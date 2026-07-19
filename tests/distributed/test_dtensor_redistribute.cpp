/**
 * @file test_dtensor_redistribute.cpp
 * @brief Single-process tests for DTensor::redistribute collective wiring (B2).
 *
 * Uses a deterministic fake PG to exercise the new code paths added in audit
 * B2: Shard→Replicate (all_gather), Partial→Replicate (all_reduce),
 * Partial→Shard (reduce_scatter), and Shard(a)→Shard(b) (all_to_all_single).
 *
 * The fake PG implements collectives as if every peer held a copy of *this*
 * rank's input — so the shapes and call counts are checkable end-to-end
 * without spawning processes. Real multi-rank reduction semantics (does
 * Shard→Replicate's all_gather actually concatenate DIFFERENT peers' real
 * data in the right order, does Partial→Replicate's all_reduce actually sum
 * DIFFERENT peers' real values, etc. -- none of which "every peer holds a
 * copy of my own input" can catch) are verified by MultiRankB2Test below: a
 * real 2-process GlooProcessGroup, launched via
 * tests/distributed/run_multirank_test.sh (FINDING 21 "plan K3" -- the
 * multi-process C++ test job this file used to defer to but which never
 * existed).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/dtensor.hpp>
#include <tenzor/distributed/device_mesh.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

// Fake PG: every peer "has" the same tensor I do. Useful for shape and
// dispatch correctness checks in-process.
class SelfAsEveryonePG : public ProcessGroupBase {
public:
    int reported_world_size;
    int my_rank;
    int all_gather_calls = 0;
    int all_reduce_calls = 0;
    int reduce_scatter_calls = 0;
    int alltoall_calls = 0;

    explicit SelfAsEveryonePG(int world_size = 2, int rank = 0)
        : reported_world_size(world_size), my_rank(rank) {}

    auto rank() const -> int override { return my_rank; }
    auto world_size() const -> int override { return reported_world_size; }

    auto all_reduce(Tensor& t, ReduceOp op) -> void override {
        ++all_reduce_calls;
        // Self-as-everyone: peer count = world_size, all identical.
        if (op == ReduceOp::SUM) {
            // result = W * local
            auto* p = t.data<float>();
            int64_t n = t.numel();
            for (int64_t i = 0; i < n; ++i) p[i] *= reported_world_size;
        } else if (op == ReduceOp::AVG) {
            // result = local (W copies of same value, avg = self)
        }
    }
    auto broadcast(Tensor&, int) -> void override {}
    auto all_gather(std::vector<Tensor>& output, const Tensor& input) -> void override {
        ++all_gather_calls;
        output.assign(static_cast<size_t>(reported_world_size), input);
    }
    auto reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void override {
        ++reduce_scatter_calls;
        // SUM-reduce W identical input chunks -> per-rank slice = W * input[my_rank].
        if (input.empty()) return;
        // Initialize output with the chunk for our rank, scaled by W.
        const Tensor& my_chunk = input[static_cast<size_t>(my_rank)];
        // Copy data: use a simple scaling clone (since copy_ doesn't exist
        // on Tensor in the public surface as we saw earlier).
        // We approximate by reassigning output (the test only checks shape
        // and the call count).
        auto out_shape = std::vector<int64_t>(my_chunk.shape().begin(),
                                              my_chunk.shape().end());
        output = empty(out_shape, my_chunk.dtype(), my_chunk.device());
        // Fill with W * (chunk values).
        auto* dst = output.data<float>();
        const auto* src = my_chunk.data<float>();
        int64_t n = my_chunk.numel();
        for (int64_t i = 0; i < n; ++i) dst[i] = src[i] * reported_world_size;
    }
    auto barrier() -> void override {}
    auto all_to_all_single(Tensor& output, const Tensor& input) -> void override {
        ++alltoall_calls;
        // Self-as-everyone: I send input[k] to peer k and receive peer k's
        // input[my_rank]. Since every peer sent the same data, the output
        // equals the input (up to ordering). Pass through.
        output = input;
    }
};

class B2Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    // Build a 1D mesh of size W with my_rank=0 and the SelfAsEveryonePG attached.
    static auto make_mesh(int world_size)
        -> std::pair<std::shared_ptr<DeviceMesh>, std::shared_ptr<SelfAsEveryonePG>> {
        auto mesh = std::make_shared<DeviceMesh>(
            Device::Type::CPU, std::vector<int64_t>{world_size},
            std::vector<std::string>{"x"}, /*mesh_rank=*/0);
        auto pg = std::make_shared<SelfAsEveryonePG>(world_size, /*rank=*/0);
        mesh->set_process_group(pg);
        return {mesh, pg};
    }

    static auto seq_tensor(std::vector<int64_t> shape) -> Tensor {
        Tensor t = zeros(shape, DType::Float32, Device::cpu());
        auto* p = t.data<float>();
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
        return t;
    }
};

} // namespace

TEST_F(B2Test, ShardToReplicate_CallsAllGatherAndCats) {
    auto [mesh, pg] = make_mesh(2);
    // Local shard: [3, 4]. After Shard(dim=0)→Replicate on a W=2 mesh, the
    // gathered tensor is shape [6, 4] (W copies of the local concatenated
    // along dim 0).
    Tensor local = seq_tensor({3, 4});
    DTensor d(local, mesh, {Shard{0}});
    auto result = d.redistribute({Replicate{}});
    EXPECT_EQ(pg->all_gather_calls, 1);
    auto shape = result.local_tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 6);
    EXPECT_EQ(shape[1], 4);
}

TEST_F(B2Test, ReplicateToShard_NoCommunication_LocalNarrow) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({4, 4});
    DTensor d(local, mesh, {Replicate{}});
    auto result = d.redistribute({Shard{0}});
    // No collective: narrow only.
    EXPECT_EQ(pg->all_gather_calls, 0);
    EXPECT_EQ(pg->all_reduce_calls, 0);
    auto shape = result.local_tensor().shape();
    EXPECT_EQ(shape[0], 2);  // 4 / 2
    EXPECT_EQ(shape[1], 4);
}

TEST_F(B2Test, PartialToReplicate_CallsAllReduce) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({3, 4});
    DTensor d(local, mesh, {Partial{DTensorReduceOp::Sum}});
    auto result = d.redistribute({Replicate{}});
    EXPECT_EQ(pg->all_reduce_calls, 1);
    auto shape = result.local_tensor().shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
    // SelfAsEveryone SUM with W=2 -> values doubled.
    auto* p = result.local_tensor().data<float>();
    EXPECT_FLOAT_EQ(p[0], 0.0f);    // 0 * 2
    EXPECT_FLOAT_EQ(p[1], 2.0f);    // 1 * 2
    EXPECT_FLOAT_EQ(p[11], 22.0f);  // 11 * 2
}

TEST_F(B2Test, PartialToShard_Sum_CallsReduceScatter) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({4, 4});  // shard_dim=0 divisible by W=2
    DTensor d(local, mesh, {Partial{DTensorReduceOp::Sum}});
    auto result = d.redistribute({Shard{0}});
    EXPECT_EQ(pg->reduce_scatter_calls, 1);
    auto shape = result.local_tensor().shape();
    EXPECT_EQ(shape[0], 2);  // local chunk: 4 / 2
    EXPECT_EQ(shape[1], 4);
}

TEST_F(B2Test, PartialToShard_Mean_FallsBackToAllReducePlusNarrow) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({4, 4});
    DTensor d(local, mesh, {Partial{DTensorReduceOp::Mean}});
    auto result = d.redistribute({Shard{0}});
    // Mean op routes through all_reduce(AVG) + narrow, not reduce_scatter.
    EXPECT_EQ(pg->reduce_scatter_calls, 0);
    EXPECT_EQ(pg->all_reduce_calls, 1);
    auto shape = result.local_tensor().shape();
    EXPECT_EQ(shape[0], 2);
    EXPECT_EQ(shape[1], 4);
}

TEST_F(B2Test, ShardToShard_CallsAllToAllSingle) {
    auto [mesh, pg] = make_mesh(2);
    // Local: shape [2, 4] (sharded on dim 0). After Shard(0)→Shard(1) on W=2
    // we move from "row 0..1 of a [4,4] global" to "col 0..1 of [4,4] global".
    // Local result shape: [4, 2].
    Tensor local = seq_tensor({2, 4});
    DTensor d(local, mesh, {Shard{0}});
    auto result = d.redistribute({Shard{1}});
    EXPECT_EQ(pg->alltoall_calls, 1);
    auto shape = result.local_tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);  // src_dim expanded by W
    EXPECT_EQ(shape[1], 2);  // dst_dim shrunk by W
}

TEST_F(B2Test, MissingProcessGroupThrowsClearError) {
    auto mesh = std::make_shared<DeviceMesh>(
        Device::Type::CPU, std::vector<int64_t>{2},
        std::vector<std::string>{"x"}, /*mesh_rank=*/0);
    // No PG attached.
    Tensor local = seq_tensor({3, 4});
    DTensor d(local, mesh, {Shard{0}});
    EXPECT_THROW(d.redistribute({Replicate{}}), std::runtime_error);
}

TEST_F(B2Test, SingleAxisMeshSizeOne_IsNoOp) {
    auto [mesh, pg] = make_mesh(1);
    Tensor local = seq_tensor({3, 4});
    DTensor d(local, mesh, {Shard{0}});
    auto result = d.redistribute({Replicate{}});
    EXPECT_EQ(pg->all_gather_calls, 0);
    auto shape = result.local_tensor().shape();
    EXPECT_EQ(shape[0], 3);
    EXPECT_EQ(shape[1], 4);
}

TEST_F(B2Test, FullTensor_RoundTripsThroughAllGather) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({3, 4});
    DTensor d(local, mesh, {Shard{0}});
    Tensor full = d.full_tensor();
    EXPECT_EQ(pg->all_gather_calls, 1);
    auto shape = full.shape();
    EXPECT_EQ(shape[0], 6);
    EXPECT_EQ(shape[1], 4);
}

TEST_F(B2Test, InvalidTransition_ShardToPartial_Throws) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({2, 4});
    DTensor d(local, mesh, {Shard{0}});
    EXPECT_THROW(d.redistribute({Partial{DTensorReduceOp::Sum}}),
                 std::runtime_error);
}

TEST_F(B2Test, InvalidTransition_ReplicateToPartial_Throws) {
    auto [mesh, pg] = make_mesh(2);
    Tensor local = seq_tensor({2, 4});
    DTensor d(local, mesh, {Replicate{}});
    EXPECT_THROW(d.redistribute({Partial{DTensorReduceOp::Sum}}),
                 std::runtime_error);
}

// ============================================================================
// Multi-rank tests (real GlooProcessGroup, real OS processes) — FINDING 21
// ============================================================================
namespace {

class MultiRankB2Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    void SetUp() override {
        const char* rank_env = std::getenv("RANK");
        const char* world_size_env = std::getenv("WORLD_SIZE");
        if (!rank_env || !world_size_env) {
            GTEST_SKIP() << "Distributed environment not available (RANK, WORLD_SIZE not set)";
        }
        rank_ = std::atoi(rank_env);
        world_size_ = std::atoi(world_size_env);
        if (world_size_ != 2) {
            GTEST_SKIP() << "MultiRankB2Test is written for exactly 2 ranks";
        }

        const char* addr_env = std::getenv("MASTER_ADDR");
        const char* port_env = std::getenv("MASTER_PORT");
        std::string addr = addr_env ? addr_env : "127.0.0.1";
        int port = port_env ? std::atoi(port_env) : 29500;
        pg_ = std::make_shared<GlooProcessGroup>(rank_, world_size_, addr, port);

        mesh_ = std::make_shared<DeviceMesh>(
            Device::Type::CPU, std::vector<int64_t>{world_size_},
            std::vector<std::string>{"x"}, /*mesh_rank=*/rank_);
        mesh_->set_process_group(pg_);
    }

    void TearDown() override {
        if (pg_) {
            pg_->barrier();
            pg_.reset();
        }
    }

    static auto seq_tensor(std::vector<int64_t> shape) -> Tensor {
        Tensor t = zeros(shape, DType::Float32, Device::cpu());
        auto* p = t.data<float>();
        int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) p[i] = static_cast<float>(i);
        return t;
    }

    int rank_{-1};
    int world_size_{-1};
    std::shared_ptr<GlooProcessGroup> pg_;
    std::shared_ptr<DeviceMesh> mesh_;
};

} // namespace

// Each rank's local shard holds DIFFERENT, rank-identifiable data (unlike
// SelfAsEveryonePG, where every "peer" is a copy of the caller's own input).
// After Shard(0)->Replicate, every rank must see the TRUE global tensor --
// rank 0's rows followed by rank 1's rows, in the correct order -- proving
// the real all_gather actually moved each peer's own data, not just that a
// collective of the right shape was called.
TEST_F(MultiRankB2Test, ShardToReplicate_RealAllGatherAcrossRanks) {
    Tensor local = zeros({3, 4}, DType::Float32, Device::cpu());
    {
        auto* p = local.data<float>();
        for (int64_t i = 0; i < 12; ++i) p[i] = static_cast<float>(rank_ * 100 + i);
    }

    DTensor d(local, mesh_, {Shard{0}});
    auto result = d.redistribute({Replicate{}});

    auto shape = result.local_tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 6);
    EXPECT_EQ(shape[1], 4);

    auto cpu_full = result.local_tensor().to(Device::cpu()).contiguous();
    const float* fp = cpu_full.data<float>();
    for (int r = 0; r < world_size_; ++r) {
        for (int64_t i = 0; i < 12; ++i) {
            float expected = static_cast<float>(r * 100 + i);
            EXPECT_FLOAT_EQ(fp[r * 12 + i], expected)
                << "rank " << rank_ << " reading peer " << r << "'s row-block, element " << i;
        }
    }
}

// Every rank contributes a DIFFERENT Partial value (ones() * (rank+1)); the
// real all_reduce SUM must combine all of them, not just replay one rank's
// own contribution scaled by world_size (which is all SelfAsEveryonePG can
// simulate).
TEST_F(MultiRankB2Test, PartialToReplicate_RealAllReduceAcrossRanks) {
    Tensor local = full({3, 4}, static_cast<double>(rank_ + 1), DType::Float32, Device::cpu());
    DTensor d(local, mesh_, {Partial{DTensorReduceOp::Sum}});
    auto result = d.redistribute({Replicate{}});

    double expected_sum = 0.0;
    for (int r = 0; r < world_size_; ++r) expected_sum += (r + 1);

    auto cpu_result = result.local_tensor().to(Device::cpu()).contiguous();
    const float* p = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(p[i], static_cast<float>(expected_sum), 1e-4f)
            << "rank " << rank_ << " element " << i;
    }
}

// Real reduce_scatter: every rank contributes a DIFFERENT Partial value: the
// resulting shard on rank r must be the SUM of every peer's contribution,
// restricted to rank r's own slice -- exercised with a genuinely different
// value per rank so a broken reduce_scatter (e.g. one that just narrows
// without reducing, or reduces without narrowing) would be caught.
TEST_F(MultiRankB2Test, PartialToShard_RealReduceScatterAcrossRanks) {
    Tensor local = full({4, 4}, static_cast<double>(rank_ + 1), DType::Float32, Device::cpu());
    DTensor d(local, mesh_, {Partial{DTensorReduceOp::Sum}});
    auto result = d.redistribute({Shard{0}});

    auto shape = result.local_tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4 / world_size_);
    EXPECT_EQ(shape[1], 4);

    double expected_sum = 0.0;
    for (int r = 0; r < world_size_; ++r) expected_sum += (r + 1);

    auto cpu_result = result.local_tensor().to(Device::cpu()).contiguous();
    const float* p = cpu_result.data<float>();
    for (int64_t i = 0; i < cpu_result.numel(); ++i) {
        EXPECT_NEAR(p[i], static_cast<float>(expected_sum), 1e-4f)
            << "rank " << rank_ << " element " << i;
    }
}

// Shard(0)->Shard(1) on a real 2-rank mesh: rank 0 holds rows [0,1] and rank
// 1 holds rows [2,3] of one shared logical [4,4] global tensor (a genuine
// distributed tensor, not a self-as-everyone fake). After the transpose-like
// redistribution, rank r must hold ALL 4 rows but only columns [2r, 2r+1] --
// verified against the known closed-form global value at every position,
// which only holds if all_to_all_single actually exchanged the two ranks'
// real, different column data.
TEST_F(MultiRankB2Test, ShardToShard_RealAllToAllAcrossRanks) {
    // This rank's local tile is rows [rank_*2, rank_*2+1] of the global
    // seq_tensor({4,4}) (value at [row, col] = row*4 + col).
    Tensor local = zeros({2, 4}, DType::Float32, Device::cpu());
    {
        auto* p = local.data<float>();
        for (int64_t local_row = 0; local_row < 2; ++local_row) {
            int64_t global_row = rank_ * 2 + local_row;
            for (int64_t col = 0; col < 4; ++col) {
                p[local_row * 4 + col] = static_cast<float>(global_row * 4 + col);
            }
        }
    }

    DTensor d(local, mesh_, {Shard{0}});
    auto result = d.redistribute({Shard{1}});

    auto shape = result.local_tensor().shape();
    ASSERT_EQ(shape.size(), 2u);
    EXPECT_EQ(shape[0], 4);
    EXPECT_EQ(shape[1], 2);

    // Result[row][local_col] must equal the global value at
    // [row, rank_*2 + local_col].
    auto cpu_result = result.local_tensor().to(Device::cpu()).contiguous();
    const float* p = cpu_result.data<float>();
    for (int64_t row = 0; row < 4; ++row) {
        for (int64_t local_col = 0; local_col < 2; ++local_col) {
            int64_t global_col = rank_ * 2 + local_col;
            float expected = static_cast<float>(row * 4 + global_col);
            EXPECT_FLOAT_EQ(p[row * 2 + local_col], expected)
                << "rank " << rank_ << " row " << row << " local_col " << local_col;
        }
    }
}
