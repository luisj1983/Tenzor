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
 * without spawning processes. Real multi-rank reduction semantics are
 * verified by plan K3's multi-process distributed test job.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/dtensor.hpp>
#include <tenzor/distributed/device_mesh.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/transform.hpp>

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
