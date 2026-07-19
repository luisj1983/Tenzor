/**
 * @file test_distributed_all_reduce_autograd.cpp
 * @brief Unit tests for tenzor::distributed_all_reduce (A5).
 *
 * The single-process public-surface tests (fake IdentityPG/DoublingPG below)
 * exercise:
 *   - null PG rejection
 *   - non-differentiable ReduceOp rejection (PRODUCT/MIN/MAX/B*)
 *   - WS=1 forward identity (output equals input value-wise)
 *   - clone semantics (mutating output does not affect input)
 *   - backward replays the same all-reduce on the incoming gradient
 *   - higher-order via backward_with_variables re-enters the autograd graph
 *
 * Multi-rank correctness (the actual collective reduction, not a hand-rolled
 * simulation of one) is exercised by MultiRankA5Test below: a real 2-process
 * GlooProcessGroup, launched via tests/distributed/run_multirank_test.sh
 * (FINDING 21 "plan K3" -- the multi-process C++ test job this file used to
 * defer to but which never existed).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function_distributed.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/ops/creation.hpp>
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

class A5Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

// Single-rank fake PG. WS=1, so collectives are identity (SUM -> x, AVG -> x).
// Records call counts so we can assert backward triggers a second all_reduce.
class IdentityPG : public ProcessGroupBase {
public:
    int all_reduce_calls = 0;
    auto rank() const -> int override { return 0; }
    auto world_size() const -> int override { return 1; }
    auto all_reduce(Tensor& /*tensor*/, ReduceOp /*op*/) -> void override {
        ++all_reduce_calls;
        // WS=1: identity, no mutation needed.
    }
    auto broadcast(Tensor&, int) -> void override {}
    auto all_gather(std::vector<Tensor>& out, const Tensor& in) -> void override {
        out.assign(1, in);
    }
    auto reduce_scatter(Tensor&, std::span<const Tensor>) -> void override {}
    auto barrier() -> void override {}
};

// Fake PG that simulates a 2-rank SUM by doubling the tensor in-place.
// Useful for confirming the gradient flow actually applies the same op.
class DoublingPG : public ProcessGroupBase {
public:
    int all_reduce_calls = 0;
    auto rank() const -> int override { return 0; }
    auto world_size() const -> int override { return 2; }
    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override {
        ++all_reduce_calls;
        if (op == ReduceOp::SUM) {
            // 2-rank SUM with identical rank-0 data == 2x of local input.
            auto* p = tensor.data<float>();
            int64_t n = tensor.numel();
            for (int64_t i = 0; i < n; ++i) p[i] *= 2.0f;
        } else if (op == ReduceOp::AVG) {
            // 2-rank AVG with identical rank-0 data == 1x of local input.
            // Already identity.
        }
    }
    auto broadcast(Tensor&, int) -> void override {}
    auto all_gather(std::vector<Tensor>& out, const Tensor& in) -> void override {
        out.assign(2, in);
    }
    auto reduce_scatter(Tensor&, std::span<const Tensor>) -> void override {}
    auto barrier() -> void override {}
};

} // namespace

TEST_F(A5Test, RejectsNullProcessGroup) {
    auto x = zeros({4}, DType::Float32, Device::cpu());
    Variable v(x, /*requires_grad=*/true);
    EXPECT_THROW(distributed_all_reduce(v, /*pg=*/nullptr, ReduceOp::SUM),
                 std::invalid_argument);
}

TEST_F(A5Test, RejectsNonDifferentiableReduceOps) {
    auto pg = std::make_shared<IdentityPG>();
    auto x = zeros({4}, DType::Float32, Device::cpu());
    Variable v(x, /*requires_grad=*/true);

    for (auto op : {ReduceOp::PRODUCT, ReduceOp::MIN, ReduceOp::MAX,
                    ReduceOp::BAND, ReduceOp::BOR, ReduceOp::BXOR}) {
        EXPECT_THROW(distributed_all_reduce(v, pg, op), std::invalid_argument)
            << "ReduceOp index " << static_cast<int>(op);
    }
}

TEST_F(A5Test, ForwardIdentityOnSingleRank) {
    auto pg = std::make_shared<IdentityPG>();
    auto x = zeros({4}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int i = 0; i < 4; ++i) xp[i] = static_cast<float>(i + 1);

    Variable v(x, /*requires_grad=*/false);
    Variable y = distributed_all_reduce(v, pg, ReduceOp::SUM);

    ASSERT_EQ(pg->all_reduce_calls, 1);
    EXPECT_EQ(y.tensor().shape().size(), 1u);
    EXPECT_EQ(y.tensor().shape()[0], 4);
    auto* yp = y.tensor().data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(yp[i], static_cast<float>(i + 1));
    }
}

TEST_F(A5Test, CloneSemantics_OutputDoesNotAliasInput) {
    auto pg = std::make_shared<IdentityPG>();
    auto x = zeros({4}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int i = 0; i < 4; ++i) xp[i] = 1.0f;

    Variable v(x, /*requires_grad=*/false);
    Variable y = distributed_all_reduce(v, pg, ReduceOp::SUM);

    // Mutate the output's storage; input must be unaffected.
    auto* yp = y.tensor().data<float>();
    for (int i = 0; i < 4; ++i) yp[i] = 99.0f;

    auto* xp_after = x.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(xp_after[i], 1.0f) << " idx=" << i;
    }
}

TEST_F(A5Test, GraphWiring_GradFnAttachedWhenRequiresGrad) {
    auto pg = std::make_shared<IdentityPG>();
    auto x = zeros({4}, DType::Float32, Device::cpu());
    Variable v(x, /*requires_grad=*/true);
    Variable y = distributed_all_reduce(v, pg, ReduceOp::SUM);
    EXPECT_TRUE(y.requires_grad());
    EXPECT_NE(y.grad_fn(), nullptr);

    // No grad_fn when requires_grad is false.
    Variable v2(x, /*requires_grad=*/false);
    Variable y2 = distributed_all_reduce(v2, pg, ReduceOp::SUM);
    EXPECT_FALSE(y2.requires_grad());
    EXPECT_EQ(y2.grad_fn(), nullptr);
}

TEST_F(A5Test, BackwardAppliesSameAllReduceOnIncomingGrad) {
    auto pg = std::make_shared<DoublingPG>();  // simulates 2-rank SUM = 2x
    auto x = zeros({4}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int i = 0; i < 4; ++i) xp[i] = 1.0f;

    Variable v(x, /*requires_grad=*/true);
    Variable y = distributed_all_reduce(v, pg, ReduceOp::SUM);
    EXPECT_EQ(pg->all_reduce_calls, 1);
    // Forward produced y = 2 * x = {2, 2, 2, 2} on rank 0.
    auto* yp = y.tensor().data<float>();
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(yp[i], 2.0f);

    // Drive the backward manually at the Function level.
    auto grad_y = zeros({4}, DType::Float32, Device::cpu());
    auto* gp = grad_y.data<float>();
    for (int i = 0; i < 4; ++i) gp[i] = 3.0f;

    auto grad_fn = y.grad_fn();
    ASSERT_NE(grad_fn, nullptr);
    auto grads = grad_fn->backward({grad_y});
    EXPECT_EQ(pg->all_reduce_calls, 2);  // forward + backward
    ASSERT_EQ(grads.size(), 1u);
    auto* g_in = grads[0].data<float>();
    // DoublingPG SUM -> 2 * 3 = 6
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(g_in[i], 6.0f);
}

TEST_F(A5Test, HigherOrderBackward_ReentersAutograd) {
    auto pg = std::make_shared<DoublingPG>();
    auto x = zeros({4}, DType::Float32, Device::cpu());
    Variable v(x, /*requires_grad=*/true);
    Variable y = distributed_all_reduce(v, pg, ReduceOp::SUM);

    auto grad_y_t = zeros({4}, DType::Float32, Device::cpu());
    auto* gp = grad_y_t.data<float>();
    for (int i = 0; i < 4; ++i) gp[i] = 1.0f;
    Variable grad_y(grad_y_t, /*requires_grad=*/true);

    auto grad_fn = y.grad_fn();
    ASSERT_NE(grad_fn, nullptr);
    auto var_grads = grad_fn->backward_with_variables({grad_y});
    ASSERT_EQ(var_grads.size(), 1u);

    // The returned Variable should itself carry a grad_fn (the second
    // DistributedAllReduceBackward in the chain), so higher-order grads
    // would continue to flow.
    EXPECT_TRUE(var_grads[0].requires_grad());
    EXPECT_NE(var_grads[0].grad_fn(), nullptr);

    // Value: DoublingPG SUM on grad=1 -> 2.
    auto* g_in = var_grads[0].tensor().data<float>();
    for (int i = 0; i < 4; ++i) EXPECT_FLOAT_EQ(g_in[i], 2.0f);
}

// ============================================================================
// Multi-rank tests (real GlooProcessGroup, real OS processes) — FINDING 21
// ============================================================================
// Run via tests/distributed/run_multirank_test.sh, which launches
// world_size copies of this binary with RANK/WORLD_SIZE/MASTER_ADDR/
// MASTER_PORT set. Skips (not fails) when those aren't set, so the binary
// still runs standalone (e.g. under plain ctest) without a hard failure.
namespace {

class MultiRankA5Test : public ::testing::Test {
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
        if (world_size_ < 2) {
            GTEST_SKIP() << "Need at least 2 processes for multi-rank A5 tests";
        }

        const char* addr_env = std::getenv("MASTER_ADDR");
        const char* port_env = std::getenv("MASTER_PORT");
        std::string addr = addr_env ? addr_env : "127.0.0.1";
        int port = port_env ? std::atoi(port_env) : 29500;
        pg_ = std::make_shared<GlooProcessGroup>(rank_, world_size_, addr, port);
    }

    void TearDown() override {
        if (pg_) {
            pg_->barrier();
            pg_.reset();
        }
    }

    int rank_{-1};
    int world_size_{-1};
    std::shared_ptr<GlooProcessGroup> pg_;
};

} // namespace

// Real collective SUM across every rank, not DoublingPG's hand-rolled "2x"
// simulation of a specific 2-rank case: each rank contributes (rank+1) in
// every element, so the correct reduced value depends on world_size and on
// every rank's real network participation, not on a single process's guess
// at what its peers would have sent.
TEST_F(MultiRankA5Test, ForwardSumAcrossRealRanks) {
    auto x = zeros({4}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int i = 0; i < 4; ++i) xp[i] = static_cast<float>(rank_ + 1);

    Variable v(x, /*requires_grad=*/false);
    Variable y = distributed_all_reduce(v, pg_, ReduceOp::SUM);

    double expected = 0.0;
    for (int r = 0; r < world_size_; ++r) expected += (r + 1);

    auto* yp = y.tensor().data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(yp[i], static_cast<float>(expected))
            << "rank " << rank_ << " element " << i;
    }
}

TEST_F(MultiRankA5Test, ForwardAvgAcrossRealRanks) {
    auto x = zeros({4}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int i = 0; i < 4; ++i) xp[i] = static_cast<float>(rank_ + 1);

    Variable v(x, /*requires_grad=*/false);
    Variable y = distributed_all_reduce(v, pg_, ReduceOp::AVG);

    double sum = 0.0;
    for (int r = 0; r < world_size_; ++r) sum += (r + 1);
    double expected = sum / world_size_;

    auto* yp = y.tensor().data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(yp[i], static_cast<float>(expected), 1e-5f)
            << "rank " << rank_ << " element " << i;
    }
}

// Every rank feeds an identical, known gradient into backward(); the result
// on every rank must be that gradient summed across every real rank's
// participation in the collective (not DoublingPG's fixed "2x").
TEST_F(MultiRankA5Test, BackwardReplaysRealAllReduce) {
    auto x = zeros({4}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int i = 0; i < 4; ++i) xp[i] = static_cast<float>(rank_ + 1);

    Variable v(x, /*requires_grad=*/true);
    Variable y = distributed_all_reduce(v, pg_, ReduceOp::SUM);

    auto grad_y = zeros({4}, DType::Float32, Device::cpu());
    auto* gp = grad_y.data<float>();
    for (int i = 0; i < 4; ++i) gp[i] = 3.0f;

    auto grad_fn = y.grad_fn();
    ASSERT_NE(grad_fn, nullptr);
    auto grads = grad_fn->backward({grad_y});
    ASSERT_EQ(grads.size(), 1u);

    double expected = 3.0 * world_size_;
    auto* g_in = grads[0].data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(g_in[i], static_cast<float>(expected))
            << "rank " << rank_ << " element " << i;
    }
}
