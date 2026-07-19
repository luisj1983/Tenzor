/**
 * @file test_sync_batchnorm_autograd_aware.cpp
 * @brief Unit tests for the audit C1 SyncBatchNorm autograd-aware path.
 *
 * Verifies that the new PG-based constructor and the rebuilt
 * `SyncBatchNorm2dBackward::backward_with_variables` produce a real
 * second-order graph for `world_size > 1`, using
 * `tenzor::distributed_all_reduce` (A5) under the hood.
 *
 * The PG here is a single-rank fake (its all_reduce is an identity, since
 * there is only one real process to reduce); multi-rank correctness -- does
 * SyncBatchNorm's output actually reflect batch statistics combined across
 * DIFFERENT ranks' real local batches, not just one rank's own data replayed
 * -- is verified by MultiRankC1Test below: a real 2-process GlooProcessGroup,
 * launched via tests/distributed/run_multirank_test.sh (FINDING 21
 * "plan K3" -- the multi-process C++ test job this file used to defer to but
 * which never existed).
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/distributed/process_group.hpp>
#include <tenzor/distributed/distributed.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../grad_flow_helpers.hpp"
#include <cstdlib>

using namespace tenzor;
using namespace tenzor::distributed;

namespace {

// Fake PG that pretends to be 2-rank but treats every collective as identity
// (so the math matches the single-process case but the *code path* under
// test is the distributed one).
class FakeWS2PG : public ProcessGroupBase {
public:
    int all_reduce_calls = 0;
    int reported_world_size = 2;
    auto rank() const -> int override { return 0; }
    auto world_size() const -> int override { return reported_world_size; }
    auto all_reduce(Tensor& /*t*/, ReduceOp /*op*/) -> void override {
        ++all_reduce_calls;
    }
    auto broadcast(Tensor&, int) -> void override {}
    auto all_gather(std::vector<Tensor>& out, const Tensor& in) -> void override {
        out.assign(static_cast<size_t>(reported_world_size), in);
    }
    auto reduce_scatter(Tensor&, std::span<const Tensor>) -> void override {}
    auto barrier() -> void override {}
};

class C1Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

} // namespace

TEST_F(C1Test, PgBasedConstructorAcceptsBareProcessGroup) {
    auto pg = std::make_shared<FakeWS2PG>();
    EXPECT_NO_THROW({
        nn::SyncBatchNorm sbn(/*num_features=*/4, pg);
    });
}

TEST_F(C1Test, PgBasedConstructorInfersWorldSizeFromPG) {
    auto pg = std::make_shared<FakeWS2PG>();
    nn::SyncBatchNorm sbn(/*num_features=*/4, pg);
    // The new constructor's `world_size=0` default forwards
    // `pg->world_size()` to the impl. We can't directly probe `world_size_`
    // (it's private) but the higher-order behavior verifies it indirectly
    // (see HigherOrderGraphBuiltWhenPGProvided below).
    sbn.train();
}

TEST_F(C1Test, HigherOrderGraphBuiltWhenPGProvided) {
    auto pg = std::make_shared<FakeWS2PG>();
    nn::SyncBatchNorm sbn(/*num_features=*/4, pg);
    sbn.train();

    auto input = Variable(zeros({2, 4, 3, 3}, DType::Float32, Device::cpu()),
                          /*requires_grad=*/true);
    // Fill with non-zero so the math has variance.
    auto* ip = input.tensor().data<float>();
    for (int64_t i = 0; i < input.tensor().numel(); ++i) {
        ip[i] = static_cast<float>((i * 7) % 11) - 5.0f;
    }

    auto output = sbn.forward(input);
    auto loss = tenzor::sum(output);

    // Critical test: with C1 + A5, create_graph=true must populate
    // grad_variable() (not just grad()) — proving the all-reduce flows
    // through autograd rather than disconnecting.
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "SyncBatchNorm (pg, world_size=2) with create_graph=true must "
           "populate grad_variable() now that the all-reduce is a Variable-level op";

    // Second-order: compute grad_norm = sum(grad^2), call .backward(), and
    // confirm the engine can walk the higher-order chain.
    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);

    // Sanity: the all-reduce was actually called (forward + backward).
    EXPECT_GE(pg->all_reduce_calls, 1);
}

TEST_F(C1Test, LegacyConstructor_NoPG_KeepsBackwardCompat) {
    // No PG -> legacy injected callback path. Forward should still work,
    // and (per the existing test_higher_order_nn coverage) world_size=1
    // higher-order remains correct.
    int seen = 0;
    nn::AllReduceFn callback = [&seen](Tensor&) { ++seen; };
    // intentionally exercising deprecated legacy SyncBatchNorm ctor
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    nn::SyncBatchNorm sbn(/*num_features=*/4, callback, /*world_size=*/1);
#pragma GCC diagnostic pop
    sbn.train();
    auto input = Variable(zeros({2, 4, 3, 3}, DType::Float32, Device::cpu()), true);
    auto out = sbn.forward(input);
    EXPECT_EQ(out.tensor().shape().size(), 4u);
    // world_size=1 path skips the callback (no all-reduce needed).
    EXPECT_EQ(seen, 0);
}

// ============================================================================
// Multi-rank tests (real GlooProcessGroup, real OS processes) — FINDING 21
// ============================================================================
namespace {

class MultiRankC1Test : public ::testing::Test {
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
            GTEST_SKIP() << "MultiRankC1Test is written for exactly 2 ranks";
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

// Every rank feeds a DIFFERENT constant local batch (rank r -> value r+1
// everywhere). Real SyncBatchNorm output must reflect statistics combined
// across BOTH ranks' real data, not just one rank's own (constant-input)
// local batch, which would normalize to all-zeros -- the single-process
// FakeWS2PG test above cannot distinguish these cases since its all_reduce
// is a no-op identity.
//
// Closed form for 2 equal-size ranks with constant local values v0=1, v1=2:
// combined mean = (v0+v1)/2 = 1.5; combined (biased/population) variance =
// ((v0-mean)^2 + (v1-mean)^2) / 2 = 0.25, so combined std = 0.5. Normalized
// (affine defaults to gamma=1, beta=0 at init):
//   rank 0: (1 - 1.5) / 0.5 = -1.0
//   rank 1: (2 - 1.5) / 0.5 = +1.0
TEST_F(MultiRankC1Test, OutputReflectsBatchStatsCombinedAcrossRealRanks) {
    nn::SyncBatchNorm sbn(/*num_features=*/4, pg_);
    sbn.train();

    auto input = Variable(
        full({2, 4, 3, 3}, static_cast<double>(rank_ + 1), DType::Float32, Device::cpu()),
        /*requires_grad=*/false);

    auto output = sbn.forward(input);
    auto cpu_out = output.tensor().to(Device::cpu()).contiguous();
    const float* p = cpu_out.data<float>();

    float expected = (rank_ == 0) ? -1.0f : 1.0f;
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(p[i], expected, 5e-2f)
            << "rank " << rank_ << " element " << i;
    }
}

// Second-order gradient flow with a REAL 2-rank process group (not
// FakeWS2PG's identity all_reduce): create_graph=true must still populate
// grad_variable() and support a second .backward() call through the real
// distributed_all_reduce path.
TEST_F(MultiRankC1Test, HigherOrderGraphBuiltWithRealProcessGroup) {
    nn::SyncBatchNorm sbn(/*num_features=*/4, pg_);
    sbn.train();

    auto input = Variable(zeros({2, 4, 3, 3}, DType::Float32, Device::cpu()),
                          /*requires_grad=*/true);
    auto* ip = input.tensor().data<float>();
    for (int64_t i = 0; i < input.tensor().numel(); ++i) {
        // Rank-distinguishable, non-constant fill so real cross-rank
        // statistics actually differ from a single rank's own.
        ip[i] = static_cast<float>((i * 7) % 11) - 5.0f + rank_ * 10.0f;
    }

    auto output = sbn.forward(input);
    auto loss = tenzor::sum(output);

    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
    ASSERT_TRUE(input.grad_variable().has_value())
        << "rank " << rank_ << ": create_graph=true must populate grad_variable() "
           "with a real multi-rank process group too";

    Variable grad_var = input.grad_variable().value();
    auto grad_norm = tenzor::sum(grad_var * grad_var);
    grad_norm.backward();
    EXPECT_GRAD_FLOWS(input);
}
