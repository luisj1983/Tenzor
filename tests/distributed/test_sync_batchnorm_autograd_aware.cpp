/**
 * @file test_sync_batchnorm_autograd_aware.cpp
 * @brief Unit tests for the audit C1 SyncBatchNorm autograd-aware path.
 *
 * Verifies that the new PG-based constructor and the rebuilt
 * `SyncBatchNorm2dBackward::backward_with_variables` produce a real
 * second-order graph for `world_size > 1`, using
 * `tenzor::distributed_all_reduce` (A5) under the hood.
 *
 * The PG is a single-rank fake; multi-rank correctness is exercised by the
 * multi-process distributed test job (plan K3).
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
    nn::SyncBatchNorm sbn(/*num_features=*/4, callback, /*world_size=*/1);
    sbn.train();
    auto input = Variable(zeros({2, 4, 3, 3}, DType::Float32, Device::cpu()), true);
    auto out = sbn.forward(input);
    EXPECT_EQ(out.tensor().shape().size(), 4u);
    // world_size=1 path skips the callback (no all-reduce needed).
    EXPECT_EQ(seen, 0);
}
