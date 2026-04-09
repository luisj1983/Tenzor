/**
 * @file test_optimizers_missing_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for RAdam, LAMB, AdamAtan2,
 *        SparseAdam, and Adadelta optimizers
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/optim/radam.hpp>
#include <tenzor/nn/optim/lamb.hpp>
#include <tenzor/nn/optim/adam_atan2.hpp>
#include <tenzor/nn/optim/sparse_adam.hpp>
#include <tenzor/nn/optim/adadelta.hpp>

#include "../../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture
// ============================================================================

// Macro (not a method) so that GTEST_SKIP's internal `return`
// statement returns from the TEST_P body rather than from a helper
// method — otherwise the test continues and fails on the first op
// that doesn't support Float16.
#define skipIfHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            GTEST_SKIP() << "Optimizer convergence unreliable in half precision"; \
        } \
    } while (0)

class MissingOptimizersMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    /**
     * @brief Create a simple parameter vector for optimizer testing.
     *
     * Returns a single 4x4 parameter initialized to ones, placed on the
     * test device with the test dtype.
     */
    std::vector<std::shared_ptr<Variable>> makeParams() {
        auto t = tenzor::ones({4, 4}, DType::Float32, device());
        if (dtype() != DType::Float32) {
            t = t.to(dtype());
        }
        auto param = std::make_shared<Variable>(t, /*requires_grad=*/true);
        return {param};
    }

    /**
     * @brief Run a simple optimization step using set_grad directly.
     *
     * Sets gradient to ones, steps the optimizer, returns the sum of absolute
     * param values before the step.
     */
    float simpleStep(std::vector<std::shared_ptr<Variable>>& params,
                     optim::Optimizer& optimizer) {
        // Compute sum of abs param values before step
        float sum = 0.0f;
        for (auto& p : params) {
            auto cpu_t = p->tensor().to(Device::cpu()).to(DType::Float32);
            auto* d = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) sum += std::abs(d[i]);
        }

        // Set gradient to ones and step
        for (auto& p : params) {
            auto shape = p->tensor().shape();
            p->set_grad(tenzor::ones({shape.begin(), shape.end()}, p->tensor().dtype(), device()));
        }
        optimizer.step();

        return sum;
    }
};

// ============================================================================
// RAdam Tests
// ============================================================================

TEST_P(MissingOptimizersMultiDTypeTest, RAdam_BasicStep) {
    skipIfHalf();
    auto params = makeParams();
    optim::RAdam optimizer(params, /*lr=*/0.01);

    float loss_before = simpleStep(params, optimizer);
    float loss_after  = simpleStep(params, optimizer);

    EXPECT_LT(loss_after, loss_before)
        << "RAdam should reduce quadratic loss after two steps";
}

TEST_P(MissingOptimizersMultiDTypeTest, RAdam_LRGetSet) {
    auto params = makeParams();
    optim::RAdam optimizer(params, /*lr=*/0.01);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);

    optimizer.set_lr(0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05);
}

TEST_P(MissingOptimizersMultiDTypeTest, RAdam_StateDictRoundtrip) {
    skipIfHalf();
    auto params = makeParams();
    optim::RAdam optimizer(params, /*lr=*/0.01);

    // Take a few steps to populate state
    for (int i = 0; i < 3; ++i) {
        simpleStep(params, optimizer);
    }

    auto state = optimizer.state_dict();
    EXPECT_FALSE(state.empty()) << "state_dict should not be empty after steps";

    // Create a fresh optimizer and load state
    auto params2 = makeParams();
    optim::RAdam optimizer2(params2, /*lr=*/0.01);
    optimizer2.load_state_dict(state);

    auto state2 = optimizer2.state_dict();
    EXPECT_EQ(state.size(), state2.size())
        << "Round-tripped state_dict should have same number of entries";
}

// ============================================================================
// LAMB Tests
// ============================================================================

TEST_P(MissingOptimizersMultiDTypeTest, LAMB_BasicStep) {
    skipIfHalf();
    auto params = makeParams();
    optim::LAMB optimizer(params, /*lr=*/0.01);

    float loss_before = simpleStep(params, optimizer);
    float loss_after  = simpleStep(params, optimizer);

    EXPECT_LT(loss_after, loss_before)
        << "LAMB should reduce quadratic loss after two steps";
}

TEST_P(MissingOptimizersMultiDTypeTest, LAMB_LRGetSet) {
    auto params = makeParams();
    optim::LAMB optimizer(params, /*lr=*/0.01);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);

    optimizer.set_lr(0.1);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.1);
}

TEST_P(MissingOptimizersMultiDTypeTest, LAMB_StateDictRoundtrip) {
    skipIfHalf();
    auto params = makeParams();
    optim::LAMB optimizer(params, /*lr=*/0.01);

    for (int i = 0; i < 3; ++i) {
        simpleStep(params, optimizer);
    }

    auto state = optimizer.state_dict();
    EXPECT_FALSE(state.empty()) << "state_dict should not be empty after steps";

    auto params2 = makeParams();
    optim::LAMB optimizer2(params2, /*lr=*/0.01);
    optimizer2.load_state_dict(state);

    auto state2 = optimizer2.state_dict();
    EXPECT_EQ(state.size(), state2.size())
        << "Round-tripped state_dict should have same number of entries";
}

// ============================================================================
// AdamAtan2 Tests
// ============================================================================

TEST_P(MissingOptimizersMultiDTypeTest, AdamAtan2_BasicStep) {
    skipIfHalf();
    auto params = makeParams();
    optim::AdamAtan2 optimizer(params, /*lr=*/0.01);

    float loss_before = simpleStep(params, optimizer);
    float loss_after  = simpleStep(params, optimizer);

    EXPECT_LT(loss_after, loss_before)
        << "AdamAtan2 should reduce quadratic loss after two steps";
}

TEST_P(MissingOptimizersMultiDTypeTest, AdamAtan2_LRGetSet) {
    auto params = makeParams();
    optim::AdamAtan2 optimizer(params, /*lr=*/0.01);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);

    optimizer.set_lr(0.002);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.002);
}

TEST_P(MissingOptimizersMultiDTypeTest, AdamAtan2_StateDictRoundtrip) {
    skipIfHalf();
    auto params = makeParams();
    optim::AdamAtan2 optimizer(params, /*lr=*/0.01);

    for (int i = 0; i < 3; ++i) {
        simpleStep(params, optimizer);
    }

    auto state = optimizer.state_dict();
    EXPECT_FALSE(state.empty()) << "state_dict should not be empty after steps";

    auto params2 = makeParams();
    optim::AdamAtan2 optimizer2(params2, /*lr=*/0.01);
    optimizer2.load_state_dict(state);

    auto state2 = optimizer2.state_dict();
    EXPECT_EQ(state.size(), state2.size())
        << "Round-tripped state_dict should have same number of entries";
}

// ============================================================================
// SparseAdam Tests
// ============================================================================

TEST_P(MissingOptimizersMultiDTypeTest, SparseAdam_BasicStep) {
    skipIfHalf();
    auto params = makeParams();
    optim::SparseAdam optimizer(params, /*lr=*/0.01);

    float loss_before = simpleStep(params, optimizer);
    float loss_after  = simpleStep(params, optimizer);

    EXPECT_LT(loss_after, loss_before)
        << "SparseAdam should reduce quadratic loss after two steps";
}

TEST_P(MissingOptimizersMultiDTypeTest, SparseAdam_LRGetSet) {
    auto params = makeParams();
    optim::SparseAdam optimizer(params, /*lr=*/0.01);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);

    optimizer.set_lr(0.03);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.03);
}

TEST_P(MissingOptimizersMultiDTypeTest, SparseAdam_StateDictRoundtrip) {
    skipIfHalf();
    auto params = makeParams();
    optim::SparseAdam optimizer(params, /*lr=*/0.01);

    for (int i = 0; i < 3; ++i) {
        simpleStep(params, optimizer);
    }

    auto state = optimizer.state_dict();
    EXPECT_FALSE(state.empty()) << "state_dict should not be empty after steps";

    auto params2 = makeParams();
    optim::SparseAdam optimizer2(params2, /*lr=*/0.01);
    optimizer2.load_state_dict(state);

    auto state2 = optimizer2.state_dict();
    EXPECT_EQ(state.size(), state2.size())
        << "Round-tripped state_dict should have same number of entries";
}

// ============================================================================
// Adadelta Tests
// ============================================================================

TEST_P(MissingOptimizersMultiDTypeTest, Adadelta_BasicStep) {
    skipIfHalf();
    auto params = makeParams();
    // Adadelta defaults to lr=1.0 but use explicit value for clarity
    optim::Adadelta optimizer(params, /*lr=*/1.0, /*rho=*/0.9, /*eps=*/1e-6);

    float loss_before = simpleStep(params, optimizer);
    float loss_after  = simpleStep(params, optimizer);

    EXPECT_LT(loss_after, loss_before)
        << "Adadelta should reduce quadratic loss after two steps";
}

TEST_P(MissingOptimizersMultiDTypeTest, Adadelta_LRGetSet) {
    auto params = makeParams();
    optim::Adadelta optimizer(params, /*lr=*/1.0);

    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 1.0);

    optimizer.set_lr(0.5);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.5);
}

TEST_P(MissingOptimizersMultiDTypeTest, Adadelta_StateDictRoundtrip) {
    skipIfHalf();
    auto params = makeParams();
    optim::Adadelta optimizer(params, /*lr=*/1.0);

    for (int i = 0; i < 3; ++i) {
        simpleStep(params, optimizer);
    }

    auto state = optimizer.state_dict();
    EXPECT_FALSE(state.empty()) << "state_dict should not be empty after steps";

    auto params2 = makeParams();
    optim::Adadelta optimizer2(params2, /*lr=*/1.0);
    optimizer2.load_state_dict(state);

    auto state2 = optimizer2.state_dict();
    EXPECT_EQ(state.size(), state2.size())
        << "Round-tripped state_dict should have same number of entries";
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MissingOptimizersMultiDTypeTest);
