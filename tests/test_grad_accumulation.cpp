/**
 * @file test_grad_accumulation.cpp
 * @brief Multi-backend tests for GradientAccumulator
 *
 * Tests the GradientAccumulator utility across all backends to ensure
 * correct step counting, flushing, sync signaling, and reset behavior.
 */

#include <gtest/gtest.h>
#include "tenzor/nn/utils/grad_accumulation.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/tenzor.hpp"
#include "backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradAccumulationTest : public BackendTest {
protected:
    static constexpr int64_t kAccumulationSteps = 4;

    /// Create a simple parameter on the test device
    std::shared_ptr<Variable> makeParam() {
        auto t = tenzor::randn({4, 4}, DType::Float32, device);
        return std::make_shared<Variable>(t, /*requires_grad=*/true);
    }

    /// Create an SGD optimizer with a single parameter
    std::unique_ptr<optim::SGD> makeOptimizer(std::shared_ptr<Variable> param) {
        std::vector<std::shared_ptr<Variable>> params = {param};
        return std::make_unique<optim::SGD>(std::move(params), /*lr=*/0.01);
    }
};

TEST_P(GradAccumulationTest, AccumulatesNSteps) {
    auto param = makeParam();
    auto optimizer = makeOptimizer(param);
    nn::utils::GradientAccumulator accum(*optimizer, kAccumulationSteps);

    // First N-1 steps should not trigger optimizer step
    for (int64_t i = 0; i < kAccumulationSteps - 1; ++i) {
        bool stepped = accum.step();
        EXPECT_FALSE(stepped) << "step() should return false at step " << i;
    }

    // The Nth step should trigger
    bool stepped = accum.step();
    EXPECT_TRUE(stepped) << "step() should return true on the accumulation boundary";
}

TEST_P(GradAccumulationTest, FlushForcesStep) {
    auto param = makeParam();
    auto optimizer = makeOptimizer(param);
    nn::utils::GradientAccumulator accum(*optimizer, kAccumulationSteps);

    // Accumulate 2 steps (less than N=4)
    accum.step();
    accum.step();

    // Flush should force a step even though we haven't reached the boundary
    bool flushed = accum.flush();
    EXPECT_TRUE(flushed) << "flush() should return true when there are accumulated gradients";
}

TEST_P(GradAccumulationTest, ShouldSync) {
    auto param = makeParam();
    auto optimizer = makeOptimizer(param);
    nn::utils::GradientAccumulator accum(*optimizer, kAccumulationSteps);

    // During accumulation (steps 1 .. N-2, exclusive of the boundary),
    // should_sync() should return false. The Nth step (step N-1 when
    // zero-indexed through current_step_) IS the boundary where
    // should_sync() flips to true; the loop therefore only covers the
    // earlier non-boundary steps.
    for (int64_t i = 0; i < kAccumulationSteps - 2; ++i) {
        accum.step();
        EXPECT_FALSE(accum.should_sync())
            << "should_sync() should be false during accumulation at step " << i;
    }

    // Take the step that reaches the boundary. After this, current_step_ ==
    // kAccumulationSteps - 1 and should_sync() returns true (the next
    // backward pass is the sync step).
    accum.step();
    EXPECT_TRUE(accum.should_sync())
        << "should_sync() should be true at the accumulation boundary";
}

TEST_P(GradAccumulationTest, CurrentStep) {
    auto param = makeParam();
    auto optimizer = makeOptimizer(param);
    nn::utils::GradientAccumulator accum(*optimizer, kAccumulationSteps);

    EXPECT_EQ(accum.current_step(), 0) << "Initial current_step should be 0";

    accum.step();
    EXPECT_EQ(accum.current_step(), 1) << "current_step should be 1 after one step";

    accum.step();
    EXPECT_EQ(accum.current_step(), 2) << "current_step should be 2 after two steps";

    accum.step();
    EXPECT_EQ(accum.current_step(), 3) << "current_step should be 3 after three steps";

    // Fourth step triggers optimizer update and resets counter
    accum.step();
    EXPECT_EQ(accum.current_step(), 0)
        << "current_step should reset to 0 after accumulation boundary";
}

TEST_P(GradAccumulationTest, Reset) {
    auto param = makeParam();
    auto optimizer = makeOptimizer(param);
    nn::utils::GradientAccumulator accum(*optimizer, kAccumulationSteps);

    // Advance a couple steps
    accum.step();
    accum.step();
    EXPECT_EQ(accum.current_step(), 2);

    // Reset should clear the counter
    accum.reset();
    EXPECT_EQ(accum.current_step(), 0)
        << "current_step should be 0 after reset";
}

INSTANTIATE_BACKEND_TESTS(GradAccumulationTest);
