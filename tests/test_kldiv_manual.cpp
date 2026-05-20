#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include <iostream>
#include "grad_flow_helpers.hpp"

using namespace tenzor;

class KLDivManualTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

TEST_F(KLDivManualTest, SimpleSubtract) {
    std::cout << "\n=== Testing simple subtraction ===" << std::endl;

    auto a = Variable(full({2, 3}, 2.0f, DType::Float32), true);
    auto b = Variable(full({2, 3}, 1.0f, DType::Float32), false);

    std::cout << "Computing a - b..." << std::endl;
    auto c = a - b;
    std::cout << "c is_leaf: " << c.is_leaf() << std::endl;
    std::cout << "c has grad_fn: " << (c.grad_fn() != nullptr) << std::endl;

    std::cout << "Computing mean(c)..." << std::endl;
    auto loss = mean(c);

    std::cout << "Calling backward..." << std::endl;
    loss.backward();

    std::cout << "Checking gradient..." << std::endl;
    EXPECT_GRAD_FLOWS(a);
    if (a.grad().has_value()) {
        std::cout << "SUCCESS: gradient computed!" << std::endl;
    }
}

TEST_F(KLDivManualTest, SimpleMultiply) {
    std::cout << "\n=== Testing simple multiplication ===" << std::endl;

    auto a = Variable(full({2, 3}, 2.0f, DType::Float32), true);
    auto b = Variable(full({2, 3}, 3.0f, DType::Float32), false);

    std::cout << "Computing a * b..." << std::endl;
    auto c = a * b;
    std::cout << "c is_leaf: " << c.is_leaf() << std::endl;

    std::cout << "Computing mean(c)..." << std::endl;
    auto loss = mean(c);

    std::cout << "Calling backward..." << std::endl;
    loss.backward();

    std::cout << "Checking gradient..." << std::endl;
    EXPECT_GRAD_FLOWS(a);
    if (a.grad().has_value()) {
        std::cout << "SUCCESS: gradient computed!" << std::endl;
    }
}

TEST_F(KLDivManualTest, StepByStepBackward) {
    std::cout << "\n=== Testing KLDiv loss manually ===" << std::endl;

    // Create inputs
    std::cout << "1. Creating input (requires_grad=true) and target..." << std::endl;
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

    std::cout << "   input is_leaf: " << input.is_leaf() << std::endl;
    std::cout << "   target is_leaf: " << target.is_leaf() << std::endl;

    // Step 1: clamp target
    std::cout << "2. Clamping target..." << std::endl;
    auto target_clamped = clamp(target, 1e-7f, 1.0f);
    std::cout << "   target_clamped is_leaf: " << target_clamped.is_leaf() << std::endl;

    // Step 2: log target
    std::cout << "3. Computing log(target_clamped)..." << std::endl;
    auto log_target = log(target_clamped);
    std::cout << "   log_target is_leaf: " << log_target.is_leaf() << std::endl;

    // Step 3: compute diff
    std::cout << "4. Computing diff = log_target - input..." << std::endl;
    auto diff = log_target - input;
    std::cout << "   diff is_leaf: " << diff.is_leaf() << std::endl;

    // Step 4: multiply by target
    std::cout << "5. Computing loss_unreduced = target * diff..." << std::endl;
    auto loss_unreduced = target * diff;
    std::cout << "   loss_unreduced is_leaf: " << loss_unreduced.is_leaf() << std::endl;

    // Step 5: take mean
    std::cout << "6. Computing loss = mean(loss_unreduced)..." << std::endl;
    auto loss = mean(loss_unreduced);
    std::cout << "   loss is_leaf: " << loss.is_leaf() << std::endl;
    std::cout << "   loss has grad_fn: " << (loss.grad_fn() != nullptr) << std::endl;

    // Step 6: backward
    std::cout << "7. Calling loss.backward()..." << std::endl;
    try {
        loss.backward();
        std::cout << "   SUCCESS: backward completed!" << std::endl;

        // Check gradient
        if (input.grad().has_value()) {
            std::cout << "   input.grad() exists!" << std::endl;
            EXPECT_TRUE(true);
        } else {
            std::cout << "   ERROR: input.grad() is null!" << std::endl;
            EXPECT_TRUE(false);
        }
    } catch (const std::exception& e) {
        std::cout << "   EXCEPTION: " << e.what() << std::endl;
        EXPECT_TRUE(false);
    }
}

TEST_F(KLDivManualTest, ExactReplicaOfFailingTest) {
    std::cout << "\n=== EXACT REPLICA of AdvancedLossTest.KLDivLoss_BackwardGradient ===" << std::endl;

    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);  // requires_grad=true
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

    auto criterion = nn::KLDivLoss("mean");
    auto loss = criterion(input, target);

    // Check that we can compute gradients
    loss.backward();
    EXPECT_GRAD_FLOWS(input);

    std::cout << "SUCCESS: Test passed!" << std::endl;
}
