/**
 * @file test_inference_mode_guard.cpp
 * @brief Unit tests for InferenceModeGuard and its interaction with NoGradGuard
 *
 * CPU-only infrastructure tests verifying RAII guard semantics,
 * nesting behaviour, and effect on Variable grad_fn attachment.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>

#include <mutex>

using namespace tenzor;

class InferenceModeGuardTest : public ::testing::Test {
protected:
    void SetUp() override {
        static std::once_flag init_flag;
        std::call_once(init_flag, []() { tenzor::initialize(); });
    }
};

// ============================================================================
// 1. DefaultOff
// ============================================================================

TEST_F(InferenceModeGuardTest, DefaultOff) {
    EXPECT_FALSE(is_inference_mode_enabled());
}

// ============================================================================
// 2. GuardEnablesInferenceMode
// ============================================================================

TEST_F(InferenceModeGuardTest, GuardEnablesInferenceMode) {
    {
        InferenceModeGuard guard;
        EXPECT_TRUE(is_inference_mode_enabled());
    }
    EXPECT_FALSE(is_inference_mode_enabled())
        << "Inference mode should be disabled after guard goes out of scope";
}

// ============================================================================
// 3. GuardDisablesGrad
// ============================================================================

TEST_F(InferenceModeGuardTest, GuardDisablesGrad) {
    // Outside the guard, grad should be enabled.
    EXPECT_TRUE(is_grad_enabled());

    {
        InferenceModeGuard guard;
        // Inference mode is stronger than NoGradGuard: grad must be off.
        EXPECT_FALSE(is_grad_enabled());
    }

    EXPECT_TRUE(is_grad_enabled());
}

// ============================================================================
// 4. NestedGuards
// ============================================================================

TEST_F(InferenceModeGuardTest, NestedGuards) {
    {
        InferenceModeGuard outer;
        EXPECT_TRUE(is_inference_mode_enabled());

        {
            InferenceModeGuard inner;
            EXPECT_TRUE(is_inference_mode_enabled());
        }

        // Inner guard destroyed, but outer still active.
        EXPECT_TRUE(is_inference_mode_enabled());
    }

    EXPECT_FALSE(is_inference_mode_enabled());
}

// ============================================================================
// 5. InteractionWithNoGrad
// ============================================================================

TEST_F(InferenceModeGuardTest, InteractionWithNoGrad) {
    // NoGradGuard alone does not enable inference mode.
    {
        NoGradGuard ng;
        EXPECT_FALSE(is_inference_mode_enabled());
        EXPECT_FALSE(is_grad_enabled());
    }

    // InferenceModeGuard inside NoGradGuard: both effects active.
    {
        NoGradGuard ng;
        {
            InferenceModeGuard ig;
            EXPECT_TRUE(is_inference_mode_enabled());
            EXPECT_FALSE(is_grad_enabled());
        }
        // Inference mode off, but NoGradGuard still active.
        EXPECT_FALSE(is_inference_mode_enabled());
        EXPECT_FALSE(is_grad_enabled());
    }

    // After both scopes: everything restored.
    EXPECT_FALSE(is_inference_mode_enabled());
    EXPECT_TRUE(is_grad_enabled());
}

// ============================================================================
// 6. OperationsDuringInferenceMode
// ============================================================================

TEST_F(InferenceModeGuardTest, OperationsDuringInferenceMode) {
    auto t = ones({4}, DType::Float32, Device::cpu());
    Variable x(t, /*requires_grad=*/true);

    Variable result;
    {
        InferenceModeGuard guard;
        // Operations performed in inference mode should not build a graph.
        result = x + x;
    }

    EXPECT_EQ(result.grad_fn(), nullptr)
        << "Operations in inference mode should not attach a grad_fn";
}
