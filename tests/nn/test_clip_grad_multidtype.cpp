/**
 * @file test_clip_grad_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for gradient clipping utilities
 *
 * Covers: clip_grad_norm_, clip_grad_value_
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/clip_grad.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class ClipGradMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Create a parameter with a known gradient
    std::shared_ptr<Variable> makeParamWithGrad(
        const std::vector<int64_t>& shape, float grad_val) {
        auto param = std::make_shared<Variable>(createRandn(shape), true);
        // Simulate a gradient by setting it
        auto grad = tenzor::full(shape, grad_val, dtype(), device());
        param->set_grad(grad);
        return param;
    }
};

// ============================================================================
// clip_grad_norm_ Tests
// ============================================================================

TEST_P(ClipGradMultiDTypeTest, ClipGradNormBasic) {
    auto p1 = makeParamWithGrad({10}, 3.0f);

    std::vector<std::shared_ptr<Variable>> params = {p1};
    double total_norm = nn::utils::clip_grad_norm_(params, 1.0, 2.0);

    // Total L2 norm of grad=[3,3,...,3] (10 elements) = 3*sqrt(10) ≈ 9.487
    EXPECT_GT(total_norm, 1.0);

    // After clipping, grad norm should be ~1.0
    auto grad = p1->grad();
    ASSERT_TRUE(grad.has_value());
    auto g_f32 = grad->to(Device::cpu()).to(DType::Float32);
    float clipped_norm = 0.0f;
    auto* d = g_f32.data<float>();
    for (int64_t i = 0; i < g_f32.numel(); ++i) {
        clipped_norm += d[i] * d[i];
    }
    clipped_norm = std::sqrt(clipped_norm);
    EXPECT_NEAR(clipped_norm, 1.0f, std::max(atol(), 0.1f));
}

TEST_P(ClipGradMultiDTypeTest, ClipGradNormReturnValue) {
    auto p1 = makeParamWithGrad({4}, 1.0f);
    std::vector<std::shared_ptr<Variable>> params = {p1};

    // L2 norm of [1,1,1,1] = 2.0
    double total_norm = nn::utils::clip_grad_norm_(params, 10.0, 2.0);
    EXPECT_NEAR(total_norm, 2.0, std::max(static_cast<double>(atol()), 0.1));
}

TEST_P(ClipGradMultiDTypeTest, ClipGradNormNoClipping) {
    auto p1 = makeParamWithGrad({4}, 0.1f);
    std::vector<std::shared_ptr<Variable>> params = {p1};

    // L2 norm of [0.1]*4 = 0.2, max_norm=10 => no clipping
    double total_norm = nn::utils::clip_grad_norm_(params, 10.0, 2.0);
    EXPECT_LT(total_norm, 10.0);

    // Gradient should be unchanged
    auto grad = p1->grad();
    ASSERT_TRUE(grad.has_value());
    auto g_f32 = grad->to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(g_f32.data<float>()[0], 0.1f, std::max(atol(), 1e-3f));
}

// ============================================================================
// clip_grad_value_ Tests
// ============================================================================

TEST_P(ClipGradMultiDTypeTest, ClipGradValueBasic) {
    auto p1 = makeParamWithGrad({4}, 5.0f);
    std::vector<std::shared_ptr<Variable>> params = {p1};

    nn::utils::clip_grad_value_(params, 1.0);

    auto grad = p1->grad();
    ASSERT_TRUE(grad.has_value());
    auto g_f32 = grad->to(Device::cpu()).to(DType::Float32);
    auto* d = g_f32.data<float>();
    for (int64_t i = 0; i < g_f32.numel(); ++i) {
        EXPECT_LE(std::abs(d[i]), 1.0f + atol()) << "Index " << i;
    }
}

TEST_P(ClipGradMultiDTypeTest, ClipGradValueNoClipping) {
    auto p1 = makeParamWithGrad({4}, 0.5f);
    std::vector<std::shared_ptr<Variable>> params = {p1};

    nn::utils::clip_grad_value_(params, 10.0);

    auto grad = p1->grad();
    ASSERT_TRUE(grad.has_value());
    auto g_f32 = grad->to(Device::cpu()).to(DType::Float32);
    // Should be unchanged
    EXPECT_NEAR(g_f32.data<float>()[0], 0.5f, std::max(atol(), 1e-3f));
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ClipGradMultiDTypeTest);
