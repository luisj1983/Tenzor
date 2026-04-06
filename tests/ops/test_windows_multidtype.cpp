/**
 * @file test_windows_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for window functions
 *
 * Covers: hann_window, hamming_window, blackman_window, bartlett_window, kaiser_window
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/windows.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Fixture
// ============================================================================

class WindowFunctionsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::vector<float> toFloats(const Tensor& t) {
        auto f32 = t.to(Device::cpu()).to(DType::Float32);
        auto* d = f32.data<float>();
        return std::vector<float>(d, d + f32.numel());
    }
};

// ============================================================================
// Hann Window Tests
// ============================================================================

TEST_P(WindowFunctionsMultiDTypeTest, HannWindowShape) {
    auto w = tenzor::hann_window(64, true, dtype(), device());
    expectShape(w, {64});
    expectDType(w);
    expectDevice(w);
}

TEST_P(WindowFunctionsMultiDTypeTest, HannWindowSymmetric) {
    // Symmetric mode: w[i] == w[N-1-i]
    auto w = tenzor::hann_window(32, false, dtype(), device());
    auto vals = toFloats(w);
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(vals[i], vals[31 - i], atol()) << "Index " << i;
    }
}

TEST_P(WindowFunctionsMultiDTypeTest, HannWindowEndpoints) {
    // Symmetric hann: endpoints are 0
    auto w = tenzor::hann_window(16, false, dtype(), device());
    auto vals = toFloats(w);
    EXPECT_NEAR(vals[0], 0.0f, atol());
    EXPECT_NEAR(vals[15], 0.0f, atol());
}

TEST_P(WindowFunctionsMultiDTypeTest, HannWindowValueRange) {
    auto w = tenzor::hann_window(64, true, dtype(), device());
    float max_val = compute_max(w);
    float min_val = compute_min(w);
    EXPECT_GE(min_val, -atol());
    EXPECT_LE(max_val, 1.0f + atol());
}

// ============================================================================
// Hamming Window Tests
// ============================================================================

TEST_P(WindowFunctionsMultiDTypeTest, HammingWindowShape) {
    auto w = tenzor::hamming_window(64, true, 0.54, 0.46, dtype(), device());
    expectShape(w, {64});
    expectDType(w);
    expectDevice(w);
}

TEST_P(WindowFunctionsMultiDTypeTest, HammingWindowSymmetric) {
    auto w = tenzor::hamming_window(32, false, 0.54, 0.46, dtype(), device());
    auto vals = toFloats(w);
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(vals[i], vals[31 - i], atol()) << "Index " << i;
    }
}

TEST_P(WindowFunctionsMultiDTypeTest, HammingWindowMinValue) {
    // Hamming window has a non-zero minimum (alpha - beta)
    auto w = tenzor::hamming_window(64, false, 0.54, 0.46, dtype(), device());
    float min_val = compute_min(w);
    // alpha - beta = 0.54 - 0.46 = 0.08
    EXPECT_GE(min_val, 0.08f - atol() - 0.01f);
}

// ============================================================================
// Blackman Window Tests
// ============================================================================

TEST_P(WindowFunctionsMultiDTypeTest, BlackmanWindowShape) {
    auto w = tenzor::blackman_window(64, true, dtype(), device());
    expectShape(w, {64});
    expectDType(w);
    expectDevice(w);
}

TEST_P(WindowFunctionsMultiDTypeTest, BlackmanWindowNearZeroEndpoints) {
    // Blackman window has near-zero endpoints
    auto w = tenzor::blackman_window(32, false, dtype(), device());
    auto vals = toFloats(w);
    EXPECT_NEAR(vals[0], 0.0f, 0.01f + atol());
    EXPECT_NEAR(vals[31], 0.0f, 0.01f + atol());
}

// ============================================================================
// Bartlett Window Tests
// ============================================================================

TEST_P(WindowFunctionsMultiDTypeTest, BartlettWindowShape) {
    auto w = tenzor::bartlett_window(64, true, dtype(), device());
    expectShape(w, {64});
    expectDType(w);
    expectDevice(w);
}

TEST_P(WindowFunctionsMultiDTypeTest, BartlettWindowPeakAtCenter) {
    // Triangular window peaks at center
    auto w = tenzor::bartlett_window(32, false, dtype(), device());
    auto vals = toFloats(w);
    float peak = *std::max_element(vals.begin(), vals.end());
    // Peak should be near 1.0 at the center
    EXPECT_NEAR(peak, 1.0f, atol() + 0.05f);
}

TEST_P(WindowFunctionsMultiDTypeTest, BartlettWindowZeroEndpoints) {
    auto w = tenzor::bartlett_window(32, false, dtype(), device());
    auto vals = toFloats(w);
    EXPECT_NEAR(vals[0], 0.0f, atol());
    EXPECT_NEAR(vals[31], 0.0f, atol());
}

// ============================================================================
// Kaiser Window Tests
// ============================================================================

TEST_P(WindowFunctionsMultiDTypeTest, KaiserWindowShape) {
    auto w = tenzor::kaiser_window(64, true, 12.0, dtype(), device());
    expectShape(w, {64});
    expectDType(w);
    expectDevice(w);
}

TEST_P(WindowFunctionsMultiDTypeTest, KaiserWindowValueRange) {
    auto w = tenzor::kaiser_window(64, true, 12.0, dtype(), device());
    float max_val = compute_max(w);
    float min_val = compute_min(w);
    EXPECT_GE(min_val, -atol());
    EXPECT_LE(max_val, 1.0f + atol());
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(WindowFunctionsMultiDTypeTest, WindowSize1) {
    auto w = tenzor::hann_window(1, true, dtype(), device());
    expectShape(w, {1});
    // Size-1 window: value depends on implementation (0 or 1 are both valid)
    auto vals = toFloats(w);
    EXPECT_TRUE(vals[0] >= -atol() && vals[0] <= 1.0f + atol());
}

TEST_P(WindowFunctionsMultiDTypeTest, PeriodicVsSymmetric) {
    auto periodic = tenzor::hann_window(8, true, dtype(), device());
    auto symmetric = tenzor::hann_window(8, false, dtype(), device());

    // They should be different (periodic shifts the window slightly)
    auto p_vals = toFloats(periodic);
    auto s_vals = toFloats(symmetric);

    bool any_different = false;
    for (size_t i = 0; i < p_vals.size(); ++i) {
        if (std::abs(p_vals[i] - s_vals[i]) > 1e-6f) {
            any_different = true;
            break;
        }
    }
    EXPECT_TRUE(any_different) << "Periodic and symmetric should differ";
}

// ============================================================================
// Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(WindowFunctionsMultiDTypeTest);
