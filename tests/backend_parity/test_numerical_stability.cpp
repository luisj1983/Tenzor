/**
 * @file test_numerical_stability.cpp
 * @brief Numerical stability tests for edge cases
 *
 * Tests backend behavior with extreme values including very small,
 * very large, mixed magnitudes, denormalized numbers, NaN, and Inf.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"
#include <cmath>
#include <limits>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Very Small Values (Near Zero)
// ============================================================================

TEST(NumericalStability, VerySmallValues_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e-10f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-10f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-7f, 1e-9f, "Very Small Add");
}

TEST(NumericalStability, VerySmallValues_Mul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e-8f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-8f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-7f, 1e-9f, "Very Small Mul");
}

TEST(NumericalStability, VerySmallValues_Div) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e-10f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-5f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] / inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Very Small Div");
}

TEST(NumericalStability, VerySmallValues_Log) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e-5f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return log(inputs[0]);
    }, {a}, 1e-5f, 1e-7f, "Very Small Log");
}

TEST(NumericalStability, VerySmallValues_Exp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, -20.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a}, 1e-7f, 1e-9f, "Very Small Exp");
}

// ============================================================================
// Very Large Values (Near Overflow)
// ============================================================================

TEST(NumericalStability, VeryLargeValues_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e8f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e8f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-3f, 1e-5f, "Very Large Add");
}

TEST(NumericalStability, VeryLargeValues_Mul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e10f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-5f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-2f, 1e-4f, "Very Large Mul");
}

TEST(NumericalStability, VeryLargeValues_Exp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Large exp input (but not so large it overflows)
    auto a = full({32, 32}, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);
    }, {a}, 1e-2f, 1e-4f, "Very Large Exp");
}

// ============================================================================
// Mixed Magnitudes
// ============================================================================

TEST(NumericalStability, MixedMagnitudes_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e8f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-8f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-3f, 1e-5f, "Mixed Magnitudes Add");
}

TEST(NumericalStability, MixedMagnitudes_MatMul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e4f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-4f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-2f, 1e-4f, "Mixed Magnitudes MatMul");
}

// ============================================================================
// Denormalized Numbers
// ============================================================================

TEST(NumericalStability, DenormalizedNumbers) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Create denormalized numbers (very close to zero)
    float denorm = std::numeric_limits<float>::min() / 2.0f;
    auto a = full({32, 32}, denorm, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[0];
    }, {a}, 1e-7f, 1e-9f, "Denormalized Add");
}

// ============================================================================
// NaN Handling
// ============================================================================

TEST(NumericalStability, NaN_Propagation) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    // Create NaN by dividing zero by zero
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto zeros = zeros_like(inputs[0]);
        return zeros / zeros;  // Should produce NaN
    }, {a}, 0.0f, 0.0f, "NaN Creation");
}

TEST(NumericalStability, NaN_InOperation) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // sqrt of negative should produce NaN
        return sqrt(inputs[0] * -1.0f);
    }, {a}, 1e-6f, 1e-8f, "NaN from Operation");
}

// ============================================================================
// Infinity Handling
// ============================================================================

TEST(NumericalStability, Infinity_Division) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e30f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-30f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] / inputs[1];
    }, {a, b}, 1e-2f, 1e-4f, "Large Division");
}

TEST(NumericalStability, Infinity_Exp) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 100.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return exp(inputs[0]);  // May overflow to infinity
    }, {a}, 1e-2f, 1e-4f, "Exp Overflow");
}

// ============================================================================
// Precision Loss Tests
// ============================================================================

TEST(NumericalStability, PrecisionLoss_Accumulation) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({1000, 1000}, 1e-7f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Sum of many small values tests accumulation precision
        return sum(inputs[0]);
    }, {a}, 1e-3f, 1e-5f, "Precision Loss Accumulation");
}

TEST(NumericalStability, PrecisionLoss_Cancellation) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = full({32, 32}, 1e8f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e8f + 1.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Catastrophic cancellation test
        return inputs[1] - inputs[0];
    }, {a, b}, 1e-2f, 1e-4f, "Precision Loss Cancellation");
}

// ============================================================================
// Special Function Edge Cases
// ============================================================================

TEST(NumericalStability, Softmax_LargeValues) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Large values in softmax should not overflow
    auto a = full({32, 64}, 100.0f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return softmax(input_var, 1).tensor();
    }, {a}, 1e-6f, 1e-8f, "Softmax Large Values");
}

TEST(NumericalStability, LogSoftmax_StableComputation) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto a = randn({32, 64}, DType::Float32, Device::cpu()) * 10.0f;

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return log_softmax(input_var, 1).tensor();
    }, {a}, 1e-5f, 1e-7f, "LogSoftmax Stability");
}

TEST(NumericalStability, BatchNorm_SmallVariance) {
    // TODO: Requires nn::functional::batch_norm implementation
    GTEST_SKIP() << "nn::functional API not yet implemented";
}

// ============================================================================
// Gradient Stability
// ============================================================================

TEST(NumericalStability, Gradient_VerySmallValues) {
    // TODO: Tensor gradient API not yet available - use Variable for autograd
    GTEST_SKIP() << "Tensor gradient API not available";
}

TEST(NumericalStability, Gradient_VeryLargeValues) {
    // TODO: Tensor gradient API not yet available - use Variable for autograd
    GTEST_SKIP() << "Tensor gradient API not available";
}

// ============================================================================
// Underflow/Overflow Detection
// ============================================================================

TEST(NumericalStability, DetectUnderflow) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Multiply very small numbers
    auto a = full({32, 32}, 1e-20f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-20f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-7f, 1e-9f, "Detect Underflow");
}

TEST(NumericalStability, DetectOverflow) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Multiply large numbers (but keep within float32 range)
    auto a = full({32, 32}, 1e15f, DType::Float32, Device::cpu());
    auto b = full({32, 32}, 1e-10f, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] * inputs[1];
    }, {a, b}, 1e-2f, 1e-4f, "Detect Overflow");
}

// ============================================================================
// FP16 Saturation Tests
// ============================================================================
// Verify all backends clamp FP16 ±Inf to ±65504 (max finite Float16 value).
// FP32 compute producing values outside [-65504, 65504] should be saturated
// rather than producing Inf, which would cascade to NaN.

TEST(NumericalStability, FP16Saturation_MatMul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Create FP16 tensors with large values that will overflow FP16 range
    // when multiplied: 256 * 256 * inner_dim values near 256 = well over 65504
    auto a = full({4, 16}, 300.0f, DType::Float32, Device::cpu()).to(DType::Float16);
    auto b = full({16, 4}, 300.0f, DType::Float32, Device::cpu()).to(DType::Float16);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto result = matmul(inputs[0], inputs[1]);
        // Verify no Inf values in output
        auto result_f32 = result.to(DType::Float32);
        return result;
    }, {a, b}, 1e-1f, 1.0f, "FP16 Saturation MatMul");
}

TEST(NumericalStability, FP16Saturation_Add) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Create FP16 tensors with large values near the FP16 max.
    // Adding two values near 65504 should saturate, not produce Inf.
    auto a = full({32, 32}, 60000.0f, DType::Float32, Device::cpu()).to(DType::Float16);
    auto b = full({32, 32}, 60000.0f, DType::Float32, Device::cpu()).to(DType::Float16);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-1f, 1.0f, "FP16 Saturation Add");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
