/**
 * @file test_numerical_stability.cpp
 * @brief Numerical stability tests for edge cases
 *
 * Tests backend behavior with extreme values including very small,
 * very large, mixed magnitudes, denormalized numbers, NaN, and Inf.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/loss/losses.hpp>
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
    // BatchNorm with very small variance — epsilon must prevent division by ~zero
    auto input = full({4, 8, 2, 2}, 42.0f, DType::Float32, Device::cpu());
    auto noise = randn({4, 8, 2, 2}, DType::Float32, Device::cpu()) * 1e-10f;
    input = add(input, noise);

    auto running_mean = full({8}, 42.0f, DType::Float32, Device::cpu());
    auto running_var = full({8}, 1e-12f, DType::Float32, Device::cpu());
    auto weight_var = Variable(ones({8}, DType::Float32, Device::cpu()), false);
    auto bias_var = Variable(zeros({8}, DType::Float32, Device::cpu()), false);

    auto input_var = Variable(input, false);
    auto result = nn::functional::batch_norm(
        input_var, running_mean, running_var,
        weight_var, bias_var,
        /*training=*/false, /*momentum=*/0.1, /*eps=*/1e-5);
    auto result_t = result.tensor();

    EXPECT_FALSE(has_inf_nan(result_t).data<bool>()[0])
        << "BatchNorm with small variance should not produce NaN/Inf";
}

TEST(NumericalStability, LayerNorm_ConstantInput) {
    // LayerNorm with all-identical elements — variance is zero
    auto input = full({4, 16}, 5.0f, DType::Float32, Device::cpu());
    auto weight_var = Variable(ones({16}, DType::Float32, Device::cpu()), false);
    auto bias_var = Variable(zeros({16}, DType::Float32, Device::cpu()), false);

    auto input_var = Variable(input, false);
    auto result = nn::functional::layer_norm(input_var, {16}, weight_var, bias_var, 1e-5);
    auto result_t = result.tensor();

    EXPECT_FALSE(has_inf_nan(result_t).data<bool>()[0])
        << "LayerNorm with constant input should not produce NaN/Inf";
}

TEST(NumericalStability, CrossEntropy_NearZeroProbabilities) {
    // Cross-entropy with extreme logits (near-0 and near-1 after softmax)
    auto logits = zeros({8, 10}, DType::Float32, Device::cpu());
    for (int i = 0; i < 8; ++i) {
        logits.data<float>()[i * 10 + i % 10] = 100.0f;
    }
    auto targets = zeros({8}, DType::Int64, Device::cpu());
    for (int i = 0; i < 8; ++i) {
        targets.data<int64_t>()[i] = i % 10;
    }

    auto logits_var = Variable(logits, false);
    auto loss = nn::functional::cross_entropy(logits_var, targets);
    auto loss_t = loss.tensor();

    EXPECT_FALSE(has_inf_nan(loss_t).data<bool>()[0])
        << "CrossEntropy with extreme logits should not produce NaN/Inf";
    EXPECT_LT(loss_t.data<float>()[0], 1.0f);
}

TEST(NumericalStability, Softmax_VeryLargeInput) {
    // Softmax with inputs in [1000, 1001] range — tests max-subtraction trick
    auto input = full({4, 32}, 1000.0f, DType::Float32, Device::cpu());
    auto noise = randn({4, 32}, DType::Float32, Device::cpu());
    input = add(input, noise);

    auto input_var = Variable(input, false);
    auto result = softmax(input_var, 1).tensor();

    EXPECT_FALSE(has_inf_nan(result).data<bool>()[0])
        << "Softmax with very large inputs should not produce NaN/Inf";

    auto row_sums = tenzor::sum(result, 1);
    for (int64_t i = 0; i < 4; ++i) {
        float row_sum = row_sums.data<float>()[i];
        EXPECT_NEAR(row_sum, 1.0f, 1e-5f)
            << "Softmax row " << i << " should sum to 1.0";
    }
}

TEST(NumericalStability, KLDiv_NearZero) {
    // KL divergence with near-zero distributions should not produce NaN/Inf
    // KL(P || Q) = sum(P * log(P / Q))
    // When Q -> 0, log(P/Q) -> inf, so stability depends on clamping
    auto p = full({4, 16}, 1.0f / 16.0f, DType::Float32, Device::cpu()); // uniform
    auto q = full({4, 16}, 1e-7f, DType::Float32, Device::cpu()); // near-zero

    auto p_var = Variable(p, false);
    auto q_var = Variable(q, false);

    // log(p) should be stable, log(q) is very negative but finite
    auto log_p = tenzor::log(p);
    auto log_q = tenzor::log(q);

    EXPECT_FALSE(has_inf_nan(log_p).data<bool>()[0])
        << "log of uniform distribution should be finite";
    EXPECT_FALSE(has_inf_nan(log_q).data<bool>()[0])
        << "log of near-zero values should be finite (very negative but not NaN/Inf)";

    // KL divergence: sum(p * (log_p - log_q))
    auto kl = tenzor::sum(tenzor::mul(p, tenzor::sub(log_p, log_q)));
    EXPECT_FALSE(has_inf_nan(kl).data<bool>()[0])
        << "KL divergence with near-zero Q should be finite";
    EXPECT_GT(kl.data<float>()[0], 0.0f)
        << "KL divergence should be positive";
}

TEST(NumericalStability, Attention_LongSequence) {
    // Scaled dot-product attention with long sequence
    // Q @ K^T / sqrt(d_k) can produce very large values before softmax
    int64_t seq_len = 4096;
    int64_t d_k = 64;
    float scale = 1.0f / std::sqrt(static_cast<float>(d_k));

    auto Q = randn({1, seq_len, d_k}, DType::Float32, Device::cpu());
    auto K = randn({1, d_k, seq_len}, DType::Float32, Device::cpu());

    // Compute attention scores: Q @ K^T * scale
    auto scores = tenzor::matmul(Q, K) * scale;

    EXPECT_FALSE(has_inf_nan(scores).data<bool>()[0])
        << "Attention scores should be finite";

    // Apply softmax — this is the critical stability test
    auto scores_2d = scores.reshape({seq_len, seq_len});
    auto scores_var = Variable(scores_2d, false);
    auto attn_weights = softmax(scores_var, 1).tensor();

    EXPECT_FALSE(has_inf_nan(attn_weights).data<bool>()[0])
        << "Attention weights after softmax should be finite";

    // Each row should sum to ~1.0
    auto row_sum = tenzor::sum(attn_weights, 1);
    float first_sum = row_sum.data<float>()[0];
    EXPECT_NEAR(first_sum, 1.0f, 1e-4f)
        << "Attention weights row should sum to 1.0";
}

// ============================================================================
// Gradient Stability
// ============================================================================

TEST(NumericalStability, Gradient_VerySmallValues) {
    tenzor::initialize();
    set_grad_enabled(true);

    // exp(very_small) ~ 1.0, so gradient d/dx exp(x) = exp(x) ~ 1.0
    auto data = full({16}, 1e-20f, DType::Float32, Device::cpu());
    Variable x(data, /*requires_grad=*/true);
    auto y = exp(x);
    auto loss = sum(y);
    loss.backward();

    auto grad = x.grad().value();
    const float* g = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(g[i])) << "gradient is NaN at index " << i;
        EXPECT_FALSE(std::isinf(g[i])) << "gradient is Inf at index " << i;
        // exp(1e-20) ~ 1.0
        EXPECT_NEAR(g[i], 1.0f, 1e-4f) << "gradient should be ~1.0 at index " << i;
    }
}

TEST(NumericalStability, Gradient_VeryLargeValues) {
    tenzor::initialize();
    set_grad_enabled(true);

    // tanh saturates for large values: tanh(1000) ~ 1.0, gradient ~ 0.0
    auto data = full({16}, 1000.0f, DType::Float32, Device::cpu());
    Variable x(data, /*requires_grad=*/true);
    auto y = tanh(x);
    auto loss = sum(y);
    loss.backward();

    auto grad = x.grad().value();
    const float* g = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(g[i])) << "gradient is NaN at index " << i;
        EXPECT_FALSE(std::isinf(g[i])) << "gradient is Inf at index " << i;
        // tanh'(1000) = 1 - tanh(1000)^2 ~ 0.0
        EXPECT_NEAR(g[i], 0.0f, 1e-6f) << "gradient should be ~0.0 at index " << i;
    }
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
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
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
