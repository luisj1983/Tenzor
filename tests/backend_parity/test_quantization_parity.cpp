/**
 * @file test_quantization_parity.cpp
 * @brief Quantization operation parity tests across backends
 *
 * Verifies that quantization-related operations produce identical results
 * across all backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/quantize.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn::quantization;

// ============================================================================
// Quantization Simulation Parity
// ============================================================================

TEST(QuantizationParity, QuantDequant_Roundtrip) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Simulate INT8 quantize → dequantize roundtrip
        auto clamped = clamp(inputs[0], -1.0f, 1.0f);
        auto scaled = clamped * 127.0f;
        auto rounded = round(scaled);
        return rounded / 127.0f;
    }, {input}, 1e-5f, 1e-6f, "Quantization roundtrip simulation");
}

TEST(QuantizationParity, SymmetricQuantization) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Symmetric quantization: scale = max(abs(input)) / 127
        auto abs_input = abs(inputs[0]);
        auto max_val = max(abs_input);
        auto scale = max_val / 127.0f;
        // Quantize
        auto quantized = round(inputs[0] / scale);
        auto clamped = clamp(quantized, -127.0f, 127.0f);
        // Dequantize
        return clamped * scale;
    }, {input}, 1e-4f, 1e-5f, "Symmetric quantization");
}

// ============================================================================
// Actual Quantization API Tests
// ============================================================================

TEST(QuantizationParity, PerTensorSymmetric_Int8) {
    auto input = randn({4, 16}, DType::Float32, Device::cpu());

    auto qt = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    auto recovered = qt.dequantize();

    // Dequantized should be close to original (within quantization error)
    auto diff = sub(recovered, input);
    auto max_err = tenzor::max(tenzor::abs(diff)).data<float>()[0];
    auto input_range = tenzor::max(tenzor::abs(input)).data<float>()[0];
    float relative_err = max_err / (input_range + 1e-10f);

    // INT8 has 256 levels, so relative error should be < ~1%
    EXPECT_LT(relative_err, 0.02f)
        << "Per-tensor symmetric INT8 roundtrip should have < 2% relative error";
}

TEST(QuantizationParity, PerTensorAsymmetric_Int8) {
    // Asymmetric input (all positive)
    auto input = tenzor::abs(randn({4, 16}, DType::Float32, Device::cpu()));

    auto qt = quantize_per_tensor_asymmetric(input, QuantDType::INT8);
    auto recovered = qt.dequantize();

    auto diff = sub(recovered, input);
    auto max_err = tenzor::max(tenzor::abs(diff)).data<float>()[0];
    auto input_range = tenzor::max(input).data<float>()[0];
    float relative_err = max_err / (input_range + 1e-10f);

    EXPECT_LT(relative_err, 0.02f)
        << "Per-tensor asymmetric INT8 roundtrip should have < 2% relative error";
}

TEST(QuantizationParity, QuantizedTensor_Properties) {
    auto input = randn({8, 8}, DType::Float32, Device::cpu());
    auto qt = quantize_per_tensor_symmetric(input, QuantDType::INT8);

    auto scale_val = qt.params().scale.data<float>()[0];
    EXPECT_GT(scale_val, 0.0f) << "Scale should be positive";
    auto zp_val = qt.params().zero_point.data<int32_t>()[0];
    EXPECT_EQ(zp_val, 0) << "Symmetric quantization has zero_point=0";
    EXPECT_EQ(qt.params().dtype, QuantDType::INT8);
}

TEST(QuantizationParity, DequantRoundtripPreservesShape) {
    auto input = randn({2, 3, 4}, DType::Float32, Device::cpu());
    auto qt = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    auto recovered = qt.dequantize();

    auto orig_shape = input.shape();
    auto recov_shape = recovered.shape();
    ASSERT_EQ(orig_shape.size(), recov_shape.size());
    for (size_t i = 0; i < orig_shape.size(); ++i) {
        EXPECT_EQ(orig_shape[i], recov_shape[i]);
    }
}

TEST(QuantizationParity, SymmetricQuantization_Backend) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto abs_input = abs(inputs[0]);
        auto max_val = max(abs_input);
        auto scale = max_val / 127.0f;
        auto quantized = round(inputs[0] / scale);
        auto clamped = clamp(quantized, -127.0f, 127.0f);
        return clamped * scale;
    }, {input}, 1e-4f, 1e-5f, "Symmetric quantization across backends");
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
