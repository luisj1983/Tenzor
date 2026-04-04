/**
 * @file test_quantization_parity.cpp
 * @brief Quantization operation parity tests across backends
 *
 * Verifies that quantization-related operations produce identical results
 * across all backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

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
