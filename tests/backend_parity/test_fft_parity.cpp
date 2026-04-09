/**
 * @file test_fft_parity.cpp
 * @brief FFT operation parity tests across backends
 *
 * Verifies that FFT/IFFT/RFFT produce identical results across
 * CPU, CUDA, ROCm, OneAPI, and Vulkan backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// 1-D FFT Parity
// ============================================================================

TEST(FFTParity, FFT_1D_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::fft::fft(inputs[0]);
    }, {input}, 1e-4f, 1e-5f, "FFT 1D");
}

TEST(FFTParity, IFFT_1D_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // FFT then IFFT should round-trip
        auto freq = tenzor::fft::fft(inputs[0]);
        return tenzor::fft::ifft(freq);
    }, {input}, 1e-4f, 1e-5f, "FFT→IFFT roundtrip");
}

TEST(FFTParity, RFFT_1D_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::fft::rfft(inputs[0]);
    }, {input}, 1e-4f, 1e-5f, "RFFT 1D");
}

TEST(FFTParity, FFT_1D_WithLength) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Zero-pad to 32 elements
        return tenzor::fft::fft(inputs[0], 32);
    }, {input}, 1e-4f, 1e-5f, "FFT 1D zero-padded");
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
