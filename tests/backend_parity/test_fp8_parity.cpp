/**
 * @file test_fp8_parity.cpp
 * @brief Backend parity for FP8 quantize/dequantize (Phase 5.3).
 *
 * FP8 is hardware-dependent: native on H100 / MI300, software-emulated
 * elsewhere. `fp8_is_native(device)` returns whether a backend has native
 * FP8 support. Parity is tested by quantizing the same FP32 tensor on each
 * backend and comparing the dequantized result — the quantization error
 * should be within a fixed relative tolerance regardless of whether the
 * kernel is native or emulated.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fp8_scaling.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

TEST(FP8Parity, QuantizeDequantize_E4M3_Roundtrip) {
    // Use values in a moderate range so FP8_E4M3 (max ~448) doesn't saturate.
    auto input = randn({8, 16}, DType::Float32, Device::cpu()) * 2.0f;

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (const auto& dev : backends) {
        try {
            auto input_dev = input.to(dev);
            auto [fp8_tensor, params] = quantize_to_fp8(input_dev,
                                                        DType::FP8_E4M3);
            auto recovered = dequantize_from_fp8(fp8_tensor, params.scale);
            dev.synchronize();

            auto recovered_cpu = recovered.to(Device::cpu());
            auto diff = sub(recovered_cpu, input);
            float max_abs = tenzor::max(tenzor::abs(diff)).item<float>();
            float input_max = tenzor::max(tenzor::abs(input)).item<float>();
            float rel_err = max_abs / (input_max + 1e-10f);

            SCOPED_TRACE(std::string("FP8_E4M3 roundtrip on ")
                         + backend_name(dev));
            // FP8_E4M3 has 3 mantissa bits → relative error ~0.125 per value
            EXPECT_LT(rel_err, 0.20f)
                << "Relative error exceeds 20% (likely native/emulated divergence)";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "FP8_E4M3 failed on " << backend_name(dev) << ": "
                      << e.what() << std::endl;
        }
    }
}

TEST(FP8Parity, QuantizeDequantize_E5M2_Roundtrip) {
    auto input = randn({8, 16}, DType::Float32, Device::cpu()) * 2.0f;

    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    for (const auto& dev : backends) {
        try {
            auto input_dev = input.to(dev);
            auto [fp8_tensor, params] = quantize_to_fp8(input_dev,
                                                        DType::FP8_E5M2);
            auto recovered = dequantize_from_fp8(fp8_tensor, params.scale);
            dev.synchronize();

            auto recovered_cpu = recovered.to(Device::cpu());
            auto diff = sub(recovered_cpu, input);
            float max_abs = tenzor::max(tenzor::abs(diff)).item<float>();
            float input_max = tenzor::max(tenzor::abs(input)).item<float>();
            float rel_err = max_abs / (input_max + 1e-10f);

            SCOPED_TRACE(std::string("FP8_E5M2 roundtrip on ")
                         + backend_name(dev));
            // FP8_E5M2 has 2 mantissa bits → relative error up to ~0.25
            EXPECT_LT(rel_err, 0.35f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "FP8_E5M2 failed on " << backend_name(dev) << ": "
                      << e.what() << std::endl;
        }
    }
}

TEST(FP8Parity, NativeQueryPerBackend) {
    // Informational: report which backends claim native FP8. Always passes.
    auto backends = get_available_backends();
    for (const auto& dev : backends) {
        bool native = fp8_is_native(dev.type);
        std::cerr << backend_name(dev) << " native FP8: "
                  << (native ? "yes" : "no") << std::endl;
    }
    SUCCEED();
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
    try { tenzor::finalize(); } catch (...) {}
    return result;
}
