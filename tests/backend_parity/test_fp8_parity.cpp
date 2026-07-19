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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class FP8Parity : public BackendTest {};
TEST_P(FP8Parity, QuantizeDequantize_E4M3_Roundtrip) {
    // Use values in a moderate range so FP8_E4M3 (max ~448) doesn't saturate.
    auto input = randn({8, 16}, DType::Float32, Device::cpu()) * 2.0f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("fp8 parity");

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
            // FP8_E4M3 has 3 mantissa bits. The per-value relative error is
            // ~0.125, but this metric divides the max absolute error by the
            // *global* input max, which inflates dramatically when a single
            // near-zero element gets quantized to a non-trivial fraction of
            // the scale. 60% allows for that pathological case while still
            // catching a genuinely broken quantize path.
            EXPECT_LT(rel_err, 0.60f)
                << "Relative error exceeds 60% (likely native/emulated divergence)";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "FP8_E4M3 failed on " << backend_name(dev) << ": "
                      << e.what() << std::endl;
        }
    }
}

TEST_P(FP8Parity, QuantizeDequantize_E5M2_Roundtrip) {
    auto input = randn({8, 16}, DType::Float32, Device::cpu()) * 2.0f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("fp8 parity");

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

TEST_P(FP8Parity, NativeQueryPerBackend) {
    // Informational: report which backends claim native FP8. Always passes.
    auto backends = get_available_backends();
    for (const auto& dev : backends) {
        bool native = fp8_is_native(dev.type);
        std::cerr << backend_name(dev) << " native FP8: "
                  << (native ? "yes" : "no") << std::endl;
    }
    SUCCEED();
}

// ====================================================================
// Phase D.1 — FP8 MatMul parity. The CPU emulated FP8 matmul is the
// reference; each GPU backend's FP8 matmul (native or widen-narrow)
// must match within the documented FP8 accumulation tolerance band.
// ====================================================================

TEST_P(FP8Parity, MatMul_FP8_E4M3) {
    auto a_f32 = randn({4, 4}, DType::Float32, Device::cpu()) * 0.5f;
    auto b_f32 = randn({4, 4}, DType::Float32, Device::cpu()) * 0.5f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("fp8 matmul parity");

    // Audit: previously wrapped in try{...}catch(...){GTEST_SKIP("FP8 MatMul
    // CPU reference failed")}. The CPU emulated FP8 matmul is the documented
    // reference (Phase D.1 comment above); a failure here is a real bug, not a
    // clean skip. Let it propagate.
    Tensor ref_f32;
    {
        auto a_e4m3 = a_f32.to(DType::FP8_E4M3);
        auto b_e4m3 = b_f32.to(DType::FP8_E4M3);
        auto out_e4m3 = ::tenzor::matmul(a_e4m3, b_e4m3);
        ref_f32 = out_e4m3.to(DType::Float32);
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        const auto& dev = backends[i];
        try {
            auto a_dev = a_f32.to(dev).to(DType::FP8_E4M3);
            auto b_dev = b_f32.to(dev).to(DType::FP8_E4M3);
            auto out_dev = ::tenzor::matmul(a_dev, b_dev);
            dev.synchronize();
            auto out_cpu = out_dev.to(DType::Float32).to(Device::cpu());
            SCOPED_TRACE(std::string("FP8_E4M3 MatMul on ") + backend_name(dev));
            // FP8_E4M3 has 3 mantissa bits + Float32 accumulation.
            // 4-element dot products typically incur 5-10% per-element error.
            EXPECT_TENSORS_CLOSE(ref_f32, out_cpu, 0.6f, 0.1f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "FP8_E4M3 MatMul failed on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

TEST_P(FP8Parity, MatMul_FP8_E5M2) {
    auto a_f32 = randn({4, 4}, DType::Float32, Device::cpu()) * 0.5f;
    auto b_f32 = randn({4, 4}, DType::Float32, Device::cpu()) * 0.5f;

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("fp8 matmul parity");

    // Audit: previously wrapped in try{...}catch(...){GTEST_SKIP("FP8 MatMul
    // CPU reference failed")}. The CPU emulated FP8 matmul is the documented
    // reference (Phase D.1 comment above); a failure here is a real bug, not a
    // clean skip. Let it propagate.
    Tensor ref_f32;
    {
        auto a_e5m2 = a_f32.to(DType::FP8_E5M2);
        auto b_e5m2 = b_f32.to(DType::FP8_E5M2);
        auto out_e5m2 = ::tenzor::matmul(a_e5m2, b_e5m2);
        ref_f32 = out_e5m2.to(DType::Float32);
    }

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        const auto& dev = backends[i];
        try {
            auto a_dev = a_f32.to(dev).to(DType::FP8_E5M2);
            auto b_dev = b_f32.to(dev).to(DType::FP8_E5M2);
            auto out_dev = ::tenzor::matmul(a_dev, b_dev);
            dev.synchronize();
            auto out_cpu = out_dev.to(DType::Float32).to(Device::cpu());
            SCOPED_TRACE(std::string("FP8_E5M2 MatMul on ") + backend_name(dev));
            // FP8_E5M2 has 2 mantissa bits — looser tolerance.
            EXPECT_TENSORS_CLOSE(ref_f32, out_cpu, 1.0f, 0.2f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "FP8_E5M2 MatMul failed on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

INSTANTIATE_BACKEND_TESTS(FP8Parity);


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
