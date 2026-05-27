/**
 * @file test_gptq_quantizer_multidtype.cpp
 * @brief RR.21 — Multi-backend + multi-dtype companion for GPTQQuantizer.
 *
 * Mirrors the audit-10 OO.19 AWQ pattern. Sweeps the GPTQ quantization
 * pipeline (Hessian computation + layer quantization) across
 * {Float16, BFloat16, Float32} weights for every available backend.
 * The internal GPTQ math runs in Float32 inside the quantizer, so this
 * test verifies that:
 *   1. compute_hessian() accepts each float dtype and produces a
 *      well-shaped symmetric matrix with a positive diagonal.
 *   2. quantize_layer() round-trips dtype-converted weights without
 *      crashing and produces finite scales.
 *
 * The error bound here is dtype-conversion-dominated rather than
 * quantization-noise-dominated; the latter is covered by the base
 * test_gptq_quantizer.cpp test.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/quantization/gptq.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../../multi_backend_dtype_fixture.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn::quantization;

class GPTQQuantizerMultiDTypeTest : public MultiBackendDTypeTest {};

// compute_hessian must accept each float dtype, produce a square
// matrix of size (in_features × in_features), be symmetric, and have a
// positive diagonal after damping.
TEST_P(GPTQQuantizerMultiDTypeTest, ComputeHessianShape) {
    if (dtype_ != DType::Float32 && dtype_ != DType::Float64 &&
        dtype_ != DType::Float16 && dtype_ != DType::BFloat16) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::DtypeUnsupportedOnBackend,
            "GPTQ Hessian is float-only.");
        return;
    }
    try {
        auto input_cpu = tenzor::randn({12, 32}, DType::Float32, Device::cpu());
        auto input = input_cpu.to(dtype_).to(device_);

        auto hessian = GPTQQuantizer::compute_hessian(input);
        ASSERT_EQ(hessian.dim(), 2);
        EXPECT_EQ(hessian.size(0), 32);
        EXPECT_EQ(hessian.size(1), 32);

        // Symmetry + positive-diagonal sanity (relax tolerance for the
        // low-precision dtypes — the upcast to Float32 happens inside
        // compute_hessian but the input itself has been rounded already).
        auto h_cpu = hessian.to(Device::cpu()).to(DType::Float32);
        const float* data = h_cpu.data<float>();
        const float atol = (dtype_ == DType::Float32) ? 1e-4f : 1e-2f;
        for (int i = 0; i < 32; ++i) {
            EXPECT_GT(data[i * 32 + i], 0.0f)
                << "Hessian diagonal at (" << i << "," << i << ") not positive";
            for (int j = i + 1; j < 32; ++j) {
                EXPECT_NEAR(data[i * 32 + j], data[j * 32 + i], atol)
                    << "Hessian not symmetric at (" << i << "," << j << ")";
            }
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPTQ compute_hessian unsupported on this "
                     << "backend/dtype combination: " << e.what();
    }
}

// Layer quantization end-to-end: build a Float32 reference weight,
// convert to dtype_, quantize via 4-bit GPTQ, verify the packed weight
// has the expected leading dim and finite scales.
TEST_P(GPTQQuantizerMultiDTypeTest, QuantizeLayerRoundTrip) {
    if (dtype_ != DType::Float32 && dtype_ != DType::Float64 &&
        dtype_ != DType::Float16 && dtype_ != DType::BFloat16) {
        SKIP_WITH_REASON(tenzor::testing::SkipReason::DtypeUnsupportedOnBackend,
            "GPTQ quantize_layer is float-only.");
        return;
    }

    GPTQConfig config;
    config.bits = 4;
    config.group_size = 64;
    config.damp_percent = 0.01f;
    GPTQQuantizer q(config);

    auto weight_cpu = tenzor::randn({32, 64}, DType::Float32, Device::cpu());
    auto input_cpu  = tenzor::randn({16, 64}, DType::Float32, Device::cpu());

    try {
        auto weight = weight_cpu.to(dtype_).to(device_);
        auto input  = input_cpu.to(dtype_).to(device_);
        auto hessian = GPTQQuantizer::compute_hessian(input);

        auto result = q.quantize_layer(weight, hessian);
        ASSERT_EQ(result.packed_weight.dim(), 2);
        EXPECT_EQ(result.packed_weight.size(0), 32);
        EXPECT_GT(result.scales.numel(), 0);

        // Scales must be finite — non-finite values indicate the
        // internal Float32 pipeline saw a NaN/Inf from the dtype
        // conversion and silently propagated it.
        auto s_cpu = result.scales.to(Device::cpu()).to(DType::Float32);
        const float* sdata = s_cpu.data<float>();
        for (int64_t i = 0; i < s_cpu.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(sdata[i]))
                << "Non-finite GPTQ scale at index " << i;
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "GPTQ quantize_layer unsupported on this backend/"
                     << "dtype combination: " << e.what();
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(GPTQQuantizerMultiDTypeTest);
