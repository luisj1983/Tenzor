/**
 * @file test_extended_math_parity.cpp
 * @brief Extended math operation parity tests across backends
 *
 * Tests 21 extended math operations (log2, log10, log1p, exp2, expm1, erf,
 * erfc, rsqrt, square, reciprocal, floor, ceil, round, trunc, frac, erfinv,
 * atan2, fmod, remainder, copysign, hypot) to ensure all backends
 * (CPU, CUDA, ROCm, Vulkan, OneAPI) produce identical results.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class ExtendedMathParity : public BackendTest {};
// ============================================================================
// Unary Extended Math Operations
// ============================================================================

TEST_P(ExtendedMathParity, Log2) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log2(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log2");
}

TEST_P(ExtendedMathParity, Log10) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log10(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log10");
}

TEST_P(ExtendedMathParity, Log1p) {

    auto a = generate_uniform_tensor({32, 32}, -0.5f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return log1p(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Log1p");
}

TEST_P(ExtendedMathParity, Exp2) {

    auto a = generate_uniform_tensor({32, 32}, -5.0f, 5.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return exp2(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Exp2");
}

TEST_P(ExtendedMathParity, Expm1) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return expm1(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Expm1");
}

TEST_P(ExtendedMathParity, Erf) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erf(inputs[0]);
    }, {a}, device, 1e-4f, 1e-6f, "Erf");
}

TEST_P(ExtendedMathParity, Erfc) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erfc(inputs[0]);
    }, {a}, device, 1e-4f, 1e-6f, "Erfc");
}

TEST_P(ExtendedMathParity, Rsqrt) {

    auto a = generate_uniform_tensor({32, 32}, 0.1f, 10.0f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return rsqrt(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Rsqrt");
}

TEST_P(ExtendedMathParity, Square) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return square(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Square");
}

TEST_P(ExtendedMathParity, Reciprocal) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return reciprocal(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Reciprocal");
}

TEST_P(ExtendedMathParity, Floor) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return floor(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Floor");
}

TEST_P(ExtendedMathParity, Ceil) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return ceil(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Ceil");
}

TEST_P(ExtendedMathParity, Round) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return round(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Round");
}

TEST_P(ExtendedMathParity, Trunc) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    // Call tenzor::trunc directly so a broken trunc kernel is actually exercised.
    // Independent reference: trunc(x) == sign(x) * floor(abs(x)), asserted on CPU.
    auto cpu_ref = sign(a) * floor(abs(a));
    auto cpu_trunc = tenzor::trunc(a);
    EXPECT_TRUE(tensors_close(cpu_ref, cpu_trunc, 0.0f, 0.0f))
        << "CPU tenzor::trunc disagrees with sign(x)*floor(abs(x)) reference";

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tenzor::trunc(inputs[0]);
    }, {a}, device, 0.0f, 0.0f, "Trunc");
}

TEST_P(ExtendedMathParity, Frac) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return frac(inputs[0]);
    }, {a}, device, 1e-5f, 1e-7f, "Frac");
}

TEST_P(ExtendedMathParity, Erfinv) {

    auto a = generate_uniform_tensor({32, 32}, -0.9f, 0.9f, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return erfinv(inputs[0]);
    }, {a}, device, 1e-3f, 1e-4f, "Erfinv");
}

// ============================================================================
// Binary Extended Math Operations
// ============================================================================

TEST_P(ExtendedMathParity, Atan2) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return atan2(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Atan2");
}

TEST_P(ExtendedMathParity, Fmod) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return fmod(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Fmod");
}

TEST_P(ExtendedMathParity, Remainder) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu()) + 2.0f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return remainder(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Remainder");
}

TEST_P(ExtendedMathParity, Copysign) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return copysign(inputs[0], inputs[1]);
    }, {a, b}, device, 0.0f, 0.0f, "Copysign");
}

TEST_P(ExtendedMathParity, Hypot) {

    auto a = randn({32, 32}, DType::Float32, Device::cpu());
    auto b = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return hypot(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-5f, 1e-7f, "Hypot");
}

// Phase 6-followup #27: gradient parity for extended math.
TEST_P(ExtendedMathParity, Log2_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return log2(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Log2_Grad");
}

TEST_P(ExtendedMathParity, Log10_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, 0.5f, 5.0f, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return log10(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Log10_Grad");
}

TEST_P(ExtendedMathParity, Log1p_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, -0.5f, 5.0f, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return log1p(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Log1p_Grad");
}

TEST_P(ExtendedMathParity, Exp2_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, -2.0f, 2.0f, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return exp2(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Exp2_Grad");
}

TEST_P(ExtendedMathParity, Expm1_GradientParity) {
    auto a = generate_uniform_tensor({16, 16}, -2.0f, 2.0f, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return expm1(in[0]); },
        {a}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "Expm1_Grad");
}

TEST_P(ExtendedMathParity, Erf_GradientParity) {
    // Slightly looser fwd tol — erf approximations differ across backends
    // by ~5e-7 (sub-Float32-precision but exceeds 1e-7 atol).
    auto a = randn({16, 16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return erf(in[0]); },
        {a}, {}, 1e-5f, 1e-6f, 1e-4f, 1e-5f, {}, "Erf_Grad");
}

TEST_P(ExtendedMathParity, Erfc_GradientParity) {
    auto a = randn({16, 16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable { return erfc(in[0]); },
        {a}, {}, 1e-5f, 1e-6f, 1e-4f, 1e-5f, {}, "Erfc_Grad");
}

// ============================================================================
// Wave A — CPU baseline native F16/BF16 paths (no widen-narrow workarounds)
// ============================================================================
//
// These tests verify that CPU kernels produce correct results for Float16 and
// BFloat16 inputs WITHOUT the tensor-wide widen-narrow workaround (input.to
// (Float32).op().to(orig)). Per-element in-register F32 promotion remains
// allowed (e.g. std::tan has no half overload), but the kernel itself must
// dispatch on the native dtype and return the requested precision directly.
//
// Current state at time of writing (Wave A):
//   - Tan_F16/BF16, Minimum_F16/BF16, Maximum_F16/BF16: characterization tests
//     — they PASS today because the widen-narrow path produces correct values.
//     They remain green after the refactor (workaround removal). They prevent
//     regression.
//   - Linspace_F16/BF16: GREEN today — the *public* linspace() in
//     src/ops/creation.cpp short-circuits CPU dispatch and handles F16/BF16
//     directly (see lines 738-762 there). The dead linspace_kernel in
//     src/backends/cpu/kernels/creation.cpp that threw on non-F32/F64 is
//     unreachable; it should be removed as cleanup but does not block this
//     wave. The tests are regression coverage for the public-API path.
// ============================================================================

namespace {

// Cast helper: tensors_close requires matching dtypes. We compare F32 reference
// to F16/BF16 result by promoting both sides to Float32 (lossless from F16/BF16).
inline auto as_f32(const Tensor& t) -> Tensor { return t.to(DType::Float32); }

}  // namespace

TEST(WaveA_CpuBaseline, Tan_F16) {
    // Scale away from the pi/2 poles to keep values within F16's +/-65504 range.
    auto x_f32 = randn({64, 128}, DType::Float32, Device::cpu()) * 1.4f;
    auto x_f16 = x_f32.to(DType::Float16);
    // Reference: compute tan in F32 on the *F16-quantized* inputs (not the
    // original F32 values), then narrow. This is the function `tan(F16->F32)`
    // — exactly what a correct native F16 kernel computes per element.
    auto x_f16_in_f32 = x_f16.to(DType::Float32);
    auto y_ref = tan(x_f16_in_f32).to(DType::Float16);
    auto y_test = tan(x_f16);
    ASSERT_EQ(y_test.dtype(), DType::Float16);
    EXPECT_TRUE(tensors_close(as_f32(y_ref), as_f32(y_test), 5e-3f, 5e-3f));
}

TEST(WaveA_CpuBaseline, Tan_BF16) {
    auto x_f32 = randn({64, 128}, DType::Float32, Device::cpu()) * 1.4f;
    auto x_bf16 = x_f32.to(DType::BFloat16);
    auto x_bf16_in_f32 = x_bf16.to(DType::Float32);
    auto y_ref = tan(x_bf16_in_f32).to(DType::BFloat16);
    auto y_test = tan(x_bf16);
    ASSERT_EQ(y_test.dtype(), DType::BFloat16);
    EXPECT_TRUE(tensors_close(as_f32(y_ref), as_f32(y_test), 1e-2f, 1e-2f));
}

TEST(WaveA_CpuBaseline, Minimum_F16) {
    auto a_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto b_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto y_ref = minimum(a_f32, b_f32).to(DType::Float16);
    auto y_test = minimum(a_f32.to(DType::Float16), b_f32.to(DType::Float16));
    ASSERT_EQ(y_test.dtype(), DType::Float16);
    EXPECT_TRUE(tensors_close(as_f32(y_ref), as_f32(y_test), 5e-3f, 5e-3f));
}

TEST(WaveA_CpuBaseline, Minimum_BF16) {
    auto a_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto b_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto y_ref = minimum(a_f32, b_f32).to(DType::BFloat16);
    auto y_test = minimum(a_f32.to(DType::BFloat16), b_f32.to(DType::BFloat16));
    ASSERT_EQ(y_test.dtype(), DType::BFloat16);
    EXPECT_TRUE(tensors_close(as_f32(y_ref), as_f32(y_test), 1e-2f, 1e-2f));
}

TEST(WaveA_CpuBaseline, Maximum_F16) {
    auto a_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto b_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto y_ref = maximum(a_f32, b_f32).to(DType::Float16);
    auto y_test = maximum(a_f32.to(DType::Float16), b_f32.to(DType::Float16));
    ASSERT_EQ(y_test.dtype(), DType::Float16);
    EXPECT_TRUE(tensors_close(as_f32(y_ref), as_f32(y_test), 5e-3f, 5e-3f));
}

TEST(WaveA_CpuBaseline, Maximum_BF16) {
    auto a_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto b_f32 = randn({64, 64}, DType::Float32, Device::cpu());
    auto y_ref = maximum(a_f32, b_f32).to(DType::BFloat16);
    auto y_test = maximum(a_f32.to(DType::BFloat16), b_f32.to(DType::BFloat16));
    ASSERT_EQ(y_test.dtype(), DType::BFloat16);
    EXPECT_TRUE(tensors_close(as_f32(y_ref), as_f32(y_test), 1e-2f, 1e-2f));
}

TEST(WaveA_CpuBaseline, Linspace_F16) {
    // Genuine red: linspace_kernel throws on Float16/BFloat16 at time of writing.
    auto y_f32 = linspace(-1.0f, 1.0f, 17, DType::Float32, Device::cpu());
    auto y_test = linspace(-1.0f, 1.0f, 17, DType::Float16, Device::cpu());
    ASSERT_EQ(y_test.dtype(), DType::Float16);
    EXPECT_TRUE(tensors_close(y_f32, as_f32(y_test), 5e-3f, 5e-3f));
}

TEST(WaveA_CpuBaseline, Linspace_BF16) {
    auto y_f32 = linspace(-1.0f, 1.0f, 17, DType::Float32, Device::cpu());
    auto y_test = linspace(-1.0f, 1.0f, 17, DType::BFloat16, Device::cpu());
    ASSERT_EQ(y_test.dtype(), DType::BFloat16);
    EXPECT_TRUE(tensors_close(y_f32, as_f32(y_test), 1e-2f, 1e-2f));
}

INSTANTIATE_BACKEND_TESTS(ExtendedMathParity);




int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
