/**
 * @file test_codegen_elementwise_cpu.cpp
 * @brief Regression coverage for every ElemOp's CPU eager fallback path.
 *
 * Phase P0 / Fix 1: the original `execute_fused` CPU fallback in
 * src/jit/codegen.cpp had a `switch` over `ElemOp` with `default: break;`
 * that silently produced unmodified output for 30 of 45 enum values. These
 * tests build a 1-step FusionGroup for each ElemOp and compare against the
 * eager `tenzor::*` reference; before the fix, 30 of them returned the
 * unmodified input (numerically wrong but not erroring).
 *
 * The tests target `execute_fused_cpu` directly so they exercise the CPU
 * eager path on CUDA-enabled builds too.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/codegen.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

#include <cmath>
#include <span>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class CodegenEltwiseEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new CodegenEltwiseEnv);
}  // namespace

using namespace tenzor;
using namespace tenzor::jit;

namespace {

auto unary_group(ElemOp op, double scalar = 0.0) -> FusionGroup {
    return build_fusion({{op, 0, -1, scalar}}, 1, DType::Float32);
}

auto binary_group(ElemOp op) -> FusionGroup {
    return build_fusion({{op, 0, 1, 0.0}}, 2, DType::Float32);
}

// Compare every element of `actual` against `expected` within `atol`.
auto expect_close(const Tensor& actual, const Tensor& expected, float atol = 1e-5f)
    -> ::testing::AssertionResult {
    if (actual.numel() != expected.numel()) {
        return ::testing::AssertionFailure()
            << "numel mismatch: " << actual.numel() << " vs " << expected.numel();
    }
    auto a = actual.to(Device::cpu()).contiguous();
    auto e = expected.to(Device::cpu()).contiguous();
    const float* ap = a.data<float>();
    const float* ep = e.data<float>();
    for (int64_t i = 0; i < a.numel(); ++i) {
        const float diff = std::abs(ap[i] - ep[i]);
        const bool both_nan = std::isnan(ap[i]) && std::isnan(ep[i]);
        if (!both_nan && diff > atol) {
            return ::testing::AssertionFailure()
                << "mismatch at i=" << i << ": got " << ap[i] << " expected " << ep[i]
                << " (diff " << diff << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

}  // namespace

// =========================================================================
// Unary ops — these all used to silently return the unmodified input.
// =========================================================================

TEST(CodegenElemOps, Unary_Sign) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Sign), {in});
    EXPECT_TRUE(expect_close(out, tenzor::sign(in)));
}

TEST(CodegenElemOps, Unary_Reciprocal) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::add(in, 2.0);  // shift away from zero so 1/x is well-defined
    auto out = execute_fused_cpu(unary_group(ElemOp::Reciprocal), {in});
    EXPECT_TRUE(expect_close(out, tenzor::reciprocal(in)));
}

TEST(CodegenElemOps, Unary_Tan) {
    auto in = tenzor::randn({8}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Tan), {in});
    EXPECT_TRUE(expect_close(out, tenzor::tan(in)));
}

TEST(CodegenElemOps, Unary_Asin) {
    auto raw = tenzor::randn({8}, DType::Float32, Device::cpu());
    // squash to [-0.9, 0.9] so asin is finite
    auto in = tenzor::mul(tenzor::tanh(raw), 0.9);
    auto out = execute_fused_cpu(unary_group(ElemOp::Asin), {in});
    EXPECT_TRUE(expect_close(out, tenzor::asin(in)));
}

TEST(CodegenElemOps, Unary_Acos) {
    auto raw = tenzor::randn({8}, DType::Float32, Device::cpu());
    auto in = tenzor::mul(tenzor::tanh(raw), 0.9);
    auto out = execute_fused_cpu(unary_group(ElemOp::Acos), {in});
    EXPECT_TRUE(expect_close(out, tenzor::acos(in)));
}

TEST(CodegenElemOps, Unary_Atan) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Atan), {in});
    EXPECT_TRUE(expect_close(out, tenzor::atan(in)));
}

TEST(CodegenElemOps, Unary_Sinh) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Sinh), {in});
    EXPECT_TRUE(expect_close(out, tenzor::sinh(in)));
}

TEST(CodegenElemOps, Unary_Cosh) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Cosh), {in});
    EXPECT_TRUE(expect_close(out, tenzor::cosh(in)));
}

TEST(CodegenElemOps, Unary_Erfc) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Erfc), {in});
    EXPECT_TRUE(expect_close(out, tenzor::erfc(in)));
}

TEST(CodegenElemOps, Unary_Log2) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::add(tenzor::abs(in), 0.1);  // log domain
    auto out = execute_fused_cpu(unary_group(ElemOp::Log2), {in});
    EXPECT_TRUE(expect_close(out, tenzor::log2(in)));
}

TEST(CodegenElemOps, Unary_Log10) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::add(tenzor::abs(in), 0.1);
    auto out = execute_fused_cpu(unary_group(ElemOp::Log10), {in});
    EXPECT_TRUE(expect_close(out, tenzor::log10(in)));
}

TEST(CodegenElemOps, Unary_Log1p) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::add(tenzor::abs(in), 0.1);
    auto out = execute_fused_cpu(unary_group(ElemOp::Log1p), {in});
    EXPECT_TRUE(expect_close(out, tenzor::log1p(in)));
}

TEST(CodegenElemOps, Unary_Exp2) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Exp2), {in});
    EXPECT_TRUE(expect_close(out, tenzor::exp2(in)));
}

TEST(CodegenElemOps, Unary_Expm1) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Expm1), {in});
    EXPECT_TRUE(expect_close(out, tenzor::expm1(in)));
}

TEST(CodegenElemOps, Unary_Floor) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::mul(in, 10.0);
    auto out = execute_fused_cpu(unary_group(ElemOp::Floor), {in});
    EXPECT_TRUE(expect_close(out, tenzor::floor(in)));
}

TEST(CodegenElemOps, Unary_Ceil) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::mul(in, 10.0);
    auto out = execute_fused_cpu(unary_group(ElemOp::Ceil), {in});
    EXPECT_TRUE(expect_close(out, tenzor::ceil(in)));
}

TEST(CodegenElemOps, Unary_Round) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::mul(in, 10.0);
    auto out = execute_fused_cpu(unary_group(ElemOp::Round), {in});
    EXPECT_TRUE(expect_close(out, tenzor::round(in)));
}

// =========================================================================
// Activations — routed through OpId dispatch in the CPU fallback. The fix
// must match eager kernel output byte-for-byte.
// =========================================================================

namespace {
// Helper: dispatch a single-input activation OpId with given attrs and return
// the resulting Tensor (the eager reference for the JIT codegen fallback).
auto dispatch_activation(OpId op, const Tensor& in, const OpAttributes& attrs = {})
    -> Tensor {
    const Tensor in_arr[1] = {in};
    return tenzor::dispatch(op, std::span<const Tensor>{in_arr, 1}, attrs)[0];
}
}  // namespace

TEST(CodegenElemOps, Activation_LeakyRelu) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    // Default negative_slope = 0.01 when step.scalar == 0.
    auto out = execute_fused_cpu(unary_group(ElemOp::LeakyRelu, 0.0), {in});
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 0.01);
    EXPECT_TRUE(expect_close(out, dispatch_activation(OpId::LeakyReLU, in, attrs)));
}

TEST(CodegenElemOps, Activation_Elu) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Elu, 0.0), {in});
    OpAttributes attrs;
    attrs.set(AttrKey::Alpha, 1.0);
    EXPECT_TRUE(expect_close(out, dispatch_activation(OpId::Elu, in, attrs)));
}

TEST(CodegenElemOps, Activation_Selu) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Selu), {in});
    EXPECT_TRUE(expect_close(out, dispatch_activation(OpId::Selu, in)));
}

TEST(CodegenElemOps, Activation_Gelu) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Gelu), {in});
    EXPECT_TRUE(expect_close(out, dispatch_activation(OpId::Gelu, in), 1e-4f));
}

TEST(CodegenElemOps, Activation_Mish) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Mish), {in});
    EXPECT_TRUE(expect_close(out, dispatch_activation(OpId::Mish, in), 1e-4f));
}

TEST(CodegenElemOps, Activation_Softplus) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::Softplus, 0.0), {in});
    OpAttributes attrs;
    attrs.set(AttrKey::Beta, 1.0);
    attrs.set(AttrKey::Threshold, 20.0);
    EXPECT_TRUE(expect_close(out, dispatch_activation(OpId::Softplus, in, attrs), 1e-4f));
}

// =========================================================================
// Binary ops — Max / Min / Fmod / Pow used to silently return the unchanged
// first input.
// =========================================================================

TEST(CodegenElemOps, Binary_Max) {
    auto a = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto b = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(binary_group(ElemOp::Max), {a, b});
    EXPECT_TRUE(expect_close(out, tenzor::maximum(a, b)));
}

TEST(CodegenElemOps, Binary_Min) {
    auto a = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto b = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(binary_group(ElemOp::Min), {a, b});
    EXPECT_TRUE(expect_close(out, tenzor::minimum(a, b)));
}

TEST(CodegenElemOps, Binary_Fmod) {
    auto a = tenzor::randn({16}, DType::Float32, Device::cpu());
    a = tenzor::mul(a, 10.0);
    auto b = tenzor::randn({16}, DType::Float32, Device::cpu());
    b = tenzor::add(tenzor::abs(b), 0.5);
    auto out = execute_fused_cpu(binary_group(ElemOp::Fmod), {a, b});
    EXPECT_TRUE(expect_close(out, tenzor::fmod(a, b)));
}

TEST(CodegenElemOps, Binary_Pow) {
    auto a = tenzor::randn({16}, DType::Float32, Device::cpu());
    a = tenzor::add(tenzor::abs(a), 0.1);  // positive base
    auto b = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(binary_group(ElemOp::Pow), {a, b});
    const Tensor in_arr[2] = {a, b};
    auto expected = tenzor::dispatch(OpId::Pow,
                                     std::span<const Tensor>{in_arr, 2}, {})[0];
    EXPECT_TRUE(expect_close(out, expected, 1e-3f));
}

// =========================================================================
// Scalar binary ops — PowScalar / ClampMin / ClampMax used to silently
// return the unchanged input.
// =========================================================================

TEST(CodegenElemOps, ScalarBinary_PowScalar) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    in = tenzor::add(tenzor::abs(in), 0.1);
    auto out = execute_fused_cpu(unary_group(ElemOp::PowScalar, 2.5), {in});
    EXPECT_TRUE(expect_close(out, tenzor::pow(in, 2.5f), 1e-4f));
}

TEST(CodegenElemOps, ScalarBinary_ClampMin) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::ClampMin, 0.5), {in});
    EXPECT_TRUE(expect_close(out, tenzor::clamp_min(in, 0.5f)));
}

TEST(CodegenElemOps, ScalarBinary_ClampMax) {
    auto in = tenzor::randn({16}, DType::Float32, Device::cpu());
    auto out = execute_fused_cpu(unary_group(ElemOp::ClampMax, 0.5), {in});
    EXPECT_TRUE(expect_close(out, tenzor::clamp_max(in, 0.5f)));
}

// =========================================================================
// Regression guard: an op that *was* previously handled correctly should
// still produce identical output (catches regressions where refactoring
// broke a previously-working case).
// =========================================================================

TEST(CodegenElemOps, Regression_ReluSigmoidUnchanged) {
    auto in = tenzor::randn({32}, DType::Float32, Device::cpu());
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::Sigmoid, -1, -1, 0.0},
    }, 1, DType::Float32);
    auto out = execute_fused_cpu(group, {in});
    auto expected = tenzor::sigmoid(tenzor::clamp_min(in, 0.0f));
    EXPECT_TRUE(expect_close(out, expected));
}
