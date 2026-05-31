/**
 * @file test_fused_conv_activation_dtype.cpp
 * @brief Stream S4: fused conv+activation kernels must apply the
 *        activation step on Float16 / BFloat16 inputs (not silently no-op).
 *
 * The CPU implementations in src/backends/cpu/kernels/fused_ops.cpp used to
 * gate their activation/clamp step on a hard-coded `dtype == Float32 ||
 * dtype == Float64` check. For Float16 / BFloat16 inputs the conv ran but
 * the activation was skipped silently — callers got plain conv output and
 * thought they had conv+ReLU / sigmoid / tanh / swish.
 *
 * S4 wraps each kernel with `tenzor::utils::widen_narrow_compute`, so half-
 * precision inputs are widened to Float32 for the activation step and
 * narrowed back to the original dtype on return.
 *
 * This test asserts:
 *   1) Float16 / BFloat16 fused output matches the Float32 reference for
 *      each of the 5 kernels (random input/weight/bias).
 *   2) With zero input and zero bias, each fused kernel returns the
 *      expected scalar value of the activation at zero:
 *        relu(0) = 0,  sigmoid(0) = 0.5,  tanh(0) = 0,  swish(0) = 0.
 *      Crucially, this catches the regression where activation is skipped
 *      on F16 — sigmoid(0)=0.5 vs. raw conv output of 0 distinguishes the
 *      two cases unambiguously.
 */

#include "backend_test_fixture.hpp"

#include <tenzor/tenzor.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/fused_ops.hpp>

#include <cmath>
#include <cstdint>
#include <random>
#include <vector>

namespace {

using tenzor::DType;
using tenzor::Tensor;
using tenzor::Float16;
using tenzor::BFloat16;

class FusedConvActivationDtype : public ::tenzor::testing::BackendTest {};

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

// Build a Float32 tensor of `shape` populated from `data` (row-major) and
// move it onto `device`. Host writes happen on CPU first, then `.to(device)`.
auto make_f32(const std::vector<int64_t>& shape,
              const std::vector<float>& data,
              const tenzor::Device& device) -> Tensor {
    Tensor host(shape, DType::Float32, tenzor::Device::cpu());
    float* dst = host.data<float>();
    for (size_t i = 0; i < data.size(); ++i) dst[i] = data[i];
    return host.to(device);
}

// Fill `shape`-sized buffer with deterministic pseudo-random Float32 in
// [-lo, hi] and place it on `device`. We keep magnitudes small so that
// BFloat16's 7-bit mantissa is the tolerance driver, not Float16 overflow.
auto rand_f32(const std::vector<int64_t>& shape, uint32_t seed,
              const tenzor::Device& device,
              float lo = -0.5f, float hi = 0.5f) -> Tensor {
    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(lo, hi);
    int64_t numel = 1;
    for (auto d : shape) numel *= d;
    std::vector<float> v(static_cast<size_t>(numel));
    for (auto& x : v) x = dist(rng);
    return make_f32(shape, v, device);
}

// Maximum absolute element-wise difference between two same-shape Float32
// tensors. Both inputs are expected to be Float32-typed already. Reads run
// on CPU copies.
auto max_abs_diff_f32(const Tensor& a, const Tensor& b) -> float {
    EXPECT_EQ(a.dtype(), DType::Float32);
    EXPECT_EQ(b.dtype(), DType::Float32);
    EXPECT_EQ(a.numel(), b.numel());
    Tensor a_cpu = a.cpu();
    Tensor b_cpu = b.cpu();
    const float* da = a_cpu.data<float>();
    const float* db = b_cpu.data<float>();
    float m = 0.0f;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        float d = std::fabs(da[i] - db[i]);
        if (d > m) m = d;
    }
    return m;
}

// Tolerances for half-precision parity vs. the Float32 reference.
// Float16: 10-bit mantissa => ~1e-3 baseline; we allow 5e-3 for chained
// matmul + activation.
// BFloat16: 7-bit mantissa => ~7e-3 baseline; we allow 1.5e-2.
constexpr float kTolF16 = 5e-3f;
constexpr float kTolBF16 = 1.5e-2f;

// Shapes shared by the conv tests:
//   N=2, C_in=3, H=8, W=8
//   C_out=4, KH=3, KW=3
//   stride=1, padding=1  -> output spatial (H, W)
constexpr int64_t kN = 2;
constexpr int64_t kCin = 3;
constexpr int64_t kH = 8;
constexpr int64_t kW = 8;
constexpr int64_t kCout = 4;
constexpr int64_t kK = 3;
constexpr int64_t kStride = 1;
constexpr int64_t kPad = 1;

// Run a fused conv+activation public-API entry point across (F32 ref,
// F16 cast, BF16 cast) and assert half precision matches the F32 ref.
template <typename FusedFn>
void check_conv_half_parity(const char* name, FusedFn&& fused, uint32_t seed,
                            const tenzor::Device& device) {
    Tensor x32 = rand_f32({kN, kCin, kH, kW}, seed, device);
    Tensor w32 = rand_f32({kCout, kCin, kK, kK}, seed + 1u, device);
    Tensor b32 = rand_f32({kCout}, seed + 2u, device);

    Tensor ref = fused(x32, w32, &b32);
    ASSERT_EQ(ref.dtype(), DType::Float32) << name << ": F32 ref dtype";

    {
        Tensor x16 = x32.to(DType::Float16);
        Tensor w16 = w32.to(DType::Float16);
        Tensor b16 = b32.to(DType::Float16);
        Tensor y16 = fused(x16, w16, &b16);
        ASSERT_EQ(y16.dtype(), DType::Float16) << name << ": F16 result dtype";
        Tensor y16_as_f32 = y16.to(DType::Float32);
        float diff = max_abs_diff_f32(y16_as_f32, ref);
        EXPECT_LT(diff, kTolF16) << name << " (Float16) max_abs_diff=" << diff;
    }

    {
        Tensor xb = x32.to(DType::BFloat16);
        Tensor wb = w32.to(DType::BFloat16);
        Tensor bb = b32.to(DType::BFloat16);
        Tensor yb = fused(xb, wb, &bb);
        ASSERT_EQ(yb.dtype(), DType::BFloat16) << name << ": BF16 result dtype";
        Tensor yb_as_f32 = yb.to(DType::Float32);
        float diff = max_abs_diff_f32(yb_as_f32, ref);
        EXPECT_LT(diff, kTolBF16) << name << " (BFloat16) max_abs_diff=" << diff;
    }
}

// Run the fused_linear_relu public-API entry point across (F32 ref, F16,
// BF16) and assert half precision matches.
void check_linear_half_parity(uint32_t seed, const tenzor::Device& device) {
    constexpr int64_t kBatch = 4;
    constexpr int64_t kInF = 6;
    constexpr int64_t kOutF = 5;
    Tensor x32 = rand_f32({kBatch, kInF}, seed, device);
    Tensor w32 = rand_f32({kOutF, kInF}, seed + 1u, device);
    Tensor b32 = rand_f32({kOutF}, seed + 2u, device);

    Tensor ref = tenzor::ops::fused_linear_relu(x32, w32, &b32);
    ASSERT_EQ(ref.dtype(), DType::Float32);

    {
        Tensor x16 = x32.to(DType::Float16);
        Tensor w16 = w32.to(DType::Float16);
        Tensor b16 = b32.to(DType::Float16);
        Tensor y16 = tenzor::ops::fused_linear_relu(x16, w16, &b16);
        ASSERT_EQ(y16.dtype(), DType::Float16);
        float diff = max_abs_diff_f32(y16.to(DType::Float32), ref);
        EXPECT_LT(diff, kTolF16) << "fused_linear_relu (F16) max_abs_diff=" << diff;
    }
    {
        Tensor xb = x32.to(DType::BFloat16);
        Tensor wb = w32.to(DType::BFloat16);
        Tensor bb = b32.to(DType::BFloat16);
        Tensor yb = tenzor::ops::fused_linear_relu(xb, wb, &bb);
        ASSERT_EQ(yb.dtype(), DType::BFloat16);
        float diff = max_abs_diff_f32(yb.to(DType::Float32), ref);
        EXPECT_LT(diff, kTolBF16) << "fused_linear_relu (BF16) max_abs_diff=" << diff;
    }
}

// Build an all-zero tensor of given shape/dtype on `device`.
auto zeros_like_shape(const std::vector<int64_t>& shape, DType dt,
                      const tenzor::Device& device) -> Tensor {
    return tenzor::zeros(shape, dt, device);
}

// Assert every element of `t` (a half-precision tensor) is close to
// `expected` (a Float32 scalar). Used for the zero-input sanity checks.
// Reads run on a CPU copy.
void expect_all_close_to_scalar(const Tensor& t, float expected, float tol,
                                 const char* what) {
    Tensor t32 = t.to(DType::Float32).cpu();
    const float* d = t32.data<float>();
    int64_t n = t32.numel();
    for (int64_t i = 0; i < n; ++i) {
        ASSERT_NEAR(d[i], expected, tol)
            << what << " elem " << i << " expected " << expected << " got " << d[i];
    }
}

// ---------------------------------------------------------------------------
// Half-precision parity: each kernel against its Float32 reference.
// ---------------------------------------------------------------------------

TEST_P(FusedConvActivationDtype, Conv2dReLUHalfParity) {
    auto fn = [](const Tensor& x, const Tensor& w, const Tensor* b) {
        return tenzor::ops::fused_conv2d_relu(x, w, b, kStride, kPad);
    };
    check_conv_half_parity("fused_conv2d_relu", fn, /*seed=*/1u, device);
}

TEST_P(FusedConvActivationDtype, Conv2dSigmoidHalfParity) {
    auto fn = [](const Tensor& x, const Tensor& w, const Tensor* b) {
        return tenzor::ops::fused_conv2d_sigmoid(x, w, b, kStride, kPad);
    };
    check_conv_half_parity("fused_conv2d_sigmoid", fn, /*seed=*/2u, device);
}

TEST_P(FusedConvActivationDtype, Conv2dTanhHalfParity) {
    auto fn = [](const Tensor& x, const Tensor& w, const Tensor* b) {
        return tenzor::ops::fused_conv2d_tanh(x, w, b, kStride, kPad);
    };
    check_conv_half_parity("fused_conv2d_tanh", fn, /*seed=*/3u, device);
}

TEST_P(FusedConvActivationDtype, Conv2dSwishHalfParity) {
    auto fn = [](const Tensor& x, const Tensor& w, const Tensor* b) {
        return tenzor::ops::fused_conv2d_swish(x, w, b, kStride, kPad);
    };
    check_conv_half_parity("fused_conv2d_swish", fn, /*seed=*/4u, device);
}

TEST_P(FusedConvActivationDtype, LinearReLUHalfParity) {
    check_linear_half_parity(/*seed=*/5u, device);
}

// ---------------------------------------------------------------------------
// Zero-input sanity: with all-zero inputs / weight / bias the activation
// step must still execute. If the activation is silently skipped, a fused
// sigmoid would return 0 instead of 0.5 — this is the smoking-gun assertion
// for the original regression.
// ---------------------------------------------------------------------------

template <DType DT>
void zero_input_conv_check(const char* name,
                           float expected,
                           float tol,
                           Tensor (*fused)(const Tensor&, const Tensor&,
                                           const Tensor*, int64_t, int64_t),
                           const tenzor::Device& device) {
    Tensor x = zeros_like_shape({kN, kCin, kH, kW}, DT, device);
    Tensor w = zeros_like_shape({kCout, kCin, kK, kK}, DT, device);
    Tensor b = zeros_like_shape({kCout}, DT, device);
    Tensor y = fused(x, w, &b, kStride, kPad);
    ASSERT_EQ(y.dtype(), DT) << name;
    expect_all_close_to_scalar(y, expected, tol, name);
}

TEST_P(FusedConvActivationDtype, Conv2dReLUZeroInputFloat16) {
    zero_input_conv_check<DType::Float16>(
        "fused_conv2d_relu f16 zero", /*expected=*/0.0f, /*tol=*/1e-4f,
        &tenzor::ops::fused_conv2d_relu, device);
}
TEST_P(FusedConvActivationDtype, Conv2dReLUZeroInputBFloat16) {
    zero_input_conv_check<DType::BFloat16>(
        "fused_conv2d_relu bf16 zero", /*expected=*/0.0f, /*tol=*/1e-4f,
        &tenzor::ops::fused_conv2d_relu, device);
}

TEST_P(FusedConvActivationDtype, Conv2dSigmoidZeroInputFloat16) {
    // sigmoid(0) = 0.5 — distinguishes "activation applied" from
    // "activation silently skipped" (which would yield 0).
    zero_input_conv_check<DType::Float16>(
        "fused_conv2d_sigmoid f16 zero", /*expected=*/0.5f, /*tol=*/2e-3f,
        &tenzor::ops::fused_conv2d_sigmoid, device);
}
TEST_P(FusedConvActivationDtype, Conv2dSigmoidZeroInputBFloat16) {
    zero_input_conv_check<DType::BFloat16>(
        "fused_conv2d_sigmoid bf16 zero", /*expected=*/0.5f, /*tol=*/8e-3f,
        &tenzor::ops::fused_conv2d_sigmoid, device);
}

TEST_P(FusedConvActivationDtype, Conv2dTanhZeroInputFloat16) {
    zero_input_conv_check<DType::Float16>(
        "fused_conv2d_tanh f16 zero", /*expected=*/0.0f, /*tol=*/1e-4f,
        &tenzor::ops::fused_conv2d_tanh, device);
}
TEST_P(FusedConvActivationDtype, Conv2dTanhZeroInputBFloat16) {
    zero_input_conv_check<DType::BFloat16>(
        "fused_conv2d_tanh bf16 zero", /*expected=*/0.0f, /*tol=*/1e-4f,
        &tenzor::ops::fused_conv2d_tanh, device);
}

TEST_P(FusedConvActivationDtype, Conv2dSwishZeroInputFloat16) {
    // swish(0) = 0 * sigmoid(0) = 0
    zero_input_conv_check<DType::Float16>(
        "fused_conv2d_swish f16 zero", /*expected=*/0.0f, /*tol=*/1e-4f,
        &tenzor::ops::fused_conv2d_swish, device);
}
TEST_P(FusedConvActivationDtype, Conv2dSwishZeroInputBFloat16) {
    zero_input_conv_check<DType::BFloat16>(
        "fused_conv2d_swish bf16 zero", /*expected=*/0.0f, /*tol=*/1e-4f,
        &tenzor::ops::fused_conv2d_swish, device);
}

template <DType DT>
void zero_input_linear_relu_check(const char* name, float tol,
                                  const tenzor::Device& device) {
    Tensor x = zeros_like_shape({4, 6}, DT, device);
    Tensor w = zeros_like_shape({5, 6}, DT, device);
    Tensor b = zeros_like_shape({5}, DT, device);
    Tensor y = tenzor::ops::fused_linear_relu(x, w, &b);
    ASSERT_EQ(y.dtype(), DT) << name;
    expect_all_close_to_scalar(y, /*expected=*/0.0f, tol, name);
}

TEST_P(FusedConvActivationDtype, LinearReLUZeroInputFloat16) {
    zero_input_linear_relu_check<DType::Float16>(
        "fused_linear_relu f16 zero", /*tol=*/1e-4f, device);
}
TEST_P(FusedConvActivationDtype, LinearReLUZeroInputBFloat16) {
    zero_input_linear_relu_check<DType::BFloat16>(
        "fused_linear_relu bf16 zero", /*tol=*/1e-4f, device);
}

INSTANTIATE_BACKEND_TESTS(FusedConvActivationDtype);

} // namespace
