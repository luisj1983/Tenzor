/**
 * @file test_depthwise_conv.cpp
 * @brief Stream S18: native CPU DepthwiseConv1d / DepthwiseConv3d kernels.
 *
 * Asserts that the new fast-path kernels (registered as OpId::DepthwiseConv1d
 * and OpId::DepthwiseConv3d in the CPU backend) produce numerically the same
 * output as the generic Conv1d / Conv3d forward path called with
 * `groups == in_channels`. Both paths take identical inputs and attributes,
 * so any divergence is a bug in the new kernel.
 *
 * The depthwise paths must support:
 *   - Float32, Float64, Float16, BFloat16 (last two with widen-narrow)
 *   - stride > 1, padding > 0, dilation > 1
 *   - bias present / absent
 *   - 1-D: NN-layer-style 4-D-with-H==1 dispatch surface (the layer pads the
 *     length axis manually, then unsqueezes — see src/nn/layers/conv.cpp).
 *   - 3-D: native 5-D dispatch with scalar + per-axis attribute keys.
 *
 * The previous registry shipped throw-stubs for these OpIds. The dispatcher
 * is therefore the regression that landing this stream prevents — calling
 * `dispatch(OpId::DepthwiseConv{1,3}d, ...)` used to raise; it must now
 * compute the correct output.
 */

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/core/dtype.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>

#include <cmath>
#include <cstdint>
#include <vector>

namespace tenzor { void initialize(); void finalize(); }

using tenzor::Tensor;
using tenzor::DType;
using tenzor::Device;
using tenzor::OpId;
using tenzor::OpAttributes;
using tenzor::AttrKey;
using tenzor::dispatch;
using tenzor::randn;
using tenzor::zeros;

namespace {

// Returns max-abs and max-rel divergence between two same-shape, same-dtype
// tensors after casting to Float64 for comparison. We use Float64 because
// the F16/BF16 paths have measurable rounding noise we want to characterise
// numerically rather than mask by quantising the comparison.
struct ToleranceReport {
    double max_abs;
    double max_rel;
};
ToleranceReport compare(const Tensor& a, const Tensor& b) {
    auto a64 = a.to(DType::Float64).contiguous();
    auto b64 = b.to(DType::Float64).contiguous();
    const double* pa = a64.data<double>();
    const double* pb = b64.data<double>();
    const int64_t n = a64.numel();
    double max_abs = 0.0;
    double max_rel = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double da = pa[i];
        const double db = pb[i];
        const double diff = std::abs(da - db);
        const double denom = std::max(std::abs(da), std::abs(db));
        if (diff > max_abs) max_abs = diff;
        if (denom > 1e-12 && diff / denom > max_rel) max_rel = diff / denom;
    }
    return {max_abs, max_rel};
}

void expect_close(const Tensor& got, const Tensor& want,
                  double abs_tol, double rel_tol, const char* tag) {
    auto rep = compare(got, want);
    EXPECT_LE(rep.max_abs, abs_tol)
        << tag << " max-abs " << rep.max_abs << " > tol " << abs_tol;
    EXPECT_LE(rep.max_rel, rel_tol)
        << tag << " max-rel " << rep.max_rel << " > tol " << rel_tol;
}

// ----- 1-D dispatch helpers -----
// The NN-layer contract: the kernel takes 4-D unsqueezed input/weight with
// H==1 and Padding==0 in attrs (the layer pads the length axis manually).
// We replicate that contract here so the kernel sees inputs identical to
// what Conv1d::forward_impl produces.

Tensor pad_1d(const Tensor& x_3d, int64_t pad) {
    // x_3d: [N, C, L] -> [N, C, L + 2*pad] zero-padded on L axis.
    if (pad == 0) return x_3d;
    auto sh = x_3d.shape();
    auto out = zeros({sh[0], sh[1], sh[2] + 2 * pad}, x_3d.dtype(), x_3d.device());
    // Slice-assign by copying through a Float64 buffer is overkill; instead
    // construct via a manual loop, since this is a test helper.
    auto x64 = x_3d.to(DType::Float64).contiguous();
    auto out64 = out.to(DType::Float64).contiguous();
    const double* px = x64.data<double>();
    double* po = out64.data<double>();
    const int64_t N = sh[0], C = sh[1], L = sh[2];
    const int64_t Lp = L + 2 * pad;
    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t l = 0; l < L; ++l) {
                po[(n * C + c) * Lp + (l + pad)] = px[(n * C + c) * L + l];
            }
        }
    }
    return out64.to(x_3d.dtype());
}

Tensor depthwise_conv1d_dispatch(const Tensor& x_3d, const Tensor& w_3d,
                                  const Tensor* bias,
                                  int64_t stride, int64_t padding, int64_t dilation,
                                  int64_t groups) {
    // Replicate the NN-layer 1-D contract: manual pad, unsqueeze.
    auto x_padded = pad_1d(x_3d, padding);
    auto x_4d = x_padded.unsqueeze(2);
    auto w_4d = w_3d.unsqueeze(2);

    std::vector<Tensor> inputs = {x_4d, w_4d};
    if (bias) inputs.push_back(*bias);

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   stride);
    attrs.set(AttrKey::Padding,  static_cast<int64_t>(0));
    attrs.set(AttrKey::Dilation, dilation);
    attrs.set(AttrKey::Groups,   groups);

    auto out_4d = dispatch(OpId::DepthwiseConv1d, inputs, attrs)[0];
    return out_4d.squeeze(2);
}

Tensor conv1d_reference(const Tensor& x_3d, const Tensor& w_3d,
                        const Tensor* bias,
                        int64_t stride, int64_t padding, int64_t dilation,
                        int64_t groups) {
    // Reference: identical NN-layer contract but dispatch to Conv2dForward.
    auto x_padded = pad_1d(x_3d, padding);
    auto x_4d = x_padded.unsqueeze(2);
    auto w_4d = w_3d.unsqueeze(2);

    std::vector<Tensor> inputs = {x_4d, w_4d};
    if (bias) inputs.push_back(*bias);

    OpAttributes attrs;
    attrs.set(AttrKey::Stride,   stride);
    attrs.set(AttrKey::Padding,  static_cast<int64_t>(0));
    attrs.set(AttrKey::Dilation, dilation);
    attrs.set(AttrKey::Groups,   groups);

    auto out_4d = dispatch(OpId::Conv2dForward, inputs, attrs)[0];
    return out_4d.squeeze(2);
}

// ----- 3-D dispatch helpers -----

void set_conv3d_attrs(OpAttributes& a,
                      int64_t sD, int64_t sH, int64_t sW,
                      int64_t pD, int64_t pH, int64_t pW,
                      int64_t dD, int64_t dH, int64_t dW,
                      int64_t groups) {
    a.set(AttrKey::Stride,    sD);  // scalar fallback
    a.set(AttrKey::Padding,   pD);
    a.set(AttrKey::Dilation,  dD);
    a.set(AttrKey::StrideD,   sD);
    a.set(AttrKey::StrideH,   sH);
    a.set(AttrKey::StrideW,   sW);
    a.set(AttrKey::PaddingD,  pD);
    a.set(AttrKey::PaddingH,  pH);
    a.set(AttrKey::PaddingW,  pW);
    a.set(AttrKey::DilationD, dD);
    a.set(AttrKey::DilationH, dH);
    a.set(AttrKey::DilationW, dW);
    a.set(AttrKey::Groups,    groups);
}

Tensor depthwise_conv3d_dispatch(const Tensor& x, const Tensor& w,
                                  const Tensor* bias,
                                  int64_t sD, int64_t sH, int64_t sW,
                                  int64_t pD, int64_t pH, int64_t pW,
                                  int64_t dD, int64_t dH, int64_t dW,
                                  int64_t groups) {
    std::vector<Tensor> inputs = {x, w};
    if (bias) inputs.push_back(*bias);
    OpAttributes attrs;
    set_conv3d_attrs(attrs, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    return dispatch(OpId::DepthwiseConv3d, inputs, attrs)[0];
}

Tensor conv3d_reference(const Tensor& x, const Tensor& w,
                        const Tensor* bias,
                        int64_t sD, int64_t sH, int64_t sW,
                        int64_t pD, int64_t pH, int64_t pW,
                        int64_t dD, int64_t dH, int64_t dW,
                        int64_t groups) {
    std::vector<Tensor> inputs = {x, w};
    if (bias) inputs.push_back(*bias);
    OpAttributes attrs;
    set_conv3d_attrs(attrs, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups);
    return dispatch(OpId::Conv3dForward, inputs, attrs)[0];
}

// ----- Test fixture -----

struct DepthwiseConvCpu : public ::testing::Test {
    Device device = Device::cpu();

    void SetUp() override {
        tenzor::initialize();
    }
};

}  // namespace

// ==========================================================================
// DepthwiseConv1d — Float32 parity (stride==1, padding==0, dilation==1, no bias)
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv1dF32_Default) {
    const int64_t N = 2, C = 8, L = 32, kL = 3;
    auto x = randn({N, C, L}, DType::Float32, device);
    auto w = randn({C, 1, kL}, DType::Float32, device);

    auto got  = depthwise_conv1d_dispatch(x, w, nullptr, 1, 0, 1, C);
    auto want = conv1d_reference(x, w, nullptr, 1, 0, 1, C);

    expect_close(got, want, 5e-5, 5e-5, "Conv1d/F32/default");
}

// stride > 1, padding > 0, with bias.
TEST_F(DepthwiseConvCpu, Conv1dF32_StridePadBias) {
    const int64_t N = 1, C = 4, L = 17, kL = 5;
    auto x    = randn({N, C, L}, DType::Float32, device);
    auto w    = randn({C, 1, kL}, DType::Float32, device);
    auto bias = randn({C}, DType::Float32, device);

    auto got  = depthwise_conv1d_dispatch(x, w, &bias, /*stride*/ 2, /*pad*/ 2, /*dil*/ 1, C);
    auto want = conv1d_reference(x, w, &bias, 2, 2, 1, C);

    expect_close(got, want, 5e-5, 5e-5, "Conv1d/F32/stride+pad+bias");
}

// dilation > 1
TEST_F(DepthwiseConvCpu, Conv1dF32_Dilation) {
    const int64_t N = 1, C = 3, L = 20, kL = 3;
    auto x = randn({N, C, L}, DType::Float32, device);
    auto w = randn({C, 1, kL}, DType::Float32, device);

    auto got  = depthwise_conv1d_dispatch(x, w, nullptr, 1, 2, 2, C);
    auto want = conv1d_reference(x, w, nullptr, 1, 2, 2, C);

    expect_close(got, want, 5e-5, 5e-5, "Conv1d/F32/dilation");
}

// ==========================================================================
// DepthwiseConv1d — Float64 parity
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv1dF64) {
    const int64_t N = 1, C = 5, L = 24, kL = 3;
    auto x = randn({N, C, L}, DType::Float64, device);
    auto w = randn({C, 1, kL}, DType::Float64, device);
    auto bias = randn({C}, DType::Float64, device);

    auto got  = depthwise_conv1d_dispatch(x, w, &bias, 2, 1, 1, C);
    auto want = conv1d_reference(x, w, &bias, 2, 1, 1, C);

    expect_close(got, want, 1e-12, 1e-12, "Conv1d/F64");
}

// ==========================================================================
// DepthwiseConv1d — Float16 parity (widen-narrow path)
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv1dF16) {
    const int64_t N = 1, C = 4, L = 16, kL = 3;
    auto x = randn({N, C, L}, DType::Float16, device);
    auto w = randn({C, 1, kL}, DType::Float16, device);

    auto got  = depthwise_conv1d_dispatch(x, w, nullptr, 1, 1, 1, C);
    auto want = conv1d_reference(x, w, nullptr, 1, 1, 1, C);

    // Both paths route F16 through widen-narrow internally, so the only
    // delta is OpenMP/SIMD reduction order — keep tolerance tight enough
    // to catch a stride/dim bug but loose enough to absorb F16 ulps.
    expect_close(got, want, 5e-3, 5e-3, "Conv1d/F16");
}

// ==========================================================================
// DepthwiseConv1d — BFloat16 parity
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv1dBF16) {
    const int64_t N = 1, C = 4, L = 16, kL = 3;
    auto x = randn({N, C, L}, DType::BFloat16, device);
    auto w = randn({C, 1, kL}, DType::BFloat16, device);

    auto got  = depthwise_conv1d_dispatch(x, w, nullptr, 1, 1, 1, C);
    auto want = conv1d_reference(x, w, nullptr, 1, 1, 1, C);

    expect_close(got, want, 5e-2, 5e-2, "Conv1d/BF16");
}

// ==========================================================================
// DepthwiseConv3d — Float32 parity, default attrs
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv3dF32_Default) {
    const int64_t N = 1, C = 4, D = 6, H = 6, W = 6;
    const int64_t kD = 3, kH = 3, kW = 3;
    auto x = randn({N, C, D, H, W}, DType::Float32, device);
    auto w = randn({C, 1, kD, kH, kW}, DType::Float32, device);

    auto got  = depthwise_conv3d_dispatch(x, w, nullptr, 1,1,1, 0,0,0, 1,1,1, C);
    auto want = conv3d_reference     (x, w, nullptr, 1,1,1, 0,0,0, 1,1,1, C);

    expect_close(got, want, 5e-5, 5e-5, "Conv3d/F32/default");
}

// stride > 1 + padding > 0 + bias
TEST_F(DepthwiseConvCpu, Conv3dF32_StridePadBias) {
    const int64_t N = 1, C = 3, D = 7, H = 7, W = 7;
    const int64_t kD = 3, kH = 3, kW = 3;
    auto x    = randn({N, C, D, H, W}, DType::Float32, device);
    auto w    = randn({C, 1, kD, kH, kW}, DType::Float32, device);
    auto bias = randn({C}, DType::Float32, device);

    auto got  = depthwise_conv3d_dispatch(x, w, &bias, 2,2,2, 1,1,1, 1,1,1, C);
    auto want = conv3d_reference     (x, w, &bias, 2,2,2, 1,1,1, 1,1,1, C);

    expect_close(got, want, 5e-5, 5e-5, "Conv3d/F32/stride+pad+bias");
}

// Dilation > 1 on a single axis (catches per-axis attr wiring).
TEST_F(DepthwiseConvCpu, Conv3dF32_DilationW) {
    const int64_t N = 1, C = 2, D = 5, H = 5, W = 9;
    const int64_t kD = 2, kH = 2, kW = 3;
    auto x = randn({N, C, D, H, W}, DType::Float32, device);
    auto w = randn({C, 1, kD, kH, kW}, DType::Float32, device);

    auto got  = depthwise_conv3d_dispatch(x, w, nullptr, 1,1,1, 0,0,2, 1,1,2, C);
    auto want = conv3d_reference     (x, w, nullptr, 1,1,1, 0,0,2, 1,1,2, C);

    expect_close(got, want, 5e-5, 5e-5, "Conv3d/F32/dilation-W");
}

// ==========================================================================
// DepthwiseConv3d — Float64
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv3dF64) {
    const int64_t N = 1, C = 3, D = 4, H = 4, W = 4;
    const int64_t kD = 3, kH = 3, kW = 3;
    auto x    = randn({N, C, D, H, W}, DType::Float64, device);
    auto w    = randn({C, 1, kD, kH, kW}, DType::Float64, device);
    auto bias = randn({C}, DType::Float64, device);

    auto got  = depthwise_conv3d_dispatch(x, w, &bias, 1,1,1, 1,1,1, 1,1,1, C);
    auto want = conv3d_reference     (x, w, &bias, 1,1,1, 1,1,1, 1,1,1, C);

    expect_close(got, want, 1e-12, 1e-12, "Conv3d/F64");
}

// ==========================================================================
// DepthwiseConv3d — Float16
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv3dF16) {
    const int64_t N = 1, C = 2, D = 4, H = 4, W = 4;
    const int64_t kD = 3, kH = 3, kW = 3;
    auto x = randn({N, C, D, H, W}, DType::Float16, device);
    auto w = randn({C, 1, kD, kH, kW}, DType::Float16, device);

    auto got  = depthwise_conv3d_dispatch(x, w, nullptr, 1,1,1, 1,1,1, 1,1,1, C);
    auto want = conv3d_reference     (x, w, nullptr, 1,1,1, 1,1,1, 1,1,1, C);

    expect_close(got, want, 5e-3, 5e-3, "Conv3d/F16");
}

// ==========================================================================
// DepthwiseConv3d — BFloat16
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv3dBF16) {
    const int64_t N = 1, C = 2, D = 4, H = 4, W = 4;
    const int64_t kD = 3, kH = 3, kW = 3;
    auto x = randn({N, C, D, H, W}, DType::BFloat16, device);
    auto w = randn({C, 1, kD, kH, kW}, DType::BFloat16, device);

    auto got  = depthwise_conv3d_dispatch(x, w, nullptr, 1,1,1, 1,1,1, 1,1,1, C);
    auto want = conv3d_reference     (x, w, nullptr, 1,1,1, 1,1,1, 1,1,1, C);

    expect_close(got, want, 5e-2, 5e-2, "Conv3d/BF16");
}

// ==========================================================================
// Output-shape sanity for the trivial kL=1 / kD=kH=kW=1 case — guards
// against an off-by-one in the output-size computation.
// ==========================================================================
TEST_F(DepthwiseConvCpu, Conv1dF32_K1OutputShape) {
    const int64_t N = 1, C = 3, L = 8;
    auto x = randn({N, C, L}, DType::Float32, device);
    auto w = randn({C, 1, 1}, DType::Float32, device);

    auto got = depthwise_conv1d_dispatch(x, w, nullptr, 1, 0, 1, C);
    ASSERT_EQ(got.shape().size(), 3u);
    EXPECT_EQ(got.shape()[0], N);
    EXPECT_EQ(got.shape()[1], C);
    EXPECT_EQ(got.shape()[2], L);
}

TEST_F(DepthwiseConvCpu, Conv3dF32_K1OutputShape) {
    const int64_t N = 1, C = 2, D = 5, H = 6, W = 7;
    auto x = randn({N, C, D, H, W}, DType::Float32, device);
    auto w = randn({C, 1, 1, 1, 1}, DType::Float32, device);

    auto got = depthwise_conv3d_dispatch(x, w, nullptr, 1,1,1, 0,0,0, 1,1,1, C);
    ASSERT_EQ(got.shape().size(), 5u);
    EXPECT_EQ(got.shape()[0], N);
    EXPECT_EQ(got.shape()[1], C);
    EXPECT_EQ(got.shape()[2], D);
    EXPECT_EQ(got.shape()[3], H);
    EXPECT_EQ(got.shape()[4], W);
}
