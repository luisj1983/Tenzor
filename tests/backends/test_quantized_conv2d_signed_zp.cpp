// =============================================================================
// tests/backends/test_quantized_conv2d_signed_zp.cpp
//
// Correctness regression for the CPU INT8 quantized_conv2d kernels
// (src/backends/cpu/kernels/quantization/quantized_conv2d.cpp).
//
// Covers three release-prep fixes:
//   A4 (signedness): dot_int8 must NOT use _mm256_maddubs_epi16 /
//       _mm512_dpbusd_epi32 (first operand treated as unsigned). Negative
//       weights/activations must accumulate as signed*signed.
//   A3 (zero-point): asymmetric quantization needs the full expansion
//       sum (q_x - input_zp)(q_w - weight_zp), i.e. the raw dot minus both
//       cross terms (input_zp*sum_w and weight_zp*sum_x) plus input_zp*
//       weight_zp*K -- not just "raw - input_zp*weight_zp*K".
//   A3 (padding): with a nonzero input_zp, im2col padding must use the
//       quantized representation of real-zero (= input_zp), otherwise padded
//       taps contribute spurious (0 - input_zp) terms.
//
// The ground-truth reference dequantizes by definition: padded input taps are
// real-zero (contribute nothing), valid taps contribute
// (q_x - input_zp)(q_w - weight_zp). It is independent of the kernel's internal
// decomposition, so it catches all three issues.
// =============================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <random>

#include "tenzor/tenzor.hpp"

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {
auto quantized_conv2d_kernel(
    const int8_t* input, const int8_t* weight, const float* bias, float* output,
    int64_t batch, int64_t in_channels, int64_t out_channels,
    int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    float input_scale, float weight_scale,
    int32_t input_zp, int32_t weight_zp,
    int64_t dil_h, int64_t dil_w, int64_t groups) -> void;
auto quantized_conv2d_per_channel_kernel(
    const int8_t* input, const int8_t* weight, const float* bias, float* output,
    int64_t batch, int64_t in_channels, int64_t out_channels,
    int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
    int64_t kernel_h, int64_t kernel_w, int64_t stride_h, int64_t stride_w,
    int64_t pad_h, int64_t pad_w,
    float input_scale, const float* weight_scales,
    int32_t input_zp, const int32_t* weight_zps,
    int64_t dil_h, int64_t dil_w, int64_t groups) -> void;
} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor

namespace {

class QuantizedConv2dEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const env =
    ::testing::AddGlobalTestEnvironment(new QuantizedConv2dEnv);

// Shapes chosen so col_width = in_channels*K*K = 72 (>= 64) to exercise the
// AVX-512 VNNI 64-wide loop and the AVX2 32-wide loop, with a scalar remainder.
constexpr int64_t kBatch = 2;
constexpr int64_t kInC = 8;
constexpr int64_t kOutC = 8;
constexpr int64_t kHin = 5, kWin = 5;
constexpr int64_t kK = 3, kStride = 1, kPad = 1, kDil = 1;
constexpr int64_t kHout = 5, kWout = 5;  // (5 + 2 - 2 - 1)/1 + 1 = 5

// Ground-truth: weight laid out [out_c][in_c][kh][kw], input [b][in_c][h][w].
// combined_scale = input_scale * weight_scale (conv has no output_scale).
void conv_reference(const std::vector<int8_t>& input,
                    const std::vector<int8_t>& weight,
                    const std::vector<float>& bias, bool use_bias,
                    float input_scale, const std::vector<float>& weight_scale_vec,
                    bool per_channel,
                    int32_t input_zp, const std::vector<int32_t>& weight_zp_vec,
                    std::vector<float>& out) {
    for (int64_t b = 0; b < kBatch; ++b) {
        for (int64_t oc = 0; oc < kOutC; ++oc) {
            const int32_t wz = per_channel ? weight_zp_vec[oc] : weight_zp_vec[0];
            const float wscale = per_channel ? weight_scale_vec[oc] : weight_scale_vec[0];
            const float combined = input_scale * wscale;
            for (int64_t oh = 0; oh < kHout; ++oh) {
                for (int64_t ow = 0; ow < kWout; ++ow) {
                    int32_t acc = 0;
                    for (int64_t ic = 0; ic < kInC; ++ic) {
                        for (int64_t kh = 0; kh < kK; ++kh) {
                            for (int64_t kw = 0; kw < kK; ++kw) {
                                int64_t ih = oh * kStride + kh * kDil - kPad;
                                int64_t iw = ow * kStride + kw * kDil - kPad;
                                if (ih < 0 || ih >= kHin || iw < 0 || iw >= kWin)
                                    continue;  // padded tap = real zero
                                int32_t qx = input[((b * kInC + ic) * kHin + ih) * kWin + iw];
                                int32_t qw = weight[((oc * kInC + ic) * kK + kh) * kK + kw];
                                acc += (qx - input_zp) * (qw - wz);
                            }
                        }
                    }
                    float v = static_cast<float>(acc) * combined;
                    if (use_bias) v += bias[oc];
                    out[((b * kOutC + oc) * kHout + oh) * kWout + ow] = v;
                }
            }
        }
    }
}

} // namespace

// Per-tensor asymmetric conv: negative weights (A4) + nonzero zero-points and
// padding (A3). Must match the dequantized ground truth bit-for-bit.
TEST(QuantizedConv2dSignedZP, PerTensorAsymmetricMatchesGroundTruth) {
    std::vector<int8_t> input(kBatch * kInC * kHin * kWin);
    std::vector<int8_t> weight(kOutC * kInC * kK * kK);
    std::vector<float> bias(kOutC);

    std::mt19937 rng(0xBEEFCAFE);
    std::uniform_int_distribution<int> idist(-128, 127);
    std::uniform_int_distribution<int> wdist(-127, 127);
    std::uniform_real_distribution<float> bdist(-0.5f, 0.5f);
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : bias) v = bdist(rng);

    const float input_scale = 0.0125f, weight_scale = 0.0078125f;
    const int32_t input_zp = 7, weight_zp = -3;

    std::vector<float> got(kBatch * kOutC * kHout * kWout, 0.0f);
    std::vector<float> ref(kBatch * kOutC * kHout * kWout, 0.0f);

    tenzor::nn::quantization::kernels::quantized_conv2d_kernel(
        input.data(), weight.data(), bias.data(), got.data(),
        kBatch, kInC, kOutC, kHin, kWin, kHout, kWout,
        kK, kK, kStride, kStride, kPad, kPad, input_scale, weight_scale,
        input_zp, weight_zp, kDil, kDil, /*groups=*/1);

    conv_reference(input, weight, bias, /*use_bias=*/true, input_scale,
                   {weight_scale}, /*per_channel=*/false, input_zp, {weight_zp}, ref);

    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_FLOAT_EQ(got[i], ref[i]) << "mismatch at i=" << i;
}

// Symmetric sanity (zp=0): the simplest case, still negative weights (A4).
TEST(QuantizedConv2dSignedZP, SymmetricNegativeWeightsMatchesGroundTruth) {
    std::vector<int8_t> input(kBatch * kInC * kHin * kWin);
    std::vector<int8_t> weight(kOutC * kInC * kK * kK);
    std::vector<float> bias(kOutC, 0.0f);

    std::mt19937 rng(0x0DDBALL);
    std::uniform_int_distribution<int> idist(-128, 127);
    std::uniform_int_distribution<int> wdist(-127, -1);  // all-negative weights
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));

    const float input_scale = 0.01f, weight_scale = 0.01f;

    std::vector<float> got(kBatch * kOutC * kHout * kWout, 0.0f);
    std::vector<float> ref(kBatch * kOutC * kHout * kWout, 0.0f);

    tenzor::nn::quantization::kernels::quantized_conv2d_kernel(
        input.data(), weight.data(), nullptr, got.data(),
        kBatch, kInC, kOutC, kHin, kWin, kHout, kWout,
        kK, kK, kStride, kStride, kPad, kPad, input_scale, weight_scale,
        0, 0, kDil, kDil, 1);

    conv_reference(input, weight, bias, /*use_bias=*/false, input_scale,
                   {weight_scale}, /*per_channel=*/false, 0, {0}, ref);

    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_FLOAT_EQ(got[i], ref[i]) << "mismatch at i=" << i;
}

// Per-channel asymmetric conv.
TEST(QuantizedConv2dSignedZP, PerChannelAsymmetricMatchesGroundTruth) {
    std::vector<int8_t> input(kBatch * kInC * kHin * kWin);
    std::vector<int8_t> weight(kOutC * kInC * kK * kK);
    std::vector<float> bias(kOutC);
    std::vector<float> weight_scales(kOutC);
    std::vector<int32_t> weight_zps(kOutC);

    std::mt19937 rng(0xC0DEC0DE);
    std::uniform_int_distribution<int> idist(-128, 127);
    std::uniform_int_distribution<int> wdist(-127, 127);
    std::uniform_int_distribution<int> wzpdist(-8, 8);
    std::uniform_real_distribution<float> sdist(0.005f, 0.05f);
    std::uniform_real_distribution<float> bdist(-0.5f, 0.5f);
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : bias) v = bdist(rng);
    for (auto& v : weight_scales) v = sdist(rng);
    for (auto& v : weight_zps) v = wzpdist(rng);

    const float input_scale = 0.0125f;
    const int32_t input_zp = 5;

    std::vector<float> got(kBatch * kOutC * kHout * kWout, 0.0f);
    std::vector<float> ref(kBatch * kOutC * kHout * kWout, 0.0f);

    tenzor::nn::quantization::kernels::quantized_conv2d_per_channel_kernel(
        input.data(), weight.data(), bias.data(), got.data(),
        kBatch, kInC, kOutC, kHin, kWin, kHout, kWout,
        kK, kK, kStride, kStride, kPad, kPad, input_scale, weight_scales.data(),
        input_zp, weight_zps.data(), kDil, kDil, /*groups=*/1);

    conv_reference(input, weight, bias, /*use_bias=*/true, input_scale,
                   weight_scales, /*per_channel=*/true, input_zp, weight_zps, ref);

    for (size_t i = 0; i < got.size(); ++i)
        ASSERT_FLOAT_EQ(got[i], ref[i]) << "mismatch at i=" << i;
}
