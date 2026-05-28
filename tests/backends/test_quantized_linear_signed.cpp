// =============================================================================
// tests/backends/test_quantized_linear_signed.cpp
//
// Signedness regression test for the CPU INT8 quantized_linear kernel
// (src/backends/cpu/kernels/quantization/quantized_linear.cpp).
//
// History: the previous AVX2 path used _mm256_maddubs_epi16, which treats its
// first operand as *unsigned* INT8. Any negative input byte (e.g. -1) aliased
// to a large positive (255), silently producing wrong accumulations for
// signed activations — which is the norm in quantization (post-relu can still
// be negative, symmetric per-tensor quant centres around zero).
//
// The fixed path widens signed INT8 -> INT16 via _mm256_cvtepi8_epi16 and uses
// _mm256_madd_epi16 (signed*signed -> int32). This test forces SIMD execution
// (in_features = 128, batch=64) and asserts the kernel matches a hand-written
// int32 scalar reference for several activation regimes:
//   - mixed-sign uniformly distributed inputs (regression for the bug)
//   - all-zero inputs (sanity)
//   - positive-only inputs (regression that the fix didn't break the case the
//     old broken code happened to handle correctly).
//
// We call the kernel function directly (free function in
// tenzor::nn::quantization::kernels), forward-declared below.
// =============================================================================

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>
#include <random>
#include <cmath>

#include "tenzor/tenzor.hpp"

// ---------------------------------------------------------------------------
// Forward declaration of the kernel under test.
// Defined in src/backends/cpu/kernels/quantization/quantized_linear.cpp,
// linked into the tenzor_backend_cpu library (transitively pulled in by
// tenzor_core).
// ---------------------------------------------------------------------------
namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {
auto quantized_linear_kernel(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    float weight_scale,
    float output_scale,
    int32_t input_zp,
    int32_t weight_zp
) -> void;
} // namespace kernels
} // namespace quantization
} // namespace nn
} // namespace tenzor

// ---------------------------------------------------------------------------
// Test environment: initialise Tenzor (mirrors tests/utils/test_widen_narrow.cpp).
// ---------------------------------------------------------------------------
namespace {

class QuantizedLinearSignedEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const env =
    ::testing::AddGlobalTestEnvironment(new QuantizedLinearSignedEnv);

// Hand-written int32 reference matching the kernel's math exactly.
// output[b][o] = (sum_k int32(input[b][k]) * int32(weight[o][k])
//                 - input_zp * weight_zp * K) * (input_scale * weight_scale / output_scale)
//                + bias[o]
void scalar_reference(
    const int8_t* input,
    const int8_t* weight,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    float weight_scale,
    float output_scale,
    int32_t input_zp,
    int32_t weight_zp)
{
    const float combined_scale = input_scale * weight_scale / output_scale;
    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            int32_t acc = 0;
            const int8_t* in_row = input + b * in_features;
            const int8_t* wt_row = weight + o * in_features;
            for (int64_t k = 0; k < in_features; ++k) {
                acc += static_cast<int32_t>(in_row[k]) *
                       static_cast<int32_t>(wt_row[k]);
            }
            acc -= input_zp * weight_zp * static_cast<int32_t>(in_features);
            float v = static_cast<float>(acc) * combined_scale;
            if (bias != nullptr) v += bias[o];
            output[b * out_features + o] = v;
        }
    }
}

// Common shape — in_features=128 is 4x32, guaranteeing the AVX2 SIMD loop body
// executes (32 bytes per _mm256_loadu_si256), and the AVX-512 path's 32-INT8
// stride also runs cleanly. batch_size=64, out_features=32 keep the test fast.
constexpr int64_t kBatch = 64;
constexpr int64_t kIn = 128;
constexpr int64_t kOut = 32;

void run_kernel_and_compare(
    const std::vector<int8_t>& input,
    const std::vector<int8_t>& weight,
    const std::vector<float>& bias_vec,
    bool use_bias,
    float input_scale, float weight_scale, float output_scale,
    int32_t input_zp, int32_t weight_zp)
{
    std::vector<float> simd_out(kBatch * kOut, 0.0f);
    std::vector<float> ref_out(kBatch * kOut, 0.0f);

    const float* bias_ptr = use_bias ? bias_vec.data() : nullptr;

    tenzor::nn::quantization::kernels::quantized_linear_kernel(
        input.data(), weight.data(), bias_ptr, simd_out.data(),
        kBatch, kIn, kOut,
        input_scale, weight_scale, output_scale,
        input_zp, weight_zp);

    scalar_reference(
        input.data(), weight.data(), bias_ptr, ref_out.data(),
        kBatch, kIn, kOut,
        input_scale, weight_scale, output_scale,
        input_zp, weight_zp);

    // INT8 dot-products are exact integer math; the only float step is a
    // single multiply by combined_scale (+ optional bias add), identical in
    // both paths. We assert bit-exact equality.
    for (int64_t i = 0; i < kBatch * kOut; ++i) {
        ASSERT_EQ(simd_out[i], ref_out[i])
            << "mismatch at i=" << i
            << " simd=" << simd_out[i]
            << " ref=" << ref_out[i];
    }
}

} // namespace

// ---------------------------------------------------------------------------
// Regression: mixed-sign INT8 activations.
//
// This is the exact case the old _mm256_maddubs_epi16 path silently
// miscalculated. Inputs drawn from full int8 range [-128, 127].
// ---------------------------------------------------------------------------
TEST(QuantizedLinearSigned, MixedSignActivationsMatchScalarReference) {
    std::vector<int8_t> input(kBatch * kIn);
    std::vector<int8_t> weight(kOut * kIn);
    std::vector<float> bias(kOut);

    std::mt19937 rng(0xC0FFEE);
    std::uniform_int_distribution<int> idist(-128, 127);
    std::uniform_int_distribution<int> wdist(-127, 127);  // exclude -128 for symmetry
    std::uniform_real_distribution<float> bdist(-0.5f, 0.5f);
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : bias) v = bdist(rng);

    // Symmetric per-tensor quant params.
    run_kernel_and_compare(input, weight, bias, /*use_bias=*/true,
                           /*input_scale=*/0.015625f,
                           /*weight_scale=*/0.0078125f,
                           /*output_scale=*/0.03125f,
                           /*input_zp=*/0,
                           /*weight_zp=*/0);
}

// ---------------------------------------------------------------------------
// Stricter regression: force a high fraction of negative input bytes. The
// old _mm256_maddubs_epi16 path would re-interpret each negative as a large
// positive, so this case maximally exposes the bug.
// ---------------------------------------------------------------------------
TEST(QuantizedLinearSigned, MostlyNegativeActivationsMatchScalarReference) {
    std::vector<int8_t> input(kBatch * kIn);
    std::vector<int8_t> weight(kOut * kIn);
    std::vector<float> bias(kOut, 0.0f);

    std::mt19937 rng(0xDEADBEEF);
    std::uniform_int_distribution<int> idist(-128, -1);  // strictly negative
    std::uniform_int_distribution<int> wdist(-127, 127);
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));

    run_kernel_and_compare(input, weight, bias, /*use_bias=*/false,
                           0.0078125f, 0.0078125f, 0.0078125f, 0, 0);
}

// ---------------------------------------------------------------------------
// Sanity: zero activations -> zero output (plus bias).
// ---------------------------------------------------------------------------
TEST(QuantizedLinearSigned, AllZeroActivationsYieldZeroOrBias) {
    std::vector<int8_t> input(kBatch * kIn, 0);
    std::vector<int8_t> weight(kOut * kIn);
    std::vector<float> bias(kOut);

    std::mt19937 rng(0xBABEFACE);
    std::uniform_int_distribution<int> wdist(-127, 127);
    std::uniform_real_distribution<float> bdist(-1.0f, 1.0f);
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));
    for (auto& v : bias) v = bdist(rng);

    run_kernel_and_compare(input, weight, bias, /*use_bias=*/true,
                           0.01f, 0.01f, 0.01f, 0, 0);

    // Independently assert the explicit zero+bias case: with all-zero inputs
    // and zero_points, output[b][o] should equal bias[o] exactly.
    std::vector<float> out(kBatch * kOut, 99.0f);
    tenzor::nn::quantization::kernels::quantized_linear_kernel(
        input.data(), weight.data(), bias.data(), out.data(),
        kBatch, kIn, kOut, 0.01f, 0.01f, 0.01f, 0, 0);
    for (int64_t b = 0; b < kBatch; ++b) {
        for (int64_t o = 0; o < kOut; ++o) {
            EXPECT_FLOAT_EQ(out[b * kOut + o], bias[o]);
        }
    }
}

// ---------------------------------------------------------------------------
// Regression: positive-only activations.
//
// This is the case the *broken* maddubs path happened to handle correctly
// (because uint8 reinterpret was a no-op on positive bytes). The fix must
// not regress here.
// ---------------------------------------------------------------------------
TEST(QuantizedLinearSigned, PositiveOnlyActivationsMatchScalarReference) {
    std::vector<int8_t> input(kBatch * kIn);
    std::vector<int8_t> weight(kOut * kIn);
    std::vector<float> bias(kOut, 0.0f);

    std::mt19937 rng(0xFEEDFACE);
    std::uniform_int_distribution<int> idist(0, 127);
    std::uniform_int_distribution<int> wdist(-127, 127);
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));

    run_kernel_and_compare(input, weight, bias, /*use_bias=*/false,
                           0.0078125f, 0.0078125f, 0.0078125f, 0, 0);
}

// ---------------------------------------------------------------------------
// Per-tensor zero-point correction still works after the signed-safe rewrite.
// ---------------------------------------------------------------------------
TEST(QuantizedLinearSigned, NonZeroZeroPointsMatchScalarReference) {
    std::vector<int8_t> input(kBatch * kIn);
    std::vector<int8_t> weight(kOut * kIn);
    std::vector<float> bias(kOut, 0.0f);

    std::mt19937 rng(0xA5A5A5A5);
    std::uniform_int_distribution<int> idist(-128, 127);
    std::uniform_int_distribution<int> wdist(-127, 127);
    for (auto& v : input) v = static_cast<int8_t>(idist(rng));
    for (auto& v : weight) v = static_cast<int8_t>(wdist(rng));

    run_kernel_and_compare(input, weight, bias, /*use_bias=*/false,
                           0.02f, 0.02f, 0.02f,
                           /*input_zp=*/5,
                           /*weight_zp=*/-3);
}
