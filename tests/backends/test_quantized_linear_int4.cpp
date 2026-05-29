// INT4 codec regression (release audit C2).
//
// Tenzor stores signed INT4 weights in two's-complement form masked to a
// nibble (range [-8, 7], `value & 0xF`), exactly as int4_utils.hpp / tensor.cpp
// / gptq.cpp / awq.cpp pack them.  quantized_linear_int4.cpp's unpack_int4 used
// to apply an incompatible offset-8 (zero-point) decode (`(packed & 0xF) - 8`),
// which silently produced wrong results for every nonzero weight.  This test
// packs known weights and checks the kernel against a reference dot product
// computed directly from the original signed weights.

#include <gtest/gtest.h>

#include <cstdint>
#include <vector>

namespace tenzor {
namespace nn {
namespace quantization {
namespace kernels {

// Free function defined in src/backends/cpu/kernels/quantization/
// quantized_linear_int4.cpp (no public header; forward-declared here).
auto quantized_linear_int4_kernel(
    const int8_t* input,
    const uint8_t* weight_packed,
    const float* bias,
    float* output,
    int64_t batch_size,
    int64_t in_features,
    int64_t out_features,
    float input_scale,
    float weight_scale,
    float output_scale) -> void;

}  // namespace kernels
}  // namespace quantization
}  // namespace nn
}  // namespace tenzor

namespace {

// Pack two signed INT4 values (each in [-8, 7]) into one byte, low nibble
// first — identical to int4_utils.hpp::pack_int4.
uint8_t pack_pair(int8_t lo, int8_t hi) {
    return static_cast<uint8_t>((lo & 0x0F) | ((hi & 0x0F) << 4));
}

// Run the kernel with unit scales and no bias, comparing against a reference
// integer dot product over the original signed weights.
void check_case(int64_t batch_size, int64_t in_features, int64_t out_features,
                const std::vector<int8_t>& input,          // [batch, in]
                const std::vector<int8_t>& weights) {       // [out, in], in [-8,7]
    ASSERT_EQ(in_features % 2, 0) << "INT4 kernel requires even in_features";

    const int64_t packed_features = in_features / 2;
    std::vector<uint8_t> packed(static_cast<size_t>(out_features * packed_features));
    for (int64_t o = 0; o < out_features; ++o) {
        for (int64_t p = 0; p < packed_features; ++p) {
            int8_t lo = weights[o * in_features + 2 * p];
            int8_t hi = weights[o * in_features + 2 * p + 1];
            packed[o * packed_features + p] = pack_pair(lo, hi);
        }
    }

    std::vector<float> output(static_cast<size_t>(batch_size * out_features), 0.0f);
    tenzor::nn::quantization::kernels::quantized_linear_int4_kernel(
        input.data(), packed.data(), /*bias=*/nullptr, output.data(),
        batch_size, in_features, out_features,
        /*input_scale=*/1.0f, /*weight_scale=*/1.0f, /*output_scale=*/1.0f);

    for (int64_t b = 0; b < batch_size; ++b) {
        for (int64_t o = 0; o < out_features; ++o) {
            int32_t ref = 0;
            for (int64_t k = 0; k < in_features; ++k) {
                ref += static_cast<int32_t>(input[b * in_features + k]) *
                       static_cast<int32_t>(weights[o * in_features + k]);
            }
            EXPECT_FLOAT_EQ(output[b * out_features + o], static_cast<float>(ref))
                << "mismatch at batch " << b << " out " << o;
        }
    }
}

// batch_size == 1 exercises the single-row (AVX2) path.
TEST(QuantizedLinearInt4, SignedWeightsBatch1) {
    const int64_t in = 8, out = 3;
    std::vector<int8_t> input = {1, -1, 2, -2, 3, -3, 4, -4};
    std::vector<int8_t> weights = {
        // full signed range incl. -8 (the value the offset-8 bug got most wrong)
        -8, 7, -1, 1, 0, 3, -4, 5,
         7, -8, 2, -2, 6, -6, 1, -1,
         0, 0, -8, -8, 7, 7, -3, 4,
    };
    check_case(1, in, out, input, weights);
}

// batch_size > 1 exercises the pre-unpack path.
TEST(QuantizedLinearInt4, SignedWeightsBatchN) {
    const int64_t in = 8, out = 3, batch = 4;
    std::vector<int8_t> input(static_cast<size_t>(batch * in));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<int8_t>((static_cast<int>(i) % 15) - 7);  // [-7, 7]
    }
    std::vector<int8_t> weights = {
        -8, 7, -1, 1, 0, 3, -4, 5,
         7, -8, 2, -2, 6, -6, 1, -1,
        -2, 2, -8, 4, 7, -5, -3, 6,
    };
    check_case(batch, in, out, input, weights);
}

// Wide in_features crosses the 16-nibble AVX2 unrolled block boundary.
TEST(QuantizedLinearInt4, WideFeaturesCrossSimdBlock) {
    const int64_t in = 40, out = 2, batch = 3;
    std::vector<int8_t> input(static_cast<size_t>(batch * in));
    for (size_t i = 0; i < input.size(); ++i) {
        input[i] = static_cast<int8_t>((static_cast<int>(i * 3) % 13) - 6);
    }
    std::vector<int8_t> weights(static_cast<size_t>(out * in));
    for (size_t i = 0; i < weights.size(); ++i) {
        weights[i] = static_cast<int8_t>((static_cast<int>(i) % 16) - 8);  // [-8, 7]
    }
    check_case(batch, in, out, input, weights);
}

}  // namespace
