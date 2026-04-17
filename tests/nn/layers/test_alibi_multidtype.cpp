/**
 * @file test_alibi_multidtype.cpp
 * @brief Multi-dtype tests for ALiBi (Attention with Linear Biases) layer
 *
 * Tests ALiBi bias generation with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct construction for power-of-2 and non-power-of-2 head counts
 * - Proper bias tensor shapes (symmetric and asymmetric)
 * - Bias values are non-positive, with zero on diagonal
 * - Monotonic decrease of bias with distance
 * - Forward passthrough behavior
 * - Dtype preservation
 * - Slope scaling with head count
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/alibi.hpp>
#include "../../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// ALiBi Multi-Backend Multi-DType Test Fixture
// ============================================================================

class ALiBiMultiDTypeTest : public MultiBackendDTypeTest {};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_P(ALiBiMultiDTypeTest, ConstructionPowerOf2Heads) {
    // 8 is a power of 2 — slopes should be exact powers of 2
    EXPECT_NO_THROW(ALiBi alibi(8));
}

TEST_P(ALiBiMultiDTypeTest, ConstructionNonPowerOf2Heads) {
    // 6 is not a power of 2 — uses interpolated slopes
    EXPECT_NO_THROW(ALiBi alibi(6));
}

// ============================================================================
// Shape Tests
// ============================================================================

TEST_P(ALiBiMultiDTypeTest, BiasShape) {
    ALiBi alibi(8);
    auto bias = alibi.get_bias(16, 16, device(), dtype());

    // Expected shape: (1, num_heads, seq_q, seq_k) = (1, 8, 16, 16)
    expectShape(bias, {1, 8, 16, 16});
}

TEST_P(ALiBiMultiDTypeTest, BiasShapeAsymmetric) {
    ALiBi alibi(4);
    auto bias = alibi.get_bias(10, 20, device(), dtype());

    // Expected shape: (1, 4, 10, 20)
    expectShape(bias, {1, 4, 10, 20});
}

// ============================================================================
// Value Tests
// ============================================================================

TEST_P(ALiBiMultiDTypeTest, BiasValuesNonPositive) {
    ALiBi alibi(8);
    auto bias = alibi.get_bias(12, 12, device(), dtype());

    // All bias values should be <= 0 (negative slopes * positive distances)
    auto bias_cpu = bias.to(Device::cpu()).to(DType::Float32);
    float max_val = compute_max(bias_cpu);
    EXPECT_LE(max_val, atol())
        << "All ALiBi bias values should be non-positive, but max value is " << max_val;
}

TEST_P(ALiBiMultiDTypeTest, BiasValuesDiagonalZero) {
    ALiBi alibi(4);
    int64_t seq_len = 8;
    auto bias = alibi.get_bias(seq_len, seq_len, device(), dtype());

    // Diagonal elements (i == j, distance = 0) should be exactly 0
    auto bias_cpu = bias.to(Device::cpu()).to(DType::Float32);
    auto* data = bias_cpu.data<float>();

    // bias shape: (1, num_heads, seq_len, seq_len)
    int64_t num_heads = 4;
    for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t i = 0; i < seq_len; ++i) {
            // Index: batch(0) * (num_heads*seq_len*seq_len) + h * (seq_len*seq_len) + i * seq_len + i
            int64_t idx = h * (seq_len * seq_len) + i * seq_len + i;
            EXPECT_NEAR(data[idx], 0.0f, atol())
                << "Diagonal element at head " << h << ", position " << i
                << " should be 0 but is " << data[idx];
        }
    }
}

TEST_P(ALiBiMultiDTypeTest, BiasValuesDecreaseWithDistance) {
    ALiBi alibi(4);
    int64_t seq_len = 8;
    auto bias = alibi.get_bias(seq_len, seq_len, device(), dtype());

    auto bias_cpu = bias.to(Device::cpu()).to(DType::Float32);
    auto* data = bias_cpu.data<float>();

    // For each head, bias[0][h][0][j] should decrease (become more negative)
    // as j increases (distance from position 0 increases)
    int64_t num_heads = 4;
    for (int64_t h = 0; h < num_heads; ++h) {
        for (int64_t j = 1; j < seq_len; ++j) {
            // bias[0][h][0][j-1] and bias[0][h][0][j]
            int64_t idx_prev = h * (seq_len * seq_len) + 0 * seq_len + (j - 1);
            int64_t idx_curr = h * (seq_len * seq_len) + 0 * seq_len + j;
            EXPECT_GE(data[idx_prev], data[idx_curr] - atol())
                << "Bias should decrease with distance at head " << h
                << ", positions j=" << (j - 1) << " vs j=" << j
                << ": " << data[idx_prev] << " vs " << data[idx_curr];
        }
    }
}

// ============================================================================
// Forward Passthrough Test
// ============================================================================

TEST_P(ALiBiMultiDTypeTest, ForwardPassthrough) {
    ALiBi alibi(4);

    // forward_impl should return the input unchanged
    auto input = createInput({2, 4, 8, 8}, true);
    auto output = alibi.forward(input);

    // Output tensor should be identical to input tensor
    auto input_cpu = input.tensor().to(Device::cpu()).to(DType::Float32);
    auto output_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);

    ASSERT_EQ(input_cpu.numel(), output_cpu.numel());
    auto* in_data = input_cpu.data<float>();
    auto* out_data = output_cpu.data<float>();

    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        EXPECT_EQ(in_data[i], out_data[i])
            << "Forward should return input unchanged, mismatch at index " << i;
    }
}

// ============================================================================
// DType Preservation Test
// ============================================================================

TEST_P(ALiBiMultiDTypeTest, DTypePreservation) {
    ALiBi alibi(4);
    auto bias = alibi.get_bias(8, 8, device(), dtype());

    EXPECT_EQ(bias.dtype(), dtype())
        << "get_bias output dtype should match requested dtype";
}

// ============================================================================
// Slope Scaling Test
// ============================================================================

TEST_P(ALiBiMultiDTypeTest, DifferentHeadCountsSlopes) {
    // ALiBi(1) has a single steep slope: m = 2^(-8*1/1) = 2^(-8) = 1/256
    // ALiBi(16) has shallower first slope: m = 2^(-8*1/16) = 2^(-0.5)
    // But the key point is that with fewer heads, the steepest slope is used,
    // making bias values more negative at the same distance.
    //
    // Actually ALiBi(1) slope = 2^(-8) which is tiny, so bias is *less* negative.
    // ALiBi(16) first head slope = 2^(-0.5) which is larger.
    // The last head of ALiBi(16) slope = 2^(-8) matches ALiBi(1).
    //
    // Better comparison: check that ALiBi(1) and ALiBi(16) produce different
    // magnitudes. Specifically, ALiBi(16) should have at least one head
    // with more negative bias than ALiBi(1) at the same distance, because
    // ALiBi(16) head 0 has slope 2^(-0.5) > 2^(-8) = ALiBi(1) slope.
    ALiBi alibi1(1);
    ALiBi alibi16(16);

    int64_t seq_len = 8;
    auto bias1 = alibi1.get_bias(seq_len, seq_len, device(), dtype());
    auto bias16 = alibi16.get_bias(seq_len, seq_len, device(), dtype());

    auto bias1_cpu = bias1.to(Device::cpu()).to(DType::Float32);
    auto bias16_cpu = bias16.to(Device::cpu()).to(DType::Float32);

    auto* data1 = bias1_cpu.data<float>();
    auto* data16 = bias16_cpu.data<float>();

    // Compare at distance 4: bias1[0][0][0][4] vs bias16[0][0][0][4] (head 0)
    // ALiBi(1) slope = 2^(-8) ~ 0.00391, bias = -0.00391 * 4 ~ -0.01563
    // ALiBi(16) head 0 slope = 2^(-0.5) ~ 0.707, bias = -0.707 * 4 ~ -2.828
    // So ALiBi(16) head 0 should be more negative than ALiBi(1) head 0
    int64_t dist = 4;
    float bias1_val = data1[0 * (seq_len * seq_len) + 0 * seq_len + dist];
    float bias16_head0_val = data16[0 * (seq_len * seq_len) + 0 * seq_len + dist];

    EXPECT_LT(bias16_head0_val, bias1_val - atol())
        << "ALiBi(16) head 0 (slope 2^-0.5) should produce more negative bias "
        << "than ALiBi(1) (slope 2^-8) at distance " << dist
        << ": bias16=" << bias16_head0_val << " vs bias1=" << bias1_val;
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ALiBiMultiDTypeTest);
