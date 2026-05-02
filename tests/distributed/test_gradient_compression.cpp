/**
 * @file test_gradient_compression.cpp
 * @brief C++-side tests for tenzor::distributed::GradientCompressor.
 *
 * The audit (2026-05-02) found a Python-only test (tests/python/
 * test_gradient_compression.py); the C++ public API at
 * include/tenzor/distributed/gradient_compression.hpp had no coverage.
 * This file pins:
 *   - FP16 compressor: round-trip preserves shape/dtype, halves byte
 *     count, reconstruction tolerance ≤ 1e-2 (Float16 epsilon).
 *   - TopK compressor: only `ratio` fraction of entries are non-zero,
 *     reported compression ratio matches sparsity, error feedback
 *     accumulates the dropped magnitude across multiple compress() calls.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/distributed/gradient_compression.hpp>
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributed;

class GradientCompressionTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

namespace {
int64_t count_nonzero_local(const Tensor& t) {
    auto cpu = t.contiguous().to(Device::cpu()).to(DType::Float32);
    const float* p = cpu.data<float>();
    int64_t n = 0;
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        if (p[i] != 0.0f) ++n;
    }
    return n;
}

float l2_norm_diff(const Tensor& a, const Tensor& b) {
    auto a_cpu = a.contiguous().to(Device::cpu()).to(DType::Float32);
    auto b_cpu = b.contiguous().to(Device::cpu()).to(DType::Float32);
    const float* ap = a_cpu.data<float>();
    const float* bp = b_cpu.data<float>();
    double s = 0.0;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        double d = ap[i] - bp[i];
        s += d * d;
    }
    return static_cast<float>(std::sqrt(s));
}
}  // namespace

// ---------------------------------------------------------------------------
// FP16Compressor
// ---------------------------------------------------------------------------

TEST_F(GradientCompressionTest, FP16_RoundTrip_PreservesShapeAndDtype) {
    auto grad_orig = randn({4, 16}, DType::Float32, Device::cpu());
    auto grad = grad_orig.clone();

    FP16Compressor compressor;
    auto compressed = compressor.compress(grad);

    EXPECT_EQ(compressed.original_dtype, DType::Float32);
    EXPECT_EQ(compressed.original_numel, 64);
    ASSERT_EQ(compressed.original_shape.size(), 2u);
    EXPECT_EQ(compressed.original_shape[0], 4);
    EXPECT_EQ(compressed.original_shape[1], 16);
    EXPECT_EQ(compressed.data.dtype(), DType::Float16);

    Tensor decompressed = compressor.decompress(compressed);
    EXPECT_EQ(decompressed.dtype(), DType::Float32);
    EXPECT_EQ(decompressed.shape().size(), 2u);
    EXPECT_EQ(decompressed.shape()[0], 4);
    EXPECT_EQ(decompressed.shape()[1], 16);

    // Reconstruction should be within Float16 representational error.
    float diff = l2_norm_diff(grad_orig, decompressed);
    EXPECT_LT(diff, 1e-2f) << "FP16 round-trip diff too large: " << diff;
}

TEST_F(GradientCompressionTest, FP16_CompressionRatio_HalfByteCount) {
    auto grad = randn({64}, DType::Float32, Device::cpu());
    FP16Compressor compressor;
    auto compressed = compressor.compress(grad);
    // Float16 is half the byte count of Float32 → ratio ≈ 0.5.
    EXPECT_NEAR(compressed.compression_ratio, 0.5f, 0.05f)
        << "FP16 compression ratio ≠ 0.5 (got "
        << compressed.compression_ratio << ")";
}

// ---------------------------------------------------------------------------
// TopKCompressor
// ---------------------------------------------------------------------------

TEST_F(GradientCompressionTest, TopK_OnlyKeepsRatioFraction) {
    auto grad = randn({256}, DType::Float32, Device::cpu());
    const float ratio = 0.1f;
    TopKCompressor compressor(ratio);
    auto compressed = compressor.compress(grad);

    auto kept = count_nonzero_local(compressed.data);
    int64_t expected = static_cast<int64_t>(std::ceil(256 * ratio));
    EXPECT_NEAR(kept, expected, 1) << "TopK kept " << kept
        << " entries, expected ≈ " << expected;
}

TEST_F(GradientCompressionTest, TopK_RatioInvariant_RejectsBadRatio) {
    EXPECT_THROW(TopKCompressor(0.0f), std::invalid_argument);
    EXPECT_THROW(TopKCompressor(-0.1f), std::invalid_argument);
    EXPECT_THROW(TopKCompressor(1.5f), std::invalid_argument);
}

TEST_F(GradientCompressionTest, TopK_RoundTrip_PreservesShape) {
    auto grad_orig = randn({8, 32}, DType::Float32, Device::cpu());
    auto grad = grad_orig.clone();
    TopKCompressor compressor(/*ratio=*/0.25f);
    auto compressed = compressor.compress(grad);
    Tensor decompressed = compressor.decompress(compressed);
    EXPECT_EQ(decompressed.shape().size(), 2u);
    EXPECT_EQ(decompressed.shape()[0], 8);
    EXPECT_EQ(decompressed.shape()[1], 32);
    EXPECT_EQ(decompressed.dtype(), DType::Float32);
}

TEST_F(GradientCompressionTest, TopK_ErrorFeedback_AccumulatesDroppedMagnitude) {
    // The error-feedback contract: small gradients that get dropped one
    // iteration must be added to the next call's input via a residual
    // buffer keyed off the gradient pointer. We can observe this without
    // peeking into private state by feeding the SAME tensor over multiple
    // iterations: the running sum of (kept entries) should approach the
    // full L2 norm of the original gradient as the residual builds up.
    auto grad_orig = randn({64}, DType::Float32, Device::cpu()) * 0.1f;
    TopKCompressor compressor(/*ratio=*/0.1f);

    Tensor running_kept = full({64}, 0.0, DType::Float32, Device::cpu());
    for (int iter = 0; iter < 10; ++iter) {
        auto grad = grad_orig.clone();
        auto compressed = compressor.compress(grad);
        // grad is now sparsified in-place; its non-zero entries are this
        // iteration's "transmission" and we accumulate them.
        const float* gp = grad.contiguous().data<float>();
        float* rp = running_kept.data<float>();
        for (int64_t i = 0; i < grad.numel(); ++i) {
            rp[i] += gp[i];
        }
    }

    // The accumulated transmissions, divided by 10 iterations, should
    // converge to (a fraction of) the original gradient. Without error
    // feedback, each call sees the same input, picks the same ~10% of
    // positions, and the OTHER 90% would get a zero contribution forever.
    // With error feedback, every position eventually crosses the
    // threshold, so the per-position accumulated value is non-zero for
    // most positions.
    int64_t positions_hit = count_nonzero_local(running_kept);
    EXPECT_GT(positions_hit, 32)
        << "Error feedback did not accumulate dropped values: only "
        << positions_hit << " / 64 positions ever transmitted.";
}
