/**
 * @file test_philox_dropout.cpp
 * @brief Tests for the Philox4x32-10 deterministic dropout mask helper.
 *
 * F13/F22-followup verification:
 *   - Same (seed, offset) reproduces the same mask bit-exactly.
 *   - Different seeds produce different masks.
 *   - The effective keep ratio matches `1 - p` within statistical bounds.
 *   - dropout_p == 0 yields an all-keep mask (= ones / (1-0) = ones).
 */

#include <gtest/gtest.h>

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/philox_dropout.hpp"
#include "tenzor/tenzor.hpp"

#include <cmath>
#include <cstdint>
#include <vector>

using namespace tenzor;

class PhiloxDropoutEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const g_env = ::testing::AddGlobalTestEnvironment(new PhiloxDropoutEnv);

TEST(PhiloxDropout, DeterministicSameSeedSameMask) {
    std::vector<int64_t> shape = {2, 4, 8, 8};  // [B, H, Sq, Sk]
    const double p = 0.25;
    const uint64_t seed = 0xDEADBEEF12345678ULL;
    const uint64_t offset = 0;

    Tensor m1 = philox_dropout_mask(shape, p, seed, offset, DType::Float32);
    Tensor m2 = philox_dropout_mask(shape, p, seed, offset, DType::Float32);

    int64_t total = 1;
    for (auto d : shape) total *= d;
    const float* p1 = m1.data<float>();
    const float* p2 = m2.data<float>();
    for (int64_t i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(p1[i], p2[i]) << "Mask differs at index " << i
            << " under identical (seed, offset)";
    }
}

TEST(PhiloxDropout, DifferentSeedsDifferentMasks) {
    std::vector<int64_t> shape = {1, 2, 8, 8};
    const double p = 0.5;

    Tensor m1 = philox_dropout_mask(shape, p, /*seed=*/1, 0);
    Tensor m2 = philox_dropout_mask(shape, p, /*seed=*/2, 0);

    int64_t total = 1;
    for (auto d : shape) total *= d;
    const float* p1 = m1.data<float>();
    const float* p2 = m2.data<float>();
    int64_t diff_count = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (p1[i] != p2[i]) ++diff_count;
    }
    // With p=0.5 on 128 elements, the two masks should differ on roughly
    // half. Require at least 25% to catch any degenerate "same output"
    // bug while staying well above the noise floor.
    EXPECT_GT(diff_count, total / 4)
        << "Two different seeds produced suspiciously similar masks";
}

TEST(PhiloxDropout, KeepRatioMatchesOneMinusP) {
    // With p = 0.3, the mask should keep ≈ 70% of entries (scaled by 1/0.7).
    std::vector<int64_t> shape = {16, 8, 16, 16};  // 32 768 elements
    const double p = 0.3;

    Tensor m = philox_dropout_mask(shape, p, /*seed=*/42, /*offset=*/0);
    int64_t total = 1;
    for (auto d : shape) total *= d;
    const float* mp = m.data<float>();

    int64_t kept = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (mp[i] > 0.0f) ++kept;
    }
    double ratio = static_cast<double>(kept) / static_cast<double>(total);
    // Allow ±2% slop given the sample size.
    EXPECT_NEAR(ratio, 1.0 - p, 0.02);
}

TEST(PhiloxDropout, ZeroPMeansAllKeep) {
    std::vector<int64_t> shape = {3, 5, 7};
    Tensor m = philox_dropout_mask(shape, /*p=*/0.0, /*seed=*/0xFF, /*offset=*/0);
    int64_t total = 1;
    for (auto d : shape) total *= d;
    const float* mp = m.data<float>();
    // p=0 → keep_scale = 1/(1-0) = 1. Every entry should be exactly 1.0.
    for (int64_t i = 0; i < total; ++i) {
        EXPECT_FLOAT_EQ(mp[i], 1.0f);
    }
}

TEST(PhiloxDropout, NewPhiloxStreamReturnsDistinctSeeds) {
    auto s1 = new_philox_stream();
    auto s2 = new_philox_stream();
    EXPECT_EQ(s1.seed.shape().size(), 1u);
    EXPECT_EQ(s1.seed.shape()[0], 1);
    EXPECT_EQ(s1.seed.dtype(), DType::Int64);
    // Two fresh streams should almost always have different seeds (std::mt19937_64).
    EXPECT_NE(s1.seed.data<int64_t>()[0], s2.seed.data<int64_t>()[0]);
    // Offset is 0 by convention.
    EXPECT_EQ(s1.offset.data<int64_t>()[0], 0);
}

TEST(PhiloxDropout, RejectsInvalidP) {
    EXPECT_THROW(philox_dropout_mask({4}, /*p=*/-0.1, 0, 0),
                 std::invalid_argument);
    EXPECT_THROW(philox_dropout_mask({4}, /*p=*/1.0, 0, 0),
                 std::invalid_argument);
    EXPECT_THROW(philox_dropout_mask({4}, /*p=*/2.0, 0, 0),
                 std::invalid_argument);
}

// F13/F22-followup: the device-side Philox kernels (OneAPI SYCL kernel +
// Vulkan compute shader) use the same Philox4x32-10 constants and the same
// (batch_head, query_idx, kv_pos, offset) counter convention as the host
// helper. Bit-exact cross-backend equivalence is verified at the algorithm
// level via shared constants in code (the SYCL kernel inlines exactly the
// same `philox_round_device` body; the Vulkan shader uses
// `umulExtended` which produces the same hi/lo 32-bit halves as the host
// 64-bit multiply). End-to-end device verification lives in the backend
// FlashAttention integration tests where the dropout-keyed forward is
// followed by a backward that replays via the host helper.

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
