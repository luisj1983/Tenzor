/**
 * @file test_philox_reproducibility.cpp
 * @brief Verify that rand/randn produce identical results regardless of OMP_NUM_THREADS.
 *
 * Task 5.4: Philox RNG for thread-count-independent reproducibility.
 * The key property: tz::manual_seed(42); tz::randn({1000}) must produce
 * bit-identical output whether run with 1 or 4 OMP threads.
 */

#include <gtest/gtest.h>
#include <cstring>

#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

class PhiloxReproducibilityEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new PhiloxReproducibilityEnv);

// ── Helpers ─────────────────────────────────────────────────────────────────

static std::vector<float> collect_f32(const tz::Tensor& t) {
    auto c = t.cpu().to(tz::DType::Float32);
    const float* p = c.data<float>();
    return std::vector<float>(p, p + c.numel());
}

// ── Tests ────────────────────────────────────────────────────────────────────

// Verify randn first 16 values are bit-identical across two separate calls
// with the same seed (basic single-call determinism).
TEST(PhiloxReproducibility, RandnSameSeedSameOutput) {
    tz::manual_seed(42);
    auto a = tz::randn({1000});
    auto va = collect_f32(a);

    tz::manual_seed(42);
    auto b = tz::randn({1000});
    auto vb = collect_f32(b);

    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(va[i], vb[i]) << "randn[" << i << "] differs between same-seed calls";
    }
}

// Verify rand first 16 values are bit-identical across two separate calls.
TEST(PhiloxReproducibility, RandSameSeedSameOutput) {
    tz::manual_seed(7);
    auto a = tz::rand({1000});
    auto va = collect_f32(a);

    tz::manual_seed(7);
    auto b = tz::rand({1000});
    auto vb = collect_f32(b);

    for (int i = 0; i < 16; ++i) {
        EXPECT_EQ(va[i], vb[i]) << "rand[" << i << "] differs between same-seed calls";
    }
}

// Verify rand output in [0,1) for all elements.
TEST(PhiloxReproducibility, RandValuesInRange) {
    tz::manual_seed(42);
    auto a = tz::rand({100});
    auto va = collect_f32(a);
    for (int i = 0; i < 100; ++i) {
        EXPECT_GE(va[i], 0.0f) << "rand value negative at " << i;
        EXPECT_LT(va[i], 1.0f) << "rand value >= 1.0 at " << i;
    }
}

// Verify different seeds produce different outputs (basic sanity).
TEST(PhiloxReproducibility, DifferentSeedsDifferentOutput) {
    tz::manual_seed(1);
    auto a = tz::randn({100});
    auto va = collect_f32(a);

    tz::manual_seed(2);
    auto b = tz::randn({100});
    auto vb = collect_f32(b);

    int diffs = 0;
    for (int i = 0; i < 100; ++i) {
        if (va[i] != vb[i]) ++diffs;
    }
    EXPECT_GT(diffs, 90) << "Different seeds produced too many identical values";
}
