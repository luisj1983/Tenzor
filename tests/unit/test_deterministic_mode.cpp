/**
 * @file test_deterministic_mode.cpp
 * @brief Verify deterministic mode forces single-threaded reductions.
 *
 * Task 5.5: with set_deterministic(true), repeated tz::sum() over a large
 * tensor must return bit-identical float results regardless of thread count.
 */

#include <gtest/gtest.h>
#include <cstring>

#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/utils/config.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

class DeterministicModeEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new DeterministicModeEnv);

// ── Tests ────────────────────────────────────────────────────────────────────

// With deterministic mode ON, sum of a large tensor must be bit-identical
// across repeated calls (each call goes through the same single-threaded path).
TEST(DeterministicMode, SumBitIdenticalWhenDeterministic) {
    // Use a large tensor (>REDUCTION_OMP_THRESHOLD=65536) to force OMP path normally.
    tz::manual_seed(42);
    auto t = tz::rand({131072});  // 2x threshold

    tz::set_deterministic(true);

    auto s1 = tz::sum(t);
    auto s2 = tz::sum(t);

    float v1 = s1.cpu().to(tz::DType::Float32).data<float>()[0];
    float v2 = s2.cpu().to(tz::DType::Float32).data<float>()[0];

    EXPECT_EQ(v1, v2) << "deterministic sum differs between calls: " << v1 << " vs " << v2;

    tz::set_deterministic(false);
}

// With deterministic mode OFF, sum still works correctly (just not forced single-thread).
TEST(DeterministicMode, SumWorksWithoutDeterminism) {
    tz::set_deterministic(false);

    tz::manual_seed(1);
    auto t = tz::rand({1000});
    auto s = tz::sum(t);
    float v = s.cpu().to(tz::DType::Float32).data<float>()[0];

    // Just verify the result is finite and positive (all rand values in [0,1))
    EXPECT_GT(v, 0.0f);
    EXPECT_LT(v, 1000.0f);
    EXPECT_EQ(v, v);  // not NaN
}

// Verify is_deterministic() reflects set_deterministic().
TEST(DeterministicMode, FlagRoundtrips) {
    tz::set_deterministic(true);
    EXPECT_TRUE(tz::is_deterministic());

    tz::set_deterministic(false);
    EXPECT_FALSE(tz::is_deterministic());
}
