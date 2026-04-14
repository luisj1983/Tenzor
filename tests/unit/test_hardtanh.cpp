/**
 * @file test_hardtanh.cpp
 * @brief Unit tests for Hardtanh activation (module and functional)
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include <cmath>

namespace tenzor {
    void initialize();
}

using namespace tenzor;
using namespace tenzor::nn;

class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

class HardtanhTest : public ::testing::Test {};

// ============================================================================
// Default range [-1, 1]
// ============================================================================

TEST_F(HardtanhTest, DefaultRangeClampsBelowMin) {
    // Values below -1 should be clamped to -1
    float data[] = {-5.0f, -2.0f, -1.5f, -1.0f};
    auto input = Variable(from_data(data, {4}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -1.0f);
    EXPECT_FLOAT_EQ(out[2], -1.0f);
    EXPECT_FLOAT_EQ(out[3], -1.0f);
}

TEST_F(HardtanhTest, DefaultRangeClampsAboveMax) {
    // Values above 1 should be clamped to 1
    float data[] = {1.0f, 1.5f, 2.0f, 5.0f};
    auto input = Variable(from_data(data, {4}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_F(HardtanhTest, DefaultRangePassesThrough) {
    // Values in [-1, 1] should pass through unchanged
    float data[] = {-0.5f, 0.0f, 0.5f, 0.99f};
    auto input = Variable(from_data(data, {4}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -0.5f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 0.99f);
}

TEST_F(HardtanhTest, DefaultRangeMixed) {
    float data[] = {-3.0f, -0.5f, 0.5f, 3.0f};
    auto input = Variable(from_data(data, {4}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -0.5f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

// ============================================================================
// Custom range
// ============================================================================

TEST_F(HardtanhTest, CustomRange) {
    float data[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    auto input = Variable(from_data(data, {5}), false);

    Hardtanh act(-2.0, 3.0);
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -2.0f);  // clamped to min
    EXPECT_FLOAT_EQ(out[1], -2.0f);  // clamped to min
    EXPECT_FLOAT_EQ(out[2], 0.0f);   // in range
    EXPECT_FLOAT_EQ(out[3], 3.0f);   // clamped to max
    EXPECT_FLOAT_EQ(out[4], 3.0f);   // clamped to max
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_F(HardtanhTest, AllValuesInRange) {
    float data[] = {-0.1f, 0.0f, 0.1f, 0.5f};
    auto input = Variable(from_data(data, {4}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    // Everything should pass through
    EXPECT_FLOAT_EQ(out[0], -0.1f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.1f);
    EXPECT_FLOAT_EQ(out[3], 0.5f);
}

TEST_F(HardtanhTest, AllValuesClamped) {
    float data[] = {-100.0f, -50.0f, 50.0f, 100.0f};
    auto input = Variable(from_data(data, {4}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -1.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_F(HardtanhTest, FunctionalHardtanh) {
    float data[] = {-3.0f, -0.5f, 0.5f, 3.0f};
    auto input = Variable(from_data(data, {4}), false);

    auto result = hardtanh(input, -1.0, 1.0);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -0.5f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_F(HardtanhTest, ExactBoundaryValues) {
    // Test exact boundary values: -1 and 1 should pass through
    float data[] = {-1.0f, 1.0f};
    auto input = Variable(from_data(data, {2}), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto* out = result.tensor().data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
}
