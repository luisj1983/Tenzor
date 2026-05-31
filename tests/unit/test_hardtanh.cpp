/**
 * @file test_hardtanh.cpp
 * @brief Unit tests for Hardtanh activation (module and functional)
 */

#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "../backend_test_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

class HardtanhTest : public ::tenzor::testing::BackendTest {};

// ============================================================================
// Default range [-1, 1]
// ============================================================================

TEST_P(HardtanhTest, DefaultRangeClampsBelowMin) {
    // Values below -1 should be clamped to -1
    float data[] = {-5.0f, -2.0f, -1.5f, -1.0f};
    auto input = Variable(from_data(data, {4}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -1.0f);
    EXPECT_FLOAT_EQ(out[2], -1.0f);
    EXPECT_FLOAT_EQ(out[3], -1.0f);
}

TEST_P(HardtanhTest, DefaultRangeClampsAboveMax) {
    // Values above 1 should be clamped to 1
    float data[] = {1.0f, 1.5f, 2.0f, 5.0f};
    auto input = Variable(from_data(data, {4}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_P(HardtanhTest, DefaultRangePassesThrough) {
    // Values in [-1, 1] should pass through unchanged
    float data[] = {-0.5f, 0.0f, 0.5f, 0.99f};
    auto input = Variable(from_data(data, {4}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -0.5f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 0.99f);
}

TEST_P(HardtanhTest, DefaultRangeMixed) {
    float data[] = {-3.0f, -0.5f, 0.5f, 3.0f};
    auto input = Variable(from_data(data, {4}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -0.5f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

// ============================================================================
// Custom range
// ============================================================================

TEST_P(HardtanhTest, CustomRange) {
    float data[] = {-10.0f, -5.0f, 0.0f, 5.0f, 10.0f};
    auto input = Variable(from_data(data, {5}, device), false);

    Hardtanh act(-2.0, 3.0);
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -2.0f);  // clamped to min
    EXPECT_FLOAT_EQ(out[1], -2.0f);  // clamped to min
    EXPECT_FLOAT_EQ(out[2], 0.0f);   // in range
    EXPECT_FLOAT_EQ(out[3], 3.0f);   // clamped to max
    EXPECT_FLOAT_EQ(out[4], 3.0f);   // clamped to max
}

// ============================================================================
// Edge cases
// ============================================================================

TEST_P(HardtanhTest, AllValuesInRange) {
    float data[] = {-0.1f, 0.0f, 0.1f, 0.5f};
    auto input = Variable(from_data(data, {4}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    // Everything should pass through
    EXPECT_FLOAT_EQ(out[0], -0.1f);
    EXPECT_FLOAT_EQ(out[1], 0.0f);
    EXPECT_FLOAT_EQ(out[2], 0.1f);
    EXPECT_FLOAT_EQ(out[3], 0.5f);
}

TEST_P(HardtanhTest, AllValuesClamped) {
    float data[] = {-100.0f, -50.0f, 50.0f, 100.0f};
    auto input = Variable(from_data(data, {4}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -1.0f);
    EXPECT_FLOAT_EQ(out[2], 1.0f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_P(HardtanhTest, FunctionalHardtanh) {
    float data[] = {-3.0f, -0.5f, 0.5f, 3.0f};
    auto input = Variable(from_data(data, {4}, device), false);

    auto result = hardtanh(input, -1.0, 1.0);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], -0.5f);
    EXPECT_FLOAT_EQ(out[2], 0.5f);
    EXPECT_FLOAT_EQ(out[3], 1.0f);
}

TEST_P(HardtanhTest, ExactBoundaryValues) {
    // Test exact boundary values: -1 and 1 should pass through
    float data[] = {-1.0f, 1.0f};
    auto input = Variable(from_data(data, {2}, device), false);

    Hardtanh act;
    auto result = act.forward_impl(input);
    auto out_cpu = result.tensor().cpu();
    auto* out = out_cpu.data<float>();

    EXPECT_FLOAT_EQ(out[0], -1.0f);
    EXPECT_FLOAT_EQ(out[1], 1.0f);
}

INSTANTIATE_BACKEND_TESTS(HardtanhTest);
