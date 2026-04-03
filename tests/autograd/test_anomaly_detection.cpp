/**
 * @file test_anomaly_detection.cpp
 * @brief Tests for enhanced anomaly detection with forward traceback
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class AnomalyDetectionTest : public BackendTest {};

TEST_P(AnomalyDetectionTest, MetadataPopulatedWhenEnabled) {
    AnomalyMode guard(true);

    auto x = Variable(ones({3}, DType::Float32, device), true);
    auto y = x * 2.0f;

    auto meta = y.creation_metadata();
    ASSERT_NE(meta, nullptr);
    EXPECT_FALSE(meta->function_name.empty());
}

TEST_P(AnomalyDetectionTest, MetadataNotPopulatedWhenDisabled) {
    set_anomaly_detection(false);

    auto x = Variable(ones({3}, DType::Float32, device), true);
    auto y = x * 2.0f;

    auto meta = y.creation_metadata();
    EXPECT_EQ(meta, nullptr);
}

TEST_P(AnomalyDetectionTest, MetadataChainsThroughOps) {
    AnomalyMode guard(true);

    auto x = Variable(ones({3}, DType::Float32, device), true);
    auto y = x * 2.0f;
    auto z = y + 1.0f;

    auto meta = z.creation_metadata();
    ASSERT_NE(meta, nullptr);

    if (meta->parent) {
        EXPECT_FALSE(meta->parent->function_name.empty());
    }
}

TEST_P(AnomalyDetectionTest, NoOverheadWhenDisabled) {
    set_anomaly_detection(false);

    auto x = Variable(ones({100}, DType::Float32, device), true);
    auto y = x;
    for (int i = 0; i < 10; ++i) {
        y = y * 1.01f;
    }
    EXPECT_EQ(y.creation_metadata(), nullptr);
}

TEST_P(AnomalyDetectionTest, MetadataContainsInputShapes) {
    AnomalyMode guard(true);

    auto a = Variable(ones({2, 3}, DType::Float32, device), true);
    auto b = Variable(ones({2, 3}, DType::Float32, device), true);
    auto c = a + b;

    auto meta = c.creation_metadata();
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(meta->input_shapes.size(), 2u);
    if (meta->input_shapes.size() >= 2) {
        EXPECT_EQ(meta->input_shapes[0], (std::vector<int64_t>{2, 3}));
        EXPECT_EQ(meta->input_shapes[1], (std::vector<int64_t>{2, 3}));
    }
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    AnomalyDetectionTest,
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi", "rocm")
);
