/**
 * @file test_anomaly_detection_multidtype.cpp
 * @brief Multi-backend, multi-dtype tests for enhanced anomaly detection
 *
 * Converted from test_anomaly_detection.cpp to exercise all backend + dtype
 * combinations via MultiBackendDTypeTest.
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/autograd/anomaly_mode.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class AnomalyDetectionMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(AnomalyDetectionMultiDTypeTest, MetadataPopulatedWhenEnabled) {
    AnomalyMode guard(true);

    auto x = Variable(ones({3}, dtype(), device()), true);
    auto y = x * 2.0f;

    auto meta = y.creation_metadata();
    ASSERT_NE(meta, nullptr);
    EXPECT_FALSE(meta->function_name.empty());
}

TEST_P(AnomalyDetectionMultiDTypeTest, MetadataNotPopulatedWhenDisabled) {
    set_anomaly_detection(false);

    auto x = Variable(ones({3}, dtype(), device()), true);
    auto y = x * 2.0f;

    auto meta = y.creation_metadata();
    EXPECT_EQ(meta, nullptr);
}

TEST_P(AnomalyDetectionMultiDTypeTest, MetadataChainsThroughOps) {
    AnomalyMode guard(true);

    auto x = Variable(ones({3}, dtype(), device()), true);
    auto y = x * 2.0f;
    auto z = y + 1.0f;

    auto meta = z.creation_metadata();
    ASSERT_NE(meta, nullptr);

    if (meta->parent) {
        EXPECT_FALSE(meta->parent->function_name.empty());
    }
}

TEST_P(AnomalyDetectionMultiDTypeTest, NoOverheadWhenDisabled) {
    set_anomaly_detection(false);

    auto x = Variable(ones({100}, dtype(), device()), true);
    auto y = x;
    for (int i = 0; i < 10; ++i) {
        y = y * 1.01f;
    }
    EXPECT_EQ(y.creation_metadata(), nullptr);
}

TEST_P(AnomalyDetectionMultiDTypeTest, MetadataContainsInputShapes) {
    AnomalyMode guard(true);

    auto a = Variable(ones({2, 3}, dtype(), device()), true);
    auto b = Variable(ones({2, 3}, dtype(), device()), true);
    auto c = a + b;

    auto meta = c.creation_metadata();
    ASSERT_NE(meta, nullptr);
    EXPECT_EQ(meta->input_shapes.size(), 2u);
    if (meta->input_shapes.size() >= 2) {
        EXPECT_EQ(meta->input_shapes[0], (std::vector<int64_t>{2, 3}));
        EXPECT_EQ(meta->input_shapes[1], (std::vector<int64_t>{2, 3}));
    }
}

// ============================================================================
// Instantiate for all available backends and dtypes
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(AnomalyDetectionMultiDTypeTest);
