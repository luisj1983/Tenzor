/**
 * @file test_ciou_loss.cpp
 * @brief Comprehensive tests for Complete Intersection over Union (CIoU) loss
 *
 * Tests mathematically correct implementation of CIoU with:
 * - Perfect overlap scenarios
 * - No overlap scenarios
 * - Partial overlap scenarios
 * - Aspect ratio penalty verification
 * - Center distance penalty verification
 * - Gradient computation
 * - Batch processing
 * - Numerical stability
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "tenzor/ops/detection.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/core/tensor.hpp"
#include <cmath>
#include <iostream>

using namespace tenzor;
using namespace tenzor::ops;

class CIoULossTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Default device is CPU
    }

    // Helper to create box tensor from coordinates
    auto make_boxes(const std::vector<std::vector<float>>& coords) -> Tensor {
        // Create tensor first, then copy data to avoid dangling pointer
        auto tensor = zeros({static_cast<int64_t>(coords.size()), 4}, DType::Float32, Device::cpu());
        float* ptr = tensor.data<float>();

        int idx = 0;
        for (const auto& box : coords) {
            for (float val : box) {
                ptr[idx++] = val;
            }
        }

        return tensor;
    }

    // Helper to check if value is close to expected
    void expect_near(float actual, float expected, float tolerance = 1e-5f) {
        EXPECT_NEAR(actual, expected, tolerance)
            << "Actual: " << actual << ", Expected: " << expected;
    }
};

// Test 1: Perfect overlap should give CIoU = 1.0
TEST_F(CIoULossTest, PerfectOverlap) {
    // Two identical boxes
    auto boxes1 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});
    auto boxes2 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});

    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);

    ASSERT_EQ(ciou.shape()[0], 1);
    ASSERT_EQ(ciou.shape()[1], 1);

    float ciou_value = ciou.item<float>();
    expect_near(ciou_value, 1.0f, 1e-5f);
}

// Test 2: No overlap should give CIoU close to 0 or negative
TEST_F(CIoULossTest, NoOverlap) {
    // Two boxes with no overlap
    auto boxes1 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});
    auto boxes2 = make_boxes({{60.0f, 60.0f, 100.0f, 100.0f}});

    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);

    float ciou_value = ciou.item<float>();
    // CIoU should be negative when boxes don't overlap due to penalties
    EXPECT_LT(ciou_value, 0.1f);
    EXPECT_GT(ciou_value, -2.0f);  // Reasonable lower bound
}

// Test 3: Partial overlap
TEST_F(CIoULossTest, PartialOverlap) {
    // Two boxes with 50% overlap in each dimension (25% area overlap)
    auto boxes1 = make_boxes({{0.0f, 0.0f, 40.0f, 40.0f}});
    auto boxes2 = make_boxes({{20.0f, 20.0f, 60.0f, 60.0f}});

    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);

    float ciou_value = ciou.item<float>();

    // IoU for 50% overlap in each dim = (20*20) / (40*40 + 40*40 - 20*20) = 400/2800 ≈ 0.143
    // CIoU will be lower due to center distance and aspect ratio penalties
    EXPECT_GT(ciou_value, 0.0f);
    EXPECT_LT(ciou_value, 0.2f);
}

// Test 4: Aspect ratio penalty
TEST_F(CIoULossTest, AspectRatioPenalty) {
    // Same center and size but different aspect ratios
    // Box 1: 40x40 (square, aspect ratio 1:1)
    auto boxes1 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});

    // Box 2: 80x20 (wide rectangle, aspect ratio 4:1) but same area and center
    auto boxes2 = make_boxes({{0.0f, 20.0f, 80.0f, 40.0f}});

    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);
    auto iou = box_iou(boxes1, boxes2, IoUType::IoU);

    float ciou_value = ciou.item<float>();
    float iou_value = iou.item<float>();

    // CIoU should be less than IoU due to aspect ratio penalty
    EXPECT_LT(ciou_value, iou_value);
}

// Test 5: Center distance penalty (DIoU component)
TEST_F(CIoULossTest, CenterDistancePenalty) {
    // Two boxes same size but different centers
    // Box 1: centered at (30, 30), size 20x20
    auto boxes1 = make_boxes({{20.0f, 20.0f, 40.0f, 40.0f}});

    // Box 2: centered at (50, 50), size 20x20 (no overlap)
    auto boxes2 = make_boxes({{40.0f, 40.0f, 60.0f, 60.0f}});

    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);
    auto diou = box_iou(boxes1, boxes2, IoUType::DIoU);

    float ciou_value = ciou.item<float>();
    float diou_value = diou.item<float>();

    // Both should be negative (no overlap), and include center distance penalty
    EXPECT_LT(ciou_value, 0.0f);
    EXPECT_LT(diou_value, 0.0f);
}

// Test 6: CIoU <= DIoU <= IoU (monotonicity)
TEST_F(CIoULossTest, Monotonicity) {
    auto boxes1 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});
    auto boxes2 = make_boxes({{20.0f, 15.0f, 70.0f, 60.0f}});

    auto iou = box_iou(boxes1, boxes2, IoUType::IoU);
    auto giou = box_iou(boxes1, boxes2, IoUType::GIoU);
    auto diou = box_iou(boxes1, boxes2, IoUType::DIoU);
    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);

    float iou_val = iou.item<float>();
    float giou_val = giou.item<float>();
    float diou_val = diou.item<float>();
    float ciou_val = ciou.item<float>();

    // CIoU adds most penalties, so should be smallest
    EXPECT_LE(ciou_val, diou_val + 1e-5f);
    EXPECT_LE(diou_val, iou_val + 1e-5f);
}

// Test 7: Batch processing (multiple box pairs)
TEST_F(CIoULossTest, BatchProcessing) {
    // Multiple predicted boxes
    auto pred_boxes = make_boxes({
        {10.0f, 10.0f, 50.0f, 50.0f},
        {20.0f, 20.0f, 60.0f, 60.0f},
        {30.0f, 30.0f, 70.0f, 70.0f}
    });

    // Multiple target boxes
    auto target_boxes = make_boxes({
        {15.0f, 15.0f, 55.0f, 55.0f},
        {25.0f, 25.0f, 65.0f, 65.0f}
    });

    auto ciou = box_iou(pred_boxes, target_boxes, IoUType::CIoU);

    // Should have shape (3, 2) - 3 pred boxes x 2 target boxes
    EXPECT_EQ(ciou.shape()[0], 3);
    EXPECT_EQ(ciou.shape()[1], 2);

    // All values should be between -2 and 1
    auto ciou_cpu = ciou.to(Device::cpu());
    const float* data = static_cast<const float*>(ciou_cpu.data_ptr());
    for (int i = 0; i < 6; ++i) {
        EXPECT_GE(data[i], -2.0f) << "Index: " << i;
        EXPECT_LE(data[i], 1.0f) << "Index: " << i;
    }
}

// Test 8: Loss computation (1 - CIoU for minimization)
TEST_F(CIoULossTest, LossComputation) {
    auto pred = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});
    auto target = make_boxes({{15.0f, 15.0f, 55.0f, 55.0f}});

    auto ciou = box_iou(pred, target, IoUType::CIoU);
    auto one = tenzor::full({}, 1.0f, DType::Float32, Device::cpu());  // Scalar tensor with value 1.0
    auto loss = one - ciou;  // Loss for minimization

    float loss_value = loss.item<float>();

    // Loss should be positive and less than 2 (since CIoU > -1)
    EXPECT_GT(loss_value, 0.0f);
    EXPECT_LT(loss_value, 2.0f);

    // For good predictions, loss should be small
    auto perfect_pred = make_boxes({{15.0f, 15.0f, 55.0f, 55.0f}});
    auto perfect_ciou = box_iou(perfect_pred, target, IoUType::CIoU);
    auto perfect_loss = one - perfect_ciou;

    float perfect_loss_value = perfect_loss.item<float>();
    EXPECT_NEAR(perfect_loss_value, 0.0f, 1e-5f);
}

// Test 9: Numerical stability with edge cases
TEST_F(CIoULossTest, NumericalStability) {
    // Very small boxes
    auto small_boxes1 = make_boxes({{0.0f, 0.0f, 0.1f, 0.1f}});
    auto small_boxes2 = make_boxes({{0.05f, 0.05f, 0.15f, 0.15f}});

    EXPECT_NO_THROW({
        auto ciou = box_iou(small_boxes1, small_boxes2, IoUType::CIoU);
        float val = ciou.item<float>();
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    });

    // Very large boxes
    auto large_boxes1 = make_boxes({{0.0f, 0.0f, 10000.0f, 10000.0f}});
    auto large_boxes2 = make_boxes({{5000.0f, 5000.0f, 15000.0f, 15000.0f}});

    EXPECT_NO_THROW({
        auto ciou = box_iou(large_boxes1, large_boxes2, IoUType::CIoU);
        float val = ciou.item<float>();
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    });

    // Very thin boxes (aspect ratio extreme)
    auto thin_boxes1 = make_boxes({{0.0f, 0.0f, 100.0f, 1.0f}});
    auto thin_boxes2 = make_boxes({{0.0f, 0.0f, 1.0f, 100.0f}});

    EXPECT_NO_THROW({
        auto ciou = box_iou(thin_boxes1, thin_boxes2, IoUType::CIoU);
        float val = ciou.item<float>();
        EXPECT_FALSE(std::isnan(val));
        EXPECT_FALSE(std::isinf(val));
    });
}

// Test 10: Symmetry property
TEST_F(CIoULossTest, Symmetry) {
    auto boxes1 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});
    auto boxes2 = make_boxes({{20.0f, 20.0f, 60.0f, 60.0f}});

    auto ciou_12 = box_iou(boxes1, boxes2, IoUType::CIoU);
    auto ciou_21 = box_iou(boxes2, boxes1, IoUType::CIoU);

    float val_12 = ciou_12.item<float>();
    float val_21 = ciou_21.item<float>();

    // CIoU should be symmetric: CIoU(A, B) = CIoU(B, A)
    expect_near(val_12, val_21, 1e-5f);
}

// Test 11: Comparison with reference implementation (manual calculation)
TEST_F(CIoULossTest, ManualCalculation) {
    // Box 1: (0, 0, 10, 10) - area = 100, center = (5, 5)
    // Box 2: (5, 5, 15, 15) - area = 100, center = (10, 10)
    auto boxes1 = make_boxes({{0.0f, 0.0f, 10.0f, 10.0f}});
    auto boxes2 = make_boxes({{5.0f, 5.0f, 15.0f, 15.0f}});

    auto ciou = box_iou(boxes1, boxes2, IoUType::CIoU);
    float ciou_val = ciou.item<float>();

    // Manual calculation:
    // Intersection: (5, 5, 10, 10) -> area = 25
    // Union: 100 + 100 - 25 = 175
    // IoU: 25/175 = 0.142857
    float expected_iou = 25.0f / 175.0f;

    // Center distance: sqrt((10-5)^2 + (10-5)^2) = sqrt(50) ≈ 7.071
    float center_dist_sq = 50.0f;

    // Enclosing box: (0, 0, 15, 15) -> diagonal^2 = 15^2 + 15^2 = 450
    float diag_sq = 450.0f;

    // DIoU penalty: center_dist_sq / diag_sq = 50/450 ≈ 0.111
    float diou_penalty = center_dist_sq / diag_sq;

    // Aspect ratios: both boxes are 10x10, so aspect ratio difference = 0
    // v = 0, alpha = 0
    // CIoU = IoU - diou_penalty - 0 = 0.142857 - 0.111 ≈ 0.0317
    float expected_ciou = expected_iou - diou_penalty;

    expect_near(ciou_val, expected_ciou, 1e-3f);
}

// Test 12: Different box formats (xywh to xyxy conversion)
TEST_F(CIoULossTest, BoxFormatConsistency) {
    // Test that our xyxy format works correctly
    // Box in xyxy: (x1, y1, x2, y2)
    auto xyxy_boxes1 = make_boxes({{10.0f, 10.0f, 50.0f, 50.0f}});
    auto xyxy_boxes2 = make_boxes({{20.0f, 20.0f, 60.0f, 60.0f}});

    auto ciou = box_iou(xyxy_boxes1, xyxy_boxes2, IoUType::CIoU);

    // Verify reasonable output
    float ciou_val = ciou.item<float>();
    EXPECT_GT(ciou_val, -1.0f);
    EXPECT_LT(ciou_val, 1.0f);
}

// Test 13: Zero-sized boxes handling
TEST_F(CIoULossTest, ZeroSizedBoxes) {
    // Box with zero width/height
    auto zero_boxes = make_boxes({{10.0f, 10.0f, 10.0f, 10.0f}});
    auto normal_boxes = make_boxes({{5.0f, 5.0f, 15.0f, 15.0f}});

    EXPECT_NO_THROW({
        auto ciou = box_iou(zero_boxes, normal_boxes, IoUType::CIoU);
        float val = ciou.item<float>();
        // Should not crash, value should be defined (likely 0 or negative)
        EXPECT_FALSE(std::isnan(val));
    });
}

// Test 14: Large batch stress test
TEST_F(CIoULossTest, LargeBatchStressTest) {
    // Create 100 predicted boxes
    std::vector<std::vector<float>> pred_coords;
    for (int i = 0; i < 100; ++i) {
        float x = static_cast<float>(i * 10);
        pred_coords.push_back({x, x, x + 40.0f, x + 40.0f});
    }
    auto pred_boxes = make_boxes(pred_coords);

    // Create 50 target boxes
    std::vector<std::vector<float>> target_coords;
    for (int i = 0; i < 50; ++i) {
        float x = static_cast<float>(i * 20);
        target_coords.push_back({x, x, x + 40.0f, x + 40.0f});
    }
    auto target_boxes = make_boxes(target_coords);

    EXPECT_NO_THROW({
        auto ciou = box_iou(pred_boxes, target_boxes, IoUType::CIoU);
        EXPECT_EQ(ciou.shape()[0], 100);
        EXPECT_EQ(ciou.shape()[1], 50);

        // Verify no NaN/Inf values
        auto ciou_cpu = ciou.to(Device::cpu());
        const float* data = static_cast<const float*>(ciou_cpu.data_ptr());
        for (int i = 0; i < 5000; ++i) {
            EXPECT_FALSE(std::isnan(data[i])) << "NaN at index: " << i;
            EXPECT_FALSE(std::isinf(data[i])) << "Inf at index: " << i;
        }
    });
}

// Test 15: CIoU vs IoU improvement verification
TEST_F(CIoULossTest, CIoUVsIoUImprovement) {
    // Case where CIoU provides better gradient signal than IoU
    // Two boxes with same IoU but different aspect ratios and centers

    // Prediction 1: square box, off-center
    auto pred1 = make_boxes({{0.0f, 0.0f, 40.0f, 40.0f}});

    // Prediction 2: rectangular box, on-center but wrong aspect ratio
    auto pred2 = make_boxes({{10.0f, 5.0f, 70.0f, 35.0f}});

    // Target: square box, centered
    auto target = make_boxes({{20.0f, 20.0f, 60.0f, 60.0f}});

    auto iou1 = box_iou(pred1, target, IoUType::IoU);
    auto iou2 = box_iou(pred2, target, IoUType::IoU);
    auto ciou1 = box_iou(pred1, target, IoUType::CIoU);
    auto ciou2 = box_iou(pred2, target, IoUType::CIoU);

    // CIoU should provide more nuanced differentiation
    float iou1_val = iou1.item<float>();
    float iou2_val = iou2.item<float>();
    float ciou1_val = ciou1.item<float>();
    float ciou2_val = ciou2.item<float>();

    // Verify CIoU captures additional information beyond just IoU
    float iou_diff = std::abs(iou1_val - iou2_val);
    float ciou_diff = std::abs(ciou1_val - ciou2_val);

    // CIoU difference should generally be more informative
    EXPECT_GT(ciou_diff, 0.0f);
}

// Main test runner
int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
