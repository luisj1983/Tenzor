/**
 * @file test_detection_components.cpp
 * @brief Unit tests for detection components
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/detection/anchors.hpp"
#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/ops/detection.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn::detection;
using namespace tenzor::ops;

class DetectionComponentsTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }

    void TearDown() override {
        tenzor::finalize();
    }
};

// ============================================================================
// AnchorGenerator Tests
// ============================================================================

TEST_F(DetectionComponentsTest, AnchorGeneratorBasic) {
    // Create anchor generator with 3 sizes and 3 aspect ratios
    AnchorGenerator anchors({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});

    // Generate anchors for 2x2 feature map with stride 16
    auto boxes = anchors.generate(2, 2, 16);

    // Should have 2*2*3*3 = 36 anchors
    EXPECT_EQ(boxes.size(0), 36);
    EXPECT_EQ(boxes.size(1), 4);

    // Check first anchor (center at (8, 8), size 32, ratio 0.5)
    auto first_anchor = boxes.slice(0, 0, 1);
    const float* data = first_anchor.data_ptr<float>();

    // w = 32 * sqrt(0.5) = 22.627
    // h = 32 / sqrt(0.5) = 45.255
    // x1 = 8 - 22.627/2 = -3.314
    // y1 = 8 - 45.255/2 = -14.627
    // x2 = 8 + 22.627/2 = 19.314
    // y2 = 8 + 45.255/2 = 30.627

    float w = 32.0f * std::sqrt(0.5f);
    float h = 32.0f / std::sqrt(0.5f);
    float expected_x1 = 8.0f - w * 0.5f;
    float expected_y1 = 8.0f - h * 0.5f;
    float expected_x2 = 8.0f + w * 0.5f;
    float expected_y2 = 8.0f + h * 0.5f;

    EXPECT_NEAR(data[0], expected_x1, 0.01f);
    EXPECT_NEAR(data[1], expected_y1, 0.01f);
    EXPECT_NEAR(data[2], expected_x2, 0.01f);
    EXPECT_NEAR(data[3], expected_y2, 0.01f);
}

TEST_F(DetectionComponentsTest, AnchorGeneratorNumAnchors) {
    AnchorGenerator anchors({32.0f, 64.0f}, {0.5f, 1.0f, 2.0f});
    EXPECT_EQ(anchors.num_anchors_per_location(), 6);

    AnchorGenerator anchors2({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});
    EXPECT_EQ(anchors2.num_anchors_per_location(), 9);
}

// ============================================================================
// Box IoU Tests
// ============================================================================

TEST_F(DetectionComponentsTest, BoxIoUStandard) {
    // Create two boxes with known IoU
    // Box1: (0, 0, 10, 10), area = 100
    // Box2: (5, 5, 15, 15), area = 100
    // Intersection: (5, 5, 10, 10), area = 25
    // Union: 100 + 100 - 25 = 175
    // IoU: 25 / 175 = 0.1429

    auto boxes1 = tenzor::tensor({{0.0f, 0.0f, 10.0f, 10.0f}});
    auto boxes2 = tenzor::tensor({{5.0f, 5.0f, 15.0f, 15.0f}});

    auto iou = box_iou(boxes1, boxes2);

    EXPECT_EQ(iou.size(0), 1);
    EXPECT_EQ(iou.size(1), 1);

    float expected_iou = 25.0f / 175.0f;
    EXPECT_NEAR(iou.item<float>(), expected_iou, 0.001f);
}

TEST_F(DetectionComponentsTest, BoxIoUMultiple) {
    // Multiple boxes
    auto boxes1 = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},
        {5.0f, 5.0f, 15.0f, 15.0f}
    });

    auto boxes2 = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},
        {20.0f, 20.0f, 30.0f, 30.0f}
    });

    auto iou = box_iou(boxes1, boxes2);

    EXPECT_EQ(iou.size(0), 2);
    EXPECT_EQ(iou.size(1), 2);

    // Box1[0] vs Box2[0]: perfect match, IoU = 1.0
    EXPECT_NEAR(iou[{0, 0}].item<float>(), 1.0f, 0.001f);

    // Box1[0] vs Box2[1]: no overlap, IoU = 0.0
    EXPECT_NEAR(iou[{0, 1}].item<float>(), 0.0f, 0.001f);
}

// ============================================================================
// Box Encoding/Decoding Tests
// ============================================================================

TEST_F(DetectionComponentsTest, BoxEncodingDecoding) {
    // Ground truth boxes
    auto boxes = tenzor::tensor({
        {10.0f, 10.0f, 20.0f, 20.0f},
        {5.0f, 5.0f, 15.0f, 15.0f}
    });

    // Anchor boxes
    auto anchors = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},
        {0.0f, 0.0f, 10.0f, 10.0f}
    });

    // Encode
    auto deltas = encode_boxes(boxes, anchors);

    EXPECT_EQ(deltas.size(0), 2);
    EXPECT_EQ(deltas.size(1), 4);

    // Decode back
    auto decoded = decode_boxes(deltas, anchors);

    // Should match original boxes
    auto boxes_data = boxes.data_ptr<float>();
    auto decoded_data = decoded.data_ptr<float>();

    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(boxes_data[i], decoded_data[i], 0.01f);
    }
}

// ============================================================================
// NMS Tests
// ============================================================================

TEST_F(DetectionComponentsTest, NMSBasic) {
    // Create overlapping boxes with different scores
    auto boxes = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},    // score: 0.9
        {1.0f, 1.0f, 11.0f, 11.0f},    // score: 0.8 (high IoU with box 0)
        {20.0f, 20.0f, 30.0f, 30.0f},  // score: 0.7 (no overlap)
        {2.0f, 2.0f, 12.0f, 12.0f}     // score: 0.6 (high IoU with box 0)
    });

    auto scores = tenzor::tensor({0.9f, 0.8f, 0.7f, 0.6f});

    // Apply NMS with IoU threshold 0.5
    auto keep = nms(boxes, scores, 0.5);

    // Should keep box 0 (highest score) and box 2 (no overlap)
    EXPECT_EQ(keep.size(0), 2);

    auto keep_data = keep.data_ptr<int64_t>();
    EXPECT_EQ(keep_data[0], 0);
    EXPECT_EQ(keep_data[1], 2);
}

TEST_F(DetectionComponentsTest, NMSNoOverlap) {
    // Non-overlapping boxes
    auto boxes = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},
        {20.0f, 20.0f, 30.0f, 30.0f},
        {40.0f, 40.0f, 50.0f, 50.0f}
    });

    auto scores = tenzor::tensor({0.9f, 0.8f, 0.7f});

    auto keep = nms(boxes, scores, 0.5);

    // Should keep all boxes
    EXPECT_EQ(keep.size(0), 3);
}

// ============================================================================
// ROIAlign Tests
// ============================================================================

TEST_F(DetectionComponentsTest, ROIAlignBasic) {
    // Create simple feature map
    auto features = tenzor::randn({1, 2, 8, 8});  // 1 batch, 2 channels, 8x8

    // Create single ROI: batch_idx=0, covering (0, 0, 8, 8) in image space
    // With spatial_scale=1.0, this maps to (0, 0, 8, 8) in feature space
    auto rois = tenzor::tensor({{0.0f, 0.0f, 0.0f, 8.0f, 8.0f}});

    // ROIAlign to 3x3 output
    ROIAlign roi_align(3, 3, 1.0, 2, true);

    auto aligned = roi_align.forward(Variable(features, false), rois);

    // Check output shape
    EXPECT_EQ(aligned.tensor().size(0), 1);   // num_rois
    EXPECT_EQ(aligned.tensor().size(1), 2);   // channels
    EXPECT_EQ(aligned.tensor().size(2), 3);   // output_h
    EXPECT_EQ(aligned.tensor().size(3), 3);   // output_w
}

TEST_F(DetectionComponentsTest, ROIAlignMultipleROIs) {
    auto features = tenzor::randn({2, 4, 16, 16});  // 2 batches, 4 channels

    // Multiple ROIs from different batches
    auto rois = tenzor::tensor({
        {0.0f, 0.0f, 0.0f, 16.0f, 16.0f},     // batch 0
        {1.0f, 8.0f, 8.0f, 24.0f, 24.0f},     // batch 1
        {0.0f, 4.0f, 4.0f, 12.0f, 12.0f}      // batch 0
    });

    ROIAlign roi_align(7, 7, 1.0, 2, true);

    auto aligned = roi_align.forward(Variable(features, false), rois);

    // Check output shape
    EXPECT_EQ(aligned.tensor().size(0), 3);   // num_rois
    EXPECT_EQ(aligned.tensor().size(1), 4);   // channels
    EXPECT_EQ(aligned.tensor().size(2), 7);   // output_h
    EXPECT_EQ(aligned.tensor().size(3), 7);   // output_w
}

TEST_F(DetectionComponentsTest, ROIAlignGradient) {
    // Test gradient flow through ROIAlign
    auto features = tenzor::randn({1, 2, 8, 8});
    auto rois = tenzor::tensor({{0.0f, 0.0f, 0.0f, 8.0f, 8.0f}});

    auto features_var = Variable(features, true);

    ROIAlign roi_align(3, 3, 1.0, 2, true);
    auto aligned = roi_align.forward(features_var, rois);

    // Backward pass
    auto grad_output = tenzor::ones_like(aligned.tensor());
    aligned.backward(grad_output);

    // Check that gradients were computed
    EXPECT_TRUE(features_var.has_grad());
    EXPECT_EQ(features_var.grad()->shape(), features.shape());
}

// ============================================================================
// Batched NMS Tests
// ============================================================================

TEST_F(DetectionComponentsTest, BatchedNMS) {
    // Boxes for multiple classes
    auto boxes = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},
        {1.0f, 1.0f, 11.0f, 11.0f},
        {20.0f, 20.0f, 30.0f, 30.0f}
    });

    // Scores for 2 classes
    auto scores = tenzor::tensor({
        {0.9f, 0.1f},  // Box 0: high score for class 0
        {0.8f, 0.2f},  // Box 1: high score for class 0 (overlaps with box 0)
        {0.1f, 0.9f}   // Box 2: high score for class 1
    });

    auto [kept_boxes, kept_scores, kept_labels] = batched_nms(
        boxes, scores, 0.5, 0.05, 100);

    // Should keep box 0 for class 0, box 2 for class 1
    EXPECT_EQ(kept_boxes.size(0), 2);
}

// ============================================================================
// Utility Functions Tests
// ============================================================================

TEST_F(DetectionComponentsTest, ClipBoxes) {
    auto boxes = tenzor::tensor({
        {-5.0f, -5.0f, 15.0f, 15.0f},
        {5.0f, 5.0f, 105.0f, 105.0f}
    });

    auto clipped = clip_boxes_to_image(boxes, 100, 100);

    auto clipped_data = clipped.data_ptr<float>();

    // Box 0 should be clipped to (0, 0, 15, 15)
    EXPECT_EQ(clipped_data[0], 0.0f);
    EXPECT_EQ(clipped_data[1], 0.0f);
    EXPECT_EQ(clipped_data[2], 15.0f);
    EXPECT_EQ(clipped_data[3], 15.0f);

    // Box 1 should be clipped to (5, 5, 100, 100)
    EXPECT_EQ(clipped_data[4], 5.0f);
    EXPECT_EQ(clipped_data[5], 5.0f);
    EXPECT_EQ(clipped_data[6], 100.0f);
    EXPECT_EQ(clipped_data[7], 100.0f);
}

TEST_F(DetectionComponentsTest, RemoveSmallBoxes) {
    auto boxes = tenzor::tensor({
        {0.0f, 0.0f, 10.0f, 10.0f},   // 10x10, valid
        {0.0f, 0.0f, 2.0f, 10.0f},    // 2x10, too narrow
        {0.0f, 0.0f, 10.0f, 2.0f},    // 10x2, too short
        {0.0f, 0.0f, 20.0f, 20.0f}    // 20x20, valid
    });

    auto scores = tenzor::ones({4});

    auto keep = remove_small_boxes(boxes, scores, 5.0);

    // Should keep boxes 0 and 3
    EXPECT_EQ(keep.size(0), 2);

    auto keep_data = keep.data_ptr<int64_t>();
    EXPECT_EQ(keep_data[0], 0);
    EXPECT_EQ(keep_data[1], 3);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
