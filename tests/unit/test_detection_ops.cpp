/**
 * @file test_detection_ops.cpp
 * @brief Extended tests for detection operations: ROI Align, NMS, etc.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/roi_ops.hpp>
#include <tenzor/nn/detection/anchors.hpp>
#include <tenzor/ops/detection.hpp>
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::ops;

class DetectionOpsTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// ROI Align Tests (Basic Shape Tests)
// ============================================================================

TEST_P(DetectionOpsTest, ROIAlignBasicForwardShape) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    // Feature map: (N, C, H, W)
    Variable features(zeros({1, 512, 28, 28}, DType::Float32, device), true);

    // ROIs: (num_rois, 5) where each row is [batch_idx, x1, y1, x2, y2]
    Tensor rois = zeros({10, 5}, DType::Float32, device);

    Variable output = roi_align.forward(features, rois);

    // Output should be (num_rois, C, pool_h, pool_w)
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 512, 7, 7}));
}

TEST_P(DetectionOpsTest, ROIAlignBasicGradientFlow) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(zeros({1, 256, 28, 28}, DType::Float32, device), true);
    Tensor rois = zeros({5, 5}, DType::Float32, device);

    Variable output = roi_align.forward(features, rois);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(features);
}

TEST_P(DetectionOpsTest, ROIAlignDifferentPoolSizes) {
    ROIAlign roi_align_7(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_14(14, 14, 1.0/16.0, 2);

    Variable features(zeros({1, 512, 56, 56}, DType::Float32, device), true);
    Tensor rois = zeros({3, 5}, DType::Float32, device);

    Variable output_7 = roi_align_7.forward(features, rois);
    auto shape_7 = output_7.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_7.begin(), shape_7.end()),
              (std::vector<int64_t>{3, 512, 7, 7}));

    Variable output_14 = roi_align_14.forward(features, rois);
    auto shape_14 = output_14.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_14.begin(), shape_14.end()),
              (std::vector<int64_t>{3, 512, 14, 14}));
}

// ============================================================================
// ROI Align Tests (Advanced)
// ============================================================================

TEST_P(DetectionOpsTest, ROIAlignForwardShape) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(zeros({1, 512, 28, 28}, DType::Float32, device), true);
    Tensor rois = zeros({10, 5}, DType::Float32, device);

    Variable output = roi_align.forward(features, rois);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 512, 7, 7}));
}

TEST_P(DetectionOpsTest, ROIAlignGradientFlow) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(zeros({1, 256, 28, 28}, DType::Float32, device), true);
    Tensor rois = zeros({5, 5}, DType::Float32, device);

    Variable output = roi_align.forward(features, rois);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(features);
}

TEST_P(DetectionOpsTest, ROIAlignDifferentSamplingRatios) {
    ROIAlign roi_align_2(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_4(7, 7, 1.0/16.0, 4);

    Variable features(zeros({1, 512, 56, 56}, DType::Float32, device), true);
    Tensor rois = zeros({3, 5}, DType::Float32, device);

    Variable output_2 = roi_align_2.forward(features, rois);
    Variable output_4 = roi_align_4.forward(features, rois);

    auto shape_2 = output_2.tensor().shape();
    auto shape_4 = output_4.tensor().shape();

    // Both should have same output shape (difference is in interpolation quality)
    EXPECT_EQ(shape_2[0], shape_4[0]);
    EXPECT_EQ(shape_2[1], shape_4[1]);
}

// ============================================================================
// Non-Maximum Suppression (NMS) Tests
// ============================================================================

TEST_P(DetectionOpsTest, NMSBasicFiltering) {
    // Boxes: (N, 4) where each row is [x1, y1, x2, y2]
    Tensor boxes = zeros({20, 4}, DType::Float32, device);

    // Scores: (N,)
    Tensor scores = zeros({20}, DType::Float32, device);

    auto keep_indices = nms(boxes, scores, 0.5);  // IOU threshold = 0.5

    // NMS should return a tensor with valid indices
    EXPECT_GT(keep_indices.shape()[0], 0);
    EXPECT_LE(keep_indices.shape()[0], 20);
}

TEST_P(DetectionOpsTest, NMSDifferentThresholds) {
    Tensor boxes = zeros({50, 4}, DType::Float32, device);
    Tensor scores = zeros({50}, DType::Float32, device);

    auto keep_low = nms(boxes, scores, 0.3);   // Stricter
    auto keep_high = nms(boxes, scores, 0.7);  // More permissive

    // Lower threshold should keep fewer or equal boxes
    EXPECT_LE(keep_low.shape()[0], keep_high.shape()[0]);
}

TEST_P(DetectionOpsTest, NMSOutputShape) {
    // Create boxes and scores
    Tensor boxes = zeros({5, 4}, DType::Float32, device);
    Tensor scores = zeros({5}, DType::Float32, device);

    auto keep_indices = nms(boxes, scores, 0.5);

    // Result should be 1D tensor of indices
    EXPECT_EQ(keep_indices.shape().size(), 1);
    EXPECT_LE(keep_indices.shape()[0], 5);
}

// ============================================================================
// Box Operations Tests
// ============================================================================

TEST_P(DetectionOpsTest, BoxIOUComputation) {
    // Box format: [x1, y1, x2, y2]
    Tensor boxes1 = zeros({5, 4}, DType::Float32, device);
    Tensor boxes2 = zeros({5, 4}, DType::Float32, device);

    auto ious = box_iou(boxes1, boxes2);

    // IOU matrix should be (5, 5)
    auto shape = ious.shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{5, 5}));
}

TEST_P(DetectionOpsTest, BoxEncodingDecoding) {
    Tensor boxes = zeros({10, 4}, DType::Float32, device);
    Tensor anchors = zeros({10, 4}, DType::Float32, device);

    // Encode boxes relative to anchors
    auto encoded = encode_boxes(boxes, anchors);

    // Decode back to original boxes
    auto decoded = decode_boxes(encoded, anchors);

    // Decoded boxes should match original boxes shape
    auto shape = decoded.shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 4}));
}

TEST_P(DetectionOpsTest, BoxEncodingShape) {
    Tensor boxes = zeros({20, 4}, DType::Float32, device);
    Tensor anchors = zeros({20, 4}, DType::Float32, device);

    auto encoded = encode_boxes(boxes, anchors);

    // Encoded should have same shape as input
    EXPECT_EQ(encoded.shape()[0], 20);
    EXPECT_EQ(encoded.shape()[1], 4);
}

// ============================================================================
// Anchor Generator Tests
// ============================================================================

TEST_P(DetectionOpsTest, AnchorGeneratorSizes) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Feature map size: 56x56
    auto anchors = anchor_gen.generate(56, 56, 16, device);  // stride=16

    // Number of anchors = H * W * (num_sizes * num_aspect_ratios)
    // 56 * 56 * (5 * 3) = 47040
    EXPECT_EQ(anchors.shape()[0], 47040);
    EXPECT_EQ(anchors.shape()[1], 4);
}

TEST_P(DetectionOpsTest, AnchorGeneratorDifferentScales) {
    std::vector<float> sizes_small = {32.0f, 64.0f};
    std::vector<float> sizes_large = {128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {1.0f};

    AnchorGenerator anchor_gen_small(sizes_small, aspect_ratios);
    AnchorGenerator anchor_gen_large(sizes_large, aspect_ratios);

    auto anchors_small = anchor_gen_small.generate(28, 28, 16, device);
    auto anchors_large = anchor_gen_large.generate(28, 28, 16, device);

    // 28 * 28 * 2 = 1568 for small
    // 28 * 28 * 3 = 2352 for large
    EXPECT_EQ(anchors_small.shape()[0], 1568);
    EXPECT_EQ(anchors_large.shape()[0], 2352);
}

TEST_P(DetectionOpsTest, AnchorGeneratorNumAnchorsPerLocation) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Should have 3 sizes * 3 aspect ratios = 9 anchors per location
    EXPECT_EQ(anchor_gen.num_anchors_per_location(), 9);
}

INSTANTIATE_BACKEND_TESTS(DetectionOpsTest);
