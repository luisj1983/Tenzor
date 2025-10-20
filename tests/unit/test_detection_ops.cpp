/**
 * @file test_detection_ops.cpp
 * @brief Extended tests for detection operations: ROI Align, NMS, etc.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/roi_ops.hpp>
#include <tenzor/nn/detection/anchors.hpp>
#include <tenzor/ops/detection.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::ops;

class DetectionOpsTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

// ============================================================================
// ROI Align Tests (Basic Shape Tests)
// ============================================================================

TEST_F(DetectionOpsTest, ROIAlignBasicForwardShape) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    // Feature map: (N, C, H, W)
    Variable features(Tensor({1, 512, 28, 28}, DType::Float32, device_), true);

    // ROIs: (num_rois, 5) where each row is [batch_idx, x1, y1, x2, y2]
    Tensor rois({10, 5}, DType::Float32, device_);

    Variable output = roi_align.forward(features, rois);

    // Output should be (num_rois, C, pool_h, pool_w)
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 512, 7, 7}));
}

TEST_F(DetectionOpsTest, ROIAlignBasicGradientFlow) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(Tensor({1, 256, 28, 28}, DType::Float32, device_), true);
    Tensor rois({5, 5}, DType::Float32, device_);

    Variable output = roi_align.forward(features, rois);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(features.grad().has_value());
}

TEST_F(DetectionOpsTest, ROIAlignDifferentPoolSizes) {
    ROIAlign roi_align_7(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_14(14, 14, 1.0/16.0, 2);

    Variable features(Tensor({1, 512, 56, 56}, DType::Float32, device_), true);
    Tensor rois({3, 5}, DType::Float32, device_);

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

TEST_F(DetectionOpsTest, ROIAlignForwardShape) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(Tensor({1, 512, 28, 28}, DType::Float32, device_), true);
    Tensor rois({10, 5}, DType::Float32, device_);

    Variable output = roi_align.forward(features, rois);

    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 512, 7, 7}));
}

TEST_F(DetectionOpsTest, ROIAlignGradientFlow) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(Tensor({1, 256, 28, 28}, DType::Float32, device_), true);
    Tensor rois({5, 5}, DType::Float32, device_);

    Variable output = roi_align.forward(features, rois);
    Variable loss = tenzor::sum(output);
    loss.backward();

    EXPECT_TRUE(features.grad().has_value());
}

TEST_F(DetectionOpsTest, ROIAlignDifferentSamplingRatios) {
    ROIAlign roi_align_2(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_4(7, 7, 1.0/16.0, 4);

    Variable features(Tensor({1, 512, 56, 56}, DType::Float32, device_), true);
    Tensor rois({3, 5}, DType::Float32, device_);

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

TEST_F(DetectionOpsTest, NMSBasicFiltering) {
    // Boxes: (N, 4) where each row is [x1, y1, x2, y2]
    Tensor boxes({20, 4}, DType::Float32, device_);

    // Scores: (N,)
    Tensor scores({20}, DType::Float32, device_);

    auto keep_indices = nms(boxes, scores, 0.5);  // IOU threshold = 0.5

    // NMS should return a tensor with valid indices
    EXPECT_GT(keep_indices.shape()[0], 0);
    EXPECT_LE(keep_indices.shape()[0], 20);
}

TEST_F(DetectionOpsTest, NMSDifferentThresholds) {
    Tensor boxes({50, 4}, DType::Float32, device_);
    Tensor scores({50}, DType::Float32, device_);

    auto keep_low = nms(boxes, scores, 0.3);   // Stricter
    auto keep_high = nms(boxes, scores, 0.7);  // More permissive

    // Lower threshold should keep fewer or equal boxes
    EXPECT_LE(keep_low.shape()[0], keep_high.shape()[0]);
}

TEST_F(DetectionOpsTest, NMSOutputShape) {
    // Create boxes and scores
    Tensor boxes({5, 4}, DType::Float32, device_);
    Tensor scores({5}, DType::Float32, device_);

    auto keep_indices = nms(boxes, scores, 0.5);

    // Result should be 1D tensor of indices
    EXPECT_EQ(keep_indices.shape().size(), 1);
    EXPECT_LE(keep_indices.shape()[0], 5);
}

// ============================================================================
// Box Operations Tests
// ============================================================================

TEST_F(DetectionOpsTest, BoxIOUComputation) {
    // Box format: [x1, y1, x2, y2]
    Tensor boxes1({5, 4}, DType::Float32, device_);
    Tensor boxes2({5, 4}, DType::Float32, device_);

    auto ious = box_iou(boxes1, boxes2);

    // IOU matrix should be (5, 5)
    auto shape = ious.shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{5, 5}));
}

TEST_F(DetectionOpsTest, BoxEncodingDecoding) {
    Tensor boxes({10, 4}, DType::Float32, device_);
    Tensor anchors({10, 4}, DType::Float32, device_);

    // Encode boxes relative to anchors
    auto encoded = encode_boxes(boxes, anchors);

    // Decode back to original boxes
    auto decoded = decode_boxes(encoded, anchors);

    // Decoded boxes should match original boxes shape
    auto shape = decoded.shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 4}));
}

TEST_F(DetectionOpsTest, BoxEncodingShape) {
    Tensor boxes({20, 4}, DType::Float32, device_);
    Tensor anchors({20, 4}, DType::Float32, device_);

    auto encoded = encode_boxes(boxes, anchors);

    // Encoded should have same shape as input
    EXPECT_EQ(encoded.shape()[0], 20);
    EXPECT_EQ(encoded.shape()[1], 4);
}

// ============================================================================
// Anchor Generator Tests
// ============================================================================

TEST_F(DetectionOpsTest, AnchorGeneratorSizes) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Feature map size: 56x56
    auto anchors = anchor_gen.generate(56, 56, 16, device_);  // stride=16

    // Number of anchors = H * W * (num_sizes * num_aspect_ratios)
    // 56 * 56 * (5 * 3) = 47040
    EXPECT_EQ(anchors.shape()[0], 47040);
    EXPECT_EQ(anchors.shape()[1], 4);
}

TEST_F(DetectionOpsTest, AnchorGeneratorDifferentScales) {
    std::vector<float> sizes_small = {32.0f, 64.0f};
    std::vector<float> sizes_large = {128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {1.0f};

    AnchorGenerator anchor_gen_small(sizes_small, aspect_ratios);
    AnchorGenerator anchor_gen_large(sizes_large, aspect_ratios);

    auto anchors_small = anchor_gen_small.generate(28, 28, 16, device_);
    auto anchors_large = anchor_gen_large.generate(28, 28, 16, device_);

    // 28 * 28 * 2 = 1568 for small
    // 28 * 28 * 3 = 2352 for large
    EXPECT_EQ(anchors_small.shape()[0], 1568);
    EXPECT_EQ(anchors_large.shape()[0], 2352);
}

TEST_F(DetectionOpsTest, AnchorGeneratorNumAnchorsPerLocation) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Should have 3 sizes * 3 aspect ratios = 9 anchors per location
    EXPECT_EQ(anchor_gen.num_anchors_per_location(), 9);
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
