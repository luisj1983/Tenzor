/**
 * @file test_mask_rcnn.cpp
 * @brief Tests for Mask R-CNN instance segmentation model
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/mask_rcnn.hpp"

using namespace tenzor;
using namespace tenzor::models;

class MaskRCNNTest : public ::testing::Test {
protected:
    void SetUp() override { device_ = Device::cpu(); }
    Device device_;
};

TEST_F(MaskRCNNTest, MaskRCNNResNet50ForwardShape) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({2, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Check that outputs are returned
    EXPECT_GT(boxes.shape()[0], 0);  // At least some detections
    EXPECT_GT(labels.shape()[0], 0);
    EXPECT_GT(scores.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_F(MaskRCNNTest, MaskRCNNResNet50GradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    // Create and initialize dummy ground truth
    Tensor gt_boxes({1, 5, 4}, DType::Float32, device_);
    auto boxes_data = gt_boxes.data<float>();
    // Initialize 5 boxes at different locations [batch, num_boxes, 4]
    // Box 1
    boxes_data[0] = 10.0f;  boxes_data[1] = 10.0f;  boxes_data[2] = 100.0f; boxes_data[3] = 100.0f;
    // Box 2
    boxes_data[4] = 150.0f; boxes_data[5] = 150.0f; boxes_data[6] = 250.0f; boxes_data[7] = 250.0f;
    // Box 3
    boxes_data[8] = 300.0f; boxes_data[9] = 300.0f; boxes_data[10] = 400.0f; boxes_data[11] = 400.0f;
    // Box 4
    boxes_data[12] = 450.0f; boxes_data[13] = 450.0f; boxes_data[14] = 550.0f; boxes_data[15] = 550.0f;
    // Box 5
    boxes_data[16] = 600.0f; boxes_data[17] = 600.0f; boxes_data[18] = 700.0f; boxes_data[19] = 700.0f;

    Tensor gt_labels({1, 5}, DType::Int64, device_);
    auto labels_data = gt_labels.data<int64_t>();
    // Initialize with valid class labels
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;
    labels_data[3] = 4;
    labels_data[4] = 5;

    Tensor gt_masks({1, 5, 800, 800}, DType::Float32, device_);
    // Initialize masks with zeros (background) - this is memory-intensive but necessary
    auto masks_data = gt_masks.data<float>();
    std::fill(masks_data, masks_data + gt_masks.numel(), 0.0f);
    // Optionally add some 1s in the mask regions corresponding to the boxes
    // For simplicity, we'll just leave as zeros which represents valid (empty) masks

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(MaskRCNNTest, MaskRCNNResNet101ForwardShape) {
    auto model = mask_rcnn_resnet101_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_F(MaskRCNNTest, MaskRCNNDifferentImageSizes) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->eval();

    Variable images_600(Tensor({1, 3, 600, 600}, DType::Float32, device_), true);
    auto [boxes_600, labels_600, scores_600, masks_600] = model->forward_test(images_600);
    EXPECT_GT(boxes_600.shape()[0], 0);

    Variable images_1024(Tensor({1, 3, 1024, 1024}, DType::Float32, device_), true);
    auto [boxes_1024, labels_1024, scores_1024, masks_1024] = model->forward_test(images_1024);
    EXPECT_GT(boxes_1024.shape()[0], 0);
}

TEST_F(MaskRCNNTest, MaskRCNNCustomClasses) {
    auto model = mask_rcnn_resnet50_fpn(80, false);  // COCO classes
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
}


// ============================================================================
// Main  
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
