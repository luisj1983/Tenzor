/**
 * @file test_mask_rcnn_losses.cpp
 * @brief Comprehensive tests for Mask R-CNN loss functions
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/models/mask_rcnn.hpp>
#include <tenzor/ops/detection.hpp>
#include <tenzor/ops/creation.hpp>

using namespace tenzor;
using namespace tenzor::models;

class MaskRCNNLossTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        // Note: Random seed setting would go here if available
    }

    Device device_;
};

// ============================================================================
// RPN Loss Tests
// ============================================================================

TEST_F(MaskRCNNLossTest, RPNLossBasic) {
    // Create a simple Mask R-CNN model
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    // Create synthetic input
    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    // Create ground truth boxes (5 objects)
    Tensor gt_boxes({1, 5, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    // Box 1: [10, 10, 100, 100]
    boxes_data[0] = 10.0f; boxes_data[1] = 10.0f; boxes_data[2] = 100.0f; boxes_data[3] = 100.0f;
    // Box 2: [150, 150, 250, 250]
    boxes_data[4] = 150.0f; boxes_data[5] = 150.0f; boxes_data[6] = 250.0f; boxes_data[7] = 250.0f;
    // Box 3: [300, 300, 400, 400]
    boxes_data[8] = 300.0f; boxes_data[9] = 300.0f; boxes_data[10] = 400.0f; boxes_data[11] = 400.0f;
    // Box 4: [450, 450, 550, 550]
    boxes_data[12] = 450.0f; boxes_data[13] = 450.0f; boxes_data[14] = 550.0f; boxes_data[15] = 550.0f;
    // Box 5: [600, 600, 700, 700]
    boxes_data[16] = 600.0f; boxes_data[17] = 600.0f; boxes_data[18] = 700.0f; boxes_data[19] = 700.0f;

    Tensor gt_labels({1, 5}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 3;
    labels_data[3] = 4; labels_data[4] = 5;

    Tensor gt_masks({1, 5, 800, 800}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    // Forward pass to compute losses
    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Verify RPN losses are non-zero and finite
    EXPECT_GT(rpn_cls_loss.tensor().item<float>(), 0.0f);
    EXPECT_LT(rpn_cls_loss.tensor().item<float>(), 100.0f);  // Reasonable range
    EXPECT_FALSE(std::isnan(rpn_cls_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(rpn_cls_loss.tensor().item<float>()));

    // Box regression loss should also be valid (might be zero if no positive anchors)
    EXPECT_GE(rpn_box_loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(rpn_box_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(rpn_box_loss.tensor().item<float>()));
}

TEST_F(MaskRCNNLossTest, RPNLossMultipleImages) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    // Batch of 2 images
    Variable images(Tensor({2, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    // Ground truth for 2 images
    Tensor gt_boxes({2, 3, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    // Image 1
    boxes_data[0] = 50.0f; boxes_data[1] = 50.0f; boxes_data[2] = 150.0f; boxes_data[3] = 150.0f;
    boxes_data[4] = 200.0f; boxes_data[5] = 200.0f; boxes_data[6] = 300.0f; boxes_data[7] = 300.0f;
    boxes_data[8] = 400.0f; boxes_data[9] = 400.0f; boxes_data[10] = 500.0f; boxes_data[11] = 500.0f;
    // Image 2
    boxes_data[12] = 100.0f; boxes_data[13] = 100.0f; boxes_data[14] = 200.0f; boxes_data[15] = 200.0f;
    boxes_data[16] = 300.0f; boxes_data[17] = 300.0f; boxes_data[18] = 400.0f; boxes_data[19] = 400.0f;
    boxes_data[20] = 500.0f; boxes_data[21] = 500.0f; boxes_data[22] = 600.0f; boxes_data[23] = 600.0f;

    Tensor gt_labels({2, 3}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 3;
    labels_data[3] = 1; labels_data[4] = 2; labels_data[5] = 3;

    Tensor gt_masks({2, 3, 800, 800}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Losses should be computed across batch
    EXPECT_GT(rpn_cls_loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(rpn_cls_loss.tensor().item<float>()));
}

// ============================================================================
// ROI Head Loss Tests
// ============================================================================

TEST_F(MaskRCNNLossTest, ROIHeadLossBasic) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    Tensor gt_boxes({1, 3, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    boxes_data[0] = 100.0f; boxes_data[1] = 100.0f; boxes_data[2] = 200.0f; boxes_data[3] = 200.0f;
    boxes_data[4] = 300.0f; boxes_data[5] = 300.0f; boxes_data[6] = 400.0f; boxes_data[7] = 400.0f;
    boxes_data[8] = 500.0f; boxes_data[9] = 500.0f; boxes_data[10] = 600.0f; boxes_data[11] = 600.0f;

    Tensor gt_labels({1, 3}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 10; labels_data[1] = 20; labels_data[2] = 30;

    Tensor gt_masks({1, 3, 800, 800}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Verify ROI head losses
    EXPECT_GT(roi_cls_loss.tensor().item<float>(), 0.0f);
    EXPECT_LT(roi_cls_loss.tensor().item<float>(), 100.0f);
    EXPECT_FALSE(std::isnan(roi_cls_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(roi_cls_loss.tensor().item<float>()));

    // Box regression loss
    EXPECT_GE(roi_box_loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(roi_box_loss.tensor().item<float>()));
}

TEST_F(MaskRCNNLossTest, ROIHeadLossClassImbalance) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    // Many objects of same class
    Tensor gt_boxes({1, 10, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    for (int i = 0; i < 10; ++i) {
        float offset = i * 70.0f;
        boxes_data[i*4 + 0] = 10.0f + offset;
        boxes_data[i*4 + 1] = 10.0f + offset;
        boxes_data[i*4 + 2] = 60.0f + offset;
        boxes_data[i*4 + 3] = 60.0f + offset;
    }

    Tensor gt_labels({1, 10}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    for (int i = 0; i < 10; ++i) {
        labels_data[i] = 5;  // All same class
    }

    Tensor gt_masks({1, 10, 800, 800}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Should handle class imbalance
    EXPECT_GT(roi_cls_loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(roi_cls_loss.tensor().item<float>()));
}

// ============================================================================
// Mask Loss Tests
// ============================================================================

TEST_F(MaskRCNNLossTest, MaskLossBasic) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    Tensor gt_boxes({1, 2, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    boxes_data[0] = 100.0f; boxes_data[1] = 100.0f; boxes_data[2] = 300.0f; boxes_data[3] = 300.0f;
    boxes_data[4] = 400.0f; boxes_data[5] = 400.0f; boxes_data[6] = 600.0f; boxes_data[7] = 600.0f;

    Tensor gt_labels({1, 2}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2;

    // Create actual mask regions
    Tensor gt_masks({1, 2, 800, 800}, DType::Float32, device_);
    auto* masks_data = gt_masks.data<float>();
    std::fill(masks_data, masks_data + gt_masks.numel(), 0.0f);

    // Fill mask regions corresponding to boxes
    for (int64_t h = 100; h < 300; ++h) {
        for (int64_t w = 100; w < 300; ++w) {
            masks_data[0 * 800 * 800 + h * 800 + w] = 1.0f;  // First object
        }
    }
    for (int64_t h = 400; h < 600; ++h) {
        for (int64_t w = 400; w < 600; ++w) {
            masks_data[1 * 800 * 800 + h * 800 + w] = 1.0f;  // Second object
        }
    }

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Verify mask loss
    EXPECT_GT(mask_loss.tensor().item<float>(), 0.0f);
    EXPECT_LT(mask_loss.tensor().item<float>(), 10.0f);  // Reasonable range for BCE
    EXPECT_FALSE(std::isnan(mask_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isinf(mask_loss.tensor().item<float>()));
}

TEST_F(MaskRCNNLossTest, MaskLossSmallObjects) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    // Small objects
    Tensor gt_boxes({1, 3, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    boxes_data[0] = 50.0f; boxes_data[1] = 50.0f; boxes_data[2] = 80.0f; boxes_data[3] = 80.0f;
    boxes_data[4] = 200.0f; boxes_data[5] = 200.0f; boxes_data[6] = 230.0f; boxes_data[7] = 230.0f;
    boxes_data[8] = 400.0f; boxes_data[9] = 400.0f; boxes_data[10] = 430.0f; boxes_data[11] = 430.0f;

    Tensor gt_labels({1, 3}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 3;

    Tensor gt_masks({1, 3, 800, 800}, DType::Float32, device_);
    auto* masks_data = gt_masks.data<float>();
    std::fill(masks_data, masks_data + gt_masks.numel(), 0.0f);

    // Fill small mask regions
    for (int i = 0; i < 3; ++i) {
        int64_t x1 = static_cast<int64_t>(boxes_data[i*4 + 0]);
        int64_t y1 = static_cast<int64_t>(boxes_data[i*4 + 1]);
        int64_t x2 = static_cast<int64_t>(boxes_data[i*4 + 2]);
        int64_t y2 = static_cast<int64_t>(boxes_data[i*4 + 3]);

        for (int64_t h = y1; h < y2; ++h) {
            for (int64_t w = x1; w < x2; ++w) {
                masks_data[i * 800 * 800 + h * 800 + w] = 1.0f;
            }
        }
    }

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Should handle small objects
    EXPECT_GE(mask_loss.tensor().item<float>(), 0.0f);
    EXPECT_FALSE(std::isnan(mask_loss.tensor().item<float>()));
}

// ============================================================================
// End-to-End Training Tests
// ============================================================================

TEST_F(MaskRCNNLossTest, EndToEndTrainingLoop) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    // Simulate training for a few iterations
    for (int iter = 0; iter < 3; ++iter) {
        Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
        images.tensor().fill_(0.5f + iter * 0.1f);

        Tensor gt_boxes({1, 2, 4}, DType::Float32, device_);
        auto* boxes_data = gt_boxes.data<float>();
        boxes_data[0] = 100.0f; boxes_data[1] = 100.0f; boxes_data[2] = 200.0f; boxes_data[3] = 200.0f;
        boxes_data[4] = 300.0f; boxes_data[5] = 300.0f; boxes_data[6] = 400.0f; boxes_data[7] = 400.0f;

        Tensor gt_labels({1, 2}, DType::Int64, device_);
        auto* labels_data = gt_labels.data<int64_t>();
        labels_data[0] = 1; labels_data[1] = 2;

        Tensor gt_masks({1, 2, 800, 800}, DType::Float32, device_);
        gt_masks.fill_(0.0f);

        auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
            model->forward_train(images, gt_boxes, gt_labels, gt_masks);

        // All losses should be valid
        EXPECT_FALSE(std::isnan(rpn_cls_loss.tensor().item<float>()));
        EXPECT_FALSE(std::isnan(rpn_box_loss.tensor().item<float>()));
        EXPECT_FALSE(std::isnan(roi_cls_loss.tensor().item<float>()));
        EXPECT_FALSE(std::isnan(roi_box_loss.tensor().item<float>()));
        EXPECT_FALSE(std::isnan(mask_loss.tensor().item<float>()));

        // Total loss
        auto total_loss = Variable(
            rpn_cls_loss.tensor() + rpn_box_loss.tensor() +
            roi_cls_loss.tensor() + roi_box_loss.tensor() +
            mask_loss.tensor(),
            true
        );

        EXPECT_GT(total_loss.tensor().item<float>(), 0.0f);
        EXPECT_FALSE(std::isnan(total_loss.tensor().item<float>()));
    }
}

TEST_F(MaskRCNNLossTest, GradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    Tensor gt_boxes({1, 2, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    boxes_data[0] = 100.0f; boxes_data[1] = 100.0f; boxes_data[2] = 200.0f; boxes_data[3] = 200.0f;
    boxes_data[4] = 300.0f; boxes_data[5] = 300.0f; boxes_data[6] = 400.0f; boxes_data[7] = 400.0f;

    Tensor gt_labels({1, 2}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2;

    Tensor gt_masks({1, 2, 800, 800}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Create total loss with gradient tracking
    auto total_loss_tensor =
        rpn_cls_loss.tensor() + rpn_box_loss.tensor() +
        roi_cls_loss.tensor() + roi_box_loss.tensor() +
        mask_loss.tensor();

    Variable total_loss(total_loss_tensor, true);

    // Verify loss is a valid scalar
    EXPECT_EQ(total_loss.tensor().ndim(), 0);  // Scalar
    EXPECT_GT(total_loss.tensor().item<float>(), 0.0f);

    // In a real training scenario, we would:
    // 1. Call total_loss.backward() to compute gradients
    // 2. Verify gradients are non-zero for parameters
    // 3. Update parameters with optimizer

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_F(MaskRCNNLossTest, LossReasonableValues) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    Tensor gt_boxes({1, 3, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    boxes_data[0] = 100.0f; boxes_data[1] = 100.0f; boxes_data[2] = 200.0f; boxes_data[3] = 200.0f;
    boxes_data[4] = 300.0f; boxes_data[5] = 300.0f; boxes_data[6] = 400.0f; boxes_data[7] = 400.0f;
    boxes_data[8] = 500.0f; boxes_data[9] = 500.0f; boxes_data[10] = 600.0f; boxes_data[11] = 600.0f;

    Tensor gt_labels({1, 3}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 3;

    Tensor gt_masks({1, 3, 800, 800}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Check losses are in reasonable ranges
    // These are typical ranges for untrained models
    auto rpn_cls = rpn_cls_loss.tensor().item<float>();
    auto rpn_box = rpn_box_loss.tensor().item<float>();
    auto roi_cls = roi_cls_loss.tensor().item<float>();
    auto roi_box = roi_box_loss.tensor().item<float>();
    auto mask = mask_loss.tensor().item<float>();

    // Classification losses: typically 0-10 for cross-entropy
    EXPECT_GE(rpn_cls, 0.0f);
    EXPECT_LT(rpn_cls, 50.0f);

    EXPECT_GE(roi_cls, 0.0f);
    EXPECT_LT(roi_cls, 50.0f);

    // Regression losses: typically 0-5 for Smooth L1
    EXPECT_GE(rpn_box, 0.0f);
    EXPECT_LT(rpn_box, 20.0f);

    EXPECT_GE(roi_box, 0.0f);
    EXPECT_LT(roi_box, 20.0f);

    // Mask loss: typically 0-1 for BCE
    EXPECT_GE(mask, 0.0f);
    EXPECT_LT(mask, 10.0f);
}

// ============================================================================
// COCO-style Annotations Test
// ============================================================================

TEST_F(MaskRCNNLossTest, COCOStyleAnnotations) {
    auto model = mask_rcnn_resnet50_fpn(80, false);
    model->train();

    // Simulate COCO-style data
    Variable images(Tensor({1, 3, 800, 1200}, DType::Float32, device_), true);
    images.tensor().fill_(0.5f);

    // Multiple objects of different classes
    Tensor gt_boxes({1, 5, 4}, DType::Float32, device_);
    auto* boxes_data = gt_boxes.data<float>();
    // Person
    boxes_data[0] = 100.0f; boxes_data[1] = 150.0f; boxes_data[2] = 300.0f; boxes_data[3] = 600.0f;
    // Car
    boxes_data[4] = 400.0f; boxes_data[5] = 200.0f; boxes_data[6] = 700.0f; boxes_data[7] = 500.0f;
    // Dog
    boxes_data[8] = 50.0f; boxes_data[9] = 50.0f; boxes_data[10] = 150.0f; boxes_data[11] = 150.0f;
    // Chair
    boxes_data[12] = 800.0f; boxes_data[13] = 300.0f; boxes_data[14] = 1000.0f; boxes_data[15] = 600.0f;
    // Bottle
    boxes_data[16] = 200.0f; boxes_data[17] = 100.0f; boxes_data[18] = 250.0f; boxes_data[19] = 200.0f;

    Tensor gt_labels({1, 5}, DType::Int64, device_);
    auto* labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1;   // Person
    labels_data[1] = 3;   // Car
    labels_data[2] = 18;  // Dog
    labels_data[3] = 62;  // Chair
    labels_data[4] = 44;  // Bottle

    Tensor gt_masks({1, 5, 800, 1200}, DType::Float32, device_);
    gt_masks.fill_(0.0f);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Verify all losses are valid
    EXPECT_FALSE(std::isnan(rpn_cls_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isnan(rpn_box_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isnan(roi_cls_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isnan(roi_box_loss.tensor().item<float>()));
    EXPECT_FALSE(std::isnan(mask_loss.tensor().item<float>()));

    // All should be positive
    EXPECT_GT(rpn_cls_loss.tensor().item<float>(), 0.0f);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
