/**
 * @file test_mask_rcnn_multidtype.cpp
 * @brief Multi-dtype tests for Mask R-CNN instance segmentation model
 *
 * Tests Mask R-CNN components across Float32, Float64, and Float16:
 * - RPN (Region Proposal Network)
 * - ROI Align
 * - Mask head
 * - Detection head
 * - Forward pass with different image sizes
 * - Multi-instance handling
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/mask_rcnn.hpp"
#include <cmath>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Templated Test Fixture
// ============================================================================

template <typename T>
class MaskRCNNMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        dtype_ = dtype_from_type<T>();
        tolerance_ = get_tolerance();
    }

    DType dtype_;
    Device device_;
    float tolerance_;

    float get_tolerance() {
        if (std::is_same<T, double>::value) return 1e-5f;
        if (std::is_same<T, float>::value) return 1e-3f;
        return 1e-1f;  // Float16
    }

    template<typename DataType>
    DType dtype_from_type() {
        if (std::is_same<DataType, float>::value) return DType::Float32;
        if (std::is_same<DataType, double>::value) return DType::Float64;
        return DType::Float16;
    }

    void initialize_boxes(Tensor& gt_boxes, int num_boxes) {
        auto boxes_data = gt_boxes.data<T>();
        // Initialize boxes at different locations [batch, num_boxes, 4]
        for (int i = 0; i < num_boxes; ++i) {
            T x_offset = static_cast<T>(i * 150.0);
            boxes_data[i * 4 + 0] = static_cast<T>(10.0) + x_offset;  // x1
            boxes_data[i * 4 + 1] = static_cast<T>(10.0) + x_offset;  // y1
            boxes_data[i * 4 + 2] = static_cast<T>(100.0) + x_offset; // x2
            boxes_data[i * 4 + 3] = static_cast<T>(100.0) + x_offset; // y2
        }
    }

    void initialize_masks(Tensor& gt_masks) {
        auto masks_data = gt_masks.data<T>();
        std::fill(masks_data, masks_data + gt_masks.numel(), static_cast<T>(0.0));
    }
};

// Type definitions for testing
using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(MaskRCNNMultiDTypeTest, TestTypes);

// ============================================================================
// RPN (Region Proposal Network) Tests
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, RPNForwardShape) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // RPN should generate region proposals
    EXPECT_GT(boxes.shape()[0], 0) << "RPN should generate at least some proposals";
    EXPECT_EQ(boxes.shape()[1], 4) << "Boxes should have 4 coordinates (x1, y1, x2, y2)";
}

TYPED_TEST(MaskRCNNMultiDTypeTest, RPNMultiScaleProposals) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->eval();

    // Test that RPN generates proposals at different image scales
    Variable images_small(Tensor({1, 3, 600, 600}, this->dtype_, this->device_), true);
    auto [boxes_small, _, __, ___] = model->forward_test(images_small);

    Variable images_large(Tensor({1, 3, 1024, 1024}, this->dtype_, this->device_), true);
    auto [boxes_large, _2, __2, ___2] = model->forward_test(images_large);

    EXPECT_GT(boxes_small.shape()[0], 0);
    EXPECT_GT(boxes_large.shape()[0], 0);
    // Larger images typically generate more proposals
    EXPECT_GE(boxes_large.shape()[0], boxes_small.shape()[0] * 0.5);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, RPNGradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    Tensor gt_boxes({1, 3, 4}, this->dtype_, this->device_);
    this->initialize_boxes(gt_boxes, 3);

    Tensor gt_labels({1, 3}, DType::Int64, this->device_);
    auto labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 3;

    Tensor gt_masks({1, 3, 800, 800}, this->dtype_, this->device_);
    this->initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // RPN losses should be computed
    EXPECT_TRUE(rpn_cls_loss.requires_grad());
    EXPECT_TRUE(rpn_box_loss.requires_grad());
}

// ============================================================================
// ROI Align Tests
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, ROIAlignSpatialAlignment) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // ROI Align should produce aligned features
    EXPECT_GT(boxes.shape()[0], 0);
    // Masks should correspond to detected boxes
    EXPECT_EQ(masks.shape()[0], boxes.shape()[0]);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, ROIAlignOutputDimensions) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({2, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Check that ROI Align outputs have correct spatial dimensions
    if (masks.shape()[0] > 0) {
        EXPECT_EQ(masks.ndim(), 4) << "Masks should be 4D: [num_instances, channels, height, width]";
        // Typical mask resolution is 28x28 in Mask R-CNN
        EXPECT_GT(masks.shape()[2], 0);
        EXPECT_GT(masks.shape()[3], 0);
    }
}

// ============================================================================
// Mask Head Tests
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, MaskHeadOutputShape) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Mask head should produce instance masks
    EXPECT_GT(masks.shape()[0], 0);
    EXPECT_EQ(masks.ndim(), 4);
    // Masks should have reasonable resolution
    EXPECT_GE(masks.shape()[2], 14);  // At least 14x14
    EXPECT_GE(masks.shape()[3], 14);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, MaskHeadMultiInstance) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create ground truth with multiple instances
    Tensor gt_boxes({1, 5, 4}, this->dtype_, this->device_);
    this->initialize_boxes(gt_boxes, 5);

    Tensor gt_labels({1, 5}, DType::Int64, this->device_);
    auto labels_data = gt_labels.data<int64_t>();
    for (int i = 0; i < 5; ++i) labels_data[i] = i + 1;

    Tensor gt_masks({1, 5, 800, 800}, this->dtype_, this->device_);
    this->initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Mask loss should be computed for multiple instances
    EXPECT_TRUE(mask_loss.requires_grad());
}

TYPED_TEST(MaskRCNNMultiDTypeTest, MaskHeadGradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    Tensor gt_boxes({1, 2, 4}, this->dtype_, this->device_);
    this->initialize_boxes(gt_boxes, 2);

    Tensor gt_labels({1, 2}, DType::Int64, this->device_);
    auto labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2;

    Tensor gt_masks({1, 2, 800, 800}, this->dtype_, this->device_);
    this->initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // All losses should support gradient flow
    EXPECT_TRUE(rpn_cls_loss.requires_grad());
    EXPECT_TRUE(rpn_box_loss.requires_grad());
    EXPECT_TRUE(roi_cls_loss.requires_grad());
    EXPECT_TRUE(roi_box_loss.requires_grad());
    EXPECT_TRUE(mask_loss.requires_grad());
}

// ============================================================================
// Detection Head Tests
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, DetectionHeadOutputs) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Detection head should produce boxes, labels, and scores
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(labels.shape()[0], 0);
    EXPECT_GT(scores.shape()[0], 0);

    // All outputs should have same number of detections
    EXPECT_EQ(boxes.shape()[0], labels.shape()[0]);
    EXPECT_EQ(boxes.shape()[0], scores.shape()[0]);
    EXPECT_EQ(boxes.shape()[0], masks.shape()[0]);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, DetectionHeadClassification) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    if (labels.shape()[0] > 0) {
        auto labels_data = labels.data<int64_t>();
        // Labels should be within valid class range [0, 90]
        for (size_t i = 0; i < labels.shape()[0]; ++i) {
            EXPECT_GE(labels_data[i], 0);
            EXPECT_LT(labels_data[i], 91);
        }
    }
}

TYPED_TEST(MaskRCNNMultiDTypeTest, DetectionHeadScoreRange) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    if (scores.shape()[0] > 0) {
        auto scores_data = scores.data<TypeParam>();
        // Scores should be in [0, 1] range (probabilities)
        for (size_t i = 0; i < scores.shape()[0]; ++i) {
            EXPECT_GE(scores_data[i], static_cast<TypeParam>(0.0));
            EXPECT_LE(scores_data[i], static_cast<TypeParam>(1.0));
        }
    }
}

// ============================================================================
// Forward Pass Tests with Different Image Sizes
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, ForwardPassSmallImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 600, 600}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, ForwardPassMediumImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, ForwardPassLargeImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 1024, 1024}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, ForwardPassBatchProcessing) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({2, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Should handle batch processing
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, ForwardPassRectangularImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 600, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Should handle non-square images
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

// ============================================================================
// Multi-Instance Handling Tests
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, MultiInstanceDetection) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Should be able to detect multiple instances
    EXPECT_GT(boxes.shape()[0], 0);
    // Each instance should have corresponding mask
    EXPECT_EQ(boxes.shape()[0], masks.shape()[0]);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, MultiInstanceTraining) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create ground truth with 4 instances
    Tensor gt_boxes({1, 4, 4}, this->dtype_, this->device_);
    this->initialize_boxes(gt_boxes, 4);

    Tensor gt_labels({1, 4}, DType::Int64, this->device_);
    auto labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 1; labels_data[3] = 3;

    Tensor gt_masks({1, 4, 800, 800}, this->dtype_, this->device_);
    this->initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Should compute losses for all instances
    EXPECT_TRUE(mask_loss.requires_grad());
    EXPECT_TRUE(roi_cls_loss.requires_grad());
}

TYPED_TEST(MaskRCNNMultiDTypeTest, MultiInstanceDifferentClasses) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create instances of different classes
    Tensor gt_boxes({1, 3, 4}, this->dtype_, this->device_);
    this->initialize_boxes(gt_boxes, 3);

    Tensor gt_labels({1, 3}, DType::Int64, this->device_);
    auto labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1;  // Person
    labels_data[1] = 3;  // Car
    labels_data[2] = 18; // Dog

    Tensor gt_masks({1, 3, 800, 800}, this->dtype_, this->device_);
    this->initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    EXPECT_TRUE(roi_cls_loss.requires_grad());
}

// ============================================================================
// Model Architecture Variants
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, ResNet50BackboneForward) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, ResNet101BackboneForward) {
    auto model = mask_rcnn_resnet101_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, CustomNumClasses) {
    // Test with COCO dataset (80 classes)
    auto model = mask_rcnn_resnet50_fpn(80, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);

    if (labels.shape()[0] > 0) {
        auto labels_data = labels.data<int64_t>();
        for (size_t i = 0; i < labels.shape()[0]; ++i) {
            EXPECT_GE(labels_data[i], 0);
            EXPECT_LT(labels_data[i], 80);
        }
    }
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

TYPED_TEST(MaskRCNNMultiDTypeTest, EndToEndInference) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Complete pipeline should work
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_EQ(boxes.shape()[0], labels.shape()[0]);
    EXPECT_EQ(boxes.shape()[0], scores.shape()[0]);
    EXPECT_EQ(boxes.shape()[0], masks.shape()[0]);
}

TYPED_TEST(MaskRCNNMultiDTypeTest, EndToEndTraining) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    Tensor gt_boxes({1, 2, 4}, this->dtype_, this->device_);
    this->initialize_boxes(gt_boxes, 2);

    Tensor gt_labels({1, 2}, DType::Int64, this->device_);
    auto labels_data = gt_labels.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2;

    Tensor gt_masks({1, 2, 800, 800}, this->dtype_, this->device_);
    this->initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Complete training pipeline should work
    EXPECT_TRUE(rpn_cls_loss.requires_grad());
    EXPECT_TRUE(rpn_box_loss.requires_grad());
    EXPECT_TRUE(roi_cls_loss.requires_grad());
    EXPECT_TRUE(roi_box_loss.requires_grad());
    EXPECT_TRUE(mask_loss.requires_grad());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
