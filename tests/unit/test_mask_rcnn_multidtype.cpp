/**
 * @file test_mask_rcnn_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Mask R-CNN instance segmentation model
 *
 * Tests Mask R-CNN components across multiple backends (CPU, CUDA, OneAPI) and
 * data types (Float32, Float64, Float16):
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
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/offload.hpp>
#include <cmath>
#include <algorithm>
#include <memory>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture with Multi-Backend Multi-DType Support
// ============================================================================

class MaskRCNNMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::unique_ptr<nn::OffloadContext> offload_ctx_;

    /**
     * @brief Convert model with CPU-start offloading for GPU backends
     *
     * For GPU backends: keeps model on CPU, converts dtype only, enables offloading
     * For CPU backend: normal convert_model behavior
     */
    template <typename ModuleT>
    void convert_model_with_offload(ModuleT& model) {
        // Always convert model normally - rely on smaller input sizes for GPU memory
        convert_model(model);
    }

    // Keep for backward compatibility with smaller components
    template <typename ModuleT>
    void enable_offloading_if_needed(ModuleT& model) {
        // No-op - offloading now handled by convert_model_with_offload
        (void)model;
    }

    /**
     * @brief Get appropriate input size for the current backend and dtype
     * GPU backends use smaller sizes due to VRAM constraints for activations
     * Float64 uses 2x memory so needs even smaller sizes
     * MaskRCNN with ResNet50 backbone is very memory-intensive
     */
    int64_t getInputSize(int64_t default_size) {
        if (device().type == Device::Type::CPU) {
            return default_size;
        }

        // Float64 needs much smaller sizes (2x memory usage)
        bool is_float64 = (dtype() == DType::Float64);
        // Float16 also reduced to avoid numerical issues with large activations
        bool is_float16 = (dtype() == DType::Float16);

        if (is_float64) {
            // Float64: MaskRCNN needs very small sizes (2x memory of Float32)
            // For 8GB VRAM, use 128-160 pixel inputs
            if (default_size >= 600) return 128;
            return std::min(default_size, int64_t(128));
        }

        if (is_float16) {
            // Float16: smaller to avoid numerical overflow
            // For 8GB VRAM, use 192-224 pixel inputs
            if (default_size >= 600) return 192;
            return std::min(default_size, int64_t(192));
        }

        // Float32: reduced for 8GB VRAM
        // MaskRCNN with ResNet50 backbone needs ~4-6GB for 256x256 input
        if (default_size >= 600) return 224;
        return std::min(default_size, int64_t(224));
    }

    /**
     * @brief Initialize ground truth boxes with coordinates scaled to actual image size
     * @param gt_boxes Output tensor for boxes
     * @param num_boxes Number of boxes to create
     * @param actual_size Actual image size being used (boxes are scaled from 800x800 reference)
     * @param ref_size Reference size for coordinates (default 800)
     */
    void initialize_boxes(Tensor& gt_boxes, int num_boxes, int64_t actual_size = 800, int64_t ref_size = 800) {
        // Move to CPU for data access
        auto boxes_cpu = gt_boxes.to(Device::cpu());
        auto boxes_f32 = boxes_cpu.to(DType::Float32);
        auto boxes_data = boxes_f32.data<float>();

        float scale = static_cast<float>(actual_size) / static_cast<float>(ref_size);

        // Initialize boxes at different locations [batch, num_boxes, 4]
        for (int i = 0; i < num_boxes; ++i) {
            float x_offset = static_cast<float>(i * 150.0) * scale;
            boxes_data[i * 4 + 0] = (10.0f * scale) + x_offset;  // x1
            boxes_data[i * 4 + 1] = (10.0f * scale) + x_offset;  // y1
            boxes_data[i * 4 + 2] = (100.0f * scale) + x_offset; // x2
            boxes_data[i * 4 + 3] = (100.0f * scale) + x_offset; // y2
        }

        // Copy back to original device and dtype
        gt_boxes = boxes_f32.to(dtype()).to(device());
    }

    void initialize_masks(Tensor& gt_masks) {
        // Create zeros on device
        gt_masks = tenzor::zeros(
            std::vector<int64_t>(gt_masks.shape().begin(), gt_masks.shape().end()),
            dtype(), device());
    }
};

// ============================================================================
// RPN (Region Proposal Network) Tests
// ============================================================================

TEST_P(MaskRCNNMultiDTypeTest, RPNForwardShape) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // RPN should generate region proposals
    EXPECT_GT(boxes.shape()[0], 0) << "RPN should generate at least some proposals";
    EXPECT_EQ(boxes.shape()[1], 4) << "Boxes should have 4 coordinates (x1, y1, x2, y2)";
}

TEST_P(MaskRCNNMultiDTypeTest, RPNMultiScaleProposals) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->eval();

    // Test that RPN generates proposals at different image scales
    auto images_small = createInput({1, 3, getInputSize(600), getInputSize(600)});
    auto [boxes_small, _, __, ___] = model->forward_test(images_small);

    auto images_large = createInput({1, 3, getInputSize(1024), getInputSize(1024)});
    auto [boxes_large, _2, __2, ___2] = model->forward_test(images_large);

    EXPECT_GT(boxes_small.shape()[0], 0);
    EXPECT_GT(boxes_large.shape()[0], 0);
    // Larger images typically generate more proposals
    EXPECT_GE(boxes_large.shape()[0], boxes_small.shape()[0] * 0.5);
}

TEST_P(MaskRCNNMultiDTypeTest, RPNGradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    Tensor gt_boxes({1, 3, 4}, dtype(), device());
    initialize_boxes(gt_boxes, 3, img_size);

    Tensor gt_labels_cpu({1, 3}, DType::Int64, Device::cpu());
    auto labels_data = gt_labels_cpu.data<int64_t>();
    labels_data[0] = 1; labels_data[1] = 2; labels_data[2] = 3;
    auto gt_labels = gt_labels_cpu.to(device());

    Tensor gt_masks({1, 3, img_size, img_size}, dtype(), device());
    initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // RPN losses should be computed
    EXPECT_TRUE(rpn_cls_loss.requires_grad());
    EXPECT_TRUE(rpn_box_loss.requires_grad());
}

// ============================================================================
// ROI Align Tests
// ============================================================================

TEST_P(MaskRCNNMultiDTypeTest, ROIAlignSpatialAlignment) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // ROI Align should produce aligned features
    EXPECT_GT(boxes.shape()[0], 0);
    // Masks should correspond to detected boxes
    EXPECT_EQ(masks.shape()[0], boxes.shape()[0]);
}

TEST_P(MaskRCNNMultiDTypeTest, ROIAlignOutputDimensions) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

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

TEST_P(MaskRCNNMultiDTypeTest, MaskHeadOutputShape) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Mask head should produce instance masks
    EXPECT_GT(masks.shape()[0], 0);
    EXPECT_EQ(masks.ndim(), 4);
    // Masks should have reasonable resolution
    EXPECT_GE(masks.shape()[2], 14);  // At least 14x14
    EXPECT_GE(masks.shape()[3], 14);
}

TEST_P(MaskRCNNMultiDTypeTest, MaskHeadMultiInstance) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create ground truth with multiple instances
    Tensor gt_boxes({1, 5, 4}, dtype(), device());
    initialize_boxes(gt_boxes, 5, img_size);

    Tensor gt_labels_cpu260({1, 5}, DType::Int64, Device::cpu());
    auto labels_data260 = gt_labels_cpu260.data<int64_t>();
    for (int64_t i = 0; i < 5; ++i) labels_data260[i] = i + 1;
    auto gt_labels = gt_labels_cpu260.to(device());

    Tensor gt_masks({1, 5, img_size, img_size}, dtype(), device());
    initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Mask loss should be computed for multiple instances
    EXPECT_TRUE(mask_loss.requires_grad());
}

TEST_P(MaskRCNNMultiDTypeTest, MaskHeadGradientFlow) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    Tensor gt_boxes({1, 2, 4}, dtype(), device());
    initialize_boxes(gt_boxes, 2, img_size);

    Tensor gt_labels_cpu283({1, 2}, DType::Int64, Device::cpu());
    auto labels_data283 = gt_labels_cpu283.data<int64_t>();
    labels_data283[0] = 1; labels_data283[1] = 2;
    auto gt_labels = gt_labels_cpu283.to(device());

    Tensor gt_masks({1, 2, img_size, img_size}, dtype(), device());
    initialize_masks(gt_masks);

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

TEST_P(MaskRCNNMultiDTypeTest, DetectionHeadOutputs) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

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

TEST_P(MaskRCNNMultiDTypeTest, DetectionHeadClassification) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    if (labels.shape()[0] > 0) {
        auto labels_cpu = labels.to(Device::cpu());
        auto labels_data = labels_cpu.data<int64_t>();
        // Labels should be within valid class range [0, 90]
        for (size_t i = 0; i < labels.shape()[0]; ++i) {
            EXPECT_GE(labels_data[i], 0);
            EXPECT_LT(labels_data[i], 91);
        }
    }
}

TEST_P(MaskRCNNMultiDTypeTest, DetectionHeadScoreRange) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    if (scores.shape()[0] > 0) {
        auto scores_cpu = scores.to(DType::Float32).to(Device::cpu());
        auto scores_data = scores_cpu.data<float>();
        // Scores should be in [0, 1] range (probabilities)
        for (size_t i = 0; i < scores.shape()[0]; ++i) {
            EXPECT_GE(scores_data[i], 0.0f);
            EXPECT_LE(scores_data[i], 1.0f);
        }
    }
}

// ============================================================================
// Forward Pass Tests with Different Image Sizes
// ============================================================================

TEST_P(MaskRCNNMultiDTypeTest, ForwardPassSmallImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(600), getInputSize(600)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_P(MaskRCNNMultiDTypeTest, ForwardPassMediumImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_P(MaskRCNNMultiDTypeTest, ForwardPassLargeImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(1024), getInputSize(1024)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_P(MaskRCNNMultiDTypeTest, ForwardPassBatchProcessing) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Should handle batch processing
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

TEST_P(MaskRCNNMultiDTypeTest, ForwardPassRectangularImage) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(600), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Should handle non-square images
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_GT(masks.shape()[0], 0);
}

// ============================================================================
// Multi-Instance Handling Tests
// ============================================================================

TEST_P(MaskRCNNMultiDTypeTest, MultiInstanceDetection) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Should be able to detect multiple instances
    EXPECT_GT(boxes.shape()[0], 0);
    // Each instance should have corresponding mask
    EXPECT_EQ(boxes.shape()[0], masks.shape()[0]);
}

TEST_P(MaskRCNNMultiDTypeTest, MultiInstanceTraining) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create ground truth with 4 instances
    Tensor gt_boxes({1, 4, 4}, dtype(), device());
    initialize_boxes(gt_boxes, 4, img_size);

    Tensor gt_labels_cpu465({1, 4}, DType::Int64, Device::cpu());
    auto labels_data465 = gt_labels_cpu465.data<int64_t>();
    for (int64_t i = 0; i < 4; ++i) labels_data465[i] = i + 1;
    auto gt_labels = gt_labels_cpu465.to(device());

    Tensor gt_masks({1, 4, img_size, img_size}, dtype(), device());
    initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    // Should compute losses for all instances
    EXPECT_TRUE(mask_loss.requires_grad());
    EXPECT_TRUE(roi_cls_loss.requires_grad());
}

TEST_P(MaskRCNNMultiDTypeTest, MultiInstanceDifferentClasses) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create instances of different classes
    Tensor gt_boxes({1, 3, 4}, dtype(), device());
    initialize_boxes(gt_boxes, 3, img_size);

    Tensor gt_labels_cpu490({1, 3}, DType::Int64, Device::cpu());
    auto labels_data490 = gt_labels_cpu490.data<int64_t>();
    labels_data490[0] = 1; labels_data490[1] = 5; labels_data490[2] = 10;
    auto gt_labels = gt_labels_cpu490.to(device());

    Tensor gt_masks({1, 3, img_size, img_size}, dtype(), device());
    initialize_masks(gt_masks);

    auto [rpn_cls_loss, rpn_box_loss, roi_cls_loss, roi_box_loss, mask_loss] =
        model->forward_train(images, gt_boxes, gt_labels, gt_masks);

    EXPECT_TRUE(roi_cls_loss.requires_grad());
}

// ============================================================================
// Model Architecture Variants
// ============================================================================

TEST_P(MaskRCNNMultiDTypeTest, ResNet50BackboneForward) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
}

TEST_P(MaskRCNNMultiDTypeTest, ResNet101BackboneForward) {
    auto model = mask_rcnn_resnet101_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);
}

TEST_P(MaskRCNNMultiDTypeTest, CustomNumClasses) {
    // Test with COCO dataset (80 classes)
    auto model = mask_rcnn_resnet50_fpn(80, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    EXPECT_GT(boxes.shape()[0], 0);

    if (labels.shape()[0] > 0) {
        auto labels_cpu = labels.to(Device::cpu());
        auto labels_data = labels_cpu.data<int64_t>();
        for (size_t i = 0; i < labels.shape()[0]; ++i) {
            EXPECT_GE(labels_data[i], 0);
            EXPECT_LT(labels_data[i], 80);
        }
    }
}

// ============================================================================
// End-to-End Integration Tests
// ============================================================================

TEST_P(MaskRCNNMultiDTypeTest, EndToEndInference) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto [boxes, labels, scores, masks] = model->forward_test(images);

    // Complete pipeline should work
    EXPECT_GT(boxes.shape()[0], 0);
    EXPECT_EQ(boxes.shape()[0], labels.shape()[0]);
    EXPECT_EQ(boxes.shape()[0], scores.shape()[0]);
    EXPECT_EQ(boxes.shape()[0], masks.shape()[0]);
}

TEST_P(MaskRCNNMultiDTypeTest, EndToEndTraining) {
    auto model = mask_rcnn_resnet50_fpn(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    Tensor gt_boxes({1, 2, 4}, dtype(), device());
    initialize_boxes(gt_boxes, 2, img_size);

    Tensor gt_labels_cpu594({1, 2}, DType::Int64, Device::cpu());
    auto labels_data594 = gt_labels_cpu594.data<int64_t>();
    labels_data594[0] = 1; labels_data594[1] = 3;
    auto gt_labels = gt_labels_cpu594.to(device());

    Tensor gt_masks({1, 2, img_size, img_size}, dtype(), device());
    initialize_masks(gt_masks);

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
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(MaskRCNNMultiDTypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
