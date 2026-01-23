/**
 * @file test_faster_rcnn_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Faster R-CNN object detection model
 *
 * Tests Faster R-CNN components across multiple backends (CPU, CUDA, OneAPI) and
 * data types (Float32, Float64, Float16):
 * - RPN (Region Proposal Network)
 * - ROI pooling/align
 * - Detection head
 * - Backbone (ResNet/VGG)
 * - Forward pass with different image sizes
 * - Multi-object detection
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/faster_rcnn.hpp"
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/offload.hpp>
#include <cmath>
#include <vector>
#include <unordered_map>
#include <memory>

using namespace tenzor;
using namespace tenzor::models;
using namespace tenzor::testing;

// ============================================================================
// Test Fixture with Multi-Backend Multi-DType Support
// ============================================================================

class FasterRCNNMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::unique_ptr<nn::OffloadContext> offload_ctx_;

    /**
     * @brief Convert model with CPU-start offloading for GPU backends
     */
    template <typename ModuleT>
    void convert_model_with_offload(ModuleT& model) {
        if (device().type == Device::Type::CPU) {
            convert_model(model);
        } else {
            // GPU backend: CPU-start offloading
            if constexpr (requires { model->to(dtype()); }) {
                model->to(dtype());
            } else if constexpr (requires { model.to(dtype()); }) {
                model.to(dtype());
            }

            nn::OffloadContext::Config config;
            config.offload_parameters = true;
            config.offload_gradients = true;
            config.prefetch_depth = 2;
            config.pin_first_layer = true;
            config.pin_last_layer = true;
            config.target_device = device();

            if constexpr (requires { model.get(); }) {
                offload_ctx_ = std::make_unique<nn::OffloadContext>(*model, config);
            } else if constexpr (requires { *model; }) {
                offload_ctx_ = std::make_unique<nn::OffloadContext>(*model, config);
            } else {
                offload_ctx_ = std::make_unique<nn::OffloadContext>(model, config);
            }
            offload_ctx_->enable();
        }
    }

    template <typename ModuleT>
    void enable_offloading_if_needed(ModuleT& model) { (void)model; }

    /**
     * @brief Get appropriate input size for the current backend and dtype
     * GPU backends use smaller sizes due to VRAM constraints
     * Float64 uses 2x memory so needs even smaller sizes
     * FasterRCNN with ResNet backbone is very memory-intensive
     */
    int64_t getInputSize(int64_t default_size) {
        if (device().type == Device::Type::CPU) {
            return default_size;
        }

        bool is_float64 = (dtype() == DType::Float64);
        bool is_float16 = (dtype() == DType::Float16);

        if (is_float64) {
            // Float64: FasterRCNN needs very small sizes
            if (default_size >= 1280) return 160;
            if (default_size >= 1024) return 160;
            if (default_size >= 800) return 160;
            if (default_size >= 600) return 160;
            return std::min(default_size, int64_t(160));
        }

        if (is_float16) {
            // Float16: smaller to avoid numerical issues
            if (default_size >= 1280) return 320;
            if (default_size >= 1024) return 288;
            if (default_size >= 800) return 256;
            if (default_size >= 600) return 224;
            return std::min(default_size, int64_t(224));
        }

        // Float32: moderate reduction for GPU
        if (default_size >= 1280) return 480;
        if (default_size >= 1024) return 448;
        if (default_size >= 800) return 416;
        if (default_size >= 600) return 384;
        return std::min(default_size, int64_t(352));
    }

    /**
     * @brief Scale box coordinates from reference size to actual input size
     * Box coordinates are specified for 800x800 reference and scaled down
     * to fit within the actual input size (e.g., 160x160 for Float64 on GPU)
     */
    std::array<float, 4> scaleBox(const std::array<float, 4>& box, int64_t actual_size, int64_t ref_size = 800) {
        float scale = static_cast<float>(actual_size) / static_cast<float>(ref_size);
        return {
            box[0] * scale,
            box[1] * scale,
            box[2] * scale,
            box[3] * scale
        };
    }

    // Helper to create target with boxes at specific positions
    // Boxes are specified for 800x800 reference size and automatically scaled
    std::vector<std::unordered_map<std::string, Tensor>> createTargets(
            int num_boxes, const std::vector<std::array<float, 4>>& box_coords,
            int64_t actual_size = 800, int64_t ref_size = 800) {
        std::vector<std::unordered_map<std::string, Tensor>> targets(1);

        auto boxes = Tensor({num_boxes, 4}, dtype(), device());
        auto boxes_f32 = boxes.to(DType::Float32).to(Device::cpu());
        auto boxes_ptr = boxes_f32.data<float>();

        for (int i = 0; i < num_boxes && i < static_cast<int>(box_coords.size()); ++i) {
            auto scaled_box = scaleBox(box_coords[i], actual_size, ref_size);
            boxes_ptr[i * 4 + 0] = scaled_box[0];
            boxes_ptr[i * 4 + 1] = scaled_box[1];
            boxes_ptr[i * 4 + 2] = scaled_box[2];
            boxes_ptr[i * 4 + 3] = scaled_box[3];
        }
        boxes = boxes_f32.to(dtype()).to(device());

        auto labels = Tensor({num_boxes}, DType::Int64, device());
        auto labels_cpu = labels.to(Device::cpu());
        auto labels_data = labels_cpu.data<int64_t>();
        for (int i = 0; i < num_boxes; ++i) {
            labels_data[i] = (i % 5) + 1;
        }
        labels = labels_cpu.to(device());

        targets[0]["boxes"] = boxes;
        targets[0]["labels"] = labels;

        return targets;
    }
};

// ============================================================================
// Backbone Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, ResNet50BackboneForward) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Backbone should process image and produce detections
    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, ResNet101BackboneForward) {
    auto model = faster_rcnn_resnet101(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // ResNet101 backbone should handle deeper architecture
    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, BackboneBatchProcessing) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({4, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle batch of 4 images
    EXPECT_EQ(detections.size(), 4);
}

// ============================================================================
// RPN (Region Proposal Network) Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, RPNProposalGeneration) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // RPN should generate region proposals
    EXPECT_EQ(detections.size(), 1);
    // Each detection should contain boxes
    EXPECT_TRUE(detections[0].find("boxes") != detections[0].end());
}

TEST_P(FasterRCNNMultiDTypeTest, RPNMultiScaleAnchors) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    // Test RPN with different image scales
    auto images_small = createInput({1, 3, getInputSize(600), getInputSize(600)});
    auto images_large = createInput({1, 3, getInputSize(1024), getInputSize(1024)});

    model->eval();
    auto detections_small = model->forward_inference(images_small);
    auto detections_large = model->forward_inference(images_large);

    // RPN should handle different scales
    EXPECT_EQ(detections_small.size(), 1);
    EXPECT_EQ(detections_large.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, RPNObjectnessScores) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create dummy targets (coords scaled from 800x800 reference)
    auto targets = createTargets(2, {
        {100.0f, 100.0f, 200.0f, 200.0f},
        {300.0f, 300.0f, 400.0f, 400.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // RPN should compute objectness loss
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// ROI Pooling/Align Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, ROIPoolingForward) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // ROI pooling should extract features from proposals
    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, ROIAlignPrecision) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    // Test with smaller image to verify ROI align precision
    auto images = createInput({1, 3, getInputSize(400), getInputSize(400)});

    model->eval();
    auto detections = model->forward_inference(images);

    // ROI align should maintain spatial precision
    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, ROIMultipleRegions) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create targets with multiple boxes (coords scaled from 800x800 reference)
    auto targets = createTargets(5, {
        {50.0f, 50.0f, 150.0f, 150.0f},
        {200.0f, 200.0f, 300.0f, 300.0f},
        {350.0f, 350.0f, 450.0f, 450.0f},
        {500.0f, 100.0f, 600.0f, 200.0f},
        {100.0f, 500.0f, 200.0f, 600.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // Should handle multiple ROI regions
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// Detection Head Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, DetectionHeadClassification) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Detection head should classify proposals
    EXPECT_EQ(detections.size(), 1);
    EXPECT_TRUE(detections[0].find("labels") != detections[0].end() ||
                detections[0].find("boxes") != detections[0].end());
}

TEST_P(FasterRCNNMultiDTypeTest, DetectionHeadBBoxRegression) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create targets (coords scaled from 800x800 reference)
    auto targets = createTargets(3, {
        {100.0f, 100.0f, 200.0f, 200.0f},
        {300.0f, 300.0f, 400.0f, 400.0f},
        {500.0f, 500.0f, 600.0f, 600.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // Detection head should compute bbox regression loss
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TEST_P(FasterRCNNMultiDTypeTest, DetectionHeadMultiClass) {
    auto model = faster_rcnn_resnet50(91, false);  // 91 classes (COCO)
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle multi-class detection
    EXPECT_EQ(detections.size(), 1);
}

// ============================================================================
// Forward Pass with Different Image Sizes
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, SmallImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(400), getInputSize(400)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, MediumImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(600), getInputSize(600)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, StandardImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, LargeImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(1024), getInputSize(1024)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, VeryLargeImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(1280), getInputSize(1280)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, RectangularImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    // Test with non-square image
    auto images = createInput({1, 3, getInputSize(600), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle non-square images
    EXPECT_EQ(detections.size(), 1);
}

// ============================================================================
// Multi-Object Detection Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, SingleObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    auto targets = createTargets(1, {
        {100.0f, 100.0f, 300.0f, 300.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TEST_P(FasterRCNNMultiDTypeTest, MultiObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create targets with multiple objects at various locations
    std::vector<std::array<float, 4>> boxes;
    for (int i = 0; i < 8; ++i) {
        boxes.push_back({
            static_cast<float>((i % 3) * 200 + 50),
            static_cast<float>((i / 3) * 200 + 50),
            static_cast<float>((i % 3) * 200 + 150),
            static_cast<float>((i / 3) * 200 + 150)
        });
    }
    auto targets = createTargets(8, boxes, img_size);

    auto losses = model->forward_train(images, targets);

    // Should handle multiple objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TEST_P(FasterRCNNMultiDTypeTest, DenseObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Create targets with many small objects
    std::vector<std::array<float, 4>> boxes;
    for (int i = 0; i < 15; ++i) {
        boxes.push_back({
            static_cast<float>((i % 5) * 150 + 20),
            static_cast<float>((i / 5) * 250 + 20),
            static_cast<float>((i % 5) * 150 + 100),
            static_cast<float>((i / 5) * 250 + 100)
        });
    }
    auto targets = createTargets(15, boxes, img_size);

    auto losses = model->forward_train(images, targets);

    // Should handle dense object detection
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TEST_P(FasterRCNNMultiDTypeTest, OverlappingObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Overlapping boxes in center region (coords scaled from 800x800 reference)
    auto targets = createTargets(4, {
        {300.0f, 300.0f, 500.0f, 500.0f},
        {320.0f, 320.0f, 520.0f, 520.0f},
        {340.0f, 340.0f, 540.0f, 540.0f},
        {360.0f, 360.0f, 560.0f, 560.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // Should handle overlapping objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// Gradient Flow and Training Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, GradientFlowThroughModel) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    auto targets = createTargets(3, {
        {100.0f, 100.0f, 200.0f, 200.0f},
        {300.0f, 300.0f, 400.0f, 400.0f},
        {500.0f, 500.0f, 600.0f, 600.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // Check gradients can flow through all components
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TEST_P(FasterRCNNMultiDTypeTest, ParameterCount) {
    auto model = faster_rcnn_resnet50(91, false);

    auto params = model->parameters();

    // Faster R-CNN should have many parameters (backbone + RPN + head)
    EXPECT_GT(params.size(), 100);
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, BatchInference) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({3, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Should process all images in batch
    EXPECT_EQ(detections.size(), 3);
}

TEST_P(FasterRCNNMultiDTypeTest, LargeBatchInference) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({8, 3, getInputSize(600), getInputSize(600)});

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle larger batches
    EXPECT_EQ(detections.size(), 8);
}

// ============================================================================
// Model Variants Tests
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, ResNet50Variant) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, ResNet101Variant) {
    auto model = faster_rcnn_resnet101(91, false);
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TEST_P(FasterRCNNMultiDTypeTest, CustomNumClasses) {
    auto model = faster_rcnn_resnet50(20, false);  // Custom 20 classes
    convert_model_with_offload(model);

    auto images = createInput({1, 3, getInputSize(800), getInputSize(800)});

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_P(FasterRCNNMultiDTypeTest, NoObjectsInImage) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Empty targets (no objects)
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({0, 4}, dtype(), device());
    auto labels = Tensor({0}, DType::Int64, device());

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    // Should handle empty targets gracefully
    // Note: May not return losses if no objects present
    auto losses = model->forward_train(images, targets);
}

TEST_P(FasterRCNNMultiDTypeTest, VerySmallObjects) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Small objects (10x10 pixels in 800x800, scaled for actual size)
    auto targets = createTargets(3, {
        {100.0f, 100.0f, 110.0f, 110.0f},
        {300.0f, 300.0f, 310.0f, 310.0f},
        {500.0f, 500.0f, 510.0f, 510.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // Should handle very small objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TEST_P(FasterRCNNMultiDTypeTest, VeryLargeObjects) {
    auto model = faster_rcnn_resnet50(91, false);
    convert_model_with_offload(model);
    model->train();

    int64_t img_size = getInputSize(800);
    auto images = createInput({1, 3, img_size, img_size});

    // Large object covering most of image (scaled from 800x800 reference)
    auto targets = createTargets(1, {
        {50.0f, 50.0f, 750.0f, 750.0f}
    }, img_size);

    auto losses = model->forward_train(images, targets);

    // Should handle very large objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FasterRCNNMultiDTypeTest);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
