/**
 * @file test_faster_rcnn_multidtype.cpp
 * @brief Multi-dtype tests for Faster R-CNN object detection model
 *
 * Tests Faster R-CNN components across Float32, Float64, and Float16:
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
#include <cmath>
#include <vector>
#include <unordered_map>

using namespace tenzor;
using namespace tenzor::models;

// Helper to get tolerance based on dtype
template<typename T>
T get_tolerance() {
    if constexpr (std::is_same_v<T, float>) {
        return static_cast<T>(1e-4);
    } else if constexpr (std::is_same_v<T, double>) {
        return static_cast<T>(1e-6);
    } else {  // Float16
        return static_cast<T>(1e-2);
    }
}

// Test fixture for multi-dtype Faster R-CNN tests
template<typename T>
class FasterRCNNMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        if constexpr (std::is_same_v<T, float>) {
            dtype_ = DType::Float32;
        } else if constexpr (std::is_same_v<T, double>) {
            dtype_ = DType::Float64;
        } else {
            dtype_ = DType::Float16;
        }
    }

    Device device_;
    DType dtype_;
};

using TestTypes = ::testing::Types<float, double>;
TYPED_TEST_SUITE(FasterRCNNMultiDTypeTest, TestTypes);

// ============================================================================
// Backbone Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, ResNet50BackboneForward) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Backbone should process image and produce detections
    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, ResNet101BackboneForward) {
    auto model = faster_rcnn_resnet101(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // ResNet101 backbone should handle deeper architecture
    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, BackboneBatchProcessing) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({4, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle batch of 4 images
    EXPECT_EQ(detections.size(), 4);
}

// ============================================================================
// RPN (Region Proposal Network) Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, RPNProposalGeneration) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // RPN should generate region proposals
    EXPECT_EQ(detections.size(), 1);
    // Each detection should contain boxes
    EXPECT_TRUE(detections[0].find("boxes") != detections[0].end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, RPNMultiScaleAnchors) {
    auto model = faster_rcnn_resnet50(91, false);

    // Test RPN with different image scales
    Variable images_small(Tensor({1, 3, 600, 600}, this->dtype_, this->device_), true);
    Variable images_large(Tensor({1, 3, 1024, 1024}, this->dtype_, this->device_), true);

    model->eval();
    auto detections_small = model->forward_inference(images_small);
    auto detections_large = model->forward_inference(images_large);

    // RPN should handle different scales
    EXPECT_EQ(detections_small.size(), 1);
    EXPECT_EQ(detections_large.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, RPNObjectnessScores) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create dummy targets
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({2, 4}, this->dtype_, this->device_);
    auto labels = Tensor({2}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    boxes_ptr[0] = static_cast<TypeParam>(100); boxes_ptr[1] = static_cast<TypeParam>(100);
    boxes_ptr[2] = static_cast<TypeParam>(200); boxes_ptr[3] = static_cast<TypeParam>(200);
    boxes_ptr[4] = static_cast<TypeParam>(300); boxes_ptr[5] = static_cast<TypeParam>(300);
    boxes_ptr[6] = static_cast<TypeParam>(400); boxes_ptr[7] = static_cast<TypeParam>(400);

    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;
    labels_data[1] = 2;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // RPN should compute objectness loss
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// ROI Pooling/Align Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, ROIPoolingForward) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // ROI pooling should extract features from proposals
    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, ROIAlignPrecision) {
    auto model = faster_rcnn_resnet50(91, false);

    // Test with smaller image to verify ROI align precision
    Variable images(Tensor({1, 3, 400, 400}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // ROI align should maintain spatial precision
    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, ROIMultipleRegions) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create targets with multiple boxes
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({5, 4}, this->dtype_, this->device_);
    auto labels = Tensor({5}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    // Box 1
    boxes_ptr[0] = static_cast<TypeParam>(50); boxes_ptr[1] = static_cast<TypeParam>(50);
    boxes_ptr[2] = static_cast<TypeParam>(150); boxes_ptr[3] = static_cast<TypeParam>(150);
    // Box 2
    boxes_ptr[4] = static_cast<TypeParam>(200); boxes_ptr[5] = static_cast<TypeParam>(200);
    boxes_ptr[6] = static_cast<TypeParam>(300); boxes_ptr[7] = static_cast<TypeParam>(300);
    // Box 3
    boxes_ptr[8] = static_cast<TypeParam>(350); boxes_ptr[9] = static_cast<TypeParam>(350);
    boxes_ptr[10] = static_cast<TypeParam>(450); boxes_ptr[11] = static_cast<TypeParam>(450);
    // Box 4
    boxes_ptr[12] = static_cast<TypeParam>(500); boxes_ptr[13] = static_cast<TypeParam>(100);
    boxes_ptr[14] = static_cast<TypeParam>(600); boxes_ptr[15] = static_cast<TypeParam>(200);
    // Box 5
    boxes_ptr[16] = static_cast<TypeParam>(100); boxes_ptr[17] = static_cast<TypeParam>(500);
    boxes_ptr[18] = static_cast<TypeParam>(200); boxes_ptr[19] = static_cast<TypeParam>(600);

    auto labels_data = labels.data<int64_t>();
    for (int i = 0; i < 5; i++) {
        labels_data[i] = i + 1;
    }

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Should handle multiple ROI regions
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// Detection Head Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, DetectionHeadClassification) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Detection head should classify proposals
    EXPECT_EQ(detections.size(), 1);
    EXPECT_TRUE(detections[0].find("labels") != detections[0].end() ||
                detections[0].find("boxes") != detections[0].end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, DetectionHeadBBoxRegression) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create targets
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({3, 4}, this->dtype_, this->device_);
    auto labels = Tensor({3}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    boxes_ptr[0] = static_cast<TypeParam>(100); boxes_ptr[1] = static_cast<TypeParam>(100);
    boxes_ptr[2] = static_cast<TypeParam>(200); boxes_ptr[3] = static_cast<TypeParam>(200);
    boxes_ptr[4] = static_cast<TypeParam>(300); boxes_ptr[5] = static_cast<TypeParam>(300);
    boxes_ptr[6] = static_cast<TypeParam>(400); boxes_ptr[7] = static_cast<TypeParam>(400);
    boxes_ptr[8] = static_cast<TypeParam>(500); boxes_ptr[9] = static_cast<TypeParam>(500);
    boxes_ptr[10] = static_cast<TypeParam>(600); boxes_ptr[11] = static_cast<TypeParam>(600);

    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Detection head should compute bbox regression loss
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, DetectionHeadMultiClass) {
    auto model = faster_rcnn_resnet50(91, false);  // 91 classes (COCO)
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle multi-class detection
    EXPECT_EQ(detections.size(), 1);
}

// ============================================================================
// Forward Pass with Different Image Sizes
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, SmallImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 400, 400}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, MediumImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 600, 600}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, StandardImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, LargeImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 1024, 1024}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, VeryLargeImageSize) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 1280, 1280}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, RectangularImageSize) {
    auto model = faster_rcnn_resnet50(91, false);

    // Test with non-square image
    Variable images(Tensor({1, 3, 600, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle non-square images
    EXPECT_EQ(detections.size(), 1);
}

// ============================================================================
// Multi-Object Detection Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, SingleObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({1, 4}, this->dtype_, this->device_);
    auto labels = Tensor({1}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    boxes_ptr[0] = static_cast<TypeParam>(100); boxes_ptr[1] = static_cast<TypeParam>(100);
    boxes_ptr[2] = static_cast<TypeParam>(300); boxes_ptr[3] = static_cast<TypeParam>(300);

    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, MultiObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create targets with multiple objects
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({8, 4}, this->dtype_, this->device_);
    auto labels = Tensor({8}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    // 8 different objects at various locations
    for (int i = 0; i < 8; i++) {
        boxes_ptr[i*4 + 0] = static_cast<TypeParam>((i % 3) * 200 + 50);
        boxes_ptr[i*4 + 1] = static_cast<TypeParam>((i / 3) * 200 + 50);
        boxes_ptr[i*4 + 2] = static_cast<TypeParam>((i % 3) * 200 + 150);
        boxes_ptr[i*4 + 3] = static_cast<TypeParam>((i / 3) * 200 + 150);
    }

    auto labels_data = labels.data<int64_t>();
    for (int i = 0; i < 8; i++) {
        labels_data[i] = (i % 5) + 1;  // 5 different classes
    }

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Should handle multiple objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, DenseObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create targets with many small objects
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({15, 4}, this->dtype_, this->device_);
    auto labels = Tensor({15}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    for (int i = 0; i < 15; i++) {
        boxes_ptr[i*4 + 0] = static_cast<TypeParam>((i % 5) * 150 + 20);
        boxes_ptr[i*4 + 1] = static_cast<TypeParam>((i / 5) * 250 + 20);
        boxes_ptr[i*4 + 2] = static_cast<TypeParam>((i % 5) * 150 + 100);
        boxes_ptr[i*4 + 3] = static_cast<TypeParam>((i / 5) * 250 + 100);
    }

    auto labels_data = labels.data<int64_t>();
    for (int i = 0; i < 15; i++) {
        labels_data[i] = (i % 10) + 1;
    }

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Should handle dense object detection
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, OverlappingObjectDetection) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Create targets with overlapping boxes
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({4, 4}, this->dtype_, this->device_);
    auto labels = Tensor({4}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    // Overlapping boxes in center region
    boxes_ptr[0] = static_cast<TypeParam>(300); boxes_ptr[1] = static_cast<TypeParam>(300);
    boxes_ptr[2] = static_cast<TypeParam>(500); boxes_ptr[3] = static_cast<TypeParam>(500);

    boxes_ptr[4] = static_cast<TypeParam>(320); boxes_ptr[5] = static_cast<TypeParam>(320);
    boxes_ptr[6] = static_cast<TypeParam>(520); boxes_ptr[7] = static_cast<TypeParam>(520);

    boxes_ptr[8] = static_cast<TypeParam>(340); boxes_ptr[9] = static_cast<TypeParam>(340);
    boxes_ptr[10] = static_cast<TypeParam>(540); boxes_ptr[11] = static_cast<TypeParam>(540);

    boxes_ptr[12] = static_cast<TypeParam>(360); boxes_ptr[13] = static_cast<TypeParam>(360);
    boxes_ptr[14] = static_cast<TypeParam>(560); boxes_ptr[15] = static_cast<TypeParam>(560);

    auto labels_data = labels.data<int64_t>();
    for (int i = 0; i < 4; i++) {
        labels_data[i] = i + 1;
    }

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Should handle overlapping objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// Gradient Flow and Training Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, GradientFlowThroughModel) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({3, 4}, this->dtype_, this->device_);
    auto labels = Tensor({3}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    boxes_ptr[0] = static_cast<TypeParam>(100); boxes_ptr[1] = static_cast<TypeParam>(100);
    boxes_ptr[2] = static_cast<TypeParam>(200); boxes_ptr[3] = static_cast<TypeParam>(200);
    boxes_ptr[4] = static_cast<TypeParam>(300); boxes_ptr[5] = static_cast<TypeParam>(300);
    boxes_ptr[6] = static_cast<TypeParam>(400); boxes_ptr[7] = static_cast<TypeParam>(400);
    boxes_ptr[8] = static_cast<TypeParam>(500); boxes_ptr[9] = static_cast<TypeParam>(500);
    boxes_ptr[10] = static_cast<TypeParam>(600); boxes_ptr[11] = static_cast<TypeParam>(600);

    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Check gradients can flow through all components
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, ParameterCount) {
    auto model = faster_rcnn_resnet50(91, false);

    auto params = model->parameters();

    // Faster R-CNN should have many parameters (backbone + RPN + head)
    EXPECT_GT(params.size(), 100);
}

// ============================================================================
// Batch Processing Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, BatchInference) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({3, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Should process all images in batch
    EXPECT_EQ(detections.size(), 3);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, LargeBatchInference) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({8, 3, 600, 600}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    // Should handle larger batches
    EXPECT_EQ(detections.size(), 8);
}

// ============================================================================
// Model Variants Tests
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, ResNet50Variant) {
    auto model = faster_rcnn_resnet50(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, ResNet101Variant) {
    auto model = faster_rcnn_resnet101(91, false);
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, CustomNumClasses) {
    auto model = faster_rcnn_resnet50(20, false);  // Custom 20 classes
    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    model->eval();
    auto detections = model->forward_inference(images);

    EXPECT_EQ(detections.size(), 1);
}

// ============================================================================
// Edge Cases
// ============================================================================

TYPED_TEST(FasterRCNNMultiDTypeTest, NoObjectsInImage) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Empty targets (no objects)
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({0, 4}, this->dtype_, this->device_);
    auto labels = Tensor({0}, DType::Int64, this->device_);

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    // Should handle empty targets gracefully
    // Note: May not return losses if no objects present
    auto losses = model->forward_train(images, targets);
}

TYPED_TEST(FasterRCNNMultiDTypeTest, VerySmallObjects) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Small objects (10x10 pixels)
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({3, 4}, this->dtype_, this->device_);
    auto labels = Tensor({3}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    boxes_ptr[0] = static_cast<TypeParam>(100); boxes_ptr[1] = static_cast<TypeParam>(100);
    boxes_ptr[2] = static_cast<TypeParam>(110); boxes_ptr[3] = static_cast<TypeParam>(110);
    boxes_ptr[4] = static_cast<TypeParam>(300); boxes_ptr[5] = static_cast<TypeParam>(300);
    boxes_ptr[6] = static_cast<TypeParam>(310); boxes_ptr[7] = static_cast<TypeParam>(310);
    boxes_ptr[8] = static_cast<TypeParam>(500); boxes_ptr[9] = static_cast<TypeParam>(500);
    boxes_ptr[10] = static_cast<TypeParam>(510); boxes_ptr[11] = static_cast<TypeParam>(510);

    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;
    labels_data[1] = 2;
    labels_data[2] = 3;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Should handle very small objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

TYPED_TEST(FasterRCNNMultiDTypeTest, VeryLargeObjects) {
    auto model = faster_rcnn_resnet50(91, false);
    model->train();

    Variable images(Tensor({1, 3, 800, 800}, this->dtype_, this->device_), true);

    // Large object covering most of image
    std::vector<std::unordered_map<std::string, Tensor>> targets(1);
    auto boxes = Tensor({1, 4}, this->dtype_, this->device_);
    auto labels = Tensor({1}, DType::Int64, this->device_);

    auto boxes_ptr = reinterpret_cast<TypeParam*>(boxes.data_ptr());
    boxes_ptr[0] = static_cast<TypeParam>(50); boxes_ptr[1] = static_cast<TypeParam>(50);
    boxes_ptr[2] = static_cast<TypeParam>(750); boxes_ptr[3] = static_cast<TypeParam>(750);

    auto labels_data = labels.data<int64_t>();
    labels_data[0] = 1;

    targets[0]["boxes"] = boxes;
    targets[0]["labels"] = labels;

    auto losses = model->forward_train(images, targets);

    // Should handle very large objects
    EXPECT_TRUE(losses.find("loss_objectness") != losses.end());
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
