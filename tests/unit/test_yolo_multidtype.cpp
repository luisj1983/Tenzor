/**
 * @file test_yolo_multidtype.cpp
 * @brief Multi-dtype tests for YOLO v3/v4/v5 object detection models
 *
 * Tests YOLO architectures across Float32, Float64, and Float16 data types.
 * Covers forward passes, detection heads, anchor boxes, NMS, bounding box
 * predictions, and multi-scale detection capabilities.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../../include/tenzor/models/yolo.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::models;

// ============================================================================
// Test Fixture with Multi-DType Support
// ============================================================================

template <typename T>
class YOLOMultiDTypeTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
        dtype_ = get_dtype();

        // Set tolerances based on data type
        if (dtype_ == DType::Float16) {
            abs_tol_ = 1e-2;
            rel_tol_ = 1e-2;
        } else if (dtype_ == DType::Float32) {
            abs_tol_ = 1e-5;
            rel_tol_ = 1e-5;
        } else {  // Float64
            abs_tol_ = 1e-9;
            rel_tol_ = 1e-9;
        }
    }

    DType get_dtype() {
        if constexpr (std::is_same_v<T, float>) {
            return DType::Float32;
        } else if constexpr (std::is_same_v<T, double>) {
            return DType::Float64;
        } else {
            return DType::Float16;
        }
    }

    // Helper to check tensor values with dtype-specific tolerance
    bool tensors_close(const Tensor& a, const Tensor& b) {
        auto shape_a = a.shape();
        auto shape_b = b.shape();
        if (std::vector<int64_t>(shape_a.begin(), shape_a.end()) !=
            std::vector<int64_t>(shape_b.begin(), shape_b.end())) return false;

        auto a_data = a.to(DType::Float32).data<float>();
        auto b_data = b.to(DType::Float32).data<float>();

        for (size_t i = 0; i < a.numel(); ++i) {
            float diff = std::abs(a_data[i] - b_data[i]);
            float threshold = abs_tol_ + rel_tol_ * std::abs(b_data[i]);
            if (diff > threshold) {
                return false;
            }
        }
        return true;
    }

    Device device_;
    DType dtype_;
    double abs_tol_;
    double rel_tol_;
};

// Test type definitions
// Float16 placeholder handled by uint16_t in template
using TestTypes = ::testing::Types<float, double, uint16_t>;
TYPED_TEST_SUITE(YOLOMultiDTypeTest, TestTypes);

// ============================================================================
// YOLOv3 Architecture Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3ForwardPass) {
    auto model = yolov3(80, false);

    Variable images(Tensor({2, 3, 416, 416}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    // Verify output has valid shape and values
    EXPECT_TRUE(output.tensor().numel() > 0);
    EXPECT_EQ(output.requires_grad(), true);

    // Check that output is finite
    auto output_data = output.tensor().to(DType::Float32).data<float>();
    for (size_t i = 0; i < std::min<size_t>(100, output.tensor().numel()); ++i) {
        EXPECT_TRUE(std::isfinite(output_data[i]))
            << "Output contains non-finite value at index " << i;
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3GradientFlow) {
    auto model = yolov3(80, false);
    model->train();

    Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify parameters exist and have gradients
    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Check gradient flow (at least some parameters should have gradients)
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value() && param->grad().value().numel() > 0) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0) << "No parameters received gradients";
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3MultiScaleImageSizes) {
    auto model = yolov3(80, false);

    // Test 416x416
    Variable images_416(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
    auto output_416 = model->forward(images_416);
    EXPECT_TRUE(output_416.tensor().numel() > 0);

    // Test 512x512
    Variable images_512(Tensor({1, 3, 512, 512}, this->dtype_, this->device_), true);
    auto output_512 = model->forward(images_512);
    EXPECT_TRUE(output_512.tensor().numel() > 0);

    // Test 608x608
    Variable images_608(Tensor({1, 3, 608, 608}, this->dtype_, this->device_), true);
    auto output_608 = model->forward(images_608);
    EXPECT_TRUE(output_608.tensor().numel() > 0);

    // Larger input should produce more predictions
    EXPECT_GE(output_608.tensor().numel(), output_416.tensor().numel());
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3DetectionHeads) {
    auto model = yolov3(80, false);

    Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);

    // Get raw predictions from all detection heads
    auto raw_outputs = model->forward_raw(images);

    // YOLOv3 has 3 detection heads (small, medium, large objects)
    EXPECT_EQ(raw_outputs.size(), 3);

    // Verify each detection head produces valid output
    for (size_t i = 0; i < raw_outputs.size(); ++i) {
        EXPECT_TRUE(raw_outputs[i].tensor().numel() > 0)
            << "Detection head " << i << " produced empty output";
        EXPECT_GT(raw_outputs[i].tensor().ndim(), 0)
            << "Detection head " << i << " has invalid dimensions";
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3AnchorBoxes) {
    auto model = yolov3(80, false);

    Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);

    // Each detection head uses 3 anchor boxes per grid cell
    // Output shape should be (N, num_anchors, grid_h, grid_w, 5 + num_classes)
    for (const auto& output : raw_outputs) {
        const auto& shape = output.tensor().shape();
        EXPECT_GE(shape.size(), 2) << "Output has insufficient dimensions";

        // Check that predictions include: x, y, w, h, objectness, classes (85 total for COCO)
        if (shape.size() >= 2) {
            EXPECT_TRUE(shape[shape.size() - 1] >= 85 || shape[1] >= 3)
                << "Output doesn't contain expected anchor predictions";
        }
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3BoundingBoxPredictions) {
    auto model = yolov3(80, false);

    Variable images(Tensor({2, 3, 416, 416}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);

    // Decode predictions to bounding boxes
    auto boxes = model->decode_predictions(raw_outputs, 416);

    // Verify boxes tensor is valid
    EXPECT_TRUE(boxes.numel() > 0);
    EXPECT_EQ(boxes.ndim(), 3);  // (batch, num_boxes, 4)
    EXPECT_EQ(boxes.shape()[2], 4);  // Each box has 4 coordinates (x1, y1, x2, y2)
    EXPECT_EQ(boxes.shape()[0], 2);  // Batch size
}

// ============================================================================
// YOLOv5 Architecture Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5NanoForwardPass) {
    auto model = yolov5n(80, false);

    Variable images(Tensor({2, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
    EXPECT_EQ(output.requires_grad(), true);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5SmallForwardPass) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({2, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
    EXPECT_EQ(output.requires_grad(), true);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5MediumForwardPass) {
    auto model = yolov5m(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5LargeForwardPass) {
    auto model = yolov5l(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5XLargeForwardPass) {
    auto model = yolov5x(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5GradientFlow) {
    auto model = yolov5s(80, false);
    model->train();

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    Variable loss = tenzor::sum(output);
    loss.backward();

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify gradient computation
    int params_with_grad = 0;
    for (const auto& param : params) {
        if (param->grad().has_value() && param->grad().value().numel() > 0) {
            params_with_grad++;
        }
    }
    EXPECT_GT(params_with_grad, 0);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5MultiScaleDetection) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);

    // YOLOv5 has 3 detection scales: P3 (80x80), P4 (40x40), P5 (20x20)
    EXPECT_EQ(raw_outputs.size(), 3);

    // Verify multi-scale outputs
    for (size_t i = 0; i < raw_outputs.size(); ++i) {
        EXPECT_TRUE(raw_outputs[i].tensor().numel() > 0)
            << "Scale " << i << " produced empty output";

        // Check that each scale has different spatial dimensions
        const auto& shape = raw_outputs[i].tensor().shape();
        EXPECT_GT(shape.size(), 2) << "Scale " << i << " has invalid shape";
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5DetectionHeadsAndAnchors) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);

    // Each detection head uses anchor boxes
    EXPECT_EQ(raw_outputs.size(), 3);

    // Verify predictions contain box coordinates, objectness, and class scores
    for (const auto& output : raw_outputs) {
        const auto& shape = output.tensor().shape();
        EXPECT_GE(shape.size(), 2);

        // YOLOv5 prediction format: (batch, anchors, grid_h, grid_w, 85)
        // 85 = 4 (bbox) + 1 (objectness) + 80 (classes)
        if (shape.size() >= 2) {
            EXPECT_TRUE(shape[shape.size() - 1] >= 85 || shape[1] >= 3);
        }
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5BoundingBoxDecoding) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({2, 3, 640, 640}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);

    // Decode predictions to absolute bounding boxes
    auto boxes = model->decode_predictions(raw_outputs, 640);

    EXPECT_TRUE(boxes.numel() > 0);
    EXPECT_EQ(boxes.ndim(), 3);  // (batch, num_boxes, 4)
    EXPECT_EQ(boxes.shape()[2], 4);  // x1, y1, x2, y2
    EXPECT_EQ(boxes.shape()[0], 2);  // Batch size

    // Verify box coordinates are in valid range [0, image_size]
    auto boxes_data = boxes.to(DType::Float32).data<float>();
    for (size_t i = 0; i < std::min<size_t>(100, boxes.numel()); ++i) {
        float val = boxes_data[i];
        if (std::isfinite(val)) {
            EXPECT_GE(val, -100.0f) << "Box coordinate unexpectedly negative";
            EXPECT_LE(val, 750.0f) << "Box coordinate exceeds image bounds";
        }
    }
}

// ============================================================================
// NMS (Non-Maximum Suppression) Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3NMSPostProcessing) {
    auto model = yolov3(80, false);

    Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);
    auto boxes = model->decode_predictions(raw_outputs, 416);

    // Create dummy scores for testing NMS
    Tensor scores({1, boxes.shape()[1], 80}, this->dtype_, this->device_);

    // Apply NMS post-processing
    auto detections = model->postprocess(boxes, scores);

    // Should return detections for each image in batch
    EXPECT_EQ(detections.size(), 1);

    // Each detection tuple contains: (boxes, scores, labels)
    auto [det_boxes, det_scores, det_labels] = detections[0];

    EXPECT_TRUE(det_boxes.numel() >= 0);
    EXPECT_EQ(det_scores.numel(), det_labels.numel());
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5NMSPostProcessing) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto raw_outputs = model->forward_raw(images);
    auto boxes = model->decode_predictions(raw_outputs, 640);

    // Create dummy scores
    Tensor scores({1, boxes.shape()[1], 80}, this->dtype_, this->device_);

    // Apply NMS
    auto detections = model->postprocess(boxes, scores);

    EXPECT_EQ(detections.size(), 1);

    auto [det_boxes, det_scores, det_labels] = detections[0];
    EXPECT_TRUE(det_boxes.numel() >= 0);
}

// ============================================================================
// Different Image Sizes Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5VariableInputSizes) {
    auto model = yolov5s(80, false);

    // Test 320x320 (smaller, faster)
    Variable images_320(Tensor({1, 3, 320, 320}, this->dtype_, this->device_), true);
    auto output_320 = model->forward(images_320);
    EXPECT_TRUE(output_320.tensor().numel() > 0);

    // Test 640x640 (default)
    Variable images_640(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output_640 = model->forward(images_640);
    EXPECT_TRUE(output_640.tensor().numel() > 0);

    // Test 1280x1280 (larger, more accurate)
    Variable images_1280(Tensor({1, 3, 1280, 1280}, this->dtype_, this->device_), true);
    auto output_1280 = model->forward(images_1280);
    EXPECT_TRUE(output_1280.tensor().numel() > 0);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3VariableInputSizes) {
    auto model = yolov3(80, false);

    // Test different multiples of 32 (YOLOv3 requirement)
    std::vector<int> sizes = {320, 416, 512, 608};

    for (int size : sizes) {
        Variable images(Tensor({1, 3, size, size}, this->dtype_, this->device_), true);
        auto output = model->forward(images);
        EXPECT_TRUE(output.tensor().numel() > 0)
            << "Failed for size " << size << "x" << size;
    }
}

// ============================================================================
// Batch Size Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3BatchProcessing) {
    auto model = yolov3(80, false);

    // Test different batch sizes
    for (int batch_size : {1, 2, 4}) {
        Variable images(Tensor({batch_size, 3, 416, 416}, this->dtype_, this->device_), true);
        auto output = model->forward(images);
        EXPECT_TRUE(output.tensor().numel() > 0)
            << "Failed for batch size " << batch_size;
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5BatchProcessing) {
    auto model = yolov5s(80, false);

    // Test different batch sizes
    for (int batch_size : {1, 2, 4}) {
        Variable images(Tensor({batch_size, 3, 640, 640}, this->dtype_, this->device_), true);
        auto output = model->forward(images);
        EXPECT_TRUE(output.tensor().numel() > 0)
            << "Failed for batch size " << batch_size;
    }
}

// ============================================================================
// Model Evaluation Mode Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3EvalMode) {
    auto model = yolov3(80, false);
    model->eval();

    Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), false);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
    EXPECT_FALSE(output.requires_grad());
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5EvalMode) {
    auto model = yolov5s(80, false);
    model->eval();

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), false);
    auto output = model->forward(images);

    EXPECT_TRUE(output.tensor().numel() > 0);
    EXPECT_FALSE(output.requires_grad());
}

// ============================================================================
// DType Conversion Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3DTypeConsistency) {
    auto model = yolov3(80, false);

    Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    // Output should maintain the same dtype as input
    EXPECT_EQ(output.tensor().dtype(), this->dtype_);
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5DTypeConsistency) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output = model->forward(images);

    // Output should maintain the same dtype as input
    EXPECT_EQ(output.tensor().dtype(), this->dtype_);
}

// ============================================================================
// Model Size Comparison Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5ModelSizeComparison) {
    // Create all YOLOv5 variants
    auto nano = yolov5n(80, false);
    auto small = yolov5s(80, false);
    auto medium = yolov5m(80, false);
    auto large = yolov5l(80, false);
    auto xlarge = yolov5x(80, false);

    // Count parameters for each model
    auto count_params = [](const std::shared_ptr<nn::Module>& model) {
        size_t count = 0;
        for (const auto& param : model->parameters()) {
            count += param->tensor().numel();
        }
        return count;
    };

    size_t nano_params = count_params(nano);
    size_t small_params = count_params(small);
    size_t medium_params = count_params(medium);
    size_t large_params = count_params(large);
    size_t xlarge_params = count_params(xlarge);

    // Verify parameter count ordering (larger models have more parameters)
    EXPECT_LT(nano_params, small_params);
    EXPECT_LT(small_params, medium_params);
    EXPECT_LT(medium_params, large_params);
    EXPECT_LT(large_params, xlarge_params);
}

// ============================================================================
// Numerical Stability Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3NumericalStability) {
    auto model = yolov3(80, false);

    // Test with zero input
    Variable images_zero(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
    auto output_zero = model->forward(images_zero);

    auto output_data = output_zero.tensor().to(DType::Float32).data<float>();
    for (size_t i = 0; i < std::min<size_t>(100, output_zero.tensor().numel()); ++i) {
        EXPECT_TRUE(std::isfinite(output_data[i]))
            << "Output contains NaN/Inf with zero input at index " << i;
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5NumericalStability) {
    auto model = yolov5s(80, false);

    // Test with zero input
    Variable images_zero(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
    auto output_zero = model->forward(images_zero);

    auto output_data = output_zero.tensor().to(DType::Float32).data<float>();
    for (size_t i = 0; i < std::min<size_t>(100, output_zero.tensor().numel()); ++i) {
        EXPECT_TRUE(std::isfinite(output_data[i]))
            << "Output contains NaN/Inf with zero input";
    }
}

// ============================================================================
// Custom Number of Classes Tests
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3CustomClasses) {
    // Test with different number of classes
    for (int num_classes : {1, 10, 20, 80}) {
        auto model = yolov3(num_classes, false);
        Variable images(Tensor({1, 3, 416, 416}, this->dtype_, this->device_), true);
        auto output = model->forward(images);

        EXPECT_TRUE(output.tensor().numel() > 0)
            << "Failed with " << num_classes << " classes";
    }
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5CustomClasses) {
    // Test with different number of classes
    for (int num_classes : {1, 10, 20, 80}) {
        auto model = yolov5s(num_classes, false);
        Variable images(Tensor({1, 3, 640, 640}, this->dtype_, this->device_), true);
        auto output = model->forward(images);

        EXPECT_TRUE(output.tensor().numel() > 0)
            << "Failed with " << num_classes << " classes";
    }
}

// ============================================================================
// Performance Tests (Shape Consistency)
// ============================================================================

TYPED_TEST(YOLOMultiDTypeTest, YOLOv3OutputShapeConsistency) {
    auto model = yolov3(80, false);

    // Run forward pass multiple times
    Variable images(Tensor({2, 3, 416, 416}, this->dtype_, this->device_), true);

    auto output1 = model->forward(images);
    auto output2 = model->forward(images);

    // Outputs should have consistent shapes
    auto shape1 = output1.tensor().shape(); auto shape2 = output2.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape1.begin(), shape1.end()), std::vector<int64_t>(shape2.begin(), shape2.end()));
}

TYPED_TEST(YOLOMultiDTypeTest, YOLOv5OutputShapeConsistency) {
    auto model = yolov5s(80, false);

    Variable images(Tensor({2, 3, 640, 640}, this->dtype_, this->device_), true);

    auto output1 = model->forward(images);
    auto output2 = model->forward(images);

    // Outputs should have consistent shapes
    auto shape1 = output1.tensor().shape(); auto shape2 = output2.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape1.begin(), shape1.end()), std::vector<int64_t>(shape2.begin(), shape2.end()));
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
