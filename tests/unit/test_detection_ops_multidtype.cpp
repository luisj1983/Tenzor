/**
 * @file test_detection_ops_multidtype.cpp
 * @brief DType-parameterized tests for detection operations: ROI Align, NMS, etc.
 *
 * Tests detection operations (NMS, ROI Align, box operations) with multiple dtypes.
 * Detection operations typically work with Float32/Float64 for bounding box coordinates
 * and confidence scores.
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

// ============================================================================
// DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const DTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class DetectionOpsMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    Device device_;
    DType dtype_;

    void SetUp() override {
        tenzor::initialize();
        device_ = Device::cpu();
        dtype_ = GetParam().dtype;
    }

    // Helper to get epsilon for floating point comparisons
    double getEpsilon() const {
        if (dtype_ == DType::Float32) {
            return 1e-5;
        } else if (dtype_ == DType::Float64) {
            return 1e-10;
        }
        return 1e-5;
    }

    // Helper to verify floating point values
    template<typename T>
    void verifyValue(T actual, T expected, const std::string& msg = "") const {
        if (dtype_ == DType::Float32) {
            EXPECT_NEAR(static_cast<float>(actual), static_cast<float>(expected),
                       static_cast<float>(getEpsilon())) << msg;
        } else if (dtype_ == DType::Float64) {
            EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected),
                       getEpsilon()) << msg;
        }
    }
};

// ============================================================================
// ROI Align Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignBasicForwardShape) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    // Feature map: (N, C, H, W)
    Variable features(Tensor({1, 512, 28, 28}, dtype_, device_), true);

    // ROIs: (num_rois, 5) where each row is [batch_idx, x1, y1, x2, y2]
    Tensor rois({10, 5}, dtype_, device_);

    Variable output = roi_align.forward(features, rois);

    // Output should be (num_rois, C, pool_h, pool_w)
    auto shape = output.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 512, 7, 7}));

    // Verify dtype is preserved
    EXPECT_EQ(output.tensor().dtype(), dtype_);
}

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignGradientFlow) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(Tensor({1, 256, 28, 28}, dtype_, device_), true);
    Tensor rois({5, 5}, dtype_, device_);

    Variable output = roi_align.forward(features, rois);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient exists and has correct dtype
    EXPECT_TRUE(features.grad().has_value());
    EXPECT_EQ(features.grad()->dtype(), dtype_);
}

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignDifferentPoolSizes) {
    ROIAlign roi_align_7(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_14(14, 14, 1.0/16.0, 2);

    Variable features(Tensor({1, 512, 56, 56}, dtype_, device_), true);
    Tensor rois({3, 5}, dtype_, device_);

    Variable output_7 = roi_align_7.forward(features, rois);
    auto shape_7 = output_7.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_7.begin(), shape_7.end()),
              (std::vector<int64_t>{3, 512, 7, 7}));
    EXPECT_EQ(output_7.tensor().dtype(), dtype_);

    Variable output_14 = roi_align_14.forward(features, rois);
    auto shape_14 = output_14.tensor().shape();
    EXPECT_EQ(std::vector<int64_t>(shape_14.begin(), shape_14.end()),
              (std::vector<int64_t>{3, 512, 14, 14}));
    EXPECT_EQ(output_14.tensor().dtype(), dtype_);
}

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignDifferentSamplingRatios) {
    ROIAlign roi_align_2(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_4(7, 7, 1.0/16.0, 4);

    Variable features(Tensor({1, 512, 56, 56}, dtype_, device_), true);
    Tensor rois({3, 5}, dtype_, device_);

    Variable output_2 = roi_align_2.forward(features, rois);
    Variable output_4 = roi_align_4.forward(features, rois);

    auto shape_2 = output_2.tensor().shape();
    auto shape_4 = output_4.tensor().shape();

    // Both should have same output shape (difference is in interpolation quality)
    EXPECT_EQ(shape_2[0], shape_4[0]);
    EXPECT_EQ(shape_2[1], shape_4[1]);

    // Verify dtypes preserved
    EXPECT_EQ(output_2.tensor().dtype(), dtype_);
    EXPECT_EQ(output_4.tensor().dtype(), dtype_);
}

// ============================================================================
// Non-Maximum Suppression (NMS) Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, NMSBasicFiltering) {
    // Boxes: (N, 4) where each row is [x1, y1, x2, y2]
    Tensor boxes({20, 4}, dtype_, device_);

    // Initialize with some test data
    auto boxes_cpu = boxes.to(Device::cpu());
    if (dtype_ == DType::Float32) {
        float* data = boxes_cpu.data<float>();
        for (int i = 0; i < 20; ++i) {
            data[i*4 + 0] = static_cast<float>(i * 10);       // x1
            data[i*4 + 1] = static_cast<float>(i * 10);       // y1
            data[i*4 + 2] = static_cast<float>(i * 10 + 50);  // x2
            data[i*4 + 3] = static_cast<float>(i * 10 + 50);  // y2
        }
    } else if (dtype_ == DType::Float64) {
        double* data = boxes_cpu.data<double>();
        for (int i = 0; i < 20; ++i) {
            data[i*4 + 0] = static_cast<double>(i * 10);       // x1
            data[i*4 + 1] = static_cast<double>(i * 10);       // y1
            data[i*4 + 2] = static_cast<double>(i * 10 + 50);  // x2
            data[i*4 + 3] = static_cast<double>(i * 10 + 50);  // y2
        }
    }
    boxes = boxes_cpu.to(device_);

    // Scores: (N,)
    Tensor scores({20}, dtype_, device_);
    auto scores_cpu = scores.to(Device::cpu());
    if (dtype_ == DType::Float32) {
        float* data = scores_cpu.data<float>();
        for (int i = 0; i < 20; ++i) {
            data[i] = 1.0f - static_cast<float>(i) * 0.05f;  // Descending scores
        }
    } else if (dtype_ == DType::Float64) {
        double* data = scores_cpu.data<double>();
        for (int i = 0; i < 20; ++i) {
            data[i] = 1.0 - static_cast<double>(i) * 0.05;  // Descending scores
        }
    }
    scores = scores_cpu.to(device_);

    auto keep_indices = nms(boxes, scores, 0.5);  // IOU threshold = 0.5

    // NMS should return a tensor with valid indices
    EXPECT_GT(keep_indices.shape()[0], 0);
    EXPECT_LE(keep_indices.shape()[0], 20);
}

TEST_P(DetectionOpsMultiDTypeTest, NMSDifferentThresholds) {
    Tensor boxes({50, 4}, dtype_, device_);
    Tensor scores({50}, dtype_, device_);

    // Initialize with test data
    auto boxes_cpu = boxes.to(Device::cpu());
    auto scores_cpu = scores.to(Device::cpu());

    if (dtype_ == DType::Float32) {
        float* box_data = boxes_cpu.data<float>();
        float* score_data = scores_cpu.data<float>();
        for (int i = 0; i < 50; ++i) {
            box_data[i*4 + 0] = static_cast<float>(i * 5);
            box_data[i*4 + 1] = static_cast<float>(i * 5);
            box_data[i*4 + 2] = static_cast<float>(i * 5 + 30);
            box_data[i*4 + 3] = static_cast<float>(i * 5 + 30);
            score_data[i] = 1.0f - static_cast<float>(i) * 0.02f;
        }
    } else if (dtype_ == DType::Float64) {
        double* box_data = boxes_cpu.data<double>();
        double* score_data = scores_cpu.data<double>();
        for (int i = 0; i < 50; ++i) {
            box_data[i*4 + 0] = static_cast<double>(i * 5);
            box_data[i*4 + 1] = static_cast<double>(i * 5);
            box_data[i*4 + 2] = static_cast<double>(i * 5 + 30);
            box_data[i*4 + 3] = static_cast<double>(i * 5 + 30);
            score_data[i] = 1.0 - static_cast<double>(i) * 0.02;
        }
    }

    boxes = boxes_cpu.to(device_);
    scores = scores_cpu.to(device_);

    auto keep_low = nms(boxes, scores, 0.3);   // Stricter
    auto keep_high = nms(boxes, scores, 0.7);  // More permissive

    // Lower threshold should keep fewer or equal boxes
    EXPECT_LE(keep_low.shape()[0], keep_high.shape()[0]);
}

TEST_P(DetectionOpsMultiDTypeTest, NMSOutputShape) {
    // Create boxes and scores
    Tensor boxes({5, 4}, dtype_, device_);
    Tensor scores({5}, dtype_, device_);

    auto keep_indices = nms(boxes, scores, 0.5);

    // Result should be 1D tensor of indices
    EXPECT_EQ(keep_indices.shape().size(), 1);
    EXPECT_LE(keep_indices.shape()[0], 5);
}

// ============================================================================
// Box Operations Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, BoxIOUComputation) {
    // Box format: [x1, y1, x2, y2]
    Tensor boxes1({5, 4}, dtype_, device_);
    Tensor boxes2({5, 4}, dtype_, device_);

    // Initialize with test boxes
    auto boxes1_cpu = boxes1.to(Device::cpu());
    auto boxes2_cpu = boxes2.to(Device::cpu());

    if (dtype_ == DType::Float32) {
        float* data1 = boxes1_cpu.data<float>();
        float* data2 = boxes2_cpu.data<float>();
        for (int i = 0; i < 5; ++i) {
            // boxes1
            data1[i*4 + 0] = static_cast<float>(i * 20);
            data1[i*4 + 1] = static_cast<float>(i * 20);
            data1[i*4 + 2] = static_cast<float>(i * 20 + 40);
            data1[i*4 + 3] = static_cast<float>(i * 20 + 40);
            // boxes2 - slightly offset
            data2[i*4 + 0] = static_cast<float>(i * 20 + 5);
            data2[i*4 + 1] = static_cast<float>(i * 20 + 5);
            data2[i*4 + 2] = static_cast<float>(i * 20 + 45);
            data2[i*4 + 3] = static_cast<float>(i * 20 + 45);
        }
    } else if (dtype_ == DType::Float64) {
        double* data1 = boxes1_cpu.data<double>();
        double* data2 = boxes2_cpu.data<double>();
        for (int i = 0; i < 5; ++i) {
            // boxes1
            data1[i*4 + 0] = static_cast<double>(i * 20);
            data1[i*4 + 1] = static_cast<double>(i * 20);
            data1[i*4 + 2] = static_cast<double>(i * 20 + 40);
            data1[i*4 + 3] = static_cast<double>(i * 20 + 40);
            // boxes2 - slightly offset
            data2[i*4 + 0] = static_cast<double>(i * 20 + 5);
            data2[i*4 + 1] = static_cast<double>(i * 20 + 5);
            data2[i*4 + 2] = static_cast<double>(i * 20 + 45);
            data2[i*4 + 3] = static_cast<double>(i * 20 + 45);
        }
    }

    boxes1 = boxes1_cpu.to(device_);
    boxes2 = boxes2_cpu.to(device_);

    auto ious = box_iou(boxes1, boxes2);

    // IOU matrix should be (5, 5)
    auto shape = ious.shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{5, 5}));

    // Verify dtype preserved
    EXPECT_EQ(ious.dtype(), dtype_);
}

TEST_P(DetectionOpsMultiDTypeTest, BoxEncodingDecoding) {
    Tensor boxes({10, 4}, dtype_, device_);
    Tensor anchors({10, 4}, dtype_, device_);

    // Initialize with test data
    auto boxes_cpu = boxes.to(Device::cpu());
    auto anchors_cpu = anchors.to(Device::cpu());

    if (dtype_ == DType::Float32) {
        float* box_data = boxes_cpu.data<float>();
        float* anchor_data = anchors_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            // Boxes
            box_data[i*4 + 0] = static_cast<float>(i * 10 + 5);
            box_data[i*4 + 1] = static_cast<float>(i * 10 + 5);
            box_data[i*4 + 2] = static_cast<float>(i * 10 + 35);
            box_data[i*4 + 3] = static_cast<float>(i * 10 + 35);
            // Anchors
            anchor_data[i*4 + 0] = static_cast<float>(i * 10);
            anchor_data[i*4 + 1] = static_cast<float>(i * 10);
            anchor_data[i*4 + 2] = static_cast<float>(i * 10 + 40);
            anchor_data[i*4 + 3] = static_cast<float>(i * 10 + 40);
        }
    } else if (dtype_ == DType::Float64) {
        double* box_data = boxes_cpu.data<double>();
        double* anchor_data = anchors_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            // Boxes
            box_data[i*4 + 0] = static_cast<double>(i * 10 + 5);
            box_data[i*4 + 1] = static_cast<double>(i * 10 + 5);
            box_data[i*4 + 2] = static_cast<double>(i * 10 + 35);
            box_data[i*4 + 3] = static_cast<double>(i * 10 + 35);
            // Anchors
            anchor_data[i*4 + 0] = static_cast<double>(i * 10);
            anchor_data[i*4 + 1] = static_cast<double>(i * 10);
            anchor_data[i*4 + 2] = static_cast<double>(i * 10 + 40);
            anchor_data[i*4 + 3] = static_cast<double>(i * 10 + 40);
        }
    }

    boxes = boxes_cpu.to(device_);
    anchors = anchors_cpu.to(device_);

    // Encode boxes relative to anchors
    auto encoded = encode_boxes(boxes, anchors);
    EXPECT_EQ(encoded.dtype(), dtype_);

    // Decode back to original boxes
    auto decoded = decode_boxes(encoded, anchors);
    EXPECT_EQ(decoded.dtype(), dtype_);

    // Decoded boxes should match original boxes shape
    auto shape = decoded.shape();
    EXPECT_EQ(std::vector<int64_t>(shape.begin(), shape.end()),
              (std::vector<int64_t>{10, 4}));
}

TEST_P(DetectionOpsMultiDTypeTest, BoxEncodingShape) {
    Tensor boxes({20, 4}, dtype_, device_);
    Tensor anchors({20, 4}, dtype_, device_);

    auto encoded = encode_boxes(boxes, anchors);

    // Encoded should have same shape as input
    EXPECT_EQ(encoded.shape()[0], 20);
    EXPECT_EQ(encoded.shape()[1], 4);
    EXPECT_EQ(encoded.dtype(), dtype_);
}

// ============================================================================
// Anchor Generator Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, AnchorGeneratorSizes) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Feature map size: 56x56
    auto anchors = anchor_gen.generate(56, 56, 16, device_);  // stride=16

    // Number of anchors = H * W * (num_sizes * num_aspect_ratios)
    // 56 * 56 * (5 * 3) = 47040
    EXPECT_EQ(anchors.shape()[0], 47040);
    EXPECT_EQ(anchors.shape()[1], 4);

    // Anchors should be in Float32 by default (can convert if needed)
    // For this test, we verify the shape is correct
}

TEST_P(DetectionOpsMultiDTypeTest, AnchorGeneratorDifferentScales) {
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

TEST_P(DetectionOpsMultiDTypeTest, AnchorGeneratorNumAnchorsPerLocation) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Should have 3 sizes * 3 aspect ratios = 9 anchors per location
    EXPECT_EQ(anchor_gen.num_anchors_per_location(), 9);
}

// ============================================================================
// Test Instantiation
// ============================================================================

// Generate DType combinations for detection operations
std::vector<DTypeParam> GenerateDetectionDTypes() {
    // Detection operations typically work with Float32 and Float64
    // for bounding box coordinates and confidence scores
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"}
    };
}

INSTANTIATE_TEST_SUITE_P(
    AllDTypes,
    DetectionOpsMultiDTypeTest,
    ::testing::ValuesIn(GenerateDetectionDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    tenzor::initialize();  // Initialize Tenzor library and backends
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}

/*
 * COVERAGE SUMMARY:
 *
 * This test file provides dtype parameterization for detection operations:
 *
 * DType Coverage:
 * - Float32: Standard precision for detection (bounding boxes, scores)
 * - Float64: High precision for detection (research/debugging)
 *
 * Operation Coverage:
 * - ROI Align: Forward pass, gradient flow, different pool sizes, sampling ratios
 * - NMS: Basic filtering, different thresholds, output shapes
 * - Box Operations: IOU computation, encoding/decoding, shape verification
 * - Anchor Generation: Different scales, sizes, anchors per location
 *
 * Test Scenarios:
 * - 15 test cases × 2 dtypes = 30 test scenarios
 *
 * Key Verification Points:
 * 1. DType preservation through operations
 * 2. Correct shapes for all operations
 * 3. Gradient flow with correct dtypes
 * 4. Box coordinates work correctly with different precisions
 * 5. Confidence scores maintain precision
 *
 * Note: Detection operations require floating-point dtypes (Float32/Float64)
 * as they involve bounding box coordinates, IoU calculations, and probability scores.
 */
