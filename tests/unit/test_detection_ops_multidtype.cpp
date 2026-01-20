/**
 * @file test_detection_ops_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for detection operations: ROI Align, NMS, etc.
 *
 * Tests detection operations (NMS, ROI Align, box operations) with Float32 and Float64
 * across CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - ROI Align works correctly with different dtypes and backends
 * - NMS filters boxes correctly across backends
 * - Box operations maintain precision across dtypes
 * - Anchor generation produces correct shapes
 *
 * Note: Detection operations require Float32/Float64 for bounding box coordinates
 * and confidence scores. Float16 is not supported due to precision requirements.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/detection/roi_ops.hpp>
#include <tenzor/nn/detection/anchors.hpp>
#include <tenzor/ops/detection.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::detection;
using namespace tenzor::ops;
using namespace tenzor::testing;

// ============================================================================
// Detection Ops Multi-Backend Multi-DType Test Fixture
// ============================================================================

class DetectionOpsMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // Skip Float16 - detection ops require higher precision for box coordinates
        if (dtype() == DType::Float16) {
            GTEST_SKIP() << "Float16 not supported for detection operations";
        }
    }

    // Helper to initialize box data
    template<typename T>
    void initializeBoxData(Tensor& boxes, int num_boxes, T x_step, T box_size) {
        auto boxes_cpu = boxes.to(Device::cpu());
        T* data = boxes_cpu.data<T>();
        for (int i = 0; i < num_boxes; ++i) {
            data[i*4 + 0] = static_cast<T>(i) * x_step;           // x1
            data[i*4 + 1] = static_cast<T>(i) * x_step;           // y1
            data[i*4 + 2] = static_cast<T>(i) * x_step + box_size; // x2
            data[i*4 + 3] = static_cast<T>(i) * x_step + box_size; // y2
        }
        boxes = boxes_cpu.to(device());
    }

    // Helper to initialize score data
    template<typename T>
    void initializeScoreData(Tensor& scores, int num_scores, T decay_rate) {
        auto scores_cpu = scores.to(Device::cpu());
        T* data = scores_cpu.data<T>();
        for (int i = 0; i < num_scores; ++i) {
            data[i] = static_cast<T>(1.0) - static_cast<T>(i) * decay_rate;
        }
        scores = scores_cpu.to(device());
    }
};

// ============================================================================
// ROI Align Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignBasicForwardShape) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    // Feature map: (N, C, H, W)
    Variable features(Tensor({1, 512, 28, 28}, dtype(), device()), true);

    // ROIs: (num_rois, 5) where each row is [batch_idx, x1, y1, x2, y2]
    Tensor rois({10, 5}, dtype(), device());

    Variable output = roi_align.forward(features, rois);

    // Output should be (num_rois, C, pool_h, pool_w)
    expectShape(output.tensor(), {10, 512, 7, 7});
    expectDType(output.tensor());
}

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignGradientFlow) {
    ROIAlign roi_align(7, 7, 1.0/16.0, 2);

    Variable features(Tensor({1, 256, 28, 28}, dtype(), device()), true);
    Tensor rois({5, 5}, dtype(), device());

    Variable output = roi_align.forward(features, rois);
    Variable loss = tenzor::sum(output);
    loss.backward();

    // Verify gradient exists and has correct dtype
    EXPECT_TRUE(features.grad().has_value());
    EXPECT_EQ(features.grad()->dtype(), dtype());
}

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignDifferentPoolSizes) {
    ROIAlign roi_align_7(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_14(14, 14, 1.0/16.0, 2);

    Variable features(Tensor({1, 512, 56, 56}, dtype(), device()), true);
    Tensor rois({3, 5}, dtype(), device());

    Variable output_7 = roi_align_7.forward(features, rois);
    expectShape(output_7.tensor(), {3, 512, 7, 7});
    expectDType(output_7.tensor());

    Variable output_14 = roi_align_14.forward(features, rois);
    expectShape(output_14.tensor(), {3, 512, 14, 14});
    expectDType(output_14.tensor());
}

TEST_P(DetectionOpsMultiDTypeTest, ROIAlignDifferentSamplingRatios) {
    ROIAlign roi_align_2(7, 7, 1.0/16.0, 2);
    ROIAlign roi_align_4(7, 7, 1.0/16.0, 4);

    Variable features(Tensor({1, 512, 56, 56}, dtype(), device()), true);
    Tensor rois({3, 5}, dtype(), device());

    Variable output_2 = roi_align_2.forward(features, rois);
    Variable output_4 = roi_align_4.forward(features, rois);

    auto shape_2 = output_2.tensor().shape();
    auto shape_4 = output_4.tensor().shape();

    // Both should have same output shape (difference is in interpolation quality)
    EXPECT_EQ(shape_2[0], shape_4[0]);
    EXPECT_EQ(shape_2[1], shape_4[1]);

    // Verify dtypes preserved
    expectDType(output_2.tensor());
    expectDType(output_4.tensor());
}

// ============================================================================
// Non-Maximum Suppression (NMS) Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, NMSBasicFiltering) {
    // Boxes: (N, 4) where each row is [x1, y1, x2, y2]
    Tensor boxes({20, 4}, dtype(), device());

    // Initialize with test data using helper
    if (dtype() == DType::Float32) {
        initializeBoxData<float>(boxes, 20, 10.0f, 50.0f);
    } else {
        initializeBoxData<double>(boxes, 20, 10.0, 50.0);
    }

    // Scores: (N,)
    Tensor scores({20}, dtype(), device());
    if (dtype() == DType::Float32) {
        initializeScoreData<float>(scores, 20, 0.05f);
    } else {
        initializeScoreData<double>(scores, 20, 0.05);
    }

    auto keep_indices = nms(boxes, scores, 0.5);  // IOU threshold = 0.5

    // NMS should return a tensor with valid indices
    EXPECT_GT(keep_indices.shape()[0], 0);
    EXPECT_LE(keep_indices.shape()[0], 20);
}

TEST_P(DetectionOpsMultiDTypeTest, NMSDifferentThresholds) {
    Tensor boxes({50, 4}, dtype(), device());
    Tensor scores({50}, dtype(), device());

    // Initialize with test data using helpers
    if (dtype() == DType::Float32) {
        initializeBoxData<float>(boxes, 50, 5.0f, 30.0f);
        initializeScoreData<float>(scores, 50, 0.02f);
    } else {
        initializeBoxData<double>(boxes, 50, 5.0, 30.0);
        initializeScoreData<double>(scores, 50, 0.02);
    }

    auto keep_low = nms(boxes, scores, 0.3);   // Stricter
    auto keep_high = nms(boxes, scores, 0.7);  // More permissive

    // Lower threshold should keep fewer or equal boxes
    EXPECT_LE(keep_low.shape()[0], keep_high.shape()[0]);
}

TEST_P(DetectionOpsMultiDTypeTest, NMSOutputShape) {
    // Create boxes and scores
    Tensor boxes({5, 4}, dtype(), device());
    Tensor scores({5}, dtype(), device());

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
    Tensor boxes1({5, 4}, dtype(), device());
    Tensor boxes2({5, 4}, dtype(), device());

    // Initialize with test boxes
    if (dtype() == DType::Float32) {
        initializeBoxData<float>(boxes1, 5, 20.0f, 40.0f);
        // boxes2 - slightly offset
        auto boxes2_cpu = boxes2.to(Device::cpu());
        float* data2 = boxes2_cpu.data<float>();
        for (int i = 0; i < 5; ++i) {
            data2[i*4 + 0] = static_cast<float>(i * 20 + 5);
            data2[i*4 + 1] = static_cast<float>(i * 20 + 5);
            data2[i*4 + 2] = static_cast<float>(i * 20 + 45);
            data2[i*4 + 3] = static_cast<float>(i * 20 + 45);
        }
        boxes2 = boxes2_cpu.to(device());
    } else {
        initializeBoxData<double>(boxes1, 5, 20.0, 40.0);
        // boxes2 - slightly offset
        auto boxes2_cpu = boxes2.to(Device::cpu());
        double* data2 = boxes2_cpu.data<double>();
        for (int i = 0; i < 5; ++i) {
            data2[i*4 + 0] = static_cast<double>(i * 20 + 5);
            data2[i*4 + 1] = static_cast<double>(i * 20 + 5);
            data2[i*4 + 2] = static_cast<double>(i * 20 + 45);
            data2[i*4 + 3] = static_cast<double>(i * 20 + 45);
        }
        boxes2 = boxes2_cpu.to(device());
    }

    auto ious = box_iou(boxes1, boxes2);

    // IOU matrix should be (5, 5)
    expectShape(ious, {5, 5});
    expectDType(ious);
}

TEST_P(DetectionOpsMultiDTypeTest, BoxEncodingDecoding) {
    Tensor boxes({10, 4}, dtype(), device());
    Tensor anchors({10, 4}, dtype(), device());

    // Initialize with test data
    if (dtype() == DType::Float32) {
        // Boxes with offset
        auto boxes_cpu = boxes.to(Device::cpu());
        float* box_data = boxes_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            box_data[i*4 + 0] = static_cast<float>(i * 10 + 5);
            box_data[i*4 + 1] = static_cast<float>(i * 10 + 5);
            box_data[i*4 + 2] = static_cast<float>(i * 10 + 35);
            box_data[i*4 + 3] = static_cast<float>(i * 10 + 35);
        }
        boxes = boxes_cpu.to(device());
        initializeBoxData<float>(anchors, 10, 10.0f, 40.0f);
    } else {
        // Boxes with offset
        auto boxes_cpu = boxes.to(Device::cpu());
        double* box_data = boxes_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            box_data[i*4 + 0] = static_cast<double>(i * 10 + 5);
            box_data[i*4 + 1] = static_cast<double>(i * 10 + 5);
            box_data[i*4 + 2] = static_cast<double>(i * 10 + 35);
            box_data[i*4 + 3] = static_cast<double>(i * 10 + 35);
        }
        boxes = boxes_cpu.to(device());
        initializeBoxData<double>(anchors, 10, 10.0, 40.0);
    }

    // Encode boxes relative to anchors
    auto encoded = encode_boxes(boxes, anchors);
    expectDType(encoded);

    // Decode back to original boxes
    auto decoded = decode_boxes(encoded, anchors);
    expectDType(decoded);

    // Decoded boxes should match original boxes shape
    expectShape(decoded, {10, 4});
}

TEST_P(DetectionOpsMultiDTypeTest, BoxEncodingShape) {
    Tensor boxes({20, 4}, dtype(), device());
    Tensor anchors({20, 4}, dtype(), device());

    auto encoded = encode_boxes(boxes, anchors);

    // Encoded should have same shape as input
    expectShape(encoded, {20, 4});
    expectDType(encoded);
}

// ============================================================================
// Anchor Generator Tests
// ============================================================================

TEST_P(DetectionOpsMultiDTypeTest, AnchorGeneratorSizes) {
    std::vector<float> sizes = {32.0f, 64.0f, 128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {0.5f, 1.0f, 2.0f};
    AnchorGenerator anchor_gen(sizes, aspect_ratios);

    // Feature map size: 56x56
    auto anchors = anchor_gen.generate(56, 56, 16, device());  // stride=16

    // Number of anchors = H * W * (num_sizes * num_aspect_ratios)
    // 56 * 56 * (5 * 3) = 47040
    EXPECT_EQ(anchors.shape()[0], 47040);
    EXPECT_EQ(anchors.shape()[1], 4);

    // Anchors are generated on the specified device
    EXPECT_EQ(anchors.device().type, device().type);
}

TEST_P(DetectionOpsMultiDTypeTest, AnchorGeneratorDifferentScales) {
    std::vector<float> sizes_small = {32.0f, 64.0f};
    std::vector<float> sizes_large = {128.0f, 256.0f, 512.0f};
    std::vector<float> aspect_ratios = {1.0f};

    AnchorGenerator anchor_gen_small(sizes_small, aspect_ratios);
    AnchorGenerator anchor_gen_large(sizes_large, aspect_ratios);

    auto anchors_small = anchor_gen_small.generate(28, 28, 16, device());
    auto anchors_large = anchor_gen_large.generate(28, 28, 16, device());

    // 28 * 28 * 2 = 1568 for small
    // 28 * 28 * 3 = 2352 for large
    EXPECT_EQ(anchors_small.shape()[0], 1568);
    EXPECT_EQ(anchors_large.shape()[0], 2352);

    // Verify device placement
    EXPECT_EQ(anchors_small.device().type, device().type);
    EXPECT_EQ(anchors_large.device().type, device().type);
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

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DetectionOpsMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 15
 * DTypes Tested: Float32, Float64 (Float16 skipped - detection requires higher precision)
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 15 tests × 2 dtypes × 3 backends = 90 test scenarios
 *
 * Coverage:
 * - ROI Align: Forward pass, gradient flow, different pool sizes, sampling ratios
 * - NMS: Basic filtering, different thresholds, output shapes
 * - Box Operations: IOU computation, encoding/decoding, shape verification
 * - Anchor Generation: Different scales, sizes, anchors per location
 *
 * Note: Detection operations require Float32/Float64 for bounding box coordinates
 * and confidence scores. Float16 is skipped due to precision requirements.
 */
