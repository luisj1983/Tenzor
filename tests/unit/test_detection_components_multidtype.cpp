/**
 * @file test_detection_components_multidtype.cpp
 * @brief Multi-dtype tests for detection components
 *
 * Tests detection operations with Float32 and Float64 dtypes
 * across CPU, CUDA, Vulkan, and OneAPI backends.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include "tenzor/nn/detection/anchors.hpp"
#include "tenzor/nn/detection/roi_ops.hpp"
#include "tenzor/ops/detection.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn::detection;
using namespace tenzor::ops;

/**
 * Multi-dtype parameterized testing for detection components
 *
 * Coverage:
 * - Dtypes: Float32, Float64
 * - Backends: CPU, CUDA, Vulkan, OneAPI
 * - Components: AnchorGenerator, ROIAlign, NMS
 */

// ============================================================================
// Backend + DType Parameterization
// ============================================================================

struct BackendDTypeParam {
    std::string backend_name;
    DType dtype;
    std::string dtype_name;
    float tolerance;

    std::string ToString() const {
        return backend_name + "_" + dtype_name;
    }
};

// Required for gtest_discover_tests to show human-readable test names
void PrintTo(const BackendDTypeParam& param, std::ostream* os) {
    *os << param.ToString();
}

class DetectionComponentsMultiDTypeTest : public ::testing::TestWithParam<BackendDTypeParam> {
protected:
    Device device;
    DType dtype;
    float tol;

    void SetUp() override {
        tenzor::initialize();

        auto param = GetParam();
        dtype = param.dtype;
        tol = param.tolerance;

        if (param.backend_name == "cpu") {
            device = Device::cpu();
        }
        else if (param.backend_name == "cuda") {
            if (!isBackendAvailable(Device::Type::CUDA)) {
                GTEST_SKIP() << "CUDA not available";
            }
            device = Device::cuda(0);
        }
        else if (param.backend_name == "vulkan") {
            if (!isBackendAvailable(Device::Type::Vulkan)) {
                GTEST_SKIP() << "Vulkan not available";
            }
            device = Device::vulkan(0);
        }
        else if (param.backend_name == "oneapi") {
            if (!isBackendAvailable(Device::Type::OneAPI)) {
                GTEST_SKIP() << "OneAPI not available";
            }
            device = Device::oneapi(0);
        }
    }

    void TearDown() override {
        tenzor::finalize();
    }

    static bool isBackendAvailable(Device::Type type) {
        try {
            Device test_device{type, 0};
            auto t = zeros({2, 2}, DType::Float32, test_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to convert tensor to test dtype
    Tensor toTestDType(const Tensor& t) {
        if (t.dtype() != dtype) {
            return t.to(dtype);
        }
        return t;
    }
};

// ============================================================================
// AnchorGenerator Tests
// ============================================================================

TEST_P(DetectionComponentsMultiDTypeTest, AnchorGeneratorBasic) {
    // Create anchor generator with 3 sizes and 3 aspect ratios
    AnchorGenerator anchors({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});

    // Generate anchors for 2x2 feature map with stride 16
    auto boxes = anchors.generate(2, 2, 16);
    boxes = toTestDType(boxes);

    // Should have 2*2*3*3 = 36 anchors
    auto shape = boxes.shape();
    EXPECT_EQ(shape[0], 36);
    EXPECT_EQ(shape[1], 4);
    EXPECT_EQ(boxes.dtype(), dtype);
}

TEST_P(DetectionComponentsMultiDTypeTest, AnchorGeneratorNumAnchors) {
    AnchorGenerator anchors({32.0f, 64.0f}, {0.5f, 1.0f, 2.0f});
    EXPECT_EQ(anchors.num_anchors_per_location(), 6);

    AnchorGenerator anchors2({32.0f, 64.0f, 128.0f}, {0.5f, 1.0f, 2.0f});
    EXPECT_EQ(anchors2.num_anchors_per_location(), 9);
}

TEST_P(DetectionComponentsMultiDTypeTest, AnchorGeneratorDifferentStrides) {
    AnchorGenerator anchors({32.0f}, {1.0f});

    // Test different strides
    auto boxes_8 = toTestDType(anchors.generate(4, 4, 8));
    auto boxes_16 = toTestDType(anchors.generate(2, 2, 16));

    EXPECT_EQ(boxes_8.shape()[0], 16);  // 4*4*1*1
    EXPECT_EQ(boxes_16.shape()[0], 4);   // 2*2*1*1
    EXPECT_EQ(boxes_8.dtype(), dtype);
    EXPECT_EQ(boxes_16.dtype(), dtype);
}

// ============================================================================
// NMS Tests
// ============================================================================

TEST_P(DetectionComponentsMultiDTypeTest, NMSBasicShape) {
    // Create boxes and scores
    Tensor boxes({10, 4}, dtype, device);
    Tensor scores({10}, dtype, device);

    // Fill with non-overlapping boxes
    auto boxes_cpu = boxes.to(Device::cpu());
    auto scores_cpu = scores.to(Device::cpu());

    if (dtype == DType::Float32) {
        float* box_data = boxes_cpu.data<float>();
        float* score_data = scores_cpu.data<float>();
        for (int i = 0; i < 10; ++i) {
            // Non-overlapping boxes
            box_data[i*4 + 0] = i * 20.0f;      // x1
            box_data[i*4 + 1] = i * 20.0f;      // y1
            box_data[i*4 + 2] = i * 20.0f + 10.0f;  // x2
            box_data[i*4 + 3] = i * 20.0f + 10.0f;  // y2
            score_data[i] = 1.0f - i * 0.05f;
        }
    } else if (dtype == DType::Float64) {
        double* box_data = boxes_cpu.data<double>();
        double* score_data = scores_cpu.data<double>();
        for (int i = 0; i < 10; ++i) {
            box_data[i*4 + 0] = i * 20.0;
            box_data[i*4 + 1] = i * 20.0;
            box_data[i*4 + 2] = i * 20.0 + 10.0;
            box_data[i*4 + 3] = i * 20.0 + 10.0;
            score_data[i] = 1.0 - i * 0.05;
        }
    }

    boxes = boxes_cpu.to(device);
    scores = scores_cpu.to(device);

    // Apply NMS
    auto keep = nms(boxes, scores, 0.5);

    // Non-overlapping boxes should all be kept
    EXPECT_EQ(keep.shape()[0], 10);
}

TEST_P(DetectionComponentsMultiDTypeTest, NMSOverlappingBoxes) {
    // Create overlapping boxes
    Tensor boxes({4, 4}, dtype, device);
    Tensor scores({4}, dtype, device);

    auto boxes_cpu = boxes.to(Device::cpu());
    auto scores_cpu = scores.to(Device::cpu());

    if (dtype == DType::Float32) {
        float* box_data = boxes_cpu.data<float>();
        float* score_data = scores_cpu.data<float>();

        // Box 0: (0, 0, 10, 10), score: 0.9
        box_data[0] = 0.0f; box_data[1] = 0.0f; box_data[2] = 10.0f; box_data[3] = 10.0f;
        score_data[0] = 0.9f;

        // Box 1: (1, 1, 11, 11), score: 0.8 (overlaps with box 0)
        box_data[4] = 1.0f; box_data[5] = 1.0f; box_data[6] = 11.0f; box_data[7] = 11.0f;
        score_data[1] = 0.8f;

        // Box 2: (20, 20, 30, 30), score: 0.7 (no overlap)
        box_data[8] = 20.0f; box_data[9] = 20.0f; box_data[10] = 30.0f; box_data[11] = 30.0f;
        score_data[2] = 0.7f;

        // Box 3: (1, 1, 10, 10), score: 0.6 (overlaps with box 0, IoU=0.81)
        box_data[12] = 1.0f; box_data[13] = 1.0f; box_data[14] = 10.0f; box_data[15] = 10.0f;
        score_data[3] = 0.6f;
    } else if (dtype == DType::Float64) {
        double* box_data = boxes_cpu.data<double>();
        double* score_data = scores_cpu.data<double>();

        box_data[0] = 0.0; box_data[1] = 0.0; box_data[2] = 10.0; box_data[3] = 10.0;
        score_data[0] = 0.9;

        box_data[4] = 1.0; box_data[5] = 1.0; box_data[6] = 11.0; box_data[7] = 11.0;
        score_data[1] = 0.8;

        box_data[8] = 20.0; box_data[9] = 20.0; box_data[10] = 30.0; box_data[11] = 30.0;
        score_data[2] = 0.7;

        // Box 3: (1, 1, 10, 10), IoU with box 0 = 0.81
        box_data[12] = 1.0; box_data[13] = 1.0; box_data[14] = 10.0; box_data[15] = 10.0;
        score_data[3] = 0.6;
    }

    boxes = boxes_cpu.to(device);
    scores = scores_cpu.to(device);

    auto keep = nms(boxes, scores, 0.5);

    // Should keep box 0 (highest score) and box 2 (no overlap)
    EXPECT_EQ(keep.shape()[0], 2);
}

// ============================================================================
// ROIAlign Tests
// ============================================================================

TEST_P(DetectionComponentsMultiDTypeTest, ROIAlignBasic) {
    // Create simple feature map
    auto features = randn({1, 2, 8, 8}, DType::Float32, device);
    features = toTestDType(features);

    // Create single ROI
    Tensor rois({1, 5}, dtype, device);
    auto rois_cpu = rois.to(Device::cpu());

    if (dtype == DType::Float32) {
        float* data = rois_cpu.data<float>();
        data[0] = 0.0f;  // batch_idx
        data[1] = 0.0f;  // x1
        data[2] = 0.0f;  // y1
        data[3] = 8.0f;  // x2
        data[4] = 8.0f;  // y2
    } else if (dtype == DType::Float64) {
        double* data = rois_cpu.data<double>();
        data[0] = 0.0;
        data[1] = 0.0;
        data[2] = 0.0;
        data[3] = 8.0;
        data[4] = 8.0;
    }

    rois = rois_cpu.to(device);

    // ROIAlign to 3x3 output
    ROIAlign roi_align(3, 3, 1.0, 2, true);
    auto aligned = roi_align.forward(Variable(features, false), rois);

    // Check output shape
    auto shape = aligned.tensor().shape();
    EXPECT_EQ(shape[0], 1);   // num_rois
    EXPECT_EQ(shape[1], 2);   // channels
    EXPECT_EQ(shape[2], 3);   // output_h
    EXPECT_EQ(shape[3], 3);   // output_w
    EXPECT_EQ(aligned.tensor().dtype(), dtype);
}

TEST_P(DetectionComponentsMultiDTypeTest, ROIAlignMultipleROIs) {
    auto features = randn({2, 4, 16, 16}, DType::Float32, device);
    features = toTestDType(features);

    // Multiple ROIs from different batches
    Tensor rois({3, 5}, dtype, device);
    auto rois_cpu = rois.to(Device::cpu());

    if (dtype == DType::Float32) {
        float* data = rois_cpu.data<float>();
        // ROI 0: batch 0
        data[0] = 0.0f; data[1] = 0.0f; data[2] = 0.0f; data[3] = 16.0f; data[4] = 16.0f;
        // ROI 1: batch 1
        data[5] = 1.0f; data[6] = 8.0f; data[7] = 8.0f; data[8] = 24.0f; data[9] = 24.0f;
        // ROI 2: batch 0
        data[10] = 0.0f; data[11] = 4.0f; data[12] = 4.0f; data[13] = 12.0f; data[14] = 12.0f;
    } else if (dtype == DType::Float64) {
        double* data = rois_cpu.data<double>();
        data[0] = 0.0; data[1] = 0.0; data[2] = 0.0; data[3] = 16.0; data[4] = 16.0;
        data[5] = 1.0; data[6] = 8.0; data[7] = 8.0; data[8] = 24.0; data[9] = 24.0;
        data[10] = 0.0; data[11] = 4.0; data[12] = 4.0; data[13] = 12.0; data[14] = 12.0;
    }

    rois = rois_cpu.to(device);

    ROIAlign roi_align(7, 7, 1.0, 2, true);
    auto aligned = roi_align.forward(Variable(features, false), rois);

    // Check output shape
    auto shape = aligned.tensor().shape();
    EXPECT_EQ(shape[0], 3);   // num_rois
    EXPECT_EQ(shape[1], 4);   // channels
    EXPECT_EQ(shape[2], 7);   // output_h
    EXPECT_EQ(shape[3], 7);   // output_w
    EXPECT_EQ(aligned.tensor().dtype(), dtype);
}

TEST_P(DetectionComponentsMultiDTypeTest, ROIAlignGradient) {
    // Test gradient flow through ROIAlign
    auto features = randn({1, 2, 8, 8}, DType::Float32, device);
    features = toTestDType(features);

    Tensor rois({1, 5}, dtype, device);
    auto rois_cpu = rois.to(Device::cpu());

    if (dtype == DType::Float32) {
        float* data = rois_cpu.data<float>();
        data[0] = 0.0f; data[1] = 0.0f; data[2] = 0.0f; data[3] = 8.0f; data[4] = 8.0f;
    } else if (dtype == DType::Float64) {
        double* data = rois_cpu.data<double>();
        data[0] = 0.0; data[1] = 0.0; data[2] = 0.0; data[3] = 8.0; data[4] = 8.0;
    }

    rois = rois_cpu.to(device);

    auto features_var = Variable(features, true);

    ROIAlign roi_align(3, 3, 1.0, 2, true);
    auto aligned = roi_align.forward(features_var, rois);

    // Backward pass
    auto grad_output = ones_like(aligned.tensor());

    EXPECT_NO_THROW({
        aligned.backward(grad_output);
    });

    // Check that gradients were computed
    EXPECT_TRUE(features_var.has_grad());
    if (features_var.has_grad()) {
        auto grad_shape = features_var.grad()->shape();
        auto feat_shape = features.shape();
        EXPECT_EQ(std::vector<int64_t>(grad_shape.begin(), grad_shape.end()),
                  std::vector<int64_t>(feat_shape.begin(), feat_shape.end()));
        EXPECT_EQ(features_var.grad()->dtype(), dtype);
    }
}

TEST_P(DetectionComponentsMultiDTypeTest, ROIAlignDifferentOutputSizes) {
    auto features = randn({1, 3, 16, 16}, DType::Float32, device);
    features = toTestDType(features);

    Tensor rois({1, 5}, dtype, device);
    auto rois_cpu = rois.to(Device::cpu());

    if (dtype == DType::Float32) {
        float* data = rois_cpu.data<float>();
        data[0] = 0.0f; data[1] = 0.0f; data[2] = 0.0f; data[3] = 16.0f; data[4] = 16.0f;
    } else if (dtype == DType::Float64) {
        double* data = rois_cpu.data<double>();
        data[0] = 0.0; data[1] = 0.0; data[2] = 0.0; data[3] = 16.0; data[4] = 16.0;
    }

    rois = rois_cpu.to(device);

    // Test different output sizes
    std::vector<int> sizes = {3, 5, 7};
    for (int size : sizes) {
        ROIAlign roi_align(size, size, 1.0, 2, true);
        auto aligned = roi_align.forward(Variable(features, false), rois);

        auto shape = aligned.tensor().shape();
        EXPECT_EQ(shape[2], size);
        EXPECT_EQ(shape[3], size);
        EXPECT_EQ(aligned.tensor().dtype(), dtype);
    }
}

// ============================================================================
// Test Instantiation
// ============================================================================

std::vector<BackendDTypeParam> GenerateDetectionComponentsTestCombinations() {
    std::vector<std::string> backends = {"cpu", "cuda", "vulkan", "oneapi"};

    std::vector<std::tuple<DType, std::string, float>> dtypes = {
        {DType::Float32, "float32", 1e-3f},
        {DType::Float64, "float64", 1e-8f},
    };

    std::vector<BackendDTypeParam> combinations;
    for (const auto& backend : backends) {
        for (const auto& [dtype, dtype_name, tolerance] : dtypes) {
            combinations.push_back({backend, dtype, dtype_name, tolerance});
        }
    }
    return combinations;
}

INSTANTIATE_TEST_SUITE_P(
    AllBackendsAllDTypes,
    DetectionComponentsMultiDTypeTest,
    ::testing::ValuesIn(GenerateDetectionComponentsTestCombinations()),
    [](const ::testing::TestParamInfo<BackendDTypeParam>& info) {
        return info.param.ToString();
    }
);

/*
 * COVERAGE SUMMARY:
 *
 * Test scenarios: 9 tests × 4 backends × 2 dtypes = 72 test scenarios
 *
 * Components Tested:
 * 1. AnchorGenerator (3 tests):
 *    - Basic anchor generation with multiple sizes/ratios
 *    - Num anchors per location verification
 *    - Different stride configurations
 *
 * 2. NMS (2 tests):
 *    - Basic NMS shape verification
 *    - Overlapping boxes filtering
 *
 * 3. ROIAlign (4 tests):
 *    - Basic ROI pooling to fixed size
 *    - Multiple ROIs from different batches
 *    - Gradient flow verification
 *    - Different output sizes (3x3, 5x5, 7x7)
 *
 * DType-Specific Testing:
 * - Float32: Standard precision (1e-3 tolerance)
 * - Float64: High precision (1e-8 tolerance)
 *
 * Backend Coverage:
 * - CPU: Reference implementation
 * - CUDA: GPU acceleration for detection pipelines
 * - Vulkan: Cross-platform GPU support
 * - OneAPI: Intel hardware optimization
 */
