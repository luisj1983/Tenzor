#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/detection.hpp>
#include <hip/hip_runtime.h>
#include <cmath>

using namespace tenzor;

/**
 * @brief Test suite for ROCm detection kernels (NMS, ROI Align)
 *
 * These tests verify NMS suppression decisions and ROIAlign bilinear-
 * interpolation output against a CPU reference (see FINDING 5 in
 * findings.txt: the previous version of this file only asserted bare
 * SUCCEED()/tensor-construction shapes and never checked an actual IoU
 * suppression or bilinear-interpolation value).
 */

namespace {

float max_abs_diff(const Tensor& a, const Tensor& b) {
    auto a_cpu = a.to(Device::cpu()).contiguous();
    auto b_cpu = b.to(Device::cpu()).contiguous();
    EXPECT_EQ(a_cpu.numel(), b_cpu.numel());
    const float* ap = a_cpu.data<float>();
    const float* bp = b_cpu.data<float>();
    float m = 0.0f;
    for (int64_t i = 0; i < a_cpu.numel(); ++i) {
        m = std::max(m, std::fabs(ap[i] - bp[i]));
    }
    return m;
}

}  // namespace

class ROCmDetectionKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Tenzor
        tenzor::initialize();

        // Check if ROCm is available
        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);

        if (error != hipSuccess || device_count == 0) {
            GTEST_SKIP() << "ROCm device not available, skipping ROCm detection kernel tests";
        }
    }
};

TEST_F(ROCmDetectionKernelsTest, NMS_SuppressesKnownOverlap) {
    // Two heavily-overlapping boxes (IoU well above 0.5) plus one disjoint
    // box. The higher-scoring overlapping box must survive, the lower-scoring
    // overlapping box must be suppressed, and the disjoint box always survives.
    std::vector<float> boxes_data = {
        0.0f, 0.0f, 10.0f, 10.0f,   // box 0: overlaps box 1 heavily
        1.0f, 1.0f, 11.0f, 11.0f,   // box 1: overlaps box 0 heavily, lower score
        50.0f, 50.0f, 60.0f, 60.0f  // box 2: disjoint from both
    };
    std::vector<float> scores_data = {0.9f, 0.8f, 0.7f};

    auto boxes_cpu = from_data(boxes_data.data(), {3, 4});
    auto scores_cpu = from_data(scores_data.data(), {3});
    auto boxes_rocm = boxes_cpu.to(Device::rocm(0));
    auto scores_rocm = scores_cpu.to(Device::rocm(0));

    auto keep_cpu = ops::nms(boxes_cpu, scores_cpu, 0.5).to(Device::cpu()).contiguous();
    auto keep_rocm = ops::nms(boxes_rocm, scores_rocm, 0.5).to(Device::cpu()).contiguous();

    ASSERT_EQ(keep_cpu.numel(), 2) << "Box 1 should be suppressed by box 0 (IoU > 0.5)";
    ASSERT_EQ(keep_rocm.numel(), keep_cpu.numel())
        << "ROCm NMS kept a different number of boxes than the CPU reference";

    const int64_t* cpu_idx = keep_cpu.data<int64_t>();
    const int64_t* rocm_idx = keep_rocm.data<int64_t>();
    for (int64_t i = 0; i < keep_cpu.numel(); ++i) {
        EXPECT_EQ(cpu_idx[i], rocm_idx[i]) << "kept index " << i << " diverges";
    }
    // Box 1 (the suppressed overlap) must not appear in either result.
    for (int64_t i = 0; i < keep_rocm.numel(); ++i) {
        EXPECT_NE(rocm_idx[i], 1) << "ROCm NMS failed to suppress the overlapping lower-score box";
    }
}

TEST_F(ROCmDetectionKernelsTest, NMS_RandomBoxesMatchesCPU) {
    // Random boxes/scores: exact kept-index agreement between CPU and ROCm
    // across several box counts.
    for (int64_t num_boxes : {10, 50, 100, 500}) {
        auto boxes_cpu = rand({num_boxes, 4}, DType::Float32, Device::cpu()) * 100.0f;
        auto scores_cpu = rand({num_boxes}, DType::Float32, Device::cpu());
        auto boxes_rocm = boxes_cpu.to(Device::rocm(0));
        auto scores_rocm = scores_cpu.to(Device::rocm(0));

        auto keep_cpu = ops::nms(boxes_cpu, scores_cpu, 0.5).to(Device::cpu()).contiguous();
        auto keep_rocm = ops::nms(boxes_rocm, scores_rocm, 0.5).to(Device::cpu()).contiguous();

        ASSERT_EQ(keep_cpu.numel(), keep_rocm.numel()) << "num_boxes=" << num_boxes;
        const int64_t* cpu_idx = keep_cpu.data<int64_t>();
        const int64_t* rocm_idx = keep_rocm.data<int64_t>();
        for (int64_t i = 0; i < keep_cpu.numel(); ++i) {
            EXPECT_EQ(cpu_idx[i], rocm_idx[i]) << "num_boxes=" << num_boxes << " kept index " << i;
        }
    }
}

TEST_F(ROCmDetectionKernelsTest, ROIAlign_MatchesCPUReference) {
    const int64_t batch_size = 2;
    const int64_t channels = 3;
    const int64_t feat_h = 32;
    const int64_t feat_w = 32;
    const int64_t output_h = 7;
    const int64_t output_w = 7;

    auto features_cpu = randn({batch_size, channels, feat_h, feat_w}, DType::Float32, Device::cpu());

    // ROIs: (batch_idx, x1, y1, x2, y2) in the original image space (feature
    // map is scaled by spatial_scale below).
    std::vector<float> rois_data = {
        0.0f, 2.0f, 2.0f, 20.0f, 20.0f,
        0.0f, 5.0f, 5.0f, 30.0f, 30.0f,
        1.0f, 0.0f, 0.0f, 32.0f, 32.0f,
        1.0f, 10.0f, 3.0f, 25.0f, 28.0f,
    };
    auto rois_cpu = from_data(rois_data.data(), {4, 5});

    auto features_rocm = features_cpu.to(Device::rocm(0));
    auto rois_rocm = rois_cpu.to(Device::rocm(0));

    nn::detection::ROIAlign roi_align_cpu(output_h, output_w, 1.0, 2, true);
    nn::detection::ROIAlign roi_align_rocm(output_h, output_w, 1.0, 2, true);

    auto out_cpu = roi_align_cpu.forward(Variable(features_cpu, false), rois_cpu).tensor();
    auto out_rocm = roi_align_rocm.forward(Variable(features_rocm, false), rois_rocm).tensor();

    ASSERT_EQ(out_cpu.shape().size(), out_rocm.shape().size());
    for (size_t i = 0; i < out_cpu.shape().size(); ++i) {
        EXPECT_EQ(out_cpu.shape()[i], out_rocm.shape()[i]);
    }
    EXPECT_EQ(out_cpu.shape()[0], 4);
    EXPECT_EQ(out_cpu.shape()[1], channels);
    EXPECT_EQ(out_cpu.shape()[2], output_h);
    EXPECT_EQ(out_cpu.shape()[3], output_w);

    EXPECT_LT(max_abs_diff(out_cpu, out_rocm), 1e-3f)
        << "ROCm ROIAlign bilinear-interpolation output diverges from CPU reference";
}

TEST_F(ROCmDetectionKernelsTest, NMS_InputShapes) {
    auto device = Device::rocm(0);

    // Test various input sizes
    std::vector<int64_t> box_counts = {10, 50, 100, 500};

    for (int64_t num_boxes : box_counts) {
        auto boxes = randn({num_boxes, 4}, DType::Float32, device);
        auto scores = rand({num_boxes}, DType::Float32, device);

        // Verify shapes
        EXPECT_EQ(boxes.shape()[0], num_boxes);
        EXPECT_EQ(boxes.shape()[1], 4);  // x1, y1, x2, y2
        EXPECT_EQ(scores.shape()[0], num_boxes);
    }
}

TEST_F(ROCmDetectionKernelsTest, ROIAlign_OutputShapes) {
    auto device = Device::rocm(0);

    int64_t batch_size = 2;
    int64_t channels = 3;
    int64_t feat_h = 64;
    int64_t feat_w = 64;
    int64_t num_rois = 10;
    int64_t output_h = 7;
    int64_t output_w = 7;

    auto features = randn({batch_size, channels, feat_h, feat_w}, DType::Float32, device);
    auto rois = randn({num_rois, 5}, DType::Float32, device);

    // Verify input shapes
    EXPECT_EQ(features.ndim(), 4);
    EXPECT_EQ(rois.shape()[0], num_rois);
    EXPECT_EQ(rois.shape()[1], 5);  // batch_idx, x1, y1, x2, y2
}

TEST_F(ROCmDetectionKernelsTest, MemoryAllocation) {
    auto device = Device::rocm(0);

    // Test that we can allocate large tensors for detection
    auto large_features = zeros({1, 256, 128, 128}, DType::Float32, device);
    auto large_boxes = zeros({1000, 4}, DType::Float32, device);

    EXPECT_EQ(large_features.numel(), 1 * 256 * 128 * 128);
    EXPECT_EQ(large_boxes.numel(), 1000 * 4);
}

TEST_F(ROCmDetectionKernelsTest, DeviceSynchronization) {
    auto device = Device::rocm(0);

    // Create tensors
    auto boxes = randn({100, 4}, DType::Float32, device);
    auto scores = rand({100}, DType::Float32, device);

    // Test synchronization
    ASSERT_EQ(hipDeviceSynchronize(), hipSuccess);

    // Copy to CPU to verify data exists
    auto boxes_cpu = boxes.to(Device::cpu());
    EXPECT_EQ(boxes_cpu.numel(), 100 * 4);
}
