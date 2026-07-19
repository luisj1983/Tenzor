#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/detection.hpp>
#include <cmath>

using namespace tenzor;

/**
 * @brief Test suite for OneAPI detection kernels (NMS, ROI Align)
 *
 * Mirrors tests/backends/test_rocm_detection_kernels.cpp (see FINDING 3 in
 * findings.txt: ROCm had a dedicated detection-kernel test file with no
 * OneAPI equivalent). Verifies NMS suppression decisions and ROIAlign
 * bilinear-interpolation output against a CPU reference.
 */

namespace {

bool is_oneapi_available() {
    try {
        auto device = Device::oneapi(0);
        auto t = zeros({1}, DType::Float32, device);
        (void)t;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

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

class OneAPIDetectionKernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        if (!is_oneapi_available()) {
            GTEST_SKIP() << "OneAPI device not available, skipping OneAPI detection kernel tests";
        }
    }
};

TEST_F(OneAPIDetectionKernelsTest, NMS_SuppressesKnownOverlap) {
    std::vector<float> boxes_data = {
        0.0f, 0.0f, 10.0f, 10.0f,   // box 0: overlaps box 1 heavily
        1.0f, 1.0f, 11.0f, 11.0f,   // box 1: overlaps box 0 heavily, lower score
        50.0f, 50.0f, 60.0f, 60.0f  // box 2: disjoint from both
    };
    std::vector<float> scores_data = {0.9f, 0.8f, 0.7f};

    auto boxes_cpu = from_data(boxes_data.data(), {3, 4});
    auto scores_cpu = from_data(scores_data.data(), {3});
    auto boxes_oneapi = boxes_cpu.to(Device::oneapi(0));
    auto scores_oneapi = scores_cpu.to(Device::oneapi(0));

    auto keep_cpu = ops::nms(boxes_cpu, scores_cpu, 0.5).to(Device::cpu()).contiguous();
    auto keep_oneapi = ops::nms(boxes_oneapi, scores_oneapi, 0.5).to(Device::cpu()).contiguous();

    ASSERT_EQ(keep_cpu.numel(), 2) << "Box 1 should be suppressed by box 0 (IoU > 0.5)";
    ASSERT_EQ(keep_oneapi.numel(), keep_cpu.numel())
        << "OneAPI NMS kept a different number of boxes than the CPU reference";

    const int64_t* cpu_idx = keep_cpu.data<int64_t>();
    const int64_t* oneapi_idx = keep_oneapi.data<int64_t>();
    for (int64_t i = 0; i < keep_cpu.numel(); ++i) {
        EXPECT_EQ(cpu_idx[i], oneapi_idx[i]) << "kept index " << i << " diverges";
    }
    for (int64_t i = 0; i < keep_oneapi.numel(); ++i) {
        EXPECT_NE(oneapi_idx[i], 1) << "OneAPI NMS failed to suppress the overlapping lower-score box";
    }
}

TEST_F(OneAPIDetectionKernelsTest, NMS_RandomBoxesMatchesCPU) {
    for (int64_t num_boxes : {10, 50, 100, 500}) {
        auto boxes_cpu = rand({num_boxes, 4}, DType::Float32, Device::cpu()) * 100.0f;
        auto scores_cpu = rand({num_boxes}, DType::Float32, Device::cpu());
        auto boxes_oneapi = boxes_cpu.to(Device::oneapi(0));
        auto scores_oneapi = scores_cpu.to(Device::oneapi(0));

        auto keep_cpu = ops::nms(boxes_cpu, scores_cpu, 0.5).to(Device::cpu()).contiguous();
        auto keep_oneapi = ops::nms(boxes_oneapi, scores_oneapi, 0.5).to(Device::cpu()).contiguous();

        ASSERT_EQ(keep_cpu.numel(), keep_oneapi.numel()) << "num_boxes=" << num_boxes;
        const int64_t* cpu_idx = keep_cpu.data<int64_t>();
        const int64_t* oneapi_idx = keep_oneapi.data<int64_t>();
        for (int64_t i = 0; i < keep_cpu.numel(); ++i) {
            EXPECT_EQ(cpu_idx[i], oneapi_idx[i]) << "num_boxes=" << num_boxes << " kept index " << i;
        }
    }
}

TEST_F(OneAPIDetectionKernelsTest, ROIAlign_MatchesCPUReference) {
    const int64_t batch_size = 2;
    const int64_t channels = 3;
    const int64_t feat_h = 32;
    const int64_t feat_w = 32;
    const int64_t output_h = 7;
    const int64_t output_w = 7;

    auto features_cpu = randn({batch_size, channels, feat_h, feat_w}, DType::Float32, Device::cpu());

    std::vector<float> rois_data = {
        0.0f, 2.0f, 2.0f, 20.0f, 20.0f,
        0.0f, 5.0f, 5.0f, 30.0f, 30.0f,
        1.0f, 0.0f, 0.0f, 32.0f, 32.0f,
        1.0f, 10.0f, 3.0f, 25.0f, 28.0f,
    };
    auto rois_cpu = from_data(rois_data.data(), {4, 5});

    auto features_oneapi = features_cpu.to(Device::oneapi(0));
    auto rois_oneapi = rois_cpu.to(Device::oneapi(0));

    nn::detection::ROIAlign roi_align_cpu(output_h, output_w, 1.0, 2, true);
    nn::detection::ROIAlign roi_align_oneapi(output_h, output_w, 1.0, 2, true);

    auto out_cpu = roi_align_cpu.forward(Variable(features_cpu, false), rois_cpu).tensor();
    auto out_oneapi = roi_align_oneapi.forward(Variable(features_oneapi, false), rois_oneapi).tensor();

    ASSERT_EQ(out_cpu.shape().size(), out_oneapi.shape().size());
    for (size_t i = 0; i < out_cpu.shape().size(); ++i) {
        EXPECT_EQ(out_cpu.shape()[i], out_oneapi.shape()[i]);
    }
    EXPECT_EQ(out_cpu.shape()[0], 4);
    EXPECT_EQ(out_cpu.shape()[1], channels);
    EXPECT_EQ(out_cpu.shape()[2], output_h);
    EXPECT_EQ(out_cpu.shape()[3], output_w);

    EXPECT_LT(max_abs_diff(out_cpu, out_oneapi), 1e-3f)
        << "OneAPI ROIAlign bilinear-interpolation output diverges from CPU reference";
}
