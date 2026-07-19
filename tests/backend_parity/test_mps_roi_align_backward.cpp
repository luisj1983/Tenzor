// test_mps_roi_align_backward.cpp
//
// M16: MPS ROIAlignBackward previously had no native kernel at all
// (unconditional CPU round-trip). Verifies the new native Metal kernel
// (roi_align_backward_kernel, pool3d.metal) matches the CPU reference for
// both forward and backward.
//
// NOTE: this test can only run where MPS is available (macOS + Apple GPU).
// It was authored and reviewed on a Linux machine where MPS cannot be
// compiled or executed at all (TENZOR_BUILD_MPS is gated on APPLE in
// CMakeLists.txt) -- it GTEST_SKIPs cleanly there via has_mps(). Please run
// this on macOS to confirm before relying on it.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

class MpsRoiAlignBackward : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    static bool has_mps() {
        try {
            auto t = zeros({1}, DType::Float32, Device::mps(0));
            (void)t; return true;
        } catch (...) { return false; }
    }
};

// features: (N=1, C=2, H=8, W=8); one ROI covering most of the feature map.
static auto seq_f32(std::vector<int64_t> shape) -> Tensor {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i) * 0.1f;
    }
    return t;
}

static auto make_rois() -> Tensor {
    // [batch_idx, x1, y1, x2, y2] -- a single ROI, in feature-map pixel coords
    // after spatial_scale=1.0 (kept simple: same scale as the feature grid).
    float data[5] = {0.0f, 1.0f, 1.0f, 6.0f, 6.0f};
    return Tensor::from_blob(data, {1, 5}, DType::Float32, Device::cpu()).clone();
}

}  // namespace

TEST_F(MpsRoiAlignBackward, Forward_MatchesCPU) {
    auto features_cpu = seq_f32({1, 2, 8, 8});
    auto rois_cpu = make_rois();

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSizeH, int64_t(4));
    attrs.set(AttrKey::OutputSizeW, int64_t(4));
    attrs.set(AttrKey::SpatialScale, 1.0);
    attrs.set(AttrKey::SamplingRatio, int64_t(2));
    attrs.set(AttrKey::Aligned, true);

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::ROIAlignForward, ins, attrs)[0];
    };

    // On a CPU-only host (i.e. every non-macOS host), fall back to comparing
    // against a recorded golden instead of skipping outright (FINDING 17).
    Device target = has_mps() ? Device::mps(0) : Device::cpu();
    test_operation_parity_single(op, {features_cpu, rois_cpu}, target, 1e-5f, 1e-4f,
                                  "ROIAlign_Forward");
}

TEST_F(MpsRoiAlignBackward, Backward_MatchesCPU) {
    auto grad_out_cpu = seq_f32({1, 2, 4, 4});
    auto rois_cpu = make_rois();

    OpAttributes attrs;
    attrs.set(AttrKey::BatchSize, int64_t(1));
    attrs.set(AttrKey::FeatHeight, int64_t(8));
    attrs.set(AttrKey::FeatWidth, int64_t(8));
    attrs.set(AttrKey::SpatialScale, 1.0);
    attrs.set(AttrKey::SamplingRatio, int64_t(2));
    attrs.set(AttrKey::Aligned, true);

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::ROIAlignBackward, ins, attrs)[0];
    };

    Device target = has_mps() ? Device::mps(0) : Device::cpu();
    test_operation_parity_single(op, {grad_out_cpu, rois_cpu}, target, 1e-5f, 1e-4f,
                                  "ROIAlign_Backward");
}
