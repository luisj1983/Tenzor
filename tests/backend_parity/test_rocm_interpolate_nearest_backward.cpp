// test_rocm_interpolate_nearest_backward.cpp
//
// Wave H4: ROCm InterpolateBackward gains native nearest-mode support via
// `interpolate_nearest_backward_kernel_hip` (atomicAdd scatter to the single
// nearest input pixel). The prior code silently routed `mode="nearest"`
// through the bilinear backward kernel, producing wrong gradients.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

class RocmInterpolateNearestBackward : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    static bool has_rocm() {
        try {
            auto t = zeros({1}, DType::Float32, Device::rocm(0));
            (void)t; return true;
        } catch (...) { return false; }
    }
};

static auto seq_f32(std::vector<int64_t> shape) -> Tensor {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t i = 0; i < t.numel(); ++i) {
        p[i] = static_cast<float>(i) * 0.5f;
    }
    return t;
}

}  // namespace

TEST_F(RocmInterpolateNearestBackward, Nearest_MatchesCPU) {
    // Forward upsample 2x2 -> 4x4 in nearest mode; backward scatters gradient
    // from each 4x4 output pixel to its 2x2 input pixel.
    auto grad_out_cpu = seq_f32({1, 2, 4, 4});  // (N, C, out_h, out_w)

    OpAttributes attrs;
    // InputShape is stored as a comma-separated string (see function_shape.cpp:303).
    attrs.set(AttrKey::InputShape, "2,2");
    attrs.set(AttrKey::Mode, "nearest");
    attrs.set(AttrKey::AlignCorners, false);

    // Audit: previously wrapped in try{...}catch(...){GTEST_SKIP("CPU
    // InterpolateBackward nearest unsupported")}. The CPU nearest backward IS
    // the reference for this parity test (file header: ROCm just gained the
    // native kernel) — a CPU reference failure is a real bug, not a clean skip.
    // Let it propagate (test_operation_parity_single always runs it first).
    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::InterpolateBackward, ins, attrs)[0];
    };

    // On a CPU-only host, fall back to comparing CPU's own result against a
    // recorded golden instead of skipping outright (FINDING 17 in
    // findings.txt: this file previously had zero golden::* integration).
    Device target = has_rocm() ? Device::rocm(0) : Device::cpu();
    test_operation_parity_single(op, {grad_out_cpu}, target, 1e-5f, 5e-5f,
                                  "InterpolateBackward_Nearest");
}

TEST_F(RocmInterpolateNearestBackward, Bilinear_MatchesCPU) {
    // Wave H4 regression check: the bilinear backward gradient on ROCm must
    // match the CPU reference, not merely return a correctly-shaped tensor.
    // A shape-only check gave false confidence given this file's history of
    // ROCm silently routing nearest through the bilinear kernel — a value
    // comparison is what actually catches a wrong-kernel/wrong-gradient bug.
    auto grad_out_cpu = seq_f32({1, 2, 4, 4});

    OpAttributes attrs;
    attrs.set(AttrKey::InputShape, "2,2");
    attrs.set(AttrKey::Mode, "bilinear");
    attrs.set(AttrKey::AlignCorners, false);

    // CPU reference (same dispatch path, CPU device). A CPU failure is a real
    // bug and must propagate, not be skipped (test_operation_parity_single
    // always runs it first).
    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::InterpolateBackward, ins, attrs)[0];
    };

    // Generous tolerance for the bilinear backward scatter (atomicAdd ordering
    // on GPU vs sequential accumulation on CPU), but tight enough that a
    // wrong-kernel routing or zeroed gradient fails. On a CPU-only host, fall
    // back to a recorded golden instead of skipping outright (FINDING 17).
    Device target = has_rocm() ? Device::rocm(0) : Device::cpu();
    test_operation_parity_single(op, {grad_out_cpu}, target, 1e-5f, 1e-3f,
                                  "InterpolateBackward_Bilinear");
}
