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

using namespace tenzor;

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
    if (!has_rocm()) GTEST_SKIP() << "ROCm not available";

    // Forward upsample 2x2 -> 4x4 in nearest mode; backward scatters gradient
    // from each 4x4 output pixel to its 2x2 input pixel.
    auto grad_out_cpu = seq_f32({1, 2, 4, 4});  // (N, C, out_h, out_w)

    OpAttributes attrs;
    // InputShape is stored as a comma-separated string (see function_shape.cpp:303).
    attrs.set(AttrKey::InputShape, "2,2");
    attrs.set(AttrKey::Mode, "nearest");
    attrs.set(AttrKey::AlignCorners, false);

    std::vector<Tensor> cpu_inputs = {grad_out_cpu};
    Tensor cpu_out;
    try {
        cpu_out = dispatch(OpId::InterpolateBackward, cpu_inputs, attrs)[0];
    } catch (const std::exception& e) {
        GTEST_SKIP() << "CPU InterpolateBackward nearest unsupported: " << e.what();
    }

    auto grad_out_rocm = grad_out_cpu.to(Device::rocm(0));
    std::vector<Tensor> rocm_inputs = {grad_out_rocm};
    Tensor rocm_out = dispatch(OpId::InterpolateBackward, rocm_inputs, attrs)[0]
                           .to(Device::cpu());

    ASSERT_EQ(cpu_out.shape().size(), rocm_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], rocm_out.shape()[i]) << " dim " << i;
    }
    auto* cp = cpu_out.data<float>();
    auto* rp = rocm_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], rp[i], 5e-5f) << " elem " << i;
    }
}

TEST_F(RocmInterpolateNearestBackward, Bilinear_PathExecutes) {
    // Wave H4 regression check: bilinear backward path still executes and
    // returns a tensor of the correct shape. (Pre-existing CPU↔ROCm bilinear
    // numerical divergence is out of scope for this wave — see
    // existing test_interpolate_parity for the parity check.)
    if (!has_rocm()) GTEST_SKIP() << "ROCm not available";

    auto grad_out_cpu = seq_f32({1, 2, 4, 4});

    OpAttributes attrs;
    attrs.set(AttrKey::InputShape, "2,2");
    attrs.set(AttrKey::Mode, "bilinear");
    attrs.set(AttrKey::AlignCorners, false);

    auto grad_out_rocm = grad_out_cpu.to(Device::rocm(0));
    std::vector<Tensor> rocm_inputs = {grad_out_rocm};
    Tensor rocm_out;
    ASSERT_NO_THROW({
        rocm_out = dispatch(OpId::InterpolateBackward, rocm_inputs, attrs)[0]
                       .to(Device::cpu());
    });
    ASSERT_EQ(rocm_out.shape().size(), 4u);
    EXPECT_EQ(rocm_out.shape()[0], 1);
    EXPECT_EQ(rocm_out.shape()[1], 2);
    EXPECT_EQ(rocm_out.shape()[2], 2);
    EXPECT_EQ(rocm_out.shape()[3], 2);
}
