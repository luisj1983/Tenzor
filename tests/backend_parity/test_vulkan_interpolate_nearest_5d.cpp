// test_vulkan_interpolate_nearest_5d.cpp
//
// L10: Vulkan's dispatchInterpolate/dispatchInterpolateBackward previously
// hard-rejected any 5D (volumetric) input for mode="nearest" (only
// mode="trilinear" had native 5D support, per the M29 fix) -- unlike every
// other backend, which implements nearest-5D forward AND backward. Verifies
// the new native Vulkan kernels (nearest_interpolate_5d.comp,
// interpolate_nearest_5d_backward.comp) match the CPU reference.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;

namespace {

class VulkanInterpolateNearest5D : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
    static bool has_vulkan() {
        try {
            auto t = zeros({1}, DType::Float32, Device::vulkan(0));
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

TEST_F(VulkanInterpolateNearest5D, Forward_MatchesCPU) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";

    // Upsample 2x2x2 -> 4x4x4 in nearest mode, volumetric (N, C, D, H, W).
    auto input_cpu = seq_f32({1, 2, 2, 2, 2});

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSize, std::string("4,4,4"));
    attrs.set(AttrKey::Mode, "nearest");
    attrs.set(AttrKey::AlignCorners, false);

    Tensor cpu_out = dispatch(OpId::Interpolate, std::vector<Tensor>{input_cpu}, attrs)[0];

    auto input_vulkan = input_cpu.to(Device::vulkan(0));
    Tensor vulkan_out = dispatch(OpId::Interpolate, std::vector<Tensor>{input_vulkan}, attrs)[0]
                             .to(Device::cpu());

    ASSERT_EQ(cpu_out.shape().size(), vulkan_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], vulkan_out.shape()[i]) << " dim " << i;
    }
    ASSERT_EQ(cpu_out.numel(), vulkan_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* vp = vulkan_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], vp[i], 5e-5f) << " elem " << i;
    }
}

TEST_F(VulkanInterpolateNearest5D, Backward_MatchesCPU) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";

    // Backward scatters gradient from each 4x4x4 output voxel to its 2x2x2
    // input voxel.
    auto grad_out_cpu = seq_f32({1, 2, 4, 4, 4});

    OpAttributes attrs;
    // InputShape is stored as a comma-separated string (see function_shape.cpp).
    attrs.set(AttrKey::InputShape, "2,2,2");
    attrs.set(AttrKey::Mode, "nearest");
    attrs.set(AttrKey::AlignCorners, false);

    Tensor cpu_out = dispatch(OpId::InterpolateBackward, std::vector<Tensor>{grad_out_cpu}, attrs)[0];

    auto grad_out_vulkan = grad_out_cpu.to(Device::vulkan(0));
    Tensor vulkan_out = dispatch(OpId::InterpolateBackward, std::vector<Tensor>{grad_out_vulkan}, attrs)[0]
                             .to(Device::cpu());

    ASSERT_EQ(cpu_out.shape().size(), vulkan_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], vulkan_out.shape()[i]) << " dim " << i;
    }
    ASSERT_EQ(cpu_out.numel(), vulkan_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* vp = vulkan_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], vp[i], 5e-5f) << " elem " << i;
    }
}

TEST_F(VulkanInterpolateNearest5D, Forward_NonDivisibleSize_MatchesCPU) {
    // Non-power-of-2 / non-evenly-divisible sizes are the case most likely to
    // expose a float-floor-vs-integer-division mismatch between the new
    // Vulkan kernel and the CPU reference.
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";

    auto input_cpu = seq_f32({1, 3, 3, 5, 7});

    OpAttributes attrs;
    attrs.set(AttrKey::OutputSize, std::string("7,11,4"));
    attrs.set(AttrKey::Mode, "nearest");
    attrs.set(AttrKey::AlignCorners, false);

    Tensor cpu_out = dispatch(OpId::Interpolate, std::vector<Tensor>{input_cpu}, attrs)[0];

    auto input_vulkan = input_cpu.to(Device::vulkan(0));
    Tensor vulkan_out = dispatch(OpId::Interpolate, std::vector<Tensor>{input_vulkan}, attrs)[0]
                             .to(Device::cpu());

    ASSERT_EQ(cpu_out.numel(), vulkan_out.numel());
    auto* cp = cpu_out.data<float>();
    auto* vp = vulkan_out.data<float>();
    for (int64_t i = 0; i < cpu_out.numel(); ++i) {
        EXPECT_NEAR(cp[i], vp[i], 5e-5f) << " elem " << i;
    }
}
