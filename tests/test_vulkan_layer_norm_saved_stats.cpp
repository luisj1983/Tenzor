/**
 * @file test_vulkan_layer_norm_saved_stats.cpp
 * @brief Phase P0 / Fix 4: Vulkan LayerNorm forward must return
 *        {output, mean, rstd} and backward must read those saved stats
 *        without crashing.
 *
 * The Vulkan compute shader at src/backends/vulkan/kernels/layer_norm.comp
 * used to compute mean/rstd in shared memory and discard them. The kernel
 * registry returned only `{output}`; the backward registration however
 * indexed `inputs[2]` and `inputs[3]` expecting saved stats → OOB crash.
 * Fix exposes mean/rstd via new SSBO bindings (4, 5) and returns them
 * from dispatchLayerNorm.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/ops/creation.hpp>

#include <span>
#include <vector>

namespace tenzor { void initialize(); }

namespace {
class VulkanLNEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new VulkanLNEnv);

auto vulkan_available() -> bool {
    return tenzor::is_op_supported(tenzor::OpId::LayerNorm, tenzor::Device::Type::Vulkan);
}
}  // namespace

using namespace tenzor;

TEST(VulkanLayerNormSavedStats, ForwardReturnsThreeTensors) {
    if (!vulkan_available()) GTEST_SKIP() << "Vulkan LayerNorm not registered";

    constexpr int64_t B = 4;
    constexpr int64_t N = 16;
    auto x_cpu = tenzor::randn({B, N}, DType::Float32, Device::cpu());
    auto w_cpu = tenzor::ones({N}, DType::Float32, Device::cpu());
    auto b_cpu = tenzor::zeros({N}, DType::Float32, Device::cpu());

    auto x = x_cpu.to(Device::vulkan());
    auto w = w_cpu.to(Device::vulkan());
    auto b = b_cpu.to(Device::vulkan());

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(N));
    attrs.set(AttrKey::Eps, 1e-5);
    const Tensor in_arr[3] = {x, w, b};
    auto out_vec = tenzor::dispatch(OpId::LayerNorm,
                                    std::span<const Tensor>{in_arr, 3}, attrs);
    ASSERT_EQ(out_vec.size(), 3u)
        << "Vulkan LayerNorm forward must return {output, mean, rstd}";
    const auto& output = out_vec[0];
    const auto& mean   = out_vec[1];
    const auto& rstd   = out_vec[2];
    ASSERT_EQ(output.numel(), B * N);
    ASSERT_EQ(mean.numel(), B) << "mean: one scalar per batch element";
    ASSERT_EQ(rstd.numel(), B) << "rstd: one scalar per batch element";

    // Mean and rstd are Float32 per the kernel-registry contract.
    EXPECT_EQ(mean.dtype(), DType::Float32);
    EXPECT_EQ(rstd.dtype(), DType::Float32);
}

TEST(VulkanLayerNormSavedStats, SavedStatsMatchCpuReference) {
    if (!vulkan_available()) GTEST_SKIP() << "Vulkan LayerNorm not registered";

    constexpr int64_t B = 4;
    constexpr int64_t N = 16;
    auto x_cpu = tenzor::randn({B, N}, DType::Float32, Device::cpu());
    auto w_cpu = tenzor::ones({N}, DType::Float32, Device::cpu());
    auto b_cpu = tenzor::zeros({N}, DType::Float32, Device::cpu());
    auto x = x_cpu.to(Device::vulkan());
    auto w = w_cpu.to(Device::vulkan());
    auto b = b_cpu.to(Device::vulkan());

    NewOpAttributes attrs;
    attrs.set(AttrKey::NormalizedShape, std::to_string(N));
    attrs.set(AttrKey::Eps, 1e-5);
    const Tensor vk_in[3] = {x, w, b};
    auto vk_out = tenzor::dispatch(OpId::LayerNorm,
                                   std::span<const Tensor>{vk_in, 3}, attrs);
    const Tensor cpu_in[3] = {x_cpu, w_cpu, b_cpu};
    auto cpu_out = tenzor::dispatch(OpId::LayerNorm,
                                    std::span<const Tensor>{cpu_in, 3}, attrs);

    // Compare per-batch mean.
    auto vk_mean_cpu  = vk_out[1].to(Device::cpu()).contiguous();
    auto cpu_mean_cpu = cpu_out[1].to(Device::cpu()).contiguous();
    const float* vm = vk_mean_cpu.data<float>();
    const float* cm = cpu_mean_cpu.data<float>();
    for (int64_t i = 0; i < B; ++i) {
        EXPECT_NEAR(vm[i], cm[i], 1e-4)
            << "Vulkan vs CPU mean diverges at batch " << i;
    }
    // Compare per-batch rstd.
    auto vk_rstd_cpu  = vk_out[2].to(Device::cpu()).contiguous();
    auto cpu_rstd_cpu = cpu_out[2].to(Device::cpu()).contiguous();
    const float* vr = vk_rstd_cpu.data<float>();
    const float* cr = cpu_rstd_cpu.data<float>();
    for (int64_t i = 0; i < B; ++i) {
        EXPECT_NEAR(vr[i], cr[i], 1e-3)
            << "Vulkan vs CPU rstd diverges at batch " << i;
    }
}
