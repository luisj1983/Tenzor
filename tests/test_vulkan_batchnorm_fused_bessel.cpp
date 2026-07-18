/**
 * @file test_vulkan_batchnorm_fused_bessel.cpp
 * @brief Regression: Vulkan's OpId::BatchNorm2dFusedTraining passed the raw
 *        biased batch_var straight into dispatchBatchNorm2dUpdateRunningStats
 *        with no Bessel correction, while CPU/CUDA/ROCm/OneAPI all multiply
 *        by N/(N-1) before updating running_var. For batch=2 (N*H*W=2),
 *        Vulkan's running_var was low by 50% relative to every other backend
 *        and to PyTorch semantics, compounding over training steps since
 *        it's an EMA.
 *
 * nn::BatchNorm2d gates the fused path to CUDA only, so this is only
 * reachable via a direct OpId::BatchNorm2dFusedTraining dispatch -- exercise
 * that directly and compare Vulkan's updated running_var against CPU's.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include <tenzor/ops/creation.hpp>

#include <span>

namespace tenzor { void initialize(); }

namespace {
class VulkanBNEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new VulkanBNEnv);

auto vulkan_available() -> bool {
    return tenzor::is_op_supported(tenzor::OpId::BatchNorm2dFusedTraining,
                                    tenzor::Device::Type::Vulkan);
}
}  // namespace

using namespace tenzor;

TEST(VulkanBatchNormFusedBessel, RunningVarMatchesCpuBesselCorrection) {
    if (!vulkan_available()) GTEST_SKIP() << "Vulkan BatchNorm2dFusedTraining not registered";

    // N=2, C=2, H=1, W=1 -- bn_count = N*H*W = 2, so the N/(N-1) correction
    // factor is exactly 2.0. A missing correction would make Vulkan's
    // running_var exactly half of every other backend's, an easy signal to
    // catch even with loose float tolerance.
    constexpr int64_t N = 2, C = 2, H = 1, W = 1;
    auto x_cpu = tenzor::randn({N, C, H, W}, DType::Float32, Device::cpu());
    auto running_mean_cpu = tenzor::zeros({C}, DType::Float32, Device::cpu());
    auto running_var_cpu = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto gamma_cpu = tenzor::ones({C}, DType::Float32, Device::cpu());
    auto beta_cpu = tenzor::zeros({C}, DType::Float32, Device::cpu());

    NewOpAttributes attrs;
    attrs.set(AttrKey::Eps, 1e-5);
    attrs.set(AttrKey::Momentum, 0.1);

    auto x_vk = x_cpu.to(Device::vulkan());
    auto rm_vk = running_mean_cpu.to(Device::vulkan());
    auto rv_vk = running_var_cpu.to(Device::vulkan());
    auto gamma_vk = gamma_cpu.to(Device::vulkan());
    auto beta_vk = beta_cpu.to(Device::vulkan());
    const Tensor vk_in[5] = {x_vk, rm_vk, rv_vk, gamma_vk, beta_vk};
    auto vk_out = tenzor::dispatch(OpId::BatchNorm2dFusedTraining,
                                   std::span<const Tensor>{vk_in, 5}, attrs);

    const Tensor cpu_in[5] = {x_cpu, running_mean_cpu, running_var_cpu, gamma_cpu, beta_cpu};
    auto cpu_out = tenzor::dispatch(OpId::BatchNorm2dFusedTraining,
                                    std::span<const Tensor>{cpu_in, 5}, attrs);

    // Return contract: {output, updated_running_mean, updated_running_var, batch_mean, batch_var}
    ASSERT_GE(vk_out.size(), 3u);
    ASSERT_GE(cpu_out.size(), 3u);
    auto vk_running_var = vk_out[2].to(Device::cpu()).contiguous();
    auto cpu_running_var = cpu_out[2].to(Device::cpu()).contiguous();
    const float* vv = vk_running_var.data<float>();
    const float* cv = cpu_running_var.data<float>();
    for (int64_t c = 0; c < C; ++c) {
        EXPECT_NEAR(vv[c], cv[c], 1e-4)
            << "Vulkan running_var[" << c << "]=" << vv[c]
            << " vs CPU running_var[" << c << "]=" << cv[c]
            << " -- Vulkan is missing the Bessel (N/(N-1)) correction if this is off by 2x";
    }
}
