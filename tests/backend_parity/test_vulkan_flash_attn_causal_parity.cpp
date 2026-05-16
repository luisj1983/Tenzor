// test_vulkan_flash_attn_causal_parity.cpp
//
// Wave D: Vulkan FlashAttention now supports causal masking natively via its
// tiled compute shader (per the shader's own contract). Previously the
// NN attention layer gated this off with `vulkan_causal_supported = false`,
// silently routing Vulkan-causal-inference to the slower composed BMM path.
//
// This test exercises the OpId::FlashAttention dispatch directly with
// causal=true on Vulkan and verifies parity against the CPU reference.

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;

namespace {

class FlashAttentionCausalParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }

    static bool has_vulkan() {
        try {
            auto t = zeros({1}, DType::Float32, Device::vulkan(0));
            (void)t;
            return true;
        } catch (...) {
            return false;
        }
    }
};

static auto random_f32(std::vector<int64_t> shape, Device dev) -> Tensor {
    auto cpu = zeros(shape, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i * 2654435761u);
        p[i] = (static_cast<float>(bits & 0xFFFFu) / 65536.0f) - 0.5f;
    }
    return (dev.type == Device::Type::CPU) ? cpu : cpu.to(dev);
}

}  // namespace

TEST_F(FlashAttentionCausalParity, Vulkan_Causal_MatchesCPU) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";

    // Shape: [batch, heads, seq_len, head_dim] (4D per FlashAttention contract).
    auto Q_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto K_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto V_cpu = random_f32({1, 2, 8, 16}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(0.25));
    attrs.set(AttrKey::Causal, true);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    std::vector<Tensor> cpu_inputs = {Q_cpu, K_cpu, V_cpu};
    auto cpu_outs = dispatch(OpId::FlashAttention, cpu_inputs, attrs);
    ASSERT_GE(cpu_outs.size(), 1u);
    Tensor cpu_out = cpu_outs[0];

    auto Q_v = Q_cpu.to(Device::vulkan(0));
    auto K_v = K_cpu.to(Device::vulkan(0));
    auto V_v = V_cpu.to(Device::vulkan(0));
    std::vector<Tensor> v_inputs = {Q_v, K_v, V_v};
    auto v_outs = dispatch(OpId::FlashAttention, v_inputs, attrs);
    ASSERT_GE(v_outs.size(), 1u);
    Tensor v_out = v_outs[0].to(Device::cpu());

    ASSERT_EQ(cpu_out.shape().size(), v_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], v_out.shape()[i]) << " dim " << i;
    }
    auto* cp = cpu_out.data<float>();
    auto* vp = v_out.data<float>();
    int64_t n = cpu_out.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(cp[i], vp[i], 5e-4f) << " elem " << i;
    }
}

TEST_F(FlashAttentionCausalParity, Vulkan_NonCausal_MatchesCPU) {
    if (!has_vulkan()) GTEST_SKIP() << "Vulkan not available";

    auto Q_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto K_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto V_cpu = random_f32({1, 2, 8, 16}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(0.25));
    attrs.set(AttrKey::Causal, false);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    std::vector<Tensor> cpu_inputs = {Q_cpu, K_cpu, V_cpu};
    Tensor cpu_out = dispatch(OpId::FlashAttention, cpu_inputs, attrs)[0];

    auto Q_v = Q_cpu.to(Device::vulkan(0));
    auto K_v = K_cpu.to(Device::vulkan(0));
    auto V_v = V_cpu.to(Device::vulkan(0));
    std::vector<Tensor> v_inputs = {Q_v, K_v, V_v};
    Tensor v_out = dispatch(OpId::FlashAttention, v_inputs, attrs)[0].to(Device::cpu());

    auto* cp = cpu_out.data<float>();
    auto* vp = v_out.data<float>();
    int64_t n = cpu_out.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(cp[i], vp[i], 5e-4f) << " elem " << i;
    }
}
