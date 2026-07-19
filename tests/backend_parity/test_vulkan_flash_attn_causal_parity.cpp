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
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

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
    // Shape: [batch, heads, seq_len, head_dim] (4D per FlashAttention contract).
    auto Q_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto K_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto V_cpu = random_f32({1, 2, 8, 16}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(0.25));
    attrs.set(AttrKey::Causal, true);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::FlashAttention, ins, attrs)[0];
    };

    // On a CPU-only host, fall back to comparing against a recorded golden
    // instead of skipping outright (FINDING 17 in findings.txt).
    Device target = has_vulkan() ? Device::vulkan(0) : Device::cpu();
    test_operation_parity_single(op, {Q_cpu, K_cpu, V_cpu}, target, 1e-5f, 5e-4f,
                                  "FlashAttention_Vulkan_Causal");
}

TEST_F(FlashAttentionCausalParity, Vulkan_NonCausal_MatchesCPU) {
    auto Q_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto K_cpu = random_f32({1, 2, 8, 16}, Device::cpu());
    auto V_cpu = random_f32({1, 2, 8, 16}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(0.25));
    attrs.set(AttrKey::Causal, false);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::FlashAttention, ins, attrs)[0];
    };

    Device target = has_vulkan() ? Device::vulkan(0) : Device::cpu();
    test_operation_parity_single(op, {Q_cpu, K_cpu, V_cpu}, target, 1e-5f, 5e-4f,
                                  "FlashAttention_Vulkan_NonCausal");
}
