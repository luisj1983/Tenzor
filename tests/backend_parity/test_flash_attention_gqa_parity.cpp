// test_flash_attention_gqa_parity.cpp
//
// JIT-R160/JIT-R161: a direct OpId::FlashAttention dispatch with H_kv < H_q
// (GQA/MQA, no pre-broadcast by the caller) previously behaved inconsistently
// across backends for the IDENTICAL call:
//   - CUDA/ROCm: correct (K/V broadcast to H_q internally before the kernel).
//   - CPU: silently WRONG (kv_bh_offset indexed with Q's head count, reading
//     past the K/V buffer -- no error, just corrupted output).
//   - OneAPI/Vulkan: threw a reshape-element-count-mismatch error instead of
//     computing the (perfectly valid) GQA-shaped attention.
//
// This test exercises OpId::FlashAttention directly (bypassing
// nn::GroupedQueryAttention's own repeat_kv, which already worked around this
// by pre-broadcasting) with H_kv < H_q on every available backend and
// verifies all agree with each other.
//
// JIT-R162 (a distinct Vulkan-only bug in OpId::FusedAttention, not
// OpId::FlashAttention): a non-divisible group size (H_q % H_kv != 0)
// silently skipped the K/V repeat/broadcast instead of throwing, unlike
// every other backend. Covered separately below.

#include <gtest/gtest.h>
#include <cmath>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;

namespace {

auto random_f32(std::vector<int64_t> shape, Device dev) -> Tensor {
    auto cpu = zeros(shape, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i * 2654435761u);
        p[i] = (static_cast<float>(bits & 0xFFFFu) / 65536.0f) - 0.5f;
    }
    return (dev.type == Device::Type::CPU) ? cpu : cpu.to(dev);
}

bool device_available(Device dev) {
    try {
        auto t = zeros({1}, DType::Float32, dev);
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

std::string device_name(const Device& d) {
    switch (d.type) {
        case Device::Type::CPU: return "cpu";
        case Device::Type::CUDA: return "cuda";
        case Device::Type::ROCm: return "rocm";
        case Device::Type::Vulkan: return "vulkan";
        case Device::Type::OneAPI: return "oneapi";
        default: return "other";
    }
}

class FlashAttentionGqaParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

}  // namespace

// batch=1, H_q=4, H_kv=2 (group size 2), seq_len=8, head_dim=16.
TEST_F(FlashAttentionGqaParity, DirectGqaCallAgreesAcrossBackends) {
    constexpr int64_t kBatch = 1, kHq = 4, kHkv = 2, kSeq = 8, kDim = 16;
    auto Q_cpu = random_f32({kBatch, kHq, kSeq, kDim}, Device::cpu());
    auto K_cpu = random_f32({kBatch, kHkv, kSeq, kDim}, Device::cpu());
    auto V_cpu = random_f32({kBatch, kHkv, kSeq, kDim}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(kDim))));
    attrs.set(AttrKey::Causal, true);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    std::vector<Tensor> cpu_inputs = {Q_cpu, K_cpu, V_cpu};
    Tensor reference;
    std::string reference_backend;
    std::vector<Device> devices = {Device::cpu(), Device::cuda(0), Device::rocm(0),
                                    Device::vulkan(0), Device::oneapi(0)};
    std::vector<std::pair<std::string, Tensor>> results;

    for (const auto& dev : devices) {
        if (!device_available(dev)) continue;
        try {
            auto Q_d = Q_cpu.to(dev);
            auto K_d = K_cpu.to(dev);
            auto V_d = V_cpu.to(dev);
            std::vector<Tensor> ins = {Q_d, K_d, V_d};
            auto outs = dispatch(OpId::FlashAttention, ins, attrs);
            ASSERT_GE(outs.size(), 1u) << device_name(dev);
            results.emplace_back(device_name(dev), outs[0].to(Device::cpu()));
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Direct GQA FlashAttention threw on " << device_name(dev)
                          << ": " << e.what();
        }
    }

    ASSERT_GE(results.size(), 2u) << "need at least 2 backends to compare";
    const auto& [ref_name, ref_out] = results[0];
    for (size_t i = 1; i < results.size(); ++i) {
        const auto& [name, out] = results[i];
        ASSERT_EQ(ref_out.shape().size(), out.shape().size())
            << ref_name << " vs " << name;
        for (size_t d = 0; d < ref_out.shape().size(); ++d) {
            EXPECT_EQ(ref_out.shape()[d], out.shape()[d])
                << ref_name << " vs " << name << " dim " << d;
        }
        auto* rp = ref_out.data<float>();
        auto* op = out.data<float>();
        for (int64_t k = 0; k < ref_out.numel(); ++k) {
            EXPECT_NEAR(rp[k], op[k], 5e-3f)
                << ref_name << " vs " << name << " elem " << k;
        }
    }
}

// JIT-R162: Vulkan's OpId::FusedAttention with a non-divisible group size
// (H_q % H_kv != 0) must throw a clear error, not silently proceed.
TEST_F(FlashAttentionGqaParity, VulkanFusedAttentionRejectsNonDivisibleGroupSize) {
    if (!device_available(Device::vulkan(0))) {
        GTEST_SKIP() << "Vulkan not available";
    }
    constexpr int64_t kBatch = 1, kHq = 5, kHkv = 2, kSeq = 4, kDim = 8;  // 5 % 2 != 0
    auto Q = random_f32({kBatch, kHq, kSeq, kDim}, Device::vulkan(0));
    auto K = random_f32({kBatch, kHkv, kSeq, kDim}, Device::vulkan(0));
    auto V = random_f32({kBatch, kHkv, kSeq, kDim}, Device::vulkan(0));

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(kDim))));
    attrs.set(AttrKey::Causal, false);

    std::vector<Tensor> inputs = {Q, K, V};
    EXPECT_THROW(
        { auto outs = dispatch(OpId::FusedAttention, inputs, attrs); },
        std::exception);
}
