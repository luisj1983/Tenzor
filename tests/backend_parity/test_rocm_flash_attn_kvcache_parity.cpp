// test_rocm_flash_attn_kvcache_parity.cpp
//
// JIT-R158/JIT-R159: ROCm's tiled Flash Attention v2 kernel
// (flash_attention_v2_kernel_hip) read ONLY Q's seq_len and passed it as
// BOTH the seq_len_q and seq_len_k kernel-launch arguments, mis-striding/
// truncating the K/V buffer whenever seq_len_q != seq_len_k (KV-cache /
// cross-attention). The per-tile causal pre-skip also lacked the bottom-
// right causal_offset the per-element mask already used, which would have
// incorrectly skipped needed tiles once the seq_len bug was fixed.
//
// This test exercises the OpId::FlashAttention dispatch directly on ROCm
// with head_dim=64 (inside the {32,64,128} tiled-kernel set) and
// seq_len_q != seq_len_k, both causal and non-causal, and verifies parity
// against the CPU reference.

#include <gtest/gtest.h>
#include <cmath>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;

namespace {

class FlashAttentionKvCacheParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

bool has_rocm() {
    try {
        auto t = zeros({1}, DType::Float32, Device::rocm(0));
        (void)t;
        return true;
    } catch (...) {
        return false;
    }
}

auto random_f32(std::vector<int64_t> shape, Device dev) -> Tensor {
    auto cpu = zeros(shape, DType::Float32, Device::cpu());
    auto* p = cpu.data<float>();
    for (int64_t i = 0; i < cpu.numel(); ++i) {
        uint32_t bits = static_cast<uint32_t>(i * 2654435761u);
        p[i] = (static_cast<float>(bits & 0xFFFFu) / 65536.0f) - 0.5f;
    }
    return (dev.type == Device::Type::CPU) ? cpu : cpu.to(dev);
}

void run_kvcache_parity(int64_t seq_len_q, int64_t seq_len_k, bool causal) {
    if (!has_rocm()) {
        GTEST_SKIP() << "ROCm not available";
    }

    // head_dim=64 is inside the tiled-kernel {32,64,128} set on ROCm.
    constexpr int64_t kHeadDim = 64;
    auto Q_cpu = random_f32({1, 2, seq_len_q, kHeadDim}, Device::cpu());
    auto K_cpu = random_f32({1, 2, seq_len_k, kHeadDim}, Device::cpu());
    auto V_cpu = random_f32({1, 2, seq_len_k, kHeadDim}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(kHeadDim))));
    attrs.set(AttrKey::Causal, causal);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    std::vector<Tensor> cpu_inputs = {Q_cpu, K_cpu, V_cpu};
    auto cpu_outs = dispatch(OpId::FlashAttention, cpu_inputs, attrs);
    ASSERT_GE(cpu_outs.size(), 1u);
    Tensor cpu_out = cpu_outs[0];

    auto Q_r = Q_cpu.to(Device::rocm(0));
    auto K_r = K_cpu.to(Device::rocm(0));
    auto V_r = V_cpu.to(Device::rocm(0));
    std::vector<Tensor> rocm_inputs = {Q_r, K_r, V_r};
    auto rocm_outs = dispatch(OpId::FlashAttention, rocm_inputs, attrs);
    ASSERT_GE(rocm_outs.size(), 1u);
    Tensor rocm_out = rocm_outs[0].to(Device::cpu());

    ASSERT_EQ(cpu_out.shape().size(), rocm_out.shape().size());
    for (size_t i = 0; i < cpu_out.shape().size(); ++i) {
        EXPECT_EQ(cpu_out.shape()[i], rocm_out.shape()[i])
            << " dim " << i << " (seq_len_q=" << seq_len_q
            << ", seq_len_k=" << seq_len_k << ", causal=" << causal << ")";
    }
    auto* cp = cpu_out.data<float>();
    auto* rp = rocm_out.data<float>();
    int64_t n = cpu_out.numel();
    for (int64_t i = 0; i < n; ++i) {
        EXPECT_NEAR(cp[i], rp[i], 5e-4f)
            << " elem " << i << " (seq_len_q=" << seq_len_q
            << ", seq_len_k=" << seq_len_k << ", causal=" << causal << ")";
    }
}

}  // namespace

// Self-attention (seq_len_q == seq_len_k): the pre-existing, already-correct
// case (causal_offset == 0) -- a no-regression baseline.
TEST_F(FlashAttentionKvCacheParity, SelfAttention_Causal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/32, /*seq_len_k=*/32, /*causal=*/true);
}

// KV-cache decode: a single new query attending over a much longer growing
// cache (seq_len_q << seq_len_k). This is exactly the JIT-R158 failure mode:
// pre-fix, ROCm's tiled kernel used seq_len_q for BOTH seq_len_q and
// seq_len_k, so K/V (actually seq_len_k=64 long) were read/strided as if
// only seq_len_q=1 long.
TEST_F(FlashAttentionKvCacheParity, KvCacheDecode_Causal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/1, /*seq_len_k=*/64, /*causal=*/true);
}

// KV-cache prefill-continuation: a chunk of new queries attending over a
// longer existing cache (seq_len_q < seq_len_k, both > 1). Exercises the
// JIT-R159 tile-skip causal_offset fix specifically (with seq_len_q=1 the
// tile-skip's kv_start > q_row check with q_row==0 never distinguishes an
// offset bug the way a multi-row query chunk does).
TEST_F(FlashAttentionKvCacheParity, KvCacheChunk_Causal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/16, /*seq_len_k=*/64, /*causal=*/true);
}

// Non-causal KV-cache cross-attention: no causal mask at all, so this
// isolates the JIT-R158 seq_len bug alone (no causal_offset involved).
TEST_F(FlashAttentionKvCacheParity, KvCacheChunk_NonCausal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/16, /*seq_len_k=*/64, /*causal=*/false);
}
