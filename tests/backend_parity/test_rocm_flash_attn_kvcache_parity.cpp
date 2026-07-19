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

#include "parity_test_utils.hpp"

using namespace tenzor;
using tenzor::testing::test_operation_parity_single;
using tenzor::testing::tensors_close;
using tenzor::testing::max_abs_diff;
namespace golden = tenzor::testing::golden;

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

// JIT-R179: escalate to FAIL() under TENZOR_REQUIRE_MULTI_BACKEND=1 instead
// of a bare skip with no signal -- this is the ONE file whose entire
// purpose is covering ROCm-only bugs (JIT-R158/159); a CI host supposed to
// have ROCm but with a silently-failed driver would otherwise report a
// clean SKIPPED with zero indication this coverage is actually missing.
void require_rocm_or_skip() {
    if (!has_rocm()) {
        if (::tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "ROCm required by TENZOR_REQUIRE_MULTI_BACKEND but not available";
        }
        GTEST_SKIP() << "ROCm not available";
    }
}

// FINDING 17: previously an unconditional GTEST_SKIP() on a host without
// ROCm, with zero golden::* integration — this whole file's coverage
// (JIT-R158/159 ROCm-only bugs) was invisible on any non-ROCm CI shard. On a
// CPU-only host, fall back to comparing against a recorded golden instead.
void run_kvcache_parity(int64_t seq_len_q, int64_t seq_len_k, bool causal,
                         const std::string& test_name) {
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

    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::FlashAttention, ins, attrs)[0];
    };

    Device target = has_rocm() ? Device::rocm(0) : Device::cpu();
    test_operation_parity_single(op, {Q_cpu, K_cpu, V_cpu}, target, 1e-5f, 5e-4f, test_name);
}

// H20: ROCm's fused Float32 FlashAttentionBackward fast path used to pass
// the ORIGINAL 4D [B,H,S,D] tensors straight into flash_attention_backward_hip
// with no reshape, even though that kernel reads Q.shape()[0] directly as
// batch_heads and Q.shape()[1] as seq_len (3D [B*H,S,D] contract) — silently
// reinterpreting the head axis as sequence length and the sequence axis as
// head_dim for any B>1 or H>1 call (i.e. any standard MHA/GQA shape). Compare
// dQ/dK/dV against the CPU reference for B=2,H=4 so the bug (if reintroduced)
// is unambiguous, not masked by a degenerate batch_heads=1 collapse.
// Three-output result (dQ, dK, dV) doesn't fit test_operation_parity_single's
// single-Tensor contract, so wire golden::maybe_record/maybe_load directly —
// one golden per output tensor, keyed by name (FINDING 17).
void run_backward_parity(int64_t B, int64_t H, int64_t seq_len, int64_t head_dim) {
    auto Q_cpu = random_f32({B, H, seq_len, head_dim}, Device::cpu());
    auto K_cpu = random_f32({B, H, seq_len, head_dim}, Device::cpu());
    auto V_cpu = random_f32({B, H, seq_len, head_dim}, Device::cpu());
    auto dO_cpu = random_f32({B, H, seq_len, head_dim}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(head_dim))));
    attrs.set(AttrKey::Causal, false);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    std::vector<Tensor> cpu_fwd_inputs = {Q_cpu, K_cpu, V_cpu};
    auto cpu_fwd = dispatch(OpId::FlashAttention, cpu_fwd_inputs, attrs);
    ASSERT_GE(cpu_fwd.size(), 2u);
    std::vector<Tensor> cpu_bwd_inputs = {dO_cpu, Q_cpu, K_cpu, V_cpu, cpu_fwd[0], cpu_fwd[1]};
    auto cpu_bwd = dispatch(OpId::FlashAttentionBackward, cpu_bwd_inputs, attrs);
    ASSERT_GE(cpu_bwd.size(), 3u);

    const char* names[3] = {"dQ", "dK", "dV"};

    if (!has_rocm()) {
        if (golden::require_multi_backend()) {
            FAIL() << "ROCm required by TENZOR_REQUIRE_MULTI_BACKEND but not available";
        }
        bool any_missing = false;
        for (int idx = 0; idx < 3; ++idx) {
            std::string test_name = std::string("FlashAttentionBackward_StandardMHA_") + names[idx];
            if (auto golden_result = golden::maybe_load(test_name, cpu_bwd_inputs)) {
                golden::note_comparison();
                Tensor cpu_g = cpu_bwd[idx];
                if (!tensors_close(cpu_g, *golden_result, 1e-5f, 5e-3f)) {
                    float max_diff = max_abs_diff(cpu_g, *golden_result);
                    FAIL() << test_name << " golden parity failed: max diff " << max_diff;
                }
            } else {
                any_missing = true;
            }
        }
        if (any_missing) {
            GTEST_SKIP() << "ROCm not available and no recorded golden for one or more of dQ/dK/dV — "
                            "nothing to compare against.";
        }
        return;
    }

    auto Q_r = Q_cpu.to(Device::rocm(0));
    auto K_r = K_cpu.to(Device::rocm(0));
    auto V_r = V_cpu.to(Device::rocm(0));
    auto dO_r = dO_cpu.to(Device::rocm(0));
    std::vector<Tensor> rocm_fwd_inputs = {Q_r, K_r, V_r};
    auto rocm_fwd = dispatch(OpId::FlashAttention, rocm_fwd_inputs, attrs);
    ASSERT_GE(rocm_fwd.size(), 2u);
    std::vector<Tensor> rocm_bwd_inputs = {dO_r, Q_r, K_r, V_r, rocm_fwd[0], rocm_fwd[1]};
    auto rocm_bwd = dispatch(OpId::FlashAttentionBackward, rocm_bwd_inputs, attrs);
    ASSERT_GE(rocm_bwd.size(), 3u);

    for (int idx = 0; idx < 3; ++idx) {
        Tensor cpu_g = cpu_bwd[idx];
        Tensor rocm_g = rocm_bwd[idx].to(Device::cpu());
        ASSERT_EQ(cpu_g.shape().size(), rocm_g.shape().size()) << names[idx];
        for (size_t i = 0; i < cpu_g.shape().size(); ++i) {
            ASSERT_EQ(cpu_g.shape()[i], rocm_g.shape()[i])
                << names[idx] << " dim " << i << " (B=" << B << ",H=" << H
                << ",S=" << seq_len << ",D=" << head_dim << ")";
        }
        auto* cp = cpu_g.data<float>();
        auto* rp = rocm_g.data<float>();
        int64_t n = cpu_g.numel();
        for (int64_t i = 0; i < n; ++i) {
            EXPECT_NEAR(cp[i], rp[i], 5e-3f)
                << names[idx] << " elem " << i << " (B=" << B << ",H=" << H
                << ",S=" << seq_len << ",D=" << head_dim << ")";
        }

        if (golden::recording_enabled()) {
            std::string test_name = std::string("FlashAttentionBackward_StandardMHA_") + names[idx];
            golden::maybe_record(test_name, cpu_bwd_inputs, rocm_g);
        }
    }
}

}  // namespace

TEST_F(FlashAttentionKvCacheParity, Backward_StandardMHA_MatchesCPU) {
    run_backward_parity(/*B=*/2, /*H=*/4, /*seq_len=*/24, /*head_dim=*/64);
}

// Self-attention (seq_len_q == seq_len_k): the pre-existing, already-correct
// case (causal_offset == 0) -- a no-regression baseline.
TEST_F(FlashAttentionKvCacheParity, SelfAttention_Causal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/32, /*seq_len_k=*/32, /*causal=*/true,
                        "FlashAttentionKvCache_SelfAttention_Causal");
}

// KV-cache decode: a single new query attending over a much longer growing
// cache (seq_len_q << seq_len_k). This is exactly the JIT-R158 failure mode:
// pre-fix, ROCm's tiled kernel used seq_len_q for BOTH seq_len_q and
// seq_len_k, so K/V (actually seq_len_k=64 long) were read/strided as if
// only seq_len_q=1 long.
TEST_F(FlashAttentionKvCacheParity, KvCacheDecode_Causal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/1, /*seq_len_k=*/64, /*causal=*/true,
                        "FlashAttentionKvCache_Decode_Causal");
}

// KV-cache prefill-continuation: a chunk of new queries attending over a
// longer existing cache (seq_len_q < seq_len_k, both > 1). Exercises the
// JIT-R159 tile-skip causal_offset fix specifically (with seq_len_q=1 the
// tile-skip's kv_start > q_row check with q_row==0 never distinguishes an
// offset bug the way a multi-row query chunk does).
TEST_F(FlashAttentionKvCacheParity, KvCacheChunk_Causal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/16, /*seq_len_k=*/64, /*causal=*/true,
                        "FlashAttentionKvCache_Chunk_Causal");
}

// Non-causal KV-cache cross-attention: no causal mask at all, so this
// isolates the JIT-R158 seq_len bug alone (no causal_offset involved).
TEST_F(FlashAttentionKvCacheParity, KvCacheChunk_NonCausal_MatchesCPU) {
    run_kvcache_parity(/*seq_len_q=*/16, /*seq_len_k=*/64, /*causal=*/false,
                        "FlashAttentionKvCache_Chunk_NonCausal");
}

// JIT-R183: every case above uses a freshly zeros()-filled, contiguous K/V
// buffer at offset 0 -- not how a REAL KV-cache is used. Production KV-cache
// usage pre-allocates a max-length buffer once and narrows a strided VIEW
// out of it at a non-zero offset as the cache grows, so a stride-ignoring
// kernel bug (this codebase's established recurring signature: a kernel
// that computes strides FROM the logical shape instead of reading the
// tensor's actual strides) would pass every test above yet still corrupt
// output against a genuine cache buffer. This test allocates a
// max_seq_len=128 K/V buffer, narrows out a `seq_len_k`-long window
// starting at a non-zero position (past_len=37) -- giving K/V a non-zero
// storage offset AND per-dimension strides that differ from what a fresh
// contiguous allocation of the narrowed SHAPE would have (the batch/head
// dims still stride by max_seq_len*head_dim, not seq_len_k*head_dim) --
// and verifies ROCm still matches the CPU reference computed over the
// IDENTICAL strided view (not a freshly-copied contiguous CPU tensor,
// which would silently sidestep the bug this test targets).
TEST_F(FlashAttentionKvCacheParity, KvCacheChunk_StridedCacheBuffer_MatchesCPU) {
    constexpr int64_t kHeadDim = 64;
    constexpr int64_t kSeqLenQ = 16;
    constexpr int64_t kSeqLenK = 64;
    constexpr int64_t kPastLen = 37;              // non-zero offset into the cache
    constexpr int64_t kMaxSeqLen = kPastLen + kSeqLenK + 27;  // extra room past what's used

    auto Q_cpu = random_f32({1, 2, kSeqLenQ, kHeadDim}, Device::cpu());
    // Full backing cache buffer, larger than what this call actually reads.
    auto K_cache_cpu = random_f32({1, 2, kMaxSeqLen, kHeadDim}, Device::cpu());
    auto V_cache_cpu = random_f32({1, 2, kMaxSeqLen, kHeadDim}, Device::cpu());
    // Strided, offset VIEW into the cache -- shares storage, does NOT copy.
    // Strides along dims 0/1 remain kMaxSeqLen*kHeadDim-based, not
    // kSeqLenK*kHeadDim-based as a fresh contiguous alloc of this shape
    // would have.
    Tensor K_cpu = K_cache_cpu.narrow(2, kPastLen, kSeqLenK);
    Tensor V_cpu = V_cache_cpu.narrow(2, kPastLen, kSeqLenK);
    ASSERT_EQ(K_cpu.shape()[2], kSeqLenK);

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(kHeadDim))));
    attrs.set(AttrKey::Causal, true);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    // .to(device) on a non-contiguous view must still produce a tensor that,
    // when handed to the ROCm kernel, is either correctly interpreted with
    // its real strides or made contiguous by the transfer -- either way the
    // dispatch below exercises the same logical (strided-source) data the
    // CPU reference used.
    auto op = [&attrs](const std::vector<Tensor>& ins) -> Tensor {
        return dispatch(OpId::FlashAttention, ins, attrs)[0];
    };

    Device target = has_rocm() ? Device::rocm(0) : Device::cpu();
    test_operation_parity_single(op, {Q_cpu, K_cpu, V_cpu}, target, 1e-5f, 5e-4f,
                                  "FlashAttentionKvCache_StridedCacheBuffer");
}
