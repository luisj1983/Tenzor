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
#include <limits>
#include <tenzor/tenzor.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>
#include "parity_test_utils.hpp"

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

// JIT-R178: honors TENZOR_SKIP_BACKENDS (unlike the old ad hoc
// device_available()), so an explicit opt-out for a known-flaky driver on
// this host is actually respected by this file.
bool device_available(Device dev) {
    return tenzor::testing::is_backend_available(dev.type, dev.index);
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

// JIT-R191: independent ground truth for causal GQA/MQA attention, computed
// directly from raw pointers in double precision -- does NOT call
// OpId::FlashAttention, tenzor::matmul, or tenzor::softmax, so it cannot
// share a bug with any backend's attention kernel (a systemic group-mapping
// error common to every backend would previously have passed undetected
// when the reference was just "whichever backend ran first").
// Head-grouping convention matches replay_gqa_repeat_kv /
// dispatch_gqa (Hkv-major, G-minor): query head hq maps to kv head
// hq / (Hq/Hkv). Causal alignment is bottom-right: kj > qi + (Sk - Sq) is
// masked out, matching every backend's flash_attention forward.
std::vector<float> naive_gqa_attention_reference(
    const float* Q, const float* K, const float* V,
    int64_t B, int64_t Hq, int64_t Hkv, int64_t Sq, int64_t Sk, int64_t D,
    double scale, bool causal) {
    const int64_t reps = Hq / Hkv;
    const int64_t causal_offset = Sk - Sq;
    std::vector<float> out(static_cast<size_t>(B * Hq * Sq * D), 0.0f);
    std::vector<double> scores(static_cast<size_t>(Sk));
    for (int64_t b = 0; b < B; ++b) {
        for (int64_t hq = 0; hq < Hq; ++hq) {
            const int64_t hkv = hq / reps;
            for (int64_t qi = 0; qi < Sq; ++qi) {
                const float* q_row = Q + ((b * Hq + hq) * Sq + qi) * D;
                double max_score = -std::numeric_limits<double>::infinity();
                for (int64_t kj = 0; kj < Sk; ++kj) {
                    if (causal && kj > qi + causal_offset) {
                        scores[static_cast<size_t>(kj)] = -std::numeric_limits<double>::infinity();
                        continue;
                    }
                    const float* k_row = K + ((b * Hkv + hkv) * Sk + kj) * D;
                    double dot = 0.0;
                    for (int64_t d = 0; d < D; ++d) {
                        dot += static_cast<double>(q_row[d]) * static_cast<double>(k_row[d]);
                    }
                    double s = dot * scale;
                    scores[static_cast<size_t>(kj)] = s;
                    if (s > max_score) max_score = s;
                }
                double sum = 0.0;
                for (int64_t kj = 0; kj < Sk; ++kj) {
                    double w = std::exp(scores[static_cast<size_t>(kj)] - max_score);
                    scores[static_cast<size_t>(kj)] = w;
                    sum += w;
                }
                float* out_row = out.data() + ((b * Hq + hq) * Sq + qi) * D;
                for (int64_t kj = 0; kj < Sk; ++kj) {
                    double w = scores[static_cast<size_t>(kj)] / sum;
                    if (w == 0.0) continue;
                    const float* v_row = V + ((b * Hkv + hkv) * Sk + kj) * D;
                    for (int64_t d = 0; d < D; ++d) {
                        out_row[d] += static_cast<float>(w * static_cast<double>(v_row[d]));
                    }
                }
            }
        }
    }
    return out;
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
                                    Device::vulkan(0), Device::oneapi(0), Device::mps(0)};
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

    if (results.empty()) {
        if (tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "No backend produced a result for GQA FlashAttention parity check "
                       "(TENZOR_REQUIRE_MULTI_BACKEND=1)";
        }
        GTEST_SKIP() << "No backend available to test GQA FlashAttention parity";
    }

    // JIT-R191: compare every available backend against an independent,
    // op-dispatch-free ground truth (not against whichever backend happened
    // to run first) -- a systemic group-mapping error shared by every
    // backend would otherwise pass undetected.
    auto ref_vec = naive_gqa_attention_reference(
        Q_cpu.data<float>(), K_cpu.data<float>(), V_cpu.data<float>(),
        kBatch, kHq, kHkv, kSeq, kSeq, kDim,
        1.0 / std::sqrt(static_cast<double>(kDim)), /*causal=*/true);

    for (const auto& [name, out] : results) {
        ASSERT_EQ(out.numel(), static_cast<int64_t>(ref_vec.size())) << name;
        auto* op = out.data<float>();
        for (int64_t k = 0; k < out.numel(); ++k) {
            EXPECT_NEAR(ref_vec[static_cast<size_t>(k)], op[k], 5e-3f)
                << name << " vs naive reference, elem " << k;
        }
    }
}

// H21: flash_attention_backward_typed/_half (CPU) indexed K/V/dK/dV using
// Q's OWN head count with no awareness that K/V may have fewer heads
// (GQA/MQA, H_kv < H_q) — an out-of-bounds heap read/write into K/V/dK/dV
// for any un-broadcast GQA-shaped call (forward already broadcasts K/V
// internally, so this was reachable the moment a caller passed genuinely
// un-broadcast GQA-shaped K/V straight into the differentiable
// tenzor::flash_attention() Variable API, not pre-expanded via repeat_kv).
// The fix broadcasts K/V to Q's head count before dispatch (mirroring the
// forward wrapper) and reduces dK/dV back down afterward. Verify this
// GQA-shaped direct call matches an independently-computed reference: call
// the SAME backward with K/V manually pre-expanded to H_q heads (a trivial,
// always-safe H_kv==H_q call), then manually reduce dK/dV by summing each
// repeated head group — this is exactly what the fix does internally, so
// agreement here confirms both "no OOB crash/corruption" and "correct
// values", not just the former.
TEST_F(FlashAttentionGqaParity, CpuBackwardGqaMatchesManuallyExpandedReference) {
    constexpr int64_t kBatch = 2, kHq = 4, kHkv = 2, kReps = kHq / kHkv;
    constexpr int64_t kSeq = 6, kDim = 8;
    auto Q = random_f32({kBatch, kHq, kSeq, kDim}, Device::cpu());
    auto K = random_f32({kBatch, kHkv, kSeq, kDim}, Device::cpu());
    auto V = random_f32({kBatch, kHkv, kSeq, kDim}, Device::cpu());
    auto dO = random_f32({kBatch, kHq, kSeq, kDim}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(kDim))));
    attrs.set(AttrKey::Causal, false);
    attrs.set(AttrKey::DropoutP, 0.0);
    attrs.set(AttrKey::IsTraining, false);

    // Direct GQA-shaped call — exercises the fix's internal broadcast+reduce.
    std::vector<Tensor> fwd_ins = {Q, K, V};
    auto fwd_out = dispatch(OpId::FlashAttention, fwd_ins, attrs);
    ASSERT_GE(fwd_out.size(), 1u);
    std::vector<Tensor> bwd_ins = {dO, Q, K, V, fwd_out[0]};
    auto gqa_grads = dispatch(OpId::FlashAttentionBackward, bwd_ins, attrs);
    ASSERT_GE(gqa_grads.size(), 3u);

    // Reference: manually pre-expand K/V to H_q heads (repeat_kv), so the
    // backward's own GQA path is never triggered (H_kv == H_q trivially).
    auto Ku = K.unsqueeze(2);
    auto Vu = V.unsqueeze(2);
    auto Ke = Ku.expand({kBatch, kHkv, kReps, kSeq, kDim}).contiguous()
                 .reshape({kBatch, kHq, kSeq, kDim});
    auto Ve = Vu.expand({kBatch, kHkv, kReps, kSeq, kDim}).contiguous()
                 .reshape({kBatch, kHq, kSeq, kDim});
    std::vector<Tensor> fwd_ins_ref = {Q, Ke, Ve};
    auto fwd_out_ref = dispatch(OpId::FlashAttention, fwd_ins_ref, attrs);
    ASSERT_GE(fwd_out_ref.size(), 1u);
    std::vector<Tensor> bwd_ins_ref = {dO, Q, Ke, Ve, fwd_out_ref[0]};
    auto full_grads = dispatch(OpId::FlashAttentionBackward, bwd_ins_ref, attrs);
    ASSERT_GE(full_grads.size(), 3u);

    // dQ needs no reduction — compare directly.
    ASSERT_EQ(gqa_grads[0].numel(), full_grads[0].numel());
    {
        auto* a = gqa_grads[0].data<float>();
        auto* b = full_grads[0].data<float>();
        for (int64_t i = 0; i < gqa_grads[0].numel(); ++i) {
            EXPECT_NEAR(a[i], b[i], 5e-4f) << "dQ elem " << i;
        }
    }

    // dK/dV: reduce the full-head reference down to H_kv heads by summing
    // each repeated group, then compare against the GQA path's own result.
    auto dK_ref = tenzor::sum(full_grads[1].reshape({kBatch, kHkv, kReps, kSeq, kDim}), 2, false);
    auto dV_ref = tenzor::sum(full_grads[2].reshape({kBatch, kHkv, kReps, kSeq, kDim}), 2, false);
    ASSERT_EQ(gqa_grads[1].numel(), dK_ref.numel());
    ASSERT_EQ(gqa_grads[2].numel(), dV_ref.numel());
    {
        auto* a = gqa_grads[1].data<float>();
        auto* b = dK_ref.data<float>();
        for (int64_t i = 0; i < gqa_grads[1].numel(); ++i) {
            EXPECT_NEAR(a[i], b[i], 5e-4f) << "dK elem " << i;
        }
    }
    {
        auto* a = gqa_grads[2].data<float>();
        auto* b = dV_ref.data<float>();
        for (int64_t i = 0; i < gqa_grads[2].numel(); ++i) {
            EXPECT_NEAR(a[i], b[i], 5e-4f) << "dV elem " << i;
        }
    }
}

// JIT-R169/JIT-R170: OneAPI and Vulkan's FlashAttention dropout+training
// composed-ops branch previously bypassed the GQA K/V head-broadcast that
// the non-dropout fast path applies, so H_kv < H_q (real GQA, not MQA) threw
// "Shapes are not broadcastable" instead of computing attention. This test
// exercises exactly that combination (dropout_p > 0, is_training = true,
// H_kv=2 < H_q=4) on every backend that implements a real dropout path and
// verifies: (a) it no longer throws, (b) all four GPU backends -- CUDA,
// ROCm, OneAPI, and Vulkan all implement the identical Philox4x32-10
// counter-based RNG (see the "same algorithm as CPU"/"CUDA device port of
// CPU Philox4x32" comments in fused_ops.cu and fused_ops.hip.cpp) -- produce
// numerically consistent output for the same fixed RNG seed. CUDA and ROCm
// were previously left out of this list even though they qualify per the
// test's own stated scope ("every backend that implements a real dropout
// path" — both register OpId::FlashAttention with real DropoutP/IsTraining
// handling).
TEST_F(FlashAttentionGqaParity, DropoutTrainingGqaDoesNotThrowAndAgreesAcrossBackends) {
    constexpr int64_t kBatch = 1, kHq = 4, kHkv = 2, kSeq = 8, kDim = 16;
    auto Q_cpu = random_f32({kBatch, kHq, kSeq, kDim}, Device::cpu());
    auto K_cpu = random_f32({kBatch, kHkv, kSeq, kDim}, Device::cpu());
    auto V_cpu = random_f32({kBatch, kHkv, kSeq, kDim}, Device::cpu());

    OpAttributes attrs;
    attrs.set(AttrKey::Scale, static_cast<double>(1.0 / std::sqrt(static_cast<double>(kDim))));
    attrs.set(AttrKey::Causal, true);
    attrs.set(AttrKey::DropoutP, 0.2);
    attrs.set(AttrKey::IsTraining, true);
    attrs.set(AttrKey::Seed, static_cast<int64_t>(0x5EED));

    // MPS intentionally excluded here (unlike the other GQA FlashAttention
    // tests in this file): this cross-check compares bitwise-identical RNG
    // output under the shared Philox dropout convention, and the MPS backend
    // has no Philox implementation (uses Metal's own RNG), so including it
    // would fail on a real implementation difference, not a bug.
    std::vector<Device> devices = {Device::cuda(0), Device::rocm(0),
                                    Device::oneapi(0), Device::vulkan(0)};
    std::vector<std::pair<std::string, Tensor>> results;

    for (const auto& dev : devices) {
        if (!device_available(dev)) continue;
        auto Q_d = Q_cpu.to(dev);
        auto K_d = K_cpu.to(dev);
        auto V_d = V_cpu.to(dev);
        std::vector<Tensor> ins = {Q_d, K_d, V_d};
        std::vector<Tensor> outs;
        EXPECT_NO_THROW({ outs = dispatch(OpId::FlashAttention, ins, attrs); })
            << "Dropout+training GQA FlashAttention on " << device_name(dev)
            << " must not throw (JIT-R169/JIT-R170 regression)";
        if (outs.empty()) continue;
        ASSERT_EQ(outs[0].shape().size(), 4u) << device_name(dev);
        EXPECT_EQ(outs[0].shape()[0], kBatch) << device_name(dev);
        EXPECT_EQ(outs[0].shape()[1], kHq) << device_name(dev);
        EXPECT_EQ(outs[0].shape()[2], kSeq) << device_name(dev);
        EXPECT_EQ(outs[0].shape()[3], kDim) << device_name(dev);
        results.emplace_back(device_name(dev), outs[0].to(Device::cpu()));
    }

    if (results.size() < 2) {
        GTEST_SKIP() << "Need at least 2 of {CUDA, ROCm, OneAPI, Vulkan} to "
                        "cross-check the shared Philox dropout convention; only "
                     << results.size() << " available";
    }
    // Compare every available backend pair against the first (not just a
    // single fixed pair) — with up to 4 backends now in the list, an
    // ADD_FAILURE per mismatching pair (not FAIL, which would abort the
    // loop) ensures a divergence on e.g. ROCm doesn't hide a separate one
    // on Vulkan in the same run.
    const auto& [ref_name, ref_out] = results[0];
    auto* rp = ref_out.data<float>();
    for (size_t i = 1; i < results.size(); ++i) {
        const auto& [name, out] = results[i];
        auto* op = out.data<float>();
        for (int64_t k = 0; k < ref_out.numel(); ++k) {
            if (std::abs(rp[k] - op[k]) > 5e-3f) {
                ADD_FAILURE() << ref_name << " vs " << name << " elem " << k
                              << ": " << rp[k] << " vs " << op[k];
            }
        }
    }
}

// JIT-R162: Vulkan's OpId::FusedAttention with a non-divisible group size
// (H_q % H_kv != 0) must throw a clear error, not silently proceed.
TEST_F(FlashAttentionGqaParity, VulkanFusedAttentionRejectsNonDivisibleGroupSize) {
    REQUIRE_BACKEND_OR_SKIP("vulkan");
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

// JIT-R197: src/backends/mps/kernels/mps_attention.mm's dead-code
// mps_flash_attention_forward now applies the same GQA/MQA K/V head-broadcast
// (unsqueeze(2)->expand->contiguous->reshape) as the CUDA/ROCm/OneAPI/Vulkan
// OpId::FlashAttention registrations above. That file is Objective-C++
// (#import <Metal/Metal.h>) and cannot be compiled or run on this Linux
// machine -- there is no Apple/Metal SDK available, so no MPS-specific gtest
// binary can exist here. This test isolates and re-executes the EXACT same
// Tensor-level expression sequence against plain CPU tensors (no MPS/Metal
// involved at all) to verify the formula itself is correct: it must match
// nn::GroupedQueryAttention::repeat_kv's canonical Variable-level broadcast
// (src/nn/layers/gqa_attention.cpp) element-for-element. This is the
// strongest verification physically possible for that fix without an
// Apple/macOS toolchain.
TEST_F(FlashAttentionGqaParity, MpsHostBroadcastFormulaMatchesCanonicalRepeatKv_JIT197) {
    const int64_t b = 2, h_kv = 2, reps = 3, h = h_kv * reps, sk = 5, d = 4, dv = 6;

    Tensor K = tenzor::randn({b, h_kv, sk, d}, DType::Float32, Device::cpu());
    Tensor V = tenzor::randn({b, h_kv, sk, dv}, DType::Float32, Device::cpu());

    // ---- verbatim copy of mps_attention.mm's mps_flash_attention_forward
    // GQA broadcast block ----
    Tensor Kc = K.is_contiguous() ? K : K.contiguous();
    Tensor Vc = V.is_contiguous() ? V : V.contiguous();
    Tensor Ku = Kc.unsqueeze(2);
    Tensor Vu = Vc.unsqueeze(2);
    Tensor Ke = Ku.expand({b, h_kv, reps, sk, d});
    Tensor Ve = Vu.expand({b, h_kv, reps, sk, dv});
    Tensor Kb = Ke.contiguous().reshape({b, h, sk, d});
    Tensor Vb = Ve.contiguous().reshape({b, h, sk, dv});
    // ---- end verbatim copy ----

    ASSERT_EQ(Kb.shape()[0], b);
    ASSERT_EQ(Kb.shape()[1], h);
    ASSERT_EQ(Kb.shape()[2], sk);
    ASSERT_EQ(Kb.shape()[3], d);
    ASSERT_EQ(Vb.shape()[1], h);
    ASSERT_EQ(Vb.shape()[3], dv);

    const float* k_src = K.data<float>();
    const float* v_src = V.data<float>();
    const float* k_out = Kb.data<float>();
    const float* v_out = Vb.data<float>();

    // Reference: nn::GroupedQueryAttention::repeat_kv's row-major
    // (h_kv, reps) -> h collapse means flat head index `hi` maps to
    // kv-head `hi / reps` -- each kv head's block repeats `reps` times
    // consecutively (kv0,kv0,kv0, kv1,kv1,kv1, ...).
    for (int64_t bi = 0; bi < b; ++bi) {
        for (int64_t hi = 0; hi < h; ++hi) {
            int64_t kv_head = hi / reps;
            for (int64_t si = 0; si < sk; ++si) {
                for (int64_t di = 0; di < d; ++di) {
                    int64_t src_idx = ((bi * h_kv + kv_head) * sk + si) * d + di;
                    int64_t out_idx = ((bi * h + hi) * sk + si) * d + di;
                    EXPECT_FLOAT_EQ(k_out[out_idx], k_src[src_idx])
                        << "K mismatch b=" << bi << " h=" << hi << " s=" << si << " d=" << di;
                }
                for (int64_t di = 0; di < dv; ++di) {
                    int64_t src_idx = ((bi * h_kv + kv_head) * sk + si) * dv + di;
                    int64_t out_idx = ((bi * h + hi) * sk + si) * dv + di;
                    EXPECT_FLOAT_EQ(v_out[out_idx], v_src[src_idx])
                        << "V mismatch b=" << bi << " h=" << hi << " s=" << si << " d=" << di;
                }
            }
        }
    }

    // Note: nn::GroupedQueryAttention::repeat_kv (src/nn/layers/gqa_attention.cpp)
    // implements the identical formula but is a private member, so it isn't
    // directly callable here; the manual element-by-element reference above
    // (matching its documented row-major (h_kv, reps)->h collapse convention)
    // serves as the independent ground truth instead.
}
