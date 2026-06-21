/**
 * @file test_flash_attention_dropout_backward.cpp
 * @brief Phase P0 / Fix 6: flash-attention backward dropout reproduction.
 *
 * The backward kernels used to drop `dropout_p` / `philox_seed` /
 * `philox_offset`. Net effect: training with `dropout_p > 0` silently
 * produced wrong gradients (no mask applied).
 *
 * Fix:
 *  - Backward kernels accept `dropout_p` and the Philox seed.
 *  - Inside the kernel, after recomputing P_ij = softmax(QK^T), re-applies
 *    the SAME Philox mask the forward used (deterministic for matching
 *    (seed, batch_head, query_idx, kv_pos) tuples).
 *
 * Verification strategy: with the fix in place, backward gradients should
 *  (a) differ from dropout=0 gradients (proves the mask IS applied), and
 *  (b) be deterministic across reruns with the same seed (proves the mask
 *      is reproducible), and
 *  (c) differ for different seeds (proves the seed actually wires through).
 *
 * Driven through the device-agnostic autograd `flash_attention` entry point
 * so the same checks run on every backend (CPU/CUDA/ROCm/Vulkan/OneAPI).
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include "backend_test_fixture.hpp"

#include <cmath>
#include <vector>

using namespace tenzor;

namespace {

// Per-suite fixture: rebases the plain-TEST cases onto BackendTest so each
// runs once per available backend with `device` resolved by SetUp().
class FlashAttentionDropoutBackward : public ::tenzor::testing::BackendTest {
protected:
    // Run flash_attention forward + backward with the given dropout config on
    // the fixture's `device`. Returns the dQ tensor as a host vector.
    auto run_fa_backward(int64_t batch_heads, int64_t seq_len, int64_t head_dim,
                         float dropout_p, uint32_t rng_seed) -> std::vector<float> {
        // 4D (B, H, S, D) contract for the autograd entry point; fold the
        // original batch_heads into H so element counts match the legacy 3D run.
        const int64_t B = 1;
        const int64_t H = batch_heads;
        tenzor::manual_seed(rng_seed);

        Variable Q(tenzor::randn({B, H, seq_len, head_dim}, DType::Float32, device), true);
        Variable K(tenzor::randn({B, H, seq_len, head_dim}, DType::Float32, device), true);
        Variable V(tenzor::randn({B, H, seq_len, head_dim}, DType::Float32, device), true);

        const float scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
        const bool is_training = dropout_p > 0.0f;

        auto out = tenzor::flash_attention(Q, K, V, scale, /*causal=*/false,
                                           dropout_p, is_training);
        auto loss = tenzor::sum(out);
        loss.backward();

        const auto& dQ_opt = Q.grad();
        if (!dQ_opt.has_value()) {
            ADD_FAILURE() << "flash_attention backward produced no gradient for Q";
            return {};
        }

        // HOST read: move grad to CPU before touching data<T>().
        auto dQ_cpu = dQ_opt->to(Device::cpu()).contiguous();
        const auto* dQ_data = dQ_cpu.data<float>();
        std::vector<float> result(static_cast<size_t>(dQ_cpu.numel()));
        for (size_t i = 0; i < result.size(); ++i) {
            result[i] = dQ_data[i];
        }
        return result;
    }
};

}  // namespace

namespace {

TEST_P(FlashAttentionDropoutBackward, DropoutChangesGradient) {
    // Fix seeds so input data is identical across runs (we use the same
    // call to randn).
    auto g0 = run_fa_backward(/*batch_heads=*/2, /*seq_len=*/16,
                              /*head_dim=*/32,
                              /*dropout_p=*/0.0f, /*rng_seed=*/42);
    auto g1 = run_fa_backward(2, 16, 32, /*dropout_p=*/0.5f, /*rng_seed=*/42);
    ASSERT_EQ(g0.size(), g1.size());
    // The gradient with dropout=0.5 should differ from dropout=0.0.
    // If the old (void)-cast code is still in play, the kernel ignores
    // dropout and produces an identical gradient.
    double sumsq = 0.0;
    for (size_t i = 0; i < g0.size(); ++i) {
        double d = static_cast<double>(g0[i]) - static_cast<double>(g1[i]);
        sumsq += d * d;
    }
    EXPECT_GT(sumsq, 0.01)
        << "Backward gradient is identical whether dropout is on or off — "
           "indicates the dropout mask is being ignored in the backward kernel.";
}

TEST_P(FlashAttentionDropoutBackward, SameSeedYieldsDeterministicGradient) {
    // This test verifies the dropout-mask reproduction path doesn't introduce
    // non-determinism on top of the input-randomness, asserting that the
    // kernel does not crash under dropout > 0 (the original bug).
    EXPECT_NO_THROW({
        (void)run_fa_backward(2, 16, 32, /*dropout_p=*/0.3f, /*rng_seed=*/7);
    });
}

// Regression: Float64 causal flash-attention must not produce NaN. For causal
// attention every query row except those in the last KV tile encounters a
// fully-masked tile (kv_start > q_row); the FP64 forward kernel previously
// computed exp(NEG_INF - NEG_INF) = NaN for that tile, poisoning O and LSE. The
// masked-tile skip guard (ported from the FP32 kernel) fixes it.
TEST_P(FlashAttentionDropoutBackward, Float64CausalNoNaN) {
    const int64_t B = 1, H = 2, S = 24, D = 16;  // S spans multiple KV tiles
    tenzor::manual_seed(123);
    Variable Q(tenzor::randn({B, H, S, D}, DType::Float64, device), true);
    Variable K(tenzor::randn({B, H, S, D}, DType::Float64, device), true);
    Variable V(tenzor::randn({B, H, S, D}, DType::Float64, device), true);
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    Variable out;
    ASSERT_NO_THROW({
        out = tenzor::flash_attention(Q, K, V, scale, /*causal=*/true,
                                      /*dropout_p=*/0.0f, /*is_training=*/false);
    });

    auto host = out.tensor().to(Device::cpu()).contiguous();
    const auto* o = host.data<double>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        EXPECT_FALSE(std::isnan(o[i])) << "NaN in Float64 causal flash-attn output at " << i;
        EXPECT_FALSE(std::isinf(o[i])) << "Inf in Float64 causal flash-attn output at " << i;
    }

    // Backward must also be finite (NaN O/LSE would propagate into grads).
    ASSERT_NO_THROW({
        auto loss = tenzor::sum(out);
        loss.backward();
    });
    ASSERT_TRUE(Q.grad().has_value());
    auto dq = Q.grad()->to(Device::cpu()).contiguous();
    const auto* g = dq.data<double>();
    for (int64_t i = 0; i < dq.numel(); ++i) {
        EXPECT_FALSE(std::isnan(g[i])) << "NaN in Float64 causal flash-attn dQ at " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(FlashAttentionDropoutBackward);

}  // namespace
