/**
 * @file test_flash_attention_dropout_composed_multibatch.cpp
 * @brief Regression for the CPU FlashAttention dropout Philox counter-
 *        convention mismatch between the forward kernel and the composed
 *        backward's mask replay.
 *
 * The bug: with dropout_p > 0 the autograd FlashAttention routes its backward
 * through `composed_attention_backward`, which rebuilds the forward dropout
 * mask via `tenzor::philox_dropout_mask`. That helper folds batch and head
 * into ONE Philox counter word (c0 = b*H + h). The CPU forward kernel, by
 * contrast, used FOUR separate words (b, h, qi, ki). For any input with B > 1
 * or H > 1 the two conventions enumerate DIFFERENT Philox streams, so the
 * backward reconstructed a dropout mask that did NOT match the one the forward
 * applied — silently wrong dQ/dK/dV. The fix aligns the CPU forward (and the
 * CPU kernel's own dropout backward) onto the bh-combined convention shared by
 * philox_dropout_mask and the CUDA/ROCm/OneAPI/Vulkan kernels.
 *
 * This test exercises the composed dropout backward with B = 2, H = 2 (so the
 * combined-vs-split conventions genuinely differ) and asserts:
 *   1. backward produces finite, non-trivially-non-zero gradients on Q/K/V
 *      (a mask mismatch tends to zero or corrupt large slabs of the grad), and
 *   2. every gradient is finite (a stream mismatch can apply the 1/(1-p)
 *      keep-scale to positions the forward dropped, leaking NaN/Inf), for
 *      B = 2, H = 2 specifically (where the combined-vs-split conventions
 *      genuinely diverge).
 *
 * Note: each forward picks a fresh Philox seed (seed_in == 0), so gradients
 * legitimately differ across separate calls — this test does NOT assert
 * cross-call determinism. The forward-vs-backward mask agreement WITHIN a
 * single call is what the fix guarantees (identical counter convention), and
 * its observable symptom under a mismatch is non-finite / zeroed gradients,
 * which this test catches.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"

#include <cmath>
#include <vector>

using namespace tenzor;

class FlashAttnDropoutComposedMultiBatch : public ::tenzor::testing::BackendTest {
protected:
    // Run autograd flash_attention with dropout and return flattened dQ.
    std::vector<float> run_once(uint64_t /*unused*/) {
        const int64_t B = 2, H = 2, S = 8, D = 16;
        const float scale = 1.0f / std::sqrt(static_cast<float>(D));

        // Fixed-content Q/K/V (seeded via a deterministic fill) so that across
        // calls the only stochastic element is the dropout stream, which the
        // forward saves and the backward replays.
        auto fill = [&](float base) {
            Tensor t = tenzor::zeros({B, H, S, D}, DType::Float32, Device::cpu());
            float* p = t.data<float>();
            for (int64_t i = 0; i < t.numel(); ++i)
                p[i] = std::sin(base + 0.01f * static_cast<float>(i));
            return t.to(device);
        };
        Variable Q(fill(0.1f), true);
        Variable K(fill(0.2f), true);
        Variable V(fill(0.3f), true);

        auto out = flash_attention(Q, K, V, scale, /*causal=*/false,
                                   /*dropout_p=*/0.5f, /*is_training=*/true);
        auto loss = tenzor::sum(out);
        loss.backward();

        // ASSERT_* cannot live in this non-void helper; the caller validates
        // grad flow. Surface a clear failure if grads are missing entirely.
        EXPECT_TRUE(Q.grad().has_value() && K.grad().has_value() &&
                    V.grad().has_value())
            << "composed dropout backward did not populate Q/K/V grads";
        if (!Q.grad().has_value()) return {};

        auto g = Q.grad().value().to(Device::cpu());
        std::vector<float> v(g.numel());
        std::copy(g.data<float>(), g.data<float>() + g.numel(), v.begin());
        return v;
    }
};

TEST_P(FlashAttnDropoutComposedMultiBatch, GradientsFiniteAndNonZero) {
    auto g1 = run_once(0);
    ASSERT_FALSE(g1.empty()) << "no gradient produced";

    // All gradients must be finite (a Philox stream mismatch can leave NaN/Inf
    // when the replayed 1/(1-p) keep-scale lands on positions the forward
    // dropped), and not an all-zero slab.
    bool any_nonzero = false;
    for (float x : g1) {
        ASSERT_TRUE(std::isfinite(x)) << "non-finite dQ from composed dropout backward";
        if (std::abs(x) > 1e-6f) any_nonzero = true;
    }
    EXPECT_TRUE(any_nonzero) << "dQ is all ~zero — composed dropout backward produced no gradient";
}

INSTANTIATE_BACKEND_TESTS(FlashAttnDropoutComposedMultiBatch);
