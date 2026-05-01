// Attention autograd grad-flow tests.
//
// These would have caught every audit C1-C6 finding (FlashAttention training
// path dead, GQA repeat_kv severed, bias_k_/bias_v_ rebuild dangling-leaf,
// etc.). The CPU MultiheadAttentionTest passes pre-fix because it didn't
// verify gradients flow — it only checked output shapes. Per
// docs/internals/attention-contract.md M9 gates.

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/engine.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/layers/gqa_attention.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::nn;

namespace {

class AttentionAutogradTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

// --- M2: tenzor::flash_attention apply helper attaches grad_fn ---
//
// Audit C1: nn::functional::scaled_dot_product_attention previously called
// dispatch<OpId::FlashAttention>() raw and wrapped output as Variable(t,
// requires_grad=true) with no grad_fn — silent zero gradients. The new
// FlashAttentionFunction-based apply helper must attach a grad_fn.
TEST_F(AttentionAutogradTest, FlashAttentionAttachesGradFn) {
    // Small Q/K/V so the test is fast on CPU (the only backend guaranteed
    // available; this test runs on whatever backend Tensor lands on).
    auto Q = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), /*requires_grad=*/true);
    auto K = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), /*requires_grad=*/true);
    auto V = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), /*requires_grad=*/true);

    float scale = 1.0f / 4.0f;  // 1/sqrt(head_dim=16)
    auto out = tenzor::flash_attention(Q, K, V, scale, /*causal=*/false,
                                       /*dropout_p=*/0.0f, /*is_training=*/false);

    ASSERT_TRUE(out.requires_grad());
    ASSERT_NE(out.grad_fn(), nullptr)
        << "flash_attention apply helper must attach a grad_fn — was wrapping "
           "output without one (audit C1).";

    auto loss = tenzor::sum(out);
    loss.backward();

    EXPECT_GRAD_FLOWS(Q);
    EXPECT_GRAD_FLOWS(K);
    EXPECT_GRAD_FLOWS(V);
}

TEST_F(AttentionAutogradTest, FusedAttentionAttachesGradFn) {
    auto Q = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), true);
    auto K = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), true);
    auto V = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), true);
    float scale = 1.0f / 4.0f;
    auto out = tenzor::fused_attention(Q, K, V, scale, /*causal=*/false,
                                       /*use_cudnn_sdpa=*/false);
    ASSERT_TRUE(out.requires_grad());
    ASSERT_NE(out.grad_fn(), nullptr);
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(Q);
    EXPECT_GRAD_FLOWS(K);
    EXPECT_GRAD_FLOWS(V);
}

// --- M2: causal flag honored for autograd path ---
TEST_F(AttentionAutogradTest, FlashAttentionCausalGradFlows) {
    auto Q = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), true);
    auto K = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), true);
    auto V = tenzor::Variable(tenzor::randn({2, 4, 8, 16}), true);
    auto out = tenzor::flash_attention(Q, K, V, 1.0f / 4.0f, /*causal=*/true);
    ASSERT_TRUE(out.requires_grad());
    auto loss = tenzor::sum(out);
    loss.backward();
    EXPECT_GRAD_FLOWS(Q);
    EXPECT_GRAD_FLOWS(K);
    EXPECT_GRAD_FLOWS(V);
}

// --- M2: MultiheadAttention bias_k_/bias_v_ shared_ptr aliasing ---
//
// Audit C5/C6: previously bias_k_ was rebuilt as a new Variable on
// device-mismatch, breaking the registered_parameters map alias and
// dropping gradient accumulation on the dangling leaf.
TEST_F(AttentionAutogradTest, MultiheadAttentionBiasKVGradFlows) {
    int64_t embed_dim = 16, num_heads = 4, batch = 2, seq_len = 8;
    auto mha = std::make_shared<tenzor::nn::MultiheadAttention>(
        embed_dim, num_heads, /*dropout=*/0.0,
        /*bias=*/true, /*add_bias_kv=*/true, /*add_zero_attn=*/false);
    mha->train(false);  // eval mode so the BMM path runs
                        // (per attention.cpp's grad_path_safe gating)

    auto x = tenzor::Variable(tenzor::randn({batch, seq_len, embed_dim}), true);
    auto [out, weights] = mha->forward(x, x, x);
    auto loss = tenzor::sum(out);
    loss.backward();

    // The registered parameter must have non-zero grad. Pull the live
    // shared_ptr from the parameter map (this is what gradient accumulation
    // updates per Module::to() semantics).
    auto bias_k_param = mha->get_parameter("bias_k");
    auto bias_v_param = mha->get_parameter("bias_v");
    ASSERT_NE(bias_k_param, nullptr);
    ASSERT_NE(bias_v_param, nullptr);

    // Inline grad-flow check (the EXPECT_GRAD_FLOWS macro's `.template`
    // syntax interacts badly with shared_ptr-deref'd Variables).
    auto check_grad_flows = [](const Variable& v, const char* name) {
        auto opt = v.grad();
        ASSERT_TRUE(opt.has_value()) << name << ": no grad after backward";
        ASSERT_GT(opt.value().numel(), 0) << name << ": empty grad tensor";
        auto g_cpu = opt.value().cpu().to(DType::Float64);
        double g_max = max(abs(g_cpu)).item<double>();
        EXPECT_GT(g_max, 0.0) << name << ": grad is identically zero — "
                                       "grad_fn likely severed (audit C5/C6).";
    };
    check_grad_flows(*bias_k_param, "bias_k");
    check_grad_flows(*bias_v_param, "bias_v");
}

// --- M2: GQA repeat_kv preserves K/V projection grads ---
//
// Audit C3 (the worst memory-cited bug): repeat_kv used raw Tensor ops on
// x.tensor() and rewrapped as Variable, severing the grad_fn chain back to
// k_proj_/v_proj_. Every GQA model with num_heads_per_group > 1 (i.e. every
// real GQA case, not MHA fallback) silently zeroed K/V projection grads.
TEST_F(AttentionAutogradTest, GQARepeatKVPreservesProjGrads) {
    int64_t embed_dim = 16, num_heads = 8, num_kv_heads = 2;  // 4-to-1 GQA
    int64_t batch = 2, seq_len = 8;
    auto gqa = std::make_shared<tenzor::nn::GroupedQueryAttention>(
        embed_dim, num_heads, num_kv_heads);
    gqa->train(true);

    auto x = tenzor::Variable(tenzor::randn({batch, seq_len, embed_dim}), true);
    auto out = gqa->forward(x);
    auto loss = tenzor::sum(out);
    loss.backward();

    // Look up k_proj/v_proj weights via named_parameters() — this walks
    // submodules so we don't need direct submodule accessors.
    std::shared_ptr<Variable> k_w_p, v_w_p;
    for (auto& [name, p] : gqa->named_parameters()) {
        if (name == "k_proj.weight") k_w_p = p;
        else if (name == "v_proj.weight") v_w_p = p;
    }
    ASSERT_NE(k_w_p, nullptr) << "GroupedQueryAttention should expose k_proj.weight";
    ASSERT_NE(v_w_p, nullptr) << "GroupedQueryAttention should expose v_proj.weight";

    auto check_grad_flows = [](const Variable& v, const char* name) {
        auto opt = v.grad();
        ASSERT_TRUE(opt.has_value()) << name;
        auto g_cpu = opt.value().cpu().to(DType::Float64);
        double g_max = max(abs(g_cpu)).item<double>();
        EXPECT_GT(g_max, 0.0) << name << ": grad zero — grad_fn severed (audit C3 GQA repeat_kv).";
    };
    check_grad_flows(*k_w_p, "k_proj.weight");
    check_grad_flows(*v_w_p, "v_proj.weight");
}

// --- M2: is_causal && attn_mask raises (was: silently double-masked) ---
TEST_F(AttentionAutogradTest, MultiheadAttentionRejectsCausalPlusMask) {
    int64_t embed_dim = 16, num_heads = 4, batch = 2, seq_len = 8;
    auto mha = std::make_shared<tenzor::nn::MultiheadAttention>(
        embed_dim, num_heads, /*dropout=*/0.0,
        /*bias=*/true, /*add_bias_kv=*/false, /*add_zero_attn=*/false,
        /*kdim=*/0, /*vdim=*/0, /*batch_first=*/true, /*is_causal=*/true);
    mha->train(false);

    auto x = tenzor::Variable(tenzor::randn({batch, seq_len, embed_dim}), true);
    tenzor::Tensor mask = tenzor::ones({seq_len, seq_len}, tenzor::DType::Float32);

    bool threw = false;
    try {
        auto pair = mha->forward(x, x, x, tenzor::Tensor{}, mask);
        (void)pair;
    } catch (const std::invalid_argument&) {
        threw = true;
    }
    EXPECT_TRUE(threw)
        << "MultiheadAttention must reject is_causal=true && attn_mask "
           "simultaneously per attention-contract.md (audit M1).";
}

}  // anonymous namespace
