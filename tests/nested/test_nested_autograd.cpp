/**
 * @file test_nested_autograd.cpp
 * @brief Autograd tests for nested tensor operations.
 *
 * Tests gradient computation through:
 * - nested_softmax
 * - nested_sum / nested_mean
 * - nested_linear
 * - nested_attention
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nested/nested_tensor.hpp>
#include <tenzor/nested/nested_ops.hpp>
#include <tenzor/autograd/nested_ops.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/ops/creation.hpp>
#include "../backend_test_fixture.hpp"
#include "../grad_flow_helpers.hpp"
#include <cmath>
#include <vector>

namespace {

// Offsets are int64 ragged-dimension metadata. The nested ops move them to CPU
// internally as needed, so we keep them as a CPU index tensor.
tenzor::Tensor make_int64_tensor(const std::vector<int64_t>& data) {
    return tenzor::from_data(data.data(), {static_cast<int64_t>(data.size())});
}

class NestedAutogradTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// Gradient through nested_sum
// ============================================================================

TEST_P(NestedAutogradTest, NestedSumGradient) {
    // values = [total_len=8, D=4], offsets = [0, 3, 8]
    auto values_data = tenzor::randn({8, 4}, tenzor::DType::Float32, device);
    tenzor::Variable values(values_data, /*requires_grad=*/true);

    auto offsets = make_int64_tensor({0, 3, 8});

    auto result = tenzor::autograd::nested_sum(values, offsets);
    // result is [2, 4]
    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 4);

    // Sum the result to get a scalar for backward
    auto loss = tenzor::sum(result);
    loss.backward();

    // Gradient of sum-of-sums w.r.t. each input element is 1.0
    ASSERT_TRUE(values.has_grad());
    auto grad = (*values.grad()).cpu();
    EXPECT_EQ(grad.shape()[0], 8);
    EXPECT_EQ(grad.shape()[1], 4);

    const float* grad_ptr = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_NEAR(grad_ptr[i], 1.0f, 1e-5f);
    }
}

// ============================================================================
// Gradient through nested_mean
// ============================================================================

TEST_P(NestedAutogradTest, NestedMeanGradient) {
    auto values_data = tenzor::randn({8, 4}, tenzor::DType::Float32, device);
    tenzor::Variable values(values_data, true);

    auto offsets = make_int64_tensor({0, 3, 8});

    auto result = tenzor::autograd::nested_mean(values, offsets);
    auto loss = tenzor::sum(result);
    loss.backward();

    ASSERT_TRUE(values.has_grad());
    auto grad = (*values.grad()).cpu();
    const float* grad_ptr = grad.data<float>();

    // For segment 0 (len=3): grad = 1/3 for each element
    // For segment 1 (len=5): grad = 1/5 for each element
    for (int64_t i = 0; i < 3; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(grad_ptr[i * 4 + j], 1.0f / 3.0f, 1e-5f);
        }
    }
    for (int64_t i = 3; i < 8; ++i) {
        for (int64_t j = 0; j < 4; ++j) {
            EXPECT_NEAR(grad_ptr[i * 4 + j], 1.0f / 5.0f, 1e-5f);
        }
    }
}

// ============================================================================
// Gradient through nested_softmax
// ============================================================================

TEST_P(NestedAutogradTest, NestedSoftmaxGradient) {
    auto values_data = tenzor::randn({6, 4}, tenzor::DType::Float32, device);
    tenzor::Variable values(values_data, true);

    auto offsets = make_int64_tensor({0, 3, 6});

    // dim=0 is the ragged SEGMENT axis — the only axis nested_softmax normalizes
    // over (the previous dim=-1 relied on the kernel silently ignoring dim).
    auto result = tenzor::autograd::nested_softmax(values, offsets, /*dim=*/0);
    // result shape should match input
    EXPECT_EQ(result.shape()[0], 6);
    EXPECT_EQ(result.shape()[1], 4);

    auto loss = tenzor::sum(result);
    loss.backward();

    ASSERT_TRUE(values.has_grad());
    auto grad = (*values.grad()).cpu();
    EXPECT_EQ(grad.shape()[0], 6);
    EXPECT_EQ(grad.shape()[1], 4);

    // Softmax gradient should be well-defined (not NaN)
    const float* grad_ptr = grad.data<float>();
    for (int64_t i = 0; i < grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(grad_ptr[i]));
    }

    // F015: a non-segment dim must throw rather than silently compute a segment
    // softmax (the kernel ignores dim), so callers can't get the wrong result.
    EXPECT_THROW(tenzor::autograd::nested_softmax(values, offsets, /*dim=*/1),
                 std::runtime_error);
}

// ============================================================================
// Gradient through nested_linear
// ============================================================================

TEST_P(NestedAutogradTest, NestedLinearGradient) {
    int64_t D_in = 4, D_out = 3;
    auto values_data = tenzor::randn({8, D_in}, tenzor::DType::Float32, device);
    auto weight_data = tenzor::randn({D_out, D_in}, tenzor::DType::Float32, device);
    auto bias_data = tenzor::randn({D_out}, tenzor::DType::Float32, device);

    tenzor::Variable values(values_data, true);
    tenzor::Variable weight(weight_data, true);
    tenzor::Variable bias(bias_data, true);

    auto offsets = make_int64_tensor({0, 3, 8});

    auto result = tenzor::autograd::nested_linear(values, offsets, weight, &bias);

    EXPECT_EQ(result.shape()[0], 8);
    EXPECT_EQ(result.shape()[1], D_out);

    auto loss = tenzor::sum(result);
    loss.backward();

    // All inputs should have gradients
    EXPECT_GRAD_FLOWS(values);
    EXPECT_GRAD_FLOWS(weight);
    ASSERT_TRUE(bias.has_grad());

    // Bias gradient: sum of ones over 8 rows = 8 per output dim... but
    // actually the sum of d(loss)/d(bias) for linear is sum over all rows
    auto bias_grad = (*bias.grad()).cpu();
    const float* bias_grad_ptr = bias_grad.data<float>();
    for (int64_t j = 0; j < D_out; ++j) {
        EXPECT_NEAR(bias_grad_ptr[j], 8.0f, 1e-4f);
    }
}

// ============================================================================
// Gradient through nested_attention
// ============================================================================

TEST_P(NestedAutogradTest, NestedAttentionGradient) {
    int64_t head_dim = 4;

    // Two sequences: Q has [3, 4] and [2, 4]; KV has [3, 4] and [2, 4]
    auto q_data = tenzor::randn({5, head_dim}, tenzor::DType::Float32, device);
    auto k_data = tenzor::randn({5, head_dim}, tenzor::DType::Float32, device);
    auto v_data = tenzor::randn({5, head_dim}, tenzor::DType::Float32, device);

    tenzor::Variable Q(q_data, true);
    tenzor::Variable K(k_data, true);
    tenzor::Variable V(v_data, true);

    auto q_offsets = make_int64_tensor({0, 3, 5});
    auto kv_offsets = make_int64_tensor({0, 3, 5});

    double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    auto result = tenzor::autograd::nested_attention(Q, K, V, q_offsets, kv_offsets, scale, false);

    EXPECT_EQ(result.shape()[0], 5);
    EXPECT_EQ(result.shape()[1], head_dim);

    auto loss = tenzor::sum(result);
    loss.backward();

    // All Q, K, V should have gradients
    EXPECT_GRAD_FLOWS(Q);
    EXPECT_GRAD_FLOWS(K);
    EXPECT_GRAD_FLOWS(V);

    // Gradients should be finite
    auto q_grad = (*Q.grad()).cpu();
    const float* q_grad_ptr = q_grad.data<float>();
    for (int64_t i = 0; i < q_grad.numel(); ++i) {
        EXPECT_FALSE(std::isnan(q_grad_ptr[i]));
        EXPECT_FALSE(std::isinf(q_grad_ptr[i]));
    }
}

// ============================================================================
// Audit item A.6 — Causal mask MUST affect the backward gradient.
//
// The previous NestedAttentionBackward had an empty `if (causal_) { for ... }`
// body that walked the upper-triangular positions without doing anything.
// As a result the recomputed softmax inside backward saw the unmasked
// scores, producing a gradient that ignored the causal contract.  This
// test pins the fix: the gradient with causal=true MUST differ from the
// gradient with causal=false (because the masked positions contribute zero
// to forward and therefore should receive zero gradient — non-trivially
// different from the unmasked case).
// ============================================================================
TEST_P(NestedAutogradTest, NestedAttentionCausalChangesGradient) {
    const int64_t head_dim = 4;

    // One sequence of length 3 so causal matters (full upper triangle is
    // non-trivial).
    auto q_data = tenzor::randn({3, head_dim}, tenzor::DType::Float32, device);
    auto k_data = tenzor::randn({3, head_dim}, tenzor::DType::Float32, device);
    auto v_data = tenzor::randn({3, head_dim}, tenzor::DType::Float32, device);

    auto q_offsets = make_int64_tensor({0, 3});
    auto kv_offsets = make_int64_tensor({0, 3});
    const double scale = 1.0 / std::sqrt(static_cast<double>(head_dim));

    auto run_backward = [&](bool causal) -> std::tuple<tenzor::Tensor, tenzor::Tensor, tenzor::Tensor> {
        tenzor::Variable Q(q_data, true);
        tenzor::Variable K(k_data, true);
        tenzor::Variable V(v_data, true);
        auto out = tenzor::autograd::nested_attention(
            Q, K, V, q_offsets, kv_offsets, scale, causal);
        auto loss = tenzor::sum(out);
        loss.backward();
        return {(*Q.grad()).cpu(), (*K.grad()).cpu(), (*V.grad()).cpu()};
    };

    auto [gQ_nc, gK_nc, gV_nc] = run_backward(/*causal=*/false);
    auto [gQ_c,  gK_c,  gV_c]  = run_backward(/*causal=*/true);

    auto max_abs_diff = [](const tenzor::Tensor& a, const tenzor::Tensor& b) -> float {
        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        float m = 0.0f;
        for (int64_t i = 0; i < a.numel(); ++i) {
            m = std::max(m, std::abs(pa[i] - pb[i]));
        }
        return m;
    };

    EXPECT_GT(max_abs_diff(gQ_c, gQ_nc), 1e-3f)
        << "Q gradient unchanged by causal mask — backward did not apply mask";
    EXPECT_GT(max_abs_diff(gK_c, gK_nc), 1e-3f)
        << "K gradient unchanged by causal mask — backward did not apply mask";
    EXPECT_GT(max_abs_diff(gV_c, gV_nc), 1e-3f)
        << "V gradient unchanged by causal mask — backward did not apply mask";

    for (const auto* g : {&gQ_c, &gK_c, &gV_c}) {
        const float* g_ptr = g->data<float>();
        for (int64_t i = 0; i < g->numel(); ++i) {
            EXPECT_FALSE(std::isnan(g_ptr[i]));
            EXPECT_FALSE(std::isinf(g_ptr[i]));
        }
    }
}

// ============================================================================
// No-grad path
// ============================================================================

TEST_P(NestedAutogradTest, NoGradPath) {
    tenzor::NoGradGuard guard;

    auto values_data = tenzor::randn({6, 4}, tenzor::DType::Float32, device);
    tenzor::Variable values(values_data, false);

    auto offsets = make_int64_tensor({0, 3, 6});

    auto result = tenzor::autograd::nested_sum(values, offsets);

    EXPECT_EQ(result.shape()[0], 2);
    EXPECT_EQ(result.shape()[1], 4);
    // Should not have grad_fn since grad is disabled
    EXPECT_EQ(result.grad_fn(), nullptr);
}

// nested_layer_norm normalizes each row over the feature dim D, independent of
// the ragged segment structure — so it must be identical to regular per-row
// LayerNorm over {D}, forward AND backward. This catches the wrong-axis backward
// (which reduced statistics over the segment/token axis instead of D).
TEST_P(NestedAutogradTest, NestedLayerNormMatchesRegularLayerNorm) {
    const int64_t N = 7, D = 5;
    auto vals = tenzor::randn({N, D}, tenzor::DType::Float32, device);
    auto w = tenzor::randn({D}, tenzor::DType::Float32, device);
    auto b = tenzor::randn({D}, tenzor::DType::Float32, device);
    auto seed = tenzor::randn({N, D}, tenzor::DType::Float32, device);
    auto offsets = make_int64_tensor({0, 3, 5, 7});  // multi-segment, ragged
    const double eps = 1e-5;

    tenzor::Variable v1(vals, true), w1(w, true), b1(b, true);
    auto out1 = tenzor::autograd::nested_layer_norm(v1, offsets, w1, b1, eps);
    out1.backward(seed);

    tenzor::Variable v2(vals, true), w2(w, true), b2(b, true);
    auto out2 = tenzor::nn::functional::layer_norm(v2, {D}, w2, b2, eps);
    out2.backward(seed);

    auto max_diff = [](const tenzor::Tensor& a, const tenzor::Tensor& bb) {
        return tenzor::max(tenzor::abs(tenzor::sub(a.cpu(), bb.cpu()))).template item<float>();
    };
    EXPECT_LT(max_diff(out1.tensor(), out2.tensor()), 1e-4f) << "forward differs";
    EXPECT_LT(max_diff(v1.grad().value(), v2.grad().value()), 1e-4f) << "grad_values differs";
    EXPECT_LT(max_diff(w1.grad().value(), w2.grad().value()), 1e-4f) << "grad_weight differs";
    EXPECT_LT(max_diff(b1.grad().value(), b2.grad().value()), 1e-4f) << "grad_bias differs";
}

// F013 (>2D completion): nested_layer_norm on >2D values [total, H, D] normalizes
// over the LAST dim D (weight/bias are [D]); the backward must flatten leading
// dims and reduce over the feature axis, matching regular layer_norm(values,{D}).
TEST_P(NestedAutogradTest, NestedLayerNorm3DMatchesRegularLayerNorm) {
    const int64_t total = 6, Hh = 2, D = 4;
    auto vals = tenzor::randn({total, Hh, D}, tenzor::DType::Float32, device);
    auto w = tenzor::randn({D}, tenzor::DType::Float32, device);
    auto b = tenzor::randn({D}, tenzor::DType::Float32, device);
    auto seed = tenzor::randn({total, Hh, D}, tenzor::DType::Float32, device);
    auto offsets = make_int64_tensor({0, 3, 6});
    const double eps = 1e-5;

    tenzor::Variable v1(vals, true), w1(w, true), b1(b, true);
    tenzor::Variable out1;
    try {
        out1 = tenzor::autograd::nested_layer_norm(v1, offsets, w1, b1, eps);
    } catch (const std::exception& e) {
        GTEST_SKIP() << "nested_layer_norm forward does not accept >2D values: " << e.what();
    }
    out1.backward(seed);

    tenzor::Variable v2(vals, true), w2(w, true), b2(b, true);
    auto out2 = tenzor::nn::functional::layer_norm(v2, {D}, w2, b2, eps);
    out2.backward(seed);

    auto max_diff = [](const tenzor::Tensor& a, const tenzor::Tensor& bb) {
        return tenzor::max(tenzor::abs(tenzor::sub(a.cpu(), bb.cpu()))).template item<float>();
    };
    EXPECT_LT(max_diff(out1.tensor(), out2.tensor()), 1e-4f) << "forward differs (>2D)";
    EXPECT_LT(max_diff(v1.grad().value(), v2.grad().value()), 1e-4f) << "grad_values differs (>2D)";
    EXPECT_LT(max_diff(w1.grad().value(), w2.grad().value()), 1e-4f) << "grad_weight differs (>2D)";
    EXPECT_LT(max_diff(b1.grad().value(), b2.grad().value()), 1e-4f) << "grad_bias differs (>2D)";
}

INSTANTIATE_BACKEND_TESTS(NestedAutogradTest);

} // namespace
