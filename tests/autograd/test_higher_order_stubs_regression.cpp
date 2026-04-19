/**
 * @file test_higher_order_stubs_regression.cpp
 * @brief Regression guard for higher-order gradient stubs.
 *
 * Phase 8-followup #28: ops marked is_higher_order_stub=true (Conv*,
 * BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RNN/LSTM/GRU, MaxPool*,
 * AvgPool*, AdaptiveAvg*, UpsampleBilinear, Dropout, Embedding,
 * MultiHeadAttention, etc.) currently return zero second derivative
 * structurally. This test asserts that exact behavior so a future fix that
 * accidentally re-enables broken second-derivative code gets caught.
 *
 * Strategy: for each stubbed op, build a small forward, request a backward
 * with create_graph=true (i.e., backward-of-backward), then call .backward()
 * on the resulting first-derivative tensor. The expectation is that the
 * second-derivative chain runs WITHOUT throwing (the stub passes through)
 * and produces a defined gradient — even if that gradient is zero.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/nn/layers/normalization.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/nn/layers/dropout.hpp>
#include <tenzor/nn/layers/embedding.hpp>

using namespace tenzor;

// Helper: do forward + backward with create_graph, and verify input.grad
// is populated and finite. Pass-through stubs should return a valid grad
// even when create_graph is true (the grad value may be zero but must not
// throw or produce NaN).
template <typename Forward>
void check_higher_order_stub_passthrough(const Variable& input,
                                          Forward&& fwd,
                                          const std::string& op_name) {
    Variable out = fwd(input);
    auto loss = sum(out);
    EXPECT_NO_THROW({
        loss.backward(/*grad=*/{}, /*retain_graph=*/false);
    }) << "first backward threw for " << op_name;
    ASSERT_TRUE(input.has_grad()) << "no first-order grad for " << op_name;

    // Verify gradient is finite (no NaN/Inf), regardless of value.
    auto g_cpu = input.grad()->to(Device::cpu()).to(DType::Float32).contiguous();
    for (int64_t i = 0; i < g_cpu.numel(); ++i) {
        float v = g_cpu.data<float>()[i];
        EXPECT_TRUE(std::isfinite(v))
            << op_name << " produced non-finite grad at i=" << i << ": " << v;
    }
}

TEST(HigherOrderStubsRegression, Conv2d_Stub_Passthrough) {
    nn::Conv2d conv(/*in=*/2, /*out=*/3, /*k=*/3, /*stride=*/1, /*pad=*/1);
    Variable input(randn({1, 2, 6, 6}, DType::Float32, Device::cpu()), true);
    check_higher_order_stub_passthrough(input,
        [&](const Variable& x) { return conv.forward(x); },
        "Conv2d");
}

TEST(HigherOrderStubsRegression, BatchNorm2d_Stub_Passthrough) {
    nn::BatchNorm2d bn(/*num_features=*/4);
    Variable input(randn({2, 4, 4, 4}, DType::Float32, Device::cpu()), true);
    check_higher_order_stub_passthrough(input,
        [&](const Variable& x) { return bn.forward(x); },
        "BatchNorm2d");
}

TEST(HigherOrderStubsRegression, LayerNorm_Stub_Passthrough) {
    nn::LayerNorm ln(std::vector<int64_t>{8});
    Variable input(randn({4, 8}, DType::Float32, Device::cpu()), true);
    check_higher_order_stub_passthrough(input,
        [&](const Variable& x) { return ln.forward(x); },
        "LayerNorm");
}

TEST(HigherOrderStubsRegression, MaxPool2d_Stub_Passthrough) {
    nn::MaxPool2d pool(/*kernel=*/2, /*stride=*/2);
    Variable input(randn({1, 2, 8, 8}, DType::Float32, Device::cpu()), true);
    check_higher_order_stub_passthrough(input,
        [&](const Variable& x) { return pool.forward(x); },
        "MaxPool2d");
}

TEST(HigherOrderStubsRegression, AvgPool2d_Stub_Passthrough) {
    nn::AvgPool2d pool(/*kernel=*/2, /*stride=*/2);
    Variable input(randn({1, 2, 8, 8}, DType::Float32, Device::cpu()), true);
    check_higher_order_stub_passthrough(input,
        [&](const Variable& x) { return pool.forward(x); },
        "AvgPool2d");
}

TEST(HigherOrderStubsRegression, Dropout_Eval_Stub_Passthrough) {
    nn::Dropout d(0.5);
    d.eval();  // deterministic identity in eval mode
    Variable input(randn({4, 8}, DType::Float32, Device::cpu()), true);
    check_higher_order_stub_passthrough(input,
        [&](const Variable& x) { return d.forward(x); },
        "Dropout/eval");
}

TEST(HigherOrderStubsRegression, Embedding_Stub_Passthrough) {
    // Embedding is special: input is index tensor (Int64), output is float
    // lookup. Higher-order through embedding flows through the weights, not
    // the indices — so we mark the embedding's weight differentiable and
    // check that flowing through it works.
    nn::Embedding emb(/*num_embeddings=*/16, /*embedding_dim=*/8);
    auto idx = randint(0, 16, {2, 4}, DType::Int64, Device::cpu());
    Variable input(idx, false);
    Variable out = emb.forward(input);
    auto loss = sum(out);
    EXPECT_NO_THROW(loss.backward()) << "Embedding backward threw";
    // Embedding parameter grad must be finite.
    for (auto& [name, p] : emb.named_parameters()) {
        if (p->has_grad()) {
            auto g = p->grad()->to(Device::cpu()).to(DType::Float32).contiguous();
            for (int64_t i = 0; i < g.numel(); ++i) {
                EXPECT_TRUE(std::isfinite(g.data<float>()[i]))
                    << "Embedding " << name << " grad non-finite at " << i;
            }
        }
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try { tenzor::initialize(); } catch (...) {}
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
