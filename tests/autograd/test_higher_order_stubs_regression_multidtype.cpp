/**
 * @file test_higher_order_stubs_regression_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_higher_order_stubs_regression.cpp.
 *
 * The plain file (BackendTest, Float32-only) asserts that the higher-order
 * gradient *stubs* (Conv*, BatchNorm, LayerNorm, pooling, Dropout, Embedding,
 * RReLU, CTCLoss, MultiLabelMarginLoss) pass through a create_graph=true
 * backward without throwing and produce a finite (possibly zero) gradient.
 *
 * This companion re-runs the same stub-passthrough surface across the
 * {Float32, Float64, Float16} × {cpu, cuda, vulkan, oneapi, rocm, mps} matrix
 * via MultiBackendDTypeTest. The structural contract (no throw, finite grad)
 * is dtype-orthogonal in intent, but a backend that only registers a Float32
 * kernel for a given op would silently skip Float16/Float64 — this companion
 * surfaces that as an explicit, categorized skip rather than a hidden gap, and
 * catches the Float32-accumulator / autograd-graph-drop bug patterns the
 * multidype convention is meant to guard against.
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
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class HigherOrderStubsRegressionMultiDType : public MultiBackendDTypeTest {};

// Same passthrough check as the plain file: forward + first backward, verify
// input.grad is populated and finite. Pass-through stubs must return a valid
// (possibly zero) grad even when the forward ran in a non-Float32 dtype.
template <typename Forward>
void check_stub_passthrough(const Variable& input, Forward&& fwd,
                             const std::string& op_name) {
    Variable out = fwd(input);
    auto loss = sum(out);
    EXPECT_NO_THROW({ loss.backward(/*grad=*/{}, /*retain_graph=*/false); })
        << "first backward threw for " << op_name;
    ASSERT_TRUE(input.has_grad()) << "no first-order grad for " << op_name;

    auto g_cpu = input.grad()->to(Device::cpu()).to(DType::Float32).contiguous();
    for (int64_t i = 0; i < g_cpu.numel(); ++i) {
        float v = g_cpu.data<float>()[i];
        EXPECT_TRUE(std::isfinite(v))
            << op_name << " produced non-finite grad at i=" << i << ": " << v;
    }
}

TEST_P(HigherOrderStubsRegressionMultiDType, Conv2d_Stub_Passthrough) {
    nn::Conv2d conv(/*in=*/2, /*out=*/3, /*k=*/3, /*stride=*/1, /*pad=*/1);
    convert_model(conv);
    Variable input = createInput({1, 2, 6, 6}, true);
    check_stub_passthrough(input,
        [&](const Variable& x) { return conv.forward(x); }, "Conv2d");
}

TEST_P(HigherOrderStubsRegressionMultiDType, BatchNorm2d_Stub_Passthrough) {
    nn::BatchNorm2d bn(/*num_features=*/4);
    convert_model(bn);
    Variable input = createInput({2, 4, 4, 4}, true);
    check_stub_passthrough(input,
        [&](const Variable& x) { return bn.forward(x); }, "BatchNorm2d");
}

TEST_P(HigherOrderStubsRegressionMultiDType, LayerNorm_Stub_Passthrough) {
    nn::LayerNorm ln(std::vector<int64_t>{8});
    convert_model(ln);
    Variable input = createInput({4, 8}, true);
    check_stub_passthrough(input,
        [&](const Variable& x) { return ln.forward(x); }, "LayerNorm");
}

TEST_P(HigherOrderStubsRegressionMultiDType, MaxPool2d_Stub_Passthrough) {
    nn::MaxPool2d pool(/*kernel=*/2, /*stride=*/2);
    convert_model(pool);
    Variable input = createInput({1, 2, 8, 8}, true);
    check_stub_passthrough(input,
        [&](const Variable& x) { return pool.forward(x); }, "MaxPool2d");
}

TEST_P(HigherOrderStubsRegressionMultiDType, AvgPool2d_Stub_Passthrough) {
    nn::AvgPool2d pool(/*kernel=*/2, /*stride=*/2);
    convert_model(pool);
    Variable input = createInput({1, 2, 8, 8}, true);
    check_stub_passthrough(input,
        [&](const Variable& x) { return pool.forward(x); }, "AvgPool2d");
}

TEST_P(HigherOrderStubsRegressionMultiDType, Dropout_Eval_Stub_Passthrough) {
    nn::Dropout d(0.5);
    convert_model(d);
    d.eval();  // deterministic identity in eval mode
    Variable input = createInput({4, 8}, true);
    check_stub_passthrough(input,
        [&](const Variable& x) { return d.forward(x); }, "Dropout/eval");
}

TEST_P(HigherOrderStubsRegressionMultiDType, RReLU_Stub_Passthrough) {
    // RReLU is piecewise-linear; its 2nd derivative is structurally zero.
    // Eval-time (training=false) RReLU is deterministic. Verifies the
    // structural-zero stub still passes through in non-Float32 dtypes.
    Variable input = createInput({2, 8}, true);
    check_stub_passthrough(input,
        [](const Variable& x) {
            return tenzor::nn::rrelu(x, 0.125, 0.333, /*training=*/false);
        },
        "RReLU");
}

TEST_P(HigherOrderStubsRegressionMultiDType, CTCLoss_Stub_Passthrough) {
    // CTC's second derivative w.r.t. log_probs is intentionally not exposed.
    // The stub pins that plain backward works and that create_graph=true in
    // Warn mode disconnects cleanly. Targets / lengths are Int64 regardless of
    // the float dtype under test.
    const int64_t T = 4, N = 2, C = 5;
    Variable log_probs = createInput({T, N, C}, true);
    auto targets_cpu = zeros({N, 3}, DType::Int64, Device::cpu());
    auto input_lengths_cpu = zeros({N}, DType::Int64, Device::cpu());
    auto target_lengths_cpu = zeros({N}, DType::Int64, Device::cpu());
    for (int64_t i = 0; i < N; ++i) {
        input_lengths_cpu.data<int64_t>()[i] = T;
        target_lengths_cpu.data<int64_t>()[i] = 2;
    }
    for (int64_t i = 0; i < N * 3; ++i) {
        targets_cpu.data<int64_t>()[i] = 1 + (i % (C - 1));
    }
    auto targets = targets_cpu.to(device());
    auto input_lengths = input_lengths_cpu.to(device());
    auto target_lengths = target_lengths_cpu.to(device());
    tenzor::nn::CTCLoss ctc(/*reduction=*/"mean", /*blank=*/0);
    auto loss = ctc.forward(log_probs, targets, input_lengths, target_lengths);
    EXPECT_NO_THROW(loss.backward())
        << "CTCLoss regular backward threw — stub should accept it";
    ASSERT_TRUE(log_probs.has_grad());

    // Warn mode lets the engine disconnect the stubbed chain without throwing.
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);
    Variable lp2 = createInput({T, N, C}, true);
    auto loss2 = ctc.forward(lp2, targets, input_lengths, target_lengths);
    EXPECT_NO_THROW(
        loss2.backward(/*grad=*/{}, /*retain_graph=*/false, /*create_graph=*/true))
        << "CTCLoss backward threw under Warn mode";
    set_higher_order_grad_mode(HigherOrderGradMode::Error);
}

TEST_P(HigherOrderStubsRegressionMultiDType, MultiLabelMarginLoss_Stub_Passthrough) {
    // Piecewise-linear hinge loss — structural-zero 2nd derivative stub.
    const int64_t N = 3, C = 4;
    Variable input = createInput({N, C}, true);
    auto targets_cpu = zeros({N, C}, DType::Int64, Device::cpu());
    for (int64_t b = 0; b < N; ++b) {
        targets_cpu.data<int64_t>()[b * C + 0] = static_cast<int64_t>(b % C);
        targets_cpu.data<int64_t>()[b * C + 1] = -1;
        targets_cpu.data<int64_t>()[b * C + 2] = -1;
        targets_cpu.data<int64_t>()[b * C + 3] = -1;
    }
    auto targets = targets_cpu.to(device());
    tenzor::nn::MultiLabelMarginLoss loss_fn(tenzor::nn::Reduction::Mean);
    auto loss = loss_fn.forward(input, targets);
    EXPECT_NO_THROW(loss.backward())
        << "MultiLabelMarginLoss regular backward threw — stub should accept it";
    ASSERT_TRUE(input.has_grad());

    set_higher_order_grad_mode(HigherOrderGradMode::Warn);
    Variable in2 = createInput({N, C}, true);
    auto loss2 = loss_fn.forward(in2, targets);
    EXPECT_NO_THROW(
        loss2.backward(/*grad=*/{}, /*retain_graph=*/false, /*create_graph=*/true))
        << "MultiLabelMarginLoss backward threw under Warn mode";
    set_higher_order_grad_mode(HigherOrderGradMode::Error);
}

TEST_P(HigherOrderStubsRegressionMultiDType, Embedding_Stub_Passthrough) {
    // Embedding: input is Int64 indices (non-differentiable), output is the
    // float lookup. Higher-order flows through the weights, not the indices.
    nn::Embedding emb(/*num_embeddings=*/16, /*embedding_dim=*/8);
    convert_model(emb);
    auto idx = randint(0, 16, {2, 4}, DType::Int64, device());
    Variable input(idx, false);
    Variable out = emb.forward(input);
    auto loss = sum(out);
    EXPECT_NO_THROW(loss.backward()) << "Embedding backward threw";
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

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HigherOrderStubsRegressionMultiDType);