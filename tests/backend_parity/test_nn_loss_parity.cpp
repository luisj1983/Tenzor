/**
 * @file test_nn_loss_parity.cpp
 * @brief Backend parity tests for neural network loss functions
 *
 * Tests 14 loss functions across all available backends to ensure
 * consistent results regardless of compute device.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/loss/contrastive.hpp>  // InfoNCELoss / NTXentLoss / TripletLoss
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"
#include <tenzor/backend/fast_dispatch.hpp>
#include <tenzor/backend/op_attributes.hpp>
#include <tenzor/ops/op_id.hpp>

using namespace tenzor;
using namespace tenzor::testing;


class NNLossParity : public BackendTest {};
// ============================================================================
// Loss Function Parity Tests
// ============================================================================

TEST_P(NNLossParity, HuberLoss) {

    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    auto target = randn({4, 10}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::HuberLoss loss;
        return loss.forward(Variable(inputs[0], false), Variable(inputs[1], false)).tensor();
    }, {pred, target}, device, 1e-5f, 1e-7f, "HuberLoss");
}

TEST_P(NNLossParity, NLLLoss) {

    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    auto target = (rand({4}, DType::Float32, Device::cpu()) * 10).to(DType::Int64);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::NLLLoss loss;
        auto log_probs = nn::log_softmax(Variable(inputs[0], false), 1);
        return loss.forward(log_probs, inputs[1]).tensor();
    }, {pred, target}, device, 1e-5f, 1e-7f, "NLLLoss");
}

TEST_P(NNLossParity, KLDivLoss) {

    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    auto target_logits = randn({4, 10}, DType::Float32, Device::cpu());
    // Convert target to probabilities via softmax
    auto target = nn::softmax(Variable(target_logits, false), 1).tensor();

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::KLDivLoss loss;
        auto log_probs = nn::log_softmax(Variable(inputs[0], false), 1);
        return loss.forward(log_probs, Variable(inputs[1], false)).tensor();
    }, {pred, target}, device, 1e-4f, 1e-5f, "KLDivLoss");
}

TEST_P(NNLossParity, CosineEmbeddingLoss) {

    auto input1 = randn({4, 32}, DType::Float32, Device::cpu());
    auto input2 = randn({4, 32}, DType::Float32, Device::cpu());
    // Target with +1/-1 values
    auto target_raw = rand({4}, DType::Float32, Device::cpu());
    auto target = sign(target_raw * 2.0f - 1.0f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::CosineEmbeddingLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false),
                          Variable(inputs[2], false)).tensor();
    }, std::vector<Tensor>{input1, input2, target}, device, 1e-5f, 1e-7f, "CosineEmbeddingLoss");
}

TEST_P(NNLossParity, TripletMarginLoss) {

    auto anchor = randn({4, 32}, DType::Float32, Device::cpu());
    auto positive = randn({4, 32}, DType::Float32, Device::cpu());
    auto negative = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::TripletMarginLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false),
                          Variable(inputs[2], false)).tensor();
    }, std::vector<Tensor>{anchor, positive, negative}, device, 1e-5f, 1e-7f, "TripletMarginLoss");
}

TEST_P(NNLossParity, MarginRankingLoss) {

    auto input1 = randn({4}, DType::Float32, Device::cpu());
    auto input2 = randn({4}, DType::Float32, Device::cpu());
    auto target_raw = rand({4}, DType::Float32, Device::cpu());
    auto target = sign(target_raw * 2.0f - 1.0f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::MarginRankingLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false),
                          Variable(inputs[2], false)).tensor();
    }, std::vector<Tensor>{input1, input2, target}, device, 1e-5f, 1e-7f, "MarginRankingLoss");
}

TEST_P(NNLossParity, SoftMarginLoss) {

    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    auto target_raw = rand({4, 10}, DType::Float32, Device::cpu());
    auto target = sign(target_raw * 2.0f - 1.0f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::SoftMarginLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false)).tensor();
    }, {pred, target}, device, 1e-5f, 1e-7f, "SoftMarginLoss");
}

TEST_P(NNLossParity, HingeEmbeddingLoss) {

    auto input = randn({4, 10}, DType::Float32, Device::cpu());
    auto target_raw = rand({4, 10}, DType::Float32, Device::cpu());
    auto target = sign(target_raw * 2.0f - 1.0f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::HingeEmbeddingLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false)).tensor();
    }, {input, target}, device, 1e-5f, 1e-7f, "HingeEmbeddingLoss");
}

TEST_P(NNLossParity, PoissonNLLLoss) {

    // pred is log-rates (any real number)
    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    // target is counts (must be positive)
    auto target = rand({4, 10}, DType::Float32, Device::cpu()) * 5.0f + 0.1f;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::PoissonNLLLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false)).tensor();
    }, {pred, target}, device, 1e-5f, 1e-7f, "PoissonNLLLoss");
}

TEST_P(NNLossParity, CTCLoss) {
    // Audit-5 Y.31: previously the entire test body was wrapped in
    // `try { ... } catch (...) { GTEST_SKIP(); }`, which silently swallowed
    // every backend failure as "not available" even though the W.3 sweep
    // landed ROCm / OneAPI / MPS CTC kernels. A real dispatch break (missing
    // kernel, wrong attr, NaN output) was indistinguishable from a clean
    // skip. Let exceptions propagate as failures so the parity matrix
    // tooling (and humans reading the ctest output) sees the truth.

    // log_probs: (T, N, C) = (50, 4, 20)
    auto log_probs_raw = randn({50, 4, 20}, DType::Float32, Device::cpu());
    auto log_probs = nn::log_softmax(Variable(log_probs_raw, false), 2).tensor();

    // targets: (N, S) = (4, 10), class indices 1-19 (0 is blank)
    auto targets = (rand({4, 10}, DType::Float32, Device::cpu()) * 19.0f + 1.0f).to(DType::Int64);

    // input_lengths: all 50
    auto input_lengths = ones({4}, DType::Int64, Device::cpu()) * 50;

    // target_lengths: all 10
    auto target_lengths = ones({4}, DType::Int64, Device::cpu()) * 10;

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::CTCLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          inputs[1], inputs[2], inputs[3]).tensor();
    }, std::vector<Tensor>{log_probs, targets, input_lengths, target_lengths}, device, 1e-4f, 1e-5f, "CTCLoss");
}

TEST_P(NNLossParity, CTCLoss_1DConcatenatedTargets) {
    // Direct OpId::CTCLossForward dispatch with 1-D concatenated (PyTorch-style)
    // Int32 targets: total length = sum(target_lengths). The old CUDA CTC assumed
    // a 2-D layout and read targets + n*(total length), OOB for n >= 1; CPU
    // handles the concatenated layout via a prefix sum.
    const int64_t T = 30, N = 3, C = 8;
    auto log_probs_raw = randn({T, N, C}, DType::Float32, Device::cpu());
    auto log_probs = nn::log_softmax(Variable(log_probs_raw, false), 2).tensor().contiguous();
    // per-sample target length 5 → concatenated targets of length 15, labels in [1, C).
    auto targets = (rand({15}, DType::Float32, Device::cpu()) * static_cast<float>(C - 2) + 1.0f).to(DType::Int32);
    auto input_lengths  = (ones({N}, DType::Int32, Device::cpu()) * static_cast<double>(T));
    auto target_lengths = (ones({N}, DType::Int32, Device::cpu()) * 5.0);

    OpAttributes attrs;
    attrs.set(AttrKey::Blank, static_cast<int64_t>(0));
    attrs.set(AttrKey::ZeroInfinity, false);

    auto run = [&](Device dev) {
        std::vector<Tensor> ins = {log_probs.to(dev), targets.to(dev),
                                   input_lengths.to(dev), target_lengths.to(dev)};
        auto outs = dispatch(OpId::CTCLossForward, ins, attrs);
        return outs[0].to(Device::cpu()).to(DType::Float64);  // loss per sample (N,)
    };
    auto cpu_loss = run(Device::cpu());
    if (device.type == Device::Type::CPU) return;  // reference == device
    auto dev_loss = run(device);
    ASSERT_EQ(cpu_loss.numel(), dev_loss.numel());
    const double* cp = cpu_loss.data<double>();
    const double* dp = dev_loss.data<double>();
    for (int64_t i = 0; i < N; ++i)
        EXPECT_NEAR(cp[i], dp[i], 1e-3) << "sample " << i << " on " << device.to_string();
}

TEST_P(NNLossParity, MultiLabelSoftMarginLoss) {

    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    // Binary target (0 or 1)
    auto target = floor(rand({4, 10}, DType::Float32, Device::cpu()) + 0.5f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::MultiLabelSoftMarginLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false)).tensor();
    }, std::vector<Tensor>{pred, target}, device, 1e-5f, 1e-7f, "MultiLabelSoftMarginLoss");
}

TEST_P(NNLossParity, CIoULoss) {
    // CIoU is available as an IoU mode in detection ops, not as an nn::CIoULoss class.
    GTEST_SKIP() << "CIoULoss not available as an nn::Module";
}

TEST_P(NNLossParity, FocalLoss) {

    auto pred = randn({4, 10}, DType::Float32, Device::cpu());
    // FocalLoss takes Variable target (probabilities / one-hot style)
    auto target = floor(rand({4, 10}, DType::Float32, Device::cpu()) + 0.5f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::FocalLoss loss(1.0, 2.0);
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false)).tensor();
    }, std::vector<Tensor>{pred, target}, device, 1e-5f, 1e-7f, "FocalLoss");
}

TEST_P(NNLossParity, DiceLoss) {

    // Input should be probabilities (after sigmoid)
    auto pred = sigmoid(randn({4, 10}, DType::Float32, Device::cpu()));
    // Target should be binary masks
    auto target = floor(rand({4, 10}, DType::Float32, Device::cpu()) + 0.5f);

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        nn::DiceLoss loss;
        return loss.forward(Variable(inputs[0], false),
                          Variable(inputs[1], false)).tensor();
    }, std::vector<Tensor>{pred, target}, device, 1e-5f, 1e-7f, "DiceLoss");
}

// ============================================================================
// Main
// ============================================================================

// Phase 6-followup #27: gradient parity for losses. Loss backward kernels
// often have their own implementation per backend; this catches divergence.
TEST_P(NNLossParity, HuberLoss_GradientParity) {
    auto pred = randn({4, 8}, DType::Float32, Device::cpu());
    auto target = randn({4, 8}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            nn::HuberLoss loss;
            return loss.forward(in[0], in[1]);
        },
        {pred, target}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "HuberLoss_Grad");
}

TEST_P(NNLossParity, KLDivLoss_GradientParity) {
    auto pred = randn({4, 8}, DType::Float32, Device::cpu());
    auto target_logits = randn({4, 8}, DType::Float32, Device::cpu());
    auto target = nn::softmax(Variable(target_logits, false), 1).tensor();
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            nn::KLDivLoss loss;
            auto log_probs = nn::log_softmax(in[0], 1);
            return loss.forward(log_probs, in[1]);
        },
        {pred, target}, {}, 1e-4f, 1e-5f, 1e-3f, 1e-4f, {}, "KLDivLoss_Grad");
}

// ============================================================================
// Audit-driven additions (2026-05-02 E): six losses with zero parity coverage
// ============================================================================

TEST_P(NNLossParity, GaussianNLLLoss) {
    auto pred   = randn({4, 8}, DType::Float32, Device::cpu());
    auto target = randn({4, 8}, DType::Float32, Device::cpu());
    // Variance must be strictly positive — sigmoid * 2 + 0.1 keeps it in
    // (0.1, 2.1) which avoids log(near-zero) instability.
    auto var    = sigmoid(randn({4, 8}, DType::Float32, Device::cpu())) * 2.0f + 0.1f;

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        nn::GaussianNLLLoss loss;
        return loss.forward(Variable(in[0], false),
                            Variable(in[1], false),
                            Variable(in[2], false)).tensor();
    }, std::vector<Tensor>{pred, target, var}, device, 1e-4f, 1e-5f, "GaussianNLLLoss");
}

TEST_P(NNLossParity, MultiLabelMarginLoss) {
    // 4 samples × 5 classes; target is multi-label with -1 padding so
    // every sample has a different number of positive labels (but at
    // least one each). Layout per row: [c0, c1, ..., -1, -1].
    auto input = randn({4, 5}, DType::Float32, Device::cpu());
    Tensor target = full({4, 5}, 0.0, DType::Int64, Device::cpu());
    int64_t* tp = target.data<int64_t>();
    int64_t pattern[4][5] = {{0, 2, -1, -1, -1},
                             {1, -1, -1, -1, -1},
                             {3, 4, -1, -1, -1},
                             {0, 1, 2, -1, -1}};
    for (int i = 0; i < 4; ++i)
        for (int j = 0; j < 5; ++j) tp[i*5+j] = pattern[i][j];

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        nn::MultiLabelMarginLoss loss;
        return loss.forward(Variable(in[0], false), in[1]).tensor();
    }, std::vector<Tensor>{input, target}, device, 1e-4f, 1e-5f, "MultiLabelMarginLoss");
}

TEST_P(NNLossParity, TripletMarginWithDistanceLoss) {
    auto anchor   = randn({4, 16}, DType::Float32, Device::cpu());
    auto positive = randn({4, 16}, DType::Float32, Device::cpu());
    auto negative = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        // L2 distance in feature space (sqrt(sum((a-b)^2, dim=-1))).
        auto distance = [](const Variable& a, const Variable& b) {
            auto d = a + (b * -1.0f);
            return sum(d * d, /*dim=*/-1);
        };
        nn::TripletMarginWithDistanceLoss loss(distance, /*margin=*/1.0);
        return loss.forward(Variable(in[0], false),
                            Variable(in[1], false),
                            Variable(in[2], false)).tensor();
    }, std::vector<Tensor>{anchor, positive, negative}, device,
       1e-4f, 1e-5f, "TripletMarginWithDistanceLoss");
}

TEST_P(NNLossParity, InfoNCELoss) {
    auto queries = randn({4, 32}, DType::Float32, Device::cpu());
    auto keys    = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        nn::InfoNCELoss loss(/*temperature=*/0.07);
        return loss.forward(Variable(in[0], false),
                            Variable(in[1], false)).tensor();
        // InfoNCE builds the similarity matrix via q @ k^T matmul
        // (src/nn/loss/contrastive.cpp:69) → FP32 GEMM floor (parity::MATMUL_*).
    }, std::vector<Tensor>{queries, keys}, device, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "InfoNCELoss");
}

TEST_P(NNLossParity, NTXentLoss) {
    auto z_i = randn({4, 32}, DType::Float32, Device::cpu());
    auto z_j = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        nn::NTXentLoss loss(/*temperature=*/0.5);
        return loss.forward(Variable(in[0], false),
                            Variable(in[1], false)).tensor();
    }, std::vector<Tensor>{z_i, z_j}, device, 1e-4f, 1e-5f, "NTXentLoss");
}

TEST_P(NNLossParity, TripletLoss_Contrastive) {
    auto anchor   = randn({4, 16}, DType::Float32, Device::cpu());
    auto positive = randn({4, 16}, DType::Float32, Device::cpu());
    auto negative = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        nn::TripletLoss loss(/*margin=*/1.0, /*p=*/2.0);
        return loss.forward(Variable(in[0], false),
                            Variable(in[1], false),
                            Variable(in[2], false)).tensor();
    }, std::vector<Tensor>{anchor, positive, negative}, device,
       1e-4f, 1e-5f, "TripletLoss_Contrastive");
}

INSTANTIATE_BACKEND_TESTS(NNLossParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
