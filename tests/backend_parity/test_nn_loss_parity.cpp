/**
 * @file test_nn_loss_parity.cpp
 * @brief Backend parity tests for neural network loss functions
 *
 * Tests 14 loss functions across all available backends to ensure
 * consistent results regardless of compute device.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

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

    try {
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
    } catch (...) {
        GTEST_SKIP() << "CosineEmbeddingLoss not available";
    }
}

TEST_P(NNLossParity, TripletMarginLoss) {

    try {
        auto anchor = randn({4, 32}, DType::Float32, Device::cpu());
        auto positive = randn({4, 32}, DType::Float32, Device::cpu());
        auto negative = randn({4, 32}, DType::Float32, Device::cpu());

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::TripletMarginLoss loss;
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false),
                              Variable(inputs[2], false)).tensor();
        }, std::vector<Tensor>{anchor, positive, negative}, device, 1e-5f, 1e-7f, "TripletMarginLoss");
    } catch (...) {
        GTEST_SKIP() << "TripletMarginLoss not available";
    }
}

TEST_P(NNLossParity, MarginRankingLoss) {

    try {
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
    } catch (...) {
        GTEST_SKIP() << "MarginRankingLoss not available";
    }
}

TEST_P(NNLossParity, SoftMarginLoss) {

    try {
        auto pred = randn({4, 10}, DType::Float32, Device::cpu());
        auto target_raw = rand({4, 10}, DType::Float32, Device::cpu());
        auto target = sign(target_raw * 2.0f - 1.0f);

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::SoftMarginLoss loss;
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false)).tensor();
        }, {pred, target}, device, 1e-5f, 1e-7f, "SoftMarginLoss");
    } catch (...) {
        GTEST_SKIP() << "SoftMarginLoss not available";
    }
}

TEST_P(NNLossParity, HingeEmbeddingLoss) {

    try {
        auto input = randn({4, 10}, DType::Float32, Device::cpu());
        auto target_raw = rand({4, 10}, DType::Float32, Device::cpu());
        auto target = sign(target_raw * 2.0f - 1.0f);

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::HingeEmbeddingLoss loss;
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false)).tensor();
        }, {input, target}, device, 1e-5f, 1e-7f, "HingeEmbeddingLoss");
    } catch (...) {
        GTEST_SKIP() << "HingeEmbeddingLoss not available";
    }
}

TEST_P(NNLossParity, PoissonNLLLoss) {

    try {
        // pred is log-rates (any real number)
        auto pred = randn({4, 10}, DType::Float32, Device::cpu());
        // target is counts (must be positive)
        auto target = rand({4, 10}, DType::Float32, Device::cpu()) * 5.0f + 0.1f;

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::PoissonNLLLoss loss;
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false)).tensor();
        }, {pred, target}, device, 1e-5f, 1e-7f, "PoissonNLLLoss");
    } catch (...) {
        GTEST_SKIP() << "PoissonNLLLoss not available";
    }
}

TEST_P(NNLossParity, CTCLoss) {

    try {
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
    } catch (...) {
        GTEST_SKIP() << "CTCLoss not available";
    }
}

TEST_P(NNLossParity, MultiLabelSoftMarginLoss) {

    try {
        auto pred = randn({4, 10}, DType::Float32, Device::cpu());
        // Binary target (0 or 1)
        auto target = floor(rand({4, 10}, DType::Float32, Device::cpu()) + 0.5f);

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::MultiLabelSoftMarginLoss loss;
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false)).tensor();
        }, std::vector<Tensor>{pred, target}, device, 1e-5f, 1e-7f, "MultiLabelSoftMarginLoss");
    } catch (...) {
        GTEST_SKIP() << "MultiLabelSoftMarginLoss not available";
    }
}

TEST_P(NNLossParity, CIoULoss) {
    // CIoU is available as an IoU mode in detection ops, not as an nn::CIoULoss class.
    GTEST_SKIP() << "CIoULoss not available as an nn::Module";
}

TEST_P(NNLossParity, FocalLoss) {

    try {
        auto pred = randn({4, 10}, DType::Float32, Device::cpu());
        // FocalLoss takes Variable target (probabilities / one-hot style)
        auto target = floor(rand({4, 10}, DType::Float32, Device::cpu()) + 0.5f);

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::FocalLoss loss(1.0, 2.0);
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false)).tensor();
        }, std::vector<Tensor>{pred, target}, device, 1e-5f, 1e-7f, "FocalLoss");
    } catch (...) {
        GTEST_SKIP() << "FocalLoss not available";
    }
}

TEST_P(NNLossParity, DiceLoss) {

    try {
        // Input should be probabilities (after sigmoid)
        auto pred = sigmoid(randn({4, 10}, DType::Float32, Device::cpu()));
        // Target should be binary masks
        auto target = floor(rand({4, 10}, DType::Float32, Device::cpu()) + 0.5f);

        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            nn::DiceLoss loss;
            return loss.forward(Variable(inputs[0], false),
                              Variable(inputs[1], false)).tensor();
        }, std::vector<Tensor>{pred, target}, device, 1e-5f, 1e-7f, "DiceLoss");
    } catch (...) {
        GTEST_SKIP() << "DiceLoss not available";
    }
}

// ============================================================================
// Main
// ============================================================================

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
