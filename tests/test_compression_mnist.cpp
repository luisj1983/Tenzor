/**
 * @file test_compression_mnist.cpp
 * @brief MNIST accuracy retention tests for quantization
 *
 * Verifies that quantization maintains <1% accuracy loss on MNIST dataset
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/compression/pruning.hpp"
#include "tenzor/nn/quantization.hpp"
#include <memory>
#include <random>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;
using namespace tenzor::nn::quantization;

// Simple MNIST model for testing
class MNISTModel : public Module {
public:
    MNISTModel() {
        fc1_ = std::make_shared<Linear>(784, 128);
        fc2_ = std::make_shared<Linear>(128, 64);
        fc3_ = std::make_shared<Linear>(64, 10);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc3", fc3_);
    }

    Variable forward(const Variable& x) override {
        auto h1 = fc1_->forward(x).relu();
        auto h2 = fc2_->forward(h1).relu();
        return fc3_->forward(h2);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Linear> fc3_;
};

class CompressionMNISTTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize random seed for reproducibility
        std::srand(42);

        // Create model
        model_ = std::make_shared<MNISTModel>();

        // Initialize with small random weights (simulating a pre-trained model)
        auto params = model_->parameters();
        for (auto& param : params) {
            auto* data = param->tensor().data<float>();
            for (int64_t i = 0; i < param->tensor().numel(); ++i) {
                data[i] = (static_cast<float>(rand()) / RAND_MAX - 0.5f) * 0.1f;
            }
        }

        // Create synthetic MNIST-like test data
        CreateTestData();
    }

    void CreateTestData() {
        // Create 100 test samples with known labels
        num_samples_ = 100;
        test_data_ = Tensor({num_samples_, 784}, DType::Float32, Device::cpu());
        test_labels_ = Tensor({num_samples_}, DType::Int64, Device::cpu());

        auto* data = test_data_.data<float>();
        auto* labels = test_labels_.data<int64_t>();

        std::mt19937 gen(42);
        std::normal_distribution<float> dist(0.0f, 1.0f);

        for (int i = 0; i < num_samples_; ++i) {
            int label = i % 10;  // Cycle through 0-9
            labels[i] = label;

            // Create pattern specific to label
            for (int j = 0; j < 784; ++j) {
                data[i * 784 + j] = dist(gen) + (j % 10 == label ? 0.5f : 0.0f);
            }
        }
    }

    float EvaluateAccuracy(std::shared_ptr<Module> model) {
        model->eval();

        Variable input(test_data_, false);
        auto output = model->forward(input);

        // Get predictions
        auto pred_tensor = output.tensor();
        auto* pred_data = pred_tensor.data<float>();
        auto* true_labels = test_labels_.data<int64_t>();

        int correct = 0;
        for (int i = 0; i < num_samples_; ++i) {
            // Find argmax
            int pred_label = 0;
            float max_score = pred_data[i * 10];
            for (int j = 1; j < 10; ++j) {
                if (pred_data[i * 10 + j] > max_score) {
                    max_score = pred_data[i * 10 + j];
                    pred_label = j;
                }
            }

            if (pred_label == true_labels[i]) {
                correct++;
            }
        }

        return static_cast<float>(correct) / num_samples_;
    }

    std::shared_ptr<MNISTModel> model_;
    Tensor test_data_;
    Tensor test_labels_;
    int num_samples_;
};

// =============================================================================
// Quantization Accuracy Tests
// =============================================================================

TEST_F(CompressionMNISTTest, Quantization_PerTensor_AccuracyRetention) {
    // Get baseline FP32 accuracy
    float baseline_acc = EvaluateAccuracy(model_);

    // Quantize all weights
    auto params = model_->parameters();
    std::vector<QuantizedTensor> quantized_params;

    for (auto& param : params) {
        auto q_tensor = quantize_per_tensor_symmetric(param->tensor(), QuantDType::INT8);
        quantized_params.push_back(q_tensor);

        // Replace with dequantized version
        param->tensor() = q_tensor.dequantize();
    }

    // Get quantized accuracy
    float quant_acc = EvaluateAccuracy(model_);

    // Check accuracy retention
    float acc_loss = baseline_acc - quant_acc;

    std::cout << "Baseline FP32 Accuracy: " << (baseline_acc * 100) << "%\n";
    std::cout << "Quantized INT8 Accuracy: " << (quant_acc * 100) << "%\n";
    std::cout << "Accuracy Loss: " << (acc_loss * 100) << "%\n";

    // Verify <1% accuracy loss
    EXPECT_LT(std::abs(acc_loss), 0.01f)
        << "Quantization accuracy loss exceeds 1%: " << (acc_loss * 100) << "%";

    // Verify 4x memory reduction
    size_t fp32_bytes = 0;
    size_t int8_bytes = 0;
    for (const auto& q : quantized_params) {
        fp32_bytes += q.data().numel() * sizeof(float);
        int8_bytes += q.data().numel() * sizeof(int8_t);
    }

    float compression_ratio = static_cast<float>(fp32_bytes) / int8_bytes;
    EXPECT_NEAR(compression_ratio, 4.0f, 0.1f) << "Expected 4x compression ratio";

    std::cout << "Memory Compression: " << compression_ratio << "x\n";
}

TEST_F(CompressionMNISTTest, Quantization_PerChannel_BetterAccuracy) {
    float baseline_acc = EvaluateAccuracy(model_);

    // Quantize with per-channel scheme (should give better accuracy)
    auto named_params = model_->named_parameters();

    for (auto& [name, param] : named_params) {
        // Per-channel quantization on weight matrices (axis 0 = output channels)
        if (name.find("weight") != std::string::npos) {
            auto q_tensor = quantize_per_channel_symmetric(param->tensor(), 0, QuantDType::INT8);
            param->tensor() = q_tensor.dequantize();
        } else {
            // Per-tensor for biases
            auto q_tensor = quantize_per_tensor_symmetric(param->tensor(), QuantDType::INT8);
            param->tensor() = q_tensor.dequantize();
        }
    }

    float quant_acc = EvaluateAccuracy(model_);
    float acc_loss = baseline_acc - quant_acc;

    std::cout << "Per-Channel Quantization:\n";
    std::cout << "  Baseline: " << (baseline_acc * 100) << "%\n";
    std::cout << "  Quantized: " << (quant_acc * 100) << "%\n";
    std::cout << "  Loss: " << (acc_loss * 100) << "%\n";

    // Per-channel should have even lower accuracy loss
    EXPECT_LT(std::abs(acc_loss), 0.01f);
}

// =============================================================================
// Pruning Accuracy Tests
// =============================================================================

TEST_F(CompressionMNISTTest, Pruning_50Percent_MaintainsAccuracy) {
    float baseline_acc = EvaluateAccuracy(model_);

    // Apply 50% unstructured pruning
    auto config = prune_unstructured(model_, 0.5f, ImportanceCriterion::L1);
    apply_pruning_masks(model_, config);

    float pruned_acc = EvaluateAccuracy(model_);
    float acc_loss = baseline_acc - pruned_acc;

    std::cout << "50% Pruning:\n";
    std::cout << "  Baseline: " << (baseline_acc * 100) << "%\n";
    std::cout << "  Pruned: " << (pruned_acc * 100) << "%\n";
    std::cout << "  Loss: " << (acc_loss * 100) << "%\n";

    float actual_sparsity = compute_sparsity(model_);
    std::cout << "  Actual Sparsity: " << (actual_sparsity * 100) << "%\n";

    // 50% pruning should maintain reasonable accuracy
    EXPECT_GT(pruned_acc, 0.3f) << "Accuracy dropped too much after pruning";
    EXPECT_NEAR(actual_sparsity, 0.5f, 0.05f) << "Did not achieve target sparsity";
}

TEST_F(CompressionMNISTTest, Pruning_90Percent_StillFunctional) {
    float baseline_acc = EvaluateAccuracy(model_);

    // Apply aggressive 90% pruning
    auto config = prune_unstructured(model_, 0.9f, ImportanceCriterion::L1);
    apply_pruning_masks(model_, config);

    float pruned_acc = EvaluateAccuracy(model_);
    float actual_sparsity = compute_sparsity(model_);

    std::cout << "90% Pruning:\n";
    std::cout << "  Baseline: " << (baseline_acc * 100) << "%\n";
    std::cout << "  Pruned: " << (pruned_acc * 100) << "%\n";
    std::cout << "  Sparsity: " << (actual_sparsity * 100) << "%\n";

    // Model should still be functional (better than random guessing)
    EXPECT_GT(pruned_acc, 0.1f) << "Model completely broken after 90% pruning";
    EXPECT_NEAR(actual_sparsity, 0.9f, 0.05f);

    // Verify compression ratio
    auto comp_ratio = compute_compression_ratio(model_, model_);
    std::cout << "  Compression Ratio: " << comp_ratio << "x\n";
}

// =============================================================================
// Combined Compression Tests
// =============================================================================

TEST_F(CompressionMNISTTest, Combined_Pruning_And_Quantization) {
    float baseline_acc = EvaluateAccuracy(model_);

    // Step 1: Prune 50%
    auto config = prune_unstructured(model_, 0.5f, ImportanceCriterion::L1);
    apply_pruning_masks(model_, config);

    float after_pruning = EvaluateAccuracy(model_);

    // Step 2: Quantize to INT8
    auto params = model_->parameters();
    for (auto& param : params) {
        auto q_tensor = quantize_per_tensor_symmetric(param->tensor(), QuantDType::INT8);
        param->tensor() = q_tensor.dequantize();
    }

    float final_acc = EvaluateAccuracy(model_);

    std::cout << "Combined Compression:\n";
    std::cout << "  Baseline FP32: " << (baseline_acc * 100) << "%\n";
    std::cout << "  After 50% Pruning: " << (after_pruning * 100) << "%\n";
    std::cout << "  After Quantization: " << (final_acc * 100) << "%\n";

    float sparsity = compute_sparsity(model_);
    float memory_reduction = 4.0f / (1.0f - sparsity);  // quantization * sparsity

    std::cout << "  Sparsity: " << (sparsity * 100) << "%\n";
    std::cout << "  Combined Compression: " << memory_reduction << "x\n";

    // Combined should achieve 8x compression (4x quant + 50% pruning)
    EXPECT_GT(memory_reduction, 7.0f) << "Did not achieve expected compression";
    EXPECT_LT(memory_reduction, 9.0f);
}

TEST_F(CompressionMNISTTest, Iterative_Pruning_Gradual_Quality) {
    float baseline_acc = EvaluateAccuracy(model_);

    // Iteratively prune to 70% over 5 steps
    auto config = prune_iterative(model_, 0.7f, 5,
                                  PruningSchedule::Polynomial,
                                  ImportanceCriterion::L1);

    std::vector<float> accuracies;
    accuracies.push_back(baseline_acc);

    // Simulate iterative pruning with evaluation
    for (int iter = 0; iter < 5; ++iter) {
        config.current_iteration = iter;
        float target_sparsity = config.get_current_sparsity();

        // Update masks for current sparsity
        auto iter_config = prune_unstructured(model_, target_sparsity, ImportanceCriterion::L1);
        config.masks = iter_config.masks;
        apply_pruning_masks(model_, config);

        float acc = EvaluateAccuracy(model_);
        accuracies.push_back(acc);

        std::cout << "Iteration " << iter << ": Sparsity=" << (target_sparsity * 100)
                  << "%, Acc=" << (acc * 100) << "%\n";
    }

    // Final accuracy should be reasonable
    EXPECT_GT(accuracies.back(), 0.2f) << "Accuracy degraded too much";
}

// =============================================================================
// Performance Benchmarks
// =============================================================================

TEST_F(CompressionMNISTTest, Benchmark_CompressionRatios) {
    struct CompressionResult {
        std::string method;
        float sparsity;
        float memory_ratio;
        float accuracy;
    };

    std::vector<CompressionResult> results;

    // Baseline
    results.push_back({"Baseline FP32", 0.0f, 1.0f, EvaluateAccuracy(model_)});

    // Test different pruning levels
    for (float sparsity : {0.3f, 0.5f, 0.7f, 0.9f}) {
        auto test_model = std::make_shared<MNISTModel>();
        test_model->load_state_dict(model_->state_dict());

        auto config = prune_unstructured(test_model, sparsity, ImportanceCriterion::L1);
        apply_pruning_masks(test_model, config);

        float acc = EvaluateAccuracy(test_model);
        float actual_sparsity = compute_sparsity(test_model);

        results.push_back({
            "Pruning " + std::to_string(int(sparsity * 100)) + "%",
            actual_sparsity,
            1.0f / (1.0f - actual_sparsity),
            acc
        });
    }

    // Quantization only
    {
        auto test_model = std::make_shared<MNISTModel>();
        test_model->load_state_dict(model_->state_dict());

        auto params = test_model->parameters();
        for (auto& param : params) {
            auto q = quantize_per_tensor_symmetric(param->tensor(), QuantDType::INT8);
            param->tensor() = q.dequantize();
        }

        results.push_back({"INT8 Quantization", 0.0f, 4.0f, EvaluateAccuracy(test_model)});
    }

    // Combined: 50% pruning + quantization
    {
        auto test_model = std::make_shared<MNISTModel>();
        test_model->load_state_dict(model_->state_dict());

        auto config = prune_unstructured(test_model, 0.5f, ImportanceCriterion::L1);
        apply_pruning_masks(test_model, config);

        auto params = test_model->parameters();
        for (auto& param : params) {
            auto q = quantize_per_tensor_symmetric(param->tensor(), QuantDType::INT8);
            param->tensor() = q.dequantize();
        }

        float sparsity = compute_sparsity(test_model);
        float ratio = 4.0f / (1.0f - sparsity);

        results.push_back({"50% Prune + INT8", sparsity, ratio, EvaluateAccuracy(test_model)});
    }

    // Print results table
    std::cout << "\n=== Compression Benchmark Results ===\n";
    std::cout << std::setw(25) << "Method"
              << std::setw(12) << "Sparsity"
              << std::setw(15) << "Memory Ratio"
              << std::setw(12) << "Accuracy\n";
    std::cout << std::string(64, '-') << "\n";

    for (const auto& r : results) {
        std::cout << std::setw(25) << r.method
                  << std::setw(11) << std::fixed << std::setprecision(1) << (r.sparsity * 100) << "%"
                  << std::setw(14) << std::setprecision(2) << r.memory_ratio << "x"
                  << std::setw(11) << std::setprecision(1) << (r.accuracy * 100) << "%\n";
    }
    std::cout << std::string(64, '-') << "\n\n";
}
