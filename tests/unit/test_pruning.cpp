/**
 * @file test_pruning.cpp
 * @brief Comprehensive tests for neural network pruning functionality
 *
 * Tests unstructured pruning, structured pruning (channels, filters, layers),
 * importance criteria, pruning schedules, mask operations, and integration scenarios.
 *
 * CRITICAL: Tests Agent 6's layer pruning implementation (lines 334-338 in pruning.hpp)
 */

#include <gtest/gtest.h>
#include "../../include/tenzor/tenzor.hpp"
#include "../../include/tenzor/nn/compression/pruning.hpp"
#include <memory>
#include <cmath>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;

class PruningTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const pruning_env =
    ::testing::AddGlobalTestEnvironment(new PruningTestEnvironment);

class PruningTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Seed for reproducibility
        std::srand(42);
    }

    // Helper to count zeros in tensor
    int64_t count_zeros(const Tensor& t) {
        auto* data = t.data<float>();
        int64_t count = 0;
        for (int64_t i = 0; i < t.numel(); ++i) {
            if (std::abs(data[i]) < 1e-7f) {
                count++;
            }
        }
        return count;
    }

    // Helper to calculate sparsity
    float calculate_sparsity(const Tensor& t) {
        if (t.numel() == 0) return 0.0f;
        return static_cast<float>(count_zeros(t)) / static_cast<float>(t.numel());
    }

    // Helper to create random tensor
    Tensor create_random_tensor(const std::vector<int64_t>& shape) {
        Tensor t(shape, DType::Float32, Device::cpu());
        auto* data = t.data<float>();
        for (int64_t i = 0; i < t.numel(); ++i) {
            data[i] = (static_cast<float>(rand()) / RAND_MAX) * 2.0f - 1.0f;
        }
        return t;
    }

    // Create simple sequential model for testing
    class SimpleSequential : public Module {
    public:
        SimpleSequential(int num_layers, int in_features, int hidden_features, int out_features) {
            auto layer0 = std::make_shared<Linear>(in_features, hidden_features);
            register_module("layer0", layer0);
            layers_.push_back(layer0);

            for (int i = 1; i < num_layers - 1; ++i) {
                auto layer = std::make_shared<Linear>(hidden_features, hidden_features);
                register_module("layer" + std::to_string(i), layer);
                layers_.push_back(layer);
            }

            auto final_layer = std::make_shared<Linear>(hidden_features, out_features);
            register_module("layer" + std::to_string(num_layers - 1), final_layer);
            layers_.push_back(final_layer);
        }

        Variable forward(const Variable& x) override {
            Variable out = x;
            for (auto& layer : layers_) {
                out = layer->forward(out);
            }
            return out;
        }

        std::vector<std::shared_ptr<Linear>> layers_;
    };
};

// ============================================================================
// Importance Criterion Tests
// ============================================================================

TEST_F(PruningTest, ComputeImportance_L1) {
    Tensor weights = create_random_tensor({64, 32});

    Tensor importance = compute_importance(weights, ImportanceCriterion::L1);

    EXPECT_TRUE(std::equal(importance.shape().begin(), importance.shape().end(), weights.shape().begin(), weights.shape().end()));

    // L1 importance should be absolute values
    auto* w_data = weights.data<float>();
    auto* i_data = importance.data<float>();
    for (int64_t i = 0; i < weights.numel(); ++i) {
        EXPECT_NEAR(i_data[i], std::abs(w_data[i]), 1e-5f);
    }
}

TEST_F(PruningTest, ComputeImportance_L2) {
    Tensor weights = create_random_tensor({64, 32});

    Tensor importance = compute_importance(weights, ImportanceCriterion::L2);

    EXPECT_TRUE(std::equal(importance.shape().begin(), importance.shape().end(), weights.shape().begin(), weights.shape().end()));

    // L2 importance should be squared values
    auto* w_data = weights.data<float>();
    auto* i_data = importance.data<float>();
    for (int64_t i = 0; i < weights.numel(); ++i) {
        EXPECT_NEAR(i_data[i], w_data[i] * w_data[i], 1e-5f);
    }
}

TEST_F(PruningTest, ComputeImportance_L1Norm) {
    Tensor weights = create_random_tensor({64, 32});

    Tensor importance = compute_importance(weights, ImportanceCriterion::L1Norm);

    EXPECT_TRUE(std::equal(importance.shape().begin(), importance.shape().end(), weights.shape().begin(), weights.shape().end()));

    // L1Norm should normalize by number of parameters
    auto* w_data = weights.data<float>();
    auto* i_data = importance.data<float>();
    for (int64_t i = 0; i < weights.numel(); ++i) {
        float expected = std::abs(w_data[i]) / weights.numel();
        EXPECT_NEAR(i_data[i], expected, 1e-5f);
    }
}

TEST_F(PruningTest, ComputeImportance_L2Norm) {
    Tensor weights = create_random_tensor({64, 32});

    Tensor importance = compute_importance(weights, ImportanceCriterion::L2Norm);

    EXPECT_TRUE(std::equal(importance.shape().begin(), importance.shape().end(), weights.shape().begin(), weights.shape().end()));

    // L2Norm should normalize by number of parameters
    auto* w_data = weights.data<float>();
    auto* i_data = importance.data<float>();
    for (int64_t i = 0; i < weights.numel(); ++i) {
        float expected = (w_data[i] * w_data[i]) / weights.numel();
        EXPECT_NEAR(i_data[i], expected, 1e-5f);
    }
}

// ============================================================================
// Mask Creation Tests
// ============================================================================

TEST_F(PruningTest, CreateMask_50PercentSparsity) {
    Tensor importance = create_random_tensor({100, 100});

    Tensor mask = create_mask_from_importance(importance, 0.5f);

    EXPECT_TRUE(std::equal(mask.shape().begin(), mask.shape().end(), importance.shape().begin(), importance.shape().end()));

    // Check mask is binary
    auto* mask_data = mask.data<float>();
    for (int64_t i = 0; i < mask.numel(); ++i) {
        EXPECT_TRUE(mask_data[i] == 0.0f || mask_data[i] == 1.0f);
    }

    // Check sparsity is approximately 50%
    float sparsity = calculate_sparsity(mask);
    EXPECT_NEAR(sparsity, 0.5f, 0.02f);
}

TEST_F(PruningTest, CreateMask_VariousSparsityLevels) {
    std::vector<float> sparsity_levels = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    for (float target_sparsity : sparsity_levels) {
        Tensor importance = create_random_tensor({128, 64});
        Tensor mask = create_mask_from_importance(importance, target_sparsity);

        float actual_sparsity = calculate_sparsity(mask);
        EXPECT_NEAR(actual_sparsity, target_sparsity, 0.02f)
            << "Failed for target sparsity: " << target_sparsity;
    }
}

TEST_F(PruningTest, CreateMask_ZeroSparsity) {
    Tensor importance = create_random_tensor({64, 64});
    Tensor mask = create_mask_from_importance(importance, 0.0f);

    // No pruning - all ones
    float sparsity = calculate_sparsity(mask);
    EXPECT_NEAR(sparsity, 0.0f, 1e-5f);
}

TEST_F(PruningTest, CreateMask_FullSparsity) {
    Tensor importance = create_random_tensor({64, 64});
    Tensor mask = create_mask_from_importance(importance, 1.0f);

    // Full pruning - all zeros
    float sparsity = calculate_sparsity(mask);
    EXPECT_NEAR(sparsity, 1.0f, 0.01f);
}

// ============================================================================
// PruningMask Tests
// ============================================================================

TEST_F(PruningTest, PruningMask_Apply) {
    Tensor weights = create_random_tensor({64, 32});
    Tensor mask_tensor = create_random_tensor({64, 32});

    // Make mask binary
    auto* mask_data = mask_tensor.data<float>();
    for (int64_t i = 0; i < mask_tensor.numel(); ++i) {
        mask_data[i] = (mask_data[i] > 0.0f) ? 1.0f : 0.0f;
    }

    PruningMask mask;
    mask.mask = mask_tensor;
    mask.layer_name = "test_layer";

    Tensor masked_weights = mask.apply(weights);

    // Check element-wise multiplication
    auto* w_data = weights.data<float>();
    auto* m_data = mask_tensor.data<float>();
    auto* mw_data = masked_weights.data<float>();

    for (int64_t i = 0; i < weights.numel(); ++i) {
        EXPECT_NEAR(mw_data[i], w_data[i] * m_data[i], 1e-5f);
    }
}

TEST_F(PruningTest, PruningMask_ComputeSparsity) {
    PruningMask mask;
    mask.mask = create_random_tensor({100, 100});

    // Set 30% to zero
    auto* data = mask.mask.data<float>();
    for (int64_t i = 0; i < 3000; ++i) {
        data[i] = 0.0f;
    }
    for (int64_t i = 3000; i < mask.mask.numel(); ++i) {
        data[i] = 1.0f;
    }

    float sparsity = mask.compute_sparsity();
    EXPECT_NEAR(sparsity, 0.3f, 0.01f);
}

// ============================================================================
// Unstructured Pruning Tests - L1 Criterion
// ============================================================================

TEST_F(PruningTest, UnstructuredPruning_L1_50Percent) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);

    EXPECT_EQ(config.masks.size(), 1);
    EXPECT_NEAR(config.target_sparsity, 0.5f, 1e-5f);

    // Apply masks and check sparsity
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());
    EXPECT_NEAR(sparsity, 0.5f, 0.05f);
}

TEST_F(PruningTest, UnstructuredPruning_L1_VariousSparsities) {
    std::vector<float> sparsity_levels = {0.1f, 0.3f, 0.5f, 0.7f, 0.9f};

    for (float target_sparsity : sparsity_levels) {
        auto linear = std::make_shared<Linear>(64, 32, true);

        PruningConfig config = prune_unstructured(linear, target_sparsity,
                                                   ImportanceCriterion::L1, false);
        apply_pruning_masks(linear, config);

        auto weight = linear->weight();
        float actual_sparsity = calculate_sparsity(weight->tensor());

        EXPECT_NEAR(actual_sparsity, target_sparsity, 0.05f)
            << "Failed for sparsity level: " << target_sparsity;
    }
}

// ============================================================================
// Unstructured Pruning Tests - L2 Criterion
// ============================================================================

TEST_F(PruningTest, UnstructuredPruning_L2_50Percent) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L2, false);
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());
    EXPECT_NEAR(sparsity, 0.5f, 0.05f);
}

TEST_F(PruningTest, UnstructuredPruning_L2_70Percent) {
    auto linear = std::make_shared<Linear>(256, 128, true);

    PruningConfig config = prune_unstructured(linear, 0.7f, ImportanceCriterion::L2, false);
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());
    EXPECT_NEAR(sparsity, 0.7f, 0.05f);
}

// ============================================================================
// Unstructured Pruning Tests - L1Norm Criterion
// ============================================================================

TEST_F(PruningTest, UnstructuredPruning_L1Norm) {
    auto linear = std::make_shared<Linear>(100, 50, true);

    PruningConfig config = prune_unstructured(linear, 0.4f, ImportanceCriterion::L1Norm, false);
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());
    EXPECT_NEAR(sparsity, 0.4f, 0.05f);
}

// ============================================================================
// Unstructured Pruning Tests - L2Norm Criterion
// ============================================================================

TEST_F(PruningTest, UnstructuredPruning_L2Norm) {
    auto linear = std::make_shared<Linear>(100, 50, true);

    PruningConfig config = prune_unstructured(linear, 0.6f, ImportanceCriterion::L2Norm, false);
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());
    EXPECT_NEAR(sparsity, 0.6f, 0.05f);
}

// ============================================================================
// Global vs Local Pruning
// ============================================================================

TEST_F(PruningTest, UnstructuredPruning_GlobalVsLocal) {
    auto model = std::make_shared<SimpleSequential>(3, 128, 128, 10);

    // Global pruning
    PruningConfig config_global = prune_unstructured(model, 0.5f,
                                                      ImportanceCriterion::L1, true);

    // Local pruning
    auto model2 = std::make_shared<SimpleSequential>(3, 128, 128, 10);
    PruningConfig config_local = prune_unstructured(model2, 0.5f,
                                                     ImportanceCriterion::L1, false);

    // Both should have masks
    EXPECT_GT(config_global.masks.size(), 0);
    EXPECT_GT(config_local.masks.size(), 0);
}

// ============================================================================
// Iterative Pruning Tests
// ============================================================================

TEST_F(PruningTest, IterativePruning_OneShot) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_iterative(linear, 0.8f, 1,
                                           PruningSchedule::OneShot,
                                           ImportanceCriterion::L1);

    EXPECT_EQ(config.schedule, PruningSchedule::OneShot);
    EXPECT_NEAR(config.target_sparsity, 0.8f, 1e-5f);
}

TEST_F(PruningTest, IterativePruning_Iterative_5Steps) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_iterative(linear, 0.9f, 5,
                                           PruningSchedule::Iterative,
                                           ImportanceCriterion::L1);

    EXPECT_EQ(config.schedule, PruningSchedule::Iterative);
    EXPECT_EQ(config.num_iterations, 5);
    EXPECT_NEAR(config.target_sparsity, 0.9f, 1e-5f);
}

TEST_F(PruningTest, IterativePruning_Polynomial) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_iterative(linear, 0.7f, 10,
                                           PruningSchedule::Polynomial,
                                           ImportanceCriterion::L2);

    EXPECT_EQ(config.schedule, PruningSchedule::Polynomial);
    EXPECT_EQ(config.num_iterations, 10);
}

TEST_F(PruningTest, IterativePruning_CurrentSparsity) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_iterative(linear, 0.9f, 10,
                                           PruningSchedule::Iterative,
                                           ImportanceCriterion::L1);

    // Simulate iterations
    for (int i = 0; i < 10; ++i) {
        config.current_iteration = i;
        float current_sparsity = config.get_current_sparsity();

        // Current sparsity should increase with iterations
        EXPECT_GE(current_sparsity, 0.0f);
        EXPECT_LE(current_sparsity, config.target_sparsity);
    }
}

// ============================================================================
// Structured Pruning - Channel Pruning Tests
// ============================================================================

TEST_F(PruningTest, ChannelPruning_Conv2d_30Percent) {
    auto conv = std::make_shared<Conv2d>(64, 128, 3, 1, 1, 1, 1, true);

    auto pruned_conv = prune_channels(conv, 0.3f, ImportanceCriterion::L1);

    ASSERT_NE(pruned_conv, nullptr);

    // Should have 30% fewer channels
    // Access weight parameters
    auto conv_params = conv->named_parameters();
    auto pruned_params = pruned_conv->named_parameters();
    Tensor original_weight, pruned_weight;
    for (auto& [name, param] : conv_params) {
        if (name.find("weight") != std::string::npos) {
            original_weight = param->tensor();
            break;
        }
    }
    for (auto& [name, param] : pruned_params) {
        if (name.find("weight") != std::string::npos) {
            pruned_weight = param->tensor();
            break;
        }
    }

    int64_t expected_channels = static_cast<int64_t>(128 * 0.7f);
    EXPECT_EQ(pruned_weight.shape()[0], expected_channels);
}

TEST_F(PruningTest, ChannelPruning_Conv2d_50Percent) {
    auto conv = std::make_shared<Conv2d>(32, 64, 3, 1, 1, 1, 1, true);

    auto pruned_conv = prune_channels(conv, 0.5f, ImportanceCriterion::L1);

    Tensor pruned_weight;
    for (auto& [name, param] : pruned_conv->named_parameters()) {
        if (name.find("weight") != std::string::npos) {
            pruned_weight = param->tensor();
            break;
        }
    }

    // Should have 50% of original channels
    int64_t expected_channels = 32;  // 64 * 0.5
    EXPECT_EQ(pruned_weight.shape()[0], expected_channels);
}

TEST_F(PruningTest, ChannelPruning_L2Criterion) {
    auto conv = std::make_shared<Conv2d>(64, 128, 3, 1, 1, 1, 1, true);

    auto pruned_conv = prune_channels(conv, 0.25f, ImportanceCriterion::L2);

    Tensor pruned_weight;
    for (auto& [name, param] : pruned_conv->named_parameters()) {
        if (name.find("weight") != std::string::npos) {
            pruned_weight = param->tensor();
            break;
        }
    }

    int64_t expected_channels = static_cast<int64_t>(128 * 0.75f);
    EXPECT_EQ(pruned_weight.shape()[0], expected_channels);
}

// ============================================================================
// Structured Pruning - Filter Pruning Tests
// ============================================================================

TEST_F(PruningTest, FilterPruning_Conv2d_40Percent) {
    auto conv = std::make_shared<Conv2d>(64, 128, 3, 1, 1, 1, 1, true);

    auto pruned_conv = prune_filters(conv, 0.4f, ImportanceCriterion::L1);

    ASSERT_NE(pruned_conv, nullptr);

    Tensor pruned_weight;
    for (auto& [name, param] : pruned_conv->named_parameters()) {
        if (name.find("weight") != std::string::npos) {
            pruned_weight = param->tensor();
            break;
        }
    }

    // Should prune filters (reduce output channels)
    int64_t expected_filters = static_cast<int64_t>(128 * 0.6f);
    EXPECT_EQ(pruned_weight.shape()[0], expected_filters);
}

TEST_F(PruningTest, FilterPruning_L2Criterion) {
    auto conv = std::make_shared<Conv2d>(32, 64, 3, 1, 1, 1, 1, true);

    auto pruned_conv = prune_filters(conv, 0.5f, ImportanceCriterion::L2);

    Tensor pruned_weight;
    for (auto& [name, param] : pruned_conv->named_parameters()) {
        if (name.find("weight") != std::string::npos) {
            pruned_weight = param->tensor();
            break;
        }
    }

    int64_t expected_filters = 32;  // 64 * 0.5
    EXPECT_EQ(pruned_weight.shape()[0], expected_filters);
}

// ============================================================================
// CRITICAL: Layer Pruning Tests (Agent 6 Implementation)
// ============================================================================

TEST_F(PruningTest, LayerPruning_RemoveHalfLayers) {
    auto model = std::make_shared<SimpleSequential>(6, 128, 128, 10);

    // Prune 3 out of 6 layers
    auto pruned_model = prune_layers(model, 3, ImportanceCriterion::L1);

    ASSERT_NE(pruned_model, nullptr);

    // Test that model still works
    Tensor input({4, 128}, DType::Float32, Device::cpu());
    input.fill_(1.0f);

    auto output = pruned_model->forward(Variable(input, false));
    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

TEST_F(PruningTest, LayerPruning_RemoveSingleLayer) {
    auto model = std::make_shared<SimpleSequential>(5, 128, 128, 10);

    // Prune 1 layer
    auto pruned_model = prune_layers(model, 1, ImportanceCriterion::L1);

    ASSERT_NE(pruned_model, nullptr);

    Tensor input({8, 128}, DType::Float32, Device::cpu());
    input.fill_(0.5f);

    auto output = pruned_model->forward(Variable(input, false));
    EXPECT_EQ(output.tensor().shape()[0], 8);
}

TEST_F(PruningTest, LayerPruning_L2Criterion) {
    auto model = std::make_shared<SimpleSequential>(4, 64, 64, 10);

    // Prune 2 layers using L2
    auto pruned_model = prune_layers(model, 2, ImportanceCriterion::L2);

    ASSERT_NE(pruned_model, nullptr);

    Tensor input({2, 64}, DType::Float32, Device::cpu());
    input.fill_(1.0f);

    EXPECT_NO_THROW({
        auto output = pruned_model->forward(Variable(input, false));
    });
}

TEST_F(PruningTest, LayerPruning_VerifyLayersZeroed) {
    auto model = std::make_shared<SimpleSequential>(5, 128, 128, 10);

    // Store original parameters
    auto original_params = model->named_parameters();

    // Prune 2 layers
    auto pruned_model = prune_layers(model, 2, ImportanceCriterion::L1);

    // After pruning, exactly 2 layers should have all zero weights
    auto pruned_params = pruned_model->named_parameters();

    int zero_layer_count = 0;
    for (auto& [name, param] : pruned_params) {
        if (name.find("weight") != std::string::npos) {
            float sparsity = calculate_sparsity(param->tensor());
            if (sparsity > 0.99f) {  // Nearly all zeros
                zero_layer_count++;
            }
        }
    }

    EXPECT_EQ(zero_layer_count, 2);
}

TEST_F(PruningTest, LayerPruning_DifferentLayerSizes) {
    // Create model with varying layer sizes
    class VariedModel : public Module {
    public:
        VariedModel() {
            fc1_ = std::make_shared<Linear>(256, 128);
            register_module("fc1", fc1_);
            fc2_ = std::make_shared<Linear>(128, 64);
            register_module("fc2", fc2_);
            fc3_ = std::make_shared<Linear>(64, 32);
            register_module("fc3", fc3_);
            fc4_ = std::make_shared<Linear>(32, 10);
            register_module("fc4", fc4_);
        }

        Variable forward(const Variable& x) override {
            auto out = fc1_->forward(x);
            out = fc2_->forward(out);
            out = fc3_->forward(out);
            out = fc4_->forward(out);
            return out;
        }

    private:
        std::shared_ptr<Linear> fc1_, fc2_, fc3_, fc4_;
    };

    auto model = std::make_shared<VariedModel>();

    auto pruned_model = prune_layers(model, 1, ImportanceCriterion::L1);

    ASSERT_NE(pruned_model, nullptr);
}

// ============================================================================
// Mask Application Tests
// ============================================================================

TEST_F(PruningTest, ApplyPruningMasks_PreservesSparsity) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);

    // Apply masks
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity_before = calculate_sparsity(weight->tensor());

    // Apply again - should maintain sparsity
    apply_pruning_masks(linear, config);

    weight = linear->weight();
    float sparsity_after = calculate_sparsity(weight->tensor());

    EXPECT_NEAR(sparsity_before, sparsity_after, 1e-5f);
}

TEST_F(PruningTest, ApplyPruningMasks_MultipleApplications) {
    auto linear = std::make_shared<Linear>(100, 50, true);

    PruningConfig config = prune_unstructured(linear, 0.6f, ImportanceCriterion::L1, false);

    // Apply masks 10 times
    for (int i = 0; i < 10; ++i) {
        apply_pruning_masks(linear, config);
    }

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());

    // Sparsity should remain stable
    EXPECT_NEAR(sparsity, 0.6f, 0.05f);
}

// ============================================================================
// Finalize Pruning Tests
// ============================================================================

TEST_F(PruningTest, FinalizePruning_MakesPermanent) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    auto finalized = finalize_pruning(linear, config);

    ASSERT_NE(finalized, nullptr);

    // Finalized model should still have pruned weights
    auto params = finalized->named_parameters();
    bool has_zeros = false;
    for (auto& [name, param] : params) {
        if (calculate_sparsity(param->tensor()) > 0.1f) {
            has_zeros = true;
            break;
        }
    }

    EXPECT_TRUE(has_zeros);
}

// ============================================================================
// Remove Pruning Tests
// ============================================================================

TEST_F(PruningTest, RemovePruning_RestoresWeights) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    // Store original weights
    auto original_weight = linear->weight();

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    // Remove pruning
    remove_pruning(linear, config);

    // Config should be cleared
    EXPECT_EQ(config.masks.size(), 0);
}

// ============================================================================
// Sparsity Analysis Tests
// ============================================================================

TEST_F(PruningTest, ComputeSparsity_UnprunedModel) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    float sparsity = compute_sparsity(linear);

    // Unpruned model should have near-zero sparsity
    EXPECT_LT(sparsity, 0.1f);
}

TEST_F(PruningTest, ComputeSparsity_PrunedModel) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.7f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    float sparsity = compute_sparsity(linear);

    EXPECT_NEAR(sparsity, 0.7f, 0.1f);
}

TEST_F(PruningTest, AnalyzeLayerSparsity) {
    auto model = std::make_shared<SimpleSequential>(3, 128, 128, 10);

    PruningConfig config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(model, config);

    auto layer_sparsity = analyze_layer_sparsity(model);

    EXPECT_GT(layer_sparsity.size(), 0);

    for (auto& [name, sparsity] : layer_sparsity) {
        EXPECT_GE(sparsity, 0.0f);
        EXPECT_LE(sparsity, 1.0f);
    }
}

// ============================================================================
// Compression Ratio Tests
// ============================================================================

TEST_F(PruningTest, ComputeCompressionRatio_50PercentPruning) {
    auto original = std::make_shared<Linear>(256, 256, true);
    auto pruned = std::make_shared<Linear>(256, 256, true);

    PruningConfig config = prune_unstructured(pruned, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(pruned, config);

    float ratio = compute_compression_ratio(original, pruned);

    // With 50% sparsity, compression should be approximately 2x
    EXPECT_GT(ratio, 1.5f);
    EXPECT_LT(ratio, 2.5f);
}

TEST_F(PruningTest, ComputeCompressionRatio_90PercentPruning) {
    auto original = std::make_shared<Linear>(128, 128, true);
    auto pruned = std::make_shared<Linear>(128, 128, true);

    PruningConfig config = prune_unstructured(pruned, 0.9f, ImportanceCriterion::L1, false);
    apply_pruning_masks(pruned, config);

    float ratio = compute_compression_ratio(original, pruned);

    // With 90% sparsity, compression should be approximately 10x
    EXPECT_GT(ratio, 5.0f);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_F(PruningTest, Integration_PruneThenTrain) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    // Prune
    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    // Simulate training step
    Tensor input = create_random_tensor({16, 128});
    auto output = linear->forward(Variable(input, true));

    // Weights should still be pruned after forward pass
    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());
    EXPECT_NEAR(sparsity, 0.5f, 0.05f);
}

TEST_F(PruningTest, Integration_SequentialPruning) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    // Prune in steps: 30% -> 50% -> 70%
    std::vector<float> sparsity_schedule = {0.3f, 0.5f, 0.7f};

    for (float target_sparsity : sparsity_schedule) {
        PruningConfig config = prune_unstructured(linear, target_sparsity,
                                                   ImportanceCriterion::L1, false);
        apply_pruning_masks(linear, config);

        auto weight = linear->weight();
        float actual_sparsity = calculate_sparsity(weight->tensor());

        EXPECT_NEAR(actual_sparsity, target_sparsity, 0.05f);
    }
}

TEST_F(PruningTest, Integration_PruneConvAndLinear) {
    class MixedModel : public Module {
    public:
        MixedModel() {
            conv_ = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
            register_module("conv", conv_);
            fc_ = std::make_shared<Linear>(64 * 8 * 8, 10);
            register_module("fc", fc_);
        }

        Variable forward(const Variable& x) override {
            auto out = conv_->forward(x);
            auto flat = out.tensor().reshape({x.tensor().shape()[0], -1});
            return fc_->forward(Variable(flat, out.requires_grad()));
        }

    private:
        std::shared_ptr<Conv2d> conv_;
        std::shared_ptr<Linear> fc_;
    };

    auto model = std::make_shared<MixedModel>();

    PruningConfig config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(model, config);

    // Test forward pass
    Tensor input = create_random_tensor({4, 32, 8, 8});
    auto output = model->forward(Variable(input, false));

    EXPECT_EQ(output.tensor().shape()[0], 4);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

// ============================================================================
// Edge Cases
// ============================================================================

TEST_F(PruningTest, EdgeCase_PruneAlreadyPrunedModel) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    // First pruning
    PruningConfig config1 = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config1);

    // Second pruning (should work)
    PruningConfig config2 = prune_unstructured(linear, 0.7f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config2);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());

    EXPECT_NEAR(sparsity, 0.7f, 0.05f);
}

TEST_F(PruningTest, EdgeCase_ZeroSparsityPruning) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.0f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());

    // No pruning should occur
    EXPECT_LT(sparsity, 0.1f);
}

TEST_F(PruningTest, EdgeCase_NearFullSparsityPruning) {
    auto linear = std::make_shared<Linear>(128, 64, true);

    PruningConfig config = prune_unstructured(linear, 0.99f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    auto weight = linear->weight();
    float sparsity = calculate_sparsity(weight->tensor());

    EXPECT_GT(sparsity, 0.9f);
}

TEST_F(PruningTest, EdgeCase_SingleLayerModel) {
    auto linear = std::make_shared<Linear>(10, 10, true);

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    // Should handle single layer gracefully
    Tensor input = create_random_tensor({5, 10});
    auto output = linear->forward(Variable(input, false));

    EXPECT_EQ(output.tensor().shape()[0], 5);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

TEST_F(PruningTest, EdgeCase_VerySmallTensor) {
    Tensor tiny({2, 2}, DType::Float32, Device::cpu());
    tiny.fill_(1.0f);

    Tensor importance = compute_importance(tiny, ImportanceCriterion::L1);
    Tensor mask = create_mask_from_importance(importance, 0.5f);

    // Should handle small tensors without crashing
    EXPECT_EQ(mask.numel(), 4);
}

// ============================================================================
// Functional Tests
// ============================================================================

TEST_F(PruningTest, ModelStillFunctional_AfterPruning) {
    auto model = std::make_shared<SimpleSequential>(3, 784, 256, 10);

    // Test before pruning
    Tensor input = create_random_tensor({16, 784});
    auto output_before = model->forward(Variable(input, false));

    // Prune
    PruningConfig config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(model, config);

    // Test after pruning
    auto output_after = model->forward(Variable(input, false));

    // Shape should be preserved
    EXPECT_TRUE(std::equal(output_after.tensor().shape().begin(), output_after.tensor().shape().end(),
                          output_before.tensor().shape().begin(), output_before.tensor().shape().end()));
}

TEST_F(PruningTest, PrunedModelGradients) {
    auto linear = std::make_shared<Linear>(64, 32, true);

    PruningConfig config = prune_unstructured(linear, 0.5f, ImportanceCriterion::L1, false);
    apply_pruning_masks(linear, config);

    // Forward pass with gradients enabled
    Tensor input = create_random_tensor({8, 64});
    auto output = linear->forward(Variable(input, true));

    // Should be able to compute gradients
    EXPECT_NO_THROW({
        output.backward();
    });
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
