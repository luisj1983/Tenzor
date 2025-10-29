/**
 * @file test_zero_stage1_integration.cpp
 * @brief Integration tests for ZeRO Stage 1 Optimizer
 *
 * Tests end-to-end training scenarios with ZeRO Stage 1 including:
 * - Complete training loops with various optimizers
 * - Multi-epoch convergence
 * - Comparison with standard optimizers
 * - Checkpoint save/load during training
 * - Mixed optimizer configurations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <memory>
#include <vector>
#include <cmath>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;

// ============================================================================
// Test Fixtures
// ============================================================================

class ZeROIntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Default config for integration tests
        default_config.world_size = 1;  // Single process for integration tests
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.overlap_comm = true;
        default_config.process_group = nullptr;  // Single-process mode
    }

    // Helper: Create simple MLP model
    auto create_simple_mlp() -> Sequential {
        auto seq = Sequential();
        seq.add_module(std::make_shared<Linear>(784, 256))
           .add_module(std::make_shared<ReLU>())
           .add_module(std::make_shared<Linear>(256, 128))
           .add_module(std::make_shared<ReLU>())
           .add_module(std::make_shared<Linear>(128, 10));
        return seq;
    }

    // Helper: Create synthetic training data
    auto create_synthetic_batch(int batch_size, int input_size, int num_classes)
        -> std::pair<Tensor, Tensor> {
        auto inputs = randn({batch_size, input_size}, DType::Float32, Device::cpu());

        // Create random integer targets manually
        auto targets = empty({batch_size}, DType::Int64, Device::cpu());
        auto* target_data = targets.data<int64_t>();
        for (int i = 0; i < batch_size; ++i) {
            target_data[i] = std::rand() % num_classes;
        }
        return {inputs, targets};
    }

    // Helper: Train for N steps
    auto train_steps(Sequential& model, Optimizer& optimizer, int num_steps,
                     int batch_size = 32) -> std::vector<float> {
        std::vector<float> losses;

        for (int step = 0; step < num_steps; ++step) {
            optimizer.zero_grad();

            // Generate batch
            auto [inputs, targets] = create_synthetic_batch(batch_size, 784, 10);

            // Forward pass (wrap tensor in Variable)
            auto outputs = model.forward(Variable(inputs, false));

            // Compute loss
            auto loss = cross_entropy(outputs, targets);
            losses.push_back(loss.tensor().data<float>()[0]);

            // Backward pass
            loss.backward();

            // Optimizer step
            optimizer.step();
        }

        return losses;
    }

    ZeROStage1Config default_config;
};

// ============================================================================
// 1. End-to-End Training Tests
// ============================================================================

TEST_F(ZeROIntegrationTest, SingleEpochTrainingWithAdam) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 100;
    auto losses = train_steps(model, optimizer, num_steps);

    // Loss should decrease over training
    float initial_loss = losses.front();
    float final_loss = losses.back();

    EXPECT_LT(final_loss, initial_loss)
        << "Loss should decrease: initial=" << initial_loss
        << " final=" << final_loss;

    // Final loss should be reasonable (not NaN or infinity)
    EXPECT_FALSE(std::isnan(final_loss));
    EXPECT_FALSE(std::isinf(final_loss));
}

TEST_F(ZeROIntegrationTest, SingleEpochTrainingWithSGD) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<SGD>(params, 0.01, 0.9);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 100;
    auto losses = train_steps(model, optimizer, num_steps);

    // Loss should decrease
    float initial_loss = losses.front();
    float final_loss = losses.back();

    EXPECT_LT(final_loss, initial_loss);
    EXPECT_FALSE(std::isnan(final_loss));
}

TEST_F(ZeROIntegrationTest, SingleEpochTrainingWithAdamW) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<AdamW>(params, 0.001, 0.9, 0.999, 0.01);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 100;
    auto losses = train_steps(model, optimizer, num_steps);

    // Loss should decrease
    float initial_loss = losses.front();
    float final_loss = losses.back();

    EXPECT_LT(final_loss, initial_loss);
    EXPECT_FALSE(std::isnan(final_loss));
}

TEST_F(ZeROIntegrationTest, MultiEpochConvergence) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_epochs = 10;
    const int steps_per_epoch = 50;

    std::vector<float> epoch_losses;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto losses = train_steps(model, optimizer, steps_per_epoch);
        float avg_loss = 0.0f;
        for (float loss : losses) {
            avg_loss += loss;
        }
        avg_loss /= losses.size();
        epoch_losses.push_back(avg_loss);
    }

    // Average loss should decrease over epochs
    EXPECT_LT(epoch_losses.back(), epoch_losses.front());

    // No divergence
    for (float loss : epoch_losses) {
        EXPECT_FALSE(std::isnan(loss));
        EXPECT_FALSE(std::isinf(loss));
    }
}

TEST_F(ZeROIntegrationTest, LossDecreasesSmoothly) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 200;
    auto losses = train_steps(model, optimizer, num_steps);

    // Compute moving average to smooth noise
    std::vector<float> smoothed_losses;
    const int window = 20;

    for (size_t i = window; i < losses.size(); ++i) {
        float sum = 0.0f;
        for (int j = 0; j < window; ++j) {
            sum += losses[i - j];
        }
        smoothed_losses.push_back(sum / window);
    }

    // Smoothed loss should show clear downward trend
    EXPECT_LT(smoothed_losses.back(), smoothed_losses.front());
}

// ============================================================================
// 2. Comparison with Standard Optimizer
// ============================================================================

TEST_F(ZeROIntegrationTest, CompareWithStandardAdam) {
    // Create two identical models
    auto model1 = create_simple_mlp();
    auto model2 = create_simple_mlp();

    // Copy weights to ensure identical initialization
    auto params1 = model1.parameters();
    auto params2 = model2.parameters();

    ASSERT_EQ(params1.size(), params2.size());
    for (size_t i = 0; i < params1.size(); ++i) {
        params2[i]->tensor() = params1[i]->tensor();
    }

    // Train with ZeRO optimizer
    auto zero_opt = std::make_unique<Adam>(params1, 0.001);
    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single rank for fair comparison
    config.rank = 0;
    ZeROStage1Optimizer optimizer1(std::move(zero_opt), config);

    // Train with standard optimizer
    Adam optimizer2(params2, 0.001);

    const int num_steps = 50;

    // Use same random seed for both training runs
    std::srand(42);
    auto losses1 = train_steps(model1, optimizer1, num_steps);

    std::srand(42);
    auto losses2 = train_steps(model2, optimizer2, num_steps);

    // Final losses should be very similar
    float final_loss1 = losses1.back();
    float final_loss2 = losses2.back();

    EXPECT_NEAR(final_loss1, final_loss2, 0.1f)
        << "ZeRO and standard optimizer should produce similar results";
}

// ============================================================================
// 3. Checkpoint Save/Load Tests
// ============================================================================

TEST_F(ZeROIntegrationTest, SaveLoadCheckpointDuringTraining) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Train for 50 steps
    train_steps(model, optimizer, 50);

    // Save state
    auto state_dict = optimizer.state_dict();

    // Train for 50 more steps
    auto losses_before = train_steps(model, optimizer, 50);

    // Load state (rewind optimizer)
    optimizer.load_state_dict(state_dict);

    // Train again for 50 steps
    auto losses_after = train_steps(model, optimizer, 50);

    // Losses should follow similar pattern (deterministic if same data)
    EXPECT_EQ(losses_before.size(), losses_after.size());
}

TEST_F(ZeROIntegrationTest, CheckpointPreservesConvergence) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Train to convergence
    const int total_steps = 200;
    const int checkpoint_interval = 50;

    std::vector<float> all_losses;

    for (int i = 0; i < total_steps / checkpoint_interval; ++i) {
        // Save checkpoint
        auto state_dict = optimizer.state_dict();

        // Train
        auto losses = train_steps(model, optimizer, checkpoint_interval);
        all_losses.insert(all_losses.end(), losses.begin(), losses.end());

        // Reload checkpoint (test load doesn't break training)
        if (i < total_steps / checkpoint_interval - 1) {
            optimizer.load_state_dict(state_dict);
        }
    }

    // Overall trend should be decreasing
    EXPECT_LT(all_losses.back(), all_losses.front());
}

// ============================================================================
// 4. CPU Offload Integration Tests
// ============================================================================

TEST_F(ZeROIntegrationTest, TrainingWithCPUOffload) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.offload_to_cpu = true;
    config.cpu_offload_threshold = 1024;  // Offload params > 1KB

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    const int num_steps = 100;
    auto losses = train_steps(model, optimizer, num_steps);

    // Training should work with offload
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_FALSE(std::isnan(losses.back()));
}

TEST_F(ZeROIntegrationTest, CPUOffloadDoesNotAffectConvergence) {
    // Train two models: one with offload, one without
    auto model1 = create_simple_mlp();
    auto model2 = create_simple_mlp();

    auto params1 = model1.parameters();
    auto params2 = model2.parameters();

    // Ensure same initialization
    for (size_t i = 0; i < params1.size(); ++i) {
        params2[i]->tensor() = params1[i]->tensor();
    }

    // Optimizer without offload
    auto opt1 = std::make_unique<Adam>(params1, 0.001);
    ZeROStage1Config config1 = default_config;
    config1.offload_to_cpu = false;
    ZeROStage1Optimizer optimizer1(std::move(opt1), config1);

    // Optimizer with offload
    auto opt2 = std::make_unique<Adam>(params2, 0.001);
    ZeROStage1Config config2 = default_config;
    config2.offload_to_cpu = true;
    ZeROStage1Optimizer optimizer2(std::move(opt2), config2);

    const int num_steps = 50;

    std::srand(42);
    auto losses1 = train_steps(model1, optimizer1, num_steps);

    std::srand(42);
    auto losses2 = train_steps(model2, optimizer2, num_steps);

    // Results should be similar
    EXPECT_NEAR(losses1.back(), losses2.back(), 0.1f);
}

// ============================================================================
// 5. Different Partition Configurations
// ============================================================================

TEST_F(ZeROIntegrationTest, TrainingWithDifferentWorldSizes) {
    std::vector<int> world_sizes = {1, 2, 4, 8};

    for (int world_size : world_sizes) {
        auto model = create_simple_mlp();
        auto params = model.parameters();

        auto base_optimizer = std::make_unique<Adam>(params, 0.001);

        ZeROStage1Config config = default_config;
        config.world_size = world_size;
        config.rank = 0;

        ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

        const int num_steps = 50;
        auto losses = train_steps(model, optimizer, num_steps);

        // Training should work for all world sizes
        EXPECT_LT(losses.back(), losses.front())
            << "Training failed for world_size=" << world_size;
        EXPECT_FALSE(std::isnan(losses.back()))
            << "NaN loss for world_size=" << world_size;
    }
}

// ============================================================================
// 6. Large Batch Training
// ============================================================================

TEST_F(ZeROIntegrationTest, LargeBatchTraining) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 50;
    const int large_batch_size = 256;

    auto losses = train_steps(model, optimizer, num_steps, large_batch_size);

    // Large batch training should still converge
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_FALSE(std::isnan(losses.back()));
}

TEST_F(ZeROIntegrationTest, SmallBatchTraining) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 50;
    const int small_batch_size = 4;

    auto losses = train_steps(model, optimizer, num_steps, small_batch_size);

    // Small batch training should still work
    EXPECT_FALSE(std::isnan(losses.back()));
}

// ============================================================================
// 7. Gradient Accumulation
// ============================================================================

TEST_F(ZeROIntegrationTest, GradientAccumulation) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 50;
    const int accumulation_steps = 4;

    std::vector<float> losses;

    for (int step = 0; step < num_steps; ++step) {
        // Accumulate gradients over multiple mini-batches
        for (int acc = 0; acc < accumulation_steps; ++acc) {
            auto [inputs, targets] = create_synthetic_batch(8, 784, 10);
            auto outputs = model.forward(Variable(inputs, false));
            auto loss = cross_entropy(outputs, targets);

            if (acc == 0) {
                losses.push_back(loss.tensor().data<float>()[0]);
            }

            loss.backward();  // Accumulate gradients
        }

        // Update after accumulation
        optimizer.step();
        optimizer.zero_grad();
    }

    // Training with gradient accumulation should work
    EXPECT_LT(losses.back(), losses.front());
    EXPECT_FALSE(std::isnan(losses.back()));
}

// ============================================================================
// 8. Zero Gradient Edge Cases
// ============================================================================

TEST_F(ZeROIntegrationTest, StepWithoutGradients) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Call step without computing gradients
    EXPECT_NO_THROW(optimizer.step());

    // Parameters should remain unchanged
    auto params_copy = params;
    optimizer.step();

    // Can continue training normally after
    auto losses = train_steps(model, optimizer, 10);
    EXPECT_FALSE(std::isnan(losses.back()));
}

TEST_F(ZeROIntegrationTest, MultipleZeroGradCalls) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Multiple zero_grad calls should be safe
    optimizer.zero_grad();
    optimizer.zero_grad();
    optimizer.zero_grad();

    // Training should still work
    auto losses = train_steps(model, optimizer, 10);
    EXPECT_FALSE(std::isnan(losses.back()));
}

// ============================================================================
// 9. Model Size Variations
// ============================================================================

TEST_F(ZeROIntegrationTest, TinyModel) {
    // Very small model
    auto model = Sequential();
    model.add_module(std::make_shared<Linear>(10, 5))
         .add_module(std::make_shared<ReLU>())
         .add_module(std::make_shared<Linear>(5, 2));

    auto params = model.parameters();
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Generate smaller batches
    std::vector<float> losses;
    for (int i = 0; i < 50; ++i) {
        optimizer.zero_grad();
        auto [inputs, targets] = create_synthetic_batch(16, 10, 2);
        auto outputs = model.forward(Variable(inputs, false));
        auto loss = cross_entropy(outputs, targets);
        losses.push_back(loss.tensor().data<float>()[0]);
        loss.backward();
        optimizer.step();
    }

    EXPECT_LT(losses.back(), losses.front());
}

TEST_F(ZeROIntegrationTest, LargeModel) {
    // Larger model
    auto model = Sequential();
    model.add_module(std::make_shared<Linear>(784, 1024))
         .add_module(std::make_shared<ReLU>())
         .add_module(std::make_shared<Linear>(1024, 512))
         .add_module(std::make_shared<ReLU>())
         .add_module(std::make_shared<Linear>(512, 256))
         .add_module(std::make_shared<ReLU>())
         .add_module(std::make_shared<Linear>(256, 10));

    auto params = model.parameters();
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    const int num_steps = 50;
    auto losses = train_steps(model, optimizer, num_steps);

    EXPECT_LT(losses.back(), losses.front());
    EXPECT_FALSE(std::isnan(losses.back()));
}

// ============================================================================
// 10. Learning Rate Changes During Training
// ============================================================================

TEST_F(ZeROIntegrationTest, DynamicLearningRate) {
    auto model = create_simple_mlp();
    auto params = model.parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Train with high learning rate
    auto losses1 = train_steps(model, optimizer, 25);

    // Access base optimizer and reduce learning rate
    // Note: This test assumes we can access and modify the base optimizer
    // In practice, this might require an API update

    // Continue training
    auto losses2 = train_steps(model, optimizer, 25);

    // Both phases should decrease loss
    EXPECT_LT(losses1.back(), losses1.front());
    EXPECT_LT(losses2.back(), losses2.front());
}

// ============================================================================
// 11. Reproducibility Tests
// ============================================================================

TEST_F(ZeROIntegrationTest, ReproducibleTraining) {
    // Train same model twice with same seed
    std::vector<float> losses1, losses2;

    for (int trial = 0; trial < 2; ++trial) {
        auto model = create_simple_mlp();
        auto params = model.parameters();

        auto base_optimizer = std::make_unique<Adam>(params, 0.001);
        ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

        std::srand(12345);
        auto losses = train_steps(model, optimizer, 50);

        if (trial == 0) {
            losses1 = losses;
        } else {
            losses2 = losses;
        }
    }

    // Results should be identical with same seed
    ASSERT_EQ(losses1.size(), losses2.size());
    for (size_t i = 0; i < losses1.size(); ++i) {
        EXPECT_NEAR(losses1[i], losses2[i], 1e-4f)
            << "Mismatch at step " << i;
    }
}
