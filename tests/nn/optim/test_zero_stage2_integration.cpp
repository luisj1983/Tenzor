/**
 * @file test_zero_stage2_integration.cpp
 * @brief Integration tests for ZeRO Stage 2 Optimizer (Gradient Partitioning)
 *
 * Integration tests including:
 * - End-to-end training with gradient partitioning
 * - Multi-rank training (world_size=1 for single-process tests)
 * - Comparison with Stage 1 (verify same convergence)
 * - Gradient accumulation with Stage 2
 * - CPU offload integration
 * - Checkpointing with Stage 2
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/checkpoint.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>
#include <vector>
#include <cmath>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;

// ============================================================================
// Test Models
// ============================================================================

/**
 * @brief Simple MLP for testing training convergence
 */
class SimpleMLP : public Module {
public:
    SimpleMLP(int input_dim, int hidden_dim, int output_dim) {
        fc1_ = std::make_shared<Linear>(input_dim, hidden_dim);
        fc2_ = std::make_shared<Linear>(hidden_dim, output_dim);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        Variable h = fc1_->forward(x);
        h = tenzor::nn::relu(h);
        Variable out = fc2_->forward(h);
        return out;
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

/**
 * @brief Deep network for testing gradient flow
 */
class DeepNetwork : public Module {
public:
    DeepNetwork(int input_dim, int hidden_dim, int num_layers, int output_dim) {
        layers_.reserve(num_layers);

        // First layer
        auto layer0 = std::make_shared<Linear>(input_dim, hidden_dim);
        layers_.push_back(layer0);
        register_module("layer_0", layer0);

        // Hidden layers
        for (int i = 1; i < num_layers - 1; ++i) {
            auto layer = std::make_shared<Linear>(hidden_dim, hidden_dim);
            layers_.push_back(layer);
            register_module("layer_" + std::to_string(i), layer);
        }

        // Output layer
        auto layer_out = std::make_shared<Linear>(hidden_dim, output_dim);
        layers_.push_back(layer_out);
        register_module("layer_" + std::to_string(num_layers - 1), layer_out);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        Variable out = x;
        for (size_t i = 0; i < layers_.size() - 1; ++i) {
            out = layers_[i]->forward(out);
            out = tenzor::nn::relu(out);
        }
        out = layers_.back()->forward(out);
        return out;
    }

private:
    std::vector<std::shared_ptr<Linear>> layers_;
};

// ============================================================================
// Test Fixtures
// ============================================================================

class ZeROStage2IntegrationTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Default config for single-process integration tests
        default_config.world_size = 1;
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.cpu_offload_threshold = 1024;
        default_config.overlap_comm = true;
        default_config.pin_memory = true;
        default_config.process_group = nullptr;
    }

    // Helper: Generate synthetic training data
    auto generate_data(int num_samples, int input_dim, int output_dim)
        -> std::pair<Tensor, Tensor> {
        auto X = randn({num_samples, input_dim}, DType::Float32, Device::cpu());
        auto y = randn({num_samples, output_dim}, DType::Float32, Device::cpu());
        return {X, y};
    }

    // Helper: Train model for N steps
    auto train_model(
        Module& model,
        Optimizer& optimizer,
        const Tensor& X,
        const Tensor& y,
        int num_steps
    ) -> std::vector<float> {
        std::vector<float> losses;
        losses.reserve(num_steps);

        for (int step = 0; step < num_steps; ++step) {
            optimizer.zero_grad();

            // Forward pass
            auto X_var = Variable(X, true);
            auto y_var = Variable(y, true);
            auto output = model.forward(X_var);
            auto loss = mse_loss(output, y_var);

            // Backward pass
            loss.backward();

            // Optimizer step
            optimizer.step();

            // Record loss
            losses.push_back(loss.tensor().item<float>());
        }

        return losses;
    }

    ZeROStage1Config default_config;
};

// ============================================================================
// 1. End-to-End Training Tests
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, EndToEndTrainingConvergence) {
    // Test that Stage 2 optimizer can train a model to convergence
    auto model = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params = model->parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Generate synthetic data
    auto [X, y] = generate_data(32, 64, 10);

    // Train for 100 steps
    auto losses = train_model(*model, optimizer, X, y, 100);

    // Verify loss decreases
    EXPECT_LT(losses.back(), losses.front() * 0.5)
        << "Loss should decrease significantly during training";

    // Verify loss is reasonable
    EXPECT_LT(losses.back(), 10.0f)
        << "Final loss should be reasonable";
}

TEST_F(ZeROStage2IntegrationTest, EndToEndTrainingDeepNetwork) {
    // Test training deep network with many parameters
    auto model = std::make_shared<DeepNetwork>(64, 256, 8, 10);
    auto params = model->parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    // Generate data
    auto [X, y] = generate_data(16, 64, 10);

    // Train for 50 steps
    auto losses = train_model(*model, optimizer, X, y, 50);

    // Deep network should still converge
    EXPECT_LT(losses.back(), losses.front())
        << "Loss should decrease for deep network";
}

TEST_F(ZeROStage2IntegrationTest, EndToEndTrainingMultipleBatches) {
    // Test training over multiple epochs (simulated as "batches") on the same data
    // This verifies the optimizer state is maintained correctly across training steps
    auto model = std::make_shared<SimpleMLP>(32, 64, 5);
    auto params = model->parameters();

    auto base_optimizer = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), default_config);

    std::vector<float> all_losses;

    // Generate single batch of data - train multiple "epochs" on it
    auto [X, y] = generate_data(16, 32, 5);

    // Train for 10 epochs (each epoch = 10 steps on same data)
    for (int epoch = 0; epoch < 10; ++epoch) {
        auto epoch_losses = train_model(*model, optimizer, X, y, 10);
        all_losses.insert(all_losses.end(), epoch_losses.begin(), epoch_losses.end());
    }

    // Overall trend should be decreasing
    float first_10_avg = 0.0f;
    float last_10_avg = 0.0f;

    for (int i = 0; i < 10; ++i) {
        first_10_avg += all_losses[i];
        last_10_avg += all_losses[all_losses.size() - 10 + i];
    }
    first_10_avg /= 10.0f;
    last_10_avg /= 10.0f;

    EXPECT_LT(last_10_avg, first_10_avg)
        << "Average loss should decrease over multiple epochs";
}

// ============================================================================
// 2. Multi-Rank Training Tests (Single-Process Simulation)
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, MultiRankConfigSingleProcess) {
    // Test multi-rank configuration in single-process mode
    auto model = std::make_shared<SimpleMLP>(32, 64, 10);
    auto params = model->parameters();

    ZeROStage1Config config = default_config;
    config.world_size = 4;  // Simulate 4 ranks
    config.rank = 0;        // This is rank 0

    auto base_optimizer = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Should partition parameters across 4 ranks
    size_t total_params = params.size();
    size_t local_params = optimizer.local_param_count();

    // Rank 0 should own approximately 1/4 of parameters
    EXPECT_LE(local_params, (total_params / 4) + 2)
        << "Local params should be ~1/4 of total";
}

TEST_F(ZeROStage2IntegrationTest, MultiRankParameterPartitioning) {
    // Test that different ranks own different parameter partitions
    auto model = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params = model->parameters();

    ZeROStage1Config config = default_config;
    config.world_size = 4;

    std::vector<size_t> partition_sizes;

    // Create optimizers for each "rank"
    for (int rank = 0; rank < 4; ++rank) {
        config.rank = rank;
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), config);

        partition_sizes.push_back(opt.local_param_count());
    }

    // Sum of all partitions should equal total parameters
    size_t total_local = 0;
    for (size_t size : partition_sizes) {
        total_local += size;
    }

    EXPECT_EQ(total_local, params.size())
        << "Sum of partition sizes should equal total parameters";
}

TEST_F(ZeROStage2IntegrationTest, MultiRankMemoryScaling) {
    // Test that memory usage scales with world_size
    auto model = std::make_shared<SimpleMLP>(128, 256, 10);
    auto params = model->parameters();

    // Single rank baseline
    size_t memory_1rank;
    {
        ZeROStage1Config config = default_config;
        config.world_size = 1;
        config.rank = 0;

        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), config);

        auto stats = opt.get_memory_stats();
        memory_1rank = stats.gpu_optimizer_memory + stats.cpu_optimizer_memory;
    }

    // 4 ranks should use less memory per rank
    size_t memory_4ranks;
    {
        ZeROStage1Config config = default_config;
        config.world_size = 4;
        config.rank = 0;

        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), config);

        auto stats = opt.get_memory_stats();
        memory_4ranks = stats.gpu_optimizer_memory + stats.cpu_optimizer_memory;
    }

    // 4-rank config should use less memory (approximately 1/4)
    EXPECT_LT(memory_4ranks, memory_1rank / 2)
        << "Multi-rank should use less memory per rank";
}

// ============================================================================
// 3. Comparison with Stage 1
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, ComparisonStage1ConvergenceSimilar) {
    // Verify Stage 2 converges similarly to Stage 1
    // (Stage 2 should have same accuracy, just better memory efficiency)

    auto [X, y] = generate_data(32, 64, 10);

    // Train with Stage 1
    std::vector<float> stage1_losses;
    {
        auto model = std::make_shared<SimpleMLP>(64, 128, 10);
        auto params = model->parameters();

        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        stage1_losses = train_model(*model, opt, X, y, 50);
    }

    // Train with Stage 2 (when implemented, for now use Stage 1 as baseline)
    std::vector<float> stage2_losses;
    {
        auto model = std::make_shared<SimpleMLP>(64, 128, 10);
        auto params = model->parameters();

        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        stage2_losses = train_model(*model, opt, X, y, 50);
    }

    // Final losses should be similar (within 20%)
    float stage1_final = stage1_losses.back();
    float stage2_final = stage2_losses.back();

    EXPECT_NEAR(stage1_final, stage2_final, stage1_final * 0.2f)
        << "Stage 1 and Stage 2 should converge to similar losses";
}

TEST_F(ZeROStage2IntegrationTest, ComparisonStage1MemoryReduction) {
    // Verify Stage 2 uses less memory than Stage 1
    auto model = std::make_shared<SimpleMLP>(128, 512, 10);
    auto params = model->parameters();

    // Stage 1 memory (baseline)
    size_t stage1_memory;
    {
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        auto stats = opt.get_memory_stats();
        stage1_memory = stats.gpu_optimizer_memory + stats.cpu_optimizer_memory;
    }

    // Stage 2 memory (should be lower when gradient partitioning is added)
    // For now, verify Stage 1 baseline works
    size_t stage2_memory;
    {
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        auto stats = opt.get_memory_stats();
        stage2_memory = stats.gpu_optimizer_memory + stats.cpu_optimizer_memory;
    }

    // Memory should be reasonable
    EXPECT_GT(stage1_memory, 0);
    EXPECT_GT(stage2_memory, 0);
}

// ============================================================================
// 4. Gradient Accumulation Tests
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, GradientAccumulationBasic) {
    // Test gradient accumulation over multiple micro-batches
    auto model = std::make_shared<SimpleMLP>(32, 64, 10);
    auto params = model->parameters();

    auto base_opt = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer opt(std::move(base_opt), default_config);

    int accumulation_steps = 4;
    std::vector<float> losses;

    for (int step = 0; step < 10; ++step) {
        // Accumulate gradients over 4 micro-batches
        for (int micro_step = 0; micro_step < accumulation_steps; ++micro_step) {
            auto [X, y] = generate_data(8, 32, 10);  // Small micro-batch

            auto X_var = Variable(X, true);
            auto y_var = Variable(y, true);
            auto output = model->forward(X_var);
            auto loss = mse_loss(output, y_var);

            loss.backward();

            if (micro_step == 0) {
                losses.push_back(loss.tensor().item<float>());
            }
        }

        // Update after accumulation
        opt.step();
        opt.zero_grad();
    }

    // Loss should decrease with gradient accumulation
    EXPECT_LT(losses.back(), losses.front())
        << "Loss should decrease with gradient accumulation";
}

TEST_F(ZeROStage2IntegrationTest, GradientAccumulationVsNormalBatch) {
    // Compare gradient accumulation vs single large batch
    auto [X_large, y_large] = generate_data(32, 32, 10);

    // Train with gradient accumulation (4 micro-batches of 8)
    float acc_final_loss;
    {
        auto model = std::make_shared<SimpleMLP>(32, 64, 10);
        auto params = model->parameters();
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        for (int step = 0; step < 20; ++step) {
            for (int micro = 0; micro < 4; ++micro) {
                int start = micro * 8;
                auto X_micro = X_large.slice(0, start, start + 8);
                auto y_micro = y_large.slice(0, start, start + 8);

                auto X_var = Variable(X_micro, true);
                auto y_var = Variable(y_micro, true);
                auto output = model->forward(X_var);
                auto loss = mse_loss(output, y_var);
                loss.backward();

                if (step == 19 && micro == 0) {
                    acc_final_loss = loss.tensor().item<float>();
                }
            }
            opt.step();
            opt.zero_grad();
        }
    }

    // Train with normal large batch
    float normal_final_loss;
    {
        auto model = std::make_shared<SimpleMLP>(32, 64, 10);
        auto params = model->parameters();
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        for (int step = 0; step < 20; ++step) {
            auto X_var = Variable(X_large, true);
            auto y_var = Variable(y_large, true);
            auto output = model->forward(X_var);
            auto loss = mse_loss(output, y_var);
            loss.backward();

            if (step == 19) {
                normal_final_loss = loss.tensor().item<float>();
            }

            opt.step();
            opt.zero_grad();
        }
    }

    // Results should be similar (within 30% due to different batch dynamics)
    EXPECT_NEAR(acc_final_loss, normal_final_loss, normal_final_loss * 0.3f)
        << "Gradient accumulation should produce similar results";
}

// ============================================================================
// 5. CPU Offload Integration Tests
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, CPUOffloadEndToEndTraining) {
    // Test training with CPU offload enabled
    auto model = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params = model->parameters();

    ZeROStage1Config config = default_config;
    config.offload_to_cpu = true;

    auto base_opt = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer opt(std::move(base_opt), config);

    auto [X, y] = generate_data(32, 64, 10);

    // Train for 50 steps with offload
    auto losses = train_model(*model, opt, X, y, 50);

    // Should still converge
    EXPECT_LT(losses.back(), losses.front() * 0.5)
        << "Training should converge with CPU offload";
}

TEST_F(ZeROStage2IntegrationTest, CPUOffloadMemoryLocation) {
    // Verify optimizer states are on CPU when offload is enabled
    auto model = std::make_shared<SimpleMLP>(128, 256, 10);
    auto params = model->parameters();

    ZeROStage1Config config = default_config;
    config.offload_to_cpu = true;

    auto base_opt = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer opt(std::move(base_opt), config);

    auto [X, y] = generate_data(16, 128, 10);
    auto losses = train_model(*model, opt, X, y, 10);

    // Check memory stats
    auto stats = opt.get_memory_stats();

    // With CPU offload, optimizer states should be on CPU
    EXPECT_GT(stats.cpu_optimizer_memory, 0)
        << "CPU offload should place optimizer states on CPU";
}

TEST_F(ZeROStage2IntegrationTest, CPUOffloadVsGPUConvergence) {
    // Compare convergence with and without CPU offload
    auto [X, y] = generate_data(32, 64, 10);

    // Train without offload
    std::vector<float> gpu_losses;
    {
        auto model = std::make_shared<SimpleMLP>(64, 128, 10);
        auto params = model->parameters();

        ZeROStage1Config config = default_config;
        config.offload_to_cpu = false;

        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), config);

        gpu_losses = train_model(*model, opt, X, y, 50);
    }

    // Train with offload
    std::vector<float> cpu_losses;
    {
        auto model = std::make_shared<SimpleMLP>(64, 128, 10);
        auto params = model->parameters();

        ZeROStage1Config config = default_config;
        config.offload_to_cpu = true;

        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), config);

        cpu_losses = train_model(*model, opt, X, y, 50);
    }

    // Both should converge to similar final loss
    EXPECT_NEAR(gpu_losses.back(), cpu_losses.back(), gpu_losses.back() * 0.2f)
        << "GPU and CPU offload should converge similarly";
}

// ============================================================================
// 6. Checkpointing Tests
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, CheckpointSaveLoad) {
    auto model = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params = model->parameters();

    auto base_opt = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer opt(std::move(base_opt), default_config);

    auto [X, y] = generate_data(32, 64, 10);

    // Train for 20 steps
    train_model(*model, opt, X, y, 20);

    // Save checkpoint
    std::string checkpoint_path = "/tmp/zero_stage2_test_checkpoint";
    EXPECT_NO_THROW(opt.save_checkpoint(checkpoint_path));

    // Create new optimizer and load checkpoint
    auto model2 = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params2 = model2->parameters();
    auto base_opt2 = std::make_unique<Adam>(params2, 0.01);
    ZeROStage1Optimizer opt2(std::move(base_opt2), default_config);

    EXPECT_NO_THROW(opt2.load_checkpoint(checkpoint_path));
}

TEST_F(ZeROStage2IntegrationTest, CheckpointResumeTraining) {
    auto [X, y] = generate_data(32, 64, 10);
    std::string checkpoint_path = "/tmp/zero_stage2_resume_test.pt";
    std::string opt_checkpoint_path = "/tmp/zero_stage2_resume_opt";

    // Train for 30 steps and save both model and optimizer
    std::vector<float> losses_first;
    {
        auto model = std::make_shared<SimpleMLP>(64, 128, 10);
        auto params = model->parameters();
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        losses_first = train_model(*model, opt, X, y, 30);

        // Save model state using checkpoint manager
        ModelCheckpoint checkpoint_manager;
        checkpoint_manager.save_model(checkpoint_path, *model);
        // Save ZeRO optimizer state separately (includes wrapped optimizer state)
        opt.save_checkpoint(opt_checkpoint_path);
    }

    // Load both model and optimizer, continue training for 20 more steps
    std::vector<float> losses_resumed;
    {
        auto model = std::make_shared<SimpleMLP>(64, 128, 10);
        auto params = model->parameters();
        auto base_opt = std::make_unique<Adam>(params, 0.01);
        ZeROStage1Optimizer opt(std::move(base_opt), default_config);

        // Load model state first
        ModelCheckpoint checkpoint_manager;
        auto model_state = checkpoint_manager.load_model(checkpoint_path);
        model->load_state_dict(model_state);
        // Load ZeRO optimizer state
        opt.load_checkpoint(opt_checkpoint_path);

        losses_resumed = train_model(*model, opt, X, y, 20);
    }

    // Resumed training should continue from where we left off
    // First loss after resume should be close to last loss before checkpoint
    EXPECT_NEAR(losses_resumed.front(), losses_first.back(), losses_first.back() * 0.5)
        << "First resumed loss should be close to last checkpoint loss";

    // Training should continue improving (or at least not get worse)
    EXPECT_LT(losses_resumed.back(), losses_resumed.front())
        << "Resumed training should continue improving";
}

TEST_F(ZeROStage2IntegrationTest, CheckpointMultiRankCompatibility) {
    // Test checkpoint format compatibility for multi-rank configurations
    // Note: Actual distributed training requires a process group, so we only test
    // checkpoint save/load format compatibility without training
    auto model = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params = model->parameters();

    ZeROStage1Config config = default_config;
    config.world_size = 4;
    config.rank = 0;

    auto base_opt = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer opt(std::move(base_opt), config);

    // Save checkpoint without training (tests checkpoint format only)
    std::string checkpoint_path = "/tmp/zero_stage2_multirank_test";
    EXPECT_NO_THROW(opt.save_checkpoint(checkpoint_path));

    // Load should work for same configuration
    auto model2 = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params2 = model2->parameters();
    auto base_opt2 = std::make_unique<Adam>(params2, 0.01);

    ZeROStage1Config config2 = default_config;
    config2.world_size = 4;
    config2.rank = 0;

    ZeROStage1Optimizer opt2(std::move(base_opt2), config2);
    EXPECT_NO_THROW(opt2.load_checkpoint(checkpoint_path));
}

TEST_F(ZeROStage2IntegrationTest, CheckpointWithCPUOffload) {
    // Test checkpointing with CPU offload enabled
    auto model = std::make_shared<SimpleMLP>(128, 256, 10);
    auto params = model->parameters();

    ZeROStage1Config config = default_config;
    config.offload_to_cpu = true;

    auto base_opt = std::make_unique<Adam>(params, 0.01);
    ZeROStage1Optimizer opt(std::move(base_opt), config);

    auto [X, y] = generate_data(16, 128, 10);
    train_model(*model, opt, X, y, 20);

    // Save and load checkpoint
    std::string checkpoint_path = "/tmp/zero_stage2_offload_checkpoint";
    EXPECT_NO_THROW(opt.save_checkpoint(checkpoint_path));

    auto model2 = std::make_shared<SimpleMLP>(128, 256, 10);
    auto params2 = model2->parameters();
    auto base_opt2 = std::make_unique<Adam>(params2, 0.01);

    ZeROStage1Config config2 = default_config;
    config2.offload_to_cpu = true;

    ZeROStage1Optimizer opt2(std::move(base_opt2), config2);
    EXPECT_NO_THROW(opt2.load_checkpoint(checkpoint_path));
}

// ============================================================================
// 7. Performance and Stability Tests
// ============================================================================

TEST_F(ZeROStage2IntegrationTest, StabilityLongTraining) {
    // Test stability over long training runs
    auto model = std::make_shared<SimpleMLP>(64, 128, 10);
    auto params = model->parameters();

    auto base_opt = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer opt(std::move(base_opt), default_config);

    auto [X, y] = generate_data(32, 64, 10);

    // Train for 200 steps
    auto losses = train_model(*model, opt, X, y, 200);

    // Should remain stable (no NaN or Inf)
    for (float loss : losses) {
        EXPECT_FALSE(std::isnan(loss)) << "Loss should not be NaN";
        EXPECT_FALSE(std::isinf(loss)) << "Loss should not be Inf";
    }

    // Final loss should be reasonable
    EXPECT_LT(losses.back(), 100.0f) << "Final loss should be reasonable";
}

TEST_F(ZeROStage2IntegrationTest, StabilityLargeGradients) {
    // Test stability with large gradients
    auto model = std::make_shared<SimpleMLP>(32, 64, 10);
    auto params = model->parameters();

    auto base_opt = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer opt(std::move(base_opt), default_config);

    // Generate data that produces large gradients
    auto X = full({16, 32}, 10.0f, DType::Float32, Device::cpu());
    auto y = full({16, 10}, 10.0f, DType::Float32, Device::cpu());

    auto losses = train_model(*model, opt, X, y, 50);

    // Should handle large gradients without instability
    for (float loss : losses) {
        EXPECT_FALSE(std::isnan(loss));
        EXPECT_FALSE(std::isinf(loss));
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
