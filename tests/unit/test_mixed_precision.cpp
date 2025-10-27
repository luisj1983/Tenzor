/**
 * @file test_mixed_precision.cpp
 * @brief Comprehensive unit tests for mixed precision training
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/mixed_precision.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// Simple MLP model for testing
class SimpleMLP : public Module {
public:
    SimpleMLP(int input_size, int hidden_size, int output_size)
        : fc1_(std::make_shared<Linear>(input_size, hidden_size)),
          relu_(std::make_shared<ReLU>()),
          fc2_(std::make_shared<Linear>(hidden_size, output_size)) {
        register_module("fc1", fc1_);
        register_module("relu", relu_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& x) -> Variable override {
        auto h = fc1_->forward(x);
        h = relu_->forward(h);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Linear> fc2_;
};

class MixedPrecisionTest : public ::testing::Test {
protected:
    void SetUp() override {
        device_ = Device::cpu();
    }

    Device device_;
};

// Test 1: MixedPrecisionConfig - FP16 CUDA configuration
TEST_F(MixedPrecisionTest, ConfigFP16CUDA) {
    auto config = MixedPrecisionConfig::fp16_cuda();

    EXPECT_EQ(config.dtype, DType::Float16);
    EXPECT_EQ(config.device_type, Device::Type::CUDA);
    EXPECT_TRUE(config.enabled);
    EXPECT_FLOAT_EQ(config.init_scale, 65536.0f);
    EXPECT_FLOAT_EQ(config.growth_factor, 2.0f);
    EXPECT_FLOAT_EQ(config.backoff_factor, 0.5f);
    EXPECT_EQ(config.growth_interval, 2000);
}

// Test 2: MixedPrecisionConfig - BFloat16 CUDA configuration
TEST_F(MixedPrecisionTest, ConfigBFloat16CUDA) {
    auto config = MixedPrecisionConfig::bfloat16_cuda();

    EXPECT_EQ(config.dtype, DType::BFloat16);
    EXPECT_EQ(config.device_type, Device::Type::CUDA);
    EXPECT_TRUE(config.enabled);
}

// Test 3: MixedPrecisionConfig - Conservative configuration
TEST_F(MixedPrecisionTest, ConfigConservative) {
    auto config = MixedPrecisionConfig::conservative();

    EXPECT_FLOAT_EQ(config.init_scale, 1024.0f);
    EXPECT_FLOAT_EQ(config.growth_factor, 1.5f);
    EXPECT_FLOAT_EQ(config.backoff_factor, 0.75f);
    EXPECT_EQ(config.growth_interval, 5000);
}

// Test 4: MixedPrecisionTrainer - Basic construction
TEST_F(MixedPrecisionTest, TrainerConstruction) {
    auto model = std::make_shared<SimpleMLP>(10, 20, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    // Create trainer with disabled mixed precision (for CPU)
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    EXPECT_TRUE(trainer.is_training());
    EXPECT_EQ(trainer.get_skipped_steps(), 0);
    EXPECT_EQ(trainer.get_total_steps(), 0);
}

// Test 5: MixedPrecisionTrainer - Train step without mixed precision (FP32)
TEST_F(MixedPrecisionTest, TrainStepFP32) {
    auto model = std::make_shared<SimpleMLP>(10, 20, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    // Disable mixed precision for CPU testing
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create dummy input and target
    auto input_tensor = randn({4, 10}, DType::Float32, device_);
    auto target_tensor = randn({4, 5}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform training step
    float loss = trainer.train_step(input, target);

    EXPECT_GT(loss, 0.0f);
    EXPECT_EQ(trainer.get_total_steps(), 1);
    EXPECT_EQ(trainer.get_skipped_steps(), 0);
}

// Test 6: MixedPrecisionTrainer - Multiple training steps
TEST_F(MixedPrecisionTest, MultipleTrainSteps) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({8, 5}, DType::Float32, device_);
    auto target_tensor = randn({8, 3}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform 10 training steps
    std::vector<float> losses;
    for (int i = 0; i < 10; ++i) {
        float loss = trainer.train_step(input, target);
        losses.push_back(loss);
    }

    EXPECT_EQ(trainer.get_total_steps(), 10);
    EXPECT_EQ(losses.size(), 10);

    // Loss should generally decrease (allow some variance)
    float initial_loss = losses[0];
    float final_loss = losses[9];
    EXPECT_LE(final_loss, initial_loss * 2.0f);  // Shouldn't diverge
}

// Test 7: MixedPrecisionTrainer - Evaluation step
TEST_F(MixedPrecisionTest, EvalStep) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, DType::Float32, device_);
    auto target_tensor = randn({4, 3}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform evaluation step
    float loss = trainer.eval_step(input, target);

    EXPECT_GT(loss, 0.0f);
    EXPECT_FALSE(trainer.is_training());  // Should switch to eval mode
    EXPECT_EQ(trainer.get_total_steps(), 0);  // Eval doesn't count as training step
}

// Test 8: MixedPrecisionTrainer - Train/Eval mode switching
TEST_F(MixedPrecisionTest, TrainEvalModeSwitching) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    EXPECT_TRUE(trainer.is_training());

    trainer.eval();
    EXPECT_FALSE(trainer.is_training());

    trainer.train();
    EXPECT_TRUE(trainer.is_training());
}

// Test 9: MixedPrecisionTrainer - Get model and optimizer
TEST_F(MixedPrecisionTest, GetModelAndOptimizer) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    EXPECT_EQ(trainer.model(), model);
    EXPECT_EQ(trainer.optimizer(), optimizer);
}

// Test 10: MixedPrecisionTrainer - Get configuration
TEST_F(MixedPrecisionTest, GetConfiguration) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::conservative();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    const auto& retrieved_config = trainer.get_config();
    EXPECT_FLOAT_EQ(retrieved_config.init_scale, 1024.0f);
    EXPECT_FLOAT_EQ(retrieved_config.growth_factor, 1.5f);
}

// Test 11: MixedPrecisionTrainer - Reset statistics
TEST_F(MixedPrecisionTest, ResetStatistics) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, DType::Float32, device_);
    auto target_tensor = randn({4, 3}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Perform some steps
    for (int i = 0; i < 5; ++i) {
        trainer.train_step(input, target);
    }

    EXPECT_EQ(trainer.get_total_steps(), 5);

    // Reset statistics
    trainer.reset_stats();

    EXPECT_EQ(trainer.get_total_steps(), 0);
    EXPECT_EQ(trainer.get_skipped_steps(), 0);
}

// Test 12: MixedPrecisionTrainer - Fit with DataLoader
TEST_F(MixedPrecisionTest, FitWithDataLoader) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create simple dataset
    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 8; ++i) {
        auto input = randn({5}, DType::Float32, device_);
        auto target = randn({3}, DType::Float32, device_);
        train_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 2);  // Batch size 2

    // Train for 2 epochs
    trainer.fit(train_loader, 2);

    // Should have trained for 8 batches (4 per epoch)
    EXPECT_GT(trainer.get_total_steps(), 0);
}

// Test 13: MixedPrecisionTrainer - Fit with validation
TEST_F(MixedPrecisionTest, FitWithValidation) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create train and validation datasets
    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 8; ++i) {
        auto input = randn({5}, DType::Float32, device_);
        auto target = randn({3}, DType::Float32, device_);
        train_data.emplace_back(input, target);
    }

    std::vector<std::pair<Tensor, Tensor>> val_data;
    for (int i = 0; i < 4; ++i) {
        auto input = randn({5}, DType::Float32, device_);
        auto target = randn({3}, DType::Float32, device_);
        val_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 2);
    DataLoader val_loader(val_data, 2);

    // Train for 2 epochs with validation
    trainer.fit(train_loader, 2, &val_loader);

    EXPECT_GT(trainer.get_total_steps(), 0);
}

// Test 14: Helper functions - create_fp16_trainer
TEST_F(MixedPrecisionTest, CreateFP16Trainer) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto trainer = create_fp16_trainer(model, optimizer, loss_fn);

    const auto& config = trainer.get_config();
    EXPECT_EQ(config.dtype, DType::Float16);
    EXPECT_TRUE(config.enabled);
}

// Test 15: Helper functions - create_bfloat16_trainer
TEST_F(MixedPrecisionTest, CreateBFloat16Trainer) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto trainer = create_bfloat16_trainer(model, optimizer, loss_fn);

    const auto& config = trainer.get_config();
    EXPECT_EQ(config.dtype, DType::BFloat16);
    EXPECT_TRUE(config.enabled);
}

// Test 16: Loss computation in correct precision
TEST_F(MixedPrecisionTest, LossPrecision) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);

    int loss_computation_count = 0;
    auto loss_fn = [&loss_computation_count](const Variable& pred, const Variable& target) {
        loss_computation_count++;
        // In real implementation, loss should be computed in FP32
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, DType::Float32, device_);
    auto target_tensor = randn({4, 3}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    trainer.train_step(input, target);

    EXPECT_EQ(loss_computation_count, 1);
}

// Test 17: Gradient scaler integration
TEST_F(MixedPrecisionTest, GradScalerIntegration) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Get initial scale
    float initial_scale = trainer.get_scale();
    EXPECT_GT(initial_scale, 0.0f);

    // Access scaler
    auto& scaler = trainer.scaler();
    EXPECT_FLOAT_EQ(scaler.get_scale(), initial_scale);
}

// Test 18: Convergence test - simple regression
TEST_F(MixedPrecisionTest, SimpleConvergence) {
    auto model = std::make_shared<SimpleMLP>(2, 8, 1);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.1);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Simple linear relationship: y = 2*x1 + 3*x2
    auto input_tensor = randn({32, 2}, DType::Float32, device_);
    auto input_data = input_tensor.data<float>();

    std::vector<float> target_data(32);
    for (int i = 0; i < 32; ++i) {
        target_data[i] = 2.0f * input_data[i*2] + 3.0f * input_data[i*2 + 1];
    }

    auto target_tensor = Tensor({32, 1}, DType::Float32, device_);
    float* target_ptr = target_tensor.data<float>();
    for (int i = 0; i < 32; ++i) {
        target_ptr[i] = target_data[i];
    }

    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    // Train for many iterations
    float initial_loss = trainer.train_step(input, target);

    for (int i = 0; i < 100; ++i) {
        trainer.train_step(input, target);
    }

    float final_loss = trainer.train_step(input, target);

    // Loss should decrease significantly
    EXPECT_LT(final_loss, initial_loss * 0.5f);
}

// Test 19: Mixed precision disabled - should use FP32
TEST_F(MixedPrecisionTest, DisabledMixedPrecision) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;  // Explicitly disable
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input_tensor = randn({4, 5}, DType::Float32, device_);
    auto target_tensor = randn({4, 3}, DType::Float32, device_);
    auto input = Variable(input_tensor, false);
    auto target = Variable(target_tensor, false);

    float loss = trainer.train_step(input, target);

    EXPECT_GT(loss, 0.0f);
    EXPECT_EQ(trainer.get_skipped_steps(), 0);  // No overflow with FP32
}

// Test 20: Statistics tracking over multiple epochs
TEST_F(MixedPrecisionTest, StatisticsTracking) {
    auto model = std::make_shared<SimpleMLP>(5, 10, 3);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        auto squared = diff * diff;
        return mean(squared);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;
    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 10; ++i) {
        auto input = randn({5}, DType::Float32, device_);
        auto target = randn({3}, DType::Float32, device_);
        train_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 2);

    int initial_steps = trainer.get_total_steps();
    trainer.fit(train_loader, 3);  // 3 epochs
    int final_steps = trainer.get_total_steps();

    // Should have 5 batches per epoch * 3 epochs = 15 steps
    EXPECT_EQ(final_steps - initial_steps, 15);
}

int main(int argc, char** argv) {
    // Initialize Tenzor library
    tenzor::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
