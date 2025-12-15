/**
 * @file mixed_precision_training.cpp
 * @brief Tutorial: Mixed Precision Training with FP16
 *
 * This example demonstrates how to use the MixedPrecisionTrainer class
 * for efficient FP16 training with automatic gradient scaling.
 *
 * Key concepts covered:
 * 1. Creating a MixedPrecisionTrainer
 * 2. Configuring FP16 vs BFloat16
 * 3. Training with automatic loss scaling
 * 4. Handling gradient overflow
 * 5. Performance monitoring
 */

#include <iostream>
#include <iomanip>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/mixed_precision.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/training.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

using namespace tenzor;
using namespace tenzor::nn;

// Simple MLP model for demonstration
class MLP : public Module {
public:
    MLP(int input_size, int hidden_size, int output_size)
        : fc1_(std::make_shared<Linear>(input_size, hidden_size)),
          relu1_(std::make_shared<ReLU>()),
          fc2_(std::make_shared<Linear>(hidden_size, hidden_size)),
          relu2_(std::make_shared<ReLU>()),
          fc3_(std::make_shared<Linear>(hidden_size, output_size)) {

        register_module("fc1", fc1_);
        register_module("relu1", relu1_);
        register_module("fc2", fc2_);
        register_module("relu2", relu2_);
        register_module("fc3", fc3_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h1 = relu1_->forward(fc1_->forward(x));
        auto h2 = relu2_->forward(fc2_->forward(h1));
        return fc3_->forward(h2);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<ReLU> relu2_;
    std::shared_ptr<Linear> fc3_;
};

// Example 1: Basic mixed precision training
void example_basic_mixed_precision() {
    std::cout << "\n=== Example 1: Basic Mixed Precision Training ===" << std::endl;

    // Create model, optimizer, and loss function
    auto model = std::make_shared<MLP>(784, 256, 10);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);

    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    // Create mixed precision trainer with FP16
    // Note: For CPU demonstration, we disable mixed precision
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;  // Disable for CPU demo

    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    std::cout << "Created MixedPrecisionTrainer with:" << std::endl;
    std::cout << "  - Dtype: " << (config.dtype == DType::Float16 ? "Float16" : "BFloat16") << std::endl;
    std::cout << "  - Initial scale: " << config.init_scale << std::endl;
    std::cout << "  - Growth factor: " << config.growth_factor << std::endl;
    std::cout << "  - Enabled: " << (config.enabled ? "true" : "false") << std::endl;

    // Create dummy training data
    auto input = randn({32, 784}, DType::Float32, Device::cpu());
    auto target = randn({32, 10}, DType::Float32, Device::cpu());

    // Train for a few steps
    std::cout << "\nTraining for 10 steps:" << std::endl;
    for (int step = 0; step < 10; ++step) {
        float loss = trainer.train_step(Variable(input, false), Variable(target, false));

        if (step % 2 == 0) {
            std::cout << "  Step " << std::setw(2) << step
                      << " - Loss: " << std::fixed << std::setprecision(6) << loss
                      << " - Scale: " << trainer.get_scale() << std::endl;
        }
    }

    std::cout << "\nFinal statistics:" << std::endl;
    std::cout << "  - Total steps: " << trainer.get_total_steps() << std::endl;
    std::cout << "  - Skipped steps: " << trainer.get_skipped_steps() << std::endl;
}

// Example 2: Comparing FP32 vs FP16 training
void example_fp32_vs_fp16_comparison() {
    std::cout << "\n=== Example 2: FP32 vs FP16 Training Comparison ===" << std::endl;

    // Shared model architecture and data
    auto create_model = []() { return std::make_shared<MLP>(100, 128, 10); };
    auto input = randn({16, 100}, DType::Float32, Device::cpu());
    auto target = randn({16, 10}, DType::Float32, Device::cpu());

    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    // FP32 Training
    std::cout << "\nFP32 Training:" << std::endl;
    auto model_fp32 = create_model();
    auto optimizer_fp32 = std::make_shared<optim::Adam>(model_fp32->parameters(), 0.001);

    auto config_fp32 = MixedPrecisionConfig::fp16_cuda();
    config_fp32.enabled = false;  // Disable mixed precision

    MixedPrecisionTrainer trainer_fp32(model_fp32, optimizer_fp32, loss_fn, config_fp32);

    float initial_loss_fp32 = trainer_fp32.train_step(Variable(input, false), Variable(target, false));
    for (int i = 0; i < 49; ++i) {
        trainer_fp32.train_step(Variable(input, false), Variable(target, false));
    }
    float final_loss_fp32 = trainer_fp32.train_step(Variable(input, false), Variable(target, false));

    std::cout << "  - Initial loss: " << initial_loss_fp32 << std::endl;
    std::cout << "  - Final loss: " << final_loss_fp32 << std::endl;
    std::cout << "  - Reduction: " << (initial_loss_fp32 - final_loss_fp32) / initial_loss_fp32 * 100 << "%" << std::endl;

    // FP16 Training (disabled for CPU, but shows API usage)
    std::cout << "\nFP16 Training (API demonstration):" << std::endl;
    auto model_fp16 = create_model();
    auto optimizer_fp16 = std::make_shared<optim::Adam>(model_fp16->parameters(), 0.001);

    auto config_fp16 = MixedPrecisionConfig::fp16_cuda();
    config_fp16.enabled = false;  // Would be true on CUDA device

    MixedPrecisionTrainer trainer_fp16(model_fp16, optimizer_fp16, loss_fn, config_fp16);

    std::cout << "  - Config dtype: Float16" << std::endl;
    std::cout << "  - Initial scale: " << trainer_fp16.get_scale() << std::endl;
    std::cout << "  - Note: FP16 provides 2-3x speedup on CUDA" << std::endl;
}

// Example 3: Monitoring gradient overflow
void example_gradient_overflow_monitoring() {
    std::cout << "\n=== Example 3: Gradient Overflow Monitoring ===" << std::endl;

    auto model = std::make_shared<MLP>(50, 64, 5);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.01);

    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    // Use aggressive scaling to demonstrate overflow handling
    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;  // Disabled for CPU
    config.init_scale = 1000.0f;
    config.growth_factor = 3.0f;
    config.backoff_factor = 0.25f;

    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input = randn({8, 50}, DType::Float32, Device::cpu());
    auto target = randn({8, 5}, DType::Float32, Device::cpu());

    std::cout << "Training with overflow monitoring:" << std::endl;
    std::cout << "Initial scale: " << trainer.get_scale() << std::endl;

    for (int step = 0; step < 20; ++step) {
        float loss = trainer.train_step(Variable(input, false), Variable(target, false));

        if (step % 5 == 0) {
            std::cout << "  Step " << std::setw(2) << step
                      << " - Loss: " << std::fixed << std::setprecision(4) << loss
                      << " - Scale: " << std::setw(8) << trainer.get_scale()
                      << " - Skipped: " << trainer.get_skipped_steps()
                      << "/" << trainer.get_total_steps() << std::endl;
        }
    }
}

// Example 4: Using DataLoader with mixed precision
void example_dataloader_integration() {
    std::cout << "\n=== Example 4: DataLoader Integration ===" << std::endl;

    auto model = std::make_shared<MLP>(20, 32, 4);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);

    auto loss_fn = [](const Variable& pred, const Variable& target) {
        auto diff = pred - target;
        return mean(diff * diff);
    };

    auto config = MixedPrecisionConfig::fp16_cuda();
    config.enabled = false;  // Disabled for CPU

    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    // Create training dataset
    std::vector<std::pair<Tensor, Tensor>> train_data;
    for (int i = 0; i < 32; ++i) {
        auto input = randn({20}, DType::Float32, Device::cpu());
        auto target = randn({4}, DType::Float32, Device::cpu());
        train_data.emplace_back(input, target);
    }

    DataLoader train_loader(train_data, 8);  // Batch size 8

    std::cout << "Training for 3 epochs with DataLoader:" << std::endl;

    // Training loop
    trainer.fit(train_loader, 3);

    std::cout << "\nTraining completed!" << std::endl;
    std::cout << "  - Total training steps: " << trainer.get_total_steps() << std::endl;
}

// Example 5: Custom configuration for stability
void example_custom_config() {
    std::cout << "\n=== Example 5: Custom Configuration ===" << std::endl;

    auto model = std::make_shared<MLP>(30, 48, 6);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.0005);

    auto loss_fn = [](const Variable& pred, const Variable& target) {
        return mean(abs(pred - target));
    };

    // Conservative configuration for stable training
    auto config = MixedPrecisionConfig::conservative();
    config.enabled = false;  // Disabled for CPU

    std::cout << "Using conservative configuration:" << std::endl;
    std::cout << "  - Initial scale: " << config.init_scale << std::endl;
    std::cout << "  - Growth factor: " << config.growth_factor << std::endl;
    std::cout << "  - Backoff factor: " << config.backoff_factor << std::endl;
    std::cout << "  - Growth interval: " << config.growth_interval << std::endl;

    MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);

    auto input = randn({12, 30}, DType::Float32, Device::cpu());
    auto target = randn({12, 6}, DType::Float32, Device::cpu());

    std::cout << "\nTraining for 15 steps:" << std::endl;
    for (int step = 0; step < 15; ++step) {
        float loss = trainer.train_step(Variable(input, false), Variable(target, false));

        if (step % 5 == 0) {
            std::cout << "  Step " << step << " - Loss: " << loss << std::endl;
        }
    }
}

// Example 6: Performance best practices
void example_best_practices() {
    std::cout << "\n=== Example 6: Best Practices ===" << std::endl;

    std::cout << "\n1. Choose the right precision:" << std::endl;
    std::cout << "   - FP16: Best for Volta/Turing/Ampere GPUs (2-3x speedup)" << std::endl;
    std::cout << "   - BFloat16: Best for Ampere+ GPUs (better stability)" << std::endl;
    std::cout << "   - FP32: Use for CPU or when precision is critical" << std::endl;

    std::cout << "\n2. Monitor gradient overflow:" << std::endl;
    std::cout << "   - Check get_skipped_steps() regularly" << std::endl;
    std::cout << "   - If >10% steps skipped, use conservative config" << std::endl;
    std::cout << "   - Adjust learning rate if frequent overflow" << std::endl;

    std::cout << "\n3. Scale configuration:" << std::endl;
    std::cout << "   - Default (65536) works for most cases" << std::endl;
    std::cout << "   - Use conservative (1024) for unstable models" << std::endl;
    std::cout << "   - Increase growth_interval for large models" << std::endl;

    std::cout << "\n4. Memory savings:" << std::endl;
    std::cout << "   - FP16 saves ~40% activation memory" << std::endl;
    std::cout << "   - Enables larger batch sizes" << std::endl;
    std::cout << "   - Parameters still stored in FP32" << std::endl;

    std::cout << "\n5. Numerical stability:" << std::endl;
    std::cout << "   - Loss always computed in FP32" << std::endl;
    std::cout << "   - Batch norm should stay in FP32" << std::endl;
    std::cout << "   - Use gradient clipping if needed" << std::endl;
}

int main() {
    // Initialize Tenzor library
    tenzor::initialize();

    std::cout << "========================================" << std::endl;
    std::cout << "   Mixed Precision Training Tutorial   " << std::endl;
    std::cout << "========================================" << std::endl;

    // Run examples
    example_basic_mixed_precision();
    example_fp32_vs_fp16_comparison();
    example_gradient_overflow_monitoring();
    example_dataloader_integration();
    example_custom_config();
    example_best_practices();

    std::cout << "\n========================================" << std::endl;
    std::cout << "   Tutorial Complete!                  " << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
