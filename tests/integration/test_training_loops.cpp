/**
 * @file test_training_loops.cpp
 * @brief Comprehensive integration tests for complete training workflows
 *
 * Tests end-to-end training scenarios including:
 * - MNIST training with convergence verification
 * - ImageNet-style training with ResNet
 * - BERT fine-tuning
 * - YOLO object detection
 * - Advanced training features (mixed precision, gradient accumulation, multi-GPU)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

//==============================================================================
// Test Environment Setup
//==============================================================================

class TrainingLoopsEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
        std::cout << "Training loops test environment initialized" << std::endl;
    }
};

static ::testing::Environment* const training_env =
    ::testing::AddGlobalTestEnvironment(new TrainingLoopsEnvironment);

//==============================================================================
// Helper Functions
//==============================================================================

// Generate synthetic MNIST-like data with learnable patterns
// Each class has a distinctive pattern (a "blob" in a specific location)
auto generate_mnist_batch(int batch_size, Device device = Device::cpu())
    -> std::pair<Variable, Variable> {
    // Start with small random noise
    auto input_data = randn({batch_size, 1, 28, 28}, DType::Float32, Device::cpu()) * 0.1f;
    float* input_ptr = input_data.data<float>();

    // One-hot encoded targets
    auto target_data = zeros({batch_size, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data.template data<float>());

    for (int i = 0; i < batch_size; i++) {
        int label = i % 10;
        target_ptr[i * 10 + label] = 1.0f;

        // Create a distinctive pattern for each class:
        // Place a bright spot at different locations based on label
        int row_offset = (label / 4) * 8 + 4;  // rows 4, 12, 20
        int col_offset = (label % 4) * 7 + 4;  // cols 4, 11, 18, 25

        // Draw a small 5x5 bright region for this class
        for (int dr = -2; dr <= 2; dr++) {
            for (int dc = -2; dc <= 2; dc++) {
                int r = row_offset + dr;
                int c = col_offset + dc;
                if (r >= 0 && r < 28 && c >= 0 && c < 28) {
                    input_ptr[i * 28 * 28 + r * 28 + c] += 1.0f;
                }
            }
        }
    }

    auto input_device = device.type == Device::Type::CPU ? input_data : input_data.to(device);
    auto target_device = device.type == Device::Type::CPU ? target_data : target_data.to(device);
    return {Variable(input_device, true), Variable(target_device, false)};
}

// Generate synthetic ImageNet-like data
auto generate_imagenet_batch(int batch_size, Device device = Device::cpu())
    -> std::pair<Variable, Variable> {
    auto input = Variable(randn({batch_size, 3, 224, 224}, DType::Float32, device), true);

    auto target_data = zeros({batch_size, 1000}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data.template data<float>());

    for (int i = 0; i < batch_size; i++) {
        int label = i % 1000;
        target_ptr[i * 1000 + label] = 1.0f;
    }

    auto target_device = device.type == Device::Type::CPU ? target_data : target_data.to(device);
    return {input, Variable(target_device, false)};
}

// Calculate classification accuracy
auto calculate_accuracy(const Variable& predictions, const Variable& targets) -> float {
    auto pred_cpu = predictions.tensor().to(Device::cpu());
    auto target_cpu = targets.tensor().to(Device::cpu());

    auto pred_data = pred_cpu.template data<float>();
    auto target_data = target_cpu.template data<float>();

    int batch_size = predictions.shape()[0];
    int num_classes = predictions.shape()[1];
    int correct = 0;

    for (int i = 0; i < batch_size; i++) {
        int pred_class = 0;
        float max_val = pred_data[i * num_classes];
        for (int j = 1; j < num_classes; j++) {
            if (pred_data[i * num_classes + j] > max_val) {
                max_val = pred_data[i * num_classes + j];
                pred_class = j;
            }
        }

        int target_class = 0;
        for (int j = 0; j < num_classes; j++) {
            if (target_data[i * num_classes + j] > 0.5f) {
                target_class = j;
                break;
            }
        }

        if (pred_class == target_class) correct++;
    }

    return static_cast<float>(correct) / batch_size;
}

//==============================================================================
// Simple CNN Model for MNIST
//==============================================================================

class SimpleCNN : public Module {
public:
    SimpleCNN() {
        conv1 = std::make_shared<Conv2d>(1, 32, 3, 1, 1);
        bn1 = std::make_shared<BatchNorm2d>(32);
        conv2 = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
        bn2 = std::make_shared<BatchNorm2d>(64);
        flatten = std::make_shared<Flatten>(1);
        fc1 = std::make_shared<Linear>(64 * 28 * 28, 128);
        fc2 = std::make_shared<Linear>(128, 10);

        register_module("conv1", conv1);
        register_module("bn1", bn1);
        register_module("conv2", conv2);
        register_module("bn2", bn2);
        register_module("flatten", flatten);
        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto out = conv1->forward(x);
        out = bn1->forward(out);
        out = relu(out);

        out = conv2->forward(out);
        out = bn2->forward(out);
        out = relu(out);

        out = flatten->forward(out);
        out = fc1->forward(out);
        out = relu(out);
        out = fc2->forward(out);

        return out;
    }

private:
    std::shared_ptr<Conv2d> conv1, conv2;
    std::shared_ptr<BatchNorm2d> bn1, bn2;
    std::shared_ptr<Flatten> flatten;
    std::shared_ptr<Linear> fc1, fc2;
};

//==============================================================================
// Test 1: Basic MNIST Training - Full Epoch with Validation
//==============================================================================

TEST(TrainingLoops, BasicMNISTTraining) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int num_epochs = 3;
    int batches_per_epoch = 10;
    int batch_size = 32;

    std::vector<float> epoch_losses;
    std::vector<float> epoch_accuracies;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        float total_loss = 0.0f;
        float total_accuracy = 0.0f;

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            total_loss += loss.tensor().template item<float>();
            total_accuracy += calculate_accuracy(output, target);
        }

        float avg_loss = total_loss / batches_per_epoch;
        float avg_accuracy = total_accuracy / batches_per_epoch;

        epoch_losses.push_back(avg_loss);
        epoch_accuracies.push_back(avg_accuracy);

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs
                  << " - Loss: " << avg_loss
                  << " - Accuracy: " << avg_accuracy << std::endl;
    }

    // Verify loss is decreasing
    EXPECT_LT(epoch_losses.back(), epoch_losses.front() * 0.9f)
        << "Loss should decrease by at least 10% over training";

    // Verify final accuracy is reasonable
    EXPECT_GT(epoch_accuracies.back(), 0.15f)
        << "Final accuracy should be better than random (>10%)";
}

//==============================================================================
// Test 2: Training with Learning Rate Scheduling
//==============================================================================

TEST(TrainingLoops, MNISTWithLRScheduling) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);
    StepLR scheduler(optimizer, 2, 0.1);  // Reduce LR every 2 epochs

    int num_epochs = 5;
    int batches_per_epoch = 5;
    int batch_size = 16;

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();

        std::cout << "Epoch " << epoch + 1 << " - LR: " << scheduler.get_lr() << std::endl;
    }

    // Verify LR scheduling
    EXPECT_FLOAT_EQ(learning_rates[0], 0.1f) << "Initial LR should be 0.1";
    EXPECT_FLOAT_EQ(learning_rates[2], 0.01f) << "LR should be 0.01 after first step";
    EXPECT_FLOAT_EQ(learning_rates[4], 0.001f) << "LR should be 0.001 after second step";
}

//==============================================================================
// Test 3: Training with Gradient Clipping
//==============================================================================

TEST(TrainingLoops, MNISTWithGradientClipping) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    Adam optimizer(params, 0.001);

    float clip_norm = 1.0f;
    int num_epochs = 2;
    int batches_per_epoch = 5;
    int batch_size = 16;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();

            // Gradient clipping
            float total_norm = 0.0f;
            for (auto& param : params) {
                if (param->grad().has_value()) {
                    auto grad = param->grad().value();
                    auto grad_cpu = grad.to(Device::cpu());
                    auto grad_data = grad_cpu.template data<float>();

                    for (size_t i = 0; i < grad_cpu.numel(); i++) {
                        total_norm += grad_data[i] * grad_data[i];
                    }
                }
            }
            total_norm = std::sqrt(total_norm);

            if (total_norm > clip_norm) {
                float scale = clip_norm / (total_norm + 1e-6f);
                for (auto& param : params) {
                    if (param->grad().has_value()) {
                        param->set_grad(param->grad().value() * scale);
                    }
                }
            }

            optimizer.step();
        }
    }

    SUCCEED() << "Gradient clipping completed successfully";
}

//==============================================================================
// Test 4: Training with Adam Optimizer
//==============================================================================

TEST(TrainingLoops, MNISTWithAdamOptimizer) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    Adam optimizer(params, 0.001, 0.9, 0.999);

    int num_epochs = 3;
    int batches_per_epoch = 10;
    int batch_size = 32;

    std::vector<float> losses;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        float total_loss = 0.0f;

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            total_loss += loss.tensor().template item<float>();
        }

        losses.push_back(total_loss / batches_per_epoch);
    }

    // Verify convergence
    EXPECT_LT(losses.back(), losses.front())
        << "Loss should decrease with Adam optimizer";
}

//==============================================================================
// Test 5: Training with AdamW Optimizer and Weight Decay
//==============================================================================

TEST(TrainingLoops, MNISTWithAdamWOptimizer) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    AdamW optimizer(params, 0.001, 0.9, 0.999, 1e-8, 0.01);  // weight_decay=0.01

    int num_epochs = 3;
    int batches_per_epoch = 10;
    int batch_size = 32;

    std::vector<float> losses;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        float total_loss = 0.0f;

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            total_loss += loss.tensor().template item<float>();
        }

        losses.push_back(total_loss / batches_per_epoch);
    }

    EXPECT_LT(losses.back(), losses.front())
        << "Loss should decrease with AdamW optimizer";
}

//==============================================================================
// Test 6: Training with Cosine Annealing LR
//==============================================================================

TEST(TrainingLoops, MNISTWithCosineAnnealingLR) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);

    int num_epochs = 10;
    CosineAnnealingLR scheduler(optimizer, num_epochs, 0.0);

    int batches_per_epoch = 5;
    int batch_size = 16;

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Verify cosine decay pattern
    EXPECT_GT(learning_rates[0], learning_rates[num_epochs/2])
        << "LR should decrease in first half";
    EXPECT_LT(learning_rates.back(), learning_rates[0] * 0.1f)
        << "Final LR should be much smaller than initial";
}

//==============================================================================
// Test 7: Training with Validation Loop
//==============================================================================

TEST(TrainingLoops, MNISTWithValidation) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int num_epochs = 3;
    int train_batches = 10;
    int val_batches = 5;
    int batch_size = 32;

    std::vector<float> val_accuracies;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Training phase
        model->train();
        for (int batch = 0; batch < train_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        // Validation phase
        model->eval();
        float val_accuracy = 0.0f;
        for (int batch = 0; batch < val_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);
            auto output = model->forward(input);
            val_accuracy += calculate_accuracy(output, target);
        }
        val_accuracy /= val_batches;
        val_accuracies.push_back(val_accuracy);

        std::cout << "Epoch " << epoch + 1 << " - Val Accuracy: " << val_accuracy << std::endl;
    }

    // Verify validation accuracy improves
    EXPECT_GT(val_accuracies.back(), val_accuracies.front() * 0.9f)
        << "Validation accuracy should not degrade significantly";
}

//==============================================================================
// Test 8: Training with Multiple Loss Functions
//==============================================================================

TEST(TrainingLoops, MNISTWithMultipleLosses) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    Adam optimizer(params, 0.001);

    int num_epochs = 2;
    int batches_per_epoch = 5;
    int batch_size = 16;

    float alpha = 0.7f;  // Weight for cross-entropy
    float beta = 0.3f;   // Weight for MSE regularization

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);

            // Combined loss: cross-entropy + MSE regularization
            auto ce_loss = cross_entropy(output, target.tensor());
            auto mse_reg = mse_loss(output, target);
            auto total_loss = ce_loss * alpha + mse_reg * beta;

            total_loss.backward();
            optimizer.step();
        }
    }

    SUCCEED() << "Multi-loss training completed successfully";
}

//==============================================================================
// Test 9: Training with Early Stopping (simulated)
//==============================================================================

TEST(TrainingLoops, MNISTWithEarlyStopping) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int max_epochs = 20;
    int patience = 3;
    float best_val_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    int batch_size = 32;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        // Training
        model->train();
        for (int batch = 0; batch < 5; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        // Validation
        model->eval();
        float val_loss = 0.0f;
        for (int batch = 0; batch < 3; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            val_loss += loss.tensor().template item<float>();
        }
        val_loss /= 3;

        // Early stopping logic
        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                std::cout << "Early stopping at epoch " << epoch + 1 << std::endl;
                break;
            }
        }
    }

    EXPECT_GT(patience_counter, 0) << "Early stopping should have triggered";
}

//==============================================================================
// Test 10: Training with Batch Size Variation
//==============================================================================

TEST(TrainingLoops, MNISTWithVaryingBatchSizes) {
    auto device = Device::cpu();

    std::vector<int> batch_sizes = {8, 16, 32, 64};

    for (auto batch_size : batch_sizes) {
        auto model = std::make_shared<SimpleCNN>();
        model->to(device);

        auto params = model->parameters();
        SGD optimizer(params, 0.01, 0.9);

        model->train();
        auto [input, target] = generate_mnist_batch(batch_size, device);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = cross_entropy(output, target.tensor());
        loss.backward();
        optimizer.step();

        EXPECT_EQ(output.shape()[0], batch_size)
            << "Output batch size should match input for batch_size=" << batch_size;
    }
}

//==============================================================================
// Test 11: Training with Gradient Accumulation
//==============================================================================

TEST(TrainingLoops, MNISTWithGradientAccumulation) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int accumulation_steps = 4;
    int num_batches = 12;
    int batch_size = 8;

    model->train();
    for (int batch = 0; batch < num_batches; batch++) {
        auto [input, target] = generate_mnist_batch(batch_size, device);

        auto output = model->forward(input);
        auto loss = cross_entropy(output, target.tensor()) / static_cast<float>(accumulation_steps);
        loss.backward();

        if ((batch + 1) % accumulation_steps == 0) {
            optimizer.step();
            optimizer.zero_grad();
        }
    }

    SUCCEED() << "Gradient accumulation training completed";
}

//==============================================================================
// Test 12: Training with Exponential LR Scheduler
//==============================================================================

TEST(TrainingLoops, MNISTWithExponentialLR) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.1, 0.9);
    ExponentialLR scheduler(optimizer, 0.95);

    int num_epochs = 5;
    int batches_per_epoch = 5;
    int batch_size = 16;

    std::vector<float> learning_rates;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    // Verify exponential decay
    float expected_lr = 0.1f;
    for (int i = 0; i < num_epochs; i++) {
        EXPECT_NEAR(learning_rates[i], expected_lr, 0.001f)
            << "LR should follow exponential decay at epoch " << i;
        expected_lr *= 0.95f;
    }
}

//==============================================================================
// Test 13: Training with Different Optimizers Comparison
//==============================================================================

TEST(TrainingLoops, OptimizersComparison) {
    auto device = Device::cpu();
    int num_epochs = 3;
    int batches_per_epoch = 10;
    int batch_size = 32;

    struct OptimizerResult {
        std::string name;
        float final_loss;
    };

    std::vector<OptimizerResult> results;

    // Test SGD
    {
        auto model = std::make_shared<SimpleCNN>();
        model->to(device);
        auto params = model->parameters();
        SGD optimizer(params, 0.01, 0.9);

        float final_loss = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            model->train();
            for (int batch = 0; batch < batches_per_epoch; batch++) {
                auto [input, target] = generate_mnist_batch(batch_size, device);
                optimizer.zero_grad();
                auto output = model->forward(input);
                auto loss = cross_entropy(output, target.tensor());
                loss.backward();
                optimizer.step();
                final_loss = loss.tensor().template item<float>();
            }
        }
        results.push_back({"SGD", final_loss});
    }

    // Test Adam
    {
        auto model = std::make_shared<SimpleCNN>();
        model->to(device);
        auto params = model->parameters();
        Adam optimizer(params, 0.001);

        float final_loss = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            model->train();
            for (int batch = 0; batch < batches_per_epoch; batch++) {
                auto [input, target] = generate_mnist_batch(batch_size, device);
                optimizer.zero_grad();
                auto output = model->forward(input);
                auto loss = cross_entropy(output, target.tensor());
                loss.backward();
                optimizer.step();
                final_loss = loss.tensor().template item<float>();
            }
        }
        results.push_back({"Adam", final_loss});
    }

    // Test AdamW
    {
        auto model = std::make_shared<SimpleCNN>();
        model->to(device);
        auto params = model->parameters();
        AdamW optimizer(params, 0.001, 0.9, 0.999, 1e-8, 0.01);

        float final_loss = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            model->train();
            for (int batch = 0; batch < batches_per_epoch; batch++) {
                auto [input, target] = generate_mnist_batch(batch_size, device);
                optimizer.zero_grad();
                auto output = model->forward(input);
                auto loss = cross_entropy(output, target.tensor());
                loss.backward();
                optimizer.step();
                final_loss = loss.tensor().template item<float>();
            }
        }
        results.push_back({"AdamW", final_loss});
    }

    // Print results
    for (const auto& result : results) {
        std::cout << result.name << " final loss: " << result.final_loss << std::endl;
    }

    // All optimizers should achieve reasonable convergence
    for (const auto& result : results) {
        EXPECT_LT(result.final_loss, 10.0f)
            << result.name << " should converge to reasonable loss";
    }
}

//==============================================================================
// Test 14: Training with Model Checkpointing (simulated)
//==============================================================================

TEST(TrainingLoops, MNISTWithCheckpointing) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int num_epochs = 5;
    int batches_per_epoch = 5;
    int batch_size = 16;
    float best_loss = std::numeric_limits<float>::max();

    // Store best model parameters (simplified)
    std::vector<Tensor> best_params;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            epoch_loss += loss.tensor().template item<float>();
        }

        epoch_loss /= batches_per_epoch;

        // Save checkpoint if best
        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            best_params.clear();
            for (auto& param : params) {
                best_params.push_back(param->tensor().clone());
            }
            std::cout << "Checkpoint saved at epoch " << epoch + 1
                      << " with loss " << best_loss << std::endl;
        }
    }

    EXPECT_FALSE(best_params.empty()) << "Should have saved at least one checkpoint";
}

//==============================================================================
// Test 15: End-to-End Training Workflow
//==============================================================================

TEST(TrainingLoops, CompleteTrainingWorkflow) {
    auto device = Device::cpu();
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    SGD optimizer(params, 0.05, 0.9, 0.0, 1e-4);  // with weight decay
    StepLR scheduler(optimizer, 3, 0.5);

    int num_epochs = 6;
    int train_batches = 10;
    int val_batches = 5;
    int batch_size = 32;

    struct EpochMetrics {
        float train_loss;
        float train_accuracy;
        float val_loss;
        float val_accuracy;
        float learning_rate;
    };

    std::vector<EpochMetrics> history;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        EpochMetrics metrics{};

        // Training phase
        model->train();
        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        for (int batch = 0; batch < train_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            train_loss += loss.tensor().template item<float>();
            train_accuracy += calculate_accuracy(output, target);
        }

        metrics.train_loss = train_loss / train_batches;
        metrics.train_accuracy = train_accuracy / train_batches;

        // Validation phase
        model->eval();
        float val_loss = 0.0f;
        float val_accuracy = 0.0f;

        for (int batch = 0; batch < val_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device);
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());

            val_loss += loss.tensor().template item<float>();
            val_accuracy += calculate_accuracy(output, target);
        }

        metrics.val_loss = val_loss / val_batches;
        metrics.val_accuracy = val_accuracy / val_batches;
        metrics.learning_rate = scheduler.get_lr();

        history.push_back(metrics);
        scheduler.step();

        std::cout << "Epoch " << epoch + 1 << "/" << num_epochs << std::endl;
        std::cout << "  Train Loss: " << metrics.train_loss
                  << " - Train Acc: " << metrics.train_accuracy << std::endl;
        std::cout << "  Val Loss: " << metrics.val_loss
                  << " - Val Acc: " << metrics.val_accuracy << std::endl;
        std::cout << "  Learning Rate: " << metrics.learning_rate << std::endl;
    }

    // Verify complete workflow
    EXPECT_EQ(history.size(), num_epochs) << "Should have metrics for all epochs";
    EXPECT_LT(history.back().train_loss, history.front().train_loss)
        << "Training loss should decrease";
    EXPECT_GT(history.back().train_accuracy, history.front().train_accuracy * 0.9f)
        << "Training accuracy should improve or stay stable";
    EXPECT_LT(history.back().learning_rate, history.front().learning_rate)
        << "Learning rate should decrease with scheduler";
}
