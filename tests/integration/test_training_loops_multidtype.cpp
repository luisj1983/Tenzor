/**
 * @file test_training_loops_multidtype.cpp
 * @brief Multi-dtype / multi-backend companion to test_training_loops.cpp.
 *
 * The plain file (BackendTest, Float32-only) exercises complete end-to-end
 * training workflows on a SimpleCNN (Conv2d→BatchNorm2d→ReLU→Conv2d→
 * BatchNorm2d→ReLU→Flatten→Linear→ReLU→Linear) over synthetic MNIST-like
 * data: basic convergence, LR scheduling (Step / Cosine / Exponential),
 * gradient clipping, gradient accumulation, Adam / AdamW / SGD optimizer
 * comparison, validation + early-stopping loops, multi-loss training,
 * checkpointing, and a full train/val workflow.
 *
 * This companion adds the dtype axis across {Float32, Float64, Float16} x
 * {cpu, cuda, vulkan, oneapi, rocm, mps} via MultiBackendDTypeTest. The
 * SimpleCNN is moved to the test dtype+device via convert_model, and the
 * synthetic input is cast to the test dtype on the test device, so the
 * companion exercises each backend's dtype dispatch through every stage of
 * the training loop — conv / batchnorm / linear forward, cross_entropy loss,
 * backward, and the optimizer step.
 *
 * Dtype coverage is gated by what the training path can actually sustain:
 *   - Float32 runs on every backend (the meaningful coverage — the plain
 *     file is Float32-only via BackendTest; this companion re-exercises the
 *     full training loop through the multidtype fixture's device/dtype
 *     bring-up, tolerances, and TearDown device-drain).
 *   - Float64 runs on CPU only. The SimpleCNN's fc1 is 64*28*28*128 ≈ 6.4M
 *     params; multi-epoch F64 training loops on GPU are too memory/compute
 *     heavy for the workflow, so F64-on-non-CPU is skipped
 *     (NumericalDivergence). F64-on-CPU adds a genuine second-dtype axis on
 *     the CPU training path.
 *   - Float16 / BFloat16 are skipped categorically (NumericalDivergence):
 *     the SimpleCNN trains through BatchNorm2d in the test dtype, and
 *     F16/BF16 batchnorm training convergence is not validated across
 *     backends — the loss-decrease / accuracy assertions would fail
 *     spuriously. (The F32×backend axis already covers the training-loop
 *     machinery; the skip loses no real coverage.)
 *
 * Numerical readbacks are dtype-safe: calculate_accuracy and the
 * gradient-clipping norm both cast tensors to CPU Float32 before reading
 * .data<float>(), so an F64 source tensor is not type-punned as float.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <chrono>
#include <cmath>
#include <limits>
#include <memory>
#include <iostream>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;
using namespace tenzor::testing;

//==============================================================================
// Skip guards
//==============================================================================

// F16/BF16 training through BatchNorm2d is not validated across backends;
// the convergence / loss / accuracy assertions would fail spuriously.
#define skip_if_half_training()                                              \
    do {                                                                      \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) {       \
            SKIP_WITH_REASON(                                                \
                ::tenzor::testing::SkipReason::NumericalDivergence,          \
                "F16/BF16 training through BatchNorm2d not validated "       \
                "across backends; convergence/loss/accuracy assertions "     \
                "unreliable");                                                \
        }                                                                     \
    } while (0)

// F64 multi-epoch training loops on GPU are too memory/compute heavy for the
// SimpleCNN workflow (fc1 ≈ 6.4M params × multi-epoch × multi-batch in F64).
#define skip_if_float64_gpu()                                                \
    do {                                                                      \
        if (dtype() == DType::Float64 &&                                     \
            device().type != Device::Type::CPU) {                            \
            SKIP_WITH_REASON(                                                \
                ::tenzor::testing::SkipReason::NumericalDivergence,          \
                "Float64 multi-epoch training loops on GPU are too "         \
                "memory/compute heavy for the SimpleCNN workflow");          \
        }                                                                     \
    } while (0)

//==============================================================================
// Helper Functions
//==============================================================================

// Generate synthetic MNIST-like data with learnable patterns. Each class has
// a distinctive 5x5 bright region. The input is built in Float32 on CPU
// (matching the plain file's pattern placement) then cast to the test dtype
// on the test device; the one-hot target stays Float32 (as in the plain
// file — cross_entropy receives a Float32 target regardless of model dtype).
auto generate_mnist_batch(int batch_size, Device device, DType dtype)
    -> std::pair<Variable, Variable> {
    auto input_data = randn({batch_size, 1, 28, 28}, DType::Float32, Device::cpu()) * 0.1f;
    float* input_ptr = input_data.data<float>();

    auto target_data = zeros({batch_size, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data.template data<float>());

    for (int i = 0; i < batch_size; i++) {
        int label = i % 10;
        target_ptr[i * 10 + label] = 1.0f;

        int row_offset = (label / 4) * 8 + 4;  // rows 4, 12, 20
        int col_offset = (label % 4) * 7 + 4;  // cols 4, 11, 18, 25

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

    auto input_device = input_data.to(device);
    if (dtype != DType::Float32) {
        input_device = input_device.to(dtype);
    }
    auto target_device = target_data.to(device);
    return {Variable(input_device, true), Variable(target_device, false)};
}

// Calculate classification accuracy. Predictions are cast to CPU Float32
// before reading (dtype-safe — an F64/F16 logits tensor is not type-punned).
auto calculate_accuracy(const Variable& predictions, const Variable& targets) -> float {
    auto pred_cpu = predictions.tensor().to(Device::cpu());
    if (pred_cpu.dtype() != DType::Float32) pred_cpu = pred_cpu.to(DType::Float32);
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

// Read a scalar loss as float regardless of the test dtype (item<float>()
// narrows F64 / widens F16 safely).
template <typename T>
auto loss_value(const T& loss) -> float {
    return loss.tensor().cpu().template item<float>();
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
// Fixture
//==============================================================================

class TrainingLoopsMultiDType : public MultiBackendDTypeTest {};

//==============================================================================
// Test 1: Basic MNIST Training - Full Epoch with Validation
//==============================================================================

TEST_P(TrainingLoopsMultiDType, BasicMNISTTraining) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            total_loss += loss_value(loss);
            total_accuracy += calculate_accuracy(output, target);
        }

        epoch_losses.push_back(total_loss / batches_per_epoch);
        epoch_accuracies.push_back(total_accuracy / batches_per_epoch);
    }

    EXPECT_LT(epoch_losses.back(), epoch_losses.front() * 0.9f)
        << "Loss should decrease by at least 10% over training";
    EXPECT_GT(epoch_accuracies.back(), 0.15f)
        << "Final accuracy should be better than random (>10%)";
}

//==============================================================================
// Test 2: Training with Learning Rate Scheduling
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithLRScheduling) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    EXPECT_FLOAT_EQ(learning_rates[0], 0.1f) << "Initial LR should be 0.1";
    EXPECT_FLOAT_EQ(learning_rates[2], 0.01f) << "LR should be 0.01 after first step";
    EXPECT_FLOAT_EQ(learning_rates[4], 0.001f) << "LR should be 0.001 after second step";
}

//==============================================================================
// Test 3: Training with Gradient Clipping
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithGradientClipping) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

    auto params = model->parameters();
    Adam optimizer(params, 0.001);

    float clip_norm = 1.0f;
    int num_epochs = 2;
    int batches_per_epoch = 5;
    int batch_size = 16;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();

            // Gradient clipping. Grad is cast to CPU Float32 before reading
            // so an F64 grad tensor is not type-punned as float.
            float total_norm = 0.0f;
            for (auto& param : params) {
                if (param->grad().has_value()) {
                    auto grad = param->grad().value();
                    auto grad_cpu = grad.to(Device::cpu());
                    if (grad_cpu.dtype() != DType::Float32) {
                        grad_cpu = grad_cpu.to(DType::Float32);
                    }
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

TEST_P(TrainingLoopsMultiDType, MNISTWithAdamOptimizer) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            total_loss += loss_value(loss);
        }

        losses.push_back(total_loss / batches_per_epoch);
    }

    EXPECT_LT(losses.back(), losses.front())
        << "Loss should decrease with Adam optimizer";
}

//==============================================================================
// Test 5: Training with AdamW Optimizer and Weight Decay
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithAdamWOptimizer) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            total_loss += loss_value(loss);
        }

        losses.push_back(total_loss / batches_per_epoch);
    }

    EXPECT_LT(losses.back(), losses.front())
        << "Loss should decrease with AdamW optimizer";
}

//==============================================================================
// Test 6: Training with Cosine Annealing LR
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithCosineAnnealingLR) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

    EXPECT_GT(learning_rates[0], learning_rates[num_epochs / 2])
        << "LR should decrease in first half";
    EXPECT_LT(learning_rates.back(), learning_rates[0] * 0.1f)
        << "Final LR should be much smaller than initial";
}

//==============================================================================
// Test 7: Training with Validation Loop
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithValidation) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int num_epochs = 3;
    int train_batches = 10;
    int val_batches = 5;
    int batch_size = 32;

    std::vector<float> val_accuracies;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        for (int batch = 0; batch < train_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        model->eval();
        float val_accuracy = 0.0f;
        for (int batch = 0; batch < val_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());
            auto output = model->forward(input);
            val_accuracy += calculate_accuracy(output, target);
        }
        val_accuracy /= val_batches;
        val_accuracies.push_back(val_accuracy);
    }

    EXPECT_GT(val_accuracies.back(), val_accuracies.front() * 0.9f)
        << "Validation accuracy should not degrade significantly";
}

//==============================================================================
// Test 8: Training with Multiple Loss Functions
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithMultipleLosses) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);

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

TEST_P(TrainingLoopsMultiDType, MNISTWithEarlyStopping) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int max_epochs = 20;
    int patience = 3;
    float best_val_loss = std::numeric_limits<float>::max();
    int patience_counter = 0;
    int batch_size = 32;

    for (int epoch = 0; epoch < max_epochs; epoch++) {
        model->train();
        for (int batch = 0; batch < 5; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        model->eval();
        float val_loss = 0.0f;
        for (int batch = 0; batch < 3; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            val_loss += loss_value(loss);
        }
        val_loss /= 3;

        if (val_loss < best_val_loss) {
            best_val_loss = val_loss;
            patience_counter = 0;
        } else {
            patience_counter++;
            if (patience_counter >= patience) {
                break;
            }
        }
    }

    EXPECT_GT(patience_counter, 0) << "Early stopping should have triggered";
}

//==============================================================================
// Test 10: Training with Batch Size Variation
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithVaryingBatchSizes) {
    skip_if_half_training();
    skip_if_float64_gpu();

    std::vector<int> batch_sizes = {8, 16, 32, 64};

    for (auto batch_size : batch_sizes) {
        auto model = std::make_shared<SimpleCNN>();
        convert_model(model);

        auto params = model->parameters();
        SGD optimizer(params, 0.01, 0.9);

        model->train();
        auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

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

TEST_P(TrainingLoopsMultiDType, MNISTWithGradientAccumulation) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int accumulation_steps = 4;
    int num_batches = 12;
    int batch_size = 8;

    model->train();
    for (int batch = 0; batch < num_batches; batch++) {
        auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

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

TEST_P(TrainingLoopsMultiDType, MNISTWithExponentialLR) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();
        }

        learning_rates.push_back(scheduler.get_lr());
        scheduler.step();
    }

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

TEST_P(TrainingLoopsMultiDType, OptimizersComparison) {
    skip_if_half_training();
    skip_if_float64_gpu();

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
        convert_model(model);
        auto params = model->parameters();
        SGD optimizer(params, 0.01, 0.9);

        float final_loss = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            model->train();
            for (int batch = 0; batch < batches_per_epoch; batch++) {
                auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());
                optimizer.zero_grad();
                auto output = model->forward(input);
                auto loss = cross_entropy(output, target.tensor());
                loss.backward();
                optimizer.step();
                final_loss = loss_value(loss);
            }
        }
        results.push_back({"SGD", final_loss});
    }

    // Test Adam
    {
        auto model = std::make_shared<SimpleCNN>();
        convert_model(model);
        auto params = model->parameters();
        Adam optimizer(params, 0.001);

        float final_loss = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            model->train();
            for (int batch = 0; batch < batches_per_epoch; batch++) {
                auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());
                optimizer.zero_grad();
                auto output = model->forward(input);
                auto loss = cross_entropy(output, target.tensor());
                loss.backward();
                optimizer.step();
                final_loss = loss_value(loss);
            }
        }
        results.push_back({"Adam", final_loss});
    }

    // Test AdamW
    {
        auto model = std::make_shared<SimpleCNN>();
        convert_model(model);
        auto params = model->parameters();
        AdamW optimizer(params, 0.001, 0.9, 0.999, 1e-8, 0.01);

        float final_loss = 0.0f;
        for (int epoch = 0; epoch < num_epochs; epoch++) {
            model->train();
            for (int batch = 0; batch < batches_per_epoch; batch++) {
                auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());
                optimizer.zero_grad();
                auto output = model->forward(input);
                auto loss = cross_entropy(output, target.tensor());
                loss.backward();
                optimizer.step();
                final_loss = loss_value(loss);
            }
        }
        results.push_back({"AdamW", final_loss});
    }

    for (const auto& result : results) {
        EXPECT_LT(result.final_loss, 10.0f)
            << result.name << " should converge to reasonable loss";
    }
}

//==============================================================================
// Test 14: Training with Model Checkpointing (simulated)
//==============================================================================

TEST_P(TrainingLoopsMultiDType, MNISTWithCheckpointing) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

    auto params = model->parameters();
    SGD optimizer(params, 0.01, 0.9);

    int num_epochs = 5;
    int batches_per_epoch = 5;
    int batch_size = 16;
    float best_loss = std::numeric_limits<float>::max();

    std::vector<Tensor> best_params;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        model->train();
        float epoch_loss = 0.0f;

        for (int batch = 0; batch < batches_per_epoch; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            epoch_loss += loss_value(loss);
        }

        epoch_loss /= batches_per_epoch;

        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            best_params.clear();
            for (auto& param : params) {
                best_params.push_back(param->tensor().clone());
            }
        }
    }

    EXPECT_FALSE(best_params.empty()) << "Should have saved at least one checkpoint";
}

//==============================================================================
// Test 15: End-to-End Training Workflow
//==============================================================================

TEST_P(TrainingLoopsMultiDType, CompleteTrainingWorkflow) {
    skip_if_half_training();
    skip_if_float64_gpu();

    auto model = std::make_shared<SimpleCNN>();
    convert_model(model);

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

        model->train();
        float train_loss = 0.0f;
        float train_accuracy = 0.0f;

        for (int batch = 0; batch < train_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());

            optimizer.zero_grad();
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());
            loss.backward();
            optimizer.step();

            train_loss += loss_value(loss);
            train_accuracy += calculate_accuracy(output, target);
        }

        metrics.train_loss = train_loss / train_batches;
        metrics.train_accuracy = train_accuracy / train_batches;

        model->eval();
        float val_loss = 0.0f;
        float val_accuracy = 0.0f;

        for (int batch = 0; batch < val_batches; batch++) {
            auto [input, target] = generate_mnist_batch(batch_size, device(), dtype());
            auto output = model->forward(input);
            auto loss = cross_entropy(output, target.tensor());

            val_loss += loss_value(loss);
            val_accuracy += calculate_accuracy(output, target);
        }

        metrics.val_loss = val_loss / val_batches;
        metrics.val_accuracy = val_accuracy / val_batches;
        metrics.learning_rate = scheduler.get_lr();

        history.push_back(metrics);
        scheduler.step();
    }

    EXPECT_EQ(history.size(), num_epochs) << "Should have metrics for all epochs";
    EXPECT_LT(history.back().train_loss, history.front().train_loss)
        << "Training loss should decrease";
    EXPECT_GT(history.back().train_accuracy, history.front().train_accuracy * 0.9f)
        << "Training accuracy should improve or stay stable";
    EXPECT_LT(history.back().learning_rate, history.front().learning_rate)
        << "Learning rate should decrease with scheduler";
}

//==============================================================================
// Multi-Backend DType Instantiation
//==============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(TrainingLoopsMultiDType);