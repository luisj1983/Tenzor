#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <iostream>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;
using namespace tenzor::testing;

//==============================================================================
// Test Environment Setup
//==============================================================================

class AcceleratorTrainingEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const accel_env =
    ::testing::AddGlobalTestEnvironment(new AcceleratorTrainingEnvironment);

//==============================================================================
// Helper Functions
//==============================================================================

// Generate synthetic MNIST-like data
auto generate_mnist_batch(int batch_size, Device device) -> std::pair<Variable, Variable> {
    // Input: [batch, 1, 28, 28]
    auto input = Variable(randn({batch_size, 1, 28, 28}, DType::Float32, device), true);

    // Target: [batch, 10] (one-hot encoded)
    auto target_data_cpu = zeros({batch_size, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data_cpu.template data<float>());

    for (int i = 0; i < batch_size; i++) {
        int label = i % 10;
        target_ptr[i * 10 + label] = 1.0f;
    }

    auto target_data = (device.type == Device::Type::CPU) ? target_data_cpu : target_data_cpu.to(device);
    auto target = Variable(target_data, false);
    return {input, target};
}

// Simple accuracy calculation
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

        if (pred_class == target_class) {
            correct++;
        }
    }

    return static_cast<float>(correct) / batch_size;
}

//==============================================================================
// Test Models
//==============================================================================

class SimpleCNN : public Module {
public:
    SimpleCNN() {
        conv1 = std::make_shared<Conv2d>(1, 32, 3, 1, 1);
        bn1 = std::make_shared<BatchNorm2d>(32);
        relu1 = std::make_shared<ReLU>();
        dropout1 = std::make_shared<Dropout>(0.25);

        conv2 = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
        bn2 = std::make_shared<BatchNorm2d>(64);
        relu2 = std::make_shared<ReLU>();
        dropout2 = std::make_shared<Dropout>(0.25);

        flatten = std::make_shared<Flatten>(1);
        fc1 = std::make_shared<Linear>(64 * 28 * 28, 128);
        relu3 = std::make_shared<ReLU>();
        dropout3 = std::make_shared<Dropout>(0.5);
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
        out = relu1->forward(out);
        out = dropout1->forward(out);

        out = conv2->forward(out);
        out = bn2->forward(out);
        out = relu2->forward(out);
        out = dropout2->forward(out);

        auto flattened = flatten->forward(out);
        auto fc_out = fc1->forward(flattened);
        fc_out = relu3->forward(fc_out);
        fc_out = dropout3->forward(fc_out);
        fc_out = fc2->forward(fc_out);

        return fc_out;
    }

private:
    std::shared_ptr<Conv2d> conv1, conv2;
    std::shared_ptr<BatchNorm2d> bn1, bn2;
    std::shared_ptr<ReLU> relu1, relu2, relu3;
    std::shared_ptr<Dropout> dropout1, dropout2, dropout3;
    std::shared_ptr<Flatten> flatten;
    std::shared_ptr<Linear> fc1, fc2;
};

class MLP : public Module {
public:
    MLP(int input_size, int hidden_size, int output_size) {
        fc1 = std::make_shared<Linear>(input_size, hidden_size);
        relu1 = std::make_shared<ReLU>();
        dropout = std::make_shared<Dropout>(0.5);
        fc2 = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1);
        register_module("fc2", fc2);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto out = fc1->forward(x);
        out = relu1->forward(out);
        out = dropout->forward(out);
        out = fc2->forward(out);
        return out;
    }

private:
    std::shared_ptr<Linear> fc1, fc2;
    std::shared_ptr<ReLU> relu1;
    std::shared_ptr<Dropout> dropout;
};

//==============================================================================
// Backend Parameterized Training Tests
//==============================================================================

class AcceleratorTrainingTest : public BackendTest {
protected:
    void TearDown() override {
        // Synchronize device after each test
        try {
            device.synchronize();
        } catch (...) {
            // Ignore synchronization errors
        }
    }
};

//==============================================================================
// Test 1: Simple CNN on MNIST-like Data
//==============================================================================

TEST_P(AcceleratorTrainingTest, SimpleCNN_MNIST) {
    auto model = std::make_shared<SimpleCNN>();
    model->to(device);

    auto params = model->parameters();
    auto optimizer = SGD(params, 0.01, 0.9);

    const int num_epochs = 3;
    const int batch_size = 32;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        auto [input, target] = generate_mnist_batch(batch_size, device);

        // Forward pass
        auto output = model->forward(input);
        EXPECT_EQ(output.shape()[0], batch_size);
        EXPECT_EQ(output.shape()[1], 10);

        // Compute loss
        auto loss = cross_entropy(output, target.tensor(), Reduction::Mean);
        EXPECT_EQ(loss.tensor().numel(), 1);

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_val = loss_cpu.template data<float>()[0];
        EXPECT_GT(loss_val, 0.0f);

        // Backward pass
        optimizer.zero_grad();
        loss.backward();

        // Check gradients were computed
        bool has_gradients = true;
        for (const auto& param : params) {
            if (param->requires_grad() && !param->has_grad()) {
                has_gradients = false;
                break;
            }
        }
        EXPECT_TRUE(has_gradients) << "Some parameters don't have gradients at epoch " << epoch;

        // Optimizer step
        optimizer.step();
    }

    SUCCEED();
}

//==============================================================================
// Test 2: Multi-Layer Perceptron Training
//==============================================================================

TEST_P(AcceleratorTrainingTest, MLP_Training) {
    const int input_size = 784;
    const int hidden_size = 256;
    const int output_size = 10;
    const int batch_size = 64;

    auto model = std::make_shared<MLP>(input_size, hidden_size, output_size);
    model->to(device);
    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    // Generate random input
    auto input = Variable(randn({batch_size, input_size}, DType::Float32, device), true);

    // Create random one-hot targets
    auto target_data_cpu = zeros({batch_size, output_size}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data_cpu.template data<float>());
    for (int i = 0; i < batch_size; i++) {
        int label = i % output_size;
        target_ptr[i * output_size + label] = 1.0f;
    }
    auto target_data = (device.type == Device::Type::CPU) ? target_data_cpu : target_data_cpu.to(device);
    auto target = Variable(target_data, false);

    // Forward pass
    auto output = model->forward(input);
    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], output_size);

    // Loss computation
    auto loss = cross_entropy(output, target.tensor(), Reduction::Mean);
    auto loss_cpu = loss.tensor().to(Device::cpu());
    float loss_value = loss_cpu.template data<float>()[0];
    EXPECT_GT(loss_value, 0.0f);

    // Backward pass
    optimizer.zero_grad();
    loss.backward();

    // Verify gradients
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->has_grad()) << "Parameter missing gradient";
        }
    }

    // Optimizer step
    optimizer.step();

    SUCCEED();
}

//==============================================================================
// Test 3: Complete Training Loop with Convergence
//==============================================================================

TEST_P(AcceleratorTrainingTest, CompleteTrainingLoop) {
    auto model = std::make_shared<MLP>(100, 50, 10);
    model->to(device);
    model->eval();  // Disable dropout for deterministic results
    auto params = model->parameters();
    auto optimizer = SGD(params, 0.001, 0.0);  // Conservative: low LR, no momentum

    const int num_epochs = 20;
    const int batch_size = 32;
    const int batches_per_epoch = 3;

    float initial_loss = 0.0f;
    float best_loss = std::numeric_limits<float>::max();

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        float epoch_loss = 0.0f;

        for (int batch_idx = 0; batch_idx < batches_per_epoch; batch_idx++) {
            // Create fixed synthetic data
            auto input_data_cpu = zeros({batch_size, 100}, DType::Float32, Device::cpu());
            auto target_data_cpu = zeros({batch_size, 10}, DType::Float32, Device::cpu());

            float* input_ptr = const_cast<float*>(input_data_cpu.template data<float>());
            float* target_ptr = const_cast<float*>(target_data_cpu.template data<float>());

            for (int i = 0; i < batch_size; i++) {
                int class_label = (i + batch_idx) % 10;

                for (int j = 0; j < 10; j++) {
                    input_ptr[i * 100 + (class_label * 10 + j)] = 1.0f;
                }

                target_ptr[i * 10 + class_label] = 1.0f;
            }

            auto input_data = (device.type == Device::Type::CPU) ? input_data_cpu : input_data_cpu.to(device);
            auto target_data = (device.type == Device::Type::CPU) ? target_data_cpu : target_data_cpu.to(device);

            auto input = Variable(input_data, true);
            auto target = Variable(target_data, false);

            // Forward
            auto output = model->forward(input);
            auto loss = mse_loss(output, target, Reduction::Mean);

            auto loss_cpu = loss.tensor().to(Device::cpu());
            float loss_val = loss_cpu.template data<float>()[0];
            epoch_loss += loss_val;

            // Backward
            optimizer.zero_grad();
            loss.backward();

            // Update
            optimizer.step();
        }

        epoch_loss /= batches_per_epoch;

        if (epoch == 0) {
            initial_loss = epoch_loss;
        }

        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
        }
    }

    // Training is working if the best loss is better than initial
    EXPECT_LT(best_loss, initial_loss) << "Model did not improve on " << device.to_string()
                                       << " (best: " << best_loss << ", initial: " << initial_loss << ")";
}

//==============================================================================
// Test 4: Gradient Flow Verification
//==============================================================================

TEST_P(AcceleratorTrainingTest, GradientFlowVerification) {
    auto model = std::make_shared<SimpleCNN>();
    model->train();
    model->to(device);
    auto params = model->parameters();
    auto optimizer = SGD(params, 0.01);

    const int batch_size = 8;
    auto [input, target] = generate_mnist_batch(batch_size, device);

    // Forward pass
    auto output = model->forward(input);
    auto loss = cross_entropy(output, target.tensor(), Reduction::Mean);

    auto loss_cpu = loss.tensor().to(Device::cpu());
    float loss_val = loss_cpu.data<float>()[0];
    EXPECT_GT(loss_val, 0.0f) << "Loss should be positive";

    // Backward pass
    optimizer.zero_grad();
    loss.backward();

    // Verify all parameters have gradients
    int params_with_grad = 0;
    int total_params = 0;
    int params_with_nonzero_grad = 0;

    for (const auto& param : params) {
        if (param->requires_grad()) {
            total_params++;
            if (param->has_grad()) {
                params_with_grad++;

                auto grad_data = param->grad().value();
                bool has_nonzero = false;

                auto grad_cpu = grad_data.to(Device::cpu());
                const float* grad_ptr = grad_cpu.data<float>();

                size_t check_count = std::min(static_cast<size_t>(100), static_cast<size_t>(grad_cpu.numel()));
                for (size_t i = 0; i < check_count; i++) {
                    if (std::abs(grad_ptr[i]) > 1e-8) {
                        has_nonzero = true;
                        break;
                    }
                }

                if (has_nonzero) {
                    params_with_nonzero_grad++;
                }
            }
        }
    }

    EXPECT_EQ(params_with_grad, total_params) << "Not all parameters received gradients on " << device.to_string();
    EXPECT_GT(params_with_nonzero_grad, 0) << "All gradients appear to be zero on " << device.to_string();
}

//==============================================================================
// Test 5: Batch Size Scaling
//==============================================================================

TEST_P(AcceleratorTrainingTest, BatchSizeScaling) {
    auto model = std::make_shared<MLP>(784, 128, 10);
    model->to(device);
    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    std::vector<int> batch_sizes = {16, 32, 64};

    for (int batch_size : batch_sizes) {
        auto input = Variable(randn({batch_size, 784}, DType::Float32, device), true);
        auto target_data = zeros({batch_size, 10}, DType::Float32, device);
        auto target = Variable(target_data, false);

        auto output = model->forward(input);
        EXPECT_EQ(output.shape()[0], batch_size);

        auto loss = mse_loss(output, target, Reduction::Mean);

        optimizer.zero_grad();
        loss.backward();
        optimizer.step();

        EXPECT_EQ(output.shape()[0], batch_size);
    }
}

//==============================================================================
// Test 6: Multi-Epoch Training with Validation
//==============================================================================

TEST_P(AcceleratorTrainingTest, MultiEpochTrainingWithValidation) {
    auto model = std::make_shared<MLP>(100, 64, 10);
    model->to(device);
    auto params = model->parameters();
    auto optimizer = Adam(params, 0.001);

    const int num_epochs = 5;
    const int train_batches = 10;
    const int val_batches = 3;
    const int batch_size = 32;

    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Training phase
        float train_loss = 0.0f;
        model->train();

        for (int batch = 0; batch < train_batches; batch++) {
            auto input = Variable(randn({batch_size, 100}, DType::Float32, device), true);
            auto target_data = zeros({batch_size, 10}, DType::Float32, device);
            auto target = Variable(target_data, false);

            auto output = model->forward(input);
            auto loss = mse_loss(output, target, Reduction::Mean);

            optimizer.zero_grad();
            loss.backward();
            optimizer.step();

            train_loss += loss.tensor().to(Device::cpu()).template data<float>()[0];
        }
        train_loss /= train_batches;

        // Validation phase
        float val_loss = 0.0f;
        model->eval();

        for (int batch = 0; batch < val_batches; batch++) {
            auto input = Variable(randn({batch_size, 100}, DType::Float32, device), false);
            auto target_data = zeros({batch_size, 10}, DType::Float32, device);
            auto target = Variable(target_data, false);

            auto output = model->forward(input);
            auto loss = mse_loss(output, target, Reduction::Mean);

            val_loss += loss.tensor().to(Device::cpu()).template data<float>()[0];
        }
        val_loss /= val_batches;

        EXPECT_GT(train_loss, 0.0f);
        EXPECT_GT(val_loss, 0.0f);
    }
}

//==============================================================================
// Test 7: Backend Result Consistency
//==============================================================================

TEST_P(AcceleratorTrainingTest, BackendResultConsistency) {
    // This test verifies that the same model produces reasonable results
    // when trained on any backend

    const int input_size = 50;
    const int hidden_size = 30;
    const int output_size = 5;
    const int batch_size = 16;

    auto model = std::make_shared<MLP>(input_size, hidden_size, output_size);
    model->to(device);

    auto input = Variable(randn({batch_size, input_size}, DType::Float32, device), true);
    auto output = model->forward(input);

    EXPECT_EQ(output.shape()[0], batch_size);
    EXPECT_EQ(output.shape()[1], output_size);
    EXPECT_EQ(output.device().type, device.type);

    // Verify output values are reasonable (not NaN, not too large)
    auto output_cpu = output.tensor().to(Device::cpu());
    auto* data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN detected in output on " << device.to_string();
        EXPECT_FALSE(std::isinf(data[i])) << "Inf detected in output on " << device.to_string();
        EXPECT_LT(std::abs(data[i]), 1e6f) << "Extremely large value in output on " << device.to_string();
    }
}

INSTANTIATE_BACKEND_TESTS(AcceleratorTrainingTest);
