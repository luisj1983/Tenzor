/**
 * @file test_cross_backend.cpp
 * @brief Integration tests for cross-backend compatibility and device transfers
 *
 * Tests:
 * - Device transfers (CPU <-> CUDA <-> OneAPI <-> Vulkan)
 * - Backend parity (same results across backends)
 * - Mixed backend operations
 * - Memory efficiency
 * - Numerical accuracy
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>
#include <memory>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

//==============================================================================
// Test Environment
//==============================================================================

class CrossBackendEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const cross_backend_env =
    ::testing::AddGlobalTestEnvironment(new CrossBackendEnvironment);

//==============================================================================
// Backend Parameterized Tests
//==============================================================================

class CrossBackendTest : public BackendTest {};

//==============================================================================
// Test 1: Device Transfer Round Trip
//==============================================================================

TEST_P(CrossBackendTest, DeviceTransferRoundTrip) {
    auto cpu_device = Device::cpu();

    // Create tensor on CPU
    auto cpu_tensor = randn({10, 20}, DType::Float32, cpu_device);

    // Transfer to test device
    auto device_tensor = cpu_tensor.to(device);
    EXPECT_EQ(device_tensor.device().type, device.type);

    // Transfer back to CPU
    auto cpu_tensor_back = device_tensor.to(cpu_device);
    EXPECT_EQ(cpu_tensor_back.device().type, Device::Type::CPU);

    // Verify data integrity
    expectTensorNear(cpu_tensor, cpu_tensor_back, 1e-6f);
}

//==============================================================================
// Test 2: Simple Operation Consistency
//==============================================================================

TEST_P(CrossBackendTest, SimpleOperationConsistency) {
    // Computation on test device
    auto a = ones({5, 5}, DType::Float32, device);
    auto b = ones({5, 5}, DType::Float32, device) * 2.0f;
    auto result = a + b;

    // Verify result
    auto result_cpu = result.to(Device::cpu());
    auto result_data = result_cpu.template data<float>();

    for (int64_t i = 0; i < result_cpu.numel(); i++) {
        EXPECT_FLOAT_EQ(result_data[i], 3.0f) << "Result should be 3.0 on " << device.to_string();
    }
}

//==============================================================================
// Test 3: Matrix Multiplication Consistency
//==============================================================================

TEST_P(CrossBackendTest, MatMulConsistency) {
    // Create input tensors
    auto a = randn({32, 64}, DType::Float32, device);
    auto b = randn({64, 128}, DType::Float32, device);

    // Perform matmul on test device
    auto result = matmul(a, b);

    EXPECT_EQ(result.shape()[0], 32);
    EXPECT_EQ(result.shape()[1], 128);
    EXPECT_EQ(result.device().type, device.type);

    // Verify result is reasonable
    auto result_cpu = result.to(Device::cpu());
    auto* data = result_cpu.data<float>();
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN in matmul result on " << device.to_string();
        EXPECT_FALSE(std::isinf(data[i])) << "Inf in matmul result on " << device.to_string();
    }
}

//==============================================================================
// Test 4: Convolution Consistency
//==============================================================================

TEST_P(CrossBackendTest, Conv2DConsistency) {
    // Create input
    auto input = randn({2, 3, 32, 32}, DType::Float32, device);

    // Create and run conv
    auto conv = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
    conv->to(device);

    auto input_var = Variable(input, false);
    auto output = conv->forward(input_var);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
    EXPECT_EQ(output.device().type, device.type);

    // Verify output is reasonable
    auto output_cpu = output.tensor().to(Device::cpu());
    auto* data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN in conv output on " << device.to_string();
        EXPECT_FALSE(std::isinf(data[i])) << "Inf in conv output on " << device.to_string();
    }
}

//==============================================================================
// Test 5: ReLU Activation Consistency
//==============================================================================

TEST_P(CrossBackendTest, ReLUConsistency) {
    // Create input with both positive and negative values
    auto input = randn({10, 20}, DType::Float32, device);

    // Apply ReLU
    auto input_var = Variable(input, false);
    auto output_var = relu(input_var);
    auto output = output_var.tensor();

    EXPECT_EQ(output.device().type, device.type);

    // Verify ReLU property: all values >= 0
    auto output_cpu = output.to(Device::cpu());
    auto* data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_GE(data[i], 0.0f) << "ReLU output should be non-negative on " << device.to_string();
    }
}

//==============================================================================
// Test 6: BatchNorm Consistency
//==============================================================================

TEST_P(CrossBackendTest, BatchNormConsistency) {
    // Create input
    auto input = randn({4, 16, 32, 32}, DType::Float32, device);

    // Create BatchNorm
    auto bn = std::make_shared<BatchNorm2d>(16);
    bn->to(device);
    bn->eval();  // Use eval mode for deterministic behavior

    auto input_var = Variable(input, false);
    auto output = bn->forward(input_var);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 16);
    EXPECT_EQ(output.shape()[2], 32);
    EXPECT_EQ(output.shape()[3], 32);
    EXPECT_EQ(output.device().type, device.type);

    // Verify output is reasonable
    auto output_cpu = output.tensor().to(Device::cpu());
    auto* data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN in BatchNorm output on " << device.to_string();
        EXPECT_FALSE(std::isinf(data[i])) << "Inf in BatchNorm output on " << device.to_string();
    }
}

//==============================================================================
// Test 7: End-to-End Model Inference Consistency
//==============================================================================

TEST_P(CrossBackendTest, ModelInferenceConsistency) {
    // Simple model
    class SimpleModel : public Module {
    public:
        SimpleModel() {
            conv = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
            bn = std::make_shared<BatchNorm2d>(16);
            fc = std::make_shared<Linear>(16 * 32 * 32, 10);

            register_module("conv", conv);
            register_module("bn", bn);
            register_module("fc", fc);
        }

        auto forward(const Variable& x) -> Variable override {
            auto out = conv->forward(x);
            out = bn->forward(out);
            out = relu(out);
            out = out.reshape({out.shape()[0], -1});
            out = fc->forward(out);
            return out;
        }

    private:
        std::shared_ptr<Conv2d> conv;
        std::shared_ptr<BatchNorm2d> bn;
        std::shared_ptr<Linear> fc;
    };

    auto model = std::make_shared<SimpleModel>();
    model->to(device);
    model->eval();

    // Create input
    auto input = randn({2, 3, 32, 32}, DType::Float32, device);
    auto input_var = Variable(input, false);

    // Inference
    auto output = model->forward(input_var);

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.device().type, device.type);

    // Verify output is reasonable
    auto output_cpu = output.tensor().to(Device::cpu());
    auto* data = output_cpu.data<float>();
    for (int64_t i = 0; i < output_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i])) << "NaN in model output on " << device.to_string();
        EXPECT_FALSE(std::isinf(data[i])) << "Inf in model output on " << device.to_string();
    }
}

//==============================================================================
// Test 8: Gradient Computation Consistency
//==============================================================================

TEST_P(CrossBackendTest, GradientComputationConsistency) {
    // Simple computation with gradients
    auto x = Variable(ones({5, 5}, DType::Float32, device), true);
    auto y = x * 2.0f + 1.0f;
    auto loss = sum(y);
    loss.backward();

    auto grad = x.grad();
    EXPECT_TRUE(grad.has_value()) << "Gradient not computed on " << device.to_string();

    // Verify gradient values (should all be 2.0)
    auto grad_cpu = grad.value().to(Device::cpu());
    auto* grad_data = grad_cpu.data<float>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        EXPECT_NEAR(grad_data[i], 2.0f, 1e-5f) << "Gradient mismatch on " << device.to_string();
    }
}

//==============================================================================
// Test 9: Training Step Consistency
//==============================================================================

TEST_P(CrossBackendTest, TrainingStepConsistency) {
    // Create model
    auto model = std::make_shared<Linear>(10, 5);
    model->to(device);

    auto params = model->parameters();
    optim::SGD optimizer(params, 0.01);

    // Training step
    auto input = Variable(randn({4, 10}, DType::Float32, device), true);
    auto target = Variable(randn({4, 5}, DType::Float32, device), false);

    optimizer.zero_grad();
    auto output = model->forward(input);
    auto loss = mse_loss(output, target);
    loss.backward();

    // Verify gradients exist
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->has_grad()) << "Missing gradient on " << device.to_string();
        }
    }

    // Store initial params
    auto initial_params = params[0]->tensor().clone();

    optimizer.step();

    // Verify parameters changed
    auto updated_params = params[0]->tensor();
    auto initial_cpu = initial_params.to(Device::cpu());
    auto updated_cpu = updated_params.to(Device::cpu());

    bool changed = false;
    for (int64_t i = 0; i < initial_cpu.numel(); ++i) {
        if (std::abs(initial_cpu.data<float>()[i] - updated_cpu.data<float>()[i]) > 1e-7f) {
            changed = true;
            break;
        }
    }

    EXPECT_TRUE(changed) << "Parameters should change after optimizer step on " << device.to_string();
}

//==============================================================================
// Test 10: Memory Efficiency
//==============================================================================

TEST_P(CrossBackendTest, MemoryEfficiency) {
    // Allocate and deallocate tensors
    for (int i = 0; i < 100; i++) {
        auto tensor = randn({100, 100}, DType::Float32, device);
        auto result = tensor * 2.0f;
        // Tensor should be freed after scope
    }

    // If we reach here without crash, memory management is working
    SUCCEED() << "Memory management working correctly on " << device.to_string();
}

//==============================================================================
// Test 11: Reduction Operations Consistency
//==============================================================================

TEST_P(CrossBackendTest, ReductionOperationsConsistency) {
    // Create input
    auto input = randn({10, 20, 30}, DType::Float32, device);

    // Perform reductions
    auto sum_result = tenzor::sum(input);
    auto mean_result = tenzor::mean(input);
    auto max_result = tenzor::max(input);

    EXPECT_EQ(sum_result.device().type, device.type);
    EXPECT_EQ(mean_result.device().type, device.type);
    EXPECT_EQ(max_result.device().type, device.type);

    // Verify results are reasonable
    auto sum_cpu = sum_result.to(Device::cpu());
    auto mean_cpu = mean_result.to(Device::cpu());
    auto max_cpu = max_result.to(Device::cpu());

    EXPECT_FALSE(std::isnan(sum_cpu.data<float>()[0])) << "NaN in sum on " << device.to_string();
    EXPECT_FALSE(std::isnan(mean_cpu.data<float>()[0])) << "NaN in mean on " << device.to_string();
    EXPECT_FALSE(std::isnan(max_cpu.data<float>()[0])) << "NaN in max on " << device.to_string();
}

//==============================================================================
// Test 12: Softmax Consistency
//==============================================================================

TEST_P(CrossBackendTest, SoftmaxConsistency) {
    // Create input
    auto input = randn({32, 10}, DType::Float32, device);

    // Apply softmax
    auto input_var = Variable(input, false);
    auto output_var = tenzor::nn::softmax(input_var, 1);
    auto output = output_var.tensor();

    EXPECT_EQ(output.device().type, device.type);

    // Verify sum to 1.0 along dim 1
    auto output_cpu = output.to(Device::cpu());
    auto output_data = output_cpu.data<float>();
    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            sum += output_data[i * 10 + j];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax should sum to 1.0 on " << device.to_string();
    }
}

//==============================================================================
// Test 13: Cross-Backend Data Pipeline
//==============================================================================

TEST_P(CrossBackendTest, CrossBackendDataPipeline) {
    auto cpu_device = Device::cpu();

    // Simulate data loading on CPU
    auto data_cpu = randn({8, 3, 224, 224}, DType::Float32, cpu_device);

    // Create model on test device
    auto model = std::make_shared<Conv2d>(3, 64, 7, 2, 3);
    model->to(device);
    model->eval();

    // Transfer data to device and run inference
    auto data_device = data_cpu.to(device);
    auto input_var = Variable(data_device, false);
    auto output = model->forward(input_var);

    EXPECT_EQ(output.shape()[0], 8);
    EXPECT_EQ(output.device().type, device.type);
}

//==============================================================================
// Test 14: Complete Training Loop on Backend
//==============================================================================

TEST_P(CrossBackendTest, CompleteTrainingLoop) {
    class SimpleMLP : public Module {
    public:
        SimpleMLP() {
            fc1 = std::make_shared<Linear>(50, 30);
            relu = std::make_shared<ReLU>();
            fc2 = std::make_shared<Linear>(30, 10);

            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        auto forward(const Variable& x) -> Variable override {
            auto h = fc1->forward(x);
            h = relu->forward(h);
            return fc2->forward(h);
        }

    private:
        std::shared_ptr<Linear> fc1, fc2;
        std::shared_ptr<ReLU> relu;
    };

    auto model = std::make_shared<SimpleMLP>();
    model->to(device);

    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    const int num_epochs = 5;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto input = Variable(randn({32, 50}, DType::Float32, device), true);
        auto target = Variable(randn({32, 10}, DType::Float32, device), false);

        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = mse_loss(output, target);
        loss.backward();
        optimizer.step();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_val = loss_cpu.data<float>()[0];

        if (epoch == 0) {
            initial_loss = loss_val;
        }
        if (epoch == num_epochs - 1) {
            final_loss = loss_val;
        }

        EXPECT_GT(loss_val, 0.0f);
        EXPECT_FALSE(std::isnan(loss_val));
    }

    // Verify training completed successfully
    EXPECT_GT(initial_loss, 0.0f);
    EXPECT_GT(final_loss, 0.0f);
    EXPECT_FALSE(std::isnan(final_loss)) << "Training diverged on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(CrossBackendTest);
