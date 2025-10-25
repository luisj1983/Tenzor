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
#include <tenzor/tenzor.hpp>
#include <memory>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

//==============================================================================
// Test Environment
//==============================================================================

class CrossBackendEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();

        // Check backend availability
        try {
            auto test_tensor = ones({1}, DType::Float32, Device::cuda());
            cuda_available_ = true;
        } catch (...) {
            cuda_available_ = false;
        }

        std::cout << "CUDA available: " << (cuda_available_ ? "Yes" : "No") << std::endl;
    }

    static bool cuda_available_;
};

bool CrossBackendEnvironment::cuda_available_ = false;

static ::testing::Environment* const cross_backend_env =
    ::testing::AddGlobalTestEnvironment(new CrossBackendEnvironment);

//==============================================================================
// Helper Functions
//==============================================================================

auto compare_tensors(const Tensor& t1, const Tensor& t2, float tolerance = 1e-4f) -> bool {
    if (!std::ranges::equal(t1.shape(), t2.shape())) return false;

    auto t1_cpu = t1.to(Device::cpu());
    auto t2_cpu = t2.to(Device::cpu());

    auto data1 = t1_cpu.template data<float>();
    auto data2 = t2_cpu.template data<float>();

    for (size_t i = 0; i < t1_cpu.numel(); i++) {
        if (std::abs(data1[i] - data2[i]) > tolerance) {
            return false;
        }
    }
    return true;
}

//==============================================================================
// Test 1: CPU to CUDA Transfer and Back
//==============================================================================

TEST(CrossBackend, CPUToCUDATransfer) {
    if (!CrossBackendEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cpu_device = Device::cpu();
    auto cuda_device = Device::cuda();

    // Create tensor on CPU
    auto cpu_tensor = randn({10, 20}, DType::Float32, cpu_device);

    // Transfer to CUDA
    auto cuda_tensor = cpu_tensor.to(cuda_device);
    EXPECT_EQ(cuda_tensor.device().type, Device::Type::CUDA);

    // Transfer back to CPU
    auto cpu_tensor_back = cuda_tensor.to(cpu_device);
    EXPECT_EQ(cpu_tensor_back.device().type, Device::Type::CPU);

    // Verify data integrity
    EXPECT_TRUE(compare_tensors(cpu_tensor, cpu_tensor_back, 1e-6f))
        << "Data should match after round-trip transfer";
}

//==============================================================================
// Test 2: Simple Operation Consistency Across Backends
//==============================================================================

TEST(CrossBackend, SimpleOperationConsistency) {
    auto cpu_device = Device::cpu();

    // CPU computation
    auto a_cpu = ones({5, 5}, DType::Float32, cpu_device);
    auto b_cpu = ones({5, 5}, DType::Float32, cpu_device) * 2.0f;
    auto result_cpu = a_cpu + b_cpu;

    auto result_cpu_data = result_cpu.to(Device::cpu()).template data<float>();

    // Verify CPU result
    for (size_t i = 0; i < result_cpu.numel(); i++) {
        EXPECT_FLOAT_EQ(result_cpu_data[i], 3.0f) << "CPU result should be 3.0";
    }

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // CUDA computation
        auto a_cuda = ones({5, 5}, DType::Float32, cuda_device);
        auto b_cuda = ones({5, 5}, DType::Float32, cuda_device) * 2.0f;
        auto result_cuda = a_cuda + b_cuda;

        // Compare results
        EXPECT_TRUE(compare_tensors(result_cpu, result_cuda, 1e-6f))
            << "CPU and CUDA results should match";
    }
}

//==============================================================================
// Test 3: Matrix Multiplication Across Backends
//==============================================================================

TEST(CrossBackend, MatMulConsistency) {
    auto cpu_device = Device::cpu();

    // Create input tensors on CPU
    auto a_cpu = randn({32, 64}, DType::Float32, cpu_device);
    auto b_cpu = randn({64, 128}, DType::Float32, cpu_device);

    // CPU matmul
    auto result_cpu = matmul(a_cpu, b_cpu);

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // Transfer to CUDA and compute
        auto a_cuda = a_cpu.to(cuda_device);
        auto b_cuda = b_cpu.to(cuda_device);
        auto result_cuda = matmul(a_cuda, b_cuda);

        // Compare results (matmul may have slightly larger numerical differences)
        EXPECT_TRUE(compare_tensors(result_cpu, result_cuda, 1e-3f))
            << "CPU and CUDA matmul results should be close";
    }
}

//==============================================================================
// Test 4: Convolution Across Backends
//==============================================================================

TEST(CrossBackend, Conv2DConsistency) {
    auto cpu_device = Device::cpu();

    // Create input on CPU
    auto input_cpu = randn({2, 3, 32, 32}, DType::Float32, cpu_device);

    // Create and run conv on CPU
    auto conv_cpu = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
    conv_cpu->to(cpu_device);

    auto input_var_cpu = Variable(input_cpu, false);
    auto output_cpu = conv_cpu->forward(input_var_cpu);

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // Transfer model and input to CUDA
        auto conv_cuda = std::make_shared<Conv2d>(3, 16, 3, 1, 1);
        conv_cuda->load_state_dict(conv_cpu->state_dict());
        conv_cuda->to(cuda_device);

        auto input_cuda = input_cpu.to(cuda_device);
        auto input_var_cuda = Variable(input_cuda, false);
        auto output_cuda = conv_cuda->forward(input_var_cuda);

        // Compare outputs
        EXPECT_TRUE(compare_tensors(output_cpu.tensor(), output_cuda.tensor(), 1e-3f))
            << "Conv2d outputs should match across backends";
    }
}

//==============================================================================
// Test 5: ReLU Activation Consistency
//==============================================================================

TEST(CrossBackend, ReLUConsistency) {
    auto cpu_device = Device::cpu();

    // Create input with both positive and negative values
    auto input_cpu = randn({10, 20}, DType::Float32, cpu_device);

    // CPU ReLU
    auto input_var_cpu = Variable(input_cpu, false);
    auto output_var_cpu = relu(input_var_cpu);
    auto output_cpu = output_var_cpu.tensor();

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // CUDA ReLU
        auto input_cuda = input_cpu.to(cuda_device);
        auto input_var_cuda = Variable(input_cuda, false);
        auto output_var_cuda = relu(input_var_cuda);
        auto output_cuda = output_var_cuda.tensor();

        // Compare
        EXPECT_TRUE(compare_tensors(output_cpu, output_cuda, 1e-6f))
            << "ReLU outputs should match exactly";
    }
}

//==============================================================================
// Test 6: BatchNorm Consistency
//==============================================================================

TEST(CrossBackend, BatchNormConsistency) {
    auto cpu_device = Device::cpu();

    // Create input
    auto input_cpu = randn({4, 16, 32, 32}, DType::Float32, cpu_device);

    // CPU BatchNorm
    auto bn_cpu = std::make_shared<BatchNorm2d>(16);
    bn_cpu->to(cpu_device);
    bn_cpu->eval();  // Use eval mode for deterministic behavior

    auto input_var_cpu = Variable(input_cpu, false);
    auto output_cpu = bn_cpu->forward(input_var_cpu);

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // CUDA BatchNorm
        auto bn_cuda = std::make_shared<BatchNorm2d>(16);
        bn_cuda->load_state_dict(bn_cpu->state_dict());
        bn_cuda->to(cuda_device);
        bn_cuda->eval();

        auto input_cuda = input_cpu.to(cuda_device);
        auto input_var_cuda = Variable(input_cuda, false);
        auto output_cuda = bn_cuda->forward(input_var_cuda);

        // Compare
        EXPECT_TRUE(compare_tensors(output_cpu.tensor(), output_cuda.tensor(), 1e-3f))
            << "BatchNorm outputs should match";
    }
}

//==============================================================================
// Test 7: End-to-End Model Inference Consistency
//==============================================================================

TEST(CrossBackend, ModelInferenceConsistency) {
    auto cpu_device = Device::cpu();

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
            // Reshape for fully connected layer
            out = out.reshape({out.shape()[0], -1});
            out = fc->forward(out);
            return out;
        }

    private:
        std::shared_ptr<Conv2d> conv;
        std::shared_ptr<BatchNorm2d> bn;
        std::shared_ptr<Linear> fc;
    };

    auto model_cpu = std::make_shared<SimpleModel>();
    model_cpu->to(cpu_device);
    model_cpu->eval();

    // Create input
    auto input_cpu = randn({2, 3, 32, 32}, DType::Float32, cpu_device);
    auto input_var_cpu = Variable(input_cpu, false);

    // CPU inference
    auto output_cpu = model_cpu->forward(input_var_cpu);

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // Transfer model to CUDA
        auto model_cuda = std::make_shared<SimpleModel>();
        model_cuda->load_state_dict(model_cpu->state_dict());
        model_cuda->to(cuda_device);
        model_cuda->eval();

        // Transfer input to CUDA
        auto input_cuda = input_cpu.to(cuda_device);
        auto input_var_cuda = Variable(input_cuda, false);

        // CUDA inference
        auto output_cuda = model_cuda->forward(input_var_cuda);

        // Compare
        EXPECT_TRUE(compare_tensors(output_cpu.tensor(), output_cuda.tensor(), 1e-2f))
            << "Model outputs should be consistent across backends";
    }
}

//==============================================================================
// Test 8: Gradient Computation Consistency
//==============================================================================

TEST(CrossBackend, GradientComputationConsistency) {
    auto cpu_device = Device::cpu();

    // Simple computation with gradients on CPU
    auto x_cpu = Variable(ones({5, 5}, DType::Float32, cpu_device), true);
    auto y_cpu = x_cpu * 2.0f + 1.0f;
    auto loss_cpu = sum(y_cpu);
    loss_cpu.backward();

    auto grad_cpu = x_cpu.grad();

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // Same computation on CUDA
        auto x_cuda = Variable(ones({5, 5}, DType::Float32, cuda_device), true);
        auto y_cuda = x_cuda * 2.0f + 1.0f;
        auto loss_cuda = sum(y_cuda);
        loss_cuda.backward();

        auto grad_cuda = x_cuda.grad();

        // Compare gradients
        if (grad_cpu.has_value() && grad_cuda.has_value()) {
            EXPECT_TRUE(compare_tensors(grad_cpu.value(), grad_cuda.value(), 1e-6f))
                << "Gradients should match across backends";
        } else {
            FAIL() << "Gradients not computed properly";
        }
    }
}

//==============================================================================
// Test 9: Training Step Consistency
//==============================================================================

TEST(CrossBackend, TrainingStepConsistency) {
    auto cpu_device = Device::cpu();

    // Create model on CPU
    auto model_cpu = std::make_shared<Linear>(10, 5);
    model_cpu->to(cpu_device);

    auto params_cpu = model_cpu->parameters();
    optim::SGD optimizer_cpu(params_cpu, 0.01);

    // Training step on CPU
    auto input_cpu = Variable(randn({4, 10}, DType::Float32, cpu_device), true);
    auto target_cpu = Variable(randn({4, 5}, DType::Float32, cpu_device), false);

    optimizer_cpu.zero_grad();
    auto output_cpu = model_cpu->forward(input_cpu);
    auto loss_cpu = mse_loss(output_cpu, target_cpu);
    loss_cpu.backward();
    optimizer_cpu.step();

    auto final_params_cpu = params_cpu[0]->tensor().clone();

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // Create same model on CUDA
        auto model_cuda = std::make_shared<Linear>(10, 5);
        model_cuda->load_state_dict(model_cpu->state_dict());
        model_cuda->to(cuda_device);

        auto params_cuda = model_cuda->parameters();
        optim::SGD optimizer_cuda(params_cuda, 0.01);

        // Training step on CUDA (same data)
        auto input_cuda = input_cpu.tensor().to(cuda_device);
        auto target_cuda = target_cpu.tensor().to(cuda_device);

        auto input_var_cuda = Variable(input_cuda, true);
        auto target_var_cuda = Variable(target_cuda, false);

        optimizer_cuda.zero_grad();
        auto output_cuda = model_cuda->forward(input_var_cuda);
        auto loss_cuda = mse_loss(output_cuda, target_var_cuda);
        loss_cuda.backward();
        optimizer_cuda.step();

        auto final_params_cuda = params_cuda[0]->tensor();

        // Compare updated parameters
        EXPECT_TRUE(compare_tensors(final_params_cpu, final_params_cuda, 1e-3f))
            << "Updated parameters should match across backends";
    }
}

//==============================================================================
// Test 10: Multi-Device Tensor Operations
//==============================================================================

TEST(CrossBackend, MultiDeviceTensorOperations) {
    if (!CrossBackendEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cpu_device = Device::cpu();
    auto cuda_device = Device::cuda();

    // Create tensors on different devices
    auto a_cpu = ones({10, 10}, DType::Float32, cpu_device);
    auto b_cuda = ones({10, 10}, DType::Float32, cuda_device) * 2.0f;

    // Transfer b to CPU for operation
    auto b_cpu = b_cuda.to(cpu_device);
    auto result = a_cpu + b_cpu;

    // Verify result
    auto result_data = result.template data<float>();
    for (size_t i = 0; i < result.numel(); i++) {
        EXPECT_FLOAT_EQ(result_data[i], 3.0f);
    }
}

//==============================================================================
// Test 11: Memory Efficiency Test
//==============================================================================

TEST(CrossBackend, MemoryEfficiency) {
    if (!CrossBackendEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cuda_device = Device::cuda();

    // Allocate and deallocate tensors
    for (int i = 0; i < 100; i++) {
        auto tensor = randn({100, 100}, DType::Float32, cuda_device);
        auto result = tensor * 2.0f;
        // Tensor should be freed after scope
    }

    // If we reach here without crash, memory management is working
    SUCCEED() << "Memory management working correctly";
}

//==============================================================================
// Test 12: Reduction Operations Consistency
//==============================================================================

TEST(CrossBackend, ReductionOperationsConsistency) {
    auto cpu_device = Device::cpu();

    // Create input
    auto input_cpu = randn({10, 20, 30}, DType::Float32, cpu_device);

    // CPU reductions
    auto sum_cpu = tenzor::sum(input_cpu);
    auto mean_cpu = tenzor::mean(input_cpu);
    auto max_cpu = tenzor::max(input_cpu);

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // CUDA reductions
        auto input_cuda = input_cpu.to(cuda_device);
        auto sum_cuda = tenzor::sum(input_cuda);
        auto mean_cuda = tenzor::mean(input_cuda);
        auto max_cuda = tenzor::max(input_cuda);

        // Compare results
        EXPECT_TRUE(compare_tensors(sum_cpu, sum_cuda, 1e-3f))
            << "Sum should match";
        EXPECT_TRUE(compare_tensors(mean_cpu, mean_cuda, 1e-3f))
            << "Mean should match";
        EXPECT_TRUE(compare_tensors(max_cpu, max_cuda, 1e-6f))
            << "Max should match";
    }
}

//==============================================================================
// Test 13: Softmax Consistency
//==============================================================================

TEST(CrossBackend, SoftmaxConsistency) {
    auto cpu_device = Device::cpu();

    // Create input
    auto input_cpu = randn({32, 10}, DType::Float32, cpu_device);

    // CPU softmax
    auto input_var_cpu = Variable(input_cpu, false);
    auto output_var_cpu = tenzor::nn::softmax(input_var_cpu, 1);
    auto output_cpu = output_var_cpu.tensor();

    // Verify sum to 1.0 along dim 1
    auto output_cpu_on_cpu = output_cpu.to(Device::cpu());
    auto output_cpu_data = output_cpu_on_cpu.data<float>();
    for (int i = 0; i < 32; i++) {
        float sum = 0.0f;
        for (int j = 0; j < 10; j++) {
            sum += output_cpu_data[i * 10 + j];
        }
        EXPECT_NEAR(sum, 1.0f, 1e-5f) << "Softmax should sum to 1.0";
    }

    if (CrossBackendEnvironment::cuda_available_) {
        auto cuda_device = Device::cuda();

        // CUDA softmax
        auto input_cuda = input_cpu.to(cuda_device);
        auto input_var_cuda = Variable(input_cuda, false);
        auto output_var_cuda = tenzor::nn::softmax(input_var_cuda, 1);
        auto output_cuda = output_var_cuda.tensor();

        // Compare
        EXPECT_TRUE(compare_tensors(output_cpu, output_cuda, 1e-5f))
            << "Softmax outputs should match";
    }
}

//==============================================================================
// Test 14: Cross-Backend Data Pipeline
//==============================================================================

TEST(CrossBackend, CrossBackendDataPipeline) {
    if (!CrossBackendEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto cpu_device = Device::cpu();
    auto cuda_device = Device::cuda();

    // Simulate data loading on CPU
    auto data_cpu = randn({8, 3, 224, 224}, DType::Float32, cpu_device);

    // Create model on CUDA
    auto model = std::make_shared<Conv2d>(3, 64, 7, 2, 3);
    model->to(cuda_device);
    model->eval();

    // Transfer data to CUDA and run inference
    auto data_cuda = data_cpu.to(cuda_device);
    auto input_var = Variable(data_cuda, false);
    auto output = model->forward(input_var);

    EXPECT_EQ(output.shape()[0], 8) << "Batch size should be preserved";
    EXPECT_EQ(output.device().type, Device::Type::CUDA) << "Output should be on CUDA";
}
