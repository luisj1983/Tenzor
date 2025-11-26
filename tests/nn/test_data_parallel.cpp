/**
 * @file test_data_parallel.cpp
 * @brief Tests for DataParallel multi-GPU gradient synchronization
 */

#include <gtest/gtest.h>
#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/nn/modules/linear.hpp"
#include "tenzor/nn/modules/sequential.hpp"
#include "tenzor/ops/loss.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::nn;

/**
 * @brief Simple test module for data parallel testing
 */
class SimpleModule : public Module {
public:
    SimpleModule(int input_size, int hidden_size, int output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto h = fc1_->forward(input);
        // Simple ReLU
        auto h_relu = Variable(h.tensor() * (h.tensor() > 0.0f));
        return fc2_->forward(h_relu);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

// Test fixture
class DataParallelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
#ifdef TENZOR_USE_CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        has_cuda_ = (device_count > 0);
        num_gpus_ = device_count;
#else
        has_cuda_ = false;
        num_gpus_ = 0;
#endif
    }

    bool has_cuda_{false};
    int num_gpus_{0};
};

TEST_F(DataParallelTest, ConstructorAutoDetectDevices) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);

    // Auto-detect all available GPUs
    EXPECT_NO_THROW({
        auto parallel = DataParallel(module);
        EXPECT_GT(parallel.device_ids().size(), 0);
        EXPECT_EQ(parallel.output_device(), parallel.device_ids()[0]);
    });
}

TEST_F(DataParallelTest, ConstructorExplicitDevices) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);

    // Explicitly specify device 0
    EXPECT_NO_THROW({
        auto parallel = DataParallel(module, {0}, 0);
        EXPECT_EQ(parallel.device_ids().size(), 1);
        EXPECT_EQ(parallel.device_ids()[0], 0);
        EXPECT_EQ(parallel.output_device(), 0);
    });
}

TEST_F(DataParallelTest, ConstructorInvalidDevice) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);

    // Try to use non-existent device
    EXPECT_THROW({
        auto parallel = DataParallel(module, {999}, 999);
    }, std::invalid_argument);
}

TEST_F(DataParallelTest, ForwardSingleGPU) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0}, 0);

    // Create input on GPU 0
    Tensor input_tensor({4, 10}, DType::Float32, Device::cuda(0));
    input_tensor.fill_(1.0f);
    Variable input(input_tensor, true);

    // Forward pass
    Variable output;
    EXPECT_NO_THROW({
        output = parallel->forward(input);
    });

    // Check output shape
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 4);
    EXPECT_EQ(output_shape[1], 5);

    // Check output is on master device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
    EXPECT_EQ(output.device().index, 0);
}

TEST_F(DataParallelTest, ForwardMultiGPU) {
    if (!has_cuda_ || num_gpus_ < 2) {
        GTEST_SKIP() << "Multiple GPUs not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0, 1}, 0);

    // Create input on GPU 0 with batch size 8 (4 per GPU)
    Tensor input_tensor({8, 10}, DType::Float32, Device::cuda(0));
    input_tensor.fill_(1.0f);
    Variable input(input_tensor, true);

    // Forward pass
    Variable output;
    EXPECT_NO_THROW({
        output = parallel->forward(input);
    });

    // Check output shape
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape.size(), 2);
    EXPECT_EQ(output_shape[0], 8);
    EXPECT_EQ(output_shape[1], 5);

    // Check output is on master device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
    EXPECT_EQ(output.device().index, 0);
}

TEST_F(DataParallelTest, ForwardBatchTooSmall) {
    if (!has_cuda_ || num_gpus_ < 2) {
        GTEST_SKIP() << "Multiple GPUs not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0, 1}, 0);

    // Create input with batch size 1 (too small for 2 GPUs)
    Tensor input_tensor({1, 10}, DType::Float32, Device::cuda(0));
    input_tensor.fill_(1.0f);
    Variable input(input_tensor, true);

    // Forward pass should fail
    EXPECT_THROW({
        auto output = parallel->forward(input);
    }, std::runtime_error);
}

TEST_F(DataParallelTest, GradientSynchronizationSingleGPU) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0}, 0);

    // Create input
    Tensor input_tensor({4, 10}, DType::Float32, Device::cuda(0));
    input_tensor.fill_(1.0f);
    Variable input(input_tensor, true);

    // Forward pass
    auto output = parallel->forward(input);

    // Backward pass
    Tensor grad_output({4, 5}, DType::Float32, Device::cuda(0));
    grad_output.fill_(1.0f);
    output.backward(grad_output);

    // Call synchronize_gradients (should be no-op for single GPU)
    EXPECT_NO_THROW({
        parallel->synchronize_gradients();
    });

    // Check that parameters have gradients
    auto params = parallel->parameters();
    for (auto* param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

TEST_F(DataParallelTest, GradientSynchronizationMultiGPU) {
    if (!has_cuda_ || num_gpus_ < 2) {
        GTEST_SKIP() << "Multiple GPUs not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0, 1}, 0);

    // Create input
    Tensor input_tensor({8, 10}, DType::Float32, Device::cuda(0));
    input_tensor.fill_(1.0f);
    Variable input(input_tensor, true);

    // Forward pass
    auto output = parallel->forward(input);

    // Backward pass
    Tensor grad_output({8, 5}, DType::Float32, Device::cuda(0));
    grad_output.fill_(1.0f);
    output.backward(grad_output);

    // Call synchronize_gradients
    EXPECT_NO_THROW({
        parallel->synchronize_gradients();
    });

    // Check that parameters have gradients
    auto params = parallel->parameters();
    for (auto* param : params) {
        EXPECT_TRUE(param->has_grad());

        // Gradients should be averaged across 2 GPUs
        // (In practice, gradient values would be half of single-GPU case)
    }
}

TEST_F(DataParallelTest, ParametersReturnMasterParams) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0}, 0);

    // Get parameters from both module and parallel wrapper
    auto module_params = module->parameters();
    auto parallel_params = parallel->parameters();

    // Should return same parameters
    EXPECT_EQ(module_params.size(), parallel_params.size());

    for (size_t i = 0; i < module_params.size(); ++i) {
        EXPECT_EQ(module_params[i], parallel_params[i]);
    }
}

TEST_F(DataParallelTest, TrainingModePropagatesToReplicas) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0}, 0);

    // Initial state: training
    EXPECT_TRUE(module->is_training());

    // Set to eval mode
    parallel->eval();
    EXPECT_FALSE(module->is_training());

    // Set back to training mode
    parallel->train();
    EXPECT_TRUE(module->is_training());
}

TEST_F(DataParallelTest, MakeDataParallelHelper) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);

    // Use helper function
    auto parallel = make_data_parallel(module);

    EXPECT_NE(parallel, nullptr);
    EXPECT_GT(parallel->device_ids().size(), 0);
}

TEST_F(DataParallelTest, UnevenBatchSplit) {
    if (!has_cuda_ || num_gpus_ < 2) {
        GTEST_SKIP() << "Multiple GPUs not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0, 1}, 0);

    // Create input with batch size 5 (splits as 3 + 2)
    Tensor input_tensor({5, 10}, DType::Float32, Device::cuda(0));
    input_tensor.fill_(1.0f);
    Variable input(input_tensor, true);

    // Forward pass should handle uneven split
    Variable output;
    EXPECT_NO_THROW({
        output = parallel->forward(input);
    });

    // Check output shape
    auto output_shape = output.shape();
    EXPECT_EQ(output_shape[0], 5);
    EXPECT_EQ(output_shape[1], 5);
}

TEST_F(DataParallelTest, DifferentBatchDimensions) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);

    // Test with different batch dimensions
    for (int dim : {0, 1}) {
        auto parallel = std::make_shared<DataParallel>(
            module, std::vector<int>{0}, 0, dim
        );

        EXPECT_EQ(parallel->batch_dim(), dim);
    }
}

// Integration test with full training loop
TEST_F(DataParallelTest, FullTrainingLoopSingleGPU) {
    if (!has_cuda_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<SimpleModule>(10, 20, 5);
    auto parallel = std::make_shared<DataParallel>(module, std::vector<int>{0}, 0);

    // Simulate training loop
    for (int epoch = 0; epoch < 3; ++epoch) {
        // Create batch
        Tensor input_tensor({4, 10}, DType::Float32, Device::cuda(0));
        input_tensor.fill_(1.0f);
        Variable input(input_tensor, true);

        Tensor target_tensor({4, 5}, DType::Float32, Device::cuda(0));
        target_tensor.fill_(0.5f);
        Variable target(target_tensor, false);

        // Forward
        auto output = parallel->forward(input);

        // Compute loss (simple MSE)
        auto diff = output.tensor() - target.tensor();
        auto loss_tensor = (diff * diff).sum();
        Variable loss(loss_tensor, true);

        // Backward
        loss.backward();

        // Synchronize gradients
        parallel->synchronize_gradients();

        // Verify gradients exist
        auto params = parallel->parameters();
        for (auto* param : params) {
            EXPECT_TRUE(param->has_grad());
        }

        // Zero gradients for next iteration
        parallel->zero_grad();
    }
}

// CPU fallback test
TEST_F(DataParallelTest, CPUFallbackThrows) {
    auto module = std::make_shared<SimpleModule>(10, 20, 5);

    // Should throw on CPU-only builds
#ifndef TENZOR_USE_CUDA
    EXPECT_THROW({
        auto parallel = DataParallel(module);
    }, std::runtime_error);
#else
    // If CUDA is available but not initialized properly, should still handle
    GTEST_SKIP() << "CUDA available, skipping CPU fallback test";
#endif
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
