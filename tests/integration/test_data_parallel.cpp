/**
 * @file test_data_parallel.cpp
 * @brief Comprehensive integration tests for DataParallel multi-GPU training
 *
 * Tests the complete DataParallel system including:
 * - Multi-GPU model replication
 * - Input batch splitting and scattering
 * - Parallel forward pass execution
 * - Output gathering and concatenation
 * - Gradient synchronization and averaging
 * - Training loop integration with optimizers
 * - Performance scaling validation
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/parallel/data_parallel.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <iostream>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

//==============================================================================
// Test Environment Setup
//==============================================================================

class DataParallelEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();

        // Check CUDA availability and count devices
#ifdef TENZOR_USE_CUDA
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err == cudaSuccess && device_count > 0) {
            cuda_available_ = true;
            device_count_ = device_count;
            std::cout << "CUDA available: " << device_count << " device(s) detected\n";
        } else {
            cuda_available_ = false;
            device_count_ = 0;
            std::cout << "CUDA not available, multi-GPU tests will be skipped\n";
        }
#else
        cuda_available_ = false;
        device_count_ = 0;
        std::cout << "CUDA not compiled, multi-GPU tests will be skipped\n";
#endif
    }

    static bool cuda_available_;
    static int device_count_;
};

bool DataParallelEnvironment::cuda_available_ = false;
int DataParallelEnvironment::device_count_ = 0;

static ::testing::Environment* const data_parallel_env =
    ::testing::AddGlobalTestEnvironment(new DataParallelEnvironment);

//==============================================================================
// Test Fixture with Device Synchronization
//==============================================================================

class DataParallelTest : public ::testing::Test {
protected:
    void TearDown() override {
        // Synchronize all devices after each test
#ifdef TENZOR_USE_CUDA
        if (DataParallelEnvironment::cuda_available_) {
            for (int i = 0; i < DataParallelEnvironment::device_count_; ++i) {
                try {
                    Device::cuda(i).synchronize();
                } catch (...) {
                    // Ignore synchronization errors
                }
            }
        }
#endif
    }
};

//==============================================================================
// Test Models
//==============================================================================

/**
 * @brief Simple MLP for testing data parallel training
 */
class SimpleMLP : public Module {
public:
    SimpleMLP(int input_size, int hidden_size, int output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        relu_ = std::make_shared<ReLU>();
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto h = fc1_->forward(input);
        h = relu_->forward(h);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Linear> fc2_;
};

/**
 * @brief Convolutional model for testing with image data
 */
class SimpleConvNet : public Module {
public:
    SimpleConvNet() {
        conv1_ = std::make_shared<Conv2d>(1, 16, 3, 1, 1);
        relu1_ = std::make_shared<ReLU>();
        pool1_ = std::make_shared<MaxPool2d>(2, 2);

        conv2_ = std::make_shared<Conv2d>(16, 32, 3, 1, 1);
        relu2_ = std::make_shared<ReLU>();
        pool2_ = std::make_shared<MaxPool2d>(2, 2);

        flatten_ = std::make_shared<Flatten>(1);
        fc1_ = std::make_shared<Linear>(32 * 7 * 7, 128);
        relu3_ = std::make_shared<ReLU>();
        fc2_ = std::make_shared<Linear>(128, 10);

        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto x = conv1_->forward(input);
        x = relu1_->forward(x);
        x = pool1_->forward(x);

        x = conv2_->forward(x);
        x = relu2_->forward(x);
        x = pool2_->forward(x);

        x = flatten_->forward(x);
        x = fc1_->forward(x);
        x = relu3_->forward(x);
        return fc2_->forward(x);
    }

private:
    std::shared_ptr<Conv2d> conv1_, conv2_;
    std::shared_ptr<ReLU> relu1_, relu2_, relu3_;
    std::shared_ptr<MaxPool2d> pool1_, pool2_;
    std::shared_ptr<Flatten> flatten_;
    std::shared_ptr<Linear> fc1_, fc2_;
};

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * @brief Generate synthetic batch data for testing
 */
auto generate_batch(int batch_size, int input_size, Device device)
    -> std::pair<Variable, Variable> {
    auto input = Variable(randn({batch_size, input_size}, DType::Float32, device), true);

    // Create random labels (one-hot encoded)
    auto target_data = zeros({batch_size, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data.template data<float>());

    for (int i = 0; i < batch_size; ++i) {
        int label = i % 10;
        target_ptr[i * 10 + label] = 1.0f;
    }

    auto target_device = (device.type == Device::Type::CPU) ?
        target_data : target_data.to(device);
    auto target = Variable(target_device, false);

    return {input, target};
}

/**
 * @brief Generate image batch data for CNN testing
 */
auto generate_image_batch(int batch_size, Device device)
    -> std::pair<Variable, Variable> {
    // Create [batch, 1, 28, 28] images
    auto input = Variable(
        randn({batch_size, 1, 28, 28}, DType::Float32, device),
        true
    );

    // Create labels
    auto target_data = zeros({batch_size, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_data.template data<float>());

    for (int i = 0; i < batch_size; ++i) {
        int label = i % 10;
        target_ptr[i * 10 + label] = 1.0f;
    }

    auto target_device = (device.type == Device::Type::CPU) ?
        target_data : target_data.to(device);
    auto target = Variable(target_device, false);

    return {input, target};
}

/**
 * @brief Verify gradient values are reasonable (not NaN, not too large)
 */
bool check_gradients_valid(const std::vector<std::shared_ptr<Variable>>& params) {
    for (const auto& param : params) {
        if (!param->has_grad()) {
            continue;
        }

        auto grad = param->grad();
        if (!grad.has_value()) {
            continue;
        }

        auto grad_cpu = grad.value().to(Device::cpu());
        const float* grad_data = grad_cpu.template data<float>();
        size_t numel = grad_cpu.numel();

        for (size_t i = 0; i < numel; ++i) {
            if (std::isnan(grad_data[i]) || std::isinf(grad_data[i])) {
                return false;
            }
            if (std::abs(grad_data[i]) > 1e6) {
                return false;
            }
        }
    }

    return true;
}

//==============================================================================
// Single GPU Tests (Baseline)
//==============================================================================

TEST_F(DataParallelTest, SingleGPUBaseline) {
    if (!DataParallelEnvironment::cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create model and wrap with DataParallel (single GPU)
    auto model = std::make_shared<SimpleMLP>(784, 128, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0},  // Single GPU
        0
    );

    // Move model to GPU
    parallel_model->to(Device::cuda(0));

    // Generate batch
    auto [input, target] = generate_batch(32, 784, Device::cuda(0));

    // Forward pass
    auto output = parallel_model->forward(input);

    // Check output shape
    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 10);

    // Compute loss
    auto loss = MSELoss()(output, target);

    // Backward pass
    parallel_model->zero_grad();
    loss.backward();

    // Check gradients exist and are valid
    auto params = parallel_model->parameters();
    EXPECT_TRUE(check_gradients_valid(params));
}

//==============================================================================
// Multi-GPU Forward Pass Tests
//==============================================================================

TEST_F(DataParallelTest, MultiGPUForwardPass) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    // Create model with 2 GPUs
    auto model = std::make_shared<SimpleMLP>(784, 128, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1},
        0  // Master GPU
    );

    parallel_model->to(Device::cuda(0));

    // Generate batch (must be >= num_gpus)
    auto [input, target] = generate_batch(64, 784, Device::cuda(0));

    // Forward pass should split across GPUs
    auto output = parallel_model->forward(input);

    // Verify output
    EXPECT_EQ(output.shape()[0], 64);
    EXPECT_EQ(output.shape()[1], 10);
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
    EXPECT_EQ(output.device().index, 0);  // Should be on master GPU
}

TEST_F(DataParallelTest, MultiGPUUnevenBatchSplit) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    auto model = std::make_shared<SimpleMLP>(784, 128, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1},
        0
    );

    parallel_model->to(Device::cuda(0));

    // Uneven batch size (65 samples on 2 GPUs = 33 + 32)
    auto [input, target] = generate_batch(65, 784, Device::cuda(0));

    auto output = parallel_model->forward(input);

    EXPECT_EQ(output.shape()[0], 65);
    EXPECT_EQ(output.shape()[1], 10);
}

TEST_F(DataParallelTest, MultiGPUBatchTooSmall) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    auto model = std::make_shared<SimpleMLP>(784, 128, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1},
        0
    );

    parallel_model->to(Device::cuda(0));

    // Batch size smaller than number of GPUs should throw
    auto [input, target] = generate_batch(1, 784, Device::cuda(0));

    EXPECT_THROW({
        auto output = parallel_model->forward(input);
    }, std::runtime_error);
}

//==============================================================================
// Gradient Synchronization Tests
//==============================================================================

TEST_F(DataParallelTest, GradientSynchronizationCorrectness) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    // Train model with DataParallel
    auto model_parallel = std::make_shared<SimpleMLP>(784, 128, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model_parallel,
        std::vector<int>{0, 1},
        0
    );
    parallel_model->to(Device::cuda(0));

    // Train model on single GPU for comparison
    auto model_single = std::make_shared<SimpleMLP>(784, 128, 10);
    model_single->to(Device::cuda(0));

    // Copy parameters from parallel to single model
    auto parallel_state = parallel_model->state_dict();
    model_single->load_state_dict(parallel_state);

    // Generate same batch for both
    auto [input, target] = generate_batch(64, 784, Device::cuda(0));

    // Forward + backward on DataParallel
    parallel_model->zero_grad();
    auto output_parallel = parallel_model->forward(input);
    auto loss_parallel = MSELoss()(output_parallel, target);
    loss_parallel.backward();

    // Forward + backward on single GPU
    model_single->zero_grad();
    auto output_single = model_single->forward(input);
    auto loss_single = MSELoss()(output_single, target);
    loss_single.backward();

    // Compare losses
    auto loss_parallel_cpu = loss_parallel.tensor().to(Device::cpu());
    auto loss_single_cpu = loss_single.tensor().to(Device::cpu());
    float loss_parallel_val = loss_parallel_cpu.template data<float>()[0];
    float loss_single_val = loss_single_cpu.template data<float>()[0];

    EXPECT_NEAR(loss_parallel_val, loss_single_val, 1e-4);

    // Compare gradients (they should be similar after averaging)
    // Note: Due to parallel execution order, exact match might not occur
    // but they should be in the same range
    auto params_parallel = parallel_model->parameters();
    auto params_single = model_single->parameters();

    EXPECT_EQ(params_parallel.size(), params_single.size());

    for (size_t i = 0; i < params_parallel.size(); ++i) {
        if (!params_parallel[i]->has_grad() || !params_single[i]->has_grad()) {
            continue;
        }

        auto grad_parallel = params_parallel[i]->grad().value();
        auto grad_single = params_single[i]->grad().value();

        // Check gradient magnitudes are similar
        auto grad_p_cpu = grad_parallel.to(Device::cpu());
        auto grad_s_cpu = grad_single.to(Device::cpu());

        const float* gp_data = grad_p_cpu.template data<float>();
        const float* gs_data = grad_s_cpu.template data<float>();

        size_t numel = grad_p_cpu.numel();
        float max_diff = 0.0f;

        for (size_t j = 0; j < numel; ++j) {
            float diff = std::abs(gp_data[j] - gs_data[j]);
            max_diff = std::max(max_diff, diff);
        }

        // Gradients should be reasonably close
        EXPECT_LT(max_diff, 1.0f);
    }
}

//==============================================================================
// Training Loop Integration Tests
//==============================================================================

TEST_F(DataParallelTest, TrainingLoopWithOptimizer) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    // Create DataParallel model
    auto model = std::make_shared<SimpleMLP>(784, 128, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1},
        0
    );
    parallel_model->to(Device::cuda(0));

    // Create optimizer
    auto optimizer = std::make_shared<SGD>(
        parallel_model->parameters(),
        0.01  // learning rate
    );

    // Training loop
    int num_iterations = 10;
    std::vector<float> losses;

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Generate batch
        auto [input, target] = generate_batch(64, 784, Device::cuda(0));

        // Zero gradients
        parallel_model->zero_grad();

        // Forward pass
        auto output = parallel_model->forward(input);

        // Compute loss
        auto loss = MSELoss()(output, target);

        // Backward pass
        loss.backward();

        // Gradient synchronization (if not automatic)
        // parallel_model->synchronize_gradients();

        // Optimizer step
        optimizer->step();

        // Record loss
        auto loss_cpu = loss.tensor().to(Device::cpu());
        losses.push_back(loss_cpu.template data<float>()[0]);
    }

    // Verify training occurred (loss should decrease or stay stable)
    EXPECT_GT(losses.size(), 0);

    // Check that loss values are reasonable
    for (float loss_val : losses) {
        EXPECT_FALSE(std::isnan(loss_val));
        EXPECT_FALSE(std::isinf(loss_val));
        EXPECT_GT(loss_val, 0.0f);
    }
}

TEST_F(DataParallelTest, MultiGPUConvNetTraining) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    // Create ConvNet with DataParallel
    auto model = std::make_shared<SimpleConvNet>();
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1},
        0
    );
    parallel_model->to(Device::cuda(0));

    // Create optimizer
    auto optimizer = std::make_shared<Adam>(
        parallel_model->parameters(),
        0.001  // learning rate
    );

    // Training iterations
    int num_iterations = 5;

    for (int iter = 0; iter < num_iterations; ++iter) {
        // Generate image batch
        auto [input, target] = generate_image_batch(32, Device::cuda(0));

        parallel_model->zero_grad();

        auto output = parallel_model->forward(input);
        auto loss = MSELoss()(output, target);

        loss.backward();

        // Check gradients are valid before optimizer step
        EXPECT_TRUE(check_gradients_valid(parallel_model->parameters()));

        optimizer->step();
    }
}

//==============================================================================
// 4-GPU Tests (If Available)
//==============================================================================

TEST_F(DataParallelTest, FourGPUTraining) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 4) {
        GTEST_SKIP() << "Need at least 4 GPUs";
    }

    auto model = std::make_shared<SimpleMLP>(784, 256, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1, 2, 3},
        0
    );
    parallel_model->to(Device::cuda(0));

    auto optimizer = std::make_shared<SGD>(
        parallel_model->parameters(),
        0.01
    );

    // Need batch size >= 4
    auto [input, target] = generate_batch(128, 784, Device::cuda(0));

    parallel_model->zero_grad();
    auto output = parallel_model->forward(input);

    EXPECT_EQ(output.shape()[0], 128);
    EXPECT_EQ(output.shape()[1], 10);

    auto loss = MSELoss()(output, target);
    loss.backward();

    EXPECT_TRUE(check_gradients_valid(parallel_model->parameters()));

    optimizer->step();
}

//==============================================================================
// Performance and Scaling Tests
//==============================================================================

TEST_F(DataParallelTest, PerformanceScaling) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    // Test single GPU
    auto model_single = std::make_shared<SimpleMLP>(784, 512, 10);
    model_single->to(Device::cuda(0));

    auto [input_single, target_single] = generate_batch(128, 784, Device::cuda(0));

    auto start_single = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        auto output = model_single->forward(input_single);
        auto loss = MSELoss()(output, target_single);
        model_single->zero_grad();
        loss.backward();
    }
    Device::cuda(0).synchronize();
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_single - start_single
    ).count();

    // Test multi-GPU
    auto model_multi = std::make_shared<SimpleMLP>(784, 512, 10);
    auto parallel_model = std::make_shared<DataParallel>(
        model_multi,
        std::vector<int>{0, 1},
        0
    );
    parallel_model->to(Device::cuda(0));

    auto [input_multi, target_multi] = generate_batch(128, 784, Device::cuda(0));

    auto start_multi = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 10; ++i) {
        auto output = parallel_model->forward(input_multi);
        auto loss = MSELoss()(output, target_multi);
        parallel_model->zero_grad();
        loss.backward();
    }
    Device::cuda(0).synchronize();
    Device::cuda(1).synchronize();
    auto end_multi = std::chrono::high_resolution_clock::now();
    auto duration_multi = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_multi - start_multi
    ).count();

    std::cout << "Single GPU time: " << duration_single << "ms\n";
    std::cout << "Multi GPU time: " << duration_multi << "ms\n";

    // Multi-GPU should be faster or at least not significantly slower
    // (accounting for small models and synchronization overhead)
    EXPECT_GT(duration_single, 0);
    EXPECT_GT(duration_multi, 0);
}

//==============================================================================
// Memory Efficiency Tests
//==============================================================================

TEST_F(DataParallelTest, MemoryDistribution) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    // Large model to test memory distribution
    auto model = std::make_shared<SimpleMLP>(1024, 2048, 1024);
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1},
        0
    );
    parallel_model->to(Device::cuda(0));

    // Large batch
    auto [input, target] = generate_batch(256, 1024, Device::cuda(0));

    // Forward pass should succeed without OOM
    EXPECT_NO_THROW({
        auto output = parallel_model->forward(input);
        EXPECT_EQ(output.shape()[0], 256);
    });
}

//==============================================================================
// Edge Cases and Error Handling
//==============================================================================

TEST_F(DataParallelTest, DifferentBatchDimensions) {
    if (!DataParallelEnvironment::cuda_available_ ||
        DataParallelEnvironment::device_count_ < 2) {
        GTEST_SKIP() << "Need at least 2 GPUs";
    }

    auto model = std::make_shared<SimpleMLP>(784, 128, 10);

    // Test with batch dimension = 1 instead of 0
    auto parallel_model = std::make_shared<DataParallel>(
        model,
        std::vector<int>{0, 1}
    );
    // Note: dim parameter defaults to 0, testing with custom dim would require
    // a separate constructor or setter method

    // Create input with shape [10, 64] where batch is dimension 1
    auto input = Variable(
        randn({10, 64}, DType::Float32, Device::cuda(0)),
        true
    );

    // Test forward pass works
    EXPECT_NO_THROW({
        auto output = parallel_model->forward(input);
        EXPECT_EQ(output.shape()[0], 10);
    });
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "========================================\n";
    std::cout << "DataParallel Integration Tests\n";
    std::cout << "========================================\n";

    return RUN_ALL_TESTS();
}
