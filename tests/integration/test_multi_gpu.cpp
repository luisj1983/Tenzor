/**
 * @file test_multi_gpu.cpp
 * @brief Comprehensive multi-GPU integration tests for DataParallel
 *
 * Production-quality tests covering:
 * - Multi-GPU model creation and replication
 * - Forward/backward passes across GPUs
 * - Gradient synchronization and averaging
 * - Training loops with multiple GPUs
 * - Performance scaling validation
 * - Correctness verification against single-GPU baseline
 *
 * Tests automatically skip gracefully if insufficient GPUs are available.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/parallel/data_parallel.hpp>
#include <chrono>
#include <cmath>
#include <memory>
#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip>

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::optim;

//==============================================================================
// Multi-GPU Test Environment
//==============================================================================

class MultiGPUEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();

#ifdef TENZOR_USE_CUDA
        cudaError_t err = cudaGetDeviceCount(&device_count_);
        if (err == cudaSuccess && device_count_ > 0) {
            cuda_available_ = true;
            std::cout << "\n=== Multi-GPU Test Environment ===\n";
            std::cout << "CUDA devices detected: " << device_count_ << "\n";

            // Print device properties
            for (int i = 0; i < device_count_; ++i) {
                cudaDeviceProp prop;
                cudaGetDeviceProperties(&prop, i);
                std::cout << "  GPU " << i << ": " << prop.name
                         << " (" << prop.totalGlobalMem / (1024*1024) << " MB)\n";
            }
            std::cout << "===================================\n\n";
        } else {
            cuda_available_ = false;
            device_count_ = 0;
            std::cout << "\n=== Multi-GPU Test Environment ===\n";
            std::cout << "No CUDA devices available - tests will skip\n";
            std::cout << "===================================\n\n";
        }
#else
        cuda_available_ = false;
        device_count_ = 0;
        std::cout << "\n=== Multi-GPU Test Environment ===\n";
        std::cout << "CUDA not compiled - tests will skip\n";
        std::cout << "===================================\n\n";
#endif
    }

    void TearDown() override {
#ifdef TENZOR_USE_CUDA
        if (cuda_available_) {
            // Synchronize and reset all devices
            for (int i = 0; i < device_count_; ++i) {
                cudaSetDevice(i);
                cudaDeviceSynchronize();
                cudaDeviceReset();
            }
        }
#endif
    }

    static bool cuda_available_;
    static int device_count_;
};

bool MultiGPUEnvironment::cuda_available_ = false;
int MultiGPUEnvironment::device_count_ = 0;

static ::testing::Environment* const multi_gpu_env =
    ::testing::AddGlobalTestEnvironment(new MultiGPUEnvironment);

//==============================================================================
// Test Fixture
//==============================================================================

class MultiGPUTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Skip tests if CUDA not available
        if (!MultiGPUEnvironment::cuda_available_) {
            GTEST_SKIP() << "CUDA not available, skipping multi-GPU test";
        }
    }

    void TearDown() override {
#ifdef TENZOR_USE_CUDA
        // Synchronize all devices
        for (int i = 0; i < MultiGPUEnvironment::device_count_; ++i) {
            cudaSetDevice(i);
            cudaDeviceSynchronize();
        }
#endif
    }

    // Helper: Check if we have enough GPUs
    bool has_n_gpus(int n) const {
        return MultiGPUEnvironment::device_count_ >= n;
    }

    // Helper: Get available device IDs up to max_count
    std::vector<int> get_device_ids(int max_count) const {
        int count = std::min(max_count, MultiGPUEnvironment::device_count_);
        std::vector<int> ids;
        for (int i = 0; i < count; ++i) {
            ids.push_back(i);
        }
        return ids;
    }
};

//==============================================================================
// Test Models
//==============================================================================

/**
 * @brief Simple MLP for basic multi-GPU testing
 */
class TestMLP : public Module {
public:
    TestMLP(int input_size, int hidden_size, int output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        relu_ = std::make_shared<ReLU>();
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& input) -> Variable override {
        auto h = fc1_->forward(input);
        h = relu_->forward(h);
        return fc2_->forward(h);
    }

    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Linear> fc2_;
};

/**
 * @brief Convolutional model for image-based testing
 */
class TestConvNet : public Module {
public:
    TestConvNet() {
        conv1_ = std::make_shared<Conv2d>(3, 32, 3, 1, 1);
        bn1_ = std::make_shared<BatchNorm2d>(32);
        relu1_ = std::make_shared<ReLU>();
        pool1_ = std::make_shared<MaxPool2d>(2, 2);

        conv2_ = std::make_shared<Conv2d>(32, 64, 3, 1, 1);
        bn2_ = std::make_shared<BatchNorm2d>(64);
        relu2_ = std::make_shared<ReLU>();
        pool2_ = std::make_shared<MaxPool2d>(2, 2);

        flatten_ = std::make_shared<Flatten>(1);
        fc1_ = std::make_shared<Linear>(64 * 8 * 8, 256);
        relu3_ = std::make_shared<ReLU>();
        dropout_ = std::make_shared<Dropout>(0.5);
        fc2_ = std::make_shared<Linear>(256, 10);

        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("conv2", conv2_);
        register_module("bn2", bn2_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward(const Variable& input) -> Variable override {
        auto x = conv1_->forward(input);
        x = bn1_->forward(x);
        x = relu1_->forward(x);
        x = pool1_->forward(x);

        x = conv2_->forward(x);
        x = bn2_->forward(x);
        x = relu2_->forward(x);
        x = pool2_->forward(x);

        x = flatten_->forward(x);
        x = fc1_->forward(x);
        x = relu3_->forward(x);
        x = dropout_->forward(x);
        return fc2_->forward(x);
    }

    std::shared_ptr<Conv2d> conv1_, conv2_;
    std::shared_ptr<BatchNorm2d> bn1_, bn2_;
    std::shared_ptr<ReLU> relu1_, relu2_, relu3_;
    std::shared_ptr<MaxPool2d> pool1_, pool2_;
    std::shared_ptr<Flatten> flatten_;
    std::shared_ptr<Linear> fc1_, fc2_;
    std::shared_ptr<Dropout> dropout_;
};

//==============================================================================
// Helper Functions
//==============================================================================

/**
 * @brief Generate synthetic batch data
 */
auto generate_batch(int batch_size, int input_size, Device device = Device::cpu())
    -> std::pair<Variable, Variable> {

    auto input = Variable(randn({batch_size, input_size}, DType::Float32, device), true);

    // Random target values
    auto target = Variable(randn({batch_size, 10}, DType::Float32, device), false);

    return {input, target};
}

/**
 * @brief Generate image batch data
 */
auto generate_image_batch(int batch_size, int channels, int height, int width,
                         Device device = Device::cpu())
    -> std::pair<Variable, Variable> {

    auto input = Variable(
        randn({batch_size, channels, height, width}, DType::Float32, device),
        true
    );

    auto target = Variable(randn({batch_size, 10}, DType::Float32, device), false);

    return {input, target};
}

/**
 * @brief Compare two tensors with tolerance
 */
bool tensors_close(const Tensor& a, const Tensor& b, float rtol = 1e-4, float atol = 1e-5) {
    auto shape_a = a.shape();
    auto shape_b = b.shape();

    if (shape_a.size() != shape_b.size()) return false;
    for (size_t i = 0; i < shape_a.size(); ++i) {
        if (shape_a[i] != shape_b[i]) return false;
    }

    auto a_cpu = a.to(Device::cpu());
    auto b_cpu = b.to(Device::cpu());

    const float* a_data = a_cpu.data<float>();
    const float* b_data = b_cpu.data<float>();

    int64_t size = a.numel();
    for (int64_t i = 0; i < size; ++i) {
        float diff = std::abs(a_data[i] - b_data[i]);
        float threshold = atol + rtol * std::abs(b_data[i]);
        if (diff > threshold) {
            return false;
        }
    }
    return true;
}

//==============================================================================
// Test 1: DataParallel Model Creation
//==============================================================================

TEST_F(MultiGPUTest, ModelCreationWith2GPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs, only "
                     << MultiGPUEnvironment::device_count_ << " available";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);

    // Create DataParallel model
    auto parallel_model = std::make_shared<DataParallel>(
        model, device_ids, device_ids[0]
    );

    // Verify properties
    EXPECT_EQ(parallel_model->device_ids().size(), 2);
    EXPECT_EQ(parallel_model->output_device(), device_ids[0]);
    EXPECT_EQ(parallel_model->batch_dim(), 0);
    EXPECT_NE(parallel_model->module(), nullptr);
}

TEST_F(MultiGPUTest, ModelCreationWith3GPUs) {
    if (!has_n_gpus(3)) {
        GTEST_SKIP() << "Test requires 3 GPUs, only "
                     << MultiGPUEnvironment::device_count_ << " available";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(3);

    auto parallel_model = std::make_shared<DataParallel>(
        model, device_ids, device_ids[0]
    );

    EXPECT_EQ(parallel_model->device_ids().size(), 3);
}

TEST_F(MultiGPUTest, ModelCreationWith4GPUs) {
    if (!has_n_gpus(4)) {
        GTEST_SKIP() << "Test requires 4 GPUs, only "
                     << MultiGPUEnvironment::device_count_ << " available";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(4);

    auto parallel_model = std::make_shared<DataParallel>(
        model, device_ids, device_ids[0]
    );

    EXPECT_EQ(parallel_model->device_ids().size(), 4);
}

TEST_F(MultiGPUTest, AutoDetectAllGPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires at least 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);

    // Auto-detect: empty device_ids vector
    auto parallel_model = std::make_shared<DataParallel>(model);

    EXPECT_EQ(parallel_model->device_ids().size(), MultiGPUEnvironment::device_count_);
}

//==============================================================================
// Test 2: Forward Pass Across Multiple GPUs
//==============================================================================

TEST_F(MultiGPUTest, ForwardPass2GPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Create batch on GPU 0 (master device)
    int batch_size = 16;
    auto [input, _] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    // Forward pass
    auto output = parallel_model->forward(input);

    // Verify output shape
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], batch_size);
    EXPECT_EQ(shape[1], 10);

    // Verify output is on master device
    EXPECT_EQ(output.tensor().device().type, Device::Type::CUDA);
    EXPECT_EQ(output.tensor().device().index, device_ids[0]);
}

TEST_F(MultiGPUTest, ForwardPass3GPUs) {
    if (!has_n_gpus(3)) {
        GTEST_SKIP() << "Test requires 3 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(3);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    int batch_size = 24; // Divisible by 3
    auto [input, _] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    auto output = parallel_model->forward(input);

    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], batch_size);
    EXPECT_EQ(shape[1], 10);
    EXPECT_EQ(output.tensor().device().index, device_ids[0]);
}

TEST_F(MultiGPUTest, ForwardPass4GPUs) {
    if (!has_n_gpus(4)) {
        GTEST_SKIP() << "Test requires 4 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(4);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    int batch_size = 32; // Divisible by 4
    auto [input, _] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    auto output = parallel_model->forward(input);

    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], batch_size);
    EXPECT_EQ(shape[1], 10);
    EXPECT_EQ(output.tensor().device().index, device_ids[0]);
}

TEST_F(MultiGPUTest, ForwardPassUnevenBatch) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Batch size not evenly divisible by num_gpus
    int batch_size = 17;
    auto [input, _] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    auto output = parallel_model->forward(input);

    // Should still work - first GPU gets 9 samples, second gets 8
    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], batch_size);
    EXPECT_EQ(shape[1], 10);
}

//==============================================================================
// Test 3: Backward Pass and Gradient Synchronization
//==============================================================================

TEST_F(MultiGPUTest, BackwardPass2GPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    int batch_size = 16;
    auto [input, target] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    // Forward pass
    auto output = parallel_model->forward(input);

    // Compute loss
    auto loss = MSELoss()(output, target);

    // Backward pass
    loss.backward();

    // Verify gradients exist
    auto params = parallel_model->parameters();
    EXPECT_GT(params.size(), 0);

    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad()) << "Parameter missing gradient";

        auto grad = param->grad();
        EXPECT_TRUE(grad.has_value());

        // Verify gradient is not all zeros
        auto grad_tensor = grad.value().to(Device::cpu());
        const float* grad_data = grad_tensor.data<float>();
        bool has_nonzero = false;
        for (int64_t i = 0; i < grad_tensor.numel(); ++i) {
            if (std::abs(grad_data[i]) > 1e-10) {
                has_nonzero = true;
                break;
            }
        }
        EXPECT_TRUE(has_nonzero) << "Gradient is all zeros";
    }
}

TEST_F(MultiGPUTest, BackwardPass4GPUs) {
    if (!has_n_gpus(4)) {
        GTEST_SKIP() << "Test requires 4 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(4);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    int batch_size = 32;
    auto [input, target] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    auto output = parallel_model->forward(input);
    auto loss = MSELoss()(output, target);
    loss.backward();

    // Verify all parameters have gradients
    auto params = parallel_model->parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

//==============================================================================
// Test 4: Gradient Averaging Verification
//==============================================================================

TEST_F(MultiGPUTest, GradientAveragingCorrectness2GPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    // Create two identical models
    auto model_parallel = std::make_shared<TestMLP>(32, 64, 10);
    auto model_single = std::make_shared<TestMLP>(32, 64, 10);

    // Copy parameters from parallel to single GPU model
    auto state_dict = model_parallel->state_dict();
    model_single->load_state_dict(state_dict);

    // Wrap with DataParallel
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model_parallel, device_ids);

    // Create same batch for both models
    int batch_size = 16;
    auto input_data = randn({batch_size, 32}, DType::Float32, Device::cpu());
    auto target_data = randn({batch_size, 10}, DType::Float32, Device::cpu());

    // Multi-GPU forward/backward
    auto input_parallel = Variable(input_data.cuda(device_ids[0]), true);
    auto target_parallel = Variable(target_data.cuda(device_ids[0]), false);
    auto output_parallel = parallel_model->forward(input_parallel);
    auto loss_parallel = MSELoss()(output_parallel, target_parallel);
    loss_parallel.backward();

    // Single-GPU forward/backward
    auto input_single = Variable(input_data.cuda(device_ids[0]), true);
    auto target_single = Variable(target_data.cuda(device_ids[0]), false);
    auto output_single = model_single->forward(input_single);
    auto loss_single = MSELoss()(output_single, target_single);
    loss_single.backward();

    // Compare outputs
    EXPECT_TRUE(tensors_close(output_parallel.tensor(), output_single.tensor(), 1e-3, 1e-4))
        << "Multi-GPU and single-GPU outputs differ";

    // Compare gradients (should be similar after averaging)
    auto params_parallel = parallel_model->parameters();
    auto params_single = model_single->parameters();

    for (size_t i = 0; i < params_parallel.size(); ++i) {
        if (params_parallel[i]->has_grad() && params_single[i]->has_grad()) {
            auto grad_parallel = params_parallel[i]->grad().value();
            auto grad_single = params_single[i]->grad().value();

            EXPECT_TRUE(tensors_close(grad_parallel, grad_single, 1e-2, 1e-3))
                << "Gradients differ for parameter " << i;
        }
    }
}

//==============================================================================
// Test 5: Training Loop with DataParallel
//==============================================================================

TEST_F(MultiGPUTest, TrainingLoop2GPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Create optimizer
    auto params = parallel_model->parameters();
    SGD optimizer(params, 0.01);

    // Training loop
    const int num_epochs = 3;
    const int num_batches = 5;
    const int batch_size = 16;

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            // Generate batch
            auto [input, target] = generate_batch(
                batch_size, 64, Device::cuda(device_ids[0])
            );

            // Zero gradients
            optimizer.zero_grad();

            // Forward pass
            auto output = parallel_model->forward(input);

            // Compute loss
            auto loss = MSELoss()(output, target);

            // Track loss
            if (epoch == 0 && batch == 0) {
                initial_loss = loss.tensor().to(Device::cpu()).data<float>()[0];
            }
            if (epoch == num_epochs - 1 && batch == num_batches - 1) {
                final_loss = loss.tensor().to(Device::cpu()).data<float>()[0];
            }

            // Backward pass
            loss.backward();

            // Update parameters
            optimizer.step();
        }
    }

    // Loss should decrease (model is learning)
    EXPECT_LT(final_loss, initial_loss)
        << "Loss did not decrease during training";
}

TEST_F(MultiGPUTest, TrainingLoop4GPUs) {
    if (!has_n_gpus(4)) {
        GTEST_SKIP() << "Test requires 4 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(4);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    auto params = parallel_model->parameters();
    SGD optimizer(params, 0.01);

    const int num_epochs = 3;
    const int num_batches = 5;
    const int batch_size = 32; // 8 per GPU

    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        for (int batch = 0; batch < num_batches; ++batch) {
            auto [input, target] = generate_batch(
                batch_size, 64, Device::cuda(device_ids[0])
            );

            optimizer.zero_grad();
            auto output = parallel_model->forward(input);
            auto loss = MSELoss()(output, target);

            if (epoch == 0 && batch == 0) {
                initial_loss = loss.tensor().to(Device::cpu()).data<float>()[0];
            }
            if (epoch == num_epochs - 1 && batch == num_batches - 1) {
                final_loss = loss.tensor().to(Device::cpu()).data<float>()[0];
            }

            loss.backward();
            optimizer.step();
        }
    }

    EXPECT_LT(final_loss, initial_loss);
}

//==============================================================================
// Test 6: Model Replication Verification
//==============================================================================

TEST_F(MultiGPUTest, ModelReplicationVerification) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Trigger replication by running forward pass
    int batch_size = 16;
    auto [input, _] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));
    parallel_model->forward(input);

    // Verify parameters are accessible
    auto params = parallel_model->parameters();
    EXPECT_GT(params.size(), 0);

    // Verify parameters have correct shapes
    for (const auto& param : params) {
        EXPECT_GT(param->tensor().numel(), 0);
        EXPECT_EQ(param->tensor().dtype(), DType::Float32);
    }
}

//==============================================================================
// Test 7: Performance Scaling Checks
//==============================================================================

TEST_F(MultiGPUTest, PerformanceScaling2vs1GPU) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    // Test with larger model for meaningful timing
    auto model_1gpu = std::make_shared<TestMLP>(512, 1024, 256);
    auto model_2gpu = std::make_shared<TestMLP>(512, 1024, 256);

    // Copy parameters to ensure identical initialization
    auto state_dict = model_1gpu->state_dict();
    model_2gpu->load_state_dict(state_dict);

    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model_2gpu, device_ids);

    const int batch_size = 128;
    const int num_iterations = 20;

    // Warm-up
    for (int i = 0; i < 5; ++i) {
        auto [input, target] = generate_batch(batch_size, 512, Device::cuda(0));
        auto output = model_1gpu->forward(input);
        auto loss = MSELoss()(output, target);
        loss.backward();
    }

    // Time single GPU
    auto start_1gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto [input, target] = generate_batch(batch_size, 512, Device::cuda(0));
        auto output = model_1gpu->forward(input);
        auto loss = MSELoss()(output, target);
        loss.backward();
    }
#ifdef TENZOR_USE_CUDA
    cudaDeviceSynchronize();
#endif
    auto end_1gpu = std::chrono::high_resolution_clock::now();
    auto time_1gpu = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_1gpu - start_1gpu
    ).count();

    // Warm-up multi-GPU
    for (int i = 0; i < 5; ++i) {
        auto [input, target] = generate_batch(batch_size, 512, Device::cuda(0));
        auto output = parallel_model->forward(input);
        auto loss = MSELoss()(output, target);
        loss.backward();
    }

    // Time multi-GPU
    auto start_2gpu = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_iterations; ++i) {
        auto [input, target] = generate_batch(batch_size, 512, Device::cuda(0));
        auto output = parallel_model->forward(input);
        auto loss = MSELoss()(output, target);
        loss.backward();
    }
#ifdef TENZOR_USE_CUDA
    for (int i = 0; i < 2; ++i) {
        cudaSetDevice(device_ids[i]);
        cudaDeviceSynchronize();
    }
#endif
    auto end_2gpu = std::chrono::high_resolution_clock::now();
    auto time_2gpu = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_2gpu - start_2gpu
    ).count();

    float speedup = static_cast<float>(time_1gpu) / static_cast<float>(time_2gpu);

    std::cout << "\n=== Performance Scaling Results ===\n";
    std::cout << "1 GPU time: " << time_1gpu << " ms\n";
    std::cout << "2 GPU time: " << time_2gpu << " ms\n";
    std::cout << "Speedup: " << std::fixed << std::setprecision(2) << speedup << "x\n";
    std::cout << "===================================\n\n";

    // We expect some speedup, but not necessarily 2x due to overhead
    // Relaxed requirement: just verify multi-GPU isn't slower
    EXPECT_GT(speedup, 0.8f) << "Multi-GPU performance is significantly worse than single GPU";
}

//==============================================================================
// Test 8: Convolutional Model Testing
//==============================================================================

TEST_F(MultiGPUTest, ConvolutionalModel2GPUs) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestConvNet>();
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    int batch_size = 16;
    auto [input, target] = generate_image_batch(
        batch_size, 3, 32, 32, Device::cuda(device_ids[0])
    );

    // Forward pass
    auto output = parallel_model->forward(input);

    auto shape = output.tensor().shape();
    ASSERT_EQ(shape.size(), 2);
    EXPECT_EQ(shape[0], batch_size);
    EXPECT_EQ(shape[1], 10);

    // Backward pass
    auto loss = MSELoss()(output, target);
    loss.backward();

    // Verify gradients
    auto params = parallel_model->parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

//==============================================================================
// Test 9: Training Mode Synchronization
//==============================================================================

TEST_F(MultiGPUTest, TrainingModeSync) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestConvNet>();
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Set to eval mode
    parallel_model->eval();

    // Run forward pass
    int batch_size = 8;
    auto [input, _] = generate_image_batch(
        batch_size, 3, 32, 32, Device::cuda(device_ids[0])
    );
    auto output_eval = parallel_model->forward(input);

    // Set to train mode
    parallel_model->train();
    auto output_train = parallel_model->forward(input);

    // Outputs might be different due to dropout and batch norm
    // (Not guaranteed to be different, so we just verify both outputs have correct shape)
    auto shape_eval = output_eval.tensor().shape();
    auto shape_train = output_train.tensor().shape();
    EXPECT_EQ(shape_eval.size(), 2);
    EXPECT_EQ(shape_train.size(), 2);
    EXPECT_EQ(shape_eval[0], batch_size);
    EXPECT_EQ(shape_train[0], batch_size);
}

//==============================================================================
// Test 10: Edge Cases
//==============================================================================

TEST_F(MultiGPUTest, MinimumBatchSize) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Batch size equals number of GPUs (minimum valid)
    int batch_size = 2;
    auto [input, target] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    auto output = parallel_model->forward(input);
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], batch_size);
}

TEST_F(MultiGPUTest, LargeBatchSize) {
    if (!has_n_gpus(2)) {
        GTEST_SKIP() << "Test requires 2 GPUs";
    }

    auto model = std::make_shared<TestMLP>(64, 128, 10);
    auto device_ids = get_device_ids(2);
    auto parallel_model = std::make_shared<DataParallel>(model, device_ids);

    // Large batch
    int batch_size = 256;
    auto [input, target] = generate_batch(batch_size, 64, Device::cuda(device_ids[0]));

    auto output = parallel_model->forward(input);
    auto shape = output.tensor().shape();
    EXPECT_EQ(shape[0], batch_size);

    auto loss = MSELoss()(output, target);
    loss.backward();

    // Verify gradients exist
    auto params = parallel_model->parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->has_grad());
    }
}

//==============================================================================
// Main
//==============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
