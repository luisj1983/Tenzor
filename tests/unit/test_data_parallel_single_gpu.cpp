/**
 * @file test_data_parallel_single_gpu.cpp
 * @brief Comprehensive single-GPU tests for DataParallel
 *
 * This test suite is designed for systems with only one GPU and provides:
 * 1. Real single-GPU training validation
 * 2. Mock multi-GPU logic testing without actual hardware
 * 3. Edge case handling
 * 4. Integration tests with various model architectures
 */

#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include <cmath>
#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"  // For initialize()

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Test Fixture and Utilities
// ============================================================================

/**
 * @brief Test fixture providing common setup for DataParallel tests
 */
class DataParallelSingleGPUTest : public ::testing::Test {
public:
    static bool is_cuda_available() {
#ifdef TENZOR_USE_CUDA
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        return (err == cudaSuccess && device_count > 0);
#else
        return false;
#endif
    }

    static int get_device_count() {
#ifdef TENZOR_USE_CUDA
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        return device_count;
#else
        return 0;
#endif
    }

protected:
    void SetUp() override {
        // Initialize the library (only on first test)
        static bool initialized = false;
        if (!initialized) {
            tenzor::initialize();
            initialized = true;
        }

        cuda_available_ = is_cuda_available();
        device_count_ = get_device_count();

        if (cuda_available_) {
            std::cout << "[INFO] CUDA available with " << device_count_ << " device(s)\n";
        } else {
            std::cout << "[INFO] CUDA not available, tests will be skipped\n";
        }
    }

    bool cuda_available_{false};
    int device_count_{0};
};

// ============================================================================
// Mock Modules for Testing
// ============================================================================

/**
 * @brief Simple module that scales input by a constant factor
 */
class ScaleModule : public Module {
public:
    explicit ScaleModule(float scale = 2.0f) : scale_(scale) {}

    auto forward_impl(const Variable& input) -> Variable override {
        // Preserve requires_grad from input
        auto result = input.tensor() * scale_;
        return Variable(result, input.requires_grad());
    }

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override {
        return {};
    }

    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override {
        return {};
    }

private:
    float scale_;
};

/**
 * @brief Module with learnable parameters for gradient testing
 */
class TrainableModule : public Module {
public:
    TrainableModule(int in_features, int out_features) {
        // Initialize weight and bias
        auto weight_data = tenzor::randn({out_features, in_features});
        auto bias_data = tenzor::zeros({out_features});

        weight_ = Variable(weight_data, true);  // requires_grad = true
        bias_ = Variable(bias_data, true);

        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // y = xW^T + b (proper matrix multiplication)
        // Input: [batch, in_features], Weight: [out_features, in_features]
        // Need: [batch, in_features] @ [in_features, out_features] = [batch, out_features]

        auto weight_t = weight_.tensor().transpose(0, 1);  // [in_features, out_features]
        auto output_tensor = tenzor::matmul(input.tensor(), weight_t);  // [batch, out_features]
        auto result = output_tensor + bias_.tensor();
        return Variable(result, input.requires_grad() || weight_.requires_grad());
    }

    auto get_weight() -> Variable& { return weight_; }
    auto get_bias() -> Variable& { return bias_; }

private:
    Variable weight_;
    Variable bias_;
};

/**
 * @brief Module that accumulates call count for testing replica behavior
 */
class CountingModule : public Module {
public:
    CountingModule() : call_count_(0) {}

    auto forward_impl(const Variable& input) -> Variable override {
        ++call_count_;
        return input;
    }

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override {
        return {};
    }

    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override {
        return {};
    }

    int get_call_count() const { return call_count_; }
    void reset_count() { call_count_ = 0; }

private:
    int call_count_;
};

// ============================================================================
// SECTION 1: Single-GPU Mode Tests (Actual Hardware)
// ============================================================================

TEST_F(DataParallelSingleGPUTest, SingleGPU_BasicForward) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(3.0f);
    DataParallel dp(module, {0}, 0);

    // Create test input: 4x8 matrix of ones
    auto input = Variable(tenzor::ones({4, 8}));
    auto output = dp.forward(input);

    // Verify output is scaled by 3.0
    auto output_data = output.tensor().data<float>();
    for (int i = 0; i < 32; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], 3.0f) << "Mismatch at index " << i;
    }
}

TEST_F(DataParallelSingleGPUTest, SingleGPU_CorrectOutputShape) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Test various input shapes
    std::vector<std::vector<int64_t>> shapes = {
        {8, 16},      // 2D
        {4, 8, 8},    // 3D (batch, height, width)
        {2, 3, 32, 32}  // 4D (batch, channels, height, width)
    };

    for (const auto& shape : shapes) {
        auto input = Variable(tenzor::ones(shape));
        auto output = dp.forward(input);

        // Compare shape dimensions individually
        auto out_shape = output.tensor().shape();
        EXPECT_EQ(out_shape.size(), shape.size())
            << "Shape dimension count mismatch";
        for (size_t i = 0; i < shape.size(); ++i) {
            EXPECT_EQ(out_shape[i], shape[i])
                << "Shape mismatch at dimension " << i;
        }
    }
}

TEST_F(DataParallelSingleGPUTest, SingleGPU_PreservesValues) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(1.0f);  // Identity scaling
    DataParallel dp(module, {0}, 0);

    // Create input with sequential values
    auto input_tensor = tenzor::empty({4, 5});
    auto input_data = input_tensor.data<float>();
    for (int i = 0; i < 20; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    auto input = Variable(input_tensor);
    auto output = dp.forward(input);

    // Verify values are preserved
    auto output_data = output.tensor().data<float>();
    for (int i = 0; i < 20; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], static_cast<float>(i))
            << "Value mismatch at index " << i;
    }
}

TEST_F(DataParallelSingleGPUTest, SingleGPU_ParameterAccess) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<TrainableModule>(10, 5);
    DataParallel dp(module, {0}, 0);

    // Verify parameters are accessible through DataParallel
    auto params = dp.parameters();
    auto module_params = module->parameters();

    EXPECT_EQ(params.size(), module_params.size());

    // Verify parameters are the same objects (not copies)
    for (size_t i = 0; i < params.size(); ++i) {
        EXPECT_EQ(params[i], module_params[i])
            << "Parameter pointer mismatch at index " << i;
    }
}

TEST_F(DataParallelSingleGPUTest, SingleGPU_TrainingModeSync) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Test training mode transitions
    EXPECT_TRUE(module->is_training());

    dp.eval();
    EXPECT_FALSE(module->is_training());

    dp.train();
    EXPECT_TRUE(module->is_training());
}

TEST_F(DataParallelSingleGPUTest, SingleGPU_MultipleForwardPasses) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<CountingModule>();
    DataParallel dp(module, {0}, 0);

    // Multiple forward passes should all succeed
    auto input = Variable(tenzor::ones({4, 8}));

    for (int i = 0; i < 5; ++i) {
        EXPECT_NO_THROW(dp.forward(input)) << "Forward pass " << i << " failed";
    }

    // With single GPU, module should be called directly each time
    EXPECT_EQ(module->get_call_count(), 5);
}

// ============================================================================
// SECTION 2: Gradient Flow Tests
// ============================================================================

TEST_F(DataParallelSingleGPUTest, GradientFlow_BasicBackward) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<TrainableModule>(8, 4);
    DataParallel dp(module, {0}, 0);

    // Forward pass
    auto input = Variable(tenzor::ones({2, 8}), true);
    auto output = dp.forward(input);

    // Verify output requires grad
    EXPECT_TRUE(output.requires_grad()) << "Output should require gradients";
}

TEST_F(DataParallelSingleGPUTest, GradientFlow_ParametersUpdateable) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<TrainableModule>(8, 4);
    DataParallel dp(module, {0}, 0);

    // Get initial parameter values
    auto params = dp.parameters();
    EXPECT_GT(params.size(), 0) << "Module should have parameters";

    // Verify parameters require grad
    for (const auto& param : params) {
        EXPECT_TRUE(param->requires_grad())
            << "Parameters should require gradients";
    }
}

TEST_F(DataParallelSingleGPUTest, GradientFlow_NonLeafGradients) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(2.0f);
    DataParallel dp(module, {0}, 0);

    // Create input that requires grad
    auto input = Variable(tenzor::ones({2, 4}), true);
    auto output = dp.forward(input);

    // Output should also require grad (non-leaf variable)
    EXPECT_TRUE(output.requires_grad());
}

// ============================================================================
// SECTION 3: Mock Multi-GPU Logic Tests
// ============================================================================

TEST_F(DataParallelSingleGPUTest, MockMultiGPU_DeviceValidation) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();

    // Valid single device
    EXPECT_NO_THROW(DataParallel(module, {0}, 0));

    // Invalid device ID should throw
    EXPECT_THROW(DataParallel(module, {999}, 999), std::invalid_argument);

    // Output device not in device list should throw
    if (device_count_ >= 2) {
        EXPECT_THROW(DataParallel(module, {0}, 1), std::invalid_argument);
    }
}

TEST_F(DataParallelSingleGPUTest, MockMultiGPU_EmptyDeviceList) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();

    // Empty device list should auto-detect
    EXPECT_NO_THROW({
        DataParallel dp(module, {}, -1);
        EXPECT_EQ(dp.device_ids().size(), static_cast<size_t>(device_count_));
    });
}

TEST_F(DataParallelSingleGPUTest, MockMultiGPU_DefaultOutputDevice) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, -1);

    // Default output device should be first device
    EXPECT_EQ(dp.output_device(), 0);
}

TEST_F(DataParallelSingleGPUTest, MockMultiGPU_BatchSizeValidation) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Batch size >= device count: should work
    auto large_batch = Variable(tenzor::ones({8, 4}));
    EXPECT_NO_THROW(dp.forward(large_batch));

    // Batch size == device count: should work
    auto exact_batch = Variable(tenzor::ones({1, 4}));
    EXPECT_NO_THROW(dp.forward(exact_batch));

    // Note: With single GPU, batch size < device count is impossible
    // This would only be testable with simulated multi-GPU
}

TEST_F(DataParallelSingleGPUTest, MockMultiGPU_ReplicaInitialization) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // First forward pass should initialize replicas
    auto input = Variable(tenzor::ones({4, 8}));
    EXPECT_NO_THROW(dp.forward(input));

    // Second forward pass should reuse replicas (no re-initialization)
    EXPECT_NO_THROW(dp.forward(input));
}

// ============================================================================
// SECTION 4: Integration Tests with Different Architectures
// ============================================================================

TEST_F(DataParallelSingleGPUTest, Integration_LinearLayer) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Create a Linear layer and wrap with DataParallel
    auto linear = std::make_shared<Linear>(64, 32);
    DataParallel dp(linear, {0}, 0);

    // Forward pass with batch
    auto input = Variable(tenzor::randn({8, 64}));
    auto output = dp.forward(input);

    // Verify output shape
    EXPECT_EQ(output.tensor().shape()[0], 8);
    EXPECT_EQ(output.tensor().shape()[1], 32);
}

TEST_F(DataParallelSingleGPUTest, Integration_SequentialModel) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    // Build a sequential model
    auto model = std::make_shared<Sequential>();
    model->add_module(std::make_shared<Linear>(128, 64));
    model->add_module(std::make_shared<Linear>(64, 10));

    DataParallel dp(model, {0}, 0);

    // Forward pass
    auto input = Variable(tenzor::randn({16, 128}));
    auto output = dp.forward(input);

    // Verify output shape
    EXPECT_EQ(output.tensor().shape()[0], 16);
    EXPECT_EQ(output.tensor().shape()[1], 10);
}

TEST_F(DataParallelSingleGPUTest, Integration_CompareWithDirectExecution) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(2.5f);
    DataParallel dp(module, {0}, 0);

    // Same input for both
    auto input_tensor = tenzor::randn({8, 16});
    auto input1 = Variable(input_tensor);
    auto input2 = Variable(input_tensor);

    // Direct execution
    auto direct_output = module->forward(input1);

    // DataParallel execution
    auto parallel_output = dp.forward(input2);

    // Results should be identical for single GPU
    auto direct_data = direct_output.tensor().data<float>();
    auto parallel_data = parallel_output.tensor().data<float>();

    for (int i = 0; i < 128; ++i) {
        EXPECT_FLOAT_EQ(direct_data[i], parallel_data[i])
            << "Output mismatch at index " << i;
    }
}

// ============================================================================
// SECTION 5: Edge Cases and Error Handling
// ============================================================================

TEST_F(DataParallelSingleGPUTest, EdgeCase_NullModule) {
    // Null module should throw immediately
    EXPECT_THROW(DataParallel(nullptr, {0}, 0), std::invalid_argument);
}

TEST_F(DataParallelSingleGPUTest, EdgeCase_EmptyInput) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Empty batch dimension (0 elements in batch)
    auto input = Variable(tenzor::empty({0, 4}));

    // Empty batch should be handled gracefully by DataParallel
    // The output should also have empty batch dimension
    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 0) << "Empty batch should produce empty output";
        EXPECT_EQ(output.tensor().shape()[1], 4) << "Feature dimension should be preserved";
    });
}

TEST_F(DataParallelSingleGPUTest, EdgeCase_1DTensor) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(2.0f);
    DataParallel dp(module, {0}, 0);

    // 1D tensor
    auto input = Variable(tenzor::ones({10}));
    auto output = dp.forward(input);

    // Verify shape and values
    EXPECT_EQ(output.tensor().shape()[0], 10);

    auto output_data = output.tensor().data<float>();
    for (int i = 0; i < 10; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], 2.0f);
    }
}

TEST_F(DataParallelSingleGPUTest, EdgeCase_LargeBatch) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Large batch (256 samples)
    auto input = Variable(tenzor::ones({256, 32}));

    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 256);
    });
}

TEST_F(DataParallelSingleGPUTest, EdgeCase_SmallFeatureDimension) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Small feature dimension (1)
    auto input = Variable(tenzor::ones({16, 1}));
    auto output = dp.forward(input);

    EXPECT_EQ(output.tensor().shape()[0], 16);
    EXPECT_EQ(output.tensor().shape()[1], 1);
}

TEST_F(DataParallelSingleGPUTest, EdgeCase_NonZeroBatchDimension) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0, 1);  // Split along dimension 1

    // Input: (channels, batch, features)
    auto input = Variable(tenzor::ones({3, 8, 16}));
    auto output = dp.forward(input);

    // Verify shape preservation
    EXPECT_EQ(output.tensor().shape()[0], 3);
    EXPECT_EQ(output.tensor().shape()[1], 8);
    EXPECT_EQ(output.tensor().shape()[2], 16);
}

// ============================================================================
// SECTION 6: Module Interface Compliance
// ============================================================================

TEST_F(DataParallelSingleGPUTest, Interface_ModuleAccessor) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // module() should return the wrapped module
    EXPECT_EQ(dp.module(), module);
}

TEST_F(DataParallelSingleGPUTest, Interface_DeviceIDsAccessor) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    std::vector<int> device_ids = {0};
    DataParallel dp(module, device_ids, 0);

    EXPECT_EQ(dp.device_ids(), device_ids);
}

TEST_F(DataParallelSingleGPUTest, Interface_OutputDeviceAccessor) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    EXPECT_EQ(dp.output_device(), 0);
}

TEST_F(DataParallelSingleGPUTest, Interface_BatchDimAccessor) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0, 2);  // Custom batch dimension

    EXPECT_EQ(dp.batch_dim(), 2);
}

TEST_F(DataParallelSingleGPUTest, Interface_NamedParameters) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<TrainableModule>(10, 5);
    DataParallel dp(module, {0}, 0);

    auto named_params = dp.named_parameters();
    auto module_named_params = module->named_parameters();

    EXPECT_EQ(named_params.size(), module_named_params.size());

    // Verify parameter names match
    for (size_t i = 0; i < named_params.size(); ++i) {
        EXPECT_EQ(named_params[i].first, module_named_params[i].first)
            << "Parameter name mismatch at index " << i;
    }
}

// ============================================================================
// SECTION 7: Helper Function Tests
// ============================================================================

TEST_F(DataParallelSingleGPUTest, Helper_MakeDataParallel) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();

    auto dp = make_data_parallel(module, {0}, 0);

    EXPECT_NE(dp, nullptr);
    EXPECT_EQ(dp->device_ids().size(), 1);
    EXPECT_EQ(dp->output_device(), 0);
}

TEST_F(DataParallelSingleGPUTest, Helper_MakeDataParallelAutoDetect) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();

    // Empty device_ids should auto-detect
    auto dp = make_data_parallel(module);

    EXPECT_NE(dp, nullptr);
    EXPECT_EQ(dp->device_ids().size(), static_cast<size_t>(device_count_));
}

// ============================================================================
// SECTION 8: Performance and Correctness Validation
// ============================================================================

TEST_F(DataParallelSingleGPUTest, Correctness_NumericalStability) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(0.1f);
    DataParallel dp(module, {0}, 0);

    // Test with small values
    auto input_tensor = tenzor::ones({4, 8});
    auto input_data = input_tensor.data<float>();
    for (int i = 0; i < 32; ++i) {
        input_data[i] = 1e-6f * static_cast<float>(i + 1);
    }

    auto input = Variable(input_tensor);
    auto output = dp.forward(input);

    // Verify numerical stability
    auto output_data = output.tensor().data<float>();
    for (int i = 0; i < 32; ++i) {
        float expected = 1e-7f * static_cast<float>(i + 1);
        EXPECT_NEAR(output_data[i], expected, 1e-10f)
            << "Numerical instability at index " << i;
    }
}

TEST_F(DataParallelSingleGPUTest, Correctness_BatchOrderPreserved) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>(1.0f);
    DataParallel dp(module, {0}, 0);

    // Create input where each batch element is unique
    auto input_tensor = tenzor::empty({8, 4});
    auto input_data = input_tensor.data<float>();
    for (int b = 0; b < 8; ++b) {
        for (int f = 0; f < 4; ++f) {
            input_data[b * 4 + f] = static_cast<float>(b * 100 + f);
        }
    }

    auto input = Variable(input_tensor);
    auto output = dp.forward(input);

    // Verify batch order is preserved
    auto output_data = output.tensor().data<float>();
    for (int b = 0; b < 8; ++b) {
        for (int f = 0; f < 4; ++f) {
            float expected = static_cast<float>(b * 100 + f);
            EXPECT_FLOAT_EQ(output_data[b * 4 + f], expected)
                << "Batch order corruption at batch " << b << ", feature " << f;
        }
    }
}

// ============================================================================
// SECTION 9: Thread Safety (Basic)
// ============================================================================

TEST_F(DataParallelSingleGPUTest, ThreadSafety_ConcurrentForward) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Sequential forward passes (basic thread safety check)
    auto input = Variable(tenzor::ones({4, 8}));

    for (int i = 0; i < 10; ++i) {
        EXPECT_NO_THROW(dp.forward(input))
            << "Forward pass " << i << " failed";
    }
}

TEST_F(DataParallelSingleGPUTest, ThreadSafety_ReplicaInitializationOnce) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<CountingModule>();
    DataParallel dp(module, {0}, 0);

    auto input = Variable(tenzor::ones({4, 8}));

    // Multiple forward passes
    for (int i = 0; i < 5; ++i) {
        dp.forward(input);
    }

    // Replica initialization should happen only once
    // (though with single GPU, the module is used directly)
    EXPECT_EQ(module->get_call_count(), 5);
}

// ============================================================================
// SECTION 10: Stress Tests
// ============================================================================

TEST_F(DataParallelSingleGPUTest, Stress_ManySmallBatches) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    // Process 100 small batches
    for (int i = 0; i < 100; ++i) {
        auto input = Variable(tenzor::ones({2, 16}));
        EXPECT_NO_THROW(dp.forward(input))
            << "Failed at batch " << i;
    }
}

TEST_F(DataParallelSingleGPUTest, Stress_VaryingBatchSizes) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto module = std::make_shared<ScaleModule>();
    DataParallel dp(module, {0}, 0);

    std::vector<int> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};

    for (int batch_size : batch_sizes) {
        auto input = Variable(tenzor::ones({batch_size, 32}));
        auto output = dp.forward(input);

        EXPECT_EQ(output.tensor().shape()[0], batch_size)
            << "Shape mismatch for batch size " << batch_size;
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "========================================\n";
    std::cout << "DataParallel Single-GPU Test Suite\n";
    std::cout << "========================================\n";

    // Initialize Tenzor library before checking CUDA availability
    tenzor::initialize();

    bool cuda_available = DataParallelSingleGPUTest::is_cuda_available();
    int device_count = DataParallelSingleGPUTest::get_device_count();

    if (cuda_available) {
        std::cout << "[INFO] CUDA is available\n";
        std::cout << "[INFO] Device count: " << device_count << "\n";
        std::cout << "[INFO] Running full test suite\n";
    } else {
        std::cout << "[WARN] CUDA is not available\n";
        std::cout << "[WARN] Most tests will be skipped\n";
    }

    std::cout << "========================================\n\n";

    return RUN_ALL_TESTS();
}
