/**
 * @file test_offload.cpp
 * @brief Comprehensive tests for ZeRO Stage 2 Parameter Offloading API
 *
 * Tests cover:
 * - OffloadContext lifecycle and configuration
 * - Parameter offloading and loading
 * - Gradient offloading
 * - ComputeContext RAII behavior
 * - Integration with Module system
 * - Real training scenarios
 * - Performance characteristics
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"  // CC.18: SKIP_WITH_REASON
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/offload.hpp>
#include <chrono>
#include <cmath>

using namespace tenzor;
using namespace tenzor::nn;

// ==========================
// Helper Functions
// ==========================

/**
 * @brief Create a simple test module with multiple linear layers.
 */
auto createTestModule(int input_size, int hidden_size, int output_size)
    -> std::unique_ptr<Sequential> {
    auto model = std::make_unique<Sequential>();
    auto linear1 = std::make_shared<Linear>(input_size, hidden_size);
    auto linear2 = std::make_shared<Linear>(hidden_size, output_size);
    model->add_module(linear1);
    model->add_module(linear2);
    return model;
}

/**
 * @brief Check if tensor is on specified device type.
 */
auto isTensorOn(const Tensor& t, Device::Type device_type) -> bool {
    return t.device().type == device_type;
}

/**
 * @brief Verify parameter data is unchanged after offload/load cycle.
 */
auto verifyParameterData(const std::vector<Tensor>& before,
                        const std::vector<Tensor>& after,
                        float tolerance = 1e-6f) -> bool {
    if (before.size() != after.size()) return false;

    for (size_t i = 0; i < before.size(); ++i) {
        auto before_cpu = before[i].to(Device::cpu());
        auto after_cpu = after[i].to(Device::cpu());

        if (before_cpu.numel() != after_cpu.numel()) return false;

        const float* before_data = before_cpu.data<float>();
        const float* after_data = after_cpu.data<float>();

        for (int64_t j = 0; j < before_cpu.numel(); ++j) {
            if (std::abs(before_data[j] - after_data[j]) > tolerance) {
                return false;
            }
        }
    }
    return true;
}

/**
 * @brief Get total memory size of parameters in MB.
 */
auto getParameterMemoryMB(const std::vector<std::shared_ptr<Variable>>& params) -> float {
    size_t total_bytes = 0;
    for (const auto& param : params) {
        total_bytes += param->tensor().numel() * sizeof(float);
    }
    return static_cast<float>(total_bytes) / (1024.0f * 1024.0f);
}

/**
 * @brief Check if CUDA is available for testing.
 */
auto cudaAvailable() -> bool {
    try {
        auto test = zeros({1}, DType::Float32, Device::cuda(0));
        return true;
    } catch (...) {
        return false;
    }
}

// ==========================
// Test Fixture
// ==========================

class ParameterOffloadTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        cuda_available = cudaAvailable();
        if (cuda_available) {
            default_config.offload_parameters = true;
            default_config.offload_gradients = false;
            default_config.offload_threshold = 0;  // Offload all
            default_config.prefetch_depth = 1;
            default_config.pin_first_layer = false;  // Don't pin first layer for testing
            default_config.pin_last_layer = false;   // Don't pin last layer for testing
        }
    }

    bool cuda_available = false;
    OffloadContext::Config default_config;
};

// ==========================
// OffloadContext Tests (6 tests)
// ==========================

TEST_F(ParameterOffloadTest, OffloadContext_Constructor) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 10);
    model->to(Device::cuda(0));

    // Constructor should not throw
    EXPECT_NO_THROW({
        OffloadContext ctx(*model, default_config);
    });
}

TEST_F(ParameterOffloadTest, OffloadContext_Enable) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 10);
    model->to(Device::cuda(0));

    OffloadContext ctx(*model, default_config);

    // Initially disabled
    EXPECT_FALSE(ctx.is_enabled());

    // Enable offloading
    ctx.enable();
    EXPECT_TRUE(ctx.is_enabled());
}

TEST_F(ParameterOffloadTest, OffloadContext_Disable) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 10);
    model->to(Device::cuda(0));

    OffloadContext ctx(*model, default_config);
    ctx.enable();

    EXPECT_TRUE(ctx.is_enabled());

    // Disable offloading
    ctx.disable();
    EXPECT_FALSE(ctx.is_enabled());
}

TEST_F(ParameterOffloadTest, OffloadContext_GetStats) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 10);
    model->to(Device::cuda(0));

    OffloadContext ctx(*model, default_config);
    ctx.enable();

    // Get statistics
    auto stats = ctx.get_stats();

    // Stats should have reasonable values
    EXPECT_GE(stats.num_parameters_offloaded, 0);
    EXPECT_GE(stats.current_cpu_memory_mb, 0);
    EXPECT_GE(stats.peak_gpu_memory_mb, 0);
    EXPECT_GE(stats.total_offload_count, 0);
}

TEST_F(ParameterOffloadTest, OffloadContext_RegisterHooks) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 10);
    model->to(Device::cuda(0));

    OffloadContext ctx(*model, default_config);

    // Enabling should register hooks
    EXPECT_NO_THROW({
        ctx.enable();
    });

    EXPECT_TRUE(ctx.is_enabled());
}

TEST_F(ParameterOffloadTest, OffloadContext_Destructor) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 10);
    model->to(Device::cuda(0));

    // Destructor should cleanup without throwing
    {
        OffloadContext ctx(*model, default_config);
        ctx.enable();
    } // ctx destroyed here

    // Model should still be valid
    EXPECT_NO_THROW({
        auto params = model->parameters();
        EXPECT_GT(params.size(), 0);
    });
}

// ==========================
// Parameter Offloading Tests (6 tests)
// ==========================

TEST_F(ParameterOffloadTest, OffloadParams_SingleLayer) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    // Create simple linear layer
    auto linear = std::make_shared<Linear>(128, 64);
    linear->to(Device::cuda(0));

    auto model = std::make_unique<Sequential>();
    model->add_module(linear);

    // Create offload context
    OffloadContext ctx(*model, default_config);
    ctx.enable();

    // Parameters should be offloaded
    auto stats = ctx.get_stats();
    EXPECT_GT(stats.num_parameters_offloaded, 0);
    EXPECT_GT(stats.current_cpu_memory_mb, 0.0f);
}

TEST_F(ParameterOffloadTest, OffloadParams_MultipleLayers) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(256, 512, 128);
    model->to(Device::cuda(0));

    auto params_before = model->parameters();
    size_t num_params = params_before.size();

    OffloadContext ctx(*model, default_config);
    ctx.enable();

    auto stats = ctx.get_stats();

    // Should offload all parameters
    EXPECT_EQ(stats.num_parameters_offloaded, num_params);
    EXPECT_GT(stats.current_cpu_memory_mb, 0.0f);
}

TEST_F(ParameterOffloadTest, OffloadParams_SelectiveThreshold) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    // Create model with small and large parameters
    auto model = std::make_unique<Sequential>();
    auto small_layer = std::make_shared<Linear>(10, 10);     // Small parameter
    auto large_layer = std::make_shared<Linear>(1000, 1000); // Large parameter

    model->add_module(small_layer);
    model->add_module(large_layer);
    model->to(Device::cuda(0));

    // Set threshold to only offload large parameters
    OffloadContext::Config config = default_config;
    config.offload_threshold = 500;  // Only offload params with >500 elements

    OffloadContext ctx(*model, config);
    ctx.enable();

    auto stats = ctx.get_stats();

    // Should only offload large parameters
    EXPECT_GT(stats.num_parameters_offloaded, 0);
    EXPECT_LT(stats.num_parameters_offloaded, model->parameters().size());
}

TEST_F(ParameterOffloadTest, OffloadParams_FirstLayerPinned) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 64);
    model->to(Device::cuda(0));

    // Pin first layer on GPU
    OffloadContext::Config config = default_config;
    config.pin_first_layer = true;

    OffloadContext ctx(*model, config);
    ctx.enable();

    auto stats = ctx.get_stats();
    auto all_params = model->parameters();

    // Should offload fewer parameters (first layer pinned)
    EXPECT_GT(stats.num_parameters_offloaded, 0);
    EXPECT_LT(stats.num_parameters_offloaded, all_params.size());
}

TEST_F(ParameterOffloadTest, OffloadParams_LastLayerPinned) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 64);
    model->to(Device::cuda(0));

    // Pin last layer on GPU
    OffloadContext::Config config = default_config;
    config.pin_last_layer = true;

    OffloadContext ctx(*model, config);
    ctx.enable();

    auto stats = ctx.get_stats();
    auto all_params = model->parameters();

    // Should offload fewer parameters (last layer pinned)
    EXPECT_GT(stats.num_parameters_offloaded, 0);
    EXPECT_LT(stats.num_parameters_offloaded, all_params.size());
}

TEST_F(ParameterOffloadTest, OffloadParams_PreservesData) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 32);
    model->to(Device::cuda(0));

    // Save parameter values before offloading
    auto params = model->parameters();
    std::vector<Tensor> params_before;
    for (const auto& param : params) {
        params_before.push_back(param->tensor().clone());
    }

    // Offload and load back
    OffloadContext ctx(*model, default_config);
    ctx.enable();
    ctx.disable();

    // Get parameters after
    std::vector<Tensor> params_after;
    for (const auto& param : params) {
        params_after.push_back(param->tensor());
    }

    // Data should be preserved
    EXPECT_TRUE(verifyParameterData(params_before, params_after));
}

// ==========================
// Gradient Offloading Tests (4 tests)
// ==========================

TEST_F(ParameterOffloadTest, OffloadGradients_AfterBackward) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 10);
    model->to(Device::cuda(0));

    // Enable gradient offloading
    OffloadContext::Config config = default_config;
    config.offload_parameters = false;
    config.offload_gradients = true;

    OffloadContext ctx(*model, config);
    ctx.enable();

    // Forward and backward pass
    auto input = Variable(randn({8, 64}, DType::Float32, Device::cuda(0)), true);
    auto output = model->forward(input);

    auto loss = sum(output);  // Use autograd sum
    loss.backward();

    auto stats = ctx.get_stats();

    // Gradients should be offloaded
    EXPECT_GT(stats.num_gradients_offloaded, 0);
}

TEST_F(ParameterOffloadTest, OffloadGradients_MultipleParams) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 64);
    model->to(Device::cuda(0));

    OffloadContext::Config config = default_config;
    config.offload_parameters = false;
    config.offload_gradients = true;

    OffloadContext ctx(*model, config);
    ctx.enable();

    // Run backward pass
    auto input = Variable(randn({4, 128}, DType::Float32, Device::cuda(0)), true);
    auto output = model->forward(input);
    auto loss = sum(output);  // Use autograd sum
    loss.backward();

    auto stats = ctx.get_stats();
    auto params = model->parameters();

    // Should offload gradients for all parameters
    EXPECT_EQ(stats.num_gradients_offloaded, params.size());
}

TEST_F(ParameterOffloadTest, OffloadGradients_PreservesValues) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(32, 64, 10);
    model->to(Device::cuda(0));

    OffloadContext::Config config = default_config;
    config.offload_parameters = false;
    config.offload_gradients = true;

    OffloadContext ctx(*model, config);
    ctx.enable();

    // Compute gradients
    auto input = Variable(randn({4, 32}, DType::Float32, Device::cuda(0)), true);
    auto output = model->forward(input);
    auto loss = sum(output);  // Use autograd sum
    loss.backward();

    // Save gradient values
    auto params = model->parameters();
    std::vector<Tensor> grads_before;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            grads_before.push_back(param->grad()->clone());
        }
    }

    // Disable offloading (should restore gradients)
    ctx.disable();

    // Verify gradients unchanged
    std::vector<Tensor> grads_after;
    for (const auto& param : params) {
        if (param->grad().has_value()) {
            grads_after.push_back(*param->grad());
        }
    }

    EXPECT_TRUE(verifyParameterData(grads_before, grads_after));
}

TEST_F(ParameterOffloadTest, OffloadGradients_PrefetchForOptimizer) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 32);
    model->to(Device::cuda(0));

    OffloadContext::Config config = default_config;
    config.offload_parameters = true;
    config.offload_gradients = true;
    config.prefetch_depth = 1;

    OffloadContext ctx(*model, config);
    ctx.enable();

    // Training iteration
    auto input = Variable(randn({8, 64}, DType::Float32, Device::cuda(0)), true);
    auto output = model->forward(input);
    auto loss = sum(output);  // Use autograd sum
    loss.backward();

    auto stats = ctx.get_stats();

    // Should have offloaded both parameters and gradients
    EXPECT_GT(stats.num_parameters_offloaded, 0);
    EXPECT_GT(stats.num_gradients_offloaded, 0);
}

// ==========================
// ComputeContext Tests (4 tests)
// ==========================

TEST_F(ParameterOffloadTest, ComputeContext_RAII_LoadsParams) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto param = randn({1000, 1000}, DType::Float32, Device::cpu());

    {
        ComputeContext ctx({&param});
        // Inside scope, param should be on GPU
        EXPECT_TRUE(isTensorOn(param, Device::Type::CUDA));
    }

    // After scope, param should be back on CPU
    EXPECT_TRUE(isTensorOn(param, Device::Type::CPU));
}

TEST_F(ParameterOffloadTest, ComputeContext_RAII_OffloadsOnDestroy) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto param1 = randn({500, 500}, DType::Float32, Device::cpu());
    auto param2 = randn({500, 500}, DType::Float32, Device::cpu());

    // Verify both on CPU initially
    EXPECT_TRUE(isTensorOn(param1, Device::Type::CPU));
    EXPECT_TRUE(isTensorOn(param2, Device::Type::CPU));

    {
        ComputeContext ctx({&param1, &param2});

        // Both should be on GPU
        EXPECT_TRUE(isTensorOn(param1, Device::Type::CUDA));
        EXPECT_TRUE(isTensorOn(param2, Device::Type::CUDA));
    }

    // Both should be back on CPU after destruction
    EXPECT_TRUE(isTensorOn(param1, Device::Type::CPU));
    EXPECT_TRUE(isTensorOn(param2, Device::Type::CPU));
}

TEST_F(ParameterOffloadTest, ComputeContext_MultipleTensors) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    std::vector<Tensor> tensors;
    const int num_tensors = 5;

    for (int i = 0; i < num_tensors; ++i) {
        tensors.push_back(randn({100, 100}, DType::Float32, Device::cpu()));
    }

    std::vector<Tensor*> tensor_ptrs;
    for (auto& t : tensors) {
        tensor_ptrs.push_back(&t);
    }

    {
        ComputeContext ctx(tensor_ptrs);

        // All should be on GPU
        for (const auto& t : tensors) {
            EXPECT_TRUE(isTensorOn(t, Device::Type::CUDA));
        }
    }

    // All should be back on CPU
    for (const auto& t : tensors) {
        EXPECT_TRUE(isTensorOn(t, Device::Type::CPU));
    }
}

TEST_F(ParameterOffloadTest, ComputeContext_NestedScopes) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto param1 = randn({200, 200}, DType::Float32, Device::cpu());
    auto param2 = randn({200, 200}, DType::Float32, Device::cpu());

    {
        ComputeContext ctx1({&param1});
        EXPECT_TRUE(isTensorOn(param1, Device::Type::CUDA));

        {
            ComputeContext ctx2({&param2});
            // Both on GPU
            EXPECT_TRUE(isTensorOn(param1, Device::Type::CUDA));
            EXPECT_TRUE(isTensorOn(param2, Device::Type::CUDA));
        }

        // param2 back on CPU, param1 still on GPU
        EXPECT_TRUE(isTensorOn(param1, Device::Type::CUDA));
        EXPECT_TRUE(isTensorOn(param2, Device::Type::CPU));
    }

    // Both back on CPU
    EXPECT_TRUE(isTensorOn(param1, Device::Type::CPU));
    EXPECT_TRUE(isTensorOn(param2, Device::Type::CPU));
}

// ==========================
// Integration Tests (3 tests)
// ==========================

TEST_F(ParameterOffloadTest, Integration_SimpleForwardPass) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 10);
    model->to(Device::cuda(0));

    OffloadContext ctx(*model, default_config);
    ctx.enable();

    // Forward pass should work with offloading
    auto input = Variable(randn({4, 128}, DType::Float32, Device::cuda(0)), true);

    EXPECT_NO_THROW({
        auto output = model->forward(input);
        EXPECT_EQ(output.shape()[0], 4);
        EXPECT_EQ(output.shape()[1], 10);
    });
}

TEST_F(ParameterOffloadTest, Integration_ForwardBackwardPass) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 10);
    model->to(Device::cuda(0));

    OffloadContext::Config config = default_config;
    config.offload_parameters = true;
    config.offload_gradients = true;

    OffloadContext ctx(*model, config);
    ctx.enable();

    // Forward and backward
    auto input = Variable(randn({8, 64}, DType::Float32, Device::cuda(0)), true);
    auto output = model->forward(input);

    EXPECT_NO_THROW({
        auto loss = sum(output);  // Use autograd sum to maintain computation graph
        loss.backward();
    });

    // Check gradients computed
    auto params = model->parameters();
    for (const auto& param : params) {
        EXPECT_TRUE(param->grad().has_value());
    }
}

TEST_F(ParameterOffloadTest, Integration_FullTrainingLoop) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    // Create model
    auto model = createTestModule(784, 256, 10);
    model->to(Device::cuda(0));

    // Enable offloading
    OffloadContext::Config config;
    config.offload_parameters = true;
    config.offload_gradients = true;
    config.offload_threshold = 0;  // Offload all parameters regardless of size
    config.prefetch_depth = 1;
    config.pin_first_layer = false;
    config.pin_last_layer = false;

    OffloadContext ctx(*model, config);
    ctx.enable();

    // Train for 10 steps
    const int num_steps = 10;
    for (int i = 0; i < num_steps; ++i) {
        auto input = Variable(randn({32, 784}, DType::Float32, Device::cuda(0)), true);
        auto output = model->forward(input);

        // Backward
        auto loss = sum(output);  // Use autograd sum
        loss.backward();

        // Check gradients exist
        auto params = model->parameters();
        for (const auto& param : params) {
            ASSERT_TRUE(param->grad().has_value());
        }

        // Zero gradients for next iteration
        model->zero_grad();
    }

    // Verify memory savings
    auto stats = ctx.get_stats();
    EXPECT_GT(stats.num_parameters_offloaded, 0);
    EXPECT_GT(stats.total_offload_count, 0);
}

// ==========================
// Performance Tests (2 tests)
// ==========================

TEST_F(ParameterOffloadTest, Performance_MemorySavings) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    // Create large model
    auto model = createTestModule(1024, 2048, 512);
    model->to(Device::cuda(0));

    auto params = model->parameters();
    float param_memory_mb = getParameterMemoryMB(params);

    // Enable offloading
    OffloadContext ctx(*model, default_config);
    ctx.enable();

    auto stats = ctx.get_stats();

    // CPU memory should roughly match parameter size
    EXPECT_GT(stats.current_cpu_memory_mb, param_memory_mb * 0.5f);

    // Should have offloaded parameters
    EXPECT_GT(stats.num_parameters_offloaded, 0);
}

TEST_F(ParameterOffloadTest, Performance_OverheadAcceptable) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(512, 1024, 256);
    model->to(Device::cuda(0));

    // Baseline: forward pass without offloading
    auto input = Variable(randn({16, 512}, DType::Float32, Device::cuda(0)), true);

    auto start_baseline = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) {
        auto output = model->forward(input);
    }
    auto end_baseline = std::chrono::high_resolution_clock::now();
    auto baseline_ms = std::chrono::duration<float, std::milli>(
        end_baseline - start_baseline).count();

    // With offloading
    OffloadContext ctx(*model, default_config);
    ctx.enable();

    auto start_offload = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < 5; ++i) {
        auto output = model->forward(input);
    }
    auto end_offload = std::chrono::high_resolution_clock::now();
    auto offload_ms = std::chrono::duration<float, std::milli>(
        end_offload - start_offload).count();

    // Overhead should be reasonable (less than 3x slowdown)
    float overhead_ratio = offload_ms / baseline_ms;
    EXPECT_LT(overhead_ratio, 3.0f);

    auto stats = ctx.get_stats();
    EXPECT_GT(stats.total_offload_count, 0);
}

// ==========================
// Edge Cases and Error Handling (3 additional tests)
// ==========================

TEST_F(ParameterOffloadTest, EdgeCase_EmptyModel) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = std::make_unique<Sequential>();

    // Should handle empty model gracefully
    EXPECT_NO_THROW({
        OffloadContext ctx(*model, default_config);
        ctx.enable();
        auto stats = ctx.get_stats();
        EXPECT_EQ(stats.num_parameters_offloaded, 0);
    });
}

TEST_F(ParameterOffloadTest, EdgeCase_AlreadyOnCPU) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 32);
    // Model already on CPU - don't move to CUDA

    OffloadContext ctx(*model, default_config);

    // Should handle CPU-only model gracefully
    EXPECT_NO_THROW({
        ctx.enable();
        auto stats = ctx.get_stats();
    });
}

TEST_F(ParameterOffloadTest, EdgeCase_MultipleEnableDisable) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 32);
    model->to(Device::cuda(0));

    OffloadContext ctx(*model, default_config);

    // Multiple enable/disable cycles
    for (int i = 0; i < 3; ++i) {
        ctx.enable();
        EXPECT_TRUE(ctx.is_enabled());

        ctx.disable();
        EXPECT_FALSE(ctx.is_enabled());
    }

    // Model should still work
    auto input = Variable(randn({4, 64}, DType::Float32, Device::cuda(0)), true);
    EXPECT_NO_THROW({
        auto output = model->forward(input);
    });
}

// ==========================
// Async offload round-trip
// ==========================

TEST_F(ParameterOffloadTest, AsyncOffloadEndsWithCommittedState) {
    // After the offload path was rewritten to issue async transfers and lazily finalize, we
    // need to confirm that:
    //   (a) enable() leaves every param fully-committed on CPU (not stuck mid-transfer);
    //   (b) get_stats() drives drain_all_pending() and reports a consistent snapshot;
    //   (c) cross-device data comparison still matches the pre-offload params bit-for-bit.
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(128, 256, 64);
    model->to(Device::cuda(0));

    auto params = model->parameters();
    std::vector<Tensor> params_before;
    params_before.reserve(params.size());
    for (const auto& p : params) {
        params_before.push_back(p->tensor().clone());  // GPU clone; we'll compare via CPU
    }

    OffloadContext ctx(*model, default_config);
    ctx.enable();  // issues async offloads internally, drains before returning

    // Every param should now live on CPU — the async TransferHandles must have all been
    // finalized by enable()'s tail drain.
    for (const auto& p : params) {
        EXPECT_EQ(p->tensor().device().type, Device::Type::CPU)
            << "After enable(), all offloaded params should sit on CPU";
    }

    // Stats reflect the committed state: count == params.size().
    auto stats = ctx.get_stats();
    EXPECT_EQ(stats.num_parameters_offloaded, params.size());

    // Bit-exact data preservation through the async offload path.
    std::vector<Tensor> params_after;
    params_after.reserve(params.size());
    for (const auto& p : params) {
        params_after.push_back(p->tensor());
    }
    EXPECT_TRUE(verifyParameterData(params_before, params_after))
        << "Async offload should preserve param data byte-for-byte";
}

// ==========================
// Audit G3: OffloadDType::Int8WithScale end-to-end.
// ==========================
//
// Previously the Int8WithScale enum value was accepted but the code path
// silently fell back to BFloat16 ("INT8 with per-tensor scale is recorded
// but the actual quant/dequant kernels need extra plumbing"). G3 wires up
// the real quantize-on-offload + dequantize-on-fetch flow.

TEST_F(ParameterOffloadTest, Int8WithScale_OffloadFetch_DataApproximatelyPreserved) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(64, 128, 32);
    model->to(Device::cuda(0));

    auto params = model->parameters();
    std::vector<Tensor> params_before;
    for (const auto& param : params) {
        params_before.push_back(param->tensor().clone());
    }

    OffloadContext::Config config = default_config;
    config.offload_dtype = OffloadContext::Config::OffloadDType::Int8WithScale;
    config.offload_threshold = 0;

    {
        // OffloadContext restores on destruction, not on disable() (which is
        // just a flag flip). Scope ctx so its destructor runs before the
        // post-check.
        OffloadContext ctx(*model, config);
        ctx.enable();
        ctx.disable();
    }  // ~OffloadContext() runs — restores Int8 cpu_copy back to Float32 GPU.

    // After round-trip via Int8 + scale, dequantized values match within
    // ~scale/2 = max(|t|)/254. Use a tolerance proportional to the largest
    // absolute parameter value seen.
    float max_abs = 0.0f;
    for (const auto& t : params_before) {
        auto host = t.to(Device::cpu()).contiguous();
        const float* d = host.data<float>();
        for (int64_t i = 0; i < host.numel(); ++i) {
            max_abs = std::max(max_abs, std::abs(d[i]));
        }
    }
    const float tol = (max_abs / 254.0f) + 1e-6f;

    for (size_t i = 0; i < params.size(); ++i) {
        auto before_cpu = params_before[i].to(Device::cpu()).contiguous();
        auto after_cpu  = params[i]->tensor().to(Device::cpu()).contiguous();
        ASSERT_EQ(before_cpu.numel(), after_cpu.numel());
        const float* b = before_cpu.data<float>();
        const float* a = after_cpu.data<float>();
        for (int64_t j = 0; j < before_cpu.numel(); ++j) {
            EXPECT_NEAR(a[j], b[j], tol)
                << "Int8 round-trip drift at param[" << i << "][" << j << "]";
        }
    }
}

TEST_F(ParameterOffloadTest, Int8WithScale_OffloadFetch_FiniteResults) {
    if (!cuda_available) SKIP_WITH_REASON(::tenzor::testing::SkipReason::BackendUnavailable, "CUDA not available");

    auto model = createTestModule(32, 64, 16);
    model->to(Device::cuda(0));

    OffloadContext::Config config = default_config;
    config.offload_dtype = OffloadContext::Config::OffloadDType::Int8WithScale;
    config.offload_threshold = 0;

    {
        OffloadContext ctx(*model, config);
        ctx.enable();
        ctx.disable();
    }  // ~OffloadContext() restores params.

    // All restored values must be finite — no NaN from a bug in the
    // quant/dequant chain (e.g. division by an all-zero scale).
    for (const auto& p : model->parameters()) {
        auto host = p->tensor().to(Device::cpu()).contiguous();
        const float* d = host.data<float>();
        for (int64_t i = 0; i < host.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(d[i]))
                << "non-finite value after Int8 round-trip at index " << i;
        }
    }
}

// ==========================
// Main
// ==========================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
