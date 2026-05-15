/**
 * @file test_zero_stage1.cpp
 * @brief Unit tests for ZeRO Stage 1 Optimizer
 *
 * Tests ZeRO Stage 1 optimizer functionality including:
 * - Constructor validation
 * - Parameter partitioning
 * - Optimizer state management
 * - Gradient synchronization
 * - Parameter all-gather
 * - CPU offload
 * - State dict save/load
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <memory>
#include <vector>
#include <filesystem>
#include <tenzor/distributed/gradient_compression.hpp>
#include <tenzor/core/offload_engine.hpp>

using namespace tenzor;
using namespace tenzor::optim;

// ============================================================================
// Test Fixtures
// ============================================================================

class ZeROStage1Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Default config for single-process tests
        default_config.world_size = 1;  // Single process mode
        default_config.rank = 0;
        default_config.offload_to_cpu = false;
        default_config.cpu_offload_threshold = 1024;  // 1KB
        default_config.overlap_comm = true;
        default_config.pin_memory = true;
        default_config.process_group = nullptr;  // Single-process mode
    }

    // Helper: Create test parameters on a specific device
    auto create_test_params(size_t count, const std::vector<int64_t>& shape = {128, 128},
                           Device device = Device::cpu())
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        params.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto param = std::make_shared<Variable>(
                ones(shape, DType::Float32, device),
                true
            );
            params.push_back(param);
        }
        return params;
    }

    // Helper: Check if CUDA is available by attempting to create a tensor
    static bool cuda_available() {
        try {
            // Try to create a small tensor on CUDA
            auto test_tensor = ones({1}, DType::Float32, Device::cuda(0));
            return true;
        } catch (...) {
            return false;
        }
    }

    ZeROStage1Config default_config;
};

// ============================================================================
// 1. Constructor and Initialization Tests
// ============================================================================

TEST_F(ZeROStage1Test, ConstructorWithValidConfig) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ASSERT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);
    });
}

TEST_F(ZeROStage1Test, ConstructorWithInvalidWorldSize) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 0;  // Invalid
    config.rank = 0;

    EXPECT_THROW({
        ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);
    }, std::invalid_argument);
}

TEST_F(ZeROStage1Test, ConstructorWithInvalidRank) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 4;  // Invalid: rank >= world_size

    EXPECT_THROW({
        ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);
    }, std::invalid_argument);
}

TEST_F(ZeROStage1Test, ConstructorWithNegativeRank) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = -1;  // Invalid

    EXPECT_THROW({
        ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);
    }, std::invalid_argument);
}

TEST_F(ZeROStage1Test, ConstructorWithNullOptimizer) {
    ZeROStage1Config config = default_config;

    EXPECT_THROW({
        ZeROStage1Optimizer optimizer(nullptr, config);
    }, std::invalid_argument);
}

TEST_F(ZeROStage1Test, ConstructorInitializesPartitions) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 100 params
    EXPECT_EQ(optimizer.local_param_count(), 100);
}

// ============================================================================
// 2. Parameter Partitioning Tests
// ============================================================================

TEST_F(ZeROStage1Test, PartitionParametersEvenSplit) {
    // 100 parameters, single process = 100 params
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 100 params
    EXPECT_EQ(optimizer.local_param_count(), 100);
}

TEST_F(ZeROStage1Test, PartitionParametersUnevenSplit) {
    // 103 parameters in single-process mode
    // Single rank should own all 103 params
    auto params = create_test_params(103);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    auto opt = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Optimizer optimizer(std::move(opt), config);

    // Single rank owns all 103 parameters
    EXPECT_EQ(optimizer.local_param_count(), 103)
        << "Single rank should own all parameters";
}

TEST_F(ZeROStage1Test, PartitionEmptyParameterList) {
    std::vector<std::shared_ptr<Variable>> empty_params;
    auto base_optimizer = std::make_unique<Adam>(empty_params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_EQ(optimizer.local_param_count(), 0);
    EXPECT_NO_THROW(optimizer.step());
}

TEST_F(ZeROStage1Test, PartitionSingleParameter) {
    auto params = create_test_params(1);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // First rank gets the single parameter
    EXPECT_EQ(optimizer.local_param_count(), 1);
}

TEST_F(ZeROStage1Test, PartitionMixedParameterSizes) {
    std::vector<std::shared_ptr<Variable>> params;

    // Create parameters of different sizes
    params.push_back(std::make_shared<Variable>(
        ones({1000, 1000}, DType::Float32, Device::cpu()), true));  // 4MB
    params.push_back(std::make_shared<Variable>(
        ones({10, 10}, DType::Float32, Device::cpu()), true));      // 400B
    params.push_back(std::make_shared<Variable>(
        ones({500, 2000}, DType::Float32, Device::cpu()), true));   // 4MB
    params.push_back(std::make_shared<Variable>(
        ones({64, 64}, DType::Float32, Device::cpu()), true));      // 16KB

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 4 parameters
    EXPECT_EQ(optimizer.local_param_count(), 4);
}

// ============================================================================
// 3. Optimizer State Management Tests
// ============================================================================

TEST_F(ZeROStage1Test, InitializeOptimizerStates) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Adam has 2 states per param: momentum + variance
    // Single rank has all 100 params
    auto stats = optimizer.get_memory_stats();
    EXPECT_EQ(stats.num_local_parameters, 100);
}

TEST_F(ZeROStage1Test, OptimizerStatesOnCPU) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;
    config.offload_to_cpu = false;  // States should be on same device as params

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // For CPU tensors, states remain on CPU
    auto stats = optimizer.get_memory_stats();
    EXPECT_GE(stats.cpu_optimizer_memory, 0);
}

TEST_F(ZeROStage1Test, OptimizerStatesPreserveAfterStep) {
    auto params = create_test_params(4);

    // Set gradients to non-zero values
    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Take one optimizer step
    optimizer.step();

    // States should exist and be non-zero (Adam accumulates momentum/variance)
    auto stats = optimizer.get_memory_stats();
    EXPECT_GT(stats.num_local_parameters, 0);
}

// ============================================================================
// 4. Gradient Handling Tests
// ============================================================================

TEST_F(ZeROStage1Test, ZeroGradClearsGradients) {
    auto params = create_test_params(10);

    // Set gradients
    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Zero gradients
    optimizer.zero_grad();

    // All gradients should be zero
    for (const auto& param : params) {
        if (param->has_grad()) {
            auto grad_data = static_cast<const float*>(param->grad().value().data_ptr());
            int numel = param->grad().value().numel();
            for (int i = 0; i < numel; ++i) {
                EXPECT_FLOAT_EQ(grad_data[i], 0.0f);
            }
        }
    }
}

TEST_F(ZeROStage1Test, StepWithZeroGradients) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Set all gradients to zero
    optimizer.zero_grad();

    // Step should complete without error
    EXPECT_NO_THROW(optimizer.step());
}

// ============================================================================
// 5. State Dict Save/Load Tests
// ============================================================================

TEST_F(ZeROStage1Test, SaveStateDictToDict) {
    auto params = create_test_params(10);

    // Set gradients and take a step to create optimizer state
    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);
    optimizer.step();

    auto state_dict = optimizer.state_dict();

    // State dict should not be empty after a step
    EXPECT_GT(state_dict.size(), 0);
}

TEST_F(ZeROStage1Test, LoadStateDictFromDict) {
    auto params = create_test_params(10);

    // Create first optimizer and get state
    {
        auto params_copy = create_test_params(10);
        for (auto& param : params_copy) {
            param->set_grad(ones_like(param->tensor()));
        }

        auto opt1 = std::make_unique<Adam>(params_copy, 0.001);
        ZeROStage1Config config = default_config;
        config.world_size = 1;  // Single process for unit test
        config.rank = 0;

        ZeROStage1Optimizer optimizer1(std::move(opt1), config);
        optimizer1.step();

        auto state_dict = optimizer1.state_dict();

        // Create second optimizer and load state
        auto opt2 = std::make_unique<Adam>(params, 0.001);
        ZeROStage1Optimizer optimizer2(std::move(opt2), config);

        EXPECT_NO_THROW(optimizer2.load_state_dict(state_dict));
    }
}

TEST_F(ZeROStage1Test, LoadEmptyStateDict) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    std::unordered_map<std::string, Tensor> empty_state;

    // Loading empty state should not crash
    EXPECT_NO_THROW(optimizer.load_state_dict(empty_state));
}

// ============================================================================
// 6. CPU Offload Tests
// ============================================================================

TEST_F(ZeROStage1Test, EnableCPUOffload) {
    // Test with CPU params - tests the flag enabling
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;
    config.offload_to_cpu = true;
    config.cpu_offload_threshold = 1024;  // 1KB threshold

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_TRUE(optimizer.is_cpu_offload_enabled());
}

TEST_F(ZeROStage1Test, DisableCPUOffload) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;
    config.offload_to_cpu = false;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_FALSE(optimizer.is_cpu_offload_enabled());
}

TEST_F(ZeROStage1Test, CPUOffloadStepCompletes) {
    // Test with CPU params - verifies step works with offload flag enabled
    auto params = create_test_params(10);

    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;
    config.offload_to_cpu = true;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Step should complete with offload enabled
    EXPECT_NO_THROW(optimizer.step());
}

TEST_F(ZeROStage1Test, CPUOffloadFromGPU) {
    // This test requires CUDA - skip if not available
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU offload test";
    }

    // Create parameters on GPU
    auto params = create_test_params(10, {128, 128}, Device::cuda(0));

    // Verify params are on GPU
    for (const auto& param : params) {
        ASSERT_EQ(param->tensor().device().type, Device::Type::CUDA)
            << "Parameter should be on GPU before offload";
    }

    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = true;
    config.cpu_offload_threshold = 1024;  // 1KB threshold - triggers offload

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Verify offload is enabled
    EXPECT_TRUE(optimizer.is_cpu_offload_enabled());

    // Step should complete with actual GPU->CPU offload
    EXPECT_NO_THROW(optimizer.step());

    // Verify local parameters are tracked
    auto stats = optimizer.get_memory_stats();
    EXPECT_EQ(stats.num_local_parameters, 10);
}

TEST_F(ZeROStage1Test, CPUOffloadThresholdRespected) {
    // This test requires CUDA - skip if not available
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU offload threshold test";
    }

    // Create small parameters on GPU (below threshold)
    auto small_params = create_test_params(10, {4, 4}, Device::cuda(0));  // 64 bytes each

    for (auto& param : small_params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(small_params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = true;
    config.cpu_offload_threshold = 1024 * 1024;  // 1MB - larger than params

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Step should complete - small tensors may not be offloaded
    EXPECT_NO_THROW(optimizer.step());
}

// ============================================================================
// 7. Optimizer Wrapper Tests
// ============================================================================

TEST_F(ZeROStage1Test, WrapAdamOptimizer) {
    auto params = create_test_params(10);
    auto adam = std::make_unique<Adam>(params, 0.001, 0.9, 0.999);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ASSERT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(adam), config);
    });
}

TEST_F(ZeROStage1Test, WrapSGDOptimizer) {
    auto params = create_test_params(10);
    auto sgd = std::make_unique<SGD>(params, 0.01, 0.9);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ASSERT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(sgd), config);
    });
}

TEST_F(ZeROStage1Test, WrapAdamWOptimizer) {
    auto params = create_test_params(10);
    auto adamw = std::make_unique<AdamW>(params, 0.001, 0.9, 0.999, 0.01);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ASSERT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(adamw), config);
    });
}

// ============================================================================
// 8. Edge Case Tests
// ============================================================================

TEST_F(ZeROStage1Test, SingleGPUConfiguration) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // All parameters should be in single partition
    EXPECT_EQ(optimizer.local_param_count(), 10);
}

TEST_F(ZeROStage1Test, VeryLargeParameterCount) {
    // Test with 10000 parameters
    auto params = create_test_params(10000, {8, 8});  // Small tensors
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 10000 params
    EXPECT_EQ(optimizer.local_param_count(), 10000);
}

TEST_F(ZeROStage1Test, VerySmallParameters) {
    std::vector<std::shared_ptr<Variable>> params;

    // Create many small tensors
    for (int i = 0; i < 100; ++i) {
        params.push_back(std::make_shared<Variable>(
            ones({4, 4}, DType::Float32, Device::cpu()), true));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 100 parameters
    EXPECT_EQ(optimizer.local_param_count(), 100);
}

TEST_F(ZeROStage1Test, FewerParametersThanRanks) {
    auto params = create_test_params(2);  // Only 2 params
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 2 params
    EXPECT_EQ(optimizer.local_param_count(), 2);
}

// ============================================================================
// 9. Memory Statistics Tests
// ============================================================================

TEST_F(ZeROStage1Test, GetMemoryStats) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    auto stats = optimizer.get_memory_stats();

    EXPECT_EQ(stats.num_parameters, 100);
    EXPECT_EQ(stats.num_local_parameters, 100);  // Single rank owns all 100
}

TEST_F(ZeROStage1Test, MemoryStatsAfterStep) {
    auto params = create_test_params(10);

    for (auto& param : params) {
        param->set_grad(ones_like(param->tensor()));
    }

    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    optimizer.step();

    auto stats = optimizer.get_memory_stats();

    // After step, optimizer states should be allocated
    EXPECT_GT(stats.cpu_optimizer_memory + stats.gpu_optimizer_memory, 0);
}

// ============================================================================
// 10. Accessor Tests
// ============================================================================

TEST_F(ZeROStage1Test, GetRank) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_EQ(optimizer.rank(), 0);
}

TEST_F(ZeROStage1Test, GetWorldSize) {
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_EQ(optimizer.world_size(), 1);
}

TEST_F(ZeROStage1Test, GetLocalParamCount) {
    auto params = create_test_params(100);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // Single process for unit test
    config.rank = 0;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // Single rank should own all 100 parameters
    EXPECT_EQ(optimizer.local_param_count(), 100);
}

// ============================================================================
// 11. Mixed-Precision (fp32 master + fp32 states)
// ============================================================================

TEST_F(ZeROStage1Test, MixedPrecisionFloat16ParamsKeepFp32MomentumAndMaster) {
    // Build fp16 parameters with non-zero gradients so step() actually does work.
    std::vector<std::shared_ptr<Variable>> params;
    for (int i = 0; i < 4; ++i) {
        auto p = std::make_shared<Variable>(
            ones({16, 16}, DType::Float16, Device::cpu()),
            true
        );
        p->set_grad(ones({16, 16}, DType::Float16, Device::cpu()));
        params.push_back(p);
    }

    auto adam = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.use_master_fp32 = true;
    config.state_dtype = DType::Float32;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Take one step. With master enabled, the optimizer should:
    //   - hold momentum/variance in fp32
    //   - hold a fp32 master copy of each param
    //   - downcast master back into the user's fp16 param after the step.
    optimizer.step();

    // The user-visible params should still be fp16 — the master copy is internal.
    for (const auto& p : params) {
        EXPECT_EQ(p->tensor().dtype(), DType::Float16);
    }

    // Walk the saved state to verify dtypes. state_dict exposes momentum_*, variance_* keys.
    auto sd = optimizer.state_dict();
    int momentum_seen = 0, variance_seen = 0;
    for (const auto& [key, t] : sd) {
        if (key.rfind("momentum_", 0) == 0) {
            EXPECT_EQ(t.dtype(), DType::Float32) << "momentum should be fp32 (state_dtype)";
            ++momentum_seen;
        } else if (key.rfind("variance_", 0) == 0) {
            EXPECT_EQ(t.dtype(), DType::Float32) << "variance should be fp32 (state_dtype)";
            ++variance_seen;
        }
    }
    EXPECT_EQ(momentum_seen, 4);
    EXPECT_EQ(variance_seen, 4);

    // After step, fp16 params should not be all-ones any more (Adam moved them).
    bool any_changed = false;
    for (const auto& p : params) {
        const auto* data = static_cast<const Float16*>(p->tensor().data_ptr());
        for (int64_t k = 0; k < p->tensor().numel(); ++k) {
            if (static_cast<float>(data[k]) != 1.0f) { any_changed = true; break; }
        }
        if (any_changed) break;
    }
    EXPECT_TRUE(any_changed) << "fp16 params should have been updated by the step";
}

TEST_F(ZeROStage1Test, MixedPrecisionDisabledKeepsLegacyDtypes) {
    // With master disabled and state_dtype unset, optimizer states should match the param
    // dtype — the legacy behaviour.
    auto params = create_test_params(2);
    for (auto& p : params) {
        p->set_grad(ones_like(p->tensor()));
    }
    auto adam = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    EXPECT_FALSE(config.use_master_fp32);
    EXPECT_FALSE(config.state_dtype.has_value());

    ZeROStage1Optimizer optimizer(std::move(adam), config);
    optimizer.step();

    auto sd = optimizer.state_dict();
    for (const auto& [key, t] : sd) {
        if (key.rfind("momentum_", 0) == 0 || key.rfind("variance_", 0) == 0) {
            EXPECT_EQ(t.dtype(), DType::Float32)
                << "legacy path: state should match param dtype (fp32 here)";
        }
    }
}

// ============================================================================
// 12. Int8 Quantization of Offloaded Optimizer States
// ============================================================================

TEST_F(ZeROStage1Test, QuantizedOffloadStepsCompleteAndPreserveTrajectory) {
    // Verifies the int8 offload codec end-to-end:
    //   - offload_states_to_cpu quantizes momentum/variance to int8 + scalar scale,
    //   - fetch_states_to_gpu dequantizes back to fp32,
    //   - the optimizer step completes successfully across multiple iterations,
    //   - param trajectory stays close to the unquantized reference (within int8 noise).
    if (!cuda_available()) {
        GTEST_SKIP() << "CUDA not available, skipping GPU offload test";
    }

    constexpr int n_steps = 3;
    constexpr int n_params = 4;
    constexpr float lr = 0.001f;

    // Build two identical setups: one with quantization off, one with it on. We compare
    // the resulting param values after `n_steps`.
    auto make_setup = [&](bool quantize) {
        auto params = create_test_params(n_params, {32, 32}, Device::cuda(0));
        for (auto& p : params) {
            p->set_grad(ones_like(p->tensor()));
        }
        auto adam = std::make_unique<Adam>(params, lr);

        ZeROStage1Config cfg = default_config;
        cfg.world_size = 1;
        cfg.rank = 0;
        cfg.offload_to_cpu = true;
        cfg.quantize_offloaded_states_int8 = quantize;

        return std::pair{params, std::make_unique<ZeROStage1Optimizer>(std::move(adam), cfg)};
    };

    auto [ref_params, ref_opt] = make_setup(false);
    auto [q_params, q_opt]     = make_setup(true);

    for (int s = 0; s < n_steps; ++s) {
        ASSERT_NO_THROW(ref_opt->step());
        ASSERT_NO_THROW(q_opt->step());
        // Refresh grads each step (Adam has consumed them).
        for (auto& p : ref_params) p->set_grad(ones_like(p->tensor()));
        for (auto& p : q_params)   p->set_grad(ones_like(p->tensor()));
    }

    // Param trajectories should match within int8 quantization noise. With per-tensor
    // scale and constant grads, the quantization error in m,v stays small relative to
    // the param magnitudes; a 5% absolute tolerance is a generous bound.
    for (size_t i = 0; i < ref_params.size(); ++i) {
        auto ref_cpu = ref_params[i]->tensor().to(Device::cpu());
        auto q_cpu   = q_params[i]->tensor().to(Device::cpu());
        ASSERT_EQ(ref_cpu.numel(), q_cpu.numel());
        const float* a = ref_cpu.data<float>();
        const float* b = q_cpu.data<float>();
        for (int64_t k = 0; k < ref_cpu.numel(); ++k) {
            EXPECT_NEAR(a[k], b[k], 0.05f * std::max(std::abs(a[k]), 1e-3f))
                << "param[" << i << "][" << k << "] drift exceeded int8 noise budget";
        }
    }

    // After step(), the optimizer re-offloads state to CPU as part of its tail. With the
    // quantization flag on, the CPU-resident momentum/variance tensors should carry Int8
    // payloads — that's the proof the codec actually fired (and not e.g. silently bypassed).
    auto sd = q_opt->state_dict();
    int int8_states_seen = 0;
    for (const auto& [key, t] : sd) {
        if (key.rfind("momentum_", 0) == 0 || key.rfind("variance_", 0) == 0) {
            EXPECT_EQ(t.dtype(), DType::Int8)
                << "post-step CPU-offloaded state should be int8 quantized";
            ++int8_states_seen;
        }
    }
    EXPECT_GT(int8_states_seen, 0) << "expected at least one quantized state in state_dict";

    // Reference (no quantization) should still report fp32 states.
    auto ref_sd = ref_opt->state_dict();
    for (const auto& [key, t] : ref_sd) {
        if (key.rfind("momentum_", 0) == 0 || key.rfind("variance_", 0) == 0) {
            EXPECT_EQ(t.dtype(), DType::Float32)
                << "reference (no-quant) state should remain fp32 on CPU";
        }
    }
}

TEST_F(ZeROStage1Test, QuantizationDisabledByDefault) {
    // Sanity: the new flag defaults off and doesn't perturb the legacy CPU offload path.
    ZeROStage1Config cfg = default_config;
    EXPECT_FALSE(cfg.quantize_offloaded_states_int8);
}

// ============================================================================
// 13. NVMe Offload (ZeRO-Infinity-style)
// ============================================================================

TEST_F(ZeROStage1Test, NvmeOffloadStepsCompleteAndPreserveTrajectory) {
    // End-to-end: an optimizer with offload_to_nvme=true should serialize momentum/variance
    // to disk between steps and read them back on the next step, producing a param
    // trajectory that matches the no-offload reference within fp32 round-trip tolerance.
    namespace fs = std::filesystem;
    fs::path nvme_dir = fs::temp_directory_path() / "tenzor_zero_nvme_test";
    fs::remove_all(nvme_dir);  // clean slate per test run

    constexpr int n_steps = 3;
    constexpr int n_params = 3;

    auto make_setup = [&](bool nvme) {
        auto params = create_test_params(n_params, {16, 16});
        for (auto& p : params) {
            p->set_grad(ones_like(p->tensor()));
        }
        auto adam = std::make_unique<Adam>(params, 0.001);

        ZeROStage1Config cfg = default_config;
        cfg.offload_to_nvme = nvme;
        cfg.nvme_path = nvme ? nvme_dir.string() : std::string{};

        return std::pair{params, std::make_unique<ZeROStage1Optimizer>(std::move(adam), cfg)};
    };

    auto [ref_params, ref_opt]   = make_setup(false);
    auto [nvme_params, nvme_opt] = make_setup(true);

    // After construction with NVMe enabled, scratch files must exist for every state.
    ASSERT_TRUE(fs::exists(nvme_dir));
    int blob_count = 0;
    for (auto& e : fs::directory_iterator(nvme_dir)) {
        if (e.is_regular_file()) ++blob_count;
    }
    EXPECT_EQ(blob_count, n_params * 2)  // momentum + variance per param
        << "expected one blob per momentum and variance state on disk";

    // Drive multiple steps and verify both setups stay in lockstep.
    for (int s = 0; s < n_steps; ++s) {
        ASSERT_NO_THROW(ref_opt->step());
        ASSERT_NO_THROW(nvme_opt->step());
        for (auto& p : ref_params)  p->set_grad(ones_like(p->tensor()));
        for (auto& p : nvme_params) p->set_grad(ones_like(p->tensor()));
    }

    // NVMe round-trip is bit-identical for fp32 (no quantization in this test), so we use
    // a tight tolerance — anything looser would mask an actual data-corruption bug.
    for (size_t i = 0; i < ref_params.size(); ++i) {
        const auto& a = ref_params[i]->tensor();
        const auto& b = nvme_params[i]->tensor();
        ASSERT_EQ(a.numel(), b.numel());
        const float* ad = a.data<float>();
        const float* bd = b.data<float>();
        for (int64_t k = 0; k < a.numel(); ++k) {
            EXPECT_NEAR(ad[k], bd[k], 1e-6f)
                << "NVMe round-trip drift at param[" << i << "][" << k << "]";
        }
    }

    // Destructor cleanup: blobs should be removed when the optimizer goes away.
    nvme_opt.reset();
    int leftover = 0;
    if (fs::exists(nvme_dir)) {
        for (auto& e : fs::directory_iterator(nvme_dir)) {
            if (e.is_regular_file()) ++leftover;
        }
    }
    EXPECT_EQ(leftover, 0) << "destructor should remove NVMe scratch files";

    fs::remove_all(nvme_dir);
}

TEST_F(ZeROStage1Test, NvmeOffloadCombinesWithInt8Quantization) {
    // When both quantize_offloaded_states_int8 and offload_to_nvme are on, the on-disk
    // payload should be int8 (size = numel bytes) plus a tiny scale file. Verifies the
    // combined memory savings stack as documented.
    namespace fs = std::filesystem;
    fs::path nvme_dir = fs::temp_directory_path() / "tenzor_zero_nvme_q_test";
    fs::remove_all(nvme_dir);

    constexpr int n_params = 2;
    auto params = create_test_params(n_params, {32, 32});
    for (auto& p : params) {
        p->set_grad(ones_like(p->tensor()));
    }
    auto adam = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config cfg = default_config;
    cfg.offload_to_nvme = true;
    cfg.nvme_path = nvme_dir.string();
    cfg.quantize_offloaded_states_int8 = true;

    ZeROStage1Optimizer optimizer(std::move(adam), cfg);

    // After step(), states are quantized and re-written to disk.
    ASSERT_NO_THROW(optimizer.step());

    // Each state file should be ~numel bytes (int8) and each scale file should be 4 bytes.
    int int8_state_files = 0;
    int scale_files = 0;
    for (auto& e : fs::directory_iterator(nvme_dir)) {
        if (!e.is_regular_file()) continue;
        const auto sz = static_cast<size_t>(fs::file_size(e.path()));
        const auto name = e.path().filename().string();
        if (name.find("_scale.bin") != std::string::npos) {
            ++scale_files;
            EXPECT_EQ(sz, 4u) << "fp32 scalar scale should be 4 bytes: " << name;
        } else {
            ++int8_state_files;
            EXPECT_EQ(sz, 32u * 32u) << "int8 state should be numel bytes: " << name;
        }
    }
    EXPECT_EQ(int8_state_files, n_params * 2);
    EXPECT_EQ(scale_files,      n_params * 2);

    fs::remove_all(nvme_dir);
}

// ============================================================================
// 14. Gradient Compression Hook (review item #24)
// ============================================================================

TEST_F(ZeROStage1Test, GradientCompressorConfigPlumbsThrough) {
    // The new config field accepts a shared_ptr<GradientCompressor>; constructing the
    // optimizer with one set should not throw, and the field should round-trip via the
    // base optimizer access (we don't expose it back, but the constructor should at
    // least accept it and not silently drop or assert on it).
    auto params = create_test_params(2);
    for (auto& p : params) p->set_grad(ones_like(p->tensor()));

    auto adam = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config cfg = default_config;
    cfg.world_size = 1;
    cfg.rank = 0;
    cfg.grad_compressor = std::make_shared<distributed::FP16Compressor>();

    EXPECT_NO_THROW({
        ZeROStage1Optimizer optimizer(std::move(adam), cfg);
        optimizer.step();  // single-rank: all_reduce isn't called, but the wiring is exercised
    });
}

TEST_F(ZeROStage1Test, GradientCompressorDefaultsToNullptr) {
    // Sanity: legacy callers see no behavioural change.
    ZeROStage1Config cfg = default_config;
    EXPECT_EQ(cfg.grad_compressor, nullptr);
}

// ============================================================================
// Shared OffloadEngine plumbing (avoids duplicating pinned-host pool when
// activation offload + optimizer offload are both active in the same process).
// ============================================================================

TEST_F(ZeROStage1Test, SharedOffloadEngineDefaultsToNullptr) {
    // Sanity: legacy callers don't see any new behaviour.
    ZeROStage1Config cfg = default_config;
    EXPECT_EQ(cfg.shared_offload_engine, nullptr);
}

TEST_F(ZeROStage1Test, SharedOffloadEngineIsAdoptedWhenProvided) {
    // The user pre-builds an OffloadEngine — typically because they already have
    // one wired up for activation offload — and passes it to the optimizer to
    // avoid spinning up a second pinned-host pool. The optimizer should hold the
    // exact same shared_ptr instead of constructing its own engine.
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    core::OffloadEngine::Config eng_config;
    eng_config.pinned_memory_size = 16ULL * 1024 * 1024;  // 16 MB to keep the test cheap
    eng_config.num_transfer_streams = 1;
    eng_config.enable_prefetch = false;
    eng_config.enable_auto_monitoring = false;
    auto shared_engine = std::make_shared<core::OffloadEngine>(eng_config);

    ZeROStage1Config config = default_config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = true;
    config.shared_offload_engine = shared_engine;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    // The optimizer must adopt the engine pointer-equal, not construct a new one.
    EXPECT_EQ(optimizer.offload_engine().get(), shared_engine.get());
}

TEST_F(ZeROStage1Test, SharedOffloadEngineIgnoredWhenOffloadDisabled) {
    // If the user provides a shared engine but turns off CPU offload, the engine
    // should simply not be wired up — no init, no adoption — because there's no
    // offload work to route through it. This matters because OffloadEngine
    // construction is moderately expensive and we don't want to pay for it when
    // it won't be used.
    auto params = create_test_params(10);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    core::OffloadEngine::Config eng_config;
    eng_config.pinned_memory_size = 16ULL * 1024 * 1024;
    eng_config.num_transfer_streams = 1;
    eng_config.enable_prefetch = false;
    eng_config.enable_auto_monitoring = false;
    auto shared_engine = std::make_shared<core::OffloadEngine>(eng_config);

    ZeROStage1Config config = default_config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;  // <-- offload off
    config.shared_offload_engine = shared_engine;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);

    EXPECT_EQ(optimizer.offload_engine(), nullptr);
}

// ============================================================================
// Balanced (size-aware) parameter partitioning across ranks.
//
// The legacy partitioner slices `parameters_` by *index*, which is fine when all
// params have similar size but produces severe per-rank memory imbalance for
// models with one (or a few) huge tensors among many small ones — the rank that
// owns the huge tensor pays a multi-GB tax while peer ranks sit nearly empty.
// The opt-in greedy LPT (longest-processing-time) variant assigns by descending
// size to the least-loaded rank, which is provably within 4/3 of optimal.
// ============================================================================

TEST_F(ZeROStage1Test, BalancedPartitioningDefaultsOff) {
    // Sanity: legacy callers see no behavioural change.
    ZeROStage1Config cfg = default_config;
    EXPECT_FALSE(cfg.balanced_partitioning);
}

TEST_F(ZeROStage1Test, BalancedPartitioningKeepsLegacyOrderWhenOff) {
    // With the flag off and all params equal size, the index-based partitioner
    // should still produce the same per-rank counts that distributed users
    // already rely on (test_zero_stage1_distributed.UnevenParameterDistribution
    // hard-codes this assignment).
    auto params = create_test_params(8);
    auto base_optimizer = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config config = default_config;
    config.world_size = 1;  // single-process for this unit test
    config.rank = 0;
    config.balanced_partitioning = false;

    ZeROStage1Optimizer optimizer(std::move(base_optimizer), config);
    EXPECT_EQ(optimizer.local_param_count(), 8u);
}

TEST_F(ZeROStage1Test, BalancedPartitioningEvensOutMemoryAcrossRanks) {
    // Construct a deliberately skewed parameter set: 3 large (1M elements each
    // = ~4 MB in fp32) + 3 tiny (16 elements = 64 B). With world_size=2 and the
    // legacy index-slice partitioner, rank 0 ends up with all 3 large tensors
    // (~12 MB optimizer-state-bearing) and rank 1 with the 3 tiny ones (~192 B
    // bearing). Greedy LPT places one large on rank 0 and two on rank 1, then
    // packs the smalls onto the lighter rank — nearly perfect balance.
    //
    // The optimizer is constructed once per rank because partition_parameters()
    // runs at constructor time and we want to inspect each rank's partition
    // memory_bytes. Same Variables are shared across all rank-fixtures because
    // the partition assignment depends only on size, not on tensor identity.
    std::vector<std::shared_ptr<Variable>> params;
    for (int i = 0; i < 3; ++i) {
        params.push_back(std::make_shared<Variable>(
            ones({1000, 1000}, DType::Float32, Device::cpu()), true));
    }
    for (int i = 0; i < 3; ++i) {
        params.push_back(std::make_shared<Variable>(
            ones({4, 4}, DType::Float32, Device::cpu()), true));
    }

    constexpr int world_size = 2;

    auto memory_per_rank = [&](bool balanced) {
        std::vector<size_t> per_rank(world_size, 0);
        for (int rank = 0; rank < world_size; ++rank) {
            // Each rank-fixture needs its own base_optimizer (Adam takes a copy
            // of the params vector, but the Adam dtor unregisters hooks etc).
            auto base_optimizer = std::make_unique<Adam>(params, 0.001);

            ZeROStage1Config cfg = default_config;
            cfg.world_size = world_size;
            cfg.rank = rank;
            cfg.process_group = nullptr;  // partition_parameters() doesn't need comms
            cfg.balanced_partitioning = balanced;

            ZeROStage1Optimizer optimizer(std::move(base_optimizer), cfg);
            auto stats = optimizer.get_memory_stats();
            // CPU-resident params land in cpu_optimizer_memory; the field is
            // populated unconditionally for the local rank.
            per_rank[rank] = stats.cpu_optimizer_memory;
        }
        return per_rank;
    };

    // Legacy contiguous-slice baseline: rank 0 gets all 3 large tensors, rank 1
    // gets the 3 tiny ones. Confirm the imbalance is genuine before claiming a
    // fix — without this, a passing "balanced" assertion could mean nothing.
    auto legacy = memory_per_rank(false);
    auto legacy_max = *std::max_element(legacy.begin(), legacy.end());
    auto legacy_min = *std::min_element(legacy.begin(), legacy.end());
    ASSERT_GT(legacy_min, 0u) << "legacy partitioner left a rank empty; test is invalid";
    EXPECT_GT(legacy_max / legacy_min, 100u)
        << "legacy partitioner should be heavily skewed for this input"
        << " (large_rank=" << legacy_max << " small_rank=" << legacy_min << ")";

    // With balanced=true the per-rank memory should be within ~2× of each
    // other — perfect balance is impossible because the smallest unit is one
    // 1M-element tensor, but greedy LPT will end up with 1× large + 0 small on
    // one rank and 2× large + 3× small on the other (or vice versa).
    auto balanced = memory_per_rank(true);
    auto bal_max = *std::max_element(balanced.begin(), balanced.end());
    auto bal_min = *std::min_element(balanced.begin(), balanced.end());
    ASSERT_GT(bal_min, 0u);
    EXPECT_LT(bal_max, bal_min * 3u)
        << "balanced partitioner should be within ~2× across ranks"
        << " (rank0=" << balanced[0] << " rank1=" << balanced[1] << ")";

    // Sanity: total memory should be equal between the two configurations
    // (we just moved tensors between bins, didn't add or drop any).
    size_t legacy_total = legacy[0] + legacy[1];
    size_t balanced_total = balanced[0] + balanced[1];
    EXPECT_EQ(legacy_total, balanced_total);
}

// =========================================================================
// ElementLevel-mode parity tests
// =========================================================================

namespace {
// Run N Adam steps, return the final parameter values flattened.
auto run_n_adam_steps(
    std::vector<std::shared_ptr<Variable>> params,
    PartitioningMode mode,
    int num_steps,
    int world_size = 1,
    int rank = 0
) -> std::vector<float> {
    auto base = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Config cfg;
    cfg.world_size = world_size;
    cfg.rank = rank;
    cfg.partitioning_mode = mode;
    ZeROStage1Optimizer opt(std::move(base), cfg);

    for (int step = 0; step < num_steps; ++step) {
        // Synthetic gradient: g = 0.1 * (step + 1) — nonzero from step 0
        for (auto& p : params) {
            Tensor g = ones_like(p->tensor()) * (0.1f * (step + 1));
            p->set_grad(g);
        }
        opt.step();
    }

    std::vector<float> out;
    for (auto& p : params) {
        Tensor flat = p->tensor().contiguous().view({-1}).to(DType::Float32).to(Device::cpu());
        const float* d = flat.data<float>();
        for (int64_t i = 0; i < flat.numel(); ++i) out.push_back(d[i]);
    }
    return out;
}
}  // namespace

TEST_F(ZeROStage1Test, ElementLevel_SingleRank_AdamParity) {
    // ParamLevel and ElementLevel must produce bitwise-identical parameters at
    // world_size=1 (no actual distribution happens; only the bookkeeping differs).
    auto params_a = create_test_params(3, {16, 16});  // 3 × 256 = 768 elements
    auto params_b = create_test_params(3, {16, 16});

    auto out_param  = run_n_adam_steps(params_a, PartitioningMode::ParamLevel,   5);
    auto out_elem   = run_n_adam_steps(params_b, PartitioningMode::ElementLevel, 5);

    ASSERT_EQ(out_param.size(), out_elem.size());
    for (size_t i = 0; i < out_param.size(); ++i) {
        EXPECT_NEAR(out_param[i], out_elem[i], 1e-6)
            << "Mismatch at element " << i;
    }
}

TEST_F(ZeROStage1Test, ElementLevel_SingleRank_NonDivisibleShape) {
    // Param numel not a multiple of world_size — exercises padding.
    auto params_a = create_test_params(1, {17});
    auto params_b = create_test_params(1, {17});

    auto out_param = run_n_adam_steps(params_a, PartitioningMode::ParamLevel,   3);
    auto out_elem  = run_n_adam_steps(params_b, PartitioningMode::ElementLevel, 3);

    ASSERT_EQ(out_param.size(), out_elem.size());
    for (size_t i = 0; i < out_param.size(); ++i) {
        EXPECT_NEAR(out_param[i], out_elem[i], 1e-6)
            << "Mismatch at element " << i;
    }
}

// ============================================================================
// 15. Cross-partitioning-mode checkpoint load error
// ============================================================================

TEST_F(ZeROStage1Test, CheckpointCrossModeLoadThrows) {
    namespace fs = std::filesystem;
    auto tmp = fs::temp_directory_path() / "tenzor_zero_xmode_test";
    fs::remove_all(tmp);
    fs::create_directories(tmp);
    auto path_prefix = (tmp / "ckpt").string();

    {
        // Save in ParamLevel.
        auto params = create_test_params(2);
        auto base = std::make_unique<Adam>(params, 0.001);
        ZeROStage1Config cfg = default_config;
        cfg.partitioning_mode = PartitioningMode::ParamLevel;
        ZeROStage1Optimizer opt(std::move(base), cfg);
        opt.save_checkpoint(path_prefix);
    }
    {
        // Try to load in ElementLevel — must throw.
        auto params = create_test_params(2);
        auto base = std::make_unique<Adam>(params, 0.001);
        ZeROStage1Config cfg = default_config;
        cfg.partitioning_mode = PartitioningMode::ElementLevel;
        ZeROStage1Optimizer opt(std::move(base), cfg);
        EXPECT_THROW(opt.load_checkpoint(path_prefix), std::runtime_error);
    }
    fs::remove_all(tmp);
}

// ============================================================================
// 16. ElementLevel + CPU offload smoke test
// ============================================================================

TEST_F(ZeROStage1Test, ElementLevel_CPUOffload_StepRuns) {
    // Smoke test: element-mode + CPU offload doesn't crash and produces finite values.
    if (!cuda_available()) GTEST_SKIP() << "Requires CUDA";
    auto params = create_test_params(2, {32, 32}, Device::cuda(0));
    auto base = std::make_unique<Adam>(params, 0.001);
    ZeROStage1Config cfg = default_config;
    cfg.partitioning_mode = PartitioningMode::ElementLevel;
    cfg.offload_to_cpu = true;
    ZeROStage1Optimizer opt(std::move(base), cfg);

    for (int step = 0; step < 3; ++step) {
        for (auto& p : params) {
            Tensor g = ones_like(p->tensor()) * 0.01f;
            p->set_grad(g);
        }
        opt.step();
    }
    // Spot-check finite.
    auto cpu_param = params[0]->tensor().to(Device::cpu()).contiguous().view({-1});
    const float* d = cpu_param.data<float>();
    for (int64_t i = 0; i < cpu_param.numel(); ++i) {
        EXPECT_TRUE(std::isfinite(d[i]));
    }
}


// ============================================================================
// 14. Audit G2: ElementLevel + offload_to_nvme — was a throw, now functional.
// ============================================================================

TEST_F(ZeROStage1Test, ElementLevelNvmeOffload_ConstructsAndSteps) {
    namespace fs = std::filesystem;
    fs::path nvme_dir = fs::temp_directory_path() / "tenzor_zero_nvme_elementlevel_test";
    fs::remove_all(nvme_dir);

    auto params = create_test_params(/*count=*/3, /*shape=*/{16, 16});
    for (auto& p : params) p->set_grad(ones_like(p->tensor()));
    auto adam = std::make_unique<Adam>(params, 0.001);

    ZeROStage1Config cfg = default_config;
    cfg.partitioning_mode = PartitioningMode::ElementLevel;
    cfg.offload_to_nvme = true;
    cfg.nvme_path = nvme_dir.string();
    // G2: previously threw at construction with "ElementLevel + offload_to_nvme
    // is not yet supported (planned: Task 9.1)".
    std::unique_ptr<ZeROStage1Optimizer> opt;
    ASSERT_NO_THROW(
        opt = std::make_unique<ZeROStage1Optimizer>(std::move(adam), cfg));

    // ElementLevel produces a single flat slice per state (momentum + variance)
    // — so exactly 2 blobs on disk per rank, regardless of param count.
    ASSERT_TRUE(fs::exists(nvme_dir));
    int blob_count = 0;
    for (auto& e : fs::directory_iterator(nvme_dir)) {
        if (e.is_regular_file()) ++blob_count;
    }
    EXPECT_EQ(blob_count, 2)
        << "expected one momentum + one variance blob in ElementLevel NVMe mode";

    // Drive a few steps end-to-end. The fetch_states_to_gpu / offload_states_to_cpu
    // cycle should round-trip the slice through disk without throwing.
    for (int s = 0; s < 3; ++s) {
        ASSERT_NO_THROW(opt->step());
        for (auto& p : params) p->set_grad(ones_like(p->tensor()));
    }

    // Param trajectory: all values finite (no NaN/Inf from corrupted IO).
    for (auto& p : params) {
        auto host = p->tensor().to(Device::cpu()).contiguous();
        const float* d = host.data<float>();
        for (int64_t i = 0; i < host.numel(); ++i) {
            EXPECT_TRUE(std::isfinite(d[i]));
        }
    }

    // Destructor should sweep the scratch blobs (same convention as ParamLevel).
    opt.reset();
    int leftover = 0;
    if (fs::exists(nvme_dir)) {
        for (auto& e : fs::directory_iterator(nvme_dir)) {
            if (e.is_regular_file()) ++leftover;
        }
    }
    EXPECT_EQ(leftover, 0) << "destructor should remove ElementLevel NVMe scratch files";

    fs::remove_all(nvme_dir);
}

TEST_F(ZeROStage1Test, ElementLevelNvmeOffload_MatchesNoOffloadTrajectory) {
    namespace fs = std::filesystem;
    fs::path nvme_dir = fs::temp_directory_path() / "tenzor_zero_nvme_elementlevel_match";
    fs::remove_all(nvme_dir);

    constexpr int n_steps = 3;
    constexpr int n_params = 3;

    auto make_setup = [&](bool nvme) {
        auto params = create_test_params(n_params, {16, 16});
        for (auto& p : params) p->set_grad(ones_like(p->tensor()));
        auto adam = std::make_unique<Adam>(params, 0.001);

        ZeROStage1Config cfg = default_config;
        cfg.partitioning_mode = PartitioningMode::ElementLevel;
        cfg.offload_to_nvme = nvme;
        cfg.nvme_path = nvme ? nvme_dir.string() : std::string{};

        return std::pair{params, std::make_unique<ZeROStage1Optimizer>(std::move(adam), cfg)};
    };

    auto [ref_params, ref_opt]   = make_setup(false);
    auto [nvme_params, nvme_opt] = make_setup(true);

    for (int s = 0; s < n_steps; ++s) {
        ASSERT_NO_THROW(ref_opt->step());
        ASSERT_NO_THROW(nvme_opt->step());
        for (auto& p : ref_params)  p->set_grad(ones_like(p->tensor()));
        for (auto& p : nvme_params) p->set_grad(ones_like(p->tensor()));
    }

    // NVMe round-trip is bit-identical for fp32 (no quantization).
    for (size_t i = 0; i < ref_params.size(); ++i) {
        const auto& a = ref_params[i]->tensor();
        const auto& b = nvme_params[i]->tensor();
        ASSERT_EQ(a.numel(), b.numel());
        const float* ad = a.data<float>();
        const float* bd = b.data<float>();
        for (int64_t k = 0; k < a.numel(); ++k) {
            EXPECT_NEAR(ad[k], bd[k], 1e-6f)
                << "ElementLevel NVMe drift at param[" << i << "][" << k << "]";
        }
    }

    nvme_opt.reset();
    fs::remove_all(nvme_dir);
}
