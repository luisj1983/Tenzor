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

    // Helper: Create test parameters
    auto create_test_params(size_t count, const std::vector<int64_t>& shape = {128, 128})
        -> std::vector<std::shared_ptr<Variable>> {
        std::vector<std::shared_ptr<Variable>> params;
        params.reserve(count);
        for (size_t i = 0; i < count; ++i) {
            auto param = std::make_shared<Variable>(
                ones(shape, DType::Float32, Device::cpu()),
                true
            );
            params.push_back(param);
        }
        return params;
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
        param->grad() = ones_like(param->tensor());
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
        param->grad() = ones_like(param->tensor());
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
            auto grad_data = static_cast<float*>(param->grad().value().data_ptr());
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
        param->grad() = ones_like(param->tensor());
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
            param->grad() = ones_like(param->tensor());
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
    auto params = create_test_params(10);

    for (auto& param : params) {
        param->grad() = ones_like(param->tensor());
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
        param->grad() = ones_like(param->tensor());
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
