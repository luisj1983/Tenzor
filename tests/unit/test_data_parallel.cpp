/**
 * @file test_data_parallel.cpp
 * @brief Comprehensive unit tests for DataParallel multi-GPU training
 *
 * RR.17 (audit-11): the 26 raw GTEST-SKIP sites with "CUDA not available..."
 * sites that used to open every test body were promoted to a single
 * SetUp()-level `SKIP_WITH_REASON(BackendUnavailable, ...)` on a
 * BackendTest-derived fixture. Each test still runs CUDA-only (DataParallel
 * itself is CUDA-specific), but the skip is now tagged with the
 * structured skip-reason taxonomy so `scripts/count_skips.py` can attribute
 * the skips correctly. Tests that additionally need ≥2 CUDA devices use
 * `SkipReason::RequiresMultiGPU` from the per-test guard.
 */
#include <gtest/gtest.h>
#include <memory>
#include <vector>
#include "tenzor/nn/parallel/data_parallel.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"
#include "../multi_backend_dtype_fixture.hpp"

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

using namespace tenzor;
using namespace tenzor::nn;
using ::tenzor::testing::BackendTest;
using ::tenzor::testing::SkipReason;

// ============================================================================
// Mock Module for Testing
// ============================================================================

/**
 * @brief Simple test module that doubles its input
 */
class DoubleModule : public Module {
public:
    DoubleModule() = default;

    auto forward_impl(const Variable& input) -> Variable override {
        // Simply multiply input by 2
        return Variable(input.tensor() * 2.0f);
    }

    auto parameters() -> std::vector<std::shared_ptr<Variable>> override {
        return {};  // No trainable parameters
    }

    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override {
        return {};
    }
};

/**
 * @brief Module with parameters for gradient testing
 */
class SimpleLinearModule : public Module {
public:
    SimpleLinearModule(int in_features, int out_features) {
        // Create weight and bias parameters
        auto weight_tensor = tenzor::randn({out_features, in_features});
        auto bias_tensor = tenzor::zeros({out_features});

        weight_ = Variable(weight_tensor, true);
        bias_ = Variable(bias_tensor, true);

        register_parameter("weight", weight_);
        register_parameter("bias", bias_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        // Simple linear transformation: y = xW^T + b
        auto output = input.tensor() * weight_.tensor();
        return Variable(output + bias_.tensor());
    }

private:
    Variable weight_;
    Variable bias_;
};

// ============================================================================
// Helper Functions
// ============================================================================

/**
 * @brief Get number of available CUDA devices
 */
static int get_device_count() {
#ifdef TENZOR_USE_CUDA
    int device_count = 0;
    cudaGetDeviceCount(&device_count);
    return device_count;
#else
    return 0;
#endif
}

// ============================================================================
// DataParallel fixture
// ============================================================================
//
// DataParallel itself is a CUDA-only construct (it dispatches replicas across
// CUDA device IDs). The fixture inherits BackendTest so all the standard env
// var handling (TENZOR_SKIP_BACKENDS, TENZOR_REQUIRE_MULTI_BACKEND) works,
// and we tag the "no CUDA at all" skip with the BackendUnavailable reason so
// scripts/count_skips.py can attribute it correctly.
//
// Per-test SkipMultiGPU() helper handles the additional skip for the four
// scatter/multi-device tests that require ≥2 CUDA devices.
class DataParallelMultiBackendTest : public BackendTest {
protected:
    void SetUp() override {
        // Run BackendTest::SetUp() first so env-var skips fire correctly.
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        if (device.type != Device::Type::CUDA) {
            // DataParallel only works on CUDA. Mark non-CUDA params as
            // unsupported with the structured reason.
            SKIP_WITH_REASON(SkipReason::BackendUnavailable,
                "DataParallel is CUDA-only");
            return;
        }
        if (get_device_count() <= 0) {
            SKIP_WITH_REASON(SkipReason::BackendUnavailable,
                "CUDA not available");
            return;
        }
    }

    // Helper: per-test extra skip for cases needing ≥2 CUDA devices.
    void SkipUnlessMultiGPU() {
        if (get_device_count() < 2) {
            SKIP_WITH_REASON(SkipReason::RequiresMultiGPU,
                "DataParallel scatter test needs ≥2 CUDA devices");
        }
    }
};

// ============================================================================
// Construction Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, ConstructionWithValidDevices) {
    auto module = std::make_shared<DoubleModule>();

    // Test construction with explicit device IDs
    EXPECT_NO_THROW({
        DataParallel dp(module, {0}, 0);
    });
}

TEST_P(DataParallelMultiBackendTest, ConstructionWithNullModule) {
    // Should throw when module is null
    EXPECT_THROW({
        DataParallel dp(nullptr, {0}, 0);
    }, std::invalid_argument);
}

TEST_P(DataParallelMultiBackendTest, ConstructionWithEmptyDeviceIdsAutoDetect) {
    auto module = std::make_shared<DoubleModule>();

    // Empty device_ids should auto-detect available devices
    EXPECT_NO_THROW({
        DataParallel dp(module, {}, -1);
        EXPECT_GT(dp.device_ids().size(), 0);
    });
}

TEST_P(DataParallelMultiBackendTest, ConstructionWithInvalidDeviceId) {
    auto module = std::make_shared<DoubleModule>();

    // Invalid device ID should throw
    EXPECT_THROW({
        DataParallel dp(module, {999}, 999);
    }, std::invalid_argument);
}

TEST_P(DataParallelMultiBackendTest, ConstructionOutputDeviceNotInList) {
    auto module = std::make_shared<DoubleModule>();

    // Output device must be in device_ids
    EXPECT_THROW({
        DataParallel dp(module, {0}, 1);
    }, std::invalid_argument);
}

TEST_P(DataParallelMultiBackendTest, ConstructionDefaultOutputDevice) {
    auto module = std::make_shared<DoubleModule>();

    // Default output device should be device_ids[0]
    DataParallel dp(module, {0}, -1);
    EXPECT_EQ(dp.output_device(), 0);
}

// ============================================================================
// Single GPU Optimization Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, SingleGPUOptimization) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Create test input
    auto input_tensor = tenzor::ones({4, 10});
    Variable input(input_tensor);

    // Forward pass with single GPU should work
    EXPECT_NO_THROW({
        auto output = dp.forward(input);

        // Output should be doubled (2.0)
        auto output_data = output.tensor().data<float>();
        for (int i = 0; i < 40; ++i) {
            EXPECT_FLOAT_EQ(output_data[i], 2.0f);
        }
    });
}

// ============================================================================
// Scatter Operation Tests (require ≥2 CUDA devices)
// ============================================================================

TEST_P(DataParallelMultiBackendTest, ScatterEvenBatchSize) {
    SkipUnlessMultiGPU();
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0, 1}, 0);

    // Batch size divisible by number of devices
    auto input_tensor = tenzor::ones({8, 10});
    Variable input(input_tensor);

    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 8);
        EXPECT_EQ(output.tensor().shape()[1], 10);
    });
}

TEST_P(DataParallelMultiBackendTest, ScatterUnevenBatchSize) {
    SkipUnlessMultiGPU();
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0, 1}, 0);

    // Batch size not divisible by number of devices
    auto input_tensor = tenzor::ones({7, 10});
    Variable input(input_tensor);

    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 7);
        EXPECT_EQ(output.tensor().shape()[1], 10);
    });
}

TEST_P(DataParallelMultiBackendTest, ScatterBatchSizeTooSmall) {
    SkipUnlessMultiGPU();
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0, 1}, 0);

    // Batch size smaller than number of devices should throw
    auto input_tensor = tenzor::ones({1, 10});
    Variable input(input_tensor);

    EXPECT_THROW({
        auto output = dp.forward(input);
    }, std::runtime_error);
}

// ============================================================================
// Gather Operation Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, GatherConcatenatesCorrectly) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Create input with known values
    auto input_tensor = tenzor::ones({4, 3});
    auto input_data = input_tensor.data<float>();
    for (int i = 0; i < 12; ++i) {
        input_data[i] = static_cast<float>(i);
    }

    Variable input(input_tensor);

    auto output = dp.forward(input);
    auto output_data = output.tensor().data<float>();

    // Check that values are doubled
    for (int i = 0; i < 12; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], static_cast<float>(i * 2));
    }
}

// ============================================================================
// Device Management Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, DeviceIDsAccessor) {
    auto module = std::make_shared<DoubleModule>();
    std::vector<int> device_ids = {0};
    DataParallel dp(module, device_ids, 0);

    auto returned_ids = dp.device_ids();
    EXPECT_EQ(returned_ids, device_ids);
}

TEST_P(DataParallelMultiBackendTest, OutputDeviceAccessor) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    EXPECT_EQ(dp.output_device(), 0);
}

TEST_P(DataParallelMultiBackendTest, BatchDimAccessor) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0, 1);  // Use dimension 1 as batch dim

    EXPECT_EQ(dp.batch_dim(), 1);
}

// ============================================================================
// Parameter Management Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, ParametersFromMasterModule) {
    auto module = std::make_shared<SimpleLinearModule>(10, 5);
    DataParallel dp(module, {0}, 0);

    // Parameters should come from master module
    auto params = dp.parameters();
    auto master_params = module->parameters();

    EXPECT_EQ(params.size(), master_params.size());
}

TEST_P(DataParallelMultiBackendTest, NamedParametersFromMasterModule) {
    auto module = std::make_shared<SimpleLinearModule>(10, 5);
    DataParallel dp(module, {0}, 0);

    auto named_params = dp.named_parameters();
    auto master_named_params = module->named_parameters();

    EXPECT_EQ(named_params.size(), master_named_params.size());
}

TEST_P(DataParallelMultiBackendTest, ModuleAccessor) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Should return the original module
    EXPECT_EQ(dp.module(), module);
}

// ============================================================================
// Training Mode Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, TrainMode) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    EXPECT_NO_THROW({
        dp.train(true);
    });
}

TEST_P(DataParallelMultiBackendTest, EvalMode) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    EXPECT_NO_THROW({
        dp.eval();
    });
}

// ============================================================================
// Edge Cases and Error Handling Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, EmptyInputThrows) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Empty tensor should throw
    auto input_tensor = tenzor::empty({0});
    Variable input(input_tensor);

    EXPECT_THROW({
        auto output = dp.forward(input);
    }, std::runtime_error);
}

TEST_P(DataParallelMultiBackendTest, MultidimensionalInput) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Test with 3D input (batch, height, width)
    auto input_tensor = tenzor::ones({4, 8, 8});
    Variable input(input_tensor);

    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 4);
        EXPECT_EQ(output.tensor().shape()[1], 8);
        EXPECT_EQ(output.tensor().shape()[2], 8);
    });
}

TEST_P(DataParallelMultiBackendTest, NonZeroBatchDimension) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0, 1);  // Split along dimension 1

    // Input shape: (channels, batch, features)
    auto input_tensor = tenzor::ones({3, 8, 10});
    Variable input(input_tensor);

    EXPECT_NO_THROW({
        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 3);
        EXPECT_EQ(output.tensor().shape()[1], 8);
        EXPECT_EQ(output.tensor().shape()[2], 10);
    });
}

// ============================================================================
// Correctness Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, ForwardPassCorrectness) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Create input with specific values
    auto input_tensor = tenzor::ones({4, 5});
    auto input_data = input_tensor.data<float>();
    for (int i = 0; i < 20; ++i) {
        input_data[i] = static_cast<float>(i + 1);
    }

    Variable input(input_tensor);
    auto output = dp.forward(input);

    // Verify all values are exactly doubled
    auto output_data = output.tensor().data<float>();
    for (int i = 0; i < 20; ++i) {
        EXPECT_FLOAT_EQ(output_data[i], static_cast<float>((i + 1) * 2));
    }
}

TEST_P(DataParallelMultiBackendTest, BatchPreservation) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Test that batch order is preserved
    std::vector<int> batch_sizes = {2, 4, 8, 16};

    for (int batch_size : batch_sizes) {
        auto input_tensor = tenzor::ones({batch_size, 10});
        Variable input(input_tensor);

        auto output = dp.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], batch_size);
        EXPECT_EQ(output.tensor().shape()[1], 10);
    }
}

// ============================================================================
// Thread Safety Tests (Basic)
// ============================================================================

TEST_P(DataParallelMultiBackendTest, ReplicaInitializationThreadSafety) {
    auto module = std::make_shared<DoubleModule>();
    DataParallel dp(module, {0}, 0);

    // Multiple forward passes should initialize replicas only once
    auto input_tensor = tenzor::ones({4, 10});
    Variable input(input_tensor);

    EXPECT_NO_THROW({
        dp.forward(input);
        dp.forward(input);
        dp.forward(input);
    });
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST_P(DataParallelMultiBackendTest, MakeDataParallelHelper) {
    auto module = std::make_shared<DoubleModule>();

    // Test helper function
    EXPECT_NO_THROW({
        auto parallel_model = make_data_parallel(module, {0}, 0);
        EXPECT_NE(parallel_model, nullptr);
    });
}

TEST_P(DataParallelMultiBackendTest, MakeDataParallelAutoDetect) {
    auto module = std::make_shared<DoubleModule>();

    // Test auto-detection
    EXPECT_NO_THROW({
        auto parallel_model = make_data_parallel(module);
        EXPECT_NE(parallel_model, nullptr);
        EXPECT_GT(parallel_model->device_ids().size(), 0);
    });
}

// ============================================================================
// Instantiate: CUDA-only (DataParallel does not support other backends).
// The fixture's SetUp() will translate "device not present" into a
// structured BackendUnavailable skip on hosts without a CUDA driver.
// ============================================================================
INSTANTIATE_TEST_SUITE_P(
    DataParallelCuda,
    DataParallelMultiBackendTest,
    ::testing::Values(std::string("cuda")),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param;
    });

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    // Print CUDA availability info
    if (get_device_count() > 0) {
        std::cout << "CUDA is available. Device count: " << get_device_count() << std::endl;
    } else {
        std::cout << "CUDA is not available. Many tests will be skipped." << std::endl;
    }

    return RUN_ALL_TESTS();
}
