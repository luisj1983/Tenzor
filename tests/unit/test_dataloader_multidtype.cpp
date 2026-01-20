/**
 * @file test_dataloader_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for dataloader
 *
 * Tests data loading operations with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends to ensure:
 * - Correct dataset creation and access
 * - Proper batching and iteration
 * - Shuffling behavior
 * - Multi-threaded loading
 * - Data correctness across dtypes and devices
 *
 * Note: DataLoader operations are CPU-based by design, but loaded tensors
 * can be transferred to other backends for computation.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/data/dataloader.hpp>
#include <tenzor/data/dataset.hpp>
#include <tenzor/data/transforms.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <chrono>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::data;
using namespace tenzor::testing;

// ============================================================================
// DataLoader Multi-Backend Multi-DType Test Fixture
// ============================================================================

class DataLoaderMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    std::shared_ptr<TensorDataset> dataset_;

    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // DataLoader works on CPU, then transfers to target device
        // Create dataset in Float32 on CPU, will be converted as needed
        std::vector<float> input_data(100);
        std::vector<float> target_data(100);

        for (size_t i = 0; i < 100; ++i) {
            input_data[i] = static_cast<float>(i);
            target_data[i] = static_cast<float>(i * 2);
        }

        auto inputs = from_data(input_data.data(), {100, 1});
        auto targets = from_data(target_data.data(), {100, 1});

        // Convert to test dtype
        if (dtype() != DType::Float32) {
            inputs = inputs.to(dtype());
            targets = targets.to(dtype());
        }

        dataset_ = std::make_shared<TensorDataset>(inputs, targets);
    }

    // Helper to move batch to test device
    Batch moveBatchToDevice(const Batch& batch) {
        return Batch{
            batch.inputs.to(device()),
            batch.targets.to(device())
        };
    }

    // Helper to get value from tensor for comparison
    float getValueAsFloat(const Tensor& t, int64_t idx = 0) {
        auto t_cpu = t.to(Device::cpu()).to(DType::Float32);
        auto slice = t_cpu.slice(0, idx, idx + 1);
        return slice.item<float>();
    }
};

// ==============================================================================
// Dataset Tests
// ==============================================================================

TEST_P(DataLoaderMultiDTypeTest, DatasetCreation) {
    ASSERT_NE(dataset_, nullptr);
    EXPECT_EQ(dataset_->size(), 100);
    EXPECT_FALSE(dataset_->empty());

    auto [input, target] = dataset_->get(0);
    EXPECT_EQ(input.shape().size(), 1);
    EXPECT_EQ(input.shape()[0], 1);
    EXPECT_EQ(input.dtype(), dtype());
    EXPECT_EQ(target.dtype(), dtype());
}

TEST_P(DataLoaderMultiDTypeTest, TensorDatasetAccess) {
    auto [input0, target0] = dataset_->get(0);
    auto [input50, target50] = dataset_->get(50);
    auto [input99, target99] = dataset_->get(99);

    // Move to test device
    input0 = input0.to(device());
    target0 = target0.to(device());
    input50 = input50.to(device());
    target50 = target50.to(device());

    // Convert back to CPU Float32 for comparison
    EXPECT_NEAR(getValueAsFloat(input0), 0.0f, atol());
    EXPECT_NEAR(getValueAsFloat(target0), 0.0f, atol());
    EXPECT_NEAR(getValueAsFloat(input50), 50.0f, atol());
    EXPECT_NEAR(getValueAsFloat(target50), 100.0f, atol());
}

// ==============================================================================
// DataLoader Tests
// ==============================================================================

TEST_P(DataLoaderMultiDTypeTest, SingleThreadedLoading) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = false;
    config.num_workers = 0;

    DataLoader loader(dataset_, config);

    EXPECT_EQ(loader.size(), 10);

    size_t batch_count = 0;
    for (const auto& batch : loader) {
        auto device_batch = moveBatchToDevice(batch);
        EXPECT_EQ(device_batch.inputs.shape()[0], 10);
        EXPECT_EQ(device_batch.inputs.shape()[1], 1);
        EXPECT_EQ(device_batch.targets.shape()[0], 10);
        EXPECT_EQ(device_batch.targets.shape()[1], 1);
        EXPECT_EQ(device_batch.inputs.dtype(), dtype());
        EXPECT_EQ(device_batch.targets.dtype(), dtype());
        batch_count++;
    }

    EXPECT_EQ(batch_count, 10);
}

TEST_P(DataLoaderMultiDTypeTest, DifferentBatchSizes) {
    // Batch size 1
    {
        DataLoader loader(dataset_, 1, false, 0);
        EXPECT_EQ(loader.size(), 100);

        size_t count = 0;
        for (const auto& batch : loader) {
            auto device_batch = moveBatchToDevice(batch);
            EXPECT_EQ(device_batch.inputs.shape()[0], 1);
            EXPECT_EQ(device_batch.inputs.dtype(), dtype());
            count++;
        }
        EXPECT_EQ(count, 100);
    }

    // Batch size 32
    {
        DataLoader loader(dataset_, 32, false, 0);
        EXPECT_EQ(loader.size(), 4);

        size_t count = 0;
        for (const auto& batch : loader) {
            auto device_batch = moveBatchToDevice(batch);
            if (count < 3) {
                EXPECT_EQ(device_batch.inputs.shape()[0], 32);
            } else {
                EXPECT_EQ(device_batch.inputs.shape()[0], 4);
            }
            EXPECT_EQ(device_batch.inputs.dtype(), dtype());
            count++;
        }
        EXPECT_EQ(count, 4);
    }
}

TEST_P(DataLoaderMultiDTypeTest, DropLastBatch) {
    DataLoaderConfig config;
    config.batch_size = 32;
    config.shuffle = false;
    config.num_workers = 0;
    config.drop_last = true;

    DataLoader loader(dataset_, config);

    EXPECT_EQ(loader.size(), 3);

    size_t batch_count = 0;
    for (const auto& batch : loader) {
        auto device_batch = moveBatchToDevice(batch);
        EXPECT_EQ(device_batch.inputs.shape()[0], 32);
        EXPECT_EQ(device_batch.inputs.dtype(), dtype());
        batch_count++;
    }

    EXPECT_EQ(batch_count, 3);
}

TEST_P(DataLoaderMultiDTypeTest, Shuffling) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = true;
    config.num_workers = 0;

    DataLoader loader(dataset_, config);

    // Collect first batch from first epoch
    std::vector<float> first_epoch_batch;
    auto it = loader.begin();
    if (it != loader.end()) {
        const auto& batch = *it;
        for (size_t i = 0; i < 10; ++i) {
            auto slice = batch.inputs.slice(0, i, i + 1);
            first_epoch_batch.push_back(getValueAsFloat(slice));
        }
    }

    // Reset and collect first batch from second epoch
    loader.reset();
    std::vector<float> second_epoch_batch;
    it = loader.begin();
    if (it != loader.end()) {
        const auto& batch = *it;
        for (size_t i = 0; i < 10; ++i) {
            auto slice = batch.inputs.slice(0, i, i + 1);
            second_epoch_batch.push_back(getValueAsFloat(slice));
        }
    }

    EXPECT_NE(first_epoch_batch, second_epoch_batch);
}

TEST_P(DataLoaderMultiDTypeTest, MultiThreadedLoading) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = false;
    config.num_workers = 4;

    DataLoader loader(dataset_, config);

    EXPECT_EQ(loader.size(), 10);

    size_t batch_count = 0;
    size_t total_samples = 0;

    for (const auto& batch : loader) {
        auto device_batch = moveBatchToDevice(batch);
        EXPECT_GT(device_batch.inputs.shape()[0], 0);
        EXPECT_LE(device_batch.inputs.shape()[0], 10);
        EXPECT_EQ(device_batch.inputs.dtype(), dtype());
        total_samples += device_batch.inputs.shape()[0];
        batch_count++;
    }

    EXPECT_EQ(batch_count, 10);
    EXPECT_EQ(total_samples, 100);
}

TEST_P(DataLoaderMultiDTypeTest, DataCorrectness) {
    DataLoader loader(dataset_, 10, false, 0);

    size_t sample_idx = 0;
    for (const auto& batch : loader) {
        auto device_batch = moveBatchToDevice(batch);
        for (size_t i = 0; i < device_batch.inputs.shape()[0]; ++i) {
            auto input_slice = device_batch.inputs.slice(0, i, i + 1);
            auto target_slice = device_batch.targets.slice(0, i, i + 1);

            float input_val = getValueAsFloat(input_slice);
            float target_val = getValueAsFloat(target_slice);

            EXPECT_NEAR(input_val, static_cast<float>(sample_idx), atol());
            EXPECT_NEAR(target_val, static_cast<float>(sample_idx * 2), atol());

            sample_idx++;
        }
    }

    EXPECT_EQ(sample_idx, 100);
}

TEST_P(DataLoaderMultiDTypeTest, ResetLoader) {
    DataLoader loader(dataset_, 10, false, 0);

    size_t first_count = 0;
    for (const auto& batch : loader) {
        auto device_batch = moveBatchToDevice(batch);
        EXPECT_EQ(device_batch.inputs.dtype(), dtype());
        first_count++;
    }
    EXPECT_EQ(first_count, 10);

    loader.reset();
    size_t second_count = 0;
    for (const auto& batch : loader) {
        auto device_batch = moveBatchToDevice(batch);
        EXPECT_EQ(device_batch.inputs.dtype(), dtype());
        second_count++;
    }
    EXPECT_EQ(second_count, 10);
}

TEST_P(DataLoaderMultiDTypeTest, DeviceTransfer) {
    DataLoader loader(dataset_, 10, false, 0);

    auto it = loader.begin();
    if (it != loader.end()) {
        const auto& batch = *it;

        // Transfer to test device
        auto device_batch = moveBatchToDevice(batch);

        // Verify on target device
        EXPECT_EQ(device_batch.inputs.device().type, device().type);
        EXPECT_EQ(device_batch.targets.device().type, device().type);
        EXPECT_EQ(device_batch.inputs.dtype(), dtype());
        EXPECT_EQ(device_batch.targets.dtype(), dtype());
    }
}

// ==============================================================================
// Test Instantiation
// ==============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DataLoaderMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 9
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 9 tests × 3 dtypes × 3 backends = 81 test scenarios
 *
 * Coverage:
 * - Dataset: creation, access
 * - DataLoader: single-threaded, multi-threaded loading
 * - Batch sizes: different sizes, drop last
 * - Shuffling: epoch-based shuffling
 * - Data correctness: value verification
 * - Device transfer: CPU to target device
 */
