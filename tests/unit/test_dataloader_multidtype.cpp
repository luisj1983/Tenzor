/**
 * @file test_dataloader_multidtype.cpp
 * @brief Multi-dtype tests for dataloader
 *
 * Coverage: DType testing for data loading operations
 * - Primary dtypes: Float32, Float64, Int32, Int64
 * - Backend: CPU (data loading is CPU-based)
 *
 * Note: DataLoader works with various dtypes for inputs and labels.
 * Testing common dtypes used in machine learning workflows.
 */

#include <gtest/gtest.h>
#include "tenzor/data/dataloader.hpp"
#include "tenzor/data/dataset.hpp"
#include "tenzor/data/transforms.hpp"
#include <chrono>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::data;

// ============================================================================
// DType Parameterization
// ============================================================================

struct DTypeParam {
    DType dtype;
    std::string dtype_name;

    std::string ToString() const {
        return dtype_name;
    }
};

class DataLoaderMultiDTypeTest : public ::testing::TestWithParam<DTypeParam> {
protected:
    DType dtype;
    std::shared_ptr<TensorDataset> dataset_;
    Device device;

    void SetUp() override {
        auto param = GetParam();
        dtype = param.dtype;
        device = Device::cpu();

        // Create simple dataset based on dtype
        if (dtype == DType::Float32) {
            std::vector<float> input_data(100);
            std::vector<float> target_data(100);

            for (size_t i = 0; i < 100; ++i) {
                input_data[i] = static_cast<float>(i);
                target_data[i] = static_cast<float>(i * 2);
            }

            auto inputs = from_data(input_data.data(), {100, 1});
            auto targets = from_data(target_data.data(), {100, 1});
            dataset_ = std::make_shared<TensorDataset>(inputs, targets);
        }
        else if (dtype == DType::Float64) {
            std::vector<double> input_data(100);
            std::vector<double> target_data(100);

            for (size_t i = 0; i < 100; ++i) {
                input_data[i] = static_cast<double>(i);
                target_data[i] = static_cast<double>(i * 2);
            }

            auto inputs = from_data(input_data.data(), {100, 1});
            auto targets = from_data(target_data.data(), {100, 1});
            dataset_ = std::make_shared<TensorDataset>(inputs, targets);
        }
        else if (dtype == DType::Int32) {
            std::vector<int32_t> input_data(100);
            std::vector<int32_t> target_data(100);

            for (size_t i = 0; i < 100; ++i) {
                input_data[i] = static_cast<int32_t>(i);
                target_data[i] = static_cast<int32_t>(i * 2);
            }

            auto inputs = from_data(input_data.data(), {100, 1});
            auto targets = from_data(target_data.data(), {100, 1});
            dataset_ = std::make_shared<TensorDataset>(inputs, targets);
        }
        else if (dtype == DType::Int64) {
            std::vector<int64_t> input_data(100);
            std::vector<int64_t> target_data(100);

            for (size_t i = 0; i < 100; ++i) {
                input_data[i] = static_cast<int64_t>(i);
                target_data[i] = static_cast<int64_t>(i * 2);
            }

            auto inputs = from_data(input_data.data(), {100, 1});
            auto targets = from_data(target_data.data(), {100, 1});
            dataset_ = std::make_shared<TensorDataset>(inputs, targets);
        }
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
    EXPECT_EQ(input.dtype(), dtype);
    EXPECT_EQ(target.dtype(), dtype);
}

TEST_P(DataLoaderMultiDTypeTest, TensorDatasetAccess) {
    auto [input0, target0] = dataset_->get(0);
    auto [input50, target50] = dataset_->get(50);
    auto [input99, target99] = dataset_->get(99);

    if (dtype == DType::Float32) {
        EXPECT_NEAR(input0.item<float>(), 0.0f, 1e-5);
        EXPECT_NEAR(target0.item<float>(), 0.0f, 1e-5);
        EXPECT_NEAR(input50.item<float>(), 50.0f, 1e-5);
        EXPECT_NEAR(target50.item<float>(), 100.0f, 1e-5);
    } else if (dtype == DType::Float64) {
        EXPECT_NEAR(input0.item<double>(), 0.0, 1e-10);
        EXPECT_NEAR(target0.item<double>(), 0.0, 1e-10);
        EXPECT_NEAR(input50.item<double>(), 50.0, 1e-10);
        EXPECT_NEAR(target50.item<double>(), 100.0, 1e-10);
    } else if (dtype == DType::Int32) {
        EXPECT_EQ(input0.item<int32_t>(), 0);
        EXPECT_EQ(target0.item<int32_t>(), 0);
        EXPECT_EQ(input50.item<int32_t>(), 50);
        EXPECT_EQ(target50.item<int32_t>(), 100);
    } else if (dtype == DType::Int64) {
        EXPECT_EQ(input0.item<int64_t>(), 0);
        EXPECT_EQ(target0.item<int64_t>(), 0);
        EXPECT_EQ(input50.item<int64_t>(), 50);
        EXPECT_EQ(target50.item<int64_t>(), 100);
    }
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
        EXPECT_EQ(batch.inputs.shape()[0], 10);
        EXPECT_EQ(batch.inputs.shape()[1], 1);
        EXPECT_EQ(batch.targets.shape()[0], 10);
        EXPECT_EQ(batch.targets.shape()[1], 1);
        EXPECT_EQ(batch.inputs.dtype(), dtype);
        EXPECT_EQ(batch.targets.dtype(), dtype);
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
            EXPECT_EQ(batch.inputs.shape()[0], 1);
            EXPECT_EQ(batch.inputs.dtype(), dtype);
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
            if (count < 3) {
                EXPECT_EQ(batch.inputs.shape()[0], 32);
            } else {
                EXPECT_EQ(batch.inputs.shape()[0], 4);
            }
            EXPECT_EQ(batch.inputs.dtype(), dtype);
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
        EXPECT_EQ(batch.inputs.shape()[0], 32);
        EXPECT_EQ(batch.inputs.dtype(), dtype);
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
    std::vector<double> first_epoch_batch;
    auto it = loader.begin();
    if (it != loader.end()) {
        const auto& batch = *it;
        for (size_t i = 0; i < 10; ++i) {
            auto slice = batch.inputs.slice(0, i, i + 1);
            if (dtype == DType::Float32) {
                first_epoch_batch.push_back(slice.item<float>());
            } else if (dtype == DType::Float64) {
                first_epoch_batch.push_back(slice.item<double>());
            } else if (dtype == DType::Int32) {
                first_epoch_batch.push_back(slice.item<int32_t>());
            } else if (dtype == DType::Int64) {
                first_epoch_batch.push_back(slice.item<int64_t>());
            }
        }
    }

    // Reset and collect first batch from second epoch
    loader.reset();
    std::vector<double> second_epoch_batch;
    it = loader.begin();
    if (it != loader.end()) {
        const auto& batch = *it;
        for (size_t i = 0; i < 10; ++i) {
            auto slice = batch.inputs.slice(0, i, i + 1);
            if (dtype == DType::Float32) {
                second_epoch_batch.push_back(slice.item<float>());
            } else if (dtype == DType::Float64) {
                second_epoch_batch.push_back(slice.item<double>());
            } else if (dtype == DType::Int32) {
                second_epoch_batch.push_back(slice.item<int32_t>());
            } else if (dtype == DType::Int64) {
                second_epoch_batch.push_back(slice.item<int64_t>());
            }
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
        EXPECT_GT(batch.inputs.shape()[0], 0);
        EXPECT_LE(batch.inputs.shape()[0], 10);
        EXPECT_EQ(batch.inputs.dtype(), dtype);
        total_samples += batch.inputs.shape()[0];
        batch_count++;
    }

    EXPECT_EQ(batch_count, 10);
    EXPECT_EQ(total_samples, 100);
}

TEST_P(DataLoaderMultiDTypeTest, DataCorrectness) {
    DataLoader loader(dataset_, 10, false, 0);

    size_t sample_idx = 0;
    for (const auto& batch : loader) {
        for (size_t i = 0; i < batch.inputs.shape()[0]; ++i) {
            auto input_slice = batch.inputs.slice(0, i, i + 1);
            auto target_slice = batch.targets.slice(0, i, i + 1);

            if (dtype == DType::Float32) {
                float input_val = input_slice.item<float>();
                float target_val = target_slice.item<float>();
                EXPECT_NEAR(input_val, static_cast<float>(sample_idx), 1e-5);
                EXPECT_NEAR(target_val, static_cast<float>(sample_idx * 2), 1e-5);
            } else if (dtype == DType::Float64) {
                double input_val = input_slice.item<double>();
                double target_val = target_slice.item<double>();
                EXPECT_NEAR(input_val, static_cast<double>(sample_idx), 1e-10);
                EXPECT_NEAR(target_val, static_cast<double>(sample_idx * 2), 1e-10);
            } else if (dtype == DType::Int32) {
                int32_t input_val = input_slice.item<int32_t>();
                int32_t target_val = target_slice.item<int32_t>();
                EXPECT_EQ(input_val, static_cast<int32_t>(sample_idx));
                EXPECT_EQ(target_val, static_cast<int32_t>(sample_idx * 2));
            } else if (dtype == DType::Int64) {
                int64_t input_val = input_slice.item<int64_t>();
                int64_t target_val = target_slice.item<int64_t>();
                EXPECT_EQ(input_val, static_cast<int64_t>(sample_idx));
                EXPECT_EQ(target_val, static_cast<int64_t>(sample_idx * 2));
            }

            sample_idx++;
        }
    }

    EXPECT_EQ(sample_idx, 100);
}

TEST_P(DataLoaderMultiDTypeTest, ResetLoader) {
    DataLoader loader(dataset_, 10, false, 0);

    size_t first_count = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.inputs.dtype(), dtype);
        first_count++;
    }
    EXPECT_EQ(first_count, 10);

    loader.reset();
    size_t second_count = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.inputs.dtype(), dtype);
        second_count++;
    }
    EXPECT_EQ(second_count, 10);
}

// ==============================================================================
// Test Instantiation
// ==============================================================================

std::vector<DTypeParam> GenerateDataLoaderDTypes() {
    return {
        {DType::Float32, "float32"},
        {DType::Float64, "float64"},
        {DType::Int32, "int32"},
        {DType::Int64, "int64"},
    };
}

INSTANTIATE_TEST_SUITE_P(
    CommonDTypes,
    DataLoaderMultiDTypeTest,
    ::testing::ValuesIn(GenerateDataLoaderDTypes()),
    [](const ::testing::TestParamInfo<DTypeParam>& info) {
        return info.param.ToString();
    }
);

// ============================================================================
// Test Environment Setup
// ============================================================================

class DataLoaderTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        initialize();
    }
};

static ::testing::Environment* const dataloader_env =
    ::testing::AddGlobalTestEnvironment(new DataLoaderTestEnvironment);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
