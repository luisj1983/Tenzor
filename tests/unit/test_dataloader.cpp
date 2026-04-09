#include <gtest/gtest.h>
#include "tenzor/data/dataloader.hpp"
#include "tenzor/data/dataset.hpp"
#include "tenzor/data/transforms.hpp"
#include <chrono>
#include <algorithm>

using namespace tenzor;
using namespace tenzor::data;

// Test fixture for DataLoader tests
class DataLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Create simple dataset: inputs [0, 1, 2, ..., 99], targets = inputs * 2
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

    std::shared_ptr<TensorDataset> dataset_;
};

// Test dataset creation and basic operations
TEST_F(DataLoaderTest, DatasetCreation) {
    ASSERT_NE(dataset_, nullptr);
    EXPECT_EQ(dataset_->size(), 100);
    EXPECT_FALSE(dataset_->empty());

    auto [input, target] = dataset_->get(0);
    EXPECT_EQ(input.shape().size(), 1);
    EXPECT_EQ(input.shape()[0], 1);
}

// Test TensorDataset access
TEST_F(DataLoaderTest, TensorDatasetAccess) {
    // Test first sample
    auto [input0, target0] = dataset_->get(0);
    EXPECT_NEAR(input0.item<float>(), 0.0f, 1e-5);
    EXPECT_NEAR(target0.item<float>(), 0.0f, 1e-5);

    // Test middle sample
    auto [input50, target50] = dataset_->get(50);
    EXPECT_NEAR(input50.item<float>(), 50.0f, 1e-5);
    EXPECT_NEAR(target50.item<float>(), 100.0f, 1e-5);

    // Test last sample
    auto [input99, target99] = dataset_->get(99);
    EXPECT_NEAR(input99.item<float>(), 99.0f, 1e-5);
    EXPECT_NEAR(target99.item<float>(), 198.0f, 1e-5);
}

// Test out of bounds access
TEST_F(DataLoaderTest, DatasetOutOfBounds) {
    EXPECT_THROW(dataset_->get(100), std::out_of_range);
    EXPECT_THROW(dataset_->get(1000), std::out_of_range);
}

// Test single-threaded DataLoader
TEST_F(DataLoaderTest, SingleThreadedLoading) {
    DataLoaderConfig config;
    config.batch_size = 10;
    config.shuffle = false;
    config.num_workers = 0;

    DataLoader loader(dataset_, config);

    EXPECT_EQ(loader.size(), 10);  // 100 samples / 10 batch_size = 10 batches

    size_t batch_count = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.inputs.shape()[0], 10);  // batch size
        EXPECT_EQ(batch.inputs.shape()[1], 1);   // feature size
        EXPECT_EQ(batch.targets.shape()[0], 10);
        EXPECT_EQ(batch.targets.shape()[1], 1);
        batch_count++;
    }

    EXPECT_EQ(batch_count, 10);
}

// Test batching with different batch sizes
TEST_F(DataLoaderTest, DifferentBatchSizes) {
    // Batch size 1
    {
        DataLoader loader(dataset_, 1, false, 0);
        EXPECT_EQ(loader.size(), 100);

        size_t count = 0;
        for (const auto& batch : loader) {
            EXPECT_EQ(batch.inputs.shape()[0], 1);
            count++;
        }
        EXPECT_EQ(count, 100);
    }

    // Batch size 32
    {
        DataLoader loader(dataset_, 32, false, 0);
        EXPECT_EQ(loader.size(), 4);  // ceil(100/32) = 4

        size_t count = 0;
        for (const auto& batch : loader) {
            if (count < 3) {
                EXPECT_EQ(batch.inputs.shape()[0], 32);
            } else {
                EXPECT_EQ(batch.inputs.shape()[0], 4);  // Last batch has 4 samples
            }
            count++;
        }
        EXPECT_EQ(count, 4);
    }

    // Batch size 100 (entire dataset)
    {
        DataLoader loader(dataset_, 100, false, 0);
        EXPECT_EQ(loader.size(), 1);

        size_t count = 0;
        for (const auto& batch : loader) {
            EXPECT_EQ(batch.inputs.shape()[0], 100);
            count++;
        }
        EXPECT_EQ(count, 1);
    }
}

// Test drop_last option
TEST_F(DataLoaderTest, DropLastBatch) {
    DataLoaderConfig config;
    config.batch_size = 32;
    config.shuffle = false;
    config.num_workers = 0;
    config.drop_last = true;

    DataLoader loader(dataset_, config);

    EXPECT_EQ(loader.size(), 3);  // floor(100/32) = 3 (drops last 4 samples)

    size_t batch_count = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.inputs.shape()[0], 32);  // All batches have full batch size
        batch_count++;
    }

    EXPECT_EQ(batch_count, 3);
}

// Test shuffling
TEST_F(DataLoaderTest, Shuffling) {
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
            first_epoch_batch.push_back(batch.inputs.slice(0, i, i + 1).item<float>());
        }
    }

    // Reset and collect first batch from second epoch
    loader.reset();
    std::vector<float> second_epoch_batch;
    it = loader.begin();
    if (it != loader.end()) {
        const auto& batch = *it;
        for (size_t i = 0; i < 10; ++i) {
            second_epoch_batch.push_back(batch.inputs.slice(0, i, i + 1).item<float>());
        }
    }

    // Batches should be different due to shuffling (with high probability)
    EXPECT_NE(first_epoch_batch, second_epoch_batch);
}

// Test multi-threaded loading
TEST_F(DataLoaderTest, MultiThreadedLoading) {
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
        total_samples += batch.inputs.shape()[0];
        batch_count++;
    }

    EXPECT_EQ(batch_count, 10);
    EXPECT_EQ(total_samples, 100);
}

// Test multi-threaded performance improvement
TEST_F(DataLoaderTest, MultiThreadedPerformance) {
    // Create larger dataset for meaningful performance test
    std::vector<float> input_data(10000);
    std::vector<float> target_data(10000);

    for (size_t i = 0; i < 10000; ++i) {
        input_data[i] = static_cast<float>(i);
        target_data[i] = static_cast<float>(i * 2);
    }

    auto inputs = from_data(input_data.data(), {10000, 1});
    auto targets = from_data(target_data.data(), {10000, 1});
    auto large_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Single-threaded timing
    auto start_single = std::chrono::high_resolution_clock::now();
    {
        DataLoader loader(large_dataset, 32, false, 0);
        for (const auto& batch : loader) {
            // Simulate some processing
            volatile float sum = 0.0f;
            for (size_t i = 0; i < batch.inputs.shape()[0]; ++i) {
                sum += batch.inputs.slice(0, i, i + 1).item<float>();
            }
        }
    }
    auto end_single = std::chrono::high_resolution_clock::now();
    auto duration_single = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_single - start_single).count();

    // Multi-threaded timing
    auto start_multi = std::chrono::high_resolution_clock::now();
    {
        DataLoader loader(large_dataset, 32, false, 4);
        for (const auto& batch : loader) {
            // Simulate some processing
            volatile float sum = 0.0f;
            for (size_t i = 0; i < batch.inputs.shape()[0]; ++i) {
                sum += batch.inputs.slice(0, i, i + 1).item<float>();
            }
        }
    }
    auto end_multi = std::chrono::high_resolution_clock::now();
    auto duration_multi = std::chrono::duration_cast<std::chrono::milliseconds>(
        end_multi - start_multi).count();

    // Multi-threaded should be comparable or faster (not strictly enforced in test)
    std::cout << "Single-threaded: " << duration_single << "ms\n";
    std::cout << "Multi-threaded: " << duration_multi << "ms\n";
    std::cout << "Speedup: " << static_cast<float>(duration_single) / duration_multi << "x\n";
}

// Test iterator functionality
TEST_F(DataLoaderTest, IteratorOperations) {
    DataLoader loader(dataset_, 10, false, 0);

    auto it = loader.begin();
    auto end = loader.end();

    EXPECT_NE(it, end);

    // Test dereference
    const auto& batch = *it;
    EXPECT_EQ(batch.inputs.shape()[0], 10);

    // Test pre-increment
    ++it;
    EXPECT_NE(it, end);

    // Test post-increment
    auto old_it = it++;
    EXPECT_NE(old_it, it);

    // Iterate to end
    size_t count = 3;  // Already advanced 3 times (fetched batches 0, 1, 2)
    while (it != end) {
        ++it;
        count++;
    }
    // Note: The iterator performs N+1 increments for N batches (including end detection)
    // After manually fetching 3 batches, we need 8 more increments to reach end (2->10)
    EXPECT_EQ(count, 11);
    EXPECT_EQ(it, end);
}

// Test reset functionality
TEST_F(DataLoaderTest, ResetLoader) {
    DataLoader loader(dataset_, 10, false, 0);

    // First iteration
    size_t first_count = 0;
    for (const auto& batch : loader) {
        first_count++;
    }
    EXPECT_EQ(first_count, 10);

    // Reset and iterate again
    loader.reset();
    size_t second_count = 0;
    for (const auto& batch : loader) {
        second_count++;
    }
    EXPECT_EQ(second_count, 10);
}

// Test ConcatDataset
TEST_F(DataLoaderTest, ConcatDataset) {
    // Create second dataset
    std::vector<float> input_data2(50);
    std::vector<float> target_data2(50);

    for (size_t i = 0; i < 50; ++i) {
        input_data2[i] = static_cast<float>(i + 100);
        target_data2[i] = static_cast<float>((i + 100) * 2);
    }

    auto inputs2 = from_data(input_data2.data(), {50, 1});
    auto targets2 = from_data(target_data2.data(), {50, 1});
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    // Concatenate datasets
    std::vector<std::shared_ptr<Dataset>> datasets = {dataset_, dataset2};
    auto concat_dataset = std::make_shared<ConcatDataset>(datasets);

    EXPECT_EQ(concat_dataset->size(), 150);

    // Test access from first dataset
    auto [input0, target0] = concat_dataset->get(0);
    EXPECT_NEAR(input0.item<float>(), 0.0f, 1e-5);

    // Test access from second dataset
    auto [input100, target100] = concat_dataset->get(100);
    EXPECT_NEAR(input100.item<float>(), 100.0f, 1e-5);

    // Test DataLoader with concatenated dataset
    DataLoader loader(concat_dataset, 25, false, 0);
    EXPECT_EQ(loader.size(), 6);  // ceil(150/25) = 6

    size_t batch_count = 0;
    for (const auto& batch : loader) {
        batch_count++;
    }
    EXPECT_EQ(batch_count, 6);
}

// Test empty dataset error handling
TEST_F(DataLoaderTest, ErrorHandling) {
    // Null dataset
    EXPECT_THROW(DataLoader(nullptr, 10), std::invalid_argument);

    // Zero batch size
    EXPECT_THROW(DataLoader(dataset_, 0), std::invalid_argument);

    // Empty ConcatDataset
    std::vector<std::shared_ptr<Dataset>> empty_datasets;
    EXPECT_THROW((ConcatDataset(empty_datasets)), std::invalid_argument);
}

// Test data correctness with batching
TEST_F(DataLoaderTest, DataCorrectness) {
    DataLoader loader(dataset_, 10, false, 0);

    size_t sample_idx = 0;
    for (const auto& batch : loader) {
        for (size_t i = 0; i < batch.inputs.shape()[0]; ++i) {
            float input_val = batch.inputs.slice(0, i, i + 1).item<float>();
            float target_val = batch.targets.slice(0, i, i + 1).item<float>();

            // Verify data integrity
            EXPECT_NEAR(input_val, static_cast<float>(sample_idx), 1e-5)
                << "Sample " << sample_idx << " input mismatch";
            EXPECT_NEAR(target_val, static_cast<float>(sample_idx * 2), 1e-5)
                << "Sample " << sample_idx << " target mismatch";

            sample_idx++;
        }
    }

    EXPECT_EQ(sample_idx, 100);
}

// Test Transform composition
TEST_F(DataLoaderTest, TransformComposition) {
    using namespace transforms;

    auto to_tensor = std::make_shared<ToTensor>();
    auto normalize = std::make_shared<Normalize>(
        std::vector<float>{0.5f}, std::vector<float>{0.5f});

    auto compose = std::make_shared<Compose>(
        std::vector<std::shared_ptr<Transform>>{to_tensor, normalize});

    auto transformed = std::make_shared<TransformedDataset>(dataset_,
        [compose](const Tensor& input, const Tensor& target) {
            return (*compose)(input, target);
        });

    EXPECT_EQ(transformed->size(), dataset_->size());

    DataLoader loader(transformed, 10, false, 0);
    size_t batch_count = 0;
    for (const auto& batch : loader) {
        EXPECT_EQ(batch.inputs.shape()[0], 10);
        batch_count++;
    }
    EXPECT_EQ(batch_count, 10);
}

// Test with MNIST-like data shape
TEST_F(DataLoaderTest, MNISTLikeData) {
    // Create MNIST-like dataset: 1000 samples of 28x28 images
    size_t num_samples = 1000;
    size_t height = 28;
    size_t width = 28;

    std::vector<float> image_data(num_samples * height * width);
    std::vector<int64_t> label_data(num_samples);

    for (size_t i = 0; i < num_samples; ++i) {
        // Random image data
        for (size_t j = 0; j < height * width; ++j) {
            image_data[i * height * width + j] = static_cast<float>(rand()) / RAND_MAX;
        }
        // Label is class index
        label_data[i] = static_cast<int64_t>(i % 10);
    }

    auto images = from_data(image_data.data(), {num_samples, height, width});
    auto labels = from_data(label_data.data(), {num_samples, 1});

    auto mnist_dataset = std::make_shared<TensorDataset>(images, labels);

    // Test DataLoader with MNIST-like data
    DataLoaderConfig config;
    config.batch_size = 64;
    config.shuffle = true;
    config.num_workers = 2;

    DataLoader loader(mnist_dataset, config);

    size_t batch_count = 0;
    size_t total_samples = 0;

    for (const auto& batch : loader) {
        // Check batch dimensions
        EXPECT_EQ(batch.inputs.ndim(), 3);  // [batch_size, height, width]
        EXPECT_GT(batch.inputs.shape()[0], 0);
        EXPECT_LE(batch.inputs.shape()[0], 64);
        EXPECT_EQ(batch.inputs.shape()[1], 28);
        EXPECT_EQ(batch.inputs.shape()[2], 28);

        EXPECT_EQ(batch.targets.ndim(), 2);  // [batch_size, 1]
        EXPECT_EQ(batch.targets.shape()[0], batch.inputs.shape()[0]);

        total_samples += batch.inputs.shape()[0];
        batch_count++;
    }

    EXPECT_EQ(total_samples, num_samples);
    EXPECT_EQ(batch_count, loader.size());
}

// Main function
int main(int argc, char** argv) {
    // Initialize Tenzor library

    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
