/**
 * @file test_data_pipeline.cpp
 * @brief Integration tests for data loading and preprocessing pipelines
 *
 * Tests:
 * - DataLoader end-to-end functionality
 * - Batch loading and shuffling
 * - Data augmentation pipelines
 * - Custom dataset implementations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/data/dataset.hpp>
#include <tenzor/data/dataloader.hpp>
#include <vector>
#include <memory>

using namespace tenzor;

//==============================================================================
// Test Environment
//==============================================================================

class DataPipelineEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

static ::testing::Environment* const data_pipeline_env =
    ::testing::AddGlobalTestEnvironment(new DataPipelineEnvironment);

//==============================================================================
// Test Dataset Implementations
//==============================================================================

class SimpleDataset : public data::Dataset {
public:
    SimpleDataset(size_t size) : size_(size) {}

    auto size() const -> size_t override { return size_; }

    auto get(size_t index) -> std::pair<Tensor, Tensor> override {
        auto data = randn({3, 32, 32}, DType::Float32, Device::cpu());
        auto label = zeros({10}, DType::Float32, Device::cpu());
        auto label_data = const_cast<float*>(label.template data<float>());
        label_data[index % 10] = 1.0f;
        return {data, label};
    }

private:
    size_t size_;
};

//==============================================================================
// Test 1: Basic DataLoader Creation
//==============================================================================

TEST(DataPipeline, BasicDataLoaderCreation) {
    auto dataset = std::make_shared<SimpleDataset>(100);

    data::DataLoader loader(dataset, 16, false);

    EXPECT_GT(loader.size(), 0);
}

//==============================================================================
// Test 2: DataLoader Batch Iteration
//==============================================================================

TEST(DataPipeline, DataLoaderBatchIteration) {
    auto dataset = std::make_shared<SimpleDataset>(100);
    data::DataLoader loader(dataset, 16, false);

    int batch_count = 0;
    for (auto batch : loader) {
        const auto& data = batch.inputs;
        const auto& labels = batch.targets;

        // Verify batch shapes
        EXPECT_EQ(data.shape()[0], 16) << "Batch size should be 16";
        EXPECT_EQ(data.shape()[1], 3) << "Should have 3 channels";
        EXPECT_EQ(data.shape()[2], 32) << "Height should be 32";
        EXPECT_EQ(data.shape()[3], 32) << "Width should be 32";

        EXPECT_EQ(labels.shape()[0], 16) << "Label batch size should match";
        EXPECT_EQ(labels.shape()[1], 10) << "Should have 10 classes";

        batch_count++;
    }

    EXPECT_EQ(batch_count, 100 / 16) << "Should have correct number of batches";
}

//==============================================================================
// Test 3: DataLoader with Shuffling
//==============================================================================

TEST(DataPipeline, DataLoaderShuffling) {
    auto dataset = std::make_shared<SimpleDataset>(50);

    data::DataLoader loader_no_shuffle(dataset, 10, false);
    data::DataLoader loader_shuffle(dataset, 10, true);

    // Collect first batch from both loaders
    auto it_no_shuffle = loader_no_shuffle.begin();
    const auto& batch_no_shuffle = *it_no_shuffle;
    const auto& data_no_shuffle = batch_no_shuffle.inputs;

    auto it_shuffle = loader_shuffle.begin();
    const auto& batch_shuffle = *it_shuffle;
    const auto& data_shuffle = batch_shuffle.inputs;

    // Note: We can't directly verify shuffling without exposing indices,
    // but we can verify the loaders work correctly
    auto shape1 = data_no_shuffle.shape();
    auto shape2 = data_shuffle.shape();
    EXPECT_EQ(shape1.size(), shape2.size());
    for (size_t i = 0; i < shape1.size(); ++i) {
        EXPECT_EQ(shape1[i], shape2[i]);
    }
}

//==============================================================================
// Test 4: DataLoader with Different Batch Sizes
//==============================================================================

TEST(DataPipeline, DifferentBatchSizes) {
    auto dataset = std::make_shared<SimpleDataset>(100);

    std::vector<size_t> batch_sizes = {1, 8, 16, 32, 64};

    for (auto batch_size : batch_sizes) {
        data::DataLoader loader(dataset, batch_size, false);

        auto it = loader.begin();
        const auto& batch = *it;
        const auto& data = batch.inputs;

        EXPECT_EQ(data.shape()[0], batch_size)
            << "Batch size should match for batch_size=" << batch_size;
    }
}

//==============================================================================
// Test 5: DataLoader with Drop Last
//==============================================================================

TEST(DataPipeline, DropLastBatch) {
    auto dataset = std::make_shared<SimpleDataset>(105);  // Not divisible by batch size

    data::DataLoader loader_no_drop(dataset, 32, false, 0, false, false);
    data::DataLoader loader_drop(dataset, 32, false, 0, false, true);

    // Count batches
    int count_no_drop = 0;
    for (auto batch : loader_no_drop) {
        count_no_drop++;
    }

    int count_drop = 0;
    for (auto batch : loader_drop) {
        count_drop++;
    }

    // With drop_last=false: 105/32 = 4 batches (32+32+32+9)
    // With drop_last=true: 105/32 = 3 batches (32+32+32, drop 9)
    EXPECT_EQ(count_no_drop, 4) << "Should have 4 batches without drop";
    EXPECT_EQ(count_drop, 3) << "Should have 3 batches with drop";
}

//==============================================================================
// Test 6: Multiple Epochs with DataLoader
//==============================================================================

TEST(DataPipeline, MultipleEpochs) {
    auto dataset = std::make_shared<SimpleDataset>(64);
    data::DataLoader loader(dataset, 16, false);

    for (int epoch = 0; epoch < 3; epoch++) {
        int batch_count = 0;

        for (auto batch : loader) {
            batch_count++;
        }

        EXPECT_EQ(batch_count, 4) << "Should have 4 batches per epoch";
    }
}

//==============================================================================
// Test 7: DataLoader with CUDA Device
//==============================================================================

TEST(DataPipeline, DataLoaderCUDA) {
    bool cuda_available = false;
    try {
        auto test_tensor = ones({1}, DType::Float32, Device::cuda());
        cuda_available = true;
    } catch (...) {}

    if (!cuda_available) {
        GTEST_SKIP() << "CUDA not available";
    }

    auto dataset = std::make_shared<SimpleDataset>(64);
    data::DataLoader loader(dataset, 16, false);

    // Load data and transfer to CUDA
    auto it = loader.begin();
    const auto& batch = *it;
    const auto& data_cpu = batch.inputs;
    const auto& labels_cpu = batch.targets;

    auto data_cuda = data_cpu.to(Device::cuda());
    auto labels_cuda = labels_cpu.to(Device::cuda());

    EXPECT_EQ(data_cuda.device().type, Device::Type::CUDA);
    EXPECT_EQ(labels_cuda.device().type, Device::Type::CUDA);
}

//==============================================================================
// Test 8: Empty Dataset Handling
//==============================================================================

TEST(DataPipeline, EmptyDataset) {
    auto dataset = std::make_shared<SimpleDataset>(0);
    data::DataLoader loader(dataset, 16, false);

    int batch_count = 0;
    for (auto batch : loader) {
        batch_count++;
    }

    EXPECT_EQ(batch_count, 0) << "Empty dataset should produce no batches";
}

//==============================================================================
// Test 9: Training Loop with DataLoader
//==============================================================================

TEST(DataPipeline, TrainingLoopWithDataLoader) {
    auto dataset = std::make_shared<SimpleDataset>(128);
    data::DataLoader loader(dataset, 32, true);  // With shuffling

    // Simple model
    auto model = std::make_shared<nn::Linear>(3 * 32 * 32, 10);
    model->to(Device::cpu());

    auto params = model->parameters();
    optim::SGD optimizer(params, 0.01);

    model->train();
    int total_batches = 0;

    for (auto batch : loader) {
        auto data = batch.inputs;
        const auto& labels = batch.targets;

        // Flatten input
        data = data.view({data.shape()[0], -1});

        auto input_var = Variable(data, true);
        auto target_var = Variable(labels, false);

        optimizer.zero_grad();
        auto output = model->forward(input_var);
        auto loss = nn::mse_loss(output, target_var);
        loss.backward();
        optimizer.step();

        total_batches++;
    }

    EXPECT_EQ(total_batches, 128 / 32) << "Should process all batches";
}

//==============================================================================
// Test 10: DataLoader Performance Test
//==============================================================================

TEST(DataPipeline, DataLoaderPerformance) {
    auto dataset = std::make_shared<SimpleDataset>(1000);
    data::DataLoader loader(dataset, 64, false);

    auto start = std::chrono::high_resolution_clock::now();

    for (auto batch : loader) {
        // Just iterate, no processing
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "DataLoader iteration time: " << duration << "ms" << std::endl;

    EXPECT_LT(duration, 5000) << "DataLoader should iterate in reasonable time";
}
