/**
 * @file test_data_pipeline_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for data loading + preprocessing pipeline
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/data/dataloader.hpp>
#include <tenzor/data/dataset.hpp>
#include <tenzor/ops/creation.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class DataPipelineMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(DataPipelineMultiDTypeTest, TensorDatasetCreation) {
    auto data = createRandn({100, 3, 32, 32});
    auto labels = tenzor::randint(0, 10, {100}, DType::Int64, device());

    data::TensorDataset dataset(data, labels);
    EXPECT_EQ(dataset.size(), 100);
}

TEST_P(DataPipelineMultiDTypeTest, TensorDatasetGetItem) {
    auto data = createRandn({50, 3, 8, 8});
    auto labels = tenzor::randint(0, 5, {50}, DType::Int64, device());

    data::TensorDataset dataset(data, labels);
    auto [sample, label] = dataset.get(0);

    expectShape(sample, {3, 8, 8});
}

TEST_P(DataPipelineMultiDTypeTest, DataLoaderIteration) {
    auto data = createRandn({20, 3, 4, 4});
    auto labels = tenzor::randint(0, 2, {20}, DType::Int64, device());

    auto dataset = std::make_shared<data::TensorDataset>(data, labels);
    data::DataLoaderConfig config;
    config.batch_size = 4;
    data::DataLoader loader(dataset, config);

    int batch_count = 0;
    for (auto& [batch_data, batch_labels] : loader) {
        EXPECT_EQ(batch_data.shape()[0], 4);
        batch_count++;
    }
    EXPECT_EQ(batch_count, 5);  // 20 / 4 = 5 batches
}

TEST_P(DataPipelineMultiDTypeTest, DataLoaderShuffle) {
    auto data = createRandn({16, 2});
    auto labels = tenzor::randint(0, 2, {16}, DType::Int64, device());

    auto dataset = std::make_shared<data::TensorDataset>(data, labels);
    data::DataLoaderConfig config;
    config.batch_size = 4;
    config.shuffle = true;
    data::DataLoader loader(dataset, config);

    int batch_count = 0;
    for (auto& [batch_data, batch_labels] : loader) {
        batch_count++;
    }
    EXPECT_EQ(batch_count, 4);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DataPipelineMultiDTypeTest);
