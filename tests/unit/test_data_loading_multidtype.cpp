/**
 * @file test_data_loading_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for data loading (Dataset, Transforms, ConcatDataset)
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/data/dataset.hpp"
#include "tenzor/data/transforms.hpp"

using namespace tenzor;
using namespace tenzor::data;
using namespace tenzor::data::transforms;
using namespace tenzor::testing;

class DataLoadingMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    Tensor createTestInputs(size_t n_samples, const std::vector<int64_t>& feature_shape) {
        std::vector<int64_t> shape = {static_cast<int64_t>(n_samples)};
        shape.insert(shape.end(), feature_shape.begin(), feature_shape.end());

        auto tensor = zeros(shape, DType::Float32, Device::cpu());
        auto* data = tensor.data<float>();
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            data[i] = static_cast<float>(i % 100) / 10.0f;
        }

        tensor = tensor.to(dtype());
        if (device() != Device::cpu()) {
            tensor = tensor.to(device());
        }
        return tensor;
    }

    Tensor createTestTargets(size_t n_samples) {
        auto tensor = zeros({static_cast<int64_t>(n_samples)}, DType::Float32, Device::cpu());
        auto* data = tensor.data<float>();
        for (size_t i = 0; i < n_samples; ++i) {
            data[i] = static_cast<float>(i % 10);
        }

        tensor = tensor.to(dtype());
        if (device() != Device::cpu()) {
            tensor = tensor.to(device());
        }
        return tensor;
    }
};

TEST_P(DataLoadingMultiDTypeTest, TensorDatasetBasicConstruction) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    TensorDataset dataset(inputs, targets);

    EXPECT_EQ(dataset.size(), 10);
    EXPECT_FALSE(dataset.empty());
}

TEST_P(DataLoadingMultiDTypeTest, TensorDatasetSingleElement) {
    auto inputs = createTestInputs(1, {3, 4});
    auto targets = createTestTargets(1);

    TensorDataset dataset(inputs, targets);
    EXPECT_EQ(dataset.size(), 1);

    auto [input, target] = dataset.get(0);
    expectShape(input, {3, 4});
}

TEST_P(DataLoadingMultiDTypeTest, TensorDatasetGetElement) {
    auto inputs = createTestInputs(20, {5, 5});
    auto targets = createTestTargets(20);

    TensorDataset dataset(inputs, targets);

    auto [input, target] = dataset.get(10);
    expectShape(input, {5, 5});
}

TEST_P(DataLoadingMultiDTypeTest, TensorDatasetOutOfRangeThrows) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    TensorDataset dataset(inputs, targets);

    EXPECT_THROW(dataset.get(10), std::out_of_range);
    EXPECT_THROW(dataset.get(100), std::out_of_range);
}

TEST_P(DataLoadingMultiDTypeTest, TensorDatasetMismatchedSizesThrows) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(15);

    EXPECT_THROW(TensorDataset(inputs, targets), std::invalid_argument);
}

TEST_P(DataLoadingMultiDTypeTest, ConcatDatasetBasicConcatenation) {
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(15, {5});
    auto targets2 = createTestTargets(15);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2};
    ConcatDataset concat_dataset(datasets);

    EXPECT_EQ(concat_dataset.size(), 25);
    EXPECT_FALSE(concat_dataset.empty());
}

TEST_P(DataLoadingMultiDTypeTest, ConcatDatasetEmptyThrows) {
    std::vector<std::shared_ptr<Dataset>> empty_datasets;
    EXPECT_THROW({ ConcatDataset concat(empty_datasets); }, std::invalid_argument);
}

TEST_P(DataLoadingMultiDTypeTest, TransformedDatasetPreservesSize) {
    auto inputs = createTestInputs(50, {10});
    auto targets = createTestTargets(50);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    auto transform_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input, target);
    };

    TransformedDataset dataset(base_dataset, transform_func);

    EXPECT_EQ(dataset.size(), base_dataset->size());
    EXPECT_EQ(dataset.size(), 50);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(DataLoadingMultiDTypeTest);
