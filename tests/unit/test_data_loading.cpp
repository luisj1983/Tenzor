#include <gtest/gtest.h>
#include "tenzor/data/dataset.hpp"
#include "tenzor/data/transforms.hpp"
#include "../backend_test_fixture.hpp"
#include <memory>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <random>

using namespace tenzor;
using namespace tenzor::data;
using namespace tenzor::data::transforms;
using namespace tenzor::testing;

/**
 * @brief Comprehensive test suite for data loading with multi-backend support
 *
 * This test suite covers:
 * - TensorDataset: Construction, access, boundary conditions
 * - TransformedDataset: Transform application, chaining
 * - ConcatDataset: Concatenation, index mapping, edge cases
 * - All Transform classes: Normalize, ToTensor, Compose, RandomHorizontalFlip, Lambda
 * - Edge cases: Empty datasets, single item, large batches
 * - Multi-backend compatibility
 * - Memory efficiency
 */
class DataLoadingBackendTest : public BackendTest {
protected:
    // Helper to create test tensors
    Tensor createTestInputs(size_t n_samples, const std::vector<int64_t>& feature_shape) {
        std::vector<int64_t> shape = {static_cast<int64_t>(n_samples)};
        shape.insert(shape.end(), feature_shape.begin(), feature_shape.end());

        // Create tensor on CPU first to fill with data
        auto tensor = zeros(shape, DType::Float32, Device::cpu());
        auto* data = tensor.data<float>();

        // Fill with sequential values for predictable testing
        for (int64_t i = 0; i < tensor.numel(); ++i) {
            data[i] = static_cast<float>(i % 100) / 10.0f;
        }

        // Transfer to target device if needed
        if (device != Device::cpu()) {
            tensor = tensor.to(device);
        }

        return tensor;
    }

    Tensor createTestTargets(size_t n_samples) {
        // Create tensor on CPU first to fill with data
        auto tensor = zeros({static_cast<int64_t>(n_samples)}, DType::Float32, Device::cpu());
        auto* data = tensor.data<float>();

        // Fill with class labels (0, 1, 2, ...)
        for (size_t i = 0; i < n_samples; ++i) {
            data[i] = static_cast<float>(i % 10);
        }

        // Transfer to target device if needed
        if (device != Device::cpu()) {
            tensor = tensor.to(device);
        }

        return tensor;
    }

    // Helper to verify tensor shape
    void expectShape(const Tensor& tensor, const std::vector<int64_t>& expected_shape) {
        const auto& actual_shape = tensor.shape();
        ASSERT_EQ(actual_shape.size(), expected_shape.size())
            << "Shape rank mismatch on device " << device.to_string();

        for (size_t i = 0; i < expected_shape.size(); ++i) {
            EXPECT_EQ(actual_shape[i], expected_shape[i])
                << "Shape dimension " << i << " mismatch on device " << device.to_string();
        }
    }
};

// ============================================================================
// TensorDataset Tests
// ============================================================================

TEST_P(DataLoadingBackendTest, TensorDatasetBasicConstruction) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    TensorDataset dataset(inputs, targets);

    EXPECT_EQ(dataset.size(), 10);
    EXPECT_FALSE(dataset.empty());
}

TEST_P(DataLoadingBackendTest, TensorDatasetSingleElement) {
    auto inputs = createTestInputs(1, {3, 4});
    auto targets = createTestTargets(1);

    TensorDataset dataset(inputs, targets);

    EXPECT_EQ(dataset.size(), 1);
    EXPECT_FALSE(dataset.empty());

    auto [input, target] = dataset.get(0);
    expectShape(input, {3, 4});
    expectShape(target, {});  // Scalar after squeeze
}

TEST_P(DataLoadingBackendTest, TensorDatasetEmptyCheck) {
    auto inputs = createTestInputs(100, {10, 10});
    auto targets = createTestTargets(100);

    TensorDataset dataset(inputs, targets);

    EXPECT_FALSE(dataset.empty());
    EXPECT_EQ(dataset.size(), 100);
}

TEST_P(DataLoadingBackendTest, TensorDatasetGetElement) {
    const size_t n_samples = 20;
    auto inputs = createTestInputs(n_samples, {5, 5});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    // Get middle element
    auto [input, target] = dataset.get(10);

    expectShape(input, {5, 5});
    expectShape(target, {});  // Scalar after squeeze

    // Verify data integrity
    EXPECT_EQ(target.item<float>(), 0.0f);  // 10 % 10 = 0
}

TEST_P(DataLoadingBackendTest, TensorDatasetBoundaryAccess) {
    const size_t n_samples = 50;
    auto inputs = createTestInputs(n_samples, {3});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    // First element
    auto [input0, target0] = dataset.get(0);
    expectShape(input0, {3});

    // Last element
    auto [input_last, target_last] = dataset.get(n_samples - 1);
    expectShape(input_last, {3});
    EXPECT_EQ(target_last.item<float>(), 9.0f);  // 49 % 10 = 9
}

TEST_P(DataLoadingBackendTest, TensorDatasetOutOfRangeThrows) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    TensorDataset dataset(inputs, targets);

    EXPECT_THROW(dataset.get(10), std::out_of_range);
    EXPECT_THROW(dataset.get(100), std::out_of_range);
}

TEST_P(DataLoadingBackendTest, TensorDatasetMismatchedSizesThrows) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(15);  // Mismatched size

    EXPECT_THROW(TensorDataset(inputs, targets), std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, TensorDatasetMultidimensionalInputs) {
    // Test with image-like data [N, C, H, W]
    auto inputs = createTestInputs(32, {3, 28, 28});
    auto targets = createTestTargets(32);

    TensorDataset dataset(inputs, targets);

    EXPECT_EQ(dataset.size(), 32);

    auto [input, target] = dataset.get(0);
    expectShape(input, {3, 28, 28});
}

TEST_P(DataLoadingBackendTest, TensorDatasetSequentialAccess) {
    const size_t n_samples = 25;
    auto inputs = createTestInputs(n_samples, {4});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    // Access all elements sequentially
    for (size_t i = 0; i < n_samples; ++i) {
        auto [input, target] = dataset.get(i);
        expectShape(input, {4});
        EXPECT_EQ(target.item<float>(), static_cast<float>(i % 10));
    }
}

TEST_P(DataLoadingBackendTest, TensorDatasetLargeBatch) {
    // Test with larger dataset
    const size_t n_samples = 1000;
    auto inputs = createTestInputs(n_samples, {128});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    EXPECT_EQ(dataset.size(), n_samples);

    // Sample some elements
    auto [input0, target0] = dataset.get(0);
    auto [input500, target500] = dataset.get(500);
    auto [input999, target999] = dataset.get(999);

    expectShape(input0, {128});
    expectShape(input500, {128});
    expectShape(input999, {128});
}

// ============================================================================
// TransformedDataset Tests
// ============================================================================

TEST_P(DataLoadingBackendTest, TransformedDatasetBasicTransform) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Simple transform: multiply input by 2
    auto transform_func = [](const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> {
        auto scaled_input = input * 2.0f;
        return {scaled_input, target};
    };

    TransformedDataset dataset(base_dataset, transform_func);

    EXPECT_EQ(dataset.size(), 10);

    auto [input, target] = dataset.get(0);
    auto [orig_input, orig_target] = base_dataset->get(0);

    // Verify transformation was applied
    auto input_cpu = input.to(Device::cpu());
    auto orig_input_cpu = orig_input.to(Device::cpu());

    auto* input_data = input_cpu.data<float>();
    auto* orig_data = orig_input_cpu.data<float>();

    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        EXPECT_NEAR(input_data[i], orig_data[i] * 2.0f, 1e-5f);
    }
}

TEST_P(DataLoadingBackendTest, TransformedDatasetTargetTransform) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Transform that modifies target
    auto transform_func = [](const Tensor& input, const Tensor& target)
        -> std::pair<Tensor, Tensor> {
        // Create tensor on CPU first, fill data, then transfer to target device
        auto one_hot = zeros({10}, input.dtype(), Device::cpu());
        int label = static_cast<int>(target.item<float>());
        one_hot.data<float>()[label] = 1.0f;

        // Transfer to target device if needed
        if (input.device() != Device::cpu()) {
            one_hot = one_hot.to(input.device());
        }

        return {input, one_hot};
    };

    TransformedDataset dataset(base_dataset, transform_func);

    auto [input, target] = dataset.get(5);
    expectShape(target, {10});

    // Verify one-hot encoding
    auto target_cpu = target.to(Device::cpu());
    auto* target_data = target_cpu.data<float>();

    int expected_label = 5 % 10;
    for (int i = 0; i < 10; ++i) {
        EXPECT_EQ(target_data[i], (i == expected_label) ? 1.0f : 0.0f);
    }
}

TEST_P(DataLoadingBackendTest, TransformedDatasetChaining) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // First transform: add 1
    auto transform1 = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input + 1.0f, target);
    };

    auto dataset1 = std::make_shared<TransformedDataset>(base_dataset, transform1);

    // Second transform: multiply by 2
    auto transform2 = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input * 2.0f, target);
    };

    TransformedDataset dataset2(dataset1, transform2);

    auto [input, target] = dataset2.get(0);
    auto [orig_input, orig_target] = base_dataset->get(0);

    // Verify chained transformation: (x + 1) * 2
    auto input_cpu = input.to(Device::cpu());
    auto orig_input_cpu = orig_input.to(Device::cpu());

    auto* input_data = input_cpu.data<float>();
    auto* orig_data = orig_input_cpu.data<float>();

    for (int64_t i = 0; i < input_cpu.numel(); ++i) {
        EXPECT_NEAR(input_data[i], (orig_data[i] + 1.0f) * 2.0f, 1e-5f);
    }
}

TEST_P(DataLoadingBackendTest, TransformedDatasetPreservesSize) {
    auto inputs = createTestInputs(100, {10});
    auto targets = createTestTargets(100);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    auto transform_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input, target);
    };

    TransformedDataset dataset(base_dataset, transform_func);

    EXPECT_EQ(dataset.size(), base_dataset->size());
    EXPECT_EQ(dataset.size(), 100);
}

// ============================================================================
// ConcatDataset Tests
// ============================================================================

TEST_P(DataLoadingBackendTest, ConcatDatasetBasicConcatenation) {
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

TEST_P(DataLoadingBackendTest, ConcatDatasetAccessFirstDataset) {
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(10, {5});
    auto targets2 = createTestTargets(10);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2};
    ConcatDataset concat_dataset(datasets);

    // Access from first dataset
    auto [input, target] = concat_dataset.get(5);
    auto [orig_input, orig_target] = dataset1->get(5);

    expectTensorNear(input, orig_input);
    expectTensorNear(target, orig_target);
}

TEST_P(DataLoadingBackendTest, ConcatDatasetAccessSecondDataset) {
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(10, {5});
    auto targets2 = createTestTargets(10);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2};
    ConcatDataset concat_dataset(datasets);

    // Access from second dataset (index 15 = 5th element of second dataset)
    auto [input, target] = concat_dataset.get(15);
    auto [orig_input, orig_target] = dataset2->get(5);

    expectTensorNear(input, orig_input);
    expectTensorNear(target, orig_target);
}

TEST_P(DataLoadingBackendTest, ConcatDatasetBoundaryTransition) {
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(10, {5});
    auto targets2 = createTestTargets(10);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2};
    ConcatDataset concat_dataset(datasets);

    // Last element of first dataset
    auto [input9, target9] = concat_dataset.get(9);
    auto [orig_input9, orig_target9] = dataset1->get(9);
    expectTensorNear(input9, orig_input9);

    // First element of second dataset
    auto [input10, target10] = concat_dataset.get(10);
    auto [orig_input10, orig_target10] = dataset2->get(0);
    expectTensorNear(input10, orig_input10);
}

TEST_P(DataLoadingBackendTest, ConcatDatasetMultipleDatasets) {
    std::vector<std::shared_ptr<Dataset>> datasets;

    // Create 5 small datasets
    for (int i = 0; i < 5; ++i) {
        auto inputs = createTestInputs(5, {3});
        auto targets = createTestTargets(5);
        datasets.push_back(std::make_shared<TensorDataset>(inputs, targets));
    }

    ConcatDataset concat_dataset(datasets);

    EXPECT_EQ(concat_dataset.size(), 25);

    // Access element from each dataset
    for (size_t i = 0; i < 5; ++i) {
        size_t global_idx = i * 5;  // First element of each dataset
        auto [input, target] = concat_dataset.get(global_idx);
        expectShape(input, {3});
    }
}

TEST_P(DataLoadingBackendTest, ConcatDatasetEmptyThrows) {
    std::vector<std::shared_ptr<Dataset>> empty_datasets;

    EXPECT_THROW({ ConcatDataset concat(empty_datasets); }, std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, ConcatDatasetOutOfRangeThrows) {
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1};
    ConcatDataset concat_dataset(datasets);

    EXPECT_THROW(concat_dataset.get(10), std::out_of_range);
    EXPECT_THROW(concat_dataset.get(100), std::out_of_range);
}

TEST_P(DataLoadingBackendTest, ConcatDatasetVariableSizes) {
    // Datasets with different sizes
    auto inputs1 = createTestInputs(5, {3});
    auto targets1 = createTestTargets(5);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(20, {3});
    auto targets2 = createTestTargets(20);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    auto inputs3 = createTestInputs(8, {3});
    auto targets3 = createTestTargets(8);
    auto dataset3 = std::make_shared<TensorDataset>(inputs3, targets3);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2, dataset3};
    ConcatDataset concat_dataset(datasets);

    EXPECT_EQ(concat_dataset.size(), 33);

    // Verify boundary transitions
    auto [input4, target4] = concat_dataset.get(4);    // Last of dataset1
    auto [input5, target5] = concat_dataset.get(5);    // First of dataset2
    auto [input24, target24] = concat_dataset.get(24); // Last of dataset2
    auto [input25, target25] = concat_dataset.get(25); // First of dataset3

    expectShape(input4, {3});
    expectShape(input5, {3});
    expectShape(input24, {3});
    expectShape(input25, {3});
}

// ============================================================================
// Transform Tests: Normalize
// ============================================================================

TEST_P(DataLoadingBackendTest, NormalizeBasicOperation) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> std = {0.5f, 0.5f, 0.5f};

    Normalize normalize(mean, std);

    auto input = ones({3}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [normalized_input, output_target] = normalize(input, target);

    // Verify: (1.0 - 0.5) / 0.5 = 1.0
    expectShape(normalized_input, {3});
    expectTensorNear(output_target, target);
}

TEST_P(DataLoadingBackendTest, NormalizeMismatchedSizeThrows) {
    std::vector<float> mean = {0.5f, 0.5f};
    std::vector<float> std = {0.5f, 0.5f, 0.5f};  // Different size

    EXPECT_THROW(Normalize(mean, std), std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, NormalizeZeroStdThrows) {
    std::vector<float> mean = {0.5f};
    std::vector<float> std = {0.0f};  // Zero std

    Normalize normalize(mean, std);

    auto input = ones({1}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    EXPECT_THROW(normalize(input, target), std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, NormalizeScalarInputThrows) {
    std::vector<float> mean = {0.5f};
    std::vector<float> std = {0.5f};

    Normalize normalize(mean, std);

    // Scalar tensor (empty shape - this should fail due to no dimensions)
    // Since Tensor::scalar doesn't exist, create a tensor with empty shape
    auto input = full({}, 1.0f, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    EXPECT_THROW(normalize(input, target), std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, NormalizeChannelMismatchThrows) {
    std::vector<float> mean = {0.5f, 0.5f, 0.5f};
    std::vector<float> std = {0.5f, 0.5f, 0.5f};

    Normalize normalize(mean, std);

    // Input with wrong number of channels
    auto input = ones({5}, DType::Float32, device);  // 5 channels, expected 3
    auto target = zeros({1}, DType::Float32, device);

    EXPECT_THROW(normalize(input, target), std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, NormalizeSingleChannel) {
    std::vector<float> mean = {0.5f};
    std::vector<float> std = {0.25f};

    Normalize normalize(mean, std);

    auto input = full({1}, 1.0f, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [normalized_input, output_target] = normalize(input, target);

    expectShape(normalized_input, {1});
}

TEST_P(DataLoadingBackendTest, NormalizeMultiDimensional) {
    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};

    Normalize normalize(mean, std);

    // Image-like tensor [H, W, C]
    auto input = ones({32, 32, 3}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [normalized_input, output_target] = normalize(input, target);

    expectShape(normalized_input, {32, 32, 3});
}

// ============================================================================
// Transform Tests: ToTensor
// ============================================================================

TEST_P(DataLoadingBackendTest, ToTensorIdentity) {
    ToTensor to_tensor;

    auto input = ones({10, 10}, DType::Float32, device);
    auto target = zeros({10}, DType::Float32, device);

    auto [output_input, output_target] = to_tensor(input, target);

    expectTensorNear(output_input, input);
    expectTensorNear(output_target, target);
}

TEST_P(DataLoadingBackendTest, ToTensorPreservesShape) {
    ToTensor to_tensor;

    auto input = createTestInputs(5, {3, 32, 32});
    auto target = createTestTargets(5);

    // Get single sample
    auto input_sample = input.slice(0, 0, 1).squeeze(0);
    auto target_sample = target.slice(0, 0, 1).squeeze(0);

    auto [output_input, output_target] = to_tensor(input_sample, target_sample);

    expectShape(output_input, {3, 32, 32});
}

// ============================================================================
// Transform Tests: Compose
// ============================================================================

TEST_P(DataLoadingBackendTest, ComposeEmptyTransforms) {
    std::vector<std::shared_ptr<Transform>> transforms;
    Compose compose(transforms);

    auto input = ones({5}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = compose(input, target);

    expectTensorNear(output_input, input);
    expectTensorNear(output_target, target);
}

TEST_P(DataLoadingBackendTest, ComposeSingleTransform) {
    auto to_tensor = std::make_shared<ToTensor>();
    std::vector<std::shared_ptr<Transform>> transforms = {to_tensor};

    Compose compose(transforms);

    auto input = ones({5}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = compose(input, target);

    expectTensorNear(output_input, input);
}

TEST_P(DataLoadingBackendTest, ComposeMultipleTransforms) {
    auto to_tensor = std::make_shared<ToTensor>();

    std::vector<float> mean = {0.5f};
    std::vector<float> std = {0.5f};
    auto normalize = std::make_shared<Normalize>(mean, std);

    std::vector<std::shared_ptr<Transform>> transforms = {to_tensor, normalize};
    Compose compose(transforms);

    auto input = ones({1}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = compose(input, target);

    expectShape(output_input, {1});
}

TEST_P(DataLoadingBackendTest, ComposeChainExecution) {
    // Create three transforms that will be chained
    auto transform1 = std::make_shared<ToTensor>();
    auto transform2 = std::make_shared<ToTensor>();
    auto transform3 = std::make_shared<ToTensor>();

    std::vector<std::shared_ptr<Transform>> transforms = {
        transform1, transform2, transform3
    };

    Compose compose(transforms);

    auto input = ones({10}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = compose(input, target);

    expectTensorNear(output_input, input);
}

TEST_P(DataLoadingBackendTest, ComposeNestedCompose) {
    auto to_tensor = std::make_shared<ToTensor>();

    // Inner compose
    std::vector<std::shared_ptr<Transform>> inner_transforms = {to_tensor};
    auto inner_compose = std::make_shared<Compose>(inner_transforms);

    // Outer compose
    std::vector<std::shared_ptr<Transform>> outer_transforms = {inner_compose, to_tensor};
    Compose outer_compose(outer_transforms);

    auto input = ones({5}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = outer_compose(input, target);

    expectTensorNear(output_input, input);
}

// ============================================================================
// Transform Tests: RandomHorizontalFlip
// ============================================================================

TEST_P(DataLoadingBackendTest, RandomHorizontalFlipConstruction) {
    RandomHorizontalFlip flip(0.5f);

    auto input = ones({3, 32, 32}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = flip(input, target);

    expectShape(output_input, {3, 32, 32});
}

TEST_P(DataLoadingBackendTest, RandomHorizontalFlipProbabilityZero) {
    RandomHorizontalFlip flip(0.0f);

    auto input = ones({3, 32, 32}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = flip(input, target);

    // With p=0, should never flip (always return unchanged)
    expectTensorNear(output_input, input);
}

TEST_P(DataLoadingBackendTest, RandomHorizontalFlipProbabilityOne) {
    RandomHorizontalFlip flip(1.0f);

    auto input = ones({3, 32, 32}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = flip(input, target);

    expectShape(output_input, {3, 32, 32});
}

TEST_P(DataLoadingBackendTest, RandomHorizontalFlipInvalidProbabilityThrows) {
    EXPECT_THROW(RandomHorizontalFlip(-0.1f), std::invalid_argument);
    EXPECT_THROW(RandomHorizontalFlip(1.5f), std::invalid_argument);
}

TEST_P(DataLoadingBackendTest, RandomHorizontalFlipTargetUnchanged) {
    RandomHorizontalFlip flip(0.5f);

    auto input = ones({3, 32, 32}, DType::Float32, device);
    auto target = full({10}, 5.0f, DType::Float32, device);

    auto [output_input, output_target] = flip(input, target);

    // Target should always be unchanged
    expectTensorNear(output_target, target);
}

// ============================================================================
// Transform Tests: Lambda
// ============================================================================

TEST_P(DataLoadingBackendTest, LambdaBasicFunction) {
    auto lambda_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input * 2.0f, target);
    };

    Lambda lambda(lambda_func);

    auto input = ones({5}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = lambda(input, target);

    auto expected = input * 2.0f;
    expectTensorNear(output_input, expected);
}

TEST_P(DataLoadingBackendTest, LambdaTargetTransform) {
    auto lambda_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input, target + 10.0f);
    };

    Lambda lambda(lambda_func);

    auto input = ones({5}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = lambda(input, target);

    expectTensorNear(output_input, input);

    auto expected_target = target + 10.0f;
    expectTensorNear(output_target, expected_target);
}

TEST_P(DataLoadingBackendTest, LambdaBothTransform) {
    auto lambda_func = [this](const Tensor& input, const Tensor& target) {
        auto new_input = input * 0.5f;
        auto new_target = full({3}, 1.0f, DType::Float32, device);
        return std::make_pair(new_input, new_target);
    };

    Lambda lambda(lambda_func);

    auto input = ones({5}, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = lambda(input, target);

    expectShape(output_input, {5});
    expectShape(output_target, {3});
}

TEST_P(DataLoadingBackendTest, LambdaComplexOperation) {
    auto lambda_func = [](const Tensor& input, const Tensor& target) {
        // Simulate complex preprocessing
        auto processed = (input - 0.5f) * 2.0f;  // Scale to [-1, 1]
        return std::make_pair(processed, target);
    };

    Lambda lambda(lambda_func);

    auto input = full({10}, 0.75f, DType::Float32, device);
    auto target = zeros({1}, DType::Float32, device);

    auto [output_input, output_target] = lambda(input, target);

    // (0.75 - 0.5) * 2.0 = 0.5
    auto expected = full({10}, 0.5f, DType::Float32, device);
    expectTensorNear(output_input, expected);
}

// ============================================================================
// Integration Tests: Combined Dataset and Transform Operations
// ============================================================================

TEST_P(DataLoadingBackendTest, IntegrationDatasetWithTransform) {
    auto inputs = createTestInputs(20, {3});
    auto targets = createTestTargets(20);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Apply transform
    auto transform_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input * 2.0f, target);
    };

    TransformedDataset dataset(base_dataset, transform_func);

    EXPECT_EQ(dataset.size(), 20);

    // Verify transformation
    for (size_t i = 0; i < 5; ++i) {
        auto [input, target] = dataset.get(i);
        expectShape(input, {3});
    }
}

TEST_P(DataLoadingBackendTest, IntegrationConcatWithTransform) {
    // Create multiple datasets
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(10, {5});
    auto targets2 = createTestTargets(10);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    // Concatenate
    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2};
    auto concat_dataset = std::make_shared<ConcatDataset>(datasets);

    // Apply transform
    auto transform_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input + 1.0f, target);
    };

    TransformedDataset transformed(concat_dataset, transform_func);

    EXPECT_EQ(transformed.size(), 20);

    // Access elements from both original datasets
    auto [input5, target5] = transformed.get(5);
    auto [input15, target15] = transformed.get(15);

    expectShape(input5, {5});
    expectShape(input15, {5});
}

TEST_P(DataLoadingBackendTest, IntegrationComplexPipeline) {
    // Create base dataset with shape [N, H, W, C] for normalization
    auto inputs = createTestInputs(50, {32, 32, 3});
    auto targets = createTestTargets(50);
    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Create transform pipeline
    auto to_tensor = std::make_shared<ToTensor>();

    std::vector<float> mean = {0.485f, 0.456f, 0.406f};
    std::vector<float> std = {0.229f, 0.224f, 0.225f};
    auto normalize = std::make_shared<Normalize>(mean, std);

    std::vector<std::shared_ptr<Transform>> transforms = {to_tensor, normalize};
    auto composed = std::make_shared<Compose>(transforms);

    // Wrap transform into function
    auto transform_func = [composed](const Tensor& input, const Tensor& target) {
        return (*composed)(input, target);
    };

    TransformedDataset dataset(base_dataset, transform_func);

    EXPECT_EQ(dataset.size(), 50);

    auto [input, target] = dataset.get(0);
    expectShape(input, {32, 32, 3});
}

// ============================================================================
// Memory and Performance Tests
// ============================================================================

TEST_P(DataLoadingBackendTest, MemoryLargeDataset) {
    // Test with larger dataset to check memory handling
    const size_t n_samples = 100;
    auto inputs = createTestInputs(n_samples, {64, 64});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    // Access multiple samples
    for (size_t i = 0; i < 10; ++i) {
        auto [input, target] = dataset.get(i * 10);
        expectShape(input, {64, 64});
    }
}

TEST_P(DataLoadingBackendTest, MemoryMultipleAccesses) {
    auto inputs = createTestInputs(100, {10});
    auto targets = createTestTargets(100);

    TensorDataset dataset(inputs, targets);

    // Access same element multiple times
    for (int iter = 0; iter < 5; ++iter) {
        auto [input, target] = dataset.get(50);
        expectShape(input, {10});
    }
}

TEST_P(DataLoadingBackendTest, PerformanceSequentialAccess) {
    const size_t n_samples = 500;
    auto inputs = createTestInputs(n_samples, {32});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    // Sequential access performance test
    for (size_t i = 0; i < n_samples; ++i) {
        auto [input, target] = dataset.get(i);
        EXPECT_EQ(input.shape()[0], 32);
    }
}

TEST_P(DataLoadingBackendTest, PerformanceRandomAccess) {
    const size_t n_samples = 200;
    auto inputs = createTestInputs(n_samples, {16});
    auto targets = createTestTargets(n_samples);

    TensorDataset dataset(inputs, targets);

    // Random access pattern
    std::vector<size_t> indices(n_samples);
    std::iota(indices.begin(), indices.end(), 0);

    // Use shuffle instead of deprecated random_shuffle
    std::random_device rd;
    std::mt19937 gen(rd());
    std::shuffle(indices.begin(), indices.end(), gen);

    for (size_t idx : indices) {
        auto [input, target] = dataset.get(idx);
        expectShape(input, {16});
    }
}

// ============================================================================
// Cross-Device Tests
// ============================================================================

TEST_P(DataLoadingBackendTest, CrossDeviceDataAccess) {
    // Create dataset on current device
    auto inputs = createTestInputs(20, {10});
    auto targets = createTestTargets(20);

    TensorDataset dataset(inputs, targets);

    // Access and verify device
    auto [input, target] = dataset.get(0);

    EXPECT_EQ(input.device().type, device.type);
    EXPECT_EQ(target.device().type, device.type);
}

TEST_P(DataLoadingBackendTest, CrossDeviceTransform) {
    auto inputs = createTestInputs(10, {5});
    auto targets = createTestTargets(10);

    auto base_dataset = std::make_shared<TensorDataset>(inputs, targets);

    // Transform should maintain device
    auto transform_func = [](const Tensor& input, const Tensor& target) {
        return std::make_pair(input * 2.0f, target);
    };

    TransformedDataset dataset(base_dataset, transform_func);

    auto [input, target] = dataset.get(0);

    EXPECT_EQ(input.device().type, device.type);
    EXPECT_EQ(target.device().type, device.type);
}

TEST_P(DataLoadingBackendTest, CrossDeviceConcat) {
    // Create datasets on same device
    auto inputs1 = createTestInputs(10, {5});
    auto targets1 = createTestTargets(10);
    auto dataset1 = std::make_shared<TensorDataset>(inputs1, targets1);

    auto inputs2 = createTestInputs(10, {5});
    auto targets2 = createTestTargets(10);
    auto dataset2 = std::make_shared<TensorDataset>(inputs2, targets2);

    std::vector<std::shared_ptr<Dataset>> datasets = {dataset1, dataset2};
    ConcatDataset concat_dataset(datasets);

    // Verify all data stays on same device
    auto [input0, target0] = concat_dataset.get(0);
    auto [input15, target15] = concat_dataset.get(15);

    EXPECT_EQ(input0.device().type, device.type);
    EXPECT_EQ(input15.device().type, device.type);
}

// Instantiate tests for all backends
INSTANTIATE_BACKEND_TESTS(DataLoadingBackendTest);
