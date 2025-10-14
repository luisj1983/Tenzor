#pragma once

#include <cstddef>
#include <memory>
#include <utility>
#include <vector>
#include <functional>
#include "../tenzor.hpp"

namespace tenzor {
namespace data {

/**
 * @brief Abstract base class for datasets
 *
 * Dataset provides a standard interface for accessing data samples.
 * Each sample consists of a pair of tensors (input, target).
 */
class Dataset {
public:
    virtual ~Dataset() = default;

    /**
     * @brief Get the size of the dataset
     * @return Number of samples in the dataset
     */
    virtual auto size() const -> size_t = 0;

    /**
     * @brief Get a sample at the specified index
     * @param index Index of the sample to retrieve
     * @return Pair of tensors (input, target)
     */
    virtual auto get(size_t index) -> std::pair<Tensor, Tensor> = 0;

    /**
     * @brief Check if dataset is empty
     * @return true if dataset has no samples
     */
    virtual auto empty() const -> bool {
        return size() == 0;
    }
};

/**
 * @brief Map-style dataset that supports random access
 *
 * Suitable for datasets where samples can be accessed by index.
 */
class MapDataset : public Dataset {
public:
    virtual ~MapDataset() = default;
};

/**
 * @brief Iterable-style dataset for streaming data
 *
 * Suitable for datasets that are generated on-the-fly or streamed.
 */
class IterableDataset : public Dataset {
public:
    virtual ~IterableDataset() = default;
};

/**
 * @brief Simple tensor dataset holding data in memory
 *
 * Stores input and target tensors directly in memory for fast access.
 */
class TensorDataset : public MapDataset {
public:
    /**
     * @brief Construct dataset from input and target tensors
     * @param inputs Input tensor with shape [N, ...]
     * @param targets Target tensor with shape [N, ...]
     */
    TensorDataset(Tensor inputs, Tensor targets)
        : inputs_(std::move(inputs)), targets_(std::move(targets)) {
        if (inputs_.shape()[0] != targets_.shape()[0]) {
            throw std::invalid_argument("Input and target batch dimensions must match");
        }
    }

    auto size() const -> size_t override {
        return inputs_.shape()[0];
    }

    auto get(size_t index) -> std::pair<Tensor, Tensor> override {
        if (index >= size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        // Extract single sample by slicing along batch dimension
        auto input = inputs_.slice(0, index, index + 1).squeeze(0);
        auto target = targets_.slice(0, index, index + 1).squeeze(0);

        return {input, target};
    }

private:
    Tensor inputs_;
    Tensor targets_;
};

/**
 * @brief Dataset wrapper that applies transforms to samples
 *
 * Wraps an existing dataset and applies a transform function to each sample.
 */
class TransformedDataset : public Dataset {
public:
    using TransformFunc = std::function<std::pair<Tensor, Tensor>(const Tensor&, const Tensor&)>;

    /**
     * @brief Construct transformed dataset
     * @param dataset Base dataset to transform
     * @param transform Function to apply to each sample
     */
    TransformedDataset(std::shared_ptr<Dataset> dataset, TransformFunc transform)
        : dataset_(std::move(dataset)), transform_(std::move(transform)) {}

    auto size() const -> size_t override {
        return dataset_->size();
    }

    auto get(size_t index) -> std::pair<Tensor, Tensor> override {
        auto [input, target] = dataset_->get(index);
        return transform_(input, target);
    }

private:
    std::shared_ptr<Dataset> dataset_;
    TransformFunc transform_;
};

/**
 * @brief Dataset that concatenates multiple datasets
 */
class ConcatDataset : public Dataset {
public:
    /**
     * @brief Construct concatenated dataset
     * @param datasets Vector of datasets to concatenate
     */
    explicit ConcatDataset(std::vector<std::shared_ptr<Dataset>> datasets)
        : datasets_(std::move(datasets)) {
        if (datasets_.empty()) {
            throw std::invalid_argument("Cannot create empty ConcatDataset");
        }

        // Precompute cumulative sizes
        cumulative_sizes_.push_back(0);
        for (const auto& dataset : datasets_) {
            cumulative_sizes_.push_back(cumulative_sizes_.back() + dataset->size());
        }
    }

    auto size() const -> size_t override {
        return cumulative_sizes_.back();
    }

    auto get(size_t index) -> std::pair<Tensor, Tensor> override {
        if (index >= size()) {
            throw std::out_of_range("Dataset index out of range");
        }

        // Binary search to find which dataset contains this index
        auto it = std::upper_bound(cumulative_sizes_.begin(), cumulative_sizes_.end(), index);
        size_t dataset_idx = std::distance(cumulative_sizes_.begin(), it) - 1;
        size_t local_idx = index - cumulative_sizes_[dataset_idx];

        return datasets_[dataset_idx]->get(local_idx);
    }

private:
    std::vector<std::shared_ptr<Dataset>> datasets_;
    std::vector<size_t> cumulative_sizes_;
};

} // namespace data
} // namespace tenzor
