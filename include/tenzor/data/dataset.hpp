#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <functional>
#include "../tenzor.hpp"

namespace tenzor {
namespace data {

/**
 * @brief Metadata describing the current DataLoader worker context
 *
 * When a DataLoader uses multiple workers, each worker thread receives
 * a WorkerInfo instance via set_worker_info() before processing begins.
 * IterableDataset subclasses can call get_worker_info() to shard their
 * data stream across workers and distributed ranks.
 */
struct WorkerInfo {
    int worker_id{0};       ///< Index of this worker within the DataLoader (0-based)
    int num_workers{1};     ///< Total number of workers in this DataLoader
    int64_t seed{0};        ///< Per-worker random seed for reproducibility
    int rank{0};            ///< Distributed rank of this process
    int world_size{1};      ///< Total number of distributed processes
};

/**
 * @brief Get the WorkerInfo for the current DataLoader worker thread
 * @return WorkerInfo if called from a worker thread, std::nullopt otherwise
 */
auto get_worker_info() -> std::optional<WorkerInfo>;

/**
 * @brief Set the WorkerInfo for the current thread
 * @param info WorkerInfo to associate with the current thread
 */
auto set_worker_info(WorkerInfo info) -> void;

/**
 * @brief Clear the WorkerInfo for the current thread
 */
auto clear_worker_info() -> void;

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
 * Unlike MapDataset, IterableDataset does not support random access or
 * known size. Subclasses should use get_worker_info() to partition their
 * data stream across DataLoader workers and distributed ranks.
 *
 * Example:
 * @code
 * class MyStream : public IterableDataset {
 * public:
 *     auto size() const -> size_t override {
 *         throw std::runtime_error("IterableDataset does not support size()");
 *     }
 *     auto get(size_t) -> std::pair<Tensor, Tensor> override {
 *         throw std::runtime_error("IterableDataset does not support random access");
 *     }
 * };
 * @endcode
 */
class IterableDataset : public Dataset {
public:
    ~IterableDataset() override = default;

    /**
     * @brief Size is unknown for iterable datasets
     * @throws std::runtime_error always
     */
    auto size() const -> size_t override {
        throw std::runtime_error("IterableDataset does not support size()");
    }

    /**
     * @brief Random access is not supported for iterable datasets
     * @throws std::runtime_error always
     */
    auto get(size_t /*index*/) -> std::pair<Tensor, Tensor> override {
        throw std::runtime_error("IterableDataset does not support random access");
    }
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
        if (inputs_.ndim() == 0 || targets_.ndim() == 0) {
            throw std::invalid_argument(
                "TensorDataset requires tensors with a batch dimension");
        }
        if (inputs_.shape()[0] != targets_.shape()[0]) {
            throw std::invalid_argument("Input and target batch dimensions must match");
        }
    }

    auto size() const -> size_t override {
        if (inputs_.ndim() == 0) {
            throw std::runtime_error(
                "TensorDataset requires tensors with a batch dimension");
        }
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
