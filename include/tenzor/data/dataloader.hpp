#pragma once

#include <memory>
#include <vector>
#include <queue>
#include <map>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include <random>
#include <algorithm>
#include <functional>
#include "dataset.hpp"

namespace tenzor {
namespace data {

/**
 * @brief Batch of samples for training
 *
 * Contains batched input and target tensors.
 */
struct Batch {
    Tensor inputs;   ///< Batched input tensor [batch_size, ...]
    Tensor targets;  ///< Batched target tensor [batch_size, ...]

    Batch() = default;
    Batch(Tensor inputs_, Tensor targets_)
        : inputs(std::move(inputs_)), targets(std::move(targets_)) {}
};

/**
 * @brief Configuration for DataLoader
 */
/**
 * @brief Custom collation function type.
 *
 * Takes a vector of (input, target) sample pairs and returns a single
 * batched Batch. If not set, the default collation (tensor stacking)
 * is used.
 */
using CollateFn = std::function<Batch(const std::vector<std::pair<Tensor, Tensor>>&)>;

struct DataLoaderConfig {
    size_t batch_size = 1;        ///< Number of samples per batch
    bool shuffle = false;          ///< Whether to shuffle data each epoch
    size_t num_workers = 0;        ///< Number of worker threads (0 = single-threaded)
    bool pin_memory = false;       ///< Pin memory for faster CUDA transfer
    bool drop_last = false;        ///< Drop last incomplete batch
    size_t prefetch_factor = 2;    ///< Number of batches to prefetch per worker
    bool persistent_workers = false; ///< Keep workers alive between epochs
    CollateFn collate_fn;          ///< Optional custom collation function

    DataLoaderConfig() = default;
};

/**
 * @brief Multi-threaded data loader for efficient batch loading
 *
 * DataLoader provides efficient data loading with:
 * - Multi-threaded loading with worker pool
 * - Automatic batching and collation
 * - Data shuffling per epoch
 * - Prefetching for pipeline efficiency
 * - Pin memory option for CUDA
 *
 * Example usage:
 * @code
 * auto dataset = std::make_shared<TensorDataset>(inputs, targets);
 * DataLoaderConfig config;
 * config.batch_size = 32;
 * config.shuffle = true;
 * config.num_workers = 4;
 *
 * DataLoader loader(dataset, config);
 * for (const auto& batch : loader) {
 *     // Train on batch.inputs and batch.targets
 * }
 * @endcode
 */
class DataLoader {
public:
    /**
     * @brief Construct DataLoader with configuration
     * @param dataset Dataset to load from
     * @param config DataLoader configuration
     */
    DataLoader(std::shared_ptr<Dataset> dataset, const DataLoaderConfig& config);

    /**
     * @brief Construct DataLoader with individual parameters
     * @param dataset Dataset to load from
     * @param batch_size Number of samples per batch
     * @param shuffle Whether to shuffle data
     * @param num_workers Number of worker threads
     * @param pin_memory Pin memory for CUDA
     * @param drop_last Drop last incomplete batch
     */
    DataLoader(std::shared_ptr<Dataset> dataset,
               size_t batch_size,
               bool shuffle = false,
               size_t num_workers = 0,
               bool pin_memory = false,
               bool drop_last = false);

    ~DataLoader();

    // Prevent copying
    DataLoader(const DataLoader&) = delete;
    DataLoader& operator=(const DataLoader&) = delete;
    // Non-movable: worker threads are bound to `this`. Moving a running loader
    // would leave its threads executing on the moved-from object (accessing its
    // now-null dataset_ / moved-from queue) — a data race / use-after-move that
    // a constructor cannot avoid (the member-init list runs before any stop).

    DataLoader(DataLoader&&) = delete;
    DataLoader& operator=(DataLoader&&) = delete;

    /**
     * @brief Iterator for DataLoader
     *
     * Provides forward iteration over batches.
     */
    class Iterator {
    public:
        using iterator_category = std::input_iterator_tag;
        using value_type = Batch;
        using difference_type = std::ptrdiff_t;
        using pointer = Batch*;
        using reference = Batch&;

        Iterator(DataLoader* loader, size_t index);

        auto operator*() -> Batch&;
        auto operator->() -> Batch*;
        auto operator++() -> Iterator&;
        auto operator++(int) -> Iterator;

        auto operator==(const Iterator& other) const -> bool;
        auto operator!=(const Iterator& other) const -> bool;

    private:
        DataLoader* loader_;
        size_t index_;
        Batch current_batch_;
        bool valid_;

        void fetch_next();
    };

    /**
     * @brief Get iterator to first batch
     * @return Iterator to beginning
     */
    auto begin() -> Iterator;

    /**
     * @brief Get iterator past last batch
     * @return Iterator to end
     */
    auto end() -> Iterator;

    /**
     * @brief Get number of batches per epoch
     * @return Number of batches
     */
    auto size() const -> size_t;

    /**
     * @brief Reset loader for new epoch
     *
     * Resets internal state and reshuffles if enabled.
     */
    void reset();

private:
    std::shared_ptr<Dataset> dataset_;
    DataLoaderConfig config_;

    // Shuffling
    std::vector<size_t> indices_;
    std::mt19937 rng_;
    size_t current_index_;

    // Multi-threading
    std::vector<std::thread> workers_;
    // Workers push (batch_idx, batch) in completion order; the consumer
    // reorders them into ascending batch_idx via reorder_buffer_ so iteration
    // is deterministic regardless of which worker finished first.
    std::queue<std::pair<size_t, Batch>> batch_queue_;
    std::map<size_t, Batch> reorder_buffer_;  ///< out-of-order completed batches awaiting in-order emit
    size_t next_output_idx_{0};               ///< next batch index the consumer will emit
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::condition_variable worker_cv_;
    std::atomic<bool> stop_workers_;
    std::atomic<bool> epoch_done_;
    std::atomic<size_t> next_batch_idx_;
    std::atomic<size_t> active_workers_;  ///< Number of workers still processing
    std::exception_ptr worker_exception_;  ///< First exception from any worker thread

    // Persistent workers synchronization
    std::condition_variable epoch_start_cv_;  ///< Signals workers to start new epoch
    /// Monotonic epoch counter. reset() increments it once per epoch so every
    /// waiting persistent worker observes a change exactly once (a one-shot bool
    /// is lost-wakeup-prone: the first worker to wake clears it before the rest
    /// re-check the predicate, collapsing parallelism to a single worker).
    uint64_t epoch_generation_{0};

    // Batch management
    size_t num_batches_;

    /**
     * @brief Initialize indices for iteration
     */
    void init_indices();

    /**
     * @brief Shuffle indices for new epoch
     */
    void shuffle_indices();

    /**
     * @brief Collate samples into batch
     * @param samples Vector of (input, target) pairs
     * @return Collated batch
     */
    auto collate_samples(const std::vector<std::pair<Tensor, Tensor>>& samples) -> Batch;

    /**
     * @brief Worker thread function
     * @param worker_id ID of this worker
     */
    void worker_thread(size_t worker_id);

    /**
     * @brief Get next batch (single-threaded)
     * @return Next batch or empty batch if done
     */
    auto get_next_batch_single_threaded() -> Batch;

    /**
     * @brief Get next batch (multi-threaded)
     * @return Next batch or empty batch if done
     */
    auto get_next_batch_multi_threaded() -> Batch;

    /**
     * @brief Start worker threads
     */
    void start_workers();

    /**
     * @brief Stop worker threads
     */
    void stop_workers();
};

} // namespace data
} // namespace tenzor
