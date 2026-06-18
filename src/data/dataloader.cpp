#include "tenzor/data/dataloader.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include <chrono>
#include <stdexcept>

namespace tenzor {
namespace data {

// DataLoader Constructor with config
DataLoader::DataLoader(std::shared_ptr<Dataset> dataset, const DataLoaderConfig& config)
    : dataset_(std::move(dataset)),
      config_(config),
      rng_(std::random_device{}()),
      current_index_(0),
      stop_workers_(false),
      epoch_done_(false),
      next_batch_idx_(0),
      active_workers_(0) {

    if (!dataset_) {
        throw std::invalid_argument("Dataset cannot be null");
    }

    if (config_.batch_size == 0) {
        throw std::invalid_argument("Batch size must be greater than 0");
    }

    // Calculate number of batches
    size_t dataset_size = dataset_->size();
    if (config_.drop_last) {
        num_batches_ = dataset_size / config_.batch_size;
    } else {
        num_batches_ = (dataset_size + config_.batch_size - 1) / config_.batch_size;
    }

    // Initialize indices
    init_indices();

    // Start worker threads if multi-threaded
    if (config_.num_workers > 0) {
        start_workers();
    }
}

// DataLoader Constructor with individual parameters
DataLoader::DataLoader(std::shared_ptr<Dataset> dataset,
                       size_t batch_size,
                       bool shuffle,
                       size_t num_workers,
                       bool pin_memory,
                       bool drop_last)
    : dataset_(std::move(dataset)),
      rng_(std::random_device{}()),
      current_index_(0),
      stop_workers_(false),
      epoch_done_(false),
      next_batch_idx_(0),
      active_workers_(0) {

    if (!dataset_) {
        throw std::invalid_argument("Dataset cannot be null");
    }

    if (batch_size == 0) {
        throw std::invalid_argument("Batch size must be greater than 0");
    }

    // Set config
    config_.batch_size = batch_size;
    config_.shuffle = shuffle;
    config_.num_workers = num_workers;
    config_.pin_memory = pin_memory;
    config_.drop_last = drop_last;
    config_.prefetch_factor = 2;

    // Calculate number of batches
    size_t dataset_size = dataset_->size();
    if (config_.drop_last) {
        num_batches_ = dataset_size / config_.batch_size;
    } else {
        num_batches_ = (dataset_size + config_.batch_size - 1) / config_.batch_size;
    }

    // Initialize indices
    init_indices();

    // Start worker threads if multi-threaded
    if (config_.num_workers > 0) {
        start_workers();
    }
}

// Destructor
DataLoader::~DataLoader() {
    stop_workers();
}

// DataLoader is non-movable (worker threads are bound to `this`); the move
// operations are = delete in the header. See the comment there.

// Initialize indices
void DataLoader::init_indices() {
    size_t dataset_size = dataset_->size();
    indices_.resize(dataset_size);
    for (size_t i = 0; i < dataset_size; ++i) {
        indices_[i] = i;
    }

    if (config_.shuffle) {
        shuffle_indices();
    }
}

// Shuffle indices
void DataLoader::shuffle_indices() {
    std::shuffle(indices_.begin(), indices_.end(), rng_);
}

// Collate samples into batch
auto DataLoader::collate_samples(const std::vector<std::pair<Tensor, Tensor>>& samples) -> Batch {
    if (samples.empty()) {
        return Batch{};
    }

    // Get shapes from first sample
    const auto& first_input = samples[0].first;
    const auto& first_target = samples[0].second;

    // Stack samples into batch: unsqueeze each sample to add a leading batch
    // dim, then concat along that dim. cat() validates that all sections have
    // matching shapes, so no separate pre-validation pass is needed here; we
    // keep a cheap rank check only to surface a clearer error message. The
    // stack+cat path is already O(N) and avoids the mutable-Tensor-view
    // machinery an in-place slice-assignment path would require.
    std::vector<Tensor> input_list, target_list;
    input_list.reserve(samples.size());
    target_list.reserve(samples.size());
    const size_t input_rank = first_input.shape().size();
    const size_t target_rank = first_target.shape().size();
    for (const auto& [input, target] : samples) {
        if (input.shape().size() != input_rank) {
            throw std::runtime_error("All input samples must have same rank");
        }
        if (target.shape().size() != target_rank) {
            throw std::runtime_error("All target samples must have same rank");
        }
        input_list.push_back(unsqueeze(input, 0));
        target_list.push_back(unsqueeze(target, 0));
    }

    // Concatenate along batch dimension (cat() validates per-section shapes).
    Tensor batch_inputs = cat(input_list, 0);
    Tensor batch_targets = cat(target_list, 0);

    // Audit J8: real memory pinning via Storage::pin().
    //
    // The old code documented the steps ("Full pinning implementation would
    // require cudaHostAlloc... setting a flag in tensor storage") and then
    // didn't do them. The Storage base class has a virtual `pin()` method
    // (`include/tenzor/core/storage.hpp:88`) that the CPU storage
    // implements via cudaHostRegister + sets the pinned flag; we just call
    // it on each batch tensor here.
    //
    // The pin() call is a no-op on non-CPU storage and on builds without
    // CUDA, so this path is safe to invoke unconditionally when
    // `config_.pin_memory == true` (the storage layer guards CUDA-only
    // operations internally).
    if (config_.pin_memory) {
        // Ensure host placement + contiguity before pinning (pin() registers
        // the buffer with the CUDA driver — it must be on CPU and own a
        // page-aligned host allocation).
        if (batch_inputs.device().type != Device::Type::CPU) {
            batch_inputs = batch_inputs.to(Device::cpu());
        }
        if (batch_targets.device().type != Device::Type::CPU) {
            batch_targets = batch_targets.to(Device::cpu());
        }
        batch_inputs  = batch_inputs.contiguous();
        batch_targets = batch_targets.contiguous();

        if (batch_inputs.storage())  batch_inputs.storage()->pin();
        if (batch_targets.storage()) batch_targets.storage()->pin();
    }

    return Batch{batch_inputs, batch_targets};
}

// Worker thread function
void DataLoader::worker_thread([[maybe_unused]] size_t worker_id) {
    // Set per-thread worker info so IterableDataset subclasses can shard
    WorkerInfo info;
    info.worker_id = static_cast<int>(worker_id);
    info.num_workers = static_cast<int>(config_.num_workers);
    info.seed = static_cast<int64_t>(rng_()) + static_cast<int64_t>(worker_id);
    // rank and world_size remain at defaults (0 and 1) for single-process;
    // distributed launchers should set them via set_worker_info() before
    // constructing the DataLoader, or the Python layer handles it.
    set_worker_info(info);

    // Each persistent worker tracks the last epoch generation it processed.
    // reset() bumps epoch_generation_ once per epoch; a worker only re-runs
    // when it observes a generation it has not yet handled. This avoids the
    // one-shot-bool lost-wakeup where a single worker clears the flag before
    // its peers re-check the predicate.
    uint64_t last_seen_generation;
    {
        std::unique_lock<std::mutex> lock(queue_mutex_);
        last_seen_generation = epoch_generation_;
    }

    try {
        do {
            while (!stop_workers_) {
                // Get next batch index to process
                size_t batch_idx = next_batch_idx_.fetch_add(1);

                if (batch_idx >= num_batches_) {
                    // No more batches to process - decrement active workers
                    size_t remaining = active_workers_.fetch_sub(1) - 1;
                    if (remaining == 0) {
                        // Last worker to finish - signal epoch done
                        std::unique_lock<std::mutex> lock(queue_mutex_);
                        epoch_done_ = true;
                        queue_cv_.notify_all();
                    }
                    break;
                }

                // Calculate sample indices for this batch
                size_t start_idx = batch_idx * config_.batch_size;
                size_t end_idx = std::min(start_idx + config_.batch_size, dataset_->size());

                // Skip if this is an incomplete batch and drop_last is true
                if (config_.drop_last && (end_idx - start_idx) < config_.batch_size) {
                    continue;
                }

                // Load samples
                std::vector<std::pair<Tensor, Tensor>> samples;
                samples.reserve(end_idx - start_idx);

                for (size_t i = start_idx; i < end_idx; ++i) {
                    size_t sample_idx = indices_[i];
                    samples.push_back(dataset_->get(sample_idx));
                }

                // Collate into batch
                Batch batch = config_.collate_fn ? config_.collate_fn(samples) : collate_samples(samples);

                // Add to queue (with backpressure)
                {
                    std::unique_lock<std::mutex> lock(queue_mutex_);

                    // Wait if queue is full (implement prefetch limit)
                    size_t max_queue_size = config_.num_workers * config_.prefetch_factor;
                    worker_cv_.wait(lock, [this, max_queue_size] {
                        return stop_workers_ || batch_queue_.size() < max_queue_size;
                    });

                    if (stop_workers_) {
                        break;
                    }

                    batch_queue_.push({batch_idx, std::move(batch)});
                    queue_cv_.notify_one();
                }
            }

            // Persistent workers: wait for next epoch instead of exiting
            if (config_.persistent_workers && !stop_workers_) {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                epoch_start_cv_.wait(lock, [this, last_seen_generation] {
                    return stop_workers_ ||
                           epoch_generation_ != last_seen_generation;
                });
                if (stop_workers_) break;
                // Record the new generation so this worker waits for the NEXT
                // epoch on its following pass; every waiter is released exactly
                // once per epoch.
                last_seen_generation = epoch_generation_;
                continue;
            }
        } while (config_.persistent_workers && !stop_workers_);

        clear_worker_info();
    } catch (...) {
        clear_worker_info();
        // Store first exception and signal epoch done so consumer unblocks
        std::unique_lock<std::mutex> lock(queue_mutex_);
        if (!worker_exception_) {
            worker_exception_ = std::current_exception();
        }
        epoch_done_ = true;
        queue_cv_.notify_all();
    }
}

// Start worker threads
void DataLoader::start_workers() {
    stop_workers_ = false;
    epoch_done_ = false;
    next_batch_idx_ = 0;
    active_workers_ = config_.num_workers;  // Initialize active worker count

    workers_.reserve(config_.num_workers);
    for (size_t i = 0; i < config_.num_workers; ++i) {
        workers_.emplace_back(&DataLoader::worker_thread, this, i);
    }
}

// Stop worker threads
void DataLoader::stop_workers() {
    if (workers_.empty()) {
        return;
    }

    // Signal workers to stop
    stop_workers_ = true;

    // Wake up all workers
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        queue_cv_.notify_all();
        worker_cv_.notify_all();
    }

    // Join all worker threads
    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }

    workers_.clear();
}

// Get next batch (single-threaded)
auto DataLoader::get_next_batch_single_threaded() -> Batch {
    if (current_index_ >= num_batches_) {
        return Batch{};  // Empty batch signals end
    }

    // Calculate sample indices for this batch
    size_t start_idx = current_index_ * config_.batch_size;
    size_t end_idx = std::min(start_idx + config_.batch_size, dataset_->size());

    // Skip if this is an incomplete batch and drop_last is true
    if (config_.drop_last && (end_idx - start_idx) < config_.batch_size) {
        current_index_ = num_batches_;  // Signal end
        return Batch{};
    }

    // Load samples
    std::vector<std::pair<Tensor, Tensor>> samples;
    samples.reserve(end_idx - start_idx);

    for (size_t i = start_idx; i < end_idx; ++i) {
        size_t sample_idx = indices_[i];
        samples.push_back(dataset_->get(sample_idx));
    }

    current_index_++;

    return config_.collate_fn ? config_.collate_fn(samples) : collate_samples(samples);
}

// Get next batch (multi-threaded)
auto DataLoader::get_next_batch_multi_threaded() -> Batch {
    std::unique_lock<std::mutex> lock(queue_mutex_);

    // Emit batches in ascending batch_idx order even though workers complete
    // out of order. Each pass drains the bounded completion queue into the
    // reorder buffer (freeing queue space so a worker holding the next-needed
    // batch can never deadlock against backpressure), then emits if the next
    // expected index is available.
    for (;;) {
        // Drain everything currently completed into the reorder buffer.
        bool drained = false;
        while (!batch_queue_.empty()) {
            auto& front = batch_queue_.front();
            reorder_buffer_.emplace(front.first, std::move(front.second));
            batch_queue_.pop();
            drained = true;
        }
        if (drained) {
            // Freed queue capacity — let backpressured workers proceed.
            worker_cv_.notify_all();
        }

        // Rethrow worker exception on the consumer thread.
        if (worker_exception_) {
            auto ex = worker_exception_;
            worker_exception_ = nullptr;
            std::rethrow_exception(ex);
        }

        // Emit the next in-order batch if we have it.
        auto it = reorder_buffer_.find(next_output_idx_);
        if (it != reorder_buffer_.end()) {
            Batch batch = std::move(it->second);
            reorder_buffer_.erase(it);
            ++next_output_idx_;
            worker_cv_.notify_all();
            return batch;
        }

        // Nothing more will ever arrive: workers done and nothing buffered.
        if (epoch_done_ && batch_queue_.empty() && reorder_buffer_.empty()) {
            return Batch{};  // Empty batch signals end of epoch
        }

        // Wait for a worker to push the next batch (or finish the epoch).
        queue_cv_.wait(lock, [this] {
            return !batch_queue_.empty() || epoch_done_;
        });
    }
}

// Begin iterator
auto DataLoader::begin() -> Iterator {
    reset();
    return Iterator(this, 0);
}

// End iterator
auto DataLoader::end() -> Iterator {
    return Iterator(this, num_batches_);
}

// Get number of batches
auto DataLoader::size() const -> size_t {
    return num_batches_;
}

// Reset for new epoch
void DataLoader::reset() {
    current_index_ = 0;

    if (config_.shuffle) {
        shuffle_indices();
    }

    if (config_.num_workers > 0) {
        if (config_.persistent_workers && !workers_.empty()) {
            // Persistent workers: signal new epoch without destroying threads
            // Clear any stored worker exception from previous epoch
            worker_exception_ = nullptr;

            // Clear queue + reorder state
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                std::queue<std::pair<size_t, Batch>> empty_queue;
                std::swap(batch_queue_, empty_queue);
                reorder_buffer_.clear();
                next_output_idx_ = 0;
            }

            // Reset state for new epoch
            epoch_done_ = false;
            next_batch_idx_ = 0;
            active_workers_ = config_.num_workers;

            // Signal workers to start new epoch: bump the generation counter
            // once under the lock so all N waiters are released exactly once.
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                ++epoch_generation_;
            }
            epoch_start_cv_.notify_all();
        } else {
            // Non-persistent: stop and restart workers
            stop_workers();

            // Clear any stored worker exception from previous epoch
            worker_exception_ = nullptr;

            // Clear queue + reorder state
            std::queue<std::pair<size_t, Batch>> empty_queue;
            std::swap(batch_queue_, empty_queue);
            reorder_buffer_.clear();
            next_output_idx_ = 0;

            start_workers();
        }
    }
}

// Iterator Constructor
DataLoader::Iterator::Iterator(DataLoader* loader, size_t index)
    : loader_(loader), index_(index), valid_(false) {
    if (loader_ && index_ < loader_->num_batches_) {
        fetch_next();
    }
}

// Fetch next batch
void DataLoader::Iterator::fetch_next() {
    if (!loader_ || index_ >= loader_->num_batches_) {
        valid_ = false;
        return;
    }

    if (loader_->config_.num_workers > 0) {
        current_batch_ = loader_->get_next_batch_multi_threaded();
    } else {
        current_batch_ = loader_->get_next_batch_single_threaded();
    }

    valid_ = !current_batch_.inputs.shape().empty();

    if (!valid_) {
        index_ = loader_->num_batches_;
    }
}

// Iterator dereference
auto DataLoader::Iterator::operator*() -> Batch& {
    return current_batch_;
}

auto DataLoader::Iterator::operator->() -> Batch* {
    return &current_batch_;
}

// Iterator increment
auto DataLoader::Iterator::operator++() -> Iterator& {
    if (index_ < loader_->num_batches_) {
        index_++;
        fetch_next();
    }
    return *this;
}

auto DataLoader::Iterator::operator++(int) -> Iterator {
    Iterator tmp = *this;
    ++(*this);
    return tmp;
}

// Iterator comparison
auto DataLoader::Iterator::operator==(const Iterator& other) const -> bool {
    return loader_ == other.loader_ && index_ == other.index_;
}

auto DataLoader::Iterator::operator!=(const Iterator& other) const -> bool {
    return !(*this == other);
}

} // namespace data
} // namespace tenzor
