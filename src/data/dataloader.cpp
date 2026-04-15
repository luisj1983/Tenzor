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
      active_workers_(0),
      new_epoch_ready_(false) {

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
      active_workers_(0),
      new_epoch_ready_(false) {

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

// Move constructor
DataLoader::DataLoader(DataLoader&& other) noexcept
    : dataset_(std::move(other.dataset_)),
      config_(other.config_),
      indices_(std::move(other.indices_)),
      rng_(std::move(other.rng_)),
      current_index_(other.current_index_),
      workers_(std::move(other.workers_)),
      batch_queue_(std::move(other.batch_queue_)),
      stop_workers_(other.stop_workers_.load()),
      epoch_done_(other.epoch_done_.load()),
      next_batch_idx_(other.next_batch_idx_.load()),
      active_workers_(other.active_workers_.load()),
      num_batches_(other.num_batches_) {
    other.dataset_ = nullptr;
}

// Move assignment
DataLoader& DataLoader::operator=(DataLoader&& other) noexcept {
    if (this != &other) {
        stop_workers();

        dataset_ = std::move(other.dataset_);
        config_ = other.config_;
        indices_ = std::move(other.indices_);
        rng_ = std::move(other.rng_);
        current_index_ = other.current_index_;
        workers_ = std::move(other.workers_);
        batch_queue_ = std::move(other.batch_queue_);
        stop_workers_ = other.stop_workers_.load();
        epoch_done_ = other.epoch_done_.load();
        next_batch_idx_ = other.next_batch_idx_.load();
        active_workers_ = other.active_workers_.load();
        num_batches_ = other.num_batches_;

        other.dataset_ = nullptr;
    }
    return *this;
}

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

    // Create batch shape: [batch_size, ...]
    auto input_shape_span = first_input.shape();
    auto target_shape_span = first_target.shape();

    // Convert spans to vectors so we can insert batch dimension
    std::vector<int64_t> input_shape(input_shape_span.begin(), input_shape_span.end());
    std::vector<int64_t> target_shape(target_shape_span.begin(), target_shape_span.end());

    input_shape.insert(input_shape.begin(), samples.size());
    target_shape.insert(target_shape.begin(), samples.size());

    // Create batch tensors
    Tensor batch_inputs = zeros(input_shape, first_input.dtype());
    Tensor batch_targets = zeros(target_shape, first_target.dtype());

    // Stack samples into batch
    for (size_t i = 0; i < samples.size(); ++i) {
        const auto& [input, target] = samples[i];

        // Validate shapes match
        auto input_shape = input.shape();
        auto first_input_shape = first_input.shape();
        bool shapes_match = (input_shape.size() == first_input_shape.size());
        if (shapes_match) {
            for (size_t j = 0; j < input_shape.size(); ++j) {
                if (input_shape[j] != first_input_shape[j]) {
                    shapes_match = false;
                    break;
                }
            }
        }
        if (!shapes_match) {
            throw std::runtime_error("All input samples must have same shape");
        }

        auto target_shape = target.shape();
        auto first_target_shape = first_target.shape();
        shapes_match = (target_shape.size() == first_target_shape.size());
        if (shapes_match) {
            for (size_t j = 0; j < target_shape.size(); ++j) {
                if (target_shape[j] != first_target_shape[j]) {
                    shapes_match = false;
                    break;
                }
            }
        }
        if (!shapes_match) {
            throw std::runtime_error("All target samples must have same shape");
        }

        // Copy data into batch
        // Use slice assignment to place sample at index i
        // batch_inputs[i] = input
        // batch_targets[i] = target

        // Note: This requires implementing slice assignment in Tensor class
        // For now, we'll create a simplified version using the stack operation
    }

    // Alternative: Use stack operation if available
    std::vector<Tensor> input_list, target_list;
    for (const auto& [input, target] : samples) {
        input_list.push_back(unsqueeze(input, 0));
        target_list.push_back(unsqueeze(target, 0));
    }

    // Concatenate along batch dimension
    batch_inputs = cat(input_list, 0);
    batch_targets = cat(target_list, 0);

    // Pin memory if requested and CUDA is available
    if (config_.pin_memory) {
        // Implement memory pinning for faster CPU-to-CUDA transfers
        // Pinned memory (page-locked memory) allows DMA transfers without paging
        // This is beneficial when transferring data to GPU

        // Check if CUDA is available by attempting device query
        try {
            Device cuda_device(Device::Type::CUDA, 0);

            // Memory pinning is typically done by:
            // 1. Allocating pinned host memory
            // 2. Copying tensor data to pinned memory
            // 3. Marking tensor storage as pinned

            // For now, we'll mark the intent by ensuring tensors are contiguous
            // and on CPU (actual pinning requires CUDA API integration)
            if (batch_inputs.device().type != Device::Type::CPU) {
                batch_inputs = batch_inputs.to(Device::cpu());
            }
            if (batch_targets.device().type != Device::Type::CPU) {
                batch_targets = batch_targets.to(Device::cpu());
            }

            // Ensure contiguous memory layout for efficient transfer
            batch_inputs = batch_inputs.contiguous();
            batch_targets = batch_targets.contiguous();

            // Note: Full pinning implementation would require:
            // - cudaHostAlloc/cudaMallocHost for pinned allocation
            // - Registering memory pages with CUDA driver
            // - Setting a flag in tensor storage to indicate pinned status
            // This is typically done at the storage/allocator level
        } catch (const std::exception&) {
            // CUDA not available, skip pinning
        }
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

                    batch_queue_.push(std::move(batch));
                    queue_cv_.notify_one();
                }
            }

            // Persistent workers: wait for next epoch instead of exiting
            if (config_.persistent_workers && !stop_workers_) {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                epoch_start_cv_.wait(lock, [this] {
                    return stop_workers_ || new_epoch_ready_.load();
                });
                if (stop_workers_) break;
                // Reset the flag only once (first worker to wake clears it)
                new_epoch_ready_ = false;
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

    // Wait for batch or epoch done
    queue_cv_.wait(lock, [this] {
        return !batch_queue_.empty() || epoch_done_;
    });

    // Rethrow worker exception on the consumer thread
    if (worker_exception_) {
        auto ex = worker_exception_;
        worker_exception_ = nullptr;
        std::rethrow_exception(ex);
    }

    if (batch_queue_.empty()) {
        return Batch{};  // Empty batch signals end
    }

    // Get batch from queue
    Batch batch = std::move(batch_queue_.front());
    batch_queue_.pop();

    // Notify workers that queue has space
    worker_cv_.notify_one();

    return batch;
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

            // Clear queue
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                std::queue<Batch> empty_queue;
                std::swap(batch_queue_, empty_queue);
            }

            // Reset state for new epoch
            epoch_done_ = false;
            next_batch_idx_ = 0;
            active_workers_ = config_.num_workers;

            // Signal workers to start new epoch
            {
                std::lock_guard<std::mutex> lock(queue_mutex_);
                new_epoch_ready_ = true;
            }
            epoch_start_cv_.notify_all();
        } else {
            // Non-persistent: stop and restart workers
            stop_workers();

            // Clear any stored worker exception from previous epoch
            worker_exception_ = nullptr;

            // Clear queue
            std::queue<Batch> empty_queue;
            std::swap(batch_queue_, empty_queue);

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
