/**
 * @file offload_engine.cpp
 * @brief Implementation of OffloadEngine for ZeRO optimizer Phase 2
 */

#include "tenzor/core/offload_engine.hpp"
#include <algorithm>
#include <stdexcept>
#include <sstream>

namespace tenzor {
namespace core {

// ============================================================================
// Constructor / Destructor
// ============================================================================

OffloadEngine::OffloadEngine(const Config& config)
    : config_(config)
    , default_gpu_device_(Device::cuda(0))
{
    // Validate configuration
    if (config_.pinned_memory_size == 0) {
        throw std::runtime_error("OffloadEngine: pinned_memory_size must be > 0");
    }
    if (config_.num_transfer_streams <= 0) {
        throw std::runtime_error("OffloadEngine: num_transfer_streams must be > 0");
    }
    if (config_.memory_fraction < 0.0f || config_.memory_fraction > 1.0f) {
        throw std::runtime_error("OffloadEngine: memory_fraction must be in [0.0, 1.0]");
    }
    if (config_.prefetch_depth <= 0) {
        throw std::runtime_error("OffloadEngine: prefetch_depth must be > 0");
    }

    try {
        // Initialize TransferEngine. Adopt the caller's pre-built engine if provided so
        // multiple cooperating subsystems (parameter offload via OffloadEngine, activation
        // offload via OffloadContext) can share a single host-pinned buffer pool — see
        // review item #17.
        if (config_.shared_transfer_engine) {
            transfer_engine_ = config_.shared_transfer_engine;
        } else {
            TransferEngine::Config transfer_config;
            transfer_config.num_streams = config_.num_transfer_streams;
            transfer_config.use_pinned_memory = true;
            transfer_config.pinned_pool_size = config_.pinned_memory_size;
            transfer_engine_ = std::make_shared<TransferEngine>(transfer_config);
        }

        // Phase C (C1): The previous code allocated a separate PinnedMemoryAllocator
        // here, but it was never used outside this constructor and get_pinned_memory_stats()
        // -- TransferEngine owns its own pinned-buffer pool (see pinned_buffers_ field at
        // include/tenzor/core/transfer_engine.hpp:325) and that's what every actual DMA
        // routes through. The duplicate 2 GB pool was pure waste of host RAM. Removed.

        // Initialize MemoryManager
        MemoryManager::Config memory_config;
        memory_config.cpu_memory_limit = config_.cpu_memory_limit;
        memory_config.gpu_memory_limit = config_.gpu_memory_limit;
        memory_config.eviction_threshold = config_.memory_fraction;
        memory_config.track_statistics = true;
        memory_manager_ = std::make_unique<MemoryManager>(memory_config);

        // Start prefetch worker thread if enabled
        if (config_.enable_prefetch) {
            stop_prefetch_worker_.store(false);
            prefetch_worker_thread_ = std::thread(&OffloadEngine::prefetch_worker, this);
        }

        // Start monitoring worker thread if enabled
        if (config_.enable_auto_monitoring) {
            stop_monitoring_.store(false);
            monitoring_thread_ = std::thread(&OffloadEngine::monitoring_worker, this);
        }

    } catch (const std::exception& e) {
        throw std::runtime_error(
            std::string("OffloadEngine initialization failed: ") + e.what()
        );
    }
}

OffloadEngine::~OffloadEngine() {
    // Stop monitoring worker
    if (config_.enable_auto_monitoring && monitoring_thread_.joinable()) {
        stop_monitoring_.store(true);
        monitoring_thread_.join();
    }

    // Stop prefetch worker
    if (config_.enable_prefetch && prefetch_worker_thread_.joinable()) {
        stop_prefetch_worker_.store(true);
        prefetch_cv_.notify_all();
        // Phase C (C5): also wake any wait_for_prefetch caller so they can observe
        // the stop flag instead of waiting forever.
        prefetch_drained_cv_.notify_all();
        prefetch_worker_thread_.join();
    }

    // Drop any in-flight prefetch handles. We deliberately do *not* dereference the user's
    // Tensor* targets here: the engine is being destroyed alongside the rest of the
    // session's state, and we have no liveness guarantees on those raw pointers (the user's
    // tensors may already have been freed). The handles themselves clean up their own
    // staging buffers when they destruct. If the caller wants the GPU results assigned
    // back to their tensors they must call wait_for_prefetch() before tearing down the
    // engine.
    {
        std::lock_guard<std::mutex> ifl_lock(in_flight_mutex_);
        in_flight_prefetches_.clear();
    }

    // Synchronize all pending transfers
    if (transfer_engine_) {
        transfer_engine_->synchronize();
    }

    // Cleanup components (unique_ptr handles this automatically)
}

// ============================================================================
// Synchronous Offload API
// ============================================================================

auto OffloadEngine::offload_to_cpu(const Tensor& gpu_tensor) -> Tensor {
    // If tensor is already on CPU, return it as-is (no-op)
    if (gpu_tensor.device().type == Device::Type::CPU) {
        return gpu_tensor;
    }

    // Validate input is a GPU device
    if (!is_gpu_device(gpu_tensor.device())) {
        throw std::runtime_error(
            "OffloadEngine::offload_to_cpu: tensor must be on GPU or CPU, got " +
            gpu_tensor.device().to_string()
        );
    }

    // Use TransferEngine for synchronous transfer
    Tensor cpu_tensor = transfer_engine_->gpu_to_cpu(gpu_tensor);

    // Update statistics
    offload_count_.fetch_add(1, std::memory_order_relaxed);

    return cpu_tensor;
}

auto OffloadEngine::load_to_gpu(const Tensor& cpu_tensor) -> Tensor {
    return load_to_gpu(cpu_tensor, default_gpu_device_);
}

auto OffloadEngine::load_to_gpu(const Tensor& cpu_tensor, Device gpu_device) -> Tensor {
    // Validate input
    if (cpu_tensor.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "OffloadEngine::load_to_gpu: tensor must be on CPU, got " +
            cpu_tensor.device().to_string()
        );
    }

    if (!is_gpu_device(gpu_device)) {
        throw std::runtime_error(
            "OffloadEngine::load_to_gpu: target device must be GPU, got " +
            gpu_device.to_string()
        );
    }

    // Check if we should trigger auto-offload first
    if (is_over_threshold()) {
        check_and_offload();
    }

    // Use TransferEngine for synchronous transfer
    Tensor gpu_tensor = transfer_engine_->cpu_to_gpu(cpu_tensor, gpu_device);

    // Update statistics
    load_count_.fetch_add(1, std::memory_order_relaxed);

    return gpu_tensor;
}

// ============================================================================
// Asynchronous Offload API
// ============================================================================

auto OffloadEngine::offload_to_cpu_async(const Tensor& gpu_tensor) -> TransferHandle {
    // Validate input
    if (!is_gpu_device(gpu_tensor.device())) {
        throw std::runtime_error(
            "OffloadEngine::offload_to_cpu_async: tensor must be on GPU, got " +
            gpu_tensor.device().to_string()
        );
    }

    // Use TransferEngine for async transfer
    TransferHandle handle = transfer_engine_->gpu_to_cpu_async(gpu_tensor);

    // Update statistics
    offload_count_.fetch_add(1, std::memory_order_relaxed);

    return handle;
}

auto OffloadEngine::load_to_gpu_async(const Tensor& cpu_tensor) -> TransferHandle {
    return load_to_gpu_async(cpu_tensor, default_gpu_device_);
}

auto OffloadEngine::load_to_gpu_async(const Tensor& cpu_tensor, Device gpu_device) -> TransferHandle {
    // Validate input
    if (cpu_tensor.device().type != Device::Type::CPU) {
        throw std::runtime_error(
            "OffloadEngine::load_to_gpu_async: tensor must be on CPU, got " +
            cpu_tensor.device().to_string()
        );
    }

    if (!is_gpu_device(gpu_device)) {
        throw std::runtime_error(
            "OffloadEngine::load_to_gpu_async: target device must be GPU, got " +
            gpu_device.to_string()
        );
    }

    // Check if we should trigger auto-offload first
    if (is_over_threshold()) {
        check_and_offload();
    }

    // Use TransferEngine for async transfer
    TransferHandle handle = transfer_engine_->cpu_to_gpu_async(cpu_tensor, gpu_device);

    // Update statistics
    load_count_.fetch_add(1, std::memory_order_relaxed);

    return handle;
}

auto OffloadEngine::prefetch_to_gpu(const std::vector<Tensor*>& tensors) -> void {
    if (!config_.enable_prefetch) {
        return;  // Prefetch disabled
    }

    std::lock_guard<std::mutex> lock(prefetch_mutex_);

    // Check queue depth limit
    if (prefetch_queue_.size() >= static_cast<size_t>(config_.prefetch_depth)) {
        return;  // Queue full, skip prefetch
    }

    // Add tensors to prefetch queue
    for (Tensor* tensor : tensors) {
        if (tensor == nullptr) {
            continue;  // Skip null pointers
        }

        // Only prefetch CPU tensors
        if (tensor->device().type != Device::Type::CPU) {
            continue;  // Already on GPU or other device
        }

        // Add to queue
        PrefetchRequest req;
        req.tensor = tensor;
        req.target_device = default_gpu_device_;
        prefetch_queue_.push(req);

        prefetch_count_.fetch_add(1, std::memory_order_relaxed);
    }

    // Notify worker thread
    prefetch_cv_.notify_one();
}

auto OffloadEngine::prefetch_to_gpu(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        return;
    }

    std::vector<Tensor*> tensors = {tensor};
    prefetch_to_gpu(tensors);
}

// ============================================================================
// Memory Management
// ============================================================================

auto OffloadEngine::get_pinned_memory_stats() -> core::PinnedMemoryStats {
    // The TransferEngine owns the host-pinned buffer pool and reports real
    // occupancy (allocated/free bytes, block counts, fragmentation, peak).
    if (transfer_engine_) {
        return transfer_engine_->get_pinned_memory_stats();
    }
    // No transfer engine (shouldn't happen post-construction): report only the
    // configured capacity, with zero live usage.
    core::PinnedMemoryStats stats{};
    stats.total_size = config_.pinned_memory_size;
    stats.free_size = config_.pinned_memory_size;
    return stats;
}

auto OffloadEngine::register_auto_offload(Tensor* tensor, OffloadPriority priority) -> void {
    if (tensor == nullptr) {
        throw std::runtime_error("OffloadEngine::register_auto_offload: tensor is null");
    }

    std::lock_guard<std::mutex> lock(registry_mutex_);

    // Check if already registered
    auto it = std::find_if(
        auto_offload_registry_.begin(),
        auto_offload_registry_.end(),
        [tensor](const AutoOffloadEntry& entry) { return entry.tensor == tensor; }
    );

    const size_t now_tick = registration_counter_.fetch_add(1, std::memory_order_relaxed);
    const size_t cur_bytes =
        static_cast<size_t>(tensor->numel()) * dtype_size(tensor->dtype());

    if (it != auto_offload_registry_.end()) {
        // Update existing entry. Touch last_access_tick too -- a re-registration
        // is itself an access signal.
        it->priority = priority;
        it->registration_time = now_tick;
        it->last_access_tick = now_tick;
        if (it->size_bytes == 0) it->size_bytes = cur_bytes;
    } else {
        // Add new entry
        AutoOffloadEntry entry;
        entry.tensor = tensor;
        entry.priority = priority;
        entry.registration_time = now_tick;
        entry.last_access_tick = now_tick;
        entry.size_bytes = cur_bytes;
        auto_offload_registry_.push_back(entry);
    }

    // Register with memory manager
    if (memory_manager_) {
        memory_manager_->register_tensor(tensor);
    }

    // Sort by priority
    sort_auto_offload_registry();
}

auto OffloadEngine::unregister_auto_offload(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(registry_mutex_);

    // Remove from registry
    auto it = std::find_if(
        auto_offload_registry_.begin(),
        auto_offload_registry_.end(),
        [tensor](const AutoOffloadEntry& entry) { return entry.tensor == tensor; }
    );

    if (it != auto_offload_registry_.end()) {
        auto_offload_registry_.erase(it);
    }

    // Unregister from memory manager
    if (memory_manager_) {
        memory_manager_->unregister_tensor(tensor);
    }
}

auto OffloadEngine::mark_accessed(Tensor* tensor) -> void {
    if (tensor == nullptr) return;
    std::lock_guard<std::mutex> lock(registry_mutex_);
    auto it = std::find_if(
        auto_offload_registry_.begin(),
        auto_offload_registry_.end(),
        [tensor](const AutoOffloadEntry& e) { return e.tensor == tensor; }
    );
    if (it != auto_offload_registry_.end()) {
        it->last_access_tick =
            registration_counter_.fetch_add(1, std::memory_order_relaxed);
    }
}

auto OffloadEngine::check_and_offload() -> size_t {
    if (!is_over_threshold()) {
        return 0;  // No action needed
    }

    // Snapshot the registry under the lock, then drop the lock before any disk/PCIe
    // transfer. The legacy code held registry_mutex_ across each gpu_to_cpu_async +
    // handle.wait(), which blocked register_auto_offload / unregister_auto_offload /
    // get_registered_tensor_count for the duration of every transfer — and on a 100ms
    // monitoring cadence that produced visible stalls. After the snapshot, mutations to
    // the registry (priority changes, registrations) take effect on the *next* tick.
    std::vector<AutoOffloadEntry> snapshot;
    {
        std::lock_guard<std::mutex> lock(registry_mutex_);
        sort_auto_offload_registry();
        snapshot = auto_offload_registry_;  // value-copy, releases the lock immediately
    }

    size_t offloaded = 0;

    // Phase C (C2): hysteresis. Once we cross config_.memory_fraction (typically 0.9),
    // keep evicting until we're under 0.75 * threshold instead of stopping the moment
    // we dip back under threshold. Without this, the next monitor tick re-fires
    // immediately because steady-state usage hovers exactly at the threshold ->
    // evict-load-evict thrash.
    const float low_water = config_.memory_fraction * 0.75f;

    // Offload tensors until we're under low_water. get_gpu_memory_pressure() reads from
    // memory_manager_ which has its own internal locking, so it's safe to call here.
    for (const auto& entry : snapshot) {
        if (get_gpu_memory_pressure() <= low_water) {
            break;  // Pressure dropped below low-water mark; stop hysteresis loop.
        }

        Tensor* tensor = entry.tensor;
        if (tensor == nullptr || !is_gpu_device(tensor->device())) {
            continue;  // Already on CPU or invalidated
        }

        try {
            // Synchronous GPU→CPU. The legacy code did gpu_to_cpu_async() followed
            // immediately by handle.wait() and handle.get_tensor() — that's the same as
            // sync with two extra allocations. Just call the sync version.
            Tensor cpu_tensor = transfer_engine_->gpu_to_cpu(*tensor);
            {
                // Serialize the user-tensor reassignment w.r.t. prefetch commits
                // so the intrusive_ptr refcount updates can't race.
                std::lock_guard<std::mutex> assign_lock(tensor_assign_mutex_);
                *tensor = cpu_tensor;
            }

            if (memory_manager_) {
                memory_manager_->update_tensor_location(tensor, Device::cpu());
            }

            offloaded++;
            auto_offload_count_.fetch_add(1, std::memory_order_relaxed);
        } catch (const std::exception& e) {
            // Silently skip on error
            continue;
        }
    }

    return offloaded;
}

auto OffloadEngine::get_gpu_memory_pressure() const -> float {
    if (!memory_manager_) {
        return 0.0f;
    }

    // Use default GPU device type instead of hardcoding CUDA
    return memory_manager_->get_memory_pressure(default_gpu_device_.type);
}

auto OffloadEngine::get_gpu_memory_pressure(Device::Type device_type) const -> float {
    if (!memory_manager_) {
        return 0.0f;
    }

    return memory_manager_->get_memory_pressure(device_type);
}

auto OffloadEngine::is_over_threshold() const -> bool {
    return get_gpu_memory_pressure() > config_.memory_fraction;
}

// ============================================================================
// Synchronization
// ============================================================================

auto OffloadEngine::synchronize() -> void {
    // Wait for prefetch queue to drain
    wait_for_prefetch();

    // Synchronize transfer engine
    if (transfer_engine_) {
        transfer_engine_->synchronize();
    }
}

auto OffloadEngine::wait_for_prefetch() -> void {
    if (!config_.enable_prefetch) {
        return;
    }

    // Phase 1 (Phase C C5): wait for the worker to drain the queue. Previously this
    // polled with a 10 ms sleep loop, adding up to 10 ms of latency per drain even
    // when the worker had already finished. Now we CV-wait on prefetch_drained_cv_,
    // which the worker notifies after every drain. The wait still uses
    // prefetch_mutex_ so the predicate read is atomic.
    {
        std::unique_lock<std::mutex> lock(prefetch_mutex_);
        prefetch_drained_cv_.wait(lock, [this] {
            return prefetch_queue_.empty()
                || stop_prefetch_worker_.load(std::memory_order_relaxed);
        });
    }

    // Phase 2: commit every in-flight handle. Snapshot the list under the in-flight lock
    // and clear the engine's state, then await + assign outside the lock so a slow PCIe
    // transfer doesn't block other engine operations (e.g. concurrent register_auto_offload
    // from a different thread).
    std::vector<InFlightPrefetch> to_commit;
    {
        std::lock_guard<std::mutex> ifl_lock(in_flight_mutex_);
        to_commit.swap(in_flight_prefetches_);
    }
    for (auto& f : to_commit) {
        try {
            Tensor result = f.handle.get_tensor();  // implicit wait
            if (f.target != nullptr) {
                // Serialize w.r.t. check_and_offload's `*tensor = cpu_tensor`
                // so concurrent refcount updates on the same Tensor are atomic.
                std::lock_guard<std::mutex> assign_lock(tensor_assign_mutex_);
                *f.target = result;
            }
        } catch (const std::exception&) {
            // Best-effort; bad transfers shouldn't poison the rest of the batch.
            continue;
        }
    }

    // Belt-and-braces: drain anything else the transfer engine might still hold
    // (host-side staging copies queued internally, etc.).
    if (transfer_engine_) {
        transfer_engine_->synchronize();
    }
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

auto OffloadEngine::get_registered_tensor_count() const -> size_t {
    std::lock_guard<std::mutex> lock(registry_mutex_);
    return auto_offload_registry_.size();
}

auto OffloadEngine::reset_statistics() -> void {
    offload_count_.store(0, std::memory_order_relaxed);
    load_count_.store(0, std::memory_order_relaxed);
    prefetch_count_.store(0, std::memory_order_relaxed);
    auto_offload_count_.store(0, std::memory_order_relaxed);

    if (transfer_engine_) {
        transfer_engine_->reset_statistics();
    }

    if (memory_manager_) {
        memory_manager_->reset_stats();
    }
}

// ============================================================================
// Private Helper Methods
// ============================================================================

auto OffloadEngine::prefetch_worker() -> void {
    while (!stop_prefetch_worker_.load(std::memory_order_relaxed)) {
        // Phase C (C4): drain ALL queued requests in one lock acquisition, then issue
        // them in a tight loop without the lock. The legacy code popped one request per
        // condvar wake even though TransferEngine has 4 streams -- effectively
        // serializing the issue side and defeating the multi-stream parallelism. Now a
        // batch of N queued requests turns into N nearly-simultaneous async DMAs that
        // round-robin across TransferEngine's stream pool.
        std::vector<PrefetchRequest> batch;
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);
            prefetch_cv_.wait(lock, [this] {
                return !prefetch_queue_.empty() ||
                       stop_prefetch_worker_.load(std::memory_order_relaxed);
            });

            if (stop_prefetch_worker_.load(std::memory_order_relaxed)) {
                break;
            }

            batch.reserve(prefetch_queue_.size());
            while (!prefetch_queue_.empty()) {
                batch.push_back(prefetch_queue_.front());
                prefetch_queue_.pop();
            }
        }

        // Process the batch outside the lock. Each cpu_to_gpu_async returns quickly
        // (just enqueues on a stream and returns the handle), so single-threaded issue
        // is sufficient -- the wins are in (a) not blocking other producers behind the
        // condvar wait per item, and (b) the multi-stream backend handling concurrent
        // DMAs.
        std::vector<std::pair<Tensor*, core::TransferHandle>> issued;
        issued.reserve(batch.size());
        for (auto& request : batch) {
            if (request.tensor == nullptr) continue;
            try {
                if (request.tensor->device().type == Device::Type::CPU) {
                    auto handle = transfer_engine_->cpu_to_gpu_async(
                        *request.tensor,
                        request.target_device
                    );
                    issued.emplace_back(request.tensor, std::move(handle));
                }
            } catch (const std::exception& e) {
                // Silently skip on error -- prefetch is a hint, not a guarantee.
                continue;
            }
        }

        if (!issued.empty()) {
            std::lock_guard<std::mutex> ifl_lock(in_flight_mutex_);
            for (auto& p : issued) {
                in_flight_prefetches_.push_back({p.first, std::move(p.second)});
            }
        }

        // Phase C (C5): notify wait_for_prefetch that the queue is now empty (we just
        // drained the entire batch above). Cheap notify_all even if no waiter is
        // present; CV-side spurious wakes are filtered by the predicate.
        prefetch_drained_cv_.notify_all();
    }
}

auto OffloadEngine::monitoring_worker() -> void {
    while (!stop_monitoring_.load(std::memory_order_relaxed)) {
        // Sleep for the configured interval
        std::this_thread::sleep_for(
            std::chrono::milliseconds(config_.monitoring_interval_ms)
        );

        // Check if we should stop
        if (stop_monitoring_.load(std::memory_order_relaxed)) {
            break;
        }

        // Check memory pressure and trigger auto-offload if needed
        if (is_over_threshold()) {
            check_and_offload();
        }
    }
}

auto OffloadEngine::get_gpu_device(const Tensor& tensor) const -> Device {
    const Device& device = tensor.device();
    if (is_gpu_device(device)) {
        return device;
    }
    return default_gpu_device_;
}

auto OffloadEngine::is_gpu_device(const Device& device) const -> bool {
    return device.type != Device::Type::CPU;
}

auto OffloadEngine::sort_auto_offload_registry() -> void {
    // Phase C (C2): eviction order = (priority asc, size_bytes desc, last_access_tick asc).
    //   - Lowest-priority evicts first (CRITICAL never evicts unless nothing else exists).
    //   - Within same priority, largest-size first (one 1 GB tensor instead of 1000 1 MB).
    //   - Within same priority + size band, oldest-access first (LRU; tensors used every
    //     step stay resident).
    // Replaces the legacy (priority, registration_time) order which evicted tensors used
    // every step before never-used ones of equal priority -- pure thrash on steady state.
    std::sort(
        auto_offload_registry_.begin(),
        auto_offload_registry_.end(),
        [](const AutoOffloadEntry& a, const AutoOffloadEntry& b) {
            if (a.priority != b.priority) {
                return static_cast<int>(a.priority) < static_cast<int>(b.priority);
            }
            if (a.size_bytes != b.size_bytes) {
                return a.size_bytes > b.size_bytes;
            }
            return a.last_access_tick < b.last_access_tick;
        }
    );
}

} // namespace core
} // namespace tenzor
