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
        // Initialize TransferEngine
        TransferEngine::Config transfer_config;
        transfer_config.num_streams = config_.num_transfer_streams;
        transfer_config.use_pinned_memory = true;
        transfer_config.pinned_pool_size = config_.pinned_memory_size;
        transfer_engine_ = std::make_unique<TransferEngine>(transfer_config);

        // Initialize PinnedMemoryAllocator
        PinnedMemoryAllocator::Config pinned_config;
        pinned_config.pool_size = config_.pinned_memory_size;
        pinned_config.allow_growth = false;
        pinned_config.enable_defragmentation = true;
        pinned_alloc_ = std::make_unique<PinnedMemoryAllocator>(pinned_config);

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
        prefetch_worker_thread_.join();
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
    if (!pinned_alloc_) {
        return core::PinnedMemoryStats{};
    }

    // Return the stats directly from pinned allocator
    return pinned_alloc_->get_stats();
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

    if (it != auto_offload_registry_.end()) {
        // Update existing entry
        it->priority = priority;
        it->registration_time = registration_counter_.fetch_add(1, std::memory_order_relaxed);
    } else {
        // Add new entry
        AutoOffloadEntry entry;
        entry.tensor = tensor;
        entry.priority = priority;
        entry.registration_time = registration_counter_.fetch_add(1, std::memory_order_relaxed);
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

auto OffloadEngine::check_and_offload() -> size_t {
    if (!is_over_threshold()) {
        return 0;  // No action needed
    }

    std::lock_guard<std::mutex> lock(registry_mutex_);

    size_t offloaded = 0;

    // Sort by priority (LOW priority first)
    sort_auto_offload_registry();

    // Offload tensors until we're under threshold
    for (const auto& entry : auto_offload_registry_) {
        if (!is_over_threshold()) {
            break;  // Pressure relieved
        }

        Tensor* tensor = entry.tensor;

        // Check if tensor is on GPU
        if (!is_gpu_device(tensor->device())) {
            continue;  // Already on CPU
        }

        // Offload to CPU
        try {
            auto handle = transfer_engine_->gpu_to_cpu_async(*tensor);

            // Wait for the transfer to complete
            handle.wait();

            // Get the resulting CPU tensor and update the original tensor
            Tensor cpu_tensor = handle.get_tensor();
            *tensor = cpu_tensor;

            // Update memory manager to reflect the new location
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

    // Wait until prefetch queue is empty
    while (true) {
        {
            std::lock_guard<std::mutex> lock(prefetch_mutex_);
            if (prefetch_queue_.empty()) {
                break;
            }
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Synchronize transfer engine to ensure all prefetches complete
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
        PrefetchRequest request;
        bool has_request = false;

        // Get next request from queue
        {
            std::unique_lock<std::mutex> lock(prefetch_mutex_);

            // Wait for requests or stop signal
            prefetch_cv_.wait(lock, [this] {
                return !prefetch_queue_.empty() ||
                       stop_prefetch_worker_.load(std::memory_order_relaxed);
            });

            if (stop_prefetch_worker_.load(std::memory_order_relaxed)) {
                break;
            }

            if (!prefetch_queue_.empty()) {
                request = prefetch_queue_.front();
                prefetch_queue_.pop();
                has_request = true;
            }
        }

        // Process request outside lock
        if (has_request && request.tensor != nullptr) {
            try {
                // Check if tensor is still on CPU
                if (request.tensor->device().type == Device::Type::CPU) {
                    // Issue async transfer
                    auto handle = transfer_engine_->cpu_to_gpu_async(
                        *request.tensor,
                        request.target_device
                    );
                    // Let transfer run in background
                    // Note: In production, we'd need to track handles and update tensors
                }
            } catch (const std::exception& e) {
                // Silently skip on error
                continue;
            }
        }
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
    // Sort by priority (LOW first, then HIGH), then by registration time
    std::sort(
        auto_offload_registry_.begin(),
        auto_offload_registry_.end(),
        [](const AutoOffloadEntry& a, const AutoOffloadEntry& b) {
            // Priority order: LOW < NORMAL < HIGH < CRITICAL
            if (a.priority != b.priority) {
                return static_cast<int>(a.priority) < static_cast<int>(b.priority);
            }
            // Within same priority, use FIFO (older first)
            return a.registration_time < b.registration_time;
        }
    );
}

} // namespace core
} // namespace tenzor
