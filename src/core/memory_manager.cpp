/**
 * @file memory_manager.cpp
 * @brief Implementation of memory manager for ZeRO offload
 */

#include "tenzor/core/memory_manager.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/checked_math.hpp"
#include <stdexcept>
#include <algorithm>

namespace tenzor {
namespace core {

// ============================================================================
// Constructor and Initialization
// ============================================================================

MemoryManager::MemoryManager(const Config& config)
    : config_(config) {
    // Initialize CPU memory limits
    cpu_memory_.memory_limit = config_.cpu_memory_limit;

    // Initialize per-device GPU memory limits
    cuda_memory_.memory_limit = config_.cuda_memory_limit;
    rocm_memory_.memory_limit = config_.rocm_memory_limit;
    oneapi_memory_.memory_limit = config_.oneapi_memory_limit;
    vulkan_memory_.memory_limit = config_.vulkan_memory_limit;

    // Initialize statistics
    stats_ = MemoryStats{};
}

// Special members are defined out-of-line here (rather than defaulted in the
// header) because tensors_ holds intrusive_ptr<TensorImpl> values whose
// destructor requires the complete TensorImpl type, available via tensor.hpp
// in this translation unit.
MemoryManager::~MemoryManager() = default;

// ============================================================================
// Tensor Registration and Tracking
// ============================================================================

auto MemoryManager::register_tensor(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        throw std::invalid_argument("Cannot register null tensor");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Key the registry on the intrusive TensorImpl identity, not the Tensor
    // wrapper address (which is copyable/movable and would dangle).
    TensorImpl* key = tensor->impl().get();
    if (key == nullptr) {
        throw std::invalid_argument("Cannot register tensor with null implementation");
    }

    // Check if already registered
    if (tensors_.find(key) != tensors_.end()) {
        return;  // Already registered
    }

    // Get tensor properties
    Device location = tensor->device();
    size_t size_bytes = calculate_tensor_size(tensor);

    // Create tensor info; hold an owning reference to keep the keyed
    // TensorImpl alive for as long as it is tracked.
    TensorInfo info(location, size_bytes);
    info.impl = tensor->impl();
    tensors_[key] = std::move(info);

    // Add to device-specific tracking
    auto& device_mem = get_device_memory(location.type);
    device_mem.memory_used += size_bytes;
    add_to_lru(tensor, device_mem);

    // Update statistics
    stats_.total_tensors++;
    switch (location.type) {
        case Device::Type::CPU:
            stats_.cpu_tensors++;
            stats_.cpu_memory_used += size_bytes;
            break;
        case Device::Type::CUDA:
            stats_.cuda_tensors++;
            stats_.cuda_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        case Device::Type::ROCm:
            stats_.rocm_tensors++;
            stats_.rocm_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        case Device::Type::OneAPI:
            stats_.oneapi_tensors++;
            stats_.oneapi_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        case Device::Type::Vulkan:
            stats_.vulkan_tensors++;
            stats_.vulkan_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        default:
            break;
    }

    update_stats(location.type);
}

auto MemoryManager::unregister_tensor(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    TensorImpl* key = tensor->impl().get();
    auto it = tensors_.find(key);
    if (it == tensors_.end()) {
        return;  // Not registered
    }

    const auto& info = it->second;
    Device location = info.location;
    size_t size_bytes = info.size_bytes;

    // Remove from device-specific tracking
    auto& device_mem = get_device_memory(location.type);
    device_mem.memory_used -= size_bytes;
    remove_from_lru(tensor, device_mem);

    // Update statistics
    stats_.total_tensors--;
    switch (location.type) {
        case Device::Type::CPU:
            stats_.cpu_tensors--;
            stats_.cpu_memory_used -= size_bytes;
            break;
        case Device::Type::CUDA:
            stats_.cuda_tensors--;
            stats_.cuda_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        case Device::Type::ROCm:
            stats_.rocm_tensors--;
            stats_.rocm_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        case Device::Type::OneAPI:
            stats_.oneapi_tensors--;
            stats_.oneapi_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        case Device::Type::Vulkan:
            stats_.vulkan_tensors--;
            stats_.vulkan_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        default:
            break;
    }

    // Remove from tracking
    tensors_.erase(it);

    update_stats(location.type);
}

auto MemoryManager::get_tensor_location(const Tensor* tensor) -> Device {
    if (tensor == nullptr) {
        throw std::invalid_argument("Cannot get location of null tensor");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tensors_.find(tensor->impl().get());
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not registered with memory manager");
    }

    return it->second.location;
}

auto MemoryManager::update_tensor_location(Tensor* tensor, Device new_location) -> void {
    if (tensor == nullptr) {
        throw std::invalid_argument("Cannot update location of null tensor");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tensors_.find(tensor->impl().get());
    if (it == tensors_.end()) {
        throw std::runtime_error("Tensor not registered with memory manager");
    }

    auto& info = it->second;
    Device old_location = info.location;

    if (old_location == new_location) {
        return;  // No change
    }

    size_t size_bytes = info.size_bytes;

    // Remove from old device tracking
    auto& old_device_mem = get_device_memory(old_location.type);
    old_device_mem.memory_used -= size_bytes;
    remove_from_lru(tensor, old_device_mem);

    // Update statistics for old location
    switch (old_location.type) {
        case Device::Type::CPU:
            stats_.cpu_tensors--;
            stats_.cpu_memory_used -= size_bytes;
            break;
        case Device::Type::CUDA:
            stats_.cuda_tensors--;
            stats_.cuda_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        case Device::Type::ROCm:
            stats_.rocm_tensors--;
            stats_.rocm_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        case Device::Type::OneAPI:
            stats_.oneapi_tensors--;
            stats_.oneapi_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        case Device::Type::Vulkan:
            stats_.vulkan_tensors--;
            stats_.vulkan_memory_used -= size_bytes;
            stats_.gpu_tensors--;
            stats_.gpu_memory_used -= size_bytes;
            break;
        default:
            break;
    }

    // Add to new device tracking
    auto& new_device_mem = get_device_memory(new_location.type);
    new_device_mem.memory_used += size_bytes;
    add_to_lru(tensor, new_device_mem);

    // Update statistics for new location
    switch (new_location.type) {
        case Device::Type::CPU:
            stats_.cpu_tensors++;
            stats_.cpu_memory_used += size_bytes;
            break;
        case Device::Type::CUDA:
            stats_.cuda_tensors++;
            stats_.cuda_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        case Device::Type::ROCm:
            stats_.rocm_tensors++;
            stats_.rocm_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        case Device::Type::OneAPI:
            stats_.oneapi_tensors++;
            stats_.oneapi_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        case Device::Type::Vulkan:
            stats_.vulkan_tensors++;
            stats_.vulkan_memory_used += size_bytes;
            stats_.gpu_tensors++;
            stats_.gpu_memory_used += size_bytes;
            break;
        default:
            break;
    }

    // Update tensor info
    info.location = new_location;
    info.last_access = std::chrono::steady_clock::now();

    update_stats(old_location.type);
    update_stats(new_location.type);
}

auto MemoryManager::is_registered(const Tensor* tensor) const -> bool {
    if (tensor == nullptr) {
        return false;
    }

    std::lock_guard<std::mutex> lock(mutex_);
    return tensors_.find(tensor->impl().get()) != tensors_.end();
}

// ============================================================================
// Memory Pressure Monitoring
// ============================================================================

auto MemoryManager::get_memory_usage(Device::Type device) const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return get_device_memory(device).memory_used;
}

auto MemoryManager::get_memory_limit(Device::Type device) const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return get_device_memory(device).memory_limit;
}

auto MemoryManager::get_memory_pressure(Device::Type device) const -> float {
    std::lock_guard<std::mutex> lock(mutex_);

    const auto& device_mem = get_device_memory(device);
    if (device_mem.memory_limit == 0) {
        return 0.0f;
    }

    return static_cast<float>(device_mem.memory_used) /
           static_cast<float>(device_mem.memory_limit);
}

auto MemoryManager::is_over_threshold(Device::Type device) const -> bool {
    return get_memory_pressure(device) > config_.eviction_threshold;
}

// ============================================================================
// LRU Eviction Policy
// ============================================================================

auto MemoryManager::evict_lru_tensors(Device::Type device, size_t target_bytes) -> std::vector<Tensor*> {
    std::lock_guard<std::mutex> lock(mutex_);

    std::vector<Tensor*> eviction_candidates;
    auto& device_mem = get_device_memory(device);

    size_t bytes_to_free = target_bytes;
    size_t freed_bytes = 0;

    // Iterate through LRU list from oldest to newest
    for (auto it = device_mem.lru_list.begin();
         it != device_mem.lru_list.end() && freed_bytes < bytes_to_free;
         ++it) {
        Tensor* tensor = *it;

        auto tensor_it = tensors_.find(tensor->impl().get());
        if (tensor_it != tensors_.end()) {
            size_t tensor_size = tensor_it->second.size_bytes;
            eviction_candidates.push_back(tensor);
            freed_bytes += tensor_size;
        }
    }

    // Update eviction statistics
    if (!eviction_candidates.empty()) {
        stats_.total_evictions += eviction_candidates.size();
    }

    return eviction_candidates;
}

auto MemoryManager::mark_tensor_used(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tensors_.find(tensor->impl().get());
    if (it == tensors_.end()) {
        return;  // Not registered
    }

    // Update last access time
    it->second.last_access = std::chrono::steady_clock::now();

    // Move to most recent in LRU list
    auto& device_mem = get_device_memory(it->second.location.type);
    move_to_recent(tensor, device_mem);

    // Update cache hit statistics
    if (config_.track_statistics) {
        stats_.total_cache_hits++;
    }
}

auto MemoryManager::get_lru_tensor(Device::Type device) -> Tensor* {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& device_mem = get_device_memory(device);
    if (device_mem.lru_list.empty()) {
        return nullptr;
    }

    // Front of list is least recently used
    return device_mem.lru_list.front();
}

// ============================================================================
// Statistics and Monitoring
// ============================================================================

auto MemoryManager::get_stats() const -> MemoryStats {
    std::lock_guard<std::mutex> lock(mutex_);
    return stats_;
}

auto MemoryManager::reset_stats() -> void {
    std::lock_guard<std::mutex> lock(mutex_);

    stats_.total_evictions = 0;
    stats_.total_cache_hits = 0;
    stats_.total_cache_misses = 0;
    stats_.peak_cpu_memory = 0;
    stats_.peak_gpu_memory = 0;
}

auto MemoryManager::get_tensor_count() const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);
    return tensors_.size();
}

auto MemoryManager::get_tensor_count(Device::Type device) const -> size_t {
    std::lock_guard<std::mutex> lock(mutex_);

    size_t count = 0;
    for (const auto& [impl, info] : tensors_) {
        if (info.location.type == device) {
            count++;
        }
    }
    return count;
}

// ============================================================================
// Private Helper Methods
// ============================================================================

auto MemoryManager::get_device_memory(Device::Type device) -> DeviceMemory& {
    switch (device) {
        case Device::Type::CPU:
            return cpu_memory_;
        case Device::Type::CUDA:
            return cuda_memory_;
        case Device::Type::ROCm:
            return rocm_memory_;
        case Device::Type::OneAPI:
            return oneapi_memory_;
        case Device::Type::Vulkan:
            return vulkan_memory_;
        default:
            // Do NOT fold unlisted device types (e.g. MPS) into the CPU pool —
            // that misattributes their memory. There is no bucket for them, so
            // reject explicitly rather than silently corrupting cpu_memory_ stats.
            throw std::invalid_argument(
                "MemoryManager::get_device_memory: no memory pool for device type " +
                std::to_string(static_cast<int>(device)));
    }
}

auto MemoryManager::get_device_memory(Device::Type device) const -> const DeviceMemory& {
    switch (device) {
        case Device::Type::CPU:
            return cpu_memory_;
        case Device::Type::CUDA:
            return cuda_memory_;
        case Device::Type::ROCm:
            return rocm_memory_;
        case Device::Type::OneAPI:
            return oneapi_memory_;
        case Device::Type::Vulkan:
            return vulkan_memory_;
        default:
            // Do NOT fold unlisted device types (e.g. MPS) into the CPU pool —
            // that misattributes their memory. There is no bucket for them, so
            // reject explicitly rather than silently corrupting cpu_memory_ stats.
            throw std::invalid_argument(
                "MemoryManager::get_device_memory: no memory pool for device type " +
                std::to_string(static_cast<int>(device)));
    }
}

auto MemoryManager::calculate_tensor_size(const Tensor* tensor) const -> size_t {
    if (tensor == nullptr) {
        return 0;
    }

    // Calculate size = num_elements * element_size, guarding against overflow
    // the same way the allocating paths (tensor.cpp) do via checked_mul.
    size_t num_elements = tensor->numel();
    size_t element_size = tensor->dtype_size();

    return static_cast<size_t>(checked_mul(static_cast<int64_t>(num_elements),
                                           static_cast<int64_t>(element_size)));
}

auto MemoryManager::update_stats(Device::Type device) -> void {
    if (!config_.track_statistics) {
        return;
    }

    const auto& device_mem = get_device_memory(device);

    // Update memory pressure and peak memory based on device type
    switch (device) {
        case Device::Type::CPU:
            if (device_mem.memory_limit > 0) {
                stats_.cpu_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                            static_cast<float>(device_mem.memory_limit);
            }
            if (device_mem.memory_used > stats_.peak_cpu_memory) {
                stats_.peak_cpu_memory = device_mem.memory_used;
            }
            break;
        case Device::Type::CUDA:
            if (device_mem.memory_limit > 0) {
                stats_.cuda_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                             static_cast<float>(device_mem.memory_limit);
            }
            if (device_mem.memory_used > stats_.peak_cuda_memory) {
                stats_.peak_cuda_memory = device_mem.memory_used;
            }
            break;
        case Device::Type::ROCm:
            if (device_mem.memory_limit > 0) {
                stats_.rocm_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                             static_cast<float>(device_mem.memory_limit);
            }
            if (device_mem.memory_used > stats_.peak_rocm_memory) {
                stats_.peak_rocm_memory = device_mem.memory_used;
            }
            break;
        case Device::Type::OneAPI:
            if (device_mem.memory_limit > 0) {
                stats_.oneapi_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                               static_cast<float>(device_mem.memory_limit);
            }
            if (device_mem.memory_used > stats_.peak_oneapi_memory) {
                stats_.peak_oneapi_memory = device_mem.memory_used;
            }
            break;
        case Device::Type::Vulkan:
            if (device_mem.memory_limit > 0) {
                stats_.vulkan_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                               static_cast<float>(device_mem.memory_limit);
            }
            if (device_mem.memory_used > stats_.peak_vulkan_memory) {
                stats_.peak_vulkan_memory = device_mem.memory_used;
            }
            break;
        default:
            break;
    }

    // Update overall GPU memory pressure (max across all GPU types)
    if (device != Device::Type::CPU) {
        float max_gpu_pressure = std::max({
            stats_.cuda_memory_pressure,
            stats_.rocm_memory_pressure,
            stats_.oneapi_memory_pressure,
            stats_.vulkan_memory_pressure
        });
        stats_.gpu_memory_pressure = max_gpu_pressure;

        // Update peak GPU memory (sum across all GPU types)
        size_t total_gpu_peak = stats_.peak_cuda_memory + stats_.peak_rocm_memory +
                               stats_.peak_oneapi_memory +
                               stats_.peak_vulkan_memory;
        if (total_gpu_peak > stats_.peak_gpu_memory) {
            stats_.peak_gpu_memory = total_gpu_peak;
        }
    }
}

auto MemoryManager::move_to_recent(Tensor* tensor, DeviceMemory& device_mem) -> void {
    auto lru_it = device_mem.lru_map.find(tensor);
    if (lru_it == device_mem.lru_map.end()) {
        return;  // Not in LRU list
    }

    // Remove from current position
    device_mem.lru_list.erase(lru_it->second);

    // Add to back (most recent)
    device_mem.lru_list.push_back(tensor);

    // Update iterator
    device_mem.lru_map[tensor] = std::prev(device_mem.lru_list.end());
}

auto MemoryManager::add_to_lru(Tensor* tensor, DeviceMemory& device_mem) -> void {
    // Add to back of list (most recent)
    device_mem.lru_list.push_back(tensor);

    // Store iterator for fast lookup
    device_mem.lru_map[tensor] = std::prev(device_mem.lru_list.end());
}

auto MemoryManager::remove_from_lru(Tensor* tensor, DeviceMemory& device_mem) -> void {
    auto lru_it = device_mem.lru_map.find(tensor);
    if (lru_it == device_mem.lru_map.end()) {
        return;  // Not in LRU list
    }

    // Remove from list
    device_mem.lru_list.erase(lru_it->second);

    // Remove from map
    device_mem.lru_map.erase(lru_it);
}

} // namespace core

// ============================================================================
// Auto-registration API
// ============================================================================

static std::atomic<bool> g_auto_tensor_registration{false};

auto enable_auto_tensor_registration() -> void {
    g_auto_tensor_registration.store(true, std::memory_order_release);
}

auto disable_auto_tensor_registration() -> void {
    g_auto_tensor_registration.store(false, std::memory_order_release);
}

auto is_auto_tensor_registration_enabled() -> bool {
    return g_auto_tensor_registration.load(std::memory_order_acquire);
}

} // namespace tenzor
