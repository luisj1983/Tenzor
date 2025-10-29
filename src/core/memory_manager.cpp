/**
 * @file memory_manager.cpp
 * @brief Implementation of memory manager for ZeRO offload
 */

#include "tenzor/core/memory_manager.hpp"
#include "tenzor/core/tensor.hpp"
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

    // Initialize GPU memory limits (unified across all GPU types)
    gpu_memory_.memory_limit = config_.gpu_memory_limit;

    // Initialize statistics
    stats_ = MemoryStats{};
}

// ============================================================================
// Tensor Registration and Tracking
// ============================================================================

auto MemoryManager::register_tensor(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        throw std::invalid_argument("Cannot register null tensor");
    }

    std::lock_guard<std::mutex> lock(mutex_);

    // Check if already registered
    if (tensors_.find(tensor) != tensors_.end()) {
        return;  // Already registered
    }

    // Get tensor properties
    Device location = tensor->device();
    size_t size_bytes = calculate_tensor_size(tensor);

    // Create tensor info
    TensorInfo info(location, size_bytes);
    tensors_[tensor] = info;

    // Add to device-specific tracking
    auto& device_mem = get_device_memory(location.type);
    device_mem.memory_used += size_bytes;
    add_to_lru(tensor, device_mem);

    // Update statistics
    stats_.total_tensors++;
    if (location.type == Device::Type::CPU) {
        stats_.cpu_tensors++;
        stats_.cpu_memory_used += size_bytes;
    } else {
        // All GPU types (CUDA, ROCm, OneAPI, etc.)
        stats_.gpu_tensors++;
        stats_.gpu_memory_used += size_bytes;
        if (location.type == Device::Type::CUDA) {
            stats_.cuda_tensors++;
            stats_.cuda_memory_used += size_bytes;
        }
    }

    update_stats(location.type);
}

auto MemoryManager::unregister_tensor(Tensor* tensor) -> void {
    if (tensor == nullptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = tensors_.find(tensor);
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
    if (location.type == Device::Type::CPU) {
        stats_.cpu_tensors--;
        stats_.cpu_memory_used -= size_bytes;
    } else {
        stats_.gpu_tensors--;
        stats_.gpu_memory_used -= size_bytes;
        if (location.type == Device::Type::CUDA) {
            stats_.cuda_tensors--;
            stats_.cuda_memory_used -= size_bytes;
        }
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

    auto it = tensors_.find(const_cast<Tensor*>(tensor));
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

    auto it = tensors_.find(tensor);
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
    if (old_location.type == Device::Type::CPU) {
        stats_.cpu_tensors--;
        stats_.cpu_memory_used -= size_bytes;
    } else {
        stats_.gpu_tensors--;
        stats_.gpu_memory_used -= size_bytes;
        if (old_location.type == Device::Type::CUDA) {
            stats_.cuda_tensors--;
            stats_.cuda_memory_used -= size_bytes;
        }
    }

    // Add to new device tracking
    auto& new_device_mem = get_device_memory(new_location.type);
    new_device_mem.memory_used += size_bytes;
    add_to_lru(tensor, new_device_mem);

    // Update statistics for new location
    if (new_location.type == Device::Type::CPU) {
        stats_.cpu_tensors++;
        stats_.cpu_memory_used += size_bytes;
    } else {
        stats_.gpu_tensors++;
        stats_.gpu_memory_used += size_bytes;
        if (new_location.type == Device::Type::CUDA) {
            stats_.cuda_tensors++;
            stats_.cuda_memory_used += size_bytes;
        }
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
    return tensors_.find(const_cast<Tensor*>(tensor)) != tensors_.end();
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

        auto tensor_it = tensors_.find(tensor);
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

    auto it = tensors_.find(tensor);
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
    for (const auto& [tensor, info] : tensors_) {
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
    if (device == Device::Type::CPU) {
        return cpu_memory_;
    } else {
        // All GPU types use unified GPU memory tracking
        return gpu_memory_;
    }
}

auto MemoryManager::get_device_memory(Device::Type device) const -> const DeviceMemory& {
    if (device == Device::Type::CPU) {
        return cpu_memory_;
    } else {
        return gpu_memory_;
    }
}

auto MemoryManager::calculate_tensor_size(const Tensor* tensor) const -> size_t {
    if (tensor == nullptr) {
        return 0;
    }

    // Calculate size = num_elements * element_size
    size_t num_elements = tensor->numel();
    size_t element_size = tensor->dtype_size();

    return num_elements * element_size;
}

auto MemoryManager::update_stats(Device::Type device) -> void {
    if (!config_.track_statistics) {
        return;
    }

    const auto& device_mem = get_device_memory(device);

    // Update memory pressure
    if (device == Device::Type::CPU) {
        if (device_mem.memory_limit > 0) {
            stats_.cpu_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                        static_cast<float>(device_mem.memory_limit);
        }

        // Update peak CPU memory
        if (device_mem.memory_used > stats_.peak_cpu_memory) {
            stats_.peak_cpu_memory = device_mem.memory_used;
        }
    } else {
        if (device_mem.memory_limit > 0) {
            stats_.gpu_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                        static_cast<float>(device_mem.memory_limit);
        }

        // Update CUDA-specific pressure if CUDA device
        if (device == Device::Type::CUDA && device_mem.memory_limit > 0) {
            stats_.cuda_memory_pressure = static_cast<float>(device_mem.memory_used) /
                                         static_cast<float>(device_mem.memory_limit);
        }

        // Update peak GPU memory
        if (device_mem.memory_used > stats_.peak_gpu_memory) {
            stats_.peak_gpu_memory = device_mem.memory_used;
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
} // namespace tenzor
