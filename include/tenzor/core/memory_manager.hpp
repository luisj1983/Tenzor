/**
 * @file memory_manager.hpp
 * @brief Memory manager for ZeRO offload with tensor tracking and eviction
 *
 * Provides comprehensive memory management for CPU/GPU tensor tracking,
 * memory pressure monitoring, and LRU-based eviction for ZeRO optimizer.
 */

#pragma once

#include <cstddef>
#include <unordered_map>
#include <list>
#include <mutex>
#include <vector>
#include <chrono>
#include "device.hpp"

namespace tenzor {

// Forward declaration
class Tensor;

namespace core {

/**
 * @brief Statistics for memory manager operations.
 */
struct MemoryStats {
    size_t total_tensors{0};           ///< Total tensors tracked
    size_t cpu_tensors{0};             ///< Tensors on CPU
    size_t cuda_tensors{0};            ///< Tensors on CUDA
    size_t rocm_tensors{0};            ///< Tensors on ROCm
    size_t oneapi_tensors{0};          ///< Tensors on OneAPI
    size_t vulkan_tensors{0};          ///< Tensors on Vulkan
    size_t gpu_tensors{0};             ///< Tensors on GPU (all types combined)
    size_t pinned_tensors{0};          ///< Tensors in pinned memory

    size_t cpu_memory_used{0};         ///< Bytes used on CPU
    size_t cuda_memory_used{0};        ///< Bytes used on CUDA
    size_t rocm_memory_used{0};        ///< Bytes used on ROCm
    size_t oneapi_memory_used{0};      ///< Bytes used on OneAPI
    size_t vulkan_memory_used{0};      ///< Bytes used on Vulkan
    size_t gpu_memory_used{0};         ///< Total GPU memory used (all types)
    size_t pinned_memory_used{0};      ///< Bytes in pinned memory

    size_t total_evictions{0};         ///< Total eviction operations
    size_t total_cache_hits{0};        ///< Cache hit count
    size_t total_cache_misses{0};      ///< Cache miss count

    float cpu_memory_pressure{0.0f};   ///< CPU memory pressure (0.0-1.0)
    float cuda_memory_pressure{0.0f};  ///< CUDA memory pressure (0.0-1.0)
    float rocm_memory_pressure{0.0f};  ///< ROCm memory pressure (0.0-1.0)
    float oneapi_memory_pressure{0.0f}; ///< OneAPI memory pressure (0.0-1.0)
    float vulkan_memory_pressure{0.0f}; ///< Vulkan memory pressure (0.0-1.0)
    float gpu_memory_pressure{0.0f};   ///< Overall GPU memory pressure (0.0-1.0)

    size_t peak_cpu_memory{0};         ///< Peak CPU memory usage
    size_t peak_cuda_memory{0};        ///< Peak CUDA memory usage
    size_t peak_rocm_memory{0};        ///< Peak ROCm memory usage
    size_t peak_oneapi_memory{0};      ///< Peak OneAPI memory usage
    size_t peak_vulkan_memory{0};      ///< Peak Vulkan memory usage
    size_t peak_gpu_memory{0};         ///< Peak GPU memory usage (all types)
};

/**
 * @brief Memory manager for tensor location tracking and eviction.
 *
 * Provides:
 * - Tensor location tracking across CPU/GPU/Pinned memory
 * - Memory pressure monitoring with configurable thresholds
 * - LRU eviction policy for memory management
 * - Thread-safe operations
 * - Statistics tracking for performance analysis
 *
 * Usage:
 * @code
 * MemoryManager::Config config;
 * config.cpu_memory_limit = 16ULL * 1024 * 1024 * 1024;  // 16 GB
 * config.gpu_memory_limit = 8ULL * 1024 * 1024 * 1024;   // 8 GB
 * config.eviction_threshold = 0.9f;                       // 90%
 *
 * MemoryManager manager(config);
 * manager.register_tensor(&tensor);
 *
 * if (manager.is_over_threshold(Device::Type::CUDA)) {
 *     auto evicted = manager.evict_lru_tensors(Device::Type::CUDA, 1024*1024*100);
 * }
 * @endcode
 */
class MemoryManager {
public:
    /**
     * @brief Configuration for memory manager.
     */
    struct Config {
        size_t cpu_memory_limit{16ULL * 1024 * 1024 * 1024};    ///< CPU memory limit (16 GB)
        size_t cuda_memory_limit{8ULL * 1024 * 1024 * 1024};    ///< CUDA memory limit (8 GB)
        size_t rocm_memory_limit{8ULL * 1024 * 1024 * 1024};    ///< ROCm memory limit (8 GB)
        size_t oneapi_memory_limit{8ULL * 1024 * 1024 * 1024};  ///< OneAPI memory limit (8 GB)
        size_t vulkan_memory_limit{8ULL * 1024 * 1024 * 1024};  ///< Vulkan memory limit (8 GB)
        size_t gpu_memory_limit{8ULL * 1024 * 1024 * 1024};     ///< Default GPU memory limit (backward compat)
        float eviction_threshold{0.9f};                         ///< Eviction threshold (0.0-1.0)
        bool track_statistics{true};                            ///< Enable statistics tracking
        bool enable_cache{true};                                ///< Enable cache tracking
    };

    /**
     * @brief Construct memory manager with configuration.
     *
     * @param config Configuration settings
     */
    explicit MemoryManager(const Config& config);

    /**
     * @brief Destructor.
     */
    ~MemoryManager() = default;

    // Disable copy, allow move
    MemoryManager(const MemoryManager&) = delete;
    MemoryManager& operator=(const MemoryManager&) = delete;
    MemoryManager(MemoryManager&&) noexcept = default;
    MemoryManager& operator=(MemoryManager&&) noexcept = default;

    // ========================================================================
    // Tensor Registration and Tracking
    // ========================================================================

    /**
     * @brief Register tensor for tracking.
     *
     * Adds tensor to tracking system and records its location.
     * Thread-safe.
     *
     * @param tensor Pointer to tensor to track
     */
    auto register_tensor(Tensor* tensor) -> void;

    /**
     * @brief Unregister tensor from tracking.
     *
     * Removes tensor from tracking system.
     * Thread-safe.
     *
     * @param tensor Pointer to tensor to untrack
     */
    auto unregister_tensor(Tensor* tensor) -> void;

    /**
     * @brief Get tensor location.
     *
     * @param tensor Pointer to tensor
     * @return Device where tensor resides
     * @throws std::runtime_error if tensor not registered
     */
    auto get_tensor_location(const Tensor* tensor) -> Device;

    /**
     * @brief Update tensor location.
     *
     * Updates tracked location when tensor moves between devices.
     * Thread-safe.
     *
     * @param tensor Pointer to tensor
     * @param new_location New device location
     */
    auto update_tensor_location(Tensor* tensor, Device new_location) -> void;

    /**
     * @brief Check if tensor is registered.
     *
     * @param tensor Pointer to tensor
     * @return true if tensor is tracked
     */
    auto is_registered(const Tensor* tensor) const -> bool;

    // ========================================================================
    // Memory Pressure Monitoring
    // ========================================================================

    /**
     * @brief Get current memory usage for device type.
     *
     * @param device Device type (CPU, CUDA, etc.)
     * @return Bytes currently used
     */
    auto get_memory_usage(Device::Type device) const -> size_t;

    /**
     * @brief Get memory limit for device type.
     *
     * @param device Device type
     * @return Configured memory limit in bytes
     */
    auto get_memory_limit(Device::Type device) const -> size_t;

    /**
     * @brief Get memory pressure for device type.
     *
     * Pressure = current_usage / memory_limit
     *
     * @param device Device type
     * @return Pressure value (0.0 = empty, 1.0 = full)
     */
    auto get_memory_pressure(Device::Type device) const -> float;

    /**
     * @brief Check if memory usage is over eviction threshold.
     *
     * @param device Device type
     * @return true if pressure > eviction_threshold
     */
    auto is_over_threshold(Device::Type device) const -> bool;

    // ========================================================================
    // LRU Eviction Policy
    // ========================================================================

    /**
     * @brief Evict least recently used tensors to free memory.
     *
     * Evicts tensors in LRU order until target_bytes are freed.
     * Does NOT actually move tensor data - returns list of candidates.
     *
     * @param device Device type to evict from
     * @param target_bytes Minimum bytes to free
     * @return Vector of tensors to evict (in LRU order)
     */
    auto evict_lru_tensors(Device::Type device, size_t target_bytes) -> std::vector<Tensor*>;

    /**
     * @brief Mark tensor as recently used.
     *
     * Updates LRU tracking to mark tensor as most recently used.
     * Thread-safe.
     *
     * @param tensor Pointer to tensor that was accessed
     */
    auto mark_tensor_used(Tensor* tensor) -> void;

    /**
     * @brief Get least recently used tensor for device.
     *
     * @param device Device type
     * @return Pointer to LRU tensor, or nullptr if none
     */
    auto get_lru_tensor(Device::Type device) -> Tensor*;

    // ========================================================================
    // Statistics and Monitoring
    // ========================================================================

    /**
     * @brief Get current statistics.
     *
     * @return Statistics snapshot
     */
    auto get_stats() const -> MemoryStats;

    /**
     * @brief Reset statistics counters.
     *
     * Resets eviction counts, cache hits/misses, and peak memory.
     */
    auto reset_stats() -> void;

    /**
     * @brief Get number of registered tensors.
     *
     * @return Total tensor count
     */
    auto get_tensor_count() const -> size_t;

    /**
     * @brief Get number of tensors on specific device.
     *
     * @param device Device type
     * @return Tensor count on device
     */
    auto get_tensor_count(Device::Type device) const -> size_t;

private:
    /**
     * @brief Internal tensor tracking information.
     */
    struct TensorInfo {
        Device location;                           ///< Current device location
        size_t size_bytes{0};                      ///< Tensor size in bytes
        std::chrono::steady_clock::time_point last_access;  ///< Last access time

        TensorInfo() : location(Device::cpu()), last_access(std::chrono::steady_clock::now()) {}
        TensorInfo(Device loc, size_t size)
            : location(loc), size_bytes(size), last_access(std::chrono::steady_clock::now()) {}
    };

    /**
     * @brief LRU list node (tensor pointer).
     */
    using LRUList = std::list<Tensor*>;
    using LRUIterator = LRUList::iterator;

    /**
     * @brief Per-device LRU tracking.
     */
    struct DeviceMemory {
        size_t memory_used{0};                     ///< Current memory usage
        size_t memory_limit{0};                    ///< Memory limit
        LRUList lru_list;                          ///< LRU list (front=oldest, back=newest)
        std::unordered_map<Tensor*, LRUIterator> lru_map;  ///< Fast LRU lookup
    };

    Config config_;                                ///< Configuration
    mutable std::mutex mutex_;                     ///< Thread safety

    // Tensor tracking
    std::unordered_map<Tensor*, TensorInfo> tensors_;  ///< All tracked tensors
    std::unordered_map<Tensor*, LRUIterator> tensor_to_lru_;  ///< Tensor to LRU iterator

    // Per-device memory tracking
    DeviceMemory cpu_memory_;                      ///< CPU memory tracking
    DeviceMemory cuda_memory_;                     ///< CUDA memory tracking
    DeviceMemory rocm_memory_;                     ///< ROCm memory tracking
    DeviceMemory oneapi_memory_;                   ///< OneAPI memory tracking
    DeviceMemory vulkan_memory_;                   ///< Vulkan memory tracking

    // Statistics
    MemoryStats stats_;                            ///< Runtime statistics

    /**
     * @brief Get device memory tracker for device type.
     *
     * @param device Device type
     * @return Reference to device memory tracker
     */
    auto get_device_memory(Device::Type device) -> DeviceMemory&;

    /**
     * @brief Get device memory tracker for device type (const).
     *
     * @param device Device type
     * @return Const reference to device memory tracker
     */
    auto get_device_memory(Device::Type device) const -> const DeviceMemory&;

    /**
     * @brief Calculate tensor size in bytes.
     *
     * @param tensor Pointer to tensor
     * @return Size in bytes
     */
    auto calculate_tensor_size(const Tensor* tensor) const -> size_t;

    /**
     * @brief Update statistics after memory change.
     *
     * @param device Device type
     */
    auto update_stats(Device::Type device) -> void;

    /**
     * @brief Move tensor in LRU list to most recent position.
     *
     * @param tensor Pointer to tensor
     * @param device_mem Device memory tracker
     */
    auto move_to_recent(Tensor* tensor, DeviceMemory& device_mem) -> void;

    /**
     * @brief Add tensor to LRU tracking.
     *
     * @param tensor Pointer to tensor
     * @param device_mem Device memory tracker
     */
    auto add_to_lru(Tensor* tensor, DeviceMemory& device_mem) -> void;

    /**
     * @brief Remove tensor from LRU tracking.
     *
     * @param tensor Pointer to tensor
     * @param device_mem Device memory tracker
     */
    auto remove_from_lru(Tensor* tensor, DeviceMemory& device_mem) -> void;
};

} // namespace core
} // namespace tenzor
