#pragma once

/**
 * @file vulkan_vma_allocator.hpp
 * @brief VMA (Vulkan Memory Allocator) based allocator for Vulkan backend
 *
 * Provides an alternative to VulkanCachingAllocator using AMD's VMA library
 * for improved suballocation, defragmentation, and memory budgeting.
 * Enabled at build time with -DTENZOR_USE_VMA=ON.
 */

#ifdef TENZOR_HAS_VMA

#include <cstddef>
#include <mutex>
#include <stdexcept>
#include <unordered_map>
#include <vulkan/vulkan.h>

// VMA configuration: use dynamic Vulkan functions, no stats by default
#define VMA_STATIC_VULKAN_FUNCTIONS 0
#define VMA_DYNAMIC_VULKAN_FUNCTIONS 1
#include <vk_mem_alloc.h>

namespace tenzor {
namespace backend {

/**
 * @brief VMA-backed allocator for Vulkan memory management.
 *
 * Uses AMD's Vulkan Memory Allocator library for:
 * - Suballocation from larger memory blocks (reduces vkAllocateMemory calls)
 * - Automatic defragmentation
 * - Memory budget tracking via VK_EXT_memory_budget
 * - Best-fit or buddy allocation strategies
 *
 * This allocator conforms to the same external interface as VulkanCachingAllocator
 * so it can be used as a drop-in replacement.
 */
class VulkanVMAAllocator {
public:
    static VulkanVMAAllocator& get() {
        static VulkanVMAAllocator instance;
        return instance;
    }

    /**
     * @brief Initialize VMA for a Vulkan device.
     */
    void initialize(VkInstance instance, VkDevice device,
                    VkPhysicalDevice physical_device, int device_index) {
        std::lock_guard<std::mutex> lock(mutex_);

        if (allocators_.count(device_index)) return;  // Already initialized

        VmaAllocatorCreateInfo create_info{};
        create_info.vulkanApiVersion = VK_API_VERSION_1_2;
        create_info.instance = instance;
        create_info.device = device;
        create_info.physicalDevice = physical_device;
        create_info.flags = VMA_ALLOCATOR_CREATE_EXT_MEMORY_BUDGET_BIT;

        VmaAllocator allocator = VK_NULL_HANDLE;
        VkResult result = vmaCreateAllocator(&create_info, &allocator);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("VMA: Failed to create allocator (VkResult " +
                                     std::to_string(result) + ")");
        }

        allocators_[device_index] = allocator;
        devices_[device_index] = device;
    }

    /**
     * @brief Allocate a buffer with VMA.
     *
     * @param size Size in bytes
     * @param device Device index
     * @param usage Vulkan buffer usage flags
     * @param properties Memory property flags
     * @return Mapped pointer to allocated memory
     */
    void* allocate(size_t size, int device = 0,
                   VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = allocators_.find(device);
        if (it == allocators_.end()) {
            throw std::runtime_error("VMA: Device " + std::to_string(device) + " not initialized");
        }

        VkBufferCreateInfo buffer_info{};
        buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        buffer_info.size = size;
        buffer_info.usage = usage;
        buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        VmaAllocationCreateInfo alloc_info{};
        alloc_info.usage = VMA_MEMORY_USAGE_AUTO;
        alloc_info.requiredFlags = properties;

        bool need_mapping = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
        if (need_mapping) {
            alloc_info.flags = VMA_ALLOCATION_CREATE_HOST_ACCESS_SEQUENTIAL_WRITE_BIT |
                               VMA_ALLOCATION_CREATE_MAPPED_BIT;
        }

        VkBuffer buffer = VK_NULL_HANDLE;
        VmaAllocation allocation = VK_NULL_HANDLE;
        VmaAllocationInfo allocation_info{};

        VkResult result = vmaCreateBuffer(it->second, &buffer_info, &alloc_info,
                                          &buffer, &allocation, &allocation_info);
        if (result != VK_SUCCESS) {
            throw std::runtime_error("VMA: Buffer allocation failed (VkResult " +
                                     std::to_string(result) + ", size=" +
                                     std::to_string(size) + ")");
        }

        // Use mapped pointer for host-visible, synthetic address for device-local
        void* ptr = allocation_info.pMappedData;
        if (!ptr) {
            // For device-local non-mapped memory, use the allocation handle as a key
            ptr = reinterpret_cast<void*>(reinterpret_cast<uintptr_t>(allocation));
        }

        // Track allocation for later deallocation
        alloc_map_[ptr] = {buffer, allocation, device};

        return ptr;
    }

    /**
     * @brief Free a VMA allocation.
     */
    void free(void* ptr, int device = 0) {
        std::lock_guard<std::mutex> lock(mutex_);

        auto it = alloc_map_.find(ptr);
        if (it == alloc_map_.end()) return;

        auto& info = it->second;
        auto alloc_it = allocators_.find(info.device);
        if (alloc_it != allocators_.end()) {
            vmaDestroyBuffer(alloc_it->second, info.buffer, info.allocation);
        }
        alloc_map_.erase(it);
    }

    /**
     * @brief Get the VkBuffer handle for an allocation.
     */
    VkBuffer get_buffer(void* ptr, int device = 0) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = alloc_map_.find(ptr);
        if (it == alloc_map_.end()) return VK_NULL_HANDLE;
        return it->second.buffer;
    }

    /**
     * @brief Run defragmentation pass.
     *
     * @warning Not implemented. A correct implementation must iterate
     * `VmaDefragmentationPassMoveInfo::pMoves`, recreate each buffer at its new
     * allocation, copy the contents via a command buffer, rebind, and update
     * `alloc_map_` so cached handles stay valid. Applying the VMA moves without
     * that bookkeeping would leave the handles in `alloc_map_` stale, so
     * subsequent get_buffer()/free() would use invalid handles. Rather than
     * silently pretend to defragment (the previous no-op implementation),
     * this entrypoint throws until real move application is implemented.
     */
    void defragment(int /*device*/ = 0) {
        throw std::runtime_error(
            "VulkanVMAAllocator::defragment: not implemented "
            "(move application and alloc_map_ rebind are required for "
            "correctness; the public entrypoint is disabled until then)");
    }

    /**
     * @brief Get memory budget information from VMA.
     */
    void print_stats(int device = 0) const {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocators_.find(device);
        if (it == allocators_.end()) return;

        VmaTotalStatistics stats{};
        vmaCalculateStatistics(it->second, &stats);

        auto& total = stats.total;
        fprintf(stderr, "[VMA] Device %d: %zu allocations, %zu bytes used, %zu bytes total\n",
                device, (size_t)total.statistics.allocationCount,
                (size_t)total.statistics.allocationBytes,
                (size_t)total.statistics.blockBytes);
    }

    /**
     * @brief Shutdown VMA for a device.
     */
    void shutdown_device(int device) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto it = allocators_.find(device);
        if (it == allocators_.end()) return;

        // Free all remaining allocations for this device
        for (auto alloc_it = alloc_map_.begin(); alloc_it != alloc_map_.end(); ) {
            if (alloc_it->second.device == device) {
                vmaDestroyBuffer(it->second, alloc_it->second.buffer,
                                 alloc_it->second.allocation);
                alloc_it = alloc_map_.erase(alloc_it);
            } else {
                ++alloc_it;
            }
        }

        vmaDestroyAllocator(it->second);
        allocators_.erase(it);
        devices_.erase(device);
    }

private:
    VulkanVMAAllocator() = default;
    ~VulkanVMAAllocator() {
        // Destroy every outstanding buffer with its owning device's allocator
        // BEFORE destroying the allocators themselves. Skipping this (the old
        // behavior) triggers a VMA leak/assertion in debug builds and leaks the
        // VkBuffer/VmaAllocation pairs in release. Mirrors shutdown_device().
        for (auto& [ptr, info] : alloc_map_) {
            auto alloc_it = allocators_.find(info.device);
            if (alloc_it != allocators_.end()) {
                vmaDestroyBuffer(alloc_it->second, info.buffer, info.allocation);
            }
        }
        alloc_map_.clear();

        for (auto& [device, allocator] : allocators_) {
            vmaDestroyAllocator(allocator);
        }
    }

    VulkanVMAAllocator(const VulkanVMAAllocator&) = delete;
    VulkanVMAAllocator& operator=(const VulkanVMAAllocator&) = delete;

    struct AllocationInfo {
        VkBuffer buffer;
        VmaAllocation allocation;
        int device;
    };

    mutable std::mutex mutex_;
    std::unordered_map<int, VmaAllocator> allocators_;
    std::unordered_map<int, VkDevice> devices_;
    std::unordered_map<void*, AllocationInfo> alloc_map_;
};

} // namespace backend
} // namespace tenzor

#endif // TENZOR_HAS_VMA
