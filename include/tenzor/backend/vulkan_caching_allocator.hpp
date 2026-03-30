#pragma once

#include <cstddef>
#include <memory>
#include <mutex>
#include <set>
#include <unordered_map>
#include <vector>

#include <vulkan/vulkan.h>

namespace tenzor {
namespace backend {

/**
 * @brief Memory block representation for the Vulkan caching allocator
 *
 * Supports sub-allocation: multiple blocks can share the same VkDeviceMemory,
 * each bound at a different offset. Only the block with owns_memory=true
 * is responsible for freeing the underlying memory.
 */
struct VulkanBlock {
    VkBuffer buffer;            // Vulkan buffer handle
    VkDeviceMemory memory;      // Device memory handle (may be shared with other blocks)
    void* mapped_ptr;           // Host-visible mapped pointer (or synthetic address)
    size_t size;                // Block size in bytes
    size_t memory_offset;       // Offset within the VkDeviceMemory where buffer is bound
    size_t memory_size;         // Total size of the underlying VkDeviceMemory allocation
    bool allocated;             // Whether block is currently allocated
    bool is_host_visible;       // Whether memory is host-visible (mappable)
    bool owns_memory;           // Whether this block owns the VkDeviceMemory (should free it)
    int device;                 // Device index
    uint32_t memory_type_index; // Memory type index for this block
    void* base_mapped_ptr;      // Base mapped pointer for the entire memory (for sub-allocation)

    VulkanBlock(VkBuffer buf, VkDeviceMemory mem, void* ptr, size_t s, int dev, uint32_t mem_type)
        : buffer(buf), memory(mem), mapped_ptr(ptr), size(s),
          memory_offset(0), memory_size(s), allocated(false), is_host_visible(true),
          owns_memory(true), device(dev), memory_type_index(mem_type), base_mapped_ptr(ptr) {}

    // Comparison for ordered containers (by size, then by pointer)
    bool operator<(const VulkanBlock& other) const {
        if (size != other.size) return size < other.size;
        return mapped_ptr < other.mapped_ptr;
    }
};

/**
 * @brief Comparator for ordering blocks by size (for best-fit allocation)
 */
struct VulkanBlockComparator {
    bool operator()(const VulkanBlock* a, const VulkanBlock* b) const {
        if (a->size != b->size) return a->size < b->size;
        return a->mapped_ptr < b->mapped_ptr;
    }
};

/**
 * @brief Statistics for Vulkan memory usage tracking
 */
struct VulkanMemoryStats {
    size_t allocated_bytes;     // Currently allocated to user
    size_t reserved_bytes;      // Total reserved (allocated + cached)
    size_t cached_bytes;        // Cached but not allocated
    size_t num_allocations;     // Total allocation calls
    size_t num_frees;           // Total free calls
    size_t num_cache_hits;      // Allocations satisfied from cache
    size_t num_splits;          // Number of block splits
    size_t num_merges;          // Number of block merges
    size_t peak_allocated;      // Peak allocated bytes

    VulkanMemoryStats()
        : allocated_bytes(0), reserved_bytes(0), cached_bytes(0),
          num_allocations(0), num_frees(0), num_cache_hits(0),
          num_splits(0), num_merges(0), peak_allocated(0) {}
};

/**
 * @brief VulkanCachingAllocator for efficient Vulkan GPU memory management
 *
 * Implements a memory pooling system that reuses freed memory blocks
 * to reduce vkAllocateMemory/vkFreeMemory overhead. Uses best-fit allocation
 * strategy with block splitting to reduce fragmentation.
 *
 * Thread-safe and supports multiple Vulkan devices.
 */
class VulkanCachingAllocator {
public:
    /**
     * @brief Get the singleton instance
     */
    static VulkanCachingAllocator& get();

    /**
     * @brief Check if the allocator singleton is still alive
     *
     * Returns false during static destruction when the allocator is being
     * or has been destroyed. Use this before calling get() to avoid
     * accessing destroyed memory.
     *
     * @return true if allocator is alive and safe to use
     */
    static bool is_alive();

    /**
     * @brief Initialize the allocator with Vulkan device context
     *
     * @param device Vulkan logical device
     * @param physical_device Vulkan physical device
     * @param device_index Device index for multi-GPU support
     */
    void initialize(VkDevice device, VkPhysicalDevice physical_device, int device_index);

    /**
     * @brief Check if allocator is initialized for a device
     */
    bool is_initialized(int device_index) const;

    /**
     * @brief Allocate memory from the cache or device
     *
     * @param size Size in bytes to allocate
     * @param device Device index
     * @param usage Buffer usage flags
     * @param properties Memory property flags
     * @return Mapped pointer to allocated memory
     * @throws std::runtime_error if allocation fails
     */
    void* allocate(size_t size, int device = 0,
                   VkBufferUsageFlags usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                                              VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                   VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT |
                                                      VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                                      VK_MEMORY_PROPERTY_HOST_COHERENT_BIT);

    /**
     * @brief Free memory back to the cache
     *
     * @param ptr Mapped pointer to free
     * @param device Device index
     */
    void free(void* ptr, int device = 0);

    /**
     * @brief Empty the cache and release all cached blocks
     *
     * @param device Device ID (-1 for all devices)
     */
    void empty_cache(int device = -1);

    /**
     * @brief Shutdown the allocator for a device before VkDevice is destroyed
     *
     * This must be called before vkDestroyDevice() to avoid use-after-free.
     * After shutdown:
     * - Releases all cached blocks properly
     * - Clears all blocks without Vulkan calls (device is about to be destroyed)
     * - Marks device as shutdown so future free() calls are safe
     *
     * @param device Device ID
     */
    void shutdown_device(int device);

    /**
     * @brief Get the VkBuffer handle for a mapped pointer
     *
     * @param ptr Mapped pointer
     * @param device Device index
     * @return VkBuffer handle
     */
    VkBuffer get_buffer(void* ptr, int device = 0) const;

    /**
     * @brief Find the VkBuffer and byte offset for a pointer that may be
     *        interior to an allocated block (e.g. tensor offset pointer).
     *
     * Searches all allocated blocks to find the one that contains @p ptr.
     * Returns the block's VkBuffer and the byte offset of ptr within it.
     *
     * @param ptr Pointer that may be base or interior to an allocation
     * @param device Device index
     * @return {VkBuffer, byte_offset} or {VK_NULL_HANDLE, 0} if not found
     */
    std::pair<VkBuffer, size_t> find_buffer_and_offset(const void* ptr, int device = 0) const;

    /**
     * @brief Check if a pointer refers to host-visible (mappable) memory
     *
     * @param ptr Mapped pointer or synthetic address
     * @param device Device index
     * @return true if memory is host-visible and can be directly accessed
     */
    bool is_memory_host_visible(void* ptr, int device = 0) const;

    /**
     * @brief Get the actual mapped pointer for host-visible memory
     *
     * For host-visible memory, returns the mapped pointer.
     * For device-local memory, returns nullptr.
     *
     * @param ptr Pointer (mapped or synthetic)
     * @param device Device index
     * @return Actual mapped pointer or nullptr
     */
    void* get_mapped_ptr(void* ptr, int device = 0) const;

    /**
     * @brief Staging buffer for CPU<->GPU transfers
     */
    struct StagingBuffer {
        VkBuffer buffer;
        VkDeviceMemory memory;
        void* mapped_ptr;
        size_t size;
        bool in_use;
    };

    /**
     * @brief Acquire a staging buffer for data transfer
     *
     * @param size Required size in bytes
     * @param device Device index
     * @return Staging buffer (caller must release when done)
     */
    StagingBuffer* acquire_staging_buffer(size_t size, int device = 0);

    /**
     * @brief Release a staging buffer back to the pool
     *
     * @param staging Staging buffer to release
     * @param device Device index
     */
    void release_staging_buffer(StagingBuffer* staging, int device = 0);

    /**
     * @brief Get currently allocated memory in bytes
     *
     * @param device Device ID (-1 for all devices)
     * @return Size in bytes
     */
    size_t memory_allocated(int device = -1) const;

    /**
     * @brief Get total reserved memory (allocated + cached) in bytes
     *
     * @param device Device ID (-1 for all devices)
     * @return Size in bytes
     */
    size_t memory_reserved(int device = -1) const;

    /**
     * @brief Get cached memory in bytes
     *
     * @param device Device ID (-1 for all devices)
     * @return Size in bytes
     */
    size_t memory_cached(int device = -1) const;

    /**
     * @brief Get memory statistics
     *
     * @param device Device ID (-1 for all devices)
     * @return Memory statistics
     */
    VulkanMemoryStats get_stats(int device = -1) const;

    /**
     * @brief Reset statistics counters
     */
    void reset_stats();

    /**
     * @brief Set memory allocation alignment (default: 256 bytes)
     *
     * @param alignment Alignment in bytes (must be power of 2)
     */
    void set_alignment(size_t alignment);

    /**
     * @brief Set maximum cached memory per device (default: unlimited)
     *
     * @param max_bytes Maximum cached memory in bytes (0 = unlimited)
     */
    void set_max_cached_memory(size_t max_bytes);

    /**
     * @brief Set minimum block size for splitting (default: 256 bytes)
     *
     * @param min_size Minimum size in bytes
     */
    void set_min_split_size(size_t min_size);

private:
    VulkanCachingAllocator();
    ~VulkanCachingAllocator();

    // Prevent copying
    VulkanCachingAllocator(const VulkanCachingAllocator&) = delete;
    VulkanCachingAllocator& operator=(const VulkanCachingAllocator&) = delete;

    /**
     * @brief Internal cache empty - assumes mutex is already held
     */
    void empty_cache_impl(int device);

    /**
     * @brief Try to allocate from cache
     */
    VulkanBlock* try_allocate_from_cache(size_t size, int device,
                                          VkBufferUsageFlags usage,
                                          VkMemoryPropertyFlags properties);

    /**
     * @brief Allocate new block from device
     */
    VulkanBlock* allocate_new_block(size_t size, int device,
                                     VkBufferUsageFlags usage,
                                     VkMemoryPropertyFlags properties);

    /**
     * @brief Split a block if it's larger than needed
     */
    bool split_block(VulkanBlock* block, size_t size);

    /**
     * @brief Try to merge adjacent free blocks sharing the same VkDeviceMemory.
     *
     * Scans the free list for blocks that are contiguous in the same
     * VkDeviceMemory allocation and merges them into larger blocks.
     * Called lazily on allocate cache-miss before falling back to new allocation.
     *
     * @param device Device index
     * @return Number of merges performed
     */
    size_t try_merge_free_blocks(int device);

    /**
     * @brief Round size up to alignment
     */
    size_t round_size(size_t size) const;

    /**
     * @brief Free blocks exceeding cache limit
     */
    void enforce_cache_limit(int device);

    /**
     * @brief Release a block back to device
     */
    void release_block(VulkanBlock* block);


    /**
     * @brief Find suitable memory type index
     */
    uint32_t find_memory_type(VkPhysicalDevice physical_device,
                              uint32_t type_filter,
                              VkMemoryPropertyFlags properties);

    // Per-device data structures
    struct DeviceAllocator {
        VkDevice device = VK_NULL_HANDLE;
        VkPhysicalDevice physical_device = VK_NULL_HANDLE;
        VkPhysicalDeviceMemoryProperties memory_properties{};
        size_t device_memory_size = 0;  // Total device-local VRAM
        bool initialized = false;
        bool shutdown = false;  // True after shutdown_device() called

        // Free blocks ordered by size (for best-fit)
        std::set<VulkanBlock*, VulkanBlockComparator> free_blocks;

        // All blocks (free and allocated) by mapped pointer
        std::unordered_map<void*, std::unique_ptr<VulkanBlock>> all_blocks;

        // Reference counts for shared VkDeviceMemory (for sub-allocation)
        // Key: VkDeviceMemory handle, Value: number of blocks referencing it
        std::unordered_map<VkDeviceMemory, size_t> memory_ref_counts;

        // Staging buffer pool for CPU<->GPU transfers
        std::vector<std::unique_ptr<StagingBuffer>> staging_pool;

        // Statistics
        VulkanMemoryStats stats;

        DeviceAllocator() = default;
    };

    /**
     * @brief Create a new staging buffer
     */
    StagingBuffer* create_staging_buffer(size_t size, int device);

    /**
     * @brief Destroy a staging buffer
     */
    void destroy_staging_buffer(StagingBuffer* staging, int device);

    // Mutex for thread safety
    mutable std::mutex mutex_;

    // Per-device allocators
    std::unordered_map<int, DeviceAllocator> device_allocators_;

    // Configuration
    size_t alignment_;
    size_t max_cached_memory_;
    size_t min_split_size_;
};

/**
 * @brief RAII wrapper for Vulkan cached memory allocation
 */
class VulkanCachedMemoryGuard {
public:
    VulkanCachedMemoryGuard(size_t size, int device = 0)
        : ptr_(nullptr), device_(device), size_(size) {
        ptr_ = VulkanCachingAllocator::get().allocate(size, device);
    }

    ~VulkanCachedMemoryGuard() {
        if (ptr_) {
            VulkanCachingAllocator::get().free(ptr_, device_);
        }
    }

    // Move semantics
    VulkanCachedMemoryGuard(VulkanCachedMemoryGuard&& other) noexcept
        : ptr_(other.ptr_), device_(other.device_), size_(other.size_) {
        other.ptr_ = nullptr;
    }

    VulkanCachedMemoryGuard& operator=(VulkanCachedMemoryGuard&& other) noexcept {
        if (this != &other) {
            if (ptr_) {
                VulkanCachingAllocator::get().free(ptr_, device_);
            }
            ptr_ = other.ptr_;
            device_ = other.device_;
            size_ = other.size_;
            other.ptr_ = nullptr;
        }
        return *this;
    }

    // Prevent copying
    VulkanCachedMemoryGuard(const VulkanCachedMemoryGuard&) = delete;
    VulkanCachedMemoryGuard& operator=(const VulkanCachedMemoryGuard&) = delete;

    void* get() const { return ptr_; }
    VkBuffer buffer() const {
        return VulkanCachingAllocator::get().get_buffer(ptr_, device_);
    }
    size_t size() const { return size_; }

private:
    void* ptr_;
    int device_;
    size_t size_;
};

} // namespace backend
} // namespace tenzor
