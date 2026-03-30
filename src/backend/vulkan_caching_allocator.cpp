#include "tenzor/backend/vulkan_caching_allocator.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cstring>
#include <iostream>
#include <atomic>

namespace tenzor {
namespace backend {

// Default configuration
constexpr size_t DEFAULT_ALIGNMENT = 256;  // Vulkan-optimized alignment
constexpr size_t DEFAULT_MIN_SPLIT_SIZE = 256;

// Flag to track if the allocator singleton is being/has been destroyed
// This prevents crashes when shutdown_device is called after the allocator is destroyed
static std::atomic<bool> g_allocator_destroyed{false};

VulkanCachingAllocator::VulkanCachingAllocator()
    : alignment_(DEFAULT_ALIGNMENT),
      max_cached_memory_(0),  // Unlimited by default
      min_split_size_(DEFAULT_MIN_SPLIT_SIZE) {
}

VulkanCachingAllocator::~VulkanCachingAllocator() {
    // Mark as destroyed FIRST to prevent any calls during destruction
    g_allocator_destroyed.store(true, std::memory_order_release);

    // Release all cached memory
    empty_cache(-1);
}

// Check if the allocator singleton is still alive
bool VulkanCachingAllocator::is_alive() {
    return !g_allocator_destroyed.load(std::memory_order_acquire);
}

VulkanCachingAllocator& VulkanCachingAllocator::get() {
    static VulkanCachingAllocator instance;
    return instance;
}

void VulkanCachingAllocator::initialize(VkDevice device, VkPhysicalDevice physical_device, int device_index) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto& device_alloc = device_allocators_[device_index];
    if (device_alloc.initialized) {
        return;  // Already initialized
    }

    device_alloc.device = device;
    device_alloc.physical_device = physical_device;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &device_alloc.memory_properties);

    // Find the largest device-local memory heap
    size_t max_device_local_memory = 0;
    for (uint32_t i = 0; i < device_alloc.memory_properties.memoryHeapCount; i++) {
        const auto& heap = device_alloc.memory_properties.memoryHeaps[i];
        if (heap.flags & VK_MEMORY_HEAP_DEVICE_LOCAL_BIT) {
            if (heap.size > max_device_local_memory) {
                max_device_local_memory = heap.size;
            }
        }
    }

    // Store device memory size for proactive cache management
    device_alloc.device_memory_size = max_device_local_memory;

    device_alloc.initialized = true;
}

bool VulkanCachingAllocator::is_initialized(int device_index) const {
    std::lock_guard<std::mutex> lock(mutex_);
    auto it = device_allocators_.find(device_index);
    return it != device_allocators_.end() && it->second.initialized;
}

void* VulkanCachingAllocator::allocate(size_t size, int device,
                                        VkBufferUsageFlags usage,
                                        VkMemoryPropertyFlags properties) {
    if (size == 0) {
        return nullptr;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_allocators_.find(device);
    if (it == device_allocators_.end() || !it->second.initialized) {
        throw std::runtime_error("VulkanCachingAllocator: Device " + std::to_string(device) +
                                 " not initialized. Call initialize() first.");
    }

    // Round size to alignment
    size = round_size(size);

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_allocations++;

    // NOTE: Proactive cache clearing disabled to match CUDA behavior.
    // CUDA uses all available memory and relies on OOM retry to clear cache.
    // The OOM retry mechanism at line 456 handles memory pressure.

    // Try to find a suitable block in cache
    VulkanBlock* block = try_allocate_from_cache(size, device, usage, properties);

    if (!block) {
        // No suitable cached block — try merging adjacent free blocks first
        if (try_merge_free_blocks(device) > 0) {
            block = try_allocate_from_cache(size, device, usage, properties);
        }
    }

    bool from_cache = (block != nullptr);

    if (!block) {
        // Still no suitable block, allocate new one
        block = allocate_new_block(size, device, usage, properties);
    } else {
        device_alloc.stats.num_cache_hits++;
    }

    // Mark block as allocated
    block->allocated = true;

    // Update statistics
    device_alloc.stats.allocated_bytes += block->size;
    if (device_alloc.stats.allocated_bytes > device_alloc.stats.peak_allocated) {
        device_alloc.stats.peak_allocated = device_alloc.stats.allocated_bytes;
    }

    // Only subtract from cached_bytes if we got the block from cache
    if (from_cache && device_alloc.stats.cached_bytes >= block->size) {
        device_alloc.stats.cached_bytes -= block->size;
    }

    return block->mapped_ptr;
}

void VulkanCachingAllocator::free(void* ptr, int device) {
    if (!ptr) {
        return;
    }

    std::lock_guard<std::mutex> lock(mutex_);

    auto dev_it = device_allocators_.find(device);
    if (dev_it == device_allocators_.end()) {
        // Device not found - might have been shutdown already
        return;
    }

    auto& device_alloc = dev_it->second;

    // If device is shutdown, the block was already cleaned up
    if (device_alloc.shutdown) {
        return;
    }

    device_alloc.stats.num_frees++;

    // Find the block
    auto it = device_alloc.all_blocks.find(ptr);
    if (it == device_alloc.all_blocks.end()) {
        // Block not found - might have been cleaned up during shutdown
        return;
    }

    VulkanBlock* block = it->second.get();
    if (!block->allocated) {
        throw std::runtime_error("VulkanCachingAllocator: Attempted to free already freed pointer");
    }

    // Mark as free
    block->allocated = false;

    // Update statistics
    if (device_alloc.stats.allocated_bytes >= block->size) {
        device_alloc.stats.allocated_bytes -= block->size;
    }

    // Always cache the block for reuse (like CUDA allocator)
    // OOM retry mechanism will clear cache when needed
    device_alloc.stats.cached_bytes += block->size;
    device_alloc.free_blocks.insert(block);

    // Enforce cache limit if set
    if (max_cached_memory_ > 0) {
        enforce_cache_limit(device);
    }
}

// Internal implementation that assumes mutex is already held
void VulkanCachingAllocator::empty_cache_impl(int device)
{
    if (device == -1) {
        // Empty all devices
        for (auto& pair : device_allocators_) {
            auto& device_alloc = pair.second;
            if (!device_alloc.initialized) continue;

            // Release all free blocks
            std::vector<VulkanBlock*> blocks_to_release;
            for (VulkanBlock* block : device_alloc.free_blocks) {
                blocks_to_release.push_back(block);
            }

            for (VulkanBlock* block : blocks_to_release) {
                release_block(block);
            }
        }
    } else {
        // Empty specific device
        auto it = device_allocators_.find(device);
        if (it == device_allocators_.end() || !it->second.initialized) {
            return;  // Device doesn't exist or not initialized
        }

        auto& device_alloc = it->second;

        std::vector<VulkanBlock*> blocks_to_release;
        for (VulkanBlock* block : device_alloc.free_blocks) {
            blocks_to_release.push_back(block);
        }

        for (VulkanBlock* block : blocks_to_release) {
            release_block(block);
        }
    }
}

void VulkanCachingAllocator::empty_cache(int device)
{
    std::lock_guard<std::mutex> lock(mutex_);
    empty_cache_impl(device);
}

void VulkanCachingAllocator::shutdown_device(int device)
{
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_allocators_.find(device);
    if (it == device_allocators_.end() || !it->second.initialized) {
        return;  // Device doesn't exist or not initialized
    }

    auto& device_alloc = it->second;

    // First, release all free blocks properly (device is still valid at this point)
    std::vector<VulkanBlock*> blocks_to_release;
    for (VulkanBlock* block : device_alloc.free_blocks) {
        blocks_to_release.push_back(block);
    }
    for (VulkanBlock* block : blocks_to_release) {
        release_block(block);
    }

    // Now mark device as shutdown - any remaining allocated blocks will be
    // cleaned up without Vulkan calls when their tensors are destroyed
    device_alloc.shutdown = true;

    // Also mark as uninitialized to prevent any future operations
    device_alloc.initialized = false;

    // Clear any remaining blocks without Vulkan calls (device about to be destroyed)
    // Just clear the containers - the VkBuffer/VkDeviceMemory handles will be
    // invalidated when vkDestroyDevice is called anyway
    device_alloc.free_blocks.clear();
    device_alloc.all_blocks.clear();
    device_alloc.memory_ref_counts.clear();

    // Set device handle to null to catch any accidental use
    device_alloc.device = VK_NULL_HANDLE;

    // Clear staging buffers (also without Vulkan calls - they'll be invalidated)
    device_alloc.staging_pool.clear();
}


VkBuffer VulkanCachingAllocator::get_buffer(void* ptr, int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto dev_it = device_allocators_.find(device);
    if (dev_it == device_allocators_.end()) {
        throw std::runtime_error("VulkanCachingAllocator: Device not found");
    }

    // If device is shutdown, return null handle
    if (dev_it->second.shutdown) {
        return VK_NULL_HANDLE;
    }

    auto it = dev_it->second.all_blocks.find(ptr);
    if (it == dev_it->second.all_blocks.end()) {
        throw std::runtime_error("VulkanCachingAllocator: Pointer not found");
    }

    return it->second->buffer;
}

std::pair<VkBuffer, size_t> VulkanCachingAllocator::find_buffer_and_offset(const void* ptr, int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto dev_it = device_allocators_.find(device);
    if (dev_it == device_allocators_.end() || dev_it->second.shutdown) {
        return {VK_NULL_HANDLE, 0};
    }

    const auto* ptr_u8 = reinterpret_cast<const uint8_t*>(ptr);
    VulkanBlock* best = nullptr;

    for (const auto& [block_ptr, block] : dev_it->second.all_blocks) {
        if (!block->allocated) continue;  // Only search actively-allocated blocks
        const auto* base = reinterpret_cast<const uint8_t*>(block->mapped_ptr);
        if (ptr_u8 >= base && ptr_u8 < base + block->size) {
            // Pick the block with the highest base (tightest enclosing)
            if (!best || base > reinterpret_cast<const uint8_t*>(best->mapped_ptr)) {
                best = block.get();
            }
        }
    }

    if (best) {
        size_t offset = static_cast<size_t>(ptr_u8 - reinterpret_cast<const uint8_t*>(best->mapped_ptr));
        return {best->buffer, offset};
    }
    return {VK_NULL_HANDLE, 0};
}

size_t VulkanCachingAllocator::memory_allocated(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            total += pair.second.stats.allocated_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats.allocated_bytes;
    }
    return 0;
}

size_t VulkanCachingAllocator::memory_reserved(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            total += pair.second.stats.reserved_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats.reserved_bytes;
    }
    return 0;
}

size_t VulkanCachingAllocator::memory_cached(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        size_t total = 0;
        for (const auto& pair : device_allocators_) {
            total += pair.second.stats.cached_bytes;
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats.cached_bytes;
    }
    return 0;
}

VulkanMemoryStats VulkanCachingAllocator::get_stats(int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    if (device == -1) {
        VulkanMemoryStats total;
        for (const auto& pair : device_allocators_) {
            const auto& stats = pair.second.stats;
            total.allocated_bytes += stats.allocated_bytes;
            total.reserved_bytes += stats.reserved_bytes;
            total.cached_bytes += stats.cached_bytes;
            total.num_allocations += stats.num_allocations;
            total.num_frees += stats.num_frees;
            total.num_cache_hits += stats.num_cache_hits;
            total.num_splits += stats.num_splits;
            total.num_merges += stats.num_merges;
            if (stats.peak_allocated > total.peak_allocated) {
                total.peak_allocated = stats.peak_allocated;
            }
        }
        return total;
    }

    auto it = device_allocators_.find(device);
    if (it != device_allocators_.end()) {
        return it->second.stats;
    }
    return VulkanMemoryStats();
}

void VulkanCachingAllocator::reset_stats() {
    std::lock_guard<std::mutex> lock(mutex_);

    for (auto& pair : device_allocators_) {
        auto& stats = pair.second.stats;
        stats.num_allocations = 0;
        stats.num_frees = 0;
        stats.num_cache_hits = 0;
        stats.num_splits = 0;
        stats.num_merges = 0;
    }
}

void VulkanCachingAllocator::set_alignment(size_t alignment) {
    if (alignment == 0 || (alignment & (alignment - 1)) != 0) {
        throw std::invalid_argument("Alignment must be a power of 2");
    }
    std::lock_guard<std::mutex> lock(mutex_);
    alignment_ = alignment;
}

void VulkanCachingAllocator::set_max_cached_memory(size_t max_bytes) {
    std::lock_guard<std::mutex> lock(mutex_);
    max_cached_memory_ = max_bytes;
}

void VulkanCachingAllocator::set_min_split_size(size_t min_size) {
    std::lock_guard<std::mutex> lock(mutex_);
    min_split_size_ = min_size;
}

VulkanBlock* VulkanCachingAllocator::try_allocate_from_cache(size_t size, int device,
                                                              VkBufferUsageFlags /*usage*/,
                                                              VkMemoryPropertyFlags properties) {
    auto& device_alloc = device_allocators_[device];
    bool need_host_visible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;

    // Find smallest free block that fits (best-fit) with compatible memory type
    // Create a temporary block for comparison
    VulkanBlock search_block(VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, size, device, 0);
    auto it = device_alloc.free_blocks.lower_bound(&search_block);

    // Iterate through candidates to find one with compatible memory type
    while (it != device_alloc.free_blocks.end()) {
        VulkanBlock* block = *it;

        // Check memory type compatibility:
        // - If HOST_VISIBLE is requested, block must be host-visible
        // - If HOST_VISIBLE is not requested, any block is acceptable
        //   (DEVICE_LOCAL is always suitable for non-HOST_VISIBLE requests)
        if (!need_host_visible || block->is_host_visible) {
            // Found compatible block, remove from free blocks
            device_alloc.free_blocks.erase(it);

            // Try to split if block is too large (reclaim unused portion)
            if (block->size >= size + min_split_size_) {
                split_block(block, size);
            }

            return block;
        }

        // Try next larger block
        ++it;
    }

    return nullptr;
}

VulkanBlock* VulkanCachingAllocator::allocate_new_block(size_t size, int device,
                                                         VkBufferUsageFlags usage,
                                                         VkMemoryPropertyFlags properties) {
    auto& device_alloc = device_allocators_[device];

    // Slab allocation: round up to reduce the number of vkAllocateMemory calls.
    // Each VkDeviceMemory has ~256KB driver overhead on NVIDIA, so fewer is better.
    // Adjacent free blocks are lazily merged in try_merge_free_blocks() on cache miss.
    constexpr size_t kSmallSlab  = 2 * 1024 * 1024;    // 2 MB for small allocs
    constexpr size_t kMediumSlab = 32 * 1024 * 1024;   // 32 MB
    constexpr size_t kLargeSlab  = 256 * 1024 * 1024;  // 256 MB

    // When memory pressure is high (>60% of device memory reserved), skip slab
    // allocation to avoid fragmentation that wastes the remaining budget.
    bool memory_pressure_high = (device_alloc.stats.reserved_bytes >
                                  device_alloc.device_memory_size * 3 / 5);

    size_t slab_size = size;
    if (!memory_pressure_high) {
        if (size < kSmallSlab) {
            slab_size = kSmallSlab;
        } else if (size < kMediumSlab) {
            slab_size = kMediumSlab;
        } else if (size < kLargeSlab) {
            slab_size = kLargeSlab;
        }
    }

    // Create buffer with slab size
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = slab_size;
    buffer_info.usage = usage;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_alloc.device, &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("VulkanCachingAllocator: Failed to create buffer, VkResult: " +
                                 std::to_string(result));
    }

    // Get memory requirements
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_alloc.device, buffer, &mem_requirements);

    // Allocate memory
    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(device_alloc.physical_device,
                                                   mem_requirements.memoryTypeBits,
                                                   properties);

    VkDeviceMemory memory;
    bool using_relaxed_props = false;
    VkMemoryPropertyFlags relaxed_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;

    result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);

    // If allocation fails with HOST_VISIBLE requested, retry with DEVICE_LOCAL only
    // This handles discrete GPUs where BAR (host-visible VRAM) is limited
    if (result != VK_SUCCESS && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        // Retry with just DEVICE_LOCAL
        alloc_info.memoryTypeIndex = find_memory_type(device_alloc.physical_device,
                                                       mem_requirements.memoryTypeBits,
                                                       relaxed_props);
        result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);
        if (result == VK_SUCCESS) {
            // Successfully allocated with device-local only
            using_relaxed_props = true;
            properties = relaxed_props;
        } else {
            // Track that we're now trying with relaxed props
            using_relaxed_props = true;
        }
    }

    // If slab allocation fails, fall back to exact size before giving up
    if (result != VK_SUCCESS && slab_size > size) {
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);

        // Recreate buffer with exact requested size
        slab_size = size;
        buffer_info.size = slab_size;
        result = vkCreateBuffer(device_alloc.device, &buffer_info, nullptr, &buffer);
        if (result == VK_SUCCESS) {
            vkGetBufferMemoryRequirements(device_alloc.device, buffer, &mem_requirements);
            alloc_info.allocationSize = mem_requirements.size;
            alloc_info.memoryTypeIndex = find_memory_type(device_alloc.physical_device,
                                                           mem_requirements.memoryTypeBits,
                                                           properties);
            result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);

            if (result != VK_SUCCESS && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
                alloc_info.memoryTypeIndex = find_memory_type(device_alloc.physical_device,
                                                               mem_requirements.memoryTypeBits,
                                                               relaxed_props);
                result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);
                if (result == VK_SUCCESS) {
                    using_relaxed_props = true;
                    properties = relaxed_props;
                } else {
                    using_relaxed_props = true;
                }
            }
        }
    }

    // If allocation still fails (OOM), try emptying the cache and retrying
    if (result == VK_ERROR_OUT_OF_DEVICE_MEMORY || result == VK_ERROR_OUT_OF_HOST_MEMORY) {
        // Use internal version since we already hold the mutex
        empty_cache_impl(device);

        // Retry allocation with exact size
        if (slab_size != size) {
            vkDestroyBuffer(device_alloc.device, buffer, nullptr);
            slab_size = size;
            buffer_info.size = slab_size;
            vkCreateBuffer(device_alloc.device, &buffer_info, nullptr, &buffer);
            vkGetBufferMemoryRequirements(device_alloc.device, buffer, &mem_requirements);
            alloc_info.allocationSize = mem_requirements.size;
        }

        result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);

        // If OOM retry succeeded with relaxed props, update properties
        if (result == VK_SUCCESS && using_relaxed_props) {
            properties = relaxed_props;
        }
    }

    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);

        // Debug: report memory state on OOM
        size_t total_allocated = 0, total_cached = 0, num_blocks = 0;
        for (auto& [ptr, blk] : device_alloc.all_blocks) {
            num_blocks++;
            if (blk->allocated) total_allocated += blk->size;
            else total_cached += blk->size;
        }
        std::ostringstream oss;
        size_t num_allocs = device_alloc.memory_ref_counts.size();
        oss << "VulkanCachingAllocator: Failed to allocate memory, VkResult: "
            << std::to_string(result)
            << " (requested=" << size << " bytes"
            << ", allocated=" << total_allocated
            << ", cached=" << total_cached
            << ", blocks=" << num_blocks
            << ", vkAllocs=" << num_allocs
            << ", reserved=" << device_alloc.stats.reserved_bytes
            << ", device_mem=" << device_alloc.device_memory_size << ")";
        throw std::runtime_error(oss.str());
    }

    // Bind buffer to memory
    result = vkBindBufferMemory(device_alloc.device, buffer, memory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device_alloc.device, memory, nullptr);
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);
        throw std::runtime_error("VulkanCachingAllocator: Failed to bind buffer memory, VkResult: " +
                                 std::to_string(result));
    }

    // Map memory if host-visible
    void* mapped_ptr = nullptr;
    bool is_host_visible = (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT) != 0;
    if (is_host_visible) {
        result = vkMapMemory(device_alloc.device, memory, 0, slab_size, 0, &mapped_ptr);
        if (result != VK_SUCCESS) {
            vkFreeMemory(device_alloc.device, memory, nullptr);
            vkDestroyBuffer(device_alloc.device, buffer, nullptr);
            throw std::runtime_error("VulkanCachingAllocator: Failed to map memory, VkResult: " +
                                     std::to_string(result));
        }
    } else {
        // For device-local only memory, use buffer handle as synthetic address
        // This allows us to track the allocation even without a mapped pointer
        mapped_ptr = reinterpret_cast<void*>(buffer);
    }

    // Create block with full slab size
    auto block = std::make_unique<VulkanBlock>(buffer, memory, mapped_ptr, slab_size, device,
                                                alloc_info.memoryTypeIndex);
    block->is_host_visible = is_host_visible;
    block->owns_memory = true;
    block->memory_offset = 0;
    block->memory_size = slab_size;
    block->base_mapped_ptr = mapped_ptr;
    VulkanBlock* block_ptr = block.get();

    // Add to all_blocks (keyed by mapped pointer or synthetic address)
    device_alloc.all_blocks[mapped_ptr] = std::move(block);

    // Initialize reference count for this memory allocation
    device_alloc.memory_ref_counts[memory] = 1;

    // Update statistics
    device_alloc.stats.reserved_bytes += slab_size;

    // Split the slab: return only the requested size, cache the remainder
    if (slab_size > size + min_split_size_) {
        split_block(block_ptr, size);
    }

    return block_ptr;
}

bool VulkanCachingAllocator::split_block(VulkanBlock* block, size_t size) {
    if (block->size < size + min_split_size_) {
        return false;
    }

    auto& device_alloc = device_allocators_[block->device];

    // Calculate remaining size after split
    size_t remaining_size = block->size - size;

    // Calculate offset for the new block within the shared memory
    size_t new_offset = block->memory_offset + size;

    // Create new buffer for remaining portion (no new memory allocation!)
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = remaining_size;
    buffer_info.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT |
                        VK_BUFFER_USAGE_TRANSFER_SRC_BIT |
                        VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer new_buffer;
    VkResult result = vkCreateBuffer(device_alloc.device, &buffer_info, nullptr, &new_buffer);
    if (result != VK_SUCCESS) {
        return false;  // Can't split, just use the whole block
    }

    // Verify memory requirements are compatible
    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_alloc.device, new_buffer, &mem_requirements);

    // Check alignment - the new offset must be aligned to the buffer's requirements
    if (new_offset % mem_requirements.alignment != 0) {
        // Alignment mismatch - can't sub-allocate at this offset
        vkDestroyBuffer(device_alloc.device, new_buffer, nullptr);
        return false;
    }

    // Bind the new buffer to the SAME memory at the calculated offset
    // This is the key to sub-allocation - no new vkAllocateMemory!
    result = vkBindBufferMemory(device_alloc.device, new_buffer, block->memory, new_offset);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_alloc.device, new_buffer, nullptr);
        return false;
    }

    // Calculate mapped pointer for the new block using pointer arithmetic
    void* new_ptr = nullptr;
    if (block->is_host_visible && block->base_mapped_ptr) {
        new_ptr = static_cast<char*>(block->base_mapped_ptr) + new_offset;
    } else {
        // For device-local memory, use synthetic address
        new_ptr = reinterpret_cast<void*>(new_buffer);
    }

    // Create new block for remaining memory (shares memory with original block)
    auto new_block = std::make_unique<VulkanBlock>(new_buffer, block->memory, new_ptr,
                                                    remaining_size, block->device,
                                                    block->memory_type_index);
    new_block->allocated = false;
    new_block->is_host_visible = block->is_host_visible;
    new_block->owns_memory = false;  // This block does NOT own the memory
    new_block->memory_offset = new_offset;
    new_block->memory_size = block->memory_size;  // Same underlying memory size
    new_block->base_mapped_ptr = block->base_mapped_ptr;  // Same base pointer
    VulkanBlock* new_block_ptr = new_block.get();

    // Increment reference count for the shared memory
    device_alloc.memory_ref_counts[block->memory]++;

    // Add to all_blocks
    device_alloc.all_blocks[new_ptr] = std::move(new_block);

    // Add to free blocks
    device_alloc.free_blocks.insert(new_block_ptr);

    // Update original block size (memory_size stays the same)
    block->size = size;

    // Update statistics - no new reserved_bytes since we're reusing memory!
    device_alloc.stats.num_splits++;
    device_alloc.stats.cached_bytes += remaining_size;
    // Note: reserved_bytes does NOT increase because we're sub-allocating

    return true;
}

size_t VulkanCachingAllocator::try_merge_free_blocks(int device) {
    auto& device_alloc = device_allocators_[device];
    size_t merges = 0;

    if (device_alloc.free_blocks.size() < 2) {
        return 0;
    }

    // Group free blocks by their underlying VkDeviceMemory
    std::unordered_map<VkDeviceMemory, std::vector<VulkanBlock*>> mem_groups;
    for (auto* block : device_alloc.free_blocks) {
        mem_groups[block->memory].push_back(block);
    }

    for (auto& [mem, blocks] : mem_groups) {
        if (blocks.size() < 2) continue;

        // Sort by memory_offset within this VkDeviceMemory
        std::sort(blocks.begin(), blocks.end(),
                  [](const VulkanBlock* a, const VulkanBlock* b) {
                      return a->memory_offset < b->memory_offset;
                  });

        // Scan for adjacent pairs and merge
        for (size_t i = 0; i + 1 < blocks.size(); ) {
            VulkanBlock* lo = blocks[i];
            VulkanBlock* hi = blocks[i + 1];

            // Check adjacency: lo's end must exactly meet hi's start
            if (lo->memory_offset + lo->size != hi->memory_offset) {
                ++i;
                continue;
            }

            // Check compatible memory type
            if (lo->memory_type_index != hi->memory_type_index) {
                ++i;
                continue;
            }

            // Merge hi into lo: extend lo's size, destroy hi's VkBuffer
            device_alloc.free_blocks.erase(lo);
            device_alloc.free_blocks.erase(hi);

            lo->size += hi->size;

            // Destroy hi's VkBuffer
            if (hi->buffer != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_alloc.device, hi->buffer, nullptr);
            }

            // Decrement ref count for the shared memory (hi no longer references it)
            auto ref_it = device_alloc.memory_ref_counts.find(mem);
            if (ref_it != device_alloc.memory_ref_counts.end()) {
                ref_it->second--;
                if (ref_it->second == 0) {
                    device_alloc.memory_ref_counts.erase(ref_it);
                }
            }

            // Remove hi from all_blocks
            device_alloc.all_blocks.erase(hi->mapped_ptr);

            // Re-insert merged lo back into free_blocks
            device_alloc.free_blocks.insert(lo);

            // Update blocks vector for continued scanning
            blocks[i] = lo;
            blocks.erase(blocks.begin() + static_cast<ptrdiff_t>(i + 1));

            merges++;
            device_alloc.stats.num_merges++;
            // Don't increment i — check if lo can merge with next block too
        }
    }

    return merges;
}

size_t VulkanCachingAllocator::round_size(size_t size) const {
    // Just apply base alignment (256 bytes) - no aggressive bucketing.
    // Bucketing was causing too much memory overhead for large models.
    return ((size + alignment_ - 1) / alignment_) * alignment_;
}

void VulkanCachingAllocator::enforce_cache_limit(int device) {
    auto& device_alloc = device_allocators_[device];

    // Release blocks until we're under the limit
    while (device_alloc.stats.cached_bytes > max_cached_memory_ &&
           !device_alloc.free_blocks.empty()) {
        // Release largest block first
        auto it = device_alloc.free_blocks.rbegin();
        VulkanBlock* block = *it;

        // Convert reverse iterator to forward iterator for erase
        auto forward_it = std::next(it).base();
        device_alloc.free_blocks.erase(forward_it);

        release_block(block);
    }
}

void VulkanCachingAllocator::release_block(VulkanBlock* block) {
    auto& device_alloc = device_allocators_[block->device];

    // Remove from free blocks if present
    device_alloc.free_blocks.erase(block);

    // Destroy the buffer (each block has its own VkBuffer even if sharing memory)
    if (block->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_alloc.device, block->buffer, nullptr);
    }

    VkDeviceMemory memory = block->memory;

    // Use reference counting to manage shared memory from split blocks
    bool should_free_memory = false;
    auto ref_it = device_alloc.memory_ref_counts.find(memory);
    if (ref_it != device_alloc.memory_ref_counts.end()) {
        ref_it->second--;
        if (ref_it->second == 0) {
            should_free_memory = true;
            device_alloc.memory_ref_counts.erase(ref_it);
        }
    } else {
        // No ref count entry means this block solely owns its memory
        should_free_memory = true;
    }

    // Free the underlying memory only when the last reference is released
    if (should_free_memory && memory != VK_NULL_HANDLE) {
        // Unmap memory only if it's actually host-visible (mapped)
        if (block->is_host_visible && block->base_mapped_ptr) {
            vkUnmapMemory(device_alloc.device, memory);
        }

        // Free the underlying device memory
        vkFreeMemory(device_alloc.device, memory, nullptr);
    }

    // Subtract from reserved_bytes only for the block's portion
    if (device_alloc.stats.reserved_bytes >= block->size) {
        device_alloc.stats.reserved_bytes -= block->size;
    }

    // Update cached_bytes
    if (device_alloc.stats.cached_bytes >= block->size) {
        device_alloc.stats.cached_bytes -= block->size;
    }

    // Remove from all_blocks
    device_alloc.all_blocks.erase(block->mapped_ptr);
}

uint32_t VulkanCachingAllocator::find_memory_type(VkPhysicalDevice physical_device,
                                                   uint32_t type_filter,
                                                   VkMemoryPropertyFlags properties) {
    VkPhysicalDeviceMemoryProperties mem_properties;
    vkGetPhysicalDeviceMemoryProperties(physical_device, &mem_properties);

    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    // If exact match not found, try without HOST_COHERENT
    VkMemoryPropertyFlags relaxed = properties & ~VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & relaxed) == relaxed) {
            return i;
        }
    }

    // Last resort: just device local
    for (uint32_t i = 0; i < mem_properties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (mem_properties.memoryTypes[i].propertyFlags & VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT)) {
            return i;
        }
    }

    throw std::runtime_error("VulkanCachingAllocator: Failed to find suitable memory type");
}

bool VulkanCachingAllocator::is_memory_host_visible(void* ptr, int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto dev_it = device_allocators_.find(device);
    if (dev_it == device_allocators_.end() || dev_it->second.shutdown) {
        return false;
    }

    auto it = dev_it->second.all_blocks.find(ptr);
    if (it == dev_it->second.all_blocks.end()) {
        return false;
    }

    return it->second->is_host_visible;
}

void* VulkanCachingAllocator::get_mapped_ptr(void* ptr, int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto dev_it = device_allocators_.find(device);
    if (dev_it == device_allocators_.end() || dev_it->second.shutdown) {
        return nullptr;
    }

    auto it = dev_it->second.all_blocks.find(ptr);
    if (it == dev_it->second.all_blocks.end()) {
        return nullptr;
    }

    // Return actual mapped pointer only if host-visible
    if (it->second->is_host_visible) {
        return it->second->mapped_ptr;
    }
    return nullptr;
}

VulkanCachingAllocator::StagingBuffer* VulkanCachingAllocator::acquire_staging_buffer(size_t size, int device) {
    std::lock_guard<std::mutex> lock(mutex_);

    auto it = device_allocators_.find(device);
    if (it == device_allocators_.end() || !it->second.initialized) {
        throw std::runtime_error("VulkanCachingAllocator: Device not initialized for staging buffer");
    }

    auto& device_alloc = it->second;

    // Try to find an existing free staging buffer that's large enough
    for (auto& staging : device_alloc.staging_pool) {
        if (!staging->in_use && staging->size >= size) {
            staging->in_use = true;
            return staging.get();
        }
    }

    // GC: if the staging pool has grown too large, evict unused buffers.
    // Cap at 64 entries to prevent unbounded growth in long-running servers.
    constexpr size_t MAX_STAGING_POOL_SIZE = 64;
    if (device_alloc.staging_pool.size() >= MAX_STAGING_POOL_SIZE) {
        auto& pool = device_alloc.staging_pool;
        auto new_end = std::remove_if(pool.begin(), pool.end(),
            [&](const std::unique_ptr<StagingBuffer>& sb) {
                if (!sb->in_use) {
                    vkDestroyBuffer(device_alloc.device, sb->buffer, nullptr);
                    vkFreeMemory(device_alloc.device, sb->memory, nullptr);
                    return true;
                }
                return false;
            });
        pool.erase(new_end, pool.end());
    }

    // No suitable buffer found, create a new one
    StagingBuffer* new_staging = create_staging_buffer(size, device);
    new_staging->in_use = true;
    return new_staging;
}

void VulkanCachingAllocator::release_staging_buffer(StagingBuffer* staging, int device) {
    std::lock_guard<std::mutex> lock(mutex_);

    if (!staging) return;

    staging->in_use = false;
}

VulkanCachingAllocator::StagingBuffer* VulkanCachingAllocator::create_staging_buffer(size_t size, int device) {
    auto& device_alloc = device_allocators_[device];

    // Round up to alignment
    size = round_size(size);

    // Create staging buffer with HOST_VISIBLE | HOST_COHERENT memory
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
    buffer_info.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    buffer_info.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VkBuffer buffer;
    VkResult result = vkCreateBuffer(device_alloc.device, &buffer_info, nullptr, &buffer);
    if (result != VK_SUCCESS) {
        throw std::runtime_error("VulkanCachingAllocator: Failed to create staging buffer");
    }

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_alloc.device, buffer, &mem_requirements);

    // Find host-visible memory type
    VkMemoryPropertyFlags properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT |
                                       VK_MEMORY_PROPERTY_HOST_COHERENT_BIT;

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = find_memory_type(device_alloc.physical_device,
                                                   mem_requirements.memoryTypeBits,
                                                   properties);

    VkDeviceMemory memory;
    result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);
        throw std::runtime_error("VulkanCachingAllocator: Failed to allocate staging buffer memory");
    }

    result = vkBindBufferMemory(device_alloc.device, buffer, memory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device_alloc.device, memory, nullptr);
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);
        throw std::runtime_error("VulkanCachingAllocator: Failed to bind staging buffer memory");
    }

    void* mapped_ptr = nullptr;
    result = vkMapMemory(device_alloc.device, memory, 0, size, 0, &mapped_ptr);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device_alloc.device, memory, nullptr);
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);
        throw std::runtime_error("VulkanCachingAllocator: Failed to map staging buffer memory");
    }

    auto staging = std::make_unique<StagingBuffer>();
    staging->buffer = buffer;
    staging->memory = memory;
    staging->mapped_ptr = mapped_ptr;
    staging->size = size;
    staging->in_use = false;

    StagingBuffer* staging_ptr = staging.get();
    device_alloc.staging_pool.push_back(std::move(staging));

    return staging_ptr;
}

void VulkanCachingAllocator::destroy_staging_buffer(StagingBuffer* staging, int device) {
    auto& device_alloc = device_allocators_[device];

    if (staging->mapped_ptr) {
        vkUnmapMemory(device_alloc.device, staging->memory);
    }
    if (staging->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_alloc.device, staging->buffer, nullptr);
    }
    if (staging->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_alloc.device, staging->memory, nullptr);
    }
}

} // namespace backend
} // namespace tenzor
