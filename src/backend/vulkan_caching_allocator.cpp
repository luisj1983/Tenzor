#include "tenzor/backend/vulkan_caching_allocator.hpp"

#include <algorithm>
#include <stdexcept>
#include <sstream>
#include <cstring>

namespace tenzor {
namespace backend {

// Default configuration
constexpr size_t DEFAULT_ALIGNMENT = 256;  // Vulkan-optimized alignment
constexpr size_t DEFAULT_MIN_SPLIT_SIZE = 256;

VulkanCachingAllocator::VulkanCachingAllocator()
    : alignment_(DEFAULT_ALIGNMENT),
      max_cached_memory_(0),  // Unlimited by default
      min_split_size_(DEFAULT_MIN_SPLIT_SIZE) {
}

VulkanCachingAllocator::~VulkanCachingAllocator() {
    // Release all cached memory
    empty_cache(-1);
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

    // Try to find a suitable block in cache
    VulkanBlock* block = try_allocate_from_cache(size, device, usage, properties);

    bool from_cache = (block != nullptr);

    if (!block) {
        // No suitable cached block, allocate new one
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

    auto& device_alloc = device_allocators_[device];
    device_alloc.stats.num_frees++;

    // Find the block
    auto it = device_alloc.all_blocks.find(ptr);
    if (it == device_alloc.all_blocks.end()) {
        throw std::runtime_error("VulkanCachingAllocator: Attempted to free pointer not allocated by this allocator");
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
    device_alloc.stats.cached_bytes += block->size;

    // Add to free blocks set
    device_alloc.free_blocks.insert(block);

    // Enforce cache limit if set
    if (max_cached_memory_ > 0) {
        enforce_cache_limit(device);
    }
}

void VulkanCachingAllocator::empty_cache(int device) {
    std::lock_guard<std::mutex> lock(mutex_);

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

VkBuffer VulkanCachingAllocator::get_buffer(void* ptr, int device) const {
    std::lock_guard<std::mutex> lock(mutex_);

    auto dev_it = device_allocators_.find(device);
    if (dev_it == device_allocators_.end()) {
        throw std::runtime_error("VulkanCachingAllocator: Device not found");
    }

    auto it = dev_it->second.all_blocks.find(ptr);
    if (it == dev_it->second.all_blocks.end()) {
        throw std::runtime_error("VulkanCachingAllocator: Pointer not found");
    }

    return it->second->buffer;
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
                                                              VkMemoryPropertyFlags /*properties*/) {
    auto& device_alloc = device_allocators_[device];

    // Find smallest free block that fits (best-fit)
    // Create a temporary block for comparison
    VulkanBlock search_block(VK_NULL_HANDLE, VK_NULL_HANDLE, nullptr, size, device, 0);
    auto it = device_alloc.free_blocks.lower_bound(&search_block);

    if (it != device_alloc.free_blocks.end()) {
        VulkanBlock* block = *it;

        // Remove from free blocks
        device_alloc.free_blocks.erase(it);

        // Try to split if block is too large
        if (block->size >= size + min_split_size_) {
            split_block(block, size);
        }

        return block;
    }

    return nullptr;
}

VulkanBlock* VulkanCachingAllocator::allocate_new_block(size_t size, int device,
                                                         VkBufferUsageFlags usage,
                                                         VkMemoryPropertyFlags properties) {
    auto& device_alloc = device_allocators_[device];

    // Create buffer
    VkBufferCreateInfo buffer_info{};
    buffer_info.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    buffer_info.size = size;
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
    result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);

    // If allocation fails with HOST_VISIBLE requested, retry with DEVICE_LOCAL only
    // This handles discrete GPUs where BAR (host-visible VRAM) is limited
    if (result != VK_SUCCESS && (properties & VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT)) {
        // Retry with just DEVICE_LOCAL
        VkMemoryPropertyFlags relaxed_props = VK_MEMORY_PROPERTY_DEVICE_LOCAL_BIT;
        alloc_info.memoryTypeIndex = find_memory_type(device_alloc.physical_device,
                                                       mem_requirements.memoryTypeBits,
                                                       relaxed_props);
        result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &memory);
        if (result == VK_SUCCESS) {
            // Successfully allocated with device-local only, clear HOST_VISIBLE from properties
            properties = relaxed_props;
        }
    }

    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_alloc.device, buffer, nullptr);
        throw std::runtime_error("VulkanCachingAllocator: Failed to allocate memory, VkResult: " +
                                 std::to_string(result));
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
        result = vkMapMemory(device_alloc.device, memory, 0, size, 0, &mapped_ptr);
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

    // Create block
    auto block = std::make_unique<VulkanBlock>(buffer, memory, mapped_ptr, size, device,
                                                alloc_info.memoryTypeIndex);
    block->is_host_visible = is_host_visible;
    VulkanBlock* block_ptr = block.get();

    // Add to all_blocks (keyed by mapped pointer or synthetic address)
    device_alloc.all_blocks[mapped_ptr] = std::move(block);

    // Update statistics
    device_alloc.stats.reserved_bytes += size;

    return block_ptr;
}

bool VulkanCachingAllocator::split_block(VulkanBlock* block, size_t size) {
    if (block->size < size + min_split_size_) {
        return false;
    }

    auto& device_alloc = device_allocators_[block->device];

    // Calculate split size
    size_t remaining_size = block->size - size;

    // Create new buffer and memory for remaining block
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

    VkMemoryRequirements mem_requirements;
    vkGetBufferMemoryRequirements(device_alloc.device, new_buffer, &mem_requirements);

    VkMemoryAllocateInfo alloc_info{};
    alloc_info.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    alloc_info.allocationSize = mem_requirements.size;
    alloc_info.memoryTypeIndex = block->memory_type_index;

    VkDeviceMemory new_memory;
    result = vkAllocateMemory(device_alloc.device, &alloc_info, nullptr, &new_memory);
    if (result != VK_SUCCESS) {
        vkDestroyBuffer(device_alloc.device, new_buffer, nullptr);
        return false;
    }

    result = vkBindBufferMemory(device_alloc.device, new_buffer, new_memory, 0);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device_alloc.device, new_memory, nullptr);
        vkDestroyBuffer(device_alloc.device, new_buffer, nullptr);
        return false;
    }

    // Map the new memory
    void* new_ptr = nullptr;
    result = vkMapMemory(device_alloc.device, new_memory, 0, remaining_size, 0, &new_ptr);
    if (result != VK_SUCCESS) {
        vkFreeMemory(device_alloc.device, new_memory, nullptr);
        vkDestroyBuffer(device_alloc.device, new_buffer, nullptr);
        return false;
    }

    // Create new block for remaining memory
    auto new_block = std::make_unique<VulkanBlock>(new_buffer, new_memory, new_ptr,
                                                    remaining_size, block->device,
                                                    block->memory_type_index);
    new_block->allocated = false;
    VulkanBlock* new_block_ptr = new_block.get();

    // Add to all_blocks
    device_alloc.all_blocks[new_ptr] = std::move(new_block);

    // Add to free blocks
    device_alloc.free_blocks.insert(new_block_ptr);

    // Update original block size
    block->size = size;

    // Update statistics
    device_alloc.stats.num_splits++;
    device_alloc.stats.cached_bytes += remaining_size;
    device_alloc.stats.reserved_bytes += remaining_size;

    return true;
}

size_t VulkanCachingAllocator::round_size(size_t size) const {
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

    // Unmap memory if mapped
    if (block->mapped_ptr) {
        vkUnmapMemory(device_alloc.device, block->memory);
    }

    // Free Vulkan resources
    if (block->buffer != VK_NULL_HANDLE) {
        vkDestroyBuffer(device_alloc.device, block->buffer, nullptr);
    }
    if (block->memory != VK_NULL_HANDLE) {
        vkFreeMemory(device_alloc.device, block->memory, nullptr);
    }

    // Update statistics
    if (device_alloc.stats.reserved_bytes >= block->size) {
        device_alloc.stats.reserved_bytes -= block->size;
    }
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

} // namespace backend
} // namespace tenzor
