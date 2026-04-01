/**
 * @file vulkan_utils.hpp
 * @brief Vulkan utility functions and helper classes
 *
 * Provides utility functions for Vulkan compute operations including
 * error checking, pipeline creation, descriptor management, and command buffers.
 */

#pragma once

#include <vulkan/vulkan.h>
#include <vector>
#include <string>
#include <memory>
#include <stdexcept>
#include <fstream>
#include <iostream>
#include <unordered_map>
#include <cstdio>

namespace tenzor {
namespace vulkan {

/**
 * @brief Exception class for Vulkan errors
 */
class VulkanError : public std::runtime_error {
public:
    VulkanError(const std::string& message, VkResult result)
        : std::runtime_error(message + " (VkResult: " + std::to_string(result) + ")"),
          result_(result) {}

    VkResult result() const { return result_; }

private:
    VkResult result_;
};

/**
 * @brief Check Vulkan result and throw exception on failure
 */
inline void checkVk(VkResult result, const std::string& operation) {
    if (result != VK_SUCCESS) {
        throw VulkanError("Vulkan operation failed: " + operation, result);
    }
}

/**
 * @brief Load compiled SPIR-V shader from file
 */
inline std::vector<uint32_t> loadShader(const std::string& filename) {
    std::ifstream file(filename, std::ios::ate | std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filename);
    }

    size_t fileSize = static_cast<size_t>(file.tellg());
    if (fileSize < 20) {  // SPIR-V minimum: 5-word header (20 bytes)
        throw std::runtime_error("Shader file too small to be valid SPIR-V: " + filename);
    }
    if (fileSize % sizeof(uint32_t) != 0) {
        throw std::runtime_error("Shader file size not aligned to 4 bytes: " + filename);
    }

    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    // Validate SPIR-V magic number
    constexpr uint32_t SPIRV_MAGIC = 0x07230203;
    if (buffer[0] != SPIRV_MAGIC) {
        char hex[9];
        snprintf(hex, sizeof(hex), "%08x", buffer[0]);
        throw std::runtime_error(
            "Invalid SPIR-V magic number in shader: " + filename +
            " (expected 0x07230203, got 0x" + hex + ")");
    }

    return buffer;
}

/**
 * @brief Reflect push constant size from compiled SPIR-V binary
 *
 * Parses SPIR-V instructions to find the push constant block and computes
 * its total size from member offset decorations and type definitions.
 * Returns 0 if the shader has no push constants.
 *
 * The size is rounded up to 4-byte alignment per Vulkan requirements.
 *
 * @param spirv The compiled SPIR-V binary (vector of uint32_t words)
 * @return Total push constant size in bytes, or 0 if none
 */
inline uint32_t reflectPushConstantSize(const std::vector<uint32_t>& spirv) {
    if (spirv.size() < 5) {
        throw std::runtime_error("reflectPushConstantSize: SPIR-V binary too small ("
                                 + std::to_string(spirv.size()) + " words, need >= 5)");
    }
    if (spirv[0] != 0x07230203) {
        throw std::runtime_error("reflectPushConstantSize: invalid SPIR-V magic number 0x"
                                 + ([&]{
                                     char buf[16];
                                     snprintf(buf, sizeof(buf), "%08X", spirv[0]);
                                     return std::string(buf);
                                 })()
                                 + " (expected 0x07230203)");
    }

    // SPIR-V opcode constants
    constexpr uint32_t SpvOpTypeFloat       = 22;
    constexpr uint32_t SpvOpTypeInt         = 21;
    constexpr uint32_t SpvOpTypeVector      = 23;
    constexpr uint32_t SpvOpTypeMatrix      = 24;
    constexpr uint32_t SpvOpTypeArray       = 28;
    constexpr uint32_t SpvOpTypeStruct      = 30;
    constexpr uint32_t SpvOpTypePointer     = 32;
    constexpr uint32_t SpvOpConstant        = 43;
    constexpr uint32_t SpvOpVariable        = 59;
    constexpr uint32_t SpvOpMemberDecorate  = 72;
    constexpr uint32_t SpvDecorationOffset  = 35;
    constexpr uint32_t SpvStorageClassPushConstant = 9;

    // Type size info: maps SPIR-V result ID to its size in bytes
    std::unordered_map<uint32_t, uint32_t> type_sizes;
    // Constant values: maps SPIR-V result ID to its uint32_t value
    std::unordered_map<uint32_t, uint32_t> constants;
    // Struct member type IDs: maps struct ID to vector of member type IDs
    std::unordered_map<uint32_t, std::vector<uint32_t>> struct_members;
    // Pointer types: maps pointer result ID to {storage_class, pointee_type_id}
    std::unordered_map<uint32_t, std::pair<uint32_t, uint32_t>> pointer_types;
    // Member offset decorations: maps {struct_id, member_index} to offset
    std::unordered_map<uint64_t, uint32_t> member_offsets;
    // The struct type ID used for push constants (0 if not found)
    uint32_t push_constant_struct_id = 0;

    // Single pass through SPIR-V instructions (skip 5-word header)
    size_t i = 5;
    while (i < spirv.size()) {
        uint32_t word = spirv[i];
        uint32_t word_count = word >> 16;
        uint32_t opcode = word & 0xFFFF;

        if (word_count == 0 || i + word_count > spirv.size()) {
            std::fprintf(stderr, "[tenzor] warning: malformed SPIR-V instruction at word %zu "
                                 "(word_count=%u, remaining=%zu)\n",
                         i, word_count, spirv.size() - i);
            break;
        }

        switch (opcode) {
            case SpvOpTypeFloat: {
                // OpTypeFloat result_id bit_width
                if (word_count >= 3) {
                    uint32_t id = spirv[i + 1];
                    uint32_t bit_width = spirv[i + 2];
                    type_sizes[id] = bit_width / 8;
                }
                break;
            }
            case SpvOpTypeInt: {
                // OpTypeInt result_id bit_width signedness
                if (word_count >= 3) {
                    uint32_t id = spirv[i + 1];
                    uint32_t bit_width = spirv[i + 2];
                    type_sizes[id] = bit_width / 8;
                }
                break;
            }
            case SpvOpTypeVector: {
                // OpTypeVector result_id component_type count
                if (word_count >= 4) {
                    uint32_t id = spirv[i + 1];
                    uint32_t component_type = spirv[i + 2];
                    uint32_t count = spirv[i + 3];
                    auto it = type_sizes.find(component_type);
                    if (it != type_sizes.end()) {
                        type_sizes[id] = it->second * count;
                    }
                }
                break;
            }
            case SpvOpTypeMatrix: {
                // OpTypeMatrix result_id column_type column_count
                if (word_count >= 4) {
                    uint32_t id = spirv[i + 1];
                    uint32_t column_type = spirv[i + 2];
                    uint32_t column_count = spirv[i + 3];
                    auto it = type_sizes.find(column_type);
                    if (it != type_sizes.end()) {
                        type_sizes[id] = it->second * column_count;
                    }
                }
                break;
            }
            case SpvOpConstant: {
                // OpConstant result_type result_id value...
                if (word_count >= 4) {
                    uint32_t id = spirv[i + 2];
                    uint32_t value = spirv[i + 3];
                    constants[id] = value;
                }
                break;
            }
            case SpvOpTypeArray: {
                // OpTypeArray result_id element_type length_id
                if (word_count >= 4) {
                    uint32_t id = spirv[i + 1];
                    uint32_t element_type = spirv[i + 2];
                    uint32_t length_id = spirv[i + 3];
                    auto elem_it = type_sizes.find(element_type);
                    auto len_it = constants.find(length_id);
                    if (elem_it != type_sizes.end() && len_it != constants.end()) {
                        type_sizes[id] = elem_it->second * len_it->second;
                    }
                }
                break;
            }
            case SpvOpTypeStruct: {
                // OpTypeStruct result_id member_type_0 member_type_1 ...
                if (word_count >= 2) {
                    uint32_t id = spirv[i + 1];
                    std::vector<uint32_t> members;
                    for (uint32_t m = 2; m < word_count; ++m) {
                        members.push_back(spirv[i + m]);
                    }
                    struct_members[id] = std::move(members);
                }
                break;
            }
            case SpvOpTypePointer: {
                // OpTypePointer result_id storage_class type
                if (word_count >= 4) {
                    uint32_t id = spirv[i + 1];
                    uint32_t storage_class = spirv[i + 2];
                    uint32_t type = spirv[i + 3];
                    pointer_types[id] = {storage_class, type};
                }
                break;
            }
            case SpvOpVariable: {
                // OpVariable result_type result_id storage_class [initializer]
                if (word_count >= 4) {
                    uint32_t result_type = spirv[i + 1];
                    uint32_t storage_class = spirv[i + 3];
                    if (storage_class == SpvStorageClassPushConstant) {
                        // Found push constant variable - resolve its struct type
                        auto ptr_it = pointer_types.find(result_type);
                        if (ptr_it != pointer_types.end()) {
                            push_constant_struct_id = ptr_it->second.second;
                        }
                    }
                }
                break;
            }
            case SpvOpMemberDecorate: {
                // OpMemberDecorate struct_type member decoration [value]
                if (word_count >= 5) {
                    uint32_t struct_type = spirv[i + 1];
                    uint32_t member = spirv[i + 2];
                    uint32_t decoration = spirv[i + 3];
                    if (decoration == SpvDecorationOffset) {
                        uint32_t offset = spirv[i + 4];
                        uint64_t key = (static_cast<uint64_t>(struct_type) << 32) | member;
                        member_offsets[key] = offset;
                    }
                }
                break;
            }
            default:
                break;
        }

        i += word_count;
    }

    // No push constant variable found
    if (push_constant_struct_id == 0) {
        return 0;
    }

    // Find the struct members
    auto struct_it = struct_members.find(push_constant_struct_id);
    if (struct_it == struct_members.end()) {
        return 0;
    }

    const auto& members = struct_it->second;
    uint32_t max_extent = 0;

    for (uint32_t m = 0; m < members.size(); ++m) {
        uint64_t key = (static_cast<uint64_t>(push_constant_struct_id) << 32) | m;
        auto off_it = member_offsets.find(key);
        if (off_it == member_offsets.end()) {
            continue;  // No offset decoration for this member
        }

        uint32_t offset = off_it->second;
        uint32_t member_size = 0;

        auto size_it = type_sizes.find(members[m]);
        if (size_it != type_sizes.end()) {
            member_size = size_it->second;
        } else {
            // Check if it's a nested struct
            auto nested_struct_it = struct_members.find(members[m]);
            if (nested_struct_it != struct_members.end()) {
                // Compute nested struct size recursively (find max extent)
                uint32_t nested_max = 0;
                for (uint32_t nm = 0; nm < nested_struct_it->second.size(); ++nm) {
                    uint64_t nkey = (static_cast<uint64_t>(members[m]) << 32) | nm;
                    auto noff_it = member_offsets.find(nkey);
                    auto nsize_it = type_sizes.find(nested_struct_it->second[nm]);
                    if (noff_it != member_offsets.end() && nsize_it != type_sizes.end()) {
                        nested_max = std::max(nested_max, noff_it->second + nsize_it->second);
                    }
                }
                member_size = nested_max;
            }
        }

        max_extent = std::max(max_extent, offset + member_size);
    }

    // Round up to 4-byte alignment (Vulkan push constant requirement)
    return (max_extent + 3) & ~3u;
}

/**
 * @brief RAII wrapper for Vulkan buffer
 */
class VulkanBuffer {
public:
    VulkanBuffer() = default;

    VulkanBuffer(VkDevice device, VkPhysicalDevice physicalDevice,
                 VkDeviceSize size, VkBufferUsageFlags usage,
                 VkMemoryPropertyFlags properties)
        : device_(device), size_(size) {

        // Create buffer
        VkBufferCreateInfo bufferInfo{};
        bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
        bufferInfo.size = size;
        bufferInfo.usage = usage;
        bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

        checkVk(vkCreateBuffer(device, &bufferInfo, nullptr, &buffer_),
                "Failed to create buffer");

        // Allocate memory
        VkMemoryRequirements memRequirements;
        vkGetBufferMemoryRequirements(device, buffer_, &memRequirements);

        VkMemoryAllocateInfo allocInfo{};
        allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
        allocInfo.allocationSize = memRequirements.size;
        allocInfo.memoryTypeIndex = findMemoryType(physicalDevice,
                                                   memRequirements.memoryTypeBits,
                                                   properties);

        VkResult allocResult = vkAllocateMemory(device, &allocInfo, nullptr, &memory_);
        if (allocResult != VK_SUCCESS) {
            vkDestroyBuffer(device, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
            checkVk(allocResult, "Failed to allocate buffer memory");
        }

        VkResult bindResult = vkBindBufferMemory(device, buffer_, memory_, 0);
        if (bindResult != VK_SUCCESS) {
            vkFreeMemory(device, memory_, nullptr);
            memory_ = VK_NULL_HANDLE;
            vkDestroyBuffer(device, buffer_, nullptr);
            buffer_ = VK_NULL_HANDLE;
            checkVk(bindResult, "Failed to bind buffer memory");
        }
    }

    ~VulkanBuffer() {
        if (buffer_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(device_, buffer_, nullptr);
        }
        if (memory_ != VK_NULL_HANDLE) {
            vkFreeMemory(device_, memory_, nullptr);
        }
    }

    // Disable copy, enable move
    VulkanBuffer(const VulkanBuffer&) = delete;
    VulkanBuffer& operator=(const VulkanBuffer&) = delete;

    VulkanBuffer(VulkanBuffer&& other) noexcept
        : device_(other.device_), buffer_(other.buffer_),
          memory_(other.memory_), size_(other.size_) {
        other.buffer_ = VK_NULL_HANDLE;
        other.memory_ = VK_NULL_HANDLE;
    }

    VulkanBuffer& operator=(VulkanBuffer&& other) noexcept {
        if (this != &other) {
            if (buffer_ != VK_NULL_HANDLE) {
                vkDestroyBuffer(device_, buffer_, nullptr);
            }
            if (memory_ != VK_NULL_HANDLE) {
                vkFreeMemory(device_, memory_, nullptr);
            }

            device_ = other.device_;
            buffer_ = other.buffer_;
            memory_ = other.memory_;
            size_ = other.size_;

            other.buffer_ = VK_NULL_HANDLE;
            other.memory_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    VkBuffer buffer() const { return buffer_; }
    VkDeviceMemory memory() const { return memory_; }
    VkDeviceSize size() const { return size_; }

    void* map() {
        void* data;
        checkVk(vkMapMemory(device_, memory_, 0, size_, 0, &data),
                "Failed to map buffer memory");
        return data;
    }

    void unmap() {
        vkUnmapMemory(device_, memory_);
    }

private:
    static uint32_t findMemoryType(VkPhysicalDevice physicalDevice,
                                   uint32_t typeFilter,
                                   VkMemoryPropertyFlags properties) {
        VkPhysicalDeviceMemoryProperties memProperties;
        vkGetPhysicalDeviceMemoryProperties(physicalDevice, &memProperties);

        for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
            if ((typeFilter & (1 << i)) &&
                (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
                return i;
            }
        }

        throw std::runtime_error("Failed to find suitable memory type");
    }

    VkDevice device_ = VK_NULL_HANDLE;
    VkBuffer buffer_ = VK_NULL_HANDLE;
    VkDeviceMemory memory_ = VK_NULL_HANDLE;
    VkDeviceSize size_ = 0;
};

/**
 * @brief RAII wrapper for Vulkan descriptor pool
 */
class DescriptorPool {
    static constexpr uint32_t MAX_DESCRIPTOR_POOL_SETS = 65536;

public:
    DescriptorPool(VkDevice device, uint32_t maxSets)
        : device_(device), max_sets_(maxSets) {

        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = maxSets * 8; // Up to 8 buffers per set

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = maxSets;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        checkVk(vkCreateDescriptorPool(device, &poolInfo, nullptr, &pool_),
                "Failed to create descriptor pool");
    }

    ~DescriptorPool() {
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, pool_, nullptr);
        }
    }

    DescriptorPool(const DescriptorPool&) = delete;
    DescriptorPool& operator=(const DescriptorPool&) = delete;

    VkDescriptorPool pool() const { return pool_; }

    /// Track a descriptor set allocation and warn if pool usage exceeds 80%.
    void trackAllocation() {
        allocated_sets_++;
        if (!warning_issued_ && max_sets_ > 0) {
            uint32_t threshold = max_sets_ * 4 / 5;  // 80%
            if (allocated_sets_ >= threshold) {
                std::cerr << "[Vulkan WARNING] Descriptor pool usage at "
                          << allocated_sets_ << "/" << max_sets_
                          << " (" << (allocated_sets_ * 100 / max_sets_)
                          << "%). Consider calling synchronize() to reclaim sets.\n";
                warning_issued_ = true;
            }
        }
    }

    /// Return current allocation count.
    uint32_t allocated_sets() const { return allocated_sets_; }

    /// Return pool capacity (maxSets).
    uint32_t max_sets() const { return max_sets_; }

    /**
     * @brief Reset the descriptor pool, freeing all allocated descriptor sets
     *
     * This should only be called when no descriptor sets are in use (i.e., after
     * synchronization). All previously allocated descriptor sets become invalid.
     */
    void reset() {
        if (pool_ != VK_NULL_HANDLE) {
            vkResetDescriptorPool(device_, pool_, 0);
            allocated_sets_ = 0;
            warning_issued_ = false;
        }
    }

    /**
     * @brief Grow the descriptor pool by destroying and recreating with larger capacity.
     *
     * Called when VK_ERROR_OUT_OF_POOL_MEMORY or VK_ERROR_FRAGMENTED_POOL is
     * encountered and a simple reset is not sufficient (e.g., live descriptor sets
     * still in use). The new pool has 2x the previous maxSets capacity.
     *
     * @warning All previously allocated descriptor sets become invalid. Callers must
     *          ensure no in-flight command buffers reference old descriptor sets.
     */
    void grow() {
        uint32_t new_max = max_sets_ * 2;
        if (new_max > MAX_DESCRIPTOR_POOL_SETS) {
            new_max = MAX_DESCRIPTOR_POOL_SETS;
            if (max_sets_ >= MAX_DESCRIPTOR_POOL_SETS) {
                std::cerr << "[Vulkan WARNING] Descriptor pool reached maximum capacity ("
                          << MAX_DESCRIPTOR_POOL_SETS << " sets)\n";
                return;
            }
        }
        std::cerr << "[Vulkan WARNING] Descriptor pool exhausted/fragmented (capacity="
                  << max_sets_ << "). Growing to " << new_max << " sets.\n";

        // Destroy old pool
        if (pool_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorPool(device_, pool_, nullptr);
            pool_ = VK_NULL_HANDLE;
        }

        // Create new, larger pool
        VkDescriptorPoolSize poolSize{};
        poolSize.type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER;
        poolSize.descriptorCount = new_max * 8;  // Up to 8 buffers per set

        VkDescriptorPoolCreateInfo poolInfo{};
        poolInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        poolInfo.poolSizeCount = 1;
        poolInfo.pPoolSizes = &poolSize;
        poolInfo.maxSets = new_max;
        poolInfo.flags = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;

        checkVk(vkCreateDescriptorPool(device_, &poolInfo, nullptr, &pool_),
                "Failed to create grown descriptor pool");

        max_sets_ = new_max;
        allocated_sets_ = 0;
        warning_issued_ = false;
    }

private:
    VkDevice device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
    uint32_t max_sets_ = 0;
    uint32_t allocated_sets_ = 0;
    bool warning_issued_ = false;
};

/**
 * @brief RAII wrapper for compute pipeline
 */
class ComputePipeline {
public:
    ComputePipeline() = default;

    ComputePipeline(VkDevice device, const std::vector<uint32_t>& shaderCode,
                   const std::vector<VkDescriptorSetLayoutBinding>& bindings,
                   const std::vector<VkPushConstantRange>& pushConstants = {},
                   VkPipelineCache pipelineCache = VK_NULL_HANDLE,
                   const VkSpecializationInfo* specializationInfo = nullptr)
        : device_(device) {

        // Create shader module
        VkShaderModuleCreateInfo shaderModuleInfo{};
        shaderModuleInfo.sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO;
        shaderModuleInfo.codeSize = shaderCode.size() * sizeof(uint32_t);
        shaderModuleInfo.pCode = shaderCode.data();

        VkShaderModule shaderModule;
        checkVk(vkCreateShaderModule(device, &shaderModuleInfo, nullptr, &shaderModule),
                "Failed to create shader module");

        // Create descriptor set layout
        VkDescriptorSetLayoutCreateInfo layoutInfo{};
        layoutInfo.sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO;
        layoutInfo.bindingCount = static_cast<uint32_t>(bindings.size());
        layoutInfo.pBindings = bindings.data();

        checkVk(vkCreateDescriptorSetLayout(device, &layoutInfo, nullptr, &descriptorSetLayout_),
                "Failed to create descriptor set layout");

        // Create pipeline layout
        VkPipelineLayoutCreateInfo pipelineLayoutInfo{};
        pipelineLayoutInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO;
        pipelineLayoutInfo.setLayoutCount = 1;
        pipelineLayoutInfo.pSetLayouts = &descriptorSetLayout_;
        pipelineLayoutInfo.pushConstantRangeCount = static_cast<uint32_t>(pushConstants.size());
        pipelineLayoutInfo.pPushConstantRanges = pushConstants.empty() ? nullptr : pushConstants.data();

        checkVk(vkCreatePipelineLayout(device, &pipelineLayoutInfo, nullptr, &pipelineLayout_),
                "Failed to create pipeline layout");

        // Create compute pipeline
        VkPipelineShaderStageCreateInfo shaderStageInfo{};
        shaderStageInfo.sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO;
        shaderStageInfo.stage = VK_SHADER_STAGE_COMPUTE_BIT;
        shaderStageInfo.module = shaderModule;
        shaderStageInfo.pName = "main";
        shaderStageInfo.pSpecializationInfo = specializationInfo;

        VkComputePipelineCreateInfo pipelineInfo{};
        pipelineInfo.sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO;
        pipelineInfo.stage = shaderStageInfo;
        pipelineInfo.layout = pipelineLayout_;

        checkVk(vkCreateComputePipelines(device, pipelineCache, 1, &pipelineInfo,
                                        nullptr, &pipeline_),
                "Failed to create compute pipeline");

        vkDestroyShaderModule(device, shaderModule, nullptr);
    }

    ~ComputePipeline() {
        if (pipeline_ != VK_NULL_HANDLE) {
            vkDestroyPipeline(device_, pipeline_, nullptr);
        }
        if (pipelineLayout_ != VK_NULL_HANDLE) {
            vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
        }
        if (descriptorSetLayout_ != VK_NULL_HANDLE) {
            vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
        }
    }

    ComputePipeline(const ComputePipeline&) = delete;
    ComputePipeline& operator=(const ComputePipeline&) = delete;

    ComputePipeline(ComputePipeline&& other) noexcept
        : device_(other.device_), pipeline_(other.pipeline_),
          pipelineLayout_(other.pipelineLayout_),
          descriptorSetLayout_(other.descriptorSetLayout_) {
        other.pipeline_ = VK_NULL_HANDLE;
        other.pipelineLayout_ = VK_NULL_HANDLE;
        other.descriptorSetLayout_ = VK_NULL_HANDLE;
    }

    ComputePipeline& operator=(ComputePipeline&& other) noexcept {
        if (this != &other) {
            if (pipeline_ != VK_NULL_HANDLE) {
                vkDestroyPipeline(device_, pipeline_, nullptr);
            }
            if (pipelineLayout_ != VK_NULL_HANDLE) {
                vkDestroyPipelineLayout(device_, pipelineLayout_, nullptr);
            }
            if (descriptorSetLayout_ != VK_NULL_HANDLE) {
                vkDestroyDescriptorSetLayout(device_, descriptorSetLayout_, nullptr);
            }

            device_ = other.device_;
            pipeline_ = other.pipeline_;
            pipelineLayout_ = other.pipelineLayout_;
            descriptorSetLayout_ = other.descriptorSetLayout_;

            other.pipeline_ = VK_NULL_HANDLE;
            other.pipelineLayout_ = VK_NULL_HANDLE;
            other.descriptorSetLayout_ = VK_NULL_HANDLE;
        }
        return *this;
    }

    VkPipeline pipeline() const { return pipeline_; }
    VkPipelineLayout layout() const { return pipelineLayout_; }
    VkDescriptorSetLayout descriptorSetLayout() const { return descriptorSetLayout_; }

private:
    VkDevice device_ = VK_NULL_HANDLE;
    VkPipeline pipeline_ = VK_NULL_HANDLE;
    VkPipelineLayout pipelineLayout_ = VK_NULL_HANDLE;
    VkDescriptorSetLayout descriptorSetLayout_ = VK_NULL_HANDLE;
};

/**
 * @brief Query optimal workgroup size for a Vulkan physical device.
 *
 * Returns the largest power-of-2 workgroup size that fits within both
 * maxComputeWorkGroupSize[0] and maxComputeWorkGroupInvocations, capped at 1024.
 * Power-of-2 sizes are preferred for efficient GPU occupancy.
 *
 * Device properties are queried once per physical device and cached to avoid
 * repeated Vulkan API calls on every dispatch.
 *
 * @param physicalDevice The Vulkan physical device to query
 * @return Optimal workgroup size (always a power of 2, at most 1024)
 */
inline uint32_t optimalWorkgroupSize(VkPhysicalDevice physicalDevice) {
    thread_local VkPhysicalDevice cachedDevice = VK_NULL_HANDLE;
    thread_local uint32_t cachedResult = 0;

    if (cachedDevice == physicalDevice && cachedResult != 0) {
        return cachedResult;
    }

    VkPhysicalDeviceProperties props;
    vkGetPhysicalDeviceProperties(physicalDevice, &props);
    // Use the minimum of maxComputeWorkGroupSize[0] and maxComputeWorkGroupInvocations
    uint32_t max_x = props.limits.maxComputeWorkGroupSize[0];
    uint32_t max_inv = props.limits.maxComputeWorkGroupInvocations;
    uint32_t optimal = std::min(max_x, max_inv);
    // Clamp to powers of 2 for efficiency
    uint32_t result = 1;
    while (result * 2 <= optimal && result * 2 <= 1024) result *= 2;

    cachedDevice = physicalDevice;
    cachedResult = result;
    return result;
}

} // namespace vulkan
} // namespace tenzor
