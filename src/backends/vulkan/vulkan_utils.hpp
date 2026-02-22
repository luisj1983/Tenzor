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
    std::vector<uint32_t> buffer(fileSize / sizeof(uint32_t));
    file.seekg(0);
    file.read(reinterpret_cast<char*>(buffer.data()), fileSize);
    file.close();

    return buffer;
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

        checkVk(vkAllocateMemory(device, &allocInfo, nullptr, &memory_),
                "Failed to allocate buffer memory");

        vkBindBufferMemory(device, buffer_, memory_, 0);
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
public:
    DescriptorPool(VkDevice device, uint32_t maxSets)
        : device_(device) {

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

    /**
     * @brief Reset the descriptor pool, freeing all allocated descriptor sets
     *
     * This should only be called when no descriptor sets are in use (i.e., after
     * synchronization). All previously allocated descriptor sets become invalid.
     */
    void reset() {
        if (pool_ != VK_NULL_HANDLE) {
            vkResetDescriptorPool(device_, pool_, 0);
        }
    }

private:
    VkDevice device_;
    VkDescriptorPool pool_ = VK_NULL_HANDLE;
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
                   VkPipelineCache pipelineCache = VK_NULL_HANDLE)
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

} // namespace vulkan
} // namespace tenzor
