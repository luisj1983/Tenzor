/**
 * @file vulkan_backend.hpp
 * @brief Vulkan compute backend implementation
 *
 * Provides cross-platform GPU compute backend using Vulkan API.
 * Supports Windows, Linux, and macOS with compute shaders.
 */

#pragma once

#include "tenzor/backend/backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/device.hpp"
#include "vulkan_utils.hpp"
#include <vulkan/vulkan.h>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <unordered_map>

namespace tenzor {

/**
 * @brief Vulkan compute backend implementation
 *
 * Implements the Backend interface using Vulkan compute shaders.
 * Provides cross-platform GPU acceleration with buffer management,
 * pipeline caching, and optimized memory transfers.
 */
class VulkanBackend : public Backend {
public:
    VulkanBackend();
    ~VulkanBackend() override;

    // Backend interface implementation
    auto name() const -> std::string_view override { return "vulkan"; }
    auto device_count() const -> int32_t override;
    auto is_available() const -> bool override;

    auto allocate(size_t bytes, int32_t device_id) -> void* override;
    auto deallocate(void* ptr) -> void override;
    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override;

    auto synchronize(int32_t device_id) -> void override;
    auto create_stream(int32_t device_id) -> StreamHandle override;
    auto destroy_stream(StreamHandle stream) -> void override;
    auto synchronize_stream(StreamHandle stream) -> void override;

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override;

private:
    // Vulkan context management
    struct DeviceContext {
        VkPhysicalDevice physicalDevice;
        VkDevice device;
        VkQueue computeQueue;
        uint32_t queueFamilyIndex;
        VkCommandPool commandPool;
        VkPhysicalDeviceMemoryProperties memoryProperties;
        std::unique_ptr<vulkan::DescriptorPool> descriptorPool;
    };

    // Staging buffer for host-device transfers
    struct StagingBuffer {
        std::unique_ptr<vulkan::VulkanBuffer> buffer;
        size_t size;
    };

    // Pipeline cache for reusing compiled shaders
    struct PipelineCache {
        std::unordered_map<std::string, std::unique_ptr<vulkan::ComputePipeline>> pipelines;
    };

    // Initialization
    void initVulkan();
    void createInstance();
    void selectPhysicalDevices();
    void createLogicalDevices();
    void loadShaders();

    // Memory management
    void* allocateDeviceMemory(size_t bytes, int32_t device_id);
    void freeDeviceMemory(void* ptr, int32_t device_id);
    StagingBuffer& getStagingBuffer(int32_t device_id, size_t size);

    // Command execution
    VkCommandBuffer beginSingleTimeCommands(int32_t device_id);
    void endSingleTimeCommands(VkCommandBuffer commandBuffer, int32_t device_id);

    // Pipeline management
    vulkan::ComputePipeline* getPipeline(const std::string& shader_name, int32_t device_id);

    // Kernel dispatch helpers
    auto dispatchBinaryOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchUnaryOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchReduction(const std::string& op_name, const Tensor& input,
                          int64_t dim, bool keepdim) -> Tensor;
    auto dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchConv2d(const Tensor& input, const Tensor& weight,
                       const Tensor* bias, int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups) -> Tensor;

    // Instance and devices
    VkInstance instance_ = VK_NULL_HANDLE;
    std::vector<DeviceContext> devices_;
    std::vector<StagingBuffer> stagingBuffers_;
    std::vector<PipelineCache> pipelineCaches_;

    // Memory tracking
    std::unordered_map<void*, std::pair<size_t, int32_t>> allocations_;

    // Shader paths
    std::string shaderPath_;
};

} // namespace tenzor
