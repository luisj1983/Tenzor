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

    // Kernel dispatch helpers - Basic operations
    auto dispatchBinaryOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchUnaryOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchUnaryOpWithParam(const std::string& op_name, const Tensor& input, float param) -> Tensor;
    auto dispatchTrigonometricOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchHyperbolicOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchComparisonOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchReduction(const std::string& op_name, const Tensor& input,
                          int64_t dim, bool keepdim) -> Tensor;
    auto dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchDot(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchConv2d(const Tensor& input, const Tensor& weight,
                       const Tensor* bias, int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups) -> Tensor;
    auto dispatchConv2dForward(const Tensor& input, const Tensor& weight, const OpAttributes& attrs) -> Tensor;

    // Conv2d backward operations
    auto dispatchConv2dBackwardInput(const Tensor& grad_output, const Tensor& weight,
                                     int64_t stride, int64_t padding, int64_t dilation,
                                     const std::vector<int64_t>& input_shape) -> Tensor;
    auto dispatchConv2dBackwardWeight(const Tensor& grad_output, const Tensor& input,
                                      int64_t stride, int64_t padding, int64_t dilation,
                                      const std::vector<int64_t>& weight_shape) -> Tensor;
    auto dispatchConv2dBackwardBias(const Tensor& grad_output) -> Tensor;

    // Vision operations
    auto dispatchIm2Col(const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchCol2Im(const Tensor& input, const OpAttributes& attrs) -> Tensor;

    // Pooling operations
    auto dispatchMaxPool2d(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                          int64_t stride_h, int64_t stride_w,
                          int64_t padding_h, int64_t padding_w) -> std::pair<Tensor, Tensor>;
    auto dispatchAvgPool2d(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                          int64_t stride_h, int64_t stride_w,
                          int64_t padding_h, int64_t padding_w) -> Tensor;
    auto dispatchAdaptiveMaxPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor>;
    auto dispatchAdaptiveAvgPool2d(const Tensor& input, int64_t out_h, int64_t out_w) -> Tensor;
    auto dispatchMaxPool2dBackward(const Tensor& grad_out, const Tensor& input,
                                   const Tensor& indices, int64_t kernel_h, int64_t kernel_w,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t padding_h, int64_t padding_w) -> Tensor;

    // New pooling operations (OpAttributes versions)
    auto dispatchAvgPool2dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchMaxPool2dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchAvgPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchMaxPool2dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor;

    // Normalization operations
    auto dispatchBatchNorm2d(const Tensor& input, const Tensor& mean, const Tensor& var,
                            const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor;
    auto dispatchBatchNorm2dBackward(const Tensor& grad_out, const Tensor& input,
                                     const Tensor& mean, const Tensor& var,
                                     const Tensor* gamma, float epsilon)
                                     -> std::tuple<Tensor, Tensor, Tensor>;
    auto dispatchBatchNorm2dForward(const Tensor& input, const Tensor& mean, const Tensor& var,
                                    const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor;
    auto dispatchBatchNorm2dMeanVar(const Tensor& input) -> std::pair<Tensor, Tensor>;
    auto dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                          const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor;
    auto dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                          const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor;

    // Softmax and loss operations
    auto dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                             int64_t reduction) -> Tensor;

    // Advanced reduction operations
    auto dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor;
    auto dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor;
    auto dispatchNorm(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchAll(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchAny(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;

    // Indexing operations
    auto dispatchEmbedding(const Tensor& weight, const Tensor& indices,
                          int64_t padding_idx) -> Tensor;
    auto dispatchGather(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor;
    auto dispatchScatter(const Tensor& input, int64_t dim, const Tensor& indices,
                        const Tensor& values, int64_t reduction) -> Tensor;
    auto dispatchIndexSelect(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor;

    // Shape operations
    auto dispatchReshape(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    auto dispatchTranspose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    auto dispatchPermute(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    auto dispatchSqueeze(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchUnsqueeze(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchContiguous(const Tensor& input) -> Tensor;

    // Memory operations
    auto dispatchZeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto dispatchFill(const Tensor& input, float value) -> Tensor;
    auto dispatchClone(const Tensor& input) -> Tensor;
    auto dispatchFull(const std::vector<int64_t>& shape, float value, DType dtype) -> Tensor;
    auto dispatchOnes(const std::vector<int64_t>& shape, DType dtype) -> Tensor;
    auto dispatchRand(const std::vector<int64_t>& shape, DType dtype) -> Tensor;
    auto dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor;

    // Repeat and masked operations
    auto dispatchRepeat(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor;
    auto dispatchMaskedSelect(const Tensor& input, const Tensor& mask) -> Tensor;
    auto dispatchMaskedFill(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto dispatchWhere(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;

    // Tensor manipulation operations
    auto dispatchExpand(const Tensor& input, const std::vector<int64_t>& shape) -> Tensor;
    auto dispatchCat(const std::vector<Tensor>& inputs, int64_t dim) -> Tensor;
    auto dispatchClamp(const Tensor& input, float min_value, float max_value) -> Tensor;

    // Forward activation operations
    auto dispatchActivation(const std::string& op_name,
                           const Tensor& input,
                           uint32_t opcode,
                           float param) -> Tensor;

    // Backward activation operations
    auto dispatchActivationBackward(const std::string& op_name,
                                     const Tensor& grad_output,
                                     const Tensor& input_or_output,
                                     uint32_t opcode,
                                     float param) -> Tensor;
    auto dispatchSwishBackward(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto dispatchSoftmaxBackward(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;
    auto dispatchLogSoftmaxBackward(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;

    // Instance and devices
    VkInstance instance_ = VK_NULL_HANDLE;
    std::vector<DeviceContext> devices_;
    std::vector<StagingBuffer> stagingBuffers_;
    std::vector<PipelineCache> pipelineCaches_;

    // Memory tracking
    std::unordered_map<void*, std::pair<size_t, int32_t>> allocations_;

    // Buffer tracking: maps tensor data pointer (void*) to actual VulkanBuffer
    std::unordered_map<void*, std::unique_ptr<vulkan::VulkanBuffer>> bufferMap_;

    // Helper to get VkBuffer from tensor data pointer
    VkBuffer getVulkanBuffer(const void* ptr) const;
    std::pair<VkBuffer, VkDeviceSize> getVulkanBufferAndOffset(const void* ptr) const;

    // Helper to allocate and bind descriptor sets
    VkDescriptorSet allocateAndWriteDescriptorSet(
        int32_t device_id,
        vulkan::ComputePipeline* pipeline,
        const std::vector<std::pair<uint32_t, VkBuffer>>& bufferBindings,
        const std::vector<size_t>& bufferSizes);

    // Shader paths
    std::string shaderPath_;
};

} // namespace tenzor
