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
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <unordered_map>
#include <mutex>

namespace tenzor {

// Performance tuning configuration
namespace vulkan_config {
    // Enable command batching to reduce submission overhead
    // When true, multiple operations are recorded into a shared command buffer
    // and submitted together, significantly reducing per-operation latency
    // Command batching for reduced per-operation overhead.
    // DEBUGGING: Re-enabled for RenderDoc capture to diagnose GPU hangs.
    constexpr bool USE_COMMAND_BATCHING = true;

    // Maximum operations to batch before auto-submit
    constexpr size_t BATCH_SIZE_THRESHOLD = 32;
}

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
    auto get_device_info(int32_t device_id) const -> DeviceInfo override;

    auto allocate(size_t bytes, int32_t device_id) -> void* override;
    auto deallocate(void* ptr) -> void override;
    auto copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void override;

    auto synchronize(int32_t device_id) -> void override;
    auto create_stream(int32_t device_id) -> StreamHandle override;
    auto destroy_stream(StreamHandle stream) -> void override;
    auto synchronize_stream(StreamHandle stream) -> void override;

    auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void override;

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
        bool canPreserveDenormsF32 = false;  // Whether GPU supports denormal preservation for float32
        bool hasAtomicInt64 = false;          // Whether GPU supports VK_EXT_shader_atomic_int64
        bool hasSubgroupArithmetic = false;   // Whether GPU supports subgroup arithmetic ops
        uint32_t subgroupSize = 0;            // Subgroup (warp) size for this device
        uint32_t workgroupSize = 256;         // Optimal 1D workgroup size (power-of-2, from device limits)
        VkPipelineCache pipelineCache = VK_NULL_HANDLE;  // Persistent pipeline cache

        // Configurable fence timeout (default 30s, override with TENZOR_VULKAN_FENCE_TIMEOUT_S)
        uint64_t fence_timeout_ns = 30'000'000'000ULL;

        // Fence-based async synchronization
        VkFence pendingFence = VK_NULL_HANDLE;  // Fence for last submitted work
        bool hasPendingWork = false;            // Whether fence needs to be waited on

        // Ring buffer of fences for true async execution
        static constexpr size_t MAX_FRAMES_IN_FLIGHT = 4;
        std::array<VkFence, MAX_FRAMES_IN_FLIGHT> frameFences{};
        std::array<VkCommandBuffer, MAX_FRAMES_IN_FLIGHT> frameCommandBuffers{};
        size_t currentFrame = 0;
        size_t submittedFrames = 0;  // Number of frames submitted but not yet waited on

        // Command buffer pool for reuse
        std::vector<VkCommandBuffer> commandBufferPool;
        size_t nextCommandBufferIndex = 0;
        static constexpr size_t COMMAND_BUFFER_POOL_SIZE = 32;

        // Active command buffer for batching multiple operations
        VkCommandBuffer activeCommandBuffer = VK_NULL_HANDLE;
        size_t operationsInBatch = 0;
        static constexpr size_t MAX_OPERATIONS_PER_BATCH = 64;

        // Per-device mutex for independent multi-GPU operation.
        mutable std::recursive_mutex mutex;

        // Set on VK_ERROR_DEVICE_LOST — prevents further submissions to lost device.
        std::atomic<bool> device_lost{false};

        DeviceContext() = default;
        DeviceContext(DeviceContext&& other) noexcept
            : physicalDevice(other.physicalDevice)
            , device(other.device)
            , computeQueue(other.computeQueue)
            , queueFamilyIndex(other.queueFamilyIndex)
            , commandPool(other.commandPool)
            , memoryProperties(other.memoryProperties)
            , descriptorPool(std::move(other.descriptorPool))
            , canPreserveDenormsF32(other.canPreserveDenormsF32)
            , hasAtomicInt64(other.hasAtomicInt64)
            , workgroupSize(other.workgroupSize)
            , pipelineCache(other.pipelineCache)
            , pendingFence(other.pendingFence)
            , hasPendingWork(other.hasPendingWork)
            , frameFences(other.frameFences)
            , frameCommandBuffers(other.frameCommandBuffers)
            , currentFrame(other.currentFrame)
            , submittedFrames(other.submittedFrames)
            , commandBufferPool(std::move(other.commandBufferPool))
            , nextCommandBufferIndex(other.nextCommandBufferIndex)
            , activeCommandBuffer(other.activeCommandBuffer)
            , operationsInBatch(other.operationsInBatch)
            // mutex is default-constructed (not movable)
            , device_lost(other.device_lost.load(std::memory_order_relaxed))
        {}
        DeviceContext& operator=(DeviceContext&&) = delete;
        DeviceContext(const DeviceContext&) = delete;
        DeviceContext& operator=(const DeviceContext&) = delete;
    };

    // Staging buffer for host-device transfers
    struct StagingBuffer {
        std::unique_ptr<vulkan::VulkanBuffer> buffer;
        size_t size = 0;
        bool in_use = false;  // Whether currently acquired by a transfer
    };

    // Pool of staging buffers per device for concurrent transfers
    struct StagingBufferPool {
        std::vector<StagingBuffer> buffers;
        std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();

        StagingBufferPool() = default;
        StagingBufferPool(StagingBufferPool&&) = default;
        StagingBufferPool& operator=(StagingBufferPool&&) = default;

        // Acquire a staging buffer of at least `size` bytes. Creates one if none available.
        // Returns index into `buffers`.
        size_t acquire(int32_t device_id, size_t size, const DeviceContext& ctx);
        // Release a staging buffer back to the pool for reuse.
        void release(size_t index);
    };

    // Deferred free entry for buffers awaiting GPU idle
    struct DeferredFree {
        void* ptr;
        size_t bytes;
        int32_t device_id;
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
    size_t acquireStagingBuffer(int32_t device_id, size_t size);
    void releaseStagingBuffer(int32_t device_id, size_t index);
    void flush_deferred_frees(int32_t device_id);

    // Command execution
    VkCommandBuffer beginSingleTimeCommands(int32_t device_id);
    void endSingleTimeCommands(VkCommandBuffer commandBuffer, int32_t device_id);

    // Async command execution (fence-based, no blocking)
    void endSingleTimeCommandsAsync(VkCommandBuffer commandBuffer, int32_t device_id);
    void ensurePendingWorkComplete(int32_t device_id);
    void initCommandBufferPool(DeviceContext& ctx);
    VkCommandBuffer acquireCommandBuffer(int32_t device_id);
    void releaseCommandBuffer(VkCommandBuffer cmdBuffer, int32_t device_id);

    // Batched command execution for improved performance
    void initFrameFences(DeviceContext& ctx);
    VkCommandBuffer getOrCreateBatchCommandBuffer(int32_t device_id);
    void recordOperationToBatch(int32_t device_id);
    void submitBatchIfNeeded(int32_t device_id, bool force = false);
    void waitForFrame(int32_t device_id, size_t frameIndex);

    // Pipeline management
    vulkan::ComputePipeline* getPipeline(const std::string& shader_name, int32_t device_id);

public:
    // Kernel dispatch helpers — used by vulkan_kernel_registry.cpp
    auto dispatchBinaryOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchUnaryOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchUnaryOpWithParam(const std::string& op_name, const Tensor& input, float param) -> Tensor;
    auto dispatchTrigonometricOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchHyperbolicOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchComparisonOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchReduction(const std::string& op_name, const Tensor& input,
                          int64_t dim, bool keepdim) -> Tensor;
    auto dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchBmm(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchDot(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchConv2d(const Tensor& input, const Tensor& weight,
                       const Tensor* bias, int64_t stride, int64_t padding,
                       int64_t dilation, int64_t groups) -> Tensor;
    auto dispatchConv2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor;
    auto dispatchConvTranspose2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor;

    // Conv2d backward operations
    auto dispatchConv2dBackwardInput(const Tensor& grad_output, const Tensor& weight,
                                     int64_t stride, int64_t padding, int64_t dilation,
                                     const std::vector<int64_t>& input_shape,
                                     int64_t groups = 1) -> Tensor;
    auto dispatchConv2dBackwardWeight(const Tensor& grad_output, const Tensor& input,
                                      int64_t stride, int64_t padding, int64_t dilation,
                                      const std::vector<int64_t>& weight_shape,
                                      int64_t groups = 1) -> Tensor;
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
    auto dispatchAdaptiveAvgPool2dBackward(const Tensor& grad_output, int64_t H_in, int64_t W_in) -> Tensor;
    auto dispatchMaxPool2dBackward(const Tensor& grad_out, const Tensor& input,
                                   const Tensor& indices, int64_t kernel_h, int64_t kernel_w,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t padding_h, int64_t padding_w) -> Tensor;
    auto dispatchMaxPool2dBackwardWithIndices(const Tensor& grad_output, const Tensor& indices,
                                              int64_t H_in, int64_t W_in) -> Tensor;

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
    auto dispatchBatchNorm2dUpdateRunningStats(std::span<const Tensor> inputs,
                                                const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchFusedRMSPropStep(std::span<const Tensor> inputs,
                                   const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchFusedAdadeltaStep(std::span<const Tensor> inputs,
                                    const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchFusedAdagradStep(std::span<const Tensor> inputs,
                                   const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchLayerNorm(const Tensor& input, int64_t normalized_shape,
                          const Tensor* gamma, const Tensor* beta, float epsilon) -> Tensor;
    auto dispatchGroupNorm(const Tensor& input, int64_t num_groups,
                          const Tensor* gamma, const Tensor* beta, float epsilon) -> std::vector<Tensor>;
    auto dispatchLayerNormBackward(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& mean, const Tensor& rstd,
                                   const Tensor* weight, int64_t normalized_shape)
                                   -> std::tuple<Tensor, Tensor, Tensor>;
    auto dispatchGroupNormBackward(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& mean, const Tensor& rstd,
                                   const Tensor* weight, int64_t num_groups)
                                   -> std::tuple<Tensor, Tensor, Tensor>;
    auto dispatchEmbeddingBackward(const Tensor& grad_output, const Tensor& indices,
                                    int64_t num_embeddings, int64_t embedding_dim) -> Tensor;
    auto dispatchRMSNorm(const Tensor& input, const Tensor& weight,
                         int64_t normalized_shape, float epsilon) -> std::pair<Tensor, Tensor>;
    auto dispatchRMSNormBackward(const Tensor& grad_output, const Tensor& input,
                                  const Tensor& rrms, const Tensor& weight,
                                  int64_t normalized_shape)
                                  -> std::pair<Tensor, Tensor>;

    // Softmax and loss operations
    auto dispatchSoftmax(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchLogSoftmax(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                             int64_t reduction) -> Tensor;

    // Advanced reduction operations
    auto dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchArgSort(const Tensor& input, int64_t dim, bool descending) -> Tensor;
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

    // Vision operations
    auto dispatchGatherRelativePositionBias(const Tensor& table, const Tensor& indices,
                                            int64_t num_positions, int64_t num_heads) -> Tensor;

    // Shape operations
    auto dispatchReshape(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    auto dispatchTranspose(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    auto dispatchPermute(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    auto dispatchSqueeze(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchUnsqueeze(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchContiguous(const Tensor& input) -> Tensor;

    // Creation operations
    auto dispatchArange(float start, float end, float step, DType dtype, const Device& device) -> Tensor;
    auto dispatchLinspace(float start, float end, int64_t steps, DType dtype, const Device& device) -> Tensor;
    auto dispatchEye(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor;

    // Memory operations
    auto dispatchZeros(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto dispatchFill(const Tensor& input, float value) -> Tensor;
    auto dispatchClone(const Tensor& input) -> Tensor;
    auto dispatchFull(const std::vector<int64_t>& shape, float value, DType dtype) -> Tensor;
    auto dispatchOnes(const std::vector<int64_t>& shape, DType dtype) -> Tensor;
    auto dispatchRand(const std::vector<int64_t>& shape, DType dtype) -> Tensor;
    auto dispatchRandn(const std::vector<int64_t>& shape, DType dtype) -> Tensor;

    // Type cast operations
    auto dispatchCast(const Tensor& input, DType target_dtype) -> Tensor;

    // Repeat and masked operations
    auto dispatchRepeat(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor;
    auto dispatchMaskedSelect(const Tensor& input, const Tensor& mask) -> Tensor;
    auto dispatchMaskedFill(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto dispatchWhere(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;

    // Tensor manipulation operations
    auto dispatchExpand(const Tensor& input, const std::vector<int64_t>& shape) -> Tensor;
    auto dispatchCat(const std::vector<Tensor>& inputs, int64_t dim) -> Tensor;
    auto dispatchClamp(const Tensor& input, float min_value, float max_value) -> Tensor;

    // Interpolation operation
    auto dispatchInterpolate(const Tensor& input, const OpAttributes& attrs) -> Tensor;

    // ROI Align operations
    auto dispatchROIAlignForward(const Tensor& features, const Tensor& rois, const OpAttributes& attrs) -> Tensor;
    auto dispatchROIAlignBackward(const Tensor& grad_output, const Tensor& rois, const OpAttributes& attrs) -> Tensor;

    // Phase 3 operations
    auto dispatchNonzero(const Tensor& input) -> Tensor;
    auto dispatchOneHot(const Tensor& indices, int64_t num_classes) -> Tensor;
    auto dispatchBoxIoU(const Tensor& boxes1, const Tensor& boxes2, int64_t iou_type) -> Tensor;

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
    VkDebugUtilsMessengerEXT debug_messenger_ = VK_NULL_HANDLE;
    std::vector<DeviceContext> devices_;
    std::vector<StagingBufferPool> stagingPools_;
    std::vector<PipelineCache> pipelineCaches_;

    // Deferred free lists (per-device) for buffers awaiting GPU idle
    std::vector<std::vector<DeferredFree>> deferred_frees_;
    // Mutex protecting deferred_frees_ (per-device vectors are accessed under device mutex,
    // but the outer vector itself needs protection during resize/init)

    // Memory tracking
    mutable std::mutex allocations_mutex_;  // Protects allocations_ map
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

    // Global mutex for cross-device operations (instance creation, shutdown, pipeline cache, dispatch).
    // Memory operations (allocate/deallocate/copy) use per-device mutexes + allocations_mutex_ instead.
    mutable std::recursive_mutex dispatch_mutex_;
};

} // namespace tenzor
