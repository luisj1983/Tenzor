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
#include "tenzor/sparse/sparse_tensor.hpp"
#include "vulkan_utils.hpp"
#include <vulkan/vulkan.h>
#include <array>
#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <tuple>
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

    /// Check if a device has been lost (GPU crash/hang/timeout).
    auto is_device_lost(int32_t device_id) const -> bool;

    /// Attempt to recover from a device-lost state by waiting for the device
    /// to become idle and resetting command pools. Returns true if recovery
    /// succeeded. If the device is truly gone (hardware failure), returns false
    /// and the device remains in a lost state.
    auto try_reset_device(int32_t device_id) -> bool;

    auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void override;

    auto dispatch(const std::string& op_name,
                 std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> std::vector<Tensor> override;

    // Broad GPU vendor classification for per-vendor workgroup / tile
    // tuning. VkPhysicalDeviceProperties::vendorID gives us the PCI
    // vendor; we collapse it to a small enum so dispatch sites can
    // branch without caring about the PCI numbers.
    enum class GpuVendor : uint8_t {
        Unknown = 0,
        Nvidia,   // vendorID 0x10DE — wavefront 32
        Amd,      // vendorID 0x1002 — wavefront 32/64 depending on arch
        Intel,    // vendorID 0x8086 — EU size 8/16/32
        Apple,    // vendorID 0x106B — SIMD width 32
        Arm,      // vendorID 0x13B5 — Mali; tiled renderer
        Qualcomm, // vendorID 0x5143 — Adreno
    };

    /**
     * @brief Return a recommended 2D tile size for the given vendor and
     *        op kind. Values are heuristic defaults chosen to match the
     *        vendor's subgroup width without inflating thread count.
     *
     * Call sites use this to *adjust dispatch counts* — each workgroup
     * still executes the tile size baked into the SPIR-V shader. A
     * deeper per-vendor tuning effort would swap shader variants via
     * specialization constants; until that lands, the `_x * _y` product
     * equals each shader's compile-time `local_size_x * local_size_y`
     * (currently 16 * 16 = 256) so the returned numbers only affect the
     * *shape* of the dispatch grid, not its total thread count.
     *
     * Phase 2.2: matmul shaders now use specialization constants
     * (constant_id 0/1 for TILE_X/TILE_Y) to receive vendor-specific
     * workgroup dimensions at pipeline creation time. Conv and other
     * ops can follow the same pattern as a follow-up.
     */
    enum class OpKind : uint8_t { Matmul, Conv, ElementWise };
    static auto recommended_workgroup_2d(GpuVendor vendor, OpKind op)
        -> std::pair<uint32_t, uint32_t>;

    /// Get the detected GPU vendor for a device (used by tests / profiling).
    auto get_device_vendor(int32_t device_id) const -> GpuVendor;

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
        bool hasAtomicFloat = false;          // Whether GPU supports VK_EXT_shader_atomic_float
        bool hasSubgroupArithmetic = false;   // Whether GPU supports subgroup arithmetic ops
        uint32_t subgroupSize = 0;            // Subgroup (warp) size for this device
        uint32_t workgroupSize = 256;         // Optimal 1D workgroup size (power-of-2, from device limits)
        GpuVendor vendor = GpuVendor::Unknown; // Collapsed vendor classification (Phase 2.2)
        uint32_t maxComputeWorkGroupCount[3] = {65535, 65535, 65535};  // Vulkan minimum guaranteed
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
            , maxComputeWorkGroupCount{other.maxComputeWorkGroupCount[0], other.maxComputeWorkGroupCount[1], other.maxComputeWorkGroupCount[2]}
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
        bool in_use = false;           // Whether currently acquired by a transfer
        uint64_t last_use_tick = 0;    // Monotonic tick for LRU eviction
    };

    // Pool of staging buffers per device for concurrent transfers.
    // Evicts oldest unused buffers when pool exceeds kMaxPoolSize.
    struct StagingBufferPool {
        static constexpr size_t kMaxPoolSize = 16;

        std::vector<StagingBuffer> buffers;
        std::unique_ptr<std::mutex> mutex = std::make_unique<std::mutex>();
        uint64_t tick_counter = 0;     // Monotonic counter for LRU ordering

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
    // Specialization-constant-aware pipeline: caches by (shader_name + spec_hash)
    vulkan::ComputePipeline* getPipelineSpecialized(
        const std::string& shader_name, int32_t device_id,
        const std::vector<VkSpecializationMapEntry>& specEntries,
        const void* specData, size_t specDataSize);

public:
    // Kernel dispatch helpers — used by vulkan_kernel_registry.cpp
    auto dispatchBinaryOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchUnaryOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchUnaryOpWithParam(const std::string& op_name, const Tensor& input, float param) -> Tensor;
    // NanToNum and bitwise shift dispatchers (vulkan_ops_misc.cpp). Replace previous CPU fallbacks.
    auto dispatchNanToNum(const Tensor& input, float nan_val, float posinf_val, float neginf_val) -> Tensor;
    auto dispatchBitwiseBinaryOp(const std::string& shader_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchHeaviside(const Tensor& input, const Tensor& values) -> Tensor;
    // Special-math dispatchers (vulkan_ops_special_math.cpp). Replace previous CPU fallbacks.
    auto dispatchSpecialMathUnary(const Tensor& input, uint32_t opcode, int32_t param_int = 0) -> Tensor;
    auto dispatchSpecialMathBinary(const Tensor& a, const Tensor& b, uint32_t opcode) -> Tensor;
    auto dispatchSpecialMathTernary(const Tensor& a, const Tensor& b, const Tensor& x) -> Tensor;
    // Grid sample / affine grid (vulkan_ops_grid_sample.cpp)
    auto dispatchGridSample(const Tensor& input, const Tensor& grid,
                            const std::string& mode_str, const std::string& padding_mode_str,
                            bool align_corners) -> Tensor;
    auto dispatchAffineGrid(const Tensor& theta, const std::vector<int64_t>& size, bool align_corners) -> Tensor;
    // Advanced (fancy) indexing (vulkan_ops_advanced_index.cpp)
    auto dispatchAdvancedIndex(const Tensor& src, const std::vector<Tensor>& indices,
                               int64_t num_indices) -> Tensor;
    auto dispatchAdvancedIndexPut(const Tensor& src, const std::vector<Tensor>& indices,
                                  const Tensor& values, int64_t num_indices) -> Tensor;
    // Sampling / statistics (vulkan_ops_sampling.cpp)
    auto dispatchBernoulli(const Tensor& probs) -> Tensor;
    auto dispatchBucketize(const Tensor& input, const Tensor& boundaries, bool right) -> Tensor;
    auto dispatchCDist(const Tensor& x1, const Tensor& x2, double p) -> Tensor;
    auto dispatchTrapezoid(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr) -> Tensor;
    auto dispatchCumulativeTrapezoid(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr) -> Tensor;
    auto dispatchGradient(const Tensor& input, int64_t dim, double spacing) -> Tensor;
    auto dispatchPairwiseDistance(const Tensor& x1, const Tensor& x2, double p) -> Tensor;
    auto dispatchPdist(const Tensor& input, double p) -> Tensor;
    auto dispatchHistogram(const Tensor& input, int64_t bins, double min_val, double max_val)
        -> std::pair<Tensor, Tensor>;
    auto dispatchHistogramdd(const Tensor& input, std::vector<int64_t> bins,
                             std::vector<std::pair<double,double>> ranges, bool density)
        -> std::pair<Tensor, std::vector<Tensor>>;
    auto dispatchMultinomial(const Tensor& probs, int64_t num_samples, bool replacement) -> Tensor;
    auto dispatchPoissonSample(const Tensor& rates) -> Tensor;
    auto dispatchNormalSample(const Tensor& mean, const Tensor& stddev) -> Tensor;
    auto dispatchExponentialSample(const Tensor& rate) -> Tensor;
    auto dispatchNestedAttention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                  const Tensor& q_offsets, const Tensor& kv_offsets,
                                  float scale, bool causal) -> Tensor;
    auto dispatchNestedAttentionBackward(const Tensor& grad_out,
                                          const Tensor& Q, const Tensor& K, const Tensor& V,
                                          const Tensor& q_offsets, const Tensor& kv_offsets,
                                          float scale, bool causal) -> std::vector<Tensor>;
    auto dispatchSTFT(const Tensor& input, int64_t n_fft, int64_t hop_length,
                      int64_t win_length, const Tensor& window, bool center,
                      bool normalized, bool onesided) -> Tensor;
    auto dispatchISTFT(const Tensor& input, int64_t n_fft, int64_t hop_length,
                       int64_t win_length, const Tensor& window, bool center,
                       bool normalized, bool onesided, int64_t length) -> Tensor;
    auto dispatchTrigonometricOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchHyperbolicOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchComparisonOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchBoolPredicateOp(const std::string& op_name, const Tensor& input) -> Tensor;
    auto dispatchLogicalOp(const std::string& op_name, const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchLerp(const Tensor& start, const Tensor& end, const Tensor& weight) -> Tensor;
    auto dispatchAddcmul(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value) -> Tensor;
    auto dispatchAddcdiv(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value) -> Tensor;
    auto dispatchCross(const Tensor& a, const Tensor& b, int64_t dim) -> Tensor;
    auto dispatchReduction(const std::string& op_name, const Tensor& input,
                          int64_t dim, bool keepdim) -> Tensor;
    auto dispatchMatmul(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchBmm(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchDot(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchConv2dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor;
    auto dispatchConv2dWinograd(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                int64_t padding, int64_t groups) -> Tensor;
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

    // Deformable Conv2d (DCNv2) operations
    auto dispatchDeformableConv2dForward(const Tensor& input, const Tensor& offset,
                                         const Tensor& weight, const Tensor& bias,
                                         const Tensor& mask,
                                         int64_t stride_h, int64_t stride_w,
                                         int64_t pad_h, int64_t pad_w,
                                         int64_t dil_h, int64_t dil_w,
                                         int64_t groups, int64_t offset_groups,
                                         bool use_mask) -> Tensor;
    auto dispatchDeformableConv2dBackwardInput(const Tensor& grad_output, const Tensor& input,
                                                const Tensor& offset, const Tensor& weight,
                                                const Tensor& mask,
                                                int64_t stride_h, int64_t stride_w,
                                                int64_t pad_h, int64_t pad_w,
                                                int64_t dil_h, int64_t dil_w,
                                                int64_t groups, int64_t offset_groups,
                                                bool use_mask) -> std::vector<Tensor>;
    auto dispatchDeformableConv2dBackwardWeight(const Tensor& grad_output, const Tensor& input,
                                                 const Tensor& offset, const Tensor& mask,
                                                 int64_t stride_h, int64_t stride_w,
                                                 int64_t pad_h, int64_t pad_w,
                                                 int64_t dil_h, int64_t dil_w,
                                                 int64_t groups, int64_t offset_groups,
                                                 bool use_mask,
                                                 const std::vector<int64_t>& weight_shape) -> Tensor;

    // Conv3d operations
    auto dispatchConv3dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor;
    auto dispatchConv3dBackwardInput(const Tensor& grad_output, const Tensor& weight,
                                     int64_t stride, int64_t padding, int64_t dilation,
                                     const std::vector<int64_t>& input_shape,
                                     int64_t groups = 1) -> Tensor;
    auto dispatchConv3dBackwardWeight(const Tensor& grad_output, const Tensor& input,
                                      int64_t stride, int64_t padding, int64_t dilation,
                                      const std::vector<int64_t>& weight_shape,
                                      int64_t groups = 1) -> Tensor;
    auto dispatchConv3dBackwardBias(const Tensor& grad_output) -> Tensor;

    // ConvTranspose3d operations (use Conv3d shader duality)
    auto dispatchConvTranspose3dForward(const Tensor& input, const Tensor& weight, const Tensor* bias, const OpAttributes& attrs) -> Tensor;
    auto dispatchConvTranspose3dBackwardInput(const Tensor& grad_output, const Tensor& weight, const OpAttributes& attrs) -> Tensor;
    auto dispatchConvTranspose3dBackwardWeight(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, const OpAttributes& attrs) -> Tensor;
    auto dispatchConvTranspose3dBackwardBias(const Tensor& grad_output) -> Tensor;

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

    // 1D pooling operations
    auto dispatchMaxPool1dForward(const Tensor& input, const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchMaxPool1dBackward(const Tensor& grad_output, const Tensor& indices,
                                   int64_t L_in) -> Tensor;
    auto dispatchAvgPool1dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchAvgPool1dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchAdaptiveMaxPool1d(const Tensor& input, int64_t output_size) -> std::pair<Tensor, Tensor>;
    auto dispatchAdaptiveAvgPool1d(const Tensor& input, int64_t output_size) -> Tensor;
    auto dispatchAdaptiveMaxPool1dBackward(const Tensor& grad_output, const Tensor& indices,
                                            const std::vector<int64_t>& input_shape) -> Tensor;
    auto dispatchAdaptiveAvgPool1dBackward(const Tensor& grad_output, int64_t L_in) -> Tensor;

    // 3D pooling operations
    auto dispatchMaxPool3dForward(const Tensor& input, const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                   int64_t D_in, int64_t H_in, int64_t W_in) -> Tensor;
    auto dispatchAvgPool3dForward(const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchAvgPool3dBackward(const Tensor& grad_output, const Tensor& input, const OpAttributes& attrs) -> Tensor;
    auto dispatchAdaptiveMaxPool3d(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w) -> std::pair<Tensor, Tensor>;
    auto dispatchAdaptiveAvgPool3d(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor;
    auto dispatchAdaptiveMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                            const std::vector<int64_t>& input_shape) -> Tensor;
    auto dispatchAdaptiveAvgPool3dBackward(const Tensor& grad_output, int64_t D_in, int64_t H_in, int64_t W_in) -> Tensor;

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
    auto dispatchNestedLogSoftmax(const Tensor& values, const Tensor& offsets, int64_t dim) -> Tensor;
    auto dispatchNestedSoftmax(const Tensor& values, const Tensor& offsets, int64_t dim) -> Tensor;
    auto dispatchNestedSum(const Tensor& values, const Tensor& offsets) -> Tensor;
    auto dispatchNestedMean(const Tensor& values, const Tensor& offsets) -> Tensor;
    auto dispatchNestedToPadded(const Tensor& values, const Tensor& offsets,
                                int64_t max_len, float padding_value) -> Tensor;
    auto dispatchNestedFromPadded(const Tensor& padded, const Tensor& offsets) -> Tensor;
    auto dispatchCrossEntropy(const Tensor& log_probs, const Tensor& targets,
                             int64_t reduction) -> Tensor;

    // Advanced reduction operations
    auto dispatchArgmax(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchArgmin(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchArgSort(const Tensor& input, int64_t dim, bool descending) -> Tensor;
    auto dispatchVariance(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor;
    auto dispatchStd(const Tensor& input, int64_t dim, bool unbiased, bool keepdim) -> Tensor;
    // Single-pass Welford implementation backing both variance and std.
    auto dispatchVarianceWelford(const Tensor& input, int64_t dim, bool unbiased,
                                 bool keepdim, bool compute_std) -> Tensor;
    auto dispatchNorm(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchProd(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchAll(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchAny(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchBooleanReduction(const std::string& op_name, const Tensor& input,
                                  int64_t dim, bool keepdim) -> Tensor;
    auto dispatchLogSumExp(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchTriuTril(const std::string& op_name, const Tensor& input,
                          int64_t diagonal) -> Tensor;
    auto dispatchDiag(const Tensor& input, int64_t diagonal) -> Tensor;
    auto dispatchFlip(const Tensor& input_orig, int64_t dim) -> Tensor;
    auto dispatchRoll(const Tensor& input, int64_t shift, int64_t dim) -> Tensor;
    auto dispatchTrace(const Tensor& input) -> Tensor;
    auto dispatchCountNonzero(const Tensor& input) -> Tensor;
    auto dispatchNansum(const Tensor& input) -> Tensor;
    auto dispatchNanmean(const Tensor& input) -> Tensor;
    auto dispatchNanVar(const Tensor& input, int64_t correction) -> Tensor;
    auto dispatchNanStd(const Tensor& input, int64_t correction) -> Tensor;
    auto dispatchAminmax(const Tensor& input) -> std::pair<Tensor, Tensor>;
    auto dispatchFrexp(const Tensor& input) -> std::pair<Tensor, Tensor>;
    auto dispatchDiagEmbed(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor;
    auto dispatchDiagflat(const Tensor& input, int64_t offset) -> Tensor;

    // Indexing operations
    auto dispatchEmbedding(const Tensor& weight, const Tensor& indices,
                          int64_t padding_idx) -> Tensor;
    auto dispatchGather(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor;
    auto dispatchScatter(const Tensor& input, int64_t dim, const Tensor& indices,
                        const Tensor& values, int64_t reduction) -> Tensor;
    auto dispatchScatterAdd(const Tensor& self, int64_t dim,
                            const Tensor& index, const Tensor& src) -> Tensor;
    auto dispatchScatterReduce(const Tensor& self, int64_t dim,
                               const Tensor& index, const Tensor& src,
                               const std::string& reduce, bool include_self) -> Tensor;
    auto dispatchIndexAdd(const Tensor& self, int64_t dim,
                          const Tensor& index, const Tensor& src) -> Tensor;
    auto dispatchIndexCopy(const Tensor& self, int64_t dim,
                           const Tensor& index, const Tensor& src) -> Tensor;
    auto dispatchIndexFill(const Tensor& self, int64_t dim,
                           const Tensor& index, float value) -> Tensor;
    auto dispatchIndexSelect(const Tensor& input, int64_t dim, const Tensor& indices) -> Tensor;
    auto dispatchMaskedSelect(const Tensor& input, const Tensor& mask) -> Tensor;
    auto dispatchMaskedFill(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto dispatchWhere(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;
    auto dispatchRepeat(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor;
    auto dispatchRepeatInterleave(const Tensor& input, int64_t repeats, int64_t dim) -> Tensor;
    auto dispatchRepeatInterleaveTensor(const Tensor& input, const Tensor& repeats, int64_t dim) -> Tensor;

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
    auto dispatchRandint(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                          DType dtype, const Device& device) -> Tensor;

    // Scan operations
    auto dispatchLogcumsumexp(const Tensor& input, int64_t dim) -> Tensor;

    // New reduction operations
    auto dispatchCumMax(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor>;
    auto dispatchCumMin(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor>;
    auto dispatchFmax(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchFmin(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchIsin(const Tensor& elements, const Tensor& test_elements) -> Tensor;
    auto dispatchKthvalue(const Tensor& input, int64_t k, int64_t dim, bool keepdim) -> std::pair<Tensor, Tensor>;
    auto dispatchQuantile(const Tensor& input, double q, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchNanquantile(const Tensor& input, double q, int64_t dim, bool keepdim) -> Tensor;
    auto dispatchNanmedian(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchHistc(const Tensor& input, int64_t bins, double min_val, double max_val) -> Tensor;
    auto dispatchUniqueConsecutive(const Tensor& input, bool return_inverse) -> std::tuple<Tensor, Tensor, Tensor>;
    auto dispatchSegmentReduce(const Tensor& data, const Tensor& offsets,
                               const std::string& reduce, int64_t axis) -> Tensor;

    // Fractional Max Pool + Max Unpool operations
    auto dispatchFractionalMaxPool2dForward(const Tensor& input, int64_t out_h, int64_t out_w,
                                            const Tensor* random_samples) -> std::pair<Tensor, Tensor>;
    auto dispatchFractionalMaxPool2dBackward(const Tensor& grad_output, const Tensor& indices,
                                             const std::vector<int64_t>& input_shape) -> Tensor;
    auto dispatchFractionalMaxPool3dForward(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w,
                                            const Tensor* random_samples) -> std::pair<Tensor, Tensor>;
    auto dispatchFractionalMaxPool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                             const std::vector<int64_t>& input_shape) -> Tensor;
    auto dispatchMaxUnpool2dForward(const Tensor& input, const Tensor& indices,
                                    int64_t out_h, int64_t out_w) -> Tensor;
    auto dispatchMaxUnpool2dBackward(const Tensor& grad_output, const Tensor& indices,
                                     const std::vector<int64_t>& input_shape) -> Tensor;
    auto dispatchMaxUnpool3dForward(const Tensor& input, const Tensor& indices,
                                    int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor;
    auto dispatchMaxUnpool3dBackward(const Tensor& grad_output, const Tensor& indices,
                                     const std::vector<int64_t>& input_shape) -> Tensor;

    // MaskedScatter with precomputed prefix sum
    auto dispatchMaskedScatterWithPrefix(const Tensor& input, const Tensor& mask,
                                         const Tensor& source, const Tensor& prefix_sum) -> Tensor;

    // Triangular index generation (native Vulkan compute shaders)
    auto dispatchTrilIndices(int64_t row, int64_t col, int64_t offset) -> Tensor;
    auto dispatchTriuIndices(int64_t row, int64_t col, int64_t offset) -> Tensor;

    // Histogram operations
    auto dispatchBincount(const Tensor& input, const std::optional<Tensor>& weights, int64_t minlength) -> Tensor;

    // Type cast operations
    auto dispatchCast(const Tensor& input, DType target_dtype) -> Tensor;

    // RNN operations (Phase 11.3)
    auto dispatchLSTMForward(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                             const Tensor& bias_ih, const Tensor& bias_hh,
                             const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;
    auto dispatchGRUForward(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                            const Tensor& bias, const Tensor& h0) -> std::vector<Tensor>;
    auto dispatchLSTMCellForward(const Tensor& input, const Tensor& hx, const Tensor& cx,
                                 const Tensor& weight_ih, const Tensor& weight_hh,
                                 const Tensor& bias_ih, const Tensor& bias_hh) -> std::vector<Tensor>;
    auto dispatchGRUCellForward(const Tensor& input, const Tensor& hx,
                                const Tensor& weight_ih, const Tensor& weight_hh,
                                const Tensor& bias_ih, const Tensor& bias_hh) -> Tensor;
    auto dispatchLSTMCellBackward(const Tensor& grad_h, const Tensor& grad_c_next,
                                  const Tensor& gates, const Tensor& c_prev,
                                  const Tensor& c_out,
                                  int64_t batch_size, int64_t hidden_size) -> std::vector<Tensor>;
    auto dispatchGRUCellBackward(const Tensor& grad_h, const Tensor& gates_x,
                                 const Tensor& gates_h, const Tensor& h_prev,
                                 int64_t batch_size, int64_t hidden_size) -> std::vector<Tensor>;
    auto dispatchLSTMMultiLayerForward(const Tensor& input,
                                       const std::vector<Tensor>& W_ih_list,
                                       const std::vector<Tensor>& W_hh_list,
                                       const std::vector<Tensor>& bias_list,
                                       const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;
    auto dispatchGRUMultiLayerForward(const Tensor& input,
                                      const std::vector<Tensor>& W_ih_list,
                                      const std::vector<Tensor>& W_hh_list,
                                      const std::vector<Tensor>& bias_list,
                                      const Tensor& h0) -> std::vector<Tensor>;
    auto dispatchBiLSTMForward(const Tensor& input, const Tensor& h0, const Tensor& c0,
                                const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
                                const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
                                const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
                                const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd) -> std::vector<Tensor>;

    // Sorting operations (Phase 11.4)
    auto dispatchSort(const Tensor& input, int64_t dim, bool descending) -> std::pair<Tensor, Tensor>;
    auto dispatchTopK(const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted) -> std::pair<Tensor, Tensor>;
    auto dispatchUnique(const Tensor& input, bool sorted, bool return_inverse, bool return_counts) -> std::vector<Tensor>;

    // Median and Mode operations
    auto dispatchMedian(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor>;
    auto dispatchMode(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor>;

    // Misc operations (Phase 11.5)
    auto dispatchStridedFill(Tensor& input, float value) -> void;
    auto dispatchToMemoryFormat(const Tensor& input, int format) -> Tensor;
    auto dispatchHasInfNan(const Tensor& input) -> Tensor;
    auto dispatchDepthwiseConv2d(const Tensor& input, const Tensor& weight,
                                  const Tensor* bias, int64_t stride,
                                  int64_t padding, int64_t dilation) -> Tensor;
    auto dispatchCumSum(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchCumProd(const Tensor& input, int64_t dim) -> Tensor;
    auto dispatchAdaptiveMaxPool2dBackward(const Tensor& grad_output, const Tensor& indices,
                                            const std::vector<int64_t>& input_shape) -> Tensor;

    // Linear/FC operations
    auto dispatchLinear(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto dispatchLinearBackward(const Tensor& grad_output, const Tensor& input, const Tensor& weight) -> std::vector<Tensor>;

    // Dropout operations
    auto dispatchDropout(const Tensor& input, float p, bool training) -> std::pair<Tensor, Tensor>;
    auto dispatchDropoutBackward(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor;

    // Slice/Split/Chunk/Flatten operations
    auto dispatchSlice(const Tensor& input, const std::vector<int64_t>& starts,
                       const std::vector<int64_t>& ends, const std::vector<int64_t>& steps) -> Tensor;
    auto dispatchSplit(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor>;
    auto dispatchChunk(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor>;
    auto dispatchFlatten(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor;

    // Tensor manipulation operations (Repeat/Masked already declared above)
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
    auto dispatchNMS(const Tensor& boxes, const Tensor& scores, float iou_threshold) -> Tensor;

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
    auto dispatchRReLU(const Tensor& input, float lower, float upper, bool training) -> Tensor;
    auto dispatchRReLUBackward(const Tensor& grad_output, const Tensor& input, float slope) -> Tensor;
    auto dispatchLogSigmoidBackward(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto dispatchSoftmaxBackward(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;
    auto dispatchLogSoftmaxBackward(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;

    // Instance Normalization operations
    auto dispatchInstanceNorm(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, float epsilon) -> std::vector<Tensor>;
    auto dispatchInstanceNormBackward(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& mean, const Tensor& rstd,
                                      const Tensor& weight)
                                      -> std::tuple<Tensor, Tensor, Tensor>;

    // EmbeddingBag operation
    auto dispatchEmbeddingBag(const Tensor& embeddings, const Tensor& offsets,
                               int64_t embedding_dim, const std::string& mode,
                               bool include_last_offset) -> Tensor;
    auto dispatchEmbeddingBagBackward(const Tensor& grad_output, const Tensor& indices,
                                       const Tensor& offsets, int64_t num_embeddings,
                                       int64_t embedding_dim, const std::string& mode,
                                       bool include_last_offset) -> Tensor;

    // Fused optimizer steps
    auto dispatchFusedSGDStep(std::span<const Tensor> inputs,
                               const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchFusedAdamStep(std::span<const Tensor> inputs,
                                const OpAttributes& attrs) -> std::vector<Tensor>;
    auto dispatchFusedAdamAtan2Step(std::span<const Tensor> inputs,
                                    const OpAttributes& attrs) -> std::vector<Tensor>;

    // Complex number operations
    auto dispatchConj(const Tensor& input) -> Tensor;
    auto dispatchReal(const Tensor& input) -> Tensor;
    auto dispatchImag(const Tensor& input) -> Tensor;
    auto dispatchAngle(const Tensor& input) -> Tensor;
    auto dispatchPolar(const Tensor& abs, const Tensor& angle) -> Tensor;
    auto dispatchComplexTensor(const Tensor& real, const Tensor& imag) -> Tensor;

    // Stack/Take/Tile/Put operations (native Vulkan shaders)
    auto dispatchStack(std::span<const Tensor> inputs, int64_t dim) -> Tensor;
    auto dispatchTake(const Tensor& input, const Tensor& indices) -> Tensor;
    auto dispatchTile(const Tensor& input, const std::vector<int64_t>& reps) -> Tensor;
    auto dispatchPut(const Tensor& input, const Tensor& indices, const Tensor& source, bool accumulate) -> Tensor;

    // FFT operations (native Vulkan compute shaders)
    auto dispatchFFT(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm) -> Tensor;
    auto dispatchIFFT(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm) -> Tensor;
    auto dispatchRFFT(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm) -> Tensor;
    auto dispatchIRFFT(const Tensor& input, int64_t dim, int64_t n,
                       const std::string& norm) -> Tensor;
    auto dispatchFFT2(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::string& norm) -> Tensor;
    auto dispatchIFFT2(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::string& norm) -> Tensor;
    auto dispatchFFTN(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::string& norm) -> Tensor;
    auto dispatchIFFTN(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::string& norm) -> Tensor;

    // FFT internal helpers
    auto runFFTButterfly(const Tensor& input, uint32_t fft_size, uint32_t direction,
                         uint32_t batch_size, uint32_t batch_stride) -> Tensor;
    auto runFFTScale(Tensor& data, uint32_t n, double scale_factor) -> void;
    auto runFFTChirpMultiply(Tensor& data, const Tensor& chirp, uint32_t n,
                              bool conjugate) -> void;
    auto runFFTChirpGen(Tensor& output, uint32_t N, int32_t sign) -> void;
    auto runFFTConjKernelGen(Tensor& output, uint32_t N, uint32_t M, int32_t sign) -> void;
    auto dispatchFFTBluestein(const Tensor& input, int64_t signal_len,
                               uint32_t direction) -> Tensor;
    auto runMixedRadixFFT(const Tensor& input, int64_t N, uint32_t direction,
                           uint32_t batch_size, uint32_t batch_stride) -> Tensor;

    // Radix sort for large arrays (> 65K elements)
    auto dispatchRadixSort(const Tensor& input, bool descending) -> std::pair<Tensor, Tensor>;

    // Linear algebra operations (single-workgroup shaders for small matrices, tiled GPU for large)
    auto dispatchLinalgDet(const Tensor& input) -> Tensor;
    auto dispatchLinalgInv(const Tensor& input) -> Tensor;
    auto dispatchLinalgSolve(const Tensor& a, const Tensor& b) -> Tensor;
    auto dispatchLinalgCholesky(const Tensor& input, bool upper) -> Tensor;
    auto dispatchLinalgQR(const Tensor& input) -> std::vector<Tensor>;
    auto dispatchLinalgSVD(const Tensor& input, bool full_matrices) -> std::vector<Tensor>;
    auto dispatchLinalgEigh(const Tensor& input) -> std::vector<Tensor>;
    auto dispatchLinalgEig(const Tensor& input) -> std::vector<Tensor>;
    auto dispatchLinalgLU(const Tensor& input) -> std::vector<Tensor>;
    auto dispatchLinalgLUSolve(const Tensor& LU_data, const Tensor& pivots,
                               const Tensor& B) -> Tensor;
    auto dispatchLinalgSolveTriangular(const Tensor& A, const Tensor& B,
                                       bool upper, bool unitriangular) -> Tensor;
    auto dispatchGeqrf(const Tensor& input) -> std::vector<Tensor>;
    auto dispatchOrmqr(const Tensor& reflectors, const Tensor& tau,
                        const Tensor& C, bool left, bool transpose_q) -> Tensor;
    auto dispatchLinalgHouseholder(const Tensor& input, const Tensor& tau) -> Tensor;
    auto dispatchLinalgLDLFactor(const Tensor& A) -> std::vector<Tensor>;
    auto dispatchLinalgLDLSolve(const Tensor& LD, const Tensor& pivots,
                                 const Tensor& B) -> Tensor;

    // Tiled blocked linalg helpers for medium matrices (33-256)
    void runBlockedLU(Tensor& A, Tensor& pivots, int64_t n,
                      int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16);
    void runBlockedCholesky(Tensor& A, int64_t n,
                            int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16);
    void runBlockedQR(Tensor& A, Tensor& tau, int64_t m, int64_t n,
                      int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16);
    void runBlockedBidiag(Tensor& A, Tensor& tau_l, Tensor& tau_r, int64_t n,
                          int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16);
    void runBlockedTridiag(Tensor& A, Tensor& tau, int64_t n,
                           int64_t batch_size, int32_t device_id, bool is_f64, bool is_f16);
    void runBlockedHessenberg(Tensor& A, Tensor& tau, int64_t n,
                              int64_t batch_size, int32_t device_id, bool is_f64);

    // SearchSorted (native GPU binary search shader)
    auto dispatchSearchSorted(const Tensor& sorted, const Tensor& values) -> Tensor;

    // Quantized operations (native Int8 GPU shaders)
    auto dispatchQuantizedLinear(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                  float input_scale, float weight_scale,
                                  int32_t input_zp, int32_t weight_zp) -> Tensor;
    auto dispatchQuantizedConv2d(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                  int64_t stride, int64_t padding,
                                  float input_scale, float weight_scale,
                                  int32_t input_zp, int32_t weight_zp) -> Tensor;

    // Sparse tensor operations (raw CSR components from kernel registry dispatch)
    auto dispatchSparseSpMM(const Tensor& crow_indices, const Tensor& col_indices,
                             const Tensor& values, const Tensor& dense,
                             int64_t M, int64_t K, int64_t N) -> Tensor;
    auto dispatchSparseSpMV(const Tensor& crow_indices, const Tensor& col_indices,
                             const Tensor& values, const Tensor& vec,
                             int64_t M, int64_t K) -> Tensor;
    auto dispatchSparseToDense(const Tensor& crow_indices, const Tensor& col_indices,
                                const Tensor& values, int64_t M, int64_t K, DType dtype) -> Tensor;
    auto dispatchSparseAdd(const Tensor& crow_indices, const Tensor& col_indices,
                            const Tensor& values, const Tensor& dense,
                            int64_t M, int64_t K) -> Tensor;
    auto dispatchDenseToSparse(const Tensor& dense) -> std::vector<Tensor>;

    // Sparse SpGEMM, Trsv, Trsm
    auto dispatchSparseSpGEMM(const Tensor& a_crow, const Tensor& a_col,
                               const Tensor& a_vals,
                               const Tensor& b_crow, const Tensor& b_col,
                               const Tensor& b_vals,
                               int64_t M, int64_t K, int64_t N) -> std::vector<Tensor>;
    auto dispatchSparseTrsv(const Tensor& crow_indices, const Tensor& col_indices,
                             const Tensor& values, const Tensor& b,
                             int64_t N, bool upper) -> Tensor;
    auto dispatchSparseTrsm(const Tensor& crow_indices, const Tensor& col_indices,
                             const Tensor& values, const Tensor& B,
                             int64_t N, int64_t K_rhs, bool upper) -> Tensor;

    // Flash Attention — composed from existing matmul + softmax shaders (not a single fused kernel)
    auto dispatchFlashAttention(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 float scale, bool causal) -> Tensor;

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

    // Helper to allocate and bind descriptor sets.
    // Accepts raw data pointers — resolves VkBuffer + byte offset internally
    // so tensor views (slices) bind at the correct offset.
    VkDescriptorSet allocateAndWriteDescriptorSet(
        int32_t device_id,
        vulkan::ComputePipeline* pipeline,
        const std::vector<std::pair<uint32_t, const void*>>& bufferPtrs,
        const std::vector<size_t>& bufferSizes);

    // Shader paths
    std::string shaderPath_;

    // Global mutex for cross-device operations (instance creation, shutdown, pipeline cache, dispatch).
    // Memory operations (allocate/deallocate/copy) use per-device mutexes + allocations_mutex_ instead.
    mutable std::recursive_mutex dispatch_mutex_;

    // Matmul autotuning infrastructure
    struct MatmulConfig {
        uint32_t tile_m{16}, tile_n{16};
        uint32_t local_size_x{16}, local_size_y{16};
    };

    /// Select matmul config based on problem dimensions (heuristic)
    static auto selectMatmulConfig(int64_t M, int64_t N, int64_t K) -> MatmulConfig {
        if (M < 64 && N < 64 && K < 64) return {8, 8, 8, 8};      // Tiny
        if (M >= 512 && N >= 512) return {32, 32, 32, 32};          // Large
        return {16, 16, 16, 16};                                     // Default
    }

    /// Cache for matmul configurations by "MxNxK" key
    std::unordered_map<std::string, MatmulConfig> matmul_autotune_cache_;
    mutable std::mutex autotune_mutex_;

    auto getMatmulConfig(int64_t M, int64_t N, int64_t K) -> const MatmulConfig& {
        std::string key = std::to_string(M) + "x" + std::to_string(N) + "x" + std::to_string(K);
        std::lock_guard lock(autotune_mutex_);
        auto it = matmul_autotune_cache_.find(key);
        if (it != matmul_autotune_cache_.end()) return it->second;
        auto config = selectMatmulConfig(M, N, K);
        auto [inserted_it, _] = matmul_autotune_cache_.emplace(key, config);
        return inserted_it->second;
    }
};

} // namespace tenzor
