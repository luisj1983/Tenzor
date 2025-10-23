#pragma once

#include <webgpu/webgpu.h>
#include <memory>
#include <vector>
#include <string>
#include <unordered_map>
#include <functional>
#include <future>

namespace tenzor {
namespace webgpu {

/**
 * @brief WebGPU Buffer wrapper with memory management
 */
class WebGPUBuffer {
public:
    WebGPUBuffer(WGPUDevice device, size_t size, WGPUBufferUsageFlags usage);
    ~WebGPUBuffer();

    // Prevent copying
    WebGPUBuffer(const WebGPUBuffer&) = delete;
    WebGPUBuffer& operator=(const WebGPUBuffer&) = delete;

    // Allow moving
    WebGPUBuffer(WebGPUBuffer&& other) noexcept;
    WebGPUBuffer& operator=(WebGPUBuffer&& other) noexcept;

    WGPUBuffer get() const { return buffer_; }
    size_t size() const { return size_; }
    WGPUBufferUsageFlags usage() const { return usage_; }

    // Async read operation
    std::future<std::vector<uint8_t>> readAsync();

    // Async write operation
    void writeAsync(const void* data, size_t size, size_t offset = 0);

    // Map/unmap operations
    void map(WGPUMapModeFlags mode, std::function<void(void*, size_t)> callback);
    void unmap();

private:
    WGPUDevice device_;
    WGPUBuffer buffer_;
    size_t size_;
    WGPUBufferUsageFlags usage_;
    bool mapped_;
};

/**
 * @brief WebGPU Compute Pipeline wrapper
 */
class WebGPUComputePipeline {
public:
    WebGPUComputePipeline(WGPUDevice device, const std::string& shaderCode,
                          const std::string& entryPoint = "main");
    ~WebGPUComputePipeline();

    // Prevent copying
    WebGPUComputePipeline(const WebGPUComputePipeline&) = delete;
    WebGPUComputePipeline& operator=(const WebGPUComputePipeline&) = delete;

    // Allow moving
    WebGPUComputePipeline(WebGPUComputePipeline&& other) noexcept;
    WebGPUComputePipeline& operator=(WebGPUComputePipeline&& other) noexcept;

    WGPUComputePipeline get() const { return pipeline_; }
    WGPUBindGroupLayout getBindGroupLayout() const { return bindGroupLayout_; }

private:
    WGPUDevice device_;
    WGPUComputePipeline pipeline_;
    WGPUBindGroupLayout bindGroupLayout_;
    WGPUShaderModule shaderModule_;
};

/**
 * @brief WebGPU Backend Configuration
 */
struct WebGPUConfig {
    // Device selection
    WGPUPowerPreference powerPreference = WGPUPowerPreference_HighPerformance;

    // Feature requirements
    bool requireTimestampQuery = false;
    bool requireF16 = false;

    // Memory limits
    size_t maxBufferSize = 1024 * 1024 * 1024; // 1GB
    size_t maxBindGroups = 4;

    // Workgroup size limits (for browser compatibility)
    uint32_t maxWorkgroupSizeX = 256;
    uint32_t maxWorkgroupSizeY = 256;
    uint32_t maxWorkgroupSizeZ = 64;
    uint32_t maxWorkgroupInvocations = 256;
};

/**
 * @brief Main WebGPU Backend class
 *
 * Provides complete WebGPU functionality for tensor operations:
 * - Device and adapter enumeration
 * - Buffer allocation and management
 * - Memory operations (copy, map, unmap)
 * - Queue management
 * - Shader compilation and execution
 * - Async operations support
 */
class WebGPUBackend {
public:
    /**
     * @brief Initialize WebGPU backend with configuration
     */
    explicit WebGPUBackend(const WebGPUConfig& config = WebGPUConfig());
    ~WebGPUBackend();

    // Prevent copying
    WebGPUBackend(const WebGPUBackend&) = delete;
    WebGPUBackend& operator=(const WebGPUBackend&) = delete;

    /**
     * @brief Initialize and enumerate devices
     * @return true if initialization successful
     */
    bool initialize();

    /**
     * @brief Check if backend is initialized
     */
    bool isInitialized() const { return initialized_; }

    /**
     * @brief Get device information
     */
    struct DeviceInfo {
        std::string name;
        std::string vendor;
        std::string architecture;
        WGPUAdapterType type;
        WGPUBackendType backendType;
        uint64_t maxBufferSize;
        uint32_t maxTextureDimension2D;
        uint32_t maxComputeWorkgroupSizeX;
        uint32_t maxComputeWorkgroupSizeY;
        uint32_t maxComputeWorkgroupSizeZ;
        uint32_t maxComputeInvocationsPerWorkgroup;
    };

    const DeviceInfo& getDeviceInfo() const { return deviceInfo_; }

    /**
     * @brief Buffer Management
     */

    // Create buffer with specified usage
    std::shared_ptr<WebGPUBuffer> createBuffer(size_t size, WGPUBufferUsageFlags usage);

    // Create storage buffer
    std::shared_ptr<WebGPUBuffer> createStorageBuffer(size_t size, bool readOnly = false);

    // Create uniform buffer
    std::shared_ptr<WebGPUBuffer> createUniformBuffer(size_t size);

    // Create staging buffer for CPU-GPU transfer
    std::shared_ptr<WebGPUBuffer> createStagingBuffer(size_t size);

    /**
     * @brief Memory Operations
     */

    // Write data to buffer
    void writeBuffer(WebGPUBuffer& buffer, const void* data, size_t size, size_t offset = 0);

    // Read data from buffer (async)
    std::future<std::vector<uint8_t>> readBuffer(WebGPUBuffer& buffer);

    // Copy buffer to buffer
    void copyBuffer(WebGPUBuffer& src, WebGPUBuffer& dst, size_t size,
                    size_t srcOffset = 0, size_t dstOffset = 0);

    /**
     * @brief Shader Management
     */

    // Load shader from WGSL source
    std::shared_ptr<WebGPUComputePipeline> loadShader(const std::string& shaderCode,
                                                       const std::string& entryPoint = "main");

    // Load shader from file
    std::shared_ptr<WebGPUComputePipeline> loadShaderFromFile(const std::string& filepath,
                                                                const std::string& entryPoint = "main");

    /**
     * @brief Compute Operations
     */

    // Create bind group for compute pipeline
    WGPUBindGroup createBindGroup(const WebGPUComputePipeline& pipeline,
                                  const std::vector<std::shared_ptr<WebGPUBuffer>>& buffers);

    // Execute compute shader
    void compute(const WebGPUComputePipeline& pipeline,
                 WGPUBindGroup bindGroup,
                 uint32_t workgroupX,
                 uint32_t workgroupY = 1,
                 uint32_t workgroupZ = 1);

    // Execute compute shader with multiple bind groups
    void compute(const WebGPUComputePipeline& pipeline,
                 const std::vector<WGPUBindGroup>& bindGroups,
                 uint32_t workgroupX,
                 uint32_t workgroupY = 1,
                 uint32_t workgroupZ = 1);

    /**
     * @brief Queue Management
     */

    // Submit commands and wait for completion
    void submit();

    // Wait for all operations to complete
    void waitIdle();

    // Get command encoder for custom commands
    WGPUCommandEncoder getCommandEncoder();

    // Submit custom command buffer
    void submitCommandBuffer(WGPUCommandBuffer commandBuffer);

    /**
     * @brief Synchronization
     */

    // Create fence for synchronization
    void fence();

    // Poll for async operations
    void poll();

    /**
     * @brief Resource Access
     */

    WGPUDevice getDevice() const { return device_; }
    WGPUQueue getQueue() const { return queue_; }
    WGPUAdapter getAdapter() const { return adapter_; }
    WGPUInstance getInstance() const { return instance_; }

    /**
     * @brief Error Handling
     */

    using ErrorCallback = std::function<void(WGPUErrorType, const char*)>;
    void setErrorCallback(ErrorCallback callback);

    const std::string& getLastError() const { return lastError_; }

private:
    // Initialization helpers
    bool createInstance();
    bool requestAdapter();
    bool requestDevice();
    void queryDeviceInfo();

    // Error handling
    static void handleDeviceError(WGPUErrorType type, const char* message, void* userdata);
    static void handleUncapturedError(WGPUErrorType type, const char* message, void* userdata);

    // Resource management
    void releaseResources();

    // Configuration
    WebGPUConfig config_;

    // WebGPU objects
    WGPUInstance instance_ = nullptr;
    WGPUAdapter adapter_ = nullptr;
    WGPUDevice device_ = nullptr;
    WGPUQueue queue_ = nullptr;

    // Device information
    DeviceInfo deviceInfo_;

    // State
    bool initialized_ = false;
    std::string lastError_;
    ErrorCallback errorCallback_;

    // Command encoding
    WGPUCommandEncoder currentEncoder_ = nullptr;

    // Resource tracking
    std::vector<WGPUBindGroup> activeBindGroups_;
};

/**
 * @brief RAII wrapper for command encoder
 */
class ScopedCommandEncoder {
public:
    explicit ScopedCommandEncoder(WebGPUBackend& backend);
    ~ScopedCommandEncoder();

    WGPUCommandEncoder get() const { return encoder_; }
    WGPUComputePassEncoder beginComputePass();
    void endComputePass();

private:
    WebGPUBackend& backend_;
    WGPUCommandEncoder encoder_;
    WGPUComputePassEncoder computePass_ = nullptr;
};

} // namespace webgpu
} // namespace tenzor
