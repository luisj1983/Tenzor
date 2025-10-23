#pragma once

#include "webgpu_backend.hpp"
#include <vector>
#include <string>
#include <memory>

namespace tenzor {
namespace webgpu {
namespace utils {

/**
 * @brief Pipeline builder for easier pipeline creation
 */
class PipelineBuilder {
public:
    explicit PipelineBuilder(WGPUDevice device);

    PipelineBuilder& setShaderCode(const std::string& code);
    PipelineBuilder& setEntryPoint(const std::string& entryPoint);
    PipelineBuilder& addBindGroupLayoutEntry(uint32_t binding, WGPUShaderStageFlags visibility,
                                              WGPUBufferBindingType type);

    std::shared_ptr<WebGPUComputePipeline> build();

private:
    WGPUDevice device_;
    std::string shaderCode_;
    std::string entryPoint_ = "main";
    std::vector<WGPUBindGroupLayoutEntry> layoutEntries_;
};

/**
 * @brief Bind group builder for easier bind group creation
 */
class BindGroupBuilder {
public:
    explicit BindGroupBuilder(WGPUDevice device, WGPUBindGroupLayout layout);

    BindGroupBuilder& addBuffer(uint32_t binding, WGPUBuffer buffer, size_t size, size_t offset = 0);
    BindGroupBuilder& addStorageBuffer(uint32_t binding, const WebGPUBuffer& buffer);
    BindGroupBuilder& addUniformBuffer(uint32_t binding, const WebGPUBuffer& buffer);

    WGPUBindGroup build();

private:
    WGPUDevice device_;
    WGPUBindGroupLayout layout_;
    std::vector<WGPUBindGroupEntry> entries_;
};

/**
 * @brief Compute pass helper for easier command encoding
 */
class ComputePassHelper {
public:
    ComputePassHelper(WGPUCommandEncoder encoder);
    ~ComputePassHelper();

    void setPipeline(WGPUComputePipeline pipeline);
    void setBindGroup(uint32_t index, WGPUBindGroup bindGroup);
    void dispatch(uint32_t x, uint32_t y = 1, uint32_t z = 1);
    void end();

private:
    WGPUCommandEncoder encoder_;
    WGPUComputePassEncoder pass_;
    bool ended_ = false;
};

/**
 * @brief Calculate optimal workgroup sizes for browser compatibility
 */
struct WorkgroupSize {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

WorkgroupSize calculateWorkgroupSize(uint32_t totalX, uint32_t totalY = 1, uint32_t totalZ = 1,
                                      uint32_t maxWorkgroupSize = 256);

/**
 * @brief Calculate number of workgroups needed
 */
struct WorkgroupCount {
    uint32_t x;
    uint32_t y;
    uint32_t z;
};

WorkgroupCount calculateWorkgroupCount(uint32_t totalX, uint32_t totalY, uint32_t totalZ,
                                        uint32_t workgroupSizeX, uint32_t workgroupSizeY,
                                        uint32_t workgroupSizeZ);

/**
 * @brief Align size to multiple of alignment
 */
inline size_t alignSize(size_t size, size_t alignment) {
    return (size + alignment - 1) / alignment * alignment;
}

/**
 * @brief Get bytes per element for data type
 */
inline size_t getBytesPerElement(const std::string& dtype) {
    if (dtype == "float32" || dtype == "int32" || dtype == "uint32") return 4;
    if (dtype == "float16" || dtype == "int16" || dtype == "uint16") return 2;
    if (dtype == "int8" || dtype == "uint8") return 1;
    if (dtype == "float64" || dtype == "int64" || dtype == "uint64") return 8;
    return 4; // default to float32
}

/**
 * @brief Create shader module from WGSL code
 */
WGPUShaderModule createShaderModule(WGPUDevice device, const std::string& code,
                                    const std::string& label = "");

/**
 * @brief Create compute pipeline
 */
WGPUComputePipeline createComputePipeline(WGPUDevice device, WGPUShaderModule module,
                                          const std::string& entryPoint = "main",
                                          WGPUPipelineLayout layout = nullptr);

/**
 * @brief Create buffer with data
 */
WGPUBuffer createBufferWithData(WGPUDevice device, WGPUQueue queue,
                                const void* data, size_t size,
                                WGPUBufferUsageFlags usage);

/**
 * @brief Read buffer data (blocking)
 */
std::vector<uint8_t> readBufferData(WGPUDevice device, WGPUBuffer buffer, size_t size);

/**
 * @brief Shader cache for compiled shaders
 */
class ShaderCache {
public:
    explicit ShaderCache(WGPUDevice device);
    ~ShaderCache();

    std::shared_ptr<WebGPUComputePipeline> getOrCreate(const std::string& shaderPath,
                                                         const std::string& entryPoint = "main");

    void clear();

private:
    WGPUDevice device_;
    std::unordered_map<std::string, std::shared_ptr<WebGPUComputePipeline>> cache_;
};

/**
 * @brief Uniform buffer helpers
 */
template<typename T>
class UniformBuffer {
public:
    UniformBuffer(WebGPUBackend& backend, const T& data)
        : backend_(backend) {
        // Align to 256 bytes for uniform buffers
        size_t alignedSize = alignSize(sizeof(T), 256);
        buffer_ = backend_.createUniformBuffer(alignedSize);
        update(data);
    }

    void update(const T& data) {
        backend_.writeBuffer(*buffer_, &data, sizeof(T));
    }

    WebGPUBuffer& getBuffer() { return *buffer_; }
    const WebGPUBuffer& getBuffer() const { return *buffer_; }

private:
    WebGPUBackend& backend_;
    std::shared_ptr<WebGPUBuffer> buffer_;
};

/**
 * @brief Storage buffer helpers
 */
template<typename T>
class StorageBuffer {
public:
    StorageBuffer(WebGPUBackend& backend, size_t count)
        : backend_(backend), count_(count) {
        buffer_ = backend_.createStorageBuffer(sizeof(T) * count);
    }

    void write(const std::vector<T>& data) {
        if (data.size() > count_) {
            throw std::runtime_error("Data size exceeds buffer capacity");
        }
        backend_.writeBuffer(*buffer_, data.data(), sizeof(T) * data.size());
    }

    std::future<std::vector<T>> read() {
        auto future = backend_.readBuffer(*buffer_);

        return std::async(std::launch::deferred, [future = std::move(future)]() mutable {
            auto bytes = future.get();
            std::vector<T> result(bytes.size() / sizeof(T));
            std::memcpy(result.data(), bytes.data(), bytes.size());
            return result;
        });
    }

    WebGPUBuffer& getBuffer() { return *buffer_; }
    const WebGPUBuffer& getBuffer() const { return *buffer_; }

    size_t count() const { return count_; }
    size_t size() const { return sizeof(T) * count_; }

private:
    WebGPUBackend& backend_;
    std::shared_ptr<WebGPUBuffer> buffer_;
    size_t count_;
};

/**
 * @brief Dispatch helper with automatic workgroup calculation
 */
class DispatchHelper {
public:
    static void dispatch1D(WebGPUBackend& backend,
                          const WebGPUComputePipeline& pipeline,
                          WGPUBindGroup bindGroup,
                          uint32_t count,
                          uint32_t workgroupSize = 256);

    static void dispatch2D(WebGPUBackend& backend,
                          const WebGPUComputePipeline& pipeline,
                          WGPUBindGroup bindGroup,
                          uint32_t width,
                          uint32_t height,
                          uint32_t workgroupSizeX = 16,
                          uint32_t workgroupSizeY = 16);

    static void dispatch3D(WebGPUBackend& backend,
                          const WebGPUComputePipeline& pipeline,
                          WGPUBindGroup bindGroup,
                          uint32_t width,
                          uint32_t height,
                          uint32_t depth,
                          uint32_t workgroupSizeX = 8,
                          uint32_t workgroupSizeY = 8,
                          uint32_t workgroupSizeZ = 8);
};

/**
 * @brief Timing utilities for profiling
 */
class TimingHelper {
public:
    explicit TimingHelper(WGPUDevice device);
    ~TimingHelper();

    void begin();
    void end();
    double getElapsedMs();

private:
    WGPUDevice device_;
    WGPUQuerySet querySet_ = nullptr;
    WGPUBuffer timestampBuffer_ = nullptr;
    bool supported_ = false;
};

/**
 * @brief Resource pool for buffer reuse
 */
class BufferPool {
public:
    explicit BufferPool(WebGPUBackend& backend);
    ~BufferPool();

    std::shared_ptr<WebGPUBuffer> acquire(size_t size, WGPUBufferUsageFlags usage);
    void release(std::shared_ptr<WebGPUBuffer> buffer);
    void clear();

private:
    WebGPUBackend& backend_;
    struct PoolEntry {
        size_t size;
        WGPUBufferUsageFlags usage;
        std::shared_ptr<WebGPUBuffer> buffer;
    };
    std::vector<PoolEntry> pool_;
};

} // namespace utils
} // namespace webgpu
} // namespace tenzor
