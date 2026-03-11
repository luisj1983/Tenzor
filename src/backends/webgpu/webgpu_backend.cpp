#include "webgpu_backend.hpp"
#include <fstream>
#include <sstream>
#include <iostream>
#include <cstring>

namespace tenzor {
namespace webgpu {

// ============================================================================
// WebGPUBuffer Implementation
// ============================================================================

WebGPUBuffer::WebGPUBuffer(WGPUDevice device, size_t size, WGPUBufferUsageFlags usage)
    : device_(device), size_(size), usage_(usage), mapped_(false) {

    WGPUBufferDescriptor descriptor = {};
    descriptor.size = size;
    descriptor.usage = usage;
    descriptor.mappedAtCreation = false;

    buffer_ = wgpuDeviceCreateBuffer(device_, &descriptor);
    if (!buffer_) {
        throw std::runtime_error("Failed to create WebGPU buffer");
    }
}

WebGPUBuffer::~WebGPUBuffer() {
    if (buffer_) {
        if (mapped_) {
            wgpuBufferUnmap(buffer_);
        }
        wgpuBufferRelease(buffer_);
    }
}

WebGPUBuffer::WebGPUBuffer(WebGPUBuffer&& other) noexcept
    : device_(other.device_),
      buffer_(other.buffer_),
      size_(other.size_),
      usage_(other.usage_),
      mapped_(other.mapped_) {
    other.buffer_ = nullptr;
    other.mapped_ = false;
}

WebGPUBuffer& WebGPUBuffer::operator=(WebGPUBuffer&& other) noexcept {
    if (this != &other) {
        if (buffer_) {
            if (mapped_) {
                wgpuBufferUnmap(buffer_);
            }
            wgpuBufferRelease(buffer_);
        }

        device_ = other.device_;
        buffer_ = other.buffer_;
        size_ = other.size_;
        usage_ = other.usage_;
        mapped_ = other.mapped_;

        other.buffer_ = nullptr;
        other.mapped_ = false;
    }
    return *this;
}

std::future<std::vector<uint8_t>> WebGPUBuffer::readAsync() {
    auto promise = std::make_shared<std::promise<std::vector<uint8_t>>>();
    auto future = promise->get_future();

    struct CallbackData {
        std::shared_ptr<std::promise<std::vector<uint8_t>>> promise;
        WGPUBuffer buffer;
        size_t size;
    };

    auto callbackData = new CallbackData{promise, buffer_, size_};

    wgpuBufferMapAsync(
        buffer_,
        WGPUMapMode_Read,
        0,
        size_,
        [](WGPUBufferMapAsyncStatus status, void* userdata) {
            auto data = static_cast<CallbackData*>(userdata);

            if (status == WGPUBufferMapAsyncStatus_Success) {
                const uint8_t* mappedData = static_cast<const uint8_t*>(
                    wgpuBufferGetConstMappedRange(data->buffer, 0, data->size)
                );

                std::vector<uint8_t> result(mappedData, mappedData + data->size);
                data->promise->set_value(std::move(result));

                wgpuBufferUnmap(data->buffer);
            } else {
                data->promise->set_exception(
                    std::make_exception_ptr(std::runtime_error("Buffer map failed"))
                );
            }

            delete data;
        },
        callbackData
    );

    return future;
}

void WebGPUBuffer::writeAsync(const void* data, size_t size, size_t offset) {
    wgpuQueueWriteBuffer(wgpuDeviceGetQueue(device_), buffer_, offset, data, size);
}

void WebGPUBuffer::map(WGPUMapModeFlags mode, std::function<void(void*, size_t)> callback) {
    struct CallbackData {
        std::function<void(void*, size_t)> callback;
        WGPUBuffer buffer;
        size_t size;
        WGPUMapModeFlags mode;
    };

    auto callbackData = new CallbackData{callback, buffer_, size_, mode};

    wgpuBufferMapAsync(
        buffer_,
        mode,
        0,
        size_,
        [](WGPUBufferMapAsyncStatus status, void* userdata) {
            auto data = static_cast<CallbackData*>(userdata);

            if (status == WGPUBufferMapAsyncStatus_Success) {
                void* mappedData = nullptr;
                if (data->mode & WGPUMapMode_Write) {
                    mappedData = wgpuBufferGetMappedRange(data->buffer, 0, data->size);
                } else {
                    mappedData = const_cast<void*>(
                        wgpuBufferGetConstMappedRange(data->buffer, 0, data->size)
                    );
                }

                data->callback(mappedData, data->size);
            }

            delete data;
        },
        callbackData
    );

    mapped_ = true;
}

void WebGPUBuffer::unmap() {
    if (mapped_) {
        wgpuBufferUnmap(buffer_);
        mapped_ = false;
    }
}

// ============================================================================
// WebGPUComputePipeline Implementation
// ============================================================================

WebGPUComputePipeline::WebGPUComputePipeline(WGPUDevice device,
                                              const std::string& shaderCode,
                                              const std::string& entryPoint)
    : device_(device), pipeline_(nullptr), bindGroupLayout_(nullptr), shaderModule_(nullptr) {

    // Create shader module
    WGPUShaderModuleWGSLDescriptor wgslDesc = {};
    wgslDesc.chain.sType = WGPUSType_ShaderModuleWGSLDescriptor;
    wgslDesc.code = shaderCode.c_str();

    WGPUShaderModuleDescriptor moduleDesc = {};
    moduleDesc.nextInChain = &wgslDesc.chain;

    shaderModule_ = wgpuDeviceCreateShaderModule(device_, &moduleDesc);
    if (!shaderModule_) {
        throw std::runtime_error("Failed to create shader module");
    }

    // Get bind group layout from pipeline
    WGPUComputePipelineDescriptor pipelineDesc = {};
    pipelineDesc.compute.module = shaderModule_;
    pipelineDesc.compute.entryPoint = entryPoint.c_str();

    pipeline_ = wgpuDeviceCreateComputePipeline(device_, &pipelineDesc);
    if (!pipeline_) {
        wgpuShaderModuleRelease(shaderModule_);
        throw std::runtime_error("Failed to create compute pipeline");
    }

    bindGroupLayout_ = wgpuComputePipelineGetBindGroupLayout(pipeline_, 0);
}

WebGPUComputePipeline::~WebGPUComputePipeline() {
    if (bindGroupLayout_) {
        wgpuBindGroupLayoutRelease(bindGroupLayout_);
    }
    if (pipeline_) {
        wgpuComputePipelineRelease(pipeline_);
    }
    if (shaderModule_) {
        wgpuShaderModuleRelease(shaderModule_);
    }
}

WebGPUComputePipeline::WebGPUComputePipeline(WebGPUComputePipeline&& other) noexcept
    : device_(other.device_),
      pipeline_(other.pipeline_),
      bindGroupLayout_(other.bindGroupLayout_),
      shaderModule_(other.shaderModule_) {
    other.pipeline_ = nullptr;
    other.bindGroupLayout_ = nullptr;
    other.shaderModule_ = nullptr;
}

WebGPUComputePipeline& WebGPUComputePipeline::operator=(WebGPUComputePipeline&& other) noexcept {
    if (this != &other) {
        if (bindGroupLayout_) {
            wgpuBindGroupLayoutRelease(bindGroupLayout_);
        }
        if (pipeline_) {
            wgpuComputePipelineRelease(pipeline_);
        }
        if (shaderModule_) {
            wgpuShaderModuleRelease(shaderModule_);
        }

        device_ = other.device_;
        pipeline_ = other.pipeline_;
        bindGroupLayout_ = other.bindGroupLayout_;
        shaderModule_ = other.shaderModule_;

        other.pipeline_ = nullptr;
        other.bindGroupLayout_ = nullptr;
        other.shaderModule_ = nullptr;
    }
    return *this;
}

// ============================================================================
// WebGPUBackend Implementation
// ============================================================================

WebGPUBackend::WebGPUBackend(const WebGPUConfig& config)
    : config_(config) {
}

WebGPUBackend::~WebGPUBackend() {
    releaseResources();
}

bool WebGPUBackend::initialize() {
    throw std::runtime_error(
        "WebGPU backend is experimental and not yet functional. "
        "Please use CPU, CUDA, Vulkan, or other backends.");

    if (initialized_) {
        return true;
    }

    if (!createInstance()) {
        lastError_ = "Failed to create WebGPU instance";
        return false;
    }

    if (!requestAdapter()) {
        lastError_ = "Failed to request WebGPU adapter";
        return false;
    }

    if (!requestDevice()) {
        lastError_ = "Failed to request WebGPU device";
        return false;
    }

    queryDeviceInfo();

    queue_ = wgpuDeviceGetQueue(device_);
    if (!queue_) {
        lastError_ = "Failed to get device queue";
        return false;
    }

    // Set error callbacks
    wgpuDeviceSetUncapturedErrorCallback(device_, handleUncapturedError, this);

    initialized_ = true;
    return true;
}

bool WebGPUBackend::createInstance() {
    WGPUInstanceDescriptor desc = {};
    instance_ = wgpuCreateInstance(&desc);
    return instance_ != nullptr;
}

bool WebGPUBackend::requestAdapter() {
    WGPURequestAdapterOptions options = {};
    options.powerPreference = config_.powerPreference;

    struct UserData {
        WGPUAdapter adapter;
        bool requestEnded;
    };

    UserData userData = {nullptr, false};

    wgpuInstanceRequestAdapter(
        instance_,
        &options,
        [](WGPURequestAdapterStatus status, WGPUAdapter adapter,
           const char* message, void* userdata) {
            auto data = static_cast<UserData*>(userdata);
            if (status == WGPURequestAdapterStatus_Success) {
                data->adapter = adapter;
            }
            data->requestEnded = true;
        },
        &userData
    );

    // Poll until request completes
    while (!userData.requestEnded) {
        #ifdef __EMSCRIPTEN__
        emscripten_sleep(1);
        #endif
    }

    adapter_ = userData.adapter;
    return adapter_ != nullptr;
}

bool WebGPUBackend::requestDevice() {
    WGPUDeviceDescriptor desc = {};

    // Set required features
    std::vector<WGPUFeatureName> requiredFeatures;
    if (config_.requireTimestampQuery) {
        requiredFeatures.push_back(WGPUFeatureName_TimestampQuery);
    }
    if (config_.requireF16) {
        requiredFeatures.push_back(WGPUFeatureName_ShaderF16);
    }

    desc.requiredFeatureCount = requiredFeatures.size();
    desc.requiredFeatures = requiredFeatures.data();

    // Set limits
    WGPURequiredLimits limits = {};
    limits.limits.maxBufferSize = config_.maxBufferSize;
    limits.limits.maxComputeWorkgroupSizeX = config_.maxWorkgroupSizeX;
    limits.limits.maxComputeWorkgroupSizeY = config_.maxWorkgroupSizeY;
    limits.limits.maxComputeWorkgroupSizeZ = config_.maxWorkgroupSizeZ;
    limits.limits.maxComputeInvocationsPerWorkgroup = config_.maxWorkgroupInvocations;
    desc.requiredLimits = &limits;

    struct UserData {
        WGPUDevice device;
        bool requestEnded;
    };

    UserData userData = {nullptr, false};

    wgpuAdapterRequestDevice(
        adapter_,
        &desc,
        [](WGPURequestDeviceStatus status, WGPUDevice device,
           const char* message, void* userdata) {
            auto data = static_cast<UserData*>(userdata);
            if (status == WGPURequestDeviceStatus_Success) {
                data->device = device;
            }
            data->requestEnded = true;
        },
        &userData
    );

    // Poll until request completes
    while (!userData.requestEnded) {
        #ifdef __EMSCRIPTEN__
        emscripten_sleep(1);
        #endif
    }

    device_ = userData.device;
    return device_ != nullptr;
}

void WebGPUBackend::queryDeviceInfo() {
    WGPUAdapterProperties properties = {};
    wgpuAdapterGetProperties(adapter_, &properties);

    deviceInfo_.name = properties.name ? properties.name : "Unknown";
    deviceInfo_.vendor = properties.vendorName ? properties.vendorName : "Unknown";
    deviceInfo_.architecture = properties.architecture ? properties.architecture : "Unknown";
    deviceInfo_.type = properties.adapterType;
    deviceInfo_.backendType = properties.backendType;

    WGPUSupportedLimits limits = {};
    wgpuDeviceGetLimits(device_, &limits);

    deviceInfo_.maxBufferSize = limits.limits.maxBufferSize;
    deviceInfo_.maxTextureDimension2D = limits.limits.maxTextureDimension2D;
    deviceInfo_.maxComputeWorkgroupSizeX = limits.limits.maxComputeWorkgroupSizeX;
    deviceInfo_.maxComputeWorkgroupSizeY = limits.limits.maxComputeWorkgroupSizeY;
    deviceInfo_.maxComputeWorkgroupSizeZ = limits.limits.maxComputeWorkgroupSizeZ;
    deviceInfo_.maxComputeInvocationsPerWorkgroup = limits.limits.maxComputeInvocationsPerWorkgroup;
}

std::shared_ptr<WebGPUBuffer> WebGPUBackend::createBuffer(size_t size, WGPUBufferUsageFlags usage) {
    return std::make_shared<WebGPUBuffer>(device_, size, usage);
}

std::shared_ptr<WebGPUBuffer> WebGPUBackend::createStorageBuffer(size_t size, bool readOnly) {
    WGPUBufferUsageFlags usage = WGPUBufferUsage_Storage | WGPUBufferUsage_CopyDst | WGPUBufferUsage_CopySrc;
    return createBuffer(size, usage);
}

std::shared_ptr<WebGPUBuffer> WebGPUBackend::createUniformBuffer(size_t size) {
    WGPUBufferUsageFlags usage = WGPUBufferUsage_Uniform | WGPUBufferUsage_CopyDst;
    return createBuffer(size, usage);
}

std::shared_ptr<WebGPUBuffer> WebGPUBackend::createStagingBuffer(size_t size) {
    WGPUBufferUsageFlags usage = WGPUBufferUsage_MapRead | WGPUBufferUsage_CopyDst;
    return createBuffer(size, usage);
}

void WebGPUBackend::writeBuffer(WebGPUBuffer& buffer, const void* data, size_t size, size_t offset) {
    wgpuQueueWriteBuffer(queue_, buffer.get(), offset, data, size);
}

std::future<std::vector<uint8_t>> WebGPUBackend::readBuffer(WebGPUBuffer& buffer) {
    return buffer.readAsync();
}

void WebGPUBackend::copyBuffer(WebGPUBuffer& src, WebGPUBuffer& dst, size_t size,
                                size_t srcOffset, size_t dstOffset) {
    auto encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    wgpuCommandEncoderCopyBufferToBuffer(encoder, src.get(), srcOffset,
                                         dst.get(), dstOffset, size);

    WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(encoder);
}

std::shared_ptr<WebGPUComputePipeline> WebGPUBackend::loadShader(const std::string& shaderCode,
                                                                   const std::string& entryPoint) {
    return std::make_shared<WebGPUComputePipeline>(device_, shaderCode, entryPoint);
}

std::shared_ptr<WebGPUComputePipeline> WebGPUBackend::loadShaderFromFile(const std::string& filepath,
                                                                           const std::string& entryPoint) {
    std::ifstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open shader file: " + filepath);
    }

    std::stringstream buffer;
    buffer << file.rdbuf();
    return loadShader(buffer.str(), entryPoint);
}

WGPUBindGroup WebGPUBackend::createBindGroup(const WebGPUComputePipeline& pipeline,
                                              const std::vector<std::shared_ptr<WebGPUBuffer>>& buffers) {
    std::vector<WGPUBindGroupEntry> entries;
    entries.reserve(buffers.size());

    for (size_t i = 0; i < buffers.size(); ++i) {
        WGPUBindGroupEntry entry = {};
        entry.binding = i;
        entry.buffer = buffers[i]->get();
        entry.offset = 0;
        entry.size = buffers[i]->size();
        entries.push_back(entry);
    }

    WGPUBindGroupDescriptor desc = {};
    desc.layout = pipeline.getBindGroupLayout();
    desc.entryCount = entries.size();
    desc.entries = entries.data();

    return wgpuDeviceCreateBindGroup(device_, &desc);
}

void WebGPUBackend::compute(const WebGPUComputePipeline& pipeline,
                             WGPUBindGroup bindGroup,
                             uint32_t workgroupX,
                             uint32_t workgroupY,
                             uint32_t workgroupZ) {
    auto encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    auto computePass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);

    wgpuComputePassEncoderSetPipeline(computePass, pipeline.get());
    wgpuComputePassEncoderSetBindGroup(computePass, 0, bindGroup, 0, nullptr);
    wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroupX, workgroupY, workgroupZ);
    wgpuComputePassEncoderEnd(computePass);

    WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuComputePassEncoderRelease(computePass);
    wgpuCommandEncoderRelease(encoder);
}

void WebGPUBackend::compute(const WebGPUComputePipeline& pipeline,
                             const std::vector<WGPUBindGroup>& bindGroups,
                             uint32_t workgroupX,
                             uint32_t workgroupY,
                             uint32_t workgroupZ) {
    auto encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    auto computePass = wgpuCommandEncoderBeginComputePass(encoder, nullptr);

    wgpuComputePassEncoderSetPipeline(computePass, pipeline.get());

    for (size_t i = 0; i < bindGroups.size(); ++i) {
        wgpuComputePassEncoderSetBindGroup(computePass, i, bindGroups[i], 0, nullptr);
    }

    wgpuComputePassEncoderDispatchWorkgroups(computePass, workgroupX, workgroupY, workgroupZ);
    wgpuComputePassEncoderEnd(computePass);

    WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);

    wgpuCommandBufferRelease(commandBuffer);
    wgpuComputePassEncoderRelease(computePass);
    wgpuCommandEncoderRelease(encoder);
}

void WebGPUBackend::submit() {
    if (currentEncoder_) {
        WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(currentEncoder_, nullptr);
        wgpuQueueSubmit(queue_, 1, &commandBuffer);
        wgpuCommandBufferRelease(commandBuffer);
        wgpuCommandEncoderRelease(currentEncoder_);
        currentEncoder_ = nullptr;
    }
}

void WebGPUBackend::waitIdle() {
    submit();

    // Submit empty command to ensure completion
    auto encoder = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    WGPUCommandBuffer commandBuffer = wgpuCommandEncoderFinish(encoder, nullptr);
    wgpuQueueSubmit(queue_, 1, &commandBuffer);
    wgpuCommandBufferRelease(commandBuffer);
    wgpuCommandEncoderRelease(encoder);

    // Poll device
    #ifdef __EMSCRIPTEN__
    emscripten_sleep(1);
    #else
    wgpuDeviceTick(device_);
    #endif
}

WGPUCommandEncoder WebGPUBackend::getCommandEncoder() {
    if (!currentEncoder_) {
        currentEncoder_ = wgpuDeviceCreateCommandEncoder(device_, nullptr);
    }
    return currentEncoder_;
}

void WebGPUBackend::submitCommandBuffer(WGPUCommandBuffer commandBuffer) {
    wgpuQueueSubmit(queue_, 1, &commandBuffer);
}

void WebGPUBackend::fence() {
    waitIdle();
}

void WebGPUBackend::poll() {
    #ifdef __EMSCRIPTEN__
    // In Emscripten, polling is handled automatically
    #else
    wgpuDeviceTick(device_);
    #endif
}

void WebGPUBackend::setErrorCallback(ErrorCallback callback) {
    errorCallback_ = callback;
}

void WebGPUBackend::handleDeviceError(WGPUErrorType type, const char* message, void* userdata) {
    auto backend = static_cast<WebGPUBackend*>(userdata);
    backend->lastError_ = message ? message : "Unknown device error";

    if (backend->errorCallback_) {
        backend->errorCallback_(type, message);
    }
}

void WebGPUBackend::handleUncapturedError(WGPUErrorType type, const char* message, void* userdata) {
    auto backend = static_cast<WebGPUBackend*>(userdata);
    backend->lastError_ = message ? message : "Unknown uncaptured error";

    if (backend->errorCallback_) {
        backend->errorCallback_(type, message);
    }

    std::cerr << "WebGPU Uncaptured Error: " << message << std::endl;
}

void WebGPUBackend::releaseResources() {
    // Release active bind groups
    for (auto bindGroup : activeBindGroups_) {
        wgpuBindGroupRelease(bindGroup);
    }
    activeBindGroups_.clear();

    if (currentEncoder_) {
        wgpuCommandEncoderRelease(currentEncoder_);
        currentEncoder_ = nullptr;
    }

    if (queue_) {
        wgpuQueueRelease(queue_);
        queue_ = nullptr;
    }

    if (device_) {
        wgpuDeviceRelease(device_);
        device_ = nullptr;
    }

    if (adapter_) {
        wgpuAdapterRelease(adapter_);
        adapter_ = nullptr;
    }

    if (instance_) {
        wgpuInstanceRelease(instance_);
        instance_ = nullptr;
    }

    initialized_ = false;
}

// ============================================================================
// ScopedCommandEncoder Implementation
// ============================================================================

ScopedCommandEncoder::ScopedCommandEncoder(WebGPUBackend& backend)
    : backend_(backend), encoder_(backend.getCommandEncoder()) {
}

ScopedCommandEncoder::~ScopedCommandEncoder() {
    if (computePass_) {
        wgpuComputePassEncoderEnd(computePass_);
        wgpuComputePassEncoderRelease(computePass_);
    }
}

WGPUComputePassEncoder ScopedCommandEncoder::beginComputePass() {
    if (!computePass_) {
        computePass_ = wgpuCommandEncoderBeginComputePass(encoder_, nullptr);
    }
    return computePass_;
}

void ScopedCommandEncoder::endComputePass() {
    if (computePass_) {
        wgpuComputePassEncoderEnd(computePass_);
        wgpuComputePassEncoderRelease(computePass_);
        computePass_ = nullptr;
    }
}

} // namespace webgpu
} // namespace tenzor
