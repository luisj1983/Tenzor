/**
 * @file transfer_engine.cpp
 * @brief Implementation of asynchronous CPU<->GPU transfer engine
 */

#include "tenzor/core/transfer_engine.hpp"
#include "tenzor/ops/creation.hpp"
#include <stdexcept>
#include <algorithm>
#include <cstring>
#include <chrono>
#include <iostream>

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err) \
            ); \
        } \
    } while(0)

#endif

#ifdef TENZOR_USE_ROCM
#include <hip/hip_runtime.h>

// HIP error checking macro
#define HIP_CHECK(call) \
    do { \
        hipError_t err = call; \
        if (err != hipSuccess) { \
            throw std::runtime_error( \
                std::string("HIP error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err) \
            ); \
        } \
    } while(0)

#endif

#ifdef TENZOR_USE_WEBGPU
#include <webgpu/webgpu.h>
#endif

#ifdef TENZOR_USE_VULKAN
// Vulkan error checking macro
#define VK_CHECK(call) \
    do { \
        VkResult err = call; \
        if (err != VK_SUCCESS) { \
            throw std::runtime_error( \
                std::string("Vulkan error at ") + __FILE__ + ":" + \
                std::to_string(__LINE__) + " - error code: " + std::to_string(err) \
            ); \
        } \
    } while(0)
#endif

namespace tenzor {
namespace core {

// ============================================================================
// WebGPU Implementation
// ============================================================================

#ifdef TENZOR_USE_WEBGPU
auto TransferEngine::initialize_webgpu_resources() -> void {
    if (wgpu_device_) {
        wgpu_queue_ = wgpuDeviceGetQueue(wgpu_device_);
    }
}

auto TransferEngine::cleanup_webgpu_resources() -> void {
    if (wgpu_queue_) {
        wgpuQueueRelease(wgpu_queue_);
        wgpu_queue_ = nullptr;
    }
}
#endif

// ============================================================================
// OneAPI/SYCL Implementation
// ============================================================================

#ifdef TENZOR_USE_ONEAPI
auto TransferEngine::initialize_oneapi_resources() -> void {
    try {
        sycl::device dev = sycl::device(sycl::gpu_selector_v);
        sycl_queues_.reserve(config_.num_streams);
        for (int i = 0; i < config_.num_streams; ++i) {
            sycl_queues_.emplace_back(dev, sycl::property::queue::in_order());
        }
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("SYCL initialization failed: ") + e.what());
    }
}

auto TransferEngine::cleanup_oneapi_resources() -> void {
    for (auto& q : sycl_queues_) {
        try {
            q.wait();
        } catch (const sycl::exception& e) {
            std::cerr << "SYCL queue wait failed during cleanup: " << e.what() << std::endl;
        }
    }
    sycl_queues_.clear();
}

auto TransferEngine::get_sycl_queue(int idx) -> sycl::queue& {
    if (idx < 0 || idx >= static_cast<int>(sycl_queues_.size())) {
        throw std::out_of_range("Invalid SYCL queue index");
    }
    return sycl_queues_[idx];
}
#endif

// ============================================================================
// Vulkan Implementation
// ============================================================================

#ifdef TENZOR_USE_VULKAN
auto TransferEngine::initialize_vulkan_resources() -> void {
    if (vk_device_ == VK_NULL_HANDLE) {
        return;
    }

    VkCommandPoolCreateInfo poolInfo{};
    poolInfo.sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO;
    poolInfo.queueFamilyIndex = vk_transfer_queue_family_index_;
    poolInfo.flags = VK_COMMAND_POOL_CREATE_RESET_COMMAND_BUFFER_BIT;
    VK_CHECK(vkCreateCommandPool(vk_device_, &poolInfo, nullptr, &vk_command_pool_));

    vk_command_buffers_.resize(config_.num_streams);
    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vk_command_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = static_cast<uint32_t>(config_.num_streams);
    VK_CHECK(vkAllocateCommandBuffers(vk_device_, &allocInfo, vk_command_buffers_.data()));

    vk_fences_.reserve(config_.num_streams * 2);
    for (int i = 0; i < config_.num_streams * 2; ++i) {
        VkFenceCreateInfo fenceInfo{};
        fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
        fenceInfo.flags = 0;
        VkFence fence;
        VK_CHECK(vkCreateFence(vk_device_, &fenceInfo, nullptr, &fence));
        vk_fences_.push_back(fence);
    }
}

auto TransferEngine::cleanup_vulkan_resources() -> void {
    if (vk_device_ == VK_NULL_HANDLE) {
        return;
    }

    vkDeviceWaitIdle(vk_device_);

    for (VkFence fence : vk_fences_) {
        vkDestroyFence(vk_device_, fence, nullptr);
    }
    vk_fences_.clear();
    vk_command_buffers_.clear();

    if (vk_command_pool_ != VK_NULL_HANDLE) {
        vkDestroyCommandPool(vk_device_, vk_command_pool_, nullptr);
        vk_command_pool_ = VK_NULL_HANDLE;
    }
}

auto TransferEngine::get_vk_fence() -> VkFence {
    std::lock_guard lock(vk_fence_mutex_);

    if (!vk_fences_.empty()) {
        VkFence fence = vk_fences_.back();
        vk_fences_.pop_back();
        return fence;
    }

    VkFenceCreateInfo fenceInfo{};
    fenceInfo.sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO;
    fenceInfo.flags = 0;
    VkFence fence;
    VK_CHECK(vkCreateFence(vk_device_, &fenceInfo, nullptr, &fence));
    return fence;
}

auto TransferEngine::return_vk_fence(VkFence fence) -> void {
    if (fence == VK_NULL_HANDLE) return;
    std::lock_guard lock(vk_fence_mutex_);
    vk_fences_.push_back(fence);
}

auto TransferEngine::get_vk_command_buffer() -> VkCommandBuffer {
    std::lock_guard lock(vk_command_buffer_mutex_);

    if (!vk_command_buffers_.empty()) {
        VkCommandBuffer cmd = vk_command_buffers_.back();
        vk_command_buffers_.pop_back();
        return cmd;
    }

    VkCommandBufferAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO;
    allocInfo.commandPool = vk_command_pool_;
    allocInfo.level = VK_COMMAND_BUFFER_LEVEL_PRIMARY;
    allocInfo.commandBufferCount = 1;
    VkCommandBuffer cmd;
    VK_CHECK(vkAllocateCommandBuffers(vk_device_, &allocInfo, &cmd));
    return cmd;
}

auto TransferEngine::return_vk_command_buffer(VkCommandBuffer cmd_buffer) -> void {
    if (cmd_buffer == VK_NULL_HANDLE) return;
    std::lock_guard lock(vk_command_buffer_mutex_);
    vkResetCommandBuffer(cmd_buffer, 0);
    vk_command_buffers_.push_back(cmd_buffer);
}

auto TransferEngine::find_memory_type(uint32_t type_filter, VkMemoryPropertyFlags properties) -> uint32_t {
    VkPhysicalDeviceMemoryProperties memProperties;
    vkGetPhysicalDeviceMemoryProperties(vk_physical_device_, &memProperties);

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if ((type_filter & (1 << i)) &&
            (memProperties.memoryTypes[i].propertyFlags & properties) == properties) {
            return i;
        }
    }

    throw std::runtime_error("Failed to find suitable Vulkan memory type");
}

auto TransferEngine::create_staging_buffer(VkDeviceSize size, VkBuffer& buffer, VkDeviceMemory& memory) -> void {
    VkBufferCreateInfo bufferInfo{};
    bufferInfo.sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO;
    bufferInfo.size = size;
    bufferInfo.usage = VK_BUFFER_USAGE_TRANSFER_SRC_BIT | VK_BUFFER_USAGE_TRANSFER_DST_BIT;
    bufferInfo.sharingMode = VK_SHARING_MODE_EXCLUSIVE;

    VK_CHECK(vkCreateBuffer(vk_device_, &bufferInfo, nullptr, &buffer));

    VkMemoryRequirements memRequirements;
    vkGetBufferMemoryRequirements(vk_device_, buffer, &memRequirements);

    VkMemoryAllocateInfo allocInfo{};
    allocInfo.sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO;
    allocInfo.allocationSize = memRequirements.size;
    allocInfo.memoryTypeIndex = find_memory_type(
        memRequirements.memoryTypeBits,
        VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT | VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
    );

    VK_CHECK(vkAllocateMemory(vk_device_, &allocInfo, nullptr, &memory));
    VK_CHECK(vkBindBufferMemory(vk_device_, buffer, memory, 0));
}
#endif

// ============================================================================
// TransferState Implementation
// ============================================================================

TransferState::~TransferState() {
#ifdef TENZOR_USE_CUDA
    if (event && engine) {
        engine->return_event(event);
    }
#endif

#ifdef TENZOR_USE_ROCM
    if (hip_event && engine) {
        engine->return_hip_event(hip_event);
    }
#endif

    if (pinned_buffer && engine) {
        engine->return_pinned_buffer(pinned_buffer);
    }

#ifdef TENZOR_USE_WEBGPU
    if (wgpu_staging_buffer) {
        wgpuBufferRelease(wgpu_staging_buffer);
        wgpu_staging_buffer = nullptr;
    }
#endif

#ifdef TENZOR_USE_VULKAN
    if (engine && has_vulkan_transfer) {
        if (vk_fence != VK_NULL_HANDLE) {
            engine->return_vk_fence(vk_fence);
        }
        if (vk_command_buffer != VK_NULL_HANDLE) {
            engine->return_vk_command_buffer(vk_command_buffer);
        }
        if (vk_staging_buffer != VK_NULL_HANDLE && engine->vk_device_ != VK_NULL_HANDLE) {
            vkDestroyBuffer(engine->vk_device_, vk_staging_buffer, nullptr);
        }
        if (vk_staging_memory != VK_NULL_HANDLE && engine->vk_device_ != VK_NULL_HANDLE) {
            vkFreeMemory(engine->vk_device_, vk_staging_memory, nullptr);
        }
    }
#endif
}

// ============================================================================
// TransferHandle Implementation
// ============================================================================

TransferHandle::TransferHandle(std::shared_ptr<TransferState> state)
    : state_(std::move(state)) {}

auto TransferHandle::is_ready() const -> bool {
    if (!state_) {
        return true;
    }

    if (state_->completed.load(std::memory_order_acquire)) {
        return true;
    }

#ifdef TENZOR_USE_CUDA
    if (state_->event) {
        cudaError_t result = cudaEventQuery(state_->event);
        if (result == cudaSuccess) {
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
        if (result != cudaErrorNotReady) {
            CUDA_CHECK(result);
        }
    }
#endif

#ifdef TENZOR_USE_ROCM
    if (state_->hip_event) {
        hipError_t result = hipEventQuery(state_->hip_event);
        if (result == hipSuccess) {
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
        if (result != hipErrorNotReady) {
            HIP_CHECK(result);
        }
    }
#endif

#ifdef TENZOR_USE_ONEAPI
    if (state_->has_sycl_event) {
        auto status = state_->sycl_event.get_info<sycl::info::event::command_execution_status>();
        if (status == sycl::info::event_command_status::complete) {
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
    }
#endif

#ifdef TENZOR_USE_VULKAN
    if (state_->has_vulkan_transfer && state_->vk_fence != VK_NULL_HANDLE && state_->engine) {
        VkResult result = vkGetFenceStatus(state_->engine->vk_device_, state_->vk_fence);
        if (result == VK_SUCCESS) {
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
    }
#endif

    return false;
}

auto TransferHandle::wait() -> void {
    if (!state_) {
        return;
    }

    if (state_->completed.load(std::memory_order_acquire)) {
        return;
    }

#ifdef TENZOR_USE_CUDA
    if (state_->event) {
        CUDA_CHECK(cudaEventSynchronize(state_->event));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

#ifdef TENZOR_USE_ROCM
    if (state_->hip_event) {
        HIP_CHECK(hipEventSynchronize(state_->hip_event));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

#ifdef TENZOR_USE_ONEAPI
    if (state_->has_sycl_event) {
        state_->sycl_event.wait();
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

#ifdef TENZOR_USE_VULKAN
    if (state_->has_vulkan_transfer && state_->vk_fence != VK_NULL_HANDLE && state_->engine) {
        VK_CHECK(vkWaitForFences(state_->engine->vk_device_, 1, &state_->vk_fence, VK_TRUE, UINT64_MAX));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

    std::unique_lock lock(state_->mutex);
    state_->cv.wait(lock, [this] {
        return state_->completed.load(std::memory_order_acquire)
#ifdef TENZOR_USE_CUDA
               || (state_->event != nullptr)
#endif
#ifdef TENZOR_USE_ROCM
               || (state_->hip_event != nullptr)
#endif
#ifdef TENZOR_USE_ONEAPI
               || state_->has_sycl_event
#endif
               ;
    });

#ifdef TENZOR_USE_CUDA
    if (state_->event) {
        lock.unlock();
        CUDA_CHECK(cudaEventSynchronize(state_->event));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

#ifdef TENZOR_USE_ROCM
    if (state_->hip_event) {
        lock.unlock();
        HIP_CHECK(hipEventSynchronize(state_->hip_event));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

#ifdef TENZOR_USE_ONEAPI
    if (state_->has_sycl_event) {
        lock.unlock();
        state_->sycl_event.wait();
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
    }
#endif
}

auto TransferHandle::get_tensor() -> Tensor {
    if (!state_) {
        throw std::runtime_error("Invalid transfer handle");
    }

    wait();

    if (state_->has_error) {
        throw std::runtime_error("Transfer failed: " + state_->error_message);
    }

    return state_->result;
}

// ============================================================================
// TransferEngine Implementation
// ============================================================================

TransferEngine::TransferEngine(const Config& config)
    : config_(config) {
    if (config_.num_streams <= 0) {
        throw std::invalid_argument("num_streams must be positive");
    }

    if (config_.queue_capacity == 0) {
        throw std::invalid_argument("queue_capacity must be positive");
    }

    initialize_cuda_resources();
    initialize_rocm_resources();

#ifdef TENZOR_USE_WEBGPU
    initialize_webgpu_resources();
#endif

#ifdef TENZOR_USE_ONEAPI
    initialize_oneapi_resources();
#endif

#ifdef TENZOR_USE_VULKAN
    initialize_vulkan_resources();
#endif

#ifdef TENZOR_USE_METAL
    initialize_metal_resources();
#endif

    worker_thread_ = std::thread(&TransferEngine::transfer_worker, this);
}

TransferEngine::~TransferEngine() {
    {
        std::lock_guard lock(queue_mutex_);
        stop_worker_.store(true, std::memory_order_release);
    }
    queue_cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    cleanup_cuda_resources();
    cleanup_rocm_resources();

#ifdef TENZOR_USE_WEBGPU
    cleanup_webgpu_resources();
#endif

#ifdef TENZOR_USE_ONEAPI
    cleanup_oneapi_resources();
#endif

#ifdef TENZOR_USE_VULKAN
    cleanup_vulkan_resources();
#endif

#ifdef TENZOR_USE_METAL
    cleanup_metal_resources();
#endif
}

auto TransferEngine::initialize_cuda_resources() -> void {
#ifdef TENZOR_USE_CUDA
    streams_.reserve(config_.num_streams);
    for (int i = 0; i < config_.num_streams; ++i) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        streams_.push_back(stream);
    }

    event_pool_.reserve(config_.num_streams * 2);
    for (int i = 0; i < config_.num_streams * 2; ++i) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        event_pool_.push_back(event);
    }

    if (config_.use_pinned_memory && config_.pinned_pool_size > 0) {
        std::vector<size_t> buffer_sizes = {
            1 * 1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024
        };

        size_t total_allocated = 0;
        for (size_t size : buffer_sizes) {
            if (total_allocated + size > config_.pinned_pool_size) {
                break;
            }

            void* ptr = nullptr;
            cudaError_t err = cudaMallocHost(&ptr, size);
            if (err == cudaSuccess) {
                pinned_buffers_.push_back({ptr, size, false});
                total_allocated += size;
            }
        }
    }
#endif
}

auto TransferEngine::cleanup_cuda_resources() -> void {
#ifdef TENZOR_USE_CUDA
    for (cudaStream_t stream : streams_) {
        cudaStreamDestroy(stream);
    }
    streams_.clear();

    for (cudaEvent_t event : event_pool_) {
        cudaEventDestroy(event);
    }
    event_pool_.clear();

    for (auto& buffer : pinned_buffers_) {
        if (buffer.ptr) {
            cudaFreeHost(buffer.ptr);
        }
    }
    pinned_buffers_.clear();
#endif
}

auto TransferEngine::initialize_rocm_resources() -> void {
#ifdef TENZOR_USE_ROCM
    hip_streams_.reserve(config_.num_streams);
    for (int i = 0; i < config_.num_streams; ++i) {
        hipStream_t stream;
        HIP_CHECK(hipStreamCreate(&stream));
        hip_streams_.push_back(stream);
    }

    hip_event_pool_.reserve(config_.num_streams * 2);
    for (int i = 0; i < config_.num_streams * 2; ++i) {
        hipEvent_t event;
        HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
        hip_event_pool_.push_back(event);
    }

#ifndef TENZOR_USE_CUDA
    if (config_.use_pinned_memory && config_.pinned_pool_size > 0) {
        std::vector<size_t> buffer_sizes = {
            1 * 1024 * 1024,
            4 * 1024 * 1024,
            16 * 1024 * 1024,
            64 * 1024 * 1024
        };

        size_t total_allocated = 0;
        for (size_t size : buffer_sizes) {
            if (total_allocated + size > config_.pinned_pool_size) {
                break;
            }

            void* ptr = nullptr;
            hipError_t err = hipHostMalloc(&ptr, size, hipHostMallocDefault);
            if (err == hipSuccess) {
                pinned_buffers_.push_back({ptr, size, false});
                total_allocated += size;
            }
        }
    }
#endif
#endif
}

auto TransferEngine::cleanup_rocm_resources() -> void {
#ifdef TENZOR_USE_ROCM
    for (hipStream_t stream : hip_streams_) {
        hipStreamDestroy(stream);
    }
    hip_streams_.clear();

    for (hipEvent_t event : hip_event_pool_) {
        hipEventDestroy(event);
    }
    hip_event_pool_.clear();

#ifndef TENZOR_USE_CUDA
    for (auto& buffer : pinned_buffers_) {
        if (buffer.ptr) {
            hipHostFree(buffer.ptr);
        }
    }
    pinned_buffers_.clear();
#endif
#endif
}

#ifdef TENZOR_USE_CUDA
auto TransferEngine::get_event() -> cudaEvent_t {
    std::lock_guard lock(event_pool_mutex_);

    if (!event_pool_.empty()) {
        cudaEvent_t event = event_pool_.back();
        event_pool_.pop_back();
        return event;
    }

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
    return event;
}

auto TransferEngine::return_event(cudaEvent_t event) -> void {
    if (!event) return;
    std::lock_guard lock(event_pool_mutex_);
    event_pool_.push_back(event);
}
#endif

#ifdef TENZOR_USE_ROCM
auto TransferEngine::get_hip_event() -> hipEvent_t {
    std::lock_guard lock(hip_event_pool_mutex_);

    if (!hip_event_pool_.empty()) {
        hipEvent_t event = hip_event_pool_.back();
        hip_event_pool_.pop_back();
        return event;
    }

    hipEvent_t event;
    HIP_CHECK(hipEventCreateWithFlags(&event, hipEventDisableTiming));
    return event;
}

auto TransferEngine::return_hip_event(hipEvent_t event) -> void {
    if (!event) return;
    std::lock_guard lock(hip_event_pool_mutex_);
    hip_event_pool_.push_back(event);
}
#endif

auto TransferEngine::get_pinned_buffer(size_t size) -> void* {
    std::lock_guard lock(pinned_mutex_);

    for (auto& buffer : pinned_buffers_) {
        if (!buffer.in_use && buffer.size >= size) {
            buffer.in_use = true;
            return buffer.ptr;
        }
    }

#ifdef TENZOR_USE_CUDA
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err == cudaSuccess) {
        pinned_buffers_.push_back({ptr, size, true});
        return ptr;
    }
#elif defined(TENZOR_USE_ROCM)
    void* ptr = nullptr;
    hipError_t err = hipHostMalloc(&ptr, size, hipHostMallocDefault);
    if (err == hipSuccess) {
        pinned_buffers_.push_back({ptr, size, true});
        return ptr;
    }
#endif

    return nullptr;
}

auto TransferEngine::return_pinned_buffer(void* ptr) -> void {
    if (!ptr) return;

    std::lock_guard lock(pinned_mutex_);

    for (auto& buffer : pinned_buffers_) {
        if (buffer.ptr == ptr) {
            buffer.in_use = false;
            return;
        }
    }
}

auto TransferEngine::allocate_tensor(
    const std::vector<int64_t>& shape,
    DType dtype,
    Device device
) -> Tensor {
    return Tensor(shape, dtype, device);
}

auto TransferEngine::record_transfer(
    size_t bytes,
    double time_ms,
    bool cpu_to_gpu
) -> void {
    stats_.total_transfers.fetch_add(1, std::memory_order_relaxed);
    stats_.bytes_transferred.fetch_add(bytes, std::memory_order_relaxed);

    if (cpu_to_gpu) {
        stats_.cpu_to_gpu_count.fetch_add(1, std::memory_order_relaxed);
    } else {
        stats_.gpu_to_cpu_count.fetch_add(1, std::memory_order_relaxed);
    }

    double current = stats_.total_time_ms.load(std::memory_order_relaxed);
    while (!stats_.total_time_ms.compare_exchange_weak(
        current,
        current + time_ms,
        std::memory_order_relaxed
    ));
}

// ============================================================================
// Synchronous Transfer API
// ============================================================================

auto TransferEngine::cpu_to_gpu(const Tensor& cpu_tensor, Device gpu_device) -> Tensor {
    if (cpu_tensor.device().type != Device::Type::CPU) {
        throw std::runtime_error("Source tensor must be on CPU");
    }

    if (gpu_device.type != Device::Type::CUDA && gpu_device.type != Device::Type::ROCm &&
        gpu_device.type != Device::Type::Vulkan && gpu_device.type != Device::Type::OneAPI) {
        throw std::runtime_error("Target device must be CUDA, ROCm, Vulkan, or OneAPI");
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto shape_span = cpu_tensor.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    Tensor gpu_tensor = allocate_tensor(shape_vec, cpu_tensor.dtype(), gpu_device);

    size_t bytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());

#ifdef TENZOR_USE_CUDA
    if (gpu_device.type == Device::Type::CUDA) {
        CUDA_CHECK(cudaSetDevice(gpu_device.index));
        CUDA_CHECK(cudaMemcpy(
            gpu_tensor.data_ptr(),
            cpu_tensor.data_ptr(),
            bytes,
            cudaMemcpyHostToDevice
        ));
    }
#endif

#ifdef TENZOR_USE_ROCM
    if (gpu_device.type == Device::Type::ROCm) {
        HIP_CHECK(hipSetDevice(gpu_device.index));
        HIP_CHECK(hipMemcpy(
            gpu_tensor.data_ptr(),
            cpu_tensor.data_ptr(),
            bytes,
            hipMemcpyHostToDevice
        ));
    }
#endif

    // Vulkan transfers via tensor .to() method
    if (gpu_device.type == Device::Type::Vulkan) {
        gpu_tensor = cpu_tensor.to(gpu_device);
    }

    // OneAPI uses USM shared memory - direct memcpy works via backend copy
    if (gpu_device.type == Device::Type::OneAPI) {
        gpu_tensor = cpu_tensor.to(gpu_device);
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    record_transfer(bytes, time_ms, true);

    return gpu_tensor;
}

auto TransferEngine::gpu_to_cpu(const Tensor& gpu_tensor) -> Tensor {
    if (gpu_tensor.device().type != Device::Type::CUDA && gpu_tensor.device().type != Device::Type::ROCm &&
        gpu_tensor.device().type != Device::Type::Vulkan && gpu_tensor.device().type != Device::Type::OneAPI) {
        throw std::runtime_error("Source tensor must be on CUDA, ROCm, Vulkan, or OneAPI");
    }

    auto start = std::chrono::high_resolution_clock::now();

    auto shape_span = gpu_tensor.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    Tensor cpu_tensor = allocate_tensor(
        shape_vec,
        gpu_tensor.dtype(),
        Device::cpu()
    );

    size_t bytes = gpu_tensor.numel() * dtype_size(gpu_tensor.dtype());

#ifdef TENZOR_USE_CUDA
    if (gpu_tensor.device().type == Device::Type::CUDA) {
        CUDA_CHECK(cudaSetDevice(gpu_tensor.device().index));
        CUDA_CHECK(cudaMemcpy(
            cpu_tensor.data_ptr(),
            gpu_tensor.data_ptr(),
            bytes,
            cudaMemcpyDeviceToHost
        ));
    }
#endif

#ifdef TENZOR_USE_ROCM
    if (gpu_tensor.device().type == Device::Type::ROCm) {
        HIP_CHECK(hipSetDevice(gpu_tensor.device().index));
        HIP_CHECK(hipMemcpy(
            cpu_tensor.data_ptr(),
            gpu_tensor.data_ptr(),
            bytes,
            hipMemcpyDeviceToHost
        ));
    }
#endif

    // Vulkan transfers via tensor .to() method
    if (gpu_tensor.device().type == Device::Type::Vulkan) {
        cpu_tensor = gpu_tensor.to(Device::cpu());
    }

    // OneAPI uses USM shared memory - transfer via .to() method
    if (gpu_tensor.device().type == Device::Type::OneAPI) {
        cpu_tensor = gpu_tensor.to(Device::cpu());
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    record_transfer(bytes, time_ms, false);

    return cpu_tensor;
}

// ============================================================================
// Asynchronous Transfer API
// ============================================================================

auto TransferEngine::cpu_to_gpu_async(
    const Tensor& cpu_tensor,
    Device gpu_device
) -> TransferHandle {
    if (cpu_tensor.device().type != Device::Type::CPU) {
        throw std::runtime_error("Source tensor must be on CPU");
    }

    if (gpu_device.type != Device::Type::CUDA &&
        gpu_device.type != Device::Type::ROCm &&
        gpu_device.type != Device::Type::OneAPI &&
        gpu_device.type != Device::Type::WebGPU) {
        throw std::runtime_error("Target device must be CUDA, ROCm, OneAPI, or WebGPU");
    }

    auto state = std::make_shared<TransferState>();
    state->engine = this;

    // Handle ROCm direct transfers
    if (gpu_device.type == Device::Type::ROCm) {
#ifdef TENZOR_USE_ROCM
        static std::atomic<int> hip_stream_counter{0};
        int stream_idx = hip_stream_counter.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
        hipStream_t stream = hip_streams_[stream_idx];

        HIP_CHECK(hipSetDevice(gpu_device.index));

        auto shape_span = cpu_tensor.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(shape_vec, cpu_tensor.dtype(), gpu_device);

        size_t bytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());
        const void* src_ptr = cpu_tensor.data_ptr();
        void* dst_ptr = gpu_tensor.data_ptr();

        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                std::memcpy(pinned, src_ptr, bytes);
                HIP_CHECK(hipMemcpyAsync(dst_ptr, pinned, bytes, hipMemcpyHostToDevice, stream));
                state->pinned_buffer = pinned;
            } else {
                HIP_CHECK(hipMemcpyAsync(dst_ptr, src_ptr, bytes, hipMemcpyHostToDevice, stream));
            }
        } else {
            HIP_CHECK(hipMemcpyAsync(dst_ptr, src_ptr, bytes, hipMemcpyHostToDevice, stream));
        }

        hipEvent_t event = get_hip_event();
        HIP_CHECK(hipEventRecord(event, stream));

        state->result = gpu_tensor;
        state->hip_event = event;
        state->hip_stream = stream;

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, true);

        return TransferHandle(state);
#else
        throw std::runtime_error("ROCm support not compiled");
#endif
    }

#ifdef TENZOR_USE_ONEAPI
    if (gpu_device.type == Device::Type::OneAPI) {
        int queue_idx = next_stream_.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
        auto& queue = get_sycl_queue(queue_idx);

        auto shape_span = cpu_tensor.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(shape_vec, cpu_tensor.dtype(), gpu_device);

        size_t bytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());

        try {
            sycl::event event = queue.memcpy(gpu_tensor.data_ptr(), cpu_tensor.data_ptr(), bytes);
            state->result = gpu_tensor;
            state->sycl_event = event;
            state->has_sycl_event = true;

            auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            record_transfer(bytes, time_ms, true);

            return TransferHandle(state);
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL memcpy failed: ") + e.what());
        }
    }
#endif

    // Queue transfer request for CUDA
    TransferRequest request;
    request.type = TransferRequest::Type::CPU_TO_GPU;
    request.source = cpu_tensor;
    request.target_device = gpu_device;
    request.state = state;

    {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return transfer_queue_.size() < config_.queue_capacity || stop_worker_.load();
        });

        if (stop_worker_.load()) {
            throw std::runtime_error("Transfer engine shutting down");
        }

        transfer_queue_.push(std::move(request));
    }

    queue_cv_.notify_one();

    return TransferHandle(state);
}

auto TransferEngine::gpu_to_cpu_async(const Tensor& gpu_tensor) -> TransferHandle {
    if (gpu_tensor.device().type != Device::Type::CUDA &&
        gpu_tensor.device().type != Device::Type::ROCm &&
        gpu_tensor.device().type != Device::Type::OneAPI &&
        gpu_tensor.device().type != Device::Type::WebGPU) {
        throw std::runtime_error("Source tensor must be on CUDA, ROCm, OneAPI, or WebGPU");
    }

    auto state = std::make_shared<TransferState>();
    state->engine = this;

    // Handle ROCm direct transfers
    if (gpu_tensor.device().type == Device::Type::ROCm) {
#ifdef TENZOR_USE_ROCM
        static std::atomic<int> hip_stream_counter{0};
        int stream_idx = hip_stream_counter.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
        hipStream_t stream = hip_streams_[stream_idx];

        HIP_CHECK(hipSetDevice(gpu_tensor.device().index));

        auto shape_span = gpu_tensor.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_result = allocate_tensor(shape_vec, gpu_tensor.dtype(), Device::cpu());

        size_t bytes = gpu_tensor.numel() * dtype_size(gpu_tensor.dtype());
        const void* src_ptr = gpu_tensor.data_ptr();
        void* dst_ptr = cpu_result.data_ptr();

        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                HIP_CHECK(hipMemcpyAsync(pinned, src_ptr, bytes, hipMemcpyDeviceToHost, stream));

                hipEvent_t event = get_hip_event();
                HIP_CHECK(hipEventRecord(event, stream));
                HIP_CHECK(hipEventSynchronize(event));

                std::memcpy(dst_ptr, pinned, bytes);

                return_hip_event(event);
                state->pinned_buffer = pinned;
                state->result = cpu_result;
                state->completed.store(true, std::memory_order_release);
            } else {
                HIP_CHECK(hipMemcpyAsync(dst_ptr, src_ptr, bytes, hipMemcpyDeviceToHost, stream));

                hipEvent_t event = get_hip_event();
                HIP_CHECK(hipEventRecord(event, stream));

                state->result = cpu_result;
                state->hip_event = event;
                state->hip_stream = stream;
            }
        } else {
            HIP_CHECK(hipMemcpyAsync(dst_ptr, src_ptr, bytes, hipMemcpyDeviceToHost, stream));

            hipEvent_t event = get_hip_event();
            HIP_CHECK(hipEventRecord(event, stream));

            state->result = cpu_result;
            state->hip_event = event;
            state->hip_stream = stream;
        }

        auto start = std::chrono::high_resolution_clock::now();
        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, false);

        return TransferHandle(state);
#else
        throw std::runtime_error("ROCm support not compiled");
#endif
    }

#ifdef TENZOR_USE_ONEAPI
    if (gpu_tensor.device().type == Device::Type::OneAPI) {
        int queue_idx = next_stream_.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
        auto& queue = get_sycl_queue(queue_idx);

        auto shape_span = gpu_tensor.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_result = allocate_tensor(shape_vec, gpu_tensor.dtype(), Device::cpu());

        size_t bytes = gpu_tensor.numel() * dtype_size(gpu_tensor.dtype());

        try {
            sycl::event event = queue.memcpy(cpu_result.data_ptr(), gpu_tensor.data_ptr(), bytes);
            state->result = cpu_result;
            state->sycl_event = event;
            state->has_sycl_event = true;

            auto start = std::chrono::high_resolution_clock::now();
            auto end = std::chrono::high_resolution_clock::now();
            double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
            record_transfer(bytes, time_ms, false);

            return TransferHandle(state);
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL memcpy failed: ") + e.what());
        }
    }
#endif

    // Queue transfer request for CUDA
    TransferRequest request;
    request.type = TransferRequest::Type::GPU_TO_CPU;
    request.source = gpu_tensor;
    request.target_device = Device::cpu();
    request.state = state;

    {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return transfer_queue_.size() < config_.queue_capacity || stop_worker_.load();
        });

        if (stop_worker_.load()) {
            throw std::runtime_error("Transfer engine shutting down");
        }

        transfer_queue_.push(std::move(request));
    }

    queue_cv_.notify_one();

    return TransferHandle(state);
}

// ============================================================================
// Worker Thread
// ============================================================================

auto TransferEngine::transfer_worker() -> void {
    while (!stop_worker_.load(std::memory_order_acquire)) {
        TransferRequest request;

        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !transfer_queue_.empty() ||
                       stop_worker_.load(std::memory_order_acquire);
            });

            if (stop_worker_.load(std::memory_order_acquire) && transfer_queue_.empty()) {
                break;
            }

            if (transfer_queue_.empty()) {
                continue;
            }

            request = std::move(transfer_queue_.front());
            transfer_queue_.pop();
            in_flight_transfers_.fetch_add(1, std::memory_order_release);
        }

        queue_cv_.notify_one();

        try {
            process_transfer(request);
        } catch (const std::exception& e) {
            request.state->has_error = true;
            request.state->error_message = e.what();
            request.state->completed.store(true, std::memory_order_release);
            request.state->cv.notify_all();
        }

        in_flight_transfers_.fetch_sub(1, std::memory_order_release);
        queue_cv_.notify_all();
    }
}

auto TransferEngine::process_transfer(const TransferRequest& request) -> void {
    auto start = std::chrono::high_resolution_clock::now();

#ifdef TENZOR_USE_CUDA
    static std::atomic<int> stream_counter{0};
    int stream_idx = stream_counter.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
    cudaStream_t stream = streams_[stream_idx];

    request.state->stream = stream;

    size_t bytes = request.source.numel() * dtype_size(request.source.dtype());

    if (request.type == TransferRequest::Type::CPU_TO_GPU) {
        Device gpu_device = request.target_device;
        CUDA_CHECK(cudaSetDevice(gpu_device.index));

        auto shape_span = request.source.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(shape_vec, request.source.dtype(), gpu_device);

        const void* src_ptr = request.source.data_ptr();
        void* dst_ptr = gpu_tensor.data_ptr();

        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                std::memcpy(pinned, src_ptr, bytes);
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, pinned, bytes, cudaMemcpyHostToDevice, stream));
                request.state->pinned_buffer = pinned;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream));
            }
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream));
        }

        cudaEvent_t event = get_event();
        CUDA_CHECK(cudaEventRecord(event, stream));
        request.state->event = event;
        request.state->result = gpu_tensor;

        request.state->cv.notify_all();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, true);

    } else {
        CUDA_CHECK(cudaSetDevice(request.source.device().index));

        auto shape_span = request.source.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_tensor = allocate_tensor(shape_vec, request.source.dtype(), Device::cpu());

        const void* src_ptr = request.source.data_ptr();
        void* dst_ptr = cpu_tensor.data_ptr();

        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                CUDA_CHECK(cudaMemcpyAsync(pinned, src_ptr, bytes, cudaMemcpyDeviceToHost, stream));

                cudaEvent_t event = get_event();
                CUDA_CHECK(cudaEventRecord(event, stream));
                CUDA_CHECK(cudaEventSynchronize(event));

                std::memcpy(dst_ptr, pinned, bytes);

                return_event(event);
                request.state->pinned_buffer = pinned;
            } else {
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream));
            }
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream));
        }

        if (!config_.use_pinned_memory || !request.state->pinned_buffer) {
            cudaEvent_t event = get_event();
            CUDA_CHECK(cudaEventRecord(event, stream));
            request.state->event = event;
        }

        request.state->result = cpu_tensor;
        request.state->cv.notify_all();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, false);
    }

    if (!request.state->event) {
        request.state->completed.store(true, std::memory_order_release);
        request.state->cv.notify_all();
    }

#else
    throw std::runtime_error("CUDA not enabled");
#endif
}

// ============================================================================
// Stream Management
// ============================================================================

auto TransferEngine::synchronize() -> void {
    {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return transfer_queue_.empty() &&
                   in_flight_transfers_.load(std::memory_order_acquire) == 0;
        });
    }

#ifdef TENZOR_USE_CUDA
    for (cudaStream_t stream : streams_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
#endif

#ifdef TENZOR_USE_ROCM
    for (hipStream_t stream : hip_streams_) {
        HIP_CHECK(hipStreamSynchronize(stream));
    }
#endif

#ifdef TENZOR_USE_ONEAPI
    for (auto& queue : sycl_queues_) {
        try {
            queue.wait();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL queue synchronization failed: ") + e.what());
        }
    }
#endif
}

auto TransferEngine::synchronize_stream(int stream_id) -> void {
    if (stream_id < 0 || stream_id >= config_.num_streams) {
        throw std::out_of_range("Invalid stream_id");
    }

    {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return transfer_queue_.empty() &&
                   in_flight_transfers_.load(std::memory_order_acquire) == 0;
        });
    }

#ifdef TENZOR_USE_CUDA
    CUDA_CHECK(cudaStreamSynchronize(streams_[stream_id]));
#endif

#ifdef TENZOR_USE_ROCM
    HIP_CHECK(hipStreamSynchronize(hip_streams_[stream_id]));
#endif

#ifdef TENZOR_USE_ONEAPI
    if (stream_id < static_cast<int>(sycl_queues_.size())) {
        try {
            sycl_queues_[stream_id].wait();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL queue synchronization failed: ") + e.what());
        }
    }
#endif
}

// ============================================================================
// Statistics
// ============================================================================

auto TransferEngine::get_average_bandwidth_gbps() const -> float {
    size_t total_bytes = stats_.bytes_transferred.load(std::memory_order_relaxed);
    double total_time_s = stats_.total_time_ms.load(std::memory_order_relaxed) / 1000.0;

    if (total_time_s <= 0.0) {
        return 0.0f;
    }

    double bytes_per_second = static_cast<double>(total_bytes) / total_time_s;
    return static_cast<float>(bytes_per_second / 1e9);
}

auto TransferEngine::get_statistics() const -> Statistics {
    Statistics stats;
    stats.total_transfers = stats_.total_transfers.load(std::memory_order_relaxed);
    stats.bytes_transferred = stats_.bytes_transferred.load(std::memory_order_relaxed);
    stats.cpu_to_gpu_count = stats_.cpu_to_gpu_count.load(std::memory_order_relaxed);
    stats.gpu_to_cpu_count = stats_.gpu_to_cpu_count.load(std::memory_order_relaxed);
    stats.total_time_ms = stats_.total_time_ms.load(std::memory_order_relaxed);
    stats.average_bandwidth_gbps = get_average_bandwidth_gbps();
    return stats;
}

auto TransferEngine::reset_statistics() -> void {
    stats_.total_transfers.store(0, std::memory_order_relaxed);
    stats_.bytes_transferred.store(0, std::memory_order_relaxed);
    stats_.cpu_to_gpu_count.store(0, std::memory_order_relaxed);
    stats_.gpu_to_cpu_count.store(0, std::memory_order_relaxed);
    stats_.total_time_ms.store(0.0, std::memory_order_relaxed);
}

} // namespace core
} // namespace tenzor
