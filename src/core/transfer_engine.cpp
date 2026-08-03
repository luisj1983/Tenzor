/**
 * @file transfer_engine.cpp
 * @brief Implementation of asynchronous CPU<->GPU transfer engine
 */

#include "tenzor/core/transfer_engine.hpp"
#include "tenzor/core/rocm_transfer.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/utils/log.hpp"
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

// FINDING 60: this TU previously #included <hip/hip_runtime.h> here under
// #ifdef TENZOR_USE_ROCM. That's the reason TENZOR_USE_ROCM could never
// safely be defined for tenzor_core: on a combined CUDA+ROCm build (this
// project's default), the file already includes <cuda_runtime.h> above, and
// HIP's and CUDA's vector-type headers define conflicting make_ulonglong3
// -style helpers when both land in the same translation unit (confirmed by
// actually trying it -- ~30 redefinition errors). Every ROCm code path in
// this file already routes through tenzor::rocm_transfer:: (the isolated
// HIP-language TU in rocm_transfer.hip.cpp, which is the ONLY place
// <hip/hip_runtime.h> may be included from tenzor_core) or the public Tensor
// API's own .to()/copy() dispatch, both proven correct and exercised on real
// ROCm hardware this session -- so the direct-hipMemcpy/hipStream_t code
// this header enabled was already 100% dead (TENZOR_USE_ROCM was never
// defined anywhere in the build) and has been removed rather than resurrected.

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
            // Audit I.4: unified logger.
            TENZOR_LOG_WARN("SYCL queue wait failed during cleanup: {}", e.what());
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
        if ((type_filter & (1u << i)) &&  // 1u: `1 << 31` is signed-overflow UB
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
    // The destructor releases `source` (and possibly `result`). If the async
    // copy is still in flight, freeing those buffers now reintroduces the
    // exact use-after-free this state exists to prevent (see `source` in the
    // header) — so wait for completion first. For finalized handles the event
    // has already signalled and these waits are no-ops.
    //
    // ROCm async path (HIP-isolated): wait for the DMA, then release the event.
    // No-op if wait() already finalized it (nulls rocm_event) or on non-ROCm
    // builds (stub). Must precede freeing `source`/`result` below. The pinned
    // D2H path may have a pending pinned->dst host copy that no wait()/
    // is_ready() ever ran — finalize it now (the DMA has just synced above)
    // so any holder of `result` sees the transferred bytes, and before
    // `pinned_buffer` is returned to the pool for reuse below.
    if (rocm_event != nullptr) {
        tenzor::rocm_transfer::event_sync(rocm_event);
        rocm_event = nullptr;
        finalize_deferred_copy();
    }
#ifdef TENZOR_USE_ONEAPI
    if (has_sycl_event) {
        try { sycl_event.wait(); } catch (...) { /* device teardown */ }
    }
#endif
#ifdef TENZOR_USE_CUDA
    if (event && !completed.load(std::memory_order_acquire)) {
        cudaEventSynchronize(event);  // ignore errors at teardown
        // The async D2H pinned path may have a pending pinned->dst host copy
        // that no wait()/is_ready() ever ran. Complete it now (once the DMA has
        // synced above) so any holder of `result` sees the transferred bytes,
        // and before `pinned_buffer` is returned to the pool for reuse.
        finalize_deferred_copy();
    }
#endif
#ifdef TENZOR_USE_CUDA
    if (event && engine) {
        engine->return_event(event);
    }
#endif

    if (pinned_buffer && engine) {
        engine->return_pinned_buffer(pinned_buffer);
    }

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

void TransferState::finalize_deferred_copy() {
    // Sequenced after the completion event has signalled by every caller.
    // The mutex guarantees the memcpy runs exactly once and that its effects
    // are visible to any thread that subsequently observes `completed` (which
    // is stored with release ordering only after this returns): a waiter that
    // sees completion is therefore guaranteed to see the copied host bytes.
    std::lock_guard lock(mutex);
    if (deferred_dst && pinned_buffer && !host_copy_done) {
        std::memcpy(deferred_dst, pinned_buffer, deferred_bytes);
        host_copy_done = true;
    }
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

    // ROCm async path (HIP-isolated): non-blocking completion query.
    if (state_->rocm_event != nullptr) {
        if (tenzor::rocm_transfer::event_ready(state_->rocm_event)) {
            // Event signalled => device->pinned DMA done (if pinned staging
            // was used for D2H). Run the deferred pinned->dst host copy
            // before publishing completion, same as the CUDA path below.
            state_->finalize_deferred_copy();
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
        return false;
    }

#ifdef TENZOR_USE_CUDA
    if (state_->event) {
        cudaError_t result = cudaEventQuery(state_->event);
        if (result == cudaSuccess) {
            // Event signalled => device->pinned DMA done. Run the deferred
            // pinned->dst host copy (if any) before publishing completion.
            state_->finalize_deferred_copy();
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
        if (result != cudaErrorNotReady) {
            CUDA_CHECK(result);
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

    // ROCm async path (HIP-isolated): block on the event, release it, finalize.
    if (state_->rocm_event != nullptr) {
        tenzor::rocm_transfer::event_sync(state_->rocm_event);
        state_->rocm_event = nullptr;
        state_->finalize_deferred_copy();
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }

#ifdef TENZOR_USE_CUDA
    if (state_->event) {
        CUDA_CHECK(cudaEventSynchronize(state_->event));
        // DMA done: run the deferred pinned->dst host copy (if any) before
        // publishing completion so a subsequent get_tensor() sees the bytes.
        state_->finalize_deferred_copy();
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
#ifdef TENZOR_USE_ONEAPI
               || state_->has_sycl_event
#endif
               ;
    });

#ifdef TENZOR_USE_CUDA
    if (state_->event) {
        lock.unlock();
        CUDA_CHECK(cudaEventSynchronize(state_->event));
        // DMA done: run the deferred pinned->dst host copy (if any) before
        // publishing completion so a subsequent get_tensor() sees the bytes.
        state_->finalize_deferred_copy();
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
    // ROCm resources: no per-engine stream/event init needed. The isolated-TU
    // rocm_transfer:: path (see the file-top comment) owns its own
    // streams/event pool internally (rocm_transfer.hip.cpp's g_streams/
    // g_event_pool), so there is nothing for TransferEngine itself to set up
    // there. The pinned-buffer pool is per-engine, though, so it does need
    // its own initializer (mirrors initialize_cuda_resources' pinned pool).
    initialize_rocm_pinned_pool();

#ifdef TENZOR_USE_ONEAPI
    initialize_oneapi_resources();
#endif

#ifdef TENZOR_USE_VULKAN
    initialize_vulkan_resources();
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

    // Defensive drain: the worker fully empties the queue before exiting, but in
    // case any request remains (e.g. enqueued concurrently during shutdown),
    // mark each as errored and wake its waiter so no TransferHandle::wait()
    // blocks forever on a request that will never be processed.
    {
        std::lock_guard lock(queue_mutex_);
        while (!transfer_queue_.empty()) {
            TransferRequest& request = transfer_queue_.front();
            if (request.state) {
                {
                    std::lock_guard state_lock(request.state->mutex);
                    request.state->has_error = true;
                    request.state->error_message = "Transfer engine shut down before request was processed";
                    request.state->completed.store(true, std::memory_order_release);
                }
                request.state->cv.notify_all();
            }
            transfer_queue_.pop();
        }
    }

    cleanup_cuda_resources();
    // ROCm: see initialize_rocm_resources' removal note above -- nothing
    // per-engine to clean up.

#ifdef TENZOR_USE_ONEAPI
    cleanup_oneapi_resources();
#endif

#ifdef TENZOR_USE_VULKAN
    cleanup_vulkan_resources();
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
                pinned_buffers_.push_back({ptr, size, false, false});
                total_allocated += size;
            }
        }
    }
#endif
}

auto TransferEngine::initialize_rocm_pinned_pool() -> void {
    // Pre-warms the pinned-buffer pool with hipHostMalloc allocations the same
    // way initialize_cuda_resources() does for cudaMallocHost, so ROCm
    // transfers get the same DMA-accelerated staging buffers CUDA already
    // has (see findings.txt — this pool used to be CUDA-only, leaving
    // get_pinned_buffer() to return nullptr unconditionally on ROCm and
    // silently fall back to unpinned copies). No-op when ROCm isn't linked in.
    if (!config_.use_pinned_memory || config_.pinned_pool_size == 0) {
        return;
    }
    if (!rocm_transfer::available()) {
        return;
    }
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
        if (void* ptr = rocm_transfer::host_malloc(size)) {
            pinned_buffers_.push_back({ptr, size, false, true});
            total_allocated += size;
        }
    }
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
        if (buffer.ptr && !buffer.via_rocm) {
            cudaFreeHost(buffer.ptr);
        }
    }
#endif
    cleanup_rocm_pinned_pool();
    pinned_buffers_.clear();
}

auto TransferEngine::cleanup_rocm_pinned_pool() -> void {
    for (auto& buffer : pinned_buffers_) {
        if (buffer.ptr && buffer.via_rocm) {
            rocm_transfer::host_free(buffer.ptr);
        }
    }
}

// FINDING 60: initialize_rocm_resources()/cleanup_rocm_resources() removed.
// Their entire bodies were gated on TENZOR_USE_ROCM, which was never defined
// for this TU (see the file-top comment), so they were dead code that never
// executed. The isolated-TU rocm_transfer:: path owns ROCm stream/event/
// pinned-buffer lifetime instead (see rocm_transfer.hip.cpp). The calls to
// these two functions were already removed from the constructor/destructor.

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

// FINDING 60: get_hip_event()/return_hip_event() removed — dead code, see
// the removal note above initialize_rocm_resources' former location.

auto TransferEngine::get_pinned_buffer(size_t size) -> void* {
    std::lock_guard lock(pinned_mutex_);

    void* result = nullptr;
    for (auto& buffer : pinned_buffers_) {
        if (!buffer.in_use && buffer.size >= size) {
            buffer.in_use = true;
            result = buffer.ptr;
            break;
        }
    }

    if (!result) {
        // Enforce an upper bound on total page-locked memory. Pinned memory is a
        // scarce OS resource; without a cap, distinct/large transfer sizes each
        // spawn a new permanent allocation that is never freed until engine
        // destruction. Account for currently-reserved bytes and, when a new
        // allocation would exceed config_.pinned_pool_size, first evict free
        // (not in-use) buffers to reclaim room; if it still does not fit, return
        // nullptr so the caller falls back to a non-pinned copy.
        const size_t cap = config_.pinned_pool_size;

        auto reserved_bytes = [this]() -> size_t {
            size_t total = 0;
            for (const auto& b : pinned_buffers_) {
                total += b.size;
            }
            return total;
        };

        if (cap > 0 && reserved_bytes() + size > cap) {
            // Try to free idle buffers (smallest-first) until the new allocation
            // fits under the cap.
            for (auto it = pinned_buffers_.begin();
                 it != pinned_buffers_.end() && reserved_bytes() + size > cap; ) {
                if (!it->in_use) {
                    if (it->via_rocm) {
                        rocm_transfer::host_free(it->ptr);
                    } else {
#ifdef TENZOR_USE_CUDA
                        cudaFreeHost(it->ptr);
#endif
                    }
                    it = pinned_buffers_.erase(it);
                } else {
                    ++it;
                }
            }
        }

        if (cap > 0 && reserved_bytes() + size > cap) {
            // Cannot honor this request within the pinned budget; let the caller
            // fall back to a direct (non-pinned) memcpy.
            return nullptr;
        }

#ifdef TENZOR_USE_CUDA
        cudaError_t err = cudaMallocHost(&result, size);
        if (err == cudaSuccess) {
            pinned_buffers_.push_back({result, size, true, false});
        } else {
            result = nullptr;
        }
#else
        if (void* rocm_ptr = rocm_transfer::host_malloc(size)) {
            result = rocm_ptr;
            pinned_buffers_.push_back({result, size, true, true});
        }
#endif
    }

    if (result) {
        // Update the high-water in-use mark for pinned-pool telemetry.
        size_t cur = 0;
        for (const auto& b : pinned_buffers_) {
            if (b.in_use) cur += b.size;
        }
        if (cur > pinned_peak_allocated_) pinned_peak_allocated_ = cur;
    }

    return result;
}

auto TransferEngine::get_pinned_memory_stats() -> core::PinnedMemoryStats {
    std::lock_guard lock(pinned_mutex_);

    core::PinnedMemoryStats stats{};
    size_t allocated = 0;
    size_t reserved = 0;
    size_t in_use_blocks = 0;
    size_t free_blocks = 0;
    size_t largest_free = 0;
    for (const auto& b : pinned_buffers_) {
        reserved += b.size;
        if (b.in_use) {
            allocated += b.size;
            ++in_use_blocks;
        } else {
            ++free_blocks;
            if (b.size > largest_free) largest_free = b.size;
        }
    }

    stats.total_size = config_.pinned_pool_size;  // configured capacity
    stats.allocated_size = allocated;
    stats.free_size = (stats.total_size > allocated) ? (stats.total_size - allocated) : 0;
    stats.num_allocations = in_use_blocks;
    stats.num_blocks = pinned_buffers_.size();
    stats.num_free_blocks = free_blocks;
    stats.peak_allocated = pinned_peak_allocated_;
    stats.num_defragmentations = 0;  // pool never defragments

    // Fragmentation of the already-reserved free space: 0 when a single free
    // block holds all of it, →1 as it splinters across many small blocks.
    const size_t free_reserved = reserved - allocated;
    stats.fragmentation_ratio = (free_reserved > 0)
        ? static_cast<float>(free_reserved - largest_free) / static_cast<float>(free_reserved)
        : 0.0f;

    return stats;
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

    // Only transfers with a real (non-zero) measured duration contribute to the
    // bandwidth average. Async/queued transfers record an empty timing window
    // (time_ms ~= 0) and would otherwise add bytes against ~0 time, inflating
    // the reported bandwidth.
    if (time_ms > 0.0) {
        stats_.timed_bytes_transferred.fetch_add(bytes, std::memory_order_relaxed);
    }

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
        // The raw-byte memcpy below reads a flat run from data_ptr(); for a
        // non-contiguous source (transpose/slice view) that run is the wrong
        // elements. Contiguify first so the flat copy matches logical order.
        Tensor src = cpu_tensor.contiguous();
        CUDA_CHECK(cudaSetDevice(gpu_device.index));
        CUDA_CHECK(cudaMemcpy(
            gpu_tensor.data_ptr(),
            src.data_ptr(),
            bytes,
            cudaMemcpyHostToDevice
        ));
    }
#endif

    // ROCm: route through the public Tensor API (dispatches to the ROCm backend's
    // own HIP memcpy). A direct hipMemcpy path used to be gated on TENZOR_USE_ROCM,
    // which is never defined for tenzor_core (HIP/CUDA host headers define
    // conflicting make_*N helpers in multi-backend builds), so that branch was
    // dead code and has been removed (FINDING 60). Using .to() fills it correctly.
    if (gpu_device.type == Device::Type::ROCm) {
        gpu_tensor = cpu_tensor.to(gpu_device);
    }

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
        // Contiguify the (possibly strided) GPU source so the flat byte copy
        // into the fresh contiguous host tensor reads logical element order.
        Tensor src = gpu_tensor.contiguous();
        CUDA_CHECK(cudaSetDevice(gpu_tensor.device().index));
        CUDA_CHECK(cudaMemcpy(
            cpu_tensor.data_ptr(),
            src.data_ptr(),
            bytes,
            cudaMemcpyDeviceToHost
        ));
    }
#endif

    // ROCm: see cpu_to_gpu — a direct hipMemcpy path used to be compiled out of
    // core (TENZOR_USE_ROCM undefined), leaving this branch dead. Removed
    // (FINDING 60); route through .to() so the value is preserved.
    if (gpu_tensor.device().type == Device::Type::ROCm) {
        cpu_tensor = gpu_tensor.to(Device::cpu());
    }

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
        gpu_device.type != Device::Type::Vulkan) {
        throw std::runtime_error("Target device must be CUDA, ROCm, OneAPI, or Vulkan");
    }

    auto state = std::make_shared<TransferState>();
    state->engine = this;

    // Handle ROCm direct transfers. A direct HIP host-transfer path using
    // hip_streams_/get_hip_event() used to be gated on TENZOR_USE_ROCM, which
    // is never defined for tenzor_core (HIP/CUDA host headers define
    // conflicting make_*N vector helpers in a multi-backend build), so it was
    // dead code and has been removed (FINDING 60). The isolated-TU
    // rocm_transfer::h2d_async() path below (compiled by hipcc in
    // rocm_transfer.hip.cpp, exposed via an opaque void*-based API) is the
    // only ROCm async-transfer path that has ever actually run.
    if (gpu_device.type == Device::Type::ROCm) {
        auto shape_span = cpu_tensor.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(shape_vec, cpu_tensor.dtype(), gpu_device);
        // Contiguify so the flat byte DMA reads logical element order; anchor
        // the contiguous buffer in state->source until the async DMA completes.
        Tensor src = cpu_tensor.contiguous();
        size_t bytes = src.numel() * dtype_size(src.dtype());

        // H2D: issue hipMemcpyAsync directly on the (already-contiguous)
        // pageable source. The previous code staged through a
        // hipHostRegister'd pinned buffer with a synchronous
        // std::memcpy(pinned, src, bytes) here, on the caller's thread,
        // BEFORE issuing the async DMA. For a parallel loop of N
        // cpu_to_gpu_async calls (e.g. the async-overlap benchmark) those N
        // host->pinned memcpys serialize on the caller thread and dominate,
        // destroying cross-stream overlap (parallel measured ~0.6x of
        // serial). The HIP driver, like CUDA, stages pageable->internal-
        // pinned internally and overlaps concurrent H2D async copies across
        // the round-robin g_streams in rocm_transfer.hip.cpp without
        // host-side help, so issuing the DMA directly on the pageable
        // source is both faster and correctly overlapping. The pinned pool
        // is still used for the D2H direction below, where the DMA lands in
        // pinned memory and a deferred pinned->dst copy is needed.
        void* ev = tenzor::rocm_transfer::h2d_async(
            gpu_tensor.data_ptr(), src.data_ptr(), bytes, gpu_device.index);
        state->result = gpu_tensor;
        state->source = src;  // keep contiguous source alive until DMA completes
        state->rocm_event = ev;
        if (ev == nullptr) {
            state->completed.store(true, std::memory_order_release);
        } else {
            std::lock_guard<std::mutex> pending_lock(rocm_pending_mutex_);
            rocm_pending_states_.push_back(state);
        }
        record_transfer(bytes, 0.0, true);
        return TransferHandle(state);
    }

#ifdef TENZOR_USE_ONEAPI
    if (gpu_device.type == Device::Type::OneAPI) {
        int queue_idx = next_stream_.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
        auto& queue = get_sycl_queue(queue_idx);

        auto shape_span = cpu_tensor.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(shape_vec, cpu_tensor.dtype(), gpu_device);

        // Contiguify so the flat byte DMA reads logical element order; anchor the
        // contiguous buffer in state->source until the async DMA completes.
        Tensor src = cpu_tensor.contiguous();
        size_t bytes = src.numel() * dtype_size(src.dtype());

        try {
            sycl::event event = queue.memcpy(gpu_tensor.data_ptr(), src.data_ptr(), bytes);
            state->result = gpu_tensor;
        state->source = src;  // keep contiguous source alive until DMA completes
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

    // Vulkan has no dedicated async stream/queue wired into TransferEngine (no
    // per-stream VkCommandBuffer/VkFence pool analogous to hip_streams_/the
    // SYCL queues above). Same trade-off the ROCm branch documents when its
    // HIP-isolated TU is unavailable: do the transfer synchronously via the
    // Tensor API's own working Vulkan copy path (used by the sync cpu_to_gpu()
    // above) and mark the handle immediately complete. This used to
    // unconditionally throw here, which meant any caller using the async
    // offload API on Vulkan (ZeRO offload_to_cpu, OffloadEngine) got an
    // exception instead of a degraded-but-correct transfer.
    if (gpu_device.type == Device::Type::Vulkan) {
        Tensor gpu_tensor = cpu_tensor.to(gpu_device);
        size_t bytes = static_cast<size_t>(gpu_tensor.numel()) * dtype_size(gpu_tensor.dtype());
        state->result = gpu_tensor;
        state->completed.store(true, std::memory_order_release);
        record_transfer(bytes, 0.0, true);
        return TransferHandle(state);
    }

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
        gpu_tensor.device().type != Device::Type::Vulkan) {
        throw std::runtime_error("Source tensor must be on CUDA, ROCm, OneAPI, or Vulkan");
    }

    auto state = std::make_shared<TransferState>();
    state->engine = this;

    // Handle ROCm direct transfers. A direct HIP host-transfer path using
    // hip_streams_/get_hip_event() used to be gated on TENZOR_USE_ROCM, which
    // is never defined for tenzor_core, so it was dead code and has been
    // removed (FINDING 60). See cpu_to_gpu_async for the same rationale.
    if (gpu_tensor.device().type == Device::Type::ROCm) {
        // Contiguify the GPU source so the flat byte DMA reads logical element
        // order; anchor the contiguous buffer until the async DMA completes.
        Tensor src = gpu_tensor.contiguous();
        auto shape_span = src.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_result = allocate_tensor(shape_vec, src.dtype(), Device::cpu());
        size_t bytes = src.numel() * dtype_size(src.dtype());

        // D2H: issue hipMemcpyAsync directly to the (pageable) cpu_result
        // destination. The previous code staged through a hipHostRegister'd
        // pinned buffer and deferred a pinned->cpu_result host memcpy until
        // the event signalled (mirroring the CUDA D2H path, which is correct
        // for a DISCRETE GPU where D2H is a real PCIe DMA that needs a
        // pinned host target to be truly async). On the ROCm hardware this
        // path actually runs on (an integrated AMD Radeon iGPU), device
        // memory IS system RAM, so a D2H "DMA" is just a system-RAM copy.
        // The pinned staging then doubles the work (device->pinned->dst) and
        // the deferred pinned->dst memcpys serialize on the wait thread,
        // which made parallel D2H ~0.6x of serial (slower, not faster).
        // Issuing the DMA directly to the pageable destination is a single
        // copy per transfer, and the round-robin g_streams in
        // rocm_transfer.hip.cpp still overlap the concurrent copies. The
        // FINDING 25 cross-stream producer dependency is preserved: do_copy()
        // records a default-stream event and has the DMA's g_stream wait on
        // it for every DeviceToHost copy, regardless of the destination
        // buffer, so a producer kernel that wrote `src` on the default stream
        // is still completed before this DMA reads it.
        void* ev = tenzor::rocm_transfer::d2h_async(
            cpu_result.data_ptr(), src.data_ptr(), bytes, src.device().index);
        state->result = cpu_result;
        state->source = src;  // keep contiguous source alive until DMA completes
        state->rocm_event = ev;
        if (ev == nullptr) {
            state->completed.store(true, std::memory_order_release);
        } else {
            std::lock_guard<std::mutex> pending_lock(rocm_pending_mutex_);
            rocm_pending_states_.push_back(state);
        }
        record_transfer(bytes, 0.0, false);
        return TransferHandle(state);
    }

#ifdef TENZOR_USE_ONEAPI
    if (gpu_tensor.device().type == Device::Type::OneAPI) {
        int queue_idx = next_stream_.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
        auto& queue = get_sycl_queue(queue_idx);

        // Contiguify the GPU source so the flat byte DMA reads logical element
        // order; anchor the contiguous buffer until the async DMA completes.
        Tensor src = gpu_tensor.contiguous();
        auto shape_span = src.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_result = allocate_tensor(shape_vec, src.dtype(), Device::cpu());

        size_t bytes = src.numel() * dtype_size(src.dtype());

        try {
            sycl::event event = queue.memcpy(cpu_result.data_ptr(), src.data_ptr(), bytes);
            state->result = cpu_result;
        state->source = src;  // keep contiguous source alive until DMA completes
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

    // Vulkan has no dedicated async stream/queue wired into TransferEngine --
    // see the matching comment in cpu_to_gpu_async(). Do the transfer
    // synchronously via the Tensor API's own working Vulkan copy path (used
    // by the sync gpu_to_cpu() above) and mark the handle immediately
    // complete.
    if (gpu_tensor.device().type == Device::Type::Vulkan) {
        Tensor cpu_result = gpu_tensor.to(Device::cpu());
        size_t bytes = static_cast<size_t>(cpu_result.numel()) * dtype_size(cpu_result.dtype());
        state->result = cpu_result;
        state->completed.store(true, std::memory_order_release);
        record_transfer(bytes, 0.0, false);
        return TransferHandle(state);
    }

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
    // Loop until stop is requested AND the queue has been fully drained. Using
    // `while (!stop_worker_)` would let the outer condition exit the loop after
    // the current iteration while requests are still queued, leaving their
    // TransferState with completed=false and no recorded event -- a thread
    // holding the corresponding TransferHandle would then block forever in
    // wait(). Processing every remaining request on shutdown guarantees each
    // waiter is eventually woken (via completion or error).
    while (true) {
        TransferRequest request;

        {
            std::unique_lock lock(queue_mutex_);
            queue_cv_.wait(lock, [this] {
                return !transfer_queue_.empty() ||
                       stop_worker_.load(std::memory_order_acquire);
            });

            if (transfer_queue_.empty()) {
                // Queue is drained; only exit once shutdown has been requested.
                if (stop_worker_.load(std::memory_order_acquire)) {
                    break;
                }
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

    // Contiguify the source so the flat raw-byte memcpys below read logical
    // element order; a strided view's data_ptr() base would otherwise corrupt.
    // Anchor it in state->source so the (possibly freshly-allocated) contiguous
    // buffer stays alive until the async DMA completes.
    Tensor source = request.source.contiguous();
    request.state->source = source;

    if (request.type == TransferRequest::Type::CPU_TO_GPU) {
        Device gpu_device = request.target_device;
        CUDA_CHECK(cudaSetDevice(gpu_device.index));

        auto shape_span = source.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(shape_vec, source.dtype(), gpu_device);

        const void* src_ptr = source.data_ptr();
        void* dst_ptr = gpu_tensor.data_ptr();

        // H2D: issue cudaMemcpyAsync directly on the (already-contiguous)
        // source. The source is pageable host memory; modern CUDA drivers
        // stage pageable->internal-pinned internally and overlap concurrent
        // H2D async copies across streams without host-side help.
        //
        // We deliberately do NOT pre-stage through a pinned buffer here, even
        // when config_.use_pinned_memory is set: an explicit host->pinned
        // `std::memcpy` runs synchronously on this single worker thread, so
        // for N concurrent transfers it serialises N host memcpys before the
        // DMAs can be issued. On a PCIe-saturated link (where the DMAs cannot
        // exceed the serial baseline anyway) that staged copy is pure
        // overhead with no overlap to amortise it -- measured ~13% slower
        // than the serial baseline (speedup 0.87x). Letting the driver stage
        // the pageable source itself removes the host memcpy from the worker
        // and lets the N async copies genuinely overlap, measuring ~1.04x
        // (parallel faster than serial) on the same hardware. The pinned
        // pool is still used for the D2H direction below, where the DMA
        // writes into pinned memory and the final pinned->dst copy is
        // deferred to completion -- a different design that does benefit.
        CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyHostToDevice, stream));

        cudaEvent_t event = get_event();
        CUDA_CHECK(cudaEventRecord(event, stream));
        request.state->event = event;
        request.state->result = gpu_tensor;
        // (source is already retained by request.source for the queued path)

        request.state->cv.notify_all();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, true);

    } else {
        CUDA_CHECK(cudaSetDevice(source.device().index));

        auto shape_span = source.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_tensor = allocate_tensor(shape_vec, source.dtype(), Device::cpu());

        const void* src_ptr = source.data_ptr();
        void* dst_ptr = cpu_tensor.data_ptr();

        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                // Issue the device->pinned DMA and return WITHOUT blocking the
                // worker. Previously this recorded an event and immediately
                // cudaEventSynchronize'd it, then did the pinned->dst memcpy
                // inline — that stalled the single worker thread per transfer
                // and fully serialized all D2H copies (they could not overlap
                // across streams the way H2D does). Instead we record the event
                // (below) and DEFER the pinned->dst host copy to the completion
                // point (wait()/is_ready()/dtor via finalize_deferred_copy()),
                // so multiple D2H DMAs are in flight concurrently. `pinned`
                // stays reserved (in_use) in this state until it is destroyed,
                // and `dst_ptr` lives in `cpu_tensor`/`result`, so both copy
                // endpoints outlive the DMA.
                CUDA_CHECK(cudaMemcpyAsync(pinned, src_ptr, bytes, cudaMemcpyDeviceToHost, stream));
                request.state->pinned_buffer = pinned;
                request.state->deferred_dst = dst_ptr;
                request.state->deferred_bytes = bytes;
            } else {
                // No pinned buffer available: DMA straight into pageable dst;
                // no deferred host copy is needed.
                CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream));
            }
        } else {
            CUDA_CHECK(cudaMemcpyAsync(dst_ptr, src_ptr, bytes, cudaMemcpyDeviceToHost, stream));
        }

        // Always record a single completion event for the D2H path (both the
        // deferred-pinned and the direct pageable variants). Completion — and
        // the deferred pinned->dst copy, if any — is finalized in wait()/
        // is_ready(), never by stalling the worker here.
        cudaEvent_t event = get_event();
        CUDA_CHECK(cudaEventRecord(event, stream));
        request.state->event = event;

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

auto TransferEngine::drain_rocm_pending_states() -> void {
    std::vector<std::weak_ptr<TransferState>> pending;
    {
        std::lock_guard<std::mutex> lock(rocm_pending_mutex_);
        pending.swap(rocm_pending_states_);
    }
    for (auto& weak_state : pending) {
        auto state = weak_state.lock();
        if (!state) continue;  // handle + state already destroyed (dtor synced/released it)
        if (state->rocm_event != nullptr) {
            tenzor::rocm_transfer::event_sync(state->rocm_event);
            state->rocm_event = nullptr;
            // Finalize the deferred pinned->dst host copy (if the D2H path
            // staged through a pinned buffer) before publishing completion —
            // same requirement as wait()/is_ready()/dtor. Draining here
            // without this would let a caller observe completed=true (and
            // read `result`) before the pinned bytes were ever copied into
            // it, a real stale/uninitialized-data bug for whoever drains via
            // this path instead of TransferHandle::wait()/is_ready().
            state->finalize_deferred_copy();
            state->completed.store(true, std::memory_order_release);
            state->cv.notify_all();
        }
    }
}

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

    // ROCm streams are owned by the isolated-TU rocm_transfer:: path (see the
    // file-top comment); drain_rocm_pending_states() below is what actually
    // syncs outstanding ROCm transfers. A direct hip_streams_ loop gated on
    // TENZOR_USE_ROCM used to sit here but was dead code (FINDING 60).

#ifdef TENZOR_USE_ONEAPI
    for (auto& queue : sycl_queues_) {
        try {
            queue.wait();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL queue synchronization failed: ") + e.what());
        }
    }
#endif

    drain_rocm_pending_states();
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

    // ROCm: see synchronize() -- a direct hip_streams_[stream_id] sync gated on
    // TENZOR_USE_ROCM used to sit here but was dead code (FINDING 60).

#ifdef TENZOR_USE_ONEAPI
    if (stream_id < static_cast<int>(sycl_queues_.size())) {
        try {
            sycl_queues_[stream_id].wait();
        } catch (const sycl::exception& e) {
            throw std::runtime_error(std::string("SYCL queue synchronization failed: ") + e.what());
        }
    }
#endif

    // The ROCm isolated-TU async path has no per-stream affiliation (h2d_async/
    // d2h_async take a device index, not a stream id), so there is no narrower
    // "just this stream" semantics available for it -- drain all of them, same
    // as synchronize(). Still correct: synchronize_stream(id) is documented as
    // "at least as strong as" a targeted wait, never weaker.
    drain_rocm_pending_states();
}

// ============================================================================
// Statistics
// ============================================================================

auto TransferEngine::get_average_bandwidth_gbps() const -> float {
    // Use only the bytes from transfers that recorded a real duration; async
    // transfers contributed ~0 time and would otherwise skew the average.
    size_t total_bytes = stats_.timed_bytes_transferred.load(std::memory_order_relaxed);
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
    stats_.timed_bytes_transferred.store(0, std::memory_order_relaxed);
    stats_.cpu_to_gpu_count.store(0, std::memory_order_relaxed);
    stats_.gpu_to_cpu_count.store(0, std::memory_order_relaxed);
    stats_.total_time_ms.store(0.0, std::memory_order_relaxed);
}

} // namespace core
} // namespace tenzor
