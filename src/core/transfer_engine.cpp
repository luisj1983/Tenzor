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

namespace tenzor {
namespace core {

// ============================================================================
// TransferState Implementation
// ============================================================================

TransferState::~TransferState() {
#ifdef TENZOR_USE_CUDA
    // Return event to pool if we have one
    if (event && engine) {
        engine->return_event(event);
    }

    // Return pinned buffer if we have one
    if (pinned_buffer && engine) {
        engine->return_pinned_buffer(pinned_buffer);
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
        return true;  // Empty handle is "ready"
    }

    // Check atomic flag first (fast path)
    if (state_->completed.load(std::memory_order_acquire)) {
        return true;
    }

#ifdef TENZOR_USE_CUDA
    // Check CUDA event if available
    if (state_->event) {
        cudaError_t result = cudaEventQuery(state_->event);
        if (result == cudaSuccess) {
            state_->completed.store(true, std::memory_order_release);
            return true;
        }
        // cudaErrorNotReady means still in progress
        if (result != cudaErrorNotReady) {
            CUDA_CHECK(result);
        }
    }
#endif

    return false;
}

auto TransferHandle::wait() -> void {
    if (!state_) {
        return;
    }

    // Fast path: already completed
    if (state_->completed.load(std::memory_order_acquire)) {
        return;
    }

#ifdef TENZOR_USE_CUDA
    // Wait on CUDA event if available
    if (state_->event) {
        CUDA_CHECK(cudaEventSynchronize(state_->event));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
        return;
    }
#endif

    // Fallback: wait on condition variable
    std::unique_lock lock(state_->mutex);
    state_->cv.wait(lock, [this] {
        // Wake up if completed OR if event has been set
        return state_->completed.load(std::memory_order_acquire)
#ifdef TENZOR_USE_CUDA
               || (state_->event != nullptr)
#endif
               ;
    });

#ifdef TENZOR_USE_CUDA
    // If event was set while we were waiting, synchronize on it now
    if (state_->event) {
        lock.unlock();  // Release lock before CUDA call
        CUDA_CHECK(cudaEventSynchronize(state_->event));
        state_->completed.store(true, std::memory_order_release);
        state_->cv.notify_all();
    }
#endif
}

auto TransferHandle::get_tensor() -> Tensor {
    if (!state_) {
        throw std::runtime_error("Invalid transfer handle");
    }

    // Wait for transfer to complete
    wait();

    // Check for errors
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

    // Initialize CUDA resources
    initialize_cuda_resources();

    // Start worker thread AFTER all initialization is complete
    // This ensures the object is fully constructed before thread starts
    worker_thread_ = std::thread(&TransferEngine::transfer_worker, this);
}

TransferEngine::~TransferEngine() {
    // Stop worker thread
    {
        std::lock_guard lock(queue_mutex_);
        stop_worker_.store(true, std::memory_order_release);
    }
    queue_cv_.notify_all();

    if (worker_thread_.joinable()) {
        worker_thread_.join();
    }

    // Cleanup CUDA resources
    cleanup_cuda_resources();
}

auto TransferEngine::initialize_cuda_resources() -> void {
#ifdef TENZOR_USE_CUDA
    // Create CUDA streams
    streams_.reserve(config_.num_streams);
    for (int i = 0; i < config_.num_streams; ++i) {
        cudaStream_t stream;
        CUDA_CHECK(cudaStreamCreate(&stream));
        streams_.push_back(stream);
    }

    // Pre-allocate some events in pool
    event_pool_.reserve(config_.num_streams * 2);
    for (int i = 0; i < config_.num_streams * 2; ++i) {
        cudaEvent_t event;
        CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        event_pool_.push_back(event);
    }

    // Pre-allocate pinned memory buffers if enabled
    if (config_.use_pinned_memory && config_.pinned_pool_size > 0) {
        // Create several buffers of varying sizes
        std::vector<size_t> buffer_sizes = {
            1 * 1024 * 1024,     // 1 MB
            4 * 1024 * 1024,     // 4 MB
            16 * 1024 * 1024,    // 16 MB
            64 * 1024 * 1024     // 64 MB
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
            // Ignore allocation failures - we'll allocate on-demand if needed
        }
    }
#endif
}

auto TransferEngine::cleanup_cuda_resources() -> void {
#ifdef TENZOR_USE_CUDA
    // Destroy streams
    for (cudaStream_t stream : streams_) {
        cudaStreamDestroy(stream);
    }
    streams_.clear();

    // Destroy events
    for (cudaEvent_t event : event_pool_) {
        cudaEventDestroy(event);
    }
    event_pool_.clear();

    // Free pinned memory
    for (auto& buffer : pinned_buffers_) {
        if (buffer.ptr) {
            cudaFreeHost(buffer.ptr);
        }
    }
    pinned_buffers_.clear();
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

    // Pool empty, create new event
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

auto TransferEngine::get_pinned_buffer(size_t size) -> void* {
    std::lock_guard lock(pinned_mutex_);

    // Find available buffer that's large enough
    for (auto& buffer : pinned_buffers_) {
        if (!buffer.in_use && buffer.size >= size) {
            buffer.in_use = true;
            return buffer.ptr;
        }
    }

#ifdef TENZOR_USE_CUDA
    // No suitable buffer, allocate new one
    void* ptr = nullptr;
    cudaError_t err = cudaMallocHost(&ptr, size);
    if (err == cudaSuccess) {
        pinned_buffers_.push_back({ptr, size, true});
        return ptr;
    }
#endif

    // Failed to allocate pinned memory
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

    // Update total time (this is approximate due to concurrent updates)
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

    if (gpu_device.type != Device::Type::CUDA) {
        throw std::runtime_error("Target device must be CUDA");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate GPU tensor
    auto shape_span = cpu_tensor.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    Tensor gpu_tensor = allocate_tensor(shape_vec, cpu_tensor.dtype(), gpu_device);

    size_t bytes = cpu_tensor.numel() * dtype_size(cpu_tensor.dtype());

#ifdef TENZOR_USE_CUDA
    // Set device
    CUDA_CHECK(cudaSetDevice(gpu_device.index));

    // Synchronous memcpy
    CUDA_CHECK(cudaMemcpy(
        gpu_tensor.data_ptr(),
        cpu_tensor.data_ptr(),
        bytes,
        cudaMemcpyHostToDevice
    ));
#else
    throw std::runtime_error("CUDA not enabled");
#endif

    auto end = std::chrono::high_resolution_clock::now();
    double time_ms = std::chrono::duration<double, std::milli>(end - start).count();

    record_transfer(bytes, time_ms, true);

    return gpu_tensor;
}

auto TransferEngine::gpu_to_cpu(const Tensor& gpu_tensor) -> Tensor {
    if (gpu_tensor.device().type != Device::Type::CUDA) {
        throw std::runtime_error("Source tensor must be on CUDA");
    }

    auto start = std::chrono::high_resolution_clock::now();

    // Allocate CPU tensor
    auto shape_span = gpu_tensor.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    Tensor cpu_tensor = allocate_tensor(
        shape_vec,
        gpu_tensor.dtype(),
        Device::cpu()
    );

    size_t bytes = gpu_tensor.numel() * dtype_size(gpu_tensor.dtype());

#ifdef TENZOR_USE_CUDA
    // Set device
    CUDA_CHECK(cudaSetDevice(gpu_tensor.device().index));

    // Synchronous memcpy
    CUDA_CHECK(cudaMemcpy(
        cpu_tensor.data_ptr(),
        gpu_tensor.data_ptr(),
        bytes,
        cudaMemcpyDeviceToHost
    ));
#else
    throw std::runtime_error("CUDA not enabled");
#endif

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

    if (gpu_device.type != Device::Type::CUDA) {
        throw std::runtime_error("Target device must be CUDA");
    }

    // Create transfer state
    auto state = std::make_shared<TransferState>();
    state->engine = this;

    // Queue transfer request
    TransferRequest request;
    request.type = TransferRequest::Type::CPU_TO_GPU;
    request.source = cpu_tensor;
    request.target_device = gpu_device;
    request.state = state;

    {
        std::unique_lock lock(queue_mutex_);

        // Wait for queue space if full (backpressure handling)
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
    if (gpu_tensor.device().type != Device::Type::CUDA) {
        throw std::runtime_error("Source tensor must be on CUDA");
    }

    // Create transfer state
    auto state = std::make_shared<TransferState>();
    state->engine = this;

    // Queue transfer request
    TransferRequest request;
    request.type = TransferRequest::Type::GPU_TO_CPU;
    request.source = gpu_tensor;
    request.target_device = Device::cpu();
    request.state = state;

    {
        std::unique_lock lock(queue_mutex_);

        // Wait for queue space if full (backpressure handling)
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

        // Get next request from queue
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

            // Increment in-flight counter (still inside lock to maintain consistency)
            in_flight_transfers_.fetch_add(1, std::memory_order_release);
        }

        // Notify that queue has space (for backpressure handling)
        queue_cv_.notify_one();

        // Process transfer
        try {
            process_transfer(request);
        } catch (const std::exception& e) {
            // Mark transfer as failed
            request.state->has_error = true;
            request.state->error_message = e.what();
            request.state->completed.store(true, std::memory_order_release);
            request.state->cv.notify_all();
        }

        // Decrement in-flight counter and notify
        in_flight_transfers_.fetch_sub(1, std::memory_order_release);
        queue_cv_.notify_all();  // Wake up threads waiting in synchronize()
    }
}

auto TransferEngine::process_transfer(const TransferRequest& request) -> void {
    auto start = std::chrono::high_resolution_clock::now();

#ifdef TENZOR_USE_CUDA
    // Select stream (round-robin)
    static std::atomic<int> stream_counter{0};
    int stream_idx = stream_counter.fetch_add(1, std::memory_order_relaxed) % config_.num_streams;
    cudaStream_t stream = streams_[stream_idx];

    request.state->stream = stream;

    size_t bytes = request.source.numel() * dtype_size(request.source.dtype());

    if (request.type == TransferRequest::Type::CPU_TO_GPU) {
        // CPU -> GPU transfer
        Device gpu_device = request.target_device;
        CUDA_CHECK(cudaSetDevice(gpu_device.index));

        // Allocate GPU tensor
        auto shape_span = request.source.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor gpu_tensor = allocate_tensor(
            shape_vec,
            request.source.dtype(),
            gpu_device
        );

        const void* src_ptr = request.source.data_ptr();
        void* dst_ptr = gpu_tensor.data_ptr();

        // Use pinned memory if available for better performance
        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                // Copy CPU -> pinned (synchronous)
                std::memcpy(pinned, src_ptr, bytes);

                // Copy pinned -> GPU (async)
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_ptr,
                    pinned,
                    bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ));

                request.state->pinned_buffer = pinned;
            } else {
                // Fallback: direct copy (slower)
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    bytes,
                    cudaMemcpyHostToDevice,
                    stream
                ));
            }
        } else {
            // Direct copy
            CUDA_CHECK(cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                bytes,
                cudaMemcpyHostToDevice,
                stream
            ));
        }

        // Record event
        cudaEvent_t event = get_event();
        CUDA_CHECK(cudaEventRecord(event, stream));
        request.state->event = event;
        request.state->result = gpu_tensor;

        // Notify condition variable in case wait() is already waiting
        request.state->cv.notify_all();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, true);

    } else {
        // GPU -> CPU transfer
        CUDA_CHECK(cudaSetDevice(request.source.device().index));

        // Allocate CPU tensor
        auto shape_span = request.source.shape();
        std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
        Tensor cpu_tensor = allocate_tensor(
            shape_vec,
            request.source.dtype(),
            Device::cpu()
        );

        const void* src_ptr = request.source.data_ptr();
        void* dst_ptr = cpu_tensor.data_ptr();

        // Use pinned memory if available
        if (config_.use_pinned_memory) {
            void* pinned = get_pinned_buffer(bytes);
            if (pinned) {
                // Copy GPU -> pinned (async)
                CUDA_CHECK(cudaMemcpyAsync(
                    pinned,
                    src_ptr,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                ));

                // Record event and copy pinned -> CPU after event completes
                cudaEvent_t event = get_event();
                CUDA_CHECK(cudaEventRecord(event, stream));

                // Wait for transfer to complete
                CUDA_CHECK(cudaEventSynchronize(event));

                // Copy pinned -> CPU (synchronous)
                std::memcpy(dst_ptr, pinned, bytes);

                return_event(event);
                request.state->pinned_buffer = pinned;
            } else {
                // Fallback: direct copy
                CUDA_CHECK(cudaMemcpyAsync(
                    dst_ptr,
                    src_ptr,
                    bytes,
                    cudaMemcpyDeviceToHost,
                    stream
                ));
            }
        } else {
            // Direct copy
            CUDA_CHECK(cudaMemcpyAsync(
                dst_ptr,
                src_ptr,
                bytes,
                cudaMemcpyDeviceToHost,
                stream
            ));
        }

        // Record event for non-pinned path or if pinned copy already completed
        if (!config_.use_pinned_memory || !request.state->pinned_buffer) {
            cudaEvent_t event = get_event();
            CUDA_CHECK(cudaEventRecord(event, stream));
            request.state->event = event;
        }

        request.state->result = cpu_tensor;

        // Notify condition variable in case wait() is already waiting
        request.state->cv.notify_all();

        auto end = std::chrono::high_resolution_clock::now();
        double time_ms = std::chrono::duration<double, std::milli>(end - start).count();
        record_transfer(bytes, time_ms, false);
    }

    // Mark as completed if no event (synchronous path)
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
    // Wait for queue to be empty AND all in-flight transfers to complete
    {
        std::unique_lock lock(queue_mutex_);
        queue_cv_.wait(lock, [this] {
            return transfer_queue_.empty() &&
                   in_flight_transfers_.load(std::memory_order_acquire) == 0;
        });
    }

#ifdef TENZOR_USE_CUDA
    // Then synchronize all streams (ensures GPU work is complete)
    for (cudaStream_t stream : streams_) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
    }
#endif
}

auto TransferEngine::synchronize_stream(int stream_id) -> void {
    if (stream_id < 0 || stream_id >= config_.num_streams) {
        throw std::out_of_range("Invalid stream_id");
    }

    // Wait for queue to be empty AND all in-flight transfers to complete
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
    return static_cast<float>(bytes_per_second / 1e9);  // Convert to GB/s
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
