/**
 * @file ddp.cpp
 * @brief Implementation of Distributed Data Parallel training wrapper
 *
 * Implements gradient bucketing and synchronization for multi-process
 * training. Parameters are grouped into buckets in reverse order
 * (matching backward pass traversal) for communication/computation overlap.
 *
 * When using a GPU backend (NCCL), all-reduce operations are launched
 * asynchronously on a dedicated communication CUDA stream, allowing
 * overlap with backward computation on the default compute stream.
 * CUDA events are used to synchronize between the two streams before
 * the optimizer step.
 */

#include "tenzor/distributed/ddp.hpp"
#include "tenzor/distributed/process_group.hpp"
#include "tenzor/core/dtype.hpp"
#include <algorithm>
#include <stdexcept>
#include <numeric>

#if defined(TENZOR_USE_CUDA)
    #include <cuda_runtime.h>
    #define DDP_CUDA_CHECK(call) \
        do { \
            cudaError_t err = call; \
            if (err != cudaSuccess) { \
                throw std::runtime_error( \
                    std::string("CUDA error in DDP: ") + \
                    cudaGetErrorString(err) + \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__) \
                ); \
            } \
        } while(0)
#elif defined(TENZOR_USE_ROCM)
    #include <hip/hip_runtime.h>
    // Map CUDA API names to HIP equivalents
    #define cudaStream_t hipStream_t
    #define cudaEvent_t hipEvent_t
    #define cudaStreamCreateWithFlags hipStreamCreateWithFlags
    #define cudaStreamNonBlocking hipStreamNonBlocking
    #define cudaStreamDestroy hipStreamDestroy
    #define cudaStreamSynchronize hipStreamSynchronize
    #define cudaStreamWaitEvent hipStreamWaitEvent
    #define cudaEventCreate hipEventCreate
    #define cudaEventCreateWithFlags hipEventCreateWithFlags
    #define cudaEventDestroy hipEventDestroy
    #define cudaEventRecord hipEventRecord
    #define cudaEventDisableTiming hipEventDisableTiming
    #define cudaSuccess hipSuccess
    #define cudaGetErrorString hipGetErrorString
    #define DDP_CUDA_CHECK(call) \
        do { \
            hipError_t err = call; \
            if (err != hipSuccess) { \
                throw std::runtime_error( \
                    std::string("HIP error in DDP: ") + \
                    hipGetErrorString(err) + \
                    " at " + __FILE__ + ":" + std::to_string(__LINE__) \
                ); \
            } \
        } while(0)
#endif

namespace tenzor::distributed {

// ============================================================================
// DistributedDataParallel Implementation
// ============================================================================

DistributedDataParallel::DistributedDataParallel(
    nn::Module& module,
    ProcessGroup& pg,
    size_t bucket_size_bytes
) : module_(module), pg_(pg) {

    // Build gradient buckets from module parameters
    build_buckets(bucket_size_bytes);

    // Detect whether the process group supports async stream operations
    // (i.e., it's an NCCL backend on GPU)
    use_gpu_comm_ = pg_.supports_async_stream();

    // Initialize the dedicated communication stream and per-bucket events
    init_comm_resources();

    // Broadcast parameters from rank 0 to ensure all processes
    // start with identical model weights
    broadcast_parameters();

    // Register backward hooks on each parameter for auto-sync
    register_grad_hooks();
}

DistributedDataParallel::~DistributedDataParallel() {
    destroy_comm_resources();
}

// ============================================================================
// Communication stream/event resource management
// ============================================================================

auto DistributedDataParallel::init_comm_resources() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (!use_gpu_comm_) {
        return;
    }

    // Create a dedicated non-blocking CUDA stream for communication.
    // cudaStreamNonBlocking ensures this stream does NOT implicitly
    // synchronize with the default (stream 0) compute stream.
    cudaStream_t stream = nullptr;
    DDP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    comm_stream_ = static_cast<void*>(stream);

    // Create one CUDA event per bucket for synchronization.
    // Events use cudaEventDisableTiming for lower overhead since
    // we only need ordering guarantees, not timing.
    bucket_events_.resize(buckets_.size(), nullptr);
    for (size_t i = 0; i < buckets_.size(); ++i) {
        cudaEvent_t event = nullptr;
        DDP_CUDA_CHECK(cudaEventCreateWithFlags(&event,
                                                 cudaEventDisableTiming));
        bucket_events_[i] = static_cast<void*>(event);
    }
#else
    // CPU-only build: no stream resources needed
    (void)use_gpu_comm_;
#endif
}

auto DistributedDataParallel::destroy_comm_resources() -> void {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (!use_gpu_comm_) {
        return;
    }

    // Destroy per-bucket events
    for (void* evt : bucket_events_) {
        if (evt) {
            cudaEventDestroy(static_cast<cudaEvent_t>(evt));
        }
    }
    bucket_events_.clear();

    // Destroy the communication stream
    if (comm_stream_) {
        cudaStreamDestroy(static_cast<cudaStream_t>(comm_stream_));
        comm_stream_ = nullptr;
    }
#endif
}

// ============================================================================
// Forward and gradient synchronization
// ============================================================================

auto DistributedDataParallel::forward(const Variable& input) -> Variable {
    // Reset bucket states at the start of each forward pass
    if (auto_sync_enabled_) {
        reset_buckets();
    }

    return module_.forward(input);
}

auto DistributedDataParallel::auto_sync_gradients(bool enabled) -> void {
    auto_sync_enabled_ = enabled;
}

auto DistributedDataParallel::reset_buckets() -> void {
    std::lock_guard<std::mutex> lock(bucket_mutex_);
    for (auto& bucket : buckets_) {
        bucket.ready = false;
        bucket.pending_count = 0;
    }
    pending_async_ops_.store(0, std::memory_order_relaxed);
}

auto DistributedDataParallel::register_grad_hooks() -> void {
    // Build the param_to_bucket_ map so hooks can find their bucket in O(1)
    for (size_t bucket_idx = 0; bucket_idx < buckets_.size(); ++bucket_idx) {
        for (const auto& param : buckets_[bucket_idx].params) {
            if (param && param->requires_grad()) {
                const void* ptr = param->tensor().data_ptr();
                param_to_bucket_[ptr] = bucket_idx;

                // Register a backward hook on this parameter.
                // The hook captures `this` and the param's data_ptr.
                // When the gradient is computed during backward(), the hook
                // fires and marks this parameter's bucket as ready.
                const void* captured_ptr = ptr;
                param->register_hook(
                    [this, captured_ptr](const Tensor& grad) -> Tensor {
                        if (auto_sync_enabled_) {
                            mark_param_ready(captured_ptr);
                        }
                        return grad;
                    }
                );
            }
        }
    }
}

auto DistributedDataParallel::mark_param_ready(const void* param_ptr) -> void {
    std::lock_guard<std::mutex> lock(bucket_mutex_);

    auto it = param_to_bucket_.find(param_ptr);
    if (it == param_to_bucket_.end()) {
        return; // Unknown parameter, skip
    }

    size_t bucket_idx = it->second;
    GradBucket& bucket = buckets_[bucket_idx];

    if (bucket.ready) {
        return; // Already all-reduced this iteration
    }

    bucket.pending_count++;

    // Check if all parameters in this bucket have their gradients ready
    if (bucket.pending_count >= bucket.params.size()) {
        // All gradients in this bucket are ready -- fire async all-reduce.
        // On GPU backends, this launches the NCCL all-reduce on the
        // dedicated comm stream, allowing it to overlap with backward
        // computation of earlier layers still running on the compute stream.
        all_reduce_bucket_async(bucket, bucket_idx);
    }
}

auto DistributedDataParallel::synchronize_gradients() -> void {
    // All-reduce each bucket's gradients.
    // On GPU backends, this launches async all-reduce for each bucket
    // on the communication stream.
    for (size_t i = 0; i < buckets_.size(); ++i) {
        all_reduce_bucket_async(buckets_[i], i);
    }

    // Wait for all async all-reduce operations to complete before
    // returning, so the caller can safely use the averaged gradients.
    sync_comm();
}

// ============================================================================
// Async all-reduce and synchronization
// ============================================================================

auto DistributedDataParallel::all_reduce_bucket_async(
    GradBucket& bucket,
    size_t bucket_idx
) -> void {
    if (!use_gpu_comm_) {
        // CPU backend: fall back to synchronous all-reduce
        all_reduce_bucket(bucket);
        return;
    }

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    for (auto& param : bucket.params) {
        if (!param || !param->has_grad()) {
            continue;
        }

        // Get the gradient tensor
        Tensor grad = param->grad().value();

        // Launch NCCL all-reduce asynchronously on the communication stream.
        // The NCCL kernel reads gradient data from GPU memory; since the
        // gradient was just computed on the default compute stream, NCCL
        // will see the correct data because ncclAllReduce on a non-blocking
        // stream does not require explicit inter-stream synchronization --
        // NCCL internally tracks data dependencies on the same GPU.
        //
        // However, we must NOT read the gradient on the compute stream
        // until the all-reduce completes, which is handled by the event
        // recorded below.
        // Use ReduceOp::AVG to fuse the division by world_size into the
        // all-reduce kernel, eliminating a separate element-wise division.
        // NCCL natively supports AVG reduction (ncclAvg), so this is a
        // single fused communication + scaling operation.
        pg_.all_reduce_async(grad, ReduceOp::AVG, comm_stream_);
    }

    // Record a CUDA event on the communication stream after this bucket's
    // all-reduce + division completes. The compute stream will later
    // wait on this event (in sync_comm()) before the optimizer step.
    if (bucket_idx < bucket_events_.size() && bucket_events_[bucket_idx]) {
        cudaEvent_t event = static_cast<cudaEvent_t>(bucket_events_[bucket_idx]);
        cudaStream_t stream = static_cast<cudaStream_t>(comm_stream_);
        DDP_CUDA_CHECK(cudaEventRecord(event, stream));
        pending_async_ops_.fetch_add(1, std::memory_order_relaxed);
    }

    bucket.ready = true;
#else
    // Should not reach here (use_gpu_comm_ would be false without CUDA/ROCm)
    all_reduce_bucket(bucket);
#endif
}

auto DistributedDataParallel::sync_comm() -> void {
    if (!use_gpu_comm_) {
        // CPU backend: all-reduce is synchronous, nothing to wait for
        return;
    }

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (pending_async_ops_.load(std::memory_order_relaxed) == 0) {
        return; // No async operations in flight
    }

    // Make the default compute stream (stream 0) wait on all bucket events.
    // This ensures the optimizer step (which runs on the compute stream)
    // does not start until all async all-reduce operations have completed
    // on the communication stream.
    //
    // Note: We use cudaStreamWaitEvent(0, event, 0) where stream 0 is the
    // default compute stream. This is a non-blocking GPU-side wait -- the
    // CPU thread returns immediately, and the GPU will stall the compute
    // stream only if the event hasn't been reached yet on the comm stream.
    for (size_t i = 0; i < bucket_events_.size(); ++i) {
        if (bucket_events_[i]) {
            cudaEvent_t event = static_cast<cudaEvent_t>(bucket_events_[i]);
            // Stream 0 (default compute stream) waits on the comm event
            DDP_CUDA_CHECK(cudaStreamWaitEvent(nullptr, event, 0));
        }
    }

    pending_async_ops_.store(0, std::memory_order_relaxed);
#endif
}

// ============================================================================
// Synchronous all-reduce fallback (CPU backends)
// ============================================================================

auto DistributedDataParallel::all_reduce_bucket(GradBucket& bucket) -> void {
    int ws = pg_.world_size();

    for (auto& param : bucket.params) {
        if (!param || !param->has_grad()) {
            continue;
        }

        // Get the gradient tensor
        Tensor grad = param->grad().value();

        // All-reduce: sum gradients across all processes
        pg_.all_reduce(grad, ReduceOp::SUM);

        // Divide by world_size to compute average gradient.
        // This is equivalent to using ReduceOp::AVG but more explicit
        // and works with backends that may not support AVG natively.
        if (ws > 1) {
            param->set_grad(grad / static_cast<double>(ws));
        }
    }

    bucket.ready = true;
}

// ============================================================================
// Bucket building and parameter broadcast
// ============================================================================

auto DistributedDataParallel::build_buckets(size_t bucket_size_bytes) -> void {
    auto params = module_.parameters();

    if (params.empty()) {
        return;
    }

    // Reverse parameter order to match backward pass traversal.
    // In the backward pass, gradients for later layers are computed first.
    // By bucketing in reverse order, we can start all-reduce on the first
    // bucket (last layers' gradients) as soon as those gradients are ready,
    // overlapping communication with computation of earlier layers' gradients.
    std::reverse(params.begin(), params.end());

    // Build buckets by accumulating parameters until the size limit
    GradBucket current_bucket;

    for (auto& param : params) {
        if (!param || !param->requires_grad()) {
            continue;
        }

        size_t param_bytes = param->tensor().numel() *
                            dtype_size(param->tensor().dtype());

        // If adding this parameter would exceed the bucket size and the
        // current bucket is non-empty, finalize the current bucket first
        if (!current_bucket.params.empty() &&
            current_bucket.size_bytes + param_bytes > bucket_size_bytes) {
            buckets_.push_back(std::move(current_bucket));
            current_bucket = GradBucket{};
        }

        current_bucket.params.push_back(param);
        current_bucket.size_bytes += param_bytes;
    }

    // Don't forget the last bucket
    if (!current_bucket.params.empty()) {
        buckets_.push_back(std::move(current_bucket));
    }
}

auto DistributedDataParallel::broadcast_parameters() -> void {
    // Broadcast all parameters from rank 0 to ensure identical starting weights
    auto params = module_.parameters();

    for (auto& param : params) {
        if (!param) {
            continue;
        }

        Tensor& data = param->tensor();
        pg_.broadcast(data, /*src_rank=*/0);
    }

    // Also broadcast buffers (e.g., batch norm running statistics)
    auto bufs = module_.buffers();

    for (auto& buf : bufs) {
        if (!buf) {
            continue;
        }

        Tensor& data = buf->tensor();
        pg_.broadcast(data, /*src_rank=*/0);
    }
}

} // namespace tenzor::distributed
