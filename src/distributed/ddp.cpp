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
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/utils/log.hpp"
#include <algorithm>
#include <cassert>
#include <chrono>
#include <stdexcept>
#include <numeric>
#include <iostream>

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
    size_t bucket_size_bytes,
    bool find_unused_parameters
) : module_(module), pg_(&pg), find_unused_parameters_(find_unused_parameters) {

    // Build gradient buckets from module parameters
    build_buckets(bucket_size_bytes);

    // Detect whether the process group supports async stream operations
    // (i.e., it's an NCCL backend on GPU)
    use_gpu_comm_ = pg_->supports_async_stream();

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

    // Audit D.9: one dedicated non-blocking CUDA stream per gradient
    // bucket. Mirrors PyTorch DDP's reducer.cpp where each bucket
    // gets its own NCCL stream so distinct buckets can all-reduce
    // concurrently and overlap with backward compute on stream 0.
    //
    // cudaStreamNonBlocking ensures these streams do NOT implicitly
    // synchronize with the default (stream 0) compute stream -- the
    // only ordering enforced is via the per-bucket event recorded
    // below and waited on in sync_comm().
    // Audit N.1: pre-size the per-bucket stream/event vectors to exactly
    // buckets_.size() and assert that every slot is populated with a
    // non-null handle before we leave init. The async all-reduce path
    // (all_reduce_bucket_async) and the sync_comm() wait loop both index
    // these vectors by bucket index; if any slot is null OR if the
    // vector sizes don't agree, the async path silently falls back to
    // synchronous all-reduce while sync_comm() never learns it must
    // skip the wait, producing a stale-event hazard.
    bucket_streams_.assign(buckets_.size(), nullptr);
    bucket_events_.assign(buckets_.size(), nullptr);
    for (size_t i = 0; i < buckets_.size(); ++i) {
        cudaStream_t stream = nullptr;
        DDP_CUDA_CHECK(cudaStreamCreateWithFlags(&stream,
                                                 cudaStreamNonBlocking));
        bucket_streams_[i] = static_cast<void*>(stream);

        // Events use cudaEventDisableTiming for lower overhead since
        // we only need ordering guarantees, not timing.
        cudaEvent_t event = nullptr;
        DDP_CUDA_CHECK(cudaEventCreateWithFlags(&event,
                                                 cudaEventDisableTiming));
        bucket_events_[i] = static_cast<void*>(event);
    }
    // Audit N.1: post-condition for the async path. The async dispatch
    // assumes (a) sizes match and (b) every slot is non-null. If either
    // invariant is broken we'd silently mix sync and async paths.
    assert(bucket_streams_.size() == buckets_.size());
    assert(bucket_events_.size() == buckets_.size());
    for (size_t i = 0; i < buckets_.size(); ++i) {
        assert(bucket_streams_[i] != nullptr &&
               "DDP bucket stream init produced a null handle");
        assert(bucket_events_[i] != nullptr &&
               "DDP bucket event init produced a null handle");
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

    // Destroy per-bucket communication streams
    for (void* s : bucket_streams_) {
        if (s) {
            cudaStreamDestroy(static_cast<cudaStream_t>(s));
        }
    }
    bucket_streams_.clear();
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

    // Pre-mark KNOWN-unused parameters as already accounted for, so a bucket can
    // still reach params.size() and fire its async all-reduce from the
    // contributing params alone — preserving comm/compute overlap. Without this,
    // an unused param's backward hook never fires, the bucket never reaches the
    // count, and the overlapped reduce silently degrades to the synchronous
    // force-reduce in synchronize_gradients().
    //
    // Gated on `unused_detection_cached_`: the unused set is only stable (and
    // thus safe to pre-count) once detected. The first iteration runs without
    // overlap (like PyTorch's reducer warmup), and dynamic graphs that call
    // invalidate_unused_cache() clear the flag, so a param that becomes used
    // again is never both pre-counted AND hook-fired (which would fire a bucket
    // before all live grads are ready).
    if (find_unused_parameters_ && unused_detection_cached_) {
        auto params = module_.parameters();
        for (size_t i : cached_unused_indices_) {
            if (i >= params.size() || !params[i]) continue;
            const void* ptr = params[i]->tensor().data_ptr();
            auto it = param_to_bucket_.find(ptr);
            if (it != param_to_bucket_.end()) {
                buckets_[it->second].pending_count++;
            }
        }
    }

    // Audit N.1: release-store so a subsequent acquire-load in
    // sync_comm()/all_reduce_bucket_async observes the reset.
    pending_async_ops_.store(0, std::memory_order_release);
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
    // When find_unused_parameters is enabled, detect and log unused params.
    // Detection is cached after first iteration for performance. Call
    // invalidate_unused_cache() for dynamic graphs where unused params change.
    if (find_unused_parameters_ && !unused_detection_cached_) {
        cached_unused_indices_.clear();
        std::vector<std::string> unused_names;
        auto params = module_.parameters();
        auto named = module_.named_parameters();

        for (size_t i = 0; i < params.size(); ++i) {
            if (!params[i]->has_grad()) {
                cached_unused_indices_.insert(i);
                if (i < named.size()) {
                    unused_names.push_back(named[i].first);
                }
            }
        }

        unused_detection_cached_ = true;

        if (!unused_names.empty() && !logged_unused_warning_) {
            logged_unused_warning_ = true;
            std::string msg = "DDP: find_unused_parameters detected " +
                              std::to_string(unused_names.size()) +
                              " parameter(s) without gradients: ";
            for (size_t i = 0; i < std::min(unused_names.size(), size_t(5)); ++i) {
                if (i > 0) msg += ", ";
                msg += unused_names[i];
            }
            if (unused_names.size() > 5) {
                msg += " ... and " + std::to_string(unused_names.size() - 5) + " more";
            }
            // Audit I.4: unified logger.
            TENZOR_LOG_WARN("{}", msg);
        }
    }

    // All-reduce each bucket's gradients.
    // On GPU backends, this launches async all-reduce for each bucket
    // on the communication stream.
    // Note: all_reduce_bucket already skips params without gradients,
    // so unused parameters are automatically excluded from communication.
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

namespace {
/// QQ.13: build/refresh the per-bucket DTypeGroup coalescing buffers.
/// Each call (re)packs every parameter gradient in `bucket.params` into a
/// flat F? buffer per distinct dtype.  PyTorch DDP issues ONE all-reduce
/// per bucket by this exact mechanism; without it we incur N collective
/// launches per bucket (one per parameter), which dominates step time at
/// even modest world sizes.
auto pack_bucket_flat(GradBucket& bucket) -> void {
    bucket.dtype_groups.clear();
    if (bucket.params.empty()) return;

    // First pass: group param indices by dtype (preserves bucket order
    // within a group so the scatter back reconstructs in the same order
    // and the offsets match).
    struct Plan {
        DType dt;
        Device dev;
        std::vector<size_t> indices;
        std::vector<size_t> numels;
        size_t total{0};
    };
    std::vector<Plan> plans;
    for (size_t i = 0; i < bucket.params.size(); ++i) {
        auto& p = bucket.params[i];
        if (!p || !p->has_grad()) continue;
        const Tensor& g = p->grad().value();
        const size_t n = static_cast<size_t>(g.numel());
        const DType dt = g.dtype();
        const Device dev = g.device();
        auto it = std::find_if(plans.begin(), plans.end(),
            [&](const Plan& pl) { return pl.dt == dt && pl.dev == dev; });
        if (it == plans.end()) {
            plans.push_back(Plan{dt, dev, {i}, {n}, n});
        } else {
            it->indices.push_back(i);
            it->numels.push_back(n);
            it->total += n;
        }
    }

    // Second pass: allocate the flat buffer per dtype group and
    // slice_scatter each gradient in.  The scatter mirrors FSDP's
    // collect_grads pattern (src/distributed/fsdp.cpp).
    for (auto& pl : plans) {
        GradBucket::DTypeGroup grp;
        grp.flat = zeros({static_cast<int64_t>(pl.total)}, pl.dt, pl.dev);
        grp.param_indices = std::move(pl.indices);
        grp.numels = std::move(pl.numels);
        grp.offsets.reserve(grp.numels.size());

        size_t offset = 0;
        for (size_t k = 0; k < grp.param_indices.size(); ++k) {
            const size_t pi = grp.param_indices[k];
            const size_t n  = grp.numels[k];
            const Tensor& g = bucket.params[pi]->grad().value();
            Tensor g_flat = g.reshape({static_cast<int64_t>(n)}).contiguous();
            grp.flat = slice_scatter(grp.flat, g_flat, /*dim=*/0,
                                     static_cast<int64_t>(offset),
                                     static_cast<int64_t>(offset + n));
            grp.offsets.push_back(offset);
            offset += n;
        }
        bucket.dtype_groups.push_back(std::move(grp));
    }
}

/// Scatter the reduced flat buffers back into per-parameter grads.
auto unpack_bucket_flat(GradBucket& bucket, double scale) -> void {
    for (auto& grp : bucket.dtype_groups) {
        for (size_t k = 0; k < grp.param_indices.size(); ++k) {
            const size_t pi = grp.param_indices[k];
            const size_t off = grp.offsets[k];
            const size_t n   = grp.numels[k];
            auto& p = bucket.params[pi];
            Tensor sliced = slice(grp.flat, /*dim=*/0,
                                  static_cast<int64_t>(off),
                                  static_cast<int64_t>(off + n));
            auto shape_view = p->tensor().shape();
            std::vector<int64_t> shape(shape_view.begin(), shape_view.end());
            Tensor reshaped = sliced.reshape(shape).contiguous();
            if (scale != 1.0) {
                p->set_grad(reshaped * scale);
            } else {
                p->set_grad(std::move(reshaped));
            }
        }
    }
    bucket.dtype_groups.clear();
}
} // anonymous namespace

auto DistributedDataParallel::all_reduce_bucket_async(
    GradBucket& bucket,
    size_t bucket_idx
) -> void {
    if (!use_gpu_comm_) {
        // CPU backend: fall back to synchronous all-reduce
        all_reduce_bucket(bucket);
        return;
    }

    if (compressor_) {
        // The async/flat-buffer path issues an uncompressed AVG all-reduce and
        // never consults compressor_. Honour a user-set compressor by routing
        // this bucket through the synchronous compression path (the only one
        // that applies the encoder); otherwise the compressor would be a silent
        // no-op on the GPU/NCCL path, defeating bandwidth reduction and (for
        // TopK) changing convergence.
        all_reduce_bucket(bucket);
        return;
    }

#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    // Audit D.9: each bucket all-reduces on its OWN dedicated stream
    // (PyTorch DDP reducer.cpp pattern). Distinct buckets' collectives
    // can therefore run concurrently and overlap with backward
    // compute on the default stream.
    if (bucket_idx >= bucket_streams_.size() ||
        !bucket_streams_[bucket_idx]) {
        // Defensive: stream not initialised for this bucket. Should
        // not happen because init_comm_resources() sizes the vector
        // to buckets_.size(), but fall back to sync all-reduce rather
        // than crashing.
        all_reduce_bucket(bucket);
        return;
    }

    void* this_bucket_stream = bucket_streams_[bucket_idx];

    // QQ.13: coalesce all of this bucket's gradients into ONE (or, for
    // mixed-dtype buckets, one-per-dtype) flat buffer before issuing the
    // collective.  PyTorch DDP's reducer.cpp does the same; without it we
    // would launch N collectives per bucket (10-100x overhead).
    pack_bucket_flat(bucket);

    for (auto& grp : bucket.dtype_groups) {
        // Launch NCCL all-reduce asynchronously on this bucket's
        // dedicated communication stream.  ReduceOp::AVG maps to ncclSum
        // (to_nccl_reduce_op) and all_reduce_async does NOT divide, so the
        // flat buffer holds an undivided SUM; sync_comm() divides by
        // world_size when unpacking, mirroring the synchronous path.
        pg_->all_reduce_async(grp.flat, ReduceOp::SUM, this_bucket_stream);
    }

    // Record a CUDA event on THIS bucket's stream after its
    // all-reduce(s) complete. The default compute stream will later
    // cudaStreamWaitEvent on this event before the optimizer step.
    //
    // Audit N.1: debug-build assertion that bucket_idx is in bounds
    // before the async path runs. init_comm_resources() pre-sizes
    // bucket_events_/bucket_streams_ to buckets_.size() and asserts
    // non-null, so by the time we get here both indices must be
    // valid; if not, the async record-and-increment guard below
    // would be skipped while sync_comm() still iterates every event
    // slot — exactly the silent-stale-event hazard the audit flags.
    assert(bucket_idx < bucket_events_.size() &&
           "DDP async path entered with out-of-bounds bucket_idx");
    assert(bucket_events_[bucket_idx] != nullptr &&
           "DDP async path entered with null event handle");

    // Audit N.1: paired record-and-increment. The invariant for
    // sync_comm() is "a bucket either records its event AND
    // increments pending_async_ops_, OR it does neither" — sync_comm()
    // uses the counter as a fast-path skip but iterates every event
    // slot, so a recorded-but-uncounted event is harmless while a
    // counted-but-unrecorded event would wait on a stale event from
    // a previous step. We therefore record FIRST and only bump the
    // counter once the record has succeeded (DDP_CUDA_CHECK throws
    // on failure, aborting before fetch_add).
    //
    // The fetch_add uses memory_order_release so that the matching
    // memory_order_acquire load in sync_comm() observes every prior
    // bucket_events_[i] write made by the launching thread before
    // it sees a non-zero counter — i.e. the counter is the
    // synchronisation point that publishes the recorded events.
    if (bucket_idx < bucket_events_.size() && bucket_events_[bucket_idx]) {
        cudaEvent_t event = static_cast<cudaEvent_t>(bucket_events_[bucket_idx]);
        cudaStream_t stream = static_cast<cudaStream_t>(this_bucket_stream);
        DDP_CUDA_CHECK(cudaEventRecord(event, stream));
        pending_async_ops_.fetch_add(1, std::memory_order_release);
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
    // Audit N.1: pair the acquire-load here with the release-fetch_add in
    // all_reduce_bucket_async(). Acquire semantics guarantee that if we
    // observe pending_async_ops_ > 0, we also observe every
    // bucket_events_[i] handle that was published before the counter
    // bump — closing the check-then-act window that a relaxed load
    // left open against a concurrent launcher.
    if (pending_async_ops_.load(std::memory_order_acquire) == 0) {
        return; // No async operations in flight
    }

    // Audit D.9: make the default compute stream (stream 0) wait on
    // every per-bucket event. Each event was recorded on that
    // bucket's own dedicated stream after its all-reduce completed,
    // so once stream 0 has waited on all of them, the optimizer
    // step (running on stream 0) is guaranteed to observe the
    // fully-reduced, averaged gradients for every bucket.
    //
    // cudaStreamWaitEvent(0, event, 0) is a non-blocking GPU-side
    // wait -- the CPU thread returns immediately, and the GPU
    // stalls stream 0 only if the corresponding bucket event has
    // not yet been reached on its bucket stream.
    for (size_t i = 0; i < bucket_events_.size(); ++i) {
        if (bucket_events_[i]) {
            cudaEvent_t event = static_cast<cudaEvent_t>(bucket_events_[i]);
            // Stream 0 (default compute stream) waits on the bucket event
            DDP_CUDA_CHECK(cudaStreamWaitEvent(nullptr, event, 0));
        }
    }

    // QQ.13: scatter each bucket's reduced flat buffer back into its
    // per-parameter grads.  The async all-reduce maps ReduceOp::AVG to
    // ncclSum (to_nccl_reduce_op) and NCCLProcessGroup::all_reduce_async
    // deliberately does NOT divide (it cannot do tensor math safely on an
    // async stream), so the buffer holds an undivided SUM. Divide here by
    // world_size to obtain the averaged gradient, matching the synchronous
    // path's scale = 1.0/ws in all_reduce_bucket().
    const int ws = pg_->world_size();
    const double scale = (ws > 1) ? (1.0 / static_cast<double>(ws)) : 1.0;
    for (auto& bucket : buckets_) {
        if (!bucket.dtype_groups.empty()) {
            unpack_bucket_flat(bucket, scale);
        }
    }

    // Audit N.1: release-store pairs with any subsequent acquire-load
    // (e.g. the next sync_comm() entry, or reset_buckets() before the
    // next step) so observers see the just-completed event waits.
    pending_async_ops_.store(0, std::memory_order_release);
#endif
}

// ============================================================================
// Synchronous all-reduce fallback (CPU backends)
// ============================================================================

auto DistributedDataParallel::all_reduce_bucket(GradBucket& bucket) -> void {
    int ws = pg_->world_size();
    auto start = std::chrono::high_resolution_clock::now();

    // QQ.13: coalesce per-bucket gradients into one collective per dtype
    // group.  Compression takes a special-case fallback because the
    // compressed payload is per-tensor — there is no equivalent of a
    // "compressed flat buffer" without re-doing the encoder.
    if (compressor_) {
        // Compression path: one collective per parameter (existing
        // behaviour) — coalescing here would require a compressor that
        // operates on concatenated buffers, which we don't have.
        for (auto& param : bucket.params) {
            if (!param || !param->has_grad()) continue;
            Tensor grad = param->grad().value();
            comm_stats_.total_bytes_transferred += grad.numel() * dtype_size(grad.dtype());
            auto compressed = compressor_->compress(grad);
            // Some backends (notably MPI) have no native Float16/BFloat16
            // collective datatype and throw in all_reduce. Widen a half
            // payload to Float32 for the reduction and narrow it back so
            // decompress() still sees the dtype it produced, mirroring the
            // widen/narrow pattern GlooBackend::all_reduce uses internally.
            DType payload_dtype = compressed.data.dtype();
            bool widen_half = (payload_dtype == DType::Float16 ||
                               payload_dtype == DType::BFloat16);
            if (widen_half) {
                compressed.data = compressed.data.to(DType::Float32);
            }
            pg_->all_reduce(compressed.data, ReduceOp::SUM);
            if (widen_half) {
                compressed.data = compressed.data.to(payload_dtype);
            }
            grad = compressor_->decompress(compressed);
            if (ws > 1) {
                param->set_grad(grad / static_cast<double>(ws));
            } else {
                param->set_grad(std::move(grad));
            }
        }
    } else {
        // Coalesced path: pack into per-dtype flat buffers, one
        // all_reduce per buffer (matches PyTorch DDP reducer.cpp).
        pack_bucket_flat(bucket);
        for (auto& grp : bucket.dtype_groups) {
            comm_stats_.total_bytes_transferred +=
                grp.flat.numel() * dtype_size(grp.flat.dtype());
            pg_->all_reduce(grp.flat, ReduceOp::SUM);
        }
        // Divide by world_size to compute average gradient.  Equivalent
        // to ReduceOp::AVG but works with backends that don't support
        // AVG natively.
        const double scale = (ws > 1) ? (1.0 / static_cast<double>(ws)) : 1.0;
        unpack_bucket_flat(bucket, scale);
    }

    auto end = std::chrono::high_resolution_clock::now();
    comm_stats_.total_comm_time_ms +=
        std::chrono::duration<double, std::milli>(end - start).count();
    ++comm_stats_.num_all_reduces;

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
        pg_->broadcast(data, /*src_rank=*/0);
    }

    // Also broadcast buffers (e.g., batch norm running statistics)
    auto bufs = module_.buffers();

    for (auto& buf : bufs) {
        if (!buf) {
            continue;
        }

        Tensor& data = buf->tensor();
        pg_->broadcast(data, /*src_rank=*/0);
    }
}

} // namespace tenzor::distributed
