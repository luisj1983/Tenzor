/**
 * @file async_ops.cpp
 * @brief Implementation of asynchronous tensor operations
 */

#include "tenzor/ops/async_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/utils/logging.hpp"
#include <cstdio>
#include <stdexcept>

#ifdef TENZOR_CUDA_ENABLED
#include <cuda_runtime.h>
#include <cstdio>
#include <stdexcept>
#include <string>

// audit-5 Y.8: local mirror of `tenzor::cuda::TENZOR_CUDA_CHECK` so the
// StreamManager doesn't silently swallow `cudaSetDevice` / `cudaStreamCreate` /
// `cudaStreamSynchronize` / `cudaStreamDestroy` failures.  We keep it local
// rather than pulling in `cuda_common.cuh` because that header drags the
// kernel-side intrinsics; we only need the runtime error check here.
#ifndef TENZOR_ASYNC_CUDA_CHECK
#define TENZOR_ASYNC_CUDA_CHECK(call)                                          \
    do {                                                                        \
        cudaError_t _e = (call);                                                \
        if (_e != cudaSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("CUDA error in StreamManager (") + #call + "): " + \
                cudaGetErrorString(_e));                                        \
        }                                                                       \
    } while (0)
#endif
#endif

namespace tenzor {

// Helper functions for tensor-based operations (not Variable-based)

namespace {

// Simple tensor-based activation implementations
auto tensor_relu(const Tensor& input) -> Tensor {
    // For Float16/BFloat16, promote to Float32, apply ReLU, then convert back
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        auto input_f32 = input.to(DType::Float32);
        auto result_f32 = clamp_min(input_f32, 0.0f);
        return result_f32.to(input.dtype());
    }
    // Use clamp_min which dispatches to the correct backend (CPU/CUDA/etc.)
    return clamp_min(input, 0.0f);
}

auto tensor_sigmoid(const Tensor& input) -> Tensor {
    // Use sigmoid() which dispatches to the correct backend (CPU/CUDA/etc.)
    return sigmoid(input);
}

auto tensor_softmax(const Tensor& input, int64_t dim) -> Tensor {
    // For simplicity, use the autograd version wrapped in a tensor operation
    // In production, this would have a dedicated tensor-only implementation
    Variable var(input, false);
    Variable result = softmax(var, dim);
    return result.tensor();
}

} // anonymous namespace

// StreamManager implementation

auto StreamManager::instance() -> StreamManager& {
    static StreamManager manager;
    return manager;
}

// Streams are created/synchronized/destroyed through the backend interface,
// so every GPU backend (CUDA streams, ROCm hipStreams, OneAPI in-order SYCL
// queues, Vulkan timeline contexts) gets a real per-device stream pool. The
// core layer holds only opaque StreamHandles; the owning backend (which links
// its native runtime) performs the actual operations.
auto StreamManager::get_stream(const Device& device) -> StreamHandle {
    if (device.type == Device::Type::CPU) {
        return nullptr;  // CPU async uses the thread pool, not a device stream.
    }

    std::lock_guard lock(global_mutex_);
    auto& device_info = device_streams_[device];

    if (device_info.streams.empty()) {
        Backend* backend = is_backend_registry_alive()
            ? backend_registry().get_backend(device.type) : nullptr;
        if (!backend) {
            return nullptr;  // backend not loaded — fall back to thread-pool path.
        }
        device_info.streams.reserve(STREAMS_PER_DEVICE);
        for (size_t i = 0; i < STREAMS_PER_DEVICE; ++i) {
            StreamHandle stream = backend->create_stream(device.index);
            if (!stream) break;
            device_info.streams.push_back(stream);
            stream_owner_[stream] = backend;
        }
    }

    if (device_info.streams.empty()) {
        return nullptr;
    }

    StreamHandle stream = device_info.streams[device_info.next_index];
    device_info.next_index = (device_info.next_index + 1) % device_info.streams.size();
    return stream;
}

auto StreamManager::synchronize_stream(StreamHandle stream) -> void {
    if (!stream) {
        return;
    }
    // The owning Backend* is only valid while the backend registry is alive;
    // after finalize()/shutdown() the cached pointer dangles. Guard before use
    // (mirrors the destructor) so a late async continuation can't deref a freed
    // backend.
    if (!is_backend_registry_alive()) {
        return;
    }
    Backend* owner = nullptr;
    {
        std::lock_guard lock(global_mutex_);
        auto it = stream_owner_.find(stream);
        if (it != stream_owner_.end()) owner = it->second;
    }
    if (owner) owner->synchronize_stream(stream);
}

auto StreamManager::synchronize_device(const Device& device) -> void {
    if (!is_backend_registry_alive()) {
        return;
    }
    std::lock_guard lock(global_mutex_);

    auto it = device_streams_.find(device);
    if (it == device_streams_.end()) {
        return;
    }
    for (StreamHandle stream : it->second.streams) {
        if (!stream) continue;
        auto owner_it = stream_owner_.find(stream);
        if (owner_it != stream_owner_.end()) owner_it->second->synchronize_stream(stream);
    }
}

auto StreamManager::reset() -> void {
    // Destroy all pooled streams through their owning backends and clear the
    // pool. Called from finalize()/shutdown() BEFORE backends are torn down so
    // a subsequent re-initialize() rebuilds streams against live backends
    // (otherwise get_stream would hand out handles from a freed backend).
    std::lock_guard lock(global_mutex_);
    if (is_backend_registry_alive()) {
        for (auto& [stream, owner] : stream_owner_) {
            if (stream && owner) {
                try { owner->destroy_stream(stream); }
                catch (...) { std::fprintf(stderr, "StreamManager::reset: destroy_stream failed\n"); }
            }
        }
    }
    stream_owner_.clear();
    device_streams_.clear();
}

StreamManager::~StreamManager() {
    // Destructors must not throw. Destroy every stream through its owning
    // backend; the backend registry may already be torn down at static-dtor
    // time, in which case the OS reclaims the streams.
    if (!is_backend_registry_alive()) {
        return;
    }
    for (auto& [stream, owner] : stream_owner_) {
        if (stream && owner) {
            try {
                owner->destroy_stream(stream);
            } catch (...) {
                std::fprintf(stderr, "StreamManager: destroy_stream failed in destructor\n");
            }
        }
    }
}

// Async operations implementations

auto async_matmul(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    // Check if tensors are on GPU
    [[maybe_unused]] Device device = a.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [a, b](StreamHandle /*stream*/) {
            // Call synchronous matmul; the wrapping async_gpu_op enqueues on
            // a dedicated CUDA stream so callers can overlap multiple matmuls.
            // A cuBLAS-with-stream variant is a future optimisation but does
            // not change the externally observable contract.
            return matmul(a, b);
        });
    }

    // CPU path
    return detail::async_cpu_op([a, b]() {
        return matmul(a, b);
    });
}

auto async_add(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [a, b](StreamHandle /*stream*/) {
            return add(a, b);
        });
    }

    return detail::async_cpu_op([a, b]() {
        return add(a, b);
    });
}

auto async_mul(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [a, b](StreamHandle /*stream*/) {
            return mul(a, b);
        });
    }

    return detail::async_cpu_op([a, b]() {
        return mul(a, b);
    });
}

auto async_sub(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [a, b](StreamHandle /*stream*/) {
            return sub(a, b);
        });
    }

    return detail::async_cpu_op([a, b]() {
        return sub(a, b);
    });
}

auto async_div(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [a, b](StreamHandle /*stream*/) {
            return div(a, b);
        });
    }

    return detail::async_cpu_op([a, b]() {
        return div(a, b);
    });
}

auto async_relu(const Tensor& input) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [input](StreamHandle /*stream*/) {
            return tensor_relu(input);
        });
    }

    return detail::async_cpu_op([input]() {
        return tensor_relu(input);
    });
}

auto async_sigmoid(const Tensor& input) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [input](StreamHandle /*stream*/) {
            return tensor_sigmoid(input);
        });
    }

    return detail::async_cpu_op([input]() {
        return tensor_sigmoid(input);
    });
}

auto async_tanh(const Tensor& input) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [input](StreamHandle /*stream*/) {
            return tanh(input);
        });
    }

    return detail::async_cpu_op([input]() {
        return tanh(input);
    });
}

auto async_softmax(const Tensor& input, int64_t dim) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

    if (device.type != Device::Type::CPU) {
        return detail::async_gpu_op(device, [input, dim](StreamHandle /*stream*/) {
            return tensor_softmax(input, dim);
        });
    }

    return detail::async_cpu_op([input, dim]() {
        return tensor_softmax(input, dim);
    });
}

auto wait_all(std::vector<Future<Tensor>>& futures) -> std::vector<Tensor> {
    std::vector<Tensor> results;
    results.reserve(futures.size());

    for (auto& future : futures) {
        results.push_back(future.wait());
    }

    return results;
}

auto wait_any(const std::vector<Future<Tensor>>& futures) -> int64_t {
    // Simple polling implementation — return the first future that becomes
    // ready. Don't return the moment one is ready, though: the unfinished
    // futures may still be mid-flight in the thread pool, capturing Tensors
    // by value. If the caller lets the `futures` vector die before those
    // background tasks finish, the captured Tensors get destroyed while the
    // async op is still writing through their storage, producing
    // "corrupted double-linked list" / SEGFAULT on teardown (seen on
    // AsyncOpsMultiDTypeTest.WaitAny / Oneapi0_Float16).
    //
    // Block on *all* futures after finding the first-ready one so background
    // work completes deterministically. `wait_any` still returns the index
    // of the future that finished first, matching the contract.
    int64_t first_ready = -1;
    while (first_ready < 0) {
        for (size_t i = 0; i < futures.size(); ++i) {
            if (futures[i].is_ready()) {
                first_ready = static_cast<int64_t>(i);
                break;
            }
        }
        if (first_ready < 0) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    // Spin until every future has signalled completion (is_ready()==true).
    // We do NOT call `.wait()` — the Future<T>::wait() API consumes the
    // shared state's value and is not const-callable, and the caller may
    // still want to call `.wait()` later.
    for (const auto& f : futures) {
        while (!f.is_ready()) {
            std::this_thread::sleep_for(std::chrono::microseconds(100));
        }
    }
    return first_ready;
}

} // namespace tenzor
