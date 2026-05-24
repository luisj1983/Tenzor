/**
 * @file async_ops.cpp
 * @brief Implementation of asynchronous tensor operations
 */

#include "tenzor/ops/async_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"

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

auto StreamManager::get_stream(const Device& device) -> StreamHandle {
    std::lock_guard lock(global_mutex_);

    auto& device_info = device_streams_[device];
    std::lock_guard device_lock(device_info.mutex);

    // Create streams if needed
    if (device_info.streams.empty()) {
        device_info.streams.reserve(STREAMS_PER_DEVICE);

#ifdef TENZOR_CUDA_ENABLED
        if (device.type == Device::Type::CUDA) {
            // Set device (audit-5 Y.8 — was unchecked).
            TENZOR_ASYNC_CUDA_CHECK(cudaSetDevice(device.index));

            // Create CUDA streams.  An unchecked failure would insert an
            // uninitialised handle into the pool and later corrupt the CUDA
            // context on the synchronize / destroy path.
            for (size_t i = 0; i < STREAMS_PER_DEVICE; ++i) {
                cudaStream_t stream;
                TENZOR_ASYNC_CUDA_CHECK(cudaStreamCreate(&stream));
                device_info.streams.push_back(static_cast<StreamHandle>(stream));
            }
        }
#endif
    }

    // Round-robin selection
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

#ifdef TENZOR_CUDA_ENABLED
    // audit-5 Y.8: was unchecked — a failure here (corrupt stream handle,
    // device lost) needs to surface, not be silently dropped.
    TENZOR_ASYNC_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
#endif
}

auto StreamManager::synchronize_device(const Device& device) -> void {
    std::lock_guard lock(global_mutex_);

    auto it = device_streams_.find(device);
    if (it == device_streams_.end()) {
        return;
    }

    auto& device_info = it->second;
    std::lock_guard device_lock(device_info.mutex);

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        for (StreamHandle stream : device_info.streams) {
            if (stream) {
                // audit-5 Y.8: was unchecked.
                TENZOR_ASYNC_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
            }
        }
    }
#endif
}

StreamManager::~StreamManager() {
#ifdef TENZOR_CUDA_ENABLED
    // Destructor cannot throw, so we log on failure instead of throwing.
    // audit-5 Y.8: previously all four CUDA calls below were unchecked.  We
    // can't propagate via exception from a destructor, but a context-lost
    // condition still warrants an error path — write to stderr so the
    // failure is visible rather than silent.
    for (auto& [device, device_info] : device_streams_) {
        if (device.type == Device::Type::CUDA) {
            cudaError_t set_err = cudaSetDevice(device.index);
            if (set_err != cudaSuccess) {
                std::fprintf(stderr,
                    "StreamManager: cudaSetDevice(%d) failed in destructor: %s\n",
                    device.index, cudaGetErrorString(set_err));
                continue;  // can't destroy streams without the right device set
            }
            for (StreamHandle stream : device_info.streams) {
                if (stream) {
                    cudaError_t dst_err = cudaStreamDestroy(static_cast<cudaStream_t>(stream));
                    if (dst_err != cudaSuccess) {
                        std::fprintf(stderr,
                            "StreamManager: cudaStreamDestroy failed: %s\n",
                            cudaGetErrorString(dst_err));
                    }
                }
            }
        }
    }
#endif
}

// Async operations implementations

auto async_matmul(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    // Check if tensors are on GPU
    [[maybe_unused]] Device device = a.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [a, b](StreamHandle stream) {
            // For now, call synchronous matmul
            // In production, this would use cuBLAS with stream parameter
            return matmul(a, b);
        });
    }
#endif

    // CPU path
    return detail::async_cpu_op([a, b]() {
        return matmul(a, b);
    });
}

auto async_conv2d(
    const Tensor& input,
    const Tensor& weight,
    [[maybe_unused]] const std::optional<Tensor>& bias,
    int64_t stride,
    int64_t padding,
    int64_t dilation,
    [[maybe_unused]] int64_t groups
) -> Future<Tensor> {
    Device device = input.device();

    // Create output tensor with proper shape calculation
    // Conv2d output: (N, out_channels, H_out, W_out)
    auto batch_size = input.shape()[0];
    auto in_height = input.shape()[2];
    auto in_width = input.shape()[3];
    auto out_channels = weight.shape()[0];
    auto kernel_h = weight.shape()[2];
    auto kernel_w = weight.shape()[3];

    auto out_height = (in_height + 2 * padding - dilation * (kernel_h - 1) - 1) / stride + 1;
    auto out_width = (in_width + 2 * padding - dilation * (kernel_w - 1) - 1) / stride + 1;

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [=](StreamHandle stream) {
            // For now, return properly shaped zero tensor
            // Full conv2d would need backend implementation
            return zeros({batch_size, out_channels, out_height, out_width}, input.dtype(), device);
        });
    }
#endif

    // CPU path: execute in thread pool
    return detail::async_cpu_op([=]() {
        // Return properly shaped zero tensor
        return zeros({batch_size, out_channels, out_height, out_width}, input.dtype(), device);
    });
}

auto async_add(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [a, b](StreamHandle stream) {
            return add(a, b);
        });
    }
#endif

    return detail::async_cpu_op([a, b]() {
        return add(a, b);
    });
}

auto async_mul(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [a, b](StreamHandle stream) {
            return mul(a, b);
        });
    }
#endif

    return detail::async_cpu_op([a, b]() {
        return mul(a, b);
    });
}

auto async_sub(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [a, b](StreamHandle stream) {
            return sub(a, b);
        });
    }
#endif

    return detail::async_cpu_op([a, b]() {
        return sub(a, b);
    });
}

auto async_div(const Tensor& a, const Tensor& b) -> Future<Tensor> {
    [[maybe_unused]] Device device = a.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [a, b](StreamHandle stream) {
            return div(a, b);
        });
    }
#endif

    return detail::async_cpu_op([a, b]() {
        return div(a, b);
    });
}

auto async_relu(const Tensor& input) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [input](StreamHandle stream) {
            return tensor_relu(input);
        });
    }
#endif

    return detail::async_cpu_op([input]() {
        return tensor_relu(input);
    });
}

auto async_sigmoid(const Tensor& input) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [input](StreamHandle stream) {
            return tensor_sigmoid(input);
        });
    }
#endif

    return detail::async_cpu_op([input]() {
        return tensor_sigmoid(input);
    });
}

auto async_tanh(const Tensor& input) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [input](StreamHandle stream) {
            return tanh(input);
        });
    }
#endif

    return detail::async_cpu_op([input]() {
        return tanh(input);
    });
}

auto async_softmax(const Tensor& input, int64_t dim) -> Future<Tensor> {
    [[maybe_unused]] Device device = input.device();

#ifdef TENZOR_CUDA_ENABLED
    if (device.type == Device::Type::CUDA) {
        return detail::async_gpu_op(device, [input, dim](StreamHandle stream) {
            return tensor_softmax(input, dim);
        });
    }
#endif

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
