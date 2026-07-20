#include "tenzor/core/gpu_stream_ops.hpp"
#include "tenzor/core/rocm_transfer.hpp"
#include "tenzor/backend/loader.hpp"
#include <stdexcept>
#include <string>

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>

#define TENZOR_GPU_STREAM_CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            throw std::runtime_error( \
                std::string("CUDA error in gpu_stream_ops: ") + cudaGetErrorString(err)); \
        } \
    } while (0)
#endif

namespace tenzor::core::gpu_stream {

auto create_stream(Device::Type type) -> void* {
    if (type == Device::Type::ROCm) {
        return tenzor::rocm_transfer::stream_create();
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        cudaStream_t stream = nullptr;
        TENZOR_GPU_STREAM_CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
        return static_cast<void*>(stream);
    }
#endif
    return nullptr;
}

auto destroy_stream(void* stream, Device::Type type) -> void {
    if (!stream) return;
    if (type == Device::Type::ROCm) {
        tenzor::rocm_transfer::stream_destroy(stream);
        return;
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        cudaStreamDestroy(static_cast<cudaStream_t>(stream));
        return;
    }
#endif
}

auto create_event(Device::Type type) -> void* {
    if (type == Device::Type::ROCm) {
        return tenzor::rocm_transfer::event_create();
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        cudaEvent_t event = nullptr;
        TENZOR_GPU_STREAM_CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));
        return static_cast<void*>(event);
    }
#endif
    return nullptr;
}

auto destroy_event(void* event, Device::Type type) -> void {
    if (!event) return;
    if (type == Device::Type::ROCm) {
        tenzor::rocm_transfer::event_destroy(event);
        return;
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        cudaEventDestroy(static_cast<cudaEvent_t>(event));
        return;
    }
#endif
}

auto record_event(void* event, void* stream, Device::Type type) -> void {
    if (!event) return;
    if (type == Device::Type::ROCm) {
        tenzor::rocm_transfer::event_record(event, stream);
        return;
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        TENZOR_GPU_STREAM_CUDA_CHECK(
            cudaEventRecord(static_cast<cudaEvent_t>(event), static_cast<cudaStream_t>(stream)));
        return;
    }
#endif
}

auto stream_wait_event(void* stream, void* event, Device::Type type) -> void {
    if (!event) return;
    if (type == Device::Type::ROCm) {
        tenzor::rocm_transfer::stream_wait_event(stream, event);
        return;
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        TENZOR_GPU_STREAM_CUDA_CHECK(cudaStreamWaitEvent(
            static_cast<cudaStream_t>(stream), static_cast<cudaEvent_t>(event), 0));
        return;
    }
#endif
}

auto synchronize_stream(void* stream, Device::Type type) -> void {
    if (!stream) return;
    if (type == Device::Type::ROCm) {
        tenzor::rocm_transfer::stream_synchronize(stream);
        return;
    }
#ifdef TENZOR_USE_CUDA
    if (type == Device::Type::CUDA) {
        TENZOR_GPU_STREAM_CUDA_CHECK(cudaStreamSynchronize(static_cast<cudaStream_t>(stream)));
        return;
    }
#endif
}

auto mem_get_info(Device device, std::size_t* free_bytes, std::size_t* total_bytes) -> bool {
    if (device.type == Device::Type::ROCm) {
        return tenzor::rocm_transfer::mem_get_info(device.index, free_bytes, total_bytes);
    }
#ifdef TENZOR_USE_CUDA
    if (device.type == Device::Type::CUDA) {
        (void)cudaSetDevice(device.index);
        return cudaMemGetInfo(free_bytes, total_bytes) == cudaSuccess;
    }
#endif
    if (device.type == Device::Type::Vulkan || device.type == Device::Type::OneAPI ||
        device.type == Device::Type::MPS) {
        // Vulkan/OneAPI/MPS have no direct cudaMemGetInfo equivalent reachable from
        // this header-only core TU without pulling in vendor headers, but all three
        // backends already implement the generic Backend::get_device_info() query
        // (real hardware-reported total_memory; available_memory is that same
        // backend's own best-effort free estimate -- Vulkan lacks a portable
        // free-memory query without the VK_EXT_memory_budget extension, SYCL has no
        // standard free-memory query, and MPS/Metal likewise reports total_memory as
        // available_memory, same as the rest of this codebase's convention for these
        // backends). Route through that instead of a CPU fallback.
        tenzor::Backend* backend = tenzor::try_get_backend(device.type);
        if (backend == nullptr) {
            return false;
        }
        tenzor::DeviceInfo info = backend->get_device_info(device.index);
        if (info.total_memory == 0) {
            return false;
        }
        if (total_bytes != nullptr) {
            *total_bytes = info.total_memory;
        }
        if (free_bytes != nullptr) {
            *free_bytes = info.available_memory;
        }
        return true;
    }
    return false;
}

}  // namespace tenzor::core::gpu_stream
