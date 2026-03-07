#include "rocm_backend.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"
#include "tenzor/utils/logging.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <sstream>

namespace tenzor {

// ROCmBackend Implementation

ROCmBackend::ROCmBackend() {
    // Check if caching allocator is enabled via environment variable
    const char* enable_caching = std::getenv("TENZOR_ENABLE_CACHING_ALLOCATOR");
    use_caching_allocator_ = (enable_caching != nullptr && std::string(enable_caching) == "1");

    // Initialize HIP runtime by querying device count
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess && err != hipErrorNoDevice) {
        throw std::runtime_error(
            std::string("Failed to initialize ROCm backend: ") + hipGetErrorString(err)
        );
    }
}

auto ROCmBackend::name() const -> std::string_view {
    return "rocm";
}

auto ROCmBackend::device_count() const -> int32_t {
    int count = 0;
    hipError_t err = hipGetDeviceCount(&count);
    if (err != hipSuccess) {
        // Return 0 if ROCm is not available or has no devices
        return 0;
    }
    return count;
}

auto ROCmBackend::is_available() const -> bool {
    return device_count() > 0;
}

auto ROCmBackend::get_device_info(int32_t device_id) const -> DeviceInfo {
    int count = device_count();
    if (device_id < 0 || device_id >= count) {
        throw std::out_of_range("Invalid ROCm device ID: " + std::to_string(device_id) +
                                " (available: 0-" + std::to_string(count - 1) + ")");
    }

    hipDeviceProp_t props;
    hipGetDeviceProperties(&props, device_id);

    DeviceInfo info;
    info.name = props.name;
    info.vendor = "AMD";

    // Get driver version
    int driver_version = 0;
    hipDriverGetVersion(&driver_version);
    info.driver_version = std::to_string(driver_version / 100) + "." +
                          std::to_string(driver_version % 100);

    // Memory info
    info.total_memory = props.totalGlobalMem;
    size_t free_mem = 0, total_mem = 0;
    int current_device;
    hipGetDevice(&current_device);
    hipSetDevice(device_id);
    hipMemGetInfo(&free_mem, &total_mem);
    hipSetDevice(current_device);
    info.available_memory = free_mem;

    // Compute info
    info.compute_units = props.multiProcessorCount;
    info.max_threads_per_block = props.maxThreadsPerBlock;
    info.max_shared_memory = static_cast<int>(props.sharedMemPerBlock);
    info.warp_size = props.warpSize;  // 64 for AMD

    // GCN/RDNA version from architecture name
    info.major_version = props.major;
    info.minor_version = props.minor;

    // Feature support - AMD GPUs generally support these
    info.supports_fp16 = true;   // GCN 3rd gen+
    info.supports_fp64 = true;   // All GCN/RDNA
    info.supports_int8 = true;   // RDNA2+

    // Device type
    info.is_integrated = (props.integrated != 0);
    info.is_discrete = !info.is_integrated;

    // PCI info
    info.pci_bus_id = props.pciBusID;
    info.pci_device_id = props.pciDeviceID;

    return info;
}

auto ROCmBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    // Handle empty tensors - HIP doesn't like 0-byte allocations
    if (bytes == 0) {
        return nullptr;
    }

    if (use_caching_allocator_) {
        return backend::rocm::RocmCachingAllocator::get().allocate(bytes, device_id);
    }

    void* ptr = nullptr;
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in allocate");
    hipError_t err = hipMalloc(&ptr, bytes);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("Failed to allocate device memory: ") + hipGetErrorString(err)
        );
    }

    // Track allocation -> device mapping
    {
        std::lock_guard<std::mutex> lock(alloc_map_mutex_);
        alloc_device_map_[ptr] = device_id;
    }

    return ptr;
}

auto ROCmBackend::deallocate(void* ptr) -> void {
    // Handle nullptr from empty tensor allocations
    if (ptr == nullptr) {
        return;
    }

    if (use_caching_allocator_) {
        // Look up device_id from our tracking map first
        int device_id = -1;
        {
            std::lock_guard<std::mutex> lock(alloc_map_mutex_);
            auto it = alloc_device_map_.find(ptr);
            if (it != alloc_device_map_.end()) {
                device_id = it->second;
                alloc_device_map_.erase(it);
            }
        }

        // Fall back to hipPointerGetAttributes if not in our map
        if (device_id < 0) {
            hipPointerAttribute_t attrs;
            if (hipPointerGetAttributes(&attrs, ptr) == hipSuccess) {
                device_id = attrs.device;
            } else {
                throw std::runtime_error("ROCm deallocate: failed to determine device for pointer " +
                    std::to_string(reinterpret_cast<uintptr_t>(ptr)) +
                    ". Pointer was not tracked and hipPointerGetAttributes failed.");
            }
        }

        backend::rocm::RocmCachingAllocator::get().free(ptr, device_id);
        return;
    }

    // Remove from tracking map for non-caching path too
    {
        std::lock_guard<std::mutex> lock(alloc_map_mutex_);
        alloc_device_map_.erase(ptr);
    }

    check_hip_error(hipFree(ptr), "hipFree");
}

auto ROCmBackend::copy(void* dst, const void* src, size_t bytes, CopyKind kind) -> void {
    // Handle empty tensors
    if (bytes == 0) {
        return;
    }

    hipMemcpyKind hip_kind;
    switch (kind) {
        case CopyKind::HostToHost:
            hip_kind = hipMemcpyHostToHost;
            break;
        case CopyKind::HostToDevice:
            hip_kind = hipMemcpyHostToDevice;
            break;
        case CopyKind::DeviceToHost:
            hip_kind = hipMemcpyDeviceToHost;
            break;
        case CopyKind::DeviceToDevice:
            hip_kind = hipMemcpyDeviceToDevice;
            break;
    }

    // Use async copy on the default stream for non-blocking transfers
    hipError_t err = hipMemcpyAsync(dst, src, bytes, hip_kind, nullptr);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP async copy failed: ") + hipGetErrorString(err)
        );
    }
    // Only synchronize for D2H copies where caller expects data to be available
    // H2D and D2D can remain async on the default stream
    if (kind == CopyKind::DeviceToHost) {
        err = hipStreamSynchronize(nullptr);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("HIP stream sync after D2H copy failed: ") + hipGetErrorString(err)
            );
        }
    }
}

auto ROCmBackend::synchronize(int32_t device_id) -> void {
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in synchronize");
    check_hip_error(hipDeviceSynchronize(), "hipDeviceSynchronize");
}

auto ROCmBackend::create_stream(int32_t device_id) -> StreamHandle {
    hipStream_t stream;
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in create_stream");
    check_hip_error(hipStreamCreate(&stream), "hipStreamCreate");
    return static_cast<StreamHandle>(stream);
}

auto ROCmBackend::destroy_stream(StreamHandle stream) -> void {
    check_hip_error(hipStreamDestroy(static_cast<hipStream_t>(stream)), "hipStreamDestroy");
}

auto ROCmBackend::synchronize_stream(StreamHandle stream) -> void {
    check_hip_error(hipStreamSynchronize(static_cast<hipStream_t>(stream)), "hipStreamSynchronize");
}

auto ROCmBackend::memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void {
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in memset");
    check_hip_error(hipMemset(ptr, value, bytes), "hipMemset");
}

auto ROCmBackend::dispatch(const std::string& op_name,
                           std::span<const Tensor> inputs,
                           const OpAttributes& attrs) -> std::vector<Tensor> {
    throw std::runtime_error("ROCmBackend::dispatch(string): operation '" + op_name +
        "' not available via legacy string dispatch. Use OpId-based dispatch instead.");
}

auto ROCmBackend::get_device_properties(int32_t device_id) const -> hipDeviceProp_t {
    hipDeviceProp_t props;
    check_hip_error(hipGetDeviceProperties(&props, device_id), "hipGetDeviceProperties");
    return props;
}

void ROCmBackend::check_hip_error(hipError_t err, const char* operation) const {
    if (err != hipSuccess) {
        std::stringstream ss;
        ss << "ROCm operation '" << operation << "' failed: " << hipGetErrorString(err);
        throw std::runtime_error(ss.str());
    }
}

// Factory function for backend creation
extern "C" {
    auto create_backend() -> std::unique_ptr<Backend> {
        return std::make_unique<ROCmBackend>();
    }
}

} // namespace tenzor
