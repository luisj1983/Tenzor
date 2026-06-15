#include "rocm_backend.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"
#include "tenzor/core/device_guard.hpp"
#include "tenzor/utils/logging.hpp"
#include <hip/hip_runtime.h>
#include <stdexcept>
#include <cstdlib>
#include <cstdint>
#include <cstdio>
#include <sstream>
#include <string>
#include <atomic>

namespace tenzor {

// ROCmBackend Implementation

// MIOpen runs HIPRTC at runtime to JIT-compile any kernels missing from its
// prebuilt DB (which on most ROCm builds covers only CDNA — gfx908/90a/942/950).
// HIPRTC's default include search path does not contain /opt/rocm/include, so
// kernels that #include <hip/hip_runtime.h> fail to compile with a "file not
// found" error and the failure can wedge the GPU. HIPRTC honours ROCM_PATH to
// locate the HIP headers, so we ensure it is set before any MIOpen call. We
// only set it if the user hasn't, and only if the candidate path actually
// contains hip/hip_runtime.h — so a non-standard ROCm install is unaffected.
static void ensure_rocm_path_for_hiprtc() {
    if (std::getenv("ROCM_PATH") != nullptr) {
        return;  // Respect user-provided path.
    }
    static constexpr const char* kCandidate = "/opt/rocm";
    std::string header_path = std::string(kCandidate) + "/include/hip/hip_runtime.h";
    if (FILE* f = std::fopen(header_path.c_str(), "r")) {
        std::fclose(f);
        ::setenv("ROCM_PATH", kCandidate, /*overwrite=*/0);
    }
}

ROCmBackend::ROCmBackend() {
    // Check if caching allocator is enabled via environment variable
    const char* enable_caching = std::getenv("TENZOR_ENABLE_CACHING_ALLOCATOR");
    use_caching_allocator_ = (enable_caching != nullptr && std::string(enable_caching) == "1");

    // Set ROCM_PATH for MIOpen's HIPRTC kernel JIT before any MIOpen call.
    ensure_rocm_path_for_hiprtc();

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
        return 0;
    }
    return count;
}

auto ROCmBackend::is_available() const -> bool {
    return device_count() > 0;
}

auto ROCmBackend::set_device(int32_t device_id) -> void {
    check_hip_error(hipSetDevice(device_id), "hipSetDevice");
}

auto ROCmBackend::get_current_device() const -> int32_t {
    int device_id = 0;
    check_hip_error(hipGetDevice(&device_id), "hipGetDevice");
    return device_id;
}

auto ROCmBackend::get_device_info(int32_t device_id) const -> DeviceInfo {
    int count = device_count();
    if (device_id < 0 || device_id >= count) {
        throw std::out_of_range("Invalid ROCm device ID: " + std::to_string(device_id) +
                                " (available: 0-" + std::to_string(count - 1) + ")");
    }

    hipDeviceProp_t props;
    check_hip_error(hipGetDeviceProperties(&props, device_id), "hipGetDeviceProperties");

    DeviceInfo info;
    info.name = props.name;
    info.vendor = "AMD";

    // Get driver version
    int driver_version = 0;
    check_hip_error(hipDriverGetVersion(&driver_version), "hipDriverGetVersion");
    info.driver_version = std::to_string(driver_version / 100) + "." +
                          std::to_string(driver_version % 100);

    // Memory info
    info.total_memory = props.totalGlobalMem;
    size_t free_mem = 0, total_mem = 0;
    {
        DeviceGuard guard(Device::rocm(device_id));
        check_hip_error(hipMemGetInfo(&free_mem, &total_mem), "hipMemGetInfo");
    }
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

// Check at runtime whether hipMallocAsync is available (HIP 5.3+)
static bool hip_async_alloc_available() {
    // Use an atomic cache so concurrent first-callers on arbitrary host threads
    // do not race on the memoized result (benign value, but a real data race
    // that trips TSan/UB checkers). The probe is idempotent, so racing threads
    // recomputing the same answer before the cache settles is harmless.
    static std::atomic<int> result{-1};
    int cached = result.load(std::memory_order_acquire);
    if (cached >= 0) return cached != 0;

    int computed;
#if HIP_VERSION >= 50300000
    // hipMallocAsync available at compile time; verify runtime support
    // by checking if the default memory pool exists
    hipMemPool_t pool = nullptr;
    hipError_t err = hipDeviceGetDefaultMemPool(&pool, 0);
    computed = (err == hipSuccess && pool != nullptr) ? 1 : 0;
#else
    computed = 0;
#endif
    result.store(computed, std::memory_order_release);
    return computed != 0;
}

auto ROCmBackend::allocate(size_t bytes, int32_t device_id) -> void* {
    // Handle empty tensors - HIP doesn't like 0-byte allocations
    if (bytes == 0) {
        return nullptr;
    }

    if (use_caching_allocator_) {
        return backend::rocm::RocmCachingAllocator::get().allocate(bytes, device_id);
    }

    if (device_id < 0 || device_id >= device_count()) {
        throw std::runtime_error(
            "ROCmBackend::allocate: invalid device_id " + std::to_string(device_id) +
            " (have " + std::to_string(device_count()) + " devices)");
    }

    void* ptr = nullptr;
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in allocate");

#if HIP_VERSION >= 50300000
    if (hip_async_alloc_available()) {
        hipStream_t stream = nullptr;  // default stream
        hipError_t err = hipMallocAsync(&ptr, bytes, stream);
        if (err == hipSuccess) {
            std::lock_guard<std::mutex> lock(alloc_map_mutex_);
            alloc_device_map_[ptr] = device_id;
            return ptr;
        }
        // Fall through to synchronous allocation on failure
    }
#endif

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

#if HIP_VERSION >= 50300000
    if (hip_async_alloc_available()) {
        hipStream_t stream = nullptr;  // default stream
        hipError_t err = hipFreeAsync(ptr, stream);
        if (err == hipSuccess) return;
        // Fall through to synchronous free on failure
    }
#endif
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
        default:
            throw std::runtime_error("Unknown CopyKind in ROCm copy");
    }

    // Use async copy on the default stream for non-blocking transfers
    hipError_t err = hipMemcpyAsync(dst, src, bytes, hip_kind, nullptr);
    if (err != hipSuccess) {
        throw std::runtime_error(
            std::string("HIP async copy failed: ") + hipGetErrorString(err)
        );
    }
    // copy() is documented as synchronous (backend.hpp). Host-visible copies must
    // therefore complete before returning: DeviceToHost (caller reads dst on the
    // host immediately) and HostToHost (an async H2H returns before completion, so
    // an immediate read of dst, or freeing/reusing src, would see stale/partial
    // data or hit a use-after-free). H2D and D2D are async-by-design on the default
    // stream and need no sync here.
    if (kind == CopyKind::DeviceToHost || kind == CopyKind::HostToHost) {
        err = hipStreamSynchronize(nullptr);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("HIP stream sync after host-visible copy failed: ") + hipGetErrorString(err)
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

auto ROCmBackend::create_event(int32_t device_id, bool enable_timing) -> EventHandle {
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in create_event");
    hipEvent_t event;
    unsigned flags = enable_timing ? hipEventDefault : hipEventDisableTiming;
    check_hip_error(hipEventCreateWithFlags(&event, flags), "hipEventCreateWithFlags");
    return static_cast<EventHandle>(event);
}

auto ROCmBackend::destroy_event(EventHandle event) -> void {
    if (event) {
        check_hip_error(hipEventDestroy(static_cast<hipEvent_t>(event)), "hipEventDestroy");
    }
}

auto ROCmBackend::record_event(EventHandle event, StreamHandle stream) -> void {
    check_hip_error(
        hipEventRecord(static_cast<hipEvent_t>(event), static_cast<hipStream_t>(stream)),
        "hipEventRecord");
}

auto ROCmBackend::wait_event(EventHandle event, StreamHandle stream) -> void {
    check_hip_error(
        hipStreamWaitEvent(static_cast<hipStream_t>(stream), static_cast<hipEvent_t>(event), 0),
        "hipStreamWaitEvent");
}

auto ROCmBackend::event_elapsed_ms(EventHandle start_event, EventHandle end_event) -> float {
    check_hip_error(hipEventSynchronize(static_cast<hipEvent_t>(end_event)), "hipEventSynchronize");
    float ms = 0.0f;
    check_hip_error(
        hipEventElapsedTime(&ms, static_cast<hipEvent_t>(start_event), static_cast<hipEvent_t>(end_event)),
        "hipEventElapsedTime");
    return ms;
}

auto ROCmBackend::synchronize_event(EventHandle event) -> void {
    if (!event) return;
    check_hip_error(hipEventSynchronize(static_cast<hipEvent_t>(event)),
                    "hipEventSynchronize");
}

auto ROCmBackend::memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void {
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in memset");
    check_hip_error(hipMemset(ptr, value, bytes), "hipMemset");
}

// Legacy string-keyed dispatch removed (audit Phase C).

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

auto ROCmBackend::create_hip_graph(int32_t device_id) -> std::unique_ptr<rocm::HIPGraph> {
    int count = device_count();
    if (count == 0 || device_id < 0 || device_id >= count) {
        throw std::runtime_error(
            "create_hip_graph: invalid device_id " + std::to_string(device_id) +
            " (available devices: " + std::to_string(count) + ")");
    }
    // Bind to the requested device before constructing the graph so capture/replay
    // happens on the intended GPU on multi-GPU hosts (mirrors allocate()).
    check_hip_error(hipSetDevice(device_id), "hipSetDevice in create_hip_graph");
    return std::make_unique<rocm::HIPGraph>();
}

// Factory function for backend creation
extern "C" {
    Backend* create_backend() {
        return new ROCmBackend();
    }
}

} // namespace tenzor
