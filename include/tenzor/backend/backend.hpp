/**
 * @file backend.hpp
 * @brief Abstract backend interface for device-specific implementations
 *
 * Defines the abstract Backend interface that all device backends (CPU, CUDA, ROCm, OneAPI)
 * must implement. Provides memory management, kernel dispatch, and synchronization primitives.
 */

#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>
#include <span>
#include <functional>
#include <unordered_map>
#include "../core/tensor.hpp"
#include "../core/device.hpp"
#include "op_attributes.hpp"

namespace tenzor {

/**
 * @brief Memory copy direction enumeration.
 *
 * Specifies the source and destination memory spaces for copy operations.
 */
enum class CopyKind {
    HostToHost,      ///< CPU to CPU copy
    HostToDevice,    ///< CPU to GPU copy
    DeviceToHost,    ///< GPU to CPU copy
    DeviceToDevice   ///< GPU to GPU copy (may be same or different devices)
};

/**
 * @brief Opaque handle for asynchronous operation streams.
 *
 * Backend-specific stream handle for managing concurrent operations.
 * Type varies by backend (cudaStream_t for CUDA, hipStream_t for ROCm, etc.).
 */
using StreamHandle = void*;

/**
 * @brief Opaque handle for synchronization events.
 *
 * Backend-specific event handle for inter-stream synchronization and timing.
 * Type varies by backend (cudaEvent_t for CUDA, hipEvent_t for ROCm, etc.).
 */
using EventHandle = void*;

/**
 * @brief Operation attributes for kernel parameters.
 *
 * Compact, cache-friendly container with SBO for up to 8 key-value pairs.
 * Uses typed AttrKey enum for O(1) lookup instead of string hashing.
 *
 * @code
 * OpAttributes attrs;
 * attrs.set(AttrKey::Algorithm, "fft");
 * attrs.set(AttrKey::WorkspaceLimit, int64_t(1073741824));  // 1GB
 * @endcode
 */
using OpAttributes = NewOpAttributes;

/**
 * @brief Device information structure.
 *
 * Contains detailed information about a specific device in a backend.
 * Properties vary by device type - GPU devices have more fields populated.
 *
 * @code
 * Backend* cuda = get_backend("cuda");
 * for (int i = 0; i < cuda->device_count(); i++) {
 *     auto info = cuda->get_device_info(i);
 *     std::cout << info.name << ": " << info.total_memory / 1e9 << " GB\n";
 * }
 * @endcode
 */
struct DeviceInfo {
    std::string name;              ///< Device name (e.g., "NVIDIA GeForce RTX 3080")
    std::string vendor;            ///< Vendor name (e.g., "NVIDIA", "AMD", "Intel")
    std::string driver_version;    ///< Driver version string

    size_t total_memory{0};        ///< Total device memory in bytes
    size_t available_memory{0};    ///< Currently available memory in bytes

    int compute_units{0};          ///< Number of compute units/SMs/CUs
    int max_threads_per_block{0};  ///< Maximum threads per block/workgroup
    int max_shared_memory{0};      ///< Maximum shared/local memory per block (bytes)
    int warp_size{0};              ///< Warp/wavefront size (32 for NVIDIA, 64 for AMD)

    int major_version{0};          ///< Compute capability major (CUDA) or similar
    int minor_version{0};          ///< Compute capability minor

    bool supports_fp16{false};     ///< Hardware FP16 support
    bool supports_fp64{false};     ///< Hardware FP64 (double) support
    bool supports_int8{false};     ///< Hardware INT8 support (tensor cores)
    bool is_integrated{false};     ///< Integrated GPU (shares system memory)
    bool is_discrete{false};       ///< Discrete GPU (dedicated memory)

    int pci_bus_id{-1};            ///< PCI bus ID (-1 if not applicable)
    int pci_device_id{-1};         ///< PCI device ID (-1 if not applicable)
};

/**
 * @brief Abstract base class for device backend implementations.
 *
 * Defines the interface that all backend implementations must provide,
 * including memory management, kernel dispatch, and synchronization.
 *
 * Concrete backends (CPUBackend, CUDABackend, etc.) inherit from this
 * class and implement all virtual methods for their specific hardware.
 *
 * @note This class uses the Strategy pattern for polymorphic dispatch.
 * @see BackendLoader for dynamic backend loading
 */
class Backend {
public:
    virtual ~Backend() = default;

    /**
     * @brief Get backend name identifier.
     *
     * @return Name string like "cpu", "cuda", "rocm", "oneapi"
     */
    virtual auto name() const -> std::string_view = 0;

    /**
     * @brief Get number of available devices.
     *
     * @return Device count (1 for CPU, N for multi-GPU systems)
     *
     * @code
     * Backend* cuda = get_backend("cuda");
     * int gpus = cuda->device_count();
     * @endcode
     */
    virtual auto device_count() const -> int32_t = 0;

    /**
     * @brief Check if backend is available and functional.
     *
     * @return true if backend can be used (drivers loaded, hardware present)
     */
    virtual auto is_available() const -> bool = 0;

    /**
     * @brief Get detailed information about a specific device.
     *
     * @param device_id Device index (0 to device_count()-1)
     * @return DeviceInfo structure with device properties
     * @throws std::out_of_range if device_id is invalid
     *
     * @code
     * Backend* cuda = get_backend("cuda");
     * for (int i = 0; i < cuda->device_count(); i++) {
     *     auto info = cuda->get_device_info(i);
     *     std::cout << "Device " << i << ": " << info.name << "\n";
     *     std::cout << "  Memory: " << info.total_memory / (1024*1024*1024) << " GB\n";
     *     std::cout << "  Compute: " << info.major_version << "." << info.minor_version << "\n";
     * }
     * @endcode
     */
    virtual auto get_device_info(int32_t device_id) const -> DeviceInfo = 0;

    /**
     * @brief Allocate device memory.
     *
     * @param bytes Number of bytes to allocate
     * @param device_id Device index for allocation
     * @return Pointer to allocated memory
     * @throws std::runtime_error if allocation fails
     *
     * @note Memory is uninitialized. Alignment is backend-specific.
     */
    virtual auto allocate(size_t bytes, int32_t device_id) -> void* = 0;

    /**
     * @brief Deallocate device memory.
     *
     * @param ptr Pointer returned by allocate()
     *
     * @note Pointer must be valid and from this backend.
     */
    virtual auto deallocate(void* ptr) -> void = 0;

    /**
     * @brief Copy memory between host and device.
     *
     * @param dst Destination pointer
     * @param src Source pointer
     * @param bytes Number of bytes to copy
     * @param kind Copy direction (host/device)
     * @throws std::runtime_error if copy fails
     *
     * @note This is a synchronous operation. Use streams for async copies.
     */
    virtual auto copy(void* dst, const void* src, size_t bytes,
                     CopyKind kind) -> void = 0;

    /**
     * @brief Synchronize device operations.
     *
     * Blocks until all operations on the device are complete.
     *
     * @param device_id Device to synchronize
     */
    virtual auto synchronize(int32_t device_id) -> void = 0;

    /**
     * @brief Create asynchronous operation stream.
     *
     * @param device_id Device for stream creation
     * @return Stream handle for async operations
     *
     * @note Caller must eventually call destroy_stream()
     */
    virtual auto create_stream(int32_t device_id) -> StreamHandle = 0;

    /**
     * @brief Destroy operation stream.
     *
     * @param stream Stream handle to destroy
     *
     * @note Stream must be synchronized before destruction.
     */
    virtual auto destroy_stream(StreamHandle stream) -> void = 0;

    /**
     * @brief Synchronize specific stream.
     *
     * Blocks until all operations in stream are complete.
     *
     * @param stream Stream to synchronize
     */
    virtual auto synchronize_stream(StreamHandle stream) -> void = 0;

    // ---- Event API for inter-stream synchronization and timing ----

    /**
     * @brief Create a synchronization event.
     *
     * @param device_id Device for event creation
     * @param enable_timing Whether the event should record timing info
     * @return Event handle for synchronization/timing
     *
     * @note Caller must eventually call destroy_event().
     *       Default implementation returns nullptr (no-op for CPU).
     */
    virtual auto create_event(int32_t device_id, bool enable_timing = true) -> EventHandle {
        (void)device_id; (void)enable_timing;
        return nullptr;
    }

    /**
     * @brief Destroy a synchronization event.
     *
     * @param event Event handle to destroy
     */
    virtual auto destroy_event(EventHandle event) -> void {
        (void)event;
    }

    /**
     * @brief Record an event on a stream.
     *
     * Marks the current point of execution in the stream. The event
     * transitions to "recorded" state when all preceding operations
     * in the stream have completed.
     *
     * @param event Event handle to record
     * @param stream Stream to record on (nullptr for default stream)
     */
    virtual auto record_event(EventHandle event, StreamHandle stream = nullptr) -> void {
        (void)event; (void)stream;
    }

    /**
     * @brief Block a stream until an event completes.
     *
     * Makes all future operations on the given stream wait until the
     * event has been recorded and all prior work in the event's stream
     * has completed.
     *
     * @param event Event to wait on
     * @param stream Stream that should wait (nullptr for default stream)
     */
    virtual auto wait_event(EventHandle event, StreamHandle stream = nullptr) -> void {
        (void)event; (void)stream;
    }

    /**
     * @brief Measure elapsed time between two events in milliseconds.
     *
     * Both events must have been recorded (via record_event) and
     * created with enable_timing=true.
     *
     * @param start_event Event recorded at the start
     * @param end_event Event recorded at the end
     * @return Elapsed time in milliseconds
     *
     * @note Default returns 0.0 for backends without timing support.
     */
    virtual auto event_elapsed_ms(EventHandle start_event, EventHandle end_event) -> float {
        (void)start_event; (void)end_event;
        return 0.0f;
    }

    /**
     * @brief Set the active device for this backend.
     *
     * Switches the hardware context to the specified device index.
     * Used by DeviceGuard for RAII device management. GPU backends
     * (CUDA, ROCm) must override this; CPU backend uses the default no-op.
     *
     * @param device_id Device index to switch to (0 to device_count()-1)
     *
     * @note This is thread-safe per CUDA/HIP specifications — each host
     *       thread maintains its own current device.
     * @see DeviceGuard for RAII usage
     */
    virtual auto set_device(int32_t device_id) -> void {
        (void)device_id;  // No-op for CPU and backends without device switching
    }

    /**
     * @brief Get the currently active device index for this backend.
     *
     * @return Current device index in the calling thread
     *
     * @note Default returns 0 (correct for CPU). GPU backends should override.
     */
    virtual auto get_current_device() const -> int32_t {
        return 0;
    }

    /**
     * @brief Fill device memory with a byte value.
     *
     * Sets all bytes in the specified memory region to the given value.
     * Primarily used for zero-initialization of GPU tensors.
     *
     * @param ptr Pointer to device memory (from allocate())
     * @param value Byte value to fill with (typically 0)
     * @param bytes Number of bytes to fill
     * @param device_id Device index where memory resides
     * @throws std::runtime_error if operation is not supported by the backend
     *
     * @note Default implementation throws. GPU backends must override.
     */
    virtual auto memset(void* ptr, int value, size_t bytes, int32_t device_id) -> void {
        (void)ptr; (void)value; (void)bytes; (void)device_id;
        throw std::runtime_error(std::string(name()) + " backend does not support memset");
    }

    // The legacy string-keyed `dispatch(op_name, inputs, attrs)` virtual was
    // removed. Production dispatch is OpId-based via DispatchTable; use
    // `dispatch_to_device(OpId::..., device.type, inputs, attrs)` instead.
    // (Removed in the pre-release audit cleanup — see audit Phase C.)
};

/**
 * @brief Factory function type for backend creation.
 *
 * Function pointer type for creating backend instances.
 * Used by plugin system for dynamic backend loading.
 *
 * @return Raw pointer to newly created backend (caller takes ownership)
 *
 * @code
 * extern "C" Backend* create_backend() {
 *     return new MyCUDABackend();
 * }
 * @endcode
 */
using BackendFactory = Backend*(*)();

} // namespace tenzor
