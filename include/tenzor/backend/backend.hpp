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
 * @brief Generic operation attributes for kernel parameters.
 *
 * String-to-string map for passing backend-specific parameters to kernels.
 * Used for stride values, algorithm hints, workspace limits, etc.
 *
 * @code
 * OpAttributes attrs;
 * attrs["algorithm"] = "fft";
 * attrs["workspace_limit"] = "1073741824";  // 1GB
 * @endcode
 */
using OpAttributes = std::unordered_map<std::string, std::string>;

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

    /**
     * @brief Dispatch operation kernel.
     *
     * Main entry point for executing operations on this backend.
     * Looks up and executes the kernel for the specified operation.
     *
     * @param op_name Operation identifier (e.g., "add", "matmul", "conv2d")
     * @param inputs Input tensors for the operation
     * @param attrs Operation-specific attributes
     * @return Output tensor(s) produced by the operation
     * @throws std::runtime_error if operation is not supported
     *
     * @code
     * std::vector<Tensor> result = backend->dispatch(
     *     "matmul", {tensor_a, tensor_b}, {{"transpose_a", "false"}}
     * );
     * @endcode
     */
    virtual auto dispatch(const std::string& op_name,
                         std::span<const Tensor> inputs,
                         const OpAttributes& attrs) -> std::vector<Tensor> = 0;
};

/**
 * @brief Factory function type for backend creation.
 *
 * Function pointer type for creating backend instances.
 * Used by plugin system for dynamic backend loading.
 *
 * @return Unique pointer to newly created backend
 *
 * @code
 * extern "C" auto create_backend() -> std::unique_ptr<Backend> {
 *     return std::make_unique<MyCUDABackend>();
 * }
 * @endcode
 */
using BackendFactory = std::unique_ptr<Backend>(*)();

} // namespace tenzor
