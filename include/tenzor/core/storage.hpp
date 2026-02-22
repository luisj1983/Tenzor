/**
 * @file storage.hpp
 * @brief Memory storage abstraction for tensor data
 *
 * Provides abstract storage interface and concrete implementations
 * for CPU and device (GPU) memory management with reference counting.
 */

#pragma once

#include <memory>
#include <cstddef>
#include "device.hpp"

namespace tenzor {

/**
 * @brief Abstract base class for tensor memory storage.
 *
 * Defines the interface for managing raw memory buffers that back
 * tensor data. Concrete implementations handle CPU and device-specific
 * memory allocation, deallocation, and lifetime management.
 *
 * Storage objects use reference counting to enable safe sharing of
 * underlying memory between multiple tensor views.
 *
 * @note This class is not thread-safe for mutation operations.
 */
class Storage {
public:
    virtual ~Storage() = default;

    /**
     * @brief Get mutable pointer to storage data.
     *
     * @return Pointer to raw memory buffer
     */
    virtual auto data() -> void* = 0;

    /**
     * @brief Get const pointer to storage data.
     *
     * @return Const pointer to raw memory buffer
     */
    virtual auto data() const -> const void* = 0;

    /**
     * @brief Get total size of storage in bytes.
     *
     * @return Size of allocated memory in bytes
     */
    virtual auto size_bytes() const -> size_t = 0;

    /**
     * @brief Get device where storage resides.
     *
     * @return Device specification (CPU, CUDA, etc.)
     */
    virtual auto device() const -> Device = 0;

};

// Forward declaration for backend
class Backend;

/**
 * @brief Device (GPU) memory storage managed by backend.
 *
 * Manages device-side memory (CUDA, ROCm, OneAPI) through backend
 * abstractions. Memory allocation and deallocation are delegated to
 * the appropriate backend implementation.
 *
 * Unlike CPUStorage, this class does not directly manage memory but
 * coordinates with the backend system for proper lifecycle management.
 *
 * @code
 * Backend* backend = get_cuda_backend();
 * void* device_ptr = backend->allocate(1024);
 * auto storage = std::make_unique<DeviceStorage>(
 *     device_ptr, 1024, Device::cuda(0), backend
 * );
 * @endcode
 *
 * @note This class is move-only (non-copyable).
 * @warning Device pointer must be valid for the backend type.
 */
class DeviceStorage : public Storage {
public:
    /**
     * @brief Construct device storage from existing allocation.
     *
     * Takes ownership of pre-allocated device memory. The backend
     * pointer must remain valid for the lifetime of this storage.
     *
     * @param device_ptr Pointer to device memory
     * @param size_bytes Size of allocation in bytes
     * @param device Device specification
     * @param backend Backend managing this memory
     */
    DeviceStorage(void* device_ptr, size_t size_bytes,
                  Device device, Backend* backend);

    /**
     * @brief Destructor frees device memory via backend.
     *
     * Delegates memory deallocation to the backend.
     */
    ~DeviceStorage() override;

    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;

    /**
     * @brief Move constructor transfers ownership.
     *
     * @param other Source storage (left in valid but unspecified state)
     */
    DeviceStorage(DeviceStorage&&) noexcept;

    /**
     * @brief Move assignment transfers ownership.
     *
     * @param other Source storage (left in valid but unspecified state)
     * @return Reference to this
     */
    DeviceStorage& operator=(DeviceStorage&&) noexcept;

    auto data() -> void* override { return device_ptr_; }
    auto data() const -> const void* override { return device_ptr_; }
    auto size_bytes() const -> size_t override { return size_; }
    auto device() const -> Device override { return device_; }

private:
    void* device_ptr_{nullptr};                   ///< Device memory pointer
    size_t size_{0};                              ///< Size in bytes
    Device device_;                               ///< Device specification
    Backend* backend_{nullptr};                   ///< Backend managing memory
};

} // namespace tenzor
