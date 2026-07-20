#include "tenzor/core/storage.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include "tenzor/core/rocm_transfer.hpp"

#ifdef TENZOR_USE_CUDA
#include <cuda_runtime.h>
#endif

namespace tenzor {

// DeviceStorage implementation
DeviceStorage::DeviceStorage(void* device_ptr, size_t size_bytes, Device device)
    : device_ptr_(device_ptr), size_(size_bytes), device_(device) {}

DeviceStorage::~DeviceStorage() {
    if (!device_ptr_) {
        return;
    }

    if (pinned_ && device_.type == Device::Type::CPU) {
        if (pinned_via_rocm_) {
            // Pinned via hipHostRegister (rocm_transfer.hip.cpp, isolated TU —
            // see rocm_transfer.hpp for why HIP headers can't appear here
            // directly). No-op if ROCm isn't linked in.
            rocm_transfer::host_unregister(device_ptr_);
        }
#ifdef TENZOR_USE_CUDA
        else {
            // If this CPU storage was pinned via cudaHostRegister, unregister
            // before deallocation. Ignore errors — if CUDA has already
            // finalized or the handle is stale, the deallocation path below
            // still runs.
            cudaHostUnregister(device_ptr_);
        }
#endif
    }

    // Atomically check if registry is alive and get backend in one call,
    // eliminating the TOCTOU race between is_backend_registry_alive() and
    // get_backend(). Returns nullptr if registry is being destroyed.
    Backend* current_backend = try_get_backend(device_.type);
    if (current_backend) {
        current_backend->deallocate(device_ptr_);
    }
    // If nullptr: registry destroyed or backend unloaded — minor leak at
    // shutdown only, acceptable since process is about to exit.
}

auto DeviceStorage::pin() -> bool {
    // Only CPU storage is pinnable — GPU memory is already accessible
    // to the CUDA driver without registration.
    if (device_.type != Device::Type::CPU) {
        return false;
    }
    if (pinned_) {
        return true;  // already pinned
    }
    if (!device_ptr_ || size_ == 0) {
        return false;
    }
#ifdef TENZOR_USE_CUDA
    // cudaHostRegister page-locks the provided host allocation so
    // subsequent cudaMemcpy calls can DMA directly into/out of it
    // without a staging copy. The kernel rejects non-writable pages
    // and over-registered regions — fall through silently on failure.
    cudaError_t err = cudaHostRegister(device_ptr_, size_, cudaHostRegisterDefault);
    if (err == cudaSuccess) {
        pinned_ = true;
        pinned_via_rocm_ = false;
        return true;
    }
    // Clear the error so it doesn't propagate to the next CUDA call.
    cudaGetLastError();
#endif
    // hipHostRegister mirrors cudaHostRegister for ROCm — page-locks the host
    // allocation so hipMemcpyAsync (rocm_transfer.hip.cpp) can DMA directly
    // into/out of it. No-op (returns false) when ROCm isn't linked in, or on
    // a build that only has CUDA. Tried after CUDA so a combined CUDA+ROCm
    // build keeps CUDA's pin as the default when both are present.
    if (rocm_transfer::host_register(device_ptr_, size_)) {
        pinned_ = true;
        pinned_via_rocm_ = true;
        return true;
    }
    return false;
}

DeviceStorage::DeviceStorage(DeviceStorage&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_),
      device_(other.device_), pinned_(other.pinned_),
      pinned_via_rocm_(other.pinned_via_rocm_) {
    // Transfer the pinned flag so the moved-to object's destructor performs the
    // matching cudaHostUnregister/hipHostUnregister; otherwise the page-locked
    // registration leaks and is_pinned() would lie. Clear it on the moved-from
    // object whose pointer is now null (its destructor early-returns).
    other.device_ptr_ = nullptr;
    other.pinned_ = false;
    other.pinned_via_rocm_ = false;
}

DeviceStorage& DeviceStorage::operator=(DeviceStorage&& other) noexcept {
    if (this != &other) {
        if (device_ptr_) {
            if (pinned_ && device_.type == Device::Type::CPU) {
                if (pinned_via_rocm_) {
                    rocm_transfer::host_unregister(device_ptr_);
                }
#ifdef TENZOR_USE_CUDA
                else {
                    // Unregister the existing target buffer if it was pinned
                    // before we overwrite the pointer, else its page-locked
                    // registration leaks.
                    cudaHostUnregister(device_ptr_);
                }
#endif
            }
            Backend* current_backend = try_get_backend(device_.type);
            if (current_backend) {
                current_backend->deallocate(device_ptr_);
            }
        }
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        device_ = other.device_;
        pinned_ = other.pinned_;
        pinned_via_rocm_ = other.pinned_via_rocm_;
        other.device_ptr_ = nullptr;
        other.pinned_ = false;
        other.pinned_via_rocm_ = false;
    }
    return *this;
}

// ExternalStorage implementation

ExternalStorage::ExternalStorage(void* ptr, size_t size_bytes, Device device,
                                 Deleter deleter)
    : ptr_(ptr), size_(size_bytes), device_(device),
      deleter_(std::move(deleter)) {}

ExternalStorage::~ExternalStorage() {
    if (ptr_ && deleter_) {
        deleter_(ptr_);
    }
    // If no deleter: external memory is not our responsibility
}

ExternalStorage::ExternalStorage(ExternalStorage&& other) noexcept
    : ptr_(other.ptr_), size_(other.size_),
      device_(other.device_), deleter_(std::move(other.deleter_)) {
    other.ptr_ = nullptr;
}

ExternalStorage& ExternalStorage::operator=(ExternalStorage&& other) noexcept {
    if (this != &other) {
        if (ptr_ && deleter_) {
            deleter_(ptr_);
        }
        ptr_ = other.ptr_;
        size_ = other.size_;
        device_ = other.device_;
        deleter_ = std::move(other.deleter_);
        other.ptr_ = nullptr;
    }
    return *this;
}

} // namespace tenzor
