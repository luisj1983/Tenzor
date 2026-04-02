#include "tenzor/core/storage.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"

namespace tenzor {

// DeviceStorage implementation
DeviceStorage::DeviceStorage(void* device_ptr, size_t size_bytes, Device device)
    : device_ptr_(device_ptr), size_(size_bytes), device_(device) {}

DeviceStorage::~DeviceStorage() {
    if (!device_ptr_) {
        return;
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

DeviceStorage::DeviceStorage(DeviceStorage&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_),
      device_(other.device_) {
    other.device_ptr_ = nullptr;
}

DeviceStorage& DeviceStorage::operator=(DeviceStorage&& other) noexcept {
    if (this != &other) {
        if (device_ptr_) {
            Backend* current_backend = try_get_backend(device_.type);
            if (current_backend) {
                current_backend->deallocate(device_ptr_);
            }
        }
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        device_ = other.device_;
        other.device_ptr_ = nullptr;
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
