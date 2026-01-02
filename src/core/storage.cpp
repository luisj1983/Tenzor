#include "tenzor/core/storage.hpp"
#include "tenzor/backend/backend.hpp"

namespace tenzor {

// DeviceStorage implementation
DeviceStorage::DeviceStorage(void* device_ptr, size_t size_bytes,
                            Device device, Backend* backend)
    : device_ptr_(device_ptr), size_(size_bytes),
      device_(device), backend_(backend) {}

DeviceStorage::~DeviceStorage() {
    if (device_ptr_ && backend_) {
        backend_->deallocate(device_ptr_);
    }
}

DeviceStorage::DeviceStorage(DeviceStorage&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_),
      device_(other.device_), backend_(other.backend_) {
    other.device_ptr_ = nullptr;
    other.backend_ = nullptr;
}

DeviceStorage& DeviceStorage::operator=(DeviceStorage&& other) noexcept {
    if (this != &other) {
        if (device_ptr_ && backend_) {
            backend_->deallocate(device_ptr_);
        }
        device_ptr_ = other.device_ptr_;
        size_ = other.size_;
        device_ = other.device_;
        backend_ = other.backend_;
        other.device_ptr_ = nullptr;
        other.backend_ = nullptr;
    }
    return *this;
}

} // namespace tenzor
