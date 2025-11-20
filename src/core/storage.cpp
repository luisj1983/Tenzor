#include "tenzor/core/storage.hpp"
#include "tenzor/backend/backend.hpp"
#include <cstdlib>
#include <cstring>

namespace tenzor {

// CPUStorage implementation
CPUStorage::CPUStorage(size_t size_bytes) : size_(size_bytes) {
    #ifdef _WIN32
        data_ = _aligned_malloc(size_bytes, alignment_);
    #else
        if (posix_memalign(&data_, alignment_, size_bytes) != 0) {
            data_ = nullptr;
        }
    #endif

    if (!data_) {
        throw std::bad_alloc();
    }

    // Initialize memory to zero to prevent uninitialized values
    std::memset(data_, 0, size_bytes);
}

CPUStorage::~CPUStorage() {
    if (data_) {
        #ifdef _WIN32
            _aligned_free(data_);
        #else
            free(data_);
        #endif
    }
}

CPUStorage::CPUStorage(CPUStorage&& other) noexcept
    : data_(other.data_), size_(other.size_) {
    other.data_ = nullptr;
    other.size_ = 0;
}

CPUStorage& CPUStorage::operator=(CPUStorage&& other) noexcept {
    if (this != &other) {
        if (data_) {
            #ifdef _WIN32
                _aligned_free(data_);
            #else
                free(data_);
            #endif
        }
        data_ = other.data_;
        size_ = other.size_;
        other.data_ = nullptr;
        other.size_ = 0;
    }
    return *this;
}

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
