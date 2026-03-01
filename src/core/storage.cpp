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

    // Check if the backend registry is still alive before accessing it.
    // During static destruction, the registry may have been destroyed before
    // this DeviceStorage instance. In that case, skip deallocation.
    if (!is_backend_registry_alive()) {
        // Registry is being or has been destroyed. Skip deallocation.
        // For Vulkan, VulkanCachingAllocator::shutdown_device() already cleared
        // all blocks before the VkDevice was destroyed. For other backends,
        // this is a minor memory leak that only occurs during program shutdown,
        // which is acceptable since the process is about to exit anyway.
        return;
    }

    // Look up the backend from the registry to check if it's still alive
    Backend* current_backend = backend_registry().get_backend(device_.type);

    if (current_backend) {
        // Backend is still alive, use it normally
        current_backend->deallocate(device_ptr_);
    }
}

DeviceStorage::DeviceStorage(DeviceStorage&& other) noexcept
    : device_ptr_(other.device_ptr_), size_(other.size_),
      device_(other.device_) {
    other.device_ptr_ = nullptr;
}

DeviceStorage& DeviceStorage::operator=(DeviceStorage&& other) noexcept {
    if (this != &other) {
        if (device_ptr_ && is_backend_registry_alive()) {
            // Use the same safe deallocation as destructor
            Backend* current_backend = backend_registry().get_backend(device_.type);
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

} // namespace tenzor
