#pragma once

#include <memory>
#include <cstddef>
#include <atomic>
#include "device.hpp"

namespace tenzor {

// Abstract storage interface
class Storage {
public:
    virtual ~Storage() = default;

    virtual auto data() -> void* = 0;
    virtual auto data() const -> const void* = 0;
    virtual auto size_bytes() const -> size_t = 0;
    virtual auto device() const -> Device = 0;
    virtual auto ref_count() const -> int64_t = 0;
};

// CPU storage with aligned allocation
class CPUStorage : public Storage {
public:
    explicit CPUStorage(size_t size_bytes);
    ~CPUStorage() override;

    CPUStorage(const CPUStorage&) = delete;
    CPUStorage& operator=(const CPUStorage&) = delete;
    CPUStorage(CPUStorage&&) noexcept;
    CPUStorage& operator=(CPUStorage&&) noexcept;

    auto data() -> void* override { return data_; }
    auto data() const -> const void* override { return data_; }
    auto size_bytes() const -> size_t override { return size_; }
    auto device() const -> Device override { return Device::cpu(); }
    auto ref_count() const -> int64_t override { return ref_count_.load(); }

private:
    void* data_{nullptr};
    size_t size_{0};
    mutable std::atomic<int64_t> ref_count_{1};
    static constexpr size_t alignment_ = 64; // Cache line aligned
};

// Forward declaration for backend
class Backend;

// Device storage (managed by backend)
class DeviceStorage : public Storage {
public:
    DeviceStorage(void* device_ptr, size_t size_bytes,
                  Device device, Backend* backend);
    ~DeviceStorage() override;

    DeviceStorage(const DeviceStorage&) = delete;
    DeviceStorage& operator=(const DeviceStorage&) = delete;
    DeviceStorage(DeviceStorage&&) noexcept;
    DeviceStorage& operator=(DeviceStorage&&) noexcept;

    auto data() -> void* override { return device_ptr_; }
    auto data() const -> const void* override { return device_ptr_; }
    auto size_bytes() const -> size_t override { return size_; }
    auto device() const -> Device override { return device_; }
    auto ref_count() const -> int64_t override { return ref_count_.load(); }

private:
    void* device_ptr_{nullptr};
    size_t size_{0};
    Device device_;
    Backend* backend_{nullptr};
    mutable std::atomic<int64_t> ref_count_{1};
};

} // namespace tenzor
