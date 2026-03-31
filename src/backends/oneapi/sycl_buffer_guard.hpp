#pragma once

/// @file sycl_buffer_guard.hpp
/// @brief RAII wrapper for SYCL device memory to prevent leaks on exception paths.

#include <sycl/sycl.hpp>
#include <stdexcept>
#include <utility>

namespace tenzor::oneapi {

/// RAII wrapper for sycl::malloc_device / sycl::free.
/// Checks for nullptr after allocation and frees on destruction.
template<typename T>
struct SyclDeviceBuffer {
    T* ptr = nullptr;
    sycl::queue* q = nullptr;

    SyclDeviceBuffer() = default;

    SyclDeviceBuffer(size_t count, sycl::queue& queue) : q(&queue) {
        if (count > 0) {
            ptr = sycl::malloc_device<T>(count, queue);
            if (!ptr) {
                throw std::runtime_error(
                    "sycl::malloc_device returned nullptr for " +
                    std::to_string(count * sizeof(T)) + " bytes");
            }
        }
    }

    ~SyclDeviceBuffer() noexcept {
        if (ptr && q) sycl::free(ptr, *q);
    }

    SyclDeviceBuffer(const SyclDeviceBuffer&) = delete;
    SyclDeviceBuffer& operator=(const SyclDeviceBuffer&) = delete;

    SyclDeviceBuffer(SyclDeviceBuffer&& other) noexcept
        : ptr(other.ptr), q(other.q) {
        other.ptr = nullptr;
    }

    SyclDeviceBuffer& operator=(SyclDeviceBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr && q) sycl::free(ptr, *q);
            ptr = other.ptr;
            q = other.q;
            other.ptr = nullptr;
        }
        return *this;
    }

    T* get() const { return ptr; }

    /// Release ownership and return the raw pointer.
    T* release() {
        T* p = ptr;
        ptr = nullptr;
        return p;
    }

    explicit operator bool() const { return ptr != nullptr; }
};

}  // namespace tenzor::oneapi
