#pragma once

/// @file hip_buffer.hpp
/// @brief RAII wrapper for HIP device memory, shared across ROCm kernel files.

#include <hip/hip_runtime.h>
#include <stdexcept>
#include <string>

#include "rocm_error.hpp"

namespace tenzor::rocm {

/// RAII wrapper for HIP device memory.
struct HipBuffer {
    void* ptr = nullptr;

    explicit HipBuffer(size_t bytes) {
        if (bytes > 0) {
            HIP_CHECK(hipMalloc(&ptr, bytes));
        }
    }

    ~HipBuffer() noexcept {
        if (ptr) hipFree(ptr);
    }

    HipBuffer(const HipBuffer&) = delete;
    HipBuffer& operator=(const HipBuffer&) = delete;

    HipBuffer(HipBuffer&& other) noexcept : ptr(other.ptr) {
        other.ptr = nullptr;
    }

    HipBuffer& operator=(HipBuffer&& other) noexcept {
        if (this != &other) {
            if (ptr) hipFree(ptr);
            ptr = other.ptr;
            other.ptr = nullptr;
        }
        return *this;
    }

    template<typename T>
    T* as() { return static_cast<T*>(ptr); }

    template<typename T>
    const T* as() const { return static_cast<const T*>(ptr); }
};

}  // namespace tenzor::rocm
