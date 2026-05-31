/**
 * @file oneapi_kernel_utils.hpp
 * @brief Shared internal helpers for OneAPI/SYCL kernel implementations.
 *
 * These helpers were previously defined as `inline` in the named namespace
 * `tenzor::oneapi` across many kernel translation units, which is ODR-fragile.
 * This header provides the single canonical definition of each so every
 * kernel file includes one consistent implementation.
 *
 * Internal-only: not referenced by tests, Python bindings, or examples.
 */
#pragma once

#include "tenzor/core/tensor.hpp"

#include <sycl/sycl.hpp>

#include <cstdint>
#include <cstring>
#include <numeric>
#include <vector>

namespace tenzor {
namespace oneapi {

// Helper function to get a typed pointer from a tensor.
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// BFloat16 <-> Float32 conversions.
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    std::memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    std::memcpy(&bits, &f32, sizeof(uint32_t));
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// Compute contiguous (row-major) strides from a shape.
inline auto calculate_strides(const std::vector<int64_t>& shape) -> std::vector<int64_t> {
    std::vector<int64_t> strides(shape.size());
    int64_t stride = 1;
    for (int64_t i = shape.size() - 1; i >= 0; --i) {
        strides[i] = stride;
        stride *= shape[i];
    }
    return strides;
}

// Compute the number of elements from a shape.
inline auto calculate_numel(const std::vector<int64_t>& shape) -> int64_t {
    int64_t numel = 1;
    for (auto s : shape) {
        numel *= s;
    }
    return numel;
}

// Saturating float-to-half conversion: clamps to Float16 representable range
// instead of producing Infinity on overflow (matches Vulkan/GPU hardware behavior)
constexpr float HALF_MAX = 65504.0f;
inline sycl::half saturate_to_half(float val) {
    return sycl::half(sycl::fmin(sycl::fmax(val, -HALF_MAX), HALF_MAX));
}

} // namespace oneapi
} // namespace tenzor
