/**
 * @file vulkan_ops_common.hpp
 * @brief Shared includes and utilities for Vulkan operation implementations
 */

#pragma once

#include "vulkan_helpers.hpp"
#include "tenzor/backend/vulkan_caching_allocator.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/utils/logging.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstring>
#include <limits>
#include <numeric>
#include <optional>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <unordered_set>
#include <vector>

// Undefine Xlib Bool macro that conflicts with DType::Bool
// vulkan_caching_allocator.hpp re-includes <vulkan/vulkan.h> which re-defines it
#ifdef Bool
#undef Bool
#endif

namespace tenzor {

// Clamps FP16 values to max finite range, replacing ±Inf with ±65504.
// This matches the CUDA/ROCm fp16_saturate behavior to prevent NaN
// propagation when FP32 compute produces values outside Float16 range.
// Returns the saturated tensor (new allocation via dispatchClamp).
static inline auto fp16_saturate_if_needed(
        VulkanBackend& backend, const Tensor& output) -> Tensor {
    if (output.dtype() == DType::Float16 && output.numel() > 0) {
        return backend.dispatchClamp(output, -65504.0f, 65504.0f);
    }
    return output;
}

inline void vulkan_assert_dtype_supported(
    const char* op_name, DType dtype, std::initializer_list<DType> supported) {
    for (auto d : supported) if (dtype == d) return;
    throw std::runtime_error(std::string(op_name) + ": unsupported dtype " +
        std::string(dtype_name(dtype)) + " on Vulkan backend");
}

// The IndexSelect/Gather/Take/Scatter compute shaders have no error-reporting
// path back to the host (unlike CUDA/ROCm's device-error-flag pattern), so an
// out-of-range index previously silently wrote 0.0 (gather/take/index_select)
// or silently skipped the write (scatter) instead of raising, diverging from
// CPU (throws std::out_of_range), CUDA (device error-flag + host throw), ROCm
// (validate_index_bounds), and OneAPI (host-side pre-check). Validate host-side
// before ever dispatching the shader, mirroring dispatchPut's own fix for the
// same root-cause gap.
inline void vulkan_validate_index_bounds(
    const Tensor& indices, int64_t dim_size, const char* op_name) {
    int64_t n = indices.numel();
    if (n == 0) return;
    Tensor idx_host = indices.contiguous().to(Device::cpu());
    auto check = [&](int64_t raw) {
        int64_t v = raw;
        if (v < 0) v += dim_size;
        if (v < 0 || v >= dim_size) {
            throw std::out_of_range(
                std::string(op_name) + ": index " + std::to_string(raw) +
                " out of range for dimension of size " + std::to_string(dim_size));
        }
    };
    if (idx_host.dtype() == DType::Int64) {
        const int64_t* p = idx_host.data<int64_t>();
        for (int64_t i = 0; i < n; ++i) check(p[i]);
    } else if (idx_host.dtype() == DType::Int32) {
        const int32_t* p = idx_host.data<int32_t>();
        for (int64_t i = 0; i < n; ++i) check(static_cast<int64_t>(p[i]));
    } else {
        throw std::invalid_argument(
            std::string(op_name) + ": index tensor must have dtype Int32 or Int64");
    }
}

} // namespace tenzor
