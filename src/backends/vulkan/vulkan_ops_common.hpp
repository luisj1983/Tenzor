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

} // namespace tenzor
