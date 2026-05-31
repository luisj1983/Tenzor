/**
 * @file gradient_utils_test_support.hpp
 * @brief Test-only gradient bucketing / memory-estimation / inspection helpers.
 *
 * These symbols were previously declared in the production header
 * include/tenzor/nn/optim/gradient_utils.hpp but were only ever used by the
 * gradient_utils tests. They were relocated here to keep the production surface
 * limited to flatten_tensors() / unflatten_into() (the only functions used by
 * zero_optimizer.cpp).
 *
 * Header-only (inline) so the existing test executables can include it without
 * any CMake changes. Lives in namespace tenzor::optim so the test call sites
 * (which do `using namespace tenzor::optim;`) remain unchanged.
 */

#pragma once

#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/transform.hpp"
#include <vector>
#include <cstddef>
#include <memory>
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tenzor {
namespace optim {

/**
 * @brief Configuration for gradient bucketing
 */
struct BucketConfig {
    size_t max_bucket_size_mb{25};      ///< Maximum bucket size in megabytes
    size_t min_bucket_size_mb{1};       ///< Minimum bucket size in megabytes
    bool align_buckets{true};           ///< Align bucket boundaries to parameter boundaries
    bool sort_by_size{false};           ///< Sort parameters by size before bucketing
};

/**
 * @brief Information about a gradient bucket
 */
struct BucketInfo {
    size_t start_idx{0};                ///< Start index in parameter list
    size_t end_idx{0};                  ///< End index (exclusive) in parameter list
    size_t num_elements{0};             ///< Total number of elements in bucket
    size_t size_bytes{0};               ///< Total size in bytes
    DType dtype{DType::Float32};        ///< Data type of tensors in bucket
    Device device{Device::cpu()};       ///< Device where bucket resides
};

/**
 * @brief Memory usage estimates for gradients
 */
struct GradientMemoryStats {
    size_t total_parameters{0};         ///< Total number of parameters
    size_t total_elements{0};           ///< Total number of elements across all tensors
    size_t total_bytes{0};              ///< Total memory in bytes
    size_t bytes_per_rank{0};           ///< Memory per rank (partitioned)
    size_t num_buckets{0};              ///< Number of communication buckets
    size_t max_bucket_bytes{0};         ///< Size of largest bucket
    size_t min_bucket_bytes{0};         ///< Size of smallest bucket
    double fragmentation_ratio{0.0};    ///< Memory fragmentation (wasted space ratio)
};

// ============================================================================
// Bucket Computation
// ============================================================================

inline auto compute_bucket_sizes(const std::vector<Tensor>& tensors,
                                 const BucketConfig& config = BucketConfig{})
    -> std::vector<BucketInfo> {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot compute buckets for empty tensor vector");
    }

    const size_t max_bucket_bytes = config.max_bucket_size_mb * 1024 * 1024;
    const size_t min_bucket_bytes = config.min_bucket_size_mb * 1024 * 1024;

    std::vector<BucketInfo> buckets;

    // Sort indices by size if requested
    std::vector<size_t> indices(tensors.size());
    std::iota(indices.begin(), indices.end(), 0);

    if (config.sort_by_size) {
        std::sort(indices.begin(), indices.end(), [&tensors](size_t a, size_t b) {
            return tensors[a].numel() * dtype_size(tensors[a].dtype()) >
                   tensors[b].numel() * dtype_size(tensors[b].dtype());
        });
    }

    // Greedily assign tensors to buckets
    BucketInfo current_bucket;
    current_bucket.start_idx = 0;
    current_bucket.dtype = tensors[indices[0]].dtype();
    current_bucket.device = tensors[indices[0]].device();

    for (size_t i = 0; i < indices.size(); ++i) {
        const auto idx = indices[i];
        const auto& tensor = tensors[idx];
        const size_t tensor_bytes = tensor.numel() * dtype_size(tensor.dtype());

        // Check if we should start a new bucket
        bool start_new_bucket = false;

        // Start new bucket if dtype mismatch
        if (tensor.dtype() != current_bucket.dtype) {
            start_new_bucket = true;
        }
        // Start new bucket if size would exceed max
        else if (current_bucket.size_bytes + tensor_bytes > max_bucket_bytes &&
                 current_bucket.size_bytes >= min_bucket_bytes) {
            start_new_bucket = true;
        }

        if (start_new_bucket && current_bucket.num_elements > 0) {
            // Finalize current bucket
            current_bucket.end_idx = i;
            buckets.push_back(current_bucket);

            // Start new bucket
            current_bucket = BucketInfo{};
            current_bucket.start_idx = i;
            current_bucket.dtype = tensor.dtype();
            current_bucket.device = tensor.device();
        }

        // Add tensor to current bucket
        current_bucket.num_elements += tensor.numel();
        current_bucket.size_bytes += tensor_bytes;
    }

    // Add final bucket
    if (current_bucket.num_elements > 0) {
        current_bucket.end_idx = indices.size();
        buckets.push_back(current_bucket);
    }

    return buckets;
}

inline auto compute_bucket_sizes(const std::vector<std::shared_ptr<Variable>>& variables,
                                 const BucketConfig& config = BucketConfig{})
    -> std::vector<BucketInfo> {
    // Extract tensors from variables
    std::vector<Tensor> tensors;
    tensors.reserve(variables.size());

    for (const auto& var : variables) {
        if (var) {
            tensors.push_back(var->tensor());
        }
    }

    if (tensors.empty()) {
        throw std::invalid_argument("No Variables to compute buckets for");
    }

    return compute_bucket_sizes(tensors, config);
}

// ============================================================================
// Memory Estimation
// ============================================================================

inline auto estimate_gradient_memory(const std::vector<Tensor>& tensors,
                                     int world_size = 1,
                                     const BucketConfig& config = BucketConfig{})
    -> GradientMemoryStats {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot estimate memory for empty tensor vector");
    }

    if (world_size <= 0) {
        throw std::invalid_argument("World size must be positive");
    }

    GradientMemoryStats stats;
    stats.total_parameters = tensors.size();

    // Calculate total memory
    for (const auto& t : tensors) {
        const size_t bytes = t.numel() * dtype_size(t.dtype());
        stats.total_elements += t.numel();
        stats.total_bytes += bytes;
    }

    // Calculate per-rank memory (partitioned)
    stats.bytes_per_rank = (stats.total_bytes + world_size - 1) / world_size;

    // Compute bucket statistics
    std::vector<BucketInfo> buckets;
    try {
        buckets = compute_bucket_sizes(tensors, config);
        stats.num_buckets = buckets.size();

        if (!buckets.empty()) {
            stats.max_bucket_bytes = buckets[0].size_bytes;
            stats.min_bucket_bytes = buckets[0].size_bytes;

            for (const auto& bucket : buckets) {
                stats.max_bucket_bytes = std::max(stats.max_bucket_bytes, bucket.size_bytes);
                stats.min_bucket_bytes = std::min(stats.min_bucket_bytes, bucket.size_bytes);
            }

            // Calculate fragmentation (wasted space due to bucketing)
            size_t ideal_bucket_size = stats.total_bytes / buckets.size();
            size_t total_deviation = 0;
            for (const auto& bucket : buckets) {
                if (bucket.size_bytes > ideal_bucket_size) {
                    total_deviation += bucket.size_bytes - ideal_bucket_size;
                } else {
                    total_deviation += ideal_bucket_size - bucket.size_bytes;
                }
            }
            stats.fragmentation_ratio = static_cast<double>(total_deviation) / stats.total_bytes;
        }
    } catch (const std::exception&) {
        // Bucketing failed, use defaults
        stats.num_buckets = 0;
        stats.max_bucket_bytes = stats.total_bytes;
        stats.min_bucket_bytes = stats.total_bytes;
        stats.fragmentation_ratio = 0.0;
    }

    return stats;
}

inline auto estimate_gradient_memory(const std::vector<std::shared_ptr<Variable>>& variables,
                                     int world_size = 1,
                                     const BucketConfig& config = BucketConfig{})
    -> GradientMemoryStats {
    // Extract tensors from variables
    std::vector<Tensor> tensors;
    tensors.reserve(variables.size());

    for (const auto& var : variables) {
        if (var) {
            tensors.push_back(var->tensor());
        }
    }

    if (tensors.empty()) {
        throw std::invalid_argument("No Variables to estimate memory for");
    }

    return estimate_gradient_memory(tensors, world_size, config);
}

// ============================================================================
// Helper Functions
// ============================================================================

inline auto all_contiguous(const std::vector<Tensor>& tensors) -> bool {
    return std::all_of(tensors.begin(), tensors.end(),
                      [](const Tensor& t) { return t.is_contiguous(); });
}

inline auto same_dtype(const std::vector<Tensor>& tensors) -> bool {
    if (tensors.empty()) {
        return true;
    }

    const auto dtype = tensors[0].dtype();
    return std::all_of(tensors.begin() + 1, tensors.end(),
                      [dtype](const Tensor& t) { return t.dtype() == dtype; });
}

inline auto same_device(const std::vector<Tensor>& tensors) -> bool {
    if (tensors.empty()) {
        return true;
    }

    const auto& device = tensors[0].device();
    return std::all_of(tensors.begin() + 1, tensors.end(),
                      [&device](const Tensor& t) {
                          const auto& t_device = t.device();
                          return t_device.type == device.type &&
                                 t_device.index == device.index;
                      });
}

inline auto total_numel(const std::vector<Tensor>& tensors) -> size_t {
    size_t total = 0;
    for (const auto& t : tensors) {
        total += t.numel();
    }
    return total;
}

inline auto total_bytes(const std::vector<Tensor>& tensors) -> size_t {
    size_t total = 0;
    for (const auto& t : tensors) {
        total += t.numel() * dtype_size(t.dtype());
    }
    return total;
}

inline auto extract_shapes(const std::vector<Tensor>& tensors)
    -> std::vector<std::vector<int64_t>> {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(tensors.size());

    for (const auto& t : tensors) {
        auto shape_span = t.shape();
        shapes.emplace_back(shape_span.begin(), shape_span.end());
    }

    return shapes;
}

inline auto make_contiguous(const std::vector<Tensor>& tensors) -> std::vector<Tensor> {
    std::vector<Tensor> result;
    result.reserve(tensors.size());

    for (const auto& t : tensors) {
        result.push_back(t.is_contiguous() ? t : t.contiguous());
    }

    return result;
}

} // namespace optim
} // namespace tenzor
