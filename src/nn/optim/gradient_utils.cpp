/**
 * @file gradient_utils.cpp
 * @brief Implementation of gradient utility functions for ZeRO Stage 2
 */

#include "tenzor/nn/optim/gradient_utils.hpp"
#include "tenzor/ops/transform.hpp"
#include <algorithm>
#include <numeric>
#include <stdexcept>

namespace tenzor {
namespace optim {

namespace {

/**
 * @brief Validate that all tensors are compatible for flattening
 */
void validate_tensors_for_flattening(const std::vector<Tensor>& tensors) {
    if (tensors.empty()) {
        throw std::invalid_argument("Cannot flatten empty tensor vector");
    }

    const auto& first = tensors[0];
    auto dtype = first.dtype();
    auto device = first.device();

    for (size_t i = 1; i < tensors.size(); ++i) {
        if (tensors[i].dtype() != dtype) {
            throw std::invalid_argument(
                "All tensors must have the same dtype for flattening. "
                "Expected " + std::string(dtype_name(dtype)) +
                " but got " + std::string(dtype_name(tensors[i].dtype())) +
                " at index " + std::to_string(i)
            );
        }
        const auto& curr_device = tensors[i].device();
        if (curr_device.type != device.type ||
            curr_device.index != device.index) {
            throw std::invalid_argument(
                "All tensors must be on the same device for flattening"
            );
        }
    }
}

} // anonymous namespace

// ============================================================================
// Tensor Flattening
// ============================================================================

auto flatten_tensors(const std::vector<Tensor>& tensors) -> Tensor {
    validate_tensors_for_flattening(tensors);

    // Flatten each tensor to 1D and concatenate using cat() which is device-safe
    // (dispatches to GPU kernel for CUDA tensors, avoids raw data_ptr() access)
    std::vector<Tensor> flat_parts;
    flat_parts.reserve(tensors.size());
    for (const auto& t : tensors) {
        flat_parts.push_back(t.flatten().contiguous());
    }
    return tenzor::cat(flat_parts, 0);
}

auto flatten_tensors(const std::vector<std::shared_ptr<Variable>>& variables) -> Tensor {
    // Extract gradients from variables
    std::vector<Tensor> gradients;
    gradients.reserve(variables.size());

    for (const auto& var : variables) {
        if (var && var->grad()) {
            gradients.push_back(*var->grad());
        }
    }

    if (gradients.empty()) {
        throw std::invalid_argument("No Variables have gradients to flatten");
    }

    return flatten_tensors(gradients);
}

// ============================================================================
// Tensor Unflattening
// ============================================================================

auto unflatten_into(const Tensor& flat_tensor,
                   const std::vector<std::vector<int64_t>>& shapes) -> std::vector<Tensor> {
    if (flat_tensor.ndim() != 1) {
        throw std::invalid_argument("Flat tensor must be 1-dimensional");
    }

    // Calculate expected total elements
    size_t expected_elements = 0;
    for (const auto& shape : shapes) {
        size_t shape_elements = 1;
        for (auto dim : shape) {
            if (dim <= 0) {
                throw std::invalid_argument("All shape dimensions must be positive");
            }
            shape_elements *= dim;
        }
        expected_elements += shape_elements;
    }

    if (static_cast<size_t>(flat_tensor.numel()) != expected_elements) {
        throw std::invalid_argument(
            "Shape mismatch: flat tensor has " + std::to_string(flat_tensor.numel()) +
            " elements but shapes require " + std::to_string(expected_elements)
        );
    }

    // Slice the flat tensor and reshape — device-safe (no raw data_ptr() access)
    std::vector<Tensor> result;
    result.reserve(shapes.size());

    int64_t offset = 0;
    for (const auto& shape : shapes) {
        int64_t numel = 1;
        for (auto dim : shape) numel *= dim;

        auto slice = flat_tensor.slice(0, offset, offset + numel).contiguous();
        result.push_back(slice.reshape(shape));
        offset += numel;
    }

    return result;
}

auto unflatten_into(const Tensor& flat_tensor,
                   std::vector<Tensor>& output_tensors) -> void {
    if (flat_tensor.ndim() != 1) {
        throw std::invalid_argument("Flat tensor must be 1-dimensional");
    }

    // Validate total size matches
    size_t expected_elements = 0;
    for (const auto& t : output_tensors) {
        expected_elements += t.numel();
    }

    if (static_cast<size_t>(flat_tensor.numel()) != expected_elements) {
        throw std::invalid_argument(
            "Size mismatch: flat tensor has " + std::to_string(flat_tensor.numel()) +
            " elements but output tensors have " + std::to_string(expected_elements)
        );
    }

    // Validate dtypes match
    const auto dtype = flat_tensor.dtype();
    for (const auto& t : output_tensors) {
        if (t.dtype() != dtype) {
            throw std::invalid_argument("Output tensor dtype does not match flat tensor dtype");
        }
    }

    // Slice flat tensor and reshape into each output — device-safe
    int64_t offset = 0;
    for (auto& t : output_tensors) {
        int64_t numel = t.numel();
        auto slice = flat_tensor.slice(0, offset, offset + numel).contiguous();
        auto shape_span = t.shape();
        std::vector<int64_t> shape(shape_span.begin(), shape_span.end());
        t = slice.reshape(shape);
        offset += numel;
    }
}

// ============================================================================
// Bucket Computation
// ============================================================================

auto compute_bucket_sizes(const std::vector<Tensor>& tensors,
                         const BucketConfig& config) -> std::vector<BucketInfo> {
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

auto compute_bucket_sizes(const std::vector<std::shared_ptr<Variable>>& variables,
                         const BucketConfig& config) -> std::vector<BucketInfo> {
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

auto estimate_gradient_memory(const std::vector<Tensor>& tensors,
                              int world_size,
                              const BucketConfig& config) -> GradientMemoryStats {
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

auto estimate_gradient_memory(const std::vector<std::shared_ptr<Variable>>& variables,
                              int world_size,
                              const BucketConfig& config) -> GradientMemoryStats {
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

auto all_contiguous(const std::vector<Tensor>& tensors) -> bool {
    return std::all_of(tensors.begin(), tensors.end(),
                      [](const Tensor& t) { return t.is_contiguous(); });
}

auto same_dtype(const std::vector<Tensor>& tensors) -> bool {
    if (tensors.empty()) {
        return true;
    }

    const auto dtype = tensors[0].dtype();
    return std::all_of(tensors.begin() + 1, tensors.end(),
                      [dtype](const Tensor& t) { return t.dtype() == dtype; });
}

auto same_device(const std::vector<Tensor>& tensors) -> bool {
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

auto total_numel(const std::vector<Tensor>& tensors) -> size_t {
    size_t total = 0;
    for (const auto& t : tensors) {
        total += t.numel();
    }
    return total;
}

auto total_bytes(const std::vector<Tensor>& tensors) -> size_t {
    size_t total = 0;
    for (const auto& t : tensors) {
        total += t.numel() * dtype_size(t.dtype());
    }
    return total;
}

auto extract_shapes(const std::vector<Tensor>& tensors) -> std::vector<std::vector<int64_t>> {
    std::vector<std::vector<int64_t>> shapes;
    shapes.reserve(tensors.size());

    for (const auto& t : tensors) {
        auto shape_span = t.shape();
        shapes.emplace_back(shape_span.begin(), shape_span.end());
    }

    return shapes;
}

auto make_contiguous(const std::vector<Tensor>& tensors) -> std::vector<Tensor> {
    std::vector<Tensor> result;
    result.reserve(tensors.size());

    for (const auto& t : tensors) {
        result.push_back(t.is_contiguous() ? t : t.contiguous());
    }

    return result;
}

} // namespace optim
} // namespace tenzor
