/**
 * @file gradient_compression.cpp
 * @brief Implementation of gradient compression strategies
 *
 * Implements FP16 and TopK gradient compression for bandwidth-efficient
 * distributed training. These compressors integrate into the DDP all-reduce
 * path to reduce communication volume.
 */

#include "tenzor/distributed/gradient_compression.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/backend.hpp"
#include "tenzor/backend/loader.hpp"
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace tenzor::distributed {

// ============================================================================
// FP16Compressor
// ============================================================================

auto FP16Compressor::compress(Tensor& gradient) -> CompressedGradient {
    CompressedGradient result;

    // Store original metadata for decompression
    result.original_shape = std::vector<int64_t>(
        gradient.shape().begin(), gradient.shape().end());
    result.original_dtype = gradient.dtype();
    result.original_device = gradient.device();
    result.original_numel = gradient.numel();

    // If gradient is already Float16 or BFloat16, no conversion needed
    if (gradient.dtype() == DType::Float16 ||
        gradient.dtype() == DType::BFloat16) {
        result.data = gradient;
        result.compression_ratio = 1.0f;
        return result;
    }

    // Cast to Float16 for 2x bandwidth reduction
    result.data = gradient.to(DType::Float16);

    // Compute compression ratio: Float16 (2 bytes) vs original dtype size
    size_t original_elem_size = dtype_size(gradient.dtype());
    size_t compressed_elem_size = dtype_size(DType::Float16);
    result.compression_ratio = static_cast<float>(compressed_elem_size) /
                               static_cast<float>(original_elem_size);

    return result;
}

auto FP16Compressor::decompress(CompressedGradient& compressed) -> Tensor {
    // If the original dtype was already Float16/BFloat16, return as-is
    if (compressed.original_dtype == DType::Float16 ||
        compressed.original_dtype == DType::BFloat16) {
        return compressed.data;
    }

    // Cast back to the original dtype
    return compressed.data.to(compressed.original_dtype);
}

// ============================================================================
// TopKCompressor
// ============================================================================

TopKCompressor::TopKCompressor(float ratio)
    : ratio_(ratio) {
    if (ratio <= 0.0f || ratio > 1.0f) {
        throw std::invalid_argument(
            "TopKCompressor ratio must be in (0.0, 1.0], got " +
            std::to_string(ratio));
    }
}

auto TopKCompressor::compress(Tensor& gradient) -> CompressedGradient {
    CompressedGradient result;

    // Store original metadata
    result.original_shape = std::vector<int64_t>(
        gradient.shape().begin(), gradient.shape().end());
    result.original_dtype = gradient.dtype();
    result.original_device = gradient.device();
    result.original_numel = gradient.numel();

    // Flatten the gradient for uniform processing
    Tensor flat = gradient.flatten();
    int64_t numel = flat.numel();

    // Apply error feedback: add residual from previous iteration.
    // The residual accumulates zeroed-out values so no gradient
    // information is permanently lost across iterations.
    const void* grad_key = gradient.data_ptr();
    auto residual_it = residuals_.find(grad_key);
    if (residual_it != residuals_.end()) {
        Tensor& residual = residual_it->second;
        // Verify the residual shape matches (parameter hasn't changed)
        if (residual.numel() == numel) {
            flat = tenzor::add(flat, residual);
        } else {
            // Shape mismatch: discard stale residual
            residuals_.erase(residual_it);
        }
    }

    // Compute K: number of elements to keep
    int64_t k = std::max(static_cast<int64_t>(1),
                         static_cast<int64_t>(std::ceil(numel * ratio_)));

    // Sort absolute values descending to determine which elements to keep.
    // The sort gives us indices that map from sorted positions back to
    // original positions: sort_indices[i] = original position of i-th
    // largest absolute value.
    Tensor abs_flat = tenzor::abs(flat);
    auto [sorted_abs, sort_indices] = tenzor::sort(abs_flat, /*dim=*/0,
                                                    /*descending=*/true);

    // Build a binary indicator in sorted space: 1.0 for the first k
    // positions (top-K), 0.0 for the rest.
    Tensor indicator;
    if (k < numel) {
        Tensor ones_k = tenzor::ones({k}, flat.dtype(), flat.device());
        Tensor zeros_rest = tenzor::zeros({numel - k}, flat.dtype(),
                                           flat.device());
        indicator = tenzor::cat({ones_k, zeros_rest}, /*dim=*/0);
    } else {
        // Keep everything (k >= numel)
        indicator = tenzor::ones({numel}, flat.dtype(), flat.device());
    }

    // "Unsort" the indicator back to original positions to get the mask.
    // argsort(sort_indices) produces the inverse permutation, and then
    // index_select(indicator, 0, inverse_perm) rearranges indicator so
    // that mask[j] = 1.0 iff position j was among the top-K.
    Tensor inverse_perm = tenzor::argsort(sort_indices, /*dim=*/0,
                                           /*descending=*/false);
    Tensor mask = tenzor::index_select(indicator, /*dim=*/0, inverse_perm);

    // Apply mask: keep top-K values, zero the rest
    Tensor compressed = tenzor::mul(flat, mask);

    // Compute and store residual (zeroed-out values) for error feedback
    Tensor residual = tenzor::sub(flat, compressed);
    residuals_[grad_key] = residual;

    // Reshape compressed back to original shape
    result.data = compressed.reshape(result.original_shape);
    result.compression_ratio = static_cast<float>(k) /
                               static_cast<float>(numel);

    // Update the input gradient in-place to the compressed version.
    //
    // Important: we must NOT reassign `gradient` to a freshly-allocated
    // tensor here. The error-feedback residual is keyed on
    // `gradient.data_ptr()`; if compress() swaps the caller's buffer out
    // for a new allocation, the next call sees a different pointer and the
    // residual cache misses. Instead, copy the compressed values into the
    // caller's existing storage so data_ptr() stays stable across calls.
    {
        Tensor compressed_reshaped = compressed.reshape(result.original_shape);
        if (gradient.is_contiguous() &&
            gradient.dtype() == compressed_reshaped.dtype() &&
            gradient.numel() == compressed_reshaped.numel() &&
            gradient.device() == compressed_reshaped.device()) {
            // In-place memcpy via the backend's copy primitive.
            auto* backend = backend_registry().get_backend(gradient.device().type);
            if (backend) {
                CopyKind kind = (gradient.device().type == Device::Type::CPU)
                                    ? CopyKind::HostToHost
                                    : CopyKind::DeviceToDevice;
                backend->copy(
                    gradient.data_ptr(),
                    compressed_reshaped.data_ptr(),
                    gradient.numel() * gradient.dtype_size(),
                    kind);
            } else {
                // No backend — fall back to assignment (loses residual
                // tracking, but at least we don't crash).
                gradient = compressed_reshaped;
            }
        } else {
            // Shape/dtype/device mismatch — caller will see the new buffer
            // and the residual cache is effectively reset, which is the
            // best we can do here.
            gradient = compressed_reshaped;
        }
    }
    result.data = gradient;

    return result;
}

auto TopKCompressor::decompress(CompressedGradient& compressed) -> Tensor {
    // TopK compression produces a full-size tensor with zeros in place of
    // pruned values. No decompression is needed -- just return the data,
    // reshaping to the original shape if necessary.
    Tensor result = compressed.data;

    // Compare shapes element-by-element since span != span is not defined
    auto current_shape = result.shape();
    const auto& target_shape = compressed.original_shape;
    bool shapes_match = (static_cast<size_t>(current_shape.size()) ==
                         target_shape.size());
    if (shapes_match) {
        for (size_t i = 0; i < target_shape.size(); ++i) {
            if (current_shape[i] != target_shape[i]) {
                shapes_match = false;
                break;
            }
        }
    }

    if (!shapes_match) {
        result = result.reshape(compressed.original_shape);
    }

    return result;
}

auto TopKCompressor::reset() -> void {
    residuals_.clear();
}

} // namespace tenzor::distributed
