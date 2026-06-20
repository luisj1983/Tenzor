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
            if (dim < 0) {
                throw std::invalid_argument("Shape dimensions must be non-negative");
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

    // Slice flat tensor and reshape into each output — device-safe.
    //
    // NOTE ON SEMANTICS: each element of `output_tensors` is *rebound* to a
    // freshly-allocated contiguous slice of `flat_tensor`; the data is NOT
    // written into the caller's pre-existing storage. The library currently
    // exposes no device-safe primitive to copy one tensor's data into another
    // tensor's existing storage (there is no Tensor::copy_ / OpId::Copy
    // in-place dispatch), and raw data_ptr() memcpy would be incorrect for
    // strided/device tensors. Consequently the caller MUST propagate the
    // rebound elements back to their owning objects after this call (e.g.
    // `param->tensor() = output_tensors[i];` or `param->set_grad(...)`), as the
    // two ZeRO callers in zero_optimizer.cpp do. Any external alias of the
    // original storage retains stale data otherwise.
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

} // namespace optim
} // namespace tenzor
