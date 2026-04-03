#include "tenzor/autograd/function.hpp"
#include "function_helpers.hpp"
#include <cassert>
#include "tenzor/autograd/ops.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/error.hpp"
#include "tenzor/utils/safe_math.hpp"
#include <cmath>
#include <iostream>
#include <string>
#include <typeinfo>
#include <unordered_set>
#ifdef __GNUC__
#include <cxxabi.h>
#include <cstdlib>
#endif

namespace tenzor {

// =========================================================================
// Shape/Indexing Backward Functions
// =========================================================================

// UnsqueezeBackward implementation
// Saves dim. backward: squeeze(grad, dim)
auto UnsqueezeBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("UnsqueezeBackward::forward should not be called");
}

auto UnsqueezeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    return {squeeze(grad, dim)};
}

auto UnsqueezeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    return {tenzor::squeeze(grad_outputs[0], dim)};
}

// ExpandBackward implementation
// Saves original shape. backward: sum_to(grad, original_shape)
auto ExpandBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ExpandBackward::forward should not be called");
}

auto ExpandBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // Original shape is saved in saved_tensors_[0] as a 1D Int64 tensor
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {reduce_grad_for_broadcasting(grad, original_shape)};
}

auto ExpandBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {reduce_grad_var_for_broadcasting(grad_outputs[0], original_shape)};
}

// DeviceTransferBackward implementation
// Transfers gradient back to the source device during backward
auto DeviceTransferBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("DeviceTransferBackward::forward should not be called");
}

auto DeviceTransferBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    if (grad.device() == source_device) {
        return {grad};
    }
    return {grad.to(source_device)};
}

// FlattenBackward implementation
// Saves original shape. backward: reshape(grad, original_shape)
auto FlattenBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FlattenBackward::forward should not be called");
}

auto FlattenBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // Original shape is saved in saved_tensors_[0] as a 1D Int64 tensor
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {reshape(grad, original_shape)};
}

auto FlattenBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    return {tenzor::reshape(grad_outputs[0], original_shape)};
}

// WhereBackward implementation
// Saves condition. backward: grad_x = grad * condition, grad_y = grad * !condition
auto WhereBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("WhereBackward::forward should not be called");
}

auto WhereBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& condition = saved_tensors_[0];  // Bool tensor

    auto shape_vec = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    auto zeros_tensor = zeros(shape_vec, grad.dtype(), grad.device());

    // grad_x = where(condition, grad, 0)
    auto grad_x = where(condition, grad, zeros_tensor);

    // grad_y = where(condition, 0, grad) = where(!condition, grad, 0)
    auto grad_y = where(condition, zeros_tensor, grad);

    return {grad_x, grad_y};
}

auto WhereBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // where backward: grad_x = where(condition, grad, 0), grad_y = where(condition, 0, grad)
    // condition is non-differentiable
    const auto& condition = saved_tensors_[0];
    auto shape_vec = std::vector<int64_t>(grad_outputs[0].shape().begin(), grad_outputs[0].shape().end());
    auto zeros_tensor = zeros(shape_vec, grad_outputs[0].tensor().dtype(), grad_outputs[0].tensor().device());

    Variable cond_var(condition, false);
    Variable zeros_var(zeros_tensor, false);

    auto grad_x = tenzor::where(cond_var, grad_outputs[0], zeros_var);
    auto grad_y = tenzor::where(cond_var, zeros_var, grad_outputs[0]);

    return {grad_x, grad_y};
}

// GatherBackward implementation
// Saves dim, index, input_shape. backward: scatter_add(zeros(input_shape), dim, index, grad)
auto GatherBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("GatherBackward::forward should not be called");
}

auto GatherBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = index
    // saved_tensors_[2] = input shape (1D Int64)
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    const auto& index = saved_tensors_[1];
    const auto& shape_tensor = saved_tensors_[2];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto input_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // scatter_add zeros with grad at index positions
    auto grad_input = zeros(input_shape, grad.dtype(), grad.device());
    grad_input = scatter_add(grad_input, dim, index, grad);

    return {grad_input};
}

auto GatherBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// ScatterBackward implementation
// Saves dim, index. backward: grad_input = scatter(grad, dim, index, zeros), grad_src = gather(grad, dim, index)
auto ScatterBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ScatterBackward::forward should not be called");
}

auto ScatterBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = index
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    const auto& index = saved_tensors_[1];

    // grad_input: zero out the scattered positions
    auto index_shape_vec = std::vector<int64_t>(index.shape().begin(), index.shape().end());
    auto zeros_src = zeros(index_shape_vec, grad.dtype(), grad.device());
    auto grad_input = scatter(grad, dim, index, zeros_src);

    // grad_src: gather from grad at the index positions
    auto grad_src = gather(grad, dim, index);

    return {grad_input, grad_src};
}

auto ScatterBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad()),
            Variable(result[1], grad_outputs[0].requires_grad())};
}

// IndexSelectBackward implementation
// Saves dim, index, input_shape. backward: create zeros, scatter_add grad at index positions
auto IndexSelectBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IndexSelectBackward::forward should not be called");
}

auto IndexSelectBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = index (1D Int64)
    // saved_tensors_[2] = input shape (1D Int64)
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    const auto& index = saved_tensors_[1];
    const auto& shape_tensor = saved_tensors_[2];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto input_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zeros of input shape
    auto grad_input = zeros(input_shape, grad.dtype(), grad.device());

    // Build a full index tensor matching grad shape for scatter_add along dim
    auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    auto full_index = zeros(grad_shape, DType::Int64, Device::cpu());
    auto* idx_ptr = full_index.data<int64_t>();
    auto* src_idx_ptr = index.to(Device::cpu()).data<int64_t>();

    int64_t total = grad.numel();
    int64_t dim_size = grad_shape[dim];
    int64_t dim_stride = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(grad_shape.size()); ++d) {
        dim_stride *= grad_shape[d];
    }

    for (int64_t i = 0; i < total; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % dim_size;
        idx_ptr[i] = src_idx_ptr[pos_in_dim];
    }

    if (grad.device() != Device::cpu()) {
        full_index = full_index.to(grad.device());
    }

    grad_input = scatter_add(grad_input, dim, full_index, grad);

    return {grad_input};
}

auto IndexSelectBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// IndexBackward implementation
// Backward of advanced (fancy) indexing: scatter_add grad into zeros of input shape
auto IndexBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("IndexBackward::forward should not be called");
}

auto IndexBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = input shape (1D Int64)
    // saved_tensors_[1..num_indices_] = index tensors (0-element = nullopt)
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto input_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zeros of input shape
    auto grad_input = zeros(input_shape, grad.dtype(), grad.device());

    // Reconstruct optional indices
    std::vector<std::optional<Tensor>> indices;
    indices.reserve(num_indices_);
    for (int64_t i = 0; i < num_indices_; ++i) {
        const auto& idx = saved_tensors_[1 + i];
        if (idx.numel() > 0) {
            indices.push_back(idx);
        } else {
            indices.push_back(std::nullopt);
        }
    }

    // index_put with accumulation: grad_input[indices] += grad
    index_put(grad_input, indices, grad);

    return {grad_input};
}

auto IndexBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// NarrowBackward implementation
// Saves dim, start, original_shape. backward: zero-pad grad to original shape
auto NarrowBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("NarrowBackward::forward should not be called");
}

auto NarrowBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = dim (scalar Int64)
    // saved_tensors_[1] = start (scalar Int64)
    // saved_tensors_[2] = original shape (1D Int64)
    int64_t dim = saved_tensors_[0].data<int64_t>()[0];
    int64_t start = saved_tensors_[1].data<int64_t>()[0];
    const auto& shape_tensor = saved_tensors_[2];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    // Create zero tensor of original shape
    auto grad_input = zeros(original_shape, grad.dtype(), grad.device());

    // Use scatter to place grad values at the correct positions
    // Build index tensor
    auto grad_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    int64_t narrow_len = grad_shape[dim];
    auto index = zeros(grad_shape, DType::Int64, Device::cpu());
    auto* idx_ptr = index.data<int64_t>();
    int64_t total = grad.numel();
    int64_t dim_stride = 1;
    for (int64_t d = dim + 1; d < static_cast<int64_t>(grad_shape.size()); ++d) {
        dim_stride *= grad_shape[d];
    }

    for (int64_t i = 0; i < total; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % narrow_len;
        idx_ptr[i] = start + pos_in_dim;
    }

    if (grad.device() != Device::cpu()) {
        index = index.to(grad.device());
    }

    grad_input = scatter(grad_input, dim, index, grad);

    return {grad_input};
}

auto NarrowBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

// FlipBackward implementation
// Saves dims. backward: flip(grad, dims)
auto FlipBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("FlipBackward::forward should not be called");
}

auto FlipBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    // saved_tensors_[0] holds the dims as a 1D Int64 tensor
    const auto& dims_tensor = saved_tensors_[0];
    auto dims_ptr = dims_tensor.data<int64_t>();
    auto dims = std::vector<int64_t>(dims_ptr, dims_ptr + dims_tensor.numel());
    return {flip(grad, dims)};
}

auto FlipBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // flip backward: flip(grad, dims)
    const auto& dims_tensor = saved_tensors_[0];
    auto dims_ptr = dims_tensor.data<int64_t>();
    auto dims = std::vector<int64_t>(dims_ptr, dims_ptr + dims_tensor.numel());
    return {tenzor::flip(grad_outputs[0], dims)};
}

// RepeatBackward implementation
// Saves original_shape and repeats. backward: sum grad over repeated dimensions
auto RepeatBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("RepeatBackward::forward should not be called");
}

auto RepeatBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];

    // saved_tensors_[0] = original shape (1D Int64)
    // saved_tensors_[1] = repeats (1D Int64)
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    const auto& repeats_tensor = saved_tensors_[1];
    auto repeats_ptr = repeats_tensor.data<int64_t>();
    auto repeats = std::vector<int64_t>(repeats_ptr, repeats_ptr + repeats_tensor.numel());

    // To compute gradient: reshape grad so each repeated dimension is split into
    // (repeat_count, original_dim_size), then sum over the repeat_count dimension.
    //
    // For each dimension i:
    //   grad_shape[i] = repeats[i] * original_shape[i]
    // We reshape to interleave repeat and original dims, then sum.

    auto ndim = original_shape.size();

    // Build reshape: [repeats[0], orig[0], repeats[1], orig[1], ...]
    std::vector<int64_t> expanded_shape;
    expanded_shape.reserve(2 * ndim);
    for (size_t i = 0; i < ndim; ++i) {
        expanded_shape.push_back(repeats[i]);
        expanded_shape.push_back(original_shape[i]);
    }

    auto grad_reshaped = reshape(grad, expanded_shape);

    // Sum over the repeat dimensions (dims 0, 2, 4, ...)
    // We need to sum from the highest dim first to avoid shifting indices
    auto result = grad_reshaped;
    for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
        int64_t repeat_dim = 2 * i;  // The repeat count dimension
        result = tenzor::sum(result, repeat_dim, false);
    }

    return {result};
}

auto RepeatBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // repeat backward: reshape grad to split repeated dims, sum over repeat dims
    const auto& shape_tensor = saved_tensors_[0];
    auto shape_ptr = shape_tensor.data<int64_t>();
    auto original_shape = std::vector<int64_t>(shape_ptr, shape_ptr + shape_tensor.numel());

    const auto& repeats_tensor = saved_tensors_[1];
    auto repeats_ptr = repeats_tensor.data<int64_t>();
    auto repeats = std::vector<int64_t>(repeats_ptr, repeats_ptr + repeats_tensor.numel());

    auto ndim = original_shape.size();

    // Build reshape: [repeats[0], orig[0], repeats[1], orig[1], ...]
    std::vector<int64_t> expanded_shape;
    expanded_shape.reserve(2 * ndim);
    for (size_t i = 0; i < ndim; ++i) {
        expanded_shape.push_back(repeats[i]);
        expanded_shape.push_back(original_shape[i]);
    }

    auto grad_reshaped = tenzor::reshape(grad_outputs[0], expanded_shape);

    // Sum over repeat dimensions (0, 2, 4, ...) from highest to avoid index shift
    Variable result = grad_reshaped;
    for (int64_t i = static_cast<int64_t>(ndim) - 1; i >= 0; --i) {
        int64_t repeat_dim = 2 * i;
        result = tenzor::sum(result, repeat_dim, false);
    }

    return {result};
}

} // namespace tenzor
