#include "tenzor/autograd/function.hpp"
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

// ReshapeBackward implementation
auto ReshapeBackward::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("ReshapeBackward::forward should not be called");
}

auto ReshapeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Reshape gradient back to input shape and ensure contiguity
    // Reshape may create non-contiguous views, which can cause issues in element-wise operations
    auto grad_input = reshape(grad_outputs[0], input_shape_).contiguous();
    return {grad_input};
}

auto ReshapeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {reshape(grad_outputs[0], input_shape_)};
}

// PermuteBackward implementation
auto PermuteBackward::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("PermuteBackward::forward should not be called");
}

auto PermuteBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Apply inverse permutation to gradient and ensure contiguity
    // Permute creates non-contiguous views, which can cause issues in element-wise operations
    auto grad_input = permute(grad_outputs[0], inv_dims_).contiguous();
    return {grad_input};
}

auto PermuteBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {permute(grad_outputs[0], inv_dims_)};
}

// TransposeBackward implementation
auto TransposeBackward::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("TransposeBackward::forward should not be called");
}

auto TransposeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Transpose is its own inverse, so apply same transpose to gradient
    auto grad_input = transpose(grad_outputs[0], dim0_, dim1_).contiguous();
    return {grad_input};
}

auto TransposeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {transpose(grad_outputs[0], dim0_, dim1_)};
}

// RollBackward implementation
auto RollBackward::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("RollBackward::forward should not be called");
}

auto RollBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Roll backward is roll with negative shift
    auto grad_input = roll(grad_outputs[0], -shifts_, dim_);
    return {grad_input};
}

auto RollBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    return {roll(grad_outputs[0], -shifts_, dim_)};
}

// SqueezeBackward implementation
auto SqueezeBackward::forward([[maybe_unused]] std::vector<Variable> inputs) -> std::vector<Variable> {
    throw std::runtime_error("SqueezeBackward::forward should not be called");
}

auto SqueezeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Unsqueeze gradient back to original shape
    auto grad_input = unsqueeze(grad_outputs[0], dim_);
    return {grad_input};
}

auto SqueezeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Use Variable-level reshape to unsqueeze back to original shape
    // This preserves the computation graph for higher-order gradients
    auto grad = grad_outputs[0];
    auto target_shape = std::vector<int64_t>(grad.shape().begin(), grad.shape().end());
    int64_t ndim_output = static_cast<int64_t>(target_shape.size()) + 1;
    int64_t dim = dim_ < 0 ? dim_ + ndim_output : dim_;
    target_shape.insert(target_shape.begin() + dim, 1);
    return {reshape(grad, target_shape)};
}

// BmmBackward implementation
auto BmmBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    save_for_backward({inputs[0].tensor(), inputs[1].tensor()});
    auto result = bmm(inputs[0].tensor(), inputs[1].tensor());
    return {Variable(result, true)};
}

auto BmmBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // For C = bmm(A, B):
    // A: (batch, n, m), B: (batch, m, p), C: (batch, n, p)
    // grad_output: (batch, n, p)
    //
    // Backward gradients:
    // grad_a = grad_output @ B^T = (batch, n, p) @ (batch, p, m) = (batch, n, m)
    // grad_b = A^T @ grad_output = (batch, m, n) @ (batch, n, p) = (batch, m, p)

    const auto& a = saved_tensors_[0];
    const auto& b = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    // Transpose last two dimensions: (batch, m, p) -> (batch, p, m)
    auto b_transposed = permute(b, {0, 2, 1});

    // grad_a = grad_output @ b^T
    auto grad_a = bmm(grad_output, b_transposed);

    // Transpose a: (batch, n, m) -> (batch, m, n)
    auto a_transposed = permute(a, {0, 2, 1});

    // grad_b = a^T @ grad_output
    auto grad_b = bmm(a_transposed, grad_output);

    return {grad_a, grad_b};
}

auto BmmBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // For C = bmm(A, B):
    // grad_a = grad_output @ B^T
    // grad_b = A^T @ grad_output
    Variable saved_a, saved_b;
    if (has_saved_variables()) {
        saved_a = saved_variables_[0];
        saved_b = saved_variables_[1];
    } else {
        saved_a = Variable(saved_tensors_[0], false);
        saved_b = Variable(saved_tensors_[1], false);
    }
    const auto& grad_out = grad_outputs[0];
    auto b_t = tenzor::transpose(saved_b, saved_b.shape().size() - 2, saved_b.shape().size() - 1);
    auto a_t = tenzor::transpose(saved_a, saved_a.shape().size() - 2, saved_a.shape().size() - 1);
    auto grad_a = tenzor::bmm(grad_out, b_t);
    auto grad_b = tenzor::bmm(a_t, grad_out);
    return {grad_a, grad_b};
}

// CatBackward implementation
auto CatBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Convert Variables to Tensors for concatenation
    std::vector<Tensor> tensors;
    tensors.reserve(inputs.size());
    for (const auto& var : inputs) {
        tensors.push_back(var.tensor());
    }

    auto result = cat(tensors, dim_);
    return {Variable(result, true)};
}

auto CatBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Split gradient back along concatenation dimension
    // grad_output shape: [..., sum(split_sizes), ...]
    // Need to split into gradients of shape [..., split_sizes[i], ...]

    const auto& grad_output = grad_outputs[0];
    std::vector<Tensor> grad_inputs;
    grad_inputs.reserve(split_sizes_.size());

    int64_t offset = 0;
    for (int64_t split_size : split_sizes_) {
        // Slice grad_output from offset to offset+split_size along dim_
        auto grad_slice = slice(grad_output, dim_, offset, offset + split_size);
        grad_inputs.push_back(grad_slice);
        offset += split_size;
    }

    return grad_inputs;
}

auto CatBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    const auto& grad_output = grad_outputs[0];
    std::vector<Variable> grad_inputs;
    grad_inputs.reserve(split_sizes_.size());
    int64_t offset = 0;
    for (int64_t split_size : split_sizes_) {
        // Use Variable-level slice to preserve computation graph for higher-order gradients
        grad_inputs.push_back(slice(grad_output, dim_, offset, offset + split_size));
        offset += split_size;
    }
    return grad_inputs;
}

// SliceBackward implementation
auto SliceBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = slice(inputs[0].tensor(), dim_, start_, end_, step_);
    return {Variable(result, true)};
}

auto SliceBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad_output_raw = grad_outputs[0];

    // grad_output here often arrives as a non-contiguous strided view —
    // typically from CatBackward, which returns slice views of its grad to
    // distribute back to each cat input. The CUDA / ROCm / OneAPI / Vulkan
    // scatter kernels read `src` with raw pointer arithmetic and silently
    // ignore strides, so a non-contig src lands the wrong element values at
    // each scatter index. The defect surfaces only in chained slice+cat
    // graphs (e.g. CircularPad2d/3d) where grad_output is a slice along a
    // non-last dim; 1D / single-axis paths are accidentally fine because
    // the view's stride matches contiguous layout there. Materialise a
    // contiguous copy up front so every backend's scatter sees stride-1
    // src memory. (See feedback_stride_bugs.md for the family.)
    auto grad_output = grad_output_raw.is_contiguous() ? grad_output_raw
                                                       : grad_output_raw.contiguous();

    // Create zero gradient tensor with original input shape
    auto grad_input = zeros(input_shape_, grad_output.dtype(), grad_output.device());

    // Build index tensor for scatter operation
    // Index tensor must have same shape as grad_output
    int64_t slice_size = grad_output.shape()[dim_];
    int64_t total_elements = grad_output.numel();

    // Create index tensor with same shape as grad_output
    auto index_shape = std::vector<int64_t>(grad_output.shape().begin(), grad_output.shape().end());
    auto index = zeros(index_shape, DType::Int64, Device::cpu());

    // Fill index tensor on CPU
    int64_t* index_ptr = index.data<int64_t>();

    // Calculate stride for the sliced dimension
    int64_t dim_stride = 1;
    for (int64_t d = dim_ + 1; d < grad_output.ndim(); ++d) {
        dim_stride *= grad_output.shape()[d];
    }

    // Fill index tensor: each element along dim_ gets mapped to (start_ + pos * step_)
    for (int64_t i = 0; i < total_elements; ++i) {
        int64_t pos_in_dim = (i / dim_stride) % slice_size;
        index_ptr[i] = start_ + pos_in_dim * step_;
    }

    // Transfer to target device if needed
    if (grad_output.device() != Device::cpu()) {
        index = index.to(grad_output.device());
    }

    // Use scatter to place gradients - dispatches to appropriate backend
    grad_input = scatter(grad_input, dim_, index, grad_output);

    return {grad_input};
}

auto SliceBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Same as tensor-level backward but wrapping with requires_grad=true
    auto result_tensors = backward({grad_outputs[0].tensor()});
    return {Variable(result_tensors[0], grad_outputs[0].requires_grad())};
}

// UpsampleBilinearBackward implementation
auto UpsampleBilinearBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    // Save input tensor for backward pass
    save_for_backward({inputs[0].tensor()});

    // Forward computation is done externally in the wrapper function
    // This method is not typically called directly
    throw std::runtime_error("UpsampleBilinearBackward::forward should not be called directly");
}

auto UpsampleBilinearBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Audit D3: device-resident bilinear scatter via OpId::InterpolateBackward.
    // The previous body did a `.to(cpu)` / scalar loop / `.to(device)` round-trip
    // (a CPU fallback on GPU backends); the new path dispatches through the
    // registered backend kernel so the math stays on the original device.
    const auto& grad_output = grad_outputs[0];
    if (grad_output.shape().size() != 4) {
        throw std::runtime_error(
            "UpsampleBilinearBackward: expected 4D gradient tensor (N, C, H, W)");
    }
    OpAttributes attrs;
    attrs.set(AttrKey::InputShape,
              std::to_string(input_h_) + "," + std::to_string(input_w_));
    attrs.set(AttrKey::Mode, "bilinear");
    attrs.set(AttrKey::AlignCorners, false);

    std::vector<Tensor> dispatch_inputs = {grad_output};
    auto results = tenzor::dispatch(OpId::InterpolateBackward, dispatch_inputs, attrs);
    return {results[0]};
}

auto UpsampleBilinearBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Audit D3: real Variable-level backward + true higher-order.
    //
    // The bilinear-backward op is a linear scatter A^T (where A is the
    // forward bilinear upsample). Its adjoint at the Variable level is
    // therefore A itself — bilinear upsample of the next-level gradient.
    //
    // Strategy: compute grad_input via the tensor-level dispatch (no CPU
    // fallback, see backward()), then attach a fresh `UpsampleBilinearAdjoint`
    // grad_fn whose backward dispatches `OpId::Interpolate` on the next-level
    // gradient. That re-enters the autograd machinery and produces a real
    // 2nd-order graph.
    const Variable& grad_out = grad_outputs[0];
    Tensor grad_input_t = backward({grad_out.tensor()})[0];

    Variable grad_input(grad_input_t, grad_out.requires_grad());
    if (grad_out.requires_grad() && is_grad_enabled()) {
        // Build the adjoint Function: its `backward` applies bilinear upsample
        // to the incoming next-level grad (shape input_h × input_w) to produce
        // a tensor of shape output_h × output_w — completing the chain.
        auto adjoint = std::make_shared<UpsampleBilinearForwardAdjoint>(
            input_h_, input_w_, output_h_, output_w_);
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(grad_out.grad_fn());
        adjoint->set_next_functions(std::move(next_funcs));
        std::vector<Variable> input_vars;
        input_vars.push_back(grad_out);
        adjoint->set_input_variables(std::move(input_vars));
        grad_input.set_grad_fn(adjoint);
    }
    return {grad_input};
}

// ============================================================================
// UpsampleBilinearForwardAdjoint (audit D3)
// ============================================================================
//
// Used by `UpsampleBilinearBackward::backward_with_variables` to express the
// 2nd-order chain: forward is *no-op* (the engine never invokes this
// directly), backward applies bilinear upsample (`OpId::Interpolate`) to
// the next-level gradient — turning a (input_h × input_w)-shaped grad into
// a (output_h × output_w)-shaped grad, matching the original forward op.

auto UpsampleBilinearForwardAdjoint::forward(std::vector<Variable> /*inputs*/)
    -> std::vector<Variable> {
    throw std::runtime_error(
        "UpsampleBilinearForwardAdjoint::forward should not be called directly");
}

auto UpsampleBilinearForwardAdjoint::backward(std::vector<Tensor> grad_outputs)
    -> std::vector<Tensor> {
    const Tensor& g = grad_outputs[0];
    OpAttributes attrs;
    attrs.set(AttrKey::OutputSize,
              std::to_string(output_h_) + "," + std::to_string(output_w_));
    attrs.set(AttrKey::Mode, "bilinear");
    attrs.set(AttrKey::AlignCorners, false);
    std::vector<Tensor> dispatch_inputs = {g};
    auto results = tenzor::dispatch(OpId::Interpolate, dispatch_inputs, attrs);
    return {results[0]};
}

auto UpsampleBilinearForwardAdjoint::backward_with_variables(
    std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Recursive higher-order: re-enter UpsampleBilinearBackward to provide
    // the backward-of-the-forward. This pairs symmetrically with
    // UpsampleBilinearBackward::backward_with_variables above.
    const Variable& g = grad_outputs[0];
    auto fwd_result = backward({g.tensor()});  // bilinear upsample, tensor-level
    Variable out(fwd_result[0], g.requires_grad());
    if (g.requires_grad() && is_grad_enabled()) {
        auto bwd = std::make_shared<UpsampleBilinearBackward>(
            input_h_, input_w_, output_h_, output_w_);
        std::vector<std::shared_ptr<Function>> next_funcs;
        next_funcs.push_back(g.grad_fn());
        bwd->set_next_functions(std::move(next_funcs));
        std::vector<Variable> input_vars;
        input_vars.push_back(g);
        bwd->set_input_variables(std::move(input_vars));
        out.set_grad_fn(bwd);
    }
    return {out};
}

// ============================================================================
// ViewAsReal / ViewAsComplex backward
// ============================================================================

auto ViewAsRealBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = tenzor::view_as_real(inputs[0].tensor());
    return {Variable(result, inputs[0].requires_grad())};
}

auto ViewAsRealBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {tenzor::view_as_complex(grad_outputs[0])};
}

auto ViewAsComplexBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = tenzor::view_as_complex(inputs[0].tensor());
    return {Variable(result, inputs[0].requires_grad())};
}

auto ViewAsComplexBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    return {tenzor::view_as_real(grad_outputs[0])};
}

} // namespace tenzor
