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
auto ReshapeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
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
auto PermuteBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
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
auto TransposeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
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
auto RollBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
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
auto SqueezeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
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
    const auto& grad_output = grad_outputs[0];

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
    // Distribute gradients from upsampled output back to input size
    // For nearest neighbor upsampling: each output pixel's gradient goes to its source input pixel

    const auto& grad_output_orig = grad_outputs[0];
    const auto& shape = grad_output_orig.shape();

    if (shape.size() != 4) {
        throw std::runtime_error("UpsampleBilinearBackward: Expected 4D gradient tensor (N, C, H, W)");
    }

    // Remember original dtype and device for output conversion
    DType original_dtype = grad_output_orig.dtype();
    Device original_device = grad_output_orig.device();

    // Convert to Float32 on CPU for computation
    Tensor grad_output = grad_output_orig.to(Device::cpu());
    if (grad_output.dtype() != DType::Float32) {
        grad_output = grad_output.to(DType::Float32);
    }

    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t H_out = shape[2];
    int64_t W_out = shape[3];

    // Create gradient tensor for input (all zeros initially) in Float32
    auto grad_input = zeros({N, C, input_h_, input_w_}, DType::Float32, Device::cpu());

    // Calculate scaling factors (align_corners=false convention)
    float scale_h = static_cast<float>(input_h_) / static_cast<float>(output_h_);
    float scale_w = static_cast<float>(input_w_) / static_cast<float>(output_w_);

    // Distribute gradients using bilinear interpolation weights
    auto* grad_in_ptr = grad_input.data<float>();
    const auto* grad_out_ptr = grad_output.data<float>();

    for (int64_t n = 0; n < N; ++n) {
        for (int64_t c = 0; c < C; ++c) {
            for (int64_t h = 0; h < H_out; ++h) {
                for (int64_t w = 0; w < W_out; ++w) {
                    // Map output pixel to input coordinate (align_corners=false)
                    float src_h = (h + 0.5f) * scale_h - 0.5f;
                    float src_w = (w + 0.5f) * scale_w - 0.5f;

                    // Bounding input pixels
                    int64_t h0 = static_cast<int64_t>(std::floor(src_h));
                    int64_t w0 = static_cast<int64_t>(std::floor(src_w));
                    int64_t h1 = h0 + 1;
                    int64_t w1 = w0 + 1;

                    // Interpolation weights from fractional part
                    float fh = src_h - h0;
                    float fw = src_w - w0;

                    float grad_val = grad_out_ptr[((n * C + c) * H_out + h) * W_out + w];
                    int64_t base = (n * C + c) * input_h_;

                    // Accumulate weighted gradient to each of the 4 neighbors
                    if (h0 >= 0 && h0 < input_h_ && w0 >= 0 && w0 < input_w_)
                        grad_in_ptr[(base + h0) * input_w_ + w0] += grad_val * (1.0f - fh) * (1.0f - fw);
                    if (h0 >= 0 && h0 < input_h_ && w1 >= 0 && w1 < input_w_)
                        grad_in_ptr[(base + h0) * input_w_ + w1] += grad_val * (1.0f - fh) * fw;
                    if (h1 >= 0 && h1 < input_h_ && w0 >= 0 && w0 < input_w_)
                        grad_in_ptr[(base + h1) * input_w_ + w0] += grad_val * fh * (1.0f - fw);
                    if (h1 >= 0 && h1 < input_h_ && w1 >= 0 && w1 < input_w_)
                        grad_in_ptr[(base + h1) * input_w_ + w1] += grad_val * fh * fw;
                }
            }
        }
    }

    // Convert back to original dtype and device
    if (grad_input.dtype() != original_dtype) {
        grad_input = grad_input.to(original_dtype);
    }
    if (grad_input.device() != original_device) {
        grad_input = grad_input.to(original_device);
    }

    return {grad_input};
}

auto UpsampleBilinearBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Upsample backward is a linear operation (weighted accumulation), so its
    // second derivative is constant. Compute at Tensor level since the bilinear
    // weights don't depend on the input values.
    auto result = backward({grad_outputs[0].tensor()});
    return {Variable(result[0], grad_outputs[0].requires_grad())};
}

} // namespace tenzor
