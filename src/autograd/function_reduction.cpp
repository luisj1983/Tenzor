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

// MaxBackward implementation
auto MaxBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    auto result = max(inputs[0].tensor(), dim_, keepdim_);
    // Save both input and output for backward
    save_for_backward({inputs[0].tensor(), result});
    return {Variable(result, true)};
}

namespace {
// Finite-difference-consistent argmax mask for max-over-`dim`, shared by the
// first-order backward() and higher-order backward_with_variables() so both
// return identical gradients. The mask is a constant w.r.t. differentiation
// (built only from saved tensors). Returns a tensor shaped like `input`.
//
//   grad_i = clamp((x_i + eps - max_{j!=i} x_j) / (2*eps), 0, 1)
//
// which matches gradcheck's numerical derivative for any input and reduces to
// the argmax one-hot when the gap exceeds eps. `max_{j!=i} x_j` equals the
// second-max when x_i is the unique argmax, otherwise the max.
Tensor compute_max_dim_mask(const Tensor& input, const Tensor& output,
                            int64_t dim, bool keepdim) {
    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());

    auto out = output;
    if (!keepdim) {
        out = unsqueeze(out, dim);
    }
    auto out_expanded = expand(out, input_shape_vec);

    double eps_val2;
    switch (input.dtype()) {
        case DType::Float64:  eps_val2 = 1e-6; break;
        case DType::Float16:
        case DType::BFloat16: eps_val2 = 1e-3; break;
        default:              eps_val2 = 5e-4; break;  // Float32
    }

    auto diff = sub(input, out_expanded);
    auto abs_diff = abs(diff);
    auto tie_epsilon = full(input_shape_vec,
                            (input.dtype() == DType::Float64) ? 1e-12
                              : (input.dtype() == DType::Float32) ? 1e-7
                              : 1e-3,
                            input.dtype(), input.device());
    auto is_argmax = lt(abs_diff, tie_epsilon);
    auto neg_inf = full(input_shape_vec,
                        -std::numeric_limits<double>::infinity(),
                        input.dtype(), input.device());
    auto x_sans_max = where(is_argmax, neg_inf, input);
    auto second_max = max(x_sans_max, dim, /*keepdim=*/true);
    auto second_max_expanded = expand(second_max, input_shape_vec);
    auto max_without = where(is_argmax, second_max_expanded, out_expanded);

    auto eps_tensor = full(input_shape_vec, eps_val2, input.dtype(), input.device());
    auto two_eps = full(input_shape_vec, 2.0 * eps_val2, input.dtype(), input.device());
    auto numerator = sub(add(input, eps_tensor), max_without);
    auto ratio = div(numerator, two_eps);
    auto mask = clamp(ratio, 0.0f, 1.0f);

    // Split the gradient equally among tied maxima so the per-dim mass sums to
    // 1, matching the global-max path (which divides by tie_count). At an exact
    // K-way tie every tied position would otherwise clamp to 1, returning mass
    // K instead of 1 and disagreeing with the central-difference derivative
    // (1/K each). Normalize by the per-dim tie count. Build the count in
    // Float32 so the sum does not saturate past 2048 ties in half precision,
    // then narrow back. Guard against a zero denominator (cannot happen for a
    // real max, but keeps the division well-defined).
    const bool is_half =
        (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
    auto mask_f32 = is_half ? mask.to(DType::Float32) : mask;
    auto tie_count = sum(mask_f32, dim, /*keepdim=*/true);
    auto tie_count_expanded = expand(tie_count, input_shape_vec);
    auto one_f32 = ones(input_shape_vec, DType::Float32, input.device());
    auto safe_tie_count = maximum(tie_count_expanded, one_f32);
    auto normalized = div(mask_f32, safe_tie_count);
    return is_half ? normalized.to(input.dtype()) : normalized;
}
}  // namespace

auto MaxBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MaxBackward: cannot compute gradient of max over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global max: gradient flows only to the maximum element
        // Create mask where input == output (broadcasted)

        // Reshape scalar output to match input dimensions before expanding
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == output (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        // Select epsilon appropriate for the tensor's precision
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Convert boolean mask to float for gradient computation. Build the
        // mask + tie_count in Float32: in Float16/BFloat16 the running sum
        // saturates past 2048 ties (and 1/tie_count underflows), silently
        // mis-normalising. Narrow the final grad back.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto zeros_tensor = zeros(input_shape_vec, DType::Float32, input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        // FIX: grad_output is also scalar, need to reshape before expanding
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);
        auto grad_f32 = is_half ? grad_broadcasted.to(DType::Float32) : grad_broadcasted;
        auto grad_input = mul(grad_f32, mask);
        if (is_half) grad_input = grad_input.to(input.dtype());

        return {grad_input};
    } else {
        // Dimension-specific max
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
        }
        auto grad_expanded = expand(grad, input_shape_vec);

        // Shared finite-difference-consistent mask: both this first-order path
        // and backward_with_variables() use compute_max_dim_mask() so that
        // .backward() and .backward(create_graph=true) return identical
        // gradients for max-over-dim.
        auto mask = compute_max_dim_mask(input, output, dim, keepdim_);
        return {mul(grad_expanded, mask)};
    }
}

auto MaxBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Higher-order Max backward: mask is a position-mask derived from saved
    // tensors (a constant wrt differentiation), so build it at Tensor level and
    // wrap as non-grad Variable. Threading the grad Variable through expand+mul
    // preserves the graph for create_graph=true.
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MaxBackward: cannot compute gradient of max over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Build mask + tie_count in Float32 so the sum doesn't saturate in
        // half precision; narrow the normalized mask back to input dtype.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto zeros_tensor = zeros(input_shape_vec, DType::Float32, input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        auto tie_count = sum(mask);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        auto grad_v = grad_var;
        if (grad_var.tensor().ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_v = tenzor::reshape(grad_v, ones_shape);
        }
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);
        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());

        auto grad_v = grad_var;
        if (!keepdim_) {
            grad_v = tenzor::unsqueeze(grad_v, dim);
        }
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);

        // Identical mask to the first-order path (a constant w.r.t.
        // differentiation), wrapped as a non-grad Variable so create_graph=true
        // still traces back through grad_expanded.
        auto mask = compute_max_dim_mask(input, output, dim, keepdim_);
        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    }
}

// MedianBackward implementation
auto MedianBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    int64_t dim = dim_.value_or(-1);
    auto [values, indices] = ::tenzor::median(inputs[0].tensor(), dim, keepdim_);
    save_for_backward({inputs[0].tensor(), values});
    return {Variable(values, true)};
}

auto MedianBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MedianBackward: cannot compute gradient of median over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global median: gradient flows only to the median element
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == median value (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Build mask + tie_count in Float32 so the sum doesn't saturate past
        // 2048 ties in half precision; narrow the final grad back.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto zeros_tensor = zeros(input_shape_vec, DType::Float32, input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);
        auto grad_f32 = is_half ? grad_broadcasted.to(DType::Float32) : grad_broadcasted;
        auto grad_input = mul(grad_f32, mask);
        if (is_half) grad_input = grad_input.to(input.dtype());

        return {grad_input};
    } else {
        // Dimension-specific median
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == median value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);

        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        // Build the tie mask + per-dim tie_count in Float32 so the sum does not
        // saturate past 2048 ties in half precision (matching the global path),
        // then narrow back to the input dtype.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto abs_diff_f32 = is_half ? abs_diff.to(DType::Float32) : abs_diff;
        auto epsilon = full(input_shape_vec, eps_val2, DType::Float32, input.device());
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto scaled_diff = div(abs_diff_f32, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        return {mul(grad_expanded, mask)};
    }
}

auto MedianBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Replicate backward() logic using Variable-level ops for higher-order gradients.
    // The mask is computed from saved tensors (constants), so only grad_output needs
    // Variable-level tracking. We wrap the mask as a non-grad Variable.
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MedianBackward: cannot compute gradient of median over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global median
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Build mask + tie_count in Float32 so the sum doesn't saturate in
        // half precision; narrow the normalized mask back to input dtype.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto zeros_tensor = zeros(input_shape_vec, DType::Float32, input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        auto tie_count = sum(mask);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        // Variable-level: broadcast grad and multiply by mask
        auto grad_v = grad_var;
        if (grad_var.tensor().ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_v = tenzor::reshape(grad_v, ones_shape);
        }
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);
        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());

        auto grad_v = grad_var;
        auto out = output;

        if (!keepdim_) {
            grad_v = tenzor::unsqueeze(grad_v, dim);
            out = unsqueeze(out, dim);
        }

        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);

        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);
        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        // Float32 tie mask + per-dim sum (half precision saturates past 2048
        // ties); narrow back to input dtype, matching the global path.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto abs_diff_f32 = is_half ? abs_diff.to(DType::Float32) : abs_diff;
        auto epsilon = full(input_shape_vec, eps_val2, DType::Float32, input.device());
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto scaled_diff = div(abs_diff_f32, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    }
}

// ModeBackward implementation
auto ModeBackward::forward(std::vector<Variable> inputs) -> std::vector<Variable> {
    int64_t dim = dim_.value_or(-1);
    auto [values, indices] = ::tenzor::mode(inputs[0].tensor(), dim, keepdim_);
    save_for_backward({inputs[0].tensor(), values});
    return {Variable(values, true)};
}

auto ModeBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "ModeBackward: cannot compute gradient of mode over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global mode: gradient flows only to the mode element(s)
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        // Create mask where input == mode value (within epsilon)
        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Build mask + tie_count in Float32 so the sum doesn't saturate past
        // 2048 ties in half precision; narrow the final grad back.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto zeros_tensor = zeros(input_shape_vec, DType::Float32, input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        // Normalize mask by tie count so gradient is split among tied elements
        auto tie_count = sum(mask);
        mask = div(mask, tie_count);

        // Broadcast grad_output to input shape
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);
        auto grad_f32 = is_half ? grad_broadcasted.to(DType::Float32) : grad_broadcasted;
        auto grad_input = mul(grad_f32, mask);
        if (is_half) grad_input = grad_input.to(input.dtype());

        return {grad_input};
    } else {
        // Dimension-specific mode
        int64_t dim = dim_.value();
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        if (!keepdim_) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == mode value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);

        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        // Float32 tie mask + per-dim sum (half precision saturates past 2048
        // ties); narrow back to input dtype, matching the global path.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto abs_diff_f32 = is_half ? abs_diff.to(DType::Float32) : abs_diff;
        auto epsilon = full(input_shape_vec, eps_val2, DType::Float32, input.device());
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto scaled_diff = div(abs_diff_f32, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        return {mul(grad_expanded, mask)};
    }
}

auto ModeBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Replicate backward() logic using Variable-level ops for higher-order gradients.
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "ModeBackward: cannot compute gradient of mode over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    if (!dim_.has_value()) {
        // Global mode
        auto output_reshaped = output;
        if (output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            output_reshaped = reshape(output, ones_shape);
        }
        auto output_expanded = expand(output_reshaped, input_shape_vec);

        auto diff = sub(input, output_expanded);
        auto abs_diff = abs(diff);
        double eps_val;
        switch (input.dtype()) {
            case DType::Float64:  eps_val = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val = 1e-3; break;
            default:              eps_val = 1e-7; break;
        }
        auto epsilon = full(input_shape_vec, eps_val, input.dtype(), input.device());
        auto mask_bool = lt(abs_diff, epsilon);
        // Build mask + tie_count in Float32 so the sum doesn't saturate in
        // half precision; narrow the normalized mask back to input dtype.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto zeros_tensor = zeros(input_shape_vec, DType::Float32, input.device());
        auto mask = where(mask_bool, ones_tensor, zeros_tensor);

        auto tie_count = sum(mask);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        // Variable-level: broadcast grad and multiply by mask
        auto grad_v = grad_var;
        if (grad_var.tensor().ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_v = tenzor::reshape(grad_v, ones_shape);
        }
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);
        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    } else {
        int64_t dim = dim_.value();
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());

        auto grad_v = grad_var;
        auto out = output;

        if (!keepdim_) {
            grad_v = tenzor::unsqueeze(grad_v, dim);
            out = unsqueeze(out, dim);
        }

        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);

        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);
        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        // Float32 tie mask + per-dim sum (half precision saturates past 2048
        // ties); narrow back to input dtype, matching the global path.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto abs_diff_f32 = is_half ? abs_diff.to(DType::Float32) : abs_diff;
        auto epsilon = full(input_shape_vec, eps_val2, DType::Float32, input.device());
        auto ones_tensor = ones(input_shape_vec, DType::Float32, input.device());
        auto scaled_diff = div(abs_diff_f32, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);
        if (is_half) mask = mask.to(input.dtype());

        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    }
}

} // namespace tenzor
