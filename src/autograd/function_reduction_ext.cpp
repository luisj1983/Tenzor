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

namespace {
// Value-match smear mask for a GLOBAL min: gradient flows to element(s) equal
// to the saved scalar min `output`. Uses the SAME `1 - clamp(|x-output|/eps)`
// smear convention as MinBackward's dim path (so global and dim agree),
// normalised by the global tie count with a safe `maximum(tie_count,1)` guard.
// The relative-to-eps smear + guard removes the all-zero-mask 0/0 NaN the old
// absolute-epsilon hard one-hot could produce for large-magnitude inputs.
// (Mirrors compute_value_match_global_mask in function_reduction.cpp; kept
// local to this TU since that helper lives in an anonymous namespace.)
Tensor compute_min_value_match_global_mask(const Tensor& input,
                                           const Tensor& output) {
    auto input_shape_vec =
        std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto output_reshaped = output;
    if (output.ndim() == 0) {
        std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
        output_reshaped = reshape(output, ones_shape);
    }
    auto output_expanded = expand(output_reshaped, input_shape_vec);

    double eps_val;
    switch (input.dtype()) {
        case DType::Float64:  eps_val = 1e-12; break;
        case DType::Float16:
        case DType::BFloat16: eps_val = 1e-3;  break;
        default:              eps_val = 1e-7;  break;
    }
    // Do the mask arithmetic in the input's natural floating precision: Float64
    // inputs must stay in Float64 (building epsilon/ones in Float32 silently
    // capped the comparison at single precision), while half precision widens
    // to Float32. Mirrors compute_value_match_global_mask in function_reduction.cpp.
    const DType work_dtype =
        (input.dtype() == DType::Float64) ? DType::Float64 : DType::Float32;
    auto abs_diff = abs(sub(input, output_expanded));
    auto abs_diff_w =
        (abs_diff.dtype() != work_dtype) ? abs_diff.to(work_dtype) : abs_diff;
    auto epsilon = full(input_shape_vec, eps_val, work_dtype, input.device());
    auto ones_tensor = ones(input_shape_vec, work_dtype, input.device());
    auto clamped = clamp(div(abs_diff_w, epsilon), 0.0, 1.0);
    auto mask = sub(ones_tensor, clamped);

    auto tie_count = sum(mask);
    auto tie_count_exp = expand(reshape(tie_count,
        std::vector<int64_t>(input_shape_vec.size(), 1)), input_shape_vec);
    auto safe_tie = maximum(tie_count_exp, ones_tensor);
    auto normalized = div(mask, safe_tie);
    return (normalized.dtype() != input.dtype()) ? normalized.to(input.dtype())
                                                 : normalized;
}
}  // namespace

// =========================================================================
// Reduction Backward Functions
// =========================================================================

// MinBackward implementation
// Same pattern as MaxBackward. Save input+output. backward: mask where input == min_val
auto MinBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("MinBackward::forward should not be called");
}

auto MinBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];  // min values
    const auto& grad_output = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MinBackward: cannot compute gradient of min over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // The saved_tensors_[2] holds dim as a scalar Int64 tensor (or not present for global min)
    bool has_dim = saved_tensors_.size() > 2;

    if (!has_dim) {
        // Global min: route through the SAME 1-clamp smear convention as the
        // dim path (compute_min_value_match_global_mask), safe tie-count guard.
        auto mask = compute_min_value_match_global_mask(input, output);
        auto grad_reshaped = grad_output;
        if (grad_output.ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_reshaped = reshape(grad_output, ones_shape);
        }
        auto grad_broadcasted = expand(grad_reshaped, input_shape_vec);
        return {mul(grad_broadcasted, mask)};
    } else {
        // Dimension-specific min
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();

        auto grad = grad_output;
        auto out = output;

        // Check if keepdim was used by comparing shapes
        bool keepdim = (output.ndim() == input.ndim());

        if (!keepdim) {
            grad = unsqueeze(grad, dim);
            out = unsqueeze(out, dim);
        }

        // Expand to input shape
        auto out_expanded = expand(out, input_shape_vec);
        auto grad_expanded = expand(grad, input_shape_vec);

        // Create mask where input == min_value
        auto diff = sub(input, out_expanded);
        auto abs_diff = abs(diff);
        double eps_val2;
        switch (input.dtype()) {
            case DType::Float64:  eps_val2 = 1e-12; break;
            case DType::Float16:
            case DType::BFloat16: eps_val2 = 1e-3; break;
            default:              eps_val2 = 1e-7; break;
        }
        // Build mask + tie_count in Float32: in Float16/BFloat16 the running
        // sum saturates past 2048 ties (and 1/tie_count underflows), silently
        // mis-normalising the gradient along long axes. Narrow the final grad back.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto abs_diff_f32 = is_half ? abs_diff.to(DType::Float32) : abs_diff;
        auto epsilon = full(input_shape_vec, eps_val2, is_half ? DType::Float32 : input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, is_half ? DType::Float32 : input.dtype(), input.device());
        auto scaled_diff = div(abs_diff_f32, epsilon);
        auto clamped = clamp(scaled_diff, 0.0f, 1.0f);
        auto mask = sub(ones_tensor, clamped);

        // Normalize mask by tie count along dim
        auto tie_count = sum(mask, dim, /*keepdim=*/true);
        mask = div(mask, tie_count);

        auto grad_f32 = is_half ? grad_expanded.to(DType::Float32) : grad_expanded;
        auto grad_input = mul(grad_f32, mask);
        if (is_half) grad_input = grad_input.to(input.dtype());

        return {grad_input};
    }
}

auto MinBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // Higher-order: mask is a position-mask (constant wrt differentiation);
    // build at Tensor level, wrap as non-grad Variable, thread the grad Variable
    // through expand+mul to preserve the graph for create_graph=true.
    const auto& input = saved_tensors_[0];
    const auto& output = saved_tensors_[1];
    const auto& grad_var = grad_outputs[0];

    TENZOR_CHECK_SHAPE(input.numel() > 0,
        "MinBackward: cannot compute gradient of min over empty tensor");

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    bool has_dim = saved_tensors_.size() > 2;

    if (!has_dim) {
        // Same 1-clamp smear global mask as the first-order path (constant wrt
        // differentiation): consistent with the dim path, safe tie-count guard.
        auto mask = compute_min_value_match_global_mask(input, output);
        auto grad_v = grad_var;
        if (grad_var.tensor().ndim() == 0) {
            std::vector<int64_t> ones_shape(input_shape_vec.size(), 1);
            grad_v = tenzor::reshape(grad_v, ones_shape);
        }
        auto grad_expanded = tenzor::expand(grad_v, input_shape_vec);
        auto mask_var = Variable(mask, false);
        return {grad_expanded * mask_var};
    } else {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += static_cast<int64_t>(input.shape().size());

        auto grad_v = grad_var;
        auto out = output;
        bool keepdim = (output.ndim() == input.ndim());

        if (!keepdim) {
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
        // Build mask + tie_count in Float32 so the sum doesn't saturate in
        // half precision; narrow the normalized mask back to input dtype.
        const bool is_half = (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16);
        auto abs_diff_f32 = is_half ? abs_diff.to(DType::Float32) : abs_diff;
        auto epsilon = full(input_shape_vec, eps_val2, is_half ? DType::Float32 : input.dtype(), input.device());
        auto ones_tensor = ones(input_shape_vec, is_half ? DType::Float32 : input.dtype(), input.device());
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

// StdBackward implementation
// Saves input and output. backward: grad * (input - mean) / (N * output)
auto StdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("StdBackward::forward should not be called");
}

auto StdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Widen Float16/BFloat16 to Float32 for the (input-mean)/(denom*std)
    // arithmetic: half precision saturates / loses the subtractive cancellation
    // in (input - mean). Compute in Float32 and narrow grad_input back.
    const auto orig_dtype = saved_tensors_[0].dtype();
    const bool is_half = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);

    const Tensor grad = is_half ? grad_outputs[0].to(DType::Float32) : grad_outputs[0];
    const Tensor input = is_half ? saved_tensors_[0].to(DType::Float32) : saved_tensors_[0];
    const Tensor std_out = is_half ? saved_tensors_[1].to(DType::Float32) : saved_tensors_[1];  // std(x)

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Determine dim and N from saved_tensors_[2] if present
    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    int64_t N;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        N = input.shape()[dim];
        keepdim = (std_out.ndim() == input.ndim());
    } else {
        N = input.numel();
        keepdim = false;
    }

    // Compute mean of input
    auto input_mean = mean(input, dim_opt, true);

    // (input - mean)
    auto diff = sub(input, expand(input_mean, input_shape_vec));

    // Expand std and grad to input shape
    auto std_expanded = std_out;
    auto grad_expanded = grad;
    if (dim_opt.has_value() && !keepdim) {
        std_expanded = unsqueeze(std_out, dim_opt.value());
        grad_expanded = unsqueeze(grad, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        if (std_out.ndim() > 0) {
            std_expanded = reshape(std_out, std::vector<int64_t>(input_shape_vec.size(), 1));
        } else {
            std_expanded = reshape(std_out, std::vector<int64_t>(input_shape_vec.size(), 1));
        }
        if (grad.ndim() > 0) {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        } else {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        }
    }

    std_expanded = expand(std_expanded, input_shape_vec);
    grad_expanded = expand(grad_expanded, input_shape_vec);

    // R.3: grad_input = grad * (input - mean) / (denom * std)
    // denom = N - 1 when unbiased=true (Bessel), N when unbiased=false.
    // Falls back to 1 when N <= 1 to avoid div-by-zero (gradient there is
    // ill-defined since variance is zero / undefined with one sample).
    double denom = unbiased_ ? ((N > 1) ? static_cast<double>(N - 1) : 1.0)
                             : ((N > 0) ? static_cast<double>(N)     : 1.0);
    auto n_std = mul(std_expanded, denom);
    auto grad_input = div(mul(grad_expanded, diff), n_std);

    if (is_half) grad_input = grad_input.to(orig_dtype);
    return {grad_input};
}

auto StdBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // std backward: grad * (input - mean) / (N * std)
    // diff and n_std are constants (don't depend on grad); compute at Tensor level
    const auto& input = saved_tensors_[0];
    const auto& std_out = saved_tensors_[1];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    int64_t N;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        N = input.shape()[dim];
        keepdim = (std_out.ndim() == input.ndim());
    } else {
        N = input.numel();
        keepdim = false;
    }

    auto input_mean = mean(input, dim_opt, true);
    auto diff = sub(input, expand(input_mean, input_shape_vec));

    auto std_expanded = std_out;
    if (dim_opt.has_value() && !keepdim) {
        std_expanded = unsqueeze(std_out, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        std_expanded = reshape(std_out, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    std_expanded = expand(std_expanded, input_shape_vec);
    // R.3: denom = N - 1 when unbiased=true (Bessel), N when unbiased=false.
    double denom = unbiased_ ? ((N > 1) ? static_cast<double>(N - 1) : 1.0)
                             : ((N > 0) ? static_cast<double>(N)     : 1.0);
    auto n_std = mul(std_expanded, denom);
    auto factor = div(diff, n_std);
    Variable factor_var(factor, false);

    // Expand grad at Variable level so reshape/expand are tracked for higher-order
    auto grad_var = grad_outputs[0];
    if (dim_opt.has_value() && !keepdim) {
        grad_var = tenzor::unsqueeze(grad_var, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        grad_var = tenzor::reshape(grad_var, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    grad_var = tenzor::expand(grad_var, input_shape_vec);

    return {grad_var * factor_var};
}

// VarBackward implementation
// Saves input. backward: grad * 2 * (input - mean) / N
auto VarBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("VarBackward::forward should not be called");
}

auto VarBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    // Widen Float16/BFloat16 to Float32 for the 2*(input-mean)/denom
    // arithmetic; half precision loses the subtractive cancellation in
    // (input - mean). Compute in Float32 and narrow grad_input back.
    const auto orig_dtype = saved_tensors_[0].dtype();
    const bool is_half = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);

    const Tensor grad = is_half ? grad_outputs[0].to(DType::Float32) : grad_outputs[0];
    const Tensor input = is_half ? saved_tensors_[0].to(DType::Float32) : saved_tensors_[0];
    const Tensor var_out = is_half ? saved_tensors_[1].to(DType::Float32) : saved_tensors_[1];

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Determine dim and N from saved_tensors_[2] if present
    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    int64_t N;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        N = input.shape()[dim];
        keepdim = (var_out.ndim() == input.ndim());
    } else {
        N = input.numel();
        keepdim = false;
    }

    // Compute mean of input
    auto input_mean = mean(input, dim_opt, true);

    // (input - mean)
    auto diff = sub(input, expand(input_mean, input_shape_vec));

    // Expand grad to input shape
    auto grad_expanded = grad;
    if (dim_opt.has_value() && !keepdim) {
        grad_expanded = unsqueeze(grad, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        if (grad.ndim() > 0) {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        } else {
            grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
        }
    }
    grad_expanded = expand(grad_expanded, input_shape_vec);

    // R.3: grad_input = grad * 2 * (input - mean) / denom
    // denom = N - 1 when unbiased=true (Bessel), N when unbiased=false.
    double denom = unbiased_ ? ((N > 1) ? static_cast<double>(N - 1) : 1.0)
                             : ((N > 0) ? static_cast<double>(N)     : 1.0);
    auto scale = 2.0 / denom;
    auto grad_input = mul(mul(grad_expanded, diff), scale);

    if (is_half) grad_input = grad_input.to(orig_dtype);
    return {grad_input};
}

auto VarBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // var backward: grad * 2 * (input - mean) / N
    const auto& input = saved_tensors_[0];
    const auto& var_out = saved_tensors_[1];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    bool has_dim = saved_tensors_.size() > 2;
    std::optional<int64_t> dim_opt;
    int64_t N;
    bool keepdim;

    if (has_dim) {
        int64_t dim = saved_tensors_[2].data<int64_t>()[0];
        if (dim < 0) dim += input.shape().size();
        dim_opt = dim;
        N = input.shape()[dim];
        keepdim = (var_out.ndim() == input.ndim());
    } else {
        N = input.numel();
        keepdim = false;
    }

    // diff = (input - mean) is a constant w.r.t. higher-order gradients
    auto input_mean = mean(input, dim_opt, true);
    auto diff = sub(input, expand(input_mean, input_shape_vec));
    // R.3: denom = N - 1 when unbiased=true (Bessel), N when unbiased=false.
    double denom = unbiased_ ? ((N > 1) ? static_cast<double>(N - 1) : 1.0)
                             : ((N > 0) ? static_cast<double>(N)     : 1.0);
    double scale = 2.0 / denom;
    auto factor = mul(diff, scale);
    Variable factor_var(factor, false);

    // Expand grad at Variable level
    auto grad_var = grad_outputs[0];
    if (dim_opt.has_value() && !keepdim) {
        grad_var = tenzor::unsqueeze(grad_var, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        grad_var = tenzor::reshape(grad_var, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    grad_var = tenzor::expand(grad_var, input_shape_vec);

    return {grad_var * factor_var};
}

// ProdBackward implementation
// Saves input and output. dy/dx_i = product over j != i of x_j.
//
// Correct handling for arbitrary numbers of zeros in the input:
//   Let S = set of indices where x == 0 along the reduction axes, and
//   prod_safe = product of x with zeros replaced by 1 (= product of non-zero entries).
//   For each position i in the reduction group:
//     * i ∉ S, |S| == 0  → dy/dx_i = prod_safe / x_i  (= full prod / x_i)
//     * i ∈ S, |S| == 1  → dy/dx_i = prod_safe       (= product of all non-zero entries)
//     * otherwise        → dy/dx_i = 0
//   This is equivalent to: factor_i = prod_safe / safe_input_i when
//   (zero_count - mask_zero_i) == 0, else 0.
namespace {

auto compute_prod_backward_factor(const Tensor& input,
                                  std::optional<int64_t> dim_opt,
                                  bool keepdim) -> Tensor {
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    auto zero_in = zeros(input_shape_vec, input.dtype(), input.device());
    auto ones_in = ones(input_shape_vec, input.dtype(), input.device());

    auto mask_zero = eq(input, zero_in);
    auto safe_input = where(mask_zero, ones_in, input);

    // prod_safe along reduction axes, then broadcast back to input shape.
    Tensor prod_safe = prod(safe_input, dim_opt, keepdim);
    Tensor prod_safe_expanded = prod_safe;
    if (dim_opt.has_value() && !keepdim) {
        prod_safe_expanded = unsqueeze(prod_safe, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        prod_safe_expanded =
            reshape(prod_safe, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    prod_safe_expanded = expand(prod_safe_expanded, input_shape_vec);

    auto factor_raw = div(prod_safe_expanded, safe_input);

    // zero_count along reduction axes, broadcast back. mask cast to Int64 so
    // arithmetic works without dtype-narrowing surprises.
    auto mask_zero_int = mask_zero.to(DType::Int64);
    Tensor zero_count = sum(mask_zero_int, dim_opt, keepdim);
    Tensor zero_count_expanded = zero_count;
    if (dim_opt.has_value() && !keepdim) {
        zero_count_expanded = unsqueeze(zero_count, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        zero_count_expanded =
            reshape(zero_count, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    zero_count_expanded = expand(zero_count_expanded, input_shape_vec);

    // remaining_for_i = zero_count - mask_zero_i. The factor is correct iff this
    // is zero (i.e. removing this element leaves no zeros in the product).
    auto remaining = sub(zero_count_expanded, mask_zero_int);
    auto zeros_int = zeros(input_shape_vec, DType::Int64, input.device());
    auto keep = eq(remaining, zeros_int);

    return where(keep, factor_raw, zero_in);
}

auto resolve_prod_dim(const Tensor& input, const Tensor& prod_out,
                      bool has_dim, const Tensor& dim_tensor)
    -> std::pair<std::optional<int64_t>, bool> {
    if (!has_dim) {
        return {std::nullopt, false};
    }
    int64_t dim = dim_tensor.data<int64_t>()[0];
    if (dim < 0) dim += input.shape().size();
    bool keepdim = (prod_out.ndim() == input.ndim());
    return {dim, keepdim};
}

}  // namespace

auto ProdBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("ProdBackward::forward should not be called");
}

auto ProdBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& prod_out = saved_tensors_[1];

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    bool has_dim = saved_tensors_.size() > 2;
    auto [dim_opt, keepdim] = resolve_prod_dim(
        input, prod_out, has_dim, has_dim ? saved_tensors_[2] : Tensor{});

    // Broadcast incoming gradient to input shape.
    auto grad_expanded = grad;
    if (dim_opt.has_value() && !keepdim) {
        grad_expanded = unsqueeze(grad, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        grad_expanded = reshape(grad, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    grad_expanded = expand(grad_expanded, input_shape_vec);

    auto factor = compute_prod_backward_factor(input, dim_opt, keepdim);
    return {mul(grad_expanded, factor)};
}

auto ProdBackward::backward_with_variables(std::vector<Variable> grad_outputs)
    -> std::vector<Variable> {
    const auto& input = saved_tensors_[0];
    const auto& prod_out = saved_tensors_[1];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    bool has_dim = saved_tensors_.size() > 2;
    auto [dim_opt, keepdim] = resolve_prod_dim(
        input, prod_out, has_dim, has_dim ? saved_tensors_[2] : Tensor{});

    // Factor is a constant w.r.t. the autograd graph (depends only on input,
    // which is saved as a tensor); wrap in a non-grad Variable.
    Variable factor_var(compute_prod_backward_factor(input, dim_opt, keepdim), false);

    auto grad_var = grad_outputs[0];
    if (dim_opt.has_value() && !keepdim) {
        grad_var = tenzor::unsqueeze(grad_var, dim_opt.value());
    } else if (!dim_opt.has_value()) {
        grad_var = tenzor::reshape(grad_var, std::vector<int64_t>(input_shape_vec.size(), 1));
    }
    grad_var = tenzor::expand(grad_var, input_shape_vec);

    return {grad_var * factor_var};
}

// LogSumExpBackward implementation
// Saves input and output. backward: grad * softmax(input, dim)
auto LogSumExpBackward::forward(std::vector<Variable>) -> std::vector<Variable> {
    throw std::runtime_error("LogSumExpBackward::forward should not be called");
}

auto LogSumExpBackward::backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> {
    const auto& grad = grad_outputs[0];
    const auto& input = saved_tensors_[0];
    const auto& lse_out = saved_tensors_[1];

    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    // Determine dim from saved_tensors_[2]
    int64_t dim = saved_tensors_[2].data<int64_t>()[0];
    if (dim < 0) dim += input.shape().size();

    bool keepdim = (lse_out.ndim() == input.ndim());

    // softmax(input, dim) = exp(input - logsumexp(input, dim))
    auto lse_expanded = lse_out;
    auto grad_expanded = grad;
    if (!keepdim) {
        lse_expanded = unsqueeze(lse_out, dim);
        grad_expanded = unsqueeze(grad, dim);
    }
    lse_expanded = expand(lse_expanded, input_shape_vec);
    grad_expanded = expand(grad_expanded, input_shape_vec);

    auto softmax_val = exp(sub(input, lse_expanded));

    return {mul(grad_expanded, softmax_val)};
}

auto LogSumExpBackward::backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> {
    // logsumexp backward: grad * softmax(input, dim) where softmax = exp(input - lse)
    // GG.1 / HH.10: on the higher-order path, recompute the softmax via the
    // Variable-level ops sourced from saved_variables_[0] (the input Variable
    // with its grad_fn chain). The previous code built softmax from raw saved
    // tensors and wrapped as `Variable(..., false)`, severing the chain — so
    // double-backward through cross-entropy (whose kernel is logsumexp) lost
    // the softmax-Jacobian path.
    const auto& input = saved_tensors_[0];
    const auto& lse_out = saved_tensors_[1];
    auto input_shape_vec = std::vector<int64_t>(input.shape().begin(), input.shape().end());

    int64_t dim = saved_tensors_[2].data<int64_t>()[0];
    if (dim < 0) dim += input.shape().size();
    bool keepdim = (lse_out.ndim() == input.ndim());

    Variable softmax_var;
    if (has_saved_variables()) {
        // Variable-level: lse = logsumexp(input, dim, keepdim); softmax =
        // exp(input - expand(lse)). grad_fn chains back through input.
        auto lse_v = tenzor::logsumexp(saved_variables_[0], dim, keepdim);
        if (!keepdim) {
            lse_v = tenzor::unsqueeze(lse_v, dim);
        }
        lse_v = tenzor::expand(lse_v, input_shape_vec);
        softmax_var = tenzor::exp(saved_variables_[0] - lse_v);
    } else {
        auto lse_expanded = lse_out;
        if (!keepdim) {
            lse_expanded = unsqueeze(lse_out, dim);
        }
        lse_expanded = expand(lse_expanded, input_shape_vec);
        auto softmax_val = exp(sub(input, lse_expanded));
        softmax_var = Variable(softmax_val, false);
    }

    // Expand grad at Variable level
    auto grad_var = grad_outputs[0];
    if (!keepdim) {
        grad_var = tenzor::unsqueeze(grad_var, dim);
    }
    grad_var = tenzor::expand(grad_var, input_shape_vec);

    return {grad_var * softmax_var};
}

} // namespace tenzor
