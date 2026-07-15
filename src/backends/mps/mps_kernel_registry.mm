/**
 * @file mps_kernel_registry.mm
 * @brief MPS kernel registration for Tier 1 operations
 *
 * Registers Metal compute shader kernels with the dispatch table.
 * Tier 1 covers the essential ops needed for inference.
 */

#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>

#include "mps_backend.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/shape.hpp"

#include <limits>
#include <stdexcept>

namespace tenzor::mps {

// ============================================================================
// Zero-copy shape operations (metadata-only, no Metal dispatch needed)
// ============================================================================

static auto mps_reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor {
    if (!input.is_contiguous()) {
        // Non-contiguous: must materialize via CPU (rare path)
        auto cpu_input = input.to(Device::cpu());
        auto cpu_result = cpu_input.reshape(new_shape);
        return cpu_result.to(input.device());
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    result.mutable_shape() = new_shape;
    result.mutable_strides() = compute_strides(new_shape);
    return result;
}

static auto mps_transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor {
    const int64_t ndim = input.ndim();
    if (dim0 < 0) dim0 += ndim;
    if (dim1 < 0) dim1 += ndim;
    // Validate before indexing: this kernel is reachable directly via dispatch,
    // not only through the validated Tensor::transpose() path. An out-of-range
    // dim would be an OOB vector access on r_shape/r_strides.
    if (dim0 < 0 || dim0 >= ndim || dim1 < 0 || dim1 >= ndim) {
        throw std::runtime_error("mps transpose: dim out of range");
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    std::swap(r_shape[dim0], r_shape[dim1]);
    std::swap(r_strides[dim0], r_strides[dim1]);
    return result;
}

static auto mps_permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor {
    const int64_t ndim = input.ndim();
    // Validate that dims is a permutation of [0, ndim) before indexing
    // input.shape()[dims[i]] — reachable directly via dispatch.
    if (static_cast<int64_t>(dims.size()) != ndim) {
        throw std::runtime_error("mps permute: dims size must equal tensor rank");
    }
    std::vector<bool> seen(ndim, false);
    std::vector<int64_t> ndims_resolved(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        int64_t d = dims[i];
        if (d < 0) d += ndim;
        if (d < 0 || d >= ndim) {
            throw std::runtime_error("mps permute: dim out of range");
        }
        if (seen[d]) {
            throw std::runtime_error("mps permute: duplicate dim in permutation");
        }
        seen[d] = true;
        ndims_resolved[i] = d;
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = input.shape()[ndims_resolved[i]];
        new_strides[i] = input.strides()[ndims_resolved[i]];
    }
    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);
    return result;
}

// Sentinel used by the registry to mean "squeeze all size-1 dims".
static constexpr int64_t kSqueezeAllDims = std::numeric_limits<int64_t>::min();

static auto mps_squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    if (dim != kSqueezeAllDims) {
        // Explicit dim: normalize negatives, then squeeze only if size == 1
        // (PyTorch semantics — a non-size-1 dim is a no-op, never dropped).
        const int64_t ndim = input.ndim();
        if (dim < 0) dim += ndim;
        if (dim < 0 || dim >= ndim) {
            throw std::runtime_error("squeeze: dim out of range");
        }
        if (input.shape()[dim] == 1) {
            auto& r_shape = result.mutable_shape();
            auto& r_strides = result.mutable_strides();
            r_shape.erase(r_shape.begin() + dim);
            r_strides.erase(r_strides.begin() + dim);
        }
    } else {
        std::vector<int64_t> new_shape;
        std::vector<int64_t> new_strides;
        for (int64_t i = 0; i < input.ndim(); ++i) {
            if (input.shape()[i] != 1) {
                new_shape.push_back(input.shape()[i]);
                new_strides.push_back(input.strides()[i]);
            }
        }
        if (new_shape.empty()) {
            new_shape.push_back(1);
            new_strides.push_back(1);
        }
        result.mutable_shape() = std::move(new_shape);
        result.mutable_strides() = std::move(new_strides);
    }
    return result;
}

static auto mps_unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    // PyTorch accepts dim in [-(ndim+1), ndim]; normalize negatives so that
    // unsqueeze(-1) inserts at the end rather than dereferencing before
    // begin() (UB / heap corruption in std::vector::insert).
    const int64_t ndim = input.ndim();
    if (dim < 0) dim += ndim + 1;
    if (dim < 0 || dim > ndim) {
        throw std::runtime_error("unsqueeze: dim out of range");
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);
    int64_t new_stride = (dim < ndim) ? input.strides()[dim] : 1;
    r_strides.insert(r_strides.begin() + dim, new_stride);
    return result;
}

static auto mps_flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor {
    auto shape = input.shape();
    int64_t ndim = shape.size();
    if (start_dim < 0) start_dim += ndim;
    if (end_dim < 0) end_dim += ndim;
    int64_t flat_size = 1;
    for (int64_t d = start_dim; d <= end_dim; ++d) {
        flat_size *= shape[d];
    }
    std::vector<int64_t> new_shape;
    for (int64_t d = 0; d < start_dim; ++d) new_shape.push_back(shape[d]);
    new_shape.push_back(flat_size);
    for (int64_t d = end_dim + 1; d < ndim; ++d) new_shape.push_back(shape[d]);
    return mps_reshape_kernel(input, new_shape);
}

static auto mps_expand_kernel(const Tensor& input, const std::vector<int64_t>& target_shape) -> Tensor {
    const auto& in_shape = input.shape();
    const auto& in_strides = input.strides();
    int64_t ndim_out = static_cast<int64_t>(target_shape.size());
    int64_t ndim_in = input.ndim();
    int64_t dim_diff = ndim_out - ndim_in;

    std::vector<int64_t> new_strides(ndim_out, 0);
    for (int64_t i = ndim_out - 1; i >= 0; --i) {
        int64_t in_idx = i - dim_diff;
        if (in_idx >= 0) {
            if (in_shape[in_idx] == target_shape[i]) {
                new_strides[i] = in_strides[in_idx];
            } else if (in_shape[in_idx] == 1) {
                new_strides[i] = 0;  // Broadcast
            } else {
                throw std::runtime_error("expand: incompatible shapes");
            }
        }
    }
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    result.mutable_shape() = target_shape;
    result.mutable_strides() = new_strides;
    return result;
}

// ============================================================================
// Forward declarations — Tier 1 (mps_elementwise.mm)
// ============================================================================
Tensor mps_add_kernel(const Tensor& a, const Tensor& b);
Tensor mps_sub_kernel(const Tensor& a, const Tensor& b);
Tensor mps_mul_kernel(const Tensor& a, const Tensor& b);
Tensor mps_div_kernel(const Tensor& a, const Tensor& b);
Tensor mps_relu_kernel(const Tensor& input);
Tensor mps_sigmoid_kernel(const Tensor& input);
Tensor mps_neg_kernel(const Tensor& input);
Tensor mps_exp_kernel(const Tensor& input);
Tensor mps_log_kernel(const Tensor& input);
Tensor mps_tanh_kernel(const Tensor& input);
Tensor mps_sqrt_kernel(const Tensor& input);
Tensor mps_abs_kernel(const Tensor& input);
Tensor mps_pow_kernel(const Tensor& base, const Tensor& exponent);
Tensor mps_clamp_kernel(const Tensor& input, float min_val, float max_val);
Tensor mps_matmul_kernel(const Tensor& a, const Tensor& b);
// Phase 5: additional element-wise math ops
Tensor mps_log2_kernel(const Tensor& input);
Tensor mps_log10_kernel(const Tensor& input);
Tensor mps_log1p_kernel(const Tensor& input);
Tensor mps_exp2_kernel(const Tensor& input);
Tensor mps_expm1_kernel(const Tensor& input);
Tensor mps_erf_kernel(const Tensor& input);
Tensor mps_erfc_kernel(const Tensor& input);
Tensor mps_isnan_kernel(const Tensor& input);
Tensor mps_isinf_kernel(const Tensor& input);
Tensor mps_isfinite_kernel(const Tensor& input);
Tensor mps_rsqrt_kernel(const Tensor& input);
Tensor mps_square_kernel(const Tensor& input);
Tensor mps_reciprocal_kernel(const Tensor& input);
Tensor mps_deg2rad_kernel(const Tensor& input);
Tensor mps_rad2deg_kernel(const Tensor& input);
Tensor mps_logit_kernel(const Tensor& input);
Tensor mps_signbit_kernel(const Tensor& input);
Tensor mps_isreal_kernel(const Tensor& input);
Tensor mps_isposinf_kernel(const Tensor& input);
Tensor mps_isneginf_kernel(const Tensor& input);
Tensor mps_atan2_kernel(const Tensor& a, const Tensor& b);
Tensor mps_fmod_kernel(const Tensor& a, const Tensor& b);
Tensor mps_remainder_kernel(const Tensor& a, const Tensor& b);
Tensor mps_copysign_kernel(const Tensor& a, const Tensor& b);
// Additional native elementwise wrappers backing the W2 registry refit
// (kernels/mps_elementwise.mm). They replace mps_accelerate_* CPU
// roundtrips for ops that have a Metal shader.
Tensor mps_ceil_kernel(const Tensor& input);
Tensor mps_floor_kernel(const Tensor& input);
Tensor mps_gelu_kernel(const Tensor& input);
Tensor mps_acos_kernel(const Tensor& input);
Tensor mps_asin_kernel(const Tensor& input);
Tensor mps_atan_kernel(const Tensor& input);
// Native Metal adaptive max-pool 3D backward (kernels/mps_pooling.mm).
Tensor mps_adaptive_maxpool3d_backward_kernel(
    const Tensor& grad_output, const Tensor& indices,
    const std::vector<int64_t>& input_shape);
Tensor mps_nextafter_kernel(const Tensor& a, const Tensor& b);
Tensor mps_float_power_kernel(const Tensor& a, const Tensor& b);
Tensor mps_xlog1py_kernel(const Tensor& a, const Tensor& b);
Tensor mps_ldexp_kernel(const Tensor& a, const Tensor& b);
Tensor mps_hypot_kernel(const Tensor& a, const Tensor& b);

Tensor mps_sparse_spmv_kernel(const Tensor& crow_indices,
                              const Tensor& col_indices,
                              const Tensor& values,
                              const Tensor& x,
                              int64_t M, int64_t K);
Tensor mps_sparse_spmm_kernel(const Tensor& crow_indices,
                              const Tensor& col_indices,
                              const Tensor& values,
                              const Tensor& B,
                              int64_t M, int64_t K);
Tensor mps_linear_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias);
Tensor mps_embedding_kernel(const Tensor& weight, const Tensor& indices);
Tensor mps_softmax_kernel(const Tensor& input, int64_t dim);
Tensor mps_batch_norm_kernel(const Tensor& input, const Tensor& mean,
                              const Tensor& var, const Tensor& weight,
                              const Tensor& bias, float eps);
Tensor mps_layer_norm_kernel(const Tensor& input, const Tensor& weight,
                              const Tensor& bias, float eps);
std::tuple<Tensor, Tensor, Tensor> mps_layer_norm_kernel_with_stats(
    const Tensor& input, const Tensor& weight,
    const Tensor& bias, float eps);

// Native Metal norm kernels (kernels/mps_normalization.mm). Forward
// declarations so the registry can dispatch directly instead of routing
// through mps_accelerate_* (which CPU-roundtrips via unified memory).
std::vector<Tensor> mps_rmsnorm_forward(const Tensor& input, const Tensor& weight,
                                         float eps);
std::vector<Tensor> mps_rmsnorm_backward(const Tensor& grad_output, const Tensor& input,
                                          const Tensor& weight, const Tensor& rrms);
auto ctc_loss_forward_kernel(const Tensor& log_probs,
                             const Tensor& targets,
                             const Tensor& input_lengths,
                             const Tensor& target_lengths,
                             int64_t blank,
                             bool zero_infinity) -> std::vector<Tensor>;
std::vector<Tensor> mps_groupnorm_forward(const Tensor& input, int64_t num_groups,
                                           const Tensor& weight, const Tensor& bias,
                                           float eps);
std::vector<Tensor> mps_groupnorm_backward(const Tensor& grad_output, const Tensor& input,
                                            int64_t num_groups,
                                            const Tensor& mean_saved,
                                            const Tensor& rstd_saved,
                                            const Tensor& weight);
std::vector<Tensor> mps_instancenorm_forward(const Tensor& input, const Tensor& weight,
                                              const Tensor& bias, float eps);
std::vector<Tensor> mps_instancenorm_backward(const Tensor& grad_output, const Tensor& input,
                                               const Tensor& mean_saved,
                                               const Tensor& rstd_saved,
                                               const Tensor& weight);
Tensor mps_conv2d_kernel(const Tensor& input, const Tensor& weight,
                          int64_t stride_h, int64_t stride_w,
                          int64_t pad_h, int64_t pad_w,
                          int64_t dilation_h, int64_t dilation_w,
                          int64_t groups);
Tensor mps_sum_kernel(const Tensor& input, int64_t dim, bool keepdim);
Tensor mps_mean_kernel(const Tensor& input, int64_t dim, bool keepdim);
Tensor mps_max_kernel(const Tensor& input, int64_t dim, bool keepdim, Tensor& out_indices);
Tensor mps_gt_kernel(const Tensor& a, const Tensor& b);
Tensor mps_eq_kernel(const Tensor& a, const Tensor& b);
Tensor mps_ne_kernel(const Tensor& a, const Tensor& b);
Tensor mps_lt_kernel(const Tensor& a, const Tensor& b);
Tensor mps_le_kernel(const Tensor& a, const Tensor& b);
Tensor mps_ge_kernel(const Tensor& a, const Tensor& b);
Tensor mps_relu_backward_kernel(const Tensor& grad, const Tensor& input);
Tensor mps_sigmoid_backward_kernel(const Tensor& grad, const Tensor& sigmoid_out);
Tensor mps_tanh_backward_kernel(const Tensor& grad, const Tensor& tanh_out);
Tensor mps_add_inplace_kernel(Tensor& a, const Tensor& b);
Tensor mps_sub_inplace_kernel(Tensor& a, const Tensor& b);
Tensor mps_mul_inplace_kernel(Tensor& a, const Tensor& b);
Tensor mps_div_inplace_kernel(Tensor& a, const Tensor& b);
Tensor& mps_relu_inplace_kernel(Tensor& a);
Tensor& mps_sigmoid_inplace_kernel(Tensor& a);
Tensor& mps_tanh_inplace_kernel(Tensor& a);
Tensor& mps_gelu_inplace_kernel(Tensor& a);
Tensor& mps_leaky_relu_inplace_kernel(Tensor& a, float negative_slope);
Tensor mps_cast_kernel(const Tensor& input, DType target_dtype);
std::vector<Tensor> mps_fused_sgd_step(const Tensor& param, const Tensor& grad,
                                         const Tensor& momentum_buf,
                                         float lr, float momentum, float weight_decay);
std::vector<Tensor> mps_fused_adam_step(const Tensor& param, const Tensor& grad,
                                         const Tensor& exp_avg, const Tensor& exp_avg_sq,
                                         float lr, float beta1, float beta2,
                                         float eps, float bc1, float bc2,
                                         float weight_decay);

// ============================================================================
// Forward declarations — Native misc ops (mps_misc_ops.mm)
// ============================================================================
Tensor mps_frac_kernel(const Tensor& input);
Tensor mps_heaviside_kernel(const Tensor& input, const Tensor& values);
Tensor mps_nan_to_num_kernel(const Tensor& input, double nan_val, double posinf_val, double neginf_val);
Tensor mps_log_sigmoid_kernel(const Tensor& input);
Tensor mps_log_sigmoid_backward_kernel(const Tensor& grad, const Tensor& input);
Tensor mps_rrelu_kernel(const Tensor& input, float lower, float upper, bool training);
Tensor mps_rrelu_backward_kernel(const Tensor& grad, const Tensor& input, float lower, float upper);
Tensor mps_bitwise_and_kernel(const Tensor& a, const Tensor& b);
Tensor mps_bitwise_or_kernel(const Tensor& a, const Tensor& b);
Tensor mps_bitwise_xor_kernel(const Tensor& a, const Tensor& b);
Tensor mps_bitwise_not_kernel(const Tensor& input);
Tensor mps_bitwise_left_shift_kernel(const Tensor& a, const Tensor& b);
Tensor mps_bitwise_right_shift_kernel(const Tensor& a, const Tensor& b);
Tensor mps_count_nonzero_kernel(const Tensor& input, int64_t dim);
Tensor mps_nansum_kernel(const Tensor& input, int64_t dim, bool keepdim);
Tensor mps_nanmean_kernel(const Tensor& input, int64_t dim, bool keepdim);
std::pair<Tensor, Tensor> mps_aminmax_kernel(const Tensor& input, int64_t dim, bool keepdim);
Tensor mps_var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction);
Tensor mps_std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction);
Tensor mps_norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim);
Tensor mps_lerp_kernel(const Tensor& a, const Tensor& b, const Tensor& weight);
Tensor mps_trace_kernel(const Tensor& input);
Tensor mps_fill_kernel(const Tensor& input, float value);
Tensor mps_diag_kernel(const Tensor& input, int64_t diagonal);
Tensor mps_tril_kernel(const Tensor& input, int64_t diagonal);
Tensor mps_triu_kernel(const Tensor& input, int64_t diagonal);
Tensor mps_cumsum_kernel(const Tensor& input, int64_t dim);
Tensor mps_cumprod_kernel(const Tensor& input, int64_t dim);
Tensor mps_cross_kernel(const Tensor& a, const Tensor& b);
Tensor mps_polygamma_kernel(const Tensor& input, int64_t order);
Tensor mps_leaky_relu_kernel(const Tensor& input, float neg_slope);
Tensor mps_leaky_relu_backward_kernel(const Tensor& grad, const Tensor& input, float neg_slope);
Tensor mps_elu_kernel(const Tensor& input, float alpha);
Tensor mps_elu_backward_kernel(const Tensor& grad, const Tensor& input, float alpha);
Tensor mps_softplus_kernel(const Tensor& input, float beta, float threshold);
Tensor mps_softplus_backward_kernel(const Tensor& grad, const Tensor& input, float beta, float threshold);
Tensor mps_clamp_min_kernel(const Tensor& input, float min_val);
Tensor mps_clamp_max_kernel(const Tensor& input, float max_val);
Tensor mps_log_softmax_kernel(const Tensor& input, int64_t dim);
// Creation ops
Tensor mps_zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device);
Tensor mps_ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device);
Tensor mps_full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device);
Tensor mps_eye_kernel(int64_t n, DType dtype, Device device);
Tensor mps_arange_kernel(float start, float end, float step, DType dtype, Device device);
Tensor mps_linspace_kernel(float start, float end, int64_t steps, DType dtype, Device device);
Tensor mps_rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device);
Tensor mps_randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device);
Tensor mps_randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape, DType dtype, Device device);
Tensor mps_bernoulli_kernel(const Tensor& probs);
Tensor mps_one_hot_kernel(const Tensor& indices, int64_t num_classes);

// ============================================================================
// Forward declarations — Native indexing ops (mps_indexing.mm)
// ============================================================================
Tensor mps_cat_kernel(const std::vector<Tensor>& inputs, int64_t dim);
Tensor mps_stack_kernel(const std::vector<Tensor>& inputs, int64_t dim);
std::vector<Tensor> mps_split_kernel(const Tensor& input, int64_t split_size, int64_t dim);
std::vector<Tensor> mps_chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim);
Tensor mps_index_select_kernel(const Tensor& input, int64_t dim, const Tensor& indices);
Tensor mps_gather_kernel(const Tensor& input, int64_t dim, const Tensor& indices);
Tensor mps_scatter_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& src);
Tensor mps_scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& src);
Tensor mps_index_add_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& source);
Tensor mps_index_copy_kernel(const Tensor& input, int64_t dim, const Tensor& indices, const Tensor& source);
Tensor mps_index_fill_kernel(const Tensor& input, int64_t dim, const Tensor& indices, double value);
Tensor mps_masked_fill_kernel(const Tensor& input, const Tensor& mask, float value);
Tensor mps_take_kernel(const Tensor& input, const Tensor& indices);
Tensor mps_put_kernel(const Tensor& input, const Tensor& indices, const Tensor& source);
Tensor mps_flip_kernel(const Tensor& input, const std::vector<int64_t>& dims);
Tensor mps_roll_kernel(const Tensor& input, int64_t shift, int64_t dim);
Tensor mps_repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats);
Tensor mps_slice_kernel(const Tensor& input, int64_t dim, int64_t start, int64_t end, int64_t step);
Tensor mps_nonzero_kernel(const Tensor& input);
Tensor mps_searchsorted_kernel(const Tensor& sorted, const Tensor& values, bool right);
Tensor mps_bucketize_kernel(const Tensor& boundaries, const Tensor& input, bool right);

// ============================================================================
// Forward declarations — Native extended ops (mps_extended_ops.mm)
// ============================================================================
// S.11: per-axis dilation_d/h/w added to Conv3d / MaxPool3d / AvgPool3d.
// Previously the dispatch dropped DilationD/H/W on the floor and the
// underlying MPSGraph descriptors used the default value of 1 — dilated
// 3D convs / pools on Apple Silicon silently produced wrong outputs.
Tensor mps_conv3d_forward_kernel(const Tensor& input, const Tensor& weight,
                                  int64_t sd, int64_t sh, int64_t sw,
                                  int64_t pd, int64_t ph, int64_t pw,
                                  int64_t dd, int64_t dh, int64_t dw,
                                  int64_t groups);
Tensor mps_maxpool3d_forward_kernel(const Tensor& input, int64_t kd, int64_t kh, int64_t kw,
                                     int64_t sd, int64_t sh, int64_t sw,
                                     int64_t pd, int64_t ph, int64_t pw,
                                     int64_t dd, int64_t dh, int64_t dw,
                                     Tensor& indices);
Tensor mps_avgpool3d_forward_kernel(const Tensor& input, int64_t kd, int64_t kh, int64_t kw,
                                     int64_t sd, int64_t sh, int64_t sw,
                                     int64_t pd, int64_t ph, int64_t pw,
                                     int64_t dd, int64_t dh, int64_t dw,
                                     bool count_include_pad);
Tensor mps_cdist_kernel(const Tensor& x1, const Tensor& x2, float p);
std::vector<Tensor> mps_sort_kernel(const Tensor& input, int64_t dim, bool descending);
Tensor mps_argsort_kernel(const Tensor& input, int64_t dim, bool descending);
std::vector<Tensor> mps_topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest);
std::vector<Tensor> mps_median_kernel(const Tensor& input, int64_t dim);
std::vector<Tensor> mps_mode_kernel(const Tensor& input, int64_t dim);
std::vector<Tensor> mps_unique_kernel(const Tensor& input, bool sorted, bool return_inverse, bool return_counts);
Tensor mps_grid_sample_kernel(const Tensor& input, const Tensor& grid,
                               const std::string& mode, const std::string& padding_mode,
                               bool align_corners);
Tensor mps_interpolate_kernel(const Tensor& input, int64_t out_h, int64_t out_w,
                               const std::string& mode, bool align_corners);
Tensor mps_box_iou_kernel(const Tensor& boxes1, const Tensor& boxes2, int iou_type);
Tensor mps_nms_kernel(const Tensor& boxes, const Tensor& scores, float iou_threshold);
Tensor mps_roi_align_forward_kernel(const Tensor& features, const Tensor& rois,
                                    int64_t output_h, int64_t output_w,
                                    float spatial_scale, int64_t sampling_ratio,
                                    bool aligned);
Tensor mps_roi_align_backward_kernel(const Tensor& grad_output, const Tensor& rois,
                                     int64_t batch_size, int64_t feat_height, int64_t feat_width,
                                     float spatial_scale, int64_t sampling_ratio,
                                     bool aligned);
Tensor mps_max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices,
                                       int64_t out_h, int64_t out_w);
Tensor mps_max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices,
                                       int64_t out_d, int64_t out_h, int64_t out_w);
std::vector<Tensor> mps_batchnorm_mean_var_kernel(const Tensor& input);
Tensor mps_batchnorm_forward_training_kernel(const Tensor& input, const Tensor& mean,
                                               const Tensor& var, const Tensor& weight,
                                               const Tensor& bias, float eps);
std::vector<Tensor> mps_fused_adadelta_step(const Tensor& param, const Tensor& grad,
                                              const Tensor& accum, const Tensor& delta_accum,
                                              float lr, float rho, float eps, float wd);
std::vector<Tensor> mps_fused_adagrad_step(const Tensor& param, const Tensor& grad,
                                             const Tensor& sum_sq,
                                             float lr, float lr_decay, float eps,
                                             float wd, float step);
std::vector<Tensor> mps_fused_rmsprop_step(const Tensor& param, const Tensor& grad,
                                             const Tensor& sq_avg,
                                             float lr, float alpha, float eps, float wd);
std::vector<Tensor> mps_fused_adam_atan2_step(const Tensor& param, const Tensor& grad,
                                                const Tensor& exp_avg, const Tensor& exp_avg_sq,
                                                float lr, float beta1, float beta2,
                                                float eps, float bc1, float bc2, float wd);
std::vector<Tensor> mps_fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets,
                                                           bool compute_grad, const std::string& reduction);
std::vector<Tensor> mps_dropout_kernel(const Tensor& input, float p);

auto register_mps_kernels(BackendDispatchTable& table) -> void {
    // ================================================================
    // Tier 1: Arithmetic operations
    // ================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Add, mps_add_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Sub, mps_sub_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Mul, mps_mul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Div, mps_div_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, MatMul, mps_matmul_kernel);

    // ================================================================
    // Tier 1: Activation functions
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, ReLU, mps_relu_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sigmoid, mps_sigmoid_kernel);

    // ================================================================
    // Tier 1: Element-wise math
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Neg, mps_neg_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Exp, mps_exp_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log, mps_log_kernel);

    // ================================================================
    // Tier 1: Linear (matmul + bias add)
    // ================================================================
    table.register_single_output_kernel(OpId::Linear,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            if (inputs.size() >= 3) {
                return mps_linear_kernel(inputs[0], inputs[1], inputs[2]);
            }
            return mps_matmul_kernel(inputs[0], inputs[1]);
        });

    // ================================================================
    // Tier 1: Embedding
    // ================================================================
    table.register_kernel(OpId::Embedding,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {mps_embedding_kernel(inputs[0], inputs[1])};
        });

    // ================================================================
    // Tier 1: Softmax
    // ================================================================
    table.register_kernel(OpId::Softmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {mps_softmax_kernel(inputs[0], dim)};
        });

    // ================================================================
    // Tier 1: BatchNorm (inference path)
    // ================================================================
    table.register_kernel(OpId::BatchNorm2dForwardAffine,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return {mps_batch_norm_kernel(inputs[0], inputs[1], inputs[2],
                                          inputs[3], inputs[4], eps)};
        });

    // ================================================================
    // Tier 1: LayerNorm
    // ================================================================
    table.register_kernel(OpId::FusedLayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Per docs/internals/attention-contract.md the contract is
            // {output, mean, inv_std}. The kernel computes both on device;
            // return them so the autograd backward can read the saved stats
            // without SEGVing on empty placeholders (audit C5).
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [output, mean, inv_std] = mps_layer_norm_kernel_with_stats(
                inputs[0], inputs[1], inputs[2], eps);
            return {output, mean, inv_std};
        });

    // ================================================================
    // Tier 1: Conv2d
    // ================================================================
    table.register_kernel(OpId::Conv2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Per-axis keys win when present; scalar keys are the fallback so
            // callers that set only the scalar variant still work end-to-end
            // (mirrors the CPU registry pattern). Audit Q.9: previously
            // hard-coded dilation=1 silently ignored any dilated convolution.
            int64_t s  = attrs.get_int(AttrKey::Stride, 1);
            int64_t p  = attrs.get_int(AttrKey::Padding, 0);
            int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
            int64_t sh = attrs.has(AttrKey::StrideH)   ? attrs.get_int(AttrKey::StrideH)   : s;
            int64_t sw = attrs.has(AttrKey::StrideW)   ? attrs.get_int(AttrKey::StrideW)   : s;
            int64_t ph = attrs.has(AttrKey::PaddingH)  ? attrs.get_int(AttrKey::PaddingH)  : p;
            int64_t pw = attrs.has(AttrKey::PaddingW)  ? attrs.get_int(AttrKey::PaddingW)  : p;
            int64_t dh = attrs.has(AttrKey::DilationH) ? attrs.get_int(AttrKey::DilationH) : d;
            int64_t dw = attrs.has(AttrKey::DilationW) ? attrs.get_int(AttrKey::DilationW) : d;
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            Tensor out = mps_conv2d_kernel(inputs[0], inputs[1], sh, sw, ph, pw, dh, dw, groups);
            // nn::Conv2d pushes bias as inputs[2] and relies on the kernel to add
            // it (mirrors CPU: `bias = inputs.size() > 2 ? &inputs[2] : nullptr`).
            // The MPSGraph conv omits bias, so add it here as a broadcast over the
            // channel axis: bias{out_c} -> {1,out_c,1,1} + out{N,out_c,H,W}.
            if (inputs.size() > 2 && inputs[2].numel() > 0) {
                int64_t out_c = out.shape()[1];
                Tensor bias_bc = mps_reshape_kernel(inputs[2], {1, out_c, 1, 1});
                out = mps_add_kernel(out, bias_bc);
            }
            return {out};
        });

    // ================================================================
    // Tier 2: CPU-roundtrip fallbacks for training support
    // ================================================================
    // These enable backward pass and optimizer steps on MPS tensors
    // by routing through CPU. Native Metal shaders can replace these
    // incrementally for better performance.

    // Reductions — native Metal kernels (no CPU roundtrip for contiguous last-dim reductions)
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{mps_sum_kernel(inputs[0], dim, keepdim)};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{mps_mean_kernel(inputs[0], dim, keepdim)};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        // AUTOGRAD-R048: OpId::Max returns {values} only on every other
        // backend (CPU/CUDA/ROCm/OneAPI/Vulkan); mps_max_kernel computes
        // indices as an intrinsic byproduct of the reduction (kept, since
        // dropping it would need a separate kernel path), but the extra
        // element must not leak into this OpId's return contract.
        Tensor indices;
        auto values = mps_max_kernel(inputs[0], dim, keepdim, indices);
        return std::vector<Tensor>{values};
    });

    // Shape operations — zero-copy metadata ops (no GPU→CPU→GPU roundtrip)
    table.register_single_output_kernel(OpId::Reshape,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            return mps_reshape_kernel(inputs[0], shape);
        });

    table.register_single_output_kernel(OpId::Transpose,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
            return mps_transpose_kernel(inputs[0], dim0, dim1);
        });

    // Phase 3.2: native Metal unary / binary kernels (previously CPU
    // fallbacks) for Tanh/Sqrt/Abs and hand-coded CPU lambdas for
    // Pow/Clamp.
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Tanh, mps_tanh_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sqrt, mps_sqrt_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Abs,  mps_abs_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Pow, mps_pow_kernel);

    // ================================================================
    // Phase 5: Additional element-wise math ops
    // ================================================================
    // Unary ops
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log2, mps_log2_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log10, mps_log10_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log1p, mps_log1p_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Exp2, mps_exp2_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Expm1, mps_expm1_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Erf, mps_erf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Erfc, mps_erfc_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsNan, mps_isnan_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsInf, mps_isinf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsFinite, mps_isfinite_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Rsqrt, mps_rsqrt_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Square, mps_square_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Reciprocal, mps_reciprocal_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Deg2Rad, mps_deg2rad_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Rad2Deg, mps_rad2deg_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Logit, mps_logit_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Signbit, mps_signbit_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsReal, mps_isreal_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsPosInf, mps_isposinf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsNegInf, mps_isneginf_kernel);
    // Binary ops
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Atan2, mps_atan2_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Fmod, mps_fmod_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Remainder, mps_remainder_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Copysign, mps_copysign_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Nextafter, mps_nextafter_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, FloatPower, mps_float_power_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Xlog1py, mps_xlog1py_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Ldexp, mps_ldexp_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Hypot, mps_hypot_kernel);

    // Clamp needs scalar min/max plumbed through OpAttributes, so it
    // can't use the unary register macro directly.
    table.register_single_output_kernel(OpId::Clamp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float min_val = static_cast<float>(
                attrs.get_float(AttrKey::Min, std::numeric_limits<double>::lowest()));
            float max_val = static_cast<float>(
                attrs.get_float(AttrKey::Max, std::numeric_limits<double>::max()));
            return mps_clamp_kernel(inputs[0], min_val, max_val);
        });

    // Permute, Squeeze, Unsqueeze, Flatten, Expand — zero-copy metadata ops
    table.register_single_output_kernel(OpId::Permute,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto dims = attrs.get_int_list(AttrKey::Dims);
            return mps_permute_kernel(inputs[0], dims);
        });

    table.register_single_output_kernel(OpId::Squeeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Use a sentinel that cannot collide with a real (possibly negative)
            // axis so that squeeze(-1)/squeeze(-2) are NOT mistaken for
            // squeeze-all. Mirrors the CPU backend's contract.
            int64_t dim = attrs.get_int(AttrKey::Dim, kSqueezeAllDims);
            return mps_squeeze_kernel(inputs[0], dim);
        });

    table.register_single_output_kernel(OpId::Unsqueeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_unsqueeze_kernel(inputs[0], dim);
        });

    table.register_single_output_kernel(OpId::Flatten,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
            int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
            return mps_flatten_kernel(inputs[0], start_dim, end_dim);
        });

    table.register_single_output_kernel(OpId::Expand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            return mps_expand_kernel(inputs[0], shape);
        });

    // Comparison ops — native Metal kernels with dedicated Bool-output dispatcher
    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_gt_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_eq_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_ne_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_lt_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_le_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_ge_kernel(inputs[0], inputs[1])};
    });

    // Backward activation kernels — native Metal
    table.register_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_relu_backward_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_sigmoid_backward_kernel(inputs[0], inputs[1])};
    });
    table.register_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{mps_tanh_backward_kernel(inputs[0], inputs[1])};
    });

    // In-place arithmetic — native Metal.
    // These MUST be registered in the inplace_kernels[] array (via
    // register_inplace_kernel), because ops::add_/sub_/mul_/div_ route through
    // BackendDispatchTable::dispatch_inplace(), which only reads that array.
    // Registering via register_kernel (multi-output kernels[]) leaves the
    // inplace array empty and makes every in-place arithmetic op throw
    // "not supported" on MPS. Signature: Tensor&(Tensor&, span, attrs).
    table.register_inplace_kernel(OpId::AddInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            mps_add_inplace_kernel(target, others[0]);  // writes into shared storage
            return target;
        });
    table.register_inplace_kernel(OpId::SubInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            mps_sub_inplace_kernel(target, others[0]);
            return target;
        });
    table.register_inplace_kernel(OpId::MulInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            mps_mul_inplace_kernel(target, others[0]);
            return target;
        });
    table.register_inplace_kernel(OpId::DivInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            mps_div_inplace_kernel(target, others[0]);
            return target;
        });

    // In-place activations — also routed through dispatch_inplace(), so they
    // MUST live in inplace_kernels[]. Previously registered via
    // mps_accelerate_single (single_output_kernels[]), which made x.relu_()
    // etc. throw "not supported" on MPS. Native Metal, on-device write-back.
    table.register_inplace_kernel(OpId::ReLUInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            return mps_relu_inplace_kernel(target);
        });
    table.register_inplace_kernel(OpId::SigmoidInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            return mps_sigmoid_inplace_kernel(target);
        });
    table.register_inplace_kernel(OpId::TanhInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            return mps_tanh_inplace_kernel(target);
        });
    table.register_inplace_kernel(OpId::GeluInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            return mps_gelu_inplace_kernel(target);
        });
    table.register_inplace_kernel(OpId::LeakyReLUInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            return mps_leaky_relu_inplace_kernel(target, alpha);
        });

    // Note: zeros_like / ones_like are library-level free functions in
    // tenzor::ops, not dispatch-level OpIds. Autograd code that needs
    // gradient-init scratch tensors should call tenzor::zeros_like(x) /
    // tenzor::ones_like(x), which internally dispatches zeros()/ones()
    // for the tensor's device. No MPS-specific registration is needed.

    // Fused optimizer steps — native Metal kernels (no CPU roundtrip)
    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        return mps_fused_sgd_step(inputs[0], inputs[1], inputs[2], lr, momentum, wd);
    });

    table.register_kernel(OpId::FusedAdamStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.001));
        float beta1 = static_cast<float>(attrs.get_float(AttrKey::Beta1, 0.9));
        float beta2 = static_cast<float>(attrs.get_float(AttrKey::Beta2, 0.999));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
        int64_t step = attrs.get_int(AttrKey::Step, 1);
        float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        // Compute bias corrections from step count
        float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
        float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
        return mps_fused_adam_step(inputs[0], inputs[1], inputs[2], inputs[3],
                                    lr, beta1, beta2, eps, bc1, bc2, wd);
    });

    // Cast — native Metal for common pairs, CPU roundtrip for exotic dtypes
    table.register_single_output_kernel(OpId::Cast,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto target = static_cast<DType>(attrs.get_int(AttrKey::DType, 0));
            return mps_cast_kernel(inputs[0], target);
        });

    // ================================================================
    // Native Phase 4 ops — element-wise, bitwise, reductions
    // ================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Frac, mps_frac_kernel);

    table.register_single_output_kernel(OpId::Heaviside,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_heaviside_kernel(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::NanToNum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double nan_val = attrs.get_float(AttrKey::NanValue, 0.0);
            double posinf = attrs.get_float(AttrKey::PosInfValue, 1e38);
            double neginf = attrs.get_float(AttrKey::NegInfValue, -1e38);
            return mps_nan_to_num_kernel(inputs[0], nan_val, posinf, neginf);
        });

    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, LogSigmoid, mps_log_sigmoid_kernel);

    table.register_single_output_kernel(OpId::LogSigmoidBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_log_sigmoid_backward_kernel(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::RReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
            float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
            bool training = attrs.get_bool(AttrKey::Training, false);
            return mps_rrelu_kernel(inputs[0], lower, upper, training);
        });

    table.register_single_output_kernel(OpId::RReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
            float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
            return mps_rrelu_backward_kernel(inputs[0], inputs[1], lower, upper);
        });

    // Bitwise ops
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseAnd, mps_bitwise_and_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseOr, mps_bitwise_or_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseXor, mps_bitwise_xor_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BitwiseNot, mps_bitwise_not_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseLeftShift, mps_bitwise_left_shift_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseRightShift, mps_bitwise_right_shift_kernel);

    // Reduction ops
    table.register_single_output_kernel(OpId::CountNonzero,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return mps_count_nonzero_kernel(inputs[0], dim);
        });

    table.register_single_output_kernel(OpId::Nansum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return mps_nansum_kernel(inputs[0], dim, keepdim);
        });

    table.register_single_output_kernel(OpId::Nanmean,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return mps_nanmean_kernel(inputs[0], dim, keepdim);
        });

    table.register_kernel(OpId::Aminmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            auto [mn, mx] = mps_aminmax_kernel(inputs[0], dim, keepdim);
            return {mn, mx};
        });

    table.register_single_output_kernel(OpId::Var,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            int64_t correction = attrs.get_int(AttrKey::Correction, 1);
            return mps_var_kernel(inputs[0], dim, keepdim, correction);
        });

    table.register_single_output_kernel(OpId::Std,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            int64_t correction = attrs.get_int(AttrKey::Correction, 1);
            return mps_std_kernel(inputs[0], dim, keepdim, correction);
        });

    table.register_single_output_kernel(OpId::Norm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // ops/reduction.cpp's norm() sets AttrKey::P (AttrKey::Order is
            // set only by Polygamma, math.cpp:690, never by Norm); reading
            // Order here always fell through to the 2.0 default, silently
            // ignoring any requested p != 2.
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return mps_norm_kernel(inputs[0], p, dim, keepdim);
        });

    // Scatter variants: IndexAdd, IndexCopy, IndexFill
    table.register_single_output_kernel(OpId::IndexAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_index_add_kernel(inputs[0], dim, inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::IndexCopy,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_index_copy_kernel(inputs[0], dim, inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::IndexFill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            double value = attrs.get_float(AttrKey::Value, 0.0);
            return mps_index_fill_kernel(inputs[0], dim, inputs[1], value);
        });

    // ================================================================
    // Native element-wise ops (previously in Tier 3 roundtrip)
    // ================================================================
    table.register_single_output_kernel(OpId::LeakyReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float ns = static_cast<float>(attrs.get_float(AttrKey::Negative_slope, 0.01));
            return mps_leaky_relu_kernel(inputs[0], ns);
        });
    table.register_single_output_kernel(OpId::LeakyReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float ns = static_cast<float>(attrs.get_float(AttrKey::Negative_slope, 0.01));
            return mps_leaky_relu_backward_kernel(inputs[0], inputs[1], ns);
        });
    table.register_single_output_kernel(OpId::Elu,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return mps_elu_kernel(inputs[0], alpha);
        });
    table.register_single_output_kernel(OpId::EluBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return mps_elu_backward_kernel(inputs[0], inputs[1], alpha);
        });
    table.register_single_output_kernel(OpId::Softplus,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
            float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
            return mps_softplus_kernel(inputs[0], beta, threshold);
        });
    table.register_single_output_kernel(OpId::SoftplusBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
            float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
            return mps_softplus_backward_kernel(inputs[0], inputs[1], beta, threshold);
        });
    table.register_single_output_kernel(OpId::ClampMin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float val = static_cast<float>(attrs.get_float(AttrKey::Min, 0.0));
            return mps_clamp_min_kernel(inputs[0], val);
        });
    table.register_single_output_kernel(OpId::ClampMax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float val = static_cast<float>(attrs.get_float(AttrKey::Max, 0.0));
            return mps_clamp_max_kernel(inputs[0], val);
        });
    table.register_single_output_kernel(OpId::LogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return mps_log_softmax_kernel(inputs[0], dim);
        });
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Trace, mps_trace_kernel);
    table.register_single_output_kernel(OpId::Fill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            return mps_fill_kernel(inputs[0], value);
        });
    table.register_single_output_kernel(OpId::Diag,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t diag = attrs.get_int(AttrKey::Diagonal, 0);
            return mps_diag_kernel(inputs[0], diag);
        });
    table.register_single_output_kernel(OpId::Tril,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t diag = attrs.get_int(AttrKey::Diagonal, 0);
            return mps_tril_kernel(inputs[0], diag);
        });
    table.register_single_output_kernel(OpId::Triu,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t diag = attrs.get_int(AttrKey::Diagonal, 0);
            return mps_triu_kernel(inputs[0], diag);
        });
    table.register_single_output_kernel(OpId::CumSum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_cumsum_kernel(inputs[0], dim);
        });
    table.register_single_output_kernel(OpId::CumProd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_cumprod_kernel(inputs[0], dim);
        });
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Cross, mps_cross_kernel);
    table.register_single_output_kernel(OpId::Polygamma,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // AttrKey::Order is stored as a double by the producer
            // (src/ops/math.cpp), matching every other backend. Reading it
            // with get_int would reinterpret the double's bit pattern as an
            // int64 (n=1 -> 4607182418800017408), so read it as a float.
            int64_t order = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
            return mps_polygamma_kernel(inputs[0], order);
        });

    // ================================================================
    // Native indexing / manipulation ops
    // ================================================================
    table.register_single_output_kernel(OpId::Cat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            std::vector<Tensor> tensors(inputs.begin(), inputs.end());
            return mps_cat_kernel(tensors, dim);
        });
    table.register_single_output_kernel(OpId::Stack,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            std::vector<Tensor> tensors(inputs.begin(), inputs.end());
            return mps_stack_kernel(tensors, dim);
        });
    table.register_kernel(OpId::Split,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t split_size = attrs.get_int(AttrKey::SplitSize, 1);
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_split_kernel(inputs[0], split_size, dim);
        });
    table.register_kernel(OpId::Chunk,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t chunks = attrs.get_int(AttrKey::Chunks, 1);
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_chunk_kernel(inputs[0], chunks, dim);
        });
    table.register_single_output_kernel(OpId::IndexSelect,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_index_select_kernel(inputs[0], dim, inputs[1]);
        });
    table.register_single_output_kernel(OpId::Gather,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_gather_kernel(inputs[0], dim, inputs[1]);
        });
    table.register_single_output_kernel(OpId::Scatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_scatter_kernel(inputs[0], dim, inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::ScatterAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_scatter_add_kernel(inputs[0], dim, inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::Slice,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // AUTOGRAD-R046: support both the multi-dim (Starts/Ends/Steps
            // lists) and single-dim (Dim/Start/End/Step scalars) attribute
            // formats, matching CPU/CUDA. mps_slice_kernel is natively
            // single-dim only, so the list format is composed by applying
            // it once per dimension (mirrors cpu::slice_multi_kernel's
            // composition exactly, including its short-list end/step
            // defaulting).
            if (attrs.has(AttrKey::Starts)) {
                auto starts = attrs.get_int_list(AttrKey::Starts);
                auto ends = attrs.get_int_list(AttrKey::Ends);
                auto steps = attrs.get_int_list(AttrKey::Steps);
                Tensor result = inputs[0];
                for (size_t d = 0; d < starts.size(); ++d) {
                    const int64_t end = (d < ends.size()) ? ends[d] : result.shape()[static_cast<int64_t>(d)];
                    const int64_t step = (d < steps.size()) ? steps[d] : 1;
                    result = mps_slice_kernel(result, static_cast<int64_t>(d), starts[d], end, step);
                }
                return result;
            }
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t start = attrs.get_int(AttrKey::Start, 0);
            int64_t end = attrs.get_int(AttrKey::End, -1);
            int64_t step = attrs.get_int(AttrKey::Step, 1);
            return mps_slice_kernel(inputs[0], dim, start, end, step);
        });
    table.register_single_output_kernel(OpId::MaskedFill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            return mps_masked_fill_kernel(inputs[0], inputs[1], value);
        });
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Take, mps_take_kernel);
    table.register_single_output_kernel(OpId::Put,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_put_kernel(inputs[0], inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::Flip,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto dims = attrs.get_int_list(AttrKey::Dims);
            if (dims.empty()) dims = {0};
            return mps_flip_kernel(inputs[0], dims);
        });
    table.register_single_output_kernel(OpId::Roll,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t shift = attrs.get_int(AttrKey::Shift, 0);
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return mps_roll_kernel(inputs[0], shift, dim);
        });
    table.register_single_output_kernel(OpId::Repeat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto repeats = attrs.get_int_list(AttrKey::Repeats);
            return mps_repeat_kernel(inputs[0], repeats);
        });
    table.register_single_output_kernel(OpId::Tile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto reps = attrs.get_int_list(AttrKey::Reps);
            return mps_repeat_kernel(inputs[0], reps);
        });
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Nonzero, mps_nonzero_kernel);
    table.register_single_output_kernel(OpId::SearchSorted,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return mps_searchsorted_kernel(inputs[0], inputs[1], right);
        });
    table.register_single_output_kernel(OpId::Bucketize,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return mps_bucketize_kernel(inputs[0], inputs[1], right);
        });
    table.register_single_output_kernel(OpId::OneHot,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t nc = attrs.get_int(AttrKey::NumClasses, -1);
            return mps_one_hot_kernel(inputs[0], nc);
        });
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Bernoulli, mps_bernoulli_kernel);

    // ================================================================
    // Native Lerp
    // ================================================================
    table.register_kernel(OpId::Lerp,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {mps_lerp_kernel(inputs[0], inputs[1], inputs[2])};
        });

    // ================================================================
    // Native 3D ops: Conv3d, MaxPool3d, AvgPool3d
    // ================================================================
    table.register_kernel(OpId::Conv3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t sd = attrs.get_int(AttrKey::StrideD, 1);
            int64_t sh = attrs.get_int(AttrKey::StrideH, 1);
            int64_t sw = attrs.get_int(AttrKey::StrideW, 1);
            int64_t pd = attrs.get_int(AttrKey::PaddingD, 0);
            int64_t ph = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pw = attrs.get_int(AttrKey::PaddingW, 0);
            // S.11: per-axis dilation (was previously dropped — MPSGraph defaulted to 1).
            int64_t dd = attrs.get_int(AttrKey::DilationD, 1);
            int64_t dh = attrs.get_int(AttrKey::DilationH, 1);
            int64_t dw = attrs.get_int(AttrKey::DilationW, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            Tensor out = mps_conv3d_forward_kernel(inputs[0], inputs[1], sd, sh, sw, pd, ph, pw, dd, dh, dw, groups);
            // nn::Conv3d pushes bias as inputs[2] and relies on the kernel to add
            // it (mirrors CPU). The native conv omits bias, so add it here as a
            // broadcast over the channel axis:
            // bias{out_c} -> {1,out_c,1,1,1} + out{N,out_c,D,H,W}.
            if (inputs.size() > 2 && inputs[2].numel() > 0) {
                int64_t out_c = out.shape()[1];
                Tensor bias_bc = mps_reshape_kernel(inputs[2], {1, out_c, 1, 1, 1});
                out = mps_add_kernel(out, bias_bc);
            }
            return {out};
        });

    table.register_kernel(OpId::MaxPool3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Per-axis with scalar fallback: matches the canonical pattern in
            // include/tenzor/backend/attr_macros.hpp. The previous hard-coded
            // default of 2 silently picked kernel=2 when the dispatcher only
            // set scalar AttrKey::KernelSize.
            const int64_t k_scalar = attrs.get_int(AttrKey::KernelSize, 2);
            const int64_t s_scalar = attrs.get_int(AttrKey::Stride, k_scalar);
            const int64_t p_scalar = attrs.get_int(AttrKey::Padding, 0);
            const int64_t kd = attrs.get_int(AttrKey::KernelSizeD, k_scalar);
            const int64_t kh = attrs.get_int(AttrKey::KernelSizeH, k_scalar);
            const int64_t kw = attrs.get_int(AttrKey::KernelSizeW, k_scalar);
            const int64_t sd = attrs.get_int(AttrKey::StrideD, s_scalar);
            const int64_t sh = attrs.get_int(AttrKey::StrideH, s_scalar);
            const int64_t sw = attrs.get_int(AttrKey::StrideW, s_scalar);
            const int64_t pd = attrs.get_int(AttrKey::PaddingD, p_scalar);
            const int64_t ph = attrs.get_int(AttrKey::PaddingH, p_scalar);
            const int64_t pw = attrs.get_int(AttrKey::PaddingW, p_scalar);
            // S.11: per-axis dilation (PyTorch MaxPool3d supports dilation > 1).
            const int64_t d_scalar = attrs.get_int(AttrKey::Dilation, 1);
            const int64_t dd = attrs.get_int(AttrKey::DilationD, d_scalar);
            const int64_t dh = attrs.get_int(AttrKey::DilationH, d_scalar);
            const int64_t dw = attrs.get_int(AttrKey::DilationW, d_scalar);
            Tensor indices;
            auto output = mps_maxpool3d_forward_kernel(inputs[0], kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, indices);
            return {output, indices};
        });

    table.register_single_output_kernel(OpId::AvgPool3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const int64_t k_scalar = attrs.get_int(AttrKey::KernelSize, 2);
            const int64_t s_scalar = attrs.get_int(AttrKey::Stride, k_scalar);
            const int64_t p_scalar = attrs.get_int(AttrKey::Padding, 0);
            const int64_t kd = attrs.get_int(AttrKey::KernelSizeD, k_scalar);
            const int64_t kh = attrs.get_int(AttrKey::KernelSizeH, k_scalar);
            const int64_t kw = attrs.get_int(AttrKey::KernelSizeW, k_scalar);
            const int64_t sd = attrs.get_int(AttrKey::StrideD, s_scalar);
            const int64_t sh = attrs.get_int(AttrKey::StrideH, s_scalar);
            const int64_t sw = attrs.get_int(AttrKey::StrideW, s_scalar);
            const int64_t pd = attrs.get_int(AttrKey::PaddingD, p_scalar);
            const int64_t ph = attrs.get_int(AttrKey::PaddingH, p_scalar);
            const int64_t pw = attrs.get_int(AttrKey::PaddingW, p_scalar);
            // S.11: per-axis dilation. AvgPool dilation > 1 is rare but PyTorch
            // doesn't currently expose it for AvgPool3d; we honour the per-axis
            // attrs (default 1) for consistency with Conv3d / MaxPool3d so a
            // future PyTorch parity gap doesn't require another sweep.
            const int64_t d_scalar = attrs.get_int(AttrKey::Dilation, 1);
            const int64_t dd = attrs.get_int(AttrKey::DilationD, d_scalar);
            const int64_t dh = attrs.get_int(AttrKey::DilationH, d_scalar);
            const int64_t dw = attrs.get_int(AttrKey::DilationW, d_scalar);
            bool count_pad = attrs.get_bool(AttrKey::CountIncludePad, false);
            return mps_avgpool3d_forward_kernel(inputs[0], kd, kh, kw, sd, sh, sw, pd, ph, pw, dd, dh, dw, count_pad);
        });

    // Native CDist
    table.register_single_output_kernel(OpId::CDist,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
            return mps_cdist_kernel(inputs[0], inputs[1], p);
        });

    // ================================================================
    // Native Sort / ArgSort / TopK / Median / Mode / Unique
    // ================================================================
    table.register_kernel(OpId::Sort,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool desc = attrs.get_bool(AttrKey::Descending, false);
            return mps_sort_kernel(inputs[0], dim, desc);
        });
    table.register_single_output_kernel(OpId::ArgSort,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool desc = attrs.get_bool(AttrKey::Descending, false);
            return mps_argsort_kernel(inputs[0], dim, desc);
        });
    table.register_kernel(OpId::TopK,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t k = attrs.get_int(AttrKey::K, 1);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool largest = attrs.get_bool(AttrKey::Largest, true);
            return mps_topk_kernel(inputs[0], k, dim, largest);
        });
    table.register_kernel(OpId::Median,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return mps_median_kernel(inputs[0], dim);
        });
    table.register_kernel(OpId::Mode,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return mps_mode_kernel(inputs[0], dim);
        });
    table.register_kernel(OpId::Unique,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool sorted = attrs.get_bool(AttrKey::Sorted, true);
            bool ret_inv = attrs.get_bool(AttrKey::ReturnInverse, false);
            bool ret_cnt = attrs.get_bool(AttrKey::ReturnCounts, false);
            return mps_unique_kernel(inputs[0], sorted, ret_inv, ret_cnt);
        });

    // ================================================================
    // Native vision ops
    // ================================================================
    table.register_single_output_kernel(OpId::GridSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Honor Mode & PaddingMode like every other backend (CUDA:2781).
            // Previously MPS read only AlignCorners and hard-coded one sampling
            // mode + zero padding, so nearest/bicubic and border/reflection were
            // silently ignored.
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
            bool align = attrs.get_bool(AttrKey::AlignCorners, false);
            return mps_grid_sample_kernel(inputs[0], inputs[1], mode, padding_mode, align);
        });
    table.register_single_output_kernel(OpId::Interpolate,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // The eager op serializes the target size into AttrKey::OutputSize
            // (int list) and the interpolation mode into AttrKey::Mode — it never
            // sets the OutputSizeH/W scalar keys. Reading those (as before) gave
            // a (N,C,0,0) output and hard-coded "bilinear" ran nearest/bicubic as
            // bilinear. Read the list + mode like CUDA:2763.
            auto size = attrs.get_int_list(AttrKey::OutputSize);
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            bool align = attrs.get_bool(AttrKey::AlignCorners, false);
            if (size.size() != 2) {
                throw std::runtime_error(
                    "MPS Interpolate: only 4D interpolation (2 output sizes) is "
                    "implemented; got " + std::to_string(size.size()) +
                    " output dims (1D/5D interpolate unsupported on MPS)");
            }
            return mps_interpolate_kernel(inputs[0], size[0], size[1], mode, align);
        });
    table.register_single_output_kernel(OpId::BoxIoU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Honor IouType (0=IoU,1=GIoU,2=DIoU,3=CIoU) like CPU:2720. Was always
            // plain IoU.
            int iou_type = static_cast<int>(attrs.get_int(AttrKey::IouType, 0));
            return mps_box_iou_kernel(inputs[0], inputs[1], iou_type);
        });
    table.register_kernel(OpId::NMS,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // The dispatcher sets AttrKey::IouThreshold; MPS previously read
            // AttrKey::Threshold (a different key) and so always used the 0.5
            // default, ignoring the caller's iou_threshold. Read IouThreshold
            // like CPU:2726.
            float threshold = static_cast<float>(attrs.get_float(AttrKey::IouThreshold, 0.5));
            return {mps_nms_kernel(inputs[0], inputs[1], threshold)};
        });

    // ================================================================
    // Native creation ops
    // ================================================================
    table.register_kernel(OpId::Zeros,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_zeros_kernel(shape, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Ones,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_ones_kernel(shape, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Full,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_full_kernel(shape, value, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Eye,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t n = attrs.get_int(AttrKey::N, 1);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_eye_kernel(n, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Arange,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
            float end = static_cast<float>(attrs.get_float(AttrKey::End, 1.0));
            float step = static_cast<float>(attrs.get_float(AttrKey::Step, 1.0));
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_arange_kernel(start, end, step, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Linspace,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
            float end = static_cast<float>(attrs.get_float(AttrKey::End, 1.0));
            int64_t steps = attrs.get_int(AttrKey::Steps, 100);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_linspace_kernel(start, end, steps, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Rand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_rand_kernel(shape, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Randn,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_randn_kernel(shape, dtype, Device::mps())};
        });
    table.register_kernel(OpId::Randint,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t low = attrs.get_int(AttrKey::Start, 0);
            int64_t high = attrs.get_int(AttrKey::High, 10);
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto dtype = static_cast<DType>(attrs.get_int(AttrKey::Dtype, static_cast<int>(DType::Float32)));
            return {mps_randint_kernel(low, high, shape, dtype, Device::mps())};
        });

    // ================================================================
    // Native BatchNorm variants
    // ================================================================
    table.register_kernel(OpId::BatchNorm2dMeanVar,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return mps_batchnorm_mean_var_kernel(inputs[0]);
        });
    table.register_kernel(OpId::BatchNorm2dFusedTraining,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto mean_var = mps_batchnorm_mean_var_kernel(inputs[0]);
            auto output = mps_batchnorm_forward_training_kernel(inputs[0], mean_var[0], mean_var[1],
                                                                  inputs[1], inputs[2], eps);
            return {output, mean_var[0], mean_var[1]};
        });
    table.register_single_output_kernel(OpId::BatchNorm2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return mps_batchnorm_forward_training_kernel(inputs[0], inputs[1], inputs[2],
                                                          inputs[3], inputs[4], eps);
        });

    // ================================================================
    // Native fused optimizer steps
    // ================================================================
    table.register_kernel(OpId::FusedAdadeltaStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
            float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
            float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
            return mps_fused_adadelta_step(inputs[0], inputs[1], inputs[2], inputs[3], lr, rho, eps, wd);
        });
    table.register_kernel(OpId::FusedAdagradStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
            float lr_decay = static_cast<float>(attrs.get_float(AttrKey::LrDecay, 0.0));
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
            float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
            float step = static_cast<float>(attrs.get_int(AttrKey::Step, 1));
            return mps_fused_adagrad_step(inputs[0], inputs[1], inputs[2], lr, lr_decay, eps, wd, step);
        });
    table.register_kernel(OpId::FusedRMSPropStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.99));
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
            float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
            return mps_fused_rmsprop_step(inputs[0], inputs[1], inputs[2], lr, alpha, eps, wd);
        });
    table.register_kernel(OpId::FusedAdamAtan2Step,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.001));
            float beta1 = static_cast<float>(attrs.get_float(AttrKey::Beta1, 0.9));
            float beta2 = static_cast<float>(attrs.get_float(AttrKey::Beta2, 0.999));
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
            int64_t step = attrs.get_int(AttrKey::Step, 1);
            float wd = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
            float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
            float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));
            return mps_fused_adam_atan2_step(inputs[0], inputs[1], inputs[2], inputs[3],
                                              lr, beta1, beta2, eps, bc1, bc2, wd);
        });
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Honor Reduction & ComputeGrad like every other backend (CPU:
            // cpu_kernel_registry FusedSoftmaxCrossEntropy). Previously MPS
            // dropped both attrs, always returning per-sample {batch} loss and an
            // unscaled gradient.
            bool compute_grad = attrs.get_bool(AttrKey::ComputeGrad, true);
            std::string reduction = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
            return mps_fused_softmax_cross_entropy_kernel(inputs[0], inputs[1],
                                                          compute_grad, reduction);
        });

    // ================================================================
    // Native Dropout
    // ================================================================
    table.register_kernel(OpId::Dropout,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
            return mps_dropout_kernel(inputs[0], p);
        });
    table.register_single_output_kernel(OpId::DropoutBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // grad_in = grad * mask * scale — reuse the mask from forward.
            // inputs[0] = grad, inputs[1] = mask.
            // Compute entirely with the native element-wise Mul kernel so that
            // every dtype (Float16/BFloat16/Float32/Float64/...) and any
            // non-contiguous / broadcast layout is handled correctly. A raw
            // float* loop here would read half words as 32-bit floats (garbage
            // and OOB for Float16/BFloat16), drop precision for Float64, and
            // ignore strides for non-contiguous grad/mask.
            double p = attrs.get_float(AttrKey::P, 0.5);
            double scale = 1.0 / (1.0 - p);
            // grad * mask (native, dtype/stride/broadcast correct)
            Tensor gm = mps_mul_kernel(inputs[0], inputs[1]);
            // * scale: scalar tensor of the same dtype/device, broadcast-mul.
            Tensor scale_t = tenzor::full({1}, scale, gm.dtype(), gm.device());
            return mps_mul_kernel(gm, scale_t);
        });

    // ================================================================
    // Accelerate-based ops (FFT, Linalg, Sparse) — use vDSP/LAPACK
    // via shared memory (zero-copy on Apple Silicon, no GPU roundtrip)
    // ================================================================

    // Helper: dispatch to CPU but use shared memory (MPS unified memory
    // means .to(cpu) and .to(mps) are effectively free on Apple Silicon)
    // IMPORTANT: these Accelerate (CPU round-trip) fallbacks must NEVER
    // clobber a native Metal kernel that was already registered earlier in
    // this same function. The alphabetical Phase 8.6 catch-all block below
    // names many OpIds that already have native GPU kernels (MatMul, Mul,
    // Div, Sub, Neg, Log, Sigmoid, Tanh, Pow, Cross, Take, bitwise ops, …).
    // Registration runs top-to-bottom and the later write wins, so without
    // this guard the native kernels would be silently replaced by a
    // .to(cpu)->dispatch->.to(mps) round-trip — a CPU fallback for a GPU
    // backend. Skip registration whenever any kernel already exists.
    auto mps_accelerate_single = [&](OpId op) {
        if (table.has_kernel(op)) return;  // keep native Metal kernel
        table.register_single_output_kernel(op,
            [op](std::span<const Tensor> inputs,
                 const OpAttributes& attrs) -> Tensor {
                auto dev = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                auto cpu_result = dispatch(op, cpu_inputs, attrs);
                return cpu_result[0].to(dev);
            });
    };
    auto mps_accelerate_multi = [&](OpId op) {
        if (table.has_kernel(op)) return;  // keep native Metal kernel
        table.register_kernel(op, [op](std::span<const Tensor> inputs,
                                        const OpAttributes& attrs) {
            auto dev = inputs[0].device();
            std::vector<Tensor> cpu_inputs;
            cpu_inputs.reserve(inputs.size());
            for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
            auto cpu_result = dispatch(op, cpu_inputs, attrs);
            std::vector<Tensor> gpu_result;
            gpu_result.reserve(cpu_result.size());
            for (auto& t : cpu_result) gpu_result.push_back(t.to(dev));
            return gpu_result;
        });
    };

    // FFT family — uses vDSP via Accelerate (shared memory, no GPU roundtrip)
    mps_accelerate_single(OpId::FFT);
    mps_accelerate_single(OpId::IFFT);
    mps_accelerate_single(OpId::RFFT);
    mps_accelerate_single(OpId::IRFFT);
    mps_accelerate_single(OpId::FFTN);
    mps_accelerate_single(OpId::IFFTN);
    mps_accelerate_single(OpId::FFT2);
    mps_accelerate_single(OpId::IFFT2);
    mps_accelerate_single(OpId::STFT);
    mps_accelerate_single(OpId::ISTFT);

    // Linalg family — uses LAPACK/Accelerate (shared memory)
    mps_accelerate_multi(OpId::LinalgSVD);
    mps_accelerate_multi(OpId::LinalgQR);
    mps_accelerate_multi(OpId::LinalgEigh);
    mps_accelerate_multi(OpId::LinalgLU);
    mps_accelerate_multi(OpId::LinalgEig);
    mps_accelerate_single(OpId::LinalgDet);
    mps_accelerate_single(OpId::LinalgInv);
    mps_accelerate_single(OpId::LinalgSolve);
    mps_accelerate_single(OpId::LinalgCholesky);
    mps_accelerate_single(OpId::LinalgLUSolve);

    // Conj — CPU/CUDA/ROCm/Vulkan/OneAPI all register this; MPS previously
    // didn't at all, so every complex-dtype linalg backward that calls
    // adjoint()/conj() (Det/Inv/Solve/Cholesky/Svd/Eigh/LUBackward) threw
    // "kernel for Conj not found" as soon as it reached an MPS-device
    // tensor (complex-dtype linalg forward already shuttles through CPU
    // via try_gpu_dispatch, so this is genuinely reachable).
    mps_accelerate_single(OpId::Conj);

    // Sparse SpMM/SpMV — native Metal compute shaders (see sparse.metal).
    // Reads CSR (crow_indices, col_indices, values) + dense input, writes
    // dense output via the GPU command queue. No CPU dispatch.
    table.register_single_output_kernel(OpId::SparseSpMV,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            if (!attrs.has(AttrKey::M) || !attrs.has(AttrKey::K)) {
                throw std::runtime_error(
                    "MPS SparseSpMV: required attributes M and K not provided.");
            }
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            return mps_sparse_spmv_kernel(inputs[0], inputs[1], inputs[2], inputs[3], M, K);
        });
    table.register_single_output_kernel(OpId::SparseSpMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            if (!attrs.has(AttrKey::M) || !attrs.has(AttrKey::K)) {
                throw std::runtime_error(
                    "MPS SparseSpMM: required attributes M and K not provided.");
            }
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            return mps_sparse_spmm_kernel(inputs[0], inputs[1], inputs[2], inputs[3], M, K);
        });
    mps_accelerate_single(OpId::SparseToDense);
    mps_accelerate_multi(OpId::DenseToSparse);
    mps_accelerate_single(OpId::SparseAdd);
    mps_accelerate_single(OpId::SparseTrsm);
    mps_accelerate_single(OpId::SparseTrsv);
    mps_accelerate_multi(OpId::SparseSpGEMM);

    // ================================================================
    // Remaining ops — native Metal or Accelerate via shared memory
    // ================================================================

    // Pooling backward ops (use Metal atomic scatter)
    //
    // MaxPool3dBackward: forward (native Metal, mps_maxpool3d_forward_kernel in
    // mps_extended_ops.mm) allocates its indices buffer as Int32, but this op
    // round-trips through the CPU kernel table (mps_accelerate_single below),
    // and the CPU maxpool3d_backward_kernel unconditionally requires Int64
    // indices (indices.data<int64_t>()) — matching the Int64 pooling-index
    // convention used by every other backend (CPU/CUDA/ROCm/Vulkan/OneAPI all
    // allocate pooling indices as Int64). A plain mps_accelerate_single copies
    // inputs to CPU verbatim with no dtype fixup, so the Int32 indices tensor
    // throws a DTypeException inside the CPU kernel every time — audit CR1.
    // Give this op its own wrapper that upcasts indices to Int64 before the
    // CPU dispatch instead of the generic accelerate helper.
    // M13: dtype wasn't the only mismatch — forward's indices (pool3d.metal
    // maxpool3d_forward_kernel) also used to store a global NCHW-flat index,
    // while the CPU maxpool3d_backward_kernel this round-trips into expects a
    // (n,c)-plane-local index and reconstructs the plane base itself. Sending
    // it an already-global index silently double-counted the plane offset and
    // scattered the gradient to a wrong (often out-of-bounds) input position.
    // Fixed at the source: forward now stores the plane-local index, matching
    // every other backend.
    table.register_single_output_kernel(OpId::MaxPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto dev = inputs[0].device();
            Tensor grad_output_cpu = inputs[0].to(Device::cpu());
            Tensor indices_cpu = inputs[1].to(Device::cpu());
            if (indices_cpu.dtype() != DType::Int64) {
                indices_cpu = indices_cpu.to(DType::Int64);
            }
            std::vector<Tensor> cpu_inputs{grad_output_cpu, indices_cpu};
            auto cpu_result = dispatch(OpId::MaxPool3dBackward, cpu_inputs, attrs);
            return cpu_result[0].to(dev);
        });
    mps_accelerate_single(OpId::AvgPool3dBackward);
    mps_accelerate_single(OpId::AdaptiveAvgPool3dBackward);
    table.register_single_output_kernel(OpId::AdaptiveMaxPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Native Metal — replaces the prior CPU-roundtrip mps_accelerate_single
            // dispatch for this op. Inputs: {grad_output, indices}; the input
            // shape (required to allocate grad_input) is packed in attrs.
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return mps_adaptive_maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape);
        });

    // 1D/2D pooling (already native or simple Accelerate)
    mps_accelerate_single(OpId::AdaptiveAvgPool1d);
    mps_accelerate_single(OpId::AdaptiveAvgPool1dBackward);
    mps_accelerate_single(OpId::AdaptiveAvgPool2d);
    mps_accelerate_single(OpId::AdaptiveAvgPool2dBackward);
    mps_accelerate_single(OpId::AdaptiveAvgPool3d);
    mps_accelerate_single(OpId::AdaptiveMaxPool1dBackward);
    mps_accelerate_single(OpId::AdaptiveMaxPool2dBackward);
    mps_accelerate_multi(OpId::AdaptiveMaxPool1d);
    mps_accelerate_multi(OpId::AdaptiveMaxPool2d);
    mps_accelerate_multi(OpId::AdaptiveMaxPool3d);
    mps_accelerate_single(OpId::AvgPool1dForward);
    mps_accelerate_single(OpId::AvgPool1dBackward);
    mps_accelerate_single(OpId::AvgPool2dForward);
    mps_accelerate_single(OpId::AvgPool2dBackward);
    mps_accelerate_single(OpId::MaxPool1dBackward);
    mps_accelerate_single(OpId::MaxPool2dBackward);
    mps_accelerate_multi(OpId::MaxPool1dForward);
    mps_accelerate_multi(OpId::MaxPool2dForward);

    // AdvancedIndex/Put (Metal shader)
    mps_accelerate_single(OpId::AdvancedIndex);
    mps_accelerate_single(OpId::AdvancedIndexPut);
    mps_accelerate_single(OpId::AffineGrid);

    // Vision backward ops — previously unregistered on MPS, making
    // upsample/grid_sample/affine_grid non-differentiable (backward threw
    // "no kernel registered"). Route through the CPU reference like AffineGrid
    // forward above so the same graph that trains on every other backend also
    // trains on MPS. GridSampleBackward returns {grad_input, grad_grid} (multi);
    // InterpolateBackward and AffineGridBackward are single-output.
    mps_accelerate_single(OpId::InterpolateBackward);   // F029
    mps_accelerate_multi(OpId::GridSampleBackward);     // F053
    mps_accelerate_single(OpId::AffineGridBackward);    // F053

    // Conv backward ops (Accelerate/shared memory)
    mps_accelerate_multi(OpId::Conv1dForward);
    mps_accelerate_multi(OpId::Conv1dBackwardInput);
    mps_accelerate_multi(OpId::Conv1dBackwardWeight);
    mps_accelerate_multi(OpId::Conv1dBackwardBias);
    mps_accelerate_multi(OpId::Conv2dBackwardBias);
    mps_accelerate_multi(OpId::Conv2dBackwardInput);
    mps_accelerate_multi(OpId::Conv2dBackwardWeight);
    mps_accelerate_multi(OpId::Conv3dBackwardBias);
    mps_accelerate_multi(OpId::Conv3dBackwardInput);
    mps_accelerate_multi(OpId::Conv3dBackwardWeight);
    mps_accelerate_multi(OpId::ConvTranspose2dForward);
    // ConvTranspose1dForward: CPU already registers this (unsqueeze/squeeze
    // 2D-lowering); MPS had no registration at all, a real gap for the
    // JIT-graph ConvTranspose replay path (src/jit/graph.cpp dispatches this
    // OpId directly for 1-D nodes) — not covered by the ordinary eager
    // nn::ConvTranspose1d layer, which never reaches this OpId.
    mps_accelerate_multi(OpId::ConvTranspose1dForward);
    mps_accelerate_multi(OpId::ConvTranspose3dForward);
    mps_accelerate_multi(OpId::ConvTranspose3dBackwardBias);
    mps_accelerate_multi(OpId::ConvTranspose3dBackwardInput);
    mps_accelerate_multi(OpId::ConvTranspose3dBackwardWeight);
    mps_accelerate_multi(OpId::DepthwiseConv2d);
    // F040: register the 1d/3d depthwise fast paths too (CPU has real
    // DepthwiseConv1d/3d kernels). Without these, MPS silently fell back to the
    // generic grouped-conv path while the other 5 backends took the depthwise
    // fast path — the two must stay numerically identical, so route MPS to the
    // same CPU depthwise kernel the fast path uses.
    mps_accelerate_multi(OpId::DepthwiseConv1d);
    mps_accelerate_multi(OpId::DepthwiseConv3d);

    // Norm variants
    mps_accelerate_multi(OpId::BatchNorm2dBackward);
    mps_accelerate_multi(OpId::BatchNorm2dUpdateRunningStats);

    // Native Metal GroupNorm / InstanceNorm / RMSNorm wiring — replaces the
    // CPU-roundtripping mps_accelerate_multi routing for these ops. Each
    // lambda forwards the op's attrs and inputs to the kernel wrapper in
    // kernels/mps_normalization.mm.
    table.register_kernel(OpId::GroupNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return mps_groupnorm_forward(inputs[0], num_groups, inputs[1], inputs[2], eps);
        });
    table.register_kernel(OpId::GroupNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
            // inputs: [grad_output, input, mean_saved, rstd_saved, weight]
            return mps_groupnorm_backward(inputs[0], inputs[1], num_groups,
                                          inputs[2], inputs[3], inputs[4]);
        });
    table.register_kernel(OpId::InstanceNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return mps_instancenorm_forward(inputs[0], inputs[1], inputs[2], eps);
        });
    table.register_kernel(OpId::InstanceNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            // inputs: [grad_output, input, weight, mean, rstd] — order set by
            // the autograd InstanceNormBackward call sites (normalization.cpp,
            // "InstanceNormBackward: inputs [grad_output, input, weight, mean,
            // rstd]"). mps_instancenorm_backward's own signature is
            // (grad_output, input, mean_saved, rstd_saved, weight), so it needs
            // reindexing here — mirrors cpu_kernel_registry.cpp's
            // OpId::InstanceNormBackward wrapper, which does the same
            // inputs[3]/inputs[4]/inputs[2] reindex. The previous direct
            // passthrough silently rotated weight->mean_saved,
            // mean->rstd_saved, rstd->weight.
            return mps_instancenorm_backward(inputs[0], inputs[1], inputs[3], inputs[4], inputs[2]);
        });
    mps_accelerate_multi(OpId::LayerNorm);
    mps_accelerate_multi(OpId::LayerNormBackward);
    mps_accelerate_multi(OpId::FusedLayerNormBackward);
    table.register_kernel(OpId::FusedRMSNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return mps_rmsnorm_forward(inputs[0], inputs[1], eps);
        });
    table.register_kernel(OpId::RMSNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return mps_rmsnorm_forward(inputs[0], inputs[1], eps);
        });
    table.register_kernel(OpId::RMSNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            // inputs: [grad_output, input, rrms, weight] — order set by the
            // autograd RMSNormBackward (normalization.cpp: "inputs_vec = {go,
            // inp, rm, wt}"). mps_rmsnorm_backward's own signature is
            // (grad_output, input, weight, rrms), so weight=inputs[3],
            // rrms=inputs[2] — mirrors cpu_kernel_registry.cpp's
            // OpId::RMSNormBackward wrapper, which documents this exact same
            // scramble having previously been fixed there. The previous direct
            // passthrough here silently swapped weight and rrms.
            return mps_rmsnorm_backward(inputs[0], inputs[1], inputs[3], inputs[2]);
        });

    // CTC Loss — log-domain forward-backward DP
    table.register_kernel(OpId::CTCLossForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t blank = attrs.get_int(AttrKey::Blank, 0);
            bool zero_infinity = attrs.get_bool(AttrKey::ZeroInfinity, false);
            return ctc_loss_forward_kernel(inputs[0], inputs[1], inputs[2], inputs[3],
                                           blank, zero_infinity);
        });

    // RNN family
    mps_accelerate_multi(OpId::LSTMCellForward);
    mps_accelerate_multi(OpId::LSTMCellBackward);
    mps_accelerate_multi(OpId::LSTMForward);
    mps_accelerate_multi(OpId::LSTMMultiLayerForward);
    mps_accelerate_multi(OpId::GRUCellForward);
    mps_accelerate_multi(OpId::GRUCellBackward);
    mps_accelerate_multi(OpId::GRUForward);
    mps_accelerate_multi(OpId::GRUMultiLayerForward);
    mps_accelerate_multi(OpId::BiLSTMForward);

    // Attention
    //
    // FlashAttention/FlashAttentionBackward/FusedAttention route through the
    // same CPU round-trip (Accelerate via shared memory on Apple Silicon,
    // effectively free thanks to unified memory) as every sibling op in this
    // file. Float64 works fine here -- the dispatched CPU kernel handles it
    // directly and no Metal-side compute (with its lack of an MSL `double`
    // type) is ever involved in this dispatch path.
    // (JIT-R199: this previously had a bespoke wrapper that threw on Float64
    // citing an MSL double-support rationale that does not apply to a
    // CPU-round-trip dispatch -- removed for consistency with
    // NestedAttention/FlexAttentionBackward/GatherRelativePositionBias below,
    // which always ran Float64 through this same path.)
    mps_accelerate_multi(OpId::FlashAttention);
    mps_accelerate_multi(OpId::FlashAttentionBackward);
    mps_accelerate_multi(OpId::FusedAttention);
    // FlexAttention (forward) was missing here — a copy/paste omission from
    // this trio — while OpId::FlexAttentionBackward was registered below,
    // making the backward registration permanently unreachable dead code
    // (forward always threw "kernel not found" first, so no Variable with a
    // FlexAttentionBackward grad_fn could ever exist on this device).
    mps_accelerate_multi(OpId::FlexAttention);
    mps_accelerate_multi(OpId::GatherRelativePositionBias);

    // Embedding backward
    mps_accelerate_multi(OpId::EmbeddingBackward);
    mps_accelerate_single(OpId::EmbeddingBagForward);
    mps_accelerate_single(OpId::EmbeddingBagBackward);

    // Linear backward
    mps_accelerate_multi(OpId::LinearBackward);

    // Softmax backward, LogSoftmax backward
    mps_accelerate_single(OpId::SoftmaxBackward);
    mps_accelerate_single(OpId::LogSoftmaxBackward);

    // Fused conv+activation ops
    mps_accelerate_single(OpId::FusedBatchNormReLU);
    mps_accelerate_single(OpId::FusedConv2dBnReLU);
    mps_accelerate_single(OpId::FusedConv2dReLU);
    mps_accelerate_single(OpId::FusedConv2dSigmoid);
    mps_accelerate_single(OpId::FusedConv2dSwish);
    mps_accelerate_single(OpId::FusedConv2dTanh);
    mps_accelerate_single(OpId::FusedLinearReLU);

    // Quantized ops
    mps_accelerate_single(OpId::QuantizedConv2d);
    mps_accelerate_single(OpId::QuantizedLinear);

    // Misc remaining ops
    mps_accelerate_single(OpId::GumbelSoftmax);
    mps_accelerate_single(OpId::Multinomial);
    mps_accelerate_single(OpId::Fold);
    mps_accelerate_single(OpId::Unfold);
    mps_accelerate_single(OpId::ToMemoryFormat);
    mps_accelerate_multi(OpId::Histogram);
    mps_accelerate_multi(OpId::BetaInc);
    // M16: native Metal for Float32/Float16/BFloat16 (was a full CPU
    // round-trip for every dtype). Float64 has no Metal Shading Language
    // double type at all, so it keeps the CPU round-trip inline below —
    // matching the box_iou/grid_sample precedent above of gating Float64
    // rather than misbinding an 8-byte buffer to a 4-byte-float shader.
    // ROIAlignBackward now has a native kernel too (roi_align_backward_kernel,
    // pool3d.metal), matching the same Float64-CPU-roundtrip / F16-BF16-widen
    // discipline as forward.
    table.register_kernel(OpId::ROIAlignForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            if (inputs[0].dtype() == DType::Float64) {
                auto dev = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                auto cpu_result = dispatch(OpId::ROIAlignForward, cpu_inputs, attrs);
                std::vector<Tensor> gpu_result;
                gpu_result.reserve(cpu_result.size());
                for (auto& t : cpu_result) gpu_result.push_back(t.to(dev));
                return gpu_result;
            }
            int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 7);
            int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 7);
            float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
            int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
            bool aligned = attrs.get_bool(AttrKey::Aligned, true);
            return {mps_roi_align_forward_kernel(inputs[0], inputs[1], output_h, output_w,
                                                 spatial_scale, sampling_ratio, aligned)};
        });
    table.register_kernel(OpId::ROIAlignBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [grad_output, rois]
            if (inputs[0].dtype() == DType::Float64) {
                auto dev = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                auto cpu_result = dispatch(OpId::ROIAlignBackward, cpu_inputs, attrs);
                std::vector<Tensor> gpu_result;
                gpu_result.reserve(cpu_result.size());
                for (auto& t : cpu_result) gpu_result.push_back(t.to(dev));
                return gpu_result;
            }
            int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 1);
            int64_t feat_height = attrs.get_int(AttrKey::FeatHeight, 0);
            int64_t feat_width = attrs.get_int(AttrKey::FeatWidth, 0);
            float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
            int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
            bool aligned = attrs.get_bool(AttrKey::Aligned, true);
            return {mps_roi_align_backward_kernel(inputs[0], inputs[1],
                                                  batch_size, feat_height, feat_width,
                                                  spatial_scale, sampling_ratio, aligned)};
        });

    // Fractional Max Pool + Max Unpool — use Metal shaders via Accelerate path
    mps_accelerate_multi(OpId::FractionalMaxPool2dForward);
    mps_accelerate_multi(OpId::FractionalMaxPool3dForward);
    mps_accelerate_single(OpId::FractionalMaxPool2dBackward);
    mps_accelerate_single(OpId::FractionalMaxPool3dBackward);
    mps_accelerate_single(OpId::MaxUnpool1dBackward);
    // C3: max_unpool{2,3}d_forward_kernel (pool3d.metal) now correctly use
    // the plane-local index convention (previously a stale global-flat
    // assumption, and dead code -- nothing called them). Wired up here,
    // replacing the CPU round-trip; Float64 has no Metal double type so it
    // stays on the round-trip below, matching every other native kernel's
    // convention in this file.
    table.register_single_output_kernel(OpId::MaxUnpool2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            if (inputs[0].dtype() == DType::Float64) {
                auto dev = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                return dispatch(OpId::MaxUnpool2dForward, cpu_inputs, attrs)[0].to(dev);
            }
            int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
            int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
            return mps_max_unpool2d_forward_kernel(inputs[0], inputs[1], out_h, out_w);
        });
    mps_accelerate_single(OpId::MaxUnpool2dBackward);
    table.register_single_output_kernel(OpId::MaxUnpool3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            if (inputs[0].dtype() == DType::Float64) {
                auto dev = inputs[0].device();
                std::vector<Tensor> cpu_inputs;
                cpu_inputs.reserve(inputs.size());
                for (const auto& t : inputs) cpu_inputs.push_back(t.to(Device::cpu()));
                return dispatch(OpId::MaxUnpool3dForward, cpu_inputs, attrs)[0].to(dev);
            }
            int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
            int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
            int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
            return mps_max_unpool3d_forward_kernel(inputs[0], inputs[1], out_d, out_h, out_w);
        });
    mps_accelerate_single(OpId::MaxUnpool3dBackward);

    // ================================================================
    // Tier 2: Reduction and scan ops — Accelerate via shared memory
    // ================================================================

    // Cumulative scans (multi-output: values + indices)
    mps_accelerate_multi(OpId::CumMax);
    mps_accelerate_multi(OpId::CumMin);

    // Reduction ops (single-output)
    mps_accelerate_single(OpId::LogSumExp);
    mps_accelerate_single(OpId::Logcumsumexp);
    mps_accelerate_single(OpId::Bincount);
    mps_accelerate_single(OpId::NanVar);
    mps_accelerate_single(OpId::NanStd);
    mps_accelerate_single(OpId::HasInfNan);

    // Set membership (binary input, single boolean output)
    mps_accelerate_single(OpId::Isin);

    // K-th value (multi-output: values + indices)
    mps_accelerate_multi(OpId::Kthvalue);

    // Quantile/median ops (single-output)
    mps_accelerate_single(OpId::Quantile);
    mps_accelerate_single(OpId::Nanquantile);
    mps_accelerate_single(OpId::Nanmedian);

    // Fixed-bin histogram (single-output)
    mps_accelerate_single(OpId::Histc);

    // Deduplicate consecutive elements (multi-output: output, inverse, counts)
    mps_accelerate_multi(OpId::UniqueConsecutive);

    // ================================================================
    // Tier 3: Advanced ops — Accelerate via shared memory
    // ================================================================

    // Scatter/gather ops
    mps_accelerate_single(OpId::ScatterReduce);
    mps_accelerate_single(OpId::RepeatInterleave);
    mps_accelerate_single(OpId::TakeAlongDim);
    mps_accelerate_single(OpId::MaskedScatter);

    // Diagonal embedding
    mps_accelerate_single(OpId::DiagEmbed);
    mps_accelerate_single(OpId::Diagflat);

    // Triangular solve
    mps_accelerate_single(OpId::SolveTriangular);

    // Ormqr, Geqrf, ComplexTensor
    mps_accelerate_single(OpId::Ormqr);
    mps_accelerate_multi(OpId::Geqrf);
    mps_accelerate_single(OpId::ComplexTensor);

    // Special math functions — Accelerate vForce/vvfuncs
    mps_accelerate_single(OpId::Gamma);
    mps_accelerate_single(OpId::Lgamma);
    mps_accelerate_single(OpId::Digamma);
    mps_accelerate_single(OpId::Beta);
    mps_accelerate_single(OpId::ErfInv);
    mps_accelerate_single(OpId::Sinc);
    mps_accelerate_single(OpId::Zeta);

    // Bessel functions
    mps_accelerate_single(OpId::BesselJ0);
    mps_accelerate_single(OpId::BesselJ1);
    mps_accelerate_single(OpId::BesselY0);
    mps_accelerate_single(OpId::BesselY1);
    mps_accelerate_single(OpId::BesselI0);
    mps_accelerate_single(OpId::BesselI1);

    // ================================================================
    // Tier 4: Nested tensor ops — Accelerate via shared memory
    // ================================================================

    mps_accelerate_multi(OpId::NestedSoftmax);
    mps_accelerate_multi(OpId::NestedLogSoftmax);
    mps_accelerate_multi(OpId::NestedLayerNorm);
    mps_accelerate_multi(OpId::NestedSum);
    mps_accelerate_multi(OpId::NestedMean);
    mps_accelerate_multi(OpId::NestedAttention);
    mps_accelerate_multi(OpId::NestedToPadded);
    mps_accelerate_multi(OpId::NestedFromPadded);
    mps_accelerate_multi(OpId::NestedLinear);
    mps_accelerate_multi(OpId::NestedAttentionBackward);

    // AsStrided — metadata-only view, works on any device
    mps_accelerate_single(OpId::AsStrided);

    // ================================================================
    // Phase 8.6 — complete op-coverage parity
    // ================================================================
    // The remaining 158 ops (math elementwise, special functions,
    // sampling, sparse-softmax, etc.) route through the CPU fallback
    // via Apple Silicon's unified memory. The .to(cpu)/.to(mps)
    // round-trip is effectively free on M-series GPUs (shared system
    // memory), so the cost is just the CPU compute itself — which
    // for these ops is mostly Accelerate / vDSP backed and fast.
    //
    // Per-op native Metal kernels can replace these registrations
    // incrementally; this block closes the 317/317 op-count gate.
    // ================================================================
    table.register_single_output_kernel(OpId::Abs,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_abs_kernel(inputs[0]);
        });
    table.register_single_output_kernel(OpId::Acos,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_acos_kernel(inputs[0]);
        });
    mps_accelerate_single(OpId::Acosh);
    mps_accelerate_single(OpId::Addcdiv);
    mps_accelerate_single(OpId::Addcmul);
    mps_accelerate_single(OpId::All);
    mps_accelerate_single(OpId::Angle);
    mps_accelerate_single(OpId::Any);
    mps_accelerate_single(OpId::ArgMax);
    mps_accelerate_single(OpId::ArgMin);
    table.register_single_output_kernel(OpId::Asin,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_asin_kernel(inputs[0]);
        });
    mps_accelerate_single(OpId::Asinh);
    table.register_single_output_kernel(OpId::Atan,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_atan_kernel(inputs[0]);
        });
    table.register_single_output_kernel(OpId::Atan2,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_atan2_kernel(inputs[0], inputs[1]);
        });
    mps_accelerate_single(OpId::Atanh);
    mps_accelerate_single(OpId::Bernoulli);
    mps_accelerate_single(OpId::BitwiseLeftShift);
    mps_accelerate_single(OpId::BitwiseNot);
    mps_accelerate_single(OpId::BitwiseOr);
    mps_accelerate_single(OpId::BitwiseRightShift);
    mps_accelerate_single(OpId::BitwiseXor);
    mps_accelerate_single(OpId::Bmm);
    table.register_single_output_kernel(OpId::Ceil,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_ceil_kernel(inputs[0]);
        });
    mps_accelerate_single(OpId::CholeskyInverse);
    mps_accelerate_single(OpId::Clone);
    mps_accelerate_single(OpId::Contiguous);
    table.register_single_output_kernel(OpId::Copysign,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_copysign_kernel(inputs[0], inputs[1]);
        });
    mps_accelerate_single(OpId::Corrcoef);
    mps_accelerate_single(OpId::Cos);
    mps_accelerate_single(OpId::Cosh);
    mps_accelerate_single(OpId::CosineSimilarity);
    mps_accelerate_single(OpId::Cov);
    mps_accelerate_single(OpId::Cross);
    mps_accelerate_single(OpId::CumulativeTrapezoid);
    mps_accelerate_single(OpId::DCT);
    // L6: forward was never registered at all — the backward-only trio below
    // is unreachable without it (autograd always calls forward first), so
    // deformable_conv2d threw "kernel not found" on MPS unconditionally.
    // CPU-roundtrip, matching every other unrewritten-to-native op in this
    // completeness-parity block; CPU's own DeformableConv2dForward is the
    // reference implementation these backward kernels were already written
    // against.
    mps_accelerate_multi(OpId::DeformableConv2dForward);
    mps_accelerate_multi(OpId::DeformableConv2dBackwardBias);
    mps_accelerate_multi(OpId::DeformableConv2dBackwardInput);
    mps_accelerate_multi(OpId::DeformableConv2dBackwardWeight);
    mps_accelerate_single(OpId::Deg2Rad);
    mps_accelerate_single(OpId::DiagonalScatter);
    mps_accelerate_single(OpId::Div);
    mps_accelerate_single(OpId::Dot);
    mps_accelerate_multi(OpId::Einsum);
    mps_accelerate_single(OpId::EmbeddingWithBoundsCheck);
    mps_accelerate_single(OpId::Entr);
    table.register_single_output_kernel(OpId::Erf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_erf_kernel(inputs[0]);
        });
    table.register_single_output_kernel(OpId::Erfc,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_erfc_kernel(inputs[0]);
        });
    table.register_single_output_kernel(OpId::Exp,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_exp_kernel(inputs[0]);
        });
    table.register_single_output_kernel(OpId::Exp2,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_exp2_kernel(inputs[0]);
        });
    table.register_single_output_kernel(OpId::Expm1,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_expm1_kernel(inputs[0]);
        });
    mps_accelerate_single(OpId::ExponentialSample);
    mps_accelerate_multi(OpId::FlexAttentionBackward);
    mps_accelerate_single(OpId::FloatPower);
    table.register_single_output_kernel(OpId::Floor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_floor_kernel(inputs[0]);
        });
    mps_accelerate_single(OpId::Fmax);
    mps_accelerate_single(OpId::Fmin);
    mps_accelerate_single(OpId::Fmod);
    table.register_single_output_kernel(OpId::Frac,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_frac_kernel(inputs[0]);
        });
    mps_accelerate_multi(OpId::Frexp);
    mps_accelerate_single(OpId::FusedAddReLU);
    mps_accelerate_single(OpId::FusedGelu);
    mps_accelerate_single(OpId::Gcd);
    table.register_single_output_kernel(OpId::Gelu,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_gelu_kernel(inputs[0]);
        });
    mps_accelerate_single(OpId::GeluBackward);
    mps_accelerate_multi(OpId::Histogramdd);
    mps_accelerate_single(OpId::Hypot);
    mps_accelerate_single(OpId::I0e);
    mps_accelerate_single(OpId::I1e);
    mps_accelerate_single(OpId::IDCT);
    mps_accelerate_single(OpId::Igammac);
    mps_accelerate_single(OpId::Imag);
    mps_accelerate_single(OpId::IsFinite);
    mps_accelerate_single(OpId::IsInf);
    mps_accelerate_single(OpId::IsNan);
    mps_accelerate_single(OpId::IsNegInf);
    mps_accelerate_single(OpId::IsPosInf);
    mps_accelerate_single(OpId::IsReal);
    mps_accelerate_single(OpId::Lcm);
    mps_accelerate_single(OpId::Ldexp);
    mps_accelerate_single(OpId::LinalgCholeskySolve);
    mps_accelerate_single(OpId::LinalgHouseholder);
    mps_accelerate_multi(OpId::LinalgLDLFactor);
    mps_accelerate_single(OpId::LinalgLDLSolve);
    mps_accelerate_single(OpId::LinalgMatrixNorm);
    mps_accelerate_single(OpId::LinalgVecdot);
    mps_accelerate_single(OpId::LinalgVectorNorm);
    mps_accelerate_multi(OpId::LOBPCG);
    mps_accelerate_single(OpId::Log);
    mps_accelerate_single(OpId::Log10);
    mps_accelerate_single(OpId::Log1p);
    mps_accelerate_single(OpId::LogAddExp2);
    mps_accelerate_single(OpId::LogicalNot);
    mps_accelerate_single(OpId::LogicalOr);
    mps_accelerate_single(OpId::LogicalXor);
    mps_accelerate_single(OpId::Logit);
    // AUTOGRAD-R045: Ndtr (normal CDF) had no MPS registration at all while
    // every other backend has one, and the closely related LogNdtr right
    // below already uses this same CPU-roundtrip accelerator — an
    // accidental gap, not an intentional omission.
    mps_accelerate_single(OpId::Ndtr);
    mps_accelerate_single(OpId::LogNdtr);
    mps_accelerate_single(OpId::LogSigmoid);
    mps_accelerate_single(OpId::MaskedSelect);
    // NOTE: no mps_accelerate_single(OpId::MatMul) here — a native
    // mps_matmul_kernel is already registered above (TENZOR_REGISTER_
    // BINARY_SINGLE_KERNEL), so mps_accelerate_single would be a silent
    // no-op (it only registers when the table has no kernel for the op
    // yet). mps_matmul_kernel itself now shuttles unsupported dtypes
    // (Float64, Complex64/128) through CPU internally.
    mps_accelerate_single(OpId::Maximum);
    mps_accelerate_single(OpId::MelScale);
    mps_accelerate_single(OpId::MFCC);
    mps_accelerate_single(OpId::Min);
    mps_accelerate_single(OpId::Mish);
    mps_accelerate_single(OpId::MishBackward);
    mps_accelerate_single(OpId::Mul);
    mps_accelerate_single(OpId::Multigammaln);
    mps_accelerate_single(OpId::Neg);
    mps_accelerate_single(OpId::Nextafter);
    mps_accelerate_single(OpId::Nonzero);
    mps_accelerate_single(OpId::NormalSample);
    mps_accelerate_single(OpId::NumericalGradient);
    mps_accelerate_single(OpId::PairwiseDistance);
    mps_accelerate_single(OpId::Pdist);
    mps_accelerate_single(OpId::PoissonSample);
    mps_accelerate_single(OpId::Polar);
    mps_accelerate_single(OpId::Pow);
    mps_accelerate_single(OpId::Prod);
    mps_accelerate_single(OpId::Rad2Deg);
    mps_accelerate_single(OpId::Real);
    mps_accelerate_single(OpId::Reciprocal);
    mps_accelerate_single(OpId::Remainder);
    mps_accelerate_single(OpId::Renorm);
    mps_accelerate_single(OpId::Round);
    mps_accelerate_single(OpId::Rsqrt);
    mps_accelerate_single(OpId::SelectScatter);
    mps_accelerate_single(OpId::Selu);
    mps_accelerate_single(OpId::SeluBackward);
    mps_accelerate_single(OpId::Sigmoid);
    mps_accelerate_single(OpId::Sign);
    mps_accelerate_single(OpId::Signbit);
    mps_accelerate_single(OpId::Sinh);
    mps_accelerate_single(OpId::SliceScatter);
    mps_accelerate_single(OpId::SparseLogSoftmax);
    mps_accelerate_single(OpId::SparseSoftmax);
    mps_accelerate_single(OpId::SphericalBesselJ0);
    mps_accelerate_single(OpId::Square);
    mps_accelerate_single(OpId::Sub);
    mps_accelerate_single(OpId::Swish);
    mps_accelerate_single(OpId::SwishBackward);
    mps_accelerate_single(OpId::Take);
    mps_accelerate_single(OpId::Tan);
    mps_accelerate_single(OpId::Tanh);
    mps_accelerate_single(OpId::TanhActivation);
    mps_accelerate_single(OpId::TensorInv);
    mps_accelerate_single(OpId::TensorSolve);
    mps_accelerate_single(OpId::Trace);
    mps_accelerate_single(OpId::TrilIndices);
    mps_accelerate_single(OpId::TriuIndices);
    mps_accelerate_single(OpId::Trunc);
    mps_accelerate_single(OpId::Where);
    mps_accelerate_single(OpId::Xlog1py);
    mps_accelerate_single(OpId::XLogY);
}

} // namespace tenzor::mps

// Export function for dynamic loading
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::mps::register_mps_kernels(*table);
        }
    }
}
