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
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    std::vector<int64_t> new_shape(ndim);
    std::vector<int64_t> new_strides(ndim);
    for (int64_t i = 0; i < ndim; ++i) {
        new_shape[i] = input.shape()[dims[i]];
        new_strides[i] = input.strides()[dims[i]];
    }
    result.mutable_shape() = std::move(new_shape);
    result.mutable_strides() = std::move(new_strides);
    return result;
}

static auto mps_squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor {
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    if (dim >= 0) {
        auto& r_shape = result.mutable_shape();
        auto& r_strides = result.mutable_strides();
        r_shape.erase(r_shape.begin() + dim);
        r_strides.erase(r_strides.begin() + dim);
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
    Tensor result;
    TensorAccessor::get_impl_mutable(result) = make_intrusive<TensorImpl>(*TensorAccessor::get_impl(input));
    auto& r_shape = result.mutable_shape();
    auto& r_strides = result.mutable_strides();
    r_shape.insert(r_shape.begin() + dim, 1);
    int64_t new_stride = (dim < input.ndim()) ? input.strides()[dim] : 1;
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
                          int64_t pad_h, int64_t pad_w, int64_t groups);
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
Tensor mps_conv3d_forward_kernel(const Tensor& input, const Tensor& weight,
                                  int64_t sd, int64_t sh, int64_t sw,
                                  int64_t pd, int64_t ph, int64_t pw, int64_t groups);
Tensor mps_maxpool3d_forward_kernel(const Tensor& input, int64_t kd, int64_t kh, int64_t kw,
                                     int64_t sd, int64_t sh, int64_t sw,
                                     int64_t pd, int64_t ph, int64_t pw, Tensor& indices);
Tensor mps_avgpool3d_forward_kernel(const Tensor& input, int64_t kd, int64_t kh, int64_t kw,
                                     int64_t sd, int64_t sh, int64_t sw,
                                     int64_t pd, int64_t ph, int64_t pw, bool count_include_pad);
Tensor mps_cdist_kernel(const Tensor& x1, const Tensor& x2, float p);
std::vector<Tensor> mps_sort_kernel(const Tensor& input, int64_t dim, bool descending);
Tensor mps_argsort_kernel(const Tensor& input, int64_t dim, bool descending);
std::vector<Tensor> mps_topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest);
std::vector<Tensor> mps_median_kernel(const Tensor& input, int64_t dim);
std::vector<Tensor> mps_mode_kernel(const Tensor& input, int64_t dim);
std::vector<Tensor> mps_unique_kernel(const Tensor& input, bool sorted, bool return_inverse, bool return_counts);
Tensor mps_grid_sample_kernel(const Tensor& input, const Tensor& grid, bool align_corners);
Tensor mps_interpolate_kernel(const Tensor& input, int64_t out_h, int64_t out_w,
                               const std::string& mode, bool align_corners);
Tensor mps_box_iou_kernel(const Tensor& boxes1, const Tensor& boxes2);
Tensor mps_nms_kernel(const Tensor& boxes, const Tensor& scores, float iou_threshold);
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
std::vector<Tensor> mps_fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets);
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
            int64_t sh = attrs.get_int(AttrKey::StrideH, 1);
            int64_t sw = attrs.get_int(AttrKey::StrideW, 1);
            int64_t ph = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pw = attrs.get_int(AttrKey::PaddingW, 0);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            return {mps_conv2d_kernel(inputs[0], inputs[1], sh, sw, ph, pw, groups)};
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
        Tensor indices;
        auto values = mps_max_kernel(inputs[0], dim, keepdim, indices);
        return std::vector<Tensor>{values, indices};
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
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
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

    // In-place arithmetic — native Metal
    table.register_kernel(OpId::AddInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];  // copy handle (shared storage — kernel writes in-place)
        return std::vector<Tensor>{mps_add_inplace_kernel(a, inputs[1])};
    });
    table.register_kernel(OpId::SubInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];
        return std::vector<Tensor>{mps_sub_inplace_kernel(a, inputs[1])};
    });
    table.register_kernel(OpId::MulInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];
        return std::vector<Tensor>{mps_mul_inplace_kernel(a, inputs[1])};
    });
    table.register_kernel(OpId::DivInplace, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor a = inputs[0];
        return std::vector<Tensor>{mps_div_inplace_kernel(a, inputs[1])};
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
            float p = static_cast<float>(attrs.get_float(AttrKey::Order, 2.0));
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
            int64_t order = attrs.get_int(AttrKey::Order, 0);
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
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            return {mps_conv3d_forward_kernel(inputs[0], inputs[1], sd, sh, sw, pd, ph, pw, groups)};
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
            Tensor indices;
            auto output = mps_maxpool3d_forward_kernel(inputs[0], kd, kh, kw, sd, sh, sw, pd, ph, pw, indices);
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
            bool count_pad = attrs.get_bool(AttrKey::CountIncludePad, false);
            return mps_avgpool3d_forward_kernel(inputs[0], kd, kh, kw, sd, sh, sw, pd, ph, pw, count_pad);
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
            bool align = attrs.get_bool(AttrKey::AlignCorners, false);
            return mps_grid_sample_kernel(inputs[0], inputs[1], align);
        });
    table.register_single_output_kernel(OpId::Interpolate,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t oh = attrs.get_int(AttrKey::OutputSizeH, 0);
            int64_t ow = attrs.get_int(AttrKey::OutputSizeW, 0);
            bool align = attrs.get_bool(AttrKey::AlignCorners, false);
            std::string mode = "bilinear"; // default
            return mps_interpolate_kernel(inputs[0], oh, ow, mode, align);
        });
    table.register_single_output_kernel(OpId::BoxIoU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return mps_box_iou_kernel(inputs[0], inputs[1]);
        });
    table.register_kernel(OpId::NMS,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 0.5));
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
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return mps_fused_softmax_cross_entropy_kernel(inputs[0], inputs[1]);
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
            // grad * mask * scale — reuse the mask from forward
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
            float scale = 1.0f / (1.0f - p);
            // inputs[0] = grad, inputs[1] = mask
            // Use element-wise multiplication via shared memory
            auto shape = inputs[0].shape();
            std::vector<int64_t> sv(shape.begin(), shape.end());
            Tensor output(sv, inputs[0].dtype(), inputs[0].device());
            const float* g = static_cast<const float*>(inputs[0].data_ptr());
            const float* m = static_cast<const float*>(inputs[1].data_ptr());
            float* o = static_cast<float*>(const_cast<void*>(output.data_ptr()));
            for (size_t i = 0; i < static_cast<size_t>(inputs[0].numel()); ++i)
                o[i] = g[i] * m[i] * scale;
            return output;
        });

    // ================================================================
    // Accelerate-based ops (FFT, Linalg, Sparse) — use vDSP/LAPACK
    // via shared memory (zero-copy on Apple Silicon, no GPU roundtrip)
    // ================================================================

    // Helper: dispatch to CPU but use shared memory (MPS unified memory
    // means .to(cpu) and .to(mps) are effectively free on Apple Silicon)
    auto mps_accelerate_single = [&](OpId op) {
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
    mps_accelerate_single(OpId::MaxPool3dBackward);
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
    mps_accelerate_multi(OpId::ConvTranspose3dForward);
    mps_accelerate_multi(OpId::ConvTranspose3dBackwardBias);
    mps_accelerate_multi(OpId::ConvTranspose3dBackwardInput);
    mps_accelerate_multi(OpId::ConvTranspose3dBackwardWeight);
    mps_accelerate_multi(OpId::DepthwiseConv2d);

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
            // inputs: [grad_output, input, mean_saved, rstd_saved, weight]
            return mps_instancenorm_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4]);
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
            // inputs: [grad_output, input, weight, rrms]
            return mps_rmsnorm_backward(inputs[0], inputs[1], inputs[2], inputs[3]);
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
    // audit A.11 (MPS): Float64 attention is unsupported on the Metal backend.
    // The Metal Shading Language specification (see Apple "Metal Shading
    // Language Specification", section "Data Types" / "Scalar Data Types")
    // defines no `double` type — only `half` (FP16) and `float` (FP32) are
    // available to compute shaders, and there is no MSL FP64 extension on any
    // shipping Apple silicon GPU. The project rule forbids both CPU fallbacks
    // for GPU backends and Float32 upcasts, so the only honest response when
    // a user invokes FlashAttention with Float64 tensors on an MPS device is
    // to throw immediately at dispatch. All non-Float64 dtypes continue to
    // route through the existing Accelerate-based path (shared memory, no
    // GPU round-trip on Apple Silicon).
    auto mps_attention_throw_on_f64 = [&](OpId op) {
        table.register_kernel(op, [op](std::span<const Tensor> inputs,
                                       const OpAttributes& attrs) {
            if (!inputs.empty() && inputs[0].dtype() == DType::Float64) {
                throw std::runtime_error(
                    "MPS FlashAttention: Metal Shading Language does not "
                    "support Float64 (no `double` type per Apple's MSL "
                    "specification, \"Scalar Data Types\"). This is a "
                    "platform limitation, not a Tenzor bug. Use a different "
                    "backend (CPU, CUDA, ROCm, OneAPI, or Vulkan with "
                    "shaderFloat64) for Float64 attention.");
            }
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
    mps_attention_throw_on_f64(OpId::FlashAttention);
    mps_attention_throw_on_f64(OpId::FlashAttentionBackward);
    mps_attention_throw_on_f64(OpId::FusedAttention);
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
    mps_accelerate_multi(OpId::ROIAlignForward);
    mps_accelerate_multi(OpId::ROIAlignBackward);

    // Fractional Max Pool + Max Unpool — use Metal shaders via Accelerate path
    mps_accelerate_multi(OpId::FractionalMaxPool2dForward);
    mps_accelerate_multi(OpId::FractionalMaxPool3dForward);
    mps_accelerate_single(OpId::FractionalMaxPool2dBackward);
    mps_accelerate_single(OpId::FractionalMaxPool3dBackward);
    mps_accelerate_single(OpId::MaxUnpool2dForward);
    mps_accelerate_single(OpId::MaxUnpool2dBackward);
    mps_accelerate_single(OpId::MaxUnpool3dForward);
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
    mps_accelerate_single(OpId::GeluInplace);
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
    mps_accelerate_single(OpId::LeakyReLUInplace);
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
    mps_accelerate_single(OpId::LogNdtr);
    mps_accelerate_single(OpId::LogSigmoid);
    mps_accelerate_single(OpId::MaskedSelect);
    mps_accelerate_single(OpId::MatMul);
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
    mps_accelerate_single(OpId::ReLUInplace);
    mps_accelerate_single(OpId::Remainder);
    mps_accelerate_single(OpId::Renorm);
    mps_accelerate_single(OpId::Round);
    mps_accelerate_single(OpId::Rsqrt);
    mps_accelerate_single(OpId::SelectScatter);
    mps_accelerate_single(OpId::Selu);
    mps_accelerate_single(OpId::SeluBackward);
    mps_accelerate_single(OpId::Sigmoid);
    mps_accelerate_single(OpId::SigmoidInplace);
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
    mps_accelerate_single(OpId::TanhInplace);
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
