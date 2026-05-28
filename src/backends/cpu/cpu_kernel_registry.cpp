/**
 * @file cpu_kernel_registry.cpp
 * @brief CPU kernel registration for O(1) dispatch
 *
 * Registers all CPU kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/attr_macros.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/nn/layers/flex_attention.hpp"  // Audit J12: find_registered_score_mod
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/detection.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include <array>
#include <cmath>
#include <cstdlib>
#include <climits>
#include <cstdint>
#include <tuple>

namespace tenzor {

// Forward declarations for CPU kernels (same as in cpu_backend.cpp)
namespace cpu {
    auto add_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto bmm_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                      double alpha, double beta) -> Tensor;
    auto addmv_kernel(const Tensor& input, const Tensor& mat, const Tensor& vec,
                      double alpha, double beta) -> Tensor;
    auto baddbmm_kernel(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
                        double alpha, double beta) -> Tensor;

    auto sqrt_kernel(const Tensor& input) -> Tensor;
    auto neg_kernel(const Tensor& input) -> Tensor;
    auto abs_kernel(const Tensor& input) -> Tensor;
    auto sign_kernel(const Tensor& input) -> Tensor;
    auto clamp_kernel(const Tensor& input, double min_val, double max_val) -> Tensor;
    auto clamp_min_kernel(const Tensor& input, double min_val) -> Tensor;
    auto clamp_max_kernel(const Tensor& input, double max_val) -> Tensor;
    auto log_kernel(const Tensor& input) -> Tensor;
    auto exp_kernel(const Tensor& input) -> Tensor;
    auto pow_kernel(const Tensor& input, double exponent) -> Tensor;
    auto reciprocal_kernel(const Tensor& input) -> Tensor;
    auto floor_kernel(const Tensor& input) -> Tensor;
    auto ceil_kernel(const Tensor& input) -> Tensor;
    auto round_kernel(const Tensor& input) -> Tensor;
    auto trunc_kernel(const Tensor& input) -> Tensor;
    auto frac_kernel(const Tensor& input) -> Tensor;
    auto heaviside_kernel(const Tensor& input, const Tensor& values) -> Tensor;
    auto nan_to_num_kernel(const Tensor& input, double nan_val, double posinf_val, double neginf_val) -> Tensor;
    auto log_sigmoid_kernel(const Tensor& input) -> Tensor;
    auto log_sigmoid_backward_kernel(const Tensor& grad, const Tensor& input) -> Tensor;
    auto rrelu_kernel(const Tensor& input, float lower, float upper, bool training) -> Tensor;
    auto rrelu_backward_kernel(const Tensor& grad, const Tensor& input, float lower, float upper) -> Tensor;
    auto bitwise_and_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto bitwise_or_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto bitwise_xor_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto bitwise_not_kernel(const Tensor& input) -> Tensor;
    auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift) -> Tensor;
    auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift) -> Tensor;
    auto count_nonzero_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto nansum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto nanmean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto aminmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> std::pair<Tensor, Tensor>;
    auto index_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor;
    auto index_copy_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source) -> Tensor;
    auto index_fill_kernel(const Tensor& input, int64_t dim, const Tensor& index, double value) -> Tensor;
    auto bincount_kernel(const Tensor& input, const Tensor* weights, int64_t minlength) -> Tensor;
    auto take_along_dim_kernel(const Tensor& input, const Tensor& indices, int64_t dim) -> Tensor;
    auto masked_scatter_kernel(const Tensor& input, const Tensor& mask, const Tensor& source) -> Tensor;
    auto tril_indices_kernel(int64_t row, int64_t col, int64_t offset) -> Tensor;
    auto triu_indices_kernel(int64_t row, int64_t col, int64_t offset) -> Tensor;

    // Trigonometric
    auto sin_kernel(const Tensor& input) -> Tensor;
    auto cos_kernel(const Tensor& input) -> Tensor;
    auto tan_kernel(const Tensor& input) -> Tensor;
    auto asin_kernel(const Tensor& input) -> Tensor;
    auto acos_kernel(const Tensor& input) -> Tensor;
    auto atan_kernel(const Tensor& input) -> Tensor;
    auto sinh_kernel(const Tensor& input) -> Tensor;
    auto cosh_kernel(const Tensor& input) -> Tensor;
    auto tanh_kernel(const Tensor& input) -> Tensor;

    // Extended math
    auto log2_kernel(const Tensor& input) -> Tensor;
    auto log10_kernel(const Tensor& input) -> Tensor;
    auto log1p_kernel(const Tensor& input) -> Tensor;
    auto exp2_kernel(const Tensor& input) -> Tensor;
    auto expm1_kernel(const Tensor& input) -> Tensor;
    auto erf_kernel(const Tensor& input) -> Tensor;
    auto erfc_kernel(const Tensor& input) -> Tensor;
    auto erfinv_kernel(const Tensor& input) -> Tensor;
    auto gamma_kernel(const Tensor& input) -> Tensor;
    auto lgamma_kernel(const Tensor& input) -> Tensor;
    auto digamma_kernel(const Tensor& input) -> Tensor;
    auto polygamma_kernel(const Tensor& input, int64_t n) -> Tensor;
    auto beta_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto betainc_kernel(std::span<const Tensor> inputs) -> Tensor;
    auto bessel_j0_kernel(const Tensor& input) -> Tensor;
    auto bessel_j1_kernel(const Tensor& input) -> Tensor;
    auto bessel_y0_kernel(const Tensor& input) -> Tensor;
    auto bessel_y1_kernel(const Tensor& input) -> Tensor;
    auto bessel_i0_kernel(const Tensor& input) -> Tensor;
    auto bessel_i1_kernel(const Tensor& input) -> Tensor;
    auto sinc_kernel(const Tensor& input) -> Tensor;
    auto zeta_kernel(const Tensor& x, const Tensor& q) -> Tensor;
    auto isnan_kernel(const Tensor& input) -> Tensor;
    auto isinf_kernel(const Tensor& input) -> Tensor;
    auto isfinite_kernel(const Tensor& input) -> Tensor;
    auto atan2_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fmod_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto lerp_kernel(std::span<const Tensor> inputs) -> Tensor;

    // New element-wise math operations
    auto rsqrt_kernel(const Tensor& input) -> Tensor;
    auto square_kernel(const Tensor& input) -> Tensor;
    auto asinh_kernel(const Tensor& input) -> Tensor;
    auto acosh_kernel(const Tensor& input) -> Tensor;
    auto atanh_kernel(const Tensor& input) -> Tensor;
    auto hypot_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto copysign_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto nextafter_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto gcd_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto lcm_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto igamma_kernel(const Tensor& a, const Tensor& x) -> Tensor;
    auto igammac_kernel(const Tensor& a, const Tensor& x) -> Tensor;
    auto addcmul_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, double alpha) -> Tensor;
    auto addcdiv_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, double alpha) -> Tensor;

    // Phase 5 Extended Math
    auto deg2rad_kernel(const Tensor& input) -> Tensor;
    auto rad2deg_kernel(const Tensor& input) -> Tensor;
    auto logit_kernel(const Tensor& input) -> Tensor;
    auto signbit_kernel(const Tensor& input) -> Tensor;
    auto isposinf_kernel(const Tensor& input) -> Tensor;
    auto isneginf_kernel(const Tensor& input) -> Tensor;
    auto isreal_kernel(const Tensor& input) -> Tensor;
    auto float_power_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto xlog1py_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ldexp_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto frexp_kernel(const Tensor& input) -> std::vector<Tensor>;
    auto diag_embed_kernel(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor;
    auto diagflat_kernel(const Tensor& input, int64_t offset) -> Tensor;
    auto nanvar_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto nanstd_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;

    // Numerical integration / gradient / distance kernels
    auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr) -> Tensor;
    auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr) -> Tensor;
    auto gradient_kernel(const Tensor& input, int64_t dim, double spacing) -> Tensor;
    auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p) -> Tensor;
    auto pdist_kernel(const Tensor& input, double p) -> Tensor;

    // New math operations (PyTorch parity)
    auto logaddexp_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto logaddexp2_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto xlogy_kernel(const Tensor& x, const Tensor& y) -> Tensor;
    auto i0e_kernel(const Tensor& input) -> Tensor;
    auto i1e_kernel(const Tensor& input) -> Tensor;
    auto entr_kernel(const Tensor& input) -> Tensor;
    auto spherical_bessel_j0_kernel(const Tensor& input) -> Tensor;
    auto renorm_kernel(const Tensor& input, double p, int64_t dim, double maxnorm) -> Tensor;
    auto cosine_similarity_kernel(const Tensor& a, const Tensor& b, int64_t dim, double eps) -> Tensor;

    auto logical_and_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto logical_or_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto logical_not_kernel(const Tensor& input) -> Tensor;
    auto logical_xor_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto minimum_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto maximum_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim) -> Tensor;

    // Complex operations
    auto conj_kernel(const Tensor& input) -> Tensor;
    auto real_kernel(const Tensor& input) -> Tensor;
    auto imag_kernel(const Tensor& input) -> Tensor;
    auto angle_kernel(const Tensor& input) -> Tensor;
    auto polar_kernel(const Tensor& abs, const Tensor& angle) -> Tensor;

    // Reductions
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor;
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending) -> Tensor;
    auto any_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto all_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto has_inf_nan_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto median_kernel(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor>;
    auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim) -> std::vector<Tensor>;

    // Comparison
    auto eq_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b) -> Tensor;

    // Inplace operations
    auto add_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto sub_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto mul_inplace_kernel(Tensor& a, const Tensor& b) -> void;
    auto div_inplace_kernel(Tensor& a, const Tensor& b) -> void;

    // Inplace activation operations
    auto relu_inplace_kernel(Tensor& input) -> void;
    auto sigmoid_inplace_kernel(Tensor& input) -> void;
    auto tanh_inplace_kernel(Tensor& input) -> void;
    auto leaky_relu_inplace_kernel(Tensor& input, double alpha) -> void;
    auto gelu_inplace_kernel(Tensor& input) -> void;

    // Activations
    auto relu_kernel(const Tensor& input) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto sigmoid_kernel(const Tensor& input) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto tanh_activation_kernel(const Tensor& input) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto gelu_kernel(const Tensor& input) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto swish_kernel(const Tensor& input) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, double alpha) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, double alpha) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha) -> Tensor;
    auto selu_kernel(const Tensor& input) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto mish_kernel(const Tensor& input) -> Tensor;
    auto hardswish_kernel(const Tensor& input) -> Tensor;
    auto hardsigmoid_kernel(const Tensor& input) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold) -> Tensor;
    auto softmax_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim) -> Tensor;

    // Shape/Transform
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim) -> Tensor;
    auto contiguous_kernel(const Tensor& input) -> Tensor;
    auto clone_kernel(const Tensor& input) -> Tensor;
    auto fill_kernel(const Tensor& input, float value) -> Tensor;
    auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim) -> Tensor;
    auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim) -> Tensor;
    auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats, int64_t dim) -> Tensor;

    // Indexing
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto scatter_reduce_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src, const std::string& reduce, bool include_self) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;
    auto slice_kernel(const Tensor& input, int64_t dim, int64_t start, int64_t end, int64_t step) -> Tensor;
    auto slice_multi_kernel(const Tensor& input, const std::vector<int64_t>& starts, const std::vector<int64_t>& ends, const std::vector<int64_t>& steps) -> Tensor;
    auto cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) -> Tensor;
    auto searchsorted_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor;

    // Normalization
    auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor>;
    auto batchnorm2d_forward_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon) -> Tensor;
    auto batchnorm2d_forward_affine_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon) -> Tensor;
    auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum) -> void;
    auto batchnorm2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon) -> std::vector<Tensor>;
    auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto layer_norm_kernel_with_stats(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> std::tuple<Tensor, Tensor, Tensor>;
    auto layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& mean, const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor>;
    auto group_norm_kernel(const Tensor& input, int64_t num_groups, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto group_norm_kernel_with_stats(const Tensor& input, int64_t num_groups, const Tensor& weight, const Tensor& bias, float eps) -> std::vector<Tensor>;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, int64_t num_groups, const Tensor& mean, const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor>;
    auto instance_norm_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto instance_norm_kernel_with_stats(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) -> std::vector<Tensor>;
    auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor>;

    // Convolution
    // Pair (per-axis) overloads used for rectangular configs.
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;
    auto conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;
    auto conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;
    // Scalar (isotropic) backward-compat wrappers.
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    // Audit I5: per-axis overload.
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t sH, int64_t sW, int64_t pH, int64_t pW, int64_t opH, int64_t opW, int64_t dH, int64_t dW, int64_t groups) -> Tensor;
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation) -> Tensor;
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride_h, int64_t stride_w,
                                 int64_t padding_h, int64_t padding_w,
                                 int64_t dilation_h, int64_t dilation_w) -> Tensor;
    // S18: depthwise Conv1d and Conv3d
    auto depthwise_conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride, int64_t padding, int64_t dilation) -> Tensor;
    auto depthwise_conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t sD, int64_t sH, int64_t sW,
                                 int64_t pD, int64_t pH, int64_t pW,
                                 int64_t dD, int64_t dH, int64_t dW) -> Tensor;

    // Deformable Conv2d (DCNv2)
    auto deformable_conv2d_forward_kernel(const Tensor& input, const Tensor& offset, const Tensor& weight, const Tensor& bias, const Tensor& mask, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups, int64_t offset_groups) -> Tensor;
    auto deformable_conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& offset, const Tensor& weight, const Tensor& mask, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups, int64_t offset_groups) -> std::vector<Tensor>;
    auto deformable_conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& offset, const Tensor& mask, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups, int64_t offset_groups, const std::vector<int64_t>& weight_shape) -> Tensor;

    // Conv3d (audit I5: per-axis overloads added; scalar overloads delegate)
    auto conv3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW, int64_t groups) -> Tensor;
    auto conv3d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv3d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW, int64_t groups) -> Tensor;
    auto conv3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW, int64_t groups) -> Tensor;
    auto conv3d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;

    // ConvTranspose3d
    auto conv_transpose3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv_transpose3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t opD, int64_t opH, int64_t opW, int64_t dD, int64_t dH, int64_t dW, int64_t groups) -> Tensor;
    auto conv_transpose3d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv_transpose3d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t opD, int64_t opH, int64_t opW, int64_t dD, int64_t dH, int64_t dW, int64_t groups) -> Tensor;
    auto conv_transpose3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv_transpose3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t opD, int64_t opH, int64_t opW, int64_t dD, int64_t dH, int64_t dW, int64_t groups) -> Tensor;

    // Pooling
    auto maxpool2d_forward_kernel(const Tensor& input,
                                   std::array<int64_t, 2> kernel_size,
                                   std::array<int64_t, 2> stride,
                                   std::array<int64_t, 2> padding,
                                   std::array<int64_t, 2> dilation) -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool2d_forward_kernel(const Tensor& input,
                                   std::array<int64_t, 2> kernel_size,
                                   std::array<int64_t, 2> stride,
                                   std::array<int64_t, 2> padding,
                                   bool count_include_pad) -> Tensor;
    auto avgpool2d_backward_kernel(const Tensor& grad_output,
                                    const std::vector<int64_t>& input_shape,
                                    std::array<int64_t, 2> kernel_size,
                                    std::array<int64_t, 2> stride,
                                    std::array<int64_t, 2> padding,
                                    bool count_include_pad) -> Tensor;
    auto adaptive_avgpool2d_kernel(const Tensor& input, int64_t output_h, int64_t output_w) -> Tensor;
    auto adaptive_avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_maxpool2d_kernel(const Tensor& input, int64_t output_h, int64_t output_w) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;

    // 1D Pooling
    auto maxpool1d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation = 1) -> std::pair<Tensor, Tensor>;
    auto maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool1d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad) -> Tensor;
    auto avgpool1d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad) -> Tensor;
    auto adaptive_avgpool1d_kernel(const Tensor& input, int64_t output_size) -> Tensor;
    auto adaptive_avgpool1d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_maxpool1d_kernel(const Tensor& input, int64_t output_size) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;

    // 3D Pooling
    auto maxpool3d_forward_kernel(const Tensor& input,
                                   std::array<int64_t, 3> kernel_size,
                                   std::array<int64_t, 3> stride,
                                   std::array<int64_t, 3> padding,
                                   std::array<int64_t, 3> dilation) -> std::pair<Tensor, Tensor>;
    auto maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool3d_forward_kernel(const Tensor& input,
                                   std::array<int64_t, 3> kernel_size,
                                   std::array<int64_t, 3> stride,
                                   std::array<int64_t, 3> padding,
                                   bool count_include_pad) -> Tensor;
    auto avgpool3d_backward_kernel(const Tensor& grad_output,
                                    const std::vector<int64_t>& input_shape,
                                    std::array<int64_t, 3> kernel_size,
                                    std::array<int64_t, 3> stride,
                                    std::array<int64_t, 3> padding,
                                    bool count_include_pad) -> Tensor;
    auto adaptive_maxpool3d_kernel(const Tensor& input, int64_t output_d, int64_t output_h, int64_t output_w) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_avgpool3d_kernel(const Tensor& input, int64_t output_d, int64_t output_h, int64_t output_w) -> Tensor;
    auto adaptive_avgpool3d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;

    // Phase 9 Pooling: Fractional Max Pool + Max Unpool
    auto fractional_maxpool2d_forward_kernel(const Tensor& input, int64_t out_h, int64_t out_w, const Tensor* random_samples) -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto fractional_maxpool3d_forward_kernel(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w, const Tensor* random_samples) -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices, int64_t out_h, int64_t out_w) -> Tensor;
    auto max_unpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices, int64_t out_d, int64_t out_h, int64_t out_w) -> Tensor;
    auto max_unpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto max_unpool1d_forward_kernel(const Tensor& input, const Tensor& indices, int64_t out_l) -> Tensor;
    auto max_unpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;

    // Fused operations
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_conv2d_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    // Per-axis overload (audit F.11).
    auto fused_conv2d_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets, bool compute_grad, const std::string& reduction = "mean") -> std::vector<Tensor>;
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> std::tuple<Tensor, Tensor, Tensor>;
    auto fused_layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& mean, const Tensor& inv_std, const Tensor& weight) -> std::vector<Tensor>;
    auto fused_conv2d_bn_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor& conv_bias, const Tensor& bn_gamma, const Tensor& bn_beta, const Tensor& bn_running_mean, const Tensor& bn_running_var, int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w, int64_t dilation_h, int64_t dilation_w, float bn_momentum, float bn_eps, bool training) -> Tensor;

    // Creation
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, double value, DType dtype, const Device& device) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto arange_kernel(double start, double end, double step, DType dtype, const Device& device) -> Tensor;
    auto linspace_kernel(double start, double end, int64_t steps, DType dtype, const Device& device) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, DType dtype, const Device& device) -> Tensor;

    // RNN - Cell operations
    auto lstm_cell_forward_kernel(const Tensor& input, const Tensor& hx, const Tensor& cx,
                                   const Tensor& weight_ih, const Tensor& weight_hh,
                                   const Tensor& bias_ih, const Tensor& bias_hh) -> std::vector<Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_hy, const Tensor& grad_cy,
                                    const Tensor& input, const Tensor& hx, const Tensor& cx,
                                    const Tensor& hy, const Tensor& cy,
                                    const Tensor& weight_ih, const Tensor& weight_hh,
                                    const Tensor& bias_ih, const Tensor& bias_hh) -> std::vector<Tensor>;
    auto gru_cell_forward_kernel(const Tensor& input, const Tensor& hx,
                                  const Tensor& weight_ih, const Tensor& weight_hh,
                                  const Tensor& bias_ih, const Tensor& bias_hh) -> Tensor;
    auto gru_cell_backward_kernel(const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
                                   const Tensor& weight_ih, const Tensor& weight_hh,
                                   const Tensor& bias_ih, const Tensor& bias_hh) -> std::vector<Tensor>;

    // RNN - Full sequence operations (fused, SIMD-optimized)
    auto lstm_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                              const Tensor& bias_ih, const Tensor& bias_hh,
                              const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;
    auto gru_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                             const Tensor& bias_ih, const Tensor& bias_hh,
                             const Tensor& h0) -> std::vector<Tensor>;
    auto bilstm_forward_kernel(const Tensor& input,
                                const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
                                const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
                                const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
                                const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
                                const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;

    // RNN - Fused multi-layer operations
    auto lstm_multilayer_forward_kernel(const Tensor& input,
                                         const std::vector<Tensor>& W_ih_list,
                                         const std::vector<Tensor>& W_hh_list,
                                         const std::vector<Tensor>& bias_list,
                                         const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;
    auto gru_multilayer_forward_kernel(const Tensor& input,
                                        const std::vector<Tensor>& W_ih_list,
                                        const std::vector<Tensor>& W_hh_list,
                                        const std::vector<Tensor>& bias_list,
                                        const Tensor& h0) -> std::vector<Tensor>;

    // Embedding
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, int64_t padding_idx = -1) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices, int64_t num_embeddings) -> Tensor;
    auto embedding_bag_forward_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor;
    auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                       const Tensor& offsets, const OpAttributes& attrs) -> Tensor;

    // Linear
    auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight) -> std::vector<Tensor>;

    // Dropout
    auto dropout_kernel(const Tensor& input, float p, bool training) -> std::pair<Tensor, Tensor>;
    auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor;

    // Vision operations
    auto interpolate_kernel(const Tensor& input, const std::vector<int64_t>& size,
                            const std::string& mode, bool align_corners) -> Tensor;
    auto interpolate_backward_kernel(const Tensor& grad_output,
                                      const std::vector<int64_t>& input_size,
                                      const std::string& mode, bool align_corners) -> Tensor;
    auto roi_align_forward_kernel(const Tensor& features, const Tensor& rois,
                                   int64_t output_h, int64_t output_w,
                                   float spatial_scale, int64_t sampling_ratio,
                                   bool aligned) -> Tensor;
    auto roi_align_backward_kernel(const Tensor& grad_output, const Tensor& rois,
                                    int64_t batch_size, int64_t feat_height, int64_t feat_width,
                                    float spatial_scale, int64_t sampling_ratio,
                                    bool aligned) -> Tensor;
    auto box_iou_kernel(const Tensor& boxes1, const Tensor& boxes2, int iou_type) -> Tensor;
    auto gather_relative_position_bias_kernel(const Tensor& bias_table, const Tensor& rel_pos_index,
                                               int64_t num_positions, int64_t num_heads) -> Tensor;
    auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                            const std::string& mode, const std::string& padding_mode,
                            bool align_corners) -> Tensor;
    auto affine_grid_kernel(const Tensor& theta, const std::vector<int64_t>& size,
                            bool align_corners) -> Tensor;
    auto grid_sample_backward_kernel(const Tensor& grad_output,
                                     const Tensor& input, const Tensor& grid,
                                     const std::string& mode,
                                     const std::string& padding_mode,
                                     bool align_corners)
        -> std::pair<Tensor, Tensor>;
    auto affine_grid_backward_kernel(const Tensor& grad_grid,
                                     const std::vector<int64_t>& size,
                                     bool align_corners) -> Tensor;
    auto unfold_kernel(const Tensor& input, int64_t kernel_size,
                       int64_t stride, int64_t padding, int64_t dilation) -> Tensor;
    auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                     int64_t kernel_size, int64_t stride, int64_t padding,
                     int64_t dilation) -> Tensor;

    // Transform operations (additional)
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& target_shape) -> Tensor;
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats) -> Tensor;
    auto stack_kernel(const std::vector<Tensor>& tensors, int64_t dim) -> Tensor;
    auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim) -> std::vector<Tensor>;
    auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim) -> std::vector<Tensor>;
    auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps) -> Tensor;
    auto to_memory_format_kernel(const Tensor& input, MemoryFormat format) -> Tensor;

    // Indexing operations (additional)
    auto nonzero_kernel(const Tensor& input) -> Tensor;
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes) -> Tensor;
    auto take_kernel(const Tensor& input, const Tensor& indices) -> Tensor;
    auto put_kernel(Tensor& input, const Tensor& indices, const Tensor& source,
                    bool accumulate) -> Tensor;

    // Advanced operations
    auto topk_kernel(const Tensor& input, int64_t k, int64_t dim,
                     bool largest, bool sorted) -> std::pair<Tensor, Tensor>;
    auto sort_kernel(const Tensor& input, int64_t dim,
                     bool descending) -> std::pair<Tensor, Tensor>;
    auto cumsum_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto logcumsumexp_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto unique_kernel(const Tensor& input, bool sorted_output,
                       bool return_inverse, bool return_counts)
        -> std::tuple<Tensor, Tensor, Tensor>;

    // New reduction operations
    auto cummax_kernel(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor>;
    auto cummin_kernel(const Tensor& input, int64_t dim) -> std::pair<Tensor, Tensor>;
    auto isin_kernel(const Tensor& elements, const Tensor& test_elements) -> Tensor;
    auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim,
                         bool keepdim) -> std::pair<Tensor, Tensor>;
    auto fmax_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fmin_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto quantile_kernel(const Tensor& input, double q, int64_t dim,
                         bool keepdim) -> Tensor;
    auto nanquantile_kernel(const Tensor& input, double q, int64_t dim,
                            bool keepdim) -> Tensor;
    auto nanmedian_kernel(const Tensor& input, int64_t dim) -> Tensor;
    auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val) -> Tensor;
    auto unique_consecutive_kernel(const Tensor& input, bool return_inverse,
                                   bool return_counts)
        -> std::tuple<Tensor, Tensor, Tensor>;
    auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets,
                               const std::string& reduce, int64_t axis) -> Tensor;

    // RMSNorm operations
    auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps)
        -> std::tuple<Tensor, Tensor>;
    auto rms_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight, const Tensor& rrms)
        -> std::tuple<Tensor, Tensor>;

    // Fused Attention
    auto fused_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                                float scale, bool causal) -> Tensor;

    // Flash Attention forward (with optional fused dropout for training).
    // Per docs/internals/attention-contract.md, returns 4 tensors:
    // [output, lse_f32, philox_seed_int64, philox_offset_int64]. seed/offset
    // are empty Tensors when dropout_p == 0.
    auto flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                  float scale, bool causal,
                                  float dropout_p = 0.0f, bool is_training = false,
                                  uint64_t seed_in = 0) -> std::vector<Tensor>;

    // Flash Attention Backward. Consumes saved Philox (seed, offset) so the
    // forward's dropout mask is reproduced bit-exactly. Returns {dQ, dK, dV}.
    auto flash_attention_backward(const Tensor& dO, const Tensor& Q, const Tensor& K,
                                   const Tensor& V, const Tensor& O,
                                   float scale, bool causal,
                                   float dropout_p = 0.0f,
                                   uint64_t philox_seed = 0,
                                   uint64_t philox_offset = 0) -> std::vector<Tensor>;

    // Fused Conv2d + Activation variants
    // Scalar (back-compat) + per-axis (audit F.11) overloads.
    auto fused_conv2d_sigmoid_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                      int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_conv2d_sigmoid_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                      int64_t stride_h, int64_t stride_w,
                                      int64_t pad_h, int64_t pad_w,
                                      int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;
    auto fused_conv2d_tanh_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                   int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_conv2d_tanh_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                   int64_t stride_h, int64_t stride_w,
                                   int64_t pad_h, int64_t pad_w,
                                   int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;
    auto fused_conv2d_swish_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                    int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_conv2d_swish_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                    int64_t stride_h, int64_t stride_w,
                                    int64_t pad_h, int64_t pad_w,
                                    int64_t dil_h, int64_t dil_w, int64_t groups) -> Tensor;

    // BatchNorm2d fused training
    auto batchnorm2d_fused_training_kernel(const Tensor& input, Tensor& running_mean, Tensor& running_var,
                                            const Tensor& gamma, const Tensor& beta,
                                            float momentum, float epsilon) -> std::vector<Tensor>;

    // Fused optimizer steps
    auto fused_sgd_step_kernel(Tensor& param, const Tensor& grad, Tensor* momentum_buffer,
                               float lr, float momentum, float weight_decay,
                               float dampening, bool nesterov) -> void;
    auto fused_adam_step_kernel(Tensor& param, const Tensor& grad,
                               Tensor& exp_avg, Tensor& exp_avg_sq,
                               double lr, double beta1, double beta2,
                               double eps, double weight_decay,
                               int64_t step, bool decoupled_weight_decay,
                               Tensor* max_exp_avg_sq, bool amsgrad) -> void;
    auto fused_adam_atan2_step_kernel(Tensor& param, const Tensor& grad,
                                      Tensor& exp_avg, Tensor& exp_avg_sq,
                                      Tensor* max_exp_avg_sq,
                                      float lr, float beta1, float beta2,
                                      float eps, float weight_decay,
                                      int64_t step, bool amsgrad) -> void;
    auto fused_rmsprop_step_kernel(Tensor& param, const Tensor& grad,
                                    Tensor& square_avg, Tensor* grad_avg,
                                    Tensor* momentum_buffer,
                                    float lr, float alpha, float eps,
                                    float weight_decay, float momentum,
                                    bool centered) -> void;
    auto fused_adadelta_step_kernel(Tensor& param, const Tensor& grad,
                                     Tensor& square_avg, Tensor& acc_delta,
                                     float rho, float eps, float lr,
                                     float weight_decay) -> void;
    auto fused_adagrad_step_kernel(Tensor& param, const Tensor& grad,
                                    Tensor& sum_sq, float lr, float lr_decay,
                                    float eps, float weight_decay,
                                    int64_t step) -> void;

    // FFT operations (MKL DFTI)
    auto fft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                    std::string_view norm) -> Tensor;
    auto ifft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                     std::string_view norm) -> Tensor;
    auto rfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                     std::string_view norm) -> Tensor;
    auto irfft_kernel(const Tensor& input, int64_t dim, int64_t signal_len,
                      std::string_view norm) -> Tensor;
    auto fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& signal_lengths,
                      std::string_view norm) -> Tensor;
    auto ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& signal_lengths,
                       std::string_view norm) -> Tensor;
    auto fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& signal_lengths,
                      std::string_view norm) -> Tensor;
    auto ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                       const std::vector<int64_t>& signal_lengths,
                       std::string_view norm) -> Tensor;

    // STFT/ISTFT
    auto stft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length,
                     const Tensor& window, bool center, bool normalized, bool onesided) -> Tensor;
    auto istft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length,
                      const Tensor& window, bool center, bool normalized, bool onesided,
                      int64_t length) -> Tensor;

    // CDist
    auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p) -> Tensor;

    // Multinomial / Bernoulli / Distribution Sampling
    auto multinomial_kernel(const Tensor& probs, int64_t num_samples, bool replacement) -> Tensor;
    auto bernoulli_kernel(const Tensor& probs) -> Tensor;
    auto normal_sample_kernel(const Tensor& mean, const Tensor& std) -> Tensor;
    auto poisson_sample_kernel(const Tensor& rates) -> Tensor;
    auto exponential_sample_kernel(const Tensor& rate) -> Tensor;

    // Histogram
    auto histogram_kernel(const Tensor& input, int64_t bins, double min_val, double max_val)
        -> std::pair<Tensor, Tensor>;

    // Multi-dimensional histogram
    auto histogramdd_kernel(const Tensor& input, std::vector<int64_t> bins,
                            std::vector<std::pair<double,double>> ranges,
                            bool density) -> std::pair<Tensor, std::vector<Tensor>>;

    // Bucketize
    auto bucketize_kernel(const Tensor& input, const Tensor& boundaries, bool right) -> Tensor;

    // Ndtr / LogNdtr / Multigammaln
    auto ndtr_kernel(const Tensor& input) -> Tensor;
    auto log_ndtr_kernel(const Tensor& input) -> Tensor;
    auto multigammaln_kernel(const Tensor& input, int64_t d) -> Tensor;

    // Nested tensor kernels
    auto nested_softmax_kernel(const Tensor& values, const Tensor& offsets, int64_t dim) -> Tensor;
    auto nested_log_softmax_kernel(const Tensor& values, const Tensor& offsets, int64_t dim) -> Tensor;
    auto nested_layer_norm_kernel(const Tensor& values, const Tensor& offsets, const Tensor& weight, const Tensor& bias, double eps) -> Tensor;
    auto nested_sum_kernel(const Tensor& values, const Tensor& offsets, int64_t dim, bool keepdim) -> Tensor;
    auto nested_mean_kernel(const Tensor& values, const Tensor& offsets, int64_t dim, bool keepdim) -> Tensor;
    auto nested_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& q_offsets, const Tensor& kv_offsets, float scale, bool causal) -> Tensor;
    auto nested_to_padded_kernel(const Tensor& values, const Tensor& offsets, int64_t max_len, float padding_value) -> Tensor;
    auto nested_from_padded_kernel(const Tensor& padded, const Tensor& offsets) -> Tensor;
    auto nested_linear_kernel(const Tensor& values, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto nested_attention_backward_kernel(const Tensor& grad_out, const Tensor& Q, const Tensor& K, const Tensor& V,
                                           const Tensor& attn_out, const Tensor& q_offsets, const Tensor& kv_offsets,
                                           float scale, bool causal) -> std::vector<Tensor>;

    // CTC loss
    auto ctc_loss_forward_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> std::vector<Tensor>;
} // namespace cpu

// Forward declarations for quantized kernels (in nn::quantization::kernels namespace)
namespace nn::quantization::kernels {
    auto quantized_linear_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, float weight_scale, float output_scale,
        int32_t input_zp, int32_t weight_zp
    ) -> void;

    auto quantized_conv2d_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_size, int64_t stride, int64_t padding,
        float input_scale, float weight_scale, int32_t input_zp, int32_t weight_zp,
        int64_t dilation, int64_t groups
    ) -> void;

    // Per-channel quantized variants
    auto quantized_linear_per_channel_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, const float* weight_scales, float output_scale,
        int32_t input_zp, const int32_t* weight_zps
    ) -> void;

    auto quantized_conv2d_per_channel_kernel(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_size, int64_t stride, int64_t padding,
        float input_scale, const float* weight_scales, int32_t input_zp, const int32_t* weight_zps,
        int64_t dilation, int64_t groups
    ) -> void;
} // namespace nn::quantization::kernels

/**
 * @brief Register CPU kernels for the "arithmetic" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_arithmetic(BackendDispatchTable& table) {
    // =========================================================================
    // Arithmetic Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Add, cpu::add_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Sub, cpu::sub_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Mul, cpu::mul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Div, cpu::div_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, MatMul, cpu::matmul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Bmm, cpu::bmm_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Dot, cpu::dot_kernel);

    // Inplace operations
    TENZOR_REGISTER_INPLACE_KERNEL(table, AddInplace, cpu::add_inplace_kernel);
    TENZOR_REGISTER_INPLACE_KERNEL(table, SubInplace, cpu::sub_inplace_kernel);
    TENZOR_REGISTER_INPLACE_KERNEL(table, MulInplace, cpu::mul_inplace_kernel);
    TENZOR_REGISTER_INPLACE_KERNEL(table, DivInplace, cpu::div_inplace_kernel);

    // Inplace activation operations
    table.register_inplace_kernel(OpId::ReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        cpu::relu_inplace_kernel(target);
        return target;
    });

    table.register_inplace_kernel(OpId::SigmoidInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        cpu::sigmoid_inplace_kernel(target);
        return target;
    });

    table.register_inplace_kernel(OpId::TanhInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        cpu::tanh_inplace_kernel(target);
        return target;
    });

    table.register_inplace_kernel(OpId::LeakyReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);
        cpu::leaky_relu_inplace_kernel(target, alpha);
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        cpu::gelu_inplace_kernel(target);
        return target;
    });

} // register_cpu_kernels_arithmetic

/**
 * @brief Register CPU kernels for the "reduction" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_reduction(BackendDispatchTable& table) {
    // =========================================================================
    // Reduction Operations
    // =========================================================================
    TENZOR_REGISTER_REDUCTION_KERNEL(table, Sum, cpu::sum_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, Mean, cpu::mean_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, Max, cpu::max_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, Min, cpu::min_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, ArgMax, cpu::argmax_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, ArgMin, cpu::argmin_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, Prod, cpu::prod_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, Any, cpu::any_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, All, cpu::all_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, HasInfNan, cpu::has_inf_nan_kernel);
    TENZOR_REGISTER_REDUCTION_KERNEL(table, LogSumExp, cpu::logsumexp_kernel);

    table.register_kernel(OpId::Median, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cpu::median_kernel(inputs[0], dim, keepdim);
    });

    table.register_kernel(OpId::Mode, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cpu::mode_kernel(inputs[0], dim, keepdim);
    });

    // Use INT64_MIN as sentinel for "reduce all dimensions" (no dim specified)
    table.register_single_output_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return cpu::var_kernel(inputs[0], dim, keepdim, correction);
    });

    table.register_single_output_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return cpu::std_kernel(inputs[0], dim, keepdim, correction);
    });

    table.register_single_output_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cpu::norm_kernel(inputs[0], p, dim, keepdim);
    });

    table.register_single_output_kernel(OpId::ArgSort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        return cpu::argsort_kernel(inputs[0], dim, descending);
    });

} // register_cpu_kernels_reduction

/**
 * @brief Register CPU kernels for the "elementwise_math" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_elementwise_math(BackendDispatchTable& table) {
    // =========================================================================
    // Element-wise Math Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sqrt, cpu::sqrt_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Neg, cpu::neg_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Abs, cpu::abs_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sign, cpu::sign_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log, cpu::log_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Exp, cpu::exp_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Reciprocal, cpu::reciprocal_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Floor, cpu::floor_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Ceil, cpu::ceil_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Round, cpu::round_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Trunc, cpu::trunc_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Frac, cpu::frac_kernel);

    table.register_single_output_kernel(OpId::Heaviside,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return cpu::heaviside_kernel(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::NanToNum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double nan_val = attrs.get_float(AttrKey::NanValue, 0.0);
            double posinf = attrs.get_float(AttrKey::PosInfValue, std::numeric_limits<double>::max());
            double neginf = attrs.get_float(AttrKey::NegInfValue, std::numeric_limits<double>::lowest());
            return cpu::nan_to_num_kernel(inputs[0], nan_val, posinf, neginf);
        });

    // LogSigmoid activation
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, LogSigmoid, cpu::log_sigmoid_kernel);
    table.register_single_output_kernel(OpId::LogSigmoidBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return cpu::log_sigmoid_backward_kernel(inputs[0], inputs[1]);
        });

    // RReLU activation
    table.register_single_output_kernel(OpId::RReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
            float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
            bool training = attrs.get_bool(AttrKey::Training, false);
            return cpu::rrelu_kernel(inputs[0], lower, upper, training);
        });
    table.register_single_output_kernel(OpId::RReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
            float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
            return cpu::rrelu_backward_kernel(inputs[0], inputs[1], lower, upper);
        });

    // Bitwise operations
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseAnd, cpu::bitwise_and_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseOr, cpu::bitwise_or_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseXor, cpu::bitwise_xor_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BitwiseNot, cpu::bitwise_not_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseLeftShift, cpu::bitwise_left_shift_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, BitwiseRightShift, cpu::bitwise_right_shift_kernel);

    // NaN-aware reductions
    table.register_single_output_kernel(OpId::CountNonzero,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return cpu::count_nonzero_kernel(inputs[0], dim);
        });
    table.register_single_output_kernel(OpId::Nansum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return cpu::nansum_kernel(inputs[0], dim, keepdim);
        });
    table.register_single_output_kernel(OpId::Nanmean,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return cpu::nanmean_kernel(inputs[0], dim, keepdim);
        });
    table.register_kernel(OpId::Aminmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            auto [mn, mx] = cpu::aminmax_kernel(inputs[0], dim, keepdim);
            return {mn, mx};
        });

    // Scatter variants: IndexAdd, IndexCopy, IndexFill
    table.register_single_output_kernel(OpId::IndexAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return cpu::index_add_kernel(inputs[0], dim, inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::IndexCopy,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return cpu::index_copy_kernel(inputs[0], dim, inputs[1], inputs[2]);
        });
    table.register_single_output_kernel(OpId::IndexFill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            double value = attrs.get_float(AttrKey::Value, 0.0);
            return cpu::index_fill_kernel(inputs[0], dim, inputs[1], value);
        });

    // SelectScatter: clone input, then copy src into the selected slice
    table.register_single_output_kernel(OpId::SelectScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src = inputs[1];
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t index = attrs.get_int(AttrKey::Index, 0);

            auto output = input.clone();
            int64_t ndim = static_cast<int64_t>(output.shape().size());
            if (dim < 0) dim += ndim;

            // select(dim, index) gives a view with dim removed;
            // we use slice(dim, index, index+1, 1) to keep the dimension,
            // then copy src (unsqueezed) into it.
            auto dst_slice = output.slice(dim, index, index + 1, 1);

            // src has the shape of output with dim removed; we need to
            // unsqueeze it back to match dst_slice shape
            auto dst_sh = dst_slice.shape();
            auto src_reshaped = src.reshape(std::vector<int64_t>(dst_sh.begin(), dst_sh.end())).contiguous();

            // Byte-level copy using element size (works for all dtypes)
            auto n = dst_slice.numel();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* dst_ptr = static_cast<char*>(dst_slice.data_ptr());
            const auto* src_ptr = static_cast<const char*>(src_reshaped.data_ptr());
            if (dst_slice.is_contiguous()) {
                std::memcpy(dst_ptr, src_ptr, n * elem_size);
            } else {
                auto dst_shape_v = dst_slice.shape();
                auto dst_strides = dst_slice.strides();
                int64_t ndims = static_cast<int64_t>(dst_shape_v.size());
                std::vector<int64_t> coord(ndims, 0);
                for (int64_t i = 0; i < n; i++) {
                    int64_t byte_offset = 0;
                    for (int64_t d = 0; d < ndims; d++) {
                        byte_offset += coord[d] * dst_strides[d] * elem_size;
                    }
                    std::memcpy(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size);
                    for (int64_t d = ndims - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < dst_shape_v[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return output;
        });

    // SliceScatter: clone input, then copy src into the sliced region
    table.register_single_output_kernel(OpId::SliceScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src = inputs[1];
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t start = attrs.get_int(AttrKey::Start, 0);
            int64_t end = attrs.get_int(AttrKey::End, -1);
            int64_t step = attrs.get_int(AttrKey::Step, 1);

            auto output = input.clone();
            int64_t ndim = static_cast<int64_t>(output.shape().size());
            if (dim < 0) dim += ndim;
            int64_t dim_size = output.shape()[dim];

            // Normalize negative indices
            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size + 1; // -1 means end of dim
            if (start < 0) start = 0;
            if (end > dim_size) end = dim_size;

            auto dst_slice = output.slice(dim, start, end, step);
            auto dst_sh = dst_slice.shape();
            auto src_reshaped = src.reshape(std::vector<int64_t>(dst_sh.begin(), dst_sh.end())).contiguous();

            auto n = dst_slice.numel();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* dst_ptr = static_cast<char*>(dst_slice.data_ptr());
            const auto* src_ptr = static_cast<const char*>(src_reshaped.data_ptr());
            if (dst_slice.is_contiguous()) {
                std::memcpy(dst_ptr, src_ptr, n * elem_size);
            } else {
                auto dst_shape_v = dst_slice.shape();
                auto dst_strides = dst_slice.strides();
                int64_t ndims = static_cast<int64_t>(dst_shape_v.size());
                std::vector<int64_t> coord(ndims, 0);
                for (int64_t i = 0; i < n; i++) {
                    int64_t byte_offset = 0;
                    for (int64_t d = 0; d < ndims; d++) {
                        byte_offset += coord[d] * dst_strides[d] * elem_size;
                    }
                    std::memcpy(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size);
                    for (int64_t d = ndims - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < dst_shape_v[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return output;
        });

    // DiagonalScatter: clone input, place src values along the diagonal
    table.register_single_output_kernel(OpId::DiagonalScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src_in = inputs[1];
            int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim1, 0);
            int64_t dim2 = attrs.get_int(AttrKey::Dim2, 1);

            auto output = input.clone();
            int64_t ndim = static_cast<int64_t>(output.shape().size());
            if (dim1 < 0) dim1 += ndim;
            if (dim2 < 0) dim2 += ndim;

            auto shape = output.shape();
            int64_t size1 = shape[dim1];
            int64_t size2 = shape[dim2];

            // Compute diagonal length
            int64_t diag_len;
            if (offset >= 0) {
                diag_len = std::min(size1, size2 - offset);
            } else {
                diag_len = std::min(size1 + offset, size2);
            }
            if (diag_len <= 0) return output;

            auto strides = output.strides();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* out_ptr = static_cast<char*>(output.data_ptr());
            auto src = src_in.contiguous();
            const auto* src_ptr = static_cast<const char*>(src.data_ptr());

            // For a general ndim tensor, iterate over all "batch" dimensions
            // (everything except dim1 and dim2), and for each batch position
            // walk the diagonal.
            int64_t batch_size = 1;
            std::vector<int64_t> batch_dims;
            for (int64_t d = 0; d < ndim; d++) {
                if (d != dim1 && d != dim2) {
                    batch_dims.push_back(d);
                    batch_size *= shape[d];
                }
            }

            std::vector<int64_t> batch_coord(batch_dims.size(), 0);
            for (int64_t b = 0; b < batch_size; b++) {
                // Compute base offset from batch coordinates
                int64_t base = 0;
                for (size_t i = 0; i < batch_dims.size(); i++) {
                    base += batch_coord[i] * strides[batch_dims[i]];
                }

                // Walk diagonal
                int64_t r0 = (offset >= 0) ? 0 : -offset;
                int64_t c0 = (offset >= 0) ? offset : 0;
                for (int64_t k = 0; k < diag_len; k++) {
                    int64_t out_elem_offset = base + (r0 + k) * strides[dim1] + (c0 + k) * strides[dim2];
                    int64_t src_elem_idx = b * diag_len + k;
                    std::memcpy(out_ptr + out_elem_offset * elem_size,
                                src_ptr + src_elem_idx * elem_size, elem_size);
                }

                // Increment batch coordinates
                for (int64_t i = static_cast<int64_t>(batch_dims.size()) - 1; i >= 0; i--) {
                    batch_coord[i]++;
                    if (batch_coord[i] < shape[batch_dims[i]]) break;
                    batch_coord[i] = 0;
                }
            }
            return output;
        });

    table.register_single_output_kernel(OpId::Cast, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        if (!attrs.has(AttrKey::TargetDtype)) {
            throw std::runtime_error("cast: missing 'target_dtype' attribute");
        }
        DType target_dtype = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
        return inputs[0].to(target_dtype);
    });

    table.register_single_output_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double exponent = attrs.get_float(AttrKey::Exponent, 2.0);
        return cpu::pow_kernel(inputs[0], exponent);
    });

    table.register_single_output_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double min_val = attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity());
        double max_val = attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity());
        return cpu::clamp_kernel(inputs[0], min_val, max_val);
    });

    table.register_single_output_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double min_val = attrs.get_float(AttrKey::Min, 0.0);
        return cpu::clamp_min_kernel(inputs[0], min_val);
    });

    table.register_single_output_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double max_val = attrs.get_float(AttrKey::Max, 0.0);
        return cpu::clamp_max_kernel(inputs[0], max_val);
    });

} // register_cpu_kernels_elementwise_math

/**
 * @brief Register CPU kernels for the "trigonometric" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_trigonometric(BackendDispatchTable& table) {
    // =========================================================================
    // Trigonometric Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sin, cpu::sin_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Cos, cpu::cos_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Tan, cpu::tan_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Asin, cpu::asin_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Acos, cpu::acos_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Atan, cpu::atan_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sinh, cpu::sinh_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Cosh, cpu::cosh_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Tanh, cpu::tanh_kernel);

} // register_cpu_kernels_trigonometric

/**
 * @brief Register CPU kernels for the "extended_math" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_extended_math(BackendDispatchTable& table) {
    // =========================================================================
    // Extended Math Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log2, cpu::log2_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log10, cpu::log10_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Log1p, cpu::log1p_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Exp2, cpu::exp2_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Expm1, cpu::expm1_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Erf, cpu::erf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Erfc, cpu::erfc_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, ErfInv, cpu::erfinv_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Gamma, cpu::gamma_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Lgamma, cpu::lgamma_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Digamma, cpu::digamma_kernel);
    table.register_single_output_kernel(OpId::Polygamma, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t n = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        return cpu::polygamma_kernel(inputs[0], n);
    });
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Beta, cpu::beta_kernel);
    table.register_kernel(OpId::BetaInc, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        return {cpu::betainc_kernel(inputs)};
    });
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BesselJ0, cpu::bessel_j0_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BesselJ1, cpu::bessel_j1_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BesselY0, cpu::bessel_y0_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BesselY1, cpu::bessel_y1_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BesselI0, cpu::bessel_i0_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, BesselI1, cpu::bessel_i1_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sinc, cpu::sinc_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Zeta, cpu::zeta_kernel);

    // Ndtr / LogNdtr / Multigammaln
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Ndtr, cpu::ndtr_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, LogNdtr, cpu::log_ndtr_kernel);
    table.register_single_output_kernel(OpId::Multigammaln, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t d = attrs.get_int(AttrKey::Dim, 1);
        return cpu::multigammaln_kernel(inputs[0], d);
    });

    // LinalgVectorNorm: delegates to existing Norm kernel
    table.register_single_output_kernel(OpId::LinalgVectorNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cpu::norm_kernel(inputs[0], p, dim, keepdim);
    });

    // LinalgMatrixNorm: Frobenius (ord=0), nuclear (ord=1), spectral (ord=2)
    table.register_single_output_kernel(OpId::LinalgMatrixNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ord = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        if (ord == 0) {
            // Frobenius norm: sqrt(sum(x^2))
            return cpu::norm_kernel(inputs[0], 2.0f, INT64_MIN, false);
        }
        // Nuclear (ord==1) or Spectral (ord==2): use SVD
        auto svd_result = linalg::svd(inputs[0], /*full_matrices=*/false);
        const auto& S = std::get<1>(svd_result);
        if (ord == 1) {
            return cpu::sum_kernel(S, INT64_MIN, false);
        }
        return cpu::max_kernel(S, INT64_MIN, false);
    });

    // LinalgVecdot: sum(a * b, dim)
    table.register_single_output_kernel(OpId::LinalgVecdot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        Tensor product = cpu::mul_kernel(inputs[0], inputs[1]);
        return cpu::sum_kernel(product, dim, false);
    });

    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsNan, cpu::isnan_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsInf, cpu::isinf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsFinite, cpu::isfinite_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Atan2, cpu::atan2_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Fmod, cpu::fmod_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Remainder, cpu::remainder_kernel);

    table.register_kernel(OpId::Lerp, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        return {cpu::lerp_kernel(inputs)};
    });

    // =========================================================================
    // New Element-wise Math Operations
    // =========================================================================

    // Unary ops
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Rsqrt, cpu::rsqrt_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Square, cpu::square_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Asinh, cpu::asinh_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Acosh, cpu::acosh_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Atanh, cpu::atanh_kernel);

    // Binary floating-point ops
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Hypot, cpu::hypot_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Copysign, cpu::copysign_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Nextafter, cpu::nextafter_kernel);

    // Binary integer ops
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Gcd, cpu::gcd_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Lcm, cpu::lcm_kernel);

    // Igamma / Igammac (regularized incomplete gamma)
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Igamma, cpu::igamma_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Igammac, cpu::igammac_kernel);

    // Ternary ops with alpha attribute
    table.register_single_output_kernel(OpId::Addcmul,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            return cpu::addcmul_kernel(inputs[0], inputs[1], inputs[2], alpha);
        });

    table.register_single_output_kernel(OpId::Addcdiv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            return cpu::addcdiv_kernel(inputs[0], inputs[1], inputs[2], alpha);
        });

} // register_cpu_kernels_extended_math

/**
 * @brief Register CPU kernels for the "logical" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_logical(BackendDispatchTable& table) {
    // =========================================================================
    // Logical Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogicalAnd, cpu::logical_and_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogicalOr, cpu::logical_or_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, LogicalNot, cpu::logical_not_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogicalXor, cpu::logical_xor_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Minimum, cpu::minimum_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Maximum, cpu::maximum_kernel);

    table.register_single_output_kernel(OpId::Cross, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cpu::cross_kernel(inputs[0], inputs[1], dim);
    });

} // register_cpu_kernels_logical

/**
 * @brief Register CPU kernels for the "comparison" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_comparison(BackendDispatchTable& table) {
    // =========================================================================
    // Comparison Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Eq, cpu::eq_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Ne, cpu::ne_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Lt, cpu::lt_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Le, cpu::le_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Gt, cpu::gt_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Ge, cpu::ge_kernel);

    // =========================================================================
    // Activation Functions
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, ReLU, cpu::relu_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, ReLUBackward, cpu::relu_backward_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Sigmoid, cpu::sigmoid_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, SigmoidBackward, cpu::sigmoid_backward_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, TanhActivation, cpu::tanh_kernel);  // Uses same kernel as Tanh
    TENZOR_REGISTER_BINARY_KERNEL(table, TanhBackward, cpu::tanh_backward_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Gelu, cpu::gelu_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, GeluBackward, cpu::gelu_backward_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Swish, cpu::swish_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, SwishBackward, cpu::swish_backward_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Selu, cpu::selu_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, SeluBackward, cpu::selu_backward_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Mish, cpu::mish_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, MishBackward, cpu::mish_backward_kernel);
    // C.3 audit batch 3: Hardswish / Hardsigmoid CPU kernels (backward
    // remains autograd-composed via clamp + mul chain; no dedicated
    // *Backward OpId — gradients flow through the composition's grad_fn).
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Hardswish, cpu::hardswish_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Hardsigmoid, cpu::hardsigmoid_kernel);

    table.register_single_output_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // alpha kept as double so Float64 inputs preserve precision; the
        // kernel converts down for Float32/Float16/BFloat16 paths.
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);
        return cpu::leaky_relu_kernel(inputs[0], alpha);
    });

    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);
        return cpu::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha);
    });

    table.register_single_output_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return cpu::elu_kernel(inputs[0], alpha);
    });

    table.register_single_output_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return cpu::elu_backward_kernel(inputs[0], inputs[1], alpha);
    });

    table.register_single_output_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
        return cpu::softplus_kernel(inputs[0], beta, threshold);
    });

    table.register_single_output_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
        return cpu::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold);
    });

    table.register_single_output_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cpu::softmax_kernel(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cpu::softmax_backward_kernel(inputs[0], inputs[1], dim);
    });

    table.register_single_output_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cpu::log_softmax_kernel(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cpu::log_softmax_backward_kernel(inputs[0], inputs[1], dim);
    });

} // register_cpu_kernels_comparison

/**
 * @brief Register CPU kernels for the "shape" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_shape(BackendDispatchTable& table) {
    // =========================================================================
    // Shape/View Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        return cpu::reshape_kernel(inputs[0], shape);
    });

    table.register_single_output_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
        return cpu::transpose_kernel(inputs[0], dim0, dim1);
    });

    table.register_single_output_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        return cpu::permute_kernel(inputs[0], dims);
    });

    table.register_single_output_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cpu::squeeze_kernel(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::unsqueeze_kernel(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
        int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
        return cpu::flatten_kernel(inputs[0], start_dim, end_dim);
    });

    TENZOR_REGISTER_UNARY_KERNEL(table, Contiguous, cpu::contiguous_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Clone, cpu::clone_kernel);

    table.register_single_output_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        return cpu::fill_kernel(inputs[0], value);
    });

} // register_cpu_kernels_shape

/**
 * @brief Register CPU kernels for the "indexing" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_indexing(BackendDispatchTable& table) {
    // =========================================================================
    // Indexing Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::index_select_kernel(inputs[0], dim, inputs[1]);
    });

    table.register_single_output_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::gather_kernel(inputs[0], dim, inputs[1]);
    });

    table.register_single_output_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::scatter_kernel(inputs[0], dim, inputs[1], inputs[2]);
    });

    table.register_single_output_kernel(OpId::ScatterAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::scatter_add_kernel(inputs[0], dim, inputs[1], inputs[2]);
    });

    table.register_single_output_kernel(OpId::ScatterReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        bool include_self = attrs.get_bool(AttrKey::IncludeSelf, true);
        return cpu::scatter_reduce_kernel(inputs[0], dim, inputs[1], inputs[2], reduce, include_self);
    });

    TENZOR_REGISTER_BINARY_KERNEL(table, MaskedSelect, cpu::masked_select_kernel);

    table.register_single_output_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double value = attrs.get_float(AttrKey::Value, 0.0);
        return cpu::masked_fill_kernel(inputs[0], inputs[1], static_cast<float>(value));
    });

    TENZOR_REGISTER_TERNARY_KERNEL(table, Where, cpu::where_kernel);

    table.register_single_output_kernel(OpId::SearchSorted, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cpu::searchsorted_kernel(inputs, attrs);
    });

    table.register_single_output_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Support both multi-dim format (CUDA-compatible) and single-dim format
        if (attrs.has(AttrKey::Starts)) {
            auto starts = attrs.get_int_list(AttrKey::Starts);
            auto ends = attrs.get_int_list(AttrKey::Ends);
            auto steps = attrs.get_int_list(AttrKey::Steps);
            return cpu::slice_multi_kernel(inputs[0], starts, ends, steps);
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        int64_t start = attrs.get_int(AttrKey::Start, 0);
        int64_t end = attrs.get_int(AttrKey::End, -1);
        int64_t step = attrs.get_int(AttrKey::Step, 1);
        return cpu::slice_kernel(inputs[0], dim, start, end, step);
    });

    table.register_single_output_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return cpu::cat_kernel(tensors, dim);
    });

    table.register_single_output_kernel(OpId::Roll, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t shift = attrs.get_int(AttrKey::Shift, 0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::roll_kernel(inputs[0], shift, dim);
    });

    table.register_single_output_kernel(OpId::RepeatInterleave, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        int64_t num_repeats = attrs.get_int(AttrKey::NumRepeats, 1);
        if (num_repeats >= 0) {
            // Scalar repeats mode
            return cpu::repeat_interleave_scalar_kernel(inputs[0], num_repeats, dim);
        } else {
            // Tensor repeats mode — repeats tensor is inputs[1]
            return cpu::repeat_interleave_tensor_kernel(inputs[0], inputs[1], dim);
        }
    });

} // register_cpu_kernels_indexing

/**
 * @brief Register CPU kernels for the "normalization" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_normalization(BackendDispatchTable& table) {
    // =========================================================================
    // Normalization Operations
    // =========================================================================
    table.register_kernel(OpId::BatchNorm2dMeanVar, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cpu::batchnorm2d_mean_var_kernel(inputs[0]);
    });

    table.register_single_output_kernel(OpId::BatchNorm2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::batchnorm2d_forward_kernel(inputs[0], inputs[1], inputs[2], epsilon);
    });

    table.register_single_output_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::batchnorm2d_forward_affine_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon);
    });

    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::batchnorm2d_backward_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon);
    });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        // inputs: running_mean, running_var, batch_mean, batch_var
        // Note: This modifies running_mean and running_var in-place
        Tensor running_mean = inputs[0];  // Copy to allow modification
        Tensor running_var = inputs[1];
        cpu::batchnorm2d_update_running_stats_kernel(running_mean, running_var, inputs[2], inputs[3], momentum);
        return std::vector<Tensor>{running_mean, running_var};
    });

    // Multi-output LayerNorm: returns {output, mean, rstd} for backward pass compatibility.
    // The single-output variant was dropped (audit 8.1) — it caused a duplicate-registration
    // warning and the backward pass always needs {mean, rstd}.
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, mean, rstd] = cpu::layer_norm_kernel_with_stats(inputs[0], normalized_shape, inputs[1], inputs[2], eps);
        return std::vector<Tensor>{output, mean, rstd};
    });

    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Standard input order from LayerNormBackward dispatcher
        // (src/nn/layers/normalization.cpp:461): [grad_output, input, mean, inv_std, weight].
        // Kernel signature is (grad_out, input, normalized_shape, mean, rstd, weight).
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        return cpu::layer_norm_backward_kernel(inputs[0], inputs[1], normalized_shape, inputs[2], inputs[3], inputs[4]);
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // GroupNorm layer sets NumGroups; functional::group_norm previously set
        // Groups. Accept either to avoid silent num_groups=1 default.
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::group_norm_kernel_with_stats(inputs[0], num_groups, inputs[1], inputs[2], eps);
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Canonical input order across all backends:
        //   [grad_output, input, mean, rstd, weight]
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        return cpu::group_norm_backward_kernel(inputs[0], inputs[1], num_groups, inputs[2], inputs[3], inputs[4]);
    });

    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::instance_norm_kernel_with_stats(inputs[0], inputs[1], inputs[2], eps);
    });

    table.register_kernel(OpId::InstanceNormBackward, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, rstd]
        return cpu::instance_norm_backward_kernel(inputs[0], inputs[1], inputs[3], inputs[4], inputs[2]);
    });

} // register_cpu_kernels_normalization

/**
 * @brief Register CPU kernels for the "convolution" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_convolution(BackendDispatchTable& table) {
    // =========================================================================
    // Convolution Operations
    // =========================================================================
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Read per-axis keys when present, fall back to the scalar keys.
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
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv2d_forward_kernel(
            inputs[0], inputs[1], bias, sh, sw, ph, pw, dh, dw, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{cpu::conv2d_backward_input_kernel(
            inputs[0], inputs[2], input_shape, sh, sw, ph, pw, dh, dw, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cpu::conv2d_backward_weight_kernel(
            inputs[0], inputs[1], weight_shape, sh, sw, ph, pw, dh, dw, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv2d_backward_bias_kernel(inputs[0])};
    });

    // Conv1d: wraps Conv2d by unsqueezing [N,C,L] -> [N,C,1,L] and delegating.
    // The NN layer handles manual 1D padding before dispatching, so the kernel
    // receives pre-padded input with padding=0 in attrs.
    table.register_kernel(OpId::Conv1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_4d = inputs[0].unsqueeze(2);
        auto weight_4d = inputs[1].unsqueeze(2);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        auto result_4d = cpu::conv2d_forward_kernel(input_4d, weight_4d, bias, stride, padding, dilation, groups);
        return std::vector<Tensor>{result_4d.squeeze(2)};
    });

    // Conv1dBackwardInput: inputs = {grad_output, input, weight}
    // Delegates to conv2d_backward_input_kernel(grad, weight, input_shape, ...)
    table.register_kernel(OpId::Conv1dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto grad_4d = inputs[0].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        // Build 4D input shape from 3D: [N,C,L] -> [N,C,1,L]
        auto in_shape = inputs[1].shape();
        std::vector<int64_t> input_shape_4d = {in_shape[0], in_shape[1], 1, in_shape[2]};
        auto result_4d = cpu::conv2d_backward_input_kernel(grad_4d, weight_4d, input_shape_4d, stride, padding, dilation, groups);
        return std::vector<Tensor>{result_4d.squeeze(2)};
    });

    // Conv1dBackwardWeight: inputs = {grad_output, input, weight}
    // Delegates to conv2d_backward_weight_kernel(grad, input, weight_shape, ...)
    table.register_kernel(OpId::Conv1dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        // Build 4D weight shape from 3D: [out,in/g,kL] -> [out,in/g,1,kL]
        auto wt_shape = inputs[2].shape();
        std::vector<int64_t> weight_shape_4d = {wt_shape[0], wt_shape[1], 1, wt_shape[2]};
        auto result_4d = cpu::conv2d_backward_weight_kernel(grad_4d, input_4d, weight_shape_4d, stride, padding, dilation, groups);
        return std::vector<Tensor>{result_4d.squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv2d_backward_bias_kernel(inputs[0].unsqueeze(2))};
    });

    // Conv3d
    // Audit I5: read per-axis attrs with scalar fallback. When per-axis
    // keys are missing the scalar is used for all three axes — backward-
    // compatible with callers that haven't yet been updated.
    table.register_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sD = attrs.get_int(AttrKey::StrideD,   s);
        int64_t sH = attrs.get_int(AttrKey::StrideH,   s);
        int64_t sW = attrs.get_int(AttrKey::StrideW,   s);
        int64_t pD = attrs.get_int(AttrKey::PaddingD,  p);
        int64_t pH = attrs.get_int(AttrKey::PaddingH,  p);
        int64_t pW = attrs.get_int(AttrKey::PaddingW,  p);
        int64_t dD = attrs.get_int(AttrKey::DilationD, d);
        int64_t dH = attrs.get_int(AttrKey::DilationH, d);
        int64_t dW = attrs.get_int(AttrKey::DilationW, d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv3d_forward_kernel(
            inputs[0], inputs[1], bias, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sD = attrs.get_int(AttrKey::StrideD,   s);
        int64_t sH = attrs.get_int(AttrKey::StrideH,   s);
        int64_t sW = attrs.get_int(AttrKey::StrideW,   s);
        int64_t pD = attrs.get_int(AttrKey::PaddingD,  p);
        int64_t pH = attrs.get_int(AttrKey::PaddingH,  p);
        int64_t pW = attrs.get_int(AttrKey::PaddingW,  p);
        int64_t dD = attrs.get_int(AttrKey::DilationD, d);
        int64_t dH = attrs.get_int(AttrKey::DilationH, d);
        int64_t dW = attrs.get_int(AttrKey::DilationW, d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{cpu::conv3d_backward_input_kernel(
            inputs[0], inputs[2], input_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sD = attrs.get_int(AttrKey::StrideD,   s);
        int64_t sH = attrs.get_int(AttrKey::StrideH,   s);
        int64_t sW = attrs.get_int(AttrKey::StrideW,   s);
        int64_t pD = attrs.get_int(AttrKey::PaddingD,  p);
        int64_t pH = attrs.get_int(AttrKey::PaddingH,  p);
        int64_t pW = attrs.get_int(AttrKey::PaddingW,  p);
        int64_t dD = attrs.get_int(AttrKey::DilationD, d);
        int64_t dH = attrs.get_int(AttrKey::DilationH, d);
        int64_t dW = attrs.get_int(AttrKey::DilationW, d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cpu::conv3d_backward_weight_kernel(
            inputs[0], inputs[1], weight_shape, sD, sH, sW, pD, pH, pW, dD, dH, dW, groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv3d_backward_bias_kernel(inputs[0])};
    });

    // ConvTranspose3d
    // Audit I5: ConvT3d registrations read per-axis attrs with scalar fallback.
    // The per-axis extraction is inlined per lambda (auto-lambdas can't be
    // captured by other lambdas without messing up the std::function type).
    table.register_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t op = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sD = attrs.get_int(AttrKey::StrideD, s),  sH = attrs.get_int(AttrKey::StrideH, s),  sW = attrs.get_int(AttrKey::StrideW, s);
        int64_t pD = attrs.get_int(AttrKey::PaddingD, p), pH = attrs.get_int(AttrKey::PaddingH, p), pW = attrs.get_int(AttrKey::PaddingW, p);
        int64_t opD= attrs.get_int(AttrKey::OutputPaddingD, op), opH = attrs.get_int(AttrKey::OutputPaddingH, op), opW = attrs.get_int(AttrKey::OutputPaddingW, op);
        int64_t dD = attrs.get_int(AttrKey::DilationD, d), dH = attrs.get_int(AttrKey::DilationH, d), dW = attrs.get_int(AttrKey::DilationW, d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv_transpose3d_forward_kernel(
            inputs[0], inputs[1], bias, sD,sH,sW, pD,pH,pW, opD,opH,opW, dD,dH,dW, groups)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t op = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sD = attrs.get_int(AttrKey::StrideD, s),  sH = attrs.get_int(AttrKey::StrideH, s),  sW = attrs.get_int(AttrKey::StrideW, s);
        int64_t pD = attrs.get_int(AttrKey::PaddingD, p), pH = attrs.get_int(AttrKey::PaddingH, p), pW = attrs.get_int(AttrKey::PaddingW, p);
        int64_t opD= attrs.get_int(AttrKey::OutputPaddingD, op), opH = attrs.get_int(AttrKey::OutputPaddingH, op), opW = attrs.get_int(AttrKey::OutputPaddingW, op);
        int64_t dD = attrs.get_int(AttrKey::DilationD, d), dH = attrs.get_int(AttrKey::DilationH, d), dW = attrs.get_int(AttrKey::DilationW, d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{cpu::conv_transpose3d_backward_input_kernel(
            inputs[0], inputs[2], input_shape, sD,sH,sW, pD,pH,pW, opD,opH,opW, dD,dH,dW, groups)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t op = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sD = attrs.get_int(AttrKey::StrideD, s),  sH = attrs.get_int(AttrKey::StrideH, s),  sW = attrs.get_int(AttrKey::StrideW, s);
        int64_t pD = attrs.get_int(AttrKey::PaddingD, p), pH = attrs.get_int(AttrKey::PaddingH, p), pW = attrs.get_int(AttrKey::PaddingW, p);
        int64_t opD= attrs.get_int(AttrKey::OutputPaddingD, op), opH = attrs.get_int(AttrKey::OutputPaddingH, op), opW = attrs.get_int(AttrKey::OutputPaddingW, op);
        int64_t dD = attrs.get_int(AttrKey::DilationD, d), dH = attrs.get_int(AttrKey::DilationH, d), dW = attrs.get_int(AttrKey::DilationW, d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cpu::conv_transpose3d_backward_weight_kernel(
            inputs[0], inputs[1], weight_shape, sD,sH,sW, pD,pH,pW, opD,opH,opW, dD,dH,dW, groups)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv3d_backward_bias_kernel(inputs[0])};
    });

    table.register_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Audit I5: per-axis with scalar fallback (ConvT2d).
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t op = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sH = attrs.get_int(AttrKey::StrideH,        s);
        int64_t sW = attrs.get_int(AttrKey::StrideW,        s);
        int64_t pH = attrs.get_int(AttrKey::PaddingH,       p);
        int64_t pW = attrs.get_int(AttrKey::PaddingW,       p);
        int64_t opH= attrs.get_int(AttrKey::OutputPaddingH, op);
        int64_t opW= attrs.get_int(AttrKey::OutputPaddingW, op);
        int64_t dH = attrs.get_int(AttrKey::DilationH,      d);
        int64_t dW = attrs.get_int(AttrKey::DilationW,      d);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv_transpose2d_forward_kernel(
            inputs[0], inputs[1], bias, sH, sW, pH, pW, opH, opW, dH, dW, groups)};
    });

    table.register_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Mirror Conv2dForward's per-axis-with-scalar-fallback pattern (audit 8.5).
        // Read scalar keys as defaults, then override with per-axis keys when present.
        int64_t s  = attrs.get_int(AttrKey::Stride, 1);
        int64_t p  = attrs.get_int(AttrKey::Padding, 0);
        int64_t d  = attrs.get_int(AttrKey::Dilation, 1);
        int64_t sh = attrs.has(AttrKey::StrideH)   ? attrs.get_int(AttrKey::StrideH)   : s;
        int64_t sw = attrs.has(AttrKey::StrideW)   ? attrs.get_int(AttrKey::StrideW)   : s;
        int64_t ph = attrs.has(AttrKey::PaddingH)  ? attrs.get_int(AttrKey::PaddingH)  : p;
        int64_t pw = attrs.has(AttrKey::PaddingW)  ? attrs.get_int(AttrKey::PaddingW)  : p;
        int64_t dh = attrs.has(AttrKey::DilationH) ? attrs.get_int(AttrKey::DilationH) : d;
        int64_t dw = attrs.has(AttrKey::DilationW) ? attrs.get_int(AttrKey::DilationW) : d;
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::depthwise_conv2d_kernel(
            inputs[0], inputs[1], bias, sh, sw, ph, pw, dh, dw)};
    });

    // S18: real DepthwiseConv1d and DepthwiseConv3d kernels.
    table.register_kernel(OpId::DepthwiseConv1d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t stride   = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding  = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return std::vector<Tensor>{cpu::depthwise_conv1d_kernel(
                inputs[0], inputs[1], bias, stride, padding, dilation)};
        });
    table.register_kernel(OpId::DepthwiseConv3d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t s = attrs.get_int(AttrKey::Stride, 1);
            int64_t p = attrs.get_int(AttrKey::Padding, 0);
            int64_t d = attrs.get_int(AttrKey::Dilation, 1);
            int64_t sD = attrs.has(AttrKey::StrideD) ? attrs.get_int(AttrKey::StrideD) : s;
            int64_t sH = attrs.has(AttrKey::StrideH) ? attrs.get_int(AttrKey::StrideH) : s;
            int64_t sW = attrs.has(AttrKey::StrideW) ? attrs.get_int(AttrKey::StrideW) : s;
            int64_t pD = attrs.has(AttrKey::PaddingD) ? attrs.get_int(AttrKey::PaddingD) : p;
            int64_t pH = attrs.has(AttrKey::PaddingH) ? attrs.get_int(AttrKey::PaddingH) : p;
            int64_t pW = attrs.has(AttrKey::PaddingW) ? attrs.get_int(AttrKey::PaddingW) : p;
            int64_t dD = attrs.has(AttrKey::DilationD) ? attrs.get_int(AttrKey::DilationD) : d;
            int64_t dH = attrs.has(AttrKey::DilationH) ? attrs.get_int(AttrKey::DilationH) : d;
            int64_t dW = attrs.has(AttrKey::DilationW) ? attrs.get_int(AttrKey::DilationW) : d;
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return std::vector<Tensor>{cpu::depthwise_conv3d_kernel(
                inputs[0], inputs[1], bias, sD, sH, sW, pD, pH, pW, dD, dH, dW)};
        });

    // DeformableConv2d (DCNv2)
    table.register_kernel(OpId::DeformableConv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: {input, offset, weight, [bias], [mask]} — bias and mask are
        // optional. std::span performs NO bounds checking, so a caller that
        // omits bias/mask (e.g. DeformableConv2dBackward::forward with
        // use_mask=false) must not be indexed at [3]/[4] directly. Substitute
        // a valid 0-element tensor for any absent trailing input; the kernel
        // treats numel()==0 bias/mask as "not present".
        if (inputs.size() < 3) {
            throw std::runtime_error(
                "DeformableConv2dForward: requires at least {input, offset, weight}");
        }
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        Tensor empty_t = Tensor({0}, inputs[0].dtype(), inputs[0].device());
        const Tensor& bias = inputs.size() > 3 ? inputs[3] : empty_t;
        const Tensor& mask = inputs.size() > 4 ? inputs[4] : empty_t;
        return std::vector<Tensor>{cpu::deformable_conv2d_forward_kernel(
            inputs[0], inputs[1], inputs[2], bias, mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups)};
    });

    table.register_kernel(OpId::DeformableConv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: {grad_output, input, offset, weight, [mask]} — mask optional.
        // span has no bounds checking; substitute an empty tensor when absent.
        if (inputs.size() < 4) {
            throw std::runtime_error(
                "DeformableConv2dBackwardInput: requires {grad_output, input, offset, weight}");
        }
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        Tensor empty_t = Tensor({0}, inputs[1].dtype(), inputs[1].device());
        const Tensor& mask = inputs.size() > 4 ? inputs[4] : empty_t;
        return cpu::deformable_conv2d_backward_input_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w, groups, offset_groups);
    });

    table.register_kernel(OpId::DeformableConv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: {grad_output, input, offset, [mask]} — mask optional.
        // span has no bounds checking; substitute an empty tensor when absent.
        if (inputs.size() < 3) {
            throw std::runtime_error(
                "DeformableConv2dBackwardWeight: requires {grad_output, input, offset}");
        }
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        Tensor empty_t = Tensor({0}, inputs[1].dtype(), inputs[1].device());
        const Tensor& mask = inputs.size() > 3 ? inputs[3] : empty_t;
        return std::vector<Tensor>{cpu::deformable_conv2d_backward_weight_kernel(
            inputs[0], inputs[1], inputs[2], mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, weight_shape)};
    });

    table.register_kernel(OpId::DeformableConv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // Reuse regular conv2d bias backward (channel-wise sum of grad_output)
        return std::vector<Tensor>{cpu::conv2d_backward_bias_kernel(inputs[0])};
    });

} // register_cpu_kernels_convolution

/**
 * @brief Register CPU kernels for the "pooling" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_pooling(BackendDispatchTable& table) {
    // =========================================================================
    // Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, /*default*/ kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation    = ::tenzor::backend::attrs::dilation_2d(attrs);
        auto [output, indices] = cpu::maxpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, /*default*/ kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        const bool count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1) != 0;
        return cpu::avgpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, count_include_pad);
    });

    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, /*default*/ kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        const bool count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1) != 0;
        return cpu::avgpool2d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding, count_include_pad);
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cpu::adaptive_avgpool2d_kernel(inputs[0], output_h, output_w);
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::adaptive_avgpool2d_backward_kernel(inputs[0], input_shape);
    });

    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        auto [output, indices] = cpu::adaptive_maxpool2d_kernel(inputs[0], output_h, output_w);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::adaptive_maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // 1D Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        auto [output, indices] = cpu::maxpool1d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::maxpool1d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::AvgPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        const bool count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1) != 0;
        return cpu::avgpool1d_forward_kernel(inputs[0], kernel_size, stride, padding, count_include_pad);
    });

    table.register_single_output_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        const bool count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1) != 0;
        return cpu::avgpool1d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding, count_include_pad);
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
        return cpu::adaptive_avgpool1d_kernel(inputs[0], output_size);
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::adaptive_avgpool1d_backward_kernel(inputs[0], input_shape);
    });

    table.register_kernel(OpId::AdaptiveMaxPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
        auto [output, indices] = cpu::adaptive_maxpool1d_kernel(inputs[0], output_size);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::adaptive_maxpool1d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // 3D Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, /*default*/ kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        const auto dilation    = ::tenzor::backend::attrs::dilation_3d(attrs);
        auto [output, indices] = cpu::maxpool3d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, /*default*/ kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        const bool count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1) != 0;
        return cpu::avgpool3d_forward_kernel(inputs[0], kernel_size, stride, padding, count_include_pad);
    });

    table.register_single_output_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, /*default*/ kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        const bool count_include_pad = attrs.get_int(AttrKey::CountIncludePad, 1) != 0;
        return cpu::avgpool3d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding, count_include_pad);
    });

    table.register_kernel(OpId::AdaptiveMaxPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        auto [output, indices] = cpu::adaptive_maxpool3d_kernel(inputs[0], output_d, output_h, output_w);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::adaptive_maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cpu::adaptive_avgpool3d_kernel(inputs[0], output_d, output_h, output_w);
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::adaptive_avgpool3d_backward_kernel(inputs[0], input_shape);
    });

    // =========================================================================
    // Fused Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_linear_relu_kernel(inputs[0], inputs[1], bias);
    });

    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: read per-axis stride/padding/dilation with scalar fallback.
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_relu_kernel(inputs[0], inputs[1], bias,
                                             stride[0], stride[1],
                                             padding[0], padding[1],
                                             dilation[0], dilation[1], groups);
    });

    table.register_single_output_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::fused_batchnorm_relu_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
    });

    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool compute_grad = attrs.get_bool(AttrKey::ComputeGrad, true);
        std::string reduction = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
        return cpu::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], compute_grad, reduction);
    });

    TENZOR_REGISTER_BINARY_KERNEL(table, FusedAddReLU, cpu::fused_add_relu_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, FusedGelu, cpu::fused_gelu_kernel);

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, mean, inv_std] = cpu::fused_layer_norm_kernel(inputs[0], normalized_shape, inputs[1], inputs[2], eps);
        return std::vector<Tensor>{output, mean, inv_std};
    });

    table.register_kernel(OpId::FusedLayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        return cpu::fused_layer_norm_backward_kernel(inputs[0], inputs[1], normalized_shape, inputs[3], inputs[4], inputs[2]);
    });

    table.register_single_output_kernel(OpId::FusedConv2dBnReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        // Audit QQ.10: per-axis stride / padding / dilation with scalar fallback,
        // mirroring the Conv2dForward / FusedConv2dReLU pattern (audit F.11).
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        float bn_momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        float bn_eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        bool training = attrs.get_bool(AttrKey::Training, false);
        return cpu::fused_conv2d_bn_relu_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            bn_momentum, bn_eps, training);
    });

} // register_cpu_kernels_pooling

/**
 * @brief Register CPU kernels for the "linear_fc" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_linear_fc(BackendDispatchTable& table) {
    // =========================================================================
    // Linear/FC Operations
    // =========================================================================
    // Use single-output registration for optimized dispatch (no vector allocation)
    table.register_single_output_kernel(OpId::Linear, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::linear_kernel(inputs[0], inputs[1], bias);
    });

    table.register_kernel(OpId::LinearBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cpu::linear_backward_kernel(inputs[0], inputs[1], inputs[2]);
    });

} // register_cpu_kernels_linear_fc

/**
 * @brief Register CPU kernels for the "embedding_dropout" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_embedding_dropout(BackendDispatchTable& table) {
    // =========================================================================
    // Embedding Operations
    // =========================================================================
    // Embedding: reads optional PaddingIdx attr; the kernel zeroes that output row.
    table.register_single_output_kernel(OpId::Embedding,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t padding_idx = attrs.has(AttrKey::PaddingIdx)
                                  ? attrs.get_int(AttrKey::PaddingIdx)
                                  : -1;
            return cpu::embedding_kernel(inputs[0], inputs[1], padding_idx);
        });

    // EmbeddingWithBoundsCheck — same as Embedding since CPU already validates indices
    table.register_single_output_kernel(OpId::EmbeddingWithBoundsCheck,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t padding_idx = attrs.has(AttrKey::PaddingIdx)
                                  ? attrs.get_int(AttrKey::PaddingIdx)
                                  : -1;
            return cpu::embedding_kernel(inputs[0], inputs[1], padding_idx);
        });

    table.register_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
        return std::vector<Tensor>{cpu::embedding_backward_kernel(inputs[0], inputs[1], num_embeddings)};
    });

    table.register_single_output_kernel(OpId::EmbeddingBagForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cpu::embedding_bag_forward_kernel(inputs, attrs);
        });

    table.register_single_output_kernel(OpId::EmbeddingBagBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // inputs[0] = grad_output, inputs[1] = indices (Int64), inputs[2] = offsets
            return cpu::embedding_bag_backward_kernel(inputs[0], inputs[1], inputs[2], attrs);
        });

    // CTC loss (audit Phase 3.7): outputs [loss_per_sample, raw_grad].
    table.register_kernel(OpId::CTCLossForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
            return cpu::ctc_loss_forward_kernel(inputs, attrs);
        });

    // =========================================================================
    // Dropout Operations
    // =========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        bool training = attrs.get_bool(AttrKey::Training, true);
        auto [output, mask] = cpu::dropout_kernel(inputs[0], p, training);
        return std::vector<Tensor>{output, mask};
    });

    table.register_single_output_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        return cpu::dropout_backward_kernel(inputs[0], inputs[1], p);
    });

    // =========================================================================
    // RNN Operations
    // =========================================================================
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cpu::lstm_cell_forward_kernel(inputs[0], inputs[1], inputs[2],
                                              inputs[3], inputs[4], inputs[5], inputs[6]);
    });

    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // inputs: [grad_hy, grad_cy, input, hx, cx, hy, cy, weight_ih, weight_hh, bias_ih, bias_hh]
        // bias_ih and bias_hh are optional (may not be present for backward compatibility)
        Tensor bias_ih = inputs.size() > 9 ? inputs[9] : Tensor({0}, inputs[0].dtype(), inputs[0].device());
        Tensor bias_hh = inputs.size() > 10 ? inputs[10] : Tensor({0}, inputs[0].dtype(), inputs[0].device());
        return cpu::lstm_cell_backward_kernel(inputs[0], inputs[1],
                                               inputs[2], inputs[3], inputs[4],
                                               inputs[5], inputs[6],
                                               inputs[7], inputs[8],
                                               bias_ih, bias_hh);
    });

    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::gru_cell_forward_kernel(inputs[0], inputs[1],
                                                                  inputs[2], inputs[3],
                                                                  inputs[4], inputs[5])};
    });

    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // inputs: [grad_hy, input, hx, weight_ih, weight_hh, bias_ih, bias_hh]
        // bias_ih and bias_hh are optional (may not be present for backward compatibility)
        Tensor bias_ih = inputs.size() > 5 ? inputs[5] : Tensor({0}, inputs[0].dtype(), inputs[0].device());
        Tensor bias_hh = inputs.size() > 6 ? inputs[6] : Tensor({0}, inputs[0].dtype(), inputs[0].device());
        return cpu::gru_cell_backward_kernel(inputs[0], inputs[1], inputs[2],
                                              inputs[3], inputs[4],
                                              bias_ih, bias_hh);
    });

    // Full-sequence RNN operations (fused, SIMD-optimized)
    // inputs: [input, W_ih, W_hh, bias_ih, bias_hh, h0, c0] for LSTM (7 inputs)
    // inputs: [input, W_ih, W_hh, bias, h0] for GRU (5 inputs)
    table.register_kernel(OpId::LSTMForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // bias_ih and bias_hh may be empty tensors if not provided
        // Combined inside kernel during cache setup for oneDNN
        return cpu::lstm_forward_kernel(inputs[0], inputs[1], inputs[2],
                                         inputs[3], inputs[4], inputs[5], inputs[6]);
    });

    table.register_kernel(OpId::GRUForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // F.2: input convention is [input, W_ih, W_hh, bias_ih, h0, bias_hh].
        // The bias_hh slot was added in Phase 8.5; older callers omit it
        // (5 inputs), in which case bias_hh is treated as empty.
        const Tensor empty_t = empty({0}, inputs[0].dtype(), inputs[0].device());
        const Tensor& bias_ih = inputs[3];
        const Tensor& h0      = inputs[4];
        const Tensor& bias_hh = inputs.size() > 5 ? inputs[5] : empty_t;
        return cpu::gru_forward_kernel(inputs[0], inputs[1], inputs[2],
                                        bias_ih, bias_hh, h0);
    });

    // Multi-layer LSTM with fused oneDNN primitive
    // inputs: [input, h0, c0, W_ih_0, W_hh_0, bias_0, W_ih_1, W_hh_1, bias_1, ...]
    // Each layer has 3 tensors: W_ih, W_hh, bias (bias may be empty)
    table.register_kernel(OpId::LSTMMultiLayerForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);

        const Tensor& input = inputs[0];
        const Tensor& h0 = inputs[1];
        const Tensor& c0 = inputs[2];

        std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
        for (int64_t l = 0; l < num_layers; ++l) {
            size_t base_idx = 3 + l * 3;  // input, h0, c0, then 3 tensors per layer
            W_ih_list.push_back(inputs[base_idx]);
            W_hh_list.push_back(inputs[base_idx + 1]);
            bias_list.push_back(inputs[base_idx + 2]);
        }

        return cpu::lstm_multilayer_forward_kernel(input, W_ih_list, W_hh_list, bias_list, h0, c0);
    });

    // Multi-layer GRU with fused oneDNN primitive
    // inputs: [input, h0, W_ih_0, W_hh_0, bias_0, W_ih_1, W_hh_1, bias_1, ...]
    // Each layer has 3 tensors: W_ih, W_hh, bias (bias may be empty)
    table.register_kernel(OpId::GRUMultiLayerForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);

        const Tensor& input = inputs[0];
        const Tensor& h0 = inputs[1];

        std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
        for (int64_t l = 0; l < num_layers; ++l) {
            size_t base_idx = 2 + l * 3;  // input, h0, then 3 tensors per layer
            W_ih_list.push_back(inputs[base_idx]);
            W_hh_list.push_back(inputs[base_idx + 1]);
            bias_list.push_back(inputs[base_idx + 2]);
        }

        return cpu::gru_multilayer_forward_kernel(input, W_ih_list, W_hh_list, bias_list, h0);
    });

    // Bidirectional LSTM
    // inputs: [input, h0, c0, W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd,
    //          W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd]
    table.register_kernel(OpId::BiLSTMForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        const Tensor& input = inputs[0];
        const Tensor& h0 = inputs[1];
        const Tensor& c0 = inputs[2];
        const Tensor& W_ih_fwd = inputs[3];
        const Tensor& W_hh_fwd = inputs[4];
        const Tensor& bias_ih_fwd = inputs[5];
        const Tensor& bias_hh_fwd = inputs[6];
        const Tensor& W_ih_bwd = inputs[7];
        const Tensor& W_hh_bwd = inputs[8];
        const Tensor& bias_ih_bwd = inputs[9];
        const Tensor& bias_hh_bwd = inputs[10];

        return cpu::bilstm_forward_kernel(
            input,
            W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd,
            W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd,
            h0, c0
        );
    });

} // register_cpu_kernels_embedding_dropout

/**
 * @brief Register CPU kernels for the "vision_pool_misc" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_vision_pool_misc(BackendDispatchTable& table) {
    // =========================================================================
    // Vision Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cpu::interpolate_kernel(inputs[0], size, mode, align_corners);
    });
    // Audit D3: device-resident bilinear (and nearest) backward.
    // Inputs: [grad_output]. Attrs: InputShape (= [in_h, in_w]), Mode, AlignCorners.
    table.register_single_output_kernel(OpId::InterpolateBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_size = attrs.get_int_list(AttrKey::InputShape);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cpu::interpolate_backward_kernel(inputs[0], input_size, mode, align_corners);
    });

    // =========================================================================
    // ROI Align Operations
    // =========================================================================
    table.register_kernel(OpId::ROIAlignForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 7);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 7);
        float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
        int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
        bool aligned = attrs.get_bool(AttrKey::Aligned, true);
        return {cpu::roi_align_forward_kernel(inputs[0], inputs[1], output_h, output_w,
                                              spatial_scale, sampling_ratio, aligned)};
    });

    table.register_kernel(OpId::ROIAlignBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 1);
        int64_t feat_height = attrs.get_int(AttrKey::FeatHeight, 0);
        int64_t feat_width = attrs.get_int(AttrKey::FeatWidth, 0);
        float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
        int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
        bool aligned = attrs.get_bool(AttrKey::Aligned, true);
        return {cpu::roi_align_backward_kernel(inputs[0], inputs[1], batch_size,
                                                feat_height, feat_width, spatial_scale,
                                                sampling_ratio, aligned)};
    });

    table.register_single_output_kernel(OpId::BoxIoU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int iou_type = static_cast<int>(attrs.get_int(AttrKey::IouType, 0));
        return cpu::box_iou_kernel(inputs[0], inputs[1], iou_type);
    });

    table.register_kernel(OpId::NMS, [](std::span<const Tensor> inputs, const OpAttributes& attrs)
        -> std::vector<Tensor> {
        float iou_threshold = static_cast<float>(attrs.get_float(AttrKey::IouThreshold, 0.5));
        return {tenzor::ops::nms(inputs[0], inputs[1], iou_threshold)};
    });

    table.register_kernel(OpId::GatherRelativePositionBias,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
            int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
            return {cpu::gather_relative_position_bias_kernel(inputs[0], inputs[1], num_positions, num_heads)};
        });

    // =========================================================================
    // Grid Sample / Affine Grid Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::GridSample, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cpu::grid_sample_kernel(inputs[0], inputs[1], mode, padding_mode, align_corners);
    });

    table.register_single_output_kernel(OpId::AffineGrid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cpu::affine_grid_kernel(inputs[0], size, align_corners);
    });

    // audit Q.4: backward kernels for grid_sample / affine_grid. Replaces
    // the previous CPU shuttle in src/autograd/function_vision.cpp.
    // Inputs (GridSampleBackward): [input, grid, grad_output]. Outputs: [grad_input, grad_grid].
    table.register_kernel(OpId::GridSampleBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            auto [gi, gg] = cpu::grid_sample_backward_kernel(
                inputs[2], inputs[0], inputs[1], mode, padding_mode, align_corners);
            return {gi, gg};
        });

    // Inputs (AffineGridBackward): [grad_grid]. Attrs: OutputSize, AlignCorners.
    table.register_single_output_kernel(OpId::AffineGridBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto size = attrs.get_int_list(AttrKey::OutputSize);
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            return cpu::affine_grid_backward_kernel(inputs[0], size, align_corners);
        });

    // =========================================================================
    // Unfold / Fold Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // audit V.12: mirror CUDA contract — read per-axis attrs and reject
        // asymmetric kernel/stride/padding/dilation. The CPU unfold_kernel is
        // scalar-only; silently using a single axis would diverge from the
        // dispatcher's per-axis intent.
        const auto kernel_size = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 3);
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        if (kernel_size[0] != kernel_size[1] || stride[0] != stride[1] ||
            padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "Unfold (CPU): backend kernel only supports symmetric kernel/stride/padding/dilation; "
                "got kernel=" + std::to_string(kernel_size[0]) + "x" + std::to_string(kernel_size[1]) +
                ", stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]));
        }
        return cpu::unfold_kernel(inputs[0],
            kernel_size[0], stride[0], padding[0], dilation[0]);
    });

    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // audit V.12: same per-axis throw contract as Unfold above.
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        const auto kernel_size = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 3);
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        if (kernel_size[0] != kernel_size[1] || stride[0] != stride[1] ||
            padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "Fold (CPU): backend kernel only supports symmetric kernel/stride/padding/dilation; "
                "got kernel=" + std::to_string(kernel_size[0]) + "x" + std::to_string(kernel_size[1]) +
                ", stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]));
        }
        return cpu::fold_kernel(inputs[0], output_size,
            kernel_size[0], stride[0], padding[0], dilation[0]);
    });

    // =========================================================================
    // Additional Transform Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        return cpu::expand_kernel(inputs[0], shape);
    });

    table.register_single_output_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto repeats = attrs.get_int_list(AttrKey::Repeats);
        return cpu::repeat_kernel(inputs[0], repeats);
    });

    table.register_single_output_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return cpu::stack_kernel(tensors, dim);
    });

    table.register_kernel(OpId::Split, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t split_size = attrs.get_int(AttrKey::SplitSize, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::split_kernel(inputs[0], split_size, dim);
    });

    table.register_kernel(OpId::Chunk, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t chunks = attrs.get_int(AttrKey::Chunks, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::chunk_kernel(inputs[0], chunks, dim);
    });

    table.register_single_output_kernel(OpId::Tile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto reps = attrs.get_int_list(AttrKey::Reps);
        return cpu::tile_kernel(inputs[0], reps);
    });

    table.register_single_output_kernel(OpId::ToMemoryFormat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int format_int = static_cast<int>(attrs.get_int(AttrKey::MemoryFormat, 0));
        MemoryFormat format = static_cast<MemoryFormat>(format_int);
        return cpu::to_memory_format_kernel(inputs[0], format);
    });

    // =========================================================================
    // Additional Indexing Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Nonzero, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return cpu::nonzero_kernel(inputs[0]);
    });

    table.register_single_output_kernel(OpId::OneHot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t num_classes = attrs.get_int(AttrKey::NumClasses, 0);
        return cpu::one_hot_kernel(inputs[0], num_classes);
    });

    table.register_single_output_kernel(OpId::Take, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return cpu::take_kernel(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::Put, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool accumulate = attrs.get_bool(AttrKey::Accumulate, false);
        Tensor input = inputs[0];
        return cpu::put_kernel(input, inputs[1], inputs[2], accumulate);
    });

    // =========================================================================
    // Advanced (Fancy) Indexing
    // =========================================================================
    table.register_single_output_kernel(OpId::AdvancedIndex,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs[0] = source tensor
        // inputs[1..N] = index tensors (0-element sentinel means full-slice)
        int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
        const auto& src = inputs[0];
        auto src_shape = src.shape();
        int64_t src_ndim = static_cast<int64_t>(src_shape.size());

        // Identify which dims are indexed vs full-slice
        std::vector<bool> is_indexed(num_indices, false);
        std::vector<int64_t> broadcast_shape;
        for (int64_t i = 0; i < num_indices; ++i) {
            const auto& idx_t = inputs[1 + i];
            if (idx_t.numel() > 0) {
                is_indexed[i] = true;
                if (broadcast_shape.empty()) {
                    auto s = idx_t.shape();
                    broadcast_shape.assign(s.begin(), s.end());
                }
            }
        }

        if (broadcast_shape.empty()) {
            throw std::runtime_error("AdvancedIndex: no non-null index tensors");
        }

        // Compute output shape:
        //   [broadcast_shape...] + [passthrough dims...]
        // passthrough dims = dims not covered by indices + nullopt-indexed dims
        std::vector<int64_t> output_shape;
        output_shape.insert(output_shape.end(), broadcast_shape.begin(), broadcast_shape.end());

        // Nullopt dims within the indices range
        for (int64_t i = 0; i < num_indices; ++i) {
            if (!is_indexed[i]) {
                output_shape.push_back(src_shape[i]);
            }
        }
        // Dims beyond the indices range
        for (int64_t i = num_indices; i < src_ndim; ++i) {
            output_shape.push_back(src_shape[i]);
        }

        // Allocate output
        auto result = tenzor::empty(output_shape, src.dtype(), src.device());
        int64_t bc_numel = 1;
        for (auto d : broadcast_shape) bc_numel *= d;

        // Compute strides for source to do flat-index arithmetic
        std::vector<int64_t> src_strides(src_ndim);
        src_strides[src_ndim - 1] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * src_shape[d + 1];
        }

        // Compute passthrough dims info
        std::vector<int64_t> pass_dims;      // which source dims are passthrough
        for (int64_t i = 0; i < num_indices; ++i) {
            if (!is_indexed[i]) pass_dims.push_back(i);
        }
        for (int64_t i = num_indices; i < src_ndim; ++i) {
            pass_dims.push_back(i);
        }

        int64_t pass_numel = 1;
        for (auto d : pass_dims) pass_numel *= src_shape[d];

        // Collect contiguous index data pointers
        std::vector<const int64_t*> idx_ptrs(num_indices, nullptr);
        std::vector<Tensor> idx_contig(num_indices); // keep alive
        for (int64_t i = 0; i < num_indices; ++i) {
            if (is_indexed[i]) {
                idx_contig[i] = inputs[1 + i].contiguous();
                idx_ptrs[i] = idx_contig[i].data<int64_t>();
            }
        }

        auto src_contig = src.contiguous();
        size_t elem_size = tenzor::dtype_size(src.dtype());
        const auto* src_bytes = static_cast<const uint8_t*>(src_contig.data_ptr());
        auto* dst_bytes = static_cast<uint8_t*>(result.data_ptr());

        // For each broadcast position, compute the base offset into source
        // from the indexed dims, then copy the passthrough slice.
        for (int64_t bc = 0; bc < bc_numel; ++bc) {
            int64_t src_offset = 0;
            for (int64_t i = 0; i < num_indices; ++i) {
                if (is_indexed[i]) {
                    int64_t idx_val = idx_ptrs[i][bc];
                    if (idx_val < 0) idx_val += src_shape[i];
                    if (idx_val < 0 || idx_val >= src_shape[i]) {
                        throw std::out_of_range(
                            "AdvancedIndex: index " + std::to_string(idx_val) +
                            " out of range for dim " + std::to_string(i) +
                            " with size " + std::to_string(src_shape[i]));
                    }
                    src_offset += idx_val * src_strides[i];
                }
            }

            // Copy all passthrough elements
            if (pass_dims.empty()) {
                // Scalar case: just copy one element
                std::memcpy(dst_bytes + bc * elem_size,
                           src_bytes + src_offset * elem_size,
                           elem_size);
            } else {
                // Iterate over all combinations of passthrough dims
                for (int64_t p = 0; p < pass_numel; ++p) {
                    int64_t pass_offset = 0;
                    int64_t remaining = p;
                    for (int64_t k = static_cast<int64_t>(pass_dims.size()) - 1; k >= 0; --k) {
                        int64_t d = pass_dims[k];
                        int64_t coord = remaining % src_shape[d];
                        remaining /= src_shape[d];
                        pass_offset += coord * src_strides[d];
                    }

                    std::memcpy(dst_bytes + (bc * pass_numel + p) * elem_size,
                               src_bytes + (src_offset + pass_offset) * elem_size,
                               elem_size);
                }
            }
        }

        return result;
    });

    table.register_single_output_kernel(OpId::AdvancedIndexPut,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs[0] = destination tensor, inputs[1] = values, inputs[2..] = index tensors
        int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
        auto result = inputs[0].clone();
        const auto& values = inputs[1];
        auto src_shape = result.shape();
        int64_t src_ndim = static_cast<int64_t>(src_shape.size());

        std::vector<bool> is_indexed(num_indices, false);
        std::vector<int64_t> broadcast_shape;
        for (int64_t i = 0; i < num_indices; ++i) {
            const auto& idx_t = inputs[2 + i];
            if (idx_t.numel() > 0) {
                is_indexed[i] = true;
                if (broadcast_shape.empty()) {
                    auto s = idx_t.shape();
                    broadcast_shape.assign(s.begin(), s.end());
                }
            }
        }

        if (broadcast_shape.empty()) {
            throw std::runtime_error("AdvancedIndexPut: no non-null index tensors");
        }

        int64_t bc_numel = 1;
        for (auto d : broadcast_shape) bc_numel *= d;

        std::vector<int64_t> src_strides(src_ndim);
        src_strides[src_ndim - 1] = 1;
        for (int64_t d = src_ndim - 2; d >= 0; --d) {
            src_strides[d] = src_strides[d + 1] * src_shape[d + 1];
        }

        std::vector<int64_t> pass_dims;
        for (int64_t i = 0; i < num_indices; ++i) {
            if (!is_indexed[i]) pass_dims.push_back(i);
        }
        for (int64_t i = num_indices; i < src_ndim; ++i) {
            pass_dims.push_back(i);
        }

        int64_t pass_numel = 1;
        for (auto d : pass_dims) pass_numel *= src_shape[d];

        std::vector<const int64_t*> idx_ptrs(num_indices, nullptr);
        std::vector<Tensor> idx_contig(num_indices);
        for (int64_t i = 0; i < num_indices; ++i) {
            if (is_indexed[i]) {
                idx_contig[i] = inputs[2 + i].contiguous();
                idx_ptrs[i] = idx_contig[i].data<int64_t>();
            }
        }

        auto result_contig = result.contiguous();
        auto values_contig = values.contiguous();
        size_t elem_size = tenzor::dtype_size(result.dtype());
        auto* dst_bytes = static_cast<uint8_t*>(result_contig.data_ptr());
        const auto* val_bytes = static_cast<const uint8_t*>(values_contig.data_ptr());

        for (int64_t bc = 0; bc < bc_numel; ++bc) {
            int64_t dst_offset = 0;
            for (int64_t i = 0; i < num_indices; ++i) {
                if (is_indexed[i]) {
                    int64_t idx_val = idx_ptrs[i][bc];
                    if (idx_val < 0) idx_val += src_shape[i];
                    if (idx_val < 0 || idx_val >= src_shape[i]) {
                        throw std::out_of_range(
                            "AdvancedIndexPut: index " + std::to_string(idx_val) +
                            " out of range for dim " + std::to_string(i) +
                            " with size " + std::to_string(src_shape[i]));
                    }
                    dst_offset += idx_val * src_strides[i];
                }
            }

            if (pass_dims.empty()) {
                std::memcpy(dst_bytes + dst_offset * elem_size,
                           val_bytes + bc * elem_size,
                           elem_size);
            } else {
                for (int64_t p = 0; p < pass_numel; ++p) {
                    int64_t pass_offset = 0;
                    int64_t remaining = p;
                    for (int64_t k = static_cast<int64_t>(pass_dims.size()) - 1; k >= 0; --k) {
                        int64_t d = pass_dims[k];
                        int64_t coord = remaining % src_shape[d];
                        remaining /= src_shape[d];
                        pass_offset += coord * src_strides[d];
                    }

                    std::memcpy(dst_bytes + (dst_offset + pass_offset) * elem_size,
                               val_bytes + (bc * pass_numel + p) * elem_size,
                               elem_size);
                }
            }
        }

        return result_contig;
    });

} // register_cpu_kernels_vision_pool_misc

/**
 * @brief Register CPU kernels for the "advanced" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_advanced(BackendDispatchTable& table) {
    // =========================================================================
    // Advanced Operations (TopK, Sort, CumSum, CumProd, Unique)
    // =========================================================================
    table.register_kernel(OpId::TopK, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool largest = attrs.get_bool(AttrKey::Largest, true);
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        auto [values, indices] = cpu::topk_kernel(inputs[0], k, dim, largest, sorted);
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::Sort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        auto [values, indices] = cpu::sort_kernel(inputs[0], dim, descending);
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::CumSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::cumsum_kernel(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::CumProd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::cumprod_kernel(inputs[0], dim);
    });

    table.register_kernel(OpId::Unique, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        bool return_inverse = attrs.get_bool(AttrKey::ReturnInverse, false);
        bool return_counts = attrs.get_bool(AttrKey::ReturnCounts, false);
        auto [unique_vals, inverse, counts] = cpu::unique_kernel(inputs[0], sorted, return_inverse, return_counts);
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    // =========================================================================
    // New Reduction Operations (CumMax, CumMin, Isin, Kthvalue, Fmax, Fmin,
    //                           Quantile, Nanquantile, Nanmedian, Histc,
    //                           UniqueConsecutive)
    // =========================================================================
    table.register_kernel(OpId::CumMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = cpu::cummax_kernel(inputs[0], dim);
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::CumMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = cpu::cummin_kernel(inputs[0], dim);
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Isin, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return cpu::isin_kernel(inputs[0], inputs[1]);
    });

    table.register_kernel(OpId::Kthvalue, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        auto [values, indices] = cpu::kthvalue_kernel(inputs[0], k, dim, keepdim);
        return std::vector<Tensor>{values, indices};
    });

    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Fmax, cpu::fmax_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Fmin, cpu::fmin_kernel);

    table.register_single_output_kernel(OpId::Quantile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double q = attrs.get_float(AttrKey::Alpha, 0.5);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cpu::quantile_kernel(inputs[0], q, dim, keepdim);
    });

    table.register_single_output_kernel(OpId::Nanquantile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double q = attrs.get_float(AttrKey::Alpha, 0.5);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cpu::nanquantile_kernel(inputs[0], q, dim, keepdim);
    });

    table.register_single_output_kernel(OpId::Nanmedian, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::nanmedian_kernel(inputs[0], dim);
    });

    table.register_single_output_kernel(OpId::Histc, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t bins = attrs.get_int(AttrKey::N, 100);
        double min_val = attrs.get_float(AttrKey::Alpha, 0.0);
        double max_val = attrs.get_float(AttrKey::Beta, 0.0);
        return cpu::histc_kernel(inputs[0], bins, min_val, max_val);
    });

    table.register_kernel(OpId::UniqueConsecutive, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool return_inverse = attrs.get_bool(AttrKey::Keepdim, false);  // reused for return_inverse
        bool return_counts = true;  // always compute; caller decides whether to use
        auto [unique_vals, inverse, counts] = cpu::unique_consecutive_kernel(inputs[0], return_inverse, return_counts);
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    table.register_single_output_kernel(OpId::SegmentReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t axis = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        return cpu::segment_reduce_kernel(inputs[0], inputs[1], reduce, axis);
    });

} // register_cpu_kernels_advanced

/**
 * @brief Register CPU kernels for the "rmsnorm_etc" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_rmsnorm_etc(BackendDispatchTable& table) {
    // =========================================================================
    // RMSNorm Operations
    // =========================================================================
    table.register_kernel(OpId::RMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, rrms] = cpu::fused_rms_norm_kernel(inputs[0], inputs[1], eps);
        return std::vector<Tensor>{output, rrms};
    });

    table.register_kernel(OpId::FusedRMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, rrms] = cpu::fused_rms_norm_kernel(inputs[0], inputs[1], eps);
        return std::vector<Tensor>{output, rrms};
    });

    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // inputs: [grad_output, input, weight, rrms]
        auto [grad_input, grad_weight] = cpu::rms_norm_backward_kernel(inputs[0], inputs[1], inputs[2], inputs[3]);
        return std::vector<Tensor>{grad_input, grad_weight};
    });

    // =========================================================================
    // Fused Attention
    // =========================================================================
    table.register_kernel(OpId::FusedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        return std::vector<Tensor>{cpu::fused_attention_kernel(inputs[0], inputs[1], inputs[2], scale, causal)};
    });

    // =========================================================================
    // Flash Attention (O(N) memory, tiled online softmax)
    // =========================================================================
    table.register_kernel(OpId::FlashAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Per docs/internals/attention-contract.md, returns 4 tensors:
        // [output, lse_f32, philox_seed_int64, philox_offset_int64].
        // seed/offset are empty Tensors when dropout_p == 0.
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        float dropout_p = static_cast<float>(attrs.get_float(AttrKey::DropoutP, 0.0));
        bool is_training = attrs.get_bool(AttrKey::IsTraining, attrs.get_bool(AttrKey::Training, false));
        uint64_t seed_in = static_cast<uint64_t>(attrs.get_int(AttrKey::Seed, 0));
        return cpu::flash_attention_forward(inputs[0], inputs[1], inputs[2],
                                            scale, causal, dropout_p, is_training, seed_in);
    });

    // =========================================================================
    // Flash Attention Backward
    // =========================================================================
    table.register_kernel(OpId::FlashAttentionBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Per contract, inputs are [dO, Q, K, V, O, L?, philox_seed?, philox_offset?].
        // L is currently unused by this CPU backward (it recomputes softmax
        // from scratch — correctness-equivalent, just more memory). seed and
        // offset, when present, replay the forward's dropout mask exactly.
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        float dropout_p = static_cast<float>(attrs.get_float(AttrKey::DropoutP, 0.0));
        uint64_t philox_seed = 0, philox_offset = 0;
        if (inputs.size() > 6 && inputs[6].is_valid() && inputs[6].numel() > 0) {
            philox_seed = static_cast<uint64_t>(inputs[6].data<int64_t>()[0]);
        }
        if (inputs.size() > 7 && inputs[7].is_valid() && inputs[7].numel() > 0) {
            philox_offset = static_cast<uint64_t>(inputs[7].data<int64_t>()[0]);
        }
        return cpu::flash_attention_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                             scale, causal, dropout_p, philox_seed, philox_offset);
    });

    // =========================================================================
    // FlexAttention (built-in score_mod registry, per
    // docs/internals/attention-contract.md + include/tenzor/ops/attention_contract.hpp)
    // =========================================================================
    // ScoreModId selects a built-in score modification:
    //   0 = identity                   (no modification — equivalent to FusedAttention)
    //   1 = causal                     (upper-triangular -inf mask before softmax)
    //   2 = sliding_window             (|i-j| > WindowSize/2 → -inf)
    //   3 = relpos_bias                (additive bias tensor passed as inputs[N+1])
    //   4 = alibi                      (-slope_h * (q_idx - k_idx); per-head canonical slopes)
    //   5 = prefix_lm                  (first PrefixLength positions bidi; remainder causal)
    //   6 = sliding_window_sym         (same as 2; spelled out symmetrically per task 2.21)
    //   7 = user_lambda                (lookup via find_registered_score_mod)
    // All mask sentinels MUST be -INFINITY (attention contract: never -1e9/-1e30).
    auto flex_attention_dispatch = [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

        // ScoreModId 0 (identity) and 1 (causal) reduce to FusedAttention —
        // route to the optimised flash kernel directly.
        if (score_mod_id == 0 || score_mod_id == 1) {
            bool causal = (score_mod_id == 1);
            return cpu::flash_attention_forward(inputs[0], inputs[1], inputs[2],
                                                scale, causal, /*dropout_p=*/0.0f,
                                                /*is_training=*/false, /*seed_in=*/0);
        }

        // -----------------------------------------------------------------
        // Composed-ops path for IDs 2-7. Same scaffolding for every mode:
        //   scores = Q @ K^T * scale
        //   scores += mask_or_bias
        //   probs  = softmax(scores, dim=-1)
        //   output = probs @ V
        // Uses tenzor::matmul (handles 3D/4D batched) instead of bmm.
        // -----------------------------------------------------------------
        const Tensor& Q = inputs[0];
        const Tensor& K = inputs[1];
        const Tensor& V = inputs[2];
        const int64_t S_q = Q.shape()[Q.shape().size() - 2];
        const int64_t S_k = K.shape()[K.shape().size() - 2];

        Tensor Kt = tenzor::transpose(K, -1, -2);
        Tensor scores = tenzor::matmul(Q, Kt);
        const auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
        Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                       scores.dtype(), scores.device());
        scores = scores * scale_t;

        // Build (S_q, S_k) row/col index grids for any positional mask.
        auto build_index_grids = [&]() -> std::pair<Tensor, Tensor> {
            Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Q.device()).to(DType::Float32);
            Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Q.device()).to(DType::Float32);
            Tensor rows_2d = tenzor::reshape(rows, std::vector<int64_t>{S_q, 1});
            Tensor cols_2d = tenzor::reshape(cols, std::vector<int64_t>{1, S_k});
            return {rows_2d, cols_2d};
        };

        // Apply a positional Bool mask (true → -inf), broadcasted over batch/head.
        auto apply_neg_inf_mask = [&](const Tensor& mask_2d) {
            Tensor neg_inf = tenzor::full(scores_shape,
                -std::numeric_limits<float>::infinity(),
                scores.dtype(), scores.device());
            scores = scores + (mask_2d.to(scores.dtype()) * neg_inf);
        };

        if (score_mod_id == 2 || score_mod_id == 6) {
            // Sliding window: mask positions where |i - j| > window/2.
            int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
            if (window_size <= 0) {
                throw std::invalid_argument(
                    "FlexAttention CPU: ScoreModId=" + std::to_string(score_mod_id) +
                    " (sliding_window) requires AttrKey::WindowSize > 0.");
            }
            auto [rows_2d, cols_2d] = build_index_grids();
            Tensor abs_diff = tenzor::abs(tenzor::sub(rows_2d, cols_2d));
            Tensor half_t = tenzor::full({1}, static_cast<double>(window_size / 2),
                                          abs_diff.dtype(), abs_diff.device());
            Tensor outside = tenzor::gt(abs_diff, half_t);
            apply_neg_inf_mask(outside);
        } else if (score_mod_id == 3) {
            // RelPosBias: additive bias tensor as the last input (after Q,K,V
            // and an optional block_mask). When only inputs[3] is present we
            // treat it as the bias; otherwise the bias is the trailing input.
            Tensor bias;
            if (inputs.size() == 4) {
                bias = inputs[3];
            } else if (inputs.size() >= 5) {
                bias = inputs.back();
            } else {
                throw std::invalid_argument(
                    "FlexAttention CPU: ScoreModId=3 (relpos_bias) requires a "
                    "bias tensor as the last input (shape broadcastable to "
                    "scores [..., S_q, S_k]).");
            }
            scores = scores + bias.to(scores.dtype());
        } else if (score_mod_id == 4) {
            // ALiBi: -slope_h * (q_idx - k_idx). Canonical slopes from
            // Press et al. 2022: slope_h = 2^(-8*(h+1)/H). Requires Q to be
            // [..., H, S_q, D] so we can infer H.
            const auto& q_shape = Q.shape();
            if (q_shape.size() < 3) {
                throw std::invalid_argument(
                    "FlexAttention CPU: ScoreModId=4 (alibi) requires Q rank >= 3 "
                    "([..., H, S_q, D]) to infer head count.");
            }
            const int64_t H = q_shape[q_shape.size() - 3];
            std::vector<float> slope_vals(static_cast<std::size_t>(H));
            for (int64_t h = 0; h < H; ++h) {
                slope_vals[static_cast<std::size_t>(h)] =
                    std::pow(2.0f, -8.0f * static_cast<float>(h + 1) / static_cast<float>(H));
            }
            Tensor slopes = Tensor::from_blob(slope_vals.data(),
                std::vector<int64_t>{H}, DType::Float32, Device::cpu()).clone().to(Q.device()).to(scores.dtype());
            // scores shape is (..., H, S_q, S_k). H sits at axis = -3.
            std::vector<int64_t> slope_shape(scores_shape.size(), 1);
            slope_shape[slope_shape.size() - 3] = H;
            slopes = tenzor::reshape(slopes, slope_shape);

            // Build positional bias (S_q, S_k) = -(q_idx - k_idx) = (k_idx - q_idx).
            auto [rows_2d, cols_2d] = build_index_grids();
            Tensor pos_bias = tenzor::sub(cols_2d, rows_2d).to(scores.dtype());
            std::vector<int64_t> pb_shape(scores_shape.size(), 1);
            pb_shape[pb_shape.size() - 2] = S_q;
            pb_shape[pb_shape.size() - 1] = S_k;
            pos_bias = tenzor::reshape(pos_bias, pb_shape);
            // alibi_bias = slope * (k - q). Per-head slope broadcasts.
            scores = scores + (slopes * pos_bias);
        } else if (score_mod_id == 5) {
            // PrefixLM: first PrefixLength positions in each row see all PrefixLength
            // columns bidirectionally; for q >= PrefixLength, mask k > q (causal).
            // mask[i, j] = (i >= PrefixLength) AND (j > i).
            int64_t prefix_len = attrs.get_int(AttrKey::PrefixLength, 0);
            if (prefix_len < 0) {
                throw std::invalid_argument(
                    "FlexAttention CPU: ScoreModId=5 (prefix_lm) requires "
                    "AttrKey::PrefixLength >= 0.");
            }
            auto [rows_2d, cols_2d] = build_index_grids();
            Tensor prefix_t = tenzor::full({1}, static_cast<double>(prefix_len),
                                            rows_2d.dtype(), rows_2d.device());
            Tensor q_after_prefix = tenzor::ge(rows_2d, prefix_t).to(DType::Float32);
            Tensor causal = tenzor::gt(cols_2d, rows_2d).to(DType::Float32);
            // Logical AND via product on 0/1 floats. Both broadcast to (S_q, S_k).
            Tensor mask_f = q_after_prefix * causal;
            apply_neg_inf_mask(mask_f);
        } else if (score_mod_id == 7) {
            auto user_fn = tenzor::nn::find_registered_score_mod(score_mod_id);
            if (!user_fn) {
                throw std::invalid_argument(
                    "FlexAttention CPU: ScoreModId=7 (user_lambda) but no "
                    "functor is registered. Call "
                    "tenzor::nn::register_score_mod(7, fn) before dispatch.");
            }
            scores = user_fn(scores, /*b=*/0, /*h=*/0, /*q_start=*/0, /*kv_start=*/0);
        } else if (score_mod_id >= 8) {
            // IDs >= 8 are reserved for user functors registered by integer ID.
            auto user_fn = tenzor::nn::find_registered_score_mod(score_mod_id);
            if (!user_fn) {
                throw std::runtime_error(
                    "FlexAttention CPU: ScoreModId=" + std::to_string(score_mod_id) +
                    " is not a built-in and no user functor is registered. "
                    "Built-ins: 0=identity, 1=causal, 2=sliding_window, "
                    "3=relpos_bias, 4=alibi, 5=prefix_lm, 6=sliding_window_sym, "
                    "7=user_lambda. Register custom IDs via "
                    "tenzor::nn::register_score_mod(id, fn).");
            }
            scores = user_fn(scores, 0, 0, 0, 0);
        } else {
            throw std::runtime_error(
                "FlexAttention CPU: unrecognised ScoreModId=" + std::to_string(score_mod_id));
        }

        NewOpAttributes sm_attrs;
        sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
        std::vector<Tensor> sm_in = {scores};
        Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
        Tensor output = tenzor::matmul(probs, V);
        return {output, Tensor{}};  // LSE not computed in composed path
    };
    table.register_kernel(OpId::FlexAttention, flex_attention_dispatch);

    // FlexAttentionBackward dispatch split:
    //   - ScoreModId 0/1 (identity/causal): fused FlashAttention backward
    //     (algebraically identical — same kernel).
    //   - ScoreModId 2..N (sliding_window, relpos_bias, ALiBi, prefix_lm,
    //     user-registered): composed backward — forward is replayed with the
    //     per-ID score-mod mask reapplied, then the softmax chain rule
    //     produces {dQ, dK, dV}.
    table.register_kernel(OpId::FlexAttentionBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

        // ScoreModId 0/1: route to fused FlashAttention backward (mathematically identical).
        if (score_mod_id == 0 || score_mod_id == 1) {
            bool causal = (score_mod_id == 1);
            return cpu::flash_attention_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                                                 scale, causal, /*dropout_p=*/0.0f,
                                                 /*philox_seed=*/0, /*philox_offset=*/0);
        }

        // Composed backward for IDs 2-7+ (replays forward to recover masked
        // scores, applies softmax-attention chain rule). Uses matmul for
        // 3D/4D batched safety. inputs: [dO, Q, K, V, O, (LSE optional), (extra...)].
        if (score_mod_id >= 2) {
            const Tensor& dO = inputs[0];
            const Tensor& Q  = inputs[1];
            const Tensor& K  = inputs[2];
            const Tensor& V  = inputs[3];
            const int64_t S_q = Q.shape()[Q.shape().size() - 2];
            const int64_t S_k = K.shape()[K.shape().size() - 2];

            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::matmul(Q, Kt);
            const auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = scores * scale_t;

            auto build_index_grids = [&]() -> std::pair<Tensor, Tensor> {
                Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Q.device()).to(DType::Float32);
                Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Q.device()).to(DType::Float32);
                Tensor rows_2d = tenzor::reshape(rows, std::vector<int64_t>{S_q, 1});
                Tensor cols_2d = tenzor::reshape(cols, std::vector<int64_t>{1, S_k});
                return {rows_2d, cols_2d};
            };
            auto apply_neg_inf_mask = [&](const Tensor& mask_2d) {
                // Use -INFINITY per attention-contract; softmax-grad
                // (attn * (dAttn - sum)) drives `attn` to 0 at masked positions,
                // so 0 * (anything) = 0 without NaN risk.
                Tensor neg_inf = tenzor::full(scores_shape,
                    -std::numeric_limits<float>::infinity(),
                    scores.dtype(), scores.device());
                scores = scores + (mask_2d.to(scores.dtype()) * neg_inf);
            };

            if (score_mod_id == 2 || score_mod_id == 6) {
                int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
                if (window_size <= 0) {
                    throw std::invalid_argument(
                        "FlexAttentionBackward CPU: ScoreModId=" +
                        std::to_string(score_mod_id) +
                        " requires AttrKey::WindowSize > 0.");
                }
                auto [rows_2d, cols_2d] = build_index_grids();
                Tensor abs_diff = tenzor::abs(tenzor::sub(rows_2d, cols_2d));
                Tensor half_t = tenzor::full({1}, static_cast<double>(window_size / 2),
                                              abs_diff.dtype(), abs_diff.device());
                Tensor outside = tenzor::gt(abs_diff, half_t);
                apply_neg_inf_mask(outside);
            } else if (score_mod_id == 3) {
                // Backward replays bias. Inputs layout:
                // [dO, Q, K, V, O, (LSE)?, (bias)]. Bias is the last input.
                if (inputs.size() < 6) {
                    throw std::invalid_argument(
                        "FlexAttentionBackward CPU: ScoreModId=3 (relpos_bias) "
                        "requires the bias tensor saved as the last input.");
                }
                const Tensor& bias = inputs.back();
                scores = scores + bias.to(scores.dtype());
            } else if (score_mod_id == 4) {
                const auto& q_shape = Q.shape();
                if (q_shape.size() < 3) {
                    throw std::invalid_argument(
                        "FlexAttentionBackward CPU: ScoreModId=4 (alibi) requires "
                        "Q rank >= 3 to infer head count.");
                }
                const int64_t H = q_shape[q_shape.size() - 3];
                std::vector<float> slope_vals(static_cast<std::size_t>(H));
                for (int64_t h = 0; h < H; ++h) {
                    slope_vals[static_cast<std::size_t>(h)] =
                        std::pow(2.0f, -8.0f * static_cast<float>(h + 1) / static_cast<float>(H));
                }
                Tensor slopes = Tensor::from_blob(slope_vals.data(),
                    std::vector<int64_t>{H}, DType::Float32, Device::cpu()).clone().to(Q.device()).to(scores.dtype());
                std::vector<int64_t> slope_shape(scores_shape.size(), 1);
                slope_shape[slope_shape.size() - 3] = H;
                slopes = tenzor::reshape(slopes, slope_shape);
                auto [rows_2d, cols_2d] = build_index_grids();
                Tensor pos_bias = tenzor::sub(cols_2d, rows_2d).to(scores.dtype());
                std::vector<int64_t> pb_shape(scores_shape.size(), 1);
                pb_shape[pb_shape.size() - 2] = S_q;
                pb_shape[pb_shape.size() - 1] = S_k;
                pos_bias = tenzor::reshape(pos_bias, pb_shape);
                scores = scores + (slopes * pos_bias);
            } else if (score_mod_id == 5) {
                int64_t prefix_len = attrs.get_int(AttrKey::PrefixLength, 0);
                if (prefix_len < 0) {
                    throw std::invalid_argument(
                        "FlexAttentionBackward CPU: ScoreModId=5 (prefix_lm) "
                        "requires AttrKey::PrefixLength >= 0.");
                }
                auto [rows_2d, cols_2d] = build_index_grids();
                Tensor prefix_t = tenzor::full({1}, static_cast<double>(prefix_len),
                                                rows_2d.dtype(), rows_2d.device());
                Tensor q_after_prefix = tenzor::ge(rows_2d, prefix_t).to(DType::Float32);
                Tensor causal = tenzor::gt(cols_2d, rows_2d).to(DType::Float32);
                Tensor mask_f = q_after_prefix * causal;
                apply_neg_inf_mask(mask_f);
            } else if (score_mod_id == 7 || score_mod_id >= 8) {
                auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
                if (!fn) {
                    throw std::runtime_error(
                        "FlexAttentionBackward CPU: no user score_mod registered for ScoreModId=" +
                        std::to_string(score_mod_id));
                }
                scores = fn(scores, 0, 0, 0, 0);
            } else {
                throw std::runtime_error(
                    "FlexAttentionBackward CPU: unrecognised ScoreModId=" +
                    std::to_string(score_mod_id));
            }

            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_in = {scores};
            Tensor attn = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];

            // dV = attn^T @ dO
            Tensor attn_t = tenzor::transpose(attn, -1, -2);
            Tensor dV = tenzor::matmul(attn_t, dO);

            // dAttn = dO @ V^T
            Tensor Vt = tenzor::transpose(V, -1, -2);
            Tensor dAttn = tenzor::matmul(dO, Vt);

            // dScores = attn * (dAttn - sum(attn * dAttn, dim=-1, keepdim=true))
            Tensor ad = tenzor::mul(attn, dAttn);
            NewOpAttributes sum_attrs;
            sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            sum_attrs.set(AttrKey::Keepdim, true);
            std::vector<Tensor> sum_inputs = {ad};
            Tensor sum_ad = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
            Tensor dScores = tenzor::mul(attn, tenzor::sub(dAttn, sum_ad));

            // Apply scale to grad.
            Tensor scale_t2 = tenzor::full(
                std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
                static_cast<double>(scale), dScores.dtype(), dScores.device());
            dScores = tenzor::mul(dScores, scale_t2);

            Tensor dQ = tenzor::matmul(dScores, K);
            Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
            Tensor dK = tenzor::matmul(dScores_t, Q);

            return {dQ, dK, dV};
        }

        throw std::runtime_error(
            "FlexAttentionBackward CPU: ScoreModId=" + std::to_string(score_mod_id) +
            " not recognised.");
    });

    // =========================================================================
    // Einsum (composed — delegates to einsum_composed to avoid dispatch loop)
    // =========================================================================
    table.register_kernel(OpId::Einsum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto equation = std::string(attrs.get_string(AttrKey::EinsumEquation, ""));
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return {einsum_composed(equation, tensors)};
    });

    // =========================================================================
    // Fused Conv2d + Activation Variants
    // =========================================================================
    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: per-axis stride/padding/dilation.
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_sigmoid_kernel(inputs[0], inputs[1], bias,
                                                stride[0], stride[1],
                                                padding[0], padding[1],
                                                dilation[0], dilation[1], groups);
    });

    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: per-axis stride/padding/dilation.
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_tanh_kernel(inputs[0], inputs[1], bias,
                                             stride[0], stride[1],
                                             padding[0], padding[1],
                                             dilation[0], dilation[1], groups);
    });

    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: per-axis stride/padding/dilation.
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_swish_kernel(inputs[0], inputs[1], bias,
                                              stride[0], stride[1],
                                              padding[0], padding[1],
                                              dilation[0], dilation[1], groups);
    });

    // =========================================================================
    // BatchNorm2d Fused Training
    // =========================================================================
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        // Shallow-copy: these must share storage with the caller's tensors so
        // in-place running-stat updates propagate back.  If Tensor copy ever
        // becomes deep, the mutation silently disappears — assert here to catch it.
        Tensor running_mean = inputs[1];
        Tensor running_var  = inputs[2];
        if (running_mean.data_ptr() != inputs[1].data_ptr())
            throw std::runtime_error(
                "BatchNorm2dFusedTraining: running_mean copy does not share storage "
                "with caller tensor — Tensor copy semantics changed to deep-copy?");
        if (running_var.data_ptr() != inputs[2].data_ptr())
            throw std::runtime_error(
                "BatchNorm2dFusedTraining: running_var copy does not share storage "
                "with caller tensor — Tensor copy semantics changed to deep-copy?");
        return cpu::batchnorm2d_fused_training_kernel(inputs[0], running_mean, running_var,
                                                       inputs[3], inputs[4], momentum, epsilon);
    });

    // =========================================================================
    // Fused Optimizer Steps
    // =========================================================================
    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        float dampening = static_cast<float>(attrs.get_float(AttrKey::Dampening, 0.0));
        bool nesterov = attrs.get_bool(AttrKey::Nesterov, false);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor* momentum_buffer = (inputs.size() > 2 && momentum > 0.0f)
            ? &const_cast<Tensor&>(inputs[2]) : nullptr;

        cpu::fused_sgd_step_kernel(param, inputs[1], momentum_buffer,
            lr, momentum, weight_decay, dampening, nesterov);
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdamStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double lr = attrs.get_float(AttrKey::Lr, 0.001);
        double beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
        double beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
        double eps = attrs.get_float(AttrKey::Eps, 1e-8);
        double weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
        int64_t step = attrs.get_int(AttrKey::Step, 1);
        bool decoupled = attrs.get_bool(AttrKey::Decoupled, false);
        bool amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& exp_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& exp_avg_sq = const_cast<Tensor&>(inputs[3]);
        Tensor* max_exp_avg_sq = (amsgrad && inputs.size() > 4)
            ? &const_cast<Tensor&>(inputs[4]) : nullptr;

        cpu::fused_adam_step_kernel(param, inputs[1], exp_avg, exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, decoupled,
            max_exp_avg_sq, amsgrad);
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdamAtan2Step, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.001));
        float beta1 = static_cast<float>(attrs.get_float(AttrKey::Beta1, 0.9));
        float beta2 = static_cast<float>(attrs.get_float(AttrKey::Beta2, 0.999));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        int64_t step = attrs.get_int(AttrKey::Step, 1);
        bool amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& exp_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& exp_avg_sq = const_cast<Tensor&>(inputs[3]);
        Tensor* max_exp_avg_sq = (amsgrad && inputs.size() > 4)
            ? &const_cast<Tensor&>(inputs[4]) : nullptr;

        cpu::fused_adam_atan2_step_kernel(param, inputs[1], exp_avg, exp_avg_sq, max_exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, amsgrad);
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedRMSPropStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.99));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-8));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        bool centered = attrs.get_bool(AttrKey::Centered, false);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& square_avg = const_cast<Tensor&>(inputs[2]);
        Tensor* grad_avg = (centered && inputs.size() > 3) ? &const_cast<Tensor&>(inputs[3]) : nullptr;
        Tensor* momentum_buffer = (momentum > 0.0f && inputs.size() > 4) ? &const_cast<Tensor&>(inputs[4]) : nullptr;

        cpu::fused_rmsprop_step_kernel(param, inputs[1], square_avg, grad_avg, momentum_buffer,
            lr, alpha, eps, weight_decay, momentum, centered);
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdadeltaStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& square_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& acc_delta = const_cast<Tensor&>(inputs[3]);

        cpu::fused_adadelta_step_kernel(param, inputs[1], square_avg, acc_delta,
            rho, eps, lr, weight_decay);
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdagradStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float lr_decay = static_cast<float>(attrs.get_float(AttrKey::LrDecay, 0.0));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        int64_t step = attrs.get_int(AttrKey::Step, 1);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& sum_sq = const_cast<Tensor&>(inputs[2]);

        cpu::fused_adagrad_step_kernel(param, inputs[1], sum_sq,
            lr, lr_decay, eps, weight_decay, step);
        return std::vector<Tensor>{param};
    });

} // register_cpu_kernels_rmsnorm_etc

/**
 * @brief Register CPU kernels for the "creation" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_creation(BackendDispatchTable& table) {
    // =========================================================================
    // Creation Operations
    // =========================================================================
    table.register_kernel(OpId::Zeros, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::zeros_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Ones, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::ones_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Full, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        double value = attrs.get_float(AttrKey::Value, 0.0);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::full_kernel(shape, value, dtype, device)};
    });

    table.register_kernel(OpId::Rand, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::rand_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Randn, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::randn_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Randint, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t low = attrs.get_int(AttrKey::Start, 0);
        int64_t high = attrs.get_int(AttrKey::End, 0);
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Int32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::randint_kernel(low, high, shape, dtype, device)};
    });

    table.register_kernel(OpId::Arange, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = attrs.get_float(AttrKey::Start, 0.0);
        double end = attrs.get_float(AttrKey::End, 0.0);
        double step = attrs.get_float(AttrKey::Step, 1.0);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::arange_kernel(start, end, step, dtype, device)};
    });

    table.register_kernel(OpId::Linspace, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = attrs.get_float(AttrKey::Start, 0.0);
        double end = attrs.get_float(AttrKey::End, 1.0);
        int64_t steps = attrs.get_int(AttrKey::Steps, 100);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::linspace_kernel(start, end, steps, dtype, device)};
    });

    table.register_kernel(OpId::Eye, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = attrs.get_int(AttrKey::N, 0);
        int64_t m = attrs.get_int(AttrKey::M, -1);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::eye_kernel(n, m, dtype, device)};
    });

    // =========================================================================
    // Tensor Manipulation Operations (triu, tril, diag, trace, flip)
    // These call the public API which already has CPU implementations inline
    // =========================================================================
    table.register_single_output_kernel(OpId::Triu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return triu(inputs[0], diagonal);
    });

    table.register_single_output_kernel(OpId::Tril, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return tril(inputs[0], diagonal);
    });

    table.register_single_output_kernel(OpId::Diag, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return diag(inputs[0], diagonal);
    });

    table.register_single_output_kernel(OpId::Trace, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // Forward to the inline tenzor::trace(Tensor) implementation.
        return tenzor::trace(inputs[0]);
    });

    table.register_single_output_kernel(OpId::Flip, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) dims = {0};
        return flip(inputs[0], dims);
    });

    // =========================================================================
    // StridedFill — fill non-contiguous tensor in-place on CPU
    // =========================================================================
    table.register_inplace_kernel(OpId::StridedFill, [](Tensor& self, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        double value = attrs.get_float(AttrKey::Value, 0.0);
        auto ndims = self.ndim();
        auto shp = self.shape();
        auto str = self.strides();
        auto elem_size = static_cast<int64_t>(dtype_size(self.dtype()));
        auto* base = static_cast<uint8_t*>(self.data_ptr());
        int64_t n = self.numel();

        // Precompute cumulative products for flat-index-to-coordinate conversion
        std::vector<int64_t> cum_sizes(ndims);
        if (ndims > 0) {
            cum_sizes[ndims - 1] = 1;
            for (int64_t d = ndims - 2; d >= 0; --d)
                cum_sizes[d] = cum_sizes[d + 1] * shp[d + 1];
        }

        auto dtype = self.dtype();
        #pragma omp parallel for if(n > 65536)
        for (int64_t i = 0; i < n; ++i) {
            int64_t byte_offset = 0;
            int64_t idx = i;
            for (int64_t d = 0; d < ndims; ++d) {
                int64_t coord = idx / cum_sizes[d];
                idx %= cum_sizes[d];
                byte_offset += coord * str[d] * elem_size;
            }
            auto* ptr = base + byte_offset;
            switch (dtype) {
                case DType::Float32:   *reinterpret_cast<float*>(ptr)    = static_cast<float>(value); break;
                case DType::Float64:   *reinterpret_cast<double*>(ptr)   = value; break;
                case DType::Int32:     *reinterpret_cast<int32_t*>(ptr)  = static_cast<int32_t>(value); break;
                case DType::Int64:     *reinterpret_cast<int64_t*>(ptr)  = static_cast<int64_t>(value); break;
                case DType::Int16:     *reinterpret_cast<int16_t*>(ptr)  = static_cast<int16_t>(value); break;
                case DType::Int8:      *reinterpret_cast<int8_t*>(ptr)   = static_cast<int8_t>(value); break;
                case DType::UInt8:     *reinterpret_cast<uint8_t*>(ptr)  = static_cast<uint8_t>(value); break;
                case DType::UInt16:    *reinterpret_cast<uint16_t*>(ptr) = static_cast<uint16_t>(value); break;
                case DType::UInt32:    *reinterpret_cast<uint32_t*>(ptr) = static_cast<uint32_t>(value); break;
                case DType::UInt64:    *reinterpret_cast<uint64_t*>(ptr) = static_cast<uint64_t>(value); break;
                case DType::Float16:   *reinterpret_cast<Float16*>(ptr)  = Float16(static_cast<float>(value)); break;
                case DType::BFloat16:  *reinterpret_cast<BFloat16*>(ptr) = BFloat16(static_cast<float>(value)); break;
                case DType::Bool:      *reinterpret_cast<bool*>(ptr)     = (value != 0.0); break;
                case DType::Complex64:
                    *reinterpret_cast<std::complex<float>*>(ptr) =
                        std::complex<float>(static_cast<float>(value), 0.0f);
                    break;
                case DType::Complex128:
                    *reinterpret_cast<std::complex<double>*>(ptr) =
                        std::complex<double>(value, 0.0);
                    break;
                case DType::FP8_E4M3:
                    *reinterpret_cast<FP8_E4M3*>(ptr) = FP8_E4M3(static_cast<float>(value));
                    break;
                case DType::FP8_E5M2:
                    *reinterpret_cast<FP8_E5M2*>(ptr) = FP8_E5M2(static_cast<float>(value));
                    break;
                case DType::QInt8: {
                    if (self.q_scale() == 0.0)
                        throw std::runtime_error("StridedFill: quantized tensor requires "
                                                 "quantization params (call set_quantization_params first)");
                    const int64_t qv = static_cast<int64_t>(std::round(value / self.q_scale()))
                                       + self.q_zero_point();
                    *reinterpret_cast<int8_t*>(ptr) = static_cast<int8_t>(
                        std::clamp(qv, static_cast<int64_t>(-128), static_cast<int64_t>(127)));
                    break;
                }
                case DType::QUInt8: {
                    if (self.q_scale() == 0.0)
                        throw std::runtime_error("StridedFill: quantized tensor requires "
                                                 "quantization params (call set_quantization_params first)");
                    const int64_t qv = static_cast<int64_t>(std::round(value / self.q_scale()))
                                       + self.q_zero_point();
                    *reinterpret_cast<uint8_t*>(ptr) = static_cast<uint8_t>(
                        std::clamp(qv, static_cast<int64_t>(0), static_cast<int64_t>(255)));
                    break;
                }
                case DType::QInt4x2: {
                    if (self.q_scale() == 0.0)
                        throw std::runtime_error("StridedFill: quantized tensor requires "
                                                 "quantization params (call set_quantization_params first)");
                    const int64_t qv = static_cast<int64_t>(std::round(value / self.q_scale()))
                                       + self.q_zero_point();
                    const int64_t clamped = std::clamp(qv, static_cast<int64_t>(-8), static_cast<int64_t>(7));
                    // QInt4x2 packs two 4-bit signed values per byte. Strides are at
                    // byte granularity (the packed shape halves the last dimension), so
                    // each element visited by this stride loop IS a whole byte. Writing
                    // both nibbles is correct: both logical values in a visited byte
                    // belong to the view. Sub-byte strides are not supported by the
                    // framework, so adjacent-nibble corruption is structurally impossible.
                    *reinterpret_cast<uint8_t*>(ptr) =
                        static_cast<uint8_t>((clamped & 0xF) | ((clamped & 0xF) << 4));
                    break;
                }
                default:
                    throw std::runtime_error(
                        std::string("StridedFill: unsupported dtype ") +
                        std::string(dtype_name(dtype)));
            }
        }
        return self;
    });

} // register_cpu_kernels_creation

/**
 * @brief Register CPU kernels for the "fft" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_fft(BackendDispatchTable& table) {
    // =========================================================================
    // FFT Operations (MKL DFTI)
    // =========================================================================
    table.register_single_output_kernel(OpId::FFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::fft_kernel(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::IFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::ifft_kernel(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::RFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::rfft_kernel(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::IRFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, 2 * (inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim] - 1));
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::irfft_kernel(inputs[0], dim, n, norm);
    });

    table.register_single_output_kernel(OpId::FFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            int64_t ndim = inputs[0].ndim();
            dims = {ndim - 2, ndim - 1};
        }
        // Build signal lengths from input shape if not provided via N
        std::vector<int64_t> signal_lengths(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            int64_t d = dims[i] < 0 ? dims[i] + inputs[0].ndim() : dims[i];
            signal_lengths[i] = inputs[0].shape()[d];
        }
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::fft2_kernel(inputs[0], dims, signal_lengths, norm);
    });

    table.register_single_output_kernel(OpId::IFFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            int64_t ndim = inputs[0].ndim();
            dims = {ndim - 2, ndim - 1};
        }
        std::vector<int64_t> signal_lengths(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            int64_t d = dims[i] < 0 ? dims[i] + inputs[0].ndim() : dims[i];
            signal_lengths[i] = inputs[0].shape()[d];
        }
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::ifft2_kernel(inputs[0], dims, signal_lengths, norm);
    });

    table.register_single_output_kernel(OpId::FFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            // Default: all dimensions
            dims.resize(inputs[0].ndim());
            for (int64_t i = 0; i < inputs[0].ndim(); ++i) dims[i] = i;
        }
        std::vector<int64_t> signal_lengths(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            int64_t d = dims[i] < 0 ? dims[i] + inputs[0].ndim() : dims[i];
            signal_lengths[i] = inputs[0].shape()[d];
        }
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::fftn_kernel(inputs[0], dims, signal_lengths, norm);
    });

    table.register_single_output_kernel(OpId::IFFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        if (dims.empty()) {
            dims.resize(inputs[0].ndim());
            for (int64_t i = 0; i < inputs[0].ndim(); ++i) dims[i] = i;
        }
        std::vector<int64_t> signal_lengths(dims.size());
        for (size_t i = 0; i < dims.size(); ++i) {
            int64_t d = dims[i] < 0 ? dims[i] + inputs[0].ndim() : dims[i];
            signal_lengths[i] = inputs[0].shape()[d];
        }
        auto norm = attrs.get_string(AttrKey::Norm, "backward");
        return cpu::ifftn_kernel(inputs[0], dims, signal_lengths, norm);
    });

    // =========================================================================
    // STFT / ISTFT
    // =========================================================================
    table.register_single_output_kernel(OpId::STFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t n_fft = attrs.get_int(AttrKey::NFft);
        int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
        int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
        bool center = attrs.get_bool(AttrKey::Centered, true);
        bool normalized = attrs.get_bool(AttrKey::Normalized, false);
        bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
        // Window is optional second input
        Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
        return cpu::stft_kernel(inputs[0], n_fft, hop_length, win_length, window, center, normalized, onesided);
    });

    table.register_single_output_kernel(OpId::ISTFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t n_fft = attrs.get_int(AttrKey::NFft);
        int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
        int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
        bool center = attrs.get_bool(AttrKey::Centered, true);
        bool normalized = attrs.get_bool(AttrKey::Normalized, false);
        bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
        int64_t length = attrs.get_int(AttrKey::N, -1);
        Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
        return cpu::istft_kernel(inputs[0], n_fft, hop_length, win_length, window, center, normalized, onesided, length);
    });

    // =========================================================================
    // DCT / IDCT
    // =========================================================================
    table.register_single_output_kernel(OpId::DCT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
        std::optional<int64_t> n = std::nullopt;
        int64_t n_val = attrs.get_int(AttrKey::N, -1);
        if (n_val > 0) n = n_val;
        return fft::dct(inputs[0], type, n, dim, norm);
    });

    table.register_single_output_kernel(OpId::IDCT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
        std::optional<int64_t> n = std::nullopt;
        int64_t n_val = attrs.get_int(AttrKey::N, -1);
        if (n_val > 0) n = n_val;
        return fft::idct(inputs[0], type, n, dim, norm);
    });

    table.register_single_output_kernel(OpId::MelScale, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
        double f_min = attrs.get_float(AttrKey::FMin, 0.0);
        double f_max = attrs.get_float(AttrKey::FMax, 0.0);
        int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
        return fft::mel_scale(inputs[0], n_mels, f_min, f_max, sample_rate);
    });

    table.register_single_output_kernel(OpId::MFCC, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
        int64_t n_mfcc = attrs.get_int(AttrKey::NumMFCC, 40);
        int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
        int64_t n_fft = attrs.get_int(AttrKey::NFft, 400);
        int64_t hop_length = attrs.get_int(AttrKey::HopLength, 160);
        double f_min = attrs.get_float(AttrKey::FMin, 0.0);
        double f_max = attrs.get_float(AttrKey::FMax, 0.0);
        return fft::mfcc(inputs[0], sample_rate, n_mfcc, n_mels, n_fft, hop_length, f_min, f_max);
    });

    // =========================================================================
    // CDist
    // =========================================================================
    table.register_single_output_kernel(OpId::CDist, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return cpu::cdist_kernel(inputs[0], inputs[1], p);
    });

    // =========================================================================
    // Trapezoid / Cumulative Trapezoid / Gradient / PairwiseDistance / Pdist
    // =========================================================================
    table.register_single_output_kernel(OpId::Trapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return cpu::trapezoid_kernel(inputs[0], dim, dx, x_ptr);
    });

    table.register_single_output_kernel(OpId::CumulativeTrapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return cpu::cumulative_trapezoid_kernel(inputs[0], dim, dx, x_ptr);
    });

    table.register_single_output_kernel(OpId::NumericalGradient, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double spacing = attrs.get_float(AttrKey::Spacing, 1.0);
        return cpu::gradient_kernel(inputs[0], dim, spacing);
    });

    table.register_single_output_kernel(OpId::PairwiseDistance, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return cpu::pairwise_distance_kernel(inputs[0], inputs[1], p);
    });

    table.register_single_output_kernel(OpId::Pdist, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return cpu::pdist_kernel(inputs[0], p);
    });

    // =========================================================================
    // Multinomial / Bernoulli
    // =========================================================================
    table.register_single_output_kernel(OpId::Multinomial, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t num_samples = attrs.get_int(AttrKey::NumSamples, 1);
        bool replacement = attrs.get_bool(AttrKey::Replacement, false);
        return cpu::multinomial_kernel(inputs[0], num_samples, replacement);
    });

    table.register_single_output_kernel(OpId::Bernoulli, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) -> Tensor {
        return cpu::bernoulli_kernel(inputs[0]);
    });

    table.register_single_output_kernel(OpId::PoissonSample, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) -> Tensor {
        return cpu::poisson_sample_kernel(inputs[0]);
    });

    table.register_single_output_kernel(OpId::NormalSample, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) -> Tensor {
        return cpu::normal_sample_kernel(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::ExponentialSample, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) -> Tensor {
        return cpu::exponential_sample_kernel(inputs[0]);
    });

    // =========================================================================
    // Histogram
    // =========================================================================
    table.register_kernel(OpId::Histogram, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t bins = attrs.get_int(AttrKey::NumBins, 10);
        double min_val = attrs.get_float(AttrKey::Min, 0.0);
        double max_val = attrs.get_float(AttrKey::Max, 0.0);
        auto [counts, edges] = cpu::histogram_kernel(inputs[0], bins, min_val, max_val);
        return {counts, edges};
    });

    // =========================================================================
    // Multi-dimensional Histogram
    // =========================================================================
    table.register_kernel(OpId::Histogramdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // Parse bins list from comma-separated string
        auto bins_list = attrs.get_int_list(AttrKey::BinsList);
        bool density = attrs.get_bool(AttrKey::Density, false);

        // Parse ranges from comma-separated string: min0,max0,min1,max1,...
        std::vector<std::pair<double,double>> ranges;
        auto ranges_str = attrs.get_string(AttrKey::RangesList, "");
        if (!ranges_str.empty()) {
            // Parse comma-separated doubles
            std::vector<double> vals;
            std::string s(ranges_str);
            size_t pos = 0;
            while (pos < s.size()) {
                size_t next = s.find(',', pos);
                if (next == std::string::npos) next = s.size();
                vals.push_back(std::stod(s.substr(pos, next - pos)));
                pos = next + 1;
            }
            for (size_t i = 0; i + 1 < vals.size(); i += 2) {
                ranges.emplace_back(vals[i], vals[i + 1]);
            }
        }

        auto [counts, edges] = cpu::histogramdd_kernel(inputs[0], bins_list, ranges, density);
        std::vector<Tensor> results;
        results.push_back(counts);
        for (auto& e : edges) results.push_back(std::move(e));
        return results;
    });

    // =========================================================================
    // Bucketize
    // =========================================================================
    table.register_single_output_kernel(OpId::Bucketize, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool right = attrs.get_bool(AttrKey::Right, false);
        return cpu::bucketize_kernel(inputs[0], inputs[1], right);
    });

} // register_cpu_kernels_fft

/**
 * @brief Register CPU kernels for the "quantization" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_quantization(BackendDispatchTable& table) {
    // =========================================================================
    // Quantized Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::QuantizedLinear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input_int8, weight_int8] or [input_int8, weight_int8, bias_f32]
        const auto& input = inputs[0];
        const auto& weight = inputs[1];

        auto input_shape = input.shape();
        auto weight_shape = weight.shape();
        int64_t batch_size = input_shape[0];
        int64_t in_features = input_shape[1];
        int64_t out_features = weight_shape[0];

        float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
        float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
        float output_scale = static_cast<float>(attrs.get_float(AttrKey::OutputScale, 1.0));
        int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
        int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));

        Tensor output({batch_size, out_features}, DType::Float32, Device::cpu());

        const int8_t* input_data = input.data<int8_t>();
        const int8_t* weight_data = weight.data<int8_t>();
        const float* bias_data = nullptr;
        if (inputs.size() > 2 && inputs[2].numel() > 0) {
            bias_data = inputs[2].data<const float>();
        }
        float* output_data = output.data<float>();

        // Per-channel path: inputs[3] = weight_scales (Float32, [out_features])
        //                   inputs[4] = weight_zero_points (Int32, [out_features]) (optional)
        bool per_channel = inputs.size() > 3 && inputs[3].numel() > 1;

        if (per_channel) {
            const float* weight_scales_data = inputs[3].data<const float>();
            const int32_t* weight_zps_data = (inputs.size() > 4 && inputs[4].numel() > 0)
                ? inputs[4].data<int32_t>() : nullptr;

            nn::quantization::kernels::quantized_linear_per_channel_kernel(
                input_data, weight_data, bias_data, output_data,
                batch_size, in_features, out_features,
                input_scale, weight_scales_data, output_scale,
                input_zp, weight_zps_data
            );
        } else {
            nn::quantization::kernels::quantized_linear_kernel(
                input_data, weight_data, bias_data, output_data,
                batch_size, in_features, out_features,
                input_scale, weight_scale, output_scale,
                input_zp, weight_zp
            );
        }

        return output;
    });

    table.register_single_output_kernel(OpId::QuantizedConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input_int8, weight_int8] or [input_int8, weight_int8, bias_f32]
        const auto& input = inputs[0];
        const auto& weight = inputs[1];

        auto input_shape = input.shape();
        int64_t batch = input_shape[0];
        int64_t in_channels = input_shape[1];
        int64_t h_in = input_shape[2];
        int64_t w_in = input_shape[3];

        auto weight_shape = weight.shape();
        int64_t out_channels = weight_shape[0];
        int64_t kernel_size = weight_shape[2];

        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);

        float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
        float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
        int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
        int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));

        int64_t h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int64_t w_out = (w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, Device::cpu());

        const int8_t* input_data = input.data<int8_t>();
        const int8_t* weight_data = weight.data<int8_t>();
        const float* bias_data = nullptr;
        if (inputs.size() > 2 && inputs[2].numel() > 0) {
            bias_data = inputs[2].data<const float>();
        }
        float* output_data = output.data<float>();

        // Per-channel path: inputs[3] = weight_scales (Float32, [out_channels])
        //                   inputs[4] = weight_zero_points (Int32, [out_channels]) (optional)
        bool per_channel = inputs.size() > 3 && inputs[3].numel() > 1;

        if (per_channel) {
            const float* weight_scales_data = inputs[3].data<const float>();
            const int32_t* weight_zps_data = (inputs.size() > 4 && inputs[4].numel() > 0)
                ? inputs[4].data<int32_t>() : nullptr;

            nn::quantization::kernels::quantized_conv2d_per_channel_kernel(
                input_data, weight_data, bias_data, output_data,
                batch, in_channels, out_channels,
                h_in, w_in, h_out, w_out,
                kernel_size, stride, padding,
                input_scale, weight_scales_data, input_zp, weight_zps_data,
                dilation, groups
            );
        } else {
            nn::quantization::kernels::quantized_conv2d_kernel(
                input_data, weight_data, bias_data, output_data,
                batch, in_channels, out_channels,
                h_in, w_in, h_out, w_out,
                kernel_size, stride, padding,
                input_scale, weight_scale, input_zp, weight_zp,
                dilation, groups
            );
        }

        return output;
    });

    // =========================================================================
    // GumbelSoftmax (composition of existing dispatched ops)
    // =========================================================================
    table.register_single_output_kernel(OpId::GumbelSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const Tensor& logits = inputs[0];
            double tau = attrs.get_float(AttrKey::Tau, 1.0);
            bool hard = attrs.get_bool(AttrKey::Hard, false);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);

            auto shape_vec = std::vector<int64_t>(logits.shape().begin(), logits.shape().end());

            // Gumbel noise: -log(-log(U)) where U ~ Uniform(0, 1)
            Tensor u = rand(shape_vec, logits.dtype(), logits.device());
            Tensor eps_tensor = full(shape_vec, 1e-20, logits.dtype(), logits.device());
            u = add(u, eps_tensor);  // avoid exact zeros

            // gumbel = -log(-log(u))
            Tensor gumbels = neg(log(neg(log(u))));

            // (logits + gumbels) / tau
            Tensor scaled = div(add(logits, gumbels),
                                full(shape_vec, tau, logits.dtype(), logits.device()));

            // Softmax
            std::array<Tensor, 1> sm_inputs = {scaled};
            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, dim);
            Tensor y_soft = dispatch<OpId::Softmax>(sm_inputs, sm_attrs)[0];

            if (!hard) {
                return y_soft;
            }

            // Straight-through estimator
            int64_t actual_dim = dim < 0 ? dim + logits.ndim() : dim;
            Tensor indices = argmax(y_soft, std::make_optional(actual_dim), /*keepdim=*/true);

            Tensor y_hard = zeros(shape_vec, logits.dtype(), logits.device());
            std::array<Tensor, 3> scatter_inputs = {y_hard, indices,
                full(std::vector<int64_t>(indices.shape().begin(), indices.shape().end()),
                     1.0, logits.dtype(), logits.device())};
            NewOpAttributes scatter_attrs;
            scatter_attrs.set(AttrKey::Dim, actual_dim);
            y_hard = dispatch<OpId::Scatter>(scatter_inputs, scatter_attrs)[0];

            // Forward: y_hard, backward: gradients flow through y_soft
            return add(sub(y_hard, y_soft.detach()), y_soft);
        });

} // register_cpu_kernels_quantization

/**
 * @brief Register CPU kernels for the "complex" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_complex(BackendDispatchTable& table) {
    // =========================================================================
    // Complex Number Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Conj, cpu::conj_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Real, cpu::real_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Imag, cpu::imag_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Angle, cpu::angle_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Polar, cpu::polar_kernel);

} // register_cpu_kernels_complex

/**
 * @brief Register CPU kernels for the "linalg" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_linalg(BackendDispatchTable& table) {
    // =========================================================================
    // Linear Algebra Operations (CPU LAPACKE)
    // =========================================================================
    // These wrap the linalg:: functions which use LAPACKE directly on CPU.
    // Registering them in the dispatch table enables uniform dispatch for all
    // backends and allows GPU→CPU fallback to go through the same code path.

    table.register_single_output_kernel(OpId::LinalgDet, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return linalg::det(inputs[0]);
    });

    table.register_single_output_kernel(OpId::LinalgInv, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return linalg::inv(inputs[0]);
    });

    table.register_single_output_kernel(OpId::LinalgSolve, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return linalg::solve(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::SolveTriangular, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, true);
        bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
        return linalg::solve_triangular(inputs[0], inputs[1], upper, unitriangular);
    });

    table.register_single_output_kernel(OpId::LinalgCholesky, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        return linalg::cholesky(inputs[0], upper);
    });

    table.register_single_output_kernel(OpId::LinalgCholeskySolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        return linalg::cholesky_solve(inputs[0], inputs[1], upper);
    });

    table.register_kernel(OpId::LinalgSVD, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        bool full_matrices = attrs.get_bool(AttrKey::FullMatrices, true);
        auto [U, S, Vt] = linalg::svd(inputs[0], full_matrices);
        return {U, S, Vt};
    });

    table.register_kernel(OpId::LinalgQR, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto [Q, R] = linalg::qr(inputs[0]);
        return {Q, R};
    });

    table.register_kernel(OpId::LinalgEigh, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto [W, V] = linalg::eigh(inputs[0]);
        return {W, V};
    });

    table.register_kernel(OpId::LinalgEig, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto [Re, Im, V] = linalg::eig(inputs[0]);
        return {Re, Im, V};
    });

    table.register_kernel(OpId::LinalgLU, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto [L, U, pivots] = linalg::lu(inputs[0]);
        return {L, U, pivots};
    });

    table.register_single_output_kernel(OpId::LinalgLUSolve, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return linalg::lu_solve(inputs[0], inputs[1], inputs[2]);
    });

} // register_cpu_kernels_linalg

/**
 * @brief Register CPU kernels for the "sparse" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_sparse(BackendDispatchTable& table) {
    // =========================================================================
    // Sparse Tensor Operations (OpIds 460-464)
    //
    // Wrapper lambdas that reconstruct SparseTensor from CSR components passed
    // as plain Tensors, then delegate to the existing sparse:: functions.
    //
    // Convention for inputs:
    //   [0] = crow_indices (Int64, shape {M+1})
    //   [1] = col_indices  (Int64, shape {NNZ})
    //   [2] = values       (Float32/Float64, shape {NNZ})
    //   [3] = dense matrix B (for SpMM/SpMV/SparseAdd)
    // Attrs: M, K, N (dimensions), NNZ
    // =========================================================================

    // SparseSpMM: sparse(M,K) @ dense(K,N) -> dense(M,N)
    table.register_single_output_kernel(OpId::SparseSpMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Audit 8.4: M and K are required; default-0 silently produced wrong shapes.
            if (!attrs.has(AttrKey::M) || !attrs.has(AttrKey::K)) {
                throw std::runtime_error(
                    "SparseSpMM: required attributes M and K not provided. "
                    "Set AttrKey::M (rows of sparse matrix) and AttrKey::K (cols) "
                    "in the OpAttributes before dispatching.");
            }
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::spmm(sp, inputs[3]);
        });

    // SparseSpMV: sparse(M,K) @ vec(K) -> vec(M)
    table.register_single_output_kernel(OpId::SparseSpMV,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // Audit 8.4: M and K are required; default-0 silently produced wrong shapes.
            if (!attrs.has(AttrKey::M) || !attrs.has(AttrKey::K)) {
                throw std::runtime_error(
                    "SparseSpMV: required attributes M and K not provided. "
                    "Set AttrKey::M (rows of sparse matrix) and AttrKey::K (cols) "
                    "in the OpAttributes before dispatching.");
            }
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::spmv(sp, inputs[3]);
        });

    // SparseToDense: CSR components -> dense tensor
    // inputs: [0]=crow_indices, [1]=col_indices, [2]=values
    // attrs: M, K (shape of the sparse matrix)
    table.register_single_output_kernel(OpId::SparseToDense,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sp.to_dense();
        });

    // DenseToSparse: dense tensor -> CSR components [crow_indices, col_indices, values]
    // inputs: [0]=dense tensor
    table.register_kernel(OpId::DenseToSparse,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto sp = SparseTensor::from_dense(inputs[0], SparseLayout::CSR);
            return {sp.crow_indices(), sp.col_indices(), sp.values()};
        });

    // SparseAdd: sparse(M,K) + dense(M,K) -> dense(M,K)
    // inputs: [0]=crow_indices, [1]=col_indices, [2]=values, [3]=dense
    table.register_single_output_kernel(OpId::SparseAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::add(sp, inputs[3]);
        });

    // SparseSpGEMM: sparse(M,K) × sparse(K,N) -> sparse(M,N) in CSR.
    // inputs: [0..2] = A's {crow, col, values}, [3..5] = B's {crow, col, values}
    // attrs: M, K, N. Returns three tensors (crow, col, values) of result.
    // The public sparse::spgemm() skips the dispatch-table path when both
    // operands are on CPU, so dispatching this kernel does not recurse.
    // Underlying CPU implementation is cpu_spgemm_typed<float/double> in
    // src/sparse/sparse_ops.cpp (already exists for Float32 and Float64).
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            int64_t N = attrs.get_int(AttrKey::N);
            auto a = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            auto b = SparseTensor::sparse_csr(inputs[3], inputs[4], inputs[5], {K, N});
            auto c = sparse::spgemm(a, b);
            return {c.crow_indices(), c.col_indices(), c.values()};
        });

    // SparseTrsv: solve L @ x = b (1D RHS), L is lower or upper triangular
    // stored in CSR. inputs: [0..2] = L's {crow, col, values}, [3] = b (N,).
    // attrs: N, Upper. Returns x (N,).
    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return sparse::sparse_triangular_solve(L, inputs[3], upper);
        });

    // SparseTrsm: solve L @ X = B (2D RHS), same semantics as Trsv but b is
    // (N, K). Dispatches into the same public entry point which forwards to
    // cpu_sparse_trsm for 2D b.
    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return sparse::sparse_triangular_solve(L, inputs[3], upper);
        });

    // SparseSoftmax: softmax over non-zero values per row (CSR)
    // inputs: [0]=crow_indices, [1]=col_indices, [2]=values
    table.register_single_output_kernel(OpId::SparseSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], shape);
            auto result = sparse::sparse_softmax(sp);
            return result.values();
        });

    // SparseLogSoftmax: log-softmax over non-zero values per row (CSR)
    table.register_single_output_kernel(OpId::SparseLogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], shape);
            auto result = sparse::sparse_log_softmax(sp);
            return result.values();
        });

} // register_cpu_kernels_sparse

/**
 * @brief Register CPU kernels for the "fused_gemm" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_fused_gemm(BackendDispatchTable& table) {
    // =========================================================================
    // Fused GEMM Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Addmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            return cpu::addmm_kernel(inputs[0], inputs[1], inputs[2], alpha, beta);
        });

    table.register_single_output_kernel(OpId::Addmv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            return cpu::addmv_kernel(inputs[0], inputs[1], inputs[2], alpha, beta);
        });

    table.register_single_output_kernel(OpId::Baddbmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            return cpu::baddbmm_kernel(inputs[0], inputs[1], inputs[2], alpha, beta);
        });

    // =========================================================================
    // Log-Cumulative-Sum-Exp
    // =========================================================================
    table.register_single_output_kernel(OpId::Logcumsumexp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::logcumsumexp_kernel(inputs[0], dim);
    });

    // =========================================================================
    // Bincount
    // =========================================================================
    table.register_single_output_kernel(OpId::Bincount, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t minlength = attrs.get_int(AttrKey::Minlength, 0);
        const Tensor* weights = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return cpu::bincount_kernel(inputs[0], weights, minlength);
    });

    // =========================================================================
    // TakeAlongDim
    // =========================================================================
    table.register_single_output_kernel(OpId::TakeAlongDim, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cpu::take_along_dim_kernel(inputs[0], inputs[1], dim);
    });

    // =========================================================================
    // MaskedScatter
    // =========================================================================
    table.register_single_output_kernel(OpId::MaskedScatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cpu::masked_scatter_kernel(inputs[0], inputs[1], inputs[2]);
    });

    // =========================================================================
    // TrilIndices
    // =========================================================================
    table.register_single_output_kernel(OpId::TrilIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return cpu::tril_indices_kernel(row, col, offset);
    });

    // =========================================================================
    // TriuIndices
    // =========================================================================
    table.register_single_output_kernel(OpId::TriuIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return cpu::triu_indices_kernel(row, col, offset);
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 2D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // RR.8: support PyTorch's output_ratio alternative.  When a positive
        // OutputRatio{H,W} attr is supplied, derive the output extent from the
        // input spatial extent (deterministic ratio mode).
        const auto& in_shape = inputs[0].shape();
        const int64_t in_h = (in_shape.size() >= 2) ? in_shape[in_shape.size() - 2] : 0;
        const int64_t in_w = (in_shape.size() >= 1) ? in_shape[in_shape.size() - 1] : 0;
        const double ratio_h = attrs.get_float(AttrKey::OutputRatioH, 0.0);
        const double ratio_w = attrs.get_float(AttrKey::OutputRatioW, 0.0);
        int64_t out_h = (ratio_h > 0.0)
            ? static_cast<int64_t>(std::floor(static_cast<double>(in_h) * ratio_h))
            : attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = (ratio_w > 0.0)
            ? static_cast<int64_t>(std::floor(static_cast<double>(in_w) * ratio_w))
            : attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = cpu::fractional_maxpool2d_forward_kernel(inputs[0], out_h, out_w, samples);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::fractional_maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 3D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto& in_shape = inputs[0].shape();  // RR.8 ratio mode
        const int64_t in_d = (in_shape.size() >= 3) ? in_shape[in_shape.size() - 3] : 0;
        const int64_t in_h = (in_shape.size() >= 2) ? in_shape[in_shape.size() - 2] : 0;
        const int64_t in_w = (in_shape.size() >= 1) ? in_shape[in_shape.size() - 1] : 0;
        const double ratio_d = attrs.get_float(AttrKey::OutputRatioD, 0.0);
        const double ratio_h = attrs.get_float(AttrKey::OutputRatioH, 0.0);
        const double ratio_w = attrs.get_float(AttrKey::OutputRatioW, 0.0);
        int64_t out_d = (ratio_d > 0.0)
            ? static_cast<int64_t>(std::floor(static_cast<double>(in_d) * ratio_d))
            : attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = (ratio_h > 0.0)
            ? static_cast<int64_t>(std::floor(static_cast<double>(in_h) * ratio_h))
            : attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = (ratio_w > 0.0)
            ? static_cast<int64_t>(std::floor(static_cast<double>(in_w) * ratio_w))
            : attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = cpu::fractional_maxpool3d_forward_kernel(inputs[0], out_d, out_h, out_w, samples);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::fractional_maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // Phase 9: Max Unpool 2D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cpu::max_unpool2d_forward_kernel(inputs[0], inputs[1], out_h, out_w);
    });

    table.register_single_output_kernel(OpId::MaxUnpool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::max_unpool2d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // Phase 9: Max Unpool 3D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cpu::max_unpool3d_forward_kernel(inputs[0], inputs[1], out_d, out_h, out_w);
    });

    table.register_single_output_kernel(OpId::MaxUnpool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::max_unpool3d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // Phase A.1: Max Unpool 1D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_l = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cpu::max_unpool1d_forward_kernel(inputs[0], inputs[1], out_l);
    });

    table.register_single_output_kernel(OpId::MaxUnpool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::max_unpool1d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    // =========================================================================
    // Phase 5: Extended Math Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Deg2Rad, cpu::deg2rad_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Rad2Deg, cpu::rad2deg_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Logit, cpu::logit_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Signbit, cpu::signbit_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsPosInf, cpu::isposinf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsNegInf, cpu::isneginf_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, IsReal, cpu::isreal_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, FloatPower, cpu::float_power_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Xlog1py, cpu::xlog1py_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Ldexp, cpu::ldexp_kernel);

    // Frexp (multi-output: mantissa + exponent)
    table.register_kernel(OpId::Frexp, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        return cpu::frexp_kernel(inputs[0]);
    });

    // DiagEmbed and Diagflat
    table.register_single_output_kernel(OpId::DiagEmbed, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim0, -2);
        int64_t dim2 = attrs.get_int(AttrKey::Dim1, -1);
        return cpu::diag_embed_kernel(inputs[0], offset, dim1, dim2);
    });
    table.register_single_output_kernel(OpId::Diagflat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return cpu::diagflat_kernel(inputs[0], offset);
    });

    // NanVar and NanStd
    table.register_single_output_kernel(OpId::NanVar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, LLONG_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return cpu::nanvar_kernel(inputs[0], dim, keepdim, correction);
    });
    table.register_single_output_kernel(OpId::NanStd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, LLONG_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return cpu::nanstd_kernel(inputs[0], dim, keepdim, correction);
    });

} // register_cpu_kernels_fused_gemm

/**
 * @brief Register CPU kernels for the "nested_misc" category.
 *
 * Stream 25 / Audit-12 split: this used to live inline inside
 * `register_cpu_kernels` (a single ~4070-line function). The body is
 * unchanged; only the enclosing function boundary moved.
 */
static void register_cpu_kernels_nested_misc(BackendDispatchTable& table) {
    // =========================================================================
    // Nested Tensor Operations (OpIds 670-679)
    //
    // Segmented kernels for offset-aware operations on jagged layout.
    // Input convention:
    //   [0] = values   (Float32/Float64, shape {total_len, *regular_dims})
    //   [1] = offsets   (Int64, shape {B+1})
    //   [2..] = additional inputs (weight, bias for layer norm)
    // =========================================================================

    table.register_kernel(OpId::NestedSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {cpu::nested_softmax_kernel(inputs[0], inputs[1], dim)};
        });

    table.register_kernel(OpId::NestedLogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {cpu::nested_log_softmax_kernel(inputs[0], inputs[1], dim)};
        });

    table.register_kernel(OpId::NestedLayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            double eps = attrs.get_float(AttrKey::Eps, 1e-5);
            return {cpu::nested_layer_norm_kernel(inputs[0], inputs[1], inputs[2], inputs[3], eps)};
        });

    table.register_kernel(OpId::NestedSum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {cpu::nested_sum_kernel(inputs[0], inputs[1], dim, keepdim)};
        });

    table.register_kernel(OpId::NestedMean,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {cpu::nested_mean_kernel(inputs[0], inputs[1], dim, keepdim)};
        });

    table.register_kernel(OpId::NestedAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float scale = attrs.get_float(AttrKey::Scale, 1.0f);
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return {cpu::nested_attention_kernel(inputs[0], inputs[1], inputs[2],
                                                 inputs[3], inputs[4], scale, causal)};
        });

    table.register_kernel(OpId::NestedToPadded,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t max_len = attrs.get_int(AttrKey::MaxLen, 0);
            float padding_value = attrs.get_float(AttrKey::PaddingValue, 0.0f);
            return {cpu::nested_to_padded_kernel(inputs[0], inputs[1], max_len, padding_value)};
        });

    table.register_kernel(OpId::NestedFromPadded,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {cpu::nested_from_padded_kernel(inputs[0], inputs[1])};
        });

    table.register_kernel(OpId::NestedLinear,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor* bias = (inputs.size() > 3) ? &inputs[3] : nullptr;
            return {cpu::nested_linear_kernel(inputs[0], inputs[2], bias)};
        });

    // =========================================================================
    // New math ops (PyTorch parity)
    // =========================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogAddExp, cpu::logaddexp_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogAddExp2, cpu::logaddexp2_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, XLogY, cpu::xlogy_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, I0e, cpu::i0e_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, I1e, cpu::i1e_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Entr, cpu::entr_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, SphericalBesselJ0, cpu::spherical_bessel_j0_kernel);

    table.register_single_output_kernel(OpId::Renorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::P, 2.0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        double maxnorm = attrs.get_float(AttrKey::MaxNorm, 1.0);
        return cpu::renorm_kernel(inputs[0], p, dim, maxnorm);
    });
    table.register_single_output_kernel(OpId::CosineSimilarity, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 1);
        double eps = attrs.get_float(AttrKey::Eps, 1e-8);
        return cpu::cosine_similarity_kernel(inputs[0], inputs[1], dim, eps);
    });

    // =========================================================================
    // ComplexTensor, Ormqr, Geqrf CPU kernel registrations
    // =========================================================================

    table.register_single_output_kernel(OpId::ComplexTensor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return tenzor::complex(inputs[0], inputs[1]);
        });

    table.register_single_output_kernel(OpId::Ormqr,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool left = attrs.get_bool(AttrKey::Left, true);
            bool transpose = attrs.get_bool(AttrKey::TransposeQ, false);
            return linalg::ormqr(inputs[0], inputs[1], inputs[2], left, transpose);
        });

    table.register_kernel(OpId::Geqrf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [result, tau] = linalg::geqrf(inputs[0]);
            return {result, tau};
        });

    // =========================================================================
    // LinalgHouseholder, LinalgLDLFactor, LinalgLDLSolve,
    // CholeskyInverse, TensorInv, TensorSolve CPU kernel registrations
    // =========================================================================

    table.register_single_output_kernel(OpId::LinalgHouseholder,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return linalg::householder_product(inputs[0], inputs[1]);
        });

    table.register_kernel(OpId::LinalgLDLFactor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [LD, pivots] = linalg::ldl_factor(inputs[0]);
            return {LD, pivots};
        });

    table.register_single_output_kernel(OpId::LinalgLDLSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return linalg::ldl_solve(inputs[0], inputs[1], inputs[2]);
        });

    table.register_single_output_kernel(OpId::CholeskyInverse,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return linalg::cholesky_inverse(inputs[0], upper);
        });

    table.register_single_output_kernel(OpId::TensorInv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t ind = attrs.get_int(AttrKey::Ind, 2);
            return linalg::tensorinv(inputs[0], ind);
        });

    table.register_single_output_kernel(OpId::TensorSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return linalg::tensorsolve(inputs[0], inputs[1]);
        });

    // =========================================================================
    // Statistical operations (Cov, Corrcoef) — composed from existing ops
    // =========================================================================
    table.register_single_output_kernel(OpId::Cov, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return cov(inputs[0], correction);
    });

    table.register_single_output_kernel(OpId::Corrcoef, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return corrcoef(inputs[0]);
    });

    // =========================================================================
    // AsStrided — metadata-only view with custom shape/strides
    // =========================================================================
    table.register_single_output_kernel(OpId::AsStrided,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto strides = attrs.get_int_list(AttrKey::Strides);
            int64_t offset = attrs.get_int(AttrKey::StorageOffset, -1);
            std::optional<int64_t> storage_offset = (offset >= 0) ? std::optional(offset) : std::nullopt;
            return tenzor::as_strided(inputs[0], shape, strides, storage_offset);
        });

    // =========================================================================
    // LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
    // =========================================================================
    table.register_kernel(OpId::LOBPCG,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t k = attrs.get_int(AttrKey::K, 1);
            int64_t max_iter = attrs.get_int(AttrKey::MaxIter, 100);
            double tol = attrs.get_float(AttrKey::Tolerance, 1e-6);
            Tensor B = inputs.size() > 2 ? inputs[2] : Tensor();
            auto [evals, evecs] = linalg::lobpcg(inputs[0], inputs[1], k, B, max_iter, tol);
            return {evals, evecs};
        });

    // =========================================================================
    // NestedAttentionBackward — backward for segmented attention
    // =========================================================================
    table.register_kernel(OpId::NestedAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [grad_out, Q, K, V, attn_out, q_offsets, kv_offsets]
            float scale = attrs.get_float(AttrKey::Scale, 1.0f);
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return cpu::nested_attention_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                inputs[5], inputs[6], scale, causal);
        });
} // register_cpu_kernels_nested_misc

/**
 * @brief Register all CPU kernels with the dispatch table.
 *
 * This function is called during initialization to populate the CPU
 * dispatch table with direct function pointers to kernel implementations.
 */
void register_cpu_kernels(BackendDispatchTable& table) {
    // Stream 25 / Audit-12: this dispatcher used to inline ~4070 lines of
    // register calls. Each per-category helper below contains the original
    // section verbatim — only the enclosing function boundary changed.
    register_cpu_kernels_arithmetic(table);
    register_cpu_kernels_reduction(table);
    register_cpu_kernels_elementwise_math(table);
    register_cpu_kernels_trigonometric(table);
    register_cpu_kernels_extended_math(table);
    register_cpu_kernels_logical(table);
    register_cpu_kernels_comparison(table);
    register_cpu_kernels_shape(table);
    register_cpu_kernels_indexing(table);
    register_cpu_kernels_normalization(table);
    register_cpu_kernels_convolution(table);
    register_cpu_kernels_pooling(table);
    register_cpu_kernels_linear_fc(table);
    register_cpu_kernels_embedding_dropout(table);
    register_cpu_kernels_vision_pool_misc(table);
    register_cpu_kernels_advanced(table);
    register_cpu_kernels_rmsnorm_etc(table);
    register_cpu_kernels_creation(table);
    register_cpu_kernels_fft(table);
    register_cpu_kernels_quantization(table);
    register_cpu_kernels_complex(table);
    register_cpu_kernels_linalg(table);
    register_cpu_kernels_sparse(table);
    register_cpu_kernels_fused_gemm(table);
    register_cpu_kernels_nested_misc(table);
}

} // namespace tenzor
