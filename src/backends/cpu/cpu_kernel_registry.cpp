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
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
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

    auto sqrt_kernel(const Tensor& input) -> Tensor;
    auto neg_kernel(const Tensor& input) -> Tensor;
    auto abs_kernel(const Tensor& input) -> Tensor;
    auto sign_kernel(const Tensor& input) -> Tensor;
    auto clamp_kernel(const Tensor& input, float min_val, float max_val) -> Tensor;
    auto clamp_min_kernel(const Tensor& input, float min_val) -> Tensor;
    auto clamp_max_kernel(const Tensor& input, float max_val) -> Tensor;
    auto log_kernel(const Tensor& input) -> Tensor;
    auto exp_kernel(const Tensor& input) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent) -> Tensor;
    auto reciprocal_kernel(const Tensor& input) -> Tensor;
    auto floor_kernel(const Tensor& input) -> Tensor;
    auto ceil_kernel(const Tensor& input) -> Tensor;
    auto round_kernel(const Tensor& input) -> Tensor;
    auto trunc_kernel(const Tensor& input) -> Tensor;

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
    auto isnan_kernel(const Tensor& input) -> Tensor;
    auto isinf_kernel(const Tensor& input) -> Tensor;
    auto isfinite_kernel(const Tensor& input) -> Tensor;
    auto atan2_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fmod_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto lerp_kernel(std::span<const Tensor> inputs) -> Tensor;
    auto logical_and_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto logical_or_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto logical_not_kernel(const Tensor& input) -> Tensor;
    auto logical_xor_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto minimum_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto maximum_kernel(const Tensor& a, const Tensor& b) -> Tensor;

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
    auto leaky_relu_inplace_kernel(Tensor& input, float alpha) -> void;
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
    auto leaky_relu_kernel(const Tensor& input, float alpha) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha) -> Tensor;
    auto selu_kernel(const Tensor& input) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input) -> Tensor;
    auto mish_kernel(const Tensor& input) -> Tensor;
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

    // Indexing
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
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
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation) -> Tensor;

    // Conv3d
    auto conv3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv3d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv3d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;

    // ConvTranspose3d
    auto conv_transpose3d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv_transpose3d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv_transpose3d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;

    // Pooling
    auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation = 1) -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto adaptive_avgpool2d_kernel(const Tensor& input, int64_t output_h, int64_t output_w) -> Tensor;
    auto adaptive_avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_maxpool2d_kernel(const Tensor& input, int64_t output_h, int64_t output_w) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;

    // 1D Pooling
    auto maxpool1d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation = 1) -> std::pair<Tensor, Tensor>;
    auto maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool1d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto avgpool1d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto adaptive_avgpool1d_kernel(const Tensor& input, int64_t output_size) -> Tensor;
    auto adaptive_avgpool1d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_maxpool1d_kernel(const Tensor& input, int64_t output_size) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;

    // 3D Pooling
    auto maxpool3d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> std::pair<Tensor, Tensor>;
    auto maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool3d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto avgpool3d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto adaptive_maxpool3d_kernel(const Tensor& input, int64_t output_d, int64_t output_h, int64_t output_w) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_avgpool3d_kernel(const Tensor& input, int64_t output_d, int64_t output_h, int64_t output_w) -> Tensor;
    auto adaptive_avgpool3d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;

    // Fused operations
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_conv2d_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets, bool compute_grad) -> std::vector<Tensor>;
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> std::tuple<Tensor, Tensor, Tensor>;
    auto fused_layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& mean, const Tensor& inv_std, const Tensor& weight) -> std::vector<Tensor>;
    auto fused_conv2d_bn_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor& conv_bias, const Tensor& bn_gamma, const Tensor& bn_beta, const Tensor& bn_running_mean, const Tensor& bn_running_var, int64_t stride, int64_t padding, float bn_momentum, float bn_eps, bool training) -> Tensor;

    // Creation
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, const Device& device) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto arange_kernel(float start, float end, float step, DType dtype, const Device& device) -> Tensor;
    auto linspace_kernel(float start, float end, int64_t steps, DType dtype, const Device& device) -> Tensor;
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
                             const Tensor& bias, const Tensor& h0) -> std::vector<Tensor>;
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
    auto embedding_kernel(const Tensor& weight, const Tensor& indices) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices, int64_t num_embeddings) -> Tensor;
    auto embedding_bag_forward_kernel(std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor;
    auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& embeddings,
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
    auto unique_kernel(const Tensor& input, bool sorted_output,
                       bool return_inverse, bool return_counts)
        -> std::tuple<Tensor, Tensor, Tensor>;

    // RMSNorm operations
    auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps)
        -> std::tuple<Tensor, Tensor>;
    auto rms_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight, const Tensor& rrms)
        -> std::tuple<Tensor, Tensor>;

    // Fused Attention
    auto fused_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                                float scale, bool causal) -> Tensor;

    // Flash Attention (with optional fused dropout for training)
    auto flash_attention_forward(const Tensor& Q, const Tensor& K, const Tensor& V,
                                  float scale, bool causal,
                                  float dropout_p = 0.0f, bool is_training = false) -> Tensor;

    // Flash Attention Backward
    auto flash_attention_backward(const Tensor& dO, const Tensor& Q, const Tensor& K,
                                   const Tensor& V, const Tensor& O,
                                   float scale, bool causal) -> std::vector<Tensor>;

    // Fused Conv2d + Activation variants
    auto fused_conv2d_sigmoid_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                      int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_conv2d_tanh_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                   int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto fused_conv2d_swish_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                    int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;

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
} // namespace nn::quantization::kernels

/**
 * @brief Register all CPU kernels with the dispatch table.
 *
 * This function is called during initialization to populate the CPU
 * dispatch table with direct function pointers to kernel implementations.
 */
void register_cpu_kernels(BackendDispatchTable& table) {
    // =========================================================================
    // Arithmetic Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Add, cpu::add_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Sub, cpu::sub_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Mul, cpu::mul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Div, cpu::div_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, MatMul, cpu::matmul_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Bmm, cpu::bmm_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Dot, cpu::dot_kernel);

    // Single-output registrations for optimized dispatch (no vector allocation)
    // These avoid ~0.5-2us overhead per call from std::vector creation
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, MatMul, cpu::matmul_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Bmm, cpu::bmm_kernel);

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
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        cpu::leaky_relu_inplace_kernel(target, alpha);
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
        cpu::gelu_inplace_kernel(target);
        return target;
    });

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

    table.register_single_output_kernel(OpId::Cast, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        if (!attrs.has(AttrKey::TargetDtype)) {
            throw std::runtime_error("cast: missing 'target_dtype' attribute");
        }
        DType target_dtype = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
        return inputs[0].to(target_dtype);
    });

    table.register_single_output_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float exponent = static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0));
        return cpu::pow_kernel(inputs[0], exponent);
    });

    table.register_single_output_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<float>::infinity()));
        float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity()));
        return cpu::clamp_kernel(inputs[0], min_val, max_val);
    });

    table.register_single_output_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, 0.0));
        return cpu::clamp_min_kernel(inputs[0], min_val);
    });

    table.register_single_output_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, 0.0));
        return cpu::clamp_max_kernel(inputs[0], max_val);
    });

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
    // Logical Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogicalAnd, cpu::logical_and_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogicalOr, cpu::logical_or_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, LogicalNot, cpu::logical_not_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, LogicalXor, cpu::logical_xor_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Minimum, cpu::minimum_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Maximum, cpu::maximum_kernel);

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

    table.register_single_output_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        return cpu::leaky_relu_kernel(inputs[0], alpha);
    });

    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
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

    // Register both multi-output and single-output versions for LayerNorm
    // Multi-output returns {output, mean, rstd} to match CUDA backend (needed for backward pass)
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, mean, rstd] = cpu::layer_norm_kernel_with_stats(inputs[0], normalized_shape, inputs[1], inputs[2], eps);
        return std::vector<Tensor>{output, mean, rstd};
    });

    // Single-output version for optimized dispatch (no vector allocation)
    table.register_single_output_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::layer_norm_kernel(inputs[0], normalized_shape, inputs[1], inputs[2], eps);
    });

    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, rstd]
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        return cpu::layer_norm_backward_kernel(inputs[0], inputs[1], normalized_shape, inputs[3], inputs[4], inputs[2]);
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = attrs.get_int(AttrKey::Groups, 1);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::group_norm_kernel_with_stats(inputs[0], num_groups, inputs[1], inputs[2], eps);
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, rstd]
        int64_t num_groups = attrs.get_int(AttrKey::Groups, 1);
        return cpu::group_norm_backward_kernel(inputs[0], inputs[1], num_groups, inputs[3], inputs[4], inputs[2]);
    });

    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::instance_norm_kernel_with_stats(inputs[0], inputs[1], inputs[2], eps);
    });

    table.register_kernel(OpId::InstanceNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, rstd]
        return cpu::instance_norm_backward_kernel(inputs[0], inputs[1], inputs[3], inputs[4], inputs[2]);
    });

    // =========================================================================
    // Convolution Operations
    // =========================================================================
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{cpu::conv2d_backward_input_kernel(inputs[0], inputs[1], input_shape, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cpu::conv2d_backward_weight_kernel(inputs[0], inputs[1], weight_shape, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv2d_backward_bias_kernel(inputs[0])};
    });

    // Conv3d
    table.register_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv3d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{cpu::conv3d_backward_input_kernel(inputs[0], inputs[1], input_shape, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cpu::conv3d_backward_weight_kernel(inputs[0], inputs[1], weight_shape, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv3d_backward_bias_kernel(inputs[0])};
    });

    // ConvTranspose3d
    table.register_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv_transpose3d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{cpu::conv_transpose3d_backward_input_kernel(inputs[0], inputs[1], input_shape, stride, padding, output_padding, dilation, groups)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cpu::conv_transpose3d_backward_weight_kernel(inputs[0], inputs[1], weight_shape, stride, padding, output_padding, dilation, groups)};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        // Reuse conv3d bias backward - same operation (sum over batch and spatial dims)
        return std::vector<Tensor>{cpu::conv3d_backward_bias_kernel(inputs[0])};
    });

    table.register_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups)};
    });

    table.register_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::depthwise_conv2d_kernel(inputs[0], inputs[1], bias, stride, padding, dilation)};
    });

    // =========================================================================
    // Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        auto [output, indices] = cpu::maxpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cpu::avgpool2d_forward_kernel(inputs[0], kernel_size, stride, padding);
    });

    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cpu::avgpool2d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding);
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
        return cpu::avgpool1d_forward_kernel(inputs[0], kernel_size, stride, padding);
    });

    table.register_single_output_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cpu::avgpool1d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding);
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
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        auto [output, indices] = cpu::maxpool3d_forward_kernel(inputs[0], kernel_size, stride, padding);
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cpu::maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape);
    });

    table.register_single_output_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cpu::avgpool3d_forward_kernel(inputs[0], kernel_size, stride, padding);
    });

    table.register_single_output_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cpu::avgpool3d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding);
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
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_relu_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups);
    });

    table.register_single_output_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cpu::fused_batchnorm_relu_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
    });

    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool compute_grad = attrs.get_bool(AttrKey::ComputeGrad, true);
        return cpu::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], compute_grad);
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
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        float bn_momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        float bn_eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        bool training = attrs.get_bool(AttrKey::Training, false);
        return cpu::fused_conv2d_bn_relu_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5], inputs[6],
                                                  stride, padding, bn_momentum, bn_eps, training);
    });

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

    // =========================================================================
    // Embedding Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_KERNEL(table, Embedding, cpu::embedding_kernel);

    // EmbeddingWithBoundsCheck — same as Embedding since CPU already validates indices
    TENZOR_REGISTER_BINARY_KERNEL(table, EmbeddingWithBoundsCheck, cpu::embedding_kernel);

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
            // inputs[0] = grad_output, inputs[1] = embeddings, inputs[2] = offsets
            return cpu::embedding_bag_backward_kernel(inputs[0], inputs[1], inputs[2], attrs);
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
        // bias may be empty tensor if not provided
        return cpu::gru_forward_kernel(inputs[0], inputs[1], inputs[2],
                                        inputs[3], inputs[4]);
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

    // =========================================================================
    // Vision Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cpu::interpolate_kernel(inputs[0], size, mode, align_corners);
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

    table.register_kernel(OpId::GatherRelativePositionBias,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
            int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
            return {cpu::gather_relative_position_bias_kernel(inputs[0], inputs[1], num_positions, num_heads)};
        });

    // =========================================================================
    // Unfold / Fold Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        return cpu::unfold_kernel(inputs[0], kernel_size, stride, padding, dilation);
    });

    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        return cpu::fold_kernel(inputs[0], output_size, kernel_size, stride, padding, dilation);
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
    // RMSNorm Operations
    // =========================================================================
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
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        float dropout_p = static_cast<float>(attrs.get_float(AttrKey::DropoutP, 0.0));
        bool is_training = attrs.get_bool(AttrKey::Training, false);
        return std::vector<Tensor>{cpu::flash_attention_forward(inputs[0], inputs[1], inputs[2], scale, causal, dropout_p, is_training)};
    });

    // =========================================================================
    // Flash Attention Backward
    // =========================================================================
    table.register_kernel(OpId::FlashAttentionBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        // inputs: [dO, Q, K, V, O]
        return cpu::flash_attention_backward(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], scale, causal);
    });

    // =========================================================================
    // Fused Conv2d + Activation Variants
    // =========================================================================
    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_sigmoid_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups);
    });

    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_tanh_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups);
    });

    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cpu::fused_conv2d_swish_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups);
    });

    // =========================================================================
    // BatchNorm2d Fused Training
    // =========================================================================
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        Tensor running_mean = inputs[1];
        Tensor running_var = inputs[2];
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

    // =========================================================================
    // Creation Operations
    // =========================================================================
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::zeros_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Ones, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::ones_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Full, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::full_kernel(shape, value, dtype, device)};
    });

    table.register_kernel(OpId::Rand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::rand_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Randn, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::randn_kernel(shape, dtype, device)};
    });

    table.register_kernel(OpId::Randint, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t low = attrs.get_int(AttrKey::Start, 0);
        int64_t high = attrs.get_int(AttrKey::End, 0);
        auto shape = attrs.get_int_list(AttrKey::Shape);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Int32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::randint_kernel(low, high, shape, dtype, device)};
    });

    table.register_kernel(OpId::Arange, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
        float end = static_cast<float>(attrs.get_float(AttrKey::End, 0.0));
        float step = static_cast<float>(attrs.get_float(AttrKey::Step, 1.0));
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::arange_kernel(start, end, step, dtype, device)};
    });

    table.register_kernel(OpId::Linspace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
        float end = static_cast<float>(attrs.get_float(AttrKey::End, 1.0));
        int64_t steps = attrs.get_int(AttrKey::Steps, 100);
        int dtype_int = static_cast<int>(attrs.get_int(AttrKey::Dtype, static_cast<int64_t>(DType::Float32)));
        DType dtype = static_cast<DType>(dtype_int);
        Device device = Device::cpu();
        return std::vector<Tensor>{cpu::linspace_kernel(start, end, steps, dtype, device)};
    });

    table.register_kernel(OpId::Eye, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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

    table.register_single_output_kernel(OpId::Trace, [](std::span<const Tensor>, const OpAttributes&) -> Tensor {
        throw std::runtime_error("trace: CPU dispatch not needed (handled inline)");
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
                case DType::Float32:  *reinterpret_cast<float*>(ptr) = static_cast<float>(value); break;
                case DType::Float64:  *reinterpret_cast<double*>(ptr) = value; break;
                case DType::Int32:    *reinterpret_cast<int32_t*>(ptr) = static_cast<int32_t>(value); break;
                case DType::Int64:    *reinterpret_cast<int64_t*>(ptr) = static_cast<int64_t>(value); break;
                case DType::Int16:    *reinterpret_cast<int16_t*>(ptr) = static_cast<int16_t>(value); break;
                case DType::Int8:     *reinterpret_cast<int8_t*>(ptr) = static_cast<int8_t>(value); break;
                case DType::UInt8:    *reinterpret_cast<uint8_t*>(ptr) = static_cast<uint8_t>(value); break;
                case DType::Float16:  *reinterpret_cast<Float16*>(ptr) = Float16(static_cast<float>(value)); break;
                case DType::BFloat16: *reinterpret_cast<BFloat16*>(ptr) = BFloat16(static_cast<float>(value)); break;
                case DType::Bool:     *reinterpret_cast<bool*>(ptr) = (value != 0.0); break;
                default: break;
            }
        }
        return self;
    });

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

        nn::quantization::kernels::quantized_linear_kernel(
            input_data, weight_data, bias_data, output_data,
            batch_size, in_features, out_features,
            input_scale, weight_scale, output_scale,
            input_zp, weight_zp
        );

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

        nn::quantization::kernels::quantized_conv2d_kernel(
            input_data, weight_data, bias_data, output_data,
            batch, in_channels, out_channels,
            h_in, w_in, h_out, w_out,
            kernel_size, stride, padding,
            input_scale, weight_scale, input_zp, weight_zp,
            dilation, groups
        );

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

    // =========================================================================
    // Complex Number Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Conj, cpu::conj_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Real, cpu::real_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Imag, cpu::imag_kernel);
    TENZOR_REGISTER_UNARY_SINGLE_KERNEL(table, Angle, cpu::angle_kernel);
    TENZOR_REGISTER_BINARY_SINGLE_KERNEL(table, Polar, cpu::polar_kernel);

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

    table.register_single_output_kernel(OpId::LinalgCholesky, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        return linalg::cholesky(inputs[0], upper);
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
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::spmm(sp, inputs[3]);
        });

    // SparseSpMV: sparse(M,K) @ vec(K) -> vec(M)
    table.register_single_output_kernel(OpId::SparseSpMV,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
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
}

} // namespace tenzor
