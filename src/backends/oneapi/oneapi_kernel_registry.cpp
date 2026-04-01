/**
 * @file oneapi_kernel_registry.cpp
 * @brief OneAPI kernel registration for O(1) dispatch
 *
 * Registers all OneAPI/SYCL kernel implementations with the dispatch table.
 * Each kernel is a direct function call - no intermediate string dispatch.
 * The SYCL queue is obtained via oneapi_internal::get_queue() which is
 * wired up by the backend constructor.
 */

#include "oneapi_internal.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include <climits>
#include <cmath>
#include <cstdint>
#include <string>
#include <tuple>
#include <vector>

namespace tenzor {

// ============================================================================
// Forward declarations for all OneAPI kernel functions
// ============================================================================
namespace oneapi {

    // ---- Math operations (kernels/math.cpp) ----
    auto add_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto bmm_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto sqrt_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto neg_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto abs_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto log_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto exp_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent, sycl::queue& queue) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // Trigonometric
    auto sin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto tan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto asin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto acos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto atan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sinh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cosh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto atan2_kernel(const Tensor& y, const Tensor& x, sycl::queue& queue) -> Tensor;

    // Rounding
    auto round_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto floor_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto ceil_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto trunc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto reciprocal_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // Extended math
    auto log2_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto log10_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto log1p_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto exp2_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto expm1_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto erf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto erfc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // Bool predicates
    auto isnan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isinf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isfinite_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // Binary math
    auto fmod_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // Ternary
    auto lerp_kernel(const Tensor& start, const Tensor& end, const Tensor& weight, sycl::queue& queue) -> Tensor;

    // Logical
    auto logical_and_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto logical_or_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto logical_not_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto logical_xor_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // Element-wise min/max
    auto minimum_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto maximum_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // Complex number operations
    auto conj_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto real_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto imag_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto angle_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto polar_kernel(const Tensor& abs_t, const Tensor& angle_t, sycl::queue& queue) -> Tensor;
    auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim, sycl::queue& queue) -> Tensor;

    // Additional math
    auto clamp_min_kernel(const Tensor& input, float min_val, sycl::queue& queue) -> Tensor;
    auto clamp_max_kernel(const Tensor& input, float max_val, sycl::queue& queue) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor;
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, sycl::queue& queue) -> Tensor;

    // In-place arithmetic
    auto add_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;
    auto sub_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;
    auto mul_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;
    auto div_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor;

    // ---- Activation functions (kernels/activations.cpp) ----
    auto relu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor;
    auto tanh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor;
    auto gelu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, sycl::queue& queue) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, sycl::queue& queue) -> Tensor;
    auto swish_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;

    // Additional activations
    auto elu_kernel(const Tensor& input, float alpha, sycl::queue& queue) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, sycl::queue& queue) -> Tensor;
    auto selu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto mish_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold, sycl::queue& queue) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, sycl::queue& queue) -> Tensor;
    auto tanh_activation_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // ---- Critical layer operations ----
    auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, sycl::queue& queue) -> Tensor;
    auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, sycl::queue& queue) -> std::vector<Tensor>;
    auto dropout_kernel(const Tensor& input, float p, bool training, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p, sycl::queue& queue) -> Tensor;
    auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                           const Tensor& weight, const Tensor& bias, float eps,
                           sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto instance_norm_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps,
                               sycl::queue& queue) -> std::vector<Tensor>;
    auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                        const Tensor& mean, const Tensor& rstd, const Tensor& weight,
                                        sycl::queue& queue) -> std::vector<Tensor>;

    // In-place activations
    auto relu_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;
    auto sigmoid_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;
    auto tanh_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;
    auto leaky_relu_inplace_kernel(Tensor& input, float alpha, sycl::queue& queue) -> void;
    auto gelu_inplace_kernel(Tensor& input, sycl::queue& queue) -> void;

    // ---- Reduction operations (kernels/reduction.cpp) ----
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto any_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto all_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto median_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> std::vector<Tensor>;
    auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> std::vector<Tensor>;

    // ---- Statistical operations (kernels/statistical.cpp) ----
    auto std_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto var_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto prod_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto norm_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // ---- Transform operations (kernels/transform.cpp) ----
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, sycl::queue& queue) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, sycl::queue& queue) -> Tensor;
    auto triu_kernel(const Tensor& input, int64_t diagonal, sycl::queue& queue) -> Tensor;
    auto tril_kernel(const Tensor& input, int64_t diagonal, sycl::queue& queue) -> Tensor;
    auto diag_kernel(const Tensor& input, int64_t diagonal, sycl::queue& queue) -> Tensor;
    auto trace_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto flip_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, sycl::queue& queue) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim, sycl::queue& queue) -> Tensor;
    auto slice_kernel(const Tensor& input, const std::vector<int64_t>& starts,
                      const std::vector<int64_t>& ends, const std::vector<int64_t>& steps,
                      sycl::queue& queue) -> Tensor;
    auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, sycl::queue& queue) -> std::vector<Tensor>;
    auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, sycl::queue& queue) -> std::vector<Tensor>;
    auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, sycl::queue& queue) -> Tensor;
    auto take_kernel(const Tensor& input, const Tensor& indices, sycl::queue& queue) -> Tensor;
    auto unfold_kernel(const Tensor& input, int64_t kernel_size, int64_t stride,
                        int64_t padding, int64_t dilation, sycl::queue& queue) -> Tensor;
    auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                      int64_t kernel_size, int64_t stride, int64_t padding,
                      int64_t dilation, sycl::queue& queue) -> Tensor;
    auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, sycl::queue& queue) -> Tensor;
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto fill_kernel(const Tensor& tensor, float value, sycl::queue& queue) -> Tensor;

    // ---- Creation operations (kernels/creation.cpp) ----
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto arange_kernel(double start, double end, double step, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto linspace_kernel(double start, double end, int64_t steps, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, sycl::queue& queue) -> Tensor;
    auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape, DType dtype, Device device, sycl::queue& queue) -> Tensor;

    // ---- Comparison operations (kernels/comparison.cpp) ----
    auto eq_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // ---- Utility operations (kernels/utilities.cpp) ----
    auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, sycl::queue& queue) -> Tensor;
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, sycl::queue& queue) -> Tensor;
    auto sign_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // ---- Indexing operations (kernels/indexing.cpp) ----
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index, sycl::queue& queue) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index, sycl::queue& queue) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src, sycl::queue& queue) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask, sycl::queue& queue) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value, sycl::queue& queue) -> Tensor;
    auto nonzero_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes, DType output_dtype, sycl::queue& queue) -> Tensor;
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, sycl::queue& queue) -> Tensor;

    // ---- Batch normalization (kernels/batchnorm.cpp) ----
    auto batchnorm2d_mean_var(const Tensor& input, sycl::queue& queue) -> std::vector<Tensor>;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var,
                                          const Tensor& batch_mean, const Tensor& batch_var,
                                          float momentum, sycl::queue& queue) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance,
                             float epsilon, sycl::queue& queue) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                    const Tensor& gamma, const Tensor& beta, float epsilon,
                                    sycl::queue& queue) -> Tensor;
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean,
                              const Tensor& variance, const Tensor& gamma, float epsilon,
                              sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;

    // Group normalization
    auto group_norm_kernel(const Tensor& input, int64_t num_groups,
                           const Tensor* weight, const Tensor* bias,
                           float eps, sycl::queue& queue) -> std::vector<Tensor>;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd,
                                    const Tensor& weight, int64_t num_groups,
                                    sycl::queue& queue) -> std::vector<Tensor>;

    // ---- Conv2d operations (kernels/conv2d.cpp) ----
    auto conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                        sycl::queue& queue) -> Tensor;
    auto conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                         int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                         bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias,
                         sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                               const std::vector<int64_t>& input_shape,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                               sycl::queue& queue) -> Tensor;
    auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& weight_shape,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                sycl::queue& queue) -> Tensor;
    auto conv2d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor;
    auto conv_transpose2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  int64_t stride, int64_t padding, int64_t output_padding,
                                  int64_t dilation, int64_t groups, sycl::queue& queue) -> Tensor;

    // ---- Conv3d operations (kernels/conv3d.cpp) ----
    auto conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                        const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                        const std::vector<int64_t>& dilation, int64_t groups,
                        sycl::queue& queue) -> Tensor;
    auto conv3d_backward_input(const Tensor& grad_output, const Tensor& weight,
                                const std::vector<int64_t>& input_shape,
                                const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                const std::vector<int64_t>& dilation, int64_t groups,
                                sycl::queue& queue) -> Tensor;
    auto conv3d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                 const std::vector<int64_t>& weight_shape,
                                 const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                 const std::vector<int64_t>& dilation, int64_t groups,
                                 sycl::queue& queue) -> Tensor;
    auto conv3d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor;

    // ---- ConvTranspose3d operations (kernels/conv3d.cpp) ----
    auto conv_transpose3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                   const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                   const std::vector<int64_t>& output_padding,
                                   const std::vector<int64_t>& dilation, int64_t groups,
                                   sycl::queue& queue) -> Tensor;
    auto conv_transpose3d_backward_input(const Tensor& grad_output, const Tensor& weight,
                                          const std::vector<int64_t>& input_shape,
                                          const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                          const std::vector<int64_t>& dilation, int64_t groups,
                                          sycl::queue& queue) -> Tensor;
    auto conv_transpose3d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                           const std::vector<int64_t>& weight_shape,
                                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                           const std::vector<int64_t>& dilation, int64_t groups,
                                           sycl::queue& queue) -> Tensor;
    auto conv_transpose3d_backward_bias(const Tensor& grad_output, sycl::queue& queue) -> Tensor;

    // ---- Pooling operations (kernels/pooling.cpp) ----
    auto avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto adaptive_avg_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto adaptive_max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto avg_pool2d_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto max_pool2d_backward_with_indices(const Tensor& grad_output, const Tensor& indices,
                                          int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor;
    auto adaptive_avgpool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in,
                                     sycl::queue& queue) -> Tensor;
    auto adaptive_maxpool2d_backward(const Tensor& grad_output, const Tensor& indices,
                                     int64_t H_in, int64_t W_in, sycl::queue& queue) -> Tensor;

    // ---- 1D Pooling operations (kernels/pooling.cpp) ----
    auto maxpool1d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                           int64_t padding, sycl::queue& queue) -> std::vector<Tensor>;
    auto maxpool1d_backward(const Tensor& grad_output, const Tensor& indices,
                             const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto avgpool1d_forward(const Tensor& input, int64_t kernel_size, int64_t stride,
                           int64_t padding, sycl::queue& queue) -> Tensor;
    auto avgpool1d_backward(const Tensor& grad_output, int64_t kernel_size, int64_t stride,
                             int64_t padding, const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto adaptive_maxpool1d_forward(const Tensor& input, int64_t output_size,
                                     sycl::queue& queue) -> std::vector<Tensor>;
    auto adaptive_maxpool1d_backward(const Tensor& grad_output, const Tensor& indices,
                                      const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto adaptive_avgpool1d_forward(const Tensor& input, int64_t output_size,
                                     sycl::queue& queue) -> Tensor;
    auto adaptive_avgpool1d_backward(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                      sycl::queue& queue) -> Tensor;

    // ---- 3D Pooling operations (kernels/pooling.cpp) ----
    auto maxpool3d_forward(const Tensor& input, const std::vector<int64_t>& kernel_size,
                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                           sycl::queue& queue) -> std::vector<Tensor>;
    auto maxpool3d_backward(const Tensor& grad_output, const Tensor& indices,
                             const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto avgpool3d_forward(const Tensor& input, const std::vector<int64_t>& kernel_size,
                           const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                           sycl::queue& queue) -> Tensor;
    auto avgpool3d_backward(const Tensor& grad_output, const std::vector<int64_t>& kernel_size,
                             const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                             const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto adaptive_maxpool3d_forward(const Tensor& input, const std::vector<int64_t>& output_size,
                                     sycl::queue& queue) -> std::vector<Tensor>;
    auto adaptive_maxpool3d_backward(const Tensor& grad_output, const Tensor& indices,
                                      const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto adaptive_avgpool3d_forward(const Tensor& input, const std::vector<int64_t>& output_size,
                                     sycl::queue& queue) -> Tensor;
    auto adaptive_avgpool3d_backward(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                      sycl::queue& queue) -> Tensor;

    // ---- Embedding operations (kernels/embedding.cpp) ----
    auto embedding_lookup_kernel(const Tensor& indices, const Tensor& weights,
                                 int64_t padding_idx, sycl::queue& queue) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   int64_t vocab_size, int64_t embedding_dim,
                                   sycl::queue& queue) -> Tensor;
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                      const std::string& mode, bool include_last_offset,
                                      sycl::queue& queue) -> Tensor;
    auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& embeddings,
                                       const Tensor& offsets, const OpAttributes& attrs,
                                       sycl::queue& queue) -> Tensor;

    // ---- Im2col/Col2im operations (kernels/im2col.cpp) ----
    auto im2col_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto col2im_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // ---- Expand operation (kernels/expand.cpp) ----
    auto expand_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // ---- Fused operations (kernels/fused_ops.cpp) ----
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                 const std::vector<int64_t>& normalized_shape, float epsilon,
                                 sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto fused_layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                          const Tensor& mean, const Tensor& inv_std, const Tensor& weight,
                                          const std::vector<int64_t>& normalized_shape,
                                          sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  sycl::queue& queue) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean,
                                     const Tensor& running_var, const Tensor& weight,
                                     const Tensor& bias, float epsilon, sycl::queue& queue) -> Tensor;
    auto fused_matmul_add_kernel(const Tensor& a, const Tensor& b, const Tensor& bias,
                                 sycl::queue& queue) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets,
                                            const std::string& reduction, sycl::queue& queue) -> Tensor;
    auto fused_rms_norm_kernel(const Tensor& input, const Tensor& weight, float eps,
                               sycl::queue& queue) -> std::tuple<Tensor, Tensor>;
    auto rms_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                   const Tensor& weight, const Tensor& rrms,
                                   sycl::queue& queue) -> std::tuple<Tensor, Tensor>;
    auto flash_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                                const Tensor* mask, float scale, bool is_causal,
                                sycl::queue& queue) -> Tensor;

    // ---- Fused optimizer steps (kernels/fused_ops.cpp) ----
    auto fused_adam_step_kernel(
        Tensor& param, const Tensor& grad, Tensor& exp_avg, Tensor& exp_avg_sq,
        double lr, double beta1, double beta2, double eps, double weight_decay,
        int64_t step, bool decoupled_weight_decay, sycl::queue& queue,
        Tensor* max_exp_avg_sq, bool amsgrad) -> void;
    auto fused_sgd_step_kernel(
        Tensor& param, const Tensor& grad, Tensor* momentum_buffer,
        float lr, float momentum, float weight_decay, float dampening,
        bool nesterov, sycl::queue& queue) -> void;
    auto fused_rmsprop_step_kernel(
        Tensor& param, const Tensor& grad, Tensor& square_avg,
        Tensor* grad_avg, Tensor* momentum_buffer,
        float lr, float alpha, float eps, float weight_decay, float momentum,
        bool centered, sycl::queue& queue) -> void;
    auto fused_adadelta_step_kernel(
        Tensor& param, const Tensor& grad, Tensor& square_avg, Tensor& acc_delta,
        float rho, float eps, float lr, float weight_decay,
        sycl::queue& queue) -> void;
    auto fused_adagrad_step_kernel(
        Tensor& param, const Tensor& grad, Tensor& sum_sq,
        float lr, float lr_decay, float eps, float weight_decay,
        int64_t step, sycl::queue& queue) -> void;

    // ---- LSTM operations (kernels/lstm.cpp) ----
    auto lstm_cell_forward_kernel(const Tensor& gates, const Tensor& c_prev,
                                  int64_t batch_size, int64_t hidden_size,
                                  sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_h, const Tensor& grad_c, const Tensor& gates,
                                   const Tensor& c_prev, const Tensor& c_out,
                                   int64_t batch_size, int64_t hidden_size,
                                   sycl::queue& queue) -> std::pair<Tensor, Tensor>;

    // ---- GRU operations (kernels/gru.cpp) ----
    auto gru_cell_forward_kernel(const Tensor& reset_gates, const Tensor& update_gates,
                                 const Tensor& new_gates_input, const Tensor& new_gates_hidden,
                                 const Tensor& h_prev, int64_t batch_size, int64_t hidden_size,
                                 sycl::queue& queue) -> Tensor;
    struct GRUBackwardOutputs {
        Tensor grad_reset;
        Tensor grad_update;
        Tensor grad_new_input;
        Tensor grad_new_hidden;
        Tensor grad_h_prev;
    };
    auto gru_cell_backward_kernel(const Tensor& grad_h, const Tensor& reset_gates,
                                  const Tensor& update_gates, const Tensor& new_gates_input,
                                  const Tensor& new_gates_hidden, const Tensor& h_prev,
                                  int64_t batch_size, int64_t hidden_size,
                                  sycl::queue& queue) -> GRUBackwardOutputs;

    // ---- Vision operations (kernels/vision.cpp) ----
    auto nms_kernel(const Tensor& boxes, const Tensor& scores, float iou_threshold,
                    sycl::queue& queue) -> Tensor;
    auto roi_align_kernel(const Tensor& features, const Tensor& rois,
                          int64_t output_height, int64_t output_width,
                          float spatial_scale, int64_t sampling_ratio, bool aligned,
                          sycl::queue& queue) -> Tensor;
    auto roi_align_backward_kernel(const Tensor& grad_output, const Tensor& rois,
                                   int64_t batch_size, int64_t channels,
                                   int64_t feat_height, int64_t feat_width,
                                   float spatial_scale, int64_t sampling_ratio, bool aligned,
                                   sycl::queue& queue) -> Tensor;
    auto gather_relative_position_bias_kernel(const Tensor& table, const Tensor& indices,
                                              int64_t num_positions, int64_t num_heads,
                                              sycl::queue& queue) -> Tensor;
    auto interpolate_kernel(const Tensor& input, const std::vector<int64_t>& size,
                            const std::string& mode, bool align_corners,
                            sycl::queue& queue) -> Tensor;
    auto box_iou_kernel(const Tensor& boxes1, const Tensor& boxes2, int iou_type,
                        sycl::queue& queue) -> Tensor;

    // ---- Quantization operations (kernels/quantization.cpp) ----
    auto quantize_kernel(const Tensor& input, float scale, int32_t zero_point,
                         sycl::queue& queue) -> Tensor;
    auto dequantize_kernel(const Tensor& input, float scale, int32_t zero_point,
                           sycl::queue& queue) -> Tensor;
    auto quantized_linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 float input_scale, int32_t input_zero_point,
                                 float weight_scale, int32_t weight_zero_point,
                                 float output_scale, int32_t output_zero_point,
                                 sycl::queue& queue) -> Tensor;
    auto quantized_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                 float input_scale, int32_t input_zero_point,
                                 float weight_scale, int32_t weight_zero_point,
                                 sycl::queue& queue) -> Tensor;

    // ---- Full-sequence RNN operations (kernels/rnn.cpp) ----
    auto lstm_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                             const Tensor& bias_ih, const Tensor& bias_hh,
                             const Tensor& h0, const Tensor& c0,
                             sycl::queue& queue) -> std::vector<Tensor>;
    auto gru_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                            const Tensor& bias, const Tensor& h0,
                            sycl::queue& queue) -> std::vector<Tensor>;
    auto lstm_multilayer_forward_kernel(const Tensor& input,
                                        const std::vector<Tensor>& W_ih_list,
                                        const std::vector<Tensor>& W_hh_list,
                                        const std::vector<Tensor>& bias_list,
                                        const Tensor& h0, const Tensor& c0,
                                        sycl::queue& queue) -> std::vector<Tensor>;
    auto gru_multilayer_forward_kernel(const Tensor& input,
                                       const std::vector<Tensor>& W_ih_list,
                                       const std::vector<Tensor>& W_hh_list,
                                       const std::vector<Tensor>& bias_list,
                                       const Tensor& h0,
                                       sycl::queue& queue) -> std::vector<Tensor>;
    auto bilstm_forward_kernel(const Tensor& input, const Tensor& h0, const Tensor& c0,
                                const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
                                const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
                                const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
                                const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
                                sycl::queue& queue) -> std::vector<Tensor>;

    // ---- Sorting / Advanced ops (kernels/reduction.cpp) ----
    auto sort_kernel(const Tensor& input, int64_t dim, bool descending,
                     sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted,
                     sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto unique_kernel(const Tensor& input, bool sorted, bool return_inverse, bool return_counts,
                       sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;

    // ---- Cast / Transform ops (kernels/transform.cpp) ----
    auto cast_kernel(const Tensor& input, DType target_dtype, sycl::queue& queue) -> Tensor;
    auto strided_fill_kernel(Tensor& self, double value, sycl::queue& queue) -> void;
    auto to_memory_format_kernel(const Tensor& input, int format_int, sycl::queue& queue) -> Tensor;

    // ---- ScatterAdd, Put, SearchSorted (kernels/indexing.cpp) ----
    auto scatter_add_kernel(const Tensor& self, int64_t dim, const Tensor& index, const Tensor& src,
                            sycl::queue& queue) -> Tensor;
    auto put_kernel(const Tensor& input, const Tensor& indices, const Tensor& source,
                    bool accumulate, sycl::queue& queue) -> Tensor;
    auto searchsorted_kernel(const Tensor& sorted_sequence, const Tensor& values,
                              bool right, sycl::queue& queue) -> Tensor;

    // ---- HasInfNan, CumSum, CumProd (kernels/math.cpp) ----
    auto has_inf_nan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cumsum_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;

    // ---- DepthwiseConv2d (kernels/conv2d.cpp) ----
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  int64_t stride, int64_t padding, int64_t dilation,
                                  sycl::queue& queue) -> Tensor;

    // ---- Linalg operations (kernels/linalg.cpp) ----
#ifdef TENZOR_HAS_ONEMKL
    auto linalg_det_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto linalg_inv_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto linalg_solve_kernel(const Tensor& A, const Tensor& B, sycl::queue& queue) -> Tensor;
    auto linalg_svd_kernel(const Tensor& input, bool full_matrices,
                            sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_qr_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto linalg_eigh_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto linalg_eig_kernel(const Tensor& input, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_cholesky_kernel(const Tensor& input, bool upper, sycl::queue& queue) -> Tensor;

    // ---- FFT operations (kernels/fft.cpp) ----
    auto fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                    const std::string& norm, sycl::queue& queue) -> Tensor;
    auto ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, sycl::queue& queue) -> Tensor;
    auto rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                     const std::string& norm, sycl::queue& queue) -> Tensor;
    auto irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                      const std::string& norm, sycl::queue& queue) -> Tensor;
    auto fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& signal_lengths,
                     const std::string& norm, sycl::queue& queue) -> Tensor;
    auto ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& signal_lengths,
                      const std::string& norm, sycl::queue& queue) -> Tensor;
    auto fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                     const std::vector<int64_t>& signal_lengths,
                     const std::string& norm, sycl::queue& queue) -> Tensor;
    auto ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                      const std::vector<int64_t>& signal_lengths,
                      const std::string& norm, sycl::queue& queue) -> Tensor;
#endif // TENZOR_HAS_ONEMKL

} // namespace oneapi


// ============================================================================
// Helper: get queue from inputs
// ============================================================================
static sycl::queue& get_q(std::span<const Tensor> inputs) {
    return oneapi_internal::get_queue(inputs[0].device().index);
}

static sycl::queue& get_q_device(int32_t device_id) {
    if (device_id < 0) {
        throw std::runtime_error("Invalid OneAPI device ID: " + std::to_string(device_id));
    }
    try {
        return oneapi_internal::get_queue(device_id);
    } catch (const std::exception& e) {
        throw std::runtime_error("Failed to get OneAPI queue for device " +
                                 std::to_string(device_id) + ": " + e.what());
    }
}

// ============================================================================
// Helper: parse DType from typed attributes
// ============================================================================
static DType parse_dtype(const OpAttributes& attrs) {
    if (!attrs.has(AttrKey::Dtype)) return DType::Float32;

    auto s = attrs.get_string(AttrKey::Dtype, "float32");
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    if (s == "float16") return DType::Float16;
    if (s == "bfloat16") return DType::BFloat16;
    if (s == "int8")    return DType::Int8;
    if (s == "int16")   return DType::Int16;
    if (s == "int32")   return DType::Int32;
    if (s == "int64")   return DType::Int64;
    if (s == "uint8")   return DType::UInt8;
    if (s == "uint16")  return DType::UInt16;
    if (s == "uint32")  return DType::UInt32;
    if (s == "uint64")  return DType::UInt64;
    if (s == "bool")    return DType::Bool;
    return DType::Float32;
}


// ============================================================================
// register_oneapi_kernels - populates the dispatch table
// ============================================================================

void register_oneapi_kernels(BackendDispatchTable& table) {

    // =========================================================================
    // Arithmetic Operations
    // =========================================================================

    table.register_kernel(OpId::Add,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::add_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Sub,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sub_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Mul,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::mul_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Div,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::div_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::MatMul,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::matmul_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Bmm,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::bmm_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Dot,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::dot_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Pow,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float exponent = static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0));
            return {oneapi::pow_kernel(inputs[0], exponent, get_q(inputs))};
        });

    // =========================================================================
    // In-place Arithmetic Operations
    // =========================================================================

    table.register_inplace_kernel(OpId::AddInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::add_inplace_kernel(target, others[0], q);
            return target;
        });

    table.register_inplace_kernel(OpId::SubInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::sub_inplace_kernel(target, others[0], q);
            return target;
        });

    table.register_inplace_kernel(OpId::MulInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::mul_inplace_kernel(target, others[0], q);
            return target;
        });

    table.register_inplace_kernel(OpId::DivInplace,
        [](Tensor& target, std::span<const Tensor> others, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::div_inplace_kernel(target, others[0], q);
            return target;
        });

    // =========================================================================
    // Unary Math Operations
    // =========================================================================

    table.register_kernel(OpId::Sqrt,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sqrt_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Neg,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::neg_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Abs,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::abs_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Sign,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sign_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Log,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::log_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Exp,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::exp_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Reciprocal,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::reciprocal_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Floor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::floor_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Ceil,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::ceil_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Round,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::round_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Trunc,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::trunc_kernel(inputs[0], get_q(inputs))};
        });

    // =========================================================================
    // Trigonometric Operations
    // =========================================================================

    table.register_kernel(OpId::Sin,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sin_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Cos,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::cos_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Tan,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::tan_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Asin,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::asin_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Acos,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::acos_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Atan,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::atan_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Sinh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sinh_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Cosh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::cosh_kernel(inputs[0], get_q(inputs))};
        });

    // Tanh (trigonometric, maps to OpId::Tanh = 58)
    table.register_kernel(OpId::Tanh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::tanh_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Atan2,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::atan2_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // Extended Math Operations
    // =========================================================================

    table.register_kernel(OpId::Log2,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::log2_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Log10,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::log10_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Log1p,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::log1p_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Exp2,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::exp2_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Expm1,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::expm1_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Erf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::erf_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Erfc,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::erfc_kernel(inputs[0], get_q(inputs))};
        });

    // =========================================================================
    // Bool Predicate Operations
    // =========================================================================

    table.register_kernel(OpId::IsNan,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::isnan_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::IsInf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::isinf_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::IsFinite,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::isfinite_kernel(inputs[0], get_q(inputs))};
        });

    // =========================================================================
    // Binary Math Operations (fmod, remainder)
    // =========================================================================

    table.register_kernel(OpId::Fmod,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::fmod_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Remainder,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::remainder_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // Ternary: Lerp
    // =========================================================================

    table.register_kernel(OpId::Lerp,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::lerp_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs))};
        });

    // =========================================================================
    // Logical Operations
    // =========================================================================

    table.register_kernel(OpId::LogicalAnd,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::logical_and_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::LogicalOr,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::logical_or_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::LogicalNot,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::logical_not_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::LogicalXor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::logical_xor_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // Element-wise Minimum / Maximum
    // =========================================================================

    table.register_kernel(OpId::Minimum,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::minimum_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Maximum,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::maximum_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // Complex Number Operations
    // =========================================================================

    table.register_kernel(OpId::Conj,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::conj_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Real,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::real_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Imag,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::imag_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Angle,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::angle_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Polar,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::polar_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_single_output_kernel(OpId::Cross,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return oneapi::cross_kernel(inputs[0], inputs[1], dim, get_q(inputs));
        });

    // =========================================================================
    // Clamp Operations
    // =========================================================================

    table.register_kernel(OpId::Clamp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, 0.0));
            float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, 1.0));
            return {oneapi::clamp_kernel(inputs[0], min_val, max_val, get_q(inputs))};
        });

    table.register_kernel(OpId::ClampMin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, 0.0));
            return {oneapi::clamp_min_kernel(inputs[0], min_val, get_q(inputs))};
        });

    table.register_kernel(OpId::ClampMax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, 1.0));
            return {oneapi::clamp_max_kernel(inputs[0], max_val, get_q(inputs))};
        });

    // =========================================================================
    // Comparison Operations
    // =========================================================================

    table.register_kernel(OpId::Eq,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::eq_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Ne,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::ne_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Lt,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::lt_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Le,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::le_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Gt,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::gt_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Ge,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::ge_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // Reduction Operations
    // =========================================================================

    table.register_kernel(OpId::Sum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::sum_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Mean,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::mean_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Max,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::max_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Min,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::min_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::ArgMax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::argmax_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::ArgMin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::argmin_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Any,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::any_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::All,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::all_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::LogSumExp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::logsumexp_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Median,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return oneapi::median_kernel(inputs[0], dim, keepdim, get_q(inputs));
        });

    table.register_kernel(OpId::Mode,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return oneapi::mode_kernel(inputs[0], dim, keepdim, get_q(inputs));
        });

    // Statistical operations: these OneAPI kernels take OpAttributes directly
    table.register_kernel(OpId::Prod,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::prod_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::Var,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::var_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::Std,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::std_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::Norm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::norm_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::ArgSort,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool descending = attrs.get_bool(AttrKey::Descending, false);
            return {oneapi::argsort_kernel(inputs[0], dim, descending, get_q(inputs))};
        });

    // =========================================================================
    // Activation Functions
    // =========================================================================

    table.register_kernel(OpId::ReLU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::relu_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::ReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::relu_backward_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Sigmoid,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sigmoid_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::SigmoidBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::sigmoid_backward_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::TanhBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::tanh_backward_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Gelu,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::gelu_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::GeluBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::gelu_backward_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Swish,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::swish_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::SwishBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::swish_backward_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::LeakyReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            return {oneapi::leaky_relu_kernel(inputs[0], alpha, get_q(inputs))};
        });

    table.register_kernel(OpId::LeakyReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            return {oneapi::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_q(inputs))};
        });

    table.register_kernel(OpId::Softmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::softmax_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::SoftmaxBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::softmax_backward_kernel(inputs[0], inputs[1], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::LogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::log_softmax_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::LogSoftmaxBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::log_softmax_backward_kernel(inputs[0], inputs[1], dim, get_q(inputs))};
        });

    // =========================================================================
    // In-place Activation Operations
    // =========================================================================

    table.register_inplace_kernel(OpId::ReLUInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::relu_inplace_kernel(target, q);
            return target;
        });

    table.register_inplace_kernel(OpId::SigmoidInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::sigmoid_inplace_kernel(target, q);
            return target;
        });

    table.register_inplace_kernel(OpId::TanhInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::tanh_inplace_kernel(target, q);
            return target;
        });

    table.register_inplace_kernel(OpId::LeakyReLUInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::leaky_relu_inplace_kernel(target, alpha, q);
            return target;
        });

    table.register_inplace_kernel(OpId::GeluInplace,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes&) -> Tensor& {
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::gelu_inplace_kernel(target, q);
            return target;
        });

    // =========================================================================
    // Shape/View Operations
    // =========================================================================

    table.register_kernel(OpId::Reshape,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            return {oneapi::reshape_kernel(inputs[0], shape, get_q(inputs))};
        });

    table.register_kernel(OpId::Transpose,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
            return {oneapi::transpose_kernel(inputs[0], dim0, dim1, get_q(inputs))};
        });

    table.register_kernel(OpId::Permute,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto dims = attrs.get_int_list(AttrKey::Dims);
            return {oneapi::permute_kernel(inputs[0], dims, get_q(inputs))};
        });

    table.register_kernel(OpId::Squeeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::squeeze_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::Unsqueeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::unsqueeze_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::Contiguous,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::contiguous_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Clone,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::clone_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Fill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            return {oneapi::fill_kernel(inputs[0], value, get_q(inputs))};
        });

    table.register_kernel(OpId::Repeat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto repeats = attrs.get_int_list(AttrKey::Repeats);
            return {oneapi::repeat_kernel(inputs[0], repeats, get_q(inputs))};
        });

    table.register_kernel(OpId::Expand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::expand_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::Cat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::cat_kernel(inputs, dim, get_q(inputs))};
        });

    table.register_kernel(OpId::Stack,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            auto& q = get_q(inputs);
            // Stack = unsqueeze each tensor at dim, then cat
            std::vector<Tensor> unsqueezed;
            unsqueezed.reserve(inputs.size());
            for (const auto& t : inputs) {
                unsqueezed.push_back(oneapi::unsqueeze_kernel(t, dim, q));
            }
            return {oneapi::cat_kernel(
                std::span<const Tensor>(unsqueezed.data(), unsqueezed.size()), dim, q)};
        });

    // =========================================================================
    // Indexing Operations
    // =========================================================================

    table.register_kernel(OpId::IndexSelect,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::index_select_kernel(inputs[0], dim, inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Gather,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::gather_kernel(inputs[0], dim, inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Scatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_kernel(OpId::MaskedSelect,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::masked_select_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::MaskedFill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            return {oneapi::masked_fill_kernel(inputs[0], inputs[1], value, get_q(inputs))};
        });

    table.register_kernel(OpId::Where,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::where_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_single_output_kernel(OpId::SearchSorted,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return oneapi::searchsorted_kernel(inputs[0], inputs[1], right, get_q(inputs));
        });

    table.register_kernel(OpId::Nonzero,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::nonzero_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::OneHot,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_classes = attrs.get_int(AttrKey::NumClasses, -1);
            DType output_dtype = DType::Float32;
            if (attrs.has(AttrKey::Dtype)) {
                auto dt = attrs.get_string(AttrKey::Dtype);
                if (dt == "float64") output_dtype = DType::Float64;
                else if (dt == "float16") output_dtype = DType::Float16;
            }
            return {oneapi::one_hot_kernel(inputs[0], num_classes, output_dtype, get_q(inputs))};
        });

    // =========================================================================
    // Convolution Operations
    // =========================================================================

    table.register_kernel(OpId::Conv2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            return {oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                                           stride, padding, dilation, groups, get_q(inputs))};
        });

    // Conv2dBackwardInput: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv2dBackwardInput,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            // inputs[0]=grad_output, inputs[2]=weight
            return {oneapi::conv2d_backward_input(inputs[0], inputs[2], input_shape,
                                                   stride, padding, dilation, groups, get_q(inputs))};
        });

    // Conv2dBackwardWeight: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv2dBackwardWeight,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            // inputs[0]=grad_output, inputs[1]=input
            return {oneapi::conv2d_backward_weight(inputs[0], inputs[1], weight_shape,
                                                    stride, padding, dilation, groups, get_q(inputs))};
        });

    table.register_kernel(OpId::Conv2dBackwardBias,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::conv2d_backward_bias(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::ConvTranspose2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            return {oneapi::conv_transpose2d_forward(inputs[0], inputs[1], bias,
                                                      stride, padding, output_padding,
                                                      dilation, groups, get_q(inputs))};
        });

    // Conv3d operations
    table.register_kernel(OpId::Conv3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto stride = attrs.get_int_list(AttrKey::Stride);
            auto padding = attrs.get_int_list(AttrKey::Padding);
            auto dilation = attrs.get_int_list(AttrKey::Dilation);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {oneapi::conv3d_forward(inputs[0], inputs[1], bias,
                                            stride, padding, dilation, groups, get_q(inputs))};
        });

    // Conv3dBackwardInput: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv3dBackwardInput,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto stride = attrs.get_int_list(AttrKey::Stride);
            auto padding = attrs.get_int_list(AttrKey::Padding);
            auto dilation = attrs.get_int_list(AttrKey::Dilation);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            // inputs[0]=grad_output, inputs[2]=weight
            return {oneapi::conv3d_backward_input(inputs[0], inputs[2], input_shape,
                                                    stride, padding, dilation, groups, get_q(inputs))};
        });

    // Conv3dBackwardWeight: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv3dBackwardWeight,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto stride = attrs.get_int_list(AttrKey::Stride);
            auto padding = attrs.get_int_list(AttrKey::Padding);
            auto dilation = attrs.get_int_list(AttrKey::Dilation);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
            // inputs[0]=grad_output, inputs[1]=input
            return {oneapi::conv3d_backward_weight(inputs[0], inputs[1], weight_shape,
                                                     stride, padding, dilation, groups, get_q(inputs))};
        });

    table.register_kernel(OpId::Conv3dBackwardBias,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::conv3d_backward_bias(inputs[0], get_q(inputs))};
        });

    // ConvTranspose3d operations
    table.register_kernel(OpId::ConvTranspose3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto stride = attrs.get_int_list(AttrKey::Stride);
            auto padding = attrs.get_int_list(AttrKey::Padding);
            auto output_padding = attrs.get_int_list(AttrKey::OutputPadding);
            auto dilation = attrs.get_int_list(AttrKey::Dilation);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {oneapi::conv_transpose3d_forward(inputs[0], inputs[1], bias,
                                                       stride, padding, output_padding,
                                                       dilation, groups, get_q(inputs))};
        });

    table.register_kernel(OpId::ConvTranspose3dBackwardInput,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto stride = attrs.get_int_list(AttrKey::Stride);
            auto padding = attrs.get_int_list(AttrKey::Padding);
            auto dilation = attrs.get_int_list(AttrKey::Dilation);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::conv_transpose3d_backward_input(inputs[0], inputs[1], input_shape,
                                                             stride, padding, dilation,
                                                             groups, get_q(inputs))};
        });

    table.register_kernel(OpId::ConvTranspose3dBackwardWeight,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto stride = attrs.get_int_list(AttrKey::Stride);
            auto padding = attrs.get_int_list(AttrKey::Padding);
            auto dilation = attrs.get_int_list(AttrKey::Dilation);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
            return {oneapi::conv_transpose3d_backward_weight(inputs[0], inputs[1], weight_shape,
                                                              stride, padding, dilation,
                                                              groups, get_q(inputs))};
        });

    table.register_kernel(OpId::ConvTranspose3dBackwardBias,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::conv_transpose3d_backward_bias(inputs[0], get_q(inputs))};
        });

    // =========================================================================
    // Pooling Operations
    // =========================================================================

    table.register_kernel(OpId::AvgPool2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::avg_pool2d_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::MaxPool2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto [output, indices] = oneapi::max_pool2d_kernel(inputs[0], attrs, get_q(inputs));
            return {output, indices};
        });

    table.register_kernel(OpId::AdaptiveAvgPool2d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::adaptive_avg_pool2d_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveMaxPool2d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::adaptive_max_pool2d_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::AvgPool2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            if (inputs.size() >= 2) {
                return {oneapi::avg_pool2d_backward_kernel(inputs[0], inputs[1], attrs, get_q(inputs))};
            } else if (attrs.has(AttrKey::InputShape)) {
                // Autograd path: input shape passed via attributes
                auto input_shape = attrs.get_int_list(AttrKey::InputShape);
                Tensor dummy_input(input_shape, inputs[0].dtype(), inputs[0].device());
                return {oneapi::avg_pool2d_backward_kernel(inputs[0], dummy_input, attrs, get_q(inputs))};
            } else {
                throw std::invalid_argument("avg_pool2d_backward requires 2 inputs or 1 input with input_shape attribute");
            }
        });

    table.register_kernel(OpId::MaxPool2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            int64_t H_in = input_shape.size() >= 3 ? input_shape[2] : 0;
            int64_t W_in = input_shape.size() >= 4 ? input_shape[3] : 0;
            return {oneapi::max_pool2d_backward_with_indices(inputs[0], inputs[1], H_in, W_in, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t H_in = attrs.get_int(AttrKey::InputH, 0);
            int64_t W_in = attrs.get_int(AttrKey::InputW, 0);
            return {oneapi::adaptive_avgpool2d_backward(inputs[0], H_in, W_in, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveMaxPool2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t H_in = attrs.get_int(AttrKey::InputH, 0);
            int64_t W_in = attrs.get_int(AttrKey::InputW, 0);
            return {oneapi::adaptive_maxpool2d_backward(inputs[0], inputs[1], H_in, W_in, get_q(inputs))};
        });

    // =========================================================================
    // 1D Pooling Operations
    // =========================================================================

    table.register_kernel(OpId::MaxPool1dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
            int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            return oneapi::maxpool1d_forward(inputs[0], kernel_size, stride, padding, get_q(inputs));
        });

    table.register_kernel(OpId::MaxPool1dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::maxpool1d_backward(inputs[0], inputs[1], input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AvgPool1dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
            int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            return {oneapi::avgpool1d_forward(inputs[0], kernel_size, stride, padding, get_q(inputs))};
        });

    table.register_kernel(OpId::AvgPool1dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
            int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::avgpool1d_backward(inputs[0], kernel_size, stride, padding, input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveMaxPool1d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
            return oneapi::adaptive_maxpool1d_forward(inputs[0], output_size, get_q(inputs));
        });

    table.register_kernel(OpId::AdaptiveMaxPool1dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::adaptive_maxpool1d_backward(inputs[0], inputs[1], input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool1d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
            return {oneapi::adaptive_avgpool1d_forward(inputs[0], output_size, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool1dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::adaptive_avgpool1d_backward(inputs[0], input_shape, get_q(inputs))};
        });

    // =========================================================================
    // 3D Pooling Operations
    // =========================================================================

    table.register_kernel(OpId::MaxPool3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto ks = attrs.get_int_list(AttrKey::KernelSize);
            auto st = attrs.get_int_list(AttrKey::Stride);
            auto pd = attrs.get_int_list(AttrKey::Padding);
            return oneapi::maxpool3d_forward(inputs[0], ks, st, pd, get_q(inputs));
        });

    table.register_kernel(OpId::MaxPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::maxpool3d_backward(inputs[0], inputs[1], input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AvgPool3dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto ks = attrs.get_int_list(AttrKey::KernelSize);
            auto st = attrs.get_int_list(AttrKey::Stride);
            auto pd = attrs.get_int_list(AttrKey::Padding);
            return {oneapi::avgpool3d_forward(inputs[0], ks, st, pd, get_q(inputs))};
        });

    table.register_kernel(OpId::AvgPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto ks = attrs.get_int_list(AttrKey::KernelSize);
            auto st = attrs.get_int_list(AttrKey::Stride);
            auto pd = attrs.get_int_list(AttrKey::Padding);
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::avgpool3d_backward(inputs[0], ks, st, pd, input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveMaxPool3d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto output_size = attrs.get_int_list(AttrKey::OutputSize);
            return oneapi::adaptive_maxpool3d_forward(inputs[0], output_size, get_q(inputs));
        });

    table.register_kernel(OpId::AdaptiveMaxPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::adaptive_maxpool3d_backward(inputs[0], inputs[1], input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool3d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto output_size = attrs.get_int_list(AttrKey::OutputSize);
            return {oneapi::adaptive_avgpool3d_forward(inputs[0], output_size, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::adaptive_avgpool3d_backward(inputs[0], input_shape, get_q(inputs))};
        });

    // =========================================================================
    // Batch Normalization Operations
    // =========================================================================

    table.register_kernel(OpId::BatchNorm2dMeanVar,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return oneapi::batchnorm2d_mean_var(inputs[0], get_q(inputs));
        });

    table.register_kernel(OpId::BatchNorm2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return {oneapi::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_q(inputs))};
        });

    table.register_kernel(OpId::BatchNorm2dForwardAffine,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return {oneapi::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2],
                                                        inputs[3], inputs[4], epsilon, get_q(inputs))};
        });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
            Tensor updated_mean = inputs[0].clone();
            Tensor updated_var = inputs[1].clone();
            auto& q = get_q(inputs);
            oneapi::batchnorm2d_update_running_stats(updated_mean, updated_var,
                                                     inputs[2], inputs[3], momentum, q);
            return {updated_mean, updated_var};
        });

    table.register_kernel(OpId::BatchNorm2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [grad_input, grad_gamma, grad_beta] = oneapi::batchnorm2d_backward(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_q(inputs));
            return {grad_input, grad_gamma, grad_beta};
        });

    // =========================================================================
    // Group Normalization Operations
    // =========================================================================

    table.register_kernel(OpId::GroupNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return oneapi::group_norm_kernel(inputs[0], num_groups, weight, bias, eps, get_q(inputs));
        });

    table.register_kernel(OpId::GroupNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
            return oneapi::group_norm_backward_kernel(inputs[0], inputs[1], inputs[2],
                                                      inputs[3], inputs[4], num_groups, get_q(inputs));
        });

    // =========================================================================
    // Creation Operations
    // =========================================================================

    table.register_kernel(OpId::Zeros,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::zeros_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Ones,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::ones_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Full,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::full_kernel(shape, value, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Rand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::rand_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Randn,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::randn_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Randint,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t low = attrs.get_int(AttrKey::Start, 0);
            int64_t high = attrs.get_int(AttrKey::End, 0);
            auto shape = attrs.get_int_list(AttrKey::Shape);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::randint_kernel(low, high, shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Arange,
        [](std::span<const Tensor>, const OpAttributes& attrs) -> std::vector<Tensor> {
            double start = attrs.get_float(AttrKey::Start, 0.0);
            double end_val = attrs.get_float(AttrKey::End, 0.0);
            double step = attrs.get_float(AttrKey::Step, 1.0);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            return {oneapi::arange_kernel(start, end_val, step, dtype,
                                           Device(Device::Type::OneAPI, device_id),
                                           get_q_device(device_id))};
        });

    table.register_kernel(OpId::Linspace,
        [](std::span<const Tensor>, const OpAttributes& attrs) -> std::vector<Tensor> {
            double start = attrs.get_float(AttrKey::Start, 0.0);
            double end_val = attrs.get_float(AttrKey::End, 1.0);
            int64_t steps = attrs.get_int(AttrKey::Steps, 100);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            return {oneapi::linspace_kernel(start, end_val, steps, dtype,
                                             Device(Device::Type::OneAPI, device_id),
                                             get_q_device(device_id))};
        });

    table.register_kernel(OpId::Eye,
        [](std::span<const Tensor>, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t n = attrs.get_int(AttrKey::N, 1);
            int64_t m = attrs.get_int(AttrKey::M, n);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
            return {oneapi::eye_kernel(n, m, dtype,
                                        Device(Device::Type::OneAPI, device_id),
                                        get_q_device(device_id))};
        });

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    table.register_kernel(OpId::Embedding,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t padding_idx = attrs.get_int(AttrKey::PaddingIdx, -1);
            // inputs[0] = weights, inputs[1] = indices (from nn::Embedding layer)
            return {oneapi::embedding_lookup_kernel(inputs[1], inputs[0], padding_idx, get_q(inputs))};
        });

    table.register_kernel(OpId::EmbeddingWithBoundsCheck,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t padding_idx = attrs.get_int(AttrKey::PaddingIdx, -1);
            return {oneapi::embedding_lookup_kernel(inputs[1], inputs[0], padding_idx, get_q(inputs))};
        });

    table.register_kernel(OpId::EmbeddingBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t vocab_size = attrs.get_int(AttrKey::NumEmbeddings, 0);
            int64_t embedding_dim = inputs[0].shape().back();
            return {oneapi::embedding_backward_kernel(inputs[0], inputs[1], vocab_size,
                                                       embedding_dim, get_q(inputs))};
        });

    // =========================================================================
    // RNN Operations
    // =========================================================================

    table.register_kernel(OpId::LSTMCellForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
            int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
            auto [h_out, c_out] = oneapi::lstm_cell_forward_kernel(
                inputs[0], inputs[1], batch_size, hidden_size, get_q(inputs));
            return {h_out, c_out};
        });

    table.register_kernel(OpId::LSTMCellBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
            int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
            auto [grad_gates, grad_c_prev] = oneapi::lstm_cell_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                batch_size, hidden_size, get_q(inputs));
            return {grad_gates, grad_c_prev};
        });

    table.register_kernel(OpId::GRUCellForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
            int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
            return {oneapi::gru_cell_forward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                batch_size, hidden_size, get_q(inputs))};
        });

    table.register_kernel(OpId::GRUCellBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
            int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
            auto outputs = oneapi::gru_cell_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
                batch_size, hidden_size, get_q(inputs));
            return {outputs.grad_reset, outputs.grad_update, outputs.grad_new_input,
                    outputs.grad_new_hidden, outputs.grad_h_prev};
        });

    // =========================================================================
    // Fused Operations
    // =========================================================================

    table.register_kernel(OpId::FusedAddReLU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::fused_add_relu_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::FusedGelu,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::fused_gelu_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::FusedLayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [output, mean, inv_std] = oneapi::fused_layer_norm_kernel(
                inputs[0], inputs[1], inputs[2], normalized_shape, epsilon, get_q(inputs));
            return {output, mean, inv_std};
        });

    table.register_kernel(OpId::LayerNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
            auto [grad_input, grad_weight, grad_bias] = oneapi::fused_layer_norm_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape, get_q(inputs));
            return {grad_input, grad_weight, grad_bias};
        });

    // Note: OpId::FusedLayerNormBackward also maps to fused_layer_norm_backward
    table.register_kernel(OpId::FusedLayerNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
            auto [grad_input, grad_weight, grad_bias] = oneapi::fused_layer_norm_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape, get_q(inputs));
            return {grad_input, grad_weight, grad_bias};
        });

    table.register_kernel(OpId::FusedLinearReLU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {oneapi::fused_linear_relu_kernel(inputs[0], inputs[1], bias, get_q(inputs))};
        });

    table.register_kernel(OpId::FusedBatchNormReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return {oneapi::fused_batchnorm_relu_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_q(inputs))};
        });

    table.register_kernel(OpId::FusedSoftmaxCrossEntropy,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string reduction = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
            return {oneapi::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], reduction, get_q(inputs))};
        });

    table.register_kernel(OpId::FusedRMSNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [output, rrms] = oneapi::fused_rms_norm_kernel(inputs[0], inputs[1], epsilon, get_q(inputs));
            return {output, rrms};
        });

    table.register_kernel(OpId::RMSNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            // Dispatch sends: {grad_output, input, rrms, weight}
            // Kernel expects: (grad_output, input, weight, rrms)
            auto [grad_input, grad_weight] = oneapi::rms_norm_backward_kernel(
                inputs[0], inputs[1], inputs[3], inputs[2], get_q(inputs));
            return {grad_input, grad_weight};
        });

    // =========================================================================
    // Fused Optimizer Steps
    // =========================================================================

    table.register_kernel(OpId::FusedSGDStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [param, grad, momentum_buffer (optional)]
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
            float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
            float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
            float dampening = static_cast<float>(attrs.get_float(AttrKey::Dampening, 0.0));
            bool nesterov = attrs.get_bool(AttrKey::Nesterov, false);

            Tensor& param = const_cast<Tensor&>(inputs[0]);
            Tensor* momentum_buffer = (inputs.size() > 2 && momentum > 0.0f)
                ? &const_cast<Tensor&>(inputs[2]) : nullptr;

            oneapi::fused_sgd_step_kernel(
                param, inputs[1], momentum_buffer,
                lr, momentum, weight_decay, dampening, nesterov,
                get_q(inputs));
            return std::vector<Tensor>{param};
        });

    table.register_kernel(OpId::FusedAdamStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [param, grad, exp_avg, exp_avg_sq, packed_params, max_exp_avg_sq (optional)]
            double lr, beta1, beta2, eps, weight_decay;
            int64_t step;
            bool decoupled, amsgrad;

            if (inputs.size() >= 5 && inputs[4].dtype() == DType::Float64 && inputs[4].numel() == 8) {
                // New packed-tensor path
                const double* p = inputs[4].data<double>();
                lr = p[0];
                beta1 = p[1];
                beta2 = p[2];
                eps = p[3];
                weight_decay = p[4];
                step = static_cast<int64_t>(p[5]);
                decoupled = p[6] != 0.0;
                amsgrad = p[7] != 0.0;
            } else {
                // Legacy attribute path
                lr = attrs.get_float(AttrKey::Lr, 0.001);
                beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
                beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
                eps = attrs.get_float(AttrKey::Eps, 1e-8);
                weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
                step = attrs.get_int(AttrKey::Step, 1);
                decoupled = attrs.get_bool(AttrKey::Decoupled, false);
                amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);
            }

            Tensor& param = const_cast<Tensor&>(inputs[0]);
            Tensor& exp_avg = const_cast<Tensor&>(inputs[2]);
            Tensor& exp_avg_sq = const_cast<Tensor&>(inputs[3]);
            Tensor* max_exp_avg_sq = (amsgrad && inputs.size() > 5)
                ? &const_cast<Tensor&>(inputs[5]) : nullptr;

            oneapi::fused_adam_step_kernel(
                param, inputs[1], exp_avg, exp_avg_sq,
                lr, beta1, beta2, eps, weight_decay, step, decoupled,
                get_q(inputs), max_exp_avg_sq, amsgrad);
            return std::vector<Tensor>{param};
        });

    table.register_kernel(OpId::FusedRMSPropStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [param, grad, square_avg, grad_avg (optional), momentum_buffer (optional)]
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

            oneapi::fused_rmsprop_step_kernel(param, inputs[1], square_avg, grad_avg, momentum_buffer,
                lr, alpha, eps, weight_decay, momentum, centered, get_q(inputs));
            return std::vector<Tensor>{param};
        });

    table.register_kernel(OpId::FusedAdadeltaStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [param, grad, square_avg, acc_delta]
            float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
            float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

            Tensor& param = const_cast<Tensor&>(inputs[0]);
            Tensor& square_avg = const_cast<Tensor&>(inputs[2]);
            Tensor& acc_delta = const_cast<Tensor&>(inputs[3]);

            oneapi::fused_adadelta_step_kernel(param, inputs[1], square_avg, acc_delta,
                rho, eps, lr, weight_decay, get_q(inputs));
            return std::vector<Tensor>{param};
        });

    table.register_kernel(OpId::FusedAdagradStep,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [param, grad, sum_sq]
            float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
            float lr_decay = static_cast<float>(attrs.get_float(AttrKey::LrDecay, 0.0));
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
            float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
            int64_t step = attrs.get_int(AttrKey::Step, 1);

            Tensor& param = const_cast<Tensor&>(inputs[0]);
            Tensor& sum_sq = const_cast<Tensor&>(inputs[2]);

            oneapi::fused_adagrad_step_kernel(param, inputs[1], sum_sq,
                lr, lr_decay, eps, weight_decay, step, get_q(inputs));
            return std::vector<Tensor>{param};
        });

    // =========================================================================
    // Fused Conv2D + Activation Variants (compose conv2d + activation)
    // =========================================================================

    table.register_single_output_kernel(OpId::FusedConv2dReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            auto& queue = get_q(inputs);
            Tensor conv_out = oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                stride, padding, dilation, groups, queue);
            return oneapi::relu_kernel(conv_out, queue);
        });

    table.register_single_output_kernel(OpId::FusedConv2dSigmoid,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            auto& queue = get_q(inputs);
            Tensor conv_out = oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                stride, padding, dilation, groups, queue);
            return oneapi::sigmoid_kernel(conv_out, queue);
        });

    table.register_single_output_kernel(OpId::FusedConv2dTanh,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            auto& queue = get_q(inputs);
            Tensor conv_out = oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                stride, padding, dilation, groups, queue);
            return oneapi::tanh_kernel(conv_out, queue);
        });

    table.register_single_output_kernel(OpId::FusedConv2dSwish,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            auto& queue = get_q(inputs);
            Tensor conv_out = oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                stride, padding, dilation, groups, queue);
            return oneapi::swish_kernel(conv_out, queue);
        });

    // =========================================================================
    // Fused Conv2D + BatchNorm + ReLU (full pipeline)
    // =========================================================================

    table.register_single_output_kernel(OpId::FusedConv2dBnReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            // inputs: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            const Tensor* bias = inputs.size() > 2 && inputs[2].numel() > 0 ? &inputs[2] : nullptr;
            auto& queue = get_q(inputs);

            // Step 1: Conv2d forward
            Tensor conv_out = oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                stride, padding, 1, 1, queue);

            // Step 2: BatchNorm forward with affine (using bn_running_mean/var as mean/var for inference-style)
            Tensor bn_out = oneapi::batchnorm2d_forward_affine(
                conv_out, inputs[5], inputs[6], inputs[3], inputs[4], eps, queue);

            // Step 3: ReLU
            return oneapi::relu_kernel(bn_out, queue);
        });

    // =========================================================================
    // Fused Attention (Q*K^T/scale -> softmax -> *V)
    // =========================================================================

    table.register_kernel(OpId::FusedAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [Q, K, V]
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            auto& queue = get_q(inputs);

            // Step 1: QK^T via batched matmul — transpose K by permuting last two dims
            auto k_shape = inputs[1].shape();
            int64_t ndim = static_cast<int64_t>(k_shape.size());
            Tensor kt = oneapi::transpose_kernel(inputs[1], ndim - 2, ndim - 1, queue);

            // Step 2: scores = Q @ K^T
            Tensor scores = oneapi::matmul_kernel(inputs[0], kt, queue);

            // Step 3: Scale by 1/scale (scale attr is typically sqrt(d_k))
            Tensor scale_tensor = oneapi::full_kernel({1}, scale, scores.dtype(), scores.device(), queue);
            scores = oneapi::div_kernel(scores, scale_tensor, queue);

            // Step 4: Softmax over last dimension
            Tensor attn_weights = oneapi::softmax_kernel(scores, -1, queue);

            // Step 5: attn_weights @ V
            Tensor output = oneapi::matmul_kernel(attn_weights, inputs[2], queue);
            return {output};
        });

    // =========================================================================
    // Flash Attention (memory-efficient tiled attention with online softmax)
    // =========================================================================

    table.register_kernel(OpId::FlashAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [Q, K, V] or [Q, K, V, mask]
            auto& queue = get_q(inputs);

            // Infer head_dim from Q shape (last dimension)
            int64_t head_dim = inputs[0].shape().back();
            float default_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, default_scale));
            bool is_causal = attrs.get_bool(AttrKey::Causal, false);

            const Tensor* mask = (inputs.size() > 3) ? &inputs[3] : nullptr;

            Tensor output = oneapi::flash_attention_kernel(
                inputs[0], inputs[1], inputs[2], mask, scale, is_causal, queue);
            return {output};
        });

    // =========================================================================
    // Flash Attention Backward (composed-ops fallback using high-level ops)
    // =========================================================================
    table.register_kernel(OpId::FlashAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [dO, Q, K, V, O] — dO = grad_output, O = forward output
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);

            const Tensor& dO = inputs[0];  // [B, H, S, D]
            const Tensor& Q = inputs[1];
            const Tensor& K = inputs[2];
            const Tensor& V = inputs[3];

            // Recompute attention weights: attn = softmax(Q @ K^T * scale)
            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::bmm(Q, Kt);  // [B, H, S, S]

            // Scale
            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = tenzor::mul(scores, scale_t);

            // Apply causal mask if needed
            if (causal) {
                int64_t seq_len = scores_shape[scores_shape.size() - 1];
                Tensor rows = tenzor::arange(0, seq_len, 1, DType::Int64, scores.device());
                Tensor cols = tenzor::arange(0, seq_len, 1, DType::Int64, scores.device());
                rows = tenzor::reshape(rows, {seq_len, 1});
                cols = tenzor::reshape(cols, {1, seq_len});
                Tensor causal_mask = tenzor::gt(cols.to(DType::Float32), rows.to(DType::Float32));
                Tensor neg_inf = tenzor::full(scores_shape, -1e9,
                                              scores.dtype(), scores.device());
                scores = tenzor::add(scores, tenzor::mul(causal_mask.to(scores.dtype()), neg_inf));
            }

            // Softmax along last dim
            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_inputs = {scores};
            Tensor attn_weights = tenzor::dispatch(OpId::Softmax, sm_inputs, sm_attrs)[0];

            // dV = attn^T @ dO
            Tensor attn_t = tenzor::transpose(attn_weights, -1, -2);
            Tensor dV = tenzor::bmm(attn_t, dO);

            // dAttn = dO @ V^T
            Tensor Vt = tenzor::transpose(V, -1, -2);
            Tensor dAttn = tenzor::bmm(dO, Vt);

            // softmax backward: ds = attn * (dAttn - sum(attn * dAttn, dim=-1, keepdim=true))
            Tensor attn_dAttn = tenzor::mul(attn_weights, dAttn);
            NewOpAttributes sum_attrs;
            sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            sum_attrs.set(AttrKey::Keepdim, true);
            std::vector<Tensor> sum_inputs = {attn_dAttn};
            Tensor sum_ad = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
            Tensor dScores = tenzor::mul(attn_weights, tenzor::sub(dAttn, sum_ad));

            // Apply scale
            Tensor scale_t2 = tenzor::full(
                std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
                static_cast<double>(scale), dScores.dtype(), dScores.device());
            dScores = tenzor::mul(dScores, scale_t2);

            // dQ = dScores @ K, dK = dScores^T @ Q
            Tensor dQ = tenzor::bmm(dScores, K);
            Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
            Tensor dK = tenzor::bmm(dScores_t, Q);

            return {dQ, dK, dV};
        });

    // =========================================================================
    // BatchNorm2d Fused Training (compose mean_var + forward_affine + update_running_stats)
    // =========================================================================

    table.register_kernel(OpId::BatchNorm2dFusedTraining,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [input, running_mean, running_var, gamma, beta]
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
            auto& queue = get_q(inputs);

            // Step 1: Compute batch mean and variance
            auto mean_var = oneapi::batchnorm2d_mean_var(inputs[0], queue);
            Tensor& mean = mean_var[0];
            Tensor& variance = mean_var[1];

            // Step 2: Normalize with affine transform
            Tensor output = oneapi::batchnorm2d_forward_affine(
                inputs[0], mean, variance, inputs[3], inputs[4], epsilon, queue);

            // Step 3: Update running stats
            Tensor running_mean = inputs[1];
            Tensor running_var = inputs[2];
            oneapi::batchnorm2d_update_running_stats(
                running_mean, running_var, mean, variance, momentum, queue);

            return {output, mean, variance, running_mean, running_var};
        });

    // =========================================================================
    // Fused Adam-Atan2 Optimizer Step
    // =========================================================================

    table.register_kernel(OpId::FusedAdamAtan2Step,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq (optional)]
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
            auto& queue = get_q(inputs);

            // Bias correction
            float bc1 = 1.0f - std::pow(beta1, static_cast<float>(step));
            float bc2 = 1.0f - std::pow(beta2, static_cast<float>(step));

            // Weight decay (decoupled)
            if (weight_decay != 0.0f) {
                Tensor wd_scale = oneapi::full_kernel({1}, 1.0f - lr * weight_decay,
                    param.dtype(), param.device(), queue);
                oneapi::mul_inplace_kernel(param, wd_scale, queue);
            }

            // Update biased first moment estimate: exp_avg = beta1 * exp_avg + (1 - beta1) * grad
            Tensor beta1_t = oneapi::full_kernel({1}, beta1, exp_avg.dtype(), exp_avg.device(), queue);
            Tensor one_minus_beta1_t = oneapi::full_kernel({1}, 1.0f - beta1, exp_avg.dtype(), exp_avg.device(), queue);
            Tensor scaled_avg = oneapi::mul_kernel(exp_avg, beta1_t, queue);
            Tensor scaled_grad = oneapi::mul_kernel(inputs[1], one_minus_beta1_t, queue);
            exp_avg = oneapi::add_kernel(scaled_avg, scaled_grad, queue);

            // Update biased second moment estimate: exp_avg_sq = beta2 * exp_avg_sq + (1 - beta2) * grad^2
            Tensor beta2_t = oneapi::full_kernel({1}, beta2, exp_avg_sq.dtype(), exp_avg_sq.device(), queue);
            Tensor one_minus_beta2_t = oneapi::full_kernel({1}, 1.0f - beta2, exp_avg_sq.dtype(), exp_avg_sq.device(), queue);
            Tensor grad_sq = oneapi::mul_kernel(inputs[1], inputs[1], queue);
            Tensor scaled_sq = oneapi::mul_kernel(exp_avg_sq, beta2_t, queue);
            Tensor scaled_grad_sq = oneapi::mul_kernel(grad_sq, one_minus_beta2_t, queue);
            exp_avg_sq = oneapi::add_kernel(scaled_sq, scaled_grad_sq, queue);

            // Bias-corrected estimates
            Tensor bc1_t = oneapi::full_kernel({1}, bc1, exp_avg.dtype(), exp_avg.device(), queue);
            Tensor bc2_t = oneapi::full_kernel({1}, bc2, exp_avg_sq.dtype(), exp_avg_sq.device(), queue);
            Tensor m_hat = oneapi::div_kernel(exp_avg, bc1_t, queue);
            Tensor v_hat = oneapi::div_kernel(exp_avg_sq, bc2_t, queue);

            if (amsgrad && max_exp_avg_sq) {
                *max_exp_avg_sq = oneapi::maximum_kernel(*max_exp_avg_sq, v_hat, queue);
                v_hat = *max_exp_avg_sq;
            }

            // Adam-Atan2 update: param -= lr * atan2(m_hat, sqrt(v_hat) + eps)
            Tensor sqrt_v = oneapi::sqrt_kernel(v_hat, queue);
            Tensor eps_t = oneapi::full_kernel({1}, eps, sqrt_v.dtype(), sqrt_v.device(), queue);
            Tensor denom = oneapi::add_kernel(sqrt_v, eps_t, queue);
            Tensor update = oneapi::atan2_kernel(m_hat, denom, queue);
            Tensor lr_t = oneapi::full_kernel({1}, lr, param.dtype(), param.device(), queue);
            Tensor step_update = oneapi::mul_kernel(update, lr_t, queue);
            oneapi::sub_inplace_kernel(param, step_update, queue);

            return std::vector<Tensor>{param};
        });

    // =========================================================================
    // Vision Operations
    // =========================================================================

    table.register_kernel(OpId::ROIAlignForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t output_height = attrs.get_int(AttrKey::OutputSizeH, 0);
            int64_t output_width = attrs.get_int(AttrKey::OutputSizeW, 0);
            float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0));
            int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
            bool aligned = attrs.get_bool(AttrKey::Aligned, false);
            return {oneapi::roi_align_kernel(inputs[0], inputs[1], output_height, output_width,
                                              spatial_scale, sampling_ratio, aligned, get_q(inputs))};
        });

    table.register_kernel(OpId::ROIAlignBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
            int64_t channels = inputs[0].shape()[1];
            int64_t feat_height = attrs.get_int(AttrKey::FeatHeight, 0);
            int64_t feat_width = attrs.get_int(AttrKey::FeatWidth, 0);
            float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0));
            int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
            bool aligned = attrs.get_bool(AttrKey::Aligned, false);
            return {oneapi::roi_align_backward_kernel(inputs[0], inputs[1], batch_size, channels,
                                                       feat_height, feat_width, spatial_scale,
                                                       sampling_ratio, aligned, get_q(inputs))};
        });

    table.register_kernel(OpId::Interpolate,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto size = attrs.get_int_list(AttrKey::OutputSize);
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            return {oneapi::interpolate_kernel(inputs[0], size, mode, align_corners, get_q(inputs))};
        });

    table.register_kernel(OpId::GatherRelativePositionBias,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
            int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
            return {oneapi::gather_relative_position_bias_kernel(
                inputs[0], inputs[1], num_positions, num_heads, get_q(inputs))};
        });

    // =========================================================================
    // Phase 10.1: Critical Layer Operations
    // =========================================================================

    table.register_single_output_kernel(OpId::Linear,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return oneapi::linear_kernel(inputs[0], inputs[1], bias, get_q(inputs));
        });

    table.register_kernel(OpId::LinearBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return oneapi::linear_backward_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs));
        });

    table.register_kernel(OpId::Dropout,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
            bool training = attrs.get_bool(AttrKey::Training, true);
            auto [output, mask] = oneapi::dropout_kernel(inputs[0], p, training, get_q(inputs));
            return {output, mask};
        });

    table.register_single_output_kernel(OpId::DropoutBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
            return oneapi::dropout_backward_kernel(inputs[0], inputs[1], p, get_q(inputs));
        });

    table.register_kernel(OpId::LayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [output, mean, rstd] = oneapi::layer_norm_kernel(
                inputs[0], normalized_shape, inputs[1], inputs[2], eps, get_q(inputs));
            return {output, mean, rstd};
        });

    table.register_kernel(OpId::InstanceNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            return oneapi::instance_norm_kernel(inputs[0], inputs[1], inputs[2], eps, get_q(inputs));
        });

    table.register_kernel(OpId::InstanceNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            // inputs: [grad_output, input, weight, mean, rstd]
            return oneapi::instance_norm_backward_kernel(
                inputs[0], inputs[1], inputs[3], inputs[4], inputs[2], get_q(inputs));
        });

    // =========================================================================
    // Phase 10.2: Activation Operations
    // =========================================================================

    table.register_single_output_kernel(OpId::Elu,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return oneapi::elu_kernel(inputs[0], alpha, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::EluBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return oneapi::elu_backward_kernel(inputs[0], inputs[1], alpha, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Selu,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::selu_kernel(inputs[0], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::SeluBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::selu_backward_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Mish,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::mish_kernel(inputs[0], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::MishBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::mish_backward_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Softplus,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
            float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
            return oneapi::softplus_kernel(inputs[0], beta, threshold, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::SoftplusBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
            float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
            return oneapi::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::TanhActivation,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::tanh_activation_kernel(inputs[0], get_q(inputs));
        });

    // =========================================================================
    // Phase 10.3: Tensor Reshape Operations
    // =========================================================================

    table.register_single_output_kernel(OpId::Flatten,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
            int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
            return oneapi::flatten_kernel(inputs[0], start_dim, end_dim, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Slice,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto starts = attrs.get_int_list(AttrKey::Starts);
            auto ends = attrs.get_int_list(AttrKey::Ends);
            auto steps = attrs.get_int_list(AttrKey::Steps);
            return oneapi::slice_kernel(inputs[0], starts, ends, steps, get_q(inputs));
        });

    table.register_kernel(OpId::Split,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t split_size = attrs.get_int(AttrKey::SplitSize, 1);
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return oneapi::split_kernel(inputs[0], split_size, dim, get_q(inputs));
        });

    table.register_kernel(OpId::Chunk,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t chunks = attrs.get_int(AttrKey::Chunks, 1);
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return oneapi::chunk_kernel(inputs[0], chunks, dim, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Tile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto reps = attrs.get_int_list(AttrKey::Reps);
            return oneapi::tile_kernel(inputs[0], reps, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Take,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::take_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Unfold,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            return oneapi::unfold_kernel(inputs[0], kernel_size, stride, padding, dilation, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Fold,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto output_size = attrs.get_int_list(AttrKey::OutputSize);
            int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            return oneapi::fold_kernel(inputs[0], output_size, kernel_size, stride, padding, dilation, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Roll,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t shift = attrs.get_int(AttrKey::Shift, 0);
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return oneapi::roll_kernel(inputs[0], shift, dim, get_q(inputs));
        });

    // =========================================================================
    // Triu / Tril / Diag / Trace / Flip
    // =========================================================================

    table.register_kernel(OpId::Triu,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
            return {oneapi::triu_kernel(inputs[0], diagonal, get_q(inputs))};
        });

    table.register_kernel(OpId::Tril,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
            return {oneapi::tril_kernel(inputs[0], diagonal, get_q(inputs))};
        });

    table.register_kernel(OpId::Diag,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
            return {oneapi::diag_kernel(inputs[0], diagonal, get_q(inputs))};
        });

    table.register_kernel(OpId::Trace,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::trace_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Flip,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::flip_kernel(inputs[0], dim, get_q(inputs))};
        });

    // =========================================================================
    // Full-Sequence RNN Operations (Phase 10.4)
    // =========================================================================

    table.register_kernel(OpId::LSTMForward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return oneapi::lstm_forward_kernel(inputs[0], inputs[1], inputs[2],
                                               inputs[3], inputs[4], inputs[5], inputs[6],
                                               get_q(inputs));
        });

    table.register_kernel(OpId::GRUForward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return oneapi::gru_forward_kernel(inputs[0], inputs[1], inputs[2],
                                              inputs[3], inputs[4], get_q(inputs));
        });

    table.register_kernel(OpId::LSTMMultiLayerForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);
            const Tensor& input = inputs[0];
            const Tensor& h0 = inputs[1];
            const Tensor& c0 = inputs[2];

            std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
            for (int64_t l = 0; l < num_layers; ++l) {
                size_t base_idx = 3 + l * 3;
                W_ih_list.push_back(inputs[base_idx]);
                W_hh_list.push_back(inputs[base_idx + 1]);
                bias_list.push_back(inputs[base_idx + 2]);
            }

            return oneapi::lstm_multilayer_forward_kernel(input, W_ih_list, W_hh_list,
                                                          bias_list, h0, c0, get_q(inputs));
        });

    table.register_kernel(OpId::GRUMultiLayerForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);
            const Tensor& input = inputs[0];
            const Tensor& h0 = inputs[1];

            std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
            for (int64_t l = 0; l < num_layers; ++l) {
                size_t base_idx = 2 + l * 3;
                W_ih_list.push_back(inputs[base_idx]);
                W_hh_list.push_back(inputs[base_idx + 1]);
                bias_list.push_back(inputs[base_idx + 2]);
            }

            return oneapi::gru_multilayer_forward_kernel(input, W_ih_list, W_hh_list,
                                                         bias_list, h0, get_q(inputs));
        });

    table.register_kernel(OpId::BiLSTMForward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return oneapi::bilstm_forward_kernel(
                inputs[0], inputs[1], inputs[2],
                inputs[3], inputs[4], inputs[5], inputs[6],
                inputs[7], inputs[8], inputs[9], inputs[10],
                get_q(inputs));
        });

    // =========================================================================
    // Sorting / Advanced Operations (Phase 10.5)
    // =========================================================================

    table.register_kernel(OpId::Sort,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool descending = attrs.get_bool(AttrKey::Descending, false);
            auto [values, indices] = oneapi::sort_kernel(inputs[0], dim, descending, get_q(inputs));
            return {values, indices};
        });

    table.register_kernel(OpId::TopK,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t k = attrs.get_int(AttrKey::K, 1);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool largest = attrs.get_bool(AttrKey::Largest, true);
            bool sorted = attrs.get_bool(AttrKey::Sorted, true);
            auto [values, indices] = oneapi::topk_kernel(inputs[0], k, dim, largest, sorted, get_q(inputs));
            return {values, indices};
        });

    table.register_kernel(OpId::Unique,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool sorted = attrs.get_bool(AttrKey::Sorted, true);
            bool return_inverse = attrs.get_bool(AttrKey::ReturnInverse, false);
            bool return_counts = attrs.get_bool(AttrKey::ReturnCounts, false);
            auto [unique_vals, inverse, counts] = oneapi::unique_kernel(
                inputs[0], sorted, return_inverse, return_counts, get_q(inputs));
            return {unique_vals, inverse, counts};
        });

    table.register_kernel(OpId::Cast,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            DType target_dtype = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
            return {oneapi::cast_kernel(inputs[0], target_dtype, get_q(inputs))};
        });

    table.register_inplace_kernel(OpId::StridedFill,
        [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
            double value = attrs.get_float(AttrKey::Value, 0.0);
            auto& q = oneapi_internal::get_queue(target.device().index);
            oneapi::strided_fill_kernel(target, value, q);
            return target;
        });

    table.register_kernel(OpId::ToMemoryFormat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int format_int = static_cast<int>(attrs.get_int(AttrKey::MemoryFormat, 0));
            return {oneapi::to_memory_format_kernel(inputs[0], format_int, get_q(inputs))};
        });

    table.register_kernel(OpId::ScatterAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::scatter_add_kernel(inputs[0], dim, inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_kernel(OpId::HasInfNan,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::has_inf_nan_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::EmbeddingBagForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "mean"));
            bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);
            return {oneapi::embedding_bag_forward_kernel(inputs[0], inputs[1], mode,
                                                         include_last_offset, get_q(inputs))};
        });

    table.register_kernel(OpId::EmbeddingBagBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [grad_output, embeddings, offsets]
            return {oneapi::embedding_bag_backward_kernel(
                inputs[0], inputs[1], inputs[2], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::DepthwiseConv2d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {oneapi::depthwise_conv2d_kernel(inputs[0], inputs[1], bias,
                                                     stride, padding, dilation, get_q(inputs))};
        });

    table.register_kernel(OpId::CumSum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::cumsum_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::CumProd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::cumprod_kernel(inputs[0], dim, get_q(inputs))};
        });

    // =========================================================================
    // Linear Algebra Operations (Phase 10.6 - oneMKL LAPACK)
    // =========================================================================
#ifdef TENZOR_HAS_ONEMKL
    table.register_kernel(OpId::LinalgDet,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::linalg_det_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::LinalgInv,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::linalg_inv_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::LinalgSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::linalg_solve_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::LinalgSVD,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool full_matrices = attrs.get_bool(AttrKey::FullMatrices, true);
            auto [U, S, Vt] = oneapi::linalg_svd_kernel(inputs[0], full_matrices, get_q(inputs));
            return {U, S, Vt};
        });

    table.register_kernel(OpId::LinalgQR,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [Q, R] = oneapi::linalg_qr_kernel(inputs[0], get_q(inputs));
            return {Q, R};
        });

    table.register_kernel(OpId::LinalgEigh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [W, V] = oneapi::linalg_eigh_kernel(inputs[0], get_q(inputs));
            return {W, V};
        });

    table.register_kernel(OpId::LinalgEig,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [WR, WI, V] = oneapi::linalg_eig_kernel(inputs[0], get_q(inputs));
            return {WR, WI, V};
        });

    table.register_kernel(OpId::LinalgCholesky,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return {oneapi::linalg_cholesky_kernel(inputs[0], upper, get_q(inputs))};
        });

    // =========================================================================
    // FFT Operations (Phase 10.7 - oneMKL DFT)
    // =========================================================================

    table.register_kernel(OpId::FFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::fft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IFFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::ifft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::RFFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::rfft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IRFFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, 2 * (inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim] - 1));
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::irfft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::FFT2,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::fft2_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IFFT2,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::ifft2_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::FFTN,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::fftn_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IFFTN,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::ifftn_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

#else // !TENZOR_HAS_ONEMKL
    // Linalg ops require oneMKL — no SYCL-native fallback
#define ONEMKL_STUB(OpName) \
    table.register_kernel(OpId::OpName, \
        [](std::span<const Tensor>, const OpAttributes&) -> std::vector<Tensor> { \
            throw std::runtime_error( \
                "Operation '" #OpName "' requires oneMKL. " \
                "Rebuild with oneMKL support enabled."); \
        })
    ONEMKL_STUB(LinalgDet);
    ONEMKL_STUB(LinalgInv);
    ONEMKL_STUB(LinalgSolve);
    ONEMKL_STUB(LinalgSVD);
    ONEMKL_STUB(LinalgQR);
    ONEMKL_STUB(LinalgEigh);
    ONEMKL_STUB(LinalgEig);
    ONEMKL_STUB(LinalgCholesky);
#undef ONEMKL_STUB

    // FFT ops: use SYCL-native Cooley-Tukey + Bluestein fallback (implemented in fft.cpp)
    table.register_kernel(OpId::FFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::fft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IFFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::ifft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::RFFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim]);
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::rfft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IRFFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            int64_t n = attrs.get_int(AttrKey::N, 2 * (inputs[0].shape()[dim < 0 ? dim + inputs[0].ndim() : dim] - 1));
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::irfft_kernel(inputs[0], dim, n, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::FFT2,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::fft2_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IFFT2,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::ifft2_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::FFTN,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::fftn_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });

    table.register_kernel(OpId::IFFTN,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            auto norm = std::string(attrs.get_string(AttrKey::Norm, "backward"));
            return {oneapi::ifftn_kernel(inputs[0], dims, signal_lengths, norm, get_q(inputs))};
        });
#endif // TENZOR_HAS_ONEMKL

    // =========================================================================
    // Put Operation (Phase 3.1)
    // =========================================================================

    table.register_kernel(OpId::Put,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool accumulate = attrs.get_bool(AttrKey::Accumulate, false);
            return {oneapi::put_kernel(inputs[0], inputs[1], inputs[2], accumulate, get_q(inputs))};
        });

    // =========================================================================
    // Box IoU Operation (Phase 3.1)
    // =========================================================================

    table.register_kernel(OpId::BoxIoU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int iou_type = static_cast<int>(attrs.get_int(AttrKey::IouType, 0));
            return {oneapi::box_iou_kernel(inputs[0], inputs[1], iou_type, get_q(inputs))};
        });

    // =========================================================================
    // NMS Operation
    // =========================================================================
    table.register_kernel(OpId::NMS,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [boxes (N,4), scores (N)]
            // attrs: IouThreshold
            float iou_threshold = static_cast<float>(attrs.get_float(AttrKey::IouThreshold, 0.5));
            return {oneapi::nms_kernel(inputs[0], inputs[1], iou_threshold, get_q(inputs))};
        });

    // =========================================================================
    // Quantized Operations (Phase 3.1)
    // =========================================================================

    table.register_kernel(OpId::QuantizedLinear,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
            int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
            float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
            int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));
            float output_scale = static_cast<float>(attrs.get_float(AttrKey::OutputScale, 1.0));
            int32_t output_zp = static_cast<int32_t>(attrs.get_int(AttrKey::OutputZeroPoint, 0));
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {oneapi::quantized_linear_kernel(inputs[0], inputs[1], bias,
                                                     input_scale, input_zp,
                                                     weight_scale, weight_zp,
                                                     output_scale, output_zp, get_q(inputs))};
        });

    table.register_kernel(OpId::QuantizedConv2d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t stride = attrs.get_int(AttrKey::Stride, 1);
            int64_t padding = attrs.get_int(AttrKey::Padding, 0);
            int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
            int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
            float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
            int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {oneapi::quantized_conv2d_kernel(inputs[0], inputs[1], bias,
                                                     stride, padding, dilation, groups,
                                                     input_scale, input_zp,
                                                     weight_scale, weight_zp, get_q(inputs))};
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

            Tensor u = rand(shape_vec, logits.dtype(), logits.device());
            Tensor eps_tensor = full(shape_vec, 1e-20, logits.dtype(), logits.device());
            u = add(u, eps_tensor);

            Tensor gumbels = neg(log(neg(log(u))));
            Tensor scaled = div(add(logits, gumbels),
                                full(shape_vec, tau, logits.dtype(), logits.device()));

            std::array<Tensor, 1> sm_inputs = {scaled};
            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, dim);
            Tensor y_soft = dispatch<OpId::Softmax>(sm_inputs, sm_attrs)[0];

            if (!hard) {
                return y_soft;
            }

            int64_t actual_dim = dim < 0 ? dim + logits.ndim() : dim;
            Tensor indices = argmax(y_soft, std::make_optional(actual_dim), /*keepdim=*/true);

            Tensor y_hard = zeros(shape_vec, logits.dtype(), logits.device());
            std::array<Tensor, 3> scatter_inputs = {y_hard, indices,
                full(std::vector<int64_t>(indices.shape().begin(), indices.shape().end()),
                     1.0, logits.dtype(), logits.device())};
            NewOpAttributes scatter_attrs;
            scatter_attrs.set(AttrKey::Dim, actual_dim);
            y_hard = dispatch<OpId::Scatter>(scatter_inputs, scatter_attrs)[0];

            return add(sub(y_hard, y_soft.detach()), y_soft);
        });

    // =========================================================================
    // Sparse Tensor Operations (OpIds 460-464)
    //
    // Wrapper lambdas that reconstruct SparseTensor from CSR components passed
    // as plain Tensors, then delegate to the existing sparse:: functions which
    // internally dispatch to oneMKL sparse when inputs are on OneAPI/SYCL.
    // =========================================================================

#ifdef TENZOR_HAS_ONEMKL
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
#else // !TENZOR_HAS_ONEMKL
    // SYCL-native CSR SpMM/SpMV fallbacks (no oneMKL dependency)
    table.register_single_output_kernel(OpId::SparseSpMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::spmm(sp, inputs[3]);
        });

    table.register_single_output_kernel(OpId::SparseSpMV,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::spmv(sp, inputs[3]);
        });
#endif // TENZOR_HAS_ONEMKL

    // SparseToDense: CSR components -> dense tensor
    table.register_single_output_kernel(OpId::SparseToDense,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sp.to_dense();
        });

    // DenseToSparse: dense tensor -> CSR components [crow_indices, col_indices, values]
    table.register_kernel(OpId::DenseToSparse,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto sp = SparseTensor::from_dense(inputs[0], SparseLayout::CSR);
            return {sp.crow_indices(), sp.col_indices(), sp.values()};
        });

    // SparseAdd: sparse(M,K) + dense(M,K) -> dense(M,K)
    table.register_single_output_kernel(OpId::SparseAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sparse::add(sp, inputs[3]);
        });

} // register_oneapi_kernels

} // namespace tenzor


// ============================================================================
// Export function for dynamic loading via dlsym
// ============================================================================
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::register_oneapi_kernels(*table);
        }
    }
}
