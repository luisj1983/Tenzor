/**
 * @file oneapi_kernel_registry.cpp
 * @brief OneAPI kernel registration for O(1) dispatch
 *
 * Registers all OneAPI/SYCL kernel implementations with the dispatch table.
 * Each kernel is a direct function call - no intermediate string dispatch.
 * The SYCL queue is obtained via oneapi_internal::get_queue() which is
 * wired up by the backend constructor.
 */

#include <limits>
#include "oneapi_internal.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "kernels/fp16_saturate.hpp"
#include <climits>
#include <cmath>
#include <cstdint>
#include <cstring>
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

    // Special math (native OneAPI — replaces previous CPU-roundtrip fallbacks)
    auto gamma_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto lgamma_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto digamma_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto polygamma_kernel(int64_t n, const Tensor& input, sycl::queue& queue) -> Tensor;
    auto beta_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto betainc_kernel(const Tensor& a, const Tensor& b, const Tensor& x, sycl::queue& queue) -> Tensor;
    auto bessel_j0_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bessel_j1_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bessel_y0_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bessel_y1_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bessel_i0_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bessel_i1_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto erfinv_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto sinc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto zeta_kernel(const Tensor& s, const Tensor& q, sycl::queue& queue) -> Tensor;
    auto i0e_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto i1e_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto entr_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto spherical_bessel_j0_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto logaddexp_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto logaddexp2_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto xlogy_kernel(const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor;
    auto cosine_similarity_kernel(const Tensor& a, const Tensor& b, int64_t dim, double eps, sycl::queue& queue) -> Tensor;
    auto renorm_kernel(const Tensor& input, double p, int64_t dim, double maxnorm, sycl::queue& queue) -> Tensor;

    // Ndtr / LogNdtr / Multigammaln
    auto ndtr_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto log_ndtr_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto multigammaln_kernel(const Tensor& input, int64_t d, sycl::queue& queue) -> Tensor;

    // Grid sample / affine grid (native OneAPI — replaces previous CPU fallbacks)
    auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                            const std::string& mode, const std::string& padding_mode,
                            bool align_corners, sycl::queue& queue) -> Tensor;
    auto affine_grid_kernel(const Tensor& theta, const std::vector<int64_t>& size,
                            bool align_corners, sycl::queue& queue) -> Tensor;

    // Sampling / statistics (native OneAPI — replaces previous CPU fallbacks)
    auto bernoulli_kernel(const Tensor& probs, sycl::queue& queue) -> Tensor;
    auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                            bool replacement, sycl::queue& queue) -> Tensor;
    auto bucketize_kernel(const Tensor& input, const Tensor& boundaries,
                          bool right, sycl::queue& queue) -> Tensor;
    auto histogram_kernel(const Tensor& input, int64_t bins,
                          double min_val, double max_val,
                          sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto histogramdd_kernel(const Tensor& input, std::vector<int64_t> bins,
                            std::vector<std::pair<double,double>> ranges, bool density,
                            sycl::queue& queue) -> std::pair<Tensor, std::vector<Tensor>>;
    auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                      sycl::queue& queue) -> Tensor;
    auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr, sycl::queue& queue) -> Tensor;
    auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr, sycl::queue& queue) -> Tensor;
    auto gradient_kernel(const Tensor& input, int64_t dim, double spacing, sycl::queue& queue) -> Tensor;
    auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p, sycl::queue& queue) -> Tensor;
    auto pdist_kernel(const Tensor& input, double p, sycl::queue& queue) -> Tensor;
    auto poisson_sample_kernel(const Tensor& rates, sycl::queue& queue) -> Tensor;
    auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, sycl::queue& queue) -> Tensor;
    auto exponential_sample_kernel(const Tensor& rate, sycl::queue& queue) -> Tensor;

    // Standalone SYCL sparse ops (no oneMKL dependency)
    auto spgemm_standalone_sycl(std::span<const Tensor> inputs, const OpAttributes& attrs,
                                sycl::queue& queue) -> std::vector<Tensor>;
    auto sparse_trsv_standalone_sycl(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                     const Tensor& b, int64_t N, bool upper,
                                     sycl::queue& queue) -> Tensor;
    auto sparse_trsm_standalone_sycl(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                     const Tensor& B, int64_t N, bool upper,
                                     sycl::queue& queue) -> Tensor;

    // oneMKL-backed sparse ops
    auto spgemm_kernel(const SparseTensor& A, const SparseTensor& B,
                       sycl::queue& queue) -> SparseTensor;
    auto sparse_trsv_kernel(const SparseTensor& L, const Tensor& b, bool upper,
                            sycl::queue& queue) -> Tensor;
    auto sparse_trsm_kernel(const SparseTensor& L, const Tensor& B, bool upper,
                            sycl::queue& queue) -> Tensor;

    // NestedAttention (native SYCL — replaces previous CPU-offset fallback)
    auto nested_attention_kernel(const Tensor& Q, const Tensor& K, const Tensor& V,
                                  const Tensor& q_offsets, const Tensor& kv_offsets,
                                  float scale, bool causal, sycl::queue& queue) -> Tensor;

    // NestedAttentionBackward (native SYCL)
    auto nested_attention_backward_kernel(const Tensor& grad_out, const Tensor& Q, const Tensor& K, const Tensor& V,
                                           const Tensor& attn_out, const Tensor& q_offsets, const Tensor& kv_offsets,
                                           float scale, bool causal, sycl::queue& queue) -> std::vector<Tensor>;

    // STFT / ISTFT (native OneAPI — replaces previous CPU fallbacks)
    auto stft_kernel(const Tensor& input, int64_t n_fft,
                     int64_t hop_length, int64_t win_length,
                     const Tensor& window, bool center,
                     bool normalized, bool onesided,
                     sycl::queue& queue) -> Tensor;
    auto istft_kernel(const Tensor& input, int64_t n_fft,
                      int64_t hop_length, int64_t win_length,
                      const Tensor& window, bool center,
                      bool normalized, bool onesided,
                      int64_t length, sycl::queue& queue) -> Tensor;

    // ---- Advanced indexing (kernels/advanced_index.cpp) ----
    auto advanced_index_oneapi_kernel(const Tensor& src,
                                      const std::vector<Tensor>& indices,
                                      int64_t num_indices,
                                      sycl::queue& queue) -> Tensor;
    auto advanced_index_put_oneapi_kernel(const Tensor& src,
                                          const std::vector<Tensor>& indices,
                                          const Tensor& values,
                                          int64_t num_indices,
                                          sycl::queue& queue) -> Tensor;

    // Bool predicates
    auto isnan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isinf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isfinite_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto signbit_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isposinf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isneginf_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto isreal_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // New unary math
    auto deg2rad_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto rad2deg_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto logit_kernel(const Tensor& input, float eps, sycl::queue& queue) -> Tensor;

    // New binary math
    auto float_power_kernel(const Tensor& base, const Tensor& exp, sycl::queue& queue) -> Tensor;
    auto xlog1py_kernel(const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor;
    auto ldexp_kernel(const Tensor& x, const Tensor& n, sycl::queue& queue) -> Tensor;
    auto frexp_kernel(const Tensor& input, sycl::queue& queue) -> std::vector<Tensor>;

    // Tensor manipulation
    auto diag_embed_kernel(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2, sycl::queue& queue) -> Tensor;
    auto diagflat_kernel(const Tensor& input, int64_t offset, sycl::queue& queue) -> Tensor;

    // Binary math
    auto fmod_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;

    // New unary math
    auto rsqrt_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto square_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto asinh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto acosh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto atanh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

    // New binary math
    auto hypot_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto copysign_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto nextafter_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto gcd_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto lcm_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto igamma_kernel(const Tensor& a, const Tensor& x, sycl::queue& queue) -> Tensor;
    auto igammac_kernel(const Tensor& a, const Tensor& x, sycl::queue& queue) -> Tensor;

    // Ternary
    auto lerp_kernel(const Tensor& start, const Tensor& end, const Tensor& weight, sycl::queue& queue) -> Tensor;
    auto addcmul_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, sycl::queue& queue) -> Tensor;
    auto addcdiv_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, sycl::queue& queue) -> Tensor;

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
    auto complex_tensor_kernel(const Tensor& real_t, const Tensor& imag_t, sycl::queue& queue) -> Tensor;
    auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim, sycl::queue& queue) -> Tensor;

    // Phase 4 math ops (native SYCL kernels in kernels/math.cpp)
    auto frac_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto heaviside_kernel(const Tensor& input, const Tensor& values, sycl::queue& queue) -> Tensor;
    auto nan_to_num_kernel(const Tensor& input, double nan_v, double posinf_v, double neginf_v, sycl::queue& queue) -> Tensor;
    auto log_sigmoid_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bitwise_and_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto bitwise_or_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto bitwise_xor_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto bitwise_not_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift, sycl::queue& queue) -> Tensor;
    auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift, sycl::queue& queue) -> Tensor;

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
    auto rrelu_kernel(const Tensor& input, float lower, float upper, bool training, sycl::queue& queue) -> Tensor;
    auto rrelu_backward_kernel(const Tensor& grad_output, const Tensor& input, float lower, float upper, sycl::queue& queue) -> Tensor;
    auto log_sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor;

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
    auto count_nonzero_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto nansum_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto nanmean_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto nanvar_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, sycl::queue& queue) -> Tensor;
    auto nanstd_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, sycl::queue& queue) -> Tensor;
    auto aminmax_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> std::vector<Tensor>;

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
    auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim, sycl::queue& queue) -> Tensor;
    auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats, int64_t dim, sycl::queue& queue) -> Tensor;
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
    auto index_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source, sycl::queue& queue) -> Tensor;
    auto index_copy_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source, sycl::queue& queue) -> Tensor;
    auto index_fill_kernel(const Tensor& input, int64_t dim, const Tensor& index, float value, sycl::queue& queue) -> Tensor;
    auto scatter_reduce_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& source, const std::string& reduce, bool include_self, sycl::queue& queue) -> Tensor;
    auto take_along_dim_kernel(const Tensor& input, const Tensor& indices, int64_t dim, sycl::queue& queue) -> Tensor;
    auto masked_scatter_kernel(const Tensor& input, const Tensor& mask, const Tensor& source, sycl::queue& queue) -> Tensor;
    auto tril_indices_kernel(int64_t row, int64_t col, int64_t offset, sycl::queue& queue) -> Tensor;
    auto triu_indices_kernel(int64_t row, int64_t col, int64_t offset, sycl::queue& queue) -> Tensor;

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
    auto adaptive_max_pool2d_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> std::vector<Tensor>;
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

    // ---- Fractional Max Pool + Max Unpool operations (kernels/pooling.cpp) ----
    auto fractional_maxpool2d_forward_kernel(const Tensor& input, int64_t out_h, int64_t out_w,
                                             const Tensor* random_samples, sycl::queue& queue)
        -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                              const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto fractional_maxpool3d_forward_kernel(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w,
                                             const Tensor* random_samples, sycl::queue& queue)
        -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                              const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices,
                                     int64_t out_h, int64_t out_w, sycl::queue& queue) -> Tensor;
    auto max_unpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                      const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;
    auto max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices,
                                     int64_t out_d, int64_t out_h, int64_t out_w, sycl::queue& queue) -> Tensor;
    auto max_unpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                      const std::vector<int64_t>& input_shape, sycl::queue& queue) -> Tensor;

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

    // ---- HasInfNan, CumSum, CumProd, Logcumsumexp, Bincount (kernels/math.cpp) ----
    auto has_inf_nan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cumsum_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto logcumsumexp_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto bincount_kernel(const Tensor& input, const Tensor* weights,
                         int64_t minlength, sycl::queue& queue) -> Tensor;

    // ---- New reduction operations (kernels/math.cpp) ----
    auto cummax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto cummin_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto fmax_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto fmin_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor;
    auto isin_kernel(const Tensor& elements, const Tensor& test_elements, sycl::queue& queue) -> Tensor;
    auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim, bool keepdim, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto quantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto nanquantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto nanmedian_kernel(const Tensor& input, int64_t dim, bool keepdim, sycl::queue& queue) -> Tensor;
    auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val, sycl::queue& queue) -> Tensor;
    auto unique_consecutive_kernel(const Tensor& input, bool return_inverse, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets, const std::string& reduce, int64_t axis, sycl::queue& queue) -> Tensor;

    // ---- DepthwiseConv2d (kernels/conv2d.cpp) ----
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                  int64_t stride, int64_t padding, int64_t dilation,
                                  sycl::queue& queue) -> Tensor;

    // ---- DeformableConv2d / DCNv2 (kernels/conv2d.cpp) ----
    auto deformable_conv2d_forward_kernel(
        const Tensor& input, const Tensor& offset, const Tensor& weight,
        const Tensor& bias, const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        sycl::queue& queue) -> Tensor;

    auto deformable_conv2d_backward_input_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& offset,
        const Tensor& weight, const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        sycl::queue& queue) -> std::vector<Tensor>;

    auto deformable_conv2d_backward_weight_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& offset,
        const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        const std::vector<int64_t>& weight_shape,
        sycl::queue& queue) -> Tensor;

    // ---- Linalg operations (kernels/linalg.cpp) ----
#ifdef TENZOR_HAS_ONEMKL
    auto linalg_det_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto linalg_inv_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto linalg_solve_kernel(const Tensor& A, const Tensor& B, sycl::queue& queue) -> Tensor;
    auto linalg_lu_kernel(const Tensor& A, sycl::queue& queue)
        -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                                const Tensor& B, sycl::queue& queue) -> Tensor;
    auto linalg_svd_kernel(const Tensor& input, bool full_matrices,
                            sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_qr_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto linalg_eigh_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor>;
    auto linalg_eig_kernel(const Tensor& input, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_cholesky_kernel(const Tensor& input, bool upper, sycl::queue& queue) -> Tensor;
    auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                         bool upper, bool unitriangular,
                                         sycl::queue& queue) -> Tensor;
    auto linalg_geqrf_kernel(const Tensor& input, sycl::queue& queue)
        -> std::pair<Tensor, Tensor>;
    auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                              const Tensor& C, bool left, bool transpose_q,
                              sycl::queue& queue) -> Tensor;
    auto linalg_ldl_factor_kernel(const Tensor& A, sycl::queue& queue)
        -> std::tuple<Tensor, Tensor>;
    auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                                  const Tensor& B, sycl::queue& queue) -> Tensor;
    auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                    sycl::queue& queue) -> Tensor;
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
// FP8 emulation helpers
// ============================================================================
static bool is_fp8(DType dt) {
    return dt == DType::FP8_E4M3 || dt == DType::FP8_E5M2;
}

static DType fp8_result_dtype(DType a, DType b) {
    // Type promotion: FP8 + FP8 → wider FP8, FP8 + Float → Float
    if (!is_fp8(a)) return a;
    if (!is_fp8(b)) return b;
    // E5M2 has wider range than E4M3
    if (a == DType::FP8_E5M2 || b == DType::FP8_E5M2) return DType::FP8_E5M2;
    return DType::FP8_E4M3;
}

// Binary FP8 emulation: widen to Float32, compute, narrow back
template<typename BinaryOp>
static Tensor fp8_binary_emulate(const Tensor& a, const Tensor& b, BinaryOp op) {
    DType out_dtype = fp8_result_dtype(a.dtype(), b.dtype());
    Tensor a_f32 = is_fp8(a.dtype()) ? a.to(DType::Float32) : a;
    Tensor b_f32 = is_fp8(b.dtype()) ? b.to(DType::Float32) : b;
    Tensor result_f32 = op(a_f32, b_f32);
    return is_fp8(out_dtype) ? result_f32.to(out_dtype) : result_f32;
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
            if (is_fp8(inputs[0].dtype()) || is_fp8(inputs[1].dtype())) {
                return {fp8_binary_emulate(inputs[0], inputs[1],
                    [&](const Tensor& a, const Tensor& b) { return oneapi::add_kernel(a, b, get_q(inputs)); })};
            }
            return {oneapi::add_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Sub,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            if (is_fp8(inputs[0].dtype()) || is_fp8(inputs[1].dtype())) {
                return {fp8_binary_emulate(inputs[0], inputs[1],
                    [&](const Tensor& a, const Tensor& b) { return oneapi::sub_kernel(a, b, get_q(inputs)); })};
            }
            return {oneapi::sub_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Mul,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            if (is_fp8(inputs[0].dtype()) || is_fp8(inputs[1].dtype())) {
                return {fp8_binary_emulate(inputs[0], inputs[1],
                    [&](const Tensor& a, const Tensor& b) { return oneapi::mul_kernel(a, b, get_q(inputs)); })};
            }
            return {oneapi::mul_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Div,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            if (is_fp8(inputs[0].dtype()) || is_fp8(inputs[1].dtype())) {
                return {fp8_binary_emulate(inputs[0], inputs[1],
                    [&](const Tensor& a, const Tensor& b) { return oneapi::div_kernel(a, b, get_q(inputs)); })};
            }
            return {oneapi::div_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::MatMul,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            if (is_fp8(inputs[0].dtype()) || is_fp8(inputs[1].dtype())) {
                return {fp8_binary_emulate(inputs[0], inputs[1],
                    [&](const Tensor& a, const Tensor& b) {
                        auto r = oneapi::matmul_kernel(a, b, get_q(inputs));
                        oneapi::fp16_saturate_if_needed(r, get_q(inputs));
                        return r;
                    })};
            }
            auto result = oneapi::matmul_kernel(inputs[0], inputs[1], get_q(inputs));
            oneapi::fp16_saturate_if_needed(result, get_q(inputs));
            return {result};
        });

    table.register_kernel(OpId::Bmm,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto result = oneapi::bmm_kernel(inputs[0], inputs[1], get_q(inputs));
            oneapi::fp16_saturate_if_needed(result, get_q(inputs));
            return {result};
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
    // Special Math Functions — native OneAPI/SYCL kernels
    // =========================================================================
#define ONEAPI_REGISTER_UNARY_SPECIAL(OP_ID, FN) \
    table.register_kernel(OpId::OP_ID, [](std::span<const Tensor> inputs, const OpAttributes&) { \
        return std::vector<Tensor>{oneapi::FN(inputs[0], get_q(inputs))}; \
    })
#define ONEAPI_REGISTER_BINARY_SPECIAL(OP_ID, FN) \
    table.register_kernel(OpId::OP_ID, [](std::span<const Tensor> inputs, const OpAttributes&) { \
        return std::vector<Tensor>{oneapi::FN(inputs[0], inputs[1], get_q(inputs))}; \
    })

    ONEAPI_REGISTER_UNARY_SPECIAL(Gamma,     gamma_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(Lgamma,    lgamma_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(Digamma,   digamma_kernel);
    ONEAPI_REGISTER_BINARY_SPECIAL(Beta,     beta_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(BesselJ0,  bessel_j0_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(BesselJ1,  bessel_j1_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(BesselY0,  bessel_y0_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(BesselY1,  bessel_y1_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(BesselI0,  bessel_i0_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(BesselI1,  bessel_i1_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(ErfInv,    erfinv_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(Sinc,      sinc_kernel);
    ONEAPI_REGISTER_BINARY_SPECIAL(Zeta,     zeta_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(I0e,       i0e_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(I1e,       i1e_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(Entr,      entr_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(SphericalBesselJ0, spherical_bessel_j0_kernel);
    ONEAPI_REGISTER_BINARY_SPECIAL(LogAddExp,  logaddexp_kernel);
    ONEAPI_REGISTER_BINARY_SPECIAL(LogAddExp2, logaddexp2_kernel);
    ONEAPI_REGISTER_BINARY_SPECIAL(XLogY,      xlogy_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(Ndtr,       ndtr_kernel);
    ONEAPI_REGISTER_UNARY_SPECIAL(LogNdtr,    log_ndtr_kernel);

    // Multigammaln (needs extra dim parameter)
    table.register_kernel(OpId::Multigammaln,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t d = attrs.get_int(AttrKey::Dim, 1);
            return {oneapi::multigammaln_kernel(inputs[0], d, get_q(inputs))};
        });

    table.register_kernel(OpId::Polygamma, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        return std::vector<Tensor>{oneapi::polygamma_kernel(n, inputs[0], get_q(inputs))};
    });
    table.register_kernel(OpId::BetaInc, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{oneapi::betainc_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs))};
    });

    // CosineSimilarity: reduction op with dim + eps parameters
    table.register_single_output_kernel(OpId::CosineSimilarity, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 1);
        double eps = attrs.get_float(AttrKey::Eps, 1e-8);
        return oneapi::cosine_similarity_kernel(inputs[0], inputs[1], dim, eps, get_q(inputs));
    });

    // Renorm: reduction op with p, dim, maxnorm parameters
    table.register_single_output_kernel(OpId::Renorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::P, 2.0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        double maxnorm = attrs.get_float(AttrKey::MaxNorm, 1.0);
        return oneapi::renorm_kernel(inputs[0], p, dim, maxnorm, get_q(inputs));
    });

#undef ONEAPI_REGISTER_UNARY_SPECIAL
#undef ONEAPI_REGISTER_BINARY_SPECIAL

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
            const auto& start = inputs[0];
            const auto& end = inputs[1];
            Tensor weight = inputs[2];
            // Broadcast weight to match start shape if needed
            if (weight.numel() == 1 && start.numel() > 1) {
                auto shape_span = start.shape();
                std::string shape_str;
                for (size_t i = 0; i < shape_span.size(); ++i) {
                    if (i > 0) shape_str += ',';
                    shape_str += std::to_string(shape_span[i]);
                }
                OpAttributes expand_attrs;
                expand_attrs.set(AttrKey::Shape, shape_str);
                weight = oneapi::expand_kernel(weight, expand_attrs, get_q(inputs));
            }
            return {oneapi::lerp_kernel(start, end, weight, get_q(inputs))};
        });

    // =========================================================================
    // New Unary Math Operations
    // =========================================================================

    table.register_kernel(OpId::Rsqrt,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::rsqrt_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Square,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::square_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Asinh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::asinh_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Acosh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::acosh_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Atanh,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::atanh_kernel(inputs[0], get_q(inputs))};
        });

    // =========================================================================
    // New Binary Math Operations
    // =========================================================================

    table.register_kernel(OpId::Hypot,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::hypot_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Copysign,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::copysign_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Nextafter,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::nextafter_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Gcd,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::gcd_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Lcm,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::lcm_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Igamma,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::igamma_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Igammac,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::igammac_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // New Ternary Operations: Addcmul / Addcdiv
    // =========================================================================

    table.register_kernel(OpId::Addcmul,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float value = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return {oneapi::addcmul_kernel(inputs[0], inputs[1], inputs[2], value, get_q(inputs))};
        });

    table.register_kernel(OpId::Addcdiv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float value = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
            return {oneapi::addcdiv_kernel(inputs[0], inputs[1], inputs[2], value, get_q(inputs))};
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

    table.register_kernel(OpId::ComplexTensor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::complex_tensor_kernel(inputs[0], inputs[1], get_q(inputs))};
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
            float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<float>::infinity()));
            float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity()));
            return {oneapi::clamp_kernel(inputs[0], min_val, max_val, get_q(inputs))};
        });

    table.register_kernel(OpId::ClampMin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, 0.0));
            return {oneapi::clamp_min_kernel(inputs[0], min_val, get_q(inputs))};
        });

    table.register_kernel(OpId::ClampMax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity()));
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
            auto result = oneapi::softmax_kernel(inputs[0], dim, get_q(inputs));
            oneapi::fp16_saturate_if_needed(result, get_q(inputs));
            return {result};
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
            auto result = oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                                           stride, padding, dilation, groups, get_q(inputs));
            oneapi::fp16_saturate_if_needed(result, get_q(inputs));
            return {result};
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

    // Conv1d: wraps Conv2d by unsqueezing height dimension [N,C,L] -> [N,C,1,L]
    table.register_kernel(OpId::Conv1dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_4d = inputs[0].unsqueeze(2);
            auto weight_4d = inputs[1].unsqueeze(2);
            std::vector<Tensor> conv2d_inputs = inputs.size() > 2
                ? std::vector<Tensor>{input_4d, weight_4d, inputs[2]}
                : std::vector<Tensor>{input_4d, weight_4d};
            auto result = tenzor::dispatch(OpId::Conv2dForward, conv2d_inputs, attrs);
            return {result[0].squeeze(2)};
        });

    table.register_kernel(OpId::Conv1dBackwardInput,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto grad_4d = inputs[0].unsqueeze(2);
            auto input_4d = inputs[1].unsqueeze(2);
            auto weight_4d = inputs[2].unsqueeze(2);
            std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
            auto result = tenzor::dispatch(OpId::Conv2dBackwardInput, conv2d_inputs, attrs);
            return {result[0].squeeze(2)};
        });

    table.register_kernel(OpId::Conv1dBackwardWeight,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto grad_4d = inputs[0].unsqueeze(2);
            auto input_4d = inputs[1].unsqueeze(2);
            auto weight_4d = inputs[2].unsqueeze(2);
            std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
            auto result = tenzor::dispatch(OpId::Conv2dBackwardWeight, conv2d_inputs, attrs);
            return {result[0].squeeze(2)};
        });

    table.register_kernel(OpId::Conv1dBackwardBias,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto grad_4d = inputs[0].unsqueeze(2);
            std::vector<Tensor> conv2d_inputs = {grad_4d};
            auto result = tenzor::dispatch(OpId::Conv2dBackwardBias, conv2d_inputs, {});
            return {result[0]};
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
            return {oneapi::conv_transpose3d_backward_input(inputs[0], inputs[2], input_shape,
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
            return oneapi::adaptive_max_pool2d_kernel(inputs[0], attrs, get_q(inputs));
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
            int64_t H_in = 0, W_in = 0;
            if (attrs.has(AttrKey::InputShape)) {
                auto input_shape = attrs.get_int_list(AttrKey::InputShape);
                H_in = input_shape.size() >= 3 ? input_shape[2] : 0;
                W_in = input_shape.size() >= 4 ? input_shape[3] : 0;
            } else {
                H_in = attrs.get_int(AttrKey::InputH, 0);
                W_in = attrs.get_int(AttrKey::InputW, 0);
            }
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
            // Expand single value to 3-element vector for isotropic pooling
            if (ks.size() == 1) ks = {ks[0], ks[0], ks[0]};
            if (st.size() == 1) st = {st[0], st[0], st[0]};
            if (pd.size() == 1) pd = {pd[0], pd[0], pd[0]};
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
            if (ks.size() == 1) ks = {ks[0], ks[0], ks[0]};
            if (st.size() == 1) st = {st[0], st[0], st[0]};
            if (pd.size() == 1) pd = {pd[0], pd[0], pd[0]};
            return {oneapi::avgpool3d_forward(inputs[0], ks, st, pd, get_q(inputs))};
        });

    table.register_kernel(OpId::AvgPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto ks = attrs.get_int_list(AttrKey::KernelSize);
            auto st = attrs.get_int_list(AttrKey::Stride);
            auto pd = attrs.get_int_list(AttrKey::Padding);
            if (ks.size() == 1) ks = {ks[0], ks[0], ks[0]};
            if (st.size() == 1) st = {st[0], st[0], st[0]};
            if (pd.size() == 1) pd = {pd[0], pd[0], pd[0]};
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::avgpool3d_backward(inputs[0], ks, st, pd, input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveMaxPool3d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::vector<int64_t> output_size;
            if (attrs.has(AttrKey::OutputSize)) {
                output_size = attrs.get_int_list(AttrKey::OutputSize);
            } else {
                output_size = {
                    attrs.get_int(AttrKey::OutputSizeD, 1),
                    attrs.get_int(AttrKey::OutputSizeH, 1),
                    attrs.get_int(AttrKey::OutputSizeW, 1)
                };
            }
            return oneapi::adaptive_maxpool3d_forward(inputs[0], output_size, get_q(inputs));
        });

    table.register_kernel(OpId::AdaptiveMaxPool3dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = attrs.get_int_list(AttrKey::InputShape);
            return {oneapi::adaptive_maxpool3d_backward(inputs[0], inputs[1], input_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool3d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::vector<int64_t> output_size;
            if (attrs.has(AttrKey::OutputSize)) {
                output_size = attrs.get_int_list(AttrKey::OutputSize);
            } else {
                output_size = {
                    attrs.get_int(AttrKey::OutputSizeD, 1),
                    attrs.get_int(AttrKey::OutputSizeH, 1),
                    attrs.get_int(AttrKey::OutputSizeW, 1)
                };
            }
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
            auto result = oneapi::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_q(inputs));
            oneapi::fp16_saturate_if_needed(result, get_q(inputs));
            return {result};
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
            // Dispatch provides: [grad_output, input, weight/gamma, mean, invstd]
            // Kernel expects:    (grad_output, input, mean, variance, gamma, epsilon)
            // The backward saves invstd = 1/sqrt(var+eps), convert back to variance
            // var = (1/invstd)^2 - eps
            const auto& grad_output = inputs[0];
            const auto& input = inputs[1];
            const auto& gamma = inputs[2];
            const auto& mean = inputs[3];
            const auto& invstd = inputs[4];

            // Convert invstd back to variance: var = 1/invstd^2 - eps
            auto invstd_sq = tenzor::mul(invstd, invstd);
            auto one_over_invstd_sq = tenzor::div(
                tenzor::ones(std::vector<int64_t>(invstd.shape().begin(), invstd.shape().end()),
                             invstd.dtype(), invstd.device()),
                invstd_sq);
            auto eps_tensor = tenzor::full(
                std::vector<int64_t>(invstd.shape().begin(), invstd.shape().end()),
                static_cast<double>(epsilon), invstd.dtype(), invstd.device());
            auto variance = tenzor::sub(one_over_invstd_sq, eps_tensor);

            auto [grad_input, grad_gamma, grad_beta] = oneapi::batchnorm2d_backward(
                grad_output, input, mean, variance, gamma, epsilon, get_q(inputs));
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

    table.register_kernel(OpId::RMSNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
            auto [output, rrms] = oneapi::fused_rms_norm_kernel(inputs[0], inputs[1], epsilon, get_q(inputs));
            return {output, rrms};
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
            // conv2d_forward drives oneDNN on an interop stream created from
            // `queue`. dnnl_stream.wait() only drains the dnnl stream — the
            // outer sycl::queue can still have pending task graph state from
            // prior SYCL dispatches in this session (e.g., device migration).
            // Drain the outer queue before submitting sigmoid_kernel so the
            // two do not race on conv_out's USM memory. Fixes the OneAPI hang
            // on FusedConv2dSigmoid (verified standalone; ReLU/Tanh paths do
            // not hit this because of different oneDNN primitive caching,
            // but the same fix is defensive and harmless there).
            queue.wait();
            Tensor sig_out = oneapi::sigmoid_kernel(conv_out, queue);
            queue.wait();
            return sig_out;
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
            oneapi::fp16_saturate_if_needed(output, queue);
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
    // Einsum (composed — delegates to einsum_composed to avoid dispatch loop)
    // =========================================================================
    table.register_kernel(OpId::Einsum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto equation = std::string(attrs.get_string(AttrKey::EinsumEquation, ""));
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return {einsum_composed(equation, tensors)};
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

    // GridSample / AffineGrid — native OneAPI kernels
    table.register_single_output_kernel(OpId::GridSample, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return oneapi::grid_sample_kernel(inputs[0], inputs[1], mode, padding_mode, align_corners, get_q(inputs));
    });
    table.register_single_output_kernel(OpId::AffineGrid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size_span = attrs.get_int_list(AttrKey::OutputSize);
        std::vector<int64_t> size(size_span.begin(), size_span.end());
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return oneapi::affine_grid_kernel(inputs[0], size, align_corners, get_q(inputs));
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
            auto result = oneapi::linear_kernel(inputs[0], inputs[1], bias, get_q(inputs));
            oneapi::fp16_saturate_if_needed(result, get_q(inputs));
            return result;
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
            oneapi::fp16_saturate_if_needed(output, get_q(inputs));
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

    table.register_single_output_kernel(OpId::RepeatInterleave,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t num_repeats = attrs.get_int(AttrKey::NumRepeats, 1);
            auto& q = get_q(inputs);
            if (num_repeats >= 0) {
                return oneapi::repeat_interleave_scalar_kernel(inputs[0], num_repeats, dim, q);
            } else {
                return oneapi::repeat_interleave_tensor_kernel(inputs[0], inputs[1], dim, q);
            }
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
            // Support both single Dim and multi-Dims (string-encoded list)
            if (attrs.has(AttrKey::Dims)) {
                auto dims = attrs.get_int_list(AttrKey::Dims);
                Tensor result = inputs[0];
                for (auto d : dims) {
                    result = oneapi::flip_kernel(result, d, get_q(inputs));
                }
                return {result};
            }
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

    // DeformableConv2d (DCNv2)
    table.register_kernel(OpId::DeformableConv2dForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: {input, offset, weight, bias, mask}
            int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
            int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
            int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
            int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
            int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
            return std::vector<Tensor>{oneapi::deformable_conv2d_forward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, offset_groups, get_q(inputs))};
        });

    table.register_kernel(OpId::DeformableConv2dBackwardInput,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: {grad_output, input, offset, weight, mask}
            int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
            int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
            int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
            int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
            int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
            return oneapi::deformable_conv2d_backward_input_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, offset_groups, get_q(inputs));
        });

    table.register_kernel(OpId::DeformableConv2dBackwardWeight,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: {grad_output, input, offset, mask}
            int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
            int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
            int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
            int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
            int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
            int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
            int64_t groups = attrs.get_int(AttrKey::Groups, 1);
            int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
            auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
            return std::vector<Tensor>{oneapi::deformable_conv2d_backward_weight_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3],
                stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
                groups, offset_groups, weight_shape, get_q(inputs))};
        });

    table.register_kernel(OpId::DeformableConv2dBackwardBias,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            // Reuse regular conv2d bias backward (channel-wise sum of grad_output)
            return {oneapi::conv2d_backward_bias(inputs[0], get_q(inputs))};
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

    table.register_kernel(OpId::LinalgLU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [L, U, pivots] = oneapi::linalg_lu_kernel(inputs[0], get_q(inputs));
            return {L, U, pivots};
        });

    table.register_kernel(OpId::LinalgLUSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::linalg_lu_solve_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs))};
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

    table.register_kernel(OpId::SolveTriangular,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool upper = attrs.get_bool(AttrKey::Upper, true);
            bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
            return {oneapi::linalg_solve_triangular_kernel(inputs[0], inputs[1], upper, unitriangular, get_q(inputs))};
        });

    table.register_kernel(OpId::LinalgCholeskySolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto& q = get_q(inputs);
            if (!upper) {
                auto Y = oneapi::linalg_solve_triangular_kernel(inputs[1], inputs[0], false, false, q);
                int64_t ndim = inputs[1].ndim();
                auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                return {oneapi::linalg_solve_triangular_kernel(Lt, Y, true, false, q)};
            } else {
                int64_t ndim = inputs[1].ndim();
                auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                auto Y = oneapi::linalg_solve_triangular_kernel(Ut, inputs[0], false, false, q);
                return {oneapi::linalg_solve_triangular_kernel(inputs[1], Y, true, false, q)};
            }
        });

    table.register_kernel(OpId::Geqrf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [result, tau] = oneapi::linalg_geqrf_kernel(inputs[0], get_q(inputs));
            return {result, tau};
        });

    table.register_single_output_kernel(OpId::Ormqr,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool left = attrs.get_bool(AttrKey::Left, true);
            bool transpose_q = attrs.get_bool(AttrKey::TransposeQ, false);
            return oneapi::linalg_ormqr_kernel(inputs[0], inputs[1], inputs[2],
                left, transpose_q, get_q(inputs));
        });

    table.register_kernel(OpId::LinalgCholeskySolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto& q = get_q(inputs);
            if (!upper) {
                auto Y = oneapi::linalg_solve_triangular_kernel(inputs[1], inputs[0], false, false, q);
                int64_t ndim = inputs[1].ndim();
                auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                return {oneapi::linalg_solve_triangular_kernel(Lt, Y, true, false, q)};
            } else {
                int64_t ndim = inputs[1].ndim();
                auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                auto Y = oneapi::linalg_solve_triangular_kernel(Ut, inputs[0], false, false, q);
                return {oneapi::linalg_solve_triangular_kernel(inputs[1], Y, true, false, q)};
            }
        });

    // =========================================================================
    // LinalgVectorNorm, LinalgMatrixNorm, LinalgVecdot
    // =========================================================================

    // LinalgVectorNorm: delegates to existing Norm kernel
    table.register_kernel(OpId::LinalgVectorNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::norm_kernel(inputs[0], attrs, get_q(inputs))};
        });

    // LinalgMatrixNorm: Frobenius (ord=0), nuclear (ord=1), spectral (ord=2)
    table.register_kernel(OpId::LinalgMatrixNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t ord = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
            auto& q = get_q(inputs);
            if (ord == 0) {
                return {oneapi::norm_kernel(inputs[0], attrs, q)};
            }
            auto [U, S, Vt] = oneapi::linalg_svd_kernel(inputs[0], false, q);
            if (ord == 1) {
                return {oneapi::sum_kernel(S, INT64_MIN, false, q)};
            }
            return {oneapi::max_kernel(S, INT64_MIN, false, q)};
        });

    // LinalgVecdot: sum(a * b, dim)
    table.register_kernel(OpId::LinalgVecdot,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            auto& q = get_q(inputs);
            Tensor product = oneapi::mul_kernel(inputs[0], inputs[1], q);
            return {oneapi::sum_kernel(product, dim, false, q)};
        });

    // =========================================================================
    // LinalgHouseholder, LinalgLDLFactor, LinalgLDLSolve,
    // CholeskyInverse, TensorInv, TensorSolve
    // =========================================================================
    table.register_single_output_kernel(OpId::LinalgHouseholder,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::linalg_householder_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    table.register_kernel(OpId::LinalgLDLFactor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [LD, pivots] = oneapi::linalg_ldl_factor_kernel(inputs[0], get_q(inputs));
            return {LD, pivots};
        });

    table.register_single_output_kernel(OpId::LinalgLDLSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::linalg_ldl_solve_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs));
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

#else // !TENZOR_HAS_ONEMKL — use native SYCL shared-memory linalg fallback
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

    table.register_kernel(OpId::LinalgLU,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [L, U, pivots] = oneapi::linalg_lu_kernel(inputs[0], get_q(inputs));
            return {L, U, pivots};
        });

    table.register_kernel(OpId::LinalgLUSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::linalg_lu_solve_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs))};
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

    table.register_kernel(OpId::SolveTriangular,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool upper = attrs.get_bool(AttrKey::Upper, true);
            bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
            return {oneapi::linalg_solve_triangular_kernel(inputs[0], inputs[1], upper, unitriangular, get_q(inputs))};
        });

    table.register_kernel(OpId::LinalgCholeskySolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto& q = get_q(inputs);
            if (!upper) {
                auto Y = oneapi::linalg_solve_triangular_kernel(inputs[1], inputs[0], false, false, q);
                int64_t ndim = inputs[1].ndim();
                auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                return {oneapi::linalg_solve_triangular_kernel(Lt, Y, true, false, q)};
            } else {
                int64_t ndim = inputs[1].ndim();
                auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                auto Y = oneapi::linalg_solve_triangular_kernel(Ut, inputs[0], false, false, q);
                return {oneapi::linalg_solve_triangular_kernel(inputs[1], Y, true, false, q)};
            }
        });

    table.register_kernel(OpId::Geqrf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [result, tau] = oneapi::linalg_geqrf_kernel(inputs[0], get_q(inputs));
            return {result, tau};
        });

    table.register_single_output_kernel(OpId::Ormqr,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool left = attrs.get_bool(AttrKey::Left, true);
            bool transpose_q = attrs.get_bool(AttrKey::TransposeQ, false);
            return oneapi::linalg_ormqr_kernel(inputs[0], inputs[1], inputs[2],
                left, transpose_q, get_q(inputs));
        });

    // =========================================================================
    // LinalgHouseholder, LinalgLDLFactor, LinalgLDLSolve,
    // CholeskyInverse, TensorInv, TensorSolve (non-onemkl fallback path)
    // =========================================================================
    table.register_single_output_kernel(OpId::LinalgHouseholder,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::linalg_householder_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    table.register_kernel(OpId::LinalgLDLFactor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto [LD, pivots] = oneapi::linalg_ldl_factor_kernel(inputs[0], get_q(inputs));
            return {LD, pivots};
        });

    table.register_single_output_kernel(OpId::LinalgLDLSolve,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::linalg_ldl_solve_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs));
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

#ifdef TENZOR_HAS_ONEMKL
    // SparseSpGEMM: sparse(M,K) × sparse(K,N) -> sparse(M,N).
    // inputs: [0..2] = A's {crow, col, values}, [3..5] = B's {crow, col, values}
    // attrs: M, K, N. Returns three tensors (crow, col, values) of result.
    // IMPORTANT: must call the backend-local `oneapi::spgemm_kernel` — calling
    // the top-level `sparse::spgemm` here re-enters this same kernel through
    // the dispatch table and recurses forever (observed as a hang on OneAPI
    // SpGEMM parity tests).
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            int64_t N = attrs.get_int(AttrKey::N);
            auto a = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            auto b = SparseTensor::sparse_csr(inputs[3], inputs[4], inputs[5], {K, N});
            auto c = oneapi::spgemm_kernel(a, b, get_q(inputs));
            return {c.crow_indices(), c.col_indices(), c.values()};
        });

    // SparseTrsv: solve L @ x = b (1D RHS), L is lower or upper triangular
    // stored in CSR. inputs: [0..2] = L's {crow, col, values}, [3] = b (N,).
    // attrs: N, Upper. Returns x (N,).
    // IMPORTANT: call the backend-local `oneapi::sparse_trsv_kernel` — calling
    // the top-level `sparse::sparse_triangular_solve` re-enters this same
    // kernel via the dispatch table and recurses forever.
    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return oneapi::sparse_trsv_kernel(L, inputs[3], upper, get_q(inputs));
        });

    // SparseTrsm: solve L @ X = B (2D RHS), same semantics as Trsv but b is
    // (N, K). Same note about recursion as SparseTrsv — call the backend-
    // local kernel directly, not the top-level `sparse::sparse_triangular_solve`.
    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return oneapi::sparse_trsm_kernel(L, inputs[3], upper, get_q(inputs));
        });
#else
    // Standalone SYCL SpGEMM/Trsv/Trsm — no oneMKL dependency
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return oneapi::spgemm_standalone_sycl(inputs, attrs, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return oneapi::sparse_trsv_standalone_sycl(
                inputs[0], inputs[1], inputs[2], inputs[3], N, upper, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return oneapi::sparse_trsm_standalone_sycl(
                inputs[0], inputs[1], inputs[2], inputs[3], N, upper, get_q(inputs));
        });
#endif // TENZOR_HAS_ONEMKL

    // SparseSoftmax: row-wise softmax on CSR sparse tensor values
    table.register_single_output_kernel(OpId::SparseSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], shape);
            auto result = sparse::sparse_softmax(sp);
            return result.values();
        });

    // SparseLogSoftmax: row-wise log-softmax on CSR sparse tensor values
    table.register_single_output_kernel(OpId::SparseLogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto shape = attrs.get_int_list(AttrKey::Shape);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], shape);
            auto result = sparse::sparse_log_softmax(sp);
            return result.values();
        });

    // ========================================================================
    // Sampling / Statistics — native OneAPI kernels
    // ========================================================================

    table.register_single_output_kernel(OpId::Bernoulli,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bernoulli_kernel(inputs[0], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::PoissonSample,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::poisson_sample_kernel(inputs[0], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::NormalSample,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::normal_sample_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::ExponentialSample,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::exponential_sample_kernel(inputs[0], get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Multinomial,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_samples = attrs.get_int(AttrKey::NumSamples, 1);
            bool replacement = attrs.get_bool(AttrKey::Replacement, false);
            return oneapi::multinomial_kernel(inputs[0], num_samples, replacement, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::Bucketize,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return oneapi::bucketize_kernel(inputs[0], inputs[1], right, get_q(inputs));
        });

    table.register_kernel(OpId::Histogram,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t bins = attrs.get_int(AttrKey::NumBins, 10);
            double min_val = attrs.get_float(AttrKey::Min, 0.0);
            double max_val = attrs.get_float(AttrKey::Max, 0.0);
            auto [counts, edges] = oneapi::histogram_kernel(inputs[0], bins, min_val, max_val, get_q(inputs));
            return {counts, edges};
        });

    table.register_kernel(OpId::Histogramdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Parse bins list from comma-separated string
            auto bins_list = attrs.get_int_list(AttrKey::BinsList);
            bool density = attrs.get_bool(AttrKey::Density, false);

            // Parse ranges from comma-separated string: min0,max0,min1,max1,...
            std::vector<std::pair<double,double>> ranges;
            auto ranges_str = attrs.get_string(AttrKey::RangesList, "");
            if (!ranges_str.empty()) {
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

            auto [counts, edges] = oneapi::histogramdd_kernel(inputs[0], bins_list, ranges, density, get_q(inputs));
            std::vector<Tensor> results;
            results.push_back(counts);
            for (auto& e : edges) results.push_back(std::move(e));
            return results;
        });

    table.register_single_output_kernel(OpId::CDist,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double p = attrs.get_float(AttrKey::DistP, 2.0);
            return oneapi::cdist_kernel(inputs[0], inputs[1], p, get_q(inputs));
        });

    // =========================================================================
    // Trapezoid / Cumulative Trapezoid / Gradient / PairwiseDistance / Pdist
    // =========================================================================
    table.register_single_output_kernel(OpId::Trapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return oneapi::trapezoid_kernel(inputs[0], dim, dx, x_ptr, get_q(inputs));
    });

    table.register_single_output_kernel(OpId::CumulativeTrapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return oneapi::cumulative_trapezoid_kernel(inputs[0], dim, dx, x_ptr, get_q(inputs));
    });

    table.register_single_output_kernel(OpId::NumericalGradient, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double spacing = attrs.get_float(AttrKey::Spacing, 1.0);
        return oneapi::gradient_kernel(inputs[0], dim, spacing, get_q(inputs));
    });

    table.register_single_output_kernel(OpId::PairwiseDistance, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return oneapi::pairwise_distance_kernel(inputs[0], inputs[1], p, get_q(inputs));
    });

    table.register_single_output_kernel(OpId::Pdist, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return oneapi::pdist_kernel(inputs[0], p, get_q(inputs));
    });

    // STFT / ISTFT — native OneAPI kernels
    table.register_single_output_kernel(OpId::STFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return oneapi::stft_kernel(inputs[0], n_fft, hop_length, win_length,
                                       window, center, normalized, onesided, get_q(inputs));
        });

    table.register_single_output_kernel(OpId::ISTFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            int64_t length_val = attrs.get_int(AttrKey::N, -1);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return oneapi::istft_kernel(inputs[0], n_fft, hop_length, win_length,
                                        window, center, normalized, onesided,
                                        length_val, get_q(inputs));
        });

    // =========================================================================
    // DCT / IDCT
    // =========================================================================
    table.register_single_output_kernel(OpId::DCT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
            std::optional<int64_t> n = std::nullopt;
            int64_t n_val = attrs.get_int(AttrKey::N, -1);
            if (n_val > 0) n = n_val;
            return fft::dct(inputs[0], type, n, dim, norm);
        });

    table.register_single_output_kernel(OpId::IDCT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
            std::optional<int64_t> n = std::nullopt;
            int64_t n_val = attrs.get_int(AttrKey::N, -1);
            if (n_val > 0) n = n_val;
            return fft::idct(inputs[0], type, n, dim, norm);
        });

    table.register_single_output_kernel(OpId::MelScale,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
            double f_min = attrs.get_float(AttrKey::FMin, 0.0);
            double f_max = attrs.get_float(AttrKey::FMax, 0.0);
            int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
            return fft::mel_scale(inputs[0], n_mels, f_min, f_max, sample_rate);
        });

    table.register_single_output_kernel(OpId::MFCC,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
            int64_t n_mfcc = attrs.get_int(AttrKey::NumMFCC, 40);
            int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
            int64_t n_fft = attrs.get_int(AttrKey::NFft, 400);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, 160);
            double f_min = attrs.get_float(AttrKey::FMin, 0.0);
            double f_max = attrs.get_float(AttrKey::FMax, 0.0);
            return fft::mfcc(inputs[0], sample_rate, n_mfcc, n_mels, n_fft, hop_length, f_min, f_max);
        });

    // AdvancedIndex (fancy indexing) — native OneAPI kernel
    table.register_single_output_kernel(OpId::AdvancedIndex,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            std::vector<Tensor> indices;
            indices.reserve(static_cast<size_t>(num_indices));
            for (int64_t i = 0; i < num_indices; ++i) {
                indices.push_back(inputs[1 + i]);
            }
            return oneapi::advanced_index_oneapi_kernel(
                inputs[0], indices, num_indices, get_q(inputs));
        });

    // AdvancedIndexPut (fancy indexing assignment) — native OneAPI kernel
    table.register_single_output_kernel(OpId::AdvancedIndexPut,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            const auto& values = inputs[1];
            std::vector<Tensor> indices;
            indices.reserve(static_cast<size_t>(num_indices));
            for (int64_t i = 0; i < num_indices; ++i) {
                indices.push_back(inputs[2 + i]);
            }
            return oneapi::advanced_index_put_oneapi_kernel(
                inputs[0], indices, values, num_indices, get_q(inputs));
        });

    // ========================================================================
    // Phase 4 ops — native SYCL dispatch (kernels in kernels/math.cpp)
    // ========================================================================
    table.register_single_output_kernel(OpId::Frac,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::frac_kernel(inputs[0], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::LogSigmoid,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::log_sigmoid_kernel(inputs[0], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::Heaviside,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::heaviside_kernel(inputs[0], inputs[1], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::NanToNum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double nan_v = attrs.get_float(AttrKey::NanValue, 0.0);
            double posinf = attrs.get_float(AttrKey::PosInfValue, std::numeric_limits<double>::max());
            double neginf = attrs.get_float(AttrKey::NegInfValue, std::numeric_limits<double>::lowest());
            return oneapi::nan_to_num_kernel(inputs[0], nan_v, posinf, neginf, get_q(inputs));
        });
    table.register_single_output_kernel(OpId::BitwiseAnd,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bitwise_and_kernel(inputs[0], inputs[1], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::BitwiseOr,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bitwise_or_kernel(inputs[0], inputs[1], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::BitwiseXor,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bitwise_xor_kernel(inputs[0], inputs[1], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::BitwiseNot,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bitwise_not_kernel(inputs[0], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::BitwiseLeftShift,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bitwise_left_shift_kernel(inputs[0], inputs[1], get_q(inputs));
        });
    table.register_single_output_kernel(OpId::BitwiseRightShift,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
            return oneapi::bitwise_right_shift_kernel(inputs[0], inputs[1], get_q(inputs));
        });

    // Native SYCL kernels for RReLU, RReLUBackward, LogSigmoidBackward
    table.register_kernel(OpId::RReLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
            float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
            bool training = attrs.get_bool(AttrKey::Training, false);
            return {oneapi::rrelu_kernel(inputs[0], lower, upper, training, get_q(inputs))};
        });

    table.register_kernel(OpId::RReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
            float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
            return {oneapi::rrelu_backward_kernel(inputs[0], inputs[1], lower, upper, get_q(inputs))};
        });

    table.register_kernel(OpId::LogSigmoidBackward,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::log_sigmoid_backward_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // NaN reductions — native SYCL kernels
    table.register_kernel(OpId::CountNonzero,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::count_nonzero_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::Nansum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::nansum_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Nanmean,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::nanmean_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Aminmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return oneapi::aminmax_kernel(inputs[0], dim, keepdim, get_q(inputs));
        });
    table.register_kernel(OpId::IndexAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::index_add_kernel(inputs[0], dim, inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_kernel(OpId::IndexCopy,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::index_copy_kernel(inputs[0], dim, inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_kernel(OpId::IndexFill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            float value = attrs.get_float(AttrKey::Value, 0.0f);
            return {oneapi::index_fill_kernel(inputs[0], dim, inputs[1], value, get_q(inputs))};
        });

    table.register_kernel(OpId::ScatterReduce,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
            bool include_self = attrs.get_bool(AttrKey::IncludeSelf, true);
            return {oneapi::scatter_reduce_kernel(inputs[0], dim, inputs[1], inputs[2], reduce, include_self, get_q(inputs))};
        });

    // SelectScatter: clone input, then copy src into the selected slice
    table.register_kernel(OpId::SelectScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            const auto& input = inputs[0];
            const auto& src = inputs[1];
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t index = attrs.get_int(AttrKey::Index, 0);

            auto output = input.clone();
            int64_t ndim = static_cast<int64_t>(output.shape().size());
            if (dim < 0) dim += ndim;

            auto dst_slice = output.slice(dim, index, index + 1, 1);
            auto dst_sh = dst_slice.shape();
            auto src_reshaped = src.reshape(std::vector<int64_t>(dst_sh.begin(), dst_sh.end())).contiguous();

            auto n = dst_slice.numel();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* dst_ptr = static_cast<char*>(dst_slice.data_ptr());
            const auto* src_ptr = static_cast<const char*>(src_reshaped.data_ptr());
            auto& q = oneapi_internal::get_queue(output.device().index);
            if (dst_slice.is_contiguous()) {
                q.memcpy(dst_ptr, src_ptr, n * elem_size).wait();
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
                    q.memcpy(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size).wait();
                    for (int64_t d = ndims - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < dst_shape_v[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return {output};
        });

    // SliceScatter: clone input, then copy src into the sliced region
    table.register_kernel(OpId::SliceScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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

            if (start < 0) start += dim_size;
            if (end < 0) end += dim_size + 1;
            if (start < 0) start = 0;
            if (end > dim_size) end = dim_size;

            auto dst_slice = output.slice(dim, start, end, step);
            auto dst_sh = dst_slice.shape();
            auto src_reshaped = src.reshape(std::vector<int64_t>(dst_sh.begin(), dst_sh.end())).contiguous();

            auto n = dst_slice.numel();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* dst_ptr = static_cast<char*>(dst_slice.data_ptr());
            const auto* src_ptr = static_cast<const char*>(src_reshaped.data_ptr());
            auto& q = oneapi_internal::get_queue(output.device().index);
            if (dst_slice.is_contiguous()) {
                q.memcpy(dst_ptr, src_ptr, n * elem_size).wait();
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
                    q.memcpy(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size).wait();
                    for (int64_t d = ndims - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < dst_shape_v[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            return {output};
        });

    // DiagonalScatter: clone input, place src values along the diagonal
    table.register_kernel(OpId::DiagonalScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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

            int64_t diag_len;
            if (offset >= 0) {
                diag_len = std::min(size1, size2 - offset);
            } else {
                diag_len = std::min(size1 + offset, size2);
            }
            if (diag_len <= 0) return {output};

            auto strides = output.strides();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* out_ptr = static_cast<char*>(output.data_ptr());
            auto src = src_in.contiguous();
            const auto* src_ptr = static_cast<const char*>(src.data_ptr());

            int64_t batch_size = 1;
            std::vector<int64_t> batch_dims;
            for (int64_t d = 0; d < ndim; d++) {
                if (d != dim1 && d != dim2) {
                    batch_dims.push_back(d);
                    batch_size *= shape[d];
                }
            }

            auto& q = oneapi_internal::get_queue(output.device().index);
            std::vector<int64_t> batch_coord(batch_dims.size(), 0);
            for (int64_t b = 0; b < batch_size; b++) {
                int64_t base = 0;
                for (size_t i = 0; i < batch_dims.size(); i++) {
                    base += batch_coord[i] * strides[batch_dims[i]];
                }

                int64_t r0 = (offset >= 0) ? 0 : -offset;
                int64_t c0 = (offset >= 0) ? offset : 0;
                for (int64_t k = 0; k < diag_len; k++) {
                    int64_t out_elem_offset = base + (r0 + k) * strides[dim1] + (c0 + k) * strides[dim2];
                    int64_t src_elem_idx = b * diag_len + k;
                    q.memcpy(out_ptr + out_elem_offset * elem_size,
                             src_ptr + src_elem_idx * elem_size, elem_size).wait();
                }

                for (int64_t i = static_cast<int64_t>(batch_dims.size()) - 1; i >= 0; i--) {
                    batch_coord[i]++;
                    if (batch_coord[i] < shape[batch_dims[i]]) break;
                    batch_coord[i] = 0;
                }
            }
            return {output};
        });

    // =========================================================================
    // Fused GEMM Operations (composed from existing OneAPI ops)
    // =========================================================================
    table.register_single_output_kernel(OpId::Addmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto& q = get_q(inputs);
            auto mm = oneapi::matmul_kernel(inputs[1], inputs[2], q);
            if (alpha != 1.0) {
                auto alpha_t = tenzor::full({1}, alpha, mm.dtype(), mm.device());
                mm = oneapi::mul_kernel(mm, alpha_t, q);
            }
            if (beta == 0.0) return mm;
            Tensor inp = inputs[0];
            if (beta != 1.0) {
                auto beta_t = tenzor::full({1}, beta, inp.dtype(), inp.device());
                inp = oneapi::mul_kernel(inp, beta_t, q);
            }
            return oneapi::add_kernel(inp, mm, q);
        });

    table.register_single_output_kernel(OpId::Addmv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto& q = get_q(inputs);
            auto vec_col = inputs[2].reshape({inputs[2].shape()[0], 1});
            auto mv = oneapi::matmul_kernel(inputs[1], vec_col, q);
            mv = mv.reshape({inputs[1].shape()[0]});
            if (alpha != 1.0) {
                auto alpha_t = tenzor::full({1}, alpha, mv.dtype(), mv.device());
                mv = oneapi::mul_kernel(mv, alpha_t, q);
            }
            if (beta == 0.0) return mv;
            Tensor inp = inputs[0];
            if (beta != 1.0) {
                auto beta_t = tenzor::full({1}, beta, inp.dtype(), inp.device());
                inp = oneapi::mul_kernel(inp, beta_t, q);
            }
            return oneapi::add_kernel(inp, mv, q);
        });

    table.register_single_output_kernel(OpId::Baddbmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto& q = get_q(inputs);
            auto bmm = oneapi::bmm_kernel(inputs[1], inputs[2], q);
            if (alpha != 1.0) {
                auto alpha_t = tenzor::full({1}, alpha, bmm.dtype(), bmm.device());
                bmm = oneapi::mul_kernel(bmm, alpha_t, q);
            }
            if (beta == 0.0) return bmm;
            Tensor inp = inputs[0];
            if (beta != 1.0) {
                auto beta_t = tenzor::full({1}, beta, inp.dtype(), inp.device());
                inp = oneapi::mul_kernel(inp, beta_t, q);
            }
            return oneapi::add_kernel(inp, bmm, q);
        });

    // =========================================================================
    // Log-Cumulative-Sum-Exp (native SYCL kernel)
    // =========================================================================
    table.register_kernel(OpId::Logcumsumexp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return {oneapi::logcumsumexp_kernel(inputs[0], dim, get_q(inputs))};
        });

    // =========================================================================
    // Bincount (native SYCL kernel)
    // =========================================================================
    table.register_kernel(OpId::Bincount,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t minlength = attrs.get_int(AttrKey::Minlength, 0);
            const Tensor* weights_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
            return {oneapi::bincount_kernel(inputs[0], weights_ptr, minlength, get_q(inputs))};
        });

    // =========================================================================
    // New Reduction Operations (CumMax, CumMin, Fmax, Fmin, Isin, Kthvalue, etc.)
    // =========================================================================

    table.register_kernel(OpId::CumMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = oneapi::cummax_kernel(inputs[0], dim, get_q(inputs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::CumMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = oneapi::cummin_kernel(inputs[0], dim, get_q(inputs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::Fmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::fmax_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Fmin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::fmin_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Isin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::isin_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Kthvalue, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        auto [values, indices] = oneapi::kthvalue_kernel(inputs[0], k, dim, keepdim, get_q(inputs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::Quantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::quantile_kernel(inputs[0], q, dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Nanquantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return {oneapi::nanquantile_kernel(inputs[0], q, dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Nanmedian,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return {oneapi::nanmedian_kernel(inputs[0], dim, false, get_q(inputs))};
        });

    table.register_kernel(OpId::Histc,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t bins = attrs.get_int(AttrKey::N, 100);
            double min_val = attrs.get_float(AttrKey::Alpha, 0.0);
            double max_val = attrs.get_float(AttrKey::Beta, 0.0);
            return {oneapi::histc_kernel(inputs[0], bins, min_val, max_val, get_q(inputs))};
        });

    table.register_kernel(OpId::UniqueConsecutive, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool return_inverse = attrs.get_bool(AttrKey::Keepdim, false);
        auto [unique_vals, inverse, counts] = oneapi::unique_consecutive_kernel(
            inputs[0], return_inverse, get_q(inputs));
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    table.register_single_output_kernel(OpId::SegmentReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t axis = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        return oneapi::segment_reduce_kernel(inputs[0], inputs[1], reduce, axis, get_q(inputs));
    });

    // =========================================================================
    // TakeAlongDim
    // =========================================================================
    table.register_single_output_kernel(OpId::TakeAlongDim, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return oneapi::take_along_dim_kernel(inputs[0], inputs[1], dim, get_q(inputs));
    });

    // =========================================================================
    // MaskedScatter
    // =========================================================================
    table.register_single_output_kernel(OpId::MaskedScatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return oneapi::masked_scatter_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs));
    });

    // =========================================================================
    // TrilIndices — CPU generation + transfer
    // =========================================================================
    table.register_single_output_kernel(OpId::TrilIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
        return oneapi::tril_indices_kernel(row, col, offset, get_q_device(device_id));
    });

    // =========================================================================
    // TriuIndices — CPU generation + transfer
    // =========================================================================
    table.register_single_output_kernel(OpId::TriuIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::DeviceId, 0));
        return oneapi::triu_indices_kernel(row, col, offset, get_q_device(device_id));
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 2D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = oneapi::fractional_maxpool2d_forward_kernel(inputs[0], out_h, out_w, samples, get_q(inputs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return oneapi::fractional_maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape, get_q(inputs));
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 3D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = oneapi::fractional_maxpool3d_forward_kernel(inputs[0], out_d, out_h, out_w, samples, get_q(inputs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return oneapi::fractional_maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape, get_q(inputs));
    });

    // =========================================================================
    // Phase 9: Max Unpool 2D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return oneapi::max_unpool2d_forward_kernel(inputs[0], inputs[1], out_h, out_w, get_q(inputs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return oneapi::max_unpool2d_backward_kernel(inputs[0], inputs[1], input_shape, get_q(inputs));
    });

    // =========================================================================
    // Phase 9: Max Unpool 3D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return oneapi::max_unpool3d_forward_kernel(inputs[0], inputs[1], out_d, out_h, out_w, get_q(inputs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return oneapi::max_unpool3d_backward_kernel(inputs[0], inputs[1], input_shape, get_q(inputs));
    });

    // =========================================================================
    // New Phase: Deg2Rad / Rad2Deg / Logit
    // =========================================================================
    table.register_kernel(OpId::Deg2Rad,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::deg2rad_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Rad2Deg,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::rad2deg_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::Logit,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
            return {oneapi::logit_kernel(inputs[0], eps, get_q(inputs))};
        });

    // =========================================================================
    // New Phase: Signbit / IsPosInf / IsNegInf / IsReal
    // =========================================================================
    table.register_kernel(OpId::Signbit,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::signbit_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::IsPosInf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::isposinf_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::IsNegInf,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::isneginf_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::IsReal,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::isreal_kernel(inputs[0], get_q(inputs))};
        });

    // =========================================================================
    // New Phase: FloatPower / Xlog1py / Ldexp
    // =========================================================================
    table.register_kernel(OpId::FloatPower,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::float_power_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Xlog1py,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::xlog1py_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Ldexp,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::ldexp_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    // =========================================================================
    // New Phase: Frexp (two-output)
    // =========================================================================
    table.register_kernel(OpId::Frexp,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return oneapi::frexp_kernel(inputs[0], get_q(inputs));
        });

    // =========================================================================
    // New Phase: DiagEmbed / Diagflat
    // =========================================================================
    table.register_kernel(OpId::DiagEmbed,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
            int64_t dim1 = attrs.get_int(AttrKey::Dim0, -2);
            int64_t dim2 = attrs.get_int(AttrKey::Dim1, -1);
            return {oneapi::diag_embed_kernel(inputs[0], offset, dim1, dim2, get_q(inputs))};
        });

    table.register_kernel(OpId::Diagflat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
            return {oneapi::diagflat_kernel(inputs[0], offset, get_q(inputs))};
        });

    // =========================================================================
    // New Phase: NanVar / NanStd
    // =========================================================================
    table.register_kernel(OpId::NanVar,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            int64_t correction = attrs.get_int(AttrKey::Correction, 1);
            return {oneapi::nanvar_kernel(inputs[0], dim, keepdim, correction, get_q(inputs))};
        });

    table.register_kernel(OpId::NanStd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            int64_t correction = attrs.get_int(AttrKey::Correction, 1);
            return {oneapi::nanstd_kernel(inputs[0], dim, keepdim, correction, get_q(inputs))};
        });

    // =========================================================================
    // Nested Tensor Operations (fallback: unbind segments, apply regular ops)
    // =========================================================================
    table.register_kernel(OpId::NestedSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor& values = inputs[0];
            const Tensor& offsets = inputs[1];
            auto shape = values.shape();
            int64_t total_len = shape[0];
            int64_t D = (shape.size() > 1) ? shape[1] : 1;
            int64_t B = offsets.numel() - 1;

            Tensor output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()),
                                          values.dtype(), values.device());
            auto& queue = oneapi_internal::get_queue(values.device().index);

            const float* vals_ptr = values.data<float>();
            const int64_t* off_ptr = offsets.data<int64_t>();
            float* out_ptr = output.data<float>();
            int64_t D_val = D;

            // One work-item per (batch, d) pair
            queue.parallel_for(sycl::range<2>(B, D), [=](sycl::id<2> id) {
                int64_t b = id[0];
                int64_t d = id[1];
                if (d >= D_val) return;

                int64_t start = off_ptr[b];
                int64_t end = off_ptr[b + 1];
                int64_t len = end - start;
                if (len <= 0) return;

                // Find max
                float max_val = -1e38f;
                for (int64_t s = 0; s < len; ++s) {
                    float v = vals_ptr[(start + s) * D_val + d];
                    if (v > max_val) max_val = v;
                }
                // Sum of exp(x - max)
                float sum = 0.0f;
                for (int64_t s = 0; s < len; ++s) {
                    sum += sycl::exp(vals_ptr[(start + s) * D_val + d] - max_val);
                }
                // Write softmax
                float inv_sum = 1.0f / sum;
                for (int64_t s = 0; s < len; ++s) {
                    int64_t idx = (start + s) * D_val + d;
                    out_ptr[idx] = sycl::exp(vals_ptr[idx] - max_val) * inv_sum;
                }
            }).wait();

            return {output};
        });

    table.register_kernel(OpId::NestedLogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor& values = inputs[0];
            const Tensor& offsets = inputs[1];
            auto shape = values.shape();
            int64_t total_len = shape[0];
            int64_t D = (shape.size() > 1) ? shape[1] : 1;
            int64_t B = offsets.numel() - 1;

            Tensor output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()),
                                          values.dtype(), values.device());
            auto& queue = oneapi_internal::get_queue(values.device().index);

            const float* vals_ptr = values.data<float>();
            const int64_t* off_ptr = offsets.data<int64_t>();
            float* out_ptr = output.data<float>();
            int64_t D_val = D;

            // One work-item per (batch, d) pair — batch in dim 0, d in dim 1
            queue.parallel_for(sycl::range<2>(B, D), [=](sycl::id<2> id) {
                int64_t b = id[0];
                int64_t d = id[1];
                if (d >= D_val) return;

                int64_t start = off_ptr[b];
                int64_t end = off_ptr[b + 1];
                int64_t len = end - start;
                if (len <= 0) return;

                // Find max
                float max_val = -1e38f;
                for (int64_t s = 0; s < len; ++s) {
                    float v = vals_ptr[(start + s) * D_val + d];
                    if (v > max_val) max_val = v;
                }
                // Sum of exp(x - max)
                float sum = 0.0f;
                for (int64_t s = 0; s < len; ++s) {
                    sum += sycl::exp(vals_ptr[(start + s) * D_val + d] - max_val);
                }
                // Write log-softmax
                float log_sum = sycl::log(sum);
                for (int64_t s = 0; s < len; ++s) {
                    int64_t idx = (start + s) * D_val + d;
                    out_ptr[idx] = (vals_ptr[idx] - max_val) - log_sum;
                }
            }).wait();

            return {output};
        });

    table.register_kernel(OpId::NestedSum,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor& values = inputs[0];
            const Tensor& offsets = inputs[1];
            auto shape = values.shape();
            int64_t D = (shape.size() > 1) ? shape[1] : 1;
            int64_t B = offsets.numel() - 1;

            // Output: one row per batch element, shape [B, D]
            Tensor output = tenzor::zeros({B, D}, values.dtype(), values.device());
            auto& queue = oneapi_internal::get_queue(values.device().index);

            const float* vals_ptr = values.data<float>();
            const int64_t* off_ptr = offsets.data<int64_t>();
            float* out_ptr = output.data<float>();
            int64_t D_val = D;

            // One work-item per (batch, d) pair
            queue.parallel_for(sycl::range<2>(B, D), [=](sycl::id<2> id) {
                int64_t b = id[0];
                int64_t d = id[1];
                if (d >= D_val) return;

                int64_t start = off_ptr[b];
                int64_t end = off_ptr[b + 1];

                float sum = 0.0f;
                for (int64_t s = start; s < end; ++s) {
                    sum += vals_ptr[s * D_val + d];
                }
                out_ptr[b * D_val + d] = sum;
            }).wait();

            return {output};
        });

    table.register_kernel(OpId::NestedMean,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor& values = inputs[0];
            const Tensor& offsets = inputs[1];
            auto shape = values.shape();
            int64_t D = (shape.size() > 1) ? shape[1] : 1;
            int64_t B = offsets.numel() - 1;

            // Output: one row per batch element, shape [B, D]
            Tensor output = tenzor::zeros({B, D}, values.dtype(), values.device());
            auto& queue = oneapi_internal::get_queue(values.device().index);

            const float* vals_ptr = values.data<float>();
            const int64_t* off_ptr = offsets.data<int64_t>();
            float* out_ptr = output.data<float>();
            int64_t D_val = D;

            // One work-item per (batch, d) pair
            queue.parallel_for(sycl::range<2>(B, D), [=](sycl::id<2> id) {
                int64_t b = id[0];
                int64_t d = id[1];
                if (d >= D_val) return;

                int64_t start = off_ptr[b];
                int64_t end = off_ptr[b + 1];
                int64_t len = end - start;
                if (len <= 0) return;

                float sum = 0.0f;
                for (int64_t s = start; s < end; ++s) {
                    sum += vals_ptr[s * D_val + d];
                }
                out_ptr[b * D_val + d] = sum / static_cast<float>(len);
            }).wait();

            return {output};
        });

    table.register_kernel(OpId::NestedLayerNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // Per-row LN on packed values (LN operates on last dim, same for all rows)
            const Tensor& values = inputs[0];
            const Tensor& offsets = inputs[1];
            const Tensor& gamma = inputs[2];  // weight [D]
            const Tensor& beta = inputs[3];   // bias [D]
            auto shape = values.shape();
            int64_t total_rows = shape[0];
            int64_t D = shape.back();
            float eps = attrs.get_float(AttrKey::Eps, 1e-5f);

            // Output same shape as values — every row is independently normalized
            Tensor output = tenzor::empty(std::vector<int64_t>(shape.begin(), shape.end()),
                                          values.dtype(), values.device());
            auto& queue = oneapi_internal::get_queue(values.device().index);

            const float* vals_ptr = values.data<float>();
            const float* gamma_ptr = gamma.data<float>();
            const float* beta_ptr = beta.data<float>();
            float* out_ptr = output.data<float>();
            int64_t D_val = D;
            float eps_val = eps;

            // One work-item per row — each row is independently layer-normed
            // Offsets not needed for the actual computation since LN is per-row,
            // but we keep total_rows from shape to iterate all packed rows.
            queue.parallel_for(sycl::range<1>(total_rows), [=](sycl::id<1> id) {
                int64_t row = id[0];
                const float* row_ptr = vals_ptr + row * D_val;
                float* row_out = out_ptr + row * D_val;

                // Compute mean
                float mean = 0.0f;
                for (int64_t d = 0; d < D_val; ++d) {
                    mean += row_ptr[d];
                }
                mean /= static_cast<float>(D_val);

                // Compute variance
                float var = 0.0f;
                for (int64_t d = 0; d < D_val; ++d) {
                    float diff = row_ptr[d] - mean;
                    var += diff * diff;
                }
                var /= static_cast<float>(D_val);

                // Normalize and apply affine
                float inv_std = sycl::rsqrt(var + eps_val);
                for (int64_t d = 0; d < D_val; ++d) {
                    row_out[d] = (row_ptr[d] - mean) * inv_std * gamma_ptr[d] + beta_ptr[d];
                }
            }).wait();

            return {output};
        });

    table.register_kernel(OpId::NestedLinear,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto result = tenzor::matmul(inputs[0], inputs[2].transpose(0, 1));
            if (inputs.size() > 3) {
                result = tenzor::add(result, inputs[3]);
            }
            return {result};
        });

    // NestedAttention — native SYCL kernel reading offsets on device
    table.register_single_output_kernel(OpId::NestedAttention,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return oneapi::nested_attention_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                scale, causal, get_q(inputs));
        });

    table.register_kernel(OpId::NestedToPadded,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            const Tensor& values = inputs[0];
            const Tensor& offsets = inputs[1];
            auto shape = values.shape();
            int64_t B = offsets.numel() - 1;
            int64_t max_len = attrs.get_int(AttrKey::MaxLen, 0);
            float padding_value = attrs.get_float(AttrKey::PaddingValue, 0.0f);
            int64_t D = (shape.size() > 1) ? shape[1] : 1;

            auto padded = tenzor::full({B, max_len, D}, padding_value, values.dtype(), values.device());
            auto& queue = oneapi_internal::get_queue(values.device().index);

            const float* vals_ptr = values.data<float>();
            const int64_t* off_ptr = offsets.data<int64_t>();
            float* pad_ptr = padded.data<float>();
            int64_t D_val = D;
            int64_t max_len_val = max_len;

            // One work-item per (batch, position, d) triple
            queue.parallel_for(sycl::range<3>(B, max_len, D), [=](sycl::id<3> id) {
                int64_t b = id[0];
                int64_t t = id[1];
                int64_t d = id[2];
                if (d >= D_val || t >= max_len_val) return;

                int64_t start = off_ptr[b];
                int64_t end = off_ptr[b + 1];
                int64_t len = end - start;

                int64_t out_idx = (b * max_len_val + t) * D_val + d;
                if (t < len) {
                    pad_ptr[out_idx] = vals_ptr[(start + t) * D_val + d];
                }
                // else: already filled with padding_value by tenzor::full
            }).wait();

            return {padded};
        });

    table.register_kernel(OpId::NestedFromPadded,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            const Tensor& padded = inputs[0];
            const Tensor& offsets = inputs[1];
            int64_t B = offsets.numel() - 1;
            int64_t max_len = padded.shape()[1];
            int64_t D = (padded.shape().size() > 2) ? padded.shape()[2] : 1;

            // Read only offsets[B] to get total_len for output allocation
            Tensor total_len_scalar = tenzor::slice(offsets, 0, B, B + 1).to(Device::cpu());
            int64_t total_len = total_len_scalar.data<int64_t>()[0];

            auto values = tenzor::empty({total_len, D}, padded.dtype(), padded.device());
            auto& queue = oneapi_internal::get_queue(padded.device().index);

            const float* pad_ptr = padded.data<float>();
            const int64_t* off_ptr = offsets.data<int64_t>();
            float* vals_ptr = values.data<float>();
            int64_t D_val = D;
            int64_t max_len_val = max_len;

            // One work-item per (batch, position, d) triple
            // Use total output elements as an upper bound — each work-item maps to
            // a (b, t, d) coordinate; we iterate batches to find the right one.
            queue.parallel_for(sycl::range<2>(B, D), [=](sycl::id<2> id) {
                int64_t b = id[0];
                int64_t d = id[1];
                if (d >= D_val) return;

                int64_t start = off_ptr[b];
                int64_t end = off_ptr[b + 1];
                int64_t len = end - start;

                for (int64_t t = 0; t < len; ++t) {
                    vals_ptr[(start + t) * D_val + d] =
                        pad_ptr[(b * max_len_val + t) * D_val + d];
                }
            }).wait();

            return {values};
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
    // NestedAttentionBackward — backward for segmented attention (native SYCL)
    // =========================================================================
    table.register_kernel(OpId::NestedAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return oneapi::nested_attention_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                inputs[5], inputs[6], scale, causal, get_q(inputs));
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
