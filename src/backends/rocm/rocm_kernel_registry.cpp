/**
 * @file rocm_kernel_registry.cpp
 * @brief ROCm kernel registration for O(1) dispatch
 *
 * Registers all ROCm/HIP kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/nn/layers/flex_attention.hpp"  // Wave C: process-wide score_mod registry
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include <sstream>
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/attr_macros.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/indexing.hpp"  // for where(), needed by composed Nansum/Nanmean dim path
#include "tenzor/core/tensor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include <hip/hip_runtime.h>
#include "rocm_error.hpp"
#include <iostream>
#include <cmath>
#include <cstdlib>
#include <charconv>
#include <limits>
#include <climits>
#include <tuple>
#include <utility>

#include "tenzor/backend/dtype_from_string.hpp"

namespace tenzor {

// Helper to extract HIP stream from attributes
inline hipStream_t get_hip_stream(const OpAttributes& attrs) {
    if (attrs.has(AttrKey::Stream)) {
        return static_cast<hipStream_t>(
            reinterpret_cast<void*>(static_cast<uintptr_t>(attrs.get_int(AttrKey::Stream)))
        );
    }
    return nullptr;  // Default stream
}


// Helper to convert device string to Device
inline Device device_from_string(std::string_view s, Device default_val = Device::rocm(0)) {
    if (s.empty()) return default_val;
    // Parse strings like "rocm:0", "cuda:1", "cpu"
    size_t colon_pos = s.find(':');
    std::string_view type_str = (colon_pos != std::string_view::npos) ? s.substr(0, colon_pos) : s;
    int32_t device_id = 0;
    if (colon_pos != std::string_view::npos) {
        auto id_str = s.substr(colon_pos + 1);
        int val = 0;
        auto [ptr, ec] = std::from_chars(id_str.data(), id_str.data() + id_str.size(), val);
        if (ec == std::errc{}) device_id = static_cast<int32_t>(val);
    }

    Device::Type type = Device::Type::CPU;
    if (type_str == "rocm" || type_str == "hip") type = Device::Type::ROCm;
    else if (type_str == "cuda") type = Device::Type::CUDA;
    else if (type_str == "vulkan") type = Device::Type::Vulkan;

    return Device{type, device_id};
}

// Forward declarations for ROCm kernels
namespace rocm {
    // DataLayout enum (must match definition in conv2d.hip.cpp)
    enum class DataLayout {
        NCHW,  // Batch, Channels, Height, Width (default)
        NHWC   // Batch, Height, Width, Channels (TensorFlow style)
    };
    // Binary operations
    auto add_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // Unary operations
    auto sqrt_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto neg_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto abs_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sign_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto log_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto exp_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto pow_kernel(const Tensor& input, double exponent, hipStream_t stream) -> Tensor;
    auto clamp_kernel(const Tensor& input, double min_val, double max_val, hipStream_t stream) -> Tensor;
    auto reciprocal_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto floor_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto ceil_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto round_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // Trigonometric operations
    auto sin_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto cos_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto tan_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto asin_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto acos_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto atan_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sinh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto cosh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // Extended math operations
    auto log2_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto log10_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto log1p_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto exp2_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto expm1_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto erf_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto erfc_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // Special math (native ROCm — replaces previous CPU-roundtrip fallbacks)
    auto gamma_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto lgamma_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto digamma_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto polygamma_kernel(int64_t n, const Tensor& input, hipStream_t stream) -> Tensor;
    auto beta_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto betainc_kernel(const Tensor& a, const Tensor& b, const Tensor& x, hipStream_t stream) -> Tensor;
    auto bessel_j0_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bessel_j1_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bessel_y0_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bessel_y1_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bessel_i0_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bessel_i1_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto erfinv_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sinc_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto zeta_kernel(const Tensor& s, const Tensor& q, hipStream_t stream) -> Tensor;

    // New math ops (PyTorch parity)
    auto logaddexp_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto logaddexp2_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto xlogy_kernel(const Tensor& x, const Tensor& y, hipStream_t stream) -> Tensor;
    auto i0e_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto i1e_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto entr_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto spherical_bessel_j0_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto cosine_similarity_kernel(const Tensor& a, const Tensor& b,
                                   int64_t dim, double eps, hipStream_t stream) -> Tensor;
    auto renorm_kernel(const Tensor& input, double p, int64_t dim,
                       double maxnorm, hipStream_t stream) -> Tensor;

    // Ndtr / LogNdtr / Multigammaln
    auto ndtr_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto log_ndtr_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto multigammaln_kernel(const Tensor& input, int d, hipStream_t stream) -> Tensor;

    // Grid sample / affine grid (native ROCm kernels — replaces previous CPU fallbacks)
    auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                            const std::string& mode, const std::string& padding_mode,
                            bool align_corners, hipStream_t stream) -> Tensor;
    auto affine_grid_kernel_host(const Tensor& theta, const std::vector<int64_t>& size,
                                  bool align_corners, hipStream_t stream) -> Tensor;
    auto grid_sample_backward_kernel_host(const Tensor& grad_output,
                                          const Tensor& input, const Tensor& grid,
                                          const std::string& mode,
                                          const std::string& padding_mode,
                                          bool align_corners, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto affine_grid_backward_kernel_host(const Tensor& grad_grid,
                                          const std::vector<int64_t>& size,
                                          bool align_corners, hipStream_t stream) -> Tensor;

    // Sampling / statistics (native ROCm — replaces previous CPU fallbacks)
    auto bernoulli_kernel(const Tensor& probs, hipStream_t stream) -> Tensor;
    auto poisson_sample_kernel(const Tensor& rates, hipStream_t stream) -> Tensor;
    auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, hipStream_t stream) -> Tensor;
    auto exponential_sample_kernel(const Tensor& rate, hipStream_t stream) -> Tensor;
    auto gamma_sample_kernel(const Tensor& concentration, const Tensor& rate, hipStream_t stream) -> Tensor;
    auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                            bool replacement, hipStream_t stream) -> Tensor;
    auto bucketize_kernel(const Tensor& input, const Tensor& boundaries,
                          bool right, hipStream_t stream) -> Tensor;
    auto histogram_kernel(const Tensor& input, int64_t bins,
                          double min_val, double max_val,
                          hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto histogramdd_kernel(const Tensor& input, std::vector<int64_t> bins,
                            std::vector<std::pair<double,double>> ranges,
                            bool density, hipStream_t stream)
        -> std::pair<Tensor, std::vector<Tensor>>;
    auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                      hipStream_t stream) -> Tensor;
    auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr, hipStream_t stream) -> Tensor;
    auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr, hipStream_t stream) -> Tensor;
    auto gradient_kernel(const Tensor& input, int64_t dim, double spacing, hipStream_t stream) -> Tensor;
    auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p, hipStream_t stream) -> Tensor;
    auto pdist_kernel(const Tensor& input, double p, hipStream_t stream) -> Tensor;

    // STFT / ISTFT (native ROCm — replaces previous CPU fallbacks)
    auto stft_kernel(const Tensor& input, int64_t n_fft,
                     int64_t hop_length, int64_t win_length,
                     const Tensor& window, bool center,
                     bool normalized, bool onesided,
                     hipStream_t stream) -> Tensor;
    auto istft_kernel(const Tensor& input, int64_t n_fft,
                      int64_t hop_length, int64_t win_length,
                      const Tensor& window, bool center,
                      bool normalized, bool onesided,
                      int64_t length, hipStream_t stream) -> Tensor;

    // Bool predicate operations
    auto isnan_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto isinf_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto isfinite_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto signbit_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto isposinf_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto isneginf_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto isreal_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // Unary math operations (new)
    auto deg2rad_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto rad2deg_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto logit_kernel(const Tensor& input, double eps, hipStream_t stream) -> Tensor;

    // Binary math operations (new)
    auto float_power_kernel(const Tensor& base, const Tensor& exp, hipStream_t stream) -> Tensor;
    auto xlog1py_kernel(const Tensor& x, const Tensor& y, hipStream_t stream) -> Tensor;
    auto ldexp_kernel(const Tensor& x, const Tensor& n, hipStream_t stream) -> Tensor;

    // Two-output operations
    auto frexp_kernel(const Tensor& input, hipStream_t stream) -> std::vector<Tensor>;

    // Tensor manipulation operations
    auto diag_embed_kernel(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2,
                           hipStream_t stream) -> Tensor;
    auto diagflat_kernel(const Tensor& input, int64_t offset, hipStream_t stream) -> Tensor;

    // Binary math operations
    auto atan2_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto fmod_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // New binary math operations
    auto hypot_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto copysign_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto nextafter_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto gcd_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto lcm_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto igamma_kernel(const Tensor& a, const Tensor& x, hipStream_t stream) -> Tensor;
    auto igammac_kernel(const Tensor& a, const Tensor& x, hipStream_t stream) -> Tensor;

    // New unary math operations
    auto rsqrt_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto square_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto asinh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto acosh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto atanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // Ternary operations
    auto lerp_kernel(const Tensor& a, const Tensor& b, const Tensor& weight, hipStream_t stream) -> Tensor;
    auto addcmul_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, hipStream_t stream) -> Tensor;
    auto addcdiv_kernel(const Tensor& input, const Tensor& tensor1, const Tensor& tensor2, float value, hipStream_t stream) -> Tensor;

    // Logical operations
    auto logical_and_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto logical_or_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto logical_not_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto logical_xor_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // Element-wise min/max
    auto minimum_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto maximum_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // Complex number operations
    auto conj_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto real_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto imag_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto angle_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto polar_kernel(const Tensor& abs_t, const Tensor& angle_t, hipStream_t stream) -> Tensor;
    auto complex_tensor_kernel(const Tensor& real_t, const Tensor& imag_t, hipStream_t stream) -> Tensor;
    auto cross_kernel(const Tensor& a, const Tensor& b, int64_t dim, hipStream_t stream) -> Tensor;

    // Dot product
    auto dot_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // In-place operations
    void add_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream);
    void sub_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream);
    void mul_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream);
    void div_inplace_kernel(Tensor& a, const Tensor& b, hipStream_t stream);

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto median_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> std::vector<Tensor>;
    auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> std::vector<Tensor>;

    // NaN-aware reductions and counting (native HIP kernels)
    auto count_nonzero_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto count_nonzero_dim_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto nansum_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto nanmean_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto nanvar_kernel(const Tensor& input, bool unbiased, hipStream_t stream) -> Tensor;
    auto nanstd_kernel(const Tensor& input, bool unbiased, hipStream_t stream) -> Tensor;
    auto aminmax_kernel(const Tensor& input, hipStream_t stream) -> std::pair<Tensor, Tensor>;

    // Activation functions
    auto relu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto gelu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, double alpha, hipStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, double alpha, hipStream_t stream) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto selu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto swish_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto mish_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto hardswish_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto hardsigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold, hipStream_t stream) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, hipStream_t stream) -> Tensor;
    auto rrelu_kernel(const Tensor& input, float lower, float upper, bool training, hipStream_t stream) -> Tensor;
    auto rrelu_backward_kernel(const Tensor& grad_output, const Tensor& input, float lower, float upper, hipStream_t stream) -> Tensor;
    auto log_sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;

    // Index operations (native HIP kernels)
    auto index_add_kernel(const Tensor& self, const Tensor& index, const Tensor& source,
                          int64_t dim, hipStream_t stream) -> Tensor;
    auto index_copy_kernel(const Tensor& self, const Tensor& index, const Tensor& source,
                           int64_t dim, hipStream_t stream) -> Tensor;
    auto index_fill_kernel(const Tensor& self, const Tensor& index,
                           int64_t dim, double value, hipStream_t stream) -> Tensor;
    auto scatter_reduce_kernel(const Tensor& self, const Tensor& index, const Tensor& source,
                               int64_t dim, const std::string& reduce, bool include_self,
                               hipStream_t stream) -> Tensor;

    // In-place activation kernels (Phase 7.1) — direct aliased-in/out launches.
    void relu_inplace_kernel(Tensor& target, hipStream_t stream);
    void sigmoid_inplace_kernel(Tensor& target, hipStream_t stream);
    void tanh_inplace_kernel(Tensor& target, hipStream_t stream);
    void leaky_relu_inplace_kernel(Tensor& target, float alpha, hipStream_t stream);
    void gelu_inplace_kernel(Tensor& target, hipStream_t stream);

    // Softmax operations
    auto softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream, float temperature = 1.0f) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, hipStream_t stream) -> Tensor;

    // Tensor creation
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, double value, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto fill_kernel(const Tensor& input, double value, hipStream_t stream) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, hipStream_t stream) -> Tensor;

    // Transform operations
    auto contiguous_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto clone_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& shape, hipStream_t stream) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, hipStream_t stream) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, hipStream_t stream) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream) -> Tensor;

    // BatchNorm operations
    void batchnorm2d_mean_var(const Tensor& input, Tensor& mean, Tensor& variance, hipStream_t stream);
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance,
                             float epsilon, hipStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance,
                                    const Tensor& gamma, const Tensor& beta,
                                    float epsilon, hipStream_t stream) -> Tensor;
    void batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var,
                                          const Tensor& batch_mean, const Tensor& batch_var,
                                          float momentum, hipStream_t stream);
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input,
                              const Tensor& mean, const Tensor& variance, const Tensor& gamma,
                              float epsilon, hipStream_t stream)
        -> std::tuple<Tensor, Tensor, Tensor>;

    // Convolution operations
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                               hipStream_t stream, DataLayout layout = DataLayout::NCHW) -> Tensor;
    auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                               const std::vector<int64_t>& input_shape,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                               hipStream_t stream) -> Tensor;
    auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& weight_shape,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                hipStream_t stream) -> Tensor;
    // Wave B3: per-axis overloads now defined in conv2d.hip.cpp natively.
    // The previous iso-or-throw shims are gone; conv2d.hip.cpp provides both
    // per-axis and scalar entry points (the scalar version forwards to per-axis
    // with duplicated values).
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               int64_t dil_h, int64_t dil_w,
                               int64_t groups, hipStream_t stream,
                               DataLayout layout = DataLayout::NCHW) -> Tensor;
    auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                               const std::vector<int64_t>& input_shape,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               int64_t dil_h, int64_t dil_w,
                               int64_t groups, hipStream_t stream) -> Tensor;
    auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& weight_shape,
                                int64_t stride_h, int64_t stride_w,
                                int64_t pad_h, int64_t pad_w,
                                int64_t dil_h, int64_t dil_w,
                                int64_t groups, hipStream_t stream) -> Tensor;
    auto conv2d_backward_bias(const Tensor& grad_output, hipStream_t stream) -> Tensor;

    // Conv3d operations (conv3d.hip.cpp)
    auto conv3d_forward_hip(const Tensor& input, const Tensor& weight, const Tensor& bias,
                            const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                            const std::vector<int64_t>& dilation, int64_t groups,
                            hipStream_t stream) -> Tensor;
    auto conv3d_backward_input_hip(const Tensor& grad_output, const Tensor& weight,
                                    const std::vector<int64_t>& input_shape,
                                    const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                    const std::vector<int64_t>& dilation, int64_t groups,
                                    hipStream_t stream) -> Tensor;
    auto conv3d_backward_weight_hip(const Tensor& grad_output, const Tensor& input,
                                     const std::vector<int64_t>& weight_shape,
                                     const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                     const std::vector<int64_t>& dilation, int64_t groups,
                                     hipStream_t stream) -> Tensor;
    auto conv3d_backward_bias_hip(const Tensor& grad_output, hipStream_t stream) -> Tensor;

    // ConvTranspose3d operations (conv3d.hip.cpp)
    auto conv_transpose3d_forward_hip(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                       const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                       const std::vector<int64_t>& output_padding,
                                       const std::vector<int64_t>& dilation, int64_t groups,
                                       hipStream_t stream) -> Tensor;
    auto conv_transpose3d_backward_input_hip(const Tensor& grad_output, const Tensor& weight,
                                              const std::vector<int64_t>& input_shape,
                                              const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                              const std::vector<int64_t>& dilation, int64_t groups,
                                              hipStream_t stream) -> Tensor;
    auto conv_transpose3d_backward_weight_hip(const Tensor& grad_output, const Tensor& input,
                                               const std::vector<int64_t>& weight_shape,
                                               const std::vector<int64_t>& stride, const std::vector<int64_t>& padding,
                                               const std::vector<int64_t>& dilation, int64_t groups,
                                               hipStream_t stream) -> Tensor;
    auto conv_transpose3d_backward_bias_hip(const Tensor& grad_output, hipStream_t stream) -> Tensor;

    // Pooling operations
    auto maxpool2d_forward_hip(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               bool return_indices,
                               hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape,
                                hipStream_t stream) -> Tensor;
    auto avgpool2d_forward_hip(const Tensor& input, int64_t kernel_h, int64_t kernel_w,
                               int64_t stride_h, int64_t stride_w,
                               int64_t pad_h, int64_t pad_w,
                               bool count_include_pad,
                               hipStream_t stream) -> Tensor;
    auto avgpool2d_backward_hip(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                int64_t kernel_h, int64_t kernel_w,
                                int64_t stride_h, int64_t stride_w,
                                int64_t pad_h, int64_t pad_w,
                                bool count_include_pad,
                                hipStream_t stream) -> Tensor;
    auto adaptive_avgpool2d_hip(const Tensor& input, int64_t output_h, int64_t output_w,
                                hipStream_t stream) -> Tensor;
    auto adaptive_maxpool2d_hip(const Tensor& input, int64_t output_h, int64_t output_w,
                                bool return_indices,
                                hipStream_t stream) -> std::pair<Tensor, Tensor>;

    // Indexing operations
    auto gather_hip(const Tensor& input, int64_t dim, const Tensor& index, hipStream_t stream) -> Tensor;
    auto scatter_hip(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src,
                     hipStream_t stream) -> Tensor;
    auto index_select_hip(const Tensor& input, int64_t dim, const Tensor& index, hipStream_t stream) -> Tensor;
    auto masked_fill_hip(const Tensor& input, const Tensor& mask, double value, hipStream_t stream) -> Tensor;
    auto masked_select_hip(const Tensor& input, const Tensor& mask, hipStream_t stream) -> Tensor;
    auto searchsorted_hip(const Tensor& sorted_sequence, const Tensor& values, bool right, hipStream_t stream) -> Tensor;

    // LSTM/GRU cell operations (operate on pre-computed gates, matching CUDA pattern)
    auto lstm_cell_forward_kernel(const Tensor& gates, const Tensor& c_prev,
                                  int64_t batch_size, int64_t hidden_size,
                                  hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_h, const Tensor& grad_c,
                                   const Tensor& gates, const Tensor& c_prev,
                                   const Tensor& c_out,
                                   int64_t batch_size, int64_t hidden_size,
                                   hipStream_t stream) -> std::pair<Tensor, Tensor>;
    struct GRUBackwardOutputs {
        Tensor grad_reset;
        Tensor grad_update;
        Tensor grad_new_input;
        Tensor grad_new_hidden;
        Tensor grad_h_prev;
    };
    auto gru_cell_forward_kernel(const Tensor& reset_gates, const Tensor& update_gates,
                                 const Tensor& new_gates_input, const Tensor& new_gates_hidden,
                                 const Tensor& h_prev,
                                 int64_t batch_size, int64_t hidden_size,
                                 hipStream_t stream) -> Tensor;
    auto gru_cell_backward_kernel(const Tensor& grad_h, const Tensor& reset_gates,
                                  const Tensor& update_gates, const Tensor& new_gates_input,
                                  const Tensor& new_gates_hidden, const Tensor& h_prev,
                                  int64_t batch_size, int64_t hidden_size,
                                  hipStream_t stream) -> GRUBackwardOutputs;

    // Full sequence LSTM/GRU operations
    auto lstm_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                            const Tensor& bias, const Tensor& h0, const Tensor& c0,
                            hipStream_t stream) -> std::vector<Tensor>;
    auto gru_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                           const Tensor& bias, const Tensor& h0, const Tensor& bias_hh,
                           hipStream_t stream) -> std::vector<Tensor>;

    // Multi-layer and bidirectional RNN operations
    auto lstm_multi_layer_forward_kernel(
        const Tensor& input,
        const std::vector<Tensor>& W_ih_list,
        const std::vector<Tensor>& W_hh_list,
        const std::vector<Tensor>& bias_list,
        const Tensor& h0, const Tensor& c0,
        hipStream_t stream) -> std::vector<Tensor>;
    auto gru_multi_layer_forward_kernel(
        const Tensor& input,
        const std::vector<Tensor>& W_ih_list,
        const std::vector<Tensor>& W_hh_list,
        const std::vector<Tensor>& bias_list,
        const std::vector<Tensor>& bias_hh_list,
        const Tensor& h0,
        hipStream_t stream) -> std::vector<Tensor>;
    auto bilstm_forward_kernel(
        const Tensor& input,
        const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
        const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
        const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
        const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
        const Tensor& h0, const Tensor& c0,
        hipStream_t stream) -> std::vector<Tensor>;

    // Additional transform operations
    auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, hipStream_t stream)
        -> std::vector<Tensor>;
    auto flip_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;

    // Comparison operations
    auto eq_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // Additional indexing operations
    auto where_hip(const Tensor& condition, const Tensor& x, const Tensor& y, hipStream_t stream) -> Tensor;
    auto slice_hip(const Tensor& input, int64_t dim, int64_t start, int64_t end, int64_t step, hipStream_t stream) -> Tensor;
    auto cat_hip(const std::vector<Tensor>& tensors, int64_t dim, hipStream_t stream) -> Tensor;
    auto take_hip(const Tensor& input, const Tensor& indices, hipStream_t stream) -> Tensor;
    auto put_hip(Tensor& input, const Tensor& indices, const Tensor& source, bool accumulate, hipStream_t stream) -> Tensor;

    // Additional transform operations
    auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim, hipStream_t stream) -> Tensor;
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, hipStream_t stream) -> Tensor;
    auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, hipStream_t stream) -> Tensor;
    auto stack_kernel(const std::vector<Tensor>& tensors, int64_t dim, hipStream_t stream) -> std::vector<Tensor>;
    auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, hipStream_t stream) -> std::vector<Tensor>;
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, void* stream_ptr) -> Tensor;
    auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, hipStream_t stream) -> Tensor;
    auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim, hipStream_t stream) -> Tensor;
    auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats, int64_t dim, hipStream_t stream) -> Tensor;

    // Creation operations
    auto arange_kernel(double start, double end, double step, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto linspace_kernel(double start, double end, int64_t steps, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, int64_t k, DType dtype, Device device, hipStream_t stream) -> Tensor;

    // Additional convolution and pooling operations
    // Q.8: per-axis dilation_h/w added (was forced to 1).
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                         int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                         int64_t output_padding_h, int64_t output_padding_w,
                                         int64_t dilation_h, int64_t dilation_w, int64_t groups, hipStream_t stream) -> Tensor;
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                 int64_t dilation_h, int64_t dilation_w, hipStream_t stream) -> Tensor;
    auto depthwise_conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride, int64_t padding, int64_t dilation, hipStream_t stream) -> Tensor;
    auto depthwise_conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW,
                                 int64_t dD, int64_t dH, int64_t dW, hipStream_t stream) -> Tensor;

    // Deformable Convolution v2 (DCNv2)
    auto deformable_conv2d_forward_kernel(
        const Tensor& input, const Tensor& offset, const Tensor& weight,
        const Tensor& bias, const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        hipStream_t stream) -> Tensor;
    auto deformable_conv2d_backward_input_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& offset,
        const Tensor& weight, const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        hipStream_t stream) -> std::vector<Tensor>;
    auto deformable_conv2d_backward_weight_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& offset,
        const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        const std::vector<int64_t>& weight_shape,
        hipStream_t stream) -> Tensor;

    auto adaptive_avgpool2d_backward_hip(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto adaptive_maxpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices, const Tensor& input, hipStream_t stream) -> Tensor;

    // 1D Pooling operations
    // Q.6: per-axis std::array<int64_t, 1> signatures (1D has only one spatial
    // axis but the API matches the per-axis sweep across 2D/3D).
    auto maxpool1d_forward_hip(const Tensor& input, std::array<int64_t, 1> kernel_size, std::array<int64_t, 1> stride,
                               std::array<int64_t, 1> padding, std::array<int64_t, 1> dilation, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto maxpool1d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto avgpool1d_forward_hip(const Tensor& input, std::array<int64_t, 1> kernel_size, std::array<int64_t, 1> stride,
                               std::array<int64_t, 1> padding, hipStream_t stream) -> Tensor;
    auto avgpool1d_backward_hip(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                std::array<int64_t, 1> kernel_size, std::array<int64_t, 1> stride, std::array<int64_t, 1> padding,
                                hipStream_t stream) -> Tensor;
    auto adaptive_maxpool1d_forward_hip(const Tensor& input, int64_t output_size,
                                        hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool1d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                          const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto adaptive_avgpool1d_forward_hip(const Tensor& input, int64_t output_size,
                                         hipStream_t stream) -> Tensor;
    auto adaptive_avgpool1d_backward_hip(const Tensor& grad_output,
                                          const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;

    // 3D Pooling operations
    // Q.6: per-axis std::array<int64_t, 3> signatures honour asymmetric D/H/W.
    auto maxpool3d_forward_hip(const Tensor& input, std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride,
                               std::array<int64_t, 3> padding, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto maxpool3d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto avgpool3d_forward_hip(const Tensor& input, std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride,
                               std::array<int64_t, 3> padding, hipStream_t stream) -> Tensor;
    auto avgpool3d_backward_hip(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding,
                                hipStream_t stream) -> Tensor;
    auto adaptive_maxpool3d_forward_hip(const Tensor& input,
                                         int64_t output_d, int64_t output_h, int64_t output_w,
                                         hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool3d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                          const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto adaptive_avgpool3d_forward_hip(const Tensor& input,
                                         int64_t output_d, int64_t output_h, int64_t output_w,
                                         hipStream_t stream) -> Tensor;
    auto adaptive_avgpool3d_backward_hip(const Tensor& grad_output,
                                          const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;

    // Fractional Max Pool + Max Unpool
    auto fractional_maxpool2d_forward_hip(const Tensor& input, int64_t out_h, int64_t out_w,
                                           const Tensor* random_samples, hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                            const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto fractional_maxpool3d_forward_hip(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w,
                                           const Tensor* random_samples, hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool3d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                            const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto max_unpool2d_forward_hip(const Tensor& input, const Tensor& indices,
                                   int64_t out_h, int64_t out_w, hipStream_t stream) -> Tensor;
    auto max_unpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                    const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto max_unpool3d_forward_hip(const Tensor& input, const Tensor& indices,
                                   int64_t out_d, int64_t out_h, int64_t out_w, hipStream_t stream) -> Tensor;
    auto max_unpool3d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                    const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto max_unpool1d_forward_hip(const Tensor& input, const Tensor& indices,
                                   int64_t out_l, hipStream_t stream) -> Tensor;
    auto max_unpool1d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                    const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;

    // Normalization operations
    auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                          const Tensor* weight, const Tensor* bias, float eps, hipStream_t stream) -> Tensor;
    auto layer_norm_kernel_with_stats(const Tensor& input,
                                      const std::vector<int64_t>& normalized_shape,
                                      const Tensor* weight, const Tensor* bias,
                                      float eps, hipStream_t stream)
        -> std::tuple<Tensor, Tensor, Tensor>;
    auto layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd, const Tensor* weight,
                                    hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto group_norm_kernel(const Tensor& input, int64_t num_groups, const Tensor* weight,
                          const Tensor* bias, float eps, hipStream_t stream) -> Tensor;
    auto group_norm_forward_with_stats(const Tensor& input, int64_t num_groups, const Tensor* weight,
                                       const Tensor* bias, float eps, hipStream_t stream) -> std::vector<Tensor>;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd, int64_t num_groups,
                                    const Tensor* weight, hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto instance_norm_kernel(const Tensor& input, const Tensor* weight, const Tensor* bias,
                             float eps, hipStream_t stream) -> Tensor;
    auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& mean, const Tensor& rstd, const Tensor* weight,
                                       hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // Fused operations
    auto fused_linear_relu_hip(const Tensor& input, const Tensor& weight, const Tensor* bias, hipStream_t stream) -> Tensor;
    auto fused_batchnorm_relu_hip(const Tensor& input, const Tensor& running_mean, const Tensor& running_var,
                                  const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_hip(const Tensor& logits, const Tensor& targets,
                                         const std::string& reduction) -> Tensor;
    auto fused_softmax_cross_entropy_grad_hip(const Tensor& logits, const Tensor& targets)
                                         -> std::pair<Tensor, Tensor>;
    auto fused_add_relu_hip(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_hip(const Tensor& input) -> Tensor;
    auto fused_layer_norm_hip(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                              const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_conv_batchnorm_relu_hip(const Tensor& conv_output, const Tensor& running_mean,
                                       const Tensor& running_var, const Tensor& weight,
                                       const Tensor& bias, float eps) -> Tensor;
    auto fused_attention_hip(const Tensor& Q, const Tensor& K, const Tensor& V,
                             float scale, bool causal = false,
                             float dropout_p = 0.0f, uint32_t rng_seed = 0u) -> std::pair<Tensor, Tensor>;
    auto flash_attention_backward_hip(
        const Tensor& dO, const Tensor& Q, const Tensor& K, const Tensor& V,
        const Tensor& O, const Tensor& L, float scale, bool causal) -> std::vector<Tensor>;

    // Audit A.11 — native Float64 FlashAttention (flash_attention_f64.hip.cpp).
    // audit V.16: stream is now plumbed so workspace zeroing and kernel launches
    // share the dispatcher's stream and HIP errors are surfaced via HIP_CHECK.
    auto fused_attention_hip_f64(const Tensor& Q, const Tensor& K, const Tensor& V,
                                 double scale, bool causal,
                                 hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto flash_attention_backward_hip_f64(
        const Tensor& dO, const Tensor& Q, const Tensor& K, const Tensor& V,
        const Tensor& O, double scale, bool causal,
        hipStream_t stream) -> std::vector<Tensor>;

    // Fused RMSNorm
    auto fused_rms_norm_hip(const Tensor& input, const Tensor& weight,
                            float eps) -> std::pair<Tensor, Tensor>;

    // Fused LayerNorm Backward
    auto fused_layer_norm_backward_hip(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& weight, const Tensor& mean,
                                       const Tensor& inv_std,
                                       const std::vector<int64_t>& normalized_shape)
        -> std::tuple<Tensor, Tensor, Tensor>;

    // Fused Conv2D + BatchNorm + ReLU (full pipeline)
    auto fused_conv2d_bn_relu_full_hip(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                        const Tensor& bn_mean, const Tensor& bn_var,
                                        const Tensor& bn_gamma, const Tensor& bn_beta,
                                        int64_t stride, int64_t padding, float eps) -> Tensor;

    // Fused optimizer steps
    auto fused_sgd_step_hip(Tensor& param, const Tensor& grad, Tensor* momentum_buffer,
                            float lr, float momentum, float weight_decay, float dampening,
                            bool nesterov, hipStream_t stream) -> void;
    auto fused_adam_step_hip(Tensor& param, const Tensor& grad, Tensor& exp_avg, Tensor& exp_avg_sq,
                             double lr, double beta1, double beta2, double eps, double weight_decay,
                             int64_t step, bool decoupled_weight_decay, hipStream_t stream,
                             Tensor* max_exp_avg_sq, bool amsgrad) -> void;
    auto fused_rmsprop_step_hip(Tensor& param, const Tensor& grad, Tensor& square_avg,
                                 Tensor* grad_avg, Tensor* momentum_buffer,
                                 float lr, float alpha, float eps, float weight_decay,
                                 float momentum, bool centered, hipStream_t stream) -> void;
    auto fused_adadelta_step_hip(Tensor& param, const Tensor& grad, Tensor& square_avg,
                                  Tensor& acc_delta, float rho, float eps, float lr,
                                  float weight_decay, hipStream_t stream) -> void;
    auto fused_adagrad_step_hip(Tensor& param, const Tensor& grad, Tensor& sum_sq,
                                 float lr, float lr_decay, float eps, float weight_decay,
                                 int64_t step, hipStream_t stream) -> void;
    auto fused_adam_atan2_step_hip(Tensor& param, const Tensor& grad, Tensor& exp_avg,
                                    Tensor& exp_avg_sq, Tensor* max_exp_avg_sq,
                                    float lr, float beta1, float beta2, float eps,
                                    float weight_decay, int64_t step, bool amsgrad,
                                    hipStream_t stream) -> void;

    // Embedding, Linear, Dropout operations
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, hipStream_t stream) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   int64_t num_embeddings, hipStream_t stream) -> Tensor;
    auto gather_relative_position_bias_kernel(const Tensor& table, const Tensor& indices,
                                              int64_t num_positions, int64_t num_heads, hipStream_t stream) -> Tensor;
    auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                       hipStream_t stream) -> Tensor;
    auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight,
                                hipStream_t stream) -> std::vector<Tensor>;
    auto dropout_kernel(const Tensor& input, float p, bool training, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p,
                                 hipStream_t stream) -> Tensor;

    // Vision operations (Unfold, Fold, Interpolate)
    auto unfold_kernel(const Tensor& input,
                       int64_t kernel_h, int64_t kernel_w,
                       int64_t stride_h, int64_t stride_w,
                       int64_t padding_h, int64_t padding_w,
                       int64_t dilation_h, int64_t dilation_w,
                       hipStream_t stream) -> Tensor;
    auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                     int64_t kernel_h, int64_t kernel_w,
                     int64_t stride_h, int64_t stride_w,
                     int64_t padding_h, int64_t padding_w,
                     int64_t dilation_h, int64_t dilation_w,
                     hipStream_t stream) -> Tensor;
    auto interpolate_kernel(const Tensor& input, const std::vector<int64_t>& size,
                            const std::string& mode, bool align_corners, hipStream_t stream) -> Tensor;
    // D3-followup ROCm: native atomicAdd-scatter bilinear backward.
    auto interpolate_backward_kernel(const Tensor& grad_output,
                                      const std::vector<int64_t>& input_size,
                                      const std::string& mode, bool align_corners,
                                      hipStream_t stream) -> Tensor;

    // FFT operations (rocFFT or native HIP fallback)
    auto rocm_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                         const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                          const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                          const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                           const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& n_vec,
                          const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& n_vec,
                           const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& n_vec,
                          const std::string& norm, hipStream_t stream) -> Tensor;
    auto rocm_ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& n_vec,
                           const std::string& norm, hipStream_t stream) -> Tensor;

    // Linear algebra operations (rocSOLVER or native HIP fallback)
    auto linalg_det_kernel(const Tensor& A, hipStream_t stream) -> Tensor;
    auto linalg_inv_kernel(const Tensor& A, hipStream_t stream) -> Tensor;
    auto linalg_solve_kernel(const Tensor& A, const Tensor& B, hipStream_t stream) -> Tensor;
    auto linalg_svd_kernel(const Tensor& A, bool full_matrices, hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_qr_kernel(const Tensor& A, hipStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto linalg_eigh_kernel(const Tensor& A, hipStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto linalg_eig_kernel(const Tensor& A, hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_cholesky_kernel(const Tensor& A, bool upper, hipStream_t stream) -> Tensor;
    auto linalg_lu_kernel(const Tensor& A, hipStream_t stream)
        -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                                const Tensor& B, hipStream_t stream) -> Tensor;
    auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                         bool upper, bool unitriangular,
                                         hipStream_t stream) -> Tensor;
    auto linalg_geqrf_kernel(const Tensor& A, hipStream_t stream)
        -> std::tuple<Tensor, Tensor>;
    auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                              const Tensor& C, bool left, bool transpose_q,
                              hipStream_t stream) -> Tensor;
    auto linalg_ldl_factor_kernel(const Tensor& A, hipStream_t stream)
        -> std::tuple<Tensor, Tensor>;
    auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                                  const Tensor& B, hipStream_t stream) -> Tensor;
    auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                    hipStream_t stream) -> Tensor;
    // Sparse operations (sparse.hip.cpp) — available with or without rocSPARSE
    auto rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) -> Tensor;
    auto rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) -> Tensor;
    auto rocm_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) -> Tensor;
#ifdef TENZOR_HAS_ROCSPARSE
    // SpGEMM / triangular solve via rocSPARSE. When rocSPARSE isn't available
    // the dispatch table below registers the standalone HIP kernels
    // (spgemm_standalone_hip / sparse_trsv_standalone_hip / sparse_trsm_…)
    // so SparseSpGEMM / SparseTrsv / SparseTrsm always have a ROCm-side
    // implementation — no CPU fallback.
    auto rocm_spgemm_kernel(const SparseTensor& a, const SparseTensor& b) -> SparseTensor;
    auto rocm_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b,
                                  bool upper) -> Tensor;
    auto rocm_sparse_trsm_kernel(const SparseTensor& L, const Tensor& B,
                                  bool upper) -> Tensor;
#endif

    // Standalone GPU implementations (always available)
    auto spgemm_standalone_hip(std::span<const Tensor> inputs, const OpAttributes& attrs,
                               hipStream_t stream) -> std::vector<Tensor>;
    auto sparse_trsv_standalone_hip(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                    const Tensor& b, int64_t N, bool upper, hipStream_t stream) -> Tensor;
    auto sparse_trsm_standalone_hip(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                    const Tensor& B, int64_t N, bool upper, hipStream_t stream) -> Tensor;

    // Sort/TopK/ArgSort/Unique operations (sort.hip.cpp)
    auto sort_kernel(const Tensor& input, int64_t dim, bool descending,
                     hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest,
                     bool sorted, hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending,
                        hipStream_t stream) -> Tensor;
    auto unique_kernel(const Tensor& input, bool sorted_output, bool return_inverse,
                       bool return_counts, hipStream_t stream)
        -> std::tuple<Tensor, Tensor, Tensor>;

    // ScatterAdd (indexing.hip.cpp)
    auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index,
                            const Tensor& src, hipStream_t stream) -> Tensor;
    auto take_along_dim_hip(const Tensor& input, const Tensor& indices, int64_t dim, hipStream_t stream) -> Tensor;
    auto masked_scatter_hip(const Tensor& input, const Tensor& mask, const Tensor& source, hipStream_t stream) -> Tensor;
    auto tril_indices_hip(int64_t row, int64_t col, int64_t offset, hipStream_t stream) -> Tensor;
    auto triu_indices_hip(int64_t row, int64_t col, int64_t offset, hipStream_t stream) -> Tensor;

    // Cast, StridedFill, ToMemoryFormat (transform.hip.cpp)
    auto cast_kernel(const Tensor& input, DType target_dtype, hipStream_t stream) -> Tensor;
    auto strided_fill_kernel(Tensor& self, double value, hipStream_t stream) -> void;
    auto to_memory_format_kernel(const Tensor& input, MemoryFormat format, void* stream_ptr) -> Tensor;

    // Any/All reductions (reduction.hip.cpp)
    auto any_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto all_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;

    // Triu/Tril/Diag/Trace (transform.hip.cpp)
    auto triu_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor;
    auto tril_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor;
    auto diag_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor;
    auto trace_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // CumSum, CumProd, HasInfNan, Logcumsumexp, Bincount (math.hip.cpp)
    auto cumsum_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto has_inf_nan_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto logcumsumexp_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto bincount_kernel(const Tensor& input, const Tensor* weights, int64_t minlength, hipStream_t stream) -> Tensor;

    // New reduction operations (math.hip.cpp)
    auto cummax_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cummin_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto fmax_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto fmin_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto isin_kernel(const Tensor& elements, const Tensor& test_elements, hipStream_t stream) -> Tensor;
    auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim, bool keepdim, hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto quantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto nanquantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto nanmedian_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val, hipStream_t stream) -> Tensor;
    auto unique_consecutive_kernel(const Tensor& input, bool return_inverse, hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets, const std::string& reduce, int64_t axis, hipStream_t stream) -> Tensor;

    // BoxIoU (vision.hip.cpp)
    auto box_iou_hip(const Tensor& boxes1, const Tensor& boxes2, int iou_type, hipStream_t stream) -> Tensor;

    // NMS (nms.hip.cpp)
    auto nms_forward(const Tensor& boxes, const Tensor& scores, float iou_threshold, hipStream_t stream) -> Tensor;

    // EmbeddingBagForward/Backward (indexing.hip.cpp)
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                       const std::string& mode, int64_t embedding_dim,
                                       bool include_last_offset, hipStream_t stream) -> std::vector<Tensor>;
    auto embedding_bag_backward_kernel(const Tensor& grad_output,
                                       const Tensor& indices,
                                       const Tensor& offsets,
                                       const OpAttributes& attrs,
                                       hipStream_t stream) -> Tensor;

    // Randint (math.hip.cpp)
    auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape,
                        DType dtype, Device device, hipStream_t stream) -> Tensor;

    // Quantized operations (quantization.hip.cpp)
    auto quantized_linear_hip(
        const Tensor& input, const Tensor& weight, const Tensor* bias,
        float input_scale, int32_t input_zero_point,
        float weight_scale, int32_t weight_zero_point,
        float output_scale, int32_t output_zero_point,
        hipStream_t stream) -> Tensor;
    auto quantized_conv2d_hip(
        const Tensor& input, const Tensor& weight, const Tensor* bias,
        int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
        float input_scale, int32_t input_zero_point,
        float weight_scale, int32_t weight_zero_point,
        float output_scale, int32_t output_zero_point,
        hipStream_t stream) -> Tensor;

    // Trunc (math.hip.cpp)
    auto trunc_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // New Phase 4 ops (math.hip.cpp)
    auto frac_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto heaviside_kernel(const Tensor& input, const Tensor& values, hipStream_t stream) -> Tensor;
    auto nan_to_num_kernel(const Tensor& input, double nan_v, double posinf_v, double neginf_v, hipStream_t stream) -> Tensor;
    auto log_sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bitwise_and_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto bitwise_or_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto bitwise_xor_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto bitwise_not_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto bitwise_left_shift_kernel(const Tensor& input, const Tensor& shift, hipStream_t stream) -> Tensor;
    auto bitwise_right_shift_kernel(const Tensor& input, const Tensor& shift, hipStream_t stream) -> Tensor;

    // OneHot, Nonzero (indexing.hip.cpp)
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes,
                        hipStream_t stream) -> Tensor;
    auto nonzero_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // ROI Align (roi_align.hip.cpp)
    auto roi_align_forward(const Tensor& features, const Tensor& rois,
                           int64_t output_h, int64_t output_w,
                           float spatial_scale, int64_t sampling_ratio,
                           bool aligned, hipStream_t stream) -> Tensor;
    auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                            int64_t batch_size, int64_t feat_height, int64_t feat_width,
                            float spatial_scale, int64_t sampling_ratio,
                            bool aligned, hipStream_t stream) -> Tensor;

    // RMSNorm backward (fused_ops.hip.cpp)
    auto fused_rms_norm_backward_hip(const Tensor& grad_output, const Tensor& input,
                                      const Tensor& weight, const Tensor& rrms)
        -> std::tuple<Tensor, Tensor>;

    // Advanced indexing (indexing.hip.cpp)
    auto advanced_index_rocm_kernel(const Tensor& src, const std::vector<Tensor>& indices,
                                    int64_t num_indices, hipStream_t stream) -> Tensor;
    auto advanced_index_put_rocm_kernel(const Tensor& src, const std::vector<Tensor>& indices,
                                        const Tensor& values, int64_t num_indices,
                                        hipStream_t stream) -> Tensor;

    // STFT / ISTFT (stft.hip.cpp)
    auto stft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length,
                     int64_t win_length, const Tensor& window, bool center,
                     bool normalized, bool onesided, hipStream_t stream) -> Tensor;
    auto istft_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length,
                      int64_t win_length, const Tensor& window, bool center,
                      bool normalized, bool onesided, int64_t length,
                      hipStream_t stream) -> Tensor;
    // Nested tensor operations
    auto nested_softmax_hip(const Tensor& values, const Tensor& offsets, int64_t dim, hipStream_t stream) -> Tensor;
    auto nested_log_softmax_hip(const Tensor& values, const Tensor& offsets, int64_t dim, hipStream_t stream) -> Tensor;
    auto nested_sum_hip(const Tensor& values, const Tensor& offsets, hipStream_t stream) -> Tensor;
    auto nested_mean_hip(const Tensor& values, const Tensor& offsets, hipStream_t stream) -> Tensor;
    auto nested_layer_norm_hip(const Tensor& values, const Tensor& offsets, const Tensor& weight, const Tensor& bias, float eps, hipStream_t stream) -> Tensor;
    auto nested_linear_hip(const Tensor& values, const Tensor& weight, const Tensor* bias, hipStream_t stream) -> Tensor;
    auto nested_attention_hip(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& q_offsets, const Tensor& kv_offsets, float scale, bool causal, hipStream_t stream) -> Tensor;
    auto nested_attention_backward_hip(const Tensor& grad_out, const Tensor& Q, const Tensor& K, const Tensor& V,
                                        const Tensor& attn_out, const Tensor& q_offsets, const Tensor& kv_offsets,
                                        float scale, bool causal, hipStream_t stream) -> std::vector<Tensor>;
    auto nested_to_padded_hip(const Tensor& values, const Tensor& offsets, int64_t max_len, float padding_value, hipStream_t stream) -> Tensor;
    auto nested_from_padded_hip(const Tensor& padded, const Tensor& offsets, hipStream_t stream) -> Tensor;

    // CTC loss
    auto ctc_loss_forward_kernel(const Tensor& log_probs, const Tensor& targets,
                                 const Tensor& input_lengths, const Tensor& target_lengths,
                                 int64_t blank, bool zero_infinity, hipStream_t stream)
        -> std::vector<Tensor>;
} // namespace rocm

/**
 * @brief Register all ROCm kernels with the dispatch table.
 *
 * Each registration wraps a rocm::* kernel call with stream handling.
 */
void register_rocm_kernels(BackendDispatchTable& table) {
    // ========================================================================
    // Binary Operations
    // ========================================================================
    // Bmm (batched matrix multiplication) uses the same kernel as MatMul
    // matmul_kernel already supports batched operations via rocblas_*gemm_strided_batched
    // ========================================================================
    // Unary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double exponent = attrs.get_float(AttrKey::Exponent, 2.0);
        return std::vector<Tensor>{rocm::pow_kernel(inputs[0], exponent, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double min_val = attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity());
        double max_val = attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity());
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], min_val, max_val, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double min_val = attrs.get_float(AttrKey::Min, -std::numeric_limits<double>::infinity());
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], min_val, std::numeric_limits<double>::infinity(), get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double max_val = attrs.get_float(AttrKey::Max, std::numeric_limits<double>::infinity());
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], -std::numeric_limits<double>::infinity(), max_val, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Trigonometric Operations
    // ========================================================================
    // ========================================================================
    // Additional Binary Operations
    // ========================================================================
    // In-place arithmetic operations (proper inplace dispatch)
    table.register_inplace_kernel(OpId::AddInplace, [](Tensor& target, std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor& {
        rocm::add_inplace_kernel(target, inputs[0], get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::SubInplace, [](Tensor& target, std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor& {
        rocm::sub_inplace_kernel(target, inputs[0], get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::MulInplace, [](Tensor& target, std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor& {
        rocm::mul_inplace_kernel(target, inputs[0], get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::DivInplace, [](Tensor& target, std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor& {
        rocm::div_inplace_kernel(target, inputs[0], get_hip_stream(attrs));
        return target;
    });

    // ========================================================================
    // Reduction Operations
    // ========================================================================
    // Note: Use INT64_MIN as default to signal "full reduction" (no dim specified)
    // This is distinct from dim=-1 which means "last dimension" in PyTorch convention
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::sum_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::mean_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::max_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::min_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::argmax_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::argmin_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::prod_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::LogSumExp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::logsumexp_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Median, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return rocm::median_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::Mode, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return rocm::mode_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
        return std::vector<Tensor>{rocm::var_kernel(inputs[0], dim, keepdim, unbiased, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
        return std::vector<Tensor>{rocm::std_kernel(inputs[0], dim, keepdim, unbiased, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::norm_kernel(inputs[0], p, dim, keepdim, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Activation Functions
    // ========================================================================
    // In-place activation operations
    // Native in-place activations: Phase 7.1 cleanup. These call the forward
    // kernel with aliased input/output pointers directly into target's storage,
    // avoiding the temporary result + D2D copy + hipStreamSynchronize dance.
    table.register_inplace_kernel(OpId::ReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        rocm::relu_inplace_kernel(target, get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::SigmoidInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        rocm::sigmoid_inplace_kernel(target, get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::TanhInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        rocm::tanh_inplace_kernel(target, get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::LeakyReLUInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        rocm::leaky_relu_inplace_kernel(target, alpha, get_hip_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        rocm::gelu_inplace_kernel(target, get_hip_stream(attrs));
        return target;
    });

    // ========================================================================
    // Softmax Operations
    // ========================================================================
    // ========================================================================
    // Tensor Creation Operations
    // ========================================================================
    table.register_kernel(OpId::Zeros, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::zeros_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Ones, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::ones_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Full, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        // Read as double so Float64 subnormals survive — narrowing to float
        // here would collapse them to zero.
        double value = attrs.get_float(AttrKey::Value, 0.0);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::full_kernel(shape, value, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Rand, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::rand_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Randn, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::randn_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Randint, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t low = attrs.get_int(AttrKey::Start, 0);
        int64_t high = attrs.get_int(AttrKey::End, 0);
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "int32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::randint_kernel(low, high, shape, dtype, device, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Transform Operations
    // ========================================================================
    // ========================================================================
    // BatchNorm Operations
    // ========================================================================
    table.register_kernel(OpId::BatchNorm2dMeanVar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = inputs[0].shape();
        int64_t C = shape[1];
        Tensor mean({C}, inputs[0].dtype(), inputs[0].device());
        Tensor variance({C}, inputs[0].dtype(), inputs[0].device());
        rocm::batchnorm2d_mean_var(inputs[0], mean, variance, get_hip_stream(attrs));
        return std::vector<Tensor>{mean, variance};
    });

    table.register_kernel(OpId::BatchNorm2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return std::vector<Tensor>{rocm::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return std::vector<Tensor>{rocm::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        Tensor running_mean = inputs[0];
        Tensor running_var = inputs[1];
        rocm::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, get_hip_stream(attrs));
        return std::vector<Tensor>{running_mean, running_var};
    });

    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Dispatch convention (matches CUDA cuDNN path and the caller in
        // src/nn/layers/batchnorm.cpp::BatchNorm2dBackward::backward):
        //   inputs = [grad_output, input, gamma, saved_mean, saved_inv_var]
        //
        // The rocm::batchnorm2d_backward function signature is
        //   (grad_output, input, mean, variance, gamma, epsilon, stream)
        // so we must re-order, and convert invstd back to raw variance
        // (variance = 1/invstd^2 - epsilon) since the kernel internally
        // redoes rsqrt(variance + epsilon).
        //
        // Previously this dispatch passed inputs positionally, which
        // silently sent `gamma` where the kernel expected `mean`,
        // `mean` where it expected `variance`, and `invstd` where it
        // expected `gamma` — producing garbage stats and NaN gradients.
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor& grad_output = inputs[0];
        const Tensor& input       = inputs[1];
        const Tensor& gamma       = inputs[2];
        const Tensor& saved_mean  = inputs[3];
        const Tensor& invstd      = inputs[4];

        // variance = 1/invstd^2 - epsilon, computed in the invstd's dtype
        auto one = tenzor::ones_like(invstd);
        auto inv_var = invstd * invstd;                        // 1/(var+eps)
        auto var_plus_eps = one / inv_var;                     // var + eps
        auto eps_t = tenzor::full_like(invstd, epsilon);
        auto variance = var_plus_eps - eps_t;

        auto [grad_input, grad_gamma, grad_beta] = rocm::batchnorm2d_backward(
            grad_output, input, saved_mean, variance, gamma,
            epsilon, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });

    // ========================================================================
    // Convolution Operations
    // ========================================================================
    // Conv2dBackwardInput: inputs = {grad_output, input, weight}
    // Audit F.11: per-axis read with scalar fallback via shared helpers.
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{rocm::conv2d_backward_input(inputs[0], inputs[2], input_shape,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs))};
    });

    // Conv2dBackwardWeight: inputs = {grad_output, input, weight}
    // Audit F.11: per-axis read with scalar fallback via shared helpers.
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{rocm::conv2d_backward_weight(inputs[0], inputs[1], weight_shape,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::conv2d_backward_bias(inputs[0], get_hip_stream(attrs))};
    });

    // Conv1d: wraps Conv2d by unsqueezing height dimension [N,C,L] -> [N,C,1,L].
    // Audit U.4: project scalar Stride/Padding/Dilation onto the W axis
    // only; pin H to neutral (stride=1, padding=0, dilation=1). See the
    // CUDA registry for the full rationale.
    table.register_kernel(OpId::Conv1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto input_4d = inputs[0].unsqueeze(2);
        auto weight_4d = inputs[1].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = inputs.size() > 2
            ? std::vector<Tensor>{input_4d, weight_4d, inputs[2]}
            : std::vector<Tensor>{input_4d, weight_4d};
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dForward, conv2d_inputs, conv2d_attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardInput, conv2d_inputs, conv2d_attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardWeight, conv2d_inputs, conv2d_attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d};
        // U.4: project to per-axis; preserves Stream / other keys for the
        // downstream Conv2dBackwardBias even though it does not consume
        // stride attrs today.
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardBias, conv2d_inputs, conv2d_attrs);
        return {result[0]};
    });

    // Audit F.11: read per-axis StrideD/H/W, PaddingD/H/W, DilationD/H/W
    // with scalar fallback via shared helpers. Anisotropic 3-D conv configs
    // (depth dilation != height dilation, etc.) silently degraded to
    // isotropic before; now routed through stride_3d_vec / padding_3d_vec /
    // dilation_3d_vec which read the per-axis keys with scalar fallback.
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto stride   = ::tenzor::backend::attrs::stride_3d_vec(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_3d_vec(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_3d_vec(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{rocm::conv3d_backward_input_hip(inputs[0], inputs[2], input_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto stride   = ::tenzor::backend::attrs::stride_3d_vec(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_3d_vec(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_3d_vec(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{rocm::conv3d_backward_weight_hip(inputs[0], inputs[1], weight_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::conv3d_backward_bias_hip(inputs[0], get_hip_stream(attrs))};
    });

    // Audit F.11: same per-axis treatment for ConvTranspose3d via shared helpers.
    table.register_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto stride   = ::tenzor::backend::attrs::stride_3d_vec(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_3d_vec(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_3d_vec(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{rocm::conv_transpose3d_backward_input_hip(inputs[0], inputs[2], input_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto stride   = ::tenzor::backend::attrs::stride_3d_vec(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_3d_vec(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_3d_vec(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{rocm::conv_transpose3d_backward_weight_hip(inputs[0], inputs[1], weight_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::conv_transpose3d_backward_bias_hip(inputs[0], get_hip_stream(attrs))};
    });

    // ========================================================================
    // Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        hipStream_t stream = get_hip_stream(attrs);
        // ROCm maxpool2d_forward_hip natively accepts per-axis values.
        auto [output, indices] = rocm::maxpool2d_forward_hip(inputs[0],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            true, stream);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        hipStream_t stream = get_hip_stream(attrs);
        auto [output, indices] = rocm::adaptive_maxpool2d_hip(inputs[0], output_h, output_w,
            true, stream);
        return std::vector<Tensor>{output, indices};
    });

    // ========================================================================
    // 1D Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Q.6: per-axis std::array<int64_t, 1> signature (was scalar W).
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_1d(attrs,
            AttrKey::Stride, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs);
        const auto dilation    = ::tenzor::backend::attrs::dilation_1d(attrs);
        auto [output, indices] = rocm::maxpool1d_forward_hip(inputs[0],
            kernel_size, stride, padding, dilation, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::maxpool1d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.6: per-axis std::array<int64_t, 1> signature.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_1d(attrs,
            AttrKey::Stride, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs);
        return rocm::avgpool1d_forward_hip(inputs[0],
            kernel_size, stride, padding, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.6: per-axis std::array<int64_t, 1> signature.
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_1d(attrs,
            AttrKey::Stride, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs);
        return rocm::avgpool1d_backward_hip(inputs[0], input_shape,
            kernel_size, stride, padding, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::AdaptiveMaxPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
        auto [output, indices] = rocm::adaptive_maxpool1d_forward_hip(inputs[0], output_size, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::adaptive_maxpool1d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
        return rocm::adaptive_avgpool1d_forward_hip(inputs[0], output_size, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::adaptive_avgpool1d_backward_hip(inputs[0], input_shape, get_hip_stream(attrs));
    });

    // ========================================================================
    // 3D Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Q.6: per-axis std::array<int64_t, 3> kernel signature accepts
        // asymmetric kernel/stride/padding across D/H/W.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        auto [output, indices] = rocm::maxpool3d_forward_hip(inputs[0],
            kernel_size, stride, padding, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::maxpool3d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.6: per-axis std::array<int64_t, 3> kernel signature.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        return rocm::avgpool3d_forward_hip(inputs[0],
            kernel_size, stride, padding, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.6: per-axis std::array<int64_t, 3> kernel signature.
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        return rocm::avgpool3d_backward_hip(inputs[0], input_shape,
            kernel_size, stride, padding, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::AdaptiveMaxPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        auto [output, indices] = rocm::adaptive_maxpool3d_forward_hip(inputs[0], output_d, output_h, output_w, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::adaptive_maxpool3d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return rocm::adaptive_avgpool3d_forward_hip(inputs[0], output_d, output_h, output_w, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::adaptive_avgpool3d_backward_hip(inputs[0], input_shape, get_hip_stream(attrs));
    });

    // ========================================================================
    // Indexing Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::SearchSorted, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool right = attrs.get_bool(AttrKey::Right, false);
        return rocm::searchsorted_hip(inputs[0], inputs[1], right, get_hip_stream(attrs));
    });

    // ========================================================================
    // Comparison Operations
    // ========================================================================
    // ========================================================================
    // RNN Operations (LSTM/GRU)
    // ========================================================================
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_hip_stream(attrs);
        // Canonical 7-input contract (matches the CPU backend and the
        // OpId-dispatch callers/tests): {input, hx, cx, w_ih, w_hh, b_ih, b_hh}.
        // Compose the gate pre-activations here, then run the cell kernel.
        // The previous 2-input {gates, c_prev} form read 4*hidden gate
        // offsets out of a [batch, in_size] tensor -> OOB / segfault on F64.
        if (inputs.size() >= 5) {
            const Tensor& input = inputs[0];
            const Tensor& hx    = inputs[1];
            const Tensor& cx    = inputs[2];
            const Tensor& w_ih  = inputs[3];
            const Tensor& w_hh  = inputs[4];
            const int64_t batch_size  = input.shape()[0];
            const int64_t hidden_size = hx.shape()[1];
            // gates = input @ w_ih^T + hx @ w_hh^T + b_ih + b_hh
            Tensor gates = rocm::add_kernel(
                rocm::matmul_kernel(input, rocm::transpose_kernel(w_ih, 0, 1, stream), stream),
                rocm::matmul_kernel(hx, rocm::transpose_kernel(w_hh, 0, 1, stream), stream), stream);
            if (inputs.size() > 5 && inputs[5].numel() > 0)
                gates = rocm::add_kernel(gates, inputs[5], stream);
            if (inputs.size() > 6 && inputs[6].numel() > 0)
                gates = rocm::add_kernel(gates, inputs[6], stream);
            auto [h_out, c_out] = rocm::lstm_cell_forward_kernel(
                gates, cx, batch_size, hidden_size, stream);
            return {h_out, c_out};
        }
        // Legacy 2-input fused form ({gates, c_prev} + size attrs), used by
        // the internal full-sequence path.
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [h_out, c_out] = rocm::lstm_cell_forward_kernel(inputs[0], inputs[1], batch_size, hidden_size, stream);
        return {h_out, c_out};
    });

    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_hip_stream(attrs);
        // Canonical 11-input contract (matches CPU/tests):
        // [d_hy, d_cy, input, hx, cx, hy, cy, w_ih, w_hh, b_ih, b_hh]
        // -> 7 grads [grad_input, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh].
        if (inputs.size() >= 11) {
            const Tensor& d_hy  = inputs[0];
            const Tensor& d_cy  = inputs[1];
            const Tensor& input = inputs[2];
            const Tensor& hx    = inputs[3];
            const Tensor& cx    = inputs[4];
            const Tensor& cy    = inputs[6];
            const Tensor& w_ih  = inputs[7];
            const Tensor& w_hh  = inputs[8];
            const int64_t batch_size  = input.shape()[0];
            const int64_t hidden_size = hx.shape()[1];
            // Recompute the gate pre-activations (cell backward needs them).
            Tensor gates = rocm::add_kernel(
                rocm::matmul_kernel(input, rocm::transpose_kernel(w_ih, 0, 1, stream), stream),
                rocm::matmul_kernel(hx, rocm::transpose_kernel(w_hh, 0, 1, stream), stream), stream);
            if (inputs[9].numel() > 0)  gates = rocm::add_kernel(gates, inputs[9], stream);
            if (inputs[10].numel() > 0) gates = rocm::add_kernel(gates, inputs[10], stream);
            // Cell backward -> grad wrt gate pre-activations and wrt c_prev.
            auto [d_gates, grad_cx] = rocm::lstm_cell_backward_kernel(
                d_hy, d_cy, gates, cx, cy, batch_size, hidden_size, stream);
            // Linear backward through gates = input@w_ih^T + hx@w_hh^T + b.
            Tensor grad_input = rocm::matmul_kernel(d_gates, w_ih, stream);      // [B,in]
            Tensor grad_hx    = rocm::matmul_kernel(d_gates, w_hh, stream);      // [B,H]
            Tensor d_gates_T  = rocm::transpose_kernel(d_gates, 0, 1, stream);   // [4H,B]
            Tensor grad_w_ih  = rocm::matmul_kernel(d_gates_T, input, stream);   // [4H,in]
            Tensor grad_w_hh  = rocm::matmul_kernel(d_gates_T, hx, stream);      // [4H,H]
            // Column-sum over the batch for the bias grads: ones[1,B] @ d_gates.
            Tensor ones = rocm::ones_kernel({1, batch_size}, d_gates.dtype(), d_gates.device(), stream);
            Tensor grad_b = rocm::matmul_kernel(ones, d_gates, stream).reshape({4 * hidden_size});
            return {grad_input, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b, grad_b};
        }
        // Legacy fused 5-input form ({grad_h, grad_c, gates, c_prev, c_out}).
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [grad_gates, grad_c_prev] = rocm::lstm_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, stream);
        return std::vector<Tensor>{grad_gates, grad_c_prev};
    });

    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_hip_stream(attrs);
        // Canonical 6-input contract (matches the CPU backend and the
        // OpId-dispatch callers/tests): {input, hx, w_ih, w_hh, b_ih, b_hh}.
        if (inputs.size() == 6) {
            const Tensor& input = inputs[0];
            const Tensor& hx    = inputs[1];
            const Tensor& w_ih  = inputs[2];
            const Tensor& w_hh  = inputs[3];
            const Tensor& b_ih  = inputs[4];
            const Tensor& b_hh  = inputs[5];
            const int64_t batch_size  = input.shape()[0];
            const int64_t hidden_size = hx.shape()[1];
            // gates_ih = input @ w_ih^T + b_ih  (shape [batch, 3H])
            Tensor gates_ih = rocm::matmul_kernel(
                input, rocm::transpose_kernel(w_ih, 0, 1, stream), stream);
            if (b_ih.numel() > 0) gates_ih = rocm::add_kernel(gates_ih, b_ih, stream);
            Tensor gates_hh = rocm::matmul_kernel(
                hx, rocm::transpose_kernel(w_hh, 0, 1, stream), stream);
            if (b_hh.numel() > 0) gates_hh = rocm::add_kernel(gates_hh, b_hh, stream);
            // Split each [batch, 3H] into reset/update/new chunks of width H.
            auto ih = rocm::split_kernel(gates_ih, hidden_size, /*dim=*/1, stream);
            auto hh = rocm::split_kernel(gates_hh, hidden_size, /*dim=*/1, stream);
            Tensor reset_gates  = rocm::add_kernel(ih[0], hh[0], stream);
            Tensor update_gates = rocm::add_kernel(ih[1], hh[1], stream);
            Tensor new_input    = ih[2];
            Tensor new_hidden   = hh[2];
            return {rocm::gru_cell_forward_kernel(
                reset_gates, update_gates, new_input, new_hidden,
                hx, batch_size, hidden_size, stream)};
        }
        // Legacy 5-input fused form, used by the internal full-sequence path.
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        return {rocm::gru_cell_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, stream)};
    });

    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_hip_stream(attrs);
        // Canonical 7-input contract (matches CPU/tests):
        // [d_hy, input, hx, w_ih, w_hh, b_ih, b_hh]
        // -> 6 grads [grad_input, grad_hx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh].
        if (inputs.size() == 7) {
            const Tensor& d_hy  = inputs[0];
            const Tensor& input = inputs[1];
            const Tensor& hx    = inputs[2];
            const Tensor& w_ih  = inputs[3];
            const Tensor& w_hh  = inputs[4];
            const Tensor& b_ih  = inputs[5];
            const Tensor& b_hh  = inputs[6];
            const int64_t batch_size  = input.shape()[0];
            const int64_t hidden_size = hx.shape()[1];
            // Recompute the gate pre-activations (per ih/hh side).
            Tensor gates_ih = rocm::matmul_kernel(input, rocm::transpose_kernel(w_ih, 0, 1, stream), stream);
            if (b_ih.numel() > 0) gates_ih = rocm::add_kernel(gates_ih, b_ih, stream);
            Tensor gates_hh = rocm::matmul_kernel(hx, rocm::transpose_kernel(w_hh, 0, 1, stream), stream);
            if (b_hh.numel() > 0) gates_hh = rocm::add_kernel(gates_hh, b_hh, stream);
            auto ih = rocm::split_kernel(gates_ih, hidden_size, /*dim=*/1, stream); // [r_i,z_i,n_i]
            auto hh = rocm::split_kernel(gates_hh, hidden_size, /*dim=*/1, stream); // [r_h,z_h,n_h]
            Tensor reset_gates  = rocm::add_kernel(ih[0], hh[0], stream);
            Tensor update_gates = rocm::add_kernel(ih[1], hh[1], stream);
            Tensor new_input    = ih[2];
            Tensor new_hidden   = hh[2];
            auto o = rocm::gru_cell_backward_kernel(
                d_hy, reset_gates, update_gates, new_input, new_hidden, hx,
                batch_size, hidden_size, stream);
            // reset/update gates receive grad on both ih and hh sides;
            // new gate's ih chunk gets grad_new_input, hh chunk grad_new_hidden.
            std::vector<Tensor> ih_parts{o.grad_reset, o.grad_update, o.grad_new_input};
            std::vector<Tensor> hh_parts{o.grad_reset, o.grad_update, o.grad_new_hidden};
            Tensor d_gates_ih = rocm::cat_hip(ih_parts, /*dim=*/1, stream);
            Tensor d_gates_hh = rocm::cat_hip(hh_parts, /*dim=*/1, stream);
            Tensor grad_input = rocm::matmul_kernel(d_gates_ih, w_ih, stream);
            Tensor grad_hx = rocm::add_kernel(rocm::matmul_kernel(d_gates_hh, w_hh, stream),
                                              o.grad_h_prev, stream);
            Tensor grad_w_ih = rocm::matmul_kernel(rocm::transpose_kernel(d_gates_ih, 0, 1, stream), input, stream);
            Tensor grad_w_hh = rocm::matmul_kernel(rocm::transpose_kernel(d_gates_hh, 0, 1, stream), hx, stream);
            Tensor ones = rocm::ones_kernel({1, batch_size}, d_gates_ih.dtype(), d_gates_ih.device(), stream);
            Tensor grad_b_ih = rocm::matmul_kernel(ones, d_gates_ih, stream).reshape({3 * hidden_size});
            Tensor grad_b_hh = rocm::matmul_kernel(ones, d_gates_hh, stream).reshape({3 * hidden_size});
            return {grad_input, grad_hx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh};
        }
        // Legacy fused 6-input form.
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto result = rocm::gru_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            batch_size, hidden_size, stream);
        return std::vector<Tensor>{result.grad_reset, result.grad_update, result.grad_new_input, result.grad_new_hidden, result.grad_h_prev};
    });

    // Full-sequence RNN operations
    // inputs: [input, W_ih, W_hh, bias_ih, bias_hh, h0, c0] for LSTM
    //   This 7-arg signature matches CPU and Vulkan dispatch (see
    //   cpu_kernel_registry + vulkan_kernel_registry). The previous comment
    //   claimed a 6-arg "combined-bias" signature and read inputs[3..5] as
    //   (bias, h0, c0); against the 7-arg caller in src/nn/layers/lstm.cpp
    //   that silently took bias_hh in the `h0` slot and h0 in the `c0` slot,
    //   producing 0.1–0.2 max-abs divergence from CPU in
    //   AllBackends/NNRNNParity.LSTM_Dropout_Eval. The ROCm kernel still
    //   takes a single `bias`, so combine bias_ih+bias_hh up front; if only
    //   one is non-empty use it as-is, and if both are empty pass through.
    // inputs: [input, W_ih, W_hh, bias, h0] for GRU
    table.register_kernel(OpId::LSTMForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor& bias_ih = inputs[3];
        const Tensor& bias_hh = inputs[4];
        Tensor bias_combined;
        if (bias_ih.numel() > 0 && bias_hh.numel() > 0) {
            bias_combined = ::tenzor::add(bias_ih, bias_hh);
        } else if (bias_ih.numel() > 0) {
            bias_combined = bias_ih;
        } else if (bias_hh.numel() > 0) {
            bias_combined = bias_hh;
        } else {
            bias_combined = bias_ih;  // empty
        }
        return rocm::lstm_forward_kernel(inputs[0], inputs[1], inputs[2],
                                         bias_combined, inputs[5], inputs[6],
                                         get_hip_stream(attrs));
    });

    table.register_kernel(OpId::GRUForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, W_ih, W_hh, bias_ih, h0, bias_hh] — biases may be
        // empty tensors if not provided; bias_hh was previously dropped, which
        // broke the reset-gated n-gate term (PyTorch GRU semantics).
        Tensor bias_hh = (inputs.size() > 5) ? inputs[5] : Tensor{};
        return rocm::gru_forward_kernel(inputs[0], inputs[1], inputs[2],
                                        inputs[3], inputs[4], bias_hh,
                                        get_hip_stream(attrs));
    });

    // ========================================================================
    // Additional Transform Operations
    // ========================================================================
    table.register_kernel(OpId::Chunk, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t chunks = attrs.get_int(AttrKey::Chunks, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::chunk_kernel(inputs[0], chunks, dim, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::Split, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t split_size = attrs.get_int(AttrKey::SplitSize, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::split_kernel(inputs[0], split_size, dim, get_hip_stream(attrs));
    });

    // ========================================================================
    // Creation Operations
    // ========================================================================
    table.register_kernel(OpId::Arange, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = attrs.get_float(AttrKey::Start, 0.0);
        double end = attrs.get_float(AttrKey::End, 1.0);
        double step = attrs.get_float(AttrKey::Step, 1.0);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        Device device = device_from_string(attrs.get_string(AttrKey::Device, ""), Device::rocm(0));
        return std::vector<Tensor>{rocm::arange_kernel(start, end, step, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Linspace, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = attrs.get_float(AttrKey::Start, 0.0);
        double end = attrs.get_float(AttrKey::End, 1.0);
        int64_t steps = attrs.get_int(AttrKey::Steps, 100);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        Device device = device_from_string(attrs.get_string(AttrKey::Device, ""), Device::rocm(0));
        return std::vector<Tensor>{rocm::linspace_kernel(start, end, steps, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Eye, []([[maybe_unused]] std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = attrs.get_int(AttrKey::N, 1);
        int64_t m = attrs.get_int(AttrKey::M, -1);
        int64_t k = attrs.get_int(AttrKey::K, 0);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        Device device = device_from_string(attrs.get_string(AttrKey::Device, ""), Device::rocm(0));
        return std::vector<Tensor>{rocm::eye_kernel(n, m, k, dtype, device, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Normalization Operations
    // ========================================================================
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Contract per docs/internals/attention-contract.md: LayerNorm forward
        // must return {output, mean, rstd}. Previously this only returned
        // {output}, forcing the nn layer to dispatch FusedLayerNorm instead —
        // that workaround is removed in src/nn/layers/normalization.cpp.
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        auto [output, mean, rstd] = rocm::layer_norm_kernel_with_stats(
            inputs[0], normalized_shape, weight, bias, eps, get_hip_stream(attrs));
        return std::vector<Tensor>{output, mean, rstd};
    });

    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_output, input, mean, rstd, [weight]
        const Tensor* weight = inputs.size() > 4 ? &inputs[4] : nullptr;
        auto [grad_input, grad_weight, grad_bias] = rocm::layer_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], weight, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::group_norm_forward_with_stats(inputs[0], num_groups, weight, bias, eps, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Accept NumGroups (layer convention) or Groups (functional fallback).
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        const Tensor* weight = inputs.size() > 4 ? &inputs[4] : nullptr;
        auto [grad_input, grad_weight, grad_bias] = rocm::group_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], num_groups, weight, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // nn layer expects {output, saved_mean, saved_rstd}. InstanceNorm is GroupNorm
        // with num_groups = C, so delegate to the stats-returning forward.
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        int64_t C = inputs[0].shape()[1];
        return rocm::group_norm_forward_with_stats(inputs[0], C, weight, bias, eps, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::InstanceNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // nn layer passes [grad_output, input, weight, mean, rstd].
        const Tensor* weight = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        auto [grad_input, grad_weight, grad_bias] = rocm::instance_norm_backward_kernel(
            inputs[0], inputs[1], inputs[3], inputs[4], weight, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    // ========================================================================
    // Fused Operations
    // ========================================================================
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [logits, targets]
        std::string reduction = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
        bool compute_grad = attrs.get_bool(AttrKey::ComputeGrad, false);
        const Tensor& logits = inputs[0];
        const Tensor& targets = inputs[1];

        // The kernels operate on rank-2 (batch, C); flatten rank-3 (N, T, C)
        // logits to (N*T, C) and reshape outputs back.
        const bool is3d = (logits.ndim() == 3);
        int64_t N = 0, T = 0, C = 0;
        Tensor lf = logits, tf = targets;
        if (is3d) {
            N = logits.shape()[0]; T = logits.shape()[1]; C = logits.shape()[2];
            lf = tenzor::reshape(logits.contiguous(), std::vector<int64_t>{N * T, C});
            tf = tenzor::reshape(targets.contiguous(), std::vector<int64_t>{N * T});
        }

        if (compute_grad) {
            // The grad kernel computes in Float32; widen Float16/BFloat16 logits
            // then narrow grad_logits back to the original dtype.
            const DType orig_dtype = logits.dtype();
            const bool is_half = (orig_dtype == DType::Float16 || orig_dtype == DType::BFloat16);
            Tensor lf32 = is_half ? lf.to(DType::Float32) : lf;
            auto [loss, grad_logits] = rocm::fused_softmax_cross_entropy_grad_hip(lf32, tf);
            if (is_half) grad_logits = grad_logits.to(orig_dtype);
            if (is3d) {
                loss = tenzor::reshape(loss, std::vector<int64_t>{N, T});
                grad_logits = tenzor::reshape(grad_logits, std::vector<int64_t>{N, T, C});
            }
            return std::vector<Tensor>{loss, grad_logits};
        }

        auto loss = rocm::fused_softmax_cross_entropy_hip(lf, tf, reduction);
        if (is3d && reduction == "none") {
            loss = tenzor::reshape(loss, std::vector<int64_t>{N, T});
        }
        return std::vector<Tensor>{loss};
    });

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Per docs/internals/attention-contract.md: returns (output, mean, rstd).
        // The kernel-level refit (layer_norm_kernel_with_stats) now writes
        // mean/rstd directly from the HIP shader so the previous host-side
        // mean/var recomputation workaround is gone. Single shader launch,
        // no extra device→host→device roundtrips.
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias   = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        auto [output, mean, rstd] = rocm::layer_norm_kernel_with_stats(
            inputs[0], normalized_shape, weight, bias, eps, get_hip_stream(attrs));
        return std::vector<Tensor>{output, mean, rstd};
    });

    // ========================================================================
    // Fused RMSNorm
    // ========================================================================
    table.register_kernel(OpId::RMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, rrms] = rocm::fused_rms_norm_hip(inputs[0], inputs[1], eps);
        return std::vector<Tensor>{output, rrms};
    });

    table.register_kernel(OpId::FusedRMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, rrms] = rocm::fused_rms_norm_hip(inputs[0], inputs[1], eps);
        return std::vector<Tensor>{output, rrms};
    });

    // ========================================================================
    // Fused Attention — Causal flag now plumbed through (audit C1 ROCm fix).
    // ========================================================================
    table.register_kernel(OpId::FusedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Mirrors CUDA M4-rem: 4D input with H_kv != H_q gets broadcast K/V
        // along the head dim before the 3D collapse. fused_attention_hip
        // assumes 3D [batch_heads, seq, dim] and would reshape-fail or
        // silently wrong otherwise. Per attention-contract.md GQA section.
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        const Tensor& Qi = inputs[0];
        const Tensor& Ki = inputs[1];
        const Tensor& Vi = inputs[2];
        if (Qi.shape().size() == 4) {
            int64_t b = Qi.shape()[0], h = Qi.shape()[1], sq = Qi.shape()[2], d = Qi.shape()[3];
            int64_t h_kv = Ki.shape()[1];
            int64_t sk = Ki.shape()[2];
            int64_t d_v = Vi.shape()[3];
            Tensor Kc = Ki.is_contiguous() ? Ki : Ki.contiguous();
            Tensor Vc = Vi.is_contiguous() ? Vi : Vi.contiguous();
            if (h_kv != h) {
                if (h % h_kv != 0) {
                    throw std::invalid_argument(
                        "FusedAttention ROCm: H_q must be a multiple of H_kv; got " +
                        std::to_string(h) + " and " + std::to_string(h_kv));
                }
                int64_t reps = h / h_kv;
                NewOpAttributes us; us.set(AttrKey::Dim, static_cast<int64_t>(2));
                Tensor Ku = tenzor::dispatch(OpId::Unsqueeze, std::vector<Tensor>{Kc}, us)[0];
                Tensor Vu = tenzor::dispatch(OpId::Unsqueeze, std::vector<Tensor>{Vc}, us)[0];
                std::vector<int64_t> exp_k = {b, h_kv, reps, sk, d};
                std::vector<int64_t> exp_v = {b, h_kv, reps, sk, d_v};
                std::string s_k, s_v;
                for (size_t i = 0; i < exp_k.size(); ++i) { if (i) s_k += ","; s_k += std::to_string(exp_k[i]); }
                for (size_t i = 0; i < exp_v.size(); ++i) { if (i) s_v += ","; s_v += std::to_string(exp_v[i]); }
                NewOpAttributes ek; ek.set(AttrKey::Shape, s_k);
                NewOpAttributes ev; ev.set(AttrKey::Shape, s_v);
                Tensor Ke = tenzor::dispatch(OpId::Expand, std::vector<Tensor>{Ku}, ek)[0];
                Tensor Ve = tenzor::dispatch(OpId::Expand, std::vector<Tensor>{Vu}, ev)[0];
                Kc = Ke.contiguous().reshape({b, h, sk, d});
                Vc = Ve.contiguous().reshape({b, h, sk, d_v});
            }
            Tensor Q3 = (Qi.is_contiguous() ? Qi : Qi.contiguous()).reshape({b * h, sq, d});
            Tensor K3 = Kc.reshape({b * h, sk, d});
            Tensor V3 = Vc.reshape({b * h, sk, d_v});
            auto [out3, lse3] = rocm::fused_attention_hip(Q3, K3, V3, scale, causal, 0.0f, 0u);
            return std::vector<Tensor>{out3.reshape({b, h, sq, d_v}), lse3};
        }
        auto [output, lse] = rocm::fused_attention_hip(Qi, Ki, Vi, scale, causal, 0.0f, 0u);
        return std::vector<Tensor>{output, lse};
    });

    // ========================================================================
    // Flash Attention (memory-efficient tiled attention) — returns 4-tuple
    // [output, lse, philox_seed, philox_offset] per attention-contract.md.
    // Dropout > 0 with is_training currently throws; full Philox replay arrives
    // with a kernel-level refit. Causal is honored (audit C1 fix).
    // ========================================================================
    table.register_kernel(OpId::FlashAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Per docs/internals/attention-contract.md: returns 4-tuple
        // [output, lse, philox_seed, philox_offset]. Causal + dropout both
        // honored kernel-side (M5-rem fix; was throwing for dropout > 0).
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        float dropout_p = static_cast<float>(attrs.get_float(AttrKey::DropoutP, 0.0));
        bool is_training = attrs.get_bool(AttrKey::IsTraining, attrs.get_bool(AttrKey::Training, false));
        bool apply_dropout = dropout_p > 0.0f && is_training;

        uint32_t rng_seed = 0u;
        if (apply_dropout) {
            int64_t seed_in = attrs.get_int(AttrKey::Seed, 0);
            uint64_t seed64 = (seed_in != 0)
                ? static_cast<uint64_t>(seed_in)
                : (reinterpret_cast<uintptr_t>(inputs[0].data_ptr()) * 2654435761ULL);
            rng_seed = static_cast<uint32_t>(seed64);
            if (rng_seed == 0) rng_seed = 1u;
        }
        // fused_attention_hip expects 3D `(batch_heads, seq_len, head_dim)`.
        // The autograd-side dispatch passes Q/K/V 4D `(B, H, S, D)`; collapse
        // leading dims for the kernel call and restore on output.
        bool is_4d = (inputs[0].shape().size() == 4);
        Tensor Qi = inputs[0], Ki = inputs[1], Vi = inputs[2];
        std::vector<int64_t> orig_q_shape;
        if (is_4d) {
            orig_q_shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            int64_t b = orig_q_shape[0], h = orig_q_shape[1];
            int64_t sq = orig_q_shape[2], d = orig_q_shape[3];
            int64_t sk = inputs[1].shape()[2];
            int64_t dv = inputs[2].shape()[3];
            Qi = tenzor::reshape(inputs[0].contiguous(), std::vector<int64_t>{b * h, sq, d});
            Ki = tenzor::reshape(inputs[1].contiguous(), std::vector<int64_t>{b * h, sk, d});
            Vi = tenzor::reshape(inputs[2].contiguous(), std::vector<int64_t>{b * h, sk, dv});
        }

        // Audit A.11 — native Float64 path. The mainline kernel upcasts to
        // Float32, which is fatal for Float64 gradcheck (forward halves
        // precision; analytical-vs-numerical diverges beyond 1e-7 tol). Route
        // to the native FP64 kernel for supported head_dims; for other
        // head_dims fall through to a composed-ops path that stays in FP64
        // via the standard op dispatch (every op dispatches to a ROCm double-
        // precision kernel — no FP32 round-trip, no CPU fallback). Dropout is
        // rejected for Float64 (gradcheck-only; dropout breaks determinism).
        Tensor output, lse;
        int64_t head_dim_q = Qi.shape().back();
        bool f64_native_supported = (head_dim_q == 16 || head_dim_q == 32 ||
                                     head_dim_q == 48 || head_dim_q == 64 ||
                                     head_dim_q == 80 || head_dim_q == 96 ||
                                     head_dim_q == 128);
        if (Qi.dtype() == DType::Float64) {
            if (apply_dropout) {
                throw std::runtime_error(
                    "FlashAttention ROCm: dropout not supported for Float64 "
                    "(gradcheck-only path; dropout is incompatible with "
                    "deterministic gradcheck).");
            }
            if (f64_native_supported) {
                auto pair = rocm::fused_attention_hip_f64(
                    Qi, Ki, Vi, static_cast<double>(scale), causal,
                    get_hip_stream(attrs));
                output = pair.first;
                lse    = pair.second;
            } else {
                // FP64 composed-ops fallback for unsupported head_dim.
                // Mirrors the CUDA path (cuda_kernel_registry.cpp A.11):
                // bmm + softmax dispatch to native ROCm FP64 kernels.
                Tensor Kt = tenzor::transpose(Ki, -1, -2);
                Tensor scores = tenzor::bmm(Qi, Kt);
                auto scores_shape = std::vector<int64_t>(scores.shape().begin(),
                                                         scores.shape().end());
                Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                              scores.dtype(), scores.device());
                scores = tenzor::mul(scores, scale_t);
                if (causal) {
                    int64_t S_q = scores_shape[scores_shape.size() - 2];
                    int64_t S_k = scores_shape[scores_shape.size() - 1];
                    Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Qi.device());
                    Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Qi.device());
                    rows = tenzor::reshape(rows, {S_q, 1});
                    cols = tenzor::reshape(cols, {1, S_k});
                    Tensor mask = tenzor::gt(cols.to(DType::Float64),
                                              rows.to(DType::Float64));
                    Tensor neg_inf = tenzor::full(scores_shape,
                        -std::numeric_limits<double>::infinity(),
                        scores.dtype(), scores.device());
                    scores = tenzor::add(scores,
                                          tenzor::mul(mask.to(scores.dtype()), neg_inf));
                }
                NewOpAttributes sm_attrs;
                sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
                std::vector<Tensor> sm_in = {scores};
                Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
                output = tenzor::bmm(probs, Vi);
                // LSE is Float32 per the contract; build a zero stub for
                // consistency with the contract (backward uses composed path).
                lse = tenzor::zeros({Qi.shape()[0], Qi.shape()[1]},
                                     DType::Float32, Qi.device());
            }
        } else {
            auto pair = rocm::fused_attention_hip(
                Qi, Ki, Vi, scale, causal,
                apply_dropout ? dropout_p : 0.0f, rng_seed);
            output = pair.first;
            lse    = pair.second;
        }

        if (is_4d) {
            int64_t b = orig_q_shape[0], h = orig_q_shape[1];
            int64_t sq = orig_q_shape[2], dv = inputs[2].shape()[3];
            if (output.is_valid()) {
                output = tenzor::reshape(output, std::vector<int64_t>{b, h, sq, dv});
            }
            if (lse.is_valid() && lse.numel() > 0) {
                lse = tenzor::reshape(lse, std::vector<int64_t>{b, h, sq});
            }
        }

        Tensor seed_t, offset_t;
        if (apply_dropout) {
            seed_t = tenzor::zeros({1}, DType::Int64, inputs[0].device());
            offset_t = tenzor::zeros({1}, DType::Int64, inputs[0].device());
            int64_t seed_host = static_cast<int64_t>(rng_seed);
            int64_t offset_host = 0;
            hipMemcpy(seed_t.data_ptr(), &seed_host, sizeof(int64_t), hipMemcpyHostToDevice);
            hipMemcpy(offset_t.data_ptr(), &offset_host, sizeof(int64_t), hipMemcpyHostToDevice);
        }
        return std::vector<Tensor>{output, lse, seed_t, offset_t};
    });

    // ========================================================================
    // Flash Attention Backward (fused tiled kernel, composed-ops fallback)
    // ========================================================================
    table.register_kernel(OpId::FlashAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            // inputs: [dO, Q, K, V, O, L] — L = logsumexp from forward
            // Falls back to [dO, Q, K, V, O] (no L) for composed-ops path
            float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
            bool causal = attrs.get_bool(AttrKey::Causal, false);

            const Tensor& dO = inputs[0];
            const Tensor& Q = inputs[1];
            const Tensor& K = inputs[2];
            const Tensor& V = inputs[3];
            const Tensor& O = inputs[4];

            // Check if we have logsumexp (L) and supported head_dim for fused kernel
            int64_t head_dim = Q.shape().back();
            bool has_lse = inputs.size() >= 6;
            bool fused_supported = (head_dim == 32 || head_dim == 64 || head_dim == 128);

            if (has_lse && fused_supported && Q.dtype() == DType::Float32) {
                const Tensor& L = inputs[5];
                return rocm::flash_attention_backward_hip(dO, Q, K, V, O, L, scale, causal);
            }

            // Audit A.11 — native Float64 backward. Recomputes per-row LSE
            // in double from Q/K, so it does NOT depend on the saved L (which
            // is Float32 anyway per the attention contract). Kernel expects
            // 3D [BH, S, D] inputs; collapse leading B,H dims if 4D.
            bool f64_native_supported = (head_dim == 16 || head_dim == 32 ||
                                         head_dim == 48 || head_dim == 64 ||
                                         head_dim == 80 || head_dim == 96 ||
                                         head_dim == 128);
            if (Q.dtype() == DType::Float64 && f64_native_supported) {
                bool is_4d_f64 = (Q.shape().size() == 4);
                std::vector<int64_t> q_shape_4d, k_shape_4d, v_shape_4d;
                Tensor Q3 = Q, K3 = K, V3 = V, O3 = O, dO3 = dO;
                if (is_4d_f64) {
                    q_shape_4d.assign(Q.shape().begin(), Q.shape().end());
                    k_shape_4d.assign(K.shape().begin(), K.shape().end());
                    v_shape_4d.assign(V.shape().begin(), V.shape().end());
                    int64_t b = q_shape_4d[0], h = q_shape_4d[1];
                    int64_t sq = q_shape_4d[2], dq = q_shape_4d[3];
                    int64_t sk = k_shape_4d[2], dk = k_shape_4d[3];
                    int64_t dv = v_shape_4d[3];
                    Q3  = tenzor::reshape(Q.contiguous(),  std::vector<int64_t>{b * h, sq, dq});
                    K3  = tenzor::reshape(K.contiguous(),  std::vector<int64_t>{b * h, sk, dk});
                    V3  = tenzor::reshape(V.contiguous(),  std::vector<int64_t>{b * h, sk, dv});
                    O3  = tenzor::reshape(O.contiguous(),  std::vector<int64_t>{b * h, sq, dv});
                    dO3 = tenzor::reshape(dO.contiguous(), std::vector<int64_t>{b * h, sq, dv});
                }
                auto grads = rocm::flash_attention_backward_hip_f64(
                    dO3, Q3, K3, V3, O3, static_cast<double>(scale), causal,
                    get_hip_stream(attrs));
                if (is_4d_f64) {
                    grads[0] = tenzor::reshape(grads[0], q_shape_4d);
                    grads[1] = tenzor::reshape(grads[1], k_shape_4d);
                    grads[2] = tenzor::reshape(grads[2], v_shape_4d);
                }
                return grads;
            }

            // Composed-ops fallback for unsupported head_dim or missing L.
            // ROCm bmm expects 3D inputs; collapse leading B,H dims and
            // restore on output, mirroring the forward dispatch path.
            bool is_4d_bw = Q.is_valid() && (Q.shape().size() == 4);
            std::vector<int64_t> q_shape4d, k_shape4d, v_shape4d;
            Tensor Q3 = Q, K3 = K, V3 = V, dO3 = dO;
            if (is_4d_bw) {
                q_shape4d.assign(Q.shape().begin(), Q.shape().end());
                k_shape4d.assign(K.shape().begin(), K.shape().end());
                v_shape4d.assign(V.shape().begin(), V.shape().end());
                int64_t b = q_shape4d[0], h = q_shape4d[1];
                int64_t sq = q_shape4d[2], dq = q_shape4d[3];
                int64_t sk = k_shape4d[2], dk = k_shape4d[3];
                int64_t dv = v_shape4d[3];
                Q3 = tenzor::reshape(Q.contiguous(), std::vector<int64_t>{b * h, sq, dq});
                K3 = tenzor::reshape(K.contiguous(), std::vector<int64_t>{b * h, sk, dk});
                V3 = tenzor::reshape(V.contiguous(), std::vector<int64_t>{b * h, sk, dv});
                dO3 = tenzor::reshape(dO.contiguous(), std::vector<int64_t>{b * h, sq, dv});
            }

            Tensor Kt = tenzor::transpose(K3, -1, -2);
            Tensor scores = tenzor::bmm(Q3, Kt);

            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = tenzor::mul(scores, scale_t);

            if (causal) {
                // Per attention-contract.md sentinel rule: -INFINITY, not -1e9
                // (FP16 saturates the latter to -65504, leaks gradient mass
                // through softmax — Systemic #3).
                int64_t seq_len = scores_shape[scores_shape.size() - 1];
                Tensor rows = tenzor::arange(0, seq_len, 1, DType::Int64, scores.device());
                Tensor cols = tenzor::arange(0, seq_len, 1, DType::Int64, scores.device());
                rows = tenzor::reshape(rows, {seq_len, 1});
                cols = tenzor::reshape(cols, {1, seq_len});
                Tensor causal_mask = tenzor::gt(cols.to(DType::Float32), rows.to(DType::Float32));
                Tensor neg_inf = tenzor::full(scores_shape,
                                              -std::numeric_limits<float>::infinity(),
                                              scores.dtype(), scores.device());
                scores = tenzor::add(scores, tenzor::mul(causal_mask.to(scores.dtype()), neg_inf));
            }

            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_inputs = {scores};
            Tensor attn_weights = tenzor::dispatch(OpId::Softmax, sm_inputs, sm_attrs)[0];

            Tensor attn_t = tenzor::transpose(attn_weights, -1, -2);
            Tensor dV = tenzor::bmm(attn_t, dO3);

            Tensor Vt = tenzor::transpose(V3, -1, -2);
            Tensor dAttn = tenzor::bmm(dO3, Vt);

            Tensor attn_dAttn = tenzor::mul(attn_weights, dAttn);
            NewOpAttributes sum_attrs;
            sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            sum_attrs.set(AttrKey::Keepdim, true);
            std::vector<Tensor> sum_inputs = {attn_dAttn};
            Tensor sum_ad = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
            Tensor dScores = tenzor::mul(attn_weights, tenzor::sub(dAttn, sum_ad));

            Tensor scale_t2 = tenzor::full(
                std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
                static_cast<double>(scale), dScores.dtype(), dScores.device());
            dScores = tenzor::mul(dScores, scale_t2);

            Tensor dQ = tenzor::bmm(dScores, K3);
            Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
            Tensor dK = tenzor::bmm(dScores_t, Q3);

            if (is_4d_bw) {
                dQ = tenzor::reshape(dQ, q_shape4d);
                dK = tenzor::reshape(dK, k_shape4d);
                dV = tenzor::reshape(dV, v_shape4d);
            }

            return {dQ, dK, dV};
        });

    // ========================================================================
    // FlexAttention (built-in score_mod registry; M8 lands the native path)
    // ========================================================================
    // Per docs/internals/attention-contract.md, ScoreModId 0=identity and
    // 1=causal reduce to FusedAttention; 2 (sliding window) and >=3 (user-
    // registered score_mods) are served via a composed bmm+softmax path below.
    // Composed-path limitation: the user score_mod is invoked once over the full
    // [B,H,Sq,Sk] score tensor with b=h=q_start=kv_start=0, so score_mods that
    // index by batch/head/position are not yet honoured (the native M8 kernel
    // will pass true tile indices).
    table.register_kernel(OpId::FlexAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

        if (score_mod_id == 0 || score_mod_id == 1) {
            bool causal = (score_mod_id == 1);
            auto [output, lse] = rocm::fused_attention_hip(inputs[0], inputs[1], inputs[2], scale, causal);
            return std::vector<Tensor>{output, lse};
        }

        if (score_mod_id == 2) {
            int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
            if (window_size <= 0) {
                throw std::invalid_argument(
                    "FlexAttention ROCm: ScoreModId=2 requires AttrKey::WindowSize > 0.");
            }
            const Tensor& Q = inputs[0]; const Tensor& K = inputs[1]; const Tensor& V = inputs[2];
            int64_t S_q = Q.shape()[Q.shape().size() - 2];
            int64_t S_k = K.shape()[K.shape().size() - 2];
            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::bmm(Q, Kt);
            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = scores * scale_t;
            int64_t half = window_size / 2;
            Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Q.device());
            Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Q.device());
            Tensor rows_2d = tenzor::reshape(rows.to(DType::Float32), std::vector<int64_t>{S_q, 1});
            Tensor cols_2d = tenzor::reshape(cols.to(DType::Float32), std::vector<int64_t>{1, S_k});
            Tensor abs_diff = tenzor::abs(tenzor::sub(rows_2d, cols_2d));
            Tensor half_t = tenzor::full({1}, static_cast<double>(half),
                                          abs_diff.dtype(), abs_diff.device());
            Tensor outside = tenzor::gt(abs_diff, half_t);
            Tensor neg_inf = tenzor::full(scores_shape,
                -std::numeric_limits<float>::infinity(),
                scores.dtype(), scores.device());
            scores = scores + (outside.to(scores.dtype()) * neg_inf);
            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_in = {scores};
            Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
            Tensor output = tenzor::bmm(probs, V);
            // LSE (Float32 per attention_contract.hpp) — composed from
            // the same scores softmax just normalised so the saved
            // tensor matches what FlexAttentionBackward needs.
            NewOpAttributes lse_attrs;
            lse_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            lse_attrs.set(AttrKey::Keepdim, false);
            Tensor scores_f32 = (scores.dtype() == DType::Float32)
                                ? scores : scores.to(DType::Float32);
            std::vector<Tensor> lse_in = {scores_f32};
            Tensor lse = tenzor::dispatch(OpId::LogSumExp, lse_in, lse_attrs)[0];
            return std::vector<Tensor>{output, lse};
        }

        // Wave C: ScoreModId >= 3 routes through the process-wide score_mod
        // registry populated by `tenzor::nn::register_score_mod`. Forward
        // composes Q@K^T → user functor → softmax → @V via tenzor:: ops
        // (which dispatch to ROCm automatically since Q/K/V live on ROCm).
        if (score_mod_id >= 3) {
            auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
            if (!fn) {
                throw std::runtime_error(
                    "FlexAttention ROCm: no user score_mod registered for ScoreModId=" +
                    std::to_string(score_mod_id) +
                    ". Register via tenzor::nn::register_score_mod(id, fn) before dispatch.");
            }
            const Tensor& Q = inputs[0]; const Tensor& K = inputs[1]; const Tensor& V = inputs[2];
            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::bmm(Q, Kt);
            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = scores * scale_t;
            Tensor modified = fn(scores, /*b=*/0, /*h=*/0, /*q_start=*/0, /*kv_start=*/0);
            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_in = {modified};
            Tensor probs = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];
            Tensor output = tenzor::bmm(probs, V);
            // LSE (Float32 per attention_contract.hpp) — composed from
            // the same scores softmax just normalised so the saved
            // tensor matches what FlexAttentionBackward needs.
            NewOpAttributes lse_attrs;
            lse_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            lse_attrs.set(AttrKey::Keepdim, false);
            Tensor scores_f32 = (scores.dtype() == DType::Float32)
                                ? scores : scores.to(DType::Float32);
            std::vector<Tensor> lse_in = {scores_f32};
            Tensor lse = tenzor::dispatch(OpId::LogSumExp, lse_in, lse_attrs)[0];
            return std::vector<Tensor>{output, lse};
        }

        throw std::runtime_error(
            "FlexAttention ROCm: ScoreModId=" + std::to_string(score_mod_id) +
            " not recognised (built-ins: 0=identity, 1=causal, 2=sliding_window; "
            "register user IDs >= 3 via tenzor::nn::register_score_mod).");
    });

    table.register_kernel(OpId::FlexAttentionBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

        // ScoreModId 0/1: route to fused FlashAttention backward.
        if (score_mod_id == 0 || score_mod_id == 1) {
            bool causal = (score_mod_id == 1);
            OpAttributes bwd_attrs;
            bwd_attrs.set(AttrKey::Scale, static_cast<double>(scale));
            bwd_attrs.set(AttrKey::Causal, causal);
            std::vector<Tensor> bwd_inputs(inputs.begin(), inputs.end());
            return tenzor::dispatch(OpId::FlashAttentionBackward, bwd_inputs, bwd_attrs);
        }

        // Wave C: ScoreModId == 2 (sliding window) or >= 3 (user functor) — composed backward.
        if (score_mod_id == 2 || score_mod_id >= 3) {
            const Tensor& dO = inputs[0];
            const Tensor& Q  = inputs[1];
            const Tensor& K  = inputs[2];
            const Tensor& V  = inputs[3];
            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::bmm(Q, Kt);
            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = scores * scale_t;

            if (score_mod_id == 2) {
                int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
                if (window_size <= 0) {
                    throw std::invalid_argument(
                        "FlexAttentionBackward ROCm: ScoreModId=2 requires AttrKey::WindowSize > 0.");
                }
                int64_t S_q = Q.shape()[Q.shape().size() - 2];
                int64_t S_k = K.shape()[K.shape().size() - 2];
                int64_t half = window_size / 2;
                Tensor rows = tenzor::arange(0, S_q, 1, DType::Int64, Q.device());
                Tensor cols = tenzor::arange(0, S_k, 1, DType::Int64, Q.device());
                Tensor rows_2d = tenzor::reshape(rows.to(DType::Float32), std::vector<int64_t>{S_q, 1});
                Tensor cols_2d = tenzor::reshape(cols.to(DType::Float32), std::vector<int64_t>{1, S_k});
                Tensor abs_diff = tenzor::abs(tenzor::sub(rows_2d, cols_2d));
                Tensor half_t = tenzor::full({1}, static_cast<double>(half),
                                              abs_diff.dtype(), abs_diff.device());
                Tensor outside = tenzor::gt(abs_diff, half_t);
                // Use large finite negative; softmax(-1e30) underflows to 0
                // (same effect as -inf) without producing NaN at in-window
                // positions via 0 * -inf.
                Tensor large_neg = tenzor::full(scores_shape, -1.0e30,
                                                 scores.dtype(), scores.device());
                scores = scores + (outside.to(scores.dtype()) * large_neg);
            } else {
                auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
                if (!fn) {
                    throw std::runtime_error(
                        "FlexAttentionBackward ROCm: no user score_mod registered for ScoreModId=" +
                        std::to_string(score_mod_id));
                }
                scores = fn(scores, 0, 0, 0, 0);
            }

            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_in = {scores};
            Tensor attn = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];

            Tensor attn_t = tenzor::transpose(attn, -1, -2);
            Tensor dV = tenzor::bmm(attn_t, dO);

            Tensor Vt = tenzor::transpose(V, -1, -2);
            Tensor dAttn = tenzor::bmm(dO, Vt);

            Tensor ad = tenzor::mul(attn, dAttn);
            NewOpAttributes sum_attrs;
            sum_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            sum_attrs.set(AttrKey::Keepdim, true);
            std::vector<Tensor> sum_inputs = {ad};
            Tensor sum_ad = tenzor::dispatch(OpId::Sum, sum_inputs, sum_attrs)[0];
            Tensor dScores = tenzor::mul(attn, tenzor::sub(dAttn, sum_ad));

            Tensor scale_t2 = tenzor::full(
                std::vector<int64_t>(dScores.shape().begin(), dScores.shape().end()),
                static_cast<double>(scale), dScores.dtype(), dScores.device());
            dScores = tenzor::mul(dScores, scale_t2);

            Tensor dQ = tenzor::bmm(dScores, K);
            Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
            Tensor dK = tenzor::bmm(dScores_t, Q);

            return {dQ, dK, dV};
        }

        throw std::runtime_error(
            "FlexAttentionBackward ROCm: ScoreModId=" + std::to_string(score_mod_id) +
            " not recognised.");
    });

    // ========================================================================
    // Einsum (composed — delegates to einsum_composed to avoid dispatch loop)
    // ========================================================================
    table.register_kernel(OpId::Einsum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto equation = std::string(attrs.get_string(AttrKey::EinsumEquation, ""));
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return {einsum_composed(equation, tensors)};
    });

    // ========================================================================
    // Fused LayerNorm Backward
    // ========================================================================
    table.register_kernel(OpId::FusedLayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        auto [grad_input, grad_weight, grad_bias] = rocm::fused_layer_norm_backward_hip(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    // ========================================================================
    // Fused Conv2D + BatchNorm + ReLU (full pipeline)
    // ========================================================================
    table.register_single_output_kernel(OpId::FusedConv2dBnReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride  = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding = ::tenzor::backend::attrs::padding_2d(attrs);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        // Phase 2.1: fused_conv2d_bn_relu_full_hip kernel is scalar-only; reject asymmetric.
        if (stride[0] != stride[1] || padding[0] != padding[1]) {
            throw std::invalid_argument(
                "FusedConv2dBnReLU (ROCm): backend kernel only supports symmetric stride/padding; "
                "got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]));
        }
        const Tensor* bias = inputs.size() > 2 && inputs[2].numel() > 0 ? &inputs[2] : nullptr;
        return rocm::fused_conv2d_bn_relu_full_hip(inputs[0], inputs[1], bias,
            inputs[5], inputs[6], inputs[3], inputs[4], stride[0], padding[0], eps);
    });

    // ========================================================================
    // Fused SGD Optimizer Step
    // ========================================================================
    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        float dampening = static_cast<float>(attrs.get_float(AttrKey::Dampening, 0.0));
        bool nesterov = attrs.get_bool(AttrKey::Nesterov, false);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor* momentum_buffer = (inputs.size() > 2 && momentum > 0.0f)
            ? &const_cast<Tensor&>(inputs[2]) : nullptr;

        rocm::fused_sgd_step_hip(param, inputs[1], momentum_buffer,
            lr, momentum, weight_decay, dampening, nesterov, get_hip_stream(attrs));
        return std::vector<Tensor>{param};
    });

    // ========================================================================
    // Fused Adam Optimizer Step
    // ========================================================================
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

        rocm::fused_adam_step_hip(param, inputs[1], exp_avg, exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, decoupled,
            get_hip_stream(attrs), max_exp_avg_sq, amsgrad);
        return std::vector<Tensor>{param};
    });

    // ========================================================================
    // Fused RMSProp Optimizer Step
    // ========================================================================
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

        rocm::fused_rmsprop_step_hip(param, inputs[1], square_avg, grad_avg, momentum_buffer,
            lr, alpha, eps, weight_decay, momentum, centered, get_hip_stream(attrs));
        return std::vector<Tensor>{param};
    });

    // ========================================================================
    // Fused Adadelta Optimizer Step
    // ========================================================================
    table.register_kernel(OpId::FusedAdadeltaStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& square_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& acc_delta = const_cast<Tensor&>(inputs[3]);

        rocm::fused_adadelta_step_hip(param, inputs[1], square_avg, acc_delta,
            rho, eps, lr, weight_decay, get_hip_stream(attrs));
        return std::vector<Tensor>{param};
    });

    // ========================================================================
    // Fused Adagrad Optimizer Step
    // ========================================================================
    table.register_kernel(OpId::FusedAdagradStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float lr_decay = static_cast<float>(attrs.get_float(AttrKey::LrDecay, 0.0));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        int64_t step = attrs.get_int(AttrKey::Step, 1);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& sum_sq = const_cast<Tensor&>(inputs[2]);

        rocm::fused_adagrad_step_hip(param, inputs[1], sum_sq,
            lr, lr_decay, eps, weight_decay, step, get_hip_stream(attrs));
        return std::vector<Tensor>{param};
    });

    // ========================================================================
    // Fused Adam-Atan2 Optimizer Step
    // ========================================================================
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

        rocm::fused_adam_atan2_step_hip(param, inputs[1], exp_avg, exp_avg_sq, max_exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, amsgrad, get_hip_stream(attrs));
        return std::vector<Tensor>{param};
    });

    // ========================================================================
    // Multi-Layer LSTM Forward
    // ========================================================================
    table.register_kernel(OpId::LSTMMultiLayerForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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

        return rocm::lstm_multi_layer_forward_kernel(input, W_ih_list, W_hh_list, bias_list, h0, c0, get_hip_stream(attrs));
    });

    // ========================================================================
    // Multi-Layer GRU Forward
    // ========================================================================
    table.register_kernel(OpId::GRUMultiLayerForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_layers = attrs.get_int(AttrKey::NumLayers, 1);
        const Tensor& input = inputs[0];
        const Tensor& h0 = inputs[1];

        // Fixed layout shared by every backend (cpu/cuda) and the canonical
        // {x, h0, then per-layer W_ih, W_hh, bias} packing: stride 3, no
        // hidden-hidden bias. Deriving stride from (inputs.size()-2)/num_layers
        // risked a silent mis-slice (integer division) if a stray tensor were
        // ever appended; the fixed stride matches the other backends exactly.
        constexpr size_t stride = 3;

        std::vector<Tensor> W_ih_list, W_hh_list, bias_list, bias_hh_list;
        for (int64_t l = 0; l < num_layers; ++l) {
            size_t base_idx = 2 + static_cast<size_t>(l) * stride;
            W_ih_list.push_back(inputs[base_idx]);
            W_hh_list.push_back(inputs[base_idx + 1]);
            bias_list.push_back(inputs[base_idx + 2]);
        }

        return rocm::gru_multi_layer_forward_kernel(input, W_ih_list, W_hh_list, bias_list, bias_hh_list, h0, get_hip_stream(attrs));
    });

    // ========================================================================
    // Bidirectional LSTM Forward
    // ========================================================================
    table.register_kernel(OpId::BiLSTMForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return rocm::bilstm_forward_kernel(
            inputs[0],
            inputs[3], inputs[4], inputs[5], inputs[6],
            inputs[7], inputs[8], inputs[9], inputs[10],
            inputs[1], inputs[2],
            get_hip_stream(attrs)
        );
    });

    // ========================================================================
    // Embedding Operations
    // ========================================================================
    table.register_kernel(OpId::GatherRelativePositionBias,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
            int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
            hipStream_t stream = get_hip_stream(attrs);
            return {rocm::gather_relative_position_bias_kernel(inputs[0], inputs[1], num_positions, num_heads, stream)};
        });

    // ========================================================================
    // Linear Operations
    // ========================================================================
    table.register_kernel(OpId::LinearBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_output, input, weight
        return rocm::linear_backward_kernel(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
    });

    // ========================================================================
    // Dropout Operations
    // ========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        bool training = attrs.get_bool(AttrKey::Training, true);
        auto [output, mask] = rocm::dropout_kernel(inputs[0], p, training, get_hip_stream(attrs));
        return std::vector<Tensor>{output, mask};
    });

    // ========================================================================
    // Vision Operations
    // ========================================================================
    // =========================================================================
    // FFT Operations (rocFFT or native HIP Cooley-Tukey/Bluestein fallback)
    // =========================================================================
    table.register_single_output_kernel(OpId::FFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_fft_kernel(inputs[0], dim, n, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::IFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_ifft_kernel(inputs[0], dim, n, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::RFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_rfft_kernel(inputs[0], dim, n, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::IRFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, 2 * (inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim] - 1));
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_irfft_kernel(inputs[0], dim, n, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ndim = inputs[0].ndim();
        std::vector<int64_t> dims = {ndim - 2, ndim - 1};
        std::vector<int64_t> n_vec = {
            inputs[0].shape()[dims[0]],
            inputs[0].shape()[dims[1]]
        };
        auto attr_n = attrs.get_int_list(AttrKey::Shape);
        if (!attr_n.empty() && attr_n.size() >= 2) {
            n_vec[0] = attr_n[0];
            n_vec[1] = attr_n[1];
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_fft2_kernel(inputs[0], dims, n_vec, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::IFFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ndim = inputs[0].ndim();
        std::vector<int64_t> dims = {ndim - 2, ndim - 1};
        std::vector<int64_t> n_vec = {
            inputs[0].shape()[dims[0]],
            inputs[0].shape()[dims[1]]
        };
        auto attr_n = attrs.get_int_list(AttrKey::Shape);
        if (!attr_n.empty() && attr_n.size() >= 2) {
            n_vec[0] = attr_n[0];
            n_vec[1] = attr_n[1];
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_ifft2_kernel(inputs[0], dims, n_vec, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ndim = inputs[0].ndim();
        std::vector<int64_t> dims(ndim);
        for (int64_t i = 0; i < ndim; ++i) dims[i] = i;
        std::vector<int64_t> n_vec(ndim);
        for (int64_t i = 0; i < ndim; ++i) n_vec[i] = inputs[0].shape()[i];
        auto attr_n = attrs.get_int_list(AttrKey::Shape);
        if (!attr_n.empty()) {
            for (size_t i = 0; i < attr_n.size() && i < n_vec.size(); ++i) {
                n_vec[i] = attr_n[i];
            }
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_fftn_kernel(inputs[0], dims, n_vec, norm, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::IFFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ndim = inputs[0].ndim();
        std::vector<int64_t> dims(ndim);
        for (int64_t i = 0; i < ndim; ++i) dims[i] = i;
        std::vector<int64_t> n_vec(ndim);
        for (int64_t i = 0; i < ndim; ++i) n_vec[i] = inputs[0].shape()[i];
        auto attr_n = attrs.get_int_list(AttrKey::Shape);
        if (!attr_n.empty()) {
            for (size_t i = 0; i < attr_n.size() && i < n_vec.size(); ++i) {
                n_vec[i] = attr_n[i];
            }
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return rocm::rocm_ifftn_kernel(inputs[0], dims, n_vec, norm, get_hip_stream(attrs));
    });

    // ========================================================================
    // Linear Algebra Operations (rocSOLVER or native HIP fallback)
    // ========================================================================
    table.register_single_output_kernel(OpId::LinalgDet, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::linalg_det_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LinalgInv, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::linalg_inv_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LinalgSolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::linalg_solve_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_kernel(OpId::LinalgSVD, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        bool full_matrices = attrs.get_bool(AttrKey::FullMatrices, true);
        auto [U, S, Vt] = rocm::linalg_svd_kernel(inputs[0], full_matrices, get_hip_stream(attrs));
        return {U, S, Vt};
    });
    table.register_kernel(OpId::LinalgQR, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [Q, R] = rocm::linalg_qr_kernel(inputs[0], get_hip_stream(attrs));
        return {Q, R};
    });
    table.register_kernel(OpId::LinalgEigh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [W, V] = rocm::linalg_eigh_kernel(inputs[0], get_hip_stream(attrs));
        return {W, V};
    });
    table.register_kernel(OpId::LinalgEig, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [WR, WI, V] = rocm::linalg_eig_kernel(inputs[0], get_hip_stream(attrs));
        return {WR, WI, V};
    });
    table.register_single_output_kernel(OpId::LinalgCholesky, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        return rocm::linalg_cholesky_kernel(inputs[0], upper, get_hip_stream(attrs));
    });
    table.register_kernel(OpId::LinalgLU,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto [L, U, pivots] = rocm::linalg_lu_kernel(inputs[0], get_hip_stream(attrs));
            return {L, U, pivots};
        });
    table.register_single_output_kernel(OpId::LinalgLUSolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::linalg_lu_solve_kernel(
                inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
        });
    table.register_single_output_kernel(OpId::SolveTriangular,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool upper = attrs.get_bool(AttrKey::Upper, true);
            bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
            return rocm::linalg_solve_triangular_kernel(
                inputs[0], inputs[1], upper, unitriangular, get_hip_stream(attrs));
        });
    table.register_single_output_kernel(OpId::LinalgCholeskySolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto stream = get_hip_stream(attrs);
            if (!upper) {
                auto Y = rocm::linalg_solve_triangular_kernel(inputs[1], inputs[0], false, false, stream);
                int64_t ndim = inputs[1].ndim();
                auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                return rocm::linalg_solve_triangular_kernel(Lt, Y, true, false, stream);
            } else {
                int64_t ndim = inputs[1].ndim();
                auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
                auto Y = rocm::linalg_solve_triangular_kernel(Ut, inputs[0], false, false, stream);
                return rocm::linalg_solve_triangular_kernel(inputs[1], Y, true, false, stream);
            }
        });
    table.register_kernel(OpId::Geqrf,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto [result, tau] = rocm::linalg_geqrf_kernel(inputs[0], get_hip_stream(attrs));
            return {result, tau};
        });
    table.register_single_output_kernel(OpId::Ormqr,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool left = attrs.get_bool(AttrKey::Left, true);
            bool transpose_q = attrs.get_bool(AttrKey::TransposeQ, false);
            return rocm::linalg_ormqr_kernel(inputs[0], inputs[1], inputs[2],
                left, transpose_q, get_hip_stream(attrs));
        });

    // ========================================================================
    // LinalgHouseholder, LinalgLDLFactor, LinalgLDLSolve,
    // CholeskyInverse, TensorInv, TensorSolve
    // ========================================================================
    table.register_single_output_kernel(OpId::LinalgHouseholder,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::linalg_householder_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });

    table.register_kernel(OpId::LinalgLDLFactor,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto [LD, pivots] = rocm::linalg_ldl_factor_kernel(inputs[0], get_hip_stream(attrs));
            return {LD, pivots};
        });

    table.register_single_output_kernel(OpId::LinalgLDLSolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::linalg_ldl_solve_kernel(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
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

    // ========================================================================
    // Sort/TopK/ArgSort/Unique Operations
    // ========================================================================
    table.register_kernel(OpId::TopK, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool largest = attrs.get_bool(AttrKey::Largest, true);
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        auto [values, indices] = rocm::topk_kernel(inputs[0], k, dim, largest, sorted, get_hip_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::Sort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        auto [values, indices] = rocm::sort_kernel(inputs[0], dim, descending, get_hip_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::Unique, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        bool return_inverse = attrs.get_bool(AttrKey::ReturnInverse, false);
        bool return_counts = attrs.get_bool(AttrKey::ReturnCounts, false);
        auto [unique_vals, inverse, counts] = rocm::unique_kernel(inputs[0], sorted, return_inverse, return_counts, get_hip_stream(attrs));
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    // ========================================================================
    // ScatterAdd Operation
    // ========================================================================
    // ========================================================================
    // Cast Operation
    // ========================================================================
    // ========================================================================
    // StridedFill Operation (in-place)
    // ========================================================================
    table.register_inplace_kernel(OpId::StridedFill, [](Tensor& self, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        double value = attrs.get_float(AttrKey::Value, 0.0);
        rocm::strided_fill_kernel(self, value, get_hip_stream(attrs));
        return self;
    });

    // ========================================================================
    // ToMemoryFormat Operation
    // ========================================================================
    // ========================================================================
    // CumSum/CumProd Operations
    // ========================================================================
    // ========================================================================
    // HasInfNan Operation
    // ========================================================================
    table.register_kernel(OpId::HasInfNan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::has_inf_nan_kernel(inputs[0], get_hip_stream(attrs))};
    });

    // ========================================================================
    // BatchNorm2dFusedTraining — compose from existing ops (no MIOpen fusion needed)
    // ========================================================================
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        auto stream = get_hip_stream(attrs);

        // Step 1: Compute batch mean and variance
        auto shape = inputs[0].shape();
        int64_t C = shape[1];
        Tensor mean({C}, inputs[0].dtype(), inputs[0].device());
        Tensor variance({C}, inputs[0].dtype(), inputs[0].device());
        rocm::batchnorm2d_mean_var(inputs[0], mean, variance, stream);

        // Step 2: Normalize with affine
        Tensor output = rocm::batchnorm2d_forward_affine(
            inputs[0], mean, variance, inputs[3], inputs[4], epsilon, stream);

        // Step 3: Update running stats
        Tensor running_mean = inputs[1];
        Tensor running_var = inputs[2];
        rocm::batchnorm2d_update_running_stats(running_mean, running_var, mean, variance, momentum, stream);

        return std::vector<Tensor>{output, mean, variance, running_mean, running_var};
    });

    // ========================================================================
    // BoxIoU Operation
    // ========================================================================
    // ========================================================================
    // NMS Operation
    // ========================================================================
    table.register_kernel(OpId::NMS, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // inputs: [boxes (N,4), scores (N)]
        // attrs: IouThreshold
        float iou_threshold = static_cast<float>(attrs.get_float(AttrKey::IouThreshold, 0.5));
        hipStream_t stream = get_hip_stream(attrs);
        return {rocm::nms_forward(inputs[0], inputs[1], iou_threshold, stream)};
    });

    // ========================================================================
    // EmbeddingBagForward Operation
    // ========================================================================
    // ========================================================================
    // EmbeddingBagBackward Operation
    // ========================================================================
    table.register_kernel(OpId::EmbeddingBagBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, indices (Int64), offsets]
        return std::vector<Tensor>{rocm::embedding_bag_backward_kernel(
            inputs[0], inputs[1], inputs[2], attrs, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Quantized Operations (INT8 inputs, INT32 accumulation, Float32 output)
    // ========================================================================
    // ========================================================================
    // Trunc Operation
    // ========================================================================
    // ========================================================================
    // Nonzero Operation
    // ========================================================================
    // ========================================================================
    // OneHot Operation
    // ========================================================================
    // ========================================================================
    // ROI Align Operations
    // ========================================================================
    table.register_kernel(OpId::ROIAlignForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 7);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 7);
        float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
        int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
        bool aligned = attrs.get_bool(AttrKey::Aligned, true);

        return {rocm::roi_align_forward(inputs[0], inputs[1],
                                        output_h, output_w, spatial_scale,
                                        sampling_ratio, aligned, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ROIAlignBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 1);
        int64_t feat_height = attrs.get_int(AttrKey::FeatHeight, 0);
        int64_t feat_width = attrs.get_int(AttrKey::FeatWidth, 0);
        float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
        int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
        bool aligned = attrs.get_bool(AttrKey::Aligned, true);

        return {rocm::roi_align_backward(inputs[0], inputs[1],
                                         batch_size, feat_height, feat_width,
                                         spatial_scale, sampling_ratio, aligned, get_hip_stream(attrs))};
    });

    // ========================================================================
    // RMSNorm Backward
    // ========================================================================
    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) {
        // inputs: [grad_output, input, rrms, weight] — order set by the autograd
        // RMSNormBackward (saved [input, rrms, weight]). The wrapper expects
        // (grad_output, input, weight, rrms), so pass weight=inputs[3], rrms=inputs[2].
        // (Previously these were swapped — weight and rrms transposed — which
        // produced wrong grad_input and out-of-bounds reads on the 1-element rrms.)
        auto [grad_input, grad_weight] = rocm::fused_rms_norm_backward_hip(
            inputs[0], inputs[1], inputs[3], inputs[2]);
        return std::vector<Tensor>{grad_input, grad_weight};
    });

    // ========================================================================
    // Fused Conv2D + Activation Variants (compose conv2d + activation)
    // ========================================================================
    // ========================================================================
    // Extended Math Operations
    // ========================================================================
    // ========================================================================
    // Special Math Functions — native ROCm device kernels
    // ========================================================================
    // Special math ops are registered below via ROCM_SINGLE_UNARY_NATIVE

    // Binary special math ops
    table.register_kernel(OpId::Beta, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::beta_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });
    table.register_kernel(OpId::Zeta, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::zeta_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Polygamma, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        return std::vector<Tensor>{rocm::polygamma_kernel(n, inputs[0], get_hip_stream(attrs))};
    });
    table.register_kernel(OpId::BetaInc, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::betainc_kernel(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs))};
    });

#undef ROCM_REGISTER_UNARY_SPECIAL
#undef ROCM_REGISTER_BINARY_SPECIAL

    // ========================================================================
    // Bool Predicate Operations
    // ========================================================================
    // ========================================================================
    // Binary Math Operations
    // ========================================================================
    // ========================================================================
    // New Unary Math Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::Rsqrt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::rsqrt_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Square, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::square_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Asinh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::asinh_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Acosh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::acosh_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Atanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::atanh_kernel(inputs[0], get_hip_stream(attrs));
    });

    // ========================================================================
    // New Binary Math Operations
    // ========================================================================
    table.register_single_output_kernel(OpId::Hypot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::hypot_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Copysign, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::copysign_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Nextafter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::nextafter_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Gcd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::gcd_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Lcm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::lcm_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Igamma, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::igamma_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Igammac, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::igammac_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    // ========================================================================
    // Ternary Operations
    // ========================================================================
    table.register_kernel(OpId::Lerp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::lerp_kernel(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs))};
    });
    table.register_kernel(OpId::Addcmul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return std::vector<Tensor>{rocm::addcmul_kernel(inputs[0], inputs[1], inputs[2], value, get_hip_stream(attrs))};
    });
    table.register_kernel(OpId::Addcdiv, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return std::vector<Tensor>{rocm::addcdiv_kernel(inputs[0], inputs[1], inputs[2], value, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Logical Operations
    // ========================================================================
    // ========================================================================
    // Element-wise Min/Max
    // ========================================================================
    // ========================================================================
    // Complex Number Operations
    // ========================================================================
    // ========================================================================
    // Any/All Reductions
    // ========================================================================
    table.register_kernel(OpId::Any, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::any_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::All, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::all_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Triu/Tril/Diag/Trace/Flip
    // ========================================================================
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
    // internally dispatch to rocSPARSE when inputs are on ROCm.
    // =========================================================================

    // SparseSpMM: sparse(M,K) @ dense(K,N) -> dense(M,N).
    //
    // NOTE: previously the rocSPARSE branch of this lambda called
    // sparse::spmm() recursively, which bounced back into tenzor_core
    // (where TENZOR_HAS_ROCSPARSE is not defined) and silently fell
    // through to the CPU path on device pointers. Call the kernel
    // directly, same as the HIP fallback path has always done.
    table.register_single_output_kernel(OpId::SparseSpMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return rocm::rocm_spmm_kernel(sp, inputs[3]);
        });

    // SparseSpMV: sparse(M,K) @ vec(K) -> vec(M).
    table.register_single_output_kernel(OpId::SparseSpMV,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return rocm::rocm_spmv_kernel(sp, inputs[3]);
        });

#ifdef TENZOR_HAS_ROCSPARSE
    // SparseSpGEMM: sparse(M,K) × sparse(K,N) -> sparse(M,N).
    // Inputs [0..2] are A's CSR components, [3..5] are B's. Lambda
    // returns the three CSR components of the product.
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            int64_t N = attrs.get_int(AttrKey::N);
            auto a = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            auto b = SparseTensor::sparse_csr(inputs[3], inputs[4], inputs[5], {K, N});
            auto c = rocm::rocm_spgemm_kernel(a, b);
            return {c.crow_indices(), c.col_indices(), c.values()};
        });

    // SparseTrsv: solve L*x = b.
    // Uses the standalone HIP kernel even when rocSPARSE is available:
    // rocSPARSE's SpSV has been observed to hang during the preprocess stage
    // on small triangular systems in recent ROCm releases.
    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return rocm::sparse_trsv_standalone_hip(inputs[0], inputs[1], inputs[2],
                                                    inputs[3], N, upper, /*stream=*/nullptr);
        });

    // SparseTrsm: solve L*X = B (multi-RHS).
    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return rocm::sparse_trsm_standalone_hip(inputs[0], inputs[1], inputs[2],
                                                    inputs[3], N, upper, /*stream=*/nullptr);
        });
#else
    // Standalone GPU SpGEMM/Trsv/Trsm — no rocSPARSE dependency
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return rocm::spgemm_standalone_hip(inputs, attrs, /*stream=*/nullptr);
        });

    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return rocm::sparse_trsv_standalone_hip(inputs[0], inputs[1], inputs[2],
                                                    inputs[3], N, upper, /*stream=*/nullptr);
        });

    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return rocm::sparse_trsm_standalone_hip(inputs[0], inputs[1], inputs[2],
                                                    inputs[3], N, upper, /*stream=*/nullptr);
        });
#endif // TENZOR_HAS_ROCSPARSE

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

    // SparseAdd: dedicated HIP kernel that operates directly on CSR data,
    // avoiding the recursive dispatch through sparse::add that previously
    // caused a stack overflow on GPU inputs.
    table.register_single_output_kernel(OpId::SparseAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return rocm::rocm_sparse_add_kernel(sp, inputs[3]);
        });

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

    // =========================================================================
    // Single-output kernel registrations
    //
    // Mirror the register_kernel entries above but via register_single_output_kernel
    // so callers that need only one Tensor avoid the std::vector allocation overhead.
    // =========================================================================

    // --- Binary Math Operations ------------------------------------------------
    table.register_single_output_kernel(OpId::Add, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::add_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Sub, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sub_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Mul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::mul_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Div, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::div_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::matmul_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::matmul_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Dot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::dot_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Atan2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::atan2_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Fmod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::fmod_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Remainder, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::remainder_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Minimum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::minimum_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Maximum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::maximum_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    // --- Unary Math Operations -------------------------------------------------
    table.register_single_output_kernel(OpId::Abs, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::abs_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Neg, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::neg_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Sign, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sign_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Sqrt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sqrt_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Reciprocal, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::reciprocal_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Log, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::log_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Log2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::log2_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Log10, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::log10_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Log1p, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::log1p_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Exp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::exp_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Exp2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::exp2_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Expm1, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::expm1_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Erf, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::erf_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Erfc, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::erfc_kernel(inputs[0], get_hip_stream(attrs));
    });

    // Special Math Functions — native ROCm device kernels (single_output path)
#define ROCM_SINGLE_UNARY_NATIVE(OP_ID, FN) \
    table.register_single_output_kernel(OpId::OP_ID, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor { \
        return rocm::FN(inputs[0], get_hip_stream(attrs)); \
    })

    ROCM_SINGLE_UNARY_NATIVE(Gamma,    gamma_kernel);
    ROCM_SINGLE_UNARY_NATIVE(Lgamma,   lgamma_kernel);
    ROCM_SINGLE_UNARY_NATIVE(Digamma,  digamma_kernel);
    ROCM_SINGLE_UNARY_NATIVE(BesselJ0, bessel_j0_kernel);
    ROCM_SINGLE_UNARY_NATIVE(BesselJ1, bessel_j1_kernel);
    ROCM_SINGLE_UNARY_NATIVE(BesselY0, bessel_y0_kernel);
    ROCM_SINGLE_UNARY_NATIVE(BesselY1, bessel_y1_kernel);
    ROCM_SINGLE_UNARY_NATIVE(BesselI0, bessel_i0_kernel);
    ROCM_SINGLE_UNARY_NATIVE(BesselI1, bessel_i1_kernel);
    ROCM_SINGLE_UNARY_NATIVE(ErfInv,   erfinv_kernel);
    ROCM_SINGLE_UNARY_NATIVE(Sinc,     sinc_kernel);
    ROCM_SINGLE_UNARY_NATIVE(Ndtr,     ndtr_kernel);
    ROCM_SINGLE_UNARY_NATIVE(LogNdtr,  log_ndtr_kernel);

#undef ROCM_SINGLE_UNARY_NATIVE

    // Multigammaln (needs extra dim parameter)
    table.register_single_output_kernel(OpId::Multigammaln, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int d = static_cast<int>(attrs.get_int(AttrKey::Dim, 1));
        return rocm::multigammaln_kernel(inputs[0], d, get_hip_stream(attrs));
    });

    // LinalgVectorNorm: delegates to existing Norm kernel
    table.register_kernel(OpId::LinalgVectorNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{rocm::norm_kernel(inputs[0], p, dim, keepdim, get_hip_stream(attrs))};
    });

    // LinalgMatrixNorm: Frobenius (ord=0), nuclear (ord=1), spectral (ord=2)
    table.register_single_output_kernel(OpId::LinalgMatrixNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ord = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        auto stream = get_hip_stream(attrs);
        if (ord == 0) {
            return rocm::norm_kernel(inputs[0], 2.0f, INT64_MIN, false, stream);
        }
        auto [U, S, Vt] = rocm::linalg_svd_kernel(inputs[0], false, stream);
        if (ord == 1) {
            return rocm::sum_kernel(S, INT64_MIN, false, stream);
        }
        return rocm::max_kernel(S, INT64_MIN, false, stream);
    });

    // LinalgVecdot: sum(a * b, dim)
    table.register_single_output_kernel(OpId::LinalgVecdot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        auto stream = get_hip_stream(attrs);
        Tensor product = rocm::mul_kernel(inputs[0], inputs[1], stream);
        return rocm::sum_kernel(product, dim, false, stream);
    });

    table.register_single_output_kernel(OpId::Floor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::floor_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Ceil, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::ceil_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Round, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::round_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Trunc, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::trunc_kernel(inputs[0], get_hip_stream(attrs));
    });

    // --- Trigonometric Operations ----------------------------------------------
    table.register_single_output_kernel(OpId::Sin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sin_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Cos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::cos_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::tan_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Asin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::asin_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Acos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::acos_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Atan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::atan_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Sinh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sinh_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Cosh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::cosh_kernel(inputs[0], get_hip_stream(attrs));
    });

    // --- Bool Predicate Operations ---------------------------------------------
    table.register_single_output_kernel(OpId::IsNan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::isnan_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IsInf, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::isinf_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IsFinite, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::isfinite_kernel(inputs[0], get_hip_stream(attrs));
    });

    // --- Logical Operations ----------------------------------------------------
    table.register_single_output_kernel(OpId::LogicalAnd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::logical_and_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogicalOr, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::logical_or_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogicalNot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::logical_not_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogicalXor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::logical_xor_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    // --- Comparison Operations -------------------------------------------------
    table.register_single_output_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::eq_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::ne_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::lt_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::le_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::gt_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::ge_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    // --- Complex Number Operations ---------------------------------------------
    table.register_single_output_kernel(OpId::Conj, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::conj_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Real, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::real_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Imag, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::imag_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Angle, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::angle_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Polar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::polar_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ComplexTensor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::complex_tensor_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Cross, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::cross_kernel(inputs[0], inputs[1], dim, get_hip_stream(attrs));
    });

    // --- Activation Functions --------------------------------------------------
    table.register_single_output_kernel(OpId::ReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::relu_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::relu_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Sigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sigmoid_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::sigmoid_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::TanhActivation, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::tanh_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::tanh_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::tanh_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);  // keep F64 precision
        return rocm::leaky_relu_kernel(inputs[0], alpha, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);
        return rocm::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Gelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::gelu_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::GeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::gelu_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return rocm::elu_kernel(inputs[0], alpha, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return rocm::elu_backward_kernel(inputs[0], inputs[1], alpha, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Selu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::selu_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::selu_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Swish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::swish_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SwishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::swish_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Mish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::mish_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::mish_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    // Forward-only (backward autograd-composed via clamp+mul, matching CPU).
    table.register_single_output_kernel(OpId::Hardswish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::hardswish_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Hardsigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::hardsigmoid_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
        return rocm::softplus_kernel(inputs[0], beta, threshold, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
        return rocm::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, get_hip_stream(attrs));
    });

    // --- Softmax Operations ----------------------------------------------------
    table.register_single_output_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::softmax_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::softmax_backward_kernel(inputs[0], inputs[1], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::log_softmax_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::log_softmax_backward_kernel(inputs[0], inputs[1], dim, get_hip_stream(attrs));
    });

    // --- Transform Operations --------------------------------------------------
    table.register_single_output_kernel(OpId::Contiguous, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::contiguous_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Clone, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::clone_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        return rocm::reshape_kernel(inputs[0], shape, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
        return rocm::transpose_kernel(inputs[0], dim0, dim1, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Permute dispatcher sets AttrKey::Dims (plural comma-separated string),
        // matching the convention used by Flip and Transpose. Reading AttrKey::Shape
        // would silently default to empty list and produce wrong axis order.
        auto dims = attrs.get_int_list(AttrKey::Dims);
        return rocm::permute_kernel(inputs[0], dims, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::squeeze_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::unsqueeze_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        return rocm::expand_kernel(inputs[0], shape, static_cast<void*>(get_hip_stream(attrs)));
    });
    table.register_single_output_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
        int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
        return rocm::flatten_kernel(inputs[0], start_dim, end_dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto repeats = attrs.get_int_list(AttrKey::Repeats);
        return rocm::repeat_kernel(inputs[0], repeats, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Tile dispatcher sets AttrKey::Reps (not Repeats — Repeats is for
        // OpId::Repeat). Reading the wrong key defaulted to empty list.
        auto reps = attrs.get_int_list(AttrKey::Reps);
        return rocm::tile_kernel(inputs[0], reps, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        auto results = rocm::stack_kernel(tensors, dim, get_hip_stream(attrs));
        return results[0];
    });
    table.register_single_output_kernel(OpId::Flip, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Phase 27-followup #40 fix: tenzor::flip sets AttrKey::Dims
        // (plural comma-separated string), not AttrKey::Dim. The wrong key
        // defaulted to 0 and flipped axis 0 regardless of the requested
        // dim. Parse and flip each dim in turn.
        auto dims_sv = attrs.get_string(AttrKey::Dims, "0");
        std::string dims_str(dims_sv);
        Tensor result = inputs[0];
        auto stream = get_hip_stream(attrs);
        std::istringstream ss(dims_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                int64_t dim = std::stoll(token);
                result = rocm::flip_kernel(result, dim, stream);
            }
        }
        return result;
    });
    table.register_single_output_kernel(OpId::Roll, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t shift = attrs.get_int(AttrKey::Shift, 0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::roll_kernel(inputs[0], shift, dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::RepeatInterleave, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        int64_t num_repeats = attrs.get_int(AttrKey::NumRepeats, 1);
        auto stream = get_hip_stream(attrs);
        if (num_repeats >= 0) {
            return rocm::repeat_interleave_scalar_kernel(inputs[0], num_repeats, dim, stream);
        } else {
            return rocm::repeat_interleave_tensor_kernel(inputs[0], inputs[1], dim, stream);
        }
    });
    table.register_single_output_kernel(OpId::ToMemoryFormat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int format_int = static_cast<int>(attrs.get_int(AttrKey::MemoryFormat, 0));
        MemoryFormat format = static_cast<MemoryFormat>(format_int);
        return rocm::to_memory_format_kernel(inputs[0], format, get_hip_stream(attrs));
    });

    // --- Creation Operations ---------------------------------------------------
    table.register_single_output_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double value = attrs.get_float(AttrKey::Value, 0.0);
        return rocm::fill_kernel(inputs[0], value, get_hip_stream(attrs));
    });

    // --- Indexing Operations ---------------------------------------------------
    table.register_single_output_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::gather_hip(inputs[0], dim, inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::scatter_hip(inputs[0], dim, inputs[1], inputs[2], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ScatterAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::scatter_add_kernel(inputs[0], dim, inputs[1], inputs[2], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::index_select_hip(inputs[0], dim, inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double value = attrs.get_float(AttrKey::Value, 0.0);
        return rocm::masked_fill_hip(inputs[0], inputs[1], value, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::masked_select_hip(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::where_hip(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        int64_t start = attrs.get_int(AttrKey::Start, 0);
        int64_t end = attrs.get_int(AttrKey::End, std::numeric_limits<int64_t>::max());
        int64_t step = attrs.get_int(AttrKey::Step, 1);
        return rocm::slice_hip(inputs[0], dim, start, end, step, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return rocm::cat_hip(tensors, dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Take, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::take_hip(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Put, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool accumulate = attrs.get_bool(AttrKey::Accumulate, false);
        Tensor input = inputs[0];
        return rocm::put_hip(input, inputs[1], inputs[2], accumulate, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Nonzero, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::nonzero_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::OneHot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t num_classes = attrs.get_int(AttrKey::NumClasses, 0);
        return rocm::one_hot_kernel(inputs[0], num_classes, get_hip_stream(attrs));
    });

    // --- Embedding Operations --------------------------------------------------
    table.register_single_output_kernel(OpId::Embedding, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::embedding_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::EmbeddingWithBoundsCheck, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::embedding_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
        return rocm::embedding_backward_kernel(inputs[0], inputs[1], num_embeddings, get_hip_stream(attrs));
    });
    // Returns {output, max_indices}; max_indices is the per-(bag,feature) global
    // argmax element index for mode="max" (empty otherwise).
    table.register_kernel(OpId::EmbeddingBagForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
        int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
        bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);
        return rocm::embedding_bag_forward_kernel(
            inputs[0], inputs[1], mode, embedding_dim, include_last_offset, get_hip_stream(attrs));
    });

    // --- Linear Operations -----------------------------------------------------
    table.register_single_output_kernel(OpId::Linear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::linear_kernel(inputs[0], inputs[1], bias, get_hip_stream(attrs));
    });

    // --- Dropout Backward (single output) --------------------------------------
    table.register_single_output_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        return rocm::dropout_backward_kernel(inputs[0], inputs[1], p, get_hip_stream(attrs));
    });

    // --- Convolution Operations ------------------------------------------------
    table.register_single_output_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: read per-axis stride/padding/dilation via shared
        // helpers, falling back to scalars. The per-axis ROCm overload
        // routes symmetric runs to the existing scalar kernel (no behavior
        // change) and throws cleanly on asymmetric until the im2col/MIOpen-
        // descriptor refactor lands — replacing the previous silent
        // wrong-output behavior.
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Phase 2.1: use shared per-axis helpers (replaces I5-followup inline
        // reads). The ROCm kernel signature accepts std::vector<int64_t> per axis.
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const auto stride   = ::tenzor::backend::attrs::stride_3d_vec(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_3d_vec(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_3d_vec(attrs);
        Tensor bias = inputs.size() > 2 ? inputs[2] : Tensor();
        return rocm::conv3d_forward_hip(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11 + Q.8: per-axis stride/padding/output_padding/dilation
        // via shared helpers. Dilation was previously omitted, silently
        // forced to 1 (PyTorch supports dilation > 1).
        const auto stride         = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding        = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto output_padding = ::tenzor::backend::attrs::output_padding_2d(attrs);
        const auto dilation       = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups      = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1],
            output_padding[0], output_padding[1],
            dilation[0], dilation[1], groups,
            get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: per-axis attrs via shared helpers (replaces inline
        // I5-followup reads). Mirrors Conv3dForward.
        const auto stride         = ::tenzor::backend::attrs::stride_3d_vec(attrs);
        const auto padding        = ::tenzor::backend::attrs::padding_3d_vec(attrs);
        const auto output_padding = ::tenzor::backend::attrs::output_padding_3d_vec(attrs);
        const auto dilation       = ::tenzor::backend::attrs::dilation_3d_vec(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        Tensor bias = inputs.size() > 2 ? inputs[2] : Tensor();
        return rocm::conv_transpose3d_forward_hip(inputs[0], inputs[1], bias,
            stride, padding, output_padding, dilation, groups, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t padding_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t padding_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dilation_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dilation_w = attrs.get_int(AttrKey::DilationW, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::depthwise_conv2d_kernel(
            inputs[0], inputs[1], bias, stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, get_hip_stream(attrs));
    });

    // Real native depthwise 1D/3D kernels (forward; backward autograd-composed).
    table.register_kernel(OpId::DepthwiseConv1d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t s = attrs.get_int(AttrKey::Stride, 1);
            int64_t p = attrs.get_int(AttrKey::Padding, 0);
            int64_t d = attrs.get_int(AttrKey::Dilation, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {rocm::depthwise_conv1d_kernel(inputs[0], inputs[1], bias, s, p, d,
                                                  get_hip_stream(attrs))};
        });
    table.register_kernel(OpId::DepthwiseConv3d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
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
            return {rocm::depthwise_conv3d_kernel(inputs[0], inputs[1], bias,
                                                  sD, sH, sW, pD, pH, pW, dD, dH, dW,
                                                  get_hip_stream(attrs))};
        });

    // --- Deformable Convolution v2 (DCNv2) ------------------------------------
    table.register_kernel(OpId::DeformableConv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: {input, offset, weight, bias, mask}
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        // bias (inputs[3]) and mask (inputs[4]) are optional; std::span has
        // no bounds checking and indexing past size() crashed (no-mask form).
        Tensor empty_t = Tensor({0}, inputs[0].dtype(), inputs[0].device());
        const Tensor& dcf_bias = inputs.size() > 3 ? inputs[3] : empty_t;
        const Tensor& dcf_mask = inputs.size() > 4 ? inputs[4] : empty_t;
        return std::vector<Tensor>{rocm::deformable_conv2d_forward_kernel(
            inputs[0], inputs[1], inputs[2], dcf_bias, dcf_mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, get_hip_stream(attrs))};
    });
    table.register_kernel(OpId::DeformableConv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: {grad_output, input, offset, weight, mask}
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        // mask (inputs[4]) is optional (no-mask form passes 4 inputs).
        Tensor empty_t = Tensor({0}, inputs[1].dtype(), inputs[1].device());
        const Tensor& dcb_mask = inputs.size() > 4 ? inputs[4] : empty_t;
        return rocm::deformable_conv2d_backward_input_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], dcb_mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, get_hip_stream(attrs));
    });
    table.register_kernel(OpId::DeformableConv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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
        // mask (inputs[3]) is optional; std::span has no bounds checking and
        // indexing past size() crashed on the no-mask call form.
        Tensor empty_t = Tensor({0}, inputs[1].dtype(), inputs[1].device());
        const Tensor& mask = inputs.size() > 3 ? inputs[3] : empty_t;
        return std::vector<Tensor>{rocm::deformable_conv2d_backward_weight_kernel(
            inputs[0], inputs[1], inputs[2], mask,
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, weight_shape, get_hip_stream(attrs))};
    });
    table.register_kernel(OpId::DeformableConv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Reuse regular conv2d bias backward (channel-wise sum of grad_output)
        return std::vector<Tensor>{rocm::conv2d_backward_bias(inputs[0], get_hip_stream(attrs))};
    });

    // --- Pooling Operations (2D) -----------------------------------------------
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: read per-axis kernel/stride/padding via shared
        // helpers. Previously replicated a single scalar into (kH,kH) etc.,
        // silently squashing asymmetric configs.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        // PyTorch/library default is count_include_pad=true (divide by full
        // window area); the nn layer always sets this attr explicitly, and on
        // direct dispatch we match CPU/CUDA which default to true.
        bool count_include_pad = attrs.get_bool(AttrKey::CountIncludePad, true);
        hipStream_t stream = get_hip_stream(attrs);
        return rocm::avgpool2d_forward_hip(inputs[0],
            kernel_size[0], kernel_size[1], stride[0], stride[1],
            padding[0], padding[1], count_include_pad, stream);
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: per-axis kernel/stride/padding via shared helpers.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        // PyTorch/library default is count_include_pad=true (divide by full
        // window area); the nn layer always sets this attr explicitly, and on
        // direct dispatch we match CPU/CUDA which default to true.
        bool count_include_pad = attrs.get_bool(AttrKey::CountIncludePad, true);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        hipStream_t stream = get_hip_stream(attrs);
        return rocm::avgpool2d_backward_hip(inputs[0], input_shape,
            kernel_size[0], kernel_size[1], stride[0], stride[1],
            padding[0], padding[1], count_include_pad, stream);
    });
    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        hipStream_t stream = get_hip_stream(attrs);
        return rocm::maxpool2d_backward_hip(inputs[0], inputs[1],
            input_shape, stream);
    });
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        hipStream_t stream = get_hip_stream(attrs);
        return rocm::adaptive_avgpool2d_hip(inputs[0], output_h, output_w, stream);
    });
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::adaptive_avgpool2d_backward_hip(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Autograd supplies (grad_output, indices) — matches the CPU / CUDA
        // registration contract. The ROCm kernel reads the input shape + dtype
        // via a placeholder tensor constructed from AttrKey::InputShape; the
        // underlying kernel only uses its .shape() / .dtype() / .device().
        auto input_shape_vec = attrs.get_int_list(AttrKey::InputShape);
        std::vector<int64_t> shape(input_shape_vec.begin(), input_shape_vec.end());
        Tensor input_placeholder(shape, inputs[0].dtype(), inputs[0].device());
        return rocm::adaptive_maxpool2d_backward_hip(
            inputs[0], inputs[1], input_placeholder, get_hip_stream(attrs));
    });

    // --- Vision Operations -----------------------------------------------------
    table.register_single_output_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // LL.3: per-axis Unfold accepts asymmetric kernel/stride/padding/dilation.
        const auto kernel_size = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 3);
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        return rocm::unfold_kernel(inputs[0],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // LL.3: per-axis Fold accepts asymmetric kernel/stride/padding/dilation.
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        const auto kernel_size = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 3);
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        return rocm::fold_kernel(inputs[0], output_size,
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "nearest"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return rocm::interpolate_kernel(inputs[0], size, mode, align_corners, get_hip_stream(attrs));
    });
    // Wave H4: native atomicAdd-scatter for both bilinear and nearest
    // backward (Float32/Float64). Nearest backward uses a single-destination
    // atomicAdd per output element (vs four bilinear corners); see
    // `interpolate_nearest_backward_kernel_hip` in vision.hip.cpp.
    table.register_single_output_kernel(OpId::InterpolateBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_size = attrs.get_int_list(AttrKey::InputShape);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return rocm::interpolate_backward_kernel(inputs[0], input_size, mode, align_corners,
                                                  get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::BoxIoU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int iou_type = static_cast<int>(attrs.get_int(AttrKey::IouType, 0));
        return rocm::box_iou_hip(inputs[0], inputs[1], iou_type, get_hip_stream(attrs));
    });

    // GridSample / AffineGrid — native ROCm kernels
    table.register_single_output_kernel(OpId::GridSample, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return rocm::grid_sample_kernel(inputs[0], inputs[1], mode, padding_mode, align_corners, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AffineGrid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size_span = attrs.get_int_list(AttrKey::OutputSize);
        std::vector<int64_t> size(size_span.begin(), size_span.end());
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return rocm::affine_grid_kernel_host(inputs[0], size, align_corners, get_hip_stream(attrs));
    });

    // audit Q.4: grid_sample / affine_grid backward.
    table.register_kernel(OpId::GridSampleBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            auto [gi, gg] = rocm::grid_sample_backward_kernel_host(
                inputs[2], inputs[0], inputs[1], mode, padding_mode, align_corners,
                get_hip_stream(attrs));
            return {gi, gg};
        });
    table.register_single_output_kernel(OpId::AffineGridBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto size_span = attrs.get_int_list(AttrKey::OutputSize);
            std::vector<int64_t> size(size_span.begin(), size_span.end());
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            return rocm::affine_grid_backward_kernel_host(
                inputs[0], size, align_corners, get_hip_stream(attrs));
        });

    // --- Cast/Dtype Operations -------------------------------------------------
    table.register_single_output_kernel(OpId::Cast, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        if (!attrs.has(AttrKey::TargetDtype)) {
            throw std::runtime_error("cast: missing 'target_dtype' attribute");
        }
        DType target_dtype = static_cast<DType>(attrs.get_int(AttrKey::TargetDtype));
        return rocm::cast_kernel(inputs[0], target_dtype, get_hip_stream(attrs));
    });

    // --- Scan Operations -------------------------------------------------------
    table.register_single_output_kernel(OpId::CumSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::cumsum_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::CumProd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::cumprod_kernel(inputs[0], dim, get_hip_stream(attrs));
    });

    // --- Sort/ArgSort ----------------------------------------------------------
    table.register_single_output_kernel(OpId::ArgSort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        return rocm::argsort_kernel(inputs[0], dim, descending, get_hip_stream(attrs));
    });

    // --- Matrix Operations -----------------------------------------------------
    table.register_single_output_kernel(OpId::Diag, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return rocm::diag_kernel(inputs[0], diagonal, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Trace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::trace_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Triu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return rocm::triu_kernel(inputs[0], diagonal, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tril, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return rocm::tril_kernel(inputs[0], diagonal, get_hip_stream(attrs));
    });

    // --- Fused Operations (single output) --------------------------------------
    table.register_single_output_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::fused_linear_relu_hip(inputs[0], inputs[1], bias, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return rocm::fused_batchnorm_relu_hip(inputs[0], inputs[1], inputs[2],
                                              inputs[3], inputs[4], eps);
    });
    table.register_single_output_kernel(OpId::FusedAddReLU, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) -> Tensor {
        return rocm::fused_add_relu_hip(inputs[0], inputs[1]);
    });
    table.register_single_output_kernel(OpId::FusedGelu, [](std::span<const Tensor> inputs, [[maybe_unused]] const OpAttributes& attrs) -> Tensor {
        return rocm::fused_gelu_hip(inputs[0]);
    });
    // FusedConv2dReLU: 3 inputs [input, weight, bias?]. Compose conv2d + relu;
    // do NOT call fused_conv_batchnorm_relu_hip here — that op needs 5 inputs
    // (input, weight, bias, bn_running_mean, bn_running_var) and is registered
    // as FusedConv2dBnReLU. Mirroring the FusedConv2dSigmoid path below.
    // Audit F.11: FusedConv2d* activation variants now read per-axis attrs
    // and dispatch through the per-axis conv2d_forward_kernel overload.
    // Previously the scalar overload silently squashed asymmetric configs.
    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs));
        return rocm::relu_kernel(result, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs));
        return rocm::sigmoid_kernel(result, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs));
        return rocm::tanh_kernel(result, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_hip_stream(attrs));
        return rocm::swish_kernel(result, get_hip_stream(attrs));
    });

    // --- Quantized Operations --------------------------------------------------
    table.register_single_output_kernel(OpId::QuantizedLinear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
        float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
        float output_scale = static_cast<float>(attrs.get_float(AttrKey::OutputScale, 1.0));
        int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
        int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));
        int32_t output_zp = static_cast<int32_t>(attrs.get_int(AttrKey::OutputZeroPoint, 0));
        return rocm::quantized_linear_hip(
            inputs[0], inputs[1], bias,
            input_scale, input_zp, weight_scale, weight_zp,
            output_scale, output_zp, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::QuantizedConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Audit F.11: read per-axis stride/padding/dilation. The underlying
        // ROCm quantized_conv2d_hip kernel only accepts scalar values, so
        // reject asymmetric configs loudly instead of silently squashing.
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "QuantizedConv2d (ROCm): backend kernel only supports symmetric "
                "stride/padding/dilation across H/W; got stride=(" +
                std::to_string(stride[0]) + "," + std::to_string(stride[1]) +
                "), padding=(" + std::to_string(padding[0]) + "," + std::to_string(padding[1]) +
                "), dilation=(" + std::to_string(dilation[0]) + "," + std::to_string(dilation[1]) + ")");
        }
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
        float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
        float output_scale = static_cast<float>(attrs.get_float(AttrKey::OutputScale, 1.0));
        int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
        int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));
        int32_t output_zp = static_cast<int32_t>(attrs.get_int(AttrKey::OutputZeroPoint, 0));
        return rocm::quantized_conv2d_hip(
            inputs[0], inputs[1], bias,
            stride[0], padding[0], dilation[0], groups,
            input_scale, input_zp, weight_scale, weight_zp,
            output_scale, output_zp, get_hip_stream(attrs));
    });

    // ========================================================================
    // Sampling / Statistics — native ROCm kernels
    // ========================================================================

    table.register_single_output_kernel(OpId::Bernoulli,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::bernoulli_kernel(inputs[0], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::PoissonSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::poisson_sample_kernel(inputs[0], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::NormalSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::normal_sample_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::ExponentialSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::exponential_sample_kernel(inputs[0], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::GammaSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::gamma_sample_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Multinomial,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_samples = attrs.get_int(AttrKey::NumSamples, 1);
            bool replacement = attrs.get_bool(AttrKey::Replacement, false);
            return rocm::multinomial_kernel(inputs[0], num_samples, replacement, get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Bucketize,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return rocm::bucketize_kernel(inputs[0], inputs[1], right, get_hip_stream(attrs));
        });

    table.register_kernel(OpId::Histogram,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t bins = attrs.get_int(AttrKey::NumBins, 10);
            double min_val = attrs.get_float(AttrKey::Min, 0.0);
            double max_val = attrs.get_float(AttrKey::Max, 0.0);
            auto [counts, edges] = rocm::histogram_kernel(inputs[0], bins, min_val, max_val, get_hip_stream(attrs));
            return {counts, edges};
        });

    // =========================================================================
    // Multi-dimensional Histogram
    // =========================================================================
    table.register_kernel(OpId::Histogramdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto bins_list = attrs.get_int_list(AttrKey::BinsList);
            bool density = attrs.get_bool(AttrKey::Density, false);

            std::vector<std::pair<double,double>> ranges;
            auto ranges_sv = attrs.get_string(AttrKey::RangesList, "");
            if (!ranges_sv.empty()) {
                std::string s(ranges_sv);
                std::vector<double> vals;
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

            auto [counts, edges] = rocm::histogramdd_kernel(inputs[0], bins_list, ranges, density, get_hip_stream(attrs));
            std::vector<Tensor> results;
            results.push_back(counts);
            for (auto& e : edges) results.push_back(std::move(e));
            return results;
        });

    table.register_single_output_kernel(OpId::CDist,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double p = attrs.get_float(AttrKey::DistP, 2.0);
            return rocm::cdist_kernel(inputs[0], inputs[1], p, get_hip_stream(attrs));
        });

    // =========================================================================
    // Trapezoid / Cumulative Trapezoid / Gradient / PairwiseDistance / Pdist
    // =========================================================================
    table.register_single_output_kernel(OpId::Trapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return rocm::trapezoid_kernel(inputs[0], dim, dx, x_ptr, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::CumulativeTrapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return rocm::cumulative_trapezoid_kernel(inputs[0], dim, dx, x_ptr, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NumericalGradient, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double spacing = attrs.get_float(AttrKey::Spacing, 1.0);
        return rocm::gradient_kernel(inputs[0], dim, spacing, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::PairwiseDistance, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return rocm::pairwise_distance_kernel(inputs[0], inputs[1], p, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::Pdist, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return rocm::pdist_kernel(inputs[0], p, get_hip_stream(attrs));
    });

    // STFT / ISTFT — native ROCm (rocFFT-backed)
    table.register_single_output_kernel(OpId::STFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return rocm::stft_kernel(inputs[0], n_fft, hop_length, win_length,
                                     window, center, normalized, onesided, get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::ISTFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            int64_t length = attrs.get_int(AttrKey::N, -1);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return rocm::istft_kernel(inputs[0], n_fft, hop_length, win_length,
                                      window, center, normalized, onesided, length, get_hip_stream(attrs));
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

    // AdvancedIndex (native ROCm fancy indexing)
    table.register_single_output_kernel(OpId::AdvancedIndex,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            std::vector<Tensor> indices(inputs.begin() + 1, inputs.end());
            return rocm::advanced_index_rocm_kernel(inputs[0], indices, num_indices, get_hip_stream(attrs));
        });

    // AdvancedIndexPut (native ROCm fancy indexing assignment)
    table.register_single_output_kernel(OpId::AdvancedIndexPut,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            // inputs[0] = destination, inputs[1] = values, inputs[2..2+N] = index tensors
            const auto& values = inputs[1];
            std::vector<Tensor> indices(inputs.begin() + 2, inputs.begin() + 2 + num_indices);
            return rocm::advanced_index_put_rocm_kernel(inputs[0], indices, values, num_indices, get_hip_stream(attrs));
        });

    // ========================================================================
    // New Phase 4 ops
    // ========================================================================
    table.register_single_output_kernel(OpId::Frac,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::frac_kernel(inputs[0], get_hip_stream(attrs));
        });
    table.register_single_output_kernel(OpId::LogSigmoid,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::log_sigmoid_kernel(inputs[0], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Heaviside,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::heaviside_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });
    table.register_single_output_kernel(OpId::NanToNum,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double nan_v = attrs.get_float(AttrKey::NanValue, 0.0);
            double posinf = attrs.get_float(AttrKey::PosInfValue, std::numeric_limits<double>::max());
            double neginf = attrs.get_float(AttrKey::NegInfValue, std::numeric_limits<double>::lowest());
            return rocm::nan_to_num_kernel(inputs[0], nan_v, posinf, neginf, get_hip_stream(attrs));
        });

    // Bitwise ops
    table.register_single_output_kernel(OpId::BitwiseAnd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::bitwise_and_kernel(inputs[0], inputs[1], get_hip_stream(attrs)); });
    table.register_single_output_kernel(OpId::BitwiseOr, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::bitwise_or_kernel(inputs[0], inputs[1], get_hip_stream(attrs)); });
    table.register_single_output_kernel(OpId::BitwiseXor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::bitwise_xor_kernel(inputs[0], inputs[1], get_hip_stream(attrs)); });
    table.register_single_output_kernel(OpId::BitwiseNot,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::bitwise_not_kernel(inputs[0], get_hip_stream(attrs));
        });
    table.register_single_output_kernel(OpId::BitwiseLeftShift, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::bitwise_left_shift_kernel(inputs[0], inputs[1], get_hip_stream(attrs)); });
    table.register_single_output_kernel(OpId::BitwiseRightShift, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::bitwise_right_shift_kernel(inputs[0], inputs[1], get_hip_stream(attrs)); });

    // RReLU, LogSigmoidBackward, NaN reductions, scatter variants — CPU dispatch
    // Each gets an explicit non-capturing dispatch function via template.
    table.register_single_output_kernel(OpId::RReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
        float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
        bool training = attrs.get_bool(AttrKey::Training, false);
        return rocm::rrelu_kernel(inputs[0], lower, upper, training, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::RReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float lower = static_cast<float>(attrs.get_float(AttrKey::Lower, 0.125));
        float upper = static_cast<float>(attrs.get_float(AttrKey::High, 0.333));
        return rocm::rrelu_backward_kernel(inputs[0], inputs[1], lower, upper, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::log_sigmoid_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    // CountNonzero: native HIP for both full reduction and dim-specific
    table.register_single_output_kernel(OpId::CountNonzero, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        if (dim < 0) {
            return rocm::count_nonzero_kernel(inputs[0], get_hip_stream(attrs));
        }
        return rocm::count_nonzero_dim_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    // Nansum / Nanmean. The native HIP nansum_kernel/nanmean_kernel only
    // implement the full-reduction (scalar) form, so a user-supplied
    // AttrKey::Dim was silently ignored — composed callers (notably
    // nanvar in src/ops/reduction.cpp) saw a scalar broadcast across all
    // rows and produced wildly wrong per-row results. Compose the
    // dim-aware path from sum + isnan + where; fall through to the
    // native scalar reducer when no dim is requested.
    table.register_single_output_kernel(OpId::Nansum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor& x = inputs[0];
        if (!attrs.has(AttrKey::Dim)) {
            return rocm::nansum_kernel(x, get_hip_stream(attrs));
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        Tensor cleaned = tenzor::where(isnan(x), zeros_like(x), x);
        return tenzor::sum(cleaned, dim, keepdim);
    });
    table.register_single_output_kernel(OpId::Nanmean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor& x = inputs[0];
        if (!attrs.has(AttrKey::Dim)) {
            return rocm::nanmean_kernel(x, get_hip_stream(attrs));
        }
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        Tensor mask = isnan(x);
        Tensor cleaned = where(mask, zeros_like(x), x);
        Tensor numer = tenzor::sum(cleaned, dim, keepdim);
        Tensor count = tenzor::sum(logical_not(mask).to(x.dtype()), dim, keepdim);
        return tenzor::div(numer, count);
    });
    // Aminmax: native HIP dual min/max reduction (returns 2 tensors)
    table.register_kernel(OpId::Aminmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [min_t, max_t] = rocm::aminmax_kernel(inputs[0], get_hip_stream(attrs));
        return {min_t, max_t};
    });
    table.register_single_output_kernel(OpId::IndexAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::index_add_kernel(inputs[0], inputs[1], inputs[2], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IndexCopy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::index_copy_kernel(inputs[0], inputs[1], inputs[2], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IndexFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        double value = attrs.get_float(AttrKey::Value, 0.0);
        return rocm::index_fill_kernel(inputs[0], inputs[1], dim, value, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ScatterReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        bool include_self = attrs.get_bool(AttrKey::IncludeSelf, true);
        return rocm::scatter_reduce_kernel(inputs[0], inputs[1], inputs[2], dim, reduce, include_self, get_hip_stream(attrs));
    });

    // SelectScatter: clone input, then copy src into the selected slice
    table.register_single_output_kernel(OpId::SelectScatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            const auto& input = inputs[0];
            const auto& src = inputs[1];
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            int64_t index = attrs.get_int(AttrKey::Index, 0);

            hipStream_t stream = get_hip_stream(attrs);
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
            if (dst_slice.is_contiguous()) {
                hipMemcpyAsync(dst_ptr, src_ptr, n * elem_size, hipMemcpyDeviceToDevice, stream);
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
                    hipMemcpyAsync(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size, hipMemcpyDeviceToDevice, stream);
                    for (int64_t d = ndims - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < dst_shape_v[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            // src_reshaped is a local temporary; sync so its buffer outlives
            // the stream-ordered copies before it is destroyed.
            hipStreamSynchronize(stream);
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

            hipStream_t stream = get_hip_stream(attrs);
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
            if (dst_slice.is_contiguous()) {
                hipMemcpyAsync(dst_ptr, src_ptr, n * elem_size, hipMemcpyDeviceToDevice, stream);
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
                    hipMemcpyAsync(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size, hipMemcpyDeviceToDevice, stream);
                    for (int64_t d = ndims - 1; d >= 0; d--) {
                        coord[d]++;
                        if (coord[d] < dst_shape_v[d]) break;
                        coord[d] = 0;
                    }
                }
            }
            // src_reshaped is a local temporary; sync so its buffer outlives
            // the stream-ordered copies before it is destroyed.
            hipStreamSynchronize(stream);
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

            hipStream_t stream = get_hip_stream(attrs);
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
            if (diag_len <= 0) return output;

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
                    hipMemcpyAsync(out_ptr + out_elem_offset * elem_size,
                                   src_ptr + src_elem_idx * elem_size, elem_size,
                                   hipMemcpyDeviceToDevice, stream);
                }

                for (int64_t i = static_cast<int64_t>(batch_dims.size()) - 1; i >= 0; i--) {
                    batch_coord[i]++;
                    if (batch_coord[i] < shape[batch_dims[i]]) break;
                    batch_coord[i] = 0;
                }
            }
            // src is a local contiguous temporary; sync so its buffer outlives
            // the stream-ordered copies before it is destroyed.
            hipStreamSynchronize(stream);
            return output;
        });

    // =========================================================================
    // Fused GEMM Operations (composed from existing ops)
    // =========================================================================
    table.register_single_output_kernel(OpId::Addmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto stream = get_hip_stream(attrs);
            // Compose: beta * input + alpha * (mat1 @ mat2)
            auto mm_result = rocm::matmul_kernel(inputs[1], inputs[2], stream);
            if (alpha != 1.0) {
                mm_result = rocm::mul_kernel(mm_result, tenzor::full({1}, alpha, mm_result.dtype(), mm_result.device()), stream);
            }
            if (beta == 0.0) {
                return mm_result;
            }
            auto scaled_input = (beta != 1.0)
                ? rocm::mul_kernel(inputs[0], tenzor::full({1}, beta, inputs[0].dtype(), inputs[0].device()), stream)
                : inputs[0];
            return rocm::add_kernel(scaled_input, mm_result, stream);
        });

    table.register_single_output_kernel(OpId::Addmv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto stream = get_hip_stream(attrs);
            // mat @ vec via matmul (treats 1D as column vector)
            auto mv_result = rocm::matmul_kernel(inputs[1], inputs[2].reshape({inputs[2].shape()[0], 1}), stream);
            mv_result = mv_result.reshape({inputs[1].shape()[0]});
            if (alpha != 1.0) {
                mv_result = rocm::mul_kernel(mv_result, tenzor::full({1}, alpha, mv_result.dtype(), mv_result.device()), stream);
            }
            if (beta == 0.0) {
                return mv_result;
            }
            auto scaled_input = (beta != 1.0)
                ? rocm::mul_kernel(inputs[0], tenzor::full({1}, beta, inputs[0].dtype(), inputs[0].device()), stream)
                : inputs[0];
            return rocm::add_kernel(scaled_input, mv_result, stream);
        });

    table.register_single_output_kernel(OpId::Baddbmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            auto stream = get_hip_stream(attrs);
            // batch1 @ batch2 via matmul (handles 3D)
            auto bmm_result = rocm::matmul_kernel(inputs[1], inputs[2], stream);
            if (alpha != 1.0) {
                bmm_result = rocm::mul_kernel(bmm_result, tenzor::full({1}, alpha, bmm_result.dtype(), bmm_result.device()), stream);
            }
            if (beta == 0.0) {
                return bmm_result;
            }
            auto scaled_input = (beta != 1.0)
                ? rocm::mul_kernel(inputs[0], tenzor::full({1}, beta, inputs[0].dtype(), inputs[0].device()), stream)
                : inputs[0];
            return rocm::add_kernel(scaled_input, bmm_result, stream);
        });

    // =========================================================================
    // Log-Cumulative-Sum-Exp
    // =========================================================================
    table.register_single_output_kernel(OpId::Logcumsumexp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return rocm::logcumsumexp_kernel(inputs[0], dim, get_hip_stream(attrs));
        });

    // =========================================================================
    // Bincount
    // =========================================================================
    table.register_single_output_kernel(OpId::Bincount,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t minlength = attrs.get_int(AttrKey::Minlength, 0);
            const Tensor* weights = (inputs.size() > 1) ? &inputs[1] : nullptr;
            return rocm::bincount_kernel(inputs[0], weights, minlength, get_hip_stream(attrs));
        });

    // =========================================================================
    // New Reduction Operations (CumMax, CumMin, Fmax, Fmin, Isin, Kthvalue, etc.)
    // =========================================================================

    table.register_kernel(OpId::CumMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = rocm::cummax_kernel(inputs[0], dim, get_hip_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::CumMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = rocm::cummin_kernel(inputs[0], dim, get_hip_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Fmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::fmax_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Fmin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::fmin_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Isin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return rocm::isin_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
        });

    table.register_kernel(OpId::Kthvalue, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        auto [values, indices] = rocm::kthvalue_kernel(inputs[0], k, dim, keepdim, get_hip_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Quantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return rocm::quantile_kernel(inputs[0], q, dim, keepdim, get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Nanquantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return rocm::nanquantile_kernel(inputs[0], q, dim, keepdim, get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Nanmedian,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return rocm::nanmedian_kernel(inputs[0], dim, false, get_hip_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Histc,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t bins = attrs.get_int(AttrKey::N, 100);
            double min_val = attrs.get_float(AttrKey::Alpha, 0.0);
            double max_val = attrs.get_float(AttrKey::Beta, 0.0);
            return rocm::histc_kernel(inputs[0], bins, min_val, max_val, get_hip_stream(attrs));
        });

    table.register_kernel(OpId::UniqueConsecutive, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool return_inverse = attrs.get_bool(AttrKey::Keepdim, false);
        auto [unique_vals, inverse, counts] = rocm::unique_consecutive_kernel(
            inputs[0], return_inverse, get_hip_stream(attrs));
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    table.register_single_output_kernel(OpId::SegmentReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t axis = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        return rocm::segment_reduce_kernel(inputs[0], inputs[1], reduce, axis, get_hip_stream(attrs));
    });

    // =========================================================================
    // TakeAlongDim
    // =========================================================================
    table.register_single_output_kernel(OpId::TakeAlongDim, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::take_along_dim_hip(inputs[0], inputs[1], dim, get_hip_stream(attrs));
    });

    // =========================================================================
    // MaskedScatter
    // =========================================================================
    table.register_single_output_kernel(OpId::MaskedScatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::masked_scatter_hip(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
    });

    // =========================================================================
    // TrilIndices
    // =========================================================================
    table.register_single_output_kernel(OpId::TrilIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return rocm::tril_indices_hip(row, col, offset, get_hip_stream(attrs));
    });

    // =========================================================================
    // TriuIndices
    // =========================================================================
    table.register_single_output_kernel(OpId::TriuIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return rocm::triu_indices_hip(row, col, offset, get_hip_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 2D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // RR.8: honour per-axis OutputRatio{H,W} when set.
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
        auto [output, indices] = rocm::fractional_maxpool2d_forward_hip(inputs[0], out_h, out_w, samples, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::fractional_maxpool2d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 3D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // RR.8: honour per-axis OutputRatio{D,H,W} when set.
        const auto& in_shape = inputs[0].shape();
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
        auto [output, indices] = rocm::fractional_maxpool3d_forward_hip(inputs[0], out_d, out_h, out_w, samples, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::fractional_maxpool3d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Max Unpool 2D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return rocm::max_unpool2d_forward_hip(inputs[0], inputs[1], out_h, out_w, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::max_unpool2d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Max Unpool 3D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return rocm::max_unpool3d_forward_hip(inputs[0], inputs[1], out_d, out_h, out_w, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::max_unpool3d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    // =========================================================================
    // Phase A.1: Max Unpool 1D (ROCm — wraps the 2D kernel via reshape).
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_l = attrs.get_int(AttrKey::OutputSizeW, 1);
        return rocm::max_unpool1d_forward_hip(inputs[0], inputs[1], out_l, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::max_unpool1d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    // =========================================================================
    // Phase 10: New Math, Bool, Binary, Reduction, and Manipulation Operations
    // =========================================================================

    // --- Unary math: Deg2Rad, Rad2Deg, Logit ---
    table.register_single_output_kernel(OpId::Deg2Rad, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::deg2rad_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Rad2Deg, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::rad2deg_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Logit, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double eps = attrs.get_float(AttrKey::Eps, -1.0);
        return rocm::logit_kernel(inputs[0], eps, get_hip_stream(attrs));
    });

    // --- Bool predicates: Signbit, IsPosInf, IsNegInf, IsReal ---
    table.register_single_output_kernel(OpId::Signbit, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::signbit_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IsPosInf, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::isposinf_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IsNegInf, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::isneginf_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::IsReal, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::isreal_kernel(inputs[0], get_hip_stream(attrs));
    });

    // --- Binary math: FloatPower, Xlog1py, Ldexp ---
    table.register_single_output_kernel(OpId::FloatPower, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::float_power_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Xlog1py, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::xlog1py_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Ldexp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::ldexp_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    // --- Two-output: Frexp ---
    table.register_kernel(OpId::Frexp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return rocm::frexp_kernel(inputs[0], get_hip_stream(attrs));
    });

    // --- Tensor manipulation: DiagEmbed, Diagflat ---
    table.register_single_output_kernel(OpId::DiagEmbed, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim0, -2);
        int64_t dim2 = attrs.get_int(AttrKey::Dim1, -1);
        return rocm::diag_embed_kernel(inputs[0], offset, dim1, dim2, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Diagflat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return rocm::diagflat_kernel(inputs[0], offset, get_hip_stream(attrs));
    });

    // --- NaN-aware reductions: NanVar, NanStd ---
    table.register_single_output_kernel(OpId::NanVar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
        return rocm::nanvar_kernel(inputs[0], unbiased, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::NanStd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool unbiased = attrs.get_bool(AttrKey::Unbiased, true);
        return rocm::nanstd_kernel(inputs[0], unbiased, get_hip_stream(attrs));
    });

    // =========================================================================
    // Nested Tensor Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::NestedSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::nested_softmax_hip(inputs[0], inputs[1], dim, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return rocm::nested_log_softmax_hip(inputs[0], inputs[1], dim, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::nested_sum_hip(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedMean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::nested_mean_hip(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = attrs.get_float(AttrKey::Eps, 1e-5f);
        return rocm::nested_layer_norm_hip(inputs[0], inputs[1], inputs[2], inputs[3], eps, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLinear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor* bias = (inputs.size() > 3) ? &inputs[3] : nullptr;
        return rocm::nested_linear_hip(inputs[0], inputs[2], bias, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float scale = attrs.get_float(AttrKey::Scale, 1.0f);
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        return rocm::nested_attention_hip(inputs[0], inputs[1], inputs[2],
                                           inputs[3], inputs[4], scale, causal, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedToPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t max_len = attrs.get_int(AttrKey::MaxLen, 0);
        float padding_value = attrs.get_float(AttrKey::PaddingValue, 0.0f);
        return rocm::nested_to_padded_hip(inputs[0], inputs[1], max_len, padding_value, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedFromPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::nested_from_padded_hip(inputs[0], inputs[1], get_hip_stream(attrs));
    });

    // =========================================================================
    // New math ops (PyTorch parity)
    // =========================================================================
    table.register_single_output_kernel(OpId::LogAddExp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::logaddexp_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogAddExp2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::logaddexp2_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::XLogY, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::xlogy_kernel(inputs[0], inputs[1], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::I0e, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::i0e_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::I1e, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::i1e_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Entr, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::entr_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SphericalBesselJ0, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::spherical_bessel_j0_kernel(inputs[0], get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Renorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::P, 2.0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        double maxnorm = attrs.get_float(AttrKey::MaxNorm, 1.0);
        return rocm::renorm_kernel(inputs[0], p, dim, maxnorm, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::CosineSimilarity, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 1);
        double eps = attrs.get_float(AttrKey::Eps, 1e-8);
        return rocm::cosine_similarity_kernel(inputs[0], inputs[1], dim, eps, get_hip_stream(attrs));
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
    // CTC Loss — log-domain forward-backward DP
    // =========================================================================
    table.register_kernel(OpId::CTCLossForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
            int64_t blank = attrs.get_int(AttrKey::Blank, 0);
            bool zero_infinity = attrs.get_bool(AttrKey::ZeroInfinity, false);
            return rocm::ctc_loss_forward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3],
                blank, zero_infinity, get_hip_stream(attrs));
        });

    // =========================================================================
    // NestedAttentionBackward — backward for segmented attention
    // =========================================================================
    table.register_kernel(OpId::NestedAttentionBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float scale = attrs.get_float(AttrKey::Scale, 1.0f);
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return rocm::nested_attention_backward_hip(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                inputs[5], inputs[6], scale, causal, get_hip_stream(attrs));
        });

    std::cout << "ROCm dispatch table initialized with O(1) lookup" << std::endl;
}

} // namespace tenzor

// Export for dynamic loading via dlsym
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::register_rocm_kernels(*table);
        }
    }
}
