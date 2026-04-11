/**
 * @file rocm_kernel_registry.cpp
 * @brief ROCm kernel registration for O(1) dispatch
 *
 * Registers all ROCm/HIP kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include <hip/hip_runtime.h>
#include "rocm_error.hpp"
#include <iostream>
#include <cstdlib>
#include <charconv>
#include <limits>
#include <climits>
#include <tuple>

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

// Helper to convert dtype string to DType enum
inline DType dtype_from_string(std::string_view s, DType default_val = DType::Float32) {
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    if (s == "float16") return DType::Float16;
    if (s == "bfloat16") return DType::BFloat16;
    if (s == "int32") return DType::Int32;
    if (s == "int64") return DType::Int64;
    if (s == "int8") return DType::Int8;
    if (s == "uint8") return DType::UInt8;
    if (s == "bool") return DType::Bool;
    return default_val;
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
    auto pow_kernel(const Tensor& input, float exponent, hipStream_t stream) -> Tensor;
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, hipStream_t stream) -> Tensor;
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

    // Grid sample / affine grid (native ROCm kernels — replaces previous CPU fallbacks)
    auto grid_sample_kernel(const Tensor& input, const Tensor& grid,
                            const std::string& mode, const std::string& padding_mode,
                            bool align_corners, hipStream_t stream) -> Tensor;
    auto affine_grid_kernel_host(const Tensor& theta, const std::vector<int64_t>& size,
                                  bool align_corners, hipStream_t stream) -> Tensor;

    // Sampling / statistics (native ROCm — replaces previous CPU fallbacks)
    auto bernoulli_kernel(const Tensor& probs, hipStream_t stream) -> Tensor;
    auto multinomial_kernel(const Tensor& probs, int64_t num_samples,
                            bool replacement, hipStream_t stream) -> Tensor;
    auto bucketize_kernel(const Tensor& input, const Tensor& boundaries,
                          bool right, hipStream_t stream) -> Tensor;
    auto histogram_kernel(const Tensor& input, int64_t bins,
                          double min_val, double max_val,
                          hipStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p,
                      hipStream_t stream) -> Tensor;

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

    // Binary math operations
    auto atan2_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto fmod_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b, hipStream_t stream) -> Tensor;

    // Ternary operations
    auto lerp_kernel(const Tensor& a, const Tensor& b, const Tensor& weight, hipStream_t stream) -> Tensor;

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

    // Activation functions
    auto relu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto gelu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto selu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto swish_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto mish_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold, hipStream_t stream) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, hipStream_t stream) -> Tensor;

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
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto fill_kernel(const Tensor& input, float value, hipStream_t stream) -> Tensor;
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
                           const Tensor& bias, const Tensor& h0,
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

    // Creation operations
    auto arange_kernel(double start, double end, double step, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto linspace_kernel(double start, double end, int64_t steps, DType dtype, Device device, hipStream_t stream) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, int64_t k, DType dtype, Device device, hipStream_t stream) -> Tensor;

    // Additional convolution and pooling operations
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                         int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                         int64_t output_padding_h, int64_t output_padding_w, hipStream_t stream) -> Tensor;
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias,
                                 int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w,
                                 int64_t dilation_h, int64_t dilation_w, hipStream_t stream) -> Tensor;
    auto adaptive_avgpool2d_backward_hip(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto adaptive_maxpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices, const Tensor& input, hipStream_t stream) -> Tensor;

    // 1D Pooling operations
    auto maxpool1d_forward_hip(const Tensor& input, int64_t kernel_size, int64_t stride,
                               int64_t padding, int64_t dilation, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto maxpool1d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto avgpool1d_forward_hip(const Tensor& input, int64_t kernel_size, int64_t stride,
                               int64_t padding, hipStream_t stream) -> Tensor;
    auto avgpool1d_backward_hip(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride, int64_t padding,
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
    auto maxpool3d_forward_hip(const Tensor& input, int64_t kernel_size, int64_t stride,
                               int64_t padding, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto maxpool3d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto avgpool3d_forward_hip(const Tensor& input, int64_t kernel_size, int64_t stride,
                               int64_t padding, hipStream_t stream) -> Tensor;
    auto avgpool3d_backward_hip(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride, int64_t padding,
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

    // Normalization operations
    auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                          const Tensor* weight, const Tensor* bias, float eps, hipStream_t stream) -> Tensor;
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
    auto fused_add_relu_hip(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_hip(const Tensor& input) -> Tensor;
    auto fused_layer_norm_hip(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                              const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_conv_batchnorm_relu_hip(const Tensor& conv_output, const Tensor& running_mean,
                                       const Tensor& running_var, const Tensor& weight,
                                       const Tensor& bias, float eps) -> Tensor;
    auto fused_attention_hip(const Tensor& Q, const Tensor& K, const Tensor& V,
                             float scale) -> std::pair<Tensor, Tensor>;
    auto flash_attention_backward_hip(
        const Tensor& dO, const Tensor& Q, const Tensor& K, const Tensor& V,
        const Tensor& O, const Tensor& L, float scale, bool causal) -> std::vector<Tensor>;

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
    auto unfold_kernel(const Tensor& input, int64_t kernel_size, int64_t stride,
                       int64_t padding, int64_t dilation, hipStream_t stream) -> Tensor;
    auto fold_kernel(const Tensor& input, const std::vector<int64_t>& output_size,
                     int64_t kernel_size, int64_t stride, int64_t padding,
                     int64_t dilation, hipStream_t stream) -> Tensor;
    auto interpolate_kernel(const Tensor& input, const std::vector<int64_t>& size,
                            const std::string& mode, bool align_corners, hipStream_t stream) -> Tensor;

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

    // Sparse operations (sparse.hip.cpp) — available with or without rocSPARSE
    auto rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) -> Tensor;
    auto rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) -> Tensor;
#ifdef TENZOR_HAS_ROCSPARSE
    // SpGEMM / triangular solve are only defined in the rocSPARSE path —
    // the HIP fallback at the bottom of sparse.hip.cpp intentionally
    // omits them. sparse_ops.cpp will fall through to the CPU path if
    // has_kernel returns false.
    auto rocm_spgemm_kernel(const SparseTensor& a, const SparseTensor& b) -> SparseTensor;
    auto rocm_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b,
                                  bool upper) -> Tensor;
    auto rocm_sparse_trsm_kernel(const SparseTensor& L, const Tensor& B,
                                  bool upper) -> Tensor;
#endif

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

    // Cast, StridedFill, ToMemoryFormat (transform.hip.cpp)
    auto cast_kernel(const Tensor& input, DType target_dtype, hipStream_t stream) -> Tensor;
    auto strided_fill_kernel(Tensor& self, float value, hipStream_t stream) -> void;
    auto to_memory_format_kernel(const Tensor& input, MemoryFormat format, void* stream_ptr) -> Tensor;

    // Any/All reductions (reduction.hip.cpp)
    auto any_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto all_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;

    // Triu/Tril/Diag/Trace (transform.hip.cpp)
    auto triu_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor;
    auto tril_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor;
    auto diag_kernel(const Tensor& input, int64_t diagonal, hipStream_t stream) -> Tensor;
    auto trace_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // CumSum, CumProd, HasInfNan (math.hip.cpp)
    auto cumsum_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
    auto has_inf_nan_kernel(const Tensor& input, hipStream_t stream) -> Tensor;

    // BoxIoU (vision.hip.cpp)
    auto box_iou_hip(const Tensor& boxes1, const Tensor& boxes2, int iou_type) -> Tensor;

    // NMS (nms.hip.cpp)
    auto nms_forward(const Tensor& boxes, const Tensor& scores, float iou_threshold, hipStream_t stream) -> Tensor;

    // EmbeddingBagForward/Backward (indexing.hip.cpp)
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                       const std::string& mode, int64_t embedding_dim,
                                       bool include_last_offset, hipStream_t stream) -> Tensor;
    auto embedding_bag_backward_kernel(const Tensor& grad_output,
                                       const Tensor& embeddings,
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
}

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
        float exponent = static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0));
        return std::vector<Tensor>{rocm::pow_kernel(inputs[0], exponent, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<float>::infinity()));
        float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity()));
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], min_val, max_val, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<float>::infinity()));
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], min_val, std::numeric_limits<float>::infinity(), get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity()));
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], -std::numeric_limits<float>::infinity(), max_val, get_hip_stream(attrs))};
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
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::zeros_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Ones, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::ones_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Full, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::full_kernel(shape, value, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Rand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::rand_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Randn, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int32_t device_id = static_cast<int32_t>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::randn_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Randint, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [grad_input, grad_gamma, grad_beta] = rocm::batchnorm2d_backward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });

    // ========================================================================
    // Convolution Operations
    // ========================================================================
    // Conv2dBackwardInput: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        // inputs[0]=grad_output, inputs[2]=weight
        return std::vector<Tensor>{rocm::conv2d_backward_input(inputs[0], inputs[2], input_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    // Conv2dBackwardWeight: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        // inputs[0]=grad_output, inputs[1]=input
        return std::vector<Tensor>{rocm::conv2d_backward_weight(inputs[0], inputs[1], weight_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::conv2d_backward_bias(inputs[0], get_hip_stream(attrs))};
    });

    // Conv3d operations
    // Conv3dBackwardInput: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto stride = attrs.get_int_list(AttrKey::Stride);
        auto padding = attrs.get_int_list(AttrKey::Padding);
        auto dilation = attrs.get_int_list(AttrKey::Dilation);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        // inputs[0]=grad_output, inputs[2]=weight
        return std::vector<Tensor>{rocm::conv3d_backward_input_hip(inputs[0], inputs[2], input_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    // Conv3dBackwardWeight: inputs = {grad_output, input, weight}
    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto stride = attrs.get_int_list(AttrKey::Stride);
        auto padding = attrs.get_int_list(AttrKey::Padding);
        auto dilation = attrs.get_int_list(AttrKey::Dilation);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        // inputs[0]=grad_output, inputs[1]=input
        return std::vector<Tensor>{rocm::conv3d_backward_weight_hip(inputs[0], inputs[1], weight_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::conv3d_backward_bias_hip(inputs[0], get_hip_stream(attrs))};
    });

    // ConvTranspose3d operations
    table.register_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto stride = attrs.get_int_list(AttrKey::Stride);
        auto padding = attrs.get_int_list(AttrKey::Padding);
        auto dilation = attrs.get_int_list(AttrKey::Dilation);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return std::vector<Tensor>{rocm::conv_transpose3d_backward_input_hip(inputs[0], inputs[1], input_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto stride = attrs.get_int_list(AttrKey::Stride);
        auto padding = attrs.get_int_list(AttrKey::Padding);
        auto dilation = attrs.get_int_list(AttrKey::Dilation);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
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
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        hipStream_t stream = get_hip_stream(attrs);
        auto [output, indices] = rocm::maxpool2d_forward_hip(inputs[0],
            kernel_size, kernel_size, stride, stride, padding, padding, true, stream);
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
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        auto [output, indices] = rocm::maxpool1d_forward_hip(inputs[0], kernel_size, stride, padding, dilation, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::maxpool1d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return rocm::avgpool1d_forward_hip(inputs[0], kernel_size, stride, padding, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return rocm::avgpool1d_backward_hip(inputs[0], input_shape, kernel_size, stride, padding, get_hip_stream(attrs));
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
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        auto [output, indices] = rocm::maxpool3d_forward_hip(inputs[0], kernel_size, stride, padding, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return rocm::maxpool3d_backward_hip(inputs[0], inputs[1], input_shape, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return rocm::avgpool3d_forward_hip(inputs[0], kernel_size, stride, padding, get_hip_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return rocm::avgpool3d_backward_hip(inputs[0], input_shape, kernel_size, stride, padding, get_hip_stream(attrs));
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
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [gates, c_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [h_out, c_out] = rocm::lstm_cell_forward_kernel(inputs[0], inputs[1], batch_size, hidden_size, get_hip_stream(attrs));
        return std::vector<Tensor>{h_out, c_out};
    });

    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, grad_c, gates, c_prev, c_out]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [grad_gates, grad_c_prev] = rocm::lstm_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_gates, grad_c_prev};
    });

    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [reset_gates, update_gates, new_gates_input, new_gates_hidden, h_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto h_out = rocm::gru_cell_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, get_hip_stream(attrs));
        return std::vector<Tensor>{h_out};
    });

    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden, h_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto result = rocm::gru_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            batch_size, hidden_size, get_hip_stream(attrs));
        return std::vector<Tensor>{result.grad_reset, result.grad_update, result.grad_new_input, result.grad_new_hidden, result.grad_h_prev};
    });

    // Full-sequence RNN operations
    // inputs: [input, W_ih, W_hh, bias, h0, c0] for LSTM
    // inputs: [input, W_ih, W_hh, bias, h0] for GRU
    table.register_kernel(OpId::LSTMForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // bias may be empty tensor if not provided
        return rocm::lstm_forward_kernel(inputs[0], inputs[1], inputs[2],
                                         inputs[3], inputs[4], inputs[5],
                                         get_hip_stream(attrs));
    });

    table.register_kernel(OpId::GRUForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // bias may be empty tensor if not provided
        return rocm::gru_forward_kernel(inputs[0], inputs[1], inputs[2],
                                        inputs[3], inputs[4],
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
    table.register_kernel(OpId::Arange, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = attrs.get_float(AttrKey::Start, 0.0);
        double end = attrs.get_float(AttrKey::End, 1.0);
        double step = attrs.get_float(AttrKey::Step, 1.0);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        Device device = device_from_string(attrs.get_string(AttrKey::Device, ""), Device::rocm(0));
        return std::vector<Tensor>{rocm::arange_kernel(start, end, step, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Linspace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = attrs.get_float(AttrKey::Start, 0.0);
        double end = attrs.get_float(AttrKey::End, 1.0);
        int64_t steps = attrs.get_int(AttrKey::Steps, 100);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        Device device = device_from_string(attrs.get_string(AttrKey::Device, ""), Device::rocm(0));
        return std::vector<Tensor>{rocm::linspace_kernel(start, end, steps, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Eye, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::layer_norm_kernel(inputs[0], normalized_shape, weight, bias, eps, get_hip_stream(attrs))};
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
        int64_t num_groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* weight = inputs.size() > 4 ? &inputs[4] : nullptr;
        auto [grad_input, grad_weight, grad_bias] = rocm::group_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], num_groups, weight, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::instance_norm_kernel(inputs[0], weight, bias, eps, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::InstanceNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const Tensor* weight = inputs.size() > 4 ? &inputs[4] : nullptr;
        auto [grad_input, grad_weight, grad_bias] = rocm::instance_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], weight, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    // ========================================================================
    // Fused Operations
    // ========================================================================
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: logits, targets
        std::string reduction = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
        return std::vector<Tensor>{rocm::fused_softmax_cross_entropy_hip(inputs[0], inputs[1], reduction)};
    });

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, weight, bias
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return std::vector<Tensor>{rocm::fused_layer_norm_hip(inputs[0], normalized_shape,
                                                              inputs[1], inputs[2], eps)};
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
    // Fused Attention
    // ========================================================================
    table.register_kernel(OpId::FusedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        auto [output, lse] = rocm::fused_attention_hip(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output};
    });

    // ========================================================================
    // Flash Attention (memory-efficient tiled attention)
    // ========================================================================
    table.register_kernel(OpId::FlashAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        auto [output, lse] = rocm::fused_attention_hip(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output, lse};
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

            // Composed-ops fallback for unsupported head_dim or missing L
            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::bmm(Q, Kt);

            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = tenzor::mul(scores, scale_t);

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

            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_inputs = {scores};
            Tensor attn_weights = tenzor::dispatch(OpId::Softmax, sm_inputs, sm_attrs)[0];

            Tensor attn_t = tenzor::transpose(attn_weights, -1, -2);
            Tensor dV = tenzor::bmm(attn_t, dO);

            Tensor Vt = tenzor::transpose(V, -1, -2);
            Tensor dAttn = tenzor::bmm(dO, Vt);

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

            Tensor dQ = tenzor::bmm(dScores, K);
            Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
            Tensor dK = tenzor::bmm(dScores_t, Q);

            return {dQ, dK, dV};
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
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* bias = inputs.size() > 2 && inputs[2].numel() > 0 ? &inputs[2] : nullptr;
        return rocm::fused_conv2d_bn_relu_full_hip(inputs[0], inputs[1], bias,
            inputs[5], inputs[6], inputs[3], inputs[4], stride, padding, eps);
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
        double lr, beta1, beta2, eps, weight_decay;
        int64_t step;
        bool decoupled, amsgrad;

        if (inputs.size() >= 5 && inputs[4].dtype() == DType::Float64 && inputs[4].numel() == 8) {
            const double* p = inputs[4].data<double>();
            lr = p[0]; beta1 = p[1]; beta2 = p[2]; eps = p[3];
            weight_decay = p[4]; step = static_cast<int64_t>(p[5]);
            decoupled = p[6] != 0.0; amsgrad = p[7] != 0.0;
        } else {
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

        std::vector<Tensor> W_ih_list, W_hh_list, bias_list;
        for (int64_t l = 0; l < num_layers; ++l) {
            size_t base_idx = 2 + l * 3;
            W_ih_list.push_back(inputs[base_idx]);
            W_hh_list.push_back(inputs[base_idx + 1]);
            bias_list.push_back(inputs[base_idx + 2]);
        }

        return rocm::gru_multi_layer_forward_kernel(input, W_ih_list, W_hh_list, bias_list, h0, get_hip_stream(attrs));
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
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
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
        // inputs: [grad_output, embeddings, offsets]
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
    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, rrms]
        auto [grad_input, grad_weight] = rocm::fused_rms_norm_backward_hip(
            inputs[0], inputs[1], inputs[2], inputs[3]);
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
#define ROCM_REGISTER_UNARY_SPECIAL(OP_ID, FN) \
    table.register_kernel(OpId::OP_ID, [](std::span<const Tensor> inputs, const OpAttributes& attrs) { \
        return std::vector<Tensor>{rocm::FN(inputs[0], get_hip_stream(attrs))}; \
    })
#define ROCM_REGISTER_BINARY_SPECIAL(OP_ID, FN) \
    table.register_kernel(OpId::OP_ID, [](std::span<const Tensor> inputs, const OpAttributes& attrs) { \
        return std::vector<Tensor>{rocm::FN(inputs[0], inputs[1], get_hip_stream(attrs))}; \
    })

    // Unary special math ops registered via ROCM_SINGLE_UNARY_NATIVE below
    // (single-output path only, avoids redundant register_kernel + register_single_output_kernel)
    ROCM_REGISTER_BINARY_SPECIAL(Beta,     beta_kernel);
    ROCM_REGISTER_BINARY_SPECIAL(Zeta,     zeta_kernel);

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
    // Ternary Operations
    // ========================================================================
    table.register_kernel(OpId::Lerp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::lerp_kernel(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs))};
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
    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return rocm::rocm_sparse_trsv_kernel(L, inputs[3], upper);
        });

    // SparseTrsm: solve L*X = B (multi-RHS).
    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return rocm::rocm_sparse_trsm_kernel(L, inputs[3], upper);
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

    // NOTE: SparseAdd on ROCm intentionally has no dedicated lambda —
    // see the matching comment in cuda_kernel_registry.cpp. The previous
    // implementation recursed through sparse::add and the dispatch table,
    // blowing the stack on GPU inputs.

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

#undef ROCM_SINGLE_UNARY_NATIVE

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
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        return rocm::leaky_relu_kernel(inputs[0], alpha, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
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
        auto dims = attrs.get_int_list(AttrKey::Shape);
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
        auto reps = attrs.get_int_list(AttrKey::Repeats);
        return rocm::tile_kernel(inputs[0], reps, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        auto results = rocm::stack_kernel(tensors, dim, get_hip_stream(attrs));
        return results[0];
    });
    table.register_single_output_kernel(OpId::Flip, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::flip_kernel(inputs[0], dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Roll, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t shift = attrs.get_int(AttrKey::Shift, 0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return rocm::roll_kernel(inputs[0], shift, dim, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ToMemoryFormat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int format_int = static_cast<int>(attrs.get_int(AttrKey::MemoryFormat, 0));
        MemoryFormat format = static_cast<MemoryFormat>(format_int);
        return rocm::to_memory_format_kernel(inputs[0], format, get_hip_stream(attrs));
    });

    // --- Creation Operations ---------------------------------------------------
    table.register_single_output_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
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
    table.register_single_output_kernel(OpId::EmbeddingBagForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
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
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto stride = attrs.get_int_list(AttrKey::Stride);
        auto padding = attrs.get_int_list(AttrKey::Padding);
        auto dilation = attrs.get_int_list(AttrKey::Dilation);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        Tensor bias = inputs.size() > 2 ? inputs[2] : Tensor();
        return rocm::conv3d_forward_hip(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, stride);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, stride);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t padding_h = attrs.get_int(AttrKey::PaddingH, padding);
        int64_t padding_w = attrs.get_int(AttrKey::PaddingW, padding);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t output_padding_h = attrs.get_int(AttrKey::OutputPaddingH, output_padding);
        int64_t output_padding_w = attrs.get_int(AttrKey::OutputPaddingW, output_padding);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        return rocm::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias,
            stride_h, stride_w, padding_h, padding_w, output_padding_h, output_padding_w,
            get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto stride = attrs.get_int_list(AttrKey::Stride);
        auto padding = attrs.get_int_list(AttrKey::Padding);
        auto output_padding = attrs.get_int_list(AttrKey::OutputPadding);
        auto dilation = attrs.get_int_list(AttrKey::Dilation);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
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

    // --- Pooling Operations (2D) -----------------------------------------------
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        bool count_include_pad = attrs.get_bool(AttrKey::CountIncludePad, true);
        hipStream_t stream = get_hip_stream(attrs);
        return rocm::avgpool2d_forward_hip(inputs[0],
            kernel_size, kernel_size, stride, stride, padding, padding, count_include_pad, stream);
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        bool count_include_pad = attrs.get_bool(AttrKey::CountIncludePad, true);
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        hipStream_t stream = get_hip_stream(attrs);
        return rocm::avgpool2d_backward_hip(inputs[0], input_shape,
            kernel_size, kernel_size, stride, stride, padding, padding, count_include_pad, stream);
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
        return rocm::adaptive_maxpool2d_backward_hip(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
    });

    // --- Vision Operations -----------------------------------------------------
    table.register_single_output_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        return rocm::unfold_kernel(inputs[0], kernel_size, stride, padding, dilation, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        return rocm::fold_kernel(inputs[0], output_size, kernel_size, stride, padding, dilation, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "nearest"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return rocm::interpolate_kernel(inputs[0], size, mode, align_corners, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::BoxIoU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int iou_type = static_cast<int>(attrs.get_int(AttrKey::IouType, 0));
        return rocm::box_iou_hip(inputs[0], inputs[1], iou_type);
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
    table.register_single_output_kernel(OpId::FusedAddReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::fused_add_relu_hip(inputs[0], inputs[1]);
    });
    table.register_single_output_kernel(OpId::FusedGelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return rocm::fused_gelu_hip(inputs[0]);
    });
    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return rocm::fused_conv_batchnorm_relu_hip(inputs[0], inputs[1], inputs[2],
                                                    inputs[3], inputs[4], eps);
    });
    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs));
        return rocm::sigmoid_kernel(result, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs));
        return rocm::tanh_kernel(result, get_hip_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        Tensor result = rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs));
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
        const Tensor* bias = (inputs.size() > 2 && inputs[2].numel() > 0) ? &inputs[2] : nullptr;
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
        float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
        float output_scale = static_cast<float>(attrs.get_float(AttrKey::OutputScale, 1.0));
        int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
        int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));
        int32_t output_zp = static_cast<int32_t>(attrs.get_int(AttrKey::OutputZeroPoint, 0));
        return rocm::quantized_conv2d_hip(
            inputs[0], inputs[1], bias,
            stride, padding, dilation, groups,
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

    table.register_single_output_kernel(OpId::CDist,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double p = attrs.get_float(AttrKey::DistP, 2.0);
            return rocm::cdist_kernel(inputs[0], inputs[1], p, get_hip_stream(attrs));
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
