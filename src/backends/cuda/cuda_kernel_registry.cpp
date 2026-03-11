/**
 * @file cuda_kernel_registry.cpp
 * @brief CUDA kernel registration for O(1) dispatch
 *
 * Registers all CUDA kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/fused_ops.hpp"
#ifdef TENZOR_HAS_CUDNN
#include "tenzor/backend/cudnn_wrapper.hpp"
#endif
#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <tuple>

namespace tenzor {

// Helper to convert dtype string to DType enum (matches creation.cpp's dtype_to_string)
inline DType dtype_from_string(std::string_view s, DType default_val = DType::Float32) {
    if (s == "float32") return DType::Float32;
    if (s == "float64") return DType::Float64;
    if (s == "float16") return DType::Float16;
    if (s == "bfloat16") return DType::BFloat16;
    if (s == "int32") return DType::Int32;
    if (s == "int64") return DType::Int64;
    if (s == "int16") return DType::Int16;
    if (s == "int8") return DType::Int8;
    if (s == "uint8") return DType::UInt8;
    if (s == "uint16") return DType::UInt16;
    if (s == "uint32") return DType::UInt32;
    if (s == "uint64") return DType::UInt64;
    if (s == "bool") return DType::Bool;
    if (s.empty()) return default_val;
    return default_val;
}

// Helper to extract CUDA stream from attributes
inline cudaStream_t get_cuda_stream(const OpAttributes& attrs) {
    if (attrs.has(AttrKey::Stream)) {
        return static_cast<cudaStream_t>(
            reinterpret_cast<void*>(static_cast<uint64_t>(attrs.get_int(AttrKey::Stream)))
        );
    }
    return nullptr;  // Default stream
}

// Forward declarations for CUDA kernels
namespace cuda {
    // Binary operations
    auto add_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto sub_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto mul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto div_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto matmul_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto dot_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // In-place operations
    auto add_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;
    auto sub_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;
    auto mul_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;
    auto div_inplace_kernel(Tensor& inout, const Tensor& other, cudaStream_t stream) -> Tensor;

    // In-place activation operations
    auto relu_inplace_kernel(Tensor& input, cudaStream_t stream) -> void;
    auto sigmoid_inplace_kernel(Tensor& input, cudaStream_t stream) -> void;
    auto tanh_inplace_kernel(Tensor& input, cudaStream_t stream) -> void;
    auto leaky_relu_inplace_kernel(Tensor& input, float alpha, cudaStream_t stream) -> void;
    auto gelu_inplace_kernel(Tensor& input, cudaStream_t stream) -> void;

    // Unary operations
    auto sqrt_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto neg_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto abs_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sign_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto log_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto exp_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto reciprocal_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto floor_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto ceil_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto round_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Operations with parameters
    auto clamp_kernel(const Tensor& input, float min_val, float max_val, cudaStream_t stream) -> Tensor;
    auto clamp_min_kernel(const Tensor& input, float min_val, cudaStream_t stream) -> Tensor;
    auto clamp_max_kernel(const Tensor& input, float max_val, cudaStream_t stream) -> Tensor;
    auto pow_kernel(const Tensor& input, float exponent, cudaStream_t stream) -> Tensor;

    // Trigonometric functions
    auto sin_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto cos_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tan_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto asin_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto acos_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto atan_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sinh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto cosh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction, cudaStream_t stream) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto any_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto all_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto logsumexp_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;

    // AMP operations
    auto has_inf_nan_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;

    // Activation functions
    auto relu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto gelu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto swish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto selu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto mish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto softplus_kernel(const Tensor& input, float beta, float threshold, cudaStream_t stream) -> Tensor;
    auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, cudaStream_t stream) -> Tensor;

    // Softmax operations
    auto softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;
    auto log_softmax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;

    // Transform operations
    auto contiguous_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto clone_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, cudaStream_t stream) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, cudaStream_t stream) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, cudaStream_t stream) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto expand_kernel(const Tensor& input, const std::vector<int64_t>& shape, void* stream) -> Tensor;
    auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, cudaStream_t stream) -> Tensor;
    auto cat_kernel(std::span<const Tensor> tensors, int64_t dim, cudaStream_t stream) -> Tensor;
    auto roll_kernel(const Tensor& input, int64_t shift, int64_t dim, cudaStream_t stream) -> Tensor;

    // Triangular / diagonal / flip operations
    auto triu_kernel(const Tensor& input, int64_t diagonal, cudaStream_t stream) -> Tensor;
    auto tril_kernel(const Tensor& input, int64_t diagonal, cudaStream_t stream) -> Tensor;
    auto diag_kernel(const Tensor& input, int64_t diagonal, cudaStream_t stream) -> Tensor;
    auto trace_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto flip_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;

    // Memory format conversion
    auto to_memory_format_kernel(const Tensor& input, MemoryFormat format, void* stream) -> Tensor;

    // Comparison operations
    auto eq_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto ne_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto lt_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto le_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto gt_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto ge_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // Indexing operations
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index, cudaStream_t stream) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index, cudaStream_t stream) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src, cudaStream_t stream) -> Tensor;
    auto scatter_add_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src, cudaStream_t stream) -> Tensor;
    auto searchsorted_kernel(const Tensor& sorted_sequence, const Tensor& values, bool right, cudaStream_t stream) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask, cudaStream_t stream) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, double value, cudaStream_t stream) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, cudaStream_t stream) -> Tensor;
    auto nonzero_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes, cudaStream_t stream) -> Tensor;
    auto take_kernel(const Tensor& input, const Tensor& indices, cudaStream_t stream) -> Tensor;
    auto put_kernel(Tensor& input, const Tensor& indices, const Tensor& source,
                    bool accumulate, cudaStream_t stream) -> Tensor;

    // Embedding operations
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, cudaStream_t stream) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices, int64_t num_embeddings, cudaStream_t stream) -> Tensor;
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets, const std::string& mode, int64_t embedding_dim, bool include_last_offset, cudaStream_t stream) -> Tensor;
    auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& embeddings, const Tensor& offsets, const OpAttributes& attrs, cudaStream_t stream) -> Tensor;

    // Linear algebra operations (cuSOLVER)
#ifdef TENZOR_HAS_CUSOLVER
    auto linalg_det_kernel(const Tensor& A, cudaStream_t stream) -> Tensor;
    auto linalg_inv_kernel(const Tensor& A, cudaStream_t stream) -> Tensor;
    auto linalg_solve_kernel(const Tensor& A, const Tensor& B, cudaStream_t stream) -> Tensor;
    auto linalg_svd_kernel(const Tensor& A, bool full_matrices, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_qr_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto linalg_eigh_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto linalg_eig_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_cholesky_kernel(const Tensor& A, bool upper, cudaStream_t stream) -> Tensor;
#endif

    // FFT operations (cuFFT)
#ifdef TENZOR_HAS_CUFFT
    auto cuda_fft_kernel(const Tensor& input, int64_t dim, int64_t n,
                         const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_ifft_kernel(const Tensor& input, int64_t dim, int64_t n,
                          const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_rfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                          const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_irfft_kernel(const Tensor& input, int64_t dim, int64_t n,
                           const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_fft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& n_vec,
                          const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_ifft2_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& n_vec,
                           const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_fftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                          const std::vector<int64_t>& n_vec,
                          const std::string& norm, cudaStream_t stream) -> Tensor;
    auto cuda_ifftn_kernel(const Tensor& input, const std::vector<int64_t>& dims,
                           const std::vector<int64_t>& n_vec,
                           const std::string& norm, cudaStream_t stream) -> Tensor;
#endif

    // Fused operations
    auto fused_conv2d_bn_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor* bias, const Tensor& bn_mean, const Tensor& bn_var, const Tensor& bn_gamma, const Tensor& bn_beta, int64_t stride, int64_t padding, float eps) -> Tensor;
    auto fused_linear_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_batchnorm_relu_cuda(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_add_relu_cuda(const Tensor& a, const Tensor& b) -> Tensor;
    auto cudnn_fused_conv2d_relu_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto fused_gelu_cuda(const Tensor& input) -> Tensor;

    // Full-sequence RNN operations
    auto lstm_forward_cuda(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                           const Tensor& bias_ih, const Tensor& bias_hh,
                           const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;
    auto gru_forward_cuda(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                          const Tensor& bias, const Tensor& h0) -> std::vector<Tensor>;
    auto lstm_multi_layer_forward_cuda(const Tensor& input,
                                       const std::vector<Tensor>& W_ih_list,
                                       const std::vector<Tensor>& W_hh_list,
                                       const std::vector<Tensor>& bias_list,
                                       const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;
    auto gru_multi_layer_forward_cuda(const Tensor& input,
                                      const std::vector<Tensor>& W_ih_list,
                                      const std::vector<Tensor>& W_hh_list,
                                      const std::vector<Tensor>& bias_list,
                                      const Tensor& h0) -> std::vector<Tensor>;
    auto bilstm_forward_cuda(const Tensor& input,
                             const Tensor& W_ih_fwd, const Tensor& W_hh_fwd,
                             const Tensor& bias_ih_fwd, const Tensor& bias_hh_fwd,
                             const Tensor& W_ih_bwd, const Tensor& W_hh_bwd,
                             const Tensor& bias_ih_bwd, const Tensor& bias_hh_bwd,
                             const Tensor& h0, const Tensor& c0) -> std::vector<Tensor>;

    // Vision/Interpolation operations
    auto interpolate_cuda(const Tensor& input, const std::vector<int64_t>& size, const std::string& mode, bool align_corners) -> Tensor;
    auto unfold_cuda(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) -> Tensor;
    auto fold_cuda(const Tensor& input, const std::vector<int64_t>& output_size, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation) -> Tensor;
    auto box_iou_cuda(const Tensor& boxes1, const Tensor& boxes2, int iou_type) -> Tensor;
    auto gather_relative_position_bias(const Tensor& table, const Tensor& indices,
                                       int64_t num_positions, int64_t num_heads) -> Tensor;

    // Advanced operations (topk, sort, cumsum, cumprod, unique)
    auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto sort_kernel(const Tensor& input, int64_t dim, bool descending, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cumsum_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto unique_kernel(const Tensor& input, bool sorted_output, bool return_inverse, bool return_counts, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // ROI Align operations
    auto roi_align_forward(const Tensor& features, const Tensor& rois,
                           int64_t output_h, int64_t output_w,
                           float spatial_scale, int64_t sampling_ratio,
                           bool aligned) -> Tensor;
    auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                            int64_t batch_size, int64_t feat_height, int64_t feat_width,
                            float spatial_scale, int64_t sampling_ratio,
                            bool aligned, cudaStream_t stream = nullptr) -> Tensor;

    // BatchNorm2d operations
    auto batchnorm2d_mean_var(const Tensor& input, Tensor& mean, Tensor& variance, cudaStream_t stream) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine_optimized(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum, cudaStream_t stream) -> void;
    auto batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // Fused LayerNorm operation
    auto fused_layer_norm_cuda(
        const Tensor& input,
        const std::vector<int64_t>& normalized_shape,
        const Tensor& weight,
        const Tensor& bias,
        float eps
    ) -> std::tuple<Tensor, Tensor, Tensor>;

    // Fused RMSNorm operation
    auto fused_rms_norm_cuda(
        const Tensor& input,
        const Tensor& weight,
        float eps
    ) -> std::tuple<Tensor, Tensor>;

    // Fused Attention operation
    auto fused_attention_cuda(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        float scale
    ) -> Tensor;

    // Fused optimizer operations
    auto fused_sgd_step_cuda(
        Tensor& param,
        const Tensor& grad,
        Tensor* momentum_buffer,
        float lr,
        float momentum,
        float weight_decay,
        float dampening,
        bool nesterov,
        cudaStream_t stream
    ) -> void;

    auto fused_adam_step_cuda(
        Tensor& param,
        const Tensor& grad,
        Tensor& exp_avg,
        Tensor& exp_avg_sq,
        double lr,
        double beta1,
        double beta2,
        double eps,
        double weight_decay,
        int64_t step,
        bool decoupled_weight_decay,
        cudaStream_t stream,
        Tensor* max_exp_avg_sq,
        bool amsgrad
    ) -> void;

    auto fused_adam_atan2_step_cuda(
        Tensor& param,
        const Tensor& grad,
        Tensor& exp_avg,
        Tensor& exp_avg_sq,
        Tensor* max_exp_avg_sq,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        int64_t step,
        bool amsgrad,
        cudaStream_t stream
    ) -> void;

    // Linear layer operations (fused cuBLAS with bias)
    auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, cudaStream_t stream) -> Tensor;
    auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, cudaStream_t stream) -> std::vector<Tensor>;

    // Conv2d operations
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;

    // Pooling operations (custom kernels - fallback)
    auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;
    auto avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;

    // 1D Pooling operations
    auto maxpool1d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto avgpool1d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;
    auto avgpool1d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;

    // 3D Pooling operations
    auto maxpool3d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto avgpool3d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;
    auto avgpool3d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;

#ifdef TENZOR_HAS_CUDNN
    // cuDNN pooling operations (faster than custom kernels)
    auto cudnn_maxpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cudnn_maxpool2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& output, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;
    auto cudnn_avgpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;
    auto cudnn_avgpool2d_backward(const Tensor& grad_output, const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;

    // cuDNN softmax operations (faster than custom kernels)
    auto cudnn_softmax_forward(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto cudnn_softmax_backward(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;
    auto cudnn_log_softmax_forward(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto cudnn_log_softmax_backward(const Tensor& grad_output, const Tensor& output, int64_t dim, cudaStream_t stream) -> Tensor;

    // cuDNN BatchNorm2d operations
    auto cudnn_batchnorm2d_forward_training(const Tensor& input, Tensor& running_mean, Tensor& running_var, const Tensor& gamma, const Tensor& beta, float momentum, float epsilon, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto cudnn_batchnorm2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& gamma, const Tensor& saved_mean, const Tensor& saved_inv_var, float epsilon, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
#endif

    // Fill operations
    auto fill_kernel(const Tensor& tensor, float value, cudaStream_t stream) -> Tensor;
    auto strided_fill_kernel(Tensor& self, float value, cudaStream_t stream) -> void;

    // Runtime cuDNN availability check
    bool is_cudnn_available() noexcept;
    bool is_cudnn_frontend_available() noexcept;

    // Conv2d backward and transpose
    auto conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto depthwise_conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> Tensor;

#ifdef TENZOR_HAS_CUDNN
    auto cudnn_conv2d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto cudnn_layer_norm_forward(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto cudnn_layer_norm_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& mean, const Tensor& inv_std, const std::vector<int64_t>& normalized_shape, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // Conv3d (cuDNN Nd)
    auto cudnn_conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv3d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // ConvTranspose3d (cuDNN Nd)
    auto cudnn_conv_transpose3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv_transpose3d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
#endif

    // Dropout operations
    auto dropout_forward_kernel(const Tensor& input, float p, bool training, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p, cudaStream_t stream) -> Tensor;

    // LSTM operations
    auto lstm_cell_forward_kernel(const Tensor& gates, const Tensor& c_prev, int64_t batch_size, int64_t hidden_size, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_h, const Tensor& grad_c, const Tensor& gates, const Tensor& c_prev, const Tensor& c_out, int64_t batch_size, int64_t hidden_size, cudaStream_t stream) -> std::pair<Tensor, Tensor>;

    // GRU operations
    struct GRUBackwardOutputs {
        Tensor grad_reset;
        Tensor grad_update;
        Tensor grad_new_input;
        Tensor grad_new_hidden;
        Tensor grad_h_prev;
    };
    auto gru_cell_forward_kernel(const Tensor& reset_gates, const Tensor& update_gates, const Tensor& new_gates_input, const Tensor& new_gates_hidden, const Tensor& h_prev, int64_t batch_size, int64_t hidden_size, cudaStream_t stream) -> Tensor;
    auto gru_cell_backward_kernel(const Tensor& grad_h, const Tensor& reset_gates, const Tensor& update_gates, const Tensor& new_gates_input, const Tensor& new_gates_hidden, const Tensor& h_prev, int64_t batch_size, int64_t hidden_size, cudaStream_t stream) -> GRUBackwardOutputs;

    // Adaptive pooling operations
    auto adaptive_avg_pool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w, cudaStream_t stream) -> Tensor;
    auto adaptive_avg_pool2d_backward(const Tensor& grad_output, int64_t H_in, int64_t W_in, cudaStream_t stream) -> Tensor;
    auto adaptive_max_pool2d_forward(const Tensor& input, int64_t output_h, int64_t output_w, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto adaptive_max_pool2d_backward(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;

    // Adaptive 1D pooling operations
    auto adaptive_maxpool1d_forward(const Tensor& input, int64_t output_size, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool1d_backward(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto adaptive_avgpool1d_forward(const Tensor& input, int64_t output_size, cudaStream_t stream) -> Tensor;
    auto adaptive_avgpool1d_backward(const Tensor& grad_output, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;

    // Adaptive 3D pooling operations
    auto adaptive_maxpool3d_forward(const Tensor& input, int64_t output_d, int64_t output_h, int64_t output_w, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool3d_backward(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto adaptive_avgpool3d_forward(const Tensor& input, int64_t output_d, int64_t output_h, int64_t output_w, cudaStream_t stream) -> Tensor;
    auto adaptive_avgpool3d_backward(const Tensor& grad_output, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;

    // GroupNorm / InstanceNorm operations
    auto group_norm_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias, int64_t num_groups, float eps, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& mean_saved, const Tensor& inv_std_saved, int64_t num_groups, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto instance_norm_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& mean_saved, const Tensor& inv_std_saved, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // RMSNorm backward
    auto fused_rms_norm_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& rrms) -> std::tuple<Tensor, Tensor>;

    // Fused LayerNorm backward
    auto fused_layer_norm_backward_cuda(const Tensor& grad_output, const Tensor& input, const Tensor& weight, const Tensor& mean, const Tensor& inv_std, const std::vector<int64_t>& normalized_shape) -> std::tuple<Tensor, Tensor, Tensor>;

    // Creation operations
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto randint_kernel(int64_t low, int64_t high, const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto arange_kernel(float start, float end, float step, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto linspace_kernel(float start, float end, int64_t steps, DType dtype, Device device, cudaStream_t stream) -> Tensor;
    auto eye_kernel(int64_t n, int64_t m, DType dtype, Device device, cudaStream_t stream) -> Tensor;

    // Transform operations
    auto flatten_kernel(const Tensor& input, int64_t start_dim, int64_t end_dim, cudaStream_t stream) -> Tensor;
    auto slice_kernel(const Tensor& input, const std::vector<int64_t>& starts, const std::vector<int64_t>& ends, const std::vector<int64_t>& steps, cudaStream_t stream) -> Tensor;
    auto stack_kernel(std::span<const Tensor> tensors, int64_t dim, cudaStream_t stream) -> Tensor;
    auto split_kernel(const Tensor& input, int64_t split_size, int64_t dim, cudaStream_t stream) -> std::vector<Tensor>;
    auto chunk_kernel(const Tensor& input, int64_t chunks, int64_t dim, cudaStream_t stream) -> std::vector<Tensor>;
    auto tile_kernel(const Tensor& input, const std::vector<int64_t>& reps, cudaStream_t stream) -> Tensor;

    // ArgSort
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending, cudaStream_t stream) -> Tensor;

    // Fused Softmax Cross Entropy
    auto fused_softmax_cross_entropy_cuda(const Tensor& logits, const Tensor& targets, bool compute_grad) -> std::tuple<Tensor, Tensor>;

    // Fused optimizer steps
    auto fused_rmsprop_step_cuda(Tensor& param, const Tensor& grad, Tensor& square_avg, Tensor* grad_avg, Tensor* momentum_buffer, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered, cudaStream_t stream) -> void;
    auto fused_adadelta_step_cuda(Tensor& param, const Tensor& grad, Tensor& square_avg, Tensor& acc_delta, float rho, float eps, float lr, float weight_decay, cudaStream_t stream) -> void;
    auto fused_adagrad_step_cuda(Tensor& param, const Tensor& grad, Tensor& sum_sq, float lr, float lr_decay, float eps, float weight_decay, int64_t step, cudaStream_t stream) -> void;

    // =========================================================================
    // Dispatch-Conformant Wrappers (SingleOutputKernelFn signature)
    // =========================================================================
    // Binary operations
    Tensor add_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor sub_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor mul_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor div_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    // matmul_dispatch and dot_dispatch use lambdas in registration since
    // matmul_kernel/dot_kernel are defined in cublas_ops.cu

    // Inplace operations (InplaceKernelFn signature)
    Tensor& add_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs);
    Tensor& sub_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs);
    Tensor& mul_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs);
    Tensor& div_inplace_dispatch(Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs);

    // Unary operations
    Tensor sqrt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor neg_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor abs_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor sign_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor log_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor exp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor reciprocal_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor floor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor ceil_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor round_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor trunc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Trigonometric operations
    Tensor sin_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor cos_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor tan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor asin_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor acos_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor atan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor sinh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor cosh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Comparison operations
    Tensor eq_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor ne_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor lt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor le_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor gt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor ge_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Activation operations
    Tensor relu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor relu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor sigmoid_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor sigmoid_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor tanh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor tanh_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor gelu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor gelu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor swish_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor swish_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor selu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor selu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor mish_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor mish_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Cast (dtype conversion) dispatch wrapper
    Tensor cast_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Extended unary math kernels
    auto log2_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto log10_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto log1p_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto exp2_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto expm1_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto erf_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto erfc_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Bool predicate kernels
    auto isnan_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto isinf_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto isfinite_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;

    // Binary math kernels
    auto atan2_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto fmod_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto remainder_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // Ternary kernel
    auto lerp_kernel(const Tensor& start, const Tensor& end, const Tensor& weight, cudaStream_t stream) -> Tensor;

    // Logical kernels
    auto logical_and_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto logical_or_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto logical_not_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto logical_xor_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // Element-wise min/max kernels
    auto minimum_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto maximum_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;

    // Extended unary math dispatch wrappers
    Tensor log2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor log10_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor log1p_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor exp2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor expm1_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor erf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor erfc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Bool predicate dispatch wrappers
    Tensor isnan_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor isinf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor isfinite_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Binary math dispatch wrappers
    Tensor atan2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor fmod_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor remainder_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Lerp dispatch wrapper
    Tensor lerp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Logical dispatch wrappers
    Tensor logical_and_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor logical_or_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor logical_not_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor logical_xor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Element-wise min/max dispatch wrappers
    Tensor minimum_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor maximum_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Complex number dispatch wrappers
    Tensor conj_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor real_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor imag_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor angle_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor polar_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

} // namespace cuda

// Forward declarations for quantized CUDA kernels (in nn::quantization::kernels namespace)
namespace nn::quantization::kernels {
    auto quantized_linear_cuda(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch_size, int64_t in_features, int64_t out_features,
        float input_scale, float weight_scale, float output_scale,
        int32_t input_zp, int32_t weight_zp, cudaStream_t stream
    ) -> void;

    auto quantized_conv2d_cuda(
        const int8_t* input, const int8_t* weight, const float* bias,
        float* output, int64_t batch, int64_t in_channels, int64_t out_channels,
        int64_t h_in, int64_t w_in, int64_t h_out, int64_t w_out,
        int64_t kernel_size, int64_t stride, int64_t padding,
        float input_scale, float weight_scale,
        int32_t input_zp, int32_t weight_zp, cudaStream_t stream
    ) -> void;
} // namespace nn::quantization::kernels

/**
 * @brief Register all CUDA kernels with the dispatch table.
 */
void register_cuda_kernels(BackendDispatchTable& table) {
    // =========================================================================
    // Arithmetic Operations (using direct function pointers - no lambda overhead)
    // =========================================================================
    table.register_single_output_kernel(OpId::Add, cuda::add_dispatch);
    table.register_single_output_kernel(OpId::Sub, cuda::sub_dispatch);
    table.register_single_output_kernel(OpId::Mul, cuda::mul_dispatch);
    table.register_single_output_kernel(OpId::Div, cuda::div_dispatch);
    table.register_single_output_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::matmul_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });
    // Bmm (batched matrix multiplication) uses the same kernel as MatMul
    // The CUDA matmul kernel already handles batched inputs via cublasSgemmStridedBatched
    table.register_single_output_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::matmul_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Dot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::dot_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    // Inplace operations (using InplaceKernelFn - no tensor copy)
    table.register_inplace_kernel(OpId::AddInplace, cuda::add_inplace_dispatch);
    table.register_inplace_kernel(OpId::SubInplace, cuda::sub_inplace_dispatch);
    table.register_inplace_kernel(OpId::MulInplace, cuda::mul_inplace_dispatch);
    table.register_inplace_kernel(OpId::DivInplace, cuda::div_inplace_dispatch);

    // Inplace activation operations (using InplaceKernelFn - no tensor copy)
    table.register_inplace_kernel(OpId::ReLUInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        cuda::relu_inplace_kernel(target, get_cuda_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::SigmoidInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        cuda::sigmoid_inplace_kernel(target, get_cuda_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::TanhInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        cuda::tanh_inplace_kernel(target, get_cuda_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::LeakyReLUInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        cuda::leaky_relu_inplace_kernel(target, alpha, get_cuda_stream(attrs));
        return target;
    });

    table.register_inplace_kernel(OpId::GeluInplace, [](Tensor& target, std::span<const Tensor> others, const OpAttributes& attrs) -> Tensor& {
        cuda::gelu_inplace_kernel(target, get_cuda_stream(attrs));
        return target;
    });

    // =========================================================================
    // Reduction Operations
    // =========================================================================
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::sum_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::mean_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::max_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::min_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::argmax_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::argmin_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::prod_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return std::vector<Tensor>{cuda::var_kernel(inputs[0], dim, keepdim, correction, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        int64_t correction = attrs.get_int(AttrKey::Correction, 1);
        return std::vector<Tensor>{cuda::std_kernel(inputs[0], dim, keepdim, correction, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::norm_kernel(inputs[0], p, dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::HasInfNan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::has_inf_nan_kernel(inputs[0], INT64_MIN, false, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Element-wise Math Operations (using direct function pointers)
    // =========================================================================
    table.register_single_output_kernel(OpId::Sqrt, cuda::sqrt_dispatch);
    table.register_single_output_kernel(OpId::Neg, cuda::neg_dispatch);
    table.register_single_output_kernel(OpId::Abs, cuda::abs_dispatch);
    table.register_single_output_kernel(OpId::Sign, cuda::sign_dispatch);
    table.register_single_output_kernel(OpId::Log, cuda::log_dispatch);
    table.register_single_output_kernel(OpId::Exp, cuda::exp_dispatch);
    table.register_single_output_kernel(OpId::Reciprocal, cuda::reciprocal_dispatch);
    table.register_single_output_kernel(OpId::Floor, cuda::floor_dispatch);
    table.register_single_output_kernel(OpId::Ceil, cuda::ceil_dispatch);
    table.register_single_output_kernel(OpId::Round, cuda::round_dispatch);
    table.register_single_output_kernel(OpId::Trunc, cuda::trunc_dispatch);
    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float exponent = static_cast<float>(attrs.get_float(AttrKey::Exponent, 2.0));
        return std::vector<Tensor>{cuda::pow_kernel(inputs[0], exponent, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, -std::numeric_limits<float>::infinity()));
        float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, std::numeric_limits<float>::infinity()));
        return std::vector<Tensor>{cuda::clamp_kernel(inputs[0], min_val, max_val, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = static_cast<float>(attrs.get_float(AttrKey::Min, 0.0));
        return std::vector<Tensor>{cuda::clamp_min_kernel(inputs[0], min_val, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float max_val = static_cast<float>(attrs.get_float(AttrKey::Max, 0.0));
        return std::vector<Tensor>{cuda::clamp_max_kernel(inputs[0], max_val, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Trigonometric Operations (using direct function pointers)
    // =========================================================================
    table.register_single_output_kernel(OpId::Sin, cuda::sin_dispatch);
    table.register_single_output_kernel(OpId::Cos, cuda::cos_dispatch);
    table.register_single_output_kernel(OpId::Tan, cuda::tan_dispatch);
    table.register_single_output_kernel(OpId::Asin, cuda::asin_dispatch);
    table.register_single_output_kernel(OpId::Acos, cuda::acos_dispatch);
    table.register_single_output_kernel(OpId::Atan, cuda::atan_dispatch);
    table.register_single_output_kernel(OpId::Sinh, cuda::sinh_dispatch);
    table.register_single_output_kernel(OpId::Cosh, cuda::cosh_dispatch);
    table.register_single_output_kernel(OpId::Tanh, cuda::tanh_dispatch);

    // =========================================================================
    // Extended Math Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Log2, cuda::log2_dispatch);
    table.register_single_output_kernel(OpId::Log10, cuda::log10_dispatch);
    table.register_single_output_kernel(OpId::Log1p, cuda::log1p_dispatch);
    table.register_single_output_kernel(OpId::Exp2, cuda::exp2_dispatch);
    table.register_single_output_kernel(OpId::Expm1, cuda::expm1_dispatch);
    table.register_single_output_kernel(OpId::Erf, cuda::erf_dispatch);
    table.register_single_output_kernel(OpId::Erfc, cuda::erfc_dispatch);

    // =========================================================================
    // Bool Predicate Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::IsNan, cuda::isnan_dispatch);
    table.register_single_output_kernel(OpId::IsInf, cuda::isinf_dispatch);
    table.register_single_output_kernel(OpId::IsFinite, cuda::isfinite_dispatch);

    // =========================================================================
    // Binary Math Operations (atan2, fmod, remainder)
    // =========================================================================
    table.register_single_output_kernel(OpId::Atan2, cuda::atan2_dispatch);
    table.register_single_output_kernel(OpId::Fmod, cuda::fmod_dispatch);
    table.register_single_output_kernel(OpId::Remainder, cuda::remainder_dispatch);

    // =========================================================================
    // Ternary Operations (lerp)
    // =========================================================================
    table.register_kernel(OpId::Lerp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::lerp_dispatch(inputs, attrs)};
    });

    // =========================================================================
    // Logical Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::LogicalAnd, cuda::logical_and_dispatch);
    table.register_single_output_kernel(OpId::LogicalOr, cuda::logical_or_dispatch);
    table.register_single_output_kernel(OpId::LogicalNot, cuda::logical_not_dispatch);
    table.register_single_output_kernel(OpId::LogicalXor, cuda::logical_xor_dispatch);

    // =========================================================================
    // Element-wise Min/Max Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Minimum, cuda::minimum_dispatch);
    table.register_single_output_kernel(OpId::Maximum, cuda::maximum_dispatch);

    // =========================================================================
    // Complex Number Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Conj, cuda::conj_dispatch);
    table.register_single_output_kernel(OpId::Real, cuda::real_dispatch);
    table.register_single_output_kernel(OpId::Imag, cuda::imag_dispatch);
    table.register_single_output_kernel(OpId::Angle, cuda::angle_dispatch);
    table.register_single_output_kernel(OpId::Polar, cuda::polar_dispatch);

    // =========================================================================
    // Comparison Operations (using direct function pointers)
    // =========================================================================
    table.register_single_output_kernel(OpId::Eq, cuda::eq_dispatch);
    table.register_single_output_kernel(OpId::Ne, cuda::ne_dispatch);
    table.register_single_output_kernel(OpId::Lt, cuda::lt_dispatch);
    table.register_single_output_kernel(OpId::Le, cuda::le_dispatch);
    table.register_single_output_kernel(OpId::Gt, cuda::gt_dispatch);
    table.register_single_output_kernel(OpId::Ge, cuda::ge_dispatch);

    // =========================================================================
    // Activation Functions (simple activations use direct function pointers)
    // =========================================================================
    table.register_single_output_kernel(OpId::ReLU, cuda::relu_dispatch);
    table.register_single_output_kernel(OpId::ReLUBackward, cuda::relu_backward_dispatch);
    table.register_single_output_kernel(OpId::Sigmoid, cuda::sigmoid_dispatch);
    table.register_single_output_kernel(OpId::SigmoidBackward, cuda::sigmoid_backward_dispatch);
    table.register_single_output_kernel(OpId::TanhActivation, cuda::tanh_dispatch);
    table.register_single_output_kernel(OpId::TanhBackward, cuda::tanh_backward_dispatch);
    table.register_single_output_kernel(OpId::Gelu, cuda::gelu_dispatch);
    table.register_single_output_kernel(OpId::GeluBackward, cuda::gelu_backward_dispatch);
    table.register_single_output_kernel(OpId::Swish, cuda::swish_dispatch);
    table.register_single_output_kernel(OpId::SwishBackward, cuda::swish_backward_dispatch);
    table.register_single_output_kernel(OpId::Selu, cuda::selu_dispatch);
    table.register_single_output_kernel(OpId::SeluBackward, cuda::selu_backward_dispatch);
    table.register_single_output_kernel(OpId::Mish, cuda::mish_dispatch);
    table.register_single_output_kernel(OpId::MishBackward, cuda::mish_backward_dispatch);

    // Parameterized activations (keep lambdas for attribute parsing)
    table.register_single_output_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        return cuda::leaky_relu_kernel(inputs[0], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 0.01));
        return cuda::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return cuda::elu_kernel(inputs[0], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = static_cast<float>(attrs.get_float(AttrKey::Alpha, 1.0));
        return cuda::elu_backward_kernel(inputs[0], inputs[1], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
        return cuda::softplus_kernel(inputs[0], beta, threshold, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = static_cast<float>(attrs.get_float(AttrKey::Beta, 1.0));
        float threshold = static_cast<float>(attrs.get_float(AttrKey::Threshold, 20.0));
        return cuda::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, get_cuda_stream(attrs));
    });

    // Softmax operations (use cuDNN when available for better performance)
    // Uses single-output registration for efficiency
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::cudnn_softmax_forward(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::cudnn_softmax_backward(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::cudnn_log_softmax_forward(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::cudnn_log_softmax_backward(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
#else
    table.register_single_output_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::softmax_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::softmax_backward_kernel(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::log_softmax_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::log_softmax_backward_kernel(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
#endif

    // =========================================================================
    // Transform Operations (single-output registration for efficiency)
    // =========================================================================
    table.register_single_output_kernel(OpId::Contiguous, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::contiguous_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Clone, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::clone_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        return cuda::fill_kernel(inputs[0], value, get_cuda_stream(attrs));
    });
    table.register_inplace_kernel(OpId::StridedFill, [](Tensor& self, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        cuda::strided_fill_kernel(self, value, get_cuda_stream(attrs));
        return self;
    });
    table.register_single_output_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        return cuda::reshape_kernel(inputs[0], shape, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        return cuda::expand_kernel(inputs[0], shape, static_cast<void*>(get_cuda_stream(attrs)));
    });
    table.register_single_output_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim0 = attrs.get_int(AttrKey::Dim0, 0);
        int64_t dim1 = attrs.get_int(AttrKey::Dim1, 1);
        return cuda::transpose_kernel(inputs[0], dim0, dim1, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = attrs.get_int_list(AttrKey::Dims);
        return cuda::permute_kernel(inputs[0], dims, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::squeeze_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::unsqueeze_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::cat_kernel(inputs, dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto repeats = attrs.get_int_list(AttrKey::Repeats);
        return cuda::repeat_kernel(inputs[0], repeats, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ToMemoryFormat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int format_int = static_cast<int>(attrs.get_int(AttrKey::MemoryFormat, 0));
        MemoryFormat format = static_cast<MemoryFormat>(format_int);
        return cuda::to_memory_format_kernel(inputs[0], format, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Roll, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t shift = attrs.get_int(AttrKey::Shift, 0);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::roll_kernel(inputs[0], shift, dim, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Indexing Operations (single-output registration for efficiency)
    // =========================================================================
    table.register_single_output_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::index_select_kernel(inputs[0], dim, inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::gather_kernel(inputs[0], dim, inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ScatterAdd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::scatter_add_kernel(inputs[0], dim, inputs[1], inputs[2], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::masked_select_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SearchSorted, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool right = attrs.get_bool(AttrKey::Right, false);
        return cuda::searchsorted_kernel(inputs[0], inputs[1], right, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double value = attrs.get_float(AttrKey::Value, 0.0);
        return cuda::masked_fill_kernel(inputs[0], inputs[1], value, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::where_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Nonzero, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::nonzero_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::OneHot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t num_classes = attrs.get_int(AttrKey::NumClasses, 0);
        return cuda::one_hot_kernel(inputs[0], num_classes, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Pooling Operations (use cuDNN when available for better performance)
    // Note: MaxPool2dForward returns 2 tensors (output + indices) so uses register_kernel
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        auto [output, indices] = cuda::cudnn_maxpool2d_forward(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::cudnn_avgpool2d_forward(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#else
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        auto [output, indices] = cuda::maxpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::avgpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#endif

    // =========================================================================
    // Pooling Backward Operations
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, input, output]
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::cudnn_maxpool2d_backward(inputs[0], inputs[1], inputs[2], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, input]
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::cudnn_avgpool2d_backward(inputs[0], inputs[1], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#else
    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, indices], attrs: input_shape
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output], attrs: input_shape, kernel_size, stride, padding
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::avgpool2d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#endif

    // =========================================================================
    // Normalization Operations
    // =========================================================================
    table.register_kernel(OpId::BatchNorm2dMeanVar, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor mean = tenzor::zeros({inputs[0].shape()[1]}, inputs[0].dtype(), inputs[0].device());
        Tensor variance = tenzor::zeros({inputs[0].shape()[1]}, inputs[0].dtype(), inputs[0].device());
        cuda::batchnorm2d_mean_var(inputs[0], mean, variance, get_cuda_stream(attrs));
        return std::vector<Tensor>{mean, variance};
    });
    table.register_kernel(OpId::BatchNorm2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return std::vector<Tensor>{cuda::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        // Use optimized vectorized kernel for inference (faster than cuDNN due to lower overhead)
        // inputs: [input, mean, variance, gamma, beta]
        return std::vector<Tensor>{cuda::batchnorm2d_forward_affine_optimized(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_cuda_stream(attrs)
        )};
    });
    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        Tensor running_mean = inputs[0];
        Tensor running_var = inputs[1];
        cuda::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, get_cuda_stream(attrs));
        return std::vector<Tensor>{running_mean, running_var};
    });

#ifdef TENZOR_HAS_CUDNN
    // Fused BatchNorm training: computes mean/var, normalizes, and updates running stats in one call
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        Tensor running_mean = inputs[1];
        Tensor running_var = inputs[2];
        auto [output, saved_mean, saved_inv_var] = cuda::cudnn_batchnorm2d_forward_training(
            inputs[0], running_mean, running_var, inputs[3], inputs[4],
            momentum, epsilon, get_cuda_stream(attrs)
        );
        return std::vector<Tensor>{output, running_mean, running_var, saved_mean, saved_inv_var};
    });

    // cuDNN BatchNorm2d backward - significantly faster than separate tensor ops
    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, gamma, saved_mean, saved_inv_var]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [grad_input, grad_gamma, grad_beta] = cuda::cudnn_batchnorm2d_backward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            epsilon, get_cuda_stream(attrs)
        );
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });
#else
    // Fallback: compose batchnorm2d operations when cuDNN is unavailable
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.1));
        auto stream = get_cuda_stream(attrs);

        // Compute batch mean and variance
        Tensor batch_mean = tenzor::zeros({inputs[0].shape()[1]}, inputs[0].dtype(), inputs[0].device());
        Tensor batch_var = tenzor::zeros({inputs[0].shape()[1]}, inputs[0].dtype(), inputs[0].device());
        cuda::batchnorm2d_mean_var(inputs[0], batch_mean, batch_var, stream);

        // Forward with affine transform
        Tensor output = cuda::batchnorm2d_forward_affine(inputs[0], batch_mean, batch_var, inputs[3], inputs[4], epsilon, stream);

        // Update running stats
        Tensor running_mean = inputs[1];
        Tensor running_var = inputs[2];
        cuda::batchnorm2d_update_running_stats(running_mean, running_var, batch_mean, batch_var, momentum, stream);

        // Compute saved_inv_var for backward pass
        // inv_var = 1 / sqrt(var + eps) — computed on device via existing kernels
        // For simplicity, use the batch_var directly (backward will recompute inv_var)
        return std::vector<Tensor>{output, running_mean, running_var, batch_mean, batch_var};
    });

    // Custom CUDA kernel backward - fallback when cuDNN is not available
    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, mean, variance, gamma]
        float epsilon = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [grad_input, grad_gamma, grad_beta] = cuda::batchnorm2d_backward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            epsilon, get_cuda_stream(attrs)
        );
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });
#endif

    // =========================================================================
    // Fused LayerNorm (optimized with warp shuffles and vectorized loads)
    // =========================================================================
    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, weight, bias]
        // attrs: normalized_shape (comma-separated), eps
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);

#ifdef TENZOR_HAS_CUDNN
        // Use optimized kernel with warp shuffles (2x faster than naive)
        auto [output, mean, inv_std] = cuda::cudnn_layer_norm_forward(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps, get_cuda_stream(attrs)
        );
#else
        // Fallback to basic fused kernel
        auto [output, mean, inv_std] = cuda::fused_layer_norm_cuda(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps
        );
#endif
        return std::vector<Tensor>{output, mean, inv_std};
    });

    // =========================================================================
    // Fused RMSNorm (single kernel launch for maximum performance)
    // =========================================================================
    table.register_kernel(OpId::FusedRMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, weight]
        // attrs: eps
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, rrms] = cuda::fused_rms_norm_cuda(inputs[0], inputs[1], eps);
        return std::vector<Tensor>{output, rrms};
    });

    // =========================================================================
    // Fused Attention (single kernel launch for maximum performance)
    // =========================================================================
    table.register_kernel(OpId::FusedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [Q, K, V]
        //   - 4D: (batch, num_heads, seq_len, head_dim) for cuDNN SDPA
        //   - 3D: (batch_heads, seq_len, head_dim) for custom kernel
        // attrs: scale, use_cudnn_sdpa (optional)
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));

#ifdef TENZOR_HAS_CUDNN_FRONTEND
        // Check if cuDNN SDPA is requested and input is 4D
        bool use_cudnn_sdpa = attrs.get_bool(AttrKey::UseCudnnSdpa, false);
        if (use_cudnn_sdpa && inputs[0].shape().size() == 4) {
            // 4D input: use cuDNN SDPA directly
            auto output = cuda::cudnn_sdpa_forward(inputs[0], inputs[1], inputs[2], scale);
            return std::vector<Tensor>{output};
        }
#endif

        // 3D input or cuDNN not available: use custom flash attention kernel
        auto output = cuda::fused_attention_cuda(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output};
    });

    // =========================================================================
    // Flash Attention (memory-efficient tiled attention)
    // =========================================================================
    table.register_kernel(OpId::FlashAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Uses same implementation as FusedAttention — both are memory-efficient
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        auto output = cuda::fused_attention_cuda(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output};
    });

    // =========================================================================
    // Fused LayerNorm Backward
    // =========================================================================
    table.register_kernel(OpId::FusedLayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        auto [grad_input, grad_weight, grad_bias] = cuda::fused_layer_norm_backward_cuda(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    // =========================================================================
    // Linear Layer (fused cuBLAS GEMM + bias for 2-3x speedup)
    // =========================================================================
    table.register_single_output_kernel(OpId::Linear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, weight] or [input, weight, bias]
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::linear_kernel(inputs[0], inputs[1], bias, get_cuda_stream(attrs));
    });

    table.register_kernel(OpId::LinearBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight]
        return cuda::linear_backward_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs));
    });

    // =========================================================================
    // Fused SGD Optimizer Step (single kernel launch for all SGD operations)
    // =========================================================================
    table.register_kernel(OpId::FusedSGDStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [param, grad, momentum_buffer (optional)]
        // attrs: lr, momentum, weight_decay, dampening, nesterov
        // Note: param and momentum_buffer are modified in-place
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float momentum = static_cast<float>(attrs.get_float(AttrKey::Momentum, 0.0));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        float dampening = static_cast<float>(attrs.get_float(AttrKey::Dampening, 0.0));
        bool nesterov = attrs.get_bool(AttrKey::Nesterov, false);

        // Cast away const for in-place modification (safe because we control the API)
        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor* momentum_buffer = (inputs.size() > 2 && momentum > 0.0f)
            ? &const_cast<Tensor&>(inputs[2]) : nullptr;

        cuda::fused_sgd_step_cuda(
            param, inputs[1], momentum_buffer,
            lr, momentum, weight_decay, dampening, nesterov,
            get_cuda_stream(attrs)
        );
        return std::vector<Tensor>{param};  // Return modified param
    });

    // =========================================================================
    // Fused Adam Optimizer Step (single kernel launch for all Adam operations)
    // =========================================================================
    table.register_kernel(OpId::FusedAdamStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [param, grad, exp_avg, exp_avg_sq, packed_params, max_exp_avg_sq (optional)]
        // packed_params is a CPU Float64 tensor: [lr, beta1, beta2, eps, weight_decay, step, decoupled, amsgrad]
        double lr, beta1, beta2, eps, weight_decay;
        int64_t step;
        bool decoupled, amsgrad;

        if (inputs.size() >= 5 && inputs[4].dtype() == DType::Float64 && inputs[4].numel() == 8) {
            // New packed-tensor path: read typed values directly (no string parsing)
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
            // Legacy string-attribute path (backwards compatibility)
            lr = attrs.get_float(AttrKey::Lr, 0.001);
            beta1 = attrs.get_float(AttrKey::Beta1, 0.9);
            beta2 = attrs.get_float(AttrKey::Beta2, 0.999);
            eps = attrs.get_float(AttrKey::Eps, 1e-8);
            weight_decay = attrs.get_float(AttrKey::WeightDecay, 0.0);
            step = attrs.get_int(AttrKey::Step, 1);
            decoupled = attrs.get_bool(AttrKey::Decoupled, false);
            amsgrad = attrs.get_bool(AttrKey::Amsgrad, false);
        }

        // Cast away const for in-place modification
        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& exp_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& exp_avg_sq = const_cast<Tensor&>(inputs[3]);
        // max_exp_avg_sq follows packed_params tensor (index 5) if amsgrad
        Tensor* max_exp_avg_sq = (amsgrad && inputs.size() > 5)
            ? &const_cast<Tensor&>(inputs[5]) : nullptr;

        cuda::fused_adam_step_cuda(
            param, inputs[1], exp_avg, exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, decoupled,
            get_cuda_stream(attrs), max_exp_avg_sq, amsgrad
        );
        return std::vector<Tensor>{param};  // Return modified param
    });

    // =========================================================================
    // Embedding Operations (lookup table for token IDs)
    // =========================================================================
    table.register_single_output_kernel(OpId::Embedding, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [weight, indices]
        return cuda::embedding_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    // EmbeddingWithBoundsCheck — CUDA embedding already uses error flag + atomicExch for OOB detection
    table.register_single_output_kernel(OpId::EmbeddingWithBoundsCheck, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::embedding_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, indices]
        // attrs: num_embeddings
        int64_t num_embeddings = attrs.get_int(AttrKey::NumEmbeddings, 0);
        return cuda::embedding_backward_kernel(inputs[0], inputs[1], num_embeddings, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::EmbeddingBagForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [embeddings, offsets]
        // attrs: Mode, EmbeddingDim, IncludeLastOffset
        std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
        int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
        bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);
        return cuda::embedding_bag_forward_kernel(inputs[0], inputs[1], mode, embedding_dim, include_last_offset, get_cuda_stream(attrs));
    });

    table.register_kernel(OpId::EmbeddingBagBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, embeddings, offsets]
        return std::vector<Tensor>{cuda::embedding_bag_backward_kernel(
            inputs[0], inputs[1], inputs[2], attrs, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Fused Operations (optimized combined kernels)
    // =========================================================================
    table.register_single_output_kernel(OpId::FusedConv2dBnReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* bias = inputs.size() > 2 && inputs[2].numel() > 0 ? &inputs[2] : nullptr;
        // CPU registration: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        // CUDA func expects: (input, weight, bias, bn_mean, bn_var, bn_gamma, bn_beta, ...)
        return cuda::fused_conv2d_bn_relu_cuda(inputs[0], inputs[1], bias,
            inputs[5], inputs[6], inputs[3], inputs[4], stride, padding, eps);
    });

    table.register_single_output_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // inputs: [input, weight] or [input, weight, bias]
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::fused_linear_relu_cuda(inputs[0], inputs[1], bias);
    });

    table.register_single_output_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, running_mean, running_var, weight, bias]
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        return cuda::fused_batchnorm_relu_cuda(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps);
    });

    table.register_single_output_kernel(OpId::FusedAddReLU, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // inputs: [a, b]
        return cuda::fused_add_relu_cuda(inputs[0], inputs[1]);
    });

    table.register_single_output_kernel(OpId::FusedGelu, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // inputs: [input]
        return cuda::fused_gelu_cuda(inputs[0]);
    });

#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_relu_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_sigmoid_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_tanh_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_swish_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
#else
    // Fallback: compose conv2d + activation when cuDNN is unavailable
    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
        return cuda::relu_kernel(result, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
        return cuda::sigmoid_kernel(result, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
        return cuda::tanh_kernel(result, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
        return cuda::swish_kernel(result, get_cuda_stream(attrs));
    });
#endif

    // =========================================================================
    // Vision/Interpolation Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input]
        // attrs: size (comma-separated), mode, align_corners
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cuda::interpolate_cuda(inputs[0], size, mode, align_corners);
    });

    // =========================================================================
    // ROI Align Operations
    // =========================================================================
    table.register_kernel(OpId::ROIAlignForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // inputs: [features, rois]
        // attrs: output_h, output_w, spatial_scale, sampling_ratio, aligned
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 7);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 7);
        float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
        int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
        bool aligned = attrs.get_bool(AttrKey::Aligned, true);

        return {cuda::roi_align_forward(inputs[0], inputs[1],
                                        output_h, output_w, spatial_scale,
                                        sampling_ratio, aligned)};
    });

    table.register_kernel(OpId::ROIAlignBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // inputs: [grad_output, rois]
        // attrs: batch_size, feat_height, feat_width, spatial_scale, sampling_ratio, aligned
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 1);
        int64_t feat_height = attrs.get_int(AttrKey::FeatHeight, 0);
        int64_t feat_width = attrs.get_int(AttrKey::FeatWidth, 0);
        float spatial_scale = static_cast<float>(attrs.get_float(AttrKey::SpatialScale, 1.0 / 16.0));
        int64_t sampling_ratio = attrs.get_int(AttrKey::SamplingRatio, 0);
        bool aligned = attrs.get_bool(AttrKey::Aligned, true);

        return {cuda::roi_align_backward(inputs[0], inputs[1],
                                         batch_size, feat_height, feat_width,
                                         spatial_scale, sampling_ratio, aligned)};
    });

    // =========================================================================
    // Conv2d Operations (Phase 1A - CRITICAL)
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, weight] or [input, weight, bias]
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv2d_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });
#else
    table.register_single_output_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });
#endif

    table.register_single_output_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::depthwise_conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Conv3d Operations (cuDNN Nd)
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv3d_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });

    // =========================================================================
    // ConvTranspose3d Operations (cuDNN Nd)
    // =========================================================================
    table.register_single_output_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv_transpose3d_forward(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv_transpose3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv_transpose3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv_transpose3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });
#endif // TENZOR_HAS_CUDNN (Conv3d/ConvTranspose3d)

    // =========================================================================
    // Dropout Operations (Phase 1B - CRITICAL)
    // =========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        bool training = attrs.get_bool(AttrKey::Training, true);
        auto [output, mask] = cuda::dropout_forward_kernel(inputs[0], p, training, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mask};
    });
    table.register_single_output_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 0.5));
        return cuda::dropout_backward_kernel(inputs[0], inputs[1], p, get_cuda_stream(attrs));
    });

    // =========================================================================
    // LSTM Operations (Phase 1C - HIGH)
    // =========================================================================
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [gates, c_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [h_out, c_out] = cuda::lstm_cell_forward_kernel(inputs[0], inputs[1], batch_size, hidden_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{h_out, c_out};
    });
    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, grad_c, gates, c_prev, c_out]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [grad_gates, grad_c_prev] = cuda::lstm_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{grad_gates, grad_c_prev};
    });

    // =========================================================================
    // GRU Operations (Phase 1C - HIGH)
    // =========================================================================
    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [reset_gates, update_gates, new_gates_input, new_gates_hidden, h_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto h_out = cuda::gru_cell_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{h_out};
    });
    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden, h_prev]
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto result = cuda::gru_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            batch_size, hidden_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{result.grad_reset, result.grad_update, result.grad_new_input, result.grad_new_hidden, result.grad_h_prev};
    });

    // =========================================================================
    // LayerNorm (non-fused) Operations (Phase 1D - HIGH)
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, weight, bias]
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        auto [output, mean, inv_std] = cuda::cudnn_layer_norm_forward(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, mean, inv_std, weight]
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_layer_norm_backward(
            inputs[0], inputs[1], inputs[4], inputs[2], inputs[3], normalized_shape, get_cuda_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });
#else
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        auto [output, mean, inv_std] = cuda::fused_layer_norm_cuda(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps);
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, mean, inv_std, weight]
        auto normalized_shape = attrs.get_int_list(AttrKey::NormalizedShape);
        auto [grad_input, grad_weight, grad_bias] = cuda::fused_layer_norm_backward_cuda(
            inputs[0], inputs[1], inputs[4], inputs[2], inputs[3], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });
#endif

    // =========================================================================
    // Adaptive Pooling Operations (Phase 1E / 3D - HIGH)
    // =========================================================================
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cuda::adaptive_avg_pool2d_forward(inputs[0], output_h, output_w, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Autograd sends "H_in" and "W_in"; also support "input_h"/"input_w" for compatibility
        int64_t H_in = attrs.get_int(AttrKey::InputH, 0);
        int64_t W_in = attrs.get_int(AttrKey::InputW, 0);
        return cuda::adaptive_avg_pool2d_backward(inputs[0], H_in, W_in, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        auto [output, indices] = cuda::adaptive_max_pool2d_forward(inputs[0], output_h, output_w, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::adaptive_max_pool2d_backward(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // 1D Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        auto [output, indices] = cuda::maxpool1d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::maxpool1d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::avgpool1d_forward_kernel(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::avgpool1d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding, get_cuda_stream(attrs));
    });

    table.register_kernel(OpId::AdaptiveMaxPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
        auto [output, indices] = cuda::adaptive_maxpool1d_forward(inputs[0], output_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::adaptive_maxpool1d_backward(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool1d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_size = attrs.get_int(AttrKey::OutputSize, 1);
        return cuda::adaptive_avgpool1d_forward(inputs[0], output_size, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::adaptive_avgpool1d_backward(inputs[0], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // 3D Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        auto [output, indices] = cuda::maxpool3d_forward_kernel(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::avgpool3d_forward_kernel(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::avgpool3d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding, get_cuda_stream(attrs));
    });

    table.register_kernel(OpId::AdaptiveMaxPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        auto [output, indices] = cuda::adaptive_maxpool3d_forward(inputs[0], output_d, output_h, output_w, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::AdaptiveMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::adaptive_maxpool3d_backward(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool3d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t output_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t output_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cuda::adaptive_avgpool3d_forward(inputs[0], output_d, output_h, output_w, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AdaptiveAvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::adaptive_avgpool3d_backward(inputs[0], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // GroupNorm / InstanceNorm Operations (Phase 3A-B - HIGH)
    // =========================================================================
    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, weight, bias]
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, mean, inv_std] = cuda::group_norm_forward_kernel(
            inputs[0], inputs[1], inputs[2], num_groups, eps, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::group_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], num_groups, get_cuda_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });
    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, mean, inv_std] = cuda::instance_norm_forward_kernel(
            inputs[0], inputs[1], inputs[2], eps, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::InstanceNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto [grad_input, grad_weight, grad_bias] = cuda::instance_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], get_cuda_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    // =========================================================================
    // RMSNorm Backward (Phase 3C - HIGH)
    // =========================================================================
    table.register_kernel(OpId::RMSNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, rrms]
        auto [grad_input, grad_weight] = cuda::fused_rms_norm_backward_cuda(
            inputs[0], inputs[1], inputs[2], inputs[3]);
        return std::vector<Tensor>{grad_input, grad_weight};
    });

    // =========================================================================
    // ArgSort - GPU implementation (Phase 2D - HIGH)
    // =========================================================================
    table.register_single_output_kernel(OpId::ArgSort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        return cuda::argsort_kernel(inputs[0], dim, descending, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Creation Operations (Phase 3E - MEDIUM)
    // =========================================================================
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        Tensor output(shape, dtype, device);
        cudaMemsetAsync(output.data_ptr(), 0, output.numel() * dtype_size(dtype), get_cuda_stream(attrs));
        return std::vector<Tensor>{output};
    });
    table.register_kernel(OpId::Ones, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        Tensor output(shape, dtype, device);
        return std::vector<Tensor>{cuda::fill_kernel(output, 1.0f, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Full, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        float value = static_cast<float>(attrs.get_float(AttrKey::Value, 0.0));
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        Tensor output(shape, dtype, device);
        return std::vector<Tensor>{cuda::fill_kernel(output, value, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Rand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::rand_kernel(shape, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Randn, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::randn_kernel(shape, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Randint, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t low = attrs.get_int(AttrKey::Start, 0);
        int64_t high = attrs.get_int(AttrKey::End, 0);
        auto shape = attrs.get_int_list(AttrKey::Shape);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "int32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::randint_kernel(low, high, shape, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Arange, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
        float end = static_cast<float>(attrs.get_float(AttrKey::End, 0.0));
        float step = static_cast<float>(attrs.get_float(AttrKey::Step, 1.0));
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::arange_kernel(start, end, step, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Linspace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float start = static_cast<float>(attrs.get_float(AttrKey::Start, 0.0));
        float end = static_cast<float>(attrs.get_float(AttrKey::End, 1.0));
        int64_t steps = attrs.get_int(AttrKey::Steps, 100);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::linspace_kernel(start, end, steps, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Eye, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = attrs.get_int(AttrKey::N, 0);
        int64_t m = attrs.get_int(AttrKey::M, -1);
        DType dtype = dtype_from_string(attrs.get_string(AttrKey::Dtype, "float32"));
        int device_idx = static_cast<int>(attrs.get_int(AttrKey::Device, 0));
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::eye_kernel(n, m, dtype, device, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Shape/Transform Operations (Phase 3F - MEDIUM)
    // =========================================================================
    table.register_single_output_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t start_dim = attrs.get_int(AttrKey::StartDim, 0);
        int64_t end_dim = attrs.get_int(AttrKey::EndDim, -1);
        return cuda::flatten_kernel(inputs[0], start_dim, end_dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto starts = attrs.get_int_list(AttrKey::Starts);
        auto ends = attrs.get_int_list(AttrKey::Ends);
        auto steps = attrs.get_int_list(AttrKey::Steps);
        return cuda::slice_kernel(inputs[0], starts, ends, steps, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::stack_kernel(inputs, dim, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Split, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t split_size = attrs.get_int(AttrKey::SplitSize, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::split_kernel(inputs[0], split_size, dim, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Chunk, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t chunks = attrs.get_int(AttrKey::Chunks, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::chunk_kernel(inputs[0], chunks, dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto reps = attrs.get_int_list(AttrKey::Reps);
        return cuda::tile_kernel(inputs[0], reps, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Any/All Reductions
    // =========================================================================
    table.register_kernel(OpId::Any, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::any_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::All, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::all_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LogSumExp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::logsumexp_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Triangular / Diagonal / Trace / Flip Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Triu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return cuda::triu_kernel(inputs[0], diagonal, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tril, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return cuda::tril_kernel(inputs[0], diagonal, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Diag, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t diagonal = attrs.get_int(AttrKey::Diagonal, 0);
        return cuda::diag_kernel(inputs[0], diagonal, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Trace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::trace_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Flip, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::flip_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Fused Softmax Cross Entropy (Phase 4A - MEDIUM)
    // =========================================================================
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [logits, targets]
        bool compute_grad = attrs.get_bool(AttrKey::ComputeGrad, true);
        auto [loss, grad_logits] = cuda::fused_softmax_cross_entropy_cuda(inputs[0], inputs[1], compute_grad);
        if (compute_grad) {
            return std::vector<Tensor>{loss, grad_logits};
        }
        return std::vector<Tensor>{loss};
    });

    // =========================================================================
    // Fused Optimizer Steps (Phase 4C - MEDIUM)
    // =========================================================================
    table.register_kernel(OpId::FusedRMSPropStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
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

        cuda::fused_rmsprop_step_cuda(param, inputs[1], square_avg, grad_avg, momentum_buffer,
            lr, alpha, eps, weight_decay, momentum, centered, get_cuda_stream(attrs));
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdadeltaStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [param, grad, square_avg, acc_delta]
        float rho = static_cast<float>(attrs.get_float(AttrKey::Rho, 0.9));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-6));
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 1.0));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& square_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& acc_delta = const_cast<Tensor&>(inputs[3]);

        cuda::fused_adadelta_step_cuda(param, inputs[1], square_avg, acc_delta,
            rho, eps, lr, weight_decay, get_cuda_stream(attrs));
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdagradStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [param, grad, sum_sq]
        float lr = static_cast<float>(attrs.get_float(AttrKey::Lr, 0.01));
        float lr_decay = static_cast<float>(attrs.get_float(AttrKey::LrDecay, 0.0));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-10));
        float weight_decay = static_cast<float>(attrs.get_float(AttrKey::WeightDecay, 0.0));
        int64_t step = attrs.get_int(AttrKey::Step, 1);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& sum_sq = const_cast<Tensor&>(inputs[2]);

        cuda::fused_adagrad_step_cuda(param, inputs[1], sum_sq,
            lr, lr_decay, eps, weight_decay, step, get_cuda_stream(attrs));
        return std::vector<Tensor>{param};
    });

    // =========================================================================
    // Fused Adam-Atan2 Optimizer Step
    // =========================================================================
    table.register_kernel(OpId::FusedAdamAtan2Step, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq (optional)]
        // attrs: lr, beta1, beta2, eps, weight_decay, step, amsgrad
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

        cuda::fused_adam_atan2_step_cuda(
            param, inputs[1], exp_avg, exp_avg_sq, max_exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, amsgrad,
            get_cuda_stream(attrs)
        );
        return std::vector<Tensor>{param};
    });

    // =========================================================================
    // Full-Sequence RNN Operations
    // =========================================================================

    // inputs: [input, W_ih, W_hh, bias_ih, bias_hh, h0, c0]
    table.register_kernel(OpId::LSTMForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cuda::lstm_forward_cuda(inputs[0], inputs[1], inputs[2],
                                       inputs[3], inputs[4], inputs[5], inputs[6]);
    });

    // inputs: [input, W_ih, W_hh, bias, h0]
    table.register_kernel(OpId::GRUForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cuda::gru_forward_cuda(inputs[0], inputs[1], inputs[2],
                                      inputs[3], inputs[4]);
    });

    // inputs: [input, h0, c0, W_ih_0, W_hh_0, bias_0, W_ih_1, W_hh_1, bias_1, ...]
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

        return cuda::lstm_multi_layer_forward_cuda(input, W_ih_list, W_hh_list, bias_list, h0, c0);
    });

    // inputs: [input, h0, W_ih_0, W_hh_0, bias_0, W_ih_1, W_hh_1, bias_1, ...]
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

        return cuda::gru_multi_layer_forward_cuda(input, W_ih_list, W_hh_list, bias_list, h0);
    });

    // inputs: [input, h0, c0, W_ih_fwd, W_hh_fwd, bias_ih_fwd, bias_hh_fwd,
    //          W_ih_bwd, W_hh_bwd, bias_ih_bwd, bias_hh_bwd]
    table.register_kernel(OpId::BiLSTMForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cuda::bilstm_forward_cuda(
            inputs[0],
            inputs[3], inputs[4], inputs[5], inputs[6],
            inputs[7], inputs[8], inputs[9], inputs[10],
            inputs[1], inputs[2]
        );
    });

    // =========================================================================
    // Take / Put Operations
    // =========================================================================

    // inputs: [input, indices]
    table.register_single_output_kernel(OpId::Take, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::take_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    // inputs: [input, indices, source]
    // attrs: accumulate (bool)
    table.register_single_output_kernel(OpId::Put, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool accumulate = attrs.get_bool(AttrKey::Accumulate, false);
        Tensor input = inputs[0];
        return cuda::put_kernel(input, inputs[1], inputs[2], accumulate, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Unfold / Fold Operations
    // =========================================================================

    // inputs: [input]
    // attrs: kernel_size, stride, padding, dilation
    table.register_single_output_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        return cuda::unfold_cuda(inputs[0], kernel_size, stride, padding, dilation);
    });

    // inputs: [input]
    // attrs: output_size, kernel_size, stride, padding, dilation
    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        return cuda::fold_cuda(inputs[0], output_size, kernel_size, stride, padding, dilation);
    });

    // =========================================================================
    // BoxIoU Operation
    // =========================================================================

    // inputs: [boxes1, boxes2]
    // attrs: iou_type (0=IoU, 1=GIoU)
    table.register_single_output_kernel(OpId::BoxIoU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int iou_type = static_cast<int>(attrs.get_int(AttrKey::IouType, 0));
        return cuda::box_iou_cuda(inputs[0], inputs[1], iou_type);
    });

    table.register_kernel(OpId::GatherRelativePositionBias,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_positions = attrs.get_int(AttrKey::NumPositions, 0);
            int64_t num_heads = attrs.get_int(AttrKey::NumHeads, 0);
            return {cuda::gather_relative_position_bias(inputs[0], inputs[1], num_positions, num_heads)};
        });

    // =========================================================================
    // Advanced Operations (topk, sort, cumsum, cumprod, unique)
    // =========================================================================
    table.register_kernel(OpId::TopK, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool largest = attrs.get_bool(AttrKey::Largest, true);
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        auto [values, indices] = cuda::topk_kernel(inputs[0], k, dim, largest, sorted, get_cuda_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });
    table.register_kernel(OpId::Sort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool descending = attrs.get_bool(AttrKey::Descending, false);
        auto [values, indices] = cuda::sort_kernel(inputs[0], dim, descending, get_cuda_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });
    table.register_single_output_kernel(OpId::CumSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::cumsum_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::CumProd, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::cumprod_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Unique, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool sorted = attrs.get_bool(AttrKey::Sorted, true);
        bool return_inverse = attrs.get_bool(AttrKey::ReturnInverse, false);
        bool return_counts = attrs.get_bool(AttrKey::ReturnCounts, false);
        auto [unique_vals, inverse, counts] = cuda::unique_kernel(inputs[0], sorted, return_inverse, return_counts, get_cuda_stream(attrs));
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    // =========================================================================
    // Type Conversion Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Cast, cuda::cast_dispatch);

    // =========================================================================
    // Linear Algebra Operations (cuSOLVER)
    // =========================================================================
#ifdef TENZOR_HAS_CUSOLVER
    table.register_single_output_kernel(OpId::LinalgDet, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::linalg_det_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LinalgInv, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::linalg_inv_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LinalgSolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::linalg_solve_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::LinalgSVD, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        bool full_matrices = attrs.get_bool(AttrKey::FullMatrices, true);
        auto [U, S, Vt] = cuda::linalg_svd_kernel(inputs[0], full_matrices, get_cuda_stream(attrs));
        return {U, S, Vt};
    });
    table.register_kernel(OpId::LinalgQR, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [Q, R] = cuda::linalg_qr_kernel(inputs[0], get_cuda_stream(attrs));
        return {Q, R};
    });
    table.register_kernel(OpId::LinalgEigh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [W, V] = cuda::linalg_eigh_kernel(inputs[0], get_cuda_stream(attrs));
        return {W, V};
    });
    table.register_kernel(OpId::LinalgEig, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [WR, WI, V] = cuda::linalg_eig_kernel(inputs[0], get_cuda_stream(attrs));
        return {WR, WI, V};
    });
    table.register_single_output_kernel(OpId::LinalgCholesky, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        return cuda::linalg_cholesky_kernel(inputs[0], upper, get_cuda_stream(attrs));
    });
#endif // TENZOR_HAS_CUSOLVER

    // =========================================================================
    // FFT Operations (cuFFT)
    // =========================================================================
#ifdef TENZOR_HAS_CUFFT
    table.register_single_output_kernel(OpId::FFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return cuda::cuda_fft_kernel(inputs[0], dim, n, norm, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::IFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return cuda::cuda_ifft_kernel(inputs[0], dim, n, norm, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::RFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim]);
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return cuda::cuda_rfft_kernel(inputs[0], dim, n, norm, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::IRFFT, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        int64_t n = attrs.get_int(AttrKey::N, 2 * (inputs[0].shape()[dim >= 0 ? dim : inputs[0].ndim() + dim] - 1));
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return cuda::cuda_irfft_kernel(inputs[0], dim, n, norm, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FFT2, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // FFT2 operates on last 2 dims by default
        int64_t ndim = inputs[0].ndim();
        std::vector<int64_t> dims = {ndim - 2, ndim - 1};
        std::vector<int64_t> n_vec = {
            inputs[0].shape()[dims[0]],
            inputs[0].shape()[dims[1]]
        };
        // Override with attrs if provided
        auto attr_n = attrs.get_int_list(AttrKey::Shape);
        if (!attr_n.empty() && attr_n.size() >= 2) {
            n_vec[0] = attr_n[0];
            n_vec[1] = attr_n[1];
        }
        std::string norm(attrs.get_string(AttrKey::Norm, "backward"));
        return cuda::cuda_fft2_kernel(inputs[0], dims, n_vec, norm, get_cuda_stream(attrs));
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
        return cuda::cuda_ifft2_kernel(inputs[0], dims, n_vec, norm, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FFTN, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ndim = inputs[0].ndim();
        // Default: all dimensions
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
        return cuda::cuda_fftn_kernel(inputs[0], dims, n_vec, norm, get_cuda_stream(attrs));
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
        return cuda::cuda_ifftn_kernel(inputs[0], dims, n_vec, norm, get_cuda_stream(attrs));
    });
#endif // TENZOR_HAS_CUFFT

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
        auto stream = get_cuda_stream(attrs);

        Tensor output({batch_size, out_features}, DType::Float32, input.device());

        const int8_t* input_data = input.data<int8_t>();
        const int8_t* weight_data = weight.data<int8_t>();
        const float* bias_data = nullptr;
        if (inputs.size() > 2 && inputs[2].numel() > 0) {
            bias_data = inputs[2].data<const float>();
        }
        float* output_data = output.data<float>();

        nn::quantization::kernels::quantized_linear_cuda(
            input_data, weight_data, bias_data, output_data,
            batch_size, in_features, out_features,
            input_scale, weight_scale, output_scale,
            input_zp, weight_zp, stream
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
        auto stream = get_cuda_stream(attrs);

        int64_t h_out = (h_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;
        int64_t w_out = (w_in + 2 * padding - dilation * (kernel_size - 1) - 1) / stride + 1;

        Tensor output({batch, out_channels, h_out, w_out}, DType::Float32, input.device());

        const int8_t* input_data = input.data<int8_t>();
        const int8_t* weight_data = weight.data<int8_t>();
        const float* bias_data = nullptr;
        if (inputs.size() > 2 && inputs[2].numel() > 0) {
            bias_data = inputs[2].data<const float>();
        }
        float* output_data = output.data<float>();

        nn::quantization::kernels::quantized_conv2d_cuda(
            input_data, weight_data, bias_data, output_data,
            batch, in_channels, out_channels,
            h_in, w_in, h_out, w_out,
            kernel_size, stride, padding,
            input_scale, weight_scale,
            input_zp, weight_zp, stream
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
}

} // namespace tenzor

// Export function for dynamic loading
extern "C" {
    void register_kernels(tenzor::BackendDispatchTable* table) {
        if (table) {
            tenzor::register_cuda_kernels(*table);
        }
    }
}
