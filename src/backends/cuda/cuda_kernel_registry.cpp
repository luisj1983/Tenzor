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
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/backend/fused_ops.hpp"
#ifdef TENZOR_HAS_CUDNN
#include "tenzor/backend/cudnn_wrapper.hpp"
#endif
#include <cuda_runtime.h>
#include <cstdlib>
#include <limits>
#include <sstream>
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
    if (s == "complex64") return DType::Complex64;
    if (s == "complex128") return DType::Complex128;
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
    auto addmm_kernel(const Tensor& input, const Tensor& mat1, const Tensor& mat2,
                      double alpha, double beta, cudaStream_t stream) -> Tensor;
    auto addmv_kernel(const Tensor& input, const Tensor& mat, const Tensor& vec,
                      double alpha, double beta, cudaStream_t stream) -> Tensor;
    auto baddbmm_kernel(const Tensor& input, const Tensor& batch1, const Tensor& batch2,
                        double alpha, double beta, cudaStream_t stream) -> Tensor;

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
    auto repeat_interleave_scalar_kernel(const Tensor& input, int64_t repeats, int64_t dim, cudaStream_t stream) -> Tensor;
    auto repeat_interleave_tensor_kernel(const Tensor& input, const Tensor& repeats, int64_t dim, cudaStream_t stream) -> Tensor;

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
    auto take_along_dim_kernel(const Tensor& input, const Tensor& indices, int64_t dim, cudaStream_t stream) -> Tensor;
    auto masked_scatter_kernel(const Tensor& input, const Tensor& mask, const Tensor& source, cudaStream_t stream) -> Tensor;
    auto tril_indices_kernel(int64_t row, int64_t col, int64_t offset, cudaStream_t stream) -> Tensor;
    auto triu_indices_kernel(int64_t row, int64_t col, int64_t offset, cudaStream_t stream) -> Tensor;

    // Embedding operations
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, cudaStream_t stream) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices, int64_t num_embeddings, cudaStream_t stream) -> Tensor;
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets, const std::string& mode, int64_t embedding_dim, bool include_last_offset, cudaStream_t stream) -> Tensor;
    auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& embeddings, const Tensor& offsets, const OpAttributes& attrs, cudaStream_t stream) -> Tensor;

    // Linear algebra operations (cuSOLVER or native CUDA fallback)
    auto linalg_det_kernel(const Tensor& A, cudaStream_t stream) -> Tensor;
    auto linalg_inv_kernel(const Tensor& A, cudaStream_t stream) -> Tensor;
    auto linalg_solve_kernel(const Tensor& A, const Tensor& B, cudaStream_t stream) -> Tensor;
    auto linalg_svd_kernel(const Tensor& A, bool full_matrices, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_qr_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto linalg_eigh_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto linalg_eig_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_cholesky_kernel(const Tensor& A, bool upper, cudaStream_t stream) -> Tensor;
    auto linalg_lu_kernel(const Tensor& A, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                                const Tensor& B, cudaStream_t stream) -> Tensor;
    auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                         bool upper, bool unitriangular,
                                         cudaStream_t stream) -> Tensor;
    auto linalg_geqrf_kernel(const Tensor& A, cudaStream_t stream)
        -> std::tuple<Tensor, Tensor>;
    auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                              const Tensor& C, bool left, bool transpose_q,
                              cudaStream_t stream) -> Tensor;
    auto linalg_ldl_factor_kernel(const Tensor& A, cudaStream_t stream)
        -> std::tuple<Tensor, Tensor>;
    auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                                  const Tensor& B, cudaStream_t stream) -> Tensor;
    auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                    cudaStream_t stream) -> Tensor;
    // FFT operations (cuFFT or native Cooley-Tukey + Bluestein fallback)
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
    auto grid_sample_cuda(const Tensor& input, const Tensor& grid,
                          const std::string& mode, const std::string& padding_mode,
                          bool align_corners) -> Tensor;
    auto affine_grid_cuda(const Tensor& theta, const std::vector<int64_t>& size,
                          bool align_corners) -> Tensor;
    auto interpolate_cuda(const Tensor& input, const std::vector<int64_t>& size, const std::string& mode, bool align_corners) -> Tensor;
    auto unfold_cuda(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> Tensor;
    auto fold_cuda(const Tensor& input, const std::vector<int64_t>& output_size, int64_t kernel_size, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> Tensor;
    auto box_iou_cuda(const Tensor& boxes1, const Tensor& boxes2, int iou_type) -> Tensor;
    auto nms_cuda_wrapper(const Tensor& boxes, const Tensor& scores, float iou_threshold) -> Tensor;
    auto gather_relative_position_bias(const Tensor& table, const Tensor& indices,
                                       int64_t num_positions, int64_t num_heads) -> Tensor;

    // Advanced operations (topk, sort, cumsum, cumprod, unique)
    auto topk_kernel(const Tensor& input, int64_t k, int64_t dim, bool largest, bool sorted, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto sort_kernel(const Tensor& input, int64_t dim, bool descending, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cumsum_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto cumprod_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto unique_kernel(const Tensor& input, bool sorted_output, bool return_inverse, bool return_counts, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto median_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> std::vector<Tensor>;
    auto mode_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> std::vector<Tensor>;
    auto logcumsumexp_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> Tensor;
    auto bincount_kernel(const Tensor& input, const Tensor* weights, int64_t minlength, cudaStream_t stream) -> Tensor;

    // New reduction operations (CumMax, CumMin, Fmax, Fmin, Isin, Kthvalue, Quantile, etc.)
    auto cummax_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cummin_kernel(const Tensor& input, int64_t dim, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto fmax_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto fmin_kernel(const Tensor& a, const Tensor& b, cudaStream_t stream) -> Tensor;
    auto isin_kernel(const Tensor& elements, const Tensor& test_elements, cudaStream_t stream) -> Tensor;
    auto kthvalue_kernel(const Tensor& input, int64_t k, int64_t dim, bool keepdim, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto quantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto nanquantile_kernel(const Tensor& input, double q, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto nanmedian_kernel(const Tensor& input, int64_t dim, bool keepdim, cudaStream_t stream) -> Tensor;
    auto histc_kernel(const Tensor& input, int64_t bins, double min_val, double max_val, cudaStream_t stream) -> Tensor;
    auto unique_consecutive_kernel(const Tensor& input, bool return_inverse, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto segment_reduce_kernel(const Tensor& data, const Tensor& offsets, const std::string& reduce, int64_t axis, cudaStream_t stream) -> Tensor;

    // Sampling / statistics operations
    auto bernoulli_kernel(const Tensor& probs, cudaStream_t stream) -> Tensor;
    auto poisson_sample_kernel(const Tensor& rates, cudaStream_t stream) -> Tensor;
    auto multinomial_kernel(const Tensor& probs, int64_t num_samples, bool replacement, cudaStream_t stream) -> Tensor;
    auto bucketize_kernel(const Tensor& input, const Tensor& boundaries, bool right, cudaStream_t stream) -> Tensor;
    auto histogram_kernel(const Tensor& input, int64_t bins, double min_val, double max_val, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto histogramdd_kernel(const Tensor& input, std::vector<int64_t> bins,
                            std::vector<std::pair<double,double>> ranges,
                            bool density, cudaStream_t stream)
        -> std::pair<Tensor, std::vector<Tensor>>;
    auto cdist_kernel(const Tensor& x1, const Tensor& x2, double p, cudaStream_t stream) -> Tensor;
    auto normal_sample_kernel(const Tensor& mean, const Tensor& stddev, cudaStream_t stream) -> Tensor;
    auto exponential_sample_kernel(const Tensor& rate, cudaStream_t stream) -> Tensor;
    auto trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr, cudaStream_t stream) -> Tensor;
    auto cumulative_trapezoid_kernel(const Tensor& y, int64_t dim, double dx, const Tensor* x_ptr, cudaStream_t stream) -> Tensor;
    auto gradient_kernel(const Tensor& input, int64_t dim, double spacing, cudaStream_t stream) -> Tensor;
    auto pairwise_distance_kernel(const Tensor& x1, const Tensor& x2, double p, cudaStream_t stream) -> Tensor;
    auto pdist_kernel(const Tensor& input, double p, cudaStream_t stream) -> Tensor;

    // Advanced indexing (native CUDA kernel)
    auto advanced_index_cuda_kernel(const Tensor& src, const std::vector<Tensor>& indices, int64_t num_indices, cudaStream_t stream) -> Tensor;
    auto advanced_index_put_cuda_kernel(const Tensor& src, const std::vector<Tensor>& indices, const Tensor& values, int64_t num_indices, cudaStream_t stream) -> Tensor;

    // STFT/ISTFT (native CUDA framing + cuFFT)
    auto stft_cuda_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor& window, bool center, bool normalized, bool onesided, cudaStream_t stream) -> Tensor;
    auto istft_cuda_kernel(const Tensor& input, int64_t n_fft, int64_t hop_length, int64_t win_length, const Tensor& window, bool center, bool normalized, bool onesided, int64_t length, cudaStream_t stream) -> Tensor;

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

    // Fused Attention operation (returns {output, logsumexp})
    auto fused_attention_cuda(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        float scale
    ) -> std::pair<Tensor, Tensor>;

    // Fused Flash Attention backward (tiled, memory-efficient)
    auto flash_attention_backward_cuda(
        const Tensor& dO,
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        const Tensor& O,
        const Tensor& L,
        float scale,
        bool causal
    ) -> std::vector<Tensor>;

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

    // Deformable Conv2d (DCNv2) operations
    auto deformable_conv2d_forward_kernel(
        const Tensor& input, const Tensor& offset, const Tensor& weight,
        const Tensor& bias, const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        cudaStream_t stream) -> Tensor;
    auto deformable_conv2d_backward_input_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& offset,
        const Tensor& weight, const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        cudaStream_t stream) -> std::vector<Tensor>;
    auto deformable_conv2d_backward_weight_kernel(
        const Tensor& grad_output, const Tensor& input, const Tensor& offset,
        const Tensor& mask,
        int64_t stride_h, int64_t stride_w,
        int64_t pad_h, int64_t pad_w,
        int64_t dil_h, int64_t dil_w,
        int64_t groups, int64_t offset_groups,
        const std::vector<int64_t>& weight_shape,
        cudaStream_t stream) -> Tensor;

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
    // ABI-safe wrappers (single Tensor return avoids tuple across nvcc/g++ boundary)
    auto cudnn_conv_transpose3d_backward_input(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv_transpose3d_backward_weight(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv_transpose3d_backward_bias(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups, cudaStream_t stream) -> Tensor;
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

    // Fractional Max Pool operations
    auto fractional_maxpool2d_forward_kernel(const Tensor& input, int64_t out_h, int64_t out_w, const Tensor* random_samples, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto fractional_maxpool3d_forward_kernel(const Tensor& input, int64_t out_d, int64_t out_h, int64_t out_w, const Tensor* random_samples, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto fractional_maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;

    // Max Unpool operations
    auto max_unpool2d_forward_kernel(const Tensor& input, const Tensor& indices, int64_t out_h, int64_t out_w, cudaStream_t stream) -> Tensor;
    auto max_unpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto max_unpool3d_forward_kernel(const Tensor& input, const Tensor& indices, int64_t out_d, int64_t out_h, int64_t out_w, cudaStream_t stream) -> Tensor;
    auto max_unpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;

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
    auto fused_softmax_cross_entropy_cuda(const Tensor& logits, const Tensor& targets, const std::string& reduction) -> Tensor;

    // Fused optimizer steps
    auto fused_rmsprop_step_cuda(Tensor& param, const Tensor& grad, Tensor& square_avg, Tensor* grad_avg, Tensor* momentum_buffer, float lr, float alpha, float eps, float weight_decay, float momentum, bool centered, cudaStream_t stream) -> void;
    auto fused_adadelta_step_cuda(Tensor& param, const Tensor& grad, Tensor& square_avg, Tensor& acc_delta, float rho, float eps, float lr, float weight_decay, cudaStream_t stream) -> void;
    auto fused_adagrad_step_cuda(Tensor& param, const Tensor& grad, Tensor& sum_sq, float lr, float lr_decay, float eps, float weight_decay, int64_t step, cudaStream_t stream) -> void;

    // Nested tensor operations
    auto nested_softmax_cuda(const Tensor& values, const Tensor& offsets, int64_t dim, cudaStream_t stream) -> Tensor;
    auto nested_log_softmax_cuda(const Tensor& values, const Tensor& offsets, int64_t dim, cudaStream_t stream) -> Tensor;
    auto nested_sum_cuda(const Tensor& values, const Tensor& offsets, cudaStream_t stream) -> Tensor;
    auto nested_mean_cuda(const Tensor& values, const Tensor& offsets, cudaStream_t stream) -> Tensor;
    auto nested_layer_norm_cuda(const Tensor& values, const Tensor& offsets, const Tensor& weight, const Tensor& bias, float eps, cudaStream_t stream) -> Tensor;
    auto nested_linear_cuda(const Tensor& values, const Tensor& weight, const Tensor* bias, cudaStream_t stream) -> Tensor;
    auto nested_attention_cuda(const Tensor& Q, const Tensor& K, const Tensor& V, const Tensor& q_offsets, const Tensor& kv_offsets, float scale, bool causal, cudaStream_t stream) -> Tensor;
    auto nested_attention_backward_cuda(const Tensor& grad_out, const Tensor& Q, const Tensor& K, const Tensor& V,
                                         const Tensor& attn_out, const Tensor& q_offsets, const Tensor& kv_offsets,
                                         float scale, bool causal, cudaStream_t stream) -> std::vector<Tensor>;
    auto nested_to_padded_cuda(const Tensor& values, const Tensor& offsets, int64_t max_len, float padding_value, cudaStream_t stream) -> Tensor;
    auto nested_from_padded_cuda(const Tensor& padded, const Tensor& offsets, cudaStream_t stream) -> Tensor;

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
    Tensor frac_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor heaviside_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor nan_to_num_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor log_sigmoid_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bitwise_and_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bitwise_or_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bitwise_xor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bitwise_not_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bitwise_left_shift_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bitwise_right_shift_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    // activations.cu
    Tensor rrelu_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor rrelu_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor log_sigmoid_backward_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    // Reductions + scatter (activations.cu)
    Tensor count_nonzero_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor nansum_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor nanmean_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    std::vector<Tensor> aminmax_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor index_add_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor index_copy_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor index_fill_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor scatter_reduce_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

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
    Tensor signbit_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor isposinf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor isneginf_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor isreal_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

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
    Tensor complex_tensor_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor cross_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Special math (native CUDA implementations — replace previous CPU-roundtrip fallbacks)
    Tensor gamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor lgamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor digamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor polygamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor beta_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor betainc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bessel_j0_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bessel_j1_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bessel_y0_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bessel_y1_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bessel_i0_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor bessel_i1_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor erfinv_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor sinc_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor zeta_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Ndtr / LogNdtr / Multigammaln
    Tensor ndtr_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor log_ndtr_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor multigammaln_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // New element-wise math ops
    Tensor rsqrt_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor square_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor asinh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor acosh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor atanh_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor hypot_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor copysign_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor nextafter_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor gcd_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor lcm_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    auto addcmul_kernel(const Tensor& input, const Tensor& t1, const Tensor& t2, double alpha, cudaStream_t stream) -> Tensor;
    auto addcdiv_kernel(const Tensor& input, const Tensor& t1, const Tensor& t2, double alpha, cudaStream_t stream) -> Tensor;
    Tensor igamma_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor igammac_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // New unary/binary math dispatch wrappers
    Tensor deg2rad_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor rad2deg_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor logit_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor float_power_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor xlog1py_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor ldexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor frexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // DiagEmbed dispatch wrapper (diagflat is implemented inline in registry)
    Tensor diag_embed_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // NaN-ignoring variance and standard deviation
    Tensor nanvar_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor nanstd_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // New math ops (OpId 680-688)
    Tensor logaddexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor logaddexp2_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor xlogy_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor i0e_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor i1e_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor entr_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor spherical_bessel_j0_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // New reduction ops (OpId 683-684)
    Tensor cosine_similarity_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor renorm_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

    // Sparse kernels (sparse.cu). Definitions exist in both the
    // TENZOR_HAS_CUSPARSE path (cuSPARSE-backed) and the fallback path
    // (hand-written CUDA CSR kernels), so the forward declarations are
    // unconditional.
    auto cuda_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) -> Tensor;
    auto cuda_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) -> Tensor;
    auto cuda_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) -> Tensor;
#ifdef TENZOR_HAS_CUSPARSE
    // SpGEMM and triangular solve are only defined when cuSPARSE is
    // available — the hand-rolled fallback intentionally doesn't cover
    // them. sparse_ops.cpp will throw std::runtime_error
    // if has_kernel(SparseSpGEMM/Trsv/Trsm) returns false.
    auto cuda_spgemm_kernel(const SparseTensor& a, const SparseTensor& b,
                            void* stream) -> SparseTensor;
    auto cuda_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b,
                                  bool upper, void* stream) -> Tensor;
    auto cuda_sparse_trsm_kernel(const SparseTensor& L, const Tensor& B,
                                  bool upper, void* stream) -> Tensor;
#endif

    // Standalone GPU implementations (always available, no cuSPARSE dependency)
    auto spgemm_standalone(std::span<const Tensor> inputs, const OpAttributes& attrs,
                           cudaStream_t stream) -> std::vector<Tensor>;
    auto sparse_trsv_standalone(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                const Tensor& b, int64_t N, bool upper, cudaStream_t stream) -> Tensor;
    auto sparse_trsm_standalone(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                const Tensor& B, int64_t N, bool upper, cudaStream_t stream) -> Tensor;

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
    // Special Math Functions — native CUDA device kernels
    // =========================================================================
    table.register_single_output_kernel(OpId::Gamma,     cuda::gamma_dispatch);
    table.register_single_output_kernel(OpId::Lgamma,    cuda::lgamma_dispatch);
    table.register_single_output_kernel(OpId::Digamma,   cuda::digamma_dispatch);
    table.register_single_output_kernel(OpId::Polygamma, cuda::polygamma_dispatch);
    table.register_single_output_kernel(OpId::Beta,      cuda::beta_dispatch);
    table.register_single_output_kernel(OpId::BesselJ0,  cuda::bessel_j0_dispatch);
    table.register_single_output_kernel(OpId::BesselJ1,  cuda::bessel_j1_dispatch);
    table.register_single_output_kernel(OpId::BesselY0,  cuda::bessel_y0_dispatch);
    table.register_single_output_kernel(OpId::BesselY1,  cuda::bessel_y1_dispatch);
    table.register_single_output_kernel(OpId::BesselI0,  cuda::bessel_i0_dispatch);
    table.register_single_output_kernel(OpId::BesselI1,  cuda::bessel_i1_dispatch);
    table.register_single_output_kernel(OpId::ErfInv,    cuda::erfinv_dispatch);
    table.register_single_output_kernel(OpId::Sinc,      cuda::sinc_dispatch);
    table.register_single_output_kernel(OpId::Zeta,      cuda::zeta_dispatch);
    table.register_single_output_kernel(OpId::Ndtr,      cuda::ndtr_dispatch);
    table.register_single_output_kernel(OpId::LogNdtr,   cuda::log_ndtr_dispatch);
    table.register_single_output_kernel(OpId::Multigammaln, cuda::multigammaln_dispatch);

    // LinalgVectorNorm: delegates to existing Norm kernel
    table.register_kernel(OpId::LinalgVectorNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = static_cast<float>(attrs.get_float(AttrKey::P, 2.0));
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return std::vector<Tensor>{cuda::norm_kernel(inputs[0], p, dim, keepdim, get_cuda_stream(attrs))};
    });

    // LinalgMatrixNorm: Frobenius (ord=0), nuclear (ord=1), spectral (ord=2)
    table.register_single_output_kernel(OpId::LinalgMatrixNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t ord = static_cast<int64_t>(attrs.get_float(AttrKey::Order, 0.0));
        if (ord == 0) {
            return cuda::norm_kernel(inputs[0], 2.0f, INT64_MIN, false, get_cuda_stream(attrs));
        }
        auto stream = get_cuda_stream(attrs);
        auto [U, S, Vt] = cuda::linalg_svd_kernel(inputs[0], false, stream);
        if (ord == 1) {
            return cuda::sum_kernel(S, INT64_MIN, false, stream);
        }
        return cuda::max_kernel(S, INT64_MIN, false, stream);
    });

    // LinalgVecdot: sum(a * b, dim)
    table.register_single_output_kernel(OpId::LinalgVecdot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        auto stream = get_cuda_stream(attrs);
        Tensor product = cuda::mul_kernel(inputs[0], inputs[1], stream);
        return cuda::sum_kernel(product, dim, false, stream);
    });

    table.register_kernel(OpId::BetaInc,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {cuda::betainc_dispatch(inputs, attrs)};
        });

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
    table.register_single_output_kernel(OpId::ComplexTensor, cuda::complex_tensor_dispatch);
    table.register_single_output_kernel(OpId::Cross, cuda::cross_dispatch);

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

    table.register_single_output_kernel(OpId::RepeatInterleave, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        int64_t num_repeats = attrs.get_int(AttrKey::NumRepeats, 1);
        auto stream = get_cuda_stream(attrs);
        if (num_repeats >= 0) {
            return cuda::repeat_interleave_scalar_kernel(inputs[0], num_repeats, dim, stream);
        } else {
            return cuda::repeat_interleave_tensor_kernel(inputs[0], inputs[1], dim, stream);
        }
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
        // inputs: [grad_output, indices, input, output]
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 2);
        int64_t stride = attrs.get_int(AttrKey::Stride, kernel_size);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        return cuda::cudnn_maxpool2d_backward(inputs[0], inputs[2], inputs[3], kernel_size, stride, padding, get_cuda_stream(attrs));
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
    table.register_kernel(OpId::RMSNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, rrms] = cuda::fused_rms_norm_cuda(inputs[0], inputs[1], eps);
        return std::vector<Tensor>{output, rrms};
    });

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
        auto [output, lse] = cuda::fused_attention_cuda(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output};
    });

    // =========================================================================
    // Flash Attention (memory-efficient tiled attention)
    // =========================================================================
    table.register_kernel(OpId::FlashAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Uses same implementation as FusedAttention — both are memory-efficient
        // Returns {O, L} where L is the row-wise logsumexp (for backward pass)
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        auto [output, lse] = cuda::fused_attention_cuda(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output, lse};
    });

    // =========================================================================
    // Flash Attention Backward (fused tiled kernel, composed-ops fallback)
    // =========================================================================
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
                return cuda::flash_attention_backward_cuda(dO, Q, K, V, O, L, scale, causal);
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

    // =========================================================================
    // Einsum (composed — delegates to einsum_composed to avoid dispatch loop)
    // =========================================================================
    table.register_kernel(OpId::Einsum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto equation = std::string(attrs.get_string(AttrKey::EinsumEquation, ""));
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return {einsum_composed(equation, tensors)};
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

    table.register_single_output_kernel(OpId::GridSample, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cuda::grid_sample_cuda(inputs[0], inputs[1], mode, padding_mode, align_corners);
    });

    table.register_single_output_kernel(OpId::AffineGrid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto size = attrs.get_int_list(AttrKey::OutputSize);
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cuda::affine_grid_cuda(inputs[0], size, align_corners);
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
    // Conv2dBackwardBias: inputs = {grad_output}
    // Bias gradient = sum of grad_output over batch and spatial dims (N,H,W)
    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        const Tensor& grad_output = inputs[0]; // (N, C, H, W)
        // Sum over dim 3 (W), then 2 (H), then 0 (N) to get (C,)
        auto t1 = tenzor::sum(grad_output, 3); // (N, C, H)
        auto t2 = tenzor::sum(t1, 2);          // (N, C)
        auto grad_bias = tenzor::sum(t2, 0);   // (C,)
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

    // Conv1d: wraps Conv2d by unsqueezing height dimension [N,C,L] -> [N,C,1,L]
    table.register_kernel(OpId::Conv1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto input_4d = inputs[0].unsqueeze(2);
        auto weight_4d = inputs[1].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = inputs.size() > 2
            ? std::vector<Tensor>{input_4d, weight_4d, inputs[2]}
            : std::vector<Tensor>{input_4d, weight_4d};
        auto result = tenzor::dispatch(OpId::Conv2dForward, conv2d_inputs, attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
        auto result = tenzor::dispatch(OpId::Conv2dBackwardInput, conv2d_inputs, attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        auto input_4d = inputs[1].unsqueeze(2);
        auto weight_4d = inputs[2].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d, input_4d, weight_4d};
        auto result = tenzor::dispatch(OpId::Conv2dBackwardWeight, conv2d_inputs, attrs);
        return {result[0].squeeze(2)};
    });

    table.register_kernel(OpId::Conv1dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        auto grad_4d = inputs[0].unsqueeze(2);
        std::vector<Tensor> conv2d_inputs = {grad_4d};
        auto result = tenzor::dispatch(OpId::Conv2dBackwardBias, conv2d_inputs, {});
        return {result[0]};
    });

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
    // Deformable Conv2d (DCNv2) Operations
    // =========================================================================

    // DeformableConv2dForward: inputs = {input, offset, weight, bias, mask}
    table.register_kernel(OpId::DeformableConv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        return std::vector<Tensor>{cuda::deformable_conv2d_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, get_cuda_stream(attrs))};
    });

    // DeformableConv2dBackwardInput: inputs = {grad_output, input, offset, weight, mask}
    table.register_kernel(OpId::DeformableConv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        return cuda::deformable_conv2d_backward_input_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, get_cuda_stream(attrs));
    });

    // DeformableConv2dBackwardWeight: inputs = {grad_output, input, offset, mask}
    table.register_kernel(OpId::DeformableConv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride_h = attrs.get_int(AttrKey::StrideH, 1);
        int64_t stride_w = attrs.get_int(AttrKey::StrideW, 1);
        int64_t pad_h = attrs.get_int(AttrKey::PaddingH, 0);
        int64_t pad_w = attrs.get_int(AttrKey::PaddingW, 0);
        int64_t dil_h = attrs.get_int(AttrKey::DilationH, 1);
        int64_t dil_w = attrs.get_int(AttrKey::DilationW, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        return std::vector<Tensor>{cuda::deformable_conv2d_backward_weight_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3],
            stride_h, stride_w, pad_h, pad_w, dil_h, dil_w,
            groups, offset_groups, weight_shape, get_cuda_stream(attrs))};
    });

    // DeformableConv2dBackwardBias: inputs = {grad_output}
    // Bias gradient = channel-wise sum of grad_output over (N, H, W)
    table.register_kernel(OpId::DeformableConv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        const Tensor& grad_output = inputs[0]; // (N, C, H, W)
        auto t1 = tenzor::sum(grad_output, 3); // (N, C, H)
        auto t2 = tenzor::sum(t1, 2);          // (N, C)
        auto grad_bias = tenzor::sum(t2, 0);   // (C,)
        return {grad_bias};
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
    // Conv3dBackwardBias: inputs = {grad_output}
    // Bias gradient = sum of grad_output over batch and spatial dims (N,D,H,W)
    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        const Tensor& grad_output = inputs[0]; // (N, C, D, H, W)
        auto t1 = tenzor::sum(grad_output, 4); // (N, C, D, H)
        auto t2 = tenzor::sum(t1, 3);          // (N, C, D)
        auto t3 = tenzor::sum(t2, 2);          // (N, C)
        auto grad_bias = tenzor::sum(t3, 0);   // (C,)
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
    // ConvTranspose3dBackward: use ABI-safe wrappers that return single Tensor
    // instead of std::tuple, avoiding potential nvcc/g++ tuple ABI mismatch.
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        return cuda::cudnn_conv_transpose3d_backward_input(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        return cuda::cudnn_conv_transpose3d_backward_weight(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    // ConvTranspose3dBackwardBias: inputs = {grad_output} (only 1 tensor)
    // Bias gradient = sum(grad_output) over batch and spatial dims (N,D,H,W).
    // Same operation as Conv3dBackwardBias.
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        const Tensor& grad_output = inputs[0]; // (N, C, D, H, W)
        auto t1 = tenzor::sum(grad_output, 4); // (N, C, D, H)
        auto t2 = tenzor::sum(t1, 3);          // (N, C, D)
        auto t3 = tenzor::sum(t2, 2);          // (N, C)
        return tenzor::sum(t3, 0);              // (C,)
    });
#else
    // -------------------------------------------------------------------------
    // Conv3d / ConvTranspose3d fallback registrations (no cuDNN).
    // Direct-convolution kernels in src/backends/cuda/kernels/conv3d.cu.
    // -------------------------------------------------------------------------
    table.register_single_output_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv3d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv3d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv3d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
        const Tensor& grad_output = inputs[0]; // (N, C, D, H, W)
        auto t1 = tenzor::sum(grad_output, 4); // (N, C, D, H)
        auto t2 = tenzor::sum(t1, 3);          // (N, C, D)
        auto t3 = tenzor::sum(t2, 2);          // (N, C)
        auto grad_bias = tenzor::sum(t3, 0);   // (C,)
        return {grad_bias};
    });

    table.register_single_output_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv_transpose3d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        return cuda::conv_transpose3d_backward_input_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t output_padding = attrs.get_int(AttrKey::OutputPadding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        return cuda::conv_transpose3d_backward_weight_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        const Tensor& grad_output = inputs[0]; // (N, C, D, H, W)
        auto t1 = tenzor::sum(grad_output, 4); // (N, C, D, H)
        auto t2 = tenzor::sum(t1, 3);          // (N, C, D)
        auto t3 = tenzor::sum(t2, 2);          // (N, C)
        return tenzor::sum(t3, 0);              // (C,)
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
        // Accept NumGroups (layer convention) or Groups (functional fallback).
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        auto [output, mean, inv_std] = cuda::group_norm_forward_kernel(
            inputs[0], inputs[1], inputs[2], num_groups, eps, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Canonical input order across all backends:
        //   [grad_output, input, mean, rstd, weight]
        // Kernel signature is (grad_output, input, weight, mean, inv_std).
        int64_t num_groups = attrs.get_int(AttrKey::NumGroups, attrs.get_int(AttrKey::Groups, 1));
        auto [grad_input, grad_weight, grad_bias] = cuda::group_norm_backward_kernel(
            inputs[0], inputs[1], inputs[4], inputs[2], inputs[3], num_groups, get_cuda_stream(attrs));
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
        // Dims is a comma-separated string of dimensions to flip
        auto dims_sv = attrs.get_string(AttrKey::Dims, "0");
        std::string dims_str(dims_sv);
        auto stream = get_cuda_stream(attrs);
        Tensor result = inputs[0];
        // Parse comma-separated dims and flip each
        std::istringstream ss(dims_str);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) {
                int64_t dim = std::stoll(token);
                result = cuda::flip_kernel(result, dim, stream);
            }
        }
        return result;
    });

    // =========================================================================
    // Fused Softmax Cross Entropy (Phase 4A - MEDIUM)
    // =========================================================================
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [logits, targets]
        // Use the reduction-aware overload that returns a single reduced loss tensor
        std::string reduction = std::string(attrs.get_string(AttrKey::Reduction, "mean"));
        auto loss = cuda::fused_softmax_cross_entropy_cuda(inputs[0], inputs[1], reduction);
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
        cudaStream_t stream = get_cuda_stream(attrs);
        return cuda::unfold_cuda(inputs[0], kernel_size, stride, padding, dilation, stream);
    });

    // inputs: [input]
    // attrs: output_size, kernel_size, stride, padding, dilation
    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        int64_t kernel_size = attrs.get_int(AttrKey::KernelSize, 3);
        int64_t stride = attrs.get_int(AttrKey::Stride, 1);
        int64_t padding = attrs.get_int(AttrKey::Padding, 0);
        int64_t dilation = attrs.get_int(AttrKey::Dilation, 1);
        cudaStream_t stream = get_cuda_stream(attrs);
        return cuda::fold_cuda(inputs[0], output_size, kernel_size, stride, padding, dilation, stream);
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

    // =========================================================================
    // NMS Operation
    // =========================================================================
    table.register_kernel(OpId::NMS, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // inputs: [boxes (N,4), scores (N)]
        // attrs: IouThreshold
        float iou_threshold = static_cast<float>(attrs.get_float(AttrKey::IouThreshold, 0.5));
        return {cuda::nms_cuda_wrapper(inputs[0], inputs[1], iou_threshold)};
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
    table.register_kernel(OpId::Median, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cuda::median_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Mode, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, INT64_MIN);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        return cuda::mode_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs));
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
    table.register_single_output_kernel(OpId::SolveTriangular, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, true);
        bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
        return cuda::linalg_solve_triangular_kernel(inputs[0], inputs[1], upper, unitriangular, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Geqrf, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [result, tau] = cuda::linalg_geqrf_kernel(inputs[0], get_cuda_stream(attrs));
        return {result, tau};
    });
    table.register_single_output_kernel(OpId::Ormqr, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool left = attrs.get_bool(AttrKey::Left, true);
        bool transpose_q = attrs.get_bool(AttrKey::TransposeQ, false);
        return cuda::linalg_ormqr_kernel(inputs[0], inputs[1], inputs[2], left, transpose_q, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LinalgCholeskySolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        auto stream = get_cuda_stream(attrs);
        if (!upper) {
            auto Y = cuda::linalg_solve_triangular_kernel(inputs[1], inputs[0], false, false, stream);
            int64_t ndim = inputs[1].ndim();
            auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
            return cuda::linalg_solve_triangular_kernel(Lt, Y, true, false, stream);
        } else {
            int64_t ndim = inputs[1].ndim();
            auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
            auto Y = cuda::linalg_solve_triangular_kernel(Ut, inputs[0], false, false, stream);
            return cuda::linalg_solve_triangular_kernel(inputs[1], Y, true, false, stream);
        }
    });
#else // !TENZOR_HAS_CUSOLVER — use native CUDA shared-memory linalg fallback
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
    table.register_single_output_kernel(OpId::SolveTriangular, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, true);
        bool unitriangular = attrs.get_bool(AttrKey::UnitTriangular, false);
        return cuda::linalg_solve_triangular_kernel(inputs[0], inputs[1], upper, unitriangular, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Geqrf, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [result, tau] = cuda::linalg_geqrf_kernel(inputs[0], get_cuda_stream(attrs));
        return {result, tau};
    });
    table.register_single_output_kernel(OpId::Ormqr, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool left = attrs.get_bool(AttrKey::Left, true);
        bool transpose_q = attrs.get_bool(AttrKey::TransposeQ, false);
        return cuda::linalg_ormqr_kernel(inputs[0], inputs[1], inputs[2], left, transpose_q, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LinalgCholeskySolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        bool upper = attrs.get_bool(AttrKey::Upper, false);
        auto stream = get_cuda_stream(attrs);
        if (!upper) {
            auto Y = cuda::linalg_solve_triangular_kernel(inputs[1], inputs[0], false, false, stream);
            int64_t ndim = inputs[1].ndim();
            auto Lt = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
            return cuda::linalg_solve_triangular_kernel(Lt, Y, true, false, stream);
        } else {
            int64_t ndim = inputs[1].ndim();
            auto Ut = tenzor::transpose(inputs[1], ndim - 2, ndim - 1).contiguous();
            auto Y = cuda::linalg_solve_triangular_kernel(Ut, inputs[0], false, false, stream);
            return cuda::linalg_solve_triangular_kernel(inputs[1], Y, true, false, stream);
        }
    });
#endif // TENZOR_HAS_CUSOLVER

    // LinalgLU / LinalgLUSolve: registered unconditionally. Both cuSOLVER and
    // the native shared-memory fallback paths provide linalg_lu_kernel /
    // linalg_lu_solve_kernel.
    table.register_kernel(OpId::LinalgLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [L, U, pivots] = cuda::linalg_lu_kernel(inputs[0], get_cuda_stream(attrs));
        return {L, U, pivots};
    });
    table.register_single_output_kernel(OpId::LinalgLUSolve, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::linalg_lu_solve_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs));
    });

    // =========================================================================
    // LinalgHouseholder, LinalgLDLFactor, LinalgLDLSolve,
    // CholeskyInverse, TensorInv, TensorSolve
    // =========================================================================
    table.register_single_output_kernel(OpId::LinalgHouseholder,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::linalg_householder_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
        });

    table.register_kernel(OpId::LinalgLDLFactor,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto [LD, pivots] = cuda::linalg_ldl_factor_kernel(inputs[0], get_cuda_stream(attrs));
            return {LD, pivots};
        });

    table.register_single_output_kernel(OpId::LinalgLDLSolve,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::linalg_ldl_solve_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs));
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
#else // !TENZOR_HAS_CUFFT — use native Cooley-Tukey + Bluestein CUDA FFT fallback

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

    // =========================================================================
    // Sparse Tensor Operations (OpIds 460-464)
    //
    // Wrapper lambdas that reconstruct SparseTensor from CSR components passed
    // as plain Tensors, then delegate to the existing sparse:: functions which
    // internally dispatch to cuSPARSE when inputs are on CUDA.
    // =========================================================================

    // SparseSpMM: sparse(M,K) @ dense(K,N) -> dense(M,N).
    //
    // NOTE: previously this lambda called `sparse::spmm(sp, inputs[3])`
    // which recursed back into the top-level sparse::spmm in tenzor_core,
    // whose cuSPARSE branch is compiled out (TENZOR_HAS_CUSPARSE is only
    // defined inside this backend .so), so it fell through to cpu_spmm()
    // and segfaulted on device pointers. Call the kernel directly — the
    // same pattern the cuSPARSE-absent fallback has always used.
    table.register_single_output_kernel(OpId::SparseSpMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return cuda::cuda_spmm_kernel(sp, inputs[3]);
        });

    // SparseSpMV: sparse(M,K) @ vec(K) -> vec(M).
    table.register_single_output_kernel(OpId::SparseSpMV,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return cuda::cuda_spmv_kernel(sp, inputs[3]);
        });

#ifdef TENZOR_HAS_CUSPARSE
    // SparseSpGEMM: sparse(M,K) × sparse(K,N) -> sparse(M,N).
    // Inputs [0..2] are A's CSR components, [3..5] are B's. M/K/N in attrs.
    // The kernel returns a SparseTensor which we unpack into 3 result
    // tensors for the multi-output dispatch interface.
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            int64_t N = attrs.get_int(AttrKey::N);
            auto a = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            auto b = SparseTensor::sparse_csr(inputs[3], inputs[4], inputs[5], {K, N});
            auto c = cuda::cuda_spgemm_kernel(a, b, /*stream=*/nullptr);
            return {c.crow_indices(), c.col_indices(), c.values()};
        });

    // SparseTrsv: solve L*x = b  (b is 1D, length N).
    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return cuda::cuda_sparse_trsv_kernel(L, inputs[3], upper, /*stream=*/nullptr);
        });

    // SparseTrsm: solve L*X = B  (B is 2D, N × K_rhs).
    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            auto L = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {N, N});
            return cuda::cuda_sparse_trsm_kernel(L, inputs[3], upper, /*stream=*/nullptr);
        });
#else
    // Standalone GPU SpGEMM/Trsv/Trsm — no cuSPARSE dependency
    table.register_kernel(OpId::SparseSpGEMM,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return cuda::spgemm_standalone(inputs, attrs, /*stream=*/nullptr);
        });

    table.register_single_output_kernel(OpId::SparseTrsv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return cuda::sparse_trsv_standalone(inputs[0], inputs[1], inputs[2],
                                                inputs[3], N, upper, /*stream=*/nullptr);
        });

    table.register_single_output_kernel(OpId::SparseTrsm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t N = attrs.get_int(AttrKey::N);
            bool upper = attrs.get_bool(AttrKey::Upper, false);
            return cuda::sparse_trsm_standalone(inputs[0], inputs[1], inputs[2],
                                                inputs[3], N, upper, /*stream=*/nullptr);
        });
#endif // TENZOR_HAS_CUSPARSE

    // SparseToDense: CSR components -> dense tensor (works on any device)
    table.register_single_output_kernel(OpId::SparseToDense,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return sp.to_dense();
        });

    // =========================================================================
    // Sampling / Statistics operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Bernoulli,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::bernoulli_kernel(inputs[0], get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::PoissonSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::poisson_sample_kernel(inputs[0], get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::NormalSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::normal_sample_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::ExponentialSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::exponential_sample_kernel(inputs[0], get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Multinomial,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_samples = attrs.get_int(AttrKey::NumSamples, 1);
            bool replacement = attrs.get_bool(AttrKey::Replacement, false);
            return cuda::multinomial_kernel(inputs[0], num_samples, replacement, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Bucketize,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            bool right = attrs.get_bool(AttrKey::Right, false);
            return cuda::bucketize_kernel(inputs[0], inputs[1], right, get_cuda_stream(attrs));
        });

    table.register_kernel(OpId::Histogram,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t bins = attrs.get_int(AttrKey::NumBins, 10);
            double min_val = attrs.get_float(AttrKey::Min, 0.0);
            double max_val = attrs.get_float(AttrKey::Max, 0.0);
            auto [counts, edges] = cuda::histogram_kernel(inputs[0], bins, min_val, max_val, get_cuda_stream(attrs));
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

            auto [counts, edges] = cuda::histogramdd_kernel(inputs[0], bins_list, ranges, density, get_cuda_stream(attrs));
            std::vector<Tensor> results;
            results.push_back(counts);
            for (auto& e : edges) results.push_back(std::move(e));
            return results;
        });

    table.register_single_output_kernel(OpId::CDist,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double p = attrs.get_float(AttrKey::DistP, 2.0);
            return cuda::cdist_kernel(inputs[0], inputs[1], p, get_cuda_stream(attrs));
        });

    // =========================================================================
    // Trapezoid / Cumulative Trapezoid / Gradient / PairwiseDistance / Pdist
    // =========================================================================
    table.register_single_output_kernel(OpId::Trapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return cuda::trapezoid_kernel(inputs[0], dim, dx, x_ptr, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::CumulativeTrapezoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double dx = attrs.get_float(AttrKey::Dx, 1.0);
        const Tensor* x_ptr = (inputs.size() > 1) ? &inputs[1] : nullptr;
        return cuda::cumulative_trapezoid_kernel(inputs[0], dim, dx, x_ptr, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NumericalGradient, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        double spacing = attrs.get_float(AttrKey::Spacing, 1.0);
        return cuda::gradient_kernel(inputs[0], dim, spacing, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::PairwiseDistance, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return cuda::pairwise_distance_kernel(inputs[0], inputs[1], p, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::Pdist, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double p = attrs.get_float(AttrKey::DistP, 2.0);
        return cuda::pdist_kernel(inputs[0], p, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Advanced indexing (native CUDA gather/scatter kernel)
    // =========================================================================
    table.register_single_output_kernel(OpId::AdvancedIndex,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            std::vector<Tensor> indices(inputs.begin() + 1, inputs.end());
            return cuda::advanced_index_cuda_kernel(inputs[0], indices, num_indices, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::AdvancedIndexPut,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t num_indices = attrs.get_int(AttrKey::NumIndices, 0);
            // inputs[0] = destination, inputs[1] = values, inputs[2..2+N] = index tensors
            const auto& values = inputs[1];
            std::vector<Tensor> indices(inputs.begin() + 2, inputs.begin() + 2 + num_indices);
            return cuda::advanced_index_put_cuda_kernel(inputs[0], indices, values, num_indices, get_cuda_stream(attrs));
        });

    // =========================================================================
    // STFT / ISTFT (native CUDA implementation)
    // =========================================================================
    table.register_single_output_kernel(OpId::STFT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_fft = attrs.get_int(AttrKey::NFft);
            int64_t hop_length = attrs.get_int(AttrKey::HopLength, -1);
            int64_t win_length = attrs.get_int(AttrKey::WinLength, -1);
            bool center = attrs.get_bool(AttrKey::Centered, true);
            bool normalized = attrs.get_bool(AttrKey::Normalized, false);
            bool onesided = attrs.get_bool(AttrKey::OnesidedAttr, true);
            Tensor window = (inputs.size() > 1) ? inputs[1] : Tensor();
            return cuda::stft_cuda_kernel(inputs[0], n_fft, hop_length, win_length,
                                          window, center, normalized, onesided,
                                          get_cuda_stream(attrs));
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
            return cuda::istft_cuda_kernel(inputs[0], n_fft, hop_length, win_length,
                                           window, center, normalized, onesided,
                                           length, get_cuda_stream(attrs));
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
            return tenzor::fft::dct(inputs[0], type, n, dim, norm);
        });

    table.register_single_output_kernel(OpId::IDCT,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int type = static_cast<int>(attrs.get_int(AttrKey::DCTType, 2));
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            std::string norm{attrs.get_string(AttrKey::Norm, "backward")};
            std::optional<int64_t> n = std::nullopt;
            int64_t n_val = attrs.get_int(AttrKey::N, -1);
            if (n_val > 0) n = n_val;
            return tenzor::fft::idct(inputs[0], type, n, dim, norm);
        });

    table.register_single_output_kernel(OpId::MelScale,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t n_mels = attrs.get_int(AttrKey::NumMels, 128);
            double f_min = attrs.get_float(AttrKey::FMin, 0.0);
            double f_max = attrs.get_float(AttrKey::FMax, 0.0);
            int64_t sample_rate = attrs.get_int(AttrKey::SampleRate, 16000);
            return tenzor::fft::mel_scale(inputs[0], n_mels, f_min, f_max, sample_rate);
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
            return tenzor::fft::mfcc(inputs[0], sample_rate, n_mfcc, n_mels, n_fft, hop_length, f_min, f_max);
        });

    // DenseToSparse: dense tensor -> CSR components [crow_indices, col_indices, values]
    table.register_kernel(OpId::DenseToSparse,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            auto sp = SparseTensor::from_dense(inputs[0], SparseLayout::CSR);
            return {sp.crow_indices(), sp.col_indices(), sp.values()};
        });

    // SparseAdd: sparse(M,K) + dense(M,K) -> dense(M,K)
    // Uses a dedicated CUDA kernel (csr_sparse_add_kernel) that clones
    // the dense operand and adds CSR non-zeros directly.  Does NOT call
    // sparse::add() to avoid recursive dispatch.
    table.register_single_output_kernel(OpId::SparseAdd,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t M = attrs.get_int(AttrKey::M);
            int64_t K = attrs.get_int(AttrKey::K);
            auto sp = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
            return cuda::cuda_sparse_add_kernel(sp, inputs[3]);
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

    // ========================================================================
    // New Phase 4 ops
    // ========================================================================
    table.register_single_output_kernel(OpId::Frac, cuda::frac_dispatch);
    table.register_single_output_kernel(OpId::Heaviside, cuda::heaviside_dispatch);
    table.register_single_output_kernel(OpId::NanToNum, cuda::nan_to_num_dispatch);
    table.register_single_output_kernel(OpId::LogSigmoid, cuda::log_sigmoid_dispatch);
    table.register_single_output_kernel(OpId::BitwiseAnd, cuda::bitwise_and_dispatch);
    table.register_single_output_kernel(OpId::BitwiseOr, cuda::bitwise_or_dispatch);
    table.register_single_output_kernel(OpId::BitwiseXor, cuda::bitwise_xor_dispatch);
    table.register_single_output_kernel(OpId::BitwiseNot, cuda::bitwise_not_dispatch);
    table.register_single_output_kernel(OpId::BitwiseLeftShift, cuda::bitwise_left_shift_dispatch);
    table.register_single_output_kernel(OpId::BitwiseRightShift, cuda::bitwise_right_shift_dispatch);

    // RReLU + LogSigmoid backward
    table.register_single_output_kernel(OpId::RReLU, cuda::rrelu_dispatch);
    table.register_single_output_kernel(OpId::RReLUBackward, cuda::rrelu_backward_dispatch);
    table.register_single_output_kernel(OpId::LogSigmoidBackward, cuda::log_sigmoid_backward_dispatch);

    // NaN-aware reductions
    table.register_single_output_kernel(OpId::CountNonzero, cuda::count_nonzero_dispatch);
    table.register_single_output_kernel(OpId::Nansum, cuda::nansum_dispatch);
    table.register_single_output_kernel(OpId::Nanmean, cuda::nanmean_dispatch);
    // Aminmax: native GPU dual min/max reduction
    table.register_kernel(OpId::Aminmax, cuda::aminmax_dispatch);

    // Scatter variants
    table.register_single_output_kernel(OpId::IndexAdd, cuda::index_add_dispatch);
    table.register_single_output_kernel(OpId::IndexCopy, cuda::index_copy_dispatch);
    table.register_single_output_kernel(OpId::IndexFill, cuda::index_fill_dispatch);
    table.register_single_output_kernel(OpId::ScatterReduce, cuda::scatter_reduce_dispatch);

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

            auto dst_slice = output.slice(dim, index, index + 1, 1);
            auto dst_sh = dst_slice.shape();
            auto src_reshaped = src.reshape(std::vector<int64_t>(dst_sh.begin(), dst_sh.end())).contiguous();

            auto n = dst_slice.numel();
            auto elem_size = static_cast<int64_t>(dtype_size(output.dtype()));
            auto* dst_ptr = static_cast<char*>(dst_slice.data_ptr());
            const auto* src_ptr = static_cast<const char*>(src_reshaped.data_ptr());
            if (dst_slice.is_contiguous()) {
                cudaMemcpy(dst_ptr, src_ptr, n * elem_size, cudaMemcpyDeviceToDevice);
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
                    cudaMemcpy(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size, cudaMemcpyDeviceToDevice);
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
                cudaMemcpy(dst_ptr, src_ptr, n * elem_size, cudaMemcpyDeviceToDevice);
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
                    cudaMemcpy(dst_ptr + byte_offset, src_ptr + i * elem_size, elem_size, cudaMemcpyDeviceToDevice);
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
                    cudaMemcpy(out_ptr + out_elem_offset * elem_size,
                               src_ptr + src_elem_idx * elem_size, elem_size, cudaMemcpyDeviceToDevice);
                }

                for (int64_t i = static_cast<int64_t>(batch_dims.size()) - 1; i >= 0; i--) {
                    batch_coord[i]++;
                    if (batch_coord[i] < shape[batch_dims[i]]) break;
                    batch_coord[i] = 0;
                }
            }
            return output;
        });

    // =========================================================================
    // Fused GEMM Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Addmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            return cuda::addmm_kernel(inputs[0], inputs[1], inputs[2], alpha, beta, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Addmv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            return cuda::addmv_kernel(inputs[0], inputs[1], inputs[2], alpha, beta, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Baddbmm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            double beta = attrs.get_float(AttrKey::Beta, 1.0);
            return cuda::baddbmm_kernel(inputs[0], inputs[1], inputs[2], alpha, beta, get_cuda_stream(attrs));
        });

    // =========================================================================
    // Log-Cumulative-Sum-Exp
    // =========================================================================
    table.register_single_output_kernel(OpId::Logcumsumexp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, 0);
            return cuda::logcumsumexp_kernel(inputs[0], dim, get_cuda_stream(attrs));
        });

    // =========================================================================
    // Bincount
    // =========================================================================
    table.register_single_output_kernel(OpId::Bincount,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t minlength = attrs.get_int(AttrKey::Minlength, 0);
            const Tensor* weights = (inputs.size() > 1) ? &inputs[1] : nullptr;
            return cuda::bincount_kernel(inputs[0], weights, minlength, get_cuda_stream(attrs));
        });

    // =========================================================================
    // New Element-wise Math Operations
    // =========================================================================

    // Unary ops
    table.register_single_output_kernel(OpId::Rsqrt, cuda::rsqrt_dispatch);
    table.register_single_output_kernel(OpId::Square, cuda::square_dispatch);
    table.register_single_output_kernel(OpId::Asinh, cuda::asinh_dispatch);
    table.register_single_output_kernel(OpId::Acosh, cuda::acosh_dispatch);
    table.register_single_output_kernel(OpId::Atanh, cuda::atanh_dispatch);

    // Binary ops
    table.register_single_output_kernel(OpId::Hypot, cuda::hypot_dispatch);
    table.register_single_output_kernel(OpId::Copysign, cuda::copysign_dispatch);
    table.register_single_output_kernel(OpId::Nextafter, cuda::nextafter_dispatch);
    table.register_single_output_kernel(OpId::Gcd, cuda::gcd_dispatch);
    table.register_single_output_kernel(OpId::Lcm, cuda::lcm_dispatch);
    table.register_single_output_kernel(OpId::Igamma, cuda::igamma_dispatch);
    table.register_single_output_kernel(OpId::Igammac, cuda::igammac_dispatch);

    // Ternary ops with alpha attribute
    table.register_single_output_kernel(OpId::Addcmul,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            return cuda::addcmul_kernel(inputs[0], inputs[1], inputs[2], alpha, get_cuda_stream(attrs));
        });
    table.register_single_output_kernel(OpId::Addcdiv,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double alpha = attrs.get_float(AttrKey::Alpha, 1.0);
            return cuda::addcdiv_kernel(inputs[0], inputs[1], inputs[2], alpha, get_cuda_stream(attrs));
        });

    // =========================================================================
    // New Reduction Operations (CumMax, CumMin, Fmax, Fmin, Isin, Kthvalue, etc.)
    // =========================================================================

    table.register_kernel(OpId::CumMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = cuda::cummax_kernel(inputs[0], dim, get_cuda_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_kernel(OpId::CumMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        auto [values, indices] = cuda::cummin_kernel(inputs[0], dim, get_cuda_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Fmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::fmax_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Fmin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::fmin_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Isin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::isin_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
        });

    table.register_kernel(OpId::Kthvalue, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t k = attrs.get_int(AttrKey::K, 1);
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
        auto [values, indices] = cuda::kthvalue_kernel(inputs[0], k, dim, keepdim, get_cuda_stream(attrs));
        return std::vector<Tensor>{values, indices};
    });

    table.register_single_output_kernel(OpId::Quantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return cuda::quantile_kernel(inputs[0], q, dim, keepdim, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Nanquantile,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            double q = attrs.get_float(AttrKey::Alpha, 0.5);
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            bool keepdim = attrs.get_bool(AttrKey::Keepdim, false);
            return cuda::nanquantile_kernel(inputs[0], q, dim, keepdim, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Nanmedian,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t dim = attrs.get_int(AttrKey::Dim, -1);
            return cuda::nanmedian_kernel(inputs[0], dim, false, get_cuda_stream(attrs));
        });

    table.register_single_output_kernel(OpId::Histc,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            int64_t bins = attrs.get_int(AttrKey::N, 100);
            double min_val = attrs.get_float(AttrKey::Alpha, 0.0);
            double max_val = attrs.get_float(AttrKey::Beta, 0.0);
            return cuda::histc_kernel(inputs[0], bins, min_val, max_val, get_cuda_stream(attrs));
        });

    table.register_kernel(OpId::UniqueConsecutive, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool return_inverse = attrs.get_bool(AttrKey::Keepdim, false);
        auto [unique_vals, inverse, counts] = cuda::unique_consecutive_kernel(
            inputs[0], return_inverse, get_cuda_stream(attrs));
        return std::vector<Tensor>{unique_vals, inverse, counts};
    });

    table.register_single_output_kernel(OpId::SegmentReduce, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t axis = attrs.get_int(AttrKey::Dim, 0);
        std::string reduce = std::string(attrs.get_string(AttrKey::Reduction, "sum"));
        return cuda::segment_reduce_kernel(inputs[0], inputs[1], reduce, axis, get_cuda_stream(attrs));
    });

    // =========================================================================
    // TakeAlongDim
    // =========================================================================
    table.register_single_output_kernel(OpId::TakeAlongDim, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, 0);
        return cuda::take_along_dim_kernel(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });

    // =========================================================================
    // MaskedScatter
    // =========================================================================
    table.register_single_output_kernel(OpId::MaskedScatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::masked_scatter_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs));
    });

    // =========================================================================
    // TrilIndices
    // =========================================================================
    table.register_single_output_kernel(OpId::TrilIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return cuda::tril_indices_kernel(row, col, offset, get_cuda_stream(attrs));
    });

    // =========================================================================
    // TriuIndices
    // =========================================================================
    table.register_single_output_kernel(OpId::TriuIndices, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t row = attrs.get_int(AttrKey::M, 0);
        int64_t col = attrs.get_int(AttrKey::N, 0);
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        return cuda::triu_indices_kernel(row, col, offset, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 2D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = cuda::fractional_maxpool2d_forward_kernel(inputs[0], out_h, out_w, samples, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::fractional_maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Fractional Max Pool 3D
    // =========================================================================
    table.register_kernel(OpId::FractionalMaxPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        const Tensor* samples = (inputs.size() > 1) ? &inputs[1] : nullptr;
        auto [output, indices] = cuda::fractional_maxpool3d_forward_kernel(inputs[0], out_d, out_h, out_w, samples, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::FractionalMaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::fractional_maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Max Unpool 2D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cuda::max_unpool2d_forward_kernel(inputs[0], inputs[1], out_h, out_w, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::max_unpool2d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Phase 9: Max Unpool 3D
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_d = attrs.get_int(AttrKey::OutputSizeD, 1);
        int64_t out_h = attrs.get_int(AttrKey::OutputSizeH, 1);
        int64_t out_w = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cuda::max_unpool3d_forward_kernel(inputs[0], inputs[1], out_d, out_h, out_w, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::max_unpool3d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // New ops: unary math, bool predicates, binary math, structural
    // =========================================================================
    table.register_single_output_kernel(OpId::Deg2Rad, cuda::deg2rad_dispatch);
    table.register_single_output_kernel(OpId::Rad2Deg, cuda::rad2deg_dispatch);
    table.register_single_output_kernel(OpId::Logit, cuda::logit_dispatch);
    table.register_single_output_kernel(OpId::Signbit, cuda::signbit_dispatch);
    table.register_single_output_kernel(OpId::IsPosInf, cuda::isposinf_dispatch);
    table.register_single_output_kernel(OpId::IsNegInf, cuda::isneginf_dispatch);
    table.register_single_output_kernel(OpId::IsReal, cuda::isreal_dispatch);
    table.register_single_output_kernel(OpId::FloatPower, cuda::float_power_dispatch);
    table.register_single_output_kernel(OpId::Xlog1py, cuda::xlog1py_dispatch);
    table.register_single_output_kernel(OpId::Ldexp, cuda::ldexp_dispatch);
    table.register_single_output_kernel(OpId::Frexp, cuda::frexp_dispatch);
    table.register_single_output_kernel(OpId::DiagEmbed, cuda::diag_embed_dispatch);
    table.register_single_output_kernel(OpId::Diagflat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t offset = attrs.get_int(AttrKey::Diagonal, 0);
        Tensor flat = inputs[0].reshape({-1});
        if (!flat.is_contiguous()) flat = flat.contiguous();
        return cuda::diag_kernel(flat, offset, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::NanVar, cuda::nanvar_dispatch);
    table.register_single_output_kernel(OpId::NanStd, cuda::nanstd_dispatch);

    // =========================================================================
    // New math operations (OpId 680-688)
    // =========================================================================
    table.register_single_output_kernel(OpId::LogAddExp, cuda::logaddexp_dispatch);
    table.register_single_output_kernel(OpId::LogAddExp2, cuda::logaddexp2_dispatch);
    table.register_single_output_kernel(OpId::XLogY, cuda::xlogy_dispatch);
    table.register_single_output_kernel(OpId::I0e, cuda::i0e_dispatch);
    table.register_single_output_kernel(OpId::I1e, cuda::i1e_dispatch);
    table.register_single_output_kernel(OpId::Entr, cuda::entr_dispatch);
    table.register_single_output_kernel(OpId::SphericalBesselJ0, cuda::spherical_bessel_j0_dispatch);
    table.register_single_output_kernel(OpId::CosineSimilarity, cuda::cosine_similarity_dispatch);
    table.register_single_output_kernel(OpId::Renorm, cuda::renorm_dispatch);

    // =========================================================================
    // Nested Tensor Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::NestedSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::nested_softmax_cuda(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = attrs.get_int(AttrKey::Dim, -1);
        return cuda::nested_log_softmax_cuda(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::nested_sum_cuda(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedMean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::nested_mean_cuda(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = attrs.get_float(AttrKey::Eps, 1e-5f);
        return cuda::nested_layer_norm_cuda(inputs[0], inputs[1], inputs[2], inputs[3], eps, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLinear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const Tensor* bias = (inputs.size() > 3) ? &inputs[3] : nullptr;
        return cuda::nested_linear_cuda(inputs[0], inputs[2], bias, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float scale = attrs.get_float(AttrKey::Scale, 1.0f);
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        return cuda::nested_attention_cuda(inputs[0], inputs[1], inputs[2],
                                           inputs[3], inputs[4], scale, causal, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedToPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t max_len = attrs.get_int(AttrKey::MaxLen, 0);
        float padding_value = attrs.get_float(AttrKey::PaddingValue, 0.0f);
        return cuda::nested_to_padded_cuda(inputs[0], inputs[1], max_len, padding_value, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedFromPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::nested_from_padded_cuda(inputs[0], inputs[1], get_cuda_stream(attrs));
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
            float scale = attrs.get_float(AttrKey::Scale, 1.0f);
            bool causal = attrs.get_bool(AttrKey::Causal, false);
            return cuda::nested_attention_backward_cuda(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                inputs[5], inputs[6], scale, causal, get_cuda_stream(attrs));
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
