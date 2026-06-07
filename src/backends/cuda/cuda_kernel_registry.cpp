/**
 * @file cuda_kernel_registry.cpp
 * @brief CUDA kernel registration for O(1) dispatch
 *
 * Registers all CUDA kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/nn/layers/flex_attention.hpp"  // F6: process-wide score_mod registry
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/backend/attr_macros.hpp"
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
#include "cuda_error.hpp"
#include <cuda_runtime.h>
#include <array>
#include <cmath>
#include <cstdlib>
#include <limits>
#include <sstream>
#include <tuple>

#include "tenzor/backend/dtype_from_string.hpp"

namespace tenzor {


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
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, Device device, cudaStream_t stream) -> Tensor;
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
    auto leaky_relu_kernel(const Tensor& input, double alpha, cudaStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, double alpha, cudaStream_t stream) -> Tensor;
    auto elu_kernel(const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, cudaStream_t stream) -> Tensor;
    auto selu_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto mish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, cudaStream_t stream) -> Tensor;
    auto hardswish_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto hardsigmoid_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
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

    // CTC loss
    auto ctc_loss_forward_kernel(const Tensor& log_probs, const Tensor& targets,
                                 const Tensor& input_lengths, const Tensor& target_lengths,
                                 int64_t blank, bool zero_infinity, cudaStream_t stream)
        -> std::vector<Tensor>;

    // Embedding operations
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, cudaStream_t stream) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices, int64_t num_embeddings, cudaStream_t stream) -> Tensor;
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets, const std::string& mode, int64_t embedding_dim, bool include_last_offset, cudaStream_t stream) -> std::vector<Tensor>;
    auto embedding_bag_backward_kernel(const Tensor& grad_output, const Tensor& indices, const Tensor& offsets, const OpAttributes& attrs, cudaStream_t stream) -> Tensor;

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
    auto fused_conv2d_bn_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor* bias, const Tensor& bn_mean, const Tensor& bn_var, const Tensor& bn_gamma, const Tensor& bn_beta, int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w, float eps) -> Tensor;
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
                          const Tensor& bias, const Tensor& h0,
                          const Tensor& bias_hh = {}) -> std::vector<Tensor>;
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
    auto grid_sample_backward_cuda(const Tensor& grad_output,
                                   const Tensor& input, const Tensor& grid,
                                   const std::string& mode,
                                   const std::string& padding_mode,
                                   bool align_corners)
        -> std::pair<Tensor, Tensor>;
    auto affine_grid_backward_cuda(const Tensor& grad_grid,
                                   const std::vector<int64_t>& size,
                                   bool align_corners) -> Tensor;
    auto interpolate_cuda(const Tensor& input, const std::vector<int64_t>& size, const std::string& mode, bool align_corners) -> Tensor;
    auto interpolate_backward_cuda(const Tensor& grad_output, const std::vector<int64_t>& input_size, const std::string& mode, bool align_corners) -> Tensor;
    auto unfold_cuda(const Tensor& input,
                     int64_t kernel_h, int64_t kernel_w,
                     int64_t stride_h, int64_t stride_w,
                     int64_t padding_h, int64_t padding_w,
                     int64_t dilation_h, int64_t dilation_w,
                     cudaStream_t stream) -> Tensor;
    auto fold_cuda(const Tensor& input, const std::vector<int64_t>& output_size,
                   int64_t kernel_h, int64_t kernel_w,
                   int64_t stride_h, int64_t stride_w,
                   int64_t padding_h, int64_t padding_w,
                   int64_t dilation_h, int64_t dilation_w,
                   cudaStream_t stream) -> Tensor;
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
    auto gamma_sample_kernel(const Tensor& concentration, const Tensor& rate, cudaStream_t stream) -> Tensor;
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

    // Fused Attention operation (returns {output, logsumexp}).
    // Defaults live in include/tenzor/backend/fused_ops.hpp per
    // docs/internals/attention-contract.md.
    auto fused_attention_cuda(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        float scale,
        bool causal,
        float dropout_p,
        uint32_t rng_seed
    ) -> std::pair<Tensor, Tensor>;

    // Fused Flash Attention backward (tiled, memory-efficient).
    // Defaults live in include/tenzor/backend/fused_ops.hpp.
    auto flash_attention_backward_cuda(
        const Tensor& dO,
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        const Tensor& O,
        const Tensor& L,
        float scale,
        bool causal,
        float dropout_p,
        const Tensor& philox_seed,
        const Tensor& philox_offset
    ) -> std::vector<Tensor>;

    // Audit A.11 — native Float64 FlashAttention forward / backward.
    // Lives in kernels/flash_attention_f64.cu. The mainline `fused_attention_cuda`
    // upcasts FP16/BF16 → FP32 internally; FP64 cannot share that path without
    // destroying precision, so a dedicated `double`-typed kernel set runs here.
    // Dropout is intentionally unsupported in the FP64 path (gradcheck-only use).
    auto fused_attention_cuda_f64(
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        double scale,
        bool causal
    ) -> std::pair<Tensor, Tensor>;

    auto flash_attention_backward_cuda_f64(
        const Tensor& dO,
        const Tensor& Q,
        const Tensor& K,
        const Tensor& V,
        const Tensor& O,
        double scale,
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
    auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad, cudaStream_t stream) -> Tensor;
    auto avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad, cudaStream_t stream) -> Tensor;

    // 1D Pooling operations
    // Q.5: per-axis std::array<int64_t, 1> signatures (1D has only one spatial
    // axis but the API matches the per-axis sweep across 2D/3D).
    auto maxpool1d_forward_kernel(const Tensor& input, std::array<int64_t, 1> kernel_size, std::array<int64_t, 1> stride, std::array<int64_t, 1> padding, std::array<int64_t, 1> dilation, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto maxpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto avgpool1d_forward_kernel(const Tensor& input, std::array<int64_t, 1> kernel_size, std::array<int64_t, 1> stride, std::array<int64_t, 1> padding, bool count_include_pad, cudaStream_t stream) -> Tensor;
    auto avgpool1d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, std::array<int64_t, 1> kernel_size, std::array<int64_t, 1> stride, std::array<int64_t, 1> padding, bool count_include_pad, cudaStream_t stream) -> Tensor;

    // 3D Pooling operations
    // Q.5: per-axis std::array<int64_t, 3> signatures honour asymmetric D/H/W
    // stride/padding (previously silently collapsed via scalar parameters).
    auto maxpool3d_forward_kernel(const Tensor& input, std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto maxpool3d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;
    auto avgpool3d_forward_kernel(const Tensor& input, std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, bool count_include_pad, cudaStream_t stream) -> Tensor;
    auto avgpool3d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, std::array<int64_t, 3> kernel_size, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, bool count_include_pad, cudaStream_t stream) -> Tensor;

#ifdef TENZOR_HAS_CUDNN
    // cuDNN pooling operations (faster than custom kernels)
    auto cudnn_maxpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> std::pair<Tensor, Tensor>;
    auto cudnn_maxpool2d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& output, int64_t kernel_size, int64_t stride, int64_t padding, cudaStream_t stream) -> Tensor;
    auto cudnn_avgpool2d_forward(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad, cudaStream_t stream) -> Tensor;
    auto cudnn_avgpool2d_backward(const Tensor& grad_output, const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding, bool count_include_pad, cudaStream_t stream) -> Tensor;

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
    auto fill_kernel(const Tensor& tensor, double value, cudaStream_t stream) -> Tensor;
    auto strided_fill_kernel(Tensor& self, double value, cudaStream_t stream) -> void;

    // Runtime cuDNN availability check
    bool is_cudnn_available() noexcept;
    bool is_cudnn_frontend_available() noexcept;

    // Conv2d backward and transpose
    auto conv2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight, int64_t stride, int64_t padding, int64_t dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    // RR.9: only the per-axis overload remains; the dispatcher calls it directly.
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride_h, int64_t stride_w, int64_t padding_h, int64_t padding_w, int64_t output_padding_h, int64_t output_padding_w, int64_t dilation_h, int64_t dilation_w, int64_t groups, cudaStream_t stream) -> Tensor;
    auto depthwise_conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> Tensor;
    // Phase 2.1: per-axis overload.
    auto depthwise_conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride_h, int64_t stride_w, int64_t pad_h, int64_t pad_w, int64_t dil_h, int64_t dil_w, cudaStream_t stream) -> Tensor;
    auto depthwise_conv1d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, cudaStream_t stream) -> Tensor;
    auto depthwise_conv3d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t sD, int64_t sH, int64_t sW, int64_t pD, int64_t pH, int64_t pW, int64_t dD, int64_t dH, int64_t dW, cudaStream_t stream) -> Tensor;

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
    auto cudnn_conv3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv3d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // ConvTranspose3d (cuDNN Nd)
    auto cudnn_conv_transpose3d_forward(const Tensor& input, const Tensor& weight, const Tensor* bias, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv_transpose3d_backward(const Tensor& grad_output, const Tensor& input, const Tensor& weight, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation, int64_t groups, bool compute_grad_input, bool compute_grad_weight, bool compute_grad_bias, cudaStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    // ABI-safe wrappers (single Tensor return avoids tuple across nvcc/g++ boundary)
    auto cudnn_conv_transpose3d_backward_input(const Tensor& grad_output, const Tensor& input, const Tensor& weight, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv_transpose3d_backward_weight(const Tensor& grad_output, const Tensor& input, const Tensor& weight, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation, int64_t groups, cudaStream_t stream) -> Tensor;
    auto cudnn_conv_transpose3d_backward_bias(const Tensor& grad_output, const Tensor& input, const Tensor& weight, std::array<int64_t, 3> stride, std::array<int64_t, 3> padding, std::array<int64_t, 3> output_padding, std::array<int64_t, 3> dilation, int64_t groups, cudaStream_t stream) -> Tensor;
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
    auto max_unpool1d_forward_kernel(const Tensor& input, const Tensor& indices, int64_t out_l, cudaStream_t stream) -> Tensor;
    auto max_unpool1d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape, cudaStream_t stream) -> Tensor;

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
    Tensor hardswish_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);
    Tensor hardsigmoid_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

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
    std::vector<Tensor> frexp_dispatch(std::span<const Tensor> inputs, const OpAttributes& attrs);

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
        int64_t kh_size, int64_t kw_size, int64_t sH, int64_t sW,
        int64_t pH, int64_t pW, int64_t dH, int64_t dW,
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
    // Forward-only (backward autograd-composed via clamp+mul, matching CPU).
    table.register_single_output_kernel(OpId::Hardswish, cuda::hardswish_dispatch);
    table.register_single_output_kernel(OpId::Hardsigmoid, cuda::hardsigmoid_dispatch);

    // Parameterized activations (keep lambdas for attribute parsing)
    table.register_single_output_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);  // keep F64 precision
        return cuda::leaky_relu_kernel(inputs[0], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double alpha = attrs.get_float(AttrKey::Alpha, 0.01);
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
        double value = attrs.get_float(AttrKey::Value, 0.0);
        return cuda::fill_kernel(inputs[0], value, get_cuda_stream(attrs));
    });
    table.register_inplace_kernel(OpId::StridedFill, [](Tensor& self, std::span<const Tensor>, const OpAttributes& attrs) -> Tensor& {
        double value = attrs.get_float(AttrKey::Value, 0.0);
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
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        auto [output, indices] = cuda::cudnn_maxpool2d_forward(inputs[0],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        return cuda::cudnn_avgpool2d_forward(inputs[0],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0,
            get_cuda_stream(attrs));
    });
#else
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation    = ::tenzor::backend::attrs::dilation_2d(attrs);
        // Phase 2.1: non-cuDNN CUDA maxpool2d kernel is scalar-only; reject
        // asymmetric input rather than silently collapsing to symmetric.
        if (kernel_size[0] != kernel_size[1] || stride[0] != stride[1] ||
            padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "MaxPool2dForward (CUDA non-cuDNN): backend kernel only supports symmetric "
                "kernel/stride/padding/dilation; got kernel=" + std::to_string(kernel_size[0]) + "x" + std::to_string(kernel_size[1]) +
                ", stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        auto [output, indices] = cuda::maxpool2d_forward_kernel(inputs[0],
            kernel_size[0], stride[0], padding[0], dilation[0], get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        if (kernel_size[0] != kernel_size[1] || stride[0] != stride[1] ||
            padding[0] != padding[1]) {
            throw std::invalid_argument(
                "AvgPool2dForward (CUDA non-cuDNN): backend kernel only supports symmetric "
                "kernel/stride/padding; got kernel=" + std::to_string(kernel_size[0]) + "x" + std::to_string(kernel_size[1]) +
                ", stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        return cuda::avgpool2d_forward_kernel(inputs[0],
            kernel_size[0], stride[0], padding[0],
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0, get_cuda_stream(attrs));
    });
#endif

    // =========================================================================
    // Pooling Backward Operations
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, indices, input, output]
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        return cuda::cudnn_maxpool2d_backward(inputs[0], inputs[2], inputs[3],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, input]
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        return cuda::cudnn_avgpool2d_backward(inputs[0], inputs[1],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0,
            get_cuda_stream(attrs));
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
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_2d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::Stride, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_2d(attrs);
        if (kernel_size[0] != kernel_size[1] || stride[0] != stride[1] ||
            padding[0] != padding[1]) {
            throw std::invalid_argument(
                "AvgPool2dBackward (CUDA non-cuDNN): backend kernel only supports symmetric "
                "kernel/stride/padding; got kernel=" + std::to_string(kernel_size[0]) + "x" + std::to_string(kernel_size[1]) +
                ", stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        return cuda::avgpool2d_backward_kernel(inputs[0], input_shape,
            kernel_size[0], stride[0], padding[0],
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0, get_cuda_stream(attrs));
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
        // attrs: scale, use_cudnn_sdpa, causal (per docs/internals/attention-contract.md)
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal_attr = attrs.get_bool(AttrKey::Causal, false);

#ifdef TENZOR_HAS_CUDNN_FRONTEND
        // Check if cuDNN SDPA is requested and input is 4D. Causal flag is now
        // plumbed through (audit C4/M5 fix): the cuDNN graph is rebuilt with
        // set_causal_mask(causal); the cache key includes causal so a non-causal
        // build never silently serves a causal call.
        bool use_cudnn_sdpa = attrs.get_bool(AttrKey::UseCudnnSdpa, false);
        if (use_cudnn_sdpa && inputs[0].shape().size() == 4) {
            auto output = cuda::cudnn_sdpa_forward(inputs[0], inputs[1], inputs[2], scale, causal_attr);
            return std::vector<Tensor>{output};
        }
#endif

        // 3D input or cuDNN not available: use custom flash attention kernel.
        // For 4D inputs falling through (cuDNN unavailable / not requested),
        // reshape to 3D before dispatch — fused_attention_cuda assumes 3D and
        // would otherwise read num_heads as seq_len_q (audit H7).
        // GQA: when H_kv < H_q, broadcast K/V along head dim before the 3D
        // collapse so the kernel sees [B*H_q, S_k, D] for both. The kernel
        // itself doesn't have GQA index math (it would need q_heads_per_kv_head
        // as a kernel param); host-level expand+contiguous is contract-correct
        // at the cost of memory (per attention-contract.md GQA section).
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
                        "FusedAttention CUDA: H_q must be a multiple of H_kv; got " +
                        std::to_string(h) + " and " + std::to_string(h_kv));
                }
                int64_t reps = h / h_kv;
                // Broadcast via unsqueeze + expand + reshape.
                std::vector<Tensor> ku = {Kc};
                NewOpAttributes us_attrs;
                us_attrs.set(AttrKey::Dim, static_cast<int64_t>(2));
                Tensor Ku = tenzor::dispatch(OpId::Unsqueeze, ku, us_attrs)[0];
                Tensor Vu = tenzor::dispatch(OpId::Unsqueeze, std::vector<Tensor>{Vc}, us_attrs)[0];
                std::vector<int64_t> exp_k = {b, h_kv, reps, sk, d};
                std::vector<int64_t> exp_v = {b, h_kv, reps, sk, d_v};
                std::string s_k, s_v;
                for (size_t i = 0; i < exp_k.size(); ++i) { if (i) s_k += ","; s_k += std::to_string(exp_k[i]); }
                for (size_t i = 0; i < exp_v.size(); ++i) { if (i) s_v += ","; s_v += std::to_string(exp_v[i]); }
                NewOpAttributes ek_attrs; ek_attrs.set(AttrKey::Shape, s_k);
                NewOpAttributes ev_attrs; ev_attrs.set(AttrKey::Shape, s_v);
                Tensor Ke = tenzor::dispatch(OpId::Expand, std::vector<Tensor>{Ku}, ek_attrs)[0];
                Tensor Ve = tenzor::dispatch(OpId::Expand, std::vector<Tensor>{Vu}, ev_attrs)[0];
                Kc = Ke.contiguous().reshape({b, h, sk, d});
                Vc = Ve.contiguous().reshape({b, h, sk, d_v});
            }

            Tensor Q3 = (Qi.is_contiguous() ? Qi : Qi.contiguous()).reshape({b * h, sq, d});
            Tensor K3 = Kc.reshape({b * h, sk, d});
            Tensor V3 = Vc.reshape({b * h, sk, d_v});
            auto [out3, lse3] = cuda::fused_attention_cuda(Q3, K3, V3, scale, causal_attr, 0.0f, 0u);
            return std::vector<Tensor>{out3.reshape({b, h, sq, d_v})};
        }
        auto [output, lse] = cuda::fused_attention_cuda(Qi, Ki, Vi, scale, causal_attr, 0.0f, 0u);
        return std::vector<Tensor>{output};
    });

    // =========================================================================
    // Flash Attention (memory-efficient tiled attention)
    // =========================================================================
    table.register_kernel(OpId::FlashAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Per docs/internals/attention-contract.md, returns 4 tensors:
        // [output, lse_f32, philox_seed_int64, philox_offset_int64].
        // Causal mask applied inline in flash_attention_v2_kernel (audit C1).
        // Philox dropout applied inline (audit M4 follow-up). seed/offset
        // returned only when dropout fires so backward can replay.
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        float dropout_p = static_cast<float>(attrs.get_float(AttrKey::DropoutP, 0.0));
        bool is_training = attrs.get_bool(AttrKey::IsTraining, attrs.get_bool(AttrKey::Training, false));
        bool apply_dropout = dropout_p > 0.0f && is_training;

        uint32_t rng_seed = 0u;
        if (apply_dropout) {
            int64_t seed_in = attrs.get_int(AttrKey::Seed, 0);
            // Derive seed from the Q data pointer hash if caller didn't supply one.
            // Lower 32 bits — Philox treats the seed as a 32-bit key.
            uint64_t seed64 = (seed_in != 0)
                ? static_cast<uint64_t>(seed_in)
                : (reinterpret_cast<uintptr_t>(inputs[0].data_ptr()) * 2654435761ULL);
            rng_seed = static_cast<uint32_t>(seed64);
            if (rng_seed == 0) rng_seed = 1u;  // 0 disables in-kernel; force non-zero.
        }

        // fused_attention_cuda expects 3D `(batch_heads, seq_len, head_dim)`.
        // The autograd-side dispatch passes Q/K/V 4D `(B, H, S, D)` — collapse
        // the leading two dims for the kernel call and restore on output.
        bool is_4d = (inputs[0].shape().size() == 4);
        Tensor Qi = inputs[0], Ki = inputs[1], Vi = inputs[2];
        std::vector<int64_t> orig_q_shape;
        if (is_4d) {
            orig_q_shape.assign(inputs[0].shape().begin(), inputs[0].shape().end());
            int64_t b = orig_q_shape[0], h = orig_q_shape[1];
            int64_t sq = orig_q_shape[2], d = orig_q_shape[3];
            int64_t sk = inputs[1].shape()[2];
            int64_t dv = inputs[2].shape()[3];
            Qi = tenzor::reshape(inputs[0], std::vector<int64_t>{b * h, sq, d});
            Ki = tenzor::reshape(inputs[1], std::vector<int64_t>{b * h, sk, d});
            Vi = tenzor::reshape(inputs[2], std::vector<int64_t>{b * h, sk, dv});
        }

        // Audit A.11 — Float64 path: route to the native FP64 CUDA kernel
        // (no FP32 upcast). Dropout is unsupported in FP64 mode; the
        // autograd-level dispatcher already routes Float64 only when
        // dropout_p == 0.0 (see src/autograd/function_attention.cpp).
        Tensor output, lse;
        if (Qi.dtype() == DType::Float64) {
            int64_t f64_head_dim = Qi.shape().back();
            bool f64_native_supported = (f64_head_dim == 16 || f64_head_dim == 32 ||
                                          f64_head_dim == 48 || f64_head_dim == 64 ||
                                          f64_head_dim == 80 || f64_head_dim == 96 ||
                                          f64_head_dim == 128);
            if (apply_dropout) {
                throw std::runtime_error(
                    "FlashAttention CUDA: dropout is not supported with Float64 inputs.");
            }
            if (f64_native_supported) {
                auto [o64, lse64] = cuda::fused_attention_cuda_f64(
                    Qi, Ki, Vi, static_cast<double>(scale), causal);
                output = o64;
                lse    = lse64;
            } else {
                // FP64 fallback for unsupported head_dim: composed-ops on GPU
                // via tenzor:: dispatch, which calls native FP64 BMM/softmax
                // kernels on CUDA. No FP32 round-trip — preserves precision
                // for gradcheck. This is NOT a CPU fallback; every op below
                // dispatches to CUDA double-precision kernels.
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
                // LSE is the true row log-sum-exp of the (masked) scores,
                // computed on-device via the FP64 LogSumExp kernel and cast to
                // Float32 per the output contract. Masked positions carry -inf,
                // so exp() contributes 0 — the causal LSE is exact. (Previously
                // this returned a zero-filled placeholder, silently corrupting
                // any consumer that reads the second FlashAttention output.)
                lse = tenzor::logsumexp(scores, /*dim=*/-1, /*keepdim=*/false)
                          .to(DType::Float32);
            }
        } else {
            auto [o, l] = cuda::fused_attention_cuda(
                Qi, Ki, Vi, scale, causal,
                apply_dropout ? dropout_p : 0.0f, rng_seed);
            output = o;
            lse    = l;
        }

        if (is_4d) {
            int64_t b = orig_q_shape[0], h = orig_q_shape[1];
            int64_t sq = orig_q_shape[2], dv = inputs[2].shape()[3];
            output = tenzor::reshape(output,
                std::vector<int64_t>{b, h, sq, dv});
            // lse is (batch_heads, seq_len_q) → (B, H, sq)
            lse = tenzor::reshape(lse, std::vector<int64_t>{b, h, sq});
        }

        Tensor seed_t, offset_t;
        if (apply_dropout) {
            seed_t = tenzor::zeros({1}, DType::Int64, inputs[0].device());
            offset_t = tenzor::zeros({1}, DType::Int64, inputs[0].device());
            // Tiny D2H/H2D — write seed/offset scalars from host.
            int64_t seed_host = static_cast<int64_t>(rng_seed);
            int64_t offset_host = 0;
            cudaMemcpy(seed_t.data_ptr(), &seed_host, sizeof(int64_t), cudaMemcpyHostToDevice);
            cudaMemcpy(offset_t.data_ptr(), &offset_host, sizeof(int64_t), cudaMemcpyHostToDevice);
        }
        return std::vector<Tensor>{output, lse, seed_t, offset_t};
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

            // Audit A.11 — native Float64 backward. Supports head_dim
            // {16, 32, 48, 64, 80, 96, 128}; falls through to composed-ops for
            // other head_dims (which is fine in FP64 because every op below
            // dispatches natively in `double` on CUDA).
            bool f64_supported = (head_dim == 16 || head_dim == 32 || head_dim == 48 ||
                                   head_dim == 64 || head_dim == 80 || head_dim == 96 ||
                                   head_dim == 128);
            if (Q.dtype() == DType::Float64 && f64_supported) {
                // Forward saves Q/K/V/O in their original (possibly 4D) layout.
                // The FP64 kernel below expects 3D (BH, S, D); collapse and
                // restore around the call.
                bool is_4d = (Q.shape().size() == 4);
                Tensor Qi = Q, Ki = K, Vi = V, Oi = O, dOi = dO;
                int64_t b = 0, h = 0;
                if (is_4d) {
                    b = Q.shape()[0]; h = Q.shape()[1];
                    int64_t sq = Q.shape()[2], d = Q.shape()[3];
                    int64_t sk = K.shape()[2], dv = V.shape()[3];
                    Qi  = tenzor::reshape(Q,  std::vector<int64_t>{b * h, sq, d});
                    Ki  = tenzor::reshape(K,  std::vector<int64_t>{b * h, sk, d});
                    Vi  = tenzor::reshape(V,  std::vector<int64_t>{b * h, sk, dv});
                    Oi  = tenzor::reshape(O,  std::vector<int64_t>{b * h, sq, dv});
                    dOi = tenzor::reshape(dO, std::vector<int64_t>{b * h, sq, dv});
                }
                auto grads = cuda::flash_attention_backward_cuda_f64(
                    dOi, Qi, Ki, Vi, Oi, static_cast<double>(scale), causal);
                if (is_4d) {
                    int64_t sq = Q.shape()[2], d = Q.shape()[3], sk = K.shape()[2], dv = V.shape()[3];
                    grads[0] = tenzor::reshape(grads[0], std::vector<int64_t>{b, h, sq, d});
                    grads[1] = tenzor::reshape(grads[1], std::vector<int64_t>{b, h, sk, d});
                    grads[2] = tenzor::reshape(grads[2], std::vector<int64_t>{b, h, sk, dv});
                }
                return grads;
            }

            // Composed-ops fallback for unsupported head_dim or missing L
            Tensor Kt = tenzor::transpose(K, -1, -2);
            Tensor scores = tenzor::bmm(Q, Kt);

            auto scores_shape = std::vector<int64_t>(scores.shape().begin(), scores.shape().end());
            Tensor scale_t = tenzor::full(scores_shape, static_cast<double>(scale),
                                           scores.dtype(), scores.device());
            scores = tenzor::mul(scores, scale_t);

            if (causal) {
                // Per attention-contract.md sentinel rule: use -INFINITY,
                // never -1e9. The latter saturates to -65504 in FP16, leaving
                // exp(-65504 + something) > 0 after softmax — leaks gradient
                // mass through masked positions (audit Systemic #3, Vulkan C15).
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
    // FlexAttention (built-in score_mod registry; full programmable backward
    // arrives in M8 with native CUDA block-sparse implementation)
    // =========================================================================
    // Per docs/internals/attention-contract.md, ScoreModId encodes a built-in:
    //   0 = identity (== FusedAttention)
    //   1 = causal  (== FusedAttention with causal=true)
    // Other IDs throw on this backend until the M8 native path lands. The same
    // composed FlashAttentionBackward kernel handles backward for both modes
    // (mathematically identical).
    table.register_kernel(OpId::FlexAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);

        // ScoreModId 0/1 reduce to FusedAttention.
        if (score_mod_id == 0 || score_mod_id == 1) {
            bool causal = (score_mod_id == 1);
            const Tensor& Qi = inputs[0];
            const Tensor& Ki = inputs[1];
            const Tensor& Vi = inputs[2];
            if (Qi.shape().size() == 4) {
                int64_t b = Qi.shape()[0], h = Qi.shape()[1], sq = Qi.shape()[2], d = Qi.shape()[3];
                int64_t sk = Ki.shape()[2];
                Tensor Q3 = (Qi.is_contiguous() ? Qi : Qi.contiguous()).reshape({b * h, sq, d});
                Tensor K3 = (Ki.is_contiguous() ? Ki : Ki.contiguous()).reshape({b * h, sk, d});
                Tensor V3 = (Vi.is_contiguous() ? Vi : Vi.contiguous()).reshape({b * h, sk, d});
                auto [out3, lse3] = cuda::fused_attention_cuda(Q3, K3, V3, scale, causal, 0.0f, 0u);
                return std::vector<Tensor>{out3.reshape({b, h, sq, d}), lse3};
            }
            auto [output, lse] = cuda::fused_attention_cuda(Qi, Ki, Vi, scale, causal, 0.0f, 0u);
            return std::vector<Tensor>{output, lse};
        }

        // ScoreModId 2 (sliding_window) — composed-ops via tenzor:: tensor API.
        // Per docs/internals/attention-contract.md: mask (i,j) where
        // |i-j| > WindowSize/2 with -INFINITY. tenzor:: ops dispatch to the
        // CUDA backend automatically since Q/K/V live on CUDA.
        if (score_mod_id == 2) {
            int64_t window_size = attrs.get_int(AttrKey::WindowSize, 0);
            if (window_size <= 0) {
                throw std::invalid_argument(
                    "FlexAttention CUDA: ScoreModId=2 (sliding_window) requires "
                    "AttrKey::WindowSize > 0.");
            }
            const Tensor& Q = inputs[0];
            const Tensor& K = inputs[1];
            const Tensor& V = inputs[2];
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

                // F6: ScoreModId >= 3 routes through the process-wide score_mod
        // registry populated by `tenzor::nn::register_score_mod` — same
        // mechanism the CPU and OneAPI backends use. Forward composes
        // Q@K^T → user functor → softmax → @V via tenzor:: ops (which
        // dispatch to CUDA automatically).
        if (score_mod_id >= 3) {
            auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
            if (!fn) {
                throw std::runtime_error(
                    "FlexAttention CUDA: no user score_mod registered for ScoreModId=" +
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
            "FlexAttention CUDA: ScoreModId=" + std::to_string(score_mod_id) +
            " not recognised (built-ins: 0=identity, 1=causal, 2=sliding_window; "
            "register user IDs >= 3 via tenzor::nn::register_score_mod).");
    });

    table.register_kernel(OpId::FlexAttentionBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        float scale = static_cast<float>(attrs.get_float(AttrKey::Scale, 1.0));
        int64_t score_mod_id = attrs.get_int(AttrKey::ScoreModId, 0);
        // F6: identity (0) and causal (1) — route to the fused FlashAttention
        // backward when applicable. Sliding-window (2) and user-registered
        // functors (>= 3) flow through a composed-ops backward (same pattern
        // as F14 OneAPI).
        if (score_mod_id == 0 || score_mod_id == 1) {
            bool causal = (score_mod_id == 1);
            const Tensor& dO = inputs[0], &Q = inputs[1], &K = inputs[2], &V = inputs[3], &O = inputs[4];
            int64_t head_dim = Q.shape().back();
            bool has_lse = inputs.size() >= 6;
            if (has_lse && (head_dim == 32 || head_dim == 64 || head_dim == 128) && Q.dtype() == DType::Float32) {
                return cuda::flash_attention_backward_cuda(dO, Q, K, V, O, inputs[5],
                                                           scale, causal, 0.0f, Tensor{}, Tensor{});
            }
            OpAttributes bwd_attrs;
            bwd_attrs.set(AttrKey::Scale, static_cast<double>(scale));
            bwd_attrs.set(AttrKey::Causal, causal);
            std::vector<Tensor> bwd_inputs(inputs.begin(), inputs.end());
            return tenzor::dispatch(OpId::FlashAttentionBackward, bwd_inputs, bwd_attrs);
        }

        if (score_mod_id == 2 || score_mod_id >= 3) {
            // Composed backward replaying the forward to recover masked scores
            // (inputs: [dO, Q, K, V, O, (L)]).
            const Tensor& dO = inputs[0];
            const Tensor& Q = inputs[1];
            const Tensor& K = inputs[2];
            const Tensor& V = inputs[3];
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
                        "FlexAttentionBackward CUDA: ScoreModId=2 requires AttrKey::WindowSize > 0.");
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
                Tensor neg_inf = tenzor::full(scores_shape,
                    -std::numeric_limits<float>::infinity(),
                    scores.dtype(), scores.device());
                scores = scores + (outside.to(scores.dtype()) * neg_inf);
            } else {
                auto fn = tenzor::nn::find_registered_score_mod(score_mod_id);
                if (!fn) {
                    throw std::runtime_error(
                        "FlexAttentionBackward CUDA: no user score_mod registered for ScoreModId=" +
                        std::to_string(score_mod_id));
                }
                scores = fn(scores, 0, 0, 0, 0);
            }

            NewOpAttributes sm_attrs;
            sm_attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            std::vector<Tensor> sm_in = {scores};
            Tensor attn = tenzor::dispatch(OpId::Softmax, sm_in, sm_attrs)[0];

            // dV = attn^T @ dO
            Tensor attn_t = tenzor::transpose(attn, -1, -2);
            Tensor dV = tenzor::bmm(attn_t, dO);

            // dAttn = dO @ V^T
            Tensor Vt = tenzor::transpose(V, -1, -2);
            Tensor dAttn = tenzor::bmm(dO, Vt);

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

            Tensor dQ = tenzor::bmm(dScores, K);
            Tensor dScores_t = tenzor::transpose(dScores, -1, -2);
            Tensor dK = tenzor::bmm(dScores_t, Q);

            return {dQ, dK, dV};
        }

        throw std::runtime_error(
            "FlexAttentionBackward CUDA: ScoreModId=" + std::to_string(score_mod_id) +
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
        // inputs: [param, grad, exp_avg, exp_avg_sq, max_exp_avg_sq (optional)]
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

        cuda::fused_adam_step_cuda(
            param, inputs[1], exp_avg, exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, decoupled,
            get_cuda_stream(attrs), max_exp_avg_sq, amsgrad
        );
        return std::vector<Tensor>{param};
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

    // Returns {output, max_indices}; max_indices is the per-(bag,feature) global
    // argmax element index for mode="max" (empty otherwise).
    table.register_kernel(OpId::EmbeddingBagForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [embeddings, offsets]
        // attrs: Mode, EmbeddingDim, IncludeLastOffset
        std::string mode{attrs.get_string(AttrKey::Mode, "sum")};
        int64_t embedding_dim = attrs.get_int(AttrKey::EmbeddingDim, 0);
        bool include_last_offset = attrs.get_bool(AttrKey::IncludeLastOffset, false);
        return cuda::embedding_bag_forward_kernel(inputs[0], inputs[1], mode, embedding_dim, include_last_offset, get_cuda_stream(attrs));
    });

    table.register_kernel(OpId::EmbeddingBagBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, indices (Int64), offsets]
        return std::vector<Tensor>{cuda::embedding_bag_backward_kernel(
            inputs[0], inputs[1], inputs[2], attrs, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // CTC Loss (audit Phase 3.7 — eliminates CPU round-trip)
    // =========================================================================
    table.register_kernel(OpId::CTCLossForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [log_probs (T, N, C), targets (N, S_max),
        //          input_lengths (N,), target_lengths (N,)]
        // attrs:  Blank, ZeroInfinity
        // outputs: [loss_per_sample (N,), raw_grad (T, N, C)]
        int64_t blank = attrs.get_int(AttrKey::Blank, 0);
        bool zero_infinity = attrs.get_bool(AttrKey::ZeroInfinity, false);
        return cuda::ctc_loss_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3],
            blank, zero_infinity, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Fused Operations (optimized combined kernels)
    // =========================================================================
    table.register_single_output_kernel(OpId::FusedConv2dBnReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        const auto stride  = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding = ::tenzor::backend::attrs::padding_2d(attrs);
        float eps = static_cast<float>(attrs.get_float(AttrKey::Eps, 1e-5));
        const Tensor* bias = inputs.size() > 2 && inputs[2].numel() > 0 ? &inputs[2] : nullptr;
        // CPU registration: [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        // CUDA func expects: (input, weight, bias, bn_mean, bn_var, bn_gamma, bn_beta, ...)
        // Per-axis stride/padding (asymmetric supported); all float dtypes handled in-kernel.
        return cuda::fused_conv2d_bn_relu_cuda(inputs[0], inputs[1], bias,
            inputs[5], inputs[6], inputs[3], inputs[4],
            stride[0], stride[1], padding[0], padding[1], eps);
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
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_relu_forward(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_sigmoid_forward(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_tanh_forward(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_fused_conv2d_swish_forward(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, get_cuda_stream(attrs));
    });
#else
    // Fallback: compose conv2d + activation when cuDNN is unavailable.
    // The CUDA conv2d_forward_kernel is scalar-only; reject asymmetric input
    // to match Phase 2.1 policy (no silent collapse to symmetric).
    table.register_single_output_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "FusedConv2dReLU (CUDA non-cuDNN): backend kernel only supports symmetric "
                "stride/padding/dilation; got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], padding[0], dilation[0], groups, get_cuda_stream(attrs));
        return cuda::relu_kernel(result, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "FusedConv2dSigmoid (CUDA non-cuDNN): backend kernel only supports symmetric "
                "stride/padding/dilation; got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], padding[0], dilation[0], groups, get_cuda_stream(attrs));
        return cuda::sigmoid_kernel(result, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dTanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "FusedConv2dTanh (CUDA non-cuDNN): backend kernel only supports symmetric "
                "stride/padding/dilation; got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], padding[0], dilation[0], groups, get_cuda_stream(attrs));
        return cuda::tanh_kernel(result, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::FusedConv2dSwish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                "FusedConv2dSwish (CUDA non-cuDNN): backend kernel only supports symmetric "
                "stride/padding/dilation; got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        Tensor result = cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], padding[0], dilation[0], groups, get_cuda_stream(attrs));
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

    // Audit D3: device-resident bilinear backward (atomicAdd scatter) —
    // no CPU fallback. Inputs: [grad_output]. Attrs: InputShape, Mode, AlignCorners.
    table.register_single_output_kernel(OpId::InterpolateBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_size = attrs.get_int_list(AttrKey::InputShape);
        std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
        bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
        return cuda::interpolate_backward_cuda(inputs[0], input_size, mode, align_corners);
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

    // audit Q.4: grid_sample / affine_grid backward.
    table.register_kernel(OpId::GridSampleBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string mode = std::string(attrs.get_string(AttrKey::Mode, "bilinear"));
            std::string padding_mode = std::string(attrs.get_string(AttrKey::PaddingMode, "zeros"));
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            auto [gi, gg] = cuda::grid_sample_backward_cuda(
                inputs[2], inputs[0], inputs[1], mode, padding_mode, align_corners);
            return {gi, gg};
        });
    table.register_single_output_kernel(OpId::AffineGridBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            auto size = attrs.get_int_list(AttrKey::OutputSize);
            bool align_corners = attrs.get_bool(AttrKey::AlignCorners, false);
            return cuda::affine_grid_backward_cuda(inputs[0], size, align_corners);
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
        // F.11: per-axis stride/padding/dilation via attr_macros helpers.
        // cuDNN supports asymmetric values natively via cudnnSetConvolution2dDescriptor.
        TENZOR_READ_CONV2D_ATTRS();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv2d_forward(inputs[0], inputs[1], bias,
                                           stride[0], stride[1], padding[0], padding[1],
                                           dilation[0], dilation[1], groups,
                                           get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        TENZOR_READ_CONV2D_ATTRS();
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        TENZOR_READ_CONV2D_ATTRS();
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2],
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, false, true, false, get_cuda_stream(attrs));
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
    // F.11: per-axis read with scalar fallback; non-cuDNN scalar kernel
    // requires symmetric stride/padding/dilation — fail loudly rather than
    // silently collapse asymmetric input.
    auto conv2d_iso_or_throw = [](const OpAttributes& attrs, const char* op_name) {
        TENZOR_READ_CONV2D_ATTRS();
        if (stride[0] != stride[1] || padding[0] != padding[1] || dilation[0] != dilation[1]) {
            throw std::invalid_argument(
                std::string(op_name) + " (CUDA non-cuDNN): backend kernel only supports symmetric "
                "stride/padding/dilation; got stride=" + std::to_string(stride[0]) + "x" + std::to_string(stride[1]) +
                ", padding=" + std::to_string(padding[0]) + "x" + std::to_string(padding[1]) +
                ", dilation=" + std::to_string(dilation[0]) + "x" + std::to_string(dilation[1]) +
                ". Build with cuDNN for asymmetric support.");
        }
        return std::make_tuple(stride[0], padding[0], dilation[0], groups);
    };
    table.register_single_output_kernel(OpId::Conv2dForward, [conv2d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto [s, p, d, g] = conv2d_iso_or_throw(attrs, "Conv2dForward");
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, s, p, d, g, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv2dBackwardInput, [conv2d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [s, p, d, g] = conv2d_iso_or_throw(attrs, "Conv2dBackwardInput");
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], s, p, d, g, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv2dBackwardWeight, [conv2d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [s, p, d, g] = conv2d_iso_or_throw(attrs, "Conv2dBackwardWeight");
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], s, p, d, g, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv2dBackwardBias, [conv2d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto [s, p, d, g] = conv2d_iso_or_throw(attrs, "Conv2dBackwardBias");
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], s, p, d, g, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });
#endif

    // Conv1d: wraps Conv2d by unsqueezing height dimension [N,C,L] -> [N,C,1,L].
    // Audit U.4: scalar Stride/Padding/Dilation from the 1D caller must be
    // projected onto the W axis only; the synthetic H axis is pinned to
    // neutral (stride=1, padding=0, dilation=1) before forwarding to
    // Conv2d. Naive forwarding lets Conv2d's scalar reader apply the same
    // value to both H and W, producing H_out=3 (padding=1) or rejecting
    // the kernel (dilation=2). conv1d_to_conv2d_attrs preserves Groups,
    // Stream, WeightShape, InputShape, etc.
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
        // U.4: project to per-axis even though Conv2dBackwardBias does not
        // currently consume stride/padding/dilation — keeps the contract
        // honest if a future Conv2dBackwardBias implementation reads them.
        const auto conv2d_attrs = ::tenzor::backend::attrs::conv1d_to_conv2d_attrs(attrs);
        auto result = tenzor::dispatch(OpId::Conv2dBackwardBias, conv2d_inputs, conv2d_attrs);
        return {result[0]};
    });

    // Audit I5-followup: per-axis honest contract. Read per-axis attrs with
    // scalar fallback; delegate to scalar wrapper when isotropic; throw clear
    // error on asymmetric (the cuDNN wrapper is scalar-only; native per-axis
    // is a deeper followup paralleling E1's cuDNN work for Conv2d).
    table.register_single_output_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // F.11: per-axis stride/padding/output_padding/dilation via attr_macros.
        // M12: native (non-cuDNN) device kernels accept per-axis values.
        TENZOR_READ_CONVT2D_ATTRS();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv_transpose2d_forward_kernel(
            inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1],
            output_padding[0], output_padding[1], dilation[0], dilation[1],
            groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        // Per-axis depthwise: underlying kernel impl natively accepts per-axis values.
        return cuda::depthwise_conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            get_cuda_stream(attrs));
    });

    // Real native depthwise 1D/3D kernels (forward; backward autograd-composed).
    table.register_kernel(OpId::DepthwiseConv1d,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t s = attrs.get_int(AttrKey::Stride, 1);
            int64_t p = attrs.get_int(AttrKey::Padding, 0);
            int64_t d = attrs.get_int(AttrKey::Dilation, 1);
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return {cuda::depthwise_conv1d_kernel(inputs[0], inputs[1], bias, s, p, d,
                                                  get_cuda_stream(attrs))};
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
            return {cuda::depthwise_conv3d_kernel(inputs[0], inputs[1], bias,
                                                  sD, sH, sW, pD, pH, pW, dD, dH, dW,
                                                  get_cuda_stream(attrs))};
        });

    // =========================================================================
    // Deformable Conv2d (DCNv2) Operations
    // =========================================================================

    // DeformableConv2dForward: inputs = {input, offset, weight, bias, mask}
    // F.11: per-axis via attr_macros (with scalar fallback).
    table.register_kernel(OpId::DeformableConv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        // bias (inputs[3]) and mask (inputs[4]) are optional; std::span has
        // no bounds checking and indexing past size() crashed (no-mask form).
        Tensor empty_t = Tensor({0}, inputs[0].dtype(), inputs[0].device());
        const Tensor& dcf_bias = inputs.size() > 3 ? inputs[3] : empty_t;
        const Tensor& dcf_mask = inputs.size() > 4 ? inputs[4] : empty_t;
        return std::vector<Tensor>{cuda::deformable_conv2d_forward_kernel(
            inputs[0], inputs[1], inputs[2], dcf_bias, dcf_mask,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, offset_groups, get_cuda_stream(attrs))};
    });

    // DeformableConv2dBackwardInput: inputs = {grad_output, input, offset, weight, mask}
    table.register_kernel(OpId::DeformableConv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        // mask (inputs[4]) is optional (no-mask form passes 4 inputs).
        Tensor empty_t = Tensor({0}, inputs[1].dtype(), inputs[1].device());
        const Tensor& dcb_mask = inputs.size() > 4 ? inputs[4] : empty_t;
        return cuda::deformable_conv2d_backward_input_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], dcb_mask,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
            groups, offset_groups, get_cuda_stream(attrs));
    });

    // DeformableConv2dBackwardWeight: inputs = {grad_output, input, offset, mask}
    table.register_kernel(OpId::DeformableConv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        int64_t offset_groups = attrs.get_int(AttrKey::OffsetGroups, 1);
        auto weight_shape = attrs.get_int_list(AttrKey::WeightShape);
        // mask (inputs[3]) is optional; std::span has no bounds checking and
        // indexing past size() crashed on the no-mask call form.
        Tensor empty_t = Tensor({0}, inputs[1].dtype(), inputs[1].device());
        const Tensor& mask = inputs.size() > 3 ? inputs[3] : empty_t;
        return std::vector<Tensor>{cuda::deformable_conv2d_backward_weight_kernel(
            inputs[0], inputs[1], inputs[2], mask,
            stride[0], stride[1], padding[0], padding[1], dilation[0], dilation[1],
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
    // Wave B4 + F.11: CUDA Conv3d honours per-axis StrideD/H/W, PaddingD/H/W,
    // DilationD/H/W natively via cudnnSetConvolutionNdDescriptor (which has
    // always accepted per-axis int[3] arrays). Reads per-axis attrs (with
    // scalar fallback) via the shared attr_macros helpers.
    table.register_single_output_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        TENZOR_READ_CONV3D_ATTRS();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv3d_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        TENZOR_READ_CONV3D_ATTRS();
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv3d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        TENZOR_READ_CONV3D_ATTRS();
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
    // Wave B4 + F.11: ConvT3d honours per-axis StrideD/H/W, PaddingD/H/W,
    // OutputPaddingD/H/W, DilationD/H/W via cuDNN Nd descriptors. Reads
    // per-axis attrs (with scalar fallback) via shared attr_macros helpers.
    table.register_single_output_kernel(OpId::ConvTranspose3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        TENZOR_READ_CONVT3D_ATTRS();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv_transpose3d_forward(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        TENZOR_READ_CONVT3D_ATTRS();
        return cuda::cudnn_conv_transpose3d_backward_input(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        TENZOR_READ_CONVT3D_ATTRS();
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
    // Wave B4 + F.11 + S.6: non-cuDNN Conv3d fallback. The kernel functions
    // accept std::array signatures and read per-axis attrs via the shared
    // attr_macros helpers, but the underlying device kernel uses scalar index
    // math — it throws std::runtime_error on asymmetric stride/padding/
    // dilation rather than silently collapsing to symmetric (see
    // conv3d.cu:780-786). Build with TENZOR_HAS_CUDNN=ON for true per-axis
    // support, or pass isotropic params; the previous registry comment
    // ("iso-assert") was misleading because the throw is in the kernel, not
    // an assert here.
    table.register_single_output_kernel(OpId::Conv3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        TENZOR_READ_CONV3D_ATTRS();
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv3d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv3dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        TENZOR_READ_CONV3D_ATTRS();
        auto [grad_input, grad_weight, grad_bias] = cuda::conv3d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv3dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        TENZOR_READ_CONV3D_ATTRS();
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

    // Wave B4 + F.11: non-cuDNN ConvT3d fallback — these device kernels still
    // take scalar stride/padding/output_padding/dilation, so we assert iso here.
    // (Native per-axis ConvT3d device kernels are a separate refactor; the
    // cuDNN path already supports asymmetric.)
    auto t3d_iso_or_throw = [](const OpAttributes& attrs, const char* op_name)
        -> std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t> {
        const auto stride         = ::tenzor::backend::attrs::stride_3d(attrs);
        const auto padding        = ::tenzor::backend::attrs::padding_3d(attrs);
        const auto output_padding = ::tenzor::backend::attrs::output_padding_3d(attrs);
        const auto dilation       = ::tenzor::backend::attrs::dilation_3d(attrs);
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        if (stride[0] != stride[1] || stride[1] != stride[2] ||
            padding[0] != padding[1] || padding[1] != padding[2] ||
            output_padding[0] != output_padding[1] || output_padding[1] != output_padding[2] ||
            dilation[0] != dilation[1] || dilation[1] != dilation[2]) {
            throw std::runtime_error(
                std::string("CUDA ") + op_name + ": asymmetric stride/padding/"
                "output_padding/dilation requires cuDNN; rebuild with TENZOR_HAS_CUDNN.");
        }
        return {stride[0], padding[0], output_padding[0], dilation[0], groups};
    };
    table.register_single_output_kernel(OpId::ConvTranspose3dForward, [t3d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto [stride, padding, output_padding, dilation, groups] = t3d_iso_or_throw(attrs, "ConvTranspose3d (non-cuDNN fallback)");
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv_transpose3d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardInput, [t3d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto [stride, padding, output_padding, dilation, groups] = t3d_iso_or_throw(attrs, "ConvTranspose3dBackwardInput (non-cuDNN fallback)");
        return cuda::conv_transpose3d_backward_input_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ConvTranspose3dBackwardWeight, [t3d_iso_or_throw](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto [stride, padding, output_padding, dilation, groups] = t3d_iso_or_throw(attrs, "ConvTranspose3dBackwardWeight (non-cuDNN fallback)");
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
    // S.7: Dropout is a scalar-probability elementwise op. There is no spatial
    // dimensionality to make per-axis, so the float `p` and bool `training`
    // attrs are the only inputs the kernel takes — this scalar API is
    // structurally correct, not a per-axis collapse.
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
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_cuda_stream(attrs);
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
            Tensor gates = cuda::add_kernel(
                cuda::matmul_kernel(input, cuda::transpose_kernel(w_ih, 0, 1, stream), stream),
                cuda::matmul_kernel(hx, cuda::transpose_kernel(w_hh, 0, 1, stream), stream), stream);
            if (inputs.size() > 5 && inputs[5].numel() > 0)
                gates = cuda::add_kernel(gates, inputs[5], stream);
            if (inputs.size() > 6 && inputs[6].numel() > 0)
                gates = cuda::add_kernel(gates, inputs[6], stream);
            auto [h_out, c_out] = cuda::lstm_cell_forward_kernel(
                gates, cx, batch_size, hidden_size, stream);
            return {h_out, c_out};
        }
        // Legacy 2-input fused form ({gates, c_prev} + size attrs), used by
        // the internal full-sequence path.
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [h_out, c_out] = cuda::lstm_cell_forward_kernel(inputs[0], inputs[1], batch_size, hidden_size, stream);
        return {h_out, c_out};
    });
    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_cuda_stream(attrs);
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
            Tensor gates = cuda::add_kernel(
                cuda::matmul_kernel(input, cuda::transpose_kernel(w_ih, 0, 1, stream), stream),
                cuda::matmul_kernel(hx, cuda::transpose_kernel(w_hh, 0, 1, stream), stream), stream);
            if (inputs[9].numel() > 0)  gates = cuda::add_kernel(gates, inputs[9], stream);
            if (inputs[10].numel() > 0) gates = cuda::add_kernel(gates, inputs[10], stream);
            // Cell backward -> grad wrt gate pre-activations and wrt c_prev.
            auto [d_gates, grad_cx] = cuda::lstm_cell_backward_kernel(
                d_hy, d_cy, gates, cx, cy, batch_size, hidden_size, stream);
            // Linear backward through gates = input@w_ih^T + hx@w_hh^T + b.
            Tensor grad_input = cuda::matmul_kernel(d_gates, w_ih, stream);      // [B,in]
            Tensor grad_hx    = cuda::matmul_kernel(d_gates, w_hh, stream);      // [B,H]
            Tensor d_gates_T  = cuda::transpose_kernel(d_gates, 0, 1, stream);   // [4H,B]
            Tensor grad_w_ih  = cuda::matmul_kernel(d_gates_T, input, stream);   // [4H,in]
            Tensor grad_w_hh  = cuda::matmul_kernel(d_gates_T, hx, stream);      // [4H,H]
            // Column-sum over the batch for the bias grads: ones[1,B] @ d_gates.
            Tensor ones = cuda::ones_kernel({1, batch_size}, d_gates.dtype(), d_gates.device(), stream);
            Tensor grad_b = cuda::matmul_kernel(ones, d_gates, stream).reshape({4 * hidden_size});
            return {grad_input, grad_hx, grad_cx, grad_w_ih, grad_w_hh, grad_b, grad_b};
        }
        // Legacy fused 5-input form ({grad_h, grad_c, gates, c_prev, c_out}).
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto [grad_gates, grad_c_prev] = cuda::lstm_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, stream);
        return std::vector<Tensor>{grad_gates, grad_c_prev};
    });

    // =========================================================================
    // GRU Operations (Phase 1C - HIGH)
    // =========================================================================
    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_cuda_stream(attrs);
        // Canonical 6-input contract (matches the CPU backend and the
        // OpId-dispatch callers/tests): {input, hx, w_ih, w_hh, b_ih, b_hh}.
        // Compose the GRU gate pre-activations here, then run the cell kernel.
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
            Tensor gates_ih = cuda::matmul_kernel(
                input, cuda::transpose_kernel(w_ih, 0, 1, stream), stream);
            if (b_ih.numel() > 0) gates_ih = cuda::add_kernel(gates_ih, b_ih, stream);
            Tensor gates_hh = cuda::matmul_kernel(
                hx, cuda::transpose_kernel(w_hh, 0, 1, stream), stream);
            if (b_hh.numel() > 0) gates_hh = cuda::add_kernel(gates_hh, b_hh, stream);
            // Split each [batch, 3H] into reset/update/new chunks of width H.
            auto ih = cuda::split_kernel(gates_ih, hidden_size, /*dim=*/1, stream);
            auto hh = cuda::split_kernel(gates_hh, hidden_size, /*dim=*/1, stream);
            Tensor reset_gates  = cuda::add_kernel(ih[0], hh[0], stream);
            Tensor update_gates = cuda::add_kernel(ih[1], hh[1], stream);
            Tensor new_input    = ih[2];
            Tensor new_hidden   = hh[2];
            return {cuda::gru_cell_forward_kernel(
                reset_gates, update_gates, new_input, new_hidden,
                hx, batch_size, hidden_size, stream)};
        }
        // Legacy 5-input fused form, used by the internal full-sequence path.
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        return {cuda::gru_cell_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, stream)};
    });
    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        auto stream = get_cuda_stream(attrs);
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
            Tensor gates_ih = cuda::matmul_kernel(input, cuda::transpose_kernel(w_ih, 0, 1, stream), stream);
            if (b_ih.numel() > 0) gates_ih = cuda::add_kernel(gates_ih, b_ih, stream);
            Tensor gates_hh = cuda::matmul_kernel(hx, cuda::transpose_kernel(w_hh, 0, 1, stream), stream);
            if (b_hh.numel() > 0) gates_hh = cuda::add_kernel(gates_hh, b_hh, stream);
            auto ih = cuda::split_kernel(gates_ih, hidden_size, /*dim=*/1, stream); // [r_i,z_i,n_i]
            auto hh = cuda::split_kernel(gates_hh, hidden_size, /*dim=*/1, stream); // [r_h,z_h,n_h]
            Tensor reset_gates  = cuda::add_kernel(ih[0], hh[0], stream);
            Tensor update_gates = cuda::add_kernel(ih[1], hh[1], stream);
            Tensor new_input    = ih[2];
            Tensor new_hidden   = hh[2];
            auto o = cuda::gru_cell_backward_kernel(
                d_hy, reset_gates, update_gates, new_input, new_hidden, hx,
                batch_size, hidden_size, stream);
            // reset/update gates receive grad on both ih and hh sides;
            // new gate's ih chunk gets grad_new_input, hh chunk grad_new_hidden.
            std::array<Tensor, 3> ih_parts{o.grad_reset, o.grad_update, o.grad_new_input};
            std::array<Tensor, 3> hh_parts{o.grad_reset, o.grad_update, o.grad_new_hidden};
            Tensor d_gates_ih = cuda::cat_kernel(ih_parts, /*dim=*/1, stream);
            Tensor d_gates_hh = cuda::cat_kernel(hh_parts, /*dim=*/1, stream);
            Tensor grad_input = cuda::matmul_kernel(d_gates_ih, w_ih, stream);
            Tensor grad_hx = cuda::add_kernel(cuda::matmul_kernel(d_gates_hh, w_hh, stream),
                                              o.grad_h_prev, stream);
            Tensor grad_w_ih = cuda::matmul_kernel(cuda::transpose_kernel(d_gates_ih, 0, 1, stream), input, stream);
            Tensor grad_w_hh = cuda::matmul_kernel(cuda::transpose_kernel(d_gates_hh, 0, 1, stream), hx, stream);
            Tensor ones = cuda::ones_kernel({1, batch_size}, d_gates_ih.dtype(), d_gates_ih.device(), stream);
            Tensor grad_b_ih = cuda::matmul_kernel(ones, d_gates_ih, stream).reshape({3 * hidden_size});
            Tensor grad_b_hh = cuda::matmul_kernel(ones, d_gates_hh, stream).reshape({3 * hidden_size});
            return {grad_input, grad_hx, grad_w_ih, grad_w_hh, grad_b_ih, grad_b_hh};
        }
        // Legacy fused 6-input form.
        int64_t batch_size = attrs.get_int(AttrKey::BatchSize, 0);
        int64_t hidden_size = attrs.get_int(AttrKey::HiddenSize, 0);
        auto result = cuda::gru_cell_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
            batch_size, hidden_size, stream);
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
    // F.11: per-axis (W-only) read via attr_macros helpers. AvgPool/MaxPool 1d
    // only have one spatial axis, but the helpers preserve the scalar-fallback
    // semantics and keep the registry uniform with 2d/3d sites.
    // =========================================================================
    table.register_kernel(OpId::MaxPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // Q.5: per-axis std::array<int64_t, 1> passes the W-axis directly to the
        // kernel (no scalar collapse).
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_1d(attrs,
            AttrKey::Stride, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs);
        const auto dilation    = ::tenzor::backend::attrs::dilation_1d(attrs);
        auto [output, indices] = cuda::maxpool1d_forward_kernel(inputs[0],
            kernel_size, stride, padding, dilation, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::maxpool1d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.5: per-axis std::array<int64_t, 1> signature.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_1d(attrs,
            AttrKey::Stride, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs);
        return cuda::avgpool1d_forward_kernel(inputs[0],
            kernel_size, stride, padding,
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.5: per-axis std::array<int64_t, 1> signature.
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_1d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_1d(attrs,
            AttrKey::Stride, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_1d(attrs);
        return cuda::avgpool1d_backward_kernel(inputs[0], input_shape,
            kernel_size, stride, padding,
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0, get_cuda_stream(attrs));
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
        // Q.5: per-axis std::array<int64_t, 3> kernel signature accepts
        // asymmetric kernel/stride/padding across D/H/W (was previously
        // collapsed to scalars and rejected at the dispatch).
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        auto [output, indices] = cuda::maxpool3d_forward_kernel(inputs[0],
            kernel_size, stride, padding, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_single_output_kernel(OpId::MaxPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::maxpool3d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.5: per-axis std::array<int64_t, 3> kernel signature.
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        return cuda::avgpool3d_forward_kernel(inputs[0],
            kernel_size, stride, padding,
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::AvgPool3dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // Q.5: per-axis std::array<int64_t, 3> kernel signature.
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        const auto kernel_size = ::tenzor::backend::attrs::kernel_size_3d(attrs);
        const auto stride      = ::tenzor::backend::attrs::read_3d(attrs,
            AttrKey::Stride, AttrKey::StrideD, AttrKey::StrideH, AttrKey::StrideW, kernel_size[0]);
        const auto padding     = ::tenzor::backend::attrs::padding_3d(attrs);
        return cuda::avgpool3d_backward_kernel(inputs[0], input_shape,
            kernel_size, stride, padding,
            attrs.get_int(AttrKey::CountIncludePad, 1) != 0, get_cuda_stream(attrs));
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
        CUDA_CHECK(cudaMemsetAsync(output.data_ptr(), 0, output.numel() * dtype_size(dtype), get_cuda_stream(attrs)));
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
        // Read as double (the attribute is stored as double) so Float64
        // subnormals survive the trip through OpAttributes.
        double value = attrs.get_float(AttrKey::Value, 0.0);
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

    // inputs: [input, W_ih, W_hh, bias_ih, h0, bias_hh?]
    // bias_hh is optional (legacy 5-input callers still work) but required
    // for PyTorch-faithful GRU math (Phase 8.5).
    table.register_kernel(OpId::GRUForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        Tensor bias_hh = (inputs.size() > 5) ? inputs[5] : Tensor{};
        return cuda::gru_forward_cuda(inputs[0], inputs[1], inputs[2],
                                      inputs[3], inputs[4], bias_hh);
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
        // LL.3: per-axis Unfold accepts asymmetric kernel/stride/padding/dilation.
        const auto kernel_size = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 3);
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        cudaStream_t stream = get_cuda_stream(attrs);
        return cuda::unfold_cuda(inputs[0],
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            stream);
    });

    // inputs: [input]
    // attrs: output_size, kernel_size, stride, padding, dilation
    table.register_single_output_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // LL.3: per-axis Fold accepts asymmetric kernel/stride/padding/dilation.
        auto output_size = attrs.get_int_list(AttrKey::OutputSize);
        const auto kernel_size = ::tenzor::backend::attrs::read_2d(attrs,
            AttrKey::KernelSize, AttrKey::KernelSizeH, AttrKey::KernelSizeW, 3);
        const auto stride   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation = ::tenzor::backend::attrs::dilation_2d(attrs);
        cudaStream_t stream = get_cuda_stream(attrs);
        return cuda::fold_cuda(inputs[0], output_size,
            kernel_size[0], kernel_size[1],
            stride[0], stride[1],
            padding[0], padding[1],
            dilation[0], dilation[1],
            stream);
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
        int64_t kh_size = weight_shape[2];
        int64_t kw_size = weight_shape[3];

        // Per-axis (asymmetric) stride/padding/dilation, supported natively.
        const auto stride_arr   = ::tenzor::backend::attrs::stride_2d(attrs);
        const auto padding_arr  = ::tenzor::backend::attrs::padding_2d(attrs);
        const auto dilation_arr = ::tenzor::backend::attrs::dilation_2d(attrs);
        int64_t sH = stride_arr[0], sW = stride_arr[1];
        int64_t pH = padding_arr[0], pW = padding_arr[1];
        int64_t dH = dilation_arr[0], dW = dilation_arr[1];
        int64_t groups = attrs.get_int(AttrKey::Groups, 1);
        (void)groups;  // grouping not modeled by this kernel (pre-existing)

        float input_scale = static_cast<float>(attrs.get_float(AttrKey::InputScale, 1.0));
        float weight_scale = static_cast<float>(attrs.get_float(AttrKey::WeightScaleQ, 1.0));
        int32_t input_zp = static_cast<int32_t>(attrs.get_int(AttrKey::InputZeroPoint, 0));
        int32_t weight_zp = static_cast<int32_t>(attrs.get_int(AttrKey::WeightZeroPoint, 0));
        auto stream = get_cuda_stream(attrs);

        int64_t h_out = (h_in + 2 * pH - dH * (kh_size - 1) - 1) / sH + 1;
        int64_t w_out = (w_in + 2 * pW - dW * (kw_size - 1) - 1) / sW + 1;

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
            kh_size, kw_size, sH, sW, pH, pW, dH, dW,
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

    table.register_single_output_kernel(OpId::GammaSample,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
            return cuda::gamma_sample_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
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
        // RR.8: honour per-axis OutputRatio{H,W} when set (PyTorch ratio mode).
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
    // S.7: FractionalMaxPool3d / MaxUnpool*d below already take per-axis
    // OutputSizeD/H/W and the underlying CUDA kernels accept (out_d, out_h,
    // out_w) as separate parameters. No stride/padding attrs apply — fractional
    // pool computes stride from input/output ratio internally, and unpool uses
    // the saved index map. No per-axis collapse here.
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
    // Phase A.1: Max Unpool 1D (CUDA — wraps the 2D kernel via reshape).
    // =========================================================================
    table.register_single_output_kernel(OpId::MaxUnpool1dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t out_l = attrs.get_int(AttrKey::OutputSizeW, 1);
        return cuda::max_unpool1d_forward_kernel(inputs[0], inputs[1], out_l, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::MaxUnpool1dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = attrs.get_int_list(AttrKey::InputShape);
        return cuda::max_unpool1d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
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
    table.register_kernel(OpId::Frexp, cuda::frexp_dispatch);  // F1: multi-output (mantissa, exponent)
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

    // F4 sub-followup: widen-narrow for F16/BF16 on the CUDA nested kernels.
    // The kernel impls support Float32 + Float64 natively (F4); F16/BF16
    // widen at the registry entry to Float32, dispatch, and narrow back.
    // (Inline check rather than a captured helper because
    // `SingleOutputKernelFn` is a raw function pointer, not a `std::function`,
    // and capturing lambdas can't decay to function pointers — same constraint
    // hit by the I5-followup CUDA Conv3d registry macros.)
    table.register_single_output_kernel(OpId::NestedSum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        DType d = inputs[0].dtype();
        if (d == DType::Float16 || d == DType::BFloat16) {
            auto v = inputs[0].to(DType::Float32);
            auto r = cuda::nested_sum_cuda(v, inputs[1], get_cuda_stream(attrs));
            return r.to(d);
        }
        return cuda::nested_sum_cuda(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedMean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        DType d = inputs[0].dtype();
        if (d == DType::Float16 || d == DType::BFloat16) {
            auto v = inputs[0].to(DType::Float32);
            auto r = cuda::nested_mean_cuda(v, inputs[1], get_cuda_stream(attrs));
            return r.to(d);
        }
        return cuda::nested_mean_cuda(inputs[0], inputs[1], get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float eps = attrs.get_float(AttrKey::Eps, 1e-5f);
        DType d = inputs[0].dtype();
        if (d == DType::Float16 || d == DType::BFloat16) {
            auto v = inputs[0].to(DType::Float32);
            auto w = inputs[2].to(DType::Float32);
            auto b = inputs[3].to(DType::Float32);
            auto r = cuda::nested_layer_norm_cuda(v, inputs[1], w, b, eps, get_cuda_stream(attrs));
            return r.to(d);
        }
        return cuda::nested_layer_norm_cuda(inputs[0], inputs[1], inputs[2], inputs[3], eps, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedLinear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // NestedLinear delegates to `cuda::nested_linear_cuda` which itself
        // routes through MatMul + Add dispatch — those already handle every
        // floating dtype, so no widen-narrow needed here.
        const Tensor* bias = (inputs.size() > 3) ? &inputs[3] : nullptr;
        return cuda::nested_linear_cuda(inputs[0], inputs[2], bias, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedAttention, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float scale = attrs.get_float(AttrKey::Scale, 1.0f);
        bool causal = attrs.get_bool(AttrKey::Causal, false);
        DType d = inputs[0].dtype();
        if (d == DType::Float16 || d == DType::BFloat16) {
            auto Q = inputs[0].to(DType::Float32);
            auto K = inputs[1].to(DType::Float32);
            auto V = inputs[2].to(DType::Float32);
            auto r = cuda::nested_attention_cuda(Q, K, V, inputs[3], inputs[4],
                                                  scale, causal, get_cuda_stream(attrs));
            return r.to(d);
        }
        return cuda::nested_attention_cuda(inputs[0], inputs[1], inputs[2],
                                           inputs[3], inputs[4], scale, causal, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedToPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t max_len = attrs.get_int(AttrKey::MaxLen, 0);
        float padding_value = attrs.get_float(AttrKey::PaddingValue, 0.0f);
        DType d = inputs[0].dtype();
        if (d == DType::Float16 || d == DType::BFloat16) {
            auto v = inputs[0].to(DType::Float32);
            auto r = cuda::nested_to_padded_cuda(v, inputs[1], max_len, padding_value, get_cuda_stream(attrs));
            return r.to(d);
        }
        return cuda::nested_to_padded_cuda(inputs[0], inputs[1], max_len, padding_value, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::NestedFromPadded, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        DType d = inputs[0].dtype();
        if (d == DType::Float16 || d == DType::BFloat16) {
            auto v = inputs[0].to(DType::Float32);
            auto r = cuda::nested_from_padded_cuda(v, inputs[1], get_cuda_stream(attrs));
            return r.to(d);
        }
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
            // F4 sub-followup: widen-narrow for F16/BF16. inputs are
            // [grad_out, Q, K, V, attn_out, q_offsets, kv_offsets].
            if (inputs[0].dtype() == DType::Float16 || inputs[0].dtype() == DType::BFloat16) {
                DType orig = inputs[0].dtype();
                auto go = inputs[0].to(DType::Float32);
                auto q  = inputs[1].to(DType::Float32);
                auto k  = inputs[2].to(DType::Float32);
                auto v  = inputs[3].to(DType::Float32);
                auto ao = inputs[4].to(DType::Float32);
                auto res = cuda::nested_attention_backward_cuda(
                    go, q, k, v, ao, inputs[5], inputs[6],
                    scale, causal, get_cuda_stream(attrs));
                std::vector<Tensor> narrowed;
                narrowed.reserve(res.size());
                for (auto& t : res) narrowed.push_back(t.to(orig));
                return narrowed;
            }
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
