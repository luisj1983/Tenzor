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
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/backend/fused_ops.hpp"
#ifdef TENZOR_HAS_CUDNN
#include "tenzor/backend/cudnn_wrapper.hpp"
#endif
#include <cuda_runtime.h>
#include <cstdlib>
#include <charconv>
#include <limits>
#include <tuple>

namespace tenzor {

// Helper to extract CUDA stream from attributes
inline cudaStream_t get_cuda_stream(const OpAttributes& attrs) {
    if (attrs.contains("stream")) {
        return static_cast<cudaStream_t>(
            reinterpret_cast<void*>(std::stoull(attrs.at("stream")))
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
    auto masked_select_kernel(const Tensor& input, const Tensor& mask, cudaStream_t stream) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, double value, cudaStream_t stream) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, cudaStream_t stream) -> Tensor;
    auto nonzero_kernel(const Tensor& input, cudaStream_t stream) -> Tensor;
    auto one_hot_kernel(const Tensor& indices, int64_t num_classes, cudaStream_t stream) -> Tensor;

    // Embedding operations
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, cudaStream_t stream) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices, int64_t num_embeddings, cudaStream_t stream) -> Tensor;

    // Fused operations
    auto fused_linear_relu_cuda(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_batchnorm_relu_cuda(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_add_relu_cuda(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_cuda(const Tensor& input) -> Tensor;

    // Vision/Interpolation operations
    auto interpolate_cuda(const Tensor& input, const std::vector<int64_t>& size, const std::string& mode, bool align_corners) -> Tensor;

    // ROI Align operations
    auto roi_align_forward(const Tensor& features, const Tensor& rois,
                           int64_t output_h, int64_t output_w,
                           float spatial_scale, int64_t sampling_ratio,
                           bool aligned) -> Tensor;
    auto roi_align_backward(const Tensor& grad_output, const Tensor& rois,
                            int64_t batch_size, int64_t feat_height, int64_t feat_width,
                            float spatial_scale, int64_t sampling_ratio,
                            bool aligned) -> Tensor;

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
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        int64_t step,
        bool decoupled_weight_decay,
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

} // namespace cuda

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

    // Inplace activation operations
    table.register_kernel(OpId::ReLUInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor input = inputs[0];  // Copy to allow modification
        cuda::relu_inplace_kernel(input, get_cuda_stream(attrs));
        return std::vector<Tensor>{input};
    });

    table.register_kernel(OpId::SigmoidInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor input = inputs[0];
        cuda::sigmoid_inplace_kernel(input, get_cuda_stream(attrs));
        return std::vector<Tensor>{input};
    });

    table.register_kernel(OpId::TanhInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor input = inputs[0];
        cuda::tanh_inplace_kernel(input, get_cuda_stream(attrs));
        return std::vector<Tensor>{input};
    });

    table.register_kernel(OpId::LeakyReLUInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        Tensor input = inputs[0];
        cuda::leaky_relu_inplace_kernel(input, alpha, get_cuda_stream(attrs));
        return std::vector<Tensor>{input};
    });

    table.register_kernel(OpId::GeluInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor input = inputs[0];
        cuda::gelu_inplace_kernel(input, get_cuda_stream(attrs));
        return std::vector<Tensor>{input};
    });

    // =========================================================================
    // Reduction Operations
    // =========================================================================
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::sum_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::mean_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::max_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::min_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::argmax_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::argmin_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::prod_kernel(inputs[0], dim, keepdim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        int64_t correction = parse_attr<int64_t>(attrs, "correction", 1);
        return std::vector<Tensor>{cuda::var_kernel(inputs[0], dim, keepdim, correction, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        int64_t correction = parse_attr<int64_t>(attrs, "correction", 1);
        return std::vector<Tensor>{cuda::std_kernel(inputs[0], dim, keepdim, correction, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = parse_attr<float>(attrs, "p", 2.0f);
        int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cuda::norm_kernel(inputs[0], p, dim, keepdim, get_cuda_stream(attrs))};
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
    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float exponent = parse_attr<float>(attrs, "exponent", 2.0f);
        return std::vector<Tensor>{cuda::pow_kernel(inputs[0], exponent, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = parse_attr<float>(attrs, "min", -std::numeric_limits<float>::infinity());
        float max_val = parse_attr<float>(attrs, "max", std::numeric_limits<float>::infinity());
        return std::vector<Tensor>{cuda::clamp_kernel(inputs[0], min_val, max_val, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = parse_attr<float>(attrs, "min", 0.0f);
        return std::vector<Tensor>{cuda::clamp_min_kernel(inputs[0], min_val, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float max_val = parse_attr<float>(attrs, "max", 0.0f);
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
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        return cuda::leaky_relu_kernel(inputs[0], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        return cuda::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = parse_attr<float>(attrs, "alpha", 1.0f);
        return cuda::elu_kernel(inputs[0], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float alpha = parse_attr<float>(attrs, "alpha", 1.0f);
        return cuda::elu_backward_kernel(inputs[0], inputs[1], alpha, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = parse_attr<float>(attrs, "beta", 1.0f);
        float threshold = parse_attr<float>(attrs, "threshold", 20.0f);
        return cuda::softplus_kernel(inputs[0], beta, threshold, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float beta = parse_attr<float>(attrs, "beta", 1.0f);
        float threshold = parse_attr<float>(attrs, "threshold", 20.0f);
        return cuda::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, get_cuda_stream(attrs));
    });

    // Softmax operations (use cuDNN when available for better performance)
    // Uses single-output registration for efficiency
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::cudnn_softmax_forward(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::cudnn_softmax_backward(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::cudnn_log_softmax_forward(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::cudnn_log_softmax_backward(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
#else
    table.register_single_output_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::softmax_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::softmax_backward_kernel(inputs[0], inputs[1], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::log_softmax_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
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
        float value = parse_attr<float>(attrs, "value", 0.0f);
        return cuda::fill_kernel(inputs[0], value, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = parse_int_list(attrs, "shape");
        return cuda::reshape_kernel(inputs[0], shape, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto shape = parse_int_list(attrs, "shape");
        return cuda::expand_kernel(inputs[0], shape, static_cast<void*>(get_cuda_stream(attrs)));
    });
    table.register_single_output_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim0 = parse_attr<int64_t>(attrs, "dim0", 0);
        int64_t dim1 = parse_attr<int64_t>(attrs, "dim1", 1);
        return cuda::transpose_kernel(inputs[0], dim0, dim1, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto dims = parse_int_list(attrs, "dims");
        return cuda::permute_kernel(inputs[0], dims, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return cuda::squeeze_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::unsqueeze_kernel(inputs[0], dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::cat_kernel(inputs, dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto repeats = parse_int_list(attrs, "repeats");
        return cuda::repeat_kernel(inputs[0], repeats, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::ToMemoryFormat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int format_int = parse_attr<int>(attrs, "memory_format", 0);
        MemoryFormat format = static_cast<MemoryFormat>(format_int);
        return cuda::to_memory_format_kernel(inputs[0], format, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Indexing Operations (single-output registration for efficiency)
    // =========================================================================
    table.register_single_output_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::index_select_kernel(inputs[0], dim, inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::gather_kernel(inputs[0], dim, inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::masked_select_kernel(inputs[0], inputs[1], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        double value = parse_attr<double>(attrs, "value", 0.0);
        return cuda::masked_fill_kernel(inputs[0], inputs[1], value, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::where_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Nonzero, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        return cuda::nonzero_kernel(inputs[0], get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::OneHot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t num_classes = parse_attr<int64_t>(attrs, "num_classes", 0);
        return cuda::one_hot_kernel(inputs[0], num_classes, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Pooling Operations (use cuDNN when available for better performance)
    // Note: MaxPool2dForward returns 2 tensors (output + indices) so uses register_kernel
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        auto [output, indices] = cuda::cudnn_maxpool2d_forward(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return cuda::cudnn_avgpool2d_forward(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#else
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        auto [output, indices] = cuda::maxpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, dilation, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return cuda::avgpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#endif

    // =========================================================================
    // Pooling Backward Operations
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, input, output]
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return cuda::cudnn_maxpool2d_backward(inputs[0], inputs[1], inputs[2], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, input]
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return cuda::cudnn_avgpool2d_backward(inputs[0], inputs[1], kernel_size, stride, padding, get_cuda_stream(attrs));
    });
#else
    table.register_single_output_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, indices], attrs: input_shape
        auto input_shape = parse_int_list(attrs, "input_shape");
        return cuda::maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output], attrs: input_shape, kernel_size, stride, padding
        auto input_shape = parse_int_list(attrs, "input_shape");
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
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
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        return std::vector<Tensor>{cuda::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        // Use optimized vectorized kernel for inference (faster than cuDNN due to lower overhead)
        // inputs: [input, mean, variance, gamma, beta]
        return std::vector<Tensor>{cuda::batchnorm2d_forward_affine_optimized(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_cuda_stream(attrs)
        )};
    });
    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = parse_attr<float>(attrs, "momentum", 0.1f);
        Tensor running_mean = inputs[0];
        Tensor running_var = inputs[1];
        cuda::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, get_cuda_stream(attrs));
        return std::vector<Tensor>{running_mean, running_var};
    });

#ifdef TENZOR_HAS_CUDNN
    // Fused BatchNorm training: computes mean/var, normalizes, and updates running stats in one call
    table.register_kernel(OpId::BatchNorm2dFusedTraining, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, running_mean, running_var, gamma, beta]
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        float momentum = parse_attr<float>(attrs, "momentum", 0.1f);
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
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        auto [grad_input, grad_gamma, grad_beta] = cuda::cudnn_batchnorm2d_backward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            epsilon, get_cuda_stream(attrs)
        );
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });
#else
    // Custom CUDA kernel backward - fallback when cuDNN is not available
    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, mean, variance, gamma]
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
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
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");

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
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
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
        float scale = parse_attr<float>(attrs, "scale", 1.0f);

#ifdef TENZOR_HAS_CUDNN_FRONTEND
        // Check if cuDNN SDPA is requested and input is 4D
        bool use_cudnn_sdpa = parse_attr<bool>(attrs, "use_cudnn_sdpa", false);
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
        float lr = parse_attr<float>(attrs, "lr", 0.01f);
        float momentum = parse_attr<float>(attrs, "momentum", 0.0f);
        float weight_decay = parse_attr<float>(attrs, "weight_decay", 0.0f);
        float dampening = parse_attr<float>(attrs, "dampening", 0.0f);
        bool nesterov = parse_attr<bool>(attrs, "nesterov", false);

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
        // inputs: [param, grad, exp_avg, exp_avg_sq]
        // attrs: lr, beta1, beta2, eps, weight_decay, step, decoupled
        float lr = parse_attr<float>(attrs, "lr", 0.001f);
        float beta1 = parse_attr<float>(attrs, "beta1", 0.9f);
        float beta2 = parse_attr<float>(attrs, "beta2", 0.999f);
        float eps = parse_attr<float>(attrs, "eps", 1e-8f);
        float weight_decay = parse_attr<float>(attrs, "weight_decay", 0.0f);
        int64_t step = parse_attr<int64_t>(attrs, "step", 1);
        bool decoupled = parse_attr<bool>(attrs, "decoupled", false);

        // Cast away const for in-place modification
        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& exp_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& exp_avg_sq = const_cast<Tensor&>(inputs[3]);

        cuda::fused_adam_step_cuda(
            param, inputs[1], exp_avg, exp_avg_sq,
            lr, beta1, beta2, eps, weight_decay, step, decoupled,
            get_cuda_stream(attrs)
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

    table.register_single_output_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [grad_output, indices]
        // attrs: num_embeddings
        int64_t num_embeddings = parse_attr<int64_t>(attrs, "num_embeddings", 0);
        return cuda::embedding_backward_kernel(inputs[0], inputs[1], num_embeddings, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Fused Operations (optimized combined kernels)
    // =========================================================================
    table.register_single_output_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        // inputs: [input, weight] or [input, weight, bias]
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::fused_linear_relu_cuda(inputs[0], inputs[1], bias);
    });

    table.register_single_output_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input, running_mean, running_var, weight, bias]
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
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

    // =========================================================================
    // Vision/Interpolation Operations
    // =========================================================================
    table.register_single_output_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        // inputs: [input]
        // attrs: size (comma-separated), mode, align_corners
        auto size = parse_int_list(attrs, "size");
        std::string mode = parse_attr<std::string>(attrs, "mode", "bilinear");
        bool align_corners = parse_attr<bool>(attrs, "align_corners", false);
        return cuda::interpolate_cuda(inputs[0], size, mode, align_corners);
    });

    // =========================================================================
    // ROI Align Operations
    // =========================================================================
    table.register_kernel(OpId::ROIAlignForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // inputs: [features, rois]
        // attrs: output_h, output_w, spatial_scale, sampling_ratio, aligned
        int64_t output_h = parse_attr<int64_t>(attrs, "output_h", 7);
        int64_t output_w = parse_attr<int64_t>(attrs, "output_w", 7);
        float spatial_scale = parse_attr<float>(attrs, "spatial_scale", 1.0f / 16.0f);
        int64_t sampling_ratio = parse_attr<int64_t>(attrs, "sampling_ratio", 0);
        bool aligned = parse_attr<bool>(attrs, "aligned", true);

        return {cuda::roi_align_forward(inputs[0], inputs[1],
                                        output_h, output_w, spatial_scale,
                                        sampling_ratio, aligned)};
    });

    table.register_kernel(OpId::ROIAlignBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        // inputs: [grad_output, rois]
        // attrs: batch_size, feat_height, feat_width, spatial_scale, sampling_ratio, aligned
        int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 1);
        int64_t feat_height = parse_attr<int64_t>(attrs, "feat_height", 0);
        int64_t feat_width = parse_attr<int64_t>(attrs, "feat_width", 0);
        float spatial_scale = parse_attr<float>(attrs, "spatial_scale", 1.0f / 16.0f);
        int64_t sampling_ratio = parse_attr<int64_t>(attrs, "sampling_ratio", 0);
        bool aligned = parse_attr<bool>(attrs, "aligned", true);

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
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::cudnn_conv2d_forward(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_conv2d_backward(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });
#else
    table.register_single_output_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, true, false, false, get_cuda_stream(attrs));
        return {grad_input};
    });
    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, true, false, get_cuda_stream(attrs));
        return {grad_weight};
    });
    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::conv2d_backward_kernel(
            inputs[0], inputs[1], inputs[2], stride, padding, dilation, groups, false, false, true, get_cuda_stream(attrs));
        return {grad_bias};
    });
#endif

    table.register_single_output_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t output_padding = parse_attr<int64_t>(attrs, "output_padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups, get_cuda_stream(attrs));
    });

    table.register_single_output_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return cuda::depthwise_conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Dropout Operations (Phase 1B - CRITICAL)
    // =========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = parse_attr<float>(attrs, "p", 0.5f);
        bool training = parse_attr<bool>(attrs, "training", true);
        auto [output, mask] = cuda::dropout_forward_kernel(inputs[0], p, training, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mask};
    });
    table.register_single_output_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        float p = parse_attr<float>(attrs, "p", 0.5f);
        return cuda::dropout_backward_kernel(inputs[0], inputs[1], p, get_cuda_stream(attrs));
    });

    // =========================================================================
    // LSTM Operations (Phase 1C - HIGH)
    // =========================================================================
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [gates, c_prev]
        int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
        int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
        auto [h_out, c_out] = cuda::lstm_cell_forward_kernel(inputs[0], inputs[1], batch_size, hidden_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{h_out, c_out};
    });
    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, grad_c, gates, c_prev, c_out]
        int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
        int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
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
        int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
        int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
        auto h_out = cuda::gru_cell_forward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
            batch_size, hidden_size, get_cuda_stream(attrs));
        return std::vector<Tensor>{h_out};
    });
    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_h, reset_gates, update_gates, new_gates_input, new_gates_hidden, h_prev]
        int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
        int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
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
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        auto [output, mean, inv_std] = cuda::cudnn_layer_norm_forward(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        auto [grad_input, grad_weight, grad_bias] = cuda::cudnn_layer_norm_backward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape, get_cuda_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });
#else
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        auto [output, mean, inv_std] = cuda::fused_layer_norm_cuda(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps);
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::LayerNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        auto [grad_input, grad_weight, grad_bias] = cuda::fused_layer_norm_backward_cuda(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape);
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });
#endif

    // =========================================================================
    // Adaptive Pooling Operations (Phase 1E / 3D - HIGH)
    // =========================================================================
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t output_h = parse_attr<int64_t>(attrs, "output_h", 1);
        int64_t output_w = parse_attr<int64_t>(attrs, "output_w", 1);
        return cuda::adaptive_avg_pool2d_forward(inputs[0], output_h, output_w, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t H_in = parse_attr<int64_t>(attrs, "input_h", 0);
        int64_t W_in = parse_attr<int64_t>(attrs, "input_w", 0);
        return cuda::adaptive_avg_pool2d_backward(inputs[0], H_in, W_in, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = parse_attr<int64_t>(attrs, "output_h", 1);
        int64_t output_w = parse_attr<int64_t>(attrs, "output_w", 1);
        auto [output, indices] = cuda::adaptive_max_pool2d_forward(inputs[0], output_h, output_w, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_single_output_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto input_shape = parse_int_list(attrs, "input_shape");
        return cuda::adaptive_max_pool2d_backward(inputs[0], inputs[1], input_shape, get_cuda_stream(attrs));
    });

    // =========================================================================
    // GroupNorm / InstanceNorm Operations (Phase 3A-B - HIGH)
    // =========================================================================
    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, weight, bias]
        int64_t num_groups = parse_attr<int64_t>(attrs, "num_groups", 1);
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        auto [output, mean, inv_std] = cuda::group_norm_forward_kernel(
            inputs[0], inputs[1], inputs[2], num_groups, eps, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, mean, inv_std};
    });
    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [grad_output, input, weight, mean, inv_std]
        int64_t num_groups = parse_attr<int64_t>(attrs, "num_groups", 1);
        auto [grad_input, grad_weight, grad_bias] = cuda::group_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], num_groups, get_cuda_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });
    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
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
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        bool descending = parse_attr<bool>(attrs, "descending", false);
        return cuda::argsort_kernel(inputs[0], dim, descending, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Creation Operations (Phase 3E - MEDIUM)
    // =========================================================================
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        Tensor output(shape, dtype, device);
        cudaMemsetAsync(output.data_ptr(), 0, output.numel() * dtype_size(dtype), get_cuda_stream(attrs));
        return std::vector<Tensor>{output};
    });
    table.register_kernel(OpId::Ones, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        Tensor output(shape, dtype, device);
        return std::vector<Tensor>{cuda::fill_kernel(output, 1.0f, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Full, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        float value = parse_attr<float>(attrs, "value", 0.0f);
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        Tensor output(shape, dtype, device);
        return std::vector<Tensor>{cuda::fill_kernel(output, value, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Rand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::rand_kernel(shape, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Randn, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::randn_kernel(shape, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Arange, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float start = parse_attr<float>(attrs, "start", 0.0f);
        float end = parse_attr<float>(attrs, "end", 0.0f);
        float step = parse_attr<float>(attrs, "step", 1.0f);
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::arange_kernel(start, end, step, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Linspace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float start = parse_attr<float>(attrs, "start", 0.0f);
        float end = parse_attr<float>(attrs, "end", 1.0f);
        int64_t steps = parse_attr<int64_t>(attrs, "steps", 100);
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::linspace_kernel(start, end, steps, dtype, device, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Eye, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = parse_attr<int64_t>(attrs, "n", 0);
        int64_t m = parse_attr<int64_t>(attrs, "m", -1);
        int dtype_int = parse_attr<int>(attrs, "dtype", static_cast<int>(DType::Float32));
        DType dtype = static_cast<DType>(dtype_int);
        int device_idx = parse_attr<int>(attrs, "device_index", 0);
        Device device = Device::cuda(device_idx);
        return std::vector<Tensor>{cuda::eye_kernel(n, m, dtype, device, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Shape/Transform Operations (Phase 3F - MEDIUM)
    // =========================================================================
    table.register_single_output_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t start_dim = parse_attr<int64_t>(attrs, "start_dim", 0);
        int64_t end_dim = parse_attr<int64_t>(attrs, "end_dim", -1);
        return cuda::flatten_kernel(inputs[0], start_dim, end_dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto starts = parse_int_list(attrs, "starts");
        auto ends = parse_int_list(attrs, "ends");
        auto steps = parse_int_list(attrs, "steps");
        return cuda::slice_kernel(inputs[0], starts, ends, steps, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::stack_kernel(inputs, dim, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Split, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t split_size = parse_attr<int64_t>(attrs, "split_size", 1);
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::split_kernel(inputs[0], split_size, dim, get_cuda_stream(attrs));
    });
    table.register_kernel(OpId::Chunk, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t chunks = parse_attr<int64_t>(attrs, "chunks", 1);
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return cuda::chunk_kernel(inputs[0], chunks, dim, get_cuda_stream(attrs));
    });
    table.register_single_output_kernel(OpId::Tile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto reps = parse_int_list(attrs, "reps");
        return cuda::tile_kernel(inputs[0], reps, get_cuda_stream(attrs));
    });

    // =========================================================================
    // Fused Softmax Cross Entropy (Phase 4A - MEDIUM)
    // =========================================================================
    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [logits, targets]
        bool compute_grad = parse_attr<bool>(attrs, "compute_grad", true);
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
        float lr = parse_attr<float>(attrs, "lr", 0.01f);
        float alpha = parse_attr<float>(attrs, "alpha", 0.99f);
        float eps = parse_attr<float>(attrs, "eps", 1e-8f);
        float weight_decay = parse_attr<float>(attrs, "weight_decay", 0.0f);
        float momentum = parse_attr<float>(attrs, "momentum", 0.0f);
        bool centered = parse_attr<bool>(attrs, "centered", false);

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
        float rho = parse_attr<float>(attrs, "rho", 0.9f);
        float eps = parse_attr<float>(attrs, "eps", 1e-6f);
        float lr = parse_attr<float>(attrs, "lr", 1.0f);
        float weight_decay = parse_attr<float>(attrs, "weight_decay", 0.0f);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& square_avg = const_cast<Tensor&>(inputs[2]);
        Tensor& acc_delta = const_cast<Tensor&>(inputs[3]);

        cuda::fused_adadelta_step_cuda(param, inputs[1], square_avg, acc_delta,
            rho, eps, lr, weight_decay, get_cuda_stream(attrs));
        return std::vector<Tensor>{param};
    });

    table.register_kernel(OpId::FusedAdagradStep, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [param, grad, sum_sq]
        float lr = parse_attr<float>(attrs, "lr", 0.01f);
        float lr_decay = parse_attr<float>(attrs, "lr_decay", 0.0f);
        float eps = parse_attr<float>(attrs, "eps", 1e-10f);
        float weight_decay = parse_attr<float>(attrs, "weight_decay", 0.0f);
        int64_t step = parse_attr<int64_t>(attrs, "step", 1);

        Tensor& param = const_cast<Tensor&>(inputs[0]);
        Tensor& sum_sq = const_cast<Tensor&>(inputs[2]);

        cuda::fused_adagrad_step_cuda(param, inputs[1], sum_sq,
            lr, lr_decay, eps, weight_decay, step, get_cuda_stream(attrs));
        return std::vector<Tensor>{param};
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
