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
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/core/tensor.hpp"
#include <climits>
#include <cstdint>
#include <sstream>
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

    // ---- Statistical operations (kernels/statistical.cpp) ----
    auto std_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto var_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto prod_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;
    auto norm_kernel(const Tensor& input, const OpAttributes& attrs, sycl::queue& queue) -> Tensor;

    // ---- Transform operations (kernels/transform.cpp) ----
    auto reshape_kernel(const Tensor& input, const std::vector<int64_t>& new_shape, sycl::queue& queue) -> Tensor;
    auto transpose_kernel(const Tensor& input, int64_t dim0, int64_t dim1, sycl::queue& queue) -> Tensor;
    auto permute_kernel(const Tensor& input, const std::vector<int64_t>& dims, sycl::queue& queue) -> Tensor;
    auto squeeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto unsqueeze_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor;
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
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

    // ---- Embedding operations (kernels/embedding.cpp) ----
    auto embedding_lookup_kernel(const Tensor& indices, const Tensor& weights,
                                 int64_t padding_idx, sycl::queue& queue) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   int64_t vocab_size, int64_t embedding_dim,
                                   sycl::queue& queue) -> Tensor;
    auto embedding_bag_forward_kernel(const Tensor& embeddings, const Tensor& offsets,
                                      const std::string& mode, bool include_last_offset,
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

} // namespace oneapi


// ============================================================================
// Helper: get queue from inputs
// ============================================================================
static sycl::queue& get_q(std::span<const Tensor> inputs) {
    return oneapi_internal::get_queue(inputs[0].device().index);
}

static sycl::queue& get_q_device(int32_t device_id) {
    return oneapi_internal::get_queue(device_id);
}

// ============================================================================
// Helper: parse comma-separated shape from attributes
// ============================================================================
static std::vector<int64_t> parse_shape(const std::string& shape_str) {
    std::vector<int64_t> shape;
    std::string str = shape_str;
    size_t pos = 0;
    while ((pos = str.find(',')) != std::string::npos) {
        shape.push_back(std::stoll(str.substr(0, pos)));
        str.erase(0, pos + 1);
    }
    if (!str.empty()) {
        shape.push_back(std::stoll(str));
    }
    return shape;
}

// ============================================================================
// Helper: parse DType from attributes
// ============================================================================
static DType parse_dtype(const OpAttributes& attrs) {
    if (!attrs.contains("dtype")) return DType::Float32;

    const auto& s = attrs.at("dtype");
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
            float exponent = parse_attr<float>(attrs, "exponent", 2.0f);
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
    // Clamp Operations
    // =========================================================================

    table.register_kernel(OpId::Clamp,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float min_val = parse_attr<float>(attrs, "min", 0.0f);
            float max_val = parse_attr<float>(attrs, "max", 1.0f);
            return {oneapi::clamp_kernel(inputs[0], min_val, max_val, get_q(inputs))};
        });

    table.register_kernel(OpId::ClampMin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float min_val = parse_attr<float>(attrs, "min", 0.0f);
            return {oneapi::clamp_min_kernel(inputs[0], min_val, get_q(inputs))};
        });

    table.register_kernel(OpId::ClampMax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float max_val = parse_attr<float>(attrs, "max", 1.0f);
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
            int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
            bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
            return {oneapi::sum_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Mean,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
            bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
            return {oneapi::mean_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Max,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
            bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
            return {oneapi::max_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::Min,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", INT64_MIN);
            bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
            return {oneapi::min_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::ArgMax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
            return {oneapi::argmax_kernel(inputs[0], dim, keepdim, get_q(inputs))};
        });

    table.register_kernel(OpId::ArgMin,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
            return {oneapi::argmin_kernel(inputs[0], dim, keepdim, get_q(inputs))};
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
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            bool descending = parse_attr<bool>(attrs, "descending", false);
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
            float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
            return {oneapi::leaky_relu_kernel(inputs[0], alpha, get_q(inputs))};
        });

    table.register_kernel(OpId::LeakyReLUBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
            return {oneapi::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_q(inputs))};
        });

    table.register_kernel(OpId::Softmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            return {oneapi::softmax_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::SoftmaxBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            return {oneapi::softmax_backward_kernel(inputs[0], inputs[1], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::LogSoftmax,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            return {oneapi::log_softmax_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::LogSoftmaxBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
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
            float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
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
            auto shape = parse_shape(attrs.at("shape"));
            return {oneapi::reshape_kernel(inputs[0], shape, get_q(inputs))};
        });

    table.register_kernel(OpId::Transpose,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim0 = parse_attr<int64_t>(attrs, "dim0", 0);
            int64_t dim1 = parse_attr<int64_t>(attrs, "dim1", 1);
            return {oneapi::transpose_kernel(inputs[0], dim0, dim1, get_q(inputs))};
        });

    table.register_kernel(OpId::Permute,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto dims = parse_shape(attrs.at("dims"));
            return {oneapi::permute_kernel(inputs[0], dims, get_q(inputs))};
        });

    table.register_kernel(OpId::Squeeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
            return {oneapi::squeeze_kernel(inputs[0], dim, get_q(inputs))};
        });

    table.register_kernel(OpId::Unsqueeze,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
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
            float value = static_cast<float>(parse_attr<double>(attrs, "value", 0.0));
            return {oneapi::fill_kernel(inputs[0], value, get_q(inputs))};
        });

    table.register_kernel(OpId::Repeat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto repeats = parse_shape(attrs.at("repeats"));
            return {oneapi::repeat_kernel(inputs[0], repeats, get_q(inputs))};
        });

    table.register_kernel(OpId::Expand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            return {oneapi::expand_kernel(inputs[0], attrs, get_q(inputs))};
        });

    table.register_kernel(OpId::Cat,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
            return {oneapi::cat_kernel(inputs, dim, get_q(inputs))};
        });

    table.register_kernel(OpId::Stack,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
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
            int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
            return {oneapi::index_select_kernel(inputs[0], dim, inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Gather,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
            return {oneapi::gather_kernel(inputs[0], dim, inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::Scatter,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
            return {oneapi::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_kernel(OpId::MaskedSelect,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::masked_select_kernel(inputs[0], inputs[1], get_q(inputs))};
        });

    table.register_kernel(OpId::MaskedFill,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float value = parse_attr<float>(attrs, "value", 0.0f);
            return {oneapi::masked_fill_kernel(inputs[0], inputs[1], value, get_q(inputs))};
        });

    table.register_kernel(OpId::Where,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::where_kernel(inputs[0], inputs[1], inputs[2], get_q(inputs))};
        });

    table.register_kernel(OpId::Nonzero,
        [](std::span<const Tensor> inputs, const OpAttributes&) -> std::vector<Tensor> {
            return {oneapi::nonzero_kernel(inputs[0], get_q(inputs))};
        });

    table.register_kernel(OpId::OneHot,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_classes = parse_attr<int64_t>(attrs, "num_classes", -1);
            DType output_dtype = DType::Float32;
            if (attrs.contains("dtype")) {
                auto dt = attrs.at("dtype");
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
            int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
            int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
            int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
            int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
            return {oneapi::conv2d_forward(inputs[0], inputs[1], bias,
                                           stride, padding, dilation, groups, get_q(inputs))};
        });

    table.register_kernel(OpId::Conv2dBackwardInput,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto input_shape = parse_int_list(attrs, "input_shape");
            int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
            int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
            int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
            int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
            return {oneapi::conv2d_backward_input(inputs[0], inputs[1], input_shape,
                                                   stride, padding, dilation, groups, get_q(inputs))};
        });

    table.register_kernel(OpId::Conv2dBackwardWeight,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto weight_shape = parse_int_list(attrs, "weight_shape");
            int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
            int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
            int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
            int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
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
            int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
            int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
            int64_t output_padding = parse_attr<int64_t>(attrs, "output_padding", 0);
            int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
            int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
            return {oneapi::conv_transpose2d_forward(inputs[0], inputs[1], bias,
                                                      stride, padding, output_padding,
                                                      dilation, groups, get_q(inputs))};
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
            } else if (attrs.contains("input_shape")) {
                // Autograd path: input shape passed via attributes
                auto input_shape = parse_shape(attrs.at("input_shape"));
                Tensor dummy_input(input_shape, inputs[0].dtype(), inputs[0].device());
                return {oneapi::avg_pool2d_backward_kernel(inputs[0], dummy_input, attrs, get_q(inputs))};
            } else {
                throw std::invalid_argument("avg_pool2d_backward requires 2 inputs or 1 input with input_shape attribute");
            }
        });

    table.register_kernel(OpId::MaxPool2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t H_in = parse_attr<int64_t>(attrs, "H_in", 0);
            int64_t W_in = parse_attr<int64_t>(attrs, "W_in", 0);
            return {oneapi::max_pool2d_backward_with_indices(inputs[0], inputs[1], H_in, W_in, get_q(inputs))};
        });

    table.register_kernel(OpId::AdaptiveAvgPool2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t H_in = parse_attr<int64_t>(attrs, "H_in", 0);
            int64_t W_in = parse_attr<int64_t>(attrs, "W_in", 0);
            return {oneapi::adaptive_avgpool2d_backward(inputs[0], H_in, W_in, get_q(inputs))};
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
            float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
            return {oneapi::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_q(inputs))};
        });

    table.register_kernel(OpId::BatchNorm2dForwardAffine,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
            return {oneapi::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2],
                                                        inputs[3], inputs[4], epsilon, get_q(inputs))};
        });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float momentum = parse_attr<float>(attrs, "momentum", 0.1f);
            Tensor updated_mean = inputs[0].clone();
            Tensor updated_var = inputs[1].clone();
            auto& q = get_q(inputs);
            oneapi::batchnorm2d_update_running_stats(updated_mean, updated_var,
                                                     inputs[2], inputs[3], momentum, q);
            return {updated_mean, updated_var};
        });

    table.register_kernel(OpId::BatchNorm2dBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
            auto [grad_input, grad_gamma, grad_beta] = oneapi::batchnorm2d_backward(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_q(inputs));
            return {grad_input, grad_gamma, grad_beta};
        });

    // =========================================================================
    // Group Normalization Operations
    // =========================================================================

    table.register_kernel(OpId::GroupNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_groups = parse_attr<int64_t>(attrs, "num_groups", 1);
            float eps = parse_attr<float>(attrs, "eps", 1e-5f);
            const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
            const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
            return oneapi::group_norm_kernel(inputs[0], num_groups, weight, bias, eps, get_q(inputs));
        });

    table.register_kernel(OpId::GroupNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_groups = parse_attr<int64_t>(attrs, "num_groups", 1);
            return oneapi::group_norm_backward_kernel(inputs[0], inputs[1], inputs[2],
                                                      inputs[3], inputs[4], num_groups, get_q(inputs));
        });

    // =========================================================================
    // Creation Operations
    // =========================================================================

    table.register_kernel(OpId::Zeros,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = parse_shape(attrs.at("shape"));
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::zeros_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Ones,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = parse_shape(attrs.at("shape"));
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::ones_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Full,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = parse_shape(attrs.at("shape"));
            float value = static_cast<float>(parse_attr<double>(attrs, "value", 0.0));
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::full_kernel(shape, value, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Rand,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = parse_shape(attrs.at("shape"));
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::rand_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Randn,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto shape = parse_shape(attrs.at("shape"));
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            Device device = inputs.empty() ? Device::oneapi(device_id) : inputs[0].device();
            return {oneapi::randn_kernel(shape, dtype, device, get_q_device(device_id))};
        });

    table.register_kernel(OpId::Arange,
        [](std::span<const Tensor>, const OpAttributes& attrs) -> std::vector<Tensor> {
            double start = parse_attr<double>(attrs, "start", 0.0);
            double end_val = parse_attr<double>(attrs, "end", 0.0);
            double step = parse_attr<double>(attrs, "step", 1.0);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            return {oneapi::arange_kernel(start, end_val, step, dtype,
                                           Device(Device::Type::OneAPI, device_id),
                                           get_q_device(device_id))};
        });

    table.register_kernel(OpId::Linspace,
        [](std::span<const Tensor>, const OpAttributes& attrs) -> std::vector<Tensor> {
            double start = parse_attr<double>(attrs, "start", 0.0);
            double end_val = parse_attr<double>(attrs, "end", 1.0);
            int64_t steps = parse_attr<int64_t>(attrs, "steps", 100);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            return {oneapi::linspace_kernel(start, end_val, steps, dtype,
                                             Device(Device::Type::OneAPI, device_id),
                                             get_q_device(device_id))};
        });

    table.register_kernel(OpId::Eye,
        [](std::span<const Tensor>, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t n = parse_attr<int64_t>(attrs, "n", 1);
            int64_t m = parse_attr<int64_t>(attrs, "m", n);
            DType dtype = parse_dtype(attrs);
            int32_t device_id = parse_attr<int>(attrs, "device_id", 0);
            return {oneapi::eye_kernel(n, m, dtype,
                                        Device(Device::Type::OneAPI, device_id),
                                        get_q_device(device_id))};
        });

    // =========================================================================
    // Embedding Operations
    // =========================================================================

    table.register_kernel(OpId::Embedding,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t padding_idx = parse_attr<int64_t>(attrs, "padding_idx", -1);
            // inputs[0] = weights, inputs[1] = indices (from nn::Embedding layer)
            return {oneapi::embedding_lookup_kernel(inputs[1], inputs[0], padding_idx, get_q(inputs))};
        });

    table.register_kernel(OpId::EmbeddingBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t vocab_size = parse_attr<int64_t>(attrs, "num_embeddings", 0);
            int64_t embedding_dim = inputs[0].shape().back();
            return {oneapi::embedding_backward_kernel(inputs[0], inputs[1], vocab_size,
                                                       embedding_dim, get_q(inputs))};
        });

    // =========================================================================
    // RNN Operations
    // =========================================================================

    table.register_kernel(OpId::LSTMCellForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
            int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
            auto [h_out, c_out] = oneapi::lstm_cell_forward_kernel(
                inputs[0], inputs[1], batch_size, hidden_size, get_q(inputs));
            return {h_out, c_out};
        });

    table.register_kernel(OpId::LSTMCellBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
            int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
            auto [grad_gates, grad_c_prev] = oneapi::lstm_cell_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                batch_size, hidden_size, get_q(inputs));
            return {grad_gates, grad_c_prev};
        });

    table.register_kernel(OpId::GRUCellForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
            int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
            return {oneapi::gru_cell_forward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4],
                batch_size, hidden_size, get_q(inputs))};
        });

    table.register_kernel(OpId::GRUCellBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
            int64_t hidden_size = parse_attr<int64_t>(attrs, "hidden_size", 0);
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
            auto normalized_shape = parse_shape(attrs.at("normalized_shape"));
            float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
            auto [output, mean, inv_std] = oneapi::fused_layer_norm_kernel(
                inputs[0], inputs[1], inputs[2], normalized_shape, epsilon, get_q(inputs));
            return {output, mean, inv_std};
        });

    table.register_kernel(OpId::LayerNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto normalized_shape = parse_shape(attrs.at("normalized_shape"));
            auto [grad_input, grad_weight, grad_bias] = oneapi::fused_layer_norm_backward_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], normalized_shape, get_q(inputs));
            return {grad_input, grad_weight, grad_bias};
        });

    // Note: OpId::FusedLayerNormBackward also maps to fused_layer_norm_backward
    table.register_kernel(OpId::FusedLayerNormBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto normalized_shape = parse_shape(attrs.at("normalized_shape"));
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
            float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
            return {oneapi::fused_batchnorm_relu_kernel(
                inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_q(inputs))};
        });

    table.register_kernel(OpId::FusedSoftmaxCrossEntropy,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            std::string reduction = parse_attr<std::string>(attrs, "reduction", "mean");
            return {oneapi::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], reduction, get_q(inputs))};
        });

    table.register_kernel(OpId::FusedRMSNorm,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            float epsilon = parse_attr<float>(attrs, "eps", 1e-5f);
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
    // Vision Operations
    // =========================================================================

    table.register_kernel(OpId::ROIAlignForward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t output_height = attrs.contains("output_height")
                ? parse_attr<int64_t>(attrs, "output_height", 0)
                : parse_attr<int64_t>(attrs, "output_h", 0);
            int64_t output_width = attrs.contains("output_width")
                ? parse_attr<int64_t>(attrs, "output_width", 0)
                : parse_attr<int64_t>(attrs, "output_w", 0);
            float spatial_scale = parse_attr<float>(attrs, "spatial_scale", 1.0f);
            int64_t sampling_ratio = parse_attr<int64_t>(attrs, "sampling_ratio", 0);
            bool aligned = parse_attr<bool>(attrs, "aligned", false);
            return {oneapi::roi_align_kernel(inputs[0], inputs[1], output_height, output_width,
                                              spatial_scale, sampling_ratio, aligned, get_q(inputs))};
        });

    table.register_kernel(OpId::ROIAlignBackward,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t batch_size = parse_attr<int64_t>(attrs, "batch_size", 0);
            int64_t channels = inputs[0].shape()[1];
            int64_t feat_height = parse_attr<int64_t>(attrs, "feat_height", 0);
            int64_t feat_width = parse_attr<int64_t>(attrs, "feat_width", 0);
            float spatial_scale = parse_attr<float>(attrs, "spatial_scale", 1.0f);
            int64_t sampling_ratio = parse_attr<int64_t>(attrs, "sampling_ratio", 0);
            bool aligned = parse_attr<bool>(attrs, "aligned", false);
            return {oneapi::roi_align_backward_kernel(inputs[0], inputs[1], batch_size, channels,
                                                       feat_height, feat_width, spatial_scale,
                                                       sampling_ratio, aligned, get_q(inputs))};
        });

    table.register_kernel(OpId::Interpolate,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            auto size = parse_int_list(attrs, "size");
            std::string mode = parse_attr<std::string>(attrs, "mode", "bilinear");
            bool align_corners = parse_attr<bool>(attrs, "align_corners", false);
            return {oneapi::interpolate_kernel(inputs[0], size, mode, align_corners, get_q(inputs))};
        });

    table.register_kernel(OpId::GatherRelativePositionBias,
        [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> std::vector<Tensor> {
            int64_t num_positions = parse_attr<int64_t>(attrs, "num_positions", 0);
            int64_t num_heads = parse_attr<int64_t>(attrs, "num_heads", 0);
            return {oneapi::gather_relative_position_bias_kernel(
                inputs[0], inputs[1], num_positions, num_heads, get_q(inputs))};
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
