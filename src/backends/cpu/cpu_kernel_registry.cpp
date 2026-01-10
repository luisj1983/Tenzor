/**
 * @file cpu_kernel_registry.cpp
 * @brief CPU kernel registration for O(1) dispatch
 *
 * Registers all CPU kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cstdlib>
#include <charconv>
#include <climits>

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

    // Reductions
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmax_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto argmin_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto prod_kernel(const Tensor& input, int64_t dim, bool keepdim) -> Tensor;
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, int64_t correction) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim) -> Tensor;
    auto argsort_kernel(const Tensor& input, int64_t dim, bool descending) -> Tensor;

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

    // Indexing
    auto index_select_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto gather_kernel(const Tensor& input, int64_t dim, const Tensor& index) -> Tensor;
    auto scatter_kernel(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src) -> Tensor;
    auto masked_select_kernel(const Tensor& input, const Tensor& mask) -> Tensor;
    auto masked_fill_kernel(const Tensor& input, const Tensor& mask, float value) -> Tensor;
    auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y) -> Tensor;
    auto slice_kernel(const Tensor& input, int64_t dim, int64_t start, int64_t end, int64_t step) -> Tensor;
    auto cat_kernel(const std::vector<Tensor>& tensors, int64_t dim) -> Tensor;

    // Normalization
    auto batchnorm2d_mean_var_kernel(const Tensor& input) -> std::vector<Tensor>;
    auto batchnorm2d_forward_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon) -> Tensor;
    auto batchnorm2d_forward_affine_kernel(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon) -> Tensor;
    auto batchnorm2d_update_running_stats_kernel(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum) -> void;
    auto batchnorm2d_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, float epsilon) -> std::vector<Tensor>;
    auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& mean, const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor>;
    auto group_norm_kernel(const Tensor& input, int64_t num_groups, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, int64_t num_groups, const Tensor& mean, const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor>;
    auto instance_norm_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& mean, const Tensor& rstd, const Tensor& weight) -> std::vector<Tensor>;

    // Convolution
    auto conv2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_input_kernel(const Tensor& grad_output, const Tensor& weight, const std::vector<int64_t>& input_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_weight_kernel(const Tensor& grad_output, const Tensor& input, const std::vector<int64_t>& weight_shape, int64_t stride, int64_t padding, int64_t dilation, int64_t groups) -> Tensor;
    auto conv2d_backward_bias_kernel(const Tensor& grad_output) -> Tensor;
    auto conv_transpose2d_forward_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t output_padding, int64_t dilation, int64_t groups) -> Tensor;
    auto depthwise_conv2d_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding, int64_t dilation) -> Tensor;

    // Pooling
    auto maxpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;
    auto avgpool2d_forward_kernel(const Tensor& input, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape, int64_t kernel_size, int64_t stride, int64_t padding) -> Tensor;
    auto adaptive_avgpool2d_kernel(const Tensor& input, int64_t output_h, int64_t output_w) -> Tensor;
    auto adaptive_avgpool2d_backward_kernel(const Tensor& grad_output, const std::vector<int64_t>& input_shape) -> Tensor;
    auto adaptive_maxpool2d_kernel(const Tensor& input, int64_t output_h, int64_t output_w) -> std::pair<Tensor, Tensor>;
    auto adaptive_maxpool2d_backward_kernel(const Tensor& grad_output, const Tensor& indices, const std::vector<int64_t>& input_shape) -> Tensor;

    // Fused operations
    auto fused_linear_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto fused_conv2d_relu_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, int64_t stride, int64_t padding) -> Tensor;
    auto fused_batchnorm_relu_kernel(const Tensor& input, const Tensor& running_mean, const Tensor& running_var, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
    auto fused_softmax_cross_entropy_kernel(const Tensor& logits, const Tensor& targets, const std::string& reduction) -> Tensor;
    auto fused_add_relu_kernel(const Tensor& a, const Tensor& b) -> Tensor;
    auto fused_gelu_kernel(const Tensor& input) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape, const Tensor& weight, const Tensor& bias, float eps) -> Tensor;

    // Creation
    auto zeros_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto ones_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto full_kernel(const std::vector<int64_t>& shape, float value, DType dtype, const Device& device) -> Tensor;
    auto rand_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
    auto randn_kernel(const std::vector<int64_t>& shape, DType dtype, const Device& device) -> Tensor;
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
                                    const Tensor& weight_ih, const Tensor& weight_hh) -> std::vector<Tensor>;
    auto gru_cell_forward_kernel(const Tensor& input, const Tensor& hx,
                                  const Tensor& weight_ih, const Tensor& weight_hh,
                                  const Tensor& bias_ih, const Tensor& bias_hh) -> Tensor;
    auto gru_cell_backward_kernel(const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
                                   const Tensor& weight_ih, const Tensor& weight_hh) -> std::vector<Tensor>;

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

    // Linear
    auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
    auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input, const Tensor& weight) -> std::vector<Tensor>;

    // Dropout
    auto dropout_kernel(const Tensor& input, float p, bool training) -> std::pair<Tensor, Tensor>;
    auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p) -> Tensor;
} // namespace cpu

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
    TENZOR_REGISTER_BINARY_KERNEL(table, Add, cpu::add_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Sub, cpu::sub_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Mul, cpu::mul_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Div, cpu::div_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, MatMul, cpu::matmul_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Bmm, cpu::bmm_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Dot, cpu::dot_kernel);

    // Single-output registrations for optimized dispatch (no vector allocation)
    // These avoid ~0.5-2us overhead per call from std::vector creation
    table.register_single_output_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return cpu::matmul_kernel(inputs[0], inputs[1]);
    });
    table.register_single_output_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes&) -> Tensor {
        return cpu::bmm_kernel(inputs[0], inputs[1]);
    });

    // Inplace operations
    TENZOR_REGISTER_INPLACE_KERNEL(table, AddInplace, cpu::add_inplace_kernel);
    TENZOR_REGISTER_INPLACE_KERNEL(table, SubInplace, cpu::sub_inplace_kernel);
    TENZOR_REGISTER_INPLACE_KERNEL(table, MulInplace, cpu::mul_inplace_kernel);
    TENZOR_REGISTER_INPLACE_KERNEL(table, DivInplace, cpu::div_inplace_kernel);

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

    // Use LLONG_MIN as sentinel for "reduce all dimensions" (no dim specified)
    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", LLONG_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        int64_t correction = parse_attr<int64_t>(attrs, "correction", 1);
        return std::vector<Tensor>{cpu::var_kernel(inputs[0], dim, keepdim, correction)};
    });

    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", LLONG_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        int64_t correction = parse_attr<int64_t>(attrs, "correction", 1);
        return std::vector<Tensor>{cpu::std_kernel(inputs[0], dim, keepdim, correction)};
    });

    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = parse_attr<float>(attrs, "p", 2.0f);
        int64_t dim = parse_attr<int64_t>(attrs, "dim", LLONG_MIN);
        bool keepdim = parse_attr<bool>(attrs, "keepdim", false);
        return std::vector<Tensor>{cpu::norm_kernel(inputs[0], p, dim, keepdim)};
    });

    table.register_kernel(OpId::ArgSort, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        bool descending = parse_attr<bool>(attrs, "descending", false);
        return std::vector<Tensor>{cpu::argsort_kernel(inputs[0], dim, descending)};
    });

    // =========================================================================
    // Element-wise Math Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_KERNEL(table, Sqrt, cpu::sqrt_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Neg, cpu::neg_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Abs, cpu::abs_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Sign, cpu::sign_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Log, cpu::log_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Exp, cpu::exp_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Reciprocal, cpu::reciprocal_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Floor, cpu::floor_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Ceil, cpu::ceil_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Round, cpu::round_kernel);

    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float exponent = parse_attr<float>(attrs, "exponent", 2.0f);
        return std::vector<Tensor>{cpu::pow_kernel(inputs[0], exponent)};
    });

    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = parse_attr<float>(attrs, "min", -std::numeric_limits<float>::infinity());
        float max_val = parse_attr<float>(attrs, "max", std::numeric_limits<float>::infinity());
        return std::vector<Tensor>{cpu::clamp_kernel(inputs[0], min_val, max_val)};
    });

    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = parse_attr<float>(attrs, "min", 0.0f);
        return std::vector<Tensor>{cpu::clamp_min_kernel(inputs[0], min_val)};
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float max_val = parse_attr<float>(attrs, "max", 0.0f);
        return std::vector<Tensor>{cpu::clamp_max_kernel(inputs[0], max_val)};
    });

    // =========================================================================
    // Trigonometric Operations
    // =========================================================================
    TENZOR_REGISTER_UNARY_KERNEL(table, Sin, cpu::sin_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Cos, cpu::cos_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Tan, cpu::tan_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Asin, cpu::asin_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Acos, cpu::acos_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Atan, cpu::atan_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Sinh, cpu::sinh_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Cosh, cpu::cosh_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Tanh, cpu::tanh_kernel);

    // =========================================================================
    // Comparison Operations
    // =========================================================================
    TENZOR_REGISTER_BINARY_KERNEL(table, Eq, cpu::eq_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Ne, cpu::ne_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Lt, cpu::lt_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Le, cpu::le_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Gt, cpu::gt_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, Ge, cpu::ge_kernel);

    // =========================================================================
    // Activation Functions
    // =========================================================================
    TENZOR_REGISTER_UNARY_KERNEL(table, ReLU, cpu::relu_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, ReLUBackward, cpu::relu_backward_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Sigmoid, cpu::sigmoid_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, SigmoidBackward, cpu::sigmoid_backward_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, TanhActivation, cpu::tanh_kernel);  // Uses same kernel as Tanh
    TENZOR_REGISTER_BINARY_KERNEL(table, TanhBackward, cpu::tanh_backward_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Gelu, cpu::gelu_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, GeluBackward, cpu::gelu_backward_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Swish, cpu::swish_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, SwishBackward, cpu::swish_backward_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Selu, cpu::selu_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, SeluBackward, cpu::selu_backward_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Mish, cpu::mish_kernel);
    TENZOR_REGISTER_BINARY_KERNEL(table, MishBackward, cpu::mish_backward_kernel);

    table.register_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        return std::vector<Tensor>{cpu::leaky_relu_kernel(inputs[0], alpha)};
    });

    table.register_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        return std::vector<Tensor>{cpu::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha)};
    });

    table.register_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 1.0f);
        return std::vector<Tensor>{cpu::elu_kernel(inputs[0], alpha)};
    });

    table.register_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 1.0f);
        return std::vector<Tensor>{cpu::elu_backward_kernel(inputs[0], inputs[1], alpha)};
    });

    table.register_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float beta = parse_attr<float>(attrs, "beta", 1.0f);
        float threshold = parse_attr<float>(attrs, "threshold", 20.0f);
        return std::vector<Tensor>{cpu::softplus_kernel(inputs[0], beta, threshold)};
    });

    table.register_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float beta = parse_attr<float>(attrs, "beta", 1.0f);
        float threshold = parse_attr<float>(attrs, "threshold", 20.0f);
        return std::vector<Tensor>{cpu::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold)};
    });

    table.register_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cpu::softmax_kernel(inputs[0], dim)};
    });

    table.register_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cpu::softmax_backward_kernel(inputs[0], inputs[1], dim)};
    });

    table.register_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cpu::log_softmax_kernel(inputs[0], dim)};
    });

    table.register_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cpu::log_softmax_backward_kernel(inputs[0], inputs[1], dim)};
    });

    // =========================================================================
    // Shape/View Operations
    // =========================================================================
    table.register_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        return std::vector<Tensor>{cpu::reshape_kernel(inputs[0], shape)};
    });

    table.register_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim0 = parse_attr<int64_t>(attrs, "dim0", 0);
        int64_t dim1 = parse_attr<int64_t>(attrs, "dim1", 1);
        return std::vector<Tensor>{cpu::transpose_kernel(inputs[0], dim0, dim1)};
    });

    table.register_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dims = parse_int_list(attrs, "dims");
        return std::vector<Tensor>{cpu::permute_kernel(inputs[0], dims)};
    });

    table.register_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cpu::squeeze_kernel(inputs[0], dim)};
    });

    table.register_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cpu::unsqueeze_kernel(inputs[0], dim)};
    });

    table.register_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t start_dim = parse_attr<int64_t>(attrs, "start_dim", 0);
        int64_t end_dim = parse_attr<int64_t>(attrs, "end_dim", -1);
        return std::vector<Tensor>{cpu::flatten_kernel(inputs[0], start_dim, end_dim)};
    });

    TENZOR_REGISTER_UNARY_KERNEL(table, Contiguous, cpu::contiguous_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, Clone, cpu::clone_kernel);

    table.register_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = parse_attr<float>(attrs, "value", 0.0f);
        return std::vector<Tensor>{cpu::fill_kernel(inputs[0], value)};
    });

    // =========================================================================
    // Indexing Operations
    // =========================================================================
    table.register_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cpu::index_select_kernel(inputs[0], dim, inputs[1])};
    });

    table.register_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cpu::gather_kernel(inputs[0], dim, inputs[1])};
    });

    table.register_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cpu::scatter_kernel(inputs[0], dim, inputs[1], inputs[2])};
    });

    TENZOR_REGISTER_BINARY_KERNEL(table, MaskedSelect, cpu::masked_select_kernel);

    table.register_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = parse_attr<float>(attrs, "value", 0.0f);
        return std::vector<Tensor>{cpu::masked_fill_kernel(inputs[0], inputs[1], value)};
    });

    TENZOR_REGISTER_TERNARY_KERNEL(table, Where, cpu::where_kernel);

    table.register_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        int64_t start = parse_attr<int64_t>(attrs, "start", 0);
        int64_t end = parse_attr<int64_t>(attrs, "end", -1);
        int64_t step = parse_attr<int64_t>(attrs, "step", 1);
        return std::vector<Tensor>{cpu::slice_kernel(inputs[0], dim, start, end, step)};
    });

    // Cat needs special handling for multiple inputs
    table.register_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return std::vector<Tensor>{cpu::cat_kernel(tensors, dim)};
    });

    // =========================================================================
    // Normalization Operations
    // =========================================================================
    table.register_kernel(OpId::BatchNorm2dMeanVar, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cpu::batchnorm2d_mean_var_kernel(inputs[0]);
    });

    table.register_kernel(OpId::BatchNorm2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        return std::vector<Tensor>{cpu::batchnorm2d_forward_kernel(inputs[0], inputs[1], inputs[2], epsilon)};
    });

    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        return std::vector<Tensor>{cpu::batchnorm2d_forward_affine_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon)};
    });

    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = parse_attr<float>(attrs, "epsilon", 1e-5f);
        return cpu::batchnorm2d_backward_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon);
    });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = parse_attr<float>(attrs, "momentum", 0.1f);
        // inputs: running_mean, running_var, batch_mean, batch_var
        // Note: This modifies running_mean and running_var in-place
        Tensor running_mean = inputs[0];  // Copy to allow modification
        Tensor running_var = inputs[1];
        cpu::batchnorm2d_update_running_stats_kernel(running_mean, running_var, inputs[2], inputs[3], momentum);
        return std::vector<Tensor>{running_mean, running_var};
    });

    // Register both multi-output and single-output versions for LayerNorm
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{cpu::layer_norm_kernel(inputs[0], normalized_shape, inputs[1], inputs[2], eps)};
    });

    // Single-output version for optimized dispatch (no vector allocation)
    table.register_single_output_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) -> Tensor {
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        return cpu::layer_norm_kernel(inputs[0], normalized_shape, inputs[1], inputs[2], eps);
    });

    table.register_kernel(OpId::GroupNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = parse_attr<int64_t>(attrs, "num_groups", 1);
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{cpu::group_norm_kernel(inputs[0], num_groups, inputs[1], inputs[2], eps)};
    });

    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{cpu::instance_norm_kernel(inputs[0], inputs[1], inputs[2], eps)};
    });

    // =========================================================================
    // Convolution Operations
    // =========================================================================
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto input_shape = parse_int_list(attrs, "input_shape");
        return std::vector<Tensor>{cpu::conv2d_backward_input_kernel(inputs[0], inputs[1], input_shape, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        auto weight_shape = parse_int_list(attrs, "weight_shape");
        return std::vector<Tensor>{cpu::conv2d_backward_weight_kernel(inputs[0], inputs[1], weight_shape, stride, padding, dilation, groups)};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::conv2d_backward_bias_kernel(inputs[0])};
    });

    table.register_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t output_padding = parse_attr<int64_t>(attrs, "output_padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        int64_t groups = parse_attr<int64_t>(attrs, "groups", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::conv_transpose2d_forward_kernel(inputs[0], inputs[1], bias, stride, padding, output_padding, dilation, groups)};
    });

    table.register_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        int64_t dilation = parse_attr<int64_t>(attrs, "dilation", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::depthwise_conv2d_kernel(inputs[0], inputs[1], bias, stride, padding, dilation)};
    });

    // =========================================================================
    // Pooling Operations
    // =========================================================================
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        auto [output, indices] = cpu::maxpool2d_forward_kernel(inputs[0], kernel_size, stride, padding);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = parse_int_list(attrs, "input_shape");
        return std::vector<Tensor>{cpu::maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape)};
    });

    table.register_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return std::vector<Tensor>{cpu::avgpool2d_forward_kernel(inputs[0], kernel_size, stride, padding)};
    });

    table.register_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = parse_int_list(attrs, "input_shape");
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return std::vector<Tensor>{cpu::avgpool2d_backward_kernel(inputs[0], input_shape, kernel_size, stride, padding)};
    });

    table.register_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = parse_attr<int64_t>(attrs, "output_h", 1);
        int64_t output_w = parse_attr<int64_t>(attrs, "output_w", 1);
        return std::vector<Tensor>{cpu::adaptive_avgpool2d_kernel(inputs[0], output_h, output_w)};
    });

    table.register_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = parse_int_list(attrs, "input_shape");
        return std::vector<Tensor>{cpu::adaptive_avgpool2d_backward_kernel(inputs[0], input_shape)};
    });

    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = parse_attr<int64_t>(attrs, "output_h", 1);
        int64_t output_w = parse_attr<int64_t>(attrs, "output_w", 1);
        auto [output, indices] = cpu::adaptive_maxpool2d_kernel(inputs[0], output_h, output_w);
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = parse_int_list(attrs, "input_shape");
        return std::vector<Tensor>{cpu::adaptive_maxpool2d_backward_kernel(inputs[0], inputs[1], input_shape)};
    });

    // =========================================================================
    // Fused Operations
    // =========================================================================
    table.register_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes&) {
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::fused_linear_relu_kernel(inputs[0], inputs[1], bias)};
    });

    table.register_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_attr<int64_t>(attrs, "stride", 1);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{cpu::fused_conv2d_relu_kernel(inputs[0], inputs[1], bias, stride, padding)};
    });

    table.register_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{cpu::fused_batchnorm_relu_kernel(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], eps)};
    });

    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        std::string reduction = parse_attr<std::string>(attrs, "reduction", "mean");
        return std::vector<Tensor>{cpu::fused_softmax_cross_entropy_kernel(inputs[0], inputs[1], reduction)};
    });

    TENZOR_REGISTER_BINARY_KERNEL(table, FusedAddReLU, cpu::fused_add_relu_kernel);
    TENZOR_REGISTER_UNARY_KERNEL(table, FusedGelu, cpu::fused_gelu_kernel);

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{cpu::fused_layer_norm_kernel(inputs[0], normalized_shape, inputs[1], inputs[2], eps)};
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

    table.register_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_embeddings = parse_attr<int64_t>(attrs, "num_embeddings", 0);
        return std::vector<Tensor>{cpu::embedding_backward_kernel(inputs[0], inputs[1], num_embeddings)};
    });

    // =========================================================================
    // Dropout Operations
    // =========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = parse_attr<float>(attrs, "p", 0.5f);
        bool training = parse_attr<bool>(attrs, "training", true);
        auto [output, mask] = cpu::dropout_kernel(inputs[0], p, training);
        return std::vector<Tensor>{output, mask};
    });

    table.register_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = parse_attr<float>(attrs, "p", 0.5f);
        return std::vector<Tensor>{cpu::dropout_backward_kernel(inputs[0], inputs[1], p)};
    });

    // =========================================================================
    // RNN Operations
    // =========================================================================
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return cpu::lstm_cell_forward_kernel(inputs[0], inputs[1], inputs[2],
                                              inputs[3], inputs[4], inputs[5], inputs[6]);
    });

    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes&) {
        return std::vector<Tensor>{cpu::gru_cell_forward_kernel(inputs[0], inputs[1],
                                                                  inputs[2], inputs[3],
                                                                  inputs[4], inputs[5])};
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
        int64_t num_layers = parse_attr<int64_t>(attrs, "num_layers", 1);

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
        int64_t num_layers = parse_attr<int64_t>(attrs, "num_layers", 1);

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

    // Note: Creation operations (Zeros, Ones, etc.) are handled differently
    // as they don't take tensor inputs. They're registered but need special
    // handling in the dispatch path for device specification.
}

} // namespace tenzor
