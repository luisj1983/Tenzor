/**
 * @file rocm_kernel_registry.cpp
 * @brief ROCm kernel registration for O(1) dispatch
 *
 * Registers all ROCm/HIP kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include <hip/hip_runtime.h>
#include <iostream>
#include <cstdlib>
#include <charconv>
#include <limits>
#include <climits>
#include <tuple>

namespace tenzor {

// Helper to extract HIP stream from attributes
inline hipStream_t get_hip_stream(const OpAttributes& attrs) {
    if (attrs.contains("stream")) {
        return static_cast<hipStream_t>(
            reinterpret_cast<void*>(std::stoull(attrs.at("stream")))
        );
    }
    return nullptr;  // Default stream
}

// Helper to parse int64_t from attributes
inline int64_t parse_int64(const OpAttributes& attrs, const std::string& key, int64_t default_val = 0) {
    if (attrs.contains(key)) {
        return std::stoll(attrs.at(key));
    }
    return default_val;
}

// Helper to parse float from attributes
inline float parse_float(const OpAttributes& attrs, const std::string& key, float default_val = 0.0f) {
    if (attrs.contains(key)) {
        return std::stof(attrs.at(key));
    }
    return default_val;
}

// Helper to parse bool from attributes
inline bool parse_bool(const OpAttributes& attrs, const std::string& key, bool default_val = false) {
    if (attrs.contains(key)) {
        return attrs.at(key) == "1" || attrs.at(key) == "true";
    }
    return default_val;
}

// Helper to parse string from attributes
inline std::string parse_string(const OpAttributes& attrs, const std::string& key, const std::string& default_val = "") {
    if (attrs.contains(key)) {
        return attrs.at(key);
    }
    return default_val;
}

// Helper to parse vector of int64_t from comma-separated string
inline std::vector<int64_t> parse_shape(const OpAttributes& attrs, const std::string& key) {
    std::vector<int64_t> result;
    if (!attrs.contains(key)) return result;

    std::string str = attrs.at(key);
    size_t pos = 0;
    while (pos < str.size()) {
        size_t comma = str.find(',', pos);
        if (comma == std::string::npos) comma = str.size();
        result.push_back(std::stoll(str.substr(pos, comma - pos)));
        pos = comma + 1;
    }
    return result;
}

// Helper to parse DType from string
inline DType parse_dtype(const OpAttributes& attrs, const std::string& key, DType default_val = DType::Float32) {
    if (!attrs.contains(key)) return default_val;
    auto dtype_str = attrs.at(key);
    if (dtype_str == "float32") return DType::Float32;
    else if (dtype_str == "float64") return DType::Float64;
    else if (dtype_str == "float16") return DType::Float16;
    else if (dtype_str == "int32") return DType::Int32;
    else if (dtype_str == "int64") return DType::Int64;
    else if (dtype_str == "int8") return DType::Int8;
    else if (dtype_str == "uint8") return DType::UInt8;
    else if (dtype_str == "bool") return DType::Bool;
    return default_val;
}

// Helper to parse double from attributes
inline double parse_double(const OpAttributes& attrs, const std::string& key, double default_val = 0.0) {
    if (attrs.contains(key)) {
        return std::stod(attrs.at(key));
    }
    return default_val;
}

// Helper to parse vector of int64_t from comma-separated string (alias for parse_shape)
inline std::vector<int64_t> parse_int64_vector(const OpAttributes& attrs, const std::string& key) {
    return parse_shape(attrs, key);
}

// Helper to parse Device from attributes
inline Device parse_device(const OpAttributes& attrs, const std::string& key, Device default_val = Device::rocm(0)) {
    if (!attrs.contains(key)) return default_val;
    auto device_str = attrs.at(key);
    // Parse strings like "rocm:0", "cuda:1", "cpu"
    size_t colon_pos = device_str.find(':');
    std::string type_str = (colon_pos != std::string::npos) ? device_str.substr(0, colon_pos) : device_str;
    int32_t device_id = (colon_pos != std::string::npos) ? std::stoi(device_str.substr(colon_pos + 1)) : 0;

    Device::Type type = Device::Type::CPU;
    if (type_str == "rocm" || type_str == "hip") type = Device::Type::ROCm;
    else if (type_str == "cuda") type = Device::Type::CUDA;
    else if (type_str == "vulkan") type = Device::Type::Vulkan;

    return Device{type, device_id};
}

// Forward declarations for ROCm kernels
namespace rocm {
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
    auto var_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor;
    auto std_kernel(const Tensor& input, int64_t dim, bool keepdim, bool unbiased, hipStream_t stream) -> Tensor;
    auto norm_kernel(const Tensor& input, float p, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;

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
                               hipStream_t stream) -> Tensor;
    auto conv2d_backward_input(const Tensor& grad_output, const Tensor& weight,
                               const std::vector<int64_t>& input_shape,
                               int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                               hipStream_t stream) -> Tensor;
    auto conv2d_backward_weight(const Tensor& grad_output, const Tensor& input,
                                const std::vector<int64_t>& weight_shape,
                                int64_t stride, int64_t padding, int64_t dilation, int64_t groups,
                                hipStream_t stream) -> Tensor;
    auto conv2d_backward_bias(const Tensor& grad_output, hipStream_t stream) -> Tensor;

    // Pooling operations
    auto maxpool2d_forward_hip(const Tensor& input, int64_t kernel_size, int64_t stride,
                               int64_t padding, int64_t dilation, hipStream_t stream)
        -> std::pair<Tensor, Tensor>;
    auto maxpool2d_backward_hip(const Tensor& grad_output, const Tensor& indices,
                                const std::vector<int64_t>& input_shape, hipStream_t stream) -> Tensor;
    auto avgpool2d_forward_hip(const Tensor& input, int64_t kernel_size, int64_t stride,
                               int64_t padding, hipStream_t stream) -> Tensor;
    auto avgpool2d_backward_hip(const Tensor& grad_output, const std::vector<int64_t>& input_shape,
                                int64_t kernel_size, int64_t stride, int64_t padding,
                                hipStream_t stream) -> Tensor;
    auto adaptive_avgpool2d_hip(const Tensor& input, int64_t output_h, int64_t output_w,
                                hipStream_t stream) -> Tensor;
    auto adaptive_maxpool2d_hip(const Tensor& input, int64_t output_h, int64_t output_w,
                                hipStream_t stream) -> std::pair<Tensor, Tensor>;

    // Indexing operations
    auto gather_hip(const Tensor& input, int64_t dim, const Tensor& index, hipStream_t stream) -> Tensor;
    auto scatter_hip(const Tensor& input, int64_t dim, const Tensor& index, const Tensor& src,
                     hipStream_t stream) -> Tensor;
    auto index_select_hip(const Tensor& input, int64_t dim, const Tensor& index, hipStream_t stream) -> Tensor;
    auto masked_fill_hip(const Tensor& input, const Tensor& mask, double value, hipStream_t stream) -> Tensor;
    auto masked_select_hip(const Tensor& input, const Tensor& mask, hipStream_t stream) -> Tensor;

    // LSTM/GRU operations
    auto lstm_cell_forward_kernel(const Tensor& input, const Tensor& hx, const Tensor& cx,
                                  const Tensor& weight_ih, const Tensor& weight_hh,
                                  const Tensor& bias_ih, const Tensor& bias_hh,
                                  hipStream_t stream) -> std::tuple<Tensor, Tensor>;
    auto lstm_cell_backward_kernel(const Tensor& grad_hy, const Tensor& grad_cy,
                                   const Tensor& input, const Tensor& hx, const Tensor& cx,
                                   const Tensor& hy, const Tensor& cy,
                                   const Tensor& weight_ih, const Tensor& weight_hh,
                                   hipStream_t stream)
        -> std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;
    auto gru_cell_forward_kernel(const Tensor& input, const Tensor& hx,
                                 const Tensor& weight_ih, const Tensor& weight_hh,
                                 const Tensor& bias_ih, const Tensor& bias_hh,
                                 hipStream_t stream) -> Tensor;
    auto gru_cell_backward_kernel(const Tensor& grad_hy, const Tensor& input, const Tensor& hx,
                                  const Tensor& hy, const Tensor& weight_ih, const Tensor& weight_hh,
                                  hipStream_t stream)
        -> std::tuple<Tensor, Tensor, Tensor, Tensor, Tensor, Tensor>;

    // Full sequence LSTM/GRU operations
    auto lstm_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                            const Tensor& bias, const Tensor& h0, const Tensor& c0,
                            hipStream_t stream) -> std::vector<Tensor>;
    auto gru_forward_kernel(const Tensor& input, const Tensor& W_ih, const Tensor& W_hh,
                           const Tensor& bias, const Tensor& h0,
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

    // Normalization operations
    auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                          const Tensor* weight, const Tensor* bias, float eps, hipStream_t stream) -> Tensor;
    auto layer_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd, const Tensor* weight,
                                    hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto group_norm_kernel(const Tensor& input, int64_t num_groups, const Tensor* weight,
                          const Tensor* bias, float eps, hipStream_t stream) -> Tensor;
    auto group_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd, int64_t num_groups,
                                    const Tensor* weight, hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;
    auto instance_norm_kernel(const Tensor& input, const Tensor* weight, const Tensor* bias,
                             float eps, hipStream_t stream) -> Tensor;
    auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                       const Tensor& mean, const Tensor& rstd, const Tensor* weight,
                                       hipStream_t stream) -> std::tuple<Tensor, Tensor, Tensor>;

    // Fused operations
    auto fused_linear_relu_hip(const Tensor& input, const Tensor& weight, const Tensor* bias) -> Tensor;
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

    // Embedding, Linear, Dropout operations
    auto embedding_kernel(const Tensor& weight, const Tensor& indices, hipStream_t stream) -> Tensor;
    auto embedding_backward_kernel(const Tensor& grad_output, const Tensor& indices,
                                   int64_t num_embeddings, hipStream_t stream) -> Tensor;
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
    table.register_kernel(OpId::Add, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::add_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Sub, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sub_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Mul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::mul_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Div, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::div_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::matmul_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    // Bmm (batched matrix multiplication) uses the same kernel as MatMul
    // matmul_kernel already supports batched operations via rocblas_*gemm_strided_batched
    table.register_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::matmul_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    // ========================================================================
    // Unary Math Operations
    // ========================================================================
    table.register_kernel(OpId::Sqrt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sqrt_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Neg, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::neg_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Abs, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::abs_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Sign, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sign_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Log, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::log_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Exp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::exp_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Pow, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float exponent = parse_float(attrs, "exponent", 2.0f);
        return std::vector<Tensor>{rocm::pow_kernel(inputs[0], exponent, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Clamp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = parse_float(attrs, "min", -std::numeric_limits<float>::infinity());
        float max_val = parse_float(attrs, "max", std::numeric_limits<float>::infinity());
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], min_val, max_val, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ClampMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float min_val = parse_float(attrs, "min", -std::numeric_limits<float>::infinity());
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], min_val, std::numeric_limits<float>::infinity(), get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ClampMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float max_val = parse_float(attrs, "max", std::numeric_limits<float>::infinity());
        return std::vector<Tensor>{rocm::clamp_kernel(inputs[0], -std::numeric_limits<float>::infinity(), max_val, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Reciprocal, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::reciprocal_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Floor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::floor_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Ceil, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::ceil_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Round, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::round_kernel(inputs[0], get_hip_stream(attrs))};
    });

    // ========================================================================
    // Trigonometric Operations
    // ========================================================================
    table.register_kernel(OpId::Sin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sin_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Cos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::cos_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Tan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::tan_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Asin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::asin_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Acos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::acos_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Atan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::atan_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Sinh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sinh_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Cosh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::cosh_kernel(inputs[0], get_hip_stream(attrs))};
    });

    // ========================================================================
    // Additional Binary Operations
    // ========================================================================
    table.register_kernel(OpId::Dot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::dot_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    // In-place operations
    table.register_kernel(OpId::AddInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor a = inputs[0];  // Copy to allow modification
        rocm::add_inplace_kernel(a, inputs[1], get_hip_stream(attrs));
        return std::vector<Tensor>{a};
    });

    table.register_kernel(OpId::SubInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor a = inputs[0];
        rocm::sub_inplace_kernel(a, inputs[1], get_hip_stream(attrs));
        return std::vector<Tensor>{a};
    });

    table.register_kernel(OpId::MulInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor a = inputs[0];
        rocm::mul_inplace_kernel(a, inputs[1], get_hip_stream(attrs));
        return std::vector<Tensor>{a};
    });

    table.register_kernel(OpId::DivInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor a = inputs[0];
        rocm::div_inplace_kernel(a, inputs[1], get_hip_stream(attrs));
        return std::vector<Tensor>{a};
    });

    // ========================================================================
    // Reduction Operations
    // ========================================================================
    // Note: Use INT64_MIN as default to signal "full reduction" (no dim specified)
    // This is distinct from dim=-1 which means "last dimension" in PyTorch convention
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::sum_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::mean_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::max_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::min_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ArgMax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::argmax_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ArgMin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::argmin_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Prod, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::prod_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Var, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        bool unbiased = parse_bool(attrs, "unbiased", true);
        return std::vector<Tensor>{rocm::var_kernel(inputs[0], dim, keepdim, unbiased, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Std, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        bool unbiased = parse_bool(attrs, "unbiased", true);
        return std::vector<Tensor>{rocm::std_kernel(inputs[0], dim, keepdim, unbiased, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Norm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float p = parse_float(attrs, "p", 2.0f);
        int64_t dim = parse_int64(attrs, "dim", INT64_MIN);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::norm_kernel(inputs[0], p, dim, keepdim, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Activation Functions
    // ========================================================================
    table.register_kernel(OpId::ReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::relu_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::relu_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Sigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sigmoid_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::sigmoid_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::TanhActivation, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::tanh_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Tanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::tanh_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::tanh_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_float(attrs, "alpha", 0.01f);
        return std::vector<Tensor>{rocm::leaky_relu_kernel(inputs[0], alpha, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_float(attrs, "alpha", 0.01f);
        return std::vector<Tensor>{rocm::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Gelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::gelu_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::GeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::gelu_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_float(attrs, "alpha", 1.0f);
        return std::vector<Tensor>{rocm::elu_kernel(inputs[0], alpha, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_float(attrs, "alpha", 1.0f);
        return std::vector<Tensor>{rocm::elu_backward_kernel(inputs[0], inputs[1], alpha, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Selu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::selu_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::SeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::selu_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Swish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::swish_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::SwishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::swish_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Mish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::mish_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::MishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::mish_backward_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float beta = parse_float(attrs, "beta", 1.0f);
        float threshold = parse_float(attrs, "threshold", 20.0f);
        return std::vector<Tensor>{rocm::softplus_kernel(inputs[0], beta, threshold, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float beta = parse_float(attrs, "beta", 1.0f);
        float threshold = parse_float(attrs, "threshold", 20.0f);
        return std::vector<Tensor>{rocm::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Softmax Operations
    // ========================================================================
    table.register_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        return std::vector<Tensor>{rocm::softmax_kernel(inputs[0], dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        return std::vector<Tensor>{rocm::softmax_backward_kernel(inputs[0], inputs[1], dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        return std::vector<Tensor>{rocm::log_softmax_kernel(inputs[0], dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        return std::vector<Tensor>{rocm::log_softmax_backward_kernel(inputs[0], inputs[1], dim, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Tensor Creation Operations
    // ========================================================================
    table.register_kernel(OpId::Zeros, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        int32_t device_id = static_cast<int32_t>(parse_int64(attrs, "device_id", 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::zeros_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Ones, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        int32_t device_id = static_cast<int32_t>(parse_int64(attrs, "device_id", 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::ones_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Full, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        float value = parse_float(attrs, "value", 0.0f);
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        int32_t device_id = static_cast<int32_t>(parse_int64(attrs, "device_id", 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::full_kernel(shape, value, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = parse_float(attrs, "value", 0.0f);
        return std::vector<Tensor>{rocm::fill_kernel(inputs[0], value, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Rand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        int32_t device_id = static_cast<int32_t>(parse_int64(attrs, "device_id", 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::rand_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Randn, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        int32_t device_id = static_cast<int32_t>(parse_int64(attrs, "device_id", 0));
        Device device = Device::rocm(device_id);
        return std::vector<Tensor>{rocm::randn_kernel(shape, dtype, device, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Transform Operations
    // ========================================================================
    table.register_kernel(OpId::Contiguous, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::contiguous_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Clone, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::clone_kernel(inputs[0], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        return std::vector<Tensor>{rocm::reshape_kernel(inputs[0], shape, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim0 = parse_int64(attrs, "dim0", 0);
        int64_t dim1 = parse_int64(attrs, "dim1", 1);
        return std::vector<Tensor>{rocm::transpose_kernel(inputs[0], dim0, dim1, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dims = parse_shape(attrs, "dims");
        return std::vector<Tensor>{rocm::permute_kernel(inputs[0], dims, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        return std::vector<Tensor>{rocm::squeeze_kernel(inputs[0], dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        return std::vector<Tensor>{rocm::unsqueeze_kernel(inputs[0], dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_shape(attrs, "shape");
        return std::vector<Tensor>{rocm::expand_kernel(inputs[0], shape, static_cast<void*>(get_hip_stream(attrs)))};
    });

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
        float epsilon = parse_float(attrs, "epsilon", 1e-5f);
        if (attrs.contains("eps")) epsilon = parse_float(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{rocm::batchnorm2d_forward(inputs[0], inputs[1], inputs[2], epsilon, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::BatchNorm2dForwardAffine, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = parse_float(attrs, "epsilon", 1e-5f);
        if (attrs.contains("eps")) epsilon = parse_float(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{rocm::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = parse_float(attrs, "momentum", 0.1f);
        Tensor running_mean = inputs[0];
        Tensor running_var = inputs[1];
        rocm::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, get_hip_stream(attrs));
        return std::vector<Tensor>{running_mean, running_var};
    });

    table.register_kernel(OpId::BatchNorm2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float epsilon = parse_float(attrs, "epsilon", 1e-5f);
        if (attrs.contains("eps")) epsilon = parse_float(attrs, "eps", 1e-5f);
        auto [grad_input, grad_gamma, grad_beta] = rocm::batchnorm2d_backward(
            inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_gamma, grad_beta};
    });

    // ========================================================================
    // Convolution Operations
    // ========================================================================
    table.register_kernel(OpId::Conv2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_int64(attrs, "stride", 1);
        int64_t padding = parse_int64(attrs, "padding", 0);
        int64_t dilation = parse_int64(attrs, "dilation", 1);
        int64_t groups = parse_int64(attrs, "groups", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::conv2d_forward_kernel(inputs[0], inputs[1], bias,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv2dBackwardInput, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_int64(attrs, "stride", 1);
        int64_t padding = parse_int64(attrs, "padding", 0);
        int64_t dilation = parse_int64(attrs, "dilation", 1);
        int64_t groups = parse_int64(attrs, "groups", 1);
        auto input_shape = parse_shape(attrs, "input_shape");
        return std::vector<Tensor>{rocm::conv2d_backward_input(inputs[0], inputs[1], input_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv2dBackwardWeight, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride = parse_int64(attrs, "stride", 1);
        int64_t padding = parse_int64(attrs, "padding", 0);
        int64_t dilation = parse_int64(attrs, "dilation", 1);
        int64_t groups = parse_int64(attrs, "groups", 1);
        auto weight_shape = parse_shape(attrs, "weight_shape");
        return std::vector<Tensor>{rocm::conv2d_backward_weight(inputs[0], inputs[1], weight_shape,
            stride, padding, dilation, groups, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Conv2dBackwardBias, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::conv2d_backward_bias(inputs[0], get_hip_stream(attrs))};
    });

    // ========================================================================
    // Pooling Operations
    // ========================================================================
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_int64(attrs, "kernel_size", 2);
        int64_t stride = parse_int64(attrs, "stride", kernel_size);
        int64_t padding = parse_int64(attrs, "padding", 0);
        int64_t dilation = parse_int64(attrs, "dilation", 1);
        auto [output, indices] = rocm::maxpool2d_forward_hip(inputs[0], kernel_size, stride,
            padding, dilation, get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::MaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto input_shape = parse_shape(attrs, "input_shape");
        return std::vector<Tensor>{rocm::maxpool2d_backward_hip(inputs[0], inputs[1],
            input_shape, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_int64(attrs, "kernel_size", 2);
        int64_t stride = parse_int64(attrs, "stride", kernel_size);
        int64_t padding = parse_int64(attrs, "padding", 0);
        return std::vector<Tensor>{rocm::avgpool2d_forward_hip(inputs[0], kernel_size, stride,
            padding, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::AvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_int64(attrs, "kernel_size", 2);
        int64_t stride = parse_int64(attrs, "stride", kernel_size);
        int64_t padding = parse_int64(attrs, "padding", 0);
        auto input_shape = parse_shape(attrs, "input_shape");
        return std::vector<Tensor>{rocm::avgpool2d_backward_hip(inputs[0], input_shape,
            kernel_size, stride, padding, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::AdaptiveAvgPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = parse_int64(attrs, "output_h", 1);
        int64_t output_w = parse_int64(attrs, "output_w", 1);
        return std::vector<Tensor>{rocm::adaptive_avgpool2d_hip(inputs[0], output_h, output_w,
            get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::AdaptiveMaxPool2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t output_h = parse_int64(attrs, "output_h", 1);
        int64_t output_w = parse_int64(attrs, "output_w", 1);
        auto [output, indices] = rocm::adaptive_maxpool2d_hip(inputs[0], output_h, output_w,
            get_hip_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });

    table.register_kernel(OpId::AdaptiveAvgPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::adaptive_avgpool2d_backward_hip(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::AdaptiveMaxPool2dBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::adaptive_maxpool2d_backward_hip(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::ConvTranspose2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride_h = parse_int64(attrs, "stride_h", 1);
        int64_t stride_w = parse_int64(attrs, "stride_w", 1);
        int64_t padding_h = parse_int64(attrs, "padding_h", 0);
        int64_t padding_w = parse_int64(attrs, "padding_w", 0);
        int64_t output_padding_h = parse_int64(attrs, "output_padding_h", 0);
        int64_t output_padding_w = parse_int64(attrs, "output_padding_w", 0);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::conv_transpose2d_forward_kernel(
            inputs[0], inputs[1], bias, stride_h, stride_w, padding_h, padding_w,
            output_padding_h, output_padding_w, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::DepthwiseConv2d, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t stride_h = parse_int64(attrs, "stride_h", 1);
        int64_t stride_w = parse_int64(attrs, "stride_w", 1);
        int64_t padding_h = parse_int64(attrs, "padding_h", 0);
        int64_t padding_w = parse_int64(attrs, "padding_w", 0);
        int64_t dilation_h = parse_int64(attrs, "dilation_h", 1);
        int64_t dilation_w = parse_int64(attrs, "dilation_w", 1);
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::depthwise_conv2d_kernel(
            inputs[0], inputs[1], bias, stride_h, stride_w, padding_h, padding_w,
            dilation_h, dilation_w, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Indexing Operations
    // ========================================================================
    table.register_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        return std::vector<Tensor>{rocm::gather_hip(inputs[0], dim, inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        return std::vector<Tensor>{rocm::scatter_hip(inputs[0], dim, inputs[1], inputs[2], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        return std::vector<Tensor>{rocm::index_select_hip(inputs[0], dim, inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double value = static_cast<double>(parse_float(attrs, "value", 0.0f));
        return std::vector<Tensor>{rocm::masked_fill_hip(inputs[0], inputs[1], value, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::masked_select_hip(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::where_hip(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Slice, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        int64_t start = parse_int64(attrs, "start", 0);
        int64_t end = parse_int64(attrs, "end", std::numeric_limits<int64_t>::max());
        int64_t step = parse_int64(attrs, "step", 1);
        return std::vector<Tensor>{rocm::slice_hip(inputs[0], dim, start, end, step, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return std::vector<Tensor>{rocm::cat_hip(tensors, dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Take, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::take_hip(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Put, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        bool accumulate = parse_bool(attrs, "accumulate", false);
        Tensor input = inputs[0];
        return std::vector<Tensor>{rocm::put_hip(input, inputs[1], inputs[2], accumulate, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Comparison Operations
    // ========================================================================
    table.register_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::eq_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::ne_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::lt_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::le_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::gt_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{rocm::ge_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    // ========================================================================
    // RNN Operations (LSTM/GRU)
    // ========================================================================
    table.register_kernel(OpId::LSTMCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, hx, cx, weight_ih, weight_hh, bias_ih, bias_hh
        auto [hy, cy] = rocm::lstm_cell_forward_kernel(inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], inputs[6], get_hip_stream(attrs));
        return std::vector<Tensor>{hy, cy};
    });

    table.register_kernel(OpId::LSTMCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_hy, grad_cy, input, hx, cx, hy, cy, weight_ih, weight_hh
        auto result = rocm::lstm_cell_backward_kernel(inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], inputs[6], inputs[7], inputs[8], get_hip_stream(attrs));
        return std::vector<Tensor>{std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result), std::get<4>(result), std::get<5>(result), std::get<6>(result)};
    });

    table.register_kernel(OpId::GRUCellForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, hx, weight_ih, weight_hh, bias_ih, bias_hh
        return std::vector<Tensor>{rocm::gru_cell_forward_kernel(inputs[0], inputs[1],
            inputs[2], inputs[3], inputs[4], inputs[5], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::GRUCellBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_hy, input, hx, hy, weight_ih, weight_hh
        auto result = rocm::gru_cell_backward_kernel(inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], get_hip_stream(attrs));
        return std::vector<Tensor>{std::get<0>(result), std::get<1>(result), std::get<2>(result),
            std::get<3>(result), std::get<4>(result), std::get<5>(result)};
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
        int64_t chunks = parse_int64(attrs, "chunks", 1);
        int64_t dim = parse_int64(attrs, "dim", 0);
        return rocm::chunk_kernel(inputs[0], chunks, dim, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::Flatten, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t start_dim = parse_int64(attrs, "start_dim", 0);
        int64_t end_dim = parse_int64(attrs, "end_dim", -1);
        return std::vector<Tensor>{rocm::flatten_kernel(inputs[0], start_dim, end_dim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto repeats = parse_int64_vector(attrs, "repeats");
        return std::vector<Tensor>{rocm::repeat_kernel(inputs[0], repeats, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Tile, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto reps = parse_int64_vector(attrs, "reps");
        return std::vector<Tensor>{rocm::tile_kernel(inputs[0], reps, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Stack, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", 0);
        std::vector<Tensor> tensors(inputs.begin(), inputs.end());
        return rocm::stack_kernel(tensors, dim, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::Split, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t split_size = parse_int64(attrs, "split_size", 1);
        int64_t dim = parse_int64(attrs, "dim", 0);
        return rocm::split_kernel(inputs[0], split_size, dim, get_hip_stream(attrs));
    });

    table.register_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto new_shape = parse_int64_vector(attrs, "shape");
        return std::vector<Tensor>{rocm::expand_kernel(inputs[0], new_shape, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Creation Operations
    // ========================================================================
    table.register_kernel(OpId::Arange, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = parse_double(attrs, "start", 0.0);
        double end = parse_double(attrs, "end", 1.0);
        double step = parse_double(attrs, "step", 1.0);
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        Device device = parse_device(attrs, "device", Device::rocm(0));
        return std::vector<Tensor>{rocm::arange_kernel(start, end, step, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Linspace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double start = parse_double(attrs, "start", 0.0);
        double end = parse_double(attrs, "end", 1.0);
        int64_t steps = parse_int64(attrs, "steps", 100);
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        Device device = parse_device(attrs, "device", Device::rocm(0));
        return std::vector<Tensor>{rocm::linspace_kernel(start, end, steps, dtype, device, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Eye, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t n = parse_int64(attrs, "n", 1);
        int64_t m = parse_int64(attrs, "m", -1);
        int64_t k = parse_int64(attrs, "k", 0);
        DType dtype = parse_dtype(attrs, "dtype", DType::Float32);
        Device device = parse_device(attrs, "device", Device::rocm(0));
        return std::vector<Tensor>{rocm::eye_kernel(n, m, k, dtype, device, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Normalization Operations
    // ========================================================================
    table.register_kernel(OpId::LayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto normalized_shape = parse_int64_vector(attrs, "normalized_shape");
        float eps = parse_float(attrs, "eps", 1e-5f);
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
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
        int64_t num_groups = parse_int64(attrs, "num_groups", 1);
        float eps = parse_float(attrs, "eps", 1e-5f);
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::group_norm_kernel(inputs[0], num_groups, weight, bias, eps, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::GroupNormBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t num_groups = parse_int64(attrs, "num_groups", 1);
        const Tensor* weight = inputs.size() > 4 ? &inputs[4] : nullptr;
        auto [grad_input, grad_weight, grad_bias] = rocm::group_norm_backward_kernel(
            inputs[0], inputs[1], inputs[2], inputs[3], num_groups, weight, get_hip_stream(attrs));
        return std::vector<Tensor>{grad_input, grad_weight, grad_bias};
    });

    table.register_kernel(OpId::InstanceNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float eps = parse_float(attrs, "eps", 1e-5f);
        const Tensor* weight = inputs.size() > 1 ? &inputs[1] : nullptr;
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
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
    table.register_kernel(OpId::FusedLinearReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, weight, [bias]
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::fused_linear_relu_hip(inputs[0], inputs[1], bias)};
    });

    table.register_kernel(OpId::FusedBatchNormReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, running_mean, running_var, weight, bias
        float eps = parse_float(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{rocm::fused_batchnorm_relu_hip(inputs[0], inputs[1], inputs[2],
                                                                  inputs[3], inputs[4], eps)};
    });

    table.register_kernel(OpId::FusedSoftmaxCrossEntropy, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: logits, targets
        std::string reduction = parse_string(attrs, "reduction", "mean");
        return std::vector<Tensor>{rocm::fused_softmax_cross_entropy_hip(inputs[0], inputs[1], reduction)};
    });

    table.register_kernel(OpId::FusedAddReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: a, b
        return std::vector<Tensor>{rocm::fused_add_relu_hip(inputs[0], inputs[1])};
    });

    table.register_kernel(OpId::FusedGelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input
        return std::vector<Tensor>{rocm::fused_gelu_hip(inputs[0])};
    });

    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, weight, bias
        auto normalized_shape = parse_int64_vector(attrs, "normalized_shape");
        float eps = parse_float(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{rocm::fused_layer_norm_hip(inputs[0], normalized_shape,
                                                              inputs[1], inputs[2], eps)};
    });

    table.register_kernel(OpId::FusedConv2dReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // This is implemented as conv output -> batchnorm -> relu
        // inputs: conv_output, running_mean, running_var, weight, bias
        float eps = parse_float(attrs, "eps", 1e-5f);
        return std::vector<Tensor>{rocm::fused_conv_batchnorm_relu_hip(inputs[0], inputs[1], inputs[2],
                                                                        inputs[3], inputs[4], eps)};
    });

    // ========================================================================
    // Embedding Operations
    // ========================================================================
    table.register_kernel(OpId::Embedding, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: weight, indices
        return std::vector<Tensor>{rocm::embedding_kernel(inputs[0], inputs[1], get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::EmbeddingBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_output, indices
        int64_t num_embeddings = parse_int64(attrs, "num_embeddings", 0);
        return std::vector<Tensor>{rocm::embedding_backward_kernel(inputs[0], inputs[1], num_embeddings, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Linear Operations
    // ========================================================================
    table.register_kernel(OpId::Linear, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input, weight, [bias]
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return std::vector<Tensor>{rocm::linear_kernel(inputs[0], inputs[1], bias, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::LinearBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_output, input, weight
        return rocm::linear_backward_kernel(inputs[0], inputs[1], inputs[2], get_hip_stream(attrs));
    });

    // ========================================================================
    // Dropout Operations
    // ========================================================================
    table.register_kernel(OpId::Dropout, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input
        float p = parse_float(attrs, "p", 0.5f);
        bool training = parse_bool(attrs, "training", true);
        auto [output, mask] = rocm::dropout_kernel(inputs[0], p, training, get_hip_stream(attrs));
        return std::vector<Tensor>{output, mask};
    });

    table.register_kernel(OpId::DropoutBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: grad_output, mask
        float p = parse_float(attrs, "p", 0.5f);
        return std::vector<Tensor>{rocm::dropout_backward_kernel(inputs[0], inputs[1], p, get_hip_stream(attrs))};
    });

    // ========================================================================
    // Vision Operations
    // ========================================================================
    table.register_kernel(OpId::Unfold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input
        int64_t kernel_size = parse_int64(attrs, "kernel_size", 3);
        int64_t stride = parse_int64(attrs, "stride", 1);
        int64_t padding = parse_int64(attrs, "padding", 0);
        int64_t dilation = parse_int64(attrs, "dilation", 1);
        return std::vector<Tensor>{rocm::unfold_kernel(inputs[0], kernel_size, stride, padding, dilation, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Fold, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input
        auto output_size = parse_int64_vector(attrs, "output_size");
        int64_t kernel_size = parse_int64(attrs, "kernel_size", 3);
        int64_t stride = parse_int64(attrs, "stride", 1);
        int64_t padding = parse_int64(attrs, "padding", 0);
        int64_t dilation = parse_int64(attrs, "dilation", 1);
        return std::vector<Tensor>{rocm::fold_kernel(inputs[0], output_size, kernel_size, stride, padding, dilation, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Interpolate, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: input
        auto size = parse_int64_vector(attrs, "size");
        std::string mode = parse_string(attrs, "mode", "nearest");
        bool align_corners = parse_bool(attrs, "align_corners", false);
        return std::vector<Tensor>{rocm::interpolate_kernel(inputs[0], size, mode, align_corners, get_hip_stream(attrs))};
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
