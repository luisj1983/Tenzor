/**
 * @file cuda_kernel_registry.cpp
 * @brief CUDA kernel registration for O(1) dispatch
 *
 * Registers all CUDA kernel implementations with the dispatch table.
 * Each kernel is a direct function pointer - no intermediate dispatch.
 */

#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/backend/kernel_registry.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"
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

    // BatchNorm2d operations
    auto batchnorm2d_mean_var(const Tensor& input, Tensor& mean, Tensor& variance, cudaStream_t stream) -> void;
    auto batchnorm2d_forward(const Tensor& input, const Tensor& mean, const Tensor& variance, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_forward_affine(const Tensor& input, const Tensor& mean, const Tensor& variance, const Tensor& gamma, const Tensor& beta, float epsilon, cudaStream_t stream) -> Tensor;
    auto batchnorm2d_update_running_stats(Tensor& running_mean, Tensor& running_var, const Tensor& batch_mean, const Tensor& batch_var, float momentum, cudaStream_t stream) -> void;

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
#endif

    // Fill operations
    auto fill_kernel(const Tensor& tensor, float value, cudaStream_t stream) -> Tensor;

} // namespace cuda

/**
 * @brief Register all CUDA kernels with the dispatch table.
 */
void register_cuda_kernels(BackendDispatchTable& table) {
    // =========================================================================
    // Arithmetic Operations
    // =========================================================================
    table.register_kernel(OpId::Add, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::add_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Sub, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sub_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Mul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::mul_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Div, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::div_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::MatMul, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::matmul_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    // Bmm (batched matrix multiplication) uses the same kernel as MatMul
    // The CUDA matmul kernel already handles batched inputs via cublasSgemmStridedBatched
    table.register_kernel(OpId::Bmm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::matmul_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Dot, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::dot_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });

    // Inplace operations
    table.register_kernel(OpId::AddInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor result = inputs[0];
        return std::vector<Tensor>{cuda::add_inplace_kernel(result, inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SubInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor result = inputs[0];
        return std::vector<Tensor>{cuda::sub_inplace_kernel(result, inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::MulInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor result = inputs[0];
        return std::vector<Tensor>{cuda::mul_inplace_kernel(result, inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::DivInplace, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        Tensor result = inputs[0];
        return std::vector<Tensor>{cuda::div_inplace_kernel(result, inputs[1], get_cuda_stream(attrs))};
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
    // Element-wise Math Operations
    // =========================================================================
    table.register_kernel(OpId::Sqrt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sqrt_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Neg, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::neg_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Abs, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::abs_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Sign, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sign_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Log, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::log_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Exp, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::exp_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Reciprocal, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::reciprocal_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Floor, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::floor_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Ceil, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::ceil_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Round, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::round_kernel(inputs[0], get_cuda_stream(attrs))};
    });
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
    // Trigonometric Operations
    // =========================================================================
    table.register_kernel(OpId::Sin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sin_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Cos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::cos_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Tan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::tan_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Asin, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::asin_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Acos, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::acos_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Atan, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::atan_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Sinh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sinh_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Cosh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::cosh_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Tanh, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::tanh_kernel(inputs[0], get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Comparison Operations
    // =========================================================================
    table.register_kernel(OpId::Eq, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::eq_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Ne, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::ne_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Lt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::lt_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Le, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::le_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Gt, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::gt_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Ge, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::ge_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Activation Functions
    // =========================================================================
    table.register_kernel(OpId::ReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::relu_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::ReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::relu_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Sigmoid, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sigmoid_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SigmoidBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::sigmoid_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::TanhActivation, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::tanh_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::TanhBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::tanh_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Gelu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::gelu_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::GeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::gelu_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Swish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::swish_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SwishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::swish_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LeakyReLU, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        return std::vector<Tensor>{cuda::leaky_relu_kernel(inputs[0], alpha, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LeakyReLUBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 0.01f);
        return std::vector<Tensor>{cuda::leaky_relu_backward_kernel(inputs[0], inputs[1], alpha, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Elu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 1.0f);
        return std::vector<Tensor>{cuda::elu_kernel(inputs[0], alpha, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::EluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float alpha = parse_attr<float>(attrs, "alpha", 1.0f);
        return std::vector<Tensor>{cuda::elu_backward_kernel(inputs[0], inputs[1], alpha, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Selu, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::selu_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SeluBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::selu_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Mish, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::mish_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::MishBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::mish_backward_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Softplus, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float beta = parse_attr<float>(attrs, "beta", 1.0f);
        float threshold = parse_attr<float>(attrs, "threshold", 20.0f);
        return std::vector<Tensor>{cuda::softplus_kernel(inputs[0], beta, threshold, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SoftplusBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float beta = parse_attr<float>(attrs, "beta", 1.0f);
        float threshold = parse_attr<float>(attrs, "threshold", 20.0f);
        return std::vector<Tensor>{cuda::softplus_backward_kernel(inputs[0], inputs[1], beta, threshold, get_cuda_stream(attrs))};
    });

    // Softmax operations (use cuDNN when available for better performance)
#ifdef TENZOR_HAS_CUDNN
    table.register_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::cudnn_softmax_forward(inputs[0], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::cudnn_softmax_backward(inputs[0], inputs[1], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::cudnn_log_softmax_forward(inputs[0], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::cudnn_log_softmax_backward(inputs[0], inputs[1], dim, get_cuda_stream(attrs))};
    });
#else
    table.register_kernel(OpId::Softmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::softmax_kernel(inputs[0], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::SoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::softmax_backward_kernel(inputs[0], inputs[1], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LogSoftmax, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::log_softmax_kernel(inputs[0], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::LogSoftmaxBackward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::log_softmax_backward_kernel(inputs[0], inputs[1], dim, get_cuda_stream(attrs))};
    });
#endif

    // =========================================================================
    // Transform Operations
    // =========================================================================
    table.register_kernel(OpId::Contiguous, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::contiguous_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Clone, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::clone_kernel(inputs[0], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Fill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float value = parse_attr<float>(attrs, "value", 0.0f);
        return std::vector<Tensor>{cuda::fill_kernel(inputs[0], value, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Reshape, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        return std::vector<Tensor>{cuda::reshape_kernel(inputs[0], shape, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Expand, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto shape = parse_int_list(attrs, "shape");
        return std::vector<Tensor>{cuda::expand_kernel(inputs[0], shape, static_cast<void*>(get_cuda_stream(attrs)))};
    });
    table.register_kernel(OpId::Transpose, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim0 = parse_attr<int64_t>(attrs, "dim0", 0);
        int64_t dim1 = parse_attr<int64_t>(attrs, "dim1", 1);
        return std::vector<Tensor>{cuda::transpose_kernel(inputs[0], dim0, dim1, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Permute, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto dims = parse_int_list(attrs, "dims");
        return std::vector<Tensor>{cuda::permute_kernel(inputs[0], dims, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Squeeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", -1);
        return std::vector<Tensor>{cuda::squeeze_kernel(inputs[0], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Unsqueeze, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cuda::unsqueeze_kernel(inputs[0], dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Cat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cuda::cat_kernel(inputs, dim, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Repeat, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        auto repeats = parse_int_list(attrs, "repeats");
        return std::vector<Tensor>{cuda::repeat_kernel(inputs[0], repeats, get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Indexing Operations
    // =========================================================================
    table.register_kernel(OpId::IndexSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cuda::index_select_kernel(inputs[0], dim, inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Gather, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cuda::gather_kernel(inputs[0], dim, inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Scatter, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_attr<int64_t>(attrs, "dim", 0);
        return std::vector<Tensor>{cuda::scatter_kernel(inputs[0], dim, inputs[1], inputs[2], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::MaskedSelect, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::masked_select_kernel(inputs[0], inputs[1], get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::MaskedFill, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        double value = parse_attr<double>(attrs, "value", 0.0);
        return std::vector<Tensor>{cuda::masked_fill_kernel(inputs[0], inputs[1], value, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::Where, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        return std::vector<Tensor>{cuda::where_kernel(inputs[0], inputs[1], inputs[2], get_cuda_stream(attrs))};
    });

    // =========================================================================
    // Pooling Operations (use cuDNN when available for better performance)
    // =========================================================================
#ifdef TENZOR_HAS_CUDNN
    table.register_kernel(OpId::MaxPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        auto [output, indices] = cuda::cudnn_maxpool2d_forward(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs));
        return std::vector<Tensor>{output, indices};
    });
    table.register_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return std::vector<Tensor>{cuda::cudnn_avgpool2d_forward(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs))};
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
    table.register_kernel(OpId::AvgPool2dForward, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t kernel_size = parse_attr<int64_t>(attrs, "kernel_size", 2);
        int64_t stride = parse_attr<int64_t>(attrs, "stride", kernel_size);
        int64_t padding = parse_attr<int64_t>(attrs, "padding", 0);
        return std::vector<Tensor>{cuda::avgpool2d_forward_kernel(inputs[0], kernel_size, stride, padding, get_cuda_stream(attrs))};
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
        return std::vector<Tensor>{cuda::batchnorm2d_forward_affine(inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], epsilon, get_cuda_stream(attrs))};
    });
    table.register_kernel(OpId::BatchNorm2dUpdateRunningStats, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        float momentum = parse_attr<float>(attrs, "momentum", 0.1f);
        Tensor running_mean = inputs[0];
        Tensor running_var = inputs[1];
        cuda::batchnorm2d_update_running_stats(running_mean, running_var, inputs[2], inputs[3], momentum, get_cuda_stream(attrs));
        return std::vector<Tensor>{running_mean, running_var};
    });

    // =========================================================================
    // Fused LayerNorm (single kernel launch for maximum performance)
    // =========================================================================
    table.register_kernel(OpId::FusedLayerNorm, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        // inputs: [input, weight, bias]
        // attrs: normalized_shape (comma-separated), eps
        float eps = parse_attr<float>(attrs, "eps", 1e-5f);
        auto normalized_shape = parse_int_list(attrs, "normalized_shape");
        auto [output, mean, inv_std] = cuda::fused_layer_norm_cuda(
            inputs[0], normalized_shape, inputs[1], inputs[2], eps
        );
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
        // inputs: [Q, K, V] - all (batch_heads, seq_len, head_dim)
        // attrs: scale
        float scale = parse_attr<float>(attrs, "scale", 1.0f);
        auto output = cuda::fused_attention_cuda(inputs[0], inputs[1], inputs[2], scale);
        return std::vector<Tensor>{output};
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
