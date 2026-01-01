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

    // Reduction operations
    auto sum_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto mean_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto max_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;
    auto min_kernel(const Tensor& input, int64_t dim, bool keepdim, hipStream_t stream) -> Tensor;

    // Activation functions
    auto relu_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_kernel(const Tensor& input, hipStream_t stream) -> Tensor;
    auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& input, hipStream_t stream) -> Tensor;
    auto leaky_relu_kernel(const Tensor& input, float alpha, hipStream_t stream) -> Tensor;
    auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, hipStream_t stream) -> Tensor;

    // Softmax operations
    auto softmax_kernel(const Tensor& input, int64_t dim, hipStream_t stream) -> Tensor;
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

    // ========================================================================
    // Reduction Operations
    // ========================================================================
    table.register_kernel(OpId::Sum, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::sum_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Mean, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::mean_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Max, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::max_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
    });

    table.register_kernel(OpId::Min, [](std::span<const Tensor> inputs, const OpAttributes& attrs) {
        int64_t dim = parse_int64(attrs, "dim", -1);
        bool keepdim = parse_bool(attrs, "keepdim", false);
        return std::vector<Tensor>{rocm::min_kernel(inputs[0], dim, keepdim, get_hip_stream(attrs))};
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
