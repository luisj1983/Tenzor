#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <cstring>
#include <random>
#include <stdexcept>
#include <tuple>
#include <vector>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/rng.hpp>
#endif

// Forward declarations for kernels in other files
namespace tenzor {
namespace oneapi {
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
    auto cast_kernel(const Tensor& input, DType target_dtype, sycl::queue& queue) -> Tensor;
    auto fused_layer_norm_kernel(const Tensor& input, const Tensor& weight, const Tensor& bias,
                                 const std::vector<int64_t>& normalized_shape, float epsilon,
                                 sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor>;
}
}

namespace tenzor {
namespace oneapi {

// Kernel class declarations for SYCL named kernels
class ReLUKernelFloat32;
class ReLUKernelFloat64;
class ReLUKernelFloat16;
class ReLUBackwardKernelFloat32;
class ReLUBackwardKernelFloat64;
class ReLUBackwardKernelFloat16;
class SigmoidKernelFloat32;
class SigmoidKernelFloat64;
class SigmoidKernelFloat16;
class SigmoidBackwardKernelFloat32;
class SigmoidBackwardKernelFloat64;
class SigmoidBackwardKernelFloat16;
class TanhKernelFloat32;
class TanhKernelFloat64;
class TanhKernelFloat16;
class TanhBackwardKernelFloat32;
class TanhBackwardKernelFloat64;
class TanhBackwardKernelFloat16;
class GeLUKernelFloat32;
class GeLUKernelFloat64;
class GeLUKernelFloat16;
class GeLUBackwardKernelFloat32;
class GeLUBackwardKernelFloat64;
class GeLUBackwardKernelFloat16;
class SoftmaxKernelFloat32;
class SoftmaxKernelFloat64;
class SoftmaxKernelFloat16;
class SoftmaxBackwardKernelFloat32;
class SoftmaxBackwardKernelFloat64;
class SoftmaxBackwardKernelFloat16;
class LeakyReLUKernelFloat32;
class LeakyReLUKernelFloat64;
class LeakyReLUKernelFloat16;
class LeakyReLUBackwardKernelFloat32;
class LeakyReLUBackwardKernelFloat64;
class LeakyReLUBackwardKernelFloat16;
class SwishKernelFloat32;
class SwishKernelFloat64;
class SwishKernelFloat16;
class SwishBackwardKernelFloat32;
class SwishBackwardKernelFloat64;
class SwishBackwardKernelFloat16;
class LogSoftmaxKernelFloat64;
class LogSoftmaxKernelFloat16;
class LogSoftmaxBackwardKernelFloat64;
class LogSoftmaxBackwardKernelFloat16;
class ReLUKernelBFloat16;
class ReLUBackwardKernelBFloat16;
class SigmoidKernelBFloat16;
class SigmoidBackwardKernelBFloat16;
class TanhKernelBFloat16;
class TanhBackwardKernelBFloat16;
class GeLUKernelBFloat16;
class GeLUBackwardKernelBFloat16;
class SoftmaxKernelBFloat16;
class SoftmaxBackwardKernelBFloat16;
class LeakyReLUKernelBFloat16;
class LeakyReLUBackwardKernelBFloat16;
class SwishKernelBFloat16;
class SwishBackwardKernelBFloat16;
class LogSoftmaxKernelBFloat16;
class LogSoftmaxBackwardKernelBFloat16;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// BFloat16 <-> Float32 conversion helpers (device-compatible)
inline float bf16_to_f32(uint16_t bf16) {
    uint32_t bits = static_cast<uint32_t>(bf16) << 16;
    float result;
    __builtin_memcpy(&result, &bits, sizeof(float));
    return result;
}

inline uint16_t f32_to_bf16(float f32) {
    uint32_t bits;
    __builtin_memcpy(&bits, &f32, sizeof(uint32_t));
    // Round to nearest even (banker's rounding) for BFloat16
    uint32_t lsb = (bits >> 16) & 1;
    uint32_t rounding_bias = 0x7FFF + lsb;
    bits += rounding_bias;
    return static_cast<uint16_t>(bits >> 16);
}

// ReLU activation forward
// IMPORTANT: Must ensure contiguous input for direct memory access
auto relu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());

    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ReLUKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(0.0f, in_ptr[idx]);
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ReLUKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(0.0, in_ptr[idx]);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ReLUKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::fmax(0.0f, val));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<ReLUKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(sycl::fmax(0.0f, val));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for relu");
    }

    return output;
}

// ReLU backward
// IMPORTANT: Must ensure contiguous inputs for direct memory access
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor grad_out_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    Tensor grad_input(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                      in_cont.dtype(), in_cont.device());

    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_out_cont);
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<ReLUBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0f ? grad_out_ptr[idx] : 0.0f;
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_out_cont);
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<ReLUBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0 ? grad_out_ptr[idx] : 0.0;
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_out_cont);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<ReLUBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = static_cast<float>(in_ptr[idx]);
            grad_in_ptr[idx] = in_val > 0.0f ? grad_out_ptr[idx] : sycl::half(0.0f);
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_out_cont);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<ReLUBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = bf16_to_f32(in_ptr[idx]);
            grad_in_ptr[idx] = in_val > 0.0f ? grad_out_ptr[idx] : f32_to_bf16(0.0f);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for relu_backward");
    }

    return grad_input;
}

// Sigmoid activation
auto sigmoid_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());

    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SigmoidKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0f / (1.0f + sycl::exp(-in_ptr[idx]));
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SigmoidKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0 / (1.0 + sycl::exp(-in_ptr[idx]));
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SigmoidKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(1.0f / (1.0f + sycl::exp(-val)));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SigmoidKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(1.0f / (1.0f + sycl::exp(-val)));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for sigmoid");
    }

    return output;
}

// Sigmoid backward
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor out_cont = output.is_contiguous() ? output : contiguous_kernel(output, queue);

    Tensor grad_input(std::vector<int64_t>(out_cont.shape().begin(), out_cont.shape().end()),
                      out_cont.dtype(), out_cont.device());

    const int64_t numel = out_cont.numel();

    if (out_cont.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_cont);
        const float* out_ptr = get_data_ptr<const float>(out_cont);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * out_ptr[idx] * (1.0f - out_ptr[idx]);
        });
    }
    else if (out_cont.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_cont);
        const double* out_ptr = get_data_ptr<const double>(out_cont);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * out_ptr[idx] * (1.0 - out_ptr[idx]);
        });
    }
    else if (out_cont.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(out_cont);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = static_cast<float>(grad_out_ptr[idx]);
            float out_val = static_cast<float>(out_ptr[idx]);
            grad_in_ptr[idx] = sycl::half(g_out * out_val * (1.0f - out_val));
        });
    }
    else if (out_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(out_cont);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = bf16_to_f32(grad_out_ptr[idx]);
            float out_val = bf16_to_f32(out_ptr[idx]);
            grad_in_ptr[idx] = f32_to_bf16(g_out * out_val * (1.0f - out_val));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for sigmoid_backward");
    }

    return grad_input;
}

// Tanh activation
auto tanh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());

    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TanhKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tanh(in_ptr[idx]);
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TanhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tanh(in_ptr[idx]);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<TanhKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::tanh(val));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<TanhKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::tanh(bf16_to_f32(in_ptr[idx])));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for tanh");
    }

    return output;
}

// Tanh backward
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor out_cont = output.is_contiguous() ? output : contiguous_kernel(output, queue);

    Tensor grad_input(std::vector<int64_t>(out_cont.shape().begin(), out_cont.shape().end()),
                      out_cont.dtype(), out_cont.device());

    const int64_t numel = out_cont.numel();

    if (out_cont.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_cont);
        const float* out_ptr = get_data_ptr<const float>(out_cont);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<TanhBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * (1.0f - out_ptr[idx] * out_ptr[idx]);
        });
    }
    else if (out_cont.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_cont);
        const double* out_ptr = get_data_ptr<const double>(out_cont);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<TanhBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * (1.0 - out_ptr[idx] * out_ptr[idx]);
        });
    }
    else if (out_cont.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(out_cont);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<TanhBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = static_cast<float>(grad_out_ptr[idx]);
            float out_val = static_cast<float>(out_ptr[idx]);
            grad_in_ptr[idx] = sycl::half(g_out * (1.0f - out_val * out_val));
        });
    }
    else if (out_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(out_cont);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<TanhBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = bf16_to_f32(grad_out_ptr[idx]);
            float out_val = bf16_to_f32(out_ptr[idx]);
            grad_in_ptr[idx] = f32_to_bf16(g_out * (1.0f - out_val * out_val));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for tanh_backward");
    }

    return grad_input;
}

// GELU activation (Gaussian Error Linear Unit)
auto gelu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<GeLUKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            out_ptr[idx] = 0.5f * x * (1.0f + sycl::tanh(inner));
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        const double sqrt_2_over_pi = 0.7978845608028654;
        const double coeff = 0.044715;

        queue.parallel_for<GeLUKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            out_ptr[idx] = 0.5 * x * (1.0 + sycl::tanh(inner));
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<GeLUKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            out_ptr[idx] = sycl::half(0.5f * x * (1.0f + sycl::tanh(inner)));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<GeLUKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            out_ptr[idx] = f32_to_bf16(0.5f * x * (1.0f + sycl::tanh(inner)));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for gelu");
    }

    return output;
}

// GELU backward
auto gelu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<GeLUBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            float x_sq = x * x;
            float x_cubed = x_sq * x;

            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_inner = sycl::tanh(inner);
            float sech_sq = 1.0f - tanh_inner * tanh_inner;

            float derivative = 0.5f * (1.0f + tanh_inner) +
                             0.5f * x * sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_sq);

            grad_in_ptr[idx] = grad_out_ptr[idx] * derivative;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        const double sqrt_2_over_pi = 0.7978845608028654;
        const double coeff = 0.044715;

        queue.parallel_for<GeLUBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            double x_sq = x * x;
            double x_cubed = x_sq * x;

            double inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            double tanh_inner = sycl::tanh(inner);
            double sech_sq = 1.0 - tanh_inner * tanh_inner;

            double derivative = 0.5 * (1.0 + tanh_inner) +
                              0.5 * x * sech_sq * sqrt_2_over_pi * (1.0 + 3.0 * coeff * x_sq);

            grad_in_ptr[idx] = grad_out_ptr[idx] * derivative;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<GeLUBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float x_sq = x * x;
            float x_cubed = x_sq * x;

            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_inner = sycl::tanh(inner);
            float sech_sq = 1.0f - tanh_inner * tanh_inner;

            float derivative = 0.5f * (1.0f + tanh_inner) +
                             0.5f * x * sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_sq);

            grad_in_ptr[idx] = sycl::half(static_cast<float>(grad_out_ptr[idx]) * derivative);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        const float sqrt_2_over_pi = 0.7978845608f;
        const float coeff = 0.044715f;

        queue.parallel_for<GeLUBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float x_sq = x * x;
            float x_cubed = x_sq * x;

            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            float tanh_inner = sycl::tanh(inner);
            float sech_sq = 1.0f - tanh_inner * tanh_inner;

            float derivative = 0.5f * (1.0f + tanh_inner) +
                             0.5f * x * sech_sq * sqrt_2_over_pi * (1.0f + 3.0f * coeff * x_sq);

            grad_in_ptr[idx] = f32_to_bf16(bf16_to_f32(grad_out_ptr[idx]) * derivative);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for gelu_backward");
    }

    return grad_input;
}

// Softmax activation
auto softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access (use OneAPI kernel to keep data on device)
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  in_cont.dtype(), in_cont.device());

    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        // Use 1D dispatch for better compatibility with various SYCL implementations
        // Total work items = outer_size * inner_size
        const int64_t total_work_items = outer_size * inner_size;
        queue.parallel_for<SoftmaxKernelFloat32>(sycl::range<1>(total_work_items), [=](sycl::id<1> idx) {
            const int64_t work_idx = idx[0];
            const int64_t outer_idx = work_idx / inner_size;
            const int64_t inner_idx = work_idx % inner_size;
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            // Find max for numerical stability
            float max_val = -3.4028235e+38f;  // -FLT_MAX (avoid INFINITY with -ffast-math)
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[base_offset + d * inner_size];
                max_val = sycl::fmax(max_val, val);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = sycl::exp(in_ptr[base_offset + d * inner_size] - max_val);
                out_ptr[base_offset + d * inner_size] = val;
                sum += val;
            }

            // Normalize
            for (int64_t d = 0; d < dim_size; ++d) {
                out_ptr[base_offset + d * inner_size] /= sum;
            }
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        // Use 1D dispatch (matching Float32 path) for consistent SYCL behavior
        const int64_t total_work_items = outer_size * inner_size;
        queue.parallel_for<SoftmaxKernelFloat64>(sycl::range<1>(total_work_items), [=](sycl::id<1> idx) {
            const int64_t work_idx = idx[0];
            const int64_t outer_idx = work_idx / inner_size;
            const int64_t inner_idx = work_idx % inner_size;
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            double max_val = -1.7976931348623157e+308;  // -DBL_MAX (avoid INFINITY with -ffast-math)
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[base_offset + d * inner_size];
                max_val = sycl::fmax(max_val, val);
            }

            double sum = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = sycl::exp(in_ptr[base_offset + d * inner_size] - max_val);
                out_ptr[base_offset + d * inner_size] = val;
                sum += val;
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                out_ptr[base_offset + d * inner_size] /= sum;
            }
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SoftmaxKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            // Use float accumulation for numerical stability
            float max_val = -3.4028235e+38f;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = static_cast<float>(in_ptr[base_offset + d * inner_size]);
                max_val = sycl::fmax(max_val, val);
            }

            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = sycl::exp(static_cast<float>(in_ptr[base_offset + d * inner_size]) - max_val);
                out_ptr[base_offset + d * inner_size] = sycl::half(val);
                sum += val;
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                float normalized = static_cast<float>(out_ptr[base_offset + d * inner_size]) / sum;
                out_ptr[base_offset + d * inner_size] = sycl::half(normalized);
            }
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        const int64_t total_work_items = outer_size * inner_size;
        queue.parallel_for<SoftmaxKernelBFloat16>(sycl::range<1>(total_work_items), [=](sycl::id<1> idx) {
            const int64_t work_idx = idx[0];
            const int64_t outer_idx = work_idx / inner_size;
            const int64_t inner_idx = work_idx % inner_size;
            const int64_t base_offset = outer_idx * dim_size * inner_size + inner_idx;

            float max_val = -3.4028235e+38f;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = bf16_to_f32(in_ptr[base_offset + d * inner_size]);
                max_val = sycl::fmax(max_val, val);
            }

            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = sycl::exp(bf16_to_f32(in_ptr[base_offset + d * inner_size]) - max_val);
                out_ptr[base_offset + d * inner_size] = f32_to_bf16(val);
                sum += val;
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                float normalized = bf16_to_f32(out_ptr[base_offset + d * inner_size]) / sum;
                out_ptr[base_offset + d * inner_size] = f32_to_bf16(normalized);
            }
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for softmax");
    }

    return output;
}

// Softmax backward
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor out_cont = output.is_contiguous() ? output : contiguous_kernel(output, queue);

    auto shape = out_cont.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()),
                      out_cont.dtype(), out_cont.device());

    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (out_cont.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_cont);
        const float* out_ptr = get_data_ptr<const float>(out_cont);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<SoftmaxBackwardKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            // Compute dot product
            float dot = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                dot += grad_out_ptr[offset + d * inner_size] * out_ptr[offset + d * inner_size];
            }

            // Compute gradient
            for (int64_t d = 0; d < dim_size; ++d) {
                grad_in_ptr[offset + d * inner_size] = out_ptr[offset + d * inner_size] *
                    (grad_out_ptr[offset + d * inner_size] - dot);
            }
        });
    }
    else if (out_cont.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_cont);
        const double* out_ptr = get_data_ptr<const double>(out_cont);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        // Use 1D dispatch (matching Float32 path) for consistent SYCL behavior
        const int64_t total_work_items = outer_size * inner_size;
        queue.parallel_for<SoftmaxBackwardKernelFloat64>(sycl::range<1>(total_work_items), [=](sycl::id<1> idx) {
            const int64_t work_idx = idx[0];
            const int64_t outer_idx = work_idx / inner_size;
            const int64_t inner_idx = work_idx % inner_size;
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            double dot = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                dot += grad_out_ptr[offset + d * inner_size] * out_ptr[offset + d * inner_size];
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                grad_in_ptr[offset + d * inner_size] = out_ptr[offset + d * inner_size] *
                    (grad_out_ptr[offset + d * inner_size] - dot);
            }
        });
    }
    else if (out_cont.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(out_cont);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<SoftmaxBackwardKernelFloat16>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            // Use float accumulation for precision
            float dot = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                dot += static_cast<float>(grad_out_ptr[offset + d * inner_size]) *
                       static_cast<float>(out_ptr[offset + d * inner_size]);
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                float out_val = static_cast<float>(out_ptr[offset + d * inner_size]);
                float grad_out_val = static_cast<float>(grad_out_ptr[offset + d * inner_size]);
                grad_in_ptr[offset + d * inner_size] = sycl::half(out_val * (grad_out_val - dot));
            }
        });
    }
    else if (out_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(out_cont);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        const int64_t total_work_items = outer_size * inner_size;
        queue.parallel_for<SoftmaxBackwardKernelBFloat16>(sycl::range<1>(total_work_items), [=](sycl::id<1> idx) {
            const int64_t work_idx = idx[0];
            const int64_t outer_idx = work_idx / inner_size;
            const int64_t inner_idx = work_idx % inner_size;
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            float dot = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                dot += bf16_to_f32(grad_out_ptr[offset + d * inner_size]) *
                       bf16_to_f32(out_ptr[offset + d * inner_size]);
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                float out_val = bf16_to_f32(out_ptr[offset + d * inner_size]);
                float grad_out_val = bf16_to_f32(grad_out_ptr[offset + d * inner_size]);
                grad_in_ptr[offset + d * inner_size] = f32_to_bf16(out_val * (grad_out_val - dot));
            }
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for softmax_backward");
    }

    return grad_input;
}

// Leaky ReLU activation
auto leaky_relu_kernel(const Tensor& input, float alpha, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<LeakyReLUKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = x > 0.0f ? x : alpha * x;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double alpha_d = static_cast<double>(alpha);

        queue.parallel_for<LeakyReLUKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = x > 0.0 ? x : alpha_d * x;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<LeakyReLUKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(x > 0.0f ? x : alpha * x);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<LeakyReLUKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(x > 0.0f ? x : alpha * x);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for leaky_relu");
    }

    return output;
}

// Leaky ReLU backward
auto leaky_relu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, sycl::queue& queue) -> Tensor {
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<LeakyReLUBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0f ? grad_out_ptr[idx] : alpha * grad_out_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);
        const double alpha_d = static_cast<double>(alpha);

        queue.parallel_for<LeakyReLUBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0 ? grad_out_ptr[idx] : alpha_d * grad_out_ptr[idx];
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<LeakyReLUBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = static_cast<float>(in_ptr[idx]);
            float grad_out_val = static_cast<float>(grad_out_ptr[idx]);
            grad_in_ptr[idx] = sycl::half(in_val > 0.0f ? grad_out_val : alpha * grad_out_val);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<LeakyReLUBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = bf16_to_f32(in_ptr[idx]);
            float g_out = bf16_to_f32(grad_out_ptr[idx]);
            grad_in_ptr[idx] = f32_to_bf16(in_val > 0.0f ? g_out : alpha * g_out);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for leaky_relu_backward");
    }

    return grad_input;
}

// LogSoftmax activation - numerically stable version
// log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
auto log_softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    // Ensure input is contiguous for correct memory access
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);

    auto shape = in_cont.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::runtime_error("LogSoftmax dimension out of range");
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), in_cont.dtype(), in_cont.device());

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        // Process each slice along the reduction dimension
        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class LogSoftmaxKernelFloat32>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Find max for numerical stability
                    float max_val = -std::numeric_limits<float>::infinity();
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        max_val = sycl::max(max_val, in_ptr[index]);
                    }

                    // Compute log(sum(exp(x - max)))
                    float sum_exp = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_exp += sycl::exp(in_ptr[index] - max_val);
                    }
                    const float log_sum_exp = sycl::log(sum_exp);

                    // Compute log_softmax = x - max - log_sum_exp
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        out_ptr[index] = in_ptr[index] - max_val - log_sum_exp;
                    }
                }
            );
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LogSoftmaxKernelFloat64>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Find max for numerical stability
                    double max_val = -std::numeric_limits<double>::infinity();
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        max_val = sycl::max(max_val, in_ptr[index]);
                    }

                    // Compute log(sum(exp(x - max)))
                    double sum_exp = 0.0;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_exp += sycl::exp(in_ptr[index] - max_val);
                    }
                    const double log_sum_exp = sycl::log(sum_exp);

                    // Compute log_softmax = x - max - log_sum_exp
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        out_ptr[index] = in_ptr[index] - max_val - log_sum_exp;
                    }
                }
            );
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LogSoftmaxKernelFloat16>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Use float accumulation for numerical stability
                    float max_val = -3.4028235e+38f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        max_val = sycl::max(max_val, static_cast<float>(in_ptr[index]));
                    }

                    float sum_exp = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_exp += sycl::exp(static_cast<float>(in_ptr[index]) - max_val);
                    }
                    const float log_sum_exp = sycl::log(sum_exp);

                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        out_ptr[index] = sycl::half(static_cast<float>(in_ptr[index]) - max_val - log_sum_exp);
                    }
                }
            );
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LogSoftmaxKernelBFloat16>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Use float accumulation for numerical stability
                    float max_val = -3.4028235e+38f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        max_val = sycl::max(max_val, bf16_to_f32(in_ptr[index]));
                    }

                    float sum_exp = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_exp += sycl::exp(bf16_to_f32(in_ptr[index]) - max_val);
                    }
                    const float log_sum_exp = sycl::log(sum_exp);

                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        out_ptr[index] = f32_to_bf16(bf16_to_f32(in_ptr[index]) - max_val - log_sum_exp);
                    }
                }
            );
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for log_softmax");
    }

    return output;
}

// LogSoftmax backward
// grad_input = grad_output - exp(log_softmax) * sum(grad_output)
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor {
    // Ensure inputs are contiguous for correct memory access
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor out_cont = output.is_contiguous() ? output : contiguous_kernel(output, queue);

    auto shape = out_cont.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), out_cont.dtype(), out_cont.device());

    if (out_cont.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_cont);
        const float* out_ptr = get_data_ptr<const float>(out_cont);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<class LogSoftmaxBackwardKernelFloat32>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Compute sum of gradients along dim
                    float sum_grad = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_grad += grad_out_ptr[index];
                    }

                    // Compute gradient: grad_output - exp(log_softmax) * sum_grad
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        grad_in_ptr[index] = grad_out_ptr[index] - sycl::exp(out_ptr[index]) * sum_grad;
                    }
                }
            );
        });
    }
    else if (out_cont.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_cont);
        const double* out_ptr = get_data_ptr<const double>(out_cont);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LogSoftmaxBackwardKernelFloat64>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Compute sum of gradients along dim
                    double sum_grad = 0.0;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_grad += grad_out_ptr[index];
                    }

                    // Compute gradient: grad_output - exp(log_softmax) * sum_grad
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        grad_in_ptr[index] = grad_out_ptr[index] - sycl::exp(out_ptr[index]) * sum_grad;
                    }
                }
            );
        });
    }
    else if (out_cont.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(out_cont);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LogSoftmaxBackwardKernelFloat16>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Use float accumulation for precision
                    float sum_grad = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_grad += static_cast<float>(grad_out_ptr[index]);
                    }

                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        float grad_out_val = static_cast<float>(grad_out_ptr[index]);
                        float out_val = static_cast<float>(out_ptr[index]);
                        grad_in_ptr[index] = sycl::half(grad_out_val - sycl::exp(out_val) * sum_grad);
                    }
                }
            );
        });
    }
    else if (out_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(out_cont);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        // Calculate dimensions
        int64_t outer_size = 1;
        for (int64_t i = 0; i < dim; ++i) {
            outer_size *= shape[i];
        }
        int64_t dim_size = shape[dim];
        int64_t inner_size = 1;
        for (int64_t i = dim + 1; i < static_cast<int64_t>(shape.size()); ++i) {
            inner_size *= shape[i];
        }

        queue.submit([&](sycl::handler& cgh) {
            cgh.parallel_for<LogSoftmaxBackwardKernelBFloat16>(
                sycl::range<2>(outer_size, inner_size),
                [=](sycl::id<2> idx) {
                    const int64_t i = idx[0];
                    const int64_t k = idx[1];

                    // Use float accumulation for precision
                    float sum_grad = 0.0f;
                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        sum_grad += bf16_to_f32(grad_out_ptr[index]);
                    }

                    for (int64_t j = 0; j < dim_size; ++j) {
                        const int64_t index = (i * dim_size + j) * inner_size + k;
                        float grad_out_val = bf16_to_f32(grad_out_ptr[index]);
                        float out_val = bf16_to_f32(out_ptr[index]);
                        grad_in_ptr[index] = f32_to_bf16(grad_out_val - sycl::exp(out_val) * sum_grad);
                    }
                }
            );
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for log_softmax_backward");
    }

    return grad_input;
}

// Swish activation (also known as SiLU)
// swish(x) = x * sigmoid(x) = x / (1 + exp(-x))
auto swish_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SwishKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float x = in_ptr[idx];
            const float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = x * sigmoid;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SwishKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double x = in_ptr[idx];
            const double sigmoid = 1.0 / (1.0 + sycl::exp(-x));
            out_ptr[idx] = x * sigmoid;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SwishKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float x = static_cast<float>(in_ptr[idx]);
            const float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = sycl::half(x * sigmoid);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SwishKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = f32_to_bf16(x * sigmoid);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for swish");
    }

    return output;
}

// Swish backward
// swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
//           = sigmoid(x) * (1 + x * (1 - sigmoid(x)))
auto swish_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<SwishBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float x = in_ptr[idx];
            const float g_out = grad_out_ptr[idx];

            // swish'(x) = sigmoid(x) + x * sigmoid(x) * (1 - sigmoid(x))
            const float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            const float swish_grad = sigmoid + x * sigmoid * (1.0f - sigmoid);

            grad_in_ptr[idx] = g_out * swish_grad;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<SwishBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double x = in_ptr[idx];
            const double g_out = grad_out_ptr[idx];

            const double sigmoid = 1.0 / (1.0 + sycl::exp(-x));
            const double swish_grad = sigmoid + x * sigmoid * (1.0 - sigmoid);

            grad_in_ptr[idx] = g_out * swish_grad;
        });
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<SwishBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float x = static_cast<float>(in_ptr[idx]);
            const float g_out = static_cast<float>(grad_out_ptr[idx]);

            const float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            const float swish_grad = sigmoid + x * sigmoid * (1.0f - sigmoid);

            grad_in_ptr[idx] = sycl::half(g_out * swish_grad);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<SwishBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            float derivative = sigmoid + x * sigmoid * (1.0f - sigmoid);
            grad_in_ptr[idx] = f32_to_bf16(bf16_to_f32(grad_out_ptr[idx]) * derivative);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for swish_backward");
    }

    return grad_input;
}

// ============================================================================
// In-place Activation Operations
// ============================================================================

// Kernel name classes for in-place activations
class ReLUInplaceKernelFloat32 {};
class ReLUInplaceKernelFloat64 {};
class ReLUInplaceKernelFloat16 {};
class SigmoidInplaceKernelFloat32 {};
class SigmoidInplaceKernelFloat64 {};
class SigmoidInplaceKernelFloat16 {};
class TanhInplaceKernelFloat32 {};
class TanhInplaceKernelFloat64 {};
class TanhInplaceKernelFloat16 {};
class LeakyReLUInplaceKernelFloat32 {};
class LeakyReLUInplaceKernelFloat64 {};
class LeakyReLUInplaceKernelFloat16 {};
class GeLUInplaceKernelFloat32 {};
class GeLUInplaceKernelFloat64 {};
class GeLUInplaceKernelFloat16 {};
class ReLUInplaceKernelBFloat16 {};
class SigmoidInplaceKernelBFloat16 {};
class TanhInplaceKernelBFloat16 {};
class LeakyReLUInplaceKernelBFloat16 {};
class GeLUInplaceKernelBFloat16 {};

auto relu_inplace_kernel(Tensor& input, sycl::queue& queue) -> void {
    const int64_t numel = input.numel();
    if (input.dtype() == DType::Float32) {
        float* ptr = get_data_ptr<float>(input);
        queue.parallel_for<ReLUInplaceKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (ptr[idx] < 0.0f) ptr[idx] = 0.0f;
        });
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        queue.parallel_for<ReLUInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (ptr[idx] < 0.0) ptr[idx] = 0.0;
        });
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<ReLUInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (float(ptr[idx]) < 0.0f) ptr[idx] = sycl::half(0.0f);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<ReLUInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (bf16_to_f32(ptr[idx]) < 0.0f) ptr[idx] = f32_to_bf16(0.0f);
        });
    }
    else {
        throw std::runtime_error("relu_inplace: unsupported dtype");
    }
}

auto sigmoid_inplace_kernel(Tensor& input, sycl::queue& queue) -> void {
    const int64_t numel = input.numel();
    if (input.dtype() == DType::Float32) {
        float* ptr = get_data_ptr<float>(input);
        queue.parallel_for<SigmoidInplaceKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            ptr[idx] = 1.0f / (1.0f + sycl::exp(-ptr[idx]));
        });
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        queue.parallel_for<SigmoidInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            ptr[idx] = 1.0 / (1.0 + sycl::exp(-ptr[idx]));
        });
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<SigmoidInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            ptr[idx] = sycl::half(1.0f / (1.0f + sycl::exp(-x)));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<SigmoidInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            ptr[idx] = f32_to_bf16(1.0f / (1.0f + sycl::exp(-x)));
        });
    }
    else {
        throw std::runtime_error("sigmoid_inplace: unsupported dtype");
    }
}

auto tanh_inplace_kernel(Tensor& input, sycl::queue& queue) -> void {
    const int64_t numel = input.numel();
    if (input.dtype() == DType::Float32) {
        float* ptr = get_data_ptr<float>(input);
        queue.parallel_for<TanhInplaceKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            ptr[idx] = sycl::tanh(ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        queue.parallel_for<TanhInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            ptr[idx] = sycl::tanh(ptr[idx]);
        });
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<TanhInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            ptr[idx] = sycl::half(sycl::tanh(x));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<TanhInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            ptr[idx] = f32_to_bf16(sycl::tanh(x));
        });
    }
    else {
        throw std::runtime_error("tanh_inplace: unsupported dtype");
    }
}

auto leaky_relu_inplace_kernel(Tensor& input, float alpha, sycl::queue& queue) -> void {
    const int64_t numel = input.numel();
    if (input.dtype() == DType::Float32) {
        float* ptr = get_data_ptr<float>(input);
        queue.parallel_for<LeakyReLUInplaceKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (ptr[idx] < 0.0f) ptr[idx] *= alpha;
        });
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        const double alpha_d = static_cast<double>(alpha);
        queue.parallel_for<LeakyReLUInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (ptr[idx] < 0.0) ptr[idx] *= alpha_d;
        });
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<LeakyReLUInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            if (x < 0.0f) ptr[idx] = sycl::half(x * alpha);
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<LeakyReLUInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            if (x < 0.0f) ptr[idx] = f32_to_bf16(x * alpha);
        });
    }
    else {
        throw std::runtime_error("leaky_relu_inplace: unsupported dtype");
    }
}

auto gelu_inplace_kernel(Tensor& input, sycl::queue& queue) -> void {
    const int64_t numel = input.numel();
    constexpr float sqrt_2_over_pi = 0.7978845608f;
    constexpr float coeff = 0.044715f;

    if (input.dtype() == DType::Float32) {
        float* ptr = get_data_ptr<float>(input);
        queue.parallel_for<GeLUInplaceKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = ptr[idx];
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            ptr[idx] = 0.5f * x * (1.0f + sycl::tanh(inner));
        });
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        constexpr double sqrt_2_over_pi_d = 0.7978845608028654;
        constexpr double coeff_d = 0.044715;
        queue.parallel_for<GeLUInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = ptr[idx];
            double x_cubed = x * x * x;
            double inner = sqrt_2_over_pi_d * (x + coeff_d * x_cubed);
            ptr[idx] = 0.5 * x * (1.0 + sycl::tanh(inner));
        });
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<GeLUInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            ptr[idx] = sycl::half(0.5f * x * (1.0f + sycl::tanh(inner)));
        });
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<GeLUInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            ptr[idx] = f32_to_bf16(0.5f * x * (1.0f + sycl::tanh(inner)));
        });
    }
    else {
        throw std::runtime_error("gelu_inplace: unsupported dtype");
    }
}

// ============================================================================
// ELU Activation: elu(x) = x if x > 0, alpha * (exp(x) - 1) if x <= 0
// ============================================================================

// Kernel name classes
class EluKernelFloat32;
class EluKernelFloat64;
class EluKernelFloat16;
class EluKernelBFloat16;
class EluBackwardKernelFloat32;
class EluBackwardKernelFloat64;
class EluBackwardKernelFloat16;
class EluBackwardKernelBFloat16;

auto elu_kernel(const Tensor& input, float alpha, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<EluKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = (x > 0.0f) ? x : alpha * (sycl::exp(x) - 1.0f);
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        double alpha_d = static_cast<double>(alpha);
        queue.parallel_for<EluKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = (x > 0.0) ? x : alpha_d * (sycl::exp(x) - 1.0);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<EluKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half((x > 0.0f) ? x : alpha * (sycl::exp(x) - 1.0f));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<EluKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16((x > 0.0f) ? x : alpha * (sycl::exp(x) - 1.0f));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for elu");
    }
    return output;
}

auto elu_backward_kernel(const Tensor& grad_output, const Tensor& input, float alpha, sycl::queue& queue) -> Tensor {
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* grad_ptr = get_data_ptr<const float>(grad_cont);
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<EluBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = grad_ptr[idx] * ((x > 0.0f) ? 1.0f : alpha * sycl::exp(x));
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* grad_ptr = get_data_ptr<const double>(grad_cont);
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        double alpha_d = static_cast<double>(alpha);
        queue.parallel_for<EluBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = grad_ptr[idx] * ((x > 0.0) ? 1.0 : alpha_d * sycl::exp(x));
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* grad_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<EluBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float g = static_cast<float>(grad_ptr[idx]);
            out_ptr[idx] = sycl::half(g * ((x > 0.0f) ? 1.0f : alpha * sycl::exp(x)));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<EluBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float g = bf16_to_f32(grad_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(g * ((x > 0.0f) ? 1.0f : alpha * sycl::exp(x)));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for elu_backward");
    }
    return output;
}

// ============================================================================
// SELU Activation: selu(x) = scale * (x if x > 0 else alpha * (exp(x) - 1))
// ============================================================================

class SeluKernelFloat32;
class SeluKernelFloat64;
class SeluKernelFloat16;
class SeluKernelBFloat16;
class SeluBackwardKernelFloat32;
class SeluBackwardKernelFloat64;
class SeluBackwardKernelFloat16;
class SeluBackwardKernelBFloat16;

auto selu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();
    constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
    constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<SeluKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = SELU_SCALE * ((x > 0.0f) ? x : SELU_ALPHA * (sycl::exp(x) - 1.0f));
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        constexpr double SELU_ALPHA_D = 1.6732632423543772848170429916717;
        constexpr double SELU_SCALE_D = 1.0507009873554804934193349852946;
        queue.parallel_for<SeluKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = SELU_SCALE_D * ((x > 0.0) ? x : SELU_ALPHA_D * (sycl::exp(x) - 1.0));
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SeluKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(SELU_SCALE * ((x > 0.0f) ? x : SELU_ALPHA * (sycl::exp(x) - 1.0f)));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SeluKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(SELU_SCALE * ((x > 0.0f) ? x : SELU_ALPHA * (sycl::exp(x) - 1.0f)));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for selu");
    }
    return output;
}

auto selu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();
    constexpr float SELU_ALPHA = 1.6732632423543772848170429916717f;
    constexpr float SELU_SCALE = 1.0507009873554804934193349852946f;

    if (in_cont.dtype() == DType::Float32) {
        const float* grad_ptr = get_data_ptr<const float>(grad_cont);
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<SeluBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = grad_ptr[idx] * SELU_SCALE * ((x > 0.0f) ? 1.0f : SELU_ALPHA * sycl::exp(x));
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* grad_ptr = get_data_ptr<const double>(grad_cont);
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        constexpr double SELU_ALPHA_D = 1.6732632423543772848170429916717;
        constexpr double SELU_SCALE_D = 1.0507009873554804934193349852946;
        queue.parallel_for<SeluBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = grad_ptr[idx] * SELU_SCALE_D * ((x > 0.0) ? 1.0 : SELU_ALPHA_D * sycl::exp(x));
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* grad_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SeluBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float g = static_cast<float>(grad_ptr[idx]);
            out_ptr[idx] = sycl::half(g * SELU_SCALE * ((x > 0.0f) ? 1.0f : SELU_ALPHA * sycl::exp(x)));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SeluBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float g = bf16_to_f32(grad_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(g * SELU_SCALE * ((x > 0.0f) ? 1.0f : SELU_ALPHA * sycl::exp(x)));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for selu_backward");
    }
    return output;
}

// ============================================================================
// Mish Activation: mish(x) = x * tanh(softplus(x)) = x * tanh(ln(1 + exp(x)))
// ============================================================================

class MishKernelFloat32;
class MishKernelFloat64;
class MishKernelFloat16;
class MishKernelBFloat16;
class MishBackwardKernelFloat32;
class MishBackwardKernelFloat64;
class MishBackwardKernelFloat16;
class MishBackwardKernelBFloat16;

auto mish_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<MishKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            float sp = (x > 20.0f) ? x : ((x < -20.0f) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            out_ptr[idx] = x * sycl::tanh(sp);
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<MishKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            double sp = (x > 20.0) ? x : ((x < -20.0) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            out_ptr[idx] = x * sycl::tanh(sp);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MishKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float sp = (x > 20.0f) ? x : ((x < -20.0f) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            out_ptr[idx] = sycl::half(x * sycl::tanh(sp));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MishKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float sp = (x > 20.0f) ? x : ((x < -20.0f) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            out_ptr[idx] = f32_to_bf16(x * sycl::tanh(sp));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for mish");
    }
    return output;
}

auto mish_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* grad_ptr = get_data_ptr<const float>(grad_cont);
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<MishBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            float sp = (x > 20.0f) ? x : ((x < -20.0f) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            float tanh_sp = sycl::tanh(sp);
            float sigmoid_x = 1.0f / (1.0f + sycl::exp(-x));
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            out_ptr[idx] = grad_ptr[idx] * (tanh_sp + x * sech2 * sigmoid_x);
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* grad_ptr = get_data_ptr<const double>(grad_cont);
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<MishBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            double sp = (x > 20.0) ? x : ((x < -20.0) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            double tanh_sp = sycl::tanh(sp);
            double sigmoid_x = 1.0 / (1.0 + sycl::exp(-x));
            double sech2 = 1.0 - tanh_sp * tanh_sp;
            out_ptr[idx] = grad_ptr[idx] * (tanh_sp + x * sech2 * sigmoid_x);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* grad_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<MishBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            float g = static_cast<float>(grad_ptr[idx]);
            float sp = (x > 20.0f) ? x : ((x < -20.0f) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            float tanh_sp = sycl::tanh(sp);
            float sigmoid_x = 1.0f / (1.0f + sycl::exp(-x));
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            out_ptr[idx] = sycl::half(g * (tanh_sp + x * sech2 * sigmoid_x));
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<MishBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float g = bf16_to_f32(grad_ptr[idx]);
            float sp = (x > 20.0f) ? x : ((x < -20.0f) ? sycl::exp(x) : sycl::log1p(sycl::exp(x)));
            float tanh_sp = sycl::tanh(sp);
            float sigmoid_x = 1.0f / (1.0f + sycl::exp(-x));
            float sech2 = 1.0f - tanh_sp * tanh_sp;
            out_ptr[idx] = f32_to_bf16(g * (tanh_sp + x * sech2 * sigmoid_x));
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for mish_backward");
    }
    return output;
}

// ============================================================================
// Softplus Activation: softplus(x) = ln(1 + exp(beta * x)) / beta
// ============================================================================

class SoftplusKernelFloat32;
class SoftplusKernelFloat64;
class SoftplusKernelFloat16;
class SoftplusKernelBFloat16;
class SoftplusBackwardKernelFloat32;
class SoftplusBackwardKernelFloat64;
class SoftplusBackwardKernelFloat16;
class SoftplusBackwardKernelBFloat16;

auto softplus_kernel(const Tensor& input, float beta, float threshold, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<SoftplusKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx] * beta;
            if (x > threshold) out_ptr[idx] = in_ptr[idx];
            else if (x < -threshold) out_ptr[idx] = sycl::exp(x) / beta;
            else out_ptr[idx] = sycl::log1p(sycl::exp(x)) / beta;
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        double beta_d = static_cast<double>(beta);
        double threshold_d = static_cast<double>(threshold);
        queue.parallel_for<SoftplusKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx] * beta_d;
            if (x > threshold_d) out_ptr[idx] = in_ptr[idx];
            else if (x < -threshold_d) out_ptr[idx] = sycl::exp(x) / beta_d;
            else out_ptr[idx] = sycl::log1p(sycl::exp(x)) / beta_d;
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SoftplusKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            float x = val * beta;
            float result;
            if (x > threshold) result = val;
            else if (x < -threshold) result = sycl::exp(x) / beta;
            else result = sycl::log1p(sycl::exp(x)) / beta;
            out_ptr[idx] = sycl::half(result);
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SoftplusKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = bf16_to_f32(in_ptr[idx]);
            float x = val * beta;
            float result;
            if (x > threshold) result = val;
            else if (x < -threshold) result = sycl::exp(x) / beta;
            else result = sycl::log1p(sycl::exp(x)) / beta;
            out_ptr[idx] = f32_to_bf16(result);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for softplus");
    }
    return output;
}

auto softplus_backward_kernel(const Tensor& grad_output, const Tensor& input, float beta, float threshold, sycl::queue& queue) -> Tensor {
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    const int64_t numel = in_cont.numel();

    if (in_cont.dtype() == DType::Float32) {
        const float* grad_ptr = get_data_ptr<const float>(grad_cont);
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<SoftplusBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx] * beta;
            float sigmoid_x;
            if (x > threshold) sigmoid_x = 1.0f;
            else if (x < -threshold) sigmoid_x = sycl::exp(x);
            else sigmoid_x = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = grad_ptr[idx] * sigmoid_x;
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* grad_ptr = get_data_ptr<const double>(grad_cont);
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        double beta_d = static_cast<double>(beta);
        double threshold_d = static_cast<double>(threshold);
        queue.parallel_for<SoftplusBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx] * beta_d;
            double sigmoid_x;
            if (x > threshold_d) sigmoid_x = 1.0;
            else if (x < -threshold_d) sigmoid_x = sycl::exp(x);
            else sigmoid_x = 1.0 / (1.0 + sycl::exp(-x));
            out_ptr[idx] = grad_ptr[idx] * sigmoid_x;
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* grad_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<SoftplusBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            float x = val * beta;
            float sigmoid_x;
            if (x > threshold) sigmoid_x = 1.0f;
            else if (x < -threshold) sigmoid_x = sycl::exp(x);
            else sigmoid_x = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = sycl::half(static_cast<float>(grad_ptr[idx]) * sigmoid_x);
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<SoftplusBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = bf16_to_f32(in_ptr[idx]);
            float x = val * beta;
            float sigmoid_x;
            if (x > threshold) sigmoid_x = 1.0f;
            else if (x < -threshold) sigmoid_x = sycl::exp(x);
            else sigmoid_x = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(grad_ptr[idx]) * sigmoid_x);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for softplus_backward");
    }
    return output;
}

// ============================================================================
// TanhActivation - same as tanh_kernel but registered under TanhActivation OpId
// ============================================================================

auto tanh_activation_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    return tanh_kernel(input, queue);
}

// ============================================================================
// Linear Forward: output = input @ weight^T + bias
// ============================================================================

class LinearKernelFloat32;
class LinearKernelFloat64;
class LinearKernelFloat16;
class LinearKernelBFloat16;

auto linear_kernel(const Tensor& input, const Tensor& weight, const Tensor* bias, sycl::queue& queue) -> Tensor {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor w_cont = weight.is_contiguous() ? weight : contiguous_kernel(weight, queue);

    auto in_shape = in_cont.shape();
    auto w_shape = w_cont.shape();
    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    std::vector<int64_t> out_shape(in_shape.begin(), in_shape.end() - 1);
    out_shape.push_back(out_features);

    Tensor output(out_shape, in_cont.dtype(), in_cont.device());

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        const float* w_ptr = get_data_ptr<const float>(w_cont);
        float* out_ptr = get_data_ptr<float>(output);
        const float* b_ptr = bias ? get_data_ptr<const float>(*bias) : nullptr;

        queue.parallel_for<LinearKernelFloat32>(
            sycl::range<2>(batch_size, out_features), [=](sycl::id<2> id) {
            int64_t b = id[0];
            int64_t o = id[1];
            float sum = 0.0f;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += in_ptr[b * in_features + i] * w_ptr[o * in_features + i];
            }
            out_ptr[b * out_features + o] = sum + (b_ptr ? b_ptr[o] : 0.0f);
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        const double* w_ptr = get_data_ptr<const double>(w_cont);
        double* out_ptr = get_data_ptr<double>(output);
        const double* b_ptr = bias ? get_data_ptr<const double>(*bias) : nullptr;

        queue.parallel_for<LinearKernelFloat64>(
            sycl::range<2>(batch_size, out_features), [=](sycl::id<2> id) {
            int64_t b = id[0];
            int64_t o = id[1];
            double sum = 0.0;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += in_ptr[b * in_features + i] * w_ptr[o * in_features + i];
            }
            out_ptr[b * out_features + o] = sum + (b_ptr ? b_ptr[o] : 0.0);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        // Compute in float32 for precision
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        const sycl::half* w_ptr = get_data_ptr<const sycl::half>(w_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        const sycl::half* b_ptr = bias ? get_data_ptr<const sycl::half>(*bias) : nullptr;

        queue.parallel_for<LinearKernelFloat16>(
            sycl::range<2>(batch_size, out_features), [=](sycl::id<2> id) {
            int64_t b = id[0];
            int64_t o = id[1];
            float sum = 0.0f;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += static_cast<float>(in_ptr[b * in_features + i]) *
                       static_cast<float>(w_ptr[o * in_features + i]);
            }
            sum += b_ptr ? static_cast<float>(b_ptr[o]) : 0.0f;
            out_ptr[b * out_features + o] = sycl::half(sum);
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        const uint16_t* w_ptr = get_data_ptr<const uint16_t>(w_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        const uint16_t* b_ptr = bias ? get_data_ptr<const uint16_t>(*bias) : nullptr;

        queue.parallel_for<LinearKernelBFloat16>(
            sycl::range<2>(batch_size, out_features), [=](sycl::id<2> id) {
            int64_t b = id[0];
            int64_t o = id[1];
            float sum = 0.0f;
            for (int64_t i = 0; i < in_features; ++i) {
                sum += bf16_to_f32(in_ptr[b * in_features + i]) *
                       bf16_to_f32(w_ptr[o * in_features + i]);
            }
            sum += b_ptr ? bf16_to_f32(b_ptr[o]) : 0.0f;
            out_ptr[b * out_features + o] = f32_to_bf16(sum);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for linear");
    }
    return output;
}

// ============================================================================
// Linear Backward: grad_input, grad_weight, grad_bias
// ============================================================================

class LinearBackwardGradInputFloat32;
class LinearBackwardGradInputFloat64;
class LinearBackwardGradInputFloat16;
class LinearBackwardGradInputBFloat16;
class LinearBackwardGradWeightFloat32;
class LinearBackwardGradWeightFloat64;
class LinearBackwardGradWeightFloat16;
class LinearBackwardGradWeightBFloat16;
class LinearBackwardGradBiasFloat32;
class LinearBackwardGradBiasFloat64;
class LinearBackwardGradBiasFloat16;
class LinearBackwardGradBiasBFloat16;

auto linear_backward_kernel(const Tensor& grad_output, const Tensor& input,
                             const Tensor& weight, sycl::queue& queue) -> std::vector<Tensor> {
    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor w_cont = weight.is_contiguous() ? weight : contiguous_kernel(weight, queue);

    auto in_shape = in_cont.shape();
    auto w_shape = w_cont.shape();
    int64_t out_features = w_shape[0];
    int64_t in_features = w_shape[1];

    int64_t batch_size = 1;
    for (size_t i = 0; i < in_shape.size() - 1; ++i) {
        batch_size *= in_shape[i];
    }

    // For FP16/BF16, upcast to FP32, compute, downcast back
    if (grad_cont.dtype() == DType::Float16) {
        DType orig = grad_cont.dtype();
        Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()), orig, in_cont.device());
        Tensor grad_weight(std::vector<int64_t>(w_shape.begin(), w_shape.end()), orig, weight.device());
        Tensor grad_bias({out_features}, orig, grad_output.device());

        const sycl::half* g_ptr = get_data_ptr<const sycl::half>(grad_cont);
        const sycl::half* i_ptr = get_data_ptr<const sycl::half>(in_cont);
        const sycl::half* w_ptr = get_data_ptr<const sycl::half>(w_cont);
        sycl::half* gi_ptr = get_data_ptr<sycl::half>(grad_input);
        sycl::half* gw_ptr = get_data_ptr<sycl::half>(grad_weight);
        sycl::half* gb_ptr = get_data_ptr<sycl::half>(grad_bias);

        queue.parallel_for<LinearBackwardGradInputFloat16>(
            sycl::range<2>(batch_size, in_features), [=](sycl::id<2> id) {
            int64_t b = id[0], i = id[1];
            float sum = 0.0f;
            for (int64_t o = 0; o < out_features; ++o)
                sum += static_cast<float>(g_ptr[b * out_features + o]) * static_cast<float>(w_ptr[o * in_features + i]);
            gi_ptr[b * in_features + i] = sycl::half(sum);
        });
        queue.parallel_for<LinearBackwardGradWeightFloat16>(
            sycl::range<2>(out_features, in_features), [=](sycl::id<2> id) {
            int64_t o = id[0], i = id[1];
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b)
                sum += static_cast<float>(g_ptr[b * out_features + o]) * static_cast<float>(i_ptr[b * in_features + i]);
            gw_ptr[o * in_features + i] = sycl::half(sum);
        });
        queue.parallel_for<LinearBackwardGradBiasFloat16>(
            sycl::range<1>(out_features), [=](sycl::id<1> o) {
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b)
                sum += static_cast<float>(g_ptr[b * out_features + o]);
            gb_ptr[o] = sycl::half(sum);
        });
        return {grad_input, grad_weight, grad_bias};
    }
    if (grad_cont.dtype() == DType::BFloat16) {
        DType orig = grad_cont.dtype();
        Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()), orig, in_cont.device());
        Tensor grad_weight(std::vector<int64_t>(w_shape.begin(), w_shape.end()), orig, weight.device());
        Tensor grad_bias({out_features}, orig, grad_output.device());

        const uint16_t* g_ptr = get_data_ptr<const uint16_t>(grad_cont);
        const uint16_t* i_ptr = get_data_ptr<const uint16_t>(in_cont);
        const uint16_t* w_ptr = get_data_ptr<const uint16_t>(w_cont);
        uint16_t* gi_ptr = get_data_ptr<uint16_t>(grad_input);
        uint16_t* gw_ptr = get_data_ptr<uint16_t>(grad_weight);
        uint16_t* gb_ptr = get_data_ptr<uint16_t>(grad_bias);

        queue.parallel_for<LinearBackwardGradInputBFloat16>(
            sycl::range<2>(batch_size, in_features), [=](sycl::id<2> id) {
            int64_t b = id[0], i = id[1];
            float sum = 0.0f;
            for (int64_t o = 0; o < out_features; ++o)
                sum += bf16_to_f32(g_ptr[b * out_features + o]) * bf16_to_f32(w_ptr[o * in_features + i]);
            gi_ptr[b * in_features + i] = f32_to_bf16(sum);
        });
        queue.parallel_for<LinearBackwardGradWeightBFloat16>(
            sycl::range<2>(out_features, in_features), [=](sycl::id<2> id) {
            int64_t o = id[0], i = id[1];
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b)
                sum += bf16_to_f32(g_ptr[b * out_features + o]) * bf16_to_f32(i_ptr[b * in_features + i]);
            gw_ptr[o * in_features + i] = f32_to_bf16(sum);
        });
        queue.parallel_for<LinearBackwardGradBiasBFloat16>(
            sycl::range<1>(out_features), [=](sycl::id<1> o) {
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b)
                sum += bf16_to_f32(g_ptr[b * out_features + o]);
            gb_ptr[o] = f32_to_bf16(sum);
        });
        return {grad_input, grad_weight, grad_bias};
    }

    Tensor grad_input(std::vector<int64_t>(in_shape.begin(), in_shape.end()),
                      in_cont.dtype(), in_cont.device());
    Tensor grad_weight(std::vector<int64_t>(w_shape.begin(), w_shape.end()),
                       weight.dtype(), weight.device());
    Tensor grad_bias({out_features}, grad_output.dtype(), grad_output.device());

    if (grad_cont.dtype() == DType::Float32) {
        const float* g_ptr = get_data_ptr<const float>(grad_cont);
        const float* i_ptr = get_data_ptr<const float>(in_cont);
        const float* w_ptr = get_data_ptr<const float>(w_cont);
        float* gi_ptr = get_data_ptr<float>(grad_input);
        float* gw_ptr = get_data_ptr<float>(grad_weight);
        float* gb_ptr = get_data_ptr<float>(grad_bias);

        // grad_input = grad_output @ weight
        queue.parallel_for<LinearBackwardGradInputFloat32>(
            sycl::range<2>(batch_size, in_features), [=](sycl::id<2> id) {
            int64_t b = id[0];
            int64_t i = id[1];
            float sum = 0.0f;
            for (int64_t o = 0; o < out_features; ++o) {
                sum += g_ptr[b * out_features + o] * w_ptr[o * in_features + i];
            }
            gi_ptr[b * in_features + i] = sum;
        });

        // grad_weight = grad_output^T @ input (accumulate)
        queue.parallel_for<LinearBackwardGradWeightFloat32>(
            sycl::range<2>(out_features, in_features), [=](sycl::id<2> id) {
            int64_t o = id[0];
            int64_t i = id[1];
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += g_ptr[b * out_features + o] * i_ptr[b * in_features + i];
            }
            gw_ptr[o * in_features + i] = sum;
        });

        // grad_bias = sum(grad_output, dim=0)
        queue.parallel_for<LinearBackwardGradBiasFloat32>(
            sycl::range<1>(out_features), [=](sycl::id<1> o) {
            float sum = 0.0f;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += g_ptr[b * out_features + o];
            }
            gb_ptr[o] = sum;
        });
    }
    else if (grad_cont.dtype() == DType::Float64) {
        const double* g_ptr = get_data_ptr<const double>(grad_cont);
        const double* i_ptr = get_data_ptr<const double>(in_cont);
        const double* w_ptr = get_data_ptr<const double>(w_cont);
        double* gi_ptr = get_data_ptr<double>(grad_input);
        double* gw_ptr = get_data_ptr<double>(grad_weight);
        double* gb_ptr = get_data_ptr<double>(grad_bias);

        queue.parallel_for<LinearBackwardGradInputFloat64>(
            sycl::range<2>(batch_size, in_features), [=](sycl::id<2> id) {
            int64_t b = id[0];
            int64_t i = id[1];
            double sum = 0.0;
            for (int64_t o = 0; o < out_features; ++o) {
                sum += g_ptr[b * out_features + o] * w_ptr[o * in_features + i];
            }
            gi_ptr[b * in_features + i] = sum;
        });

        queue.parallel_for<LinearBackwardGradWeightFloat64>(
            sycl::range<2>(out_features, in_features), [=](sycl::id<2> id) {
            int64_t o = id[0];
            int64_t i = id[1];
            double sum = 0.0;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += g_ptr[b * out_features + o] * i_ptr[b * in_features + i];
            }
            gw_ptr[o * in_features + i] = sum;
        });

        queue.parallel_for<LinearBackwardGradBiasFloat64>(
            sycl::range<1>(out_features), [=](sycl::id<1> o) {
            double sum = 0.0;
            for (int64_t b = 0; b < batch_size; ++b) {
                sum += g_ptr[b * out_features + o];
            }
            gb_ptr[o] = sum;
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for linear_backward");
    }

    return {grad_input, grad_weight, grad_bias};
}

// ============================================================================
// Dropout Forward: randomly zero elements with probability p, scale by 1/(1-p)
// ============================================================================

class DropoutKernelFloat32;
class DropoutKernelFloat64;
class DropoutKernelFloat16;
class DropoutKernelBFloat16;
class DropoutBackwardKernelFloat32;
class DropoutBackwardKernelFloat64;
class DropoutBackwardKernelFloat16;
class DropoutBackwardKernelBFloat16;
class DropoutThresholdScaleKernel;
class DropoutPhiloxMaskKernel;

auto dropout_kernel(const Tensor& input, float p, bool training, sycl::queue& queue)
    -> std::pair<Tensor, Tensor> {
    if (!training || p == 0.0f) {
        return {input, Tensor()};
    }

    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    Tensor output(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                  in_cont.dtype(), in_cont.device());
    Tensor mask(std::vector<int64_t>(in_cont.shape().begin(), in_cont.shape().end()),
                DType::Float32, in_cont.device());

    const int64_t numel = in_cont.numel();
    float scale = 1.0f / (1.0f - p);
    float* mask_ptr = get_data_ptr<float>(mask);

#ifdef TENZOR_HAS_ONEMKL
    // Generate random values on device using oneMKL Philox RNG
    auto seed = tenzor::get_global_seed();
    // Allocate temporary buffer for uniform random values
    float* rand_buf = sycl::malloc_device<float>(numel, queue);

    ::oneapi::mkl::rng::philox4x32x10 engine(queue, seed);
    ::oneapi::mkl::rng::uniform<float> dist(0.0f, 1.0f);
    ::oneapi::mkl::rng::generate(dist, engine, numel, rand_buf);

    // Fused threshold + scale kernel: rand < p -> 0, else -> scale
    float p_thresh = p;
    queue.parallel_for<DropoutThresholdScaleKernel>(
        sycl::range<1>(numel), [=](sycl::id<1> idx) {
            mask_ptr[idx] = (rand_buf[idx] < p_thresh) ? 0.0f : scale;
        }
    ).wait();

    sycl::free(rand_buf, queue);
#else
    // Device-side Philox 4x32-10 RNG fallback (no oneMKL dependency)
    auto seed_val = static_cast<uint32_t>(tenzor::get_global_seed());
    float p_thresh = p;

    queue.parallel_for<DropoutPhiloxMaskKernel>(
        sycl::range<1>(numel), [=](sycl::id<1> idx) {
            uint32_t counter = static_cast<uint32_t>(idx[0]);
            uint32_t key = seed_val;

            // Philox 4x32-10 counter-based RNG
            uint32_t c0 = counter;
            uint32_t c1 = ~counter;
            uint32_t k0 = key;
            uint32_t k1 = ~key;

            constexpr uint32_t M0 = 0xD2511F53u;
            constexpr uint32_t M1 = 0xCD9E8D57u;
            constexpr uint32_t BUMP = 0x9E3779B9u;
            constexpr uint32_t BUMP1 = 0xBB67AE85u;

            for (int round = 0; round < 10; ++round) {
                uint32_t hi0 = static_cast<uint32_t>((static_cast<uint64_t>(c0) * M0) >> 32);
                uint32_t hi1 = static_cast<uint32_t>((static_cast<uint64_t>(c1) * M1) >> 32);
                uint32_t lo1 = c1 * M1;

                c0 = hi1 ^ k0 ^ c0;
                c1 = lo1;
                c1 = hi0 ^ c1 ^ k1;
                k0 += BUMP;
                k1 += BUMP1;
            }

            // Convert to uniform [0,1)
            constexpr float INV = 2.3283064365386963e-10f; // 1/2^32
            float u = (static_cast<float>(c0) + 0.5f) * INV;
            mask_ptr[idx] = (u < p_thresh) ? 0.0f : scale;
        }
    ).wait();
#endif

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<DropoutKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[idx] * mask_ptr[idx];
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<DropoutKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = in_ptr[idx] * static_cast<double>(mask_ptr[idx]);
        });
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<DropoutKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(in_ptr[idx]) * mask_ptr[idx]);
        });
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<DropoutKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(in_ptr[idx]) * mask_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for dropout");
    }

    return {output, mask};
}

auto dropout_backward_kernel(const Tensor& grad_output, const Tensor& mask, float p, sycl::queue& queue) -> Tensor {
    if (!mask.impl() || p == 0.0f) {
        return grad_output;
    }

    Tensor grad_cont = grad_output.is_contiguous() ? grad_output : contiguous_kernel(grad_output, queue);
    Tensor output(std::vector<int64_t>(grad_cont.shape().begin(), grad_cont.shape().end()),
                  grad_cont.dtype(), grad_cont.device());
    const int64_t numel = grad_cont.numel();
    const float* mask_ptr = get_data_ptr<const float>(mask);

    if (grad_cont.dtype() == DType::Float32) {
        const float* grad_ptr = get_data_ptr<const float>(grad_cont);
        float* out_ptr = get_data_ptr<float>(output);
        queue.parallel_for<DropoutBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = grad_ptr[idx] * mask_ptr[idx];
        });
    }
    else if (grad_cont.dtype() == DType::Float64) {
        const double* grad_ptr = get_data_ptr<const double>(grad_cont);
        double* out_ptr = get_data_ptr<double>(output);
        queue.parallel_for<DropoutBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = grad_ptr[idx] * static_cast<double>(mask_ptr[idx]);
        });
    }
    else if (grad_cont.dtype() == DType::Float16) {
        const sycl::half* grad_ptr = get_data_ptr<const sycl::half>(grad_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);
        queue.parallel_for<DropoutBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::half(static_cast<float>(grad_ptr[idx]) * mask_ptr[idx]);
        });
    }
    else if (grad_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_ptr = get_data_ptr<const uint16_t>(grad_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);
        queue.parallel_for<DropoutBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(bf16_to_f32(grad_ptr[idx]) * mask_ptr[idx]);
        });
    }
    else {
        throw std::runtime_error("Unsupported dtype for dropout_backward");
    }
    return output;
}

// ============================================================================
// LayerNorm Forward
// ============================================================================

class LayerNormKernelFloat32;
class LayerNormKernelFloat64;

auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                       const Tensor& weight, const Tensor& bias, float eps,
                       sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    // Delegate to fused_layer_norm_kernel which already exists
    return fused_layer_norm_kernel(input, weight, bias, normalized_shape, eps, queue);
}

// ============================================================================
// InstanceNorm Forward
// ============================================================================

class InstanceNormKernelFloat32;
class InstanceNormKernelFloat64;

auto instance_norm_kernel(const Tensor& input, const Tensor& weight,
                           const Tensor& bias, float eps,
                           sycl::queue& queue) -> std::vector<Tensor> {
    Tensor in_cont = input.is_contiguous() ? input : contiguous_kernel(input, queue);
    auto shape = in_cont.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  in_cont.dtype(), in_cont.device());
    Tensor mean_t({N, C}, DType::Float32, in_cont.device());
    Tensor inv_std_t({N, C}, DType::Float32, in_cont.device());

    bool has_weight = weight.impl() != nullptr;
    bool has_bias = bias.impl() != nullptr;

    if (in_cont.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(in_cont);
        float* out_ptr = get_data_ptr<float>(output);
        float* mean_ptr = get_data_ptr<float>(mean_t);
        float* inv_std_ptr = get_data_ptr<float>(inv_std_t);
        const float* w_ptr = has_weight ? get_data_ptr<const float>(weight) : nullptr;
        const float* b_ptr = has_bias ? get_data_ptr<const float>(bias) : nullptr;

        // Each work-item handles one (n, c) pair
        queue.parallel_for<InstanceNormKernelFloat32>(
            sycl::range<2>(N, C), [=](sycl::id<2> id) {
            int64_t n = id[0];
            int64_t c = id[1];
            int64_t base = (n * C + c) * spatial_size;

            // Compute mean
            float sum = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                sum += in_ptr[base + s];
            }
            float m = sum / static_cast<float>(spatial_size);

            // Compute variance
            float var = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                float diff = in_ptr[base + s] - m;
                var += diff * diff;
            }
            var /= static_cast<float>(spatial_size);
            float istd = 1.0f / sycl::sqrt(var + eps);

            mean_ptr[n * C + c] = m;
            inv_std_ptr[n * C + c] = istd;

            float w = w_ptr ? w_ptr[c] : 1.0f;
            float b = b_ptr ? b_ptr[c] : 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                out_ptr[base + s] = (in_ptr[base + s] - m) * istd * w + b;
            }
        });
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);
        float* mean_ptr = get_data_ptr<float>(mean_t);
        float* inv_std_ptr = get_data_ptr<float>(inv_std_t);
        const double* w_ptr = has_weight ? get_data_ptr<const double>(weight) : nullptr;
        const double* b_ptr = has_bias ? get_data_ptr<const double>(bias) : nullptr;

        queue.parallel_for<InstanceNormKernelFloat64>(
            sycl::range<2>(N, C), [=](sycl::id<2> id) {
            int64_t n = id[0];
            int64_t c = id[1];
            int64_t base = (n * C + c) * spatial_size;

            double sum = 0.0;
            for (int64_t s = 0; s < spatial_size; ++s) {
                sum += in_ptr[base + s];
            }
            double m = sum / static_cast<double>(spatial_size);

            double var = 0.0;
            for (int64_t s = 0; s < spatial_size; ++s) {
                double diff = in_ptr[base + s] - m;
                var += diff * diff;
            }
            var /= static_cast<double>(spatial_size);
            double istd = 1.0 / sycl::sqrt(var + static_cast<double>(eps));

            mean_ptr[n * C + c] = static_cast<float>(m);
            inv_std_ptr[n * C + c] = static_cast<float>(istd);

            double w = w_ptr ? w_ptr[c] : 1.0;
            double b = b_ptr ? b_ptr[c] : 0.0;
            for (int64_t s = 0; s < spatial_size; ++s) {
                out_ptr[base + s] = (in_ptr[base + s] - m) * istd * w + b;
            }
        });
    }
    else {
        // Float16/BFloat16: upcast to Float32, compute, downcast
        DType orig_dtype = in_cont.dtype();
        Tensor in_f32 = cast_kernel(in_cont, DType::Float32, queue);
        Tensor w_f32 = (has_weight && weight.dtype() != DType::Float32)
            ? cast_kernel(weight, DType::Float32, queue) : weight;
        Tensor b_f32 = (has_bias && bias.dtype() != DType::Float32)
            ? cast_kernel(bias, DType::Float32, queue) : bias;
        auto results = instance_norm_kernel(in_f32, w_f32, b_f32, eps, queue);
        results[0] = cast_kernel(results[0], orig_dtype, queue);
        return results;
    }

    return {output, mean_t, inv_std_t};
}

// ============================================================================
// InstanceNorm Backward
// ============================================================================

class InstanceNormBackwardKernelFloat32;
class InstanceNormBackwardKernelFloat64;

auto instance_norm_backward_kernel(const Tensor& grad_output, const Tensor& input,
                                    const Tensor& mean, const Tensor& rstd,
                                    const Tensor& weight,
                                    sycl::queue& queue) -> std::vector<Tensor> {
    auto shape = input.shape();
    int64_t N = shape[0];
    int64_t C = shape[1];
    int64_t spatial_size = 1;
    for (size_t i = 2; i < shape.size(); ++i) {
        spatial_size *= shape[i];
    }

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()),
                      input.dtype(), input.device());
    // grad_weight and grad_bias need atomic reduction across N
    // Allocate on host, compute, copy back
    Tensor grad_weight({C}, weight.dtype(), weight.device());
    Tensor grad_bias({C}, weight.dtype(), weight.device());

    bool has_weight = weight.impl() != nullptr;
    const float* mean_ptr = get_data_ptr<const float>(mean);
    const float* rstd_ptr = get_data_ptr<const float>(rstd);

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        const float* g_ptr = get_data_ptr<const float>(grad_output);
        const float* w_ptr = has_weight ? get_data_ptr<const float>(weight) : nullptr;
        float* gi_ptr = get_data_ptr<float>(grad_input);

        // Compute grad_input on device
        queue.parallel_for<InstanceNormBackwardKernelFloat32>(
            sycl::range<2>(N, C), [=](sycl::id<2> id) {
            int64_t n = id[0];
            int64_t c = id[1];
            int64_t base = (n * C + c) * spatial_size;
            float m = mean_ptr[n * C + c];
            float r = rstd_ptr[n * C + c];
            float w = w_ptr ? w_ptr[c] : 1.0f;

            float ds = 0.0f, db = 0.0f;
            for (int64_t s = 0; s < spatial_size; ++s) {
                float dy = g_ptr[base + s];
                float x_hat = (in_ptr[base + s] - m) * r;
                ds += dy * w * x_hat;
                db += dy * w;
            }

            float inv_ss = 1.0f / static_cast<float>(spatial_size);
            for (int64_t s = 0; s < spatial_size; ++s) {
                float dy = g_ptr[base + s];
                float x_hat = (in_ptr[base + s] - m) * r;
                gi_ptr[base + s] = r * (dy * w - inv_ss * (db + x_hat * ds));
            }
        });

        // Compute grad_weight and grad_bias on device (reduction across N per channel)
        float* gw_ptr = get_data_ptr<float>(grad_weight);
        float* gb_ptr = get_data_ptr<float>(grad_bias);

        queue.parallel_for<class InstanceNormBackwardGradWBFloat32>(
            sycl::range<1>(C), [=](sycl::id<1> id) {
            int64_t c = id[0];
            float gw_sum = 0.0f;
            float gb_sum = 0.0f;
            for (int64_t n = 0; n < N; ++n) {
                int64_t base = (n * C + c) * spatial_size;
                float m = mean_ptr[n * C + c];
                float r = rstd_ptr[n * C + c];
                for (int64_t s = 0; s < spatial_size; ++s) {
                    float x_hat = (in_ptr[base + s] - m) * r;
                    gw_sum += g_ptr[base + s] * x_hat;
                    gb_sum += g_ptr[base + s];
                }
            }
            gw_ptr[c] = gw_sum;
            gb_ptr[c] = gb_sum;
        });
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        const double* g_ptr = get_data_ptr<const double>(grad_output);
        const double* w_ptr = has_weight ? get_data_ptr<const double>(weight) : nullptr;
        double* gi_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<InstanceNormBackwardKernelFloat64>(
            sycl::range<2>(N, C), [=](sycl::id<2> id) {
            int64_t n = id[0];
            int64_t c = id[1];
            int64_t base = (n * C + c) * spatial_size;
            float mf = mean_ptr[n * C + c];
            float rf = rstd_ptr[n * C + c];
            double m = static_cast<double>(mf);
            double r = static_cast<double>(rf);
            double w = w_ptr ? w_ptr[c] : 1.0;

            double ds = 0.0, db = 0.0;
            for (int64_t s = 0; s < spatial_size; ++s) {
                double dy = g_ptr[base + s];
                double x_hat = (in_ptr[base + s] - m) * r;
                ds += dy * w * x_hat;
                db += dy * w;
            }

            double inv_ss = 1.0 / static_cast<double>(spatial_size);
            for (int64_t s = 0; s < spatial_size; ++s) {
                double dy = g_ptr[base + s];
                double x_hat = (in_ptr[base + s] - m) * r;
                gi_ptr[base + s] = r * (dy * w - inv_ss * (db + x_hat * ds));
            }
        });

        // Compute grad_weight and grad_bias on device (reduction across N per channel)
        double* gw_ptr = get_data_ptr<double>(grad_weight);
        double* gb_ptr = get_data_ptr<double>(grad_bias);

        queue.parallel_for<class InstanceNormBackwardGradWBFloat64>(
            sycl::range<1>(C), [=](sycl::id<1> id) {
            int64_t c = id[0];
            double gw_sum = 0.0;
            double gb_sum = 0.0;
            for (int64_t n = 0; n < N; ++n) {
                int64_t base = (n * C + c) * spatial_size;
                double m = static_cast<double>(mean_ptr[n * C + c]);
                double r = static_cast<double>(rstd_ptr[n * C + c]);
                for (int64_t s = 0; s < spatial_size; ++s) {
                    double x_hat = (in_ptr[base + s] - m) * r;
                    gw_sum += g_ptr[base + s] * x_hat;
                    gb_sum += g_ptr[base + s];
                }
            }
            gw_ptr[c] = gw_sum;
            gb_ptr[c] = gb_sum;
        });
    }
    else {
        // Float16/BFloat16: upcast to Float32, compute, downcast
        DType orig_dtype = input.dtype();
        Tensor in_f32 = cast_kernel(input, DType::Float32, queue);
        Tensor go_f32 = cast_kernel(grad_output, DType::Float32, queue);
        Tensor w_f32 = (has_weight && weight.dtype() != DType::Float32)
            ? cast_kernel(weight, DType::Float32, queue) : weight;
        auto results = instance_norm_backward_kernel(go_f32, in_f32, mean, rstd, w_f32, queue);
        results[0] = cast_kernel(results[0], orig_dtype, queue);
        return results;
    }

    return {grad_input, grad_weight, grad_bias};
}

} // namespace oneapi
} // namespace tenzor
