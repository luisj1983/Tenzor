#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>

// Forward declaration for contiguous kernel
namespace tenzor {
namespace oneapi {
    auto contiguous_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;
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
        }).wait();
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ReLUKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(0.0, in_ptr[idx]);
        }).wait();
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<ReLUKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::fmax(0.0f, val));
        }).wait();
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<ReLUKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(sycl::fmax(0.0f, val));
        }).wait();
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
        }).wait();
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_out_cont);
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<ReLUBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0 ? grad_out_ptr[idx] : 0.0;
        }).wait();
    }
    else if (in_cont.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_out_cont);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(in_cont);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<ReLUBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = static_cast<float>(in_ptr[idx]);
            grad_in_ptr[idx] = in_val > 0.0f ? grad_out_ptr[idx] : sycl::half(0.0f);
        }).wait();
    }
    else if (in_cont.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_out_cont);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(in_cont);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<ReLUBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = bf16_to_f32(in_ptr[idx]);
            grad_in_ptr[idx] = in_val > 0.0f ? grad_out_ptr[idx] : f32_to_bf16(0.0f);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for relu_backward");
    }

    return grad_input;
}

// Sigmoid activation
auto sigmoid_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SigmoidKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0f / (1.0f + sycl::exp(-in_ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SigmoidKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0 / (1.0 + sycl::exp(-in_ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SigmoidKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(1.0f / (1.0f + sycl::exp(-val)));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SigmoidKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(1.0f / (1.0f + sycl::exp(-val)));
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for sigmoid");
    }

    return output;
}

// Sigmoid backward
auto sigmoid_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor {
    Tensor grad_input(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                      output.dtype(), output.device());

    const int64_t numel = output.numel();

    if (output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* out_ptr = get_data_ptr<const float>(output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * out_ptr[idx] * (1.0f - out_ptr[idx]);
        }).wait();
    }
    else if (output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* out_ptr = get_data_ptr<const double>(output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * out_ptr[idx] * (1.0 - out_ptr[idx]);
        }).wait();
    }
    else if (output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = static_cast<float>(grad_out_ptr[idx]);
            float out_val = static_cast<float>(out_ptr[idx]);
            grad_in_ptr[idx] = sycl::half(g_out * out_val * (1.0f - out_val));
        }).wait();
    }
    else if (output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<SigmoidBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = bf16_to_f32(grad_out_ptr[idx]);
            float out_val = bf16_to_f32(out_ptr[idx]);
            grad_in_ptr[idx] = f32_to_bf16(g_out * out_val * (1.0f - out_val));
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for sigmoid_backward");
    }

    return grad_input;
}

// Tanh activation
auto tanh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TanhKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tanh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TanhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tanh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<TanhKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float val = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(sycl::tanh(val));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<TanhKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = f32_to_bf16(sycl::tanh(bf16_to_f32(in_ptr[idx])));
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for tanh");
    }

    return output;
}

// Tanh backward
auto tanh_backward_kernel(const Tensor& grad_output, const Tensor& output, sycl::queue& queue) -> Tensor {
    Tensor grad_input(std::vector<int64_t>(output.shape().begin(), output.shape().end()),
                      output.dtype(), output.device());

    const int64_t numel = output.numel();

    if (output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* out_ptr = get_data_ptr<const float>(output);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for<TanhBackwardKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * (1.0f - out_ptr[idx] * out_ptr[idx]);
        }).wait();
    }
    else if (output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* out_ptr = get_data_ptr<const double>(output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<TanhBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * (1.0 - out_ptr[idx] * out_ptr[idx]);
        }).wait();
    }
    else if (output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(output);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<TanhBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = static_cast<float>(grad_out_ptr[idx]);
            float out_val = static_cast<float>(out_ptr[idx]);
            grad_in_ptr[idx] = sycl::half(g_out * (1.0f - out_val * out_val));
        }).wait();
    }
    else if (output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(output);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<TanhBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float g_out = bf16_to_f32(grad_out_ptr[idx]);
            float out_val = bf16_to_f32(out_ptr[idx]);
            grad_in_ptr[idx] = f32_to_bf16(g_out * (1.0f - out_val * out_val));
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else if (in_cont.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(in_cont);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SoftmaxKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
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
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for softmax");
    }

    return output;
}

// Softmax backward
auto softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()),
                      output.dtype(), output.device());

    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* out_ptr = get_data_ptr<const float>(output);
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
        }).wait();
    }
    else if (output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* out_ptr = get_data_ptr<const double>(output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for<SoftmaxBackwardKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            double dot = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                dot += grad_out_ptr[offset + d * inner_size] * out_ptr[offset + d * inner_size];
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                grad_in_ptr[offset + d * inner_size] = out_ptr[offset + d * inner_size] *
                    (grad_out_ptr[offset + d * inner_size] - dot);
            }
        }).wait();
    }
    else if (output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(output);
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
        }).wait();
    }
    else if (output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(output);
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double alpha_d = static_cast<double>(alpha);

        queue.parallel_for<LeakyReLUKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            double x = in_ptr[idx];
            out_ptr[idx] = x > 0.0 ? x : alpha_d * x;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<LeakyReLUKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(in_ptr[idx]);
            out_ptr[idx] = sycl::half(x > 0.0f ? x : alpha * x);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<LeakyReLUKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            out_ptr[idx] = f32_to_bf16(x > 0.0f ? x : alpha * x);
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);
        const double alpha_d = static_cast<double>(alpha);

        queue.parallel_for<LeakyReLUBackwardKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0 ? grad_out_ptr[idx] : alpha_d * grad_out_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* grad_in_ptr = get_data_ptr<sycl::half>(grad_input);

        queue.parallel_for<LeakyReLUBackwardKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = static_cast<float>(in_ptr[idx]);
            float grad_out_val = static_cast<float>(grad_out_ptr[idx]);
            grad_in_ptr[idx] = sycl::half(in_val > 0.0f ? grad_out_val : alpha * grad_out_val);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* grad_in_ptr = get_data_ptr<uint16_t>(grad_input);

        queue.parallel_for<LeakyReLUBackwardKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float in_val = bf16_to_f32(in_ptr[idx]);
            float g_out = bf16_to_f32(grad_out_ptr[idx]);
            grad_in_ptr[idx] = f32_to_bf16(in_val > 0.0f ? g_out : alpha * g_out);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for leaky_relu_backward");
    }

    return grad_input;
}

// LogSoftmax activation - numerically stable version
// log(softmax(x)) = x - max(x) - log(sum(exp(x - max(x))))
auto log_softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    if (dim < 0 || dim >= static_cast<int64_t>(shape.size())) {
        throw std::runtime_error("LogSoftmax dimension out of range");
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
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
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
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
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
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
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for log_softmax");
    }

    return output;
}

// LogSoftmax backward
// grad_input = grad_output - exp(log_softmax) * sum(grad_output)
auto log_softmax_backward_kernel(const Tensor& grad_output, const Tensor& output, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = output.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    Tensor grad_input(std::vector<int64_t>(shape.begin(), shape.end()), output.dtype(), output.device());

    if (output.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* out_ptr = get_data_ptr<const float>(output);
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
        }).wait();
    }
    else if (output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* out_ptr = get_data_ptr<const double>(output);
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
        }).wait();
    }
    else if (output.dtype() == DType::Float16) {
        const sycl::half* grad_out_ptr = get_data_ptr<const sycl::half>(grad_output);
        const sycl::half* out_ptr = get_data_ptr<const sycl::half>(output);
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
        }).wait();
    }
    else if (output.dtype() == DType::BFloat16) {
        const uint16_t* grad_out_ptr = get_data_ptr<const uint16_t>(grad_output);
        const uint16_t* out_ptr = get_data_ptr<const uint16_t>(output);
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SwishKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const double x = in_ptr[idx];
            const double sigmoid = 1.0 / (1.0 + sycl::exp(-x));
            out_ptr[idx] = x * sigmoid;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        const sycl::half* in_ptr = get_data_ptr<const sycl::half>(input);
        sycl::half* out_ptr = get_data_ptr<sycl::half>(output);

        queue.parallel_for<SwishKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            const float x = static_cast<float>(in_ptr[idx]);
            const float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = sycl::half(x * sigmoid);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        const uint16_t* in_ptr = get_data_ptr<const uint16_t>(input);
        uint16_t* out_ptr = get_data_ptr<uint16_t>(output);

        queue.parallel_for<SwishKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(in_ptr[idx]);
            float sigmoid = 1.0f / (1.0f + sycl::exp(-x));
            out_ptr[idx] = f32_to_bf16(x * sigmoid);
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        queue.parallel_for<ReLUInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (ptr[idx] < 0.0) ptr[idx] = 0.0;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<ReLUInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (float(ptr[idx]) < 0.0f) ptr[idx] = sycl::half(0.0f);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<ReLUInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (bf16_to_f32(ptr[idx]) < 0.0f) ptr[idx] = f32_to_bf16(0.0f);
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        queue.parallel_for<SigmoidInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            ptr[idx] = 1.0 / (1.0 + sycl::exp(-ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<SigmoidInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            ptr[idx] = sycl::half(1.0f / (1.0f + sycl::exp(-x)));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<SigmoidInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            ptr[idx] = f32_to_bf16(1.0f / (1.0f + sycl::exp(-x)));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        queue.parallel_for<TanhInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            ptr[idx] = sycl::tanh(ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<TanhInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            ptr[idx] = sycl::half(sycl::tanh(x));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<TanhInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            ptr[idx] = f32_to_bf16(sycl::tanh(x));
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        double* ptr = get_data_ptr<double>(input);
        const double alpha_d = static_cast<double>(alpha);
        queue.parallel_for<LeakyReLUInplaceKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            if (ptr[idx] < 0.0) ptr[idx] *= alpha_d;
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<LeakyReLUInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            if (x < 0.0f) ptr[idx] = sycl::half(x * alpha);
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<LeakyReLUInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            if (x < 0.0f) ptr[idx] = f32_to_bf16(x * alpha);
        }).wait();
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
        }).wait();
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
        }).wait();
    }
    else if (input.dtype() == DType::Float16) {
        sycl::half* ptr = get_data_ptr<sycl::half>(input);
        queue.parallel_for<GeLUInplaceKernelFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = static_cast<float>(ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            ptr[idx] = sycl::half(0.5f * x * (1.0f + sycl::tanh(inner)));
        }).wait();
    }
    else if (input.dtype() == DType::BFloat16) {
        uint16_t* ptr = get_data_ptr<uint16_t>(input);
        queue.parallel_for<GeLUInplaceKernelBFloat16>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = bf16_to_f32(ptr[idx]);
            float x_cubed = x * x * x;
            float inner = sqrt_2_over_pi * (x + coeff * x_cubed);
            ptr[idx] = f32_to_bf16(0.5f * x * (1.0f + sycl::tanh(inner)));
        }).wait();
    }
    else {
        throw std::runtime_error("gelu_inplace: unsupported dtype");
    }
}

} // namespace oneapi
} // namespace tenzor
