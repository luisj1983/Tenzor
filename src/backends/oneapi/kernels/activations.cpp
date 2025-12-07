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
class ReLUBackwardKernelFloat32;
class ReLUBackwardKernelFloat64;
class SigmoidKernelFloat32;
class SigmoidKernelFloat64;
class SigmoidBackwardKernelFloat32;
class SigmoidBackwardKernelFloat64;
class TanhKernelFloat32;
class TanhKernelFloat64;
class TanhBackwardKernelFloat32;
class TanhBackwardKernelFloat64;
class GeLUKernelFloat32;
class GeLUKernelFloat64;
class GeLUBackwardKernelFloat32;
class GeLUBackwardKernelFloat64;
class SoftmaxKernelFloat32;
class SoftmaxKernelFloat64;
class SoftmaxBackwardKernelFloat32;
class SoftmaxBackwardKernelFloat64;
class LeakyReLUKernelFloat32;
class LeakyReLUKernelFloat64;
class LeakyReLUBackwardKernelFloat32;
class LeakyReLUBackwardKernelFloat64;
class SwishKernelFloat32;
class SwishKernelFloat64;
class SwishBackwardKernelFloat32;
class SwishBackwardKernelFloat64;

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
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
    else {
        throw std::runtime_error("Unsupported dtype for gelu_backward");
    }

    return grad_input;
}

// Softmax activation
auto softmax_kernel(const Tensor& input, int64_t dim, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (dim < 0) {
        dim += shape.size();
    }

    Tensor output(std::vector<int64_t>(shape.begin(), shape.end()),
                  input.dtype(), input.device());

    const int64_t outer_size = std::accumulate(shape.begin(), shape.begin() + dim, 1LL, std::multiplies<>());
    const int64_t dim_size = shape[dim];
    const int64_t inner_size = std::accumulate(shape.begin() + dim + 1, shape.end(), 1LL, std::multiplies<>());

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SoftmaxKernelFloat32>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            // Find max for numerical stability
            float max_val = -3.4028235e+38f;  // -FLT_MAX (avoid INFINITY with -ffast-math)
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = in_ptr[offset + d * inner_size];
                max_val = sycl::fmax(max_val, val);
            }

            // Compute exp and sum
            float sum = 0.0f;
            for (int64_t d = 0; d < dim_size; ++d) {
                float val = sycl::exp(in_ptr[offset + d * inner_size] - max_val);
                out_ptr[offset + d * inner_size] = val;
                sum += val;
            }

            // Normalize
            for (int64_t d = 0; d < dim_size; ++d) {
                out_ptr[offset + d * inner_size] /= sum;
            }
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SoftmaxKernelFloat64>(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
            const int64_t outer_idx = idx[0];
            const int64_t inner_idx = idx[1];
            const int64_t offset = (outer_idx * dim_size * inner_size) + inner_idx;

            double max_val = -1.7976931348623157e+308;  // -DBL_MAX (avoid INFINITY with -ffast-math)
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = in_ptr[offset + d * inner_size];
                max_val = sycl::fmax(max_val, val);
            }

            double sum = 0.0;
            for (int64_t d = 0; d < dim_size; ++d) {
                double val = sycl::exp(in_ptr[offset + d * inner_size] - max_val);
                out_ptr[offset + d * inner_size] = val;
                sum += val;
            }

            for (int64_t d = 0; d < dim_size; ++d) {
                out_ptr[offset + d * inner_size] /= sum;
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
    else {
        throw std::runtime_error("Unsupported dtype for swish_backward");
    }

    return grad_input;
}

} // namespace oneapi
} // namespace tenzor
