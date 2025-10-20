#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
#include <cmath>
#include <stdexcept>

namespace tenzor {
namespace oneapi {

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// ReLU activation forward
auto relu_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(0.0f, in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(0.0, in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for relu");
    }

    return output;
}

// ReLU backward
auto relu_backward_kernel(const Tensor& grad_output, const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor grad_input(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                      input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* grad_out_ptr = get_data_ptr<const float>(grad_output);
        const float* in_ptr = get_data_ptr<const float>(input);
        float* grad_in_ptr = get_data_ptr<float>(grad_input);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0f ? grad_out_ptr[idx] : 0.0f;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0f / (1.0f + sycl::exp(-in_ptr[idx]));
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * out_ptr[idx] * (1.0f - out_ptr[idx]);
        }).wait();
    }
    else if (output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* out_ptr = get_data_ptr<const double>(output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tanh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = grad_out_ptr[idx] * (1.0f - out_ptr[idx] * out_ptr[idx]);
        }).wait();
    }
    else if (output.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* out_ptr = get_data_ptr<const double>(output);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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

        queue.parallel_for(sycl::range<2>(outer_size, inner_size), [=](sycl::id<2> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            float x = in_ptr[idx];
            out_ptr[idx] = x > 0.0f ? x : alpha * x;
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double alpha_d = static_cast<double>(alpha);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0f ? grad_out_ptr[idx] : alpha * grad_out_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* grad_out_ptr = get_data_ptr<const double>(grad_output);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* grad_in_ptr = get_data_ptr<double>(grad_input);
        const double alpha_d = static_cast<double>(alpha);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            grad_in_ptr[idx] = in_ptr[idx] > 0.0 ? grad_out_ptr[idx] : alpha_d * grad_out_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for leaky_relu_backward");
    }

    return grad_input;
}

} // namespace oneapi
} // namespace tenzor
