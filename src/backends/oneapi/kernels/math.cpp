#include "tenzor/core/tensor.hpp"
#include <CL/sycl.hpp>
#include <cmath>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace tenzor {
namespace oneapi {

// Helper function to get typed pointer from tensor
template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

// Helper to calculate total elements
inline auto calculate_numel(const std::vector<int64_t>& shape) -> int64_t {
    int64_t numel = 1;
    for (auto s : shape) {
        numel *= s;
    }
    return numel;
}

// Element-wise addition kernel
auto add_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Validate inputs
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensor shapes must match for addition");
    }

    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("Tensor dtypes must match for addition");
    }

    // Create output tensor
    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    // Dispatch based on dtype
    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for addition");
    }

    return output;
}

// Element-wise subtraction kernel
auto sub_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensor shapes must match for subtraction");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for subtraction");
    }

    return output;
}

// Element-wise multiplication kernel
auto mul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensor shapes must match for multiplication");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for multiplication");
    }

    return output;
}

// Element-wise division kernel
auto div_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (!std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end())) {
        throw std::invalid_argument("Tensor shapes must match for division");
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for division");
    }

    return output;
}

// Matrix multiplication kernel
auto matmul_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    auto a_shape = a.shape();
    auto b_shape = b.shape();

    // Validate dimensions
    if (a_shape.size() < 2 || b_shape.size() < 2) {
        throw std::invalid_argument("matmul requires at least 2D tensors");
    }

    const int64_t m = a_shape[a_shape.size() - 2];
    const int64_t k = a_shape[a_shape.size() - 1];
    const int64_t k2 = b_shape[b_shape.size() - 2];
    const int64_t n = b_shape[b_shape.size() - 1];

    if (k != k2) {
        throw std::invalid_argument("Inner dimensions must match for matmul");
    }

    // Create output shape
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i < a_shape.size() - 2; ++i) {
        out_shape.push_back(a_shape[i]);
    }
    out_shape.push_back(m);
    out_shape.push_back(n);

    Tensor output(out_shape, a.dtype(), a.device());

#ifdef TENZOR_HAS_ONEMKL
    // Use oneMKL for optimized GEMM
    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        oneapi::mkl::blas::gemm(queue, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans,
                                n, m, k,
                                alpha, b_ptr, n, a_ptr, k,
                                beta, out_ptr, n);
        queue.wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        const double alpha = 1.0;
        const double beta = 0.0;

        oneapi::mkl::blas::gemm(queue, oneapi::mkl::transpose::nontrans,
                                oneapi::mkl::transpose::nontrans,
                                n, m, k,
                                alpha, b_ptr, n, a_ptr, k,
                                beta, out_ptr, n);
        queue.wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for matmul with oneMKL");
    }
#else
    // Fallback naive implementation for Float32
    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        // Simple parallel matrix multiplication
        queue.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            float sum = 0.0f;
            for (int64_t p = 0; p < k; ++p) {
                sum += a_ptr[i * k + p] * b_ptr[p * n + j];
            }
            out_ptr[i * n + j] = sum;
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
            const int64_t i = idx[0];
            const int64_t j = idx[1];

            double sum = 0.0;
            for (int64_t p = 0; p < k; ++p) {
                sum += a_ptr[i * k + p] * b_ptr[p * n + j];
            }
            out_ptr[i * n + j] = sum;
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for matmul");
    }
#endif

    return output;
}

// Square root kernel
auto sqrt_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for sqrt");
    }

    return output;
}

// Negation kernel
auto neg_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for negation");
    }

    return output;
}

// Absolute value kernel
auto abs_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for abs");
    }

    return output;
}

// Natural logarithm kernel
auto log_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for log");
    }

    return output;
}

// Exponential kernel
auto exp_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for exp");
    }

    return output;
}

// Power kernel
auto pow_kernel(const Tensor& input, float exponent, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::pow(in_ptr[idx], exponent);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double exp_d = static_cast<double>(exponent);

        queue.parallel_for(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::pow(in_ptr[idx], exp_d);
        }).wait();
    }
    else {
        throw std::runtime_error("Unsupported dtype for pow");
    }

    return output;
}

} // namespace oneapi
} // namespace tenzor
