#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <cmath>
#include <stdexcept>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

// Forward declaration for broadcasting fallback
namespace tenzor {
    auto add(const Tensor& a, const Tensor& b) -> Tensor;
    auto sub(const Tensor& a, const Tensor& b) -> Tensor;
    auto mul(const Tensor& a, const Tensor& b) -> Tensor;
    auto div(const Tensor& a, const Tensor& b) -> Tensor;
}

namespace tenzor {
namespace oneapi {

// SYCL Kernel name classes (using functors for SYCL 2025.2 compatibility)
struct AddKernelFloat32 {};
struct AddKernelFloat64 {};
struct SubKernelFloat32 {};
struct SubKernelFloat64 {};
struct MulKernelFloat32 {};
struct MulKernelFloat64 {};
struct DivKernelFloat32 {};
struct DivKernelFloat64 {};
struct MatMulKernelFloat32 {};
struct MatMulKernelFloat64 {};
struct SqrtKernelFloat32 {};
struct SqrtKernelFloat64 {};
struct NegKernelFloat32 {};
struct NegKernelFloat64 {};
struct AbsKernelFloat32 {};
struct AbsKernelFloat64 {};
struct LogKernelFloat32 {};
struct LogKernelFloat64 {};
struct ExpKernelFloat32 {};
struct ExpKernelFloat64 {};
struct PowKernelFloat32 {};
struct PowKernelFloat64 {};

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

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    // If shapes don't match, fall back to CPU for broadcasting
    // TODO: Implement proper SYCL broadcasting kernels
    if (!same_shape) {
        // Copy to CPU, perform operation, copy back
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());

        // CPU backend handles broadcasting
        auto result_cpu = tenzor::add(a_cpu, b_cpu);

        // Copy result back to OneAPI device
        return result_cpu.to(a.device());
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

        queue.parallel_for<AddKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] + b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AddKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    // If shapes don't match, fall back to CPU for broadcasting
    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::sub(a_cpu, b_cpu);
        return result_cpu.to(a.device());
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SubKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] - b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SubKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    // If shapes don't match, fall back to CPU for broadcasting
    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::mul(a_cpu, b_cpu);
        return result_cpu.to(a.device());
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<MulKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] * b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<MulKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

    // Check if shapes match exactly
    bool same_shape = std::equal(a_shape.begin(), a_shape.end(), b_shape.begin(), b_shape.end());

    // If shapes don't match, fall back to CPU for broadcasting
    if (!same_shape) {
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto result_cpu = tenzor::div(a_cpu, b_cpu);
        return result_cpu.to(a.device());
    }

    Tensor output(std::vector<int64_t>(a.shape().begin(), a.shape().end()),
                  a.dtype(), a.device());

    const int64_t numel = a.numel();

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<DivKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = a_ptr[idx] / b_ptr[idx];
        }).wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<DivKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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
    // oneMKL GEMM: C = alpha * op(A) * op(B) + beta * C
    // where A is (m x k), B is (k x n), C is (m x n)
    // oneMKL uses column-major layout, so we compute C^T = B^T * A^T
    // This means: gemm(transB, transA, n, m, k, alpha, B, ldb, A, lda, beta, C, ldc)

    if (a.dtype() == DType::Float32) {
        const float* a_ptr = get_data_ptr<const float>(a);
        const float* b_ptr = get_data_ptr<const float>(b);
        float* out_ptr = get_data_ptr<float>(output);

        const float alpha = 1.0f;
        const float beta = 0.0f;

        // oneMKL expects column-major, we have row-major
        // For row-major C = A * B (A: m x k, B: k x n, C: m x n)
        // Equivalent to column-major C^T = B^T * A^T
        ::oneapi::mkl::blas::column_major::gemm(
            queue,
            ::oneapi::mkl::transpose::nontrans,  // B^T is not transposed (since B is already in memory as we want)
            ::oneapi::mkl::transpose::nontrans,  // A^T is not transposed
            n,        // number of rows of op(B) and C in column-major
            m,        // number of columns of op(A) and C in column-major
            k,        // number of columns of op(B) and rows of op(A)
            alpha,
            b_ptr, n, // B, leading dimension n
            a_ptr, k, // A, leading dimension k
            beta,
            out_ptr, n // C, leading dimension n
        );
        queue.wait();
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_ptr = get_data_ptr<const double>(a);
        const double* b_ptr = get_data_ptr<const double>(b);
        double* out_ptr = get_data_ptr<double>(output);

        const double alpha = 1.0;
        const double beta = 0.0;

        ::oneapi::mkl::blas::column_major::gemm(
            queue,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::transpose::nontrans,
            n, m, k,
            alpha,
            b_ptr, n,
            a_ptr, k,
            beta,
            out_ptr, n
        );
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
        queue.parallel_for<MatMulKernelFloat32>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
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

        queue.parallel_for<MatMulKernelFloat64>(sycl::range<2>(m, n), [=](sycl::id<2> idx) {
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

        queue.parallel_for<SqrtKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sqrt(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SqrtKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for<NegKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = -in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<NegKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for<AbsKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fabs(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AbsKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for<LogKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::log(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<LogKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for<ExpKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::exp(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ExpKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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

        queue.parallel_for<PowKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::pow(in_ptr[idx], exponent);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double exp_d = static_cast<double>(exponent);

        queue.parallel_for<PowKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
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
