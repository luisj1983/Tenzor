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

// Trigonometric kernel name classes
struct SinKernelFloat32 {};
struct SinKernelFloat64 {};
struct CosKernelFloat32 {};
struct CosKernelFloat64 {};
struct TanKernelFloat32 {};
struct TanKernelFloat64 {};
struct AsinKernelFloat32 {};
struct AsinKernelFloat64 {};
struct AcosKernelFloat32 {};
struct AcosKernelFloat64 {};
struct AtanKernelFloat32 {};
struct AtanKernelFloat64 {};
struct SinhKernelFloat32 {};
struct SinhKernelFloat64 {};
struct CoshKernelFloat32 {};
struct CoshKernelFloat64 {};
struct Atan2KernelFloat32 {};
struct Atan2KernelFloat64 {};

// Rounding kernel name classes
struct RoundKernelFloat32 {};
struct RoundKernelFloat64 {};
struct FloorKernelFloat32 {};
struct FloorKernelFloat64 {};
struct CeilKernelFloat32 {};
struct CeilKernelFloat64 {};
struct TruncKernelFloat32 {};
struct TruncKernelFloat64 {};
struct ReciprocalKernelFloat32 {};
struct ReciprocalKernelFloat64 {};

// Utility kernel name classes
struct ClampMinKernelFloat32 {};
struct ClampMinKernelFloat64 {};
struct ClampMaxKernelFloat32 {};
struct ClampMaxKernelFloat64 {};
struct WhereKernelFloat32 {};
struct WhereKernelFloat64 {};
struct RepeatKernelFloat32 {};
struct RepeatKernelFloat64 {};

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

    // Handle 1D vector × 2D matrix (vector-matrix multiplication)
    if (a_shape.size() == 1 && b_shape.size() == 2) {
        const int64_t n = a_shape[0];  // Vector size
        const int64_t k = b_shape[0];  // Matrix rows
        const int64_t m = b_shape[1];  // Matrix cols

        if (n != k) {
            throw std::invalid_argument(
                "matmul dimension mismatch: vector(" + std::to_string(n) +
                ") @ matrix(" + std::to_string(k) + "×" + std::to_string(m) + ")"
            );
        }

        // Treat 1D vector as row vector (1, n) and perform matmul to get (1, m), then return as (m,)
        Tensor output({m}, a.dtype(), a.device());

#ifdef TENZOR_HAS_ONEMKL
        if (a.dtype() == DType::Float32) {
            const float* a_ptr = get_data_ptr<const float>(a);
            const float* b_ptr = get_data_ptr<const float>(b);
            float* out_ptr = get_data_ptr<float>(output);

            const float alpha = 1.0f;
            const float beta = 0.0f;

            // For row-major: result[m] = vector[n] × matrix[n, m]
            // In oneMKL column-major: C^T[m] = B^T[m, n] × A^T[n]
            ::oneapi::mkl::blas::column_major::gemm(
                queue,
                ::oneapi::mkl::transpose::nontrans,
                ::oneapi::mkl::transpose::nontrans,
                m,        // rows of result
                1,        // cols of result (single vector)
                n,        // inner dimension
                alpha,
                b_ptr, m, // B matrix
                a_ptr, n, // A vector
                beta,
                out_ptr, m // output
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
                m, 1, n,
                alpha,
                b_ptr, m,
                a_ptr, n,
                beta,
                out_ptr, m
            );
            queue.wait();
        }
        else {
            throw std::runtime_error("Unsupported dtype for 1D×2D matmul with oneMKL");
        }
#else
        // Fallback naive implementation
        if (a.dtype() == DType::Float32) {
            const float* a_ptr = get_data_ptr<const float>(a);
            const float* b_ptr = get_data_ptr<const float>(b);
            float* out_ptr = get_data_ptr<float>(output);

            queue.parallel_for<class MatMulKernelVector>(sycl::range<1>(m), [=](sycl::id<1> idx) {
                const int64_t j = idx[0];
                float sum = 0.0f;
                for (int64_t p = 0; p < n; ++p) {
                    sum += a_ptr[p] * b_ptr[p * m + j];
                }
                out_ptr[j] = sum;
            }).wait();
        }
        else {
            throw std::runtime_error("Unsupported dtype for 1D×2D matmul fallback");
        }
#endif
        return output;
    }

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

// Dot product kernel - element-wise multiply then sum
auto dot_kernel(const Tensor& a, const Tensor& b, sycl::queue& queue) -> Tensor {
    // Verify both tensors are 1D
    if (a.ndim() != 1 || b.ndim() != 1) {
        throw std::invalid_argument("dot: inputs must be 1D tensors");
    }

    // Verify same shape
    if (a.shape()[0] != b.shape()[0]) {
        throw std::invalid_argument("dot: inputs must have the same length");
    }

    // Verify same dtype
    if (a.dtype() != b.dtype()) {
        throw std::invalid_argument("dot: inputs must have the same dtype");
    }

    int64_t n = a.shape()[0];

    // Create scalar output tensor
    Tensor output({1}, a.dtype(), a.device());

    if (a.dtype() == DType::Float32) {
        const float* a_data = get_data_ptr<const float>(a);
        const float* b_data = get_data_ptr<const float>(b);

        // Copy to host and compute dot product
        std::vector<float> a_host(n);
        std::vector<float> b_host(n);
        queue.memcpy(a_host.data(), a_data, n * sizeof(float)).wait();
        queue.memcpy(b_host.data(), b_data, n * sizeof(float)).wait();

        float sum = 0.0f;
        for (int64_t i = 0; i < n; ++i) {
            sum += a_host[i] * b_host[i];
        }

        // Copy result back to device
        float* out_ptr = get_data_ptr<float>(output);
        queue.fill(out_ptr, sum, 1).wait();

        return output;
    }
    else if (a.dtype() == DType::Float64) {
        const double* a_data = get_data_ptr<const double>(a);
        const double* b_data = get_data_ptr<const double>(b);

        // Copy to host and compute dot product
        std::vector<double> a_host(n);
        std::vector<double> b_host(n);
        queue.memcpy(a_host.data(), a_data, n * sizeof(double)).wait();
        queue.memcpy(b_host.data(), b_data, n * sizeof(double)).wait();

        double sum = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            sum += a_host[i] * b_host[i];
        }

        // Copy result back to device
        double* out_ptr = get_data_ptr<double>(output);
        queue.fill(out_ptr, sum, 1).wait();

        return output;
    }
    else {
        throw std::runtime_error("dot: only Float32 and Float64 are supported");
    }
}

// ============================================================================
// Trigonometric Functions
// ============================================================================

// Sine kernel
auto sin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SinKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sin(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sin(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("sin: unsupported dtype");
    }

    return output;
}

// Cosine kernel
auto cos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<CosKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cos(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CosKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cos(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("cos: unsupported dtype");
    }

    return output;
}

// Tangent kernel
auto tan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TanKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tan(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::tan(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("tan: unsupported dtype");
    }

    return output;
}

// Arc sine kernel
auto asin_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AsinKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asin(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AsinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::asin(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("asin: unsupported dtype");
    }

    return output;
}

// Arc cosine kernel
auto acos_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AcosKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acos(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AcosKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::acos(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("acos: unsupported dtype");
    }

    return output;
}

// Arc tangent kernel
auto atan_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<AtanKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<AtanKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("atan: unsupported dtype");
    }

    return output;
}

// Hyperbolic sine kernel
auto sinh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<SinhKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sinh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<SinhKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::sinh(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("sinh: unsupported dtype");
    }

    return output;
}

// Hyperbolic cosine kernel
auto cosh_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<CoshKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cosh(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CoshKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::cosh(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("cosh: unsupported dtype");
    }

    return output;
}

// Arc tangent 2 (atan2) kernel - binary operation
auto atan2_kernel(const Tensor& y, const Tensor& x, sycl::queue& queue) -> Tensor {
    if (y.dtype() != x.dtype()) {
        throw std::invalid_argument("atan2: input dtypes must match");
    }

    Tensor output(std::vector<int64_t>(y.shape().begin(), y.shape().end()),
                  y.dtype(), y.device());

    const int64_t numel = y.numel();

    if (y.dtype() == DType::Float32) {
        const float* y_ptr = get_data_ptr<const float>(y);
        const float* x_ptr = get_data_ptr<const float>(x);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<Atan2KernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(y_ptr[idx], x_ptr[idx]);
        }).wait();
    }
    else if (y.dtype() == DType::Float64) {
        const double* y_ptr = get_data_ptr<const double>(y);
        const double* x_ptr = get_data_ptr<const double>(x);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<Atan2KernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::atan2(y_ptr[idx], x_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("atan2: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Rounding Functions
// ============================================================================

// Round kernel
auto round_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<RoundKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::round(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<RoundKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::round(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("round: unsupported dtype");
    }

    return output;
}

// Floor kernel
auto floor_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<FloorKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::floor(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<FloorKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::floor(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("floor: unsupported dtype");
    }

    return output;
}

// Ceil kernel
auto ceil_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<CeilKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ceil(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<CeilKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::ceil(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("ceil: unsupported dtype");
    }

    return output;
}

// Trunc kernel
auto trunc_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<TruncKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::trunc(in_ptr[idx]);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<TruncKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::trunc(in_ptr[idx]);
        }).wait();
    }
    else {
        throw std::runtime_error("trunc: unsupported dtype");
    }

    return output;
}

// Reciprocal kernel
auto reciprocal_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ReciprocalKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0f / in_ptr[idx];
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<ReciprocalKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = 1.0 / in_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("reciprocal: unsupported dtype");
    }

    return output;
}

// ============================================================================
// Utility Functions
// ============================================================================

// Clamp min kernel
auto clamp_min_kernel(const Tensor& input, float min_val, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ClampMinKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(in_ptr[idx], min_val);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double min_d = static_cast<double>(min_val);

        queue.parallel_for<ClampMinKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmax(in_ptr[idx], min_d);
        }).wait();
    }
    else {
        throw std::runtime_error("clamp_min: unsupported dtype");
    }

    return output;
}

// Clamp max kernel
auto clamp_max_kernel(const Tensor& input, float max_val, sycl::queue& queue) -> Tensor {
    Tensor output(std::vector<int64_t>(input.shape().begin(), input.shape().end()),
                  input.dtype(), input.device());

    const int64_t numel = input.numel();

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<ClampMaxKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(in_ptr[idx], max_val);
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);
        const double max_d = static_cast<double>(max_val);

        queue.parallel_for<ClampMaxKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = sycl::fmin(in_ptr[idx], max_d);
        }).wait();
    }
    else {
        throw std::runtime_error("clamp_max: unsupported dtype");
    }

    return output;
}

// Where kernel (conditional selection)
auto where_kernel(const Tensor& condition, const Tensor& x, const Tensor& y, sycl::queue& queue) -> Tensor {
    if (x.dtype() != y.dtype()) {
        throw std::invalid_argument("where: x and y must have the same dtype");
    }

    Tensor output(std::vector<int64_t>(x.shape().begin(), x.shape().end()),
                  x.dtype(), x.device());

    const int64_t numel = x.numel();

    if (x.dtype() == DType::Float32) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const float* x_ptr = get_data_ptr<const float>(x);
        const float* y_ptr = get_data_ptr<const float>(y);
        float* out_ptr = get_data_ptr<float>(output);

        queue.parallel_for<WhereKernelFloat32>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        }).wait();
    }
    else if (x.dtype() == DType::Float64) {
        const bool* cond_ptr = get_data_ptr<const bool>(condition);
        const double* x_ptr = get_data_ptr<const double>(x);
        const double* y_ptr = get_data_ptr<const double>(y);
        double* out_ptr = get_data_ptr<double>(output);

        queue.parallel_for<WhereKernelFloat64>(sycl::range<1>(numel), [=](sycl::id<1> idx) {
            out_ptr[idx] = cond_ptr[idx] ? x_ptr[idx] : y_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("where: unsupported dtype");
    }

    return output;
}

// Repeat kernel - repeats tensor along specified dimensions
auto repeat_kernel(const Tensor& input, const std::vector<int64_t>& repeats, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    if (repeats.size() != shape.size()) {
        throw std::invalid_argument("repeat: repeats size must match tensor dimensions");
    }

    // Calculate output shape
    std::vector<int64_t> out_shape(shape.size());
    for (size_t i = 0; i < shape.size(); ++i) {
        out_shape[i] = shape[i] * repeats[i];
    }

    Tensor output(out_shape, input.dtype(), input.device());

    // For simplicity, use a strided copy approach
    // This handles general N-dimensional repeat
    const int64_t ndim = shape.size();
    const int64_t out_numel = output.numel();

    // Pre-compute strides
    std::vector<int64_t> in_strides(ndim);
    std::vector<int64_t> out_strides(ndim);
    int64_t in_stride = 1;
    int64_t out_stride = 1;
    for (int64_t i = ndim - 1; i >= 0; --i) {
        in_strides[i] = in_stride;
        out_strides[i] = out_stride;
        in_stride *= shape[i];
        out_stride *= out_shape[i];
    }

    if (input.dtype() == DType::Float32) {
        const float* in_ptr = get_data_ptr<const float>(input);
        float* out_ptr = get_data_ptr<float>(output);

        // Copy shape and strides to device-accessible memory
        auto shape_buf = sycl::buffer<int64_t, 1>(shape.data(), sycl::range<1>(ndim));
        auto out_shape_buf = sycl::buffer<int64_t, 1>(out_shape.data(), sycl::range<1>(ndim));
        auto in_strides_buf = sycl::buffer<int64_t, 1>(in_strides.data(), sycl::range<1>(ndim));
        auto out_strides_buf = sycl::buffer<int64_t, 1>(out_strides.data(), sycl::range<1>(ndim));

        queue.submit([&](sycl::handler& h) {
            auto shape_acc = shape_buf.get_access<sycl::access::mode::read>(h);
            auto out_shape_acc = out_shape_buf.get_access<sycl::access::mode::read>(h);
            auto in_strides_acc = in_strides_buf.get_access<sycl::access::mode::read>(h);
            auto out_strides_acc = out_strides_buf.get_access<sycl::access::mode::read>(h);

            h.parallel_for<RepeatKernelFloat32>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t out_idx = idx[0];
                int64_t in_idx = 0;

                // Convert output index to input index using modular arithmetic
                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t coord = (out_idx / out_strides_acc[d]) % out_shape_acc[d];
                    int64_t in_coord = coord % shape_acc[d];
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        }).wait();
    }
    else if (input.dtype() == DType::Float64) {
        const double* in_ptr = get_data_ptr<const double>(input);
        double* out_ptr = get_data_ptr<double>(output);

        auto shape_buf = sycl::buffer<int64_t, 1>(shape.data(), sycl::range<1>(ndim));
        auto out_shape_buf = sycl::buffer<int64_t, 1>(out_shape.data(), sycl::range<1>(ndim));
        auto in_strides_buf = sycl::buffer<int64_t, 1>(in_strides.data(), sycl::range<1>(ndim));
        auto out_strides_buf = sycl::buffer<int64_t, 1>(out_strides.data(), sycl::range<1>(ndim));

        queue.submit([&](sycl::handler& h) {
            auto shape_acc = shape_buf.get_access<sycl::access::mode::read>(h);
            auto out_shape_acc = out_shape_buf.get_access<sycl::access::mode::read>(h);
            auto in_strides_acc = in_strides_buf.get_access<sycl::access::mode::read>(h);
            auto out_strides_acc = out_strides_buf.get_access<sycl::access::mode::read>(h);

            h.parallel_for<RepeatKernelFloat64>(sycl::range<1>(out_numel), [=](sycl::id<1> idx) {
                int64_t out_idx = idx[0];
                int64_t in_idx = 0;

                for (int64_t d = 0; d < ndim; ++d) {
                    int64_t coord = (out_idx / out_strides_acc[d]) % out_shape_acc[d];
                    int64_t in_coord = coord % shape_acc[d];
                    in_idx += in_coord * in_strides_acc[d];
                }

                out_ptr[out_idx] = in_ptr[in_idx];
            });
        }).wait();
    }
    else {
        throw std::runtime_error("repeat: unsupported dtype");
    }

    return output;
}

// In-place operation kernel name classes
struct AddInplaceKernelFloat32 {};
struct AddInplaceKernelFloat64 {};
struct SubInplaceKernelFloat32 {};
struct SubInplaceKernelFloat64 {};
struct MulInplaceKernelFloat32 {};
struct MulInplaceKernelFloat64 {};
struct DivInplaceKernelFloat32 {};
struct DivInplaceKernelFloat64 {};

// In-place add kernel
auto add_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);

        queue.parallel_for<AddInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] += other_ptr[idx];
        }).wait();
    }
    else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);

        queue.parallel_for<AddInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] += other_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("add_inplace: unsupported dtype");
    }

    return inout;
}

// In-place sub kernel
auto sub_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);

        queue.parallel_for<SubInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] -= other_ptr[idx];
        }).wait();
    }
    else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);

        queue.parallel_for<SubInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] -= other_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("sub_inplace: unsupported dtype");
    }

    return inout;
}

// In-place mul kernel
auto mul_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);

        queue.parallel_for<MulInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] *= other_ptr[idx];
        }).wait();
    }
    else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);

        queue.parallel_for<MulInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] *= other_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("mul_inplace: unsupported dtype");
    }

    return inout;
}

// In-place div kernel
auto div_inplace_kernel(Tensor& inout, const Tensor& other, sycl::queue& queue) -> Tensor {
    const int64_t n = inout.numel();

    if (inout.dtype() == DType::Float32) {
        float* data = get_data_ptr<float>(inout);
        const float* other_ptr = get_data_ptr<const float>(other);

        queue.parallel_for<DivInplaceKernelFloat32>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] /= other_ptr[idx];
        }).wait();
    }
    else if (inout.dtype() == DType::Float64) {
        double* data = get_data_ptr<double>(inout);
        const double* other_ptr = get_data_ptr<const double>(other);

        queue.parallel_for<DivInplaceKernelFloat64>(sycl::range<1>(n), [=](sycl::id<1> idx) {
            data[idx] /= other_ptr[idx];
        }).wait();
    }
    else {
        throw std::runtime_error("div_inplace: unsupported dtype");
    }

    return inout;
}

} // namespace oneapi
} // namespace tenzor
