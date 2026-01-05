#include "tenzor/ops/math.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstring>
#include <stdexcept>
#include <iostream>
#include <vector>
#ifdef _OPENMP
#include <omp.h>
#endif

// Intel MKL for optimized BLAS operations
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#endif

namespace tenzor {

// Math operation implementations - dispatched to backend kernels

auto add(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    // Permute and reshape can create non-contiguous views
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Add>(inputs)[0];
}

auto sub(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Sub>(inputs)[0];
}

auto mul(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Mul>(inputs)[0];
}

auto div(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise operation
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Div>(inputs)[0];
}

auto matmul(const Tensor& a, const Tensor& b) -> Tensor {
    // Handle batched matrix multiplication (3D+ tensors)
    if (a.shape().size() >= 3 && b.shape().size() >= 3) {
        // Both are batched - use bmm
        return bmm(a, b);
    }
    // Standard 2D matmul
    std::vector<Tensor> inputs = {a, b};
    return dispatch<OpId::MatMul>(inputs)[0];
}

auto bmm(const Tensor& a, const Tensor& b) -> Tensor {
    // Validate inputs are 3D
    if (a.shape().size() != 3 || b.shape().size() != 3) {
        throw std::runtime_error(
            "bmm requires 3D tensors, got shapes: [" +
            std::to_string(a.shape().size()) + "D] and [" +
            std::to_string(b.shape().size()) + "D]");
    }

    int64_t batch_size = a.shape()[0];
    int64_t M = a.shape()[1];  // rows of A
    int64_t K = a.shape()[2];  // cols of A = rows of B

    if (b.shape()[0] != batch_size || b.shape()[1] != K) {
        throw std::runtime_error(
            "bmm dimension mismatch: expected b.shape=[" +
            std::to_string(batch_size) + ", " + std::to_string(K) + ", *], got [" +
            std::to_string(b.shape()[0]) + ", " + std::to_string(b.shape()[1]) + ", " +
            std::to_string(b.shape()[2]) + "]");
    }

    int64_t N = b.shape()[2];  // cols of B

    // Validate dtype support
    if (a.dtype() != DType::Float16 && a.dtype() != DType::Float32 && a.dtype() != DType::Float64) {
        throw std::runtime_error(
            "bmm currently only supports Float16, Float32, and Float64 dtypes, got: " +
            std::to_string(static_cast<int>(a.dtype())));
    }

    // For non-CPU devices, use the slice/reshape approach for autograd compatibility
    if (a.device().type != Device::Type::CPU) {
        std::vector<Tensor> batch_results;
        batch_results.reserve(batch_size);

        for (int64_t batch = 0; batch < batch_size; ++batch) {
            Tensor a_slice = slice(a, 0, batch, batch + 1);
            Tensor a_batch = reshape(a_slice, {M, K});
            Tensor b_slice = slice(b, 0, batch, batch + 1);
            Tensor b_batch = reshape(b_slice, {K, N});
            batch_results.push_back(matmul(a_batch, b_batch));
        }
        return stack(batch_results, 0);
    }

    // Optimized CPU path: direct memory access with OpenMP parallelization
    // Make inputs contiguous
    Tensor a_cont = a.is_contiguous() ? a : a.contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();

    // Create output tensor
    Tensor output = zeros({batch_size, M, N}, a.dtype(), Device::cpu());

    // Batch strides
    int64_t a_batch_stride = M * K;
    int64_t b_batch_stride = K * N;
    int64_t c_batch_stride = M * N;

    if (a.dtype() == DType::Float32) {
        const float* a_data = a_cont.data<float>();
        const float* b_data = b_cont.data<float>();
        float* c_data = output.data<float>();

        // Fallback: Parallelize across batches with naive GEMM
        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const float* a_batch = a_data + batch * a_batch_stride;
            const float* b_batch = b_data + batch * b_batch_stride;
            float* c_batch = c_data + batch * c_batch_stride;

            std::memset(c_batch, 0, c_batch_stride * sizeof(float));

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t k = 0; k < K; ++k) {
                    float a_ik = a_batch[i * K + k];
                    for (int64_t j = 0; j < N; ++j) {
                        c_batch[i * N + j] += a_ik * b_batch[k * N + j];
                    }
                }
            }
        }
    } else if (a.dtype() == DType::Float64) {
        const double* a_data = a_cont.data<double>();
        const double* b_data = b_cont.data<double>();
        double* c_data = output.data<double>();

        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const double* a_batch = a_data + batch * a_batch_stride;
            const double* b_batch = b_data + batch * b_batch_stride;
            double* c_batch = c_data + batch * c_batch_stride;

            std::memset(c_batch, 0, c_batch_stride * sizeof(double));

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t k = 0; k < K; ++k) {
                    double a_ik = a_batch[i * K + k];
                    for (int64_t j = 0; j < N; ++j) {
                        c_batch[i * N + j] += a_ik * b_batch[k * N + j];
                    }
                }
            }
        }
    } else if (a.dtype() == DType::Float16) {
        const Float16* a_data = a_cont.data<Float16>();
        const Float16* b_data = b_cont.data<Float16>();
        Float16* c_data = output.data<Float16>();

        #pragma omp parallel for if(batch_size > 1)
        for (int64_t batch = 0; batch < batch_size; ++batch) {
            const Float16* a_batch = a_data + batch * a_batch_stride;
            const Float16* b_batch = b_data + batch * b_batch_stride;
            Float16* c_batch = c_data + batch * c_batch_stride;

            // Use float accumulator for FP16
            std::vector<float> c_fp32(c_batch_stride, 0.0f);

            for (int64_t i = 0; i < M; ++i) {
                for (int64_t k = 0; k < K; ++k) {
                    float a_ik = static_cast<float>(a_batch[i * K + k]);
                    for (int64_t j = 0; j < N; ++j) {
                        c_fp32[i * N + j] += a_ik * static_cast<float>(b_batch[k * N + j]);
                    }
                }
            }

            // Convert back to FP16
            for (int64_t i = 0; i < c_batch_stride; ++i) {
                c_batch[i] = Float16(c_fp32[i]);
            }
        }
    }

    return output;
}

auto dot(const Tensor& a, const Tensor& b) -> Tensor {
    std::vector<Tensor> inputs = {a, b};
    return dispatch<OpId::Dot>(inputs)[0];
}

auto pow(const Tensor& input, float exponent) -> Tensor {
    OpAttributes attrs;
    // Use scientific notation to preserve precision
    char exp_buf[32];
    snprintf(exp_buf, sizeof(exp_buf), "%.9e", exponent);
    attrs["exponent"] = std::string(exp_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Pow, inputs, attrs)[0];
}

auto exp(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Exp>(inputs)[0];
}

auto log(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Log>(inputs)[0];
}

auto sqrt(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sqrt>(inputs)[0];
}

auto sin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sin>(inputs)[0];
}

auto cos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Cos>(inputs)[0];
}

auto tan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Tan>(inputs)[0];
}

auto tanh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Tanh>(inputs)[0];
}

auto abs(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Abs>(inputs)[0];
}

auto neg(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Neg>(inputs)[0];
}

auto reciprocal(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Reciprocal>(inputs)[0];
}

auto sign(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sign>(inputs)[0];
}

auto sigmoid(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sigmoid>(inputs)[0];
}

auto minimum(const Tensor& a, const Tensor& b) -> Tensor {
    // minimum(a, b) = (a + b - abs(a - b)) / 2
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    auto sum_ab = add(a_contiguous, b_contiguous);
    auto diff_ab = sub(a_contiguous, b_contiguous);
    auto abs_diff = abs(diff_ab);
    auto numerator = sub(sum_ab, abs_diff);
    // Divide by 2 using scalar multiplication
    auto shape_span = numerator.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    auto half = full(shape_vec, 0.5f, numerator.dtype(), numerator.device());
    return mul(numerator, half);
}

auto maximum(const Tensor& a, const Tensor& b) -> Tensor {
    // maximum(a, b) = (a + b + abs(a - b)) / 2
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    auto sum_ab = add(a_contiguous, b_contiguous);
    auto diff_ab = sub(a_contiguous, b_contiguous);
    auto abs_diff = abs(diff_ab);
    auto numerator = add(sum_ab, abs_diff);
    // Divide by 2 using scalar multiplication
    auto shape_span = numerator.shape();
    std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
    auto half = full(shape_vec, 0.5f, numerator.dtype(), numerator.device());
    return mul(numerator, half);
}

auto floor(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Floor>(inputs)[0];
}

auto ceil(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Ceil>(inputs)[0];
}

auto round(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Round>(inputs)[0];
}

auto clamp(const Tensor& input, float min, float max) -> Tensor {
    OpAttributes attrs;
    // Use scientific notation to preserve precision
    char min_buf[32], max_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9e", min);
    snprintf(max_buf, sizeof(max_buf), "%.9e", max);
    attrs["min"] = std::string(min_buf);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::Clamp, inputs, attrs)[0];
}

auto clamp_min(const Tensor& input, float min) -> Tensor {
    OpAttributes attrs;
    char min_buf[32];
    snprintf(min_buf, sizeof(min_buf), "%.9e", min);
    attrs["min"] = std::string(min_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMin, inputs, attrs)[0];
}

auto clamp_max(const Tensor& input, float max) -> Tensor {
    OpAttributes attrs;
    char max_buf[32];
    snprintf(max_buf, sizeof(max_buf), "%.9e", max);
    attrs["max"] = std::string(max_buf);
    std::vector<Tensor> inputs = {input};
    return dispatch(OpId::ClampMax, inputs, attrs)[0];
}

auto sinh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Sinh>(inputs)[0];
}

auto cosh(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Cosh>(inputs)[0];
}

auto atan(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Atan>(inputs)[0];
}

auto asin(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Asin>(inputs)[0];
}

auto acos(const Tensor& input) -> Tensor {
    std::vector<Tensor> inputs = {input};
    return dispatch<OpId::Acos>(inputs)[0];
}

// Comparison operations
auto eq(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Eq>(inputs)[0];
}

auto ne(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Ne>(inputs)[0];
}

auto lt(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Lt>(inputs)[0];
}

auto le(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Le>(inputs)[0];
}

auto gt(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Gt>(inputs)[0];
}

auto ge(const Tensor& a, const Tensor& b) -> Tensor {
    // Ensure tensors are contiguous before element-wise comparison
    Tensor a_contiguous = a.is_contiguous() ? a : a.contiguous();
    Tensor b_contiguous = b.is_contiguous() ? b : b.contiguous();
    std::vector<Tensor> inputs = {a_contiguous, b_contiguous};
    return dispatch<OpId::Ge>(inputs)[0];
}

// In-place operations
auto add_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place add requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = dispatch<OpId::AddInplace>(inputs);

    // Result should be same tensor modified in-place
    // Copy result data back to self if backend created new tensor
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

auto mul_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place mul requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = dispatch<OpId::MulInplace>(inputs);

    // Result should be same tensor modified in-place
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

auto sub_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place sub requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = dispatch<OpId::SubInplace>(inputs);

    // Result should be same tensor modified in-place
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

auto div_(Tensor& self, const Tensor& other) -> Tensor& {
    // Ensure self is contiguous for in-place modification
    if (!self.is_contiguous()) {
        throw std::runtime_error("In-place div requires contiguous tensor");
    }

    Tensor other_contiguous = other.is_contiguous() ? other : other.contiguous();
    std::vector<Tensor> inputs = {self, other_contiguous};

    // Dispatch to backend in-place operation
    auto result = dispatch<OpId::DivInplace>(inputs);

    // Result should be same tensor modified in-place
    if (result[0].data<float>() != self.data<float>()) {
        self = result[0];
    }

    return self;
}

} // namespace tenzor
