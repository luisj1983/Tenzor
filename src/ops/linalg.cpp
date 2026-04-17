#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#ifdef TENZOR_USE_MKL
#include <mkl.h>
#include <mkl_lapacke.h>
#elif defined(TENZOR_USE_LAPACKE)
#include <lapacke.h>
#endif
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <functional>
#include <limits>
#include <vector>
#include <algorithm>

namespace tenzor::linalg {

namespace {

/// Try GPU dispatch for a single-output linalg op. Returns true and sets result on success.
/// CPU linalg ops are registered in the CPU dispatch table (cpu_kernel_registry.cpp) as
/// wrappers around the LAPACKE implementations below. GPU backends (CUDA/ROCm/Vulkan/OneAPI)
/// register native kernels.
/// Note: The CPU dispatch table wrappers call these functions, which check device==CPU
/// and skip try_gpu_dispatch, so there is no circular dispatch.
bool try_gpu_dispatch(OpId op, std::span<const Tensor> inputs,
                      const OpAttributes& attrs, Tensor& result) {
    if (inputs[0].device().type == Device::Type::CPU) return false;
    result = dispatch_single(op, inputs, attrs);
    return true;
}

/// Try GPU dispatch for a multi-output linalg op. Returns true and sets results on success.
bool try_gpu_dispatch_multi(OpId op, std::span<const Tensor> inputs,
                            const OpAttributes& attrs, std::vector<Tensor>& results) {
    if (inputs[0].device().type == Device::Type::CPU) return false;
    results = dispatch(op, inputs, attrs);
    return true;
}

// Check if dtype is a low-precision float that needs upcasting for LAPACK
bool needs_upcast(DType dt) {
    return dt == DType::Float16 || dt == DType::BFloat16;
}

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
[[noreturn]] void throw_no_lapack(const char* fn) {
    throw std::runtime_error(
        std::string("linalg::") + fn + ": requires MKL or LAPACKE. "
        "Build with TENZOR_USE_MKL or install liblapacke-dev.");
}
#endif

// Ensure tensor is contiguous Float32 or Float64 on CPU, return a working copy.
// Float16 and BFloat16 inputs are upcast to Float32.
// Only called for CPU tensors — GPU tensors are handled by try_gpu_dispatch.
auto prepare_matrix(const Tensor& A) -> Tensor {
    if (A.device().type != Device::Type::CPU) {
        throw std::logic_error("prepare_matrix: expected CPU tensor, got GPU tensor. "
                               "GPU linalg ops should go through backend dispatch.");
    }
    auto dt = A.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16) {
        throw std::runtime_error("linalg: only Float32, Float64, Float16, and BFloat16 supported");
    }
    // Upcast low-precision floats to Float32 for LAPACK compatibility
    auto cpu_tensor = A;
    if (needs_upcast(cpu_tensor.dtype())) {
        cpu_tensor = cpu_tensor.to(DType::Float32);
    }
    return cpu_tensor.contiguous().clone();
}

// Downcast result tensor back to original dtype if it was upcast
auto maybe_downcast(const Tensor& result, DType original_dtype) -> Tensor {
    if (needs_upcast(original_dtype) && result.dtype() != original_dtype) {
        return result.to(original_dtype);
    }
    return result;
}

auto check_square(const Tensor& A) -> std::pair<int64_t, int64_t> {
    auto shape = A.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("linalg: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    if (m != n) {
        throw std::invalid_argument("linalg: expected square matrix, got " +
            std::to_string(m) + "x" + std::to_string(n));
    }
    return {m, ndim};
}

// Compute batch size (product of all dims except last two)
auto batch_size(const Tensor& A) -> int64_t {
    auto shape = A.shape();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= shape[i];
    }
    return batch;
}

} // anonymous namespace

auto det(const Tensor& A) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch(OpId::LinalgDet, inputs, {}, result)) return result;
    }
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("det");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    // Output shape: all dims except last two
    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        out_shape.push_back(shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = zeros(out_shape, work.dtype(), Device::cpu());

    std::vector<lapack_int> ipiv(n);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* res_data = result.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::det: invalid argument");

            float d = 1.0f;
            for (int64_t i = 0; i < n; ++i) {
                d *= mat[i * n + i];
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) d = -d;
            }
            res_data[b] = d;
        }
    } else {
        double* data = work.data<double>();
        double* res_data = result.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::det: invalid argument");

            double d = 1.0;
            for (int64_t i = 0; i < n; ++i) {
                d *= mat[i * n + i];
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) d = -d;
            }
            res_data[b] = d;
        }
    }

    return maybe_downcast(result, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto inv(const Tensor& A) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch(OpId::LinalgInv, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("inv");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<lapack_int> ipiv(n);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: LU factorization failed (singular matrix)");

            info = LAPACKE_sgetri(LAPACK_ROW_MAJOR, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: inversion failed");
        }
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, ln, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: LU factorization failed (singular matrix)");

            info = LAPACKE_dgetri(LAPACK_ROW_MAJOR, ln, mat, ln, ipiv.data());
            if (info != 0) throw std::runtime_error("linalg::inv: inversion failed");
        }
    }

    return maybe_downcast(work, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto solve(const Tensor& A, const Tensor& B) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 2> inputs = {A, B};
        if (try_gpu_dispatch(OpId::LinalgSolve, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("solve");
#else
    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve: B must be at least 1D");

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    std::vector<lapack_int> ipiv(n);

    if (work_a.dtype() == DType::Float32) {
        float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * n * n;
            float* b_mat = b_data + b * n * nrhs;
            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);

            lapack_int info = LAPACKE_sgesv(LAPACK_ROW_MAJOR, ln, lnrhs,
                a_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::solve: solution failed");
        }
    } else {
        double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * n * n;
            double* b_mat = b_data + b * n * nrhs;
            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);

            lapack_int info = LAPACKE_dgesv(LAPACK_ROW_MAJOR, ln, lnrhs,
                a_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::solve: solution failed");
        }
    }

    return maybe_downcast(work_b, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto solve_triangular(const Tensor& A, const Tensor& B, bool upper, bool unitriangular) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 2> inputs = {A, B};
        OpAttributes attrs;
        attrs.set(AttrKey::Upper, upper);
        attrs.set(AttrKey::UnitTriangular, unitriangular);
        if (try_gpu_dispatch(OpId::SolveTriangular, inputs, attrs, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    // Fallback: forward/back substitution without LAPACK
    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve_triangular: B must be at least 1D");

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    if (work_a.dtype() == DType::Float32) {
        const float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            const float* A_mat = a_data + batch * n * n;
            float* X_mat = b_data + batch * n * nrhs;
            if (upper) {
                for (int64_t i = n - 1; i >= 0; --i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        float sum = X_mat[i * nrhs + j];
                        for (int64_t k = i + 1; k < n; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        float sum = X_mat[i * nrhs + j];
                        for (int64_t k = 0; k < i; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            }
        }
    } else {
        const double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            const double* A_mat = a_data + batch * n * n;
            double* X_mat = b_data + batch * n * nrhs;
            if (upper) {
                for (int64_t i = n - 1; i >= 0; --i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        double sum = X_mat[i * nrhs + j];
                        for (int64_t k = i + 1; k < n; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            } else {
                for (int64_t i = 0; i < n; ++i) {
                    for (int64_t j = 0; j < nrhs; ++j) {
                        double sum = X_mat[i * nrhs + j];
                        for (int64_t k = 0; k < i; ++k)
                            sum -= A_mat[i * n + k] * X_mat[k * nrhs + j];
                        X_mat[i * nrhs + j] = unitriangular ? sum : sum / A_mat[i * n + i];
                    }
                }
            }
        }
    }
    return maybe_downcast(work_b, original_dtype);
#else
    // Use CBLAS trsm for triangular solve
    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve_triangular: B must be at least 1D");

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    auto uplo = upper ? CblasUpper : CblasLower;
    auto diag = unitriangular ? CblasUnit : CblasNonUnit;
    auto ln = static_cast<int>(n);
    auto lnrhs = static_cast<int>(nrhs);

    if (work_a.dtype() == DType::Float32) {
        const float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            cblas_strsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                        ln, lnrhs, 1.0f,
                        a_data + batch * n * n, ln,
                        b_data + batch * n * nrhs, lnrhs);
        }
    } else {
        const double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();
        for (int64_t batch = 0; batch < nbatch; ++batch) {
            cblas_dtrsm(CblasRowMajor, CblasLeft, uplo, CblasNoTrans, diag,
                        ln, lnrhs, 1.0,
                        a_data + batch * n * n, ln,
                        b_data + batch * n * nrhs, lnrhs);
        }
    }

    return maybe_downcast(work_b, original_dtype);
#endif
}

auto cholesky(const Tensor& A, bool upper) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        OpAttributes attrs;
        attrs.set(AttrKey::Upper, upper);
        if (try_gpu_dispatch(OpId::LinalgCholesky, inputs, attrs, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("cholesky");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    char uplo = upper ? 'U' : 'L';

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_spotrf(LAPACK_ROW_MAJOR, uplo, ln, mat, ln);
            if (info != 0) throw std::runtime_error("linalg::cholesky: factorization failed (not positive definite)");

            // Zero out the other triangle
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (upper ? (i > j) : (i < j)) {
                        mat[i * n + j] = 0.0f;
                    }
                }
            }
        }
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dpotrf(LAPACK_ROW_MAJOR, uplo, ln, mat, ln);
            if (info != 0) throw std::runtime_error("linalg::cholesky: factorization failed (not positive definite)");

            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (upper ? (i > j) : (i < j)) {
                        mat[i * n + j] = 0.0;
                    }
                }
            }
        }
    }

    return maybe_downcast(work, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto norm(const Tensor& A, const std::string& ord) -> Tensor {
    if (A.device().type != Device::Type::CPU) {
        throw std::runtime_error("linalg::norm: only CPU tensors supported");
    }

    if (ord == "fro") {
        // Frobenius norm: sqrt(sum(x^2))
        auto flat = A.contiguous().reshape({A.numel()});
        auto sq = flat * flat;
        auto s = tenzor::sum(sq);
        return tenzor::sqrt(s);
    }

    throw std::runtime_error("linalg::norm: unsupported norm order '" + ord + "' (supported: 'fro')");
}

auto slogdet(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // GPU path: derive slogdet from det (which dispatches to backend kernels).
    // sign = sign(det), logabsdet = log(|det|). This avoids a dedicated
    // slogdet OpId + per-backend kernels while still running on-device.
    if (A.device().type != Device::Type::CPU) {
        Tensor d = tenzor::linalg::det(A);
        Tensor sign_t = tenzor::sign(d);
        Tensor logabsdet_t = tenzor::log(tenzor::abs(d));
        return {sign_t, logabsdet_t};
    }
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("slogdet");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        out_shape.push_back(shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto sign_result = zeros(out_shape, work.dtype(), Device::cpu());
    auto logabsdet_result = zeros(out_shape, work.dtype(), Device::cpu());

    std::vector<lapack_int> ipiv(n);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* sign_data = sign_result.data<float>();
        float* logabs_data = logabsdet_result.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::slogdet: invalid argument");

            float sign = 1.0f;
            float logabsdet = 0.0f;
            bool is_zero = false;
            for (int64_t i = 0; i < n; ++i) {
                float diag = mat[i * n + i];
                if (diag == 0.0f) { is_zero = true; break; }
                if (diag < 0.0f) { sign = -sign; diag = -diag; }
                logabsdet += std::log(diag);
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) sign = -sign;
            }
            if (is_zero) {
                sign_data[b] = 0.0f;
                logabs_data[b] = -std::numeric_limits<float>::infinity();
            } else {
                sign_data[b] = sign;
                logabs_data[b] = logabsdet;
            }
        }
    } else {
        double* data = work.data<double>();
        double* sign_data = sign_result.data<double>();
        double* logabs_data = logabsdet_result.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR,
                static_cast<lapack_int>(n), static_cast<lapack_int>(n),
                mat, static_cast<lapack_int>(n), ipiv.data());

            if (info < 0) throw std::runtime_error("linalg::slogdet: invalid argument");

            double sign = 1.0;
            double logabsdet = 0.0;
            bool is_zero = false;
            for (int64_t i = 0; i < n; ++i) {
                double diag = mat[i * n + i];
                if (diag == 0.0) { is_zero = true; break; }
                if (diag < 0.0) { sign = -sign; diag = -diag; }
                logabsdet += std::log(diag);
                if (ipiv[i] != static_cast<lapack_int>(i + 1)) sign = -sign;
            }
            if (is_zero) {
                sign_data[b] = 0.0;
                logabs_data[b] = -std::numeric_limits<double>::infinity();
            } else {
                sign_data[b] = sign;
                logabs_data[b] = logabsdet;
            }
        }
    }

    return {maybe_downcast(sign_result, original_dtype),
            maybe_downcast(logabsdet_result, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto svd(const Tensor& A, bool full_matrices) -> std::tuple<Tensor, Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        OpAttributes attrs;
        attrs.set(AttrKey::FullMatrices, full_matrices);
        if (try_gpu_dispatch_multi(OpId::LinalgSVD, inputs, attrs, results)) {
            return {results[0], results[1], results[2]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("svd");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::svd: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    // Batch dims
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Output shapes
    std::vector<int64_t> u_shape = batch_dims;
    std::vector<int64_t> s_shape = batch_dims;
    std::vector<int64_t> vt_shape = batch_dims;

    s_shape.push_back(k);
    if (full_matrices) {
        u_shape.push_back(m); u_shape.push_back(m);
        vt_shape.push_back(n_cols); vt_shape.push_back(n_cols);
    } else {
        u_shape.push_back(m); u_shape.push_back(k);
        vt_shape.push_back(k); vt_shape.push_back(n_cols);
    }

    auto U = zeros(u_shape, work.dtype(), Device::cpu());
    auto S = zeros(s_shape, work.dtype(), Device::cpu());
    auto Vt = zeros(vt_shape, work.dtype(), Device::cpu());

    char jobz = full_matrices ? 'A' : 'S';

    // superb array for ?gesvd
    int64_t superb_size = k - 1;
    if (superb_size < 1) superb_size = 1;

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* u_data = U.data<float>();
        float* s_data = S.data<float>();
        float* vt_data = Vt.data<float>();
        std::vector<float> superb(superb_size);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        // Row-major leading dimensions: ldu = #cols of U, ldvt = #cols of Vt
        auto ldu = full_matrices ? lm : static_cast<lapack_int>(k);
        auto ldvt = full_matrices ? ln : ln;
        int64_t u_stride = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * m * n_cols;
            float* u_mat = u_data + b * u_stride;
            float* s_vec = s_data + b * k;
            float* vt_mat = vt_data + b * vt_stride;

            lapack_int info = LAPACKE_sgesvd(LAPACK_ROW_MAJOR, jobz, jobz,
                lm, ln, a_mat, ln, s_vec, u_mat, ldu, vt_mat, ldvt, superb.data());
            if (info != 0) throw std::runtime_error("linalg::svd: computation failed (info=" + std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* u_data = U.data<double>();
        double* s_data = S.data<double>();
        double* vt_data = Vt.data<double>();
        std::vector<double> superb(superb_size);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto ldu = full_matrices ? lm : static_cast<lapack_int>(k);
        auto ldvt = full_matrices ? ln : ln;
        int64_t u_stride = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * m * n_cols;
            double* u_mat = u_data + b * u_stride;
            double* s_vec = s_data + b * k;
            double* vt_mat = vt_data + b * vt_stride;

            lapack_int info = LAPACKE_dgesvd(LAPACK_ROW_MAJOR, jobz, jobz,
                lm, ln, a_mat, ln, s_vec, u_mat, ldu, vt_mat, ldvt, superb.data());
            if (info != 0) throw std::runtime_error("linalg::svd: computation failed (info=" + std::to_string(info) + ")");
        }
    }

    return {maybe_downcast(U, original_dtype),
            maybe_downcast(S, original_dtype),
            maybe_downcast(Vt, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto qr(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgQR, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("qr");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::qr: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Q is (m, k), R is (k, n)
    std::vector<int64_t> q_shape = batch_dims;
    q_shape.push_back(m); q_shape.push_back(k);
    std::vector<int64_t> r_shape = batch_dims;
    r_shape.push_back(k); r_shape.push_back(n_cols);

    auto Q = zeros(q_shape, work.dtype(), Device::cpu());
    auto R = zeros(r_shape, work.dtype(), Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* q_data = Q.data<float>();
        float* r_data = R.data<float>();
        std::vector<float> tau(k);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * m * n_cols;
            float* q_mat = q_data + b * m * k;
            float* r_mat = r_data + b * k * n_cols;

            // Compute QR factorization in-place
            lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: factorization failed");

            // Extract R (upper triangle of A)
            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < n_cols; ++j) {
                    r_mat[i * n_cols + j] = (j >= i) ? a_mat[i * n_cols + j] : 0.0f;
                }
            }

            // Generate Q from Householder reflectors
            info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, lm, static_cast<lapack_int>(k),
                static_cast<lapack_int>(k), a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: Q generation failed");

            // Copy Q columns
            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    q_mat[i * k + j] = a_mat[i * n_cols + j];
                }
            }
        }
    } else {
        double* a_data = work.data<double>();
        double* q_data = Q.data<double>();
        double* r_data = R.data<double>();
        std::vector<double> tau(k);

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * m * n_cols;
            double* q_mat = q_data + b * m * k;
            double* r_mat = r_data + b * k * n_cols;

            lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: factorization failed");

            for (int64_t i = 0; i < k; ++i) {
                for (int64_t j = 0; j < n_cols; ++j) {
                    r_mat[i * n_cols + j] = (j >= i) ? a_mat[i * n_cols + j] : 0.0;
                }
            }

            info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, lm, static_cast<lapack_int>(k),
                static_cast<lapack_int>(k), a_mat, ln, tau.data());
            if (info != 0) throw std::runtime_error("linalg::qr: Q generation failed");

            for (int64_t i = 0; i < m; ++i) {
                for (int64_t j = 0; j < k; ++j) {
                    q_mat[i * k + j] = a_mat[i * n_cols + j];
                }
            }
        }
    }

    return {maybe_downcast(Q, original_dtype), maybe_downcast(R, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto eigh(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgEigh, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("eigh");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Eigenvalues shape: (..., N)
    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, work.dtype(), Device::cpu());

    // Eigenvectors are stored in work (overwritten by dsyev/ssyev)

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* w_data = W.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = a_data + b * n * n;
            float* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'V', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigh: computation failed (info=" + std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* w_data = W.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = a_data + b * n * n;
            double* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'V', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigh: computation failed (info=" + std::to_string(info) + ")");
        }
    }

    // work now contains eigenvectors (columns of orthogonal matrix)
    return {maybe_downcast(W, original_dtype), maybe_downcast(work, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto eigvalsh(const Tensor& A) -> Tensor {
    // Try GPU dispatch via eigh (compute eigenvalues + eigenvectors, return only eigenvalues)
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgEigh, inputs, {}, results)) {
            return results[0];  // eigenvalues
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("eigvalsh");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, work.dtype(), Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* w_data = W.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = a_data + b * n * n;
            float* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_ssyev(LAPACK_ROW_MAJOR, 'N', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigvalsh: computation failed");
        }
    } else {
        double* a_data = work.data<double>();
        double* w_data = W.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = a_data + b * n * n;
            double* w_vec = w_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dsyev(LAPACK_ROW_MAJOR, 'N', 'U', ln, mat, ln, w_vec);
            if (info != 0) throw std::runtime_error("linalg::eigvalsh: computation failed");
        }
    }

    return maybe_downcast(W, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto eig(const Tensor& A) -> std::tuple<Tensor, Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgEig, inputs, {}, results)) {
            return {results[0], results[1], results[2]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("eig");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Eigenvalues shape: (..., N) — real and imaginary parts separately
    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);

    // Eigenvectors shape: (..., N, N)
    std::vector<int64_t> v_shape = batch_dims;
    v_shape.push_back(n);
    v_shape.push_back(n);

    auto Wr = zeros(w_shape, work.dtype(), Device::cpu());  // real part of eigenvalues
    auto Wi = zeros(w_shape, work.dtype(), Device::cpu());  // imaginary part of eigenvalues
    auto Vr = zeros(v_shape, work.dtype(), Device::cpu());  // right eigenvectors

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* wr_data = Wr.data<float>();
        float* wi_data = Wi.data<float>();
        float* vr_data = Vr.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = a_data + b * n * n;
            float* wr_vec = wr_data + b * n;
            float* wi_vec = wi_data + b * n;
            float* vr_mat = vr_data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            // Compute eigenvalues and right eigenvectors (no left eigenvectors)
            lapack_int info = LAPACKE_sgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                ln, mat, ln, wr_vec, wi_vec,
                nullptr, ln, vr_mat, ln);
            if (info != 0) {
                throw std::runtime_error("linalg::eig: computation failed (info=" +
                    std::to_string(info) + ")");
            }
        }
    } else {
        double* a_data = work.data<double>();
        double* wr_data = Wr.data<double>();
        double* wi_data = Wi.data<double>();
        double* vr_data = Vr.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = a_data + b * n * n;
            double* wr_vec = wr_data + b * n;
            double* wi_vec = wi_data + b * n;
            double* vr_mat = vr_data + b * n * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dgeev(LAPACK_ROW_MAJOR, 'N', 'V',
                ln, mat, ln, wr_vec, wi_vec,
                nullptr, ln, vr_mat, ln);
            if (info != 0) {
                throw std::runtime_error("linalg::eig: computation failed (info=" +
                    std::to_string(info) + ")");
            }
        }
    }

    return {maybe_downcast(Wr, original_dtype),
            maybe_downcast(Wi, original_dtype),
            maybe_downcast(Vr, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto matrix_power(const Tensor& A, int64_t n) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::matrix_power: expected at least 2D tensor");
    }
    auto shape = A.shape();
    int64_t rows = shape[A.ndim() - 2];
    int64_t cols = shape[A.ndim() - 1];
    if (rows != cols) {
        throw std::invalid_argument("linalg::matrix_power: matrix must be square, got " +
            std::to_string(rows) + "x" + std::to_string(cols));
    }

    if (n == 0) {
        // A^0 = identity matrix (batched if needed)
        return tenzor::eye(rows, std::nullopt, A.dtype(), A.device());
    }

    // For negative exponents, invert first then exponentiate
    Tensor base = (n < 0) ? inv(A) : A;
    int64_t exp = std::abs(n);

    // Binary exponentiation: O(log n) matmuls
    Tensor result = base;
    exp--;
    while (exp > 0) {
        if (exp & 1) {
            result = tenzor::matmul(result, base);
        }
        base = tenzor::matmul(base, base);
        exp >>= 1;
    }

    return result;
}

auto lu(const Tensor& A) -> std::tuple<Tensor, Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgLU, inputs, {}, results)) {
            return {results[0], results[1], results[2]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("lu");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto shape = A.shape();
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> mat_shape = batch_dims;
    mat_shape.push_back(n); mat_shape.push_back(n);

    std::vector<int64_t> piv_shape = batch_dims;
    piv_shape.push_back(n);

    auto L = zeros(mat_shape, work.dtype(), Device::cpu());
    auto U = zeros(mat_shape, work.dtype(), Device::cpu());
    auto pivots_out = zeros(piv_shape, DType::Int32, Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* l_data = L.data<float>();
        float* u_data = U.data<float>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * n * n;
            float* l_mat = l_data + b * n * n;
            float* u_mat = u_data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_sgetrf(LAPACK_ROW_MAJOR, ln, ln, a_mat, ln, ipiv.data());
            if (info < 0) throw std::runtime_error("linalg::lu: invalid argument " + std::to_string(-info));

            // Extract L (unit lower triangular) and U (upper triangular) from packed result
            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (i > j) {
                        l_mat[i * n + j] = a_mat[i * n + j];
                        u_mat[i * n + j] = 0.0f;
                    } else if (i == j) {
                        l_mat[i * n + j] = 1.0f;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    } else {
                        l_mat[i * n + j] = 0.0f;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    }
                }
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
            }
        }
    } else {
        double* a_data = work.data<double>();
        double* l_data = L.data<double>();
        double* u_data = U.data<double>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * n * n;
            double* l_mat = l_data + b * n * n;
            double* u_mat = u_data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dgetrf(LAPACK_ROW_MAJOR, ln, ln, a_mat, ln, ipiv.data());
            if (info < 0) throw std::runtime_error("linalg::lu: invalid argument " + std::to_string(-info));

            for (int64_t i = 0; i < n; ++i) {
                for (int64_t j = 0; j < n; ++j) {
                    if (i > j) {
                        l_mat[i * n + j] = a_mat[i * n + j];
                        u_mat[i * n + j] = 0.0;
                    } else if (i == j) {
                        l_mat[i * n + j] = 1.0;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    } else {
                        l_mat[i * n + j] = 0.0;
                        u_mat[i * n + j] = a_mat[i * n + j];
                    }
                }
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
            }
        }
    }

    return {maybe_downcast(L, original_dtype), maybe_downcast(U, original_dtype), pivots_out};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto lu_solve(const Tensor& LU_data, const Tensor& pivots,
              const Tensor& B) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 3> inputs = {LU_data, pivots, B};
        if (try_gpu_dispatch(OpId::LinalgLUSolve, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("lu_solve");
#else
    auto original_dtype = B.dtype();
    auto work_lu = prepare_matrix(LU_data);
    auto work_b = prepare_matrix(B);

    auto lu_shape = LU_data.shape();
    auto b_shape = B.shape();
    auto lu_ndim = static_cast<int64_t>(lu_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (lu_ndim < 2 || b_ndim < 2)
        throw std::invalid_argument("linalg::lu_solve: inputs must be at least 2D");

    int64_t n = lu_shape[lu_ndim - 1];
    int64_t nrhs = b_shape[b_ndim - 1];
    int64_t nbatch = batch_size(work_lu);

    auto piv_cpu = pivots.to(Device::cpu()).contiguous();

    if (work_lu.dtype() == DType::Float32) {
        float* lu_ptr = work_lu.data<float>();
        float* b_ptr = work_b.data<float>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* lu_mat = lu_ptr + b * n * n;
            float* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_sgetrs(LAPACK_ROW_MAJOR, 'N', ln, lnrhs,
                lu_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::lu_solve: solve failed");
        }
    } else {
        double* lu_ptr = work_lu.data<double>();
        double* b_ptr = work_b.data<double>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* lu_mat = lu_ptr + b * n * n;
            double* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_dgetrs(LAPACK_ROW_MAJOR, 'N', ln, lnrhs,
                lu_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0) throw std::runtime_error("linalg::lu_solve: solve failed");
        }
    }

    return maybe_downcast(work_b, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

// ============================================================================
// lstsq / pinv / matrix_exp — higher-level linalg routines composed on top of
// the primitives above. These are CPU-only today; they fall back through the
// LAPACKE path and do not currently dispatch to GPU backends.
// ============================================================================

auto lstsq(const Tensor& A, const Tensor& B) -> std::tuple<Tensor, Tensor> {
#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("lstsq");
#else
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    if (a_shape.size() != 2) {
        throw std::invalid_argument("linalg::lstsq: A must be 2D (got ndim=" +
            std::to_string(a_shape.size()) + ")");
    }
    if (b_shape.size() != 1 && b_shape.size() != 2) {
        throw std::invalid_argument("linalg::lstsq: B must be 1D or 2D");
    }
    const int64_t m = a_shape[0];
    const int64_t n = a_shape[1];
    if (b_shape[0] != m) {
        throw std::invalid_argument("linalg::lstsq: A rows (" + std::to_string(m) +
            ") must match B rows (" + std::to_string(b_shape[0]) + ")");
    }

    // LAPACKE_?gels overwrites B in-place with the solution (top n rows) and
    // residuals (remaining max(m, n) - n rows, squared sum per column).
    // We need a workspace the size of max(m, n) rows × nrhs cols.
    const int64_t nrhs = (b_shape.size() == 2) ? b_shape[1] : 1;
    const int64_t ldb  = std::max(m, n);

    auto original_dtype = A.dtype();
    auto work_a = prepare_matrix(A);

    // Build B work buffer of shape [ldb, nrhs], copy B into the first m rows.
    const std::vector<int64_t> b_work_shape =
        (b_shape.size() == 2) ? std::vector<int64_t>{ldb, nrhs}
                              : std::vector<int64_t>{ldb};
    auto work_b = zeros(b_work_shape, work_a.dtype(), Device::cpu());
    auto src_b = B;
    if (needs_upcast(src_b.dtype())) src_b = src_b.to(DType::Float32);
    src_b = src_b.contiguous();

    auto copy_b = [&](auto src_type, auto dst_type) {
        using SrcT [[maybe_unused]] = decltype(src_type);
        using DstT [[maybe_unused]] = decltype(dst_type);
    };
    (void)copy_b;

    auto ln   = static_cast<lapack_int>(n);
    auto lm   = static_cast<lapack_int>(m);
    auto lnrhs = static_cast<lapack_int>(nrhs);
    // In LAPACK_ROW_MAJOR, the "leading dimension" is the row stride, which
    // for a shape-(rows, cols) buffer equals `cols`. So ldb_lapack == nrhs
    // even though our physical buffer has `ldb = max(m, n)` rows. This is
    // the same convention used by linalg::solve at LAPACKE_sgesv above.
    auto lldb_lapack = static_cast<lapack_int>(nrhs);

    if (work_a.dtype() == DType::Float32) {
        float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();
        const float* src_ptr = src_b.data<float>();
        // Fill first m rows of work_b with B's contents (col-for-col).
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < nrhs; ++j) {
                b_data[i * nrhs + j] = src_ptr[i * nrhs + j];
            }
        }
        lapack_int info = LAPACKE_sgels(LAPACK_ROW_MAJOR, 'N', lm, ln, lnrhs,
            a_data, ln, b_data, lldb_lapack);
        if (info < 0) {
            throw std::runtime_error("linalg::lstsq: invalid LAPACKE argument");
        }
        if (info > 0) {
            throw std::runtime_error(
                "linalg::lstsq: A does not have full rank — the least-squares "
                "solution could not be computed (info=" + std::to_string(info) + ")");
        }
    } else {
        double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();
        const double* src_ptr = src_b.data<double>();
        for (int64_t i = 0; i < m; ++i) {
            for (int64_t j = 0; j < nrhs; ++j) {
                b_data[i * nrhs + j] = src_ptr[i * nrhs + j];
            }
        }
        lapack_int info = LAPACKE_dgels(LAPACK_ROW_MAJOR, 'N', lm, ln, lnrhs,
            a_data, ln, b_data, lldb_lapack);
        if (info < 0) {
            throw std::runtime_error("linalg::lstsq: invalid LAPACKE argument");
        }
        if (info > 0) {
            throw std::runtime_error(
                "linalg::lstsq: A does not have full rank — the least-squares "
                "solution could not be computed (info=" + std::to_string(info) + ")");
        }
    }

    // Extract solution (first n rows) and residuals (rows n..m when m > n).
    std::vector<int64_t> sol_shape;
    sol_shape.push_back(n);
    if (b_shape.size() == 2) sol_shape.push_back(nrhs);
    auto solution = zeros(sol_shape, work_a.dtype(), Device::cpu());

    // Residuals: for 2D B, shape (nrhs,); for 1D B, shape (1,) with a single
    // scalar value. Empty tensor if m <= n (underdetermined, no residuals).
    std::vector<int64_t> res_shape;
    if (m > n) {
        res_shape.push_back(nrhs);
    }
    auto residuals = res_shape.empty()
        ? zeros({0}, work_a.dtype(), Device::cpu())
        : zeros(res_shape, work_a.dtype(), Device::cpu());

    if (work_a.dtype() == DType::Float32) {
        const float* b_data = work_b.data<float>();
        float* sol = solution.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < nrhs; ++j) {
                sol[i * nrhs + j] = b_data[i * nrhs + j];
            }
        }
        if (m > n) {
            float* res = residuals.data<float>();
            for (int64_t j = 0; j < nrhs; ++j) {
                float sum = 0.0f;
                for (int64_t i = n; i < m; ++i) {
                    float v = b_data[i * nrhs + j];
                    sum += v * v;
                }
                res[j] = sum;
            }
        }
    } else {
        const double* b_data = work_b.data<double>();
        double* sol = solution.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < nrhs; ++j) {
                sol[i * nrhs + j] = b_data[i * nrhs + j];
            }
        }
        if (m > n) {
            double* res = residuals.data<double>();
            for (int64_t j = 0; j < nrhs; ++j) {
                double sum = 0.0;
                for (int64_t i = n; i < m; ++i) {
                    double v = b_data[i * nrhs + j];
                    sum += v * v;
                }
                res[j] = sum;
            }
        }
    }

    return {maybe_downcast(solution, original_dtype),
            maybe_downcast(residuals, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto pinv(const Tensor& A, double rcond) -> Tensor {
    // pinv(A) = V @ diag(s_inv) @ U^T, where s_inv[i] = 1/s[i] when
    // s[i] > rcond * max(s), else 0. Built on top of the existing svd().
    auto a_shape = A.shape();
    if (a_shape.size() != 2) {
        throw std::invalid_argument("linalg::pinv: A must be 2D (got ndim=" +
            std::to_string(a_shape.size()) + ")");
    }

    auto original_dtype = A.dtype();
    auto original_device = A.device();
    // Use reduced SVD (full_matrices=false): U is (M, K), S is (K,), Vt is (K, N).
    auto [U, S, Vt] = svd(A, /*full_matrices=*/false);

    const int64_t k = S.numel();
    if (k == 0) {
        // Degenerate case: return a zero (N, M) pseudoinverse on the input device.
        return zeros({a_shape[1], a_shape[0]}, original_dtype, original_device);
    }

    // Build a diagonal scaling vector: s_inv[i] = 1/s[i] if s[i] > cutoff, else 0.
    auto compute_dtype = S.dtype();  // Float32 or Float64 (maybe upcast from F16).

    // Extract max(s) to compute cutoff. Move S to CPU because the per-element
    // scan below uses raw host-pointer reads; previously this crashed for
    // non-CPU inputs (S would be a device-resident tensor).
    auto s_cpu = S.device().type == Device::Type::CPU
        ? S.contiguous() : S.to(Device::cpu()).contiguous();
    auto s_contig = s_cpu;
    double max_s = 0.0;
    if (compute_dtype == DType::Float32) {
        const float* sp = s_contig.data<float>();
        for (int64_t i = 0; i < k; ++i) {
            if (static_cast<double>(sp[i]) > max_s) max_s = sp[i];
        }
    } else {
        const double* sp = s_contig.data<double>();
        for (int64_t i = 0; i < k; ++i) {
            if (sp[i] > max_s) max_s = sp[i];
        }
    }
    const double cutoff = rcond * max_s;

    // Build s_inv as a (k,) vector.
    auto s_inv = zeros({k}, compute_dtype, Device::cpu());
    if (compute_dtype == DType::Float32) {
        const float* sp = s_contig.data<float>();
        float* sip = s_inv.data<float>();
        for (int64_t i = 0; i < k; ++i) {
            sip[i] = (static_cast<double>(sp[i]) > cutoff) ? (1.0f / sp[i]) : 0.0f;
        }
    } else {
        const double* sp = s_contig.data<double>();
        double* sip = s_inv.data<double>();
        for (int64_t i = 0; i < k; ++i) {
            sip[i] = (sp[i] > cutoff) ? (1.0 / sp[i]) : 0.0;
        }
    }

    // pinv = Vt^T @ diag(s_inv) @ U^T
    // Compute V = Vt^T (shape (N, K)) and scale its columns by s_inv.
    // Vt.transpose(-2, -1) gives (N, K); then element-wise multiply by s_inv
    // broadcast along rows; then matmul with U^T (K, M).
    //
    // The assembly below uses raw host pointer loops, so we need CPU-resident
    // Vt/U data. Move them off-device if needed.
    auto Vt_contig = Vt.device().type == Device::Type::CPU
        ? Vt.contiguous() : Vt.to(Device::cpu()).contiguous();
    const int64_t n_cols = a_shape[1];
    const int64_t m_rows = a_shape[0];

    auto V = zeros({n_cols, k}, compute_dtype, Device::cpu());
    if (compute_dtype == DType::Float32) {
        const float* vt = Vt_contig.data<float>();
        float* v = V.data<float>();
        const float* sip = s_inv.data<float>();
        for (int64_t i = 0; i < n_cols; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                v[i * k + j] = vt[j * n_cols + i] * sip[j];
            }
        }
    } else {
        const double* vt = Vt_contig.data<double>();
        double* v = V.data<double>();
        const double* sip = s_inv.data<double>();
        for (int64_t i = 0; i < n_cols; ++i) {
            for (int64_t j = 0; j < k; ++j) {
                v[i * k + j] = vt[j * n_cols + i] * sip[j];
            }
        }
    }

    // UT = U^T, shape (K, M). Same host-pointer assumption as Vt above.
    auto U_contig = U.device().type == Device::Type::CPU
        ? U.contiguous() : U.to(Device::cpu()).contiguous();
    auto UT = zeros({k, m_rows}, compute_dtype, Device::cpu());
    if (compute_dtype == DType::Float32) {
        const float* u = U_contig.data<float>();
        float* ut = UT.data<float>();
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t j = 0; j < m_rows; ++j) {
                ut[i * m_rows + j] = u[j * k + i];
            }
        }
    } else {
        const double* u = U_contig.data<double>();
        double* ut = UT.data<double>();
        for (int64_t i = 0; i < k; ++i) {
            for (int64_t j = 0; j < m_rows; ++j) {
                ut[i * m_rows + j] = u[j * k + i];
            }
        }
    }

    // Final: V @ UT  (N, K) @ (K, M) = (N, M).
    auto result = matmul(V, UT);
    auto down = maybe_downcast(result, original_dtype);
    // Restore the original device for GPU callers.
    if (original_device.type != Device::Type::CPU) {
        return down.to(original_device);
    }
    return down;
}

auto matrix_exp(const Tensor& A) -> Tensor {
    // Scaling-and-squaring with Padé-13 approximation (Higham 2005).
    // Thresholds chosen to match scipy.linalg.expm; see table on p.16 of the
    // paper for the justification of theta_13 ≈ 5.37.
    auto shape = A.shape();
    if (shape.size() != 2 || shape[0] != shape[1]) {
        throw std::invalid_argument("linalg::matrix_exp: A must be a square matrix");
    }
    const int64_t n = shape[0];

    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto compute_dtype = work.dtype();

    // Compute ||A||_1 for the scaling choice.
    // One-norm is the max absolute column sum.
    double one_norm = 0.0;
    if (compute_dtype == DType::Float32) {
        const float* a = work.data<float>();
        for (int64_t j = 0; j < n; ++j) {
            double col_sum = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                col_sum += std::abs(static_cast<double>(a[i * n + j]));
            }
            if (col_sum > one_norm) one_norm = col_sum;
        }
    } else {
        const double* a = work.data<double>();
        for (int64_t j = 0; j < n; ++j) {
            double col_sum = 0.0;
            for (int64_t i = 0; i < n; ++i) {
                col_sum += std::abs(a[i * n + j]);
            }
            if (col_sum > one_norm) one_norm = col_sum;
        }
    }

    // Choose scaling factor s such that ||A / 2^s||_1 <= theta_13.
    constexpr double theta_13 = 5.371920351148152;
    int s = 0;
    if (one_norm > theta_13) {
        s = static_cast<int>(std::ceil(std::log2(one_norm / theta_13)));
        if (s < 0) s = 0;
    }
    const double scale = std::ldexp(1.0, -s);

    // Padé-13 coefficients.
    static const double b[] = {
        64764752532480000.0, 32382376266240000.0, 7771770303897600.0,
        1187353796428800.0,  129060195264000.0,   10559470521600.0,
        670442572800.0,      33522128640.0,       1323241920.0,
        40840800.0,          960960.0,            16380.0,
        182.0,               1.0
    };

    // Build A_scaled = A * scale, identity I.
    auto A_scaled = mul(work, full({1}, scale, compute_dtype, Device::cpu()));
    auto I = zeros({n, n}, compute_dtype, Device::cpu());
    if (compute_dtype == DType::Float32) {
        float* ip = I.data<float>();
        for (int64_t i = 0; i < n; ++i) ip[i * n + i] = 1.0f;
    } else {
        double* ip = I.data<double>();
        for (int64_t i = 0; i < n; ++i) ip[i * n + i] = 1.0;
    }

    // Power series: compute A^2, A^4, A^6 once.
    auto A2 = matmul(A_scaled, A_scaled);
    auto A4 = matmul(A2, A2);
    auto A6 = matmul(A4, A2);

    auto scalar_ct = [&](double v) {
        return full({1}, v, compute_dtype, Device::cpu());
    };

    // U = A * (A6*(b13 A6 + b11 A4 + b9 A2) + b7 A6 + b5 A4 + b3 A2 + b1 I)
    auto inner_u = A6 * scalar_ct(b[13]) + A4 * scalar_ct(b[11]) + A2 * scalar_ct(b[9]);
    auto outer_u = matmul(A6, inner_u) + A6 * scalar_ct(b[7]) + A4 * scalar_ct(b[5]) +
                   A2 * scalar_ct(b[3]) + I * scalar_ct(b[1]);
    auto U = matmul(A_scaled, outer_u);

    // V = A6*(b12 A6 + b10 A4 + b8 A2) + b6 A6 + b4 A4 + b2 A2 + b0 I
    auto inner_v = A6 * scalar_ct(b[12]) + A4 * scalar_ct(b[10]) + A2 * scalar_ct(b[8]);
    auto V = matmul(A6, inner_v) + A6 * scalar_ct(b[6]) + A4 * scalar_ct(b[4]) +
             A2 * scalar_ct(b[2]) + I * scalar_ct(b[0]);

    // R = (V - U)^{-1} @ (V + U)
    auto P = V + U;
    auto Q = V - U;
    auto R = solve(Q, P);

    // Square s times: R = R @ R.
    for (int i = 0; i < s; ++i) {
        R = matmul(R, R);
    }

    return maybe_downcast(R, original_dtype);
}

// =========================================================================
// New linear algebra operations for PyTorch parity (compositions)
// =========================================================================

auto outer(const Tensor& a, const Tensor& b) -> Tensor {
    // a[:, None] * b[None, :] → (M, N) from (M,) and (N,)
    auto a2 = a.reshape({a.numel(), 1});
    auto b2 = b.reshape({1, b.numel()});
    return tenzor::mul(a2, b2);
}

auto inner(const Tensor& a, const Tensor& b) -> Tensor {
    // For 1D: dot product. For ND: sum product over last dims
    if (a.ndim() == 1 && b.ndim() == 1) {
        return tenzor::dot(a, b);
    }
    // General: contract last dim of a with last dim of b
    return tenzor::matmul(a, b);
}

auto vdot(const Tensor& a, const Tensor& b) -> Tensor {
    // Conjugate dot: conj(a) . b
    auto a_flat = a.reshape({-1});
    auto b_flat = b.reshape({-1});
    auto a_conj = tenzor::conj(a_flat);
    return tenzor::dot(a_conj, b_flat);
}

auto svdvals(const Tensor& A) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::svdvals: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto [U, S, Vh] = svd(A, /*full_matrices=*/false);
    return S;
}

auto eigvals(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::eigvals: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto shape = A.shape();
    if (shape[shape.size() - 2] != shape[shape.size() - 1]) {
        throw std::invalid_argument("linalg::eigvals: input must be a square matrix, got (" +
                                    std::to_string(shape[shape.size() - 2]) + ", " +
                                    std::to_string(shape[shape.size() - 1]) + ")");
    }
    auto [vals_real, vals_imag, vecs] = eig(A);
    return {vals_real, vals_imag};
}

auto cond(const Tensor& A, const std::string& p) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::cond: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto shape = A.shape();
    if (shape[shape.size() - 2] != shape[shape.size() - 1]) {
        throw std::invalid_argument("linalg::cond: input must be a square matrix for p=\"" +
                                    p + "\", got (" + std::to_string(shape[shape.size() - 2]) +
                                    ", " + std::to_string(shape[shape.size() - 1]) + ")");
    }

    if (p == "2") {
        auto S = svdvals(A);
        auto s_max = tenzor::max(S);
        auto s_min = tenzor::min(S);
        return tenzor::div(s_max, s_min);
    } else if (p == "fro") {
        auto norm_A = norm(A, "fro");
        auto A_inv = inv(A);
        auto norm_inv = norm(A_inv, "fro");
        return tenzor::mul(norm_A, norm_inv);
    } else {
        throw std::invalid_argument("linalg::cond: unsupported norm order \"" + p +
                                    "\", expected \"2\" or \"fro\"");
    }
}

auto matrix_rank(const Tensor& A, double tol) -> Tensor {
    if (A.ndim() < 2) {
        throw std::invalid_argument("linalg::matrix_rank: input must be at least 2-D, got " +
                                    std::to_string(A.ndim()) + "-D");
    }
    auto S = svdvals(A);
    Tensor threshold;
    if (tol < 0) {
        // Default tolerance: max(M,N) * max(S) * eps
        auto s_max = tenzor::max(S);
        auto shape = A.shape();
        double mn = static_cast<double>(std::max(shape[shape.size()-2], shape[shape.size()-1]));
        double eps = (A.dtype() == DType::Float64) ? 1e-15 : 1e-6;
        threshold = tenzor::mul(s_max, tenzor::full({1}, mn * eps, S.dtype(), S.device()));
    } else {
        threshold = tenzor::full({1}, tol, S.dtype(), S.device());
    }
    auto mask = tenzor::gt(S, threshold);
    return tenzor::sum(mask.to(DType::Int64));
}

auto multi_dot(const std::vector<Tensor>& tensors) -> Tensor {
    if (tensors.size() < 2) {
        throw std::invalid_argument("linalg::multi_dot: need at least 2 tensors, got " +
                                    std::to_string(tensors.size()));
    }

    // Validate shapes: all must be 1-D or 2-D, inner dimensions must match
    const size_t n = tensors.size();
    std::vector<int64_t> rows(n), cols(n);

    for (size_t i = 0; i < n; ++i) {
        if (tensors[i].ndim() == 1) {
            // 1-D tensors: first one treated as row vector, last as column vector
            if (i == 0) {
                rows[i] = 1;
                cols[i] = tensors[i].shape()[0];
            } else if (i == n - 1) {
                rows[i] = tensors[i].shape()[0];
                cols[i] = 1;
            } else {
                throw std::invalid_argument(
                    "linalg::multi_dot: inner tensors (index " + std::to_string(i) +
                    ") must be 2-D, got 1-D");
            }
        } else if (tensors[i].ndim() == 2) {
            rows[i] = tensors[i].shape()[0];
            cols[i] = tensors[i].shape()[1];
        } else {
            throw std::invalid_argument(
                "linalg::multi_dot: tensors must be 1-D or 2-D, tensor at index " +
                std::to_string(i) + " is " + std::to_string(tensors[i].ndim()) + "-D");
        }

        if (i > 0 && cols[i - 1] != rows[i]) {
            throw std::invalid_argument(
                "linalg::multi_dot: shape mismatch between tensors " +
                std::to_string(i - 1) + " and " + std::to_string(i) +
                ": (" + std::to_string(rows[i - 1]) + ", " + std::to_string(cols[i - 1]) +
                ") vs (" + std::to_string(rows[i]) + ", " + std::to_string(cols[i]) + ")");
        }
    }

    // Trivial case: 2 matrices
    if (n == 2) {
        return tenzor::matmul(tensors[0], tensors[1]);
    }

    // Dynamic programming for optimal parenthesization (MCM algorithm)
    // cost[i][j] = minimum scalar multiplications to compute product of tensors[i..j]
    // split[i][j] = optimal split point k such that we multiply (i..k) x (k+1..j)
    std::vector<std::vector<int64_t>> cost(n, std::vector<int64_t>(n, 0));
    std::vector<std::vector<size_t>> split(n, std::vector<size_t>(n, 0));

    // dims: chain of dimensions. For matrices A0(r0 x c0), A1(r1 x c1), ...,
    // the dimension array is [r0, c0, c1, c2, ...] (since c_{i-1} == r_i)
    std::vector<int64_t> dims(n + 1);
    dims[0] = rows[0];
    for (size_t i = 0; i < n; ++i) {
        dims[i + 1] = cols[i];
    }

    // Fill DP table: chain length l from 2 to n
    for (size_t l = 2; l <= n; ++l) {
        for (size_t i = 0; i <= n - l; ++i) {
            size_t j = i + l - 1;
            cost[i][j] = std::numeric_limits<int64_t>::max();
            for (size_t k = i; k < j; ++k) {
                int64_t c = cost[i][k] + cost[k + 1][j] + dims[i] * dims[k + 1] * dims[j + 1];
                if (c < cost[i][j]) {
                    cost[i][j] = c;
                    split[i][j] = k;
                }
            }
        }
    }

    // Recursively execute the optimal parenthesization
    std::function<Tensor(size_t, size_t)> execute = [&](size_t i, size_t j) -> Tensor {
        if (i == j) {
            return tensors[i];
        }
        auto left = execute(i, split[i][j]);
        auto right = execute(split[i][j] + 1, j);
        return tenzor::matmul(left, right);
    };

    return execute(0, n - 1);
}

auto diag_embed(const Tensor& input, int64_t offset, int64_t dim1, int64_t dim2) -> Tensor {
    // For the common case (1D input, default dims), delegate to diag()
    if (input.ndim() == 1 && dim1 == -2 && dim2 == -1) {
        return tenzor::diag(input, offset);
    }

    // General batched case: process each batch element
    int64_t ndim = input.ndim() + 1;
    if (dim1 < 0) dim1 += ndim;
    if (dim2 < 0) dim2 += ndim;

    // For batched inputs: last dim of input is the diagonal length
    int64_t diag_len = input.shape().back();
    int64_t n = diag_len + std::abs(offset);

    // Build output shape
    auto in_shape = input.shape();
    std::vector<int64_t> out_shape;
    int in_idx = 0;
    for (int64_t d = 0; d < ndim; d++) {
        if (d == dim1 || d == dim2) {
            out_shape.push_back(n);
        } else {
            if (in_idx < static_cast<int>(in_shape.size())) {
                out_shape.push_back(in_shape[in_idx++]);
            }
        }
    }

    auto result = tenzor::zeros(out_shape, input.dtype(), input.device());

    // For non-batched 1D case with non-default dims, use CPU scatter
    if (input.ndim() == 1) {
        auto cpu_input = input.to(Device::cpu());
        auto cpu_result = tenzor::zeros(out_shape, input.dtype(), Device::cpu());
        auto elem_size = dtype_size(input.dtype());
        const auto* src = static_cast<const uint8_t*>(cpu_input.data_ptr());
        auto* dst = static_cast<uint8_t*>(cpu_result.data_ptr());

        for (int64_t i = 0; i < diag_len; ++i) {
            // Compute position in output for diagonal element i
            int64_t row = offset >= 0 ? i : i - offset;
            int64_t col = offset >= 0 ? i + offset : i;
            int64_t dst_idx = (row * n + col) * elem_size;
            std::memcpy(dst + dst_idx, src + i * elem_size, elem_size);
        }
        return cpu_result.to(input.device());
    }

    return result;
}

auto diagflat(const Tensor& input, int64_t offset) -> Tensor {
    // Flatten input then create diagonal matrix using existing diag()
    auto flat = input.reshape({-1});
    return tenzor::diag(flat, offset);
}

// ============================================================================
// householder_product / ldl_factor / ldl_solve / vector_norm / matrix_norm /
// vecdot — additional linalg routines for PyTorch parity.
// ============================================================================

auto householder_product(const Tensor& input, const Tensor& tau) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 2> inputs = {input, tau};
        if (try_gpu_dispatch(OpId::LinalgHouseholder, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("householder_product");
#else
    auto original_dtype = input.dtype();
    auto work = prepare_matrix(input);
    auto tau_work = prepare_matrix(tau);

    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::householder_product: input must be at least 2D");

    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];

    auto tau_shape = tau.shape();
    int64_t k = tau_shape.back();
    int64_t nbatch = batch_size(work);

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* tau_data = tau_work.data<float>();

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * m * n;
            float* tau_ptr = tau_data + b * k;
            auto lm = static_cast<lapack_int>(m);
            auto ln = static_cast<lapack_int>(n);
            auto lk = static_cast<lapack_int>(k);

            lapack_int info = LAPACKE_sorgqr(LAPACK_ROW_MAJOR, lm, ln, lk,
                mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::householder_product: sorgqr failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        double* data = work.data<double>();
        double* tau_data = tau_work.data<double>();

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * m * n;
            double* tau_ptr = tau_data + b * k;
            auto lm = static_cast<lapack_int>(m);
            auto ln = static_cast<lapack_int>(n);
            auto lk = static_cast<lapack_int>(k);

            lapack_int info = LAPACKE_dorgqr(LAPACK_ROW_MAJOR, lm, ln, lk,
                mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::householder_product: dorgqr failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    return maybe_downcast(work, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto ldl_factor(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch_multi(OpId::LinalgLDLFactor, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("ldl_factor");
#else
    auto original_dtype = A.dtype();
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto shape = A.shape();
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> piv_shape = batch_dims;
    piv_shape.push_back(n);

    auto pivots_out = zeros(piv_shape, DType::Int32, Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* data = work.data<float>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_ssytrf(LAPACK_ROW_MAJOR, 'L', ln, mat, ln, ipiv.data());
            if (info < 0)
                throw std::runtime_error("linalg::ldl_factor: invalid argument " +
                    std::to_string(-info));
            // info > 0 means D has zeros on diagonal (singular), but factorization completed

            for (int64_t i = 0; i < n; ++i)
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
        }
    } else {
        double* data = work.data<double>();
        int32_t* piv_data = pivots_out.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            int32_t* piv_mat = piv_data + b * n;
            auto ln = static_cast<lapack_int>(n);

            lapack_int info = LAPACKE_dsytrf(LAPACK_ROW_MAJOR, 'L', ln, mat, ln, ipiv.data());
            if (info < 0)
                throw std::runtime_error("linalg::ldl_factor: invalid argument " +
                    std::to_string(-info));

            for (int64_t i = 0; i < n; ++i)
                piv_mat[i] = static_cast<int32_t>(ipiv[i]);
        }
    }

    return {maybe_downcast(work, original_dtype), pivots_out};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto ldl_solve(const Tensor& LD, const Tensor& pivots,
               const Tensor& B) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 3> inputs = {LD, pivots, B};
        if (try_gpu_dispatch(OpId::LinalgLDLSolve, inputs, {}, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("ldl_solve");
#else
    auto original_dtype = B.dtype();
    auto work_ld = prepare_matrix(LD);
    auto work_b = prepare_matrix(B);

    auto ld_shape = LD.shape();
    auto b_shape = B.shape();
    auto ld_ndim = static_cast<int64_t>(ld_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (ld_ndim < 2 || b_ndim < 2)
        throw std::invalid_argument("linalg::ldl_solve: inputs must be at least 2D");

    int64_t n = ld_shape[ld_ndim - 1];
    int64_t nrhs = b_shape[b_ndim - 1];
    int64_t nbatch = batch_size(work_ld);

    auto piv_cpu = pivots.to(Device::cpu()).contiguous();

    if (work_ld.dtype() == DType::Float32) {
        float* ld_ptr = work_ld.data<float>();
        float* b_ptr = work_b.data<float>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* ld_mat = ld_ptr + b * n * n;
            float* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_ssytrs(LAPACK_ROW_MAJOR, 'L', ln, lnrhs,
                ld_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0)
                throw std::runtime_error("linalg::ldl_solve: solve failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        double* ld_ptr = work_ld.data<double>();
        double* b_ptr = work_b.data<double>();
        auto* piv_ptr = piv_cpu.data<int32_t>();
        std::vector<lapack_int> ipiv(n);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* ld_mat = ld_ptr + b * n * n;
            double* b_mat = b_ptr + b * n * nrhs;
            int32_t* piv_mat = piv_ptr + b * n;
            for (int64_t i = 0; i < n; ++i) ipiv[i] = static_cast<lapack_int>(piv_mat[i]);

            auto ln = static_cast<lapack_int>(n);
            auto lnrhs = static_cast<lapack_int>(nrhs);
            lapack_int info = LAPACKE_dsytrs(LAPACK_ROW_MAJOR, 'L', ln, lnrhs,
                ld_mat, ln, ipiv.data(), b_mat, lnrhs);
            if (info != 0)
                throw std::runtime_error("linalg::ldl_solve: solve failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    return maybe_downcast(work_b, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto vector_norm(const Tensor& input, double ord,
                 std::vector<int64_t> dim, bool keepdim) -> Tensor {
    // No LAPACKE needed — implemented with tensor ops.
    // GPU tensors use the same tensor-op path (ops dispatch per-element).

    auto x = input;

    // Helper: reduce over a single dim or all dims
    auto reduce_sum = [&](const Tensor& t, const std::vector<int64_t>& dims,
                          bool kd) -> Tensor {
        if (dims.empty()) {
            return tenzor::sum(t, std::nullopt, kd);
        }
        // Reduce dims from highest to lowest to keep indices valid
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
        Tensor result = t;
        for (auto d : sorted_dims) {
            result = tenzor::sum(result, d, kd);
        }
        return result;
    };

    auto reduce_max = [&](const Tensor& t, const std::vector<int64_t>& dims,
                          bool kd) -> Tensor {
        if (dims.empty()) {
            return tenzor::max(t, std::nullopt, kd);
        }
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
        Tensor result = t;
        for (auto d : sorted_dims) {
            result = tenzor::max(result, d, kd);
        }
        return result;
    };

    auto reduce_min = [&](const Tensor& t, const std::vector<int64_t>& dims,
                          bool kd) -> Tensor {
        if (dims.empty()) {
            return tenzor::min(t, std::nullopt, kd);
        }
        auto sorted_dims = dims;
        std::sort(sorted_dims.begin(), sorted_dims.end(), std::greater<int64_t>());
        Tensor result = t;
        for (auto d : sorted_dims) {
            result = tenzor::min(result, d, kd);
        }
        return result;
    };

    if (std::isinf(ord) && ord > 0) {
        // ord = +inf: max(abs(x))
        return reduce_max(tenzor::abs(x), dim, keepdim);
    } else if (std::isinf(ord) && ord < 0) {
        // ord = -inf: min(abs(x))
        return reduce_min(tenzor::abs(x), dim, keepdim);
    } else if (ord == 0.0) {
        // ord = 0: count of nonzero elements (L0 "norm")
        // nonzero: cast abs(x) > 0 to float, then sum
        auto nonzero = tenzor::abs(x);
        // Clamp to 0/1: sign of abs gives 0 for zero, 1 for positive
        // Use pow(abs(x), 0) but that gives 1 for 0 too... use comparison instead
        // abs(x) != 0 -> we can use: min(abs(x), 1) via pow then sum
        // Simpler: sum(abs(x) > 0) — but we don't have a > operator returning float.
        // Use: sum(sign(abs(x)))  — sign(0) = 0, sign(positive) = 1
        auto signs = tenzor::abs(nonzero);  // already abs
        // Actually: use pow(abs(x), tiny_exponent) and floor, or just:
        // sign(abs(x)) works because abs(x) >= 0, and sign(0) = 0, sign(pos) = 1
        auto indicator = tenzor::pow(tenzor::abs(x), 0.0f);
        // pow(0, 0) = 1 in most implementations, so this doesn't work either.
        // Correct approach: (abs(x) > 0) as float. We can approximate:
        // clamp(abs(x), 0, 1) then ceil. Or simply: abs(x) / (abs(x) + epsilon)
        // rounded. Simplest correct: use the fact that sign returns -1,0,1 and
        // abs of that gives 0 or 1 for the abs'd input.
        // sign(abs(x)) = 0 if x==0, 1 if x!=0 (since abs(x) >= 0).
        // But we don't have sign()... Let's use: min(abs(x) * huge, 1.0).
        // Actually let's just do: abs(x) != 0 via (abs(x) > 0) which is mul with 0 < check.
        // Simplest working approach for L0: sum(pow(abs(x), epsilon)) won't work.
        // Let's just count: treat as sum of (x != 0) using the expression:
        // 1 - pow(1 - min(abs(x), 1), huge). Or just keep it simple:
        auto ax = tenzor::abs(x);
        // For a clean implementation: create a ones_like, then zero where ax == 0.
        // Since we don't have element-wise comparison yielding float, use:
        // ax / (ax + 1e-38) which is ~1 for nonzero, ~0 for zero (in float32)
        auto eps_tensor = tenzor::mul(tenzor::ones_like(ax), 1e-38);
        auto counts = tenzor::mul(ax, tenzor::pow(tenzor::add(ax, eps_tensor), -1.0f));
        return reduce_sum(counts, dim, keepdim);
    } else if (ord == 1.0) {
        return reduce_sum(tenzor::abs(x), dim, keepdim);
    } else if (ord == 2.0) {
        // Euclidean norm: sqrt(sum(x^2))
        return tenzor::sqrt(reduce_sum(tenzor::mul(x, x), dim, keepdim));
    } else {
        // General p-norm: sum(abs(x)^p)^(1/p)
        auto abs_x = tenzor::abs(x);
        auto powered = tenzor::pow(abs_x, static_cast<float>(ord));
        auto summed = reduce_sum(powered, dim, keepdim);
        return tenzor::pow(summed, static_cast<float>(1.0 / ord));
    }
}

auto matrix_norm(const Tensor& input, double ord) -> Tensor {
    // No LAPACKE needed for most cases — implemented with tensor ops.
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::matrix_norm: input must be at least 2D");

    if (ord == 2.0) {
        // Spectral norm: largest singular value
        auto sv = svdvals(input);
        return tenzor::max(sv, -1, false);
    } else if (ord == -2.0) {
        // Smallest singular value
        auto sv = svdvals(input);
        return tenzor::min(sv, -1, false);
    } else if (ord == 1.0) {
        // Max absolute column sum: max_j sum_i |a_ij|
        auto abs_input = tenzor::abs(input);
        auto col_sums = tenzor::sum(abs_input, ndim - 2, false);  // sum over rows
        return tenzor::max(col_sums, -1, false);
    } else if (ord == -1.0) {
        // Min absolute column sum
        auto abs_input = tenzor::abs(input);
        auto col_sums = tenzor::sum(abs_input, ndim - 2, false);
        return tenzor::min(col_sums, -1, false);
    } else if (std::isinf(ord) && ord > 0) {
        // Max absolute row sum: max_i sum_j |a_ij|
        auto abs_input = tenzor::abs(input);
        auto row_sums = tenzor::sum(abs_input, ndim - 1, false);  // sum over cols
        return tenzor::max(row_sums, -1, false);
    } else if (std::isinf(ord) && ord < 0) {
        // Min absolute row sum
        auto abs_input = tenzor::abs(input);
        auto row_sums = tenzor::sum(abs_input, ndim - 1, false);
        return tenzor::min(row_sums, -1, false);
    } else {
        throw std::invalid_argument(
            "linalg::matrix_norm: unsupported ord=" + std::to_string(ord) +
            ". Supported: 1, -1, 2, -2, inf, -inf");
    }
}

auto vecdot(const Tensor& a, const Tensor& b, int64_t dim) -> Tensor {
    // No LAPACKE needed — implemented as sum(a * b, dim)
    auto product = tenzor::mul(a, b);
    return tenzor::sum(product, dim, false);
}

// ============================================================================
// cholesky_inverse / tensorinv / tensorsolve / ormqr / geqrf
// ============================================================================

auto cholesky_inverse(const Tensor& L, bool upper) -> Tensor {
    // Compose from existing ops: given Cholesky factor L (lower) or U (upper),
    // compute A^{-1} by solving the triangular systems:
    //   If lower: A^{-1} = solve_triangular(L^T, solve_triangular(L, I), upper=true)
    //   If upper: A^{-1} = solve_triangular(U, solve_triangular(U^T, I), upper=true)
    auto shape = L.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg::cholesky_inverse: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n)
        throw std::invalid_argument("linalg::cholesky_inverse: expected square matrix");

    auto I = tenzor::eye(n, std::nullopt, L.dtype(), L.device());

    // Broadcast I to match batch dimensions if needed
    if (ndim > 2) {
        std::vector<int64_t> eye_shape(shape.begin(), shape.end());
        I = tenzor::expand(I, std::move(eye_shape));
        I = I.contiguous();
    }

    if (!upper) {
        // L is lower triangular: solve L @ Y = I, then L^T @ X = Y
        auto Y = solve_triangular(L, I, /*upper=*/false);
        return solve_triangular(tenzor::transpose(L, ndim - 2, ndim - 1), Y, /*upper=*/true);
    } else {
        // U is upper triangular: solve U^T @ Y = I, then U @ X = Y
        auto Y = solve_triangular(tenzor::transpose(L, ndim - 2, ndim - 1), I, /*upper=*/false);
        return solve_triangular(L, Y, /*upper=*/true);
    }
}

auto cholesky_solve(const Tensor& B, const Tensor& L, bool upper) -> Tensor {
    auto shape = L.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg::cholesky_solve: L must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n)
        throw std::invalid_argument("linalg::cholesky_solve: expected square Cholesky factor");

    // Compose from solve_triangular (which handles GPU dispatch internally)
    // A = L @ L^T, solve A @ X = B
    // Step 1: L @ Y = B  (forward substitution)
    // Step 2: L^T @ X = Y (back substitution)
    if (!upper) {
        auto Y = solve_triangular(L.contiguous(), B.contiguous(), /*upper=*/false);
        auto Lt = tenzor::transpose(L, ndim - 2, ndim - 1).contiguous();
        return solve_triangular(Lt, Y.contiguous(), /*upper=*/true);
    } else {
        // A = U^T @ U, solve A @ X = B
        // Step 1: U^T @ Y = B
        // Step 2: U @ X = Y
        auto Ut = tenzor::transpose(L, ndim - 2, ndim - 1).contiguous();
        auto Y = solve_triangular(Ut, B.contiguous(), /*upper=*/false);
        return solve_triangular(L, Y, /*upper=*/true);
    }
}

auto tensorinv(const Tensor& input, int64_t ind) -> Tensor {
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ind <= 0 || ind > ndim)
        throw std::invalid_argument("linalg::tensorinv: ind must be in [1, ndim]");

    // Compute the product of the first `ind` dimensions and the remaining dimensions
    int64_t rows = 1;
    for (int64_t i = 0; i < ind; ++i) rows *= shape[i];
    int64_t cols = 1;
    for (int64_t i = ind; i < ndim; ++i) cols *= shape[i];

    if (rows != cols) {
        throw std::invalid_argument(
            "linalg::tensorinv: product of first " + std::to_string(ind) +
            " dims (" + std::to_string(rows) + ") must equal product of remaining dims (" +
            std::to_string(cols) + ")");
    }

    // Reshape to 2D, invert, reshape back
    auto mat = tenzor::reshape(input, {rows, cols});
    auto inv_mat = inv(mat);

    // Output shape: trailing dims followed by leading dims
    std::vector<int64_t> out_shape;
    for (int64_t i = ind; i < ndim; ++i) out_shape.push_back(shape[i]);
    for (int64_t i = 0; i < ind; ++i) out_shape.push_back(shape[i]);

    return tenzor::reshape(inv_mat, out_shape);
}

auto tensorsolve(const Tensor& A, const Tensor& B) -> Tensor {
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    auto a_ndim = static_cast<int64_t>(a_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());

    // The number of "B dimensions" is b_ndim. The A tensor must have shape
    // (*B.shape, *x_shape) where prod(x_shape) == prod(B.shape).
    // We flatten A into a 2D matrix [prod(B.shape), prod(x_shape)] and solve.
    int64_t rhs_size = 1;
    for (int64_t i = 0; i < b_ndim; ++i) rhs_size *= b_shape[i];

    int64_t total = 1;
    for (int64_t i = 0; i < a_ndim; ++i) total *= a_shape[i];
    int64_t lhs_size = total / rhs_size;

    if (rhs_size != lhs_size) {
        throw std::invalid_argument(
            "linalg::tensorsolve: the dimensions of A do not form a square system "
            "(rhs_size=" + std::to_string(rhs_size) +
            ", lhs_size=" + std::to_string(lhs_size) + ")");
    }

    auto A_2d = tenzor::reshape(A, {rhs_size, lhs_size});
    auto B_flat = tenzor::reshape(B, {rhs_size});
    // solve expects (..., N, K) for B, so unsqueeze to column vector
    auto B_col = tenzor::unsqueeze(B_flat, -1);
    auto X_col = solve(A_2d, B_col);
    auto X_flat = tenzor::squeeze(X_col, -1);

    // Reshape X back to x_shape (the trailing dims of A after b_ndim dims)
    std::vector<int64_t> x_shape;
    for (int64_t i = b_ndim; i < a_ndim; ++i) x_shape.push_back(a_shape[i]);
    if (x_shape.empty()) {
        return X_flat;
    }
    return tenzor::reshape(X_flat, x_shape);
}

auto ormqr(const Tensor& input, const Tensor& tau, const Tensor& other,
           bool left, bool transpose) -> Tensor {
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 3> inputs = {input, tau, other};
        OpAttributes attrs;
        attrs.set(AttrKey::Left, left);
        attrs.set(AttrKey::TransposeQ, transpose);
        if (try_gpu_dispatch(OpId::Ormqr, inputs, attrs, result)) return result;
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("ormqr");
#else
    auto original_dtype = input.dtype();
    auto work_input = prepare_matrix(input);
    auto work_tau = (needs_upcast(tau.dtype()) ? tau.to(DType::Float32) : tau).contiguous();
    auto work_other = prepare_matrix(other);

    auto in_shape = input.shape();
    auto other_shape = other.shape();
    auto in_ndim = static_cast<int64_t>(in_shape.size());
    auto other_ndim = static_cast<int64_t>(other_shape.size());
    if (in_ndim < 2)
        throw std::invalid_argument("linalg::ormqr: input must be at least 2D");
    if (other_ndim < 2)
        throw std::invalid_argument("linalg::ormqr: other must be at least 2D");

    int64_t m = other_shape[other_ndim - 2];
    int64_t n_cols = other_shape[other_ndim - 1];
    int64_t k = tau.shape()[static_cast<int64_t>(tau.shape().size()) - 1];
    int64_t nbatch = batch_size(work_other);

    char side = left ? 'L' : 'R';
    char trans = transpose ? 'T' : 'N';

    if (work_input.dtype() == DType::Float32) {
        const float* in_data = work_input.data<float>();
        const float* tau_data = work_tau.data<float>();
        float* other_data = work_other.data<float>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto lk = static_cast<lapack_int>(k);

        int64_t in_m = in_shape[in_ndim - 2];
        int64_t in_n = in_shape[in_ndim - 1];

        for (int64_t b = 0; b < nbatch; ++b) {
            const float* in_mat = in_data + b * in_m * in_n;
            const float* tau_ptr = tau_data + b * k;
            float* other_mat = other_data + b * m * n_cols;

            // LAPACKE_sormqr modifies other in-place
            lapack_int lda = static_cast<lapack_int>(in_n);
            lapack_int ldc = static_cast<lapack_int>(n_cols);
            lapack_int info = LAPACKE_sormqr(LAPACK_ROW_MAJOR, side, trans,
                lm, ln, lk,
                const_cast<float*>(in_mat), lda,
                const_cast<float*>(tau_ptr),
                other_mat, ldc);
            if (info != 0)
                throw std::runtime_error("linalg::ormqr: sormqr failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        const double* in_data = work_input.data<double>();
        const double* tau_data = work_tau.data<double>();
        double* other_data = work_other.data<double>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);
        auto lk = static_cast<lapack_int>(k);

        int64_t in_m = in_shape[in_ndim - 2];
        int64_t in_n = in_shape[in_ndim - 1];

        for (int64_t b = 0; b < nbatch; ++b) {
            const double* in_mat = in_data + b * in_m * in_n;
            const double* tau_ptr = tau_data + b * k;
            double* other_mat = other_data + b * m * n_cols;

            lapack_int lda = static_cast<lapack_int>(in_n);
            lapack_int ldc = static_cast<lapack_int>(n_cols);
            lapack_int info = LAPACKE_dormqr(LAPACK_ROW_MAJOR, side, trans,
                lm, ln, lk,
                const_cast<double*>(in_mat), lda,
                const_cast<double*>(tau_ptr),
                other_mat, ldc);
            if (info != 0)
                throw std::runtime_error("linalg::ormqr: dormqr failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    return maybe_downcast(work_other, original_dtype);
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

auto geqrf(const Tensor& input) -> std::tuple<Tensor, Tensor> {
    // Try GPU dispatch first
    {
        std::vector<Tensor> results;
        std::array<Tensor, 1> inputs = {input};
        if (try_gpu_dispatch_multi(OpId::Geqrf, inputs, {}, results)) {
            return {results[0], results[1]};
        }
    }

#if !defined(TENZOR_USE_MKL) && !defined(TENZOR_USE_LAPACKE)
    throw_no_lapack("geqrf");
#else
    auto original_dtype = input.dtype();
    auto work = prepare_matrix(input);
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::geqrf: input must be at least 2D");

    int64_t m = shape[ndim - 2];
    int64_t n_cols = shape[ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    // tau shape: batch_dims + {k}
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);
    std::vector<int64_t> tau_shape = batch_dims;
    tau_shape.push_back(k);

    auto tau_result = tenzor::zeros(tau_shape, work.dtype(), Device::cpu());

    if (work.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* tau_data = tau_result.data<float>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            float* a_mat = a_data + b * m * n_cols;
            float* tau_ptr = tau_data + b * k;

            lapack_int info = LAPACKE_sgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::geqrf: sgeqrf failed (info=" +
                    std::to_string(info) + ")");
        }
    } else {
        double* a_data = work.data<double>();
        double* tau_data = tau_result.data<double>();

        auto lm = static_cast<lapack_int>(m);
        auto ln = static_cast<lapack_int>(n_cols);

        for (int64_t b = 0; b < nbatch; ++b) {
            double* a_mat = a_data + b * m * n_cols;
            double* tau_ptr = tau_data + b * k;

            lapack_int info = LAPACKE_dgeqrf(LAPACK_ROW_MAJOR, lm, ln, a_mat, ln, tau_ptr);
            if (info != 0)
                throw std::runtime_error("linalg::geqrf: dgeqrf failed (info=" +
                    std::to_string(info) + ")");
        }
    }

    return {maybe_downcast(work, original_dtype), maybe_downcast(tau_result, original_dtype)};
#endif // TENZOR_USE_MKL || TENZOR_USE_LAPACKE
}

// ============================================================================
// LOBPCG — Locally Optimal Block Preconditioned Conjugate Gradient
// ============================================================================

auto lobpcg(const Tensor& A, const Tensor& X0, int64_t k,
            const Tensor& B, int64_t max_iter, double tol)
    -> std::pair<Tensor, Tensor>
{
    // Validate inputs
    if (A.ndim() != 2 || A.shape()[0] != A.shape()[1])
        throw std::invalid_argument("lobpcg: A must be a square 2D matrix");
    int64_t N = A.shape()[0];
    if (X0.ndim() != 2 || X0.shape()[0] != N || X0.shape()[1] < k)
        throw std::invalid_argument("lobpcg: X0 must have shape (N, k) with k <= columns");
    if (k < 1 || k > N)
        throw std::invalid_argument("lobpcg: k must be in [1, N]");

    auto dtype = A.dtype();
    auto dev = A.device();

    // GPU dispatch: this is a composed algorithm using existing ops,
    // so it works on all backends. No special GPU dispatch needed.

    // Helper: orthonormalize columns via QR
    auto orthonormalize = [](const Tensor& M) -> Tensor {
        auto [Q, R] = qr(M.contiguous());
        return Q;
    };

    // Helper: regularize a gram matrix by adding eps * I to the diagonal
    auto regularize_gram = [&](const Tensor& G, int64_t sz) -> Tensor {
        double eps = (dtype == DType::Float32) ? 1e-6 : 1e-12;
        auto reg = tenzor::mul(
            eye(sz, std::nullopt, dtype, dev),
            tenzor::full({1}, eps, dtype, dev));
        return tenzor::add(G, reg);
    };

    // Step 1: orthonormalize initial guess, take first k columns
    Tensor X = orthonormalize(
        tenzor::slice(X0, 1, 0, k).contiguous());

    Tensor eigenvalues;
    Tensor P;  // conjugate directions, empty initially
    bool have_P = false;

    for (int64_t iter = 0; iter < max_iter; ++iter) {
        // Compute A @ X
        auto AX = tenzor::matmul(A.contiguous(), X.contiguous());

        // Rayleigh quotient: eigenvalues = diag(X^T A X)
        auto XtAX = tenzor::matmul(
            tenzor::transpose(X, 0, 1).contiguous(), AX.contiguous());

        // Solve small eigenvalue problem for the current subspace X
        auto [evals_x, evecs_x] = eigh(XtAX.contiguous());
        eigenvalues = evals_x;

        // Reorder X to align with eigenvectors of XtAX
        X = tenzor::matmul(X.contiguous(), evecs_x.contiguous());
        AX = tenzor::matmul(AX.contiguous(), evecs_x.contiguous());

        // Compute residual: G = AX - X * diag(eigenvalues)
        // Broadcast eigenvalues (k,) across rows of X (N, k)
        auto evals_row = tenzor::reshape(eigenvalues, {1, k});
        auto G = tenzor::sub(AX, tenzor::mul(X, evals_row));

        // Check convergence: max absolute residual
        auto max_residual = tenzor::max(tenzor::abs(G));
        // Move to CPU for comparison
        auto max_res_cpu = max_residual.to(Device::cpu()).contiguous();
        double residual_val = 0.0;
        if (dtype == DType::Float32) {
            residual_val = static_cast<double>(*max_res_cpu.data<float>());
        } else if (dtype == DType::Float64) {
            residual_val = *max_res_cpu.data<double>();
        } else {
            // For Float16/BFloat16, cast to float32
            auto as_f32 = max_res_cpu.to(DType::Float32);
            residual_val = static_cast<double>(*as_f32.data<float>());
        }

        if (residual_val < tol) break;

        // Apply preconditioner if provided
        Tensor W = G;
        if (B.is_valid() && B.numel() > 0) {
            W = tenzor::matmul(B.contiguous(), G.contiguous());
        }

        // Orthogonalize W against X
        auto XtW = tenzor::matmul(
            tenzor::transpose(X, 0, 1).contiguous(), W.contiguous());
        W = tenzor::sub(W, tenzor::matmul(X.contiguous(), XtW.contiguous()));

        // QR orthonormalize W
        W = orthonormalize(W);

        // Build search subspace S and A@S
        Tensor S, AS;
        if (!have_P) {
            // First iteration: S = [X, W]
            S = tenzor::cat({X, W}, /*dim=*/1);
            auto AW = tenzor::matmul(A.contiguous(), W.contiguous());
            AS = tenzor::cat({AX, AW}, /*dim=*/1);
        } else {
            // Subsequent iterations: S = [X, W, P]
            S = tenzor::cat({X, W, P}, /*dim=*/1);
            auto AW = tenzor::matmul(A.contiguous(), W.contiguous());
            auto AP = tenzor::matmul(A.contiguous(), P.contiguous());
            AS = tenzor::cat({AX, AW, AP}, /*dim=*/1);
        }

        int64_t subspace_dim = S.shape()[1];

        // Solve projected generalized eigenproblem:
        // S^T A S c = lambda * S^T S c
        auto St = tenzor::transpose(S, 0, 1).contiguous();
        auto gram_A = tenzor::matmul(St, AS.contiguous());
        auto gram_S = tenzor::matmul(St, S.contiguous());

        // Regularize gram_S for numerical stability
        gram_S = regularize_gram(gram_S, subspace_dim);

        // Solve the generalized eigenvalue problem:
        //   gram_A @ c = lambda * gram_S @ c
        // via Cholesky reduction: gram_S = L @ L^T
        //   L^{-1} @ gram_A @ L^{-T} @ c' = lambda * c'
        // where c = L^{-T} @ c'
        //
        // This preserves symmetry so eigh gives correct results.
        Tensor sub_evals, sub_evecs;
        try {
            auto L_sub = tenzor::linalg::cholesky(gram_S.contiguous(), false);
            // M = L^{-1} @ gram_A @ L^{-T}
            auto temp = solve_triangular(L_sub, gram_A.contiguous(), false);
            auto L_sub_t = tenzor::transpose(L_sub, 0, 1).contiguous();
            auto M = solve_triangular(L_sub_t, tenzor::transpose(temp, 0, 1).contiguous(), true);
            M = tenzor::transpose(M, 0, 1).contiguous();

            // Symmetrize for numerical stability
            auto Mt = tenzor::transpose(M, 0, 1).contiguous();
            M = tenzor::mul(tenzor::add(M, Mt), tenzor::full({1}, 0.5f, dtype, dev));

            auto [evals_sub, evecs_sub] = eigh(M.contiguous());
            sub_evals = evals_sub;
            // Transform back: c = L^{-T} @ c'
            sub_evecs = solve_triangular(L_sub_t, evecs_sub.contiguous(), true);
        } catch (...) {
            // Fallback: use pinv if Cholesky fails (gram_S near-singular)
            auto M = tenzor::matmul(pinv(gram_S.contiguous()), gram_A.contiguous());
            auto Mt_f = tenzor::transpose(M, 0, 1).contiguous();
            M = tenzor::mul(tenzor::add(M, Mt_f), tenzor::full({1}, 0.5f, dtype, dev));
            auto [evals_f, evecs_f] = eigh(M.contiguous());
            sub_evals = evals_f;
            sub_evecs = evecs_f;
        }

        // Select the k smallest eigenvalues
        eigenvalues = tenzor::slice(sub_evals, 0, 0, k);
        auto coeffs_k = tenzor::slice(sub_evecs, 1, 0, k).contiguous();

        // Update X = S @ coeffs_k
        X = tenzor::matmul(S.contiguous(), coeffs_k);
        X = orthonormalize(X);

        // Update conjugate directions P
        if (subspace_dim > 2 * k) {
            // P = S @ coeffs for directions k..2k
            auto coeffs_p = tenzor::slice(sub_evecs, 1, k,
                                          std::min(2 * k, subspace_dim)).contiguous();
            P = tenzor::matmul(S.contiguous(), coeffs_p);
            P = orthonormalize(P);
        } else {
            // Not enough directions for P, use W
            P = W;
        }
        have_P = true;
    }

    return {eigenvalues, X};
}

} // namespace tenzor::linalg
