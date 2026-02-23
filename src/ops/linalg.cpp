#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <mkl.h>
#include <mkl_lapacke.h>
#include <stdexcept>
#include <cstring>
#include <vector>

namespace tenzor::linalg {

namespace {

// Ensure tensor is contiguous Float32 or Float64 on CPU, return a working copy
auto prepare_matrix(const Tensor& A) -> Tensor {
    if (A.device().type != Device::Type::CPU) {
        throw std::runtime_error("linalg: only CPU tensors supported");
    }
    if (A.dtype() != DType::Float32 && A.dtype() != DType::Float64) {
        throw std::runtime_error("linalg: only Float32 and Float64 supported");
    }
    return A.contiguous().clone();
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

    auto result = zeros(out_shape, A.dtype(), Device::cpu());

    std::vector<lapack_int> ipiv(n);

    if (A.dtype() == DType::Float32) {
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

    return result;
}

auto inv(const Tensor& A) -> Tensor {
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<lapack_int> ipiv(n);

    if (A.dtype() == DType::Float32) {
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

    return work;
}

auto solve(const Tensor& A, const Tensor& B) -> Tensor {
    auto work_a = prepare_matrix(A);
    auto work_b = prepare_matrix(B);
    auto [n, ndim_a] = check_square(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (b_ndim < 1) throw std::invalid_argument("linalg::solve: B must be at least 1D");

    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    std::vector<lapack_int> ipiv(n);

    if (A.dtype() == DType::Float32) {
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

    return work_b;
}

auto cholesky(const Tensor& A, bool upper) -> Tensor {
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    char uplo = upper ? 'U' : 'L';

    if (A.dtype() == DType::Float32) {
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

    return work;
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

} // namespace tenzor::linalg
