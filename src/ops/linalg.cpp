#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <mkl.h>
#include <mkl_lapacke.h>
#include <stdexcept>
#include <cstring>
#include <cmath>
#include <limits>
#include <vector>
#include <algorithm>

namespace tenzor::linalg {

namespace {

/// Try GPU dispatch for a single-output linalg op. Returns true and sets result on success.
bool try_gpu_dispatch(OpId op, std::span<const Tensor> inputs,
                      const OpAttributes& attrs, Tensor& result) {
    if (inputs[0].device().type == Device::Type::CPU) return false;
    try {
        result = dispatch_single(op, inputs, attrs);
        return true;
    } catch (...) {
        return false;
    }
}

/// Try GPU dispatch for a multi-output linalg op. Returns true and sets results on success.
bool try_gpu_dispatch_multi(OpId op, std::span<const Tensor> inputs,
                            const OpAttributes& attrs, std::vector<Tensor>& results) {
    if (inputs[0].device().type == Device::Type::CPU) return false;
    try {
        results = dispatch(op, inputs, attrs);
        return true;
    } catch (...) {
        return false;
    }
}

// Ensure tensor is contiguous Float32 or Float64 on CPU, return a working copy
auto prepare_matrix(const Tensor& A) -> Tensor {
    if (A.dtype() != DType::Float32 && A.dtype() != DType::Float64) {
        throw std::runtime_error("linalg: only Float32 and Float64 supported");
    }
    // If not on CPU, move to CPU for LAPACK fallback
    auto cpu_tensor = (A.device().type != Device::Type::CPU) ? A.to(Device::cpu()) : A;
    return cpu_tensor.contiguous().clone();
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
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        if (try_gpu_dispatch(OpId::LinalgInv, inputs, {}, result)) return result;
    }

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
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 2> inputs = {A, B};
        if (try_gpu_dispatch(OpId::LinalgSolve, inputs, {}, result)) return result;
    }

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
    // Try GPU dispatch first
    {
        Tensor result;
        std::array<Tensor, 1> inputs = {A};
        OpAttributes attrs;
        attrs.set(AttrKey::Upper, upper);
        if (try_gpu_dispatch(OpId::LinalgCholesky, inputs, attrs, result)) return result;
    }

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

auto slogdet(const Tensor& A) -> std::tuple<Tensor, Tensor> {
    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        out_shape.push_back(shape[i]);
    }
    if (out_shape.empty()) out_shape.push_back(1);

    auto sign_result = zeros(out_shape, A.dtype(), Device::cpu());
    auto logabsdet_result = zeros(out_shape, A.dtype(), Device::cpu());

    std::vector<lapack_int> ipiv(n);

    if (A.dtype() == DType::Float32) {
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

    return {sign_result, logabsdet_result};
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

    auto U = zeros(u_shape, A.dtype(), Device::cpu());
    auto S = zeros(s_shape, A.dtype(), Device::cpu());
    auto Vt = zeros(vt_shape, A.dtype(), Device::cpu());

    char jobz = full_matrices ? 'A' : 'S';

    // superb array for ?gesvd
    int64_t superb_size = k - 1;
    if (superb_size < 1) superb_size = 1;

    if (A.dtype() == DType::Float32) {
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

    return {U, S, Vt};
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

    auto Q = zeros(q_shape, A.dtype(), Device::cpu());
    auto R = zeros(r_shape, A.dtype(), Device::cpu());

    if (A.dtype() == DType::Float32) {
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

    return {Q, R};
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

    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    // Eigenvalues shape: (..., N)
    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, A.dtype(), Device::cpu());

    // Eigenvectors are stored in work (overwritten by dsyev/ssyev)

    if (A.dtype() == DType::Float32) {
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
    return {W, work};
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

    auto work = prepare_matrix(A);
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); ++i) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, A.dtype(), Device::cpu());

    if (A.dtype() == DType::Float32) {
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

    return W;
}

} // namespace tenzor::linalg
