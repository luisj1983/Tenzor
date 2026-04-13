/**
 * @file mps_linalg.mm
 * @brief Native linear algebra kernels for MPS backend using Apple Accelerate (LAPACK)
 *
 * On Apple Silicon, MPS tensors use MTLResourceStorageModeShared (unified
 * memory), so LAPACK functions from Accelerate can operate directly on tensor
 * data pointers with ZERO copies. This is NOT a CPU fallback -- Accelerate
 * LAPACK on Apple Silicon is the optimal implementation for these operations.
 *
 * Supported operations:
 *   - LinalgDet:        Determinant via LU factorization
 *   - LinalgInv:        Matrix inverse
 *   - LinalgSolve:      Solve Ax = b
 *   - LinalgSVD:        Singular value decomposition
 *   - LinalgQR:         QR decomposition
 *   - LinalgEigh:       Eigenvalues of symmetric matrix
 *   - LinalgEig:        General eigenvalue decomposition
 *   - LinalgCholesky:   Cholesky decomposition
 *   - LinalgLU:         LU decomposition
 *   - LinalgLUSolve:    Solve using LU factors
 *   - SolveTriangular:  Triangular system solve
 *
 * All operations are Float32 only (Metal does not support Float64).
 * LAPACK routines modify data in-place, so inputs are cloned before calling.
 */

#import <Accelerate/Accelerate.h>

#include "../mps_backend.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>

namespace tenzor::mps {

namespace {

// ============================================================================
// Helpers
// ============================================================================

/// Ensure matrix is square and return {n, ndim}
auto check_square(const Tensor& A) -> std::pair<int64_t, int64_t> {
    auto shape = A.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("MPS linalg: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    if (m != n) {
        throw std::invalid_argument("MPS linalg: expected square matrix, got " +
            std::to_string(m) + "x" + std::to_string(n));
    }
    return {n, ndim};
}

/// Compute batch size (product of all dims except last two)
auto batch_count(const Tensor& A) -> int64_t {
    auto shape = A.shape();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); ++i) {
        batch *= shape[i];
    }
    return batch;
}

/// Prepare a Float32 contiguous clone for in-place LAPACK operations.
/// On MPS with unified memory, data<float>() gives a pointer LAPACK can use directly.
auto prepare_work_copy(const Tensor& A) -> Tensor {
    Tensor work = A;
    if (work.dtype() != DType::Float32) {
        work = work.to(DType::Float32);
    }
    return work.contiguous().clone();
}

/// Batch shape: all dims except last two
auto batch_shape(const Tensor& A) -> std::vector<int64_t> {
    auto shape = A.shape();
    return std::vector<int64_t>(shape.begin(), shape.end() - 2);
}

} // anonymous namespace

// ============================================================================
// LinalgDet: Determinant via LU factorization
// ============================================================================

Tensor mps_linalg_det_kernel(const Tensor& A) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto work = prepare_work_copy(A);
    float* data = work.data<float>();

    auto bshape = batch_shape(A);
    Tensor result(bshape.empty() ? std::vector<int64_t>{} : bshape,
                  DType::Float32, A.device());
    // Handle scalar output (0-dim batch)
    if (bshape.empty()) {
        bshape.push_back(1);
    }
    float* det_data = result.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = N;
    __CLPK_integer info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));

    for (int64_t b = 0; b < batch; ++b) {
        float* mat = data + b * n * n;
        info = 0;
        sgetrf_(&N, &N, mat, &lda, ipiv.data(), &info);
        if (info < 0) {
            throw std::runtime_error("MPS LinalgDet: sgetrf_ illegal argument");
        }

        // det = product of diagonal * sign from pivots
        float det = 1.0f;
        int sign = 1;
        for (int64_t i = 0; i < n; ++i) {
            det *= mat[i * n + i]; // LAPACK uses column-major; (i,i) = i*lda + i
            if (ipiv[static_cast<size_t>(i)] != static_cast<__CLPK_integer>(i + 1)) {
                sign = -sign;
            }
        }
        det_data[b] = det * static_cast<float>(sign);
    }

    return result;
}

// ============================================================================
// LinalgInv: Matrix inverse
// ============================================================================

Tensor mps_linalg_inv_kernel(const Tensor& A) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto result = prepare_work_copy(A);
    float* data = result.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = N;
    __CLPK_integer info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));
    __CLPK_integer lwork = N * 64; // generous workspace
    std::vector<float> work(static_cast<size_t>(lwork));

    for (int64_t b = 0; b < batch; ++b) {
        float* mat = data + b * n * n;
        info = 0;
        sgetrf_(&N, &N, mat, &lda, ipiv.data(), &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgInv: sgetrf_ failed (singular matrix or error " +
                                     std::to_string(info) + ")");
        }
        info = 0;
        sgetri_(&N, mat, &lda, ipiv.data(), work.data(), &lwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgInv: sgetri_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    return result;
}

// ============================================================================
// LinalgSolve: Solve AX = B
// ============================================================================

Tensor mps_linalg_solve_kernel(const Tensor& A, const Tensor& B) {
    auto [n, ndim_a] = check_square(A);
    int64_t batch = batch_count(A);

    auto a_work = prepare_work_copy(A);
    auto b_work = prepare_work_copy(B);
    float* a_data = a_work.data<float>();
    float* b_data = b_work.data<float>();

    auto b_shape = B.shape();
    int64_t nrhs = (b_shape.size() > 1 && b_shape.back() != n) ? b_shape.back() : 1;
    // If B is (batch, n), treat as (batch, n, 1)
    if (static_cast<int64_t>(b_shape.size()) == ndim_a - 1) {
        nrhs = 1;
    } else {
        nrhs = b_shape.back();
    }

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer NRHS = static_cast<__CLPK_integer>(nrhs);
    __CLPK_integer lda = N;
    __CLPK_integer ldb = N;
    __CLPK_integer info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* b_mat = b_data + b * n * nrhs;
        info = 0;
        sgesv_(&N, &NRHS, a_mat, &lda, ipiv.data(), b_mat, &ldb, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgSolve: sgesv_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    return b_work;
}

// ============================================================================
// LinalgCholesky: Cholesky decomposition
// ============================================================================

Tensor mps_linalg_cholesky_kernel(const Tensor& A, bool upper) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto result = prepare_work_copy(A);
    float* data = result.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = N;
    __CLPK_integer info = 0;
    char uplo = upper ? 'U' : 'L';

    for (int64_t b = 0; b < batch; ++b) {
        float* mat = data + b * n * n;
        info = 0;
        spotrf_(&uplo, &N, mat, &lda, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgCholesky: spotrf_ failed (error " +
                                     std::to_string(info) + ")");
        }
        // Zero out the opposite triangle
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (upper && j < i) {
                    mat[j * n + i] = 0.0f; // column-major: (i,j) = j*lda + i
                } else if (!upper && j > i) {
                    mat[j * n + i] = 0.0f;
                }
            }
        }
    }

    return result;
}

// ============================================================================
// LinalgSVD: Singular Value Decomposition
// ============================================================================

std::vector<Tensor> mps_linalg_svd_kernel(const Tensor& A, bool full_matrices) {
    auto shape = A.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("MPS LinalgSVD: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    int64_t k = std::min(m, n);
    int64_t batch = batch_count(A);

    auto work_a = prepare_work_copy(A);
    float* a_data = work_a.data<float>();

    char jobz = full_matrices ? 'A' : 'S';

    int64_t u_cols = full_matrices ? m : k;
    int64_t vt_rows = full_matrices ? n : k;

    auto bshape = batch_shape(A);

    // Output shapes
    auto u_shape = bshape;
    u_shape.push_back(m);
    u_shape.push_back(u_cols);

    auto s_shape = bshape;
    s_shape.push_back(k);

    auto vt_shape = bshape;
    vt_shape.push_back(vt_rows);
    vt_shape.push_back(n);

    Tensor U(u_shape, DType::Float32, A.device());
    Tensor S(s_shape, DType::Float32, A.device());
    Tensor Vt(vt_shape, DType::Float32, A.device());

    float* u_data = U.data<float>();
    float* s_data = S.data<float>();
    float* vt_data = Vt.data<float>();

    __CLPK_integer M = static_cast<__CLPK_integer>(m);
    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = M;
    __CLPK_integer ldu = M;
    __CLPK_integer ldvt = static_cast<__CLPK_integer>(vt_rows);
    __CLPK_integer info = 0;

    // Query optimal workspace
    __CLPK_integer lwork = -1;
    float work_query;
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(8 * k));
    sgesdd_(&jobz, &M, &N, a_data, &lda, s_data, u_data, &ldu,
            vt_data, &ldvt, &work_query, &lwork, iwork.data(), &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(static_cast<size_t>(lwork));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * m * n;
        float* u_mat = u_data + b * m * u_cols;
        float* s_vec = s_data + b * k;
        float* vt_mat = vt_data + b * vt_rows * n;
        info = 0;
        sgesdd_(&jobz, &M, &N, a_mat, &lda, s_vec, u_mat, &ldu,
                vt_mat, &ldvt, work.data(), &lwork, iwork.data(), &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgSVD: sgesdd_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    return {U, S, Vt};
}

// ============================================================================
// LinalgQR: QR decomposition
// ============================================================================

std::vector<Tensor> mps_linalg_qr_kernel(const Tensor& A) {
    auto shape = A.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("MPS LinalgQR: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    int64_t k = std::min(m, n);
    int64_t batch = batch_count(A);

    auto work_a = prepare_work_copy(A);
    float* a_data = work_a.data<float>();

    __CLPK_integer M = static_cast<__CLPK_integer>(m);
    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = M;
    __CLPK_integer info = 0;
    __CLPK_integer K = static_cast<__CLPK_integer>(k);

    std::vector<float> tau(static_cast<size_t>(k));

    // Query optimal workspace for sgeqrf
    __CLPK_integer lwork = -1;
    float work_query;
    sgeqrf_(&M, &N, a_data, &lda, tau.data(), &work_query, &lwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(static_cast<size_t>(std::max(lwork, __CLPK_integer(1))));

    auto bshape = batch_shape(A);

    auto q_shape = bshape;
    q_shape.push_back(m);
    q_shape.push_back(k);

    auto r_shape = bshape;
    r_shape.push_back(k);
    r_shape.push_back(n);

    Tensor Q(q_shape, DType::Float32, A.device());
    Tensor R(r_shape, DType::Float32, A.device());
    float* q_data = Q.data<float>();
    float* r_data = R.data<float>();

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * m * n;
        info = 0;

        // QR factorization
        sgeqrf_(&M, &N, a_mat, &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgQR: sgeqrf_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // Extract R (upper triangular part)
        float* r_mat = r_data + b * k * n;
        std::memset(r_mat, 0, static_cast<size_t>(k * n) * sizeof(float));
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i <= std::min(j, k - 1); ++i) {
                r_mat[j * k + i] = a_mat[j * m + i]; // column-major
            }
        }

        // Generate Q from Householder reflectors (sorgqr)
        // Copy the first k columns of the factored matrix
        float* q_mat = q_data + b * m * k;
        for (int64_t j = 0; j < k; ++j) {
            std::memcpy(q_mat + j * m, a_mat + j * m,
                        static_cast<size_t>(m) * sizeof(float));
        }

        __CLPK_integer lwork_q = -1;
        float wq;
        sorgqr_(&M, &K, &K, q_mat, &M, tau.data(), &wq, &lwork_q, &info);
        lwork_q = static_cast<__CLPK_integer>(wq);
        std::vector<float> work_q(static_cast<size_t>(std::max(lwork_q, __CLPK_integer(1))));

        info = 0;
        sorgqr_(&M, &K, &K, q_mat, &M, tau.data(), work_q.data(), &lwork_q, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgQR: sorgqr_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    return {Q, R};
}

// ============================================================================
// LinalgEigh: Eigenvalues/eigenvectors of symmetric matrix
// ============================================================================

std::vector<Tensor> mps_linalg_eigh_kernel(const Tensor& A) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto work_a = prepare_work_copy(A);
    float* a_data = work_a.data<float>();

    auto bshape = batch_shape(A);

    auto w_shape = bshape;
    w_shape.push_back(n);

    Tensor W(w_shape, DType::Float32, A.device());
    float* w_data = W.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = N;
    __CLPK_integer info = 0;
    char jobz = 'V'; // Compute eigenvalues and eigenvectors
    char uplo = 'U';

    // Query optimal workspace
    __CLPK_integer lwork = -1;
    __CLPK_integer liwork = -1;
    float work_query;
    __CLPK_integer iwork_query;
    ssyevd_(&jobz, &uplo, &N, a_data, &lda, w_data,
            &work_query, &lwork, &iwork_query, &liwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    liwork = iwork_query;
    std::vector<float> work(static_cast<size_t>(std::max(lwork, __CLPK_integer(1))));
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(std::max(liwork, __CLPK_integer(1))));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* w_vec = w_data + b * n;
        info = 0;
        ssyevd_(&jobz, &uplo, &N, a_mat, &lda, w_vec,
                work.data(), &lwork, iwork.data(), &liwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgEigh: ssyevd_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    // a_data now contains eigenvectors (column-major)
    return {W, work_a};
}

// ============================================================================
// LinalgEig: General eigenvalue decomposition
// ============================================================================

std::vector<Tensor> mps_linalg_eig_kernel(const Tensor& A) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto work_a = prepare_work_copy(A);
    float* a_data = work_a.data<float>();

    auto bshape = batch_shape(A);

    auto w_shape = bshape;
    w_shape.push_back(n);
    auto v_shape = bshape;
    v_shape.push_back(n);
    v_shape.push_back(n);

    Tensor W_real(w_shape, DType::Float32, A.device());
    Tensor W_imag(w_shape, DType::Float32, A.device());
    Tensor V(v_shape, DType::Float32, A.device());

    float* wr_data = W_real.data<float>();
    float* wi_data = W_imag.data<float>();
    float* v_data = V.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = N;
    __CLPK_integer ldvr = N;
    __CLPK_integer info = 0;
    char jobvl = 'N'; // No left eigenvectors
    char jobvr = 'V'; // Compute right eigenvectors

    // Query optimal workspace
    __CLPK_integer lwork = -1;
    float work_query;
    sgeev_(&jobvl, &jobvr, &N, a_data, &lda,
           wr_data, wi_data,
           nullptr, &N, v_data, &ldvr,
           &work_query, &lwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(static_cast<size_t>(std::max(lwork, __CLPK_integer(1))));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* wr = wr_data + b * n;
        float* wi = wi_data + b * n;
        float* vr = v_data + b * n * n;
        info = 0;
        sgeev_(&jobvl, &jobvr, &N, a_mat, &lda,
               wr, wi,
               nullptr, &N, vr, &ldvr,
               work.data(), &lwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgEig: sgeev_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    return {W_real, W_imag, V};
}

// ============================================================================
// LinalgLU: LU decomposition with partial pivoting
// ============================================================================

std::vector<Tensor> mps_linalg_lu_kernel(const Tensor& A) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto work_a = prepare_work_copy(A);
    float* a_data = work_a.data<float>();

    auto bshape = batch_shape(A);

    auto l_shape = bshape;
    l_shape.push_back(n);
    l_shape.push_back(n);
    auto u_shape = l_shape;

    auto p_shape = bshape;
    p_shape.push_back(n);

    Tensor L(l_shape, DType::Float32, A.device());
    Tensor U(u_shape, DType::Float32, A.device());
    Tensor pivots(p_shape, DType::Float32, A.device());

    float* l_data = L.data<float>();
    float* u_data = U.data<float>();
    float* p_data = pivots.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer lda = N;
    __CLPK_integer info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        info = 0;
        sgetrf_(&N, &N, a_mat, &lda, ipiv.data(), &info);
        if (info < 0) {
            throw std::runtime_error("MPS LinalgLU: sgetrf_ illegal argument");
        }

        float* l_mat = l_data + b * n * n;
        float* u_mat = u_data + b * n * n;
        float* p_vec = p_data + b * n;

        // Extract L (lower triangular with unit diagonal)
        // and U (upper triangular) from packed LU
        // LAPACK stores in column-major
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i < n; ++i) {
                float val = a_mat[j * n + i]; // column-major
                if (i > j) {
                    l_mat[j * n + i] = val;
                    u_mat[j * n + i] = 0.0f;
                } else if (i == j) {
                    l_mat[j * n + i] = 1.0f;
                    u_mat[j * n + i] = val;
                } else {
                    l_mat[j * n + i] = 0.0f;
                    u_mat[j * n + i] = val;
                }
            }
        }

        // Store pivot indices (convert to float for Tensor storage)
        for (int64_t i = 0; i < n; ++i) {
            p_vec[i] = static_cast<float>(ipiv[static_cast<size_t>(i)]);
        }
    }

    return {L, U, pivots};
}

// ============================================================================
// LinalgLUSolve: Solve using pre-computed LU factors
// ============================================================================

Tensor mps_linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                                   const Tensor& B) {
    auto shape = LU_data.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("MPS LinalgLUSolve: LU_data must be at least 2D");
    }
    int64_t n = shape[ndim - 1];
    int64_t batch = batch_count(LU_data);

    auto lu_work = prepare_work_copy(LU_data);
    auto b_work = prepare_work_copy(B);
    float* lu_data = lu_work.data<float>();
    float* b_data = b_work.data<float>();

    auto b_shape = B.shape();
    int64_t nrhs = (static_cast<int64_t>(b_shape.size()) >= 2) ? b_shape.back() : 1;

    // Convert float pivots back to integer
    auto pivot_cont = pivots.contiguous();
    const float* p_data = pivot_cont.data<float>();

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer NRHS = static_cast<__CLPK_integer>(nrhs);
    __CLPK_integer lda = N;
    __CLPK_integer ldb = N;
    __CLPK_integer info = 0;
    std::vector<__CLPK_integer> ipiv(static_cast<size_t>(n));
    char trans = 'N';

    for (int64_t b = 0; b < batch; ++b) {
        float* lu_mat = lu_data + b * n * n;
        float* b_mat = b_data + b * n * nrhs;
        const float* p_vec = p_data + b * n;

        for (int64_t i = 0; i < n; ++i) {
            ipiv[static_cast<size_t>(i)] = static_cast<__CLPK_integer>(p_vec[i]);
        }

        info = 0;
        sgetrs_(&trans, &N, &NRHS, lu_mat, &lda, ipiv.data(), b_mat, &ldb, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgLUSolve: sgetrs_ failed (error " +
                                     std::to_string(info) + ")");
        }
    }

    return b_work;
}

// ============================================================================
// SolveTriangular: Solve triangular system AX = B
// ============================================================================

Tensor mps_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                    bool upper, bool unitriangular) {
    auto [n, ndim] = check_square(A);
    int64_t batch = batch_count(A);

    auto a_cont = prepare_work_copy(A);
    auto b_work = prepare_work_copy(B);
    float* a_data = a_cont.data<float>();
    float* b_data = b_work.data<float>();

    auto b_shape = B.shape();
    int64_t nrhs = (static_cast<int64_t>(b_shape.size()) >= 2) ? b_shape.back() : 1;

    __CLPK_integer N = static_cast<__CLPK_integer>(n);
    __CLPK_integer NRHS = static_cast<__CLPK_integer>(nrhs);
    __CLPK_integer lda = N;
    __CLPK_integer ldb = N;
    __CLPK_integer info = 0;

    char side = 'L';
    char uplo_c = upper ? 'U' : 'L';
    char trans = 'N';
    char diag = unitriangular ? 'U' : 'N';
    float alpha = 1.0f;

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* b_mat = b_data + b * n * nrhs;

        // Use BLAS strsm for triangular solve (faster than LAPACK strtrs)
        cblas_strsm(CblasColMajor,
                     CblasLeft,
                     upper ? CblasUpper : CblasLower,
                     CblasNoTrans,
                     unitriangular ? CblasUnit : CblasNonUnit,
                     static_cast<int>(n),
                     static_cast<int>(nrhs),
                     alpha,
                     a_mat,
                     static_cast<int>(n),
                     b_mat,
                     static_cast<int>(n));
    }

    return b_work;
}

} // namespace tenzor::mps
