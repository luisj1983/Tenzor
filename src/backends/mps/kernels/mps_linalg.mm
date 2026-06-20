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

// ----------------------------------------------------------------------------
// Row-major <-> column-major conversion helpers.
//
// Tenzor tensors are row-major contiguous, but Fortran CLAPACK (sgetrf_, sgesv_,
// sgesdd_, ...) interprets buffers as COLUMN-major. To keep cross-backend parity
// with the CPU reference (which uses LAPACKE row-major), we explicitly transpose
// the row-major data into a column-major scratch buffer before each LAPACK call
// and transpose any matrix results back to row-major afterwards.
// ----------------------------------------------------------------------------

/// Convert a row-major (rows x cols) matrix at src into column-major layout at dst.
/// dst[j*rows + i] = src[i*cols + j]
inline void row_to_col_major(const float* src, float* dst, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[j * rows + i] = src[i * cols + j];
        }
    }
}

/// Convert a column-major (rows x cols) matrix at src into row-major layout at dst.
/// dst[i*cols + j] = src[j*rows + i]
inline void col_to_row_major(const float* src, float* dst, int64_t rows, int64_t cols) {
    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            dst[i * cols + j] = src[j * rows + i];
        }
    }
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
    // If B is (batch, n) it represents a single RHS column; otherwise the trailing
    // dimension is the number of right-hand-side columns.
    int64_t nrhs;
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

    // Scratch column-major buffers. Tenzor storage is row-major; Fortran CLAPACK
    // expects column-major, so transpose into scratch, solve, transpose result back.
    std::vector<float> a_col(static_cast<size_t>(n * n));
    std::vector<float> b_col(static_cast<size_t>(n * nrhs));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* b_mat = b_data + b * n * nrhs;

        row_to_col_major(a_mat, a_col.data(), n, n);
        row_to_col_major(b_mat, b_col.data(), n, nrhs);

        info = 0;
        sgesv_(&N, &NRHS, a_col.data(), &lda, ipiv.data(), b_col.data(), &ldb, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgSolve: sgesv_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // b_col is the column-major solution (n x nrhs); write back row-major.
        col_to_row_major(b_col.data(), b_mat, n, nrhs);
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

    // Column-major scratch buffer; transpose row-major input in, factor, transpose out.
    std::vector<float> a_col(static_cast<size_t>(n * n));

    for (int64_t b = 0; b < batch; ++b) {
        float* mat = data + b * n * n; // row-major storage

        row_to_col_major(mat, a_col.data(), n, n);

        info = 0;
        spotrf_(&uplo, &N, a_col.data(), &lda, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgCholesky: spotrf_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // Write factor back to row-major storage.
        col_to_row_major(a_col.data(), mat, n, n);

        // Zero out the opposite triangle (row-major indexing: (i,j) = i*n + j).
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                if (upper && j < i) {
                    mat[i * n + j] = 0.0f;
                } else if (!upper && j > i) {
                    mat[i * n + j] = 0.0f;
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

    // Column-major scratch buffers. Tenzor storage is row-major; Fortran sgesdd_
    // expects column-major. We transpose A into a_col, run the decomposition into
    // column-major U/Vt scratch, then transpose those back into the row-major
    // Tenzor outputs so the result matches the CPU (LAPACKE row-major) reference.
    std::vector<float> a_col(static_cast<size_t>(m * n));
    std::vector<float> u_col(static_cast<size_t>(m * u_cols));
    std::vector<float> vt_col(static_cast<size_t>(vt_rows * n));

    // Query optimal workspace
    __CLPK_integer lwork = -1;
    float work_query;
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(8 * k));
    sgesdd_(&jobz, &M, &N, a_col.data(), &lda, s_data, u_col.data(), &ldu,
            vt_col.data(), &ldvt, &work_query, &lwork, iwork.data(), &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(static_cast<size_t>(lwork));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * m * n;
        float* u_mat = u_data + b * m * u_cols;
        float* s_vec = s_data + b * k;
        float* vt_mat = vt_data + b * vt_rows * n;

        row_to_col_major(a_mat, a_col.data(), m, n);

        info = 0;
        sgesdd_(&jobz, &M, &N, a_col.data(), &lda, s_vec, u_col.data(), &ldu,
                vt_col.data(), &ldvt, work.data(), &lwork, iwork.data(), &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgSVD: sgesdd_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // U is column-major (m x u_cols), Vt is column-major (vt_rows x n).
        col_to_row_major(u_col.data(), u_mat, m, u_cols);
        col_to_row_major(vt_col.data(), vt_mat, vt_rows, n);
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

    // Column-major scratch buffers: a_col holds A and the packed QR factors;
    // q_col holds the explicitly-formed Q. Tenzor storage is row-major, so we
    // transpose in/out around the Fortran CLAPACK calls.
    std::vector<float> a_col(static_cast<size_t>(m * n));
    std::vector<float> q_col(static_cast<size_t>(m * k));

    // Query optimal workspace for sgeqrf
    __CLPK_integer lwork = -1;
    float work_query;
    sgeqrf_(&M, &N, a_col.data(), &lda, tau.data(), &work_query, &lwork, &info);
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

        row_to_col_major(a_mat, a_col.data(), m, n);

        // QR factorization (column-major in a_col)
        sgeqrf_(&M, &N, a_col.data(), &lda, tau.data(), work.data(), &lwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgQR: sgeqrf_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // Extract R (k x n upper-triangular) from the column-major factored matrix
        // and store it row-major: R[i,j] = a_col[j*m + i] for i <= j.
        float* r_mat = r_data + b * k * n;
        std::memset(r_mat, 0, static_cast<size_t>(k * n) * sizeof(float));
        for (int64_t j = 0; j < n; ++j) {
            for (int64_t i = 0; i <= std::min(j, k - 1); ++i) {
                r_mat[i * n + j] = a_col[j * m + i]; // col-major source -> row-major dest
            }
        }

        // Generate Q (m x k) from Householder reflectors. Copy the first k columns
        // of the column-major factored matrix into q_col, then call sorgqr_.
        std::memcpy(q_col.data(), a_col.data(),
                    static_cast<size_t>(m * k) * sizeof(float));

        __CLPK_integer lwork_q = -1;
        float wq;
        sorgqr_(&M, &K, &K, q_col.data(), &M, tau.data(), &wq, &lwork_q, &info);
        lwork_q = static_cast<__CLPK_integer>(wq);
        std::vector<float> work_q(static_cast<size_t>(std::max(lwork_q, __CLPK_integer(1))));

        info = 0;
        sorgqr_(&M, &K, &K, q_col.data(), &M, tau.data(), work_q.data(), &lwork_q, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgQR: sorgqr_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // q_col is column-major (m x k); store row-major into Q.
        float* q_mat = q_data + b * m * k;
        col_to_row_major(q_col.data(), q_mat, m, k);
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

    // Column-major scratch buffer. We transpose A in (uplo='U' then reads the true
    // upper triangle of A, matching the CPU LAPACKE row-major reference), and
    // transpose the eigenvector matrix back to row-major afterwards.
    std::vector<float> a_col(static_cast<size_t>(n * n));

    // Query optimal workspace
    __CLPK_integer lwork = -1;
    __CLPK_integer liwork = -1;
    float work_query;
    __CLPK_integer iwork_query;
    ssyevd_(&jobz, &uplo, &N, a_col.data(), &lda, w_data,
            &work_query, &lwork, &iwork_query, &liwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    liwork = iwork_query;
    std::vector<float> work(static_cast<size_t>(std::max(lwork, __CLPK_integer(1))));
    std::vector<__CLPK_integer> iwork(static_cast<size_t>(std::max(liwork, __CLPK_integer(1))));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* w_vec = w_data + b * n;

        row_to_col_major(a_mat, a_col.data(), n, n);

        info = 0;
        ssyevd_(&jobz, &uplo, &N, a_col.data(), &lda, w_vec,
                work.data(), &lwork, iwork.data(), &liwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgEigh: ssyevd_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // a_col now holds the eigenvectors as columns (column-major); store the
        // eigenvector matrix row-major into work_a so columns remain eigenvectors.
        col_to_row_major(a_col.data(), a_mat, n, n);
    }

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

    // Column-major scratch buffers. Transpose A in, transpose the right
    // eigenvector matrix out, reproducing the CPU LAPACKE row-major convention
    // (including the packed real/imag storage for complex-conjugate pairs).
    std::vector<float> a_col(static_cast<size_t>(n * n));
    std::vector<float> vr_col(static_cast<size_t>(n * n));

    // Query optimal workspace
    __CLPK_integer lwork = -1;
    float work_query;
    sgeev_(&jobvl, &jobvr, &N, a_col.data(), &lda,
           wr_data, wi_data,
           nullptr, &N, vr_col.data(), &ldvr,
           &work_query, &lwork, &info);
    lwork = static_cast<__CLPK_integer>(work_query);
    std::vector<float> work(static_cast<size_t>(std::max(lwork, __CLPK_integer(1))));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* wr = wr_data + b * n;
        float* wi = wi_data + b * n;
        float* vr = v_data + b * n * n;

        row_to_col_major(a_mat, a_col.data(), n, n);

        info = 0;
        sgeev_(&jobvl, &jobvr, &N, a_col.data(), &lda,
               wr, wi,
               nullptr, &N, vr_col.data(), &ldvr,
               work.data(), &lwork, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgEig: sgeev_ failed (error " +
                                     std::to_string(info) + ")");
        }

        // vr_col is column-major (n x n); store row-major into V.
        col_to_row_major(vr_col.data(), vr, n, n);
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

    // Column-major scratch buffer holding A and the packed LU factors. Transpose
    // the row-major input in; the resulting factorization (PA = LU, partial
    // pivoting) matches the CPU LAPACKE row-major reference.
    std::vector<float> a_col(static_cast<size_t>(n * n));

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;

        row_to_col_major(a_mat, a_col.data(), n, n);

        info = 0;
        sgetrf_(&N, &N, a_col.data(), &lda, ipiv.data(), &info);
        if (info < 0) {
            throw std::runtime_error("MPS LinalgLU: sgetrf_ illegal argument");
        }

        float* l_mat = l_data + b * n * n;
        float* u_mat = u_data + b * n * n;
        float* p_vec = p_data + b * n;

        // Extract L (unit lower triangular) and U (upper triangular) from the
        // packed column-major factored matrix, storing them row-major:
        //   element (i,j) lives at a_col[j*n + i] (column-major) and is written
        //   to l_mat/u_mat at [i*n + j] (row-major).
        for (int64_t i = 0; i < n; ++i) {
            for (int64_t j = 0; j < n; ++j) {
                float val = a_col[j * n + i]; // column-major source
                if (i > j) {
                    l_mat[i * n + j] = val;
                    u_mat[i * n + j] = 0.0f;
                } else if (i == j) {
                    l_mat[i * n + j] = 1.0f;
                    u_mat[i * n + j] = val;
                } else {
                    l_mat[i * n + j] = 0.0f;
                    u_mat[i * n + j] = val;
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

    // The packed LU factors and RHS are row-major; transpose into column-major
    // scratch for Fortran sgetrs_, then transpose the solution back to row-major.
    std::vector<float> lu_col(static_cast<size_t>(n * n));
    std::vector<float> b_col(static_cast<size_t>(n * nrhs));

    for (int64_t b = 0; b < batch; ++b) {
        float* lu_mat = lu_data + b * n * n;
        float* b_mat = b_data + b * n * nrhs;
        const float* p_vec = p_data + b * n;

        for (int64_t i = 0; i < n; ++i) {
            ipiv[static_cast<size_t>(i)] = static_cast<__CLPK_integer>(p_vec[i]);
        }

        row_to_col_major(lu_mat, lu_col.data(), n, n);
        row_to_col_major(b_mat, b_col.data(), n, nrhs);

        info = 0;
        sgetrs_(&trans, &N, &NRHS, lu_col.data(), &lda, ipiv.data(), b_col.data(), &ldb, &info);
        if (info != 0) {
            throw std::runtime_error("MPS LinalgLUSolve: sgetrs_ failed (error " +
                                     std::to_string(info) + ")");
        }

        col_to_row_major(b_col.data(), b_mat, n, nrhs);
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

    // Tenzor storage is row-major; use the row-major CBLAS layout (matching the
    // CPU reference) so no transpose is needed. Row-major leading dimensions:
    // lda = #cols of A = n, ldb = #cols of B = nrhs.
    const float alpha = 1.0f;
    const auto ln = static_cast<int>(n);
    const auto lnrhs = static_cast<int>(nrhs);

    for (int64_t b = 0; b < batch; ++b) {
        float* a_mat = a_data + b * n * n;
        float* b_mat = b_data + b * n * nrhs;

        // Use BLAS strsm for triangular solve (faster than LAPACK strtrs)
        cblas_strsm(CblasRowMajor,
                     CblasLeft,
                     upper ? CblasUpper : CblasLower,
                     CblasNoTrans,
                     unitriangular ? CblasUnit : CblasNonUnit,
                     ln,
                     lnrhs,
                     alpha,
                     a_mat,
                     ln,
                     b_mat,
                     lnrhs);
    }

    return b_work;
}

} // namespace tenzor::mps
