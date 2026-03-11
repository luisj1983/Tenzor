/**
 * @file linalg.cpp
 * @brief OneAPI/SYCL linear algebra kernels via oneMKL LAPACK
 *
 * Implements SVD, QR, Eigendecomposition, Solve, Inverse, Determinant,
 * and Cholesky factorization using oneMKL LAPACK APIs.
 * Guarded by TENZOR_HAS_ONEMKL_LAPACK (subset of TENZOR_HAS_ONEMKL).
 */

#include "tenzor/core/tensor.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <vector>
#include <cmath>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace tenzor {
namespace oneapi {

template<typename T>
inline auto get_data_ptr(const Tensor& t) -> T* {
    return static_cast<T*>(const_cast<void*>(t.data_ptr()));
}

auto clone_kernel(const Tensor& input, sycl::queue& queue) -> Tensor;

#ifdef TENZOR_HAS_ONEMKL

// ============================================================================
// LinalgDet - Determinant via LU factorization (getrf)
// ============================================================================
auto linalg_det_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];

    // Copy input for in-place LU
    Tensor a = clone_kernel(input, queue);
    Tensor output({1}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        // Column-major: transpose for oneMKL
        std::vector<float> h_a(n * n);
        queue.memcpy(h_a.data(), a.data_ptr(), n * n * sizeof(float)).wait();

        // Transpose to column-major
        std::vector<float> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        // Allocate device buffers
        float* d_a = sycl::malloc_device<float>(n * n, queue);
        std::int64_t* d_ipiv = sycl::malloc_device<std::int64_t>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(float)).wait();

        auto scratchpad_size = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        float* scratchpad = sycl::malloc_device<float>(scratchpad_size, queue);

        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a, n, d_ipiv, scratchpad, scratchpad_size).wait();

        // Det = product of diagonal * sign from pivots
        std::vector<float> h_lu(n * n);
        std::vector<std::int64_t> h_ipiv(n);
        queue.memcpy(h_lu.data(), d_a, n * n * sizeof(float)).wait();
        queue.memcpy(h_ipiv.data(), d_ipiv, n * sizeof(std::int64_t)).wait();

        float det = 1.0f;
        int swaps = 0;
        for (int64_t i = 0; i < n; ++i) {
            det *= h_lu[i * n + i]; // diagonal of column-major
            if (h_ipiv[i] != i + 1) swaps++;
        }
        if (swaps % 2) det = -det;

        queue.memcpy(const_cast<void*>(output.data_ptr()), &det, sizeof(float)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_ipiv, queue);
        sycl::free(scratchpad, queue);
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(n * n);
        queue.memcpy(h_a.data(), a.data_ptr(), n * n * sizeof(double)).wait();

        std::vector<double> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(n * n, queue);
        std::int64_t* d_ipiv = sycl::malloc_device<std::int64_t>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(double)).wait();

        auto scratchpad_size = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        double* scratchpad = sycl::malloc_device<double>(scratchpad_size, queue);

        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a, n, d_ipiv, scratchpad, scratchpad_size).wait();

        std::vector<double> h_lu(n * n);
        std::vector<std::int64_t> h_ipiv(n);
        queue.memcpy(h_lu.data(), d_a, n * n * sizeof(double)).wait();
        queue.memcpy(h_ipiv.data(), d_ipiv, n * sizeof(std::int64_t)).wait();

        double det = 1.0;
        int swaps = 0;
        for (int64_t i = 0; i < n; ++i) {
            det *= h_lu[i * n + i];
            if (h_ipiv[i] != i + 1) swaps++;
        }
        if (swaps % 2) det = -det;

        queue.memcpy(const_cast<void*>(output.data_ptr()), &det, sizeof(double)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_ipiv, queue);
        sycl::free(scratchpad, queue);
    } else {
        throw std::runtime_error("linalg_det: only Float32 and Float64 supported");
    }

    return output;
}

// ============================================================================
// LinalgInv - Matrix inverse via LU (getrf + getri)
// ============================================================================
auto linalg_inv_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];

    if (input.dtype() == DType::Float32) {
        std::vector<float> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(float)).wait();

        // Row-major to column-major
        std::vector<float> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        float* d_a = sycl::malloc_device<float>(n * n, queue);
        std::int64_t* d_ipiv = sycl::malloc_device<std::int64_t>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(float)).wait();

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        float* scratch_rf = sycl::malloc_device<float>(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a, n, d_ipiv, scratch_rf, sp_rf).wait();
        sycl::free(scratch_rf, queue);

        auto sp_ri = ::oneapi::mkl::lapack::getri_scratchpad_size<float>(queue, n, n);
        float* scratch_ri = sycl::malloc_device<float>(sp_ri, queue);
        ::oneapi::mkl::lapack::getri(queue, n, d_a, n, d_ipiv, scratch_ri, sp_ri).wait();
        sycl::free(scratch_ri, queue);

        // Column-major back to row-major
        std::vector<float> col_result(n * n);
        queue.memcpy(col_result.data(), d_a, n * n * sizeof(float)).wait();

        std::vector<float> h_out(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_out[i * n + j] = col_result[j * n + i];

        Tensor output({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), n * n * sizeof(float)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_ipiv, queue);
        return output;
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(double)).wait();

        std::vector<double> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(n * n, queue);
        std::int64_t* d_ipiv = sycl::malloc_device<std::int64_t>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(double)).wait();

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        double* scratch_rf = sycl::malloc_device<double>(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a, n, d_ipiv, scratch_rf, sp_rf).wait();
        sycl::free(scratch_rf, queue);

        auto sp_ri = ::oneapi::mkl::lapack::getri_scratchpad_size<double>(queue, n, n);
        double* scratch_ri = sycl::malloc_device<double>(sp_ri, queue);
        ::oneapi::mkl::lapack::getri(queue, n, d_a, n, d_ipiv, scratch_ri, sp_ri).wait();
        sycl::free(scratch_ri, queue);

        std::vector<double> col_result(n * n);
        queue.memcpy(col_result.data(), d_a, n * n * sizeof(double)).wait();

        std::vector<double> h_out(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_out[i * n + j] = col_result[j * n + i];

        Tensor output({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), n * n * sizeof(double)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_ipiv, queue);
        return output;
    } else {
        throw std::runtime_error("linalg_inv: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgSolve - Solve Ax=B via LU (getrf + getrs)
// ============================================================================
auto linalg_solve_kernel(const Tensor& A, const Tensor& B, sycl::queue& queue) -> Tensor {
    auto a_shape = A.shape();
    int64_t n = a_shape[a_shape.size() - 1];
    auto b_shape = B.shape();
    int64_t nrhs = (b_shape.size() > 1) ? b_shape[b_shape.size() - 1] : 1;

    if (A.dtype() == DType::Float32) {
        std::vector<float> h_a(n * n), h_b(n * nrhs);
        queue.memcpy(h_a.data(), A.data_ptr(), n * n * sizeof(float)).wait();
        queue.memcpy(h_b.data(), B.data_ptr(), n * nrhs * sizeof(float)).wait();

        // Transpose to column-major
        std::vector<float> col_a(n * n), col_b(n * nrhs);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < nrhs; ++j)
                col_b[j * n + i] = h_b[i * nrhs + j];

        float* d_a = sycl::malloc_device<float>(n * n, queue);
        float* d_b = sycl::malloc_device<float>(n * nrhs, queue);
        std::int64_t* d_ipiv = sycl::malloc_device<std::int64_t>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(float)).wait();
        queue.memcpy(d_b, col_b.data(), n * nrhs * sizeof(float)).wait();

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        float* scratch_rf = sycl::malloc_device<float>(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a, n, d_ipiv, scratch_rf, sp_rf).wait();
        sycl::free(scratch_rf, queue);

        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<float>(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        float* scratch_rs = sycl::malloc_device<float>(sp_rs, queue);
        ::oneapi::mkl::lapack::getrs(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, d_a, n, d_ipiv, d_b, n, scratch_rs, sp_rs).wait();
        sycl::free(scratch_rs, queue);

        // Column-major result back to row-major
        std::vector<float> col_result(n * nrhs);
        queue.memcpy(col_result.data(), d_b, n * nrhs * sizeof(float)).wait();

        std::vector<float> h_out(n * nrhs);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < nrhs; ++j)
                h_out[i * nrhs + j] = col_result[j * n + i];

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A.dtype(), A.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), n * nrhs * sizeof(float)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_b, queue);
        sycl::free(d_ipiv, queue);
        return output;
    } else if (A.dtype() == DType::Float64) {
        std::vector<double> h_a(n * n), h_b(n * nrhs);
        queue.memcpy(h_a.data(), A.data_ptr(), n * n * sizeof(double)).wait();
        queue.memcpy(h_b.data(), B.data_ptr(), n * nrhs * sizeof(double)).wait();

        std::vector<double> col_a(n * n), col_b(n * nrhs);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < nrhs; ++j)
                col_b[j * n + i] = h_b[i * nrhs + j];

        double* d_a = sycl::malloc_device<double>(n * n, queue);
        double* d_b = sycl::malloc_device<double>(n * nrhs, queue);
        std::int64_t* d_ipiv = sycl::malloc_device<std::int64_t>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(double)).wait();
        queue.memcpy(d_b, col_b.data(), n * nrhs * sizeof(double)).wait();

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        double* scratch_rf = sycl::malloc_device<double>(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a, n, d_ipiv, scratch_rf, sp_rf).wait();
        sycl::free(scratch_rf, queue);

        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<double>(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        double* scratch_rs = sycl::malloc_device<double>(sp_rs, queue);
        ::oneapi::mkl::lapack::getrs(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, d_a, n, d_ipiv, d_b, n, scratch_rs, sp_rs).wait();
        sycl::free(scratch_rs, queue);

        std::vector<double> col_result(n * nrhs);
        queue.memcpy(col_result.data(), d_b, n * nrhs * sizeof(double)).wait();

        std::vector<double> h_out(n * nrhs);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < nrhs; ++j)
                h_out[i * nrhs + j] = col_result[j * n + i];

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A.dtype(), A.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), n * nrhs * sizeof(double)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_b, queue);
        sycl::free(d_ipiv, queue);
        return output;
    } else {
        throw std::runtime_error("linalg_solve: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgSVD - Singular Value Decomposition via gesvd
// ============================================================================
auto linalg_svd_kernel(const Tensor& input, bool full_matrices, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = std::min(m, n);

    auto jobz = full_matrices ? ::oneapi::mkl::jobsvd::vectors : ::oneapi::mkl::jobsvd::somevec;
    int64_t u_cols = full_matrices ? m : k;
    int64_t vt_rows = full_matrices ? n : k;

    if (input.dtype() == DType::Float32) {
        std::vector<float> h_a(m * n);
        queue.memcpy(h_a.data(), input.data_ptr(), m * n * sizeof(float)).wait();

        // Transpose to column-major for gesvd
        std::vector<float> col_a(m * n);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * m + i] = h_a[i * n + j];

        float* d_a = sycl::malloc_device<float>(m * n, queue);
        float* d_s = sycl::malloc_device<float>(k, queue);
        float* d_u = sycl::malloc_device<float>(m * u_cols, queue);
        float* d_vt = sycl::malloc_device<float>(vt_rows * n, queue);
        queue.memcpy(d_a, col_a.data(), m * n * sizeof(float)).wait();

        auto sp = ::oneapi::mkl::lapack::gesvd_scratchpad_size<float>(queue, jobz, jobz, m, n, m, m, n);
        float* scratch = sycl::malloc_device<float>(sp, queue);
        ::oneapi::mkl::lapack::gesvd(queue, jobz, jobz, m, n, d_a, m, d_s, d_u, m, d_vt, n, scratch, sp).wait();

        Tensor S({k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(S.data_ptr()), d_s, k * sizeof(float)).wait();

        // U: column-major (m x u_cols) -> row-major
        std::vector<float> col_u(m * u_cols);
        queue.memcpy(col_u.data(), d_u, m * u_cols * sizeof(float)).wait();
        std::vector<float> h_u(m * u_cols);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < u_cols; ++j)
                h_u[i * u_cols + j] = col_u[j * m + i];
        Tensor U({m, u_cols}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(U.data_ptr()), h_u.data(), m * u_cols * sizeof(float)).wait();

        // Vt: column-major (vt_rows x n) -> row-major
        std::vector<float> col_vt(vt_rows * n);
        queue.memcpy(col_vt.data(), d_vt, vt_rows * n * sizeof(float)).wait();
        std::vector<float> h_vt(vt_rows * n);
        for (int64_t i = 0; i < vt_rows; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_vt[i * n + j] = col_vt[j * vt_rows + i];
        Tensor Vt({vt_rows, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(Vt.data_ptr()), h_vt.data(), vt_rows * n * sizeof(float)).wait();

        sycl::free(d_a, queue); sycl::free(d_s, queue);
        sycl::free(d_u, queue); sycl::free(d_vt, queue);
        sycl::free(scratch, queue);

        return {U, S, Vt};
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(m * n);
        queue.memcpy(h_a.data(), input.data_ptr(), m * n * sizeof(double)).wait();

        std::vector<double> col_a(m * n);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * m + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(m * n, queue);
        double* d_s = sycl::malloc_device<double>(k, queue);
        double* d_u = sycl::malloc_device<double>(m * u_cols, queue);
        double* d_vt = sycl::malloc_device<double>(vt_rows * n, queue);
        queue.memcpy(d_a, col_a.data(), m * n * sizeof(double)).wait();

        auto sp = ::oneapi::mkl::lapack::gesvd_scratchpad_size<double>(queue, jobz, jobz, m, n, m, m, n);
        double* scratch = sycl::malloc_device<double>(sp, queue);
        ::oneapi::mkl::lapack::gesvd(queue, jobz, jobz, m, n, d_a, m, d_s, d_u, m, d_vt, n, scratch, sp).wait();

        Tensor S({k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(S.data_ptr()), d_s, k * sizeof(double)).wait();

        std::vector<double> col_u(m * u_cols);
        queue.memcpy(col_u.data(), d_u, m * u_cols * sizeof(double)).wait();
        std::vector<double> h_u(m * u_cols);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < u_cols; ++j)
                h_u[i * u_cols + j] = col_u[j * m + i];
        Tensor U({m, u_cols}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(U.data_ptr()), h_u.data(), m * u_cols * sizeof(double)).wait();

        std::vector<double> col_vt(vt_rows * n);
        queue.memcpy(col_vt.data(), d_vt, vt_rows * n * sizeof(double)).wait();
        std::vector<double> h_vt(vt_rows * n);
        for (int64_t i = 0; i < vt_rows; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_vt[i * n + j] = col_vt[j * vt_rows + i];
        Tensor Vt({vt_rows, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(Vt.data_ptr()), h_vt.data(), vt_rows * n * sizeof(double)).wait();

        sycl::free(d_a, queue); sycl::free(d_s, queue);
        sycl::free(d_u, queue); sycl::free(d_vt, queue);
        sycl::free(scratch, queue);

        return {U, S, Vt};
    } else {
        throw std::runtime_error("linalg_svd: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgQR - QR Decomposition via geqrf + orgqr
// ============================================================================
auto linalg_qr_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = std::min(m, n);

    if (input.dtype() == DType::Float32) {
        std::vector<float> h_a(m * n);
        queue.memcpy(h_a.data(), input.data_ptr(), m * n * sizeof(float)).wait();

        std::vector<float> col_a(m * n);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * m + i] = h_a[i * n + j];

        float* d_a = sycl::malloc_device<float>(m * n, queue);
        float* d_tau = sycl::malloc_device<float>(k, queue);
        queue.memcpy(d_a, col_a.data(), m * n * sizeof(float)).wait();

        auto sp_qr = ::oneapi::mkl::lapack::geqrf_scratchpad_size<float>(queue, m, n, m);
        float* scratch_qr = sycl::malloc_device<float>(sp_qr, queue);
        ::oneapi::mkl::lapack::geqrf(queue, m, n, d_a, m, d_tau, scratch_qr, sp_qr).wait();
        sycl::free(scratch_qr, queue);

        // Extract R (upper triangular from d_a, column-major)
        std::vector<float> col_qr(m * n);
        queue.memcpy(col_qr.data(), d_a, m * n * sizeof(float)).wait();

        std::vector<float> h_r(k * n, 0.0f);
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = i; j < n; ++j)
                h_r[i * n + j] = col_qr[j * m + i]; // col-major to row-major

        Tensor R({k, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(R.data_ptr()), h_r.data(), k * n * sizeof(float)).wait();

        // Generate Q via orgqr
        auto sp_oq = ::oneapi::mkl::lapack::orgqr_scratchpad_size<float>(queue, m, k, k, m);
        float* scratch_oq = sycl::malloc_device<float>(sp_oq, queue);
        ::oneapi::mkl::lapack::orgqr(queue, m, k, k, d_a, m, d_tau, scratch_oq, sp_oq).wait();
        sycl::free(scratch_oq, queue);

        std::vector<float> col_q(m * k);
        queue.memcpy(col_q.data(), d_a, m * k * sizeof(float)).wait();

        std::vector<float> h_q(m * k);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < k; ++j)
                h_q[i * k + j] = col_q[j * m + i];

        Tensor Q({m, k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(Q.data_ptr()), h_q.data(), m * k * sizeof(float)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_tau, queue);
        return {Q, R};
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(m * n);
        queue.memcpy(h_a.data(), input.data_ptr(), m * n * sizeof(double)).wait();

        std::vector<double> col_a(m * n);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * m + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(m * n, queue);
        double* d_tau = sycl::malloc_device<double>(k, queue);
        queue.memcpy(d_a, col_a.data(), m * n * sizeof(double)).wait();

        auto sp_qr = ::oneapi::mkl::lapack::geqrf_scratchpad_size<double>(queue, m, n, m);
        double* scratch_qr = sycl::malloc_device<double>(sp_qr, queue);
        ::oneapi::mkl::lapack::geqrf(queue, m, n, d_a, m, d_tau, scratch_qr, sp_qr).wait();
        sycl::free(scratch_qr, queue);

        std::vector<double> col_qr(m * n);
        queue.memcpy(col_qr.data(), d_a, m * n * sizeof(double)).wait();

        std::vector<double> h_r(k * n, 0.0);
        for (int64_t i = 0; i < k; ++i)
            for (int64_t j = i; j < n; ++j)
                h_r[i * n + j] = col_qr[j * m + i];

        Tensor R({k, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(R.data_ptr()), h_r.data(), k * n * sizeof(double)).wait();

        auto sp_oq = ::oneapi::mkl::lapack::orgqr_scratchpad_size<double>(queue, m, k, k, m);
        double* scratch_oq = sycl::malloc_device<double>(sp_oq, queue);
        ::oneapi::mkl::lapack::orgqr(queue, m, k, k, d_a, m, d_tau, scratch_oq, sp_oq).wait();
        sycl::free(scratch_oq, queue);

        std::vector<double> col_q(m * k);
        queue.memcpy(col_q.data(), d_a, m * k * sizeof(double)).wait();

        std::vector<double> h_q(m * k);
        for (int64_t i = 0; i < m; ++i)
            for (int64_t j = 0; j < k; ++j)
                h_q[i * k + j] = col_q[j * m + i];

        Tensor Q({m, k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(Q.data_ptr()), h_q.data(), m * k * sizeof(double)).wait();

        sycl::free(d_a, queue);
        sycl::free(d_tau, queue);
        return {Q, R};
    } else {
        throw std::runtime_error("linalg_qr: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgEigh - Symmetric eigendecomposition via syevd
// ============================================================================
auto linalg_eigh_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];

    if (input.dtype() == DType::Float32) {
        std::vector<float> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(float)).wait();

        std::vector<float> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        float* d_a = sycl::malloc_device<float>(n * n, queue);
        float* d_w = sycl::malloc_device<float>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(float)).wait();

        auto sp = ::oneapi::mkl::lapack::syevd_scratchpad_size<float>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower, n, n);
        float* scratch = sycl::malloc_device<float>(sp, queue);
        ::oneapi::mkl::lapack::syevd(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower,
                                   n, d_a, n, d_w, scratch, sp).wait();

        Tensor W({n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(W.data_ptr()), d_w, n * sizeof(float)).wait();

        std::vector<float> col_v(n * n);
        queue.memcpy(col_v.data(), d_a, n * n * sizeof(float)).wait();
        std::vector<float> h_v(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_v[i * n + j] = col_v[j * n + i];

        Tensor V({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(V.data_ptr()), h_v.data(), n * n * sizeof(float)).wait();

        sycl::free(d_a, queue); sycl::free(d_w, queue); sycl::free(scratch, queue);
        return {W, V};
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(double)).wait();

        std::vector<double> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(n * n, queue);
        double* d_w = sycl::malloc_device<double>(n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(double)).wait();

        auto sp = ::oneapi::mkl::lapack::syevd_scratchpad_size<double>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower, n, n);
        double* scratch = sycl::malloc_device<double>(sp, queue);
        ::oneapi::mkl::lapack::syevd(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower,
                                   n, d_a, n, d_w, scratch, sp).wait();

        Tensor W({n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(W.data_ptr()), d_w, n * sizeof(double)).wait();

        std::vector<double> col_v(n * n);
        queue.memcpy(col_v.data(), d_a, n * n * sizeof(double)).wait();
        std::vector<double> h_v(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_v[i * n + j] = col_v[j * n + i];

        Tensor V({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(V.data_ptr()), h_v.data(), n * n * sizeof(double)).wait();

        sycl::free(d_a, queue); sycl::free(d_w, queue); sycl::free(scratch, queue);
        return {W, V};
    } else {
        throw std::runtime_error("linalg_eigh: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgEig - Non-symmetric eigendecomposition via geev
// ============================================================================
auto linalg_eig_kernel(const Tensor& input, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];

    if (input.dtype() == DType::Float32) {
        std::vector<float> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(float)).wait();

        // Row-major to column-major
        std::vector<float> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        float* d_a = sycl::malloc_device<float>(n * n, queue);
        float* d_wr = sycl::malloc_device<float>(n, queue);
        float* d_wi = sycl::malloc_device<float>(n, queue);
        float* d_vl = sycl::malloc_device<float>(n * n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(float)).wait();

        auto sp = ::oneapi::mkl::lapack::geev_scratchpad_size<float>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec, n, n, n, n);
        float* scratch = sycl::malloc_device<float>(sp, queue);

        // Left eigenvectors of A^T (col-major) = right eigenvectors of A (row-major)
        ::oneapi::mkl::lapack::geev(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec,
                                   n, d_a, n, d_wr, d_wi,
                                   d_vl, n, nullptr, n,
                                   scratch, sp).wait();

        Tensor WR({n}, input.dtype(), input.device());
        Tensor WI({n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(WR.data_ptr()), d_wr, n * sizeof(float));
        queue.memcpy(const_cast<void*>(WI.data_ptr()), d_wi, n * sizeof(float)).wait();

        // Transpose VL from column-major back to row-major
        std::vector<float> col_v(n * n);
        queue.memcpy(col_v.data(), d_vl, n * n * sizeof(float)).wait();
        std::vector<float> h_v(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_v[i * n + j] = col_v[j * n + i];

        Tensor V({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(V.data_ptr()), h_v.data(), n * n * sizeof(float)).wait();

        sycl::free(d_a, queue); sycl::free(d_wr, queue); sycl::free(d_wi, queue);
        sycl::free(d_vl, queue); sycl::free(scratch, queue);
        return {WR, WI, V};
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(double)).wait();

        std::vector<double> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(n * n, queue);
        double* d_wr = sycl::malloc_device<double>(n, queue);
        double* d_wi = sycl::malloc_device<double>(n, queue);
        double* d_vl = sycl::malloc_device<double>(n * n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(double)).wait();

        auto sp = ::oneapi::mkl::lapack::geev_scratchpad_size<double>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec, n, n, n, n);
        double* scratch = sycl::malloc_device<double>(sp, queue);

        ::oneapi::mkl::lapack::geev(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec,
                                   n, d_a, n, d_wr, d_wi,
                                   d_vl, n, nullptr, n,
                                   scratch, sp).wait();

        Tensor WR({n}, input.dtype(), input.device());
        Tensor WI({n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(WR.data_ptr()), d_wr, n * sizeof(double));
        queue.memcpy(const_cast<void*>(WI.data_ptr()), d_wi, n * sizeof(double)).wait();

        std::vector<double> col_v(n * n);
        queue.memcpy(col_v.data(), d_vl, n * n * sizeof(double)).wait();
        std::vector<double> h_v(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_v[i * n + j] = col_v[j * n + i];

        Tensor V({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(V.data_ptr()), h_v.data(), n * n * sizeof(double)).wait();

        sycl::free(d_a, queue); sycl::free(d_wr, queue); sycl::free(d_wi, queue);
        sycl::free(d_vl, queue); sycl::free(scratch, queue);
        return {WR, WI, V};
    } else {
        throw std::runtime_error("linalg_eig: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgCholesky - Cholesky factorization via potrf
// ============================================================================
auto linalg_cholesky_kernel(const Tensor& input, bool upper, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];
    auto uplo = upper ? ::oneapi::mkl::uplo::upper : ::oneapi::mkl::uplo::lower;

    if (input.dtype() == DType::Float32) {
        std::vector<float> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(float)).wait();

        std::vector<float> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        float* d_a = sycl::malloc_device<float>(n * n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(float)).wait();

        auto sp = ::oneapi::mkl::lapack::potrf_scratchpad_size<float>(queue, uplo, n, n);
        float* scratch = sycl::malloc_device<float>(sp, queue);
        ::oneapi::mkl::lapack::potrf(queue, uplo, n, d_a, n, scratch, sp).wait();

        std::vector<float> col_result(n * n);
        queue.memcpy(col_result.data(), d_a, n * n * sizeof(float)).wait();

        // Convert back to row-major and zero out the other triangle
        std::vector<float> h_out(n * n, 0.0f);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_out[i * n + j] = col_result[j * n + i];

        // Zero out appropriate triangle
        if (upper) {
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < i; ++j)
                    h_out[i * n + j] = 0.0f;
        } else {
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = i + 1; j < n; ++j)
                    h_out[i * n + j] = 0.0f;
        }

        Tensor output({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), n * n * sizeof(float)).wait();

        sycl::free(d_a, queue); sycl::free(scratch, queue);
        return output;
    } else if (input.dtype() == DType::Float64) {
        std::vector<double> h_a(n * n);
        queue.memcpy(h_a.data(), input.data_ptr(), n * n * sizeof(double)).wait();

        std::vector<double> col_a(n * n);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                col_a[j * n + i] = h_a[i * n + j];

        double* d_a = sycl::malloc_device<double>(n * n, queue);
        queue.memcpy(d_a, col_a.data(), n * n * sizeof(double)).wait();

        auto sp = ::oneapi::mkl::lapack::potrf_scratchpad_size<double>(queue, uplo, n, n);
        double* scratch = sycl::malloc_device<double>(sp, queue);
        ::oneapi::mkl::lapack::potrf(queue, uplo, n, d_a, n, scratch, sp).wait();

        std::vector<double> col_result(n * n);
        queue.memcpy(col_result.data(), d_a, n * n * sizeof(double)).wait();

        std::vector<double> h_out(n * n, 0.0);
        for (int64_t i = 0; i < n; ++i)
            for (int64_t j = 0; j < n; ++j)
                h_out[i * n + j] = col_result[j * n + i];

        if (upper) {
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = 0; j < i; ++j)
                    h_out[i * n + j] = 0.0;
        } else {
            for (int64_t i = 0; i < n; ++i)
                for (int64_t j = i + 1; j < n; ++j)
                    h_out[i * n + j] = 0.0;
        }

        Tensor output({n, n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(output.data_ptr()), h_out.data(), n * n * sizeof(double)).wait();

        sycl::free(d_a, queue); sycl::free(scratch, queue);
        return output;
    } else {
        throw std::runtime_error("linalg_cholesky: only Float32 and Float64 supported");
    }
}

#endif // TENZOR_HAS_ONEMKL

} // namespace oneapi
} // namespace tenzor
