/**
 * @file linalg.cpp
 * @brief OneAPI/SYCL linear algebra kernels via oneMKL LAPACK
 *
 * Implements SVD, QR, Eigendecomposition, Solve, Inverse, Determinant,
 * and Cholesky factorization using oneMKL LAPACK APIs.
 * Guarded by TENZOR_HAS_ONEMKL_LAPACK (subset of TENZOR_HAS_ONEMKL).
 */

#include "tenzor/core/tensor.hpp"
#include "../sycl_buffer_guard.hpp"
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

// SYCL kernel name types for device-side transpose
class SyclTransposeF32;
class SyclTransposeF64;
class SyclTransposeF32Back;
class SyclTransposeF64Back;

/**
 * @brief Device-side tiled matrix transpose.
 *
 * Uses 16x16 tile blocking with local memory to avoid non-coalesced global
 * memory accesses.  Replaces the host memcpy + loop transposes that were
 * creating D→H→D round-trips in every linalg kernel.
 *
 * @tparam T         Element type (float / double)
 * @tparam KernelName  Unique SYCL kernel name type
 * @param dst        Destination device pointer (row-major output)
 * @param src        Source device pointer (row-major input)
 * @param rows       Number of rows in the source matrix
 * @param cols       Number of columns in the source matrix
 * @param queue      SYCL queue
 */
template<typename T, typename KernelName>
static void sycl_transpose(T* dst, const T* src, int64_t rows, int64_t cols,
                            sycl::queue& queue) {
    constexpr int TILE = 16;
    // Pad grid to full tiles
    const int64_t grid_rows = ((rows + TILE - 1) / TILE) * TILE;
    const int64_t grid_cols = ((cols + TILE - 1) / TILE) * TILE;

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<T, 2> tile(sycl::range<2>(TILE, TILE), cgh);

        cgh.parallel_for<KernelName>(
            sycl::nd_range<2>({static_cast<size_t>(grid_rows),
                               static_cast<size_t>(grid_cols)},
                              {TILE, TILE}),
            [=](sycl::nd_item<2> item) {
                const int64_t r = item.get_global_id(0);
                const int64_t c = item.get_global_id(1);
                const int lr = item.get_local_id(0);
                const int lc = item.get_local_id(1);

                // Load from src[r][c] into shared tile[lr][lc]
                if (r < rows && c < cols) {
                    tile[lr][lc] = src[r * cols + c];
                }
                sycl::group_barrier(item.get_group());

                // Transposed global position
                const int64_t tile_row_start = item.get_group(0) * TILE;
                const int64_t tile_col_start = item.get_group(1) * TILE;
                const int64_t dst_r = tile_col_start + lr;  // swapped
                const int64_t dst_c = tile_row_start + lc;  // swapped

                if (dst_r < cols && dst_c < rows) {
                    // Read tile[lc][lr] (transposed indices)
                    dst[dst_r * rows + dst_c] = tile[lc][lr];
                }
            });
    }).wait();
}

/**
 * @brief Helper: copy row-major device data to column-major device buffer.
 *
 * Equivalent to transposing src(rows x cols) into dst(cols x rows) in
 * column-major layout (i.e. dst[j*rows + i] = src[i*cols + j]).
 * This IS a plain transpose, so we just call sycl_transpose.
 */
template<typename T, typename KernelName>
static void row_to_col_major(T* dst, const T* src, int64_t rows, int64_t cols,
                              sycl::queue& queue) {
    sycl_transpose<T, KernelName>(dst, src, rows, cols, queue);
}

/**
 * @brief Helper: copy column-major device data to row-major device buffer.
 *
 * col_src is (rows x cols) stored in column-major order.
 * We need row_dst[i*cols + j] = col_src[j*rows + i], which is
 * transpose of col_src viewed as a (cols x rows) row-major matrix.
 */
template<typename T, typename KernelName>
static void col_to_row_major(T* dst, const T* src, int64_t rows, int64_t cols,
                              sycl::queue& queue) {
    // src is (cols x rows) when read row-major, transpose gives (rows x cols)
    sycl_transpose<T, KernelName>(dst, src, cols, rows, queue);
}

#ifdef TENZOR_HAS_ONEMKL

// ============================================================================
// LinalgDet - Determinant via LU factorization (getrf)
// ============================================================================
// Additional kernel names for multiple transpose calls in different functions
class SyclTransposeDetF32;
class SyclTransposeDetF64;
class SyclTransposeInvF32;
class SyclTransposeInvF64;
class SyclTransposeInvBackF32;
class SyclTransposeInvBackF64;
class SyclTransposeSolveAF32;
class SyclTransposeSolveAF64;
class SyclTransposeSolveBF32;
class SyclTransposeSolveBF64;
class SyclTransposeSolveBackF32;
class SyclTransposeSolveBackF64;
class SyclTransposeSvdAF32;
class SyclTransposeSvdAF64;
class SyclTransposeSvdUF32;
class SyclTransposeSvdUF64;
class SyclTransposeSvdVtF32;
class SyclTransposeSvdVtF64;
class SyclTransposeQrAF32;
class SyclTransposeQrAF64;
class SyclTransposeQrRF32;
class SyclTransposeQrRF64;
class SyclTransposeQrQF32;
class SyclTransposeQrQF64;
class SyclTransposeEighF32;
class SyclTransposeEighF64;
class SyclTransposeEighBackF32;
class SyclTransposeEighBackF64;
class SyclTransposeCholeskyF32;
class SyclTransposeCholeskyF64;
class SyclTransposeCholeskyBackF32;
class SyclTransposeCholeskyBackF64;
class SyclTransposeEigF32;
class SyclTransposeEigF64;
class SyclTransposeEigBackF32;
class SyclTransposeEigBackF64;
class SyclDiagExtractDetF32;
class SyclDiagExtractDetF64;
class SyclDetReduceF32;
class SyclDetReduceF64;
class SyclDetCombineF32;
class SyclDetCombineF64;

auto linalg_det_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];

    // Copy input for in-place LU
    Tensor a = clone_kernel(input, queue);
    Tensor output({1}, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        // Row-major to column-major on device
        row_to_col_major<float, SyclTransposeDetF32>(
            d_a.get(), get_data_ptr<const float>(a), n, n, queue);

        auto scratchpad_size = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        SyclDeviceBuffer<float> scratchpad(scratchpad_size, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratchpad.get(), scratchpad_size).wait();

        // Extract diagonal on device (avoids O(n^2) memcpy)
        SyclDeviceBuffer<float> d_diag(n, queue);
        auto* d_diag_ptr = d_diag.get();
        auto* d_a_ptr = d_a.get();
        queue.parallel_for<SyclDiagExtractDetF32>(sycl::range<1>(n), [=](sycl::id<1> i) {
            d_diag_ptr[i] = d_a_ptr[i * n + i];  // column-major diagonal
        }).wait();

        // Device-side reduction: product of diagonal * (-1)^swaps
        auto* prod_buf = sycl::malloc_shared<float>(1, queue);
        auto* swap_buf = sycl::malloc_shared<int32_t>(1, queue);
        prod_buf[0] = 1.0f;
        swap_buf[0] = 0;

        auto* ipiv_ptr = d_ipiv.get();
        queue.parallel_for<SyclDetReduceF32>(
            sycl::range<1>(n),
            sycl::reduction(prod_buf, sycl::multiplies<float>()),
            sycl::reduction(swap_buf, sycl::plus<int32_t>()),
            [=](sycl::id<1> i, auto& prod, auto& swaps) {
                prod *= d_diag_ptr[i];
                swaps += (ipiv_ptr[i] != static_cast<std::int64_t>(i[0]) + 1) ? 1 : 0;
            }).wait();

        auto* out_ptr = static_cast<float*>(const_cast<void*>(output.data_ptr()));
        queue.single_task<SyclDetCombineF32>([=]() {
            out_ptr[0] = prod_buf[0] * ((swap_buf[0] % 2) ? -1.0f : 1.0f);
        }).wait();

        sycl::free(prod_buf, queue);
        sycl::free(swap_buf, queue);
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<double, SyclTransposeDetF64>(
            d_a.get(), get_data_ptr<const double>(a), n, n, queue);

        auto scratchpad_size = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        SyclDeviceBuffer<double> scratchpad(scratchpad_size, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratchpad.get(), scratchpad_size).wait();

        // Extract diagonal on device (avoids O(n^2) memcpy)
        SyclDeviceBuffer<double> d_diag(n, queue);
        auto* d_diag_ptr = d_diag.get();
        auto* d_a_ptr = d_a.get();
        queue.parallel_for<SyclDiagExtractDetF64>(sycl::range<1>(n), [=](sycl::id<1> i) {
            d_diag_ptr[i] = d_a_ptr[i * n + i];  // column-major diagonal
        }).wait();

        // Device-side reduction: product of diagonal * (-1)^swaps
        auto* prod_buf = sycl::malloc_shared<double>(1, queue);
        auto* swap_buf = sycl::malloc_shared<int32_t>(1, queue);
        prod_buf[0] = 1.0;
        swap_buf[0] = 0;

        auto* ipiv_ptr = d_ipiv.get();
        queue.parallel_for<SyclDetReduceF64>(
            sycl::range<1>(n),
            sycl::reduction(prod_buf, sycl::multiplies<double>()),
            sycl::reduction(swap_buf, sycl::plus<int32_t>()),
            [=](sycl::id<1> i, auto& prod, auto& swaps) {
                prod *= d_diag_ptr[i];
                swaps += (ipiv_ptr[i] != static_cast<std::int64_t>(i[0]) + 1) ? 1 : 0;
            }).wait();

        auto* out_ptr = static_cast<double*>(const_cast<void*>(output.data_ptr()));
        queue.single_task<SyclDetCombineF64>([=]() {
            out_ptr[0] = prod_buf[0] * ((swap_buf[0] % 2) ? -1.0 : 1.0);
        }).wait();

        sycl::free(prod_buf, queue);
        sycl::free(swap_buf, queue);
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
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<float, SyclTransposeInvF32>(
            d_a.get(), get_data_ptr<const float>(input), n, n, queue);

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        SyclDeviceBuffer<float> scratch_rf(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratch_rf.get(), sp_rf).wait();

        auto sp_ri = ::oneapi::mkl::lapack::getri_scratchpad_size<float>(queue, n, n);
        SyclDeviceBuffer<float> scratch_ri(sp_ri, queue);
        ::oneapi::mkl::lapack::getri(queue, n, d_a.get(), n, d_ipiv.get(), scratch_ri.get(), sp_ri).wait();

        Tensor output({n, n}, input.dtype(), input.device());
        col_to_row_major<float, SyclTransposeInvBackF32>(
            get_data_ptr<float>(output), d_a.get(), n, n, queue);

        return output;
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<double, SyclTransposeInvF64>(
            d_a.get(), get_data_ptr<const double>(input), n, n, queue);

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        SyclDeviceBuffer<double> scratch_rf(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratch_rf.get(), sp_rf).wait();

        auto sp_ri = ::oneapi::mkl::lapack::getri_scratchpad_size<double>(queue, n, n);
        SyclDeviceBuffer<double> scratch_ri(sp_ri, queue);
        ::oneapi::mkl::lapack::getri(queue, n, d_a.get(), n, d_ipiv.get(), scratch_ri.get(), sp_ri).wait();

        Tensor output({n, n}, input.dtype(), input.device());
        col_to_row_major<double, SyclTransposeInvBackF64>(
            get_data_ptr<double>(output), d_a.get(), n, n, queue);

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
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<float> d_b(n * nrhs, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<float, SyclTransposeSolveAF32>(
            d_a.get(), get_data_ptr<const float>(A), n, n, queue);
        row_to_col_major<float, SyclTransposeSolveBF32>(
            d_b.get(), get_data_ptr<const float>(B), n, nrhs, queue);

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        SyclDeviceBuffer<float> scratch_rf(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratch_rf.get(), sp_rf).wait();

        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<float>(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        SyclDeviceBuffer<float> scratch_rs(sp_rs, queue);
        ::oneapi::mkl::lapack::getrs(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, d_a.get(), n, d_ipiv.get(), d_b.get(), n, scratch_rs.get(), sp_rs).wait();

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A.dtype(), A.device());
        col_to_row_major<float, SyclTransposeSolveBackF32>(
            get_data_ptr<float>(output), d_b.get(), n, nrhs, queue);

        return output;
    } else if (A.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<double> d_b(n * nrhs, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<double, SyclTransposeSolveAF64>(
            d_a.get(), get_data_ptr<const double>(A), n, n, queue);
        row_to_col_major<double, SyclTransposeSolveBF64>(
            d_b.get(), get_data_ptr<const double>(B), n, nrhs, queue);

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        SyclDeviceBuffer<double> scratch_rf(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratch_rf.get(), sp_rf).wait();

        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<double>(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        SyclDeviceBuffer<double> scratch_rs(sp_rs, queue);
        ::oneapi::mkl::lapack::getrs(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, d_a.get(), n, d_ipiv.get(), d_b.get(), n, scratch_rs.get(), sp_rs).wait();

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A.dtype(), A.device());
        col_to_row_major<double, SyclTransposeSolveBackF64>(
            get_data_ptr<double>(output), d_b.get(), n, nrhs, queue);

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
        SyclDeviceBuffer<float> d_a(m * n, queue);
        SyclDeviceBuffer<float> d_s(k, queue);
        SyclDeviceBuffer<float> d_u(m * u_cols, queue);
        SyclDeviceBuffer<float> d_vt(vt_rows * n, queue);

        row_to_col_major<float, SyclTransposeSvdAF32>(
            d_a.get(), get_data_ptr<const float>(input), m, n, queue);

        auto sp = ::oneapi::mkl::lapack::gesvd_scratchpad_size<float>(queue, jobz, jobz, m, n, m, m, n);
        SyclDeviceBuffer<float> scratch(sp, queue);
        ::oneapi::mkl::lapack::gesvd(queue, jobz, jobz, m, n, d_a.get(), m, d_s.get(), d_u.get(), m, d_vt.get(), n, scratch.get(), sp).wait();

        Tensor S({k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(S.data_ptr()), d_s.get(), k * sizeof(float)).wait();

        // U: column-major (m x u_cols) -> row-major on device
        Tensor U({m, u_cols}, input.dtype(), input.device());
        col_to_row_major<float, SyclTransposeSvdUF32>(
            get_data_ptr<float>(U), d_u.get(), m, u_cols, queue);

        // Vt: column-major (vt_rows x n) -> row-major on device
        Tensor Vt({vt_rows, n}, input.dtype(), input.device());
        col_to_row_major<float, SyclTransposeSvdVtF32>(
            get_data_ptr<float>(Vt), d_vt.get(), vt_rows, n, queue);

        return {U, S, Vt};
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(m * n, queue);
        SyclDeviceBuffer<double> d_s(k, queue);
        SyclDeviceBuffer<double> d_u(m * u_cols, queue);
        SyclDeviceBuffer<double> d_vt(vt_rows * n, queue);

        row_to_col_major<double, SyclTransposeSvdAF64>(
            d_a.get(), get_data_ptr<const double>(input), m, n, queue);

        auto sp = ::oneapi::mkl::lapack::gesvd_scratchpad_size<double>(queue, jobz, jobz, m, n, m, m, n);
        SyclDeviceBuffer<double> scratch(sp, queue);
        ::oneapi::mkl::lapack::gesvd(queue, jobz, jobz, m, n, d_a.get(), m, d_s.get(), d_u.get(), m, d_vt.get(), n, scratch.get(), sp).wait();

        Tensor S({k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(S.data_ptr()), d_s.get(), k * sizeof(double)).wait();

        Tensor U({m, u_cols}, input.dtype(), input.device());
        col_to_row_major<double, SyclTransposeSvdUF64>(
            get_data_ptr<double>(U), d_u.get(), m, u_cols, queue);

        Tensor Vt({vt_rows, n}, input.dtype(), input.device());
        col_to_row_major<double, SyclTransposeSvdVtF64>(
            get_data_ptr<double>(Vt), d_vt.get(), vt_rows, n, queue);

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
        SyclDeviceBuffer<float> d_a(m * n, queue);
        SyclDeviceBuffer<float> d_tau(k, queue);

        row_to_col_major<float, SyclTransposeQrAF32>(
            d_a.get(), get_data_ptr<const float>(input), m, n, queue);

        auto sp_qr = ::oneapi::mkl::lapack::geqrf_scratchpad_size<float>(queue, m, n, m);
        SyclDeviceBuffer<float> scratch_qr(sp_qr, queue);
        ::oneapi::mkl::lapack::geqrf(queue, m, n, d_a.get(), m, d_tau.get(), scratch_qr.get(), sp_qr).wait();

        // Extract R: upper triangular from d_a (column-major) -> row-major on device
        // R[i][j] = d_a[j*m + i] for j >= i, else 0
        Tensor R({k, n}, input.dtype(), input.device());
        float* r_ptr = get_data_ptr<float>(R);
        auto* d_a_ptr = d_a.get();
        queue.memset(r_ptr, 0, k * n * sizeof(float)).wait();
        queue.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> id) {
            int64_t i = id[0], j = id[1];
            if (j >= i) {
                r_ptr[i * n + j] = d_a_ptr[j * m + i]; // col-major read
            }
        }).wait();

        // Generate Q via orgqr
        auto sp_oq = ::oneapi::mkl::lapack::orgqr_scratchpad_size<float>(queue, m, k, k, m);
        SyclDeviceBuffer<float> scratch_oq(sp_oq, queue);
        ::oneapi::mkl::lapack::orgqr(queue, m, k, k, d_a.get(), m, d_tau.get(), scratch_oq.get(), sp_oq).wait();

        Tensor Q({m, k}, input.dtype(), input.device());
        col_to_row_major<float, SyclTransposeQrQF32>(
            get_data_ptr<float>(Q), d_a.get(), m, k, queue);

        return {Q, R};
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(m * n, queue);
        SyclDeviceBuffer<double> d_tau(k, queue);

        row_to_col_major<double, SyclTransposeQrAF64>(
            d_a.get(), get_data_ptr<const double>(input), m, n, queue);

        auto sp_qr = ::oneapi::mkl::lapack::geqrf_scratchpad_size<double>(queue, m, n, m);
        SyclDeviceBuffer<double> scratch_qr(sp_qr, queue);
        ::oneapi::mkl::lapack::geqrf(queue, m, n, d_a.get(), m, d_tau.get(), scratch_qr.get(), sp_qr).wait();

        Tensor R({k, n}, input.dtype(), input.device());
        double* r_ptr = get_data_ptr<double>(R);
        auto* d_a_ptr = d_a.get();
        queue.memset(r_ptr, 0, k * n * sizeof(double)).wait();
        queue.parallel_for(sycl::range<2>(k, n), [=](sycl::id<2> id) {
            int64_t i = id[0], j = id[1];
            if (j >= i) {
                r_ptr[i * n + j] = d_a_ptr[j * m + i];
            }
        }).wait();

        auto sp_oq = ::oneapi::mkl::lapack::orgqr_scratchpad_size<double>(queue, m, k, k, m);
        SyclDeviceBuffer<double> scratch_oq(sp_oq, queue);
        ::oneapi::mkl::lapack::orgqr(queue, m, k, k, d_a.get(), m, d_tau.get(), scratch_oq.get(), sp_oq).wait();

        Tensor Q({m, k}, input.dtype(), input.device());
        col_to_row_major<double, SyclTransposeQrQF64>(
            get_data_ptr<double>(Q), d_a.get(), m, k, queue);

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
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<float> d_w(n, queue);

        row_to_col_major<float, SyclTransposeEighF32>(
            d_a.get(), get_data_ptr<const float>(input), n, n, queue);

        auto sp = ::oneapi::mkl::lapack::syevd_scratchpad_size<float>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower, n, n);
        SyclDeviceBuffer<float> scratch(sp, queue);
        ::oneapi::mkl::lapack::syevd(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower,
                                   n, d_a.get(), n, d_w.get(), scratch.get(), sp).wait();

        Tensor W({n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(W.data_ptr()), d_w.get(), n * sizeof(float)).wait();

        Tensor V({n, n}, input.dtype(), input.device());
        col_to_row_major<float, SyclTransposeEighBackF32>(
            get_data_ptr<float>(V), d_a.get(), n, n, queue);

        return {W, V};
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<double> d_w(n, queue);

        row_to_col_major<double, SyclTransposeEighF64>(
            d_a.get(), get_data_ptr<const double>(input), n, n, queue);

        auto sp = ::oneapi::mkl::lapack::syevd_scratchpad_size<double>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower, n, n);
        SyclDeviceBuffer<double> scratch(sp, queue);
        ::oneapi::mkl::lapack::syevd(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower,
                                   n, d_a.get(), n, d_w.get(), scratch.get(), sp).wait();

        Tensor W({n}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(W.data_ptr()), d_w.get(), n * sizeof(double)).wait();

        Tensor V({n, n}, input.dtype(), input.device());
        col_to_row_major<double, SyclTransposeEighBackF64>(
            get_data_ptr<double>(V), d_a.get(), n, n, queue);

        return {W, V};
    } else {
        throw std::runtime_error("linalg_eigh: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgEig - Non-symmetric eigendecomposition via geev
// ============================================================================
#ifdef TENZOR_HAS_ONEMKL_GEEV
auto linalg_eig_kernel(const Tensor& input, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];

    // Compute batch dimensions (all dims except last two)
    int64_t nbatch = 1;
    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); i++) {
        batch_dims.push_back(shape[i]);
        nbatch *= shape[i];
    }
    if (nbatch == 0) nbatch = 1;

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    std::vector<int64_t> v_shape = batch_dims;
    v_shape.push_back(n);
    v_shape.push_back(n);

    if (n == 0) {
        return {Tensor(w_shape, input.dtype(), input.device()),
                Tensor(w_shape, input.dtype(), input.device()),
                Tensor(v_shape, input.dtype(), input.device())};
    }

    // Row-major input: feeding to column-major geev gives A^T.
    // Left eigenvectors of A^T = right eigenvectors of A.
    // Output VL in column-major = right eigenvectors in row-major.
    auto work = clone_kernel(input, queue);

    auto WR = Tensor(w_shape, input.dtype(), input.device());
    auto WI = Tensor(w_shape, input.dtype(), input.device());
    auto V = Tensor(v_shape, input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        float* a_data = get_data_ptr<float>(work);
        float* wr_data = get_data_ptr<float>(WR);
        float* wi_data = get_data_ptr<float>(WI);
        float* v_data = get_data_ptr<float>(V);

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = a_data + b * n * n;
            float* wr_vec = wr_data + b * n;
            float* wi_vec = wi_data + b * n;
            float* vl = v_data + b * n * n;

            auto sp = ::oneapi::mkl::lapack::geev_scratchpad_size<float>(
                queue, ::oneapi::mkl::compvl::vec, ::oneapi::mkl::compvr::novec,
                n, n);
            SyclDeviceBuffer<float> scratch(sp, queue);

            ::oneapi::mkl::lapack::geev(
                queue, ::oneapi::mkl::compvl::vec, ::oneapi::mkl::compvr::novec,
                n, mat, n, wr_vec, wi_vec,
                vl, n,       // VL (left eigenvectors of A^T = right of A)
                nullptr, n,  // VR (not computed)
                scratch.get(), sp).wait();
        }
    } else if (input.dtype() == DType::Float64) {
        double* a_data = get_data_ptr<double>(work);
        double* wr_data = get_data_ptr<double>(WR);
        double* wi_data = get_data_ptr<double>(WI);
        double* v_data = get_data_ptr<double>(V);

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = a_data + b * n * n;
            double* wr_vec = wr_data + b * n;
            double* wi_vec = wi_data + b * n;
            double* vl = v_data + b * n * n;

            auto sp = ::oneapi::mkl::lapack::geev_scratchpad_size<double>(
                queue, ::oneapi::mkl::compvl::vec, ::oneapi::mkl::compvr::novec,
                n, n);
            SyclDeviceBuffer<double> scratch(sp, queue);

            ::oneapi::mkl::lapack::geev(
                queue, ::oneapi::mkl::compvl::vec, ::oneapi::mkl::compvr::novec,
                n, mat, n, wr_vec, wi_vec,
                vl, n,
                nullptr, n,
                scratch.get(), sp).wait();
        }
    } else {
        throw std::runtime_error("linalg_eig: only Float32 and Float64 supported");
    }

    // V contains left eigenvectors of A^T (= right eigenvectors of A) in column-major
    // which is the same as right eigenvectors in row-major — no transpose needed
    return {WR, WI, V};
}
#else
auto linalg_eig_kernel(const Tensor& input, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    (void)input; (void)queue;
    throw std::runtime_error(
        "linalg_eig: oneapi::mkl::lapack::geev is not available in the installed oneMKL version. "
        "Non-symmetric eigendecomposition requires a newer oneMKL or the CPU backend.");
}
#endif // TENZOR_HAS_ONEMKL_GEEV

// ============================================================================
// LinalgCholesky - Cholesky factorization via potrf
// ============================================================================
auto linalg_cholesky_kernel(const Tensor& input, bool upper, sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];
    auto uplo = upper ? ::oneapi::mkl::uplo::upper : ::oneapi::mkl::uplo::lower;

    if (input.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_a(n * n, queue);

        row_to_col_major<float, SyclTransposeCholeskyF32>(
            d_a.get(), get_data_ptr<const float>(input), n, n, queue);

        auto sp = ::oneapi::mkl::lapack::potrf_scratchpad_size<float>(queue, uplo, n, n);
        SyclDeviceBuffer<float> scratch(sp, queue);
        ::oneapi::mkl::lapack::potrf(queue, uplo, n, d_a.get(), n, scratch.get(), sp).wait();

        // Transpose back to row-major and zero the other triangle on device
        Tensor output({n, n}, input.dtype(), input.device());
        float* out_ptr = get_data_ptr<float>(output);
        col_to_row_major<float, SyclTransposeCholeskyBackF32>(out_ptr, d_a.get(), n, n, queue);

        // Zero out the appropriate triangle on device
        bool is_upper = upper;
        queue.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> id) {
            int64_t i = id[0], j = id[1];
            if (is_upper) {
                if (j < i) out_ptr[i * n + j] = 0.0f;
            } else {
                if (j > i) out_ptr[i * n + j] = 0.0f;
            }
        }).wait();

        return output;
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);

        row_to_col_major<double, SyclTransposeCholeskyF64>(
            d_a.get(), get_data_ptr<const double>(input), n, n, queue);

        auto sp = ::oneapi::mkl::lapack::potrf_scratchpad_size<double>(queue, uplo, n, n);
        SyclDeviceBuffer<double> scratch(sp, queue);
        ::oneapi::mkl::lapack::potrf(queue, uplo, n, d_a.get(), n, scratch.get(), sp).wait();

        Tensor output({n, n}, input.dtype(), input.device());
        double* out_ptr = get_data_ptr<double>(output);
        col_to_row_major<double, SyclTransposeCholeskyBackF64>(out_ptr, d_a.get(), n, n, queue);

        bool is_upper = upper;
        queue.parallel_for(sycl::range<2>(n, n), [=](sycl::id<2> id) {
            int64_t i = id[0], j = id[1];
            if (is_upper) {
                if (j < i) out_ptr[i * n + j] = 0.0;
            } else {
                if (j > i) out_ptr[i * n + j] = 0.0;
            }
        }).wait();

        return output;
    } else {
        throw std::runtime_error("linalg_cholesky: only Float32 and Float64 supported");
    }
}

#endif // TENZOR_HAS_ONEMKL

} // namespace oneapi
} // namespace tenzor
