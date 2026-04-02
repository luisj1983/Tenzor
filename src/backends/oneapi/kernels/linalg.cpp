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

#else // !TENZOR_HAS_ONEMKL — native SYCL shared-memory linalg fallback kernels

// ============================================================================
// Constants and utilities (mirrored from CUDA/ROCm native fallback)
// ============================================================================

namespace {

/// Max matrix dimension for local-memory fallback kernels.
/// Local memory usage is ~2*N*N*sizeof(T) + scratch, capped at 48 KB.
constexpr int MAX_N_FLOAT  = 90;
constexpr int MAX_N_DOUBLE = 64;

/// Convert span to vector.
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

/// Get batch count from shape (product of all dims except last two).
int64_t batch_size(const Tensor& t) {
    auto shape = t.shape();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch *= shape[i];
    return batch;
}

/// Get square matrix size and validate.
std::pair<int64_t, int64_t> check_square(const Tensor& t) {
    auto shape = t.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    if (m != n) throw std::invalid_argument("linalg: expected square matrix");
    return {m, ndim};
}

/// Validate dtype for linalg ops.
void validate_linalg_dtype(const Tensor& t, const std::string& op_name) {
    auto dt = t.dtype();
    if (dt != DType::Float32 && dt != DType::Float64 &&
        dt != DType::Float16 && dt != DType::BFloat16) {
        throw std::invalid_argument(
            "linalg::" + op_name + ": unsupported dtype " +
            std::string(dtype_name(dt)) +
            ". Supported: Float32, Float64 (Float16/BFloat16 auto-upcast to Float32).");
    }
}

/// Check matrix size limit for local-memory fallback.
template<typename T>
void check_size_limit(int64_t n, const std::string& op_name) {
    constexpr int max_n = std::is_same_v<T, float> ? MAX_N_FLOAT : MAX_N_DOUBLE;
    if (n > max_n) {
        throw std::runtime_error(
            "linalg::" + op_name + ": matrix size " + std::to_string(n) +
            " exceeds native SYCL fallback limit of " + std::to_string(max_n) +
            " (build with oneMKL for larger matrices)");
    }
}

} // anonymous namespace

// ============================================================================
// Determinant
// ============================================================================

auto linalg_det_kernel(const Tensor& A, sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(A, "det");
    if (A.dtype() == DType::Float16)
        return linalg_det_kernel(A.to(DType::Float32), queue).to(DType::Float16);
    if (A.dtype() == DType::BFloat16)
        return linalg_det_kernel(A.to(DType::Float32), queue).to(DType::BFloat16);

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) out_shape.push_back(shape[i]);
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = zeros(out_shape, A.dtype(), A.device());

    SyclDeviceBuffer<int> d_pivots(nbatch * n, queue);
    SyclDeviceBuffer<int> d_info(nbatch, queue);
    queue.memset(d_info.ptr, 0, nbatch * sizeof(int)).wait();

    auto launch_det = [&](auto* work_ptr, auto* res_ptr, auto dummy) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        check_size_limit<T>(n, "det");
        size_t smem_lu = n * n * sizeof(T) + 4 * sizeof(T);
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;

        // LU kernel: one work-group per batch element
        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_lu), h);
            auto* data = work_ptr;
            auto* pivots = d_pivots.ptr;
            auto* info = d_info.ptr;
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);

                    char* smem_raw = smem.get_pointer();
                    T* A = reinterpret_cast<T*>(smem_raw);
                    T* scratch = A + n_ * n_;

                    T* batch_data = data + batch_idx * n_ * n_;
                    int* batch_pivots = pivots + batch_idx * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        A[idx] = batch_data[idx];
                    sycl::group_barrier(item.get_group());

                    int sign = 1;
                    for (int k = 0; k < n_; k++) {
                        if (tid == 0) {
                            T max_val = sycl::fabs(A[k * n_ + k]);
                            int max_row = k;
                            for (int i = k + 1; i < n_; i++) {
                                T val = sycl::fabs(A[i * n_ + k]);
                                if (val > max_val) { max_val = val; max_row = i; }
                            }
                            batch_pivots[k] = max_row + 1;
                            scratch[0] = static_cast<T>(max_row);
                            if (max_val == T(0) && info)
                                info[batch_idx] = k + 1;
                        }
                        sycl::group_barrier(item.get_group());

                        int pivot_row = static_cast<int>(scratch[0]);
                        if (pivot_row != k) {
                            for (int j = tid; j < n_; j += num_threads) {
                                T tmp = A[k * n_ + j];
                                A[k * n_ + j] = A[pivot_row * n_ + j];
                                A[pivot_row * n_ + j] = tmp;
                            }
                            if (tid == 0) sign = -sign;
                            sycl::group_barrier(item.get_group());
                        }

                        T diag = A[k * n_ + k];
                        if (diag != T(0)) {
                            if (tid == 0) {
                                for (int i = k + 1; i < n_; i++)
                                    A[i * n_ + k] /= diag;
                            }
                            sycl::group_barrier(item.get_group());
                            for (int i = k + 1 + tid; i < n_; i += num_threads) {
                                T mult = A[i * n_ + k];
                                for (int j = k + 1; j < n_; j++)
                                    A[i * n_ + j] -= mult * A[k * n_ + j];
                            }
                            sycl::group_barrier(item.get_group());
                        }
                    }

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        batch_data[idx] = A[idx];
                    if (tid == 0 && info) {
                        if (info[batch_idx] == 0)
                            info[batch_idx] = sign > 0 ? 0 : -1;
                    }
                });
        }).wait();

        // Det kernel: product of diagonal * pivot sign
        queue.submit([&](sycl::handler& h) {
            auto* lu_data = work_ptr;
            auto* pivots = d_pivots.ptr;
            auto* det_out = res_ptr;
            int n_ = static_cast<int>(n);
            int nb = static_cast<int>(nbatch);
            h.parallel_for(sycl::range<1>(nbatch), [=](sycl::id<1> id) {
                int b = id[0];
                const T* mat = lu_data + b * n_ * n_;
                const int* piv = pivots + b * n_;
                T d = T(1);
                for (int i = 0; i < n_; i++) {
                    d *= mat[i * n_ + i];
                    if (piv[i] != i + 1) d = -d;
                }
                det_out[b] = d;
            });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch_det(work.data<float>(), result.data<float>(), float{});
    else
        launch_det(work.data<double>(), result.data<double>(), double{});

    return result;
}

// ============================================================================
// Matrix Inverse
// ============================================================================

auto linalg_inv_kernel(const Tensor& A, sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(A, "inv");
    if (A.dtype() == DType::Float16)
        return linalg_inv_kernel(A.to(DType::Float32), queue);
    if (A.dtype() == DType::BFloat16)
        return linalg_inv_kernel(A.to(DType::Float32), queue);

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto result = zeros(to_vec(work.shape()), A.dtype(), A.device());

    SyclDeviceBuffer<int> d_pivots(nbatch * n, queue);
    SyclDeviceBuffer<int> d_info(nbatch, queue);
    queue.memset(d_info.ptr, 0, nbatch * sizeof(int)).wait();

    auto launch_inv = [&](auto* work_ptr, auto* res_ptr) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        check_size_limit<T>(n, "inv");
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;

        // LU factorize
        size_t smem_lu = n * n * sizeof(T) + 4 * sizeof(T);
        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_lu), h);
            auto* data = work_ptr;
            auto* pivots = d_pivots.ptr;
            auto* info = d_info.ptr;
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* A = reinterpret_cast<T*>(smem_raw);
                    T* scratch = A + n_ * n_;
                    T* batch_data = data + batch_idx * n_ * n_;
                    int* batch_pivots = pivots + batch_idx * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        A[idx] = batch_data[idx];
                    sycl::group_barrier(item.get_group());

                    for (int k = 0; k < n_; k++) {
                        if (tid == 0) {
                            T max_val = sycl::fabs(A[k * n_ + k]);
                            int max_row = k;
                            for (int i = k + 1; i < n_; i++) {
                                T val = sycl::fabs(A[i * n_ + k]);
                                if (val > max_val) { max_val = val; max_row = i; }
                            }
                            batch_pivots[k] = max_row + 1;
                            scratch[0] = static_cast<T>(max_row);
                        }
                        sycl::group_barrier(item.get_group());
                        int pivot_row = static_cast<int>(scratch[0]);
                        if (pivot_row != k) {
                            for (int j = tid; j < n_; j += num_threads) {
                                T tmp = A[k * n_ + j];
                                A[k * n_ + j] = A[pivot_row * n_ + j];
                                A[pivot_row * n_ + j] = tmp;
                            }
                            sycl::group_barrier(item.get_group());
                        }
                        T diag = A[k * n_ + k];
                        if (diag != T(0)) {
                            if (tid == 0)
                                for (int i = k + 1; i < n_; i++) A[i * n_ + k] /= diag;
                            sycl::group_barrier(item.get_group());
                            for (int i = k + 1 + tid; i < n_; i += num_threads) {
                                T mult = A[i * n_ + k];
                                for (int j = k + 1; j < n_; j++)
                                    A[i * n_ + j] -= mult * A[k * n_ + j];
                            }
                            sycl::group_barrier(item.get_group());
                        }
                    }
                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        batch_data[idx] = A[idx];
                });
        }).wait();

        // Invert via LU solve with identity
        size_t smem_inv = 2 * n * n * sizeof(T);
        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_inv), h);
            auto* lu_data = work_ptr;
            auto* pivots = d_pivots.ptr;
            auto* inv_out = res_ptr;
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* LU = reinterpret_cast<T*>(smem_raw);
                    T* X = LU + n_ * n_;
                    const T* batch_lu = lu_data + batch_idx * n_ * n_;
                    const int* batch_piv = pivots + batch_idx * n_;
                    T* batch_inv = inv_out + batch_idx * n_ * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        LU[idx] = batch_lu[idx];
                    for (int idx = tid; idx < n_ * n_; idx += num_threads) {
                        int row = idx / n_, col = idx % n_;
                        X[idx] = (row == col) ? T(1) : T(0);
                    }
                    sycl::group_barrier(item.get_group());

                    if (tid == 0) {
                        for (int i = 0; i < n_; i++) {
                            int piv_row = batch_piv[i] - 1;
                            if (piv_row != i)
                                for (int j = 0; j < n_; j++) {
                                    T tmp = X[i * n_ + j]; X[i * n_ + j] = X[piv_row * n_ + j]; X[piv_row * n_ + j] = tmp;
                                }
                        }
                        for (int k = 0; k < n_; k++)
                            for (int i = k + 1; i < n_; i++) {
                                T mult = LU[i * n_ + k];
                                for (int j = 0; j < n_; j++) X[i * n_ + j] -= mult * X[k * n_ + j];
                            }
                        for (int k = n_ - 1; k >= 0; k--) {
                            T diag = LU[k * n_ + k];
                            for (int j = 0; j < n_; j++) X[k * n_ + j] /= diag;
                            for (int i = 0; i < k; i++) {
                                T mult = LU[i * n_ + k];
                                for (int j = 0; j < n_; j++) X[i * n_ + j] -= mult * X[k * n_ + j];
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        batch_inv[idx] = X[idx];
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch_inv(work.data<float>(), result.data<float>());
    else
        launch_inv(work.data<double>(), result.data<double>());

    return result;
}

// ============================================================================
// Linear System Solve (AX = B)
// ============================================================================

auto linalg_solve_kernel(const Tensor& A, const Tensor& B, sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(A, "solve");
    if (A.dtype() == DType::Float16)
        return linalg_solve_kernel(A.to(DType::Float32), B.to(DType::Float32), queue);
    if (A.dtype() == DType::BFloat16)
        return linalg_solve_kernel(A.to(DType::Float32), B.to(DType::Float32), queue);

    auto work_a = A.contiguous().clone();
    auto work_b = B.contiguous().clone();
    auto [n, ndim_a] = check_square(work_a);
    int64_t nbatch = batch_size(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;

    SyclDeviceBuffer<int> d_pivots(nbatch * n, queue);
    SyclDeviceBuffer<int> d_info(nbatch, queue);
    queue.memset(d_info.ptr, 0, nbatch * sizeof(int)).wait();

    auto launch_solve = [&](auto* a_ptr, auto* b_ptr) {
        using T = std::remove_pointer_t<decltype(a_ptr)>;
        check_size_limit<T>(n, "solve");
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;

        // LU factorize
        size_t smem_lu = n * n * sizeof(T) + 4 * sizeof(T);
        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_lu), h);
            auto* data = a_ptr;
            auto* pivots = d_pivots.ptr;
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* A = reinterpret_cast<T*>(smem_raw);
                    T* scratch = A + n_ * n_;
                    T* batch_data = data + batch_idx * n_ * n_;
                    int* batch_pivots = pivots + batch_idx * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        A[idx] = batch_data[idx];
                    sycl::group_barrier(item.get_group());
                    for (int k = 0; k < n_; k++) {
                        if (tid == 0) {
                            T max_val = sycl::fabs(A[k * n_ + k]);
                            int max_row = k;
                            for (int i = k + 1; i < n_; i++) {
                                T val = sycl::fabs(A[i * n_ + k]);
                                if (val > max_val) { max_val = val; max_row = i; }
                            }
                            batch_pivots[k] = max_row + 1;
                            scratch[0] = static_cast<T>(max_row);
                        }
                        sycl::group_barrier(item.get_group());
                        int pivot_row = static_cast<int>(scratch[0]);
                        if (pivot_row != k) {
                            for (int j = tid; j < n_; j += num_threads) {
                                T tmp = A[k * n_ + j]; A[k * n_ + j] = A[pivot_row * n_ + j]; A[pivot_row * n_ + j] = tmp;
                            }
                            sycl::group_barrier(item.get_group());
                        }
                        T diag = A[k * n_ + k];
                        if (diag != T(0)) {
                            if (tid == 0) for (int i = k + 1; i < n_; i++) A[i * n_ + k] /= diag;
                            sycl::group_barrier(item.get_group());
                            for (int i = k + 1 + tid; i < n_; i += num_threads) {
                                T mult = A[i * n_ + k];
                                for (int j = k + 1; j < n_; j++) A[i * n_ + j] -= mult * A[k * n_ + j];
                            }
                            sycl::group_barrier(item.get_group());
                        }
                    }
                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        batch_data[idx] = A[idx];
                });
        }).wait();

        // Solve via forward/back substitution
        size_t smem_solve = (n * n + n * nrhs) * sizeof(T);
        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_solve), h);
            auto* lu_data = a_ptr;
            auto* pivots = d_pivots.ptr;
            auto* b_data = b_ptr;
            int n_ = static_cast<int>(n);
            int nrhs_ = static_cast<int>(nrhs);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* LU = reinterpret_cast<T*>(smem_raw);
                    T* B = LU + n_ * n_;
                    const T* batch_lu = lu_data + batch_idx * n_ * n_;
                    const int* batch_piv = pivots + batch_idx * n_;
                    T* batch_b = b_data + batch_idx * n_ * nrhs_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        LU[idx] = batch_lu[idx];
                    for (int idx = tid; idx < n_ * nrhs_; idx += num_threads)
                        B[idx] = batch_b[idx];
                    sycl::group_barrier(item.get_group());

                    if (tid == 0) {
                        for (int i = 0; i < n_; i++) {
                            int piv_row = batch_piv[i] - 1;
                            if (piv_row != i)
                                for (int j = 0; j < nrhs_; j++) {
                                    T tmp = B[i * nrhs_ + j]; B[i * nrhs_ + j] = B[piv_row * nrhs_ + j]; B[piv_row * nrhs_ + j] = tmp;
                                }
                        }
                        for (int k = 0; k < n_; k++)
                            for (int i = k + 1; i < n_; i++) {
                                T mult = LU[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++) B[i * nrhs_ + j] -= mult * B[k * nrhs_ + j];
                            }
                        for (int k = n_ - 1; k >= 0; k--) {
                            T diag = LU[k * n_ + k];
                            for (int j = 0; j < nrhs_; j++) B[k * nrhs_ + j] /= diag;
                            for (int i = 0; i < k; i++) {
                                T mult = LU[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++) B[i * nrhs_ + j] -= mult * B[k * nrhs_ + j];
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < n_ * nrhs_; idx += num_threads)
                        batch_b[idx] = B[idx];
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch_solve(work_a.data<float>(), work_b.data<float>());
    else
        launch_solve(work_a.data<double>(), work_b.data<double>());

    return work_b;
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

auto linalg_cholesky_kernel(const Tensor& A, bool upper, sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(A, "cholesky");
    if (A.dtype() == DType::Float16)
        return linalg_cholesky_kernel(A.to(DType::Float32), upper, queue);
    if (A.dtype() == DType::BFloat16)
        return linalg_cholesky_kernel(A.to(DType::Float32), upper, queue);

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto launch_cholesky = [&](auto* data_ptr) {
        using T = std::remove_pointer_t<decltype(data_ptr)>;
        check_size_limit<T>(n, "cholesky");
        size_t smem_bytes = n * n * sizeof(T);
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_bytes), h);
            auto* data = data_ptr;
            int n_ = static_cast<int>(n);
            bool upper_ = upper;
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* A = reinterpret_cast<T*>(smem_raw);
                    T* batch_data = data + batch_idx * n_ * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        A[idx] = batch_data[idx];
                    sycl::group_barrier(item.get_group());

                    if (tid == 0) {
                        for (int j = 0; j < n_; j++) {
                            T sum = A[j * n_ + j];
                            for (int k = 0; k < j; k++) sum -= A[j * n_ + k] * A[j * n_ + k];
                            if (sum <= T(0)) { A[j * n_ + j] = T(0); continue; }
                            A[j * n_ + j] = sycl::sqrt(sum);
                            T diag = A[j * n_ + j];
                            for (int i = j + 1; i < n_; i++) {
                                T s = A[i * n_ + j];
                                for (int k = 0; k < j; k++) s -= A[i * n_ + k] * A[j * n_ + k];
                                A[i * n_ + j] = s / diag;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < n_ * n_; idx += num_threads) {
                        int row = idx / n_, col = idx % n_;
                        if (upper_) {
                            batch_data[row * n_ + col] = (row <= col) ? A[col * n_ + row] : T(0);
                        } else {
                            batch_data[row * n_ + col] = (row >= col) ? A[row * n_ + col] : T(0);
                        }
                    }
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch_cholesky(work.data<float>());
    else
        launch_cholesky(work.data<double>());

    return work;
}

// ============================================================================
// QR Decomposition
// ============================================================================

auto linalg_qr_kernel(const Tensor& A, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    validate_linalg_dtype(A, "qr");
    if (A.dtype() == DType::Float16) {
        auto [Q, R] = linalg_qr_kernel(A.to(DType::Float32), queue);
        return {Q, R};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [Q, R] = linalg_qr_kernel(A.to(DType::Float32), queue);
        return {Q, R};
    }

    auto work = A.contiguous().clone();
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::qr: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> q_shape = batch_dims;
    q_shape.push_back(m); q_shape.push_back(k);
    std::vector<int64_t> r_shape = batch_dims;
    r_shape.push_back(k); r_shape.push_back(n_cols);

    auto Q = zeros(q_shape, A.dtype(), A.device());
    auto R = zeros(r_shape, A.dtype(), A.device());

    auto launch_qr = [&](auto* work_ptr, auto* q_ptr, auto* r_ptr) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        check_size_limit<T>(std::max(m, n_cols), "qr");
        size_t smem_bytes = (m * n_cols + m * m + 4) * sizeof(T);
        int threads = std::min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_bytes), h);
            int m_ = static_cast<int>(m), nc_ = static_cast<int>(n_cols), k_ = static_cast<int>(k);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* R_s = reinterpret_cast<T*>(smem_raw);
                    T* Q_s = R_s + m_ * nc_;
                    T* scratch = Q_s + m_ * m_;

                    const T* A = work_ptr + batch_idx * m_ * nc_;
                    T* Q_batch = q_ptr + batch_idx * m_ * k_;
                    T* R_batch = r_ptr + batch_idx * k_ * nc_;

                    for (int idx = tid; idx < m_ * nc_; idx += num_threads) R_s[idx] = A[idx];
                    for (int idx = tid; idx < m_ * m_; idx += num_threads) {
                        int row = idx / m_, col = idx % m_;
                        Q_s[idx] = (row == col) ? T(1) : T(0);
                    }
                    sycl::group_barrier(item.get_group());

                    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

                    for (int j = 0; j < k_; j++) {
                        if (tid == 0) {
                            T sigma = T(0);
                            for (int i = j + 1; i < m_; i++) sigma += R_s[i * nc_ + j] * R_s[i * nc_ + j];
                            T x0 = R_s[j * nc_ + j];
                            T norm_x = sycl::sqrt(x0 * x0 + sigma);
                            if (norm_x < zero_tol || sigma < zero_tol) {
                                scratch[1] = T(0);
                            } else {
                                T alpha = -sycl::copysign(norm_x, x0);
                                T v0 = x0 - alpha;
                                T v_norm_sq = v0 * v0 + sigma;
                                scratch[0] = v0;
                                scratch[1] = T(2) / v_norm_sq;
                                scratch[2] = alpha;
                            }
                        }
                        sycl::group_barrier(item.get_group());

                        T tau = scratch[1];
                        if (tau == T(0)) { sycl::group_barrier(item.get_group()); continue; }
                        T v0 = scratch[0];
                        T alpha = scratch[2];

                        for (int col = j + tid; col < nc_; col += num_threads) {
                            T dot = v0 * R_s[j * nc_ + col];
                            for (int i = j + 1; i < m_; i++) dot += R_s[i * nc_ + j] * R_s[i * nc_ + col];
                            dot *= tau;
                            R_s[j * nc_ + col] -= v0 * dot;
                            for (int i = j + 1; i < m_; i++) R_s[i * nc_ + col] -= R_s[i * nc_ + j] * dot;
                        }
                        sycl::group_barrier(item.get_group());

                        for (int row = tid; row < m_; row += num_threads) {
                            T dot = v0 * Q_s[row * m_ + j];
                            for (int i = j + 1; i < m_; i++) dot += R_s[i * nc_ + j] * Q_s[row * m_ + i];
                            dot *= tau;
                            Q_s[row * m_ + j] -= v0 * dot;
                            for (int i = j + 1; i < m_; i++) Q_s[row * m_ + i] -= R_s[i * nc_ + j] * dot;
                        }
                        sycl::group_barrier(item.get_group());

                        if (tid == 0) {
                            R_s[j * nc_ + j] = alpha;
                            for (int i = j + 1; i < m_; i++) R_s[i * nc_ + j] = T(0);
                        }
                        sycl::group_barrier(item.get_group());
                    }

                    for (int idx = tid; idx < m_ * k_; idx += num_threads) {
                        int row = idx / k_, col = idx % k_;
                        Q_batch[idx] = Q_s[row * m_ + col];
                    }
                    for (int idx = tid; idx < k_ * nc_; idx += num_threads)
                        R_batch[idx] = R_s[idx];
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch_qr(work.data<float>(), Q.data<float>(), R.data<float>());
    else
        launch_qr(work.data<double>(), Q.data<double>(), R.data<double>());

    return {Q, R};
}

// ============================================================================
// Symmetric Eigendecomposition (eigh)
// ============================================================================

auto linalg_eigh_kernel(const Tensor& A, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    validate_linalg_dtype(A, "eigh");
    if (A.dtype() == DType::Float16) { auto [W,V] = linalg_eigh_kernel(A.to(DType::Float32), queue); return {W,V}; }
    if (A.dtype() == DType::BFloat16) { auto [W,V] = linalg_eigh_kernel(A.to(DType::Float32), queue); return {W,V}; }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, A.dtype(), A.device());

    auto launch_eigh = [&](auto* data_ptr, auto* w_ptr) {
        using T = std::remove_pointer_t<decltype(data_ptr)>;
        check_size_limit<T>(n, "eigh");
        size_t smem_bytes = (2 * n * n + 2 * n + 4) * sizeof(T);
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        int max_iter = 30 * static_cast<int>(n);

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_bytes), h);
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
                    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* A = reinterpret_cast<T*>(smem_raw);
                    T* Q = A + n_ * n_;
                    T* d = Q + n_ * n_;
                    T* e = d + n_;
                    T* scratch = e + n_;

                    T* batch_data = data_ptr + batch_idx * n_ * n_;
                    T* batch_eig = w_ptr + batch_idx * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads) {
                        A[idx] = batch_data[idx];
                        int row = idx / n_, col = idx % n_;
                        Q[idx] = (row == col) ? T(1) : T(0);
                    }
                    sycl::group_barrier(item.get_group());

                    // Tridiagonalize via Householder reflections
                    for (int k = 0; k + 2 < n_; k++) {
                        if (tid == 0) {
                            T sigma = T(0);
                            for (int i = k + 1; i < n_; i++) sigma += A[i * n_ + k] * A[i * n_ + k];
                            T norm_x = sycl::sqrt(sigma);
                            if (norm_x < zero_tol) { scratch[1] = T(0); }
                            else {
                                T a = -sycl::copysign(norm_x, A[(k+1)*n_+k]);
                                T v0_val = A[(k+1)*n_+k] - a;
                                T v_norm_sq = v0_val * v0_val;
                                for (int i = k+2; i < n_; i++) v_norm_sq += A[i*n_+k]*A[i*n_+k];
                                if (v_norm_sq < zero_tol) scratch[1] = T(0);
                                else { scratch[0]=v0_val; scratch[1]=T(2)/v_norm_sq; scratch[2]=a; }
                            }
                        }
                        sycl::group_barrier(item.get_group());
                        T tau = scratch[1];
                        if (tau == T(0)) { sycl::group_barrier(item.get_group()); continue; }
                        T v0 = scratch[0]; T alpha_val = scratch[2];

                        for (int j = k + tid; j < n_; j += num_threads) {
                            T dot = v0 * A[(k+1)*n_+j];
                            for (int i = k+2; i < n_; i++) dot += A[i*n_+k] * A[i*n_+j];
                            dot *= tau;
                            A[(k+1)*n_+j] -= v0 * dot;
                            for (int i = k+2; i < n_; i++) A[i*n_+j] -= A[i*n_+k] * dot;
                        }
                        sycl::group_barrier(item.get_group());
                        for (int i = tid; i < n_; i += num_threads) {
                            T dot = v0 * A[i*n_+(k+1)];
                            for (int j = k+2; j < n_; j++) dot += A[j*n_+k] * A[i*n_+j];
                            dot *= tau;
                            A[i*n_+(k+1)] -= v0 * dot;
                            for (int j = k+2; j < n_; j++) A[i*n_+j] -= A[j*n_+k] * dot;
                        }
                        sycl::group_barrier(item.get_group());
                        for (int i = tid; i < n_; i += num_threads) {
                            T dot = v0 * Q[i*n_+(k+1)];
                            for (int j = k+2; j < n_; j++) dot += A[j*n_+k] * Q[i*n_+j];
                            dot *= tau;
                            Q[i*n_+(k+1)] -= v0 * dot;
                            for (int j = k+2; j < n_; j++) Q[i*n_+j] -= A[j*n_+k] * dot;
                        }
                        sycl::group_barrier(item.get_group());
                        if (tid == 0) {
                            A[(k+1)*n_+k] = alpha_val; A[k*n_+(k+1)] = alpha_val;
                            for (int i = k+2; i < n_; i++) { A[i*n_+k]=T(0); A[k*n_+i]=T(0); }
                        }
                        sycl::group_barrier(item.get_group());
                    }

                    if (tid == 0) {
                        for (int i = 0; i < n_; i++) d[i] = A[i*n_+i];
                        for (int i = 0; i < n_-1; i++) e[i] = A[(i+1)*n_+i];
                        e[n_-1] = T(0);
                    }
                    sycl::group_barrier(item.get_group());

                    // QL iteration with Wilkinson shifts
                    if (tid == 0) {
                        for (int l = 0; l < n_; l++) {
                            int iter = 0;
                            while (iter < max_iter) {
                                int m_idx = l;
                                for (int i = l; i < n_-1; i++) {
                                    T tst = sycl::fabs(d[i]) + sycl::fabs(d[i+1]);
                                    if (tst == T(0)) tst = T(1);
                                    if (sycl::fabs(e[i]) < eps * tst) break;
                                    m_idx = i + 1;
                                }
                                if (m_idx == l) break;

                                T g = (d[l+1] - d[l]) / (T(2) * e[l]);
                                T r = sycl::sqrt(g*g + T(1));
                                T shift = d[m_idx] - d[l] + e[l] / (g + sycl::copysign(r, g));
                                T s_val = T(1), c_val = T(1), p = T(0);

                                for (int i = m_idx - 1; i >= l; i--) {
                                    T f = s_val * e[i]; T b = c_val * e[i];
                                    if (sycl::fabs(f) >= sycl::fabs(shift)) {
                                        c_val = shift/f; r = sycl::sqrt(c_val*c_val+T(1));
                                        e[i+1] = f*r; s_val = T(1)/r; c_val *= s_val;
                                    } else {
                                        s_val = f/shift; r = sycl::sqrt(s_val*s_val+T(1));
                                        e[i+1] = shift*r; c_val = T(1)/r; s_val *= c_val;
                                    }
                                    shift = d[i+1] - p;
                                    r = (d[i] - shift)*s_val + T(2)*c_val*b;
                                    p = s_val * r; d[i+1] = shift + p; shift = c_val*r - b;
                                    for (int row = 0; row < n_; row++) {
                                        T tmp = Q[row*n_+(i+1)];
                                        Q[row*n_+(i+1)] = s_val * Q[row*n_+i] + c_val * tmp;
                                        Q[row*n_+i] = c_val * Q[row*n_+i] - s_val * tmp;
                                    }
                                }
                                d[l] -= p; e[l] = shift; e[m_idx] = T(0);
                                iter++;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < n_; idx += num_threads) batch_eig[idx] = d[idx];
                    for (int idx = tid; idx < n_ * n_; idx += num_threads) batch_data[idx] = Q[idx];
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32) launch_eigh(work.data<float>(), W.data<float>());
    else launch_eigh(work.data<double>(), W.data<double>());

    return {W, work};
}

// ============================================================================
// SVD (Singular Value Decomposition)
// ============================================================================

auto linalg_svd_kernel(const Tensor& A, bool full_matrices, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "svd");
    if (A.dtype() == DType::Float16) { auto [U,S,Vt] = linalg_svd_kernel(A.to(DType::Float32), full_matrices, queue); return {U,S,Vt}; }
    if (A.dtype() == DType::BFloat16) { auto [U,S,Vt] = linalg_svd_kernel(A.to(DType::Float32), full_matrices, queue); return {U,S,Vt}; }

    auto work = A.contiguous().clone();
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::svd: input must be at least 2D");
    int64_t m = shape[a_ndim-2], n_cols = shape[a_ndim-1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);
    std::vector<int64_t> u_shape = batch_dims, s_shape = batch_dims, vt_shape = batch_dims;
    s_shape.push_back(k);
    if (full_matrices) {
        u_shape.push_back(m); u_shape.push_back(m);
        vt_shape.push_back(n_cols); vt_shape.push_back(n_cols);
    } else {
        u_shape.push_back(m); u_shape.push_back(k);
        vt_shape.push_back(k); vt_shape.push_back(n_cols);
    }
    auto U = zeros(u_shape, A.dtype(), A.device());
    auto S = zeros(s_shape, A.dtype(), A.device());
    auto Vt = zeros(vt_shape, A.dtype(), A.device());

    auto launch_svd = [&](auto* work_ptr, auto* u_ptr, auto* s_ptr, auto* vt_ptr) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        check_size_limit<T>(std::max(m, n_cols), "svd");
        size_t smem_bytes = (m * n_cols + m * m + n_cols * n_cols + 4) * sizeof(T);
        int threads = std::min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        int max_iter = 30 * static_cast<int>(std::max(m, n_cols));
        int u_rows = static_cast<int>(m);
        int u_cols = full_matrices ? static_cast<int>(m) : static_cast<int>(k);
        int vt_rows = full_matrices ? static_cast<int>(n_cols) : static_cast<int>(k);
        int vt_cols_i = static_cast<int>(n_cols);

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_bytes), h);
            int m_ = static_cast<int>(m), nc_ = static_cast<int>(n_cols), k_ = static_cast<int>(k);
            bool full = full_matrices;
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
                    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* As = reinterpret_cast<T*>(smem_raw);
                    T* Us = As + m_ * nc_;
                    T* Vts = Us + m_ * m_;
                    T* scratch = Vts + nc_ * nc_;

                    const T* batch_A = work_ptr + batch_idx * m_ * nc_;
                    T* batch_U = u_ptr + batch_idx * u_rows * u_cols;
                    T* batch_S = s_ptr + batch_idx * k_;
                    T* batch_Vt = vt_ptr + batch_idx * vt_rows * vt_cols_i;

                    for (int idx = tid; idx < m_ * nc_; idx += num_threads) As[idx] = batch_A[idx];
                    for (int idx = tid; idx < m_ * m_; idx += num_threads) { int r=idx/m_, c=idx%m_; Us[idx]=(r==c)?T(1):T(0); }
                    for (int idx = tid; idx < nc_ * nc_; idx += num_threads) { int r=idx/nc_, c=idx%nc_; Vts[idx]=(r==c)?T(1):T(0); }
                    sycl::group_barrier(item.get_group());

                    // Bidiagonalization
                    for (int j = 0; j < k_; j++) {
                        // Left Householder
                        if (tid == 0) {
                            T sigma = T(0);
                            for (int i = j+1; i < m_; i++) sigma += As[i*nc_+j]*As[i*nc_+j];
                            T x0 = As[j*nc_+j];
                            T norm_x = sycl::sqrt(x0*x0+sigma);
                            if (norm_x < zero_tol || sigma < zero_tol) scratch[1]=T(0);
                            else {
                                T alpha = -sycl::copysign(norm_x, x0);
                                T v0 = x0 - alpha;
                                scratch[0]=v0; scratch[1]=T(2)/(v0*v0+sigma); scratch[2]=alpha;
                            }
                        }
                        sycl::group_barrier(item.get_group());
                        T tau = scratch[1];
                        if (tau != T(0)) {
                            T v0 = scratch[0]; T alpha = scratch[2];
                            for (int col = j+tid; col < nc_; col += num_threads) {
                                T dot = v0 * As[j*nc_+col];
                                for (int i = j+1; i < m_; i++) dot += As[i*nc_+j] * As[i*nc_+col];
                                dot *= tau;
                                As[j*nc_+col] -= v0*dot;
                                for (int i = j+1; i < m_; i++) As[i*nc_+col] -= As[i*nc_+j]*dot;
                            }
                            sycl::group_barrier(item.get_group());
                            for (int row = tid; row < m_; row += num_threads) {
                                T dot = v0 * Us[row*m_+j];
                                for (int i = j+1; i < m_; i++) dot += As[i*nc_+j] * Us[row*m_+i];
                                dot *= tau;
                                Us[row*m_+j] -= v0*dot;
                                for (int i = j+1; i < m_; i++) Us[row*m_+i] -= As[i*nc_+j]*dot;
                            }
                            sycl::group_barrier(item.get_group());
                            if (tid == 0) { As[j*nc_+j]=alpha; for (int i=j+1;i<m_;i++) As[i*nc_+j]=T(0); }
                            sycl::group_barrier(item.get_group());
                        }
                        // Right Householder
                        if (j+1 < nc_) {
                            if (tid == 0) {
                                T sigma = T(0);
                                for (int i = j+2; i < nc_; i++) sigma += As[j*nc_+i]*As[j*nc_+i];
                                T x0 = As[j*nc_+(j+1)];
                                T norm_x = sycl::sqrt(x0*x0+sigma);
                                if (norm_x < zero_tol || sigma < zero_tol) scratch[1]=T(0);
                                else {
                                    T alpha = -sycl::copysign(norm_x, x0);
                                    T v0 = x0-alpha;
                                    scratch[0]=v0; scratch[1]=T(2)/(v0*v0+sigma); scratch[2]=alpha;
                                }
                            }
                            sycl::group_barrier(item.get_group());
                            tau = scratch[1];
                            if (tau != T(0)) {
                                T v0 = scratch[0]; T alpha = scratch[2];
                                for (int row = j+tid; row < m_; row += num_threads) {
                                    T dot = v0 * As[row*nc_+(j+1)];
                                    for (int i = j+2; i < nc_; i++) dot += As[j*nc_+i] * As[row*nc_+i];
                                    dot *= tau;
                                    As[row*nc_+(j+1)] -= v0*dot;
                                    for (int i = j+2; i < nc_; i++) As[row*nc_+i] -= As[j*nc_+i]*dot;
                                }
                                sycl::group_barrier(item.get_group());
                                for (int col = tid; col < nc_; col += num_threads) {
                                    T dot = v0 * Vts[(j+1)*nc_+col];
                                    for (int i = j+2; i < nc_; i++) dot += As[j*nc_+i] * Vts[i*nc_+col];
                                    dot *= tau;
                                    Vts[(j+1)*nc_+col] -= v0*dot;
                                    for (int i = j+2; i < nc_; i++) Vts[i*nc_+col] -= As[j*nc_+i]*dot;
                                }
                                sycl::group_barrier(item.get_group());
                                if (tid == 0) { As[j*nc_+(j+1)]=alpha; for (int i=j+2;i<nc_;i++) As[j*nc_+i]=T(0); }
                                sycl::group_barrier(item.get_group());
                            }
                        }
                    }

                    // Bidiagonal QR iteration
                    if (tid == 0) {
                        for (int iter = 0; iter < max_iter * k_; iter++) {
                            int q = k_-1;
                            while (q > 0 && sycl::fabs(As[(q-1)*nc_+q]) < eps*(sycl::fabs(As[(q-1)*nc_+(q-1)])+sycl::fabs(As[q*nc_+q]))) q--;
                            if (q == 0) break;
                            int p = q-1;
                            while (p > 0 && sycl::fabs(As[(p-1)*nc_+p]) >= eps*(sycl::fabs(As[(p-1)*nc_+(p-1)])+sycl::fabs(As[p*nc_+p]))) p--;

                            T d_q = As[q*nc_+q], d_qm1 = As[(q-1)*nc_+(q-1)], e_qm1 = As[(q-1)*nc_+q];
                            T t11 = d_qm1*d_qm1;
                            if (q-2 >= p) t11 += As[(q-2)*nc_+(q-1)]*As[(q-2)*nc_+(q-1)];
                            T t22 = d_q*d_q+e_qm1*e_qm1, t12 = d_qm1*e_qm1;
                            T trace = t11+t22, det = t11*t22-t12*t12;
                            T disc = trace*trace-T(4)*det;
                            if (disc < T(0)) disc = T(0);
                            T sq = sycl::sqrt(disc);
                            T l1 = T(0.5)*(trace+sq), l2 = T(0.5)*(trace-sq);
                            T mu = (sycl::fabs(l1-t22)<sycl::fabs(l2-t22)) ? l1 : l2;

                            T d_p = As[p*nc_+p];
                            T y = d_p*d_p-mu, z = d_p*As[p*nc_+(p+1)];

                            for (int i = p; i < q; i++) {
                                T r = sycl::sqrt(y*y+z*z); T c_v=y/r, s_v=-z/r;
                                for (int row = 0; row < k_; row++) {
                                    if (row>=i-1 && row<=i+1) {
                                        T a1=As[row*nc_+i], a2=As[row*nc_+(i+1)];
                                        As[row*nc_+i]=c_v*a1-s_v*a2; As[row*nc_+(i+1)]=s_v*a1+c_v*a2;
                                    }
                                }
                                for (int col = 0; col < nc_; col++) {
                                    T v1=Vts[i*nc_+col], v2=Vts[(i+1)*nc_+col];
                                    Vts[i*nc_+col]=c_v*v1-s_v*v2; Vts[(i+1)*nc_+col]=s_v*v1+c_v*v2;
                                }
                                y = As[i*nc_+i]; z = As[(i+1)*nc_+i];
                                r = sycl::sqrt(y*y+z*z); c_v=y/r; s_v=-z/r;
                                for (int col = 0; col < nc_; col++) {
                                    if (col>=i && col<=i+2) {
                                        T a1=As[i*nc_+col], a2=As[(i+1)*nc_+col];
                                        As[i*nc_+col]=c_v*a1-s_v*a2; As[(i+1)*nc_+col]=s_v*a1+c_v*a2;
                                    }
                                }
                                for (int row = 0; row < m_; row++) {
                                    T u1=Us[row*m_+i], u2=Us[row*m_+(i+1)];
                                    Us[row*m_+i]=c_v*u1-s_v*u2; Us[row*m_+(i+1)]=s_v*u1+c_v*u2;
                                }
                                if (i+1 < q) { y=As[i*nc_+(i+1)]; z=As[i*nc_+(i+2)]; }
                            }
                        }
                        for (int i = 0; i < k_; i++) {
                            if (As[i*nc_+i] < T(0)) {
                                As[i*nc_+i] = -As[i*nc_+i];
                                for (int row = 0; row < m_; row++) Us[row*m_+i] = -Us[row*m_+i];
                            }
                        }
                        for (int i = 0; i < k_-1; i++) {
                            int max_idx = i; T max_val = As[i*nc_+i];
                            for (int j = i+1; j < k_; j++) if (As[j*nc_+j]>max_val) { max_val=As[j*nc_+j]; max_idx=j; }
                            if (max_idx != i) {
                                T tmp = As[i*nc_+i]; As[i*nc_+i]=As[max_idx*nc_+max_idx]; As[max_idx*nc_+max_idx]=tmp;
                                for (int row = 0; row < m_; row++) { T t=Us[row*m_+i]; Us[row*m_+i]=Us[row*m_+max_idx]; Us[row*m_+max_idx]=t; }
                                for (int col = 0; col < nc_; col++) { T t=Vts[i*nc_+col]; Vts[i*nc_+col]=Vts[max_idx*nc_+col]; Vts[max_idx*nc_+col]=t; }
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < k_; idx += num_threads) batch_S[idx] = As[idx*nc_+idx];
                    int u_out_size = u_rows * u_cols;
                    for (int idx = tid; idx < u_out_size; idx += num_threads) {
                        int r = idx/u_cols, c = idx%u_cols;
                        batch_U[idx] = Us[r*m_+c];
                    }
                    int vt_out_size = vt_rows * vt_cols_i;
                    for (int idx = tid; idx < vt_out_size; idx += num_threads) {
                        int r = idx/vt_cols_i, c = idx%vt_cols_i;
                        batch_Vt[idx] = Vts[r*nc_+c];
                    }
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32) launch_svd(work.data<float>(), U.data<float>(), S.data<float>(), Vt.data<float>());
    else launch_svd(work.data<double>(), U.data<double>(), S.data<double>(), Vt.data<double>());

    return {U, S, Vt};
}

// ============================================================================
// Non-symmetric Eigendecomposition (eig)
// ============================================================================

auto linalg_eig_kernel(const Tensor& A, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "eig");
    if (A.dtype() == DType::Float16) { auto [wr,wi,V] = linalg_eig_kernel(A.to(DType::Float32), queue); return {wr,wi,V}; }
    if (A.dtype() == DType::BFloat16) { auto [wr,wi,V] = linalg_eig_kernel(A.to(DType::Float32), queue); return {wr,wi,V}; }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims; w_shape.push_back(n);
    auto WR = zeros(w_shape, A.dtype(), A.device());
    auto WI = zeros(w_shape, A.dtype(), A.device());

    std::vector<int64_t> v_shape = batch_dims; v_shape.push_back(n); v_shape.push_back(n);
    auto V = zeros(v_shape, A.dtype(), A.device());

    auto launch_eig = [&](auto* work_ptr, auto* wr_ptr, auto* wi_ptr, auto* v_ptr) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        check_size_limit<T>(n, "eig");
        size_t smem_bytes = (2 * n * n + 4) * sizeof(T);
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        int max_iter = 30 * static_cast<int>(n);

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_bytes), h);
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
                    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* H = reinterpret_cast<T*>(smem_raw);
                    T* Q = H + n_ * n_;
                    T* scratch = Q + n_ * n_;

                    const T* As = work_ptr + batch_idx * n_ * n_;
                    T* wr = wr_ptr + batch_idx * n_;
                    T* wi = wi_ptr + batch_idx * n_;
                    T* Vs = v_ptr + batch_idx * n_ * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads) {
                        H[idx] = As[idx];
                        int row = idx/n_, col = idx%n_;
                        Q[idx] = (row==col) ? T(1) : T(0);
                    }
                    sycl::group_barrier(item.get_group());

                    // Hessenberg reduction
                    for (int k = 0; k + 2 < n_; k++) {
                        if (tid == 0) {
                            T sigma = T(0);
                            for (int i = k+1; i < n_; i++) sigma += H[i*n_+k]*H[i*n_+k];
                            T norm_x = sycl::sqrt(sigma);
                            if (norm_x < zero_tol) scratch[1]=T(0);
                            else {
                                T a = -sycl::copysign(norm_x, H[(k+1)*n_+k]);
                                T v0_val = H[(k+1)*n_+k]-a;
                                T v_norm_sq = v0_val*v0_val;
                                for (int i = k+2; i < n_; i++) v_norm_sq += H[i*n_+k]*H[i*n_+k];
                                if (v_norm_sq < zero_tol) scratch[1]=T(0);
                                else { scratch[0]=v0_val; scratch[1]=T(2)/v_norm_sq; scratch[2]=a; }
                            }
                        }
                        sycl::group_barrier(item.get_group());
                        T tau = scratch[1];
                        if (tau == T(0)) { sycl::group_barrier(item.get_group()); continue; }
                        T v0 = scratch[0]; T alpha = scratch[2];

                        for (int j = k+tid; j < n_; j += num_threads) {
                            T dot = v0 * H[(k+1)*n_+j];
                            for (int i = k+2; i < n_; i++) dot += H[i*n_+k]*H[i*n_+j];
                            dot *= tau;
                            H[(k+1)*n_+j] -= v0*dot;
                            for (int i = k+2; i < n_; i++) H[i*n_+j] -= H[i*n_+k]*dot;
                        }
                        sycl::group_barrier(item.get_group());
                        for (int i = tid; i < n_; i += num_threads) {
                            T dot = v0 * H[i*n_+(k+1)];
                            for (int j = k+2; j < n_; j++) dot += H[j*n_+k]*H[i*n_+j];
                            dot *= tau;
                            H[i*n_+(k+1)] -= v0*dot;
                            for (int j = k+2; j < n_; j++) H[i*n_+j] -= H[j*n_+k]*dot;
                        }
                        sycl::group_barrier(item.get_group());
                        for (int i = tid; i < n_; i += num_threads) {
                            T dot = v0 * Q[i*n_+(k+1)];
                            for (int j = k+2; j < n_; j++) dot += H[j*n_+k]*Q[i*n_+j];
                            dot *= tau;
                            Q[i*n_+(k+1)] -= v0*dot;
                            for (int j = k+2; j < n_; j++) Q[i*n_+j] -= H[j*n_+k]*dot;
                        }
                        sycl::group_barrier(item.get_group());
                        if (tid == 0) { H[(k+1)*n_+k]=alpha; for (int i=k+2;i<n_;i++) H[i*n_+k]=T(0); }
                        sycl::group_barrier(item.get_group());
                    }

                    // QR iteration with implicit double shifts
                    if (tid == 0) { scratch[3] = static_cast<T>(n_); }
                    sycl::group_barrier(item.get_group());

                    for (int iter = 0; iter < max_iter; iter++) {
                        int nn = static_cast<int>(scratch[3]);
                        if (nn <= 0) break;

                        if (tid == 0) {
                            bool deflated = false;
                            if (nn >= 2) {
                                T tst = sycl::fabs(H[(nn-2)*n_+(nn-2)])+sycl::fabs(H[(nn-1)*n_+(nn-1)]);
                                if (tst == T(0)) tst = T(1);
                                if (sycl::fabs(H[(nn-1)*n_+(nn-2)]) < eps*tst) {
                                    wr[nn-1]=H[(nn-1)*n_+(nn-1)]; wi[nn-1]=T(0);
                                    H[(nn-1)*n_+(nn-2)]=T(0); scratch[3]=static_cast<T>(nn-1); deflated=true;
                                }
                            }
                            if (!deflated && nn >= 3) {
                                T tst = sycl::fabs(H[(nn-3)*n_+(nn-3)])+sycl::fabs(H[(nn-2)*n_+(nn-2)]);
                                if (tst == T(0)) tst = T(1);
                                if (sycl::fabs(H[(nn-2)*n_+(nn-3)]) < eps*tst) {
                                    T a=H[(nn-2)*n_+(nn-2)], b=H[(nn-2)*n_+(nn-1)];
                                    T c=H[(nn-1)*n_+(nn-2)], dd=H[(nn-1)*n_+(nn-1)];
                                    T trace=a+dd, det=a*dd-b*c, disc=trace*trace-T(4)*det;
                                    if (disc>=T(0)) {
                                        T sq=sycl::sqrt(disc);
                                        wr[nn-2]=T(0.5)*(trace+sq); wi[nn-2]=T(0);
                                        wr[nn-1]=T(0.5)*(trace-sq); wi[nn-1]=T(0);
                                    } else {
                                        T sq=sycl::sqrt(-disc);
                                        wr[nn-2]=T(0.5)*trace; wi[nn-2]=T(0.5)*sq;
                                        wr[nn-1]=T(0.5)*trace; wi[nn-1]=T(-0.5)*sq;
                                    }
                                    H[(nn-2)*n_+(nn-3)]=T(0); scratch[3]=static_cast<T>(nn-2); deflated=true;
                                }
                            }
                            if (!deflated && nn == 1) { wr[0]=H[0]; wi[0]=T(0); scratch[3]=T(0); deflated=true; }
                            scratch[2] = deflated ? T(1) : T(0);
                        }
                        sycl::group_barrier(item.get_group());
                        if (scratch[2] != T(0)) continue;

                        nn = static_cast<int>(scratch[3]);
                        T x, y, z;
                        if (tid == 0) {
                            T s = H[(nn-2)*n_+(nn-2)]+H[(nn-1)*n_+(nn-1)];
                            T t = H[(nn-2)*n_+(nn-2)]*H[(nn-1)*n_+(nn-1)]-H[(nn-2)*n_+(nn-1)]*H[(nn-1)*n_+(nn-2)];
                            scratch[0] = H[0]*H[0]+H[1]*H[n_]-s*H[0]+t;
                            scratch[1] = H[n_]*(H[0]+H[n_+1]-s);
                            scratch[2] = (nn>2) ? H[n_]*H[2*n_+1] : T(0);
                        }
                        sycl::group_barrier(item.get_group());
                        x=scratch[0]; y=scratch[1]; z=scratch[2];

                        for (int k = 0; k+2 < nn; k++) {
                            T norm_v = sycl::sqrt(x*x+y*y+z*z);
                            if (norm_v < zero_tol) {
                                if (tid == 0) {
                                    x=H[(k+1)*n_+k]; y=(k+2<nn)?H[(k+2)*n_+k]:T(0); z=(k+3<nn)?H[(k+3)*n_+k]:T(0);
                                    scratch[0]=x; scratch[1]=y; scratch[2]=z;
                                }
                                sycl::group_barrier(item.get_group());
                                x=scratch[0]; y=scratch[1]; z=scratch[2]; continue;
                            }
                            T alpha_h = -sycl::copysign(norm_v, x);
                            T v0h=x-alpha_h, v1h=y, v2h=z;
                            T v_sq = v0h*v0h+v1h*v1h+v2h*v2h;
                            if (v_sq < zero_tol) {
                                if (tid == 0) {
                                    x=H[(k+1)*n_+k]; y=(k+2<nn)?H[(k+2)*n_+k]:T(0); z=(k+3<nn)?H[(k+3)*n_+k]:T(0);
                                    scratch[0]=x; scratch[1]=y; scratch[2]=z;
                                }
                                sycl::group_barrier(item.get_group());
                                x=scratch[0]; y=scratch[1]; z=scratch[2]; continue;
                            }
                            T tau_h = T(2)/v_sq;
                            int m_lim = (k+4<nn) ? k+4 : nn;

                            for (int j = k+tid; j < nn; j += num_threads) {
                                T dot = v0h*H[k*n_+j]+v1h*H[(k+1)*n_+j];
                                if (k+2<nn) dot += v2h*H[(k+2)*n_+j];
                                dot *= tau_h;
                                H[k*n_+j]-=v0h*dot; H[(k+1)*n_+j]-=v1h*dot;
                                if (k+2<nn) H[(k+2)*n_+j]-=v2h*dot;
                            }
                            sycl::group_barrier(item.get_group());
                            for (int i = tid; i < m_lim; i += num_threads) {
                                T dot = v0h*H[i*n_+k]+v1h*H[i*n_+(k+1)];
                                if (k+2<nn) dot += v2h*H[i*n_+(k+2)];
                                dot *= tau_h;
                                H[i*n_+k]-=v0h*dot; H[i*n_+(k+1)]-=v1h*dot;
                                if (k+2<nn) H[i*n_+(k+2)]-=v2h*dot;
                            }
                            sycl::group_barrier(item.get_group());
                            for (int i = tid; i < n_; i += num_threads) {
                                T dot = v0h*Q[i*n_+k]+v1h*Q[i*n_+(k+1)];
                                if (k+2<nn) dot += v2h*Q[i*n_+(k+2)];
                                dot *= tau_h;
                                Q[i*n_+k]-=v0h*dot; Q[i*n_+(k+1)]-=v1h*dot;
                                if (k+2<nn) Q[i*n_+(k+2)]-=v2h*dot;
                            }
                            sycl::group_barrier(item.get_group());
                            if (tid == 0) {
                                if (k>0) H[k*n_+(k-1)]=alpha_h;
                                if (k+2<nn) H[(k+2)*n_+k]=T(0);
                                if (k+3<nn) H[(k+3)*n_+k]=T(0);
                                x=H[(k+1)*n_+k]; y=(k+2<nn)?H[(k+2)*n_+k]:T(0); z=(k+3<nn)?H[(k+3)*n_+k]:T(0);
                                scratch[0]=x; scratch[1]=y; scratch[2]=z;
                            }
                            sycl::group_barrier(item.get_group());
                            x=scratch[0]; y=scratch[1]; z=scratch[2];
                        }

                        if (nn >= 2) {
                            T norm_v = sycl::sqrt(x*x+y*y);
                            if (norm_v > zero_tol) {
                                int kk = nn-2;
                                T c_v=x/norm_v, s_v=-y/norm_v;
                                for (int j = kk+tid; j < nn; j += num_threads) {
                                    T tmp = c_v*H[kk*n_+j]-s_v*H[(kk+1)*n_+j];
                                    H[(kk+1)*n_+j] = s_v*H[kk*n_+j]+c_v*H[(kk+1)*n_+j];
                                    H[kk*n_+j] = tmp;
                                }
                                sycl::group_barrier(item.get_group());
                                for (int i = tid; i < nn; i += num_threads) {
                                    T tmp = c_v*H[i*n_+kk]-s_v*H[i*n_+(kk+1)];
                                    H[i*n_+(kk+1)] = s_v*H[i*n_+kk]+c_v*H[i*n_+(kk+1)];
                                    H[i*n_+kk] = tmp;
                                }
                                sycl::group_barrier(item.get_group());
                                for (int i = tid; i < n_; i += num_threads) {
                                    T tmp = c_v*Q[i*n_+kk]-s_v*Q[i*n_+(kk+1)];
                                    Q[i*n_+(kk+1)] = s_v*Q[i*n_+kk]+c_v*Q[i*n_+(kk+1)];
                                    Q[i*n_+kk] = tmp;
                                }
                                sycl::group_barrier(item.get_group());
                            }
                        }
                    }

                    if (tid == 0) {
                        int nn = static_cast<int>(scratch[3]);
                        if (nn == 1) { wr[0]=H[0]; wi[0]=T(0); }
                        else if (nn == 2) {
                            T a=H[0], b=H[1], c=H[n_], dd=H[n_+1];
                            T trace=a+dd, det=a*dd-b*c, disc=trace*trace-T(4)*det;
                            if (disc>=T(0)) {
                                T sq=sycl::sqrt(disc);
                                wr[0]=T(0.5)*(trace+sq); wi[0]=T(0);
                                wr[1]=T(0.5)*(trace-sq); wi[1]=T(0);
                            } else {
                                T sq=sycl::sqrt(-disc);
                                wr[0]=T(0.5)*trace; wi[0]=T(0.5)*sq;
                                wr[1]=T(0.5)*trace; wi[1]=T(-0.5)*sq;
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < n_ * n_; idx += num_threads) Vs[idx] = Q[idx];
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32) launch_eig(work.data<float>(), WR.data<float>(), WI.data<float>(), V.data<float>());
    else launch_eig(work.data<double>(), WR.data<double>(), WI.data<double>(), V.data<double>());

    return {WR, WI, V};
}

#endif // TENZOR_HAS_ONEMKL

} // namespace oneapi
} // namespace tenzor
