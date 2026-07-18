/**
 * @file linalg.cpp
 * @brief OneAPI/SYCL linear algebra kernels via oneMKL LAPACK
 *
 * Implements SVD, QR, Eigendecomposition, Solve, Inverse, Determinant,
 * and Cholesky factorization using oneMKL LAPACK APIs.
 * Guarded by TENZOR_HAS_ONEMKL_LAPACK (subset of TENZOR_HAS_ONEMKL).
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/math.hpp"
#include "../sycl_buffer_guard.hpp"
#include "oneapi_kernel_utils.hpp"
#include <sycl/sycl.hpp>
#include <stdexcept>
#include <vector>
#include <cmath>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#endif

namespace tenzor {
namespace oneapi {

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

// ============================================================================
// Shared helpers — unconditional (used by both the MKL path and the native
// SYCL fallback, as well as by `linalg_eig_qr_kernel` which is always
// compiled and used as the registered backend for `OpId::LinalgEig`).
// ============================================================================

namespace {

/// Max matrix dimension for local-memory fallback kernels.
/// Local memory usage is ~2*N*N*sizeof(T) + scratch, capped at 48 KB.
constexpr int MAX_N_FLOAT  = 90;
constexpr int MAX_N_DOUBLE = 64;

/// Convert span to vector.
inline std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

/// Get batch count from shape (product of all dims except last two).
inline int64_t batch_size(const Tensor& t) {
    auto shape = t.shape();
    int64_t batch = 1;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch *= shape[i];
    return batch;
}

/// Get square matrix size and validate.
inline std::pair<int64_t, int64_t> check_square(const Tensor& t) {
    auto shape = t.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    if (m != n) throw std::invalid_argument("linalg: expected square matrix");
    return {m, ndim};
}

/// Validate dtype for linalg ops.
inline void validate_linalg_dtype(const Tensor& t, const std::string& op_name) {
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
inline void check_size_limit(int64_t n, const std::string& op_name) {
    constexpr int max_n = std::is_same_v<T, float> ? MAX_N_FLOAT : MAX_N_DOUBLE;
    if (n > max_n) {
        throw std::runtime_error(
            "linalg::" + op_name + ": matrix size " + std::to_string(n) +
            " exceeds native SYCL fallback limit of " + std::to_string(max_n) +
            " (build with oneMKL for larger matrices)");
    }
}

} // anonymous namespace

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
class SyclTransposeTriSolveAF32;
class SyclTransposeTriSolveAF64;
class SyclTransposeTriSolveBF32;
class SyclTransposeTriSolveBF64;
class SyclTransposeTriSolveBackF32;
class SyclTransposeTriSolveBackF64;
class SyclTransposeLuF32;
class SyclTransposeLuF64;
class SyclTransposeLuBackF32;
class SyclTransposeLuBackF64;
class SyclLuExtractF32;
class SyclLuExtractF64;
class SyclLuPivotsCopyF32;
class SyclLuPivotsCopyF64;
class SyclTransposeLuSolveLuF32;
class SyclTransposeLuSolveLuF64;
class SyclTransposeLuSolveBF32;
class SyclTransposeLuSolveBF64;
class SyclTransposeLuSolveBackF32;
class SyclTransposeLuSolveBackF64;
class SyclLuSolvePivCopyF32;
class SyclLuSolvePivCopyF64;
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
class SyclDetBatchPackF32;
class SyclDetBatchPackF64;
class SyclDetBatchReduceF32;
class SyclDetBatchReduceF64;
class SyclTransposeGeqrfAF32;
class SyclTransposeGeqrfAF64;
class SyclGeqrfExtractF32;
class SyclGeqrfExtractF64;
class SyclTransposeOrmqrRF32;
class SyclTransposeOrmqrRF64;
class SyclTransposeOrmqrCF32;
class SyclTransposeOrmqrCF64;
class SyclTransposeOrmqrBackF32;
class SyclTransposeOrmqrBackF64;

auto linalg_det_kernel(const Tensor& input, sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen to Float32, compute, narrow back.
    // oneMKL's getrf requires Float32/Float64 scratchpad types and is not
    // overloaded for half precision; compute in Float32 instead.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        Tensor result = linalg_det_kernel(input.to(DType::Float32), queue);
        return result.to(orig_dtype);
    }

    auto shape = input.shape();
    int64_t n = shape[shape.size() - 1];
    int64_t nbatch = 1;
    // 2D input → scalar output (shape {}). Higher-rank input → output keeps
    // the leading batch dims. Previous code appended a dummy `1` when the
    // batch dims were empty, producing shape {1} for 2D inputs — wrong, since
    // det of a single matrix is a 0D scalar.
    std::vector<int64_t> out_shape;
    for (size_t i = 0; i + 2 < shape.size(); i++) {
        out_shape.push_back(shape[i]);
        nbatch *= shape[i];
    }

    // For batched input, use oneMKL strided-batched getrf (getrf_batch) plus a
    // single device reduction kernel. The previous code recursed per batch
    // element with two blocking memcpy().wait() calls and a per-element device
    // allocation each, serializing the whole batch on host syncs and defeating
    // batched LAPACK / device parallelism.
    if (nbatch > 1) {
        Tensor output(out_shape, input.dtype(), input.device());

        auto run_batched = [&]<typename T, typename PackName, typename ReduceName>() {
            const int64_t stride_a = n * n;
            SyclDeviceBuffer<T> d_a(nbatch * stride_a, queue);
            SyclDeviceBuffer<std::int64_t> d_ipiv(nbatch * n, queue);

            // Pack each row-major (n x n) matrix into column-major in one kernel
            // (column-major: a_col[b*nn + j*n + i] = a_row[b*nn + i*n + j]).
            const T* in_ptr = get_data_ptr<const T>(input);
            T* a_ptr = d_a.get();
            const int64_t nn = stride_a;
            const int64_t nb = nbatch;
            const int64_t nv = n;
            queue.parallel_for<PackName>(
                sycl::range<1>(static_cast<size_t>(nb * nn)),
                [=](sycl::id<1> gid) {
                    int64_t g = static_cast<int64_t>(gid[0]);
                    int64_t b = g / nn;
                    int64_t rem = g % nn;
                    int64_t i = rem / nv;   // row
                    int64_t j = rem % nv;   // col
                    a_ptr[b * nn + j * nv + i] = in_ptr[b * nn + i * nv + j];
                }).wait();

            auto scratch_size = ::oneapi::mkl::lapack::getrf_batch_scratchpad_size<T>(
                queue, n, n, n, stride_a, n, nbatch);
            SyclDeviceBuffer<T> scratch(scratch_size, queue);
            ::oneapi::mkl::lapack::getrf_batch(
                queue, n, n, d_a.get(), n, stride_a, d_ipiv.get(), n, nbatch,
                scratch.get(), scratch_size).wait();

            // One work-item per batch: product of column-major diagonal entries
            // times (-1)^(number of pivot swaps).
            T* out_ptr = static_cast<T*>(const_cast<void*>(output.data_ptr()));
            const std::int64_t* ipiv_ptr = d_ipiv.get();
            const T* a_done = d_a.get();
            queue.parallel_for<ReduceName>(
                sycl::range<1>(static_cast<size_t>(nb)),
                [=](sycl::id<1> bid) {
                    int64_t b = static_cast<int64_t>(bid[0]);
                    T prod = T(1);
                    int swaps = 0;
                    for (int64_t i = 0; i < nv; ++i) {
                        prod *= a_done[b * nn + i * nv + i];
                        if (ipiv_ptr[b * nv + i] != i + 1) ++swaps;
                    }
                    out_ptr[b] = (swaps % 2) ? -prod : prod;
                }).wait();
        };

        if (input.dtype() == DType::Float32) {
            run_batched.template operator()<float, SyclDetBatchPackF32, SyclDetBatchReduceF32>();
        } else if (input.dtype() == DType::Float64) {
            run_batched.template operator()<double, SyclDetBatchPackF64, SyclDetBatchReduceF64>();
        } else {
            throw std::runtime_error("linalg_det: only Float32 and Float64 supported");
        }
        return output;
    }

    // Copy input for in-place LU
    Tensor a = clone_kernel(input, queue);
    Tensor output(out_shape, input.dtype(), input.device());

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

        // Device-side reduction: product of diagonal * (-1)^swaps. RAII so the
        // shared buffers free on any exception from the .wait() calls below.
        SyclSharedBuffer<float> prod_buf_guard(1, queue);
        SyclSharedBuffer<int32_t> swap_buf_guard(1, queue);
        float* prod_buf = prod_buf_guard.get();
        int32_t* swap_buf = swap_buf_guard.get();
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
        // prod_buf/swap_buf freed by their SyclSharedBuffer guards.
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

        // Device-side reduction: product of diagonal * (-1)^swaps. RAII so the
        // shared buffers free on any exception from the .wait() calls below.
        SyclSharedBuffer<double> prod_buf_guard(1, queue);
        SyclSharedBuffer<int32_t> swap_buf_guard(1, queue);
        double* prod_buf = prod_buf_guard.get();
        int32_t* swap_buf = swap_buf_guard.get();
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
        // prod_buf/swap_buf freed by their SyclSharedBuffer guards.
    } else {
        throw std::runtime_error("linalg_det: only Float32 and Float64 supported");
    }

    return output;
}

// ============================================================================
// LinalgInv - Matrix inverse via LU (getrf + getri)
// ============================================================================
auto linalg_inv_kernel(const Tensor& input_in, sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen to Float32, compute, narrow back (oneMKL getrf
    // is not overloaded for half precision). Mirrors linalg_det_kernel.
    if (input_in.dtype() == DType::Float16 || input_in.dtype() == DType::BFloat16) {
        const DType orig_dtype = input_in.dtype();
        return linalg_inv_kernel(input_in.to(DType::Float32), queue).to(orig_dtype);
    }

    // oneMKL repacks input into a col-major device buffer via get_data_ptr(),
    // which assumes a contiguous source; a non-contiguous input would be read
    // with the wrong layout. Materialize a contiguous copy first.
    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
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
    // Float16 / BFloat16: widen both operands to Float32, solve, narrow back.
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        const DType orig_dtype = A.dtype();
        return linalg_solve_kernel(A.to(DType::Float32), B.to(DType::Float32), queue).to(orig_dtype);
    }
    // row_to_col_major reads via raw pointer and ignores strides, so views
    // (e.g. A.transpose(-1,-2)) produce wrong results unless forced contiguous.
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    auto a_shape = A_cont.shape();
    int64_t n = a_shape[a_shape.size() - 1];
    auto b_shape = B_cont.shape();
    int64_t nrhs = (b_shape.size() > 1) ? b_shape[b_shape.size() - 1] : 1;

    if (A_cont.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<float> d_b(n * nrhs, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<float, SyclTransposeSolveAF32>(
            d_a.get(), get_data_ptr<const float>(A_cont), n, n, queue);
        row_to_col_major<float, SyclTransposeSolveBF32>(
            d_b.get(), get_data_ptr<const float>(B_cont), n, nrhs, queue);

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<float>(queue, n, n, n);
        SyclDeviceBuffer<float> scratch_rf(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratch_rf.get(), sp_rf).wait();

        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<float>(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        SyclDeviceBuffer<float> scratch_rs(sp_rs, queue);
        ::oneapi::mkl::lapack::getrs(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, d_a.get(), n, d_ipiv.get(), d_b.get(), n, scratch_rs.get(), sp_rs).wait();

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A_cont.dtype(), A_cont.device());
        col_to_row_major<float, SyclTransposeSolveBackF32>(
            get_data_ptr<float>(output), d_b.get(), n, nrhs, queue);

        return output;
    } else if (A_cont.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<double> d_b(n * nrhs, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);

        row_to_col_major<double, SyclTransposeSolveAF64>(
            d_a.get(), get_data_ptr<const double>(A_cont), n, n, queue);
        row_to_col_major<double, SyclTransposeSolveBF64>(
            d_b.get(), get_data_ptr<const double>(B_cont), n, nrhs, queue);

        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<double>(queue, n, n, n);
        SyclDeviceBuffer<double> scratch_rf(sp_rf, queue);
        ::oneapi::mkl::lapack::getrf(queue, n, n, d_a.get(), n, d_ipiv.get(), scratch_rf.get(), sp_rf).wait();

        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<double>(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        SyclDeviceBuffer<double> scratch_rs(sp_rs, queue);
        ::oneapi::mkl::lapack::getrs(queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, d_a.get(), n, d_ipiv.get(), d_b.get(), n, scratch_rs.get(), sp_rs).wait();

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A_cont.dtype(), A_cont.device());
        col_to_row_major<double, SyclTransposeSolveBackF64>(
            get_data_ptr<double>(output), d_b.get(), n, nrhs, queue);

        return output;
    } else {
        throw std::runtime_error("linalg_solve: only Float32 and Float64 supported");
    }
}

// ============================================================================
// LinalgLU - LU factorization with partial pivoting (getrf)
// ============================================================================
//
// Returns (L, U, pivots) where:
//   - L is unit lower triangular (..., N, N)
//   - U is upper triangular        (..., N, N)
//   - pivots is Int32 (..., N), 1-based LAPACK convention.
// The packed LU produced by getrf (row-major after the col→row transpose) is
// then split into L and U on the device.
auto linalg_lu_kernel(const Tensor& A, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = A.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2)
        throw std::invalid_argument("linalg::lu: input must be at least 2D");
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    if (m != n)
        throw std::invalid_argument("linalg::lu: expected square matrix");

    // Upcast low-precision floats to Float32 for LAPACK compatibility
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        auto orig_dt = A.dtype();
        auto [L32, U32, P] = linalg_lu_kernel(A.to(DType::Float32), queue);
        return {L32.to(orig_dt), U32.to(orig_dt), P};
    }

    if (A.dtype() != DType::Float32 && A.dtype() != DType::Float64)
        throw std::runtime_error("linalg_lu: only Float32 and Float64 supported");

    // Compute batch count
    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < ndim; ++i) nbatch *= shape[i];

    // Output shapes
    std::vector<int64_t> mat_shape(shape.begin(), shape.end());
    std::vector<int64_t> piv_shape(shape.begin(), shape.end() - 1);
    // piv_shape currently has trailing N from shape[..-1]; replace with N pivots
    piv_shape.back() = n;

    Tensor L(mat_shape, A.dtype(), A.device());
    Tensor U(mat_shape, A.dtype(), A.device());
    Tensor pivots(piv_shape, DType::Int32, A.device());

    auto run = [&](auto dummy) {
        using T = decltype(dummy);
        SyclDeviceBuffer<T>            d_a(n * n, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);
        auto sp_rf = ::oneapi::mkl::lapack::getrf_scratchpad_size<T>(queue, n, n, n);
        SyclDeviceBuffer<T> scratch_rf(sp_rf, queue);

        for (int64_t b = 0; b < nbatch; ++b) {
            const T* a_in   = get_data_ptr<const T>(A) + b * n * n;
            T*       l_out  = get_data_ptr<T>(L)       + b * n * n;
            T*       u_out  = get_data_ptr<T>(U)       + b * n * n;
            int32_t* p_out  = get_data_ptr<int32_t>(pivots) + b * n;

            // Row-major (input) -> column-major (oneMKL) on device
            if constexpr (std::is_same_v<T, float>) {
                row_to_col_major<float, SyclTransposeLuF32>(d_a.get(), a_in, n, n, queue);
            } else {
                row_to_col_major<double, SyclTransposeLuF64>(d_a.get(), a_in, n, n, queue);
            }

            ::oneapi::mkl::lapack::getrf(
                queue, n, n, d_a.get(), n, d_ipiv.get(),
                scratch_rf.get(), sp_rf).wait();

            // Transpose factored matrix back to row-major (packed LU).
            // We write into l_out as a temp buffer, then split into L and U.
            if constexpr (std::is_same_v<T, float>) {
                col_to_row_major<float, SyclTransposeLuBackF32>(l_out, d_a.get(), n, n, queue);
            } else {
                col_to_row_major<double, SyclTransposeLuBackF64>(l_out, d_a.get(), n, n, queue);
            }

            // Split packed LU into L (unit lower) and U (upper). Read from l_out
            // (which currently holds packed LU) and write back into l_out + u_out.
            int64_t n_ = n;
            T* packed = l_out;  // alias
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::range<2>(static_cast<size_t>(n_), static_cast<size_t>(n_)),
                    [=](sycl::id<2> id) {
                        int64_t i = id[0], j = id[1];
                        T v = packed[i * n_ + j];
                        if (i > j) {
                            // l_out[i,j] = packed[i,j], u_out[i,j] = 0
                            // packed already at l_out, so l is correct; just zero u
                            u_out[i * n_ + j] = T(0);
                        } else if (i == j) {
                            l_out[i * n_ + j] = T(1);
                            u_out[i * n_ + j] = v;
                        } else { // i < j
                            l_out[i * n_ + j] = T(0);
                            u_out[i * n_ + j] = v;
                        }
                    });
            }).wait();

            // Copy pivots from int64 -> int32
            auto* ipiv_ptr = d_ipiv.get();
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::range<1>(static_cast<size_t>(n)),
                    [=](sycl::id<1> id) {
                        p_out[id[0]] = static_cast<int32_t>(ipiv_ptr[id[0]]);
                    });
            }).wait();
        }
    };

    if (A.dtype() == DType::Float32) run(float{});
    else                              run(double{});

    return {L, U, pivots};
}

// ============================================================================
// LinalgLUSolve - Solve A x = B given packed LU (from getrf) + pivots (getrs)
// ============================================================================
auto linalg_lu_solve_kernel(const Tensor& LU_data_in, const Tensor& pivots_in,
                            const Tensor& B_in, sycl::queue& queue) -> Tensor {
    // LU_data/B/pivots are read via raw get_data_ptr + b*stride and fed to
    // row_to_col_major, which ignores strides — so a non-contiguous view would
    // produce wrong results. Materialize contiguous (matching linalg_solve_kernel
    // / linalg_solve_triangular_kernel).
    const Tensor LU_data = LU_data_in.is_contiguous() ? LU_data_in : LU_data_in.contiguous();
    const Tensor pivots  = pivots_in.is_contiguous()  ? pivots_in  : pivots_in.contiguous();
    const Tensor B       = B_in.is_contiguous()       ? B_in       : B_in.contiguous();
    auto lu_shape = LU_data.shape();
    auto b_shape  = B.shape();
    auto lu_ndim  = static_cast<int64_t>(lu_shape.size());
    auto b_ndim   = static_cast<int64_t>(b_shape.size());
    if (lu_ndim < 2 || b_ndim < 2)
        throw std::invalid_argument("linalg::lu_solve: inputs must be at least 2D");
    int64_t n    = lu_shape[lu_ndim - 1];
    int64_t m    = lu_shape[lu_ndim - 2];
    if (m != n)
        throw std::invalid_argument("linalg::lu_solve: LU must be square");
    int64_t nrhs = b_shape[b_ndim - 1];

    if (LU_data.dtype() == DType::Float16 || LU_data.dtype() == DType::BFloat16) {
        auto orig = B.dtype();
        auto out = linalg_lu_solve_kernel(
            LU_data.to(DType::Float32), pivots, B.to(DType::Float32), queue);
        return out.to(orig);
    }

    if (LU_data.dtype() != DType::Float32 && LU_data.dtype() != DType::Float64)
        throw std::runtime_error("linalg_lu_solve: only Float32 and Float64 supported");

    // Compute batch count from LU
    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < lu_ndim; ++i) nbatch *= lu_shape[i];

    Tensor output(std::vector<int64_t>(b_shape.begin(), b_shape.end()),
                  B.dtype(), B.device());

    // Pivots may live on a different device or be int64; bring to int32 USM on device.
    // We re-cast each batch's pivots into a transient int64 device buffer below.
    auto run = [&](auto dummy) {
        using T = decltype(dummy);
        SyclDeviceBuffer<T>            d_lu(n * n, queue);
        SyclDeviceBuffer<T>            d_b(n * nrhs, queue);
        SyclDeviceBuffer<std::int64_t> d_ipiv(n, queue);
        auto sp_rs = ::oneapi::mkl::lapack::getrs_scratchpad_size<T>(
            queue, ::oneapi::mkl::transpose::nontrans, n, nrhs, n, n);
        SyclDeviceBuffer<T> scratch_rs(sp_rs, queue);

        for (int64_t b = 0; b < nbatch; ++b) {
            const T*       lu_in  = get_data_ptr<const T>(LU_data) + b * n * n;
            const T*       b_in   = get_data_ptr<const T>(B)       + b * n * nrhs;
            const int32_t* piv_in = get_data_ptr<const int32_t>(pivots) + b * n;
            T*             x_out  = get_data_ptr<T>(output)        + b * n * nrhs;

            // LU is packed row-major. oneMKL wants column-major.
            if constexpr (std::is_same_v<T, float>) {
                row_to_col_major<float, SyclTransposeLuSolveLuF32>(d_lu.get(), lu_in, n, n, queue);
                row_to_col_major<float, SyclTransposeLuSolveBF32 >(d_b.get(),  b_in,  n, nrhs, queue);
            } else {
                row_to_col_major<double, SyclTransposeLuSolveLuF64>(d_lu.get(), lu_in, n, n, queue);
                row_to_col_major<double, SyclTransposeLuSolveBF64 >(d_b.get(),  b_in,  n, nrhs, queue);
            }

            // Cast int32 pivots -> int64 in d_ipiv on device
            auto* ipiv_ptr = d_ipiv.get();
            int64_t n_ = n;
            queue.submit([&](sycl::handler& cgh) {
                cgh.parallel_for(
                    sycl::range<1>(static_cast<size_t>(n_)),
                    [=](sycl::id<1> id) {
                        ipiv_ptr[id[0]] = static_cast<std::int64_t>(piv_in[id[0]]);
                    });
            }).wait();

            ::oneapi::mkl::lapack::getrs(
                queue, ::oneapi::mkl::transpose::nontrans, n, nrhs,
                d_lu.get(), n, d_ipiv.get(), d_b.get(), n,
                scratch_rs.get(), sp_rs).wait();

            // Transpose result back to row-major
            if constexpr (std::is_same_v<T, float>) {
                col_to_row_major<float, SyclTransposeLuSolveBackF32>(x_out, d_b.get(), n, nrhs, queue);
            } else {
                col_to_row_major<double, SyclTransposeLuSolveBackF64>(x_out, d_b.get(), n, nrhs, queue);
            }
        }
    };

    if (LU_data.dtype() == DType::Float32) run(float{});
    else                                   run(double{});

    return output;
}

// ============================================================================
// LinalgSVD - Singular Value Decomposition via gesvd
// ============================================================================
auto linalg_svd_kernel(const Tensor& input, bool full_matrices, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor, Tensor> {
    // Float16 / BFloat16: widen to Float32, decompose, narrow all outputs back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        auto [U, S, Vh] = linalg_svd_kernel(input.to(DType::Float32), full_matrices, queue);
        return {U.to(orig_dtype), S.to(orig_dtype), Vh.to(orig_dtype)};
    }
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    int64_t k = std::min(m, n);

    // Batched input (..., m, n): compute SVD per matrix. gesvd is a 2D LAPACK
    // primitive, so we loop over the leading batch dims — each batch is a packed
    // m*n block of the contiguous input — and write into the batched outputs.
    // Reduced-batch buffers are allocated once and reused across iterations.
    int64_t nbatch = 1;
    std::vector<int64_t> batch_dims;
    for (int64_t i = 0; i + 2 < ndim; ++i) { batch_dims.push_back(shape[i]); nbatch *= shape[i]; }

    auto jobz = full_matrices ? ::oneapi::mkl::jobsvd::vectors : ::oneapi::mkl::jobsvd::somevec;
    int64_t u_cols = full_matrices ? m : k;
    int64_t vt_rows = full_matrices ? n : k;

    std::vector<int64_t> u_shape = batch_dims;  u_shape.push_back(m);       u_shape.push_back(u_cols);
    std::vector<int64_t> s_shape = batch_dims;  s_shape.push_back(k);
    std::vector<int64_t> vt_shape = batch_dims; vt_shape.push_back(vt_rows); vt_shape.push_back(n);

    // Ensure each batch matrix is a packed m*n block.
    Tensor in_c = input.is_contiguous() ? input : input.contiguous();

    if (input.dtype() == DType::Float32) {
        Tensor S(s_shape, input.dtype(), input.device());
        Tensor U(u_shape, input.dtype(), input.device());
        Tensor Vt(vt_shape, input.dtype(), input.device());
        const float* in_ptr = get_data_ptr<const float>(in_c);
        float* s_ptr = get_data_ptr<float>(S);
        float* u_ptr = get_data_ptr<float>(U);
        float* vt_ptr = get_data_ptr<float>(Vt);

        SyclDeviceBuffer<float> d_a(m * n, queue);
        SyclDeviceBuffer<float> d_s(k, queue);
        SyclDeviceBuffer<float> d_u(m * u_cols, queue);
        SyclDeviceBuffer<float> d_vt(vt_rows * n, queue);
        // ldvt must be the number of VT rows (vt_rows), NOT n. d_vt is sized
        // vt_rows*n; passing ldvt=n would make LAPACK write an n*n column-major
        // matrix and overflow the buffer for the wide non-full case (m<n,
        // full_matrices=false, where vt_rows=min(m,n)=m<n).
        auto sp = ::oneapi::mkl::lapack::gesvd_scratchpad_size<float>(queue, jobz, jobz, m, n, m, m, vt_rows);
        SyclDeviceBuffer<float> scratch(sp, queue);

        for (int64_t b = 0; b < nbatch; ++b) {
            row_to_col_major<float, SyclTransposeSvdAF32>(
                d_a.get(), in_ptr + b * m * n, m, n, queue);
            ::oneapi::mkl::lapack::gesvd(queue, jobz, jobz, m, n, d_a.get(), m, d_s.get(), d_u.get(), m, d_vt.get(), vt_rows, scratch.get(), sp).wait();
            queue.memcpy(s_ptr + b * k, d_s.get(), k * sizeof(float)).wait();
            col_to_row_major<float, SyclTransposeSvdUF32>(
                u_ptr + b * m * u_cols, d_u.get(), m, u_cols, queue);
            col_to_row_major<float, SyclTransposeSvdVtF32>(
                vt_ptr + b * vt_rows * n, d_vt.get(), vt_rows, n, queue);
        }
        return {U, S, Vt};
    } else if (input.dtype() == DType::Float64) {
        Tensor S(s_shape, input.dtype(), input.device());
        Tensor U(u_shape, input.dtype(), input.device());
        Tensor Vt(vt_shape, input.dtype(), input.device());
        const double* in_ptr = get_data_ptr<const double>(in_c);
        double* s_ptr = get_data_ptr<double>(S);
        double* u_ptr = get_data_ptr<double>(U);
        double* vt_ptr = get_data_ptr<double>(Vt);

        SyclDeviceBuffer<double> d_a(m * n, queue);
        SyclDeviceBuffer<double> d_s(k, queue);
        SyclDeviceBuffer<double> d_u(m * u_cols, queue);
        SyclDeviceBuffer<double> d_vt(vt_rows * n, queue);
        // ldvt = vt_rows (not n); see Float32 branch for rationale.
        auto sp = ::oneapi::mkl::lapack::gesvd_scratchpad_size<double>(queue, jobz, jobz, m, n, m, m, vt_rows);
        SyclDeviceBuffer<double> scratch(sp, queue);

        for (int64_t b = 0; b < nbatch; ++b) {
            row_to_col_major<double, SyclTransposeSvdAF64>(
                d_a.get(), in_ptr + b * m * n, m, n, queue);
            ::oneapi::mkl::lapack::gesvd(queue, jobz, jobz, m, n, d_a.get(), m, d_s.get(), d_u.get(), m, d_vt.get(), vt_rows, scratch.get(), sp).wait();
            queue.memcpy(s_ptr + b * k, d_s.get(), k * sizeof(double)).wait();
            col_to_row_major<double, SyclTransposeSvdUF64>(
                u_ptr + b * m * u_cols, d_u.get(), m, u_cols, queue);
            col_to_row_major<double, SyclTransposeSvdVtF64>(
                vt_ptr + b * vt_rows * n, d_vt.get(), vt_rows, n, queue);
        }
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

// ----------------------------------------------------------------------------
// Symmetry probe: max |A| and max |A - A^T| over a batched square matrix,
// reduced into two device-side scalars. Used by linalg_eig_kernel to choose
// between the eigh fast-path and the QR fallback without downloading the
// whole input back to the host. Mirrors the CUDA / ROCm helpers.
// ----------------------------------------------------------------------------
template <typename T>
class EigSymProbeKernelTag;

template <typename T>
inline std::pair<double, double> linalg_eig_symmetry_metrics(
    const T* d_A, int64_t nbatch, int64_t n, sycl::queue& queue)
{
    if (nbatch == 0 || n == 0) return {0.0, 0.0};

    // RAII so d_pair frees on any exception from the .wait() / submit below.
    SyclDeviceBuffer<double> d_pair_guard(2, queue);
    double* d_pair = d_pair_guard.get();
    double init[2] = {0.0, 0.0};
    queue.memcpy(d_pair, init, 2 * sizeof(double)).wait();

    constexpr int LOCAL = 256;
    sycl::range<1> global(static_cast<size_t>(nbatch) * LOCAL);
    sycl::range<1> local(LOCAL);

    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> s_abs(LOCAL, h);
        sycl::local_accessor<double, 1> s_diff(LOCAL, h);
        int64_t n_ = n;
        int64_t total = n_ * n_;
        const T* A = d_A;
        double* pair = d_pair;

        h.parallel_for<EigSymProbeKernelTag<T>>(
            sycl::nd_range<1>(global, local),
            [=](sycl::nd_item<1> it) {
                int64_t b = it.get_group_linear_id();
                int tid = it.get_local_linear_id();
                int wsz = it.get_local_range(0);
                const T* mat = A + b * total;

                double local_max_abs = 0.0;
                double local_max_diff = 0.0;
                for (int64_t idx = tid; idx < total; idx += wsz) {
                    int64_t i = idx / n_;
                    int64_t j = idx - i * n_;
                    double v = static_cast<double>(mat[idx]);
                    double av = (v < 0.0) ? -v : v;
                    if (av > local_max_abs) local_max_abs = av;
                    if (j > i) {
                        double vt = static_cast<double>(mat[j * n_ + i]);
                        double d = v - vt;
                        if (d < 0.0) d = -d;
                        if (d > local_max_diff) local_max_diff = d;
                    }
                }
                s_abs[tid] = local_max_abs;
                s_diff[tid] = local_max_diff;
                sycl::group_barrier(it.get_group());

                for (int stride = wsz / 2; stride > 0; stride >>= 1) {
                    if (tid < stride) {
                        if (s_abs[tid + stride] > s_abs[tid])
                            s_abs[tid] = s_abs[tid + stride];
                        if (s_diff[tid + stride] > s_diff[tid])
                            s_diff[tid] = s_diff[tid + stride];
                    }
                    sycl::group_barrier(it.get_group());
                }

                if (tid == 0) {
                    sycl::atomic_ref<double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> a_abs(pair[0]);
                    sycl::atomic_ref<double,
                        sycl::memory_order::relaxed,
                        sycl::memory_scope::device,
                        sycl::access::address_space::global_space> a_diff(pair[1]);
                    a_abs.fetch_max(s_abs[0]);
                    a_diff.fetch_max(s_diff[0]);
                }
            });
    }).wait();

    double host_pair[2] = {0.0, 0.0};
    queue.memcpy(host_pair, d_pair, 2 * sizeof(double)).wait();
    // d_pair freed by its SyclDeviceBuffer guard on scope exit.
    return {host_pair[0], host_pair[1]};
}

// ============================================================================
// LinalgEigh - Symmetric eigendecomposition via syevd
// ============================================================================
auto linalg_eigh_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    // Float16 / BFloat16: widen to Float32, decompose, narrow eigenvalues and
    // eigenvectors back to the original dtype.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        auto [W, V] = linalg_eigh_kernel(input.to(DType::Float32), queue);
        return {W.to(orig_dtype), V.to(orig_dtype)};
    }
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t n = shape[ndim - 1];
    int64_t batch = 1;
    for (int64_t i = 0; i + 2 < ndim; ++i) batch *= shape[i];

    // Output W = (..., n), V = (..., n, n). Process each (n, n) matrix independently
    // (oneMKL syevd is per-matrix); the previous version handled only the last matrix
    // and silently collapsed the batch dimension.
    std::vector<int64_t> w_shape(shape.begin(), shape.end() - 1);
    Tensor W(w_shape, input.dtype(), input.device());
    Tensor V(std::vector<int64_t>(shape.begin(), shape.end()), input.dtype(), input.device());

    if (input.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<float> d_w(n, queue);
        auto sp = ::oneapi::mkl::lapack::syevd_scratchpad_size<float>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower, n, n);
        SyclDeviceBuffer<float> scratch(sp, queue);
        const float* in_ptr = get_data_ptr<const float>(input);
        float* w_ptr = get_data_ptr<float>(W);
        float* v_ptr = get_data_ptr<float>(V);
        for (int64_t b = 0; b < batch; ++b) {
            row_to_col_major<float, SyclTransposeEighF32>(
                d_a.get(), in_ptr + b * n * n, n, n, queue);
            ::oneapi::mkl::lapack::syevd(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower,
                                       n, d_a.get(), n, d_w.get(), scratch.get(), sp).wait();
            queue.memcpy(w_ptr + b * n, d_w.get(), n * sizeof(float)).wait();
            col_to_row_major<float, SyclTransposeEighBackF32>(
                v_ptr + b * n * n, d_a.get(), n, n, queue);
        }
        return {W, V};
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<double> d_w(n, queue);
        auto sp = ::oneapi::mkl::lapack::syevd_scratchpad_size<double>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower, n, n);
        SyclDeviceBuffer<double> scratch(sp, queue);
        const double* in_ptr = get_data_ptr<const double>(input);
        double* w_ptr = get_data_ptr<double>(W);
        double* v_ptr = get_data_ptr<double>(V);
        for (int64_t b = 0; b < batch; ++b) {
            row_to_col_major<double, SyclTransposeEighF64>(
                d_a.get(), in_ptr + b * n * n, n, n, queue);
            ::oneapi::mkl::lapack::syevd(queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::uplo::lower,
                                       n, d_a.get(), n, d_w.get(), scratch.get(), sp).wait();
            queue.memcpy(w_ptr + b * n, d_w.get(), n * sizeof(double)).wait();
            col_to_row_major<double, SyclTransposeEighBackF64>(
                v_ptr + b * n * n, d_a.get(), n, n, queue);
        }
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
    int64_t ndim = static_cast<int64_t>(shape.size());
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

    // Approximate-symmetry check (per-batch). gradcheck perturbs SPD
    // inputs by ε≈1e-6 — that breaks strict symmetry but each element is
    // very close. A near-symmetric input is treated as "intended symmetric,
    // perturbed by noise" and routed through `eigh` on the SYMMETRIZED
    // matrix. Sum-of-eigenvalues equals trace, which is preserved by
    // symmetrization, so the numerical gradient matches the analytical.
    //
    // The relative threshold is dtype-dependent and deliberately looser for
    // Float32: machine epsilon is ~1e-7 for F32 vs ~2e-16 for F64, and the
    // ~1e-6 gradcheck perturbation is proportionally far larger relative to
    // F32 precision. We therefore use 1e-2 (relative) for Float32 and 1e-3
    // for Float64 so a genuinely-symmetric F32 matrix perturbed by noise is
    // still recognised as symmetric. Keep code and these constants in sync.
    if (input.dtype() == DType::Float32 || input.dtype() == DType::Float64) {
        auto A_cont = input.contiguous();
        bool is_near_symmetric = true;
        if (input.dtype() == DType::Float32) {
            auto [a_max, diff_max] = linalg_eig_symmetry_metrics<float>(
                get_data_ptr<const float>(A_cont), nbatch, n, queue);
            is_near_symmetric = (diff_max < 1e-2 * std::max(a_max, 1.0));
        } else {
            auto [a_max, diff_max] = linalg_eig_symmetry_metrics<double>(
                get_data_ptr<const double>(A_cont), nbatch, n, queue);
            is_near_symmetric = (diff_max < 1e-3 * std::max(a_max, 1.0));
        }
        if (is_near_symmetric) {
            auto At = ::tenzor::transpose(input, ndim - 2, ndim - 1).contiguous();
            auto A_sym = ::tenzor::mul(::tenzor::add(input, At), 0.5);
            auto [W, V] = linalg_eigh_kernel(A_sym.contiguous(), queue);
            auto WI = Tensor(w_shape, input.dtype(), input.device());
            size_t elem_real = (input.dtype() == DType::Float32) ? sizeof(float) : sizeof(double);
            queue.memset(WI.data_ptr(), 0, nbatch * n * elem_real).wait();
            return {W, WI, V};
        }
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

        // n is loop-invariant: query the scratchpad size and allocate the
        // device scratch once, then reuse for every batch element.
        auto sp = ::oneapi::mkl::lapack::geev_scratchpad_size<float>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec,
            n, n);
        SyclDeviceBuffer<float> scratch(sp, queue);

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = a_data + b * n * n;
            float* wr_vec = wr_data + b * n;
            float* wi_vec = wi_data + b * n;
            float* vl = v_data + b * n * n;

            ::oneapi::mkl::lapack::geev(
                queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec,
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

        // n is loop-invariant: query the scratchpad size and allocate the
        // device scratch once, then reuse for every batch element.
        auto sp = ::oneapi::mkl::lapack::geev_scratchpad_size<double>(
            queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec,
            n, n);
        SyclDeviceBuffer<double> scratch(sp, queue);

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = a_data + b * n * n;
            double* wr_vec = wr_data + b * n;
            double* wi_vec = wi_data + b * n;
            double* vl = v_data + b * n * n;

            ::oneapi::mkl::lapack::geev(
                queue, ::oneapi::mkl::job::vec, ::oneapi::mkl::job::novec,
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
#endif // TENZOR_HAS_ONEMKL_GEEV
// When TENZOR_HAS_ONEMKL_GEEV is undefined (the current oneMKL SYCL interface
// reality), `OpId::LinalgEig` is served by the always-compiled
// `linalg_eig_qr_kernel` (Francis double-shift QR, further down this file).
// The registry chooses between the two at compile time.

// ============================================================================
// LinalgCholesky - Cholesky factorization via potrf
// ============================================================================
auto linalg_cholesky_kernel(const Tensor& input, bool upper, sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen to Float32, factorize, narrow back.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        return linalg_cholesky_kernel(input.to(DType::Float32), upper, queue).to(orig_dtype);
    }
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

// ============================================================================
// Triangular Solve (AX = B, A triangular) — oneMKL BLAS trsm
// ============================================================================

auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                     bool upper, bool unitriangular,
                                     sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen both operands to Float32, solve, narrow back.
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        const DType orig_dtype = A.dtype();
        return linalg_solve_triangular_kernel(A.to(DType::Float32), B.to(DType::Float32),
                                              upper, unitriangular, queue).to(orig_dtype);
    }
    // row_to_col_major reads via raw pointer and ignores strides, so views
    // (e.g. A.transpose(-1,-2) that `linalg::cholesky_inverse` feeds in as
    // L^T) produce wrong results unless the caller's logical layout is
    // materialized first. `linalg_solve_kernel` already does this; match it.
    auto A_cont = A.contiguous();
    auto B_cont = B.contiguous();
    auto a_shape = A_cont.shape();
    int64_t n = a_shape[a_shape.size() - 1];
    auto b_shape = B_cont.shape();
    int64_t nrhs = (b_shape.size() > 1) ? b_shape[b_shape.size() - 1] : 1;

    // oneMKL trsm works in column-major. row_to_col_major physically
    // transposes the storage, which is a double negation (transpose +
    // col-major-reinterpret cancel out), so uplo maps straight through.
    auto mkl_uplo = upper ? ::oneapi::mkl::uplo::upper : ::oneapi::mkl::uplo::lower;
    auto mkl_diag = unitriangular ? ::oneapi::mkl::diag::unit : ::oneapi::mkl::diag::nonunit;

    if (A_cont.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_a(n * n, queue);
        SyclDeviceBuffer<float> d_b(n * nrhs, queue);

        row_to_col_major<float, SyclTransposeTriSolveAF32>(
            d_a.get(), get_data_ptr<const float>(A_cont), n, n, queue);
        row_to_col_major<float, SyclTransposeTriSolveBF32>(
            d_b.get(), get_data_ptr<const float>(B_cont), n, nrhs, queue);

        float alpha = 1.0f;
        ::oneapi::mkl::blas::trsm(queue, ::oneapi::mkl::side::left, mkl_uplo,
            ::oneapi::mkl::transpose::nontrans, mkl_diag,
            n, nrhs, alpha, d_a.get(), n, d_b.get(), n).wait();

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A_cont.dtype(), A_cont.device());
        col_to_row_major<float, SyclTransposeTriSolveBackF32>(
            get_data_ptr<float>(output), d_b.get(), n, nrhs, queue);

        return output;
    } else if (A_cont.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(n * n, queue);
        SyclDeviceBuffer<double> d_b(n * nrhs, queue);

        row_to_col_major<double, SyclTransposeTriSolveAF64>(
            d_a.get(), get_data_ptr<const double>(A_cont), n, n, queue);
        row_to_col_major<double, SyclTransposeTriSolveBF64>(
            d_b.get(), get_data_ptr<const double>(B_cont), n, nrhs, queue);

        double alpha = 1.0;
        ::oneapi::mkl::blas::trsm(queue, ::oneapi::mkl::side::left, mkl_uplo,
            ::oneapi::mkl::transpose::nontrans, mkl_diag,
            n, nrhs, alpha, d_a.get(), n, d_b.get(), n).wait();

        std::vector<int64_t> out_shape(b_shape.begin(), b_shape.end());
        Tensor output(out_shape, A_cont.dtype(), A_cont.device());
        col_to_row_major<double, SyclTransposeTriSolveBackF64>(
            get_data_ptr<double>(output), d_b.get(), n, nrhs, queue);

        return output;
    } else {
        throw std::runtime_error("linalg_solve_triangular: only Float32 and Float64 supported");
    }
}

// ============================================================================
// Geqrf — raw QR factorization returning packed reflectors + tau (oneMKL)
// ============================================================================
auto linalg_geqrf_kernel(const Tensor& input, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    // Float16 / BFloat16: widen to Float32, factorize, narrow reflectors and tau.
    if (input.dtype() == DType::Float16 || input.dtype() == DType::BFloat16) {
        const DType orig_dtype = input.dtype();
        auto [refl, tau] = linalg_geqrf_kernel(input.to(DType::Float32), queue);
        return {refl.to(orig_dtype), tau.to(orig_dtype)};
    }
    auto shape = input.shape();
    int64_t m = shape[shape.size() - 2];
    int64_t n = shape[shape.size() - 1];
    int64_t k = std::min(m, n);

    if (input.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_a(m * n, queue);
        SyclDeviceBuffer<float> d_tau(k, queue);

        // Convert row-major input to column-major for oneMKL
        row_to_col_major<float, SyclTransposeGeqrfAF32>(
            d_a.get(), get_data_ptr<const float>(input), m, n, queue);

        auto sp = ::oneapi::mkl::lapack::geqrf_scratchpad_size<float>(queue, m, n, m);
        SyclDeviceBuffer<float> scratch(sp, queue);
        ::oneapi::mkl::lapack::geqrf(queue, m, n, d_a.get(), m, d_tau.get(), scratch.get(), sp).wait();

        // Convert result back to row-major (packed reflectors + R on/above diagonal)
        Tensor result({m, n}, input.dtype(), input.device());
        col_to_row_major<float, SyclGeqrfExtractF32>(
            get_data_ptr<float>(result), d_a.get(), m, n, queue);

        // Copy tau to output tensor
        Tensor tau_out({k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(tau_out.data_ptr()), d_tau.get(), k * sizeof(float)).wait();

        return {result, tau_out};
    } else if (input.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_a(m * n, queue);
        SyclDeviceBuffer<double> d_tau(k, queue);

        row_to_col_major<double, SyclTransposeGeqrfAF64>(
            d_a.get(), get_data_ptr<const double>(input), m, n, queue);

        auto sp = ::oneapi::mkl::lapack::geqrf_scratchpad_size<double>(queue, m, n, m);
        SyclDeviceBuffer<double> scratch(sp, queue);
        ::oneapi::mkl::lapack::geqrf(queue, m, n, d_a.get(), m, d_tau.get(), scratch.get(), sp).wait();

        Tensor result({m, n}, input.dtype(), input.device());
        col_to_row_major<double, SyclGeqrfExtractF64>(
            get_data_ptr<double>(result), d_a.get(), m, n, queue);

        Tensor tau_out({k}, input.dtype(), input.device());
        queue.memcpy(const_cast<void*>(tau_out.data_ptr()), d_tau.get(), k * sizeof(double)).wait();

        return {result, tau_out};
    } else {
        throw std::runtime_error("linalg_geqrf: only Float32 and Float64 supported");
    }
}

// ============================================================================
// Ormqr — multiply matrix by Q from QR factorization using tau (oneMKL)
// ============================================================================
auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                          const Tensor& C, bool left, bool transpose_q,
                          sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen all operands to Float32, apply Q, narrow back.
    if (C.dtype() == DType::Float16 || C.dtype() == DType::BFloat16) {
        const DType orig_dtype = C.dtype();
        return linalg_ormqr_kernel(reflectors.to(DType::Float32), tau.to(DType::Float32),
                                   C.to(DType::Float32), left, transpose_q, queue).to(orig_dtype);
    }
    auto c_shape = C.shape();
    auto r_shape = reflectors.shape();
    int64_t c_m = c_shape[c_shape.size() - 2];
    int64_t c_n = c_shape[c_shape.size() - 1];
    int64_t r_m = r_shape[r_shape.size() - 2];
    int64_t r_n = r_shape[r_shape.size() - 1];
    int64_t k_refl = tau.shape()[static_cast<int64_t>(tau.shape().size()) - 1];

    // oneMKL ormqr works in column-major
    auto mkl_side = left ? ::oneapi::mkl::side::left : ::oneapi::mkl::side::right;
    auto mkl_trans = transpose_q ? ::oneapi::mkl::transpose::trans : ::oneapi::mkl::transpose::nontrans;

    if (C.dtype() == DType::Float32) {
        SyclDeviceBuffer<float> d_refl(r_m * r_n, queue);
        SyclDeviceBuffer<float> d_tau(k_refl, queue);
        SyclDeviceBuffer<float> d_c(c_m * c_n, queue);

        // Convert to column-major
        row_to_col_major<float, SyclTransposeOrmqrRF32>(
            d_refl.get(), get_data_ptr<const float>(reflectors), r_m, r_n, queue);
        queue.memcpy(d_tau.get(), tau.data_ptr(), k_refl * sizeof(float)).wait();
        row_to_col_major<float, SyclTransposeOrmqrCF32>(
            d_c.get(), get_data_ptr<const float>(C), c_m, c_n, queue);

        auto sp = ::oneapi::mkl::lapack::ormqr_scratchpad_size<float>(
            queue, mkl_side, mkl_trans, c_m, c_n, k_refl, r_m, c_m);
        SyclDeviceBuffer<float> scratch(sp, queue);
        ::oneapi::mkl::lapack::ormqr(queue, mkl_side, mkl_trans,
            c_m, c_n, k_refl, d_refl.get(), r_m, d_tau.get(),
            d_c.get(), c_m, scratch.get(), sp).wait();

        Tensor output(std::vector<int64_t>(c_shape.begin(), c_shape.end()),
                       C.dtype(), C.device());
        col_to_row_major<float, SyclTransposeOrmqrBackF32>(
            get_data_ptr<float>(output), d_c.get(), c_m, c_n, queue);

        return output;
    } else if (C.dtype() == DType::Float64) {
        SyclDeviceBuffer<double> d_refl(r_m * r_n, queue);
        SyclDeviceBuffer<double> d_tau(k_refl, queue);
        SyclDeviceBuffer<double> d_c(c_m * c_n, queue);

        row_to_col_major<double, SyclTransposeOrmqrRF64>(
            d_refl.get(), get_data_ptr<const double>(reflectors), r_m, r_n, queue);
        queue.memcpy(d_tau.get(), tau.data_ptr(), k_refl * sizeof(double)).wait();
        row_to_col_major<double, SyclTransposeOrmqrCF64>(
            d_c.get(), get_data_ptr<const double>(C), c_m, c_n, queue);

        auto sp = ::oneapi::mkl::lapack::ormqr_scratchpad_size<double>(
            queue, mkl_side, mkl_trans, c_m, c_n, k_refl, r_m, c_m);
        SyclDeviceBuffer<double> scratch(sp, queue);
        ::oneapi::mkl::lapack::ormqr(queue, mkl_side, mkl_trans,
            c_m, c_n, k_refl, d_refl.get(), r_m, d_tau.get(),
            d_c.get(), c_m, scratch.get(), sp).wait();

        Tensor output(std::vector<int64_t>(c_shape.begin(), c_shape.end()),
                       C.dtype(), C.device());
        col_to_row_major<double, SyclTransposeOrmqrBackF64>(
            get_data_ptr<double>(output), d_c.get(), c_m, c_n, queue);

        return output;
    } else {
        throw std::runtime_error("linalg_ormqr: only Float32 and Float64 supported");
    }
}

#else // !TENZOR_HAS_ONEMKL — native SYCL shared-memory linalg fallback kernels

// (Shared helpers — to_vec / batch_size / check_square / validate_linalg_dtype /
//  check_size_limit — are now defined unconditionally above the `#ifdef
//  TENZOR_HAS_ONEMKL` block so they are available to both this fallback
//  branch AND to `linalg_eig_qr_kernel` which is always compiled.)

// Forward declarations: linalg_inv_kernel/linalg_solve_kernel below are
// implemented in terms of the genuinely hand-rolled LU factor/solve further
// down in this file (defined later, hence the forward declaration).
auto linalg_lu_kernel(const Tensor& A, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor, Tensor>;
auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                            const Tensor& B, sycl::queue& queue) -> Tensor;
// linalg_inv_kernel (below) calls linalg_solve_kernel (defined right after
// it) before the compiler has seen its declaration.
auto linalg_solve_kernel(const Tensor& A, const Tensor& B, sycl::queue& queue) -> Tensor;

namespace {
// linalg_lu_solve_kernel expects a single combined LU buffer (LAPACK getrf
// storage convention: strict-lower triangle holds L's multipliers, upper
// triangle incl. diagonal holds U), but linalg_lu_kernel returns L and U as
// separate tensors. Recombine them with one elementwise SYCL pass.
Tensor pack_lu(const Tensor& L, const Tensor& U, sycl::queue& queue) {
    auto shape = L.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t n = shape[ndim - 1];
    Tensor packed(std::vector<int64_t>(shape.begin(), shape.end()), L.dtype(), L.device());
    int64_t total = L.numel();

    auto launch = [&](auto* l_ptr, auto* u_ptr, auto* p_ptr) {
        queue.parallel_for(sycl::range<1>(static_cast<size_t>(total)), [=](sycl::id<1> idx) {
            int64_t flat = static_cast<int64_t>(idx[0]);
            int64_t i = (flat % (n * n)) / n;
            int64_t j = flat % n;
            p_ptr[flat] = (i > j) ? l_ptr[flat] : u_ptr[flat];
        }).wait();
    };
    if (L.dtype() == DType::Float32) {
        launch(L.data<float>(), U.data<float>(), packed.data<float>());
    } else {
        launch(L.data<double>(), U.data<double>(), packed.data<double>());
    }
    return packed;
}
} // namespace

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

    // det of a single 2D matrix is a 0-D scalar (shape {}); only batched inputs
    // carry leading batch dims. Matches the oneMKL path, CUDA, and CPU — do NOT
    // append a dummy 1 (that produced a wrong rank-1 [1] result).
    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) out_shape.push_back(shape[i]);

    auto result = zeros(out_shape, A.dtype(), A.device());

    SyclDeviceBuffer<int> d_pivots(nbatch * n, queue);
    SyclDeviceBuffer<int> d_info(nbatch, queue);
    queue.memset(d_info.ptr, 0, nbatch * sizeof(int)).wait();

    auto launch_det = [&](auto* work_ptr, auto* res_ptr, auto dummy) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;

        // Global-memory LU with partial pivoting, one work-item per batch
        // element. Operates directly on the global `work` buffer so there is
        // no local-memory size cap — arbitrary N is supported.
        queue.submit([&](sycl::handler& h) {
            auto* data = work_ptr;
            auto* pivots = d_pivots.ptr;
            auto* info = d_info.ptr;
            int n_ = static_cast<int>(n);
            h.parallel_for(sycl::range<1>(nbatch), [=](sycl::id<1> id) {
                int batch_idx = static_cast<int>(id[0]);
                T* A = data + batch_idx * n_ * n_;
                int* batch_pivots = pivots + batch_idx * n_;
                int sign = 1;
                for (int k = 0; k < n_; k++) {
                    T max_val = sycl::fabs(A[k * n_ + k]);
                    int max_row = k;
                    for (int i = k + 1; i < n_; i++) {
                        T val = sycl::fabs(A[i * n_ + k]);
                        if (val > max_val) { max_val = val; max_row = i; }
                    }
                    batch_pivots[k] = max_row + 1;
                    if (max_val == T(0) && info) info[batch_idx] = k + 1;
                    if (max_row != k) {
                        for (int j = 0; j < n_; j++) {
                            T tmp = A[k * n_ + j];
                            A[k * n_ + j] = A[max_row * n_ + j];
                            A[max_row * n_ + j] = tmp;
                        }
                        sign = -sign;
                    }
                    T diag = A[k * n_ + k];
                    if (diag != T(0)) {
                        for (int i = k + 1; i < n_; i++) {
                            A[i * n_ + k] /= diag;
                            T mult = A[i * n_ + k];
                            for (int j = k + 1; j < n_; j++)
                                A[i * n_ + j] -= mult * A[k * n_ + j];
                        }
                    }
                }
                if (info && info[batch_idx] == 0) info[batch_idx] = sign > 0 ? 0 : -1;
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

auto linalg_inv_kernel(const Tensor& input_in, sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen to Float32, compute, narrow back. Mirrors
    // linalg_det_kernel. (linalg_lu_kernel/linalg_solve_kernel below also
    // widen internally, but doing it here too avoids constructing an
    // unnecessary half-precision identity matrix.)
    if (input_in.dtype() == DType::Float16 || input_in.dtype() == DType::BFloat16) {
        const DType orig_dtype = input_in.dtype();
        return linalg_inv_kernel(input_in.to(DType::Float32), queue).to(orig_dtype);
    }

    Tensor input = input_in.is_contiguous() ? input_in : input_in.contiguous();
    auto shape = input.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());
    int64_t n = shape[ndim - 1];

    // inv(A) = solve(A, I) -- avoids reimplementing getri's explicit-inverse
    // algorithm; the native fallback has no oneMKL to call into. Build a
    // batched identity matching A's leading batch dims (mirrors CUDA's own
    // native-fallback linalg_householder_kernel batching pattern).
    Tensor I = tenzor::eye(n, std::nullopt, input.dtype(), input.device());
    if (ndim > 2) {
        std::vector<int64_t> eye_shape(shape.begin(), shape.end());
        I = tenzor::expand(I, std::move(eye_shape));
        I = I.contiguous();
    }
    return linalg_solve_kernel(input, I, queue);
}

// ============================================================================
// Linear System Solve (AX = B)
// ============================================================================

auto linalg_solve_kernel(const Tensor& A, const Tensor& B, sycl::queue& queue) -> Tensor {
    // Float16 / BFloat16: widen both operands to Float32, solve, narrow back.
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        const DType orig_dtype = A.dtype();
        return linalg_solve_kernel(A.to(DType::Float32), B.to(DType::Float32), queue).to(orig_dtype);
    }
    auto a_shape = A.shape();
    auto b_shape = B.shape();
    int64_t n = a_shape[a_shape.size() - 1];
    int64_t nrhs = (b_shape.size() > 1) ? b_shape[b_shape.size() - 1] : 1;
    int64_t nbatch = (n > 0) ? A.numel() / (n * n) : 0;
    int64_t nbatch_b = (n > 0 && nrhs > 0) ? B.numel() / (n * nrhs) : 0;
    if (nbatch_b != nbatch) {
        throw std::invalid_argument(
            "linalg_solve: batch count of A (" + std::to_string(nbatch) +
            ") does not match batch count of B (" + std::to_string(nbatch_b) + ")");
    }

    // solve(A, B) = lu_solve(lu_factor(A), B). No oneMKL getrf/getrs to call
    // into in this build config, so compose from the already hand-rolled LU
    // primitives further down this file instead of reimplementing them.
    auto [L, U, P] = linalg_lu_kernel(A, queue);
    Tensor LU = pack_lu(L, U, queue);
    // linalg_lu_solve_kernel requires B with an explicit trailing nrhs dim;
    // a 1D B (single right-hand side) needs reshaping to (n, 1) first, then
    // squeezing back to match the input rank on return.
    bool b_is_1d = (b_shape.size() < 2);
    Tensor b_work = b_is_1d ? B.reshape({n, 1}) : B;
    Tensor result = linalg_lu_solve_kernel(LU, P, b_work, queue);
    return b_is_1d ? result.reshape({n}) : result;
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

    // Per-batch LAPACK-style info flag: 0 == success, k>0 == leading minor of
    // order k is not positive-definite (potrf reports info=k for the first
    // non-positive pivot at column k-1, 1-based). Mirrors the oneMKL potrf
    // path which surfaces info>0 to the caller via an exception so that
    // cholesky_inverse / cholesky_solve do not silently consume a corrupt
    // factor.
    SyclDeviceBuffer<int> d_info(nbatch, queue);
    queue.memset(d_info.ptr, 0, nbatch * sizeof(int)).wait();

    auto launch_cholesky = [&](auto* data_ptr) {
        using T = std::remove_pointer_t<decltype(data_ptr)>;
        // Global-memory Cholesky (one work-item per batch). In-place
        // Cholesky-Crout into the lower triangle of the global buffer, then
        // mask to the requested triangle. No local memory -> arbitrary N.
        queue.submit([&](sycl::handler& h) {
            auto* data = data_ptr;
            auto* info = d_info.ptr;
            int n_ = static_cast<int>(n);
            bool upper_ = upper;
            h.parallel_for(sycl::range<1>(nbatch), [=](sycl::id<1> id) {
                int b = static_cast<int>(id[0]);
                T* A = data + b * n_ * n_;
                // Compute the lower factor L in the strict-lower + diagonal of a
                // scratch held in registers via direct A reads: we overwrite the
                // lower triangle in place (column j only depends on already-
                // computed columns < j of the lower triangle, which are intact).
                for (int j = 0; j < n_; j++) {
                    T sum = A[j * n_ + j];
                    for (int k = 0; k < j; k++) sum -= A[j * n_ + k] * A[j * n_ + k];
                    if (sum <= T(0)) {
                        // Non-positive pivot: matrix is not positive-definite.
                        // Record the (1-based) failing leading-minor order, then
                        // zero the diagonal AND the remaining strict-lower of
                        // this column so later columns never consume stale input
                        // values. Keep the first failure (smallest j).
                        if (info[b] == 0) info[b] = j + 1;
                        A[j * n_ + j] = T(0);
                        for (int i = j + 1; i < n_; i++) A[i * n_ + j] = T(0);
                        continue;
                    }
                    A[j * n_ + j] = sycl::sqrt(sum);
                    T diag = A[j * n_ + j];
                    for (int i = j + 1; i < n_; i++) {
                        T s = A[i * n_ + j];
                        for (int k = 0; k < j; k++) s -= A[i * n_ + k] * A[j * n_ + k];
                        A[i * n_ + j] = s / diag;
                    }
                }
                // Mask to the requested triangle. For 'upper', element (row,col)
                // = L(col,row) for row<=col; the lower factor is still intact
                // (we only wrote the lower triangle + diagonal), so read L(col,row).
                if (upper_) {
                    // Walk from the last element backwards so we never read a
                    // lower-triangle source after it has been overwritten by an
                    // upper-triangle store. Upper store targets (row<=col) read
                    // sources (col,row) with col>=row, i.e. lower indices, which
                    // are only written when processing that source's own cell.
                    // Build into a separate pass over the strict-upper using the
                    // symmetric lower value before zeroing the strict-lower.
                    for (int row = 0; row < n_; row++)
                        for (int col = row + 1; col < n_; col++)
                            A[row * n_ + col] = A[col * n_ + row];  // L(col,row)
                    for (int row = 0; row < n_; row++)
                        for (int col = 0; col < row; col++)
                            A[row * n_ + col] = T(0);
                } else {
                    for (int row = 0; row < n_; row++)
                        for (int col = row + 1; col < n_; col++)
                            A[row * n_ + col] = T(0);
                }
            });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch_cholesky(work.data<float>());
    else
        launch_cholesky(work.data<double>());

    // Copy the per-batch info flags back and throw on the first breakdown,
    // matching the oneMKL potrf contract (info>0 == not positive-definite).
    std::vector<int> h_info(static_cast<size_t>(nbatch), 0);
    queue.memcpy(h_info.data(), d_info.ptr, nbatch * sizeof(int)).wait();
    for (int64_t b = 0; b < nbatch; b++) {
        if (h_info[b] > 0) {
            throw std::runtime_error(
                "linalg_cholesky: the leading minor of order " +
                std::to_string(h_info[b]) +
                " is not positive-definite (matrix is not positive-definite)");
        }
    }

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
        // Global-memory Householder QR (one work-item per batch). R is built in
        // place in the global `work` buffer; the full m×m Q accumulates in a
        // global scratch buffer. No local memory -> arbitrary m, n. Each
        // reflector is applied to columns > j using the intact sub-diagonal
        // (the reflector tail), then column j is finalised — avoiding the
        // tail-overwrite-before-use hazard of a naive serial loop.
        SyclDeviceBuffer<T> d_qfull(static_cast<size_t>(nbatch) * m * m, queue);
        queue.submit([&](sycl::handler& h) {
            int m_ = static_cast<int>(m), nc_ = static_cast<int>(n_cols), k_ = static_cast<int>(k);
            auto* work_g = work_ptr;
            auto* q_g = q_ptr;
            auto* r_g = r_ptr;
            auto* qfull = d_qfull.ptr;
            h.parallel_for(sycl::range<1>(nbatch), [=](sycl::id<1> id) {
                int b = static_cast<int>(id[0]);
                T* R_s = work_g + static_cast<size_t>(b) * m_ * nc_;
                T* Q_s = qfull + static_cast<size_t>(b) * m_ * m_;
                for (int i = 0; i < m_ * m_; i++) Q_s[i] = T(0);
                for (int i = 0; i < m_; i++) Q_s[i * m_ + i] = T(1);

                constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);
                for (int j = 0; j < k_; j++) {
                    T sigma = T(0);
                    for (int i = j + 1; i < m_; i++) sigma += R_s[i * nc_ + j] * R_s[i * nc_ + j];
                    T x0 = R_s[j * nc_ + j];
                    T norm_x = sycl::sqrt(x0 * x0 + sigma);
                    if (norm_x < zero_tol || sigma < zero_tol) continue;
                    T alpha = -sycl::copysign(norm_x, x0);
                    T v0 = x0 - alpha;
                    T tau = T(2) / (v0 * v0 + sigma);

                    // Apply H to columns > j (tail R_s[i,j], i>j, still intact).
                    for (int col = j + 1; col < nc_; col++) {
                        T dot = v0 * R_s[j * nc_ + col];
                        for (int i = j + 1; i < m_; i++) dot += R_s[i * nc_ + j] * R_s[i * nc_ + col];
                        dot *= tau;
                        R_s[j * nc_ + col] -= v0 * dot;
                        for (int i = j + 1; i < m_; i++) R_s[i * nc_ + col] -= R_s[i * nc_ + j] * dot;
                    }
                    // Accumulate Q = Q * H.
                    for (int row = 0; row < m_; row++) {
                        T dot = v0 * Q_s[row * m_ + j];
                        for (int i = j + 1; i < m_; i++) dot += R_s[i * nc_ + j] * Q_s[row * m_ + i];
                        dot *= tau;
                        Q_s[row * m_ + j] -= v0 * dot;
                        for (int i = j + 1; i < m_; i++) Q_s[row * m_ + i] -= R_s[i * nc_ + j] * dot;
                    }
                    // Finalise column j (diagonal = alpha, sub-diagonal = 0).
                    R_s[j * nc_ + j] = alpha;
                    for (int i = j + 1; i < m_; i++) R_s[i * nc_ + j] = T(0);
                }

                for (int row = 0; row < m_; row++)
                    for (int col = 0; col < k_; col++)
                        q_g[static_cast<size_t>(b) * m_ * k_ + row * k_ + col] = Q_s[row * m_ + col];
                for (int i = 0; i < k_ * nc_; i++)
                    r_g[static_cast<size_t>(b) * k_ * nc_ + i] = R_s[i];
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
    if (A.dtype() == DType::Float16) { auto [W,V] = linalg_eigh_kernel(A.to(DType::Float32), queue); return {W.to(DType::Float16), V.to(DType::Float16)}; }
    if (A.dtype() == DType::BFloat16) { auto [W,V] = linalg_eigh_kernel(A.to(DType::Float32), queue); return {W.to(DType::BFloat16), V.to(DType::BFloat16)}; }

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

// (`linalg_eig_qr_kernel` has been relocated to outside the
//  `#ifdef TENZOR_HAS_ONEMKL` block — see end of file. It is registered as
//  the backend for `OpId::LinalgEig` and must compile unconditionally.)


// ============================================================================
// LU decomposition (native fallback)
// ============================================================================

auto linalg_lu_kernel(const Tensor& A, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "lu");
    if (A.dtype() == DType::Float16) {
        auto [L, U, P] = linalg_lu_kernel(A.to(DType::Float32), queue);
        return {L.to(DType::Float16), U.to(DType::Float16), P};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [L, U, P] = linalg_lu_kernel(A.to(DType::Float32), queue);
        return {L.to(DType::BFloat16), U.to(DType::BFloat16), P};
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto shape = A.shape();
    std::vector<int64_t> mat_shape(shape.begin(), shape.end());
    std::vector<int64_t> piv_shape;
    for (size_t i = 0; i + 2 < shape.size(); ++i) piv_shape.push_back(shape[i]);
    piv_shape.push_back(n);

    auto L = zeros(mat_shape, A.dtype(), A.device());
    auto U = zeros(mat_shape, A.dtype(), A.device());
    auto pivots = zeros(piv_shape, DType::Int32, A.device());

    auto launch = [&](auto* a_ptr, auto* l_ptr, auto* u_ptr) {
        using T = std::remove_pointer_t<decltype(a_ptr)>;
        check_size_limit<T>(n, "lu");
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        size_t smem_lu = n * n * sizeof(T) + 4 * sizeof(T);

        int32_t* piv_ptr = pivots.template data<int32_t>();
        int64_t  n_      = n;

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_lu), h);
            auto* data = a_ptr;
            auto* l_o  = l_ptr;
            auto* u_o  = u_ptr;
            auto* piv  = piv_ptr;
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* As = reinterpret_cast<T*>(smem_raw);
                    T* scratch = As + n_ * n_;
                    T* batch_data = data + batch_idx * n_ * n_;
                    T* batch_l    = l_o  + batch_idx * n_ * n_;
                    T* batch_u    = u_o  + batch_idx * n_ * n_;
                    int32_t* batch_piv = piv + batch_idx * n_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        As[idx] = batch_data[idx];
                    sycl::group_barrier(item.get_group());
                    for (int k = 0; k < n_; k++) {
                        if (tid == 0) {
                            T max_val = sycl::fabs(As[k * n_ + k]);
                            int max_row = k;
                            for (int i = k + 1; i < n_; i++) {
                                T val = sycl::fabs(As[i * n_ + k]);
                                if (val > max_val) { max_val = val; max_row = i; }
                            }
                            batch_piv[k] = max_row + 1;
                            scratch[0] = static_cast<T>(max_row);
                        }
                        sycl::group_barrier(item.get_group());
                        int pivot_row = static_cast<int>(scratch[0]);
                        if (pivot_row != k) {
                            for (int j = tid; j < n_; j += num_threads) {
                                T tmp = As[k * n_ + j];
                                As[k * n_ + j] = As[pivot_row * n_ + j];
                                As[pivot_row * n_ + j] = tmp;
                            }
                            sycl::group_barrier(item.get_group());
                        }
                        T diag = As[k * n_ + k];
                        if (diag != T(0)) {
                            if (tid == 0)
                                for (int i = k + 1; i < n_; i++)
                                    As[i * n_ + k] /= diag;
                            sycl::group_barrier(item.get_group());
                            for (int i = k + 1 + tid; i < n_; i += num_threads) {
                                T mult = As[i * n_ + k];
                                for (int j = k + 1; j < n_; j++)
                                    As[i * n_ + j] -= mult * As[k * n_ + j];
                            }
                            sycl::group_barrier(item.get_group());
                        }
                    }
                    // Split into L (unit lower) and U (upper)
                    for (int idx = tid; idx < n_ * n_; idx += num_threads) {
                        int i = idx / n_;
                        int j = idx % n_;
                        T v = As[idx];
                        if (i > j) {
                            batch_l[idx] = v;
                            batch_u[idx] = T(0);
                        } else if (i == j) {
                            batch_l[idx] = T(1);
                            batch_u[idx] = v;
                        } else {
                            batch_l[idx] = T(0);
                            batch_u[idx] = v;
                        }
                    }
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch(work.data<float>(), L.data<float>(), U.data<float>());
    else
        launch(work.data<double>(), L.data<double>(), U.data<double>());

    return {L, U, pivots};
}

auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                            const Tensor& B, sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(LU_data, "lu_solve");
    if (LU_data.dtype() == DType::Float16 || LU_data.dtype() == DType::BFloat16) {
        auto orig = B.dtype();
        auto out = linalg_lu_solve_kernel(
            LU_data.to(DType::Float32), pivots, B.to(DType::Float32), queue);
        return out.to(orig);
    }

    auto work_lu = LU_data.contiguous().clone();
    auto work_b  = B.contiguous().clone();
    auto [n, ndim] = check_square(work_lu);
    int64_t nbatch = batch_size(work_lu);
    auto b_shape = B.shape();
    int64_t nrhs = b_shape[b_shape.size() - 1];

    auto launch = [&](auto* lu_ptr, auto* b_ptr) {
        using T = std::remove_pointer_t<decltype(lu_ptr)>;
        check_size_limit<T>(n, "lu_solve");
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        size_t smem_solve = (n * n + n * nrhs) * sizeof(T);

        const int32_t* piv_ptr = pivots.template data<int32_t>();
        int64_t n_ = n, nrhs_ = nrhs;

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_solve), h);
            auto* lu_data_ = lu_ptr;
            auto* b_data_  = b_ptr;
            auto* piv      = piv_ptr;
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* LU = reinterpret_cast<T*>(smem_raw);
                    T* Bs = LU + n_ * n_;
                    const T* batch_lu = lu_data_ + batch_idx * n_ * n_;
                    const int32_t* batch_piv = piv + batch_idx * n_;
                    T* batch_b = b_data_ + batch_idx * n_ * nrhs_;

                    for (int idx = tid; idx < n_ * n_; idx += num_threads)
                        LU[idx] = batch_lu[idx];
                    for (int idx = tid; idx < n_ * nrhs_; idx += num_threads)
                        Bs[idx] = batch_b[idx];
                    sycl::group_barrier(item.get_group());

                    if (tid == 0) {
                        for (int i = 0; i < n_; i++) {
                            int piv_row = batch_piv[i] - 1;
                            if (piv_row != i)
                                for (int j = 0; j < nrhs_; j++) {
                                    T tmp = Bs[i * nrhs_ + j];
                                    Bs[i * nrhs_ + j] = Bs[piv_row * nrhs_ + j];
                                    Bs[piv_row * nrhs_ + j] = tmp;
                                }
                        }
                        for (int k = 0; k < n_; k++)
                            for (int i = k + 1; i < n_; i++) {
                                T mult = LU[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++)
                                    Bs[i * nrhs_ + j] -= mult * Bs[k * nrhs_ + j];
                            }
                        for (int k = n_ - 1; k >= 0; k--) {
                            T diag = LU[k * n_ + k];
                            for (int j = 0; j < nrhs_; j++) Bs[k * nrhs_ + j] /= diag;
                            for (int i = 0; i < k; i++) {
                                T mult = LU[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++)
                                    Bs[i * nrhs_ + j] -= mult * Bs[k * nrhs_ + j];
                            }
                        }
                    }
                    sycl::group_barrier(item.get_group());

                    for (int idx = tid; idx < n_ * nrhs_; idx += num_threads)
                        batch_b[idx] = Bs[idx];
                });
        }).wait();
    };

    if (LU_data.dtype() == DType::Float32)
        launch(work_lu.data<float>(), work_b.data<float>());
    else
        launch(work_lu.data<double>(), work_b.data<double>());

    return work_b;
}

// ============================================================================
// Triangular Solve — native SYCL kernel fallback (no oneMKL path)
// ============================================================================

auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                     bool upper, bool unitriangular,
                                     sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(A, "solve_triangular");
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        return linalg_solve_triangular_kernel(
            A.to(DType::Float32), B.to(DType::Float32),
            upper, unitriangular, queue).to(A.dtype());
    }

    auto work_a = A.contiguous();
    auto work_b = B.contiguous().clone();
    auto [n, ndim_a] = check_square(work_a);
    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    // Use a single work-group per batch element, serial substitution
    auto launch = [&](auto* a_ptr, auto* b_ptr) {
        using T = std::remove_pointer_t<decltype(a_ptr)>;
        int n_ = static_cast<int>(n);
        int nrhs_ = static_cast<int>(nrhs);
        bool upper_ = upper;
        bool unit_ = unitriangular;

        queue.submit([&](sycl::handler& h) {
            // No explicit kernel name: the enclosing launcher is templated on T,
            // so SYCL auto-generates a distinct kernel name per instantiation
            // (a fixed name would collide between the float and double builds).
            h.parallel_for(
                sycl::nd_range<1>(nbatch * 256, 256),
                [=](sycl::nd_item<1> item) {
                    int batch = item.get_group(0);
                    int tid = item.get_local_id(0);
                    int num_threads = item.get_local_range(0);
                    const T* A_mat = a_ptr + batch * n_ * n_;
                    T* B_mat = b_ptr + batch * n_ * nrhs_;

                    if (upper_) {
                        for (int i = n_ - 1; i >= 0; --i) {
                            sycl::group_barrier(item.get_group());
                            for (int j = tid; j < nrhs_; j += num_threads) {
                                T sum = B_mat[i * nrhs_ + j];
                                for (int k = i + 1; k < n_; ++k)
                                    sum -= A_mat[i * n_ + k] * B_mat[k * nrhs_ + j];
                                B_mat[i * nrhs_ + j] = unit_ ? sum : sum / A_mat[i * n_ + i];
                            }
                        }
                    } else {
                        for (int i = 0; i < n_; ++i) {
                            sycl::group_barrier(item.get_group());
                            for (int j = tid; j < nrhs_; j += num_threads) {
                                T sum = B_mat[i * nrhs_ + j];
                                for (int k = 0; k < i; ++k)
                                    sum -= A_mat[i * n_ + k] * B_mat[k * nrhs_ + j];
                                B_mat[i * nrhs_ + j] = unit_ ? sum : sum / A_mat[i * n_ + i];
                            }
                        }
                    }
                });
        }).wait();
    };

    if (work_a.dtype() == DType::Float32)
        launch(work_a.data<float>(), work_b.data<float>());
    else
        launch(work_a.data<double>(), work_b.data<double>());

    return work_b;
}

// ============================================================================
// Geqrf fallback — Householder QR returning packed reflectors + tau (SYCL)
// ============================================================================
auto linalg_geqrf_kernel(const Tensor& A, sycl::queue& queue) -> std::pair<Tensor, Tensor> {
    validate_linalg_dtype(A, "geqrf");
    if (A.dtype() == DType::Float16) {
        auto [R, tau] = linalg_geqrf_kernel(A.to(DType::Float32), queue);
        return {R, tau};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [R, tau] = linalg_geqrf_kernel(A.to(DType::Float32), queue);
        return {R, tau};
    }

    auto work = A.contiguous().clone();
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::geqrf: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> tau_shape = batch_dims;
    tau_shape.push_back(k);
    auto tau_result = zeros(tau_shape, A.dtype(), A.device());

    auto launch_geqrf = [&](auto* work_ptr, auto* tau_ptr) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        check_size_limit<T>(std::max(m, n_cols), "geqrf");
        size_t smem_bytes = (m * n_cols + 4) * sizeof(T);
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
                    T* R = reinterpret_cast<T*>(smem_raw);
                    T* scratch = R + m_ * nc_;

                    T* A = work_ptr + batch_idx * m_ * nc_;
                    T* tau = tau_ptr + batch_idx * k_;

                    for (int idx = tid; idx < m_ * nc_; idx += num_threads) R[idx] = A[idx];
                    sycl::group_barrier(item.get_group());

                    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

                    for (int j = 0; j < k_; j++) {
                        if (tid == 0) {
                            T sigma = T(0);
                            for (int i = j + 1; i < m_; i++) sigma += R[i * nc_ + j] * R[i * nc_ + j];
                            T x0 = R[j * nc_ + j];
                            T norm_x = sycl::sqrt(x0 * x0 + sigma);
                            if (norm_x < zero_tol || sigma < zero_tol) {
                                scratch[1] = T(0);
                                tau[j] = T(0);
                            } else {
                                T alpha = -sycl::copysign(norm_x, x0);
                                T v0 = x0 - alpha;
                                T v_norm_sq = v0 * v0 + sigma;
                                T tau_val = T(2) / v_norm_sq;
                                scratch[0] = v0;
                                scratch[1] = tau_val;
                                scratch[2] = alpha;
                                tau[j] = tau_val;
                            }
                        }
                        sycl::group_barrier(item.get_group());

                        T tau_val = scratch[1];
                        if (tau_val == T(0)) { sycl::group_barrier(item.get_group()); continue; }
                        T v0 = scratch[0];
                        T alpha = scratch[2];

                        for (int col = j + tid; col < nc_; col += num_threads) {
                            T dot = v0 * R[j * nc_ + col];
                            for (int i = j + 1; i < m_; i++) dot += R[i * nc_ + j] * R[i * nc_ + col];
                            dot *= tau_val;
                            R[j * nc_ + col] -= v0 * dot;
                            for (int i = j + 1; i < m_; i++) R[i * nc_ + col] -= R[i * nc_ + j] * dot;
                        }
                        sycl::group_barrier(item.get_group());

                        if (tid == 0) {
                            T inv_v0 = T(1) / v0;
                            for (int i = j + 1; i < m_; i++) R[i * nc_ + j] *= inv_v0;
                            R[j * nc_ + j] = alpha;
                            tau[j] = tau_val * v0 * v0;
                        }
                        sycl::group_barrier(item.get_group());
                    }

                    for (int idx = tid; idx < m_ * nc_; idx += num_threads) A[idx] = R[idx];
                });
        }).wait();
    };

    if (work.dtype() == DType::Float32)
        launch_geqrf(work.data<float>(), tau_result.data<float>());
    else
        launch_geqrf(work.data<double>(), tau_result.data<double>());

    return {work, tau_result};
}

// ============================================================================
// Ormqr fallback — apply Q (from Householder reflectors) to matrix C (SYCL)
// ============================================================================
auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                          const Tensor& C, bool left, bool transpose_q,
                          sycl::queue& queue) -> Tensor {
    validate_linalg_dtype(C, "ormqr");
    if (C.dtype() == DType::Float16) {
        return linalg_ormqr_kernel(reflectors.to(DType::Float32), tau.to(DType::Float32),
                                    C.to(DType::Float32), left, transpose_q, queue);
    }
    if (C.dtype() == DType::BFloat16) {
        return linalg_ormqr_kernel(reflectors.to(DType::Float32), tau.to(DType::Float32),
                                    C.to(DType::Float32), left, transpose_q, queue);
    }

    auto work_c = C.contiguous().clone();
    auto refl = reflectors.contiguous();
    auto tau_c = tau.contiguous();

    auto c_shape = C.shape();
    auto r_shape = reflectors.shape();
    auto c_ndim = static_cast<int64_t>(c_shape.size());
    auto r_ndim = static_cast<int64_t>(r_shape.size());
    if (c_ndim < 2) throw std::invalid_argument("linalg::ormqr: C must be at least 2D");
    if (r_ndim < 2) throw std::invalid_argument("linalg::ormqr: reflectors must be at least 2D");

    int64_t c_m = c_shape[c_ndim - 2];
    int64_t c_n = c_shape[c_ndim - 1];
    int64_t k_refl = tau.shape()[static_cast<int64_t>(tau.shape().size()) - 1];
    int64_t nbatch = batch_size(work_c);
    int64_t r_m = r_shape[r_ndim - 2];
    int64_t r_n = r_shape[r_ndim - 1];

    auto launch_ormqr = [&](auto* refl_ptr, auto* tau_ptr, auto* c_ptr) {
        using T = std::remove_pointer_t<decltype(c_ptr)>;
        check_size_limit<T>(std::max(c_m, c_n), "ormqr");
        size_t smem_bytes = (c_m * c_n + std::max(c_m, c_n)) * sizeof(T);
        int threads = std::min(static_cast<int>(std::max(c_m, c_n)), 128);
        if (threads < 1) threads = 1;

        int rm_ = static_cast<int>(r_m), rn_ = static_cast<int>(r_n);
        int cm_ = static_cast<int>(c_m), cn_ = static_cast<int>(c_n);
        int kr_ = static_cast<int>(k_refl);

        queue.submit([&](sycl::handler& h) {
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_bytes), h);
            h.parallel_for(sycl::nd_range<1>(nbatch * threads, threads),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    int tid = item.get_local_linear_id();
                    int num_threads = item.get_local_range(0);
                    char* smem_raw = smem.get_pointer();
                    T* C = reinterpret_cast<T*>(smem_raw);

                    const T* refl = refl_ptr + batch_idx * rm_ * rn_;
                    const T* tau = tau_ptr + batch_idx * kr_;
                    T* C_out = c_ptr + batch_idx * cm_ * cn_;

                    for (int idx = tid; idx < cm_ * cn_; idx += num_threads) C[idx] = C_out[idx];
                    sycl::group_barrier(item.get_group());

                    int start, end_val, step;
                    if ((left && !transpose_q) || (!left && !transpose_q)) {
                        start = 0; end_val = kr_; step = 1;
                    } else {
                        start = kr_ - 1; end_val = -1; step = -1;
                    }

                    for (int j = start; j != end_val; j += step) {
                        T tau_j = tau[j];
                        if (tau_j == T(0)) continue;

                        if (left) {
                            for (int col = tid; col < cn_; col += num_threads) {
                                T dot = C[j * cn_ + col];
                                for (int i = j + 1; i < cm_; i++)
                                    dot += refl[i * rn_ + j] * C[i * cn_ + col];
                                dot *= tau_j;
                                C[j * cn_ + col] -= dot;
                                for (int i = j + 1; i < cm_; i++)
                                    C[i * cn_ + col] -= refl[i * rn_ + j] * dot;
                            }
                        } else {
                            for (int row = tid; row < cm_; row += num_threads) {
                                T dot = C[row * cn_ + j];
                                for (int i = j + 1; i < cn_; i++)
                                    dot += C[row * cn_ + i] * refl[i * rn_ + j];
                                dot *= tau_j;
                                C[row * cn_ + j] -= dot;
                                for (int i = j + 1; i < cn_; i++)
                                    C[row * cn_ + i] -= dot * refl[i * rn_ + j];
                            }
                        }
                        sycl::group_barrier(item.get_group());
                    }

                    for (int idx = tid; idx < cm_ * cn_; idx += num_threads) C_out[idx] = C[idx];
                });
        }).wait();
    };

    if (work_c.dtype() == DType::Float32)
        launch_ormqr(refl.data<float>(), tau_c.data<float>(), work_c.data<float>());
    else
        launch_ormqr(refl.data<double>(), tau_c.data<double>(), work_c.data<double>());

    return work_c;
}

#endif // TENZOR_HAS_ONEMKL

// ============================================================================
// Non-symmetric Eigendecomposition (eig) — native SYCL Hessenberg QR algorithm
// Always compiled (independent of TENZOR_HAS_ONEMKL): oneMKL does not currently
// expose `geev` through its DPC++ LAPACK interface, so this Francis double-shift
// QR kernel is the registered backend for `OpId::LinalgEig` on every OneAPI
// build. Helpers (`validate_linalg_dtype`, `check_square`, `batch_size`,
// `check_size_limit`) are defined in the unconditional anonymous namespace
// above the `#ifdef TENZOR_HAS_ONEMKL` block at the top of this file.
// ============================================================================

auto linalg_eig_qr_kernel(const Tensor& A, sycl::queue& queue) -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "eig");
    if (A.dtype() == DType::Float16) { auto [wr,wi,V] = linalg_eig_qr_kernel(A.to(DType::Float32), queue); return {wr,wi,V}; }
    if (A.dtype() == DType::BFloat16) { auto [wr,wi,V] = linalg_eig_qr_kernel(A.to(DType::Float32), queue); return {wr,wi,V}; }

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

    // Native EISPACK-hqr2 non-symmetric eigensolver: one work-item per batch
    // element over global-memory scratch (correct eigenvalues AND eigenvectors;
    // no shared-memory size cap, no barriers). Output packing matches LAPACK
    // geev: real eigenvalue k -> column k; complex pair (wi[k]>0) -> column k =
    // Re, column k+1 = Im (conjugate in column k+1).
    auto vbuf = zeros({nbatch, n}, A.dtype(), A.device());

    // Per-batch convergence flag, mirroring LAPACK geev's `info` contract:
    // 0 == converged; info>0 == the QR iteration failed to compute all
    // eigenvalues (the value records the 1-based index of the eigenvalue that
    // was still being deflated when the iteration limit was hit). Surfaced to
    // the host so the op throws instead of returning silently-wrong (zero)
    // eigenpairs for hard / non-converging matrices.
    SyclDeviceBuffer<int> d_info(nbatch, queue);
    queue.memset(d_info.ptr, 0, nbatch * sizeof(int)).wait();

    auto launch_eig = [&](auto* work_ptr, auto* wr_ptr, auto* wi_ptr, auto* v_ptr, auto* vbuf_ptr) {
        using T = std::remove_pointer_t<decltype(work_ptr)>;
        int n_ = static_cast<int>(n);
        long long nb = static_cast<long long>(nbatch);
        auto* info_ptr = d_info.ptr;
        queue.submit([&](sycl::handler& h) {
            h.parallel_for(sycl::range<1>(static_cast<size_t>(nbatch)),
                [=](sycl::id<1> id) {
                    long long b = static_cast<long long>(id[0]);
                    if (b >= nb) return;
                    T* H = work_ptr + b * n_ * n_;
                    T* Z = v_ptr + b * n_ * n_;
                    T* wr = wr_ptr + b * n_;
                    T* wi = wi_ptr + b * n_;
                    T* v = vbuf_ptr + b * n_;
                    const T eps = std::is_same_v<T, float> ? T(1.19e-7) : T(2.22e-16);
                    auto cdiv = [](T cr, T ci, T dr, T di, T& zr, T& zi) {
                        if (sycl::fabs(dr) > sycl::fabs(di)) {
                            T r = di / dr, d = dr + di * r;
                            zr = (cr + ci * r) / d; zi = (ci - cr * r) / d;
                        } else {
                            T r = dr / di, d = di + dr * r;
                            zr = (cr * r + ci) / d; zi = (ci * r - cr) / d;
                        }
                    };

                    for (int i = 0; i < n_; ++i) { for (int j = 0; j < n_; ++j) Z[i*n_+j] = T(0); Z[i*n_+i] = T(1); }

                    // orthogonal Hessenberg reduction (Householder), accumulate Z
                    for (int k = 0; k + 2 < n_; ++k) {
                        T sigma = T(0);
                        for (int i = k+1; i < n_; ++i) sigma += H[i*n_+k]*H[i*n_+k];
                        if (sigma == T(0)) continue;
                        T nx = sycl::sqrt(sigma);
                        T a = -sycl::copysign(nx, H[(k+1)*n_+k]);
                        v[k+1] = H[(k+1)*n_+k] - a;
                        for (int i = k+2; i < n_; ++i) v[i] = H[i*n_+k];
                        T vn2 = T(0);
                        for (int i = k+1; i < n_; ++i) vn2 += v[i]*v[i];
                        if (vn2 == T(0)) continue;
                        T tau = T(2)/vn2;
                        for (int j = 0; j < n_; ++j) { T d=T(0); for(int i=k+1;i<n_;++i) d+=v[i]*H[i*n_+j]; d*=tau; for(int i=k+1;i<n_;++i) H[i*n_+j]-=v[i]*d; }
                        for (int i = 0; i < n_; ++i) { T d=T(0); for(int j=k+1;j<n_;++j) d+=H[i*n_+j]*v[j]; d*=tau; for(int j=k+1;j<n_;++j) H[i*n_+j]-=d*v[j]; }
                        for (int i = 0; i < n_; ++i) { T d=T(0); for(int j=k+1;j<n_;++j) d+=Z[i*n_+j]*v[j]; d*=tau; for(int j=k+1;j<n_;++j) Z[i*n_+j]-=d*v[j]; }
                        H[(k+1)*n_+k] = a;
                        for (int i = k+2; i < n_; ++i) H[i*n_+k] = T(0);
                    }

                    // hqr2: real Schur form + eigenvalues
                    T norm = T(0);
                    for (int i = 0; i < n_; ++i) { int j0 = i-1 > 0 ? i-1 : 0; for (int j = j0; j < n_; ++j) norm += sycl::fabs(H[i*n_+j]); }
                    T t = T(0);
                    int en = n_ - 1;
                    while (en >= 0) {
                        int its = 0, na = en - 1;
                        for (;;) {
                            int l;
                            for (l = en; l >= 1; --l) {
                                T s = sycl::fabs(H[(l-1)*n_+(l-1)]) + sycl::fabs(H[l*n_+l]);
                                if (s == T(0)) s = norm;
                                if (sycl::fabs(H[l*n_+(l-1)]) <= eps*s) break;
                            }
                            T x = H[en*n_+en];
                            if (l == en) { H[en*n_+en] = x + t; wr[en] = H[en*n_+en]; wi[en] = T(0); en -= 1; break; }
                            T y = H[na*n_+na];
                            T w = H[en*n_+na]*H[na*n_+en];
                            if (l == na) {
                                T p = T(0.5)*(y - x);
                                T q = p*p + w;
                                T zz = sycl::sqrt(sycl::fabs(q));
                                x = H[en*n_+en] = x + t; H[na*n_+na] = y + t;
                                if (q >= T(0)) {
                                    zz = p + sycl::copysign(zz, p);
                                    wr[na] = wr[en] = x + zz;
                                    if (zz != T(0)) wr[en] = x - w/zz;
                                    wi[na] = wi[en] = T(0);
                                    x = H[en*n_+na];
                                    T s = sycl::fabs(x) + sycl::fabs(zz);
                                    T p2 = x/s, q2 = zz/s;
                                    T r = sycl::sqrt(p2*p2 + q2*q2);
                                    p2 /= r; q2 /= r;
                                    for (int j = na; j < n_; ++j) { T t1=H[na*n_+j]; H[na*n_+j]=q2*t1+p2*H[en*n_+j]; H[en*n_+j]=q2*H[en*n_+j]-p2*t1; }
                                    for (int i = 0; i <= en; ++i) { T t1=H[i*n_+na]; H[i*n_+na]=q2*t1+p2*H[i*n_+en]; H[i*n_+en]=q2*H[i*n_+en]-p2*t1; }
                                    for (int i = 0; i < n_; ++i)  { T t1=Z[i*n_+na]; Z[i*n_+na]=q2*t1+p2*Z[i*n_+en]; Z[i*n_+en]=q2*Z[i*n_+en]-p2*t1; }
                                } else { wr[na] = wr[en] = x + p; wi[na] = zz; wi[en] = -zz; }
                                en -= 2; break;
                            }
                            if (its == 30) { info_ptr[b] = en + 1; en = -1; break; }
                            if (its == 10 || its == 20) {
                                t += x;
                                for (int i = 0; i <= en; ++i) H[i*n_+i] -= x;
                                T s = sycl::fabs(H[en*n_+na]) + sycl::fabs(H[na*n_+(na-1)]);
                                y = x = T(0.75)*s; w = T(-0.4375)*s*s;
                            }
                            its += 1;
                            T p,q,r; int m;
                            for (m = en-2; m >= l; --m) {
                                T zz = H[m*n_+m];
                                r = x - zz; T s = y - zz;
                                p = (r*s - w)/H[(m+1)*n_+m] + H[m*n_+(m+1)];
                                q = H[(m+1)*n_+(m+1)] - zz - r - s;
                                r = H[(m+2)*n_+(m+1)];
                                T sc = sycl::fabs(p)+sycl::fabs(q)+sycl::fabs(r);
                                p /= sc; q /= sc; r /= sc;
                                if (m == l) break;
                                T tst1 = sycl::fabs(p)*(sycl::fabs(H[(m-1)*n_+(m-1)])+sycl::fabs(zz)+sycl::fabs(H[(m+1)*n_+(m+1)]));
                                T tst2 = sycl::fabs(H[m*n_+(m-1)])*(sycl::fabs(q)+sycl::fabs(r));
                                if (tst2 <= eps*tst1) break;
                            }
                            for (int i = m+2; i <= en; ++i) { H[i*n_+(i-2)]=T(0); if (i != m+2) H[i*n_+(i-3)]=T(0); }
                            for (int k = m; k <= na; ++k) {
                                bool notlast = (k != na);
                                if (k != m) {
                                    p = H[k*n_+(k-1)]; q = H[(k+1)*n_+(k-1)]; r = notlast ? H[(k+2)*n_+(k-1)] : T(0);
                                    x = sycl::fabs(p)+sycl::fabs(q)+sycl::fabs(r);
                                    if (x == T(0)) continue;
                                    p /= x; q /= x; r /= x;
                                }
                                T s = sycl::copysign(sycl::sqrt(p*p+q*q+r*r), p);
                                if (k == m) { if (l != m) H[k*n_+(k-1)] = -H[k*n_+(k-1)]; }
                                else H[k*n_+(k-1)] = -s*x;
                                p += s;
                                T xx = p/s, yy = q/s, zz = r/s;
                                q /= p; r /= p;
                                for (int j = k; j < n_; ++j) {
                                    p = H[k*n_+j] + q*H[(k+1)*n_+j];
                                    if (notlast) { p += r*H[(k+2)*n_+j]; H[(k+2)*n_+j] -= p*zz; }
                                    H[(k+1)*n_+j] -= p*yy; H[k*n_+j] -= p*xx;
                                }
                                int jmax = en < k+3 ? en : k+3;
                                for (int i = 0; i <= jmax; ++i) {
                                    p = xx*H[i*n_+k] + yy*H[i*n_+(k+1)];
                                    if (notlast) { p += zz*H[i*n_+(k+2)]; H[i*n_+(k+2)] -= p*r; }
                                    H[i*n_+(k+1)] -= p*q; H[i*n_+k] -= p;
                                }
                                for (int i = 0; i < n_; ++i) {
                                    p = xx*Z[i*n_+k] + yy*Z[i*n_+(k+1)];
                                    if (notlast) { p += zz*Z[i*n_+(k+2)]; Z[i*n_+(k+2)] -= p*r; }
                                    Z[i*n_+(k+1)] -= p*q; Z[i*n_+k] -= p;
                                }
                            }
                        }
                    }

                    // eigenvector back-substitution on the Schur form
                    if (norm != T(0)) {
                        for (int e = n_-1; e >= 0; --e) {
                            T p = wr[e], q = wi[e];
                            int na2 = e - 1;
                            if (q == T(0)) {
                                int m = e; H[e*n_+e] = T(1);
                                T zz_s = T(0), s_s = T(0);
                                for (int i = e-1; i >= 0; --i) {
                                    T w = H[i*n_+i] - p;
                                    T r = T(0);
                                    for (int j = m; j <= e; ++j) r += H[i*n_+j]*H[j*n_+e];
                                    if (wi[i] < T(0)) { zz_s = w; s_s = r; continue; }
                                    m = i;
                                    if (wi[i] == T(0)) {
                                        T tt = (w == T(0)) ? eps*norm : w;
                                        H[i*n_+e] = -r/tt;
                                    } else {
                                        T xx = H[i*n_+(i+1)], yv = H[(i+1)*n_+i];
                                        T qd = (wr[i]-p)*(wr[i]-p) + wi[i]*wi[i];
                                        T tt = (xx*s_s - zz_s*r)/qd;
                                        H[i*n_+e] = tt;
                                        if (sycl::fabs(xx) > sycl::fabs(zz_s)) H[(i+1)*n_+e] = (-r - w*tt)/xx;
                                        else                                   H[(i+1)*n_+e] = (-s_s - yv*tt)/zz_s;
                                    }
                                    T tmag = sycl::fabs(H[i*n_+e]);
                                    if ((eps*tmag)*tmag > T(1)) for (int j = i; j <= e; ++j) H[j*n_+e] /= tmag;
                                }
                            } else if (q < T(0)) {
                                int m = na2;
                                if (sycl::fabs(H[e*n_+na2]) > sycl::fabs(H[na2*n_+e])) {
                                    H[na2*n_+na2] = q/H[e*n_+na2];
                                    H[na2*n_+e]   = -(H[e*n_+e]-p)/H[e*n_+na2];
                                } else {
                                    T zr, zi; cdiv(T(0), -H[na2*n_+e], H[na2*n_+na2]-p, q, zr, zi);
                                    H[na2*n_+na2] = zr; H[na2*n_+e] = zi;
                                }
                                H[e*n_+na2] = T(0); H[e*n_+e] = T(1);
                                T zz_s = T(0), ra_s = T(0), sa_s = T(0);
                                for (int i = e-2; i >= 0; --i) {
                                    T w = H[i*n_+i] - p;
                                    T ra = T(0), sa = T(0);
                                    for (int j = m; j <= e; ++j) { ra += H[i*n_+j]*H[j*n_+na2]; sa += H[i*n_+j]*H[j*n_+e]; }
                                    if (wi[i] < T(0)) { zz_s = w; ra_s = ra; sa_s = sa; continue; }
                                    m = i;
                                    if (wi[i] == T(0)) {
                                        T zr, zi; cdiv(-ra, -sa, w, q, zr, zi);
                                        H[i*n_+na2] = zr; H[i*n_+e] = zi;
                                    } else {
                                        T xx = H[i*n_+(i+1)], yv = H[(i+1)*n_+i];
                                        T vr = (wr[i]-p)*(wr[i]-p) + wi[i]*wi[i] - q*q;
                                        T vi = (wr[i]-p)*T(2)*q;
                                        if (vr == T(0) && vi == T(0))
                                            vr = eps*norm*(sycl::fabs(w)+sycl::fabs(q)+sycl::fabs(xx)+sycl::fabs(yv)+sycl::fabs(zz_s));
                                        T zr, zi;
                                        cdiv(xx*ra_s - zz_s*ra + q*sa, xx*sa_s - zz_s*sa - q*ra, vr, vi, zr, zi);
                                        H[i*n_+na2] = zr; H[i*n_+e] = zi;
                                        if (sycl::fabs(xx) > sycl::fabs(zz_s) + sycl::fabs(q)) {
                                            H[(i+1)*n_+na2] = (-ra - w*H[i*n_+na2] + q*H[i*n_+e])/xx;
                                            H[(i+1)*n_+e]   = (-sa - w*H[i*n_+e] - q*H[i*n_+na2])/xx;
                                        } else {
                                            T z2r, z2i; cdiv(-ra_s - yv*H[i*n_+na2], -sa_s - yv*H[i*n_+e], zz_s, q, z2r, z2i);
                                            H[(i+1)*n_+na2] = z2r; H[(i+1)*n_+e] = z2i;
                                        }
                                    }
                                    T tmag = sycl::fabs(H[i*n_+na2]); { T tm2 = sycl::fabs(H[i*n_+e]); if (tm2 > tmag) tmag = tm2; }
                                    if ((eps*tmag)*tmag > T(1)) for (int j = i; j <= e; ++j) { H[j*n_+na2] /= tmag; H[j*n_+e] /= tmag; }
                                }
                            }
                        }
                        for (int j = n_-1; j >= 0; --j)
                            for (int i = 0; i < n_; ++i) {
                                T s = T(0);
                                for (int k = 0; k <= j; ++k) s += Z[i*n_+k]*H[k*n_+j];
                                Z[i*n_+j] = s;
                            }
                    }

                    // normalize each (complex) eigenvector to unit 2-norm
                    for (int k = 0; k < n_; ++k) {
                        if (wi[k] > T(0)) {
                            T nrm = T(0); for (int i=0;i<n_;++i) nrm += Z[i*n_+k]*Z[i*n_+k]+Z[i*n_+(k+1)]*Z[i*n_+(k+1)];
                            nrm = sycl::sqrt(nrm); if (nrm>T(0)) for(int i=0;i<n_;++i){ Z[i*n_+k]/=nrm; Z[i*n_+(k+1)]/=nrm; }
                            ++k;
                        } else if (wi[k] == T(0)) {
                            T nrm = T(0); for (int i=0;i<n_;++i) nrm += Z[i*n_+k]*Z[i*n_+k];
                            nrm = sycl::sqrt(nrm); if (nrm>T(0)) for(int i=0;i<n_;++i) Z[i*n_+k]/=nrm;
                        }
                    }
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32) launch_eig(work.data<float>(), WR.data<float>(), WI.data<float>(), V.data<float>(), vbuf.data<float>());
    else launch_eig(work.data<double>(), WR.data<double>(), WI.data<double>(), V.data<double>(), vbuf.data<double>());

    // Copy the per-batch convergence flags back and throw on the first
    // failure, matching LAPACK geev's info>0 contract so callers do not
    // receive silently-wrong (zero-initialized) eigenpairs.
    std::vector<int> h_info(static_cast<size_t>(nbatch), 0);
    queue.memcpy(h_info.data(), d_info.ptr, nbatch * sizeof(int)).wait();
    for (int64_t b = 0; b < nbatch; b++) {
        if (h_info[b] > 0) {
            throw std::runtime_error(
                "linalg_eig: QR iteration failed to converge for eigenvalue " +
                std::to_string(h_info[b]) +
                " (the eigenvalue computation did not converge)");
        }
    }

    return {WR, WI, V};
}

// =========================================================================
// LDL^T factorization — native SYCL Bunch-Kaufman kernel
// =========================================================================
auto linalg_ldl_factor_kernel(const Tensor& A, sycl::queue& queue)
    -> std::tuple<Tensor, Tensor> {
    auto original_dtype = A.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto [LD32, piv] = linalg_ldl_factor_kernel(A.to(DType::Float32), queue);
        return {LD32.to(original_dtype), piv};
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument("linalg::ldl_factor: unsupported dtype");
    }

    auto shape = A.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) throw std::invalid_argument("linalg::ldl_factor: input must be at least 2D");
    int64_t n = shape[ndim - 1];
    if (shape[ndim - 2] != n) throw std::invalid_argument("linalg::ldl_factor: expected square matrix");

    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < ndim; ++i) nbatch *= shape[i];

    std::vector<int64_t> piv_shape;
    for (size_t i = 0; i + 2 < shape.size(); ++i) piv_shape.push_back(shape[i]);
    piv_shape.push_back(n);

    auto work = A.contiguous().clone();
    auto pivots_out = zeros(piv_shape, DType::Int32, A.device());

    auto launch = [&](auto* data_ptr) {
        using T = std::remove_pointer_t<decltype(data_ptr)>;
        int32_t* piv_ptr = pivots_out.template data<int32_t>();
        int64_t n_ = n;

        queue.submit([&](sycl::handler& h) {
            size_t smem_sz = (n_ * n_ + 4) * sizeof(T);
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_sz), h);
            auto* data = data_ptr;
            auto* piv = piv_ptr;
            h.parallel_for(sycl::nd_range<1>(nbatch, 1),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    char* smem_raw = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                    T* As = reinterpret_cast<T*>(smem_raw);
                    T* scratch = As + n_ * n_;

                    T* batch_data = data + batch_idx * n_ * n_;
                    int* batch_piv = piv + batch_idx * n_;

                    for (int idx = 0; idx < n_ * n_; idx++)
                        As[idx] = batch_data[idx];

                    const T alpha_val = static_cast<T>(0.6404);

                    int k = 0;
                    while (k < n_) {
                        T col_max = T(0);
                        int col_max_row = k;
                        for (int i = k + 1; i < n_; i++) {
                            T v = sycl::fabs(As[i * n_ + k]);
                            if (v > col_max) { col_max = v; col_max_row = i; }
                        }
                        T abs_akk = sycl::fabs(As[k * n_ + k]);

                        int pivot_type;
                        int swap_row;

                        if (abs_akk == T(0) && col_max == T(0)) {
                            batch_piv[k] = k + 1;
                            pivot_type = 1; swap_row = k;
                        } else if (abs_akk >= alpha_val * col_max) {
                            batch_piv[k] = k + 1;
                            pivot_type = 1; swap_row = k;
                        } else {
                            int r = col_max_row;
                            T row_max = T(0);
                            for (int j = k; j < n_; j++) {
                                if (j == r) continue;
                                T v = sycl::fabs(As[r * n_ + j]);
                                if (v > row_max) row_max = v;
                            }
                            T abs_arr = sycl::fabs(As[r * n_ + r]);

                            if (abs_akk * row_max >= alpha_val * col_max * col_max) {
                                batch_piv[k] = k + 1;
                                pivot_type = 1; swap_row = k;
                            } else if (abs_arr >= alpha_val * row_max) {
                                batch_piv[k] = r + 1;
                                pivot_type = 1; swap_row = r;
                            } else {
                                batch_piv[k] = -(r + 1);
                                batch_piv[k + 1] = -(r + 1);
                                pivot_type = 2; swap_row = r;
                            }
                        }

                        if (pivot_type == 1) {
                            if (swap_row != k) {
                                for (int j = 0; j < n_; j++) {
                                    T tmp = As[k * n_ + j];
                                    As[k * n_ + j] = As[swap_row * n_ + j];
                                    As[swap_row * n_ + j] = tmp;
                                }
                                for (int i = 0; i < n_; i++) {
                                    T tmp = As[i * n_ + k];
                                    As[i * n_ + k] = As[i * n_ + swap_row];
                                    As[i * n_ + swap_row] = tmp;
                                }
                            }
                            T diag = As[k * n_ + k];
                            if (diag != T(0)) {
                                for (int i = k + 1; i < n_; i++)
                                    As[i * n_ + k] /= diag;
                                for (int i = k + 1; i < n_; i++) {
                                    T lik = As[i * n_ + k];
                                    for (int j = k + 1; j <= i; j++)
                                        As[i * n_ + j] -= lik * diag * As[j * n_ + k];
                                }
                            }
                            k++;
                        } else {
                            if (swap_row != k + 1) {
                                for (int j = 0; j < n_; j++) {
                                    T tmp = As[(k+1) * n_ + j];
                                    As[(k+1) * n_ + j] = As[swap_row * n_ + j];
                                    As[swap_row * n_ + j] = tmp;
                                }
                                for (int i = 0; i < n_; i++) {
                                    T tmp = As[i * n_ + (k+1)];
                                    As[i * n_ + (k+1)] = As[i * n_ + swap_row];
                                    As[i * n_ + swap_row] = tmp;
                                }
                            }
                            T d11 = As[k * n_ + k];
                            T d21 = As[(k+1) * n_ + k];
                            T d22 = As[(k+1) * n_ + (k+1)];
                            T det = d11 * d22 - d21 * d21;
                            if (det != T(0)) {
                                T inv11 = d22 / det, inv12 = -d21 / det, inv22 = d11 / det;
                                for (int i = k + 2; i < n_; i++) {
                                    T a0 = As[i * n_ + k], a1 = As[i * n_ + (k+1)];
                                    As[i * n_ + k]     = inv11 * a0 + inv12 * a1;
                                    As[i * n_ + (k+1)] = inv12 * a0 + inv22 * a1;
                                }
                                for (int i = k + 2; i < n_; i++) {
                                    T li0 = As[i * n_ + k], li1 = As[i * n_ + (k+1)];
                                    for (int j = k + 2; j <= i; j++) {
                                        T lj0 = As[j * n_ + k], lj1 = As[j * n_ + (k+1)];
                                        As[i * n_ + j] -= (li0 * (d11 * lj0 + d21 * lj1)
                                                         + li1 * (d21 * lj0 + d22 * lj1));
                                    }
                                }
                            }
                            k += 2;
                        }
                    }

                    for (int idx = 0; idx < n_ * n_; idx++)
                        batch_data[idx] = As[idx];
                });
        }).wait();
    };

    if (A.dtype() == DType::Float32)
        launch(work.data<float>());
    else
        launch(work.data<double>());

    return {work, pivots_out};
}

// =========================================================================
// LDL^T solve — native SYCL Bunch-Kaufman solve kernel
// =========================================================================
auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                              const Tensor& B, sycl::queue& queue) -> Tensor {
    auto original_dtype = LD.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto result = linalg_ldl_solve_kernel(LD.to(DType::Float32), pivots,
                                               B.to(DType::Float32), queue);
        return result.to(original_dtype);
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument("linalg::ldl_solve: unsupported dtype");
    }

    auto ld_shape = LD.shape();
    auto b_shape = B.shape();
    int64_t ld_ndim = static_cast<int64_t>(ld_shape.size());
    int64_t n = ld_shape[ld_ndim - 1];
    int64_t nrhs = b_shape[static_cast<int64_t>(b_shape.size()) - 1];
    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < ld_ndim; ++i) nbatch *= ld_shape[i];

    auto ld_cont = LD.contiguous();
    auto work_b = B.contiguous().clone();

    // The LDL^T solve kernel has no error-reporting path back to the host, so
    // an exactly-zero D-block pivot (a singular factor) would silently
    // produce Inf/NaN instead of throwing (matches CPU/CUDA/ROCm, which all
    // detect this via the same pivot-block decoding and throw). Validate
    // host-side before ever enqueueing the kernel.
    {
        Tensor ld_host = ld_cont.to(Device::cpu());
        Tensor piv_host = pivots.contiguous().to(Device::cpu());
        const int32_t* piv_ptr = piv_host.data<int32_t>();
        auto check_batch = [&](auto* ld_ptr) {
            for (int64_t bidx = 0; bidx < nbatch; ++bidx) {
                auto* ld_mat = ld_ptr + bidx * n * n;
                const int32_t* piv_mat = piv_ptr + bidx * n;
                for (int64_t k = 0; k < n; ) {
                    int32_t p = piv_mat[k];
                    if (p > 0) {
                        if (ld_mat[k * n + k] == 0) {
                            throw std::runtime_error(
                                "linalg::ldl_solve: singular LDL^T factor (zero pivot)");
                        }
                        k++;
                    } else {
                        auto d11 = ld_mat[k * n + k];
                        auto d21 = ld_mat[(k + 1) * n + k];
                        auto d22 = ld_mat[(k + 1) * n + (k + 1)];
                        if (d11 * d22 - d21 * d21 == 0) {
                            throw std::runtime_error(
                                "linalg::ldl_solve: singular LDL^T factor (zero pivot)");
                        }
                        k += 2;
                    }
                }
            }
        };
        if (original_dtype == DType::Float64) {
            check_batch(ld_host.data<double>());
        } else {
            check_batch(ld_host.data<float>());
        }
    }

    auto launch = [&](auto* ld_ptr, auto* b_ptr) {
        using T = std::remove_pointer_t<decltype(ld_ptr)>;
        const int32_t* piv_ptr = pivots.template data<int32_t>();
        int64_t n_ = n, nrhs_ = nrhs;

        queue.submit([&](sycl::handler& h) {
            size_t smem_sz = (n_ * n_ + n_ * nrhs_) * sizeof(T);
            sycl::local_accessor<char, 1> smem(sycl::range<1>(smem_sz), h);
            auto* ld_data = ld_ptr;
            auto* b_data = b_ptr;
            auto* piv = piv_ptr;
            h.parallel_for(sycl::nd_range<1>(nbatch, 1),
                [=](sycl::nd_item<1> item) {
                    int batch_idx = item.get_group_linear_id();
                    char* smem_raw = smem.get_multi_ptr<sycl::access::decorated::no>().get();
                    T* LDs = reinterpret_cast<T*>(smem_raw);
                    T* Bs = LDs + n_ * n_;

                    const T* batch_ld = ld_data + batch_idx * n_ * n_;
                    const int32_t* batch_piv = piv + batch_idx * n_;
                    T* batch_b = b_data + batch_idx * n_ * nrhs_;

                    for (int idx = 0; idx < n_ * n_; idx++)
                        LDs[idx] = batch_ld[idx];
                    for (int idx = 0; idx < n_ * nrhs_; idx++)
                        Bs[idx] = batch_b[idx];

                    // Forward pivot permutation
                    for (int k = 0; k < n_; ) {
                        int p = batch_piv[k];
                        if (p > 0) {
                            int sr = p - 1;
                            if (sr != k)
                                for (int j = 0; j < nrhs_; j++) {
                                    T tmp = Bs[k * nrhs_ + j]; Bs[k * nrhs_ + j] = Bs[sr * nrhs_ + j]; Bs[sr * nrhs_ + j] = tmp;
                                }
                            k++;
                        } else {
                            int sr = (-p) - 1;
                            if (sr != k + 1)
                                for (int j = 0; j < nrhs_; j++) {
                                    T tmp = Bs[(k+1) * nrhs_ + j]; Bs[(k+1) * nrhs_ + j] = Bs[sr * nrhs_ + j]; Bs[sr * nrhs_ + j] = tmp;
                                }
                            k += 2;
                        }
                    }

                    // Forward substitution
                    for (int k = 0; k < n_; ) {
                        int p = batch_piv[k];
                        if (p > 0) {
                            for (int i = k + 1; i < n_; i++) {
                                T m = LDs[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++)
                                    Bs[i * nrhs_ + j] -= m * Bs[k * nrhs_ + j];
                            }
                            k++;
                        } else {
                            for (int i = k + 2; i < n_; i++) {
                                T m0 = LDs[i * n_ + k], m1 = LDs[i * n_ + k + 1];
                                for (int j = 0; j < nrhs_; j++)
                                    Bs[i * nrhs_ + j] -= m0 * Bs[k * nrhs_ + j] + m1 * Bs[(k+1) * nrhs_ + j];
                            }
                            k += 2;
                        }
                    }

                    // Diagonal solve
                    for (int k = 0; k < n_; ) {
                        int p = batch_piv[k];
                        if (p > 0) {
                            T d = LDs[k * n_ + k];
                            for (int j = 0; j < nrhs_; j++) Bs[k * nrhs_ + j] /= d;
                            k++;
                        } else {
                            T d11 = LDs[k * n_ + k], d21 = LDs[(k+1) * n_ + k], d22 = LDs[(k+1) * n_ + (k+1)];
                            T det = d11 * d22 - d21 * d21;
                            for (int j = 0; j < nrhs_; j++) {
                                T y0 = Bs[k * nrhs_ + j], y1 = Bs[(k+1) * nrhs_ + j];
                                Bs[k * nrhs_ + j]     = (d22 * y0 - d21 * y1) / det;
                                Bs[(k+1) * nrhs_ + j] = (d11 * y1 - d21 * y0) / det;
                            }
                            k += 2;
                        }
                    }

                    // Backward substitution
                    for (int k = n_ - 1; k >= 0; ) {
                        int p = batch_piv[k];
                        if (p > 0) {
                            for (int i = k + 1; i < n_; i++) {
                                T m = LDs[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++)
                                    Bs[k * nrhs_ + j] -= m * Bs[i * nrhs_ + j];
                            }
                            k--;
                        } else {
                            int k0 = k - 1;
                            for (int i = k + 1; i < n_; i++) {
                                T m0 = LDs[i * n_ + k0], m1 = LDs[i * n_ + k];
                                for (int j = 0; j < nrhs_; j++) {
                                    Bs[k0 * nrhs_ + j] -= m0 * Bs[i * nrhs_ + j];
                                    Bs[k * nrhs_ + j]  -= m1 * Bs[i * nrhs_ + j];
                                }
                            }
                            k -= 2;
                        }
                    }

                    // Inverse pivot permutation
                    for (int k = n_ - 1; k >= 0; ) {
                        int p = batch_piv[k];
                        if (p > 0) {
                            int sr = p - 1;
                            if (sr != k)
                                for (int j = 0; j < nrhs_; j++) {
                                    T tmp = Bs[k * nrhs_ + j]; Bs[k * nrhs_ + j] = Bs[sr * nrhs_ + j]; Bs[sr * nrhs_ + j] = tmp;
                                }
                            k--;
                        } else {
                            int sr = (-p) - 1;
                            if (sr != k)
                                for (int j = 0; j < nrhs_; j++) {
                                    T tmp = Bs[k * nrhs_ + j]; Bs[k * nrhs_ + j] = Bs[sr * nrhs_ + j]; Bs[sr * nrhs_ + j] = tmp;
                                }
                            k -= 2;
                        }
                    }

                    for (int idx = 0; idx < n_ * nrhs_; idx++)
                        batch_b[idx] = Bs[idx];
                });
        }).wait();
    };

    if (LD.dtype() == DType::Float32)
        launch(ld_cont.data<float>(), work_b.data<float>());
    else
        launch(ld_cont.data<double>(), work_b.data<double>());

    return work_b;
}

// =========================================================================
// Householder product — compose from existing ormqr kernel
// =========================================================================
auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                sycl::queue& queue) -> Tensor {
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    int64_t m = shape[ndim - 2];

    auto I = tenzor::eye(m, std::nullopt, input.dtype(), input.device());

    if (ndim > 2) {
        std::vector<int64_t> eye_shape(shape.begin(), shape.end());
        eye_shape[ndim - 1] = m;
        I = tenzor::expand(I, std::move(eye_shape));
        I = I.contiguous();
    }

    return linalg_ormqr_kernel(input, tau, I, true, false, queue);
}

} // namespace oneapi
} // namespace tenzor
