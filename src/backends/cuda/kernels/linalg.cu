/**
 * @file linalg.cu
 * @brief CUDA kernels for linear algebra operations using cuSOLVER.
 *
 * Provides GPU-accelerated implementations of:
 * - det (determinant via LU factorization)
 * - inv (matrix inverse via LU factorization)
 * - solve (linear system solve via LU factorization)
 * - svd (singular value decomposition)
 * - qr (QR decomposition)
 * - eigh (symmetric eigendecomposition)
 * - cholesky (Cholesky factorization)
 */

#ifdef TENZOR_HAS_CUSOLVER

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/backend/caching_allocator.hpp"
#include "../cusolver_handle_pool.hpp"

#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <cusolverDn.h>
#include <tuple>
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>

namespace tenzor {
namespace cuda {

namespace {

#ifndef CUDA_CHECK_LINALG
#define CUDA_CHECK_LINALG(call)                                                 \
    do {                                                                          \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            throw std::runtime_error(                                            \
                std::string("CUDA error in linalg at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));     \
        }                                                                        \
    } while (0)
#endif

/// Convert span to vector (nvcc doesn't support implicit span→vector conversion).
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

/// Check cuSOLVER info value (copied from device to host).
void check_cusolver_info(int* d_info, const std::string& op_name) {
    int h_info = 0;
    CUDA_CHECK_LINALG(cudaMemcpy(&h_info, d_info, sizeof(int), cudaMemcpyDeviceToHost));
    if (h_info < 0) {
        throw std::runtime_error("linalg::" + op_name + ": invalid argument (info=" +
                                 std::to_string(h_info) + ")");
    } else if (h_info > 0) {
        throw std::runtime_error("linalg::" + op_name + ": computation failed (info=" +
                                 std::to_string(h_info) + ")");
    }
}

/// Simple RAII wrapper for a device-side int (for cuSOLVER info output).
/// Routes through the caching allocator to avoid cudaMalloc/cudaFree overhead.
struct DeviceInt {
    int* ptr = nullptr;
    DeviceInt() {
        ptr = static_cast<int*>(backend::CachingAllocator::get().allocate(sizeof(int)));
    }
    ~DeviceInt() { if (ptr) backend::CachingAllocator::get().free(ptr); }
    DeviceInt(const DeviceInt&) = delete;
    DeviceInt& operator=(const DeviceInt&) = delete;
};

/// RAII wrapper for device workspace.
/// Routes through the caching allocator to avoid cudaMalloc/cudaFree overhead.
struct DeviceWorkspace {
    void* ptr = nullptr;
    DeviceWorkspace(size_t bytes) {
        if (bytes > 0) ptr = backend::CachingAllocator::get().allocate(bytes);
    }
    ~DeviceWorkspace() { if (ptr) backend::CachingAllocator::get().free(ptr); }
    DeviceWorkspace(const DeviceWorkspace&) = delete;
    DeviceWorkspace& operator=(const DeviceWorkspace&) = delete;
};

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

/// CUDA kernel to set a batched identity matrix on device.
template<typename T>
__global__ void set_identity_kernel(T* data, int64_t n, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t mat_offset = idx % (n * n);
    int64_t row = mat_offset / n;
    int64_t col = mat_offset % n;
    data[idx] = (row == col) ? T(1) : T(0);
}

/// CUDA kernel to compute determinant from LU diagonal + pivot info.
__global__ void det_from_lu_f32(const float* lu_data, const int* ipiv,
                                 float* det_out, int n, int nbatch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbatch) return;

    const float* mat = lu_data + b * n * n;
    const int* piv = ipiv + b * n;
    float d = 1.0f;
    for (int i = 0; i < n; i++) {
        d *= mat[i * n + i];
        // LAPACK-style 1-based pivots
        if (piv[i] != i + 1) d = -d;
    }
    det_out[b] = d;
}

__global__ void det_from_lu_f64(const double* lu_data, const int* ipiv,
                                 double* det_out, int n, int nbatch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbatch) return;

    const double* mat = lu_data + b * n * n;
    const int* piv = ipiv + b * n;
    double d = 1.0;
    for (int i = 0; i < n; i++) {
        d *= mat[i * n + i];
        if (piv[i] != i + 1) d = -d;
    }
    det_out[b] = d;
}

/// CUDA kernel to zero out one triangle after Cholesky.
__global__ void zero_triangle_f32(float* data, int n, int nbatch, bool upper) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * n * n;
    if (idx >= total) return;

    int b = idx / (n * n);
    int rem = idx % (n * n);
    int i = rem / n;
    int j = rem % n;

    float* mat = data + b * n * n;
    if (upper ? (i > j) : (i < j)) {
        mat[i * n + j] = 0.0f;
    }
}

__global__ void zero_triangle_f64(double* data, int n, int nbatch, bool upper) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * n * n;
    if (idx >= total) return;

    int b = idx / (n * n);
    int rem = idx % (n * n);
    int i = rem / n;
    int j = rem % n;

    double* mat = data + b * n * n;
    if (upper ? (i > j) : (i < j)) {
        mat[i * n + j] = 0.0;
    }
}

/// CUDA kernel to extract R (upper triangle) from QR factorization result.
__global__ void extract_r_f32(const float* a_data, float* r_data,
                               int m, int n_cols, int k, int nbatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * k * n_cols;
    if (idx >= total) return;

    int b = idx / (k * n_cols);
    int rem = idx % (k * n_cols);
    int i = rem / n_cols;
    int j = rem % n_cols;

    const float* a_mat = a_data + b * m * n_cols;
    float* r_mat = r_data + b * k * n_cols;
    r_mat[i * n_cols + j] = (j >= i) ? a_mat[i * n_cols + j] : 0.0f;
}

__global__ void extract_r_f64(const double* a_data, double* r_data,
                               int m, int n_cols, int k, int nbatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * k * n_cols;
    if (idx >= total) return;

    int b = idx / (k * n_cols);
    int rem = idx % (k * n_cols);
    int i = rem / n_cols;
    int j = rem % n_cols;

    const double* a_mat = a_data + b * m * n_cols;
    double* r_mat = r_data + b * k * n_cols;
    r_mat[i * n_cols + j] = (j >= i) ? a_mat[i * n_cols + j] : 0.0;
}

/// CUDA kernel to copy Q columns from householder result.
__global__ void copy_q_columns_f32(const float* a_data, float* q_data,
                                    int m, int n_cols, int k, int nbatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * m * k;
    if (idx >= total) return;

    int b = idx / (m * k);
    int rem = idx % (m * k);
    int i = rem / k;
    int j = rem % k;

    const float* a_mat = a_data + b * m * n_cols;
    float* q_mat = q_data + b * m * k;
    q_mat[i * k + j] = a_mat[i * n_cols + j];
}

__global__ void copy_q_columns_f64(const double* a_data, double* q_data,
                                    int m, int n_cols, int k, int nbatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * m * k;
    if (idx >= total) return;

    int b = idx / (m * k);
    int rem = idx % (m * k);
    int i = rem / k;
    int j = rem % k;

    const double* a_mat = a_data + b * m * n_cols;
    double* q_mat = q_data + b * m * k;
    q_mat[i * k + j] = a_mat[i * n_cols + j];
}

/// CUDA kernel to split a row-major packed LU matrix into separate L (unit
/// lower triangular) and U (upper triangular) tensors. One thread per element.
template<typename T>
__global__ void extract_lu_kernel(const T* packed, T* L, T* U,
                                   int64_t n, int64_t nbatch) {
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t total = nbatch * n * n;
    if (idx >= total) return;

    int64_t rem = idx % (n * n);
    int64_t i = rem / n;
    int64_t j = rem % n;

    if (i > j) {
        L[idx] = packed[idx];
        U[idx] = T(0);
    } else if (i == j) {
        L[idx] = T(1);
        U[idx] = packed[idx];
    } else {
        L[idx] = T(0);
        U[idx] = packed[idx];
    }
}

} // anonymous namespace

// ============================================================================
// Determinant
// ============================================================================

auto linalg_det_kernel(const Tensor& A, cudaStream_t stream) -> Tensor {
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) out_shape.push_back(shape[i]);
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = zeros(out_shape, A.dtype(), A.device());
    auto handle = CuSOLVERHandlePool::get(stream);

    // Allocate pivot array and info on device (via caching allocator)
    backend::CachedMemoryGuard ipiv_guard(nbatch * n * sizeof(int));
    int* d_ipiv = static_cast<int*>(ipiv_guard.get());
    DeviceInt d_info;

    if (A.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = data + b * n * n;
            int* piv = d_ipiv + b * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, mat, n,
                static_cast<float*>(workspace.ptr), piv, d_info.ptr));
        }

        int threads = 256;
        int blocks = (nbatch + threads - 1) / threads;
        det_from_lu_f32<<<blocks, threads, 0, stream>>>(
            data, d_ipiv, result.data<float>(), n, nbatch);
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = data + b * n * n;
            int* piv = d_ipiv + b * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, mat, n,
                static_cast<double*>(workspace.ptr), piv, d_info.ptr));
        }

        int threads = 256;
        int blocks = (nbatch + threads - 1) / threads;
        det_from_lu_f64<<<blocks, threads, 0, stream>>>(
            data, d_ipiv, result.data<double>(), n, nbatch);
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return result;
}

// ============================================================================
// Matrix Inverse
// ============================================================================

auto linalg_inv_kernel(const Tensor& A, cudaStream_t stream) -> Tensor {
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);
    auto handle = CuSOLVERHandlePool::get(stream);

    backend::CachedMemoryGuard ipiv_guard(n * sizeof(int));
    int* d_ipiv = static_cast<int*>(ipiv_guard.get());
    DeviceInt d_info;

    // Create identity matrix on device for getrs-based inversion
    auto identity = zeros(to_vec(work.shape()), A.dtype(), A.device());

    if (A.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* id_data = identity.data<float>();

        // Set identity matrix on device
        {
            int64_t total = nbatch * n * n;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            set_identity_kernel<<<blocks, threads, 0, stream>>>(id_data, n, total);
            CUDA_CHECK_LINALG(cudaGetLastError());
        }

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = data + b * n * n;
            float* id_mat = id_data + b * n * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, mat, n,
                static_cast<float*>(workspace.ptr), d_ipiv, d_info.ptr));
            check_cusolver_info(d_info.ptr, "inv");

            CUSOLVER_CHECK(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, n,
                mat, n, d_ipiv, id_mat, n, d_info.ptr));
            check_cusolver_info(d_info.ptr, "inv");
        }
    } else {
        double* data = work.data<double>();
        double* id_data = identity.data<double>();

        // Set identity matrix on device
        {
            int64_t total = nbatch * n * n;
            int threads = 256;
            int blocks = (total + threads - 1) / threads;
            set_identity_kernel<<<blocks, threads, 0, stream>>>(id_data, n, total);
            CUDA_CHECK_LINALG(cudaGetLastError());
        }

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = data + b * n * n;
            double* id_mat = id_data + b * n * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, mat, n,
                static_cast<double*>(workspace.ptr), d_ipiv, d_info.ptr));
            check_cusolver_info(d_info.ptr, "inv");

            CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, n,
                mat, n, d_ipiv, id_mat, n, d_info.ptr));
            check_cusolver_info(d_info.ptr, "inv");
        }
    }

    // (Phase 7.2) No trailing cudaStreamSynchronize needed: check_cusolver_info
    // inside the batch loop does a synchronous cudaMemcpy for d_info, which is
    // a full stream sync, and no kernels are launched after the loop.
    return identity;
}

// ============================================================================
// Linear System Solve (AX = B)
// ============================================================================

auto linalg_solve_kernel(const Tensor& A, const Tensor& B, cudaStream_t stream) -> Tensor {
    auto work_a = A.contiguous().clone();
    auto work_b = B.contiguous().clone();
    auto [n, ndim_a] = check_square(work_a);
    int64_t nbatch = batch_size(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;

    auto handle = CuSOLVERHandlePool::get(stream);

    backend::CachedMemoryGuard ipiv_guard(n * sizeof(int));
    int* d_ipiv = static_cast<int*>(ipiv_guard.get());
    DeviceInt d_info;

    if (A.dtype() == DType::Float32) {
        float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * n * n;
            float* b_mat = b_data + b * n * nrhs;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, a_mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, a_mat, n,
                static_cast<float*>(workspace.ptr), d_ipiv, d_info.ptr));
            check_cusolver_info(d_info.ptr, "solve");

            CUSOLVER_CHECK(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, nrhs,
                a_mat, n, d_ipiv, b_mat, n, d_info.ptr));
            check_cusolver_info(d_info.ptr, "solve");
        }
    } else {
        double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * n * n;
            double* b_mat = b_data + b * n * nrhs;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, a_mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, a_mat, n,
                static_cast<double*>(workspace.ptr), d_ipiv, d_info.ptr));
            check_cusolver_info(d_info.ptr, "solve");

            CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs,
                a_mat, n, d_ipiv, b_mat, n, d_info.ptr));
            check_cusolver_info(d_info.ptr, "solve");
        }
    }

    // (Phase 7.2) Redundant trailing sync removed; check_cusolver_info above
    // already performs a synchronous cudaMemcpy for d_info each iteration.
    return work_b;
}

// ============================================================================
// SVD (Singular Value Decomposition)
// ============================================================================

auto linalg_svd_kernel(const Tensor& A, bool full_matrices, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    auto work = A.contiguous().clone();
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::svd: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

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

    auto U = zeros(u_shape, A.dtype(), A.device());
    auto S = zeros(s_shape, A.dtype(), A.device());
    auto Vt = zeros(vt_shape, A.dtype(), A.device());

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    // cuSOLVER uses column-major; we use row-major. For row-major A (m x n),
    // cuSOLVER sees A^T (n x m) in column-major. SVD(A^T) = V S U^T,
    // so we swap U and Vt roles and transpose dimensions.
    // Simpler approach: use gesvd with column-major layout by treating
    // our row-major m×n as column-major n×m.

    signed char jobu = full_matrices ? 'A' : 'S';
    signed char jobvt = full_matrices ? 'A' : 'S';

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* s_data = S.data<float>();
        float* u_data = U.data<float>();
        float* vt_data = Vt.data<float>();

        int64_t u_stride = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;
            float* s_vec = s_data + b * k;
            float* u_mat = u_data + b * u_stride;
            float* vt_mat = vt_data + b * vt_stride;

            // Row-major A[m][n] => column-major A^T[n][m]
            // gesvd(A^T) => V^T * S * U^T where U,Vt are swapped
            int ldu = full_matrices ? m : m;
            int ldvt = full_matrices ? n_cols : k;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, m, n_cols, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            // For row-major: we pass n_cols as m and m as n to treat as col-major A^T
            // Then U output is actually Vt, and Vt output is actually U
            float* rwork = nullptr;  // not needed for real SVD
            CUSOLVER_CHECK(cusolverDnSgesvd(handle, jobu, jobvt,
                n_cols, m,  // swapped: col-major sees our row-major as transposed
                a_mat, n_cols,  // lda = n_cols (stride between columns in row-major)
                s_vec,
                vt_mat, ldvt,  // "U" output → our Vt
                u_mat, ldu,    // "Vt" output → our U
                static_cast<float*>(workspace.ptr), lwork,
                rwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "svd");
        }
    } else {
        double* a_data = work.data<double>();
        double* s_data = S.data<double>();
        double* u_data = U.data<double>();
        double* vt_data = Vt.data<double>();

        int64_t u_stride = full_matrices ? m * m : m * k;
        int64_t vt_stride = full_matrices ? n_cols * n_cols : k * n_cols;

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * m * n_cols;
            double* s_vec = s_data + b * k;
            double* u_mat = u_data + b * u_stride;
            double* vt_mat = vt_data + b * vt_stride;

            int ldvt = full_matrices ? n_cols : k;
            int ldu = full_matrices ? m : m;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(handle, m, n_cols, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            double* rwork = nullptr;
            CUSOLVER_CHECK(cusolverDnDgesvd(handle, jobu, jobvt,
                n_cols, m,
                a_mat, n_cols,
                s_vec,
                vt_mat, ldvt,
                u_mat, ldu,
                static_cast<double*>(workspace.ptr), lwork,
                rwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "svd");
        }
    }

    // (Phase 7.2) Redundant trailing sync removed — check_cusolver_info above
    // already performs a synchronous cudaMemcpy per batch iteration.
    return {U, S, Vt};
}

// ============================================================================
// QR Decomposition
// ============================================================================

auto linalg_qr_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor> {
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

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* q_data = Q.data<float>();
        float* r_data = R.data<float>();

        // Allocate tau on device (via caching allocator)
        backend::CachedMemoryGuard tau_guard(k * sizeof(float));
        float* d_tau = static_cast<float*>(tau_guard.get());

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;

            // cuSOLVER geqrf works in column-major. For row-major m×n,
            // we pass n_cols as m and m as n (treating as A^T in col-major).
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, n_cols, m, a_mat, n_cols, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgeqrf(handle, n_cols, m,
                a_mat, n_cols, d_tau,
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

            // Extract R from upper triangle
            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f32<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            // Generate Q using orgqr
            int lwork_q = 0;
            CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(handle, n_cols, k, k,
                a_mat, n_cols, d_tau, &lwork_q));
            DeviceWorkspace workspace_q(lwork_q * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSorgqr(handle, n_cols, k, k,
                a_mat, n_cols, d_tau,
                static_cast<float*>(workspace_q.ptr), lwork_q, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

            // Copy Q columns
            int total_q = m * k;
            blocks = (total_q + threads - 1) / threads;
            copy_q_columns_f32<<<blocks, threads, 0, stream>>>(
                a_mat, q_data + b * m * k, m, n_cols, k, 1);
        }
    } else {
        double* a_data = work.data<double>();
        double* q_data = Q.data<double>();
        double* r_data = R.data<double>();

        backend::CachedMemoryGuard tau_guard(k * sizeof(double));
        double* d_tau = static_cast<double*>(tau_guard.get());

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * m * n_cols;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(handle, n_cols, m, a_mat, n_cols, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgeqrf(handle, n_cols, m,
                a_mat, n_cols, d_tau,
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f64<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            int lwork_q = 0;
            CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(handle, n_cols, k, k,
                a_mat, n_cols, d_tau, &lwork_q));
            DeviceWorkspace workspace_q(lwork_q * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDorgqr(handle, n_cols, k, k,
                a_mat, n_cols, d_tau,
                static_cast<double*>(workspace_q.ptr), lwork_q, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

            int total_q = m * k;
            blocks = (total_q + threads - 1) / threads;
            copy_q_columns_f64<<<blocks, threads, 0, stream>>>(
                a_mat, q_data + b * m * k, m, n_cols, k, 1);
        }
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {Q, R};
}

// ============================================================================
// Symmetric Eigendecomposition (eigh)
// ============================================================================

auto linalg_eigh_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, A.dtype(), A.device());

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    // syevd computes eigenvalues and eigenvectors of symmetric matrix
    cusolverEigMode_t jobz = CUSOLVER_EIG_MODE_VECTOR;
    cublasFillMode_t uplo = CUBLAS_FILL_MODE_UPPER;

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* w_data = W.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = a_data + b * n * n;
            float* w_vec = w_data + b * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSsyevd_bufferSize(handle, jobz, uplo,
                n, mat, n, w_vec, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSsyevd(handle, jobz, uplo,
                n, mat, n, w_vec,
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "eigh");
        }
    } else {
        double* a_data = work.data<double>();
        double* w_data = W.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = a_data + b * n * n;
            double* w_vec = w_data + b * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDsyevd_bufferSize(handle, jobz, uplo,
                n, mat, n, w_vec, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDsyevd(handle, jobz, uplo,
                n, mat, n, w_vec,
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "eigh");
        }
    }

    // (Phase 7.2) Redundant trailing sync removed — check_cusolver_info above
    // already performs a synchronous cudaMemcpy per batch iteration.
    // work now contains eigenvectors (columns of orthogonal matrix)
    return {W, work};
}

// ============================================================================
// Non-symmetric Eigendecomposition (eig)
// ============================================================================

auto linalg_eig_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
#if CUDA_VERSION >= 11010
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto WR = zeros(w_shape, A.dtype(), A.device());
    auto WI = zeros(w_shape, A.dtype(), A.device());

    std::vector<int64_t> v_shape = batch_dims;
    v_shape.push_back(n);
    v_shape.push_back(n);
    auto V = zeros(v_shape, A.dtype(), A.device());

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    // Row-major input: A stored row-major = A^T stored column-major
    // Left eigenvectors of A^T = right eigenvectors of A
    cusolverEigMode_t jobvl = CUSOLVER_EIG_MODE_VECTOR;
    cusolverEigMode_t jobvr = CUSOLVER_EIG_MODE_NOVECTOR;

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* wr_data = WR.data<float>();
        float* wi_data = WI.data<float>();
        float* v_data = V.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = a_data + b * n * n;
            float* wr_vec = wr_data + b * n;
            float* wi_vec = wi_data + b * n;
            float* vl = v_data + b * n * n;  // left eigvecs of A^T = right eigvecs of A

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgeev_bufferSize(handle, jobvl, jobvr,
                n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgeev(handle, jobvl, jobvr,
                n, mat, n, wr_vec, wi_vec,
                vl, n,      // VL (left eigenvectors — what we want)
                nullptr, n,  // VR (not computed)
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "eig");
        }
    } else {
        double* a_data = work.data<double>();
        double* wr_data = WR.data<double>();
        double* wi_data = WI.data<double>();
        double* v_data = V.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = a_data + b * n * n;
            double* wr_vec = wr_data + b * n;
            double* wi_vec = wi_data + b * n;
            double* vl = v_data + b * n * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgeev_bufferSize(handle, jobvl, jobvr,
                n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgeev(handle, jobvl, jobvr,
                n, mat, n, wr_vec, wi_vec,
                vl, n,
                nullptr, n,
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "eig");
        }
    }

    // (Phase 7.2) Redundant trailing sync removed — check_cusolver_info above
    // already performs a synchronous cudaMemcpy per batch iteration.
    // V contains left eigenvectors of A^T (= right eigenvectors of A) in column-major
    // which is the same as right eigenvectors in row-major — exactly what we want
    return {WR, WI, V};
#else
    (void)A; (void)stream;
    throw std::runtime_error("eig: cusolverDnGeev requires CUDA 11.1+");
#endif
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

auto linalg_cholesky_kernel(const Tensor& A, bool upper, cudaStream_t stream) -> Tensor {
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    cublasFillMode_t uplo_mode = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;

    if (A.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = data + b * n * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSpotrf_bufferSize(handle, uplo_mode, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSpotrf(handle, uplo_mode, n, mat, n,
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "cholesky");
        }

        // Zero out the other triangle
        int total = nbatch * n * n;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        zero_triangle_f32<<<blocks, threads, 0, stream>>>(data, n, nbatch, upper);
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = data + b * n * n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDpotrf_bufferSize(handle, uplo_mode, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDpotrf(handle, uplo_mode, n, mat, n,
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "cholesky");
        }

        int total = nbatch * n * n;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        zero_triangle_f64<<<blocks, threads, 0, stream>>>(data, n, nbatch, upper);
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work;
}

// ============================================================================
// LU Factorization (PA = LU)
// ============================================================================

auto linalg_lu_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    // Validate dtype: support Float32/Float64 directly, upcast Float16/BFloat16.
    auto original_dtype = A.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto [L32, U32, piv] = linalg_lu_kernel(A.to(DType::Float32), stream);
        return {L32.to(original_dtype), U32.to(original_dtype), piv};
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument(
            "linalg::lu: unsupported dtype, expected Float32 or Float64");
    }

    auto [n, ndim] = check_square(A);  // throws on <2D or non-square
    int64_t nbatch = batch_size(A);

    // Build output shapes (batch_dims + [n, n] for L/U; batch_dims + [n] for pivots).
    auto a_shape = A.shape();
    std::vector<int64_t> mat_shape;
    for (size_t i = 0; i + 2 < a_shape.size(); ++i) mat_shape.push_back(a_shape[i]);
    std::vector<int64_t> piv_shape = mat_shape;
    mat_shape.push_back(n); mat_shape.push_back(n);
    piv_shape.push_back(n);

    // Step 1: transpose A so that its row-major storage becomes the column-major
    // representation of A. cuSOLVER will then factor A directly.
    auto a_t = tenzor::transpose(A, -2, -1).contiguous();

    auto handle = CuSOLVERHandlePool::get(stream);
    backend::CachedMemoryGuard ipiv_guard(nbatch * n * sizeof(int));
    int* d_ipiv = static_cast<int*>(ipiv_guard.get());
    DeviceInt d_info;

    if (original_dtype == DType::Float32) {
        float* data = a_t.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            int* piv = d_ipiv + b * n;
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgetrf_bufferSize(handle, n, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));
            CUSOLVER_CHECK(cusolverDnSgetrf(handle, n, n, mat, n,
                static_cast<float*>(workspace.ptr), piv, d_info.ptr));
            check_cusolver_info(d_info.ptr, "lu");
        }
    } else {
        double* data = a_t.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            int* piv = d_ipiv + b * n;
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgetrf_bufferSize(handle, n, n, mat, n, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));
            CUSOLVER_CHECK(cusolverDnDgetrf(handle, n, n, mat, n,
                static_cast<double*>(workspace.ptr), piv, d_info.ptr));
            check_cusolver_info(d_info.ptr, "lu");
        }
    }

    // Step 2: transpose the column-major packed factors back to row-major
    // LAPACK packed format (L below diagonal, U on/above, unit diag implicit).
    auto packed_rm = tenzor::transpose(a_t, -2, -1).contiguous();

    // Step 3: split packed factors into L (unit lower) and U (upper).
    auto L = zeros(mat_shape, original_dtype, A.device());
    auto U = zeros(mat_shape, original_dtype, A.device());
    int64_t total = nbatch * n * n;
    int threads = 256;
    int blocks = static_cast<int>((total + threads - 1) / threads);
    if (original_dtype == DType::Float32) {
        extract_lu_kernel<float><<<blocks, threads, 0, stream>>>(
            packed_rm.data<float>(), L.data<float>(), U.data<float>(), n, nbatch);
    } else {
        extract_lu_kernel<double><<<blocks, threads, 0, stream>>>(
            packed_rm.data<double>(), L.data<double>(), U.data<double>(), n, nbatch);
    }
    CUDA_CHECK_LINALG(cudaGetLastError());

    // Step 4: copy pivots (cuSOLVER returns 1-based int) into Int32 tensor.
    auto pivots_out = zeros(piv_shape, DType::Int32, A.device());
    CUDA_CHECK_LINALG(cudaMemcpyAsync(pivots_out.data<int32_t>(), d_ipiv,
        nbatch * n * sizeof(int), cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {L, U, pivots_out};
}

// ============================================================================
// LU Solve (X = A^{-1} B given packed LU + pivots from getrf)
// ============================================================================

auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                             const Tensor& B, cudaStream_t stream) -> Tensor {
    auto original_dtype = B.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        return linalg_lu_solve_kernel(
            LU_data.to(DType::Float32), pivots,
            B.to(DType::Float32), stream).to(original_dtype);
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument(
            "linalg::lu_solve: unsupported dtype, expected Float32 or Float64");
    }

    auto lu_shape = LU_data.shape();
    auto b_shape = B.shape();
    auto lu_ndim = static_cast<int64_t>(lu_shape.size());
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    if (lu_ndim < 2 || b_ndim < 2) {
        throw std::invalid_argument("linalg::lu_solve: inputs must be at least 2D");
    }
    int64_t n = lu_shape[lu_ndim - 1];
    if (lu_shape[lu_ndim - 2] != n) {
        throw std::invalid_argument("linalg::lu_solve: LU_data must be square");
    }
    int64_t nrhs = b_shape[b_ndim - 1];
    int64_t nbatch = batch_size(LU_data);

    // Convert row-major packed LU into column-major form for cuSOLVER getrs:
    // transposing the row-major tensor produces storage that, viewed as
    // column-major, is the packed factorization in the col-major convention.
    auto lu_cm = tenzor::transpose(LU_data, -2, -1).contiguous();
    // Same trick for B: row-major (n, nrhs) → transpose to (nrhs, n) row-major,
    // whose storage interpreted col-major is the (n, nrhs) col-major form of B.
    auto b_cm = tenzor::transpose(B, -2, -1).contiguous();

    // Pivots: cuSOLVER expects int*, our pivots are Int32 (same size). Make
    // sure they live on the device of A; copy to a contiguous buffer.
    auto piv_dev = pivots.to(B.device()).contiguous();
    const int* d_ipiv_base = reinterpret_cast<const int*>(piv_dev.data<int32_t>());

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    if (original_dtype == DType::Float32) {
        float* lu_ptr = lu_cm.data<float>();
        float* b_ptr = b_cm.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            float* lu_mat = lu_ptr + b * n * n;
            float* b_mat = b_ptr + b * n * nrhs;
            const int* piv = d_ipiv_base + b * n;
            CUSOLVER_CHECK(cusolverDnSgetrs(handle, CUBLAS_OP_N, n, nrhs,
                lu_mat, n, const_cast<int*>(piv), b_mat, n, d_info.ptr));
            check_cusolver_info(d_info.ptr, "lu_solve");
        }
    } else {
        double* lu_ptr = lu_cm.data<double>();
        double* b_ptr = b_cm.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            double* lu_mat = lu_ptr + b * n * n;
            double* b_mat = b_ptr + b * n * nrhs;
            const int* piv = d_ipiv_base + b * n;
            CUSOLVER_CHECK(cusolverDnDgetrs(handle, CUBLAS_OP_N, n, nrhs,
                lu_mat, n, const_cast<int*>(piv), b_mat, n, d_info.ptr));
            check_cusolver_info(d_info.ptr, "lu_solve");
        }
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    // b_cm has shape (..., nrhs, n) row-major; transpose back to (..., n, nrhs).
    return tenzor::transpose(b_cm, -2, -1).contiguous();
}

} // namespace cuda
} // namespace tenzor

#else // !TENZOR_HAS_CUSOLVER — native CUDA fallback kernels

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include <cuda_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>

namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace cuda {

namespace {

#ifndef CUDA_CHECK_LINALG
#define CUDA_CHECK_LINALG(call)                                                 \
    do {                                                                          \
        cudaError_t err = (call);                                                \
        if (err != cudaSuccess) {                                                \
            throw std::runtime_error(                                            \
                std::string("CUDA error in linalg at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + cudaGetErrorString(err));     \
        }                                                                        \
    } while (0)
#endif

/// Max matrix dimension for shared-memory fallback kernels.
/// Shared memory usage is 2*N*N*sizeof(T) + scratch, capped at 48KB default.
constexpr int MAX_N_FLOAT  = 90;
constexpr int MAX_N_DOUBLE = 64;

/// Convert span to vector (nvcc doesn't support implicit span→vector conversion).
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

/// Validate dtype for linalg ops: Float32, Float64, Float16, BFloat16 only.
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

/// Check matrix size limit for shared-memory fallback.
template<typename T>
void check_size_limit(int64_t n, const std::string& op_name) {
    constexpr int max_n = std::is_same_v<T, float> ? MAX_N_FLOAT : MAX_N_DOUBLE;
    if (n > max_n) {
        throw std::runtime_error(
            "linalg::" + op_name + ": matrix size " + std::to_string(n) +
            " exceeds native CUDA fallback limit of " + std::to_string(max_n) +
            " (build with cuSOLVER for larger matrices)");
    }
}

// ============================================================================
// LU decomposition kernel with partial pivoting (shared memory)
// One block per batch element. Thread 0 does pivoting, all threads cooperate
// on row eliminations. Shared memory layout: A[n*n] + pivot[n] (as T).
// ============================================================================

template<typename T>
__global__ void lu_kernel(
    T* __restrict__ data,
    int* __restrict__ pivots,
    int* __restrict__ info_out,
    int n)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* A = reinterpret_cast<T*>(smem_raw);
    // scratch[0] = pivot row, scratch[1] = swap flag
    T* scratch = A + n * n;

    T* batch_data = data + batch_idx * n * n;
    int* batch_pivots = pivots + batch_idx * n;

    // Load matrix into shared memory
    for (int idx = tid; idx < n * n; idx += num_threads) {
        A[idx] = batch_data[idx];
    }
    __syncthreads();

    int sign = 1;

    for (int k = 0; k < n; k++) {
        // Thread 0: find pivot (max abs in column k, rows k..n-1)
        if (tid == 0) {
            T max_val = fabs(A[k * n + k]);
            int max_row = k;
            for (int i = k + 1; i < n; i++) {
                T val = fabs(A[i * n + k]);
                if (val > max_val) {
                    max_val = val;
                    max_row = i;
                }
            }
            batch_pivots[k] = max_row + 1;  // 1-based (LAPACK convention)
            scratch[0] = static_cast<T>(max_row);
            if (max_val == T(0)) {
                // Singular matrix — record in info (1-based index)
                if (info_out) info_out[batch_idx] = k + 1;
            }
        }
        __syncthreads();

        int pivot_row = static_cast<int>(scratch[0]);

        // Swap rows k and pivot_row (parallel across columns)
        if (pivot_row != k) {
            for (int j = tid; j < n; j += num_threads) {
                T tmp = A[k * n + j];
                A[k * n + j] = A[pivot_row * n + j];
                A[pivot_row * n + j] = tmp;
            }
            if (tid == 0) sign = -sign;
            __syncthreads();
        }

        // Compute multipliers and eliminate (parallel across rows below pivot)
        T diag = A[k * n + k];
        if (diag != T(0)) {
            // Thread 0 computes multipliers
            if (tid == 0) {
                for (int i = k + 1; i < n; i++) {
                    A[i * n + k] /= diag;
                }
            }
            __syncthreads();

            // All threads apply elimination to remaining submatrix
            for (int i = k + 1 + tid; i < n; i += num_threads) {
                T mult = A[i * n + k];
                for (int j = k + 1; j < n; j++) {
                    A[i * n + j] -= mult * A[k * n + j];
                }
            }
            __syncthreads();
        }
    }

    // Write back to global memory
    for (int idx = tid; idx < n * n; idx += num_threads) {
        batch_data[idx] = A[idx];
    }
    if (tid == 0 && info_out) {
        // Store sign in info if no singularity was detected
        // We abuse a negative info to store the sign: 0 means positive sign, -1 means negative
        if (info_out[batch_idx] == 0) {
            info_out[batch_idx] = sign > 0 ? 0 : -1;
        }
    }
}

// ============================================================================
// Determinant from LU: product of diagonal * pivot sign
// ============================================================================

template<typename T>
__global__ void lu_det_kernel(
    const T* __restrict__ lu_data,
    const int* __restrict__ pivots,
    T* __restrict__ det_out,
    int n, int nbatch)
{
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbatch) return;

    const T* mat = lu_data + b * n * n;
    const int* piv = pivots + b * n;
    T d = T(1);
    for (int i = 0; i < n; i++) {
        d *= mat[i * n + i];
        if (piv[i] != i + 1) d = -d;
    }
    det_out[b] = d;
}

// ============================================================================
// LU-based solve kernel: forward and back substitution
// One block per batch element. Shared memory: LU[n*n] + B[n*nrhs] + pivot[n]
// ============================================================================

template<typename T>
__global__ void lu_solve_kernel(
    const T* __restrict__ lu_data,
    const int* __restrict__ pivots,
    T* __restrict__ b_data,
    int n, int nrhs)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* LU = reinterpret_cast<T*>(smem_raw);
    T* B = LU + n * n;

    const T* batch_lu = lu_data + batch_idx * n * n;
    const int* batch_piv = pivots + batch_idx * n;
    T* batch_b = b_data + batch_idx * n * nrhs;

    // Load LU and B into shared memory
    for (int idx = tid; idx < n * n; idx += num_threads) {
        LU[idx] = batch_lu[idx];
    }
    for (int idx = tid; idx < n * nrhs; idx += num_threads) {
        B[idx] = batch_b[idx];
    }
    __syncthreads();

    // Apply pivot permutation to B (thread 0, sequential)
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            int piv_row = batch_piv[i] - 1;  // convert to 0-based
            if (piv_row != i) {
                for (int j = 0; j < nrhs; j++) {
                    T tmp = B[i * nrhs + j];
                    B[i * nrhs + j] = B[piv_row * nrhs + j];
                    B[piv_row * nrhs + j] = tmp;
                }
            }
        }
    }
    __syncthreads();

    // Forward substitution: solve L*Y = P*B (thread 0, sequential per column)
    if (tid == 0) {
        for (int k = 0; k < n; k++) {
            for (int i = k + 1; i < n; i++) {
                T mult = LU[i * n + k];
                for (int j = 0; j < nrhs; j++) {
                    B[i * nrhs + j] -= mult * B[k * nrhs + j];
                }
            }
        }
    }
    __syncthreads();

    // Back substitution: solve U*X = Y (thread 0, sequential)
    if (tid == 0) {
        for (int k = n - 1; k >= 0; k--) {
            T diag = LU[k * n + k];
            for (int j = 0; j < nrhs; j++) {
                B[k * nrhs + j] /= diag;
            }
            for (int i = 0; i < k; i++) {
                T mult = LU[i * n + k];
                for (int j = 0; j < nrhs; j++) {
                    B[i * nrhs + j] -= mult * B[k * nrhs + j];
                }
            }
        }
    }
    __syncthreads();

    // Write B back
    for (int idx = tid; idx < n * nrhs; idx += num_threads) {
        batch_b[idx] = B[idx];
    }
}

// ============================================================================
// LU-based inverse kernel: solve A*X = I in shared memory
// One block per batch element.
// ============================================================================

template<typename T>
__global__ void lu_inv_kernel(
    const T* __restrict__ lu_data,
    const int* __restrict__ pivots,
    T* __restrict__ inv_out,
    int n)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* LU = reinterpret_cast<T*>(smem_raw);
    T* X = LU + n * n;  // n*n for identity/result

    const T* batch_lu = lu_data + batch_idx * n * n;
    const int* batch_piv = pivots + batch_idx * n;
    T* batch_inv = inv_out + batch_idx * n * n;

    // Load LU into shared memory
    for (int idx = tid; idx < n * n; idx += num_threads) {
        LU[idx] = batch_lu[idx];
    }
    // Initialize X = I
    for (int idx = tid; idx < n * n; idx += num_threads) {
        int row = idx / n;
        int col = idx % n;
        X[idx] = (row == col) ? T(1) : T(0);
    }
    __syncthreads();

    // Apply pivot permutation to identity (thread 0)
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            int piv_row = batch_piv[i] - 1;
            if (piv_row != i) {
                for (int j = 0; j < n; j++) {
                    T tmp = X[i * n + j];
                    X[i * n + j] = X[piv_row * n + j];
                    X[piv_row * n + j] = tmp;
                }
            }
        }

        // Forward substitution: L * Y = P * I
        for (int k = 0; k < n; k++) {
            for (int i = k + 1; i < n; i++) {
                T mult = LU[i * n + k];
                for (int j = 0; j < n; j++) {
                    X[i * n + j] -= mult * X[k * n + j];
                }
            }
        }

        // Back substitution: U * X_out = Y
        for (int k = n - 1; k >= 0; k--) {
            T diag = LU[k * n + k];
            for (int j = 0; j < n; j++) {
                X[k * n + j] /= diag;
            }
            for (int i = 0; i < k; i++) {
                T mult = LU[i * n + k];
                for (int j = 0; j < n; j++) {
                    X[i * n + j] -= mult * X[k * n + j];
                }
            }
        }
    }
    __syncthreads();

    // Write result
    for (int idx = tid; idx < n * n; idx += num_threads) {
        batch_inv[idx] = X[idx];
    }
}

// ============================================================================
// Cholesky factorization kernel (shared memory)
// One block per batch element.
// ============================================================================

template<typename T>
__global__ void cholesky_kernel(
    T* __restrict__ data,
    int n, bool upper)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* A = reinterpret_cast<T*>(smem_raw);

    T* batch_data = data + batch_idx * n * n;

    // Load matrix
    for (int idx = tid; idx < n * n; idx += num_threads) {
        A[idx] = batch_data[idx];
    }
    __syncthreads();

    // Cholesky factorization (thread 0 drives, sequential)
    if (tid == 0) {
        for (int j = 0; j < n; j++) {
            T sum = A[j * n + j];
            for (int k = 0; k < j; k++) {
                sum -= A[j * n + k] * A[j * n + k];
            }
            if (sum <= T(0)) {
                // Not positive definite — store what we have and let caller handle
                A[j * n + j] = T(0);
                continue;
            }
            A[j * n + j] = sqrt(sum);
            T diag = A[j * n + j];

            for (int i = j + 1; i < n; i++) {
                T s = A[i * n + j];
                for (int k = 0; k < j; k++) {
                    s -= A[i * n + k] * A[j * n + k];
                }
                A[i * n + j] = s / diag;
            }
        }
    }
    __syncthreads();

    // Zero the appropriate triangle and write back
    for (int idx = tid; idx < n * n; idx += num_threads) {
        int row = idx / n;
        int col = idx % n;
        if (upper) {
            // Transpose L to get U, zero lower
            if (row <= col) {
                batch_data[row * n + col] = A[col * n + row];  // L^T
            } else {
                batch_data[row * n + col] = T(0);
            }
        } else {
            // Keep L, zero upper
            if (row >= col) {
                batch_data[row * n + col] = A[row * n + col];
            } else {
                batch_data[row * n + col] = T(0);
            }
        }
    }
}

// ============================================================================
// Householder QR kernel (shared memory)
// One block per batch element. Computes Q and R.
// Shared memory: A[m*n] + Q[m*m] + scratch[4]
// ============================================================================

template<typename T>
__global__ void householder_qr_kernel(
    const T* __restrict__ A_in,
    T* __restrict__ Q_out,
    T* __restrict__ R_out,
    int m, int n_cols, int k)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* R = reinterpret_cast<T*>(smem_raw);
    T* Q = R + m * n_cols;
    T* scratch = Q + m * m;  // scratch[0]=v0, scratch[1]=tau

    const T* A = A_in + batch_idx * m * n_cols;
    T* Q_batch = Q_out + batch_idx * m * k;
    T* R_batch = R_out + batch_idx * k * n_cols;

    // Load A into R workspace
    for (int idx = tid; idx < m * n_cols; idx += num_threads) {
        R[idx] = A[idx];
    }
    // Initialize Q = I (m x m)
    for (int idx = tid; idx < m * m; idx += num_threads) {
        int row = idx / m;
        int col = idx % m;
        Q[idx] = (row == col) ? T(1) : T(0);
    }
    __syncthreads();

    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

    // Householder QR: for each column j, compute reflector
    for (int j = 0; j < k; j++) {
        // Thread 0 computes Householder parameters
        if (tid == 0) {
            T sigma = T(0);
            for (int i = j + 1; i < m; i++) {
                sigma += R[i * n_cols + j] * R[i * n_cols + j];
            }
            T x0 = R[j * n_cols + j];
            T norm_x = sqrt(x0 * x0 + sigma);
            if (norm_x < zero_tol || sigma < zero_tol) {
                scratch[1] = T(0);  // tau = 0, skip
            } else {
                T alpha = -copysign(norm_x, x0);
                T v0 = x0 - alpha;
                T v_norm_sq = v0 * v0 + sigma;
                scratch[0] = v0;
                scratch[1] = T(2) / v_norm_sq;  // tau
                scratch[2] = alpha;
            }
        }
        __syncthreads();

        T tau = scratch[1];
        if (tau == T(0)) {
            __syncthreads();
            continue;
        }
        T v0 = scratch[0];
        T alpha = scratch[2];

        // Apply reflector to R: R = (I - tau*v*v^T) * R
        // v = [v0, R[j+1][j], R[j+2][j], ...] (column j below diagonal)
        for (int col = j + static_cast<int>(tid); col < n_cols; col += num_threads) {
            T dot = v0 * R[j * n_cols + col];
            for (int i = j + 1; i < m; i++) {
                dot += R[i * n_cols + j] * R[i * n_cols + col];
            }
            dot *= tau;
            R[j * n_cols + col] -= v0 * dot;
            for (int i = j + 1; i < m; i++) {
                R[i * n_cols + col] -= R[i * n_cols + j] * dot;
            }
        }
        __syncthreads();

        // Accumulate Q = Q * (I - tau*v*v^T)
        for (int row = static_cast<int>(tid); row < m; row += num_threads) {
            T dot = v0 * Q[row * m + j];
            for (int i = j + 1; i < m; i++) {
                dot += R[i * n_cols + j] * Q[row * m + i];
            }
            dot *= tau;
            Q[row * m + j] -= v0 * dot;
            for (int i = j + 1; i < m; i++) {
                Q[row * m + i] -= R[i * n_cols + j] * dot;
            }
        }
        __syncthreads();

        // Clean up: set R diagonal and zero below
        if (tid == 0) {
            R[j * n_cols + j] = alpha;
            for (int i = j + 1; i < m; i++) {
                R[i * n_cols + j] = T(0);
            }
        }
        __syncthreads();
    }

    // Write Q (first k columns) and R (first k rows)
    for (int idx = tid; idx < m * k; idx += num_threads) {
        int row = idx / k;
        int col = idx % k;
        Q_batch[idx] = Q[row * m + col];
    }
    for (int idx = tid; idx < k * n_cols; idx += num_threads) {
        R_batch[idx] = R[idx];  // First k rows already contain R
    }
}

// ============================================================================
// Symmetric eigendecomposition (tridiagonal reduction + QL iteration)
// One block per batch element.
// Shared memory: A[n*n] + Q[n*n] + d[n] + e[n] + scratch[4]
// ============================================================================

template<typename T>
__global__ void eigh_kernel(
    T* __restrict__ data,
    T* __restrict__ eigenvalues_out,
    int n, int max_iterations)
{
    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* A = reinterpret_cast<T*>(smem_raw);
    T* Q = A + n * n;
    T* d = Q + n * n;       // diagonal
    T* e = d + n;           // subdiagonal
    T* scratch = e + n;

    T* batch_data = data + batch_idx * n * n;
    T* batch_eig = eigenvalues_out + batch_idx * n;

    // Load symmetric matrix and init Q = I
    for (int idx = tid; idx < n * n; idx += num_threads) {
        A[idx] = batch_data[idx];
        int row = idx / n;
        int col = idx % n;
        Q[idx] = (row == col) ? T(1) : T(0);
    }
    __syncthreads();

    // Step 1: Tridiagonalize via Householder reflections
    for (int k = 0; k + 2 < n; k++) {
        if (tid == 0) {
            T sigma = T(0);
            for (int i = k + 1; i < n; i++) {
                sigma += A[i * n + k] * A[i * n + k];
            }
            T norm_x = sqrt(sigma);
            if (norm_x < zero_tol) {
                scratch[1] = T(0);
            } else {
                T a = -copysign(norm_x, A[(k + 1) * n + k]);
                T v0_val = A[(k + 1) * n + k] - a;
                T v_norm_sq = v0_val * v0_val;
                for (int i = k + 2; i < n; i++) {
                    v_norm_sq += A[i * n + k] * A[i * n + k];
                }
                if (v_norm_sq < zero_tol) {
                    scratch[1] = T(0);
                } else {
                    scratch[0] = v0_val;
                    scratch[1] = T(2) / v_norm_sq;
                    scratch[2] = a;
                }
            }
        }
        __syncthreads();

        T tau = scratch[1];
        if (tau == T(0)) { __syncthreads(); continue; }
        T v0 = scratch[0];
        T alpha_val = scratch[2];

        // Symmetric: A = (I - tau*v*v^T) * A * (I - tau*v*v^T)
        // Left multiply: A = (I - tau*v*v^T) * A
        for (int j = k + static_cast<int>(tid); j < n; j += num_threads) {
            T dot = v0 * A[(k + 1) * n + j];
            for (int i = k + 2; i < n; i++) {
                dot += A[i * n + k] * A[i * n + j];
            }
            dot *= tau;
            A[(k + 1) * n + j] -= v0 * dot;
            for (int i = k + 2; i < n; i++) {
                A[i * n + j] -= A[i * n + k] * dot;
            }
        }
        __syncthreads();

        // Right multiply: A = A * (I - tau*v*v^T)
        for (int i = static_cast<int>(tid); i < n; i += num_threads) {
            T dot = v0 * A[i * n + (k + 1)];
            for (int j = k + 2; j < n; j++) {
                dot += A[j * n + k] * A[i * n + j];
            }
            dot *= tau;
            A[i * n + (k + 1)] -= v0 * dot;
            for (int j = k + 2; j < n; j++) {
                A[i * n + j] -= A[j * n + k] * dot;
            }
        }
        __syncthreads();

        // Accumulate Q
        for (int i = static_cast<int>(tid); i < n; i += num_threads) {
            T dot = v0 * Q[i * n + (k + 1)];
            for (int j = k + 2; j < n; j++) {
                dot += A[j * n + k] * Q[i * n + j];
            }
            dot *= tau;
            Q[i * n + (k + 1)] -= v0 * dot;
            for (int j = k + 2; j < n; j++) {
                Q[i * n + j] -= A[j * n + k] * dot;
            }
        }
        __syncthreads();

        // Clean up
        if (tid == 0) {
            A[(k + 1) * n + k] = alpha_val;
            A[k * n + (k + 1)] = alpha_val;
            for (int i = k + 2; i < n; i++) {
                A[i * n + k] = T(0);
                A[k * n + i] = T(0);
            }
        }
        __syncthreads();
    }

    // Extract diagonal and subdiagonal
    if (tid == 0) {
        for (int i = 0; i < n; i++) {
            d[i] = A[i * n + i];
        }
        for (int i = 0; i < n - 1; i++) {
            e[i] = A[(i + 1) * n + i];
        }
        e[n - 1] = T(0);
    }
    __syncthreads();

    // Step 2: QL iteration with implicit Wilkinson shifts on tridiagonal matrix
    if (tid == 0) {
        for (int l = 0; l < n; l++) {
            int iter = 0;
            while (iter < max_iterations) {
                // Find small subdiagonal element
                int m_idx = l;
                for (int i = l; i < n - 1; i++) {
                    T tst = fabs(d[i]) + fabs(d[i + 1]);
                    if (tst == T(0)) tst = T(1);
                    if (fabs(e[i]) < eps * tst) {
                        break;
                    }
                    m_idx = i + 1;
                }
                if (m_idx == l) break;  // converged

                // Wilkinson shift
                T g = (d[l + 1] - d[l]) / (T(2) * e[l]);
                T r = sqrt(g * g + T(1));
                T shift = d[m_idx] - d[l] + e[l] / (g + copysign(r, g));

                T s_val = T(1), c_val = T(1), p = T(0);

                for (int i = m_idx - 1; i >= l; i--) {
                    T f = s_val * e[i];
                    T b = c_val * e[i];

                    if (fabs(f) >= fabs(shift)) {
                        c_val = shift / f;
                        r = sqrt(c_val * c_val + T(1));
                        e[i + 1] = f * r;
                        s_val = T(1) / r;
                        c_val *= s_val;
                    } else {
                        s_val = f / shift;
                        r = sqrt(s_val * s_val + T(1));
                        e[i + 1] = shift * r;
                        c_val = T(1) / r;
                        s_val *= c_val;
                    }

                    shift = d[i + 1] - p;
                    r = (d[i] - shift) * s_val + T(2) * c_val * b;
                    p = s_val * r;
                    d[i + 1] = shift + p;
                    shift = c_val * r - b;

                    // Accumulate eigenvector rotations
                    for (int row = 0; row < n; row++) {
                        T tmp = Q[row * n + (i + 1)];
                        Q[row * n + (i + 1)] = s_val * Q[row * n + i] + c_val * tmp;
                        Q[row * n + i] = c_val * Q[row * n + i] - s_val * tmp;
                    }
                }

                d[l] -= p;
                e[l] = shift;
                e[m_idx] = T(0);
                iter++;
            }
        }
    }
    __syncthreads();

    // Write eigenvalues and eigenvectors
    for (int idx = tid; idx < n; idx += num_threads) {
        batch_eig[idx] = d[idx];
    }
    for (int idx = tid; idx < n * n; idx += num_threads) {
        batch_data[idx] = Q[idx];
    }
}

// ============================================================================
// Bidiagonalization + bidiagonal QR iteration for SVD
// One block per batch element.
// Shared memory: A[m*n] + U[m*m] + Vt[n*n] + scratch
// ============================================================================

template<typename T>
__global__ void svd_kernel(
    const T* __restrict__ A_in,
    T* __restrict__ U_out,
    T* __restrict__ S_out,
    T* __restrict__ Vt_out,
    int m, int n_cols, int k,
    bool full_matrices,
    int max_iterations)
{
    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* A = reinterpret_cast<T*>(smem_raw);
    T* U = A + m * n_cols;
    T* Vt = U + m * m;
    T* scratch = Vt + n_cols * n_cols;  // scratch[0..3]

    int u_rows = m;
    int u_cols = full_matrices ? m : k;
    int vt_rows = full_matrices ? n_cols : k;
    int vt_cols = n_cols;

    const T* batch_A = A_in + batch_idx * m * n_cols;
    T* batch_U = U_out + batch_idx * u_rows * u_cols;
    T* batch_S = S_out + batch_idx * k;
    T* batch_Vt = Vt_out + batch_idx * vt_rows * vt_cols;

    // Load A, init U = I(m x m), Vt = I(n x n)
    for (int idx = tid; idx < m * n_cols; idx += num_threads) {
        A[idx] = batch_A[idx];
    }
    for (int idx = tid; idx < m * m; idx += num_threads) {
        int r = idx / m, c = idx % m;
        U[idx] = (r == c) ? T(1) : T(0);
    }
    for (int idx = tid; idx < n_cols * n_cols; idx += num_threads) {
        int r = idx / n_cols, c = idx % n_cols;
        Vt[idx] = (r == c) ? T(1) : T(0);
    }
    __syncthreads();

    // Step 1: Bidiagonalization via Householder reflections
    for (int j = 0; j < k; j++) {
        // Left Householder: zero out column j below diagonal
        if (tid == 0) {
            T sigma = T(0);
            for (int i = j + 1; i < m; i++) {
                sigma += A[i * n_cols + j] * A[i * n_cols + j];
            }
            T x0 = A[j * n_cols + j];
            T norm_x = sqrt(x0 * x0 + sigma);
            if (norm_x < zero_tol || sigma < zero_tol) {
                scratch[1] = T(0);
            } else {
                T alpha = -copysign(norm_x, x0);
                T v0 = x0 - alpha;
                T v_sq = v0 * v0 + sigma;
                scratch[0] = v0;
                scratch[1] = T(2) / v_sq;
                scratch[2] = alpha;
            }
        }
        __syncthreads();

        T tau = scratch[1];
        if (tau != T(0)) {
            T v0 = scratch[0];
            T alpha = scratch[2];

            // A = (I - tau*v*v^T) * A
            for (int col = j + static_cast<int>(tid); col < n_cols; col += num_threads) {
                T dot = v0 * A[j * n_cols + col];
                for (int i = j + 1; i < m; i++) {
                    dot += A[i * n_cols + j] * A[i * n_cols + col];
                }
                dot *= tau;
                A[j * n_cols + col] -= v0 * dot;
                for (int i = j + 1; i < m; i++) {
                    A[i * n_cols + col] -= A[i * n_cols + j] * dot;
                }
            }
            __syncthreads();

            // Accumulate U = U * (I - tau*v*v^T)
            for (int row = static_cast<int>(tid); row < m; row += num_threads) {
                T dot = v0 * U[row * m + j];
                for (int i = j + 1; i < m; i++) {
                    dot += A[i * n_cols + j] * U[row * m + i];
                }
                dot *= tau;
                U[row * m + j] -= v0 * dot;
                for (int i = j + 1; i < m; i++) {
                    U[row * m + i] -= A[i * n_cols + j] * dot;
                }
            }
            __syncthreads();

            if (tid == 0) {
                A[j * n_cols + j] = alpha;
                for (int i = j + 1; i < m; i++) A[i * n_cols + j] = T(0);
            }
            __syncthreads();
        }

        // Right Householder: zero out row j to the right of superdiagonal
        if (j + 1 < n_cols) {
            if (tid == 0) {
                T sigma = T(0);
                for (int i = j + 2; i < n_cols; i++) {
                    sigma += A[j * n_cols + i] * A[j * n_cols + i];
                }
                T x0 = A[j * n_cols + (j + 1)];
                T norm_x = sqrt(x0 * x0 + sigma);
                if (norm_x < zero_tol || sigma < zero_tol) {
                    scratch[1] = T(0);
                } else {
                    T alpha = -copysign(norm_x, x0);
                    T v0 = x0 - alpha;
                    T v_sq = v0 * v0 + sigma;
                    scratch[0] = v0;
                    scratch[1] = T(2) / v_sq;
                    scratch[2] = alpha;
                }
            }
            __syncthreads();

            tau = scratch[1];
            if (tau != T(0)) {
                T v0 = scratch[0];
                T alpha = scratch[2];

                // A = A * (I - tau*v*v^T)
                for (int row = j + static_cast<int>(tid); row < m; row += num_threads) {
                    T dot = v0 * A[row * n_cols + (j + 1)];
                    for (int i = j + 2; i < n_cols; i++) {
                        dot += A[j * n_cols + i] * A[row * n_cols + i];
                    }
                    dot *= tau;
                    A[row * n_cols + (j + 1)] -= v0 * dot;
                    for (int i = j + 2; i < n_cols; i++) {
                        A[row * n_cols + i] -= A[j * n_cols + i] * dot;
                    }
                }
                __syncthreads();

                // Accumulate Vt: Vt = (I - tau*v*v^T) * Vt
                for (int col = static_cast<int>(tid); col < n_cols; col += num_threads) {
                    T dot = v0 * Vt[(j + 1) * n_cols + col];
                    for (int i = j + 2; i < n_cols; i++) {
                        dot += A[j * n_cols + i] * Vt[i * n_cols + col];
                    }
                    dot *= tau;
                    Vt[(j + 1) * n_cols + col] -= v0 * dot;
                    for (int i = j + 2; i < n_cols; i++) {
                        Vt[i * n_cols + col] -= A[j * n_cols + i] * dot;
                    }
                }
                __syncthreads();

                if (tid == 0) {
                    A[j * n_cols + (j + 1)] = alpha;
                    for (int i = j + 2; i < n_cols; i++) A[j * n_cols + i] = T(0);
                }
                __syncthreads();
            }
        }
    }

    // Step 2: Bidiagonal QR iteration (thread 0 drives, Givens rotations)
    if (tid == 0) {
        // Extract bidiagonal elements: diagonal d[0..k-1], superdiagonal e[0..k-2]
        // Reuse scratch area + some of A for temporary storage
        // We'll work in-place on A's diagonal/superdiagonal
        for (int iter = 0; iter < max_iterations * k; iter++) {
            // Find active range [p, q] where e[i] != 0
            int q = k - 1;
            while (q > 0 && fabs(A[(q - 1) * n_cols + q]) < eps * (fabs(A[(q - 1) * n_cols + (q - 1)]) + fabs(A[q * n_cols + q]))) {
                q--;
            }
            if (q == 0) break;  // all converged

            int p = q - 1;
            while (p > 0 && fabs(A[(p - 1) * n_cols + p]) >= eps * (fabs(A[(p - 1) * n_cols + (p - 1)]) + fabs(A[p * n_cols + p]))) {
                p--;
            }

            // Wilkinson shift from trailing 2x2 of B^T*B
            T d_q = A[q * n_cols + q];
            T d_qm1 = A[(q - 1) * n_cols + (q - 1)];
            T e_qm1 = A[(q - 1) * n_cols + q];
            T mu;
            {
                T t11 = d_qm1 * d_qm1;
                if (q - 2 >= p) t11 += A[(q - 2) * n_cols + (q - 1)] * A[(q - 2) * n_cols + (q - 1)];
                T t22 = d_q * d_q + e_qm1 * e_qm1;
                T t12 = d_qm1 * e_qm1;
                T trace = t11 + t22;
                T det = t11 * t22 - t12 * t12;
                T disc = trace * trace - T(4) * det;
                if (disc < T(0)) disc = T(0);
                T sq = sqrt(disc);
                T l1 = T(0.5) * (trace + sq);
                T l2 = T(0.5) * (trace - sq);
                mu = (fabs(l1 - t22) < fabs(l2 - t22)) ? l1 : l2;
            }

            // Chase the bulge
            T d_p = A[p * n_cols + p];
            T y = d_p * d_p - mu;
            T z = d_p * A[p * n_cols + (p + 1)];

            for (int i = p; i < q; i++) {
                // Right Givens rotation to zero z (acts on columns i, i+1)
                T r = sqrt(y * y + z * z);
                T c_val = y / r;
                T s_val = -z / r;

                // Apply to A columns i, i+1
                for (int row = 0; row < k; row++) {
                    if (row >= i - 1 && row <= i + 1) {
                        T a1 = A[row * n_cols + i];
                        T a2 = A[row * n_cols + (i + 1)];
                        A[row * n_cols + i] = c_val * a1 - s_val * a2;
                        A[row * n_cols + (i + 1)] = s_val * a1 + c_val * a2;
                    }
                }
                // Apply to Vt rows i, i+1
                for (int col = 0; col < n_cols; col++) {
                    T v1 = Vt[i * n_cols + col];
                    T v2 = Vt[(i + 1) * n_cols + col];
                    Vt[i * n_cols + col] = c_val * v1 - s_val * v2;
                    Vt[(i + 1) * n_cols + col] = s_val * v1 + c_val * v2;
                }

                // Left Givens rotation to re-zero the sub-subdiagonal
                y = A[i * n_cols + i];
                z = A[(i + 1) * n_cols + i];
                r = sqrt(y * y + z * z);
                c_val = y / r;
                s_val = -z / r;

                // Apply to A rows i, i+1
                for (int col = 0; col < n_cols; col++) {
                    if (col >= i && col <= i + 2) {
                        T a1 = A[i * n_cols + col];
                        T a2 = A[(i + 1) * n_cols + col];
                        A[i * n_cols + col] = c_val * a1 - s_val * a2;
                        A[(i + 1) * n_cols + col] = s_val * a1 + c_val * a2;
                    }
                }
                // Apply to U columns i, i+1
                for (int row = 0; row < m; row++) {
                    T u1 = U[row * m + i];
                    T u2 = U[row * m + (i + 1)];
                    U[row * m + i] = c_val * u1 - s_val * u2;
                    U[row * m + (i + 1)] = s_val * u1 + c_val * u2;
                }

                if (i + 1 < q) {
                    y = A[i * n_cols + (i + 1)];
                    z = A[i * n_cols + (i + 2)];
                }
            }
        }

        // Make singular values positive (flip sign of U column if needed)
        for (int i = 0; i < k; i++) {
            if (A[i * n_cols + i] < T(0)) {
                A[i * n_cols + i] = -A[i * n_cols + i];
                for (int row = 0; row < m; row++) {
                    U[row * m + i] = -U[row * m + i];
                }
            }
        }

        // Sort singular values in descending order (simple selection sort)
        for (int i = 0; i < k - 1; i++) {
            int max_idx = i;
            T max_val = A[i * n_cols + i];
            for (int j = i + 1; j < k; j++) {
                if (A[j * n_cols + j] > max_val) {
                    max_val = A[j * n_cols + j];
                    max_idx = j;
                }
            }
            if (max_idx != i) {
                // Swap singular values
                T tmp = A[i * n_cols + i];
                A[i * n_cols + i] = A[max_idx * n_cols + max_idx];
                A[max_idx * n_cols + max_idx] = tmp;
                // Swap U columns
                for (int row = 0; row < m; row++) {
                    T t = U[row * m + i];
                    U[row * m + i] = U[row * m + max_idx];
                    U[row * m + max_idx] = t;
                }
                // Swap Vt rows
                for (int col = 0; col < n_cols; col++) {
                    T t = Vt[i * n_cols + col];
                    Vt[i * n_cols + col] = Vt[max_idx * n_cols + col];
                    Vt[max_idx * n_cols + col] = t;
                }
            }
        }
    }
    __syncthreads();

    // Write outputs
    for (int idx = tid; idx < k; idx += num_threads) {
        batch_S[idx] = A[idx * n_cols + idx];
    }
    // U output: full_matrices ? m*m : m*k
    int u_out_size = u_rows * u_cols;
    for (int idx = tid; idx < u_out_size; idx += num_threads) {
        int r = idx / u_cols;
        int c = idx % u_cols;
        batch_U[idx] = U[r * m + c];
    }
    // Vt output: full_matrices ? n*n : k*n
    int vt_out_size = vt_rows * vt_cols;
    for (int idx = tid; idx < vt_out_size; idx += num_threads) {
        int r = idx / vt_cols;
        int c = idx % vt_cols;
        batch_Vt[idx] = Vt[r * n_cols + c];
    }
}

// ============================================================================
// Non-symmetric eigendecomposition: Hessenberg + Francis QR
// Ported from the ROCm qr_eig_fallback_kernel.
// One block per batch element.
// ============================================================================

template<typename T>
__global__ void qr_eig_fallback_kernel(
    const T* __restrict__ A_in,
    T* __restrict__ wr_out,
    T* __restrict__ wi_out,
    T* __restrict__ V_out,
    int n, int max_iterations)
{
    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

    uint32_t tid = threadIdx.x;
    uint32_t num_threads = blockDim.x;
    uint32_t batch_idx = blockIdx.x;

    extern __shared__ char smem_raw[];
    T* H = reinterpret_cast<T*>(smem_raw);
    T* Q = H + n * n;
    T* scratch = Q + n * n;  // scratch[0]=v0, scratch[1]=tau, scratch[2]=alpha, scratch[3]=flag

    const T* A = A_in + batch_idx * n * n;
    T* wr = wr_out + batch_idx * n;
    T* wi = wi_out + batch_idx * n;
    T* V = V_out + batch_idx * n * n;

    // Load matrix and init Q = I
    for (int idx = static_cast<int>(tid); idx < n * n; idx += static_cast<int>(num_threads)) {
        H[idx] = A[idx];
        int row = idx / n;
        int col = idx % n;
        Q[idx] = (row == col) ? T(1) : T(0);
    }
    __syncthreads();

    // Step 1: Reduce to upper Hessenberg form via Householder reflections
    for (int k = 0; k + 2 < n; k++) {
        if (tid == 0) {
            T sigma = T(0);
            for (int i = k + 1; i < n; i++) {
                sigma += H[i * n + k] * H[i * n + k];
            }
            T norm_x = sqrt(sigma);
            if (norm_x < zero_tol) {
                scratch[1] = T(0);
            } else {
                T a = -copysign(norm_x, H[(k + 1) * n + k]);
                T v0_val = H[(k + 1) * n + k] - a;
                T v_norm_sq = v0_val * v0_val;
                for (int i = k + 2; i < n; i++) {
                    v_norm_sq += H[i * n + k] * H[i * n + k];
                }
                if (v_norm_sq < zero_tol) {
                    scratch[1] = T(0);
                } else {
                    scratch[0] = v0_val;
                    scratch[1] = T(2) / v_norm_sq;
                    scratch[2] = a;
                }
            }
        }
        __syncthreads();

        T tau = scratch[1];
        if (tau == T(0)) { __syncthreads(); continue; }
        T v0 = scratch[0];
        T alpha = scratch[2];

        // Left: H = (I - tau*v*v^T) * H
        for (int j = k + static_cast<int>(tid); j < n; j += static_cast<int>(num_threads)) {
            T dot = v0 * H[(k + 1) * n + j];
            for (int i = k + 2; i < n; i++) {
                dot += H[i * n + k] * H[i * n + j];
            }
            dot *= tau;
            H[(k + 1) * n + j] -= v0 * dot;
            for (int i = k + 2; i < n; i++) {
                H[i * n + j] -= H[i * n + k] * dot;
            }
        }
        __syncthreads();

        // Right: H = H * (I - tau*v*v^T)
        for (int i = static_cast<int>(tid); i < n; i += static_cast<int>(num_threads)) {
            T dot = v0 * H[i * n + (k + 1)];
            for (int j = k + 2; j < n; j++) {
                dot += H[j * n + k] * H[i * n + j];
            }
            dot *= tau;
            H[i * n + (k + 1)] -= v0 * dot;
            for (int j = k + 2; j < n; j++) {
                H[i * n + j] -= H[j * n + k] * dot;
            }
        }
        __syncthreads();

        // Accumulate Q
        for (int i = static_cast<int>(tid); i < n; i += static_cast<int>(num_threads)) {
            T dot = v0 * Q[i * n + (k + 1)];
            for (int j = k + 2; j < n; j++) {
                dot += H[j * n + k] * Q[i * n + j];
            }
            dot *= tau;
            Q[i * n + (k + 1)] -= v0 * dot;
            for (int j = k + 2; j < n; j++) {
                Q[i * n + j] -= H[j * n + k] * dot;
            }
        }
        __syncthreads();

        if (tid == 0) {
            H[(k + 1) * n + k] = alpha;
            for (int i = k + 2; i < n; i++) {
                H[i * n + k] = T(0);
            }
        }
        __syncthreads();
    }

    // Step 2: QR iteration with implicit double shifts
    if (tid == 0) {
        scratch[3] = static_cast<T>(n);
    }
    __syncthreads();

    for (int iter = 0; iter < max_iterations; iter++) {
        int nn = static_cast<int>(scratch[3]);
        if (nn <= 0) break;

        // Deflation checks (thread 0)
        if (tid == 0) {
            bool deflated = false;

            if (nn >= 2) {
                T tst = fabs(H[(nn - 2) * n + (nn - 2)]) + fabs(H[(nn - 1) * n + (nn - 1)]);
                if (tst == T(0)) tst = T(1);
                if (fabs(H[(nn - 1) * n + (nn - 2)]) < eps * tst) {
                    wr[nn - 1] = H[(nn - 1) * n + (nn - 1)];
                    wi[nn - 1] = T(0);
                    H[(nn - 1) * n + (nn - 2)] = T(0);
                    scratch[3] = static_cast<T>(nn - 1);
                    deflated = true;
                }
            }

            if (!deflated && nn >= 3) {
                T tst = fabs(H[(nn - 3) * n + (nn - 3)]) + fabs(H[(nn - 2) * n + (nn - 2)]);
                if (tst == T(0)) tst = T(1);
                if (fabs(H[(nn - 2) * n + (nn - 3)]) < eps * tst) {
                    T a = H[(nn - 2) * n + (nn - 2)], b = H[(nn - 2) * n + (nn - 1)];
                    T c = H[(nn - 1) * n + (nn - 2)], d = H[(nn - 1) * n + (nn - 1)];
                    T trace = a + d;
                    T det = a * d - b * c;
                    T disc = trace * trace - T(4) * det;

                    if (disc >= T(0)) {
                        T sq = sqrt(disc);
                        wr[nn - 2] = T(0.5) * (trace + sq);
                        wi[nn - 2] = T(0);
                        wr[nn - 1] = T(0.5) * (trace - sq);
                        wi[nn - 1] = T(0);
                    } else {
                        T sq = sqrt(-disc);
                        wr[nn - 2] = T(0.5) * trace;
                        wi[nn - 2] = T(0.5) * sq;
                        wr[nn - 1] = T(0.5) * trace;
                        wi[nn - 1] = T(-0.5) * sq;
                    }

                    H[(nn - 2) * n + (nn - 3)] = T(0);
                    scratch[3] = static_cast<T>(nn - 2);
                    deflated = true;
                }
            }

            if (!deflated && nn == 1) {
                wr[0] = H[0];
                wi[0] = T(0);
                scratch[3] = T(0);
                deflated = true;
            }

            scratch[2] = deflated ? T(1) : T(0);
        }
        __syncthreads();

        if (scratch[2] != T(0)) continue;

        nn = static_cast<int>(scratch[3]);

        // Francis double-shift QR step
        T x, y, z;
        if (tid == 0) {
            T s = H[(nn - 2) * n + (nn - 2)] + H[(nn - 1) * n + (nn - 1)];
            T t = H[(nn - 2) * n + (nn - 2)] * H[(nn - 1) * n + (nn - 1)] -
                  H[(nn - 2) * n + (nn - 1)] * H[(nn - 1) * n + (nn - 2)];

            scratch[0] = H[0] * H[0] + H[1] * H[n] - s * H[0] + t;
            scratch[1] = H[n] * (H[0] + H[n + 1] - s);
            scratch[2] = (nn > 2) ? H[n] * H[2 * n + 1] : T(0);
        }
        __syncthreads();
        x = scratch[0]; y = scratch[1]; z = scratch[2];

        for (int k = 0; k + 2 < nn; k++) {
            T norm_v = sqrt(x * x + y * y + z * z);
            if (norm_v < zero_tol) {
                if (tid == 0) {
                    x = H[(k + 1) * n + k];
                    y = (k + 2 < nn) ? H[(k + 2) * n + k] : T(0);
                    z = (k + 3 < nn) ? H[(k + 3) * n + k] : T(0);
                    scratch[0] = x; scratch[1] = y; scratch[2] = z;
                }
                __syncthreads();
                x = scratch[0]; y = scratch[1]; z = scratch[2];
                continue;
            }

            T alpha_h = -copysign(norm_v, x);
            T v0 = x - alpha_h;
            T v1 = y;
            T v2 = z;
            T v_sq = v0 * v0 + v1 * v1 + v2 * v2;
            if (v_sq < zero_tol) {
                if (tid == 0) {
                    x = H[(k + 1) * n + k];
                    y = (k + 2 < nn) ? H[(k + 2) * n + k] : T(0);
                    z = (k + 3 < nn) ? H[(k + 3) * n + k] : T(0);
                    scratch[0] = x; scratch[1] = y; scratch[2] = z;
                }
                __syncthreads();
                x = scratch[0]; y = scratch[1]; z = scratch[2];
                continue;
            }
            T tau_h = T(2) / v_sq;
            int m_lim = (k + 4 < nn) ? k + 4 : nn;

            // Left reflection
            for (int j = k + static_cast<int>(tid); j < nn; j += static_cast<int>(num_threads)) {
                T dot = v0 * H[k * n + j] + v1 * H[(k + 1) * n + j];
                if (k + 2 < nn) dot += v2 * H[(k + 2) * n + j];
                dot *= tau_h;
                H[k * n + j] -= v0 * dot;
                H[(k + 1) * n + j] -= v1 * dot;
                if (k + 2 < nn) H[(k + 2) * n + j] -= v2 * dot;
            }
            __syncthreads();

            // Right reflection
            for (int i = static_cast<int>(tid); i < m_lim; i += static_cast<int>(num_threads)) {
                T dot = v0 * H[i * n + k] + v1 * H[i * n + (k + 1)];
                if (k + 2 < nn) dot += v2 * H[i * n + (k + 2)];
                dot *= tau_h;
                H[i * n + k] -= v0 * dot;
                H[i * n + (k + 1)] -= v1 * dot;
                if (k + 2 < nn) H[i * n + (k + 2)] -= v2 * dot;
            }
            __syncthreads();

            // Accumulate Q
            for (int i = static_cast<int>(tid); i < n; i += static_cast<int>(num_threads)) {
                T dot = v0 * Q[i * n + k] + v1 * Q[i * n + (k + 1)];
                if (k + 2 < nn) dot += v2 * Q[i * n + (k + 2)];
                dot *= tau_h;
                Q[i * n + k] -= v0 * dot;
                Q[i * n + (k + 1)] -= v1 * dot;
                if (k + 2 < nn) Q[i * n + (k + 2)] -= v2 * dot;
            }
            __syncthreads();

            if (tid == 0) {
                if (k > 0) H[k * n + (k - 1)] = alpha_h;
                if (k + 2 < nn) H[(k + 2) * n + k] = T(0);
                if (k + 3 < nn) H[(k + 3) * n + k] = T(0);

                x = H[(k + 1) * n + k];
                y = (k + 2 < nn) ? H[(k + 2) * n + k] : T(0);
                z = (k + 3 < nn) ? H[(k + 3) * n + k] : T(0);
                scratch[0] = x; scratch[1] = y; scratch[2] = z;
            }
            __syncthreads();
            x = scratch[0]; y = scratch[1]; z = scratch[2];
        }

        // Final 2x2 Givens rotation
        if (nn >= 2) {
            T norm_v = sqrt(x * x + y * y);
            if (norm_v > zero_tol) {
                int k = nn - 2;
                T c_val = x / norm_v;
                T s_val = -y / norm_v;

                for (int j = k + static_cast<int>(tid); j < nn; j += static_cast<int>(num_threads)) {
                    T tmp = c_val * H[k * n + j] - s_val * H[(k + 1) * n + j];
                    H[(k + 1) * n + j] = s_val * H[k * n + j] + c_val * H[(k + 1) * n + j];
                    H[k * n + j] = tmp;
                }
                __syncthreads();

                for (int i = static_cast<int>(tid); i < nn; i += static_cast<int>(num_threads)) {
                    T tmp = c_val * H[i * n + k] - s_val * H[i * n + (k + 1)];
                    H[i * n + (k + 1)] = s_val * H[i * n + k] + c_val * H[i * n + (k + 1)];
                    H[i * n + k] = tmp;
                }
                __syncthreads();

                for (int i = static_cast<int>(tid); i < n; i += static_cast<int>(num_threads)) {
                    T tmp = c_val * Q[i * n + k] - s_val * Q[i * n + (k + 1)];
                    Q[i * n + (k + 1)] = s_val * Q[i * n + k] + c_val * Q[i * n + (k + 1)];
                    Q[i * n + k] = tmp;
                }
                __syncthreads();
            }
        }
    }

    // Handle remaining blocks
    if (tid == 0) {
        int nn = static_cast<int>(scratch[3]);
        if (nn == 1) {
            wr[0] = H[0];
            wi[0] = T(0);
        } else if (nn == 2) {
            T a = H[0], b = H[1], c = H[n], d = H[n + 1];
            T trace = a + d;
            T det = a * d - b * c;
            T disc = trace * trace - T(4) * det;
            if (disc >= T(0)) {
                T sq = sqrt(disc);
                wr[0] = T(0.5) * (trace + sq);
                wi[0] = T(0);
                wr[1] = T(0.5) * (trace - sq);
                wi[1] = T(0);
            } else {
                T sq = sqrt(-disc);
                wr[0] = T(0.5) * trace;
                wi[0] = T(0.5) * sq;
                wr[1] = T(0.5) * trace;
                wi[1] = T(-0.5) * sq;
            }
        }
    }
    __syncthreads();

    // Write eigenvector matrix Q to output
    for (int idx = static_cast<int>(tid); idx < n * n; idx += static_cast<int>(num_threads)) {
        V[idx] = Q[idx];
    }
}

} // anonymous namespace

// ============================================================================
// Public API implementations (match cuSOLVER wrapper signatures)
// ============================================================================

// ============================================================================
// Determinant
// ============================================================================

auto linalg_det_kernel(const Tensor& A, cudaStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "det");
    if (A.dtype() == DType::Float16) {
        return linalg_det_kernel(A.to(DType::Float32), stream).to(DType::Float16);
    }
    if (A.dtype() == DType::BFloat16) {
        return linalg_det_kernel(A.to(DType::Float32), stream).to(DType::BFloat16);
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) out_shape.push_back(shape[i]);
    if (out_shape.empty()) out_shape.push_back(1);

    auto result = zeros(out_shape, A.dtype(), A.device());

    // Allocate pivots and info on device
    int* d_pivots = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK_LINALG(cudaMalloc(&d_pivots, nbatch * n * sizeof(int)));
    CUDA_CHECK_LINALG(cudaMalloc(&d_info, nbatch * sizeof(int)));
    CUDA_CHECK_LINALG(cudaMemset(d_info, 0, nbatch * sizeof(int)));

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "det");
        size_t smem = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        int det_threads = 256;
        int det_blocks = (nbatch + det_threads - 1) / det_threads;
        lu_det_kernel<float><<<det_blocks, det_threads, 0, stream>>>(
            work.data<float>(), d_pivots, result.data<float>(), n, nbatch);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "det");
        size_t smem = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        int det_threads = 256;
        int det_blocks = (nbatch + det_threads - 1) / det_threads;
        lu_det_kernel<double><<<det_blocks, det_threads, 0, stream>>>(
            work.data<double>(), d_pivots, result.data<double>(), n, nbatch);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    cudaFree(d_pivots);
    cudaFree(d_info);
    return result;
}

// ============================================================================
// Matrix Inverse
// ============================================================================

auto linalg_inv_kernel(const Tensor& A, cudaStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "inv");
    if (A.dtype() == DType::Float16) {
        return linalg_inv_kernel(A.to(DType::Float32), stream);
    }
    if (A.dtype() == DType::BFloat16) {
        return linalg_inv_kernel(A.to(DType::Float32), stream);
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto result = zeros(to_vec(work.shape()), A.dtype(), A.device());

    int* d_pivots = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK_LINALG(cudaMalloc(&d_pivots, nbatch * n * sizeof(int)));
    CUDA_CHECK_LINALG(cudaMalloc(&d_info, nbatch * sizeof(int)));
    CUDA_CHECK_LINALG(cudaMemset(d_info, 0, nbatch * sizeof(int)));

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "inv");
        // LU factorize
        size_t smem_lu = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem_lu, stream>>>(
            work.data<float>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        // Invert via LU solve with identity
        size_t smem_inv = 2 * n * n * sizeof(float);
        lu_inv_kernel<float><<<nbatch, threads, smem_inv, stream>>>(
            work.data<float>(), d_pivots, result.data<float>(), n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "inv");
        size_t smem_lu = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem_lu, stream>>>(
            work.data<double>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        size_t smem_inv = 2 * n * n * sizeof(double);
        lu_inv_kernel<double><<<nbatch, threads, smem_inv, stream>>>(
            work.data<double>(), d_pivots, result.data<double>(), n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    cudaFree(d_pivots);
    cudaFree(d_info);
    return result;
}

// ============================================================================
// Linear System Solve (AX = B)
// ============================================================================

auto linalg_solve_kernel(const Tensor& A, const Tensor& B, cudaStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "solve");
    if (A.dtype() == DType::Float16) {
        return linalg_solve_kernel(A.to(DType::Float32), B.to(DType::Float32), stream);
    }
    if (A.dtype() == DType::BFloat16) {
        return linalg_solve_kernel(A.to(DType::Float32), B.to(DType::Float32), stream);
    }

    auto work_a = A.contiguous().clone();
    auto work_b = B.contiguous().clone();
    auto [n, ndim_a] = check_square(work_a);
    int64_t nbatch = batch_size(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;

    int* d_pivots = nullptr;
    int* d_info = nullptr;
    CUDA_CHECK_LINALG(cudaMalloc(&d_pivots, nbatch * n * sizeof(int)));
    CUDA_CHECK_LINALG(cudaMalloc(&d_info, nbatch * sizeof(int)));
    CUDA_CHECK_LINALG(cudaMemset(d_info, 0, nbatch * sizeof(int)));

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "solve");
        // LU factorize
        size_t smem_lu = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem_lu, stream>>>(
            work_a.data<float>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        // Solve via forward/back substitution
        size_t smem_solve = (n * n + n * nrhs) * sizeof(float);
        lu_solve_kernel<float><<<nbatch, threads, smem_solve, stream>>>(
            work_a.data<float>(), d_pivots, work_b.data<float>(), n, nrhs);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "solve");
        size_t smem_lu = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem_lu, stream>>>(
            work_a.data<double>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        size_t smem_solve = (n * n + n * nrhs) * sizeof(double);
        lu_solve_kernel<double><<<nbatch, threads, smem_solve, stream>>>(
            work_a.data<double>(), d_pivots, work_b.data<double>(), n, nrhs);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    cudaFree(d_pivots);
    cudaFree(d_info);
    return work_b;
}

// ============================================================================
// SVD (Singular Value Decomposition)
// ============================================================================

auto linalg_svd_kernel(const Tensor& A, bool full_matrices, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "svd");
    if (A.dtype() == DType::Float16) {
        auto [U, S, Vt] = linalg_svd_kernel(A.to(DType::Float32), full_matrices, stream);
        return {U, S, Vt};  // Keep as Float32
    }
    if (A.dtype() == DType::BFloat16) {
        auto [U, S, Vt] = linalg_svd_kernel(A.to(DType::Float32), full_matrices, stream);
        return {U, S, Vt};
    }

    auto work = A.contiguous().clone();
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::svd: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

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

    auto U = zeros(u_shape, A.dtype(), A.device());
    auto S = zeros(s_shape, A.dtype(), A.device());
    auto Vt = zeros(vt_shape, A.dtype(), A.device());

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(std::max(m, n_cols), "svd");
        // Shared memory: A[m*n] + U[m*m] + Vt[n*n] + scratch[4]
        size_t smem = (m * n_cols + m * m + n_cols * n_cols + 4) * sizeof(float);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        svd_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), U.data<float>(), S.data<float>(), Vt.data<float>(),
            m, n_cols, k, full_matrices, 30 * std::max(m, n_cols));
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(std::max(m, n_cols), "svd");
        size_t smem = (m * n_cols + m * m + n_cols * n_cols + 4) * sizeof(double);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        svd_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), U.data<double>(), S.data<double>(), Vt.data<double>(),
            m, n_cols, k, full_matrices, 30 * std::max(m, n_cols));
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {U, S, Vt};
}

// ============================================================================
// QR Decomposition
// ============================================================================

auto linalg_qr_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    validate_linalg_dtype(A, "qr");
    if (A.dtype() == DType::Float16) {
        auto [Q, R] = linalg_qr_kernel(A.to(DType::Float32), stream);
        return {Q, R};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [Q, R] = linalg_qr_kernel(A.to(DType::Float32), stream);
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

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(std::max(m, n_cols), "qr");
        // Shared memory: R[m*n] + Q[m*m] + scratch[4]
        size_t smem = (m * n_cols + m * m + 4) * sizeof(float);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_qr_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), Q.data<float>(), R.data<float>(), m, n_cols, k);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(std::max(m, n_cols), "qr");
        size_t smem = (m * n_cols + m * m + 4) * sizeof(double);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_qr_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), Q.data<double>(), R.data<double>(), m, n_cols, k);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {Q, R};
}

// ============================================================================
// Symmetric Eigendecomposition (eigh)
// ============================================================================

auto linalg_eigh_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    validate_linalg_dtype(A, "eigh");
    if (A.dtype() == DType::Float16) {
        auto [W, V] = linalg_eigh_kernel(A.to(DType::Float32), stream);
        return {W, V};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [W, V] = linalg_eigh_kernel(A.to(DType::Float32), stream);
        return {W, V};
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto W = zeros(w_shape, A.dtype(), A.device());

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "eigh");
        // Shared memory: A[n*n] + Q[n*n] + d[n] + e[n] + scratch[4]
        size_t smem = (2 * n * n + 2 * n + 4) * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        eigh_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), W.data<float>(), n, 30 * n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "eigh");
        size_t smem = (2 * n * n + 2 * n + 4) * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        eigh_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), W.data<double>(), n, 30 * n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {W, work};
}

// ============================================================================
// Non-symmetric Eigendecomposition (eig)
// ============================================================================

auto linalg_eig_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "eig");
    if (A.dtype() == DType::Float16) {
        auto [wr, wi, V] = linalg_eig_kernel(A.to(DType::Float32), stream);
        return {wr, wi, V};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [wr, wi, V] = linalg_eig_kernel(A.to(DType::Float32), stream);
        return {wr, wi, V};
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    std::vector<int64_t> batch_dims;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) batch_dims.push_back(shape[i]);

    std::vector<int64_t> w_shape = batch_dims;
    w_shape.push_back(n);
    auto WR = zeros(w_shape, A.dtype(), A.device());
    auto WI = zeros(w_shape, A.dtype(), A.device());

    std::vector<int64_t> v_shape = batch_dims;
    v_shape.push_back(n);
    v_shape.push_back(n);
    auto V = zeros(v_shape, A.dtype(), A.device());

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "eig");
        // Shared memory: H[n*n] + Q[n*n] + scratch[4]
        size_t smem = (2 * n * n + 4) * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        qr_eig_fallback_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), WR.data<float>(), WI.data<float>(),
            V.data<float>(), n, 30 * n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "eig");
        size_t smem = (2 * n * n + 4) * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        qr_eig_fallback_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), WR.data<double>(), WI.data<double>(),
            V.data<double>(), n, 30 * n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {WR, WI, V};
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

auto linalg_cholesky_kernel(const Tensor& A, bool upper, cudaStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "cholesky");
    if (A.dtype() == DType::Float16) {
        return linalg_cholesky_kernel(A.to(DType::Float32), upper, stream);
    }
    if (A.dtype() == DType::BFloat16) {
        return linalg_cholesky_kernel(A.to(DType::Float32), upper, stream);
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "cholesky");
        size_t smem = n * n * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        cholesky_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), n, upper);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "cholesky");
        size_t smem = n * n * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        cholesky_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), n, upper);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work;
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUSOLVER
