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
#include "../cusolver_handle_pool.hpp"

// Forward-declare zeros to avoid including creation.hpp (which pulls in
// loader.hpp using std::expected, unsupported by nvcc)
namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}
#include <cusolverDn.h>
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
struct DeviceInt {
    int* ptr = nullptr;
    DeviceInt() { CUDA_CHECK_LINALG(cudaMalloc(&ptr, sizeof(int))); }
    ~DeviceInt() { if (ptr) cudaFree(ptr); }
    DeviceInt(const DeviceInt&) = delete;
    DeviceInt& operator=(const DeviceInt&) = delete;
};

/// RAII wrapper for device workspace.
struct DeviceWorkspace {
    void* ptr = nullptr;
    DeviceWorkspace(size_t bytes) {
        if (bytes > 0) CUDA_CHECK_LINALG(cudaMalloc(&ptr, bytes));
    }
    ~DeviceWorkspace() { if (ptr) cudaFree(ptr); }
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

    // Allocate pivot array and info on device
    int* d_ipiv = nullptr;
    CUDA_CHECK_LINALG(cudaMalloc(&d_ipiv, nbatch * n * sizeof(int)));
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

    cudaFree(d_ipiv);
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

    int* d_ipiv = nullptr;
    CUDA_CHECK_LINALG(cudaMalloc(&d_ipiv, n * sizeof(int)));
    DeviceInt d_info;

    // Create identity matrix on device for getrs-based inversion
    auto identity = zeros(to_vec(work.shape()), A.dtype(), A.device());

    if (A.dtype() == DType::Float32) {
        float* data = work.data<float>();
        float* id_data = identity.data<float>();

        // Set identity matrix (copy to host, set, copy back per batch)
        auto id_cpu = zeros(to_vec(work.shape()), A.dtype(), Device::cpu());
        float* id_cpu_data = id_cpu.data<float>();
        for (int64_t b = 0; b < nbatch; b++) {
            for (int64_t i = 0; i < n; i++) {
                id_cpu_data[b * n * n + i * n + i] = 1.0f;
            }
        }
        CUDA_CHECK_LINALG(cudaMemcpy(id_data, id_cpu_data,
            nbatch * n * n * sizeof(float), cudaMemcpyHostToDevice));

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

        auto id_cpu = zeros(to_vec(work.shape()), A.dtype(), Device::cpu());
        double* id_cpu_data = id_cpu.data<double>();
        for (int64_t b = 0; b < nbatch; b++) {
            for (int64_t i = 0; i < n; i++) {
                id_cpu_data[b * n * n + i * n + i] = 1.0;
            }
        }
        CUDA_CHECK_LINALG(cudaMemcpy(id_data, id_cpu_data,
            nbatch * n * n * sizeof(double), cudaMemcpyHostToDevice));

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

    cudaFree(d_ipiv);
    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
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

    int* d_ipiv = nullptr;
    CUDA_CHECK_LINALG(cudaMalloc(&d_ipiv, n * sizeof(int)));
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

    cudaFree(d_ipiv);
    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
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
            int lda = m;  // leading dim in col-major = number of rows of col-major = m
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

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
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

        // Allocate tau on device
        float* d_tau = nullptr;
        CUDA_CHECK_LINALG(cudaMalloc(&d_tau, k * sizeof(float)));

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

        cudaFree(d_tau);
    } else {
        double* a_data = work.data<double>();
        double* q_data = Q.data<double>();
        double* r_data = R.data<double>();

        double* d_tau = nullptr;
        CUDA_CHECK_LINALG(cudaMalloc(&d_tau, k * sizeof(double)));

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

        cudaFree(d_tau);
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

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    // work now contains eigenvectors (columns of orthogonal matrix)
    return {W, work};
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

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUSOLVER
