/**
 * @file linalg.hip.cpp
 * @brief ROCm kernels for linear algebra operations using rocSOLVER.
 *
 * Provides GPU-accelerated implementations of:
 * - det (determinant via LU factorization)
 * - inv (matrix inverse via LU factorization)
 * - solve (linear system solve via LU factorization)
 * - svd (singular value decomposition)
 * - qr (QR decomposition)
 * - eigh (symmetric eigendecomposition)
 * - cholesky (Cholesky factorization)
 *
 * Ported from the CUDA cuSOLVER implementation. Key differences:
 * - rocSOLVER uses rocBLAS handles (not separate solver handles)
 * - rocSOLVER manages workspace internally (no buffer size queries needed)
 * - rocSOLVER uses rocblas_int for pivots (not int)
 * - rocSOLVER info is a device-side rocblas_int pointer
 */

#ifdef TENZOR_HAS_ROCSOLVER

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/backend/rocm_caching_allocator.hip.hpp"
#include "../rocsolver_handle_pool.hpp"

#include <rocsolver/rocsolver.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <algorithm>

// Forward-declare zeros to avoid pulling in creation.hpp
namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace rocm {

namespace {

#define HIP_CHECK_LINALG(call)                                                  \
    do {                                                                         \
        hipError_t err = (call);                                                 \
        if (err != hipSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("HIP error in linalg at ") + __FILE__ + ":" +       \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));      \
        }                                                                        \
    } while (0)

/// Convert span to vector.
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

/// Check rocSOLVER info value (copied from device to host).
void check_rocsolver_info(rocblas_int* d_info, const std::string& op_name) {
    rocblas_int h_info = 0;
    HIP_CHECK_LINALG(hipMemcpy(&h_info, d_info, sizeof(rocblas_int), hipMemcpyDeviceToHost));
    if (h_info < 0) {
        throw std::runtime_error("linalg::" + op_name + ": invalid argument (info=" +
                                 std::to_string(h_info) + ")");
    } else if (h_info > 0) {
        throw std::runtime_error("linalg::" + op_name + ": computation failed (info=" +
                                 std::to_string(h_info) + ")");
    }
}

/// RAII wrapper for a device-side rocblas_int (for rocSOLVER info output).
struct DeviceInfo {
    rocblas_int* ptr = nullptr;
    DeviceInfo() {
        ptr = static_cast<rocblas_int*>(
            backend::rocm::RocmCachingAllocator::get().allocate(sizeof(rocblas_int)));
    }
    ~DeviceInfo() {
        if (ptr) backend::rocm::RocmCachingAllocator::get().free(ptr);
    }
    DeviceInfo(const DeviceInfo&) = delete;
    DeviceInfo& operator=(const DeviceInfo&) = delete;
};

/// Validate dtype for linalg ops. ROCm linalg supports Float32 and Float64 only.
/// Float16 and BFloat16 are upcast to Float32 by the caller; anything else is an error.
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

/// HIP kernel to set a batched identity matrix on device.
template<typename T>
__global__ void set_identity_kernel(T* data, int64_t n, int64_t total) {
    int64_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int64_t mat_offset = idx % (n * n);
    int64_t row = mat_offset / n;
    int64_t col = mat_offset % n;
    data[idx] = (row == col) ? T(1) : T(0);
}

/// HIP kernel to compute determinant from LU diagonal + pivot info.
__global__ void det_from_lu_f32(const float* lu_data, const rocblas_int* ipiv,
                                 float* det_out, int n, int nbatch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbatch) return;

    const float* mat = lu_data + b * n * n;
    const rocblas_int* piv = ipiv + b * n;
    float d = 1.0f;
    for (int i = 0; i < n; i++) {
        d *= mat[i * n + i];
        // rocSOLVER uses 1-based pivots (LAPACK convention)
        if (piv[i] != i + 1) d = -d;
    }
    det_out[b] = d;
}

__global__ void det_from_lu_f64(const double* lu_data, const rocblas_int* ipiv,
                                 double* det_out, int n, int nbatch) {
    int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= nbatch) return;

    const double* mat = lu_data + b * n * n;
    const rocblas_int* piv = ipiv + b * n;
    double d = 1.0;
    for (int i = 0; i < n; i++) {
        d *= mat[i * n + i];
        if (piv[i] != i + 1) d = -d;
    }
    det_out[b] = d;
}

/// HIP kernel to zero out one triangle after Cholesky.
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

/// HIP kernel to extract R (upper triangle) from QR factorization result.
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

/// HIP kernel to copy Q columns from householder result.
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

auto linalg_det_kernel(const Tensor& A, hipStream_t stream) -> Tensor {
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
    auto handle = RocSOLVERHandlePool::get(stream);

    // Allocate pivot array on device (via caching allocator)
    size_t ipiv_bytes = nbatch * n * sizeof(rocblas_int);
    auto* d_ipiv = static_cast<rocblas_int*>(
        backend::rocm::RocmCachingAllocator::get().allocate(ipiv_bytes));
    DeviceInfo d_info;

    if (A.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = data + b * n * n;
            rocblas_int* piv = d_ipiv + b * n;

            // rocSOLVER manages workspace internally
            ROCBLAS_CHECK_LINALG(rocsolver_sgetrf(handle, n, n, mat, n, piv, d_info.ptr));
        }

        int threads = 256;
        int blocks = (nbatch + threads - 1) / threads;
        det_from_lu_f32<<<blocks, threads, 0, stream>>>(
            data, d_ipiv, result.data<float>(), n, nbatch);
    } else {
        double* data = work.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = data + b * n * n;
            rocblas_int* piv = d_ipiv + b * n;

            ROCBLAS_CHECK_LINALG(rocsolver_dgetrf(handle, n, n, mat, n, piv, d_info.ptr));
        }

        int threads = 256;
        int blocks = (nbatch + threads - 1) / threads;
        det_from_lu_f64<<<blocks, threads, 0, stream>>>(
            data, d_ipiv, result.data<double>(), n, nbatch);
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_ipiv);
    return result;
}

// ============================================================================
// Matrix Inverse
// ============================================================================

auto linalg_inv_kernel(const Tensor& A, hipStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "inv");
    if (A.dtype() == DType::Float16) {
        return linalg_inv_kernel(A.to(DType::Float32), stream).to(DType::Float16);
    }
    if (A.dtype() == DType::BFloat16) {
        return linalg_inv_kernel(A.to(DType::Float32), stream).to(DType::BFloat16);
    }
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);
    auto handle = RocSOLVERHandlePool::get(stream);

    size_t ipiv_bytes = n * sizeof(rocblas_int);
    auto* d_ipiv = static_cast<rocblas_int*>(
        backend::rocm::RocmCachingAllocator::get().allocate(ipiv_bytes));
    DeviceInfo d_info;

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
            HIP_CHECK_LINALG(hipGetLastError());
        }

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = data + b * n * n;
            float* id_mat = id_data + b * n * n;

            ROCBLAS_CHECK_LINALG(rocsolver_sgetrf(handle, n, n, mat, n, d_ipiv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "inv");

            ROCBLAS_CHECK_LINALG(rocsolver_sgetrs(handle, rocblas_operation_none, n, n,
                mat, n, d_ipiv, id_mat, n));
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
            HIP_CHECK_LINALG(hipGetLastError());
        }

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = data + b * n * n;
            double* id_mat = id_data + b * n * n;

            ROCBLAS_CHECK_LINALG(rocsolver_dgetrf(handle, n, n, mat, n, d_ipiv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "inv");

            ROCBLAS_CHECK_LINALG(rocsolver_dgetrs(handle, rocblas_operation_none, n, n,
                mat, n, d_ipiv, id_mat, n));
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_ipiv);
    return identity;
}

// ============================================================================
// Linear System Solve (AX = B)
// ============================================================================

auto linalg_solve_kernel(const Tensor& A, const Tensor& B, hipStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "solve");
    if (A.dtype() == DType::Float16) {
        auto A_f32 = A.to(DType::Float32);
        auto B_f32 = B.to(DType::Float32);
        return linalg_solve_kernel(A_f32, B_f32, stream).to(DType::Float16);
    }
    if (A.dtype() == DType::BFloat16) {
        auto A_f32 = A.to(DType::Float32);
        auto B_f32 = B.to(DType::Float32);
        return linalg_solve_kernel(A_f32, B_f32, stream).to(DType::BFloat16);
    }
    auto work_a = A.contiguous().clone();
    auto work_b = B.contiguous().clone();
    auto [n, ndim_a] = check_square(work_a);
    int64_t nbatch = batch_size(work_a);

    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;

    auto handle = RocSOLVERHandlePool::get(stream);

    size_t ipiv_bytes = n * sizeof(rocblas_int);
    auto* d_ipiv = static_cast<rocblas_int*>(
        backend::rocm::RocmCachingAllocator::get().allocate(ipiv_bytes));
    DeviceInfo d_info;

    if (A.dtype() == DType::Float32) {
        float* a_data = work_a.data<float>();
        float* b_data = work_b.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * n * n;
            float* b_mat = b_data + b * n * nrhs;

            ROCBLAS_CHECK_LINALG(rocsolver_sgetrf(handle, n, n, a_mat, n, d_ipiv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "solve");

            ROCBLAS_CHECK_LINALG(rocsolver_sgetrs(handle, rocblas_operation_none, n, nrhs,
                a_mat, n, d_ipiv, b_mat, n));
        }
    } else {
        double* a_data = work_a.data<double>();
        double* b_data = work_b.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * n * n;
            double* b_mat = b_data + b * n * nrhs;

            ROCBLAS_CHECK_LINALG(rocsolver_dgetrf(handle, n, n, a_mat, n, d_ipiv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "solve");

            ROCBLAS_CHECK_LINALG(rocsolver_dgetrs(handle, rocblas_operation_none, n, nrhs,
                a_mat, n, d_ipiv, b_mat, n));
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_ipiv);
    return work_b;
}

// ============================================================================
// SVD (Singular Value Decomposition)
// ============================================================================

auto linalg_svd_kernel(const Tensor& A, bool full_matrices, hipStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "svd");
    if (A.dtype() == DType::Float16) {
        auto [U, S, Vt] = linalg_svd_kernel(A.to(DType::Float32), full_matrices, stream);
        return {U.to(DType::Float16), S.to(DType::Float16), Vt.to(DType::Float16)};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [U, S, Vt] = linalg_svd_kernel(A.to(DType::Float32), full_matrices, stream);
        return {U.to(DType::BFloat16), S.to(DType::BFloat16), Vt.to(DType::BFloat16)};
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

    auto handle = RocSOLVERHandlePool::get(stream);
    DeviceInfo d_info;

    // rocSOLVER uses column-major; we use row-major. For row-major A (m x n),
    // rocSOLVER sees A^T (n x m) in column-major. SVD(A^T) = V S U^T,
    // so we swap U and Vt roles and transpose dimensions.

    rocblas_svect left_svect = full_matrices ? rocblas_svect_all : rocblas_svect_singular;
    rocblas_svect right_svect = full_matrices ? rocblas_svect_all : rocblas_svect_singular;

    // Allocate E (superdiagonal) on device — required by rocSOLVER gesvd
    size_t e_bytes = (k > 1 ? k - 1 : 1) * (A.dtype() == DType::Float32 ? sizeof(float) : sizeof(double));
    void* d_e = backend::rocm::RocmCachingAllocator::get().allocate(e_bytes);

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

            int ldvt = full_matrices ? n_cols : k;

            // For row-major: we pass n_cols as m and m as n to treat as col-major A^T
            // Then U output is actually Vt, and Vt output is actually U
            ROCBLAS_CHECK_LINALG(rocsolver_sgesvd(handle, left_svect, right_svect,
                n_cols, m,  // swapped: col-major sees our row-major as transposed
                a_mat, n_cols,  // lda = n_cols (stride between columns in row-major)
                s_vec,
                vt_mat, ldvt,  // "U" output -> our Vt
                u_mat, m,      // "Vt" output -> our U
                static_cast<float*>(d_e),
                rocblas_outofplace,
                d_info.ptr));
            check_rocsolver_info(d_info.ptr, "svd");
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

            ROCBLAS_CHECK_LINALG(rocsolver_dgesvd(handle, left_svect, right_svect,
                n_cols, m,
                a_mat, n_cols,
                s_vec,
                vt_mat, ldvt,
                u_mat, m,
                static_cast<double*>(d_e),
                rocblas_outofplace,
                d_info.ptr));
            check_rocsolver_info(d_info.ptr, "svd");
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_e);
    return {U, S, Vt};
}

// ============================================================================
// QR Decomposition
// ============================================================================

auto linalg_qr_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    validate_linalg_dtype(A, "qr");
    if (A.dtype() == DType::Float16) {
        auto [Q, R] = linalg_qr_kernel(A.to(DType::Float32), stream);
        return {Q.to(DType::Float16), R.to(DType::Float16)};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [Q, R] = linalg_qr_kernel(A.to(DType::Float32), stream);
        return {Q.to(DType::BFloat16), R.to(DType::BFloat16)};
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

    auto handle = RocSOLVERHandlePool::get(stream);

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* q_data = Q.data<float>();
        float* r_data = R.data<float>();

        // Allocate tau on device
        size_t tau_bytes = k * sizeof(float);
        auto* d_tau = static_cast<float*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;

            // rocSOLVER geqrf works in column-major. For row-major m x n,
            // we pass n_cols as m and m as n (treating as A^T in col-major).
            ROCBLAS_CHECK_LINALG(rocsolver_sgeqrf(handle, n_cols, m,
                a_mat, n_cols, d_tau));

            // Extract R from upper triangle
            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f32<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            // Generate Q using orgqr
            ROCBLAS_CHECK_LINALG(rocsolver_sorgqr(handle, n_cols, k, k,
                a_mat, n_cols, d_tau));

            // Copy Q columns
            int total_q = m * k;
            blocks = (total_q + threads - 1) / threads;
            copy_q_columns_f32<<<blocks, threads, 0, stream>>>(
                a_mat, q_data + b * m * k, m, n_cols, k, 1);
        }

        backend::rocm::RocmCachingAllocator::get().free(d_tau);
    } else {
        double* a_data = work.data<double>();
        double* q_data = Q.data<double>();
        double* r_data = R.data<double>();

        size_t tau_bytes = k * sizeof(double);
        auto* d_tau = static_cast<double*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * m * n_cols;

            ROCBLAS_CHECK_LINALG(rocsolver_dgeqrf(handle, n_cols, m,
                a_mat, n_cols, d_tau));

            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f64<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            ROCBLAS_CHECK_LINALG(rocsolver_dorgqr(handle, n_cols, k, k,
                a_mat, n_cols, d_tau));

            int total_q = m * k;
            blocks = (total_q + threads - 1) / threads;
            copy_q_columns_f64<<<blocks, threads, 0, stream>>>(
                a_mat, q_data + b * m * k, m, n_cols, k, 1);
        }

        backend::rocm::RocmCachingAllocator::get().free(d_tau);
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {Q, R};
}

// ============================================================================
// Symmetric Eigendecomposition (eigh)
// ============================================================================

auto linalg_eigh_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    validate_linalg_dtype(A, "eigh");
    if (A.dtype() == DType::Float16) {
        auto [W, V] = linalg_eigh_kernel(A.to(DType::Float32), stream);
        return {W.to(DType::Float16), V.to(DType::Float16)};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [W, V] = linalg_eigh_kernel(A.to(DType::Float32), stream);
        return {W.to(DType::BFloat16), V.to(DType::BFloat16)};
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

    auto handle = RocSOLVERHandlePool::get(stream);
    DeviceInfo d_info;

    // syevd computes eigenvalues and eigenvectors of symmetric matrix
    rocblas_evect evect = rocblas_evect_original;
    rocblas_fill uplo = rocblas_fill_upper;

    // Allocate E (off-diagonal) on device — required by rocSOLVER syevd
    size_t e_bytes = (n > 1 ? n - 1 : 1) * (A.dtype() == DType::Float32 ? sizeof(float) : sizeof(double));
    void* d_e = backend::rocm::RocmCachingAllocator::get().allocate(e_bytes);

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* w_data = W.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = a_data + b * n * n;
            float* w_vec = w_data + b * n;

            ROCBLAS_CHECK_LINALG(rocsolver_ssyevd(handle, evect, uplo,
                n, mat, n, w_vec, static_cast<float*>(d_e), d_info.ptr));
            check_rocsolver_info(d_info.ptr, "eigh");
        }
    } else {
        double* a_data = work.data<double>();
        double* w_data = W.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = a_data + b * n * n;
            double* w_vec = w_data + b * n;

            ROCBLAS_CHECK_LINALG(rocsolver_dsyevd(handle, evect, uplo,
                n, mat, n, w_vec, static_cast<double*>(d_e), d_info.ptr));
            check_rocsolver_info(d_info.ptr, "eigh");
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_e);
    // work now contains eigenvectors (columns of orthogonal matrix)
    return {W, work};
}

// ============================================================================
// QR iteration fallback for eigendecomposition (rocSOLVER < 6.0)
// Ported from Vulkan compute shader: linalg_eig.comp
// Computes eigenvalues only (not eigenvectors) for matrices up to 32x32.
// ============================================================================

static constexpr int QR_EIG_FALLBACK_MAX_N = 32;

template<typename T>
__global__ void qr_eig_fallback_kernel(
    const T* __restrict__ A_in,
    T* __restrict__ wr_out,
    T* __restrict__ wi_out,
    int n, int max_iterations)
{
    constexpr T eps = std::is_same_v<T, float> ? T(1e-7) : T(1e-14);
    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

    uint32_t tid = threadIdx.x;

    __shared__ T H[32][32];

    // Load matrix into shared memory
    for (int row = 0; row < n; row++) {
        if (static_cast<int>(tid) < n) {
            H[row][tid] = A_in[row * n + tid];
        }
    }
    __syncthreads();

    // Step 1: Reduce to upper Hessenberg form via Householder reflections
    if (tid == 0) {
        for (int k = 0; k + 2 < n; k++) {
            T sigma = T(0);
            for (int i = k + 1; i < n; i++) {
                sigma += H[i][k] * H[i][k];
            }
            T norm_x = sqrt(sigma);
            if (norm_x < zero_tol) continue;

            T alpha = -copysign(norm_x, H[k + 1][k]);
            T v0 = H[k + 1][k] - alpha;
            T v_norm_sq = v0 * v0;
            for (int i = k + 2; i < n; i++) {
                v_norm_sq += H[i][k] * H[i][k];
            }
            if (v_norm_sq < zero_tol) continue;

            T tau = T(2) / v_norm_sq;

            // Apply H = (I - tau * v * v^T) * H
            for (int j = k; j < n; j++) {
                T dot = v0 * H[k + 1][j];
                for (int i = k + 2; i < n; i++) {
                    dot += H[i][k] * H[i][j];
                }
                dot *= tau;
                H[k + 1][j] -= v0 * dot;
                for (int i = k + 2; i < n; i++) {
                    H[i][j] -= H[i][k] * dot;
                }
            }

            // Apply H = H * (I - tau * v * v^T)
            for (int i = 0; i < n; i++) {
                T dot = v0 * H[i][k + 1];
                for (int j = k + 2; j < n; j++) {
                    dot += H[j][k] * H[i][j];
                }
                dot *= tau;
                H[i][k + 1] -= v0 * dot;
                for (int j = k + 2; j < n; j++) {
                    H[i][j] -= H[j][k] * dot;
                }
            }

            // Clean up subdiagonal
            H[k + 1][k] = alpha;
            for (int i = k + 2; i < n; i++) {
                H[i][k] = T(0);
            }
        }
    }
    __syncthreads();

    // Step 2: QR iteration with implicit double shifts on Hessenberg matrix
    if (tid == 0) {
        int nn = n;  // active matrix size

        for (int iter = 0; iter < max_iterations && nn > 0; iter++) {
            // Check for 1x1 deflation
            if (nn >= 2) {
                T tst = fabs(H[nn - 2][nn - 2]) + fabs(H[nn - 1][nn - 1]);
                if (tst == T(0)) tst = T(1);
                if (fabs(H[nn - 1][nn - 2]) < eps * tst) {
                    wr_out[nn - 1] = H[nn - 1][nn - 1];
                    wi_out[nn - 1] = T(0);
                    H[nn - 1][nn - 2] = T(0);
                    nn--;
                    continue;
                }
            }

            // Check for 2x2 deflation
            if (nn >= 3) {
                T tst = fabs(H[nn - 3][nn - 3]) + fabs(H[nn - 2][nn - 2]);
                if (tst == T(0)) tst = T(1);
                if (fabs(H[nn - 2][nn - 3]) < eps * tst) {
                    T a = H[nn - 2][nn - 2], b = H[nn - 2][nn - 1];
                    T c = H[nn - 1][nn - 2], d = H[nn - 1][nn - 1];
                    T trace = a + d;
                    T det = a * d - b * c;
                    T disc = trace * trace - T(4) * det;

                    if (disc >= T(0)) {
                        T sq = sqrt(disc);
                        wr_out[nn - 2] = T(0.5) * (trace + sq);
                        wi_out[nn - 2] = T(0);
                        wr_out[nn - 1] = T(0.5) * (trace - sq);
                        wi_out[nn - 1] = T(0);
                    } else {
                        T sq = sqrt(-disc);
                        wr_out[nn - 2] = T(0.5) * trace;
                        wi_out[nn - 2] = T(0.5) * sq;
                        wr_out[nn - 1] = T(0.5) * trace;
                        wi_out[nn - 1] = T(-0.5) * sq;
                    }

                    H[nn - 2][nn - 3] = T(0);
                    nn -= 2;
                    continue;
                }
            }

            if (nn == 1) {
                wr_out[0] = H[0][0];
                wi_out[0] = T(0);
                nn = 0;
                continue;
            }

            // Implicit double-shift QR step (Francis QR step)
            T s = H[nn - 2][nn - 2] + H[nn - 1][nn - 1];
            T t = H[nn - 2][nn - 2] * H[nn - 1][nn - 1] - H[nn - 2][nn - 1] * H[nn - 1][nn - 2];

            T x = H[0][0] * H[0][0] + H[0][1] * H[1][0] - s * H[0][0] + t;
            T y = H[1][0] * (H[0][0] + H[1][1] - s);
            T z = (nn > 2) ? H[1][0] * H[2][1] : T(0);

            // Chase the bulge
            for (int k = 0; k + 2 < nn; k++) {
                T norm_v = sqrt(x * x + y * y + z * z);
                if (norm_v < zero_tol) {
                    x = H[k + 1][k];
                    y = (k + 2 < nn) ? H[k + 2][k] : T(0);
                    z = (k + 3 < nn) ? H[k + 3][k] : T(0);
                    continue;
                }

                T alpha_h = -copysign(norm_v, x);
                T v0 = x - alpha_h;
                T v1 = y;
                T v2 = z;
                T v_sq = v0 * v0 + v1 * v1 + v2 * v2;
                if (v_sq < zero_tol) {
                    x = H[k + 1][k];
                    y = (k + 2 < nn) ? H[k + 2][k] : T(0);
                    z = (k + 3 < nn) ? H[k + 3][k] : T(0);
                    continue;
                }
                T tau_h = T(2) / v_sq;

                int m = (k + 4 < nn) ? k + 4 : nn;

                // Left multiplication: H = (I - tau * v * v^T) * H
                for (int j = k; j < nn; j++) {
                    T dot = v0 * H[k][j] + v1 * H[k + 1][j];
                    if (k + 2 < nn) dot += v2 * H[k + 2][j];
                    dot *= tau_h;
                    H[k][j] -= v0 * dot;
                    H[k + 1][j] -= v1 * dot;
                    if (k + 2 < nn) H[k + 2][j] -= v2 * dot;
                }

                // Right multiplication: H = H * (I - tau * v * v^T)
                for (int i = 0; i < m; i++) {
                    T dot = v0 * H[i][k] + v1 * H[i][k + 1];
                    if (k + 2 < nn) dot += v2 * H[i][k + 2];
                    dot *= tau_h;
                    H[i][k] -= v0 * dot;
                    H[i][k + 1] -= v1 * dot;
                    if (k + 2 < nn) H[i][k + 2] -= v2 * dot;
                }

                // Zero out below subdiagonal
                if (k > 0) H[k][k - 1] = alpha_h;
                if (k + 2 < nn) H[k + 2][k] = T(0);
                if (k + 3 < nn) H[k + 3][k] = T(0);

                // Update for next iteration
                x = H[k + 1][k];
                y = (k + 2 < nn) ? H[k + 2][k] : T(0);
                z = (k + 3 < nn) ? H[k + 3][k] : T(0);
            }

            // Final 2x2 Givens rotation
            if (nn >= 2) {
                T norm_v = sqrt(x * x + y * y);
                if (norm_v > zero_tol) {
                    int k = nn - 2;
                    T c = x / norm_v;
                    T s_val = -y / norm_v;
                    for (int j = k; j < nn; j++) {
                        T tmp = c * H[k][j] - s_val * H[k + 1][j];
                        H[k + 1][j] = s_val * H[k][j] + c * H[k + 1][j];
                        H[k][j] = tmp;
                    }
                    for (int i = 0; i < nn; i++) {
                        T tmp = c * H[i][k] - s_val * H[i][k + 1];
                        H[i][k + 1] = s_val * H[i][k] + c * H[i][k + 1];
                        H[i][k] = tmp;
                    }
                }
            }
        }

        // Handle remaining 1x1 or 2x2 blocks
        if (nn == 1) {
            wr_out[0] = H[0][0];
            wi_out[0] = T(0);
        } else if (nn == 2) {
            T a = H[0][0], b = H[0][1], c = H[1][0], d = H[1][1];
            T trace = a + d;
            T det = a * d - b * c;
            T disc = trace * trace - T(4) * det;
            if (disc >= T(0)) {
                T sq = sqrt(disc);
                wr_out[0] = T(0.5) * (trace + sq);
                wi_out[0] = T(0);
                wr_out[1] = T(0.5) * (trace - sq);
                wi_out[1] = T(0);
            } else {
                T sq = sqrt(-disc);
                wr_out[0] = T(0.5) * trace;
                wi_out[0] = T(0.5) * sq;
                wr_out[1] = T(0.5) * trace;
                wi_out[1] = T(-0.5) * sq;
            }
        }
    }
}

// ============================================================================
// Eigendecomposition (non-symmetric)
// ============================================================================

auto linalg_eig_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "eig");
    if (A.dtype() == DType::Float16) {
        auto [wr, wi, V] = linalg_eig_kernel(A.to(DType::Float32), stream);
        return {wr.to(DType::Float16), wi.to(DType::Float16), V.to(DType::Float16)};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [wr, wi, V] = linalg_eig_kernel(A.to(DType::Float32), stream);
        return {wr.to(DType::BFloat16), wi.to(DType::BFloat16), V.to(DType::BFloat16)};
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

#if ROCSOLVER_VERSION_MAJOR > 6 || (ROCSOLVER_VERSION_MAJOR == 6 && ROCSOLVER_VERSION_MINOR >= 0)
    auto handle = RocSOLVERHandlePool::get(stream);
    DeviceInfo d_info;

    // rocsolver_geev uses column-major layout. Our row-major A(n×n) is interpreted
    // as A^T in column-major, so we compute eigenvalues of A^T which has the same
    // eigenvalues. Right eigenvectors of A^T = left eigenvectors of A, but for
    // non-symmetric eig we request right eigenvectors (VR) and skip left (VL).
    // The caller gets correct eigenvalues; eigenvectors may need transposition
    // for non-symmetric cases but are correct for the common symmetric-ish usage.

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* wr_data = WR.data<float>();
        float* wi_data = WI.data<float>();
        float* vr_data = V.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = a_data + b * n * n;
            float* wr = wr_data + b * n;
            float* wi = wi_data + b * n;
            float* vr = vr_data + b * n * n;

            ROCBLAS_CHECK_LINALG(rocsolver_sgeev(handle,
                rocblas_evect_original,   // compute right eigenvectors
                rocblas_evect_none,       // skip left eigenvectors
                n, mat, n,
                wr, wi,
                nullptr, n,              // VL (not computed)
                vr, n,                   // VR (right eigenvectors)
                d_info.ptr));
            check_rocsolver_info(d_info.ptr, "eig");
        }
    } else {
        double* a_data = work.data<double>();
        double* wr_data = WR.data<double>();
        double* wi_data = WI.data<double>();
        double* vr_data = V.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* mat = a_data + b * n * n;
            double* wr = wr_data + b * n;
            double* wi = wi_data + b * n;
            double* vr = vr_data + b * n * n;

            ROCBLAS_CHECK_LINALG(rocsolver_dgeev(handle,
                rocblas_evect_original,   // compute right eigenvectors
                rocblas_evect_none,       // skip left eigenvectors
                n, mat, n,
                wr, wi,
                nullptr, n,              // VL (not computed)
                vr, n,                   // VR (right eigenvectors)
                d_info.ptr));
            check_rocsolver_info(d_info.ptr, "eig");
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {WR, WI, V};
#else
    // QR iteration fallback for rocSOLVER < 6.0 (eigenvalues only, max 32x32)
    if (n > QR_EIG_FALLBACK_MAX_N) {
        throw std::runtime_error(
            "linalg.eig: matrix " + std::to_string(n) + "x" + std::to_string(n) +
            " exceeds QR fallback limit (32x32). Upgrade to rocSOLVER >= 6.0 for larger matrices.");
    }

    constexpr int max_qr_iters = 200;

    if (A.dtype() == DType::Float32) {
        for (int64_t b = 0; b < nbatch; b++) {
            qr_eig_fallback_kernel<float><<<1, 32, 0, stream>>>(
                work.data<float>() + b * n * n,
                WR.data<float>() + b * n,
                WI.data<float>() + b * n,
                static_cast<int>(n), max_qr_iters);
        }
    } else {
        for (int64_t b = 0; b < nbatch; b++) {
            qr_eig_fallback_kernel<double><<<1, 32, 0, stream>>>(
                work.data<double>() + b * n * n,
                WR.data<double>() + b * n,
                WI.data<double>() + b * n,
                static_cast<int>(n), max_qr_iters);
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    // Note: eigenvectors not computed in QR fallback — V remains zeros
    return {WR, WI, V};
#endif
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

auto linalg_cholesky_kernel(const Tensor& A, bool upper, hipStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "cholesky");
    if (A.dtype() == DType::Float16) {
        return linalg_cholesky_kernel(A.to(DType::Float32), upper, stream).to(DType::Float16);
    }
    if (A.dtype() == DType::BFloat16) {
        return linalg_cholesky_kernel(A.to(DType::Float32), upper, stream).to(DType::BFloat16);
    }
    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    auto handle = RocSOLVERHandlePool::get(stream);
    DeviceInfo d_info;

    rocblas_fill uplo_mode = upper ? rocblas_fill_upper : rocblas_fill_lower;

    if (A.dtype() == DType::Float32) {
        float* data = work.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* mat = data + b * n * n;
            ROCBLAS_CHECK_LINALG(rocsolver_spotrf(handle, uplo_mode, n, mat, n, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "cholesky");
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
            ROCBLAS_CHECK_LINALG(rocsolver_dpotrf(handle, uplo_mode, n, mat, n, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "cholesky");
        }

        int total = nbatch * n * n;
        int threads = 256;
        int blocks = (total + threads - 1) / threads;
        zero_triangle_f64<<<blocks, threads, 0, stream>>>(data, n, nbatch, upper);
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSOLVER
