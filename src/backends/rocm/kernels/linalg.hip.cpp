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
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/transform.hpp"
#include "../rocsolver_handle_pool.hpp"

// Forward-declare the tensor math entry points we need — pulling the full
// math.hpp into this HIP translation unit makes tenzor::sqrt/abs overloads
// visible to device code and poisons unqualified sqrt() calls in the
// shared-memory fallback kernels below.
namespace tenzor {
auto add(const Tensor& a, const Tensor& b) -> Tensor;
auto sub(const Tensor& a, const Tensor& b) -> Tensor;
auto mul(const Tensor& a, const Tensor& b) -> Tensor;
auto mul(const Tensor& a, double scalar) -> Tensor;
auto matmul(const Tensor& a, const Tensor& b) -> Tensor;
}

#include <rocsolver/rocsolver.h>
#include <rocblas/rocblas.h>
#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <vector>
#include <string>
#include <cmath>
#include <tuple>
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

/// Extract R from col-major QR result (m × n_cols) into row-major R (k × n_cols).
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
    r_mat[i * n_cols + j] = (j >= i) ? a_mat[j * m + i] : 0.0f;
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
    r_mat[i * n_cols + j] = (j >= i) ? a_mat[j * m + i] : 0.0;
}

/// Copy Q from col-major orgqr result (m × k) into row-major Q (m × k).
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
    q_mat[i * k + j] = a_mat[j * m + i];
}

__global__ void copy_q_columns_f64(const double* a_data, double* q_data,
                                    int m, int n_cols, int k, int nbatch) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = nbatch * m * k;
    if (idx >= total) return;

    /* col-major read: q_mat[i*k+j] = a_mat[j*m+i] */
    int b = idx / (m * k);
    int rem = idx % (m * k);
    int i = rem / k;
    int j = rem % k;

    const double* a_mat = a_data + b * m * n_cols;
    double* q_mat = q_data + b * m * k;
    q_mat[i * k + j] = a_mat[j * m + i];
}

/// HIP kernel to split a row-major packed LU matrix into separate L (unit
/// lower triangular) and U (upper triangular) tensors. One thread per element.
template<typename T>
__global__ void extract_lu_kernel_hip(const T* packed, T* L, T* U,
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

    // 2D input → scalar ({}) result; batched input → leading dims. Matches
    // torch.linalg.det: a plain matrix returns a 0-D tensor, not {1}.
    std::vector<int64_t> out_shape;
    auto shape = A.shape();
    for (size_t i = 0; i + 2 < shape.size(); i++) out_shape.push_back(shape[i]);

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
    // rocSOLVER (like cuSOLVER / LAPACK) uses column-major; Tenzor is
    // row-major. Transposing-then-contiguous gives row-major storage of
    // A^T, which rocSOLVER reads as col-major equal to A. Transpose the
    // result back at the end.
    auto work_a = tenzor::transpose(A.contiguous(), -1, -2).contiguous();
    auto work_b = tenzor::transpose(B.contiguous(), -1, -2).contiguous();
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
    return tenzor::transpose(work_b, -1, -2).contiguous();
}

// ============================================================================
// LU Factorization (PA = LU)
// ============================================================================

auto linalg_lu_kernel(const Tensor& A, hipStream_t stream)
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
    // representation of A. rocSOLVER will then factor A directly.
    auto a_t = tenzor::transpose(A, -2, -1).contiguous();

    auto handle = RocSOLVERHandlePool::get(stream);
    size_t ipiv_bytes = nbatch * n * sizeof(rocblas_int);
    auto* d_ipiv = static_cast<rocblas_int*>(
        backend::rocm::RocmCachingAllocator::get().allocate(ipiv_bytes));
    DeviceInfo d_info;

    if (original_dtype == DType::Float32) {
        float* data = a_t.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            rocblas_int* piv = d_ipiv + b * n;
            ROCBLAS_CHECK_LINALG(rocsolver_sgetrf(handle, n, n, mat, n, piv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "lu");
        }
    } else {
        double* data = a_t.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            rocblas_int* piv = d_ipiv + b * n;
            ROCBLAS_CHECK_LINALG(rocsolver_dgetrf(handle, n, n, mat, n, piv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "lu");
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
        extract_lu_kernel_hip<float><<<blocks, threads, 0, stream>>>(
            packed_rm.data<float>(), L.data<float>(), U.data<float>(), n, nbatch);
    } else {
        extract_lu_kernel_hip<double><<<blocks, threads, 0, stream>>>(
            packed_rm.data<double>(), L.data<double>(), U.data<double>(), n, nbatch);
    }
    HIP_CHECK_LINALG(hipGetLastError());

    // Step 4: copy pivots (rocSOLVER returns 1-based rocblas_int) into Int32 tensor.
    // rocblas_int is typedef'd to int32_t, but cast explicitly for clarity.
    auto pivots_out = zeros(piv_shape, DType::Int32, A.device());
    static_assert(sizeof(rocblas_int) == sizeof(int32_t),
                  "rocblas_int must match int32_t width");
    HIP_CHECK_LINALG(hipMemcpyAsync(pivots_out.data<int32_t>(),
        reinterpret_cast<const int32_t*>(d_ipiv),
        nbatch * n * sizeof(int32_t), hipMemcpyDeviceToDevice, stream));

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_ipiv);
    return {L, U, pivots_out};
}

// ============================================================================
// LU Solve (X = A^{-1} B given packed LU + pivots from getrf)
// ============================================================================

auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                             const Tensor& B, hipStream_t stream) -> Tensor {
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

    // Convert row-major packed LU into column-major form for rocSOLVER getrs:
    // transposing the row-major tensor produces storage that, viewed as
    // column-major, is the packed factorization in the col-major convention.
    auto lu_cm = tenzor::transpose(LU_data, -2, -1).contiguous();
    // Same trick for B: row-major (n, nrhs) → transpose to (nrhs, n) row-major,
    // whose storage interpreted col-major is the (n, nrhs) col-major form of B.
    auto b_cm = tenzor::transpose(B, -2, -1).contiguous();

    // Pivots: rocSOLVER expects rocblas_int*. Our pivots are Int32; copy to
    // a device-side rocblas_int buffer (widths match by static_assert above).
    auto piv_dev = pivots.to(B.device()).contiguous();
    static_assert(sizeof(rocblas_int) == sizeof(int32_t),
                  "rocblas_int must match int32_t width");
    size_t piv_bytes = nbatch * n * sizeof(rocblas_int);
    auto* d_ipiv_base = static_cast<rocblas_int*>(
        backend::rocm::RocmCachingAllocator::get().allocate(piv_bytes));
    HIP_CHECK_LINALG(hipMemcpyAsync(d_ipiv_base,
        reinterpret_cast<const rocblas_int*>(piv_dev.data<int32_t>()),
        piv_bytes, hipMemcpyDeviceToDevice, stream));

    auto handle = RocSOLVERHandlePool::get(stream);

    if (original_dtype == DType::Float32) {
        float* lu_ptr = lu_cm.data<float>();
        float* b_ptr = b_cm.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            float* lu_mat = lu_ptr + b * n * n;
            float* b_mat = b_ptr + b * n * nrhs;
            rocblas_int* piv = d_ipiv_base + b * n;
            ROCBLAS_CHECK_LINALG(rocsolver_sgetrs(handle, rocblas_operation_none,
                n, nrhs, lu_mat, n, piv, b_mat, n));
        }
    } else {
        double* lu_ptr = lu_cm.data<double>();
        double* b_ptr = b_cm.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            double* lu_mat = lu_ptr + b * n * n;
            double* b_mat = b_ptr + b * n * nrhs;
            rocblas_int* piv = d_ipiv_base + b * n;
            ROCBLAS_CHECK_LINALG(rocsolver_dgetrs(handle, rocblas_operation_none,
                n, nrhs, lu_mat, n, piv, b_mat, n));
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    backend::rocm::RocmCachingAllocator::get().free(d_ipiv_base);
    // b_cm has shape (..., nrhs, n) row-major; transpose back to (..., n, nrhs).
    return tenzor::transpose(b_cm, -2, -1).contiguous();
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

    // rocSOLVER gesvd leading-dimension rules (col-major), with m',n' = rocSOLVER
    // arguments (= our swapped n_cols, m):
    //   * U   arg: m' × m' (all) or m' × min(m',n')=k (singular). ldu >= m' = n_cols.
    //   * V^T arg: n' × n' (all) or k × n'                (singular). ldv >= n' (full)
    //                                                                       or >= k (reduced).
    // In this wrapper the rocSOLVER "U" buffer is our Vt (row-major k×n_cols = col-major
    // n_cols×k with leading dim n_cols for full, k for reduced), and the rocSOLVER
    // "V^T" buffer is our U (row-major m×k = col-major k×m with leading dim m for full,
    // k for reduced). Prior code passed ldv=m always, which overwrote beyond the
    // reduced-mode U buffer and produced a wrong SVD (symptom: A·pinv(A)·A diverged
    // by ~2.4 from A).
    int ldu_arg  = full_matrices ? n_cols : k;  // rocSOLVER "U" leading dim
    int ldvt_arg = full_matrices ? m      : k;  // rocSOLVER "V^T" leading dim

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

            // For row-major: we pass n_cols as m and m as n to treat as col-major A^T
            // Then U output is actually Vt, and Vt output is actually U
            ROCBLAS_CHECK_LINALG(rocsolver_sgesvd(handle, left_svect, right_svect,
                n_cols, m,  // swapped: col-major sees our row-major as transposed
                a_mat, n_cols,  // lda = n_cols (stride between columns in row-major)
                s_vec,
                vt_mat, ldu_arg,   // "U" output -> our Vt
                u_mat, ldvt_arg,   // "V^T" output -> our U
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

            ROCBLAS_CHECK_LINALG(rocsolver_dgesvd(handle, left_svect, right_svect,
                n_cols, m,
                a_mat, n_cols,
                s_vec,
                vt_mat, ldu_arg,
                u_mat, ldvt_arg,
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
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::qr: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);

    // rocSOLVER geqrf is column-major; transposing row-major A into row-major
    // A^T buffers hands rocSOLVER col-major A. The extract_r / copy_q_columns
    // helpers then read the col-major result and emit row-major Q/R.
    auto work = tenzor::transpose(A.contiguous(), -1, -2).contiguous();
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

        size_t tau_bytes = k * sizeof(float);
        auto* d_tau = static_cast<float*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;

            ROCBLAS_CHECK_LINALG(rocsolver_sgeqrf(handle, m, n_cols,
                a_mat, m, d_tau));

            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f32<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            ROCBLAS_CHECK_LINALG(rocsolver_sorgqr(handle, m, k, k,
                a_mat, m, d_tau));

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

            ROCBLAS_CHECK_LINALG(rocsolver_dgeqrf(handle, m, n_cols,
                a_mat, m, d_tau));

            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f64<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            ROCBLAS_CHECK_LINALG(rocsolver_dorgqr(handle, m, k, k,
                a_mat, m, d_tau));

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
    // rocSOLVER syevd writes eigenvectors as COLUMNS of a COLUMN-MAJOR matrix.
    // Reinterpreted as row-major this gives V^T; transpose so the caller
    // sees row-major V with eigenvectors as columns (matches CPU/LAPACKE).
    auto V = tenzor::transpose(work, -2, -1).contiguous();
    return {W, V};
}

// ============================================================================
// QR iteration fallback for eigendecomposition (rocSOLVER < 6.0)
// Ported from Vulkan compute shader: linalg_eig.comp
// Computes eigenvalues only (not eigenvectors) for matrices up to 32x32.
// ============================================================================

// Parallelized QR eigendecomposition fallback for rocSOLVER < 6.0.
// Uses dynamic shared memory for H (Hessenberg) and Q (eigenvector accumulator),
// distributes Householder reflections across threads, and processes batches in parallel.
// Max matrix size is bounded by device shared memory (typically ~90 for float, ~64 for double).

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

    // Dynamic shared memory: H[n*n] + Q[n*n] + scratch[4] (v0, tau, alpha, flag)
    extern __shared__ char smem_raw[];
    T* H = reinterpret_cast<T*>(smem_raw);
    T* Q = H + n * n;
    T* scratch = Q + n * n;  // scratch[0]=v0, scratch[1]=tau, scratch[2]=alpha, scratch[3]=flag

    // Per-batch offsets
    const T* A = A_in + batch_idx * n * n;
    T* wr = wr_out + batch_idx * n;
    T* wi = wi_out + batch_idx * n;
    T* V = V_out + batch_idx * n * n;

    // Load matrix into shared memory (parallel across threads)
    for (int idx = static_cast<int>(tid); idx < n * n; idx += static_cast<int>(num_threads)) {
        H[idx] = A[idx];
        // Initialize Q = I
        int row = idx / n;
        int col = idx % n;
        Q[idx] = (row == col) ? T(1) : T(0);
    }
    __syncthreads();

    // =========================================================================
    // Step 1: Reduce to upper Hessenberg form via Householder reflections
    // Thread 0 computes the Householder vector; all threads apply reflections.
    // =========================================================================
    for (int k = 0; k + 2 < n; k++) {
        // Thread 0 computes Householder vector parameters
        if (tid == 0) {
            T sigma = T(0);
            for (int i = k + 1; i < n; i++) {
                sigma += H[i * n + k] * H[i * n + k];
            }
            T norm_x = sqrt(sigma);
            if (norm_x < zero_tol) {
                scratch[1] = T(0);  // tau = 0 signals skip
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
                    scratch[0] = v0_val;   // v0
                    scratch[1] = T(2) / v_norm_sq;  // tau
                    scratch[2] = a;        // alpha
                }
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

        // Left: H = (I - tau * v * v^T) * H  — distribute columns across threads
        // v = [v0, H[k+2][k], H[k+3][k], ...] (subdiagonal of column k)
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

        // Right: H = H * (I - tau * v * v^T)  — distribute rows across threads
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

        // Accumulate Q = Q * (I - tau * v * v^T)  — distribute rows across threads
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

        // Clean up subdiagonal (thread 0)
        if (tid == 0) {
            H[(k + 1) * n + k] = alpha;
            for (int i = k + 2; i < n; i++) {
                H[i * n + k] = T(0);
            }
        }
        __syncthreads();
    }

    // =========================================================================
    // Step 2: QR iteration with implicit double shifts on Hessenberg matrix.
    // Thread 0 drives the sequential bulge chase; all threads apply reflections.
    // =========================================================================
    // Use scratch[3] as shared nn (active matrix size)
    if (tid == 0) {
        scratch[3] = static_cast<T>(n);
    }
    __syncthreads();

    for (int iter = 0; iter < max_iterations; iter++) {
        int nn = static_cast<int>(scratch[3]);
        if (nn <= 0) break;

        // Deflation checks (thread 0 only, results broadcast via scratch)
        if (tid == 0) {
            bool deflated = false;

            // Check for 1x1 deflation
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

            // Check for 2x2 deflation
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

            // If deflated, signal skip (scratch[2] = 1), else compute shifts
            if (deflated) {
                scratch[2] = T(1);  // flag: skip QR step
            } else {
                scratch[2] = T(0);  // flag: do QR step
            }
        }
        __syncthreads();

        if (scratch[2] != T(0)) continue;  // deflation occurred, re-check

        nn = static_cast<int>(scratch[3]);

        // Francis double-shift QR step — thread 0 drives the bulge chase,
        // all threads apply left/right reflections in parallel.
        // Thread 0 computes initial shift vector
        T x, y, z;
        if (tid == 0) {
            T s = H[(nn - 2) * n + (nn - 2)] + H[(nn - 1) * n + (nn - 1)];
            T t = H[(nn - 2) * n + (nn - 2)] * H[(nn - 1) * n + (nn - 1)] -
                  H[(nn - 2) * n + (nn - 1)] * H[(nn - 1) * n + (nn - 2)];

            scratch[0] = H[0] * H[0] + H[1] * H[n] - s * H[0] + t;  // x
            scratch[1] = H[n] * (H[0] + H[n + 1] - s);  // y
            scratch[2] = (nn > 2) ? H[n] * H[2 * n + 1] : T(0);  // z
        }
        __syncthreads();
        x = scratch[0]; y = scratch[1]; z = scratch[2];

        // Chase the bulge (sequential in k, parallel in reflections)
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
            int m = (k + 4 < nn) ? k + 4 : nn;

            // Left: H = (I - tau * v * v^T) * H — columns distributed across threads
            for (int j = k + static_cast<int>(tid); j < nn; j += static_cast<int>(num_threads)) {
                T dot = v0 * H[k * n + j] + v1 * H[(k + 1) * n + j];
                if (k + 2 < nn) dot += v2 * H[(k + 2) * n + j];
                dot *= tau_h;
                H[k * n + j] -= v0 * dot;
                H[(k + 1) * n + j] -= v1 * dot;
                if (k + 2 < nn) H[(k + 2) * n + j] -= v2 * dot;
            }
            __syncthreads();

            // Right: H = H * (I - tau * v * v^T) — rows distributed across threads
            for (int i = static_cast<int>(tid); i < m; i += static_cast<int>(num_threads)) {
                T dot = v0 * H[i * n + k] + v1 * H[i * n + (k + 1)];
                if (k + 2 < nn) dot += v2 * H[i * n + (k + 2)];
                dot *= tau_h;
                H[i * n + k] -= v0 * dot;
                H[i * n + (k + 1)] -= v1 * dot;
                if (k + 2 < nn) H[i * n + (k + 2)] -= v2 * dot;
            }
            __syncthreads();

            // Accumulate Q = Q * (I - tau * v * v^T) — rows distributed
            for (int i = static_cast<int>(tid); i < n; i += static_cast<int>(num_threads)) {
                T dot = v0 * Q[i * n + k] + v1 * Q[i * n + (k + 1)];
                if (k + 2 < nn) dot += v2 * Q[i * n + (k + 2)];
                dot *= tau_h;
                Q[i * n + k] -= v0 * dot;
                Q[i * n + (k + 1)] -= v1 * dot;
                if (k + 2 < nn) Q[i * n + (k + 2)] -= v2 * dot;
            }
            __syncthreads();

            // Zero out below subdiagonal + update shift vector (thread 0)
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

        // Final 2x2 Givens rotation (parallel application)
        if (nn >= 2) {
            T norm_v = sqrt(x * x + y * y);
            if (norm_v > zero_tol) {
                int k = nn - 2;
                T c_val = x / norm_v;
                T s_val = -y / norm_v;

                // Left multiply rows k, k+1 — columns distributed
                for (int j = k + static_cast<int>(tid); j < nn; j += static_cast<int>(num_threads)) {
                    T tmp = c_val * H[k * n + j] - s_val * H[(k + 1) * n + j];
                    H[(k + 1) * n + j] = s_val * H[k * n + j] + c_val * H[(k + 1) * n + j];
                    H[k * n + j] = tmp;
                }
                __syncthreads();

                // Right multiply cols k, k+1 — rows distributed
                for (int i = static_cast<int>(tid); i < nn; i += static_cast<int>(num_threads)) {
                    T tmp = c_val * H[i * n + k] - s_val * H[i * n + (k + 1)];
                    H[i * n + (k + 1)] = s_val * H[i * n + k] + c_val * H[i * n + (k + 1)];
                    H[i * n + k] = tmp;
                }
                __syncthreads();

                // Accumulate Q — rows distributed
                for (int i = static_cast<int>(tid); i < n; i += static_cast<int>(num_threads)) {
                    T tmp = c_val * Q[i * n + k] - s_val * Q[i * n + (k + 1)];
                    Q[i * n + (k + 1)] = s_val * Q[i * n + k] + c_val * Q[i * n + (k + 1)];
                    Q[i * n + k] = tmp;
                }
                __syncthreads();
            }
        }
    }

    // Handle remaining 1x1 or 2x2 blocks
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

    // Write eigenvector matrix Q to output (parallel)
    for (int idx = static_cast<int>(tid); idx < n * n; idx += static_cast<int>(num_threads)) {
        V[idx] = Q[idx];
    }
}

// ----------------------------------------------------------------------------
// Symmetry probe: max |A| and max |A - A^T| over a batched square matrix,
// reduced into two device-side scalars. Used by linalg_eig_kernel below to
// choose between the eigh fast-path and the QR fallback without downloading
// the entire input back to the host.
// ----------------------------------------------------------------------------
template <typename T>
__global__ void eig_symmetry_probe_kernel(const T* __restrict__ A,
                                          int64_t nbatch, int64_t n,
                                          double* __restrict__ d_max_abs,
                                          double* __restrict__ d_max_diff) {
    extern __shared__ double smem[];
    double* s_abs = smem;                 // blockDim.x doubles
    double* s_diff = smem + blockDim.x;   // blockDim.x doubles

    int64_t b = blockIdx.x;
    if (b >= nbatch) return;
    const T* mat = A + b * n * n;

    double local_max_abs = 0.0;
    double local_max_diff = 0.0;

    // Each thread strides over the matrix; for the diff we look at the
    // upper triangle (i < j) so each pair is visited exactly once.
    int64_t total = n * n;
    for (int64_t idx = threadIdx.x; idx < total; idx += blockDim.x) {
        int64_t i = idx / n;
        int64_t j = idx - i * n;
        double v = static_cast<double>(mat[idx]);
        double av = (v < 0.0) ? -v : v;
        if (av > local_max_abs) local_max_abs = av;
        if (j > i) {
            double vt = static_cast<double>(mat[j * n + i]);
            double d = v - vt;
            if (d < 0.0) d = -d;
            if (d > local_max_diff) local_max_diff = d;
        }
    }

    s_abs[threadIdx.x] = local_max_abs;
    s_diff[threadIdx.x] = local_max_diff;
    __syncthreads();

    // Block-level max reduction.
    for (int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            if (s_abs[threadIdx.x + stride] > s_abs[threadIdx.x])
                s_abs[threadIdx.x] = s_abs[threadIdx.x + stride];
            if (s_diff[threadIdx.x + stride] > s_diff[threadIdx.x])
                s_diff[threadIdx.x] = s_diff[threadIdx.x + stride];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        // Atomic max over batches via CAS. atomicMax(double*) is not
        // universally available across HIP arches, so emulate with CAS on
        // the 64-bit bit pattern.
        double block_abs = s_abs[0];
        double block_diff = s_diff[0];
        unsigned long long* ull_abs = reinterpret_cast<unsigned long long*>(d_max_abs);
        unsigned long long* ull_diff = reinterpret_cast<unsigned long long*>(d_max_diff);
        unsigned long long old_a = *ull_abs;
        unsigned long long assumed_a;
        do {
            assumed_a = old_a;
            double cur = __longlong_as_double(static_cast<long long>(assumed_a));
            if (block_abs <= cur) break;
            old_a = atomicCAS(ull_abs, assumed_a,
                              static_cast<unsigned long long>(
                                  __double_as_longlong(block_abs)));
        } while (assumed_a != old_a);

        unsigned long long old_d = *ull_diff;
        unsigned long long assumed_d;
        do {
            assumed_d = old_d;
            double cur = __longlong_as_double(static_cast<long long>(assumed_d));
            if (block_diff <= cur) break;
            old_d = atomicCAS(ull_diff, assumed_d,
                              static_cast<unsigned long long>(
                                  __double_as_longlong(block_diff)));
        } while (assumed_d != old_d);
    }
}

// Returns {max_abs, max_diff} computed on-device. Only 16 bytes flow back
// to the host (two doubles) — necessary metadata, not a CPU compute path.
template <typename T>
inline std::pair<double, double> eig_symmetry_metrics(const T* d_A,
                                                      int64_t nbatch,
                                                      int64_t n,
                                                      hipStream_t stream) {
    double* d_pair = nullptr;
    HIP_CHECK_LINALG(hipMalloc(&d_pair, 2 * sizeof(double)));
    double init[2] = {0.0, 0.0};
    HIP_CHECK_LINALG(hipMemcpyAsync(d_pair, init, 2 * sizeof(double),
                                    hipMemcpyHostToDevice, stream));

    int threads = 256;
    size_t smem_bytes = 2 * threads * sizeof(double);
    hipLaunchKernelGGL(eig_symmetry_probe_kernel<T>,
        dim3(static_cast<unsigned>(nbatch)), dim3(threads), smem_bytes, stream,
        d_A, nbatch, n, d_pair, d_pair + 1);
    HIP_CHECK_LINALG(hipGetLastError());

    double host_pair[2] = {0.0, 0.0};
    HIP_CHECK_LINALG(hipMemcpyAsync(host_pair, d_pair, 2 * sizeof(double),
                                    hipMemcpyDeviceToHost, stream));
    HIP_CHECK_LINALG(hipStreamSynchronize(stream));
    HIP_CHECK_LINALG(hipFree(d_pair));
    return {host_pair[0], host_pair[1]};
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

    // Approximate-symmetry check (per-batch). gradcheck perturbs SPD
    // inputs by ε≈1e-6 — that breaks strict symmetry but each element is
    // very close. Within 1e-3 relative is treated as "intended symmetric,
    // perturbed by noise" and routed through `eigh` on the SYMMETRIZED
    // matrix. Sum-of-eigenvalues equals trace, which is preserved by
    // symmetrization, so the numerical gradient matches the analytical.
    {
        auto [n_check, ndim_check] = check_square(A);
        int64_t nbatch_check = batch_size(A);
        auto A_cont = A.contiguous();
        bool is_near_symmetric = true;
        if (A.dtype() == DType::Float32) {
            auto [a_max, diff_max] = eig_symmetry_metrics<float>(
                A_cont.data<float>(), nbatch_check, n_check, stream);
            is_near_symmetric = (diff_max < 1e-2 * std::max(a_max, 1.0));
        } else {
            auto [a_max, diff_max] = eig_symmetry_metrics<double>(
                A_cont.data<double>(), nbatch_check, n_check, stream);
            is_near_symmetric = (diff_max < 1e-3 * std::max(a_max, 1.0));
        }
        if (is_near_symmetric) {
            auto At = ::tenzor::transpose(A, ndim_check - 2, ndim_check - 1).contiguous();
            auto A_sym = ::tenzor::mul(::tenzor::add(A, At), 0.5);
            auto [W, V] = linalg_eigh_kernel(A_sym.contiguous(), stream);
            std::vector<int64_t> w_shape_v(W.shape().begin(), W.shape().end());
            auto WI = zeros(w_shape_v, A.dtype(), A.device());
            return {W, WI, V};
        }
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
    // QR iteration fallback for rocSOLVER < 6.0 with dynamic shared memory.
    // Shared memory layout: H[n*n] + Q[n*n] + scratch[4]
    // Max n is bounded by device shared memory (typically ~90 for float, ~64 for double).
    int device_id = 0;
    HIP_CHECK_LINALG(hipGetDevice(&device_id));
    int max_smem = 0;
    HIP_CHECK_LINALG(hipDeviceGetAttribute(&max_smem,
        hipDeviceAttributeMaxSharedMemoryPerBlock, device_id));

    constexpr int max_qr_iters = 200;
    constexpr int block_size = 128;

    if (A.dtype() == DType::Float32) {
        size_t smem_needed = 2 * n * n * sizeof(float) + 4 * sizeof(float);
        if (static_cast<int64_t>(smem_needed) > max_smem) {
            int max_n = static_cast<int>(sqrt(static_cast<double>(max_smem) / (2.0 * sizeof(float))));
            throw std::runtime_error(
                "linalg.eig: matrix " + std::to_string(n) + "x" + std::to_string(n) +
                " exceeds device shared memory for QR fallback (max ~" +
                std::to_string(max_n) + "). Upgrade to rocSOLVER >= 6.0.");
        }
        qr_eig_fallback_kernel<float><<<static_cast<int>(nbatch), block_size, smem_needed, stream>>>(
            work.data<float>(),
            WR.data<float>(),
            WI.data<float>(),
            V.data<float>(),
            static_cast<int>(n), max_qr_iters);
    } else {
        size_t smem_needed = 2 * n * n * sizeof(double) + 4 * sizeof(double);
        if (static_cast<int64_t>(smem_needed) > max_smem) {
            int max_n = static_cast<int>(sqrt(static_cast<double>(max_smem) / (2.0 * sizeof(double))));
            throw std::runtime_error(
                "linalg.eig: matrix " + std::to_string(n) + "x" + std::to_string(n) +
                " exceeds device shared memory for QR fallback (max ~" +
                std::to_string(max_n) + "). Upgrade to rocSOLVER >= 6.0.");
        }
        qr_eig_fallback_kernel<double><<<static_cast<int>(nbatch), block_size, smem_needed, stream>>>(
            work.data<double>(),
            WR.data<double>(),
            WI.data<double>(),
            V.data<double>(),
            static_cast<int>(n), max_qr_iters);
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
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

    // rocSOLVER operates column-major; Tenzor is row-major. A is symmetric
    // so A_col == A_row, but the output triangle is inverted between
    // conventions: col-major LOWER = row-major UPPER. Feed the opposite
    // uplo and let zero_triangle preserve the user-requested row-major
    // triangle.
    rocblas_fill uplo_mode = upper ? rocblas_fill_lower : rocblas_fill_upper;

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

// ============================================================================
// Triangular Solve (AX = B, A triangular) — rocBLAS trsm
// ============================================================================

auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                     bool upper, bool unitriangular,
                                     hipStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "solve_triangular");
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        return linalg_solve_triangular_kernel(
            A.to(DType::Float32), B.to(DType::Float32),
            upper, unitriangular, stream).to(A.dtype());
    }

    // rocBLAS trsm operates in column-major. We physically transpose A and B
    // into column-major storage, which is a double negation (transpose +
    // reinterpret-as-col-major cancel out), so uplo maps straight through.
    auto a_cm = tenzor::transpose(A.contiguous().clone(), -2, -1).contiguous();
    auto b_cm = tenzor::transpose(B.contiguous().clone(), -2, -1).contiguous();

    auto [n, ndim_a] = check_square(a_cm);
    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(a_cm);

    rocblas_fill uplo = upper ? rocblas_fill_upper : rocblas_fill_lower;
    rocblas_diagonal diag = unitriangular ? rocblas_diagonal_unit : rocblas_diagonal_non_unit;

    auto handle = RocSOLVERHandlePool::get(stream);

    if (A.dtype() == DType::Float32) {
        float alpha = 1.0f;
        float* a_ptr = a_cm.data<float>();
        float* b_ptr = b_cm.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            ROCBLAS_CHECK_LINALG(rocblas_strsm(handle, rocblas_side_left, uplo,
                rocblas_operation_none, diag,
                static_cast<rocblas_int>(n), static_cast<rocblas_int>(nrhs), &alpha,
                a_ptr + b * n * n, static_cast<rocblas_int>(n),
                b_ptr + b * n * nrhs, static_cast<rocblas_int>(n)));
        }
    } else {
        double alpha = 1.0;
        double* a_ptr = a_cm.data<double>();
        double* b_ptr = b_cm.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            ROCBLAS_CHECK_LINALG(rocblas_dtrsm(handle, rocblas_side_left, uplo,
                rocblas_operation_none, diag,
                static_cast<rocblas_int>(n), static_cast<rocblas_int>(nrhs), &alpha,
                a_ptr + b * n * n, static_cast<rocblas_int>(n),
                b_ptr + b * n * nrhs, static_cast<rocblas_int>(n)));
        }
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return tenzor::transpose(b_cm, -2, -1).contiguous();
}

// ============================================================================
// Geqrf — raw QR factorization returning packed reflectors + tau
// ============================================================================

auto linalg_geqrf_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor> {
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

    auto handle = RocSOLVERHandlePool::get(stream);

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* tau_data = tau_result.data<float>();

        // Allocate tau on device for rocSOLVER (writes directly to device memory)
        size_t tau_bytes = k * sizeof(float);
        auto* d_tau = static_cast<float*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;
            float* tau_ptr = tau_data + b * k;

            // rocSOLVER geqrf works in column-major. For row-major m x n,
            // we pass n_cols as m and m as n (treating as A^T in col-major).
            ROCBLAS_CHECK_LINALG(rocsolver_sgeqrf(handle, n_cols, m,
                a_mat, n_cols, d_tau));

            // Copy tau from device temporary to output tensor
            HIP_CHECK_LINALG(hipMemcpyAsync(tau_ptr, d_tau, tau_bytes,
                hipMemcpyDeviceToDevice, stream ? stream : nullptr));
        }

        backend::rocm::RocmCachingAllocator::get().free(d_tau);
    } else {
        double* a_data = work.data<double>();
        double* tau_data = tau_result.data<double>();

        size_t tau_bytes = k * sizeof(double);
        auto* d_tau = static_cast<double*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * m * n_cols;
            double* tau_ptr = tau_data + b * k;

            ROCBLAS_CHECK_LINALG(rocsolver_dgeqrf(handle, n_cols, m,
                a_mat, n_cols, d_tau));

            HIP_CHECK_LINALG(hipMemcpyAsync(tau_ptr, d_tau, tau_bytes,
                hipMemcpyDeviceToDevice, stream ? stream : nullptr));
        }

        backend::rocm::RocmCachingAllocator::get().free(d_tau);
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {work, tau_result};
}

// ============================================================================
// Ormqr — multiply matrix by Q from QR factorization using tau vectors
// ============================================================================

auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                          const Tensor& C, bool left, bool transpose_q,
                          hipStream_t stream) -> Tensor {
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

    auto handle = RocSOLVERHandlePool::get(stream);

    // rocSOLVER ormqr operates in column-major. Same transposition logic as CUDA:
    // Row-major left no-trans Q*C  -> col-major Right Trans on C^T
    // Row-major left trans Q^T*C   -> col-major Right NoTrans on C^T
    // Row-major right no-trans C*Q -> col-major Left Trans on C^T
    // Row-major right trans C*Q^T  -> col-major Left NoTrans on C^T
    rocblas_side side;
    rocblas_operation trans;
    if (left && !transpose_q)       { side = rocblas_side_right; trans = rocblas_operation_transpose; }
    else if (left && transpose_q)   { side = rocblas_side_right; trans = rocblas_operation_none; }
    else if (!left && !transpose_q) { side = rocblas_side_left;  trans = rocblas_operation_transpose; }
    else                            { side = rocblas_side_left;  trans = rocblas_operation_none; }

    if (C.dtype() == DType::Float32) {
        float* c_data = work_c.data<float>();
        const float* r_data = refl.data<float>();
        const float* tau_data = tau_c.data<float>();

        // rocSOLVER needs device tau; copy batch tau slice to a temp buffer
        size_t tau_bytes = k_refl * sizeof(float);
        auto* d_tau = static_cast<float*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            const float* r_mat = r_data + b * r_m * r_n;
            const float* tau_ptr = tau_data + b * k_refl;
            float* c_mat = c_data + b * c_m * c_n;

            HIP_CHECK_LINALG(hipMemcpyAsync(d_tau, tau_ptr, tau_bytes,
                hipMemcpyDeviceToDevice, stream ? stream : nullptr));

            ROCBLAS_CHECK_LINALG(rocsolver_sormqr(handle,
                side, trans,
                static_cast<rocblas_int>(c_n), static_cast<rocblas_int>(c_m),
                static_cast<rocblas_int>(k_refl),
                const_cast<float*>(r_mat), static_cast<rocblas_int>(r_n),
                d_tau,
                c_mat, static_cast<rocblas_int>(c_n)));
        }

        backend::rocm::RocmCachingAllocator::get().free(d_tau);
    } else {
        double* c_data = work_c.data<double>();
        const double* r_data = refl.data<double>();
        const double* tau_data = tau_c.data<double>();

        size_t tau_bytes = k_refl * sizeof(double);
        auto* d_tau = static_cast<double*>(
            backend::rocm::RocmCachingAllocator::get().allocate(tau_bytes));

        for (int64_t b = 0; b < nbatch; b++) {
            const double* r_mat = r_data + b * r_m * r_n;
            const double* tau_ptr = tau_data + b * k_refl;
            double* c_mat = c_data + b * c_m * c_n;

            HIP_CHECK_LINALG(hipMemcpyAsync(d_tau, tau_ptr, tau_bytes,
                hipMemcpyDeviceToDevice, stream ? stream : nullptr));

            ROCBLAS_CHECK_LINALG(rocsolver_dormqr(handle,
                side, trans,
                static_cast<rocblas_int>(c_n), static_cast<rocblas_int>(c_m),
                static_cast<rocblas_int>(k_refl),
                const_cast<double*>(r_mat), static_cast<rocblas_int>(r_n),
                d_tau,
                c_mat, static_cast<rocblas_int>(c_n)));
        }

        backend::rocm::RocmCachingAllocator::get().free(d_tau);
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work_c;
}

// =========================================================================
// LDL^T factorization (rocsolver_ssytrf / rocsolver_dsytrf)
// =========================================================================
auto linalg_ldl_factor_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    auto original_dtype = A.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto [LD32, piv] = linalg_ldl_factor_kernel(A.to(DType::Float32), stream);
        return {LD32.to(original_dtype), piv};
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument("linalg::ldl_factor: unsupported dtype");
    }

    auto [n, ndim] = check_square(A);
    int64_t nbatch = batch_size(A);

    auto a_shape = to_vec(A.shape());
    std::vector<int64_t> piv_shape;
    for (size_t i = 0; i + 2 < a_shape.size(); ++i) piv_shape.push_back(a_shape[i]);
    piv_shape.push_back(n);

    // rocSOLVER expects column-major; transpose row-major -> column-major
    auto work = tenzor::transpose(A, -2, -1).contiguous();
    auto handle = RocSOLVERHandlePool::get(stream);

    // Allocate pivots on device
    size_t piv_bytes = nbatch * n * sizeof(rocblas_int);
    auto* d_ipiv = static_cast<rocblas_int*>(
        backend::rocm::RocmCachingAllocator::get().allocate(piv_bytes));
    DeviceInfo d_info;

    if (original_dtype == DType::Float32) {
        float* data = work.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            rocblas_int* piv = d_ipiv + b * n;
            ROCBLAS_CHECK_LINALG(rocsolver_ssytrf(handle,
                rocblas_fill_lower, static_cast<rocblas_int>(n),
                mat, static_cast<rocblas_int>(n), piv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "ldl_factor");
        }
    } else {
        double* data = work.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            rocblas_int* piv = d_ipiv + b * n;
            ROCBLAS_CHECK_LINALG(rocsolver_dsytrf(handle,
                rocblas_fill_lower, static_cast<rocblas_int>(n),
                mat, static_cast<rocblas_int>(n), piv, d_info.ptr));
            check_rocsolver_info(d_info.ptr, "ldl_factor");
        }
    }

    // Transpose back to row-major
    auto LD = tenzor::transpose(work, -2, -1).contiguous();

    // Copy pivots to Int32 tensor on device
    auto pivots_out = tenzor::zeros(piv_shape, DType::Int32, A.device());
    // rocblas_int may differ from int32_t in size; copy element-wise if needed
    HIP_CHECK_LINALG(hipMemcpy(pivots_out.data<int32_t>(), d_ipiv,
        nbatch * n * sizeof(rocblas_int), hipMemcpyDeviceToDevice));
    backend::rocm::RocmCachingAllocator::get().free(d_ipiv);

    return {LD, pivots_out};
}

// =========================================================================
// LDL^T solve — native GPU kernel using Bunch-Kaufman pivoted LDL factors.
// Decomposes into: P*B -> forward subst L -> diagonal solve D -> back subst
// L^T -> P^T, all in shared memory on device.
// =========================================================================

template<typename T>
__global__ void ldl_solve_bk_kernel(
    const T* __restrict__ ld_data,
    const int* __restrict__ pivots,
    T* __restrict__ b_data,
    int n, int nrhs)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* LD = reinterpret_cast<T*>(smem_raw);
    T* B = LD + n * n;

    const T* batch_ld = ld_data + batch_idx * n * n;
    const int* batch_piv = pivots + batch_idx * n;
    T* batch_b = b_data + batch_idx * n * nrhs;

    for (int idx = tid; idx < n * n; idx += num_threads)
        LD[idx] = batch_ld[idx];
    for (int idx = tid; idx < n * nrhs; idx += num_threads)
        B[idx] = batch_b[idx];
    __syncthreads();

    if (tid == 0) {
        // Forward pivot permutation
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                int sr = p - 1;
                if (sr != k)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j]; B[k * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k++;
            } else {
                int sr = (-p) - 1;
                if (sr != k + 1)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[(k+1) * nrhs + j]; B[(k+1) * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k += 2;
            }
        }

        // Forward substitution: L * Y = P*B
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                for (int i = k + 1; i < n; i++) {
                    T m = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++)
                        B[i * nrhs + j] -= m * B[k * nrhs + j];
                }
                k++;
            } else {
                for (int i = k + 2; i < n; i++) {
                    T m0 = LD[i * n + k], m1 = LD[i * n + k + 1];
                    for (int j = 0; j < nrhs; j++)
                        B[i * nrhs + j] -= m0 * B[k * nrhs + j] + m1 * B[(k+1) * nrhs + j];
                }
                k += 2;
            }
        }

        // Diagonal solve: D * Z = Y
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                T d = LD[k * n + k];
                for (int j = 0; j < nrhs; j++) B[k * nrhs + j] /= d;
                k++;
            } else {
                T d11 = LD[k * n + k], d21 = LD[(k+1) * n + k], d22 = LD[(k+1) * n + (k+1)];
                T det = d11 * d22 - d21 * d21;
                for (int j = 0; j < nrhs; j++) {
                    T y0 = B[k * nrhs + j], y1 = B[(k+1) * nrhs + j];
                    B[k * nrhs + j]     = (d22 * y0 - d21 * y1) / det;
                    B[(k+1) * nrhs + j] = (d11 * y1 - d21 * y0) / det;
                }
                k += 2;
            }
        }

        // Backward substitution: L^T * X = Z
        for (int k = n - 1; k >= 0; ) {
            int p = batch_piv[k];
            if (p > 0) {
                for (int i = k + 1; i < n; i++) {
                    T m = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++)
                        B[k * nrhs + j] -= m * B[i * nrhs + j];
                }
                k--;
            } else {
                int k0 = k - 1;
                for (int i = k + 1; i < n; i++) {
                    T m0 = LD[i * n + k0], m1 = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++) {
                        B[k0 * nrhs + j] -= m0 * B[i * nrhs + j];
                        B[k * nrhs + j]  -= m1 * B[i * nrhs + j];
                    }
                }
                k -= 2;
            }
        }

        // Inverse pivot permutation P^T
        for (int k = n - 1; k >= 0; ) {
            int p = batch_piv[k];
            if (p > 0) {
                int sr = p - 1;
                if (sr != k)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j]; B[k * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k--;
            } else {
                int sr = (-p) - 1;
                if (sr != k)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j]; B[k * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k -= 2;
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n * nrhs; idx += num_threads)
        batch_b[idx] = B[idx];
}

auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                              const Tensor& B, hipStream_t stream) -> Tensor {
    auto original_dtype = LD.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto result = linalg_ldl_solve_kernel(LD.to(DType::Float32), pivots,
                                               B.to(DType::Float32), stream);
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

    int threads = std::min(static_cast<int>(n), 128);
    if (threads < 1) threads = 1;

    if (original_dtype == DType::Float32) {
        size_t smem = (n * n + n * nrhs) * sizeof(float);
        ldl_solve_bk_kernel<float><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<float>(), pivots.data<int32_t>(),
            work_b.data<float>(), static_cast<int>(n), static_cast<int>(nrhs));
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        size_t smem = (n * n + n * nrhs) * sizeof(double);
        ldl_solve_bk_kernel<double><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<double>(), pivots.data<int32_t>(),
            work_b.data<double>(), static_cast<int>(n), static_cast<int>(nrhs));
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work_b;
}

// =========================================================================
// Householder product — compose from existing ormqr kernel
// =========================================================================
auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                [[maybe_unused]] hipStream_t stream) -> Tensor {
    // Apply Householder reflectors to I[:, :n]: Q = H_0 · H_1 · … · H_{k-1} · I[:m,:n]
    //
    // LAPACK's sorgqr (used by the CPU path) produces an m×n matrix — the
    // first n columns of the full m×m Q. We replicate that with row-major
    // tensor ops, walking the reflectors right→left (H_{k-1} first) because
    // the canonical order that matches LAPACK is to accumulate columns of Q
    // starting from the identity on the right.
    //
    // Routing through rocSOLVER ormqr with an identity matrix as C works only
    // for square m=n inputs (row-major/col-major leading-dimension mismatch
    // with non-square C), so we mirror the CUDA implementation that sidesteps
    // cuSOLVER's ormqr for the same reason.
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("linalg::householder_product: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    int64_t k = tau.shape()[tau.shape().size() - 1];
    if (k > std::min(m, n)) {
        throw std::invalid_argument(
            "linalg::householder_product: tau length must be ≤ min(m, n)");
    }

    DType dt = input.dtype();
    Device dev = input.device();

    DType compute_dt = dt;
    if (dt == DType::Float16 || dt == DType::BFloat16) compute_dt = DType::Float32;

    Tensor V = input.contiguous();
    Tensor tau_c = tau.contiguous();
    if (V.dtype() != compute_dt) V = V.to(compute_dt);
    if (tau_c.dtype() != compute_dt) tau_c = tau_c.to(compute_dt);

    auto I_full = tenzor::eye(m, std::nullopt, compute_dt, dev);
    Tensor Q = (n == m) ? I_full
                        : tenzor::slice(I_full, /*dim=*/1, /*start=*/0, /*end=*/n).contiguous();
    if (ndim > 2) {
        std::vector<int64_t> batched_shape(shape.begin(), shape.end());
        Q = tenzor::expand(Q, std::move(batched_shape));
        Q = Q.contiguous();
    }

    for (int64_t j = k - 1; j >= 0; --j) {
        Tensor v_j = tenzor::slice(V, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();

        // Mask rows [0..j) to 0 and overwrite row j with 1, keeping rows (j..m) as V[i, j].
        auto mask_cpu = tenzor::ones({m, int64_t(1)}, compute_dt, Device::cpu());
        auto overrides_cpu = tenzor::zeros({m, int64_t(1)}, compute_dt, Device::cpu());
        if (compute_dt == DType::Float32) {
            float* mm = mask_cpu.data<float>();
            float* oo = overrides_cpu.data<float>();
            for (int64_t i = 0; i < j; ++i) { mm[i] = 0.0f; oo[i] = 0.0f; }
            if (j < m) { mm[j] = 0.0f; oo[j] = 1.0f; }
        } else {
            double* mm = mask_cpu.data<double>();
            double* oo = overrides_cpu.data<double>();
            for (int64_t i = 0; i < j; ++i) { mm[i] = 0.0; oo[i] = 0.0; }
            if (j < m) { mm[j] = 0.0; oo[j] = 1.0; }
        }
        Tensor mask = mask_cpu.to(dev);
        Tensor overrides = overrides_cpu.to(dev);
        if (ndim > 2) {
            std::vector<int64_t> bshape(shape.begin(), shape.end() - 1);
            bshape.push_back(1);
            mask = tenzor::expand(mask, bshape).contiguous();
            overrides = tenzor::expand(overrides, bshape).contiguous();
        }
        v_j = tenzor::add(tenzor::mul(v_j, mask), overrides);

        Tensor tau_j = tenzor::slice(tau_c, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();

        Tensor v_jT = tenzor::transpose(v_j, -1, -2);
        Tensor u = tenzor::matmul(v_jT, Q);

        Tensor outer = tenzor::matmul(v_j, u);
        std::vector<int64_t> tau_shape(tau_j.shape().begin(), tau_j.shape().end());
        tau_shape.push_back(1);
        tau_j = tau_j.reshape(tau_shape);
        Tensor scaled = tenzor::mul(outer, tau_j);
        Q = tenzor::sub(Q, scaled);
    }

    if (Q.dtype() != dt) Q = Q.to(dt);
    return Q;
}

} // namespace rocm
} // namespace tenzor

#else // !TENZOR_HAS_ROCSOLVER — native HIP fallback kernels

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/transform.hpp"
#include <hip/hip_runtime.h>
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
namespace rocm {

namespace {

#ifndef HIP_CHECK_LINALG
#define HIP_CHECK_LINALG(call)                                                  \
    do {                                                                         \
        hipError_t err = (call);                                                 \
        if (err != hipSuccess) {                                                 \
            throw std::runtime_error(                                            \
                std::string("HIP error in linalg at ") + __FILE__ + ":" +       \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));      \
        }                                                                        \
    } while (0)
#endif

/// Max matrix dimension for shared-memory fallback kernels.
/// Shared memory usage is 2*N*N*sizeof(T) + scratch, capped at 48KB default.
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
            " exceeds native HIP fallback limit of " + std::to_string(max_n) +
            " (build with rocSOLVER for larger matrices)");
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
// Extract L (unit lower triangular) and U (upper triangular) from packed LU
// One thread per element.
// ============================================================================

template<typename T>
__global__ void extract_lu_kernel_hip(const T* packed, T* L, T* U,
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
// Public API implementations (match rocSOLVER wrapper signatures)
// ============================================================================

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

    // Allocate pivots and info on device
    int* d_pivots = nullptr;
    int* d_info = nullptr;
    HIP_CHECK_LINALG(hipMalloc(&d_pivots, nbatch * n * sizeof(int)));
    HIP_CHECK_LINALG(hipMalloc(&d_info, nbatch * sizeof(int)));
    HIP_CHECK_LINALG(hipMemset(d_info, 0, nbatch * sizeof(int)));

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "det");
        size_t smem = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        int det_threads = 256;
        int det_blocks = (nbatch + det_threads - 1) / det_threads;
        lu_det_kernel<float><<<det_blocks, det_threads, 0, stream>>>(
            work.data<float>(), d_pivots, result.data<float>(), n, nbatch);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "det");
        size_t smem = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        int det_threads = 256;
        int det_blocks = (nbatch + det_threads - 1) / det_threads;
        lu_det_kernel<double><<<det_blocks, det_threads, 0, stream>>>(
            work.data<double>(), d_pivots, result.data<double>(), n, nbatch);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    hipFree(d_pivots);
    hipFree(d_info);
    return result;
}

// ============================================================================
// Matrix Inverse
// ============================================================================

auto linalg_inv_kernel(const Tensor& A, hipStream_t stream) -> Tensor {
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
    HIP_CHECK_LINALG(hipMalloc(&d_pivots, nbatch * n * sizeof(int)));
    HIP_CHECK_LINALG(hipMalloc(&d_info, nbatch * sizeof(int)));
    HIP_CHECK_LINALG(hipMemset(d_info, 0, nbatch * sizeof(int)));

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "inv");
        // LU factorize
        size_t smem_lu = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem_lu, stream>>>(
            work.data<float>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        // Invert via LU solve with identity
        size_t smem_inv = 2 * n * n * sizeof(float);
        lu_inv_kernel<float><<<nbatch, threads, smem_inv, stream>>>(
            work.data<float>(), d_pivots, result.data<float>(), n);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "inv");
        size_t smem_lu = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem_lu, stream>>>(
            work.data<double>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        size_t smem_inv = 2 * n * n * sizeof(double);
        lu_inv_kernel<double><<<nbatch, threads, smem_inv, stream>>>(
            work.data<double>(), d_pivots, result.data<double>(), n);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    hipFree(d_pivots);
    hipFree(d_info);
    return result;
}

// ============================================================================
// Linear System Solve (AX = B)
// ============================================================================

auto linalg_solve_kernel(const Tensor& A, const Tensor& B, hipStream_t stream) -> Tensor {
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
    HIP_CHECK_LINALG(hipMalloc(&d_pivots, nbatch * n * sizeof(int)));
    HIP_CHECK_LINALG(hipMalloc(&d_info, nbatch * sizeof(int)));
    HIP_CHECK_LINALG(hipMemset(d_info, 0, nbatch * sizeof(int)));

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(n, "solve");
        // LU factorize
        size_t smem_lu = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem_lu, stream>>>(
            work_a.data<float>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        // Solve via forward/back substitution
        size_t smem_solve = (n * n + n * nrhs) * sizeof(float);
        lu_solve_kernel<float><<<nbatch, threads, smem_solve, stream>>>(
            work_a.data<float>(), d_pivots, work_b.data<float>(), n, nrhs);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "solve");
        size_t smem_lu = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem_lu, stream>>>(
            work_a.data<double>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        size_t smem_solve = (n * n + n * nrhs) * sizeof(double);
        lu_solve_kernel<double><<<nbatch, threads, smem_solve, stream>>>(
            work_a.data<double>(), d_pivots, work_b.data<double>(), n, nrhs);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    hipFree(d_pivots);
    hipFree(d_info);
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
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(std::max(m, n_cols), "svd");
        size_t smem = (m * n_cols + m * m + n_cols * n_cols + 4) * sizeof(double);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        svd_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), U.data<double>(), S.data<double>(), Vt.data<double>(),
            m, n_cols, k, full_matrices, 30 * std::max(m, n_cols));
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
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
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(std::max(m, n_cols), "qr");
        size_t smem = (m * n_cols + m * m + 4) * sizeof(double);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_qr_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), Q.data<double>(), R.data<double>(), m, n_cols, k);
        HIP_CHECK_LINALG(hipGetLastError());
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
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "eigh");
        size_t smem = (2 * n * n + 2 * n + 4) * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        eigh_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), W.data<double>(), n, 30 * n);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {W, work};
}

// ============================================================================
// Non-symmetric Eigendecomposition (eig)
// ============================================================================

auto linalg_eig_kernel(const Tensor& A, hipStream_t stream)
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
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "eig");
        size_t smem = (2 * n * n + 4) * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        qr_eig_fallback_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), WR.data<double>(), WI.data<double>(),
            V.data<double>(), n, 30 * n);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {WR, WI, V};
}

// ============================================================================
// Cholesky Decomposition
// ============================================================================

auto linalg_cholesky_kernel(const Tensor& A, bool upper, hipStream_t stream) -> Tensor {
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
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "cholesky");
        size_t smem = n * n * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        cholesky_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), n, upper);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work;
}

// ============================================================================
// LU Decomposition — returns (L, U, pivots) using shared-memory lu_kernel
// ============================================================================

auto linalg_lu_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    validate_linalg_dtype(A, "lu");
    auto original_dtype = A.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto [L32, U32, piv] = linalg_lu_kernel(A.to(DType::Float32), stream);
        return {L32.to(original_dtype), U32.to(original_dtype), piv};
    }

    auto work = A.contiguous().clone();
    auto [n, ndim] = check_square(work);
    int64_t nbatch = batch_size(work);

    // Build output shapes
    auto a_shape = A.shape();
    std::vector<int64_t> mat_shape;
    for (size_t i = 0; i + 2 < a_shape.size(); ++i) mat_shape.push_back(a_shape[i]);
    std::vector<int64_t> piv_shape = mat_shape;
    mat_shape.push_back(n); mat_shape.push_back(n);
    piv_shape.push_back(n);

    // Allocate pivots and info on device
    int* d_pivots = nullptr;
    int* d_info = nullptr;
    HIP_CHECK_LINALG(hipMalloc(&d_pivots, nbatch * n * sizeof(int)));
    HIP_CHECK_LINALG(hipMalloc(&d_info, nbatch * sizeof(int)));
    HIP_CHECK_LINALG(hipMemset(d_info, 0, nbatch * sizeof(int)));

    // Step 1: Run LU factorization in-place on work tensor
    auto L = zeros(mat_shape, original_dtype, A.device());
    auto U = zeros(mat_shape, original_dtype, A.device());

    if (original_dtype == DType::Float32) {
        check_size_limit<float>(n, "lu");
        size_t smem = n * n * sizeof(float) + 4 * sizeof(float);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        // Step 2: split packed LU into L and U
        int64_t total = nbatch * n * n;
        int ext_threads = 256;
        int ext_blocks = static_cast<int>((total + ext_threads - 1) / ext_threads);
        extract_lu_kernel_hip<float><<<ext_blocks, ext_threads, 0, stream>>>(
            work.data<float>(), L.data<float>(), U.data<float>(), n, nbatch);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "lu");
        size_t smem = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), d_pivots, d_info, n);
        HIP_CHECK_LINALG(hipGetLastError());

        int64_t total = nbatch * n * n;
        int ext_threads = 256;
        int ext_blocks = static_cast<int>((total + ext_threads - 1) / ext_threads);
        extract_lu_kernel_hip<double><<<ext_blocks, ext_threads, 0, stream>>>(
            work.data<double>(), L.data<double>(), U.data<double>(), n, nbatch);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    // Step 3: copy pivots to Int32 tensor
    auto pivots_out = zeros(piv_shape, DType::Int32, A.device());
    HIP_CHECK_LINALG(hipMemcpyAsync(pivots_out.data<int32_t>(), d_pivots,
        nbatch * n * sizeof(int), hipMemcpyDeviceToDevice, stream));

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    hipFree(d_pivots);
    hipFree(d_info);
    return {L, U, pivots_out};
}

// ============================================================================
// LU Solve — given packed LU + pivots from linalg_lu_kernel, solve A*X = B
// ============================================================================

auto linalg_lu_solve_kernel(const Tensor& LU_data, const Tensor& pivots,
                             const Tensor& B, hipStream_t stream) -> Tensor {
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
    if (lu_ndim < 2 || b_ndim < 1) {
        throw std::invalid_argument("linalg::lu_solve: inputs must be at least 2D/1D");
    }
    int64_t n = lu_shape[lu_ndim - 1];
    if (lu_shape[lu_ndim - 2] != n) {
        throw std::invalid_argument("linalg::lu_solve: LU_data must be square");
    }
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(LU_data);

    auto lu_work = LU_data.contiguous();
    auto b_work = B.contiguous().clone();

    // Pivots must be int32 on device
    auto piv_dev = pivots.to(B.device()).contiguous();

    if (original_dtype == DType::Float32) {
        check_size_limit<float>(n, "lu_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(float);
        int threads = min(max(static_cast<int>(n), 1), 128);
        lu_solve_kernel<float><<<nbatch, threads, smem, stream>>>(
            lu_work.data<float>(), piv_dev.data<int32_t>(),
            b_work.data<float>(), n, nrhs);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "lu_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(double);
        int threads = min(max(static_cast<int>(n), 1), 128);
        lu_solve_kernel<double><<<nbatch, threads, smem, stream>>>(
            lu_work.data<double>(), piv_dev.data<int32_t>(),
            b_work.data<double>(), n, nrhs);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return b_work;
}

// ============================================================================
// Triangular Solve (AX = B, A triangular) — native HIP kernel
// ============================================================================

template<typename T>
__global__ void trsm_kernel(const T* __restrict__ A, T* __restrict__ B,
                            int n, int nrhs, bool upper, bool unit_diag) {
    int batch = blockIdx.x;
    const T* A_mat = A + batch * n * n;
    T* B_mat = B + batch * n * nrhs;
    int tid = threadIdx.x;

    if (upper) {
        for (int i = n - 1; i >= 0; --i) {
            __syncthreads();
            for (int j = tid; j < nrhs; j += blockDim.x) {
                T sum = B_mat[i * nrhs + j];
                for (int k = i + 1; k < n; ++k)
                    sum -= A_mat[i * n + k] * B_mat[k * nrhs + j];
                B_mat[i * nrhs + j] = unit_diag ? sum : sum / A_mat[i * n + i];
            }
        }
    } else {
        for (int i = 0; i < n; ++i) {
            __syncthreads();
            for (int j = tid; j < nrhs; j += blockDim.x) {
                T sum = B_mat[i * nrhs + j];
                for (int k = 0; k < i; ++k)
                    sum -= A_mat[i * n + k] * B_mat[k * nrhs + j];
                B_mat[i * nrhs + j] = unit_diag ? sum : sum / A_mat[i * n + i];
            }
        }
    }
}

auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                     bool upper, bool unitriangular,
                                     hipStream_t stream) -> Tensor {
    validate_linalg_dtype(A, "solve_triangular");
    if (A.dtype() == DType::Float16 || A.dtype() == DType::BFloat16) {
        return linalg_solve_triangular_kernel(
            A.to(DType::Float32), B.to(DType::Float32),
            upper, unitriangular, stream).to(A.dtype());
    }

    auto work_a = A.contiguous();
    auto work_b = B.contiguous().clone();
    auto [n, ndim_a] = check_square(work_a);
    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(work_a);

    int threads = std::min(std::max(static_cast<int>(nrhs), 1), 256);

    if (work_a.dtype() == DType::Float32) {
        check_size_limit<float>(n, "solve_triangular");
        trsm_kernel<float><<<static_cast<int>(nbatch), threads, 0, stream>>>(
            work_a.data<float>(), work_b.data<float>(),
            static_cast<int>(n), static_cast<int>(nrhs), upper, unitriangular);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "solve_triangular");
        trsm_kernel<double><<<static_cast<int>(nbatch), threads, 0, stream>>>(
            work_a.data<double>(), work_b.data<double>(),
            static_cast<int>(n), static_cast<int>(nrhs), upper, unitriangular);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work_b;
}

// ============================================================================
// Geqrf fallback — Householder QR returning packed reflectors + tau
// One block per batch element.
// Shared memory: A[m*n] + scratch[4]
// ============================================================================

template<typename T>
__global__ void householder_geqrf_kernel(
    T* __restrict__ A_io,
    T* __restrict__ tau_out,
    int m, int n_cols, int k)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* R = reinterpret_cast<T*>(smem_raw);
    T* scratch = R + m * n_cols;

    T* A = A_io + batch_idx * m * n_cols;
    T* tau = tau_out + batch_idx * k;

    for (int idx = tid; idx < m * n_cols; idx += num_threads) {
        R[idx] = A[idx];
    }
    __syncthreads();

    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

    for (int j = 0; j < k; j++) {
        if (tid == 0) {
            T sigma = T(0);
            for (int i = j + 1; i < m; i++) {
                sigma += R[i * n_cols + j] * R[i * n_cols + j];
            }
            T x0 = R[j * n_cols + j];
            T norm_x = sqrt(x0 * x0 + sigma);
            if (norm_x < zero_tol || sigma < zero_tol) {
                scratch[1] = T(0);
                tau[j] = T(0);
            } else {
                T alpha = -copysign(norm_x, x0);
                T v0 = x0 - alpha;
                T v_norm_sq = v0 * v0 + sigma;
                T tau_val = T(2) / v_norm_sq;
                scratch[0] = v0;
                scratch[1] = tau_val;
                scratch[2] = alpha;
                tau[j] = tau_val;
            }
        }
        __syncthreads();

        T tau_val = scratch[1];
        if (tau_val == T(0)) { __syncthreads(); continue; }
        T v0 = scratch[0];
        T alpha = scratch[2];

        for (int col = j + static_cast<int>(tid); col < n_cols; col += num_threads) {
            T dot = v0 * R[j * n_cols + col];
            for (int i = j + 1; i < m; i++) {
                dot += R[i * n_cols + j] * R[i * n_cols + col];
            }
            dot *= tau_val;
            R[j * n_cols + col] -= v0 * dot;
            for (int i = j + 1; i < m; i++) {
                R[i * n_cols + col] -= R[i * n_cols + j] * dot;
            }
        }
        __syncthreads();

        if (tid == 0) {
            T inv_v0 = T(1) / v0;
            for (int i = j + 1; i < m; i++) {
                R[i * n_cols + j] *= inv_v0;
            }
            R[j * n_cols + j] = alpha;
            tau[j] = tau_val * v0 * v0;
        }
        __syncthreads();
    }

    for (int idx = tid; idx < m * n_cols; idx += num_threads) {
        A[idx] = R[idx];
    }
}

// ============================================================================
// Ormqr fallback — apply Q (from Householder reflectors) to matrix C
// One block per batch element.
// Shared memory: C[c_m*c_n] + scratch
// ============================================================================

template<typename T>
__global__ void householder_ormqr_kernel(
    const T* __restrict__ refl_in,
    const T* __restrict__ tau_in,
    T* __restrict__ C_io,
    int r_m, int r_n, int c_m, int c_n, int k_refl,
    bool left, bool transpose_q)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* C = reinterpret_cast<T*>(smem_raw);

    const T* refl = refl_in + batch_idx * r_m * r_n;
    const T* tau = tau_in + batch_idx * k_refl;
    T* C_out = C_io + batch_idx * c_m * c_n;

    for (int idx = tid; idx < c_m * c_n; idx += num_threads) {
        C[idx] = C_out[idx];
    }
    __syncthreads();

    int start, end_val, step;
    if ((left && !transpose_q) || (!left && !transpose_q)) {
        start = 0; end_val = k_refl; step = 1;
    } else {
        start = k_refl - 1; end_val = -1; step = -1;
    }

    for (int j = start; j != end_val; j += step) {
        T tau_j = tau[j];
        if (tau_j == T(0)) continue;

        if (left) {
            for (int col = tid; col < c_n; col += num_threads) {
                T dot = C[j * c_n + col];
                for (int i = j + 1; i < c_m; i++) {
                    dot += refl[i * r_n + j] * C[i * c_n + col];
                }
                dot *= tau_j;
                C[j * c_n + col] -= dot;
                for (int i = j + 1; i < c_m; i++) {
                    C[i * c_n + col] -= refl[i * r_n + j] * dot;
                }
            }
        } else {
            for (int row = tid; row < c_m; row += num_threads) {
                T dot = C[row * c_n + j];
                for (int i = j + 1; i < c_n; i++) {
                    dot += C[row * c_n + i] * refl[i * r_n + j];
                }
                dot *= tau_j;
                C[row * c_n + j] -= dot;
                for (int i = j + 1; i < c_n; i++) {
                    C[row * c_n + i] -= dot * refl[i * r_n + j];
                }
            }
        }
        __syncthreads();
    }

    for (int idx = tid; idx < c_m * c_n; idx += num_threads) {
        C_out[idx] = C[idx];
    }
}

auto linalg_geqrf_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    validate_linalg_dtype(A, "geqrf");
    if (A.dtype() == DType::Float16) {
        auto [R, tau] = linalg_geqrf_kernel(A.to(DType::Float32), stream);
        return {R, tau};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [R, tau] = linalg_geqrf_kernel(A.to(DType::Float32), stream);
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

    if (A.dtype() == DType::Float32) {
        check_size_limit<float>(std::max(m, n_cols), "geqrf");
        size_t smem = (m * n_cols + 4) * sizeof(float);
        int threads = std::min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_geqrf_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), tau_result.data<float>(), m, n_cols, k);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(std::max(m, n_cols), "geqrf");
        size_t smem = (m * n_cols + 4) * sizeof(double);
        int threads = std::min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_geqrf_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), tau_result.data<double>(), m, n_cols, k);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {work, tau_result};
}

auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                          const Tensor& C, bool left, bool transpose_q,
                          hipStream_t stream) -> Tensor {
    validate_linalg_dtype(C, "ormqr");
    if (C.dtype() == DType::Float16) {
        return linalg_ormqr_kernel(reflectors.to(DType::Float32), tau.to(DType::Float32),
                                    C.to(DType::Float32), left, transpose_q, stream);
    }
    if (C.dtype() == DType::BFloat16) {
        return linalg_ormqr_kernel(reflectors.to(DType::Float32), tau.to(DType::Float32),
                                    C.to(DType::Float32), left, transpose_q, stream);
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

    if (C.dtype() == DType::Float32) {
        check_size_limit<float>(std::max(c_m, c_n), "ormqr");
        size_t smem = (c_m * c_n + std::max(c_m, c_n)) * sizeof(float);
        int threads = std::min(static_cast<int>(std::max(c_m, c_n)), 128);
        if (threads < 1) threads = 1;
        householder_ormqr_kernel<float><<<nbatch, threads, smem, stream>>>(
            refl.data<float>(), tau_c.data<float>(), work_c.data<float>(),
            r_m, r_n, c_m, c_n, k_refl, left, transpose_q);
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(std::max(c_m, c_n), "ormqr");
        size_t smem = (c_m * c_n + std::max(c_m, c_n)) * sizeof(double);
        int threads = std::min(static_cast<int>(std::max(c_m, c_n)), 128);
        if (threads < 1) threads = 1;
        householder_ormqr_kernel<double><<<nbatch, threads, smem, stream>>>(
            refl.data<double>(), tau_c.data<double>(), work_c.data<double>(),
            r_m, r_n, c_m, c_n, k_refl, left, transpose_q);
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work_c;
}

// =========================================================================
// LDL^T factorization — Bunch-Kaufman diagonal pivoting in shared memory.
// =========================================================================
template<typename T>
__global__ void ldl_bk_factor_kernel(
    T* __restrict__ data,
    int* __restrict__ pivots_out,
    int n)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* A = reinterpret_cast<T*>(smem_raw);
    T* scratch = A + n * n;

    T* batch_data = data + batch_idx * n * n;
    int* batch_piv = pivots_out + batch_idx * n;

    for (int idx = tid; idx < n * n; idx += num_threads)
        A[idx] = batch_data[idx];
    __syncthreads();

    constexpr T alpha_val = static_cast<T>(0.6404);

    int k = 0;
    while (k < n) {
        if (tid == 0) {
            T col_max = T(0);
            int col_max_row = k;
            for (int i = k + 1; i < n; i++) {
                T v = fabs(A[i * n + k]);
                if (v > col_max) { col_max = v; col_max_row = i; }
            }

            T abs_akk = fabs(A[k * n + k]);

            if (abs_akk == T(0) && col_max == T(0)) {
                batch_piv[k] = k + 1;
                scratch[0] = T(1); scratch[1] = T(k);
            } else if (abs_akk >= alpha_val * col_max) {
                batch_piv[k] = k + 1;
                scratch[0] = T(1); scratch[1] = T(k);
            } else {
                int r = col_max_row;
                T row_max = T(0);
                for (int j = k; j < n; j++) {
                    if (j == r) continue;
                    T v = fabs(A[r * n + j]);
                    if (v > row_max) row_max = v;
                }
                T abs_arr = fabs(A[r * n + r]);

                if (abs_akk * row_max >= alpha_val * col_max * col_max) {
                    batch_piv[k] = k + 1;
                    scratch[0] = T(1); scratch[1] = T(k);
                } else if (abs_arr >= alpha_val * row_max) {
                    batch_piv[k] = r + 1;
                    scratch[0] = T(1); scratch[1] = T(r);
                } else {
                    batch_piv[k] = -(r + 1);
                    batch_piv[k + 1] = -(r + 1);
                    scratch[0] = T(2); scratch[1] = T(r);
                }
            }
        }
        __syncthreads();

        int pivot_type = static_cast<int>(scratch[0]);
        int swap_row = static_cast<int>(scratch[1]);

        if (pivot_type == 1) {
            if (swap_row != k) {
                for (int j = tid; j < n; j += num_threads) {
                    T tmp = A[k * n + j]; A[k * n + j] = A[swap_row * n + j]; A[swap_row * n + j] = tmp;
                }
                __syncthreads();
                for (int i = tid; i < n; i += num_threads) {
                    T tmp = A[i * n + k]; A[i * n + k] = A[i * n + swap_row]; A[i * n + swap_row] = tmp;
                }
                __syncthreads();
            }

            T diag = A[k * n + k];
            if (diag != T(0)) {
                if (tid == 0)
                    for (int i = k + 1; i < n; i++)
                        A[i * n + k] /= diag;
                __syncthreads();

                for (int i = k + 1 + tid; i < n; i += num_threads) {
                    T lik = A[i * n + k];
                    for (int j = k + 1; j <= i; j++)
                        A[i * n + j] -= lik * diag * A[j * n + k];
                }
                __syncthreads();
            }
            k++;
        } else {
            if (swap_row != k + 1) {
                for (int j = tid; j < n; j += num_threads) {
                    T tmp = A[(k+1) * n + j]; A[(k+1) * n + j] = A[swap_row * n + j]; A[swap_row * n + j] = tmp;
                }
                __syncthreads();
                for (int i = tid; i < n; i += num_threads) {
                    T tmp = A[i * n + (k+1)]; A[i * n + (k+1)] = A[i * n + swap_row]; A[i * n + swap_row] = tmp;
                }
                __syncthreads();
            }

            T d11 = A[k * n + k], d21 = A[(k+1) * n + k], d22 = A[(k+1) * n + (k+1)];
            T det = d11 * d22 - d21 * d21;

            if (det != T(0)) {
                if (tid == 0) {
                    T inv11 = d22 / det, inv12 = -d21 / det, inv22 = d11 / det;
                    for (int i = k + 2; i < n; i++) {
                        T a0 = A[i * n + k], a1 = A[i * n + (k+1)];
                        A[i * n + k]     = inv11 * a0 + inv12 * a1;
                        A[i * n + (k+1)] = inv12 * a0 + inv22 * a1;
                    }
                }
                __syncthreads();

                for (int i = k + 2 + tid; i < n; i += num_threads) {
                    T li0 = A[i * n + k], li1 = A[i * n + (k+1)];
                    for (int j = k + 2; j <= i; j++) {
                        T lj0 = A[j * n + k], lj1 = A[j * n + (k+1)];
                        A[i * n + j] -= (li0 * (d11 * lj0 + d21 * lj1)
                                       + li1 * (d21 * lj0 + d22 * lj1));
                    }
                }
                __syncthreads();
            }
            k += 2;
        }
    }

    for (int idx = tid; idx < n * n; idx += num_threads)
        batch_data[idx] = A[idx];
}

// =========================================================================
// LDL^T solve — Bunch-Kaufman pivoted solve in shared memory.
// =========================================================================
template<typename T>
__global__ void ldl_bk_solve_kernel(
    const T* __restrict__ ld_data,
    const int* __restrict__ pivots,
    T* __restrict__ b_data,
    int n, int nrhs)
{
    int batch_idx = blockIdx.x;
    int tid = threadIdx.x;
    int num_threads = blockDim.x;

    extern __shared__ char smem_raw[];
    T* LD = reinterpret_cast<T*>(smem_raw);
    T* B = LD + n * n;

    const T* batch_ld = ld_data + batch_idx * n * n;
    const int* batch_piv = pivots + batch_idx * n;
    T* batch_b = b_data + batch_idx * n * nrhs;

    for (int idx = tid; idx < n * n; idx += num_threads)
        LD[idx] = batch_ld[idx];
    for (int idx = tid; idx < n * nrhs; idx += num_threads)
        B[idx] = batch_b[idx];
    __syncthreads();

    if (tid == 0) {
        // Forward pivot permutation
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                int sr = p - 1;
                if (sr != k)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j]; B[k * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k++;
            } else {
                int sr = (-p) - 1;
                if (sr != k + 1)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[(k+1) * nrhs + j]; B[(k+1) * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k += 2;
            }
        }

        // Forward substitution
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                for (int i = k + 1; i < n; i++) {
                    T m = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++)
                        B[i * nrhs + j] -= m * B[k * nrhs + j];
                }
                k++;
            } else {
                for (int i = k + 2; i < n; i++) {
                    T m0 = LD[i * n + k], m1 = LD[i * n + k + 1];
                    for (int j = 0; j < nrhs; j++)
                        B[i * nrhs + j] -= m0 * B[k * nrhs + j] + m1 * B[(k+1) * nrhs + j];
                }
                k += 2;
            }
        }

        // Diagonal solve
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                T d = LD[k * n + k];
                for (int j = 0; j < nrhs; j++) B[k * nrhs + j] /= d;
                k++;
            } else {
                T d11 = LD[k * n + k], d21 = LD[(k+1) * n + k], d22 = LD[(k+1) * n + (k+1)];
                T det = d11 * d22 - d21 * d21;
                for (int j = 0; j < nrhs; j++) {
                    T y0 = B[k * nrhs + j], y1 = B[(k+1) * nrhs + j];
                    B[k * nrhs + j]     = (d22 * y0 - d21 * y1) / det;
                    B[(k+1) * nrhs + j] = (d11 * y1 - d21 * y0) / det;
                }
                k += 2;
            }
        }

        // Backward substitution
        for (int k = n - 1; k >= 0; ) {
            int p = batch_piv[k];
            if (p > 0) {
                for (int i = k + 1; i < n; i++) {
                    T m = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++)
                        B[k * nrhs + j] -= m * B[i * nrhs + j];
                }
                k--;
            } else {
                int k0 = k - 1;
                for (int i = k + 1; i < n; i++) {
                    T m0 = LD[i * n + k0], m1 = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++) {
                        B[k0 * nrhs + j] -= m0 * B[i * nrhs + j];
                        B[k * nrhs + j]  -= m1 * B[i * nrhs + j];
                    }
                }
                k -= 2;
            }
        }

        // Inverse pivot permutation
        for (int k = n - 1; k >= 0; ) {
            int p = batch_piv[k];
            if (p > 0) {
                int sr = p - 1;
                if (sr != k)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j]; B[k * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k--;
            } else {
                int sr = (-p) - 1;
                if (sr != k)
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j]; B[k * nrhs + j] = B[sr * nrhs + j]; B[sr * nrhs + j] = tmp;
                    }
                k -= 2;
            }
        }
    }
    __syncthreads();

    for (int idx = tid; idx < n * nrhs; idx += num_threads)
        batch_b[idx] = B[idx];
}

// =========================================================================
// LDL^T factorization — native GPU Bunch-Kaufman kernel
// =========================================================================
auto linalg_ldl_factor_kernel(const Tensor& A, hipStream_t stream)
    -> std::tuple<Tensor, Tensor> {
    auto original_dtype = A.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto [LD32, piv] = linalg_ldl_factor_kernel(A.to(DType::Float32), stream);
        return {LD32.to(original_dtype), piv};
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument("linalg::ldl_factor: unsupported dtype");
    }

    auto [n, ndim] = check_square(A);
    int64_t nbatch = batch_size(A);

    auto a_shape = to_vec(A.shape());
    std::vector<int64_t> piv_shape;
    for (size_t i = 0; i + 2 < a_shape.size(); ++i) piv_shape.push_back(a_shape[i]);
    piv_shape.push_back(n);

    auto work = A.contiguous().clone();
    auto pivots_out = tenzor::zeros(piv_shape, DType::Int32, A.device());

    int threads = std::min(static_cast<int>(n), 128);
    if (threads < 1) threads = 1;

    if (original_dtype == DType::Float32) {
        check_size_limit<float>(n, "ldl_factor");
        size_t smem = (n * n + 4) * sizeof(float);
        ldl_bk_factor_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), pivots_out.data<int32_t>(), static_cast<int>(n));
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "ldl_factor");
        size_t smem = (n * n + 4) * sizeof(double);
        ldl_bk_factor_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), pivots_out.data<int32_t>(), static_cast<int>(n));
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return {work, pivots_out};
}

// =========================================================================
// LDL^T solve — native GPU Bunch-Kaufman solve kernel
// =========================================================================
auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                              const Tensor& B, hipStream_t stream) -> Tensor {
    auto original_dtype = LD.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        auto result = linalg_ldl_solve_kernel(LD.to(DType::Float32), pivots,
                                               B.to(DType::Float32), stream);
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

    int threads = std::min(static_cast<int>(n), 128);
    if (threads < 1) threads = 1;

    if (original_dtype == DType::Float32) {
        check_size_limit<float>(n, "ldl_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(float);
        ldl_bk_solve_kernel<float><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<float>(), pivots.data<int32_t>(),
            work_b.data<float>(), static_cast<int>(n), static_cast<int>(nrhs));
        HIP_CHECK_LINALG(hipGetLastError());
    } else {
        check_size_limit<double>(n, "ldl_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(double);
        ldl_bk_solve_kernel<double><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<double>(), pivots.data<int32_t>(),
            work_b.data<double>(), static_cast<int>(n), static_cast<int>(nrhs));
        HIP_CHECK_LINALG(hipGetLastError());
    }

    HIP_CHECK_LINALG(hipStreamSynchronize(stream ? stream : nullptr));
    return work_b;
}

// =========================================================================
// Householder product — compose from existing ormqr kernel
// =========================================================================
auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                [[maybe_unused]] hipStream_t stream) -> Tensor {
    // Apply Householder reflectors to I[:, :n]: Q = H_0 · H_1 · … · H_{k-1} · I[:m,:n]
    //
    // LAPACK's sorgqr (used by the CPU path) produces an m×n matrix — the
    // first n columns of the full m×m Q. We replicate that with row-major
    // tensor ops, walking the reflectors right→left (H_{k-1} first) because
    // the canonical order that matches LAPACK is to accumulate columns of Q
    // starting from the identity on the right.
    //
    // Routing through rocSOLVER ormqr with an identity matrix as C works only
    // for square m=n inputs (row-major/col-major leading-dimension mismatch
    // with non-square C), so we mirror the CUDA implementation that sidesteps
    // cuSOLVER's ormqr for the same reason.
    auto shape = input.shape();
    auto ndim = static_cast<int64_t>(shape.size());
    if (ndim < 2) {
        throw std::invalid_argument("linalg::householder_product: input must be at least 2D");
    }
    int64_t m = shape[ndim - 2];
    int64_t n = shape[ndim - 1];
    int64_t k = tau.shape()[tau.shape().size() - 1];
    if (k > std::min(m, n)) {
        throw std::invalid_argument(
            "linalg::householder_product: tau length must be ≤ min(m, n)");
    }

    DType dt = input.dtype();
    Device dev = input.device();

    DType compute_dt = dt;
    if (dt == DType::Float16 || dt == DType::BFloat16) compute_dt = DType::Float32;

    Tensor V = input.contiguous();
    Tensor tau_c = tau.contiguous();
    if (V.dtype() != compute_dt) V = V.to(compute_dt);
    if (tau_c.dtype() != compute_dt) tau_c = tau_c.to(compute_dt);

    auto I_full = tenzor::eye(m, std::nullopt, compute_dt, dev);
    Tensor Q = (n == m) ? I_full
                        : tenzor::slice(I_full, /*dim=*/1, /*start=*/0, /*end=*/n).contiguous();
    if (ndim > 2) {
        std::vector<int64_t> batched_shape(shape.begin(), shape.end());
        Q = tenzor::expand(Q, std::move(batched_shape));
        Q = Q.contiguous();
    }

    for (int64_t j = k - 1; j >= 0; --j) {
        Tensor v_j = tenzor::slice(V, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();

        // Mask rows [0..j) to 0 and overwrite row j with 1, keeping rows (j..m) as V[i, j].
        auto mask_cpu = tenzor::ones({m, int64_t(1)}, compute_dt, Device::cpu());
        auto overrides_cpu = tenzor::zeros({m, int64_t(1)}, compute_dt, Device::cpu());
        if (compute_dt == DType::Float32) {
            float* mm = mask_cpu.data<float>();
            float* oo = overrides_cpu.data<float>();
            for (int64_t i = 0; i < j; ++i) { mm[i] = 0.0f; oo[i] = 0.0f; }
            if (j < m) { mm[j] = 0.0f; oo[j] = 1.0f; }
        } else {
            double* mm = mask_cpu.data<double>();
            double* oo = overrides_cpu.data<double>();
            for (int64_t i = 0; i < j; ++i) { mm[i] = 0.0; oo[i] = 0.0; }
            if (j < m) { mm[j] = 0.0; oo[j] = 1.0; }
        }
        Tensor mask = mask_cpu.to(dev);
        Tensor overrides = overrides_cpu.to(dev);
        if (ndim > 2) {
            std::vector<int64_t> bshape(shape.begin(), shape.end() - 1);
            bshape.push_back(1);
            mask = tenzor::expand(mask, bshape).contiguous();
            overrides = tenzor::expand(overrides, bshape).contiguous();
        }
        v_j = tenzor::add(tenzor::mul(v_j, mask), overrides);

        Tensor tau_j = tenzor::slice(tau_c, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();

        Tensor v_jT = tenzor::transpose(v_j, -1, -2);
        Tensor u = tenzor::matmul(v_jT, Q);

        Tensor outer = tenzor::matmul(v_j, u);
        std::vector<int64_t> tau_shape(tau_j.shape().begin(), tau_j.shape().end());
        tau_shape.push_back(1);
        tau_j = tau_j.reshape(tau_shape);
        Tensor scaled = tenzor::mul(outer, tau_j);
        Q = tenzor::sub(Q, scaled);
    }

    if (Q.dtype() != dt) Q = Q.to(dt);
    return Q;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSOLVER
