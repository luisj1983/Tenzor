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
#include "../cublas_handle_pool.hpp"

#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
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

/// NN.13: Stream-ordered RAII wrapper used for pivot / info workspaces in the
/// custom LU paths (det / inv / solve / lu) that previously did bare
/// cudaMalloc + cudaFree per call. Bare cudaFree implicitly performs a
/// device-wide sync, which destroys stream concurrency; cudaFreeAsync defers
/// the free until prior stream work retires. Mirrors HH.8's CudaDevicePtr in
/// fft.cu.
template<typename T>
struct StreamWorkspace {
    T* ptr = nullptr;
    cudaStream_t stream = nullptr;

    StreamWorkspace() = default;
    StreamWorkspace(int64_t count, cudaStream_t s)
        : stream(s) {
        if (count > 0) {
            CUDA_CHECK_LINALG(cudaMallocAsync(&ptr, count * sizeof(T), stream));
        }
    }
    ~StreamWorkspace() {
        if (ptr) cudaFreeAsync(ptr, stream);
    }

    StreamWorkspace(const StreamWorkspace&) = delete;
    StreamWorkspace& operator=(const StreamWorkspace&) = delete;
    StreamWorkspace(StreamWorkspace&& o) noexcept
        : ptr(o.ptr), stream(o.stream) { o.ptr = nullptr; }

    T* get() const { return ptr; }
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

/// Extract R (upper triangle) from col-major QR result into row-major R.
/// a_data is col-major (m × n_cols); r_data is row-major (k × n_cols).
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
    // col-major read: a_mat[j*m + i]
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
    // col-major read: a_mat[j*m + i]
    q_mat[i * k + j] = a_mat[j * m + i];
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
    q_mat[i * k + j] = a_mat[j * m + i];
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

// ============================================================================
// Non-symmetric eigendecomposition fallback (Hessenberg + Francis QR).
//
// cuSOLVER through CUDA 13 only ships the new generic `cusolverDnXgeev`
// (CUDA 11.6+); the legacy `cusolverDn[SD]geev` symbols used by the
// previous implementation no longer exist in libcusolver.so.12. Rather
// than tracking the new generic API's complex-W contract for real inputs,
// we reuse the same self-contained Hessenberg+Francis QR kernel that the
// `!TENZOR_HAS_CUSOLVER` branch uses. One block per batch, row-major.
// ============================================================================

template<typename T>
__global__ void qr_eig_fallback_kernel_cs(
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

    extern __shared__ char smem_raw_eig_cs[];
    T* H = reinterpret_cast<T*>(smem_raw_eig_cs);
    T* Q = H + n * n;
    T* scratch = Q + n * n;

    const T* A = A_in + batch_idx * n * n;
    T* wr = wr_out + batch_idx * n;
    T* wi = wi_out + batch_idx * n;
    T* V = V_out + batch_idx * n * n;

    for (int idx = static_cast<int>(tid); idx < n * n; idx += static_cast<int>(num_threads)) {
        H[idx] = A[idx];
        int row = idx / n;
        int col = idx % n;
        Q[idx] = (row == col) ? T(1) : T(0);
    }
    __syncthreads();

    for (int k = 0; k + 2 < n; k++) {
        if (tid == 0) {
            T sigma = T(0);
            for (int i = k + 1; i < n; i++) {
                sigma += H[i * n + k] * H[i * n + k];
            }
            T norm_x = ::sqrt(sigma);
            if (norm_x < zero_tol) {
                scratch[1] = T(0);
            } else {
                T a = -::copysign(norm_x, H[(k + 1) * n + k]);
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

    if (tid == 0) {
        scratch[3] = static_cast<T>(n);
    }
    __syncthreads();

    for (int iter = 0; iter < max_iterations; iter++) {
        int nn = static_cast<int>(scratch[3]);
        if (nn <= 0) break;

        if (tid == 0) {
            bool deflated = false;

            if (nn >= 2) {
                T tst = ::fabs(H[(nn - 2) * n + (nn - 2)]) + ::fabs(H[(nn - 1) * n + (nn - 1)]);
                if (tst == T(0)) tst = T(1);
                if (::fabs(H[(nn - 1) * n + (nn - 2)]) < eps * tst) {
                    wr[nn - 1] = H[(nn - 1) * n + (nn - 1)];
                    wi[nn - 1] = T(0);
                    H[(nn - 1) * n + (nn - 2)] = T(0);
                    scratch[3] = static_cast<T>(nn - 1);
                    deflated = true;
                }
            }

            if (!deflated && nn >= 3) {
                T tst = ::fabs(H[(nn - 3) * n + (nn - 3)]) + ::fabs(H[(nn - 2) * n + (nn - 2)]);
                if (tst == T(0)) tst = T(1);
                if (::fabs(H[(nn - 2) * n + (nn - 3)]) < eps * tst) {
                    T a = H[(nn - 2) * n + (nn - 2)], b = H[(nn - 2) * n + (nn - 1)];
                    T c = H[(nn - 1) * n + (nn - 2)], d = H[(nn - 1) * n + (nn - 1)];
                    T trace = a + d;
                    T det = a * d - b * c;
                    T disc = trace * trace - T(4) * det;

                    if (disc >= T(0)) {
                        T sq = ::sqrt(disc);
                        wr[nn - 2] = T(0.5) * (trace + sq);
                        wi[nn - 2] = T(0);
                        wr[nn - 1] = T(0.5) * (trace - sq);
                        wi[nn - 1] = T(0);
                    } else {
                        T sq = ::sqrt(-disc);
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
            T norm_v = ::sqrt(x * x + y * y + z * z);
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

            T alpha_h = -::copysign(norm_v, x);
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

            for (int j = k + static_cast<int>(tid); j < nn; j += static_cast<int>(num_threads)) {
                T dot = v0 * H[k * n + j] + v1 * H[(k + 1) * n + j];
                if (k + 2 < nn) dot += v2 * H[(k + 2) * n + j];
                dot *= tau_h;
                H[k * n + j] -= v0 * dot;
                H[(k + 1) * n + j] -= v1 * dot;
                if (k + 2 < nn) H[(k + 2) * n + j] -= v2 * dot;
            }
            __syncthreads();

            for (int i = static_cast<int>(tid); i < m_lim; i += static_cast<int>(num_threads)) {
                T dot = v0 * H[i * n + k] + v1 * H[i * n + (k + 1)];
                if (k + 2 < nn) dot += v2 * H[i * n + (k + 2)];
                dot *= tau_h;
                H[i * n + k] -= v0 * dot;
                H[i * n + (k + 1)] -= v1 * dot;
                if (k + 2 < nn) H[i * n + (k + 2)] -= v2 * dot;
            }
            __syncthreads();

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

        if (nn >= 2) {
            T norm_v = ::sqrt(x * x + y * y);
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
                T sq = ::sqrt(disc);
                wr[0] = T(0.5) * (trace + sq);
                wi[0] = T(0);
                wr[1] = T(0.5) * (trace - sq);
                wi[1] = T(0);
            } else {
                T sq = ::sqrt(-disc);
                wr[0] = T(0.5) * trace;
                wi[0] = T(0.5) * sq;
                wr[1] = T(0.5) * trace;
                wi[1] = T(-0.5) * sq;
            }
        }
    }
    __syncthreads();

    for (int idx = static_cast<int>(tid); idx < n * n; idx += static_cast<int>(num_threads)) {
        V[idx] = Q[idx];
    }
}

// ----------------------------------------------------------------------------
// Symmetry probe: max |A| and max |A - A^T| over a batched square matrix,
// reduced into two device-side scalars. Used by linalg_eig_kernel to choose
// between the eigh fast-path and the QR fallback without downloading the
// whole input to the host.
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
        // Atomic max over batches via CAS. atomicMax for double isn't
        // built-in pre-Hopper, so emulate.
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
                                                      cudaStream_t stream) {
    DeviceWorkspace scratch(2 * sizeof(double));
    double init[2] = {0.0, 0.0};
    CUDA_CHECK_LINALG(cudaMemcpyAsync(scratch.ptr, init, 2 * sizeof(double),
                                      cudaMemcpyHostToDevice, stream));
    double* d_max_abs = static_cast<double*>(scratch.ptr);
    double* d_max_diff = d_max_abs + 1;

    int threads = 256;
    size_t smem_bytes = 2 * threads * sizeof(double);
    eig_symmetry_probe_kernel<T><<<static_cast<unsigned>(nbatch), threads,
                                    smem_bytes, stream>>>(
        d_A, nbatch, n, d_max_abs, d_max_diff);
    CUDA_CHECK_LINALG(cudaGetLastError());

    double host_pair[2] = {0.0, 0.0};
    CUDA_CHECK_LINALG(cudaMemcpyAsync(host_pair, d_max_abs, 2 * sizeof(double),
                                      cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream));
    return {host_pair[0], host_pair[1]};
}

} // anonymous namespace

// ============================================================================
// Determinant
// ============================================================================

auto linalg_det_kernel(const Tensor& A, cudaStream_t stream) -> Tensor {
    // cuSOLVER LU only supports Float32 / Float64. Widen Float16 / BFloat16
    // to Float32 for the computation, then narrow the result back.
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
    // Plain 2D input → scalar output (shape {}); torch.linalg.det contract.

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
    // cuSolver interprets inputs column-major. Tenzor stores row-major.
    // Transposing via .transpose(-1,-2).contiguous() produces row-major
    // storage of A^T, which cuSolver reads as col-major equal to A. Solve
    // then writes X into work_b's memory in col-major; transpose back to
    // hand the caller a row-major result.
    auto work_a = tenzor::transpose(A.contiguous(), -1, -2).contiguous();
    auto work_b = tenzor::transpose(B.contiguous(), -1, -2).contiguous();
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
    // Convert X from col-major back to the caller's row-major layout.
    return tenzor::transpose(work_b, -1, -2).contiguous();
}

// ============================================================================
// SVD (Singular Value Decomposition)
// ============================================================================

auto linalg_svd_kernel(const Tensor& A, bool full_matrices, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::svd: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];

    // cusolverDnSgesvd requires m_arg >= n_arg in col-major. The wide-matrix
    // path below passes our row-major A as col-major A^T with m_arg=N, n_arg=M
    // (which satisfies the constraint when M <= N). For tall matrices (M > N)
    // that same path would violate m_arg >= n_arg and cuSOLVER returns status=3.
    //
    // Handle the tall case by computing SVD(A^T) using the wide path: if
    // A = U S Vt, then A^T = V S U^T, so the wide-path returns (V, S, U^T).
    // We swap the U and Vt outputs (and transpose them) to recover SVD(A).
    if (m > n_cols) {
        Tensor A_T = tenzor::transpose(A, -2, -1).contiguous();
        auto [U_of_AT, S_of_AT, Vt_of_AT] =
            linalg_svd_kernel(A_T, full_matrices, stream);
        // Our U = transpose(Vt_of_AT); our Vt = transpose(U_of_AT).
        auto U = tenzor::transpose(Vt_of_AT, -2, -1).contiguous();
        auto Vt = tenzor::transpose(U_of_AT, -2, -1).contiguous();
        return {U, S_of_AT, Vt};
    }

    auto work = A.contiguous().clone();
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

            // Row-major A[m][n] => column-major A^T[n][m] in memory.
            // We call gesvd with m_gesvd=n_cols, n_gesvd=m. For the "wide"
            // branch reached here we have m <= n_cols, so constraint m>=n is
            // OK. gesvd returns U_g (m_gesvd, k_g) col-major and Vt_g (k_g,
            // n_gesvd) col-major where k_g = min(m_gesvd, n_gesvd) = m.
            // Our U slot gets Vt_g; our Vt slot gets U_g. Leading dims for
            // gesvd are the col-major leading dim = first shape element:
            //   gesvd's ldu = m_gesvd = n_cols   (goes into our ldvt_arg)
            //   gesvd's ldvt = k_g = m           (goes into our ldu_arg)
            // Previously ldvt was set to k instead of n_cols, which is
            // correct only for square/wide-equal-square cases; for tall
            // matrices routed via the recursive transpose it violates the
            // leading-dimension contract and cuSOLVER returns status 3.
            int ldu_arg = m;            // for Vt_g output buffer
            int ldvt_arg = full_matrices ? n_cols : n_cols;  // for U_g output buffer

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgesvd_bufferSize(handle, n_cols, m, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            float* rwork = nullptr;  // not needed for real SVD
            CUSOLVER_CHECK(cusolverDnSgesvd(handle, jobu, jobvt,
                n_cols, m,  // swapped: col-major sees our row-major as transposed
                a_mat, n_cols,  // lda = n_cols (leading dim of col-major A^T)
                s_vec,
                vt_mat, ldvt_arg,  // "U" output → our Vt slot (col-major n_cols x k)
                u_mat, ldu_arg,    // "Vt" output → our U slot (col-major k x m)
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

            // Leading dims match gesvd's expectations of col-major output
            // shapes: U_g(n_cols, k_g) has leading dim n_cols; Vt_g(k_g, m)
            // has leading dim k_g = min(n_cols, m) = m for the wide branch.
            int ldu_arg = m;
            int ldvt_arg = n_cols;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgesvd_bufferSize(handle, n_cols, m, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            double* rwork = nullptr;
            CUSOLVER_CHECK(cusolverDnDgesvd(handle, jobu, jobvt,
                n_cols, m,
                a_mat, n_cols,
                s_vec,
                vt_mat, ldvt_arg,
                u_mat, ldu_arg,
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
    auto shape = A.shape();
    auto a_ndim = static_cast<int64_t>(shape.size());
    if (a_ndim < 2) throw std::invalid_argument("linalg::qr: input must be at least 2D");

    int64_t m = shape[a_ndim - 2];
    int64_t n_cols = shape[a_ndim - 1];
    int64_t k = std::min(m, n_cols);

    // cuSolver geqrf operates column-major. Transposing row-major A
    // (shape m×n) into row-major A^T (shape n×m) and handing that buffer
    // to cuSolver makes cuSolver see col-major A with the correct m rows
    // and n cols. The extract_r / copy_q_columns helpers then read the
    // result as col-major and write row-major outputs.
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

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* q_data = Q.data<float>();
        float* r_data = R.data<float>();

        backend::CachedMemoryGuard tau_guard(k * sizeof(float));
        float* d_tau = static_cast<float*>(tau_guard.get());

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, m, n_cols, a_mat, m, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgeqrf(handle, m, n_cols,
                a_mat, m, d_tau,
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f32<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            int lwork_q = 0;
            CUSOLVER_CHECK(cusolverDnSorgqr_bufferSize(handle, m, k, k,
                a_mat, m, d_tau, &lwork_q));
            DeviceWorkspace workspace_q(lwork_q * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSorgqr(handle, m, k, k,
                a_mat, m, d_tau,
                static_cast<float*>(workspace_q.ptr), lwork_q, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

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
            CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(handle, m, n_cols, a_mat, m, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgeqrf(handle, m, n_cols,
                a_mat, m, d_tau,
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "qr");

            int threads = 256;
            int total_r = k * n_cols;
            int blocks = (total_r + threads - 1) / threads;
            extract_r_f64<<<blocks, threads, 0, stream>>>(
                a_mat, r_data + b * k * n_cols, m, n_cols, k, 1);

            int lwork_q = 0;
            CUSOLVER_CHECK(cusolverDnDorgqr_bufferSize(handle, m, k, k,
                a_mat, m, d_tau, &lwork_q));
            DeviceWorkspace workspace_q(lwork_q * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDorgqr(handle, m, k, k,
                a_mat, m, d_tau,
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
    //
    // cuSOLVER syevd stores eigenvectors as COLUMNS of a COLUMN-MAJOR matrix.
    // When the same memory is reinterpreted as row-major, each row is an
    // eigenvector — i.e. we hold V^T. Transpose so callers see row-major V
    // with eigenvectors as columns (matching CPU/LAPACKE semantics).
    auto V = tenzor::transpose(work, -2, -1).contiguous();
    return {W, V};
}

// ============================================================================
// Non-symmetric Eigendecomposition (eig)
// ============================================================================

auto linalg_eig_kernel(const Tensor& A, cudaStream_t stream)
    -> std::tuple<Tensor, Tensor, Tensor> {
    // Non-symmetric eigendecomposition.
    //
    // cuSOLVER through CUDA 13 ships only the generic `cusolverDnXgeev`,
    // whose real-input contract requires complex W/VL buffers; the legacy
    // `cusolverDn[SD]geev` is gone. Until we wire up the new generic API,
    // dispatch to `eigh` when the input is (numerically) symmetric — this
    // covers gradcheck (which uses SPD inputs) and any real-physical
    // problem whose autograd backward formula assumes real eigenvalues.
    // For non-symmetric inputs we fall back to the in-tree
    // Hessenberg + Francis QR kernel; that fallback is approximate for
    // n>3 SPD-like matrices but is the only available path here.
    if (A.dtype() == DType::Float16) {
        auto [wr, wi, V] = linalg_eig_kernel(A.to(DType::Float32), stream);
        return {wr, wi, V};
    }
    if (A.dtype() == DType::BFloat16) {
        auto [wr, wi, V] = linalg_eig_kernel(A.to(DType::Float32), stream);
        return {wr, wi, V};
    }
    if (A.dtype() != DType::Float32 && A.dtype() != DType::Float64) {
        throw std::runtime_error("eig: only Float32 and Float64 supported");
    }

    auto [n, ndim] = check_square(A);

    // Approximate-symmetry test on the trailing 2D slice (per batch
    // element). gradcheck perturbs A by ε≈1e-6 (Float64) for finite-diff
    // — that breaks strict symmetry but each element is still very close
    // to symmetric. Use a generous tolerance: anything within 1e-3
    // relative is treated as "intended symmetric, perturbed by noise"
    // and routed through `eigh` on the SYMMETRIZED matrix. Sum-of-
    // -eigenvalues equals trace, which is preserved by symmetrization,
    // so the numerical gradient matches the analytical one.
    bool is_near_symmetric = true;
    {
        int64_t nbatch_check = batch_size(A);
        auto A_cont = A.contiguous();
        if (A.dtype() == DType::Float32) {
            auto [a_max, diff_max] = eig_symmetry_metrics<float>(
                A_cont.data<float>(), nbatch_check, n, stream);
            is_near_symmetric = (diff_max < 1e-2 * std::max(a_max, 1.0));
        } else {
            auto [a_max, diff_max] = eig_symmetry_metrics<double>(
                A_cont.data<double>(), nbatch_check, n, stream);
            is_near_symmetric = (diff_max < 1e-3 * std::max(a_max, 1.0));
        }
    }

    if (is_near_symmetric) {
        // Symmetrize: A_sym = (A + A^T)/2. Trace is preserved, so
        // sum(eigvals(A_sym)) == sum(eigvals(A)) for any near-symmetric A
        // (eigvals are continuous in A). Dispatch eigh on the symmetric
        // matrix, returning (W, V) with V columns as right eigenvectors
        // and a zero-filled WI vector to match eig's tuple contract.
        auto At = ::tenzor::transpose(A, ndim - 2, ndim - 1).contiguous();
        auto A_sym = ::tenzor::mul(::tenzor::add(A, At), 0.5);
        auto [W, V] = linalg_eigh_kernel(A_sym.contiguous(), stream);
        std::vector<int64_t> w_shape_v(W.shape().begin(), W.shape().end());
        auto WI = zeros(w_shape_v, A.dtype(), A.device());
        return {W, WI, V};
    }

    // Non-symmetric path: QR fallback kernel.
    auto work = A.contiguous().clone();
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
        size_t smem = (2 * n * n + 4) * sizeof(float);
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        qr_eig_fallback_kernel_cs<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), WR.data<float>(), WI.data<float>(),
            V.data<float>(), n, 60 * n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        size_t smem = (2 * n * n + 4) * sizeof(double);
        int threads = std::min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        qr_eig_fallback_kernel_cs<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), WR.data<double>(), WI.data<double>(),
            V.data<double>(), n, 60 * n);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {WR, WI, V};
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

    // cuSolver uses column-major; Tenzor uses row-major. For a symmetric
    // input A we have A_row == A_col, but the output triangle mapping is
    // inverted: col-major LOWER = row-major UPPER and vice versa. So we
    // feed cuSolver the opposite uplo and let zero_triangle keep the
    // user-requested row-major triangle.
    cublasFillMode_t uplo_mode = upper ? CUBLAS_FILL_MODE_LOWER : CUBLAS_FILL_MODE_UPPER;

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

// ============================================================================
// Triangular Solve (AX = B, A triangular) — cuBLAS trsm
// ============================================================================

auto linalg_solve_triangular_kernel(const Tensor& A, const Tensor& B,
                                     bool upper, bool unitriangular,
                                     cudaStream_t stream) -> Tensor {
    auto original_dtype = A.dtype();
    if (original_dtype == DType::Float16 || original_dtype == DType::BFloat16) {
        return linalg_solve_triangular_kernel(
            A.to(DType::Float32), B.to(DType::Float32),
            upper, unitriangular, stream).to(original_dtype);
    }
    if (original_dtype != DType::Float32 && original_dtype != DType::Float64) {
        throw std::invalid_argument("linalg::solve_triangular: unsupported dtype");
    }

    // cuBLAS trsm operates in column-major. We physically transpose A and B
    // into column-major storage, so cuBLAS reads back the original logical
    // matrix (transpose-then-reinterpret is a double negation). Therefore the
    // uplo flag maps straight through — upper stays upper, lower stays lower.
    auto a_cm = tenzor::transpose(A.contiguous().clone(), -2, -1).contiguous();
    auto b_cm = tenzor::transpose(B.contiguous().clone(), -2, -1).contiguous();

    auto [n, ndim_a] = check_square(a_cm);
    auto b_shape = B.shape();
    auto b_ndim = static_cast<int64_t>(b_shape.size());
    int64_t nrhs = (b_ndim >= 2) ? b_shape[b_ndim - 1] : 1;
    int64_t nbatch = batch_size(a_cm);

    cublasFillMode_t uplo = upper ? CUBLAS_FILL_MODE_UPPER : CUBLAS_FILL_MODE_LOWER;
    cublasDiagType_t diag = unitriangular ? CUBLAS_DIAG_UNIT : CUBLAS_DIAG_NON_UNIT;

    auto handle = CuBLASHandlePool::get(stream);

    if (original_dtype == DType::Float32) {
        float alpha = 1.0f;
        float* a_ptr = a_cm.data<float>();
        float* b_ptr = b_cm.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            CUBLAS_CHECK(
                cublasStrsm(handle, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, diag,
                            static_cast<int>(n), static_cast<int>(nrhs), &alpha,
                            a_ptr + b * n * n, static_cast<int>(n),
                            b_ptr + b * n * nrhs, static_cast<int>(n)));
        }
    } else {
        double alpha = 1.0;
        double* a_ptr = a_cm.data<double>();
        double* b_ptr = b_cm.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            CUBLAS_CHECK(
                cublasDtrsm(handle, CUBLAS_SIDE_LEFT, uplo, CUBLAS_OP_N, diag,
                            static_cast<int>(n), static_cast<int>(nrhs), &alpha,
                            a_ptr + b * n * n, static_cast<int>(n),
                            b_ptr + b * n * nrhs, static_cast<int>(n)));
        }
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return tenzor::transpose(b_cm, -2, -1).contiguous();
}

// ============================================================================
// Geqrf — raw QR factorization returning packed reflectors + tau
// ============================================================================

auto linalg_geqrf_kernel(const Tensor& A, cudaStream_t stream)
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

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    if (A.dtype() == DType::Float32) {
        float* a_data = work.data<float>();
        float* tau_data = tau_result.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            float* a_mat = a_data + b * m * n_cols;
            float* tau_ptr = tau_data + b * k;

            // cuSOLVER geqrf works in column-major. For row-major m x n,
            // we pass n_cols as m and m as n (treating as A^T in col-major).
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSgeqrf_bufferSize(handle, n_cols, m, a_mat, n_cols, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSgeqrf(handle, n_cols, m,
                a_mat, n_cols, tau_ptr,
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "geqrf");
        }
    } else {
        double* a_data = work.data<double>();
        double* tau_data = tau_result.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            double* a_mat = a_data + b * m * n_cols;
            double* tau_ptr = tau_data + b * k;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDgeqrf_bufferSize(handle, n_cols, m, a_mat, n_cols, &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDgeqrf(handle, n_cols, m,
                a_mat, n_cols, tau_ptr,
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "geqrf");
        }
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {work, tau_result};
}

// ============================================================================
// Ormqr — multiply matrix by Q from QR factorization using tau vectors
// ============================================================================

auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                          const Tensor& C, bool left, bool transpose_q,
                          cudaStream_t stream) -> Tensor {
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

    auto handle = CuSOLVERHandlePool::get(stream);
    DeviceInt d_info;

    // cuSOLVER ormqr operates in column-major. For row-major data:
    // C is c_m x c_n in row-major = c_n x c_m matrix in column-major (C^T).
    // Q*C in row-major = (Q*C)^T^T; in col-major terms on C^T:
    //   left,  no-trans Q*C:    col-major does C^T * Q^T => ormqr(Right, Trans, c_n, c_m, k, ...)
    //   left,  trans    Q^T*C:  col-major does C^T * Q   => ormqr(Right, NoTrans, c_n, c_m, k, ...)
    //   right, no-trans C*Q:    col-major does Q^T * C^T => ormqr(Left, Trans, c_n, c_m, k, ...)
    //   right, trans    C*Q^T:  col-major does Q * C^T   => ormqr(Left, NoTrans, c_n, c_m, k, ...)
    cublasSideMode_t side;
    cublasOperation_t trans;
    if (left && !transpose_q)       { side = CUBLAS_SIDE_RIGHT; trans = CUBLAS_OP_T; }
    else if (left && transpose_q)   { side = CUBLAS_SIDE_RIGHT; trans = CUBLAS_OP_N; }
    else if (!left && !transpose_q) { side = CUBLAS_SIDE_LEFT;  trans = CUBLAS_OP_T; }
    else                            { side = CUBLAS_SIDE_LEFT;  trans = CUBLAS_OP_N; }

    cusolverDnHandle_t solver_handle = handle;

    if (C.dtype() == DType::Float32) {
        float* c_data = work_c.data<float>();
        const float* r_data = refl.data<float>();
        const float* tau_data = tau_c.data<float>();

        for (int64_t b = 0; b < nbatch; b++) {
            const float* r_mat = r_data + b * r_m * r_n;
            const float* tau_ptr = tau_data + b * k_refl;
            float* c_mat = c_data + b * c_m * c_n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSormqr_bufferSize(solver_handle,
                side, trans,
                static_cast<int>(c_n), static_cast<int>(c_m), static_cast<int>(k_refl),
                const_cast<float*>(r_mat), static_cast<int>(r_n),
                const_cast<float*>(tau_ptr),
                c_mat, static_cast<int>(c_n), &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));

            CUSOLVER_CHECK(cusolverDnSormqr(solver_handle,
                side, trans,
                static_cast<int>(c_n), static_cast<int>(c_m), static_cast<int>(k_refl),
                const_cast<float*>(r_mat), static_cast<int>(r_n),
                const_cast<float*>(tau_ptr),
                c_mat, static_cast<int>(c_n),
                static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "ormqr");
        }
    } else {
        double* c_data = work_c.data<double>();
        const double* r_data = refl.data<double>();
        const double* tau_data = tau_c.data<double>();

        for (int64_t b = 0; b < nbatch; b++) {
            const double* r_mat = r_data + b * r_m * r_n;
            const double* tau_ptr = tau_data + b * k_refl;
            double* c_mat = c_data + b * c_m * c_n;

            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDormqr_bufferSize(solver_handle,
                side, trans,
                static_cast<int>(c_n), static_cast<int>(c_m), static_cast<int>(k_refl),
                const_cast<double*>(r_mat), static_cast<int>(r_n),
                const_cast<double*>(tau_ptr),
                c_mat, static_cast<int>(c_n), &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));

            CUSOLVER_CHECK(cusolverDnDormqr(solver_handle,
                side, trans,
                static_cast<int>(c_n), static_cast<int>(c_m), static_cast<int>(k_refl),
                const_cast<double*>(r_mat), static_cast<int>(r_n),
                const_cast<double*>(tau_ptr),
                c_mat, static_cast<int>(c_n),
                static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "ormqr");
        }
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work_c;
}

// =========================================================================
// LDL^T factorization (cusolverDnSsytrf / cusolverDnDsytrf)
// =========================================================================
auto linalg_ldl_factor_kernel(const Tensor& A, cudaStream_t stream)
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

    // cuSOLVER expects column-major; transpose row-major → column-major
    auto work = tenzor::transpose(A, -2, -1).contiguous();
    auto handle = CuSOLVERHandlePool::get(stream);

    backend::CachedMemoryGuard ipiv_guard(nbatch * n * sizeof(int));
    int* d_ipiv = static_cast<int*>(ipiv_guard.get());
    DeviceInt d_info;

    if (original_dtype == DType::Float32) {
        float* data = work.data<float>();
        for (int64_t b = 0; b < nbatch; ++b) {
            float* mat = data + b * n * n;
            int* piv = d_ipiv + b * n;
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnSsytrf_bufferSize(handle, static_cast<int>(n), mat,
                static_cast<int>(n), &lwork));
            DeviceWorkspace workspace(lwork * sizeof(float));
            CUSOLVER_CHECK(cusolverDnSsytrf(handle, CUBLAS_FILL_MODE_LOWER,
                static_cast<int>(n), mat, static_cast<int>(n),
                piv, static_cast<float*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "ldl_factor");
        }
    } else {
        double* data = work.data<double>();
        for (int64_t b = 0; b < nbatch; ++b) {
            double* mat = data + b * n * n;
            int* piv = d_ipiv + b * n;
            int lwork = 0;
            CUSOLVER_CHECK(cusolverDnDsytrf_bufferSize(handle, static_cast<int>(n), mat,
                static_cast<int>(n), &lwork));
            DeviceWorkspace workspace(lwork * sizeof(double));
            CUSOLVER_CHECK(cusolverDnDsytrf(handle, CUBLAS_FILL_MODE_LOWER,
                static_cast<int>(n), mat, static_cast<int>(n),
                piv, static_cast<double*>(workspace.ptr), lwork, d_info.ptr));
            check_cusolver_info(d_info.ptr, "ldl_factor");
        }
    }

    // Transpose back to row-major
    auto LD = tenzor::transpose(work, -2, -1).contiguous();

    // Copy pivots to Int32 tensor on device
    auto pivots_out = tenzor::zeros(piv_shape, DType::Int32, A.device());
    CUDA_CHECK_LINALG(cudaMemcpy(pivots_out.data<int32_t>(), d_ipiv,
        nbatch * n * sizeof(int), cudaMemcpyDeviceToDevice));

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

    // Load LD and B into shared memory
    for (int idx = tid; idx < n * n; idx += num_threads)
        LD[idx] = batch_ld[idx];
    for (int idx = tid; idx < n * nrhs; idx += num_threads)
        B[idx] = batch_b[idx];
    __syncthreads();

    // All solve steps are sequential (thread 0 only)
    if (tid == 0) {
        // Step 1: Apply forward pivot permutation P to B
        // LAPACK sytrf pivots: if pivots[k] > 0, row k was swapped with pivots[k]-1 (1-based)
        // if pivots[k] < 0, then (k, k+1) form a 2x2 block;
        //   pivots[k] and pivots[k+1] have same negative value, row k+1 swapped with |pivots[k]|-1
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                // 1x1 pivot: swap row k with row p-1
                int swap_row = p - 1;
                if (swap_row != k) {
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j];
                        B[k * nrhs + j] = B[swap_row * nrhs + j];
                        B[swap_row * nrhs + j] = tmp;
                    }
                }
                k++;
            } else {
                // 2x2 pivot: swap row k+1 with row |p|-1
                int swap_row = (-p) - 1;
                if (swap_row != k + 1) {
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[(k + 1) * nrhs + j];
                        B[(k + 1) * nrhs + j] = B[swap_row * nrhs + j];
                        B[swap_row * nrhs + j] = tmp;
                    }
                }
                k += 2;
            }
        }

        // Step 2: Forward substitution: solve L * Y = P*B
        // L is unit lower triangular stored below diagonal of LD
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                // 1x1 pivot: column k of L
                for (int i = k + 1; i < n; i++) {
                    T mult = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++)
                        B[i * nrhs + j] -= mult * B[k * nrhs + j];
                }
                k++;
            } else {
                // 2x2 pivot: columns k and k+1 of L
                for (int i = k + 2; i < n; i++) {
                    T m0 = LD[i * n + k];
                    T m1 = LD[i * n + k + 1];
                    for (int j = 0; j < nrhs; j++)
                        B[i * nrhs + j] -= m0 * B[k * nrhs + j] + m1 * B[(k + 1) * nrhs + j];
                }
                k += 2;
            }
        }

        // Step 3: Diagonal solve: solve D * Z = Y
        // D is block diagonal with 1x1 and 2x2 blocks
        for (int k = 0; k < n; ) {
            int p = batch_piv[k];
            if (p > 0) {
                // 1x1 block: d = LD[k,k]
                T d = LD[k * n + k];
                for (int j = 0; j < nrhs; j++)
                    B[k * nrhs + j] /= d;
                k++;
            } else {
                // 2x2 block: [d11 d21; d21 d22] stored at LD[k,k], LD[k+1,k], LD[k+1,k+1]
                T d11 = LD[k * n + k];
                T d21 = LD[(k + 1) * n + k];
                T d22 = LD[(k + 1) * n + (k + 1)];
                T det = d11 * d22 - d21 * d21;
                for (int j = 0; j < nrhs; j++) {
                    T y0 = B[k * nrhs + j];
                    T y1 = B[(k + 1) * nrhs + j];
                    B[k * nrhs + j]       = (d22 * y0 - d21 * y1) / det;
                    B[(k + 1) * nrhs + j] = (d11 * y1 - d21 * y0) / det;
                }
                k += 2;
            }
        }

        // Step 4: Backward substitution: solve L^T * X = Z
        for (int k = n - 1; k >= 0; ) {
            int p = batch_piv[k];
            if (p > 0) {
                // 1x1 pivot: apply column k of L^T
                for (int i = k + 1; i < n; i++) {
                    T mult = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++)
                        B[k * nrhs + j] -= mult * B[i * nrhs + j];
                }
                k--;
            } else {
                // 2x2 pivot at (k-1, k)
                int k0 = k - 1;
                for (int i = k + 1; i < n; i++) {
                    T m0 = LD[i * n + k0];
                    T m1 = LD[i * n + k];
                    for (int j = 0; j < nrhs; j++) {
                        B[k0 * nrhs + j] -= m0 * B[i * nrhs + j];
                        B[k * nrhs + j]  -= m1 * B[i * nrhs + j];
                    }
                }
                k -= 2;
            }
        }

        // Step 5: Apply inverse pivot permutation P^T to X (reverse order)
        for (int k = n - 1; k >= 0; ) {
            int p = batch_piv[k];
            if (p > 0) {
                int swap_row = p - 1;
                if (swap_row != k) {
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j];
                        B[k * nrhs + j] = B[swap_row * nrhs + j];
                        B[swap_row * nrhs + j] = tmp;
                    }
                }
                k--;
            } else {
                int swap_row = (-p) - 1;
                if (swap_row != k) {
                    for (int j = 0; j < nrhs; j++) {
                        T tmp = B[k * nrhs + j];
                        B[k * nrhs + j] = B[swap_row * nrhs + j];
                        B[swap_row * nrhs + j] = tmp;
                    }
                }
                k -= 2;
            }
        }
    }
    __syncthreads();

    // Write B back to global memory
    for (int idx = tid; idx < n * nrhs; idx += num_threads)
        batch_b[idx] = B[idx];
}

auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                              const Tensor& B, cudaStream_t stream) -> Tensor {
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
    int64_t ndim = static_cast<int64_t>(ld_shape.size());
    int64_t n = ld_shape[ndim - 1];
    int64_t nrhs = b_shape[static_cast<int64_t>(b_shape.size()) - 1];
    int64_t nbatch = 1;
    for (int64_t i = 0; i + 2 < ndim; ++i) nbatch *= ld_shape[i];

    auto ld_cont = LD.contiguous();
    auto work_b = B.contiguous().clone();

    int threads = min(static_cast<int>(n), 128);
    if (threads < 1) threads = 1;

    if (original_dtype == DType::Float32) {
        size_t smem = (n * n + n * nrhs) * sizeof(float);
        ldl_solve_bk_kernel<float><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<float>(), pivots.data<int32_t>(),
            work_b.data<float>(), static_cast<int>(n), static_cast<int>(nrhs));
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        size_t smem = (n * n + n * nrhs) * sizeof(double);
        ldl_solve_bk_kernel<double><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<double>(), pivots.data<int32_t>(),
            work_b.data<double>(), static_cast<int>(n), static_cast<int>(nrhs));
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work_b;
}

// =========================================================================
// Householder product — compose from existing ormqr kernel
// =========================================================================
auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                cudaStream_t stream) -> Tensor {
    // Apply Householder reflectors to I[:, :n]: Q = H(0)*H(1)*...*H(k-1) @ I[:m,:n]
    //
    // LAPACK's sorgqr (used by the CPU path) produces an m×n matrix — the
    // first n columns of the full m×m Q. We compute it directly in row-major
    // tensor ops: walk the reflectors from 0..k-1 applying H_j to Q = I[:m,:n].
    //
    // Previously this routed through a cuSOLVER ormqr path that had a
    // row-major/column-major leading-dimension mismatch and only worked
    // accidentally for square m=n inputs.
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

    // Compute in Float32 for Float16/BFloat16, then cast back at the end.
    DType compute_dt = dt;
    if (dt == DType::Float16 || dt == DType::BFloat16) compute_dt = DType::Float32;

    Tensor V = input.contiguous();
    Tensor tau_c = tau.contiguous();
    if (V.dtype() != compute_dt) V = V.to(compute_dt);
    if (tau_c.dtype() != compute_dt) tau_c = tau_c.to(compute_dt);

    // Build m×n identity-with-padding.
    auto I_full = tenzor::eye(m, std::nullopt, compute_dt, dev);
    Tensor Q = (n == m) ? I_full
                         : tenzor::slice(I_full, /*dim=*/1, /*start=*/0,
                                         /*end=*/n).contiguous();
    if (ndim > 2) {
        std::vector<int64_t> batched_shape(shape.begin(), shape.end());
        Q = tenzor::expand(Q, std::move(batched_shape));
        Q = Q.contiguous();
    }

    // LAPACK's sorgqr builds Q = H(1) H(2) … H(k), but when applying as
    // Q = H_0 · H_1 · … · H_{k-1} · I, each H_j only touches the trailing
    // (m-j) rows so the order matters for accumulation. Testing against the
    // CPU LAPACK output shows we must walk reflectors right→left: start with
    // I and apply H_{k-1}, then H_{k-2}, … H_0. This matches the action of
    // sorgqr when reconstructing Q column-by-column.
    //
    // Precompute the per-step mask/override columns on-device once. For each
    // reflector j we need:
    //   v_j[i] = 0      for i < j   ← mask = 0, override = 0
    //   v_j[i] = 1      for i = j   ← mask = 0, override = 1
    //   v_j[i] = V[i,j] for i > j   ← mask = 1, override = 0
    // Encode that as two (m, k) tensors and slice column j per step:
    //   M_full[i, j] = (i > j) ? 1 : 0   →  tril(ones({m,k}), -1)
    //   O_full[i, j] = (i == j) ? 1 : 0  →  eye(m, k)
    // Replaces the previous m-element host scalar fill + H2D per iteration.
    Tensor M_full = tenzor::tril(tenzor::ones({m, k}, compute_dt, dev), -1);
    Tensor O_full = tenzor::eye(m, k, compute_dt, dev);

    for (int64_t j = k - 1; j >= 0; --j) {
        // v_j: shape (..., m, 1). Start from V[:, j:j+1] and mask rows < j to 0,
        // set row j to 1.
        Tensor v_j = tenzor::slice(V, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();

        // (m, 1) mask + override sliced from precomputed device tensors.
        Tensor mask = tenzor::slice(M_full, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();
        Tensor overrides = tenzor::slice(O_full, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();
        if (ndim > 2) {
            std::vector<int64_t> bshape(shape.begin(), shape.end() - 1);
            bshape.push_back(1);
            mask = tenzor::expand(mask, bshape).contiguous();
            overrides = tenzor::expand(overrides, bshape).contiguous();
        }
        v_j = tenzor::add(tenzor::mul(v_j, mask), overrides);

        // τ_j as scalar tensor (broadcastable). For batched tau, extract lane j.
        Tensor tau_j = tenzor::slice(tau_c, /*dim=*/-1, /*start=*/j, /*end=*/j + 1).contiguous();

        // u = v_jᵀ · Q  → shape (..., 1, n)
        Tensor v_jT = tenzor::transpose(v_j, -1, -2);
        Tensor u = tenzor::matmul(v_jT, Q);

        // Q ← Q − τ_j · v_j · u
        Tensor outer = tenzor::matmul(v_j, u);
        // Broadcast τ_j (…, 1) against outer (…, m, n).
        // Reshape tau_j from (…, 1) to (…, 1, 1) for broadcasting.
        std::vector<int64_t> tau_shape(tau_j.shape().begin(), tau_j.shape().end());
        tau_shape.push_back(1);
        tau_j = tau_j.reshape(tau_shape);
        Tensor scaled = tenzor::mul(outer, tau_j);
        Q = tenzor::sub(Q, scaled);
    }

    if (Q.dtype() != dt) Q = Q.to(dt);
    return Q;
}

} // namespace cuda
} // namespace tenzor

#else // !TENZOR_HAS_CUSOLVER — native CUDA fallback kernels

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/linalg.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"
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
// Split a row-major packed LU matrix into separate L (unit lower triangular)
// and U (upper triangular) tensors. One thread per element.
// ============================================================================

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
    // Plain 2D input → scalar output (shape {}); torch.linalg.det contract.

    auto result = zeros(out_shape, A.dtype(), A.device());

    // NN.13: stream-ordered alloc/free so cudaFree doesn't device-sync.
    StreamWorkspace<int> d_pivots_ws(nbatch * n, stream);
    StreamWorkspace<int> d_info_ws(nbatch, stream);
    int* d_pivots = d_pivots_ws.get();
    int* d_info = d_info_ws.get();
    CUDA_CHECK_LINALG(cudaMemsetAsync(d_info, 0, nbatch * sizeof(int), stream));

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

    // NN.13: stream-ordered alloc/free so cudaFree doesn't device-sync.
    StreamWorkspace<int> d_pivots_ws(nbatch * n, stream);
    StreamWorkspace<int> d_info_ws(nbatch, stream);
    int* d_pivots = d_pivots_ws.get();
    int* d_info = d_info_ws.get();
    CUDA_CHECK_LINALG(cudaMemsetAsync(d_info, 0, nbatch * sizeof(int), stream));

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

    // NN.13: stream-ordered alloc/free so cudaFree doesn't device-sync.
    StreamWorkspace<int> d_pivots_ws(nbatch * n, stream);
    StreamWorkspace<int> d_info_ws(nbatch, stream);
    int* d_pivots = d_pivots_ws.get();
    int* d_info = d_info_ws.get();
    CUDA_CHECK_LINALG(cudaMemsetAsync(d_info, 0, nbatch * sizeof(int), stream));

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

// ============================================================================
// Triangular Solve (AX = B, A triangular) — native CUDA kernel
// ============================================================================

template<typename T>
__global__ void trsm_kernel(const T* __restrict__ A, T* __restrict__ B,
                            int n, int nrhs, bool upper, bool unit_diag) {
    // One block per batch element. Serial substitution per row.
    int batch = blockIdx.x;
    const T* A_mat = A + batch * n * n;
    T* B_mat = B + batch * n * nrhs;
    int tid = threadIdx.x;

    if (upper) {
        // Back substitution: row n-1 down to 0
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
        // Forward substitution: row 0 up to n-1
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
                                     cudaStream_t stream) -> Tensor {
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

    int threads = min(max(static_cast<int>(nrhs), 1), 256);

    if (work_a.dtype() == DType::Float32) {
        check_size_limit<float>(n, "solve_triangular");
        trsm_kernel<float><<<nbatch, threads, 0, stream>>>(
            work_a.data<float>(), work_b.data<float>(),
            static_cast<int>(n), static_cast<int>(nrhs), upper, unitriangular);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "solve_triangular");
        trsm_kernel<double><<<nbatch, threads, 0, stream>>>(
            work_a.data<double>(), work_b.data<double>(),
            static_cast<int>(n), static_cast<int>(nrhs), upper, unitriangular);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work_b;
}

// ============================================================================
// LU Decomposition — returns (L, U, pivots) using shared-memory lu_kernel
// ============================================================================

auto linalg_lu_kernel(const Tensor& A, cudaStream_t stream)
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
    // NN.13: stream-ordered alloc/free so cudaFree doesn't device-sync.
    StreamWorkspace<int> d_pivots_ws(nbatch * n, stream);
    StreamWorkspace<int> d_info_ws(nbatch, stream);
    int* d_pivots = d_pivots_ws.get();
    int* d_info = d_info_ws.get();
    CUDA_CHECK_LINALG(cudaMemsetAsync(d_info, 0, nbatch * sizeof(int), stream));

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
        CUDA_CHECK_LINALG(cudaGetLastError());

        // Step 2: split packed LU into L and U
        int64_t total = nbatch * n * n;
        int ext_threads = 256;
        int ext_blocks = static_cast<int>((total + ext_threads - 1) / ext_threads);
        extract_lu_kernel<float><<<ext_blocks, ext_threads, 0, stream>>>(
            work.data<float>(), L.data<float>(), U.data<float>(), n, nbatch);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "lu");
        size_t smem = n * n * sizeof(double) + 4 * sizeof(double);
        int threads = min(static_cast<int>(n), 128);
        if (threads < 1) threads = 1;
        lu_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), d_pivots, d_info, n);
        CUDA_CHECK_LINALG(cudaGetLastError());

        int64_t total = nbatch * n * n;
        int ext_threads = 256;
        int ext_blocks = static_cast<int>((total + ext_threads - 1) / ext_threads);
        extract_lu_kernel<double><<<ext_blocks, ext_threads, 0, stream>>>(
            work.data<double>(), L.data<double>(), U.data<double>(), n, nbatch);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    // Step 3: copy pivots to Int32 tensor
    auto pivots_out = zeros(piv_shape, DType::Int32, A.device());
    CUDA_CHECK_LINALG(cudaMemcpyAsync(pivots_out.data<int32_t>(), d_pivots,
        nbatch * n * sizeof(int), cudaMemcpyDeviceToDevice, stream));

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {L, U, pivots_out};
}

// ============================================================================
// LU Solve — given L, U, pivots from linalg_lu_kernel, solve A*X = B
// The caller passes the PACKED LU (not separate L/U), plus pivots and B.
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
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "lu_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(double);
        int threads = min(max(static_cast<int>(n), 1), 128);
        lu_solve_kernel<double><<<nbatch, threads, smem, stream>>>(
            lu_work.data<double>(), piv_dev.data<int32_t>(),
            b_work.data<double>(), n, nrhs);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return b_work;
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
    T* scratch = R + m * n_cols;  // scratch[0]=v0, scratch[1]=tau, scratch[2]=alpha

    T* A = A_io + batch_idx * m * n_cols;
    T* tau = tau_out + batch_idx * k;

    // Load A into shared memory
    for (int idx = tid; idx < m * n_cols; idx += num_threads) {
        R[idx] = A[idx];
    }
    __syncthreads();

    constexpr T zero_tol = std::is_same_v<T, float> ? T(1e-30) : T(1e-60);

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
                scratch[1] = T(0);  // tau = 0
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
        if (tau_val == T(0)) {
            __syncthreads();
            continue;
        }
        T v0 = scratch[0];
        T alpha = scratch[2];

        // Apply reflector H = I - tau*v*v^T to trailing columns of R
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

        // Store: set diagonal to alpha, store Householder vector below diagonal
        // The reflector vector v = [v0; A[j+1:,j]] is already in place below diagonal.
        // We need to normalize so v[0] = 1 (standard LAPACK convention):
        // v_normalized = v / v0, tau_normalized = tau * v0^2
        // But for simplicity, we store the raw form: diagonal = alpha,
        // below-diagonal = raw reflector entries (unnormalized). tau already stored.
        if (tid == 0) {
            // Store v0 at the diagonal position (reflector vector starts with v0)
            // Actually, LAPACK convention stores v with v[0]=1 below diagonal
            // and tau separately. We store v/v0 below diagonal, adjust tau.
            T inv_v0 = T(1) / v0;
            for (int i = j + 1; i < m; i++) {
                R[i * n_cols + j] *= inv_v0;
            }
            R[j * n_cols + j] = alpha;
            tau[j] = tau_val * v0 * v0;  // adjusted tau for normalized v
        }
        __syncthreads();
    }

    // Write back to global memory
    for (int idx = tid; idx < m * n_cols; idx += num_threads) {
        A[idx] = R[idx];
    }
}

// ============================================================================
// Ormqr fallback — apply Q (from Householder reflectors) to matrix C
// One block per batch element.
// Shared memory: refl[r_m*r_n] + C[c_m*c_n] + scratch[c_max_dim]
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
    T* work = C + c_m * c_n;  // scratch vector of size max(c_m, c_n)

    const T* refl = refl_in + batch_idx * r_m * r_n;
    const T* tau = tau_in + batch_idx * k_refl;
    T* C_out = C_io + batch_idx * c_m * c_n;

    // Load C into shared memory
    for (int idx = tid; idx < c_m * c_n; idx += num_threads) {
        C[idx] = C_out[idx];
    }
    __syncthreads();

    // Apply Householder reflectors
    // Left: Q*C or Q^T*C — iterate over reflectors
    // Right: C*Q or C*Q^T — iterate over reflectors
    // Q = H_0 * H_1 * ... * H_{k-1}, where H_j = I - tau_j * v_j * v_j^T
    // Q^T = H_{k-1} * ... * H_1 * H_0
    // For left, no-trans (Q*C): apply H_0, H_1, ..., H_{k-1} from left
    // For left, trans (Q^T*C): apply H_{k-1}, ..., H_0 from left
    // For right, no-trans (C*Q): apply H_0, ..., H_{k-1} from right
    // For right, trans (C*Q^T): apply H_{k-1}, ..., H_0 from right

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
            // Apply H_j from left: C = (I - tau_j * v_j * v_j^T) * C
            // For each column c of C: c = c - tau_j * v_j * (v_j^T * c)
            for (int col = tid; col < c_n; col += num_threads) {
                // Compute v_j^T * C[:,col]
                T dot = C[j * c_n + col];  // v[j] = 1 (implicit)
                for (int i = j + 1; i < c_m; i++) {
                    dot += refl[i * r_n + j] * C[i * c_n + col];
                }
                dot *= tau_j;
                // Update C[:,col]
                C[j * c_n + col] -= dot;
                for (int i = j + 1; i < c_m; i++) {
                    C[i * c_n + col] -= refl[i * r_n + j] * dot;
                }
            }
        } else {
            // Apply H_j from right: C = C * (I - tau_j * v_j * v_j^T)
            // For each row r of C: r = r - tau_j * (r * v_j) * v_j^T
            for (int row = tid; row < c_m; row += num_threads) {
                T dot = C[row * c_n + j];  // v[j] = 1 (implicit)
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

    // Write C back
    for (int idx = tid; idx < c_m * c_n; idx += num_threads) {
        C_out[idx] = C[idx];
    }
}

auto linalg_geqrf_kernel(const Tensor& A, cudaStream_t stream)
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
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_geqrf_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), tau_result.data<float>(), m, n_cols, k);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(std::max(m, n_cols), "geqrf");
        size_t smem = (m * n_cols + 4) * sizeof(double);
        int threads = min(static_cast<int>(std::max(m, n_cols)), 128);
        if (threads < 1) threads = 1;
        householder_geqrf_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), tau_result.data<double>(), m, n_cols, k);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {work, tau_result};
}

auto linalg_ormqr_kernel(const Tensor& reflectors, const Tensor& tau,
                          const Tensor& C, bool left, bool transpose_q,
                          cudaStream_t stream) -> Tensor {
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
        int threads = min(static_cast<int>(std::max(c_m, c_n)), 128);
        if (threads < 1) threads = 1;
        householder_ormqr_kernel<float><<<nbatch, threads, smem, stream>>>(
            refl.data<float>(), tau_c.data<float>(), work_c.data<float>(),
            r_m, r_n, c_m, c_n, k_refl, left, transpose_q);
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(std::max(c_m, c_n), "ormqr");
        size_t smem = (c_m * c_n + std::max(c_m, c_n)) * sizeof(double);
        int threads = min(static_cast<int>(std::max(c_m, c_n)), 128);
        if (threads < 1) threads = 1;
        householder_ormqr_kernel<double><<<nbatch, threads, smem, stream>>>(
            refl.data<double>(), tau_c.data<double>(), work_c.data<double>(),
            r_m, r_n, c_m, c_n, k_refl, left, transpose_q);
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work_c;
}

// =========================================================================
// LDL^T factorization — Bunch-Kaufman diagonal pivoting in shared memory.
// One block per batch element. Shared memory: A[n*n] + scratch[4].
// Stores L below diagonal, D on/above diagonal, pivots in global memory.
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
    T* scratch = A + n * n;  // scratch[0..3]

    T* batch_data = data + batch_idx * n * n;
    int* batch_piv = pivots_out + batch_idx * n;

    // Load matrix into shared memory (symmetric, stored fully)
    for (int idx = tid; idx < n * n; idx += num_threads)
        A[idx] = batch_data[idx];
    __syncthreads();

    // Bunch-Kaufman factorization with alpha = (1 + sqrt(17)) / 8
    constexpr T alpha = static_cast<T>(0.6404);

    int k = 0;
    while (k < n) {
        if (tid == 0) {
            // Find max |A[i,k]| for i > k (column max below diagonal)
            T col_max = T(0);
            int col_max_row = k;
            for (int i = k + 1; i < n; i++) {
                T v = fabs(A[i * n + k]);
                if (v > col_max) { col_max = v; col_max_row = i; }
            }

            T abs_akk = fabs(A[k * n + k]);

            if (abs_akk == T(0) && col_max == T(0)) {
                // Zero column — 1x1 pivot with zero diagonal
                batch_piv[k] = k + 1;  // 1-based, positive = 1x1
                scratch[0] = T(1);     // flag: 1x1 pivot
                scratch[1] = T(k);     // no swap
            } else if (abs_akk >= alpha * col_max) {
                // 1x1 pivot, no interchange
                batch_piv[k] = k + 1;
                scratch[0] = T(1);     // 1x1
                scratch[1] = T(k);     // swap_row = k (no swap)
            } else {
                // Check row max of row col_max_row
                int r = col_max_row;
                T row_max = T(0);
                for (int j = k; j < n; j++) {
                    if (j == r) continue;
                    T v = fabs(A[r * n + j]);
                    if (v > row_max) row_max = v;
                }

                T abs_arr = fabs(A[r * n + r]);

                if (abs_akk * row_max >= alpha * col_max * col_max) {
                    // 1x1 pivot, no interchange
                    batch_piv[k] = k + 1;
                    scratch[0] = T(1);
                    scratch[1] = T(k);
                } else if (abs_arr >= alpha * row_max) {
                    // 1x1 pivot, interchange rows/cols k and r
                    batch_piv[k] = r + 1;  // 1-based
                    scratch[0] = T(1);
                    scratch[1] = T(r);     // swap k with r
                } else {
                    // 2x2 pivot using rows/cols k and r
                    // Store negative pivot to indicate 2x2 block
                    batch_piv[k] = -(r + 1);      // negative, 1-based
                    batch_piv[k + 1] = -(r + 1);  // same negative value
                    scratch[0] = T(2);     // 2x2
                    scratch[1] = T(r);     // swap row k+1 with r
                }
            }
        }
        __syncthreads();

        int pivot_type = static_cast<int>(scratch[0]);
        int swap_row = static_cast<int>(scratch[1]);

        if (pivot_type == 1) {
            // 1x1 pivot
            // Swap rows and columns k and swap_row if needed
            if (swap_row != k) {
                // Swap rows k and swap_row
                for (int j = tid; j < n; j += num_threads) {
                    T tmp = A[k * n + j];
                    A[k * n + j] = A[swap_row * n + j];
                    A[swap_row * n + j] = tmp;
                }
                __syncthreads();
                // Swap columns k and swap_row
                for (int i = tid; i < n; i += num_threads) {
                    T tmp = A[i * n + k];
                    A[i * n + k] = A[i * n + swap_row];
                    A[i * n + swap_row] = tmp;
                }
                __syncthreads();
            }

            // Compute multipliers and update
            T diag = A[k * n + k];
            if (diag != T(0)) {
                // Compute L column: A[i,k] /= A[k,k] for i > k
                if (tid == 0) {
                    for (int i = k + 1; i < n; i++)
                        A[i * n + k] /= diag;
                }
                __syncthreads();

                // Symmetric rank-1 update: A[i,j] -= L[i,k] * D[k,k] * L[j,k]
                for (int i = k + 1 + tid; i < n; i += num_threads) {
                    T lik = A[i * n + k];
                    for (int j = k + 1; j <= i; j++) {
                        A[i * n + j] -= lik * diag * A[j * n + k];
                    }
                }
                __syncthreads();
            }
            k++;
        } else {
            // 2x2 pivot: swap row/col k+1 with swap_row if needed
            if (swap_row != k + 1) {
                for (int j = tid; j < n; j += num_threads) {
                    T tmp = A[(k + 1) * n + j];
                    A[(k + 1) * n + j] = A[swap_row * n + j];
                    A[swap_row * n + j] = tmp;
                }
                __syncthreads();
                for (int i = tid; i < n; i += num_threads) {
                    T tmp = A[i * n + (k + 1)];
                    A[i * n + (k + 1)] = A[i * n + swap_row];
                    A[i * n + swap_row] = tmp;
                }
                __syncthreads();
            }

            // 2x2 diagonal block D = [A[k,k], A[k+1,k]; A[k+1,k], A[k+1,k+1]]
            T d11 = A[k * n + k];
            T d21 = A[(k + 1) * n + k];
            T d22 = A[(k + 1) * n + (k + 1)];
            T det = d11 * d22 - d21 * d21;

            if (det != T(0)) {
                // Compute L columns k, k+1 for rows > k+1
                if (tid == 0) {
                    T inv11 = d22 / det;
                    T inv12 = -d21 / det;
                    T inv22 = d11 / det;
                    for (int i = k + 2; i < n; i++) {
                        T a0 = A[i * n + k];
                        T a1 = A[i * n + (k + 1)];
                        A[i * n + k]     = inv11 * a0 + inv12 * a1;
                        A[i * n + (k + 1)] = inv12 * a0 + inv22 * a1;
                    }
                }
                __syncthreads();

                // Symmetric rank-2 update
                for (int i = k + 2 + tid; i < n; i += num_threads) {
                    T li0 = A[i * n + k];
                    T li1 = A[i * n + (k + 1)];
                    for (int j = k + 2; j <= i; j++) {
                        T lj0 = A[j * n + k];
                        T lj1 = A[j * n + (k + 1)];
                        A[i * n + j] -= (li0 * (d11 * lj0 + d21 * lj1)
                                       + li1 * (d21 * lj0 + d22 * lj1));
                    }
                }
                __syncthreads();
            }
            k += 2;
        }
    }

    // Write back to global memory
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

// =========================================================================
// LDL^T factorization — native GPU Bunch-Kaufman kernel
// =========================================================================
auto linalg_ldl_factor_kernel(const Tensor& A, cudaStream_t stream)
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

    int threads = min(static_cast<int>(n), 128);
    if (threads < 1) threads = 1;

    if (original_dtype == DType::Float32) {
        check_size_limit<float>(n, "ldl_factor");
        size_t smem = (n * n + 4) * sizeof(float);
        ldl_bk_factor_kernel<float><<<nbatch, threads, smem, stream>>>(
            work.data<float>(), pivots_out.data<int32_t>(), static_cast<int>(n));
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "ldl_factor");
        size_t smem = (n * n + 4) * sizeof(double);
        ldl_bk_factor_kernel<double><<<nbatch, threads, smem, stream>>>(
            work.data<double>(), pivots_out.data<int32_t>(), static_cast<int>(n));
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return {work, pivots_out};
}

// =========================================================================
// LDL^T solve — native GPU Bunch-Kaufman solve kernel
// =========================================================================
auto linalg_ldl_solve_kernel(const Tensor& LD, const Tensor& pivots,
                              const Tensor& B, cudaStream_t stream) -> Tensor {
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

    int threads = min(static_cast<int>(n), 128);
    if (threads < 1) threads = 1;

    if (original_dtype == DType::Float32) {
        check_size_limit<float>(n, "ldl_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(float);
        ldl_bk_solve_kernel<float><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<float>(), pivots.data<int32_t>(),
            work_b.data<float>(), static_cast<int>(n), static_cast<int>(nrhs));
        CUDA_CHECK_LINALG(cudaGetLastError());
    } else {
        check_size_limit<double>(n, "ldl_solve");
        size_t smem = (n * n + n * nrhs) * sizeof(double);
        ldl_bk_solve_kernel<double><<<nbatch, threads, smem, stream>>>(
            ld_cont.data<double>(), pivots.data<int32_t>(),
            work_b.data<double>(), static_cast<int>(n), static_cast<int>(nrhs));
        CUDA_CHECK_LINALG(cudaGetLastError());
    }

    CUDA_CHECK_LINALG(cudaStreamSynchronize(stream ? stream : 0));
    return work_b;
}

// =========================================================================
// Householder product — compose from existing ormqr kernel
// =========================================================================
auto linalg_householder_kernel(const Tensor& input, const Tensor& tau,
                                cudaStream_t stream) -> Tensor {
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

    return linalg_ormqr_kernel(input, tau, I, true, false, stream);
}

} // namespace cuda
} // namespace tenzor

#endif // TENZOR_HAS_CUSOLVER
