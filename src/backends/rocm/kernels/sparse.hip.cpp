/**
 * @file sparse.hip.cpp
 * @brief ROCm kernels for sparse tensor operations using rocSPARSE.
 *
 * Provides GPU-accelerated implementations of:
 * - spmm (sparse-dense matrix multiplication) via rocsparse_spmm()
 * - spmv (sparse-dense matrix-vector multiplication) via rocsparse_spmv()
 *
 * Uses CSR format descriptors for rocSPARSE API compatibility.
 * Both COO and CSR inputs are supported; COO is converted to CSR internally.
 */

#ifdef TENZOR_HAS_ROCSPARSE

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/math.hpp"

#include "tenzor/backend/loader_fwd.hpp"

#include <rocsparse/rocsparse.h>
#include <hip/hip_runtime.h>
#include "../hip_buffer.hpp"
#include "../rocsparse_handle_pool.hpp"
#include <array>
#include <climits>
#include <cstdint>
#include <span>
#include <functional>
#include <limits>
#include <list>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <thrust/device_ptr.h>
#include <thrust/sort.h>
#include <thrust/sequence.h>
#include <thrust/gather.h>
#include <thrust/transform.h>
#include <thrust/reduce.h>
#include <thrust/execution_policy.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/binary_search.h>

// Forward-declare zeros to avoid including creation.hpp
namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace rocm {

namespace {

#define HIP_CHECK_SPARSE(call)                                                  \
    do {                                                                         \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("HIP error in sparse at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));     \
        }                                                                       \
    } while (0)

// ROCSPARSE_CHECK is provided by rocsparse_handle_pool.hpp

/// Convert span to vector (HIP compiler may not do implicit span->vector).
std::vector<int64_t> to_vec(std::span<const int64_t> s) {
    return {s.begin(), s.end()};
}

// HipBuffer is provided by ../hip_buffer.hpp (in namespace tenzor::rocm)

/// RAII guard for rocSPARSE sparse matrix descriptor.
struct SpMatGuard {
    rocsparse_spmat_descr desc = nullptr;
    explicit SpMatGuard(rocsparse_spmat_descr d) : desc(d) {}
    ~SpMatGuard() { if (desc) rocsparse_destroy_spmat_descr(desc); }
    SpMatGuard(const SpMatGuard&) = delete;
    SpMatGuard& operator=(const SpMatGuard&) = delete;
};

/// RAII guard for rocSPARSE dense matrix descriptor.
struct DnMatGuard {
    rocsparse_dnmat_descr desc = nullptr;
    explicit DnMatGuard(rocsparse_dnmat_descr d) : desc(d) {}
    ~DnMatGuard() { if (desc) rocsparse_destroy_dnmat_descr(desc); }
    DnMatGuard(const DnMatGuard&) = delete;
    DnMatGuard& operator=(const DnMatGuard&) = delete;
};

/// RAII guard for rocSPARSE dense vector descriptor.
struct DnVecGuard {
    rocsparse_dnvec_descr desc = nullptr;
    explicit DnVecGuard(rocsparse_dnvec_descr d) : desc(d) {}
    ~DnVecGuard() { if (desc) rocsparse_destroy_dnvec_descr(desc); }
    DnVecGuard(const DnVecGuard&) = delete;
    DnVecGuard& operator=(const DnVecGuard&) = delete;
};

/// RAII guard for rocSPARSE SpMV descriptor (v2 API).
struct SpMVDescrGuard {
    rocsparse_spmv_descr desc = nullptr;
    explicit SpMVDescrGuard(rocsparse_spmv_descr d) : desc(d) {}
    ~SpMVDescrGuard() { if (desc) rocsparse_destroy_spmv_descr(desc); }
    SpMVDescrGuard(const SpMVDescrGuard&) = delete;
    SpMVDescrGuard& operator=(const SpMVDescrGuard&) = delete;
};

/// Per-thread rocSPARSE handle — forwards to the shared pool.
inline rocsparse_handle get_rocsparse_handle() {
    return RocSPARSEHandlePool::get();
}

/// Get rocSPARSE data type from DType.
rocsparse_datatype get_rocsparse_data_type(DType dtype) {
    switch (dtype) {
        case DType::Float32: return rocsparse_datatype_f32_r;
        case DType::Float64: return rocsparse_datatype_f64_r;
        default:
            throw std::runtime_error("rocm_sparse: unsupported dtype " +
                                     std::string(dtype_name(dtype)));
    }
}

/// HIP kernel: convert Int64 row indices to Int32.
__global__ void cast_i64_to_i32(const int64_t* __restrict__ src,
                                 int32_t* __restrict__ dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int32_t>(src[i]);
}

/// HIP kernel: convert Int32 crow_indices to Int64.
__global__ void cast_i32_to_i64(const int32_t* __restrict__ src,
                                 int64_t* __restrict__ dst, int64_t n) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) dst[i] = static_cast<int64_t>(src[i]);
}

/// HIP kernel: check if any int64 value overflows int32 range.
__global__ void check_i64_overflow(const int64_t* __restrict__ src,
                                    int64_t n, int* overflow_flag) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (src[i] > INT32_MAX || src[i] < INT32_MIN) {
            atomicOr(overflow_flag, 1);
        }
    }
}

/// Device kernel: flag any column index outside [0, ncols).
__global__ void check_col_range(const int64_t* __restrict__ src, int64_t n,
                                int64_t ncols, int* range_flag) {
    int64_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        if (src[i] < 0 || src[i] >= ncols) {
            atomicOr(range_flag, 1);
        }
    }
}

/// Validate that all column indices lie in [0, ncols). Throws on failure.
static void verify_col_in_range(const int64_t* d_data, int64_t n, int64_t ncols,
                                hipStream_t stream = nullptr) {
    if (n == 0) return;
    HipBuffer flag_buf(sizeof(int));
    int* d_flag = flag_buf.as<int>();
    HIP_CHECK_SPARSE(hipMemsetAsync(d_flag, 0, sizeof(int), stream));

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(check_col_range, dim3(blocks), dim3(threads),
                       0, stream, d_data, n, ncols, d_flag);
    HIP_CHECK_SPARSE(hipGetLastError());

    int h_flag = 0;
    HIP_CHECK_SPARSE(hipMemcpyAsync(&h_flag, d_flag, sizeof(int),
                                     hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE(hipStreamSynchronize(stream));

    if (h_flag != 0) {
        throw std::out_of_range(
            "rocm_sparse: COO column index out of range [0, ncols)");
    }
}

/// Check that all int64 indices fit in int32. Throws std::overflow_error on failure.
static void verify_i64_fits_i32(const int64_t* d_data, int64_t n,
                                 hipStream_t stream = nullptr) {
    if (n == 0) return;
    HipBuffer flag_buf(sizeof(int));
    int* d_flag = flag_buf.as<int>();
    HIP_CHECK_SPARSE(hipMemsetAsync(d_flag, 0, sizeof(int), stream));

    int threads = 256;
    int blocks = static_cast<int>((n + threads - 1) / threads);
    hipLaunchKernelGGL(check_i64_overflow, dim3(blocks), dim3(threads),
                       0, stream, d_data, n, d_flag);
    HIP_CHECK_SPARSE(hipGetLastError());

    int h_flag = 0;
    HIP_CHECK_SPARSE(hipMemcpyAsync(&h_flag, d_flag, sizeof(int),
                                     hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE(hipStreamSynchronize(stream));

    if (h_flag != 0) {
        throw std::overflow_error(
            "rocm_sparse: int64 index value exceeds int32 range");
    }
}

/// Helper to build a CSR SparseTensor on GPU from a COO SparseTensor.
SparseTensor ensure_csr_on_gpu(const SparseTensor& sparse) {
    auto sp = (sparse.device().type != Device::Type::ROCm)
              ? sparse.to(Device::rocm())
              : sparse;

    if (sp.layout() == SparseLayout::COO) {
        auto sp_shape = sp.shape();
        int64_t nrows = sp_shape[0];
        int64_t ncols = sp_shape[1];
        int64_t nnz = sp.nnz();

        Tensor indices = sp.indices().contiguous();
        Tensor values = sp.values().contiguous();

        const int64_t* indices_ptr = indices.data<int64_t>();
        const int64_t* row_indices_ptr = indices_ptr;
        const int64_t* col_indices_ptr = indices_ptr + nnz;

        // rocSPARSE coo2csr requires Int32
        HipBuffer row_i32_buf(nnz * sizeof(int32_t));
        HipBuffer crow_i32_buf((nrows + 1) * sizeof(int32_t));

        int threads = 256;
        int blocks_nnz = static_cast<int>((nnz + threads - 1) / threads);
        verify_i64_fits_i32(row_indices_ptr, nnz);
        cast_i64_to_i32<<<blocks_nnz, threads>>>(row_indices_ptr, row_i32_buf.as<int32_t>(), nnz);
        HIP_CHECK_SPARSE(hipGetLastError());

        // Convert COO row indices to CSR row pointers on GPU
        if (nnz > static_cast<int64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("rocm_sparse: nnz exceeds int32 range for rocsparse_coo2csr");
        if (nrows > static_cast<int64_t>(std::numeric_limits<int>::max()))
            throw std::overflow_error("rocm_sparse: nrows exceeds int32 range for rocsparse_coo2csr");
        rocsparse_handle handle = get_rocsparse_handle();
        ROCSPARSE_CHECK(rocsparse_coo2csr(
            handle, row_i32_buf.as<int32_t>(), static_cast<int>(nnz), static_cast<int>(nrows),
            crow_i32_buf.as<int32_t>(), rocsparse_index_base_zero));

        // Convert Int32 crow_indices back to Int64 on GPU
        Tensor crow_indices = zeros(std::vector<int64_t>{nrows + 1}, DType::Int64, Device::rocm());
        int blocks_crow = static_cast<int>((nrows + 1 + threads - 1) / threads);
        cast_i32_to_i64<<<blocks_crow, threads>>>(crow_i32_buf.as<int32_t>(), crow_indices.data<int64_t>(), nrows + 1);
        HIP_CHECK_SPARSE(hipGetLastError());

        // Validate column indices before handing them to rocSPARSE; an
        // out-of-range column would index the dense operand out of bounds.
        verify_col_in_range(col_indices_ptr, nnz, ncols);

        // Copy col indices
        Tensor col_idx = zeros(std::vector<int64_t>{nnz}, DType::Int64, Device::rocm());
        HIP_CHECK_SPARSE(hipMemcpy(col_idx.data<int64_t>(), col_indices_ptr,
                                   nnz * sizeof(int64_t), hipMemcpyDeviceToDevice));

        return SparseTensor::sparse_csr(
            crow_indices, col_idx, values,
            std::vector<int64_t>{nrows, ncols});
    }
    return sp;
}

// CSR SparseAdd kernel: one thread per row, adds sparse values into dense output
template <typename T>
__global__ void csr_sparse_add_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    T* __restrict__ out_ptr,
    int64_t nrows, int64_t ncols)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        // Guard against malformed/untrusted CSR col indices: an out-of-range
        // column would be an out-of-bounds device write into out_ptr.
        int64_t c = col_ptr[j];
        if (c < 0 || c >= ncols) continue;
        out_ptr[row * ncols + c] += val_ptr[j];
    }
}

// ============================================================================
// SpSV LRU cache: amortises rocsparse_spsv preprocess across calls.
//
// Each cache entry pins the i32-converted CSR buffers, the rocSPARSE matrix
// descriptor (already analysed) and the preprocess workspace. The trsm loop
// only needs to run rocsparse_spsv_stage_compute per right-hand-side once
// the descriptor is in the cache; on a hot key, all O(nnz) preprocess work
// is reused. On a miss, we build the entry and insert; on overflow, we
// evict the least-recently-used entry. Cache key is the (data-pointer triple
// + N + nnz + dtype + fill_mode + diag_type) of the L matrix: the data
// pointers are stable identifiers for the underlying device buffer, so the
// same L instance hits the cache while a re-allocated copy correctly misses.
// ============================================================================

struct SpSVCacheKey {
    const void* crow_ptr;
    const void* col_ptr;
    const void* vals_ptr;
    int64_t N;
    int64_t nnz;
    DType dtype;
    rocsparse_fill_mode fill_mode;
    rocsparse_diag_type diag_type;

    bool operator==(const SpSVCacheKey& o) const noexcept {
        return crow_ptr == o.crow_ptr && col_ptr == o.col_ptr
            && vals_ptr == o.vals_ptr && N == o.N && nnz == o.nnz
            && dtype == o.dtype && fill_mode == o.fill_mode
            && diag_type == o.diag_type;
    }
};

struct SpSVCacheKeyHash {
    size_t operator()(const SpSVCacheKey& k) const noexcept {
        auto mix = [](size_t a, size_t b) noexcept {
            return a ^ (b + 0x9e3779b97f4a7c15ULL + (a << 6) + (a >> 2));
        };
        size_t h = std::hash<const void*>{}(k.crow_ptr);
        h = mix(h, std::hash<const void*>{}(k.col_ptr));
        h = mix(h, std::hash<const void*>{}(k.vals_ptr));
        h = mix(h, std::hash<int64_t>{}(k.N));
        h = mix(h, std::hash<int64_t>{}(k.nnz));
        h = mix(h, std::hash<int>{}(static_cast<int>(k.dtype)));
        h = mix(h, std::hash<int>{}(static_cast<int>(k.fill_mode)));
        h = mix(h, std::hash<int>{}(static_cast<int>(k.diag_type)));
        return h;
    }
};

struct SpSVCacheEntry {
    // Pinned i32 index buffers — must outlive the descriptor.
    std::unique_ptr<HipBuffer> crow_i32;
    std::unique_ptr<HipBuffer> col_i32;
    // The cached matrix descriptor; preprocess has already been run against
    // it with the workspace below.
    rocsparse_spmat_descr mat_descr = nullptr;
    std::unique_ptr<HipBuffer> workspace;
    size_t workspace_size = 0;
    // Dense vector descriptors used as placeholders during preprocess. Kept
    // alive so the descriptor's internal references remain valid.
    rocsparse_dnvec_descr placeholder_x = nullptr;
    rocsparse_dnvec_descr placeholder_y = nullptr;
    std::unique_ptr<HipBuffer> placeholder_x_buf;
    std::unique_ptr<HipBuffer> placeholder_y_buf;

    SpSVCacheEntry() = default;
    SpSVCacheEntry(const SpSVCacheEntry&) = delete;
    SpSVCacheEntry& operator=(const SpSVCacheEntry&) = delete;
    SpSVCacheEntry(SpSVCacheEntry&&) = default;
    SpSVCacheEntry& operator=(SpSVCacheEntry&&) = default;

    ~SpSVCacheEntry() {
        if (placeholder_x) rocsparse_destroy_dnvec_descr(placeholder_x);
        if (placeholder_y) rocsparse_destroy_dnvec_descr(placeholder_y);
        if (mat_descr) rocsparse_destroy_spmat_descr(mat_descr);
    }
};

class SpSVLruCache {
public:
    static constexpr size_t kMaxEntries = 64;

    SpSVCacheEntry* get(const SpSVCacheKey& key) {
        std::lock_guard<std::mutex> lock(mu_);
        auto it = map_.find(key);
        if (it == map_.end()) return nullptr;
        // Move accessed entry to the front of the LRU list.
        lru_.splice(lru_.begin(), lru_, it->second.lru_it);
        return it->second.entry.get();
    }

    SpSVCacheEntry* insert(const SpSVCacheKey& key,
                           std::unique_ptr<SpSVCacheEntry> entry) {
        std::lock_guard<std::mutex> lock(mu_);
        if (map_.size() >= kMaxEntries) {
            // Evict LRU (tail of list).
            const SpSVCacheKey& victim = lru_.back();
            map_.erase(victim);
            lru_.pop_back();
        }
        lru_.push_front(key);
        SpSVCacheEntry* raw = entry.get();
        map_.emplace(key, Slot{std::move(entry), lru_.begin()});
        return raw;
    }

private:
    struct Slot {
        std::unique_ptr<SpSVCacheEntry> entry;
        std::list<SpSVCacheKey>::iterator lru_it;
    };
    std::mutex mu_;
    std::list<SpSVCacheKey> lru_;
    std::unordered_map<SpSVCacheKey, Slot, SpSVCacheKeyHash> map_;
};

inline SpSVLruCache& spsv_lru_cache() {
    static SpSVLruCache c;
    return c;
}

// Build a fresh cached entry: convert CSR row/col indices to i32, create
// the descriptor with explicit i32 indextypes, set fill/diag attributes,
// run buffer_size + preprocess. The returned entry holds all device-side
// state needed for subsequent stage_compute calls.
SpSVCacheEntry* sparse_trsv_lru_lookup_or_build(
    const Tensor& L_crow, const Tensor& L_col, const Tensor& L_vals,
    int64_t N, int64_t nnz, DType dtype,
    rocsparse_fill_mode fill_mode, rocsparse_diag_type diag_type)
{
    SpSVCacheKey key{
        L_crow.data_ptr(), L_col.data_ptr(), L_vals.data_ptr(),
        N, nnz, dtype, fill_mode, diag_type
    };
    if (auto* hit = spsv_lru_cache().get(key)) return hit;

    if (nnz > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
            "rocm_sparse_trsm: nnz exceeds int32 range required for rocSPARSE SpSV");
    }
    if (N + 1 > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
        throw std::overflow_error(
            "rocm_sparse_trsm: N exceeds int32 range required for rocSPARSE SpSV");
    }
    verify_i64_fits_i32(L_crow.data<int64_t>(), N + 1);
    verify_i64_fits_i32(L_col.data<int64_t>(), nnz);

    auto entry = std::make_unique<SpSVCacheEntry>();
    entry->crow_i32 = std::make_unique<HipBuffer>((N + 1) * sizeof(int32_t));
    entry->col_i32  = std::make_unique<HipBuffer>(nnz * sizeof(int32_t));

    {
        int threads = 256;
        int blocks_crow = static_cast<int>((N + 1 + threads - 1) / threads);
        cast_i64_to_i32<<<blocks_crow, threads>>>(
            L_crow.data<int64_t>(), entry->crow_i32->as<int32_t>(), N + 1);
        HIP_CHECK_SPARSE(hipGetLastError());
        if (nnz > 0) {
            int blocks_nnz = static_cast<int>((nnz + threads - 1) / threads);
            cast_i64_to_i32<<<blocks_nnz, threads>>>(
                L_col.data<int64_t>(), entry->col_i32->as<int32_t>(), nnz);
            HIP_CHECK_SPARSE(hipGetLastError());
        }
    }

    const rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &entry->mat_descr, N, N, nnz,
        entry->crow_i32->ptr,
        entry->col_i32->ptr,
        const_cast<void*>(L_vals.data_ptr()),
        rocsparse_indextype_i32, rocsparse_indextype_i32,
        rocsparse_index_base_zero, roc_dtype));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        entry->mat_descr, rocsparse_spmat_fill_mode,
        &fill_mode, sizeof(fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        entry->mat_descr, rocsparse_spmat_diag_type,
        &diag_type, sizeof(diag_type)));

    // Allocate placeholder dense vectors for buffer_size + preprocess. We
    // can't pass nullptr — rocSPARSE validates the descriptor pointers up
    // front. The vec_x / vec_y used at compute-time will replace these.
    size_t elem_size = (dtype == DType::Float32) ? sizeof(float) : sizeof(double);
    entry->placeholder_x_buf = std::make_unique<HipBuffer>(N * elem_size);
    entry->placeholder_y_buf = std::make_unique<HipBuffer>(N * elem_size);
    HIP_CHECK_SPARSE(hipMemsetAsync(
        entry->placeholder_x_buf->ptr, 0, N * elem_size, nullptr));
    HIP_CHECK_SPARSE(hipMemsetAsync(
        entry->placeholder_y_buf->ptr, 0, N * elem_size, nullptr));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &entry->placeholder_x, N, entry->placeholder_x_buf->ptr, roc_dtype));
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &entry->placeholder_y, N, entry->placeholder_y_buf->ptr, roc_dtype));

    rocsparse_handle handle = get_rocsparse_handle();
    float  alpha_f = 1.0f;
    double alpha_d = 1.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);

    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, entry->mat_descr,
        entry->placeholder_x, entry->placeholder_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_buffer_size, &buffer_size, nullptr));
    entry->workspace_size = buffer_size;
    entry->workspace = std::make_unique<HipBuffer>(buffer_size);

    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, entry->mat_descr,
        entry->placeholder_x, entry->placeholder_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_preprocess, &buffer_size,
        entry->workspace ? entry->workspace->ptr : nullptr));

    return spsv_lru_cache().insert(key, std::move(entry));
}

} // anonymous namespace

Tensor rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("rocm_spmm: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("rocm_spmm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(dense.shape()[0]) + ")");
    }

    DType dtype = dense.dtype();

    // Complex64/Complex128: rocSPARSE doesn't expose native complex SpMM.
    // Decompose on-device into 4 real SpMMs (no host roundtrip):
    //   A = A_re + i*A_im,  B = B_re + i*B_im
    //   C_re = A_re@B_re - A_im@B_im
    //   C_im = A_re@B_im + A_im@B_re
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        if (sparse.values().dtype() != dtype) {
            throw std::runtime_error(
                "rocm_spmm: complex path requires sparse.values().dtype() == dense.dtype()");
        }
        DType real_dtype = (dtype == DType::Complex64) ? DType::Float32 : DType::Float64;

        Tensor val_real_view = ::tenzor::view_as_real(sparse.values());      // (nnz, 2)
        Tensor val_re = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor val_im = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        Tensor dense_real_view = ::tenzor::view_as_real(dense);              // (K, N, 2)
        Tensor B_re = ::tenzor::select(dense_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor B_im = ::tenzor::select(dense_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        auto build_real_sparse = [&](const Tensor& v) {
            return (sparse.layout() == SparseLayout::COO)
                ? SparseTensor::sparse_coo(sparse.indices(), v, sparse.shape())
                : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                           v, sparse.shape());
        };
        SparseTensor A_re = build_real_sparse(val_re);
        SparseTensor A_im = build_real_sparse(val_im);

        Tensor ARe_BRe = rocm_spmm_kernel(A_re, B_re);
        Tensor AIm_BIm = rocm_spmm_kernel(A_im, B_im);
        Tensor ARe_BIm = rocm_spmm_kernel(A_re, B_im);
        Tensor AIm_BRe = rocm_spmm_kernel(A_im, B_re);

        Tensor C_re = ::tenzor::sub(ARe_BRe, AIm_BIm);
        Tensor C_im = ::tenzor::add(ARe_BIm, AIm_BRe);

        // Stack real and imag into a (M, N, 2) real tensor, then view as
        // complex to get (M, N) of the original complex dtype.
        std::array<Tensor, 2> parts = {C_re, C_im};
        std::span<const Tensor> parts_span(parts.data(), parts.size());
        Tensor stacked = ::tenzor::stack(parts_span, /*dim=*/-1).contiguous();
        (void)real_dtype;
        return ::tenzor::view_as_complex(stacked);
    }

    // Wave G6 (deferred → landed): F16/BF16 supported via widen-narrow
    // through F32. rocSPARSE doesn't expose half-precision SpMM directly;
    // widening at the dispatch boundary keeps correctness at the cost of
    // one extra cast pass. Same pattern documented in the codebase memory.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        // Widen sparse + dense to F32, recurse, narrow result back.
        // Use the layout-appropriate constructor directly (H8 fix —
        // previously built CSR with COO indices in a "build-then-overwrite"
        // pattern that was fragile against stricter validation).
        auto sparse_f32_vals = sparse.values().to(DType::Float32);
        SparseTensor sparse_f32 = (sparse.layout() == SparseLayout::COO)
            ? SparseTensor::sparse_coo(sparse.indices(), sparse_f32_vals, sparse.shape())
            : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                       sparse_f32_vals, sparse.shape());
        auto dense_f32 = dense.to(DType::Float32);
        auto result_f32 = rocm_spmm_kernel(sparse_f32, dense_f32);
        return result_f32.to(dtype);
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_spmm: only Float32/Float64/Float16/BFloat16 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto csr = ensure_csr_on_gpu(sparse);
    int64_t nnz = csr.nnz();

    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous()
                     : dense.contiguous();

    auto result = zeros({M, N}, dtype, Device::rocm());

    rocsparse_handle handle = get_rocsparse_handle();
    rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    // Create CSR sparse matrix descriptor
    rocsparse_spmat_descr mat_sparse;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_sparse,
        M, K, nnz,
        const_cast<void*>(static_cast<const void*>(crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(col.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(vals.data_ptr())),
        rocsparse_indextype_i64,
        rocsparse_indextype_i64,
        rocsparse_index_base_zero,
        roc_dtype
    ));
    SpMatGuard sparse_guard(mat_sparse);

    // Dense matrix descriptor (row-major)
    rocsparse_dnmat_descr mat_dense;
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &mat_dense,
        K, N, N,
        const_cast<void*>(dense_gpu.data_ptr()),
        roc_dtype,
        rocsparse_order_row
    ));
    DnMatGuard dense_guard(mat_dense);

    rocsparse_dnmat_descr mat_result;
    ROCSPARSE_CHECK(rocsparse_create_dnmat_descr(
        &mat_result,
        M, N, N,
        result.data_ptr(),
        roc_dtype,
        rocsparse_order_row
    ));
    DnMatGuard result_guard(mat_result);

    // Determine buffer size
    size_t buffer_size = 0;
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    ROCSPARSE_CHECK(rocsparse_spmm(
        handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        roc_dtype,
        rocsparse_spmm_alg_default,
        rocsparse_spmm_stage_buffer_size,
        &buffer_size,
        nullptr
    ));

    HipBuffer workspace(buffer_size);

    ROCSPARSE_CHECK(rocsparse_spmm(
        handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        roc_dtype,
        rocsparse_spmm_alg_default,
        rocsparse_spmm_stage_preprocess,
        &buffer_size,
        workspace.ptr
    ));

    ROCSPARSE_CHECK(rocsparse_spmm(
        handle,
        rocsparse_operation_none,
        rocsparse_operation_none,
        alpha_ptr,
        mat_sparse,
        mat_dense,
        beta_ptr,
        mat_result,
        roc_dtype,
        rocsparse_spmm_alg_default,
        rocsparse_spmm_stage_compute,
        &buffer_size,
        workspace.ptr
    ));

    return result;
}

Tensor rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("rocm_spmv: sparse must be 2D, vec must be 1D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("rocm_spmv: dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(vec.shape()[0]) + ")");
    }

    DType dtype = vec.dtype();

    // Complex64/Complex128: decompose into 4 real SpMVs on device. Mirrors
    // the SpMM complex path above. No host roundtrip.
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        if (sparse.values().dtype() != dtype) {
            throw std::runtime_error(
                "rocm_spmv: complex path requires sparse.values().dtype() == vec.dtype()");
        }

        Tensor val_real_view = ::tenzor::view_as_real(sparse.values());      // (nnz, 2)
        Tensor val_re = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor val_im = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        Tensor vec_real_view = ::tenzor::view_as_real(vec);                  // (K, 2)
        Tensor v_re = ::tenzor::select(vec_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor v_im = ::tenzor::select(vec_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        auto build_real_sparse = [&](const Tensor& v) {
            return (sparse.layout() == SparseLayout::COO)
                ? SparseTensor::sparse_coo(sparse.indices(), v, sparse.shape())
                : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                           v, sparse.shape());
        };
        SparseTensor A_re = build_real_sparse(val_re);
        SparseTensor A_im = build_real_sparse(val_im);

        Tensor ARe_vRe = rocm_spmv_kernel(A_re, v_re);
        Tensor AIm_vIm = rocm_spmv_kernel(A_im, v_im);
        Tensor ARe_vIm = rocm_spmv_kernel(A_re, v_im);
        Tensor AIm_vRe = rocm_spmv_kernel(A_im, v_re);

        Tensor y_re = ::tenzor::sub(ARe_vRe, AIm_vIm);
        Tensor y_im = ::tenzor::add(ARe_vIm, AIm_vRe);

        std::array<Tensor, 2> parts = {y_re, y_im};
        std::span<const Tensor> parts_span(parts.data(), parts.size());
        Tensor stacked = ::tenzor::stack(parts_span, /*dim=*/-1).contiguous();
        return ::tenzor::view_as_complex(stacked);
    }

    // Wave G6 (deferred → landed): F16/BF16 via widen-narrow.
    if (dtype == DType::Float16 || dtype == DType::BFloat16) {
        auto sparse_f32_vals = sparse.values().to(DType::Float32);
        SparseTensor sparse_f32 = (sparse.layout() == SparseLayout::COO)
            ? SparseTensor::sparse_coo(sparse.indices(), sparse_f32_vals, sparse.shape())
            : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                       sparse_f32_vals, sparse.shape());
        auto vec_f32 = vec.to(DType::Float32);
        auto result_f32 = rocm_spmv_kernel(sparse_f32, vec_f32);
        return result_f32.to(dtype);
    }
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_spmv: only Float32/Float64/Float16/BFloat16 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto csr = ensure_csr_on_gpu(sparse);
    int64_t nnz = csr.nnz();

    auto vec_gpu = (vec.device().type != Device::Type::ROCm)
                   ? vec.to(Device::rocm()).contiguous()
                   : vec.contiguous();

    auto result = zeros({M}, dtype, Device::rocm());

    rocsparse_handle handle = get_rocsparse_handle();
    rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    rocsparse_spmat_descr mat_sparse;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_sparse,
        M, K, nnz,
        const_cast<void*>(static_cast<const void*>(crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(col.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(vals.data_ptr())),
        rocsparse_indextype_i64,
        rocsparse_indextype_i64,
        rocsparse_index_base_zero,
        roc_dtype
    ));
    SpMatGuard sparse_guard(mat_sparse);

    rocsparse_dnvec_descr vec_x;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_x,
        K,
        const_cast<void*>(vec_gpu.data_ptr()),
        roc_dtype
    ));
    DnVecGuard vec_x_guard(vec_x);

    rocsparse_dnvec_descr vec_y;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_y,
        M,
        result.data_ptr(),
        roc_dtype
    ));
    DnVecGuard vec_y_guard(vec_y);

    // Set up v2 SpMV descriptor
    float alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0, beta_d = 0.0;
    void* alpha_ptr = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f) : static_cast<void*>(&alpha_d);
    void* beta_ptr = (dtype == DType::Float32) ? static_cast<void*>(&beta_f) : static_cast<void*>(&beta_d);

    rocsparse_spmv_descr spmv_descr;
    ROCSPARSE_CHECK(rocsparse_create_spmv_descr(&spmv_descr));
    SpMVDescrGuard spmv_guard(spmv_descr);

    rocsparse_operation op = rocsparse_operation_none;
    rocsparse_spmv_alg alg = rocsparse_spmv_alg_default;
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_operation, &op, sizeof(op), nullptr));
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_alg, &alg, sizeof(alg), nullptr));
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_scalar_datatype, &roc_dtype, sizeof(roc_dtype), nullptr));
    ROCSPARSE_CHECK(rocsparse_spmv_set_input(handle, spmv_descr,
        rocsparse_spmv_input_compute_datatype, &roc_dtype, sizeof(roc_dtype), nullptr));

    // Analysis stage: get buffer size and run analysis
    size_t analysis_buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(
        handle, spmv_descr, mat_sparse, vec_x, vec_y,
        rocsparse_v2_spmv_stage_analysis, &analysis_buffer_size, nullptr));

    HipBuffer analysis_workspace(analysis_buffer_size);

    ROCSPARSE_CHECK(rocsparse_v2_spmv(
        handle, spmv_descr, alpha_ptr, mat_sparse, vec_x, beta_ptr, vec_y,
        rocsparse_v2_spmv_stage_analysis, analysis_buffer_size, analysis_workspace.ptr, nullptr));

    // Compute stage: get buffer size and run compute
    size_t compute_buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_v2_spmv_buffer_size(
        handle, spmv_descr, mat_sparse, vec_x, vec_y,
        rocsparse_v2_spmv_stage_compute, &compute_buffer_size, nullptr));

    HipBuffer compute_workspace(compute_buffer_size);

    ROCSPARSE_CHECK(rocsparse_v2_spmv(
        handle, spmv_descr, alpha_ptr, mat_sparse, vec_x, beta_ptr, vec_y,
        rocsparse_v2_spmv_stage_compute, compute_buffer_size, compute_workspace.ptr, nullptr));

    return result;
}

// ============================================================================
// SpGEMM — sparse × sparse → sparse (CSR × CSR → CSR) via rocsparse_spgemm
// ============================================================================

/// RAII guard for rocSPARSE csr2csr compression descriptors — not used here
/// but the same pattern applies for any rocSPARSE handle.

SparseTensor rocm_spgemm_kernel(const SparseTensor& a, const SparseTensor& b) {
    auto a_shape = a.shape();
    auto b_shape = b.shape();
    if (a_shape.size() != 2 || b_shape.size() != 2) {
        throw std::runtime_error("rocm_spgemm: both inputs must be 2D");
    }
    const int64_t M = a_shape[0];
    const int64_t K = a_shape[1];
    const int64_t N = b_shape[1];
    if (K != b_shape[0]) {
        throw std::runtime_error("rocm_spgemm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(b_shape[0]) + ")");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("rocm_spgemm: dtype mismatch");
    }
    const DType dtype = a.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_spgemm: only Float32/Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto a_csr = ensure_csr_on_gpu(a);
    auto b_csr = ensure_csr_on_gpu(b);
    const int64_t nnz_a = a_csr.nnz();
    const int64_t nnz_b = b_csr.nnz();

    auto a_crow = a_csr.crow_indices().contiguous();
    auto a_col  = a_csr.col_indices().contiguous();
    auto a_vals = a_csr.values().contiguous();
    auto b_crow = b_csr.crow_indices().contiguous();
    auto b_col  = b_csr.col_indices().contiguous();
    auto b_vals = b_csr.values().contiguous();

    rocsparse_handle handle = get_rocsparse_handle();
    const rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    // A descriptor.
    rocsparse_spmat_descr mat_a;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_a, M, K, nnz_a,
        const_cast<void*>(static_cast<const void*>(a_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(a_col.data<int64_t>())),
        const_cast<void*>(a_vals.data_ptr()),
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard a_guard(mat_a);

    // B descriptor.
    rocsparse_spmat_descr mat_b;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_b, K, N, nnz_b,
        const_cast<void*>(static_cast<const void*>(b_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(b_col.data<int64_t>())),
        const_cast<void*>(b_vals.data_ptr()),
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard b_guard(mat_b);

    // C descriptor: start with an empty CSR with a pre-allocated row ptr
    // buffer. rocsparse_spgemm nnz stage fills it in; compute stage fills
    // col_ind / values once they are allocated.
    auto c_crow = zeros(std::vector<int64_t>{M + 1}, DType::Int64, Device::rocm());
    rocsparse_spmat_descr mat_c;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_c, M, N, 0,
        c_crow.data<int64_t>(), nullptr, nullptr,
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard c_guard(mat_c);

    // D descriptor is a nullptr alias for "no D matrix" — beta must be 0.
    // rocsparse_spgemm requires a valid D descriptor; reuse C for this
    // purpose since it will be read only when beta != 0.
    rocsparse_spmat_descr mat_d;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_d, M, N, 0,
        c_crow.data<int64_t>(), nullptr, nullptr,
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard d_guard(mat_d);

    const rocsparse_operation op_none = rocsparse_operation_none;

    float  alpha_f = 1.0f, beta_f = 0.0f;
    double alpha_d = 1.0,  beta_d = 0.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);
    void* beta  = (dtype == DType::Float32) ? static_cast<void*>(&beta_f)
                                            : static_cast<void*>(&beta_d);

    // Stage 1: buffer size.
    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_spgemm(
        handle, op_none, op_none, alpha, mat_a, mat_b, beta, mat_d, mat_c,
        roc_dtype, rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_buffer_size, &buffer_size, nullptr));
    HipBuffer workspace(buffer_size);

    // Stage 2: nnz. Fills in c_crow and computes the nnz of C.
    ROCSPARSE_CHECK(rocsparse_spgemm(
        handle, op_none, op_none, alpha, mat_a, mat_b, beta, mat_d, mat_c,
        roc_dtype, rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_nnz, &buffer_size, workspace.ptr));

    int64_t c_rows = 0, c_cols = 0, c_nnz = 0;
    ROCSPARSE_CHECK(rocsparse_spmat_get_size(mat_c, &c_rows, &c_cols, &c_nnz));

    auto c_col  = zeros(std::vector<int64_t>{c_nnz}, DType::Int64, Device::rocm());
    auto c_vals = zeros(std::vector<int64_t>{c_nnz}, dtype,         Device::rocm());

    ROCSPARSE_CHECK(rocsparse_csr_set_pointers(
        mat_c,
        c_crow.data<int64_t>(),
        c_col.data<int64_t>(),
        c_vals.data_ptr()));

    // Stage 3: compute.
    ROCSPARSE_CHECK(rocsparse_spgemm(
        handle, op_none, op_none, alpha, mat_a, mat_b, beta, mat_d, mat_c,
        roc_dtype, rocsparse_spgemm_alg_default,
        rocsparse_spgemm_stage_compute, &buffer_size, workspace.ptr));

    HIP_CHECK_SPARSE(hipDeviceSynchronize());

    return SparseTensor::sparse_csr(
        c_crow, c_col, c_vals,
        std::vector<int64_t>{M, N});
}

// ============================================================================
// SpSV — triangular solve L*x = b (single RHS) via rocsparse_spsv
// ============================================================================

Tensor rocm_sparse_trsv_kernel(const SparseTensor& L, const Tensor& b, bool upper) {
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("rocm_sparse_trsv: L must be square 2D");
    }
    if (b.ndim() != 1) {
        throw std::runtime_error("rocm_sparse_trsv: b must be 1D");
    }
    const int64_t N = L_shape[0];
    if (b.shape()[0] != N) {
        throw std::runtime_error("rocm_sparse_trsv: dimension mismatch");
    }
    const DType dtype = b.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_sparse_trsv: only Float32/Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }

    auto L_csr = ensure_csr_on_gpu(L);
    const int64_t nnz = L_csr.nnz();
    auto L_crow = L_csr.crow_indices().contiguous();
    auto L_col  = L_csr.col_indices().contiguous();
    auto L_vals = L_csr.values().contiguous();

    auto b_gpu = (b.device().type != Device::Type::ROCm)
                   ? b.to(Device::rocm()).contiguous()
                   : b.contiguous();
    auto result = zeros(std::vector<int64_t>{N}, dtype, Device::rocm());

    rocsparse_handle handle = get_rocsparse_handle();
    const rocsparse_datatype roc_dtype = get_rocsparse_data_type(dtype);

    // L matrix descriptor.
    rocsparse_spmat_descr mat_L;
    ROCSPARSE_CHECK(rocsparse_create_csr_descr(
        &mat_L, N, N, nnz,
        const_cast<void*>(static_cast<const void*>(L_crow.data<int64_t>())),
        const_cast<void*>(static_cast<const void*>(L_col.data<int64_t>())),
        const_cast<void*>(L_vals.data_ptr()),
        rocsparse_indextype_i64, rocsparse_indextype_i64,
        rocsparse_index_base_zero, roc_dtype));
    SpMatGuard L_guard(mat_L);

    // Fill mode + diagonal type.
    rocsparse_fill_mode fill_mode = upper ? rocsparse_fill_mode_upper
                                          : rocsparse_fill_mode_lower;
    rocsparse_diag_type diag_type = rocsparse_diag_type_non_unit;
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        mat_L, rocsparse_spmat_fill_mode, &fill_mode, sizeof(fill_mode)));
    ROCSPARSE_CHECK(rocsparse_spmat_set_attribute(
        mat_L, rocsparse_spmat_diag_type, &diag_type, sizeof(diag_type)));

    // Dense vector descriptors.
    rocsparse_dnvec_descr vec_x;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_x, N, const_cast<void*>(b_gpu.data_ptr()), roc_dtype));
    DnVecGuard x_guard(vec_x);

    rocsparse_dnvec_descr vec_y;
    ROCSPARSE_CHECK(rocsparse_create_dnvec_descr(
        &vec_y, N, result.data_ptr(), roc_dtype));
    DnVecGuard y_guard(vec_y);

    float  alpha_f = 1.0f;
    double alpha_d = 1.0;
    void* alpha = (dtype == DType::Float32) ? static_cast<void*>(&alpha_f)
                                            : static_cast<void*>(&alpha_d);

    // Stage 1: buffer size.
    size_t buffer_size = 0;
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, mat_L, vec_x, vec_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_buffer_size, &buffer_size, nullptr));
    HipBuffer workspace(buffer_size);

    // Stage 2: preprocess.
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, mat_L, vec_x, vec_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_preprocess, &buffer_size, workspace.ptr));

    // Stage 3: compute.
    ROCSPARSE_CHECK(rocsparse_spsv(
        handle, rocsparse_operation_none, alpha, mat_L, vec_x, vec_y,
        roc_dtype, rocsparse_spsv_alg_default,
        rocsparse_spsv_stage_compute, &buffer_size, workspace.ptr));

    HIP_CHECK_SPARSE(hipDeviceSynchronize());
    return result;
}

// Multi-RHS triangular solve: loops per column calling SpSV. As on the
// CUDA side, this is O(K) suboptimal — rocsparse has SpSM (multi-RHS)
// but caching descriptors and workspace across K calls is a follow-up.
Tensor rocm_sparse_trsm_kernel(const SparseTensor& L, const Tensor& B, bool upper) {
    auto L_shape = L.shape();
    if (L_shape.size() != 2 || L_shape[0] != L_shape[1]) {
        throw std::runtime_error("rocm_sparse_trsm: L must be square 2D");
    }
    if (B.ndim() != 2) {
        throw std::runtime_error("rocm_sparse_trsm: B must be 2D");
    }
    const int64_t N = L_shape[0];
    const int64_t K = B.shape()[1];
    if (B.shape()[0] != N) {
        throw std::runtime_error("rocm_sparse_trsm: dimension mismatch");
    }
    const DType dtype = B.dtype();
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error("rocm_sparse_trsm: only Float32/Float64 supported");
    }

    auto B_gpu = (B.device().type != Device::Type::ROCm)
                   ? B.to(Device::rocm()).contiguous()
                   : B.contiguous();
    auto X = zeros(std::vector<int64_t>{N, K}, dtype, Device::rocm());

    for (int64_t k = 0; k < K; ++k) {
        // Gather B[:, k] into a contiguous 1D buffer on GPU. B is (N, K)
        // row-major, so column k is a strided view; materialise it with a
        // single contiguous copy instead of one hipMemcpy per element.
        auto b_col = B_gpu.slice(1, k, k + 1).squeeze(1).contiguous();

        auto x_col = rocm_sparse_trsv_kernel(L, b_col, upper);

        // Scatter x_col back into X[:, k] with a single strided 2D copy
        // (matching sparse_trsm_standalone_hip) instead of N tiny memcpys.
        if (dtype == DType::Float32) {
            HIP_CHECK_SPARSE(hipMemcpy2DAsync(
                X.data<float>() + k, K * sizeof(float),
                x_col.data<float>(), sizeof(float),
                sizeof(float), N, hipMemcpyDeviceToDevice, 0));
        } else {
            HIP_CHECK_SPARSE(hipMemcpy2DAsync(
                X.data<double>() + k, K * sizeof(double),
                x_col.data<double>(), sizeof(double),
                sizeof(double), N, hipMemcpyDeviceToDevice, 0));
        }
    }

    HIP_CHECK_SPARSE(hipDeviceSynchronize());
    return X;
}

// ============================================================================
// GPU-native sparse format conversions (called from SparseTensor methods)
// ============================================================================

SparseTensor rocm_coo_to_csr(const SparseTensor& sparse) {
    return ensure_csr_on_gpu(sparse);
}

SparseTensor rocm_coalesce(const SparseTensor& sparse) {
    // Coalesce COO on GPU using thrust (HIP-compatible) sort + reduce_by_key.
    if (sparse.layout() != SparseLayout::COO) return sparse;
    if (sparse.nnz() == 0) return sparse;

    auto sp_shape = sparse.shape();
    int64_t sparse_dim = static_cast<int64_t>(sp_shape.size());
    int64_t nnz = sparse.nnz();

    Tensor indices = sparse.indices().contiguous();
    Tensor values = sparse.values().contiguous();
    const int64_t* idx_ptr = indices.data<int64_t>();

    // Compute strides for linearized keys
    std::vector<int64_t> h_strides(sparse_dim);
    if (sparse_dim > 0) {
        h_strides[sparse_dim - 1] = 1;
        for (int64_t d = sparse_dim - 2; d >= 0; --d) {
            h_strides[d] = h_strides[d + 1] * sp_shape[d + 1];
        }
    }

    // Build linearized keys on GPU
    Tensor keys_t = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* keys_ptr = keys_t.data<int64_t>();

    for (int64_t d = 0; d < sparse_dim; ++d) {
        if (h_strides[d] == 0) continue;
        const int64_t* dim_idx = idx_ptr + d * nnz;
        int64_t stride = h_strides[d];
        auto keys_dptr = thrust::device_pointer_cast(keys_ptr);
        auto dim_dptr = thrust::device_pointer_cast(dim_idx);
        thrust::transform(thrust::hip::par,
                          keys_dptr, keys_dptr + nnz,
                          dim_dptr,
                          keys_dptr,
                          [stride] __device__ (int64_t k, int64_t idx_val) {
                              return k + idx_val * stride;
                          });
    }

    // Sort by key with permutation
    Tensor perm_t = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* perm_ptr = perm_t.data<int64_t>();
    thrust::sequence(thrust::hip::par, thrust::device_pointer_cast(perm_ptr),
                     thrust::device_pointer_cast(perm_ptr + nnz));
    thrust::sort_by_key(thrust::hip::par,
                        thrust::device_pointer_cast(keys_ptr),
                        thrust::device_pointer_cast(keys_ptr + nnz),
                        thrust::device_pointer_cast(perm_ptr));

    // Gather sorted indices and values
    Tensor sorted_indices = zeros({sparse_dim, nnz}, DType::Int64, Device::rocm());
    int64_t* si_ptr = sorted_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(idx_ptr + d * nnz),
                       thrust::device_pointer_cast(si_ptr + d * nnz));
    }

    Tensor sorted_vals = zeros({nnz}, values.dtype(), Device::rocm());
    if (values.dtype() == DType::Float32) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<float>()),
                       thrust::device_pointer_cast(sorted_vals.data<float>()));
    } else if (values.dtype() == DType::Float64) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<double>()),
                       thrust::device_pointer_cast(sorted_vals.data<double>()));
    }

    // Reduce duplicates by key
    Tensor out_keys = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* ok_ptr = out_keys.data<int64_t>();
    Tensor out_vals = zeros({nnz}, values.dtype(), Device::rocm());
    int64_t new_nnz = 0;

    if (values.dtype() == DType::Float32) {
        auto end = thrust::reduce_by_key(
            thrust::hip::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<float>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<float>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    } else if (values.dtype() == DType::Float64) {
        auto end = thrust::reduce_by_key(
            thrust::hip::par,
            thrust::device_pointer_cast(keys_ptr),
            thrust::device_pointer_cast(keys_ptr + nnz),
            thrust::device_pointer_cast(sorted_vals.data<double>()),
            thrust::device_pointer_cast(ok_ptr),
            thrust::device_pointer_cast(out_vals.data<double>()));
        new_nnz = end.first - thrust::device_pointer_cast(ok_ptr);
    }

    // Decode linearized keys back to multi-dim indices
    Tensor new_indices = zeros({sparse_dim, new_nnz}, DType::Int64, Device::rocm());
    int64_t* ni_ptr = new_indices.data<int64_t>();
    for (int64_t d = 0; d < sparse_dim; ++d) {
        int64_t stride = h_strides[d];
        int64_t extent = sp_shape[d];
        auto ok_dptr = thrust::device_pointer_cast(ok_ptr);
        auto ni_dptr = thrust::device_pointer_cast(ni_ptr + d * new_nnz);
        // Decode coordinate for dim d as (key / stride_d) % extent_d. Using the
        // dim's own extent (not the next dim's, nor a degenerate `key % 1` for
        // the last dim) is required for non-square / multi-dim tensors.
        thrust::transform(thrust::hip::par,
                          ok_dptr, ok_dptr + new_nnz,
                          ni_dptr,
                          [stride, extent] __device__ (int64_t key) {
                              return (key / stride) % extent;
                          });
    }

    Tensor final_vals = zeros({new_nnz}, values.dtype(), Device::rocm());
    if (values.dtype() == DType::Float32) {
        HIP_CHECK_SPARSE(hipMemcpy(final_vals.data<float>(), out_vals.data<float>(),
                                   new_nnz * sizeof(float), hipMemcpyDeviceToDevice));
    } else if (values.dtype() == DType::Float64) {
        HIP_CHECK_SPARSE(hipMemcpy(final_vals.data<double>(), out_vals.data<double>(),
                                   new_nnz * sizeof(double), hipMemcpyDeviceToDevice));
    }

    return SparseTensor::sparse_coo(new_indices, final_vals,
                                    std::vector<int64_t>(sp_shape.begin(), sp_shape.end()));
}

SparseTensor rocm_coo_to_csc(const SparseTensor& sparse) {
    if (sparse.layout() == SparseLayout::CSC) return sparse;

    auto sp_shape = sparse.shape();
    int64_t nrows = sp_shape[0];
    int64_t ncols = sp_shape[1];

    auto coo = rocm_coalesce(sparse);
    int64_t nnz = coo.nnz();

    Tensor indices = coo.indices().contiguous();
    Tensor values = coo.values().contiguous();
    const int64_t* idx_ptr = indices.data<int64_t>();
    const int64_t* row_ptr_src = idx_ptr;
    const int64_t* col_ptr_src = idx_ptr + nnz;

    // Sort by (col, row): key = col * nrows + row
    Tensor sort_keys = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* sk_ptr = sort_keys.data<int64_t>();
    {
        auto sk_dptr = thrust::device_pointer_cast(sk_ptr);
        auto col_dptr = thrust::device_pointer_cast(col_ptr_src);
        auto row_dptr = thrust::device_pointer_cast(row_ptr_src);
        thrust::transform(thrust::hip::par,
                          col_dptr, col_dptr + nnz,
                          row_dptr,
                          sk_dptr,
                          [nrows] __device__ (int64_t c, int64_t r) {
                              return c * nrows + r;
                          });
    }

    Tensor perm_t = zeros({nnz}, DType::Int64, Device::rocm());
    int64_t* perm_ptr = perm_t.data<int64_t>();
    thrust::sequence(thrust::hip::par, thrust::device_pointer_cast(perm_ptr),
                     thrust::device_pointer_cast(perm_ptr + nnz));
    thrust::sort_by_key(thrust::hip::par,
                        thrust::device_pointer_cast(sk_ptr),
                        thrust::device_pointer_cast(sk_ptr + nnz),
                        thrust::device_pointer_cast(perm_ptr));

    // Gather sorted row/col indices
    Tensor sorted_rows = zeros({nnz}, DType::Int64, Device::rocm());
    thrust::gather(thrust::hip::par,
                   thrust::device_pointer_cast(perm_ptr),
                   thrust::device_pointer_cast(perm_ptr + nnz),
                   thrust::device_pointer_cast(row_ptr_src),
                   thrust::device_pointer_cast(sorted_rows.data<int64_t>()));

    Tensor sorted_cols = zeros({nnz}, DType::Int64, Device::rocm());
    thrust::gather(thrust::hip::par,
                   thrust::device_pointer_cast(perm_ptr),
                   thrust::device_pointer_cast(perm_ptr + nnz),
                   thrust::device_pointer_cast(col_ptr_src),
                   thrust::device_pointer_cast(sorted_cols.data<int64_t>()));

    // Build ccol_indices using upper_bound
    Tensor ccol = zeros({ncols + 1}, DType::Int64, Device::rocm());
    int64_t* ccol_ptr = ccol.data<int64_t>();
    Tensor boundaries = zeros({ncols}, DType::Int64, Device::rocm());
    int64_t* bounds_ptr = boundaries.data<int64_t>();
    thrust::upper_bound(thrust::hip::par,
                        thrust::device_pointer_cast(sorted_cols.data<int64_t>()),
                        thrust::device_pointer_cast(sorted_cols.data<int64_t>() + nnz),
                        thrust::counting_iterator<int64_t>(0),
                        thrust::counting_iterator<int64_t>(ncols),
                        thrust::device_pointer_cast(bounds_ptr));
    HIP_CHECK_SPARSE(hipMemset(ccol_ptr, 0, sizeof(int64_t)));
    HIP_CHECK_SPARSE(hipMemcpy(ccol_ptr + 1, bounds_ptr,
                               ncols * sizeof(int64_t), hipMemcpyDeviceToDevice));

    // Gather sorted values
    Tensor sorted_vals = zeros({nnz}, values.dtype(), Device::rocm());
    if (values.dtype() == DType::Float32) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<float>()),
                       thrust::device_pointer_cast(sorted_vals.data<float>()));
    } else if (values.dtype() == DType::Float64) {
        thrust::gather(thrust::hip::par,
                       thrust::device_pointer_cast(perm_ptr),
                       thrust::device_pointer_cast(perm_ptr + nnz),
                       thrust::device_pointer_cast(values.data<double>()),
                       thrust::device_pointer_cast(sorted_vals.data<double>()));
    }

    return SparseTensor::sparse_csc(ccol, sorted_rows, sorted_vals,
                                    std::vector<int64_t>{nrows, ncols});
}

Tensor rocm_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2)
        throw std::runtime_error("rocm_sparse_add: both inputs must be 2D");
    int64_t M = sp_shape[0], K = sp_shape[1];
    if (M != dense.shape()[0] || K != dense.shape()[1])
        throw std::runtime_error("rocm_sparse_add: shape mismatch");

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);
    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous() : dense.contiguous();
    auto result = dense_gpu.clone();

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_sparse_add_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            result.data<double>(), M, K);
    } else {
        throw std::runtime_error("rocm_sparse_add: only Float32 and Float64 supported");
    }
    HIP_CHECK_SPARSE(hipGetLastError());
    return result;
}

} // namespace rocm
} // namespace tenzor

#else // !TENZOR_HAS_ROCSPARSE — native HIP CSR SpMM/SpMV fallbacks

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"

#include <hip/hip_runtime.h>
#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace tenzor {
auto zeros(std::vector<int64_t> shape, DType dtype, Device device) -> Tensor;
}

namespace tenzor {
namespace rocm {

namespace {

#define HIP_CHECK_SPARSE(call)                                                  \
    do {                                                                         \
        hipError_t err = (call);                                                \
        if (err != hipSuccess) {                                                \
            throw std::runtime_error(                                           \
                std::string("HIP error in sparse at ") + __FILE__ + ":" +      \
                std::to_string(__LINE__) + " - " + hipGetErrorString(err));     \
        }                                                                       \
    } while (0)

// CSR SpMV kernel: one thread per row, accumulates dot product
template <typename T>
__global__ void csr_spmv_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    const T* __restrict__ x_ptr,
    T* __restrict__ y_ptr,
    int64_t nrows, int64_t ncols)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;

    T sum = static_cast<T>(0);
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        // x_ptr has length == the CSR column count (ncols). Guard both bounds:
        // an out-of-range col index would be an out-of-bounds device read of
        // x_ptr.
        int64_t k = col_ptr[j];
        if (k < 0 || k >= ncols) continue;
        sum += val_ptr[j] * x_ptr[k];
    }
    y_ptr[row] = sum;
}

// CSR SpMM kernel: one thread per output element (row, col)
template <typename T>
__global__ void csr_spmm_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    const T* __restrict__ b_ptr,
    T* __restrict__ c_ptr,
    int64_t nrows, int64_t ncols_b, int64_t nrows_b)
{
    int64_t idx = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    int64_t row = idx / ncols_b;
    int64_t col = idx % ncols_b;
    if (row >= nrows) return;

    T sum = static_cast<T>(0);
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        // Guard against malformed/untrusted CSR col indices: an out-of-range
        // column would be an out-of-bounds device read of b_ptr. The CSR column
        // count equals B's row count (nrows_b == ncols_a).
        int64_t k = col_ptr[j];
        if (k < 0 || k >= nrows_b) continue;
        sum += val_ptr[j] * b_ptr[k * ncols_b + col];
    }
    c_ptr[row * ncols_b + col] = sum;
}

// CSR SparseAdd kernel: one thread per row, adds sparse values into dense output
template <typename T>
__global__ void csr_sparse_add_kernel(
    const int64_t* __restrict__ crow_ptr,
    const int64_t* __restrict__ col_ptr,
    const T* __restrict__ val_ptr,
    T* __restrict__ out_ptr,
    int64_t nrows, int64_t ncols)
{
    int64_t row = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (row >= nrows) return;
    int64_t row_start = crow_ptr[row];
    int64_t row_end = crow_ptr[row + 1];
    for (int64_t j = row_start; j < row_end; ++j) {
        // Guard against malformed/untrusted CSR col indices: an out-of-range
        // column would be an out-of-bounds device write into out_ptr.
        int64_t c = col_ptr[j];
        if (c < 0 || c >= ncols) continue;
        out_ptr[row * ncols + c] += val_ptr[j];
    }
}

/// Ensure SparseTensor is in CSR format on ROCm device.
SparseTensor ensure_csr_on_gpu(const SparseTensor& sparse) {
    auto sp = (sparse.device().type != Device::Type::ROCm)
              ? sparse.to(Device::rocm())
              : sparse;
    if (sp.layout() != SparseLayout::CSR) {
        throw std::runtime_error("rocm native sparse fallback requires CSR format");
    }
    return sp;
}

} // anonymous namespace

Tensor rocm_spmm_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("rocm_spmm: both inputs must be 2D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("rocm_spmm: inner dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(dense.shape()[0]) + ")");
    }

    DType dtype = dense.dtype();

    // Complex64/Complex128: decompose into 4 real SpMMs on device — same
    // strategy as the rocSPARSE path, since this fallback only has real
    // CSR SpMM kernels.
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        if (sparse.values().dtype() != dtype) {
            throw std::runtime_error(
                "rocm_spmm (HIP fallback): complex path requires "
                "sparse.values().dtype() == dense.dtype()");
        }
        Tensor val_real_view = ::tenzor::view_as_real(sparse.values());
        Tensor val_re = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor val_im = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        Tensor dense_real_view = ::tenzor::view_as_real(dense);
        Tensor B_re = ::tenzor::select(dense_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor B_im = ::tenzor::select(dense_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        auto build_real_sparse = [&](const Tensor& v) {
            return (sparse.layout() == SparseLayout::COO)
                ? SparseTensor::sparse_coo(sparse.indices(), v, sparse.shape())
                : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                           v, sparse.shape());
        };
        SparseTensor A_re = build_real_sparse(val_re);
        SparseTensor A_im = build_real_sparse(val_im);

        Tensor ARe_BRe = rocm_spmm_kernel(A_re, B_re);
        Tensor AIm_BIm = rocm_spmm_kernel(A_im, B_im);
        Tensor ARe_BIm = rocm_spmm_kernel(A_re, B_im);
        Tensor AIm_BRe = rocm_spmm_kernel(A_im, B_re);

        Tensor C_re = ::tenzor::sub(ARe_BRe, AIm_BIm);
        Tensor C_im = ::tenzor::add(ARe_BIm, AIm_BRe);

        std::array<Tensor, 2> parts = {C_re, C_im};
        std::span<const Tensor> parts_span(parts.data(), parts.size());
        Tensor stacked = ::tenzor::stack(parts_span, /*dim=*/-1).contiguous();
        return ::tenzor::view_as_complex(stacked);
    }

    auto csr = ensure_csr_on_gpu(sparse);

    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous()
                     : dense.contiguous();

    auto result = zeros({M, N}, dtype, Device::rocm());

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int64_t total = M * N;
    int blocks = static_cast<int>((total + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_spmm_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            dense_gpu.data<float>(), result.data<float>(), M, N, K);
    } else if (dtype == DType::Float64) {
        csr_spmm_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            dense_gpu.data<double>(), result.data<double>(), M, N, K);
    } else {
        throw std::runtime_error("rocm_spmm: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    HIP_CHECK_SPARSE(hipGetLastError());

    return result;
}

Tensor rocm_spmv_kernel(const SparseTensor& sparse, const Tensor& vec) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("rocm_spmv: sparse must be 2D, vec must be 1D");
    }

    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("rocm_spmv: dimensions must match ("
            + std::to_string(K) + " vs " + std::to_string(vec.shape()[0]) + ")");
    }

    DType dtype = vec.dtype();

    // Complex64/Complex128: same on-device decomposition as SpMM.
    if (dtype == DType::Complex64 || dtype == DType::Complex128) {
        if (sparse.values().dtype() != dtype) {
            throw std::runtime_error(
                "rocm_spmv (HIP fallback): complex path requires "
                "sparse.values().dtype() == vec.dtype()");
        }
        Tensor val_real_view = ::tenzor::view_as_real(sparse.values());
        Tensor val_re = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor val_im = ::tenzor::select(val_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        Tensor vec_real_view = ::tenzor::view_as_real(vec);
        Tensor v_re = ::tenzor::select(vec_real_view, /*dim=*/-1, /*index=*/0).contiguous();
        Tensor v_im = ::tenzor::select(vec_real_view, /*dim=*/-1, /*index=*/1).contiguous();

        auto build_real_sparse = [&](const Tensor& v) {
            return (sparse.layout() == SparseLayout::COO)
                ? SparseTensor::sparse_coo(sparse.indices(), v, sparse.shape())
                : SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(),
                                           v, sparse.shape());
        };
        SparseTensor A_re = build_real_sparse(val_re);
        SparseTensor A_im = build_real_sparse(val_im);

        Tensor ARe_vRe = rocm_spmv_kernel(A_re, v_re);
        Tensor AIm_vIm = rocm_spmv_kernel(A_im, v_im);
        Tensor ARe_vIm = rocm_spmv_kernel(A_re, v_im);
        Tensor AIm_vRe = rocm_spmv_kernel(A_im, v_re);

        Tensor y_re = ::tenzor::sub(ARe_vRe, AIm_vIm);
        Tensor y_im = ::tenzor::add(ARe_vIm, AIm_vRe);

        std::array<Tensor, 2> parts = {y_re, y_im};
        std::span<const Tensor> parts_span(parts.data(), parts.size());
        Tensor stacked = ::tenzor::stack(parts_span, /*dim=*/-1).contiguous();
        return ::tenzor::view_as_complex(stacked);
    }

    auto csr = ensure_csr_on_gpu(sparse);

    auto vec_gpu = (vec.device().type != Device::Type::ROCm)
                   ? vec.to(Device::rocm()).contiguous()
                   : vec.contiguous();

    auto result = zeros({M}, dtype, Device::rocm());

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_spmv_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            vec_gpu.data<float>(), result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_spmv_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            vec_gpu.data<double>(), result.data<double>(), M, K);
    } else {
        throw std::runtime_error("rocm_spmv: only Float32 and Float64 supported, got "
            + std::string(dtype_name(dtype)));
    }
    HIP_CHECK_SPARSE(hipGetLastError());

    return result;
}

Tensor rocm_sparse_add_kernel(const SparseTensor& sparse, const Tensor& dense) {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2)
        throw std::runtime_error("rocm_sparse_add: both inputs must be 2D");
    int64_t M = sp_shape[0], K = sp_shape[1];
    if (M != dense.shape()[0] || K != dense.shape()[1])
        throw std::runtime_error("rocm_sparse_add: shape mismatch");

    DType dtype = dense.dtype();
    auto csr = ensure_csr_on_gpu(sparse);
    auto dense_gpu = (dense.device().type != Device::Type::ROCm)
                     ? dense.to(Device::rocm()).contiguous() : dense.contiguous();
    auto result = dense_gpu.clone();

    auto crow = csr.crow_indices().contiguous();
    auto col = csr.col_indices().contiguous();
    auto vals = csr.values().contiguous();

    int threads = 256;
    int blocks = static_cast<int>((M + threads - 1) / threads);

    if (dtype == DType::Float32) {
        csr_sparse_add_kernel<float><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<float>(),
            result.data<float>(), M, K);
    } else if (dtype == DType::Float64) {
        csr_sparse_add_kernel<double><<<blocks, threads>>>(
            crow.data<int64_t>(), col.data<int64_t>(), vals.data<double>(),
            result.data<double>(), M, K);
    } else {
        throw std::runtime_error("rocm_sparse_add: only Float32 and Float64 supported");
    }
    HIP_CHECK_SPARSE(hipGetLastError());
    return result;
}

} // namespace rocm
} // namespace tenzor

#endif // TENZOR_HAS_ROCSPARSE

// ============================================================================
// Standalone GPU SpGEMM and SparseTrsv/Trsm (no rocSPARSE dependency)
// ============================================================================

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include <hip/hip_runtime.h>
#include <hipcub/hipcub.hpp>
#include <cstdint>
#include <stdexcept>
#include <vector>

namespace tenzor {
namespace rocm {

#ifndef HIP_CHECK_SPARSE_SA
#define HIP_CHECK_SPARSE_SA(call)                                              \
    do {                                                                        \
        hipError_t err = (call);                                               \
        if (err != hipSuccess) {                                               \
            throw std::runtime_error(                                          \
                std::string("HIP error in sparse_standalone at ") +            \
                __FILE__ + ":" + std::to_string(__LINE__) + " - " +            \
                hipGetErrorString(err));                                        \
        }                                                                      \
    } while (0)
#endif

// SpGEMM count kernel
template <typename T>
__global__ void spgemm_count_hip(
    const int64_t* __restrict__ a_crow, const int64_t* __restrict__ a_col,
    const int64_t* __restrict__ b_crow, const int64_t* __restrict__ b_col,
    int64_t* __restrict__ row_nnz, int64_t M, int64_t N)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int64_t count = 0;
    for (int64_t ja = a_crow[row]; ja < a_crow[row + 1]; ++ja) {
        int64_t k = a_col[ja];
        count += b_crow[k + 1] - b_crow[k];
    }
    row_nnz[row] = count;
}

// SpGEMM fill kernel with dedup
template <typename T>
__global__ void spgemm_fill_hip(
    const int64_t* __restrict__ a_crow, const int64_t* __restrict__ a_col,
    const T* __restrict__ a_vals,
    const int64_t* __restrict__ b_crow, const int64_t* __restrict__ b_col,
    const T* __restrict__ b_vals,
    const int64_t* __restrict__ c_crow, int64_t* __restrict__ c_col,
    T* __restrict__ c_vals, int64_t* __restrict__ c_row_nnz,
    int64_t M, int64_t N)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int64_t c_start = c_crow[row];
    int64_t write_pos = c_start;
    for (int64_t ja = a_crow[row]; ja < a_crow[row + 1]; ++ja) {
        int64_t k = a_col[ja];
        T a_val = a_vals[ja];
        for (int64_t jb = b_crow[k]; jb < b_crow[k + 1]; ++jb) {
            int64_t col = b_col[jb];
            T val = a_val * b_vals[jb];
            bool found = false;
            for (int64_t p = c_start; p < write_pos; ++p) {
                if (c_col[p] == col) { c_vals[p] += val; found = true; break; }
            }
            if (!found) { c_col[write_pos] = col; c_vals[write_pos] = val; write_pos++; }
        }
    }
    c_row_nnz[row] = write_pos - c_start;
}

// Compact kernel
template <typename T>
__global__ void spgemm_compact_hip(
    const int64_t* __restrict__ old_crow, const int64_t* __restrict__ new_crow,
    const int64_t* __restrict__ old_col, const T* __restrict__ old_vals,
    int64_t* __restrict__ new_col, T* __restrict__ new_vals,
    const int64_t* __restrict__ row_nnz, int64_t M)
{
    int64_t row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= M) return;
    int64_t src = old_crow[row], dst = new_crow[row], count = row_nnz[row];
    for (int64_t i = 0; i < count; ++i) {
        new_col[dst + i] = old_col[src + i];
        new_vals[dst + i] = old_vals[src + i];
    }
}

template <typename T>
auto spgemm_standalone_typed_hip(
    const Tensor& a_crow, const Tensor& a_col, const Tensor& a_vals,
    const Tensor& b_crow, const Tensor& b_col, const Tensor& b_vals,
    int64_t M, int64_t K, int64_t N, hipStream_t stream) -> std::vector<Tensor>
{
    constexpr int BLK = 256;
    int64_t nblk = (M + BLK - 1) / BLK;

    // Pass 1: count
    int64_t* d_rnnz; HIP_CHECK_SPARSE_SA(hipMalloc(&d_rnnz, M * sizeof(int64_t)));
    hipLaunchKernelGGL(spgemm_count_hip<T>, dim3(nblk), dim3(BLK), 0, stream,
        a_crow.data<int64_t>(), a_col.data<int64_t>(),
        b_crow.data<int64_t>(), b_col.data<int64_t>(), d_rnnz, M, N);

    // Pass 2: prefix sum
    int64_t* d_crow_ub; HIP_CHECK_SPARSE_SA(hipMalloc(&d_crow_ub, (M + 1) * sizeof(int64_t)));
    void* d_tmp = nullptr; size_t tmp_bytes = 0;
    hipcub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_rnnz, d_crow_ub, M, stream);
    HIP_CHECK_SPARSE_SA(hipMalloc(&d_tmp, tmp_bytes));
    hipcub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_rnnz, d_crow_ub, M, stream);
    HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
    hipFree(d_tmp);

    int64_t lp = 0, lc = 0;
    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(&lp, d_crow_ub + M - 1, sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(&lc, d_rnnz + M - 1, sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
    int64_t total_ub = lp + lc;
    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(d_crow_ub + M, &total_ub, sizeof(int64_t), hipMemcpyHostToDevice, stream));

    if (total_ub == 0) {
        HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
        hipFree(d_rnnz); hipFree(d_crow_ub);
        return {tenzor::zeros({M+1}, DType::Int64, a_crow.device()),
                tenzor::empty({0}, DType::Int64, a_crow.device()),
                tenzor::empty({0}, a_vals.dtype(), a_crow.device())};
    }

    // Pass 3: fill
    int64_t* d_col_ub; T* d_vals_ub; int64_t* d_annz;
    HIP_CHECK_SPARSE_SA(hipMalloc(&d_col_ub, total_ub * sizeof(int64_t)));
    HIP_CHECK_SPARSE_SA(hipMalloc(&d_vals_ub, total_ub * sizeof(T)));
    HIP_CHECK_SPARSE_SA(hipMalloc(&d_annz, M * sizeof(int64_t)));

    hipLaunchKernelGGL(spgemm_fill_hip<T>, dim3(nblk), dim3(BLK), 0, stream,
        a_crow.data<int64_t>(), a_col.data<int64_t>(), a_vals.data<T>(),
        b_crow.data<int64_t>(), b_col.data<int64_t>(), b_vals.data<T>(),
        d_crow_ub, d_col_ub, d_vals_ub, d_annz, M, N);

    // Compact
    int64_t* d_crow_f; HIP_CHECK_SPARSE_SA(hipMalloc(&d_crow_f, (M + 1) * sizeof(int64_t)));
    d_tmp = nullptr; tmp_bytes = 0;
    hipcub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_annz, d_crow_f, M, stream);
    HIP_CHECK_SPARSE_SA(hipMalloc(&d_tmp, tmp_bytes));
    hipcub::DeviceScan::ExclusiveSum(d_tmp, tmp_bytes, d_annz, d_crow_f, M, stream);
    HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
    hipFree(d_tmp);

    int64_t alp = 0, alc = 0;
    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(&alp, d_crow_f + M - 1, sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(&alc, d_annz + M - 1, sizeof(int64_t), hipMemcpyDeviceToHost, stream));
    HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
    int64_t total_nnz = alp + alc;
    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(d_crow_f + M, &total_nnz, sizeof(int64_t), hipMemcpyHostToDevice, stream));

    auto c_crow_t = Tensor({M + 1}, DType::Int64, a_crow.device());
    auto c_col_t = Tensor({total_nnz}, DType::Int64, a_crow.device());
    auto c_vals_t = Tensor({total_nnz}, a_vals.dtype(), a_crow.device());

    HIP_CHECK_SPARSE_SA(hipMemcpyAsync(c_crow_t.data<int64_t>(), d_crow_f,
        (M + 1) * sizeof(int64_t), hipMemcpyDeviceToDevice, stream));

    if (total_nnz > 0) {
        hipLaunchKernelGGL(spgemm_compact_hip<T>, dim3(nblk), dim3(BLK), 0, stream,
            d_crow_ub, d_crow_f, d_col_ub, d_vals_ub,
            c_col_t.data<int64_t>(), c_vals_t.data<T>(), d_annz, M);
    }

    HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
    hipFree(d_rnnz); hipFree(d_crow_ub); hipFree(d_col_ub);
    hipFree(d_vals_ub); hipFree(d_annz); hipFree(d_crow_f);
    return {c_crow_t, c_col_t, c_vals_t};
}

auto spgemm_standalone_hip(std::span<const Tensor> inputs, const OpAttributes& attrs,
                           hipStream_t stream) -> std::vector<Tensor> {
    int64_t M = attrs.get_int(AttrKey::M);
    int64_t K = attrs.get_int(AttrKey::K);
    int64_t N = attrs.get_int(AttrKey::N);
    if (inputs[2].dtype() == DType::Float32)
        return spgemm_standalone_typed_hip<float>(inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], M, K, N, stream);
    else if (inputs[2].dtype() == DType::Float64)
        return spgemm_standalone_typed_hip<double>(inputs[0], inputs[1], inputs[2],
            inputs[3], inputs[4], inputs[5], M, K, N, stream);
    throw std::runtime_error("spgemm_standalone_hip: only Float32/Float64");
}

// Sparse triangular solve: single-thread sequential kernel.
//
// Parallel triangular solve on GPU is tricky: the obvious "one thread per row +
// atomic spin-wait on dependencies" approach deadlocks under SIMT execution —
// divergent threads within a warp that spin waiting on a dependency stall the
// thread that would write the dependency (they're all in the same warp and
// only one divergent branch runs at a time). One-block-per-row also can't
// guarantee forward progress when the number of rows exceeds resident block
// capacity.
//
// For correctness across all hardware we run the solve on a single GPU thread.
// This is sequential (O(nnz)) but bit-correct and matches CPU/CUDA results.
// rocSPARSE's SpSV has been observed to hang on small triangular systems in
// recent ROCm releases, so we deliberately do not delegate to it.
template <typename T>
__global__ void sparse_trsv_sequential_kernel(
    const int64_t* __restrict__ crow, const int64_t* __restrict__ col,
    const T* __restrict__ vals, const T* __restrict__ b,
    T* __restrict__ x, int64_t N, bool upper)
{
    if (blockIdx.x != 0 || threadIdx.x != 0) return;
    for (int64_t ii = 0; ii < N; ++ii) {
        int64_t row = upper ? (N - 1 - ii) : ii;
        T rhs = b[row];
        T diag = T(1);
        for (int64_t j = crow[row]; j < crow[row + 1]; ++j) {
            int64_t c = col[j];
            if (c == row) {
                diag = vals[j];
            } else if (upper ? (c > row) : (c < row)) {
                rhs -= vals[j] * x[c];
            }
        }
        x[row] = rhs / diag;
    }
}

auto sparse_trsv_standalone_hip(const Tensor& crow, const Tensor& col_idx,
    const Tensor& vals, const Tensor& b, int64_t N, bool upper,
    hipStream_t stream) -> Tensor
{
    auto x = tenzor::zeros({N}, vals.dtype(), vals.device());
    if (N == 0) return x;

    if (vals.dtype() == DType::Float32) {
        hipLaunchKernelGGL(sparse_trsv_sequential_kernel<float>, dim3(1), dim3(1), 0, stream,
            crow.data<int64_t>(), col_idx.data<int64_t>(), vals.data<float>(),
            b.data<float>(), x.data<float>(), N, upper);
    } else if (vals.dtype() == DType::Float64) {
        hipLaunchKernelGGL(sparse_trsv_sequential_kernel<double>, dim3(1), dim3(1), 0, stream,
            crow.data<int64_t>(), col_idx.data<int64_t>(), vals.data<double>(),
            b.data<double>(), x.data<double>(), N, upper);
    } else {
        throw std::runtime_error("sparse_trsv_standalone_hip: only Float32/Float64");
    }
    HIP_CHECK_SPARSE_SA(hipGetLastError());
    return x;
}

auto sparse_trsm_standalone_hip(const Tensor& crow, const Tensor& col_idx,
    const Tensor& vals, const Tensor& B, int64_t N, bool upper,
    hipStream_t stream) -> Tensor
{
    int64_t K = B.shape()[1];
    auto X = tenzor::zeros({N, K}, vals.dtype(), vals.device());
    for (int64_t k = 0; k < K; ++k) {
        // B is (N, K) row-major, so column k is a strided view. The trsv kernel
        // assumes contiguous b, so materialise the slice before solving.
        auto b_col = B.slice(1, k, k + 1).squeeze(1).contiguous();
        auto x_col = sparse_trsv_standalone_hip(crow, col_idx, vals, b_col, N, upper, stream);
        if (vals.dtype() == DType::Float32) {
            HIP_CHECK_SPARSE_SA(hipMemcpy2DAsync(
                X.data<float>() + k, K * sizeof(float),
                x_col.data<float>(), sizeof(float),
                sizeof(float), N, hipMemcpyDeviceToDevice, stream));
        } else {
            HIP_CHECK_SPARSE_SA(hipMemcpy2DAsync(
                X.data<double>() + k, K * sizeof(double),
                x_col.data<double>(), sizeof(double),
                sizeof(double), N, hipMemcpyDeviceToDevice, stream));
        }
    }
    HIP_CHECK_SPARSE_SA(hipStreamSynchronize(stream));
    return X;
}

} // namespace rocm
} // namespace tenzor
