/**
 * @file sparse.cpp
 * @brief OneAPI/SYCL sparse tensor kernels using oneMKL sparse BLAS
 *
 * Provides SpMV (sparse matrix-vector multiply), SpMM (sparse matrix-matrix
 * multiply), sparse-to-dense, dense-to-sparse, and sparse addition operations
 * for CSR format sparse tensors on Intel GPU/CPU via SYCL.
 */

#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/creation.hpp"
#include "../sycl_prefix_sum.hpp"
#include "../sycl_buffer_guard.hpp"
#include <sycl/sycl.hpp>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <vector>

#ifdef TENZOR_HAS_ONEMKL
#include <oneapi/mkl.hpp>
#include <oneapi/mkl/spblas.hpp>
#endif

namespace tenzor {
namespace oneapi {

// ============================================================================
// SpMV: y = A * x  (CSR sparse A, dense vector x)
// ============================================================================

auto spmv_kernel(const SparseTensor& A, const Tensor& x, sycl::queue& queue) -> Tensor {
    // SYCL-native CSR SpMV (one work-item per row).
    //
    // NOTE: The oneMKL sparse SYCL API (`set_csr_data`) takes 32-bit `int*`
    // index pointers, but Tenzor's `SparseTensor` contract mandates Int64
    // crow/col buffers. Converting would require an extra scratch allocation
    // per call and defeats the perf win. The native SYCL path below handles
    // Int64 end-to-end.
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spmv_kernel requires CSR format");
    }

    // Assert A.values dtype == x dtype before any dtype branch. Each branch
    // reinterprets x's bytes according to the sparse values dtype, so a
    // mismatch (e.g. a Float32 x with a BF16 sparse A) would silently misread
    // x rather than error. Mirrors the spmm_kernel guard below.
    if (A.values().dtype() != x.dtype()) {
        throw std::runtime_error(
            "oneapi spmv_kernel: sparse values dtype (" +
            std::string(dtype_name(A.values().dtype())) +
            ") must match dense x dtype (" +
            std::string(dtype_name(x.dtype())) + ")");
    }

    // E.3: native sycl::half / bfloat16 path. Each work-item loads FP16
    // values, casts to FP32 for accumulation (standard mixed-precision
    // pattern for sparse FP16 — FP16 lacks the dynamic range for row
    // sums), and writes back FP16. No tensor-level widen-narrow.
    if (A.values().dtype() == DType::Float16) {
        const auto& shape = A.shape();
        int64_t m = shape[0];
        Tensor y({m}, DType::Float16, A.values().device());
        // Defensive contiguity: the kernel reads crow/col/vals/x via flat raw
        // pointers, so a non-contiguous view would silently misread. Match CUDA.
        auto crow = A.crow_indices().contiguous();
        auto col = A.col_indices().contiguous();
        auto vals = A.values().contiguous();
        Tensor x_cont = x.is_contiguous() ? x : x.contiguous();
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr  = col.data<std::int64_t>();
        // Tenzor stores Float16 as `tenzor::Float16` (2-byte struct).
        // reinterpret to sycl::half* for the SYCL kernel — both are
        // identical 2-byte IEEE-754 binary16 representations.
        const auto* val_ptr_c = reinterpret_cast<const sycl::half*>(vals.data<tenzor::Float16>());
        const auto* x_ptr_c   = reinterpret_cast<const sycl::half*>(x_cont.data<tenzor::Float16>());
        auto*       y_ptr     = reinterpret_cast<sycl::half*>(y.data<tenzor::Float16>());
        const auto* val_ptr   = val_ptr_c;
        const auto* x_ptr     = x_ptr_c;
        queue.parallel_for(sycl::range<1>(static_cast<size_t>(m)),
            [=](sycl::id<1> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                float sum = 0.0f;
                for (std::int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += static_cast<float>(val_ptr[j]) *
                           static_cast<float>(x_ptr[col_ptr[j]]);
                }
                // L.2: write-back via the SYCL half ctor (round-to-nearest-even),
                // not static_cast<sycl::half> which is permitted to truncate on
                // PVC Gen1 and diverges from CUDA's __float2half_rn round-trip.
                // Saturate ±Inf → ±65504 (max finite half) and propagate NaN so
                // reductions of mixed magnitudes don't silently produce ±Inf.
                constexpr float kHalfMax = 65504.0f;
                float clamped = sycl::isnan(sum)
                                    ? sum
                                    : sycl::fmin(sycl::fmax(sum, -kHalfMax), kHalfMax);
                y_ptr[row] = sycl::half{clamped};
            }).wait();
        return y;
    }
    if (A.values().dtype() == DType::BFloat16) {
        // SYCL bfloat16 storage as uint16_t with bit_cast for arithmetic.
        const auto& shape = A.shape();
        int64_t m = shape[0];
        Tensor y({m}, DType::BFloat16, A.values().device());
        auto crow = A.crow_indices().contiguous();
        auto col = A.col_indices().contiguous();
        auto vals = A.values().contiguous();
        Tensor x_cont = x.is_contiguous() ? x : x.contiguous();
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr  = col.data<std::int64_t>();
        auto* val_ptr  = vals.data<uint16_t>();
        auto* x_ptr    = x_cont.data<uint16_t>();
        auto* y_ptr    = y.data<uint16_t>();
        queue.parallel_for(sycl::range<1>(static_cast<size_t>(m)),
            [=](sycl::id<1> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                float sum = 0.0f;
                auto bf_to_f32 = [](uint16_t bits) -> float {
                    uint32_t expanded = static_cast<uint32_t>(bits) << 16;
                    float result;
                    std::memcpy(&result, &expanded, sizeof(result));
                    return result;
                };
                auto f32_to_bf = [](float v) -> uint16_t {
                    uint32_t bits;
                    std::memcpy(&bits, &v, sizeof(bits));
                    // Round-to-nearest-even, IEEE convention.
                    uint32_t rounding_bias = 0x7FFF + ((bits >> 16) & 1u);
                    return static_cast<uint16_t>((bits + rounding_bias) >> 16);
                };
                for (std::int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += bf_to_f32(val_ptr[j]) * bf_to_f32(x_ptr[col_ptr[j]]);
                }
                y_ptr[row] = f32_to_bf(sum);
            }).wait();
        return y;
    }

    // No native complex SYCL sparse SpMV kernel (unlike the F16/BF16 paths
    // above, complex accumulation isn't a trivial widen-narrow). Relocate to
    // CPU — sparse::spmv already has complex<float>/complex<double>
    // instantiations (src/sparse/sparse_ops.cpp) — compute there, and move
    // the result back. Mirrors Vulkan's F104 CPU-offload-for-complex
    // convention for sparse ops (GLSL/SYCL device code has no native complex
    // arithmetic support, so this is the accepted pattern for GPU sparse
    // backends here, not a general GPU->CPU fallback).
    if (A.values().dtype() == DType::Complex64 || A.values().dtype() == DType::Complex128) {
        Device orig_dev = A.values().device();
        auto A_cpu = SparseTensor::sparse_csr(A.crow_indices().to(Device::cpu()),
                                              A.col_indices().to(Device::cpu()),
                                              A.values().to(Device::cpu()), A.shape());
        Tensor result = sparse::spmv(A_cpu, x.to(Device::cpu()));
        return result.to(orig_dev);
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];

    auto crow = A.crow_indices().contiguous();
    auto col = A.col_indices().contiguous();
    auto vals = A.values().contiguous();
    Tensor x_cont = x.is_contiguous() ? x : x.contiguous();

    Tensor y({m}, vals.dtype(), vals.device());

    if (vals.dtype() == DType::Float32) {
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<float>();
        auto* x_ptr = x_cont.data<float>();
        auto* y_ptr = y.data<float>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(m)),
            [=](sycl::id<1> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                float sum = 0.0f;
                for (std::int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * x_ptr[col_ptr[j]];
                }
                y_ptr[row] = sum;
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<double>();
        auto* x_ptr = x_cont.data<double>();
        auto* y_ptr = y.data<double>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(m)),
            [=](sycl::id<1> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                double sum = 0.0;
                for (std::int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * x_ptr[col_ptr[j]];
                }
                y_ptr[row] = sum;
            }).wait();
    } else {
        throw std::runtime_error("oneapi spmv_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return y;
}

// ============================================================================
// SpMM: C = A * B  (CSR sparse A, dense matrix B)
// ============================================================================

auto spmm_kernel(const SparseTensor& A, const Tensor& B, sycl::queue& queue) -> Tensor {
    // SYCL-native CSR SpMM (one work-item per output element).
    //
    // See spmv_kernel for why the oneMKL sparse SYCL API is not used here
    // (Int32 index pointer mismatch with Tenzor's Int64 SparseTensor API).
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spmm_kernel requires CSR format");
    }

    // Wave F5 (deferred → landed): F16/BF16 via widen-narrow through F32.
    if (A.values().dtype() == DType::Float16 || A.values().dtype() == DType::BFloat16) {
        DType orig = A.values().dtype();
        auto vals_f32 = A.values().to(DType::Float32);
        auto A_f32 = SparseTensor::sparse_csr(A.crow_indices(), A.col_indices(),
                                              vals_f32, A.shape());
        auto B_f32 = B.to(DType::Float32);
        auto C_f32 = spmm_kernel(A_f32, B_f32, queue);
        return C_f32.to(orig);
    }

    // No native complex SYCL sparse SpMM kernel; relocate to CPU (sparse::spmm
    // already has complex instantiations) and move the result back. See
    // spmv_kernel above for the full rationale (matches Vulkan's F104
    // CPU-offload-for-complex convention).
    if (A.values().dtype() == DType::Complex64 || A.values().dtype() == DType::Complex128) {
        Device orig_dev = A.values().device();
        auto A_cpu = SparseTensor::sparse_csr(A.crow_indices().to(Device::cpu()),
                                              A.col_indices().to(Device::cpu()),
                                              A.values().to(Device::cpu()), A.shape());
        Tensor result = sparse::spmm(A_cpu, B.to(Device::cpu()));
        return result.to(orig_dev);
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    // Validate B is a 2D matrix whose row count matches A's column count before
    // reading B.shape()[1]; otherwise b_ptr indexing reads out of bounds.
    if (B.ndim() != 2 || B.shape()[0] != shape[1]) {
        throw std::runtime_error(
            "oneapi spmm_kernel: dense B must be 2D with B.shape()[0] == A.shape()[1]");
    }
    int64_t n = B.shape()[1];

    // Defensive contiguity: the kernel reads crow/col/vals/B via flat raw
    // pointers, so a non-contiguous view would silently misread. Match CUDA.
    auto crow = A.crow_indices().contiguous();
    auto col = A.col_indices().contiguous();
    auto vals = A.values().contiguous();
    Tensor B_cont = B.is_contiguous() ? B : B.contiguous();

    // M7 fix: assert A.values dtype == B dtype before kernel dispatch.
    // The kernel reinterprets buffers via `data<float>()` / `data<double>()`
    // based on a single dtype branch, so a mismatch would silently
    // misread one of the operands.
    if (vals.dtype() != B.dtype()) {
        throw std::runtime_error(
            "oneapi spmm_kernel: sparse values dtype (" +
            std::string(dtype_name(vals.dtype())) +
            ") must match dense B dtype (" +
            std::string(dtype_name(B.dtype())) + ")");
    }
    Tensor C({m, n}, vals.dtype(), vals.device());

    if (vals.dtype() == DType::Float32) {
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<float>();
        auto* b_ptr = B_cont.data<float>();
        auto* c_ptr = C.data<float>();

        queue.parallel_for(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(n)),
            [=](sycl::id<2> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                int64_t c = static_cast<int64_t>(idx[1]);
                float sum = 0.0f;
                for (std::int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * b_ptr[col_ptr[j] * n + c];
                }
                c_ptr[row * n + c] = sum;
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<double>();
        auto* b_ptr = B_cont.data<double>();
        auto* c_ptr = C.data<double>();

        queue.parallel_for(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(n)),
            [=](sycl::id<2> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                int64_t c = static_cast<int64_t>(idx[1]);
                double sum = 0.0;
                for (std::int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * b_ptr[col_ptr[j] * n + c];
                }
                c_ptr[row * n + c] = sum;
            }).wait();
    } else {
        throw std::runtime_error("oneapi spmm_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return C;
}

// ============================================================================
// Sparse to Dense: convert CSR sparse tensor to dense tensor
// ============================================================================

auto sparse_to_dense_kernel(const SparseTensor& A, sycl::queue& queue) -> Tensor {
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_to_dense_kernel requires CSR format");
    }

    // E.3: F16/BF16 via widen-to-F32.
    if (A.values().dtype() == DType::Float16 ||
        A.values().dtype() == DType::BFloat16) {
        DType orig = A.values().dtype();
        auto vals_f32 = A.values().to(DType::Float32);
        auto A_f32 = SparseTensor::sparse_csr(A.crow_indices(), A.col_indices(),
                                              vals_f32, A.shape());
        return sparse_to_dense_kernel(A_f32, queue).to(orig);
    }

    // No native complex SYCL to-dense scatter kernel; relocate to CPU
    // (SparseTensor::to_dense() already handles complex generically) and
    // move the result back. See spmv_kernel above for the full rationale.
    if (A.values().dtype() == DType::Complex64 || A.values().dtype() == DType::Complex128) {
        Device orig_dev = A.values().device();
        auto A_cpu = SparseTensor::sparse_csr(A.crow_indices().to(Device::cpu()),
                                              A.col_indices().to(Device::cpu()),
                                              A.values().to(Device::cpu()), A.shape());
        return A_cpu.to_dense().to(orig_dev);
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t n = shape[1];

    auto crow = A.crow_indices().contiguous();
    auto col = A.col_indices().contiguous();
    auto vals = A.values().contiguous();
    int64_t nnz = A.nnz();

    Tensor dense({m, n}, vals.dtype(), vals.device());

    if (vals.dtype() == DType::Float32) {
        auto* dense_ptr = dense.data<float>();
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<float>();
        int64_t total = m * n;

        // Zero the output
        queue.memset(dense_ptr, 0, static_cast<size_t>(total) * sizeof(float)).wait();

        // Scatter nonzero values
        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                // Binary search for row
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= i) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = col_ptr[i];
                // Accumulate rather than overwrite: duplicate column entries
                // within a CSR row must sum (matches CPU/CUDA/ROCm's
                // SparseTensor::to_dense() scatter_add semantics). Different
                // work-items can race on the same (row,c) slot for such
                // duplicates, so the write must be atomic.
                sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>
                    atomic_val(dense_ptr[row * n + c]);
                atomic_val.fetch_add(val_ptr[i]);
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* dense_ptr = dense.data<double>();
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<double>();
        int64_t total = m * n;

        queue.memset(dense_ptr, 0, static_cast<size_t>(total) * sizeof(double)).wait();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= i) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = col_ptr[i];
                // See the Float32 branch above: accumulate, not overwrite,
                // since duplicate columns within a row must sum and this is
                // a genuine cross-work-item race otherwise.
                sycl::atomic_ref<double, sycl::memory_order::relaxed,
                                  sycl::memory_scope::device>
                    atomic_val(dense_ptr[row * n + c]);
                atomic_val.fetch_add(val_ptr[i]);
            }).wait();
    } else {
        throw std::runtime_error("oneapi sparse_to_dense_kernel: unsupported dtype");
    }

    return dense;
}

// ============================================================================
// Sparse Add: result = sparse + dense (output is dense)
// ============================================================================

auto sparse_add_kernel(const SparseTensor& A, const Tensor& B, sycl::queue& queue) -> Tensor {
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_add_kernel requires CSR format");
    }

    // E.3: F16/BF16 via widen-to-F32 (mirrors spmm/spmv).
    if (A.values().dtype() == DType::Float16 ||
        A.values().dtype() == DType::BFloat16) {
        DType orig = A.values().dtype();
        auto vals_f32 = A.values().to(DType::Float32);
        auto A_f32 = SparseTensor::sparse_csr(A.crow_indices(), A.col_indices(),
                                              vals_f32, A.shape());
        auto B_f32 = B.to(DType::Float32);
        return sparse_add_kernel(A_f32, B_f32, queue).to(orig);
    }

    // No native complex SYCL sparse-add kernel; relocate to CPU (sparse::add
    // already has complex instantiations) and move the result back. See
    // spmv_kernel above for the full rationale.
    if (A.values().dtype() == DType::Complex64 || A.values().dtype() == DType::Complex128) {
        Device orig_dev = A.values().device();
        auto A_cpu = SparseTensor::sparse_csr(A.crow_indices().to(Device::cpu()),
                                              A.col_indices().to(Device::cpu()),
                                              A.values().to(Device::cpu()), A.shape());
        Tensor result = sparse::add(A_cpu, B.to(Device::cpu()));
        return result.to(orig_dev);
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t n = shape[1];

    auto crow = A.crow_indices().contiguous();
    auto col = A.col_indices().contiguous();
    auto vals = A.values().contiguous();
    int64_t nnz = A.nnz();

    // Start with a contiguous copy of the dense tensor: out_ptr indexing below
    // is flat row-major, so a non-contiguous B must be materialized contiguous.
    Tensor result = B.is_contiguous() ? B.clone() : B.contiguous();

    if (vals.dtype() == DType::Float32) {
        auto* out_ptr = result.data<float>();
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<float>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                // Binary search for row
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= i) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = col_ptr[i];
                // Guard against malformed/untrusted CSR col indices: an
                // out-of-range column would be an out-of-bounds device write
                // into out_ptr. Mirrors ROCm's csr_sparse_add_kernel guard.
                if (c < 0 || c >= n) return;
                out_ptr[row * n + c] += val_ptr[i];
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* out_ptr = result.data<double>();
        auto* crow_ptr = crow.data<std::int64_t>();
        auto* col_ptr = col.data<std::int64_t>();
        auto* val_ptr = vals.data<double>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= i) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = col_ptr[i];
                // See the Float32 branch above: guard against an
                // out-of-range CSR column index before writing.
                if (c < 0 || c >= n) return;
                out_ptr[row * n + c] += val_ptr[i];
            }).wait();
    } else {
        throw std::runtime_error("oneapi sparse_add_kernel: unsupported dtype");
    }

    return result;
}

// ============================================================================
// SpGEMM: C = A * B  (CSR sparse A, CSR sparse B -> CSR sparse C)
// ============================================================================

auto spgemm_kernel(const SparseTensor& A, const SparseTensor& B,
                   sycl::queue& queue) -> SparseTensor {
    // E.3: F16/BF16 via widen-to-F32. Hoisted above the #if so both
    // historical-oneMKL and native-SYCL paths share the precision dispatch.
    if (A.values().dtype() == DType::Float16 ||
        A.values().dtype() == DType::BFloat16) {
        DType orig = A.values().dtype();
        auto a_vals_f32 = A.values().to(DType::Float32);
        auto A_f32 = SparseTensor::sparse_csr(A.crow_indices(), A.col_indices(),
                                              a_vals_f32, A.shape());
        auto b_vals_f32 = B.values().to(DType::Float32);
        auto B_f32 = SparseTensor::sparse_csr(B.crow_indices(), B.col_indices(),
                                              b_vals_f32, B.shape());
        auto C_f32 = spgemm_kernel(A_f32, B_f32, queue);
        auto C_vals_orig = C_f32.values().to(orig);
        return SparseTensor::sparse_csr(C_f32.crow_indices(),
                                        C_f32.col_indices(),
                                        C_vals_orig, C_f32.shape());
    }

    // No native complex SYCL SpGEMM kernel; relocate both operands to CPU
    // (sparse::spgemm already has complex instantiations) and move the
    // result's components back. See spmv_kernel above for the full
    // rationale.
    if (A.values().dtype() == DType::Complex64 || A.values().dtype() == DType::Complex128) {
        Device orig_dev = A.values().device();
        auto A_cpu = SparseTensor::sparse_csr(A.crow_indices().to(Device::cpu()),
                                              A.col_indices().to(Device::cpu()),
                                              A.values().to(Device::cpu()), A.shape());
        auto B_cpu = SparseTensor::sparse_csr(B.crow_indices().to(Device::cpu()),
                                              B.col_indices().to(Device::cpu()),
                                              B.values().to(Device::cpu()), B.shape());
        auto C_cpu = sparse::spgemm(A_cpu, B_cpu);
        return SparseTensor::sparse_csr(C_cpu.crow_indices().to(orig_dev),
                                        C_cpu.col_indices().to(orig_dev),
                                        C_cpu.values().to(orig_dev),
                                        C_cpu.shape());
    }
#if defined(TENZOR_HAS_ONEMKL) && 0
    // Historical oneMKL path: dense-intermediate approach using `spmm_kernel`
    // and `sparse_to_dense_kernel`. Disabled because those inner helpers use
    // Int32 CSR indices internally, while Tenzor's SparseTensor API requires
    // Int64. Converting in/out would defeat the perf win; we use the native
    // SYCL path below, which already works in Int64 end-to-end.
    if (A.layout() != SparseLayout::CSR || B.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spgemm_kernel requires CSR format for both inputs");
    }

    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t N = b_shape[1];

    if (K != b_shape[0]) {
        throw std::runtime_error("oneapi spgemm_kernel: inner dimensions must match");
    }
    if (A.dtype() != B.dtype()) {
        throw std::runtime_error("oneapi spgemm_kernel: dtype mismatch");
    }

    auto a_crow = A.crow_indices();
    auto a_col  = A.col_indices();
    auto a_vals = A.values();

    auto b_crow = B.crow_indices();
    auto b_col  = B.col_indices();
    auto b_vals = B.values();

    DType dtype = a_vals.dtype();

    // Step 1: Convert B to dense
    Tensor B_dense = sparse_to_dense_kernel(B, queue);

    // Step 2: Use SpMM (sparse A * dense B) to get dense C
    Tensor C_dense = spmm_kernel(A, B_dense, queue);

    // Step 3: Convert dense C back to sparse CSR (entirely on device)
    // 3-pass approach: count NNZ per row → prefix sum → compact nonzeros
    // Note: Tenzor's SparseTensor API uses int64 crow/col indices, so the
    // output tensors must be Int64 even though we use int32 scratch buffers.
    auto dense_to_csr_device = [&]<typename T>() -> SparseTensor {
        const T* d_data = C_dense.data<T>();
        Device dev = C_dense.device();

        // Pass 1: Count nonzeros per row
        std::int64_t* d_row_nnz = sycl::malloc_device<std::int64_t>(M, queue);
        queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
            int64_t row = idx[0];
            std::int64_t count = 0;
            for (int64_t j = 0; j < N; ++j) {
                if (d_data[row * N + j] != static_cast<T>(0)) {
                    count++;
                }
            }
            d_row_nnz[row] = count;
        }).wait();

        // Pass 2: Exclusive prefix sum → crow_indices (device-side)
        std::int64_t* d_crow = sycl::malloc_device<std::int64_t>(M + 1, queue);
        queue.memcpy(d_crow, d_row_nnz, M * sizeof(std::int64_t)).wait();
        queue.memset(d_crow + M, 0, sizeof(std::int64_t)).wait();

        int64_t nnz = sycl_exclusive_prefix_sum<std::int64_t>(d_crow, M + 1, queue);

        Tensor c_crow({M + 1}, DType::Int64, dev);
        Tensor c_col({nnz}, DType::Int64, dev);
        Tensor c_vals({nnz}, dtype, dev);

        queue.memcpy(c_crow.data<std::int64_t>(), d_crow,
                     static_cast<size_t>(M + 1) * sizeof(std::int64_t)).wait();

        if (nnz > 0) {
            // Pass 3: Compact nonzeros into col_indices and values arrays
            std::int64_t* out_col = c_col.data<std::int64_t>();
            T* out_vals = c_vals.data<T>();

            queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
                int64_t row = idx[0];
                std::int64_t write_pos = d_crow[row];
                for (int64_t j = 0; j < N; ++j) {
                    T v = d_data[row * N + j];
                    if (v != static_cast<T>(0)) {
                        out_col[write_pos] = static_cast<std::int64_t>(j);
                        out_vals[write_pos] = v;
                        write_pos++;
                    }
                }
            }).wait();
        }

        sycl::free(d_row_nnz, queue);
        sycl::free(d_crow, queue);
        return SparseTensor::sparse_csr(c_crow, c_col, c_vals, {M, N});
    };

    if (dtype == DType::Float32) {
        return dense_to_csr_device.template operator()<float>();
    } else if (dtype == DType::Float64) {
        return dense_to_csr_device.template operator()<double>();
    } else {
        throw std::runtime_error("oneapi spgemm_kernel: unsupported dtype (requires Float32 or Float64)");
    }
#else
    // ========================================================================
    // Standalone SYCL SpGEMM — no oneMKL dependency
    // 3-pass algorithm: count → prefix sum → fill with dedup → compact
    // ========================================================================
    if (A.layout() != SparseLayout::CSR || B.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spgemm_kernel requires CSR format for both inputs");
    }

    const auto& a_shape = A.shape();
    const auto& b_shape = B.shape();
    int64_t M = a_shape[0];
    int64_t K = a_shape[1];
    int64_t N = b_shape[1];

    if (K != b_shape[0]) {
        throw std::runtime_error("oneapi spgemm_kernel: inner dimensions must match");
    }
    if (A.dtype() != B.dtype()) {
        throw std::runtime_error("oneapi spgemm_kernel: dtype mismatch");
    }

    DType dtype = A.values().dtype();
    Device dev = A.values().device();

    // Validate the dtype BEFORE allocating any device memory. Previously the
    // unsupported-dtype error was thrown deep inside the compute lambda, after
    // five device buffers had been allocated, leaking all of them.
    if (dtype != DType::Float32 && dtype != DType::Float64) {
        throw std::runtime_error(
            "oneapi spgemm_kernel: only Float32/Float64 supported");
    }

    // Defensive contiguity: CSR components are read via flat raw pointers below.
    auto a_crow = A.crow_indices().contiguous();
    auto a_col  = A.col_indices().contiguous();
    auto a_vals = A.values().contiguous();
    auto b_crow = B.crow_indices().contiguous();
    auto b_col  = B.col_indices().contiguous();
    auto b_vals = B.values().contiguous();

    if (M == 0) {
        auto c_crow = tenzor::zeros({1}, DType::Int64, dev);
        auto c_col  = tenzor::empty({0}, DType::Int64, dev);
        auto c_vals = tenzor::empty({0}, dtype, dev);
        return SparseTensor::sparse_csr(c_crow, c_col, c_vals, {M, N});
    }

    // Pass 1: Count nnz upper bound per row. All device scratch buffers below
    // use RAII (SyclDeviceBuffer) so they are freed on every exit path,
    // including async-exception throws from .wait().
    SyclDeviceBuffer<int64_t> d_row_nnz_buf(M, queue);
    int64_t* d_row_nnz = d_row_nnz_buf.get();

    const int64_t* ac = a_crow.data<int64_t>();
    const int64_t* acol = a_col.data<int64_t>();
    const int64_t* bc = b_crow.data<int64_t>();

    queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
        int64_t row = idx[0];
        int64_t count = 0;
        int64_t a_start = ac[row];
        int64_t a_end = ac[row + 1];
        for (int64_t ja = a_start; ja < a_end; ++ja) {
            int64_t k = acol[ja];
            count += bc[k + 1] - bc[k];
        }
        d_row_nnz[row] = count;
    }).wait();

    // Pass 2: Exclusive prefix sum → crow upper bound (device-side)
    // crow[0] = 0, crow[i+1] = crow[i] + row_nnz[i]
    // We build this by placing row_nnz into crow[0..M-1], running exclusive scan
    // on M+1 elements with crow[M] temporarily set to 0. The exclusive scan of
    // [nnz0, nnz1, ..., nnzM-1, 0] gives [0, nnz0, nnz0+nnz1, ..., total, total]
    // but that's M+1 entries which is exactly what we want (last entry = total).
    //
    // Simpler: copy row_nnz into d_crow_ub[0..M-1], set d_crow_ub[M]=0, run
    // exclusive scan on all M+1 elements. Result: [0, nnz0, nnz0+nnz1, ..., total].
    SyclDeviceBuffer<int64_t> d_crow_ub_buf(M + 1, queue);
    int64_t* d_crow_ub = d_crow_ub_buf.get();
    queue.memcpy(d_crow_ub, d_row_nnz, M * sizeof(int64_t)).wait();
    queue.memset(d_crow_ub + M, 0, sizeof(int64_t)).wait();  // d_crow_ub[M] = 0

    // Device-side exclusive prefix sum; returns total as a single scalar D2H (acceptable)
    int64_t total_nnz_ub = sycl_exclusive_prefix_sum<int64_t>(d_crow_ub, M + 1, queue);

    if (total_nnz_ub == 0) {
        auto c_crow = tenzor::zeros({M + 1}, DType::Int64, dev);
        auto c_col  = tenzor::empty({0}, DType::Int64, dev);
        auto c_vals = tenzor::empty({0}, dtype, dev);
        return SparseTensor::sparse_csr(c_crow, c_col, c_vals, {M, N});
    }

    // Allocate upper-bound output
    SyclDeviceBuffer<int64_t> d_col_ub_buf(total_nnz_ub, queue);
    int64_t* d_col_ub = d_col_ub_buf.get();
    SyclDeviceBuffer<int64_t> d_actual_nnz_buf(M, queue);
    int64_t* d_actual_nnz = d_actual_nnz_buf.get();

    // Pass 3 + Compact: Fill with dedup (templated by dtype).
    //
    // Dedup is performed entirely within each row's own slice of the upper-bound
    // output buffers (d_col_ub / d_vals_ub), which the prefix-sum sized to that
    // row's product-count upper bound. Each work-item owns one row, so it owns a
    // disjoint, contiguous slice [c_start, c_start + row_nnz_ub) — no atomics and
    // no cross-row sharing.
    //
    // Within a row the written columns are kept sorted ascending. For each new
    // product (col, val) we binary-search the already-written prefix:
    //   - hit  -> accumulate into the existing slot (O(log R));
    //   - miss -> insert at the sorted position, shifting the tail right (O(R)).
    // R is the row's sparse product count, not N, so the cost is proportional to
    // the actual sparsity rather than the dense M*N marker the old code used
    // (that marker requested O(M*N) global memory and a full M*N memset, OOMing
    // for large sparse operands and defeating the point of a sparse format).
    //
    // Sorting the columns as we go also makes the SYCL output column-ordered,
    // matching the CPU reference (cpu_spgemm_typed sorts each row by column).
    auto run_fill_and_compact = [&]<typename T>() {
        SyclDeviceBuffer<T> d_vals_ub_buf(total_nnz_ub, queue);
        T* d_vals_ub = d_vals_ub_buf.get();
        const T* av = a_vals.data<T>();
        const T* bv = b_vals.data<T>();
        const int64_t* bcol = b_col.data<int64_t>();

        queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
            int64_t row = idx[0];
            int64_t c_start = d_crow_ub[row];
            int64_t a_start = ac[row];
            int64_t a_end = ac[row + 1];
            // Number of distinct columns written so far for this row; the slots
            // [c_start, c_start + len) stay sorted by column at all times.
            int64_t len = 0;

            for (int64_t ja = a_start; ja < a_end; ++ja) {
                int64_t k = acol[ja];
                T a_val = av[ja];
                int64_t b_start = bc[k];
                int64_t b_end = bc[k + 1];

                for (int64_t jb = b_start; jb < b_end; ++jb) {
                    int64_t col = bcol[jb];
                    T val = a_val * bv[jb];

                    // Binary search for `col` in the sorted prefix
                    // d_col_ub[c_start .. c_start+len). `lo` ends as the
                    // insertion point (lower_bound).
                    int64_t lo = 0;
                    int64_t hi = len;
                    while (lo < hi) {
                        int64_t mid = lo + (hi - lo) / 2;
                        int64_t mid_col = d_col_ub[c_start + mid];
                        if (mid_col < col) {
                            lo = mid + 1;
                        } else {
                            hi = mid;
                        }
                    }

                    if (lo < len && d_col_ub[c_start + lo] == col) {
                        // Existing column: accumulate.
                        d_vals_ub[c_start + lo] += val;
                    } else {
                        // New column: shift the sorted tail right by one to
                        // open a slot at `lo`, then insert.
                        for (int64_t s = len; s > lo; --s) {
                            d_col_ub[c_start + s]  = d_col_ub[c_start + s - 1];
                            d_vals_ub[c_start + s] = d_vals_ub[c_start + s - 1];
                        }
                        d_col_ub[c_start + lo]  = col;
                        d_vals_ub[c_start + lo] = val;
                        ++len;
                    }
                }
            }
            d_actual_nnz[row] = len;
        }).wait();

        // Compact: device-side prefix sum on actual nnz → final crow indices
        SyclDeviceBuffer<int64_t> d_crow_final_buf(M + 1, queue);
        int64_t* d_crow_final = d_crow_final_buf.get();
        queue.memcpy(d_crow_final, d_actual_nnz, M * sizeof(int64_t)).wait();
        queue.memset(d_crow_final + M, 0, sizeof(int64_t)).wait();

        // Device-side exclusive prefix sum; single scalar D2H for total (acceptable)
        int64_t total_nnz = sycl_exclusive_prefix_sum<int64_t>(d_crow_final, M + 1, queue);

        auto c_crow_t = Tensor({M + 1}, DType::Int64, dev);
        auto c_col_t  = Tensor({total_nnz}, DType::Int64, dev);
        auto c_vals_t = Tensor({total_nnz}, dtype, dev);

        queue.memcpy(c_crow_t.data<int64_t>(), d_crow_final,
                     (M + 1) * sizeof(int64_t)).wait();

        if (total_nnz > 0) {
            int64_t* out_col = c_col_t.data<int64_t>();
            T* out_vals = c_vals_t.data<T>();

            queue.parallel_for(sycl::range<1>(M), [=](sycl::id<1> idx) {
                int64_t row = idx[0];
                int64_t src = d_crow_ub[row];
                int64_t dst = d_crow_final[row];
                int64_t count = d_actual_nnz[row];
                for (int64_t i = 0; i < count; ++i) {
                    out_col[dst + i] = d_col_ub[src + i];
                    out_vals[dst + i] = d_vals_ub[src + i];
                }
            }).wait();
        }

        return SparseTensor::sparse_csr(c_crow_t, c_col_t, c_vals_t, {M, N});
    };

    SparseTensor result = [&]() {
        if (dtype == DType::Float32) {
            return run_fill_and_compact.template operator()<float>();
        } else if (dtype == DType::Float64) {
            return run_fill_and_compact.template operator()<double>();
        } else {
            // Unreachable: dtype validated at function entry.
            throw std::runtime_error("oneapi spgemm_kernel: only Float32/Float64 supported");
        }
    }();

    // Scratch buffers (d_row_nnz, d_crow_ub, d_col_ub, d_actual_nnz) are freed
    // by their SyclDeviceBuffer guards on scope exit. The dedup no longer needs
    // a dense per-(row,col) marker, so there is no O(M*N) scratch to free.
    return result;
#endif
}

// ============================================================================
// SparseTrsv: solve L*x = b  (or U*x = b) for a single RHS vector
// ============================================================================

auto sparse_trsv_kernel(const SparseTensor& L, const Tensor& b, bool upper,
                        sycl::queue& queue) -> Tensor {
    // E.3: F16/BF16 via widen-to-F32.
    if (L.values().dtype() == DType::Float16 ||
        L.values().dtype() == DType::BFloat16) {
        DType orig = L.values().dtype();
        auto vals_f32 = L.values().to(DType::Float32);
        auto L_f32 = SparseTensor::sparse_csr(L.crow_indices(), L.col_indices(),
                                              vals_f32, L.shape());
        auto b_f32 = b.to(DType::Float32);
        return sparse_trsv_kernel(L_f32, b_f32, upper, queue).to(orig);
    }

    // No native complex SYCL triangular-solve kernel; relocate to CPU
    // (sparse::sparse_triangular_solve already has complex instantiations)
    // and move the result back. See spmv_kernel above for the full
    // rationale — applied here so OneAPI's sparse Trsv is consistent with
    // its own SpMV/SpMM/ToDense/Add/SpGEMM complex handling above.
    if (L.values().dtype() == DType::Complex64 || L.values().dtype() == DType::Complex128) {
        Device orig_dev = L.values().device();
        auto L_cpu = SparseTensor::sparse_csr(L.crow_indices().to(Device::cpu()),
                                              L.col_indices().to(Device::cpu()),
                                              L.values().to(Device::cpu()), L.shape());
        Tensor result = sparse::sparse_triangular_solve(L_cpu, b.to(Device::cpu()), upper);
        return result.to(orig_dev);
    }
#if defined(TENZOR_HAS_ONEMKL) && 0
    // Historical oneMKL path: disabled because oneMKL's sparse::trsv here
    // is wired for Int32 CSR indices, while Tenzor's SparseTensor API uses
    // Int64. The native SYCL `#else` path handles Int64 end-to-end.
    if (L.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_trsv_kernel requires CSR format");
    }

    const auto& shape = L.shape();
    int64_t N = shape[0];
    if (shape[1] != N) {
        throw std::runtime_error("oneapi sparse_trsv_kernel: L must be square");
    }
    if (b.shape()[0] != N || b.ndim() != 1) {
        throw std::runtime_error("oneapi sparse_trsv_kernel: b must be 1D with length N");
    }

    auto crow = L.crow_indices();
    auto col  = L.col_indices();
    auto vals = L.values();
    DType dtype = vals.dtype();

    Tensor x({N}, dtype, vals.device());

    ::oneapi::mkl::sparse::matrix_handle_t handle = nullptr;

    auto uplo = upper ? ::oneapi::mkl::uplo::upper : ::oneapi::mkl::uplo::lower;

    const auto nnz = static_cast<std::int64_t>(L.nnz());

    if (dtype == DType::Float32) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(N),
            static_cast<std::int64_t>(N),
            nnz,
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<float>()).wait();

        ::oneapi::mkl::sparse::trsv(
            queue, uplo,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::diag::nonunit,
            1.0f, handle,
            b.data<float>(),
            x.data<float>()).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else if (dtype == DType::Float64) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(N),
            static_cast<std::int64_t>(N),
            nnz,
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<double>()).wait();

        ::oneapi::mkl::sparse::trsv(
            queue, uplo,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::diag::nonunit,
            1.0, handle,
            b.data<double>(),
            x.data<double>()).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else {
        throw std::runtime_error("oneapi sparse_trsv_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return x;
#else
    // ========================================================================
    // Standalone SYCL SparseTrsv — single-work-item sequential solve.
    //
    // A device-side spin-wait on a per-row "solved" flag deadlocks under SIMT
    // lockstep: work-items in the same sub-group execute in lockstep, so a
    // later row that spins waiting on an earlier row in the SAME sub-group can
    // block the very work-item that would set the flag, and neither makes
    // progress. We therefore run the entire triangular solve in a single
    // work-item that processes rows in strict dependency order
    // (forward substitution for lower, backward substitution for upper).
    // This trades parallelism for guaranteed-correct, deadlock-free behavior,
    // matching the established pattern used by the ROCm backend.
    // ========================================================================
    if (L.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_trsv_kernel requires CSR format");
    }

    const auto& shape = L.shape();
    int64_t N = shape[0];
    if (shape[1] != N) {
        throw std::runtime_error("oneapi sparse_trsv_kernel: L must be square");
    }
    if (b.shape()[0] != N || b.ndim() != 1) {
        throw std::runtime_error("oneapi sparse_trsv_kernel: b must be 1D with length N");
    }

    // Defensive contiguity: CSR components and b are read via flat raw pointers.
    auto crow = L.crow_indices().contiguous();
    auto col  = L.col_indices().contiguous();
    auto vals = L.values().contiguous();
    Tensor b_cont = b.is_contiguous() ? b : b.contiguous();
    DType dtype = vals.dtype();

    auto x = tenzor::zeros({N}, dtype, vals.device());

    auto run_trsv = [&]<typename T>() {
        const int64_t* cr = crow.data<int64_t>();
        const int64_t* cl = col.data<int64_t>();
        const T* v = vals.data<T>();
        const T* b_ptr = b_cont.data<T>();
        T* x_ptr = x.data<T>();
        bool is_upper = upper;
        int64_t n = N;

        // Single work-item: rows are visited in dependency order so every
        // x[c] referenced below has already been written. No cross-work-item
        // synchronization, hence no possibility of an SIMT spin-wait deadlock.
        queue.single_task([=]() {
            for (int64_t r = 0; r < n; ++r) {
                // Lower: forward substitution (rows 0..n-1).
                // Upper: backward substitution (rows n-1..0).
                int64_t row = is_upper ? (n - 1 - r) : r;

                int64_t row_start = cr[row];
                int64_t row_end = cr[row + 1];

                // x[row] = (b[row] - sum(L[row,c]*x[c])) / L[row,row]
                // Missing diagonal entry is treated as unit diagonal.
                T rhs = b_ptr[row];
                T diag = T(1);
                for (int64_t j = row_start; j < row_end; ++j) {
                    int64_t c = cl[j];
                    if (c == row) {
                        diag = v[j];
                    } else if (is_upper ? (c > row) : (c < row)) {
                        rhs -= v[j] * x_ptr[c];
                    }
                }
                x_ptr[row] = rhs / diag;
            }
        }).wait();
    };

    if (dtype == DType::Float32) {
        run_trsv.template operator()<float>();
    } else if (dtype == DType::Float64) {
        run_trsv.template operator()<double>();
    } else {
        throw std::runtime_error("oneapi sparse_trsv_kernel: only Float32/Float64 supported");
    }

    return x;
#endif
}

// ============================================================================
// SparseTrsm: solve L*X = B  (or U*X = B) for multiple RHS columns
// ============================================================================

auto sparse_trsm_kernel(const SparseTensor& L, const Tensor& B, bool upper,
                        sycl::queue& queue) -> Tensor {
    // E.3: F16/BF16 via widen-to-F32.
    if (L.values().dtype() == DType::Float16 ||
        L.values().dtype() == DType::BFloat16) {
        DType orig = L.values().dtype();
        auto vals_f32 = L.values().to(DType::Float32);
        auto L_f32 = SparseTensor::sparse_csr(L.crow_indices(), L.col_indices(),
                                              vals_f32, L.shape());
        auto B_f32 = B.to(DType::Float32);
        return sparse_trsm_kernel(L_f32, B_f32, upper, queue).to(orig);
    }

    // No native complex SYCL triangular-solve kernel; relocate to CPU and
    // move the result back. sparse::sparse_triangular_solve handles the 2D
    // (multi-RHS / trsm) case via b.ndim()==2 internally. See
    // sparse_trsv_kernel above for the full rationale.
    if (L.values().dtype() == DType::Complex64 || L.values().dtype() == DType::Complex128) {
        Device orig_dev = L.values().device();
        auto L_cpu = SparseTensor::sparse_csr(L.crow_indices().to(Device::cpu()),
                                              L.col_indices().to(Device::cpu()),
                                              L.values().to(Device::cpu()), L.shape());
        Tensor result = sparse::sparse_triangular_solve(L_cpu, B.to(Device::cpu()), upper);
        return result.to(orig_dev);
    }
#if defined(TENZOR_HAS_ONEMKL) && 0
    // Historical oneMKL path: disabled because oneMKL's sparse::trsm here
    // is wired for Int32 CSR indices, while Tenzor's SparseTensor API uses
    // Int64. The native SYCL `#else` path handles Int64 end-to-end.
    if (L.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_trsm_kernel requires CSR format");
    }

    const auto& shape = L.shape();
    int64_t N = shape[0];
    if (shape[1] != N) {
        throw std::runtime_error("oneapi sparse_trsm_kernel: L must be square");
    }
    if (B.ndim() != 2 || B.shape()[0] != N) {
        throw std::runtime_error("oneapi sparse_trsm_kernel: B must be 2D with first dim N");
    }

    int64_t K = B.shape()[1];  // number of RHS columns

    auto crow = L.crow_indices();
    auto col  = L.col_indices();
    auto vals = L.values();
    DType dtype = vals.dtype();

    Tensor X({N, K}, dtype, vals.device());

    ::oneapi::mkl::sparse::matrix_handle_t handle = nullptr;

    auto uplo = upper ? ::oneapi::mkl::uplo::upper : ::oneapi::mkl::uplo::lower;

    const auto nnz = static_cast<std::int64_t>(L.nnz());

    if (dtype == DType::Float32) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(N),
            static_cast<std::int64_t>(N),
            nnz,
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<float>()).wait();

        ::oneapi::mkl::sparse::trsm(
            queue, ::oneapi::mkl::layout::row_major,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::transpose::nontrans,
            uplo,
            ::oneapi::mkl::diag::nonunit,
            1.0f, handle,
            B.data<float>(),
            static_cast<std::int64_t>(K),   // columns in B
            static_cast<std::int64_t>(K),   // ldx (row-major leading dim of B)
            X.data<float>(),
            static_cast<std::int64_t>(K)).wait();  // ldy

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else if (dtype == DType::Float64) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(N),
            static_cast<std::int64_t>(N),
            nnz,
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<double>()).wait();

        ::oneapi::mkl::sparse::trsm(
            queue, ::oneapi::mkl::layout::row_major,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::transpose::nontrans,
            uplo,
            ::oneapi::mkl::diag::nonunit,
            1.0, handle,
            B.data<double>(),
            static_cast<std::int64_t>(K),
            static_cast<std::int64_t>(K),
            X.data<double>(),
            static_cast<std::int64_t>(K)).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else {
        throw std::runtime_error("oneapi sparse_trsm_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return X;
#else
    // ========================================================================
    // Standalone SYCL SparseTrsm — solve column-by-column via Trsv
    // ========================================================================
    if (L.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_trsm_kernel requires CSR format");
    }

    const auto& shape = L.shape();
    int64_t N = shape[0];
    if (shape[1] != N) {
        throw std::runtime_error("oneapi sparse_trsm_kernel: L must be square");
    }
    if (B.ndim() != 2 || B.shape()[0] != N) {
        throw std::runtime_error("oneapi sparse_trsm_kernel: B must be 2D with first dim N");
    }

    int64_t K = B.shape()[1];
    DType dtype = L.values().dtype();

    auto X = tenzor::zeros({N, K}, dtype, L.values().device());

    // Solve column-by-column. The trsv kernel reads `b.data<T>()` with
    // contiguous indexing (b_ptr[row]), so a strided slice view of a
    // {N,K} row-major B would be read at wrong offsets. Materialize each
    // column as a contiguous 1D tensor.
    for (int64_t k = 0; k < K; ++k) {
        auto b_col = B.slice(1, k, k + 1).squeeze(1).contiguous();
        auto x_col = sparse_trsv_kernel(L, b_col, upper, queue);

        // Copy x_col into X[:, k]
        if (dtype == DType::Float32) {
            const float* src = x_col.data<float>();
            float* dst = X.data<float>();
            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i[0] * K + k] = src[i[0]];
            }).wait();
        } else if (dtype == DType::Float64) {
            const double* src = x_col.data<double>();
            double* dst = X.data<double>();
            queue.parallel_for(sycl::range<1>(N), [=](sycl::id<1> i) {
                dst[i[0] * K + k] = src[i[0]];
            }).wait();
        }
    }

    return X;
#endif
}

// ============================================================================
// Standalone SYCL sparse dispatch wrappers (for kernel registry)
// These provide the same interface as the CUDA/ROCm standalone functions.
// ============================================================================

auto spgemm_standalone_sycl(std::span<const Tensor> inputs, const OpAttributes& attrs,
                            sycl::queue& queue) -> std::vector<Tensor> {
    int64_t M = attrs.get_int(AttrKey::M);
    int64_t K = attrs.get_int(AttrKey::K);
    int64_t N = attrs.get_int(AttrKey::N);

    auto a = SparseTensor::sparse_csr(inputs[0], inputs[1], inputs[2], {M, K});
    auto b = SparseTensor::sparse_csr(inputs[3], inputs[4], inputs[5], {K, N});
    auto c = spgemm_kernel(a, b, queue);
    return {c.crow_indices(), c.col_indices(), c.values()};
}

auto sparse_trsv_standalone_sycl(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                 const Tensor& b, int64_t N, bool upper,
                                 sycl::queue& queue) -> Tensor {
    auto L = SparseTensor::sparse_csr(crow, col_idx, vals, {N, N});
    return sparse_trsv_kernel(L, b, upper, queue);
}

auto sparse_trsm_standalone_sycl(const Tensor& crow, const Tensor& col_idx, const Tensor& vals,
                                 const Tensor& B, int64_t N, bool upper,
                                 sycl::queue& queue) -> Tensor {
    auto L = SparseTensor::sparse_csr(crow, col_idx, vals, {N, N});
    return sparse_trsm_kernel(L, B, upper, queue);
}

} // namespace oneapi
} // namespace tenzor
