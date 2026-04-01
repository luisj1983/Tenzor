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
#include <sycl/sycl.hpp>
#include <cstdint>
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
#ifdef TENZOR_HAS_ONEMKL
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spmv_kernel requires CSR format");
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t k = shape[1];

    auto crow = A.crow_indices();
    auto col = A.col_indices();
    auto vals = A.values();

    Tensor y({m}, vals.dtype(), vals.device());

    ::oneapi::mkl::sparse::matrix_handle_t handle = nullptr;

    if (vals.dtype() == DType::Float32) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(m),
            static_cast<std::int64_t>(k),
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<float>()).wait();

        ::oneapi::mkl::sparse::gemv(
            queue, ::oneapi::mkl::transpose::nontrans,
            1.0f, handle,
            x.data<float>(),
            0.0f,
            y.data<float>()).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else if (vals.dtype() == DType::Float64) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(m),
            static_cast<std::int64_t>(k),
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<double>()).wait();

        ::oneapi::mkl::sparse::gemv(
            queue, ::oneapi::mkl::transpose::nontrans,
            1.0, handle,
            x.data<double>(),
            0.0,
            y.data<double>()).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else {
        throw std::runtime_error("oneapi spmv_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return y;
#else
    // SYCL-native CSR SpMV fallback (one work-item per row)
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spmv_kernel requires CSR format");
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];

    auto crow = A.crow_indices();
    auto col = A.col_indices();
    auto vals = A.values();

    Tensor y({m}, vals.dtype(), vals.device());

    if (vals.dtype() == DType::Float32) {
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<float>();
        auto* x_ptr = x.data<float>();
        auto* y_ptr = y.data<float>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(m)),
            [=](sycl::id<1> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                float sum = 0.0f;
                for (std::int32_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * x_ptr[col_ptr[j]];
                }
                y_ptr[row] = sum;
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<double>();
        auto* x_ptr = x.data<double>();
        auto* y_ptr = y.data<double>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(m)),
            [=](sycl::id<1> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                double sum = 0.0;
                for (std::int32_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * x_ptr[col_ptr[j]];
                }
                y_ptr[row] = sum;
            }).wait();
    } else {
        throw std::runtime_error("oneapi spmv_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return y;
#endif
}

// ============================================================================
// SpMM: C = A * B  (CSR sparse A, dense matrix B)
// ============================================================================

auto spmm_kernel(const SparseTensor& A, const Tensor& B, sycl::queue& queue) -> Tensor {
#ifdef TENZOR_HAS_ONEMKL
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spmm_kernel requires CSR format");
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t k = shape[1];
    int64_t n = B.shape()[1];

    auto crow = A.crow_indices();
    auto col = A.col_indices();
    auto vals = A.values();

    Tensor C({m, n}, B.dtype(), vals.device());

    ::oneapi::mkl::sparse::matrix_handle_t handle = nullptr;

    if (vals.dtype() == DType::Float32) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(m),
            static_cast<std::int64_t>(k),
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<float>()).wait();

        ::oneapi::mkl::sparse::gemm(
            queue, ::oneapi::mkl::layout::row_major,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::transpose::nontrans,
            1.0f, handle,
            B.data<float>(), n,
            static_cast<std::int64_t>(n),
            0.0f,
            C.data<float>(),
            static_cast<std::int64_t>(n)).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else if (vals.dtype() == DType::Float64) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(m),
            static_cast<std::int64_t>(k),
            ::oneapi::mkl::index_base::zero,
            crow.data<std::int32_t>(),
            col.data<std::int32_t>(),
            vals.data<double>()).wait();

        ::oneapi::mkl::sparse::gemm(
            queue, ::oneapi::mkl::layout::row_major,
            ::oneapi::mkl::transpose::nontrans,
            ::oneapi::mkl::transpose::nontrans,
            1.0, handle,
            B.data<double>(), n,
            static_cast<std::int64_t>(n),
            0.0,
            C.data<double>(),
            static_cast<std::int64_t>(n)).wait();

        ::oneapi::mkl::sparse::release_matrix_handle(queue, &handle).wait();
    } else {
        throw std::runtime_error("oneapi spmm_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return C;
#else
    // SYCL-native CSR SpMM fallback (one work-item per output element)
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi spmm_kernel requires CSR format");
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t k = shape[1];
    int64_t n = B.shape()[1];

    auto crow = A.crow_indices();
    auto col = A.col_indices();
    auto vals = A.values();

    Tensor C({m, n}, B.dtype(), vals.device());

    if (vals.dtype() == DType::Float32) {
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<float>();
        auto* b_ptr = B.data<float>();
        auto* c_ptr = C.data<float>();

        queue.parallel_for(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(n)),
            [=](sycl::id<2> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                int64_t c = static_cast<int64_t>(idx[1]);
                float sum = 0.0f;
                for (std::int32_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * b_ptr[col_ptr[j] * n + c];
                }
                c_ptr[row * n + c] = sum;
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<double>();
        auto* b_ptr = B.data<double>();
        auto* c_ptr = C.data<double>();

        queue.parallel_for(sycl::range<2>(static_cast<size_t>(m), static_cast<size_t>(n)),
            [=](sycl::id<2> idx) {
                int64_t row = static_cast<int64_t>(idx[0]);
                int64_t c = static_cast<int64_t>(idx[1]);
                double sum = 0.0;
                for (std::int32_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += val_ptr[j] * b_ptr[col_ptr[j] * n + c];
                }
                c_ptr[row * n + c] = sum;
            }).wait();
    } else {
        throw std::runtime_error("oneapi spmm_kernel: unsupported dtype (requires Float32 or Float64)");
    }

    return C;
#endif
}

// ============================================================================
// Sparse to Dense: convert CSR sparse tensor to dense tensor
// ============================================================================

auto sparse_to_dense_kernel(const SparseTensor& A, sycl::queue& queue) -> Tensor {
    if (A.layout() != SparseLayout::CSR) {
        throw std::runtime_error("oneapi sparse_to_dense_kernel requires CSR format");
    }

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t n = shape[1];

    auto crow = A.crow_indices();
    auto col = A.col_indices();
    auto vals = A.values();
    int64_t nnz = A.nnz();

    Tensor dense({m, n}, vals.dtype(), vals.device());

    if (vals.dtype() == DType::Float32) {
        auto* dense_ptr = dense.data<float>();
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
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
                    if (crow_ptr[mid + 1] <= static_cast<std::int32_t>(i)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = static_cast<int64_t>(col_ptr[i]);
                dense_ptr[row * n + c] = val_ptr[i];
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* dense_ptr = dense.data<double>();
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<double>();
        int64_t total = m * n;

        queue.memset(dense_ptr, 0, static_cast<size_t>(total) * sizeof(double)).wait();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= static_cast<std::int32_t>(i)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = static_cast<int64_t>(col_ptr[i]);
                dense_ptr[row * n + c] = val_ptr[i];
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

    const auto& shape = A.shape();
    int64_t m = shape[0];
    int64_t n = shape[1];

    auto crow = A.crow_indices();
    auto col = A.col_indices();
    auto vals = A.values();
    int64_t nnz = A.nnz();

    // Start with a copy of the dense tensor
    Tensor result = B.clone();

    if (vals.dtype() == DType::Float32) {
        auto* out_ptr = result.data<float>();
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<float>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                // Binary search for row
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= static_cast<std::int32_t>(i)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = static_cast<int64_t>(col_ptr[i]);
                out_ptr[row * n + c] += val_ptr[i];
            }).wait();
    } else if (vals.dtype() == DType::Float64) {
        auto* out_ptr = result.data<double>();
        auto* crow_ptr = crow.data<std::int32_t>();
        auto* col_ptr = col.data<std::int32_t>();
        auto* val_ptr = vals.data<double>();

        queue.parallel_for(sycl::range<1>(static_cast<size_t>(nnz)),
            [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t lo = 0, hi = m;
                while (lo < hi) {
                    int64_t mid = (lo + hi) / 2;
                    if (crow_ptr[mid + 1] <= static_cast<std::int32_t>(i)) {
                        lo = mid + 1;
                    } else {
                        hi = mid;
                    }
                }
                int64_t row = lo;
                int64_t c = static_cast<int64_t>(col_ptr[i]);
                out_ptr[row * n + c] += val_ptr[i];
            }).wait();
    } else {
        throw std::runtime_error("oneapi sparse_add_kernel: unsupported dtype");
    }

    return result;
}

} // namespace oneapi
} // namespace tenzor
