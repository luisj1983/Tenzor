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

// ============================================================================
// SpGEMM: C = A * B  (CSR sparse A, CSR sparse B -> CSR sparse C)
// ============================================================================

auto spgemm_kernel(const SparseTensor& A, const SparseTensor& B,
                   sycl::queue& queue) -> SparseTensor {
#ifdef TENZOR_HAS_ONEMKL
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

    // oneMKL sparse::gemm for sparse-sparse multiply requires setting up
    // two matrix handles (A, B) and calling gemm with a dense output.
    // The oneMKL sparse::gemm signature operates on sparse A * dense B,
    // so for true SpGEMM we convert B to dense, compute, then sparsify.
    // This is the standard approach when the vendor library doesn't expose
    // a native sparse-sparse multiply with sparse output.
    //
    // Step 1: Convert B to dense
    Tensor B_dense = sparse_to_dense_kernel(B, queue);

    // Step 2: Use SpMM (sparse A * dense B) to get dense C
    Tensor C_dense = spmm_kernel(A, B_dense, queue);

    // Step 3: Convert dense C back to sparse CSR
    // Build CSR from dense: scan for nonzeros
    if (dtype == DType::Float32) {
        // Transfer to host for CSR construction
        std::vector<float> host_data(static_cast<size_t>(M * N));
        queue.memcpy(host_data.data(), C_dense.data<float>(),
                     static_cast<size_t>(M * N) * sizeof(float)).wait();

        std::vector<std::int32_t> crow(static_cast<size_t>(M + 1), 0);
        std::vector<std::int32_t> cols;
        std::vector<float> vals;

        for (int64_t i = 0; i < M; ++i) {
            crow[static_cast<size_t>(i + 1)] = crow[static_cast<size_t>(i)];
            for (int64_t j = 0; j < N; ++j) {
                float v = host_data[static_cast<size_t>(i * N + j)];
                if (v != 0.0f) {
                    cols.push_back(static_cast<std::int32_t>(j));
                    vals.push_back(v);
                    crow[static_cast<size_t>(i + 1)]++;
                }
            }
        }

        int64_t nnz = static_cast<int64_t>(vals.size());
        Tensor c_crow({M + 1}, DType::Int32, C_dense.device());
        Tensor c_col({nnz}, DType::Int32, C_dense.device());
        Tensor c_vals({nnz}, dtype, C_dense.device());

        queue.memcpy(c_crow.data<std::int32_t>(), crow.data(),
                     static_cast<size_t>(M + 1) * sizeof(std::int32_t)).wait();
        queue.memcpy(c_col.data<std::int32_t>(), cols.data(),
                     static_cast<size_t>(nnz) * sizeof(std::int32_t)).wait();
        queue.memcpy(c_vals.data<float>(), vals.data(),
                     static_cast<size_t>(nnz) * sizeof(float)).wait();

        return SparseTensor::sparse_csr(c_crow, c_col, c_vals, {M, N});
    } else if (dtype == DType::Float64) {
        std::vector<double> host_data(static_cast<size_t>(M * N));
        queue.memcpy(host_data.data(), C_dense.data<double>(),
                     static_cast<size_t>(M * N) * sizeof(double)).wait();

        std::vector<std::int32_t> crow(static_cast<size_t>(M + 1), 0);
        std::vector<std::int32_t> cols;
        std::vector<double> vals;

        for (int64_t i = 0; i < M; ++i) {
            crow[static_cast<size_t>(i + 1)] = crow[static_cast<size_t>(i)];
            for (int64_t j = 0; j < N; ++j) {
                double v = host_data[static_cast<size_t>(i * N + j)];
                if (v != 0.0) {
                    cols.push_back(static_cast<std::int32_t>(j));
                    vals.push_back(v);
                    crow[static_cast<size_t>(i + 1)]++;
                }
            }
        }

        int64_t nnz = static_cast<int64_t>(vals.size());
        Tensor c_crow({M + 1}, DType::Int32, C_dense.device());
        Tensor c_col({nnz}, DType::Int32, C_dense.device());
        Tensor c_vals({nnz}, dtype, C_dense.device());

        queue.memcpy(c_crow.data<std::int32_t>(), crow.data(),
                     static_cast<size_t>(M + 1) * sizeof(std::int32_t)).wait();
        queue.memcpy(c_col.data<std::int32_t>(), cols.data(),
                     static_cast<size_t>(nnz) * sizeof(std::int32_t)).wait();
        queue.memcpy(c_vals.data<double>(), vals.data(),
                     static_cast<size_t>(nnz) * sizeof(double)).wait();

        return SparseTensor::sparse_csr(c_crow, c_col, c_vals, {M, N});
    } else {
        throw std::runtime_error("oneapi spgemm_kernel: unsupported dtype (requires Float32 or Float64)");
    }
#else
    throw std::runtime_error("oneapi spgemm_kernel requires oneMKL (TENZOR_HAS_ONEMKL not defined)");
#endif
}

// ============================================================================
// SparseTrsv: solve L*x = b  (or U*x = b) for a single RHS vector
// ============================================================================

auto sparse_trsv_kernel(const SparseTensor& L, const Tensor& b, bool upper,
                        sycl::queue& queue) -> Tensor {
#ifdef TENZOR_HAS_ONEMKL
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

    if (dtype == DType::Float32) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(N),
            static_cast<std::int64_t>(N),
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
    throw std::runtime_error("oneapi sparse_trsv_kernel requires oneMKL (TENZOR_HAS_ONEMKL not defined)");
#endif
}

// ============================================================================
// SparseTrsm: solve L*X = B  (or U*X = B) for multiple RHS columns
// ============================================================================

auto sparse_trsm_kernel(const SparseTensor& L, const Tensor& B, bool upper,
                        sycl::queue& queue) -> Tensor {
#ifdef TENZOR_HAS_ONEMKL
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

    if (dtype == DType::Float32) {
        ::oneapi::mkl::sparse::init_matrix_handle(&handle);
        ::oneapi::mkl::sparse::set_csr_data(
            queue, handle,
            static_cast<std::int64_t>(N),
            static_cast<std::int64_t>(N),
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
    throw std::runtime_error("oneapi sparse_trsm_kernel requires oneMKL (TENZOR_HAS_ONEMKL not defined)");
#endif
}

} // namespace oneapi
} // namespace tenzor
