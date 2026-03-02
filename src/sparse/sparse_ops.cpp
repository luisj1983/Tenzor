#include "tenzor/sparse/sparse_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include <cstring>
#include <stdexcept>

namespace tenzor {
namespace sparse {

auto spmm(const SparseTensor& sparse, const Tensor& dense) -> Tensor {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || dense.ndim() != 2) {
        throw std::runtime_error("spmm: both inputs must be 2D");
    }
    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    int64_t N = dense.shape()[1];
    if (K != dense.shape()[0]) {
        throw std::runtime_error("spmm: inner dimensions must match");
    }

    auto result = zeros({M, N}, dense.dtype(), dense.device());
    auto dense_c = dense.contiguous();

    if (sparse.layout() == SparseLayout::COO) {
        auto coo = sparse.is_coalesced() ? sparse : sparse.coalesce();
        auto idx = coo.indices().contiguous();
        auto vals = coo.values().contiguous();
        auto* idx_ptr = idx.data<int64_t>();
        int64_t nnz = coo.nnz();

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* d = dense_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t row = idx_ptr[i];
                int64_t col = idx_ptr[nnz + i];
                float val = v[i];
                for (int64_t j = 0; j < N; ++j) {
                    r[row * N + j] += val * d[col * N + j];
                }
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* d = dense_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                int64_t row = idx_ptr[i];
                int64_t col = idx_ptr[nnz + i];
                double val = v[i];
                for (int64_t j = 0; j < N; ++j) {
                    r[row * N + j] += val * d[col * N + j];
                }
            }
        }
    } else {
        // CSR format
        auto crow = sparse.crow_indices().contiguous();
        auto col = sparse.col_indices().contiguous();
        auto vals = sparse.values().contiguous();
        auto* crow_ptr = crow.data<int64_t>();
        auto* col_ptr = col.data<int64_t>();

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* d = dense_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    int64_t c = col_ptr[j];
                    float val = v[j];
                    for (int64_t n = 0; n < N; ++n) {
                        r[row * N + n] += val * d[c * N + n];
                    }
                }
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* d = dense_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t row = 0; row < M; ++row) {
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    int64_t c = col_ptr[j];
                    double val = v[j];
                    for (int64_t n = 0; n < N; ++n) {
                        r[row * N + n] += val * d[c * N + n];
                    }
                }
            }
        }
    }

    return result;
}

auto spmv(const SparseTensor& sparse, const Tensor& vec) -> Tensor {
    auto sp_shape = sparse.shape();
    if (sp_shape.size() != 2 || vec.ndim() != 1) {
        throw std::runtime_error("spmv: sparse must be 2D, vec must be 1D");
    }
    int64_t M = sp_shape[0];
    int64_t K = sp_shape[1];
    if (K != vec.shape()[0]) {
        throw std::runtime_error("spmv: dimensions must match");
    }

    auto result = zeros({M}, vec.dtype(), vec.device());
    auto vec_c = vec.contiguous();

    if (sparse.layout() == SparseLayout::COO) {
        auto coo = sparse.is_coalesced() ? sparse : sparse.coalesce();
        auto idx = coo.indices().contiguous();
        auto vals = coo.values().contiguous();
        auto* idx_ptr = idx.data<int64_t>();
        int64_t nnz = coo.nnz();

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* x = vec_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t i = 0; i < nnz; ++i) {
                r[idx_ptr[i]] += v[i] * x[idx_ptr[nnz + i]];
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* x = vec_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                r[idx_ptr[i]] += v[i] * x[idx_ptr[nnz + i]];
            }
        }
    } else {
        auto crow = sparse.crow_indices().contiguous();
        auto col = sparse.col_indices().contiguous();
        auto vals = sparse.values().contiguous();
        auto* crow_ptr = crow.data<int64_t>();
        auto* col_ptr = col.data<int64_t>();

        if (vals.dtype() == DType::Float32) {
            auto* v = vals.data<float>();
            auto* x = vec_c.data<float>();
            auto* r = result.data<float>();
            for (int64_t row = 0; row < M; ++row) {
                float sum = 0;
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += v[j] * x[col_ptr[j]];
                }
                r[row] = sum;
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* v = vals.data<double>();
            auto* x = vec_c.data<double>();
            auto* r = result.data<double>();
            for (int64_t row = 0; row < M; ++row) {
                double sum = 0;
                for (int64_t j = crow_ptr[row]; j < crow_ptr[row + 1]; ++j) {
                    sum += v[j] * x[col_ptr[j]];
                }
                r[row] = sum;
            }
        }
    }

    return result;
}

auto add(const SparseTensor& sparse, const Tensor& dense) -> Tensor {
    auto result = sparse.to_dense();
    auto dense_c = dense.contiguous();
    auto result_c = result.contiguous();

    int64_t n = result_c.numel();
    if (n != dense_c.numel()) {
        throw std::runtime_error("sparse::add: shape mismatch");
    }

    if (result_c.dtype() == DType::Float32) {
        auto* r = result_c.data<float>();
        auto* d = dense_c.data<float>();
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    } else if (result_c.dtype() == DType::Float64) {
        auto* r = result_c.data<double>();
        auto* d = dense_c.data<double>();
        for (int64_t i = 0; i < n; ++i) {
            r[i] += d[i];
        }
    }

    return result_c;
}

auto add(const SparseTensor& a, const SparseTensor& b) -> SparseTensor {
    if (a.shape() != b.shape()) {
        throw std::runtime_error("sparse::add: shape mismatch");
    }
    if (a.dtype() != b.dtype()) {
        throw std::runtime_error("sparse::add: dtype mismatch - a is " +
            std::string(dtype_name(a.dtype())) + ", b is " +
            std::string(dtype_name(b.dtype())));
    }

    // Convert both to COO, concatenate indices and values
    auto a_coo = a.to_coo();
    auto b_coo = b.to_coo();

    int64_t nnz_a = a_coo.nnz();
    int64_t nnz_b = b_coo.nnz();
    int64_t total_nnz = nnz_a + nnz_b;
    int64_t sparse_dim = a_coo.sparse_dim();

    if (total_nnz == 0) {
        auto indices = Tensor({sparse_dim, int64_t(0)}, DType::Int64, a.device());
        auto values = Tensor({int64_t(0)}, a.dtype(), a.device());
        return SparseTensor::sparse_coo(indices, values, std::vector<int64_t>(a.shape().begin(), a.shape().end()));
    }

    // Concatenate indices
    auto new_indices = Tensor({sparse_dim, total_nnz}, DType::Int64, a.device());
    auto* ni_ptr = new_indices.data<int64_t>();

    if (nnz_a > 0) {
        auto a_idx = a_coo.indices().contiguous();
        auto* a_ptr = a_idx.data<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            std::memcpy(ni_ptr + d * total_nnz, a_ptr + d * nnz_a, nnz_a * sizeof(int64_t));
        }
    }
    if (nnz_b > 0) {
        auto b_idx = b_coo.indices().contiguous();
        auto* b_ptr = b_idx.data<int64_t>();
        for (int64_t d = 0; d < sparse_dim; ++d) {
            std::memcpy(ni_ptr + d * total_nnz + nnz_a, b_ptr + d * nnz_b, nnz_b * sizeof(int64_t));
        }
    }

    // Concatenate values
    auto new_values = Tensor({total_nnz}, a.dtype(), a.device());
    if (a.dtype() == DType::Float32) {
        auto* nv = new_values.data<float>();
        if (nnz_a > 0) {
            auto av = a_coo.values().contiguous();
            std::memcpy(nv, av.data<float>(), nnz_a * sizeof(float));
        }
        if (nnz_b > 0) {
            auto bv = b_coo.values().contiguous();
            std::memcpy(nv + nnz_a, bv.data<float>(), nnz_b * sizeof(float));
        }
    } else if (a.dtype() == DType::Float64) {
        auto* nv = new_values.data<double>();
        if (nnz_a > 0) {
            auto av = a_coo.values().contiguous();
            std::memcpy(nv, av.data<double>(), nnz_a * sizeof(double));
        }
        if (nnz_b > 0) {
            auto bv = b_coo.values().contiguous();
            std::memcpy(nv + nnz_a, bv.data<double>(), nnz_b * sizeof(double));
        }
    }

    auto shape_vec = std::vector<int64_t>(a.shape().begin(), a.shape().end());
    return SparseTensor::sparse_coo(new_indices, new_values, shape_vec).coalesce();
}

auto mul(const SparseTensor& sparse, double scalar) -> SparseTensor {
    if (sparse.layout() == SparseLayout::COO) {
        auto vals = sparse.values().contiguous();
        int64_t nnz = sparse.nnz();

        auto new_values = Tensor({nnz}, vals.dtype(), vals.device());
        if (vals.dtype() == DType::Float32) {
            auto* src = vals.data<float>();
            auto* dst = new_values.data<float>();
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * s;
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* src = vals.data<double>();
            auto* dst = new_values.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * scalar;
            }
        }

        auto shape_vec = std::vector<int64_t>(sparse.shape().begin(), sparse.shape().end());
        auto result = SparseTensor::sparse_coo(sparse.indices(), new_values, shape_vec);
        return result;
    } else {
        // CSR
        auto vals = sparse.values().contiguous();
        int64_t nnz = sparse.nnz();

        auto new_values = Tensor({nnz}, vals.dtype(), vals.device());
        if (vals.dtype() == DType::Float32) {
            auto* src = vals.data<float>();
            auto* dst = new_values.data<float>();
            float s = static_cast<float>(scalar);
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * s;
            }
        } else if (vals.dtype() == DType::Float64) {
            auto* src = vals.data<double>();
            auto* dst = new_values.data<double>();
            for (int64_t i = 0; i < nnz; ++i) {
                dst[i] = src[i] * scalar;
            }
        }

        auto shape_vec = std::vector<int64_t>(sparse.shape().begin(), sparse.shape().end());
        return SparseTensor::sparse_csr(sparse.crow_indices(), sparse.col_indices(), new_values, shape_vec);
    }
}

} // namespace sparse
} // namespace tenzor
