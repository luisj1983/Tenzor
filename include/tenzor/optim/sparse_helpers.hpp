/// \file sparse_helpers.hpp
/// \brief Helpers for sparse-aware optimizers (SparseAdam, future SparseSGD).
///
/// The autograd `Variable` class carries an optional `sparse_grad_` slot
/// (`std::optional<SparseTensor>`) populated by ops whose backward
/// produces a sparse gradient — currently `SpGEMMBackward` (function_sparse.cpp)
/// and `EmbeddingBackward` (embedding.cpp). The audit (2026-05-17) found that
/// `SparseAdam` was reading the dense `param.grad()` and approximating
/// sparsity by `any(grad_2d != 0, dim=1)`, which:
///
///   1. Treats fully-dense gradients with sentinel-zero rows as if those
///      rows were inactive (statistically wrong; m,v moments stop
///      advancing on rows that should still update).
///   2. Misses the cost savings the sparse pipeline was designed for —
///      iterating over `total_elements` instead of `nnz`.
///
/// This header centralises the read pattern so SparseAdam and future
/// sparse-aware optimizers have a single canonical entry point.

#pragma once

#include "tenzor/autograd/variable.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"

#include <optional>

namespace tenzor::optim::detail {

/// Extract the sparse gradient from a Variable, if one exists.
///
/// Returns `std::optional<SparseTensor>`:
///   * `std::nullopt` if the Variable has no `sparse_grad_` slot
///     populated (i.e. the producing op did not write one). Callers
///     should fall back to the dense `param.grad()` path.
///   * The wrapped `SparseTensor` otherwise. Layout is whatever the
///     producing op chose (COO for embedding-style row-sparse grads,
///     CSR for SpGEMM-style structured grads).
[[nodiscard]] inline auto
extract_sparse_grad(const tenzor::Variable& v)
        -> std::optional<tenzor::SparseTensor> {
    if (!v.has_sparse_grad()) {
        return std::nullopt;
    }
    return v.sparse_grad();  // returns const std::optional<SparseTensor>&
}

}  // namespace tenzor::optim::detail
