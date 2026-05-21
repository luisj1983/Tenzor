/**
 * @file sparse_linear.hpp
 * @brief Linear layer with sparse weight matrix
 *
 * Implements y = sparse_matmul(x, W^T) + b where W is stored in CSR format
 * for efficient sparse-dense matrix multiplication (SpMM).
 */

#pragma once

#include "../module.hpp"
#include "tenzor/sparse/sparse_tensor.hpp"
#include <optional>

namespace tenzor {
namespace nn {

/**
 * @brief Linear layer with sparse weight storage.
 *
 * Uses CSR-format sparse weight matrix for memory-efficient computation
 * when the weight matrix has high sparsity (>70% zeros).
 *
 * Forward: y = spmm(sparse_weight, x^T)^T + bias
 *
 * Shape transformations:
 * - Input: (*, in_features)
 * - Output: (*, out_features)
 * - Sparse weight: (out_features, in_features) in CSR format
 */
class SparseLinear : public Module {
public:
    /**
     * @brief Construct sparse linear layer with random sparsity.
     *
     * @param in_features Input feature dimension
     * @param out_features Output feature dimension
     * @param density Fraction of non-zero weights (0.0 to 1.0, default: 0.1)
     * @param bias If true, add learnable bias (default: true)
     */
    SparseLinear(int64_t in_features, int64_t out_features,
                 double density = 0.1, bool bias = true);

    /**
     * @brief Construct sparse linear layer from pre-built sparse weight.
     *
     * @param sparse_weight Pre-built sparse weight tensor (out_features, in_features)
     * @param bias If true, add learnable bias (default: true)
     */
    SparseLinear(const SparseTensor& sparse_weight, bool bias = true);

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in_features=" + std::to_string(in_features_) +
               ", out_features=" + std::to_string(out_features_) +
               ", density=" + std::to_string(density_) +
               ", bias=" + (has_bias_ ? "True" : "False");
    }

    auto sparse_weight() const -> const SparseTensor& { return sparse_weight_.value(); }
    auto has_bias() const -> bool { return has_bias_; }
    auto density() const -> double { return density_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    double density_;
    bool has_bias_;
    // CSR-format sparse weight.  The CSR *pattern* (crow_indices, col_indices,
    // shape) is fixed for the lifetime of the layer — the audit's premise:
    // "sparse pattern is fixed which is standard for SparseLinear".  The
    // *values* are owned by the `"sparse_weight.values"` Parameter so that
    // optimisers (SGD/Adam/Adagrad/...) can update them by treating the values
    // vector as a dense Variable.  Before every forward we rebuild
    // `sparse_weight_` so its `values()` Tensor reflects whatever Tensor the
    // optimiser last wrote into the Parameter — this is necessary because
    // optimisers replace `Variable::tensor()` rather than mutating in place.
    std::optional<SparseTensor> sparse_weight_;

    auto reset_parameters() -> void;
    /// Refresh `sparse_weight_` so its `values()` matches the current
    /// `"sparse_weight.values"` Parameter tensor.  No-op if pattern + values
    /// are already in sync (same storage pointer).
    auto sync_sparse_weight_values() -> void;
};

} // namespace nn
} // namespace tenzor
