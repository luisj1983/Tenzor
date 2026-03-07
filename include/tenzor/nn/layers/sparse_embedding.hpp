/**
 * @file sparse_embedding.hpp
 * @brief Embedding layer with sparse gradient accumulation
 *
 * Unlike standard Embedding which produces dense gradients for the entire
 * embedding table, SparseEmbedding only accumulates gradients for accessed
 * rows, using COO format for gradient storage.
 */

#pragma once

#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Embedding layer with sparse gradient support.
 *
 * Functionally identical to Embedding for forward pass, but backward
 * pass produces sparse gradients (only rows that were looked up get
 * gradient entries). This is critical for large embedding tables where
 * only a small fraction of rows are accessed per batch.
 *
 * Compatible with sparse-aware optimizers (SparseAdam, etc.).
 */
class SparseEmbedding : public Module {
public:
    /**
     * @brief Construct sparse embedding layer.
     *
     * @param num_embeddings Size of the dictionary (vocabulary size)
     * @param embedding_dim Size of each embedding vector
     * @param padding_idx If >= 0, entries at this index are zeroed and not updated
     */
    SparseEmbedding(int64_t num_embeddings, int64_t embedding_dim,
                    int64_t padding_idx = -1);

    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "num_embeddings=" + std::to_string(num_embeddings_) +
               ", embedding_dim=" + std::to_string(embedding_dim_) +
               ", sparse=True" +
               (padding_idx_ >= 0 ? ", padding_idx=" + std::to_string(padding_idx_) : "");
    }

    auto weight() -> Variable& { return *parameters_.at("weight"); }
    auto weight() const -> const Variable& { return *parameters_.at("weight"); }
    auto num_embeddings() const -> int64_t { return num_embeddings_; }
    auto embedding_dim() const -> int64_t { return embedding_dim_; }

private:
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
};

} // namespace nn
} // namespace tenzor
