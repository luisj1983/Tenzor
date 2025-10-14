/**
 * @file embedding.hpp
 * @brief Embedding layers for discrete input representations
 *
 * Implements lookup table embeddings for converting discrete tokens (e.g., words, categories)
 * into dense vector representations. Essential for NLP, recommendation systems, and
 * categorical feature encoding.
 */

#pragma once

#include <memory>
#include <string>
#include "../module.hpp"
#include "../../core/tensor.hpp"
#include "../../autograd/variable.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Lookup table embedding layer
 *
 * Stores a fixed-size dictionary of embeddings as a learnable (num_embeddings × embedding_dim) matrix.
 * Each input index maps to a corresponding row (embedding vector).
 *
 * **Use Cases:**
 * - Word embeddings (Word2Vec, GloVe initialization)
 * - Token embeddings in transformers
 * - Categorical feature encoding
 * - Entity embeddings for tabular data
 *
 * **Mathematical Operation:**
 * Given input indices \f$I = [i_1, i_2, ..., i_n]\f$ where \f$0 \le i_j < \text{num\_embeddings}\f$:
 * \f[
 * \text{Output}[j] = W[i_j, :] \quad \text{(lookup row } i_j \text{ from weight matrix)}
 * \f]
 *
 * **Key Features:**
 * - **padding_idx**: Special index whose embedding is always zero (no gradient)
 * - **max_norm**: Renormalizes embeddings exceeding specified norm
 * - **scale_grad_by_freq**: Scale gradients by inverse token frequency
 *
 * **Example:**
 * Example:
 * ```cpp
 * // Vocabulary of 10000 words, 300-dimensional embeddings
 * auto embedding = std::make_shared<Embedding>(10000, 300, 0);
 *
 * // Input: batch of token indices [batch_size, sequence_length]
 * // Variable tokens = ... // Shape: [32, 50] (32 sequences of 50 tokens)
 * // Variable embedded = embedding->forward(tokens); // Shape: [32, 50, 300]
 * ```
 *
 * @param num_embeddings Size of vocabulary/dictionary
 * @param embedding_dim Dimension of embedding vectors
 * @param padding_idx Index representing padding (default: -1, disabled)
 * @param max_norm Maximum norm for embeddings (default: 0.0, disabled)
 * @param norm_type Type of norm (default: 2.0 for L2)
 * @param scale_grad_by_freq Scale gradients by frequency (default: false)
 * @param sparse Use sparse gradient updates (default: false)
 *
 * Complexity:
 * - Forward: O(N) where N is number of input indices
 * - Backward: O(N x D) where D is embedding_dim
 * - Memory: O(V x D) where V is num_embeddings
 *
 * @see EmbeddingBag for efficient bag-of-words embeddings
 */
class Embedding : public Module {
public:
    /**
     * @brief Construct embedding layer
     *
     * @param num_embeddings Size of embedding dictionary (vocabulary size)
     * @param embedding_dim Dimension of each embedding vector
     * @param padding_idx Index to set to zero (no gradient). -1 disables padding
     * @param max_norm If set, renormalizes embeddings to have norm ≤ max_norm
     * @param norm_type Type of norm for max_norm (1.0 for L1, 2.0 for L2, etc.)
     * @param scale_grad_by_freq Scale gradients by inverse frequency of indices
     * @param sparse Use sparse gradient updates (for large vocabularies)
     */
    Embedding(int64_t num_embeddings, int64_t embedding_dim,
              int64_t padding_idx = -1, double max_norm = 0.0,
              double norm_type = 2.0, bool scale_grad_by_freq = false,
              bool sparse = false);

    /**
     * @brief Forward pass: lookup embeddings for input indices
     *
     * @param input Tensor of indices. Shape: any (e.g., [batch, seq_len])
     * @return Embeddings. Shape: input.shape() + [embedding_dim]
     *
     * @pre All indices must satisfy: 0 <= index < num_embeddings
     * @post Output shape appends embedding_dim to input shape
     *
     * @code
     * // Input: [batch=4, seq_len=10]
     * // Output: [4, 10, embedding_dim]
     * auto out = embedding->forward(input);
     * @endcode
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get embedding weight matrix (learnable parameters)
     *
     * @return Reference to weight tensor [num_embeddings, embedding_dim]
     *
     * @code
     * // Load pretrained word vectors
     * auto& W = embedding->weight();
     * W.copy_(pretrained_glove_embeddings);
     * @endcode
     */
    auto weight() -> Variable&;

    /**
     * @brief Get const reference to weight matrix
     */
    auto weight() const -> const Variable&;

private:
    Variable weight_;  ///< Embedding matrix [num_embeddings, embedding_dim]
    int64_t num_embeddings_;
    int64_t embedding_dim_;
    int64_t padding_idx_;
    double max_norm_;
    double norm_type_;
    bool scale_grad_by_freq_;
    bool sparse_;

    /**
     * @brief Initialize embedding weights (Normal(0, 1))
     */
    auto initialize_weights() -> void;

    /**
     * @brief Renormalize embeddings exceeding max_norm
     */
    auto renorm_embeddings(const Tensor& indices) -> void;
};

/**
 * @brief Embedding bag for variable-length sequences
 *
 * Efficiently computes embeddings for bags (sets) of indices, aggregating them via
 * sum, mean, or max. Useful for bag-of-words models, document embeddings, and
 * multi-hot categorical features.
 *
 * **Mathematical Operation:**
 * Given bag of indices \f$B = \{i_1, ..., i_k\}\f$:
 * \f[
 * \text{Output} = \text{aggregate}(W[i_1], W[i_2], ..., W[i_k])
 * \f]
 * where aggregate is sum, mean, or max.
 *
 * **Key Difference from Embedding:**
 * - Embedding: Returns individual vectors for each index
 * - EmbeddingBag: Returns single aggregated vector per bag
 *
 * **Use Cases:**
 * - Document embeddings (sum/mean of word embeddings)
 * - User/item embeddings from interaction history
 * - Multi-hot feature encoding with aggregation
 *
 * **Example:**
 * @code
 * // Bag-of-words document embeddings
 * auto embedding_bag = std::make_shared<EmbeddingBag>(10000, 300, 0.0, 2.0, false, "mean");
 *
 * // Document as bag of word indices
 * Variable word_indices = ... // Shape: [total_words] (flattened across docs)
 * Variable offsets = ...      // Shape: [num_docs] (start index of each doc)
 *
 * // Output: one embedding per document
 * Variable doc_embeddings = embedding_bag->forward(word_indices, offsets);
 * // Shape: [num_docs, 300]
 * @endcode
 *
 * @param num_embeddings Size of vocabulary
 * @param embedding_dim Dimension of embeddings
 * @param max_norm Maximum embedding norm (0.0 disables)
 * @param norm_type Type of norm for max_norm
 * @param scale_grad_by_freq Scale gradients by frequency
 * @param mode Aggregation mode: "sum", "mean", or "max"
 * @param sparse Use sparse gradients
 * @param include_last_offset Whether offsets includes final boundary
 *
 * @par Complexity
 * - Forward: O(N × D) where N is total indices, D is embedding_dim
 * - More efficient than Embedding + manual aggregation
 *
 * @see Embedding
 */
class EmbeddingBag : public Module {
public:
    /**
     * @brief Construct embedding bag layer
     *
     * @param num_embeddings Size of embedding dictionary
     * @param embedding_dim Dimension of embeddings
     * @param max_norm Maximum norm for embeddings (0.0 disables)
     * @param norm_type Type of norm (1.0 for L1, 2.0 for L2)
     * @param scale_grad_by_freq Scale gradients by frequency
     * @param mode Aggregation mode: "sum", "mean", or "max"
     * @param sparse Use sparse gradient updates
     * @param include_last_offset If true, offsets[n] = input.size()
     */
    EmbeddingBag(int64_t num_embeddings, int64_t embedding_dim,
                 double max_norm = 0.0, double norm_type = 2.0,
                 bool scale_grad_by_freq = false, const std::string& mode = "mean",
                 bool sparse = false, bool include_last_offset = false);

    /**
     * @brief Forward pass: aggregate embeddings for bags
     *
     * @param input Flattened tensor of indices [total_elements]
     * @param offsets Starting index of each bag [num_bags]. If empty, treats input as single bag
     * @return Aggregated embeddings [num_bags, embedding_dim] or [1, embedding_dim]
     *
     * **Offset Convention:**
     * - offsets[i] = starting index of bag i in input
     * - bag i contains indices from offsets[i] to offsets[i+1]-1
     * - If include_last_offset=false, last bag goes to end of input
     *
     * @code
     * // 3 documents with variable number of words
     * Variable input = Tensor::from_blob({0, 5, 12, 3, 7, 8, 15});  // Word IDs
     * Variable offsets = Tensor::from_blob({0, 3, 5});  // Doc starts at 0, 3, 5
     * // Bag 0: [0, 5, 12]
     * // Bag 1: [3, 7]
     * // Bag 2: [8, 15]
     * auto out = embedding_bag->forward(input, offsets); // [3, embedding_dim]
     * @endcode
     */
    auto forward(const Variable& input, const Variable& offsets = Variable{}) -> Variable;

    /**
     * @brief Default forward pass (required by Module base class)
     */
    auto forward(const Variable& input) -> Variable override;

    /**
     * @brief Get embedding weight matrix
     */
    auto weight() -> Variable&;

    /**
     * @brief Get const reference to weight matrix
     */
    auto weight() const -> const Variable&;

private:
    std::shared_ptr<Embedding> embedding_;  ///< Underlying embedding layer
    std::string mode_;                      ///< Aggregation mode: "sum", "mean", "max"
    bool include_last_offset_;

    /**
     * @brief Aggregate embeddings within each bag
     */
    auto aggregate_embeddings(const Variable& embeddings, const Variable& offsets) -> Variable;
};

} // namespace nn
} // namespace tenzor
