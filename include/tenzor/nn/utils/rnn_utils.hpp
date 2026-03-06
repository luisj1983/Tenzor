/**
 * @file rnn_utils.hpp
 * @brief Utilities for variable-length sequence processing in RNNs.
 *
 * Provides PackedSequence and packing/unpacking functions for efficient
 * processing of batches with variable-length sequences.
 */

#pragma once

#include "../../core/tensor.hpp"
#include <vector>

namespace tenzor {
namespace nn {

/**
 * @brief Packed representation of variable-length sequences.
 *
 * PackedSequence stores sequences sorted by length with batch_sizes
 * indicating how many sequences are active at each timestep. This avoids
 * wasted computation on padding tokens.
 *
 * @code
 * // Pack padded sequences
 * Tensor padded({3, 5, 10});  // 3 sequences, max length 5, 10 features
 * Tensor lengths = tensor({5, 3, 1});  // actual lengths
 * auto packed = pack_padded_sequence(padded, lengths, true);
 *
 * // Process with LSTM
 * auto [output_packed, hidden] = lstm.forward(packed, {});
 *
 * // Unpack back to padded
 * auto [output, output_lengths] = pad_packed_sequence(output_packed, true);
 * @endcode
 */
struct PackedSequence {
    Tensor data;              ///< (total_elements, features) - packed data
    Tensor batch_sizes;       ///< (max_seq_len,) int64 - batch size at each timestep
    Tensor sorted_indices;    ///< (batch,) int64 - indices used to sort by length
    Tensor unsorted_indices;  ///< (batch,) int64 - indices to restore original order
};

/**
 * @brief Pack a padded batch of variable-length sequences.
 *
 * @param input Padded input tensor of shape (batch, seq_len, features) if batch_first,
 *              or (seq_len, batch, features) otherwise
 * @param lengths 1D tensor of sequence lengths for each batch element
 * @param batch_first If true, input is (batch, seq_len, features)
 * @param enforce_sorted If true (default), sequences must be sorted by length descending.
 *                       If false, will sort internally.
 * @return PackedSequence containing packed data
 */
auto pack_padded_sequence(const Tensor& input, const Tensor& lengths,
                          bool batch_first = false, bool enforce_sorted = true)
    -> PackedSequence;

/**
 * @brief Unpack a PackedSequence back to padded tensor.
 *
 * @param packed PackedSequence to unpack
 * @param batch_first If true, output is (batch, seq_len, features)
 * @param padding_value Value to fill padding positions (default: 0.0)
 * @param total_length Total output sequence length. If -1, uses max length from packed.
 * @return Pair of (padded_output, lengths) where lengths is the sequence lengths tensor
 */
auto pad_packed_sequence(const PackedSequence& packed,
                         bool batch_first = false,
                         float padding_value = 0.0f,
                         int64_t total_length = -1)
    -> std::pair<Tensor, Tensor>;

/**
 * @brief Pack a list of variable-length tensors.
 *
 * @param sequences Vector of tensors, each of shape (length_i, features)
 * @param enforce_sorted If true, sequences must be sorted by length descending
 * @return PackedSequence
 */
auto pack_sequence(const std::vector<Tensor>& sequences,
                   bool enforce_sorted = true)
    -> PackedSequence;

} // namespace nn
} // namespace tenzor
