/**
 * @file sequence_parallel.hpp
 * @brief Sequence parallelism for long-context LLM training
 *
 * Implements sequence-parallel utilities that split the sequence dimension
 * across the tensor-parallel (TP) group. This reduces activation memory
 * for operations like LayerNorm and Dropout that operate independently
 * on each token position.
 *
 * Communication pattern in a Transformer block:
 *   1. LayerNorm operates on local sequence partition (no comm)
 *   2. Pre-attention: all-gather sequence dim so each rank has full sequence
 *   3. ColumnParallelLinear QKV projection (TP-sharded, local compute)
 *   4. Attention (local per TP rank)
 *   5. RowParallelLinear output projection (TP-sharded, all-reduce replaced
 *      by reduce-scatter into sequence-parallel partitions)
 *   6. Dropout operates on local sequence partition (no comm)
 *   7. Residual add on local partition
 */

#pragma once

#include "device_mesh.hpp"
#include "distributed.hpp"
#include "../core/tensor.hpp"
#include <memory>
#include <string>

namespace tenzor::distributed {

/**
 * @brief Sequence parallel utilities for splitting sequence dimension across TP group.
 *
 * In long-context LLM training, the sequence dimension is partitioned across
 * the tensor-parallel group for memory-bound operations (LayerNorm, Dropout,
 * residual connections). Before attention, the sequence is all-gathered so
 * each rank sees the full context; after attention, reduce-scatter distributes
 * the output back to sequence partitions.
 *
 * Usage:
 * @code
 * auto mesh = std::make_shared<DeviceMesh>(
 *     Device::Type::CUDA, std::vector<int64_t>{4, 2}, {"dp", "tp"});
 *
 * SequenceParallel sp(mesh, "tp");
 *
 * // input: [batch, local_seq_len, hidden]
 * auto full_seq = sp.pre_attention_gather(input);   // [batch, global_seq_len, hidden]
 * auto attn_out = attention(full_seq);               // [batch, global_seq_len, hidden]
 * auto local_out = sp.post_attention_scatter(attn_out); // [batch, local_seq_len, hidden]
 * @endcode
 */
class SequenceParallel {
public:
    /**
     * @brief Construct sequence parallel handler.
     *
     * @param mesh Device mesh containing the TP dimension
     * @param tp_dim_name Name of the tensor-parallel dimension in the mesh
     * @throws std::invalid_argument if mesh is null or tp_dim_name is not found
     */
    explicit SequenceParallel(std::shared_ptr<DeviceMesh> mesh,
                               const std::string& tp_dim_name = "tp");

    ~SequenceParallel() = default;

    /**
     * @brief Scatter (split) the sequence dimension across the TP group.
     *
     * Splits input along seq_dim into tp_size() equal chunks, and this rank
     * keeps its chunk. In single-process mode, returns input unchanged.
     *
     * @param input Tensor of shape (..., global_seq_len, ...)
     * @param seq_dim Dimension index of the sequence dimension (default: 1)
     * @return Tensor of shape (..., local_seq_len, ...)
     */
    auto scatter_sequence(const Tensor& input, int64_t seq_dim = 1) -> Tensor;

    /**
     * @brief Gather the sequence dimension from all TP ranks.
     *
     * All-gathers along seq_dim to reconstruct the full sequence.
     * In single-process mode, returns input unchanged.
     *
     * @param input Tensor of shape (..., local_seq_len, ...)
     * @param seq_dim Dimension index of the sequence dimension (default: 1)
     * @return Tensor of shape (..., global_seq_len, ...)
     */
    auto gather_sequence(const Tensor& input, int64_t seq_dim = 1) -> Tensor;

    /**
     * @brief All-gather sequence dimension before attention.
     *
     * Semantically identical to gather_sequence(), but named for clarity
     * in the Transformer block communication pattern.
     *
     * @param input Tensor with local sequence partition
     * @param seq_dim Sequence dimension index (default: 1)
     * @return Tensor with full sequence
     */
    auto pre_attention_gather(const Tensor& input, int64_t seq_dim = 1) -> Tensor;

    /**
     * @brief Reduce-scatter sequence dimension after attention.
     *
     * Reduces (sums) the attention output across TP ranks and scatters
     * the result so each rank gets its local sequence partition.
     * This replaces the all-reduce in the RowParallelLinear output projection
     * when sequence parallelism is enabled.
     *
     * In single-process mode, returns input unchanged.
     *
     * @param input Tensor with full sequence output from attention
     * @param seq_dim Sequence dimension index (default: 1)
     * @return Tensor with local sequence partition
     */
    auto post_attention_scatter(const Tensor& input, int64_t seq_dim = 1) -> Tensor;

    /**
     * @brief Compute local sequence length from global sequence length.
     *
     * @param global_seq_len Total sequence length
     * @return Local sequence length (global_seq_len / tp_size())
     * @throws std::invalid_argument if global_seq_len is not divisible by tp_size()
     */
    auto local_seq_len(int64_t global_seq_len) const -> int64_t;

    /**
     * @brief Get the tensor-parallel world size.
     */
    auto tp_size() const -> int64_t { return tp_size_; }

    /**
     * @brief Get this rank's index in the TP group.
     */
    auto tp_rank() const -> int64_t { return tp_rank_; }

private:
    std::shared_ptr<DeviceMesh> mesh_;
    std::string tp_dim_name_;
    int64_t tp_size_ = 1;
    int64_t tp_rank_ = 0;
};

} // namespace tenzor::distributed
