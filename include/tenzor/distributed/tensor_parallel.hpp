/**
 * @file tensor_parallel.hpp
 * @brief Tensor Parallelism for model-parallel distributed training
 *
 * Implements Megatron-LM style tensor parallelism where individual layers
 * are split across multiple GPUs. Column-parallel splits the output dimension,
 * row-parallel splits the input dimension, and parallel attention distributes
 * attention heads across ranks.
 */

#pragma once

#include "distributed.hpp"
#include "../nn/module.hpp"
#include "../autograd/variable.hpp"
#include <memory>

namespace tenzor::distributed {

/**
 * @brief Column-parallel linear layer.
 *
 * Splits the weight matrix along the output dimension (columns):
 *   Full weight: [out_features, in_features]
 *   Per-rank:    [out_features / world_size, in_features]
 *
 * Forward: local matmul produces partial output, then all-gather across
 * ranks to reconstruct the full output.
 *
 * Backward: gradient with respect to output is split (reduce-scatter),
 * and gradient with respect to input is all-reduced.
 *
 * Usage:
 * @code
 * auto pg = DistributedContext::get_process_group();
 * ColumnParallelLinear layer(1024, 4096, *pg, true);
 * auto output = layer.forward(input);  // Shape: {batch, 4096}
 * @endcode
 */
class ColumnParallelLinear : public nn::Module {
public:
    /**
     * @brief Construct column-parallel linear layer.
     *
     * @param in_features Input feature dimension (not split)
     * @param out_features Full output feature dimension (split across ranks)
     * @param pg Process group for tensor parallelism
     * @param bias Whether to include bias (default: true)
     * @param gather_output Whether to all-gather output (default: true).
     *        Set to false when feeding into RowParallelLinear.
     */
    ColumnParallelLinear(int64_t in_features, int64_t out_features,
                         ProcessGroup& pg, bool bias = true,
                         bool gather_output = true);

    /**
     * @brief Forward pass: local matmul + optional all-gather.
     *
     * @param input Input variable of shape (*, in_features)
     * @return Output variable of shape (*, out_features) if gather_output,
     *         else (*, out_features / world_size)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override;

    /** @brief Get local output features (out_features / world_size) */
    auto local_out_features() const -> int64_t { return local_out_features_; }

    /** @brief Get full output features */
    auto out_features() const -> int64_t { return out_features_; }

    /** @brief Get input features */
    auto in_features() const -> int64_t { return in_features_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    int64_t local_out_features_;
    ProcessGroup& pg_;
    bool has_bias_;
    bool gather_output_;
};

/**
 * @brief Row-parallel linear layer.
 *
 * Splits the weight matrix along the input dimension (rows):
 *   Full weight: [out_features, in_features]
 *   Per-rank:    [out_features, in_features / world_size]
 *
 * Forward: each rank computes a partial output from its input shard,
 * then all-reduce to sum partial results across ranks.
 *
 * Backward: gradient with respect to output is broadcast to all ranks,
 * gradient with respect to input shard is computed locally.
 *
 * Usage:
 * @code
 * auto pg = DistributedContext::get_process_group();
 * RowParallelLinear layer(4096, 1024, *pg, true);
 * auto output = layer.forward(input_shard);  // Input split across ranks
 * @endcode
 */
class RowParallelLinear : public nn::Module {
public:
    /**
     * @brief Construct row-parallel linear layer.
     *
     * @param in_features Full input feature dimension (split across ranks)
     * @param out_features Output feature dimension (not split)
     * @param pg Process group for tensor parallelism
     * @param bias Whether to include bias (default: true). Bias is only
     *        added on rank 0 to avoid double-counting after all-reduce.
     * @param input_is_parallel Whether the input is already split across ranks
     *        (default: true). If false, the input is split in forward.
     */
    RowParallelLinear(int64_t in_features, int64_t out_features,
                      ProcessGroup& pg, bool bias = true,
                      bool input_is_parallel = true);

    /**
     * @brief Forward pass: local matmul on input shard + all-reduce.
     *
     * @param input Input variable. If input_is_parallel, shape is
     *        (*, in_features / world_size). Otherwise, (*, in_features).
     * @return Output variable of shape (*, out_features)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override;

    /** @brief Get local input features (in_features / world_size) */
    auto local_in_features() const -> int64_t { return local_in_features_; }

    /** @brief Get full input features */
    auto in_features() const -> int64_t { return in_features_; }

    /** @brief Get output features */
    auto out_features() const -> int64_t { return out_features_; }

private:
    int64_t in_features_;
    int64_t out_features_;
    int64_t local_in_features_;
    ProcessGroup& pg_;
    bool has_bias_;
    bool input_is_parallel_;
};

/**
 * @brief Parallel multi-head attention distributed across ranks.
 *
 * Distributes attention heads across ranks: each rank computes
 * num_heads / world_size heads. Q, K, V projections use ColumnParallelLinear,
 * and the output projection uses RowParallelLinear.
 *
 * This matches the Megatron-LM attention parallelism pattern where:
 * - Q/K/V projections split output dim (each rank gets its heads' projections)
 * - Attention computation is local per rank (no communication)
 * - Output projection splits input dim and all-reduces
 *
 * Usage:
 * @code
 * auto pg = DistributedContext::get_process_group();
 * ParallelAttention attn(512, 8, *pg);  // 512 dim, 8 heads
 * auto output = attn.forward(input);    // Shape: {batch, seq_len, 512}
 * @endcode
 */
class ParallelAttention : public nn::Module {
public:
    /**
     * @brief Construct parallel attention.
     *
     * @param embed_dim Total embedding dimension
     * @param num_heads Total number of attention heads (must be divisible by world_size)
     * @param pg Process group for tensor parallelism
     * @param dropout Dropout probability (default: 0.0)
     */
    ParallelAttention(int64_t embed_dim, int64_t num_heads,
                      ProcessGroup& pg, float dropout = 0.0f);

    /**
     * @brief Forward pass for parallel attention.
     *
     * @param input Input variable of shape (batch, seq_len, embed_dim)
     * @return Output variable of shape (batch, seq_len, embed_dim)
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override;

    /** @brief Get local number of heads on this rank */
    auto local_num_heads() const -> int64_t { return local_num_heads_; }

    /** @brief Get head dimension */
    auto head_dim() const -> int64_t { return head_dim_; }

private:
    int64_t embed_dim_;
    int64_t num_heads_;
    int64_t local_num_heads_;
    int64_t head_dim_;
    float dropout_;
    ProcessGroup& pg_;

    /** @brief Q/K/V projections (column-parallel, gather_output=false) */
    std::shared_ptr<ColumnParallelLinear> q_proj_;
    std::shared_ptr<ColumnParallelLinear> k_proj_;
    std::shared_ptr<ColumnParallelLinear> v_proj_;

    /** @brief Output projection (row-parallel) */
    std::shared_ptr<RowParallelLinear> out_proj_;
};

} // namespace tenzor::distributed
