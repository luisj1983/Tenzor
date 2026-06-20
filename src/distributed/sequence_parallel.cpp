/**
 * @file sequence_parallel.cpp
 * @brief Implementation of sequence parallelism for long-context LLM training
 *
 * Splits the sequence dimension across the tensor-parallel group so that
 * memory-bound operations (LayerNorm, Dropout, residual connections) operate
 * on local sequence partitions. Communication (all-gather / reduce-scatter)
 * happens at the boundaries of attention blocks.
 *
 * In single-process mode (tp_size == 1), all operations are pass-through.
 */

#include "tenzor/distributed/sequence_parallel.hpp"
#include "tenzor/distributed/process_group.hpp"  // ProcessGroupBase (B3)
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include <stdexcept>

namespace tenzor::distributed {

// ============================================================================
// SequenceParallel Implementation
// ============================================================================

SequenceParallel::SequenceParallel(std::shared_ptr<DeviceMesh> mesh,
                                     const std::string& tp_dim_name)
    : mesh_(std::move(mesh)), tp_dim_name_(tp_dim_name) {

    if (!mesh_) {
        throw std::invalid_argument(
            "SequenceParallel: mesh must not be null");
    }

    // Validate the TP dimension exists (get_dim throws if not found)
    auto dim_idx = mesh_->get_dim(tp_dim_name_);

    tp_size_ = mesh_->shape()[dim_idx];

    // Determine this rank's position along the TP dimension
    auto local_rank = mesh_->get_local_rank();
    auto coord = mesh_->get_coordinate(local_rank);
    tp_rank_ = coord[dim_idx];
}

auto SequenceParallel::scatter_sequence(const Tensor& input,
                                         int64_t seq_dim) -> Tensor {
    if (tp_size_ <= 1) {
        return input;  // Single-process: no-op
    }

    auto seq_len = input.shape()[seq_dim];
    if (seq_len % tp_size_ != 0) {
        throw std::invalid_argument(
            "SequenceParallel::scatter_sequence: sequence length (" +
            std::to_string(seq_len) + ") must be divisible by tp_size (" +
            std::to_string(tp_size_) + ")");
    }

    // Split along seq_dim into tp_size chunks, take this rank's chunk
    auto chunks = input.chunk(tp_size_, seq_dim);
    return chunks[tp_rank_];
}

auto SequenceParallel::gather_sequence(const Tensor& input,
                                        int64_t seq_dim) -> Tensor {
    if (tp_size_ <= 1) {
        return input;  // Single-process: no-op
    }

    // Audit B3: communicate via the per-axis sub-PG, not the global PG.
    // The mesh returns the sub-PG spanning the TP axis (created lazily via
    // ProcessGroupBase::split on first access). Falling back to the global
    // PG would collect tensors across ranks that don't share this rank's
    // non-TP coordinates, producing wrong results on multi-axis meshes.
    auto pg = mesh_->process_group_for_dim(tp_dim_name_);
    if (!pg) {
        throw std::runtime_error(
            "SequenceParallel::gather_sequence: mesh dim '" + tp_dim_name_ +
            "' has tp_size " + std::to_string(tp_size_) +
            " but no ProcessGroup is attached to the mesh. Call "
            "DeviceMesh::set_process_group() before constructing "
            "SequenceParallel.");
    }

    // ProcessGroupBase::all_gather requires the output vector to be
    // pre-sized to world_size with correctly-shaped placeholder tensors;
    // every concrete backend (NCCL/MPI/Gloo) throws std::invalid_argument
    // if output.size() != world_size. Each rank contributes an identically
    // shaped local chunk, so empty_like(input) is the correct placeholder
    // (same pattern as post_attention_scatter below and tensor_parallel.cpp).
    std::vector<Tensor> gathered(static_cast<size_t>(tp_size_));
    for (auto& slot : gathered) {
        slot = empty_like(input);
    }
    pg->all_gather(gathered, input);
    // Concatenate all chunks along the sequence dimension.
    return cat(gathered, seq_dim);
}

auto SequenceParallel::pre_attention_gather(const Tensor& input,
                                             int64_t seq_dim) -> Tensor {
    return gather_sequence(input, seq_dim);
}

auto SequenceParallel::post_attention_scatter(const Tensor& input,
                                               int64_t seq_dim) -> Tensor {
    if (tp_size_ <= 1) {
        return input;  // Single-process: no-op
    }

    auto seq_len = input.shape()[seq_dim];
    if (seq_len % tp_size_ != 0) {
        throw std::invalid_argument(
            "SequenceParallel::post_attention_scatter: sequence length (" +
            std::to_string(seq_len) + ") must be divisible by tp_size (" +
            std::to_string(tp_size_) + ")");
    }

    // Audit B3: use the per-axis sub-PG (see gather_sequence above).
    auto pg = mesh_->process_group_for_dim(tp_dim_name_);
    if (!pg) {
        throw std::runtime_error(
            "SequenceParallel::post_attention_scatter: mesh dim '" +
            tp_dim_name_ + "' has tp_size " + std::to_string(tp_size_) +
            " but no ProcessGroup is attached to the mesh.");
    }

    // Reduce-scatter: split the input along seq_dim into tp_size chunks,
    // reduce (sum) corresponding chunks across ranks, and each rank
    // keeps its chunk. ProcessGroupBase::reduce_scatter signature is
    // (Tensor& output, span<const Tensor> input).
    auto chunks = input.chunk(tp_size_, seq_dim);
    std::vector<Tensor> chunk_vec(chunks.begin(), chunks.end());
    Tensor output = empty_like(chunk_vec[tp_rank_]);
    pg->reduce_scatter(output, chunk_vec);
    return output;
}

auto SequenceParallel::local_seq_len(int64_t global_seq_len) const -> int64_t {
    if (global_seq_len % tp_size_ != 0) {
        throw std::invalid_argument(
            "SequenceParallel::local_seq_len: global_seq_len (" +
            std::to_string(global_seq_len) + ") must be divisible by tp_size (" +
            std::to_string(tp_size_) + ")");
    }
    return global_seq_len / tp_size_;
}

} // namespace tenzor::distributed
