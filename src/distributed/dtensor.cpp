/**
 * @file dtensor.cpp
 * @brief DTensor implementation
 */

#include "tenzor/distributed/dtensor.hpp"
#include "tenzor/distributed/distributed.hpp"       // for ReduceOp
#include "tenzor/distributed/process_group.hpp"     // for ProcessGroupBase
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"

#include <stdexcept>
#include <format>

namespace tenzor::distributed {

namespace {

// Map DTensor's user-facing reduction op to the collective ReduceOp enum.
auto to_collective_op(DTensorReduceOp op) -> ReduceOp {
    switch (op) {
        case DTensorReduceOp::Sum:  return ReduceOp::SUM;
        case DTensorReduceOp::Mean: return ReduceOp::AVG;
        case DTensorReduceOp::Max:  return ReduceOp::MAX;
        case DTensorReduceOp::Min:  return ReduceOp::MIN;
    }
    throw std::runtime_error("DTensor: unknown DTensorReduceOp");
}

} // namespace

DTensor::DTensor(Tensor local_tensor,
                 std::shared_ptr<DeviceMesh> mesh,
                 std::vector<Placement> placements)
    : local_tensor_(std::move(local_tensor)),
      mesh_(std::move(mesh)),
      placements_(std::move(placements)) {

    if (static_cast<int64_t>(placements_.size()) != mesh_->ndim()) {
        throw std::invalid_argument(std::format(
            "DTensor: expected {} placements (one per mesh dimension), got {}",
            mesh_->ndim(), placements_.size()));
    }
}

auto DTensor::local_tensor() const -> const Tensor& {
    return local_tensor_;
}

auto DTensor::local_tensor() -> Tensor& {
    return local_tensor_;
}

auto DTensor::shape() const -> std::vector<int64_t> {
    auto local_shape = local_tensor_.shape();
    std::vector<int64_t> global_shape(local_shape.begin(), local_shape.end());

    for (int64_t mesh_dim = 0; mesh_dim < mesh_->ndim(); ++mesh_dim) {
        if (auto* shard = std::get_if<Shard>(&placements_[mesh_dim])) {
            auto tensor_dim = shard->dim;
            if (tensor_dim < 0 || tensor_dim >= static_cast<int64_t>(global_shape.size())) {
                throw std::out_of_range(std::format(
                    "DTensor::shape: Shard dim {} is out of range for tensor with {} dimensions",
                    tensor_dim, global_shape.size()));
            }
            global_shape[tensor_dim] *= mesh_->shape()[mesh_dim];
        }
        // Replicate and Partial don't change the global shape
    }

    return global_shape;
}

auto DTensor::mesh() const -> const DeviceMesh& {
    return *mesh_;
}

auto DTensor::placements() const -> const std::vector<Placement>& {
    return placements_;
}

auto DTensor::redistribute(const std::vector<Placement>& new_placements) -> DTensor {
    if (static_cast<int64_t>(new_placements.size()) != mesh_->ndim()) {
        throw std::invalid_argument(std::format(
            "DTensor::redistribute: expected {} placements, got {}",
            mesh_->ndim(), new_placements.size()));
    }

    Tensor current = local_tensor_;

    for (int64_t mesh_dim = 0; mesh_dim < mesh_->ndim(); ++mesh_dim) {
        const auto& old_p = placements_[mesh_dim];
        const auto& new_p = new_placements[mesh_dim];
        const auto mesh_size = mesh_->shape()[mesh_dim];
        const auto& dim_name = mesh_->dim_names()[mesh_dim];

        // Same-placement no-op (with one subtle case: Shard(a)→Shard(a)).
        if (old_p.index() == new_p.index()) {
            if (auto* old_s = std::get_if<Shard>(&old_p)) {
                auto* new_s = std::get_if<Shard>(&new_p);
                if (old_s->dim == new_s->dim) {
                    continue;
                }
                // Shard(a) -> Shard(b): all-to-all (audit B2).
                // Falls through to the case below.
            } else {
                continue;  // Replicate->Replicate or Partial->Partial
            }
        }

        // Single-axis size 1 reduces every collective to a no-op except the
        // local-narrow case for Replicate→Shard (which is also trivially a
        // no-op since chunk_size == dim_size).
        if (mesh_size == 1) {
            if (std::holds_alternative<Replicate>(old_p) &&
                std::holds_alternative<Shard>(new_p)) {
                // No work: 1 rank, the whole tensor is already "this rank's"
                // shard.
                continue;
            }
            continue;
        }

        // For every non-trivial transition we need a process group. Fetch the
        // per-dim PG (currently the full-mesh PG; A3-extended will produce
        // true per-axis sub-groups). Throw a clear error if absent.
        auto pg = mesh_->process_group_for_dim(dim_name);
        const auto need_pg = [&]() {
            if (!pg) {
                throw std::runtime_error(std::format(
                    "DTensor::redistribute: mesh dim '{}' has size {} but no "
                    "ProcessGroup attached. Attach one via "
                    "DeviceMesh::set_process_group() before redistributing.",
                    dim_name, mesh_size));
            }
        };

        // ---- Shard(a) -> Shard(b): all-to-all (audit B2). -----------------
        if (auto* old_s = std::get_if<Shard>(&old_p)) {
            if (auto* new_s = std::get_if<Shard>(&new_p);
                new_s && old_s->dim != new_s->dim) {
                need_pg();
                const int64_t src_dim = old_s->dim;
                const int64_t dst_dim = new_s->dim;
                const int64_t src_extent = current.size(src_dim);  // = A
                const int64_t dst_local  = current.size(dst_dim);  // = B
                if (dst_local % mesh_size != 0) {
                    throw std::runtime_error(std::format(
                        "DTensor::redistribute: Shard({}) -> Shard({}) requires "
                        "the destination dim ({}) of the local tensor to be "
                        "divisible by mesh_size ({}); got {}.",
                        src_dim, dst_dim, dst_dim, mesh_size, dst_local));
                }
                const int64_t chunk_b = dst_local / mesh_size;

                // Split local along dst_dim into mesh_size chunks. Chunk k
                // is the data this rank should send to peer k.
                std::vector<Tensor> send_chunks;
                send_chunks.reserve(static_cast<size_t>(mesh_size));
                for (int64_t k = 0; k < mesh_size; ++k) {
                    send_chunks.push_back(
                        current.narrow(dst_dim, k * chunk_b, chunk_b));
                }
                // The all_to_all_single API expects a single contiguous buffer
                // where dim 0 is the peer axis. Bring dst_dim to the front,
                // pack the W chunks contiguously, then call alltoall.
                Tensor packed = cat(send_chunks, dst_dim).contiguous();
                // Move dst_dim to dim 0 so the per-peer split happens on dim 0.
                // We assemble the contiguous "[peer * chunk_b, ...]" layout by
                // permuting dst_dim to position 0.
                std::vector<int64_t> perm;
                perm.reserve(packed.ndim());
                perm.push_back(dst_dim);
                for (int64_t d = 0; d < packed.ndim(); ++d) {
                    if (d != dst_dim) perm.push_back(d);
                }
                Tensor packed_first = packed.permute(perm).contiguous();
                Tensor recv_first = empty(
                    std::vector<int64_t>(packed_first.shape().begin(),
                                         packed_first.shape().end()),
                    packed_first.dtype(),
                    packed_first.device());
                pg->all_to_all_single(recv_first, packed_first);
                // Inverse permute to restore original axis order.
                std::vector<int64_t> inv_perm(perm.size());
                for (size_t i = 0; i < perm.size(); ++i) {
                    inv_perm[static_cast<size_t>(perm[i])] =
                        static_cast<int64_t>(i);
                }
                Tensor recv = recv_first.permute(inv_perm).contiguous();
                // Now recv has shape [..., dst_dim=chunk_b*W (W peer chunks
                // stacked), ...]. Split along dst_dim back into peer chunks
                // and cat them along src_dim to expand the source dim to A.
                std::vector<Tensor> recv_chunks;
                recv_chunks.reserve(static_cast<size_t>(mesh_size));
                for (int64_t k = 0; k < mesh_size; ++k) {
                    recv_chunks.push_back(
                        recv.narrow(dst_dim, k * chunk_b, chunk_b));
                }
                // Each chunk has src_dim of size A/W (since peer k held the
                // k-th src-dim block). Concat along src_dim to restore A.
                current = cat(recv_chunks, src_dim).contiguous();
                // dst_dim now has size chunk_b == B/W (our shard slice on the
                // destination axis). The redistribute loop's invariant holds.
                (void)src_extent;
                continue;
            }
        }

        // ---- Shard -> Replicate: all-gather + cat (audit B2). ------------
        if (std::holds_alternative<Shard>(old_p) &&
            std::holds_alternative<Replicate>(new_p)) {
            need_pg();
            const int64_t shard_dim = std::get<Shard>(old_p).dim;
            // ProcessGroupBase::all_gather requires the output vector to be
            // pre-sized to the group size (each rank's slot is filled in place);
            // passing an empty vector made every real multi-rank gather throw.
            std::vector<Tensor> gathered(static_cast<size_t>(mesh_size));
            pg->all_gather(gathered, current);
            if (static_cast<int64_t>(gathered.size()) != mesh_size) {
                throw std::runtime_error(std::format(
                    "DTensor::redistribute: all_gather returned {} tensors "
                    "but mesh dim '{}' has size {}",
                    gathered.size(), dim_name, mesh_size));
            }
            current = cat(gathered, shard_dim).contiguous();
            continue;
        }

        // ---- Replicate -> Shard: pure local narrow (no comms). ----------
        if (std::holds_alternative<Replicate>(old_p) &&
            std::holds_alternative<Shard>(new_p)) {
            const int64_t shard_dim = std::get<Shard>(new_p).dim;
            const int64_t dim_size = current.size(shard_dim);
            if (dim_size % mesh_size != 0) {
                throw std::runtime_error(std::format(
                    "DTensor::redistribute: cannot shard dimension {} of size {} "
                    "across {} devices evenly",
                    shard_dim, dim_size, mesh_size));
            }
            const int64_t chunk_size = dim_size / mesh_size;
            auto my_coord = mesh_->get_coordinate(mesh_->get_local_rank());
            const int64_t my_idx = my_coord[mesh_dim];
            current = current.narrow(shard_dim, my_idx * chunk_size, chunk_size);
            continue;
        }

        // ---- Partial -> Replicate: all-reduce (audit B2). ----------------
        if (std::holds_alternative<Partial>(old_p) &&
            std::holds_alternative<Replicate>(new_p)) {
            need_pg();
            const auto op = to_collective_op(std::get<Partial>(old_p).op);
            // all_reduce is in-place; clone to avoid mutating the caller's
            // local_tensor when `current` still aliases it.
            current = current.clone();
            pg->all_reduce(current, op);
            continue;
        }

        // ---- Partial -> Shard: reduce-scatter (audit B2). ---------------
        if (std::holds_alternative<Partial>(old_p) &&
            std::holds_alternative<Shard>(new_p)) {
            need_pg();
            const int64_t shard_dim = std::get<Shard>(new_p).dim;
            const auto p_op = std::get<Partial>(old_p).op;
            if (p_op != DTensorReduceOp::Sum) {
                // reduce_scatter is conventionally SUM-only on the backend
                // primitives we expose. For other ops, do all_reduce + slice.
                const auto op = to_collective_op(p_op);
                current = current.clone();
                pg->all_reduce(current, op);
                const int64_t dim_size = current.size(shard_dim);
                if (dim_size % mesh_size != 0) {
                    throw std::runtime_error(std::format(
                        "DTensor::redistribute: cannot shard dimension {} of size {} "
                        "across {} devices evenly",
                        shard_dim, dim_size, mesh_size));
                }
                const int64_t chunk_size = dim_size / mesh_size;
                auto my_coord = mesh_->get_coordinate(mesh_->get_local_rank());
                const int64_t my_idx = my_coord[mesh_dim];
                current = current.narrow(shard_dim, my_idx * chunk_size, chunk_size);
                continue;
            }

            // SUM path: native reduce_scatter is bandwidth-optimal.
            const int64_t dim_size = current.size(shard_dim);
            if (dim_size % mesh_size != 0) {
                throw std::runtime_error(std::format(
                    "DTensor::redistribute: Partial -> Shard({}) requires "
                    "the shard dim ({}) to be divisible by mesh_size ({}); "
                    "got {}.",
                    shard_dim, shard_dim, mesh_size, dim_size));
            }
            const int64_t chunk_size = dim_size / mesh_size;
            std::vector<Tensor> chunks;
            chunks.reserve(static_cast<size_t>(mesh_size));
            for (int64_t k = 0; k < mesh_size; ++k) {
                chunks.push_back(
                    current.narrow(shard_dim, k * chunk_size, chunk_size)
                           .contiguous());
            }
            // Allocate the per-rank output (this rank's chunk of the global).
            auto out_shape_v = std::vector<int64_t>(
                chunks[0].shape().begin(), chunks[0].shape().end());
            Tensor out = empty(out_shape_v, current.dtype(), current.device());
            pg->reduce_scatter(out, chunks);
            current = out;
            continue;
        }

        // ---- Invalid transitions. ----------------------------------------
        if (std::holds_alternative<Shard>(old_p) &&
            std::holds_alternative<Partial>(new_p)) {
            throw std::runtime_error(
                "DTensor::redistribute: Shard -> Partial is not a valid "
                "placement transition");
        }
        if (std::holds_alternative<Replicate>(old_p) &&
            std::holds_alternative<Partial>(new_p)) {
            throw std::runtime_error(
                "DTensor::redistribute: Replicate -> Partial is not a valid "
                "placement transition");
        }
    }

    return DTensor(std::move(current), mesh_, new_placements);
}

auto DTensor::full_tensor() const -> Tensor {
    // Build all-Replicate placements
    std::vector<Placement> all_replicate(mesh_->ndim(), Replicate{});

    // Create a mutable copy and redistribute
    DTensor mutable_copy(local_tensor_, mesh_, placements_);
    auto gathered = mutable_copy.redistribute(all_replicate);
    return gathered.local_tensor();
}

auto DTensor::from_global(const Tensor& tensor,
                           std::shared_ptr<DeviceMesh> mesh,
                           const std::vector<Placement>& placements) -> DTensor {
    if (static_cast<int64_t>(placements.size()) != mesh->ndim()) {
        throw std::invalid_argument(std::format(
            "DTensor::from_global: expected {} placements, got {}",
            mesh->ndim(), placements.size()));
    }

    Tensor local = tensor;

    // Store NORMALIZED shard dims so shape()/redistribute (which reject/misuse
    // raw negative dims) see the same non-negative dims from_global computed here.
    std::vector<Placement> norm_placements = placements;
    for (int64_t mesh_dim = 0; mesh_dim < mesh->ndim(); ++mesh_dim) {
        if (auto* shard = std::get_if<Shard>(&placements[mesh_dim])) {
            // Shard stores a raw dim; normalize Python-style negatives against
            // the local tensor rank before indexing.
            auto shard_dim = shard->dim < 0 ? shard->dim + local.ndim() : shard->dim;
            std::get_if<Shard>(&norm_placements[mesh_dim])->dim = shard_dim;
            auto mesh_size = mesh->shape()[mesh_dim];
            auto dim_size = local.size(shard_dim);

            if (dim_size % mesh_size != 0) {
                throw std::runtime_error(std::format(
                    "DTensor::from_global: cannot shard dimension {} of size {} "
                    "across {} devices evenly",
                    shard_dim, dim_size, mesh_size));
            }

            auto chunk_size = dim_size / mesh_size;
            auto my_coord = mesh->get_coordinate(mesh->get_local_rank());
            auto my_idx = my_coord[mesh_dim];

            local = local.narrow(shard_dim, my_idx * chunk_size, chunk_size);
        }
        // Replicate: keep the full tensor (no-op)
        // Partial: the caller is responsible for providing partial data
    }

    return DTensor(std::move(local), std::move(mesh), std::move(norm_placements));
}

} // namespace tenzor::distributed
