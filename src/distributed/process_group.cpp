/**
 * @file process_group.cpp
 * @brief Implementation of simplified ProcessGroupBase wrappers
 *
 * Implements GlooProcessGroup (and NCCLProcessGroup when NCCL is available)
 * by delegating to the existing ProcessGroup + CommunicationBackend infrastructure
 * (Gloo) or by directly owning and managing NCCL communicators (NCCL).
 */

#include "tenzor/distributed/process_group.hpp"
#include "tenzor/distributed/distributed.hpp"
#include "tenzor/distributed/gloo_backend.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"   // round() for integer ReduceOp::AVG rounding
#include <atomic>
#include <stdexcept>
#include "tenzor/utils/error.hpp"  // NotImplementedError (S25 / audit-12)
#include <memory>
#include <cstdlib>
#include <cstring>
#include <limits>

#ifdef TENZOR_HAS_MPI
#include "tenzor/distributed/mpi_backend.hpp"
#include <mpi.h>
#endif

namespace tenzor::distributed {

// ============================================================================
// GlooProcessGroup Implementation
// ============================================================================

GlooProcessGroup::GlooProcessGroup(int rank, int world_size,
                                   const std::string& master_addr,
                                   int master_port)
    : rank_(rank), world_size_(world_size),
      master_addr_(master_addr), master_port_(master_port) {

    if (rank < 0 || rank >= world_size) {
        throw std::invalid_argument(
            "GlooProcessGroup: rank " + std::to_string(rank) +
            " must be in range [0, " + std::to_string(world_size) + ")"
        );
    }

    if (world_size <= 0) {
        throw std::invalid_argument(
            "GlooProcessGroup: world_size must be positive, got " +
            std::to_string(world_size)
        );
    }

    // Validate the port before it reaches htons() in the rendezvous socket
    // setup; an out-of-range value would otherwise be silently truncated to 16
    // bits, routing the group to an unintended/colliding port.
    if (master_port < 1 || master_port > 65535) {
        throw std::invalid_argument(
            "GlooProcessGroup: master_port " + std::to_string(master_port) +
            " out of range [1, 65535]"
        );
    }

    // Create the underlying ProcessGroup using Gloo backend
    pg_ = ProcessGroup::create_process_group(
        Backend::GLOO, rank, world_size, master_addr, master_port
    );
}

GlooProcessGroup::~GlooProcessGroup() = default;

auto GlooProcessGroup::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    pg_->all_reduce(tensor, op);
}

auto GlooProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
    pg_->broadcast(tensor, src_rank);
}

auto GlooProcessGroup::all_gather(std::vector<Tensor>& output, const Tensor& input) -> void {
    pg_->all_gather(input, output);
}

auto GlooProcessGroup::reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void {
    std::vector<Tensor> input_vec(input.begin(), input.end());
    pg_->reduce_scatter(input_vec, output);
}

auto GlooProcessGroup::barrier() -> void {
    pg_->barrier();
}

auto GlooProcessGroup::all_to_all_single(Tensor& output, const Tensor& input) -> void {
    // Inf-F5: native paired send/recv all_to_all on Gloo's TCP transport.
    //
    // Sends only chunk-sized payloads per peer (vs. the base-class
    // all_gather fallback which sends full input to every peer — O(W²)
    // chunk volume on the wire). With paired ordering (low-rank
    // sends-first; high-rank recvs-first) the loop is deadlock-free.
    //
    // Contract validation mirrors the base class so callers see a single
    // consistent error shape regardless of override path.
    const int ws = world_size();
    const int my_rank = rank();
    if (input.shape().empty()) {
        throw std::invalid_argument(
            "GlooProcessGroup::all_to_all_single: input must have at least 1 dimension");
    }
    const int64_t total = input.shape()[0];
    if (total % ws != 0) {
        throw std::invalid_argument(
            "GlooProcessGroup::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(ws) + ")");
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) ||
        output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "GlooProcessGroup::all_to_all_single: output must match input shape and dtype");
    }

    const int64_t chunk = total / ws;

    // Build per-peer receive buffers (fresh contiguous tensors); the
    // i-th slot becomes the chunk received from peer i (with i == my_rank
    // being the local-copy fast path).
    auto chunk_shape = std::vector<int64_t>(input.shape().begin(), input.shape().end());
    chunk_shape[0] = chunk;
    std::vector<Tensor> per_peer(ws);

    // Local copy: my chunk of `input` ends up in `per_peer[my_rank]` —
    // no network round-trip.
    per_peer[my_rank] =
        input.slice(0, my_rank * chunk, (my_rank + 1) * chunk).contiguous();

    // For each remote peer, exchange one chunk. Pair (a, b) where a<b:
    // a sends first then receives; b receives first then sends. This
    // avoids the deadlock you'd get if both peers blocked on send.
    for (int peer = 0; peer < ws; ++peer) {
        if (peer == my_rank) continue;
        auto send_chunk =
            input.slice(0, peer * chunk, (peer + 1) * chunk).contiguous();
        Tensor recv_chunk = zeros(chunk_shape, input.dtype(), input.device());
        if (my_rank < peer) {
            pg_->send(send_chunk, peer);
            pg_->recv(recv_chunk, peer);
        } else {
            pg_->recv(recv_chunk, peer);
            pg_->send(send_chunk, peer);
        }
        per_peer[peer] = std::move(recv_chunk);
    }

    // Assemble: peer-ordered concatenation along dim 0.
    // H5 fix: write into the caller's pre-allocated `output` storage in-place
    // instead of rebinding (which would orphan any autograd Variable / view
    // aliased to it). Use the stride-aware zero_()/+= pair rather than a flat
    // std::memcpy so a non-contiguous (strided) output is written correctly
    // through its own strides instead of being corrupted as if contiguous.
    auto assembled = cat(per_peer, 0).reshape(
        std::vector<int64_t>(output.shape().begin(), output.shape().end()));
    output.zero_();
    output += assembled;
}

// Inf-F4: collective sub-PG creation. Builds a fresh GlooProcessGroup on
// a derived TCP port for ranks sharing the same color, ordered by key.
auto GlooProcessGroup::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase>
{
    // Build per-rank (color, key, rank) and all_gather over the parent.
    // Pack as Int32 triples to keep the all_gather simple.
    auto local = zeros({3}, DType::Int32, Device::cpu());
    auto* lp = local.data<int32_t>();
    lp[0] = static_cast<int32_t>(color);
    lp[1] = static_cast<int32_t>(key);
    lp[2] = static_cast<int32_t>(rank_);

    std::vector<Tensor> gathered(world_size_);
    all_gather(gathered, local);

    // For each peer, extract (color, key, peer_rank).
    struct Member { int color; int key; int rank; };
    std::vector<Member> members;
    members.reserve(world_size_);
    for (int r = 0; r < world_size_; ++r) {
        auto* g = gathered[r].data<int32_t>();
        members.push_back({g[0], g[1], g[2]});
    }

    // M3 fix: previously used `master_port_ + 1000 + color` which collides
    // on nested splits or same-color splits from sibling parent PGs. Now
    // derive an offset that depends on both the parent's port AND a
    // process-local counter of splits performed so far, so every split-
    // derived child PG gets a unique port within the process.
    //
    // For multi-process distribution the counter is process-local but the
    // sequence is identical across ranks (since they all execute split()
    // collectively in the same order) — so the derived port matches.
    //
    // CRITICAL: every rank MUST advance this counter in lockstep, INCLUDING
    // ranks that opt out (color < 0). If the opt-out early-return below
    // skipped the fetch_add, an opting-out rank's counter would lag, and on
    // the next collective split() the participating ranks would compute a
    // different my_split_id (hence a different child port) than they did on
    // ranks that took this same split — colliding/mismatched ports and a
    // hung child rendezvous. Increment BEFORE the opt-out check.
    static std::atomic<int> split_counter{0};
    const int my_split_id = split_counter.fetch_add(1, std::memory_order_relaxed);

    // Opt-out path: this rank doesn't participate. The counter was already
    // advanced above so this rank stays in sync with participating peers.
    if (color < 0) {
        return nullptr;
    }

    // Collect members sharing my color, sort by (key, rank) for deterministic
    // new-rank ordering (matches MPI_Comm_split convention).
    std::vector<Member> my_group;
    for (const auto& m : members) {
        if (m.color == color) my_group.push_back(m);
    }
    std::sort(my_group.begin(), my_group.end(),
              [](const Member& a, const Member& b) {
                  if (a.key != b.key) return a.key < b.key;
                  return a.rank < b.rank;
              });

    // Find my position in the sorted group → new rank.
    int new_rank = -1;
    for (size_t i = 0; i < my_group.size(); ++i) {
        if (my_group[i].rank == rank_) {
            new_rank = static_cast<int>(i);
            break;
        }
    }
    if (new_rank < 0) {
        throw std::runtime_error(
            "GlooProcessGroup::split: internal — local rank not found in own color group");
    }
    int new_world_size = static_cast<int>(my_group.size());

    // Use a large enough stride (10000) so colors within one split don't
    // collide with the next split's color-0.
    const int new_port = master_port_ + 1000 + my_split_id * 10000 + color;

    // Validate the derived port fits in a TCP port (uint16). With stride 10000,
    // only a few splits exhaust the range from the default 29500; previously the
    // out-of-range value was silently truncated by htons() in the child PG's
    // sockaddr, routing it to an unintended/colliding port (hang or collision).
    // Fail fast instead, mirroring the NCCL bootstrap path's [1,65535] guard.
    if (new_port < 1 || new_port > 65535) {
        throw std::runtime_error(
            "GlooProcessGroup::split: derived port " + std::to_string(new_port) +
            " out of range [1, 65535] (master_port=" + std::to_string(master_port_) +
            ", split_id=" + std::to_string(my_split_id) + ", color=" +
            std::to_string(color) + "); too many splits for the available port range");
    }

    return std::make_shared<GlooProcessGroup>(
        new_rank, new_world_size, master_addr_, new_port);
}

// ============================================================================
// ProcessGroupBase::split default: throw with a documented contract.
//
// M8 note: this is intentionally a non-pure-virtual default so:
//   (a) future backends that don't support sub-group creation can still
//       inherit from ProcessGroupBase without forcing them to provide a
//       split implementation,
//   (b) tests that use a mock ProcessGroup don't need to override split,
//   (c) the existing override set (NCCL via ncclCommSplit, MPI via
//       MPI_Comm_split, Gloo via gather + new rendezvous) demonstrates
//       the contract for backends that DO support split.
//
// If you're adding a new ProcessGroup subclass that supports sub-group
// creation, override this method.
// ============================================================================
auto ProcessGroupBase::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase>
{
    (void)color; (void)key;
    throw std::runtime_error(
        "ProcessGroupBase::split: not implemented for this backend. "
        "Currently supported: NCCLProcessGroup (ncclCommSplit), "
        "MPIProcessGroup (MPI_Comm_split), GlooProcessGroup (gather + "
        "rendezvous). Add a `split()` override on your custom backend.");
}

// ============================================================================
// ProcessGroupBase non-pure default: all_to_all_single via all_gather + slice.
// ============================================================================
//
// Contract:
//   * `input` and `output` must have the same shape and dtype.
//   * `input.shape()[0]` must be divisible by `world_size()`.
//   * On return, `output` is rebound to a freshly-allocated tensor produced
//     by `cat()`; the caller's pre-allocated buffer is discarded. Backends
//     that override this method (e.g. NCCL) preserve the pre-allocated buffer.
//
// This default is bandwidth-suboptimal (uses world_size^2 chunk volume across
// the wire instead of world_size). Production GPU paths should override
// with a native group-Send/Recv-pair primitive — see NCCLProcessGroup.
auto ProcessGroupBase::all_to_all_single(Tensor& output, const Tensor& input) -> void {
    const int ws = world_size();
    if (input.shape().empty()) {
        throw std::invalid_argument(
            "ProcessGroupBase::all_to_all_single: input must have at least 1 dimension");
    }
    const int64_t total = input.shape()[0];
    if (total % ws != 0) {
        throw std::invalid_argument(
            "ProcessGroupBase::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(ws) + ")"
        );
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) || output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "ProcessGroupBase::all_to_all_single: output must match input shape and dtype");
    }

    // 1. all_gather: every rank receives every peer's full input.
    std::vector<Tensor> gathered(ws);
    all_gather(gathered, input);

    // 2. From peer r's gathered tensor, take the slice destined for *my* rank.
    //    That slice goes into position r of the output (we receive peer-ordered
    //    contributions).
    const int my_rank = rank();
    const int64_t chunk = total / ws;
    std::vector<Tensor> per_peer(ws);
    for (int r = 0; r < ws; ++r) {
        per_peer[r] = gathered[r].slice(0, my_rank * chunk, (my_rank + 1) * chunk);
    }
    // H5 fix: in-place writeback so autograd aliases survive. Write THROUGH the
    // caller's storage with the stride-aware zero_()/+= pair (operator+= honours
    // output's strides) instead of a flat std::memcpy: a non-contiguous output
    // (a strided view) would otherwise have the assembled bytes written into it
    // as if contiguous, corrupting the view and any aliasing tensor. Reshape the
    // 1-D assembled buffer to the output shape so the elementwise add lines up.
    auto assembled_b = cat(per_peer, 0).reshape(
        std::vector<int64_t>(output.shape().begin(), output.shape().end()));
    output.zero_();
    output += assembled_b;
}

// ============================================================================
// Inf-F1: MPIProcessGroup Implementation
// ============================================================================

#ifdef TENZOR_HAS_MPI

namespace {

inline auto reduce_op_to_mpi(ReduceOp op) -> MPI_Op {
    switch (op) {
        case ReduceOp::SUM:     return MPI_SUM;
        case ReduceOp::AVG:     return MPI_SUM;  // divide by world_size on caller
        case ReduceOp::PRODUCT: return MPI_PROD;
        case ReduceOp::MIN:     return MPI_MIN;
        case ReduceOp::MAX:     return MPI_MAX;
        default:
            throw std::invalid_argument(
                "MPIProcessGroup: unsupported ReduceOp");
    }
}

inline auto dtype_to_mpi(DType dtype) -> MPI_Datatype {
    switch (dtype) {
        case DType::Float32: return MPI_FLOAT;
        case DType::Float64: return MPI_DOUBLE;
        case DType::Int32:   return MPI_INT;
        case DType::Int64:   return MPI_LONG_LONG_INT;
        case DType::Int8:    return MPI_INT8_T;
        case DType::UInt8:   return MPI_UINT8_T;
        case DType::Int16:   return MPI_INT16_T;
        case DType::UInt16:  return MPI_UINT16_T;
        case DType::UInt32:  return MPI_UINT32_T;
        case DType::UInt64:  return MPI_UINT64_T;
        case DType::Bool:    return MPI_C_BOOL;
        default:
            // Sentinel — callers must validate dtype before reaching this
            // path. Returning MPI_BYTE was unsafe because MPI_SUM on raw
            // bytes is undefined. Use `validate_mpi_reducible_dtype` below.
            return MPI_BYTE;
    }
}

// H3 fix: explicit gate that throws on any dtype MPI cannot natively
// reduce (F16/BF16/Complex/quantized). Callers must widen to F32 first.
inline auto validate_mpi_reducible_dtype(DType dtype, const char* op) -> void {
    switch (dtype) {
        case DType::Float32: case DType::Float64:
        case DType::Int8:  case DType::Int16: case DType::Int32: case DType::Int64:
        case DType::UInt8: case DType::UInt16: case DType::UInt32: case DType::UInt64:
        case DType::Bool:
            return;
        default:
            throw std::runtime_error(
                std::string("MPIProcessGroup::") + op +
                ": dtype " + std::string(dtype_name(dtype)) +
                " has no native MPI representation. Widen to Float32 before reducing.");
    }
}

#define MPI_PG_CHECK(call) \
    do { \
        int err__ = (call); \
        if (err__ != MPI_SUCCESS) { \
            char err_buf__[MPI_MAX_ERROR_STRING]; int err_len__ = 0; \
            MPI_Error_string(err__, err_buf__, &err_len__); \
            throw std::runtime_error( \
                std::string("MPIProcessGroup: MPI call failed: ") + err_buf__); \
        } \
    } while (0)

// MPI's classic collective APIs take the per-rank element count as an `int`.
// tensor.numel() is int64; for tensors with >2^31 elements the narrowing cast
// would silently truncate (or wrap negative) and MPI would transfer the wrong
// element count, silently corrupting the result. Validate the range and throw a
// clear message instead. (Mirrors MPIBackend::checked_mpi_count.)
inline auto checked_mpi_pg_count(int64_t numel, const char* op) -> int {
    if (numel < 0 || numel > static_cast<int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            std::string("MPIProcessGroup::") + op + ": element count " +
            std::to_string(numel) + " exceeds INT_MAX; tensors with more than " +
            std::to_string(std::numeric_limits<int>::max()) +
            " elements require MPI large-count APIs or chunking");
    }
    return static_cast<int>(numel);
}

// MPI reads/writes the buffer as flat-contiguous. A CUDA (or other non-CPU)
// tensor's data_ptr() is a device pointer that non-CUDA-aware MPI would
// dereference on the host (crash/garbage); a non-contiguous tensor would be
// transferred as if contiguous (corruption). MPIProcessGroup uses MPI_COMM_WORLD
// directly without the CUDA-aware staging the MPIBackend performs, so require
// the tensor to be a contiguous CPU tensor and fail loudly otherwise. Callers
// must stage GPU tensors to host (tensor.to(cpu())) and make views contiguous
// before invoking these collectives.
inline auto require_mpi_pg_buffer(const Tensor& t, const char* op) -> void {
    if (t.device().type != Device::Type::CPU) {
        throw std::invalid_argument(
            std::string("MPIProcessGroup::") + op +
            ": tensor must be on CPU (MPI_COMM_WORLD path is not CUDA-aware here); "
            "stage GPU tensors to host before reducing");
    }
    if (!t.is_contiguous()) {
        throw std::invalid_argument(
            std::string("MPIProcessGroup::") + op +
            ": tensor must be contiguous; MPI transfers the buffer as flat "
            "contiguous and would corrupt a strided view");
    }
}

}  // anonymous namespace

MPIProcessGroup::MPIProcessGroup(int rank, int world_size,
                                 const std::string& /*master_addr*/,
                                 int /*master_port*/)
    : rank_(rank), world_size_(world_size) {
    // Ensure MPI is initialized — match MPIBackend::initialize semantics.
    int inited = 0;
    MPI_PG_CHECK(MPI_Initialized(&inited));
    if (!inited) {
        int provided = 0;
        MPI_PG_CHECK(MPI_Init_thread(nullptr, nullptr,
                                     MPI_THREAD_MULTIPLE, &provided));
    }
    MPI_Comm world = MPI_COMM_WORLD;
    comm_ = static_cast<void*>(world);
    owns_comm_ = false;  // MPI_COMM_WORLD is owned by MPI itself

    // Sanity-check rank/world_size against the communicator.
    int comm_rank = -1, comm_size = -1;
    MPI_PG_CHECK(MPI_Comm_rank(world, &comm_rank));
    MPI_PG_CHECK(MPI_Comm_size(world, &comm_size));
    if (comm_rank != rank || comm_size != world_size) {
        throw std::invalid_argument(
            "MPIProcessGroup: requested (rank=" + std::to_string(rank) +
            ", world_size=" + std::to_string(world_size) +
            ") does not match MPI_COMM_WORLD (rank=" +
            std::to_string(comm_rank) + ", size=" + std::to_string(comm_size) + ")");
    }
}

MPIProcessGroup::MPIProcessGroup(int rank, int world_size,
                                 void* comm, bool owns)
    : rank_(rank), world_size_(world_size),
      comm_(comm), owns_comm_(owns) {}

MPIProcessGroup::~MPIProcessGroup() {
    if (owns_comm_ && comm_ != nullptr) {
        MPI_Comm c = reinterpret_cast<MPI_Comm>(comm_);
        if (c != MPI_COMM_NULL) {
            MPI_Comm_free(&c);  // ignore error at destruction time
        }
    }
}

auto MPIProcessGroup::validate_initialized() const -> void {
    if (comm_ == nullptr) {
        throw std::runtime_error(
            "MPIProcessGroup: communicator is null (uninitialized)");
    }
}

auto MPIProcessGroup::all_reduce(Tensor& tensor, ReduceOp op) -> void {
    validate_initialized();
    validate_mpi_reducible_dtype(tensor.dtype(), "all_reduce");
    require_mpi_pg_buffer(tensor, "all_reduce");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    MPI_PG_CHECK(MPI_Allreduce(MPI_IN_PLACE, tensor.data_ptr(),
                               checked_mpi_pg_count(tensor.numel(), "all_reduce"),
                               dtype_to_mpi(tensor.dtype()),
                               reduce_op_to_mpi(op), comm));
    if (op == ReduceOp::AVG && world_size_ > 1) {
        // In-place divide preserves the caller's storage pointer so aliased
        // Variables/views see the result. Integer AVG must round to nearest
        // (matching Gloo / MPIBackend), not truncate via integer division:
        // widen to Float64, divide, round, narrow back, write through.
        if (is_integer_type(tensor.dtype())) {
            const DType orig = tensor.dtype();
            Tensor widened = tensor.to(DType::Float64);
            auto divisor = tenzor::full({1}, static_cast<double>(world_size_),
                                        DType::Float64, widened.device());
            widened /= divisor;
            Tensor rounded = tenzor::round(widened).to(orig);
            tensor.zero_();
            tensor += rounded.to(tensor.device());
        } else {
            auto scalar = tenzor::full({1}, static_cast<double>(world_size_),
                                        tensor.dtype(), tensor.device());
            tensor /= scalar;
        }
    }
}

auto MPIProcessGroup::broadcast(Tensor& tensor, int src_rank) -> void {
    validate_initialized();
    require_mpi_pg_buffer(tensor, "broadcast");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    MPI_PG_CHECK(MPI_Bcast(tensor.data_ptr(),
                           checked_mpi_pg_count(tensor.numel(), "broadcast"),
                           dtype_to_mpi(tensor.dtype()),
                           src_rank, comm));
}

auto MPIProcessGroup::all_gather(std::vector<Tensor>& output,
                                 const Tensor& input) -> void {
    validate_initialized();
    if (static_cast<int>(output.size()) != world_size_) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_gather: output.size() must equal world_size");
    }
    require_mpi_pg_buffer(input, "all_gather");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    // Use a flat contiguous buffer for the gather, then slice into output[].
    int64_t per_rank_numel = input.numel();
    const int gather_count = checked_mpi_pg_count(per_rank_numel, "all_gather");
    Tensor flat = zeros({static_cast<int64_t>(world_size_) * per_rank_numel},
                        input.dtype(), input.device());
    MPI_PG_CHECK(MPI_Allgather(input.data_ptr(),
                               gather_count,
                               dtype_to_mpi(input.dtype()),
                               flat.data_ptr(),
                               gather_count,
                               dtype_to_mpi(input.dtype()),
                               comm));
    auto in_shape = input.shape();
    for (int r = 0; r < world_size_; ++r) {
        output[r] = flat.slice(0, r * per_rank_numel, (r + 1) * per_rank_numel)
                        .reshape(std::vector<int64_t>(in_shape.begin(), in_shape.end()));
    }
}

auto MPIProcessGroup::reduce_scatter(Tensor& output,
                                     std::span<const Tensor> input) -> void {
    validate_initialized();
    validate_mpi_reducible_dtype(output.dtype(), "reduce_scatter");
    require_mpi_pg_buffer(output, "reduce_scatter");
    if (static_cast<int>(input.size()) != world_size_) {
        throw std::invalid_argument(
            "MPIProcessGroup::reduce_scatter: input.size() must equal world_size");
    }
    for (const auto& chunk : input) {
        if (chunk.dtype() != output.dtype()) {
            throw std::invalid_argument(
                "MPIProcessGroup::reduce_scatter: input chunk dtype must match output dtype");
        }
    }
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    // MPI_Reduce_scatter_block expects a contiguous send buffer with one
    // chunk per peer. Concatenate input[] along dim 0 first.
    std::vector<Tensor> in_vec(input.begin(), input.end());
    Tensor send_buf = cat(in_vec, 0).contiguous();
    // The send buffer must hold exactly output.numel() * world_size elements;
    // a caller-contract violation would otherwise yield an out-of-bounds read.
    if (send_buf.numel() != output.numel() * world_size_) {
        throw std::invalid_argument(
            "MPIProcessGroup::reduce_scatter: concatenated input numel " +
            std::to_string(send_buf.numel()) +
            " must equal output numel * world_size (" +
            std::to_string(output.numel()) + " * " +
            std::to_string(world_size_) + ")");
    }
    // SUM-only by interface contract (see ProcessGroupBase::reduce_scatter);
    // averaging is performed caller-side by dividing the output by world_size.
    MPI_PG_CHECK(MPI_Reduce_scatter_block(
        send_buf.data_ptr(), output.data_ptr(),
        checked_mpi_pg_count(output.numel(), "reduce_scatter"),
        dtype_to_mpi(output.dtype()),
        MPI_SUM, comm));
}

auto MPIProcessGroup::all_to_all_single(Tensor& output,
                                        const Tensor& input) -> void {
    // Inf-F3: native MPI_Alltoallv on the owning communicator. Validates
    // the same contract as the base class.
    validate_initialized();
    if (input.shape().empty()) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_to_all_single: input must have at least 1 dim");
    }
    const int64_t total = input.shape()[0];
    if (total % world_size_ != 0) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_to_all_single: input.shape[0] (" +
            std::to_string(total) + ") must be divisible by world_size (" +
            std::to_string(world_size_) + ")");
    }
    auto same_shape = [](std::span<const int64_t> a, std::span<const int64_t> b) {
        return a.size() == b.size() && std::equal(a.begin(), a.end(), b.begin());
    };
    if (!same_shape(output.shape(), input.shape()) ||
        output.dtype() != input.dtype()) {
        throw std::invalid_argument(
            "MPIProcessGroup::all_to_all_single: output must match input shape and dtype");
    }
    require_mpi_pg_buffer(input, "all_to_all_single");
    require_mpi_pg_buffer(output, "all_to_all_single");
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    const int chunk_elems =
        checked_mpi_pg_count(input.numel() / world_size_, "all_to_all_single");
    // Equal-size chunks → MPI_Alltoall suffices and avoids per-peer displs.
    MPI_PG_CHECK(MPI_Alltoall(input.data_ptr(),  chunk_elems,
                              dtype_to_mpi(input.dtype()),
                              output.data_ptr(), chunk_elems,
                              dtype_to_mpi(output.dtype()),
                              comm));
}

auto MPIProcessGroup::split(int color, int key)
    -> std::shared_ptr<ProcessGroupBase> {
    // Inf-F2: native MPI_Comm_split. Maps `color < 0` to MPI_UNDEFINED so
    // the opting-out rank receives MPI_COMM_NULL → returns nullptr (same
    // contract as NCCLProcessGroup::split).
    validate_initialized();
    auto parent = reinterpret_cast<MPI_Comm>(comm_);
    int mpi_color = (color < 0) ? MPI_UNDEFINED : color;
    MPI_Comm new_comm = MPI_COMM_NULL;
    MPI_PG_CHECK(MPI_Comm_split(parent, mpi_color, key, &new_comm));
    if (new_comm == MPI_COMM_NULL) {
        return nullptr;  // we opted out
    }
    int new_rank = -1, new_size = -1;
    MPI_PG_CHECK(MPI_Comm_rank(new_comm, &new_rank));
    MPI_PG_CHECK(MPI_Comm_size(new_comm, &new_size));
    // shared_ptr can't reach the private ctor; use new directly.
    return std::shared_ptr<MPIProcessGroup>(
        new MPIProcessGroup(new_rank, new_size,
                            static_cast<void*>(new_comm),
                            /*owns=*/true));
}

auto MPIProcessGroup::barrier() -> void {
    validate_initialized();
    auto comm = reinterpret_cast<MPI_Comm>(comm_);
    MPI_PG_CHECK(MPI_Barrier(comm));
}

#endif  // TENZOR_HAS_MPI

}  // namespace tenzor::distributed
