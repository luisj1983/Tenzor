/**
 * @file process_group.hpp
 * @brief Process group abstraction for distributed communication
 *
 * Provides a simplified abstract interface for distributed collective
 * communication primitives. This is the user-facing API that wraps
 * the underlying CommunicationBackend implementations (NCCL, Gloo).
 */

#pragma once

#include "../core/tensor.hpp"
#include <vector>
#include <memory>
#include <optional>
#include <span>
#include <string>

namespace tenzor::distributed {

/** @brief Reduction operations for collective communication */
enum class ReduceOp;  // Forward declaration - defined in distributed.hpp

/**
 * @brief Abstract base for distributed communication backends.
 *
 * Provides collective communication primitives (all-reduce, broadcast, etc.)
 * for distributed training across multiple processes/GPUs.
 *
 * This is a simplified interface compared to the full CommunicationBackend.
 * Use ProcessGroup::create_process_group() from distributed.hpp for
 * the full-featured version, or use these convenience wrappers for
 * common DDP use cases.
 */
class ProcessGroupBase {
public:
    virtual ~ProcessGroupBase() = default;

    /** @brief Get rank of current process */
    virtual auto rank() const -> int = 0;

    /** @brief Get total number of processes */
    virtual auto world_size() const -> int = 0;

    /** @brief In-place all-reduce: combine tensor across all processes */
    virtual auto all_reduce(Tensor& tensor, ReduceOp op) -> void = 0;

    /**
     * @brief Asynchronous in-place all-reduce on a given CUDA stream.
     *
     * Launches the all-reduce operation on the provided stream without
     * synchronizing (no cudaDeviceSynchronize). The caller is responsible
     * for synchronization via CUDA events.
     *
     * The default implementation falls back to synchronous all_reduce().
     *
     * @param tensor Tensor to reduce in-place
     * @param op Reduction operation
     * @param stream CUDA stream (as void*) to launch the operation on.
     *              Pass nullptr for the default stream.
     */
    virtual auto all_reduce_async(Tensor& tensor, ReduceOp op,
                                  void* stream) -> void {
        // Default: fall back to synchronous all-reduce (ignores stream)
        (void)stream;
        all_reduce(tensor, op);
    }

    /**
     * @brief Check if this process group supports async stream-based operations.
     *
     * @return true if all_reduce_async uses the provided stream (GPU backends)
     */
    virtual auto supports_async_stream() const -> bool { return false; }

    /** @brief Broadcast tensor from src_rank to all processes */
    virtual auto broadcast(Tensor& tensor, int src_rank) -> void = 0;

    /** @brief Gather tensors from all processes */
    virtual auto all_gather(std::vector<Tensor>& output, const Tensor& input) -> void = 0;

    /** @brief Reduce-scatter: reduce then scatter */
    virtual auto reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void = 0;

    /**
     * @brief Create a sub-process-group containing only the ranks that
     *        supply the same `color`.
     *
     * Audit A3-extended: enables true per-axis sub-PGs for DeviceMesh.
     *
     * This is a **collective operation** — every rank in this PG must
     * invoke `split(color, key)` together with consistent values. Ranks
     * sharing the same `color` end up in the same new sub-PG, ordered by
     * `key` (lower `key` → lower new rank). Pass `color == -1` to indicate
     * "do not participate"; that rank receives `nullptr`.
     *
     * The default implementation throws — backends that genuinely support
     * sub-group creation (NCCL via `ncclCommSplit`, MPI via
     * `MPI_Comm_split`) override this.
     *
     * @param color Group identifier; ranks sharing `color` form one sub-PG.
     *              Use `-1` to opt out (returns nullptr).
     * @param key   Order within the new sub-PG. Lower keys map to lower
     *              ranks in the result.
     * @return Shared pointer to the new sub-PG, or `nullptr` if
     *         `color == -1`. The returned PG's `world_size()` equals the
     *         number of ranks sharing `color`.
     */
    virtual auto split(int color, int key)
        -> std::shared_ptr<ProcessGroupBase>;

    /**
     * @brief Equal-size all-to-all on a single contiguous buffer.
     *
     * Splits `input` evenly into `world_size()` chunks along dimension 0,
     * sends chunk[k] to rank k for every k, and receives the analogous
     * chunk from every peer into `output`. `output[r * chunk .. (r+1) * chunk]`
     * receives from rank r.
     *
     * Both `input` and `output` must have the same shape, and
     * `input.shape()[0]` must be divisible by `world_size()`.
     *
     * The default implementation routes the operation through `all_gather`,
     * which is bandwidth-suboptimal (uses world_size * input_size) but
     * correct on any backend implementing all_gather. Backends that have a
     * native primitive (NCCL, MPI) override this method.
     *
     * Required by DTensor's Shard(a) -> Shard(b) redistribute.
     *
     * @param output Pre-allocated output tensor, same shape & dtype as input.
     * @param input  Local input buffer to be split across peers.
     */
    virtual auto all_to_all_single(Tensor& output, const Tensor& input) -> void;

    /** @brief Synchronization barrier */
    virtual auto barrier() -> void = 0;
};

#ifdef TENZOR_HAS_NCCL
/**
 * @brief NCCL-based process group for GPU communication.
 *
 * Owns its own ncclComm_t communicator, initialized via ncclCommInitRank().
 * The constructor performs a TCP-based bootstrap to exchange the ncclUniqueId
 * from rank 0 to all other ranks, then each rank creates its communicator.
 *
 * Requires CUDA or ROCm with NCCL installed.
 */
class NCCLProcessGroup : public ProcessGroupBase {
public:
    /**
     * @brief Construct NCCL process group and initialize communicator.
     *
     * Rank 0 generates an ncclUniqueId, then broadcasts it to all other
     * ranks via TCP sockets. All ranks then call ncclCommInitRank() to
     * create their communicator.
     *
     * @param rank This process's rank in [0, world_size)
     * @param world_size Total number of processes
     * @param master_addr Address of rank 0 for bootstrap (default: "localhost")
     * @param master_port TCP port for bootstrap exchange (default: 29500)
     */
    NCCLProcessGroup(int rank, int world_size,
                     const std::string& master_addr = "localhost",
                     int master_port = 29500);

    /**
     * @brief Destructor - destroys the owned ncclComm_t.
     *
     * Calls ncclCommDestroy() if the backend registry is still alive.
     * If the runtime is already shutting down, cleanup is skipped to
     * avoid use-after-free of CUDA/NCCL resources.
     */
    ~NCCLProcessGroup() override;

    // Non-copyable, non-movable (NCCL communicator is not safely movable)
    NCCLProcessGroup(const NCCLProcessGroup&) = delete;
    NCCLProcessGroup& operator=(const NCCLProcessGroup&) = delete;
    NCCLProcessGroup(NCCLProcessGroup&&) = delete;
    NCCLProcessGroup& operator=(NCCLProcessGroup&&) = delete;

    auto rank() const -> int override { return rank_; }
    auto world_size() const -> int override { return world_size_; }

    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override;
    auto all_reduce_async(Tensor& tensor, ReduceOp op,
                          void* stream) -> void override;
    auto supports_async_stream() const -> bool override { return true; }
    auto broadcast(Tensor& tensor, int src_rank) -> void override;
    auto all_gather(std::vector<Tensor>& output, const Tensor& input) -> void override;
    auto reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void override;
    auto all_to_all_single(Tensor& output, const Tensor& input) -> void override;
    auto split(int color, int key)
        -> std::shared_ptr<ProcessGroupBase> override;
    auto barrier() -> void override;

private:
    int rank_;
    int world_size_;
    void* comm_{nullptr};  // ncclComm_t stored as void* to avoid header dep
    // L7 fix: cached single-element dummy tensor reused across barrier()
    // calls. Lazy-initialized on first barrier() to avoid the per-call
    // allocation on the distributed hot path. Sized for whichever device
    // the runtime is bound to.
    mutable std::optional<Tensor> barrier_dummy_;

    /// Private constructor used by `split()` to wrap an already-split comm.
    NCCLProcessGroup(int rank, int world_size, void* comm);

    /** @brief TCP bootstrap: exchange ncclUniqueId across all ranks */
    auto bootstrap_unique_id(const std::string& master_addr, int master_port) -> void;

    /** @brief Validate that a tensor is on a GPU device */
    auto validate_gpu_tensor(const Tensor& tensor) const -> void;

    /** @brief Get the GPU device index from a tensor */
    auto get_device_id(const Tensor& tensor) const -> int;
};
#endif

#ifdef TENZOR_HAS_MPI
/**
 * @brief MPI-based process group for CPU and (CUDA-aware) GPU communication.
 *
 * Inf-F1: adapter that exposes the `ProcessGroupBase` interface backed by
 * the existing `MPIBackend`. Adds two native operations beyond the
 * backend's collective set:
 *   - `split(color, key)` → native `MPI_Comm_split`. Returns `nullptr`
 *     when `color < 0` (per `MPI_UNDEFINED` semantics).
 *   - `all_to_all_single(output, input)` → native `MPI_Alltoallv` on the
 *     owning communicator.
 *
 * The remaining PG methods (all_reduce, broadcast, all_gather,
 * reduce_scatter, barrier) delegate to the same MPI backend.
 *
 * Requires the build to define `TENZOR_HAS_MPI`.
 */
class MPIProcessGroup : public ProcessGroupBase {
public:
    /** @brief Construct an MPIProcessGroup over MPI_COMM_WORLD.
     *
     * Initialises an owned `MPIBackend` (calls `MPI_Init` if not already
     * initialised). The communicator used for subsequent ops is the
     * world communicator unless `split()` is called.
     */
    MPIProcessGroup(int rank, int world_size,
                    const std::string& master_addr = "",
                    int master_port = 0);
    ~MPIProcessGroup() override;

    // Non-copyable.
    MPIProcessGroup(const MPIProcessGroup&) = delete;
    MPIProcessGroup& operator=(const MPIProcessGroup&) = delete;

    auto rank() const -> int override { return rank_; }
    auto world_size() const -> int override { return world_size_; }

    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override;
    auto broadcast(Tensor& tensor, int src_rank) -> void override;
    auto all_gather(std::vector<Tensor>& output, const Tensor& input) -> void override;
    auto reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void override;
    auto all_to_all_single(Tensor& output, const Tensor& input) -> void override;
    auto split(int color, int key)
        -> std::shared_ptr<ProcessGroupBase> override;
    auto barrier() -> void override;

private:
    int rank_;
    int world_size_;
    // Communicator stored as void* so this header does not need <mpi.h>.
    // Set to MPI_COMM_WORLD by the public ctor; overwritten by `split()`
    // via the private split-wrapping constructor.
    void* comm_{nullptr};
    bool owns_comm_{false};  ///< true → MPI_Comm_free at destruction

    /// Private constructor used by `split()` to wrap an already-split comm.
    MPIProcessGroup(int rank, int world_size, void* comm, bool owns);

    auto validate_initialized() const -> void;
};
#endif  // TENZOR_HAS_MPI

/**
 * @brief CPU-based process group using TCP/IP sockets.
 *
 * Wraps the GlooBackend via ProcessGroup for simplified access.
 * Works on any hardware as a fallback when GPU communication
 * is unavailable. Uses ring all-reduce for bandwidth-optimal
 * communication.
 */
class GlooProcessGroup : public ProcessGroupBase {
public:
    GlooProcessGroup(int rank, int world_size,
                     const std::string& master_addr = "localhost",
                     int master_port = 29500);
    ~GlooProcessGroup() override;

    auto rank() const -> int override { return rank_; }
    auto world_size() const -> int override { return world_size_; }

    auto all_reduce(Tensor& tensor, ReduceOp op) -> void override;
    auto broadcast(Tensor& tensor, int src_rank) -> void override;
    auto all_gather(std::vector<Tensor>& output, const Tensor& input) -> void override;
    auto reduce_scatter(Tensor& output, std::span<const Tensor> input) -> void override;
    auto all_to_all_single(Tensor& output, const Tensor& input) -> void override;
    auto barrier() -> void override;
    // Inf-F4 (deferred → landed): collective sub-PG creation.
    // Gloo has no native MPI_Comm_split equivalent; this implementation
    // all_gathers (rank, color, key) triples over the parent, computes
    // new rank/size locally per color, and spins a fresh GlooBackend
    // RendezvousStore on a per-color derived TCP port. Ranks with
    // color < 0 opt out and receive nullptr.
    auto split(int color, int key)
        -> std::shared_ptr<ProcessGroupBase> override;

private:
    int rank_;
    int world_size_;
    // Inf-F4: rendezvous metadata kept for `split()` to derive the
    // child PG's TCP port.
    std::string master_addr_;
    int master_port_;
    // Delegate to existing ProcessGroup from distributed.hpp
    std::shared_ptr<class ProcessGroup> pg_;
};

} // namespace tenzor::distributed
