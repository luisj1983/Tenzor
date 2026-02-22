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
    auto barrier() -> void override;

private:
    int rank_;
    int world_size_;
    void* comm_{nullptr};  // ncclComm_t stored as void* to avoid header dep

    /** @brief TCP bootstrap: exchange ncclUniqueId across all ranks */
    auto bootstrap_unique_id(const std::string& master_addr, int master_port) -> void;

    /** @brief Validate that a tensor is on a GPU device */
    auto validate_gpu_tensor(const Tensor& tensor) const -> void;

    /** @brief Get the GPU device index from a tensor */
    auto get_device_id(const Tensor& tensor) const -> int;
};
#endif

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
    auto barrier() -> void override;

private:
    int rank_;
    int world_size_;
    // Delegate to existing ProcessGroup from distributed.hpp
    std::shared_ptr<class ProcessGroup> pg_;
};

} // namespace tenzor::distributed
