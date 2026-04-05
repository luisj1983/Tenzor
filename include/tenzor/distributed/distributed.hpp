/**
 * @file distributed.hpp
 * @brief Distributed training API and process group management
 *
 * Provides a unified interface for distributed training across multiple nodes,
 * supporting NCCL (GPU) and Gloo (CPU) backends for collective communication.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <mutex>
#include <functional>
#include "../core/device.hpp"
#include "../autograd/variable.hpp"

namespace tenzor {
namespace distributed {

/**
 * @brief Reduction operation types for collective operations.
 */
enum class ReduceOp {
    SUM,     ///< Sum reduction
    PRODUCT, ///< Product reduction
    MIN,     ///< Minimum reduction
    MAX,     ///< Maximum reduction
    BAND,    ///< Bitwise AND
    BOR,     ///< Bitwise OR
    BXOR,    ///< Bitwise XOR
    AVG      ///< Average (sum / world_size)
};

/**
 * @brief Backend types for distributed communication.
 */
enum class Backend {
    NCCL,  ///< NVIDIA NCCL for GPU-to-GPU communication
    GLOO,  ///< Gloo for CPU and fallback communication
    MPI    ///< MPI backend (not implemented)
};

/**
 * @brief Convert backend enum to string.
 */
auto backend_to_string(Backend backend) -> std::string;

/**
 * @brief Parse backend from string.
 */
auto string_to_backend(const std::string& backend) -> Backend;

/**
 * @brief Abstract communication backend interface.
 *
 * Defines the contract that all communication backends (NCCL, Gloo, MPI)
 * must implement for collective operations.
 */
class CommunicationBackend {
public:
    virtual ~CommunicationBackend() = default;

    /**
     * @brief Initialize backend for given rank and world size.
     */
    virtual auto initialize(int rank, int world_size, const std::string& master_addr,
                           int master_port) -> void = 0;

    /**
     * @brief Broadcast tensor from source rank to all ranks.
     */
    virtual auto broadcast(Tensor& tensor, int src_rank) -> void = 0;

    /**
     * @brief All-reduce operation across all processes.
     */
    virtual auto all_reduce(Tensor& tensor, ReduceOp op) -> void = 0;

    /**
     * @brief Asynchronous all-reduce on a caller-provided CUDA stream.
     *
     * Launches the all-reduce without synchronizing. The caller is
     * responsible for synchronization via CUDA events. The default
     * implementation falls back to the synchronous all_reduce().
     *
     * @param tensor Tensor to reduce in-place
     * @param op Reduction operation
     * @param stream CUDA stream (as void*) to launch the operation on
     */
    virtual auto all_reduce_async(Tensor& tensor, ReduceOp op,
                                  void* stream) -> void {
        (void)stream;
        all_reduce(tensor, op);
    }

    /**
     * @brief Check if this backend supports async stream-based operations.
     *
     * @return true for GPU backends (NCCL), false for CPU backends (Gloo)
     */
    virtual auto supports_async_stream() const -> bool { return false; }

    /**
     * @brief Reduce tensor to destination rank.
     */
    virtual auto reduce(Tensor& tensor, int dst_rank, ReduceOp op) -> void = 0;

    /**
     * @brief All-gather tensors from all processes.
     */
    virtual auto all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void = 0;

    /**
     * @brief Gather tensors to destination rank.
     */
    virtual auto gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void = 0;

    /**
     * @brief Scatter tensor from source rank to all ranks.
     */
    virtual auto scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void = 0;

    /**
     * @brief Reduce-scatter operation.
     */
    virtual auto reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op) -> void = 0;

    /**
     * @brief Point-to-point send to a specific rank.
     *
     * Sends the tensor to the destination rank. Blocks until the send
     * buffer is safe to reuse (but the receiver may not have completed
     * the receive yet on some backends).
     *
     * @param tensor Tensor to send
     * @param dst_rank Destination rank
     */
    virtual auto send(const Tensor& tensor, int dst_rank) -> void = 0;

    /**
     * @brief Point-to-point receive from a specific rank.
     *
     * Receives a tensor from the source rank into the provided tensor.
     * The tensor must be pre-allocated with the correct shape and dtype.
     * Blocks until the receive is complete.
     *
     * @param tensor Pre-allocated tensor to receive into
     * @param src_rank Source rank
     */
    virtual auto recv(Tensor& tensor, int src_rank) -> void = 0;

    /**
     * @brief Barrier synchronization.
     */
    virtual auto barrier() -> void = 0;

    /**
     * @brief Cleanup and destroy backend resources.
     */
    virtual auto finalize() -> void = 0;

    /**
     * @brief Get backend type.
     */
    virtual auto backend_type() const -> Backend = 0;

    /**
     * @brief Check if backend supports device type.
     */
    virtual auto supports_device(Device::Type device_type) const -> bool = 0;
};

/**
 * @brief Process group for distributed training.
 *
 * Manages communication backend, process ranks, and provides high-level
 * collective operation APIs. Thread-safe and supports multiple devices.
 *
 * Usage:
 * @code
 * auto pg = ProcessGroup::create_process_group(Backend::NCCL, 0, 4);
 * Tensor data = torch::randn({10, 10}, Device::cuda(0));
 * pg->all_reduce(data, ReduceOp::SUM);
 * @endcode
 */
class ProcessGroup {
public:
    /**
     * @brief Construct process group with backend.
     */
    ProcessGroup(std::unique_ptr<CommunicationBackend> backend, int rank, int world_size);

    /**
     * @brief Destructor - cleanup resources.
     */
    ~ProcessGroup();

    // Collective operations

    /**
     * @brief Broadcast tensor from source rank to all ranks.
     *
     * @param tensor Tensor to broadcast (modified in-place on all ranks)
     * @param src_rank Source process rank (default: 0)
     */
    auto broadcast(Tensor& tensor, int src_rank = 0) -> void;

    /**
     * @brief All-reduce operation across all processes.
     *
     * @param tensor Tensor to reduce (modified in-place)
     * @param op Reduction operation
     */
    auto all_reduce(Tensor& tensor, ReduceOp op = ReduceOp::SUM) -> void;

    /**
     * @brief Asynchronous all-reduce on a caller-provided CUDA stream.
     *
     * Launches the all-reduce without synchronizing. The caller is
     * responsible for synchronization via CUDA events.
     *
     * @param tensor Tensor to reduce (modified in-place)
     * @param op Reduction operation
     * @param stream CUDA stream (as void*) to launch the operation on
     */
    auto all_reduce_async(Tensor& tensor, ReduceOp op, void* stream) -> void;

    /**
     * @brief Check if the underlying backend supports async stream operations.
     *
     * @return true for GPU backends (NCCL), false for CPU backends (Gloo)
     */
    auto supports_async_stream() const -> bool;

    /**
     * @brief Reduce tensor to destination rank.
     *
     * @param tensor Tensor to reduce
     * @param dst_rank Destination rank
     * @param op Reduction operation
     */
    auto reduce(Tensor& tensor, int dst_rank, ReduceOp op = ReduceOp::SUM) -> void;

    /**
     * @brief All-gather tensors from all processes.
     *
     * @param tensor Local tensor to gather
     * @param output Vector of tensors (one per rank)
     */
    auto all_gather(const Tensor& tensor, std::vector<Tensor>& output) -> void;

    /**
     * @brief Gather tensors to destination rank.
     *
     * @param tensor Local tensor
     * @param output Vector of gathered tensors (only valid on dst_rank)
     * @param dst_rank Destination rank
     */
    auto gather(const Tensor& tensor, std::vector<Tensor>& output, int dst_rank) -> void;

    /**
     * @brief Scatter tensor from source rank to all ranks.
     *
     * @param tensors Input tensors (only valid on src_rank)
     * @param output Output tensor (valid on all ranks)
     * @param src_rank Source rank
     */
    auto scatter(const std::vector<Tensor>& tensors, Tensor& output, int src_rank) -> void;

    /**
     * @brief Reduce-scatter operation.
     *
     * @param tensors Input tensors
     * @param output Output tensor
     * @param op Reduction operation
     */
    auto reduce_scatter(const std::vector<Tensor>& tensors, Tensor& output, ReduceOp op = ReduceOp::SUM) -> void;

    /**
     * @brief Point-to-point send to a specific rank.
     *
     * @param tensor Tensor to send
     * @param dst_rank Destination rank
     */
    auto send(const Tensor& tensor, int dst_rank) -> void;

    /**
     * @brief Point-to-point receive from a specific rank.
     *
     * @param tensor Pre-allocated tensor to receive into
     * @param src_rank Source rank
     */
    auto recv(Tensor& tensor, int src_rank) -> void;

    /**
     * @brief Barrier synchronization across all processes.
     */
    auto barrier() -> void;

    // Accessors

    /**
     * @brief Get process rank.
     */
    auto rank() const -> int { return rank_; }

    /**
     * @brief Get world size.
     */
    auto world_size() const -> int { return world_size_; }

    /**
     * @brief Get backend type.
     */
    auto backend() const -> Backend { return backend_->backend_type(); }

    /**
     * @brief Check if this is the master process (rank 0).
     */
    auto is_master() const -> bool { return rank_ == 0; }

    /**
     * @brief Check if backend supports device type.
     */
    auto supports_device(Device::Type device_type) const -> bool {
        return backend_->supports_device(device_type);
    }

    // Factory methods

    /**
     * @brief Create process group from environment variables.
     *
     * Reads RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from environment.
     *
     * @param backend Backend type
     * @return Shared pointer to process group
     */
    static auto create_from_env(Backend backend) -> std::shared_ptr<ProcessGroup>;

    /**
     * @brief Create process group with explicit parameters.
     *
     * @param backend Backend type
     * @param rank Process rank
     * @param world_size Total number of processes
     * @param master_addr Master node address (default: "localhost")
     * @param master_port Master node port (default: 29500)
     * @return Shared pointer to process group
     */
    static auto create_process_group(
        Backend backend,
        int rank,
        int world_size,
        const std::string& master_addr = "localhost",
        int master_port = 29500
    ) -> std::shared_ptr<ProcessGroup>;

private:
    std::unique_ptr<CommunicationBackend> backend_;
    int rank_;
    int world_size_;
    mutable std::mutex mutex_;
};

/**
 * @brief Gradient bucket for efficient communication.
 *
 * Groups multiple gradients together to amortize communication overhead
 * and improve bandwidth utilization during all-reduce operations.
 */
class GradientBucket {
public:
    /**
     * @brief Construct gradient bucket.
     *
     * @param max_size_mb Maximum bucket size in megabytes
     */
    explicit GradientBucket(size_t max_size_mb = 25);

    /**
     * @brief Add gradient to bucket.
     *
     * @param gradient Gradient tensor
     * @return true if bucket is full after adding
     */
    auto add_gradient(const Tensor& gradient) -> bool;

    /**
     * @brief Get all gradients in bucket.
     */
    auto gradients() const -> const std::vector<Tensor>& { return gradients_; }

    /**
     * @brief Get total size in bytes.
     */
    auto size_bytes() const -> size_t { return size_bytes_; }

    /**
     * @brief Check if bucket is full.
     */
    auto is_full() const -> bool { return size_bytes_ >= max_size_bytes_; }

    /**
     * @brief Check if bucket is empty.
     */
    auto is_empty() const -> bool { return gradients_.empty(); }

    /**
     * @brief Reset bucket.
     */
    auto reset() -> void;

private:
    std::vector<Tensor> gradients_;
    size_t size_bytes_{0};
    size_t max_size_bytes_;
};

/**
 * @brief Global process group management.
 */
class DistributedContext {
public:
    /**
     * @brief Initialize distributed context.
     */
    static auto initialize(Backend backend, int rank, int world_size,
                          const std::string& master_addr = "localhost",
                          int master_port = 29500) -> void;

    /**
     * @brief Initialize from environment variables.
     */
    static auto initialize_from_env(Backend backend = Backend::NCCL) -> void;

    /**
     * @brief Get global process group.
     */
    static auto get_process_group() -> std::shared_ptr<ProcessGroup>;

    /**
     * @brief Check if distributed is initialized.
     */
    static auto is_initialized() -> bool;

    /**
     * @brief Get rank.
     */
    static auto get_rank() -> int;

    /**
     * @brief Get world size.
     */
    static auto get_world_size() -> int;

    /**
     * @brief Finalize distributed context.
     */
    static auto finalize() -> void;

    /**
     * @brief Replace the global process group for elastic training recovery.
     *
     * Called when the training group is rebuilt after a worker failure.
     * Existing DDP/FSDP wrappers should call reset_process_group()
     * after this to pick up the new group.
     *
     * @param new_pg New process group
     */
    static auto replace_process_group(std::shared_ptr<ProcessGroup> new_pg) -> void {
        std::lock_guard<std::mutex> lock(mutex_);
        global_process_group_ = std::move(new_pg);
    }

private:
    static std::shared_ptr<ProcessGroup> global_process_group_;
    static std::mutex mutex_;
};

// Convenience functions

/**
 * @brief Initialize distributed training.
 *
 * @param backend Backend to use ("nccl", "gloo", or "mpi")
 * @param rank Process rank (use -1 to read from environment)
 * @param world_size World size (use -1 to read from environment)
 * @param master_addr Master address (default: "localhost")
 * @param master_port Master port (default: 29500)
 */
auto init_process_group(
    const std::string& backend = "nccl",
    int rank = -1,
    int world_size = -1,
    const std::string& master_addr = "localhost",
    int master_port = 29500
) -> void;

/**
 * @brief Destroy process group and cleanup.
 */
auto destroy_process_group() -> void;

/**
 * @brief Get current rank.
 */
auto get_rank() -> int;

/**
 * @brief Get world size.
 */
auto get_world_size() -> int;

/**
 * @brief Check if distributed is initialized.
 */
auto is_initialized() -> bool;

/**
 * @brief Barrier synchronization.
 */
auto barrier() -> void;

/**
 * @brief All-reduce operation on tensor.
 */
auto all_reduce(Tensor& tensor, ReduceOp op = ReduceOp::SUM) -> void;

/**
 * @brief Broadcast tensor from source rank.
 */
auto broadcast(Tensor& tensor, int src_rank = 0) -> void;

/**
 * @brief Point-to-point send tensor to destination rank.
 */
auto send(const Tensor& tensor, int dst_rank) -> void;

/**
 * @brief Point-to-point receive tensor from source rank.
 */
auto recv(Tensor& tensor, int src_rank) -> void;

} // namespace distributed
} // namespace tenzor
