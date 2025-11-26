/**
 * @file distributed_data_parallel.hpp
 * @brief DistributedDataParallel for multi-node training with NCCL/RCCL
 *
 * Implements distributed data parallelism across multiple nodes using NCCL (NVIDIA)
 * or RCCL (AMD) for efficient gradient synchronization and parameter broadcasting.
 */

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include <unordered_map>
#include <thread>
#include <condition_variable>
#include "../../nn/module.hpp"
#include "../../autograd/variable.hpp"
#include "../../core/device.hpp"

// Forward declare NCCL/RCCL types
#ifdef TENZOR_USE_ROCM
    #include <rccl/rccl.h>
    using ncclComm_t = ncclComm_t;
    using ncclUniqueId = ncclUniqueId;
    using ncclDataType_t = ncclDataType_t;
    using ncclRedOp_t = ncclRedOp_t;
#elif defined(TENZOR_HAS_NCCL)
    #include <nccl.h>
#else
    // Stub definitions for builds without NCCL/RCCL
    typedef void* ncclComm_t;
    typedef struct { char internal[128]; } ncclUniqueId;
    typedef enum { ncclFloat32 = 0, ncclFloat64 = 1 } ncclDataType_t;
    typedef enum { ncclSum = 0, ncclProd = 1, ncclMax = 2, ncclMin = 3 } ncclRedOp_t;
#endif

namespace tenzor {
namespace nn {

/**
 * @brief Process group for distributed training coordination.
 *
 * Manages NCCL/RCCL communicators, process ranks, and world size for
 * coordinating multi-node distributed training operations.
 *
 * Thread Safety: All methods are thread-safe and can be called concurrently.
 */
class ProcessGroup {
public:
    /**
     * @brief Construct a process group.
     *
     * @param rank Process rank (0 to world_size-1)
     * @param world_size Total number of processes
     * @param backend Communication backend ("nccl", "gloo", or "mpi")
     */
    ProcessGroup(int rank, int world_size, const std::string& backend = "nccl");

    /**
     * @brief Destructor - cleans up NCCL communicators.
     */
    ~ProcessGroup();

    /**
     * @brief Get process rank.
     *
     * @return Process rank (0-indexed)
     */
    auto rank() const -> int { return rank_; }

    /**
     * @brief Get world size.
     *
     * @return Total number of processes
     */
    auto world_size() const -> int { return world_size_; }

    /**
     * @brief Get backend name.
     *
     * @return Backend string ("nccl", "gloo", or "mpi")
     */
    auto backend() const -> const std::string& { return backend_; }

    /**
     * @brief Get NCCL communicator for a device.
     *
     * @param device_id GPU device ID
     * @return NCCL communicator handle
     */
    auto get_communicator(int device_id) -> ncclComm_t;

    /**
     * @brief Initialize NCCL communicator for a device.
     *
     * @param device_id GPU device ID
     * @param unique_id NCCL unique ID for group initialization
     */
    auto init_communicator(int device_id, const ncclUniqueId& unique_id) -> void;

    /**
     * @brief Broadcast tensor from source rank to all ranks.
     *
     * @param tensor Tensor to broadcast
     * @param src_rank Source process rank
     * @param device_id GPU device ID
     */
    auto broadcast(Tensor& tensor, int src_rank, int device_id) -> void;

    /**
     * @brief All-reduce operation across all processes.
     *
     * @param tensor Tensor to reduce
     * @param op Reduction operation (sum, prod, max, min)
     * @param device_id GPU device ID
     */
    auto all_reduce(Tensor& tensor, ncclRedOp_t op, int device_id) -> void;

    /**
     * @brief Barrier synchronization across all processes.
     */
    auto barrier() -> void;

private:
    int rank_;                                          ///< Process rank
    int world_size_;                                    ///< Total number of processes
    std::string backend_;                               ///< Communication backend
    std::unordered_map<int, ncclComm_t> communicators_; ///< Device ID -> NCCL communicator
    mutable std::mutex comm_mutex_;                     ///< Protect communicator access
};

/**
 * @brief Gradient bucket for efficient communication.
 *
 * Buckets group multiple gradients together to amortize communication overhead
 * and improve bandwidth utilization during all-reduce operations.
 *
 * Design Pattern: Bucket-based gradient synchronization (PyTorch DDP style)
 */
class GradientBucket {
public:
    /**
     * @brief Construct a gradient bucket.
     *
     * @param bucket_size_mb Maximum bucket size in megabytes
     */
    explicit GradientBucket(size_t bucket_size_mb = 25);

    /**
     * @brief Add a parameter's gradient to the bucket.
     *
     * @param param Parameter variable with gradient
     * @return true if bucket is full and should be synchronized
     */
    auto add_gradient(Variable* param) -> bool;

    /**
     * @brief Get all parameters in this bucket.
     *
     * @return Vector of parameter pointers
     */
    auto parameters() const -> const std::vector<Variable*>& { return params_; }

    /**
     * @brief Get total size of gradients in bytes.
     *
     * @return Bucket size in bytes
     */
    auto size_bytes() const -> size_t { return size_bytes_; }

    /**
     * @brief Reset bucket for next iteration.
     */
    auto reset() -> void;

    /**
     * @brief Check if bucket is full.
     *
     * @return true if bucket has reached size limit
     */
    auto is_full() const -> bool { return size_bytes_ >= max_size_bytes_; }

    /**
     * @brief Check if bucket is empty.
     *
     * @return true if bucket contains no gradients
     */
    auto is_empty() const -> bool { return params_.empty(); }

private:
    std::vector<Variable*> params_;  ///< Parameters in this bucket
    size_t size_bytes_{0};           ///< Current bucket size in bytes
    size_t max_size_bytes_;          ///< Maximum bucket size in bytes
};

/**
 * @brief Distributed data parallel wrapper for multi-node training.
 *
 * Implements PyTorch-style DistributedDataParallel with:
 * - NCCL/RCCL all-reduce for gradient synchronization
 * - Automatic gradient bucketing for efficient communication
 * - Multi-node, multi-GPU training support
 * - Synchronous training with gradient averaging
 * - Backward pass hooks for automatic synchronization
 *
 * Algorithm:
 * 1. Replicate model to local GPUs on each node
 * 2. Each process trains on local data shard
 * 3. During backward pass:
 *    a. Bucket gradients as they become ready
 *    b. All-reduce bucketed gradients across all processes
 *    c. Average gradients by dividing by world_size
 * 4. Optimizer updates local copy (all processes have same gradients)
 *
 * Design Pattern: SPMD (Single Program Multiple Data)
 * Thread Safety: Forward/backward passes use NCCL streams for parallelism
 *
 * Example:
 * @code
 * // Initialize process group (typically done via torch.distributed.init_process_group)
 * auto process_group = std::make_shared<ProcessGroup>(
 *     rank,        // From environment: RANK or LOCAL_RANK
 *     world_size,  // From environment: WORLD_SIZE
 *     "nccl"       // Backend
 * );
 *
 * // Create model and wrap with DDP
 * auto model = std::make_shared<MyModel>();
 * auto ddp_model = std::make_shared<DistributedDataParallel>(
 *     model,
 *     process_group,
 *     std::vector<int>{0},  // Local device IDs
 *     0                     // Output device
 * );
 *
 * // Training loop (same as single-GPU, but data is sharded)
 * for (auto& batch : dataloader) {
 *     // Each process sees different data
 *     Variable output = ddp_model->forward(batch.input);
 *     Variable loss = criterion(output, batch.target);
 *     loss.backward();  // Gradients automatically synchronized!
 *     optimizer.step();
 * }
 * @endcode
 */
class DistributedDataParallel : public Module {
public:
    /**
     * @brief Construct DistributedDataParallel wrapper.
     *
     * @param module Module to replicate across processes and devices
     * @param process_group Process group for distributed coordination
     * @param device_ids Local GPU device IDs to use on this process
     * @param output_device Master GPU device ID (default: device_ids[0])
     * @param broadcast_buffers Whether to broadcast buffers at start (default: true)
     * @param find_unused_parameters Find unused parameters for optimization (default: false)
     * @param gradient_as_bucket_view Use bucket view for gradients (default: false)
     * @param bucket_size_mb Gradient bucket size in MB (default: 25)
     * @throws std::runtime_error if device_ids is empty or devices unavailable
     */
    DistributedDataParallel(
        std::shared_ptr<Module> module,
        std::shared_ptr<ProcessGroup> process_group,
        std::vector<int> device_ids = {},
        int output_device = -1,
        bool broadcast_buffers = true,
        bool find_unused_parameters = false,
        bool gradient_as_bucket_view = false,
        size_t bucket_size_mb = 25
    );

    /**
     * @brief Destructor.
     */
    ~DistributedDataParallel() override = default;

    /**
     * @brief Forward pass with distributed data parallelism.
     *
     * Steps:
     * 1. Validate input is on master device
     * 2. Execute forward pass using local module replica
     * 3. Register backward hooks for gradient synchronization
     * 4. Return output (gradient sync happens during backward)
     *
     * @param input Input variable (must be on master device)
     * @return Output variable on master device
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get underlying module.
     *
     * @return Shared pointer to original module
     */
    auto module() -> std::shared_ptr<Module> { return module_; }

    /**
     * @brief Get process group.
     *
     * @return Shared pointer to process group
     */
    auto process_group() -> std::shared_ptr<ProcessGroup> { return process_group_; }

    /**
     * @brief Get device IDs.
     *
     * @return Vector of local GPU device IDs
     */
    auto device_ids() const -> const std::vector<int>& { return device_ids_; }

    /**
     * @brief Get master device ID.
     *
     * @return Output device ID
     */
    auto output_device() const -> int { return output_device_; }

    /**
     * @brief Override parameters to return master module params.
     *
     * @return Vector of parameter pointers from master module
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Override named_parameters for master module.
     *
     * @return Vector of (name, parameter) pairs from master module
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

    /**
     * @brief Set module to training mode.
     *
     * @param mode Training mode flag
     */
    auto train(bool mode = true) -> void;

    /**
     * @brief Set module to evaluation mode.
     */
    auto eval() -> void;

    /**
     * @brief Join all processes (wait for all to finish current iteration).
     *
     * Used with context managers to handle uneven iterations across processes.
     */
    auto join() -> void;

private:
    std::shared_ptr<Module> module_;                ///< Original module
    std::shared_ptr<ProcessGroup> process_group_;   ///< Process group for communication
    std::vector<int> device_ids_;                   ///< Local GPU device IDs
    int output_device_;                             ///< Master GPU device ID
    bool broadcast_buffers_;                        ///< Whether to broadcast buffers
    bool find_unused_parameters_;                   ///< Find unused parameters
    bool gradient_as_bucket_view_;                  ///< Use bucket view for gradients
    size_t bucket_size_mb_;                         ///< Bucket size in MB

    // Gradient bucketing
    std::vector<GradientBucket> buckets_;           ///< Gradient buckets for communication
    std::vector<Variable*> parameters_to_sync_;     ///< Parameters requiring synchronization
    mutable std::mutex sync_mutex_;                 ///< Protect gradient synchronization

    // Synchronization state
    bool first_forward_{true};                      ///< First forward pass flag
    std::atomic<int> num_ready_gradients_{0};       ///< Number of ready gradients
    std::atomic<bool> backward_in_progress_{false}; ///< Backward pass in progress

    /**
     * @brief Initialize distributed training.
     *
     * - Broadcasts model parameters from rank 0 to all processes
     * - Broadcasts buffers if broadcast_buffers_ is true
     * - Creates gradient buckets
     * - Sets up backward hooks for gradient synchronization
     */
    auto initialize_distributed() -> void;

    /**
     * @brief Broadcast parameters from rank 0 to all processes.
     *
     * Ensures all processes start with the same model weights.
     */
    auto broadcast_parameters() -> void;

    /**
     * @brief Broadcast buffers from rank 0 to all processes.
     *
     * Synchronizes batch norm running stats and other buffers.
     */
    auto broadcast_buffers() -> void;

    /**
     * @brief Create gradient buckets for efficient communication.
     *
     * Groups parameters into buckets based on size and reverse order
     * (parameters used later in backward pass are bucketed first).
     */
    auto create_gradient_buckets() -> void;

    /**
     * @brief Register backward hooks for gradient synchronization.
     *
     * Hooks trigger all-reduce when gradients become ready.
     */
    auto register_backward_hooks() -> void;

    /**
     * @brief Synchronize gradients for a bucket using all-reduce.
     *
     * @param bucket Gradient bucket to synchronize
     */
    auto synchronize_bucket(GradientBucket& bucket) -> void;

    /**
     * @brief Synchronize all gradients using NCCL all-reduce.
     *
     * Called automatically during backward pass via hooks.
     * Implements efficient ring all-reduce across all processes.
     */
    auto synchronize_gradients() -> void;

    /**
     * @brief Validate device availability.
     *
     * @throws std::runtime_error if CUDA/ROCm not available or device invalid
     */
    auto validate_devices() -> void;

    /**
     * @brief Get NCCL data type from tensor dtype.
     *
     * @param dtype Tensor data type
     * @return NCCL data type enumeration
     */
    auto get_nccl_datatype(DType dtype) -> ncclDataType_t;
};

/**
 * @brief Helper function to create DistributedDataParallel module.
 *
 * Convenience function that simplifies DDP creation with sensible defaults.
 *
 * @param module Module to parallelize
 * @param process_group Process group for coordination
 * @param device_ids Local GPU device IDs (empty = auto-detect all available)
 * @param output_device Master GPU (default: -1 = use device_ids[0])
 * @return Shared pointer to DistributedDataParallel wrapper
 *
 * @code
 * auto model = std::make_shared<ResNet50>();
 * auto ddp_model = make_distributed_data_parallel(
 *     model,
 *     process_group,
 *     {0}  // Use GPU 0 on this process
 * );
 * @endcode
 */
auto make_distributed_data_parallel(
    std::shared_ptr<Module> module,
    std::shared_ptr<ProcessGroup> process_group,
    std::vector<int> device_ids = {},
    int output_device = -1
) -> std::shared_ptr<DistributedDataParallel>;

/**
 * @brief Initialize distributed training environment.
 *
 * Reads environment variables (RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT)
 * and initializes the process group.
 *
 * @param backend Communication backend ("nccl", "gloo", or "mpi")
 * @return Shared pointer to initialized process group
 * @throws std::runtime_error if environment variables are missing
 *
 * Environment variables:
 * - RANK: Process rank (0 to WORLD_SIZE-1)
 * - WORLD_SIZE: Total number of processes
 * - MASTER_ADDR: Master node IP address (default: "localhost")
 * - MASTER_PORT: Master node port (default: "29500")
 *
 * @code
 * // Typically called at program start
 * auto process_group = init_process_group("nccl");
 * @endcode
 */
auto init_process_group(const std::string& backend = "nccl") -> std::shared_ptr<ProcessGroup>;

/**
 * @brief Destroy process group and cleanup resources.
 *
 * Should be called at program exit to properly cleanup NCCL communicators.
 *
 * @param process_group Process group to destroy
 */
auto destroy_process_group(std::shared_ptr<ProcessGroup> process_group) -> void;

} // namespace nn
} // namespace tenzor
