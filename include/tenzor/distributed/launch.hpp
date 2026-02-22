/**
 * @file launch.hpp
 * @brief Launch utility for distributed training
 *
 * Provides utilities for spawning multiple processes for distributed
 * training, similar to PyTorch's torch.distributed.launch / torchrun.
 * Supports both fork-based and environment-based process spawning.
 */

#pragma once

#include "distributed.hpp"
#include <functional>
#include <string>
#include <vector>
#include <memory>

namespace tenzor::distributed {

/**
 * @brief Configuration for distributed training launch.
 */
struct LaunchConfig {
    /** @brief Number of processes to spawn (typically = number of GPUs) */
    int nproc_per_node{1};

    /** @brief Total number of nodes in the cluster */
    int nnodes{1};

    /** @brief Index of this node in the cluster (0-based) */
    int node_rank{0};

    /** @brief Address of the master node for rendezvous */
    std::string master_addr{"localhost"};

    /** @brief Port on the master node for rendezvous */
    int master_port{29500};

    /** @brief Communication backend to use ("nccl", "gloo") */
    std::string backend{"nccl"};

    /** @brief Whether to redirect worker stdout/stderr to log files */
    bool log_to_file{false};

    /** @brief Log directory for worker output (when log_to_file is true) */
    std::string log_dir{"/tmp/tenzor_dist_logs"};

    /**
     * @brief Compute total world size.
     * @return nproc_per_node * nnodes
     */
    auto world_size() const -> int {
        return nproc_per_node * nnodes;
    }

    /**
     * @brief Compute global rank for a local rank on this node.
     * @param local_rank Local rank on this node (0 to nproc_per_node-1)
     * @return Global rank
     */
    auto global_rank(int local_rank) const -> int {
        return node_rank * nproc_per_node + local_rank;
    }
};

/**
 * @brief Worker function signature for distributed training.
 *
 * @param rank Global rank of this process
 * @param world_size Total number of processes
 */
using WorkerFn = std::function<void(int rank, int world_size)>;

/**
 * @brief Spawn multiple processes for distributed training.
 *
 * Forks nproc_per_node child processes on the current node, each assigned
 * a unique rank. Each child process calls worker_fn with its global rank
 * and the world size. The parent process waits for all children to complete.
 *
 * Environment variables set for each child:
 * - RANK: Global rank
 * - LOCAL_RANK: Local rank on this node
 * - WORLD_SIZE: Total number of processes
 * - MASTER_ADDR: Master node address
 * - MASTER_PORT: Master node port
 *
 * @param config Launch configuration
 * @param worker_fn Function to execute in each worker process
 * @return Exit codes from all child processes (0 = success)
 *
 * @code
 * LaunchConfig config;
 * config.nproc_per_node = 4;
 * config.backend = "nccl";
 *
 * auto results = spawn(config, [](int rank, int world_size) {
 *     // Initialize distributed
 *     init_process_group("nccl", rank, world_size);
 *
 *     // Create model and DDP wrapper
 *     auto model = create_model();
 *     auto pg = DistributedContext::get_process_group();
 *     DistributedDataParallel ddp(*model, *pg);
 *
 *     // Training loop...
 *
 *     destroy_process_group();
 * });
 * @endcode
 */
auto spawn(const LaunchConfig& config, WorkerFn worker_fn) -> std::vector<int>;

/**
 * @brief Initialize distributed training from environment variables.
 *
 * Reads RANK, LOCAL_RANK, WORLD_SIZE, MASTER_ADDR, MASTER_PORT from
 * environment variables (as set by spawn() or external launchers like
 * torchrun) and initializes the distributed context.
 *
 * @param backend Communication backend ("nccl" or "gloo")
 *
 * @code
 * // When launched via: tenzor_launch --nproc_per_node=4 ./my_training
 * init_from_env("nccl");
 * int rank = get_rank();
 * int ws = get_world_size();
 * @endcode
 */
auto init_from_env(const std::string& backend = "nccl") -> void;

/**
 * @brief Get local rank (rank within this node).
 *
 * @return Local rank from LOCAL_RANK environment variable, or 0 if not set
 */
auto get_local_rank() -> int;

} // namespace tenzor::distributed
