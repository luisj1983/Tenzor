/**
 * @file zero_optimizer.hpp
 * @brief ZeRO Stage 1 Optimizer with state partitioning and CPU offload
 *
 * Implements DeepSpeed ZeRO Stage 1: Optimizer State Partitioning
 * across distributed ranks for memory-efficient training.
 *
 * @see https://arxiv.org/abs/1910.02054
 */

#pragma once

#include "optimizer.hpp"
#include "../../distributed/distributed.hpp"
#include "../../core/offload_engine.hpp"
#include <memory>
#include <vector>
#include <map>
#include <mutex>

namespace tenzor {
namespace optim {

/**
 * @brief Configuration for ZeRO Stage 1 Optimizer
 */
struct ZeROStage1Config {
    int world_size{1};                      ///< Number of distributed ranks
    int rank{0};                            ///< Current rank ID
    bool offload_to_cpu{false};             ///< Offload optimizer states to CPU
    size_t cpu_offload_threshold{1024};     ///< Min bytes to offload (default: 1KB)
    bool overlap_comm{true};                ///< Overlap communication with computation
    bool pin_memory{true};                  ///< Use pinned memory for transfers
    std::shared_ptr<distributed::ProcessGroup> process_group{nullptr}; ///< Communication group

    ZeROStage1Config() = default;
};

/**
 * @brief ZeRO Stage 1: Optimizer State Partitioning
 *
 * Partitions optimizer states (momentum, variance) across distributed ranks
 * to reduce memory usage by N-fold where N = world_size.
 *
 * **Algorithm**:
 * 1. Parameters are replicated on all ranks
 * 2. Optimizer states are partitioned (each rank owns 1/N)
 * 3. Gradients are all-reduced before optimizer step
 * 4. Each rank updates its parameter partition
 * 5. Parameters are all-gathered after update
 *
 * **Memory Savings**:
 * - Adam: 4x reduction in optimizer states (2 states per param)
 * - SGD with momentum: 2x reduction
 *
 * **CPU Offload**:
 * - Optionally offload optimizer states to CPU RAM
 * - States are fetched to GPU before update, then offloaded back
 * - Enables training models larger than GPU memory
 *
 * @code
 * // Example: Distributed training with ZeRO Stage 1
 * distributed::init_process_group("nccl");
 * auto rank = distributed::get_rank();
 * auto world_size = distributed::get_world_size();
 *
 * // Create base optimizer
 * auto adam = std::make_unique<Adam>(model.parameters(), 1e-3);
 *
 * // Wrap with ZeRO Stage 1
 * ZeROStage1Config config;
 * config.world_size = world_size;
 * config.rank = rank;
 * config.offload_to_cpu = true;
 * auto zero_optimizer = ZeROStage1Optimizer(std::move(adam), config);
 *
 * // Training loop
 * for (auto& batch : dataloader) {
 *     zero_optimizer.zero_grad();
 *     auto output = model.forward(batch.input);
 *     auto loss = criterion(output, batch.target);
 *     loss.backward();
 *     zero_optimizer.step();  // Handles all distributed communication
 * }
 * @endcode
 *
 * @see ZeROStage2Optimizer, ZeROStage3Optimizer
 */
class ZeROStage1Optimizer : public Optimizer {
public:
    /**
     * @brief Construct ZeRO Stage 1 optimizer
     *
     * @param base_optimizer Base optimizer (Adam, SGD, etc.) - ownership transferred
     * @param config ZeRO configuration
     * @throws std::invalid_argument if rank >= world_size or base_optimizer is null
     */
    ZeROStage1Optimizer(
        std::unique_ptr<Optimizer> base_optimizer,
        const ZeROStage1Config& config
    );

    /**
     * @brief Destructor - cleanup resources
     */
    ~ZeROStage1Optimizer() override;

    /**
     * @brief Perform optimizer step with distributed state partitioning
     *
     * Algorithm:
     * 1. All-reduce gradients across ranks (sum)
     * 2. If CPU offload: Fetch local state partition to GPU
     * 3. Update local parameter partition with base optimizer
     * 4. If CPU offload: Offload states back to CPU
     * 5. All-gather updated parameters across ranks
     *
     * @throws std::runtime_error if distributed not initialized
     */
    auto step() -> void override;

    /**
     * @brief Zero all parameter gradients
     */
    auto zero_grad() -> void;

    /**
     * @brief Get optimizer state dictionary
     *
     * Returns only the local partition of optimizer states.
     * Use save_checkpoint() to save full distributed state.
     *
     * @return Map of state variable names to tensors
     */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /**
     * @brief Load optimizer state dictionary
     *
     * Loads the local partition of optimizer states.
     * Use load_checkpoint() to load full distributed state.
     *
     * @param state State dictionary to load
     */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

    /**
     * @brief Save distributed checkpoint (all ranks)
     *
     * Each rank saves its partition. Master rank saves metadata.
     *
     * @param path_prefix Checkpoint path prefix (rank ID appended)
     */
    auto save_checkpoint(const std::string& path_prefix) const -> void;

    /**
     * @brief Load distributed checkpoint (all ranks)
     *
     * Each rank loads its partition. Master rank loads metadata.
     *
     * @param path_prefix Checkpoint path prefix
     */
    auto load_checkpoint(const std::string& path_prefix) -> void;

    // Accessors

    /**
     * @brief Get current rank ID
     */
    auto rank() const -> int { return config_.rank; }

    /**
     * @brief Get world size
     */
    auto world_size() const -> int { return config_.world_size; }

    /**
     * @brief Check if CPU offload is enabled
     */
    auto is_cpu_offload_enabled() const -> bool { return config_.offload_to_cpu; }

    /**
     * @brief Get number of parameters in local partition
     */
    auto local_param_count() const -> size_t;

    /**
     * @brief Get memory usage statistics
     */
    struct MemoryStats {
        size_t gpu_optimizer_memory{0};     ///< GPU memory for optimizer states (bytes)
        size_t cpu_optimizer_memory{0};     ///< CPU memory for optimizer states (bytes)
        size_t gpu_gradient_memory{0};      ///< GPU memory for gradients (bytes)
        size_t num_parameters{0};           ///< Total number of parameters
        size_t num_local_parameters{0};     ///< Parameters in local partition
    };

    /**
     * @brief Get memory usage statistics
     */
    auto get_memory_stats() const -> MemoryStats;

    /**
     * @brief Get base optimizer (const)
     */
    auto base_optimizer() const -> const Optimizer& {
        return *base_optimizer_;
    }

private:
    /**
     * @brief State partition for a single rank
     */
    struct StatePartition {
        int rank{0};                                ///< Rank that owns this partition
        std::vector<std::shared_ptr<Variable>> params;  ///< Parameters in partition
        std::vector<Tensor> momentum;               ///< Momentum states (if applicable)
        std::vector<Tensor> variance;               ///< Variance states (if applicable)
        Device device{Device::cpu()};               ///< Where states are stored
        size_t memory_bytes{0};                     ///< Total memory usage
    };

    // Core components
    std::unique_ptr<Optimizer> base_optimizer_;     ///< Wrapped optimizer
    ZeROStage1Config config_;                       ///< Configuration
    std::vector<StatePartition> partitions_;        ///< State partitions for all ranks
    std::shared_ptr<core::OffloadEngine> offload_engine_;  ///< CPU offload engine

    // Communication handles for async operations
    std::vector<Tensor> gradient_buffers_;          ///< Buffers for gradient all-reduce
    std::vector<Tensor> param_buffers_;             ///< Buffers for parameter all-gather

    // Synchronization
    mutable std::mutex mutex_;                      ///< Thread safety

    // Initialization

    /**
     * @brief Partition parameters across ranks
     *
     * Assigns each parameter to a rank based on parameter index.
     * Tries to balance memory across ranks.
     */
    auto partition_parameters() -> void;

    /**
     * @brief Initialize optimizer states for local partition
     *
     * Creates momentum/variance buffers for base optimizer.
     */
    auto initialize_optimizer_states() -> void;

    /**
     * @brief Initialize CPU offload engine (if enabled)
     */
    auto initialize_offload_engine() -> void;

    // Communication

    /**
     * @brief All-reduce gradients across ranks (sum)
     *
     * Synchronizes gradients before optimizer step.
     */
    auto all_reduce_gradients() -> void;

    /**
     * @brief All-gather parameters after update
     *
     * Reconstructs full parameter set from partitions.
     */
    auto all_gather_parameters() -> void;

    // State management

    /**
     * @brief Update local partition of optimizer states
     *
     * Calls base optimizer step() on local partition only.
     */
    auto update_local_partition() -> void;

    /**
     * @brief Fetch optimizer states from CPU to GPU (if offloaded)
     */
    auto fetch_states_to_gpu() -> void;

    /**
     * @brief Offload optimizer states from GPU to CPU (if enabled)
     */
    auto offload_states_to_cpu() -> void;

    /**
     * @brief Get local partition for current rank
     */
    auto local_partition() -> StatePartition& {
        return partitions_[config_.rank];
    }

    /**
     * @brief Get local partition for current rank (const)
     */
    auto local_partition() const -> const StatePartition& {
        return partitions_[config_.rank];
    }
};

} // namespace optim
} // namespace tenzor
