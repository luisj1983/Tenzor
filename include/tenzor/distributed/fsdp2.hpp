/**
 * @file fsdp2.hpp
 * @brief Fully Sharded Data Parallel v2 (FSDP2) with per-parameter DTensor sharding
 *
 * Unlike FSDP v1 which flattens all parameters into a single contiguous buffer,
 * FSDP2 uses DTensor to shard each parameter independently on a named mesh
 * dimension. This enables composability with tensor parallelism and other
 * parallelism strategies expressed through the DeviceMesh.
 *
 * Key differences from FSDP v1:
 * - Per-parameter sharding via DTensor (no flat-parameter concatenation)
 * - Uses DeviceMesh for flexible multi-dimensional parallelism
 * - Mixed precision expressed per-parameter through DTensor placements
 * - Cleaner composability with TP and PP through mesh dimensions
 */

#pragma once

#include "dtensor.hpp"
#include "device_mesh.hpp"
#include "distributed.hpp"
#include "../nn/module.hpp"
#include "../autograd/variable.hpp"
#include <memory>
#include <functional>
#include <vector>
#include <string>
#include <unordered_map>

namespace tenzor::distributed {

/**
 * @brief Mixed precision configuration for FSDP2.
 *
 * Controls the dtypes used at each stage of the FSDP2 training loop.
 */
struct MixedPrecisionPolicy {
    /** @brief Dtype for parameter storage (master weights) */
    DType param_dtype = DType::Float32;

    /** @brief Dtype for gradient reduction (all-reduce / reduce-scatter) */
    DType reduce_dtype = DType::Float32;

    /** @brief Dtype for buffers (running mean, etc.) */
    DType buffer_dtype = DType::Float32;
};

/**
 * @brief Configuration for FSDP2 wrapping.
 */
struct FSDP2Config {
    /** @brief Device mesh for sharding */
    std::shared_ptr<DeviceMesh> mesh;

    /** @brief Which mesh dimension to shard parameters on */
    std::string shard_mesh_dim = "dp";

    /** @brief Whether to free full params after forward (saves memory, costs re-gather in backward) */
    bool reshard_after_forward = true;

    /** @brief Mixed precision policy */
    MixedPrecisionPolicy mixed_precision{};

    /** @brief Offload sharded parameters to CPU when not in use */
    bool cpu_offload = false;

    /** @brief Prefetch the next FSDP unit's params during forward */
    bool forward_prefetch = true;

    /** @brief Prefetch params for backward during forward */
    bool backward_prefetch = true;

    /** @brief Wrap the wrapped module's forward in autograd::checkpoint to recompute
     *         activations on backward instead of saving them.
     *
     *  This is the activation-memory complement of FSDP2's parameter sharding: param sharding
     *  cuts model-state memory by world_size; activation checkpointing cuts the *activations*
     *  (the dominant memory consumer for transformers above ~3B params) by trading a single
     *  extra forward pass per backward. Together they're how DeepSpeed/FSDP train multi-tens-
     *  of-billions-of-params on a single node.
     *
     *  Memory impact: activations between this FSDP2's input and output are not saved during
     *  forward; only the input is saved. Recomputed on first backward call. Typical savings
     *  for transformer blocks: 4-8× activation memory at the cost of ~33% extra forward FLOPs
     *  on backward.
     *
     *  Default off — activation memory is rarely the bottleneck below ~1B params.
     */
    bool activation_checkpointing = false;
};

/**
 * @brief Fully Sharded Data Parallel v2 with DTensor-based per-parameter sharding.
 *
 * Wraps an nn::Module and shards each parameter as a DTensor along the specified
 * mesh dimension. During forward, parameters are all-gathered (unsharded) for
 * computation, then optionally resharded to free memory. During backward,
 * gradients are reduce-scattered so each rank holds only its gradient shard.
 *
 * Usage:
 * @code
 * // Create a 2D mesh: 4-way data parallel, 2-way tensor parallel
 * auto mesh = std::make_shared<DeviceMesh>(
 *     Device::Type::CUDA, std::vector<int64_t>{4, 2}, {"dp", "tp"});
 *
 * auto model = std::make_shared<TransformerBlock>();
 *
 * FSDP2Config config;
 * config.mesh = mesh;
 * config.shard_mesh_dim = "dp";
 * config.reshard_after_forward = true;
 *
 * FSDP2 fsdp(model, config);
 * fsdp.shard_parameters();
 *
 * // Training loop
 * auto output = fsdp.forward(input);
 * auto loss = criterion(output, target);
 * loss.backward();
 * fsdp.backward_hook();
 * optimizer.step();
 * @endcode
 */
class FSDP2 {
public:
    /**
     * @brief Construct FSDP2 wrapper for a module.
     *
     * @param module Module to wrap (shared ownership)
     * @param config FSDP2 configuration with mesh and sharding options
     * @throws std::invalid_argument if mesh is null or shard_mesh_dim is not
     *         a valid dimension name in the mesh
     */
    explicit FSDP2(std::shared_ptr<nn::Module> module, FSDP2Config config);

    ~FSDP2();

    // Non-copyable
    FSDP2(const FSDP2&) = delete;
    FSDP2& operator=(const FSDP2&) = delete;

    // Movable
    FSDP2(FSDP2&&) noexcept = default;
    FSDP2& operator=(FSDP2&&) noexcept = default;

    /**
     * @brief Shard all module parameters as DTensors along the dp mesh dimension.
     *
     * Iterates over all parameters in the wrapped module and replaces each
     * with a DTensor sharded along dimension 0 on the configured mesh dimension.
     * In single-process mode, this is a no-op (parameters are left as-is).
     */
    auto shard_parameters() -> void;

    /**
     * @brief Forward pass with automatic parameter unsharding.
     *
     * For each sharded parameter:
     * 1. All-gather to reconstruct the full parameter (unshard)
     * 2. Run the module's forward pass
     * 3. If reshard_after_forward, free full params and keep only local shards
     *
     * @param input Input variable
     * @return Output variable from the wrapped module
     */
    auto forward(const Variable& input) -> Variable;

    /**
     * @brief Finalize backward pass by reduce-scattering gradients.
     *
     * After loss.backward() has been called, this reduce-scatters each
     * parameter's gradient so that each rank holds only its gradient shard
     * (matching the parameter sharding). The gradient is also averaged
     * across the shard group.
     *
     * In single-process mode, this is a no-op.
     */
    auto backward_hook() -> void;

    /**
     * @brief Get reference to the wrapped module.
     */
    auto module() -> nn::Module& { return *module_; }
    auto module() const -> const nn::Module& { return *module_; }

    /**
     * @brief Get all sharded parameters as DTensors.
     *
     * Useful for passing to an optimizer that understands DTensor sharding,
     * enabling each rank to update only its local shard.
     *
     * @return Vector of sharded DTensor parameters
     */
    auto sharded_parameters() -> std::vector<DTensor>;

    /**
     * @brief Summon full (unsharded) parameters temporarily.
     *
     * All-gathers all parameters so they can be inspected or saved.
     * Call release_full_params() when done.
     */
    auto summon_full_params() -> void;

    /**
     * @brief Release full parameters and return to sharded state.
     */
    auto release_full_params() -> void;

    /**
     * @brief Get the FSDP2 configuration.
     */
    auto config() const -> const FSDP2Config& { return config_; }

private:
    std::shared_ptr<nn::Module> module_;
    FSDP2Config config_;

    /** @brief Sharded DTensor for each parameter, keyed by parameter name */
    std::vector<std::pair<std::string, DTensor>> sharded_params_;

    /** @brief Whether parameters are currently in full (unsharded) state */
    bool params_unsharded_ = false;

    /** @brief Whether shard_parameters() has been called */
    bool is_sharded_ = false;

    /** @brief Shard size along the sharding mesh dimension */
    int64_t shard_world_size_ = 1;

    /** @brief This rank's index along the sharding mesh dimension */
    int64_t shard_rank_ = 0;

    /**
     * @brief Unshard (all-gather) all parameters to reconstruct full tensors.
     */
    auto unshard_params() -> void;

    /**
     * @brief Reshard parameters: free full tensors, keep only local shards.
     */
    auto reshard_params() -> void;

    /**
     * @brief Cast a tensor to the specified dtype if different.
     */
    static auto maybe_cast(const Tensor& tensor, DType target) -> Tensor;
};

} // namespace tenzor::distributed
