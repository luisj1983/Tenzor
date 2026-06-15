/**
 * @file data_parallel.hpp
 * @brief DataParallel for multi-GPU training support
 *
 * Implements data parallelism across multiple GPUs by replicating the model,
 * splitting input batches, and synchronizing gradients.
 */

#pragma once

#include <functional>
#include <vector>
#include <memory>
#include <mutex>
#include "../../nn/module.hpp"
#include "../../autograd/variable.hpp"
#include "../../core/device.hpp"
#include "../../distributed/distributed.hpp"  // ReduceOp

// Forward declaration — A1: real replica creation via factory + PG all_reduce.
namespace tenzor::distributed { class ProcessGroupBase; }

namespace tenzor {
namespace nn {

/**
 * @brief Data parallelism wrapper for multi-GPU training.
 *
 * Replicates a module across multiple GPUs, splits input batches, and
 * synchronizes gradients during backward pass for efficient multi-GPU training.
 *
 * Algorithm:
 * 1. Replicate model to all specified GPUs
 * 2. Split input batch into chunks (one per GPU)
 * 3. Scatter chunks to respective GPUs
 * 4. Execute forward pass in parallel on each GPU
 * 5. Gather outputs to master GPU
 * 6. Backward pass computes gradients on each GPU
 * 7. All-reduce gradients and average across GPUs
 * 8. Optimizer updates master model parameters
 *
 * Design Pattern: Decorator, SPMD (Single Program Multiple Data)
 * Thread Safety: Forward/backward passes use GPU streams for parallelism
 *
 * Example:
 * @code
 * // Create model
 * auto model = std::make_shared<MyModel>();
 *
 * // Wrap with DataParallel for GPUs 0, 1, 2, 3
 * auto parallel_model = std::make_shared<DataParallel>(
 *     model,
 *     std::vector<int>{0, 1, 2, 3},  // device_ids
 *     0  // output_device (master GPU)
 * );
 *
 * // Training loop (no changes needed)
 * for (auto& batch : dataloader) {
 *     Variable output = parallel_model->forward(batch.input);
 *     Variable loss = criterion(output, batch.target);
 *     loss.backward();
 *     optimizer.step();
 * }
 * @endcode
 */
/**
 * @brief Module factory — returns a fresh instance of the user's model.
 *
 * A1/B5: invoked by DataParallel during replica construction so each device
 * gets an independent module instance with its own parameter tensors. The
 * factory is expected to construct the module identically on each call
 * (same architecture, same default-initialized parameters); DataParallel
 * then synchronizes initial state across replicas by broadcasting from the
 * master.
 */
using ModuleFactory = std::function<std::shared_ptr<Module>()>;

class DataParallel : public Module {
public:
    /**
     * @brief Construct DataParallel wrapper.
     *
     * @param module Module to replicate across GPUs
     * @param device_ids List of GPU device IDs to use
     * @param output_device Master GPU device ID (default: device_ids[0])
     * @param dim Batch dimension to split (default: 0)
     * @throws std::runtime_error if device_ids is empty or devices unavailable
     */
    DataParallel(
        std::shared_ptr<Module> module,
        std::vector<int> device_ids = {},
        int output_device = -1,
        int dim = 0,
        ::tenzor::distributed::ReduceOp reduce_op = ::tenzor::distributed::ReduceOp::AVG
    );

    /**
     * @brief Construct DataParallel with a real replication factory and/or a
     *        ProcessGroup for gradient all-reduce.
     *
     * The legacy ctor leaves `module_factory_` and `pg_` null, so replicate()
     * aliases one shared module across devices and the PG all-reduce path
     * (and MAX/MIN reductions) are dead. This ctor wires both, enabling true
     * per-device replicas and real all-reduce.
     *
     * @param module Master module to replicate
     * @param module_factory Factory that materializes an independent replica
     * @param pg Optional process group (NCCL/Gloo/MPI) for grad all-reduce
     * @param device_ids List of GPU device IDs to use
     * @param output_device Master GPU device ID (default: device_ids[0])
     * @param dim Batch dimension to split (default: 0)
     * @param reduce_op Gradient reduction op
     */
    DataParallel(
        std::shared_ptr<Module> module,
        ModuleFactory module_factory,
        std::shared_ptr<::tenzor::distributed::ProcessGroupBase> pg,
        std::vector<int> device_ids = {},
        int output_device = -1,
        int dim = 0,
        ::tenzor::distributed::ReduceOp reduce_op = ::tenzor::distributed::ReduceOp::AVG
    );

    /**
     * @brief Destructor.
     */
    ~DataParallel() override = default;

    /**
     * @brief Forward pass with data parallelism.
     *
     * Steps:
     * 1. Validate input is on master device
     * 2. Replicate module to all devices (if needed)
     * 3. Split input along batch dimension
     * 4. Scatter input chunks to devices
     * 5. Execute forward pass in parallel on each device
     * 6. Gather outputs to master device
     * 7. Concatenate outputs along batch dimension
     *
     * @param input Input variable (must be on master device)
     * @return Output variable on master device
     * @throws std::runtime_error if batch_size < num_devices
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Get underlying module.
     *
     * @return Shared pointer to original module
     */
    auto module() -> std::shared_ptr<Module> { return module_; }

    /**
     * @brief Get device IDs.
     *
     * @return Vector of GPU device IDs
     */
    auto device_ids() const -> const std::vector<int>& { return device_ids_; }

    /**
     * @brief Get master device ID.
     *
     * @return Output device ID
     */
    auto output_device() const -> int { return output_device_; }

    /**
     * @brief Get batch dimension.
     *
     * @return Dimension index to split batches
     */
    auto batch_dim() const -> int { return dim_; }

    /**
     * @brief Override parameters to return master module params.
     *
     * @return Vector of parameter shared_ptrs from master module
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Override named_parameters for master module.
     *
     * @return Vector of (name, parameter shared_ptr) pairs from master module
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

    /**
     * @brief Set module to training mode.
     *
     * @param mode Training mode flag
     */
    auto train(bool mode = true) -> void override;

    /**
     * @brief Set module to evaluation mode.
     */
    auto eval() -> void override;

    /**
     * @brief Reduce per-replica gradients onto the master parameters.
     *
     * Most paths reduce gradients automatically during backward() via the
     * hooks installed by `register_grad_hooks()` (shared-module accumulation
     * and the ProcessGroup all-reduce), and calling this method on those
     * paths is a no-op.
     *
     * The factory/independent-replica path WITHOUT a ProcessGroup is the
     * exception: each replica owns its own parameter Variables, so their
     * gradients are invisible to the master after backward(). For that path
     * the training loop MUST call this method after `loss.backward()` and
     * before `optimizer.step()`. It sums every replica's per-parameter
     * gradient onto the corresponding master parameter (positional
     * correspondence established in `replicate()`), then normalizes per the
     * configured `reduce_op_` (AVG divides by the number of replicas; SUM
     * leaves the sum unchanged), so the optimizer steps on the correct
     * full-batch gradient.
     */
    auto synchronize_gradients() -> void;

private:
    std::shared_ptr<Module> module_;           ///< Original module (master)
    std::vector<int> device_ids_;              ///< GPU device IDs
    int output_device_;                        ///< Master GPU device ID
    int dim_;                                  ///< Batch dimension to split

    // Replicated modules (one per device)
    std::vector<std::shared_ptr<Module>> replicas_;
    bool replicas_initialized_{false};

    // Parameters to synchronize gradients across devices
    std::vector<std::shared_ptr<Variable>> parameters_to_sync_;

    // Per-replica parameter lists (factory/independent-replica path only),
    // aligned 1:1 with replicas_/device_ids_. The master-device slot stores an
    // empty list because it aliases module_ (its grad is the master parameter's
    // own grad). Each non-master slot holds that replica's parameters() in the
    // same order as parameters_to_sync_, giving positional replica->master
    // correspondence for synchronize_gradients(). Populated by replicate().
    std::vector<std::vector<std::shared_ptr<Variable>>> replica_parameters_;

    // True only for the factory/independent-replica path WITHOUT a ProcessGroup,
    // where synchronize_gradients() must explicitly sum replica grads onto the
    // master (the per-parameter hooks cannot, and would mis-normalize). Set in
    // replicate().
    bool independent_replica_sync_{false};

    mutable std::mutex replicas_mutex_;        ///< Protect replica creation

    // A1/B5: optional factory (set by the factory ctor) used to materialize
    // independent replicas during `replicate()`. When null, the legacy
    // shared-module path is used (the old ctor).
    ModuleFactory module_factory_;

    // A1/B5: optional process group for real grad all_reduce in
    // `synchronize_gradients`. When null, the legacy `grad *= 1/N` workaround
    // is used (the old ctor's behavior).
    std::shared_ptr<::tenzor::distributed::ProcessGroupBase> pg_;

    // Reduction op for gradient synchronization across replicas.
    // SUM:  raw sum of grads (no normalization) — total gradient.
    // AVG / MEAN: sum then divide by num_devices_ — mean gradient (default,
    //             matches the original implicit behavior of `grad *= 1/N`).
    // MAX / MIN: element-wise max / min across replicas (requires a real PG).
    // PRODUCT, BAND, BOR, BXOR: rejected — undefined semantics for gradients.
    ::tenzor::distributed::ReduceOp reduce_op_{::tenzor::distributed::ReduceOp::AVG};

    // Hook IDs registered on the master module's parameters so that
    // `synchronize_gradients` is invoked automatically when each parameter's
    // gradient lands in the autograd engine. Indexed in lockstep with
    // parameters_to_sync_; populated by register_grad_hooks().
    std::vector<size_t> grad_hook_ids_;

    /**
     * @brief Replicate module to all devices.
     *
     * Creates shallow copies of the module on each device, sharing
     * parameter references with the master module.
     */
    auto replicate() -> void;

    /**
     * @brief Scatter input tensor to multiple devices.
     *
     * Splits input along specified dimension and copies chunks to devices.
     *
     * @param input Input tensor on master device
     * @return Vector of input chunks (one per device)
     * @throws std::runtime_error if split is not possible
     */
    auto scatter(const Variable& input) -> std::vector<Variable>;

    /**
     * @brief Execute forward pass in parallel.
     *
     * Launches forward computation on each device using CUDA streams
     * for concurrent execution.
     *
     * @param inputs Input chunks (one per device)
     * @return Output chunks (one per device)
     */
    auto parallel_apply(const std::vector<Variable>& inputs) -> std::vector<Variable>;

    /**
     * @brief Gather outputs from all devices.
     *
     * Concatenates output chunks from all devices back to master device.
     *
     * @param outputs Output chunks from each device
     * @return Concatenated output on master device
     */
    auto gather(const std::vector<Variable>& outputs) -> Variable;

    /**
     * @brief Wire automatic gradient sync into the autograd engine.
     *
     * Registers a `Variable::register_hook` on every parameter so that the
     * configured `reduce_op_` all-reduce runs as soon as the parameter's
     * gradient is computed during backward(). Replaces the requirement to
     * call `synchronize_gradients()` manually after `loss.backward()`.
     *
     * Called once from `replicate()` after `parameters_to_sync_` is
     * populated; subsequent forward passes reuse the already-registered
     * hooks. Safe to call multiple times (no-op if already wired).
     */
    auto register_grad_hooks() -> void;

    /**
     * @brief Validate device availability.
     *
     * @throws std::runtime_error if CUDA not available or device invalid
     */
    auto validate_devices() -> void;

    /**
     * @brief Validate reduce_op_ against the available reduction path.
     *
     * Without a ProcessGroup only SUM/AVG are realizable; MAX/MIN/PRODUCT/etc.
     * are rejected at construction rather than throwing on first backward().
     */
    auto validate_reduce_op() const -> void;

    /**
     * @brief Check if input batch can be split.
     *
     * @param batch_size Size of input batch
     * @return true if batch_size >= num_devices
     */
    auto can_split_batch(int64_t batch_size) const -> bool;
};

/**
 * @brief Helper function to create DataParallel module.
 *
 * Convenience function that auto-detects available GPUs if device_ids
 * is not provided.
 *
 * @param module Module to parallelize
 * @param device_ids GPU device IDs (empty = auto-detect all available)
 * @param output_device Master GPU (default: -1 = use device_ids[0])
 * @return Shared pointer to DataParallel wrapper
 *
 * @code
 * auto model = std::make_shared<ResNet50>();
 * auto parallel_model = make_data_parallel(model);  // Use all GPUs
 * @endcode
 */
auto make_data_parallel(
    std::shared_ptr<Module> module,
    std::vector<int> device_ids = {},
    int output_device = -1
) -> std::shared_ptr<DataParallel>;

} // namespace nn
} // namespace tenzor
