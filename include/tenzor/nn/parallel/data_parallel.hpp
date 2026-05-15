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

/**
 * @brief Tag type that disambiguates the factory-based DataParallel ctor
 *        from the shared-module ctor (otherwise `DataParallel(nullptr, ...)`
 *        is ambiguous because both `shared_ptr<Module>` and `std::function`
 *        accept implicit conversion from nullptr_t).
 *
 * Usage: `DataParallel(use_factory, [&]() { return ...; }, {0,1,2,3})`.
 */
struct UseFactoryTag {};
inline constexpr UseFactoryTag use_factory{};

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
        int dim = 0
    );

    /**
     * @brief Construct DataParallel via a module factory (A1/B5).
     *
     * @param tag `tenzor::nn::use_factory` — disambiguates this ctor from
     *            the shared-module ctor (so `DataParallel(nullptr, ...)`
     *            stays unambiguous).
     * @param factory Callable that returns a fresh module instance — called
     *                once per device to materialize an independent replica.
     *                The user is responsible for ensuring the factory builds
     *                the same architecture on each call.
     * @param device_ids List of GPU device IDs to use.
     * @param output_device Master GPU device ID (default: device_ids[0]).
     * @param dim Batch dimension to split (default: 0).
     * @param pg Optional process group for gradient all_reduce (B5). When
     *           non-null, `synchronize_gradients` performs a real all_reduce
     *           on each parameter's gradient and divides by world_size,
     *           replacing the `grad *= 1/N` workaround.
     *
     * Compared to the constructor that takes a single shared `module`, this
     * variant produces *independent* per-device replicas (the factory is
     * called N times) and is the recommended path for new code.
     */
    DataParallel(
        UseFactoryTag tag,
        ModuleFactory factory,
        std::vector<int> device_ids,
        int output_device = -1,
        int dim = 0,
        std::shared_ptr<::tenzor::distributed::ProcessGroupBase> pg = nullptr
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
    auto train(bool mode = true) -> void;

    /**
     * @brief Set module to evaluation mode.
     */
    auto eval() -> void;

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

    mutable std::mutex replicas_mutex_;        ///< Protect replica creation

    // A1/B5: optional factory (set by the factory ctor) used to materialize
    // independent replicas during `replicate()`. When null, the legacy
    // shared-module path is used (the old ctor).
    ModuleFactory module_factory_;

    // A1/B5: optional process group for real grad all_reduce in
    // `synchronize_gradients`. When null, the legacy `grad *= 1/N` workaround
    // is used (the old ctor's behavior).
    std::shared_ptr<::tenzor::distributed::ProcessGroupBase> pg_;

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
     * @brief Synchronize gradients across devices.
     *
     * After backward pass, averages gradients from all replicas
     * into the master module's parameters.
     *
     * Called automatically during backward pass via autograd hooks.
     */
    auto synchronize_gradients() -> void;

    /**
     * @brief Validate device availability.
     *
     * @throws std::runtime_error if CUDA not available or device invalid
     */
    auto validate_devices() -> void;

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
