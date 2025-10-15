/**
 * @file data_parallel.hpp
 * @brief DataParallel for multi-GPU training support
 *
 * Implements data parallelism across multiple GPUs by replicating the model,
 * splitting input batches, and synchronizing gradients.
 */

#pragma once

#include <vector>
#include <memory>
#include <mutex>
#include "../../nn/module.hpp"
#include "../../autograd/variable.hpp"
#include "../../core/device.hpp"

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
    auto forward(const Variable& input) -> Variable override;

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
