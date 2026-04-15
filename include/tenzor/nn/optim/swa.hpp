/**
 * @file swa.hpp
 * @brief Stochastic Weight Averaging (SWA) utilities
 *
 * Provides AveragedModel for maintaining a running average of model parameters,
 * and SWALR scheduler that holds a constant learning rate after an optional
 * warmup period.
 *
 * Reference: Izmailov et al., "Averaging Weights Leads to Wider Optima and
 * Better Generalization", UAI 2018.
 */

#pragma once

#include "optimizer.hpp"
#include "scheduler.hpp"
#include <vector>
#include <memory>

namespace tenzor {
namespace optim {

/**
 * @brief Maintains a running average of model parameters for SWA.
 *
 * AveragedModel stores a separate copy of parameters and updates them
 * as a running mean each time update_parameters() is called:
 *
 * \f[
 * \bar{\theta}_{n+1} = \frac{\bar{\theta}_n \cdot n + \theta}{n + 1}
 * \f]
 *
 * **Typical Usage:**
 * 1. Train with a base optimizer for a warm-up phase
 * 2. Switch to SWA learning rate
 * 3. Call update_parameters() at the end of each epoch
 * 4. After training, use the averaged parameters for evaluation
 *
 * @code
 * // Create averaged model from current parameters
 * AveragedModel avg_model(model.parameters());
 *
 * for (int epoch = swa_start; epoch < total_epochs; ++epoch) {
 *     train_one_epoch();
 *     avg_model.update_parameters(model.parameters());
 * }
 *
 * // Copy averaged parameters back to model
 * avg_model.apply_to(model.parameters());
 * @endcode
 *
 * @see SWALR
 */
class AveragedModel {
public:
    /**
     * @brief Construct from a list of model parameters.
     *
     * Clones the current parameter values as the initial average.
     *
     * @param params Model parameters to average
     */
    explicit AveragedModel(const std::vector<std::shared_ptr<Variable>>& params);

    /**
     * @brief Update the running average with current model parameters.
     *
     * Computes: avg = (avg * n + param) / (n + 1) for each parameter,
     * then increments the averaging counter.
     *
     * @param params Current model parameters
     */
    auto update_parameters(const std::vector<std::shared_ptr<Variable>>& params) -> void;

    /**
     * @brief Copy averaged parameters back to model parameters.
     *
     * Overwrites the tensor data in the given parameters with the
     * averaged values.
     *
     * @param params Model parameters to overwrite
     */
    auto apply_to(std::vector<std::shared_ptr<Variable>>& params) const -> void;

    /**
     * @brief Get the averaged parameter tensors.
     * @return Const reference to the averaged tensors
     */
    auto averaged_params() const -> const std::vector<Tensor>& { return averaged_params_; }

    /**
     * @brief Get the number of times parameters have been averaged.
     * @return Number of update_parameters() calls
     */
    auto n_averaged() const -> int64_t { return n_averaged_; }

private:
    std::vector<Tensor> averaged_params_;  ///< Running averaged parameter values
    int64_t n_averaged_{0};                ///< Number of models averaged so far
};

/**
 * @brief SWA Learning Rate scheduler
 *
 * Anneals the learning rate to a fixed swa_lr value. Optionally performs
 * a linear warmup from the current optimizer LR to swa_lr over
 * anneal_epochs epochs, then holds swa_lr constant.
 *
 * \f[
 * \eta_t = \begin{cases}
 * \eta_{current} + \frac{t}{T_{anneal}} (\eta_{swa} - \eta_{current}) & t < T_{anneal} \\
 * \eta_{swa} & t \geq T_{anneal}
 * \end{cases}
 * \f]
 *
 * @param optimizer Optimizer whose LR will be adjusted
 * @param swa_lr Target SWA learning rate
 * @param anneal_epochs Number of epochs for linear annealing (default: 10)
 * @param anneal_strategy "linear" or "cos" annealing (default: "linear")
 *
 * @code
 * auto optimizer = Adam(model.parameters(), 1e-3);
 * SWALR swa_scheduler(optimizer, 0.05, 5);  // Anneal to 0.05 over 5 epochs
 *
 * for (int epoch = 0; epoch < total_epochs; ++epoch) {
 *     if (epoch >= swa_start) {
 *         swa_scheduler.step();
 *     }
 * }
 * @endcode
 *
 * @see AveragedModel
 */
class SWALR : public LRScheduler {
public:
    /**
     * @brief Construct SWALR scheduler.
     *
     * @param optimizer Optimizer to schedule
     * @param swa_lr Target constant learning rate
     * @param anneal_epochs Epochs to anneal from current LR to swa_lr (default: 10)
     * @param anneal_strategy "linear" or "cos" (default: "linear")
     */
    SWALR(Optimizer& optimizer, double swa_lr,
          int anneal_epochs = 10,
          const std::string& anneal_strategy = "linear");

    /** @brief Step the scheduler (typically once per epoch) */
    auto step() -> void override;

    /** @brief Get the last computed learning rate */
    auto get_last_lr() const -> double override { return last_lr_; }

private:
    Optimizer& optimizer_;
    double swa_lr_;
    double initial_lr_;
    double last_lr_;
    int anneal_epochs_;
    std::string anneal_strategy_;
    int epoch_{0};
};

} // namespace optim
} // namespace tenzor
