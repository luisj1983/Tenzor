/**
 * @file sam.hpp
 * @brief Sharpness-Aware Minimization (SAM) optimizer wrapper
 *
 * SAM wraps any base optimizer and seeks parameters that lie in neighborhoods
 * with uniformly low loss, resulting in better generalization.
 *
 * Reference: Foret et al., "Sharpness-Aware Minimization for Efficiently
 * Improving Generalization", ICLR 2021.
 */

#pragma once

#include "optimizer.hpp"
#include <memory>

namespace tenzor {
namespace optim {

/**
 * @brief Sharpness-Aware Minimization (SAM) optimizer
 *
 * SAM wraps any base optimizer and performs a two-step update:
 *
 * 1. **first_step()**: Compute perturbation epsilon = rho * grad / ||grad||_2
 *    and add it to weights (moving to the worst-case neighborhood point)
 * 2. Caller does forward + backward at the perturbed point
 * 3. **second_step()**: Restore original weights and step the base optimizer
 *    using gradients computed at the perturbed point
 *
 * The perturbation is computed per-parameter, using the global gradient norm
 * across all parameters.
 *
 * @par Training Loop
 * @code
 * auto base_opt = std::make_shared<Adam>(model.parameters(), 1e-3);
 * SAM sam(base_opt, 0.05);
 *
 * for (auto& batch : dataloader) {
 *     // First forward-backward (standard)
 *     sam.zero_grad();
 *     auto loss = model.forward(batch);
 *     loss.backward();
 *
 *     // Perturb weights
 *     sam.first_step();
 *
 *     // Second forward-backward at perturbed point
 *     sam.zero_grad();
 *     auto loss2 = model.forward(batch);
 *     loss2.backward();
 *
 *     // Restore weights and step base optimizer
 *     sam.second_step();
 * }
 * @endcode
 *
 * @param base_optimizer Shared pointer to any Optimizer (Adam, SGD, etc.)
 * @param rho Perturbation radius (default: 0.05)
 *
 * @see Adam, SGD
 */
class SAM : public Optimizer {
public:
    /**
     * @brief Construct SAM wrapper around a base optimizer.
     *
     * @param base_optimizer The underlying optimizer that performs the actual step
     * @param rho Neighborhood size for perturbation (default: 0.05)
     */
    SAM(std::shared_ptr<Optimizer> base_optimizer, double rho = 0.05);

    /**
     * @brief Compute and apply weight perturbation.
     *
     * Computes epsilon = rho * grad / ||grad||_2 for each parameter and
     * adds it to the current weights. The original weights are saved
     * internally for restoration in second_step().
     *
     * @pre Gradients must be computed via backward()
     * @post Weights are perturbed; caller should do forward+backward again
     */
    auto first_step() -> void;

    /**
     * @brief Restore original weights and step the base optimizer.
     *
     * Subtracts the stored perturbation from current weights (restoring
     * the original values) and then calls the base optimizer's step()
     * using the gradients computed at the perturbed point.
     *
     * @pre first_step() must have been called, followed by forward+backward
     * @post Weights are updated by the base optimizer
     */
    auto second_step() -> void;

    /**
     * @brief Standard step (calls first_step then second_step with closure).
     *
     * This is the step_impl required by Optimizer base. For SAM, the
     * preferred usage is to call first_step() and second_step() explicitly.
     * Calling step() directly without a closure will throw.
     */
    auto step_impl() -> void override;

    /**
     * @brief Polymorphic SAM step using the standard Optimizer::step(closure)
     *        interface.
     *
     * SAM requires two forward+backward passes around a weight perturbation,
     * so we override step(closure) (rather than step_impl) to drive both:
     *   1. Run closure to get gradients at the original weights
     *   2. first_step() — perturb weights toward the gradient ascent direction
     *   3. Run closure again to get gradients at the perturbed weights
     *   4. second_step() — restore original weights and step the base optimizer
     *      using the gradients from the perturbed point
     *
     * Audit item D.7: previously SAM only worked via the manual
     * first_step()/second_step() API; calling the polymorphic
     * Optimizer::step(closure) threw at step_impl().  Now SAM behaves
     * polymorphically with the rest of the optimizer hierarchy.
     */
    auto step(std::function<Variable()> closure) -> Variable override;

    /** @brief Set learning rate on the base optimizer */
    auto set_lr(double lr) -> void override;

    /** @brief Get learning rate from the base optimizer */
    auto get_lr() const -> double override;

    /** @brief Get perturbation radius */
    auto get_rho() const -> double { return rho_; }

    /** @brief Set perturbation radius */
    auto set_rho(double rho) -> void { rho_ = rho; }

    /** @brief Get base optimizer state for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load base optimizer state */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

    /** @brief Get reference to the base optimizer */
    auto base_optimizer() -> Optimizer& { return *base_optimizer_; }
    auto base_optimizer() const -> const Optimizer& { return *base_optimizer_; }

private:
    std::shared_ptr<Optimizer> base_optimizer_;
    double rho_;
    std::vector<Tensor> epsilon_;  ///< Stored perturbations for each parameter
};

} // namespace optim
} // namespace tenzor
