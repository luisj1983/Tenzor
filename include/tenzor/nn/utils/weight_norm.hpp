/**
 * @file weight_norm.hpp
 * @brief Weight normalization for neural network layers
 *
 * Implements weight normalization (Salimans & Kingma, 2016) which reparameterizes
 * weight vectors as w = g * (v / ||v||), decoupling magnitude (g) from direction (v).
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "../../core/tensor.hpp"
#include "../module.hpp"

namespace tenzor::nn::utils {

/**
 * @brief Weight normalization wrapper for a module parameter.
 *
 * Decomposes a weight parameter into magnitude (g) and direction (v):
 *   w = g * (v / ||v||)
 *
 * Registers a forward pre-hook that recomputes the weight from g and v.
 *
 * @code
 * auto linear = std::make_shared<Linear>(128, 64);
 * auto wn = WeightNorm::apply(linear, "weight");
 * // linear->forward() now uses weight-normalized weights
 * wn->remove();  // Restore original weight
 * @endcode
 */
class WeightNorm {
public:
    /**
     * @brief Apply weight normalization to a module parameter.
     *
     * @param module Target module
     * @param name Parameter name to normalize (default: "weight")
     * @param dim Dimension over which to compute the norm (default: 0)
     * @return Shared pointer to WeightNorm instance (for later removal)
     */
    static auto apply(std::shared_ptr<Module> module,
                      const std::string& name = "weight",
                      int64_t dim = 0) -> std::shared_ptr<WeightNorm>;

    /**
     * @brief Remove weight normalization and restore the original weight.
     */
    auto remove() -> void;

    ~WeightNorm() = default;

private:
    WeightNorm(std::shared_ptr<Module> module, std::string name, int64_t dim);

    /**
     * @brief Compute the normalized weight: w = g * (v / ||v||)
     */
    auto compute_weight() -> Tensor;

    /**
     * @brief Compute the normalized weight as a Variable (autograd-tracked).
     *
     * w = g * (v / (||v|| + eps))
     *
     * The returned Variable carries a grad_fn back to `weight_g_` and
     * `weight_v_`, so gradients flow into the registered g/v parameters
     * on backward(). This is the value the forward pre-hook writes into
     * the layer's `weight` slot every forward pass.
     */
    auto compute_weight_variable() -> Variable;

    std::shared_ptr<Module> module_;
    std::shared_ptr<Variable> weight_g_;     ///< Magnitude parameter (leaf, registered)
    std::shared_ptr<Variable> weight_v_;     ///< Direction parameter (leaf, registered)
    std::shared_ptr<Variable> weight_slot_;  ///< Layer's weight slot — pre-hook writes derived Variable here
    std::string param_name_;
    int64_t dim_;
    size_t hook_id_{0};

public:
    /// Accessors used by tests / introspection. (Public so tests can verify
    /// the slot is being mutated by the pre-hook.)
    auto weight_g() const -> std::shared_ptr<Variable> { return weight_g_; }
    auto weight_v() const -> std::shared_ptr<Variable> { return weight_v_; }
};

} // namespace tenzor::nn::utils
