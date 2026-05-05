/**
 * @file parametrize.hpp
 * @brief General parameter reparameterization framework
 *
 * Provides a composable system for registering arbitrary parameter
 * transformations on nn::Module parameters. Multiple parametrizations
 * can be chained on the same parameter, and they are applied in order
 * during each forward pass.
 *
 * This is a generalization of spectral_norm and weight_norm. Those
 * existing utilities are standalone and remain compatible.
 *
 * Usage:
 * @code
 * // Create a symmetric positive definite parametrization
 * class Symmetric : public Parametrization {
 * public:
 *     auto forward(const Tensor& X) -> Tensor override {
 *         return matmul(X, transpose(X, 0, 1));  // X @ X^T
 *     }
 * };
 *
 * auto linear = std::make_shared<Linear>(4, 4);
 * register_parametrization(linear, "weight", std::make_shared<Symmetric>());
 * // linear->forward() now uses X @ X^T as the weight
 *
 * remove_parametrizations(linear, "weight");  // Restore original
 * @endcode
 */

#pragma once

#include "../../core/tensor.hpp"
#include "../module.hpp"
#include <string>
#include <memory>
#include <vector>
#include <unordered_map>

namespace tenzor::nn::utils {

/**
 * @brief Base class for parameter parametrizations.
 *
 * Subclass and override forward() to define a parameter transformation.
 * Optionally override right_inverse() to define how to initialize
 * the unconstrained parameter from a desired constrained value.
 */
class Parametrization : public Module {
public:
    ~Parametrization() override = default;

    /**
     * @brief Apply the parametrization to transform the parameter.
     *
     * @param X The unconstrained parameter value
     * @return The constrained/transformed parameter value
     */
    virtual auto forward(const Tensor& X) -> Tensor = 0;

    /**
     * @brief Compute the right inverse of the parametrization.
     *
     * Given a desired output Y, compute X such that forward(X) ≈ Y.
     * Default implementation returns Y unchanged (identity inverse).
     *
     * @param Y Desired constrained value
     * @return Unconstrained parameter value
     */
    virtual auto right_inverse(const Tensor& Y) -> Tensor {
        return Y;
    }

    /**
     * @brief Default forward_impl: applies forward(Tensor) and returns a
     * non-grad Variable.
     *
     * NOTE: Parametrizations are currently treated as non-trainable
     * transforms — gradients do NOT flow back from the parametrized output
     * into the unconstrained source parameter. This matches the existing
     * register_parametrization() implementation in parametrize.cpp, which
     * memcpys the parametrized value into the parameter buffer via a
     * forward pre-hook (so the autograd graph for the transform itself is
     * never built).
     *
     * If you need a Parametrization whose constrained output participates
     * in autograd (so gradients reach the unconstrained parameter), do NOT
     * rely on this base — register a forward pre-hook that produces a
     * Variable via Variable-level ops, mirroring weight_norm / spectral_norm.
     */
    auto forward_impl(const Variable& input) -> Variable override {
        return Variable(forward(input.tensor()), false);
    }
};

/**
 * @brief Registry of parametrizations applied to a module.
 *
 * Manages the chain of parametrizations for each parameter name.
 * Stored as a member of the parametrized module.
 */
struct ParametrizationList {
    /** @brief Original parameter value (before any parametrization) */
    Tensor original;

    /** @brief Chain of parametrizations applied in order */
    std::vector<std::shared_ptr<Parametrization>> chain;

    /** @brief Hook ID for the forward pre-hook */
    size_t hook_id{0};
};

/**
 * @brief Register a parametrization on a module's parameter.
 *
 * The original parameter is stored as "{name}_orig" and the
 * parametrization chain is applied via a forward pre-hook.
 * Multiple parametrizations can be registered on the same parameter;
 * they are composed in registration order.
 *
 * @param module Module containing the parameter
 * @param param_name Name of the parameter to parametrize
 * @param parametrization Parametrization to apply
 */
void register_parametrization(std::shared_ptr<Module> module,
                              const std::string& param_name,
                              std::shared_ptr<Parametrization> parametrization);

/**
 * @brief Remove all parametrizations from a module's parameter.
 *
 * @param module Module containing the parametrized parameter
 * @param param_name Name of the parameter
 * @param leave_parametrized If true (default), leaves the parameter at its
 *        current parametrized value. If false, restores the original value.
 */
void remove_parametrizations(std::shared_ptr<Module> module,
                             const std::string& param_name,
                             bool leave_parametrized = true);

/**
 * @brief Check if a module has parametrizations on any or a specific parameter.
 *
 * @param module Module to check
 * @param param_name If empty, checks any parameter; otherwise checks specific name
 * @return true if parametrizations are registered
 */
auto is_parametrized(const Module& module,
                     const std::string& param_name = "") -> bool;

/**
 * @brief Clear the entire parametrization registry. Test-only helper —
 *        the registry keys on raw Module* and stale entries can survive
 *        Module destruction and pollute later modules that reuse the same
 *        address. Call between test cases.
 */
void clear_parametrization_registry();

} // namespace tenzor::nn::utils
