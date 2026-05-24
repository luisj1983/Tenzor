/**
 * @file lazy_linear.hpp
 * @brief Lazy fully connected (linear) neural network layer
 *
 * Implements a lazy linear layer that defers weight initialization until
 * the first forward pass, when the input feature dimension becomes known.
 * This is useful when building networks where intermediate dimensions
 * are not known at construction time.
 */

#pragma once

#include <optional>
#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Lazy fully connected (linear/dense) layer.
 *
 * A variant of Linear that infers in_features from the first input tensor.
 * Weight and bias parameters are not allocated until the first forward() call,
 * at which point they are initialized using Xavier uniform initialization.
 *
 * After materialization, this layer behaves identically to nn::Linear,
 * computing y = xW^T + b.
 *
 * Shape transformations (after materialization):
 * - Input: (*, in_features) where * is any number of dimensions
 * - Output: (*, out_features)
 * - Weight: (out_features, in_features)
 * - Bias: (out_features) if enabled
 *
 * @code
 * // Create lazy layer - in_features determined at first forward()
 * LazyLinear lazy_fc(128);
 *
 * // Parameters not yet allocated
 * assert(lazy_fc.parameters().empty());
 * assert(!lazy_fc.is_materialized());
 *
 * // First forward call materializes parameters
 * Variable x(Tensor({batch, 784}, DType::Float32, Device::cpu()), true);
 * Variable output = lazy_fc.forward(x);  // Shape: {batch, 128}
 *
 * // Now parameters exist
 * assert(lazy_fc.is_materialized());
 * assert(!lazy_fc.parameters().empty());
 * @endcode
 *
 * @see Linear for the eager initialization variant
 * @see Module for base class interface
 */
class LazyLinear : public Module {
public:
    /**
     * @brief Construct lazy linear layer.
     *
     * Only out_features is specified; in_features is inferred from
     * the first input tensor's last dimension.
     *
     * @param out_features Size of each output sample
     * @param bias If true, add learnable bias after materialization (default: true)
     *
     * @code
     * LazyLinear layer(10, true);   // With bias
     * LazyLinear layer_no_bias(10, false);  // Without bias
     * @endcode
     */
    LazyLinear(int64_t out_features, bool bias = true);

    /**
     * @brief Forward pass through lazy linear layer.
     *
     * On the first call, inspects input.shape()[-1] to determine in_features,
     * materializes weight and bias parameters, then computes y = xW^T + b.
     * Subsequent calls behave identically to nn::Linear::forward().
     *
     * @param input Input variable of shape (*, in_features)
     * @return Output variable of shape (*, out_features)
     *
     * @throws std::runtime_error if input has fewer than 1 dimension
     * @throws std::runtime_error if input's last dimension is <= 0
     * @throws std::runtime_error if subsequent calls have mismatched in_features
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Check if parameters have been materialized.
     *
     * @return true if weight (and optional bias) have been initialized
     */
    auto is_materialized() const -> bool { return materialized_; }

    /**
     * @brief Get all parameters.
     *
     * Returns empty vector if not yet materialized, otherwise returns
     * weight and optional bias parameters.
     *
     * @return Vector of shared pointers to parameters
     */
    auto parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get only this module's direct parameters (not submodules').
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of shared pointers to this module's own parameters only
     */
    auto own_parameters() -> std::vector<std::shared_ptr<Variable>> override;

    /**
     * @brief Get all parameters with names.
     *
     * Returns empty vector if not yet materialized.
     *
     * @return Vector of (name, shared_ptr to parameter) pairs
     */
    auto named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> override;

    /**
     * @brief Check if layer has bias.
     */
    auto has_bias() const -> bool { return has_bias_; }

    /**
     * @brief Get weight parameter.
     *
     * @return Const reference to shared_ptr to weight variable
     * @throws std::runtime_error if not yet materialized
     */
    auto weight() const -> const std::shared_ptr<Variable>&;

    /**
     * @brief Get bias parameter (if present).
     *
     * @return Shared pointer to bias variable, or nullptr if no bias or not materialized
     */
    auto bias() const -> std::shared_ptr<Variable>;

    /**
     * @brief Get the inferred in_features (only valid after materialization).
     *
     * @return The inferred input feature dimension
     * @throws std::runtime_error if not yet materialized
     */
    auto in_features() const -> int64_t;

    /**
     * @brief Get out_features.
     *
     * @return The output feature dimension
     */
    auto out_features() const -> int64_t { return out_features_; }

    /**
     * @brief Capture a requested dtype to honour at materialisation time.
     *
     * V.29: pre-materialisation `model.to(BFloat16)` was silently dropped
     * because there were no parameters/buffers yet for Module::to to walk.
     * LazyLinear now intercepts the call and stashes the request; on the
     * first forward pass, materialize() allocates weight/bias at this
     * dtype instead of the hardcoded Float32 default.  Post-materialisation,
     * we delegate to Module::to so the live parameters are converted in
     * place (and the request is updated for future inspection).
     */
    auto to(DType dtype) -> void override;
    auto to(Device device) -> void override;

private:
    int64_t out_features_;                      ///< Output feature dimension
    bool has_bias_;                             ///< Whether this layer has bias
    bool materialized_{false};                 ///< Whether parameters have been created
    int64_t in_features_{0};                   ///< Input feature dimension (set on first forward)
    /**
     * @brief V.29: dtype requested via `to(DType)` before materialisation.
     *
     * `std::nullopt` until the user calls `model.to(some_dtype)`; after
     * that, materialize() honours this dtype instead of the Float32 default.
     */
    std::optional<DType> requested_dtype_{};
    /**
     * @brief V.29: device requested via `to(Device)` before materialisation.
     *
     * The materialise path already takes the device from the first input
     * tensor (which is the right behaviour when no `to(Device)` was called),
     * so this field exists only to keep the requested-state visible for
     * debugging / introspection.  We do not need to consult it in
     * materialize() because the first-input device dominates.
     */
    std::optional<Device> requested_device_{};

    /**
     * @brief Materialize weight and bias parameters.
     *
     * Called on first forward pass. Creates weight tensor of shape
     * (out_features, in_features) and optional bias tensor of shape
     * (out_features), both initialized with Xavier uniform.
     *
     * @param in_features Inferred input feature dimension
     * @param device Device to create parameters on
     */
    auto materialize(int64_t in_features, Device device) -> void;
};

} // namespace nn
} // namespace tenzor
