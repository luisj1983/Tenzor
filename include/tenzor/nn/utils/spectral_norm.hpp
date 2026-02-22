/**
 * @file spectral_norm.hpp
 * @brief Spectral normalization for neural network weight matrices
 *
 * Implements spectral normalization (Miyato et al., 2018) which constrains
 * the spectral norm of weight matrices to stabilize GAN training.
 * Uses power iteration to efficiently estimate the largest singular value.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include "../../core/tensor.hpp"
#include "../module.hpp"

namespace tenzor::nn::utils {

/**
 * @brief Spectral normalization wrapper for a module parameter.
 *
 * Applies spectral normalization to a named weight parameter of a module.
 * On each forward call, the weight is normalized by its estimated spectral
 * norm (largest singular value) computed via power iteration.
 *
 * The wrapper registers a forward pre-hook on the target module that
 * replaces the weight parameter with W / sigma(W) before each forward pass.
 *
 * @code
 * auto linear = std::make_shared<Linear>(128, 64);
 * auto sn = SpectralNorm::apply(linear, "weight");
 * // linear->forward() now uses spectrally-normalized weights
 *
 * // Remove spectral normalization
 * sn->remove();
 * @endcode
 */
class SpectralNorm {
public:
    /**
     * @brief Apply spectral normalization to a module parameter.
     *
     * @param module Target module
     * @param name Parameter name to normalize (default: "weight")
     * @param n_power_iterations Number of power iteration steps per forward (default: 1)
     * @param eps Epsilon for numerical stability (default: 1e-12)
     * @return Shared pointer to the SpectralNorm instance (for later removal)
     */
    static auto apply(std::shared_ptr<Module> module,
                      const std::string& name = "weight",
                      int64_t n_power_iterations = 1,
                      double eps = 1e-12) -> std::shared_ptr<SpectralNorm>;

    /**
     * @brief Remove spectral normalization from the module.
     *
     * Restores the original weight parameter (un-normalized).
     */
    auto remove() -> void;

    /**
     * @brief Get the current estimated spectral norm (largest singular value).
     *
     * @return Estimated sigma(W)
     */
    auto sigma() const -> Tensor { return sigma_; }

    /**
     * @brief Get the left singular vector u.
     */
    auto u() const -> const Tensor& { return u_; }

    /**
     * @brief Get the right singular vector v.
     */
    auto v() const -> const Tensor& { return v_; }

    ~SpectralNorm() = default;

private:
    SpectralNorm(std::shared_ptr<Module> module,
                 std::string name,
                 int64_t n_power_iterations,
                 double eps);

    /**
     * @brief Perform power iteration to estimate sigma(W).
     *
     * Updates u and v vectors in-place.
     *
     * @param weight The weight tensor (2D or reshaped to 2D)
     */
    auto power_iteration(const Tensor& weight) -> void;

    /**
     * @brief Compute the normalized weight W / sigma(W).
     *
     * @param weight Original weight tensor
     * @return Spectrally normalized weight
     */
    auto compute_weight(const Tensor& weight) -> Tensor;

    std::shared_ptr<Module> module_;            ///< Target module (kept alive)
    std::shared_ptr<Variable> param_;           ///< Cached parameter pointer
    std::string param_name_;                    ///< Parameter name
    int64_t n_power_iterations_;                ///< Power iteration steps per forward
    double eps_;                                ///< Numerical stability epsilon
    Tensor u_;                                  ///< Left singular vector (out_features,)
    Tensor v_;                                  ///< Right singular vector (in_features,)
    Tensor sigma_;                              ///< Current estimated spectral norm
    size_t hook_id_{0};                         ///< Registered hook ID for removal
    std::vector<int64_t> original_shape_;       ///< Original weight shape before reshape
};

} // namespace tenzor::nn::utils
