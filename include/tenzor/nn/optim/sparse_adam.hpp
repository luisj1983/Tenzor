/**
 * @file sparse_adam.hpp
 * @brief SparseAdam optimizer for sparse gradient updates
 */

#pragma once

#include "optimizer.hpp"

namespace tenzor {
namespace optim {

/**
 * @brief SparseAdam optimizer for efficient embedding training
 *
 * Implements a variant of Adam that only updates momentum buffers for
 * non-zero gradient entries. For dense gradients, falls back to standard
 * Adam behavior.
 *
 * Ideal for large embedding layers where only a small subset of rows
 * are accessed per batch.
 *
 * When gradients are sparse (contain many zero rows), SparseAdam avoids
 * decaying momentum for unvisited rows, which prevents the bias toward
 * zero that standard Adam introduces for infrequently-updated embeddings.
 *
 * **Algorithm (sparse path):**
 * For each non-zero gradient row i:
 * \f[
 * m_i = \beta_1 m_i + (1 - \beta_1) g_i \\
 * v_i = \beta_2 v_i + (1 - \beta_2) g_i^2 \\
 * \hat{m}_i = \frac{m_i}{1 - \beta_1^t} \\
 * \hat{v}_i = \frac{v_i}{1 - \beta_2^t} \\
 * \theta_i = \theta_i - \frac{\eta}{\sqrt{\hat{v}_i} + \epsilon} \hat{m}_i
 * \f]
 *
 * Rows with zero gradients are left untouched (no momentum decay).
 *
 * **Recommended Hyperparameters:**
 * - lr: 1e-3 (default)
 * - beta1: 0.9
 * - beta2: 0.999
 * - eps: 1e-8
 *
 * @param params Parameters to optimize
 * @param lr Learning rate (default: 1e-3)
 * @param beta1 First moment decay rate (default: 0.9)
 * @param beta2 Second moment decay rate (default: 0.999)
 * @param eps Term for numerical stability (default: 1e-8)
 *
 * @par Complexity
 * - Time: O(P) per step (dense), O(nnz) per step (sparse)
 * - Space: O(2P) for moment estimates
 *
 * @code
 * auto optimizer = SparseAdam(embedding.parameters(), 1e-3);
 * @endcode
 *
 * @see Adam, AdamW
 */
class SparseAdam : public Optimizer {
public:
    SparseAdam(std::vector<std::shared_ptr<Variable>> params,
               double lr = 1e-3,
               double beta1 = 0.9,
               double beta2 = 0.999,
               double eps = 1e-8);

    /** @brief Perform single SparseAdam step */
    auto step_impl() -> void override;

    /** @brief Set new learning rate */
    auto set_lr(double lr) -> void override;

    /** @brief Get current learning rate */
    auto get_lr() const -> double override;

    /** @brief Get optimizer state (moment estimates) for serialization */
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;

    /** @brief Load optimizer state from dictionary */
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    double lr_;
    double beta1_;
    double beta2_;
    double eps_;

    int64_t step_count_{0};
    std::vector<Tensor> exp_avg_;       ///< First moment estimates
    std::vector<Tensor> exp_avg_sq_;    ///< Second moment estimates

    auto initialize_buffers() -> void;
};

} // namespace optim
} // namespace tenzor
