/**
 * @file lbfgs.hpp
 * @brief L-BFGS quasi-Newton optimizer
 *
 * Limited-memory Broyden-Fletcher-Goldfarb-Shanno (L-BFGS) is a quasi-Newton
 * method suited to classical (deterministic) full-batch optimization. Unlike
 * SGD/Adam, it requires a closure that (re-)computes the loss and gradients
 * multiple times per step during the line search.
 */

#pragma once

#include "optimizer.hpp"
#include <deque>

namespace tenzor {
namespace optim {

/**
 * @brief Line search variant for LBFGS.
 *
 * - `Armijo`: sufficient-decrease backtracking. Cheaper per iteration but
 *   cannot guarantee the curvature condition, so L-BFGS history updates can
 *   skip more often on non-quadratic losses.
 * - `StrongWolfe`: Nocedal-Wright Algorithms 3.5 + 3.6 (bracketing + zoom)
 *   with cubic interpolation. Finds an alpha satisfying both Armijo
 *   (f(x + alpha*p) <= f + c1*alpha*grad.p) and the strong curvature
 *   condition (|grad(x + alpha*p).p| <= c2*|grad.p|). Default — matches
 *   `torch.optim.LBFGS(line_search_fn="strong_wolfe")`.
 */
enum class LBFGSLineSearch : std::uint8_t {
    Armijo = 0,
    StrongWolfe = 1,
};

/**
 * @brief L-BFGS optimizer matching torch.optim.LBFGS.
 *
 * Implements the two-loop recursion from Nocedal-Wright (Algorithm 7.4). The
 * full-batch setting is assumed; each call to step(closure) performs up to
 * `max_iter` inner iterations, each invoking the closure to recompute loss
 * and gradient.
 *
 * Defaults match torch.optim.LBFGS:
 *   lr=1.0, max_iter=20, max_eval=None (1.25*max_iter), tolerance_grad=1e-7,
 *   tolerance_change=1e-9, history_size=100, line_search=StrongWolfe.
 *
 * @code
 * auto optimizer = LBFGS(model.parameters(), 1.0);
 * auto loss = optimizer.step([&] {
 *     optimizer.zero_grad();
 *     auto y = model.forward(x);
 *     auto loss = mse_loss(y, target);
 *     loss.backward();
 *     return loss;
 * });
 * @endcode
 */
class LBFGS : public Optimizer {
public:
    LBFGS(std::vector<std::shared_ptr<Variable>> params,
          double lr = 1.0,
          int max_iter = 20,
          int max_eval = -1,          // -1 → 1.25 * max_iter
          double tolerance_grad = 1e-7,
          double tolerance_change = 1e-9,
          int history_size = 100,
          LBFGSLineSearch line_search = LBFGSLineSearch::StrongWolfe);

    /**
     * @brief step() without a closure is not supported.
     *
     * L-BFGS requires the closure form because line search needs multiple
     * evaluations of f + grad. Calling step_impl() throws at runtime.
     */
    auto step_impl() -> void override;

    /**
     * @brief Perform a single L-BFGS step (may internally call closure many
     * times during line search). Returns the final loss.
     *
     * Hides the base-class step(closure); callers with an LBFGS reference get
     * this override, callers with a base Optimizer& would use the base form
     * which won't converge for LBFGS.
     */
    auto step(std::function<Variable()> closure) -> Variable;

    auto set_lr(double lr) -> void { lr_ = lr; }
    auto get_lr() const -> double { return lr_; }

    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    double lr_;
    int max_iter_;
    int max_eval_;
    double tolerance_grad_;
    double tolerance_change_;
    int history_size_;
    LBFGSLineSearch line_search_;

    // Persistent state across step() calls.
    std::deque<Tensor> s_history_;   // parameter deltas
    std::deque<Tensor> y_history_;   // gradient deltas
    std::deque<double> rho_history_; // 1 / <y, s>
    int n_iter_ = 0;                 // cumulative inner iterations
    bool has_prev_state_ = false;
    Tensor prev_flat_params_;
    Tensor prev_flat_grad_;
    double prev_loss_ = 0.0;

    // Helpers
    auto gather_flat_params() const -> Tensor;
    auto gather_flat_grad() const -> Tensor;
    auto apply_flat_delta(const Tensor& delta) -> void;  // params += delta
    auto two_loop_recursion(const Tensor& grad) const -> Tensor;  // returns search direction
};

} // namespace optim
} // namespace tenzor
