/**
 * @file lbfgs.cpp
 * @brief L-BFGS quasi-Newton optimizer implementation.
 *
 * Two-loop recursion (Nocedal-Wright Algorithm 7.4) with either Armijo
 * backtracking or a strong-Wolfe bracketing-and-zoom line search
 * (Algorithms 3.5 + 3.6, with cubic interpolation inside zoom). The
 * implementation flattens all parameter tensors into a single 1D vector for
 * the curvature-history updates, then scatters the updated vector back into
 * the original parameter tensors.
 */

#include "tenzor/nn/optim/lbfgs.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"   // narrow
#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>

namespace tenzor::optim {

namespace {

// Dot product of two 1D tensors → scalar double (host).
auto flat_dot(const Tensor& a, const Tensor& b) -> double {
    auto product = a * b;
    auto sum_t = sum(product);
    auto sum_cpu = (sum_t.device() == Device::cpu()) ? sum_t : sum_t.to(Device::cpu());
    if (sum_cpu.dtype() == DType::Float64) {
        return static_cast<double>(sum_cpu.data<double>()[0]);
    }
    if (sum_cpu.dtype() != DType::Float32) {
        sum_cpu = sum_cpu.to(DType::Float32);
    }
    return static_cast<double>(sum_cpu.data<float>()[0]);
}

auto flat_max_abs(const Tensor& a) -> double {
    auto absolute = abs(a);
    auto m = max(absolute);  // global max over all elements
    auto cpu = (m.device() == Device::cpu()) ? m : m.to(Device::cpu());
    if (cpu.dtype() == DType::Float64) {
        return static_cast<double>(cpu.data<double>()[0]);
    }
    if (cpu.dtype() != DType::Float32) {
        cpu = cpu.to(DType::Float32);
    }
    return static_cast<double>(cpu.data<float>()[0]);
}

// Scalar tensor on the same device/dtype as a reference.
auto scalar_like(double value, const Tensor& ref) -> Tensor {
    return full({1}, value, ref.dtype(), ref.device());
}

auto loss_to_double(const Variable& v) -> double {
    auto lt = (v.tensor().device() == Device::cpu())
        ? v.tensor() : v.tensor().to(Device::cpu());
    if (lt.dtype() == DType::Float64) {
        return lt.data<double>()[0];
    }
    if (lt.dtype() != DType::Float32) {
        lt = lt.to(DType::Float32);
    }
    return static_cast<double>(lt.data<float>()[0]);
}

// Cubic interpolation minimizer between two points (x1, f1, g1) and
// (x2, f2, g2), where f is the 1D function value and g is its derivative.
// Falls back to bisection when the cubic is degenerate or the minimizer
// falls outside the bracket. Follows Nocedal-Wright eq. (3.59).
auto cubic_interpolate(double x1, double f1, double g1,
                       double x2, double f2, double g2,
                       double bound_min, double bound_max) -> double {
    // Degenerate bracket — nothing to interpolate.
    const double width = x2 - x1;
    if (std::fabs(width) < 1e-30) {
        return x1;
    }
    const double d1 = g1 + g2 - 3.0 * (f1 - f2) / width;
    const double d2_sq = d1 * d1 - g1 * g2;
    if (d2_sq >= 0.0) {
        const double d2 = std::sqrt(d2_sq);
        double x_min;
        if (x1 <= x2) {
            x_min = x2 - (x2 - x1) * ((g2 + d2 - d1) / (g2 - g1 + 2.0 * d2));
        } else {
            x_min = x1 - (x1 - x2) * ((g1 + d2 - d1) / (g1 - g2 + 2.0 * d2));
        }
        // Clamp to the bracket bounds — reject extrapolation.
        return std::min(std::max(x_min, bound_min), bound_max);
    }
    // Fallback: bisection.
    return 0.5 * (bound_min + bound_max);
}

} // anonymous namespace

LBFGS::LBFGS(std::vector<std::shared_ptr<Variable>> params,
             double lr,
             int max_iter,
             int max_eval,
             double tolerance_grad,
             double tolerance_change,
             int history_size,
             LBFGSLineSearch line_search)
    : Optimizer(std::move(params)),
      lr_(lr),
      max_iter_(max_iter),
      max_eval_(max_eval >= 0 ? max_eval : static_cast<int>(max_iter * 5 / 4)),
      tolerance_grad_(tolerance_grad),
      tolerance_change_(tolerance_change),
      history_size_(history_size),
      line_search_(line_search) {
    if (parameters_.empty()) {
        throw std::invalid_argument("LBFGS: parameter list must be non-empty");
    }
    if (history_size <= 0) {
        throw std::invalid_argument("LBFGS: history_size must be positive");
    }
}

LBFGS::LBFGS(std::vector<optim::ParamGroup> groups,
             double default_lr,
             int default_max_iter,
             int default_max_eval,
             double default_tolerance_grad,
             double default_tolerance_change,
             int default_history_size,
             LBFGSLineSearch default_line_search)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      max_iter_(default_max_iter),
      max_eval_(default_max_eval >= 0
                    ? default_max_eval
                    : static_cast<int>(default_max_iter * 5 / 4)),
      tolerance_grad_(default_tolerance_grad),
      tolerance_change_(default_tolerance_change),
      history_size_(default_history_size),
      line_search_(default_line_search) {
    // PyTorch parity: torch.optim.LBFGS rejects multi-group configurations
    // because the two-loop recursion operates on a single flattened state
    // vector. Per-group hyperparameter dispatch would require running an
    // independent line search per group, which defeats the algorithm.
    if (param_groups_.size() != 1) {
        throw std::invalid_argument(
            "LBFGS: exactly one ParamGroup is supported (got " +
            std::to_string(param_groups_.size()) + ")");
    }
    if (parameters_.empty()) {
        throw std::invalid_argument("LBFGS: parameter list must be non-empty");
    }
    if (history_size_ <= 0) {
        throw std::invalid_argument("LBFGS: history_size must be positive");
    }
}

auto LBFGS::step_impl() -> void {
    throw std::runtime_error(
        "LBFGS requires a closure. Call step(closure) instead of step().");
}

// Audit K.1: L-BFGS operates on a single flat parameter vector and the
// constructor enforces exactly one ParamGroup (matching PyTorch's
// torch.optim.LBFGS).  Allowing add_param_group to silently push a
// second group would break the two-loop recursion's single-state
// contract — refuse explicitly.
auto LBFGS::on_parameters_appended_(size_t /*old_count*/, size_t /*new_count*/) -> void {
    throw std::runtime_error(
        "LBFGS::add_param_group: L-BFGS requires exactly one ParamGroup "
        "because its two-loop recursion operates on a single flattened "
        "state vector. Adding a second group after construction would "
        "violate that invariant; re-construct the optimiser with all "
        "parameters in one group instead. See audit K.1.");
}

// BB.12: when any parameter is F16/BF16 we promote the entire flat
// curvature-history path to Float32. The two-loop recursion accumulates
// over ~history_size past gradients; storing s/y at F16 underflows for
// typical L-BFGS step deltas (1e-3 squared ≈ 1e-6 → 0 in F16) and erodes
// the rho_history = 1/<y,s> denominator to noise after a few outer
// iterations.
auto LBFGS::flat_state_dtype_() const -> DType {
    for (const auto& p : parameters_) {
        if (!p) continue;
        if (optim_state_dtype(p->tensor().dtype()) == DType::Float32 &&
            p->tensor().dtype() != DType::Float32) {
            return DType::Float32;
        }
    }
    // No half-precision params; let the gather use the params' native dtype.
    // First non-null param dictates.
    for (const auto& p : parameters_) {
        if (!p) continue;
        return p->tensor().dtype();
    }
    return DType::Float32;
}

auto LBFGS::gather_flat_params() const -> Tensor {
    // Concatenate flattened views of every parameter into one 1D tensor.
    // BB.12: promote to F32 if any param is half-precision so the
    // curvature-history math runs at full precision.
    const DType target_dt = flat_state_dtype_();
    std::vector<Tensor> flat_parts;
    flat_parts.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        if (!p) continue;
        auto& t = p->tensor();
        auto flat = reshape(t, {t.numel()});
        if (flat.dtype() != target_dt) {
            flat = flat.to(target_dt);
        }
        flat_parts.push_back(flat);
    }
    if (flat_parts.empty()) {
        throw std::runtime_error("LBFGS: no parameters to optimize");
    }
    return cat(flat_parts, /*dim=*/0);
}

auto LBFGS::gather_flat_grad() const -> Tensor {
    // BB.12: mirror gather_flat_params — F16/BF16 grads get upcast to F32
    // before entering the two-loop recursion so y_history_ deltas are kept
    // at F32 precision.
    const DType target_dt = flat_state_dtype_();
    std::vector<Tensor> flat_parts;
    flat_parts.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        if (!p) continue;
        auto& t = p->tensor();
        if (p->has_grad()) {
            const auto& g = *p->grad();
            auto flat = reshape(g, {g.numel()});
            if (flat.dtype() != target_dt) {
                flat = flat.to(target_dt);
            }
            flat_parts.push_back(flat);
        } else {
            // Treat missing gradient as zero; this matches PyTorch semantics.
            flat_parts.push_back(zeros({t.numel()}, target_dt, t.device()));
        }
    }
    if (flat_parts.empty()) {
        throw std::runtime_error("LBFGS: no parameters to optimize");
    }
    return cat(flat_parts, /*dim=*/0);
}

auto LBFGS::apply_flat_delta(const Tensor& delta) -> void {
    // BB.12: delta arrives in the F32 master-state dtype (or the params'
    // native dtype if no half-precision params exist). Add to each param
    // in F32, then cast the final tensor back to the param's dtype — only
    // the public write to param.tensor() narrows precision, the math itself
    // stays at F32.
    int64_t offset = 0;
    for (auto& p : parameters_) {
        if (!p) continue;
        auto& t = p->tensor();
        int64_t n = t.numel();
        auto slice_1d = narrow(delta, /*dim=*/0, offset, n);
        auto shape_vec = std::vector<int64_t>(t.shape().begin(), t.shape().end());
        auto slice_shaped = reshape(slice_1d, shape_vec);
        if (slice_shaped.device() != t.device()) {
            slice_shaped = slice_shaped.to(t.device());
        }
        const DType param_dt = t.dtype();
        if (slice_shaped.dtype() != param_dt) {
            // Compute t + delta in delta's dtype (F32), then cast back to
            // the param dtype. This keeps the addition at F32 precision so
            // the small-step delta isn't rounded to zero against a much
            // larger param value in F16/BF16.
            Tensor t_promoted = t.to(slice_shaped.dtype());
            t = (t_promoted + slice_shaped).to(param_dt);
        } else {
            t = t + slice_shaped;
        }
        offset += n;
    }
}

auto LBFGS::two_loop_recursion(const Tensor& grad) const -> Tensor {
    // Algorithm 7.4 from Nocedal-Wright: compute direction = -H * grad where
    // H is the implicit inverse-Hessian approximation built from recent (s,y).
    const size_t m = s_history_.size();
    if (m == 0) {
        // First iteration: steepest descent direction.
        return Tensor(grad) * scalar_like(-1.0, grad);
    }

    Tensor q = grad.clone();
    std::vector<double> alphas(m, 0.0);

    for (size_t i = m; i-- > 0;) {
        double alpha = rho_history_[i] * flat_dot(s_history_[i], q);
        alphas[i] = alpha;
        q = q - y_history_[i] * scalar_like(alpha, q);
    }

    // Initial Hessian approximation: γ * I, where γ = <s_{m-1}, y_{m-1}> /
    // <y_{m-1}, y_{m-1}>, clamped for numerical stability.
    double s_last_y_last = flat_dot(s_history_.back(), y_history_.back());
    double y_last_sq = flat_dot(y_history_.back(), y_history_.back());
    double gamma = (y_last_sq > 1e-30) ? (s_last_y_last / y_last_sq) : 1.0;
    Tensor r = q * scalar_like(gamma, q);

    for (size_t i = 0; i < m; ++i) {
        double beta = rho_history_[i] * flat_dot(y_history_[i], r);
        r = r + s_history_[i] * scalar_like(alphas[i] - beta, r);
    }

    // Search direction is -H*grad.
    return r * scalar_like(-1.0, r);
}

auto LBFGS::step(std::function<Variable()> closure) -> Variable {
    // Audit D.4: when constructed from a ParamGroup, the group's `lr`
    // overrides the optimiser-wide default for *this* step. LBFGS is
    // restricted to a single group (enforced in the ctor), so a single
    // resolved scalar suffices — no per-parameter loop needed.
    const double lr = param_groups_.empty() ? lr_ : param_groups_[0].lr;

    // Initial evaluation.
    Variable loss_var = closure();
    double loss = loss_to_double(loss_var);

    int func_evals = 1;
    auto flat_grad = gather_flat_grad();
    double opt_cond = flat_max_abs(flat_grad);

    // Early exit: already at optimum.
    if (opt_cond <= tolerance_grad_) {
        return loss_var;
    }

    const double c1 = 1e-4;
    const double c2 = 0.9;   // Standard strong-Wolfe curvature constant for L-BFGS.

    // Inner L-BFGS iteration loop.
    for (int iter = 0; iter < max_iter_; ++iter) {
        n_iter_ += 1;

        // Update the (s, y) history from the previous iteration.
        if (has_prev_state_) {
            auto s = gather_flat_params() - prev_flat_params_;
            auto y = flat_grad - prev_flat_grad_;
            double ys = flat_dot(y, s);
            if (ys > 1e-10) {  // curvature condition
                if (static_cast<int>(s_history_.size()) == history_size_) {
                    s_history_.pop_front();
                    y_history_.pop_front();
                    rho_history_.pop_front();
                }
                s_history_.push_back(s);
                y_history_.push_back(y);
                rho_history_.push_back(1.0 / ys);
            }
        }

        // Compute search direction via two-loop recursion.
        Tensor direction = two_loop_recursion(flat_grad);
        double gtd = flat_dot(flat_grad, direction);  // directional derivative
        if (gtd > -tolerance_change_) {
            // Non-descent direction — reset history and fall back to
            // steepest descent for the next iteration.
            s_history_.clear();
            y_history_.clear();
            rho_history_.clear();
            direction = flat_grad * scalar_like(-1.0, flat_grad);
            gtd = flat_dot(flat_grad, direction);
        }

        // Save current state before the trial step.
        prev_flat_params_ = gather_flat_params();
        prev_flat_grad_ = flat_grad;
        prev_loss_ = loss;
        has_prev_state_ = true;

        // Track the applied step size so each line-search probe can move
        // from the current position directly to the target alpha.
        double current_alpha = 0.0;
        const double loss_0 = loss;
        const double gtd_0 = gtd;

        // Helper: move parameters to target_alpha and re-evaluate loss
        // and gradient. Returns (loss, gtd_new, flat_grad_new, loss_var_new).
        auto eval_at = [&](double target_alpha)
            -> std::tuple<double, double, Tensor, Variable> {
            const double delta_alpha = target_alpha - current_alpha;
            if (std::fabs(delta_alpha) > 0.0) {
                Tensor delta = direction * scalar_like(delta_alpha, direction);
                apply_flat_delta(delta);
                current_alpha = target_alpha;
            }
            Variable new_loss_var = closure();
            func_evals += 1;
            double new_loss_val = loss_to_double(new_loss_var);
            Tensor new_flat_grad = gather_flat_grad();
            double new_gtd = flat_dot(new_flat_grad, direction);
            return {new_loss_val, new_gtd, new_flat_grad, new_loss_var};
        };

        Variable new_loss_var = loss_var;
        double new_loss = loss;
        Tensor new_flat_grad = flat_grad;
        double accepted_alpha = 0.0;

        if (line_search_ == LBFGSLineSearch::StrongWolfe) {
            // --- Bracketing (Nocedal-Wright Algorithm 3.5) ---
            double alpha_prev = 0.0;
            double phi_prev = loss_0;
            double gtd_prev = gtd_0;

            double alpha_curr = lr;
            double alpha_max = std::max(lr * 10.0, 10.0);

            // Each probe spends one func eval; bail out if budget exhausted.
            const int ls_budget = std::max(1, max_eval_ - func_evals);
            int ls_remaining = ls_budget;
            const int max_ls_probes = 25;

            bool bracket_found = false;
            double alpha_lo = 0.0, alpha_hi = 0.0;
            double phi_lo = 0.0, phi_hi = 0.0;
            double gtd_lo = 0.0, gtd_hi = 0.0;
            double phi_curr = 0.0, gtd_curr = 0.0;
            Tensor grad_curr = flat_grad;
            Variable loss_var_curr = loss_var;

            int ls_iter = 0;
            // bracketing_done: a Wolfe-satisfying alpha was found outright
            // (no zoom needed) or a bracket [lo, hi] was built.
            bool wolfe_ok = false;
            while (ls_iter < max_ls_probes && ls_remaining > 0) {
                auto [p, g, fg, lv] = eval_at(alpha_curr);
                phi_curr = p;
                gtd_curr = g;
                grad_curr = fg;
                loss_var_curr = lv;
                ls_remaining -= 1;
                ls_iter += 1;

                const bool armijo_fail =
                    phi_curr > loss_0 + c1 * alpha_curr * gtd_0;
                const bool non_decrease = (ls_iter > 1 && phi_curr >= phi_prev);
                if (armijo_fail || non_decrease) {
                    alpha_lo = alpha_prev; alpha_hi = alpha_curr;
                    phi_lo  = phi_prev;    phi_hi  = phi_curr;
                    gtd_lo  = gtd_prev;    gtd_hi  = gtd_curr;
                    bracket_found = true;
                    break;
                }
                if (std::fabs(gtd_curr) <= -c2 * gtd_0) {
                    // Strong Wolfe satisfied — done.
                    wolfe_ok = true;
                    accepted_alpha = alpha_curr;
                    new_loss = phi_curr;
                    new_flat_grad = grad_curr;
                    new_loss_var = loss_var_curr;
                    break;
                }
                if (gtd_curr >= 0.0) {
                    alpha_lo = alpha_curr; alpha_hi = alpha_prev;
                    phi_lo  = phi_curr;    phi_hi  = phi_prev;
                    gtd_lo  = gtd_curr;    gtd_hi  = gtd_prev;
                    bracket_found = true;
                    break;
                }
                // Step forward: cubic-extrapolate between (alpha_prev, alpha_curr).
                double next_alpha = cubic_interpolate(
                    alpha_prev, phi_prev, gtd_prev,
                    alpha_curr, phi_curr, gtd_curr,
                    alpha_curr * 2.0, alpha_max);
                if (!(next_alpha > alpha_curr + 1e-12)) {
                    // Ensure progress — at least double the step.
                    next_alpha = std::min(alpha_curr * 2.0, alpha_max);
                }
                alpha_prev = alpha_curr;
                phi_prev = phi_curr;
                gtd_prev = gtd_curr;
                alpha_curr = next_alpha;
                if (alpha_curr >= alpha_max) {
                    // Reached the hard upper bound without Wolfe — use it as
                    // the best-effort alpha and stop.
                    alpha_curr = alpha_max;
                    auto [p2, g2, fg2, lv2] = eval_at(alpha_curr);
                    ls_remaining -= 1;
                    accepted_alpha = alpha_curr;
                    new_loss = p2;
                    new_flat_grad = fg2;
                    new_loss_var = lv2;
                    wolfe_ok = false;
                    break;
                }
            }

            // --- Zoom (Nocedal-Wright Algorithm 3.6) ---
            if (!wolfe_ok && bracket_found) {
                int zoom_iter = 0;
                const int max_zoom = 25;
                // Best-effort: remember the lowest-loss probe so we always
                // finish with *something* better than alpha=0.
                double best_alpha = alpha_lo;
                double best_loss = phi_lo;
                Tensor best_grad = flat_grad;  // dummy; will overwrite below.
                Variable best_loss_var = loss_var;

                // Seed "best" with current loss/grad at alpha_lo if we know
                // the grad there. alpha_lo was loaded from alpha_prev, which
                // we evaluated only before the last probe — grad_prev wasn't
                // captured. Safest: re-evaluate at alpha_lo so we own a
                // consistent (loss, grad) pair for fallback.
                {
                    auto [p_lo, g_lo, fg_lo, lv_lo] = eval_at(alpha_lo);
                    ls_remaining -= 1;
                    phi_lo = p_lo; gtd_lo = g_lo;
                    best_alpha = alpha_lo;
                    best_loss = p_lo;
                    best_grad = fg_lo;
                    best_loss_var = lv_lo;
                }

                while (zoom_iter < max_zoom && ls_remaining > 0) {
                    const double bracket_min = std::min(alpha_lo, alpha_hi);
                    const double bracket_max = std::max(alpha_lo, alpha_hi);
                    const double width = bracket_max - bracket_min;
                    // Reject a vanishing bracket.
                    if (width < 1e-12) break;

                    double alpha_j = cubic_interpolate(
                        alpha_lo, phi_lo, gtd_lo,
                        alpha_hi, phi_hi, gtd_hi,
                        bracket_min, bracket_max);
                    // Safeguarding (More-Thuente style): force alpha_j into
                    // the half of the bracket adjacent to alpha_lo (the
                    // side with the lower loss), guaranteeing at least a
                    // bisection-rate contraction each iteration. Without
                    // this, cubic interpolation on ill-conditioned losses
                    // can stall near the high-loss end of the bracket.
                    const double eps_inner = 0.1 * width;
                    double clamp_min, clamp_max;
                    if (alpha_lo < alpha_hi) {
                        clamp_min = bracket_min + eps_inner;
                        clamp_max = bracket_min + 0.5 * width;
                    } else {
                        clamp_min = bracket_max - 0.5 * width;
                        clamp_max = bracket_max - eps_inner;
                    }
                    if (clamp_min > clamp_max) clamp_min = clamp_max;
                    if (alpha_j < clamp_min) alpha_j = clamp_min;
                    if (alpha_j > clamp_max) alpha_j = clamp_max;

                    auto [p_j, g_j, fg_j, lv_j] = eval_at(alpha_j);
                    ls_remaining -= 1;
                    zoom_iter += 1;

                    if (p_j < best_loss) {
                        best_loss = p_j;
                        best_alpha = alpha_j;
                        best_grad = fg_j;
                        best_loss_var = lv_j;
                    }

                    const bool armijo_fail_j =
                        p_j > loss_0 + c1 * alpha_j * gtd_0;
                    if (armijo_fail_j || p_j >= phi_lo) {
                        alpha_hi = alpha_j;
                        phi_hi = p_j;
                        gtd_hi = g_j;
                    } else {
                        if (std::fabs(g_j) <= -c2 * gtd_0) {
                            accepted_alpha = alpha_j;
                            new_loss = p_j;
                            new_flat_grad = fg_j;
                            new_loss_var = lv_j;
                            wolfe_ok = true;
                            break;
                        }
                        if (g_j * (alpha_hi - alpha_lo) >= 0.0) {
                            alpha_hi = alpha_lo;
                            phi_hi = phi_lo;
                            gtd_hi = gtd_lo;
                        }
                        alpha_lo = alpha_j;
                        phi_lo = p_j;
                        gtd_lo = g_j;
                    }
                }

                if (!wolfe_ok) {
                    // Best-effort: fall back to the lowest-loss probe seen.
                    // eval_at() handles the move-to-target from current_alpha.
                    auto [p_b, g_b, fg_b, lv_b] = eval_at(best_alpha);
                    accepted_alpha = best_alpha;
                    new_loss = p_b;
                    new_flat_grad = fg_b;
                    new_loss_var = lv_b;
                }
            } else if (!wolfe_ok && !bracket_found) {
                // Budget exhausted before finding a bracket or a Wolfe alpha.
                // Use whatever the final probe produced.
                accepted_alpha = alpha_curr;
                new_loss = phi_curr;
                new_flat_grad = grad_curr;
                new_loss_var = loss_var_curr;
            }
        } else {
            // --- Armijo backtracking (unchanged semantics) ---
            double t = lr;
            auto [p0, g0, fg0, lv0] = eval_at(t);
            new_loss = p0;
            new_flat_grad = fg0;
            new_loss_var = lv0;

            int ls_iter = 0;
            const int ls_max = 25;
            while (new_loss > loss_0 + c1 * t * gtd_0 && ls_iter < ls_max) {
                t *= 0.5;
                if (t < 1e-12) break;
                auto [p, g, fg, lv] = eval_at(t);
                new_loss = p;
                new_flat_grad = fg;
                new_loss_var = lv;
                ls_iter += 1;
                if (func_evals >= max_eval_) break;
            }
            accepted_alpha = t;
        }

        // Refresh loss / grad / optimality check with the accepted point.
        loss_var = new_loss_var;
        loss = new_loss;
        flat_grad = new_flat_grad;
        opt_cond = flat_max_abs(flat_grad);

        // Convergence checks.
        if (opt_cond <= tolerance_grad_) break;
        double step_size = std::fabs(accepted_alpha) * flat_max_abs(direction);
        if (step_size <= tolerance_change_) break;
        if (std::fabs(loss - prev_loss_) < tolerance_change_) break;
        if (func_evals >= max_eval_) break;
    }

    return loss_var;
}

// Audit item D.5: persist every LBFGS field needed to resume training
// from a checkpoint — hyperparameters, the previous-step flat grad and
// scalar loss, the has_prev_state_ guard, plus the existing s/y/rho
// history.  Previously only n_iter and the histories were stored, so a
// non-default LBFGS lost its tolerances/limits and any in-flight
// line-search state when round-tripped.
auto LBFGS::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    auto scalar_i64 = [](int64_t v) {
        Tensor t({1}, DType::Int64, Device::cpu());
        t.data<int64_t>()[0] = v;
        return t;
    };
    auto scalar_f64 = [](double v) {
        Tensor t({1}, DType::Float64, Device::cpu());
        t.data<double>()[0] = v;
        return t;
    };

    state["n_iter"] = scalar_i64(static_cast<int64_t>(n_iter_));

    if (!s_history_.empty()) {
        std::vector<Tensor> s_vec(s_history_.begin(), s_history_.end());
        std::vector<Tensor> y_vec(y_history_.begin(), y_history_.end());
        state["s"] = stack(s_vec, /*dim=*/0);
        state["y"] = stack(y_vec, /*dim=*/0);
        Tensor rho_t({static_cast<int64_t>(rho_history_.size())}, DType::Float64, Device::cpu());
        auto* rho_data = rho_t.data<double>();
        for (size_t i = 0; i < rho_history_.size(); ++i) rho_data[i] = rho_history_[i];
        state["rho"] = rho_t;
    }

    // Hyperparameters.
    state["lr"]                = scalar_f64(lr_);
    state["max_iter"]          = scalar_i64(static_cast<int64_t>(max_iter_));
    state["max_eval"]          = scalar_i64(static_cast<int64_t>(max_eval_));
    state["tolerance_grad"]    = scalar_f64(tolerance_grad_);
    state["tolerance_change"]  = scalar_f64(tolerance_change_);
    state["history_size"]      = scalar_i64(static_cast<int64_t>(history_size_));

    // Convergence state.
    state["has_prev_state"]    = scalar_i64(has_prev_state_ ? 1 : 0);
    state["prev_loss"]         = scalar_f64(prev_loss_);
    if (has_prev_state_ && prev_flat_grad_.numel() > 0) {
        state["prev_flat_grad"] = prev_flat_grad_.clone();
    }

    return state;
}

auto LBFGS::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    s_history_.clear();
    y_history_.clear();
    rho_history_.clear();
    has_prev_state_ = false;

    auto get_i64 = [&](const std::string& key, int64_t fallback) -> int64_t {
        auto it = state.find(key);
        if (it == state.end()) return fallback;
        auto cpu = (it->second.device() == Device::cpu()) ? it->second : it->second.to(Device::cpu());
        if (cpu.dtype() == DType::Int64)  return cpu.data<int64_t>()[0];
        if (cpu.dtype() == DType::Int32)  return static_cast<int64_t>(cpu.data<int32_t>()[0]);
        return fallback;
    };
    auto get_f64 = [&](const std::string& key, double fallback) -> double {
        auto it = state.find(key);
        if (it == state.end()) return fallback;
        auto cpu = (it->second.device() == Device::cpu()) ? it->second : it->second.to(Device::cpu());
        if (cpu.dtype() == DType::Float64) return cpu.data<double>()[0];
        if (cpu.dtype() == DType::Float32) return static_cast<double>(cpu.data<float>()[0]);
        return fallback;
    };

    n_iter_           = static_cast<int>(get_i64("n_iter",       static_cast<int64_t>(n_iter_)));
    lr_               = get_f64("lr",                lr_);
    max_iter_         = static_cast<int>(get_i64("max_iter",     static_cast<int64_t>(max_iter_)));
    max_eval_         = static_cast<int>(get_i64("max_eval",     static_cast<int64_t>(max_eval_)));
    tolerance_grad_   = get_f64("tolerance_grad",    tolerance_grad_);
    tolerance_change_ = get_f64("tolerance_change",  tolerance_change_);
    history_size_     = static_cast<int>(get_i64("history_size", static_cast<int64_t>(history_size_)));

    has_prev_state_ = (get_i64("has_prev_state", 0) != 0);
    prev_loss_      = get_f64("prev_loss",      prev_loss_);
    auto it_pg = state.find("prev_flat_grad");
    if (it_pg != state.end()) {
        prev_flat_grad_ = it_pg->second.clone();
    }

    auto it_s = state.find("s");
    auto it_y = state.find("y");
    auto it_rho = state.find("rho");
    if (it_s != state.end() && it_y != state.end() && it_rho != state.end()) {
        const auto& s_stack = it_s->second;
        const auto& y_stack = it_y->second;
        const auto& rho_t = it_rho->second;
        int64_t m = s_stack.shape()[0];
        int64_t n = s_stack.shape().size() > 1 ? s_stack.shape()[1] : 0;
        for (int64_t i = 0; i < m; ++i) {
            // Each history entry is a 1D flat parameter delta; narrow returns
            // a 2D [1, N] slice which we reshape back to [N].
            auto s_slice = narrow(s_stack, /*dim=*/0, i, 1);
            auto y_slice = narrow(y_stack, /*dim=*/0, i, 1);
            s_history_.push_back(reshape(s_slice, {n}));
            y_history_.push_back(reshape(y_slice, {n}));
        }
        auto rho_cpu = (rho_t.device() == Device::cpu()) ? rho_t : rho_t.to(Device::cpu());
        const auto* rho_data = rho_cpu.data<double>();
        for (int64_t i = 0; i < m; ++i) rho_history_.push_back(rho_data[i]);
    }
}

} // namespace tenzor::optim
