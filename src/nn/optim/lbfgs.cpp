/**
 * @file lbfgs.cpp
 * @brief L-BFGS quasi-Newton optimizer implementation.
 *
 * Two-loop recursion (Nocedal-Wright Algorithm 7.4) with an Armijo backtracking
 * line search. The implementation flattens all parameter tensors into a single
 * 1D vector for the curvature-history updates, then scatters the updated
 * vector back into the original parameter tensors.
 */

#include "tenzor/nn/optim/lbfgs.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/indexing.hpp"   // narrow
#include <cmath>
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

} // anonymous namespace

LBFGS::LBFGS(std::vector<std::shared_ptr<Variable>> params,
             double lr,
             int max_iter,
             int max_eval,
             double tolerance_grad,
             double tolerance_change,
             int history_size)
    : Optimizer(std::move(params)),
      lr_(lr),
      max_iter_(max_iter),
      max_eval_(max_eval >= 0 ? max_eval : static_cast<int>(max_iter * 5 / 4)),
      tolerance_grad_(tolerance_grad),
      tolerance_change_(tolerance_change),
      history_size_(history_size) {
    if (parameters_.empty()) {
        throw std::invalid_argument("LBFGS: parameter list must be non-empty");
    }
    if (history_size <= 0) {
        throw std::invalid_argument("LBFGS: history_size must be positive");
    }
}

auto LBFGS::step_impl() -> void {
    throw std::runtime_error(
        "LBFGS requires a closure. Call step(closure) instead of step().");
}

auto LBFGS::gather_flat_params() const -> Tensor {
    // Concatenate flattened views of every parameter into one 1D tensor.
    std::vector<Tensor> flat_parts;
    flat_parts.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        if (!p) continue;
        auto& t = p->tensor();
        flat_parts.push_back(reshape(t, {t.numel()}));
    }
    if (flat_parts.empty()) {
        throw std::runtime_error("LBFGS: no parameters to optimize");
    }
    return cat(flat_parts, /*dim=*/0);
}

auto LBFGS::gather_flat_grad() const -> Tensor {
    std::vector<Tensor> flat_parts;
    flat_parts.reserve(parameters_.size());
    for (const auto& p : parameters_) {
        if (!p) continue;
        auto& t = p->tensor();
        if (p->has_grad()) {
            const auto& g = *p->grad();
            flat_parts.push_back(reshape(g, {g.numel()}));
        } else {
            // Treat missing gradient as zero; this matches PyTorch semantics.
            flat_parts.push_back(zeros({t.numel()}, t.dtype(), t.device()));
        }
    }
    if (flat_parts.empty()) {
        throw std::runtime_error("LBFGS: no parameters to optimize");
    }
    return cat(flat_parts, /*dim=*/0);
}

auto LBFGS::apply_flat_delta(const Tensor& delta) -> void {
    // Unflatten delta back to per-parameter shapes and add to each tensor.
    int64_t offset = 0;
    for (auto& p : parameters_) {
        if (!p) continue;
        auto& t = p->tensor();
        int64_t n = t.numel();
        auto slice_1d = narrow(delta, /*dim=*/0, offset, n);
        auto shape_vec = std::vector<int64_t>(t.shape().begin(), t.shape().end());
        auto slice_shaped = reshape(slice_1d, shape_vec);
        if (slice_shaped.dtype() != t.dtype()) {
            slice_shaped = slice_shaped.to(t.dtype());
        }
        if (slice_shaped.device() != t.device()) {
            slice_shaped = slice_shaped.to(t.device());
        }
        t = t + slice_shaped;
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
    // Initial evaluation.
    Variable loss_var = closure();
    double loss = 0.0;
    {
        auto lt = (loss_var.tensor().device() == Device::cpu())
            ? loss_var.tensor() : loss_var.tensor().to(Device::cpu());
        if (lt.dtype() == DType::Float64) {
            loss = static_cast<double>(lt.data<double>()[0]);
        } else {
            if (lt.dtype() != DType::Float32) lt = lt.to(DType::Float32);
            loss = static_cast<double>(lt.data<float>()[0]);
        }
    }

    int func_evals = 1;
    auto flat_grad = gather_flat_grad();
    double opt_cond = flat_max_abs(flat_grad);

    // Early exit: already at optimum.
    if (opt_cond <= tolerance_grad_) {
        return loss_var;
    }

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

        // Armijo backtracking line search.
        double t = lr_;
        const double c1 = 1e-4;
        bool step_accepted = false;
        Tensor trial_step = direction * scalar_like(t, direction);
        apply_flat_delta(trial_step);

        Variable new_loss_var = closure();
        func_evals += 1;

        auto loss_to_double = [](const Variable& v) -> double {
            auto lt = (v.tensor().device() == Device::cpu())
                ? v.tensor() : v.tensor().to(Device::cpu());
            if (lt.dtype() == DType::Float64) return lt.data<double>()[0];
            if (lt.dtype() != DType::Float32) lt = lt.to(DType::Float32);
            return static_cast<double>(lt.data<float>()[0]);
        };

        double new_loss = loss_to_double(new_loss_var);

        // Backtracking loop.
        int ls_iter = 0;
        const int ls_max = 25;
        while (new_loss > loss + c1 * t * gtd && ls_iter < ls_max) {
            // Undo trial step and shrink.
            auto undo_step = trial_step * scalar_like(-1.0, trial_step);
            apply_flat_delta(undo_step);

            t *= 0.5;
            if (t < 1e-12) break;

            trial_step = direction * scalar_like(t, direction);
            apply_flat_delta(trial_step);

            new_loss_var = closure();
            func_evals += 1;
            new_loss = loss_to_double(new_loss_var);
            ls_iter += 1;

            if (func_evals >= max_eval_) break;
        }
        step_accepted = true;

        // Refresh gradient and loss after the accepted step.
        loss_var = new_loss_var;
        loss = new_loss;
        flat_grad = gather_flat_grad();
        opt_cond = flat_max_abs(flat_grad);

        // Convergence checks.
        if (opt_cond <= tolerance_grad_) break;
        double step_size = std::fabs(t) * flat_max_abs(direction);
        if (step_size <= tolerance_change_) break;
        if (std::fabs(loss - prev_loss_) < tolerance_change_) break;
        if (func_evals >= max_eval_) break;
        (void)step_accepted;
    }

    return loss_var;
}

auto LBFGS::state_dict() const -> std::unordered_map<std::string, Tensor> {
    // Persist history as stacked tensors. s and y are stored as [m, N]; rho
    // as a 1D tensor. n_iter is stored as a 1-element Int64 tensor.
    std::unordered_map<std::string, Tensor> state;
    state["n_iter"] = full({1}, static_cast<double>(n_iter_), DType::Int64, Device::cpu());

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
    return state;
}

auto LBFGS::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    s_history_.clear();
    y_history_.clear();
    rho_history_.clear();
    has_prev_state_ = false;

    auto it_n = state.find("n_iter");
    if (it_n != state.end()) {
        auto cpu = (it_n->second.device() == Device::cpu())
            ? it_n->second : it_n->second.to(Device::cpu());
        n_iter_ = static_cast<int>(cpu.data<int64_t>()[0]);
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
