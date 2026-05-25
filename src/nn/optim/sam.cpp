#include "tenzor/nn/optim/sam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::optim {

SAM::SAM(std::shared_ptr<Optimizer> base_optimizer, double rho)
    : Optimizer(base_optimizer->parameters()),
      base_optimizer_(std::move(base_optimizer)),
      rho_(rho) {
    if (rho_ < 0.0) {
        throw std::invalid_argument("SAM: rho must be non-negative");
    }
}

SAM::SAM(std::vector<optim::ParamGroup> groups,
         std::shared_ptr<Optimizer> base_optimizer,
         double default_rho)
    : Optimizer(std::move(groups)),
      base_optimizer_(std::move(base_optimizer)),
      rho_(default_rho) {
    if (rho_ < 0.0) {
        throw std::invalid_argument("SAM: rho must be non-negative");
    }
}

auto SAM::first_step() -> void {
    // Audit D.4: per-parameter `rho` resolves from the active ParamGroup
    // (when one was set up via the ParamGroup-list constructor) or falls
    // through to the optimiser-wide default stored on this SAM instance.
    auto resolve = [&](size_t i) -> double {
        if (const auto* g = find_group_for_param(i)) {
            return ParamGroup::or_else(g->rho, rho_);
        }
        return rho_;
    };

    // Compute global gradient norm across all parameters
    double grad_norm_sq = 0.0;

    for (auto& param_ptr : parameters_) {
        if (!param_ptr || !param_ptr->has_grad()) continue;
        const Tensor& grad = param_ptr->grad().value();
        // Sum of squared elements
        auto norm_sq = sum(grad * grad);
        // Move to CPU before reading host-side; data<T>() on a GPU tensor returns
        // a device pointer and dereferencing it from host code segfaults.
        auto norm_sq_cpu = norm_sq.to(DType::Float64).to(Device::cpu());
        grad_norm_sq += static_cast<double>(norm_sq_cpu.data<double>()[0]);
    }

    double grad_norm = std::sqrt(grad_norm_sq) + 1e-12;  // Avoid division by zero

    // Compute and apply perturbation for each parameter
    epsilon_.clear();
    epsilon_.reserve(parameters_.size());

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];

        if (!param_ptr || !param_ptr->has_grad()) {
            epsilon_.emplace_back();  // Empty tensor placeholder
            continue;
        }

        const Tensor& grad = param_ptr->grad().value();

        // Per-parameter rho (from ParamGroup or instance default); global norm.
        const double rho_i = resolve(i);
        const double scale = rho_i / grad_norm;

        // epsilon_i = rho_i * grad_i / ||grad||_2
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, param_ptr->tensor().dtype(), param_ptr->tensor().device());
        };

        Tensor eps = grad * scalar(scale);
        epsilon_.push_back(eps.clone());

        // Perturb weights: w = w + epsilon
        param_ptr->tensor() = param_ptr->tensor() + eps;
    }
}

auto SAM::second_step() -> void {
    if (epsilon_.empty()) {
        throw std::runtime_error("SAM::second_step: first_step() must be called first");
    }

    // Restore original weights: w = w - epsilon
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i] || epsilon_[i].numel() == 0) continue;
        parameters_[i]->tensor() = parameters_[i]->tensor() - epsilon_[i];
    }

    // Step the base optimizer with the gradients from the perturbed point
    base_optimizer_->step();

    // Clear stored perturbations
    epsilon_.clear();
}

// Audit K.1: SAM's own epsilon_ buffer is rebuilt lazily inside
// first_step(), so we just discard any stale entries here.  The
// wrapped base optimiser, however, keeps real per-parameter state
// (momentum / exp_avg / ...) that MUST be extended in lockstep —
// forward the just-appended ParamGroup to it.
auto SAM::on_parameters_appended_(size_t /*old_count*/, size_t /*new_count*/) -> void {
    epsilon_.clear();
    if (!base_optimizer_) {
        throw std::runtime_error(
            "SAM::on_parameters_appended_: base optimizer is null; "
            "cannot extend its state buffers for the new ParamGroup.");
    }
    // The base class already moved the new ParamGroup into param_groups_,
    // so the most recent entry is what we need to thread through.
    if (param_groups_.empty()) {
        throw std::runtime_error(
            "SAM::on_parameters_appended_: parameter group list is empty "
            "after append — invariant violation.");
    }
    base_optimizer_->add_param_group(param_groups_.back());
}

auto SAM::step_impl() -> void {
    // step_impl is only invoked by the no-closure Optimizer::step(),
    // which can't drive SAM's two-pass requirement.  Users should call
    // step(closure) (which dispatches to our override below) or the
    // explicit first_step() / second_step() API.
    throw std::runtime_error(
        "SAM::step_impl: SAM requires two forward+backward passes around "
        "a weight perturbation, so step() with no closure is not supported. "
        "Either pass a closure to step(), or call first_step() and "
        "second_step() explicitly with the forward+backward pass in between.");
}

auto SAM::step(std::function<Variable()> closure) -> Variable {
    // 1. The caller's closure is expected to zero grads, run forward,
    //    backward, and return the loss at the current (unperturbed)
    //    weights.  We invoke it once to compute the initial gradients.
    if (!closure) {
        throw std::runtime_error(
            "SAM::step: closure must be non-null — SAM requires a way to "
            "re-evaluate the model at the perturbed weights.");
    }
    auto initial_loss = closure();

    // 2. Move weights along the gradient direction (ascent) to find a
    //    nearby maximum of the loss landscape.
    first_step();

    // 3. Re-run the closure at the perturbed weights to recompute the
    //    gradients at this neighbour-maximum.  Discard the perturbed
    //    loss value — what matters is that grads now reflect the
    //    perturbed point.
    {
        auto _ = closure();
        (void)_;
    }

    // 4. Restore the original weights and step the base optimizer using
    //    the (now perturbation-aware) gradients.
    second_step();

    // AA.3: SAM's step(closure) drives the underlying optimizer via
    // first_step()/second_step() — neither bumps the SAM-level step counter
    // nor fires post-step hooks registered on the SAM wrapper itself.
    // Pruning, EMA, etc. hooks that callers attach to the SAM optimizer
    // would otherwise never re-fire after the initial registration, so
    // re-impose dense masks would silently disappear. Replicate the
    // bookkeeping that Optimizer::step() normally performs.
    step_count_total_.fetch_add(1, std::memory_order_release);
    fire_post_step_hooks_();

    return initial_loss;
}

auto SAM::set_lr(double lr) -> void {
    base_optimizer_->set_lr(lr);
}

auto SAM::get_lr() const -> double {
    return base_optimizer_->get_lr();
}

auto SAM::state_dict() const -> std::unordered_map<std::string, Tensor> {
    auto state = base_optimizer_->state_dict();

    // Add SAM-specific state
    state["sam_rho"] = Tensor({1}, DType::Float64, Device::cpu());
    state["sam_rho"].data<double>()[0] = rho_;

    return state;
}

auto SAM::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    if (state.count("sam_rho")) {
        rho_ = state.at("sam_rho").data<double>()[0];
    }

    // S.17: strip the SAM-only key before forwarding so base optimisers
    // that hash-check their state dict (e.g. validating known keys) don't
    // see the extra sam_rho entry. The base optimiser only cares about
    // its own state — SAM owns sam_rho.
    auto base_state = state;
    base_state.erase("sam_rho");
    base_optimizer_->load_state_dict(base_state);
}

} // namespace tenzor::optim
