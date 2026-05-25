#include "tenzor/nn/optim/sam.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
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
    // audit-7 FF.11: SAM::step(closure) drives perturbation via first_step()
    // without going through Optimizer::step(), so the base's clip_gradients_()
    // pre-step would never run.  Apply SAM-level clipping here so the
    // perturbation pass sees clipped gradients (i.e. clipping participates
    // in finding the neighbour-maximum), matching the contract documented in
    // Optimizer::step().
    clip_gradients_();

    // Audit D.4: per-parameter `rho` resolves from the active ParamGroup
    // (when one was set up via the ParamGroup-list constructor) or falls
    // through to the optimiser-wide default stored on this SAM instance.
    auto resolve = [&](size_t i) -> double {
        if (const auto* g = find_group_for_param(i)) {
            return ParamGroup::or_else(g->rho, rho_);
        }
        return rho_;
    };

    // Compute global gradient norm across all parameters.
    // BB.11: when the param dtype is F16/BF16, take the squared-norm
    // accumulator at Float32 — the per-element grad*grad multiply in
    // half precision underflows for small grads (1e-4 squared ≈ 1e-8 →
    // 0 in F16), biasing ||grad|| toward zero and inflating epsilon.
    double grad_norm_sq = 0.0;

    for (auto& param_ptr : parameters_) {
        if (!param_ptr || !param_ptr->has_grad()) continue;
        const Tensor& grad = param_ptr->grad().value();
        const DType state_dt = optim_state_dtype(grad.dtype());
        Tensor grad_compute = (grad.dtype() == state_dt) ? grad : grad.to(state_dt);
        // Sum of squared elements
        auto norm_sq = sum(grad_compute * grad_compute);
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

        const DType param_dt = param_ptr->tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);

        if (state_dt != param_dt) {
            // BB.11: F16/BF16 path — upcast param + grad to Float32, build
            // epsilon at F32, perturb at F32, then cast the resulting
            // tensor back to the param dtype. Storing epsilon at param
            // dtype rounds 1e-7 (typical rho * grad / norm magnitude) to
            // zero in F16 and erodes BF16 momentum across steps.
            Tensor grad_f32 = grad.to(state_dt);
            Tensor scale_f32 = full({1}, scale, state_dt, param_ptr->tensor().device());
            Tensor eps_f32 = grad_f32 * scale_f32;

            // epsilon_ stays at F32 so second_step subtracts the exact F32
            // perturbation (re-casting back to F16/BF16 would lose precision
            // again on the restore).
            epsilon_.push_back(eps_f32.clone());

            Tensor param_f32 = param_ptr->tensor().to(state_dt);
            param_ptr->tensor() = (param_f32 + eps_f32).to(param_dt);
        } else {
            // epsilon_i = rho_i * grad_i / ||grad||_2 (param dtype is already
            // at least Float32 precision — no upcast needed).
            auto scalar = [&](double value) -> Tensor {
                return full({1}, value, param_dt, param_ptr->tensor().device());
            };

            Tensor eps = grad * scalar(scale);
            epsilon_.push_back(eps.clone());

            // Perturb weights: w = w + epsilon
            param_ptr->tensor() = param_ptr->tensor() + eps;
        }
    }
}

auto SAM::second_step() -> void {
    if (epsilon_.empty()) {
        throw std::runtime_error("SAM::second_step: first_step() must be called first");
    }

    // Restore original weights: w = w - epsilon.
    // BB.11: epsilon_ for F16/BF16 params is stored at Float32 to preserve
    // the perturbation magnitude across the round-trip; subtract in F32 and
    // cast the result back to the param dtype so the restored weights
    // exactly match the pre-perturbation value.
    for (size_t i = 0; i < parameters_.size(); ++i) {
        if (!parameters_[i] || epsilon_[i].numel() == 0) continue;
        auto& param_t = parameters_[i]->tensor();
        const DType param_dt = param_t.dtype();
        if (epsilon_[i].dtype() != param_dt) {
            // F16/BF16 master-weights restore path.
            Tensor param_f32 = param_t.to(epsilon_[i].dtype());
            param_t = (param_f32 - epsilon_[i]).to(param_dt);
        } else {
            param_t = param_t - epsilon_[i];
        }
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
    // audit-7 FF.12: match base Optimizer::step()'s memory ordering
    // (relaxed) so SAM doesn't impose stricter visibility than the base
    // counter; the counter is monotonic-but-not-load-bearing for ordering.
    step_count_total_.fetch_add(1, std::memory_order_relaxed);
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
        // CC.10: state_dict() writes sam_rho as Float64 / CPU, but a checkpoint
        // round-tripped through multi-rank serialization (or through a backend
        // that adapts scalar tensors to its own device/dtype on save) can come
        // back with a different dtype (Float32, Float16) or backend device.
        // Reading `data<double>()` on a Float32 / GPU tensor would either
        // mis-interpret bytes or trip a host-pointer assertion. Apply the V.27
        // adapter pattern: force Float64 + CPU before pulling the scalar host
        // value out.
        const Tensor& rho_state = state.at("sam_rho");
        rho_ = rho_state.to(DType::Float64).to(Device::cpu()).data<double>()[0];
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
