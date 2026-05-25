#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/master_weights.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/foreach.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"

namespace tenzor::optim {

SGD::SGD(std::vector<std::shared_ptr<Variable>> params, double lr, double momentum,
        double dampening, double weight_decay, bool nesterov)
    : Optimizer(std::move(params)), lr_(lr), momentum_(momentum),
      dampening_(dampening), weight_decay_(weight_decay), nesterov_(nesterov) {
    initialize_buffers();
}

SGD::SGD(std::vector<optim::ParamGroup> groups,
         double default_lr, double default_momentum, double default_dampening,
         double default_weight_decay, bool default_nesterov)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      momentum_(default_momentum),
      dampening_(default_dampening),
      weight_decay_(default_weight_decay),
      nesterov_(default_nesterov) {
    initialize_buffers();
}

auto SGD::step_impl() -> void {
    // Audit D.4: resolve each parameter's hyperparameters from the
    // active ParamGroup (with optimizer-member fallback). When the
    // optimizer was constructed from a flat parameter list (no groups),
    // `find_group_for_param` returns nullptr and we use the SGD member
    // defaults — preserving pre-D.4 behaviour bit-for-bit.
    struct HP { double lr, momentum, weight_decay, dampening; bool nesterov; };
    auto resolve = [this](size_t i) -> HP {
        HP hp{lr_, momentum_, weight_decay_, dampening_, nesterov_};
        const auto* g = find_group_for_param(i);
        if (g) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.momentum     = ParamGroup::or_else(g->momentum,  momentum_);
            hp.dampening    = ParamGroup::or_else(g->dampening, dampening_);
            hp.nesterov     = ParamGroup::or_else(g->nesterov,  nesterov_);
        }
        return hp;
    };

    // Collect CPU parameters eligible for _foreach_* batch path.
    // CUDA params go through the fused kernel; all others are batched.
    struct CpuEntry {
        Tensor* param;
        Tensor  grad;
        size_t  idx;
        HP      hp;
    };
    std::vector<CpuEntry> cpu;

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;
        HP hp = resolve(i);

        Tensor& param_tensor = param.tensor();
        const Tensor& grad_tensor = *param.grad();

        // ── CUDA path: fused single-kernel dispatch ──────────────────────────
        // R.16: fused CUDA kernel only handles Float32/Float64 (half-precision
        // params have Float32 state via make_optim_state and need the upcast path).
        if (param_tensor.device().type == Device::Type::CUDA &&
            (param_tensor.dtype() == DType::Float32 || param_tensor.dtype() == DType::Float64)) {
            std::vector<Tensor> inputs = {param_tensor, grad_tensor};
            if (hp.momentum > 0.0) {
                inputs.push_back(velocity_buffers_[i]);
            }
            NewOpAttributes attrs;
            attrs.set(AttrKey::Lr, hp.lr);
            attrs.set(AttrKey::Momentum, hp.momentum);
            attrs.set(AttrKey::WeightDecay, hp.weight_decay);
            attrs.set(AttrKey::Dampening, hp.dampening);
            attrs.set(AttrKey::Nesterov, hp.nesterov);
            dispatch(OpId::FusedSGDStep, inputs, attrs);
            continue;
        }

        // R.16: half-precision params hold Float32 velocity buffers; run
        // the SGD update in Float32 and downcast on write-back. The foreach
        // batch path below requires uniform dtype between param/grad/velocity
        // so we handle halves inline.
        const DType param_dt = param_tensor.dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        if (state_dt != param_dt) {
            auto scalar = [&](double value) -> Tensor {
                return full({1}, value, state_dt, param_tensor.device());
            };
            Tensor param_hi = param_tensor.to(state_dt);
            Tensor grad_hi  = grad_tensor.to(state_dt);
            if (hp.weight_decay > 0.0) {
                grad_hi = grad_hi + param_hi * scalar(hp.weight_decay);
            }
            Tensor updated;
            if (hp.momentum > 0.0) {
                velocity_buffers_[i] = velocity_buffers_[i] * scalar(hp.momentum)
                                     + grad_hi * scalar(1.0 - hp.dampening);
                Tensor eff_grad = hp.nesterov
                    ? grad_hi + velocity_buffers_[i] * scalar(hp.momentum)
                    : velocity_buffers_[i];
                updated = param_hi - eff_grad * scalar(hp.lr);
            } else {
                updated = param_hi - grad_hi * scalar(hp.lr);
            }
            param_tensor = updated.to(param_dt);
            continue;
        }

        // ── CPU: clone grad and apply weight decay ────────────────────────────
        auto g = grad_tensor.clone();
        if (hp.weight_decay > 0.0) {
            g = g + param_tensor *
                full({1}, hp.weight_decay, param_tensor.dtype(), param_tensor.device());
        }
        cpu.push_back({&param_tensor, std::move(g), i, hp});
    }

    if (cpu.empty()) return;

    // Audit D.4: detect "any momentum" across the batch; the foreach
    // fast path can only be taken when every CPU param has momentum=0
    // (otherwise we'd need per-param velocity buffers which the batched
    // _foreach_mul/_sub_ kernels don't support).  This is a strict
    // generalisation of the pre-D.4 single-momentum check.
    bool any_momentum = false;
    for (const auto& e : cpu) if (e.hp.momentum > 0.0) { any_momentum = true; break; }

    if (any_momentum) {
        // Per-parameter update; each entry carries its resolved
        // hyperparams so different groups can use different momentum.
        for (const auto& e : cpu) {
            auto& vel  = velocity_buffers_[e.idx];
            auto& p    = *e.param;
            auto dtype = p.dtype();
            auto dev   = p.device();

            if (e.hp.momentum > 0.0) {
                vel = vel * full({1}, e.hp.momentum, dtype, dev) +
                      e.grad * full({1}, 1.0 - e.hp.dampening, dtype, dev);

                Tensor eff_grad = e.hp.nesterov
                    ? e.grad + vel * full({1}, e.hp.momentum, dtype, dev)
                    : vel;

                p = p - eff_grad * full({1}, e.hp.lr, dtype, dev);
            } else {
                // momentum == 0 inside a mixed-momentum batch: vanilla SGD.
                p = p - e.grad * full({1}, e.hp.lr, dtype, dev);
            }
        }
        return;
    }

    // No momentum anywhere in the batch: vanilla SGD via _foreach_*.
    std::vector<Tensor> params_view;
    std::vector<Tensor> grads;
    std::vector<Tensor> lr_list;
    params_view.reserve(cpu.size());
    grads.reserve(cpu.size());
    lr_list.reserve(cpu.size());
    for (const auto& e : cpu) {
        params_view.push_back(*e.param);
        grads.push_back(e.grad);
        lr_list.push_back(full({1}, e.hp.lr, e.param->dtype(), e.param->device()));
    }
    auto scaled = foreach_mul(grads, lr_list);
    foreach_sub_(params_view, scaled);
    for (size_t k = 0; k < cpu.size(); ++k) {
        *cpu[k].param = params_view[k];
    }
}

auto SGD::set_lr(double lr) -> void {
    // HH.14: rescale every ParamGroup's lr by lr/old_lr so per-group
    // relative LRs survive scheduler.step() (PyTorch convention).
    const double old_lr = lr_;
    lr_ = lr;
    if (old_lr == 0.0) {
        for (auto& g : param_groups_) g.lr = lr;
    } else {
        const double scale = lr / old_lr;
        for (auto& g : param_groups_) g.lr *= scale;
    }
}

auto SGD::get_lr() const -> double {
    return lr_;
}

auto SGD::initialize_buffers() -> void {
    velocity_buffers_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 velocity buffers.
            velocity_buffers_.push_back(make_optim_state(param->tensor()));
        }
    }
}

// Audit K.1: extend velocity_buffers_ when add_param_group appends new
// params.  velocity_buffers_ is indexed by parameter position so the
// new entries must match parameters_.size() after the append.
auto SGD::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    velocity_buffers_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see SGD::initialize_buffers for dtype rationale.
            velocity_buffers_.push_back(make_optim_state(param->tensor()));
        } else {
            velocity_buffers_.push_back(Tensor{});
        }
    }
}

auto SGD::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Save optimizer configuration as tensors
    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["momentum"] = Tensor({1}, DType::Float64, Device::cpu());
    state["momentum"].data<double>()[0] = momentum_;

    state["dampening"] = Tensor({1}, DType::Float64, Device::cpu());
    state["dampening"].data<double>()[0] = dampening_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    state["nesterov"] = Tensor({1}, DType::Int64, Device::cpu());
    state["nesterov"].data<int64_t>()[0] = nesterov_ ? 1 : 0;

    // Save velocity buffers
    for (size_t i = 0; i < velocity_buffers_.size(); ++i) {
        state["velocity_" + std::to_string(i)] = velocity_buffers_[i].clone();
    }

    // Audit item D.5: buffer-count guard (matches Adam pattern).
    state["num_params"] = Tensor({1}, DType::Int64, Device::cpu());
    state["num_params"].data<int64_t>()[0] = static_cast<int64_t>(velocity_buffers_.size());

    return state;
}

auto SGD::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Audit item D.5: validate buffer count before any restore so we
    // never silently mis-align velocity buffers to the wrong parameters.
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.rfind("velocity_", 0) == 0) ++saved_count;
    }
    if (saved_count > 0 && saved_count != velocity_buffers_.size()) {
        throw std::runtime_error(
            "SGD::load_state_dict: velocity buffer count mismatch - "
            "saved " + std::to_string(saved_count) + " but have " +
            std::to_string(velocity_buffers_.size()) + " parameters");
    }
    if (state.count("num_params")) {
        const int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(velocity_buffers_.size())) {
            throw std::runtime_error(
                "SGD::load_state_dict: parameter count mismatch - saved " +
                std::to_string(expected) + " but have " +
                std::to_string(velocity_buffers_.size()));
        }
    }

    // Load optimizer configuration
    if (state.count("lr")) {
        lr_ = state.at("lr").data<double>()[0];
    }

    if (state.count("momentum")) {
        momentum_ = state.at("momentum").data<double>()[0];
    }

    if (state.count("dampening")) {
        dampening_ = state.at("dampening").data<double>()[0];
    }

    if (state.count("weight_decay")) {
        weight_decay_ = state.at("weight_decay").data<double>()[0];
    }

    if (state.count("nesterov")) {
        nesterov_ = state.at("nesterov").data<int64_t>()[0] != 0;
    }

    // Load velocity buffers.
    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < velocity_buffers_.size(); ++i) {
        std::string velocity_key = "velocity_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(velocity_key)) {
            velocity_buffers_[i] = state.at(velocity_key).to(state_dt);
        }
    }
}

} // namespace tenzor::optim
