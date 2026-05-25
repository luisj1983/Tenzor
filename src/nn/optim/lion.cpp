#include "tenzor/nn/optim/lion.hpp"
#include "tenzor/nn/optim/master_weights.hpp"

#include "tenzor/core/device.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor::optim {

Lion::Lion(std::vector<std::shared_ptr<Variable>> params,
           double lr, double beta1, double beta2, double weight_decay)
    : Optimizer(std::move(params)),
      lr_(lr), beta1_(beta1), beta2_(beta2), weight_decay_(weight_decay) {
    initialize_buffers();
}

Lion::Lion(std::vector<optim::ParamGroup> groups,
           double default_lr, double default_beta1,
           double default_beta2, double default_weight_decay)
    : Optimizer(std::move(groups)),
      lr_(default_lr),
      beta1_(default_beta1),
      beta2_(default_beta2),
      weight_decay_(default_weight_decay) {
    initialize_buffers();
}

auto Lion::step_impl() -> void {
    step_count_++;

    // Audit D.4: per-parameter hyperparameters resolve from the active
    // ParamGroup (when one was set up) or fall through to the optimiser-wide
    // defaults stored on this Lion instance.
    struct LionHP {
        double lr;
        double beta1;
        double beta2;
        double weight_decay;
    };

    auto resolve = [&](size_t i) -> LionHP {
        LionHP hp{lr_, beta1_, beta2_, weight_decay_};
        if (const auto* g = find_group_for_param(i)) {
            hp.lr           = g->lr;
            hp.weight_decay = g->weight_decay;
            hp.beta1        = ParamGroup::or_else(g->beta1, beta1_);
            hp.beta2        = ParamGroup::or_else(g->beta2, beta2_);
        }
        return hp;
    };

    for (size_t i = 0; i < parameters_.size(); ++i) {
        auto& param_ptr = parameters_[i];
        if (!param_ptr || !param_ptr->has_grad()) continue;
        auto& param = *param_ptr;

        const LionHP hp = resolve(i);

        const Tensor& grad = param.grad().value();

        // R.16: run math in optimiser state dtype (Float32 for halves).
        const DType param_dt = param.tensor().dtype();
        const DType state_dt = optim_state_dtype(param_dt);
        const bool needs_upcast = (state_dt != param_dt);

        // Dtype-matched scalar helper (preserves Float64 precision that
        // static_cast<float> would truncate).
        auto scalar = [&](double value) -> Tensor {
            return full({1}, value, state_dt, param.tensor().device());
        };

        Tensor param_hi = needs_upcast ? param.tensor().to(state_dt) : param.tensor();
        Tensor grad_hi  = needs_upcast ? grad.to(state_dt) : grad;

        // c_t = beta1 * m_{t-1} + (1 - beta1) * g_t   — update direction
        auto c = momentum_[i] * scalar(hp.beta1) + grad_hi * scalar(1.0 - hp.beta1);

        // theta_t = theta_{t-1} - lr * (sign(c_t) + wd * theta_{t-1})
        Tensor step = sign(c);
        if (hp.weight_decay > 0.0) {
            step = step + param_hi * scalar(hp.weight_decay);
        }
        Tensor updated = param_hi - step * scalar(hp.lr);
        param.tensor() = needs_upcast ? updated.to(param_dt) : updated;

        // m_t = beta2 * m_{t-1} + (1 - beta2) * g_t   — momentum state
        momentum_[i] = momentum_[i] * scalar(hp.beta2) + grad_hi * scalar(1.0 - hp.beta2);
    }
}

auto Lion::initialize_buffers() -> void {
    momentum_.clear();
    for (auto& param : parameters_) {
        if (param) {
            // R.16: half-precision params get Float32 state buffers.
            momentum_.push_back(make_optim_state(param->tensor()));
        }
    }
}

// Audit K.1: extend momentum_ for parameters appended via add_param_group.
auto Lion::on_parameters_appended_(size_t old_count, size_t new_count) -> void {
    momentum_.reserve(new_count);
    for (size_t i = old_count; i < new_count; ++i) {
        const auto& param = parameters_[i];
        if (param) {
            // R.16: see Lion::initialize_buffers for dtype rationale.
            momentum_.push_back(make_optim_state(param->tensor()));
        } else {
            momentum_.push_back(Tensor{});
        }
    }
}

auto Lion::set_lr(double lr) -> void {
    // HH.14: rescale every ParamGroup's lr by lr/old_lr.
    const double old_lr = lr_;
    lr_ = lr;
    if (old_lr == 0.0) {
        for (auto& g : param_groups_) g.lr = lr;
    } else {
        const double scale = lr / old_lr;
        for (auto& g : param_groups_) g.lr *= scale;
    }
}
auto Lion::get_lr() const -> double { return lr_; }

auto Lion::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    state["step_count"] = Tensor({1}, DType::Int64, Device::cpu());
    state["step_count"].data<int64_t>()[0] = step_count_;

    state["lr"] = Tensor({1}, DType::Float64, Device::cpu());
    state["lr"].data<double>()[0] = lr_;

    state["beta1"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta1"].data<double>()[0] = beta1_;

    state["beta2"] = Tensor({1}, DType::Float64, Device::cpu());
    state["beta2"].data<double>()[0] = beta2_;

    state["weight_decay"] = Tensor({1}, DType::Float64, Device::cpu());
    state["weight_decay"].data<double>()[0] = weight_decay_;

    // Z.12: persist parameter count for fail-fast load (mirrors Adam / LAMB).
    Tensor n({1}, DType::Int64, Device::cpu());
    n.data<int64_t>()[0] = static_cast<int64_t>(momentum_.size());
    state["num_params"] = n;

    for (size_t i = 0; i < momentum_.size(); ++i) {
        state["momentum_" + std::to_string(i)] = momentum_[i].clone();
    }
    return state;
}

auto Lion::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Z.12: explicit num_params guard mirroring Adam / LAMB / SparseAdam.
    if (state.count("num_params")) {
        int64_t expected = state.at("num_params").data<int64_t>()[0];
        if (expected != static_cast<int64_t>(momentum_.size())) {
            throw std::runtime_error(
                "Lion::load_state_dict: parameter count mismatch (state has " +
                std::to_string(expected) + ", optimiser has " +
                std::to_string(momentum_.size()) + ")");
        }
    }

    if (state.count("step_count"))  step_count_   = state.at("step_count").data<int64_t>()[0];
    if (state.count("lr"))          lr_           = state.at("lr").data<double>()[0];
    if (state.count("beta1"))       beta1_        = state.at("beta1").data<double>()[0];
    if (state.count("beta2"))       beta2_        = state.at("beta2").data<double>()[0];
    if (state.count("weight_decay")) weight_decay_ = state.at("weight_decay").data<double>()[0];

    // Validate momentum buffer count matches current parameter count
    size_t saved_count = 0;
    for (const auto& [key, _] : state) {
        if (key.starts_with("momentum_")) ++saved_count;
    }
    if (saved_count > 0 && saved_count != momentum_.size()) {
        throw std::runtime_error(
            "Lion::load_state_dict: momentum buffer count mismatch - saved " +
            std::to_string(saved_count) + " but have " +
            std::to_string(momentum_.size()) + " parameters");
    }

    // V.27: cast to R.16 master-weights dtype on load.
    for (size_t i = 0; i < momentum_.size(); ++i) {
        std::string key = "momentum_" + std::to_string(i);
        const DType state_dt = (i < parameters_.size() && parameters_[i])
            ? optim_state_dtype(parameters_[i]->tensor().dtype())
            : DType::Float32;
        if (state.count(key)) {
            momentum_[i] = state.at(key).to(state_dt);
        }
    }
}

} // namespace tenzor::optim
