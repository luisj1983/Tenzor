#include "tenzor/nn/optim/optimizer.hpp"
#include "tenzor/nn/utils/clip_grad.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/device.hpp"
#include <algorithm>
#include "tenzor/backend/loader.hpp"
#include <cstring>
#include <stdexcept>

namespace tenzor::optim {

Optimizer::Optimizer(std::vector<std::shared_ptr<Variable>> params)
    : parameters_(std::move(params)) {}

Optimizer::Optimizer(std::vector<ParamGroup> groups)
    : param_groups_(std::move(groups)) {
    // Flatten all group params into the main parameters_ list
    for (auto& group : param_groups_) {
        for (auto& p : group.params) {
            parameters_.push_back(p);
        }
    }
}

auto Optimizer::step() -> void {
    clip_gradients_();

    // Coalesce sparse gradients before the optimizer step.
    // This ensures sparse-aware optimizers receive coalesced indices
    // (no duplicate indices), which is required for correct updates.
    for (auto& param : parameters_) {
        if (param && param->has_sparse_grad()) {
            auto& sg = param->mutable_sparse_grad().value();
            if (!sg.is_coalesced()) {
                sg = sg.coalesce();
            }
        }
    }

    step_impl();

    // Audit G.10: fire post-step hooks (e.g. pruning mask
    // reapplication) after the parameter update has been applied.
    // Hooks fire in registration order; exceptions propagate.
    fire_post_step_hooks_();
}

auto Optimizer::register_post_step_hook(PostStepHook hook) -> uint64_t {
    uint64_t id = next_hook_id_.fetch_add(1, std::memory_order_relaxed);
    post_step_hooks_.emplace_back(id, std::move(hook));
    return id;
}

auto Optimizer::remove_post_step_hook(uint64_t hook_id) -> bool {
    auto it = std::find_if(post_step_hooks_.begin(), post_step_hooks_.end(),
                            [hook_id](const auto& entry) {
                                return entry.first == hook_id;
                            });
    if (it == post_step_hooks_.end()) {
        return false;
    }
    post_step_hooks_.erase(it);
    return true;
}

auto Optimizer::fire_post_step_hooks_() -> void {
    for (auto& [_id, hook] : post_step_hooks_) {
        if (hook) {
            hook();
        }
    }
}

auto Optimizer::find_group_for_param(size_t param_index) const -> const ParamGroup* {
    // Audit D.4: linear lookup of the ParamGroup owning parameters_[i].
    // For the typical 1–5 param groups this is fine; if profiling shows
    // it as a hotspot, swap with a cached index built at flatten time.
    if (param_index >= parameters_.size()) return nullptr;
    const auto& target = parameters_[param_index];
    for (const auto& g : param_groups_) {
        for (const auto& p : g.params) {
            if (p.get() == target.get()) {
                return &g;
            }
        }
    }
    return nullptr;
}

auto Optimizer::step(std::function<Variable()> closure) -> Variable {
    auto loss = closure();
    step();
    return loss;
}

auto Optimizer::clip_gradients_() -> void {
    switch (clip_config_.mode) {
        case ClipMode::None:
            break;
        case ClipMode::Norm:
            nn::utils::clip_grad_norm_(parameters_, clip_config_.max_norm, clip_config_.norm_type);
            break;
        case ClipMode::Value:
            nn::utils::clip_grad_value_(parameters_, clip_config_.max_norm);
            break;
    }
}

auto Optimizer::set_clip_config(const ClipConfig& config) -> void {
    clip_config_ = config;
}

auto Optimizer::clip_config() const -> const ClipConfig& {
    return clip_config_;
}

auto Optimizer::add_param_group(ParamGroup group) -> void {
    // Audit K.1: extend optimizer state buffers when new parameters
    // are added mid-training.  Without this hook, derived optimisers
    // (Adam / SGD / Adagrad / ...) silently OOB-read exp_avg_[i],
    // momentum_buffer_[i], sum_[i], etc. on the next step_impl().
    const size_t old_count = parameters_.size();
    for (auto& p : group.params) {
        parameters_.push_back(p);
    }
    const size_t new_count = parameters_.size();
    param_groups_.push_back(std::move(group));
    if (new_count > old_count) {
        on_parameters_appended_(old_count, new_count);
    }
}

auto Optimizer::on_parameters_appended_(size_t /*old_count*/, size_t /*new_count*/) -> void {
    // Default refuses to silently no-op — every concrete optimiser is
    // required to override and extend its state buffers.  See K.1.
    throw std::runtime_error(
        "Optimizer::on_parameters_appended_: derived optimizer did not "
        "override the state-extension hook required by add_param_group(). "
        "Without an override, the next step_impl() would OOB-read state "
        "buffers indexed by parameter position.  See audit K.1.");
}

auto Optimizer::param_groups() -> std::vector<ParamGroup>& {
    return param_groups_;
}

auto Optimizer::param_groups() const -> const std::vector<ParamGroup>& {
    return param_groups_;
}

auto Optimizer::zero_grad() -> void {
    for (auto& param : parameters_) {
        if (!param) continue;

        // Clear sparse gradient if present
        if (param->has_sparse_grad()) {
            param->clear_sparse_grad();
        }

        if (param->has_grad()) {
            auto& grad = param->mutable_grad().value();

            // CPU path: use direct memset for maximum performance
            if (grad.device().type == Device::Type::CPU) {
                std::memset(grad.data_ptr(), 0, grad.numel() * dtype_size(grad.dtype()));
            } else {
                // GPU path: in-place zero via backend memset (avoids allocation)
                auto* backend = backend_registry().get_backend(grad.device().type);
                if (!backend) {
                    throw std::runtime_error(
                        "Optimizer::zero_grad: no backend registered for device "
                        + grad.device().to_string());
                }
                backend->memset(grad.data_ptr(), 0,
                                grad.numel() * dtype_size(grad.dtype()),
                                grad.device().index);
            }
        }
    }
}

auto Optimizer::parameters() const -> const std::vector<std::shared_ptr<Variable>>& {
    return parameters_;
}

auto Optimizer::replace_parameters(std::vector<std::shared_ptr<Variable>> new_params) -> void {
    if (new_params.size() != parameters_.size()) {
        throw std::invalid_argument(
            "replace_parameters: size mismatch (" + std::to_string(new_params.size()) +
            " vs " + std::to_string(parameters_.size()) + ")");
    }
    parameters_ = std::move(new_params);
}

auto Optimizer::save_state(const std::string& path) const -> void {
    auto state = state_dict();
    nn::Serializer::save(state, path);
}

auto Optimizer::load_state(const std::string& path) -> void {
    auto state = nn::Serializer::load(path);
    load_state_dict(state);
}

auto Optimizer::set_lr(double lr) -> void {
    // Default implementation: write lr into every ParamGroup so any optimizer
    // that uses the standard group container picks it up on the next step().
    // Optimizers without param_groups_ should override this method.
    if (param_groups_.empty()) {
        throw std::runtime_error(
            "Optimizer::set_lr(): no parameter groups registered; "
            "derived optimizer must override set_lr() or populate param_groups_.");
    }
    for (auto& group : param_groups_) {
        group.lr = lr;
    }
}

auto Optimizer::get_lr() const -> double {
    // Default implementation: return the lr of the first ParamGroup. Mirrors
    // PyTorch's convention (param_groups[0]['lr']). When groups carry distinct
    // learning rates, callers that need per-group lrs should use param_groups()
    // directly. Optimizers without param_groups_ should override.
    if (param_groups_.empty()) {
        throw std::runtime_error(
            "Optimizer::get_lr(): no parameter groups registered; "
            "derived optimizer must override get_lr() or populate param_groups_.");
    }
    return param_groups_.front().lr;
}

auto Optimizer::defaults() const -> std::unordered_map<std::string, double> {
    // Base implementation: the universal hyperparameters every group carries
    // (lr, weight_decay), read from the first ParamGroup. Concrete optimizers
    // override to merge in their own state (momentum/eps/beta*/etc.).
    if (param_groups_.empty()) {
        return {};
    }
    const auto& g = param_groups_.front();
    return {
        {"lr",           g.lr},
        {"weight_decay", g.weight_decay},
    };
}

} // namespace tenzor::optim
