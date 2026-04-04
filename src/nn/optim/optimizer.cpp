#include "tenzor/nn/optim/optimizer.hpp"
#include "tenzor/nn/utils/clip_grad.hpp"
#include "tenzor/nn/serialize.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/device.hpp"
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
    for (auto& p : group.params) {
        parameters_.push_back(p);
    }
    param_groups_.push_back(std::move(group));
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

auto Optimizer::save_state(const std::string& path) const -> void {
    auto state = state_dict();
    nn::Serializer::save(state, path);
}

auto Optimizer::load_state(const std::string& path) -> void {
    auto state = nn::Serializer::load(path);
    load_state_dict(state);
}

auto Optimizer::set_lr(double /*lr*/) -> void {
    throw std::runtime_error("Optimizer::set_lr() not implemented for this optimizer type");
}

auto Optimizer::get_lr() const -> double {
    throw std::runtime_error("Optimizer::get_lr() not implemented for this optimizer type");
}

} // namespace tenzor::optim
