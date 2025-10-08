#include "tenzor/nn/module.hpp"

namespace tenzor::nn {

auto Module::parameters() -> std::vector<Variable*> {
    std::vector<Variable*> params;

    // Add own parameters
    for (auto& [name, param] : parameters_) {
        params.push_back(&param);
    }

    // Add submodule parameters
    for (auto& [name, module] : submodules_) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }

    return params;
}

auto Module::named_parameters() -> std::vector<std::pair<std::string, Variable*>> {
    std::vector<std::pair<std::string, Variable*>> params;

    for (auto& [name, param] : parameters_) {
        params.emplace_back(name, &param);
    }

    for (auto& [name, module] : submodules_) {
        auto sub_params = module->named_parameters();
        for (auto& [sub_name, sub_param] : sub_params) {
            params.emplace_back(name + "." + sub_name, sub_param);
        }
    }

    return params;
}

auto Module::train(bool mode) -> void {
    training_ = mode;
    for (auto& [_, module] : submodules_) {
        module->train(mode);
    }
}

auto Module::eval() -> void {
    train(false);
}

auto Module::to(Device device) -> void {
    for (auto& [_, param] : parameters_) {
        param.tensor() = param.tensor().to(device);
    }

    for (auto& [_, module] : submodules_) {
        module->to(device);
    }
}

auto Module::cuda(int device_id) -> void {
    to(Device::cuda(device_id));
}

auto Module::cpu() -> void {
    to(Device::cpu());
}

auto Module::zero_grad() -> void {
    for (auto* param : parameters()) {
        param->zero_grad();
    }
}

auto Module::register_parameter(std::string name, Variable param) -> void {
    parameters_[std::move(name)] = std::move(param);
}

auto Module::register_buffer(std::string name, Variable buffer) -> void {
    buffers_[std::move(name)] = std::move(buffer);
}

auto Module::register_module(std::string name, std::shared_ptr<Module> module) -> void {
    submodules_[std::move(name)] = std::move(module);
}

// Sequential implementation
auto Sequential::add_module(std::shared_ptr<Module> module) -> Sequential& {
    modules_.push_back(std::move(module));
    return *this;
}

auto Sequential::forward(const Variable& input) -> Variable {
    auto output = input;
    for (auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
}

} // namespace tenzor::nn
