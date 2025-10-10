#include "tenzor/nn/module.hpp"
#include "tenzor/nn/serialize.hpp"
#include <iostream>
#include <algorithm>

namespace tenzor::nn {

auto Module::parameters() -> std::vector<Variable*> {
    std::vector<Variable*> params;

    // Add own parameters in a consistent order (weight before bias)
    // This ensures tests can rely on params[0] being weight
    if (parameters_.find("weight") != parameters_.end()) {
        params.push_back(&parameters_["weight"]);
    }
    if (parameters_.find("bias") != parameters_.end()) {
        params.push_back(&parameters_["bias"]);
    }

    // Add any other parameters not named "weight" or "bias"
    for (auto& [name, param] : parameters_) {
        if (name != "weight" && name != "bias") {
            params.push_back(&param);
        }
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
    // Transfer parameters
    for (auto& [_, param] : parameters_) {
        param.tensor() = param.tensor().to(device);
    }

    // Transfer buffers (running_mean, running_var, etc.)
    for (auto& [_, buffer] : buffers_) {
        buffer.tensor() = buffer.tensor().to(device);
    }

    // Recursively transfer submodules
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

auto Module::buffers() -> std::vector<Variable*> {
    std::vector<Variable*> bufs;

    for (auto& [name, buffer] : buffers_) {
        bufs.push_back(&buffer);
    }

    for (auto& [name, module] : submodules_) {
        auto sub_bufs = module->buffers();
        bufs.insert(bufs.end(), sub_bufs.begin(), sub_bufs.end());
    }

    return bufs;
}

auto Module::named_buffers() -> std::vector<std::pair<std::string, Variable*>> {
    std::vector<std::pair<std::string, Variable*>> bufs;

    for (auto& [name, buffer] : buffers_) {
        bufs.emplace_back(name, &buffer);
    }

    for (auto& [name, module] : submodules_) {
        auto sub_bufs = module->named_buffers();
        for (auto& [sub_name, sub_buf] : sub_bufs) {
            bufs.emplace_back(name + "." + sub_name, sub_buf);
        }
    }

    return bufs;
}

auto Module::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;

    // Add own parameters
    for (const auto& [name, param] : parameters_) {
        state[name] = param.tensor().clone();
    }

    // Add own buffers
    for (const auto& [name, buffer] : buffers_) {
        state[name] = buffer.tensor().clone();
    }

    // Add submodule state with prefixed names
    for (const auto& [name, module] : submodules_) {
        auto sub_state = module->state_dict();
        for (auto& [sub_name, tensor] : sub_state) {
            state[name + "." + sub_name] = std::move(tensor);
        }
    }

    return state;
}

auto Module::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load own parameters
    for (auto& [name, param] : parameters_) {
        auto it = state.find(name);
        if (it != state.end()) {
            // Verify shapes match
            const auto& loaded_tensor = it->second;
            if (param.tensor().shape().size() != loaded_tensor.shape().size() ||
                !std::equal(param.tensor().shape().begin(), param.tensor().shape().end(),
                           loaded_tensor.shape().begin())) {
                throw std::runtime_error("Shape mismatch for parameter '" + name + "'");
            }
            if (param.tensor().dtype() != loaded_tensor.dtype()) {
                throw std::runtime_error("DType mismatch for parameter '" + name + "'");
            }
            // Copy data
            param.tensor() = loaded_tensor.clone();
        }
    }

    // Load own buffers
    for (auto& [name, buffer] : buffers_) {
        auto it = state.find(name);
        if (it != state.end()) {
            const auto& loaded_tensor = it->second;
            if (buffer.tensor().shape().size() != loaded_tensor.shape().size() ||
                !std::equal(buffer.tensor().shape().begin(), buffer.tensor().shape().end(),
                           loaded_tensor.shape().begin())) {
                throw std::runtime_error("Shape mismatch for buffer '" + name + "'");
            }
            if (buffer.tensor().dtype() != loaded_tensor.dtype()) {
                throw std::runtime_error("DType mismatch for buffer '" + name + "'");
            }
            buffer.tensor() = loaded_tensor.clone();
        }
    }

    // Load submodule state
    for (auto& [name, module] : submodules_) {
        // Extract submodule state with matching prefix
        std::unordered_map<std::string, Tensor> sub_state;
        std::string prefix = name + ".";
        for (const auto& [key, tensor] : state) {
            if (key.rfind(prefix, 0) == 0) {
                // Key starts with prefix
                std::string sub_key = key.substr(prefix.length());
                sub_state[sub_key] = tensor;
            }
        }
        module->load_state_dict(sub_state);
    }
}

auto Module::save(const std::string& path) const -> void {
    auto state = state_dict();
    Serializer::save(state, path);
}

auto Module::load(const std::string& path) -> void {
    auto state = Serializer::load(path);
    load_state_dict(state);
}

// Sequential implementation
auto Sequential::add_module(std::shared_ptr<Module> module) -> Sequential& {
    // Generate unique name for this module
    std::string name = "module_" + std::to_string(modules_.size());

    // Add to both modules_ vector (for forward pass) and submodules_ map (for state_dict)
    modules_.push_back(module);
    register_module(name, module);

    return *this;
}

auto Sequential::forward(const Variable& input) -> Variable {
    auto output = input;
    for (auto& module : modules_) {
        output = module->forward(output);
    }
    return output;
}

auto Sequential::parameters() -> std::vector<Variable*> {
    std::vector<Variable*> params;
    // Iterate over modules_ in order (not submodules_ which is unordered)
    for (auto& module : modules_) {
        auto sub_params = module->parameters();
        params.insert(params.end(), sub_params.begin(), sub_params.end());
    }
    return params;
}

auto Sequential::named_parameters() -> std::vector<std::pair<std::string, Variable*>> {
    std::vector<std::pair<std::string, Variable*>> params;
    // Use module index to generate names in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i);
        auto sub_params = modules_[i]->named_parameters();
        for (auto& [sub_name, sub_param] : sub_params) {
            params.emplace_back(prefix + "." + sub_name, sub_param);
        }
    }
    return params;
}

auto Sequential::state_dict() const -> std::unordered_map<std::string, Tensor> {
    std::unordered_map<std::string, Tensor> state;
    // Use module index to generate names in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i);
        auto sub_state = modules_[i]->state_dict();
        for (auto& [sub_name, tensor] : sub_state) {
            state[prefix + "." + sub_name] = std::move(tensor);
        }
    }
    return state;
}

auto Sequential::load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void {
    // Load state for each module in order
    for (size_t i = 0; i < modules_.size(); ++i) {
        std::string prefix = "module_" + std::to_string(i) + ".";

        // Extract submodule state with matching prefix
        std::unordered_map<std::string, Tensor> sub_state;
        for (const auto& [key, tensor] : state) {
            if (key.rfind(prefix, 0) == 0) {
                // Key starts with prefix
                std::string sub_key = key.substr(prefix.length());
                sub_state[sub_key] = tensor;
            }
        }

        // Load state into this module
        modules_[i]->load_state_dict(sub_state);
    }
}

} // namespace tenzor::nn
