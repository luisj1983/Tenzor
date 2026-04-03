/**
 * @file parametrize.cpp
 * @brief Implementation of general parameter reparameterization framework
 */

#include "tenzor/nn/utils/parametrize.hpp"
#include <stdexcept>
#include <iostream>
#include <cstring>

namespace tenzor::nn::utils {

// Global map from module pointer to its parametrization lists
// Using raw pointer as key (modules are always accessed via shared_ptr)
static std::unordered_map<Module*, std::unordered_map<std::string, ParametrizationList>>&
parametrization_registry() {
    static std::unordered_map<Module*, std::unordered_map<std::string, ParametrizationList>> reg;
    return reg;
}

void register_parametrization(std::shared_ptr<Module> module,
                              const std::string& param_name,
                              std::shared_ptr<Parametrization> parametrization) {
    if (!module) {
        throw std::runtime_error("register_parametrization: module is null");
    }
    if (!parametrization) {
        throw std::runtime_error("register_parametrization: parametrization is null");
    }

    auto* mod_ptr = module.get();
    auto& mod_params = parametrization_registry()[mod_ptr];

    if (mod_params.find(param_name) == mod_params.end()) {
        // First parametrization on this parameter — save original
        auto param = module->get_parameter(param_name);
        if (!param) {
            throw std::runtime_error(
                "register_parametrization: parameter '" + param_name + "' not found");
        }

        ParametrizationList list;
        list.original = param->tensor().clone();
        list.chain.push_back(parametrization);

        // Register a forward pre-hook that applies the parametrization chain
        auto hook_id = module->register_forward_pre_hook(
            [mod_ptr, param_name](Module* /*self*/, const Variable& /*input*/) {
                auto& reg = parametrization_registry();
                auto mod_it = reg.find(mod_ptr);
                if (mod_it == reg.end()) return;
                auto param_it = mod_it->second.find(param_name);
                if (param_it == mod_it->second.end()) return;

                // Apply chain: original -> p1 -> p2 -> ... -> pN
                Tensor value = param_it->second.original;
                for (auto& p : param_it->second.chain) {
                    value = p->forward(value);
                }

                // Update the parameter in-place
                // This is done by modifying the parameter's data
                auto param = mod_ptr->get_parameter(param_name);
                if (param) {
                    param->tensor().fill_(0.0);  // Clear
                    // Copy parametrized value
                    // Use add_ to write the new values
                    auto& t = param->tensor();
                    // Simple approach: update via the underlying storage
                    auto src = value.contiguous();
                    auto dst_numel = t.numel();
                    auto src_numel = src.numel();
                    if (dst_numel == src_numel && t.dtype() == src.dtype()) {
                        std::memcpy(t.data<uint8_t>(), src.data<uint8_t>(),
                                    src_numel * dtype_size(src.dtype()));
                    }
                }
            }
        );
        list.hook_id = hook_id;
        mod_params[param_name] = std::move(list);
    } else {
        // Additional parametrization — append to chain
        mod_params[param_name].chain.push_back(parametrization);
    }

    // Note: parametrization modules are tracked in the registry.
    // For full state_dict integration, the module would need to expose
    // register_module publicly or provide a friend declaration.
}

void remove_parametrizations(std::shared_ptr<Module> module,
                             const std::string& param_name,
                             bool leave_parametrized) {
    if (!module) return;

    auto* mod_ptr = module.get();
    auto& reg = parametrization_registry();
    auto mod_it = reg.find(mod_ptr);
    if (mod_it == reg.end()) return;

    auto param_it = mod_it->second.find(param_name);
    if (param_it == mod_it->second.end()) return;

    auto& list = param_it->second;

    if (!leave_parametrized) {
        // Restore original parameter value
        auto param = module->get_parameter(param_name);
        if (param) {
            auto src = list.original.contiguous();
            auto& t = param->tensor();
            if (t.numel() == src.numel() && t.dtype() == src.dtype()) {
                std::memcpy(t.data<uint8_t>(), src.data<uint8_t>(),
                            src.numel() * dtype_size(src.dtype()));
            }
        }
    }

    // Remove the hook
    module->remove_hook(list.hook_id);

    // Clean up registry
    mod_it->second.erase(param_it);
    if (mod_it->second.empty()) {
        reg.erase(mod_it);
    }
}

auto is_parametrized(const Module& module, const std::string& param_name) -> bool {
    auto& reg = parametrization_registry();
    auto mod_it = reg.find(const_cast<Module*>(&module));
    if (mod_it == reg.end()) return false;

    if (param_name.empty()) {
        return !mod_it->second.empty();
    }
    return mod_it->second.find(param_name) != mod_it->second.end();
}

} // namespace tenzor::nn::utils
