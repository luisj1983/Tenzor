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
        // First parametrization on this parameter — save original.
        auto param = module->get_parameter(param_name);
        if (!param) {
            throw std::runtime_error(
                "register_parametrization: parameter '" + param_name + "' not found");
        }

        ParametrizationList list;
        list.original = param->tensor().clone();
        // Promote the original parameter Variable to a stable leaf that
        // owns its own buffer. Gradients accumulate here during backward;
        // the parameter slot inside the module is repointed each forward
        // to hold chain(original_var) so the autograd graph survives.
        list.original_var = std::make_shared<Variable>(
            list.original.clone(), param->requires_grad());
        list.chain.push_back(parametrization);

        // Register an autograd-aware forward pre-hook. The hook builds
        // chain.forward_impl(original_var) as a *Variable* computation,
        // then swaps the parameter slot's tensor + grad_fn so downstream
        // ops consume the parametrized value while still being able to
        // backpropagate through the chain into original_var (and into
        // any internal parameters owned by the parametrization modules
        // themselves, e.g. WeightNorm's weight_g / weight_v).
        auto hook_id = module->register_forward_pre_hook(
            [mod_ptr, param_name](Module* /*self*/, const Variable& /*input*/) {
                auto& reg = parametrization_registry();
                auto mod_it = reg.find(mod_ptr);
                if (mod_it == reg.end()) return;
                auto param_it = mod_it->second.find(param_name);
                if (param_it == mod_it->second.end()) return;

                auto& list = param_it->second;
                if (!list.original_var) return;

                // Run the chain as a Variable computation. Each
                // parametrization's forward_impl(Variable) is responsible
                // for building autograd-tracked ops; if a parametrization
                // overrides only forward(Tensor) the base default falls
                // back to a non-grad Variable (matching prior behavior).
                Variable value = *list.original_var;
                for (auto& p : list.chain) {
                    value = p->forward_impl(value);
                }

                // Repoint the module's parameter slot at the chain
                // output. set_data_view swaps the tensor storage without
                // disturbing other autograd state on the slot; set_grad_fn
                // installs the chain's tail so loss.backward() walks back
                // through every Variable op in the chain.
                auto param = mod_ptr->get_parameter(param_name);
                if (param) {
                    param->set_data_view(value.tensor());
                    param->set_grad_fn(value.grad_fn());
                    param->set_requires_grad(value.requires_grad());
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

    // Detach the autograd graph from the parameter slot — after this
    // call the slot must be a plain leaf again (no grad_fn pointing into
    // the now-defunct chain).
    auto param = module->get_parameter(param_name);
    if (param) {
        param->set_grad_fn(nullptr);
        if (leave_parametrized) {
            // Materialise the current chain output as a plain tensor so
            // the parameter is a self-contained leaf going forward.
            // (Cloning detaches it from list.original_var's storage.)
            param->set_data_view(param->tensor().clone());
        } else {
            // Restore the original (pre-parametrization) tensor.
            param->set_data_view(list.original.clone());
        }
        // Restore the requires_grad flag from the saved leaf so an
        // unused-but-tracked parameter doesn't silently become inert.
        if (list.original_var) {
            param->set_requires_grad(list.original_var->requires_grad());
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

// Test/teardown helper: clear the entire registry. Needed because the
// registry keys on raw Module* — when a Module is destroyed and a new one
// later gets the same address (typical std::allocator reuse), stale entries
// cause is_parametrized() to return true for what should be a fresh module.
// Call between TEST_P iterations.
void clear_parametrization_registry() {
    parametrization_registry().clear();
}

} // namespace tenzor::nn::utils
