#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include "../autograd/variable.hpp"
#include "../core/device.hpp"

namespace tenzor {
namespace nn {

// Base module class
class Module {
public:
    virtual ~Module() = default;

    // Forward pass (pure virtual)
    virtual auto forward(const Variable& input) -> Variable = 0;

    // Convenience operator
    auto operator()(const Variable& input) -> Variable {
        return forward(input);
    }

    // Parameter management
    virtual auto parameters() -> std::vector<Variable*>;
    virtual auto named_parameters() -> std::vector<std::pair<std::string, Variable*>>;
    auto buffers() -> std::vector<Variable*>;
    auto named_buffers() -> std::vector<std::pair<std::string, Variable*>>;

    // Training mode
    auto train(bool mode = true) -> void;
    auto eval() -> void;
    auto is_training() const -> bool { return training_; }

    // Device management
    auto to(Device device) -> void;
    auto cuda(int device_id = 0) -> void;
    auto cpu() -> void;

    // Zero gradients
    auto zero_grad() -> void;

    // State management
    virtual auto state_dict() const -> std::unordered_map<std::string, Tensor>;
    virtual auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void;

    // Serialization
    auto save(const std::string& path) const -> void;
    auto load(const std::string& path) -> void;

protected:
    // Register parameters
    auto register_parameter(std::string name, Variable param) -> void;
    auto register_buffer(std::string name, Variable buffer) -> void;
    auto register_module(std::string name, std::shared_ptr<Module> module) -> void;

    bool training_{true};
    std::unordered_map<std::string, Variable> parameters_;
    std::unordered_map<std::string, Variable> buffers_;
    std::unordered_map<std::string, std::shared_ptr<Module>> submodules_;
};

// Sequential container
class Sequential : public Module {
public:
    Sequential() = default;

    // Variadic template constructor
    template<typename... Modules>
    explicit Sequential(std::shared_ptr<Modules>... modules) {
        (add_module(modules), ...);
    }

    // Add module
    auto add_module(std::shared_ptr<Module> module) -> Sequential&;

    // Forward pass through all modules
    auto forward(const Variable& input) -> Variable override;

    // Override to preserve module order
    auto parameters() -> std::vector<Variable*> override;
    auto named_parameters() -> std::vector<std::pair<std::string, Variable*>> override;
    auto state_dict() const -> std::unordered_map<std::string, Tensor> override;
    auto load_state_dict(const std::unordered_map<std::string, Tensor>& state) -> void override;

private:
    std::vector<std::shared_ptr<Module>> modules_;
};

} // namespace nn
} // namespace tenzor
