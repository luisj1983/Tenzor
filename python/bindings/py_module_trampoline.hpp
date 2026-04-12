// PyModule trampoline class for Python subclassing of nn::Module.
//
// Shared header so both bindings.cpp (which defines PYBIND11_MODULE)
// and bindings_nn.cpp (which registers the nn submodule) can see the
// trampoline type.

#pragma once

#include <pybind11/pybind11.h>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/module.hpp>

namespace py = pybind11;

using VariablePtr = std::shared_ptr<tenzor::Variable>;
using VariablePtrVec = std::vector<VariablePtr>;
using NamedParamPair = std::pair<std::string, VariablePtr>;
using NamedParamVec = std::vector<NamedParamPair>;
using StateDict = std::unordered_map<std::string, tenzor::Tensor>;

class PyModule : public tenzor::nn::Module {
public:
    using tenzor::nn::Module::Module;

    auto forward_impl(const tenzor::Variable& input) -> tenzor::Variable override {
        py::gil_scoped_acquire gil;
        py::function forward_override = py::get_override(this, "forward");
        if (forward_override) {
            py::object result = forward_override(input);
            if (!py::isinstance<tenzor::Variable>(result)) {
                throw std::runtime_error(
                    std::string("forward() must return a tenzor.Variable, got ") +
                    std::string(py::str(py::type::handle_of(result).attr("__name__"))));
            }
            return result.cast<tenzor::Variable>();
        }
        PYBIND11_OVERRIDE_PURE(
            tenzor::Variable,
            tenzor::nn::Module,
            forward_impl,
            input
        );
    }

    auto parameters() -> VariablePtrVec override {
        PYBIND11_OVERRIDE(VariablePtrVec, tenzor::nn::Module, parameters);
    }

    auto own_parameters() -> VariablePtrVec override {
        PYBIND11_OVERRIDE(VariablePtrVec, tenzor::nn::Module, own_parameters);
    }

    auto named_parameters() -> NamedParamVec override {
        PYBIND11_OVERRIDE(NamedParamVec, tenzor::nn::Module, named_parameters);
    }

    auto state_dict() const -> StateDict override {
        PYBIND11_OVERRIDE(StateDict, tenzor::nn::Module, state_dict);
    }

    auto load_state_dict(const StateDict& state) -> void override {
        PYBIND11_OVERRIDE(void, tenzor::nn::Module, load_state_dict, state);
    }

    void py_register_parameter(const std::string& name, tenzor::Variable param) {
        register_parameter(name, std::move(param));
    }

    void py_register_buffer(const std::string& name, tenzor::Variable buffer) {
        register_buffer(name, std::move(buffer));
    }

    void py_register_module(const std::string& name, std::shared_ptr<tenzor::nn::Module> module) {
        register_module(name, std::move(module));
    }
};
