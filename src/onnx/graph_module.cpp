#include "tenzor/onnx/graph_module.hpp"
#include <stdexcept>

namespace tenzor {
namespace onnx {

void GraphModule::add_op(GraphOp op) {
    // Register any sub-module for parameter tracking
    if (op.module) {
        register_module(op.name, op.module);
    }
    ops_.push_back(std::move(op));
}

void GraphModule::add_constant(const std::string& name, Tensor value) {
    constants_[name] = std::move(value);
}

void GraphModule::set_input_names(std::vector<std::string> names) {
    input_names_ = std::move(names);
}

void GraphModule::set_output_names(std::vector<std::string> names) {
    output_names_ = std::move(names);
}

auto GraphModule::forward_impl(const Variable& input) -> Variable {
    // Value map: name -> Variable. Keeping Variables (not bare Tensors)
    // preserves the autograd grad_fn chain across both module-based and
    // functional ops, so gradients flow back through submodules and params.
    std::unordered_map<std::string, Variable> values;

    // Seed constants (graph-internal weights/constants do not require grad).
    for (auto& [name, tensor] : constants_) {
        values[name] = Variable(tensor, false);
    }

    // Seed graph inputs (propagate the real input's requires_grad).
    if (!input_names_.empty()) {
        values[input_names_[0]] = input;
    }

    // Execute ops in topological order
    for (auto& op : ops_) {
        // Gather input Variables
        std::vector<Variable> input_vars;
        input_vars.reserve(op.inputs.size());
        for (auto& input_name : op.inputs) {
            auto it = values.find(input_name);
            if (it == values.end()) {
                throw std::runtime_error(
                    "GraphModule: value '" + input_name + "' not found when "
                    "executing op '" + op.name + "' (" + op.op_type + ")");
            }
            input_vars.push_back(it->second);
        }

        // Execute
        std::vector<Variable> output_vars;

        if (op.module) {
            // Module-based op: nn::Module::forward takes a single Variable.
            // Multi-input module ops are not representable, so reject them
            // rather than silently dropping inputs[1..].
            if (input_vars.size() != 1) {
                throw std::runtime_error(
                    "GraphModule: module op '" + op.name + "' (" + op.op_type +
                    ") has " + std::to_string(input_vars.size()) +
                    " inputs but nn::Module::forward accepts exactly 1; route "
                    "multi-input ops through compute_fn instead");
            }
            auto var_output = op.module->forward(input_vars[0]);
            output_vars.push_back(var_output);
        } else if (op.compute_fn) {
            // Functional op (operates on/returns Variables, preserving grad_fn)
            output_vars = op.compute_fn(input_vars);
        } else {
            throw std::runtime_error(
                "GraphModule: op '" + op.name + "' has neither module nor compute_fn");
        }

        // Store outputs
        for (size_t i = 0; i < op.outputs.size() && i < output_vars.size(); ++i) {
            values[op.outputs[i]] = output_vars[i];
        }
    }

    // Collect graph output
    if (output_names_.empty()) {
        throw std::runtime_error("GraphModule: no output names set");
    }

    auto it = values.find(output_names_[0]);
    if (it == values.end()) {
        throw std::runtime_error(
            "GraphModule: output '" + output_names_[0] + "' not found after execution");
    }

    // Return the computed output Variable directly, preserving its grad_fn.
    // (Single-output contract of nn::Module::forward; use forward_multi for the
    // remaining outputs of a multi-output graph.)
    return it->second;
}

auto GraphModule::forward_multi(const Variable& input) -> std::vector<Variable> {
    // Reuse the single-output forward to execute the graph once, then look up
    // every declared output by name. forward_impl already validates the graph
    // and binds output_names_[0]; here we additionally surface outputs[1..].
    // We re-run the execution loop so all outputs are collected from the same
    // value map rather than relying on side state.
    std::unordered_map<std::string, Variable> values;

    for (auto& [name, tensor] : constants_) {
        values[name] = Variable(tensor, false);
    }
    if (!input_names_.empty()) {
        values[input_names_[0]] = input;
    }

    for (auto& op : ops_) {
        std::vector<Variable> input_vars;
        input_vars.reserve(op.inputs.size());
        for (auto& input_name : op.inputs) {
            auto it = values.find(input_name);
            if (it == values.end()) {
                throw std::runtime_error(
                    "GraphModule: value '" + input_name + "' not found when "
                    "executing op '" + op.name + "' (" + op.op_type + ")");
            }
            input_vars.push_back(it->second);
        }

        std::vector<Variable> output_vars;
        if (op.module) {
            if (input_vars.size() != 1) {
                throw std::runtime_error(
                    "GraphModule: module op '" + op.name + "' (" + op.op_type +
                    ") has " + std::to_string(input_vars.size()) +
                    " inputs but nn::Module::forward accepts exactly 1; route "
                    "multi-input ops through compute_fn instead");
            }
            output_vars.push_back(op.module->forward(input_vars[0]));
        } else if (op.compute_fn) {
            output_vars = op.compute_fn(input_vars);
        } else {
            throw std::runtime_error(
                "GraphModule: op '" + op.name + "' has neither module nor compute_fn");
        }

        for (size_t i = 0; i < op.outputs.size() && i < output_vars.size(); ++i) {
            values[op.outputs[i]] = output_vars[i];
        }
    }

    if (output_names_.empty()) {
        throw std::runtime_error("GraphModule: no output names set");
    }

    std::vector<Variable> outputs;
    outputs.reserve(output_names_.size());
    for (const auto& out_name : output_names_) {
        auto it = values.find(out_name);
        if (it == values.end()) {
            throw std::runtime_error(
                "GraphModule: output '" + out_name + "' not found after execution");
        }
        outputs.push_back(it->second);
    }
    return outputs;
}

} // namespace onnx
} // namespace tenzor
