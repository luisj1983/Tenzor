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
    // Value map: name -> tensor
    std::unordered_map<std::string, Tensor> values;

    // Seed constants
    for (auto& [name, tensor] : constants_) {
        values[name] = tensor;
    }

    // Seed graph inputs
    if (!input_names_.empty()) {
        values[input_names_[0]] = input.tensor();
    }

    // Execute ops in topological order
    for (auto& op : ops_) {
        // Gather input tensors
        std::vector<Tensor> input_tensors;
        input_tensors.reserve(op.inputs.size());
        for (auto& input_name : op.inputs) {
            auto it = values.find(input_name);
            if (it == values.end()) {
                throw std::runtime_error(
                    "GraphModule: value '" + input_name + "' not found when "
                    "executing op '" + op.name + "' (" + op.op_type + ")");
            }
            input_tensors.push_back(it->second);
        }

        // Execute
        std::vector<Tensor> output_tensors;

        if (op.module) {
            // Module-based op: pass first input through forward
            Variable var_input(input_tensors[0], false);
            auto var_output = op.module->forward(var_input);
            output_tensors.push_back(var_output.tensor());
        } else if (op.compute_fn) {
            // Functional op
            output_tensors = op.compute_fn(input_tensors);
        } else {
            throw std::runtime_error(
                "GraphModule: op '" + op.name + "' has neither module nor compute_fn");
        }

        // Store outputs
        for (size_t i = 0; i < op.outputs.size() && i < output_tensors.size(); ++i) {
            values[op.outputs[i]] = output_tensors[i];
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

    return Variable(it->second, input.requires_grad());
}

} // namespace onnx
} // namespace tenzor
