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

auto GraphModule::run_graph(const std::vector<Variable>& inputs)
    -> std::unordered_map<std::string, Variable> {
    // Value map: name -> Variable. Keeping Variables (not bare Tensors)
    // preserves the autograd grad_fn chain across both module-based and
    // functional ops, so gradients flow back through submodules and params.
    //
    // Single source of truth for graph execution: every forward entry point
    // calls this exactly once, so the graph (and any stateful submodules such
    // as Dropout / BatchNorm running stats) executes once per forward — never
    // twice as the old duplicated forward_multi loop did.
    std::unordered_map<std::string, Variable> values;

    // Seed constants (graph-internal weights/constants do not require grad).
    for (auto& [name, tensor] : constants_) {
        values[name] = Variable(tensor, false);
    }

    // Bind every supplied input to its declared name in order. The count must
    // match exactly so a missing/extra input is reported here rather than as a
    // confusing "value not found" mid-execution.
    if (inputs.size() != input_names_.size()) {
        throw std::runtime_error(
            "GraphModule: graph declares " + std::to_string(input_names_.size()) +
            " input(s) but " + std::to_string(inputs.size()) +
            " were supplied");
    }
    for (size_t i = 0; i < input_names_.size(); ++i) {
        values[input_names_[i]] = inputs[i];
    }

    // Execute ops in topological order.
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
            output_vars.push_back(op.module->forward(input_vars[0]));
        } else if (op.compute_fn) {
            // Functional op (operates on/returns Variables, preserving grad_fn)
            output_vars = op.compute_fn(input_vars);
        } else {
            throw std::runtime_error(
                "GraphModule: op '" + op.name + "' has neither module nor compute_fn");
        }

        // The op must produce exactly as many Variables as it declares output
        // names. Validate here so an under-/over-producing op is reported at the
        // producer (with op name/type and expected-vs-produced counts) rather
        // than surfacing later as a misleading "value not found" / "output not
        // found" at a downstream consumer. Mirrors the early count checks used
        // for inputs (line 47) and module inputs (line 76).
        if (output_vars.size() != op.outputs.size()) {
            throw std::runtime_error(
                "GraphModule: op '" + op.name + "' (" + op.op_type +
                ") declares " + std::to_string(op.outputs.size()) +
                " output(s) but produced " + std::to_string(output_vars.size()));
        }
        for (size_t i = 0; i < op.outputs.size(); ++i) {
            values[op.outputs[i]] = output_vars[i];
        }
    }

    return values;
}

auto GraphModule::forward_impl(const Variable& input) -> Variable {
    // Single-Variable entry point can only seed input_names_[0]; reject graphs
    // with more inputs instead of silently leaving inputs[1..] unbound (which
    // surfaced as a confusing "value not found" at the first consuming op).
    if (input_names_.size() > 1) {
        throw std::runtime_error(
            "GraphModule: graph has " + std::to_string(input_names_.size()) +
            " inputs; call forward_multi(std::vector<Variable>) to supply them all");
    }

    // A graph with no declared inputs ignores the supplied Variable (constants
    // only), matching the prior behaviour that seeded inputs only when present.
    auto values = input_names_.empty() ? run_graph({}) : run_graph({input});

    if (output_names_.empty()) {
        throw std::runtime_error("GraphModule: no output names set");
    }
    auto it = values.find(output_names_[0]);
    if (it == values.end()) {
        throw std::runtime_error(
            "GraphModule: output '" + output_names_[0] + "' not found after execution");
    }
    // Return the first output, preserving its grad_fn. Use forward_multi for the
    // remaining outputs of a multi-output graph.
    return it->second;
}

auto GraphModule::forward_multi(const Variable& input) -> std::vector<Variable> {
    if (input_names_.size() > 1) {
        throw std::runtime_error(
            "GraphModule: graph has " + std::to_string(input_names_.size()) +
            " inputs; call forward_multi(std::vector<Variable>) to supply them all");
    }
    return forward_multi(input_names_.empty() ? std::vector<Variable>{}
                                              : std::vector<Variable>{input});
}

auto GraphModule::forward_multi(const std::vector<Variable>& inputs)
    -> std::vector<Variable> {
    auto values = run_graph(inputs);

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
