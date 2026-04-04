#include "tenzor/autograd/graph_viz.hpp"
#include "tenzor/autograd/function.hpp"
#include <sstream>
#include <fstream>
#include <typeinfo>
#include <unordered_set>
#include <queue>

namespace tenzor {

auto make_dot(const Variable& root,
              const std::unordered_map<std::string, Variable>& params,
              const GraphVizOptions& options) -> std::string {
    std::ostringstream out;
    out << "digraph computation_graph {\n";
    out << "  rankdir=BT;\n";  // Bottom to top (loss at top)
    out << "  node [shape=box, style=filled, fontsize=10];\n\n";

    // Build param lookup: grad_fn pointer → param name
    std::unordered_map<const void*, std::string> param_names;
    for (auto& [name, var] : params) {
        param_names[static_cast<const void*>(&var)] = name;
    }

    // BFS traversal of the computation graph
    std::unordered_set<const Function*> visited;
    std::queue<std::shared_ptr<Function>> queue;

    // Root node (the output variable)
    auto shape = root.shape();
    std::string shape_str;
    for (size_t i = 0; i < shape.size(); ++i) {
        if (i > 0) shape_str += "x";
        shape_str += std::to_string(shape[i]);
    }
    out << "  output [label=\"output\\n[" << shape_str << "]\", fillcolor=\"#caff70\"];\n";

    if (root.grad_fn()) {
        queue.push(root.grad_fn());
        out << "  node_" << reinterpret_cast<uintptr_t>(root.grad_fn().get())
            << " -> output;\n";
    }

    while (!queue.empty()) {
        auto fn = queue.front();
        queue.pop();

        if (!fn || visited.count(fn.get())) continue;
        visited.insert(fn.get());

        uintptr_t fn_id = reinterpret_cast<uintptr_t>(fn.get());
        std::string fn_name = typeid(*fn).name();

        // Try to demangle common patterns
        // Strip leading digits (mangled length prefix) and namespace prefixes
        auto last_colon = fn_name.rfind(':');
        if (last_colon != std::string::npos) {
            fn_name = fn_name.substr(last_colon + 1);
        }

        out << "  node_" << fn_id
            << " [label=\"" << fn_name << "\", fillcolor=\"#add8e6\"];\n";

        // Traverse next_functions (predecessors in the computation graph)
        for (auto& next_fn : fn->next_functions()) {
            if (next_fn) {
                uintptr_t next_id = reinterpret_cast<uintptr_t>(next_fn.get());
                out << "  node_" << next_id << " -> node_" << fn_id << ";\n";
                if (!visited.count(next_fn.get())) {
                    queue.push(next_fn);
                }
            }
        }

        // Show input variables as leaf nodes
        for (auto& input_var : fn->input_variables()) {
            if (!input_var.grad_fn() && input_var.requires_grad()) {
                uintptr_t var_id = reinterpret_cast<uintptr_t>(&input_var);
                std::string label = "param";

                // Check if this is a named parameter
                auto it = param_names.find(static_cast<const void*>(&input_var));
                if (it != param_names.end()) {
                    label = it->second;
                }

                auto var_shape = input_var.shape();
                std::string var_shape_str;
                for (size_t i = 0; i < var_shape.size(); ++i) {
                    if (i > 0) var_shape_str += "x";
                    var_shape_str += std::to_string(var_shape[i]);
                }

                std::string extra_info;
                if (options.show_dtypes) {
                    extra_info += "\\n" + std::string(dtype_name(input_var.dtype()));
                }
                if (options.show_memory_usage) {
                    size_t bytes = input_var.tensor().numel() * dtype_size(input_var.dtype());
                    if (bytes >= 1024 * 1024) {
                        extra_info += "\\n" + std::to_string(bytes / (1024 * 1024)) + " MB";
                    } else if (bytes >= 1024) {
                        extra_info += "\\n" + std::to_string(bytes / 1024) + " KB";
                    } else {
                        extra_info += "\\n" + std::to_string(bytes) + " B";
                    }
                }
                if (options.show_sparse_annotations && input_var.has_sparse_grad()) {
                    extra_info += "\\n[sparse grad]";
                }

                out << "  var_" << var_id
                    << " [label=\"" << label << "\\n[" << var_shape_str
                    << "]" << extra_info << "\", fillcolor=\"#ffcccc\", shape=ellipse];\n";
                out << "  var_" << var_id << " -> node_"
                    << reinterpret_cast<uintptr_t>(fn.get()) << ";\n";
            }
        }
    }

    out << "}\n";
    return out.str();
}

auto save_dot(const std::string& dot, const std::string& path) -> void {
    std::ofstream file(path);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file for writing: " + path);
    }
    file << dot;
}

} // namespace tenzor
