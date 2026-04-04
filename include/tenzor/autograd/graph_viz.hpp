#pragma once

#include "tenzor/autograd/variable.hpp"
#include <string>
#include <unordered_map>

namespace tenzor {

/**
 * @brief Options for graph visualization.
 */
struct GraphVizOptions {
    bool show_gradient_values{false};   ///< Display gradient norms on edges
    bool show_memory_usage{false};      ///< Display tensor memory footprint in nodes
    bool show_sparse_annotations{false}; ///< Mark sparse tensors/gradients
    bool show_dtypes{false};            ///< Show tensor dtypes on edges
};

/**
 * @brief Generate a Graphviz DOT representation of an autograd computation graph.
 *
 * Traverses the computation graph from a Variable's grad_fn backward through
 * next_functions, emitting nodes for each Function and edges for data flow.
 *
 * @param root The output Variable to trace from (typically loss)
 * @param params Optional named parameters to label in the graph
 * @param options Visualization options for additional annotations
 * @return DOT format string suitable for rendering with graphviz
 *
 * Usage:
 *   auto dot = make_dot(loss, {{"weight", weight_var}, {"bias", bias_var}});
 *   // Write to file and render: dot -Tpng graph.dot -o graph.png
 */
auto make_dot(const Variable& root,
              const std::unordered_map<std::string, Variable>& params = {},
              const GraphVizOptions& options = {}) -> std::string;

/**
 * @brief Save DOT output to a file.
 *
 * @param dot DOT string from make_dot()
 * @param path Output file path (e.g., "graph.dot")
 */
auto save_dot(const std::string& dot, const std::string& path) -> void;

} // namespace tenzor
