#pragma once

#include "tenzor/autograd/variable.hpp"
#include <string>
#include <unordered_map>

namespace tenzor {

/**
 * @brief Generate a Graphviz DOT representation of an autograd computation graph.
 *
 * Traverses the computation graph from a Variable's grad_fn backward through
 * next_functions, emitting nodes for each Function and edges for data flow.
 *
 * @param root The output Variable to trace from (typically loss)
 * @param params Optional named parameters to label in the graph
 * @return DOT format string suitable for rendering with graphviz
 *
 * Usage:
 *   auto dot = make_dot(loss, {{"weight", weight_var}, {"bias", bias_var}});
 *   // Write to file and render: dot -Tpng graph.dot -o graph.png
 */
auto make_dot(const Variable& root,
              const std::unordered_map<std::string, Variable>& params = {}) -> std::string;

} // namespace tenzor
