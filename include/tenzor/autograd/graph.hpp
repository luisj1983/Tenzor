#pragma once

#include <memory>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include "function.hpp"

namespace tenzor {

// Computational graph node
struct GraphNode {
    std::shared_ptr<Function> function;
    std::vector<std::weak_ptr<GraphNode>> next_nodes;
    int ref_count{0};
};

// Computational graph
class ComputationGraph {
public:
    ComputationGraph() = default;

    // Add node to graph
    auto add_node(std::shared_ptr<Function> func) -> std::shared_ptr<GraphNode>;

    // Connect nodes
    auto connect(std::shared_ptr<GraphNode> from,
                std::shared_ptr<GraphNode> to) -> void;

    // Topological sort for backward pass
    auto topological_sort(std::shared_ptr<GraphNode> root)
        -> std::vector<std::shared_ptr<GraphNode>>;

    // Clear graph
    auto clear() -> void;

    // Graph statistics
    auto node_count() const -> size_t;
    auto edge_count() const -> size_t;

private:
    std::unordered_map<Function*, std::shared_ptr<GraphNode>> nodes_;
    size_t edge_count_{0};
};

} // namespace tenzor
