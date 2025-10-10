#include "tenzor/autograd/graph.hpp"
#include <algorithm>
#include <functional>
#include <unordered_set>
#include <stdexcept>

namespace tenzor {

auto ComputationGraph::add_node(std::shared_ptr<Function> func) -> std::shared_ptr<GraphNode> {
    auto node = std::make_shared<GraphNode>();
    node->function = func;
    nodes_[func.get()] = node;
    return node;
}

auto ComputationGraph::connect(std::shared_ptr<GraphNode> from,
                               std::shared_ptr<GraphNode> to) -> void {
    from->next_nodes.push_back(to);
    to->ref_count++;
    edge_count_++;
}

auto ComputationGraph::topological_sort(std::shared_ptr<GraphNode> root)
    -> std::vector<std::shared_ptr<GraphNode>> {
    std::vector<std::shared_ptr<GraphNode>> sorted;
    std::unordered_set<GraphNode*> visited;
    std::unordered_set<GraphNode*> recursion_stack;

    // DFS-based topological sort
    std::function<void(std::shared_ptr<GraphNode>)> dfs;
    dfs = [&](std::shared_ptr<GraphNode> node) {
        if (!node) return;

        // Check for cycles
        if (recursion_stack.count(node.get())) {
            throw std::runtime_error("Cycle detected in computation graph");
        }

        // Already visited
        if (visited.count(node.get())) {
            return;
        }

        visited.insert(node.get());
        recursion_stack.insert(node.get());

        // Visit all next nodes
        for (const auto& next_weak : node->next_nodes) {
            auto next = next_weak.lock();
            if (next) {
                dfs(next);
            }
        }

        recursion_stack.erase(node.get());
        sorted.push_back(node);
    };

    dfs(root);
    return sorted;
}

auto ComputationGraph::clear() -> void {
    nodes_.clear();
    edge_count_ = 0;
}

auto ComputationGraph::node_count() const -> size_t {
    return nodes_.size();
}

auto ComputationGraph::edge_count() const -> size_t {
    return edge_count_;
}

} // namespace tenzor
