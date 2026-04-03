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

auto ComputationGraph::remove_node(std::shared_ptr<GraphNode> node) -> bool {
    if (!node || !node->function) return false;

    auto it = nodes_.find(node->function.get());
    if (it == nodes_.end()) return false;

    // Count edges being removed
    edge_count_ -= node->next_nodes.size();

    // Remove from map
    nodes_.erase(it);
    return true;
}

auto ComputationGraph::replace_node(std::shared_ptr<GraphNode> old_node,
                                    std::shared_ptr<GraphNode> new_node) -> bool {
    if (!old_node || !old_node->function || !new_node) return false;

    auto it = nodes_.find(old_node->function.get());
    if (it == nodes_.end()) return false;

    // Transfer outgoing edges from old to new
    new_node->next_nodes = std::move(old_node->next_nodes);
    new_node->ref_count = old_node->ref_count;

    // Remove old entry, add new
    nodes_.erase(it);
    if (new_node->function) {
        nodes_[new_node->function.get()] = new_node;
    }

    // Update all nodes that reference old_node in their next_nodes
    // Since next_nodes uses weak_ptr, stale refs will simply expire.
    // But we need to update nodes that point to old_node to point to new_node.
    for (auto& [_, n] : nodes_) {
        for (auto& next_weak : n->next_nodes) {
            auto next = next_weak.lock();
            if (next.get() == old_node.get()) {
                next_weak = new_node;
            }
        }
    }

    return true;
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
