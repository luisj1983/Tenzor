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
    std::unordered_set<GraphNode*> on_stack;  // nodes on the current DFS path (cycle detection)

    if (!root) return sorted;

    // Iterative post-order DFS with an explicit stack — avoids one recursion
    // frame per graph node, which overflowed the C++ stack on deep grad_fn
    // chains (deep residual stacks). Mirrors BackwardEngine::topological_sort's
    // iterative invariant. Each frame carries an `entered` flag: false = first
    // visit (push children), true = children done (emit node, leave the path).
    std::vector<std::pair<std::shared_ptr<GraphNode>, bool>> stack;
    stack.emplace_back(root, false);

    while (!stack.empty()) {
        auto node = stack.back().first;
        const bool entered = stack.back().second;
        if (!node) { stack.pop_back(); continue; }

        if (entered) {
            on_stack.erase(node.get());
            sorted.push_back(node);
            stack.pop_back();
            continue;
        }

        if (visited.count(node.get())) { stack.pop_back(); continue; }
        visited.insert(node.get());
        on_stack.insert(node.get());
        stack.back().second = true;  // emit this node after its descendants

        for (const auto& next_weak : node->next_nodes) {
            auto next = next_weak.lock();
            if (!next) continue;
            if (on_stack.count(next.get())) {
                throw std::runtime_error("Cycle detected in computation graph");
            }
            if (!visited.count(next.get())) {
                stack.emplace_back(next, false);
            }
        }
    }
    return sorted;
}

auto ComputationGraph::remove_node(std::shared_ptr<GraphNode> node) -> bool {
    if (!node || !node->function) return false;

    auto it = nodes_.find(node->function.get());
    if (it == nodes_.end()) return false;

    // Outgoing edges: each edge bumped its target's ref_count at add_edge time,
    // so drop that reference and remove the edge from the running total. Leaving
    // the targets' ref_counts stale (the previous behaviour) made them
    // un-collectable and skewed every later edge accounting.
    for (const auto& next_weak : node->next_nodes) {
        if (auto next = next_weak.lock()) {
            if (next->ref_count > 0) next->ref_count--;
            // Only drop this edge from the running total when the target is
            // still alive. If the target was already removed, its own
            // remove_node() already subtracted this in-edge via its ref_count
            // (the incoming-edge loop below), so decrementing again here would
            // double-count and corrupt edge_count_.
            if (edge_count_ > 0) edge_count_--;
        }
    }

    // Incoming edges: other nodes still list this node in their next_nodes
    // (now-dangling weak_ptrs that simply expire). Each such edge bumped THIS
    // node's ref_count at add_edge time, so subtract that many edges too —
    // otherwise edge_count_ permanently over-counts the removed in-edges.
    for (int k = 0; k < node->ref_count && edge_count_ > 0; ++k) {
        edge_count_--;
    }

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
