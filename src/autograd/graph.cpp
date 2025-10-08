#include "tenzor/autograd/graph.hpp"
#include <algorithm>

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
    // TODO: Implement proper topological sort
    return {root};
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
