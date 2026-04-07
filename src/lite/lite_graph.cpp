/**
 * @file lite_graph.cpp
 * @brief LiteGraph node management implementation
 *
 * Provides storage operations for the lite-runtime execution plan.
 * Note: execute() is not implemented here — the lite runtime drives
 * execution via the kernel dispatcher in runtime.cpp.
 */

#include "tenzor/lite/lite_graph.hpp"
#include <stdexcept>

namespace tenzor::lite {

auto LiteGraph::add_node(LiteNode node) -> void {
    nodes_.push_back(std::move(node));
}

auto LiteGraph::num_nodes() const -> size_t {
    return nodes_.size();
}

}  // namespace tenzor::lite
