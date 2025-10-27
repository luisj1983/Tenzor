#include "tenzor/autograd/graph_optimizer.hpp"
#include <typeinfo>
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace tenzor {

// ============================================================================
// Main Optimization Entry Point
// ============================================================================

auto GraphOptimizer::optimize(ComputationGraph& graph) -> void {
    // Apply optimization passes in order
    // Fusion passes first (may create dead code)
    fuse_linear_relu(graph);
    fuse_conv_batchnorm(graph);

    // Dead code elimination last (cleanup)
    eliminate_dead_code(graph);
}

// ============================================================================
// Operation Fusion: Linear + ReLU
// ============================================================================

auto GraphOptimizer::fuse_linear_relu(ComputationGraph& graph) -> size_t {
    size_t fusion_count = 0;

    // Get all nodes in the graph
    auto all_nodes = get_all_nodes(graph);

    // Pattern: MatMul -> ReLU
    OpPattern pattern({"MatMulBackward", "ReLUBackward"}, "FusedLinearReLU");

    // Track nodes to remove (can't modify during iteration)
    std::vector<std::shared_ptr<GraphNode>> nodes_to_remove;
    std::vector<std::pair<std::shared_ptr<GraphNode>, std::shared_ptr<GraphNode>>> fusions;

    // Find all MatMul nodes that are followed by ReLU
    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;

        // Check if this is a MatMul operation
        if (!is_operation_type(node, "MatMulBackward")) continue;

        // Check if it has exactly one consumer (ReLU)
        if (!has_single_consumer(node)) continue;

        // Get the single consumer
        if (node->next_nodes.empty()) continue;
        auto next_node = node->next_nodes[0].lock();
        if (!next_node) continue;

        // Check if consumer is ReLU
        if (!is_operation_type(next_node, "ReLUBackward")) continue;

        // Found a fusion opportunity: MatMul -> ReLU
        // Note: In a real implementation, we would create a FusedLinearReLUBackward
        // For now, we just track the fusion for statistics
        fusions.push_back({node, next_node});
        fusion_count++;
    }

    // Update statistics
    stats_.linear_relu_fused += fusion_count;
    stats_.total_optimizations += fusion_count;

    return fusion_count;
}

// ============================================================================
// Operation Fusion: Convolution + BatchNorm
// ============================================================================

auto GraphOptimizer::fuse_conv_batchnorm(ComputationGraph& graph) -> size_t {
    size_t fusion_count = 0;

    // Get all nodes in the graph
    auto all_nodes = get_all_nodes(graph);

    // Pattern: Conv -> BatchNorm (we look for patterns in function names)
    // Track nodes for fusion
    std::vector<std::pair<std::shared_ptr<GraphNode>, std::shared_ptr<GraphNode>>> fusions;

    // Find all Conv nodes that are followed by BatchNorm
    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;

        // Check for Conv operation (Conv2d, Conv1d, etc.)
        // We check the type name for "Conv" substring
        const char* node_type = typeid(*node->function).name();
        std::string type_name(node_type);

        // Skip if not a convolution
        if (type_name.find("Conv") == std::string::npos) continue;

        // Check if it has exactly one consumer
        if (!has_single_consumer(node)) continue;

        // Get the single consumer
        if (node->next_nodes.empty()) continue;
        auto next_node = node->next_nodes[0].lock();
        if (!next_node) continue;

        // Check if consumer is BatchNorm
        const char* next_type = typeid(*next_node->function).name();
        std::string next_type_name(next_type);

        if (next_type_name.find("BatchNorm") == std::string::npos) continue;

        // Found a fusion opportunity: Conv -> BatchNorm
        fusions.push_back({node, next_node});
        fusion_count++;
    }

    // Update statistics
    stats_.conv_batchnorm_fused += fusion_count;
    stats_.total_optimizations += fusion_count;

    return fusion_count;
}

// ============================================================================
// Dead Code Elimination
// ============================================================================

auto GraphOptimizer::eliminate_dead_code(ComputationGraph& graph) -> size_t {
    // Compute reachable nodes via backward traversal from outputs
    auto reachable = compute_reachable_nodes(graph);

    // Get all nodes
    auto all_nodes = get_all_nodes(graph);

    // Count dead nodes (not in reachable set)
    size_t dead_count = 0;
    for (const auto& node : all_nodes) {
        if (!node) continue;

        if (reachable.find(node.get()) == reachable.end()) {
            dead_count++;
            // Note: In a full implementation, we would actually remove the node
            // from the graph. For now, we just count it.
        }
    }

    // Update statistics
    stats_.dead_nodes_removed += dead_count;
    stats_.total_optimizations += dead_count;

    return dead_count;
}

// ============================================================================
// Pattern Matching Utilities
// ============================================================================

auto GraphOptimizer::match_pattern(std::shared_ptr<GraphNode> start,
                                   const OpPattern& pattern,
                                   const ComputationGraph& graph) const
    -> std::vector<std::shared_ptr<GraphNode>> {

    std::vector<std::shared_ptr<GraphNode>> matched;
    auto current = start;

    // Try to match each operation in the pattern sequence
    for (const auto& op_type : pattern.op_sequence) {
        if (!current || !current->function) {
            return {};  // Pattern match failed
        }

        // Check if current node matches expected operation type
        if (!is_operation_type(current, op_type)) {
            return {};  // Pattern match failed
        }

        matched.push_back(current);

        // Move to next node (if not at end of pattern)
        if (matched.size() < pattern.op_sequence.size()) {
            // Current node should have exactly one consumer for pattern to match
            if (current->next_nodes.size() != 1) {
                return {};  // Pattern match failed
            }

            current = current->next_nodes[0].lock();
        }
    }

    return matched;  // Successfully matched pattern
}

auto GraphOptimizer::is_operation_type(std::shared_ptr<GraphNode> node,
                                       const std::string& op_type) const -> bool {
    if (!node || !node->function) return false;

    // Use RTTI to get the type name
    const char* type_name = typeid(*node->function).name();
    std::string actual_type(type_name);

    // Check if the type name contains the expected operation type
    // Note: RTTI names are compiler-dependent, so this checks for substring match
    return actual_type.find(op_type) != std::string::npos;
}

auto GraphOptimizer::has_single_consumer(std::shared_ptr<GraphNode> node) const -> bool {
    if (!node) return false;

    // A node has a single consumer if it has exactly one outgoing edge
    // and that edge points to a valid node
    if (node->next_nodes.size() != 1) return false;

    auto consumer = node->next_nodes[0].lock();
    return consumer != nullptr;
}

// ============================================================================
// Graph Analysis Utilities
// ============================================================================

auto GraphOptimizer::compute_reachable_nodes(const ComputationGraph& graph) const
    -> std::unordered_set<GraphNode*> {

    std::unordered_set<GraphNode*> reachable;
    std::queue<GraphNode*> worklist;

    // Get all nodes
    auto all_nodes = get_all_nodes(graph);

    // Find output nodes (nodes that have consumers or are terminal)
    // We consider nodes with outgoing edges as potential outputs
    for (const auto& node : all_nodes) {
        if (!node) continue;

        // If node has consumers, add to worklist
        if (!node->next_nodes.empty()) {
            for (const auto& next_weak : node->next_nodes) {
                auto next = next_weak.lock();
                if (next && reachable.find(next.get()) == reachable.end()) {
                    reachable.insert(next.get());
                    worklist.push(next.get());
                }
            }
        }
    }

    // If no nodes with consumers found, consider all nodes as potentially reachable
    if (reachable.empty()) {
        for (const auto& node : all_nodes) {
            if (node) {
                reachable.insert(node.get());
            }
        }
        return reachable;
    }

    // Backward traversal from output nodes
    // We need to traverse backward, but GraphNode only has forward edges (next_nodes)
    // So we first build a reverse graph
    std::unordered_map<GraphNode*, std::vector<GraphNode*>> reverse_edges;

    for (const auto& node : all_nodes) {
        if (!node) continue;

        for (const auto& next_weak : node->next_nodes) {
            auto next = next_weak.lock();
            if (next) {
                reverse_edges[next.get()].push_back(node.get());
            }
        }
    }

    // BFS backward from output nodes using reverse edges
    std::unordered_set<GraphNode*> visited;
    while (!worklist.empty()) {
        auto* current = worklist.front();
        worklist.pop();

        if (visited.find(current) != visited.end()) {
            continue;
        }
        visited.insert(current);

        // Add predecessors to worklist
        auto it = reverse_edges.find(current);
        if (it != reverse_edges.end()) {
            for (auto* pred : it->second) {
                if (reachable.find(pred) == reachable.end()) {
                    reachable.insert(pred);
                    worklist.push(pred);
                }
            }
        }
    }

    return reachable;
}

auto GraphOptimizer::get_all_nodes(const ComputationGraph& graph) const
    -> std::vector<std::shared_ptr<GraphNode>> {

    std::vector<std::shared_ptr<GraphNode>> nodes;
    nodes.reserve(graph.nodes_.size());

    // Access the private nodes_ map via friend declaration
    // Extract all shared_ptr<GraphNode> values from the map
    for (const auto& [func_ptr, node_ptr] : graph.nodes_) {
        if (node_ptr) {
            nodes.push_back(node_ptr);
        }
    }

    return nodes;
}

auto GraphOptimizer::replace_nodes(const std::vector<std::shared_ptr<GraphNode>>& nodes,
                                   std::shared_ptr<GraphNode> fused_node,
                                   ComputationGraph& graph) -> void {
    if (nodes.empty() || !fused_node) return;

    // Get first and last nodes in the sequence
    auto first = nodes.front();
    auto last = nodes.back();

    // Update fused node's connections
    // Input edges: connect fused node to first node's predecessors
    // Output edges: connect fused node to last node's successors

    // Connect fused node outputs to last node's consumers
    fused_node->next_nodes = last->next_nodes;

    // Update reference counts for successors
    for (const auto& next_weak : fused_node->next_nodes) {
        auto next = next_weak.lock();
        if (next) {
            next->ref_count++;
        }
    }

    // In a full implementation, we would:
    // 1. Find all nodes that point to 'first' and redirect them to 'fused_node'
    // 2. Update the graph's node map
    // 3. Decrement reference counts for removed nodes
    // 4. Remove the old nodes from the graph

    // Note: This requires more extensive graph manipulation capabilities
    // than the current ComputationGraph API provides
}

} // namespace tenzor
