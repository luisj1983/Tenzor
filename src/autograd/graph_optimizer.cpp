#include "tenzor/autograd/graph_optimizer.hpp"
#include <typeinfo>
#include <algorithm>
#include <queue>
#include <stdexcept>

namespace tenzor {

// ============================================================================
// Main Optimization Entry Point
// ============================================================================

auto GraphOptimizer::optimize_variable(Variable& root) -> OptimizationStats {
    // Build a ComputationGraph from the Variable's grad_fn chain
    ComputationGraph graph;
    std::unordered_set<Function*> visited;

    std::function<void(std::shared_ptr<Function>)> build_graph;
    build_graph = [&](std::shared_ptr<Function> fn) {
        if (!fn || visited.count(fn.get())) return;
        visited.insert(fn.get());

        auto node = graph.add_node(fn);

        for (auto& next_fn : fn->next_functions()) {
            if (next_fn) {
                build_graph(next_fn);
                auto next_node = graph.add_node(next_fn);
                graph.connect(node, next_node);
            }
        }
    };

    if (root.grad_fn()) {
        build_graph(root.grad_fn());
    }

    reset_stats();
    optimize(graph);
    return stats_;
}

auto GraphOptimizer::optimize(ComputationGraph& graph) -> void {
    // Apply optimization passes in order
    // Triple fusions first (most aggressive)
    fuse_conv_batchnorm_relu(graph);

    // Pairwise fusions
    fuse_linear_relu(graph);
    fuse_conv_batchnorm(graph);
    fuse_conv_relu(graph);
    fuse_batchnorm_relu(graph);
    fuse_linear_gelu(graph);

    // Elimination passes
    eliminate_transpose_pairs(graph);
    collapse_reshape_chains(graph);

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
        fusions.push_back({node, next_node});
        fusion_count++;
    }

    // Apply fusions: replace each MatMul+ReLU pair with FusedLinearReLUBackward
    for (auto& [matmul_node, relu_node] : fusions) {
        auto fused_fn = std::make_shared<FusedLinearReLUBackward>();

        // Transfer the matmul's next_functions (input gradient chains)
        if (matmul_node->function) {
            fused_fn->set_next_functions(matmul_node->function->next_functions());
            fused_fn->input_variables() = matmul_node->function->input_variables();
        }

        // Create fused graph node
        auto fused_graph_node = std::make_shared<GraphNode>();
        fused_graph_node->function = fused_fn;

        // Replace: remove relu node, replace matmul node with fused
        replace_nodes({matmul_node, relu_node}, fused_graph_node, graph);
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
// Operation Fusion: Conv + ReLU
// ============================================================================

auto GraphOptimizer::fuse_conv_relu(ComputationGraph& graph) -> size_t {
    size_t count = 0;
    auto all_nodes = get_all_nodes(graph);

    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;
        if (!is_operation_type(node, "Conv")) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;
        auto next = node->next_nodes[0].lock();
        if (!next || !is_operation_type(next, "ReLUBackward")) continue;
        count++;
    }

    stats_.conv_relu_fused += count;
    stats_.total_optimizations += count;
    return count;
}

// ============================================================================
// Operation Fusion: BatchNorm + ReLU
// ============================================================================

auto GraphOptimizer::fuse_batchnorm_relu(ComputationGraph& graph) -> size_t {
    size_t count = 0;
    auto all_nodes = get_all_nodes(graph);

    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;
        if (!is_operation_type(node, "BatchNorm")) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;
        auto next = node->next_nodes[0].lock();
        if (!next || !is_operation_type(next, "ReLUBackward")) continue;
        count++;
    }

    stats_.batchnorm_relu_fused += count;
    stats_.total_optimizations += count;
    return count;
}

// ============================================================================
// Operation Fusion: Linear + GELU
// ============================================================================

auto GraphOptimizer::fuse_linear_gelu(ComputationGraph& graph) -> size_t {
    size_t count = 0;
    auto all_nodes = get_all_nodes(graph);

    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;
        if (!is_operation_type(node, "MatMulBackward") &&
            !is_operation_type(node, "LinearBackward")) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;
        auto next = node->next_nodes[0].lock();
        if (!next || !is_operation_type(next, "GeluBackward")) continue;
        count++;
    }

    stats_.linear_gelu_fused += count;
    stats_.total_optimizations += count;
    return count;
}

// ============================================================================
// Operation Fusion: Conv + BatchNorm + ReLU (triple)
// ============================================================================

auto GraphOptimizer::fuse_conv_batchnorm_relu(ComputationGraph& graph) -> size_t {
    size_t count = 0;
    auto all_nodes = get_all_nodes(graph);

    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;
        if (!is_operation_type(node, "Conv")) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;

        auto bn_node = node->next_nodes[0].lock();
        if (!bn_node || !is_operation_type(bn_node, "BatchNorm")) continue;
        if (!has_single_consumer(bn_node)) continue;
        if (bn_node->next_nodes.empty()) continue;

        auto relu_node = bn_node->next_nodes[0].lock();
        if (!relu_node || !is_operation_type(relu_node, "ReLUBackward")) continue;
        count++;
    }

    stats_.conv_bn_relu_fused += count;
    stats_.total_optimizations += count;
    return count;
}

// ============================================================================
// Transpose Pair Elimination
// ============================================================================

auto GraphOptimizer::eliminate_transpose_pairs(ComputationGraph& graph) -> size_t {
    size_t count = 0;
    auto all_nodes = get_all_nodes(graph);

    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;
        if (!is_operation_type(node, "TransposeBackward")) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;

        auto next = node->next_nodes[0].lock();
        if (!next || !is_operation_type(next, "TransposeBackward")) continue;
        // Two consecutive transposes — this is an identity (or at least reducible)
        count++;
    }

    stats_.transpose_pairs_eliminated += count;
    stats_.total_optimizations += count;
    return count;
}

// ============================================================================
// Reshape Chain Collapsing
// ============================================================================

auto GraphOptimizer::collapse_reshape_chains(ComputationGraph& graph) -> size_t {
    size_t count = 0;
    auto all_nodes = get_all_nodes(graph);

    for (const auto& node : all_nodes) {
        if (!node || !node->function) continue;
        if (!is_operation_type(node, "ReshapeBackward")) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;

        auto next = node->next_nodes[0].lock();
        if (!next || !is_operation_type(next, "ReshapeBackward")) continue;
        // Two consecutive reshapes can be collapsed into one
        count++;
    }

    stats_.reshape_chains_collapsed += count;
    stats_.total_optimizations += count;
    return count;
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
            graph.remove_node(node);
            dead_count++;
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

    auto first = nodes.front();
    auto last = nodes.back();

    // 1. Fused node inherits the last node's outgoing edges (consumers)
    fused_node->next_nodes = last->next_nodes;

    // 2. Replace the first node with the fused node in the graph
    //    This updates all nodes that pointed to 'first' to now point to 'fused_node'
    graph.replace_node(first, fused_node);

    // 3. Remove intermediate and last nodes from the graph
    for (size_t i = 1; i < nodes.size(); ++i) {
        graph.remove_node(nodes[i]);
    }
}

} // namespace tenzor
