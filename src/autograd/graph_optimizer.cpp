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
        // Audit A.2: OpId-based pattern match (stronger than RTTI-substring).
        if (!is_operation_type(node, OpId::MatMul)) continue;

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
        // Audit fix (LOW): FusedLinearReLUBackward::backward()/backward_with_variables()
        // unconditionally dereference relu_output_ and saved_tensors_[0/1] (input,
        // weight). Previously the fused node was created without these, leaving
        // relu_output_ a null Tensor and no saved tensors — executing it would fault.
        // Populate the mask source (ReLU's saved output) and the linear operands
        // (MatMul's saved input/weight) at fusion time so the node is self-consistent.
        if (!matmul_node->function || !relu_node->function) continue;

        // ReLUBackward saves the pre-ReLU input as its sole saved tensor. The
        // ReLU gate mask (z > 0) is identical whether evaluated on the input or
        // the output (output > 0 iff input > 0), so this serves as the mask
        // source consumed by FusedLinearReLUBackward as relu_output_.
        const auto& relu_saved = relu_node->function->saved_tensors();
        if (relu_saved.empty()) continue;

        // MatMulBackward saves input (x) and weight (W) as saved_tensors_[0/1].
        const auto& matmul_saved = matmul_node->function->saved_tensors();
        if (matmul_saved.size() < 2) continue;

        auto fused_fn = std::make_shared<FusedLinearReLUBackward>();

        // Mask source for the ReLU gate.
        fused_fn->set_relu_output(relu_saved[0]);

        // Linear operands consumed by backward as saved_tensors_[0]=x, [1]=W.
        fused_fn->save_for_backward({matmul_saved[0], matmul_saved[1]});

        // Transfer the matmul's next_functions (input gradient chains)
        fused_fn->set_next_functions(matmul_node->function->next_functions());
        fused_fn->input_variables() = matmul_node->function->input_variables();

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

    // ANALYSIS-ONLY: this pass only *detects* Conv->BatchNorm patterns; it does
    // not yet rewrite the graph (no replace_nodes/remove_node). Returning the
    // detected count is useful for diagnostics, but we must NOT bump
    // stats_.conv_batchnorm_fused / stats_.total_optimizations, since doing so
    // would make OptimizationStats over-report optimizations that never
    // happened. (Only fuse_linear_relu and eliminate_dead_code actually mutate
    // the graph and update stats.)
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

    // ANALYSIS-ONLY: detects Conv->ReLU but does not rewrite the graph; do not
    // pollute stats with optimizations that never happened.
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

    // ANALYSIS-ONLY: detects BatchNorm->ReLU but does not rewrite the graph; do
    // not pollute stats with optimizations that never happened.
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
        // Audit A.2: OpId-based pattern match.
        if (!is_operation_type(node, OpId::MatMul) &&
            !is_operation_type(node, OpId::Linear)) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;
        auto next = node->next_nodes[0].lock();
        // Audit A.2: OpId-based pattern match.
        if (!next || !is_operation_type(next, OpId::Gelu)) continue;
        count++;
    }

    // ANALYSIS-ONLY: detects Linear->GELU but does not rewrite the graph; do
    // not pollute stats with optimizations that never happened.
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

    // ANALYSIS-ONLY: detects Conv->BatchNorm->ReLU but does not rewrite the
    // graph; do not pollute stats with optimizations that never happened.
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
        // Audit A.2: OpId-based pattern match.
        if (!is_operation_type(node, OpId::Transpose)) continue;
        if (!has_single_consumer(node)) continue;
        if (node->next_nodes.empty()) continue;

        auto next = node->next_nodes[0].lock();
        // Audit A.2: OpId-based pattern match.
        if (!next || !is_operation_type(next, OpId::Transpose)) continue;
        // Two consecutive transposes — this is an identity (or at least reducible)
        count++;
    }

    // ANALYSIS-ONLY: detects transpose pairs but does not rewrite the graph; do
    // not pollute stats with optimizations that never happened.
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

    // ANALYSIS-ONLY: detects reshape chains but does not rewrite the graph; do
    // not pollute stats with optimizations that never happened.
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
                                   [[maybe_unused]] const ComputationGraph& graph) const
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

auto GraphOptimizer::is_operation_type(std::shared_ptr<GraphNode> node,
                                       OpId op_id) const -> bool {
    // Audit A.2: OpId-based pattern matching. Strictly stronger than the
    // RTTI-substring overload: OpId is a compile-time enum value, so a
    // miss-by-renaming or compiler-dependent type-mangling change can't
    // silently break the matcher. Subclasses that haven't opted in to
    // `op_id()` return OpId::Unknown, which is treated by every matcher
    // as "do not match" — so adding new Functions or renaming existing
    // ones is safe (it never silently mis-matches).
    if (!node || !node->function) return false;
    if (op_id == OpId::Unknown) {
        // Refuse to match the sentinel "no opted-in OpId" value so that
        // callers passing the default never accidentally collect every
        // un-opted-in Function in the graph. Caller bug, surface it.
        return false;
    }
    return node->function->op_id() == op_id;
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

    // Snapshot the last node's outgoing edges (the real downstream consumers)
    // BEFORE any mutation, since remove_node below would clear them.
    auto consumer_edges = last->next_nodes;

    // 1. Replace the first node with the fused node in the graph.
    //    This updates all nodes that pointed to 'first' to now point to
    //    'fused_node'. NOTE: replace_node move-assigns first->next_nodes onto
    //    fused_node->next_nodes (graph.cpp), so we MUST set the real consumer
    //    edges AFTER this call, otherwise they get clobbered with the first
    //    node's (intermediate) edges and the graph is severed downstream.
    graph.replace_node(first, fused_node);

    // 2. Fused node inherits the last node's outgoing edges (consumers).
    fused_node->next_nodes = consumer_edges;

    // 3. Remove intermediate and last nodes from the graph
    for (size_t i = 1; i < nodes.size(); ++i) {
        graph.remove_node(nodes[i]);
    }
}

} // namespace tenzor
