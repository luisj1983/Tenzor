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

    // Iterative DFS (explicit stack) — one recursion frame per grad_fn node
    // overflowed the C++ stack on deep chains (deep residual stacks). Mirrors
    // the engine's iterative topological_sort invariant.
    auto build_graph = [&](const std::shared_ptr<Function>& start) {
        std::vector<std::shared_ptr<Function>> stack{start};
        while (!stack.empty()) {
            auto fn = stack.back();
            stack.pop_back();
            if (!fn || visited.count(fn.get())) continue;
            visited.insert(fn.get());
            auto node = graph.add_node(fn);
            for (auto& next_fn : fn->next_functions()) {
                if (next_fn) {
                    auto next_node = graph.add_node(next_fn);
                    graph.connect(node, next_node);
                    if (!visited.count(next_fn.get())) stack.push_back(next_fn);
                }
            }
        }
    };

    if (root.grad_fn()) {
        build_graph(root.grad_fn());
    }

    reset_stats();
    optimize(graph);

    // Live-graph rewrite: actually eliminate provably-identity inverse transpose
    // pairs from the EXECUTABLE grad_fn chain (the analysis passes above only
    // detect patterns on the throwaway ComputationGraph). transpose(a,b) followed
    // by transpose(a,b) is the identity; its backward — TransposeBackward∘
    // TransposeBackward with the same dims — is likewise the identity on the
    // gradient. So splicing a consumer directly onto the pair's child preserves
    // gradients EXACTLY. We only touch pairs where both nodes are single-consumer
    // (no other path references them) and the dims match, so nothing else is
    // affected. The engine traverses next_functions() for both the value and the
    // higher-order (var) gradient, so rewriting next_functions covers both.
    {
        std::unordered_map<Function*, std::shared_ptr<Function>> reachable;
        std::unordered_map<Function*, int> consumers;
        // Iterative DFS (explicit stack) — avoids stack overflow on deep chains.
        auto walk = [&](const std::shared_ptr<Function>& start) {
            std::vector<std::shared_ptr<Function>> stack{start};
            while (!stack.empty()) {
                auto fn = stack.back();
                stack.pop_back();
                if (!fn || reachable.count(fn.get())) continue;
                reachable[fn.get()] = fn;
                for (const auto& nf : fn->next_functions()) {
                    if (nf) {
                        consumers[nf.get()]++;
                        if (!reachable.count(nf.get())) stack.push_back(nf);
                    }
                }
            }
        };
        if (root.grad_fn()) walk(root.grad_fn());

        auto is_transpose = [](const std::shared_ptr<Function>& f) {
            return f && f->op_id() == OpId::Transpose;
        };
        auto tdims = [](const std::shared_ptr<Function>& f) {
            OpAttributes a = f->saved_attributes();
            return std::pair<int64_t, int64_t>(a.get_int(AttrKey::Dim0, 0),
                                               a.get_int(AttrKey::Dim1, 0));
        };
        // A node accumulates a gradient LOCALLY (outside its next_functions) if
        // any of its input variables is a leaf-with-grad or retains its grad.
        // Eliminating such a node would silently drop that accumulation, so it
        // is NOT safe to splice it out — keep it in the chain.
        auto accumulates_locally = [](const std::shared_ptr<Function>& f) {
            for (const auto& v : f->input_variables()) {
                if (v.requires_grad() && (v.is_leaf() || v.retains_grad())) {
                    return true;
                }
                // A user backward hook registered on a spliced-out non-leaf
                // intermediate would be silently dropped by the transpose-pair
                // elimination. Suppress the splice so the hook still fires.
                if (v.has_hooks()) {
                    return true;
                }
            }
            return false;
        };

        for (auto& [ptr, P] : reachable) {
            (void)ptr;
            std::vector<std::shared_ptr<Function>> nfs = P->next_functions();
            bool changed = false;
            for (size_t i = 0; i < nfs.size(); ++i) {
                const std::shared_ptr<Function> T1 = nfs[i];
                if (!is_transpose(T1) || consumers[T1.get()] != 1) continue;
                if (T1->next_functions().size() != 1) continue;
                std::shared_ptr<Function> T2 = T1->next_functions()[0];
                if (!is_transpose(T2) || consumers[T2.get()] != 1) continue;
                if (tdims(T1) != tdims(T2)) continue;  // must be an inverse pair
                // Don't drop a leaf / retained-grad accumulation held by either
                // transpose, and require a real (non-null) downstream node to
                // reconnect to.
                if (accumulates_locally(T1) || accumulates_locally(T2)) continue;
                const auto& t2_next = T2->next_functions();
                if (t2_next.empty() || !t2_next[0]) continue;
                nfs[i] = t2_next[0];
                changed = true;
                ++stats_.transpose_pairs_eliminated;
                ++stats_.total_optimizations;
            }
            if (changed) P->set_next_functions(nfs);
        }
    }

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
        fusion_count++;
    }

    // ANALYSIS-ONLY: this pass *detects* MatMul->ReLU fusion opportunities but
    // does NOT apply them to the executable autograd graph.
    //
    // optimize_variable() builds a throwaway ComputationGraph from the live
    // grad_fn chain; replace_nodes() (below, formerly invoked here) only mutates
    // that ComputationGraph's GraphNode adjacency. The backward engine
    // (src/autograd/engine.cpp) traverses Function::next_functions() directly and
    // never references ComputationGraph/GraphNode. It does not rewrite the
    // consumer Function's next_functions to point at the fused node, nor update
    // root.grad_fn(), so a FusedLinearReLUBackward node would be unreachable from
    // the live graph and never executed by .backward(). Bumping
    // stats_.linear_relu_fused / stats_.total_optimizations here therefore
    // over-reported optimizations that never happened — the exact defect the
    // sibling passes (fuse_conv_*, fuse_batchnorm_relu, fuse_linear_gelu,
    // eliminate_transpose_pairs, collapse_reshape_chains) were corrected to
    // avoid. We mirror those passes: detect and return the count for diagnostics,
    // but do not mutate the graph and do not bump stats. (Only eliminate_dead_code
    // actually mutates the ComputationGraph and updates stats.)
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
