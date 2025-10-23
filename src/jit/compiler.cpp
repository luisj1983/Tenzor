/**
 * @file compiler.cpp
 * @brief Implementation of graph optimization passes
 */

#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <cmath>

namespace tenzor {
namespace jit {

// ============================================================================
// Dead Code Elimination Pass
// ============================================================================

auto DeadCodeEliminationPass::run(Graph& graph) -> bool {
    auto reachable = mark_reachable_nodes(graph);

    bool modified = false;
    auto& nodes = const_cast<std::vector<std::shared_ptr<Node>>&>(graph.nodes());

    // Remove unreachable nodes
    auto it = nodes.begin();
    while (it != nodes.end()) {
        if (reachable.find(it->get()) == reachable.end()) {
            it = nodes.erase(it);
            modified = true;
        } else {
            ++it;
        }
    }

    return modified;
}

auto DeadCodeEliminationPass::mark_reachable_nodes(const Graph& graph) -> std::unordered_set<Node*> {
    std::unordered_set<Node*> reachable;
    std::vector<Node*> worklist;

    // Start from output nodes
    for (const auto& output : graph.outputs()) {
        auto producer = output->node();
        if (producer) {
            worklist.push_back(producer.get());
        }
    }

    // Backward traversal
    while (!worklist.empty()) {
        auto node = worklist.back();
        worklist.pop_back();

        if (reachable.insert(node).second) {
            // Visit inputs
            for (const auto& input : node->inputs()) {
                auto producer = input->node();
                if (producer && reachable.find(producer.get()) == reachable.end()) {
                    worklist.push_back(producer.get());
                }
            }
        }
    }

    return reachable;
}

// ============================================================================
// Common Subexpression Elimination Pass
// ============================================================================

auto CommonSubexpressionEliminationPass::run(Graph& graph) -> bool {
    bool modified = false;
    std::unordered_map<size_t, std::vector<std::shared_ptr<Node>>> hash_map;

    // Build hash map of nodes
    for (const auto& node : graph.nodes()) {
        size_t hash = compute_node_hash(*node);
        hash_map[hash].push_back(node);
    }

    // Find and merge equivalent nodes
    std::unordered_map<std::string, std::string> value_replacements;

    for (const auto& [hash, nodes] : hash_map) {
        if (nodes.size() < 2) continue;

        for (size_t i = 0; i < nodes.size(); ++i) {
            for (size_t j = i + 1; j < nodes.size(); ++j) {
                if (nodes_equivalent(*nodes[i], *nodes[j])) {
                    // Mark outputs of nodes[j] to be replaced by outputs of nodes[i]
                    auto& outputs_i = nodes[i]->outputs();
                    auto& outputs_j = nodes[j]->outputs();

                    for (size_t k = 0; k < std::min(outputs_i.size(), outputs_j.size()); ++k) {
                        value_replacements[outputs_j[k]->id()] = outputs_i[k]->id();
                    }

                    modified = true;
                }
            }
        }
    }

    // Apply replacements (would need graph rewriting support)
    // This is a simplified version - full implementation would update all uses

    return modified;
}

auto CommonSubexpressionEliminationPass::compute_node_hash(const Node& node) -> size_t {
    size_t hash = std::hash<int>{}(static_cast<int>(node.op_type()));

    // Hash inputs
    for (const auto& input : node.inputs()) {
        hash ^= std::hash<std::string>{}(input->id()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    return hash;
}

auto CommonSubexpressionEliminationPass::nodes_equivalent(const Node& a, const Node& b) -> bool {
    // Same operation type
    if (a.op_type() != b.op_type()) return false;

    // Same number of inputs
    if (a.inputs().size() != b.inputs().size()) return false;

    // Same inputs
    for (size_t i = 0; i < a.inputs().size(); ++i) {
        if (a.inputs()[i]->id() != b.inputs()[i]->id()) return false;
    }

    // Note: Should also check attributes, but simplified for now

    return true;
}

// ============================================================================
// Constant Folding Pass
// ============================================================================

auto ConstantFoldingPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (const auto& node : graph.nodes()) {
        if (can_fold(*node)) {
            try {
                Tensor result = evaluate_constant(*node);

                // Create constant node
                auto const_node = graph.create_node(OpType::Constant);
                const_node->set_tensor_attr("value", result);

                // Replace node with constant (simplified - would need graph rewriting)
                modified = true;
            } catch (...) {
                // Failed to fold - continue
            }
        }
    }

    return modified;
}

auto ConstantFoldingPass::can_fold(const Node& node) -> bool {
    // Check if all inputs are constants
    for (const auto& input : node.inputs()) {
        auto producer = input->node();
        if (!producer || producer->op_type() != OpType::Constant) {
            return false;
        }
    }

    // Only fold simple arithmetic operations
    switch (node.op_type()) {
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
            return true;
        default:
            return false;
    }
}

auto ConstantFoldingPass::evaluate_constant(const Node& node) -> Tensor {
    // Get input tensors
    std::vector<Tensor> inputs;
    for (const auto& input : node.inputs()) {
        auto producer = input->node();
        if (producer) {
            inputs.push_back(producer->get_tensor_attr("value"));
        }
    }

    // Evaluate operation
    switch (node.op_type()) {
        case OpType::Add:
            return inputs[0] + inputs[1];
        case OpType::Sub:
            return inputs[0] - inputs[1];
        case OpType::Mul:
            return inputs[0] * inputs[1];
        case OpType::Div:
            return inputs[0] / inputs[1];
        case OpType::Exp:
            return tenzor::exp(inputs[0]);
        case OpType::Log:
            return tenzor::log(inputs[0]);
        case OpType::Sqrt:
            return tenzor::sqrt(inputs[0]);
        default:
            throw std::runtime_error("Unsupported operation for constant folding");
    }
}

// ============================================================================
// Conv-BatchNorm Fusion Pass
// ============================================================================

auto FuseConvBatchNormPass::run(Graph& graph) -> bool {
    bool modified = false;

    // Find Conv2d -> BatchNorm2d patterns
    for (size_t i = 0; i + 1 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];

        if (node1->op_type() == OpType::Conv2d && node2->op_type() == OpType::BatchNorm2d) {
            // Check if conv output is only used by batchnorm
            bool can_fuse = true;
            if (!node1->outputs().empty()) {
                auto conv_out = node1->outputs()[0];
                if (conv_out->uses().size() > 1) {
                    can_fuse = false;
                }
            }

            if (can_fuse && fuse_pair(node1, node2, graph)) {
                modified = true;
            }
        }
    }

    return modified;
}

auto FuseConvBatchNormPass::fuse_pair(std::shared_ptr<Node> conv_node,
                                       std::shared_ptr<Node> bn_node,
                                       Graph& graph) -> bool {
    // Get BatchNorm parameters
    Tensor gamma = bn_node->get_tensor_attr("weight");
    Tensor beta = bn_node->get_tensor_attr("bias");
    Tensor running_mean = bn_node->get_tensor_attr("running_mean");
    Tensor running_var = bn_node->get_tensor_attr("running_var");
    float eps = bn_node->get_attr("eps");

    // Get Conv parameters
    Tensor conv_weight = conv_node->get_tensor_attr("weight");
    Tensor conv_bias = conv_node->get_tensor_attr("bias");

    // Compute fused parameters
    // w_fused = gamma * w_conv / sqrt(var + eps)
    // b_fused = gamma * (b_conv - mean) / sqrt(var + eps) + beta

    // This is a simplified version - actual implementation would need:
    // 1. Proper broadcasting for gamma/beta
    // 2. Channel-wise operations
    // 3. Handling of optional bias

    // For now, just mark as fused (real implementation would rewrite the graph)
    conv_node->set_bool_attr("fused_bn", true);

    return true;
}

// ============================================================================
// Conv-ReLU Fusion Pass
// ============================================================================

auto FuseConvReluPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (size_t i = 0; i + 1 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];

        if (node1->op_type() == OpType::Conv2d && node2->op_type() == OpType::ReLU) {
            if (fuse_pair(node1, node2, graph)) {
                modified = true;
            }
        }
    }

    return modified;
}

auto FuseConvReluPass::fuse_pair(std::shared_ptr<Node> conv_node,
                                  std::shared_ptr<Node> relu_node,
                                  Graph& graph) -> bool {
    // Mark conv as having fused ReLU
    conv_node->set_bool_attr("fused_relu", true);

    // Would need to remove ReLU node and rewire connections
    // Simplified for now
    return true;
}

// ============================================================================
// Linear-ReLU Fusion Pass
// ============================================================================

auto FuseLinearReluPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (size_t i = 0; i + 1 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];

        if (node1->op_type() == OpType::Linear && node2->op_type() == OpType::ReLU) {
            if (fuse_pair(node1, node2, graph)) {
                modified = true;
            }
        }
    }

    return modified;
}

auto FuseLinearReluPass::fuse_pair(std::shared_ptr<Node> linear_node,
                                    std::shared_ptr<Node> relu_node,
                                    Graph& graph) -> bool {
    linear_node->set_bool_attr("fused_relu", true);
    return true;
}

// ============================================================================
// Algebraic Simplification Pass
// ============================================================================

auto AlgebraicSimplificationPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (const auto& node : graph.nodes()) {
        if (simplify_binary_op(node, graph)) {
            modified = true;
        }
        if (simplify_unary_op(node, graph)) {
            modified = true;
        }
    }

    return modified;
}

auto AlgebraicSimplificationPass::simplify_binary_op(std::shared_ptr<Node> node, Graph& graph) -> bool {
    if (node->inputs().size() < 2) return false;

    // Check for constant inputs
    auto input0 = node->inputs()[0];
    auto input1 = node->inputs()[1];

    auto producer0 = input0->node();
    auto producer1 = input1->node();

    bool is_const0 = producer0 && producer0->op_type() == OpType::Constant;
    bool is_const1 = producer1 && producer1->op_type() == OpType::Constant;

    if (!is_const0 && !is_const1) return false;

    // Simplification rules
    switch (node->op_type()) {
        case OpType::Add:
            if (is_const1) {
                Tensor val = producer1->get_tensor_attr("value");
                // Check if adding zero (simplified check)
                // Would replace node with identity
                return true;
            }
            break;

        case OpType::Mul:
            if (is_const1) {
                Tensor val = producer1->get_tensor_attr("value");
                // Check if multiplying by 1 or 0
                return true;
            }
            break;

        default:
            break;
    }

    return false;
}

auto AlgebraicSimplificationPass::simplify_unary_op(std::shared_ptr<Node> node, Graph& graph) -> bool {
    // Simplify patterns like log(exp(x)) = x
    if (node->op_type() == OpType::Log && !node->inputs().empty()) {
        auto producer = node->inputs()[0]->node();
        if (producer && producer->op_type() == OpType::Exp) {
            // Would replace with identity
            return true;
        }
    }

    if (node->op_type() == OpType::Exp && !node->inputs().empty()) {
        auto producer = node->inputs()[0]->node();
        if (producer && producer->op_type() == OpType::Log) {
            // Would replace with identity
            return true;
        }
    }

    return false;
}

// ============================================================================
// Reshape Elimination Pass
// ============================================================================

auto ReshapeEliminationPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (const auto& node : graph.nodes()) {
        if (node->op_type() != OpType::Reshape) continue;

        if (node->inputs().empty()) continue;
        auto input = node->inputs()[0];

        // Check if reshape is redundant (same shape)
        auto target_shape = node->get_vec_attr("shape");
        if (input->shape() == target_shape) {
            // Would remove reshape node
            modified = true;
        }

        // Check for consecutive reshapes
        auto producer = input->node();
        if (producer && producer->op_type() == OpType::Reshape) {
            // Would merge reshapes
            modified = true;
        }
    }

    return modified;
}

// ============================================================================
// Compiler Implementation
// ============================================================================

Compiler::Compiler(bool enable_default_passes) {
    if (enable_default_passes) {
        add_pass(std::make_unique<DeadCodeEliminationPass>());
        add_pass(std::make_unique<ConstantFoldingPass>());
        add_pass(std::make_unique<FuseConvBatchNormPass>());
        add_pass(std::make_unique<FuseConvReluPass>());
        add_pass(std::make_unique<FuseLinearReluPass>());
        add_pass(std::make_unique<AlgebraicSimplificationPass>());
        add_pass(std::make_unique<ReshapeEliminationPass>());
        add_pass(std::make_unique<CommonSubexpressionEliminationPass>());
    }
}

auto Compiler::add_pass(std::unique_ptr<Pass> pass) -> void {
    passes_.push_back(std::move(pass));
}

auto Compiler::optimize(Graph& graph, int max_iterations) -> int {
    int total_changes = 0;

    for (int iter = 0; iter < max_iterations; ++iter) {
        int changes = run_passes(graph);
        total_changes += changes;

        if (changes == 0) {
            // Converged
            break;
        }
    }

    return total_changes;
}

auto Compiler::run_passes(Graph& graph) -> int {
    int changes = 0;

    for (const auto& pass : passes_) {
        bool modified = pass->run(graph);

        if (modified) {
            changes++;
            pass_stats_[pass->name()]++;

            if (verbose_) {
                std::cout << "Pass '" << pass->name() << "' made changes\n";
            }
        }
    }

    return changes;
}

// ============================================================================
// Convenience Function
// ============================================================================

auto optimize_graph(Graph& graph) -> int {
    Compiler compiler(true);
    return compiler.optimize(graph);
}

} // namespace jit
} // namespace tenzor
