/**
 * @file compiler.cpp
 * @brief Implementation of graph optimization passes
 */

#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
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

    // Apply replacements: redirect consumers and remove dead duplicate nodes
    for (const auto& [old_id, new_id] : value_replacements) {
        graph.replace_value(old_id, new_id);
    }

    // Remove duplicate nodes (those whose outputs were all replaced)
    std::unordered_set<std::string> replaced_outputs;
    for (const auto& [old_id, new_id] : value_replacements) {
        replaced_outputs.insert(old_id);
    }

    // Collect nodes to remove (those whose ALL outputs have been replaced)
    std::vector<std::shared_ptr<Node>> to_remove;
    for (const auto& node : graph.nodes()) {
        bool all_replaced = !node->outputs().empty();
        for (const auto& output : node->outputs()) {
            if (replaced_outputs.find(output->id()) == replaced_outputs.end()) {
                all_replaced = false;
                break;
            }
        }
        if (all_replaced) {
            to_remove.push_back(node);
        }
    }

    for (auto& node : to_remove) {
        graph.remove_node(node);
    }

    return modified;
}

auto CommonSubexpressionEliminationPass::compute_node_hash(const Node& node) -> size_t {
    size_t hash = std::hash<int>{}(static_cast<int>(node.op_type()));

    // Hash inputs
    for (const auto& input : node.inputs()) {
        hash ^= std::hash<std::string>{}(input->id()) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Hash all node attributes
    auto all_attrs = const_cast<Node&>(node).get_all_attrs();
    auto& attrs = std::get<0>(all_attrs);
    auto& int_attrs = std::get<1>(all_attrs);
    auto& vec_attrs = std::get<2>(all_attrs);
    auto& bool_attrs = std::get<3>(all_attrs);

    // Hash float attributes
    for (const auto& [name, val] : attrs) {
        hash ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<float>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Hash int attributes
    for (const auto& [name, val] : int_attrs) {
        hash ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<int64_t>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
    }

    // Hash vector attributes
    for (const auto& [name, vec] : vec_attrs) {
        hash ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        for (auto v : vec) {
            hash ^= std::hash<int64_t>{}(v) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        }
    }

    // Hash bool attributes
    for (const auto& [name, val] : bool_attrs) {
        hash ^= std::hash<std::string>{}(name) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
        hash ^= std::hash<bool>{}(val) + 0x9e3779b9 + (hash << 6) + (hash >> 2);
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

    // Compare all attributes
    auto all_attrs_a = const_cast<Node&>(a).get_all_attrs();
    auto all_attrs_b = const_cast<Node&>(b).get_all_attrs();

    auto& attrs_a = std::get<0>(all_attrs_a);
    auto& int_attrs_a = std::get<1>(all_attrs_a);
    auto& vec_attrs_a = std::get<2>(all_attrs_a);
    auto& bool_attrs_a = std::get<3>(all_attrs_a);

    auto& attrs_b = std::get<0>(all_attrs_b);
    auto& int_attrs_b = std::get<1>(all_attrs_b);
    auto& vec_attrs_b = std::get<2>(all_attrs_b);
    auto& bool_attrs_b = std::get<3>(all_attrs_b);

    // Compare float attributes
    if (attrs_a.size() != attrs_b.size()) return false;
    for (const auto& [name, val] : attrs_a) {
        auto it = attrs_b.find(name);
        if (it == attrs_b.end() || it->second != val) return false;
    }

    // Compare int attributes
    if (int_attrs_a.size() != int_attrs_b.size()) return false;
    for (const auto& [name, val] : int_attrs_a) {
        auto it = int_attrs_b.find(name);
        if (it == int_attrs_b.end() || it->second != val) return false;
    }

    // Compare vector attributes
    if (vec_attrs_a.size() != vec_attrs_b.size()) return false;
    for (const auto& [name, vec] : vec_attrs_a) {
        auto it = vec_attrs_b.find(name);
        if (it == vec_attrs_b.end() || it->second != vec) return false;
    }

    // Compare bool attributes
    if (bool_attrs_a.size() != bool_attrs_b.size()) return false;
    for (const auto& [name, val] : bool_attrs_a) {
        auto it = bool_attrs_b.find(name);
        if (it == bool_attrs_b.end() || it->second != val) return false;
    }

    return true;
}

// ============================================================================
// Constant Folding Pass
// ============================================================================

auto ConstantFoldingPass::run(Graph& graph) -> bool {
    bool modified = false;

    // Collect nodes to fold first (avoid modifying graph while iterating)
    std::vector<std::shared_ptr<Node>> foldable;
    for (const auto& node : graph.nodes()) {
        if (can_fold(*node)) {
            foldable.push_back(node);
        }
    }

    for (auto& node : foldable) {
        try {
            Tensor result = evaluate_constant(*node);

            // Create constant node with the folded result
            auto const_node = graph.create_node(OpType::Constant);
            const_node->set_tensor_attr("value", result);

            // Create output value for the constant node
            if (!node->outputs().empty()) {
                auto old_output = node->outputs()[0];
                std::string const_val_id = const_node->name() + "_out";
                auto const_val = graph.create_value(
                    const_val_id, old_output->shape(), old_output->dtype(), old_output->device());
                const_val->set_node(const_node);
                const_node->add_output(const_val);

                // Add the constant node to the graph
                graph.add_node(const_node);

                // Redirect all consumers of the original node to use the constant
                graph.replace_value(old_output->id(), const_val_id);

                // Remove the original node
                graph.remove_node(node);

                modified = true;
            }
        } catch (...) {
            // Failed to fold - continue
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

    // Validate that we have the necessary tensors
    if (gamma.numel() == 0 || running_var.numel() == 0 || conv_weight.numel() == 0) {
        return false;
    }

    // Compute fused parameters:
    //   inv_std = 1 / sqrt(var + eps)
    //   w' = gamma * w / sqrt(var + eps) = gamma * inv_std * w
    //   b' = gamma * (b - mean) / sqrt(var + eps) + beta
    Tensor var_plus_eps = running_var + eps;
    Tensor inv_std = tenzor::sqrt(var_plus_eps);
    // inv_std = 1 / sqrt(var + eps): compute scale = gamma / sqrt(var + eps)
    Tensor scale = gamma / inv_std;

    // Fuse weights: scale each output channel
    // conv_weight shape: [out_channels, in_channels, kH, kW]
    // scale shape: [out_channels]
    // We need to reshape scale to [out_channels, 1, 1, 1] for broadcasting
    auto w_shape = conv_weight.shape();
    if (w_shape.size() == 4) {
        Tensor scale_reshaped = scale.reshape({w_shape[0], 1, 1, 1});
        Tensor fused_weight = conv_weight * scale_reshaped;
        conv_node->set_tensor_attr("weight", fused_weight);
    } else {
        // Fallback: attempt direct element-wise (may not broadcast correctly)
        conv_node->set_tensor_attr("weight", conv_weight * scale);
    }

    // Fuse bias: b' = scale * (b - mean) + beta
    if (conv_bias.numel() > 0) {
        Tensor fused_bias = scale * (conv_bias - running_mean) + beta;
        conv_node->set_tensor_attr("bias", fused_bias);
    } else {
        // No conv bias: b' = scale * (0 - mean) + beta = -scale * mean + beta
        Tensor fused_bias = beta - scale * running_mean;
        conv_node->set_tensor_attr("bias", fused_bias);
    }

    conv_node->set_bool_attr("fused_bn", true);

    // Redirect consumers of BN output to use conv output, then remove BN node
    if (!conv_node->outputs().empty() && !bn_node->outputs().empty()) {
        graph.replace_value(bn_node->outputs()[0]->id(), conv_node->outputs()[0]->id());
        graph.remove_node(bn_node);
    }

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
    // Mark conv as having fused ReLU activation
    conv_node->set_bool_attr("fused_relu", true);

    // Redirect consumers of relu output to use conv output, then remove relu
    if (!conv_node->outputs().empty() && !relu_node->outputs().empty()) {
        graph.replace_value(relu_node->outputs()[0]->id(), conv_node->outputs()[0]->id());
        graph.remove_node(relu_node);
    }

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
    // Mark linear as having fused ReLU activation
    linear_node->set_bool_attr("fused_relu", true);

    // Redirect consumers of relu output to use linear output, then remove relu
    if (!linear_node->outputs().empty() && !relu_node->outputs().empty()) {
        graph.replace_value(relu_node->outputs()[0]->id(), linear_node->outputs()[0]->id());
        graph.remove_node(relu_node);
    }

    return true;
}

// ============================================================================
// MatMul + Add Fusion Pass
// ============================================================================

auto FuseMatMulAddPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (size_t i = 0; i + 1 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];

        if (node1->op_type() == OpType::MatMul && node2->op_type() == OpType::Add) {
            // Check if matmul output is used as one of the Add inputs
            if (!node1->outputs().empty() && node2->inputs().size() >= 2) {
                auto matmul_out_id = node1->outputs()[0]->id();
                auto add_in0_id = node2->inputs()[0]->id();
                auto add_in1_id = node2->inputs()[1]->id();

                // Determine which Add input is the matmul result and which is the bias
                std::shared_ptr<Value> bias_value;
                if (add_in0_id == matmul_out_id) {
                    bias_value = node2->inputs()[1];
                } else if (add_in1_id == matmul_out_id) {
                    bias_value = node2->inputs()[0];
                } else {
                    continue;  // MatMul output is not consumed by this Add
                }

                // Check that bias is 1D (typical bias pattern)
                if (bias_value->shape().size() == 1) {
                    // Check that matmul output is only used by this Add
                    auto matmul_out = node1->outputs()[0];
                    if (matmul_out->uses().size() <= 1) {
                        if (fuse_pair(node1, node2, graph)) {
                            modified = true;
                        }
                    }
                }
            }
        }
    }

    return modified;
}

auto FuseMatMulAddPass::fuse_pair(std::shared_ptr<Node> matmul_node,
                                    std::shared_ptr<Node> add_node,
                                    Graph& graph) -> bool {
    // Determine which Add input is the bias (not the matmul output)
    auto matmul_out_id = matmul_node->outputs()[0]->id();
    std::shared_ptr<Value> bias_value;
    if (add_node->inputs()[0]->id() == matmul_out_id) {
        bias_value = add_node->inputs()[1];
    } else {
        bias_value = add_node->inputs()[0];
    }

    // If the bias comes from a Constant node, store the tensor directly
    auto bias_producer = bias_value->node();
    if (bias_producer && bias_producer->op_type() == OpType::Constant) {
        Tensor bias_tensor = bias_producer->get_tensor_attr("value");
        if (bias_tensor.numel() > 0) {
            matmul_node->set_tensor_attr("fused_bias", bias_tensor);
        }
    }

    // Mark the matmul as having a fused bias and add the bias as a third input
    matmul_node->set_bool_attr("fused_bias", true);
    matmul_node->add_input(bias_value);

    // Redirect consumers of Add output to use MatMul output, then remove Add
    if (!matmul_node->outputs().empty() && !add_node->outputs().empty()) {
        graph.replace_value(add_node->outputs()[0]->id(), matmul_node->outputs()[0]->id());
        graph.remove_node(add_node);
    }

    return true;
}

// ============================================================================
// Conv + BatchNorm + ReLU Triple Fusion Pass
// ============================================================================

auto FuseConvBatchNormReluPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (size_t i = 0; i + 2 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];
        auto& node3 = graph.nodes()[i + 2];

        if (node1->op_type() == OpType::Conv2d &&
            node2->op_type() == OpType::BatchNorm2d &&
            node3->op_type() == OpType::ReLU) {
            // Verify data flow: conv -> bn -> relu
            bool flow_valid = true;
            if (!node1->outputs().empty() && !node2->inputs().empty()) {
                if (node1->outputs()[0]->id() != node2->inputs()[0]->id()) {
                    flow_valid = false;
                }
                // Conv output should only be used by BN
                if (node1->outputs()[0]->uses().size() > 1) {
                    flow_valid = false;
                }
            } else {
                flow_valid = false;
            }
            if (!node2->outputs().empty() && !node3->inputs().empty()) {
                if (node2->outputs()[0]->id() != node3->inputs()[0]->id()) {
                    flow_valid = false;
                }
                // BN output should only be used by ReLU
                if (node2->outputs()[0]->uses().size() > 1) {
                    flow_valid = false;
                }
            } else {
                flow_valid = false;
            }

            if (flow_valid && fuse_triple(node1, node2, node3, graph)) {
                modified = true;
                // Skip over the removed nodes
            }
        }
    }

    return modified;
}

auto FuseConvBatchNormReluPass::fuse_triple(std::shared_ptr<Node> conv_node,
                                              std::shared_ptr<Node> bn_node,
                                              std::shared_ptr<Node> relu_node,
                                              Graph& graph) -> bool {
    // Fuse BatchNorm parameters into Conv weights (same logic as FuseConvBatchNormPass)
    Tensor gamma = bn_node->get_tensor_attr("weight");
    Tensor beta = bn_node->get_tensor_attr("bias");
    Tensor running_mean = bn_node->get_tensor_attr("running_mean");
    Tensor running_var = bn_node->get_tensor_attr("running_var");
    float eps = bn_node->get_attr("eps");

    Tensor conv_weight = conv_node->get_tensor_attr("weight");
    Tensor conv_bias = conv_node->get_tensor_attr("bias");

    if (gamma.numel() == 0 || running_var.numel() == 0 || conv_weight.numel() == 0) {
        return false;
    }

    Tensor var_plus_eps = running_var + eps;
    Tensor inv_std = tenzor::sqrt(var_plus_eps);
    Tensor scale = gamma / inv_std;

    auto w_shape = conv_weight.shape();
    if (w_shape.size() == 4) {
        Tensor scale_reshaped = scale.reshape({w_shape[0], 1, 1, 1});
        Tensor fused_weight = conv_weight * scale_reshaped;
        conv_node->set_tensor_attr("weight", fused_weight);
    } else {
        conv_node->set_tensor_attr("weight", conv_weight * scale);
    }

    if (conv_bias.numel() > 0) {
        Tensor fused_bias = scale * (conv_bias - running_mean) + beta;
        conv_node->set_tensor_attr("bias", fused_bias);
    } else {
        Tensor fused_bias = beta - scale * running_mean;
        conv_node->set_tensor_attr("bias", fused_bias);
    }

    // Mark as triple-fused
    conv_node->set_bool_attr("fused_bn", true);
    conv_node->set_bool_attr("fused_relu", true);

    // Redirect consumers of ReLU output to use Conv output
    if (!conv_node->outputs().empty() && !relu_node->outputs().empty()) {
        graph.replace_value(relu_node->outputs()[0]->id(), conv_node->outputs()[0]->id());
        // Also redirect any remaining BN output references
        if (!bn_node->outputs().empty()) {
            graph.replace_value(bn_node->outputs()[0]->id(), conv_node->outputs()[0]->id());
        }
        graph.remove_node(relu_node);
        graph.remove_node(bn_node);
    }

    return true;
}

// ============================================================================
// LayerNorm + Activation Fusion Pass
// ============================================================================

auto FuseLayerNormActivationPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (size_t i = 0; i + 1 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];

        if (node1->op_type() == OpType::LayerNorm &&
            (node2->op_type() == OpType::ReLU || node2->op_type() == OpType::GELU)) {
            // Verify data flow: layernorm output feeds into activation
            bool flow_valid = false;
            if (!node1->outputs().empty() && !node2->inputs().empty()) {
                if (node1->outputs()[0]->id() == node2->inputs()[0]->id()) {
                    // LayerNorm output should only be used by the activation
                    if (node1->outputs()[0]->uses().size() <= 1) {
                        flow_valid = true;
                    }
                }
            }

            if (flow_valid && fuse_pair(node1, node2, graph)) {
                modified = true;
            }
        }
    }

    return modified;
}

auto FuseLayerNormActivationPass::fuse_pair(std::shared_ptr<Node> ln_node,
                                              std::shared_ptr<Node> act_node,
                                              Graph& graph) -> bool {
    // Determine activation type string for the fused attribute
    std::string activation_name;
    switch (act_node->op_type()) {
        case OpType::ReLU:
            activation_name = "relu";
            break;
        case OpType::GELU:
            activation_name = "gelu";
            break;
        default:
            return false;
    }

    // Mark LayerNorm as having a fused activation
    ln_node->set_bool_attr("fused_activation", true);
    // Store the activation type as an int attribute for efficient dispatch:
    // 1 = relu, 2 = gelu
    ln_node->set_int_attr("fused_activation_type",
                          act_node->op_type() == OpType::ReLU ? 1 : 2);

    // Redirect consumers of activation output to use LayerNorm output
    if (!ln_node->outputs().empty() && !act_node->outputs().empty()) {
        graph.replace_value(act_node->outputs()[0]->id(), ln_node->outputs()[0]->id());
        graph.remove_node(act_node);
    }

    return true;
}

// ============================================================================
// Algebraic Simplification Pass - Helpers
// ============================================================================

/**
 * @brief Check if a Constant node holds a tensor with all zero values.
 *
 * @param node Node to check (must be OpType::Constant)
 * @return true if the tensor is all zeros
 */
static auto is_all_zeros(const Node& node) -> bool {
    if (node.op_type() != OpType::Constant) return false;
    const Tensor& t = node.get_tensor_attr("value");
    if (t.numel() == 0) return false;

    // Check all elements are zero by comparing sum of absolute values
    try {
        Tensor abs_sum = tenzor::sum(tenzor::abs(t));
        // Access the scalar value - for Float32
        if (t.dtype() == DType::Float32) {
            return abs_sum.item<float>() == 0.0f;
        } else if (t.dtype() == DType::Float64) {
            return abs_sum.item<double>() == 0.0;
        }
        // For integer types, try float conversion
        return abs_sum.item<float>() == 0.0f;
    } catch (...) {
        return false;
    }
}

/**
 * @brief Check if a Constant node holds a tensor with all one values.
 *
 * @param node Node to check (must be OpType::Constant)
 * @return true if the tensor is all ones
 */
static auto is_all_ones(const Node& node) -> bool {
    if (node.op_type() != OpType::Constant) return false;
    const Tensor& t = node.get_tensor_attr("value");
    if (t.numel() == 0) return false;

    // Check all elements are one: sum(abs(t - 1)) == 0
    try {
        Tensor diff = t - 1.0;
        Tensor abs_sum = tenzor::sum(tenzor::abs(diff));
        if (t.dtype() == DType::Float32) {
            return abs_sum.item<float>() == 0.0f;
        } else if (t.dtype() == DType::Float64) {
            return abs_sum.item<double>() == 0.0;
        }
        return abs_sum.item<float>() == 0.0f;
    } catch (...) {
        return false;
    }
}

// ============================================================================
// Algebraic Simplification Pass
// ============================================================================

auto AlgebraicSimplificationPass::run(Graph& graph) -> bool {
    bool modified = false;

    // Collect nodes to process (we may modify the graph during iteration)
    std::vector<std::shared_ptr<Node>> nodes_to_process(graph.nodes().begin(), graph.nodes().end());

    for (const auto& node : nodes_to_process) {
        // Check if node is still in the graph (may have been removed by previous simplification)
        auto& current_nodes = graph.nodes();
        if (std::find(current_nodes.begin(), current_nodes.end(), node) == current_nodes.end()) {
            continue;
        }

        if (simplify_binary_op(node, graph)) {
            modified = true;
        } else if (simplify_unary_op(node, graph)) {
            modified = true;
        }
    }

    return modified;
}

auto AlgebraicSimplificationPass::simplify_binary_op(std::shared_ptr<Node> node, Graph& graph) -> bool {
    if (node->inputs().size() < 2) return false;

    auto input0 = node->inputs()[0];
    auto input1 = node->inputs()[1];

    auto producer0 = input0->node();
    auto producer1 = input1->node();

    bool is_const0 = producer0 && producer0->op_type() == OpType::Constant;
    bool is_const1 = producer1 && producer1->op_type() == OpType::Constant;

    if (!is_const0 && !is_const1) return false;

    switch (node->op_type()) {
        case OpType::Add:
            // x + 0 = x
            if (is_const1 && is_all_zeros(*producer1)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            // 0 + x = x
            if (is_const0 && is_all_zeros(*producer0)) {
                graph.replace_node_with_value(node, input1->id());
                return true;
            }
            break;

        case OpType::Sub:
            // x - 0 = x
            if (is_const1 && is_all_zeros(*producer1)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            break;

        case OpType::Mul:
            // x * 1 = x
            if (is_const1 && is_all_ones(*producer1)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            // 1 * x = x
            if (is_const0 && is_all_ones(*producer0)) {
                graph.replace_node_with_value(node, input1->id());
                return true;
            }
            // x * 0 = 0 (redirect to the zero constant)
            if (is_const1 && is_all_zeros(*producer1)) {
                graph.replace_node_with_value(node, input1->id());
                return true;
            }
            // 0 * x = 0
            if (is_const0 && is_all_zeros(*producer0)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            break;

        case OpType::Div:
            // x / 1 = x
            if (is_const1 && is_all_ones(*producer1)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            break;

        default:
            break;
    }

    return false;
}

auto AlgebraicSimplificationPass::simplify_unary_op(std::shared_ptr<Node> node, Graph& graph) -> bool {
    if (node->inputs().empty()) return false;

    // log(exp(x)) = x
    if (node->op_type() == OpType::Log) {
        auto producer = node->inputs()[0]->node();
        if (producer && producer->op_type() == OpType::Exp && !producer->inputs().empty()) {
            // Bypass both log and exp: redirect to exp's input
            graph.replace_node_with_value(node, producer->inputs()[0]->id());
            return true;
        }
    }

    // exp(log(x)) = x
    if (node->op_type() == OpType::Exp) {
        auto producer = node->inputs()[0]->node();
        if (producer && producer->op_type() == OpType::Log && !producer->inputs().empty()) {
            // Bypass both exp and log: redirect to log's input
            graph.replace_node_with_value(node, producer->inputs()[0]->id());
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

    // Collect reshape nodes (we may modify the graph during iteration)
    std::vector<std::shared_ptr<Node>> reshape_nodes;
    for (const auto& node : graph.nodes()) {
        if (node->op_type() == OpType::Reshape) {
            reshape_nodes.push_back(node);
        }
    }

    for (auto& node : reshape_nodes) {
        // Check if node is still in the graph
        auto& current_nodes = graph.nodes();
        if (std::find(current_nodes.begin(), current_nodes.end(), node) == current_nodes.end()) {
            continue;
        }

        if (node->inputs().empty()) continue;
        auto input = node->inputs()[0];

        // Case 1: Same-shape reshape bypass
        // If output shape == input shape, the reshape is a no-op
        auto target_shape = node->get_vec_attr("shape");
        if (input->shape() == target_shape) {
            graph.replace_node_with_value(node, input->id());
            modified = true;
            continue;
        }

        // Case 2: Consecutive reshapes - merge
        // reshape(reshape(x, s1), s2) -> reshape(x, s2)
        auto producer = input->node();
        if (producer && producer->op_type() == OpType::Reshape && !producer->inputs().empty()) {
            // Rewire: this reshape takes the original input of the first reshape
            auto original_input = producer->inputs()[0];
            node->replace_input(0, original_input);
            modified = true;
        }
    }

    return modified;
}

// ============================================================================
// Memory Planning Pass
// ============================================================================

auto MemoryPlanningPass::run(Graph& graph) -> bool {
    MemoryPlanner planner;
    planner.set_alignment(alignment_);
    plan_ = planner.plan(graph);

    // Memory planning annotates values but does not structurally modify
    // the graph, so return false to avoid triggering re-optimization.
    return false;
}

// ============================================================================
// Compiler Implementation
// ============================================================================

Compiler::Compiler(bool enable_default_passes) {
    if (enable_default_passes) {
        add_pass(std::make_unique<DeadCodeEliminationPass>());
        add_pass(std::make_unique<ConstantFoldingPass>());
        add_pass(std::make_unique<FuseConvBatchNormReluPass>());  // Triple fusion before individual passes
        add_pass(std::make_unique<FuseConvBatchNormPass>());
        add_pass(std::make_unique<FuseConvReluPass>());
        add_pass(std::make_unique<FuseLinearReluPass>());
        add_pass(std::make_unique<FuseMatMulAddPass>());
        add_pass(std::make_unique<FuseLayerNormActivationPass>());
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

    // Determine whether to run memory planning:
    // - If the user explicitly set it via set_memory_planning(), use that setting.
    // - Otherwise, auto-enable when the graph has enough nodes to benefit.
    bool should_plan = enable_memory_planning_;
    if (!memory_planning_explicit_) {
        should_plan = graph.num_nodes() >= memory_planning_threshold_;
        if (verbose_ && should_plan) {
            std::cout << "MemoryPlanning: auto-enabled for graph with "
                      << graph.num_nodes() << " nodes (threshold: "
                      << memory_planning_threshold_ << ")\n";
        }
    }

    // Run memory planning as the final step after all structural passes
    if (should_plan) {
        memory_plan_ = plan_memory(graph);

        if (verbose_) {
            std::cout << "MemoryPlanning: " << memory_plan_.num_values_planned << " values planned, "
                      << memory_plan_.num_values_reused << " sharing buffers, "
                      << memory_plan_.pool_sizes.size() << " pool buffers, "
                      << memory_plan_.total_memory << " bytes total\n";
        }
    }

    return total_changes;
}

auto Compiler::plan_memory(Graph& graph) -> MemoryPlan {
    MemoryPlanner planner;
    return planner.plan(graph);
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
