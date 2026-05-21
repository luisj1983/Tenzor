/**
 * @file fusion_optimizer.cpp
 * @brief Implementation of graph-level kernel fusion optimization
 */

#include "tenzor/ops/fusion_optimizer.hpp"
#include "tenzor/ops/fused_ops.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include <algorithm>
#include <queue>
#include <stack>
#include <sstream>
#include <stdexcept>
#include <cmath>

namespace tenzor {
namespace ops {

// ==============================================================================
// Helper Functions
// ==============================================================================

auto string_to_op_type(const std::string& op_name) -> OpType {
    static const std::unordered_map<std::string, OpType> name_map = {
        {"matmul", OpType::MatMul},
        {"linear", OpType::Linear},
        {"conv2d", OpType::Conv2d},
        {"relu", OpType::ReLU},
        {"gelu", OpType::GELU},
        {"sigmoid", OpType::Sigmoid},
        {"tanh", OpType::Tanh},
        {"batchnorm2d", OpType::BatchNorm2d},
        {"layernorm", OpType::LayerNorm},
        {"add", OpType::Add},
        {"mul", OpType::Mul},
        {"sub", OpType::Sub},
        {"div", OpType::Div},
        {"softmax", OpType::Softmax},
        {"dropout", OpType::Dropout},
        {"crossentropy", OpType::CrossEntropy}
    };

    auto it = name_map.find(op_name);
    return (it != name_map.end()) ? it->second : OpType::Unknown;
}

auto op_type_to_string(OpType op_type) -> std::string {
    switch (op_type) {
        case OpType::MatMul: return "matmul";
        case OpType::Linear: return "linear";
        case OpType::Conv2d: return "conv2d";
        case OpType::ReLU: return "relu";
        case OpType::GELU: return "gelu";
        case OpType::Sigmoid: return "sigmoid";
        case OpType::Tanh: return "tanh";
        case OpType::BatchNorm2d: return "batchnorm2d";
        case OpType::LayerNorm: return "layernorm";
        case OpType::Add: return "add";
        case OpType::Mul: return "mul";
        case OpType::Sub: return "sub";
        case OpType::Div: return "div";
        case OpType::Softmax: return "softmax";
        case OpType::Dropout: return "dropout";
        case OpType::CrossEntropy: return "crossentropy";
        default: return "unknown";
    }
}

// ==============================================================================
// FusionGraph Implementation
// ==============================================================================

auto FusionGraph::add_node(
    OpType op_type,
    const std::string& op_name,
    const std::vector<size_t>& inputs,
    const std::unordered_map<std::string, std::string>& attributes
) -> size_t {
    size_t node_id = next_id_++;
    auto node = std::make_unique<FusionNode>(node_id, op_type, op_name);
    node->inputs = inputs;
    node->attributes = attributes;

    // Add edges from inputs to this node
    for (size_t input_id : inputs) {
        adjacency_list_[input_id].insert(node_id);
        if (input_id < nodes_.size() && nodes_[input_id]) {
            nodes_[input_id]->outputs.push_back(node_id);
        }
    }

    nodes_.push_back(std::move(node));
    return node_id;
}

auto FusionGraph::add_edge(size_t from, size_t to) -> void {
    adjacency_list_[from].insert(to);

    if (from < nodes_.size() && nodes_[from]) {
        nodes_[from]->outputs.push_back(to);
    }
    if (to < nodes_.size() && nodes_[to]) {
        nodes_[to]->inputs.push_back(from);
    }
}

auto FusionGraph::get_node(size_t node_id) const -> const FusionNode& {
    if (node_id >= nodes_.size() || !nodes_[node_id]) {
        throw std::out_of_range("FusionGraph::get_node: invalid node ID " +
                               std::to_string(node_id));
    }
    return *nodes_[node_id];
}

auto FusionGraph::get_node_mut(size_t node_id) -> FusionNode& {
    if (node_id >= nodes_.size() || !nodes_[node_id]) {
        throw std::out_of_range("FusionGraph::get_node_mut: invalid node ID " +
                               std::to_string(node_id));
    }
    return *nodes_[node_id];
}

auto FusionGraph::get_nodes() const
    -> const std::vector<std::unique_ptr<FusionNode>>& {
    return nodes_;
}

auto FusionGraph::get_inputs(size_t node_id) const -> std::vector<size_t> {
    if (node_id >= nodes_.size() || !nodes_[node_id]) {
        return {};
    }
    return nodes_[node_id]->inputs;
}

auto FusionGraph::get_outputs(size_t node_id) const -> std::vector<size_t> {
    if (node_id >= nodes_.size() || !nodes_[node_id]) {
        return {};
    }
    return nodes_[node_id]->outputs;
}

auto FusionGraph::topological_sort() const -> std::vector<size_t> {
    std::vector<size_t> sorted;
    std::unordered_set<size_t> visited;
    std::stack<size_t> stack;

    // Helper function for DFS
    std::function<void(size_t)> dfs = [&](size_t node_id) {
        if (visited.count(node_id)) return;
        visited.insert(node_id);

        auto it = adjacency_list_.find(node_id);
        if (it != adjacency_list_.end()) {
            for (size_t neighbor : it->second) {
                dfs(neighbor);
            }
        }

        stack.push(node_id);
    };

    // Visit all nodes
    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i] && !visited.count(i)) {
            dfs(i);
        }
    }

    // Extract sorted order
    while (!stack.empty()) {
        sorted.push_back(stack.top());
        stack.pop();
    }

    return sorted;
}

auto FusionGraph::has_cycle() const -> bool {
    std::unordered_set<size_t> visited;
    std::unordered_set<size_t> rec_stack;

    for (size_t i = 0; i < nodes_.size(); ++i) {
        if (nodes_[i] && !visited.count(i)) {
            if (has_cycle_util(i, visited, rec_stack)) {
                return true;
            }
        }
    }
    return false;
}

auto FusionGraph::has_cycle_util(
    size_t node,
    std::unordered_set<size_t>& visited,
    std::unordered_set<size_t>& rec_stack
) const -> bool {
    visited.insert(node);
    rec_stack.insert(node);

    auto it = adjacency_list_.find(node);
    if (it != adjacency_list_.end()) {
        for (size_t neighbor : it->second) {
            if (!visited.count(neighbor)) {
                if (has_cycle_util(neighbor, visited, rec_stack)) {
                    return true;
                }
            } else if (rec_stack.count(neighbor)) {
                return true;
            }
        }
    }

    rec_stack.erase(node);
    return false;
}

auto FusionGraph::clear() -> void {
    nodes_.clear();
    adjacency_list_.clear();
    next_id_ = 0;
}

auto FusionGraph::to_dot() const -> std::string {
    std::ostringstream ss;
    ss << "digraph FusionGraph {\n";
    ss << "  rankdir=TB;\n";
    ss << "  node [shape=box, style=rounded];\n\n";

    // Add nodes
    for (const auto& node : nodes_) {
        if (!node) continue;

        ss << "  n" << node->id << " [label=\"" << node->op_name;
        if (node->is_fused) {
            ss << "\\n(fused: " << node->fusion_group << ")";
        }
        ss << "\"];\n";
    }

    ss << "\n";

    // Add edges
    for (const auto& [from, tos] : adjacency_list_) {
        for (size_t to : tos) {
            ss << "  n" << from << " -> n" << to << ";\n";
        }
    }

    ss << "}\n";
    return ss.str();
}

// ==============================================================================
// FusionPattern Implementation
// ==============================================================================

FusionPattern::FusionPattern(const std::string& pattern_name)
    : pattern_name_(pattern_name) {

    // Configure pattern-specific parameters
    if (pattern_name == "linear_relu") {
        pattern_ops_ = {OpType::Linear, OpType::ReLU};
        expected_speedup_ = 1.6f;
    } else if (pattern_name == "conv_bn_relu") {
        pattern_ops_ = {OpType::Conv2d, OpType::BatchNorm2d, OpType::ReLU};
        expected_speedup_ = 2.2f;
    } else if (pattern_name == "matmul_add") {
        pattern_ops_ = {OpType::MatMul, OpType::Add};
        expected_speedup_ = 1.4f;
    } else if (pattern_name == "elementwise_chain") {
        // Variable pattern - any chain of element-wise ops
        expected_speedup_ = 1.8f;
    } else if (pattern_name == "attention") {
        pattern_ops_ = {OpType::MatMul, OpType::Softmax, OpType::Dropout, OpType::MatMul};
        expected_speedup_ = 2.5f;
    }
}

auto FusionPattern::match(const FusionGraph& graph, size_t start_node) const
    -> Match {

    if (pattern_name_ == "linear_relu") {
        return match_linear_relu(graph, start_node);
    } else if (pattern_name_ == "conv_bn_relu") {
        return match_conv_bn_relu(graph, start_node);
    } else if (pattern_name_ == "matmul_add") {
        return match_matmul_add(graph, start_node);
    } else if (pattern_name_ == "elementwise_chain") {
        return match_elementwise_chain(graph, start_node);
    } else if (pattern_name_ == "attention") {
        return match_attention(graph, start_node);
    }

    return Match{};
}

auto FusionPattern::match_linear_relu(const FusionGraph& graph, size_t start) const
    -> Match {
    try {
        const auto& node = graph.get_node(start);
        if (node.op_type != OpType::Linear || node.is_fused) {
            return Match{};
        }

        // Check for single ReLU consumer
        auto outputs = node.outputs;
        if (outputs.size() != 1) {
            return Match{};
        }

        const auto& relu_node = graph.get_node(outputs[0]);
        if (relu_node.op_type != OpType::ReLU || relu_node.is_fused) {
            return Match{};
        }

        // Successful match
        Match match;
        match.matched_nodes = {start, outputs[0]};
        match.confidence = 1.0f;
        match.pattern_name = "linear_relu";
        return match;

    } catch (const std::out_of_range&) {
        return Match{};
    }
}

auto FusionPattern::match_conv_bn_relu(const FusionGraph& graph, size_t start) const
    -> Match {
    try {
        const auto& conv_node = graph.get_node(start);
        if (conv_node.op_type != OpType::Conv2d || conv_node.is_fused) {
            return Match{};
        }

        auto outputs = conv_node.outputs;
        if (outputs.size() != 1) {
            return Match{};
        }

        const auto& bn_node = graph.get_node(outputs[0]);
        if (bn_node.op_type != OpType::BatchNorm2d || bn_node.is_fused) {
            return Match{};
        }

        auto bn_outputs = bn_node.outputs;
        if (bn_outputs.size() != 1) {
            return Match{};
        }

        const auto& relu_node = graph.get_node(bn_outputs[0]);
        if (relu_node.op_type != OpType::ReLU || relu_node.is_fused) {
            return Match{};
        }

        // Successful match
        Match match;
        match.matched_nodes = {start, outputs[0], bn_outputs[0]};
        match.confidence = 1.0f;
        match.pattern_name = "conv_bn_relu";
        return match;

    } catch (const std::out_of_range&) {
        return Match{};
    }
}

auto FusionPattern::match_matmul_add(const FusionGraph& graph, size_t start) const
    -> Match {
    try {
        const auto& matmul_node = graph.get_node(start);
        if (matmul_node.op_type != OpType::MatMul || matmul_node.is_fused) {
            return Match{};
        }

        auto outputs = matmul_node.outputs;
        if (outputs.size() != 1) {
            return Match{};
        }

        const auto& add_node = graph.get_node(outputs[0]);
        if (add_node.op_type != OpType::Add || add_node.is_fused) {
            return Match{};
        }

        // Check that add is a bias addition (one input is matmul, other is smaller)
        // This is a simplified check - more sophisticated checks could be added
        if (add_node.inputs.size() != 2) {
            return Match{};
        }

        Match match;
        match.matched_nodes = {start, outputs[0]};
        match.confidence = 0.9f;  // Slightly lower confidence as pattern is ambiguous
        match.pattern_name = "matmul_add";
        return match;

    } catch (const std::out_of_range&) {
        return Match{};
    }
}

auto FusionPattern::match_elementwise_chain(const FusionGraph& graph, size_t start) const
    -> Match {
    try {
        const auto& start_node = graph.get_node(start);

        // Check if node is element-wise
        static const std::unordered_set<OpType> elementwise_ops = {
            OpType::Add, OpType::Mul, OpType::Sub, OpType::Div,
            OpType::ReLU, OpType::GELU, OpType::Sigmoid, OpType::Tanh
        };

        if (elementwise_ops.find(start_node.op_type) == elementwise_ops.end() ||
            start_node.is_fused) {
            return Match{};
        }

        // Follow chain of element-wise operations
        std::vector<size_t> chain = {start};
        size_t current = start;

        while (true) {
            auto outputs = graph.get_outputs(current);
            if (outputs.size() != 1) break;

            const auto& next_node = graph.get_node(outputs[0]);
            if (elementwise_ops.find(next_node.op_type) == elementwise_ops.end() ||
                next_node.is_fused) {
                break;
            }

            chain.push_back(outputs[0]);
            current = outputs[0];

            // Limit chain length to avoid overly aggressive fusion
            if (chain.size() >= 5) break;
        }

        // Only fuse if we have at least 3 operations
        if (chain.size() >= 3) {
            Match match;
            match.matched_nodes = chain;
            match.confidence = 0.85f;
            match.pattern_name = "elementwise_chain";
            return match;
        }

        return Match{};

    } catch (const std::out_of_range&) {
        return Match{};
    }
}

auto FusionPattern::match_attention(const FusionGraph& graph, size_t start) const
    -> Match {
    try {
        // Simplified attention pattern: Q @ K.T -> Softmax -> Dropout -> @ V
        const auto& qk_matmul = graph.get_node(start);
        if (qk_matmul.op_type != OpType::MatMul || qk_matmul.is_fused) {
            return Match{};
        }

        auto outputs = qk_matmul.outputs;
        if (outputs.size() != 1) {
            return Match{};
        }

        const auto& softmax_node = graph.get_node(outputs[0]);
        if (softmax_node.op_type != OpType::Softmax || softmax_node.is_fused) {
            return Match{};
        }

        outputs = softmax_node.outputs;
        if (outputs.size() != 1) {
            return Match{};
        }

        // Dropout might be optional in inference
        const auto& next_node = graph.get_node(outputs[0]);
        size_t dropout_id = 0;
        size_t matmul_v_id = 0;

        if (next_node.op_type == OpType::Dropout && !next_node.is_fused) {
            dropout_id = outputs[0];
            outputs = next_node.outputs;
            if (outputs.size() != 1) {
                return Match{};
            }
            matmul_v_id = outputs[0];
        } else if (next_node.op_type == OpType::MatMul && !next_node.is_fused) {
            matmul_v_id = outputs[0];
        } else {
            return Match{};
        }

        const auto& matmul_v = graph.get_node(matmul_v_id);
        if (matmul_v.op_type != OpType::MatMul || matmul_v.is_fused) {
            return Match{};
        }

        Match match;
        match.matched_nodes = {start, softmax_node.id, matmul_v_id};
        if (dropout_id > 0) {
            match.matched_nodes.insert(match.matched_nodes.begin() + 2, dropout_id);
        }
        match.confidence = 0.95f;
        match.pattern_name = "attention";
        return match;

    } catch (const std::out_of_range&) {
        return Match{};
    }
}

// ==============================================================================
// FusionOptimizer Implementation
// ==============================================================================

FusionOptimizer::FusionOptimizer(bool enable_aggressive)
    : aggressive_mode_(enable_aggressive) {}

auto FusionOptimizer::add_pattern(const std::string& pattern_name) -> bool {
    if (pattern_name == "all") {
        add_pattern("linear_relu");
        add_pattern("conv_bn_relu");
        add_pattern("matmul_add");
        add_pattern("elementwise_chain");
        add_pattern("attention");
        return true;
    }

    // Check if pattern already exists
    for (const auto& pattern : patterns_) {
        if (pattern->get_name() == pattern_name) {
            return false;  // Already added
        }
    }

    // Check if pattern is supported
    static const std::unordered_set<std::string> supported = {
        "linear_relu", "conv_bn_relu", "matmul_add",
        "elementwise_chain", "attention"
    };

    if (supported.find(pattern_name) == supported.end()) {
        return false;
    }

    patterns_.push_back(std::make_unique<FusionPattern>(pattern_name));
    return true;
}

auto FusionOptimizer::remove_pattern(const std::string& pattern_name) -> bool {
    auto it = std::remove_if(patterns_.begin(), patterns_.end(),
        [&pattern_name](const auto& pattern) {
            return pattern->get_name() == pattern_name;
        });

    if (it != patterns_.end()) {
        patterns_.erase(it, patterns_.end());
        return true;
    }
    return false;
}

auto FusionOptimizer::optimize(const FusionGraph& graph) -> FusionGraph {
    // Reset statistics
    reset_statistics();
    stats_.num_nodes_original = graph.size();

    // Detect fusion opportunities
    auto fusions = detect_patterns(graph);

    // Validate fusions
    std::vector<FusedOp> valid_fusions;
    for (const auto& fusion : fusions) {
        if (validate_fusion(graph, fusion)) {
            valid_fusions.push_back(fusion);
        }
    }

    // Rewrite graph with fused operations
    auto optimized = rewrite_graph(graph, valid_fusions);

    // Compute statistics
    compute_statistics(graph, optimized, valid_fusions);

    return optimized;
}

auto FusionOptimizer::detect_patterns(const FusionGraph& graph)
    -> std::vector<FusedOp> {

    std::vector<FusedOp> detected_fusions;
    std::unordered_set<size_t> fused_nodes;

    // Get topological order for systematic traversal
    auto sorted = graph.topological_sort();

    for (size_t node_id : sorted) {
        // Skip already fused nodes
        if (fused_nodes.count(node_id)) {
            continue;
        }

        // Try each pattern
        for (const auto& pattern : patterns_) {
            auto match = pattern->match(graph, node_id);
            if (match) {
                // Check if any matched nodes are already fused
                bool conflict = false;
                for (size_t matched_id : match.matched_nodes) {
                    if (fused_nodes.count(matched_id)) {
                        conflict = true;
                        break;
                    }
                }

                if (!conflict) {
                    // Create fused operation
                    FusedOp fused_op(match.pattern_name, match.matched_nodes);
                    fused_op.estimated_speedup = pattern->get_speedup();

                    // Mark nodes as fused
                    for (size_t matched_id : match.matched_nodes) {
                        fused_nodes.insert(matched_id);
                    }

                    detected_fusions.push_back(fused_op);
                    break;  // Don't try other patterns for this node
                }
            }
        }
    }

    return detected_fusions;
}

auto FusionOptimizer::rewrite_graph(
    const FusionGraph& graph,
    const std::vector<FusedOp>& fusions
) -> FusionGraph {

    FusionGraph optimized;
    std::unordered_map<size_t, size_t> old_to_new_id;
    std::unordered_set<size_t> fused_nodes;

    // Mark all nodes that are part of fusions
    for (const auto& fusion : fusions) {
        for (size_t node_id : fusion.node_ids) {
            fused_nodes.insert(node_id);
        }
    }

    // Process fusions first
    for (size_t fusion_idx = 0; fusion_idx < fusions.size(); ++fusion_idx) {
        const auto& fusion = fusions[fusion_idx];

        // Collect inputs from first node in fusion
        std::vector<size_t> fused_inputs;
        for (size_t node_id : fusion.node_ids) {
            const auto& node = graph.get_node(node_id);
            for (size_t input_id : node.inputs) {
                if (fused_nodes.find(input_id) == fused_nodes.end()) {
                    // Map to new ID if already processed
                    if (old_to_new_id.count(input_id)) {
                        fused_inputs.push_back(old_to_new_id[input_id]);
                    }
                }
            }
        }

        // Remove duplicates
        std::sort(fused_inputs.begin(), fused_inputs.end());
        fused_inputs.erase(
            std::unique(fused_inputs.begin(), fused_inputs.end()),
            fused_inputs.end()
        );

        // Create fused node
        std::unordered_map<std::string, std::string> attrs;
        attrs["fusion_type"] = fusion.fused_op_name;

        size_t new_id = optimized.add_node(
            OpType::Unknown,  // Fused ops have special type
            "fused_" + fusion.fused_op_name,
            fused_inputs,
            attrs
        );

        // Map all fused nodes to this new node
        for (size_t old_id : fusion.node_ids) {
            old_to_new_id[old_id] = new_id;
        }
    }

    // Copy remaining nodes
    for (const auto& node : graph.get_nodes()) {
        if (!node || fused_nodes.count(node->id)) {
            continue;
        }

        // Map input IDs
        std::vector<size_t> new_inputs;
        for (size_t input_id : node->inputs) {
            if (old_to_new_id.count(input_id)) {
                new_inputs.push_back(old_to_new_id[input_id]);
            }
        }

        size_t new_id = optimized.add_node(
            node->op_type,
            node->op_name,
            new_inputs,
            node->attributes
        );

        old_to_new_id[node->id] = new_id;
    }

    return optimized;
}

auto FusionOptimizer::validate_fusion(
    const FusionGraph& graph,
    const FusedOp& fusion
) -> bool {
    // Check if all nodes exist
    for (size_t node_id : fusion.node_ids) {
        try {
            graph.get_node(node_id);
        } catch (const std::out_of_range&) {
            return false;
        }
    }

    // Additional validation could be added here:
    // - Check tensor shapes are compatible
    // - Check data types are supported
    // - Check device compatibility
    // - Check memory constraints

    return true;
}

auto FusionOptimizer::estimate_speedup(const FusedOp& fusion) const -> float {
    return fusion.estimated_speedup;
}

auto FusionOptimizer::compute_statistics(
    const FusionGraph& original,
    const FusionGraph& optimized,
    const std::vector<FusedOp>& fusions
) -> void {

    stats_.num_nodes_optimized = optimized.size();
    stats_.num_fusions = fusions.size();

    // Estimate kernel launches saved
    size_t kernels_saved = 0;
    for (const auto& fusion : fusions) {
        // Each fusion saves (N-1) kernel launches where N is number of fused ops
        kernels_saved += fusion.node_ids.size() - 1;
        stats_.pattern_counts[fusion.fused_op_name]++;
    }
    stats_.num_kernel_launches_saved = kernels_saved;

    // Estimate overall speedup (weighted average)
    float total_speedup = 0.0f;
    for (const auto& fusion : fusions) {
        total_speedup += fusion.estimated_speedup;
    }

    if (!fusions.empty()) {
        stats_.expected_speedup = total_speedup / fusions.size();
    } else {
        stats_.expected_speedup = 1.0f;
    }

    // Estimate memory bandwidth reduction
    // Each fused operation saves approximately 30-50% of memory traffic
    stats_.memory_bandwidth_reduction =
        (kernels_saved * 0.4f) / std::max(1.0f, static_cast<float>(original.size()));
}

auto FusionOptimizer::get_statistics() const -> const Statistics& {
    return stats_;
}

auto FusionOptimizer::reset_statistics() -> void {
    stats_ = Statistics{};
}

auto FusionOptimizer::set_aggressive_mode(bool enable) -> void {
    aggressive_mode_ = enable;
}

auto FusionOptimizer::set_backend(const std::string& backend, bool enable) -> void {
    if (backend == "cuda") {
        cuda_enabled_ = enable;
    } else if (backend == "rocm") {
        rocm_enabled_ = enable;
    } else if (backend == "cpu") {
        cpu_enabled_ = enable;
    } else if (backend == "all") {
        cuda_enabled_ = enable;
        rocm_enabled_ = enable;
        cpu_enabled_ = enable;
    }
}

auto FusionOptimizer::is_pattern_supported(const std::string& pattern_name) const
    -> bool {
    static const std::unordered_set<std::string> supported = {
        "linear_relu", "conv_bn_relu", "matmul_add",
        "elementwise_chain", "attention"
    };
    return supported.find(pattern_name) != supported.end();
}

auto FusionOptimizer::get_supported_patterns() const -> std::vector<std::string> {
    return {
        "linear_relu",
        "conv_bn_relu",
        "matmul_add",
        "elementwise_chain",
        "attention"
    };
}

// ==============================================================================
// Helper Functions
// ==============================================================================

// Audit C.5: map a Function's forward `OpId` to the fusion-graph's
// `OpType`. Only the OpIds that fusion patterns can actually match
// produce a concrete OpType; everything else returns `OpType::Unknown`,
// so the fusion-graph builder still records the node (preserving
// connectivity) but the optimiser's pattern matchers correctly skip it.
//
// The mapping is intentionally minimal — extending it lets new fusion
// patterns activate (e.g. add OpId::Conv1dForward → OpType::Conv2d
// once a Conv1d-aware fusion exists).
static auto fusion_op_type_from_op_id(OpId id) -> OpType {
    switch (id) {
        case OpId::MatMul:           return OpType::MatMul;
        case OpId::Linear:           return OpType::Linear;
        case OpId::Conv2dForward:
        case OpId::Conv2dBackwardInput:
        case OpId::Conv2dBackwardWeight:
                                     return OpType::Conv2d;
        case OpId::Gelu:             return OpType::GELU;
        case OpId::Sigmoid:          return OpType::Sigmoid;
        case OpId::Tanh:             return OpType::Tanh;
        case OpId::BatchNorm2dForward:
        case OpId::BatchNorm2dForwardAffine:
        case OpId::BatchNorm2dBackward:
                                     return OpType::BatchNorm2d;
        case OpId::LayerNorm:        return OpType::LayerNorm;
        case OpId::Add:              return OpType::Add;
        case OpId::Mul:              return OpType::Mul;
        case OpId::Sub:              return OpType::Sub;
        case OpId::Div:              return OpType::Div;
        case OpId::Softmax:          return OpType::Softmax;
        case OpId::Dropout:          return OpType::Dropout;
        case OpId::FusedSoftmaxCrossEntropy:
                                     return OpType::CrossEntropy;
        default:                     return OpType::Unknown;
    }
}

auto build_fusion_graph_from_autograd(
    [[maybe_unused]] const ComputationGraph& comp_graph,
    const std::shared_ptr<GraphNode>& root
) -> FusionGraph {
    FusionGraph fusion_graph;
    std::unordered_map<GraphNode*, size_t> node_map;

    // BFS traversal of autograd graph
    std::queue<std::shared_ptr<GraphNode>> queue;
    std::unordered_set<GraphNode*> visited;

    queue.push(root);
    visited.insert(root.get());

    while (!queue.empty()) {
        auto node = queue.front();
        queue.pop();

        // Create fusion node
        std::vector<size_t> inputs;
        for (const auto& weak_next : node->next_nodes) {
            if (auto next = weak_next.lock()) {
                if (node_map.count(next.get())) {
                    inputs.push_back(node_map[next.get()]);
                }
            }
        }

        // Audit C.5: replace the legacy "op_name = unknown; op_type =
        // Unknown" stub with a real OpId-based mapping. The Function
        // base class now exposes op_id() (audit A.2); subclasses that
        // have opted in return their canonical forward OpId, which we
        // translate into the fusion-graph's OpType. Functions that
        // haven't opted in (OpId::Unknown) still land in OpType::
        // Unknown, so the fusion node exists for connectivity tracking
        // but matchers correctly skip it.
        std::string op_name = "unknown";
        OpType op_type = OpType::Unknown;
        if (node->function) {
            op_name = node->function->name();
            op_type = fusion_op_type_from_op_id(node->function->op_id());
        }

        size_t fusion_id = fusion_graph.add_node(op_type, op_name, inputs);
        node_map[node.get()] = fusion_id;

        // Add neighbors to queue
        for (const auto& weak_next : node->next_nodes) {
            if (auto next = weak_next.lock()) {
                if (!visited.count(next.get())) {
                    visited.insert(next.get());
                    queue.push(next);
                }
            }
        }
    }

    return fusion_graph;
}

auto execute_fused_op(
    const FusedOp& fused_op,
    const std::vector<Tensor>& inputs,
    const std::unordered_map<std::string, std::string>& attributes
) -> std::vector<Tensor> {

    // Route to appropriate fused operation implementation
    if (fused_op.fused_op_name == "linear_relu") {
        if (inputs.size() < 2) {
            throw std::runtime_error("execute_fused_op: linear_relu requires at least 2 inputs");
        }
        const Tensor* bias = inputs.size() > 2 ? &inputs[2] : nullptr;
        return {fused_linear_relu(inputs[0], inputs[1], bias)};

    } else if (fused_op.fused_op_name == "conv_bn_relu") {
        // Canonical input order (matches OpId::FusedConv2dBnReLU dispatch):
        //   [input, weight, conv_bias, bn_gamma, bn_beta, bn_running_mean, bn_running_var]
        // conv_bias is required (use a zero tensor when conv had no bias).
        if (inputs.size() != 7) {
            throw std::runtime_error(
                "execute_fused_op: conv_bn_relu requires exactly 7 inputs "
                "(input, weight, conv_bias, bn_gamma, bn_beta, "
                "bn_running_mean, bn_running_var), got " +
                std::to_string(inputs.size())
            );
        }

        // Parse stride/padding/eps attributes.  Inference fusion ⇒ training=false,
        // running stats are used directly; momentum is irrelevant but must be passed.
        int64_t stride = 1;
        int64_t padding = 0;
        double eps = 1e-5;

        if (auto it = attributes.find("stride"); it != attributes.end()) {
            stride = std::stoll(it->second);
        }
        if (auto it = attributes.find("padding"); it != attributes.end()) {
            padding = std::stoll(it->second);
        }
        if (auto it = attributes.find("eps"); it != attributes.end()) {
            eps = std::stod(it->second);
        }

        OpAttributes attrs;
        attrs.set(AttrKey::Stride, stride);
        attrs.set(AttrKey::Padding, padding);
        attrs.set(AttrKey::Momentum, 0.1);
        attrs.set(AttrKey::Eps, eps);
        attrs.set(AttrKey::Training, false);

        // Dispatch the single-pass fused kernel: y = ReLU(BN(Conv(x,w,b))).
        return {dispatch(OpId::FusedConv2dBnReLU, inputs, attrs)[0]};

    } else if (fused_op.fused_op_name == "matmul_add") {
        // Expected inputs: [A, B, bias]
        if (inputs.size() < 2) {
            throw std::runtime_error(
                "execute_fused_op: matmul_add requires at least 2 inputs (A, B), got " +
                std::to_string(inputs.size())
            );
        }

        const Tensor* bias = inputs.size() >= 3 ? &inputs[2] : nullptr;

        // Use matmul + add composition
        Tensor matmul_result = matmul(inputs[0], inputs[1]);

        if (bias != nullptr) {
            return {add(matmul_result, *bias)};
        } else {
            return {matmul_result};
        }

    } else if (fused_op.fused_op_name == "elementwise_chain") {
        // Expected inputs: variable length, at least 2 tensors
        // Attributes should contain "op_sequence" describing the chain
        if (inputs.size() < 2) {
            throw std::runtime_error(
                "execute_fused_op: elementwise_chain requires at least 2 inputs, got " +
                std::to_string(inputs.size())
            );
        }

        // Parse operation type from attributes
        int op_type = 0;  // Default: (a + b) * c with relu
        auto op_type_it = attributes.find("op_type");
        if (op_type_it != attributes.end()) {
            op_type = std::stoi(op_type_it->second);
        }

        // For 3-tensor chains, use optimized fusion
        if (inputs.size() == 3) {
            Tensor result;
            switch (op_type) {
                case 0:  // ReLU((a + b) * c)
                    result = mul(add(inputs[0], inputs[1]), inputs[2]);
                    break;
                case 1:  // ReLU((a * b) + c)
                    result = add(mul(inputs[0], inputs[1]), inputs[2]);
                    break;
                default:
                    throw std::runtime_error(
                        "execute_fused_op: elementwise_chain op_type " +
                        std::to_string(op_type) + " not supported"
                    );
            }
            std::vector<Tensor> relu_inputs = {result};
            return {dispatch(OpId::ReLU, relu_inputs)[0]};
        }

        // For longer chains, compose element-wise add sequentially.
        Tensor result = inputs[0];
        for (size_t i = 1; i < inputs.size(); ++i) {
            result = add(result, inputs[i]);
        }

        // Apply final activation if specified
        auto activation_it = attributes.find("activation");
        if (activation_it != attributes.end() && activation_it->second == "relu") {
            std::vector<Tensor> relu_inputs = {result};
            return {dispatch(OpId::ReLU, relu_inputs)[0]};
        }

        return {result};

    } else if (fused_op.fused_op_name == "attention") {
        // Expected inputs: [Q, K, V] or [Q, K, V, mask].
        //
        // Mask convention (matches PyTorch scaled_dot_product_attention(attn_mask=…)):
        //   mask is ADDED to the scaled scores before softmax.  Float mask uses
        //   0 for kept positions and -inf for masked positions.
        if (inputs.size() < 3) {
            throw std::runtime_error(
                "execute_fused_op: attention requires at least 3 inputs (Q, K, V), got " +
                std::to_string(inputs.size())
            );
        }

        // Parse scale from attributes
        float scale = 1.0f;
        auto scale_it = attributes.find("scale");
        if (scale_it != attributes.end()) {
            scale = std::stof(scale_it->second);
        } else {
            // Auto-compute scale as 1/sqrt(d_k)
            int64_t d_k = inputs[0].shape().back();  // Last dimension of Q
            scale = 1.0f / std::sqrt(static_cast<float>(d_k));
        }

        // Q @ K.T
        Tensor scores = matmul(inputs[0], inputs[1].transpose(-1, -2));

        // Scale
        if (scale != 1.0f) {
            auto shape_span = scores.shape();
            std::vector<int64_t> shape_vec(shape_span.begin(), shape_span.end());
            Tensor scale_tensor = full(shape_vec, scale, scores.dtype(), scores.device());
            scores = mul(scores, scale_tensor);
        }

        // Additive mask BEFORE softmax — this is the contract that lets
        // masked positions reach zero probability while preserving
        // normalisation across the kept positions.
        if (inputs.size() >= 4) {
            scores = add(scores, inputs[3]);
        }

        // softmax(scores) along the last (key) dimension
        Variable scores_var(scores);
        Variable attention_weights_var = nn::softmax(scores_var, -1);
        Tensor attention_weights = attention_weights_var.tensor();

        // attention_weights @ V
        Tensor output = matmul(attention_weights, inputs[2]);

        return {output};
    }

    throw std::runtime_error("execute_fused_op: unknown fusion type " + fused_op.fused_op_name);
}

} // namespace ops
} // namespace tenzor
