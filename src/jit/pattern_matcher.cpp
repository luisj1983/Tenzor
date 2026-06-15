/**
 * @file pattern_matcher.cpp
 * @brief Implementation of graph pattern matching for extended fusion
 */

#include "tenzor/jit/pattern_matcher.hpp"
#include <algorithm>
#include <sstream>
#include <unordered_set>

namespace tenzor {
namespace jit {

// ============================================================================
// Utility helpers
// ============================================================================

auto PatternMatcher::has_single_use(const std::shared_ptr<Node>& node) -> bool {
    if (node->outputs().empty()) return false;
    return node->outputs()[0]->uses().size() == 1;
}

auto PatternMatcher::consumes_output(const std::shared_ptr<Node>& producer,
                                     const std::shared_ptr<Node>& consumer) -> bool {
    if (!producer || !consumer) return false;
    std::unordered_set<Value*> producer_outputs;
    for (const auto& out : producer->outputs()) {
        if (out) producer_outputs.insert(out.get());
    }
    for (const auto& inp : consumer->inputs()) {
        if (inp && producer_outputs.count(inp.get())) return true;
    }
    return false;
}

auto PatternMatcher::estimate_elements(
    const std::vector<std::shared_ptr<Value>>& inputs) -> int64_t {
    if (inputs.empty()) return FusionMatch::kUnknownElements;
    const auto& shape = inputs[0]->shape();
    if (shape.empty()) return FusionMatch::kUnknownElements;
    int64_t elements = 1;
    for (auto d : shape) {
        // <= 0 encodes a dynamic/unknown/invalid dim; a real element count
        // cannot be computed, so report it as unknown rather than letting a
        // 0 or negative placeholder propagate into the cost model.
        if (d <= 0) return FusionMatch::kUnknownElements;
        elements *= d;
    }
    return elements;
}

auto PatternMatcher::is_elementwise(OpType op) -> bool {
    switch (op) {
        case OpType::Add: case OpType::Sub: case OpType::Mul: case OpType::Div:
        case OpType::Exp: case OpType::Log: case OpType::Sqrt: case OpType::Pow:
        case OpType::Abs: case OpType::Neg: case OpType::Clamp:
        case OpType::ReLU: case OpType::Sigmoid: case OpType::Tanh: case OpType::GELU:
            return true;
        default:
            return false;
    }
}

auto PatternMatcher::is_reduction(OpType op) -> bool {
    switch (op) {
        case OpType::Sum: case OpType::Mean: case OpType::Max: case OpType::Min:
            return true;
        default:
            return false;
    }
}

auto PatternMatcher::is_activation(OpType op) -> bool {
    switch (op) {
        case OpType::ReLU: case OpType::Sigmoid: case OpType::Tanh: case OpType::GELU:
            return true;
        default:
            return false;
    }
}

auto PatternMatcher::collect_external_inputs(
    const std::vector<std::shared_ptr<Node>>& nodes)
    -> std::vector<std::shared_ptr<Value>> {
    std::unordered_set<Value*> internal_values;
    for (auto& n : nodes) {
        for (auto& out : n->outputs()) {
            internal_values.insert(out.get());
        }
    }

    std::vector<std::shared_ptr<Value>> external;
    std::unordered_set<Value*> seen;
    for (auto& n : nodes) {
        for (auto& inp : n->inputs()) {
            if (inp && internal_values.count(inp.get()) == 0 && seen.insert(inp.get()).second) {
                external.push_back(inp);
            }
        }
    }
    return external;
}

auto PatternMatcher::collect_external_outputs(
    const std::vector<std::shared_ptr<Node>>& nodes)
    -> std::vector<std::shared_ptr<Value>> {
    std::unordered_set<Node*> node_set;
    for (auto& n : nodes) node_set.insert(n.get());

    std::vector<std::shared_ptr<Value>> external;
    for (auto& n : nodes) {
        for (auto& out : n->outputs()) {
            for (auto& use : out->uses()) {
                if (auto user = use.lock()) {
                    if (node_set.count(user.get()) == 0) {
                        external.push_back(out);
                        break;  // Only add once per output
                    }
                }
            }
        }
    }
    return external;
}

// ============================================================================
// FusionMatch signature
// ============================================================================

auto FusionMatch::compute_signature() -> std::string {
    std::ostringstream ss;
    ss << "ext_" << static_cast<int>(kind);
    for (auto& n : nodes) {
        ss << "_" << static_cast<int>(n->op_type());
    }
    if (!inputs.empty()) {
        ss << "_d" << static_cast<int>(inputs[0]->dtype());
    }
    signature = ss.str();
    return signature;
}

// ============================================================================
// Main pattern finder
// ============================================================================

auto PatternMatcher::find_all(const Graph& graph) -> std::vector<FusionMatch> {
    std::vector<FusionMatch> matches;
    std::unordered_set<Node*> used;
    auto& nodes = graph.nodes();

    // Scan in topological order, try most specific patterns first
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (used.count(nodes[i].get())) continue;

        // Try patterns from most specific to least specific
        if (auto m = match_softmax(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_layer_norm(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_rms_norm(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_swiglu(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_rotary_embedding(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_gelu_variant(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_small_mlp(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_gemm_epilogue(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        } else if (auto m = match_reduction_chain(graph, i, used)) {
            for (auto& n : m->nodes) used.insert(n.get());
            matches.push_back(std::move(*m));
        }
    }

    return matches;
}

// ============================================================================
// Softmax pattern: Max -> Sub -> Exp -> Sum -> Div
// Matches the numerically stable softmax decomposition.
// ============================================================================

auto PatternMatcher::match_softmax(const Graph& graph, size_t start_idx,
                                    const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 4 >= nodes.size()) return std::nullopt;

    auto& n0 = nodes[start_idx];
    if (n0->op_type() != OpType::Max || used.count(n0.get())) return std::nullopt;

    // Look for Sub(input, max_result) immediately after. Beyond op-type and
    // single-use, require a real data edge n0->n1 so a positionally-adjacent
    // but data-independent Sub does not spuriously match.
    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Sub || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;
    if (!consumes_output(n0, n1)) return std::nullopt;

    // Exp after Sub
    auto& n2 = nodes[start_idx + 2];
    if (n2->op_type() != OpType::Exp || used.count(n2.get())) return std::nullopt;
    if (!has_single_use(n1)) return std::nullopt;
    if (!consumes_output(n1, n2)) return std::nullopt;

    // Sum after Exp
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Sum || used.count(n3.get())) return std::nullopt;
    if (!has_single_use(n2)) return std::nullopt;
    if (!consumes_output(n2, n3)) return std::nullopt;

    // Div after Sum
    if (start_idx + 4 >= nodes.size()) return std::nullopt;
    auto& n4 = nodes[start_idx + 4];
    if (n4->op_type() != OpType::Div || used.count(n4.get())) return std::nullopt;
    if (!has_single_use(n3)) return std::nullopt;
    // Div consumes both Exp's output (numerator) and Sum's output (denominator);
    // require the n3->n4 (denominator) edge.
    if (!consumes_output(n3, n4)) return std::nullopt;

    FusionMatch match;
    match.kind = FusionKind::Softmax;
    match.nodes = {n0, n1, n2, n3, n4};
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// LayerNorm pattern: Mean -> Sub -> Pow/Mul -> Mean -> Add(eps) -> Sqrt/Rsqrt -> Div/Mul -> Mul(gamma) -> Add(beta)
// We match a simplified version: Mean -> Sub -> (variance ops) -> mul(gamma) -> add(beta)
// ============================================================================

auto PatternMatcher::match_layer_norm(const Graph& graph, size_t start_idx,
                                       const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 5 >= nodes.size()) return std::nullopt;

    auto& n0 = nodes[start_idx];
    if (n0->op_type() != OpType::Mean || used.count(n0.get())) return std::nullopt;

    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Sub || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;

    // After sub(input, mean), look for variance computation:
    // Pow(2) or Mul(self, self) -> Mean -> Add(eps) -> Sqrt -> Div
    // Match loosely: next should be Pow or Mul
    auto& n2 = nodes[start_idx + 2];
    if ((n2->op_type() != OpType::Pow && n2->op_type() != OpType::Mul) ||
        used.count(n2.get())) return std::nullopt;

    // Mean of squared differences
    if (start_idx + 3 >= nodes.size()) return std::nullopt;
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Mean || used.count(n3.get())) return std::nullopt;

    // Look for the remaining chain: could be Add(eps) -> Sqrt -> Div -> Mul(gamma) -> Add(beta)
    // or variations. Collect contiguous matching nodes.
    std::vector<std::shared_ptr<Node>> matched = {n0, n1, n2, n3};
    size_t idx = start_idx + 4;

    // Consume remaining ops that form the normalization pattern
    while (idx < nodes.size() && matched.size() < 10) {
        auto& nk = nodes[idx];
        if (used.count(nk.get())) break;

        auto op = nk->op_type();
        if (op == OpType::Add || op == OpType::Sqrt || op == OpType::Div ||
            op == OpType::Mul || op == OpType::Neg) {
            matched.push_back(nk);
            ++idx;

            // If we've reached a Mul followed by Add that look like gamma/beta affine,
            // and we have at least 6 nodes, consider the pattern complete
            if (matched.size() >= 6 && op == OpType::Add) {
                break;
            }
        } else {
            break;
        }
    }

    // Need at least 6 nodes for a meaningful LayerNorm fusion
    if (matched.size() < 6) return std::nullopt;

    // Verify single-use constraint for intermediate nodes
    for (size_t i = 0; i + 1 < matched.size(); ++i) {
        if (!has_single_use(matched[i])) return std::nullopt;
    }

    FusionMatch match;
    match.kind = FusionKind::LayerNorm;
    match.nodes = std::move(matched);
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// RMSNorm pattern: Pow/Mul(self) -> Mean -> Add(eps) -> Rsqrt/Sqrt -> Mul
// ============================================================================

auto PatternMatcher::match_rms_norm(const Graph& graph, size_t start_idx,
                                     const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 4 >= nodes.size()) return std::nullopt;

    auto& n0 = nodes[start_idx];
    if ((n0->op_type() != OpType::Pow && n0->op_type() != OpType::Mul) ||
        used.count(n0.get())) return std::nullopt;

    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Mean || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;

    // Add(eps) or directly Sqrt/Rsqrt
    size_t next = start_idx + 2;
    std::vector<std::shared_ptr<Node>> matched = {n0, n1};

    if (next < nodes.size() && nodes[next]->op_type() == OpType::Add &&
        !used.count(nodes[next].get())) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Sqrt or Rsqrt (which is Div(1, Sqrt) or Pow(-0.5))
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Sqrt &&
        !used.count(nodes[next].get())) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Div (input / sqrt_result) or Mul (input * rsqrt_result)
    if (next < nodes.size() &&
        (nodes[next]->op_type() == OpType::Div || nodes[next]->op_type() == OpType::Mul) &&
        !used.count(nodes[next].get())) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Optional: Mul(gamma)
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Mul &&
        !used.count(nodes[next].get())) {
        matched.push_back(nodes[next]);
    }

    if (matched.size() < 4) return std::nullopt;

    for (size_t i = 0; i + 1 < matched.size(); ++i) {
        if (!has_single_use(matched[i])) return std::nullopt;
    }

    FusionMatch match;
    match.kind = FusionKind::RMSNorm;
    match.nodes = std::move(matched);
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// GEMM Epilogue: MatMul/Linear + Add(bias) + activation
// ============================================================================

auto PatternMatcher::match_gemm_epilogue(const Graph& graph, size_t start_idx,
                                          const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    auto& n0 = nodes[start_idx];

    if ((n0->op_type() != OpType::MatMul && n0->op_type() != OpType::Linear) ||
        used.count(n0.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;

    std::vector<std::shared_ptr<Node>> matched = {n0};
    size_t next = start_idx + 1;

    // Optional: Add(bias)
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Add &&
        !used.count(nodes[next].get())) {
        matched.push_back(nodes[next]);
        ++next;
        if (matched.size() > 1 && !has_single_use(matched.back())) {
            // Add has multiple uses — can still fuse MatMul+Add but not activation
        }
    }

    // Optional: activation
    if (next < nodes.size() && is_activation(nodes[next]->op_type()) &&
        !used.count(nodes[next].get())) {
        // Only fuse if the previous node has single use
        if (matched.size() >= 1 && has_single_use(matched.back())) {
            matched.push_back(nodes[next]);
        }
    }

    // Need at least MatMul + one more op to justify fusion
    if (matched.size() < 2) return std::nullopt;

    FusionMatch match;
    match.kind = FusionKind::GemmEpilogue;
    match.nodes = std::move(matched);
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// Small MLP: Linear -> activation -> Linear
// ============================================================================

auto PatternMatcher::match_small_mlp(const Graph& graph, size_t start_idx,
                                      const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 2 >= nodes.size()) return std::nullopt;

    auto& n0 = nodes[start_idx];
    if (n0->op_type() != OpType::Linear || used.count(n0.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;

    auto& n1 = nodes[start_idx + 1];
    if (!is_activation(n1->op_type()) || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n1)) return std::nullopt;
    if (!consumes_output(n0, n1)) return std::nullopt;

    auto& n2 = nodes[start_idx + 2];
    if (n2->op_type() != OpType::Linear || used.count(n2.get())) return std::nullopt;
    if (!consumes_output(n1, n2)) return std::nullopt;

    // Check hidden dimension constraint
    if (!n0->outputs().empty()) {
        auto& shape = n0->outputs()[0]->shape();
        if (!shape.empty()) {
            int64_t hidden = shape.back();
            if (hidden > max_mlp_hidden_) return std::nullopt;
        }
    }

    FusionMatch match;
    match.kind = FusionKind::SmallMLP;
    match.nodes = {n0, n1, n2};
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// Reduction chain: element-wise* -> reduction -> element-wise*
// ============================================================================

auto PatternMatcher::match_reduction_chain(const Graph& graph, size_t start_idx,
                                            const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();

    // Start must be an element-wise op followed eventually by a reduction
    auto& n0 = nodes[start_idx];
    if (!is_elementwise(n0->op_type()) || used.count(n0.get())) return std::nullopt;

    std::vector<std::shared_ptr<Node>> matched = {n0};
    size_t idx = start_idx + 1;
    bool found_reduction = false;

    // Collect pre-reduction element-wise ops
    while (idx < nodes.size() && !found_reduction) {
        auto& nk = nodes[idx];
        if (used.count(nk.get())) break;

        if (is_reduction(nk->op_type())) {
            matched.push_back(nk);
            found_reduction = true;
            ++idx;
        } else if (is_elementwise(nk->op_type()) && has_single_use(matched.back())) {
            matched.push_back(nk);
            ++idx;
        } else {
            break;
        }
    }

    if (!found_reduction) return std::nullopt;

    // Collect post-reduction element-wise ops
    while (idx < nodes.size() && matched.size() < 8) {
        auto& nk = nodes[idx];
        if (used.count(nk.get()) || !is_elementwise(nk->op_type())) break;
        if (!has_single_use(matched.back())) break;
        matched.push_back(nk);
        ++idx;
    }

    // Need at least 3 nodes (pre + reduce + post) to justify fusion
    if (matched.size() < 3) return std::nullopt;

    // Verify single-use for all intermediate nodes
    for (size_t i = 0; i + 1 < matched.size(); ++i) {
        if (!has_single_use(matched[i])) return std::nullopt;
    }

    FusionMatch match;
    match.kind = FusionKind::Reduction;
    match.nodes = std::move(matched);
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// SwiGLU pattern: Linear -> Slice -> Sigmoid -> Mul (gate) -> Linear
// The gated linear unit with SiLU activation: gate = sigmoid(x_slice) * x_slice
// then output = Linear(gate). Slice represents the split operation.
// ============================================================================

auto PatternMatcher::match_swiglu(const Graph& graph, size_t start_idx,
                                   const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 4 >= nodes.size()) return std::nullopt;

    auto& n0 = nodes[start_idx];
    if (n0->op_type() != OpType::Linear || used.count(n0.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;

    // Slice (split the linear output into two halves)
    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Slice || used.count(n1.get())) return std::nullopt;
    if (!consumes_output(n0, n1)) return std::nullopt;

    // Sigmoid (the SiLU activation component: silu(x) = x * sigmoid(x))
    auto& n2 = nodes[start_idx + 2];
    if (n2->op_type() != OpType::Sigmoid || used.count(n2.get())) return std::nullopt;
    if (!has_single_use(n1)) return std::nullopt;
    if (!consumes_output(n1, n2)) return std::nullopt;

    // Mul (gating: element-wise product of the two split halves after activation)
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Mul || used.count(n3.get())) return std::nullopt;
    if (!has_single_use(n2)) return std::nullopt;
    if (!consumes_output(n2, n3)) return std::nullopt;

    // Final Linear projection
    auto& n4 = nodes[start_idx + 4];
    if (n4->op_type() != OpType::Linear || used.count(n4.get())) return std::nullopt;
    if (!has_single_use(n3)) return std::nullopt;
    if (!consumes_output(n3, n4)) return std::nullopt;

    FusionMatch match;
    match.kind = FusionKind::SwiGLU;
    match.nodes = {n0, n1, n2, n3, n4};
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// GELU variant pattern (tanh-based approximation):
//   Pow(x, 3) -> Mul(0.044715) -> Add(x) -> Mul(sqrt(2/pi)) -> Tanh -> Add(1) -> Mul(0.5) -> Mul(x)
// Also matches erf-based variant:
//   Mul(x, 0.7071) -> Erf-like chain -> Add(1) -> Mul(0.5) -> Mul(x)
// We match the tanh-based form: starts with Pow, walks through the chain.
// ============================================================================

auto PatternMatcher::match_gelu_variant(const Graph& graph, size_t start_idx,
                                         const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 5 >= nodes.size()) return std::nullopt;

    auto& n0 = nodes[start_idx];
    if (n0->op_type() != OpType::Pow || used.count(n0.get())) return std::nullopt;

    // Mul (scale the cubic term by 0.044715)
    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Mul || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;
    if (!consumes_output(n0, n1)) return std::nullopt;

    // Add (x + 0.044715 * x^3)
    auto& n2 = nodes[start_idx + 2];
    if (n2->op_type() != OpType::Add || used.count(n2.get())) return std::nullopt;
    if (!has_single_use(n1)) return std::nullopt;
    if (!consumes_output(n1, n2)) return std::nullopt;

    // Mul (scale by sqrt(2/pi)) or Tanh directly if pre-scaled
    size_t next = start_idx + 3;
    std::vector<std::shared_ptr<Node>> matched = {n0, n1, n2};

    // Optional: another Mul for the sqrt(2/pi) scaling
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Mul &&
        !used.count(nodes[next].get()) && has_single_use(matched.back())) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Tanh
    if (next >= nodes.size() || nodes[next]->op_type() != OpType::Tanh ||
        used.count(nodes[next].get()) || !has_single_use(matched.back())) {
        return std::nullopt;
    }
    matched.push_back(nodes[next]);
    ++next;

    // Consume remaining Add(1) -> Mul(0.5) -> Mul(x) tail
    while (next < nodes.size() && matched.size() < 9) {
        auto& nk = nodes[next];
        if (used.count(nk.get())) break;

        auto op = nk->op_type();
        if (op == OpType::Add || op == OpType::Mul) {
            if (!has_single_use(matched.back())) break;
            matched.push_back(nk);
            ++next;
        } else {
            break;
        }
    }

    // Need at least 6 nodes: Pow -> Mul -> Add -> Tanh -> Add -> Mul
    if (matched.size() < 6) return std::nullopt;

    // Verify single-use for all intermediate nodes
    for (size_t i = 0; i + 1 < matched.size(); ++i) {
        if (!has_single_use(matched[i])) return std::nullopt;
    }

    FusionMatch match;
    match.kind = FusionKind::GeluVariant;
    match.nodes = std::move(matched);
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

// ============================================================================
// Rotary embedding pattern: cos/sin position rotation applied to Q/K tensors.
// Typical decomposition: Slice -> Mul -> Slice -> Neg -> Mul -> Add
// This applies: out = x_even * cos(theta) + (-x_odd) * sin(theta) for even,
//               out = x_even * sin(theta) + x_odd * cos(theta) for odd.
// We match: Slice -> Mul -> Slice -> Neg -> Mul -> Add
// ============================================================================

auto PatternMatcher::match_rotary_embedding(const Graph& graph, size_t start_idx,
                                             const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();
    if (start_idx + 5 >= nodes.size()) return std::nullopt;

    // Slice (extract even-indexed elements or first half)
    auto& n0 = nodes[start_idx];
    if (n0->op_type() != OpType::Slice || used.count(n0.get())) return std::nullopt;

    // Mul (x_even * cos(theta)) — consumes the even-slice output.
    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Mul || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;
    if (!consumes_output(n0, n1)) return std::nullopt;

    // Slice (extract odd-indexed elements or second half) — a separate branch,
    // not fed by n1, so no n1->n2 data edge is expected.
    auto& n2 = nodes[start_idx + 2];
    if (n2->op_type() != OpType::Slice || used.count(n2.get())) return std::nullopt;

    // Neg (negate for the rotation) — consumes the odd-slice output.
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Neg || used.count(n3.get())) return std::nullopt;
    if (!has_single_use(n2)) return std::nullopt;
    if (!consumes_output(n2, n3)) return std::nullopt;

    // Mul ((-x_odd) * sin(theta)) — consumes the negated odd slice.
    auto& n4 = nodes[start_idx + 4];
    if (n4->op_type() != OpType::Mul || used.count(n4.get())) return std::nullopt;
    if (!has_single_use(n3)) return std::nullopt;
    if (!consumes_output(n3, n4)) return std::nullopt;

    // Add (combine the two rotated components) — consumes n4's output.
    auto& n5 = nodes[start_idx + 5];
    if (n5->op_type() != OpType::Add || used.count(n5.get())) return std::nullopt;
    if (!has_single_use(n4)) return std::nullopt;
    if (!consumes_output(n4, n5)) return std::nullopt;

    // Optionally consume a second rotation half (odd output):
    // Slice -> Mul -> Slice -> Mul -> Add
    std::vector<std::shared_ptr<Node>> matched = {n0, n1, n2, n3, n4, n5};
    size_t next = start_idx + 6;

    // Try to match the second rotation component (for the other half)
    if (next + 4 < nodes.size()) {
        auto& m0 = nodes[next];
        auto& m1 = nodes[next + 1];
        auto& m2 = nodes[next + 2];
        auto& m3 = nodes[next + 3];
        auto& m4 = nodes[next + 4];

        if (m0->op_type() == OpType::Slice && !used.count(m0.get()) &&
            m1->op_type() == OpType::Mul && !used.count(m1.get()) &&
            m2->op_type() == OpType::Slice && !used.count(m2.get()) &&
            m3->op_type() == OpType::Mul && !used.count(m3.get()) &&
            m4->op_type() == OpType::Add && !used.count(m4.get()) &&
            has_single_use(m0) && has_single_use(m1) &&
            has_single_use(m2) && has_single_use(m3)) {
            matched.push_back(m0);
            matched.push_back(m1);
            matched.push_back(m2);
            matched.push_back(m3);
            matched.push_back(m4);
        }
    }

    // Verify single-use for all intermediate nodes
    for (size_t i = 0; i + 1 < matched.size(); ++i) {
        if (!has_single_use(matched[i])) return std::nullopt;
    }

    FusionMatch match;
    match.kind = FusionKind::RotaryEmbedding;
    match.nodes = std::move(matched);
    match.inputs = collect_external_inputs(match.nodes);
    match.outputs = collect_external_outputs(match.nodes);
    match.estimated_elements = estimate_elements(match.inputs);
    match.compute_signature();
    return match;
}

} // namespace jit
} // namespace tenzor
