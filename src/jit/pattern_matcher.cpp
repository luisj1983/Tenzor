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
        // NOTE: SwiGLU / RotaryEmbedding / GeluVariant are intentionally NOT
        // matched here. The extended-fusion executor (execute_extended_fused /
        // ExtendedKernelCodegen::generate) has no generator or launch geometry
        // for those FusionKinds — the ExtendedFusionGroup ABI carries none of
        // the required structure (slice offsets, the second Linear's weights, or
        // the cos/sin rotation tensors). A matched-but-ungeneratable kind is a
        // hard runtime throw ("unsupported fusion kind") once codegen is the real
        // execution path, and the SwiGLU cost model even actively selected it.
        // Removing them from the matcher (and the cost model) makes their
        // constituent ops run via normal, correct op dispatch instead. Element-
        // wise GELU is still fused by the separate element-wise codegen path.
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

    // Sum after Exp. Do NOT require Exp to be single-use here: a numerically
    // stable softmax reuses exp(x-max) for BOTH the denominator (Sum) and the
    // numerator (Div), so Exp legitimately has two consumers, both inside the
    // pattern. Requiring single-use rejected every real softmax and left this
    // fusion dead. The "no escaping use" check below enforces the real
    // constraint (no consumer outside the matched group).
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Sum || used.count(n3.get())) return std::nullopt;
    if (!consumes_output(n2, n3)) return std::nullopt;

    // Div after Sum
    if (start_idx + 4 >= nodes.size()) return std::nullopt;
    auto& n4 = nodes[start_idx + 4];
    if (n4->op_type() != OpType::Div || used.count(n4.get())) return std::nullopt;
    if (!has_single_use(n3)) return std::nullopt;
    // Div consumes both Exp's output (numerator) and Sum's output (denominator);
    // require BOTH the n3->n4 (denominator) and n2->n4 (numerator) edges so the
    // matched Div is the real softmax normalization, not an unrelated divide.
    if (!consumes_output(n3, n4)) return std::nullopt;
    if (!consumes_output(n2, n4)) return std::nullopt;

    // Every intermediate output (Max, Sub, Exp, Sum) must be consumed only within
    // the matched group; an external consumer would make the eager collapse
    // lossy. Exp is allowed its two internal uses (Sum + Div) under this rule.
    {
        std::unordered_set<Node*> grp{n0.get(), n1.get(), n2.get(),
                                      n3.get(), n4.get()};
        for (const auto* n : {&n0, &n1, &n2, &n3}) {
            if ((*n)->outputs().empty()) return std::nullopt;
            for (const auto& u : (*n)->outputs()[0]->uses()) {
                auto up = u.lock();
                if (up && grp.count(up.get()) == 0) return std::nullopt;
            }
        }
    }

    // The fused softmax kernel runs over a SINGLE reduction axis (it copies the
    // Max node's `dim`). A numerically-stable softmax always reduces Max (n0)
    // and Sum (n3) over the same axis, but a hand-built or atypically-decomposed
    // graph could reduce them over different axes and still match purely on
    // op-type/connectivity — fusing it would silently use Max's axis for both,
    // computing a denominator summed over the wrong axis. Require Max and Sum to
    // carry the same reduction axis (and keepdim) before accepting the match.
    // Absent `dim`/`keepdim` attrs mean the op's canonical default, so two nodes
    // both lacking the attr agree.
    auto reduce_dim = [](const std::shared_ptr<Node>& n) -> std::optional<int64_t> {
        return n->has_int_attr("dim")
                   ? std::optional<int64_t>{n->get_int_attr("dim")}
                   : std::nullopt;
    };
    auto reduce_keepdim = [](const std::shared_ptr<Node>& n) -> bool {
        return n->has_attr("keepdim") ? n->get_bool_attr("keepdim") : false;
    };
    if (reduce_dim(n0) != reduce_dim(n3) ||
        reduce_keepdim(n0) != reduce_keepdim(n3)) {
        return std::nullopt;
    }

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
    // Require a real data edge n0->n1 (mirrors match_softmax/match_swiglu) so a
    // positionally-adjacent but data-independent chain does not spuriously
    // match; extended_codegen replaces a LayerNorm match with hardcoded
    // LayerNorm math, so a false match yields silently wrong results.
    if (!consumes_output(n0, n1)) return std::nullopt;

    // After sub(input, mean), look for variance computation:
    // Pow(2) or Mul(self, self) -> Mean -> Add(eps) -> Sqrt -> Div
    // Match loosely: next should be Pow or Mul
    auto& n2 = nodes[start_idx + 2];
    if ((n2->op_type() != OpType::Pow && n2->op_type() != OpType::Mul) ||
        used.count(n2.get())) return std::nullopt;
    if (!consumes_output(n1, n2)) return std::nullopt;

    // Mean of squared differences
    if (start_idx + 3 >= nodes.size()) return std::nullopt;
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Mean || used.count(n3.get())) return std::nullopt;
    if (!consumes_output(n2, n3)) return std::nullopt;

    // Both Means must reduce the SAME axis. The canonical fused LayerNorm kernel
    // normalizes over a single axis; a structurally similar chain whose mean and
    // variance reductions are over DIFFERENT axes is not a LayerNorm and must
    // not be silently replaced by the hardcoded kernel. Normalize negative dims
    // before comparing so dim=-1 and dim=rank-1 count as the same axis.
    {
        int64_t rank = static_cast<int64_t>(
            n0->inputs().empty() ? 0 : n0->inputs()[0]->shape().size());
        auto mean_axis = [rank](const std::shared_ptr<Node>& n)
                -> std::optional<int64_t> {
            std::optional<int64_t> a;
            if (n->has_int_attr("dim")) a = n->get_int_attr("dim");
            else if (n->has_vec_attr("dim")) {
                auto v = n->get_vec_attr("dim");
                if (v.size() == 1) a = v[0];
                else return std::nullopt;  // multi-axis mean: handled below
            }
            if (a && *a < 0) a = *a + rank;
            return a;
        };
        if (mean_axis(n0) != mean_axis(n3)) return std::nullopt;
    }

    // Look for the remaining chain: could be Add(eps) -> Sqrt -> Div -> Mul(gamma) -> Add(beta)
    // or variations. Collect contiguous matching nodes.
    std::vector<std::shared_ptr<Node>> matched = {n0, n1, n2, n3};
    size_t idx = start_idx + 4;

    // Consume remaining ops that form the normalization pattern
    while (idx < nodes.size() && matched.size() < 10) {
        auto& nk = nodes[idx];
        if (used.count(nk.get())) break;

        auto op = nk->op_type();
        // Require a real data edge from the previously matched node into nk so
        // an unrelated tail of arithmetic ops is not absorbed into the match.
        if (op == OpType::Add || op == OpType::Sqrt || op == OpType::Div ||
            op == OpType::Mul || op == OpType::Neg) {
            if (!consumes_output(matched.back(), nk)) break;
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

    // Intermediates may be consumed by MULTIPLE nodes inside the pattern — the
    // canonical LayerNorm reuses (x - mean) for both the variance branch and the
    // final normalization, so Sub has two internal consumers. Requiring strict
    // single-use rejected every real LayerNorm and left this fusion dead. Enforce
    // the real constraint instead: no intermediate output may escape the matched
    // group (an external consumer would make the eager collapse lossy).
    {
        std::unordered_set<Node*> grp;
        for (const auto& n : matched) grp.insert(n.get());
        for (size_t i = 0; i + 1 < matched.size(); ++i) {
            if (matched[i]->outputs().empty()) return std::nullopt;
            for (const auto& u : matched[i]->outputs()[0]->uses()) {
                auto up = u.lock();
                if (up && grp.count(up.get()) == 0) return std::nullopt;
            }
        }
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
    // Require a real data edge n0->n1 (mirrors match_softmax/match_swiglu);
    // extended_codegen replaces an RMSNorm match with hardcoded RMSNorm math,
    // so a spurious match of an unrelated Pow/Mean chain is silently wrong.
    if (!consumes_output(n0, n1)) return std::nullopt;

    // Add(eps) or directly Sqrt/Rsqrt
    size_t next = start_idx + 2;
    std::vector<std::shared_ptr<Node>> matched = {n0, n1};

    if (next < nodes.size() && nodes[next]->op_type() == OpType::Add &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Sqrt or Rsqrt (which is Div(1, Sqrt) or Pow(-0.5))
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Sqrt &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Div (input / sqrt_result) or Mul (input * rsqrt_result)
    if (next < nodes.size() &&
        (nodes[next]->op_type() == OpType::Div || nodes[next]->op_type() == OpType::Mul) &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Optional: Mul(gamma)
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Mul &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
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

    // A Linear op already carries its bias internally (inputs [x, W, b]); the
    // epilogue kernel adds exactly one per-column bias. So only ATTACH a
    // separate Add as the bias when the base is a bias-less MatMul (or a Linear
    // with no bias input). Absorbing an Add after a bias-carrying Linear would
    // either double-count a bias or, for a residual Add (a full-tensor operand,
    // not a [cols] vector), be misapplied as a per-column bias — both wrong.
    const bool base_linear_has_bias =
        n0->op_type() == OpType::Linear && n0->inputs().size() >= 3;

    // Optional: Add(bias). Require a real data-dependency edge (the Add must
    // consume the MatMul/Linear output) — matching purely on op-type and
    // topological position fuses an unrelated Add, dangling its true producer's
    // output. Every sibling matcher uses consumes_output() for this reason.
    if (!base_linear_has_bias && next < nodes.size() &&
        nodes[next]->op_type() == OpType::Add &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Optional: activation — must consume the previous matched node's output and
    // that node must have a single use.
    if (next < nodes.size() && is_activation(nodes[next]->op_type()) &&
        !used.count(nodes[next].get()) &&
        has_single_use(matched.back()) &&
        consumes_output(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
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

} // namespace jit
} // namespace tenzor
