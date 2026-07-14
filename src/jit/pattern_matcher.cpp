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
        // Guard the multiply against signed-overflow UB (JIT-020): a wrapped
        // negative product would flow into the cost model. On overflow, report
        // the count as unknown (the cost model already treats that safely).
        if (__builtin_mul_overflow(elements, d, &elements)) {
            return FusionMatch::kUnknownElements;
        }
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
        // matched here — those FusionKind values were removed entirely (R1-10,
        // 2026-07-12; previously they existed but nothing ever matched or
        // generated them). The extended-fusion executor (execute_extended_fused /
        // ExtendedKernelCodegen::generate) never had a generator or launch
        // geometry for them — the ExtendedFusionGroup ABI carries none of the
        // required structure (slice offsets, the second Linear's weights, or
        // the cos/sin rotation tensors). A matched-but-ungeneratable kind would
        // have been a hard runtime throw ("unsupported fusion kind") once
        // codegen is the real execution path, and the SwiGLU cost model even
        // actively selected it. Their constituent ops run via normal, correct
        // op dispatch instead. Element-wise GELU is still fused by the
        // separate element-wise codegen path.
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
    // Normalize negative dims to [0,rank) before comparing so a graph carrying
    // dim=-1 on Max and dim=rank-1 on Sum (the same axis) is recognized as a
    // match instead of being rejected — mirrors match_layer_norm.
    int64_t sm_rank = static_cast<int64_t>(
        n0->inputs().empty() ? 0 : n0->inputs()[0]->shape().size());
    auto reduce_dim = [sm_rank](const std::shared_ptr<Node>& n)
            -> std::optional<int64_t> {
        if (!n->has_int_attr("dim")) return std::nullopt;
        int64_t d = n->get_int_attr("dim");
        if (d < 0) d += sm_rank;
        return d;
    };
    auto reduce_keepdim = [](const std::shared_ptr<Node>& n) -> bool {
        return n->has_attr("keepdim") ? n->get_bool_attr("keepdim") : false;
    };
    if (reduce_dim(n0) != reduce_dim(n3) ||
        reduce_keepdim(n0) != reduce_keepdim(n3)) {
        return std::nullopt;
    }

    // The fused Softmax kernel only normalizes over the LAST dimension:
    // execute_extended_fused throws for any other axis and execute_fused_gpu_node
    // has NO eager fallback, so fusing a non-last-axis softmax would hard-crash at
    // runtime on CUDA/ROCm (JIT-F034). Only fuse when the reduction axis is
    // provably the last dim; otherwise leave the softmax unfused so it runs
    // correctly over its true axis. An absent `dim` cannot be confirmed as last,
    // so it is conservatively left unfused.
    {
        auto sm_dim = reduce_dim(n0);
        if (!sm_dim.has_value() || *sm_dim != sm_rank - 1) return std::nullopt;
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
    // If the square step is a Mul, it must be a genuine self-square (x*x): the
    // fused LayerNorm kernel reads a single input stream and computes x*x. A
    // Mul(a,b) with distinct operands would silently square only `a`, drop `b`,
    // and mis-collect `b` as gamma. Reject so the correct eager path runs.
    if (n2->op_type() == OpType::Mul &&
        !(n2->inputs().size() == 2 &&
          n2->inputs()[0].get() == n2->inputs()[1].get())) {
        return std::nullopt;
    }
    // A Pow square step must have exponent EXACTLY 2 (JIT-004). The Mul case is
    // guarded above; the Pow case previously accepted any exponent, so a
    // norm-shaped graph using Pow(x-mean, 3) (or 0.5) would be replaced by the
    // hardcoded-squaring fused kernel and silently compute the wrong variance.
    // The traced Pow carries its exponent as a float "exponent" attr; if that is
    // absent (e.g. an earlier pass moved the exponent to a Constant input) we
    // cannot confirm it here, so reject and let the correct eager path run.
    if (n2->op_type() == OpType::Pow) {
        float e = n2->get_attr("exponent");
        if (!n2->has_float_attr("exponent") || e < 1.999f || e > 2.001f) {
            return std::nullopt;
        }
    }

    // Mean of squared differences
    if (start_idx + 3 >= nodes.size()) return std::nullopt;
    auto& n3 = nodes[start_idx + 3];
    if (n3->op_type() != OpType::Mean || used.count(n3.get())) return std::nullopt;
    if (!consumes_output(n2, n3)) return std::nullopt;

    // Both Means must reduce the SAME axis set, and the fused LayerNorm kernel
    // folds every dim from the first normalized axis to the end into one
    // contiguous instance length (norm_size = suffix_prod(axis)). That is only
    // correct when the normalized axes are a CONTIGUOUS BLOCK ENDING AT THE LAST
    // dim. Read the axes from whichever key the tracer emitted — "dim" (int) for
    // a single axis, "dims" (vec) for multi-axis. The guard previously read only
    // "dim" and silently passed nullopt==nullopt for a multi-axis "dims" list,
    // defeating the same-axis check (JIT-F010). A non-last / non-contiguous /
    // unknown axis set would read the wrong (strided) elements — silently wrong
    // on GPU (JIT-F035) — so reject and let the eager path run.
    // norm_size = the fused LayerNorm kernel's per-instance element count
    // (product of the normalized trailing axes). Computed here (alongside the
    // axis-contiguity check that already derives it) so the gamma/beta
    // absorption loop below can validate an affine operand's shape against it
    // (JIT-R111 -- mirrors match_gemm_epilogue's is_per_column_bias guard).
    int64_t norm_size = 0;
    {
        const int64_t rank = static_cast<int64_t>(
            n0->inputs().empty() ? 0 : n0->inputs()[0]->shape().size());
        auto reduction_axes = [rank](const std::shared_ptr<Node>& n)
                -> std::optional<std::vector<int64_t>> {
            std::vector<int64_t> axes;
            if (n->has_int_attr("dim")) {
                axes.push_back(n->get_int_attr("dim"));
            } else if (n->has_vec_attr("dims")) {
                axes = n->get_vec_attr("dims");
            } else if (n->has_vec_attr("dim")) {
                axes = n->get_vec_attr("dim");
            } else {
                return std::nullopt;
            }
            for (auto& a : axes) if (a < 0) a += rank;
            std::sort(axes.begin(), axes.end());
            axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
            return axes;
        };
        auto a0 = reduction_axes(n0);
        auto a3 = reduction_axes(n3);
        if (!a0.has_value() || !a3.has_value() || *a0 != *a3) return std::nullopt;
        const auto& ax = *a0;
        const int64_t last = rank - 1;
        const bool trailing_contiguous =
            !ax.empty() && ax.back() == last &&
            ax.front() == last - static_cast<int64_t>(ax.size()) + 1;
        if (!trailing_contiguous) return std::nullopt;
        const auto& in_shape = n0->inputs()[0]->shape();
        int64_t prod = 1;
        bool overflowed = false;
        for (auto a : ax) {
            if (a < 0 || a >= static_cast<int64_t>(in_shape.size())) {
                overflowed = true;
                break;
            }
            // JIT-R113: <= 0 encodes a dynamic/unknown dim (this file's
            // established sentinel, see estimate_elements() above) -- a real
            // norm_size cannot be computed from it. Letting it multiply
            // through silently produced a 0/negative norm_size that could
            // spuriously equal an unrelated candidate operand's own
            // sentinel-tainted numel below, passing the affine-shape guard
            // for an operand that doesn't actually match the real
            // (still-unresolved) normalized-axes structure.
            if (in_shape[static_cast<size_t>(a)] <= 0 ||
                __builtin_mul_overflow(prod, in_shape[static_cast<size_t>(a)], &prod)) {
                overflowed = true;
                break;
            }
        }
        if (overflowed) return std::nullopt;
        norm_size = prod;
    }

    // Look for the remaining chain: could be Add(eps) -> Sqrt -> Div -> Mul(gamma) -> Add(beta)
    // or variations. Collect contiguous matching nodes.
    std::vector<std::shared_ptr<Node>> matched = {n0, n1, n2, n3};
    size_t idx = start_idx + 4;

    // JIT-R111: an Add/Mul absorbed into the tail chain below may be the
    // gamma/beta affine step, whose OTHER operand (not the running chain
    // value) becomes an external input to the fused kernel -- indexed as
    // gamma[i]/beta[i] for i in [0, norm_size) by extended_codegen's
    // generate_layer_norm. Absorbing purely on op-type + data-edge
    // connectivity (as this loop otherwise does) let a full-tensor operand
    // (e.g. a residual Add after a bias-less LayerNorm, or a gating Mul) be
    // silently misread as a per-channel affine parameter and truncated to
    // its first norm_size elements, broadcast to every outer instance --
    // wrong output, no error, on the CUDA/ROCm fused path. Mirrors match_
    // gemm_epilogue's is_per_column_bias guard: require the other operand to
    // be either scalar (numel==1 -- covers eps-style constants) or exactly
    // norm_size elements; anything else stops absorption here instead of
    // silently including it.
    // findings.txt JIT-R117: the numel-only check above was NOT actually
    // equivalent to is_per_column_bias -- it accepted ANY shape whose total
    // element count happened to equal norm_size, regardless of which axis
    // carried that count (e.g. a [norm_size, 1] or [N, 1] with N==norm_size
    // per-row/per-token operand, an entirely different broadcast semantics
    // than a genuine [norm_size] gamma or, for multi-axis normalization, a
    // genuine [d1, ..., dK] gamma matching the K trailing normalized axes
    // exactly -- e.g. LayerNorm(normalized_shape=(H,W)) needs a [H, W]
    // gamma, not just any [*, *] shape whose product is H*W). Determine the
    // genuine "normalized axes" structure from the CHAIN's own (un-reduced)
    // shape -- not from the candidate operand -- then require the operand,
    // after stripping any leading size-1 dims, to match that structure
    // element-wise (or be a true scalar).
    auto other_operand_is_valid_affine_shape =
        [&](const std::shared_ptr<Node>& chain_tail,
            const std::shared_ptr<Node>& nk) -> bool {
        std::unordered_set<Value*> chain_outputs;
        for (const auto& out : chain_tail->outputs()) {
            if (out) chain_outputs.insert(out.get());
        }
        const Value* other = nullptr;
        for (const auto& inp : nk->inputs()) {
            if (inp && !chain_outputs.count(inp.get())) { other = inp.get(); break; }
        }
        if (!other) return true;  // no external operand to validate
        const auto& oshape = other->shape();
        int64_t onumel = 1;
        for (auto d : oshape) {
            // JIT-R113: <= 0 encodes a dynamic/unknown dim (this file's
            // established sentinel) -- cannot validate this operand's shape
            // against norm_size without a real count, so reject absorption
            // rather than let a 0/negative placeholder multiply through and
            // possibly coincide with norm_size's own value.
            if (d <= 0 || __builtin_mul_overflow(onumel, d, &onumel)) return false;
        }
        if (onumel == 1) return true;
        if (onumel != norm_size) return false;
        if (chain_tail->outputs().empty() || !chain_tail->outputs()[0]) return false;
        const auto& chain_shape = chain_tail->outputs()[0]->shape();
        // Find the smallest trailing run of chain_shape whose product is
        // norm_size -- that run IS the genuine normalized-axes structure
        // (K dims) a real gamma/beta must match.
        int64_t suffix_prod = 1;
        size_t k = 0;
        bool found = false;
        for (size_t i = chain_shape.size(); i-- > 0; ) {
            if (__builtin_mul_overflow(suffix_prod, chain_shape[i], &suffix_prod)) return false;
            ++k;
            if (suffix_prod == norm_size) { found = true; break; }
            if (suffix_prod > norm_size) break;
        }
        if (!found) return false;
        // Strip oshape's own leading size-1 dims, then require exactly K
        // dims remain, matching chain_shape's trailing K dims element-wise.
        size_t lead = 0;
        while (lead < oshape.size() && oshape[lead] == 1) ++lead;
        if (oshape.size() - lead != k) return false;
        for (size_t i = 0; i < k; ++i) {
            if (oshape[lead + i] != chain_shape[chain_shape.size() - k + i]) return false;
        }
        return true;
    };

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
            if ((op == OpType::Add || op == OpType::Mul) &&
                !other_operand_is_valid_affine_shape(matched.back(), nk)) {
                break;
            }
            // JIT-R114: the normalizing Div ((x-mean)/std) was absorbed with
            // NO validation at all -- unlike Add/Mul's gamma/beta guard
            // above. other_operand_is_valid_affine_shape isn't the right
            // check here (Div's non-Sqrt operand is x-mean from several
            // steps earlier in the chain, not chain_tail's own output, and
            // has the full batched shape, not norm_size -- reusing that
            // guard would reject every genuine LayerNorm). The actual
            // invariant a real LayerNorm's Div must satisfy: EVERY operand
            // is either already part of this matched chain, or is the
            // pattern's own original input (x, feeding n0) -- never a
            // genuinely external tensor. An unrelated Div pulling in a
            // foreign operand (e.g. an adjacent unrelated scaling division
            // that happens to consume the running std value) would
            // otherwise be silently absorbed and fed through the fused
            // kernel as if it were part of the LayerNorm math.
            if (op == OpType::Div) {
                std::unordered_set<Value*> chain_value_set;
                for (const auto& m : matched) {
                    for (const auto& out : m->outputs()) {
                        if (out) chain_value_set.insert(out.get());
                    }
                }
                Value* original_input =
                    n0->inputs().empty() ? nullptr : n0->inputs()[0].get();
                bool all_internal = true;
                for (const auto& inp : nk->inputs()) {
                    if (!inp) continue;
                    if (chain_value_set.count(inp.get())) continue;
                    if (original_input && inp.get() == original_input) continue;
                    all_internal = false;
                    break;
                }
                if (!all_internal) break;
            }
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

    // A real LayerNorm DIVIDES the centered input by the standard deviation:
    // (x - mean) / sqrt(var + eps). A plain variance/std subgraph
    // (Mean,Sub,Pow,Mean,Add,Sqrt) has the same 6-node op-type prefix but NO such
    // division — it is not a LayerNorm. Without this guard extended_codegen would
    // replace that std computation with the hardcoded full-LayerNorm kernel and
    // silently return normalized+affine output instead of a standard deviation
    // (JIT-F009). Require the normalizing Div to be present in the matched chain.
    if (std::none_of(matched.begin(), matched.end(),
                     [](const std::shared_ptr<Node>& n) {
                         return n->op_type() == OpType::Div;
                     })) {
        return std::nullopt;
    }
    // JIT-R166: the Div-presence check above only guards against a MISSING
    // division; it does not verify the divisor is actually std = sqrt(var+eps).
    // A chain dividing by (var+eps) directly -- i.e. Mean,Sub,Pow,Mean,Add,Div,
    // with the Sqrt/Rsqrt step simply absent -- also satisfies "matched.size()
    // >= 6" and "a Div is present", and would be silently misabsorbed into
    // extended_codegen's hardcoded sqrt-based LayerNorm kernel (generate_layer_
    // norm computes inv_std = 1/sqrt(m2/count + eps) unconditionally). Require a
    // Sqrt or Rsqrt to also be present so the matched chain's actual arithmetic
    // matches what the fused kernel will compute.
    if (std::none_of(matched.begin(), matched.end(),
                     [](const std::shared_ptr<Node>& n) {
                         return n->op_type() == OpType::Sqrt ||
                                n->op_type() == OpType::Rsqrt;
                     })) {
        return std::nullopt;
    }

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
    // A Mul square step must be a genuine self-square (x*x); the fused RMSNorm
    // kernel reads a single input stream. A Mul(a,b) with distinct operands
    // would silently square only `a`, drop `b`, and mis-collect `b` as gamma.
    if (n0->op_type() == OpType::Mul &&
        !(n0->inputs().size() == 2 &&
          n0->inputs()[0].get() == n0->inputs()[1].get())) {
        return std::nullopt;
    }
    // A Pow square step must have exponent EXACTLY 2 (JIT-004): the fused RMSNorm
    // kernel hardcodes squaring, so a Pow(x, 3)/Pow(x, 0.5) would be mis-fused.
    if (n0->op_type() == OpType::Pow) {
        float e = n0->get_attr("exponent");
        if (!n0->has_float_attr("exponent") || e < 1.999f || e > 2.001f) {
            return std::nullopt;
        }
    }

    auto& n1 = nodes[start_idx + 1];
    if (n1->op_type() != OpType::Mean || used.count(n1.get())) return std::nullopt;
    if (!has_single_use(n0)) return std::nullopt;
    // Require a real data edge n0->n1 (mirrors match_softmax/match_swiglu);
    // extended_codegen replaces an RMSNorm match with hardcoded RMSNorm math,
    // so a spurious match of an unrelated Pow/Mean chain is silently wrong.
    if (!consumes_output(n0, n1)) return std::nullopt;

    // The fused RMSNorm kernel folds every dim from the normalized axis to the
    // end into one instance length (norm_size = suffix_prod(axis)), correct only
    // for a CONTIGUOUS TRAILING axis block ending at the last dim. Read the Mean's
    // axes from "dim" (int) or "dims"/"dim" (vec) — the plural key is what the
    // tracer emits for multi-axis — and reject a non-last / non-contiguous /
    // unknown axis set so the eager path runs instead of the mis-indexed kernel
    // (JIT-F035).
    // norm_size = the fused RMSNorm kernel's per-instance element count
    // (product of the normalized trailing axes) -- computed here so the
    // eps/gamma absorption below can validate an operand's shape against it
    // (JIT-R111 -- mirrors match_gemm_epilogue's is_per_column_bias guard).
    int64_t norm_size = 0;
    {
        const int64_t rank = static_cast<int64_t>(
            n0->inputs().empty() ? 0 : n0->inputs()[0]->shape().size());
        std::vector<int64_t> axes;
        if (n1->has_int_attr("dim")) {
            axes.push_back(n1->get_int_attr("dim"));
        } else if (n1->has_vec_attr("dims")) {
            axes = n1->get_vec_attr("dims");
        } else if (n1->has_vec_attr("dim")) {
            axes = n1->get_vec_attr("dim");
        } else {
            return std::nullopt;
        }
        for (auto& a : axes) if (a < 0) a += rank;
        std::sort(axes.begin(), axes.end());
        axes.erase(std::unique(axes.begin(), axes.end()), axes.end());
        const int64_t last = rank - 1;
        const bool trailing_contiguous =
            !axes.empty() && axes.back() == last &&
            axes.front() == last - static_cast<int64_t>(axes.size()) + 1;
        if (!trailing_contiguous) return std::nullopt;
        const auto& in_shape = n0->inputs()[0]->shape();
        int64_t prod = 1;
        bool overflowed = false;
        for (auto a : axes) {
            if (a < 0 || a >= static_cast<int64_t>(in_shape.size())) {
                overflowed = true;
                break;
            }
            // JIT-R113: see match_layer_norm's identical guard for the full
            // rationale -- <= 0 encodes a dynamic/unknown dim and must not
            // silently multiply through into norm_size.
            if (in_shape[static_cast<size_t>(a)] <= 0 ||
                __builtin_mul_overflow(prod, in_shape[static_cast<size_t>(a)], &prod)) {
                overflowed = true;
                break;
            }
        }
        if (overflowed) return std::nullopt;
        norm_size = prod;
    }

    // JIT-R111: see match_layer_norm's identical guard for the full
    // rationale -- an absorbed Add/Mul's non-chain operand must be either
    // scalar (eps-style) or exactly norm_size elements (a genuine per-
    // channel gamma), not a full-tensor residual/gate silently truncated to
    // its first norm_size elements by generate_rms_norm.
    // findings.txt JIT-R117: see match_layer_norm's identical fix for the
    // full rationale -- the numel-only check accepted a [norm_size, 1] or
    // [N, 1] (N==norm_size) per-row/per-token operand as if it were a
    // genuine [norm_size] gamma, and a naive "trailing dim == norm_size"
    // check would in turn wrongly reject a genuine multi-axis [d1,...,dK]
    // gamma (e.g. [H, W] for normalized_shape=(H,W)). Determine the genuine
    // normalized-axes structure from the CHAIN's own shape and require the
    // operand (leading 1s stripped) to match it element-wise.
    auto other_operand_is_valid_affine_shape =
        [&](const std::shared_ptr<Node>& chain_tail,
            const std::shared_ptr<Node>& nk) -> bool {
        std::unordered_set<Value*> chain_outputs;
        for (const auto& out : chain_tail->outputs()) {
            if (out) chain_outputs.insert(out.get());
        }
        const Value* other = nullptr;
        for (const auto& inp : nk->inputs()) {
            if (inp && !chain_outputs.count(inp.get())) { other = inp.get(); break; }
        }
        if (!other) return true;
        const auto& oshape = other->shape();
        int64_t onumel = 1;
        for (auto d : oshape) {
            // JIT-R113: <= 0 encodes a dynamic/unknown dim (this file's
            // established sentinel) -- cannot validate this operand's shape
            // against norm_size without a real count, so reject absorption
            // rather than let a 0/negative placeholder multiply through and
            // possibly coincide with norm_size's own value.
            if (d <= 0 || __builtin_mul_overflow(onumel, d, &onumel)) return false;
        }
        if (onumel == 1) return true;
        if (onumel != norm_size) return false;
        if (chain_tail->outputs().empty() || !chain_tail->outputs()[0]) return false;
        const auto& chain_shape = chain_tail->outputs()[0]->shape();
        int64_t suffix_prod = 1;
        size_t k = 0;
        bool found = false;
        for (size_t i = chain_shape.size(); i-- > 0; ) {
            if (__builtin_mul_overflow(suffix_prod, chain_shape[i], &suffix_prod)) return false;
            ++k;
            if (suffix_prod == norm_size) { found = true; break; }
            if (suffix_prod > norm_size) break;
        }
        if (!found) return false;
        size_t lead = 0;
        while (lead < oshape.size() && oshape[lead] == 1) ++lead;
        if (oshape.size() - lead != k) return false;
        for (size_t i = 0; i < k; ++i) {
            if (oshape[lead + i] != chain_shape[chain_shape.size() - k + i]) return false;
        }
        return true;
    };

    // Add(eps) or directly Sqrt/Rsqrt
    size_t next = start_idx + 2;
    std::vector<std::shared_ptr<Node>> matched = {n0, n1};

    if (next < nodes.size() && nodes[next]->op_type() == OpType::Add &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next]) &&
        other_operand_is_valid_affine_shape(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Sqrt, or the literal Rsqrt op (JIT-R166: previously only Sqrt was
    // recognized here despite the comment above claiming both are matched;
    // a genuine RMSNorm traced via the non-decomposed Rsqrt op instead of
    // Div(1,Sqrt)/Pow(-0.5) was never absorbed, missing an otherwise-valid
    // fusion opportunity).
    if (next < nodes.size() &&
        (nodes[next]->op_type() == OpType::Sqrt || nodes[next]->op_type() == OpType::Rsqrt) &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
        ++next;
    }

    // Div (input / sqrt_result) or Mul (input * rsqrt_result) -- the
    // normalizing step itself, distinct from the LATER "Optional: Mul
    // (gamma)" absorption below.
    //
    // JIT-R114: this step previously had NO operand validation at all,
    // unlike the eps-Add and gamma-Mul steps around it. other_operand_is_
    // valid_affine_shape isn't the right check here either (this step's
    // non-chain-tail operand is the ORIGINAL input x, which has the full
    // batched shape, not norm_size -- that guard would reject every genuine
    // RMSNorm). The actual invariant: every operand is either already part
    // of the matched chain, or is the pattern's own original input (x,
    // feeding n0) -- never a genuinely external tensor. Mirrors match_
    // layer_norm's identical Div guard.
    if (next < nodes.size() &&
        (nodes[next]->op_type() == OpType::Div || nodes[next]->op_type() == OpType::Mul) &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        std::unordered_set<Value*> chain_value_set;
        for (const auto& m : matched) {
            for (const auto& out : m->outputs()) {
                if (out) chain_value_set.insert(out.get());
            }
        }
        Value* original_input = n0->inputs().empty() ? nullptr : n0->inputs()[0].get();
        bool all_internal = true;
        for (const auto& inp : nodes[next]->inputs()) {
            if (!inp) continue;
            if (chain_value_set.count(inp.get())) continue;
            if (original_input && inp.get() == original_input) continue;
            all_internal = false;
            break;
        }
        if (all_internal) {
            matched.push_back(nodes[next]);
            ++next;
        }
    }

    // Optional: Mul(gamma)
    if (next < nodes.size() && nodes[next]->op_type() == OpType::Mul &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next]) &&
        other_operand_is_valid_affine_shape(matched.back(), nodes[next])) {
        matched.push_back(nodes[next]);
    }

    if (matched.size() < 4) return std::nullopt;

    // JIT-R143: the eps-Add, Sqrt/Rsqrt, and normalizing Div/Mul steps above
    // are each independently optional -- nothing tied them together, so a
    // chain computing `variance = mean(x*x); variance_eps = variance + eps;
    // y = x * variance_eps` (no square root at all -- a bug, or simply a
    // non-RMS variance-scaling graph) could satisfy matched.size() >= 4 and
    // be silently misabsorbed into extended_codegen's hardcoded generate_
    // rms_norm kernel, which unconditionally computes
    // `x * rsqrt(mean_sq + eps)` regardless of what was actually matched.
    // Require a Sqrt or Rsqrt to actually be present in the matched chain,
    // mirroring match_layer_norm's identical fix.
    if (std::none_of(matched.begin(), matched.end(),
                     [](const std::shared_ptr<Node>& n) {
                         return n->op_type() == OpType::Sqrt ||
                                n->op_type() == OpType::Rsqrt;
                     })) {
        return std::nullopt;
    }

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
    //
    // Additionally require the Add's OTHER operand to be a genuine per-column
    // bias: a rank-1 [cols] vector (or a leading-1 broadcast such as [1, cols]).
    // The epilogue kernel indexes it as bias[idx % cols], so a full-tensor
    // residual (e.g. y = matmul(x, W) + skip with skip [rows, cols]) would be
    // silently broadcast from its first row — numerically wrong on the fused
    // CUDA/ROCm path while eager stays correct. If shapes aren't inferred yet,
    // do NOT fuse; leave the Add for the correct (eager) elementwise path.
    if (!base_linear_has_bias && next < nodes.size() &&
        nodes[next]->op_type() == OpType::Add &&
        !used.count(nodes[next].get()) &&
        consumes_output(matched.back(), nodes[next])) {
        std::unordered_set<Value*> mm_outputs;
        const std::vector<int64_t>* mm_shape = nullptr;
        for (const auto& out : matched.back()->outputs()) {
            if (out) {
                mm_outputs.insert(out.get());
                if (!mm_shape) mm_shape = &out->shape();
            }
        }
        const Value* bias_val = nullptr;
        for (const auto& inp : nodes[next]->inputs()) {
            if (inp && !mm_outputs.count(inp.get())) { bias_val = inp.get(); break; }
        }
        bool is_per_column_bias = false;
        if (bias_val && mm_shape && !mm_shape->empty()) {
            const auto& bshape = bias_val->shape();
            const int64_t cols = mm_shape->back();
            if (!bshape.empty() && cols > 0 && bshape.back() == cols) {
                // JIT-R017: guard against signed-overflow UB, same class as
                // estimate_elements' __builtin_mul_overflow guard above
                // (JIT-020) applied to this structurally identical raw
                // dimension-product loop. An overflowed (UB) bnumel could
                // spuriously equal `cols`, misclassifying a full-tensor
                // residual as a per-column bias and fusing it into
                // GemmEpilogue, which indexes bias[idx % cols] — silently
                // broadcasting the residual from its first row only.
                int64_t bnumel = 1;
                bool overflowed = false;
                for (auto d : bshape) {
                    if (__builtin_mul_overflow(bnumel, d, &bnumel)) {
                        overflowed = true;
                        break;
                    }
                }
                // A per-column bias has exactly `cols` elements (all leading
                // dims == 1); a [rows, cols] residual has rows*cols != cols.
                is_per_column_bias = !overflowed && (bnumel == cols);
            }
        }
        if (is_per_column_bias) {
            matched.push_back(nodes[next]);
            ++next;
        }
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

    // Check hidden dimension constraint. Must reject a non-positive `hidden`
    // (this file's sentinel for a dynamic/unknown/invalid dim, see
    // estimate_elements above) the same way match_gemm_epilogue's analogous
    // per-column-bias check does (`cols > 0 && ...`) -- extended_codegen.cpp
    // uses this value directly for CUDA/ROCm dynamic shared-memory sizing
    // (`hidden_dim * dtype_size`, cast to unsigned) and as a kernel loop
    // bound. hidden==0 would silently skip the hidden-layer computation
    // (wrong result, output collapses to just bias2, no error); a negative
    // value cast to unsigned would request a huge shared-mem allocation,
    // failing the kernel launch -- either way, a GPU-only divergence from
    // the correct (never-fused) CPU eager path for a degenerate MLP.
    if (n0->outputs().empty()) return std::nullopt;
    {
        auto& shape = n0->outputs()[0]->shape();
        if (shape.empty()) return std::nullopt;
        int64_t hidden = shape.back();
        if (hidden <= 0 || hidden > max_mlp_hidden_) return std::nullopt;
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

namespace {
// R1-09: extended_codegen's generate_reduction (and its mirror,
// ExtendedFusionPass::to_elem in compiler.cpp) can only lower a pre/post
// element-wise step that is either a genuine UNARY op, or a Mul that is a
// same-operand self-square (x*x) — a Mul/Add/Sub/Div with two DISTINCT
// operands has no second data stream in the fused kernel's single-input
// contract. match_reduction_chain used to accept anything is_elementwise()
// allowed (including arbitrary binary ops), relying entirely on compiler.cpp
// re-deriving and re-checking this same allowlist before committing the
// fusion. That is a correct safety net today (a non-representable match is
// discarded there, not mis-fused), but it means correctness for this matcher
// rests on a second file staying in sync rather than being enforced at the
// point of the match itself. Mirror match_layer_norm's self-square guard here
// so an unrepresentable step is structurally rejected at match time too.
auto is_reduction_representable_elem(const std::shared_ptr<Node>& n) -> bool {
    switch (n->op_type()) {
        case OpType::Exp: case OpType::Log: case OpType::Sqrt:
        case OpType::Abs: case OpType::Neg:
        case OpType::ReLU: case OpType::Sigmoid: case OpType::Tanh:
        case OpType::GELU:
            return true;
        case OpType::Mul:
            return n->inputs().size() == 2 &&
                   n->inputs()[0].get() == n->inputs()[1].get();
        default:
            return false;
    }
}
}  // namespace

auto PatternMatcher::match_reduction_chain(const Graph& graph, size_t start_idx,
                                            const std::unordered_set<Node*>& used)
    -> std::optional<FusionMatch> {
    auto& nodes = graph.nodes();

    // Start must be an element-wise op followed eventually by a reduction
    auto& n0 = nodes[start_idx];
    if (!is_reduction_representable_elem(n0) || used.count(n0.get())) return std::nullopt;

    std::vector<std::shared_ptr<Node>> matched = {n0};
    size_t idx = start_idx + 1;
    bool found_reduction = false;

    // Collect pre-reduction element-wise ops. Each appended node must actually
    // consume the previously matched node's output: matching purely on
    // positional adjacency + single-use (as this did before) let a run of
    // data-INDEPENDENT single-use ops fuse into one Reduction match, which
    // execute_extended_fused evaluates over a single input and collapses
    // several independent graph outputs into one (silently wrong / dropped
    // subgraph). Mirror match_layer_norm/match_softmax and require a real data
    // edge.
    while (idx < nodes.size() && !found_reduction) {
        auto& nk = nodes[idx];
        if (used.count(nk.get())) break;
        if (!consumes_output(matched.back(), nk)) break;

        if (is_reduction(nk->op_type())) {
            // The fused kernel reduces exactly ONE explicit axis (mirrors
            // compiler.cpp's ExtendedFusionPass::run has_int_attr("dim")
            // check) -- a full reduction (no "dim") or multi-axis ("dims")
            // would silently reduce only the last axis if matched here.
            if (!nk->has_int_attr("dim")) break;
            matched.push_back(nk);
            found_reduction = true;
            ++idx;
        } else if (is_reduction_representable_elem(nk) && has_single_use(matched.back())) {
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
        if (used.count(nk.get()) || !is_reduction_representable_elem(nk)) break;
        if (!has_single_use(matched.back())) break;
        if (!consumes_output(matched.back(), nk)) break;
        matched.push_back(nk);
        ++idx;
    }

    // Need at least 3 nodes (pre + reduce + post) to justify fusion
    if (matched.size() < 3) return std::nullopt;

    // Verify single-use for all intermediate nodes
    for (size_t i = 0; i + 1 < matched.size(); ++i) {
        if (!has_single_use(matched[i])) return std::nullopt;
    }

    // No intermediate output may escape the matched group: an external consumer
    // would lose its input when the group is replaced by the single fused
    // reduction kernel. Mirrors the group check in match_layer_norm.
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
