/**
 * @file compiler.cpp
 * @brief Implementation of graph optimization passes
 */

#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/extended_codegen.hpp"
#include "../../include/tenzor/jit/fusion_cost_model.hpp"
#include "../../include/tenzor/ops/math.hpp"
#include "../../include/tenzor/ops/reduction.hpp"
#include "../../include/tenzor/ops/transform.hpp"  // Audit J5-followup: reshape
#include "../../include/tenzor/ops/creation.hpp"
#include "../../include/tenzor/backend/fast_dispatch.hpp"  // is_op_supported
#include <iostream>
#include <algorithm>
#include <unordered_set>
#include <cmath>
#include <cstring>

namespace tenzor {
namespace jit {

// ============================================================================
// Dead Code Elimination Pass
// ============================================================================

auto DeadCodeEliminationPass::run(Graph& graph) -> bool {
    // Reachability is rooted at the graph's declared outputs. A graph that has
    // nodes but no declared outputs gives us no roots, so every node would look
    // unreachable and be erased. That is never the intent (it would gut a
    // subgraph whose outputs were not populated), so leave such graphs alone.
    if (graph.outputs().empty()) return false;

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

    // Recurse into control-flow subgraphs of the surviving nodes. Each subgraph
    // is a self-contained graph (it receives its inputs through forward(), it
    // does not capture outer-graph values directly), so DCE can be applied to it
    // independently, rooted at that subgraph's own outputs.
    for (const auto& node : nodes) {
        if (reachable.find(node.get()) == reachable.end()) continue;
        auto& then_g = node->then_branch();
        auto& else_g = node->else_branch();
        auto& body_g = node->body();
        if (then_g) modified |= run(*then_g);
        if (else_g) modified |= run(*else_g);
        if (body_g) modified |= run(*body_g);
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
            // NOTE: subgraph (then_branch/else_branch/body) nodes are NOT marked
            // reachable here. Each subgraph is a self-contained graph and is
            // dead-code-eliminated independently (rooted at its own outputs) by
            // the recursive DeadCodeEliminationPass::run call. Marking every
            // subgraph node reachable would both prevent intra-body DCE and keep
            // otherwise-dead outer producers alive.
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

    // Resolve each replacement target to its transitive root before applying.
    // With three (or more) mutually-equivalent nodes n0,n1,n2, the pairwise loop
    // above leaves a chain like {o1->o0, o2->o1} (pair (1,2) overwrites (0,2)).
    // Applying that directly would redirect o2's consumers to o1, whose producer
    // n1 is then removed as a duplicate — leaving a dangling value reference.
    // Collapsing every target to the canonical root (o0) makes the map
    // {o1->o0, o2->o0}, so the apply order no longer matters. A canonical output
    // (the first node in a bucket) is never itself replaced, so the walk
    // terminates; the `seen` guard is a defensive cycle break.
    {
        auto resolve_root = [&](std::string id) {
            std::unordered_set<std::string> seen;
            auto it = value_replacements.find(id);
            while (it != value_replacements.end() && seen.insert(id).second) {
                id = it->second;
                it = value_replacements.find(id);
            }
            return id;
        };
        for (auto& [old_id, new_id] : value_replacements) {
            (void)old_id;
            new_id = resolve_root(new_id);
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

    // Recurse into subgraphs of If/Loop nodes
    for (const auto& node : graph.nodes()) {
        auto& then_g = node->then_branch();
        auto& else_g = node->else_branch();
        auto& body_g = node->body();
        if (then_g) modified |= run(*then_g);
        if (else_g) modified |= run(*else_g);
        if (body_g) modified |= run(*body_g);
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

namespace {
// Ops whose result depends on hidden state (RNG draws, running statistics,
// etc.) and therefore must NOT be deduplicated even when their inputs and
// attributes are identical. CSE'ing two of these would silently turn two
// independent random draws into one shared draw — the classic JIT
// correctness bug (Phase P0 / JIT correctness fix).
auto is_stateful_op(OpType op) -> bool {
    switch (op) {
        case OpType::Dropout:    // each call draws a fresh Bernoulli mask
        case OpType::BatchNorm2d: // training-mode batch stats update + running-stat read
            return true;
        default:
            return false;
    }
}
}  // namespace

auto CommonSubexpressionEliminationPass::nodes_equivalent(const Node& a, const Node& b) -> bool {
    // Same operation type
    if (a.op_type() != b.op_type()) return false;

    // Stateful ops: never equivalent, even if inputs match. Two independent
    // Dropout draws must produce independent random masks.
    if (is_stateful_op(a.op_type())) return false;

    // Control-flow nodes (If/Loop) carry their semantics in subgraph bodies
    // (then_branch/else_branch/body) which are NOT part of get_all_attrs() and
    // are therefore never compared below. Two If/Loop nodes with identical
    // inputs and scalar attrs but structurally different branch/body graphs are
    // distinct computations and must never be merged. Bail out whenever either
    // node carries any subgraph.
    if (a.then_branch() || a.else_branch() || a.body() ||
        b.then_branch() || b.else_branch() || b.body()) {
        return false;
    }

    // Nodes carrying tensor attributes (e.g. Constant's "value" payload) are
    // never deduplicated: the tensor data is not part of the comparison below,
    // so two such nodes differing only in their tensor payload would otherwise
    // be wrongly merged (every Constant would collapse into the first one).
    if (!std::get<4>(const_cast<Node&>(a).get_all_attrs()).empty() ||
        !std::get<4>(const_cast<Node&>(b).get_all_attrs()).empty()) {
        return false;
    }

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

            if (node->outputs().empty()) continue;
            auto old_output = node->outputs()[0];

            // Keep the folded constant on the device the original output lived
            // on so downstream consumers see the expected device.
            if (result.device() != old_output->device()) {
                result = result.to(old_output->device());
            }

            // Create constant node with the folded result
            auto const_node = graph.create_node(OpType::Constant);
            const_node->set_tensor_attr("value", result);

            std::string const_val_id = const_node->name() + "_out";
            // Label the new Value with the RESULT's actual dtype/shape, not the
            // old output's. An op like Mean/Div can promote int->float (or change
            // rank), so stamping the stale label would desync the Value from its
            // real payload and mis-size downstream buffers in every backend.
            std::vector<int64_t> result_shape(result.shape().begin(),
                                               result.shape().end());
            auto const_val = graph.create_value(
                const_val_id, result_shape, result.dtype(),
                old_output->device());
            const_val->set_node(const_node);
            const_node->add_output(const_val);

            // Add the constant node to the graph
            graph.add_node(const_node);

            // Redirect all consumers of the original node to use the constant
            graph.replace_value(old_output->id(), const_val_id);

            // Remove the original node
            graph.remove_node(node);

            modified = true;
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

    // Audit J5 + J5-followup: extended op coverage for ConstantFolding.
    // Phase 1 (J5): Add/Sub/Mul/Div/Exp/Log/Sqrt/Neg/Abs/MatMul/Bmm.
    // Phase 2 (J5-followup): Sum/Mean/Max/Min reductions (with dim+keepdim
    // attrs), Pow (via tenzor::float_power Tensor,Tensor overload), Reshape
    // (when the shape input is itself a constant).
    // Verify arity before declaring foldable. can_fold's loop above only
    // checks that *existing* inputs are constants; it does not verify the node
    // has the expected number of inputs. evaluate_constant unconditionally
    // accesses inputs[1] for binary ops, so a malformed 1-input Add (etc.)
    // would otherwise index a 1-element vector out of bounds (operator[] UB,
    // not contained by the surrounding try/catch).
    const size_t arity = node.inputs().size();
    switch (node.op_type()) {
        // Binary ops: require at least 2 inputs (read inputs[0] and inputs[1]).
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::MatMul:
        case OpType::Bmm:
        case OpType::Pow:
        case OpType::Reshape:
            return arity >= 2;
        // Unary / reduction ops: require at least 1 input (read inputs[0]).
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
        case OpType::Neg:
        case OpType::Abs:
        case OpType::Sum:
        case OpType::Mean:
        case OpType::Max:
        case OpType::Min:
            return arity >= 1;
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

    // Audit J5-followup: reduction helper reads `keepdim` (default false).
    auto reduce_keepdim = [&node]() -> bool {
        return node.has_attr("keepdim") ? node.get_bool_attr("keepdim") : false;
    };

    // Normalize a multi-axis "dims" reduction list against the input rank:
    // convert negatives, drop out-of-range entries, sort ascending, dedup.
    // Mirrors normalize_reduce_dims() in graph.cpp so constant-folding agrees
    // with the native graph executor and symbolic shape inference.
    auto normalize_dims = [](const std::vector<int64_t>& dims, int64_t rank)
            -> std::vector<int64_t> {
        std::vector<int64_t> norm;
        norm.reserve(dims.size());
        for (auto d : dims) {
            if (d < 0) d += rank;
            if (d >= 0 && d < rank) norm.push_back(d);
        }
        std::sort(norm.begin(), norm.end());
        norm.erase(std::unique(norm.begin(), norm.end()), norm.end());
        return norm;
    };

    // Evaluate a reduction node whose axis is either a scalar "dim", a vector
    // "dims" (multi-axis), or absent (full reduction). For the multi-axis case
    // reduce one axis at a time from highest to lowest (non-keepdim) so earlier
    // erasures don't shift the indices still to remove — exactly the sequential
    // strategy the interpreter uses (graph.cpp Sum/Mean/Max/Min cases).
    auto fold_reduction = [&](const Tensor& in,
                              auto reduce_fn) -> Tensor {
        bool keepdim = reduce_keepdim();
        if (node.has_int_attr("dim")) {
            return reduce_fn(in, std::optional<int64_t>{node.get_int_attr("dim")},
                             keepdim);
        }
        if (node.has_vec_attr("dims")) {
            auto norm = normalize_dims(
                node.get_vec_attr("dims"),
                static_cast<int64_t>(in.shape().size()));
            Tensor acc = in;
            for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                acc = reduce_fn(acc, std::optional<int64_t>{*it}, keepdim);
            }
            return acc;
        }
        return reduce_fn(in, std::nullopt, keepdim);
    };

    // Evaluate operation
    switch (node.op_type()) {
        case OpType::Add:    return inputs[0] + inputs[1];
        case OpType::Sub:    return inputs[0] - inputs[1];
        case OpType::Mul:    return inputs[0] * inputs[1];
        case OpType::Div:    return inputs[0] / inputs[1];
        case OpType::Exp:    return tenzor::exp(inputs[0]);
        case OpType::Log:    return tenzor::log(inputs[0]);
        case OpType::Sqrt:   return tenzor::sqrt(inputs[0]);
        // Audit J5: new constant-foldable ops.
        case OpType::Neg:    return tenzor::neg(inputs[0]);
        case OpType::Abs:    return tenzor::abs(inputs[0]);
        case OpType::MatMul: return tenzor::matmul(inputs[0], inputs[1]);
        case OpType::Bmm:    return tenzor::bmm(inputs[0], inputs[1]);
        // Audit J5-followup: reductions with dim/keepdim attrs. Multi-axis
        // "dims" reductions fold one axis at a time (see fold_reduction) so a
        // node like x.sum(dim=[1,2]) over constants yields the correct partial
        // shape/value instead of being collapsed to a full reduction.
        case OpType::Sum:
            return fold_reduction(inputs[0],
                [](const Tensor& t, std::optional<int64_t> d, bool k) {
                    return tenzor::sum(t, d, k); });
        case OpType::Mean:
            return fold_reduction(inputs[0],
                [](const Tensor& t, std::optional<int64_t> d, bool k) {
                    return tenzor::mean(t, d, k); });
        case OpType::Max:
            return fold_reduction(inputs[0],
                [](const Tensor& t, std::optional<int64_t> d, bool k) {
                    return tenzor::max(t, d, k); });
        case OpType::Min:
            return fold_reduction(inputs[0],
                [](const Tensor& t, std::optional<int64_t> d, bool k) {
                    return tenzor::min(t, d, k); });
        // Audit J5-followup: Pow via float_power (Tensor, Tensor overload).
        case OpType::Pow:    return tenzor::float_power(inputs[0], inputs[1]);
        // Audit J5-followup: Reshape — second input is a 1-D Int64 shape tensor.
        case OpType::Reshape: {
            const Tensor& shape_t = inputs[1];
            if (shape_t.dtype() != DType::Int64) {
                throw std::runtime_error(
                    "Reshape constant folding: shape input must be Int64");
            }
            // The shape operand may be device-resident; data<int64_t>() returns a
            // raw host pointer, so read it on the host. contiguous() guards the
            // linear read.
            Tensor shape_host = shape_t.to(Device::cpu()).contiguous();
            const int64_t* p = shape_host.data<int64_t>();
            std::vector<int64_t> shape_vec(p, p + shape_host.numel());
            return tenzor::reshape(inputs[0], std::move(shape_vec));
        }
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
            // Data-flow guard: the BN node must actually consume the conv
            // output (mirrors FuseConvBatchNormReluPass). Without this check a
            // positionally-adjacent but data-independent (Conv, BN) pair would
            // have its conv weights folded with BN stats and BN's real
            // consumers redirected to the unmodified conv output — silently
            // wrong results.
            if (node1->outputs().empty() || node2->inputs().empty() ||
                node1->outputs()[0]->id() != node2->inputs()[0]->id()) {
                can_fuse = false;
            }
            if (can_fuse) {
                auto conv_out = node1->outputs()[0];
                if (conv_out->uses().size() > 1) {
                    can_fuse = false;
                }
            }

            // Phase P0 / JIT correctness fix: Conv+BN fusion is only valid
            // in eval/inference mode. In training, BN uses *batch* mean and
            // variance computed from the live input, not the saved running
            // statistics — folding running_mean/var into the conv weights
            // would silently produce different output than the eager path.
            // Tracer marks training mode via the bool attribute "training";
            // if present and true, refuse to fuse.
            if (can_fuse && node2->get_bool_attr("training")) {
                can_fuse = false;
            }

            if (can_fuse && fuse_pair(node1, node2, graph)) {
                modified = true;
            }
        }
    }

    return modified;
}

namespace {

// Fold BatchNorm affine params into a Conv's weight/bias at compile time.
//   scale = gamma / sqrt(var + eps)
//   w' = w * scale   (per output channel)
//   b' = scale * (b - mean) + beta   (or beta - scale*mean when the conv has no bias)
//
// Precision: eager BatchNorm INFERENCE computes invstd and the normalization in a
// Float32 accumulator for Float16/BFloat16 inputs (see batchnorm.cpp
// "normalize in a float accumulator for half types", and the oneDNN path which
// converts gamma/beta/mean/var to Float32). Folding scale directly in Float16
// rounds `var + eps` to f16 BEFORE the sqrt and does the divide in f16, so the
// folded constants diverge from eager for half-precision models (JIT-076). We
// therefore widen f16/bf16 params to Float32 for the math and narrow the FINAL
// weight/bias back to the conv tensors' original dtypes. For Float32/Float64
// params the widen dtype equals the param dtype, so this is a bit-for-bit no-op
// (no behavioural change on the common path).
void fold_bn_into_conv(const std::shared_ptr<Node>& conv_node,
                       const Tensor& gamma, const Tensor& beta,
                       const Tensor& running_mean, const Tensor& running_var,
                       float eps,
                       const Tensor& conv_weight, const Tensor& conv_bias) {
    const DType pdt = running_var.dtype();
    const DType cdt = (pdt == DType::Float16 || pdt == DType::BFloat16)
                          ? DType::Float32
                          : pdt;
    auto widen = [cdt](const Tensor& t) {
        return t.dtype() == cdt ? t : t.to(cdt);
    };

    const Tensor gamma_c = widen(gamma);
    const Tensor var_c   = widen(running_var);
    // scale = gamma / sqrt(var + eps), all in the (possibly widened) compute dtype.
    const Tensor scale = gamma_c / tenzor::sqrt(var_c + eps);

    // Fuse weights: scale each output channel. conv_weight is [out,in,kH,kW];
    // reshape scale to [out,1,1,1] for broadcasting.
    const DType wdt = conv_weight.dtype();
    const Tensor cw = conv_weight.dtype() == cdt ? conv_weight
                                                 : conv_weight.to(cdt);
    const auto w_shape = cw.shape();
    Tensor fused_weight = (w_shape.size() == 4)
        ? cw * scale.reshape({w_shape[0], 1, 1, 1})
        : cw * scale;  // fallback: direct element-wise
    if (fused_weight.dtype() != wdt) fused_weight = fused_weight.to(wdt);
    conv_node->set_tensor_attr("weight", fused_weight);

    // Fuse bias.
    const Tensor beta_c = widen(beta);
    const Tensor mean_c = widen(running_mean);
    Tensor fused_bias;
    if (conv_bias.numel() > 0) {
        const Tensor cb = conv_bias.dtype() == cdt ? conv_bias
                                                   : conv_bias.to(cdt);
        fused_bias = scale * (cb - mean_c) + beta_c;
    } else {
        fused_bias = beta_c - scale * mean_c;
    }
    // Store the bias in the dtype a downstream add would have used: the conv's
    // own bias dtype when present, otherwise BN's beta dtype.
    const DType bdt = conv_bias.numel() > 0 ? conv_bias.dtype() : beta.dtype();
    if (fused_bias.dtype() != bdt) fused_bias = fused_bias.to(bdt);
    conv_node->set_tensor_attr("bias", fused_bias);
}

}  // namespace

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

    // Fold BN into the conv weight/bias (widens f16/bf16 to match eager — JIT-076).
    fold_bn_into_conv(conv_node, gamma, beta, running_mean, running_var, eps,
                      conv_weight, conv_bias);

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
            // Data-flow guard: the ReLU must consume the conv output and the
            // conv output must have a single use (mirrors FuseMatMulAddPass /
            // FuseConvBatchNormReluPass). Otherwise a positionally-adjacent but
            // data-independent (Conv, ReLU) pair would attach an unwanted ReLU
            // to the conv output and redirect the real ReLU's consumers to the
            // conv output, dropping the genuine ReLU — silently wrong results.
            if (!node1->outputs().empty() && !node2->inputs().empty() &&
                node1->outputs()[0]->id() == node2->inputs()[0]->id() &&
                node1->outputs()[0]->uses().size() == 1) {
                if (fuse_pair(node1, node2, graph)) {
                    modified = true;
                }
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
            // Data-flow guard: the ReLU must consume the linear output and the
            // linear output must have a single use (mirrors FuseMatMulAddPass /
            // FuseConvBatchNormReluPass). Otherwise a positionally-adjacent but
            // data-independent (Linear, ReLU) pair would attach an unwanted
            // ReLU to the linear output and redirect the real ReLU's consumers
            // to the linear output, dropping the genuine ReLU.
            if (!node1->outputs().empty() && !node2->inputs().empty() &&
                node1->outputs()[0]->id() == node2->inputs()[0]->id() &&
                node1->outputs()[0]->uses().size() == 1) {
                if (fuse_pair(node1, node2, graph)) {
                    modified = true;
                }
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

            // Phase P0 / JIT correctness fix (5th-audit sibling-bug A3): mirror
            // the training-mode guard from the Conv+BN pair-fusion pass (line
            // ~429). In training, BN uses live batch mean/variance; folding the
            // running statistics into conv weights would silently produce wrong
            // outputs compared to the eager path.
            if (flow_valid && node2->get_bool_attr("training")) {
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

    // Fold BN into the conv weight/bias (widens f16/bf16 to match eager — JIT-076).
    fold_bn_into_conv(conv_node, gamma, beta, running_mean, running_var, eps,
                      conv_weight, conv_bias);

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
            !node1->get_bool_attr("fused_activation") &&
            (node2->op_type() == OpType::ReLU || node2->op_type() == OpType::GELU)) {
            // Only ONE activation may be folded into a LayerNorm. Without the
            // fused_activation guard above, `relu(gelu(layer_norm(x)))` fuses
            // twice — LN+GELU (marks gelu, deletes GELU), then LN+ReLU
            // (OVERWRITES the marker with relu, deletes ReLU) — silently dropping
            // the GELU. Fusing only the first adjacent activation and leaving the
            // rest as normal nodes keeps replay == eager.
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
// Flash Attention Fusion Pass
// ============================================================================

auto FuseAttentionPass::validate_attention_shapes(
    const std::vector<int64_t>& q_shape,
    const std::vector<int64_t>& k_shape,
    const std::vector<int64_t>& v_shape) -> bool {
    // Q, K, V must be 3D (batch, seq, dim) or 4D (batch, heads, seq, dim)
    if (q_shape.size() != k_shape.size() || q_shape.size() != v_shape.size()) {
        return false;
    }
    if (q_shape.size() != 3 && q_shape.size() != 4) {
        return false;
    }

    if (q_shape.size() == 4) {
        // 4D: batch dims and head dims must match
        if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]) return false;
        if (q_shape[1] != k_shape[1] || q_shape[1] != v_shape[1]) return false;
        // Q and K must have same head dim (last dim)
        if (q_shape[3] != k_shape[3]) return false;
        // K and V must have same seq len
        if (k_shape[2] != v_shape[2]) return false;
    } else {
        // 3D: batch dims must match
        if (q_shape[0] != k_shape[0] || q_shape[0] != v_shape[0]) return false;
        // Q and K must have same head dim (last dim)
        if (q_shape[2] != k_shape[2]) return false;
        // K and V must have same seq len
        if (k_shape[1] != v_shape[1]) return false;
    }

    return true;
}

auto FuseAttentionPass::run(Graph& graph) -> bool {
    bool modified = false;
    const auto& nodes = graph.nodes();

    // Look for the pattern: MatMul -> Mul(scale) -> [optional Add(mask)] -> Softmax -> MatMul
    // We scan for Softmax nodes and trace backward/forward to verify the pattern.
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (nodes[i]->op_type() != OpType::Softmax) continue;

        auto softmax_node = nodes[i];
        if (softmax_node->inputs().empty()) continue;

        // FlashAttention hardwires the softmax over the LAST (key) axis. Only
        // fuse when the traced softmax actually normalizes the last dim; over any
        // other axis the fused kernel would compute different probabilities.
        // (No "dim" attr == default -1 == last axis, which is fine.)
        {
            int64_t srank = static_cast<int64_t>(
                softmax_node->inputs()[0]->shape().size());
            int64_t sdim = softmax_node->has_int_attr("dim")
                               ? softmax_node->get_int_attr("dim")
                               : -1;
            if (sdim < 0) sdim += srank;
            if (srank > 0 && sdim != srank - 1) continue;
        }

        // Trace backward from Softmax: expect Scale (Mul) or Add (mask) then Mul
        auto softmax_input = softmax_node->inputs()[0];
        auto pre_softmax = softmax_input->node();
        if (!pre_softmax) continue;

        std::shared_ptr<Node> mask_add_node;
        std::shared_ptr<Node> scale_node;

        if (pre_softmax->op_type() == OpType::Add) {
            // Optional mask: Add(scaled_qk, mask)
            mask_add_node = pre_softmax;
            if (mask_add_node->inputs().empty()) continue;
            auto add_input = mask_add_node->inputs()[0];
            auto maybe_scale = add_input->node();
            if (!maybe_scale || maybe_scale->op_type() != OpType::Mul) continue;
            scale_node = maybe_scale;
        } else if (pre_softmax->op_type() == OpType::Mul) {
            scale_node = pre_softmax;
        } else {
            continue;
        }

        if (!scale_node || scale_node->inputs().size() < 2) continue;

        // One input to Mul should be a constant scalar (the scale factor),
        // the other should come from MatMul(Q, K^T)
        std::shared_ptr<Node> qk_matmul;
        for (size_t inp = 0; inp < scale_node->inputs().size(); ++inp) {
            auto producer = scale_node->inputs()[inp]->node();
            if (producer && producer->op_type() == OpType::MatMul) {
                qk_matmul = producer;
                break;
            }
        }
        if (!qk_matmul || qk_matmul->inputs().size() < 2) continue;

        // The other operand of the scale Mul (the one not produced by the QK
        // MatMul) MUST be a scalar (numel()==1) Constant for this to be a true
        // softmax scale. If it is a non-scalar tensor (an elementwise mask/scale),
        // a computed value, or a Constant with numel()!=1, the fused
        // FlashAttention executor (which only applies scores*scale with a scalar
        // scale) would silently drop the elementwise-multiply semantics and
        // diverge from eager. In that case, skip fusion.
        {
            bool has_scalar_const_scale = false;
            auto qk_out_id = qk_matmul->outputs().empty()
                                 ? std::string{}
                                 : qk_matmul->outputs()[0]->id();
            for (size_t inp = 0; inp < scale_node->inputs().size(); ++inp) {
                auto value = scale_node->inputs()[inp];
                // Skip the operand coming from the QK MatMul.
                if (!qk_out_id.empty() && value->id() == qk_out_id) continue;
                auto producer = value->node();
                if (producer && producer->op_type() == OpType::Constant) {
                    Tensor scale_tensor = producer->get_tensor_attr("value");
                    if (scale_tensor.numel() == 1) {
                        has_scalar_const_scale = true;
                    }
                }
                // Whether or not it was a scalar const, it is the non-MatMul
                // operand; we only fuse if it qualified as a scalar scale.
                break;
            }
            if (!has_scalar_const_scale) continue;
        }

        // Verify Q*K^T matmul output is only used by the scale node
        if (!qk_matmul->outputs().empty() && qk_matmul->outputs()[0]->uses().size() > 1) {
            continue;
        }

        // Trace forward from Softmax: expect MatMul(attn, V)
        if (softmax_node->outputs().empty()) continue;
        auto softmax_output = softmax_node->outputs()[0];

        // Find the MatMul consuming the softmax output
        std::shared_ptr<Node> av_matmul;
        for (const auto& use : softmax_output->uses()) {
            auto user = use.lock();
            if (user && user->op_type() == OpType::MatMul) {
                av_matmul = user;
                break;
            }
        }
        if (!av_matmul || av_matmul->inputs().size() < 2) continue;

        // Identify Q, K, V tensors and validate shapes
        auto q_value = qk_matmul->inputs()[0];
        auto k_value = qk_matmul->inputs()[1];

        // V is the other input to the av_matmul (not the softmax output)
        std::shared_ptr<Value> v_value;
        for (const auto& inp : av_matmul->inputs()) {
            if (inp->id() != softmax_output->id()) {
                v_value = inp;
                break;
            }
        }
        if (!v_value) continue;

        if (!validate_attention_shapes(q_value->shape(), k_value->shape(), v_value->shape())) {
            continue;
        }

        // Pattern matched - fuse
        if (fuse_attention(qk_matmul, scale_node, softmax_node, av_matmul, mask_add_node, graph)) {
            modified = true;
        }
    }

    return modified;
}

auto FuseAttentionPass::fuse_attention(
    std::shared_ptr<Node> qk_matmul,
    std::shared_ptr<Node> scale_node,
    std::shared_ptr<Node> softmax_node,
    std::shared_ptr<Node> av_matmul,
    std::shared_ptr<Node> mask_add_node,
    Graph& graph) -> bool {

    // Create FlashAttention node
    auto flash_node = graph.create_node(OpType::FlashAttention, "flash_attention");

    // Inputs: Q, K, V from the original matmuls
    auto q_value = qk_matmul->inputs()[0];
    auto k_value = qk_matmul->inputs()[1];

    // V is the non-softmax input to av_matmul
    std::shared_ptr<Value> v_value;
    auto softmax_out_id = softmax_node->outputs()[0]->id();
    for (const auto& inp : av_matmul->inputs()) {
        if (inp->id() != softmax_out_id) {
            v_value = inp;
            break;
        }
    }
    if (!v_value) return false;

    flash_node->add_input(q_value);
    flash_node->add_input(k_value);
    flash_node->add_input(v_value);

    // Extract scale factor from the Mul node. The scale constant may be traced
    // in any float precision (Float16/BFloat16/Float64), so convert to Float32
    // before item<float>() — a bare item<float>() throws DTypeException on a
    // non-Float32 tensor, aborting the whole compile for e.g. an fp16 attention.
    for (const auto& inp : scale_node->inputs()) {
        auto producer = inp->node();
        if (producer && producer->op_type() == OpType::Constant) {
            Tensor scale_tensor = producer->get_tensor_attr("value");
            if (scale_tensor.numel() == 1) {
                flash_node->set_attr(
                    "scale", scale_tensor.to(DType::Float32).item<float>());
            }
            break;
        }
    }

    // If there's a mask, add it as the 4th input
    if (mask_add_node) {
        // The mask is the input to Add that is NOT the scale output
        auto scale_out_id = scale_node->outputs()[0]->id();
        for (const auto& inp : mask_add_node->inputs()) {
            if (inp->id() != scale_out_id) {
                flash_node->add_input(inp);
                flash_node->set_bool_attr("has_mask", true);
                break;
            }
        }
    }

    // Create output value matching av_matmul's output
    if (!av_matmul->outputs().empty()) {
        auto old_output = av_matmul->outputs()[0];
        std::string out_id = flash_node->name() + "_out";
        auto new_output = graph.create_value(
            out_id, old_output->shape(), old_output->dtype(), old_output->device());
        new_output->set_node(flash_node);
        flash_node->add_output(new_output);

        graph.add_node(flash_node);

        // Redirect consumers
        graph.replace_value(old_output->id(), out_id);

        // Remove fused nodes in reverse dependency order
        graph.remove_node(av_matmul);
        graph.remove_node(softmax_node);
        if (mask_add_node) {
            graph.remove_node(mask_add_node);
        }
        graph.remove_node(scale_node);
        graph.remove_node(qk_matmul);
    }

    return true;
}

// ============================================================================
// Residual Add Fusion Pass
// ============================================================================

auto FuseResidualAddPass::is_sublayer_op(const Node& node) -> bool {
    switch (node.op_type()) {
        case OpType::Linear:
        case OpType::Conv2d:
        case OpType::LayerNorm:
        case OpType::BatchNorm2d:
        case OpType::MatMul:
        case OpType::Softmax:
        case OpType::GELU:
        case OpType::ReLU:
        case OpType::FlashAttention:
        case OpType::FusedFFN:
            return true;
        default:
            // Also consider any node with fused attributes
            return node.get_bool_attr("fused_bn") ||
                   node.get_bool_attr("fused_relu") ||
                   node.get_bool_attr("fused_activation");
    }
}

auto FuseResidualAddPass::value_feeds_into(
    const std::string& source_value,
    const std::shared_ptr<Value>& target_value,
    int max_depth) -> bool {
    if (max_depth <= 0) return false;

    auto producer = target_value->node();
    if (!producer) return false;

    for (const auto& input : producer->inputs()) {
        if (input->id() == source_value) {
            return true;
        }
        if (value_feeds_into(source_value, input, max_depth - 1)) {
            return true;
        }
    }
    return false;
}

auto FuseResidualAddPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (const auto& node : graph.nodes()) {
        if (node->op_type() != OpType::Add) continue;
        if (node->inputs().size() < 2) continue;

        // Already marked as residual
        if (node->get_bool_attr("residual")) continue;

        auto input0 = node->inputs()[0];
        auto input1 = node->inputs()[1];

        // Check if one input feeds into the chain producing the other.
        // Pattern: x + sublayer(x) where sublayer is recognized.
        // Check both orderings.
        bool is_residual = false;

        // Case 1: input0 = x, input1 = sublayer(x)
        auto producer1 = input1->node();
        if (producer1 && is_sublayer_op(*producer1)) {
            if (value_feeds_into(input0->id(), input1, 10)) {
                is_residual = true;
            }
        }

        // Case 2: input1 = x, input0 = sublayer(x)
        if (!is_residual) {
            auto producer0 = input0->node();
            if (producer0 && is_sublayer_op(*producer0)) {
                if (value_feeds_into(input1->id(), input0, 10)) {
                    is_residual = true;
                }
            }
        }

        if (is_residual) {
            node->set_bool_attr("residual", true);
            modified = true;
        }
    }

    return modified;
}

// ============================================================================
// Feed-Forward Network Fusion Pass
// ============================================================================

auto FuseFFNPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (size_t i = 0; i + 2 < graph.nodes().size(); ++i) {
        auto& node1 = graph.nodes()[i];
        auto& node2 = graph.nodes()[i + 1];
        auto& node3 = graph.nodes()[i + 2];

        if (node1->op_type() == OpType::Linear &&
            (node2->op_type() == OpType::GELU || node2->op_type() == OpType::ReLU) &&
            node3->op_type() == OpType::Linear) {

            // Verify data flow: linear1 -> activation -> linear2
            bool flow_valid = true;
            if (!node1->outputs().empty() && !node2->inputs().empty()) {
                if (node1->outputs()[0]->id() != node2->inputs()[0]->id()) {
                    flow_valid = false;
                }
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
                if (node2->outputs()[0]->uses().size() > 1) {
                    flow_valid = false;
                }
            } else {
                flow_valid = false;
            }

            if (flow_valid && fuse_triple(node1, node2, node3, graph)) {
                modified = true;
            }
        }
    }

    return modified;
}

auto FuseFFNPass::fuse_triple(std::shared_ptr<Node> linear1,
                               std::shared_ptr<Node> act_node,
                               std::shared_ptr<Node> linear2,
                               Graph& graph) -> bool {
    // Create FusedFFN node
    auto ffn_node = graph.create_node(OpType::FusedFFN, "fused_ffn");

    // Transfer inputs from linear1 (the data input)
    if (!linear1->inputs().empty()) {
        ffn_node->add_input(linear1->inputs()[0]);
    }

    // Store weights from both linear layers
    Tensor w1 = linear1->get_tensor_attr("weight");
    Tensor b1 = linear1->get_tensor_attr("bias");
    Tensor w2 = linear2->get_tensor_attr("weight");
    Tensor b2 = linear2->get_tensor_attr("bias");

    if (w1.numel() > 0) ffn_node->set_tensor_attr("weight1", w1);
    if (b1.numel() > 0) ffn_node->set_tensor_attr("bias1", b1);
    if (w2.numel() > 0) ffn_node->set_tensor_attr("weight2", w2);
    if (b2.numel() > 0) ffn_node->set_tensor_attr("bias2", b2);

    // Store activation type
    ffn_node->set_int_attr("activation_type",
                           act_node->op_type() == OpType::ReLU ? 1 : 2);

    // Create output value matching linear2's output
    if (!linear2->outputs().empty()) {
        auto old_output = linear2->outputs()[0];
        std::string out_id = ffn_node->name() + "_out";
        auto new_output = graph.create_value(
            out_id, old_output->shape(), old_output->dtype(), old_output->device());
        new_output->set_node(ffn_node);
        ffn_node->add_output(new_output);

        graph.add_node(ffn_node);

        // Redirect consumers
        graph.replace_value(old_output->id(), out_id);

        // Also redirect any remaining references to intermediate values
        if (!linear1->outputs().empty()) {
            graph.replace_value(linear1->outputs()[0]->id(), out_id);
        }
        if (!act_node->outputs().empty()) {
            graph.replace_value(act_node->outputs()[0]->id(), out_id);
        }

        // Remove fused nodes
        graph.remove_node(linear2);
        graph.remove_node(act_node);
        graph.remove_node(linear1);
    }

    return true;
}

// ============================================================================
// Shape Guard Insertion Pass
// ============================================================================

auto ShapeGuardInsertionPass::run(Graph& graph) -> bool {
    bool modified = false;

    // Insert a ShapeGuard node for each graph input
    auto inputs = graph.inputs();  // Copy since we'll modify the graph
    for (size_t idx = 0; idx < inputs.size(); ++idx) {
        auto input_value = inputs[idx];

        // Skip if a guard already exists for this input
        bool already_guarded = false;
        for (const auto& use : input_value->uses()) {
            auto user = use.lock();
            if (user && user->op_type() == OpType::ShapeGuard) {
                already_guarded = true;
                break;
            }
        }
        if (already_guarded) continue;

        // Skip inputs with unknown shapes
        if (input_value->shape().empty()) continue;

        // Create ShapeGuard node
        auto guard_node = graph.create_node(OpType::ShapeGuard,
                                             "shape_guard_" + std::to_string(idx));

        // Store expected shape
        guard_node->set_vec_attr("expected_shape", input_value->shape());
        guard_node->set_int_attr("input_index", static_cast<int64_t>(idx));

        // Guard takes the input value and produces a new value with the same
        // shape/dtype/device (identity semantics)
        guard_node->add_input(input_value);

        std::string guard_out_id = guard_node->name() + "_out";
        auto guard_output = graph.create_value(
            guard_out_id, input_value->shape(), input_value->dtype(), input_value->device());
        guard_output->set_node(guard_node);
        guard_node->add_output(guard_output);

        graph.add_node(guard_node);

        // Redirect all uses of the input (except by the guard itself) to use
        // the guard output instead
        graph.replace_value(input_value->id(), guard_out_id);

        // Restore the guard's own input to point back to the original input value
        guard_node->replace_input(0, input_value);

        modified = true;
    }

    if (modified) {
        graph.topological_sort();
    }

    return modified;
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

    // Only redirect consumers to a surviving operand when that operand already
    // has the node's OUTPUT shape. Otherwise the identity/absorbing operand is
    // the broadcasting one (e.g. x:(3,4) * 0:scalar), and redirecting to it
    // would silently change the result's shape. In that case we conservatively
    // skip the rewrite rather than corrupt the graph.
    if (node->outputs().empty()) return false;
    const auto& out_shape = node->outputs()[0]->shape();
    auto same_shape = [&](const std::shared_ptr<Value>& v) -> bool {
        return v && v->shape() == out_shape;
    };

    switch (node->op_type()) {
        case OpType::Add:
            // x + 0 = x
            if (is_const1 && is_all_zeros(*producer1) && same_shape(input0)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            // 0 + x = x
            if (is_const0 && is_all_zeros(*producer0) && same_shape(input1)) {
                graph.replace_node_with_value(node, input1->id());
                return true;
            }
            break;

        case OpType::Sub:
            // x - 0 = x
            if (is_const1 && is_all_zeros(*producer1) && same_shape(input0)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            break;

        case OpType::Mul:
            // x * 1 = x
            if (is_const1 && is_all_ones(*producer1) && same_shape(input0)) {
                graph.replace_node_with_value(node, input0->id());
                return true;
            }
            // 1 * x = x
            if (is_const0 && is_all_ones(*producer0) && same_shape(input1)) {
                graph.replace_node_with_value(node, input1->id());
                return true;
            }
            // x * 0 = 0 — UNSAFE: IEEE-754 gives NaN*0 = NaN and Inf*0 = NaN,
            // so folding to the literal zero changes results for non-finite x.
            // Gated behind allow_unsafe_algebra_ (default off). The shape guard
            // also avoids replacing x:(M,N) with a broadcasting scalar zero.
            if (allow_unsafe_algebra_) {
                if (is_const1 && is_all_zeros(*producer1) && same_shape(input1)) {
                    graph.replace_node_with_value(node, input1->id());
                    return true;
                }
                // 0 * x = 0
                if (is_const0 && is_all_zeros(*producer0) && same_shape(input0)) {
                    graph.replace_node_with_value(node, input0->id());
                    return true;
                }
            }
            break;

        case OpType::Div:
            // x / 1 = x
            if (is_const1 && is_all_ones(*producer1) && same_shape(input0)) {
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

    // Both transforms below change observable output for non-finite / out-of-
    // domain inputs and are therefore gated behind allow_unsafe_algebra_:
    //   log(exp(x)) = x   — original overflows exp(x)->+Inf, log(+Inf)=+Inf for
    //                       large x; the rewrite yields the finite x instead.
    //   exp(log(x)) = x   — only valid for x>0; for x<=0 the original produces
    //                       NaN/-Inf, the rewrite returns the (finite) x.
    if (!allow_unsafe_algebra_) return false;

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
// Strength Reduction Pass
// ============================================================================

/// Try to extract a scalar float value from a Constant node.
/// Returns std::nullopt if the node is not a scalar constant.
static auto get_scalar_float(const Node& node) -> std::optional<float> {
    if (node.op_type() != OpType::Constant) return std::nullopt;
    try {
        const Tensor& t = node.get_tensor_attr("value");
        if (t.numel() != 1) return std::nullopt;
        if (t.dtype() == DType::Float32) return t.item<float>();
        if (t.dtype() == DType::Float64) return static_cast<float>(t.item<double>());
        return std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

/// Like get_scalar_float but preserves full double precision for Float64
/// constants (used when computing a reciprocal that will be baked back into a
/// Float64 tensor — narrowing to float first would lose precision).
static auto get_scalar_double(const Node& node) -> std::optional<double> {
    if (node.op_type() != OpType::Constant) return std::nullopt;
    try {
        const Tensor& t = node.get_tensor_attr("value");
        if (t.numel() != 1) return std::nullopt;
        if (t.dtype() == DType::Float32) return static_cast<double>(t.item<float>());
        if (t.dtype() == DType::Float64) return t.item<double>();
        return std::nullopt;
    } catch (...) {
        return std::nullopt;
    }
}

auto StrengthReductionPass::run(Graph& graph) -> bool {
    bool modified = false;
    std::vector<std::shared_ptr<Node>> nodes(graph.nodes().begin(), graph.nodes().end());

    for (const auto& node : nodes) {
        auto& current_nodes = graph.nodes();
        if (std::find(current_nodes.begin(), current_nodes.end(), node) == current_nodes.end()) {
            continue;
        }
        if (reduce_node(node, graph)) {
            modified = true;
        }
    }
    return modified;
}

auto StrengthReductionPass::reduce_node(std::shared_ptr<Node> node, Graph& graph) -> bool {
    if (node->inputs().size() < 2) {
        // Unary ops don't have strength reduction opportunities here
        return false;
    }

    auto input0 = node->inputs()[0];
    auto input1 = node->inputs()[1];
    auto producer1 = input1->node();

    // ---- Div(x, const) -> Mul(x, 1/const) ----
    // Multiplying by a precomputed reciprocal is not bit-identical to division,
    // so this rewrite only fires under fast-math (allow_unsafe_algebra_).
    if (allow_unsafe_algebra_ && node->op_type() == OpType::Div && producer1 &&
        producer1->op_type() == OpType::Constant) {
        auto val = get_scalar_double(*producer1);
        if (val && *val != 0.0) {
            // Compute the reciprocal in the operand's precision (double for
            // Float64) so a Float64 divisor keeps its full mantissa. full()
            // narrows to Float32 once if that is the operand dtype.
            double reciprocal = 1.0 / *val;
            Tensor recip_tensor = tenzor::full({1}, reciprocal,
                input1->dtype(), input1->device());

            auto recip_node = graph.create_node(OpType::Constant);
            recip_node->set_tensor_attr("value", recip_tensor);

            std::string recip_val_id = recip_node->name() + "_out";
            // The reciprocal tensor is built with shape {1} above, so the graph
            // Value must declare the same shape. The divisor constant may carry
            // a different numel()==1 shape (e.g. {}, {1,1}); using input1->shape()
            // would desync the Value's declared shape from its tensor payload.
            auto recip_val = graph.create_value(recip_val_id, std::vector<int64_t>{1},
                input1->dtype(), input1->device());
            recip_val->set_node(recip_node);
            recip_node->add_output(recip_val);
            graph.add_node(recip_node);

            // Create Mul(x, 1/const) node replacing Div
            auto mul_node = graph.create_node(OpType::Mul);
            mul_node->add_input(input0);
            mul_node->add_input(recip_val);

            if (!node->outputs().empty()) {
                auto old_output = node->outputs()[0];
                std::string mul_val_id = mul_node->name() + "_out";
                auto mul_val = graph.create_value(mul_val_id, old_output->shape(),
                    old_output->dtype(), old_output->device());
                mul_val->set_node(mul_node);
                mul_node->add_output(mul_val);
                graph.add_node(mul_node);
                graph.replace_value(old_output->id(), mul_val_id);
                graph.remove_node(node);
                return true;
            }
        }
    }

    // Pow(x,2)->Mul(x,x) and Pow(x,0.5)->Sqrt(x) are NOT bit-identical to the
    // eager Pow kernel: libm pow(x,2)/pow(x,0.5) differ from x*x / sqrt(x) by
    // ~1 ULP, and sqrt(-0.0) == -0.0 whereas pow(-0.0, 0.5) == +0.0. Gate behind
    // allow_unsafe_algebra_ (default off), like the Div->Mul-reciprocal rewrite,
    // so the JIT matches eager exactly by default.
    // ---- Pow(x, 2) -> Mul(x, x) ----
    if (allow_unsafe_algebra_ && node->op_type() == OpType::Pow && producer1 &&
        producer1->op_type() == OpType::Constant) {
        auto val = get_scalar_float(*producer1);
        if (val) {
            if (*val == 2.0f) {
                // Replace with Mul(x, x)
                auto mul_node = graph.create_node(OpType::Mul);
                mul_node->add_input(input0);
                mul_node->add_input(input0);  // x * x

                if (!node->outputs().empty()) {
                    auto old_output = node->outputs()[0];
                    std::string mul_val_id = mul_node->name() + "_out";
                    auto mul_val = graph.create_value(mul_val_id, old_output->shape(),
                        old_output->dtype(), old_output->device());
                    mul_val->set_node(mul_node);
                    mul_node->add_output(mul_val);
                    graph.add_node(mul_node);
                    graph.replace_value(old_output->id(), mul_val_id);
                    graph.remove_node(node);
                    return true;
                }
            }

            // ---- Pow(x, 0.5) -> Sqrt(x) ----
            if (*val == 0.5f) {
                auto sqrt_node = graph.create_node(OpType::Sqrt);
                sqrt_node->add_input(input0);

                if (!node->outputs().empty()) {
                    auto old_output = node->outputs()[0];
                    std::string sqrt_val_id = sqrt_node->name() + "_out";
                    auto sqrt_val = graph.create_value(sqrt_val_id, old_output->shape(),
                        old_output->dtype(), old_output->device());
                    sqrt_val->set_node(sqrt_node);
                    sqrt_node->add_output(sqrt_val);
                    graph.add_node(sqrt_node);
                    graph.replace_value(old_output->id(), sqrt_val_id);
                    graph.remove_node(node);
                    return true;
                }
            }
        }
    }

    // ---- Mul(x, 2) -> Add(x, x) ----
    if (node->op_type() == OpType::Mul) {
        auto producer0 = input0->node();
        bool is_const0 = producer0 && producer0->op_type() == OpType::Constant;
        bool is_const1 = producer1 && producer1->op_type() == OpType::Constant;

        std::optional<float> val;
        std::shared_ptr<Value> non_const_input;

        if (is_const1 && !is_const0) {
            val = get_scalar_float(*producer1);
            non_const_input = input0;
        } else if (is_const0 && !is_const1) {
            val = get_scalar_float(*producer0);
            non_const_input = input1;
        }

        if (val && *val == 2.0f && non_const_input) {
            // Replace Mul(x, 2) with Add(x, x)
            auto add_node = graph.create_node(OpType::Add);
            add_node->add_input(non_const_input);
            add_node->add_input(non_const_input);

            if (!node->outputs().empty()) {
                auto old_output = node->outputs()[0];
                std::string add_val_id = add_node->name() + "_out";
                auto add_val = graph.create_value(add_val_id, old_output->shape(),
                    old_output->dtype(), old_output->device());
                add_val->set_node(add_node);
                add_node->add_output(add_val);
                graph.add_node(add_node);
                graph.replace_value(old_output->id(), add_val_id);
                graph.remove_node(node);
                return true;
            }
        }
    }

    return false;
}

// ============================================================================
// Loop Unrolling Pass
// ============================================================================

auto LoopUnrollingPass::run(Graph& graph) -> bool {
    bool modified = false;

    std::vector<std::shared_ptr<Node>> loop_nodes;
    for (const auto& node : graph.nodes()) {
        if (node->op_type() == OpType::Loop) {
            loop_nodes.push_back(node);
        }
    }

    for (auto& loop_node : loop_nodes) {
        auto& current_nodes = graph.nodes();
        if (std::find(current_nodes.begin(), current_nodes.end(), loop_node) == current_nodes.end()) {
            continue;
        }

        if (loop_node->inputs().empty()) continue;
        auto bound_value = loop_node->inputs()[0];
        auto bound_producer = bound_value->node();
        if (!bound_producer || bound_producer->op_type() != OpType::Constant) continue;

        // JIT Constant nodes store their payload as a TENSOR attr named
        // "value" (see set_tensor_attr/get_tensor_attr everywhere else), not as
        // an int attr — int and tensor attrs live in separate maps. Reading
        // get_int_attr("value") here always returned 0, so the guard below
        // always continued and no loop was ever unrolled. Read the scalar from
        // the tensor attr instead, handling integer and floating payloads.
        const Tensor& bound_tensor = bound_producer->get_tensor_attr("value");
        if (bound_tensor.numel() != 1) continue;
        int64_t trip_count = 0;
        try {
            switch (bound_tensor.dtype()) {
                case DType::Int64:
                    trip_count = bound_tensor.item<int64_t>();
                    break;
                case DType::Int32:
                    trip_count = static_cast<int64_t>(bound_tensor.item<int32_t>());
                    break;
                case DType::Float32:
                    trip_count = static_cast<int64_t>(bound_tensor.item<float>());
                    break;
                case DType::Float64:
                    trip_count = static_cast<int64_t>(bound_tensor.item<double>());
                    break;
                default:
                    continue;  // Unsupported bound dtype — leave the loop intact.
            }
        } catch (...) {
            continue;
        }
        if (trip_count <= 0 || trip_count > max_unroll_) continue;

        auto body = loop_node->body();
        if (!body || body->nodes().empty()) continue;

        auto loop_inputs = loop_node->inputs();
        auto loop_outputs = loop_node->outputs();

        // Carried values start as the loop's non-bound inputs
        std::vector<std::string> carried_ids;
        for (size_t i = 1; i < loop_inputs.size(); ++i) {
            carried_ids.push_back(loop_inputs[i]->id());
        }

        // For each iteration, clone body nodes with remapped IDs
        for (int64_t iter = 0; iter < trip_count; ++iter) {
            std::unordered_map<std::string, std::string> id_remap;

            // Loop convention: body_inputs = [iter, cond, carried_0, ...];
            // carried_ids tracks the loop's [cond, carried_0, ...]. Previously
            // body_inputs[i] was mapped to carried_ids[i], which aligned the
            // iteration counter (body_inputs[0]) with cond and shifted every
            // carried value by one — computing wrong results. Map the counter to
            // a per-iteration constant, and cond/carried (body_inputs[1..]) to
            // carried_ids[0..].
            auto& body_inputs = body->inputs();
            if (!body_inputs.empty()) {
                auto iter_in = body_inputs[0];
                // Build the counter constant on the host (data<int64_t>() is a raw
                // host pointer; writing into a device-allocated tensor would be
                // UB), then move it to the loop's device so the constant lands
                // where the body runs.
                Tensor iter_tensor({1}, DType::Int64, Device::cpu());
                iter_tensor.data<int64_t>()[0] = iter;
                if (iter_in->device().type != Device::Type::CPU) {
                    iter_tensor = iter_tensor.to(iter_in->device());
                }
                auto iter_node = graph.create_node(OpType::Constant);
                iter_node->set_tensor_attr("value", iter_tensor);
                std::string iter_val_id =
                    iter_node->name() + "_iter_" + std::to_string(iter);
                auto iter_val = graph.create_value(iter_val_id, {1}, DType::Int64,
                                                   iter_in->device());
                iter_val->set_node(iter_node);
                iter_node->add_output(iter_val);
                graph.add_node(iter_node);
                id_remap[iter_in->id()] = iter_val_id;
            }
            for (size_t j = 0; j + 1 < body_inputs.size() && j < carried_ids.size(); ++j) {
                id_remap[body_inputs[j + 1]->id()] = carried_ids[j];
            }

            for (const auto& body_node : body->nodes()) {
                auto cloned = graph.create_node(body_node->op_type());

                for (const auto& input : body_node->inputs()) {
                    auto remap_it = id_remap.find(input->id());
                    std::string actual_id = (remap_it != id_remap.end()) ? remap_it->second : input->id();
                    auto val = graph.get_value(actual_id);
                    if (!val) {
                        // Body-local captured leaf (a constant/parameter that
                        // lives only in the loop-body subgraph, not the outer
                        // graph). Previously this input was silently dropped,
                        // leaving the cloned op with a missing operand (JIT-054).
                        // Materialize the body's constant in the outer graph so
                        // the unrolled op has a real operand.
                        auto cit = body->constants().find(input->id());
                        if (cit != body->constants().end()) {
                            // Materialize with a node-unique id (cloned->name() is
                            // globally unique) so a body-local constant id that
                            // happens to collide across DIFFERENT loops does not
                            // reuse the wrong outer value.
                            const std::string mat_id =
                                cloned->name() + "_uconst_" + input->id();
                            val = graph.get_value(mat_id);
                            if (!val) {
                                const auto sh = cit->second.shape();
                                val = graph.create_value(
                                    mat_id,
                                    std::vector<int64_t>(sh.begin(), sh.end()),
                                    cit->second.dtype(), cit->second.device());
                                graph.set_constant(mat_id, cit->second);
                            }
                        }
                    }
                    if (val) cloned->add_input(val);
                }

                for (const auto& output : body_node->outputs()) {
                    std::string new_id = output->id() + "_unroll_" + std::to_string(iter);
                    auto new_val = graph.create_value(new_id, output->shape(), output->dtype(), output->device());
                    new_val->set_node(cloned);
                    cloned->add_output(new_val);
                    id_remap[output->id()] = new_id;
                }
                graph.add_node(cloned);
            }

            auto& body_outputs = body->outputs();
            for (size_t i = 0; i < std::min(body_outputs.size(), carried_ids.size()); ++i) {
                auto remap_it = id_remap.find(body_outputs[i]->id());
                if (remap_it != id_remap.end()) {
                    carried_ids[i] = remap_it->second;
                }
            }
        }

        // loop_outputs are the final carried values [v_0_final, ...]; they map to
        // carried_ids[1..] (carried_ids[0] is the loop cond, which is NOT a loop
        // output). Skip the cond slot — previously loop_outputs[i] was wired to
        // carried_ids[i], returning cond/shifted values.
        for (size_t i = 0; i < loop_outputs.size() && (i + 1) < carried_ids.size(); ++i) {
            graph.replace_value(loop_outputs[i]->id(), carried_ids[i + 1]);
        }
        graph.remove_node(loop_node);
        modified = true;
    }
    return modified;
}

// ============================================================================
// LICM Pass (Loop-Invariant Code Motion)
// ============================================================================

auto LICMPass::run(Graph& graph) -> bool {
    bool modified = false;

    for (const auto& node : graph.nodes()) {
        if (node->op_type() != OpType::Loop) continue;

        auto body = node->body();
        if (!body || body->nodes().empty()) continue;

        // Collect value IDs defined inside the loop body
        std::unordered_set<std::string> body_defined_ids;
        for (const auto& body_node : body->nodes()) {
            for (const auto& output : body_node->outputs()) {
                body_defined_ids.insert(output->id());
            }
        }
        for (const auto& input : body->inputs()) {
            body_defined_ids.insert(input->id());
        }

        // Find loop-invariant nodes
        std::vector<std::shared_ptr<Node>> invariant_nodes;
        for (const auto& body_node : body->nodes()) {
            bool all_external = true;
            for (const auto& input : body_node->inputs()) {
                if (body_defined_ids.count(input->id()) > 0) {
                    all_external = false;
                    break;
                }
            }
            // Exclude stateful ops (Dropout, BatchNorm in training, etc.) — the
            // same guard CSE uses. Hoisting one out of the loop computes it once
            // and threads it as a passthrough, collapsing per-iteration random
            // draws / running-stat updates into a single value. That is a
            // wrong-output transform the moment the executor honors training-mode
            // semantics, even if the value happens to be loop-invariant.
            if (all_external && !body_node->inputs().empty() &&
                !is_stateful_op(body_node->op_type())) {
                invariant_nodes.push_back(body_node);
            }
        }

        // Hoist invariant nodes before the loop.
        //
        // A loop-invariant body node computes the same value every iteration, so
        // we evaluate it once in the OUTER graph and thread the result into the
        // loop body as an extra loop-carried (passthrough) value. This keeps the
        // body's consumers valid: they continue to reference the very same Value
        // object, which we convert from a body-internal result into a body INPUT.
        //
        // The executor (graph.cpp Loop case) treats the body positionally:
        //   Loop node inputs : [max_iter, cond, carried_0, carried_1, ...]
        //   body inputs      : [iter,     cond, carried_0, carried_1, ...]
        //   body outputs     : [cond,           carried_0, carried_1, ...]
        // Appending one new value to the Loop node inputs, the body inputs, and
        // the body outputs keeps these three lists index-consistent so the new
        // carried value is initialised from the hoisted result and passed through
        // unchanged on every iteration.
        for (auto& inv_node : invariant_nodes) {
            // The invariant node must produce exactly one output for the carried
            // threading below to be well-defined.
            if (inv_node->outputs().size() != 1) continue;

            // Only hoist when EVERY input resolves in the OUTER graph (JIT-054).
            // A loop-body node whose operands are body-local captured leaves
            // (constants/parameters that exist only in the subgraph, not in the
            // outer graph) would otherwise be hoisted with those inputs silently
            // dropped — emitting a zero-/missing-input op that yields garbage or
            // crashes. Leave such a node in the body.
            bool all_resolvable = true;
            for (const auto& input : inv_node->inputs()) {
                if (!graph.get_value(input->id())) { all_resolvable = false; break; }
            }
            if (!all_resolvable) continue;

            // 1. Re-create the node in the outer graph with a FRESH unique output
            //    id (ids must not be shared across the outer and body scopes).
            auto hoisted = graph.create_node(inv_node->op_type());
            for (const auto& input : inv_node->inputs()) {
                hoisted->add_input(graph.get_value(input->id()));
            }
            // Copy scalar/vector/bool/tensor attrs so the hoisted op is identical.
            {
                auto src = inv_node->get_all_attrs();
                auto dst = hoisted->get_all_attrs();
                std::get<0>(dst) = std::get<0>(src);
                std::get<1>(dst) = std::get<1>(src);
                std::get<2>(dst) = std::get<2>(src);
                std::get<3>(dst) = std::get<3>(src);
                std::get<4>(dst) = std::get<4>(src);
            }

            auto body_output = inv_node->outputs()[0];
            std::string hoisted_id = hoisted->name() + "_licm";
            auto hoisted_val = graph.create_value(
                hoisted_id, body_output->shape(), body_output->dtype(),
                body_output->device());
            hoisted_val->set_node(hoisted);
            hoisted->add_output(hoisted_val);
            graph.add_node(hoisted);

            // 2. Remove the producer from the body. The body Value object stays
            //    alive (its consumers still hold it) and becomes a graph input.
            body->remove_node(inv_node);
            body_output->set_node(nullptr);

            // 3. Thread the hoisted result into the body as a new loop-carried
            //    value: append to body inputs (so consumers read it), append the
            //    same value to body outputs (passthrough to keep the carried set
            //    stable across iterations), and append the hoisted outer value to
            //    the Loop node inputs (the initial carried value).
            auto body_inputs = body->inputs();
            body_inputs.push_back(body_output);
            body->set_inputs(std::move(body_inputs));

            auto body_outputs = body->outputs();
            body_outputs.push_back(body_output);
            body->set_outputs(std::move(body_outputs));

            node->add_input(hoisted_val);

            modified = true;
        }
    }
    return modified;
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
// Extended Fusion Pass
// ============================================================================

auto ExtendedFusionPass::run(Graph& graph) -> bool {
    // Extended fusions are lowered to NVRTC/HIPRTC GPU kernels
    // (execute_extended_fused), which run ONLY on CUDA/ROCm. Never form them for
    // a non-GPU graph: the fused node would throw at execution time (there is no
    // — and must not be a — CPU kernel masquerading as the GPU fusion). A CPU
    // graph simply runs the ops eagerly/unfused, which is correct. The cost
    // model's device is not reliably set here, so gate on the graph's own device.
    {
        Device::Type dev = Device::Type::CPU;
        const auto& gin = graph.inputs();
        if (!gin.empty()) dev = gin[0]->device().type;
        if (dev != Device::Type::CUDA && dev != Device::Type::ROCm) {
            return false;
        }
        // NOTE: the cost model is intentionally left on its permissive default
        // (CPU) heuristic rather than being pinned to `dev`. This pass only ever
        // runs on GPU graphs (gated above), and a fused extended kernel is
        // numerically identical to the unfused ops — so the fuse/no-fuse choice
        // is purely a performance decision, never a cross-backend correctness
        // one. Pinning the GPU heuristic here makes small-tensor patterns decline
        // to fuse (estimate_speedup < 1.0), which only removes a valid, correct
        // fusion opportunity; the permissive default fuses readily on GPU, which
        // is the intended behaviour.
    }

    PatternMatcher matcher;
    matcher.set_max_mlp_hidden_dim(max_mlp_hidden_);

    auto matches = matcher.find_all(graph);
    if (matches.empty()) return false;

    bool modified = false;

    for (auto& match : matches) {
        // Check profitability
        FusionCandidate candidate;
        candidate.num_ops = match.nodes.size();
        candidate.total_elements = match.estimated_elements;
        candidate.num_memory_accesses = match.nodes.size() * 2;  // Rough estimate: 1 read + 1 write per op
        candidate.kind = match.kind;

        if (!cost_model_.should_fuse(candidate)) continue;

        // Skip fusion if device-aware cost model predicts no speedup
        if (cost_model_.estimate_speedup(candidate) < 1.0) continue;

        // The extended kernels only load/store Float32/Float64/Float16/BFloat16
        // (math runs in float/double). Fusing an Int*/Complex pattern would form
        // a node the GPU codegen cannot faithfully run, and eager collapse of
        // GemmEpilogue/SmallMLP/Reduction drops computation — so refuse to fuse
        // unsupported dtypes and let the ops run via normal op dispatch.
        if (!match.inputs.empty()) {
            DType dt = match.inputs[0]->dtype();
            if (dt != DType::Float32 && dt != DType::Float64 &&
                dt != DType::Float16 && dt != DType::BFloat16) {
                continue;
            }
        }

        // For a Reduction match, pre/post element-wise ops must be lowered into
        // the fused kernel's single input stream. Build their ElemStep sequences
        // NOW (before rewiring the graph) so an unrepresentable op — a binary
        // op needing a distinct second operand — aborts the fusion cleanly rather
        // than silently dropping the op at execution. The serialized quads
        // (op, input_idx, second_input_idx, scalar_bits) are attached to the
        // fused node and reconstructed by the executor into ExtendedFusionGroup
        // pre_ops/post_ops. Mirrors ExtendedKernelCodegen::generate_reduction's
        // representability rules (only unary math + a same-operand self-square).
        std::vector<int64_t> pre_ops_serial, post_ops_serial;
        if (match.kind == FusionKind::Reduction) {
            auto to_elem = [](const std::shared_ptr<Node>& n,
                              std::optional<ElemStep>& step) -> bool {
                auto push_unary = [&](ElemOp e) {
                    step = ElemStep{e, -1, -1, 0.0};
                };
                switch (n->op_type()) {
                    case OpType::Exp:     push_unary(ElemOp::Exp);     return true;
                    case OpType::Log:     push_unary(ElemOp::Log);     return true;
                    case OpType::Sqrt:    push_unary(ElemOp::Sqrt);    return true;
                    case OpType::Abs:     push_unary(ElemOp::Abs);     return true;
                    case OpType::Neg:     push_unary(ElemOp::Neg);     return true;
                    case OpType::ReLU:    push_unary(ElemOp::Relu);    return true;
                    case OpType::Sigmoid: push_unary(ElemOp::Sigmoid); return true;
                    case OpType::Tanh:    push_unary(ElemOp::Tanh);    return true;
                    case OpType::GELU:    push_unary(ElemOp::Gelu);    return true;
                    case OpType::Mul:
                        // Only a same-operand self-square x*x is representable in
                        // the single-stream reduction kernel (RMS/L2 pre-op). A
                        // Mul of two distinct tensors has no second stream.
                        if (n->inputs().size() == 2 &&
                            n->inputs()[0].get() == n->inputs()[1].get()) {
                            step = ElemStep{ElemOp::Mul, 0, 0, 0.0};
                            return true;
                        }
                        return false;
                    default:
                        return false;
                }
            };
            auto append = [](std::vector<int64_t>& out, const ElemStep& s) {
                int64_t bits;
                std::memcpy(&bits, &s.scalar, sizeof(bits));
                out.push_back(static_cast<int64_t>(s.op));
                out.push_back(static_cast<int64_t>(s.input_idx));
                out.push_back(static_cast<int64_t>(s.second_input_idx));
                out.push_back(bits);
            };
            bool seen_reduction = false;
            bool representable = true;
            for (auto& n : match.nodes) {
                OpType op = n->op_type();
                if (op == OpType::Sum || op == OpType::Mean ||
                    op == OpType::Max || op == OpType::Min) {
                    seen_reduction = true;
                    // The fused kernel reduces exactly ONE explicit axis. A full
                    // reduction (no "dim") or a multi-axis reduction ("dims" vec)
                    // cannot be represented — fusing it would silently reduce only
                    // the last axis. Refuse so the ops run via correct dispatch.
                    if (!n->has_int_attr("dim")) { representable = false; break; }
                    continue;
                }
                std::optional<ElemStep> step;
                if (!to_elem(n, step) || !step) { representable = false; break; }
                append(seen_reduction ? post_ops_serial : pre_ops_serial, *step);
            }
            if (!representable) continue;  // leave the ops unfused (correct eager)
        }

        // A LayerNorm/RMSNorm that FuseLayerNormActivationPass folded a trailing
        // ReLU/GELU into (marked fused_activation) CANNOT be represented by the
        // extended fused kernel — generate_layer_norm / generate_rms_norm emit no
        // activation. Fusing it here would silently DROP the activation on the
        // GPU path while the interpreter path applies it, diverging across
        // backends. Refuse the fusion so the LayerNorm+activation runs through
        // normal dispatch, where execute_node applies the fused activation
        // uniformly on every backend.
        {
            bool has_folded_activation = false;
            for (auto& n : match.nodes) {
                if ((n->op_type() == OpType::LayerNorm ||
                     n->op_type() == OpType::RMSNorm) &&
                    n->get_bool_attr("fused_activation")) {
                    has_folded_activation = true;
                    break;
                }
            }
            if (has_folded_activation) continue;  // interpreter applies the activation
        }

        // Map the match kind to the appropriate fused OpType
        OpType fused_type;
        switch (match.kind) {
            case FusionKind::Softmax:      fused_type = OpType::Softmax; break;
            case FusionKind::GemmEpilogue: fused_type = OpType::Linear; break;
            case FusionKind::SmallMLP:     fused_type = OpType::FusedFFN; break;
            default:
                // For LayerNorm, RMSNorm, Reduction: use the existing op type
                // of the dominant operation as the fused node type
                fused_type = match.nodes[0]->op_type();
                break;
        }

        // Create a fused node that replaces the matched subgraph
        auto fused_node = std::make_shared<Node>(fused_type);

        // Copy inputs from the first matched node
        for (auto& input : match.inputs) {
            fused_node->add_input(input);
        }

        // Copy outputs from the last matched node
        if (!match.nodes.empty()) {
            auto& last = match.nodes.back();
            for (auto& output : last->outputs()) {
                fused_node->add_output(output);
                output->set_node(fused_node);
            }
        }

        // Store the fusion signature as an attribute for the execution engine
        fused_node->set_name("fused_" + match.signature);
        fused_node->set_attr("fusion_kind", static_cast<float>(static_cast<int>(match.kind)));
        fused_node->set_int_attr("fusion_num_ops", static_cast<int64_t>(match.nodes.size()));

        // Copy key attributes from matched nodes (eps, momentum for normalization)
        for (auto& node : match.nodes) {
            float eps_val = node->get_attr("eps");
            if (eps_val != 0.0f) fused_node->set_attr("eps", eps_val);
            float momentum_val = node->get_attr("momentum");
            if (momentum_val != 0.0f) fused_node->set_attr("momentum", momentum_val);
        }

        // ------------------------------------------------------------------
        // Propagate the structural attributes that the extended codegen reads
        // off the reconstructed ExtendedFusionGroup. Without these the executor
        // would build a default group (dim=-1, no affine/bias, ReLU activation,
        // Sum reduction) and silently generate the wrong kernel — e.g. a Mean
        // fusion emitted as a plain Sum, a non-trailing softmax run over the
        // wrong axis, or a GEMM epilogue that drops its bias/activation.
        //
        // Attr name -> ExtendedFusionGroup field mapping:
        //   fused_reduce_dim       -> reduce_dim / softmax_dim / norm_axis
        //   fused_keepdim          -> keepdim
        //   fused_reduce_kind      -> reduce_kind (OpType of the reduction op)
        //   fused_has_affine       -> has_affine
        //   fused_has_bias         -> has_bias
        //   fused_has_activation   -> has_activation
        //   fused_activation_type  -> activation_type (OpType of the activation)
        // ------------------------------------------------------------------
        auto is_reduction_op = [](OpType op) {
            return op == OpType::Sum || op == OpType::Mean ||
                   op == OpType::Max || op == OpType::Min;
        };
        auto is_activation_op = [](OpType op) {
            return op == OpType::ReLU || op == OpType::Sigmoid ||
                   op == OpType::Tanh || op == OpType::GELU;
        };

        switch (match.kind) {
            case FusionKind::Reduction:
            case FusionKind::Softmax:
            case FusionKind::LayerNorm:
            case FusionKind::RMSNorm: {
                // Find the reduction/normalization op in the matched chain and
                // copy its dim/keepdim and (for plain reductions) its op kind.
                for (auto& node : match.nodes) {
                    OpType op = node->op_type();
                    bool is_norm = (op == OpType::LayerNorm || op == OpType::Softmax);
                    if (is_reduction_op(op) || is_norm) {
                        if (node->has_attr("dim")) {
                            fused_node->set_int_attr("fused_reduce_dim",
                                                     node->get_int_attr("dim"));
                        }
                        if (node->has_attr("keepdim")) {
                            fused_node->set_bool_attr("fused_keepdim",
                                                      node->get_bool_attr("keepdim"));
                        }
                        if (is_reduction_op(op)) {
                            fused_node->set_int_attr("fused_reduce_kind",
                                                     static_cast<int64_t>(op));
                        }
                        break;
                    }
                }
                // LayerNorm/RMSNorm affine presence: gamma/beta arrive as extra
                // external inputs beyond the single normalized tensor.
                if (match.kind == FusionKind::LayerNorm ||
                    match.kind == FusionKind::RMSNorm) {
                    fused_node->set_bool_attr("fused_has_affine",
                                              match.inputs.size() > 1);
                }
                break;
            }
            case FusionKind::GemmEpilogue: {
                // A Linear base carries its bias internally (external inputs are
                // [x, W, b]); a MatMul base expresses bias as a separate Add
                // node. Either source must set has_bias so the executor adds the
                // [cols] bias vector at input index 2 — otherwise the fused
                // kernel silently drops the bias and diverges from eager.
                const bool base_linear_has_bias =
                    match.nodes[0]->op_type() == OpType::Linear &&
                    match.nodes[0]->inputs().size() >= 3;
                bool has_bias = base_linear_has_bias;
                bool has_activation = false;
                OpType act_type = OpType::ReLU;
                for (auto& node : match.nodes) {
                    OpType op = node->op_type();
                    if (op == OpType::Add) {
                        has_bias = true;
                    } else if (is_activation_op(op)) {
                        has_activation = true;
                        act_type = op;
                    }
                }
                fused_node->set_bool_attr("fused_has_bias", has_bias);
                fused_node->set_bool_attr("fused_has_activation", has_activation);
                if (has_activation) {
                    fused_node->set_int_attr("fused_activation_type",
                                             static_cast<int64_t>(act_type));
                }
                // The fused GEMM kernel computes a plain A @ B. When the base op
                // is a Linear (y = x @ W^T), the executor must transpose the
                // second operand before the matmul, otherwise the product is
                // wrong. Record which base semantics applied.
                fused_node->set_bool_attr("fused_gemm_transpose_b",
                                          match.nodes[0]->op_type() == OpType::Linear);
                break;
            }
            case FusionKind::SmallMLP: {
                // nodes: [Linear1, activation, Linear2]. The fused MLP kernel
                // takes {w1, b1, w2, b2} params and always reads b1/b2, so record
                // the activation, the hidden dim, and whether each Linear carried
                // a bias (executor synthesizes zero bias when absent).
                const auto& lin1 = match.nodes[0];
                const auto& act = match.nodes[1];
                const auto& lin2 = match.nodes[2];
                int64_t hidden = 0;
                if (!lin1->outputs().empty() &&
                    !lin1->outputs()[0]->shape().empty()) {
                    hidden = lin1->outputs()[0]->shape().back();
                }
                fused_node->set_int_attr("fused_hidden_dim", hidden);
                fused_node->set_int_attr("fused_mlp_activation",
                                         static_cast<int64_t>(act->op_type()));
                fused_node->set_bool_attr("fused_mlp_has_bias1",
                                          lin1->inputs().size() >= 3);
                fused_node->set_bool_attr("fused_mlp_has_bias2",
                                          lin2->inputs().size() >= 3);
                break;
            }
            default:
                break;
        }

        // Persist the reduction pre/post ElemStep sequences (built + representability
        // checked above) so the executor reconstructs the exact fused reduction.
        if (match.kind == FusionKind::Reduction) {
            fused_node->set_vec_attr("fused_pre_ops", std::move(pre_ops_serial));
            fused_node->set_vec_attr("fused_post_ops", std::move(post_ops_serial));
        }

        // Replace first matched node with fused node, remove the rest.
        // Graph::replace_node only rewires values/consumers; it does not insert
        // the new node into nodes_. Register it first (mirrors the other fusion
        // passes, e.g. FlashAttention/GEMM) so it survives topo-sort/execution.
        graph.add_node(fused_node);
        graph.replace_node(match.nodes[0], fused_node);
        for (size_t i = 1; i < match.nodes.size(); ++i) {
            graph.remove_node(match.nodes[i]);
        }
        modified = true;
    }

    return modified;
}

// ============================================================================
// Symbolic Trace Pass
// ============================================================================

auto SymbolicTracePass::mark_dynamic(int input_idx, int dim, const std::string& name) -> void {
    dynamic_dims_.push_back({input_idx, dim, name});
}

auto SymbolicTracePass::run(Graph& graph) -> bool {
    if (dynamic_dims_.empty()) {
        return false;
    }

    auto& inputs = graph.inputs();
    bool changed = false;

    // Step 1: For each graph input matching a dynamic_dims_ entry, replace
    // the concrete dim with a symbolic dim in the input's symbolic shape.
    for (const auto& dd : dynamic_dims_) {
        if (dd.input_idx < 0 || static_cast<size_t>(dd.input_idx) >= inputs.size()) {
            continue;
        }

        auto& input = inputs[static_cast<size_t>(dd.input_idx)];

        // Start from existing symbolic shape, or create one from concrete shape
        SymbolicShape sym_shape = input->has_symbolic_shape()
            ? input->symbolic_shape()
            : SymbolicShape::from_concrete(input->shape());

        if (dd.dim < 0 || static_cast<size_t>(dd.dim) >= sym_shape.rank()) {
            continue;
        }

        auto& target_dim = sym_shape[static_cast<size_t>(dd.dim)];
        auto new_dim = SymbolicDim::symbolic(dd.name);

        if (target_dim != new_dim) {
            target_dim = new_dim;
            changed = true;
        }

        input->set_symbolic_shape(std::move(sym_shape));
    }

    if (!changed) {
        return false;
    }

    // Step 2: Propagate symbolic shapes through all nodes in topological order
    // using SymbolicShapeInference.
    SymbolicShapeInference inference;

    for (const auto& node : graph.nodes()) {
        auto output_shapes = inference.infer(node.get());

        auto& outputs = node->outputs();
        for (size_t i = 0; i < output_shapes.size() && i < outputs.size(); ++i) {
            if (outputs[i]->symbolic_shape() != output_shapes[i]) {
                outputs[i]->set_symbolic_shape(std::move(output_shapes[i]));
            }
        }
    }

    return true;
}

// ============================================================================
// Layout Optimization Pass
// ============================================================================

auto LayoutOptimizationPass::benefits_from_channels_last(OpType op) const -> bool {
    return op == OpType::Conv2d || op == OpType::BatchNorm2d;
}

auto LayoutOptimizationPass::is_format_agnostic(OpType op) const -> bool {
    switch (op) {
        case OpType::ReLU:
        case OpType::Sigmoid:
        case OpType::Tanh:
        case OpType::GELU:
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
        case OpType::Abs:
        case OpType::Neg:
        case OpType::Clamp:
        case OpType::Dropout:
        case OpType::ResidualAdd:
            return true;
        default:
            return false;
    }
}

auto LayoutOptimizationPass::run(Graph& graph) -> bool {
    bool changed = false;

    // Only apply channels-last layout optimization on backends whose Conv
    // kernels actually honor memory_format (CUDA/cuDNN, ROCm/MIOpen). On a
    // backend that ignores memory_format but still receives the physically
    // reordered LayoutConvert output, the Conv would read channels-last bytes
    // as NCHW and silently produce wrong results — so restrict to CUDA/ROCm.
    auto& inputs = graph.inputs();
    if (inputs.empty()) return false;

    auto device = inputs[0]->device();
    if (device.type != Device::Type::CUDA && device.type != Device::Type::ROCm) {
        return false;
    }

    // Phase 1: Identify nodes that benefit from channels-last and mark them.
    // Also propagate through format-agnostic ops connected to marked nodes.
    std::unordered_set<Node*> channels_last_nodes;

    for (auto& node : graph.nodes()) {
        if (benefits_from_channels_last(node->op_type())) {
            // Check that inputs are 4D (NCHW)
            auto& node_inputs = node->inputs();
            if (!node_inputs.empty() && node_inputs[0]->shape().size() == 4) {
                channels_last_nodes.insert(node.get());
            }
        }
    }

    if (channels_last_nodes.empty()) return false;

    // Phase 2: Propagate channels-last format through format-agnostic ops
    // that sit between two channels-last ops (avoid unnecessary conversions).
    bool propagated = true;
    while (propagated) {
        propagated = false;
        for (auto& node : graph.nodes()) {
            if (channels_last_nodes.count(node.get()) > 0) continue;
            if (!is_format_agnostic(node->op_type())) continue;

            // Check if any input comes from a channels-last node
            bool has_cl_input = false;
            for (auto& input : node->inputs()) {
                auto producer = input->node();
                if (producer && channels_last_nodes.count(producer.get()) > 0) {
                    has_cl_input = true;
                    break;
                }
            }

            // Check if any consumer is a channels-last node
            bool has_cl_consumer = false;
            for (auto& output : node->outputs()) {
                for (auto& use_wp : output->uses()) {
                    auto use = use_wp.lock();
                    if (use && channels_last_nodes.count(use.get()) > 0) {
                        has_cl_consumer = true;
                        break;
                    }
                }
                if (has_cl_consumer) break;
            }

            if (has_cl_input && has_cl_consumer) {
                channels_last_nodes.insert(node.get());
                propagated = true;
            }
        }
    }

    // Phase 3: Annotate all channels-last nodes
    for (auto& node : graph.nodes()) {
        if (channels_last_nodes.count(node.get()) > 0) {
            // 1 = ChannelsLast (matches MemoryFormat enum)
            node->set_int_attr("memory_format", 1);
            changed = true;
        }
    }

    // Phase 4: Insert LayoutConvert nodes at format boundaries.
    // A boundary exists where a non-channels-last node consumes output
    // from a channels-last node, or vice versa.
    //
    // We collect insertions first, then apply them to avoid iterator
    // invalidation.
    struct ConvertInsertion {
        std::shared_ptr<Value> value;       // Value crossing the boundary
        std::shared_ptr<Node> consumer;     // Node consuming the value
        size_t input_index;                 // Which input of consumer
        int64_t target_format;              // 0=Contiguous, 1=ChannelsLast
    };
    std::vector<ConvertInsertion> insertions;

    for (auto& node : graph.nodes()) {
        bool node_is_cl = channels_last_nodes.count(node.get()) > 0;

        for (size_t i = 0; i < node->inputs().size(); ++i) {
            auto& input_val = node->inputs()[i];
            auto producer = input_val->node();

            if (!producer) continue;  // Graph input, no conversion needed
            if (input_val->shape().size() != 4) continue;  // Only 4D tensors

            bool producer_is_cl = channels_last_nodes.count(producer.get()) > 0;

            if (producer_is_cl && !node_is_cl) {
                // channels-last -> contiguous boundary
                insertions.push_back({input_val, node, i, 0});
            } else if (!producer_is_cl && node_is_cl) {
                // contiguous -> channels-last boundary
                insertions.push_back({input_val, node, i, 1});
            }
        }
    }

    // Apply insertions
    for (auto& ins : insertions) {
        // Create a new LayoutConvert node
        auto convert_output = std::make_shared<Value>(
            ins.value->id() + "_fmt",
            ins.value->shape(),
            ins.value->dtype(),
            ins.value->device());

        auto convert_node = std::make_shared<Node>(OpType::LayoutConvert);
        convert_node->add_input(ins.value);
        convert_node->add_output(convert_output);
        convert_node->set_int_attr("target_format", ins.target_format);
        convert_output->set_node(convert_node);

        // Replace the input on the consumer. replace_input already registers
        // convert_output->add_use(consumer) (Node::replace_input calls
        // val->add_use); a second add_use here would double-count the single
        // real consumer, breaking single-use checks and over-extending
        // memory-planner liveness for the converted value.
        ins.consumer->replace_input(ins.input_index, convert_output);

        // Add the convert node to the graph
        graph.add_node(convert_node);
        changed = true;
    }

    if (changed) {
        graph.topological_sort();
    }

    return changed;
}

// ============================================================================
// DType Optimization Pass
// ============================================================================

auto DTypeOptimizationPass::is_compute_heavy(OpType op) const -> bool {
    return op == OpType::MatMul || op == OpType::Linear ||
           op == OpType::Conv2d || op == OpType::Bmm;
}

auto DTypeOptimizationPass::is_stability_critical(OpType op) const -> bool {
    switch (op) {
        case OpType::Softmax:
        case OpType::LogSoftmax:
        case OpType::LayerNorm:
        case OpType::BatchNorm2d:
        case OpType::Sum:
        case OpType::Mean:
        case OpType::Log:
        case OpType::Exp:
        case OpType::Pow:
        case OpType::Norm:
            return true;
        default:
            return false;
    }
}

auto DTypeOptimizationPass::run(Graph& graph) -> bool {
    if (!enabled_) return false;

    auto& inputs = graph.inputs();
    if (inputs.empty()) return false;

    // Apply uniformly across devices. Previously this early-returned on CPU,
    // which made the SAME compiled graph diverge CPU-vs-GPU (CPU stayed Float32
    // while GPU downcast to the target dtype). This is an explicit, opt-in
    // mixed-precision transform (not in the default pipeline and no longer
    // auto-injected by any mode), so it must behave identically regardless of
    // the target backend; CPU executes the reduced dtype via widen-compute-narrow.

    // Only optimize Float32 graphs
    bool has_fp32 = false;
    for (auto& input : inputs) {
        if (input->dtype() == DType::Float32) {
            has_fp32 = true;
            break;
        }
    }
    if (!has_fp32) return false;

    bool changed = false;

    // Track which values have been downcast
    std::unordered_set<std::string> downcast_values;

    // Collect cast insertions to avoid modifying graph during iteration
    struct CastInsertion {
        std::shared_ptr<Value> value;
        std::shared_ptr<Node> consumer;
        size_t input_index;
        DType target_dtype;
    };
    std::vector<CastInsertion> insertions;

    for (auto& node : graph.nodes()) {
        auto op = node->op_type();

        if (is_compute_heavy(op)) {
            // Downcast inputs to target_dtype for compute-heavy ops. An input
            // cast is inserted only for Float32 inputs (or inputs already in
            // the downcast set). A Float64 (or other non-F32) input gets NO
            // cast, so the kernel still produces its original dtype.
            bool any_input_downcast = false;
            for (size_t i = 0; i < node->inputs().size(); ++i) {
                auto& input_val = node->inputs()[i];
                if (downcast_values.count(input_val->id()) > 0) {
                    // Already downcast upstream — output will be in target dtype.
                    any_input_downcast = true;
                } else if (input_val->dtype() == DType::Float32) {
                    insertions.push_back({input_val, node, i, target_dtype_});
                    any_input_downcast = true;
                }
            }
            // Only relabel the output (and treat it as downcast downstream) when
            // at least one input was actually cast to target_dtype. Relabeling
            // unconditionally would desync the value's dtype label from what the
            // kernel produces for ops with no Float32 input (e.g. a pure Float64
            // matmul), corrupting downstream upcast / output-cast decisions that
            // key off downcast_values.
            if (any_input_downcast) {
                for (auto& output : node->outputs()) {
                    output->set_dtype(target_dtype_);
                    downcast_values.insert(output->id());
                }
                changed = true;
            }
        } else if (is_stability_critical(op)) {
            // Upcast any downcast inputs back to Float32
            for (size_t i = 0; i < node->inputs().size(); ++i) {
                auto& input_val = node->inputs()[i];
                if (downcast_values.count(input_val->id()) > 0) {
                    insertions.push_back({input_val, node, i, DType::Float32});
                }
            }
            // Outputs remain Float32
            for (auto& output : node->outputs()) {
                downcast_values.erase(output->id());
            }
        } else {
            // Neutral ops (e.g. elementwise Add/Sub/Mul/Div): inherit dtype from
            // the primary input. These ops may be multi-input, so relabeling the
            // output to target_dtype_ based on the primary input alone would
            // desync the label from the actual computed dtype whenever a sibling
            // input is still Float32 (or some other dtype) — corrupting the
            // downstream upcast/output-cast decisions keyed off downcast_values.
            //
            // To keep the label honest we mirror the compute-heavy branch:
            // when the primary input is downcast, every Float32 sibling is cast
            // to target_dtype_ so the whole op truly runs in the downcast dtype.
            // If a sibling carries a dtype that is neither downcast nor Float32
            // (e.g. Float64), the operands cannot be unified safely, so we leave
            // the output label untouched (it keeps its original dtype).
            if (!node->inputs().empty() &&
                downcast_values.count(node->inputs()[0]->id()) > 0) {
                bool can_downcast = true;
                for (size_t i = 0; i < node->inputs().size(); ++i) {
                    auto& input_val = node->inputs()[i];
                    if (downcast_values.count(input_val->id()) > 0) continue;
                    if (input_val->dtype() != DType::Float32) {
                        can_downcast = false;
                        break;
                    }
                }
                if (can_downcast) {
                    for (size_t i = 0; i < node->inputs().size(); ++i) {
                        auto& input_val = node->inputs()[i];
                        if (downcast_values.count(input_val->id()) > 0) continue;
                        // Remaining non-downcast inputs are all Float32 here.
                        insertions.push_back({input_val, node, i, target_dtype_});
                    }
                    for (auto& output : node->outputs()) {
                        output->set_dtype(target_dtype_);
                        downcast_values.insert(output->id());
                    }
                    changed = true;
                }
            }
        }
    }

    // Audit J4: insert Cast nodes at graph outputs to bring downcast (Float16/
    // BFloat16) values back to Float32. Previously this branch only marked
    // the insertions and then the apply-loop bailed on the `nullptr consumer`
    // case (`// Skip output-only casts for now`). The result was that
    // downcast outputs leaked Float16 to callers that expected Float32 —
    // breaking eager/JIT parity for autocast-traced models.
    //
    // Phase 1: identify which graph outputs need casting. Build a map from
    //   {original value id → cast value id} so we can rewrite the outputs
    //   vector after we walk through it.
    std::unordered_map<std::string, std::shared_ptr<Value>> output_casts;
    for (auto& output : graph.outputs()) {
        if (downcast_values.count(output->id()) > 0 && output->node()) {
            auto cast_output = std::make_shared<Value>(
                output->id() + "_cast_out",
                output->shape(),
                DType::Float32,
                output->device());

            auto cast_node = std::make_shared<Node>(OpType::Cast);
            // add_input already registers output->add_use(cast_node)
            // (Node::add_input calls val->add_use). A second add_use here would
            // double-count, inflating output->uses().size() to 2 for a single
            // real consumer and breaking single-use checks (PatternMatcher::
            // has_single_use) plus over-extending memory-planner liveness — the
            // same hazard the input-side path below documents avoiding.
            cast_node->add_input(output);
            cast_node->add_output(cast_output);
            cast_node->set_int_attr("target_dtype", static_cast<int64_t>(DType::Float32));
            cast_output->set_node(cast_node);

            graph.add_node(cast_node);
            output_casts[output->id()] = cast_output;
            changed = true;
        }
    }
    // Phase 2: rewrite graph outputs to point at the cast values.
    if (!output_casts.empty()) {
        std::vector<std::shared_ptr<Value>> new_outputs;
        new_outputs.reserve(graph.outputs().size());
        for (auto& output : graph.outputs()) {
            auto it = output_casts.find(output->id());
            new_outputs.push_back(it != output_casts.end() ? it->second : output);
        }
        graph.set_outputs(std::move(new_outputs));
    }

    // Apply input-side cast insertions (downcast-to-target for compute-heavy
    // ops + upcast-to-Float32 for stability-critical ops). Output-side casts
    // are handled separately above.
    size_t cast_uid = 0;
    for (auto& ins : insertions) {
        if (!ins.consumer) continue;  // Output-only entries: handled above.

        // Make the cast value id unique per insertion. When one source value
        // feeds several compute-heavy consumers, this loop emits one cast per
        // (value, consumer); a shared "<id>_cast" id would collide so that
        // GraphWriter::write_values (which dedups Values by id) collapses the
        // distinct casts into one record, breaking save/load of dtype-optimized
        // graphs ("references input value that does not exist"). Suffix with an
        // incrementing counter to guarantee uniqueness.
        auto cast_output = std::make_shared<Value>(
            ins.value->id() + "_cast_" + std::to_string(cast_uid++),
            ins.value->shape(),
            ins.target_dtype,
            ins.value->device());

        auto cast_node = std::make_shared<Node>(OpType::Cast);
        cast_node->add_input(ins.value);
        cast_node->add_output(cast_output);
        cast_node->set_int_attr("target_dtype", static_cast<int64_t>(ins.target_dtype));
        cast_output->set_node(cast_node);

        // replace_input already registers `cast_output`'s use of the consumer
        // (Node::replace_input calls val->add_use). A second add_use here would
        // double-count, inflating cast_output->uses().size() to 2 for a single
        // real consumer and breaking single-use checks (PatternMatcher::
        // has_single_use) that suppress legitimate cast-chain fusion/elimination.
        ins.consumer->replace_input(ins.input_index, cast_output);

        graph.add_node(cast_node);
    }

    if (changed) {
        graph.topological_sort();
    }

    return changed;
}

// ============================================================================
// Compiler Implementation
// ============================================================================

Compiler::Compiler(bool enable_default_passes) {
    if (enable_default_passes) {
        add_pass(std::make_unique<DeadCodeEliminationPass>());
        add_pass(std::make_unique<ConstantFoldingPass>());
        add_pass(std::make_unique<StrengthReductionPass>());      // Div→Mul, Pow→Mul/Sqrt
        add_pass(std::make_unique<FuseConvBatchNormReluPass>());  // Triple fusion before individual passes
        add_pass(std::make_unique<FuseConvBatchNormPass>());
        add_pass(std::make_unique<FuseConvReluPass>());
        add_pass(std::make_unique<FuseLinearReluPass>());
        add_pass(std::make_unique<FuseMatMulAddPass>());
        add_pass(std::make_unique<FuseLayerNormActivationPass>());
        add_pass(std::make_unique<FuseAttentionPass>());        // Transformer attention fusion
        add_pass(std::make_unique<FuseFFNPass>());              // FFN fusion (Linear->Act->Linear)
        add_pass(std::make_unique<FuseResidualAddPass>());      // Residual connection marking
        add_pass(std::make_unique<ExtendedFusionPass>());       // Multi-node extended fusion (reduction, softmax, norm, MLP)
        add_pass(std::make_unique<AlgebraicSimplificationPass>());
        add_pass(std::make_unique<ReshapeEliminationPass>());
        add_pass(std::make_unique<CommonSubexpressionEliminationPass>());
        add_pass(std::make_unique<LoopUnrollingPass>());
        add_pass(std::make_unique<LICMPass>());
        add_pass(std::make_unique<ShapeGuardInsertionPass>());  // Dynamic shape guards (runs once)
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

    // Passes that append nodes (ConstantFolding, StrengthReduction, fusion,
    // LoopUnroll) push them onto the end of nodes_ without restoring dependency
    // order. Graph::forward executes nodes_ in stored order, so a newly appended
    // producer could run after its consumer. Re-sort once here after all passes.
    graph.topological_sort();

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
// QuantizationPass — Post-training INT8 quantization
// ============================================================================

auto QuantizationPass::compute_scale_and_zero(const Tensor& weight)
    -> std::pair<float, int64_t> {
    // Symmetric quantization: scale = max(|weight|) / 127
    auto abs_weight = tenzor::abs(weight).to(DType::Float32);
    // item<T>() performs the documented device->host copy; data<float>()[0] on a
    // device tensor would dereference a device pointer on the host (UB on GPU).
    float max_val = tenzor::max(abs_weight).item<float>();
    if (max_val == 0.0f) max_val = 1.0f;  // avoid division by zero
    float scale = max_val / 127.0f;
    return {scale, 0};  // symmetric quantization: zero_point = 0
}

auto QuantizationPass::quantize_linear(std::shared_ptr<Node> node, Graph& /*graph*/) -> bool {
    // A traced Linear carries its operands as INPUTS in dispatch order:
    // [x, weight(, bias)] — the weight is a captured-parameter constant at
    // input[1], NOT a node attribute. Retag the node to QuantizedLinear keeping
    // the same inputs/outputs; the JIT interpreter (Graph::execute_node) runs it
    // via nn::quantization::quantized_linear_dynamic, which dynamically quantizes
    // the weight+activation to int8 and dispatches OpId::QuantizedLinear (a real
    // kernel on every backend), matching eager nn::QuantizedLinear numerically.
    // int8 is a deliberate lossy approximation of the fp32 Linear.
    if (node->inputs().size() < 2) return false;  // need at least [x, weight]

    // Retag IN PLACE (keep inputs/outputs/identity). Creating a replacement node
    // that reuses the same output Value objects breaks graph.replace_node (its
    // replace_value(old,new) is a self no-op and remove_node then orphans the
    // shared output), leaving the graph output uncomputed.
    node->set_op_type(OpType::QuantizedLinear);
    node->set_bool_attr("quantized", true);
    return true;
}

auto QuantizationPass::quantize_conv2d(std::shared_ptr<Node> node, Graph& graph) -> bool {
    // A traced Conv2d carries [x, weight(, bias)] as inputs and its
    // stride/padding/dilation/groups as node attrs. Retag IN PLACE to
    // QuantizedConv2d (keeping inputs/outputs/attrs); the JIT interpreter runs
    // it via nn::quantization::quantized_conv2d_dynamic. (See quantize_linear
    // for why in-place beats a replacement node.)
    (void)graph;
    if (node->inputs().size() < 2) return false;  // need at least [x, weight]
    node->set_op_type(OpType::QuantizedConv2d);
    node->set_bool_attr("quantized", true);
    return true;
}

auto QuantizationPass::run(Graph& graph) -> bool {
    bool modified = false;

    // Backend gating: only quantize an op when the TARGET backend actually
    // provides the quantized kernel. Rewriting Linear/Conv2d to their quantized
    // form on a backend that lacks QuantizedLinear/QuantizedConv2d would leave
    // the graph referencing a missing/divergent kernel. Skipping leaves the op
    // in its correct dense form (runs identically everywhere) — this is not a
    // CPU fallback, it is declining an unsupported optimization on that device.
    if (graph.inputs().empty()) return false;
    const Device::Type dev = graph.inputs()[0]->device().type;
    const bool has_qlinear = is_op_supported(OpId::QuantizedLinear, dev);
    const bool has_qconv   = is_op_supported(OpId::QuantizedConv2d, dev);

    // Collect nodes to quantize (can't modify graph while iterating)
    std::vector<std::shared_ptr<Node>> linear_nodes;
    std::vector<std::shared_ptr<Node>> conv_nodes;

    for (auto& node : graph.nodes()) {
        if (has_qlinear && node->op_type() == OpType::Linear && !node->get_bool_attr("quantized")) {
            linear_nodes.push_back(node);
        } else if (has_qconv && node->op_type() == OpType::Conv2d && !node->get_bool_attr("quantized")) {
            conv_nodes.push_back(node);
        }
    }

    for (auto& node : linear_nodes) {
        if (quantize_linear(node, graph)) modified = true;
    }
    for (auto& node : conv_nodes) {
        if (quantize_conv2d(node, graph)) modified = true;
    }

    return modified;
}

// ============================================================================
// SparsePass — Sparse weight optimization
// ============================================================================

auto SparsePass::compute_sparsity(const Tensor& weight) -> float {
    // Move to the host before the linear scan: data<float>() is a raw host
    // pointer, so iterating it over a device tensor is UB on GPU backends.
    // contiguous() guards the flat index.
    auto w_f32 = weight.to(DType::Float32).to(Device::cpu()).contiguous();
    int64_t total = w_f32.numel();
    if (total == 0) return 0.0f;

    // Count zeros
    const float* data = w_f32.data<float>();
    int64_t zero_count = 0;
    for (int64_t i = 0; i < total; ++i) {
        if (data[i] == 0.0f) ++zero_count;
    }
    return static_cast<float>(zero_count) / static_cast<float>(total);
}

auto SparsePass::convert_to_sparse(std::shared_ptr<Node> node, Graph& graph) -> bool {
    // A traced Linear carries [x, weight(, bias)] as inputs; the weight is a
    // captured constant at input[1]. Read it to measure sparsity and retag IN
    // PLACE to SparseMatMul only when the weight is sparse enough to benefit —
    // the interpreter runs it via tenzor::sparse::sparse_matmul_dynamic
    // (spmm(from_dense(W), xᵀ)ᵀ + bias), which is EXACT (lossless), so a dense
    // weight would still be correct, just slower; hence the threshold gate.
    // (In-place retag; see quantize_linear for why replacement nodes break.)
    if (node->inputs().size() < 2) return false;
    const auto& consts = graph.constants();
    auto it = consts.find(node->inputs()[1]->id());
    if (it == consts.end()) return false;  // weight not a known compile-time constant
    if (compute_sparsity(it->second) < threshold_) return false;

    node->set_op_type(OpType::SparseMatMul);
    node->set_bool_attr("is_sparse", true);
    return true;
}

auto SparsePass::run(Graph& graph) -> bool {
    bool modified = false;

    // Backend gating: the SparseMatMul executor dispatches SparseSpMM. Only
    // convert Linear->SparseMatMul on a backend that provides that kernel;
    // otherwise leave the dense Linear (correct everywhere). Not a CPU fallback.
    if (graph.inputs().empty()) return false;
    if (!is_op_supported(OpId::SparseSpMM, graph.inputs()[0]->device().type)) {
        return false;
    }

    // Collect candidate nodes
    std::vector<std::shared_ptr<Node>> candidates;
    for (auto& node : graph.nodes()) {
        // Target traced Linear nodes (x @ Wᵀ) — the SparseMatMul executor is
        // Linear-shaped. A raw MatMul (x @ W, no transpose) would need a
        // different orientation, so it is intentionally excluded.
        if (node->op_type() == OpType::Linear &&
            !node->get_bool_attr("is_sparse") &&
            node->inputs().size() >= 2) {
            candidates.push_back(node);
        }
    }

    for (auto& node : candidates) {
        if (convert_to_sparse(node, graph)) modified = true;
    }

    return modified;
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
