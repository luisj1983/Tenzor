/**
 * @file tracer.cpp
 * @brief Implementation of JIT operation tracing
 */

#include "../../include/tenzor/jit/tracer.hpp"
#include "../../include/tenzor/jit/compiler.hpp"
#include "../../include/tenzor/jit/tracing_interceptor.hpp"
#include "../../include/tenzor/backend/dispatch_interceptor.hpp"
#include "../../include/tenzor/core/jit_hooks.hpp"
#include "../../include/tenzor/ops/creation.hpp"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <sstream>
#include <iomanip>
#include <stdexcept>

namespace tenzor {
namespace jit {

// ============================================================================
// OpType conversion functions
// ============================================================================

auto op_type_to_string(OpType type) -> std::string {
    switch (type) {
        case OpType::Add: return "Add";
        case OpType::Sub: return "Sub";
        case OpType::Mul: return "Mul";
        case OpType::Div: return "Div";
        case OpType::MatMul: return "MatMul";
        case OpType::Bmm: return "Bmm";
        case OpType::ReLU: return "ReLU";
        case OpType::Sigmoid: return "Sigmoid";
        case OpType::Tanh: return "Tanh";
        case OpType::Softmax: return "Softmax";
        case OpType::LogSoftmax: return "LogSoftmax";
        case OpType::MaxPool2d: return "MaxPool2d";
        case OpType::AvgPool2d: return "AvgPool2d";
        case OpType::AdaptiveAvgPool2d: return "AdaptiveAvgPool2d";
        case OpType::Conv2d: return "Conv2d";
        case OpType::ConvTranspose: return "ConvTranspose";
        case OpType::BatchNorm2d: return "BatchNorm2d";
        case OpType::LayerNorm: return "LayerNorm";
        case OpType::Reshape: return "Reshape";
        case OpType::Transpose: return "Transpose";
        case OpType::Permute: return "Permute";
        case OpType::Squeeze: return "Squeeze";
        case OpType::Unsqueeze: return "Unsqueeze";
        case OpType::Flatten: return "Flatten";
        case OpType::Sum: return "Sum";
        case OpType::Mean: return "Mean";
        case OpType::Max: return "Max";
        case OpType::Min: return "Min";
        case OpType::Exp: return "Exp";
        case OpType::Log: return "Log";
        case OpType::Sqrt: return "Sqrt";
        case OpType::Pow: return "Pow";
        case OpType::Abs: return "Abs";
        case OpType::Neg: return "Neg";
        case OpType::Clamp: return "Clamp";
        case OpType::Sin: return "Sin";
        case OpType::Cos: return "Cos";
        case OpType::Rsqrt: return "Rsqrt";
        case OpType::Slice: return "Slice";
        case OpType::Cat: return "Cat";
        case OpType::Dropout: return "Dropout";
        case OpType::Linear: return "Linear";
        case OpType::Embedding: return "Embedding";
        case OpType::GELU: return "GELU";
        case OpType::Det: return "Det";
        case OpType::Inv: return "Inv";
        case OpType::Solve: return "Solve";
        case OpType::Cholesky: return "Cholesky";
        case OpType::Svd: return "Svd";
        case OpType::Qr: return "Qr";
        case OpType::Eigh: return "Eigh";
        case OpType::Eigvalsh: return "Eigvalsh";
        case OpType::Norm: return "Norm";
        case OpType::Slogdet: return "Slogdet";
        case OpType::FlashAttention: return "FlashAttention";
        case OpType::FusedFFN: return "FusedFFN";
        case OpType::ResidualAdd: return "ResidualAdd";
        case OpType::ShapeGuard: return "ShapeGuard";
        case OpType::SwapOut: return "SwapOut";
        case OpType::SwapIn: return "SwapIn";
        case OpType::Constant: return "Constant";
        case OpType::Input: return "Input";
        case OpType::Output: return "Output";
        case OpType::If: return "If";
        case OpType::Loop: return "Loop";
        case OpType::LayoutConvert: return "LayoutConvert";
        case OpType::Cast: return "Cast";
        // Phase 13 / MVP-1 additions
        case OpType::SiLU: return "SiLU";
        case OpType::Where: return "Where";
        case OpType::Stack: return "Stack";
        case OpType::Broadcast: return "Broadcast";
        case OpType::IndexSelect: return "IndexSelect";
        case OpType::RMSNorm: return "RMSNorm";
        case OpType::GQA: return "GQA";
        case OpType::RoPE: return "RoPE";
        case OpType::Padding: return "Padding";
        case OpType::Interpolate: return "Interpolate";
        default: return "Unknown";
    }
}

auto string_to_op_type(const std::string& str) -> OpType {
    static const std::unordered_map<std::string, OpType> string_to_type = {
        {"Add", OpType::Add},
        {"Sub", OpType::Sub},
        {"Mul", OpType::Mul},
        {"Div", OpType::Div},
        {"MatMul", OpType::MatMul},
        {"Bmm", OpType::Bmm},
        {"ReLU", OpType::ReLU},
        {"Sigmoid", OpType::Sigmoid},
        {"Tanh", OpType::Tanh},
        {"Softmax", OpType::Softmax},
        {"LogSoftmax", OpType::LogSoftmax},
        {"MaxPool2d", OpType::MaxPool2d},
        {"AvgPool2d", OpType::AvgPool2d},
        {"AdaptiveAvgPool2d", OpType::AdaptiveAvgPool2d},
        {"Conv2d", OpType::Conv2d},
        {"ConvTranspose", OpType::ConvTranspose},
        {"BatchNorm2d", OpType::BatchNorm2d},
        {"LayerNorm", OpType::LayerNorm},
        {"Reshape", OpType::Reshape},
        {"Transpose", OpType::Transpose},
        {"Permute", OpType::Permute},
        {"Squeeze", OpType::Squeeze},
        {"Unsqueeze", OpType::Unsqueeze},
        {"Flatten", OpType::Flatten},
        {"Sum", OpType::Sum},
        {"Mean", OpType::Mean},
        {"Max", OpType::Max},
        {"Min", OpType::Min},
        {"Exp", OpType::Exp},
        {"Log", OpType::Log},
        {"Sqrt", OpType::Sqrt},
        {"Pow", OpType::Pow},
        {"Abs", OpType::Abs},
        {"Neg", OpType::Neg},
        {"Clamp", OpType::Clamp},
        {"Sin", OpType::Sin},
        {"Cos", OpType::Cos},
        {"Rsqrt", OpType::Rsqrt},
        {"Slice", OpType::Slice},
        {"Cat", OpType::Cat},
        {"Dropout", OpType::Dropout},
        {"Linear", OpType::Linear},
        {"Embedding", OpType::Embedding},
        {"GELU", OpType::GELU},
        {"Det", OpType::Det},
        {"Inv", OpType::Inv},
        {"Solve", OpType::Solve},
        {"Cholesky", OpType::Cholesky},
        {"Svd", OpType::Svd},
        {"Qr", OpType::Qr},
        {"Eigh", OpType::Eigh},
        {"Eigvalsh", OpType::Eigvalsh},
        {"Norm", OpType::Norm},
        {"Slogdet", OpType::Slogdet},
        {"FlashAttention", OpType::FlashAttention},
        {"FusedFFN", OpType::FusedFFN},
        {"ResidualAdd", OpType::ResidualAdd},
        {"ShapeGuard", OpType::ShapeGuard},
        {"SwapOut", OpType::SwapOut},
        {"SwapIn", OpType::SwapIn},
        {"Constant", OpType::Constant},
        {"Input", OpType::Input},
        {"Output", OpType::Output},
        {"If", OpType::If},
        {"Loop", OpType::Loop},
        {"LayoutConvert", OpType::LayoutConvert},
        {"Cast", OpType::Cast},
        // Phase 13 / MVP-1 additions
        {"SiLU", OpType::SiLU},
        {"Where", OpType::Where},
        {"Stack", OpType::Stack},
        {"Broadcast", OpType::Broadcast},
        {"IndexSelect", OpType::IndexSelect},
        {"RMSNorm", OpType::RMSNorm},
        {"GQA", OpType::GQA},
        {"RoPE", OpType::RoPE},
        {"Padding", OpType::Padding},
        {"Interpolate", OpType::Interpolate},
    };

    auto it = string_to_type.find(str);
    if (it == string_to_type.end()) {
        throw std::runtime_error("Unknown operation type: " + str);
    }
    return it->second;
}

// ============================================================================
// Tracer implementation
// ============================================================================

namespace {
// Read TENZOR_JIT_STRICT from the environment. Accepts any non-empty
// string that is not "0" / "false" as "enabled".
bool read_strict_mode_from_env() {
    const char* env = std::getenv("TENZOR_JIT_STRICT");
    if (env == nullptr) return false;
    if (env[0] == '\0') return false;
    if (std::strcmp(env, "0") == 0) return false;
    if (std::strcmp(env, "false") == 0) return false;
    if (std::strcmp(env, "False") == 0) return false;
    if (std::strcmp(env, "FALSE") == 0) return false;
    return true;
}
} // anonymous namespace

auto Tracer::start_trace() -> void {
    clear();
    tracing_ = true;
    // Default strict mode from environment — can be overridden after
    // start_trace() via set_strict_mode().
    strict_mode_ = read_strict_mode_from_env();
    graph_break_count_ = 0;
}

auto Tracer::record_graph_break(const std::string& reason) -> void {
    if (!tracing_) return;
    ++graph_break_count_;

    const std::string msg =
        std::string("tenzor::jit tracer graph break: ") + reason +
        ". The captured graph bakes in whichever branch/value was taken "
        "at trace time and will be incorrect for inputs that would take a "
        "different path. Replace with tenzor::jit::cond / jit::while_loop "
        "to record both sides of the branch, or keep the op outside the "
        "traced region.";

    if (strict_mode_) {
        // Turn tracing off so the thrown exception unwinds cleanly
        // through the tracing guard.
        tracing_ = false;
        throw std::runtime_error(msg);
    }
    std::fprintf(stderr, "[tenzor.jit] warning: %s\n", msg.c_str());
}

auto Tracer::end_trace(const std::vector<Variable>& inputs,
                       const std::vector<Variable>& outputs) -> std::shared_ptr<Graph> {
    tracing_ = false;

    // Create graph
    auto graph = std::make_shared<Graph>();

    // Forward the declared trainable parameters onto the graph so replay can
    // rebind the live Variables (see classify_captured below and
    // Graph::forward's parameter-leaf binding).
    graph->set_parameters(parameters_);

    // Classify a non-produced op input (one that has no producing node and is
    // not a graph input): a captured module PARAMETER (storage matches a
    // declared parameter) becomes a parameter leaf bound to the live Variable;
    // anything else is frozen as an opaque constant. This is the single point
    // where a closure-captured parameter stops being baked as a frozen value.
    auto classify_captured = [&](Graph& g, const std::string& input_id) {
        auto storage_it = tensor_storage_.find(input_id);
        if (storage_it == tensor_storage_.end()) return;
        const void* ident = storage_it->second.data_ptr();
        auto pit = param_storage_index_.find(ident);
        if (pit != param_storage_index_.end()) {
            g.add_param_leaf(input_id, pit->second);
        } else {
            g.set_constant(input_id, storage_it->second);
        }
    };

    // Map tensor IDs to graph values
    std::unordered_map<std::string, std::shared_ptr<Value>> value_map;

    // Create input values
    std::vector<std::shared_ptr<Value>> graph_inputs;
    for (const auto& input : inputs) {
        auto tensor_id = register_tensor(input);
        const auto& info = tensor_info_[tensor_id];
        auto value = graph->create_value(tensor_id, info.shape, info.dtype, info.device);
        value_map[tensor_id] = value;
        graph_inputs.push_back(value);
    }
    graph->set_inputs(graph_inputs);

    // trace_if and trace_loop record the branch/body ops INLINE before the
    // control-flow op itself. To give the graph executor real subgraph
    // dispatch, we bundle those inlined slices into sub-Graphs and skip
    // them in the main graph.
    //
    // The attrs stored on If ops: then_ops_start / then_ops_end /
    //                             else_ops_start / else_ops_end.
    // On Loop: body_ops_start / body_ops_end.
    std::vector<bool> skip(ops_.size(), false);
    for (size_t i = 0; i < ops_.size(); ++i) {
        const auto& op = ops_[i];
        if (op.type == OpType::If) {
            auto a = op.int_attrs.find("then_ops_start");
            auto b = op.int_attrs.find("then_ops_end");
            auto c = op.int_attrs.find("else_ops_end");
            if (a != op.int_attrs.end() && b != op.int_attrs.end() &&
                c != op.int_attrs.end()) {
                int64_t ts = a->second, te = b->second, ee = c->second;
                for (int64_t k = ts; k < te && k >= 0 && (size_t)k < ops_.size(); ++k) skip[k] = true;
                for (int64_t k = te; k < ee && k >= 0 && (size_t)k < ops_.size(); ++k) skip[k] = true;
            }
        } else if (op.type == OpType::Loop) {
            auto a = op.int_attrs.find("body_ops_start");
            auto b = op.int_attrs.find("body_ops_end");
            if (a != op.int_attrs.end() && b != op.int_attrs.end()) {
                int64_t s = a->second, e = b->second;
                for (int64_t k = s; k < e && k >= 0 && (size_t)k < ops_.size(); ++k) skip[k] = true;
            }
        }
    }

    // Build the loop-body output list for a Loop op: the executor convention is
    // `body outputs = [cond, carried...]`, so the loop condition (if recorded)
    // is surfaced as the FIRST body output. Shared between the outermost Loop
    // attach below and the recursive nested-Loop attach inside build_subgraph.
    auto loop_body_output_ids = [](const TracedOp& op) -> std::vector<std::string> {
        std::vector<std::string> ids;
        if (!op.loop_cond_output.empty()) ids.push_back(op.loop_cond_output);
        ids.insert(ids.end(), op.outputs.begin(), op.outputs.end());
        return ids;
    };

    // Recursive: a nested If/Loop inside a branch/body slice must itself be
    // built as a nested sub-graph (its inner branch/body ops attached via
    // set_then_branch/set_else_branch/set_body), NOT flattened inline into the
    // enclosing sub-graph. std::function so the lambda can call itself.
    std::function<std::shared_ptr<Graph>(int64_t, int64_t,
                                         const std::vector<std::string>&,
                                         const std::vector<std::string>&)>
        build_subgraph;
    build_subgraph = [&](int64_t start, int64_t end,
                         const std::vector<std::string>& carried_ids,
                         const std::vector<std::string>& output_ids)
            -> std::shared_ptr<Graph> {
        auto sub = std::make_shared<Graph>();
        // Forward the declared trainable parameters onto the sub-graph too, so a
        // closure-captured parameter used INSIDE an If branch / Loop body is
        // rebound to the live Variable at replay (grad flows to param->grad(),
        // and inference sees updated weights) rather than frozen as a constant.
        sub->set_parameters(parameters_);
        std::unordered_map<std::string, std::shared_ptr<Value>> sub_values;

        // Carried inputs become sub-graph inputs.
        std::vector<std::shared_ptr<Value>> sub_inputs;
        for (const auto& id : carried_ids) {
            const auto& info = tensor_info_[id];
            auto v = sub->create_value(id, info.shape, info.dtype, info.device);
            sub_values[id] = v;
            sub_inputs.push_back(v);
        }
        sub->set_inputs(sub_inputs);

        // A nested If/Loop records its own branch/body ops INLINE in the global
        // op stream just before the control-flow op itself. Those inner ops must
        // not be replayed flat into THIS sub-graph — they belong to the nested
        // node's own sub-graph(s) and are attached recursively below. Mark their
        // index ranges so the replay loop skips them. (Only the directly-nested
        // ranges need marking; deeper nesting is handled by the recursion.)
        std::vector<bool> inline_skip(ops_.size(), false);
        for (int64_t k = start; k < end && k >= 0 && (size_t)k < ops_.size(); ++k) {
            const auto& nop = ops_[k];
            if (nop.type == OpType::If) {
                auto a = nop.int_attrs.find("then_ops_start");
                auto c = nop.int_attrs.find("else_ops_end");
                if (a != nop.int_attrs.end() && c != nop.int_attrs.end()) {
                    for (int64_t j = a->second; j < c->second && j >= 0 &&
                                                (size_t)j < ops_.size(); ++j)
                        inline_skip[j] = true;
                }
            } else if (nop.type == OpType::Loop) {
                auto a = nop.int_attrs.find("body_ops_start");
                auto b = nop.int_attrs.find("body_ops_end");
                if (a != nop.int_attrs.end() && b != nop.int_attrs.end()) {
                    for (int64_t j = a->second; j < b->second && j >= 0 &&
                                                (size_t)j < ops_.size(); ++j)
                        inline_skip[j] = true;
                }
            }
        }

        // Replay the ops slice inside the sub-graph.
        for (int64_t k = start; k < end && k >= 0 && (size_t)k < ops_.size(); ++k) {
            if (inline_skip[k]) continue;
            const auto& op = ops_[k];
            auto node = sub->create_node(op.type);
            for (const auto& iid : op.inputs) {
                auto it = sub_values.find(iid);
                if (it != sub_values.end()) {
                    node->add_input(it->second);
                } else {
                    const auto& info = tensor_info_[iid];
                    auto v = sub->create_value(iid, info.shape, info.dtype, info.device);
                    sub_values[iid] = v;
                    node->add_input(v);
                    // Same parameter-vs-constant classification as the top-level
                    // graph: a captured module parameter becomes a parameter leaf
                    // (live-rebound at replay); anything else is a frozen constant.
                    classify_captured(*sub, iid);
                }
            }
            for (const auto& oid : op.outputs) {
                const auto& info = tensor_info_[oid];
                auto v = sub->create_value(oid, info.shape, info.dtype, info.device);
                v->set_node(node);
                node->add_output(v);
                sub_values[oid] = v;
            }
            for (const auto& [n, val] : op.attrs)       node->set_attr(n, val);
            for (const auto& [n, val] : op.int_attrs)   node->set_int_attr(n, val);
            for (const auto& [n, val] : op.vec_attrs)   node->set_vec_attr(n, val);
            for (const auto& [n, val] : op.bool_attrs)  node->set_bool_attr(n, val);
            for (const auto& [n, val] : op.tensor_attrs) node->set_tensor_attr(n, val);

            // Recursively nest inner control-flow subgraphs instead of
            // flattening them (mirrors the outermost attach in end_trace).
            if (op.type == OpType::If) {
                auto ts = op.int_attrs.find("then_ops_start");
                auto te = op.int_attrs.find("then_ops_end");
                auto es = op.int_attrs.find("else_ops_start");
                auto ee = op.int_attrs.find("else_ops_end");
                if (ts != op.int_attrs.end() && te != op.int_attrs.end() &&
                    es != op.int_attrs.end() && ee != op.int_attrs.end()) {
                    // op.inputs[0] is the condition, inputs[1..] are carried.
                    std::vector<std::string> inner_carried(op.inputs.begin() + 1,
                                                           op.inputs.end());
                    node->set_then_branch(build_subgraph(
                        ts->second, te->second, inner_carried, op.outputs));
                    const auto& inner_else_ids = op.else_outputs.empty()
                                                     ? op.outputs
                                                     : op.else_outputs;
                    node->set_else_branch(build_subgraph(
                        es->second, ee->second, inner_carried, inner_else_ids));
                }
            } else if (op.type == OpType::Loop) {
                auto bs = op.int_attrs.find("body_ops_start");
                auto be = op.int_attrs.find("body_ops_end");
                if (bs != op.int_attrs.end() && be != op.int_attrs.end()) {
                    node->set_body(build_subgraph(
                        bs->second, be->second, op.inputs, loop_body_output_ids(op)));
                }
            }
            sub->add_node(node);
        }

        // Outputs of the sub-graph are the output tensor IDs the caller
        // asked to surface. If an output didn't get produced inside the
        // slice (branch leaves a variable unchanged), fall back to the
        // matching carried input value.
        std::vector<std::shared_ptr<Value>> sub_outputs;
        for (const auto& oid : output_ids) {
            auto it = sub_values.find(oid);
            if (it != sub_values.end()) sub_outputs.push_back(it->second);
        }
        sub->set_outputs(sub_outputs);
        sub->topological_sort();
        sub->infer_types();
        return sub;
    };

    // Process recorded operations
    for (size_t op_idx = 0; op_idx < ops_.size(); ++op_idx) {
        if (skip[op_idx]) continue;
        const auto& op = ops_[op_idx];
        // Create node
        auto node = graph->create_node(op.type);

        // Add inputs
        for (const auto& input_id : op.inputs) {
            auto it = value_map.find(input_id);
            if (it != value_map.end()) {
                node->add_input(it->second);
            } else {
                const auto& info = tensor_info_[input_id];
                auto value = graph->create_value(input_id, info.shape, info.dtype, info.device);
                value_map[input_id] = value;
                node->add_input(value);
                classify_captured(*graph, input_id);
            }
        }

        // Add outputs
        for (const auto& output_id : op.outputs) {
            const auto& info = tensor_info_[output_id];
            auto value = graph->create_value(output_id, info.shape, info.dtype, info.device);
            value->set_node(node);
            node->add_output(value);
            value_map[output_id] = value;
        }

        // Copy attributes
        for (const auto& [name, val] : op.attrs)       node->set_attr(name, val);
        for (const auto& [name, val] : op.int_attrs)   node->set_int_attr(name, val);
        for (const auto& [name, val] : op.vec_attrs)   node->set_vec_attr(name, val);
        for (const auto& [name, val] : op.bool_attrs)  node->set_bool_attr(name, val);
        for (const auto& [name, val] : op.tensor_attrs) node->set_tensor_attr(name, val);

        // Attach subgraphs for control-flow nodes.
        if (op.type == OpType::If) {
            auto ts = op.int_attrs.find("then_ops_start");
            auto te = op.int_attrs.find("then_ops_end");
            auto es = op.int_attrs.find("else_ops_start");
            auto ee = op.int_attrs.find("else_ops_end");
            if (ts != op.int_attrs.end() && te != op.int_attrs.end() &&
                es != op.int_attrs.end() && ee != op.int_attrs.end()) {
                // op.inputs[0] is the condition, inputs[1..] are carried.
                std::vector<std::string> carried(op.inputs.begin() + 1, op.inputs.end());
                node->set_then_branch(build_subgraph(
                    ts->second, te->second, carried, op.outputs));
                // The else branch produces its own outputs; fall back to
                // the then-branch output ids if the tracer didn't record
                // them (older traces).
                const auto& else_out_ids = op.else_outputs.empty()
                                               ? op.outputs
                                               : op.else_outputs;
                node->set_else_branch(build_subgraph(
                    es->second, ee->second, carried, else_out_ids));
            }
        } else if (op.type == OpType::Loop) {
            auto bs = op.int_attrs.find("body_ops_start");
            auto be = op.int_attrs.find("body_ops_end");
            if (bs != op.int_attrs.end() && be != op.int_attrs.end()) {
                node->set_body(build_subgraph(
                    bs->second, be->second, op.inputs, loop_body_output_ids(op)));
            }
        }

        graph->add_node(node);
    }

    // Set outputs
    std::vector<std::shared_ptr<Value>> graph_outputs;
    for (const auto& output : outputs) {
        auto tensor_id = register_tensor(output);
        auto it = value_map.find(tensor_id);
        if (it != value_map.end()) {
            graph_outputs.push_back(it->second);
        }
    }
    graph->set_outputs(graph_outputs);

    // Topological sort
    graph->topological_sort();

    // Type inference
    graph->infer_types();

    return graph;
}

auto Tracer::record_op(TracedOp op) -> void {
    if (tracing_) {
        ops_.push_back(std::move(op));
    }
}

namespace {
// Map an in-place OpId to the FUNCTIONAL OpType used to replay it. Graph replay
// is functional (new = op(old, others...)), so an in-place add_ replays as a
// plain Add producing a fresh value — exactly the SSA-renamed semantics.
auto inplace_opid_to_optype(OpId op) -> std::optional<OpType> {
    switch (op) {
        case OpId::AddInplace:     return OpType::Add;
        case OpId::SubInplace:     return OpType::Sub;
        case OpId::MulInplace:     return OpType::Mul;
        case OpId::DivInplace:     return OpType::Div;
        case OpId::ReLUInplace:    return OpType::ReLU;
        case OpId::SigmoidInplace: return OpType::Sigmoid;
        case OpId::TanhInplace:    return OpType::Tanh;
        case OpId::GeluInplace:    return OpType::GELU;
        default:                   return std::nullopt;
    }
}
} // namespace

auto Tracer::record_inplace(OpId op, Tensor& target,
                            std::span<const Tensor> others,
                            const OpAttributes& attrs) -> void {
    if (!tracing_) return;

    auto op_type = inplace_opid_to_optype(op);
    if (!op_type) {
        // Un-mappable in-place op: a graph break (strict throws) rather than a
        // silently dropped mutation.
        record_graph_break(
            "in-place operation (OpId=" +
            std::to_string(static_cast<int>(op)) +
            ") has no IR OpType mapping");
        return;
    }

    // Old (pre-mutation) value id. An in-place op leaves storage/shape/strides
    // — and therefore the tracer fingerprint — unchanged, so the fingerprint
    // still resolves to the pre-op value at this point.
    std::string old_id = register_tensor(target);

    std::vector<std::string> input_ids;
    input_ids.reserve(others.size() + 1);
    input_ids.push_back(old_id);
    for (const auto& o : others) {
        input_ids.push_back(register_tensor(o));
    }

    // Mint a fresh SSA value for the mutated tensor and REMAP its fingerprint
    // to that new id, so every later read of `target` resolves to the post-op
    // value. Without this, the node's own output would alias its input (a
    // self-referential node) and downstream reads would see the stale value.
    std::string new_id = register_new_tensor(target);

    TracedOp traced(*op_type, std::move(input_ids), {new_id});
    // Carry the scalar attrs the functional replay may need.
    if (attrs.has(AttrKey::Min)) {
        traced.attrs["min"] = static_cast<float>(attrs.get_float(AttrKey::Min));
    }
    if (attrs.has(AttrKey::Max)) {
        traced.attrs["max"] = static_cast<float>(attrs.get_float(AttrKey::Max));
    }
    if (attrs.has(AttrKey::Negative_slope)) {
        traced.attrs["negative_slope"] =
            static_cast<float>(attrs.get_float(AttrKey::Negative_slope));
    }
    record_op(std::move(traced));
}

// Identity of a *logical* tensor view. The same storage viewed with different
// shape/strides (e.g. a square-matrix transpose, which keeps the same data_ptr
// AND shape but flips strides) is a DISTINCT value and must get its own id;
// keying on data_ptr alone silently aliased them and dropped the view op.
static auto tensor_fingerprint(const Tensor& tensor) -> std::string {
    std::ostringstream ss;
    ss << const_cast<void*>(tensor.data_ptr());
    ss << '#' << static_cast<int>(tensor.dtype()) << '#';
    for (auto s : tensor.shape())   ss << s << ',';
    ss << '#';
    for (auto s : tensor.strides()) ss << s << ',';
    return ss.str();
}

auto Tracer::register_tensor(const Tensor& tensor) -> std::string {
    // Reuse an id only when the FULL logical view matches (ptr + dtype + shape +
    // strides). Two views sharing storage but differing in strides now get
    // distinct ids, and each stays stable across re-uses (the fingerprint, not a
    // single ptr slot, is the key).
    const std::string fp = tensor_fingerprint(tensor);
    auto it = tensor_id_map_.find(fp);
    if (it != tensor_id_map_.end()) {
        return it->second;
    }

    std::vector<int64_t> shape;
    shape.reserve(tensor.shape().size());
    for (auto s : tensor.shape()) {
        shape.push_back(s);
    }

    auto id = generate_tensor_id();
    tensor_id_map_[fp] = id;
    tensor_info_[id] = TensorInfo(shape, tensor.dtype(), tensor.device());
    // Phase 6.4: retain the full Tensor so end_trace() can later decide
    // which tensors are parameters (constants) vs intermediates vs
    // graph inputs. Shallow Tensor copy is cheap — intrusive refcount
    // on the underlying storage.
    tensor_storage_[id] = tensor;
    return id;
}

auto Tracer::register_tensor(const Variable& var) -> std::string {
    return register_tensor(var.tensor());
}

auto Tracer::register_new_tensor(const Tensor& tensor) -> std::string {
    // Always allocate a fresh ID even if this exact logical view was seen
    // before — used by view/layer ops that must materialize a distinct graph
    // value. Map this view's fingerprint to the new id so later consumers of
    // the SAME logical view resolve to it. Because the fingerprint includes
    // strides, a view no longer collides with its base (different strides), so
    // this does not disturb the base tensor's mapping.
    auto id = generate_tensor_id();
    std::vector<int64_t> shape;
    for (auto s : tensor.shape()) {
        shape.push_back(s);
    }
    tensor_info_[id] = TensorInfo(shape, tensor.dtype(), tensor.device());
    tensor_storage_[id] = tensor;
    tensor_id_map_[tensor_fingerprint(tensor)] = id;
    return id;
}

auto Tracer::alias_tensor(const Tensor& alias, const std::string& existing_id) -> void {
    // Point this tensor's fingerprint at an already-registered value id so any
    // later consumer that registers `alias` resolves to the SAME graph value,
    // WITHOUT minting a new id or recording a node. Used for transparent
    // value-identity ops (e.g. Contiguous): contiguous(x) has identical values
    // to x but — when x was non-contiguous — a fresh storage/fingerprint, so it
    // cannot be deduped by register_tensor. Eliding it here keeps such ops from
    // forcing a whole-graph eager fallback (they are pure layout materializations
    // that do not change values, so the downstream replay is numerically exact).
    tensor_id_map_[tensor_fingerprint(alias)] = existing_id;
}

auto Tracer::set_parameters(std::vector<std::shared_ptr<Variable>> params) -> void {
    parameters_ = std::move(params);
    param_storage_index_.clear();
    for (size_t i = 0; i < parameters_.size(); ++i) {
        const auto& p = parameters_[i];
        if (!p) continue;
        // Key on the parameter tensor's data pointer — the same identity a
        // traced op input that reads this parameter will present at
        // register_tensor() time. A view/reshape of the parameter created
        // INSIDE the traced forward produces its own graph value (the view op
        // is recorded), so only the parameter's own leaf value matches here.
        const void* id = p->tensor().data_ptr();
        if (id != nullptr) {
            // First declaration wins if two params alias the same storage
            // (they shouldn't); keep the lower index deterministically.
            param_storage_index_.emplace(id, i);
        }
    }
}

auto Tracer::get_tensor_info(const std::string& tensor_id) const -> const TensorInfo& {
    auto it = tensor_info_.find(tensor_id);
    if (it == tensor_info_.end()) {
        throw std::runtime_error("Tensor ID not found: " + tensor_id);
    }
    return it->second;
}

auto Tracer::clear() -> void {
    ops_.clear();
    tensor_info_.clear();
    tensor_id_map_.clear();
    tensor_storage_.clear();
    parameters_.clear();
    param_storage_index_.clear();
    next_tensor_id_ = 0;
    graph_break_count_ = 0;
    // clear() is the tracer's full reset — it stops tracing too, so
    // TracingGuard's destructor leaves the global tracer in a clean
    // state for the next guard. The Tracer API contract is:
    //   - start_trace() turns tracing on
    //   - end_trace() / clear() turn it off
    // Previously clear() only purged recorded state; tracing_ was left
    // on, so back-to-back TracingGuards bled state into each other.
    tracing_ = false;
}

auto Tracer::get_instance() -> Tracer& {
    thread_local Tracer instance;
    return instance;
}

auto Tracer::generate_tensor_id() -> std::string {
    std::ostringstream oss;
    oss << "t" << next_tensor_id_++;
    return oss.str();
}

// ============================================================================
// TracingGuard implementation
// ============================================================================

TracingGuard::TracingGuard() : tracer_(Tracer::get_instance()) {
    tracer_.start_trace();

    // Install a dispatch interceptor that records every op passing
    // through the dispatcher. Without this, `Tracer::record_op` is
    // never called and `end_trace` returns an empty graph — the
    // "CompiledModule produced no outputs" symptom observed in
    // Phase 6.5. The mirror pattern in compile.cpp's
    // CompiledFunction::trace_and_compile does this correctly.
    // Route unmappable ops through record_graph_break so they are counted (and
    // throw in strict mode) instead of being silently dropped — which would
    // otherwise bake the unmapped op's output into the graph as a constant.
    auto interceptor = make_tracing_interceptor(
        tracer_,
        [this](OpId op) {
            tracer_.record_graph_break(
                "unmapped operation (OpId=" +
                std::to_string(static_cast<int>(op)) +
                ") has no IR OpType mapping");
        });
    DispatchInterceptorStack::push(std::move(interceptor));
    interceptor_installed_ = true;

    // Install a graph-break hook so that leaf operations in tenzor_core
    // (notably Tensor::item(), but also any future data-dependent path)
    // can notify the tracer without a cyclic include. See
    // include/tenzor/core/jit_hooks.hpp.
    tenzor::detail::set_graph_break_hook(
        [this](const std::string& reason) {
            tracer_.record_graph_break(reason);
        });

    // Install the in-place op hook. In-place kernels dispatch through
    // dispatch_inplace, which bypasses the DispatchInterceptorStack; without
    // this hook their mutations are invisible to the trace and later reads
    // resolve to the pre-op value. record_inplace() records a value-versioned
    // functional node so replay applies the mutation.
    tenzor::detail::set_inplace_op_hook(
        [this](OpId op, Tensor& target, const Tensor* others,
               std::size_t num_others, const OpAttributes& attrs) {
            tracer_.record_inplace(
                op, target,
                std::span<const Tensor>(others, num_others), attrs);
        });
}

TracingGuard::~TracingGuard() {
    // Tear down the hooks so subsequent non-traced calls (e.g. .item(),
    // add_) don't walk into a stale Tracer reference.
    tenzor::detail::set_graph_break_hook(nullptr);
    tenzor::detail::set_inplace_op_hook(nullptr);

    if (interceptor_installed_) {
        DispatchInterceptorStack::pop();
    }
    if (tracer_.is_tracing()) {
        tracer_.clear();
    }
}

auto TracingGuard::get_graph(const std::vector<Variable>& inputs,
                              const std::vector<Variable>& outputs) -> std::shared_ptr<Graph> {
    return tracer_.end_trace(inputs, outputs);
}

// ============================================================================
// Trace functions
// ============================================================================

auto trace(std::shared_ptr<nn::Module> module,
           const Variable& dummy_input) -> std::shared_ptr<Graph> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    // Set module to eval mode
    module->eval();

    // Start tracing
    TracingGuard guard;

    // Run forward pass
    Variable output = module->forward(dummy_input);

    // End tracing and get graph
    return guard.get_graph({dummy_input}, {output});
}

auto trace(std::function<std::vector<Variable>(const std::vector<Variable>&)> func,
           const std::vector<Variable>& inputs) -> std::shared_ptr<Graph> {
    if (!func) {
        throw std::runtime_error("Cannot trace null function");
    }

    // Start tracing
    TracingGuard guard;

    // Run function
    auto outputs = func(inputs);

    // End tracing and get graph
    return guard.get_graph(inputs, outputs);
}

auto trace(std::shared_ptr<nn::Module> module,
           const Tensor& dummy_input) -> std::shared_ptr<CompiledModule> {
    if (!module) {
        throw std::runtime_error("Cannot trace null module");
    }

    Variable input_var(dummy_input, false);
    // Route through CompiledModule::trace so the traced device/dtype metadata
    // is captured — otherwise forward() cannot detect device/dtype mismatches.
    return CompiledModule::trace(module, input_var);
}

// ============================================================================
// Control Flow: trace_if
// ============================================================================

auto Tracer::trace_if(const Tensor& condition,
                      std::function<std::vector<Variable>(const std::vector<Variable>&)> then_fn,
                      std::function<std::vector<Variable>(const std::vector<Variable>&)> else_fn,
                      const std::vector<Variable>& inputs) -> std::vector<Variable> {

    // Register condition tensor
    auto cond_id = register_tensor(condition);

    // Register input tensors
    std::vector<std::string> input_ids;
    for (const auto& v : inputs) {
        input_ids.push_back(register_tensor(v));
    }

    // Save current ops state to isolate subgraphs
    auto ops_before_then = ops_.size();

    // Trace the then-branch by executing it
    auto then_outputs = then_fn(inputs);

    // Record then-branch ops range
    auto then_ops_end = ops_.size();

    // Trace the else-branch
    auto else_outputs = else_fn(inputs);
    auto else_ops_end = ops_.size();

    // Register output tensors
    std::vector<std::string> output_ids;
    for (const auto& v : then_outputs) {
        output_ids.push_back(register_tensor(v));
    }
    std::vector<std::string> else_output_ids;
    for (const auto& v : else_outputs) {
        else_output_ids.push_back(register_tensor(v));
    }

    // Create the If operation with subgraph info as attributes
    TracedOp if_op(OpType::If, input_ids, output_ids);
    if_op.else_outputs = std::move(else_output_ids);
    if_op.inputs.insert(if_op.inputs.begin(), cond_id);  // condition is first input
    if_op.int_attrs["then_ops_start"] = static_cast<int64_t>(ops_before_then);
    if_op.int_attrs["then_ops_end"] = static_cast<int64_t>(then_ops_end);
    if_op.int_attrs["else_ops_start"] = static_cast<int64_t>(then_ops_end);
    if_op.int_attrs["else_ops_end"] = static_cast<int64_t>(else_ops_end);
    if_op.int_attrs["num_outputs"] = static_cast<int64_t>(then_outputs.size());

    record_op(std::move(if_op));

    // At trace time, we return then-branch outputs (arbitrary choice)
    // The compiled module will evaluate the condition at runtime
    return then_outputs;
}

// ============================================================================
// Control Flow: trace_loop
// ============================================================================

auto Tracer::trace_loop(int64_t max_iter,
                        std::function<Tensor(const std::vector<Variable>&)> cond_fn,
                        std::function<std::vector<Variable>(const std::vector<Variable>&)> body_fn,
                        const std::vector<Variable>& carried) -> std::vector<Variable> {

    // The interpreter (graph.cpp Loop case) implements ONNX-style loop
    // semantics and expects:
    //   * node inputs      = [max_iter, cond, carried...]
    //   * body sub-graph in = [iter, cond, carried...]
    //   * body sub-graph out= [cond, carried...]   (first output is the new cond)
    // and it evaluates the condition BEFORE each body run (`i < max_iter &&
    // cond`), so a zero-iteration loop returns the initial carried state —
    // matching eager `while_loop` (control_flow.cpp), which checks cond first.
    // We build a Loop node + body subgraph that satisfies this contract exactly.

    // Register carried state. These become node inputs 2.. and body inputs 2..
    std::vector<std::string> carried_ids;
    carried_ids.reserve(carried.size());
    for (const auto& v : carried) {
        carried_ids.push_back(register_tensor(v));
    }

    // Loop trip-count constant (node input[0]). Built on CPU where `full` is a
    // direct fill with NO OpId dispatch, so it does not record a spurious op
    // into the trace. The interpreter reads it via `.to(Float32).item()`.
    Tensor max_iter_tensor = tenzor::full({1}, static_cast<double>(max_iter),
                                          DType::Float32, Device::cpu());
    std::string max_iter_id = register_tensor(max_iter_tensor);

    // Entry condition (node input[1]) evaluated on the ENTRY state, BEFORE the
    // body runs. Recording these ops here — before body_ops_start — keeps them
    // in the PARENT op stream so they are replayed in the outer graph (they are
    // outside every [body_ops_start, body_ops_end) skip range). This is what
    // makes a zero-iteration loop possible: if the entry cond is false the
    // interpreter never enters the body.
    Tensor entry_cond = cond_fn(carried);
    std::string entry_cond_id = register_tensor(entry_cond);

    // Trace one iteration of the body.
    auto body_ops_start = ops_.size();
    auto body_outputs = body_fn(carried);

    // Post-body exit condition, evaluated on the updated state. Recorded INSIDE
    // the body op range so it becomes part of the body subgraph and is surfaced
    // as the body's FIRST output (loop_cond_output) — the interpreter reads
    // body_outputs[0] as the next-iteration condition.
    Tensor post_cond = cond_fn(body_outputs);
    std::string post_cond_id = register_tensor(post_cond);

    auto body_ops_end = ops_.size();

    // Register output IDs (updated carried state).
    std::vector<std::string> output_ids;
    output_ids.reserve(body_outputs.size());
    for (const auto& v : body_outputs) {
        output_ids.push_back(register_tensor(v));
    }

    // Loop node inputs: [max_iter, entry_cond, carried...]. build_subgraph uses
    // op.inputs verbatim as the body subgraph inputs, giving body inputs
    // [iter, cond, carried...] — the interpreter binds the runtime iter/cond
    // into the first two placeholders (the body never references them here) and
    // the carried into the rest.
    std::vector<std::string> loop_inputs;
    loop_inputs.reserve(2 + carried_ids.size());
    loop_inputs.push_back(max_iter_id);
    loop_inputs.push_back(entry_cond_id);
    loop_inputs.insert(loop_inputs.end(), carried_ids.begin(), carried_ids.end());

    TracedOp loop_op(OpType::Loop, std::move(loop_inputs), output_ids);
    loop_op.loop_cond_output = post_cond_id;
    loop_op.int_attrs["max_iter"] = max_iter;
    loop_op.int_attrs["body_ops_start"] = static_cast<int64_t>(body_ops_start);
    loop_op.int_attrs["body_ops_end"] = static_cast<int64_t>(body_ops_end);
    loop_op.int_attrs["num_carried"] = static_cast<int64_t>(carried.size());

    record_op(std::move(loop_op));

    // Return body outputs as the trace-time result (represents final carried
    // state; the compiled graph surfaces output_ids -> final carried).
    return body_outputs;
}

} // namespace jit
} // namespace tenzor
