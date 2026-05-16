/**
 * @file exporter.cpp
 * @brief Phase 3 nn::Module -> .tzlite exporter.
 *
 * Strategy: a direct module walker. We recursively dispatch on the runtime
 * type of each submodule and emit corresponding LiteNodes + weight tensors.
 * This is simpler and more transparent than routing through the JIT tracer;
 * Phase 5 may add a JIT-based exporter once the tracer's stability is
 * established for the full Lite op coverage.
 */

#include "tenzor/lite/exporter.hpp"

#include "tenzor/autograd/variable.hpp"
#include "tenzor/lite/lite_graph.hpp"
#include "tenzor/lite/model_format.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/ops/op_id.hpp"

#include <stdexcept>
#include <string>
#include <typeinfo>
#include <vector>

namespace tenzor::lite {

namespace {

// ---------------------------------------------------------------------------
// GraphBuilder — accumulates LiteNodes and weights while keeping fresh
// tensor_ids in order.
// ---------------------------------------------------------------------------
struct GraphBuilder {
    LiteGraph graph;
    WriteOptions opts;
    int16_t next_id{0};

    auto fresh() -> int16_t { return next_id++; }

    auto declare_input(const std::vector<int64_t>& shape, DType dtype) -> int16_t {
        auto id = fresh();
        opts.input_ids.push_back(id);
        opts.input_specs[id] = {dtype, shape};
        return id;
    }

    auto add_weight(const Tensor& t) -> int16_t {
        auto id = fresh();
        opts.weights.emplace(id, t);
        return id;
    }

    auto declare_output(int16_t id) -> void {
        opts.output_ids.push_back(id);
    }
};

// ---------------------------------------------------------------------------
// Layer-specific emitters.
// ---------------------------------------------------------------------------

auto emit_linear(nn::Linear& lin, GraphBuilder& b, int16_t in_id) -> int16_t {
    if (!lin.weight()) {
        throw std::runtime_error(
            "export_to_tzlite: nn::Linear has no weight parameter (uninitialised)");
    }
    auto w_id = b.add_weight(lin.weight()->tensor());

    LiteNode node;
    node.op = OpId::Linear;
    if (lin.has_bias() && lin.bias() != nullptr) {
        auto bias_id = b.add_weight(lin.bias()->tensor());
        node.input_ids = {in_id, w_id, bias_id};
    } else {
        node.input_ids = {in_id, w_id};
    }
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// Emit a unary activation op (ReLU / Sigmoid / Tanh / GELU).
auto emit_unary(OpId op, GraphBuilder& b, int16_t in_id) -> int16_t {
    LiteNode node;
    node.op = op;
    node.input_ids = {in_id};
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// H2 fix: emit a unary activation that takes a scalar parameter via attrs.f[0].
// LeakyReLU(negative_slope), ELU(alpha), CELU(alpha) all use this shape.
auto emit_unary_with_alpha(OpId op, GraphBuilder& b, int16_t in_id,
                           double alpha) -> int16_t {
    LiteNode node;
    node.op = op;
    node.input_ids = {in_id};
    node.attrs.f[0] = static_cast<float>(alpha);
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// Forward-declared recursive visitor.
auto visit(nn::Module& m, GraphBuilder& b, int16_t in_id) -> int16_t;

auto emit_sequential(nn::Sequential& seq, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    int16_t cur = in_id;
    for (const auto& child : seq.modules()) {
        cur = visit(*child, b, cur);
    }
    return cur;
}

// ---------------------------------------------------------------------------
// Visitor — runtime dispatch by module type.
// ---------------------------------------------------------------------------
auto visit(nn::Module& m, GraphBuilder& b, int16_t in_id) -> int16_t {
    if (auto* seq = dynamic_cast<nn::Sequential*>(&m)) {
        return emit_sequential(*seq, b, in_id);
    }
    if (auto* lin = dynamic_cast<nn::Linear*>(&m)) {
        return emit_linear(*lin, b, in_id);
    }
    // Wave Inf-E6 (deferred → landed): expanded activation coverage.
    // All zero-attribute unary activations dispatch through emit_unary —
    // the underlying OpId already encodes the math; no attrs needed.
    if (dynamic_cast<nn::ReLU*>(&m))      return emit_unary(OpId::ReLU,     b, in_id);
    if (dynamic_cast<nn::Sigmoid*>(&m))   return emit_unary(OpId::Sigmoid,  b, in_id);
    if (dynamic_cast<nn::Tanh*>(&m))      return emit_unary(OpId::Tanh,     b, in_id);
    if (dynamic_cast<nn::GELU*>(&m))      return emit_unary(OpId::Gelu,     b, in_id);
    if (dynamic_cast<nn::Mish*>(&m))      return emit_unary(OpId::Mish,     b, in_id);
    if (dynamic_cast<nn::SELU*>(&m))      return emit_unary(OpId::Selu,     b, in_id);
    // H2 fix: Hardswish has different math from Swish (sigmoid·x). It is
    // `x · clamp(x+3, 0, 6) / 6`. Until Hardswish has its own dispatch
    // OpId (Inf-D deferred), refuse to export rather than silently emit
    // the wrong math.
    if (dynamic_cast<nn::Hardswish*>(&m)) {
        throw std::runtime_error(
            "export_to_tzlite: nn::Hardswish has no dedicated Lite OpId yet. "
            "Either replace with nn::Swish (different math, sigmoid·x) or "
            "wait for the Hardswish OpId to land (Inf-D follow-up).");
    }
    // H2 fix: load the activation parameter from the Module member before
    // emitting. Previously emitted with attrs.f[0] = 0 which silently
    // executed ReLU instead of LeakyReLU / ELU.
    if (auto* lr = dynamic_cast<nn::LeakyReLU*>(&m)) {
        return emit_unary_with_alpha(OpId::LeakyReLU, b, in_id,
                                     lr->negative_slope());
    }
    if (auto* el = dynamic_cast<nn::ELU*>(&m)) {
        return emit_unary_with_alpha(OpId::Elu, b, in_id, el->alpha());
    }

    throw std::runtime_error(
        std::string{"export_to_tzlite: unsupported module type '"} +
        typeid(m).name() +
        "'. Lite exporter supports nn::Linear, nn::Sequential, and the "
        "parameter-free / single-alpha activations (ReLU, Sigmoid, Tanh, "
        "GELU, Mish, SELU, Hardswish, LeakyReLU, ELU). File an issue or "
        "extend exporter.cpp to add this layer.");
}

}  // anonymous namespace

auto export_to_tzlite(nn::Module& module,
                      const std::string& path,
                      const ExportOptions& opts) -> void {
    if (opts.input_shape.empty()) {
        throw std::invalid_argument(
            "export_to_tzlite: ExportOptions::input_shape must be non-empty");
    }
    module.eval();

    GraphBuilder b;
    auto in_id  = b.declare_input(opts.input_shape, opts.input_dtype);
    auto out_id = visit(module, b, in_id);
    b.declare_output(out_id);

    b.graph.set_input_ids(b.opts.input_ids);
    b.graph.set_output_ids(b.opts.output_ids);

    b.opts.metadata = opts.metadata;
    b.opts.metadata.emplace("framework", "tenzor-lite");
    b.opts.metadata.emplace("device", opts.device.to_string());

    TZLiteWriter::save(b.graph, path, b.opts);
}

}  // namespace tenzor::lite
