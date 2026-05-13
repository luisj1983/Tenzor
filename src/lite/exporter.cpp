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
    if (dynamic_cast<nn::ReLU*>(&m)) {
        return emit_unary(OpId::ReLU, b, in_id);
    }
    if (dynamic_cast<nn::Sigmoid*>(&m)) {
        return emit_unary(OpId::Sigmoid, b, in_id);
    }
    if (dynamic_cast<nn::Tanh*>(&m)) {
        return emit_unary(OpId::Tanh, b, in_id);
    }
    if (dynamic_cast<nn::GELU*>(&m)) {
        return emit_unary(OpId::Gelu, b, in_id);
    }

    throw std::runtime_error(
        std::string{"export_to_tzlite: unsupported module type '"} +
        typeid(m).name() +
        "'. Phase 3 supports nn::Linear, nn::ReLU, nn::Sigmoid, nn::Tanh, "
        "nn::GELU, and nn::Sequential. File an issue or extend exporter.cpp "
        "to add this layer.");
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
