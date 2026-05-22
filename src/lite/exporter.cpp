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
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/dropout.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/nn/layers/flatten.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/pooling.hpp"
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

// ---------------------------------------------------------------------------
// C.3 audit emitters — Conv1/2/3d, BatchNorm2d, LayerNorm, GroupNorm, Flatten,
// MaxPool2d, AvgPool2d, Embedding. Dropout is handled inline in visit() as
// pass-through (inference-only).
// ---------------------------------------------------------------------------

// Helper: fetch a registered parameter tensor by name. Throws if missing.
auto get_param_tensor(nn::Module& m, const char* name,
                      const char* layer) -> Tensor {
    auto var = m.get_parameter(name);
    if (!var) {
        throw std::runtime_error(
            std::string{"export_to_tzlite: "} + layer +
            " is missing parameter '" + name + "'.");
    }
    return var->tensor();
}

// Helper: fetch a registered buffer tensor by name. Throws if missing.
auto get_buffer_tensor(nn::Module& m, const char* name,
                       const char* layer) -> Tensor {
    auto var = m.get_buffer(name);
    if (!var) {
        throw std::runtime_error(
            std::string{"export_to_tzlite: "} + layer +
            " is missing buffer '" + name + "'.");
    }
    return var->tensor();
}

// Common helper for Conv kernels: weight + optional bias, plus per-axis
// stride/padding/dilation already populated into attrs.
auto emit_conv_node(OpId op, nn::Module& m, GraphBuilder& b, int16_t in_id,
                    LiteAttributes attrs, const char* layer) -> int16_t {
    auto w_id = b.add_weight(get_param_tensor(m, "weight", layer));
    LiteNode node;
    node.op = op;
    node.attrs = std::move(attrs);
    auto bias_var = m.get_parameter("bias");
    if (bias_var) {
        auto bias_id = b.add_weight(bias_var->tensor());
        node.input_ids = {in_id, w_id, bias_id};
    } else {
        node.input_ids = {in_id, w_id};
    }
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_conv2d(nn::Conv2d& c, GraphBuilder& b, int16_t in_id) -> int16_t {
    LiteAttributes attrs;
    // Scalar fallback uses H-axis values (so symmetric kernels still work
    // on backends that ignore per-axis keys); per-axis extras carry the
    // exact values.
    attrs.i[0] = c.stride_h();
    attrs.i[1] = c.padding_h();
    attrs.i[2] = c.dilation_h();
    attrs.i[3] = c.groups();
    // Read kernel size from the weight shape: [O, I/g, kH, kW].
    auto w_shape = get_param_tensor(c, "weight", "nn::Conv2d").shape();
    if (w_shape.size() != 4) {
        throw std::runtime_error(
            "export_to_tzlite: nn::Conv2d weight must be rank-4, got rank " +
            std::to_string(w_shape.size()));
    }
    int64_t kH = w_shape[2];
    int64_t kW = w_shape[3];
    attrs.extra_i = {
        c.stride_h(), c.stride_w(),
        c.padding_h(), c.padding_w(),
        c.dilation_h(), c.dilation_w(),
        c.groups(),
        kH, kW,
    };
    return emit_conv_node(OpId::Conv2dForward, c, b, in_id, std::move(attrs),
                          "nn::Conv2d");
}

auto emit_conv1d(nn::Conv1d& c, GraphBuilder& b, int16_t in_id) -> int16_t {
    LiteAttributes attrs;
    attrs.i[0] = c.stride();
    attrs.i[1] = c.padding();
    attrs.i[2] = c.dilation();
    attrs.i[3] = c.groups();
    auto w_shape = get_param_tensor(c, "weight", "nn::Conv1d").shape();
    if (w_shape.size() != 3) {
        throw std::runtime_error(
            "export_to_tzlite: nn::Conv1d weight must be rank-3, got rank " +
            std::to_string(w_shape.size()));
    }
    int64_t kW = w_shape[2];
    attrs.extra_i = {
        c.stride(), c.padding(), c.dilation(), c.groups(), kW,
    };
    return emit_conv_node(OpId::Conv1dForward, c, b, in_id, std::move(attrs),
                          "nn::Conv1d");
}

auto emit_conv3d(nn::Conv3d& c, GraphBuilder& b, int16_t in_id) -> int16_t {
    LiteAttributes attrs;
    attrs.i[0] = c.stride_d();
    attrs.i[1] = c.padding_d();
    attrs.i[2] = c.dilation_d();
    attrs.i[3] = c.groups();
    auto w_shape = get_param_tensor(c, "weight", "nn::Conv3d").shape();
    if (w_shape.size() != 5) {
        throw std::runtime_error(
            "export_to_tzlite: nn::Conv3d weight must be rank-5, got rank " +
            std::to_string(w_shape.size()));
    }
    int64_t kD = w_shape[2];
    int64_t kH = w_shape[3];
    int64_t kW = w_shape[4];
    attrs.extra_i = {
        c.stride_d(), c.stride_h(), c.stride_w(),
        c.padding_d(), c.padding_h(), c.padding_w(),
        c.dilation_d(), c.dilation_h(), c.dilation_w(),
        c.groups(),
        kD, kH, kW,
    };
    return emit_conv_node(OpId::Conv3dForward, c, b, in_id, std::move(attrs),
                          "nn::Conv3d");
}

auto emit_batchnorm2d(nn::BatchNorm2d& bn, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    // Inference contract: BatchNorm2dForwardAffine takes
    //   (x, mean, var, weight, bias)  with attrs.Eps.
    // Use running_mean / running_var (the only valid eval-time stats); fall
    // back to (un-affine) BatchNorm2dForward if affine=false.
    auto mean_id = b.add_weight(
        get_buffer_tensor(bn, "running_mean", "nn::BatchNorm2d"));
    auto var_id = b.add_weight(
        get_buffer_tensor(bn, "running_var", "nn::BatchNorm2d"));

    LiteNode node;
    node.attrs.f[0] = static_cast<float>(bn.eps());

    auto weight_var = bn.get_parameter("weight");
    auto bias_var   = bn.get_parameter("bias");
    if (weight_var && bias_var) {
        auto w_id = b.add_weight(weight_var->tensor());
        auto b_id = b.add_weight(bias_var->tensor());
        node.op = OpId::BatchNorm2dForwardAffine;
        node.input_ids = {in_id, mean_id, var_id, w_id, b_id};
    } else {
        node.op = OpId::BatchNorm2dForward;
        node.input_ids = {in_id, mean_id, var_id};
    }
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_layernorm(nn::LayerNorm& ln, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    // Backend LayerNorm kernel contract: (x, weight, bias) -> (out, mean, rstd).
    // NormalizedShape is carried as the comma-separated int list via the
    // dispatch_bridge extras → AttrKey::NormalizedShape mapping. eps in f[0].
    //
    // LayerNorm always allocates weight/bias parameters (ones/zeros when
    // elementwise_affine=false), so they are always registered — there is no
    // need to branch.
    auto w_id = b.add_weight(get_param_tensor(ln, "weight", "nn::LayerNorm"));
    auto bias_id = b.add_weight(get_param_tensor(ln, "bias", "nn::LayerNorm"));

    LiteNode node;
    node.op = OpId::LayerNorm;
    node.input_ids = {in_id, w_id, bias_id};
    node.attrs.f[0] = static_cast<float>(ln.eps());
    // Reach into the public method that exposes the normalized_shape via the
    // forward pass. LayerNorm doesn't have a getter for it, so reconstruct
    // from the weight tensor's shape (which always matches normalized_shape).
    auto w_shape = ln.get_parameter("weight")->tensor().shape();
    node.attrs.extra_i.assign(w_shape.begin(), w_shape.end());
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_groupnorm(nn::GroupNorm& gn, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    // GroupNorm kernel: (x, weight, bias) -> (out, mean, rstd). Attrs: Eps,
    // NumGroups. Slot layout: f[0]=eps, i[0]=num_groups (matches dispatch
    // bridge GroupNorm case).
    auto w_id = b.add_weight(get_param_tensor(gn, "weight", "nn::GroupNorm"));
    auto bias_id = b.add_weight(get_param_tensor(gn, "bias", "nn::GroupNorm"));

    LiteNode node;
    node.op = OpId::GroupNorm;
    node.input_ids = {in_id, w_id, bias_id};
    node.attrs.f[0] = static_cast<float>(gn.eps());
    node.attrs.i[0] = gn.num_groups();
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_flatten(nn::Flatten& fl, GraphBuilder& b, int16_t in_id) -> int16_t {
    LiteNode node;
    node.op = OpId::Flatten;
    node.input_ids = {in_id};
    node.attrs.i[0] = fl.start_dim();
    node.attrs.i[1] = fl.end_dim();
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_maxpool2d(nn::MaxPool2d& mp, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    auto ks = mp.get_kernel_size();
    auto st = mp.get_stride();
    auto pd = mp.get_padding();
    LiteNode node;
    node.op = OpId::MaxPool2dForward;
    node.input_ids = {in_id};
    // Scalar slots (H-axis canonical) + per-axis extras.
    node.attrs.i[0] = ks[0];
    node.attrs.i[1] = st[0];
    node.attrs.i[2] = pd[0];
    node.attrs.i[3] = 1;  // dilation (MaxPool2d module has none surfaced)
    node.attrs.extra_i = {
        ks[0], ks[1],
        st[0], st[1],
        pd[0], pd[1],
        1, 1,                          // dilation_h, dilation_w
        mp.get_ceil_mode() ? 1 : 0,
    };
    // MaxPool2dForward returns (output, indices). Allocate both, but the
    // graph only consumes [0]; the indices output is fine to leave unread.
    auto out_id  = b.fresh();
    auto idx_id  = b.fresh();
    node.output_ids = {out_id, idx_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_avgpool2d(nn::AvgPool2d& ap, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    auto ks = ap.get_kernel_size();
    auto st = ap.get_stride();
    auto pd = ap.get_padding();
    LiteNode node;
    node.op = OpId::AvgPool2dForward;
    node.input_ids = {in_id};
    node.attrs.i[0] = ks[0];
    node.attrs.i[1] = st[0];
    node.attrs.i[2] = pd[0];
    // AvgPool2d's module currently doesn't surface ceil_mode/count_include_pad
    // accessors — the underlying CPU kernel ignores both. Encode defaults
    // (PyTorch defaults: ceil_mode=false, count_include_pad=true).
    node.attrs.extra_i = {
        ks[0], ks[1],
        st[0], st[1],
        pd[0], pd[1],
        0,  // ceil_mode (default false)
        1,  // count_include_pad (default true)
    };
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_embedding(nn::Embedding& em, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    // Kernel contract: (weight, indices) -> output. The graph's incoming
    // `in_id` is the indices tensor.
    auto w_id = b.add_weight(em.weight().tensor());
    LiteNode node;
    node.op = OpId::Embedding;
    node.input_ids = {w_id, in_id};
    // The nn::Embedding module doesn't surface a public padding_idx getter,
    // so we encode -1 (= no padding row) by default. Future work: extend
    // Embedding to expose the index.
    node.attrs.i[0] = -1;
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
    // C.3 audit: layer-type coverage expansion. Order matters — match the
    // most-derived types first; Conv1d / MaxPool1d (etc.) currently have no
    // export path, so they fall through to the catch-all error.
    if (auto* conv = dynamic_cast<nn::Conv2d*>(&m)) {
        return emit_conv2d(*conv, b, in_id);
    }
    if (auto* conv = dynamic_cast<nn::Conv1d*>(&m)) {
        return emit_conv1d(*conv, b, in_id);
    }
    if (auto* conv = dynamic_cast<nn::Conv3d*>(&m)) {
        return emit_conv3d(*conv, b, in_id);
    }
    if (auto* bn = dynamic_cast<nn::BatchNorm2d*>(&m)) {
        return emit_batchnorm2d(*bn, b, in_id);
    }
    if (auto* ln = dynamic_cast<nn::LayerNorm*>(&m)) {
        return emit_layernorm(*ln, b, in_id);
    }
    if (auto* gn = dynamic_cast<nn::GroupNorm*>(&m)) {
        return emit_groupnorm(*gn, b, in_id);
    }
    if (auto* fl = dynamic_cast<nn::Flatten*>(&m)) {
        return emit_flatten(*fl, b, in_id);
    }
    if (dynamic_cast<nn::Dropout*>(&m)) {
        // Inference-only Lite runtime — dropout is identity. Emit nothing
        // and forward the tensor id (zero-copy passthrough).
        return in_id;
    }
    if (auto* mp = dynamic_cast<nn::MaxPool2d*>(&m)) {
        return emit_maxpool2d(*mp, b, in_id);
    }
    if (auto* ap = dynamic_cast<nn::AvgPool2d*>(&m)) {
        return emit_avgpool2d(*ap, b, in_id);
    }
    if (auto* em = dynamic_cast<nn::Embedding*>(&m)) {
        return emit_embedding(*em, b, in_id);
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
        "'. Lite exporter supports nn::Linear, nn::Sequential, nn::Conv1d, "
        "nn::Conv2d, nn::Conv3d, nn::BatchNorm2d, nn::LayerNorm, "
        "nn::GroupNorm, nn::Flatten, nn::Dropout, nn::MaxPool2d, "
        "nn::AvgPool2d, nn::Embedding, and the parameter-free / "
        "single-alpha activations (ReLU, Sigmoid, Tanh, GELU, Mish, SELU, "
        "LeakyReLU, ELU). File an issue or extend exporter.cpp to add this "
        "layer.");
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
