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
#include "tenzor/nn/layers/identity.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/layers/upsample.hpp"
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
    // C.3 audit batch 3: cache the graph-input shape so emitters that need
    // to resolve relative sizes (Upsample.scale_factor) can do so at export
    // time. Captured by declare_input(); empty before that call.
    std::vector<int64_t> input_shape;

    auto fresh() -> int16_t { return next_id++; }

    auto declare_input(const std::vector<int64_t>& shape, DType dtype) -> int16_t {
        auto id = fresh();
        opts.input_ids.push_back(id);
        opts.input_specs[id] = {dtype, shape};
        input_shape = shape;
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

// Helper: fetch an *optional* registered parameter. Returns nullptr if the
// parameter has never been registered (e.g. bias-less conv). Module::
// get_parameter throws on missing, so we wrap in try/catch to recover the
// optional semantics that callers want.
auto try_get_parameter(nn::Module& m, const char* name)
    -> std::shared_ptr<Variable> {
    try {
        return m.get_parameter(name);
    } catch (const std::out_of_range&) {
        return nullptr;
    }
}

// Common helper for Conv kernels: weight + optional bias, plus per-axis
// stride/padding/dilation already populated into attrs.
auto emit_conv_node(OpId op, nn::Module& m, GraphBuilder& b, int16_t in_id,
                    LiteAttributes attrs, const char* layer) -> int16_t {
    auto w_id = b.add_weight(get_param_tensor(m, "weight", layer));
    LiteNode node;
    node.op = op;
    node.attrs = std::move(attrs);
    auto bias_var = try_get_parameter(m, "bias");
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

    auto weight_var = try_get_parameter(bn, "weight");
    auto bias_var   = try_get_parameter(bn, "bias");
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

// ---------------------------------------------------------------------------
// Batch 2: ConvTranspose / InstanceNorm / 1d-3d pool extensions / adaptive
// pools / Identity emitters.
// ---------------------------------------------------------------------------

auto emit_conv_transpose2d(nn::ConvTranspose2d& c, GraphBuilder& b,
                           int16_t in_id) -> int16_t {
    LiteAttributes attrs;
    // Scalar fallback (H-axis canonical) + per-axis extras carry the
    // exact per-dim values for backends that read AttrKey::*H/*W.
    attrs.i[0] = c.stride_h();
    attrs.i[1] = c.padding_h();
    attrs.i[2] = c.dilation_h();
    attrs.i[3] = c.groups();
    auto w_shape = get_param_tensor(c, "weight", "nn::ConvTranspose2d").shape();
    if (w_shape.size() != 4) {
        throw std::runtime_error(
            "export_to_tzlite: nn::ConvTranspose2d weight must be rank-4, got rank " +
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
        c.output_padding_h(), c.output_padding_w(),
    };
    return emit_conv_node(OpId::ConvTranspose2dForward, c, b, in_id,
                          std::move(attrs), "nn::ConvTranspose2d");
}

auto emit_conv_transpose3d(nn::ConvTranspose3d& c, GraphBuilder& b,
                           int16_t in_id) -> int16_t {
    LiteAttributes attrs;
    attrs.i[0] = c.stride_d();
    attrs.i[1] = c.padding_d();
    attrs.i[2] = c.dilation_d();
    attrs.i[3] = c.groups();
    auto w_shape = get_param_tensor(c, "weight", "nn::ConvTranspose3d").shape();
    if (w_shape.size() != 5) {
        throw std::runtime_error(
            "export_to_tzlite: nn::ConvTranspose3d weight must be rank-5, got rank " +
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
        c.output_padding_d(), c.output_padding_h(), c.output_padding_w(),
    };
    return emit_conv_node(OpId::ConvTranspose3dForward, c, b, in_id,
                          std::move(attrs), "nn::ConvTranspose3d");
}

// Shared InstanceNorm emitter — the InstanceNorm CPU kernel reads N from
// shape[0], C from shape[1], and accumulates spatial dims from shape[2..],
// so a single emitter handles InstanceNorm1d/2d/3d transparently.
auto emit_instancenorm(nn::Module& m, GraphBuilder& b, int16_t in_id,
                       double eps, const char* layer) -> int16_t {
    // affine=false leaves the InstanceNorm module without weight/bias
    // parameters. The CPU kernel signature is (x, weight, bias, eps) with
    // weight/bias allowed to be uninitialized — but the lite runtime
    // can't pass empty tensors. So when there are no affine params we
    // synthesize ones/zeros constants of shape [C].
    auto weight_var = try_get_parameter(m, "weight");
    auto bias_var   = try_get_parameter(m, "bias");

    int16_t w_id, b_id;
    if (weight_var && bias_var) {
        w_id = b.add_weight(weight_var->tensor());
        b_id = b.add_weight(bias_var->tensor());
    } else {
        // Use Tensor.shape() info from a buffer-less layer is tricky; fall
        // back to the running stats shape if available. For InstanceNorm
        // there are no running stats — so we must read num_features from
        // the input shape at runtime, which we don't have. Refuse for now.
        throw std::runtime_error(
            std::string{"export_to_tzlite: "} + layer +
            " was built with affine=false. Lite export currently requires "
            "affine=true so that weight/bias tensors are available as "
            "constants — synthesized identity tensors would need a shape "
            "we cannot recover here.");
    }

    LiteNode node;
    node.op = OpId::InstanceNorm;
    node.input_ids = {in_id, w_id, b_id};
    node.attrs.f[0] = static_cast<float>(eps);
    // InstanceNorm kernel returns {output, mean, rstd}; allocate three IDs
    // so the graph references the output position correctly.
    auto out_id   = b.fresh();
    auto mean_id  = b.fresh();
    auto rstd_id  = b.fresh();
    node.output_ids = {out_id, mean_id, rstd_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// ---------------------------------------------------------------------------
// 1d/3d pooling emitters — extend the 2d patterns.
// ---------------------------------------------------------------------------

auto emit_maxpool1d(nn::MaxPool1d& mp, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    LiteNode node;
    node.op = OpId::MaxPool1dForward;
    node.input_ids = {in_id};
    node.attrs.i[0] = mp.get_kernel_size();
    node.attrs.i[1] = mp.get_stride();
    node.attrs.i[2] = mp.get_padding();
    node.attrs.i[3] = 1;  // dilation (MaxPool1d module surfaces none)
    // MaxPool1d kernel returns {output, indices}; allocate both ids.
    auto out_id = b.fresh();
    auto idx_id = b.fresh();
    node.output_ids = {out_id, idx_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_avgpool1d(nn::AvgPool1d& ap, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    LiteNode node;
    node.op = OpId::AvgPool1dForward;
    node.input_ids = {in_id};
    node.attrs.i[0] = ap.get_kernel_size();
    node.attrs.i[1] = ap.get_stride();
    node.attrs.i[2] = ap.get_padding();
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_maxpool3d(nn::MaxPool3d& mp, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    auto ks = mp.get_kernel_size();
    auto st = mp.get_stride();
    auto pd = mp.get_padding();
    LiteNode node;
    node.op = OpId::MaxPool3dForward;
    node.input_ids = {in_id};
    // Scalar slots (D-axis canonical) — the 3D kernel still consults the
    // per-axis keys when set, falling back to the scalar otherwise. We
    // populate per-axis extras to carry asymmetric values precisely.
    node.attrs.i[0] = ks[0];
    node.attrs.i[1] = st[0];
    node.attrs.i[2] = pd[0];
    node.attrs.i[3] = 1;  // dilation
    node.attrs.extra_i = {
        ks[0], ks[1], ks[2],
        st[0], st[1], st[2],
        pd[0], pd[1], pd[2],
        1, 1, 1,                          // dilation_d/h/w
        mp.get_ceil_mode() ? 1 : 0,
    };
    auto out_id = b.fresh();
    auto idx_id = b.fresh();
    node.output_ids = {out_id, idx_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_avgpool3d(nn::AvgPool3d& ap, GraphBuilder& b, int16_t in_id)
    -> int16_t {
    auto ks = ap.get_kernel_size();
    auto st = ap.get_stride();
    auto pd = ap.get_padding();
    LiteNode node;
    node.op = OpId::AvgPool3dForward;
    node.input_ids = {in_id};
    node.attrs.i[0] = ks[0];
    node.attrs.i[1] = st[0];
    node.attrs.i[2] = pd[0];
    node.attrs.extra_i = {
        ks[0], ks[1], ks[2],
        st[0], st[1], st[2],
        pd[0], pd[1], pd[2],
        0,  // ceil_mode (default)
        1,  // count_include_pad (default)
    };
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// ---------------------------------------------------------------------------
// Adaptive pool emitters — output_size attr carried per axis. 1D uses the
// single AttrKey::OutputSize; 2D/3D use OutputSizeH/W and OutputSizeD/H/W.
// Encoded in extra_i positionally so dispatch_bridge can translate.
// ---------------------------------------------------------------------------

auto emit_adaptive_avg_pool1d(nn::AdaptiveAvgPool1d& p, GraphBuilder& b,
                              int16_t in_id) -> int16_t {
    LiteNode node;
    node.op = OpId::AdaptiveAvgPool1d;
    node.input_ids = {in_id};
    node.attrs.i[0] = p.get_output_size();
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_adaptive_max_pool1d(nn::AdaptiveMaxPool1d& p, GraphBuilder& b,
                              int16_t in_id) -> int16_t {
    LiteNode node;
    node.op = OpId::AdaptiveMaxPool1d;
    node.input_ids = {in_id};
    node.attrs.i[0] = p.get_output_size();
    // AdaptiveMaxPool returns {output, indices}.
    auto out_id = b.fresh();
    auto idx_id = b.fresh();
    node.output_ids = {out_id, idx_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_adaptive_avg_pool2d(nn::AdaptiveAvgPool2d& p, GraphBuilder& b,
                              int16_t in_id) -> int16_t {
    auto os = p.get_output_size();
    LiteNode node;
    node.op = OpId::AdaptiveAvgPool2d;
    node.input_ids = {in_id};
    node.attrs.i[0] = os[0];           // legacy scalar (height) fallback
    node.attrs.extra_i = {os[0], os[1]};  // [H, W]
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_adaptive_max_pool2d(nn::AdaptiveMaxPool2d& p, GraphBuilder& b,
                              int16_t in_id) -> int16_t {
    auto os = p.get_output_size();
    LiteNode node;
    node.op = OpId::AdaptiveMaxPool2d;
    node.input_ids = {in_id};
    node.attrs.i[0] = os[0];
    node.attrs.extra_i = {os[0], os[1]};
    auto out_id = b.fresh();
    auto idx_id = b.fresh();
    node.output_ids = {out_id, idx_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_adaptive_avg_pool3d(nn::AdaptiveAvgPool3d& p, GraphBuilder& b,
                              int16_t in_id) -> int16_t {
    auto os = p.get_output_size();
    LiteNode node;
    node.op = OpId::AdaptiveAvgPool3d;
    node.input_ids = {in_id};
    node.attrs.i[0] = os[0];
    node.attrs.extra_i = {os[0], os[1], os[2]};
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_adaptive_max_pool3d(nn::AdaptiveMaxPool3d& p, GraphBuilder& b,
                              int16_t in_id) -> int16_t {
    auto os = p.get_output_size();
    LiteNode node;
    node.op = OpId::AdaptiveMaxPool3d;
    node.input_ids = {in_id};
    node.attrs.i[0] = os[0];
    node.attrs.extra_i = {os[0], os[1], os[2]};
    auto out_id = b.fresh();
    auto idx_id = b.fresh();
    node.output_ids = {out_id, idx_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// ---------------------------------------------------------------------------
// C.3 audit batch 3: Softmax / LogSoftmax / RMSNorm / Upsample emitters.
// ---------------------------------------------------------------------------

// Emit a Softmax-like op (Softmax / LogSoftmax) keyed by AttrKey::Dim.
auto emit_softmax_like(OpId op, GraphBuilder& b, int16_t in_id, int64_t dim)
    -> int16_t {
    LiteNode node;
    node.op = op;
    node.input_ids = {in_id};
    node.attrs.i[0] = dim;
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

auto emit_rmsnorm(nn::RMSNorm& rn, GraphBuilder& b, int16_t in_id) -> int16_t {
    // RMSNorm kernel contract: (x, weight) -> output. eps via AttrKey::Eps.
    auto w_id = b.add_weight(get_param_tensor(rn, "weight", "nn::RMSNorm"));
    LiteNode node;
    node.op = OpId::RMSNorm;
    node.input_ids = {in_id, w_id};
    node.attrs.f[0] = static_cast<float>(rn.eps());
    auto out_id = b.fresh();
    node.output_ids = {out_id};
    b.graph.add_node(std::move(node));
    return out_id;
}

// Map nn::Upsample's mode string to the integer code consumed by the
// dispatch bridge. Unknown modes raise — the kernel would fail later
// anyway and a clear export-time error is more debuggable.
auto upsample_mode_code(const std::string& mode) -> int64_t {
    if (mode == "nearest")       return 0;
    if (mode == "bilinear")      return 1;
    if (mode == "trilinear")     return 2;
    if (mode == "bicubic")       return 3;
    if (mode == "linear")        return 4;
    if (mode == "area")          return 5;
    if (mode == "nearest-exact") return 6;
    throw std::runtime_error(
        std::string{"export_to_tzlite: nn::Upsample has unsupported mode '"} +
        mode + "'. Supported: nearest, bilinear, trilinear, bicubic, "
        "linear, area, nearest-exact.");
}

auto emit_upsample(nn::Upsample& up, GraphBuilder& b, int16_t in_id) -> int16_t {
    const auto& input_shape = b.input_shape;
    // Resolve target spatial sizes at export time.  Upsample stores either
    // an absolute `size` (list of target spatial dims) or a multiplicative
    // `scale_factor` against the input's spatial dims.  The exporter has
    // `input_shape` (declared by the caller via ExportOptions), so we can
    // always materialise the absolute size — the lite runtime then needs
    // no shape inference for this op.
    std::vector<int64_t> target_size;
    if (up.size().has_value()) {
        target_size = *up.size();
    } else if (up.scale_factor().has_value()) {
        // Expect input shape (N, C, [D,] H, W); spatial dims = last
        // (rank - 2) axes.  Need at least rank 3.
        if (input_shape.size() < 3) {
            throw std::runtime_error(
                "export_to_tzlite: nn::Upsample with scale_factor requires "
                "input rank >= 3 (got " + std::to_string(input_shape.size()) +
                ")");
        }
        double sf = *up.scale_factor();
        for (size_t k = 2; k < input_shape.size(); ++k) {
            target_size.push_back(
                static_cast<int64_t>(static_cast<double>(input_shape[k]) * sf));
        }
    } else {
        throw std::runtime_error(
            "export_to_tzlite: nn::Upsample needs either size or scale_factor");
    }

    LiteNode node;
    node.op = OpId::Interpolate;
    node.input_ids = {in_id};
    node.attrs.i[0] = up.align_corners() ? 1 : 0;
    node.attrs.i[1] = upsample_mode_code(up.mode());
    node.attrs.extra_i = target_size;
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
    // Batch 2: 1d / 3d pool, adaptive pool, ConvTranspose, InstanceNorm, Identity.
    if (auto* mp = dynamic_cast<nn::MaxPool1d*>(&m)) {
        return emit_maxpool1d(*mp, b, in_id);
    }
    if (auto* mp = dynamic_cast<nn::MaxPool3d*>(&m)) {
        return emit_maxpool3d(*mp, b, in_id);
    }
    if (auto* ap = dynamic_cast<nn::AvgPool1d*>(&m)) {
        return emit_avgpool1d(*ap, b, in_id);
    }
    if (auto* ap = dynamic_cast<nn::AvgPool3d*>(&m)) {
        return emit_avgpool3d(*ap, b, in_id);
    }
    // Adaptive pools — order: 2d before 3d/1d (concrete first), but the
    // classes share no base so dynamic_cast order is purely cosmetic.
    if (auto* p = dynamic_cast<nn::AdaptiveAvgPool2d*>(&m)) {
        return emit_adaptive_avg_pool2d(*p, b, in_id);
    }
    if (auto* p = dynamic_cast<nn::AdaptiveMaxPool2d*>(&m)) {
        return emit_adaptive_max_pool2d(*p, b, in_id);
    }
    if (auto* p = dynamic_cast<nn::AdaptiveAvgPool1d*>(&m)) {
        return emit_adaptive_avg_pool1d(*p, b, in_id);
    }
    if (auto* p = dynamic_cast<nn::AdaptiveMaxPool1d*>(&m)) {
        return emit_adaptive_max_pool1d(*p, b, in_id);
    }
    if (auto* p = dynamic_cast<nn::AdaptiveAvgPool3d*>(&m)) {
        return emit_adaptive_avg_pool3d(*p, b, in_id);
    }
    if (auto* p = dynamic_cast<nn::AdaptiveMaxPool3d*>(&m)) {
        return emit_adaptive_max_pool3d(*p, b, in_id);
    }
    // ConvTranspose 2d/3d — ConvTranspose1d is deferred: its module wraps
    // ConvTranspose2d via runtime unsqueeze, which we cannot replicate at
    // export time without a dedicated OpId or a Reshape/Unsqueeze pass.
    if (auto* c = dynamic_cast<nn::ConvTranspose2d*>(&m)) {
        return emit_conv_transpose2d(*c, b, in_id);
    }
    if (auto* c = dynamic_cast<nn::ConvTranspose3d*>(&m)) {
        return emit_conv_transpose3d(*c, b, in_id);
    }
    // InstanceNorm — the CPU kernel reads N from shape[0], C from shape[1],
    // and accumulates spatial dims from shape[2..], so the same OpId
    // (InstanceNorm) covers 1d/2d/3d transparently.
    if (auto* in1 = dynamic_cast<nn::InstanceNorm1d*>(&m)) {
        return emit_instancenorm(*in1, b, in_id, in1->eps(),
                                 "nn::InstanceNorm1d");
    }
    if (auto* in2 = dynamic_cast<nn::InstanceNorm2d*>(&m)) {
        return emit_instancenorm(*in2, b, in_id, in2->eps(),
                                 "nn::InstanceNorm2d");
    }
    if (auto* in3 = dynamic_cast<nn::InstanceNorm3d*>(&m)) {
        // The affine params live on the inner in2d_ submodule (registered
        // under the name "in2d"). Route the parameter lookups through it.
        const auto& subs = in3->get_submodules();
        auto it = subs.find("in2d");
        if (it == subs.end() || !it->second) {
            throw std::runtime_error(
                "export_to_tzlite: nn::InstanceNorm3d has no registered "
                "'in2d' submodule - internal contract changed.");
        }
        return emit_instancenorm(*it->second, b, in_id, in3->eps(),
                                 "nn::InstanceNorm3d");
    }
    if (auto* em = dynamic_cast<nn::Embedding*>(&m)) {
        return emit_embedding(*em, b, in_id);
    }
    if (dynamic_cast<nn::Identity*>(&m)) {
        // Identity is a strict passthrough at inference. Emit no node and
        // forward the tensor id, mirroring the Dropout passthrough above.
        return in_id;
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
    if (dynamic_cast<nn::Swish*>(&m))     return emit_unary(OpId::Swish,    b, in_id);
    // C.3 audit batch 3: Hardswish / Hardsigmoid now have dedicated OpIds
    // (98, 99) with CPU kernels registered; other backends will be wired
    // in the F.11 pass once parity is validated.
    if (dynamic_cast<nn::Hardswish*>(&m))   return emit_unary(OpId::Hardswish,   b, in_id);
    if (dynamic_cast<nn::Hardsigmoid*>(&m)) return emit_unary(OpId::Hardsigmoid, b, in_id);
    // C.3 audit batch 3: Softmax / LogSoftmax — dim attribute via Module
    // accessor; dispatch_bridge already maps i[0] → AttrKey::Dim.
    if (auto* sm = dynamic_cast<nn::Softmax*>(&m)) {
        return emit_softmax_like(OpId::Softmax, b, in_id, sm->dim());
    }
    if (auto* lsm = dynamic_cast<nn::LogSoftmax*>(&m)) {
        return emit_softmax_like(OpId::LogSoftmax, b, in_id, lsm->dim());
    }
    // C.3 audit batch 3: RMSNorm — weight-only norm with eps attribute.
    if (auto* rn = dynamic_cast<nn::RMSNorm*>(&m)) {
        return emit_rmsnorm(*rn, b, in_id);
    }
    // C.3 audit batch 3: Upsample — emits OpId::Interpolate with mode +
    // align_corners + resolved spatial output_size (we resolve scale
    // factor against b.input_shape so the runtime needs no shape
    // inference). Caveat: the resolved size assumes the export-time input
    // shape; consumers that vary input rank between export and inference
    // must use the absolute `size` constructor rather than `scale_factor`.
    if (auto* up = dynamic_cast<nn::Upsample*>(&m)) {
        return emit_upsample(*up, b, in_id);
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
        "'. Lite exporter supports nn::Linear, nn::Sequential, nn::Identity, "
        "nn::Conv1d, nn::Conv2d, nn::Conv3d, "
        "nn::ConvTranspose2d, nn::ConvTranspose3d, "
        "nn::BatchNorm2d, nn::LayerNorm, nn::GroupNorm, nn::RMSNorm, "
        "nn::InstanceNorm1d, nn::InstanceNorm2d, nn::InstanceNorm3d, "
        "nn::Flatten, nn::Dropout, "
        "nn::MaxPool1d, nn::MaxPool2d, nn::MaxPool3d, "
        "nn::AvgPool1d, nn::AvgPool2d, nn::AvgPool3d, "
        "nn::AdaptiveAvgPool1d/2d/3d, nn::AdaptiveMaxPool1d/2d/3d, "
        "nn::Upsample, nn::Embedding, "
        "nn::Softmax, nn::LogSoftmax, "
        "and the parameter-free / single-alpha activations "
        "(ReLU, Sigmoid, Tanh, GELU, Mish, SELU, Swish, Hardswish, "
        "Hardsigmoid, LeakyReLU, ELU). "
        "Deferred: nn::ConvTranspose1d (needs runtime unsqueeze), "
        "nn::BatchNorm1d / nn::BatchNorm3d (need Reshape pre/post nodes "
        "or dedicated 1d/3d OpIds), nn::EmbeddingBag (two-input forward "
        "doesn't match the single-input visitor), nn::ReflectionPad* / "
        "nn::ReplicationPad* / nn::ConstantPad* / nn::CircularPad* / "
        "nn::ZeroPad2d (no OpId::Pad yet; compose via pad_dim_* autograd "
        "ops), nn::PixelShuffle / nn::PixelUnshuffle / nn::ChannelShuffle "
        "(no dedicated OpIds). File an issue or extend exporter.cpp to "
        "add this layer.");
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
