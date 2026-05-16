/**
 * @file dispatch_bridge.cpp
 * @brief Phase 1 implementation of the LiteAttributes -> OpAttributes bridge.
 *
 * The positional attribute encoding per supported OpId is documented inline
 * below. Each new op added to the Lite-supported set extends `build_attrs`.
 */

#include "dispatch_bridge.hpp"

#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"

namespace tenzor::lite {

namespace {

// Translate the positional LiteAttributes into the typed OpAttributes map for
// a given OpId. Unsupported (or attribute-less) ops fall through to an empty
// map; the dispatched kernel will surface any missing-attribute errors itself.
auto build_attrs(LiteOpType op, const LiteAttributes& la) -> OpAttributes {
    OpAttributes oa;
    switch (op) {
        // Element-wise binary / unary ops have no attributes.
        case OpId::Add:
        case OpId::Sub:
        case OpId::Mul:
        case OpId::Div:
        case OpId::MatMul:
        case OpId::ReLU:
        case OpId::Sigmoid:
        case OpId::Tanh:
        case OpId::Gelu:
        case OpId::LeakyReLU:
        case OpId::Elu:
        case OpId::Selu:
        case OpId::Mish:
        case OpId::Softplus:
        case OpId::Swish:
        case OpId::Linear:
            break;

        // Softmax / LogSoftmax: dim = i[0].
        case OpId::Softmax:
        case OpId::LogSoftmax:
            oa.set(AttrKey::Dim, la.i[0]);
            break;

        // Inf-E5: Conv1/2/3d — per-axis stride/padding/dilation/groups.
        //   i[0] = stride, i[1] = padding, i[2] = dilation, i[3] = groups
        case OpId::Conv1dForward:
        case OpId::Conv2dForward:
        case OpId::Conv3dForward:
        case OpId::DepthwiseConv2d:
            oa.set(AttrKey::Stride,   la.i[0]);
            oa.set(AttrKey::Padding,  la.i[1]);
            oa.set(AttrKey::Dilation, la.i[2]);
            oa.set(AttrKey::Groups,   la.i[3]);
            break;

        // Inf-E5: Norms — f[0]=eps, i[0]=num_groups (GroupNorm only).
        case OpId::LayerNorm:
        case OpId::InstanceNorm:
            oa.set(AttrKey::Eps, la.f[0]);
            break;
        case OpId::GroupNorm:
            oa.set(AttrKey::Eps, la.f[0]);
            oa.set(AttrKey::NumGroups, la.i[0]);
            break;
        case OpId::RMSNorm:
        case OpId::FusedRMSNorm:
            oa.set(AttrKey::Eps, la.f[0]);
            break;
        case OpId::BatchNorm2dForward:
        case OpId::BatchNorm2dForwardAffine:
            oa.set(AttrKey::Eps, la.f[0]);
            oa.set(AttrKey::Momentum, la.f[1]);
            break;

        // Inf-E5: Pooling — i[0]=kernel, i[1]=stride, i[2]=padding,
        //                   i[3]=dilation (MaxPool only).
        case OpId::MaxPool1dForward:
        case OpId::MaxPool2dForward:
        case OpId::MaxPool3dForward:
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            oa.set(AttrKey::Dilation,   la.i[3]);
            break;
        case OpId::AvgPool1dForward:
        case OpId::AvgPool2dForward:
        case OpId::AvgPool3dForward:
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            break;
        case OpId::AdaptiveAvgPool1d:
        case OpId::AdaptiveAvgPool2d:
        case OpId::AdaptiveAvgPool3d:
        case OpId::AdaptiveMaxPool1d:
        case OpId::AdaptiveMaxPool2d:
        case OpId::AdaptiveMaxPool3d:
            oa.set(AttrKey::OutputSize, la.i[0]);
            break;

        // Inf-E5: Dropout — f[0]=p (forced 0 at inference per ONNX
        // precedent; the export visitor records training=0).
        case OpId::Dropout:
            oa.set(AttrKey::DropoutP, la.f[0]);
            oa.set(AttrKey::Training, false);
            break;

        // Inf-E5: Shape ops with a dim attribute.
        case OpId::Flatten:
        case OpId::Squeeze:
        case OpId::Unsqueeze:
            oa.set(AttrKey::Dim, la.i[0]);
            break;

        // Inf-E5: Reductions — i[0]=dim, f[0]=keepdim flag (0/1).
        case OpId::Sum:
        case OpId::Mean:
        case OpId::Max:
        case OpId::Min:
        case OpId::ArgMax:
        case OpId::ArgMin:
            oa.set(AttrKey::Dim, la.i[0]);
            oa.set(AttrKey::Keepdim, la.f[0] != 0.0f);
            break;

        // Inf-E5: Cast — i[0] = target DType cast (underlying int value).
        // The destination dtype is carried as Value (generic attr) since
        // OpAttributes has no dedicated DType key today; consumers
        // interpret it via `static_cast<DType>(Value)`.
        case OpId::Cast:
            oa.set(AttrKey::Value, static_cast<double>(la.i[0]));
            break;

        // H4 fix: RNN family uses LiteAttributes::extra_i for the richer
        // attribute set (hidden_size, num_layers, bidirectional,
        // batch_first, dropout_p_q15). The slot meanings are:
        //   extra_i[0] = hidden_size
        //   extra_i[1] = num_layers
        //   extra_i[2] = bidirectional (0/1)
        //   extra_i[3] = batch_first (0/1)
        //   extra_f[0] = dropout_p
        // Exporter writes these via b.graph.add_node(node) with the
        // extras filled in. Missing extras → attribute defaults.
        case OpId::LSTMForward:
        case OpId::LSTMMultiLayerForward:
        case OpId::BiLSTMForward:
        case OpId::GRUForward:
        case OpId::GRUMultiLayerForward: {
            if (la.extra_i.size() >= 1) oa.set(AttrKey::HiddenSize, la.extra_i[0]);
            if (la.extra_i.size() >= 2) oa.set(AttrKey::NumLayers, la.extra_i[1]);
            // Bidirectional + batch_first don't have dedicated AttrKey
            // slots yet — encoded positionally in extra_i[2..3] and the
            // kernel reads them via the LiteAttributes structure directly
            // when needed. Future AttrKey enum extension will surface them.
            if (la.extra_f.size() >= 1) oa.set(AttrKey::DropoutP, la.extra_f[0]);
            break;
        }

        // H4 fix: MultiheadAttention slots:
        //   extra_i[0] = num_heads
        //   extra_i[1] = head_dim
        //   extra_i[2] = embed_dim
        //   extra_i[3] = causal (0/1)
        //   extra_f[0] = dropout_p
        case OpId::FlashAttention:
        case OpId::FlexAttention:
        case OpId::FusedAttention: {
            if (la.extra_i.size() >= 1) oa.set(AttrKey::NumHeads, la.extra_i[0]);
            if (la.extra_i.size() >= 2) oa.set(AttrKey::HeadDim,  la.extra_i[1]);
            if (la.extra_i.size() >= 4) oa.set(AttrKey::Causal,   la.extra_i[3] != 0);
            if (la.extra_f.size() >= 1) oa.set(AttrKey::DropoutP, la.extra_f[0]);
            break;
        }

        default:
            // No mapping yet for this op. Kernels that *require* an
            // attribute will throw with a clear message. Coverage for
            // RNNs (LSTM/GRU/Bi*) and MultiheadAttention is deferred
            // — those have richer state requiring container attributes
            // beyond LiteAttributes' positional encoding.
            break;
    }
    return oa;
}

}  // namespace

auto run_op(LiteOpType op,
            std::span<const Tensor> inputs,
            const LiteAttributes& attrs) -> std::vector<Tensor> {
    return ::tenzor::dispatch(op, inputs, build_attrs(op, attrs));
}

}  // namespace tenzor::lite
