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

        // Inf-E5 / C.3 audit: Conv1/2/3d — per-axis stride/padding/dilation/groups.
        // Scalar fallback (used when extras are empty / symmetric):
        //   i[0] = stride, i[1] = padding, i[2] = dilation, i[3] = groups
        // Per-axis extras (when present) override the scalar via per-axis keys:
        //   Conv1d:  extra_i = [stride_w, padding_w, dilation_w, groups, kernel_w]
        //   Conv2d:  extra_i = [stride_h, stride_w, padding_h, padding_w,
        //                       dilation_h, dilation_w, groups,
        //                       kernel_h, kernel_w]
        //   Conv3d:  extra_i = [stride_d, stride_h, stride_w,
        //                       padding_d, padding_h, padding_w,
        //                       dilation_d, dilation_h, dilation_w,
        //                       groups, kernel_d, kernel_h, kernel_w]
        // DepthwiseConv2d uses the Conv2d layout.
        case OpId::Conv1dForward: {
            oa.set(AttrKey::Stride,   la.i[0]);
            oa.set(AttrKey::Padding,  la.i[1]);
            oa.set(AttrKey::Dilation, la.i[2]);
            oa.set(AttrKey::Groups,   la.i[3]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::StrideW,   ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::PaddingW,  ei[1]);
            if (ei.size() >= 3) oa.set(AttrKey::DilationW, ei[2]);
            if (ei.size() >= 4) oa.set(AttrKey::Groups,    ei[3]);
            if (ei.size() >= 5) oa.set(AttrKey::KernelSizeW, ei[4]);
            break;
        }
        case OpId::Conv2dForward:
        case OpId::DepthwiseConv2d: {
            oa.set(AttrKey::Stride,   la.i[0]);
            oa.set(AttrKey::Padding,  la.i[1]);
            oa.set(AttrKey::Dilation, la.i[2]);
            oa.set(AttrKey::Groups,   la.i[3]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::StrideH,   ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::StrideW,   ei[1]);
            if (ei.size() >= 3) oa.set(AttrKey::PaddingH,  ei[2]);
            if (ei.size() >= 4) oa.set(AttrKey::PaddingW,  ei[3]);
            if (ei.size() >= 5) oa.set(AttrKey::DilationH, ei[4]);
            if (ei.size() >= 6) oa.set(AttrKey::DilationW, ei[5]);
            if (ei.size() >= 7) oa.set(AttrKey::Groups,    ei[6]);
            if (ei.size() >= 8) oa.set(AttrKey::KernelSizeH, ei[7]);
            if (ei.size() >= 9) oa.set(AttrKey::KernelSizeW, ei[8]);
            break;
        }
        case OpId::Conv3dForward: {
            oa.set(AttrKey::Stride,   la.i[0]);
            oa.set(AttrKey::Padding,  la.i[1]);
            oa.set(AttrKey::Dilation, la.i[2]);
            oa.set(AttrKey::Groups,   la.i[3]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::StrideD,   ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::StrideH,   ei[1]);
            if (ei.size() >= 3) oa.set(AttrKey::StrideW,   ei[2]);
            if (ei.size() >= 4) oa.set(AttrKey::PaddingD,  ei[3]);
            if (ei.size() >= 5) oa.set(AttrKey::PaddingH,  ei[4]);
            if (ei.size() >= 6) oa.set(AttrKey::PaddingW,  ei[5]);
            if (ei.size() >= 7) oa.set(AttrKey::DilationD, ei[6]);
            if (ei.size() >= 8) oa.set(AttrKey::DilationH, ei[7]);
            if (ei.size() >= 9) oa.set(AttrKey::DilationW, ei[8]);
            if (ei.size() >= 10) oa.set(AttrKey::Groups,        ei[9]);
            if (ei.size() >= 11) oa.set(AttrKey::KernelSizeD,   ei[10]);
            if (ei.size() >= 12) oa.set(AttrKey::KernelSizeH,   ei[11]);
            if (ei.size() >= 13) oa.set(AttrKey::KernelSizeW,   ei[12]);
            break;
        }

        // C.3 audit batch 2: ConvTranspose 2d/3d.
        // ConvTranspose2d: extra_i layout matches Conv2d plus 2 trailing
        // output_padding slots:
        //   [sH, sW, pH, pW, dH, dW, groups, kH, kW, opH, opW]
        // ConvTranspose3d:
        //   [sD, sH, sW, pD, pH, pW, dD, dH, dW, groups, kD, kH, kW,
        //    opD, opH, opW]
        case OpId::ConvTranspose2dForward: {
            oa.set(AttrKey::Stride,   la.i[0]);
            oa.set(AttrKey::Padding,  la.i[1]);
            oa.set(AttrKey::Dilation, la.i[2]);
            oa.set(AttrKey::Groups,   la.i[3]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1)  oa.set(AttrKey::StrideH,        ei[0]);
            if (ei.size() >= 2)  oa.set(AttrKey::StrideW,        ei[1]);
            if (ei.size() >= 3)  oa.set(AttrKey::PaddingH,       ei[2]);
            if (ei.size() >= 4)  oa.set(AttrKey::PaddingW,       ei[3]);
            if (ei.size() >= 5)  oa.set(AttrKey::DilationH,      ei[4]);
            if (ei.size() >= 6)  oa.set(AttrKey::DilationW,      ei[5]);
            if (ei.size() >= 7)  oa.set(AttrKey::Groups,         ei[6]);
            if (ei.size() >= 8)  oa.set(AttrKey::KernelSizeH,    ei[7]);
            if (ei.size() >= 9)  oa.set(AttrKey::KernelSizeW,    ei[8]);
            if (ei.size() >= 10) oa.set(AttrKey::OutputPaddingH, ei[9]);
            if (ei.size() >= 11) oa.set(AttrKey::OutputPaddingW, ei[10]);
            break;
        }
        case OpId::ConvTranspose3dForward: {
            oa.set(AttrKey::Stride,   la.i[0]);
            oa.set(AttrKey::Padding,  la.i[1]);
            oa.set(AttrKey::Dilation, la.i[2]);
            oa.set(AttrKey::Groups,   la.i[3]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1)  oa.set(AttrKey::StrideD,        ei[0]);
            if (ei.size() >= 2)  oa.set(AttrKey::StrideH,        ei[1]);
            if (ei.size() >= 3)  oa.set(AttrKey::StrideW,        ei[2]);
            if (ei.size() >= 4)  oa.set(AttrKey::PaddingD,       ei[3]);
            if (ei.size() >= 5)  oa.set(AttrKey::PaddingH,       ei[4]);
            if (ei.size() >= 6)  oa.set(AttrKey::PaddingW,       ei[5]);
            if (ei.size() >= 7)  oa.set(AttrKey::DilationD,      ei[6]);
            if (ei.size() >= 8)  oa.set(AttrKey::DilationH,      ei[7]);
            if (ei.size() >= 9)  oa.set(AttrKey::DilationW,      ei[8]);
            if (ei.size() >= 10) oa.set(AttrKey::Groups,         ei[9]);
            if (ei.size() >= 11) oa.set(AttrKey::KernelSizeD,    ei[10]);
            if (ei.size() >= 12) oa.set(AttrKey::KernelSizeH,    ei[11]);
            if (ei.size() >= 13) oa.set(AttrKey::KernelSizeW,    ei[12]);
            if (ei.size() >= 14) oa.set(AttrKey::OutputPaddingD, ei[13]);
            if (ei.size() >= 15) oa.set(AttrKey::OutputPaddingH, ei[14]);
            if (ei.size() >= 16) oa.set(AttrKey::OutputPaddingW, ei[15]);
            break;
        }

        // Inf-E5: Norms — f[0]=eps, i[0]=num_groups (GroupNorm only).
        case OpId::InstanceNorm:
            oa.set(AttrKey::Eps, la.f[0]);
            break;
        // C.3 audit: LayerNorm — f[0]=eps, normalized_shape carried in
        //   extra_i as the trailing dims to reduce. The kernel reads it via
        //   AttrKey::NormalizedShape (comma-separated int list).
        case OpId::LayerNorm: {
            oa.set(AttrKey::Eps, la.f[0]);
            if (!la.extra_i.empty()) {
                std::string norm_shape_str;
                for (size_t k = 0; k < la.extra_i.size(); ++k) {
                    if (k > 0) norm_shape_str += ",";
                    norm_shape_str += std::to_string(la.extra_i[k]);
                }
                oa.set(AttrKey::NormalizedShape, std::string_view(norm_shape_str));
            }
            break;
        }
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

        // Inf-E5 / C.3: Pooling — i[0]=kernel, i[1]=stride, i[2]=padding,
        //   i[3]=dilation (MaxPool only).
        // Per-axis 2D extras (when present) override scalar via per-axis keys:
        //   MaxPool2d: extra_i = [kH, kW, sH, sW, pH, pW, dH, dW,
        //                         ceil_mode (0/1)]
        //   AvgPool2d: extra_i = [kH, kW, sH, sW, pH, pW,
        //                         ceil_mode (0/1), count_include_pad (0/1)]
        case OpId::MaxPool1dForward:
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            oa.set(AttrKey::Dilation,   la.i[3]);
            break;
        case OpId::MaxPool2dForward: {
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            oa.set(AttrKey::Dilation,   la.i[3]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::KernelSizeH, ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::KernelSizeW, ei[1]);
            if (ei.size() >= 3) oa.set(AttrKey::StrideH,     ei[2]);
            if (ei.size() >= 4) oa.set(AttrKey::StrideW,     ei[3]);
            if (ei.size() >= 5) oa.set(AttrKey::PaddingH,    ei[4]);
            if (ei.size() >= 6) oa.set(AttrKey::PaddingW,    ei[5]);
            if (ei.size() >= 7) oa.set(AttrKey::DilationH,   ei[6]);
            if (ei.size() >= 8) oa.set(AttrKey::DilationW,   ei[7]);
            if (ei.size() >= 9) oa.set(AttrKey::CeilMode,    ei[8] != 0);
            break;
        }
        case OpId::MaxPool3dForward: {
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            oa.set(AttrKey::Dilation,   la.i[3]);
            // Per-axis extras (C.3 audit batch 2):
            //   extra_i = [kD, kH, kW, sD, sH, sW, pD, pH, pW,
            //              dD, dH, dW, ceil_mode (0/1)]
            const auto& ei = la.extra_i;
            if (ei.size() >= 1)  oa.set(AttrKey::KernelSizeD, ei[0]);
            if (ei.size() >= 2)  oa.set(AttrKey::KernelSizeH, ei[1]);
            if (ei.size() >= 3)  oa.set(AttrKey::KernelSizeW, ei[2]);
            if (ei.size() >= 4)  oa.set(AttrKey::StrideD,     ei[3]);
            if (ei.size() >= 5)  oa.set(AttrKey::StrideH,     ei[4]);
            if (ei.size() >= 6)  oa.set(AttrKey::StrideW,     ei[5]);
            if (ei.size() >= 7)  oa.set(AttrKey::PaddingD,    ei[6]);
            if (ei.size() >= 8)  oa.set(AttrKey::PaddingH,    ei[7]);
            if (ei.size() >= 9)  oa.set(AttrKey::PaddingW,    ei[8]);
            if (ei.size() >= 10) oa.set(AttrKey::DilationD,   ei[9]);
            if (ei.size() >= 11) oa.set(AttrKey::DilationH,   ei[10]);
            if (ei.size() >= 12) oa.set(AttrKey::DilationW,   ei[11]);
            if (ei.size() >= 13) oa.set(AttrKey::CeilMode,    ei[12] != 0);
            break;
        }
        case OpId::AvgPool1dForward:
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            break;
        case OpId::AvgPool2dForward: {
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::KernelSizeH, ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::KernelSizeW, ei[1]);
            if (ei.size() >= 3) oa.set(AttrKey::StrideH,     ei[2]);
            if (ei.size() >= 4) oa.set(AttrKey::StrideW,     ei[3]);
            if (ei.size() >= 5) oa.set(AttrKey::PaddingH,    ei[4]);
            if (ei.size() >= 6) oa.set(AttrKey::PaddingW,    ei[5]);
            if (ei.size() >= 7) oa.set(AttrKey::CeilMode,        ei[6] != 0);
            if (ei.size() >= 8) oa.set(AttrKey::CountIncludePad, ei[7] != 0);
            break;
        }
        case OpId::AvgPool3dForward: {
            oa.set(AttrKey::KernelSize, la.i[0]);
            oa.set(AttrKey::Stride,     la.i[1]);
            oa.set(AttrKey::Padding,    la.i[2]);
            // Per-axis extras (C.3 audit batch 2):
            //   extra_i = [kD, kH, kW, sD, sH, sW, pD, pH, pW,
            //              ceil_mode (0/1), count_include_pad (0/1)]
            const auto& ei = la.extra_i;
            if (ei.size() >= 1)  oa.set(AttrKey::KernelSizeD, ei[0]);
            if (ei.size() >= 2)  oa.set(AttrKey::KernelSizeH, ei[1]);
            if (ei.size() >= 3)  oa.set(AttrKey::KernelSizeW, ei[2]);
            if (ei.size() >= 4)  oa.set(AttrKey::StrideD,     ei[3]);
            if (ei.size() >= 5)  oa.set(AttrKey::StrideH,     ei[4]);
            if (ei.size() >= 6)  oa.set(AttrKey::StrideW,     ei[5]);
            if (ei.size() >= 7)  oa.set(AttrKey::PaddingD,    ei[6]);
            if (ei.size() >= 8)  oa.set(AttrKey::PaddingH,    ei[7]);
            if (ei.size() >= 9)  oa.set(AttrKey::PaddingW,    ei[8]);
            if (ei.size() >= 10) oa.set(AttrKey::CeilMode,        ei[9]  != 0);
            if (ei.size() >= 11) oa.set(AttrKey::CountIncludePad, ei[10] != 0);
            break;
        }
        // C.3 audit batch 2: Adaptive pooling.
        // 1d kernels read AttrKey::OutputSize (scalar).
        // 2d kernels read AttrKey::OutputSizeH + OutputSizeW.
        // 3d kernels read AttrKey::OutputSizeD + OutputSizeH + OutputSizeW.
        // The exporter packs the per-axis values into `extra_i`:
        //   1d: extra_i = [output] (also i[0] as scalar fallback)
        //   2d: extra_i = [output_h, output_w]
        //   3d: extra_i = [output_d, output_h, output_w]
        case OpId::AdaptiveAvgPool1d:
        case OpId::AdaptiveMaxPool1d:
            oa.set(AttrKey::OutputSize, la.i[0]);
            break;
        case OpId::AdaptiveAvgPool2d:
        case OpId::AdaptiveMaxPool2d: {
            // Backward-compatible default: also emit the legacy scalar key
            // so any consumer keyed on AttrKey::OutputSize still resolves.
            oa.set(AttrKey::OutputSize, la.i[0]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::OutputSizeH, ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::OutputSizeW, ei[1]);
            break;
        }
        case OpId::AdaptiveAvgPool3d:
        case OpId::AdaptiveMaxPool3d: {
            oa.set(AttrKey::OutputSize, la.i[0]);
            const auto& ei = la.extra_i;
            if (ei.size() >= 1) oa.set(AttrKey::OutputSizeD, ei[0]);
            if (ei.size() >= 2) oa.set(AttrKey::OutputSizeH, ei[1]);
            if (ei.size() >= 3) oa.set(AttrKey::OutputSizeW, ei[2]);
            break;
        }

        // C.3 audit fix: the CPU Dropout kernel reads AttrKey::P (not
        // DropoutP) and AttrKey::Training. Inference-only Lite runtime
        // forces training=false so the kernel becomes identity (scaled
        // mask of all-ones).
        case OpId::Dropout:
            oa.set(AttrKey::P,        la.f[0]);
            oa.set(AttrKey::Training, false);
            break;

        // C.3 audit fix: Flatten reads StartDim / EndDim, not Dim. Encoded
        //   i[0] = start_dim, i[1] = end_dim
        case OpId::Flatten:
            oa.set(AttrKey::StartDim, la.i[0]);
            oa.set(AttrKey::EndDim,   la.i[1]);
            break;

        // Inf-E5: Shape ops with a single dim attribute.
        case OpId::Squeeze:
        case OpId::Unsqueeze:
            oa.set(AttrKey::Dim, la.i[0]);
            break;

        // C.3 audit: Embedding reads optional PaddingIdx; encoded
        //   i[0] = padding_idx (negative = none)
        case OpId::Embedding:
        case OpId::EmbeddingWithBoundsCheck:
            if (la.i[0] >= 0) oa.set(AttrKey::PaddingIdx, la.i[0]);
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
