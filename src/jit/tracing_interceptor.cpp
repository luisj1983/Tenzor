/**
 * @file tracing_interceptor.cpp
 * @brief Implementation of the dispatch-level tracing interceptor
 */

#include "tenzor/jit/tracing_interceptor.hpp"
#include <sstream>

namespace tenzor {
namespace jit {

// ============================================================================
// OpId -> OpType mapping
// ============================================================================

auto opid_to_optype(OpId op) -> std::optional<OpType> {
    switch (op) {
        // Arithmetic
        case OpId::Add:        return OpType::Add;
        case OpId::Sub:        return OpType::Sub;
        case OpId::Mul:        return OpType::Mul;
        case OpId::Div:        return OpType::Div;

        // Matrix operations
        case OpId::MatMul:     return OpType::MatMul;
        case OpId::Bmm:        return OpType::Bmm;

        // Activations
        case OpId::ReLU:       return OpType::ReLU;
        case OpId::Sigmoid:    return OpType::Sigmoid;
        case OpId::Tanh:       return OpType::Tanh;
        case OpId::Softmax:    return OpType::Softmax;
        case OpId::LogSoftmax: return OpType::LogSoftmax;
        case OpId::Gelu:       return OpType::GELU;

        // Pooling
        case OpId::MaxPool2dForward:       return OpType::MaxPool2d;
        case OpId::AvgPool2dForward:       return OpType::AvgPool2d;
        case OpId::AdaptiveAvgPool2d:      return OpType::AdaptiveAvgPool2d;

        // Convolution
        case OpId::Conv2dForward:     return OpType::Conv2d;
        // 1D/3D convs map to the same ONNX "Conv" op; the kernel_shape rank
        // (emitted from the kernel_size vec attr) distinguishes them.
        case OpId::Conv1dForward:     return OpType::Conv2d;
        case OpId::Conv3dForward:     return OpType::Conv2d;
        // Transposed (fractionally-strided) convs map to ONNX "ConvTranspose".
        case OpId::ConvTranspose1dForward: return OpType::ConvTranspose;
        case OpId::ConvTranspose2dForward: return OpType::ConvTranspose;
        case OpId::ConvTranspose3dForward: return OpType::ConvTranspose;

        // Normalization
        case OpId::BatchNorm2dForward:       return OpType::BatchNorm2d;
        case OpId::BatchNorm2dForwardAffine: return OpType::BatchNorm2d;
        case OpId::LayerNorm:  return OpType::LayerNorm;

        // Reductions
        case OpId::Sum:        return OpType::Sum;
        case OpId::Mean:       return OpType::Mean;
        case OpId::Max:        return OpType::Max;
        case OpId::Min:        return OpType::Min;

        // Element-wise math
        case OpId::Exp:        return OpType::Exp;
        case OpId::Log:        return OpType::Log;
        case OpId::Sqrt:       return OpType::Sqrt;
        case OpId::Pow:        return OpType::Pow;
        case OpId::Abs:        return OpType::Abs;
        case OpId::Neg:        return OpType::Neg;
        case OpId::Clamp:      return OpType::Clamp;

        // Shape operations
        case OpId::Reshape:    return OpType::Reshape;
        case OpId::Transpose:  return OpType::Transpose;
        case OpId::Permute:    return OpType::Permute;
        case OpId::Squeeze:    return OpType::Squeeze;
        case OpId::Unsqueeze:  return OpType::Unsqueeze;
        case OpId::Flatten:    return OpType::Flatten;
        case OpId::Stack:      return OpType::Stack;
        // Eager broadcast_to dispatches OpId::Expand; the IR represents
        // it with OpType::Broadcast (handled by handle_broadcast which
        // lowers to stablehlo.broadcast_in_dim).
        case OpId::Expand:     return OpType::Broadcast;

        // Indexing
        case OpId::Slice:      return OpType::Slice;
        case OpId::Cat:        return OpType::Cat;
        case OpId::Where:      return OpType::Where;

        // Linear
        case OpId::Linear:     return OpType::Linear;

        // Embedding
        case OpId::Embedding:  return OpType::Embedding;

        // Dropout
        case OpId::Dropout:    return OpType::Dropout;

        // Cast (GPU dispatch path; CPU rewrites in Tensor::to(dtype)).
        case OpId::Cast:       return OpType::Cast;

        // Index ops
        case OpId::IndexSelect: return OpType::IndexSelect;

        // Vision
        case OpId::Interpolate: return OpType::Interpolate;

        default:
            return std::nullopt;
    }
}

// ============================================================================
// Tracing interceptor factory
// ============================================================================

auto make_tracing_interceptor(
    Tracer& tracer,
    std::function<void(OpId)> on_graph_break)
    -> DispatchInterceptor {

    return [&tracer, on_graph_break = std::move(on_graph_break)](
        OpId op,
        std::span<const Tensor> inputs,
        const OpAttributes& attrs,
        DispatchNext next) -> std::vector<Tensor> {

        // Try to map dispatch OpId to IR OpType
        auto op_type = opid_to_optype(op);

        // Execute eagerly first (we always run the op)
        auto results = next(op, inputs, attrs);

        if (!op_type) {
            // Graph break: unmappable operation
            if (on_graph_break) {
                on_graph_break(op);
            }
            return results;
        }

        // Register input tensors
        std::vector<std::string> input_ids;
        input_ids.reserve(inputs.size());
        for (auto& t : inputs) {
            input_ids.push_back(tracer.register_tensor(t));
        }

        // Reorder BN2d's affine forward inputs from the eager kernel's
        // canonical (x, mean, var, weight, bias) into the IR's expected
        // (x, weight, bias, mean, var) so the MLIR lowering's BN2d
        // handler (which mirrors stablehlo.batch_norm_inference's
        // operand order) wires up the right scale/offset/mean/var
        // tensors. Without this the lowered graph silently treats the
        // running stats as scale/bias and vice versa.
        if (op == OpId::BatchNorm2dForwardAffine && input_ids.size() == 5) {
            auto [x_id, m_id, v_id, w_id, b_id] = std::tuple{
                input_ids[0], input_ids[1], input_ids[2],
                input_ids[3], input_ids[4]};
            input_ids = {x_id, w_id, b_id, m_id, v_id};
        }

        // Register output tensors. Some fused forward kernels return
        // auxiliary tensors (saved-mean, saved-rrms, indices, …) that
        // the IR side doesn't model; surface only the *primary* output
        // (results[0]) to avoid building Values with no consumer that
        // later passes can't infer shapes for.
        std::vector<std::string> output_ids;
        if (op == OpId::BatchNorm2dForwardAffine ||
            op == OpId::BatchNorm2dForward) {
            if (!results.empty()) {
                output_ids.push_back(tracer.register_tensor(results[0]));
            }
        } else {
            output_ids.reserve(results.size());
            for (auto& t : results) {
                output_ids.push_back(tracer.register_tensor(t));
            }
        }

        // Record the operation
        TracedOp traced(*op_type, std::move(input_ids), std::move(output_ids));

        // Transfer attributes from OpAttributes into the TracedOp's string-keyed
        // maps so graph.cpp's replay (which looks them up by name) can find
        // them. Previously the traced op lost all attributes, so ops like Pow
        // replayed with exponent=0 and mean/sum(dim) replayed with dim=0.
        auto copy_float = [&](AttrKey k, const char* name) {
            if (attrs.has(k)) traced.attrs[name] = static_cast<float>(attrs.get_float(k));
        };
        auto copy_int = [&](AttrKey k, const char* name) {
            if (attrs.has(k)) traced.int_attrs[name] = attrs.get_int(k);
        };
        // For 2-D conv-family ops the eager side sets both a scalar (e.g.
        // AttrKey::Padding = padding_h_) and a pair (PaddingH/PaddingW).
        // The scalar alone drops the W axis; record the pair as a vec so
        // consumers that want rectangular configs (e.g. the ONNX exporter's
        // JIT→ONNX Conv2d translator) find it.
        auto copy_hw_pair = [&](AttrKey kh, AttrKey kw, const char* name) {
            if (attrs.has(kh) && attrs.has(kw)) {
                traced.vec_attrs[name] = {attrs.get_int(kh), attrs.get_int(kw)};
            }
        };
        // 3D (depth, height, width) variant for Conv3d etc. Called after the
        // 2D pairs so a present D dim upgrades the vec to [D, H, W].
        auto copy_dhw_triple = [&](AttrKey kd, AttrKey kh, AttrKey kw,
                                   const char* name) {
            if (attrs.has(kd) && attrs.has(kh) && attrs.has(kw)) {
                traced.vec_attrs[name] = {attrs.get_int(kd), attrs.get_int(kh),
                                          attrs.get_int(kw)};
            }
        };
        copy_float(AttrKey::Exponent, "exponent");
        copy_float(AttrKey::Alpha,    "alpha");
        copy_float(AttrKey::Beta,     "beta");
        copy_float(AttrKey::Min,      "min");
        copy_float(AttrKey::Max,      "max");
        copy_float(AttrKey::Eps,      "eps");
        copy_float(AttrKey::Negative_slope, "negative_slope");
        copy_int(AttrKey::Dim,         "dim");
        // Reduction keepdim flag — graph.cpp's infer_types() needs this to
        // correctly compute reduced shapes (without it, infer_types
        // defaults to keepdim=false and silently drops the kept dim).
        auto copy_bool = [&](AttrKey k, const char* name) {
            if (attrs.has(k)) traced.bool_attrs[name] = attrs.get_bool(k, false);
        };
        copy_bool(AttrKey::Keepdim, "keepdim");
        // Slice indices: Start/End/Step are scalar ints used by Slice and
        // a handful of indexing ops. The MLIR lowerer needs them to
        // emit stablehlo.slice.
        copy_int(AttrKey::Start, "start");
        copy_int(AttrKey::End,   "end");
        copy_int(AttrKey::Step,  "step");
        // Cast op target dtype (stored as uint8 cast to int64).
        copy_int(AttrKey::TargetDtype, "target_dtype");
        // Reshape stores its target shape as a comma-separated string under
        // AttrKey::Shape; parse to an int-list so consumers don't have to.
        auto copy_int_list_to_vec = [&](AttrKey k, const char* name) {
            if (attrs.has(k)) {
                auto v = attrs.get_int_list(k);
                if (!v.empty()) traced.vec_attrs[name] = std::move(v);
            }
        };
        copy_int_list_to_vec(AttrKey::Shape, "shape");
        copy_int_list_to_vec(AttrKey::Dims,  "dims");
        copy_int_list_to_vec(AttrKey::Starts, "starts");
        copy_int_list_to_vec(AttrKey::Ends,   "ends");
        copy_int(AttrKey::KernelSize,  "kernel_size");
        copy_int(AttrKey::Stride,      "stride");
        copy_int(AttrKey::Padding,     "padding");
        copy_int(AttrKey::Dilation,    "dilation");
        copy_int(AttrKey::OutputPadding, "output_padding");
        copy_int(AttrKey::Groups,      "groups");
        copy_hw_pair(AttrKey::KernelSizeH, AttrKey::KernelSizeW, "kernel_size");
        copy_hw_pair(AttrKey::StrideH,     AttrKey::StrideW,     "stride");
        copy_hw_pair(AttrKey::PaddingH,    AttrKey::PaddingW,    "padding");
        copy_hw_pair(AttrKey::DilationH,   AttrKey::DilationW,   "dilation");
        copy_hw_pair(AttrKey::OutputPaddingH, AttrKey::OutputPaddingW, "output_padding");
        copy_dhw_triple(AttrKey::KernelSizeD, AttrKey::KernelSizeH, AttrKey::KernelSizeW, "kernel_size");
        copy_dhw_triple(AttrKey::StrideD,     AttrKey::StrideH,     AttrKey::StrideW,     "stride");
        copy_dhw_triple(AttrKey::PaddingD,    AttrKey::PaddingH,    AttrKey::PaddingW,    "padding");
        copy_dhw_triple(AttrKey::DilationD,   AttrKey::DilationH,   AttrKey::DilationW,   "dilation");
        copy_dhw_triple(AttrKey::OutputPaddingD, AttrKey::OutputPaddingH, AttrKey::OutputPaddingW, "output_padding");
        // 1-D conv-family ops set only scalar KernelSize/Stride/Padding/
        // Dilation/OutputPadding (no per-axis pairs/triples fire above). Emit
        // them as rank-1 vecs so the ONNX exporter produces a faithful
        // 1-spatial-dim Conv/ConvTranspose (rank-1 kernel_shape/pads/etc).
        if (op == OpId::Conv1dForward || op == OpId::ConvTranspose1dForward) {
            auto vec1 = [&](AttrKey k, const char* name) {
                if (attrs.has(k)) traced.vec_attrs[name] = {attrs.get_int(k)};
            };
            vec1(AttrKey::KernelSize,    "kernel_size");
            vec1(AttrKey::Stride,        "stride");
            vec1(AttrKey::Padding,       "padding");
            vec1(AttrKey::Dilation,      "dilation");
            vec1(AttrKey::OutputPadding, "output_padding");
        }

        tracer.record_op(std::move(traced));

        return results;
    };
}

} // namespace jit
} // namespace tenzor
