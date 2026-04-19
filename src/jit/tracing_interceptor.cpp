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

        // Normalization
        case OpId::BatchNorm2dForward: return OpType::BatchNorm2d;
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

        // Indexing
        case OpId::Slice:      return OpType::Slice;
        case OpId::Cat:        return OpType::Cat;

        // Linear
        case OpId::Linear:     return OpType::Linear;

        // Embedding
        case OpId::Embedding:  return OpType::Embedding;

        // Dropout
        case OpId::Dropout:    return OpType::Dropout;

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

        // Register output tensors
        std::vector<std::string> output_ids;
        output_ids.reserve(results.size());
        for (auto& t : results) {
            output_ids.push_back(tracer.register_tensor(t));
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
        copy_float(AttrKey::Exponent, "exponent");
        copy_float(AttrKey::Alpha,    "alpha");
        copy_float(AttrKey::Beta,     "beta");
        copy_float(AttrKey::Min,      "min");
        copy_float(AttrKey::Max,      "max");
        copy_float(AttrKey::Eps,      "eps");
        copy_float(AttrKey::Negative_slope, "negative_slope");
        copy_int(AttrKey::Dim,         "dim");
        copy_int(AttrKey::KernelSize,  "kernel_size");
        copy_int(AttrKey::Stride,      "stride");
        copy_int(AttrKey::Padding,     "padding");
        copy_int(AttrKey::Dilation,    "dilation");
        copy_int(AttrKey::Groups,      "groups");
        copy_hw_pair(AttrKey::KernelSizeH, AttrKey::KernelSizeW, "kernel_size");
        copy_hw_pair(AttrKey::StrideH,     AttrKey::StrideW,     "stride");
        copy_hw_pair(AttrKey::PaddingH,    AttrKey::PaddingW,    "padding");
        copy_hw_pair(AttrKey::DilationH,   AttrKey::DilationW,   "dilation");

        tracer.record_op(std::move(traced));

        return results;
    };
}

} // namespace jit
} // namespace tenzor
