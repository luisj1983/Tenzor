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

        // Transfer relevant attributes from OpAttributes to TracedOp
        // This is a best-effort mapping of common attributes
        // (kernel_size, stride, padding, etc. for conv/pool ops)
        tracer.record_op(std::move(traced));

        return results;
    };
}

} // namespace jit
} // namespace tenzor
