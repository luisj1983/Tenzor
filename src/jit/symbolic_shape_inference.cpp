/**
 * @file symbolic_shape_inference.cpp
 * @brief Implementation of standalone symbolic shape inference for JIT nodes
 */

#include "tenzor/jit/symbolic_shape_inference.hpp"
#include "tenzor/jit/tracer.hpp"

#include <algorithm>
#include <stdexcept>

namespace tenzor {
namespace jit {

// ============================================================================
// Helper: gather symbolic shapes from node inputs
// ============================================================================

auto SymbolicShapeInference::gather_input_shapes(const Node* node) -> std::vector<SymbolicShape> {
    std::vector<SymbolicShape> shapes;
    shapes.reserve(node->inputs().size());
    for (const auto& input : node->inputs()) {
        if (input->has_symbolic_shape()) {
            shapes.push_back(input->symbolic_shape());
        } else {
            shapes.push_back(SymbolicShape::from_concrete(input->shape()));
        }
    }
    return shapes;
}

// ============================================================================
// Main dispatch
// ============================================================================

auto SymbolicShapeInference::infer(const Node* node) -> std::vector<SymbolicShape> {
    switch (node->op_type()) {
        // Binary elementwise: broadcast
        case OpType::Add:
        case OpType::Sub:
        case OpType::Mul:
        case OpType::Div:
            return infer_elementwise(node);

        // Unary elementwise: preserve shape
        case OpType::ReLU:
        case OpType::Sigmoid:
        case OpType::Tanh:
        case OpType::Exp:
        case OpType::Log:
        case OpType::Sqrt:
        case OpType::Pow:
        case OpType::Abs:
        case OpType::Neg:
        case OpType::Clamp:
        case OpType::Sin:
        case OpType::Cos:
        case OpType::Rsqrt:
        case OpType::Dropout:
        case OpType::GELU:
        case OpType::Softmax:
        case OpType::LogSoftmax:
        case OpType::BatchNorm2d:
        case OpType::LayerNorm: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                return {input_shapes[0]};
            }
            return {};
        }

        case OpType::MatMul:
        case OpType::Bmm:
            return infer_matmul(node);

        case OpType::Conv2d:
            return infer_conv2d(node);

        case OpType::Reshape:
            return infer_reshape(node);

        case OpType::Sum:
        case OpType::Mean:
        case OpType::Max:
        case OpType::Min:
        // Var/Std/Prod are axis reductions carrying the same dim/dims/keepdim
        // attrs (the tracing interceptor copies them generically), so they share
        // the reduction shape rule. Without this they fell to `default -> {}` and
        // froze downstream dynamic dims to the trace-time size in SymbolicTracePass.
        case OpType::Prod:
        case OpType::Var:
        case OpType::Std:
            return infer_reduction(node);

        case OpType::Transpose:
            return infer_transpose(node);

        case OpType::Permute:
            return infer_permute(node);

        case OpType::Linear:
            return infer_linear(node);

        default:
            // For unhandled ops, return empty (no inference available)
            return {};
    }
}

// ============================================================================
// Elementwise: binary broadcast or unary pass-through
// ============================================================================

auto SymbolicShapeInference::infer_elementwise(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);

    if (input_shapes.size() >= 2) {
        // Binary op: broadcast the two input shapes
        return {broadcast_symbolic_shapes(input_shapes[0], input_shapes[1])};
    }
    if (!input_shapes.empty()) {
        // Unary op: preserve input shape
        return {input_shapes[0]};
    }
    return {};
}

// ============================================================================
// MatMul / Bmm
// ============================================================================

auto SymbolicShapeInference::infer_matmul(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.size() < 2) {
        return {};
    }

    auto& lhs = input_shapes[0];
    auto& rhs = input_shapes[1];

    if (node->op_type() == OpType::Bmm) {
        // Batch matmul: (B, M, K) @ (B, K, N) -> (B, M, N)
        if (lhs.rank() == 3 && rhs.rank() == 3) {
            return {SymbolicShape({lhs[0], lhs[1], rhs[2]})};
        }
        return {};
    }

    // torch.matmul 1-D operand promotion (JIT-047): a 1-D operand is promoted by
    // prepending (lhs) / appending (rhs) a 1, then the added dim is dropped from
    // the result. Previously these produced no symbolic shape, freezing dims.
    if (lhs.rank() == 1 && rhs.rank() == 1) {
        return {SymbolicShape(std::vector<SymbolicDim>{})};  // dot -> scalar
    }
    if (lhs.rank() == 1 && rhs.rank() >= 2) {
        // (K,) @ (..., K, N) -> (..., N)
        SymbolicShape out(std::vector<SymbolicDim>(
            rhs.dims().begin(), rhs.dims().end() - 2));
        out.push_back(rhs[rhs.rank() - 1]);
        return {std::move(out)};
    }
    if (lhs.rank() >= 2 && rhs.rank() == 1) {
        // (..., M, K) @ (K,) -> (..., M)
        SymbolicShape out(std::vector<SymbolicDim>(
            lhs.dims().begin(), lhs.dims().end() - 1));
        return {std::move(out)};
    }

    // General MatMul: (..., M, K) @ (..., K, N) -> (..., M, N)
    // Broadcast the leading batch dims (everything but the last two) of both
    // operands, mirroring torch.matmul and the concrete infer_types path in
    // graph.cpp. The naive "lhs with last dim replaced" form is wrong when rhs
    // has more batch dims (e.g. (M,K) @ (B,K,N) -> (B,M,N)).
    if (lhs.rank() >= 2 && rhs.rank() >= 2) {
        const SymbolicDim M = lhs[lhs.rank() - 2];
        const SymbolicDim N = rhs[rhs.rank() - 1];
        SymbolicShape lhs_batch(std::vector<SymbolicDim>(
            lhs.dims().begin(), lhs.dims().end() - 2));
        SymbolicShape rhs_batch(std::vector<SymbolicDim>(
            rhs.dims().begin(), rhs.dims().end() - 2));
        SymbolicShape out_shape = broadcast_symbolic_shapes(lhs_batch, rhs_batch);
        out_shape.push_back(M);
        out_shape.push_back(N);
        return {std::move(out_shape)};
    }
    return {};
}

// ============================================================================
// Conv2d
// ============================================================================

auto SymbolicShapeInference::infer_conv2d(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty() || input_shapes[0].rank() != 4) {
        return {};
    }

    auto& in_shape = input_shapes[0];  // [N, C, H, W]

    // Prefer reading out_channels / kernel_h / kernel_w from the weight tensor's
    // shape (input 1: [out_channels, in_channels/groups, kernel_h, kernel_w]).
    // The dispatch tracer never emits the out_channels/kernel_* int attrs (it
    // records kernel_size/stride/padding/dilation as {h,w} vec attrs), so the
    // legacy int-attr path is only a fallback for hand-built graphs. This
    // mirrors the concrete inference in graph.cpp's OpType::Conv2d case.
    // Track resolution: a value is only usable if it came from an explicit
    // int-attr or from a concrete weight dim. get_int_attr returns 0 for
    // missing keys, so an unresolved 0 must not be fabricated into a shape.
    bool have_out_channels = node->has_attr("out_channels");
    bool have_kernel_h = node->has_attr("kernel_h");
    bool have_kernel_w = node->has_attr("kernel_w");
    auto out_channels = node->get_int_attr("out_channels");
    auto kernel_h = node->get_int_attr("kernel_h");
    auto kernel_w = node->get_int_attr("kernel_w");
    if (input_shapes.size() >= 2 && input_shapes[1].rank() == 4) {
        const auto& w_shape = input_shapes[1];
        if (w_shape[0].is_concrete()) { out_channels = w_shape[0].value(); have_out_channels = true; }
        if (w_shape[2].is_concrete()) { kernel_h = w_shape[2].value(); have_kernel_h = true; }
        if (w_shape[3].is_concrete()) { kernel_w = w_shape[3].value(); have_kernel_w = true; }
    }

    // If out_channels / kernel_h / kernel_w could not be resolved from either
    // the weight tensor or int-attrs, do not fabricate a confidently-wrong
    // output shape (kernel_term would collapse and out_channels would be 0).
    // Mirror the other branches that return {} on missing information.
    if (!have_out_channels || !have_kernel_h || !have_kernel_w) {
        return {};
    }

    // Honor both the paired *_h/*_w int attrs and the pair-as-vec form
    // ("stride"/"padding"/"dilation" = {h, w}) the dispatch interceptor emits.
    auto pair_from = [&](const char* h_key, const char* w_key,
                          const char* vec_key,
                          int64_t default_v) -> std::pair<int64_t, int64_t> {
        if (node->has_attr(h_key) && node->has_attr(w_key)) {
            return {node->get_int_attr(h_key), node->get_int_attr(w_key)};
        }
        if (node->has_attr(vec_key)) {
            auto v = node->get_vec_attr(vec_key);
            if (v.size() == 2) return {v[0], v[1]};
            if (v.size() == 1) return {v[0], v[0]};
        }
        return {default_v, default_v};
    };
    auto [stride_h, stride_w]     = pair_from("stride_h",  "stride_w",  "stride",   1);
    auto [padding_h, padding_w]   = pair_from("padding_h", "padding_w", "padding",  0);
    auto [dilation_h, dilation_w] = pair_from("dilation_h", "dilation_w", "dilation", 1);

    // Batch dim (N) propagates symbolically
    auto N_dim = in_shape[0];
    auto H_dim = in_shape[2];
    auto W_dim = in_shape[3];

    // H_out = (H + 2*padding - dilation*(kernel-1) - 1) / stride + 1
    auto padding_h_2 = SymbolicDim::concrete(2 * padding_h);
    auto kernel_term_h = SymbolicDim::concrete(dilation_h * (kernel_h - 1) + 1);
    auto stride_h_dim = SymbolicDim::concrete(stride_h);
    auto H_out = (H_dim + padding_h_2 - kernel_term_h) / stride_h_dim + SymbolicDim::concrete(1);

    auto padding_w_2 = SymbolicDim::concrete(2 * padding_w);
    auto kernel_term_w = SymbolicDim::concrete(dilation_w * (kernel_w - 1) + 1);
    auto stride_w_dim = SymbolicDim::concrete(stride_w);
    auto W_out = (W_dim + padding_w_2 - kernel_term_w) / stride_w_dim + SymbolicDim::concrete(1);

    return {SymbolicShape({
        N_dim,
        SymbolicDim::concrete(out_channels),
        H_out,
        W_out
    })};
}

// ============================================================================
// Reshape
// ============================================================================

auto SymbolicShapeInference::infer_reshape(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) {
        return {};
    }

    if (!node->has_attr("shape")) {
        return {};
    }

    auto target_shape = node->get_vec_attr("shape");
    auto& in_shape = input_shapes[0];

    // Find the index of -1 (wildcard dimension), if any
    int64_t wildcard_idx = -1;
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] == -1) {
            wildcard_idx = static_cast<int64_t>(i);
            break;
        }
    }

    if (wildcard_idx < 0) {
        // No explicit -1 wildcard. If the input is fully static, the concrete
        // target IS the output shape.
        bool input_static = true;
        for (size_t i = 0; i < in_shape.rank(); ++i) {
            if (in_shape[i].is_symbolic()) { input_static = false; break; }
        }
        if (input_static) {
            std::vector<SymbolicDim> out_dims;
            out_dims.reserve(target_shape.size());
            for (size_t i = 0; i < target_shape.size(); ++i) {
                out_dims.emplace_back(target_shape[i]);
            }
            return {SymbolicShape(std::move(out_dims))};
        }

        // Dynamic input, no wildcard. Treating the LEADING dim as the free
        // data-dependent dim (the universal (B, ...) reshape) is only valid when
        // there is positive evidence of a batch-preserving reshape: the input's
        // non-leading dims are all static and their product equals the product of
        // the target's non-leading dims, so only the leading/batch dim varies.
        // Otherwise this is a genuinely fixed reshape and its concrete target IS
        // the output shape — overwriting target[0] with total_in/trailing would
        // wrongly symbolize a fixed leading dim (JIT-F022).
        bool trailing_static = true;
        int64_t in_trailing = 1;
        for (size_t i = 1; i < in_shape.rank(); ++i) {
            if (in_shape[i].is_symbolic()) { trailing_static = false; break; }
            in_trailing *= in_shape[i].value();
        }
        int64_t tgt_trailing = 1;
        for (size_t i = 1; i < target_shape.size(); ++i) {
            tgt_trailing *= target_shape[i];
        }
        bool batch_reshape = trailing_static && !target_shape.empty() &&
                             in_trailing == tgt_trailing;
        if (!batch_reshape) {
            std::vector<SymbolicDim> out_dims;
            out_dims.reserve(target_shape.size());
            for (size_t i = 0; i < target_shape.size(); ++i) {
                out_dims.emplace_back(target_shape[i]);
            }
            return {SymbolicShape(std::move(out_dims))};
        }

        // Batch-preserving reshape: derive the leading dim symbolically from numel
        // so the result is numel-preserving for every binding:
        // out[0] = total_in / prod(target[1..]); trailing dims stay concrete.
        SymbolicDim total_in = SymbolicDim::concrete(1);
        for (size_t i = 0; i < in_shape.rank(); ++i) total_in = total_in * in_shape[i];

        SymbolicDim trailing = SymbolicDim::concrete(1);
        for (size_t i = 1; i < target_shape.size(); ++i) {
            trailing = trailing * SymbolicDim::concrete(target_shape[i]);
        }

        std::vector<SymbolicDim> out_dims;
        out_dims.reserve(target_shape.size());
        if (trailing.is_concrete() && trailing.value() == 0) {
            out_dims.push_back(SymbolicDim::concrete(0));
        } else {
            out_dims.push_back(total_in / trailing);
        }
        for (size_t i = 1; i < target_shape.size(); ++i) {
            out_dims.emplace_back(target_shape[i]);
        }
        return {SymbolicShape(std::move(out_dims))};
    }

    // Has a -1 wildcard: compute the wildcard dim from total elements.
    // If all input dims are concrete, we can compute the wildcard concretely.
    // If any input dim is symbolic, the wildcard becomes a symbolic expression.
    SymbolicDim total_in = SymbolicDim::concrete(1);
    for (size_t i = 0; i < in_shape.rank(); ++i) {
        total_in = total_in * in_shape[i];
    }

    SymbolicDim known_product = SymbolicDim::concrete(1);
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (target_shape[i] != -1) {
            known_product = known_product * SymbolicDim::concrete(target_shape[i]);
        }
    }

    // Guard against a zero known-product (a target dim of 0 alongside a -1
    // wildcard): dividing total_in by it is undefined. Leave the wildcard at 0
    // rather than dividing by zero.
    SymbolicDim wildcard_dim = SymbolicDim::concrete(0);
    if (!(known_product.is_concrete() && known_product.value() == 0)) {
        wildcard_dim = total_in / known_product;
    }

    std::vector<SymbolicDim> out_dims;
    out_dims.reserve(target_shape.size());
    for (size_t i = 0; i < target_shape.size(); ++i) {
        if (i == static_cast<size_t>(wildcard_idx)) {
            out_dims.push_back(wildcard_dim);
        } else {
            out_dims.emplace_back(target_shape[i]);
        }
    }
    return {SymbolicShape(std::move(out_dims))};
}

// ============================================================================
// Reduction (Sum, Mean, Max, Min)
// ============================================================================

auto SymbolicShapeInference::infer_reduction(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) {
        return {};
    }

    auto sym_shape = input_shapes[0];

    auto keepdim = node->get_bool_attr("keepdim");
    auto rank = static_cast<int64_t>(sym_shape.rank());

    if (node->has_int_attr("dim")) {
        // Single-axis reduction recorded via the scalar "dim" attr.
        auto dim = node->get_int_attr("dim");
        if (dim < 0) {
            dim += rank;
        }

        if (dim >= 0 && dim < rank) {
            if (keepdim) {
                sym_shape[static_cast<size_t>(dim)] = SymbolicDim::concrete(1);
            } else {
                sym_shape.erase(static_cast<size_t>(dim));
            }
        }
    } else if (node->has_vec_attr("dims")) {
        // Multi-axis reduction: the tracing interceptor emits the axis list as
        // a vec attr "dims" (copy_int_list_to_vec(AttrKey::Dims,"dims")) with no
        // scalar "dim". Previously this fell into the else branch below and was
        // mis-inferred as a full reduce-to-scalar. Normalize each axis and
        // either set it to 1 (keepdim) or erase it. Erase from highest index to
        // lowest so earlier erasures don't shift the indices still to remove.
        auto dims = node->get_vec_attr("dims");
        std::vector<int64_t> norm;
        norm.reserve(dims.size());
        for (auto d : dims) {
            if (d < 0) d += rank;
            if (d >= 0 && d < rank) norm.push_back(d);
        }
        std::sort(norm.begin(), norm.end());
        norm.erase(std::unique(norm.begin(), norm.end()), norm.end());

        if (keepdim) {
            for (auto d : norm) {
                sym_shape[static_cast<size_t>(d)] = SymbolicDim::concrete(1);
            }
        } else {
            for (auto it = norm.rbegin(); it != norm.rend(); ++it) {
                sym_shape.erase(static_cast<size_t>(*it));
            }
        }
    } else {
        // No "dim" and no "dims": full reduction over all axes -> scalar output.
        sym_shape = SymbolicShape();
    }

    // A dimensioned Max/Min produces TWO outputs (values and indices) which share
    // the reduced shape. Emit one symbolic shape per node output so the indices
    // output is not left unshaped (and later frozen to the trace-time size).
    size_t n_out = node->outputs().empty() ? 1 : node->outputs().size();
    std::vector<SymbolicShape> result;
    result.reserve(n_out);
    for (size_t i = 0; i + 1 < n_out; ++i) {
        result.push_back(sym_shape);
    }
    result.push_back(std::move(sym_shape));
    return result;
}

// ============================================================================
// Transpose
// ============================================================================

auto SymbolicShapeInference::infer_transpose(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) {
        return {};
    }

    auto sym_shape = input_shapes[0];
    auto rank = static_cast<int64_t>(sym_shape.rank());

    int64_t dim0;
    int64_t dim1;
    if (node->has_int_attr("dim0") && node->has_int_attr("dim1")) {
        dim0 = node->get_int_attr("dim0");
        dim1 = node->get_int_attr("dim1");
        if (dim0 < 0) dim0 += rank;
        if (dim1 < 0) dim1 += rank;
    } else {
        // The tracing interceptor now copies AttrKey::Dim0/Dim1 -> "dim0"/"dim1",
        // and the autograd shape-op recorder emits them too, so a freshly traced
        // Transpose carries the axes. This fallback only fires for legacy graphs
        // that predate that fix: without the keys, get_int_attr would silently
        // return 0 for both, making the swap a no-op (the symbolic shape would
        // equal the input — transpose not reflected) and corrupting downstream
        // matmul/linear shape checks. Fall back to swapping the last two dims,
        // matching the concrete graph executor and MLIR lowering.
        if (rank < 2) {
            return {std::move(sym_shape)};
        }
        dim0 = rank - 2;
        dim1 = rank - 1;
    }

    if (dim0 >= 0 && dim0 < rank && dim1 >= 0 && dim1 < rank) {
        std::swap(sym_shape[static_cast<size_t>(dim0)],
                  sym_shape[static_cast<size_t>(dim1)]);
    }

    return {std::move(sym_shape)};
}


auto SymbolicShapeInference::infer_permute(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty() || !node->has_attr("dims")) {
        return {};
    }

    auto dims = node->get_vec_attr("dims");
    auto& in_shape = input_shapes[0];
    int64_t rank = static_cast<int64_t>(in_shape.rank());
    // A permutation must be a bijection over ALL axes, so its length must equal
    // the input rank. A wrong-length permutation (e.g. dims={0,2} on a rank-3
    // input) would otherwise yield a wrong-rank symbolic shape that mismatches
    // the concrete executor output and corrupts downstream reasoning.
    if (static_cast<int64_t>(dims.size()) != rank) {
        return {};
    }
    std::vector<SymbolicDim> out_dims(dims.size());

    // Mirror Graph::infer_symbolic_types (graph.cpp) and the concrete infer_types
    // path: normalize negative dims and require every index in range. A bare
    // `< rank` signed check would let negatives index out of bounds and leave
    // out-of-range axes as default-constructed dims.
    for (size_t i = 0; i < dims.size(); ++i) {
        int64_t d = dims[i];
        if (d < 0) d += rank;
        if (d >= 0 && d < rank) {
            out_dims[i] = in_shape[static_cast<size_t>(d)];
        } else {
            return {};  // invalid permutation index — no confident inference
        }
    }

    return {SymbolicShape(std::move(out_dims))};
}

// ============================================================================
// Linear: (*, in_features) -> (*, out_features)
// ============================================================================

auto SymbolicShapeInference::infer_linear(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.size() < 2) {
        return {};
    }

    auto out_shape = input_shapes[0];

    // A Linear activation must be rank >= 1 (*, in_features). Guard out_shape's
    // rank before computing rank()-1 (size_t underflow to SIZE_MAX would index
    // out of bounds via the unchecked operator[]). Weight shape is
    // [out_features, in_features]; replace the trailing dim with out_features.
    if (out_shape.rank() >= 1 && input_shapes[1].rank() > 0) {
        out_shape[out_shape.rank() - 1] = input_shapes[1][0];
    }

    return {std::move(out_shape)};
}

} // namespace jit
} // namespace tenzor
