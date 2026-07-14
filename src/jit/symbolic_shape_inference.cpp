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

namespace {

// JIT-R010: local re-implementations of graph.cpp's anonymous-namespace
// attr_vec_or_scalar()/pool_output_dim() helpers (that file's helpers are not
// exported/shared across translation units, and infer_conv2d above already
// established the pattern of this file carrying its own self-contained
// symbolic formula rather than depending on graph.cpp).

auto attr_vec_or_scalar(const Node& node, const char* name, size_t rank,
                        int64_t default_value) -> std::vector<int64_t> {
    auto values = node.get_vec_attr(name);
    if (!values.empty()) {
        if (values.size() == 1 && rank > 1) {
            values.resize(rank, values[0]);
        }
        return values;
    }
    if (node.has_int_attr(name)) {
        return std::vector<int64_t>(rank, node.get_int_attr(name));
    }
    return std::vector<int64_t>(rank, default_value);
}

// Mirrors graph.cpp's pool_output_dim exactly (same concrete formula with
// ceil_mode boundary correction; falls back to plain floor-division for a
// genuinely dynamic input dim, matching that file's documented limitation).
auto pool_output_dim(const SymbolicDim& in, int64_t k, int64_t s, int64_t p, int64_t d,
                     bool ceil_mode) -> SymbolicDim {
    if (in.is_concrete()) {
        const int64_t in_v = in.value();
        const int64_t num = in_v + 2 * p - d * (k - 1) - 1;
        int64_t o = ceil_mode ? (num + s - 1) / s + 1 : num / s + 1;
        if (ceil_mode && (o - 1) * s >= in_v + p) --o;
        return SymbolicDim::concrete(o);
    }
    SymbolicDim pad2 = SymbolicDim::concrete(2 * p);
    SymbolicDim kernel_term = SymbolicDim::concrete(d * (k - 1) + 1);
    SymbolicDim stride_dim = SymbolicDim::concrete(s);
    return (in + pad2 - kernel_term) / stride_dim + SymbolicDim::concrete(1);
}

}  // namespace

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
        // Comparison / logical-binary ops broadcast their two inputs exactly like
        // arithmetic elementwise ops (the result is Bool but the SHAPE follows the
        // same broadcast rule). Without these they fell to `default -> {}` and
        // froze downstream dynamic dims to the trace-time size (JIT-F040).
        case OpType::Eq:
        case OpType::Ne:
        case OpType::Lt:
        case OpType::Le:
        case OpType::Gt:
        case OpType::Ge:
        case OpType::LogicalAnd:
        case OpType::LogicalOr:
        // JIT-R010: Fmod broadcasts like every other binary elementwise op
        // (mirrors graph.cpp's infer_symbolic_types Fmod case).
        case OpType::Fmod:
        // findings.txt JIT-R118: Minimum broadcasts identically (mirrors
        // graph.cpp's infer_symbolic_types Minimum case, JIT-R064).
        case OpType::Minimum:
        // JIT-R147: Add+ReLU epilogue broadcasts like Add.
        case OpType::FusedAddReLU:
            return infer_elementwise(node);

        // Unary elementwise / shape-preserving: preserve input[0] shape.
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
        case OpType::Reciprocal:
        case OpType::FlashAttention:
        case OpType::SliceScatter:
        case OpType::Dropout:
        case OpType::GELU:
        case OpType::Softmax:
        case OpType::LogSoftmax:
        case OpType::BatchNorm2d:
        // The following are all shape-preserving but were previously omitted, so
        // a graph containing any of them (e.g. a single Cast for mixed precision,
        // or a GroupNorm/RMSNorm/activation) froze the marked dynamic dim to the
        // trace-time size in SymbolicTracePass (JIT-F040).
        case OpType::Cast:
        case OpType::ToDevice:
        case OpType::LeakyReLU:
        case OpType::ELU:
        case OpType::Mish:
        case OpType::Softplus:
        case OpType::GroupNorm:
        case OpType::InstanceNorm:
        case OpType::RMSNorm:
        case OpType::Flip:
        case OpType::LogicalNot:
        // JIT-R010: shape-preserving ops previously missing from this
        // switch (mirrors graph.cpp's infer_symbolic_types, which groups
        // all of these as simple "output shape == input[0]" cases).
        case OpType::Round:
        case OpType::SiLU:
        case OpType::RoPE:
        case OpType::Roll:
        case OpType::Scatter:
        case OpType::Dequantize:
        case OpType::Quantize:
        case OpType::DenseToSparse:
        case OpType::ResidualAdd:
        // findings.txt JIT-R118: all of the following are shape-preserving
        // (mirrors graph.cpp's infer_symbolic_types, which groups each of
        // these as a plain "output shape == input[0]" case) but were
        // missing here, silently freezing any dynamic dim downstream of one
        // of these ops to its trace-time size (the same JIT-F040 class the
        // rest of this group was already fixed for).
        case OpType::Swish:          // JIT-R084: x*sigmoid(x)
        case OpType::RReLU:          // JIT-R084
        case OpType::LogSigmoid:     // JIT-R084
        case OpType::IndexCopy:      // JIT-R064: writes rows into self at index
        case OpType::ScatterAdd:     // accumulates into self at index
        case OpType::GQA:            // Grouped-Query Attention: Q's shape
        case OpType::ShapeGuard:
        case OpType::GuardNode:
        case OpType::SwapOut:
        case OpType::SwapIn:
        case OpType::LayoutConvert:
        case OpType::LayerNorm:
        // JIT-R147: unary; shape-preserving.
        case OpType::Sign:
        // JIT-R147: cumulative reduction along dim; shape-preserving (dim
        // size is unchanged by a cumulative sum).
        case OpType::CumSum: {
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
        // JIT-R010: QuantizedConv2d shares Conv2d's shape rule exactly
        // (mirrors graph.cpp's infer_symbolic_types, JIT-026).
        case OpType::QuantizedConv2d:
        // JIT-R147: FusedConv2dReLU/FusedConv2dBnReLU share Conv2d's exact
        // [x, weight, ...] shape rule (same generic stride/padding attrs,
        // same inputs[0]/inputs[1] = x/weight layout).
        case OpType::FusedConv2dReLU:
        case OpType::FusedConv2dBnReLU:
            return infer_conv2d(node);

        // JIT-R010: previously missing entirely -- fell to `default -> {}`
        // and froze downstream dynamic dims (the same JIT-F040 pattern this
        // file's other additions document).
        case OpType::ConvTranspose:
            return infer_conv_transpose(node);

        case OpType::MaxPool2d:
        case OpType::AvgPool2d:
            return infer_pool2d(node);

        case OpType::AdaptiveAvgPool2d:
            return infer_adaptive_avg_pool2d(node);

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
        // JIT-R010: QuantizedLinear/QuantizedLinearStatic share Linear's
        // [x, weight] -> (*, out_features) shape rule exactly (mirrors
        // graph.cpp's infer_symbolic_types, JIT-026).
        case OpType::QuantizedLinear:
        case OpType::QuantizedLinearStatic:
        // findings.txt JIT-R118: SparseMatMul shares the identical
        // [x, weight(, bias)] -> (*, out_features) rule (mirrors graph.cpp's
        // infer_symbolic_types, which groups it with QuantizedLinearStatic).
        case OpType::SparseMatMul:
        // JIT-R147: Linear+ReLU epilogue shares Linear's exact [x, weight]
        // shape rule.
        case OpType::FusedLinearReLU:
            return infer_linear(node);

        // Rank-changing / select ops that previously fell to `default -> {}` and
        // froze downstream dynamic dims (JIT-F040). A CNN classifier's
        // conv->Flatten->Linear is the canonical case.
        case OpType::Where:
            return infer_where(node);
        case OpType::Squeeze:
            return infer_squeeze(node);
        case OpType::Unsqueeze:
            return infer_unsqueeze(node);
        case OpType::Flatten:
            return infer_flatten(node);

        case OpType::AsStrided:
            // Output shape is the explicit target `size`, always fully
            // concrete (as_strided's API takes concrete int64_t sizes) —
            // independent of the input's shape.
            if (node->has_attr("size")) {
                return {SymbolicShape::from_concrete(node->get_vec_attr("size"))};
            }
            return {};

        case OpType::Triu:
        case OpType::Tril: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                return {input_shapes[0]};
            }
            return {};
        }

        case OpType::Trace:
            // 2D -> scalar (0-D), known regardless of symbolic-ness of dims.
            return {SymbolicShape(std::vector<SymbolicDim>{})};

        case OpType::Diag: {
            // Only handle the fully-concrete case (matches graph.cpp's
            // infer_symbolic_types formulas); a genuinely symbolic leading
            // dim is left unresolved rather than guessed at.
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].is_fully_concrete()) {
                int64_t d = node->get_int_attr("diagonal");
                auto in_shape = input_shapes[0].to_concrete();
                if (in_shape.size() == 1) {
                    int64_t n = in_shape[0];
                    int64_t size = n + std::abs(d);
                    return {SymbolicShape::from_concrete({size, size})};
                }
                if (in_shape.size() == 2) {
                    int64_t rows = in_shape[0];
                    int64_t cols = in_shape[1];
                    int64_t start_row = d >= 0 ? 0 : -d;
                    int64_t start_col = d >= 0 ? d : 0;
                    int64_t diag_len = std::min(rows - start_row, cols - start_col);
                    return {SymbolicShape::from_concrete(
                        {std::max<int64_t>(diag_len, 0)})};
                }
            }
            return {};
        }

        case OpType::ViewAsReal: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                std::vector<SymbolicDim> dims;
                dims.reserve(input_shapes[0].rank() + 1);
                for (size_t i = 0; i < input_shapes[0].rank(); ++i) {
                    dims.push_back(input_shapes[0][i]);
                }
                dims.push_back(SymbolicDim::concrete(2));
                return {SymbolicShape(std::move(dims))};
            }
            return {};
        }

        case OpType::ViewAsComplex: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() > 0) {
                std::vector<SymbolicDim> dims;
                dims.reserve(input_shapes[0].rank() - 1);
                for (size_t i = 0; i + 1 < input_shapes[0].rank(); ++i) {
                    dims.push_back(input_shapes[0][i]);
                }
                return {SymbolicShape(std::move(dims))};
            }
            return {};
        }

        // ====================================================================
        // JIT-R010: remaining ops previously missing from this switch,
        // mirroring graph.cpp's infer_symbolic_types formulas exactly.
        // ====================================================================

        case OpType::Embedding: {
            // weight [num_embeddings, embedding_dim], indices [*] ->
            // indices_shape + [embedding_dim].
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2) {
                auto out_shape = input_shapes[1];
                if (input_shapes[0].rank() >= 2) {
                    out_shape.push_back(input_shapes[0][1]);
                }
                return {std::move(out_shape)};
            }
            return {};
        }

        case OpType::EmbeddingBag: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2 && input_shapes[1].rank() >= 1 &&
                node->has_int_attr("embedding_dim")) {
                bool include_last = node->has_bool_attr("include_last_offset") &&
                                    node->get_bool_attr("include_last_offset");
                SymbolicDim num_bags = input_shapes[1][0];
                if (include_last && num_bags.is_concrete()) {
                    num_bags = SymbolicDim::concrete(
                        std::max<int64_t>(num_bags.value() - 1, 0));
                }
                return {SymbolicShape({
                    num_bags, SymbolicDim::concrete(node->get_int_attr("embedding_dim"))})};
            }
            return {};
        }

        case OpType::Eigvalsh: {
            // (..., N, N) -> (..., N)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto sym_shape = input_shapes[0];
                sym_shape.erase(sym_shape.rank() - 1);
                return {std::move(sym_shape)};
            }
            return {};
        }

        case OpType::Norm:
            // Scalar output.
            return {SymbolicShape()};

        case OpType::SolveTriangular:
        case OpType::LinalgLUSolve:
        case OpType::LinalgLDLSolve: {
            // A/LU_data/LD: (..., N, N), B: (..., N, K) -> (..., N, K)
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2) {
                return {input_shapes.back()};
            }
            return {};
        }

        case OpType::LinalgHouseholder: {
            // input: (..., N, K), tau: (..., K) -> Q: (..., N, K)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                return {input_shapes[0]};
            }
            return {};
        }

        case OpType::Ormqr: {
            // input: (..., N, K), tau: (..., K), other: (..., M, P) ->
            // result: (..., M, P)
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 3) {
                return {input_shapes[2]};
            }
            return {};
        }

        case OpType::LinalgEig: {
            // (..., N, N) -> Wr: (..., N), Wi: (..., N), V: (..., N, N)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto N_dim = s[s.rank() - 1];
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto w_dims = batch_dims;
                w_dims.push_back(N_dim);
                return {SymbolicShape(w_dims), SymbolicShape(std::move(w_dims)),
                        input_shapes[0]};
            }
            return {};
        }

        case OpType::LinalgLU: {
            // (..., N, N) -> L: (..., N, N), U: (..., N, N), pivots: (..., N)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto N_dim = s[s.rank() - 1];
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto piv_dims = batch_dims;
                piv_dims.push_back(N_dim);
                return {input_shapes[0], input_shapes[0],
                        SymbolicShape(std::move(piv_dims))};
            }
            return {};
        }

        case OpType::LinalgLDLFactor: {
            // (..., N, N) -> LD: (..., N, N), pivots: (..., N)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto N_dim = s[s.rank() - 1];
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto piv_dims = batch_dims;
                piv_dims.push_back(N_dim);
                return {input_shapes[0], SymbolicShape(std::move(piv_dims))};
            }
            return {};
        }

        case OpType::Geqrf: {
            // (..., M, N) -> A_factored: (..., M, N), tau: (..., min(M,N))
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto M = s[s.rank() - 2];
                auto N_dim = s[s.rank() - 1];
                auto min_dim = [](const SymbolicDim& a, const SymbolicDim& b) -> SymbolicDim {
                    if (a.is_concrete() && b.is_concrete())
                        return SymbolicDim(std::min(a.value(), b.value()));
                    return a.is_concrete() ? a : b;
                };
                SymbolicDim K = min_dim(M, N_dim);
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto tau_dims = batch_dims;
                tau_dims.push_back(K);
                return {input_shapes[0], SymbolicShape(std::move(tau_dims))};
            }
            return {};
        }

        case OpType::Slogdet: {
            // (..., N, N) -> sign: (...), logabsdet: (...)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto sym_shape = input_shapes[0];
                sym_shape.erase(sym_shape.rank() - 1);
                sym_shape.erase(sym_shape.rank() - 1);
                return {sym_shape, sym_shape};
            }
            return {};
        }

        case OpType::If: {
            std::vector<SymbolicShape> out;
            if (node->then_branch()) {
                for (const auto& o : node->then_branch()->outputs()) {
                    out.push_back(o->symbolic_shape());
                }
            }
            return out;
        }

        case OpType::Loop: {
            // Loop-carried values preserve shapes.
            auto input_shapes = gather_input_shapes(node);
            std::vector<SymbolicShape> out;
            for (size_t i = 2; i < input_shapes.size(); ++i) {
                out.push_back(input_shapes[i]);
            }
            return out;
        }

        case OpType::FusedFFN: {
            // (*, in) -> (*, out) via the second linear's weight (input[2]).
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 3) {
                auto out_shape = input_shapes[0];
                if (out_shape.rank() >= 1 && input_shapes[2].rank() > 0) {
                    out_shape[out_shape.rank() - 1] = input_shapes[2][0];
                }
                return {std::move(out_shape)};
            }
            if (!input_shapes.empty()) {
                return {input_shapes[0]};
            }
            return {};
        }

        // ====================================================================
        // findings.txt JIT-R118: the following cases were entirely absent
        // from this switch (silently freezing any dynamic dim downstream of
        // one of these ops, per the JIT-F040 class documented throughout
        // this file) despite graph.cpp's infer_symbolic_types() -- the
        // OTHER, non-production symbolic-shape implementation -- already
        // handling every one of them. Ported directly from there, adapted
        // to this function's gather_input_shapes()+return convention.
        // ====================================================================

        case OpType::Gather: {
            // torch.gather / take_along_dim: output shape == indices shape.
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2) {
                return {input_shapes[1]};
            }
            return {};
        }

        case OpType::OneHot: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto sym_shape = input_shapes[0];
                sym_shape.push_back(SymbolicDim::concrete(node->get_int_attr("num_classes")));
                return {std::move(sym_shape)};
            }
            return {};
        }

        case OpType::Slice: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto sym_shape = input_shapes[0];
                int64_t rank = static_cast<int64_t>(sym_shape.rank());
                int64_t dim = node->get_int_attr("dim");
                int64_t start = node->get_int_attr("start");
                int64_t end = node->get_int_attr("end");
                int64_t step = node->has_attr("step") ? node->get_int_attr("step") : 1;
                if (dim < 0) dim += rank;
                if (dim >= 0 && dim < rank) {
                    if (step <= 0) step = 1;
                    const SymbolicDim& cur = sym_shape[static_cast<size_t>(dim)];
                    if (cur.is_concrete()) {
                        const int64_t dim_size = cur.value();
                        if (start < 0) start += dim_size;
                        if (end   < 0) end   += dim_size;
                        if (start < 0) start = 0;
                        if (start > dim_size) start = dim_size;
                        if (end   < start) end = start;
                        if (end   > dim_size) end = dim_size;
                        int64_t len = (end - start + step - 1) / step;
                        sym_shape[static_cast<size_t>(dim)] = SymbolicDim::concrete(len);
                    } else if (start >= 0 && end >= start) {
                        int64_t len = (end - start + step - 1) / step;
                        sym_shape[static_cast<size_t>(dim)] = SymbolicDim::concrete(len);
                    }
                }
                return {std::move(sym_shape)};
            }
            return {};
        }

        case OpType::Cat: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                auto out_shape = input_shapes[0];
                int64_t rank = static_cast<int64_t>(out_shape.rank());
                if (dim < 0) dim += rank;
                if (dim >= 0 && dim < rank) {
                    SymbolicDim total = SymbolicDim::concrete(0);
                    for (auto& s : input_shapes) {
                        if (dim < static_cast<int64_t>(s.rank())) {
                            total = total + s[static_cast<size_t>(dim)];
                        }
                    }
                    out_shape[static_cast<size_t>(dim)] = total;
                }
                return {std::move(out_shape)};
            }
            return {};
        }

        case OpType::Stack: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto sym_shape = input_shapes[0];
                int64_t rank = static_cast<int64_t>(sym_shape.rank());
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                const int64_t limit = rank + 1;
                if (dim < 0) dim += limit;
                if (dim < 0) dim = 0;
                if (dim > rank) dim = rank;
                sym_shape.insert(static_cast<size_t>(dim),
                                  SymbolicDim::concrete(static_cast<int64_t>(input_shapes.size())));
                return {std::move(sym_shape)};
            }
            return {};
        }

        case OpType::Broadcast: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto target = node->get_vec_attr("shape");
                if (target.empty()) return {input_shapes[0]};
                return {SymbolicShape::from_concrete(target)};
            }
            return {};
        }

        case OpType::IndexSelect: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2) {
                auto out_shape = input_shapes[0];
                int64_t rank = static_cast<int64_t>(out_shape.rank());
                int64_t dim = node->has_attr("dim") ? node->get_int_attr("dim") : 0;
                if (dim < 0) dim += rank;
                // JIT-R151b: match the concrete infer_types ground truth
                // (graph.cpp's IndexSelect case, which uses the throwing
                // normalize_dim_for_rank) and Graph::infer_symbolic_types'
                // mirror of it -- an out-of-range IndexSelect dim is a
                // malformed graph, and silently clamping to rank-1 here
                // used to paper over that by producing a shape for the
                // WRONG dimension instead of surfacing the bug.
                if (dim < 0 || dim >= rank) {
                    throw std::out_of_range("JIT graph dimension out of range");
                }
                SymbolicDim selected = SymbolicDim::concrete(1);
                for (size_t i = 0; i < input_shapes[1].rank(); ++i) {
                    selected = selected * input_shapes[1][i];
                }
                out_shape[static_cast<size_t>(dim)] = selected;
                return {std::move(out_shape)};
            }
            return {};
        }

        case OpType::Interpolate: {
            // JIT-R018 (see graph.cpp's identical case): only push a shape
            // when an explicit output_size is present; the scale_factor-only
            // form deliberately leaves the trace-time symbolic shape intact
            // rather than overwrite it with the unscaled input shape.
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto size = node->get_vec_attr("output_size");
                if (!size.empty()) {
                    auto out_shape = input_shapes[0];
                    if (out_shape.rank() >= size.size() + 2) {
                        const auto offset = out_shape.rank() - size.size();
                        for (size_t i = 0; i < size.size(); ++i) {
                            out_shape[offset + i] = SymbolicDim::concrete(size[i]);
                        }
                    }
                    return {std::move(out_shape)};
                }
            }
            return {};
        }

        case OpType::Padding: {
            // Padding ENLARGES dims; must not be treated as shape-preserving
            // (mirrors graph.cpp's infer_symbolic_types Padding case, JIT-044).
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto out_shape = input_shapes[0];
                const size_t rank = out_shape.rank();
                if (node->has_attr("padding")) {
                    auto v = node->get_vec_attr("padding");
                    std::vector<int64_t> low(rank, 0), high(rank, 0);
                    if (v.size() == 2 * rank) {
                        for (size_t i = 0; i < rank; ++i) {
                            low[i] = v[2 * i];
                            high[i] = v[2 * i + 1];
                        }
                    } else if (v.size() % 2 == 0) {
                        const size_t pairs = v.size() / 2;
                        for (size_t i = 0; i < pairs && i < rank; ++i) {
                            const size_t dim = rank - 1 - i;
                            low[dim] = v[2 * i];
                            high[dim] = v[2 * i + 1];
                        }
                    }
                    for (size_t i = 0; i < rank; ++i) {
                        out_shape[i] = out_shape[i] + SymbolicDim::concrete(low[i] + high[i]);
                    }
                }
                return {std::move(out_shape)};
            }
            return {};
        }

        case OpType::Det: {
            // (..., N, N) -> (...)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto sym_shape = input_shapes[0];
                sym_shape.erase(sym_shape.rank() - 1);
                sym_shape.erase(sym_shape.rank() - 1);
                return {std::move(sym_shape)};
            }
            return {};
        }

        case OpType::Inv:
        case OpType::Cholesky: {
            // (..., N, N) -> (..., N, N)
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                return {input_shapes[0]};
            }
            return {};
        }

        case OpType::Solve: {
            // A: (..., N, N), B: (..., N, K) -> (..., N, K)
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2) {
                return {input_shapes[1]};
            }
            return {};
        }

        case OpType::Svd: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto M = s[s.rank() - 2];
                auto N_dim = s[s.rank() - 1];
                auto min_dim = [](const SymbolicDim& a, const SymbolicDim& b) -> SymbolicDim {
                    if (a.is_concrete() && b.is_concrete())
                        return SymbolicDim(std::min(a.value(), b.value()));
                    return a.is_concrete() ? a : b;
                };
                SymbolicDim K = min_dim(M, N_dim);
                bool full = node->has_attr("full_matrices")
                                ? node->get_bool_attr("full_matrices") : true;
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto u_dims = batch_dims;
                u_dims.push_back(M);
                u_dims.push_back(full ? M : K);
                auto s_dims = batch_dims;
                s_dims.push_back(K);
                auto vt_dims = batch_dims;
                vt_dims.push_back(full ? N_dim : K);
                vt_dims.push_back(N_dim);
                return {SymbolicShape(std::move(u_dims)), SymbolicShape(std::move(s_dims)),
                        SymbolicShape(std::move(vt_dims))};
            }
            return {};
        }

        case OpType::Qr: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto M = s[s.rank() - 2];
                auto N_dim = s[s.rank() - 1];
                auto min_dim = [](const SymbolicDim& a, const SymbolicDim& b) -> SymbolicDim {
                    if (a.is_concrete() && b.is_concrete())
                        return SymbolicDim(std::min(a.value(), b.value()));
                    return a.is_concrete() ? a : b;
                };
                SymbolicDim K = min_dim(M, N_dim);
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto q_dims = batch_dims;
                q_dims.push_back(M);
                q_dims.push_back(K);
                auto r_dims = batch_dims;
                r_dims.push_back(K);
                r_dims.push_back(N_dim);
                return {SymbolicShape(std::move(q_dims)), SymbolicShape(std::move(r_dims))};
            }
            return {};
        }

        case OpType::Eigh: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() >= 2) {
                auto& s = input_shapes[0];
                auto N_dim = s[s.rank() - 1];
                std::vector<SymbolicDim> batch_dims;
                for (size_t d = 0; d + 2 < s.rank(); ++d) batch_dims.push_back(s[d]);
                auto w_dims = batch_dims;
                w_dims.push_back(N_dim);
                return {SymbolicShape(std::move(w_dims)), input_shapes[0]};
            }
            return {};
        }

        case OpType::SparseSpMM: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 4 && node->has_int_attr("m")) {
                std::vector<SymbolicDim> dims = {SymbolicDim::concrete(node->get_int_attr("m"))};
                auto& dense_sym = input_shapes[3];
                for (size_t i = 1; i < dense_sym.rank(); ++i) dims.push_back(dense_sym[i]);
                return {SymbolicShape(std::move(dims))};
            }
            return {};
        }

        case OpType::SparseSpMV: {
            if (node->has_int_attr("m")) {
                return {SymbolicShape({SymbolicDim::concrete(node->get_int_attr("m"))})};
            }
            return {};
        }

        case OpType::SparseAdd:
        case OpType::SparseTrsv:
        case OpType::SparseTrsm: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 4) {
                return {input_shapes[3]};
            }
            return {};
        }

        case OpType::Constant: {
            if (node->has_attr("value")) {
                auto& t = node->get_tensor_attr("value");
                return {SymbolicShape::from_concrete(
                    std::vector<int64_t>(t.shape().begin(), t.shape().end()))};
            }
            return {};
        }

        case OpType::Input:
        case OpType::Output: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                return {input_shapes[0]};
            }
            return {};
        }

        // ====================================================================
        // JIT-R147: previously-missing OpTypes -- ops with a real, data-
        // independent shape rule (mirrors graph.cpp's infer_symbolic_types()
        // identical additions).
        // ====================================================================

        // Sort: (values, indices) -- both shape-preserving (== input[0]).
        case OpType::Sort: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                return {input_shapes[0], input_shapes[0]};
            }
            return {};
        }

        // GridSample: input [N,C,Hin,Win], grid [N,Hout,Wout,2] -> [N,C,Hout,Wout].
        case OpType::GridSample: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2 &&
                input_shapes[0].rank() == 4 && input_shapes[1].rank() == 4) {
                auto& in = input_shapes[0];
                auto& grid = input_shapes[1];
                return {SymbolicShape({in[0], in[1], grid[1], grid[2]})};
            }
            return {};
        }

        // AffineGrid: theta [N,2,3], "output_size" attr = {N,C,H,W} -> grid [N,H,W,2].
        case OpType::AffineGrid: {
            if (node->has_vec_attr("output_size")) {
                auto sz = node->get_vec_attr("output_size");
                if (sz.size() >= 4) {
                    return {SymbolicShape({
                        SymbolicDim::concrete(sz[0]), SymbolicDim::concrete(sz[2]),
                        SymbolicDim::concrete(sz[3]), SymbolicDim::concrete(2)})};
                }
            }
            return {};
        }

        // BoxIoU: [boxes1, boxes2] -> (boxes1_count, boxes2_count).
        case OpType::BoxIoU: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2 &&
                input_shapes[0].rank() >= 1 && input_shapes[1].rank() >= 1) {
                return {SymbolicShape({input_shapes[0][0], input_shapes[1][0]})};
            }
            return {};
        }

        // ROIAlignForward: [features, rois], "output_size" attr = [h, w]
        // -> (num_rois, C, h, w).
        case OpType::ROIAlignForward: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2 &&
                input_shapes[0].rank() >= 2 && input_shapes[1].rank() >= 1 &&
                node->has_vec_attr("output_size")) {
                auto sz = node->get_vec_attr("output_size");
                if (sz.size() >= 2) {
                    return {SymbolicShape({
                        input_shapes[1][0], input_shapes[0][1],
                        SymbolicDim::concrete(sz[0]), SymbolicDim::concrete(sz[1])})};
                }
            }
            return {};
        }

        // FusedSoftmaxCrossEntropy: [logits, labels]; "reduction" attr
        // (0=mean,1=sum,2=none) determines scalar vs per-sample shape.
        case OpType::FusedSoftmaxCrossEntropy: {
            auto input_shapes = gather_input_shapes(node);
            if (input_shapes.size() >= 2) {
                int64_t red = node->has_int_attr("reduction")
                                  ? node->get_int_attr("reduction") : 0;
                if (red == 2) {
                    return {input_shapes[1]};
                }
                return {SymbolicShape({})};
            }
            return {};
        }

        // FFT/IFFT: complex-to-complex, shape-preserving unless the "n" attr
        // overrides the transformed axis's size.
        case OpType::FFT:
        case OpType::IFFT: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty()) {
                auto out_shape = input_shapes[0];
                if (node->has_int_attr("n") && out_shape.rank() > 0) {
                    int64_t dim = node->has_int_attr("dim") ? node->get_int_attr("dim") : -1;
                    int64_t rank_i = static_cast<int64_t>(out_shape.rank());
                    size_t axis = static_cast<size_t>(dim < 0 ? rank_i + dim : dim);
                    if (axis < out_shape.rank()) {
                        out_shape[axis] = SymbolicDim::concrete(node->get_int_attr("n"));
                    }
                }
                return {std::move(out_shape)};
            }
            return {};
        }

        // RFFT: real -> complex; transformed axis size becomes n_used/2+1
        // where n_used = "n" attr if given else the input's size along dim.
        case OpType::RFFT: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() > 0) {
                auto out_shape = input_shapes[0];
                int64_t dim = node->has_int_attr("dim") ? node->get_int_attr("dim") : -1;
                int64_t rank_i = static_cast<int64_t>(out_shape.rank());
                size_t axis = static_cast<size_t>(dim < 0 ? rank_i + dim : dim);
                if (axis < out_shape.rank()) {
                    if (node->has_int_attr("n")) {
                        out_shape[axis] = SymbolicDim::concrete(node->get_int_attr("n") / 2 + 1);
                        return {std::move(out_shape)};
                    }
                    if (out_shape[axis].is_concrete()) {
                        out_shape[axis] = SymbolicDim::concrete(out_shape[axis].value() / 2 + 1);
                        return {std::move(out_shape)};
                    }
                }
            }
            return {};
        }

        // IRFFT: complex -> real; transformed axis size = "n" attr if given
        // else 2*(input_size_along_dim - 1).
        case OpType::IRFFT: {
            auto input_shapes = gather_input_shapes(node);
            if (!input_shapes.empty() && input_shapes[0].rank() > 0) {
                auto out_shape = input_shapes[0];
                int64_t dim = node->has_int_attr("dim") ? node->get_int_attr("dim") : -1;
                int64_t rank_i = static_cast<int64_t>(out_shape.rank());
                size_t axis = static_cast<size_t>(dim < 0 ? rank_i + dim : dim);
                if (axis < out_shape.rank()) {
                    if (node->has_int_attr("n")) {
                        out_shape[axis] = SymbolicDim::concrete(node->get_int_attr("n"));
                        return {std::move(out_shape)};
                    }
                    if (out_shape[axis].is_concrete()) {
                        out_shape[axis] = SymbolicDim::concrete(2 * (out_shape[axis].value() - 1));
                        return {std::move(out_shape)};
                    }
                }
            }
            return {};
        }

        // JIT-R147: ops with NO safe static shape rule -- see graph.cpp's
        // infer_symbolic_types() for the detailed per-op rationale (each
        // either has a genuinely data-dependent output size that cannot be
        // derived from input shapes alone, or -- the Sparse* structural
        // conversions -- doesn't fit the dense single-tensor SymbolicShape
        // model at all). Falls through to the same `default: return {}`
        // behavior this function already documents for unhandled ops.
        case OpType::RepeatInterleave:
        case OpType::Nonzero:
        case OpType::Bincount:
        case OpType::NMS:
        case OpType::AnchorGenerate:
        case OpType::CTCLossForward:
        case OpType::SparseFromDense:
        case OpType::SparseToDense:
        case OpType::SparseCoalesce:
        case OpType::SparseTranspose:
        case OpType::SparseToCsr:
        case OpType::SparseToCsc:
        case OpType::SparseToBsr:
            return {};

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
// Where / Squeeze / Unsqueeze / Flatten (JIT-F040: previously default -> {},
// which froze downstream dynamic dims). Mirrors the concrete symbolic logic in
// Graph's infer_types so SymbolicTracePass propagates dynamic dims through them.
// ============================================================================

auto SymbolicShapeInference::infer_where(const Node* node)
    -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) return {};
    SymbolicShape out = input_shapes[0];
    for (size_t i = 1; i < input_shapes.size(); ++i) {
        out = broadcast_symbolic_shapes(out, input_shapes[i]);
    }
    return {out};
}

auto SymbolicShapeInference::infer_squeeze(const Node* node)
    -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) return {};
    auto sym_shape = input_shapes[0];
    const int64_t rank = static_cast<int64_t>(sym_shape.rank());
    // Only a CONCRETE size-1 dim can be squeezed; an unknown symbolic dim is
    // conservatively kept (matches Graph::infer_types).
    auto is_one = [](const SymbolicDim& d) {
        return d.is_concrete() && d.value() == 1;
    };
    if (node->has_attr("dims")) {
        auto dims = node->get_vec_attr("dims");
        std::vector<int64_t> norm;
        for (int64_t d : dims) { if (d < 0) d += rank; norm.push_back(d); }
        std::sort(norm.begin(), norm.end(),
                  [](int64_t a, int64_t b) { return a > b; });
        // JIT-R021: a duplicate index (e.g. {0,0}) would otherwise erase two
        // elements on the second pass through an already-shrunk vector,
        // producing a wrong-rank symbolic shape. std::unique needs a
        // pre-sorted range (already true above) to remove adjacent dups.
        norm.erase(std::unique(norm.begin(), norm.end()), norm.end());
        for (int64_t d : norm) {
            if (d >= 0 && d < static_cast<int64_t>(sym_shape.rank()) &&
                is_one(sym_shape[static_cast<size_t>(d)])) {
                sym_shape.erase(static_cast<size_t>(d));
            }
        }
    } else if (node->has_attr("dim")) {
        int64_t dim = node->get_int_attr("dim");
        if (dim < 0) dim += rank;
        if (dim >= 0 && dim < rank &&
            is_one(sym_shape[static_cast<size_t>(dim)])) {
            sym_shape.erase(static_cast<size_t>(dim));
        }
    } else {
        for (int64_t i = static_cast<int64_t>(sym_shape.rank()) - 1; i >= 0; --i) {
            if (is_one(sym_shape[static_cast<size_t>(i)])) {
                sym_shape.erase(static_cast<size_t>(i));
            }
        }
    }
    return {std::move(sym_shape)};
}

auto SymbolicShapeInference::infer_unsqueeze(const Node* node)
    -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) return {};
    auto sym_shape = input_shapes[0];
    int64_t dim = node->get_int_attr("dim");
    if (dim < 0) dim += static_cast<int64_t>(sym_shape.rank()) + 1;
    if (dim >= 0 && dim <= static_cast<int64_t>(sym_shape.rank())) {
        sym_shape.insert(static_cast<size_t>(dim), SymbolicDim::concrete(1));
    }
    return {std::move(sym_shape)};
}

auto SymbolicShapeInference::infer_flatten(const Node* node)
    -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty()) return {};
    auto& in_shape = input_shapes[0];
    int64_t start_dim = node->has_attr("start_dim")
                            ? node->get_int_attr("start_dim") : 0;
    int64_t end_dim = node->has_attr("end_dim")
                          ? node->get_int_attr("end_dim") : -1;
    const int64_t rank = static_cast<int64_t>(in_shape.rank());
    if (start_dim < 0) start_dim += rank;
    if (end_dim < 0) end_dim += rank;
    std::vector<SymbolicDim> out_dims;
    SymbolicDim flat = SymbolicDim::concrete(1);
    for (int64_t i = 0; i < rank; ++i) {
        if (i < start_dim || i > end_dim) {
            out_dims.push_back(in_shape[static_cast<size_t>(i)]);
        } else {
            flat = flat * in_shape[static_cast<size_t>(i)];
            if (i == end_dim) out_dims.push_back(flat);
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


// JIT-R010: transposed/deconv N-D convolution. Mirrors graph.cpp's
// conv_sym_output_shape's ConvTranspose formula: out = (in-1)*stride -
// 2*padding + dilation*(kernel-1) + output_padding + 1, out_channels =
// weight.shape[1] * groups.
auto SymbolicShapeInference::infer_conv_transpose(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.size() < 2 || input_shapes[0].rank() < 3 ||
        input_shapes[1].rank() != input_shapes[0].rank()) {
        return {};
    }
    const auto& in_shape = input_shapes[0];
    const auto& w_shape = input_shapes[1];
    const size_t spatial_rank = in_shape.rank() - 2;
    const int64_t groups = node->has_attr("groups") ? node->get_int_attr("groups") : 1;
    auto stride = attr_vec_or_scalar(*node, "stride", spatial_rank, 1);
    auto padding = attr_vec_or_scalar(*node, "padding", spatial_rank, 0);
    auto dilation = attr_vec_or_scalar(*node, "dilation", spatial_rank, 1);
    auto output_padding = attr_vec_or_scalar(*node, "output_padding", spatial_rank, 0);

    std::vector<SymbolicDim> dims;
    dims.reserve(in_shape.rank());
    dims.push_back(in_shape[0]);
    dims.push_back(w_shape[1] * SymbolicDim::concrete(groups));
    for (size_t i = 0; i < spatial_rank; ++i) {
        auto in_size = in_shape[i + 2];
        auto kernel = w_shape[i + 2];
        dims.push_back((in_size - SymbolicDim::concrete(1)) *
                       SymbolicDim::concrete(stride[i]) -
                       SymbolicDim::concrete(2 * padding[i]) +
                       SymbolicDim::concrete(dilation[i]) *
                       (kernel - SymbolicDim::concrete(1)) +
                       SymbolicDim::concrete(output_padding[i] + 1));
    }
    return {SymbolicShape(std::move(dims))};
}

// JIT-R010: MaxPool2d/AvgPool2d. Mirrors graph.cpp's rank-4 pooling case
// exactly (rectangular kernel/stride/padding/dilation via scalar-or-[h,w]
// attrs, ceil_mode).
auto SymbolicShapeInference::infer_pool2d(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty() || input_shapes[0].rank() != 4) {
        return {};
    }
    auto& in_shape = input_shapes[0];
    auto read_pair = [&](const char* name, int64_t dflt) -> std::pair<int64_t, int64_t> {
        if (node->has_vec_attr(name)) {
            auto v = node->get_vec_attr(name);
            if (v.size() >= 2) return {v[0], v[1]};
            if (v.size() == 1) return {v[0], v[0]};
        }
        if (node->has_int_attr(name)) {
            int64_t s = node->get_int_attr(name);
            return {s, s};
        }
        return {dflt, dflt};
    };
    auto [kh, kw] = read_pair("kernel_size", 1);
    bool has_stride = node->has_int_attr("stride") || node->has_vec_attr("stride");
    auto [sh, sw] = has_stride ? read_pair("stride", 1)
                               : std::pair<int64_t, int64_t>{kh, kw};
    auto [ph, pw] = read_pair("padding", 0);
    auto [dh, dw] = read_pair("dilation", 1);
    bool ceil_mode = node->has_int_attr("ceil_mode") &&
                     node->get_int_attr("ceil_mode") != 0;

    SymbolicDim H_out = pool_output_dim(in_shape[2], kh, sh, ph, dh, ceil_mode);
    SymbolicDim W_out = pool_output_dim(in_shape[3], kw, sw, pw, dw, ceil_mode);
    return {SymbolicShape({in_shape[0], in_shape[1], H_out, W_out})};
}

// JIT-R010: AdaptiveAvgPool2d -- output spatial dims come straight from the
// "output_size" attr (always concrete: the eager API takes a concrete
// target size), batch/channel dims propagate symbolically.
auto SymbolicShapeInference::infer_adaptive_avg_pool2d(const Node* node) -> std::vector<SymbolicShape> {
    auto input_shapes = gather_input_shapes(node);
    if (input_shapes.empty() || input_shapes[0].rank() != 4) {
        return {};
    }
    auto output_size = node->get_vec_attr("output_size");
    if (output_size.size() < 2) {
        return {};
    }
    return {SymbolicShape({
        input_shapes[0][0], input_shapes[0][1],
        SymbolicDim::concrete(output_size[0]),
        SymbolicDim::concrete(output_size[1])})};
}

} // namespace jit
} // namespace tenzor
