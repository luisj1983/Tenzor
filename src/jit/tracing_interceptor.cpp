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
        // nn::functional::layer_norm dispatches OpId::FusedLayerNorm (x, gamma,
        // beta + NormalizedShape/Eps). Without this mapping every traced
        // LayerNorm was an unmappable op → graph break → the whole graph was
        // discarded and the call silently fell back to eager (no JIT, and no
        // training-through-JIT). Map it to the same OpType::LayerNorm node the
        // executor already knows how to run/differentiate.
        case OpId::FusedLayerNorm: return OpType::LayerNorm;

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
        case OpId::Sin:        return OpType::Sin;
        case OpId::Cos:        return OpType::Cos;
        case OpId::Rsqrt:      return OpType::Rsqrt;

        // Normalization (dispatched form; the nn RMSNorm layer's SIMD fast
        // path records via jit_record_rms_norm, but a raw OpId::RMSNorm
        // dispatch must map too so its output is a real node, not a frozen
        // constant).
        case OpId::RMSNorm:    return OpType::RMSNorm;

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

        // Comparison (Bool output) and logical ops. Mapping these makes a
        // data-dependent predicate (a > b, logical_and(...)) a real graph node
        // instead of an unmapped op that graph-breaks and gets frozen as a
        // trace-time constant — the bug that made traced cond()/while_loop()
        // ignore their runtime inputs.
        case OpId::Eq:         return OpType::Eq;
        case OpId::Ne:         return OpType::Ne;
        case OpId::Lt:         return OpType::Lt;
        case OpId::Le:         return OpType::Le;
        case OpId::Gt:         return OpType::Gt;
        case OpId::Ge:         return OpType::Ge;
        case OpId::LogicalAnd: return OpType::LogicalAnd;
        case OpId::LogicalOr:  return OpType::LogicalOr;
        case OpId::LogicalNot: return OpType::LogicalNot;

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

        // Cast: dispatched by Tensor::to(dtype) on EVERY backend (CPU and GPU),
        // so a mixed-precision trace records a Cast node uniformly.
        case OpId::Cast:       return OpType::Cast;

        // Index ops
        case OpId::IndexSelect: return OpType::IndexSelect;

        // Vision
        case OpId::Interpolate: return OpType::Interpolate;

        default:
            return std::nullopt;
    }
}

// A deterministic tensor-CREATION op takes no tensor inputs and produces a
// value that is fully determined at trace time (a fill / range / identity). On
// CPU some of these are direct fills that never dispatch an OpId, but on GPU
// they DO dispatch (e.g. Variable::operator*(double) builds its scalar operand
// via `full`, which lowers to OpId::Full on CUDA/ROCm). Without special
// handling the tracer sees an unmapped OpId and declares a GRAPH BREAK, which
// discards the ENTIRE compiled graph and silently falls back to eager — so any
// compiled function that uses a scalar op (`x * 2.0`, `sum(z) * scale`, …) or a
// literal creation op was un-JIT-able on GPU. These outputs are genuine
// constants: registering them (so a downstream consumer resolves them via
// end_trace's constant-baking path) is correct and device-independent. Random
// creation ops are deliberately EXCLUDED — baking a single draw would freeze
// the randomness, so those still break.
static auto is_constant_creation_op(OpId op) -> bool {
    switch (op) {
        case OpId::Full:
        case OpId::Zeros:
        case OpId::Ones:
        case OpId::Arange:
        case OpId::Linspace:
        case OpId::Eye:
            return true;
        default:
            return false;
    }
}

// A transparent value-identity op returns a tensor with the SAME values as its
// single input but possibly different storage/strides — a pure layout
// materialization with no IR OpType. contiguous() is the canonical case: bias-
// less Linear lowers to permute+matmul and matmul materializes a contiguous
// copy of the permuted weight, so without this every bias-less Linear would
// graph-break and fall back to eager. The tracer aliases the output to its
// input and records no node (values are unchanged, so replay stays exact).
static auto is_value_identity_op(OpId op) -> bool {
    switch (op) {
        case OpId::Contiguous:
            return true;
        default:
            return false;
    }
}

// A pure view/shape op returns a tensor that shares its input's storage and,
// when the target shape/strides match exactly, the SAME logical view — a
// genuine no-op the tracer should drop. Any OTHER op whose output aliases an
// input is an in-place COMPUTE mutation (data changed under the same storage)
// and MUST be recorded as a value-versioned node instead of being dropped.
static auto is_view_optype(OpType t) -> bool {
    switch (t) {
        case OpType::Reshape:
        case OpType::Transpose:
        case OpType::Permute:
        case OpType::Squeeze:
        case OpType::Unsqueeze:
        case OpType::Flatten:
        case OpType::Broadcast:
            return true;
        default:
            return false;
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
            // A deterministic, input-less creation op (e.g. GPU `full` behind a
            // scalar operation) is NOT a graph break: its output is a constant.
            // Register the output tensors so a downstream consumer resolves them
            // through end_trace's constant-baking path, and record NO node. This
            // makes scalar/creation ops JIT-able uniformly on CPU and GPU.
            if (inputs.empty() && is_constant_creation_op(op)) {
                for (auto& t : results) {
                    tracer.register_tensor(t);
                }
                return results;
            }
            // A transparent value-identity op (e.g. Contiguous) is NOT a graph
            // break: it produces the same values as its input. Alias each output
            // to the input's traced value and record no node so downstream
            // consumers resolve through the input and the graph stays intact.
            if (!inputs.empty() && !results.empty() && is_value_identity_op(op)) {
                const std::string in_id = tracer.register_tensor(inputs[0]);
                for (auto& t : results) {
                    tracer.alias_tensor(t, in_id);
                }
                return results;
            }
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

        // Skip pure identity/no-op ops: a view op that returns its input
        // unchanged (e.g. expand/reshape/squeeze to the SAME shape+strides, which
        // share the input's tensor id) produces no new value. Recording it would
        // create a self-referential node AND a duplicate graph value id (two
        // Values both named e.g. "t2"), which DCE/topological-sort then
        // mishandle — pruning the real producer and leaving a dangling output.
        // This is exactly how a batched matmul (which decomposes into identity
        // expand/reshape around a bmm) lost its bmm node. The output tensor id
        // still resolves to the real producer, so dropping the no-op is correct.
        //
        // BUT this must fire only for genuine VIEW no-ops. A COMPUTE op whose
        // output aliases an input is an in-place mutation (relu_-style): the
        // storage is the same but the data changed, so it must be recorded as a
        // value-versioned node (new = op(old)) rather than dropped — otherwise
        // later reads of the mutated tensor resolve to the pre-op value.
        if (!output_ids.empty()) {
            bool all_alias_input = true;
            for (const auto& oid : output_ids) {
                bool is_input = false;
                for (const auto& iid : input_ids) {
                    if (iid == oid) { is_input = true; break; }
                }
                if (!is_input) { all_alias_input = false; break; }
            }
            if (all_alias_input) {
                if (is_view_optype(*op_type)) {
                    // Genuine identity view op — drop it.
                    return results;
                }
                // In-place compute mutation reaching the interceptor: mint fresh
                // SSA values for the outputs and REMAP their fingerprints so
                // downstream reads resolve to the post-op value. Fall through to
                // record a real node (new = op(old, ...)).
                std::vector<std::string> versioned;
                versioned.reserve(results.size());
                for (auto& t : results) {
                    versioned.push_back(tracer.register_new_tensor(t));
                }
                output_ids = std::move(versioned);
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
        // Transpose axes: eager transpose dispatches with Dim0/Dim1 and the
        // graph executor + shape inference read "dim0"/"dim1". Without these
        // the traced transpose replayed as transpose(0,0) — a silent no-op.
        copy_int(AttrKey::Dim0,        "dim0");
        copy_int(AttrKey::Dim1,        "dim1");
        // Flatten range: eager flatten dispatches with StartDim/EndDim and the
        // executor reads "start_dim"/"end_dim". Without these the traced
        // flatten replayed over the full [0, -1] range regardless of the
        // recorded sub-range.
        copy_int(AttrKey::StartDim,    "start_dim");
        copy_int(AttrKey::EndDim,      "end_dim");
        // Reduction keepdim flag — graph.cpp's infer_types() needs this to
        // correctly compute reduced shapes (without it, infer_types
        // defaults to keepdim=false and silently drops the kept dim).
        auto copy_bool = [&](AttrKey k, const char* name) {
            if (attrs.has(k)) traced.bool_attrs[name] = attrs.get_bool(k, false);
        };
        copy_bool(AttrKey::Keepdim, "keepdim");
        copy_bool(AttrKey::AlignCorners, "align_corners");
        // AvgPool2d's count_include_pad option (default true). Stored by the
        // pooling layer as an INT (set(AttrKey::CountIncludePad, int64_t{0|1})),
        // so it must be copied via copy_int — copy_bool/get_bool misreads an
        // Int64-tagged AttrValue. Without capturing it the traced node dropped
        // the flag, so both the MLIR lowering and the interpreted executor
        // silently used count_include_pad=true, diverging from an eager
        // avg_pool2d(count_include_pad=false) with non-zero padding.
        copy_int(AttrKey::CountIncludePad, "count_include_pad");
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
        copy_int_list_to_vec(AttrKey::OutputSize, "output_size");
        // LayerNorm normalized_shape: dispatched as a comma-separated STRING
        // (AttrKey::NormalizedShape). Parse it to an int-list so the executor's
        // LayerNorm node (which reads the "normalized_shape" vec attr) replays
        // over the correct trailing dims instead of defaulting to the last dim.
        if (attrs.has(AttrKey::NormalizedShape)) {
            const std::string ns{attrs.get_string(AttrKey::NormalizedShape, "")};
            std::vector<int64_t> dims;
            size_t start = 0;
            while (start < ns.size()) {
                size_t comma = ns.find(',', start);
                std::string tok = ns.substr(start, comma == std::string::npos
                                                        ? std::string::npos
                                                        : comma - start);
                if (!tok.empty()) {
                    try { dims.push_back(std::stoll(tok)); } catch (...) {}
                }
                if (comma == std::string::npos) break;
                start = comma + 1;
            }
            if (!dims.empty()) traced.vec_attrs["normalized_shape"] = std::move(dims);
        }
        copy_int_list_to_vec(AttrKey::Dims,  "dims");
        copy_int_list_to_vec(AttrKey::Starts, "starts");
        copy_int_list_to_vec(AttrKey::Ends,   "ends");
        if (attrs.has(AttrKey::Mode)) {
            const auto mode = attrs.get_string(AttrKey::Mode, "bilinear");
            if (mode == "nearest") {
                traced.int_attrs["mode"] = 0;
            } else if (mode == "bilinear") {
                traced.int_attrs["mode"] = 1;
            } else if (mode == "bicubic") {
                traced.int_attrs["mode"] = 2;
            } else if (mode == "trilinear") {
                traced.int_attrs["mode"] = 3;
            } else {
                traced.int_attrs["mode"] = -1;
            }
        }
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
