/**
 * @file tracing_interceptor.cpp
 * @brief Implementation of the dispatch-level tracing interceptor
 */

#include "tenzor/jit/tracing_interceptor.hpp"
#include <algorithm>
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

        // JIT-R054: scaled_dot_product_attention()'s fast path dispatches
        // OpId::FlashAttention directly (not just via FuseAttentionPass
        // synthesizing the node from a decomposed MatMul/Softmax pattern).
        // Without this mapping, any traced call taking the fast path (4D
        // Q/K/V, no explicit mask, supported dtype — the common case)
        // unconditionally graph-broke the entire surrounding trace.
        case OpId::FlashAttention: return OpType::FlashAttention;

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
        // JIT-R091 (found via CrossEntropyLoss's segmentation-target ignore-
        // mask denominator, tenzor::clamp_min(sum(mask), 1.0)): both were
        // entirely unmapped, causing an unconditional hard graph break on
        // ANY traced clamp_min/clamp_max call. Both reuse OpType::Clamp
        // as-is — clamp_min sets only AttrKey::Min, clamp_max sets only
        // AttrKey::Max, and execute_node's Clamp case already treats a
        // missing bound as unbounded on that side (see graph.cpp), so no
        // new OpType/execute_node case is needed.
        case OpId::ClampMin:   return OpType::Clamp;
        case OpId::ClampMax:   return OpType::Clamp;
        case OpId::Sin:        return OpType::Sin;
        case OpId::Cos:        return OpType::Cos;
        case OpId::Rsqrt:      return OpType::Rsqrt;
        case OpId::Reciprocal: return OpType::Reciprocal;
        // Both previously unmapped -> any traced use of round()/fmod(), including
        // internally within tenzor::float_power's negative-base decomposition,
        // graph-broke the ENTIRE compiled graph regardless of how small the use.
        case OpId::Round:      return OpType::Round;
        case OpId::Fmod:       return OpType::Fmod;

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
        // JIT-R090: slice_scatter() is the primary correctly-traceable
        // mechanism for KV-cache-style incremental updates (see
        // src/nn/utils/kv_cache.cpp). Without this mapping every traced
        // cache update graph-broke, permanently blocking JIT compilation of
        // any autoregressive decode loop using it.
        case OpId::SliceScatter: return OpType::SliceScatter;

        // Linear
        case OpId::Linear:     return OpType::Linear;

        // Embedding
        case OpId::Embedding:  return OpType::Embedding;
        // JIT-R056 EXTENDED: EmbeddingBagForward's GPU dispatch sites had no
        // opid_to_optype case, so a traced call graph-broke (safely, but
        // permanently blocking JIT compilation of any model using
        // EmbeddingBag on GPU).
        case OpId::EmbeddingBagForward: return OpType::EmbeddingBag;

        // Dropout
        case OpId::Dropout:    return OpType::Dropout;

        // Cast: dispatched by Tensor::to(dtype) on EVERY backend (CPU and GPU),
        // so a mixed-precision trace records a Cast node uniformly.
        case OpId::Cast:       return OpType::Cast;

        // Index ops
        case OpId::IndexSelect: return OpType::IndexSelect;

        // JIT review C2: map previously-unmapped single-dispatch ops so they are
        // captured (and replayed through the interpreter) instead of graph-breaking
        // to eager. Attributes are copied generically below.
        case OpId::Var:          return OpType::Var;
        case OpId::Std:          return OpType::Std;
        case OpId::Prod:         return OpType::Prod;
        case OpId::LeakyReLU:    return OpType::LeakyReLU;
        case OpId::Elu:          return OpType::ELU;
        case OpId::Mish:         return OpType::Mish;
        case OpId::Softplus:     return OpType::Softplus;
        // JIT-R084: Swish/RReLU/LogSigmoid's no-grad-fast-path dispatch<>()
        // call was previously unmapped here, graph-breaking the whole trace
        // (a "masked landmine," not silent-wrong-answer) — reachable in real
        // deployed inference via nn::Swish (MobileNetV3/EfficientNet-style
        // blocks). Mirrors Mish/Softplus's identical treatment.
        case OpId::Swish:        return OpType::Swish;
        case OpId::RReLU:        return OpType::RReLU;
        case OpId::LogSigmoid:   return OpType::LogSigmoid;
        case OpId::GroupNorm:    return OpType::GroupNorm;
        case OpId::InstanceNorm: return OpType::InstanceNorm;
        case OpId::Gather:       return OpType::Gather;
        case OpId::Scatter:      return OpType::Scatter;
        // JIT-R064: Minimum/IndexCopy's dispatch<>() call was previously
        // unmapped here, graph-breaking the whole trace — reachable in real
        // deployed inference via Embedding::renorm_embeddings' max_norm
        // clamp (scale = min(1, max_norm/norm); weight = index_copy(weight,
        // 0, idx, rows * scale)), which runs on every forward call for any
        // Embedding(max_norm>0) module.
        case OpId::Minimum:      return OpType::Minimum;
        case OpId::IndexCopy:    return OpType::IndexCopy;
        case OpId::Flip:         return OpType::Flip;
        case OpId::Roll:         return OpType::Roll;

        // Triu/Tril/Diag/Trace: now dispatch unconditionally on every device
        // (see ops/transform.cpp) instead of a CPU-only manual path that
        // bypassed dispatch — mapping them here is what actually lets the
        // JIT tracer capture them, matching Flip (JIT-R038).
        case OpId::Triu:         return OpType::Triu;
        case OpId::Tril:         return OpType::Tril;
        case OpId::Diag:         return OpType::Diag;
        case OpId::Trace:        return OpType::Trace;

        // Vision
        case OpId::Interpolate: return OpType::Interpolate;

        // JIT-R091: was entirely unmapped, making every OneHot call an
        // unconditional hard graph-break (abort_trace_unmappable) — masking
        // CrossEntropyLoss/NLLLoss's segmentation-target path as a landmine
        // rather than the "live" freeze the raw-permute half of the bug
        // implied (the permute code was unreachable during tracing since
        // OneHot always aborted first). Output shape: input_shape appended
        // with num_classes (see infer_types' OneHot case).
        case OpId::OneHot: return OpType::OneHot;

        // JIT-R055: the direct dispatch<OpId::SparseSpMM>(crow,col,values,dense)
        // convention (used by SparseLinear and the JVP sparse rules), distinct
        // from OpType::SparseMatMul (a dense-weight-retagged-sparse node
        // synthesized only by SparsePass, with a [x, weight] input layout).
        case OpId::SparseSpMM: return OpType::SparseSpMM;
        // JIT-R098: same direct-dispatch CSR convention, extended from
        // spmm-only (JIT-R055) to the other autograd/ops.cpp sparse entry
        // points (spmv/sparse_add/sparse_triangular_solve) so they are no
        // longer zero-dispatch/tracer-invisible.
        case OpId::SparseSpMV: return OpType::SparseSpMV;
        case OpId::SparseAdd: return OpType::SparseAdd;
        case OpId::SparseTrsv: return OpType::SparseTrsv;
        case OpId::SparseTrsm: return OpType::SparseTrsm;
        // JIT-R098 (continued): unblocks sparse_add's manually-recorded node
        // — see the OpType::ScatterAdd/RepeatInterleave doc comments in
        // tracer.hpp for why these needed mapping too.
        case OpId::ScatterAdd: return OpType::ScatterAdd;
        case OpId::RepeatInterleave: return OpType::RepeatInterleave;

        // JIT-R100: same rationale as ScatterAdd/RepeatInterleave above —
        // needed so CumSum/Sort/Nonzero/Bincount calls nested inside the
        // SparseTensor structural-conversion methods (from_dense/coalesce/
        // to_csr/to_csc) don't graph-break the whole trace. Also a general
        // win for any other traced code calling these directly.
        case OpId::CumSum: return OpType::CumSum;
        case OpId::Sort: return OpType::Sort;
        case OpId::Nonzero: return OpType::Nonzero;
        case OpId::Bincount: return OpType::Bincount;

        // JIT-R082/R086/R087: box_iou/nms/ROIAlignOp::apply already dispatch
        // through these OpIds unconditionally (see src/ops/detection.cpp,
        // src/nn/detection/roi_ops.cpp) — mapping them lets the generic
        // interceptor auto-record, same treatment as CumSum/Sort/Nonzero/
        // Bincount above. Must land together with JIT-R085's tracer-
        // visibility fixes for encode_boxes/decode_boxes/clip_boxes_to_image/
        // remove_small_boxes/select/nonzero/AnchorGenerator::generate — see
        // findings.txt's documented ordering hazard.
        case OpId::BoxIoU: return OpType::BoxIoU;
        case OpId::NMS: return OpType::NMS;
        case OpId::ROIAlignForward: return OpType::ROIAlignForward;

        // JIT-R058b: OpId::QuantizedLinear is deliberately left UNMAPPED here.
        // QuantizedLinear::forward_quantized's GPU fast path calls
        // dispatch<OpId::QuantizedLinear>(input_int8, weight_int8, ...)
        // directly, but that int8 activation is produced by quantize_tensor()
        // — a raw host pointer loop with zero dispatch() calls — so it has no
        // traced lineage back to the original float Variable regardless of
        // whether this raw dispatch call itself is mapped. Auto-mapping it
        // here would record a node with a frozen/wrong-lineage input AND
        // double-record alongside the correctly-lineaged node that
        // QuantizedLinear::forward_impl now records manually (via
        // jit_record_quantized_linear_static, from the RAW FLOAT input,
        // re-quantized inside execute_node — see graph.cpp's
        // OpType::QuantizedLinearStatic case). Leaving this unmapped means a
        // trace that reaches this raw dispatch call through any OTHER path
        // (bypassing forward_impl) safely aborts to eager instead of
        // recording a wrong node — matches OpType::QuantizedLinear above
        // (dynamic quantization of a dense weight, synthesized only by
        // QuantizationPass with a [x, weight] input layout) also NOT being
        // reused here: that would silently reinterpret pre-quantized int8
        // data as a dense float weight to re-quantize.
        // case OpId::QuantizedLinear: intentionally not mapped.

        // findings.txt JIT-R115: MultiheadAttention::scaled_dot_product_attention's
        // cuDNN/fused-SDPA fast path dispatches OpId::FusedAttention directly
        // (registered on every backend, not just CUDA) — without this mapping
        // every traced attention forward using that fast path unconditionally
        // hard graph-broke the whole trace. FlashAttention's execute_node case
        // already decomposes to softmax(scale*QK^T [+ causal mask])V from
        // exactly [Q,K,V(,mask)] + scale/causal/dropout_p attrs, the same
        // inputs/attrs this dispatch site provides, so reusing that node type
        // replays it correctly without a fused kernel.
        case OpId::FusedAttention: return OpType::FlashAttention;

        // findings.txt JIT-R132: tenzor::ops::norm() (src/ops/reduction.cpp)
        // dispatches unconditionally on EVERY device (no gating at all,
        // unlike most other ops here) — was entirely unmapped, hard
        // graph-breaking any traced norm()/Tensor::norm() call on every
        // backend including CPU.
        case OpId::Norm: return OpType::Norm;

        // findings.txt JIT-R133: the linalg family (det/inv/solve/cholesky/
        // svd/qr/eigh) already has full OpType + execute_node +
        // infer_types/infer_symbolic_types support (used elsewhere), just no
        // opid_to_optype mapping — GPU/complex calls (which reach here via
        // try_gpu_dispatch's dispatch_single()) hard graph-broke; the CPU
        // real-dtype path (which never reached dispatch() at all) is fixed
        // separately via manual jit_record_* calls in src/ops/linalg.cpp.
        case OpId::LinalgDet:      return OpType::Det;
        case OpId::LinalgInv:      return OpType::Inv;
        case OpId::LinalgSolve:    return OpType::Solve;
        case OpId::LinalgCholesky: return OpType::Cholesky;
        case OpId::LinalgSVD:      return OpType::Svd;
        case OpId::LinalgQR:       return OpType::Qr;
        case OpId::LinalgEigh:     return OpType::Eigh;

        // JIT-R110: same pattern as the block above -- these 9 LAPACK-
        // backed ops already had real backend kernels (dispatched via
        // try_gpu_dispatch/dispatch()) but no OpType/opid_to_optype mapping
        // at all, so GPU/complex calls hard graph-broke the whole trace; the
        // CPU real-dtype path (which never reaches dispatch()) is fixed
        // separately via manual jit_record_* calls in src/ops/linalg.cpp.
        case OpId::SolveTriangular:   return OpType::SolveTriangular;
        case OpId::LinalgEig:         return OpType::LinalgEig;
        case OpId::LinalgLU:          return OpType::LinalgLU;
        case OpId::LinalgLUSolve:     return OpType::LinalgLUSolve;
        case OpId::LinalgHouseholder: return OpType::LinalgHouseholder;
        case OpId::LinalgLDLFactor:   return OpType::LinalgLDLFactor;
        case OpId::LinalgLDLSolve:    return OpType::LinalgLDLSolve;
        case OpId::Ormqr:             return OpType::Ormqr;
        case OpId::Geqrf:             return OpType::Geqrf;

        // Needed for slogdet()'s GPU-dispatch composition (sign(det) *
        // log(abs(det))) to be fully traceable now that Det is mapped above —
        // tenzor::sign() dispatches OpId::Sign unconditionally and was
        // otherwise unmapped, same hard-graph-break exposure.
        case OpId::Sign: return OpType::Sign;

        // findings.txt JIT-R134: real, reachable dispatch call sites
        // (src/ops/vision.cpp's grid_sample/affine_grid, src/ops/fft.cpp's
        // fft(), src/nn/loss/losses_advanced.cpp's CTCLoss) with no prior
        // mapping — hard graph-broke any trace using spatial-transformer
        // primitives, FFT, or CTC loss. execute_node re-dispatches these
        // directly for inference replay (see findings.txt for why no
        // decomposition is attempted).
        case OpId::GridSample:      return OpType::GridSample;
        case OpId::AffineGrid:      return OpType::AffineGrid;
        case OpId::FFT:             return OpType::FFT;
        // IFFT/RFFT/IRFFT share FFT's exact [input] + dim/n/norm attr shape
        // (src/ops/fft.cpp's ifft()/rfft()/irfft() all set the same 3
        // AttrKeys) -- same unmapped-OpId gap, fixed the same way.
        case OpId::IFFT:            return OpType::IFFT;
        case OpId::RFFT:            return OpType::RFFT;
        case OpId::IRFFT:           return OpType::IRFFT;
        case OpId::CTCLossForward:  return OpType::CTCLossForward;

        // findings.txt JIT-R116: ops::FusionOptimizer's fused-op dispatch
        // sites (src/ops/fused_ops.cpp, src/ops/fusion_optimizer.cpp) hit
        // this exact same unmapped-OpId hard-break mechanism if FusionOptimizer
        // is ever wired into a JIT-traced path (currently unreachable — see
        // findings.txt — but mapped here defensively so that combination
        // fails safely-and-traceably instead of hard-breaking, matching the
        // other Fused* treatment above).
        case OpId::FusedLinearReLU:          return OpType::FusedLinearReLU;
        case OpId::FusedConv2dReLU:          return OpType::FusedConv2dReLU;
        case OpId::FusedConv2dBnReLU:        return OpType::FusedConv2dBnReLU;
        case OpId::FusedSoftmaxCrossEntropy: return OpType::FusedSoftmaxCrossEntropy;
        case OpId::FusedAddReLU:             return OpType::FusedAddReLU;

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
// graph-break and fall back to eager. clone() is the other case: it exists at
// the eager Tensor API purely to give the caller a storage-independent copy
// for later in-place mutation safety, a hazard that doesn't exist in the
// traced graph's pure-dataflow model (node outputs are immutable values), so
// its traced VALUE is identical to its input's. The tracer aliases the
// output to its input and records no node (values are unchanged, so replay
// stays exact).
static auto is_value_identity_op(OpId op) -> bool {
    switch (op) {
        case OpId::Contiguous:
        case OpId::Clone:
            return true;
        default:
            return false;
    }
}

// R2-T1 regression (found via the first real trace-through-CompiledFunction
// numeric test for nn::RMSNorm — every prior RMSNorm MLIR/nvrtc test built
// its Graph by hand, bypassing the tracer entirely, so this never fired):
// RMSNorm::forward_impl's CUDA/Vulkan inference fast paths and its standard
// (training-capable, all-device) path all call `dispatch<OpId::FusedRMSNorm>`
// directly, then separately call `jit_record_rms_norm` (normalization.cpp) to
// record the correctly-shaped (x, weight) -> output OpType::RMSNorm node —
// unlike LayerNorm, which relies solely on OpId::FusedLayerNorm being mapped
// in opid_to_optype. OpId::FusedRMSNorm is deliberately left UNMAPPED there
// (mapping it would make the interceptor ALSO auto-record a node for the same
// dispatch call, double-recording alongside jit_record_rms_norm's manual one).
// But leaving it unmapped meant every one of those `dispatch<>` calls fell
// through to the interceptor's "Graph break: unmappable operation" branch —
// which fires unconditionally, before jit_record_rms_norm's call afterward
// ever gets a chance to matter — so tracing ANY nn::RMSNorm forward on CUDA,
// Vulkan, or via the standard/training path (i.e. every path except the CPU
// Float32 no-grad pointer fast path, which never calls dispatch<>() at all)
// permanently broke the entire surrounding JIT trace on both the nvrtc and
// mlir backends, silently falling back to eager (or hard-failing in
// strict/fullgraph mode). This predicate marks such "handled by a manual
// jit_record_* call right after this dispatch returns" ops so the interceptor
// can defer to that call instead of registering anything or breaking.
static auto is_manually_recorded_op(OpId op) -> bool {
    switch (op) {
        case OpId::FusedRMSNorm:
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

        // JIT-F016: if a mappable op mutates one of its inputs in place, the eager
        // execution below overwrites that input's storage BEFORE we register the
        // inputs, so end_trace would bake the POST-mutation data for a captured
        // constant leaf. Snapshot the pre-op value of every REGISTERED input (a
        // potential captured leaf/constant) now, so the in-place branch can
        // restore it. Only clones already-tracked inputs, and only during tracing
        // (a one-time cost that never touches the inference/replay hot path).
        std::vector<std::optional<Tensor>> pre_inputs;
        if (op_type && !inputs.empty()) {
            pre_inputs.resize(inputs.size());
            for (std::size_t k = 0; k < inputs.size(); ++k) {
                if (tracer.is_registered(inputs[k])) {
                    pre_inputs[k] = inputs[k].clone();
                }
            }
        }

        // Execute eagerly first (we always run the op)
        auto results = next(op, inputs, attrs);

        if (!op_type) {
            // A deterministic, input-less creation op (e.g. GPU `full` behind a
            // scalar operation) is NOT a graph break: its output is a constant.
            // Register the output tensors so a downstream consumer resolves them
            // through end_trace's constant-baking path, and record NO node. This
            // makes scalar/creation ops JIT-able uniformly on CPU and GPU.
            // Must use register_new_tensor, NOT the fingerprint-deduping
            // register_tensor: a freshly-allocated creation-op result (e.g.
            // ones_like(z_gate) inside a per-timestep RNN loop) can and does
            // land at an address a PRIOR, now-freed intermediate used —
            // register_tensor would then silently alias the new constant to
            // that unrelated dead tensor's id (confirmed: ones_like's output
            // inside QuantizedGRU's gate loop collided with an earlier
            // gate-split tensor's id, corrupting the graph — JIT-R058a
            // regression testing).
            if (inputs.empty() && is_constant_creation_op(op)) {
                for (auto& t : results) {
                    tracer.register_new_tensor(t);
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
            // A "manually recorded" op (e.g. FusedRMSNorm) is NOT a graph
            // break either: its caller records the real (correctly-shaped,
            // correctly-attributed) node itself immediately after this
            // dispatch call returns (see jit_record_rms_norm). Register
            // nothing and defer entirely to that call.
            if (is_manually_recorded_op(op)) {
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

        // Snapshot the input/id correspondence BEFORE any op-specific
        // reordering below (e.g. BatchNorm2dForwardAffine) mutates
        // input_ids, so register_output's index-based inputs[k]<->id
        // lookup stays correct regardless.
        const std::vector<std::string> orig_input_ids = input_ids;

        // Register one output tensor's id. A freshly-computed output must
        // ALWAYS get a fresh id (register_new_tensor) UNLESS it is a
        // genuine identity alias of one of this op's own inputs — the FULL
        // logical view matches (data_ptr + dtype + device + shape +
        // strides), checked locally against THIS op's inputs, not via
        // tensor_fingerprint's global dedup lookup. Shape/strides must be
        // part of the check, not just data_ptr+device: a view op like
        // Squeeze/Unsqueeze/Permute shares its input's storage (same
        // data_ptr) while presenting a genuinely DIFFERENT shape — matching
        // on data_ptr alone would misclassify every such op as a same-value
        // no-op and drop it (is_view_optype below), silently truncating the
        // graph. Global fingerprint-based dedup (register_tensor) is
        // separately unsafe here regardless: once a temporary tensor from
        // an earlier op is freed, the allocator can and does hand the SAME
        // address to a later, semantically unrelated output of matching
        // shape/dtype/device (e.g. every iteration of a per-timestep RNN
        // loop allocates same-shaped gate tensors) — register_tensor would
        // then silently alias the new value to the dead tensor's old id,
        // corrupting the graph. Both failure modes were confirmed via
        // QuantizedGRU's per-timestep gate computation (JIT-R058a
        // regression testing): the global-dedup version wrongly aliased
        // across loop iterations; a data_ptr-only local version wrongly
        // dropped every Squeeze/Unsqueeze as a no-op.
        auto same_view = [](const Tensor& a, const Tensor& b) -> bool {
            if (a.data_ptr() != b.data_ptr() || a.dtype() != b.dtype() ||
                !(a.device() == b.device())) {
                return false;
            }
            if (a.shape().size() != b.shape().size() ||
                !std::equal(a.shape().begin(), a.shape().end(), b.shape().begin())) {
                return false;
            }
            if (a.strides().size() != b.strides().size() ||
                !std::equal(a.strides().begin(), a.strides().end(), b.strides().begin())) {
                return false;
            }
            return true;
        };
        auto register_output = [&](const Tensor& t) -> std::string {
            for (std::size_t k = 0; k < inputs.size(); ++k) {
                if (same_view(t, inputs[k])) {
                    return orig_input_ids[k];
                }
            }
            return tracer.register_new_tensor(t);
        };

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
            op == OpId::BatchNorm2dForward ||
            // JIT-R014: GroupNorm/InstanceNorm kernels also return auxiliary
            // (mean, inv_std) tensors on every backend, but Graph::execute_node's
            // OpType::GroupNorm/InstanceNorm cases (graph.cpp) only ever produce
            // 1 output Variable at replay -- registering all 3 here built 2 Values
            // with no producer-side replay support (no consumer could safely use
            // them, and the output-binding loop silently dropped them anyway).
            // Surface only the primary output, matching BatchNorm2d above.
            op == OpId::GroupNorm ||
            op == OpId::InstanceNorm ||
            // findings.txt JIT-R125: LayerNorm/FusedLayerNorm's registered
            // kernels also return the {output, mean, rstd} triple on every
            // backend (see docs/internals/attention-contract.md), but
            // execute_node's OpType::LayerNorm case only ever produces 1
            // output Variable at replay -- same missing-sibling gap as the
            // GroupNorm/InstanceNorm fix above. OpId::RMSNorm is included
            // too for hygiene (it has the identical {output, rrms} shape),
            // even though it has no live dispatch call site today.
            op == OpId::LayerNorm ||
            op == OpId::FusedLayerNorm ||
            op == OpId::RMSNorm ||
            // JIT-R155-adjacent: FlashAttention/FusedAttention/FlexAttention's
            // dispatch wrappers (run_flash_dispatch/run_fused_dispatch/
            // run_flex_dispatch in function_attention.cpp) always return a
            // fixed-size vector, padding any auxiliary output the current
            // call didn't produce (logsumexp L, dropout seed/offset) with an
            // EMPTY/invalid filler Tensor -- e.g. any inference call
            // (is_training=false) or dropout_p==0 leaves L/seed/offset
            // invalid. Without this branch, the generic loop below called
            // register_output on every result including those invalid
            // fillers, and register_output's same_view() helper immediately
            // calls .data_ptr()/.dtype()/.device() on the tensor, throwing
            // "Operation on uninitialized tensor" -- i.e. tracing a plain
            // inference-mode tenzor::flash_attention()/fused_attention()
            // call directly (not via FuseAttentionPass) crashed unconditionally.
            // Surface only the primary output, matching LayerNorm/RMSNorm above.
            op == OpId::FlashAttention ||
            op == OpId::FusedAttention ||
            op == OpId::FlexAttention) {
            if (!results.empty()) {
                output_ids.push_back(register_output(results[0]));
            }
        } else {
            output_ids.reserve(results.size());
            for (auto& t : results) {
                output_ids.push_back(register_output(t));
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
                // JIT-F016: restore each mutated input's PRE-op value under its
                // id, so a captured constant leaf read pre-mutation elsewhere
                // bakes its pre-mutation value (not the post-op data). An input
                // that was NOT mutated gets its identical clone back (a no-op).
                for (std::size_t k = 0;
                     k < input_ids.size() && k < pre_inputs.size(); ++k) {
                    if (pre_inputs[k]) {
                        tracer.set_pre_op_snapshot(input_ids[k], *pre_inputs[k]);
                    }
                }
            }
        }

        // Record the operation
        TracedOp traced(*op_type, std::move(input_ids), std::move(output_ids));

        // Transfer attributes from OpAttributes into the TracedOp's string-keyed
        // maps so graph.cpp's replay (which looks them up by name) can find
        // them. Previously the traced op lost all attributes, so ops like Pow
        // replayed with exponent=0 and mean/sum(dim) replayed with dim=0.
        auto copy_float = [&](AttrKey k, const char* name) {
            // Store at full double precision (get_float returns double); the
            // node's scalar attrs are now double so f64 graphs keep the exact
            // value (pow exponent, clamp bounds, eps) — JIT-F057.
            if (attrs.has(k)) traced.attrs[name] = attrs.get_float(k);
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
        copy_float(AttrKey::Lower, "lower");  // JIT-R084: RReLU
        copy_float(AttrKey::High,  "high");   // JIT-R084: RReLU
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
        copy_bool(AttrKey::Unbiased, "unbiased");   // Var/Std
        copy_int(AttrKey::NumGroups, "num_groups");  // GroupNorm
        copy_int(AttrKey::Shift, "shift");           // Roll
        copy_int(AttrKey::Diagonal, "diagonal");     // Triu/Tril/Diag
        copy_int(AttrKey::M, "m");                   // SparseSpMM/SparseSpMV/SparseAdd
        copy_int(AttrKey::K, "k");                   // SparseSpMM/SparseSpMV/SparseAdd
        copy_int(AttrKey::N, "n");                   // SparseTrsv/SparseTrsm (JIT-R098)
        copy_bool(AttrKey::Upper, "upper");           // SparseTrsv/SparseTrsm (JIT-R098)
        copy_int(AttrKey::NumRepeats, "num_repeats"); // RepeatInterleave (JIT-R098)
        copy_bool(AttrKey::Descending, "descending"); // Sort (JIT-R100)
        copy_int(AttrKey::Minlength, "minlength");    // Bincount (JIT-R100)
        copy_int(AttrKey::IouType, "iou_type");             // BoxIoU (JIT-R086)
        copy_float(AttrKey::IouThreshold, "iou_threshold"); // NMS (JIT-R087)
        copy_float(AttrKey::SpatialScale, "spatial_scale"); // ROIAlignForward (JIT-R082)
        copy_int(AttrKey::SamplingRatio, "sampling_ratio"); // ROIAlignForward (JIT-R082)
        copy_bool(AttrKey::Aligned, "aligned");              // ROIAlignForward (JIT-R082)
        copy_int(AttrKey::NumClasses, "num_classes"); // OneHot (JIT-R091)
        // Quantization scale/zero-point attrs: scale is Float64-tagged,
        // zero-point is Int64-tagged. Currently unused by any OpId this
        // interceptor auto-maps (QuantizedLinearStatic (JIT-R058b) is
        // recorded manually via jit_record_quantized_linear_static, which
        // sets these directly on the TracedOp and bypasses this generic
        // dispatch-interceptor path entirely — see quantized_layers.cpp).
        // Left in place as a harmless no-op for any future OpId that both
        // (a) is safe to auto-map here and (b) carries these same attrs.
        copy_float(AttrKey::InputScale, "input_scale");
        copy_float(AttrKey::WeightScaleQ, "weight_scale");
        copy_float(AttrKey::OutputScale, "output_scale");
        copy_int(AttrKey::InputZeroPoint, "input_zero_point");
        copy_int(AttrKey::WeightZeroPoint, "weight_zero_point");
        copy_bool(AttrKey::AlignCorners, "align_corners");
        // FlashAttention: scale/causal/dropout_p, read by execute_node's
        // OpType::FlashAttention case (graph.cpp) to reproduce the eager
        // scale, apply the causal mask, and refuse to replay dropout_p>0
        // (JIT-R054).
        copy_float(AttrKey::Scale, "scale");
        copy_bool(AttrKey::Causal, "causal");
        copy_float(AttrKey::DropoutP, "dropout_p");
        // JIT-R013: Dropout's own attrs use DIFFERENT AttrKeys (P/Training,
        // not DropoutP) — nested_dropout (nested_ops.cpp, the only real
        // producer of an OpType::Dropout node) sets these, but nothing
        // copied them into the traced node, so handle_dropout/graph.cpp's
        // interpreter had no way to see training=true even after being
        // taught to check for it.
        copy_float(AttrKey::P, "p");
        copy_bool(AttrKey::Training, "training");
        // AvgPool2d's count_include_pad option (default true). Stored by the
        // pooling layer as an INT (set(AttrKey::CountIncludePad, int64_t{0|1})),
        // so it must be copied via copy_int — copy_bool/get_bool misreads an
        // Int64-tagged AttrValue. Without capturing it the traced node dropped
        // the flag, so both the MLIR lowering and the interpreted executor
        // silently used count_include_pad=true, diverging from an eager
        // avg_pool2d(count_include_pad=false) with non-zero padding.
        copy_int(AttrKey::CountIncludePad, "count_include_pad");
        // MaxPool2d's ceil_mode (JIT-R060). Same INT64-tagged storage as
        // CountIncludePad above (set(AttrKey::CeilMode, int64_t{0|1})) — must
        // use copy_int, not copy_bool. Without this the traced node silently
        // used the floor output-size formula regardless of the real
        // ceil_mode, producing a wrong-shaped output on replay (Vulkan/MPS,
        // the only backends where ceil_mode=true is reachable).
        copy_int(AttrKey::CeilMode, "ceil_mode");
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
        // AdaptiveAvgPool2d/MaxPool dispatch the target size as per-axis
        // OutputSizeH/OutputSizeW (not the OutputSize list), so also emit the
        // "output_size" vec from that pair; otherwise the executor found no
        // output_size and produced a wrong (default) shape + severed grad
        // (JIT-056).
        copy_hw_pair(AttrKey::OutputSizeH, AttrKey::OutputSizeW, "output_size");
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
        // A multi-dim strided slice sets per-dim Steps too; emit them so the
        // traced starts/ends slice replays with the right stride (JIT-013). Gate
        // on Starts so this only fires for the slice form — AttrKey::Steps is
        // overloaded (e.g. Linspace uses it as a SCALAR count), and emitting a
        // "steps" vec for those would attach a spurious per-dim stride.
        if (attrs.has(AttrKey::Starts)) {
            copy_int_list_to_vec(AttrKey::Steps, "steps");
        }
        // Interpolate's "mode" string enum. AttrKey::Mode is reused with a
        // DIFFERENT string enum by EmbeddingBagForward ("sum"/"mean"/"max") —
        // exclude it here so it doesn't get misencoded via this block's
        // unrelated nearest/bilinear/bicubic/trilinear mapping (falling into
        // the `else -> -1` branch below); it gets its own dedicated encoding
        // right after (JIT-R056).
        if (op != OpId::EmbeddingBagForward && attrs.has(AttrKey::Mode)) {
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
        // EmbeddingBag's own AttrKey::Mode encoding ("sum"/"mean"/"max"),
        // under a distinct key so it never collides with Interpolate's
        // "mode" above. embedding_dim/include_last_offset complete the
        // shape-inference/execute_node contract (JIT-R056).
        if (op == OpId::EmbeddingBagForward && attrs.has(AttrKey::Mode)) {
            const auto eb_mode = attrs.get_string(AttrKey::Mode, "sum");
            traced.int_attrs["embedding_bag_mode"] =
                (eb_mode == "mean") ? 1 : (eb_mode == "max") ? 2 : 0;
        }
        // findings.txt JIT-R134: GridSample's mode ("bilinear"/"nearest"/
        // "bicubic", a subset of Interpolate's set) is intentionally left
        // to fall through to the generic AttrKey::Mode encoding above (same
        // 0/1/2 values, GridSample never uses "trilinear") -- no separate
        // branch needed. Its OWN padding_mode string ("zeros"/"border"/
        // "reflection") has no existing encoding anywhere, so it gets one
        // here under a distinct key.
        if (attrs.has(AttrKey::PaddingMode)) {
            const auto pm = attrs.get_string(AttrKey::PaddingMode, "zeros");
            traced.int_attrs["padding_mode"] =
                (pm == "border") ? 1 : (pm == "reflection") ? 2 : 0;
        }
        // findings.txt JIT-R134: FFT's norm string ("backward"/"forward"/"ortho").
        if (attrs.has(AttrKey::Norm)) {
            const auto norm = attrs.get_string(AttrKey::Norm, "backward");
            traced.int_attrs["norm"] =
                (norm == "forward") ? 1 : (norm == "ortho") ? 2 : 0;
        }
        // findings.txt JIT-R116: FusedSoftmaxCrossEntropy's reduction string
        // ("mean"/"sum"/"none").
        if (attrs.has(AttrKey::Reduction)) {
            const auto red = attrs.get_string(AttrKey::Reduction, "mean");
            traced.int_attrs["reduction"] =
                (red == "sum") ? 1 : (red == "none") ? 2 : 0;
        }
        // findings.txt JIT-R116/R134: HasBias (FusedLinearReLU/FusedConv2dReLU),
        // Momentum (FusedConv2dBnReLU), Blank/ZeroInfinity (CTCLossForward).
        copy_bool(AttrKey::HasBias, "has_bias");
        copy_float(AttrKey::Momentum, "momentum");
        copy_int(AttrKey::Blank, "blank");
        copy_bool(AttrKey::ZeroInfinity, "zero_infinity");
        copy_int(AttrKey::EmbeddingDim, "embedding_dim");
        copy_bool(AttrKey::IncludeLastOffset, "include_last_offset");
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
