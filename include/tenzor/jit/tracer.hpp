/**
 * @file tracer.hpp
 * @brief JIT operation tracing system for recording computation graphs
 *
 * Provides trace mode execution that records operations during forward pass
 * to build an intermediate representation (IR) graph. This is the primary
 * JIT compilation mode for Tenzor.
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <functional>
#include "../core/tensor.hpp"
#include "../core/jit_hooks.hpp"  // OpId / OpAttributes fwd decls for record_inplace
#include "../autograd/variable.hpp"
#include "../nn/module.hpp"
#include "graph.hpp"

namespace tenzor {
namespace jit {

/**
 * @brief Operation types supported by the tracer.
 *
 * Each type corresponds to a kernel operation that can be traced
 * and compiled into the IR graph.
 */
enum class OpType {
    // Arithmetic operations
    Add,
    Sub,
    Mul,
    Div,

    // Matrix operations
    MatMul,
    Bmm,

    // Activations
    ReLU,
    Sigmoid,
    Tanh,
    Softmax,
    LogSoftmax,

    // Pooling
    MaxPool2d,
    AvgPool2d,
    AdaptiveAvgPool2d,

    // Convolution
    Conv2d,
    ConvTranspose,

    // Normalization
    BatchNorm2d,
    LayerNorm,

    // Shape operations
    Reshape,
    Transpose,
    Permute,
    Squeeze,
    Unsqueeze,
    Flatten,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Element-wise
    Exp,
    Log,
    Sqrt,
    Pow,
    Abs,
    Neg,
    Clamp,
    Sin,      ///< Elementwise sine
    Cos,      ///< Elementwise cosine
    Rsqrt,    ///< Elementwise reciprocal square root (1/sqrt(x))
    Reciprocal, ///< Elementwise reciprocal (1/x)
    Round,    ///< Elementwise round-to-nearest-even
    Fmod,     ///< Elementwise C-style truncated modulo (sign follows dividend)

    // Indexing
    Slice,
    Cat,
    SliceScatter,  ///< Copy of input with src written into a dim slice [start,end,step)
    AsStrided,     ///< Arbitrary-stride view: (size, stride, storage_offset)
    ViewAsReal,    ///< Complex tensor -> real tensor with a trailing dim of 2
    ViewAsComplex, ///< Real tensor (trailing dim of 2) -> complex tensor

    // Other
    Dropout,
    Linear,
    Embedding,
    EmbeddingBag,  ///< Embedding lookup + per-bag aggregation (sum/mean/max);
                   ///< inputs [weight, offsets], attrs embedding_bag_mode/
                   ///< embedding_dim/include_last_offset

    // Activations (extended)
    GELU,

    // Linear algebra
    Det,
    Inv,
    Solve,
    Cholesky,
    Svd,
    Qr,
    Eigh,
    Eigvalsh,
    Norm,
    Slogdet,

    // Constants
    Constant,
    Input,
    Output,

    // Fused operations
    FlashAttention,   ///< Fused multi-head attention (Q*K^T -> scale -> softmax -> *V)
    FusedFFN,         ///< Fused feed-forward network (Linear -> GELU/ReLU -> Linear)
    ResidualAdd,      ///< Residual connection marker (x + sublayer(x))

    // Shape guard (for dynamic shape support)
    ShapeGuard,       ///< Runtime shape check that triggers re-trace on mismatch
    GuardNode,        ///< Data-dependent branch guard; failure triggers retrace

    // Memory management pseudo-ops
    SwapOut,          ///< GPU -> CPU async transfer for memory pressure relief
    SwapIn,           ///< CPU -> GPU async prefetch before reuse

    // Quantized operations
    QuantizedLinear,  ///< INT8 quantized linear layer — Retagged from Linear
                      ///< by QuantizationPass (dynamic quantization of a DENSE
                      ///< weight at replay time); inputs [x, weight(, bias)].
                      ///< NOT the same convention as OpId::QuantizedLinear's
                      ///< actual dispatch contract — see QuantizedLinearStatic.
    QuantizedLinearStatic, ///< The real dispatch<OpId::QuantizedLinear> contract:
                           ///< inputs [input_int8, weight_int8, bias_f32(, wscale,
                           ///< wzp)], attrs input_scale/weight_scale/output_scale
                           ///< (float) + input_zero_point/weight_zero_point (int).
                           ///< Pre-quantized (static) int8 tensors — distinct from
                           ///< QuantizedLinear above.
    QuantizedConv2d,  ///< INT8 quantized 2D convolution
    Dequantize,       ///< INT8 -> Float32 dequantization
    Quantize,         ///< Float32 -> INT8 quantization

    // Sparse operations
    SparseMatMul,     ///< Sparse-dense matrix multiplication (SpMM) — dense
                      ///< weight retagged sparse by SparsePass; inputs [x, weight(, bias)]
    SparseSpMM,       ///< CSR sparse-dense matmul; inputs [crow, col, values, dense],
                      ///< attrs M/K — the direct OpId::SparseSpMM dispatch convention
                      ///< (distinct from SparseMatMul above)
    DenseToSparse,    ///< Convert dense weight to sparse format

    // Control flow
    If,     ///< Conditional branch: cond → then_branch / else_branch subgraphs
    Loop,   ///< Loop: (max_iter, cond, carried...) → body subgraph → (carried...)

    // Layout and type conversion (inserted by optimization passes)
    LayoutConvert, ///< Convert memory format (e.g., NCHW -> NHWC)
    Cast,          ///< Convert dtype (e.g., Float32 -> Float16)
    ToDevice,      ///< Cross-device transfer (e.g., CPU -> CUDA:0). Interpreter-
                   ///< only: MLIR/IREE compiles to a single HAL target, so a
                   ///< graph containing this node cannot be lowered there and
                   ///< must run through the interpreter or fall back to eager.

    // ── Phase 13 / MVP-1 additions ──
    SiLU,             ///< x * sigmoid(x)
    Where,            ///< Elementwise select: where(cond, a, b)
    Stack,            ///< Cat along a new dim
    Broadcast,        ///< Explicit broadcast (shape-only)
    IndexSelect,      ///< Select along a dim by index tensor
    RMSNorm,          ///< Tenzor dialect op
    GQA,              ///< Grouped-Query Attention — Tenzor dialect op
    RoPE,             ///< Rotary positional embedding — Tenzor dialect op
    Padding,          ///< Constant/reflect/replicate padding
    Interpolate,      ///< Bilinear/nearest spatial resize

    // Comparison (produce Bool) and logical ops. Without these, a data-dependent
    // control-flow predicate (a > b, a == b, logical_and(...)) dispatched an
    // unmapped OpId → graph break → the predicate's output had no producer and
    // was frozen as a trace-time constant, so traced cond()/while_loop() ignored
    // their actual inputs. Mapping them makes the predicate a real graph node.
    Eq,               ///< Elementwise a == b  (Bool)
    Ne,               ///< Elementwise a != b  (Bool)
    Lt,               ///< Elementwise a <  b  (Bool)
    Le,               ///< Elementwise a <= b  (Bool)
    Gt,               ///< Elementwise a >  b  (Bool)
    Ge,               ///< Elementwise a >= b  (Bool)
    LogicalAnd,       ///< Elementwise a && b  (Bool)
    LogicalOr,        ///< Elementwise a || b  (Bool)
    LogicalNot,       ///< Elementwise !a      (Bool)
    // JIT review C2: previously-unmapped ops, now first-class IR nodes so they
    // are CAPTURED and replayed through the JIT (not eager-fallback).
    Var,              ///< Variance reduction
    Std,              ///< Standard-deviation reduction
    Prod,             ///< Product reduction
    LeakyReLU,        ///< max(x, negative_slope*x)
    ELU,              ///< Exponential linear unit
    Mish,             ///< x * tanh(softplus(x))
    Softplus,         ///< (1/beta) * log(1 + exp(beta*x))
    GroupNorm,        ///< Group normalization
    InstanceNorm,     ///< Instance normalization
    Gather,           ///< Gather along dim by index tensor
    Scatter,          ///< Scatter src into input along dim by index tensor
    Flip,             ///< Reverse along dims
    Roll,             ///< Circular shift along dim
    Triu,             ///< Upper triangle (zero below diagonal), shape-preserving
    Tril,             ///< Lower triangle (zero above diagonal), shape-preserving
    Diag,             ///< 1D->2D diagonal matrix construction, or 2D->1D diagonal extraction
    Trace,            ///< Sum of the main diagonal of a 2D matrix -> scalar

    // JIT-R091: appended at the end (not inserted mid-enum) so no existing
    // OpType's ordinal value shifts — see git history/findings.txt for the
    // regression an earlier mid-enum insertion caused.
    OneHot,           ///< Integer class-index -> one-hot encoding (appends a
                      ///< trailing class axis: [N, d1, ..., dk] ->
                      ///< [N, d1, ..., dk, num_classes]); attrs: num_classes (int)

    // JIT-R098: appended at the end (see JIT-R091's comment above on why —
    // ordinal stability for existing entries). Same direct-dispatch CSR
    // convention as SparseSpMM: inputs [crow, col, values, dense/vec/b],
    // attrs as noted per op.
    SparseSpMV,       ///< CSR sparse-dense matvec; inputs [crow, col, values, vec],
                      ///< attrs M/K, output shape (M,)
    SparseAdd,        ///< CSR sparse + dense elementwise add; inputs
                      ///< [crow, col, values, dense], attrs M/K, output shape == dense
    SparseTrsv,       ///< CSR triangular solve, 1D RHS; inputs [crow, col, values, b],
                      ///< attrs N/upper, output shape == b
    SparseTrsm,       ///< CSR triangular solve, 2D RHS; inputs [crow, col, values, B],
                      ///< attrs N/upper, output shape == B

    // JIT-R098 (continued): mapping these general-purpose ops was required
    // to unblock SparseAdd's manually-recorded node above — sparse::add()'s
    // CPU path calls SparseTensor::to_dense(), which itself dispatches
    // these two internally, and (being dispatched, not manually recorded)
    // they were reaching the interceptor unmapped and poisoning the WHOLE
    // trace with a spurious graph break. Mapping them is also a general win
    // for any other traced code using scatter_add()/repeat_interleave()
    // directly. RepeatInterleave's shape rule is intentionally NOT added to
    // infer_types below — its output size can depend on the runtime VALUES
    // of a per-element `repeats` tensor, not just input shapes, so the safe
    // choice is to leave each node's shape at whatever was actually
    // observed during tracing rather than guess at a (possibly wrong)
    // symbolic rule.
    ScatterAdd,       ///< accumulate src into input at index positions along
                      ///< dim; inputs [input, index, src], attrs dim;
                      ///< shape-preserving (== input)
    RepeatInterleave, ///< repeat each element `repeats` times along dim;
                      ///< inputs [input] (attrs dim/num_repeats, scalar mode)
                      ///< or [input, repeats] (num_repeats==-1, tensor mode)

    // JIT-R084: elementwise unary activations whose OpId was unmapped here,
    // so their no-grad-fast-path dispatch<OpId::X>() call graph-broke the
    // whole trace (a "masked landmine," not silent-wrong-answer) — reachable
    // in real deployed inference via nn::Swish (MobileNetV3/EfficientNet-
    // style blocks). Shape/dtype-preserving elementwise ops, mirroring
    // GELU/Mish/Softplus's existing treatment.
    Swish,       ///< x * sigmoid(x); shape/dtype-preserving elementwise
    RReLU,       ///< randomized leaky ReLU; attrs lower/upper (eval-mode
                 ///< fixed slope = (lower+upper)/2); shape/dtype-preserving
    LogSigmoid,  ///< log(sigmoid(x)) = -softplus(-x); shape/dtype-preserving

    // JIT-R064: OpId::Minimum/OpId::IndexCopy were unmapped, so
    // Embedding::renorm_embeddings' non-differentiable max_norm clamp
    // (scale = min(1, max_norm/norm); weight = index_copy(weight, 0, idx,
    // rows * scale)) graph-broke the WHOLE trace on every forward call for
    // any Embedding(max_norm>0) module, deterministically blocking such
    // models from ever JIT-compiling.
    Minimum,    ///< elementwise min(a, b); broadcasts like Add; no autograd
                ///< overload exists anywhere in the codebase (mirrors Fmod)
    IndexCopy,  ///< writes source rows into input at index positions along
                ///< dim; inputs [input, index, source], attrs dim;
                ///< shape-preserving (== input[0])

    // JIT-R100: general-purpose ops needed to unblock the SparseTensor
    // structural-conversion nodes below — their real implementations
    // internally dispatch these on the CPU/OneAPI/Vulkan on-device
    // conversion paths (CUDA/ROCm mostly use opaque native cusparse/
    // rocsparse kernels that bypass dispatch entirely). Same rationale as
    // ScatterAdd/RepeatInterleave under JIT-R098: left unmapped, each would
    // poison the WHOLE trace with a spurious graph break even though the
    // outer Sparse* node's own manual recording (below) would otherwise
    // succeed. Interpreter-replay only (grad_mode unconditionally
    // unsupported), matching ScatterAdd/RepeatInterleave's policy.
    CumSum,     ///< cumulative sum along dim; inputs [x], attrs dim;
                ///< shape-preserving (== input)
    Sort,       ///< sort along dim; inputs [x], attrs dim/descending; TWO
                ///< outputs [sorted_values, indices], shape-preserving
    Nonzero,    ///< positions of non-zero elements; inputs [x]; ONE output;
                ///< data-dependent output size — shape kept at whatever was
                ///< actually observed during tracing (RepeatInterleave policy)
    Bincount,   ///< per-value occurrence counts; inputs [x], attrs minlength;
                ///< ONE output; data-dependent output size (same policy)

    // JIT-R100: SparseTensor structural-conversion ops (from_dense/to_dense/
    // coalesce/transpose/to_csr/to_csc/to_bsr) — manually recorded directly
    // inside the corresponding SparseTensor method, the same way SparseAdd
    // is (JIT-R098), since these methods' real inputs/outputs are raw
    // component tensors (crow/col/values, etc.), not autograd Variables, and
    // are NEVER auto-mapped from a generic dispatch<OpId> call site (so
    // there is deliberately no opid_to_optype case for any of these —
    // OpId::SparseToDense/DenseToSparse's own pre-existing, unrelated
    // dispatch-kernel convention, used only by the JVP rules, is left as-is
    // and continues to safely graph-break under tracing, unchanged).
    // Every input/output tensor list uses this fixed per-layout ordering
    // (an int attr records which layout applies on each side, since
    // transpose()/coalesce() can change layout, e.g. CSC transposes to CSR):
    //   COO=[indices,values]  CSR=[crow,col,values]  CSC=[ccol,row,values]
    //   BSR=[row_ptr,col_ind,values] (+ int_attrs block_h/block_w)
    // Interpreter-replay only (grad_mode unconditionally unsupported,
    // matching SparseAdd/SparseSpMM/SparseTrsv's policy — none of these 7
    // methods are ever called through Variable/autograd, only on raw
    // SparseTensor).
    SparseFromDense,  ///< dense -> sparse; inputs [dense], attrs out_layout
                      ///< (+ out_block_h/out_block_w if BSR); outputs per
                      ///< out_layout's convention above
    SparseToDense,    ///< sparse -> dense; inputs per in_layout convention,
                      ///< attrs in_layout (+ in_block_h/in_block_w if BSR);
                      ///< ONE output [dense]
    SparseCoalesce,   ///< COO dedup+sort (no-op passthrough for non-COO
                      ///< layouts); inputs/outputs per in_layout/out_layout
                      ///< convention (data-dependent nnz when recomputed)
    SparseTranspose,  ///< 2D sparse transpose; inputs per in_layout
                      ///< convention, attrs in_layout; outputs per
                      ///< out_layout convention (NOT always == in_layout:
                      ///< CSC transposes to CSR, BSR to COO)
    SparseToCsr,      ///< inputs per in_layout convention, attrs in_layout;
                      ///< outputs [crow,col,values]
    SparseToCsc,      ///< inputs per in_layout convention, attrs in_layout;
                      ///< outputs [ccol,row,values]
    SparseToBsr,      ///< inputs per in_layout convention, attrs in_layout,
                      ///< target block_h/block_w; outputs [row_ptr,col_ind,values]

    // JIT-R082/R086/R087: object-detection primitives (src/ops/detection.cpp,
    // src/nn/detection/roi_ops.cpp) already dispatch through OpId — mapping
    // them here lets the generic dispatch<OpId> interceptor auto-record them,
    // same OpId-mapping treatment as JIT-R100's CumSum/Sort/Nonzero/Bincount.
    // grad_mode replay unconditionally unsupported for BoxIoU/NMS (both
    // produce index/score tensors with no autograd meaning); ROIAlignForward
    // DOES support differentiable replay (see execute_node) since its real
    // backward (ROIAlignBackward) is a registered, tested autograd Function.
    BoxIoU,           ///< pairwise IoU; inputs [boxes1, boxes2], attrs
                      ///< iou_type (int); ONE output [num_boxes1, num_boxes2]
    NMS,              ///< non-max suppression; inputs [boxes, scores], attrs
                      ///< iou_threshold (float); ONE output, data-dependent
                      ///< size (Nonzero/Bincount policy)
    ROIAlignForward,  ///< RoIAlign pooling; inputs [features, rois], attrs
                      ///< output_size (vec [h,w]), spatial_scale (float),
                      ///< sampling_ratio (int), aligned (bool)

    // JIT-R085: AnchorGenerator::generate has NO tensor inputs at all (a pure
    // function of host-side scalars feat_h/feat_w/stride plus the generator's
    // own sizes_/aspect_ratios_ config) and zero dispatch() calls internally
    // — manually recorded directly inside AnchorGenerator::generate, mirroring
    // JIT-R098/R100's manual-recording pattern. sizes_/aspect_ratios_ travel
    // as small CPU Float32 tensor_attrs (not vec_attrs, since they're floats)
    // so execute_node can reconstruct and re-call generate() exactly.
    AnchorGenerate,   ///< no tensor inputs; attrs feat_h/feat_w/stride (int),
                      ///< tensor_attrs sizes/aspect_ratios (1D Float32); ONE
                      ///< output [total_anchors, 4], always requires_grad=false

    // findings.txt JIT-R132/R133: Norm/linalg family already had a full
    // OpType + execute_node + infer_types/infer_symbolic_types implementation
    // (used by the direct dispatch<OpId::LinalgX> replay path elsewhere) but
    // no opid_to_optype mapping, so tracing any real call unconditionally
    // hard-broke. Sign is the one additional elementwise op needed for
    // slogdet()'s device-dispatch path (sign(det)/log(abs(det))) to be fully
    // traceable now that Det itself is mapped.
    Sign,             ///< elementwise sign(x) in {-1, 0, 1} (complex: x/|x|)

    // findings.txt JIT-R116/R134: previously-unmapped ops whose dispatch call
    // sites are real (GridSample/AffineGrid: src/ops/vision.cpp; FFT:
    // src/ops/fft.cpp; CTCLossForward: src/nn/loss/losses_advanced.cpp;
    // FusedLinearReLU/FusedConv2dReLU/FusedConv2dBnReLU/
    // FusedSoftmaxCrossEntropy/FusedAddReLU: src/ops/fused_ops.cpp /
    // src/ops/fusion_optimizer.cpp), unconditionally hard-breaking any trace
    // that reaches them. Unlike FlashAttention/FusedFFN these are not
    // decomposed at replay time (no simpler equivalent composition exists,
    // or the ops are themselves the atomic primitive) -- execute_node
    // instead re-dispatches the identical OpId with the traced attrs for
    // inference (non-grad) replay, and throws to force an eager fallback for
    // differentiable (grad_mode) replay, mirroring the existing Dropout/
    // BatchNorm2d/linalg-SVD "not wired for differentiable replay" pattern.
    GridSample,          ///< bilinear/nearest/bicubic sampling of input at
                         ///< grid locations; inputs [input, grid], attrs
                         ///< mode/padding_mode/align_corners
    AffineGrid,          ///< builds a sampling grid from an affine theta;
                         ///< inputs [theta], attrs output_size/align_corners
    FFT,                 ///< 1D fast Fourier transform; inputs [input],
                         ///< attrs dim/n/norm
    IFFT,                ///< 1D inverse FFT; inputs [input], attrs dim/n/norm
    RFFT,                ///< 1D real-to-complex FFT; inputs [input], attrs
                         ///< dim/n/norm
    IRFFT,               ///< 1D inverse real FFT (complex-to-real); inputs
                         ///< [input], attrs dim/n/norm
    CTCLossForward,      ///< CTC loss; inputs [log_probs, targets,
                         ///< input_lengths, target_lengths], attrs
                         ///< blank/zero_infinity; outputs
                         ///< [losses_per_sample, raw_grad] (raw_grad only
                         ///< meaningful for the eager custom-autograd path,
                         ///< dropped on inference replay)
    FusedLinearReLU,     ///< relu(linear(input, weight, bias?)); inputs
                         ///< [input, weight(, bias)], attrs has_bias
    FusedConv2dReLU,     ///< relu(conv2d(input, weight, bias?, stride,
                         ///< padding)); inputs [input, weight(, bias)],
                         ///< attrs has_bias/stride/padding
    FusedConv2dBnReLU,   ///< relu(batch_norm(conv2d(...))); inputs [input,
                         ///< weight, conv_bias, bn_gamma, bn_beta,
                         ///< bn_running_mean, bn_running_var], attrs
                         ///< stride/padding/momentum/eps
    FusedSoftmaxCrossEntropy, ///< cross_entropy(logits, targets); inputs
                         ///< [logits, targets], attrs reduction
    FusedAddReLU,        ///< relu(a + b); inputs [a, b]

    // JIT-R110: these LAPACK-backed linalg ops previously bypassed the
    // tracer entirely on CPU (direct LAPACKE calls, no dispatch()) and hard-
    // broke the trace on CUDA/ROCm/Vulkan/OneAPI (real GPU kernels exist,
    // dispatched via try_gpu_dispatch/dispatch(), but had no OpType to map
    // to). Mirrors the already-fixed Det/Inv/Solve/Cholesky/Svd/Qr/Eigh/
    // Eigvalsh/Slogdet pattern.
    SolveTriangular,      ///< triangular solve; inputs [A, B], attrs
                         ///< upper/unitriangular
    LinalgEig,            ///< general (non-symmetric) eigendecomposition;
                         ///< inputs [A]; outputs [eigenvalues_real,
                         ///< eigenvalues_imag, eigenvectors]
    LinalgLU,             ///< LU decomposition with partial pivoting;
                         ///< inputs [A]; outputs [L, U, pivots]
    LinalgLUSolve,        ///< solve via a precomputed LU factorization;
                         ///< inputs [LU_data, pivots, B]
    LinalgHouseholder,    ///< reconstruct Q from Householder reflectors
                         ///< (torch.linalg.householder_product); inputs
                         ///< [input, tau]
    LinalgLDLFactor,      ///< symmetric indefinite LDL^T factorization;
                         ///< inputs [A]; outputs [LD, pivots]
    LinalgLDLSolve,       ///< solve via a precomputed LDL^T factorization;
                         ///< inputs [LD, pivots, B]
    Ormqr,                ///< apply Householder-reflector-represented Q to
                         ///< a matrix; inputs [input, tau, other], attrs
                         ///< left/transpose
    Geqrf                 ///< raw LAPACK QR factorization (factored form +
                         ///< tau, not the explicit Q/R of Qr); inputs
                         ///< [input]; outputs [A_factored, tau]
};

/**
 * @brief Converts OpType to string for debugging and serialization.
 *
 * @param type Operation type
 * @return String representation
 */
auto op_type_to_string(OpType type) -> std::string;

/**
 * @brief Converts string to OpType for deserialization.
 *
 * @param str String representation
 * @return Operation type
 * @throws std::runtime_error if string is not recognized
 */
auto string_to_op_type(const std::string& str) -> OpType;

/**
 * @brief Recorded operation during tracing.
 *
 * Captures all information needed to reconstruct and optimize
 * an operation in the IR graph.
 */
struct TracedOp {
    OpType type;                                     ///< Operation type
    std::vector<std::string> inputs;                 ///< Input tensor IDs
    std::vector<std::string> outputs;                ///< Output tensor IDs
    std::unordered_map<std::string, double> attrs;   ///< Scalar float attributes, stored as
                                                     ///< double so f64 graphs keep full precision
                                                     ///< (e.g. pow exponent, clamp bounds) — JIT-F057
    std::unordered_map<std::string, int64_t> int_attrs;  ///< Int attributes (e.g., dimensions)
    std::unordered_map<std::string, std::vector<int64_t>> vec_attrs;  ///< Vector attributes (e.g., shape)
    std::unordered_map<std::string, bool> bool_attrs;  ///< Boolean attributes (e.g., bias)
    std::unordered_map<std::string, Tensor> tensor_attrs;  ///< Tensor constants (e.g., weights)

    /// Control-flow sidecar: for `If` ops, `outputs` holds then-branch
    /// output tensor IDs and `else_outputs` holds else-branch output
    /// tensor IDs (empty for all other op types). end_trace uses this
    /// to surface the correct per-branch outputs from each subgraph.
    std::vector<std::string> else_outputs;

    /// Control-flow sidecar: for `Loop` ops, the tensor ID produced by the
    /// loop's condition function (`cond_fn`) on the post-body state. The body
    /// subgraph must surface this as its FIRST output so the executor
    /// (graph.cpp Loop case: `body outputs = [cond, carried...]`) can decide
    /// whether to continue iterating, instead of running to `max_iter`.
    /// Empty for all other op types.
    std::string loop_cond_output;

    /**
     * @brief Construct traced operation.
     *
     * @param op_type Type of operation
     * @param input_ids Input tensor identifiers
     * @param output_ids Output tensor identifiers
     */
    TracedOp(OpType op_type, std::vector<std::string> input_ids, std::vector<std::string> output_ids)
        : type(op_type), inputs(std::move(input_ids)), outputs(std::move(output_ids)) {}
};

/**
 * @brief Tensor metadata tracked during tracing.
 *
 * Records shape, dtype, and device information for each tensor
 * in the computation graph.
 */
struct TensorInfo {
    std::vector<int64_t> shape;  ///< Tensor shape
    DType dtype;                 ///< Data type
    Device device;               ///< Device location
    bool is_param{false};        ///< True if this is a model parameter

    TensorInfo() = default;
    TensorInfo(std::vector<int64_t> s, DType dt, Device dev, bool param = false)
        : shape(std::move(s)), dtype(dt), device(dev), is_param(param) {}
};

/**
 * @brief Tracing context that records operations during execution.
 *
 * The Tracer maintains a global state while tracing is active,
 * recording all operations performed on tensors and variables.
 * It builds a complete IR graph that can be optimized and serialized.
 *
 * Thread-local storage ensures thread safety for multi-threaded tracing.
 *
 * @code
 * Tracer tracer;
 * tracer.start_trace();
 *
 * // Operations are automatically recorded
 * Variable output = model(input);
 *
 * auto graph = tracer.end_trace({input}, {output});
 * @endcode
 */
class Tracer {
public:
    /**
     * @brief Construct tracer instance.
     */
    Tracer() = default;

    /**
     * @brief Start recording operations.
     *
     * Activates tracing mode globally. All tensor/variable operations
     * will be recorded until end_trace() is called.
     */
    auto start_trace() -> void;

    /**
     * @brief Stop recording and build IR graph.
     *
     * Finalizes the trace and constructs an optimized IR graph.
     *
     * @param inputs Input tensors/variables that started the trace
     * @param outputs Output tensors/variables produced by the trace
     * @return IR graph representing the computation
     */
    auto end_trace(const std::vector<Variable>& inputs,
                   const std::vector<Variable>& outputs) -> std::shared_ptr<Graph>;

    /**
     * @brief Record an operation during tracing.
     *
     * Called automatically by overloaded operators and functions
     * when tracing is active.
     *
     * @param op Traced operation to record
     */
    auto record_op(TracedOp op) -> void;

    /**
     * @brief Record an in-place mutation as a value-versioned node.
     *
     * In-place ops (add_/sub_/mul_/div_/relu_ …) mutate `target` in place, so
     * its storage — and therefore its tracer fingerprint — is unchanged. If we
     * simply looked the tensor up again, later reads would resolve to the
     * PRE-mutation graph value. Instead this records a functional node
     * `new = op(old, others...)` and remaps the tensor's id to `new` (SSA
     * renaming), so subsequent reads of `target` see the post-op value. Called
     * by the in-place hook installed in TracingGuard; no-op when not tracing.
     *
     * @param op     The in-place OpId (mapped to its functional OpType).
     * @param target The just-mutated tensor.
     * @param others Additional inputs the op consumed.
     * @param attrs  Op attributes (clamp bounds, leaky-relu slope, …).
     * @param pre_snapshot Deep copy of `target` captured before the mutation
     *        (may be null). Stored under the pre-op value id so a captured
     *        constant leaf bakes its PRE-mutation value, not the post-op one.
     */
    auto record_inplace(OpId op, Tensor& target,
                        std::span<const Tensor> others,
                        const OpAttributes& attrs,
                        const Tensor* pre_snapshot = nullptr) -> void;

    /**
     * @brief Register a tensor in the trace.
     *
     * Assigns a unique ID to the tensor and records its metadata.
     *
     * @param tensor Tensor to register
     * @return Unique tensor ID
     */
    auto register_tensor(const Tensor& tensor) -> std::string;

    /**
     * @brief Register a variable in the trace.
     *
     * @param var Variable to register
     * @return Unique tensor ID
     */
    auto register_tensor(const Variable& var) -> std::string;

    /**
     * @brief Force-register a tensor under a fresh unique ID.
     *
     * Unlike `register_tensor`, this does NOT dedup on `data_ptr()` — it
     * always allocates a new ID and stores fresh `TensorInfo` and a
     * (shallow) copy of the tensor. This is the right primitive for
     * recording the OUTPUT of view-creating ops (reshape, transpose,
     * permute, slice, cat) where the output shares `data_ptr()` with
     * an input but has a different logical shape/strides; without it
     * the tracer would alias the output to its input and lose the
     * shape change.
     *
     * @param tensor Tensor to register
     * @return Newly-allocated unique tensor ID
     */
    auto register_new_tensor(const Tensor& tensor) -> std::string;

    /**
     * @brief Whether `tensor`'s logical-view fingerprint is already registered.
     *
     * Cheap map lookup used by the tracing interceptor to decide, BEFORE running
     * an op, whether an input is a tracked value (a potential captured
     * leaf/constant) and therefore worth deep-copying so an in-place mutation can
     * preserve its pre-op value (JIT-F016).
     */
    auto is_registered(const Tensor& tensor) const -> bool;

    /**
     * @brief Overwrite the retained storage for an existing value id.
     *
     * Used by the interceptor's in-place path to store a PRE-mutation snapshot
     * under the mutated input's id, so end_trace's constant-baking sees the
     * pre-op value rather than the post-mutation data (JIT-F016). No-op if `id`
     * is not present.
     */
    auto set_pre_op_snapshot(const std::string& id, const Tensor& snapshot)
        -> void;

    /**
     * @brief Alias a tensor to an already-registered value id.
     *
     * Maps `alias`'s fingerprint to `existing_id` so a later consumer that
     * registers `alias` resolves to the SAME graph value — recording no node and
     * minting no id. Used for transparent value-identity ops (e.g. Contiguous):
     * `contiguous(x)` has identical values to `x` but, when `x` was
     * non-contiguous, a distinct storage/fingerprint, so `register_tensor`
     * cannot dedup it. Aliasing lets the tracer elide the op instead of treating
     * it as an unmappable graph break (which would force an eager fallback).
     *
     * @param alias        The value-identity op's output tensor.
     * @param existing_id  The id the op's input was registered under.
     */
    auto alias_tensor(const Tensor& alias, const std::string& existing_id) -> void;

    /**
     * @brief Declare the trainable parameters the traced function closes over.
     *
     * Records each parameter's storage identity so end_trace() can classify a
     * non-produced op input that reads a parameter as a PARAMETER LEAF (bound
     * to the parameter index) instead of freezing it as an opaque constant.
     * The parameters vector is forwarded onto the built Graph so replay can
     * rebind the live Variables. Must be called AFTER start_trace() (which
     * clear()s any previously declared parameters). No-op with an empty list.
     *
     * @param params Trainable parameter Variables (e.g. nn::Module::parameters()).
     */
    auto set_parameters(std::vector<std::shared_ptr<Variable>> params) -> void;

    /**
     * @brief Declare the non-trainable buffers the traced function closes
     *        over (JIT-R005 fix — mirrors set_parameters() exactly).
     *
     * Records each buffer's storage identity so end_trace() can classify a
     * non-produced op input that reads a buffer as a BUFFER LEAF (bound to
     * the buffer index, rebound to its live value on every replay) instead
     * of freezing it as an opaque constant — the gap that let BatchNorm/
     * InstanceNorm running_mean_/running_var_ go stale across replays of a
     * compiled graph. Must be called AFTER start_trace(). No-op with an
     * empty list.
     *
     * @param buffers Non-trainable buffer Variables (e.g. nn::Module::buffers()).
     */
    auto set_buffers(std::vector<std::shared_ptr<Variable>> buffers) -> void;

    /**
     * @brief Get metadata for a tensor ID.
     *
     * @param tensor_id Tensor identifier
     * @return Tensor metadata
     */
    auto get_tensor_info(const std::string& tensor_id) const -> const TensorInfo&;

    /**
     * @brief Check if tracing is currently active.
     *
     * @return true if operations are being recorded
     */
    auto is_tracing() const -> bool { return tracing_; }

    /**
     * @brief Enable or disable strict mode for the current tracer.
     *
     * When strict mode is ON, any un-traceable operation encountered
     * during a trace (scalar extraction via `.item()`, data-dependent
     * `if`/`while`, any op that isn't representable in the IR) throws
     * a `std::runtime_error` with a pointer to `tenzor::jit::cond` /
     * `while_loop` as the replacement.
     *
     * When OFF (default), un-traceable ops record a warning and
     * increment `graph_break_count()` but tracing continues; the
     * compiled graph bakes in whichever branch was taken and becomes
     * brittle for inputs that would take a different branch.
     *
     * Strict mode defaults to the value of the `TENZOR_JIT_STRICT`
     * environment variable read at `start_trace()` time via the shared
     * jit_strict_mode_enabled() parser (see its doc comment) -- absent/
     * empty/"0"/"false" (case-insensitive) disable it, anything else
     * enables it.
     */
    auto set_strict_mode(bool strict) -> void { strict_mode_ = strict; }

    /// Current strict-mode state (see set_strict_mode()).
    auto is_strict_mode() const -> bool { return strict_mode_; }


    /**
     * @brief Number of graph breaks recorded during the current trace.
     *
     * Useful after `end_trace()` to surface a diagnostic count without
     * inspecting logs. Reset by `start_trace()`/`clear()`.
     */
    [[nodiscard]] auto graph_break_count() const -> int64_t { return graph_break_count_; }

    /**
     * @brief Report a graph-break during tracing.
     *
     * Called by the scalar-extraction hook, by data-dependent control
     * flow, and by any other leaf operation that cannot be cleanly
     * represented in the IR. In strict mode this throws; otherwise it
     * logs to stderr and increments `graph_break_count()`.
     *
     * @param reason Human-readable description of what broke. Used in
     *               the strict-mode exception message and stderr log.
     */
    auto record_graph_break(const std::string& reason) -> void;

    /// Abort the trace unconditionally for an op that has no IR mapping (its
    /// output/mutation would otherwise be silently baked as a constant). Always
    /// throws; only used inside the trace machinery, whose caller falls back to
    /// eager. See the implementation for why this differs from a `.item()` break.
    auto abort_trace_unmappable(const std::string& reason) -> void;

    /**
     * @brief Trace a conditional branch (if/else).
     *
     * Records both branches as subgraphs in the IR. At execution time,
     * the condition tensor determines which branch to evaluate.
     *
     * @param condition Boolean scalar tensor (evaluated at runtime)
     * @param then_fn Function executed when condition is true
     * @param else_fn Function executed when condition is false
     * @param inputs Input variables available to both branches
     * @return Output variables from the selected branch
     */
    auto trace_if(const Tensor& condition,
                  std::function<std::vector<Variable>(const std::vector<Variable>&)> then_fn,
                  std::function<std::vector<Variable>(const std::vector<Variable>&)> else_fn,
                  const std::vector<Variable>& inputs) -> std::vector<Variable>;

    /**
     * @brief Trace a loop with carried state.
     *
     * Records the loop body as a subgraph. The loop executes up to max_iter
     * times, with carried variables passed between iterations.
     *
     * @param max_iter Maximum number of iterations
     * @param cond_fn Function returning bool tensor (continue condition)
     * @param body_fn Function computing one iteration (takes carried, returns updated carried)
     * @param carried Initial carried state variables
     * @return Final carried state after loop completes
     */
    auto trace_loop(int64_t max_iter,
                    std::function<Tensor(const std::vector<Variable>&)> cond_fn,
                    std::function<std::vector<Variable>(const std::vector<Variable>&)> body_fn,
                    const std::vector<Variable>& carried) -> std::vector<Variable>;

    /**
     * @brief Clear all recorded operations and reset state.
     */
    auto clear() -> void;

    /**
     * @brief Get global tracer instance (thread-local).
     *
     * @return Reference to thread-local tracer
     */
    static auto get_instance() -> Tracer&;

private:
    bool tracing_{false};                                       ///< Tracing active flag
    bool strict_mode_{false};                                   ///< Throw on graph breaks
    int64_t graph_break_count_{0};                              ///< Diagnostic counter
    std::vector<TracedOp> ops_;                                 ///< Recorded operations
    std::unordered_map<std::string, TensorInfo> tensor_info_;   ///< Tensor metadata
    /// Logical-view fingerprint (data_ptr + dtype + shape + strides) -> ID.
    /// Keying on data_ptr alone aliased distinct views that share storage but
    /// differ only in strides (e.g. a square-matrix transpose), silently
    /// dropping the view op from the dataflow.
    std::unordered_map<std::string, std::string> tensor_id_map_;
    /// JIT-R148: fingerprints genuinely remapped by record_inplace() (an
    /// actual in-place mutation), tracked SEPARATELY from tensor_id_map_'s
    /// general "current mapping" state. trace_if's assert_no_inplace_on_
    /// shared previously inferred "was this shared tensor mutated in-place"
    /// by diffing tensor_id_map_ before/after a branch -- but
    /// register_new_tensor (used by ANY freshly-computed op output, not just
    /// record_inplace) unconditionally overwrites tensor_id_map_[fingerprint]
    /// too. If an unrelated tensor from before the branch is freed and a
    /// branch's ordinary (non-in-place) op output happens to land at the same
    /// address with matching dtype/shape/strides/device (same fingerprint),
    /// that diff-based check misidentified it as an in-place mutation and
    /// aborted the trace. This set records ONLY fingerprints record_inplace()
    /// itself remapped, so the check can ask the precise question instead of
    /// inferring it indirectly.
    std::unordered_set<std::string> inplace_remapped_fingerprints_;
    /// Phase 6.4: retain the actual Tensor (not just metadata) for
    /// every tensor seen during tracing. end_trace() uses this to
    /// capture module parameters as graph constants that live inside
    /// the Graph and get pre-populated into value_map at forward time.
    std::unordered_map<std::string, Tensor> tensor_storage_;
    /// Declared trainable parameters the traced function closes over. Forwarded
    /// onto the built Graph by end_trace() so replay can rebind live Variables.
    std::vector<std::shared_ptr<Variable>> parameters_;
    /// Parameter storage identity (tensor data pointer) -> parameter index.
    /// end_trace() uses this to classify a non-produced op input as a parameter
    /// leaf when its storage matches a declared parameter.
    std::unordered_map<const void*, size_t> param_storage_index_;
    /// Declared non-trainable buffers the traced function closes over
    /// (JIT-R005 fix). Mirrors parameters_ exactly.
    std::vector<std::shared_ptr<Variable>> buffers_;
    /// Buffer storage identity (tensor data pointer) -> buffer index. Mirrors
    /// param_storage_index_ exactly.
    std::unordered_map<const void*, size_t> buffer_storage_index_;
    int64_t next_tensor_id_{0};                                 ///< Counter for unique IDs

    /**
     * @brief Generate unique tensor ID.
     *
     * @return Unique identifier string
     */
    auto generate_tensor_id() -> std::string;
};

/**
 * @brief RAII guard for tracing scope.
 *
 * Automatically starts tracing on construction and stops on destruction.
 * Ensures tracing is properly cleaned up even if exceptions occur.
 *
 * @code
 * {
 *     TracingGuard guard;
 *     Variable output = model(input);
 *     auto graph = guard.get_graph({input}, {output});
 * }
 * @endcode
 */
class TracingGuard {
public:
    /**
     * @brief Start tracing.
     */
    TracingGuard();

    /**
     * @brief Stop tracing and clean up.
     */
    ~TracingGuard();

    /**
     * @brief Get traced graph.
     *
     * @param inputs Input variables
     * @param outputs Output variables
     * @return IR graph
     */
    auto get_graph(const std::vector<Variable>& inputs,
                   const std::vector<Variable>& outputs) -> std::shared_ptr<Graph>;

    TracingGuard(const TracingGuard&) = delete;
    TracingGuard& operator=(const TracingGuard&) = delete;

private:
    Tracer& tracer_;
    bool interceptor_installed_{false};
};

/**
 * @brief Trace a module's forward pass.
 *
 * Records all operations during module execution and returns an
 * optimized IR graph. This is the main entry point for JIT compilation.
 *
 * @param module Module to trace (must have forward() method)
 * @param dummy_input Example input for tracing
 * @param out_result Optional out-param receiving the REAL output computed
 * while tracing (tracing genuinely executes module->forward(dummy_input),
 * it does not just symbolically record it). Callers that need the actual
 * result for this exact call (e.g. a retrace triggered mid-forward()) should
 * capture it here and reuse it instead of replaying the returned graph a
 * second time on the same input (JIT-R138: replaying after tracing runs the
 * module's computation twice for one logical call).
 * @return Traced and compiled graph
 *
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * Variable dummy = Variable(Tensor({1, 3, 224, 224}, DType::Float32, Device::cpu()));
 * auto traced = trace(model, dummy);
 * traced->save("model.pt");
 * @endcode
 */
auto trace(std::shared_ptr<nn::Module> module,
           const Variable& dummy_input,
           Variable* out_result = nullptr) -> std::shared_ptr<Graph>;

/**
 * @brief Trace a function.
 *
 * Records operations from a callable function.
 *
 * @param func Function to trace
 * @param inputs Input variables
 * @param out_results Optional out-param receiving the REAL outputs computed
 * while tracing (see the module-tracing overload's out_result doc for why).
 * @return Traced graph
 */
auto trace(std::function<std::vector<Variable>(const std::vector<Variable>&)> func,
           const std::vector<Variable>& inputs,
           std::vector<Variable>* out_results = nullptr) -> std::shared_ptr<Graph>;

// Forward declaration
class CompiledModule;

/**
 * @brief Trace a module with a raw Tensor input, returning a CompiledModule.
 *
 * Convenience overload that wraps the input in a Variable and returns
 * a CompiledModule for optimized inference. This is the primary entry
 * point for JIT compilation.
 *
 * @param module Module to trace
 * @param dummy_input Example input tensor for tracing
 * @return Compiled module wrapping the traced graph
 *
 * @code
 * auto model = std::make_shared<MyNetwork>();
 * Tensor input({1, 3, 224, 224}, DType::Float32, Device::cpu());
 * auto traced = trace(model, input);
 * traced->forward(new_input);
 * @endcode
 */
auto trace(std::shared_ptr<nn::Module> module,
           const Tensor& dummy_input) -> std::shared_ptr<CompiledModule>;

/**
 * @brief Centralized TENZOR_JIT_STRICT + per-call strict-mode resolution.
 *
 * Strict mode is on if @p config_strict is true OR the TENZOR_JIT_STRICT env
 * var is set to a value other than absent/empty/"0"/"false" (case-
 * insensitive). Shared by the tracer (trace-time graph-break handling, via
 * Tracer::start_trace()) and the compiler (compile-failure / graph-break /
 * replay-failure fallback handling in compile.cpp) so every JIT strict-mode
 * decision — whether made while tracing or while compiling/replaying —
 * applies the identical rule. Before this was centralized, tracer.cpp had
 * its own separate implementation that treated "false"/"False"/"FALSE" as
 * disabled while compile.cpp's copies treated any non-"0" string (including
 * "false") as enabled — the same env var could mean opposite things
 * depending which stage of the pipeline read it (JIT-R119/JIT-R121).
 */
auto jit_strict_mode_enabled(bool config_strict) -> bool;

} // namespace jit
} // namespace tenzor
