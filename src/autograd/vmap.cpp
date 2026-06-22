#include "tenzor/autograd/vmap.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include <vector>
#include <mutex>

namespace tenzor {

// Global registry of batching rules
static std::unordered_map<std::string, BatchingRule>& batching_rules() {
    static std::unordered_map<std::string, BatchingRule> rules;
    return rules;
}

// Audit A.3: OpId-keyed registry. Looked up before the string-keyed
// registry, so an opted-in Function (op_id() != Unknown) goes through
// here. The string registry remains for the long-tail Function
// subclasses that haven't yet opted in to op_id().
static std::unordered_map<OpId, BatchingRule>& batching_rules_by_opid() {
    static std::unordered_map<OpId, BatchingRule> rules;
    return rules;
}

static std::once_flag init_flag;

void register_batching_rule(const std::string& op_name, BatchingRule rule) {
    batching_rules()[op_name] = std::move(rule);
}

auto has_batching_rule(const std::string& op_name) -> bool {
    return batching_rules().count(op_name) > 0;
}

void register_batching_rule(OpId op_id, BatchingRule rule) {
    // Refuse OpId::Unknown to surface the caller's bug instead of
    // silently installing a catch-all rule on the sentinel.
    if (op_id == OpId::Unknown) {
        throw std::runtime_error(
            "register_batching_rule: OpId::Unknown is reserved as the "
            "default sentinel and cannot have a rule registered against "
            "it. Use the string-name overload for un-opted-in classes.");
    }
    batching_rules_by_opid()[op_id] = std::move(rule);
}

auto has_batching_rule(OpId op_id) -> bool {
    return batching_rules_by_opid().count(op_id) > 0;
}

// Forward decl: defined below; used as the fallback path when a
// dim-aware passthrough rule sees no dim attribute (e.g. full reduction)
// or otherwise can't safely shift.
static auto vmap_loop_and_stack(const std::function<Variable(const Variable&)>& func,
                                const Variable& batched_input,
                                int64_t batch_dim) -> Variable;

auto dim_shifted_passthrough(AttrKey dim_attr_key) -> BatchingRule {
    // Audit J.1. The legacy passthrough_rule was wrong for any op that
    // takes a `dim` argument: with vmap prepending a batch axis, the
    // user's `dim` indexes the *unbatched* view, but we hand the kernel
    // the full batched tensor. Without a shift, `sum(x, dim=0)` over a
    // (B, N) tensor reduces over the batch axis instead of N.
    //
    // The rule below probes the user function on a single slice to
    // recover the forward OpId and saved OpAttributes, normalises the
    // saved dim against the *unbatched* ndim, shifts it past batch_dim,
    // and dispatches the op directly with the rebuilt attributes on the
    // full batched input.
    return [dim_attr_key](const std::function<Variable(const Variable&)>& func,
                          const Variable& batched_input,
                          int64_t batch_dim) -> Variable {
        (void)dim_attr_key;
        // Audit (graph-severing fix): the previous fast path re-ran the op
        // via the raw `dispatch()` (reading the saved dim attribute, shifting
        // it past the batch axis, redispatching) and rewrapped the result as
        // `Variable(outputs[0], requires_grad)` with NO grad_fn. That severed
        // autograd for every dim-carrying op routed through this rule (Sum,
        // Mean, Prod, Var, Std, Max, Min, Softmax, LogSoftmax, TopK, Sort,
        // ArgSort, CumSum, CumProd, Roll): backward through a vmap of these
        // yielded zero/absent input grads and lost second-order graphs under
        // create_graph. We instead use vmap_loop_and_stack, which calls the
        // user's `func` per slice (so the user's own `dim` already indexes the
        // correct unbatched axis) and stitches the per-slice results with
        // Variable-level autograd::unsqueeze + autograd::cat — preserving the
        // grad_fn chain at the cost of B sequential dispatches.
        //
        // Audit (probe removal): both branches of the former probe (grad_fn
        // present or absent) returned vmap_loop_and_stack, so probing via an
        // extra `func(slice0)` never changed the path — it just ran the user
        // function one redundant time (and the vmap dispatcher already probed
        // once upstream). Drop the probe and call loop-and-stack directly.
        return vmap_loop_and_stack(func, batched_input, batch_dim);
    };
}

void init_builtin_batching_rules() {
    // Element-wise operations: just pass the batched tensor through directly.
    // These ops naturally vectorize over all dimensions.
    auto passthrough_rule = [](const std::function<Variable(const Variable&)>& func,
                               const Variable& batched_input,
                               int64_t /*batch_dim*/) -> Variable {
        return func(batched_input);
    };

    // Register passthrough for all element-wise activations and math ops.
    // These ops apply independently to each element, so batching is trivial.
    for (const auto& name : {
        // Core arithmetic
        "AddBackward", "SubBackward", "MulBackward", "DivBackward",
        "NegBackward",
        // Activations
        "ReLUBackward", "SigmoidBackward", "TanhBackward",
        "GeluBackward", "EluBackward", "SeluBackward",
        "MishBackward", "LeakyReluBackward", "SoftplusBackward",
        // Math functions
        "ExpBackward", "LogBackward", "SqrtBackward",
        "AbsBackward", "SinBackward", "CosBackward",
        "TanBackward", "AsinBackward", "AcosBackward", "AtanBackward",
        "SinhBackward", "CoshBackward",
        "Log2Backward", "Log10Backward", "Log1pBackward",
        "Exp2Backward", "Expm1Backward",
        "ReciprocalBackward", "PowBackward", "ClampBackward",
        "ErfBackward", "ErfcBackward",
        // Special math
        "GammaBackward", "LgammaBackward", "DigammaBackward",
        "ErfInvBackward", "BesselI0Backward", "BesselI1Backward",
        "SincBackward"
    }) {
        register_batching_rule(name, passthrough_rule);
    }

    // Audit A.3: also register the OpId-keyed passthrough for the opted-in
    // arithmetic + activation + math Functions (see A.2 commits). The vmap
    // dispatch path tries the OpId registry first, so these are the
    // primary hit-path entries for the corresponding Backward classes.
    for (OpId op : {
        OpId::Add, OpId::Sub, OpId::Mul, OpId::Div, OpId::Neg,
        OpId::Gelu, OpId::Elu, OpId::Selu, OpId::Mish, OpId::Softplus,
        OpId::Exp, OpId::Log, OpId::Sqrt, OpId::Abs,
        OpId::Sin, OpId::Cos, OpId::Tan,
        OpId::Asin, OpId::Acos, OpId::Atan,
        OpId::Sinh, OpId::Cosh,
        OpId::Log2, OpId::Log10, OpId::Log1p,
        OpId::Exp2, OpId::Expm1,
        OpId::Reciprocal, OpId::Pow,
        OpId::Erf, OpId::Lgamma, OpId::Digamma,
        OpId::Conj, OpId::Real, OpId::Imag,
    }) {
        register_batching_rule(op, passthrough_rule);
    }

    // Softmax/LogSoftmax: operates on a specific dim, dispatch via the
    // J.1 dim_shifted_passthrough so the user's `dim` shifts past the
    // batch axis (raw passthrough would reduce/normalise across the
    // batch dim instead — R.4).
    auto softmax_dim_rule = dim_shifted_passthrough(AttrKey::Dim);
    register_batching_rule("SoftmaxBackward", softmax_dim_rule);
    register_batching_rule("LogSoftmaxBackward", softmax_dim_rule);
    register_batching_rule(OpId::Softmax, softmax_dim_rule);
    register_batching_rule(OpId::LogSoftmax, softmax_dim_rule);

    // ====================================================================
    // MatMul: promote to BMM when batch_dim == 0
    // ====================================================================
    register_batching_rule("MatMulBackward", [](
        const std::function<Variable(const Variable&)>& func,
        const Variable& batched_input,
        int64_t batch_dim) -> Variable {
        if (batch_dim == 0) {
            // Already batch-first: matmul naturally handles leading batch dims
            return func(batched_input);
        }
        // Move batch_dim to front, apply, move back. BOTH transposes must use
        // the Variable-level autograd::transpose so the chain stays intact: the
        // leading move on the raw tensor (`Variable(transpose(input.tensor()),
        // requires_grad)`) produced a fresh leaf with no grad_fn back to
        // batched_input, so backward accumulated into a discarded accumulator
        // and batched_input.grad() stayed empty (audit-5 X.2 follow-up).
        Variable permuted = tenzor::transpose(batched_input, 0, batch_dim);
        auto result = func(permuted);
        return tenzor::transpose(result, 0, batch_dim);
    });

    // ====================================================================
    // Linear: batch dim is naturally dim 0 (batch x features)
    // ====================================================================
    register_batching_rule("LinearBackward", passthrough_rule);

    // ====================================================================
    // Shape ops: reshape, transpose, permute, squeeze, unsqueeze, flatten
    // These need dim adjustment when batch_dim affects the target dim
    // For batch_dim == 0, passthrough works because shape ops preserve
    // leading dimensions
    // ====================================================================
    auto shape_passthrough = [](
        const std::function<Variable(const Variable&)>& func,
        const Variable& batched_input,
        int64_t batch_dim) -> Variable {
        if (batch_dim == 0) {
            return func(batched_input);
        }
        // Move batch to front, apply, move back. BOTH transposes must use the
        // Variable-level autograd::transpose so the chain stays intact across
        // ~25 shape ops (Conv1/2/3d, BN, LN, GN, IN, attention variants, pool
        // variants, RNN/LSTM/GRU cells, padding variants). Doing the leading
        // move on the raw tensor and rewrapping as a fresh leaf severed the
        // chain back to batched_input, so backward never reached it
        // (audit-5 X.2 follow-up).
        Variable permuted = tenzor::transpose(batched_input, 0, batch_dim);
        auto result = func(permuted);
        return tenzor::transpose(result, 0, batch_dim);
    };

    register_batching_rule("ReshapeBackward", shape_passthrough);
    register_batching_rule("TransposeBackward", shape_passthrough);
    register_batching_rule("PermuteBackward", shape_passthrough);
    register_batching_rule("FlattenBackward", shape_passthrough);
    register_batching_rule("SqueezeBackward", shape_passthrough);
    register_batching_rule("UnsqueezeBackward", shape_passthrough);
    register_batching_rule("ExpandBackward", shape_passthrough);

    // ====================================================================
    // Reduction ops: sum, mean, prod, var, std, max, min
    // Audit J.1: these all carry a `dim` attribute that must shift past
    // the batch axis. dim_shifted_passthrough does the read-shift-redispatch
    // dance using saved_attributes() (wired in A.4). When the saved attrs
    // lack a dim entry (full reduction), the rule falls back to loop-and-
    // stack so per-sample full reductions still work.
    // ====================================================================
    auto reduction_dim_rule = dim_shifted_passthrough(AttrKey::Dim);
    register_batching_rule("SumBackward",  reduction_dim_rule);
    register_batching_rule("MeanBackward", reduction_dim_rule);
    register_batching_rule("ProdBackward", reduction_dim_rule);
    register_batching_rule("VarBackward",  reduction_dim_rule);
    register_batching_rule("StdBackward",  reduction_dim_rule);
    register_batching_rule("MaxBackward",  reduction_dim_rule);
    register_batching_rule("MinBackward",  reduction_dim_rule);
    // OpId-keyed entries — these are the primary hit path now that the
    // reduction Function subclasses opt in to op_id() (audit A.2).
    register_batching_rule(OpId::Sum,  reduction_dim_rule);
    register_batching_rule(OpId::Mean, reduction_dim_rule);
    register_batching_rule(OpId::Prod, reduction_dim_rule);
    register_batching_rule(OpId::Var,  reduction_dim_rule);
    register_batching_rule(OpId::Std,  reduction_dim_rule);
    register_batching_rule(OpId::Max,  reduction_dim_rule);
    register_batching_rule(OpId::Min,  reduction_dim_rule);

    // ====================================================================
    // Concatenation/Indexing ops
    // ====================================================================
    register_batching_rule("CatBackward", passthrough_rule);
    register_batching_rule("SliceBackward", passthrough_rule);
    // R.4: Gather / IndexSelect carry a `dim` attr AND an index tensor that
    // the user captures in the closure. The captured index is shaped for
    // the *unbatched* input; broadcasting it across the batch axis would
    // need extra plumbing not exposed by saved_attributes alone, so we use
    // the loop-and-stack fallback to call func per slice — correct (each
    // slice sees its own index) at the cost of B sequential dispatches.
    auto multi_input_dim_loop_rule = [](
        const std::function<Variable(const Variable&)>& func,
        const Variable& batched_input,
        int64_t batch_dim) -> Variable {
        return vmap_loop_and_stack(func, batched_input, batch_dim);
    };
    register_batching_rule("GatherBackward",      multi_input_dim_loop_rule);
    register_batching_rule("IndexSelectBackward", multi_input_dim_loop_rule);
    register_batching_rule(OpId::Gather,      multi_input_dim_loop_rule);
    register_batching_rule(OpId::IndexSelect, multi_input_dim_loop_rule);
    register_batching_rule("WhereBackward", passthrough_rule);

    // ====================================================================
    // Fused ops
    // ====================================================================
    register_batching_rule("FusedLinearReLUBackward", passthrough_rule);
    register_batching_rule("BmmBackward", passthrough_rule);

    // ====================================================================
    // Convolution ops: batch dim is naturally dim 0 (N in NCHW/NCDHW)
    // For batch_dim != 0, move batch to front then back
    // ====================================================================
    register_batching_rule("Conv1dBackward", shape_passthrough);
    register_batching_rule("Conv2dBackward", shape_passthrough);
    register_batching_rule("Conv3dBackward", shape_passthrough);
    register_batching_rule("ConvTranspose1dBackward", shape_passthrough);
    register_batching_rule("ConvTranspose2dBackward", shape_passthrough);
    register_batching_rule("ConvTranspose3dBackward", shape_passthrough);
    register_batching_rule("DepthwiseConv2dBackward", shape_passthrough);

    // ====================================================================
    // Normalization ops: batch dim is naturally dim 0
    // BatchNorm, LayerNorm, GroupNorm, InstanceNorm, RMSNorm all expect
    // batch as dim 0
    // ====================================================================
    register_batching_rule("BatchNorm1dBackward", shape_passthrough);
    register_batching_rule("BatchNorm2dBackward", shape_passthrough);
    register_batching_rule("BatchNorm3dBackward", shape_passthrough);
    register_batching_rule("LayerNormBackward", shape_passthrough);
    register_batching_rule("GroupNormBackward", shape_passthrough);
    register_batching_rule("InstanceNorm1dBackward", shape_passthrough);
    register_batching_rule("InstanceNorm2dBackward", shape_passthrough);
    register_batching_rule("InstanceNorm3dBackward", shape_passthrough);
    register_batching_rule("RMSNormBackward", shape_passthrough);

    // ====================================================================
    // Regularization: element-wise masking, trivially batchable
    // ====================================================================
    register_batching_rule("DropoutBackward", passthrough_rule);
    register_batching_rule("Dropout2dBackward", passthrough_rule);
    register_batching_rule("AlphaDropoutBackward", passthrough_rule);
    register_batching_rule("DropPathBackward", passthrough_rule);

    // ====================================================================
    // Embedding: lookup is per-element on the index tensor
    // ====================================================================
    register_batching_rule("EmbeddingBackward", passthrough_rule);

    // ====================================================================
    // Attention: batch dim is naturally dim 0 (batch x heads x seq x dim)
    // ====================================================================
    register_batching_rule("MultiheadAttentionBackward", shape_passthrough);
    register_batching_rule("FusedAttentionBackward", shape_passthrough);
    register_batching_rule("FlashAttentionBackward", shape_passthrough);
    register_batching_rule("FlexAttentionBackward", shape_passthrough);
    register_batching_rule("NestedAttentionBackward", shape_passthrough);

    // ====================================================================
    // TopK / Sort / ArgSort: operate on a specific dim.
    // Audit J.1: same fix as the reductions above — the dim recorded by
    // the user must be shifted past the batch axis. TopK and Sort are
    // multi-output (values, indices); the user's `func` returns one of
    // those branches and the rule picks output slot 0 (values branch).
    // ArgSort is non-differentiable so its grad_fn never appears in a
    // probe; the registration is harmless and the loop-and-stack path
    // continues to cover argsort-as-tensor-op callers.
    // ====================================================================
    register_batching_rule("TopKBackward",    reduction_dim_rule);
    register_batching_rule("SortBackward",    reduction_dim_rule);
    register_batching_rule("ArgSortBackward", reduction_dim_rule);
    register_batching_rule(OpId::TopK,    reduction_dim_rule);
    register_batching_rule(OpId::Sort,    reduction_dim_rule);
    register_batching_rule(OpId::ArgSort, reduction_dim_rule);

    // ====================================================================
    // Scatter operations.
    // R.4: Scatter / ScatterAdd carry a `dim` attr plus index + src tensors
    // captured in the user's closure (and src is *not* saved on the
    // backward, only the index). Like Gather/IndexSelect we fall back to
    // loop-and-stack so each slice sees its own captured index/src.
    // ====================================================================
    register_batching_rule("ScatterBackward",    multi_input_dim_loop_rule);
    register_batching_rule("ScatterAddBackward", multi_input_dim_loop_rule);
    register_batching_rule(OpId::Scatter,    multi_input_dim_loop_rule);
    register_batching_rule(OpId::ScatterAdd, multi_input_dim_loop_rule);

    // ====================================================================
    // Pooling ops: batch dim is naturally dim 0 (NCHW)
    // ====================================================================
    register_batching_rule("MaxPool2dBackward", shape_passthrough);
    register_batching_rule("AvgPool2dBackward", shape_passthrough);
    register_batching_rule("AdaptiveAvgPool2dBackward", shape_passthrough);
    register_batching_rule("AdaptiveMaxPool2dBackward", shape_passthrough);

    // ====================================================================
    // RNN ops: batch dim is dim 0 (batch_first) or dim 1
    // Use shape_passthrough to handle both conventions
    // ====================================================================
    register_batching_rule("RNNCellBackward", shape_passthrough);
    register_batching_rule("LSTMCellBackward", shape_passthrough);
    register_batching_rule("GRUCellBackward", shape_passthrough);

    // ====================================================================
    // Cumulative ops: operate along a dim. Raw passthrough cumulates across
    // the batch axis when the user passes a positive within-sample dim; the
    // dim_shifted_passthrough rule (J.1, R.4) reads AttrKey::Dim from
    // saved_attributes() and shifts it past the batch axis.
    // ====================================================================
    auto cumulative_dim_rule = dim_shifted_passthrough(AttrKey::Dim);
    register_batching_rule("CumSumBackward", cumulative_dim_rule);
    register_batching_rule("CumProdBackward", cumulative_dim_rule);
    register_batching_rule(OpId::CumSum, cumulative_dim_rule);
    register_batching_rule(OpId::CumProd, cumulative_dim_rule);

    // ====================================================================
    // Padding ops: batch dim is naturally dim 0
    // ====================================================================
    register_batching_rule("ConstantPad2dBackward", shape_passthrough);
    register_batching_rule("ReflectionPad2dBackward", shape_passthrough);
    register_batching_rule("ReplicationPad2dBackward", shape_passthrough);

    // ====================================================================
    // Loss functions: operate element-wise or along a dim
    // ====================================================================
    register_batching_rule("MSELossBackward", passthrough_rule);
    register_batching_rule("L1LossBackward", passthrough_rule);
    register_batching_rule("CrossEntropyLossBackward", passthrough_rule);
    register_batching_rule("BCELossBackward", passthrough_rule);
    register_batching_rule("NLLLossBackward", passthrough_rule);

    // ====================================================================
    // Linalg ops: batch dim is naturally the leading dimension
    // ====================================================================
    register_batching_rule("DetBackward", passthrough_rule);
    register_batching_rule("InvBackward", passthrough_rule);
    register_batching_rule("SolveBackward", passthrough_rule);
    register_batching_rule("CholeskyBackward", passthrough_rule);
    register_batching_rule("SvdBackward", passthrough_rule);
    register_batching_rule("QrBackward", passthrough_rule);

    // ====================================================================
    // FFT ops: operate on specific dim, batch-independent
    // ====================================================================
    register_batching_rule("FFTBackward", passthrough_rule);
    register_batching_rule("IFFTBackward", passthrough_rule);
    register_batching_rule("RFFTBackward", passthrough_rule);
    register_batching_rule("IRFFTBackward", passthrough_rule);

    // ====================================================================
    // Additional indexing ops
    // ====================================================================
    // V.9: Narrow/Roll/Diag/Triu/Tril Backwards carry a per-axis attribute
    // (dim, shift, diagonal) saved in their constructor. The old
    // `passthrough_rule` ignored these, so `vmap` over `batch_dim != 0`
    // applied the op along the unbatched-rank axis — silently wrong (it
    // would re-narrow/re-roll along the batch axis).
    //
    // Roll has a true `dim` attribute that shifts past the batch axis;
    // dim_shifted_passthrough(AttrKey::Dim) does the read-shift-redispatch
    // for it (the saved Shift attribute carries through unchanged).
    //
    // Narrow has no forward OpId at the dispatch layer (it decomposes to
    // slice on the autograd side); fall back to the loop-and-stack form
    // (V.1 fix made that path graph-preserving), which is correct at the
    // cost of B per-slice dispatches.
    //
    // Diag / Triu / Tril operate on the last two axes; for batch_dim
    // below ndim-2 (the common vmap-over-batch case) a plain
    // re-dispatch with the unchanged Diagonal attribute on the batched
    // tensor gives correct per-slice results. For batch_dim >= ndim-2
    // the loop fallback is the safe path.
    auto narrow_rule = [](
        const std::function<Variable(const Variable&)>& func,
        const Variable& batched_input,
        int64_t batch_dim) -> Variable {
        return vmap_loop_and_stack(func, batched_input, batch_dim);
    };
    register_batching_rule("NarrowBackward", narrow_rule);
    register_batching_rule("IndexBackward", passthrough_rule);
    register_batching_rule("MaskedFillBackward", passthrough_rule);
    register_batching_rule("MaskedSelectBackward", passthrough_rule);
    auto roll_dim_rule = dim_shifted_passthrough(AttrKey::Dim);
    register_batching_rule("RollBackward", roll_dim_rule);
    register_batching_rule(OpId::Roll, roll_dim_rule);

    // ====================================================================
    // Diag/Trace/Triangular ops
    // ====================================================================
    // Diag/Triu/Tril operate on the trailing two axes; for batch_dim
    // outside those axes, loop-and-stack is the conservative choice
    // (correct for arbitrary batch_dim, no axis-shift bookkeeping).
    auto last2_dim_loop_rule = [](
        const std::function<Variable(const Variable&)>& func,
        const Variable& batched_input,
        int64_t batch_dim) -> Variable {
        return vmap_loop_and_stack(func, batched_input, batch_dim);
    };
    register_batching_rule("DiagBackward", last2_dim_loop_rule);
    register_batching_rule("TraceBackward", last2_dim_loop_rule);
    register_batching_rule("TriuBackward", last2_dim_loop_rule);
    register_batching_rule("TrilBackward", last2_dim_loop_rule);
    register_batching_rule(OpId::Diag, last2_dim_loop_rule);
    register_batching_rule(OpId::Triu, last2_dim_loop_rule);
    register_batching_rule(OpId::Tril, last2_dim_loop_rule);
    register_batching_rule(OpId::Trace, last2_dim_loop_rule);

    // ====================================================================
    // Sparse ops
    // ====================================================================
    register_batching_rule("SpMMBackward", shape_passthrough);
    register_batching_rule("SpMVBackward", shape_passthrough);
    register_batching_rule("SparseAddBackward", passthrough_rule);

    // ====================================================================
    // Upsample/Interpolation
    // ====================================================================
    register_batching_rule("UpsampleBilinearBackward", shape_passthrough);
    register_batching_rule("UpsampleNearestBackward", shape_passthrough);
}

// Install the built-in batching rules eagerly at load time, before any user
// code runs. Previously init ran lazily on the first vmap() call; a caller
// that registered a custom rule (e.g. an OpId::Mul override) *before* its
// first vmap had that rule silently clobbered when the deferred init then
// re-registered the built-in passthrough for the same key. Running init at
// static-init time guarantees user registrations always take precedence,
// since they happen after main() starts. vmap()'s own call_once becomes a
// no-op once the flag is consumed here.
namespace {
struct BuiltinBatchingRulesInitializer {
    BuiltinBatchingRulesInitializer() {
        std::call_once(init_flag, init_builtin_batching_rules);
    }
} g_builtin_batching_rules_initializer;
} // namespace

// Audit A.3: legacy helpers detect_op_name / detect_op_id collapsed
// into the inline probe-and-dispatch logic inside `vmap()` below
// (which now reads both the OpId *and* the name from a single probe
// run, avoiding the double-probe of the previous design).

// Loop-and-stack fallback implementation
//
// V.1: build the result via Variable-level cat over unsqueezed per-slice
// outputs. The previous implementation captured `output.tensor()` and
// stacked raw tensors, so the returned Variable had no `grad_fn` — under
// `vmap` inside `create_graph=true` this silently discarded second-order
// info. Using `autograd::unsqueeze` + `autograd::cat` keeps the graph
// intact: backward through the cat scatters gradients back to each
// per-slice Variable, whose grad_fn chains back into the user's `func`.
static auto vmap_loop_and_stack(const std::function<Variable(const Variable&)>& func,
                                const Variable& batched_input,
                                int64_t batch_dim) -> Variable {
    auto input_tensor = batched_input.tensor();
    auto shape = input_tensor.shape();
    int64_t batch_size = shape[batch_dim];

    std::vector<Variable> per_slice_outputs;
    per_slice_outputs.reserve(batch_size);

    // The insertion axis must be valid for the per-slice OUTPUT rank, which can
    // be smaller than the input rank for rank-reducing funcs (e.g. a full
    // reduction to a scalar). `unsqueeze(output, batch_dim)` throws when
    // batch_dim >= output.ndim()+1, so clamp the insertion axis to the output
    // rank. batch_dim is interpreted against the input rank; for rank-dropping
    // funcs we stack along the highest valid output axis instead. All slices
    // share the same output rank, so a single clamped axis is consistent for
    // both the unsqueeze and the cat.
    int64_t insert_axis = batch_dim;
    for (int64_t i = 0; i < batch_size; ++i) {
        // Build the per-slice input through DIFFERENTIABLE ops (narrow + squeeze)
        // so the backward edge reaches batched_input. Raw tenzor::select made
        // slice_var a fresh leaf with no grad_fn, so backward through a vmapped
        // function never propagated gradients to the batched input.
        Variable slice_var = tenzor::squeeze(
            tenzor::narrow(batched_input, batch_dim, i, 1), batch_dim);
        auto output = func(slice_var);
        if (i == 0) {
            insert_axis = std::min<int64_t>(batch_dim, output.tensor().ndim());
        }
        // Add a unit-size batch axis at the (clamped) position so a single
        // `cat` along that axis reconstructs the batched layout that the
        // raw `tenzor::stack` would have produced.
        per_slice_outputs.push_back(tenzor::unsqueeze(output, insert_axis));
    }

    return tenzor::cat(per_slice_outputs, insert_axis);
}

auto vmap(std::function<Variable(const Variable&)> func,
          const Variable& batched_input,
          int64_t batch_dim) -> Variable {
    auto input_tensor = batched_input.tensor();
    auto shape = input_tensor.shape();
    int64_t ndim = static_cast<int64_t>(shape.size());

    // Normalize negative batch_dim
    if (batch_dim < 0) {
        batch_dim += ndim;
    }

    // Initialize built-in rules on first call
    std::call_once(init_flag, init_builtin_batching_rules);

    // Audit A.3: OpId-first lookup, name-string fallback. The Function
    // base class now exposes a canonical forward OpId via op_id() (audit
    // A.2), so we can find the batching rule without going through the
    // class-name string. For un-opted-in Functions, op_id() returns
    // OpId::Unknown and we fall back to the name-string registry — the
    // existing path that's been working until the migration completes.
    if (!batching_rules().empty() || !batching_rules_by_opid().empty()) {
        // Create a probe slice to detect the op
        auto probe_tensor = tenzor::select(input_tensor, batch_dim, 0);
        Variable probe(probe_tensor, batched_input.requires_grad());

        // Run the probe once and inspect both the OpId and the name. We
        // duplicate the func() call instead of caching the Variable
        // because detect_op_id and detect_op_name each run their own
        // try/catch path; merging the probe into a single call here also
        // avoids running the probe twice when the OpId path hits.
        auto probe_result = func(probe);
        auto grad_fn = probe_result.grad_fn();

        if (grad_fn) {
            OpId probed_op_id = grad_fn->op_id();
            if (probed_op_id != OpId::Unknown) {
                auto it = batching_rules_by_opid().find(probed_op_id);
                if (it != batching_rules_by_opid().end()) {
                    return it->second(func, batched_input, batch_dim);
                }
            }
            // Fall back to the name-string registry for un-opted-in
            // Function subclasses or OpIds without an OpId-keyed rule yet.
            auto name = grad_fn->name();
            if (!name.empty()) {
                auto it = batching_rules().find(name);
                if (it != batching_rules().end()) {
                    return it->second(func, batched_input, batch_dim);
                }
            }
        }
    }

    // Fallback: loop-and-stack
    return vmap_loop_and_stack(func, batched_input, batch_dim);
}

} // namespace tenzor
