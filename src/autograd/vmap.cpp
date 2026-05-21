#include "tenzor/autograd/vmap.hpp"
#include "tenzor/autograd/function.hpp"
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

    // Softmax/LogSoftmax: operates on a specific dim, naturally batch-aware
    // (dim parameter refers to within-sample dimension, batch dim is separate)
    register_batching_rule("SoftmaxBackward", passthrough_rule);
    register_batching_rule("LogSoftmaxBackward", passthrough_rule);
    // Audit A.3: OpId-keyed entries for Softmax/LogSoftmax.
    register_batching_rule(OpId::Softmax, passthrough_rule);
    register_batching_rule(OpId::LogSoftmax, passthrough_rule);

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
        // Move batch_dim to front, apply, move back
        auto perm_in = tenzor::transpose(batched_input.tensor(), 0, batch_dim);
        Variable permuted(perm_in, batched_input.requires_grad());
        auto result = func(permuted);
        auto perm_out = tenzor::transpose(result.tensor(), 0, batch_dim);
        return Variable(perm_out, result.requires_grad());
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
        // Move batch to front, apply, move back
        auto perm_in = tenzor::transpose(batched_input.tensor(), 0, batch_dim);
        Variable permuted(perm_in, batched_input.requires_grad());
        auto result = func(permuted);
        auto perm_out = tenzor::transpose(result.tensor(), 0, batch_dim);
        return Variable(perm_out, result.requires_grad());
    };

    register_batching_rule("ReshapeBackward", shape_passthrough);
    register_batching_rule("TransposeBackward", shape_passthrough);
    register_batching_rule("PermuteBackward", shape_passthrough);
    register_batching_rule("FlattenBackward", shape_passthrough);
    register_batching_rule("SqueezeBackward", shape_passthrough);
    register_batching_rule("UnsqueezeBackward", shape_passthrough);
    register_batching_rule("ExpandBackward", shape_passthrough);

    // ====================================================================
    // Reduction ops: sum, mean, etc.
    // These reduce over a specific dim; batch dim is separate
    // ====================================================================
    register_batching_rule("SumBackward", passthrough_rule);
    register_batching_rule("MeanBackward", passthrough_rule);
    register_batching_rule("ProdBackward", passthrough_rule);
    register_batching_rule("VarBackward", passthrough_rule);
    register_batching_rule("StdBackward", passthrough_rule);
    register_batching_rule("MaxBackward", passthrough_rule);
    register_batching_rule("MinBackward", passthrough_rule);

    // ====================================================================
    // Concatenation/Indexing ops
    // ====================================================================
    register_batching_rule("CatBackward", passthrough_rule);
    register_batching_rule("SliceBackward", passthrough_rule);
    register_batching_rule("GatherBackward", passthrough_rule);
    register_batching_rule("IndexSelectBackward", passthrough_rule);
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
    // TopK/Sort: operate on a specific dim, batch-independent
    // ====================================================================
    register_batching_rule("TopKBackward", passthrough_rule);
    register_batching_rule("SortBackward", passthrough_rule);
    register_batching_rule("ArgsortBackward", passthrough_rule);

    // ====================================================================
    // Scatter operations
    // ====================================================================
    register_batching_rule("ScatterBackward", passthrough_rule);
    register_batching_rule("ScatterAddBackward", passthrough_rule);

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
    // Cumulative ops: operate along a dim, batch-independent
    // ====================================================================
    register_batching_rule("CumSumBackward", passthrough_rule);
    register_batching_rule("CumProdBackward", passthrough_rule);

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
    register_batching_rule("NarrowBackward", passthrough_rule);
    register_batching_rule("IndexBackward", passthrough_rule);
    register_batching_rule("MaskedFillBackward", passthrough_rule);
    register_batching_rule("MaskedSelectBackward", passthrough_rule);
    register_batching_rule("RollBackward", passthrough_rule);

    // ====================================================================
    // Diag/Trace/Triangular ops
    // ====================================================================
    register_batching_rule("DiagBackward", passthrough_rule);
    register_batching_rule("TraceBackward", passthrough_rule);
    register_batching_rule("TriuBackward", passthrough_rule);
    register_batching_rule("TrilBackward", passthrough_rule);

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

// Audit A.3: legacy helpers detect_op_name / detect_op_id collapsed
// into the inline probe-and-dispatch logic inside `vmap()` below
// (which now reads both the OpId *and* the name from a single probe
// run, avoiding the double-probe of the previous design).

// Loop-and-stack fallback implementation
static auto vmap_loop_and_stack(const std::function<Variable(const Variable&)>& func,
                                const Variable& batched_input,
                                int64_t batch_dim) -> Variable {
    auto input_tensor = batched_input.tensor();
    auto shape = input_tensor.shape();
    int64_t batch_size = shape[batch_dim];

    std::vector<Tensor> results;
    results.reserve(batch_size);

    for (int64_t i = 0; i < batch_size; ++i) {
        auto slice = tenzor::select(input_tensor, batch_dim, i);
        Variable slice_var(slice, batched_input.requires_grad());
        auto output = func(slice_var);
        results.push_back(output.tensor());
    }

    auto stacked = tenzor::stack(std::span<const Tensor>(results), batch_dim);
    return Variable(stacked, batched_input.requires_grad());
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
