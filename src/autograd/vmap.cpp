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

static std::once_flag init_flag;

void register_batching_rule(const std::string& op_name, BatchingRule rule) {
    batching_rules()[op_name] = std::move(rule);
}

auto has_batching_rule(const std::string& op_name) -> bool {
    return batching_rules().count(op_name) > 0;
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

    // Softmax/LogSoftmax: operates on a specific dim, naturally batch-aware
    // (dim parameter refers to within-sample dimension, batch dim is separate)
    register_batching_rule("SoftmaxBackward", passthrough_rule);
    register_batching_rule("LogSoftmaxBackward", passthrough_rule);

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
    register_batching_rule("LSTMBackward", shape_passthrough);
    register_batching_rule("GRUBackward", shape_passthrough);

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

// Try to detect the operation name from a probe run's grad_fn
static auto detect_op_name(const std::function<Variable(const Variable&)>& func,
                           const Variable& probe) -> std::string {
    try {
        auto result = func(probe);
        if (result.grad_fn()) {
            return result.grad_fn()->name();
        }
    } catch (...) {
        // Probe failed — fall back to loop-and-stack
    }
    return "";
}

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

    // Try to detect the operation and apply a batching rule
    if (!batching_rules().empty()) {
        // Create a probe slice to detect the op
        auto probe_tensor = tenzor::select(input_tensor, batch_dim, 0);
        Variable probe(probe_tensor, batched_input.requires_grad());

        auto op_name = detect_op_name(func, probe);
        if (!op_name.empty()) {
            auto it = batching_rules().find(op_name);
            if (it != batching_rules().end()) {
                return it->second(func, batched_input, batch_dim);
            }
        }
    }

    // Fallback: loop-and-stack
    return vmap_loop_and_stack(func, batched_input, batch_dim);
}

} // namespace tenzor
