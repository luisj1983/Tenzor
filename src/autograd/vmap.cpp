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
    // Normalization ops: batch dim is naturally dim 0
    // BatchNorm, LayerNorm, GroupNorm all expect batch as dim 0
    // ====================================================================
    register_batching_rule("FusedLinearReLUBackward", passthrough_rule);
    register_batching_rule("BmmBackward", passthrough_rule);
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
