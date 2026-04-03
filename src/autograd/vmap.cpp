#include "tenzor/autograd/vmap.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/transform.hpp"
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

    // Register passthrough for all element-wise activations and math ops
    for (const auto& name : {
        "ReLUBackward", "SigmoidBackward", "TanhBackward",
        "GeluBackward", "AddBackward", "SubBackward",
        "MulBackward", "DivBackward", "ExpBackward",
        "LogBackward", "SqrtBackward", "AbsBackward",
        "NegBackward", "SinBackward", "CosBackward"
    }) {
        register_batching_rule(name, passthrough_rule);
    }

    // Softmax: operates on a specific dim, naturally batch-aware
    register_batching_rule("SoftmaxBackward", passthrough_rule);
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
