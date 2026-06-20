#include "tenzor/nn/layers/lazy_linear.hpp"
#include "tenzor/nn/init.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/utils/logging.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

// Namespace alias for autograd operations (matches linear.cpp pattern)
namespace autograd = tenzor;


// Helper function to compute linear using matmul (fallback for backends without fused linear)
static auto linear_via_matmul(const Variable& input, const Variable& weight,
                               const Variable* bias) -> Variable {
    auto weight_t = autograd::permute(weight, {1, 0});
    auto output = autograd::matmul(input, weight_t);

    if (bias) {
        output = output + *bias;
    }

    return output;
}

// Check if fused linear kernel is available for this backend
static bool has_fused_linear_kernel(Device device) {
    return is_op_supported(OpId::Linear, device.type);
}

LazyLinear::LazyLinear(int64_t out_features, bool bias)
    : out_features_(out_features), has_bias_(bias) {

    if (out_features <= 0) {
        throw std::runtime_error("LazyLinear: out_features must be positive, got " +
            std::to_string(out_features));
    }

    // No parameters registered here - deferred to first forward()
}

auto LazyLinear::materialize(int64_t in_features, Device device) -> void {
    if (in_features <= 0) {
        throw std::runtime_error("LazyLinear: inferred in_features must be positive, got " +
            std::to_string(in_features));
    }

    in_features_ = in_features;

    // S27 fix: honour `requested_device_` if the user called `.to(device)`
    // before materialisation. Previously this branch defaulted to the
    // `device` argument the caller (typically forward_impl) passed in, which
    // came from the first input's device — meaning a `model.to("cuda")`
    // followed by a CPU input silently materialised weights on CPU. The
    // explicit `.to()` is a user intent that should win over the input's
    // incidental device. Note: forward_impl already pre-resolves this when
    // it has access to both signals; this guard catches any other entry
    // point (e.g. tests / future direct callers) that calls materialize()
    // with only an input device.
    Device target_device = device;
    if (requested_device_.has_value()) {
        target_device = requested_device_.value();
    }

    // V.29: honour any pre-materialisation `to(DType)` call.  We initialise
    // weight/bias at Float32 (so Xavier / uniform_ — which currently expect a
    // floating-point storage they can index as `float*` — work correctly),
    // then cast down to the requested dtype.  This keeps the initialisation
    // distribution identical to the default Float32 path while ensuring the
    // live parameter ends up at the dtype the caller requested.
    const DType target_dtype = requested_dtype_.value_or(DType::Float32);

    // Create weight tensor and initialize with Xavier uniform
    auto weight_tensor = zeros({out_features_, in_features_}, DType::Float32, target_device);
    init::xavier_uniform_(weight_tensor);
    if (target_dtype != DType::Float32) {
        weight_tensor = weight_tensor.to(target_dtype);
    }
    Variable weight(std::move(weight_tensor), true);
    register_parameter("weight", std::move(weight));

    // Create bias tensor and initialize with Xavier uniform-derived bounds
    if (has_bias_) {
        auto bias_tensor = zeros({out_features_}, DType::Float32, target_device);
        // Use uniform initialization matching Xavier convention:
        // bound = 1 / sqrt(in_features) (same as PyTorch Linear default)
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        init::uniform_(bias_tensor, -bound, bound);
        if (target_dtype != DType::Float32) {
            bias_tensor = bias_tensor.to(target_dtype);
        }
        Variable bias_var(std::move(bias_tensor), true);
        register_parameter("bias", std::move(bias_var));
    }

    materialized_ = true;
}

auto LazyLinear::to(DType dtype) -> void {
    // V.29: stash the request so a pre-materialisation `to(DType)` survives
    // until materialize() runs.  Once materialised, parameters are live and
    // Module::to(DType) walks them in-place — the requested_dtype_ keeps
    // tracking so a subsequent re-init / clone still sees the latest dtype.
    requested_dtype_ = dtype;
    Module::to(dtype);
}

auto LazyLinear::to(Device device) -> void {
    // V.29: capture the requested device for introspection.  We do not
    // consult requested_device_ inside materialize() because the standard
    // recipe is "construct on CPU, call to(device) before training, then
    // forward an input that's also on `device`" — in that flow the
    // first-input device matches requested_device_ and the materialise path
    // would already create the parameters on the right device.  If a future
    // recipe wants device-without-input we can revisit.
    requested_device_ = device;
    Module::to(device);
}

auto LazyLinear::forward_impl(const Variable& input) -> Variable {
    auto input_shape = input.shape();

    if (input_shape.empty()) {
        throw std::runtime_error("LazyLinear: input must have at least 1 dimension");
    }

    int64_t last_dim = input_shape.back();

    // Materialize parameters on first forward call
    if (!materialized_) {
        if (last_dim <= 0) {
            throw std::runtime_error("LazyLinear: input last dimension must be positive, got " +
                std::to_string(last_dim));
        }
        // Z.11: honour an explicit pre-materialisation `to(Device)` call.
        // V.29 captured `requested_device_` but materialise() always used the
        // input tensor's device, so a `to(GPU0)` followed by a CPU input
        // silently allocated weights on the CPU. Prefer the explicit request
        // and warn once if it disagrees with the input's device.
        const Device input_dev = input.tensor().device();
        Device target_dev = input_dev;
        if (requested_device_.has_value()) {
            target_dev = requested_device_.value();
            if (target_dev.type != input_dev.type ||
                target_dev.index != input_dev.index) {
                TENZOR_WARN_ONCE(
                    "LazyLinear::materialize: requested device differs from "
                    "first-input device; honouring the explicit `to(Device)` "
                    "request. Pass the input on the matching device to silence "
                    "this warning.");
            }
        }
        materialize(last_dim, target_dev);
    } else {
        // Verify input dimension matches materialized in_features
        if (last_dim != in_features_) {
            throw std::runtime_error(
                "LazyLinear: input features (" + std::to_string(last_dim) +
                ") don't match materialized in_features (" + std::to_string(in_features_) + ")");
        }
    }

    // From here on, behave identically to Linear::forward_impl
    const bool is_2d = (input_shape.size() == 2);

    auto& weight_ptr = parameters_["weight"];
    auto& weight = *weight_ptr;

    // Fast path: 2D input - skip reshape operations entirely
    if (is_2d) {
        DType compute_dtype = input.dtype();

        // Handle device mismatch - transfer weight/bias to the input's device
        // via the autograd-aware to_device (mirrors Linear::forward_impl). This
        // keeps computation AND the gradient on the input's device, inserting a
        // DeviceTransferBackward for the weight so its gradient is moved back to
        // the weight's device during backward. Moving the input instead would
        // route the input gradient to the wrong device (it would land on the
        // weight's device while the input's grad_fn expects the input device).
        Variable weight_device = weight;
        if (input.tensor().device() != weight.tensor().device()) {
            weight_device = autograd::to_device(weight, input.tensor().device());
        }

        // Handle dtype mismatch
        Variable weight_matched = variable_cast(weight_device, compute_dtype);

        Variable bias_matched;
        Variable* bias_ptr = nullptr;
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            Variable bias_device = *bias_it->second;
            if (input.tensor().device() != bias_device.tensor().device()) {
                bias_device = autograd::to_device(bias_device, input.tensor().device());
            }
            bias_matched = variable_cast(bias_device, compute_dtype);
            bias_ptr = &bias_matched;
        }

        if (has_fused_linear_kernel(input.tensor().device())) {
            Variable zero_bias_var;
            if (!bias_ptr) {
                auto zero_bias = zeros({out_features_}, compute_dtype, input.tensor().device());
                zero_bias_var = Variable(zero_bias, false);
                return autograd::linear(input, weight_matched, zero_bias_var);
            } else {
                return autograd::linear(input, weight_matched, *bias_ptr);
            }
        } else {
            return linear_via_matmul(input, weight_matched, bias_ptr);
        }
    }

    // General path: N-D input requires reshape
    std::vector<int64_t> original_shape(input_shape.begin(), input_shape.end());
    DType compute_dtype = input.dtype();

    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    auto input_2d = autograd::reshape(input, flat_shape);

    // Handle device mismatch - transfer weight/bias to the input's device via
    // the autograd-aware to_device (mirrors Linear::forward_impl), keeping the
    // computation and the input gradient on the input's device.
    Variable weight_device = weight;
    if (input_2d.tensor().device() != weight.tensor().device()) {
        weight_device = autograd::to_device(weight, input_2d.tensor().device());
    }

    Variable weight_matched = variable_cast(weight_device, compute_dtype);

    Variable bias_matched;
    Variable* bias_ptr = nullptr;
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        Variable bias_device = *bias_it->second;
        if (input_2d.tensor().device() != bias_device.tensor().device()) {
            bias_device = autograd::to_device(bias_device, input_2d.tensor().device());
        }
        bias_matched = variable_cast(bias_device, compute_dtype);
        bias_ptr = &bias_matched;
    }

    Variable output_2d;
    if (has_fused_linear_kernel(input_2d.tensor().device())) {
        Variable zero_bias_var;
        if (!bias_ptr) {
            auto zero_bias = zeros({out_features_}, compute_dtype, input_2d.tensor().device());
            zero_bias_var = Variable(zero_bias, false);
            output_2d = autograd::linear(input_2d, weight_matched, zero_bias_var);
        } else {
            output_2d = autograd::linear(input_2d, weight_matched, *bias_ptr);
        }
    } else {
        output_2d = linear_via_matmul(input_2d, weight_matched, bias_ptr);
    }

    std::vector<int64_t> output_shape = original_shape;
    output_shape.back() = out_features_;
    return autograd::reshape(output_2d, output_shape);
}

auto LazyLinear::parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!materialized_) {
        return {};
    }
    return Module::parameters();
}

auto LazyLinear::own_parameters() -> std::vector<std::shared_ptr<Variable>> {
    if (!materialized_) {
        return {};
    }
    return Module::own_parameters();
}

auto LazyLinear::named_parameters() -> std::vector<std::pair<std::string, std::shared_ptr<Variable>>> {
    if (!materialized_) {
        return {};
    }
    return Module::named_parameters();
}

auto LazyLinear::weight() const -> const std::shared_ptr<Variable>& {
    if (!materialized_) {
        throw std::runtime_error("LazyLinear: cannot access weight before materialization");
    }
    return parameters_.at("weight");
}

auto LazyLinear::bias() const -> std::shared_ptr<Variable> {
    if (!materialized_ || !has_bias_) return nullptr;
    auto it = parameters_.find("bias");
    return (it != parameters_.end()) ? it->second : nullptr;
}

auto LazyLinear::in_features() const -> int64_t {
    if (!materialized_) {
        throw std::runtime_error("LazyLinear: in_features not known until first forward pass");
    }
    return in_features_;
}

} // namespace tenzor::nn
