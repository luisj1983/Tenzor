#include "tenzor/nn/layers/lazy_linear.hpp"
#include "tenzor/nn/init.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include <cmath>
#include <stdexcept>

namespace tenzor::nn {

// Namespace alias for autograd operations (matches linear.cpp pattern)
namespace autograd = tenzor;

// TypeCastBackward - same as in linear.cpp for dtype conversion with gradient flow
class LazyLinearTypeCastBackward : public Function {
public:
    DType original_dtype_ = DType::Float32;

    LazyLinearTypeCastBackward() = default;

    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override {
        throw std::runtime_error("LazyLinearTypeCastBackward::forward should not be called");
    }

    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {grad.to(original_dtype_)};
        }
        return {grad};
    }

    auto backward_with_variables(std::vector<Variable> grad_outputs) -> std::vector<Variable> override {
        auto& grad = grad_outputs[0];
        if (grad.dtype() != original_dtype_) {
            return {Variable(grad.tensor().to(original_dtype_), grad.requires_grad())};
        }
        return {grad};
    }
};

// Helper function to cast a Variable to a new dtype with autograd support
static auto variable_cast(const Variable& input, DType target_dtype) -> Variable {
    if (input.dtype() == target_dtype) {
        return input;
    }

    auto converted_tensor = input.tensor().to(target_dtype);
    Variable result(converted_tensor, input.requires_grad());

    if (input.requires_grad() && is_grad_enabled()) {
        auto grad_fn = std::make_shared<LazyLinearTypeCastBackward>();
        grad_fn->original_dtype_ = input.dtype();

        std::vector<Variable> input_vars = {input};
        grad_fn->set_input_variables(input_vars);

        std::vector<std::shared_ptr<Function>> next_funcs;
        if (input.grad_fn()) {
            next_funcs.push_back(input.grad_fn());
        }
        grad_fn->set_next_functions(next_funcs);

        result.set_grad_fn(grad_fn);
    }

    return result;
}

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
    return device.type == Device::Type::CPU || device.type == Device::Type::CUDA;
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

    // Create weight tensor and initialize with Xavier uniform
    auto weight_tensor = zeros({out_features_, in_features_}, DType::Float32, device);
    init::xavier_uniform_(weight_tensor);
    Variable weight(std::move(weight_tensor), true);
    register_parameter("weight", std::move(weight));

    // Create bias tensor and initialize with Xavier uniform-derived bounds
    if (has_bias_) {
        auto bias_tensor = zeros({out_features_}, DType::Float32, device);
        // Use uniform initialization matching Xavier convention:
        // bound = 1 / sqrt(in_features) (same as PyTorch Linear default)
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features_));
        init::uniform_(bias_tensor, -bound, bound);
        Variable bias_var(std::move(bias_tensor), true);
        register_parameter("bias", std::move(bias_var));
    }

    materialized_ = true;
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
        materialize(last_dim, input.tensor().device());
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

        // Handle device mismatch - transfer input to weight's device
        Variable input_device = input;
        if (input.tensor().device() != weight.tensor().device()) {
            auto input_transferred = input.tensor().to(weight.tensor().device());
            input_device = Variable(input_transferred, input.requires_grad());
            input_device.set_grad_fn(input.grad_fn());
        }

        // Handle dtype mismatch
        Variable weight_matched = variable_cast(weight, compute_dtype);

        Variable bias_matched;
        Variable* bias_ptr = nullptr;
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            bias_matched = variable_cast(*bias_it->second, compute_dtype);
            bias_ptr = &bias_matched;
        }

        if (has_fused_linear_kernel(weight.tensor().device())) {
            Variable zero_bias_var;
            if (!bias_ptr) {
                auto zero_bias = zeros({out_features_}, compute_dtype, input_device.tensor().device());
                zero_bias_var = Variable(zero_bias, false);
                return autograd::linear(input_device, weight_matched, zero_bias_var);
            } else {
                return autograd::linear(input_device, weight_matched, *bias_ptr);
            }
        } else {
            return linear_via_matmul(input_device, weight_matched, bias_ptr);
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

    Variable input_2d_device = input_2d;
    if (input_2d.tensor().device() != weight.tensor().device()) {
        auto input_transferred = input_2d.tensor().to(weight.tensor().device());
        input_2d_device = Variable(input_transferred, input_2d.requires_grad());
        input_2d_device.set_grad_fn(input_2d.grad_fn());
    }

    Variable weight_matched = variable_cast(weight, compute_dtype);

    Variable bias_matched;
    Variable* bias_ptr = nullptr;
    auto bias_it = parameters_.find("bias");
    if (bias_it != parameters_.end()) {
        bias_matched = variable_cast(*bias_it->second, compute_dtype);
        bias_ptr = &bias_matched;
    }

    Variable output_2d;
    if (has_fused_linear_kernel(weight.tensor().device())) {
        Variable zero_bias_var;
        if (!bias_ptr) {
            auto zero_bias = zeros({out_features_}, compute_dtype, input_2d_device.tensor().device());
            zero_bias_var = Variable(zero_bias, false);
            output_2d = autograd::linear(input_2d_device, weight_matched, zero_bias_var);
        } else {
            output_2d = autograd::linear(input_2d_device, weight_matched, *bias_ptr);
        }
    } else {
        output_2d = linear_via_matmul(input_2d_device, weight_matched, bias_ptr);
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
