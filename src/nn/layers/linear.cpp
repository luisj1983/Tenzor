#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/utils/variable_cast.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <iostream>

namespace tenzor::nn {

// Namespace alias for autograd operations
namespace autograd = tenzor;
using tenzor::nn::variable_cast;

Linear::Linear(int64_t in_features, int64_t out_features, bool bias)
    : in_features_(in_features), out_features_(out_features), has_bias_(bias) {

    if (in_features <= 0) {
        throw std::runtime_error("Linear: in_features must be positive, got " +
            std::to_string(in_features));
    }
    if (out_features <= 0) {
        throw std::runtime_error("Linear: out_features must be positive, got " +
            std::to_string(out_features));
    }

    // Initialize weight with Kaiming uniform (matches PyTorch default)
    // PyTorch uses U(-bound, bound) where bound = sqrt(1 / fan_in) for linear layers
    float bound = std::sqrt(1.0f / static_cast<float>(in_features));
    Variable weight(rand({out_features, in_features}) * (2.0f * bound) - bound, true);
    register_parameter("weight", std::move(weight));

    // Initialize bias with uniform(-1/sqrt(in_features), 1/sqrt(in_features))
    if (bias) {
        float bound = 1.0f / std::sqrt(static_cast<float>(in_features));
        Variable bias_var(rand({out_features}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

// Helper function to compute linear using matmul (fallback for backends without fused linear)
static auto linear_via_matmul(const Variable& input, const Variable& weight,
                               const Variable* bias) -> Variable {
    // Transpose weight: [out_features, in_features] -> [in_features, out_features]
    auto weight_t = autograd::permute(weight, {1, 0});

    // Matrix multiplication: input @ weight.T
    auto output = autograd::matmul(input, weight_t);

    // Add bias if present
    if (bias) {
        output = output + *bias;
    }

    return output;
}

// Check if fused linear kernel is available for this backend
static bool has_fused_linear_kernel(Device device) {
    return is_op_supported(OpId::Linear, device.type);
}

auto Linear::forward_impl(const Variable& input) -> Variable {
    // input: [*, in_features] where * can be any number of dimensions
    // weight: [out_features, in_features]
    // output: [*, out_features]

    auto input_shape = input.shape();
    if (input_shape.empty()) {
        throw std::invalid_argument("Linear: input tensor must have at least 1 dimension");
    }
    if (input_shape.back() != in_features_) {
        throw std::invalid_argument("Linear: expected input last dim=" +
            std::to_string(in_features_) + ", got " + std::to_string(input_shape.back()));
    }
    const bool is_2d = (input_shape.size() == 2);

    auto& weight = *parameters_.at("weight");

    // Fast path: 2D input - skip reshape operations entirely
    // This eliminates 2 ReshapeBackward allocations per forward pass
    if (is_2d) {
        DType compute_dtype = input.dtype();

        // Handle device mismatch - transfer weight/bias to input's device via autograd
        // This keeps computation on the input's device (e.g., GPU), avoiding
        // cross-device data access issues with backend-specific runtimes
        Variable weight_device = weight;
        if (input.tensor().device() != weight.tensor().device()) {
            weight_device = tenzor::to_device(weight, input.tensor().device());
        }

        // Handle dtype mismatch - convert weight/bias to input's dtype using gradient-aware cast
        // This ensures gradients flow back to weight with proper dtype conversion
        Variable weight_matched = variable_cast(weight_device, compute_dtype);

        // Get bias and convert if needed
        Variable bias_matched;
        Variable* bias_ptr = nullptr;
        if (has_bias_) {
            auto bias_it = parameters_.find("bias");
            if (bias_it != parameters_.end()) {
                Variable bias_device = *bias_it->second;
                if (input.tensor().device() != bias_device.tensor().device()) {
                    bias_device = tenzor::to_device(bias_device, input.tensor().device());
                }
                bias_matched = variable_cast(bias_device, compute_dtype);
                bias_ptr = &bias_matched;
            }
        }

        // Use fused linear kernel if available (CPU with MKL, CUDA with cuBLAS)
        // Falls back to matmul + add for other backends
        if (has_fused_linear_kernel(input.tensor().device())) {
            if (!bias_ptr) {
                // No bias: use matmul path to avoid unnecessary zero tensor allocation
                return linear_via_matmul(input, weight_matched, nullptr);
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

    // Calculate total batch size
    int64_t batch_total = 1;
    for (size_t i = 0; i < original_shape.size() - 1; ++i) {
        batch_total *= original_shape[i];
    }

    // Flatten input to 2D
    std::vector<int64_t> flat_shape = {batch_total, in_features_};
    auto input_2d = autograd::reshape(input, flat_shape);

    // Handle device mismatch - transfer weight/bias to input's device via autograd
    Variable weight_device = weight;
    if (input_2d.tensor().device() != weight.tensor().device()) {
        weight_device = tenzor::to_device(weight, input_2d.tensor().device());
    }

    // Handle dtype mismatch - convert weight/bias to input's dtype using gradient-aware cast
    Variable weight_matched = variable_cast(weight_device, compute_dtype);

    // Get bias and convert if needed
    Variable bias_matched;
    Variable* bias_ptr = nullptr;
    if (has_bias_) {
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            Variable bias_device = *bias_it->second;
            if (input_2d.tensor().device() != bias_device.tensor().device()) {
                bias_device = tenzor::to_device(bias_device, input_2d.tensor().device());
            }
            bias_matched = variable_cast(bias_device, compute_dtype);
            bias_ptr = &bias_matched;
        }
    }

    // Compute linear operation
    Variable output_2d;
    if (has_fused_linear_kernel(input_2d.tensor().device()) && bias_ptr) {
        // Use fused linear kernel (CPU with MKL, CUDA with cuBLAS) — only when bias present
        output_2d = autograd::linear(input_2d, weight_matched, *bias_ptr);
    } else {
        // Fallback: use matmul + add for other backends
        output_2d = linear_via_matmul(input_2d, weight_matched, bias_ptr);
    }

    // Reshape output back
    std::vector<int64_t> output_shape = original_shape;
    output_shape.back() = out_features_;
    return autograd::reshape(output_2d, output_shape);
}

auto Linear::reset_parameters() -> void {
    float bound = std::sqrt(1.0f / static_cast<float>(in_features_));

    // Update existing parameters in-place to preserve shared_ptr identity
    // (external references like optimizers hold the same shared_ptr)
    auto weight_it = parameters_.find("weight");
    if (weight_it != parameters_.end()) {
        *weight_it->second = Variable(rand({out_features_, in_features_}) * (2.0f * bound) - bound, true);
    } else {
        Variable weight(rand({out_features_, in_features_}) * (2.0f * bound) - bound, true);
        register_parameter("weight", std::move(weight));
    }

    if (has_bias_) {
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            *bias_it->second = Variable(rand({out_features_}) * (2.0f * bound) - bound, true);
        } else {
            Variable bias_var(rand({out_features_}) * (2.0f * bound) - bound, true);
            register_parameter("bias", std::move(bias_var));
        }
    }
}

// ============================================================================
// Bilinear
// ============================================================================

Bilinear::Bilinear(int64_t in1_features, int64_t in2_features,
                   int64_t out_features, bool bias)
    : in1_features_(in1_features),
      in2_features_(in2_features),
      out_features_(out_features),
      has_bias_(bias) {
    if (in1_features <= 0 || in2_features <= 0 || out_features <= 0) {
        throw std::runtime_error("Bilinear: all feature sizes must be positive");
    }

    // PyTorch uses bound = 1/sqrt(in1_features) for both weight and bias.
    float bound = 1.0f / std::sqrt(static_cast<float>(in1_features));
    Variable weight(
        rand({out_features, in1_features, in2_features}) * (2.0f * bound) - bound,
        true);
    register_parameter("weight", std::move(weight));

    if (bias) {
        Variable bias_var(rand({out_features}) * (2.0f * bound) - bound, true);
        register_parameter("bias", std::move(bias_var));
    }
}

auto Bilinear::forward_impl(const Variable& /*input*/) -> Variable {
    throw std::runtime_error(
        "Bilinear takes two inputs; call forward(x1, x2) instead of forward(x).");
}

auto Bilinear::forward(const Variable& input1, const Variable& input2) -> Variable {
    // Shapes (PyTorch semantics):
    //   input1: [B, in1], input2: [B, in2]
    //   weight: [out, in1, in2], bias: [out]
    // Identity:
    //   y[b, k] = Σ_{j,l} W[k, j, l] * x1[b, j] * x2[b, l] + b[k]
    //           = Σ_{jl} W_flat[k, jl] * (x1 ⊗ x2)[b, jl] + b[k]
    // Implementation: build outer product then matmul with flattened weight.
    auto x1_shape = input1.shape();
    auto x2_shape = input2.shape();
    if (x1_shape.empty() || x2_shape.empty()) {
        throw std::invalid_argument("Bilinear: inputs must have ≥1 dimension");
    }
    if (x1_shape.back() != in1_features_ || x2_shape.back() != in2_features_) {
        throw std::invalid_argument(
            "Bilinear: input feature dims do not match layer configuration");
    }
    if (x1_shape.size() != x2_shape.size()) {
        throw std::invalid_argument("Bilinear: inputs must have same rank");
    }
    // Batch dims (all except trailing feature dim) must match.
    for (size_t i = 0; i + 1 < x1_shape.size(); ++i) {
        if (x1_shape[i] != x2_shape[i]) {
            throw std::invalid_argument(
                "Bilinear: inputs must share all batch dims");
        }
    }

    // Flatten leading dims into a single batch B.
    int64_t B = 1;
    for (size_t i = 0; i + 1 < x1_shape.size(); ++i) B *= x1_shape[i];

    auto x1_2d = autograd::reshape(input1, {B, in1_features_});
    auto x2_2d = autograd::reshape(input2, {B, in2_features_});

    // Outer product: [B, in1, 1] * [B, 1, in2] → [B, in1, in2]
    auto x1_3d = autograd::reshape(x1_2d, {B, in1_features_, 1});
    auto x2_3d = autograd::reshape(x2_2d, {B, 1, in2_features_});
    auto outer = x1_3d * x2_3d;  // broadcasts to [B, in1, in2]

    // Flatten to [B, in1*in2] and multiply by weight reshaped to [out, in1*in2].
    auto outer_flat = autograd::reshape(outer, {B, in1_features_ * in2_features_});

    auto& weight = *parameters_.at("weight");
    auto w_flat = autograd::reshape(weight, {out_features_, in1_features_ * in2_features_});
    // [B, J*L] @ [J*L, out] = [B, out]
    auto w_flat_t = autograd::permute(w_flat, {1, 0});
    auto output_2d = autograd::matmul(outer_flat, w_flat_t);

    if (has_bias_) {
        auto& bias = *parameters_.at("bias");
        output_2d = output_2d + bias;
    }

    // Restore leading dims.
    std::vector<int64_t> output_shape(x1_shape.begin(), x1_shape.end());
    output_shape.back() = out_features_;
    return autograd::reshape(output_2d, output_shape);
}

auto Bilinear::reset_parameters() -> void {
    float bound = 1.0f / std::sqrt(static_cast<float>(in1_features_));
    auto weight_it = parameters_.find("weight");
    if (weight_it != parameters_.end()) {
        *weight_it->second = Variable(
            rand({out_features_, in1_features_, in2_features_}) * (2.0f * bound) - bound,
            true);
    }
    if (has_bias_) {
        auto bias_it = parameters_.find("bias");
        if (bias_it != parameters_.end()) {
            *bias_it->second = Variable(
                rand({out_features_}) * (2.0f * bound) - bound, true);
        }
    }
}

} // namespace tenzor::nn
