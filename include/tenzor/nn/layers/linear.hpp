/**
 * @file linear.hpp
 * @brief Fully connected (linear) neural network layer
 *
 * Implements dense linear transformation y = xW^T + b with learnable
 * weights and optional bias.
 */

#pragma once

#include <optional>
#include "../module.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Fully connected (linear/dense) layer.
 *
 * Applies a linear transformation to incoming data: y = xW^T + b
 *
 * This layer implements a standard fully connected neural network layer
 * where each output neuron is connected to all input neurons through
 * learnable weights.
 *
 * Shape transformations:
 * - Input: (*, in_features) where * is any number of dimensions
 * - Output: (*, out_features)
 * - Weight: (out_features, in_features)
 * - Bias: (out_features) if enabled
 *
 * Parameters are initialized using Kaiming uniform initialization:
 * - Weights: U(-sqrt(k), sqrt(k)) where k = 1/in_features
 * - Bias: U(-sqrt(k), sqrt(k))
 *
 * @code
 * // Create 784 -> 128 fully connected layer
 * Linear fc1(784, 128);
 *
 * // Forward pass
 * Variable x(Tensor({batch, 784}, DType::Float32, Device::cpu()), true);
 * Variable output = fc1.forward(x);  // Shape: {batch, 128}
 *
 * // Access parameters
 * auto& weights = fc1.weight();
 * auto& bias_opt = fc1.bias();
 * @endcode
 *
 * @see Module for base class interface
 */
class Linear : public Module {
public:
    /**
     * @brief Construct linear layer.
     *
     * @param in_features Size of each input sample
     * @param out_features Size of each output sample
     * @param bias If true, add learnable bias (default: true)
     *
     * @code
     * Linear layer(784, 10, true);  // With bias
     * Linear layer_no_bias(784, 10, false);  // Without bias
     * @endcode
     */
    Linear(int64_t in_features, int64_t out_features, bool bias = true);

    /**
     * @brief Forward pass through linear layer.
     *
     * Computes y = xW^T + b where:
     * - x is input tensor of shape (*, in_features)
     * - W is weight matrix of shape (out_features, in_features)
     * - b is bias vector of shape (out_features)
     *
     * @param input Input variable of shape (*, in_features)
     * @return Output variable of shape (*, out_features)
     *
     * @throws std::runtime_error if input shape is incompatible
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in_features=" + std::to_string(in_features_) +
               ", out_features=" + std::to_string(out_features_) +
               ", bias=" + (has_bias_ ? "True" : "False");
    }

    /**
     * @brief Get weight parameter.
     *
     * @return Const reference to shared_ptr to weight variable
     */
    auto weight() const -> const std::shared_ptr<Variable>& { return parameters_.at("weight"); }

    /**
     * @brief Check if layer has bias.
     */
    auto has_bias() const -> bool { return has_bias_; }

    /**
     * @brief Get bias parameter (if present).
     *
     * @return Shared pointer to bias variable, or nullptr if no bias
     */
    auto bias() const -> std::shared_ptr<Variable> {
        if (!has_bias_) return nullptr;
        auto it = parameters_.find("bias");
        return (it != parameters_.end()) ? it->second : nullptr;
    }

private:
    int64_t in_features_;                   ///< Input feature dimension
    int64_t out_features_;                  ///< Output feature dimension
    bool has_bias_;                         ///< Whether this layer has bias

    /**
     * @brief Initialize parameters using Kaiming uniform.
     */
    auto reset_parameters() -> void;
};

/**
 * @brief Bilinear layer: y[k] = x1^T · W[k] · x2 + b[k].
 *
 * Implements the bilinear transformation matching torch.nn.Bilinear.
 *
 * Shape transformations:
 * - Input1: (*, in1_features), Input2: (*, in2_features)
 * - Output: (*, out_features)
 * - Weight: (out_features, in1_features, in2_features)
 * - Bias:   (out_features) if enabled
 *
 * This implementation is a Variable-level composition that materializes the
 * outer product of x1 and x2, then applies a reshaped linear projection. The
 * autograd graph threads naturally through the existing matmul and elementwise
 * backward paths — no new OpId or kernel required.
 */
class Bilinear : public Module {
public:
    Bilinear(int64_t in1_features, int64_t in2_features,
             int64_t out_features, bool bias = true);

    /**
     * @brief Forward pass taking two input Variables.
     *
     * The Module base class's single-input forward(...) is not used for this
     * layer; callers should invoke forward(x1, x2) directly.
     */
    auto forward(const Variable& input1, const Variable& input2) -> Variable;

    /**
     * @brief Overridden single-input forward — throws, since Bilinear needs
     * two inputs. Use forward(x1, x2) instead.
     */
    auto forward_impl(const Variable& input) -> Variable override;

    auto extra_repr() const -> std::string override {
        return "in1_features=" + std::to_string(in1_features_) +
               ", in2_features=" + std::to_string(in2_features_) +
               ", out_features=" + std::to_string(out_features_) +
               ", bias=" + (has_bias_ ? "True" : "False");
    }

    auto weight() const -> const std::shared_ptr<Variable>& {
        return parameters_.at("weight");
    }
    auto has_bias() const -> bool { return has_bias_; }
    auto bias() const -> std::shared_ptr<Variable> {
        if (!has_bias_) return nullptr;
        auto it = parameters_.find("bias");
        return (it != parameters_.end()) ? it->second : nullptr;
    }

private:
    int64_t in1_features_;
    int64_t in2_features_;
    int64_t out_features_;
    bool has_bias_;
    auto reset_parameters() -> void;
};

} // namespace nn
} // namespace tenzor
