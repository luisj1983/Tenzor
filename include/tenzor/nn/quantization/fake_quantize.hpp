/**
 * @file fake_quantize.hpp
 * @brief Fake quantization for quantization-aware training (QAT)
 *
 * Provides fake quantization layers that simulate quantization effects
 * during training by quantizing and immediately dequantizing tensors.
 * This allows gradients to flow while modeling quantization errors.
 */

#pragma once

#include <memory>
#include "quantize.hpp"
#include "observer.hpp"
#include "../module.hpp"
#include "../../autograd/variable.hpp"
#include "../../autograd/function.hpp"

namespace tenzor {
namespace nn {
namespace quantization {

/**
 * @brief Fake quantization module for QAT.
 *
 * FakeQuantize simulates quantization by:
 * 1. Quantizing input to integer range
 * 2. Immediately dequantizing back to floating point
 * 3. Passing result through with gradient flow
 *
 * During training, this models quantization error so the network learns
 * to be robust to quantization. The straight-through estimator (STE)
 * allows gradients to flow despite the non-differentiable quantization.
 *
 * The quantization parameters can be:
 * - Fixed: Set manually and not updated
 * - Learned: Updated via backpropagation as trainable parameters
 * - Observed: Updated based on observed activation statistics
 *
 * @code
 * // Create fake quantize layer
 * auto fake_quant = std::make_shared<FakeQuantize>(
 *     QuantDType::INT8,
 *     QuantizationScheme::PerTensorSymmetric,
 *     true  // learnable parameters
 * );
 *
 * // Use in training
 * Variable activations = some_layer->forward(input);
 * Variable quant_activations = fake_quant->forward(activations);
 * // Network learns to handle quantization noise
 * @endcode
 */
class FakeQuantize : public Module {
public:
    /**
     * @brief Construct fake quantize module.
     *
     * @param dtype Target quantized data type
     * @param scheme Quantization scheme
     * @param learnable If true, scale and zero_point are learnable (default: false)
     * @param observer_enabled If true, use observer to update qparams (default: true)
     * @param axis Channel axis for per-channel quantization (default: -1)
     */
    explicit FakeQuantize(
        QuantDType dtype = QuantDType::INT8,
        QuantizationScheme scheme = QuantizationScheme::PerTensorSymmetric,
        bool learnable = false,
        bool observer_enabled = true,
        int64_t axis = -1
    );

    /**
     * @brief Forward pass with fake quantization.
     *
     * In training mode with observer enabled:
     * 1. Observe input to update quantization parameters
     * 2. Apply fake quantization with current parameters
     *
     * In eval mode or with observer disabled:
     * 1. Apply fake quantization with fixed parameters
     *
     * @param input Input variable
     * @return Fake-quantized output variable
     */
    auto forward_impl(const Variable& input) -> Variable override;

    /**
     * @brief Enable or disable observer.
     *
     * @param enabled If true, observer updates qparams based on data
     */
    auto enable_observer(bool enabled = true) -> void { observer_enabled_ = enabled; }

    /**
     * @brief Disable observer (fix quantization parameters).
     */
    auto disable_observer() -> void { observer_enabled_ = false; }

    /**
     * @brief Enable or disable fake quantization.
     *
     * When disabled, acts as identity (useful for comparison).
     *
     * @param enabled If true, apply fake quantization
     */
    auto enable_fake_quant(bool enabled = true) -> void { fake_quant_enabled_ = enabled; }

    /**
     * @brief Disable fake quantization (acts as identity).
     */
    auto disable_fake_quant() -> void { fake_quant_enabled_ = false; }

    /**
     * @brief Get current quantization parameters.
     */
    auto get_qparams() const -> const QuantizationParams& { return *qparams_; }

    /**
     * @brief Set quantization parameters manually.
     *
     * @param params New quantization parameters
     */
    auto set_qparams(const QuantizationParams& params) -> void;

    /**
     * @brief Calculate and freeze quantization parameters from observer.
     *
     * Computes final qparams from observer statistics and disables observer.
     * Call this after calibration before starting QAT.
     */
    auto calculate_qparams() -> void;

    /**
     * @brief Reset observer statistics.
     */
    auto reset_observer() -> void;

    /**
     * @brief Get underlying observer.
     */
    auto observer() -> Observer* { return observer_.get(); }

    /**
     * @brief Check if parameters are learnable.
     */
    auto is_learnable() const -> bool { return learnable_; }

private:
    QuantDType dtype_;                              ///< Quantized data type
    QuantizationScheme scheme_;                     ///< Quantization scheme
    bool learnable_;                                ///< Learnable parameters flag
    bool observer_enabled_;                         ///< Observer enabled flag
    bool fake_quant_enabled_{true};                 ///< Fake quantization enabled
    int64_t axis_;                                  ///< Channel axis
    std::unique_ptr<Observer> observer_;            ///< Statistics observer
    std::unique_ptr<QuantizationParams> qparams_;   ///< Current quantization params

    /**
     * @brief Apply fake quantization to tensor.
     */
    auto apply_fake_quantization(const Tensor& input) const -> Tensor;
};

/**
 * @brief Fake quantization function for activations.
 *
 * Functional interface for fake quantization without creating a module.
 * Useful for ad-hoc quantization simulation.
 *
 * @param input Input tensor
 * @param scale Scale factor
 * @param zero_point Zero point
 * @param dtype Quantized data type
 * @return Fake-quantized tensor
 *
 * @code
 * Tensor scale({1}, DType::Float32, Device::cpu());
 * Tensor zero_point({1}, DType::Int32, Device::cpu());
 * scale.fill_(0.1f);
 * zero_point.fill_(0);
 *
 * Tensor quantized = fake_quantize_activation(input, scale, zero_point,
 *                                            QuantDType::INT8);
 * @endcode
 */
auto fake_quantize_activation(
    const Tensor& input,
    const Tensor& scale,
    const Tensor& zero_point,
    QuantDType dtype
) -> Tensor;

/**
 * @brief Fake quantization function for weights (per-channel).
 *
 * Applies per-channel fake quantization to weight tensors.
 *
 * @param weight Weight tensor
 * @param scale Per-channel scale factors
 * @param zero_point Per-channel zero points
 * @param axis Channel axis (typically 0)
 * @param dtype Quantized data type
 * @return Fake-quantized weight tensor
 */
auto fake_quantize_weight(
    const Tensor& weight,
    const Tensor& scale,
    const Tensor& zero_point,
    int64_t axis,
    QuantDType dtype
) -> Tensor;

/**
 * @brief Learnable fake quantization with trainable parameters.
 *
 * Extends fake quantization with learnable scale and zero_point parameters
 * that are optimized during training for optimal quantization.
 *
 * @code
 * auto learnable_fq = std::make_shared<LearnableFakeQuantize>();
 * learnable_fq->train();
 *
 * // During training, scale and zero_point are updated via gradients
 * Variable output = learnable_fq->forward(input);
 * loss.backward();
 * optimizer.step();  // Updates scale and zero_point
 * @endcode
 */
class LearnableFakeQuantize : public FakeQuantize {
public:
    /**
     * @brief Construct learnable fake quantization module.
     *
     * @param dtype Target quantized data type
     * @param scheme Quantization scheme
     * @param axis Channel axis for per-channel quantization (default: -1)
     */
    explicit LearnableFakeQuantize(
        QuantDType dtype = QuantDType::INT8,
        QuantizationScheme scheme = QuantizationScheme::PerTensorSymmetric,
        int64_t axis = -1
    );

    /**
     * @brief Initialize learnable parameters from observer statistics.
     *
     * Call after an initial observation phase to set good initial values
     * for scale and zero_point before enabling gradient updates.
     */
    auto init_from_observer() -> void;
};

/**
 * @brief Quantization-aware training helper.
 *
 * Manages fake quantization for an entire model, providing convenience
 * methods for preparing models for QAT.
 *
 * @code
 * QATHelper qat;
 * qat.prepare_qat(model, QuantDType::INT8);
 *
 * // Training loop with QAT
 * for (auto& batch : training_data) {
 *     qat.enable_observer();
 *     auto output = model.forward(batch);
 *     // ... backward pass ...
 * }
 *
 * // Disable observers before final training
 * qat.disable_observer();
 * @endcode
 */
class QATHelper {
public:
    QATHelper() = default;

    /**
     * @brief Prepare model for quantization-aware training.
     *
     * Inserts fake quantization modules after activations and before
     * quantizable layers. Configures observers for calibration.
     *
     * @param model Model to prepare
     * @param dtype Target quantized data type
     * @param scheme Quantization scheme
     * @param learnable Use learnable fake quantization
     */
    auto prepare_qat(
        Module& model,
        QuantDType dtype = QuantDType::INT8,
        QuantizationScheme scheme = QuantizationScheme::PerTensorSymmetric,
        bool learnable = false
    ) -> void;

    /**
     * @brief Enable observers for all fake quantization modules.
     */
    auto enable_observer() -> void;

    /**
     * @brief Disable observers (fix quantization parameters).
     */
    auto disable_observer() -> void;

    /**
     * @brief Calculate and freeze all quantization parameters.
     *
     * Call after calibration phase to freeze qparams before fine-tuning.
     */
    auto freeze_bn_stats() -> void;

    /**
     * @brief Convert QAT model to quantized inference model.
     *
     * Replaces fake quantization with actual quantized operations.
     *
     * @param model Model to convert
     * @return Quantized model ready for inference
     */
    auto convert_to_quantized(Module& model) -> std::shared_ptr<Module>;

private:
    std::vector<std::shared_ptr<FakeQuantize>> fake_quant_modules_;  ///< Tracked modules
};

/**
 * @brief Straight-through estimator gradient for quantization.
 *
 * Implements the gradient pass-through for the non-differentiable
 * quantization operation. During backward pass, gradients flow through
 * as if quantization was identity, except for values outside the
 * quantization range which get zero gradient.
 */
class StraightThroughEstimator {
public:
    /**
     * @brief Apply STE forward pass.
     *
     * @param input Input tensor
     * @param quant_min Minimum quantized value
     * @param quant_max Maximum quantized value
     * @return Clamped tensor for gradient computation
     */
    static auto forward(const Tensor& input, float quant_min, float quant_max) -> Tensor;

    /**
     * @brief Apply STE backward pass.
     *
     * @param grad_output Gradient from next layer
     * @param input Original input
     * @param quant_min Minimum quantized value
     * @param quant_max Maximum quantized value
     * @return Gradient for input
     */
    static auto backward(const Tensor& grad_output, const Tensor& input,
                        float quant_min, float quant_max) -> Tensor;
};

/**
 * @brief Autograd Function for fake quantization with straight-through estimator.
 *
 * Implements quantize-then-dequantize in the forward pass. In the backward pass,
 * uses the straight-through estimator: gradients pass through unchanged for values
 * within the quantization range [quant_min, quant_max], and are zeroed for
 * out-of-range values (clamped by the quantization).
 *
 * This allows training to adapt to quantization noise while maintaining valid
 * gradient flow for most of the parameter space.
 *
 * @code
 * auto fn = std::make_shared<FakeQuantizeFunction>(scale, zero_point, quant_min, quant_max);
 * auto outputs = fn->forward({input_var});
 * // backward() will use STE gradient
 * @endcode
 */
class FakeQuantizeFunction : public tenzor::Function {
public:
    /**
     * @brief Construct with quantization parameters.
     *
     * @param scale Quantization scale factor
     * @param zero_point Quantization zero point
     * @param quant_min Minimum quantized value (e.g. -128 for INT8)
     * @param quant_max Maximum quantized value (e.g. 127 for INT8)
     */
    FakeQuantizeFunction(float scale, float zero_point,
                         float quant_min, float quant_max);

    /**
     * @brief Forward: quantize then dequantize, saving input for backward.
     */
    auto forward(std::vector<Variable> inputs) -> std::vector<Variable> override;

    /**
     * @brief Backward: STE gradient (pass through within range, zero outside).
     */
    auto backward(std::vector<Tensor> grad_outputs) -> std::vector<Tensor> override;

private:
    float scale_;
    float zero_point_;
    float quant_min_;
    float quant_max_;
};

/**
 * @brief Convenience function to apply fake quantize with autograd support.
 *
 * Creates a FakeQuantizeFunction, runs forward, and returns the result
 * with proper gradient tracking for STE backward.
 *
 * @param input Input variable
 * @param scale Quantization scale
 * @param zero_point Quantization zero point
 * @param quant_min Minimum quantized value
 * @param quant_max Maximum quantized value
 * @return Fake-quantized variable with STE gradient
 */
auto fake_quantize_with_grad(
    const Variable& input,
    float scale,
    float zero_point,
    float quant_min,
    float quant_max
) -> Variable;

/**
 * @brief Fold BatchNorm2d into a preceding Conv2d layer.
 *
 * Walks the model looking for Conv2d -> BatchNorm2d patterns and folds the
 * BN parameters into the Conv2d weights and bias using the standard formulas:
 *
 *   w_folded = gamma / sqrt(var + eps) * w_conv
 *   b_folded = gamma / sqrt(var + eps) * (b_conv - mean) + beta
 *
 * This eliminates the BatchNorm layer entirely, reducing both computation
 * and memory during inference. The folded model produces identical outputs.
 *
 * @param model Model to fold (modified in-place via state_dict)
 * @note Only folds Conv2d->BatchNorm2d pairs found in Sequential containers.
 *       Nested or non-sequential patterns are left unchanged.
 */
auto fold_bn(Module& model) -> void;

} // namespace quantization
} // namespace nn
} // namespace tenzor
