/**
 * @file mixed_precision.hpp
 * @brief Mixed precision training utilities for FP16/BF16 training
 *
 * Provides high-level MixedPrecisionTrainer class that combines automatic
 * mixed precision (AMP) with gradient scaling for efficient and stable training.
 */

#pragma once

#include <memory>
#include <functional>
#include <optional>
#include "module.hpp"
#include "optim/optimizer.hpp"
#include "amp/autocast.hpp"
#include "amp/grad_scaler.hpp"
#include "training.hpp"
#include <tenzor/autograd/variable.hpp>

namespace tenzor {
namespace nn {

/**
 * @brief Mixed precision training configuration
 */
struct MixedPrecisionConfig {
    /// Target dtype for mixed precision (Float16 or BFloat16)
    DType dtype = DType::Float16;

    /// Device type to apply mixed precision to; std::nullopt (default) applies
    /// autocast to whatever device the model/tensors are actually on (see
    /// amp::Autocast, whose own nullopt means "any device"). Only set this to
    /// restrict autocast to one specific device type.
    std::optional<Device::Type> device_type = std::nullopt;

    /// Enable automatic mixed precision
    bool enabled = true;

    /// Initial loss scale for gradient scaler
    float init_scale = 65536.0f;

    /// Scale growth factor on successful iterations
    float growth_factor = 2.0f;

    /// Scale backoff factor on overflow
    float backoff_factor = 0.5f;

    /// Iterations before attempting scale growth
    int growth_interval = 2000;

    /// Maintain FP32 master copies of parameters for numerical stability.
    /// When enabled, the optimizer updates FP32 master weights, which are
    /// then copied to the model's lower-precision working parameters.
    bool use_master_weights = false;

    /**
     * @brief Create default FP16 configuration, device-agnostic (autocast
     * applies to whichever device the model/tensors are on).
     */
    static auto fp16() -> MixedPrecisionConfig {
        return MixedPrecisionConfig{
            DType::Float16,
            std::nullopt,
            true,
            65536.0f,
            2.0f,
            0.5f,
            2000
        };
    }

    /**
     * @brief Create default BFloat16 configuration, device-agnostic.
     */
    static auto bfloat16() -> MixedPrecisionConfig {
        return MixedPrecisionConfig{
            DType::BFloat16,
            std::nullopt,
            true,
            65536.0f,
            2.0f,
            0.5f,
            2000
        };
    }

    /**
     * @brief Create default FP16 configuration explicitly restricted to CUDA.
     * Prefer fp16() unless you specifically want autocast to skip non-CUDA
     * tensors.
     */
    static auto fp16_cuda() -> MixedPrecisionConfig {
        return MixedPrecisionConfig{
            DType::Float16,
            Device::Type::CUDA,
            true,
            65536.0f,
            2.0f,
            0.5f,
            2000
        };
    }

    /**
     * @brief Create BFloat16 configuration explicitly restricted to CUDA.
     * Prefer bfloat16() unless you specifically want autocast to skip
     * non-CUDA tensors.
     */
    static auto bfloat16_cuda() -> MixedPrecisionConfig {
        return MixedPrecisionConfig{
            DType::BFloat16,
            Device::Type::CUDA,
            true,
            65536.0f,
            2.0f,
            0.5f,
            2000
        };
    }

    /**
     * @brief Create conservative configuration (slower scale growth),
     * device-agnostic.
     */
    static auto conservative() -> MixedPrecisionConfig {
        return MixedPrecisionConfig{
            DType::Float16,
            std::nullopt,
            true,
            1024.0f,
            1.5f,
            0.75f,
            5000
        };
    }
};

/**
 * @brief Manages FP32 master copies of model parameters for mixed-precision training.
 *
 * When training in FP16/BF16, parameter updates accumulate rounding errors.
 * Master weights maintain FP32 copies: the optimizer updates the FP32 masters,
 * which are then copied to the model's lower-precision working parameters
 * before each forward pass.
 */
class MasterWeightManager {
public:
    /**
     * @brief Initialize master weights from model parameters.
     * @param model Model whose parameters will be managed
     */
    explicit MasterWeightManager(std::shared_ptr<Module> model);

    /// Copy FP32 master weights -> model's working parameters (before forward)
    auto sync_to_working() -> void;

    /// Copy model's working parameters -> FP32 masters (after optimizer step)
    auto sync_from_working() -> void;

    /// Zero the gradients of the working (FP16/BF16) parameters. The optimizer's
    /// zero_grad() only touches the FP32 master Variables, so the working leaves
    /// that backward() accumulates into must be zeroed independently each step.
    auto zero_working_grads() -> void;

    /// Get the FP32 master parameters (for optimizer construction)
    auto master_params() -> std::vector<std::shared_ptr<Variable>>&;

private:
    std::shared_ptr<Module> model_;
    std::vector<std::shared_ptr<Variable>> master_variables_;
    std::vector<std::shared_ptr<Variable>> working_refs_;  ///< References to model params
};

/**
 * @brief High-level mixed precision training wrapper
 *
 * MixedPrecisionTrainer provides a complete training API with automatic
 * mixed precision (AMP) and gradient scaling. It handles:
 * - Automatic casting of operations to FP16/BF16
 * - Loss scaling to prevent gradient underflow
 * - Gradient unscaling and overflow detection
 * - Dynamic loss scale adjustment
 * - Standard training loop patterns
 *
 * **Key Features:**
 * - Forward pass automatically uses FP16/BF16 where beneficial
 * - Loss computation in FP32 for numerical stability
 * - Automatic gradient scaling prevents underflow
 * - Skips optimizer updates on gradient overflow
 * - 2-3x training speedup on modern GPUs
 *
 * **Workflow:**
 * 1. Forward pass in mixed precision (autocast enabled)
 * 2. Loss computation in FP32 (autocast disabled)
 * 3. Scale loss before backward pass
 * 4. Compute scaled gradients
 * 5. Unscale gradients and check for overflow
 * 6. Update parameters if no overflow
 * 7. Adjust loss scale for next iteration
 *
 * **Example Usage:**
 * @code
 * // Create model, optimizer, and loss
 * auto model = std::make_shared<MyModel>();
 * auto optimizer = std::make_shared<Adam>(model->parameters(), 0.001);
 * auto loss_fn = [](const Variable& pred, const Variable& target) {
 *     return mse_loss(pred, target);
 * };
 *
 * // Create mixed precision trainer
 * auto config = MixedPrecisionConfig::fp16();
 * MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);
 *
 * // Train for 10 epochs
 * DataLoader train_loader(train_data, 32);
 * trainer.fit(train_loader, 10);
 * @endcode
 *
 * **Performance:**
 * - FP16 training: 2-3x speedup on Volta/Turing/Ampere GPUs
 * - BF16 training: 1.5-2x speedup on Ampere+ GPUs
 * - Memory savings: ~40% reduction in activation memory
 *
 * **Numerical Stability:**
 * - Loss always computed in FP32
 * - Dynamic loss scaling prevents gradient underflow
 * - Automatic overflow detection and handling
 * - Proven stable on most modern architectures
 *
 * @see amp::Autocast, amp::GradScaler, NeuralNetwork
 */
class MixedPrecisionTrainer {
public:
    /**
     * @brief Construct mixed precision trainer
     *
     * @param model Neural network model (any Module subclass)
     * @param optimizer Optimization algorithm (SGD, Adam, etc.)
     * @param loss_fn Loss function callable
     * @param config Mixed precision configuration
     *
     * @par Requirements
     * - model must have forward() method
     * - optimizer must be initialized with model parameters
     * - loss_fn must be callable with (predictions, targets)
     * - GPU with FP16/BF16 support recommended
     *
     * @code
     * auto model = std::make_shared<ResNet50>();
     * auto optimizer = std::make_shared<Adam>(model->parameters(), 0.001);
     * auto loss_fn = [](const Variable& pred, const Variable& target) {
     *     return cross_entropy_loss(pred, target);
     * };
     * auto config = MixedPrecisionConfig::fp16();
     * MixedPrecisionTrainer trainer(model, optimizer, loss_fn, config);
     * @endcode
     */
    template<typename LossFn>
    MixedPrecisionTrainer(
        std::shared_ptr<Module> model,
        std::shared_ptr<optim::Optimizer> optimizer,
        LossFn&& loss_fn,
        const MixedPrecisionConfig& config = MixedPrecisionConfig::fp16()
    ) : model_(std::move(model)),
        optimizer_(std::move(optimizer)),
        loss_fn_(std::forward<LossFn>(loss_fn)),
        config_(config),
        scaler_(config.init_scale, config.growth_factor,
                config.backoff_factor, config.growth_interval),
        training_(true),
        skipped_steps_(0),
        total_steps_(0) {
        if (config_.use_master_weights && config_.enabled) {
            master_weights_ = std::make_unique<MasterWeightManager>(model_);
            // Swap optimizer to use FP32 master params instead of model's FP16 params
            optimizer_->replace_parameters(master_weights_->master_params());
        }
    }

    /**
     * @brief Perform single training step with mixed precision
     *
     * Executes complete mixed precision training iteration:
     * 1. Enable autocast for forward pass
     * 2. Forward pass: predictions = model(input) [FP16/BF16]
     * 3. Disable autocast for loss computation
     * 4. Loss computation: loss = loss_fn(predictions, target) [FP32]
     * 5. Scale loss: scaled_loss = scaler.scale(loss)
     * 6. Zero gradients: optimizer.zero_grad()
     * 7. Backward pass: scaled_loss.backward()
     * 8. Unscale gradients and check overflow
     * 9. Update parameters if no overflow: optimizer.step()
     * 10. Update loss scale: scaler.update()
     *
     * @param input Input batch tensor
     * @param target Target batch tensor
     * @return Loss value as float (unscaled)
     *
     * @par Complexity
     * - Time: O(forward + backward + update)
     * - Space: O(gradients)
     *
     * @code
     * Tensor inputs = get_batch_inputs();
     * Tensor targets = get_batch_targets();
     * float loss = trainer.train_step(inputs, targets);
     * std::cout << "Loss: " << loss << std::endl;
     * @endcode
     *
     * @note Returns unscaled loss for logging consistency
     * @note Automatically handles gradient overflow (skips update)
     */
    auto train_step(const Variable& input, const Variable& target) -> float;

    /**
     * @brief Perform evaluation step (no mixed precision)
     *
     * Executes evaluation in FP32 for accuracy:
     * 1. Set model to evaluation mode
     * 2. Disable gradients (NoGradGuard)
     * 3. Forward pass: predictions = model(input) [FP32]
     * 4. Loss computation: loss = loss_fn(predictions, target) [FP32]
     *
     * @param input Input batch tensor
     * @param target Target batch tensor
     * @return Loss value as float
     *
     * @par Complexity
     * - Time: O(forward)
     * - Space: O(1) - no gradients
     *
     * @code
     * float val_loss = trainer.eval_step(val_inputs, val_targets);
     * @endcode
     *
     * @note Evaluation uses FP32 for numerical accuracy
     */
    auto eval_step(const Variable& input, const Variable& target) -> float;

    /**
     * @brief Train model for multiple epochs with mixed precision
     *
     * Complete training loop with:
     * - Epoch iteration
     * - Mixed precision training batches
     * - Optional FP32 validation
     * - Callback invocation for monitoring
     * - Automatic overflow handling
     * - Loss scale tracking
     *
     * @param train_loader DataLoader for training data
     * @param epochs Number of epochs to train
     * @param val_loader Optional DataLoader for validation
     * @param callbacks Optional list of callbacks for monitoring
     *
     * @code
     * // Basic training
     * trainer.fit(train_loader, 10);
     *
     * // With validation
     * trainer.fit(train_loader, 10, &val_loader);
     *
     * // With callbacks
     * auto progress = std::make_shared<ProgressCallback>();
     * trainer.fit(train_loader, 10, &val_loader, {progress});
     * @endcode
     */
    auto fit(DataLoader& train_loader,
            int epochs,
            DataLoader* val_loader = nullptr,
            std::vector<std::shared_ptr<Callback>> callbacks = {}) -> void;

    /**
     * @brief Set model to training mode
     *
     * Enables training-specific behaviors and mixed precision.
     */
    auto train() -> void {
        training_ = true;
        model_->train();
    }

    /**
     * @brief Set model to evaluation mode
     *
     * Disables training-specific behaviors and mixed precision.
     */
    auto eval() -> void {
        training_ = false;
        model_->eval();
    }

    /**
     * @brief Check if model is in training mode
     */
    auto is_training() const -> bool {
        return training_;
    }

    /**
     * @brief Get underlying model
     */
    auto model() -> std::shared_ptr<Module> {
        return model_;
    }

    /**
     * @brief Get optimizer
     */
    auto optimizer() -> std::shared_ptr<optim::Optimizer> {
        return optimizer_;
    }

    /**
     * @brief Get gradient scaler
     */
    auto scaler() -> amp::GradScaler& {
        return scaler_;
    }

    /**
     * @brief Get current loss scale
     */
    auto get_scale() const -> float {
        return scaler_.get_scale();
    }

    /**
     * @brief Get number of skipped steps due to overflow
     */
    auto get_skipped_steps() const -> int {
        return skipped_steps_;
    }

    /**
     * @brief Get total number of training steps
     */
    auto get_total_steps() const -> int {
        return total_steps_;
    }

    /**
     * @brief Get mixed precision configuration
     */
    auto get_config() const -> const MixedPrecisionConfig& {
        return config_;
    }

    /**
     * @brief Reset training statistics
     */
    auto reset_stats() -> void {
        skipped_steps_ = 0;
        total_steps_ = 0;
    }

private:
    std::shared_ptr<Module> model_;                                        ///< Neural network model
    std::shared_ptr<optim::Optimizer> optimizer_;                          ///< Parameter optimizer
    std::function<Variable(const Variable&, const Variable&)> loss_fn_;    ///< Loss function
    MixedPrecisionConfig config_;                                          ///< Mixed precision configuration
    amp::GradScaler scaler_;                                               ///< Gradient scaler
    bool training_{true};                                                  ///< Training mode flag
    int skipped_steps_{0};                                                 ///< Number of skipped steps
    int total_steps_{0};                                                   ///< Total training steps
    std::unique_ptr<MasterWeightManager> master_weights_;                  ///< FP32 master copies (optional)
};

/**
 * @brief Helper function to create FP16 mixed precision trainer
 *
 * @code
 * auto trainer = create_fp16_trainer(model, optimizer, loss_fn);
 * @endcode
 */
inline auto create_fp16_trainer(
    std::shared_ptr<Module> model,
    std::shared_ptr<optim::Optimizer> optimizer,
    std::function<Variable(const Variable&, const Variable&)> loss_fn
) -> MixedPrecisionTrainer {
    return MixedPrecisionTrainer(model, optimizer, loss_fn,
                                  MixedPrecisionConfig::fp16());
}

/**
 * @brief Helper function to create BF16 mixed precision trainer
 *
 * @code
 * auto trainer = create_bfloat16_trainer(model, optimizer, loss_fn);
 * @endcode
 */
inline auto create_bfloat16_trainer(
    std::shared_ptr<Module> model,
    std::shared_ptr<optim::Optimizer> optimizer,
    std::function<Variable(const Variable&, const Variable&)> loss_fn
) -> MixedPrecisionTrainer {
    return MixedPrecisionTrainer(model, optimizer, loss_fn,
                                  MixedPrecisionConfig::bfloat16());
}

} // namespace nn
} // namespace tenzor
