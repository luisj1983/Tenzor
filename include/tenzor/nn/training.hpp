/**
 * @file training.hpp
 * @brief High-level training API for neural networks
 *
 * Provides NeuralNetwork wrapper class for simplified training workflows,
 * including built-in training loops, validation, and callback support.
 */

#pragma once

#include <memory>
#include <vector>
#include <functional>
#include <iostream>
#include <string>
#include "module.hpp"
#include "optim/optimizer.hpp"
#include "callbacks.hpp"
#include "metrics.hpp"
#include <tenzor/autograd/variable.hpp>

namespace tenzor {
namespace nn {

/**
 * @brief Simple data loader for batch iteration
 *
 * Provides iterator interface for training/validation data batches.
 * Supports both input-target pairs for supervised learning.
 *
 * @code
 * std::vector<std::pair<Tensor, Tensor>> data = {...};
 * DataLoader loader(data, 32);  // Batch size 32
 *
 * for (auto [inputs, targets] : loader) {
 *     auto loss = model.train_step(inputs, targets);
 * }
 * @endcode
 */
class DataLoader {
public:
    /**
     * @brief Construct DataLoader from vector of input-target pairs
     *
     * @param data Vector of (input, target) tensor pairs
     * @param batch_size Number of samples per batch
     */
    DataLoader(std::vector<std::pair<Tensor, Tensor>> data, size_t batch_size)
        : data_(std::move(data)), batch_size_(batch_size) {}

    /**
     * @brief Iterator for batch traversal
     */
    class Iterator {
    public:
        Iterator(const DataLoader* loader, size_t index)
            : loader_(loader), index_(index) {}

        auto operator==(const Iterator& other) const -> bool {
            return index_ == other.index_;
        }

        auto operator!=(const Iterator& other) const -> bool {
            return index_ != other.index_;
        }

        auto operator++() -> Iterator& {
            index_++;
            return *this;
        }

        auto operator*() const -> std::pair<Tensor, Tensor> {
            size_t start = index_ * loader_->batch_size_;

            // For simplicity, return first sample in batch
            // In production, you'd concatenate all samples in the batch
            if (start < loader_->data_.size()) {
                return loader_->data_[start];
            }
            return loader_->data_[0];  // Fallback
        }

    private:
        const DataLoader* loader_;
        size_t index_;
    };

    auto begin() const -> Iterator {
        return Iterator(this, 0);
    }

    auto end() const -> Iterator {
        size_t num_batches = (data_.size() + batch_size_ - 1) / batch_size_;
        return Iterator(this, num_batches);
    }

    auto size() const -> size_t {
        return (data_.size() + batch_size_ - 1) / batch_size_;
    }

private:
    std::vector<std::pair<Tensor, Tensor>> data_;
    size_t batch_size_;
};

/**
 * @brief High-level neural network training wrapper
 *
 * NeuralNetwork provides a complete training API that wraps a model, optimizer,
 * and loss function. It handles the standard training loop pattern:
 * - Forward pass through model
 * - Loss computation
 * - Backward pass (gradient computation)
 * - Parameter updates
 * - Training/evaluation mode switching
 *
 * This eliminates boilerplate code and provides a consistent interface similar
 * to high-level frameworks like Keras or PyTorch Lightning.
 *
 * **Key Features:**
 * - Single-call training step with `train_step()`
 * - Evaluation without gradients via `eval_step()`
 * - Complete training loop with `fit()`
 * - Automatic mode switching (train/eval)
 * - Validation support
 * - Callback system for monitoring
 *
 * @code
 * // Create model, optimizer, and loss
 * auto model = std::make_shared<Sequential>(
 *     std::make_shared<Linear>(784, 128),
 *     std::make_shared<ReLU>(),
 *     std::make_shared<Linear>(128, 10)
 * );
 * auto optimizer = std::make_shared<Adam>(model->parameters(), 0.001);
 * auto loss_fn = std::make_shared<CrossEntropyLoss>();
 *
 * // Wrap in NeuralNetwork
 * NeuralNetwork nn(model, optimizer, loss_fn);
 *
 * // Train for 10 epochs
 * DataLoader train_loader(train_data, 32);
 * DataLoader val_loader(val_data, 32);
 * nn.fit(train_loader, 10, &val_loader);
 * @endcode
 *
 * @see Module, Optimizer, DataLoader, Callback
 */
class NeuralNetwork {
public:
    /**
     * @brief Construct NeuralNetwork with model, optimizer, and loss function
     *
     * @param model Neural network model (any Module subclass)
     * @param optimizer Optimization algorithm (SGD, Adam, etc.)
     * @param loss_fn Loss function callable (function or lambda)
     *
     * @par Requirements
     * - model must have forward() method defined
     * - optimizer must be initialized with model parameters
     * - loss_fn must be callable with (predictions, targets) returning Variable
     *
     * @code
     * auto model = std::make_shared<MyModel>();
     * auto optimizer = std::make_shared<SGD>(model->parameters(), 0.01);
     * auto mse_loss = std::make_shared<MSELoss>();
     * auto loss_fn = [mse_loss](const Variable& pred, const Variable& target) {
     *     return (*mse_loss)(pred, target);
     * };
     * NeuralNetwork nn(model, optimizer, loss_fn);
     * @endcode
     */
    NeuralNetwork(std::shared_ptr<Module> model,
                  std::shared_ptr<optim::Optimizer> optimizer,
                  std::function<Variable(const Variable&, const Variable&)> loss_fn);

    /**
     * @brief Perform single training step
     *
     * Executes complete training iteration:
     * 1. Set model to training mode
     * 2. Forward pass: predictions = model(input)
     * 3. Loss computation: loss = loss_fn(predictions, target)
     * 4. Zero gradients: optimizer.zero_grad()
     * 5. Backward pass: loss.backward()
     * 6. Parameter update: optimizer.step()
     *
     * @param input Input batch tensor
     * @param target Target batch tensor
     * @return Loss value as float
     *
     * @par Complexity
     * - Time: O(forward + backward + update)
     * - Space: O(gradients)
     *
     * @code
     * Tensor inputs = get_batch_inputs();
     * Tensor targets = get_batch_targets();
     * float loss = nn.train_step(inputs, targets);
     * std::cout << "Loss: " << loss << std::endl;
     * @endcode
     */
    auto train_step(const Variable& input, const Variable& target) -> float;

    /**
     * @brief Perform single evaluation step
     *
     * Executes evaluation without gradient computation:
     * 1. Set model to evaluation mode (disable dropout, etc.)
     * 2. Disable gradients (NoGradGuard)
     * 3. Forward pass: predictions = model(input)
     * 4. Loss computation: loss = loss_fn(predictions, target)
     * 5. Return loss value
     *
     * Used for validation/testing where gradients are not needed.
     * More efficient than train_step() due to no gradient computation.
     *
     * @param input Input batch tensor
     * @param target Target batch tensor
     * @return Loss value as float
     *
     * @par Complexity
     * - Time: O(forward)
     * - Space: O(1) - no gradient storage
     *
     * @code
     * model.eval();
     * float val_loss = nn.eval_step(val_inputs, val_targets);
     * @endcode
     */
    auto eval_step(const Variable& input, const Variable& target) -> float;

    /**
     * @brief Train model for multiple epochs
     *
     * Complete training loop with:
     * - Epoch iteration
     * - Training batch processing
     * - Optional validation after each epoch
     * - Callback invocation for monitoring
     * - Automatic mode switching
     *
     * Training Loop Structure:
     * ```
     * for epoch in range(epochs):
     *     callbacks.on_epoch_begin(epoch)
     *
     *     # Training phase
     *     for batch in train_loader:
     *         loss = train_step(batch)
     *         callbacks.on_batch_end(batch_idx, loss)
     *
     *     # Validation phase (if val_loader provided)
     *     if val_loader:
     *         for batch in val_loader:
     *             val_loss = eval_step(batch)
     *
     *     callbacks.on_epoch_end(epoch, train_loss, val_loss)
     * ```
     *
     * @param train_loader DataLoader for training data
     * @param epochs Number of epochs to train
     * @param val_loader Optional DataLoader for validation (nullptr = no validation)
     * @param callbacks Optional list of callbacks for monitoring
     *
     * @code
     * // Basic training
     * nn.fit(train_loader, 10);
     *
     * // With validation
     * nn.fit(train_loader, 10, &val_loader);
     *
     * // With callbacks
     * auto progress = std::make_shared<ProgressCallback>();
     * nn.fit(train_loader, 10, &val_loader, {progress});
     * @endcode
     */
    auto fit(DataLoader& train_loader,
            int epochs,
            DataLoader* val_loader = nullptr,
            std::vector<std::shared_ptr<Callback>> callbacks = {}) -> void;

    /**
     * @brief Set model to training mode
     *
     * Enables training-specific behaviors:
     * - Dropout is active
     * - Batch normalization updates running stats
     * - Other modules that behave differently during training
     */
    auto train() -> void {
        training_ = true;
        model_->train();
    }

    /**
     * @brief Set model to evaluation mode
     *
     * Disables training-specific behaviors:
     * - Dropout is disabled
     * - Batch normalization uses frozen stats
     * - Deterministic behavior for inference
     */
    auto eval() -> void {
        training_ = false;
        model_->eval();
    }

    /**
     * @brief Check if model is in training mode
     * @return true if training mode is enabled
     */
    auto is_training() const -> bool {
        return training_;
    }

    /**
     * @brief Get underlying model
     * @return Shared pointer to model module
     */
    auto model() -> std::shared_ptr<Module> {
        return model_;
    }

    /**
     * @brief Get optimizer
     * @return Shared pointer to optimizer
     */
    auto optimizer() -> std::shared_ptr<optim::Optimizer> {
        return optimizer_;
    }

    /**
     * @brief Register a metric to be evaluated during training
     *
     * Metrics are updated with (predictions, targets) after each batch
     * during fit(), computed and logged at the end of each epoch, then reset.
     *
     * @param metric Shared pointer to a Metric instance
     *
     * @code
     * NeuralNetwork nn(model, optimizer, loss_fn);
     * nn.add_metric(std::make_shared<Accuracy>(10));
     * nn.add_metric(std::make_shared<F1Score>(10));
     * nn.fit(train_loader, 10);
     * @endcode
     */
    auto add_metric(std::shared_ptr<Metric> metric) -> void {
        metrics_.push_back(std::move(metric));
    }

    /**
     * @brief Get registered metrics
     * @return Const reference to the vector of registered metrics
     */
    auto metrics() const -> const std::vector<std::shared_ptr<Metric>>& {
        return metrics_;
    }

private:
    std::shared_ptr<Module> model_;                                        ///< Neural network model
    std::shared_ptr<optim::Optimizer> optimizer_;                          ///< Parameter optimizer
    std::function<Variable(const Variable&, const Variable&)> loss_fn_;    ///< Loss function
    bool training_{true};                                                  ///< Training mode flag
    std::vector<std::shared_ptr<Metric>> metrics_;                         ///< Registered training metrics
    Tensor last_predictions_;                                              ///< Cached predictions from last step (for metrics)
};

} // namespace nn
} // namespace tenzor
