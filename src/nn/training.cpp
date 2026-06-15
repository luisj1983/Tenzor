/**
 * @file training.cpp
 * @brief Implementation of high-level training API
 */

#include <tenzor/nn/training.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/reduction.hpp>
#include <stdexcept>
#include <iostream>
#include <iomanip>

namespace tenzor {
namespace nn {

NeuralNetwork::NeuralNetwork(std::shared_ptr<Module> model,
                             std::shared_ptr<optim::Optimizer> optimizer,
                             std::function<Variable(const Variable&, const Variable&)> loss_fn)
    : model_(std::move(model)),
      optimizer_(std::move(optimizer)),
      loss_fn_(std::move(loss_fn)),
      training_(true) {

    if (!model_) {
        throw std::invalid_argument("NeuralNetwork: model cannot be null");
    }
    if (!optimizer_) {
        throw std::invalid_argument("NeuralNetwork: optimizer cannot be null");
    }
    if (!loss_fn_) {
        throw std::invalid_argument("NeuralNetwork: loss_fn cannot be null");
    }
}

auto NeuralNetwork::train_step(const Variable& input, const Variable& target) -> float {
    // 1. Set model to training mode
    model_->train();

    // 2. Forward pass through model
    Variable predictions = model_->forward(input);

    // 3. Cache predictions tensor for metric evaluation
    last_predictions_ = predictions.tensor();

    // 4. Compute loss
    Variable loss = loss_fn_(predictions, target);

    // 5. Zero gradients before backward pass
    optimizer_->zero_grad();

    // 6. Backward pass - compute gradients.
    // The autograd engine requires a scalar root: it throws when numel() != 1.
    // A loss_fn configured with Reduction::None (or any non-scalar output)
    // would otherwise crash here, so reduce to a scalar via the Variable-level
    // autograd mean, preserving the grad_fn chain back to the parameters.
    Variable scalar_loss = (loss.tensor().numel() == 1) ? loss : tenzor::mean(loss);
    scalar_loss.backward();

    // 7. Update parameters
    optimizer_->step();

    // 8. Extract and return loss value as float
    // Handle both scalar and multi-element loss tensors
    const auto& loss_tensor = loss.tensor();
    if (loss_tensor.numel() == 1) {
        // Scalar loss - extract directly
        return loss_tensor.to(DType::Float32).item<float>();
    } else {
        // Multi-element loss - compute mean using dispatched reduction
        // (safe for GPU tensors — avoids invalid data_ptr() access on device memory)
        auto mean_tensor = tenzor::mean(loss_tensor);
        return mean_tensor.to(DType::Float32).item<float>();
    }
}

auto NeuralNetwork::eval_step(const Variable& input, const Variable& target) -> float {
    // 1. Set model to evaluation mode
    model_->eval();

    // 2. Disable gradient computation for efficiency
    NoGradGuard no_grad;

    // 3. Forward pass through model
    Variable predictions = model_->forward(input);

    // 4. Cache predictions tensor for metric evaluation
    last_predictions_ = predictions.tensor();

    // 5. Compute loss (no gradients)
    Variable loss = loss_fn_(predictions, target);

    // 6. Extract and return loss value as float
    const auto& loss_tensor = loss.tensor();
    if (loss_tensor.numel() == 1) {
        // Scalar loss - extract directly
        return loss_tensor.to(DType::Float32).item<float>();
    } else {
        // Multi-element loss - compute mean using dispatched reduction
        auto mean_tensor = tenzor::mean(loss_tensor);
        return mean_tensor.to(DType::Float32).item<float>();
    }
}

auto NeuralNetwork::fit(DataLoader& train_loader,
                       int epochs,
                       DataLoader* val_loader,
                       std::vector<std::shared_ptr<Callback>> callbacks) -> void {

    // Add default progress callback if no callbacks provided
    if (callbacks.empty()) {
        callbacks.push_back(std::make_shared<ProgressCallback>());
    }

    // === Callbacks: on_train_begin (audit G.9 — previously unfired) ===
    for (auto& callback : callbacks) {
        callback->on_train_begin();
    }

    // Main training loop
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // === Callbacks: on_epoch_begin ===
        for (auto& callback : callbacks) {
            callback->on_epoch_begin(epoch);
        }

        // === Training Phase ===
        model_->train();
        double running_train_loss = 0.0;
        int train_batch_count = 0;

        for (auto [inputs, targets] : train_loader) {
            // === Callbacks: on_batch_begin (audit G.9 — previously unfired) ===
            for (auto& callback : callbacks) {
                callback->on_batch_begin(train_batch_count);
            }

            // Convert tensors to variables
            Variable input_var(inputs, false);   // Input doesn't need gradients
            Variable target_var(targets, false); // Target doesn't need gradients

            // Perform training step
            float batch_loss = train_step(input_var, target_var);
            running_train_loss += batch_loss;
            train_batch_count++;

            // === Update metrics with batch predictions ===
            if (!metrics_.empty()) {
                for (auto& metric : metrics_) {
                    metric->update(last_predictions_, targets);
                }
            }

            // === Callbacks: on_batch_end ===
            for (auto& callback : callbacks) {
                callback->on_batch_end(train_batch_count - 1, batch_loss);
            }
        }

        // Compute average training loss
        double avg_train_loss = train_batch_count > 0
            ? running_train_loss / train_batch_count
            : 0.0;

        // === Validation Phase ===
        double avg_val_loss = 0.0;

        if (val_loader != nullptr) {
            model_->eval();
            double running_val_loss = 0.0;
            int val_batch_count = 0;

            for (auto [inputs, targets] : *val_loader) {
                // Convert tensors to variables
                Variable input_var(inputs, false);
                Variable target_var(targets, false);

                // Perform evaluation step
                float batch_loss = eval_step(input_var, target_var);
                running_val_loss += batch_loss;
                val_batch_count++;
            }

            // Compute average validation loss
            avg_val_loss = val_batch_count > 0
                ? running_val_loss / val_batch_count
                : 0.0;
        }

        // === Compute and log metrics at epoch end ===
        if (!metrics_.empty()) {
            std::cout << "  Metrics:";
            for (auto& metric : metrics_) {
                auto value = metric->compute();
                std::cout << " " << metric->name() << "="
                          << std::fixed << std::setprecision(4) << value.to(DType::Float32).item<float>();
                metric->reset();
            }
            std::cout << std::endl;
        }

        // === Callbacks: on_epoch_end ===
        for (auto& callback : callbacks) {
            callback->on_epoch_end(epoch, avg_train_loss, avg_val_loss);
        }
    }

    // === Callbacks: on_train_end (audit G.9 — previously unfired) ===
    for (auto& callback : callbacks) {
        callback->on_train_end();
    }

    // Final summary
    std::cout << "\nTraining completed successfully!" << std::endl;
}

} // namespace nn
} // namespace tenzor
