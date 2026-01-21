/**
 * @file mixed_precision.cpp
 * @brief Implementation of mixed precision training utilities
 */

#include <tenzor/nn/mixed_precision.hpp>
#include <iostream>

namespace tenzor {
namespace nn {

// Constructor is now template in header

auto MixedPrecisionTrainer::train_step(const Variable& input, const Variable& target) -> float {
    // Ensure model is in training mode
    model_->train();
    training_ = true;

    // Zero gradients from previous iteration
    optimizer_->zero_grad();

    Variable loss;
    float loss_value = 0.0f;

    if (config_.enabled) {
        // Mixed precision training path
        Variable output;

        // Forward pass in mixed precision (FP16/BF16)
        {
            amp::Autocast autocast(true, config_.dtype, config_.device_type);
            output = model_->forward(input);
        }

        // Loss computation in FP32 for numerical stability
        {
            amp::AutocastDisabled no_autocast;
            loss = loss_fn_(output, target);
        }

        // Store unscaled loss value for logging (convert to Float32 for item<float>)
        loss_value = loss.tensor().to(DType::Float32).item<float>();

        // Scale loss to prevent gradient underflow
        auto scaled_loss = scaler_.scale(loss);

        // Backward pass with scaled gradients
        scaled_loss.backward();

        // Unscale gradients, check for overflow, and update parameters
        bool step_successful = scaler_.step(*optimizer_);

        // Update loss scale for next iteration
        scaler_.update();

        // Track statistics
        total_steps_++;
        if (!step_successful) {
            skipped_steps_++;
        }
    } else {
        // Standard FP32 training path (fallback)
        auto output = model_->forward(input);
        loss = loss_fn_(output, target);
        loss_value = loss.tensor().to(DType::Float32).item<float>();

        // Backward pass
        loss.backward();

        // Update parameters
        optimizer_->step();

        total_steps_++;
    }

    return loss_value;
}

auto MixedPrecisionTrainer::eval_step(const Variable& input, const Variable& target) -> float {
    // Set model to evaluation mode
    model_->eval();
    training_ = false;

    // Disable gradient computation for evaluation
    NoGradGuard no_grad;

    // Forward pass in FP32 for numerical accuracy
    auto output = model_->forward(input);

    // Compute loss in FP32
    auto loss = loss_fn_(output, target);

    return loss.tensor().to(DType::Float32).item<float>();
}

auto MixedPrecisionTrainer::fit(
    DataLoader& train_loader,
    int epochs,
    DataLoader* val_loader,
    std::vector<std::shared_ptr<Callback>> callbacks
) -> void {
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Invoke epoch begin callbacks
        for (auto& callback : callbacks) {
            callback->on_epoch_begin(epoch);
        }

        // Training phase
        model_->train();
        training_ = true;
        float epoch_train_loss = 0.0f;
        int num_train_batches = 0;
        int skipped_in_epoch = 0;
        int initial_skipped = skipped_steps_;

        for (auto [batch_input, batch_target] : train_loader) {
            int batch_idx = num_train_batches;

            // Invoke batch begin callbacks
            for (auto& callback : callbacks) {
                callback->on_batch_begin(batch_idx);
            }

            // Perform training step (convert tensors to Variables)
            float loss = train_step(Variable(batch_input, false), Variable(batch_target, false));
            epoch_train_loss += loss;
            num_train_batches++;

            // Check if step was skipped due to overflow
            if (skipped_steps_ > initial_skipped + skipped_in_epoch) {
                skipped_in_epoch++;
            }

            // Invoke batch end callbacks
            for (auto& callback : callbacks) {
                callback->on_batch_end(batch_idx, loss);
            }
        }

        float avg_train_loss = num_train_batches > 0 ?
                               epoch_train_loss / num_train_batches : 0.0f;

        // Validation phase (if validation data provided)
        float avg_val_loss = 0.0f;
        if (val_loader != nullptr) {
            model_->eval();
            training_ = false;
            float epoch_val_loss = 0.0f;
            int num_val_batches = 0;

            for (auto [batch_input, batch_target] : *val_loader) {
                float loss = eval_step(Variable(batch_input, false), Variable(batch_target, false));
                epoch_val_loss += loss;
                num_val_batches++;
            }

            avg_val_loss = num_val_batches > 0 ?
                           epoch_val_loss / num_val_batches : 0.0f;
        }

        // Invoke epoch end callbacks
        for (auto& callback : callbacks) {
            callback->on_epoch_end(epoch, avg_train_loss, avg_val_loss);
        }

        // Print epoch summary
        std::cout << "Epoch " << epoch + 1 << "/" << epochs
                  << " - train_loss: " << avg_train_loss;

        if (val_loader != nullptr) {
            std::cout << " - val_loss: " << avg_val_loss;
        }

        // Print mixed precision statistics
        if (config_.enabled) {
            std::cout << " - scale: " << scaler_.get_scale()
                      << " - skipped: " << skipped_in_epoch
                      << "/" << num_train_batches;
        }

        std::cout << std::endl;
    }
}

} // namespace nn
} // namespace tenzor
