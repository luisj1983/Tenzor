/**
 * @file mixed_precision.cpp
 * @brief Implementation of mixed precision training utilities
 */

#include <tenzor/nn/mixed_precision.hpp>
#include <iostream>

namespace tenzor {
namespace nn {

// ============================================================================
// MasterWeightManager
// ============================================================================

MasterWeightManager::MasterWeightManager(std::shared_ptr<Module> model)
    : model_(std::move(model)) {
    auto params = model_->parameters();
    master_variables_.reserve(params.size());
    working_refs_.reserve(params.size());

    for (auto& param : params) {
        working_refs_.push_back(param);
        // Create FP32 master copy
        Tensor master_data = param->tensor().to(DType::Float32).clone();
        auto master_var = std::make_shared<Variable>(master_data, true);
        master_variables_.push_back(master_var);
    }
}

auto MasterWeightManager::sync_to_working() -> void {
    for (size_t i = 0; i < master_variables_.size(); ++i) {
        auto& working = working_refs_[i];
        auto& master = master_variables_[i];
        // Copy FP32 master -> working precision (FP16/BF16) in-place
        Tensor converted = master->tensor().to(working->dtype());
        working->tensor().zero_();
        working->tensor() += converted;
    }
}

auto MasterWeightManager::sync_from_working() -> void {
    // When using replace_parameters(), the optimizer updates master params directly
    // in FP32. This method copies gradients from the working (FP16) params to the
    // master (FP32) params so the optimizer can use them.
    for (size_t i = 0; i < master_variables_.size(); ++i) {
        auto& working = working_refs_[i];
        auto& master = master_variables_[i];
        if (working->has_grad()) {
            master->set_grad(working->grad()->to(DType::Float32));
        }
    }
}

auto MasterWeightManager::master_params() -> std::vector<std::shared_ptr<Variable>>& {
    return master_variables_;
}

// ============================================================================
// MixedPrecisionTrainer
// ============================================================================

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

        // Copy gradients to master weights if using master weights
        if (master_weights_) {
            master_weights_->sync_from_working();
        }

        // Unscale gradients, check for overflow, and update FP32 master parameters
        bool step_successful = scaler_.step(*optimizer_);

        // Sync FP32 master weights -> model's working FP16 params after update
        if (master_weights_ && step_successful) {
            master_weights_->sync_to_working();
        }

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
