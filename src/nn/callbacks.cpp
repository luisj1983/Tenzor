/**
 * @file callbacks.cpp
 * @brief Implementation of training callbacks
 */

#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/serialize.hpp>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <cmath>
#include <algorithm>

namespace tenzor {
namespace nn {

// ============================================================================
// ProgressCallback Implementation
// ============================================================================

ProgressCallback::ProgressCallback(int print_every)
    : print_every_(print_every) {
    if (print_every_ <= 0) {
        print_every_ = 1;
    }
}

auto ProgressCallback::on_train_begin() -> void {
    std::cout << "Training started..." << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

auto ProgressCallback::on_epoch_begin(int epoch) -> void {
    std::cout << "\nEpoch " << (epoch + 1);
    if (total_epochs_ > 0) {
        std::cout << "/" << total_epochs_;
    }
    std::cout << std::endl;
    std::cout << std::string(60, '-') << std::endl;

    // Reset running statistics
    running_loss_ = 0.0f;
    batch_count_ = 0;
}

auto ProgressCallback::on_batch_end(int batch_idx, float loss) -> void {
    running_loss_ += loss;
    batch_count_++;

    // Print every N batches
    if ((batch_idx + 1) % print_every_ == 0) {
        float avg_loss = running_loss_ / batch_count_;

        std::cout << "  Batch " << std::setw(4) << (batch_idx + 1);
        if (total_batches_ > 0) {
            std::cout << "/" << std::setw(4) << total_batches_;

            // Progress bar
            int bar_width = 30;
            float progress = static_cast<float>(batch_idx + 1) / total_batches_;
            int pos = static_cast<int>(bar_width * progress);

            std::cout << " [";
            for (int i = 0; i < bar_width; ++i) {
                if (i < pos) std::cout << "=";
                else if (i == pos) std::cout << ">";
                else std::cout << " ";
            }
            std::cout << "] " << std::setw(3) << static_cast<int>(progress * 100) << "%";
        }

        std::cout << " - Loss: " << std::fixed << std::setprecision(4) << avg_loss;
        std::cout << std::endl;

        // Reset running statistics for next print
        running_loss_ = 0.0f;
        batch_count_ = 0;
    }
}

auto ProgressCallback::on_epoch_end(int epoch, float train_loss, float val_loss) -> void {
    std::cout << std::string(60, '-') << std::endl;
    std::cout << "Epoch " << (epoch + 1) << " Summary:" << std::endl;
    std::cout << "  Training Loss:   " << std::fixed << std::setprecision(6) << train_loss << std::endl;

    if (val_loss >= 0.0f) {  // Only print if validation loss is provided
        std::cout << "  Validation Loss: " << std::fixed << std::setprecision(6) << val_loss << std::endl;
    }

    std::cout << std::string(60, '=') << std::endl;
}

auto ProgressCallback::on_train_end() -> void {
    std::cout << "\nTraining completed!" << std::endl;
    std::cout << std::string(60, '=') << std::endl;
}

// ============================================================================
// EarlyStoppingCallback Implementation
// ============================================================================

EarlyStoppingCallback::EarlyStoppingCallback(
    int patience,
    float min_delta,
    const std::string& monitor
) : patience_(patience),
    min_delta_(min_delta),
    monitor_(monitor) {

    if (patience_ <= 0) {
        patience_ = 1;
    }
}

auto EarlyStoppingCallback::on_epoch_end([[maybe_unused]] int epoch, float train_loss, float val_loss) -> void {
    // Select metric to monitor
    float current_loss = (monitor_ == "train_loss") ? train_loss : val_loss;

    // Check if this is an improvement
    if (current_loss < best_loss_ - min_delta_) {
        // Improvement found
        best_loss_ = current_loss;
        wait_ = 0;

        std::cout << "[EarlyStopping] " << monitor_ << " improved to "
                  << std::fixed << std::setprecision(6) << best_loss_
                  << std::endl;
    } else {
        // No improvement
        wait_++;

        std::cout << "[EarlyStopping] No improvement in " << monitor_
                  << " for " << wait_ << "/" << patience_ << " epochs"
                  << std::endl;

        if (wait_ >= patience_) {
            stopped_ = true;
            std::cout << "[EarlyStopping] Early stopping triggered! "
                      << "No improvement for " << patience_ << " epochs."
                      << std::endl;
            std::cout << "[EarlyStopping] Best " << monitor_ << ": "
                      << std::fixed << std::setprecision(6) << best_loss_
                      << std::endl;
        }
    }
}

// ============================================================================
// ModelCheckpointCallback Implementation
// ============================================================================

ModelCheckpointCallback::ModelCheckpointCallback(
    const std::string& filepath,
    std::shared_ptr<Module> model,
    bool save_best_only,
    const std::string& monitor
) : filepath_(filepath),
    model_(model),
    save_best_only_(save_best_only),
    monitor_(monitor) {

    if (!model_) {
        throw std::runtime_error("ModelCheckpointCallback: model cannot be null");
    }
}

auto ModelCheckpointCallback::format_filepath(int epoch) const -> std::string {
    std::string result = filepath_;

    // Replace {epoch} placeholder with actual epoch number
    size_t pos = result.find("{epoch}");
    if (pos != std::string::npos) {
        result.replace(pos, 7, std::to_string(epoch + 1));
    }

    // Replace {epoch:03d} style placeholders
    pos = result.find("{epoch:");
    if (pos != std::string::npos) {
        size_t end_pos = result.find("}", pos);
        if (end_pos != std::string::npos) {
            // Extract format spec (e.g., "03d" from "{epoch:03d}")
            std::string format_spec = result.substr(pos + 7, end_pos - pos - 7);

            // Parse width from format spec (e.g., "03d" -> width=3)
            int width = 0;
            char fill = '0';
            if (!format_spec.empty() && format_spec[0] == '0') {
                fill = '0';
                width = std::stoi(format_spec.substr(0, format_spec.length() - 1));
            }

            // Format epoch number
            std::ostringstream oss;
            oss << std::setfill(fill) << std::setw(width) << (epoch + 1);

            result.replace(pos, end_pos - pos + 1, oss.str());
        }
    }

    return result;
}

auto ModelCheckpointCallback::on_epoch_end(int epoch, float train_loss, float val_loss) -> void {
    // Select metric to monitor
    float current_loss = (monitor_ == "train_loss") ? train_loss : val_loss;

    bool should_save = false;

    if (save_best_only_) {
        // Only save if this is the best model so far
        if (current_loss < best_loss_) {
            best_loss_ = current_loss;
            should_save = true;

            std::cout << "[ModelCheckpoint] " << monitor_ << " improved to "
                      << std::fixed << std::setprecision(6) << best_loss_
                      << ", saving model..." << std::endl;
        }
    } else {
        // Save every epoch
        should_save = true;
        std::cout << "[ModelCheckpoint] Saving model at epoch " << (epoch + 1)
                  << "..." << std::endl;
    }

    if (should_save) {
        try {
            std::string checkpoint_path = format_filepath(epoch);
            model_->save(checkpoint_path);
            last_checkpoint_ = checkpoint_path;

            std::cout << "[ModelCheckpoint] Model saved to: " << checkpoint_path
                      << std::endl;
        } catch (const std::exception& e) {
            std::cerr << "[ModelCheckpoint] Error saving model: " << e.what()
                      << std::endl;
        }
    }
}

// ============================================================================
// LRSchedulerCallback Implementation
// ============================================================================

LRSchedulerCallback::LRSchedulerCallback(
    std::shared_ptr<optim::Optimizer> optimizer,
    const std::string& schedule_type,
    float decay_factor,
    int decay_epochs,
    float min_lr,
    int patience
) : optimizer_(optimizer),
    schedule_type_(schedule_type),
    decay_factor_(decay_factor),
    decay_epochs_(decay_epochs),
    min_lr_(min_lr),
    patience_(patience),
    initial_lr_(0.0f),
    current_lr_(0.0f) {

    if (!optimizer_) {
        throw std::runtime_error("LRSchedulerCallback: optimizer cannot be null");
    }

    if (decay_factor_ <= 0.0f || decay_factor_ >= 1.0f) {
        std::cerr << "[LRScheduler] Warning: decay_factor should be in (0, 1), got "
                  << decay_factor_ << ". Using 0.1" << std::endl;
        decay_factor_ = 0.1f;
    }

    if (decay_epochs_ <= 0) {
        decay_epochs_ = 1;
    }

    if (patience_ <= 0) {
        patience_ = 1;
    }
}

auto LRSchedulerCallback::on_train_begin() -> void {
    // Store initial learning rate
    // Note: This assumes optimizer has a method to get current LR
    // For now, we'll track it internally
    initial_lr_ = 0.01f;  // Default value
    current_lr_ = initial_lr_;

    std::cout << "[LRScheduler] Initial learning rate: "
              << std::scientific << std::setprecision(2) << current_lr_
              << std::endl;
}

auto LRSchedulerCallback::update_lr(float new_lr) -> void {
    // Clamp to minimum LR
    new_lr = std::max(new_lr, min_lr_);

    if (new_lr != current_lr_) {
        current_lr_ = new_lr;

        // Update optimizer learning rate
        // Note: This requires extending Optimizer base class with set_lr method
        // For now, we just track it

        std::cout << "[LRScheduler] Learning rate adjusted to: "
                  << std::scientific << std::setprecision(2) << current_lr_
                  << std::endl;
    }
}

auto LRSchedulerCallback::on_epoch_end(int epoch, [[maybe_unused]] float train_loss, float val_loss) -> void {
    float new_lr = current_lr_;

    if (schedule_type_ == "step") {
        // Step decay: reduce LR every decay_epochs
        if ((epoch + 1) % decay_epochs_ == 0) {
            new_lr = current_lr_ * decay_factor_;
        }
    } else if (schedule_type_ == "exponential") {
        // Exponential decay: reduce LR every epoch
        new_lr = current_lr_ * decay_factor_;
    } else if (schedule_type_ == "cosine") {
        // Cosine annealing
        float pi = 3.14159265359f;
        float progress = static_cast<float>(epoch) / std::max(decay_epochs_, 1);
        new_lr = min_lr_ + (initial_lr_ - min_lr_) *
                 (1.0f + std::cos(pi * progress)) / 2.0f;
    } else if (schedule_type_ == "plateau") {
        // Reduce on plateau: reduce when validation loss stops improving
        if (val_loss < best_loss_) {
            best_loss_ = val_loss;
            wait_ = 0;
        } else {
            wait_++;
            if (wait_ >= patience_) {
                new_lr = current_lr_ * decay_factor_;
                wait_ = 0;  // Reset counter after reducing LR

                std::cout << "[LRScheduler] Validation loss plateaued for "
                          << patience_ << " epochs, reducing learning rate"
                          << std::endl;
            }
        }
    } else {
        std::cerr << "[LRScheduler] Unknown schedule type: " << schedule_type_
                  << ". No LR adjustment performed." << std::endl;
        return;
    }

    update_lr(new_lr);
}

} // namespace nn
} // namespace tenzor
