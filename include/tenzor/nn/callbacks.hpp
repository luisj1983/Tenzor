/**
 * @file callbacks.hpp
 * @brief Training callback system for monitoring and controlling training loops
 *
 * Provides extensible callback interface for training customization including:
 * - Progress monitoring and logging
 * - Early stopping based on validation metrics
 * - Model checkpointing (save best/periodic models)
 * - Learning rate scheduling
 */

#pragma once

#include <memory>
#include <string>
#include <vector>
#include <limits>
#include <functional>
#include "module.hpp"
#include "optim/optimizer.hpp"

namespace tenzor {
namespace nn {

/**
 * @brief Base callback interface for training loop hooks
 *
 * Callbacks allow custom behavior at specific points during training:
 * - Beginning and end of training
 * - Beginning and end of each epoch
 * - Beginning and end of each batch
 *
 * Derive from this class to implement custom training behavior.
 *
 * @code
 * class CustomCallback : public Callback {
 *     auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override {
 *         std::cout << "Epoch " << epoch << " complete!\n";
 *     }
 * };
 * @endcode
 */
class Callback {
public:
    virtual ~Callback() = default;

    /**
     * @brief Called at the beginning of each epoch
     * @param epoch Current epoch number (0-indexed)
     */
    virtual auto on_epoch_begin(int epoch) -> void {}

    /**
     * @brief Called at the end of each epoch
     * @param epoch Current epoch number (0-indexed)
     * @param train_loss Average training loss for the epoch
     * @param val_loss Average validation loss for the epoch
     */
    virtual auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void {}

    /**
     * @brief Called at the beginning of each batch
     * @param batch_idx Current batch index within epoch
     */
    virtual auto on_batch_begin(int batch_idx) -> void {}

    /**
     * @brief Called at the end of each batch
     * @param batch_idx Current batch index within epoch
     * @param loss Loss value for this batch
     */
    virtual auto on_batch_end(int batch_idx, float loss) -> void {}

    /**
     * @brief Called at the beginning of training
     */
    virtual auto on_train_begin() -> void {}

    /**
     * @brief Called at the end of training
     */
    virtual auto on_train_end() -> void {}
};

/**
 * @brief Callback for printing training progress
 *
 * Displays training progress including:
 * - Current epoch information
 * - Batch progress within epoch
 * - Training and validation losses
 *
 * @code
 * auto progress = std::make_shared<ProgressCallback>(10); // Print every 10 batches
 * progress->set_total_batches(100);
 * trainer.fit(train_loader, val_loader, {progress});
 * @endcode
 */
class ProgressCallback : public Callback {
public:
    /**
     * @brief Construct progress callback
     * @param print_every Print progress every N batches (default: 1)
     */
    explicit ProgressCallback(int print_every = 1);

    /**
     * @brief Set total number of batches per epoch for progress display
     * @param total Total batches per epoch
     */
    auto set_total_batches(int total) -> void { total_batches_ = total; }

    /**
     * @brief Set total number of epochs for progress display
     * @param total Total epochs for training
     */
    auto set_total_epochs(int total) -> void { total_epochs_ = total; }

    auto on_epoch_begin(int epoch) -> void override;
    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override;
    auto on_batch_end(int batch_idx, float loss) -> void override;
    auto on_train_begin() -> void override;
    auto on_train_end() -> void override;

private:
    int print_every_;
    int total_batches_{0};
    int total_epochs_{0};
    float running_loss_{0.0f};
    int batch_count_{0};
};

/**
 * @brief Callback for early stopping based on validation loss
 *
 * Monitors validation loss and stops training when no improvement is seen
 * for a specified number of epochs (patience).
 *
 * @code
 * auto early_stop = std::make_shared<EarlyStoppingCallback>(
 *     5,      // patience: stop after 5 epochs without improvement
 *     0.001   // min_delta: minimum change to qualify as improvement
 * );
 *
 * for (int epoch = 0; epoch < max_epochs; ++epoch) {
 *     // ... training code ...
 *     early_stop->on_epoch_end(epoch, train_loss, val_loss);
 *     if (early_stop->should_stop()) {
 *         std::cout << "Early stopping triggered!\n";
 *         break;
 *     }
 * }
 * @endcode
 */
class EarlyStoppingCallback : public Callback {
public:
    /**
     * @brief Construct early stopping callback
     * @param patience Number of epochs with no improvement before stopping
     * @param min_delta Minimum change in validation loss to qualify as improvement
     * @param monitor Metric to monitor: "val_loss" (default) or "train_loss"
     */
    explicit EarlyStoppingCallback(
        int patience = 5,
        float min_delta = 0.0f,
        const std::string& monitor = "val_loss"
    );

    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override;

    /**
     * @brief Check if training should stop
     * @return true if stopping criteria met
     */
    auto should_stop() const -> bool { return stopped_; }

    /**
     * @brief Get best loss value seen so far
     * @return Best validation loss
     */
    auto best_loss() const -> float { return best_loss_; }

    /**
     * @brief Get number of epochs since last improvement
     * @return Wait counter value
     */
    auto wait_count() const -> int { return wait_; }

private:
    int patience_;
    float min_delta_;
    std::string monitor_;
    float best_loss_{std::numeric_limits<float>::max()};
    int wait_{0};
    bool stopped_{false};
};

/**
 * @brief Callback for saving model checkpoints during training
 *
 * Saves model state to disk either:
 * - After every epoch (save_best_only=false)
 * - Only when validation loss improves (save_best_only=true)
 *
 * Checkpoint files include model parameters and can be loaded later.
 *
 * @code
 * auto checkpoint = std::make_shared<ModelCheckpointCallback>(
 *     "model_epoch_{epoch:03d}.pt",  // filepath template
 *     model,                          // model to save
 *     true                            // save_best_only
 * );
 * @endcode
 */
class ModelCheckpointCallback : public Callback {
public:
    /**
     * @brief Construct model checkpoint callback
     * @param filepath Path template for checkpoint files (can include {epoch})
     * @param model Model to save (shared pointer)
     * @param save_best_only If true, only save when validation loss improves
     * @param monitor Metric to monitor for best model: "val_loss" or "train_loss"
     */
    explicit ModelCheckpointCallback(
        const std::string& filepath,
        std::shared_ptr<Module> model,
        bool save_best_only = true,
        const std::string& monitor = "val_loss"
    );

    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override;

    /**
     * @brief Get best loss value for saved model
     * @return Best loss value
     */
    auto best_loss() const -> float { return best_loss_; }

    /**
     * @brief Get path of last saved checkpoint
     * @return Filepath of last checkpoint
     */
    auto last_checkpoint() const -> std::string { return last_checkpoint_; }

private:
    std::string filepath_;
    std::shared_ptr<Module> model_;
    bool save_best_only_;
    std::string monitor_;
    float best_loss_{std::numeric_limits<float>::max()};
    std::string last_checkpoint_;

    auto format_filepath(int epoch) const -> std::string;
};

/**
 * @brief Callback for adjusting learning rate during training
 *
 * Implements various learning rate scheduling strategies:
 * - "step": Multiply LR by factor every N epochs
 * - "exponential": Multiply LR by factor every epoch
 * - "cosine": Cosine annealing schedule
 * - "plateau": Reduce LR when metric plateaus (validation loss)
 *
 * @code
 * auto scheduler = std::make_shared<LRSchedulerCallback>(
 *     optimizer,
 *     "step",      // schedule type
 *     0.1,         // decay factor
 *     10           // decay every 10 epochs
 * );
 * @endcode
 */
class LRSchedulerCallback : public Callback {
public:
    /**
     * @brief Construct learning rate scheduler callback
     * @param optimizer Optimizer to adjust learning rate for
     * @param schedule_type Type of schedule: "step", "exponential", "cosine", "plateau"
     * @param decay_factor Factor to multiply learning rate by
     * @param decay_epochs For "step": decay every N epochs. For "cosine": total epochs
     * @param min_lr Minimum learning rate (for cosine/plateau schedules)
     * @param patience For "plateau": epochs to wait before reducing LR
     */
    explicit LRSchedulerCallback(
        std::shared_ptr<optim::Optimizer> optimizer,
        const std::string& schedule_type = "step",
        float decay_factor = 0.1f,
        int decay_epochs = 10,
        float min_lr = 0.0f,
        int patience = 5
    );

    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override;
    auto on_train_begin() -> void override;

    /**
     * @brief Get current learning rate
     * @return Current LR value
     */
    auto current_lr() const -> float { return current_lr_; }

private:
    std::shared_ptr<optim::Optimizer> optimizer_;
    std::string schedule_type_;
    float decay_factor_;
    int decay_epochs_;
    float min_lr_;
    int patience_;
    float initial_lr_;
    float current_lr_;
    float best_loss_{std::numeric_limits<float>::max()};
    int wait_{0};

    auto update_lr(float new_lr) -> void;
};

/**
 * @brief Collection of callbacks for training
 *
 * Convenience container for managing multiple callbacks.
 * Automatically calls all callbacks at appropriate hook points.
 *
 * @code
 * CallbackList callbacks;
 * callbacks.add(std::make_shared<ProgressCallback>());
 * callbacks.add(std::make_shared<EarlyStoppingCallback>());
 * callbacks.on_epoch_end(0, 0.5f, 0.4f);
 * @endcode
 */
class CallbackList {
public:
    /**
     * @brief Add a callback to the list
     * @param callback Callback to add
     */
    auto add(std::shared_ptr<Callback> callback) -> void {
        callbacks_.push_back(callback);
    }

    /**
     * @brief Call on_epoch_begin for all callbacks
     */
    auto on_epoch_begin(int epoch) -> void {
        for (auto& cb : callbacks_) cb->on_epoch_begin(epoch);
    }

    /**
     * @brief Call on_epoch_end for all callbacks
     */
    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void {
        for (auto& cb : callbacks_) cb->on_epoch_end(epoch, train_loss, val_loss);
    }

    /**
     * @brief Call on_batch_begin for all callbacks
     */
    auto on_batch_begin(int batch_idx) -> void {
        for (auto& cb : callbacks_) cb->on_batch_begin(batch_idx);
    }

    /**
     * @brief Call on_batch_end for all callbacks
     */
    auto on_batch_end(int batch_idx, float loss) -> void {
        for (auto& cb : callbacks_) cb->on_batch_end(batch_idx, loss);
    }

    /**
     * @brief Call on_train_begin for all callbacks
     */
    auto on_train_begin() -> void {
        for (auto& cb : callbacks_) cb->on_train_begin();
    }

    /**
     * @brief Call on_train_end for all callbacks
     */
    auto on_train_end() -> void {
        for (auto& cb : callbacks_) cb->on_train_end();
    }

    /**
     * @brief Get all callbacks
     */
    auto callbacks() const -> const std::vector<std::shared_ptr<Callback>>& {
        return callbacks_;
    }

private:
    std::vector<std::shared_ptr<Callback>> callbacks_;
};

} // namespace nn
} // namespace tenzor
