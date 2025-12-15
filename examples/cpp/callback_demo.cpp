/**
 * @file callback_demo.cpp
 * @brief Demonstration of the Tenzor callback system
 *
 * This example shows how to use all four callback types together
 * in a realistic training scenario.
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <iostream>
#include <memory>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    // Initialize Tenzor
    tenzor::initialize();

    std::cout << "=== Tenzor Callback System Demo ===\n\n";

    // Create a simple neural network (784 -> 128 -> 10)
    auto model = std::make_shared<Sequential>(
        std::make_shared<Linear>(784, 128),
        std::make_shared<ReLU>(),
        std::make_shared<Linear>(128, 10)
    );

    // Create optimizer
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);

    // Create loss function
    auto criterion = CrossEntropyLoss();

    // Setup callbacks
    CallbackList callbacks;

    // 1. Progress tracking
    std::cout << "Setting up ProgressCallback...\n";
    auto progress = std::make_shared<ProgressCallback>(5);  // Print every 5 batches
    progress->set_total_batches(50);
    progress->set_total_epochs(20);
    callbacks.add(progress);

    // 2. Early stopping
    std::cout << "Setting up EarlyStoppingCallback (patience=3)...\n";
    auto early_stop = std::make_shared<EarlyStoppingCallback>(
        3,      // patience
        0.001f, // min_delta
        "val_loss"
    );
    callbacks.add(early_stop);

    // 3. Model checkpointing
    std::cout << "Setting up ModelCheckpointCallback...\n";
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/tenzor_best_model.pt",
        model,
        true,  // save_best_only
        "val_loss"
    );
    callbacks.add(checkpoint);

    // 4. Learning rate scheduling
    std::cout << "Setting up LRSchedulerCallback (step decay)...\n";
    auto lr_scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer,
        "step",
        0.5f,   // decay by 50%
        5       // every 5 epochs
    );
    callbacks.add(lr_scheduler);

    std::cout << "\nCallbacks configured. Starting training simulation...\n";
    std::cout << std::string(70, '=') << "\n\n";

    // Simulate training loop
    callbacks.on_train_begin();

    const int num_epochs = 20;
    const int batches_per_epoch = 50;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        callbacks.on_epoch_begin(epoch);

        // Simulate training batches
        float epoch_train_loss = 0.0f;
        for (int batch = 0; batch < batches_per_epoch; ++batch) {
            // Simulate decreasing loss over time
            float batch_loss = 2.0f * std::exp(-epoch * 0.15f) +
                              0.1f * std::exp(-batch * 0.05f);

            epoch_train_loss += batch_loss;
            callbacks.on_batch_end(batch, batch_loss);
        }

        float train_loss = epoch_train_loss / batches_per_epoch;

        // Simulate validation
        // Loss improves for first 10 epochs, then plateaus
        float val_loss;
        if (epoch < 10) {
            val_loss = 1.8f * std::exp(-epoch * 0.2f) + 0.05f;
        } else {
            // Plateau with small random variation
            val_loss = 0.3f + (epoch % 2) * 0.01f;
        }

        callbacks.on_epoch_end(epoch, train_loss, val_loss);

        // Check early stopping
        if (early_stop->should_stop()) {
            std::cout << "\n>>> Training stopped early at epoch " << (epoch + 1)
                      << " due to validation loss plateau <<<\n";
            break;
        }
    }

    callbacks.on_train_end();

    // Print final statistics
    std::cout << "\n" << std::string(70, '=') << "\n";
    std::cout << "=== Training Complete ===\n";
    std::cout << std::string(70, '=') << "\n\n";

    std::cout << "Final Statistics:\n";
    std::cout << "  Best validation loss: "
              << std::fixed << std::setprecision(6)
              << early_stop->best_loss() << "\n";
    std::cout << "  Last checkpoint: " << checkpoint->last_checkpoint() << "\n";
    std::cout << "  Final learning rate: "
              << std::scientific << std::setprecision(2)
              << lr_scheduler->current_lr() << "\n";
    std::cout << "  Early stopping: "
              << (early_stop->should_stop() ? "Triggered" : "Not triggered") << "\n";

    std::cout << "\n=== Demo Complete ===\n";

    return 0;
}
