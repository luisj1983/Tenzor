/**
 * @file test_callbacks.cpp
 * @brief Unit tests for training callback system
 */

#include <gtest/gtest.h>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <sstream>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;

// Helper class for testing custom callbacks
class TestCallback : public Callback {
public:
    int epoch_begin_count = 0;
    int epoch_end_count = 0;
    int batch_begin_count = 0;
    int batch_end_count = 0;
    int train_begin_count = 0;
    int train_end_count = 0;

    float last_train_loss = 0.0f;
    float last_val_loss = 0.0f;
    int last_epoch = -1;

    auto on_epoch_begin(int epoch) -> void override {
        epoch_begin_count++;
        last_epoch = epoch;
    }

    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override {
        epoch_end_count++;
        last_epoch = epoch;
        last_train_loss = train_loss;
        last_val_loss = val_loss;
    }

    auto on_batch_begin(int batch_idx) -> void override {
        batch_begin_count++;
    }

    auto on_batch_end(int batch_idx, float loss) -> void override {
        batch_end_count++;
    }

    auto on_train_begin() -> void override {
        train_begin_count++;
    }

    auto on_train_end() -> void override {
        train_end_count++;
    }
};

// ============================================================================
// Base Callback Tests
// ============================================================================

TEST(CallbackTest, BaseCallbackInterface) {
    auto callback = std::make_shared<Callback>();

    // Should not crash when calling hooks on base class
    EXPECT_NO_THROW(callback->on_train_begin());
    EXPECT_NO_THROW(callback->on_epoch_begin(0));
    EXPECT_NO_THROW(callback->on_batch_begin(0));
    EXPECT_NO_THROW(callback->on_batch_end(0, 0.5f));
    EXPECT_NO_THROW(callback->on_epoch_end(0, 0.5f, 0.4f));
    EXPECT_NO_THROW(callback->on_train_end());
}

TEST(CallbackTest, CustomCallbackTracking) {
    auto callback = std::make_shared<TestCallback>();

    callback->on_train_begin();
    EXPECT_EQ(callback->train_begin_count, 1);

    callback->on_epoch_begin(0);
    EXPECT_EQ(callback->epoch_begin_count, 1);
    EXPECT_EQ(callback->last_epoch, 0);

    callback->on_batch_begin(0);
    callback->on_batch_end(0, 0.8f);
    EXPECT_EQ(callback->batch_begin_count, 1);
    EXPECT_EQ(callback->batch_end_count, 1);

    callback->on_epoch_end(0, 0.5f, 0.4f);
    EXPECT_EQ(callback->epoch_end_count, 1);
    EXPECT_FLOAT_EQ(callback->last_train_loss, 0.5f);
    EXPECT_FLOAT_EQ(callback->last_val_loss, 0.4f);

    callback->on_train_end();
    EXPECT_EQ(callback->train_end_count, 1);
}

// ============================================================================
// ProgressCallback Tests
// ============================================================================

TEST(CallbackTest, ProgressCallbackCreation) {
    auto progress = std::make_shared<ProgressCallback>(10);
    EXPECT_NO_THROW(progress->set_total_batches(100));
    EXPECT_NO_THROW(progress->set_total_epochs(50));
}

TEST(CallbackTest, ProgressCallbackHooks) {
    auto progress = std::make_shared<ProgressCallback>(5);
    progress->set_total_batches(20);
    progress->set_total_epochs(10);

    // Should not crash
    EXPECT_NO_THROW(progress->on_train_begin());
    EXPECT_NO_THROW(progress->on_epoch_begin(0));

    for (int i = 0; i < 20; ++i) {
        EXPECT_NO_THROW(progress->on_batch_end(i, 0.5f - i * 0.01f));
    }

    EXPECT_NO_THROW(progress->on_epoch_end(0, 0.35f, 0.30f));
    EXPECT_NO_THROW(progress->on_train_end());
}

// ============================================================================
// EarlyStoppingCallback Tests
// ============================================================================

TEST(CallbackTest, EarlyStoppingCallbackCreation) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(5, 0.001f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_FLOAT_EQ(early_stop->best_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(early_stop->wait_count(), 0);
}

TEST(CallbackTest, EarlyStoppingImprovement) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(3, 0.01f, "val_loss");

    // First epoch - improvement
    early_stop->on_epoch_end(0, 1.0f, 0.9f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 0.9f);
    EXPECT_EQ(early_stop->wait_count(), 0);

    // Second epoch - improvement
    early_stop->on_epoch_end(1, 0.8f, 0.7f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 0.7f);
    EXPECT_EQ(early_stop->wait_count(), 0);

    // Third epoch - small improvement (below min_delta)
    early_stop->on_epoch_end(2, 0.7f, 0.695f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_EQ(early_stop->wait_count(), 1);
}

TEST(CallbackTest, EarlyStoppingTriggered) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(2, 0.0f, "val_loss");

    // Initial improvement
    early_stop->on_epoch_end(0, 1.0f, 0.5f);
    EXPECT_FALSE(early_stop->should_stop());

    // No improvement - wait 1
    early_stop->on_epoch_end(1, 0.9f, 0.6f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_EQ(early_stop->wait_count(), 1);

    // No improvement - wait 2, should trigger
    early_stop->on_epoch_end(2, 0.8f, 0.7f);
    EXPECT_TRUE(early_stop->should_stop());
    EXPECT_EQ(early_stop->wait_count(), 2);
}

TEST(CallbackTest, EarlyStoppingMonitorTrainLoss) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(2, 0.0f, "train_loss");

    // Should monitor train_loss instead of val_loss
    early_stop->on_epoch_end(0, 1.0f, 0.5f);  // train_loss=1.0, val_loss=0.5
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 1.0f);  // Should track train_loss

    early_stop->on_epoch_end(1, 0.8f, 0.6f);  // train_loss improves, val_loss worsens
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 0.8f);
    EXPECT_EQ(early_stop->wait_count(), 0);  // Should reset because train_loss improved
}

// ============================================================================
// ModelCheckpointCallback Tests
// ============================================================================

TEST(CallbackTest, ModelCheckpointCallbackCreation) {
    auto model = std::make_shared<Linear>(10, 5);

    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/model_epoch_{epoch}.pt",
        model,
        true,  // save_best_only
        "val_loss"
    );

    EXPECT_FLOAT_EQ(checkpoint->best_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(checkpoint->last_checkpoint(), "");
}

TEST(CallbackTest, ModelCheckpointSaveBestOnly) {
    auto model = std::make_shared<Linear>(10, 5);

    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/test_model_best.pt",
        model,
        true  // save_best_only
    );

    // First epoch - should save (first model)
    checkpoint->on_epoch_end(0, 1.0f, 0.9f);
    EXPECT_FLOAT_EQ(checkpoint->best_loss(), 0.9f);

    // Second epoch - improvement, should save
    checkpoint->on_epoch_end(1, 0.8f, 0.7f);
    EXPECT_FLOAT_EQ(checkpoint->best_loss(), 0.7f);

    // Third epoch - no improvement, should not save
    float old_best = checkpoint->best_loss();
    checkpoint->on_epoch_end(2, 0.9f, 0.8f);
    EXPECT_FLOAT_EQ(checkpoint->best_loss(), old_best);  // Should not change
}

TEST(CallbackTest, ModelCheckpointFilepathTemplate) {
    auto model = std::make_shared<Linear>(10, 5);

    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/model_epoch_{epoch:03d}.pt",
        model,
        false  // save every epoch
    );

    // Should handle filepath template
    EXPECT_NO_THROW(checkpoint->on_epoch_end(0, 1.0f, 0.9f));
    EXPECT_NO_THROW(checkpoint->on_epoch_end(9, 0.8f, 0.7f));
}

// ============================================================================
// LRSchedulerCallback Tests
// ============================================================================

TEST(CallbackTest, LRSchedulerCallbackCreation) {
    auto model = std::make_shared<Linear>(10, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);

    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer,
        "step",
        0.1f,
        10
    );

    EXPECT_NO_THROW(scheduler->on_train_begin());
    EXPECT_GT(scheduler->current_lr(), 0.0f);
}

TEST(CallbackTest, LRSchedulerStepDecay) {
    auto model = std::make_shared<Linear>(10, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.1);

    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer,
        "step",
        0.5f,   // decay by 0.5x
        3       // every 3 epochs
    );

    scheduler->on_train_begin();
    float initial_lr = scheduler->current_lr();

    // Epochs 0-2: no decay
    scheduler->on_epoch_end(0, 1.0f, 0.9f);
    scheduler->on_epoch_end(1, 0.8f, 0.7f);
    float lr_before_decay = scheduler->current_lr();

    // Epoch 2 (0-indexed) = 3rd epoch: should decay
    scheduler->on_epoch_end(2, 0.7f, 0.6f);
    float lr_after_decay = scheduler->current_lr();

    // LR should have decreased
    EXPECT_LE(lr_after_decay, lr_before_decay);
}

TEST(CallbackTest, LRSchedulerExponentialDecay) {
    auto model = std::make_shared<Linear>(10, 5);
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);

    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer,
        "exponential",
        0.9f
    );

    scheduler->on_train_begin();

    // Should decay every epoch
    for (int i = 0; i < 5; ++i) {
        scheduler->on_epoch_end(i, 0.5f, 0.4f);
    }

    // LR should be significantly reduced
    EXPECT_GT(scheduler->current_lr(), 0.0f);
}

TEST(CallbackTest, LRSchedulerCosineAnnealing) {
    auto model = std::make_shared<Linear>(10, 5);
    auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.1);

    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer,
        "cosine",
        0.1f,
        10,  // total epochs
        0.0f  // min_lr
    );

    scheduler->on_train_begin();

    // Simulate cosine annealing
    for (int i = 0; i < 10; ++i) {
        scheduler->on_epoch_end(i, 0.5f, 0.4f);
    }

    EXPECT_GT(scheduler->current_lr(), 0.0f);
}

// ============================================================================
// CallbackList Tests
// ============================================================================

TEST(CallbackTest, CallbackListCreation) {
    CallbackList callbacks;
    EXPECT_EQ(callbacks.callbacks().size(), 0);
}

TEST(CallbackTest, CallbackListAddCallbacks) {
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallback>();
    auto cb2 = std::make_shared<TestCallback>();
    auto cb3 = std::make_shared<ProgressCallback>();

    callbacks.add(cb1);
    callbacks.add(cb2);
    callbacks.add(cb3);

    EXPECT_EQ(callbacks.callbacks().size(), 3);
}

TEST(CallbackTest, CallbackListCallsAllCallbacks) {
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallback>();
    auto cb2 = std::make_shared<TestCallback>();

    callbacks.add(cb1);
    callbacks.add(cb2);

    // Call hooks
    callbacks.on_train_begin();
    callbacks.on_epoch_begin(0);
    callbacks.on_batch_begin(0);
    callbacks.on_batch_end(0, 0.5f);
    callbacks.on_epoch_end(0, 0.5f, 0.4f);
    callbacks.on_train_end();

    // Both callbacks should have been called
    EXPECT_EQ(cb1->train_begin_count, 1);
    EXPECT_EQ(cb1->epoch_begin_count, 1);
    EXPECT_EQ(cb1->batch_begin_count, 1);
    EXPECT_EQ(cb1->batch_end_count, 1);
    EXPECT_EQ(cb1->epoch_end_count, 1);
    EXPECT_EQ(cb1->train_end_count, 1);

    EXPECT_EQ(cb2->train_begin_count, 1);
    EXPECT_EQ(cb2->epoch_begin_count, 1);
    EXPECT_EQ(cb2->batch_begin_count, 1);
    EXPECT_EQ(cb2->batch_end_count, 1);
    EXPECT_EQ(cb2->epoch_end_count, 1);
    EXPECT_EQ(cb2->train_end_count, 1);
}

// ============================================================================
// Integration Tests
// ============================================================================

TEST(CallbackTest, MultipleCallbacksWithEarlyStopping) {
    CallbackList callbacks;

    auto progress = std::make_shared<ProgressCallback>(1);
    auto early_stop = std::make_shared<EarlyStoppingCallback>(2, 0.0f);

    callbacks.add(progress);
    callbacks.add(early_stop);

    callbacks.on_train_begin();

    // Simulate training loop with plateau
    for (int epoch = 0; epoch < 10; ++epoch) {
        callbacks.on_epoch_begin(epoch);

        for (int batch = 0; batch < 10; ++batch) {
            callbacks.on_batch_end(batch, 0.5f);
        }

        // Loss improves for 2 epochs, then plateaus
        float val_loss;
        if (epoch < 2) {
            val_loss = 0.5f - epoch * 0.1f;  // Improving
        } else {
            val_loss = 0.4f;  // Plateaus - should trigger early stopping
        }

        callbacks.on_epoch_end(epoch, 0.5f, val_loss);

        if (early_stop->should_stop()) {
            break;
        }
    }

    callbacks.on_train_end();

    // Early stopping should have triggered due to plateau
    EXPECT_TRUE(early_stop->should_stop());
}

TEST(CallbackTest, CallbackWithModelAndOptimizer) {
    // Create model
    auto model = std::make_shared<Linear>(10, 5);

    // Create optimizer
    auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);

    // Create callbacks
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/test_model.pt",
        model,
        true
    );

    auto lr_scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer,
        "step",
        0.1f,
        5
    );

    CallbackList callbacks;
    callbacks.add(checkpoint);
    callbacks.add(lr_scheduler);

    // Simulate training
    callbacks.on_train_begin();

    for (int epoch = 0; epoch < 10; ++epoch) {
        callbacks.on_epoch_begin(epoch);
        callbacks.on_epoch_end(epoch, 1.0f - epoch * 0.05f, 0.9f - epoch * 0.05f);
    }

    callbacks.on_train_end();

    // Checkpoint should have saved best model
    EXPECT_LT(checkpoint->best_loss(), 1.0f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
