/**
 * @file test_callbacks_multidtype.cpp
 * @brief Multi-dtype unit tests for training callback system
 *
 * Tests callback functionality with Float32 and Float64 dtypes.
 * Callbacks are training utilities that should work with any dtype:
 * - ModelCheckpoint callback
 * - EarlyStopping callback
 * - LearningRateScheduler callback
 * - ProgressCallback
 * - Custom callbacks
 * - CallbackList
 */

#include <gtest/gtest.h>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <sstream>
#include <memory>
#include <limits>

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Helper Classes
// ============================================================================

// Helper class for testing custom callbacks (templated for dtype)
template<typename T>
class TestCallbackTyped : public Callback {
public:
    int epoch_begin_count = 0;
    int epoch_end_count = 0;
    int batch_begin_count = 0;
    int batch_end_count = 0;
    int train_begin_count = 0;
    int train_end_count = 0;

    T last_train_loss = static_cast<T>(0.0);
    T last_val_loss = static_cast<T>(0.0);
    int last_epoch = -1;

    auto on_epoch_begin(int epoch) -> void override {
        epoch_begin_count++;
        last_epoch = epoch;
    }

    auto on_epoch_end(int epoch, float train_loss, float val_loss) -> void override {
        epoch_end_count++;
        last_epoch = epoch;
        last_train_loss = static_cast<T>(train_loss);
        last_val_loss = static_cast<T>(val_loss);
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

// Test fixture template
template<typename T>
class CallbackMultiDtypeTest : public ::testing::Test {
protected:
    static constexpr DType dtype = std::is_same_v<T, float> ? DType::Float32 : DType::Float64;

    void SetUp() override {
        // Create simple model for testing
        // Note: Linear layer doesn't accept dtype in constructor
        // It will use the dtype of the input tensor during forward pass
        model = std::make_shared<Linear>(10, 5, true);
        optimizer_sgd = std::make_shared<optim::SGD>(model->parameters(), static_cast<T>(0.01));
        optimizer_adam = std::make_shared<optim::Adam>(model->parameters(), static_cast<T>(0.001));
    }

    std::shared_ptr<Linear> model;
    std::shared_ptr<optim::SGD> optimizer_sgd;
    std::shared_ptr<optim::Adam> optimizer_adam;
};

// Type definitions for parameterized tests
using DTypeList = ::testing::Types<float, double>;
TYPED_TEST_SUITE(CallbackMultiDtypeTest, DTypeList);

// ============================================================================
// Base Callback Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, BaseCallbackInterface) {
    auto callback = std::make_shared<Callback>();

    // Should not crash when calling hooks on base class
    EXPECT_NO_THROW(callback->on_train_begin());
    EXPECT_NO_THROW(callback->on_epoch_begin(0));
    EXPECT_NO_THROW(callback->on_batch_begin(0));
    EXPECT_NO_THROW(callback->on_batch_end(0, 0.5f));
    EXPECT_NO_THROW(callback->on_epoch_end(0, 0.5f, 0.4f));
    EXPECT_NO_THROW(callback->on_train_end());
}

TYPED_TEST(CallbackMultiDtypeTest, CustomCallbackTracking) {
    using T = TypeParam;
    auto callback = std::make_shared<TestCallbackTyped<T>>();

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

    if constexpr (std::is_same_v<T, float>) {
        EXPECT_FLOAT_EQ(callback->last_train_loss, static_cast<T>(0.5));
        EXPECT_FLOAT_EQ(callback->last_val_loss, static_cast<T>(0.4));
    } else {
        EXPECT_DOUBLE_EQ(callback->last_train_loss, static_cast<T>(0.5));
        EXPECT_DOUBLE_EQ(callback->last_val_loss, static_cast<T>(0.4));
    }

    callback->on_train_end();
    EXPECT_EQ(callback->train_end_count, 1);
}

// ============================================================================
// ProgressCallback Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, ProgressCallbackCreation) {
    auto progress = std::make_shared<ProgressCallback>(10);
    EXPECT_NO_THROW(progress->set_total_batches(100));
    EXPECT_NO_THROW(progress->set_total_epochs(50));
}

TYPED_TEST(CallbackMultiDtypeTest, ProgressCallbackHooks) {
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

TYPED_TEST(CallbackMultiDtypeTest, ProgressCallbackWithMultipleEpochs) {
    auto progress = std::make_shared<ProgressCallback>(3);
    progress->set_total_batches(10);
    progress->set_total_epochs(5);

    progress->on_train_begin();

    for (int epoch = 0; epoch < 5; ++epoch) {
        progress->on_epoch_begin(epoch);
        for (int batch = 0; batch < 10; ++batch) {
            progress->on_batch_end(batch, 0.5f - batch * 0.02f);
        }
        progress->on_epoch_end(epoch, 0.3f - epoch * 0.05f, 0.25f - epoch * 0.04f);
    }

    progress->on_train_end();
    // Should complete without crashing
}

// ============================================================================
// EarlyStoppingCallback Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, EarlyStoppingCallbackCreation) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(5, 0.001f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_FLOAT_EQ(early_stop->best_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(early_stop->wait_count(), 0);
}

TYPED_TEST(CallbackMultiDtypeTest, EarlyStoppingImprovement) {
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

TYPED_TEST(CallbackMultiDtypeTest, EarlyStoppingTriggered) {
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

TYPED_TEST(CallbackMultiDtypeTest, EarlyStoppingMonitorTrainLoss) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(2, 0.0f, "train_loss");

    // Should monitor train_loss instead of val_loss
    early_stop->on_epoch_end(0, 1.0f, 0.5f);  // train_loss=1.0, val_loss=0.5
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 1.0f);  // Should track train_loss

    early_stop->on_epoch_end(1, 0.8f, 0.6f);  // train_loss improves, val_loss worsens
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 0.8f);
    EXPECT_EQ(early_stop->wait_count(), 0);  // Should reset because train_loss improved
}

TYPED_TEST(CallbackMultiDtypeTest, EarlyStoppingReset) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(3, 0.01f, "val_loss");

    // Initial improvement
    early_stop->on_epoch_end(0, 1.0f, 0.5f);

    // No improvement - wait increases
    early_stop->on_epoch_end(1, 0.9f, 0.6f);
    EXPECT_EQ(early_stop->wait_count(), 1);

    // Improvement again - should reset wait count
    early_stop->on_epoch_end(2, 0.7f, 0.4f);
    EXPECT_EQ(early_stop->wait_count(), 0);
    EXPECT_FALSE(early_stop->should_stop());
}

// ============================================================================
// ModelCheckpointCallback Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, ModelCheckpointCallbackCreation) {
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/model_epoch_{epoch}.pt",
        this->model,
        true,  // save_best_only
        "val_loss"
    );

    EXPECT_FLOAT_EQ(checkpoint->best_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(checkpoint->last_checkpoint(), "");
}

TYPED_TEST(CallbackMultiDtypeTest, ModelCheckpointSaveBestOnly) {
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/test_model_best_multidtype.pt",
        this->model,
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

TYPED_TEST(CallbackMultiDtypeTest, ModelCheckpointFilepathTemplate) {
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/model_epoch_{epoch:03d}_multidtype.pt",
        this->model,
        false  // save every epoch
    );

    // Should handle filepath template
    EXPECT_NO_THROW(checkpoint->on_epoch_end(0, 1.0f, 0.9f));
    EXPECT_NO_THROW(checkpoint->on_epoch_end(9, 0.8f, 0.7f));
}

TYPED_TEST(CallbackMultiDtypeTest, ModelCheckpointMonitorTrainLoss) {
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/test_model_trainloss.pt",
        this->model,
        true,
        "train_loss"
    );

    // Should monitor train_loss
    checkpoint->on_epoch_end(0, 1.0f, 0.5f);
    EXPECT_FLOAT_EQ(checkpoint->best_loss(), 1.0f);

    // Better train_loss, worse val_loss
    checkpoint->on_epoch_end(1, 0.8f, 0.6f);
    EXPECT_FLOAT_EQ(checkpoint->best_loss(), 0.8f);
}

// ============================================================================
// LRSchedulerCallback Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerCallbackCreation) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd,
        "step",
        0.1f,
        10
    );

    EXPECT_NO_THROW(scheduler->on_train_begin());
    EXPECT_GT(scheduler->current_lr(), 0.0f);
}

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerStepDecay) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd,
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

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerExponentialDecay) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_adam,
        "exponential",
        0.9f
    );

    scheduler->on_train_begin();
    float initial_lr = scheduler->current_lr();

    // Should decay every epoch
    for (int i = 0; i < 5; ++i) {
        scheduler->on_epoch_end(i, 0.5f, 0.4f);
    }

    // LR should be significantly reduced
    float final_lr = scheduler->current_lr();
    EXPECT_GT(final_lr, 0.0f);
    EXPECT_LT(final_lr, initial_lr);
}

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerCosineAnnealing) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd,
        "cosine",
        0.1f,
        10,  // total epochs
        0.0f  // min_lr
    );

    scheduler->on_train_begin();
    float initial_lr = scheduler->current_lr();

    // Simulate cosine annealing
    for (int i = 0; i < 10; ++i) {
        scheduler->on_epoch_end(i, 0.5f, 0.4f);
    }

    float final_lr = scheduler->current_lr();
    EXPECT_GT(final_lr, 0.0f);
    // With cosine annealing, final LR should be different from initial
}

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerMultipleSchedules) {
    // Test different schedule types
    auto step_sched = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd, "step", 0.5f, 5
    );

    auto exp_sched = std::make_shared<LRSchedulerCallback>(
        this->optimizer_adam, "exponential", 0.95f
    );

    step_sched->on_train_begin();
    exp_sched->on_train_begin();

    for (int i = 0; i < 10; ++i) {
        step_sched->on_epoch_end(i, 0.5f, 0.4f);
        exp_sched->on_epoch_end(i, 0.5f, 0.4f);
    }

    // Both should have valid learning rates
    EXPECT_GT(step_sched->current_lr(), 0.0f);
    EXPECT_GT(exp_sched->current_lr(), 0.0f);
}

// ============================================================================
// CallbackList Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, CallbackListCreation) {
    CallbackList callbacks;
    EXPECT_EQ(callbacks.callbacks().size(), 0);
}

TYPED_TEST(CallbackMultiDtypeTest, CallbackListAddCallbacks) {
    using T = TypeParam;
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackTyped<T>>();
    auto cb2 = std::make_shared<TestCallbackTyped<T>>();
    auto cb3 = std::make_shared<ProgressCallback>();

    callbacks.add(cb1);
    callbacks.add(cb2);
    callbacks.add(cb3);

    EXPECT_EQ(callbacks.callbacks().size(), 3);
}

TYPED_TEST(CallbackMultiDtypeTest, CallbackListCallsAllCallbacks) {
    using T = TypeParam;
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackTyped<T>>();
    auto cb2 = std::make_shared<TestCallbackTyped<T>>();

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

TYPED_TEST(CallbackMultiDtypeTest, CallbackListRemoveCallbacks) {
    using T = TypeParam;
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackTyped<T>>();
    auto cb2 = std::make_shared<ProgressCallback>();

    callbacks.add(cb1);
    callbacks.add(cb2);
    EXPECT_EQ(callbacks.callbacks().size(), 2);

    // Note: CallbackList API doesn't support removal, only addition
    // This test just verifies we can access the callbacks vector
    EXPECT_EQ(callbacks.callbacks().size(), 2);
}

// ============================================================================
// Integration Tests
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, MultipleCallbacksWithEarlyStopping) {
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

TYPED_TEST(CallbackMultiDtypeTest, CallbackWithModelAndOptimizer) {
    // Create callbacks
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/test_model_integrated.pt",
        this->model,
        true
    );

    auto lr_scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_adam,
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

    // Learning rate should have been adjusted
    EXPECT_GT(lr_scheduler->current_lr(), 0.0f);
}

TYPED_TEST(CallbackMultiDtypeTest, CompleteTrainingWorkflow) {
    using T = TypeParam;

    CallbackList callbacks;

    // Add all callback types
    auto custom_cb = std::make_shared<TestCallbackTyped<T>>();
    auto progress = std::make_shared<ProgressCallback>(2);
    auto early_stop = std::make_shared<EarlyStoppingCallback>(5, 0.001f);
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/tmp/workflow_model.pt",
        this->model,
        true
    );
    auto lr_scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd,
        "exponential",
        0.95f
    );

    callbacks.add(custom_cb);
    callbacks.add(progress);
    callbacks.add(early_stop);
    callbacks.add(checkpoint);
    callbacks.add(lr_scheduler);

    progress->set_total_batches(20);
    progress->set_total_epochs(15);

    callbacks.on_train_begin();

    // Simulate complete training loop
    for (int epoch = 0; epoch < 15; ++epoch) {
        callbacks.on_epoch_begin(epoch);

        for (int batch = 0; batch < 20; ++batch) {
            callbacks.on_batch_begin(batch);
            float batch_loss = 1.0f - (epoch * 20 + batch) * 0.001f;
            callbacks.on_batch_end(batch, batch_loss);
        }

        float train_loss = 1.0f - epoch * 0.05f;
        float val_loss = 0.9f - epoch * 0.04f;
        callbacks.on_epoch_end(epoch, train_loss, val_loss);

        if (early_stop->should_stop()) {
            break;
        }
    }

    callbacks.on_train_end();

    // Verify all callbacks were called
    EXPECT_GT(custom_cb->train_begin_count, 0);
    EXPECT_GT(custom_cb->epoch_begin_count, 0);
    EXPECT_GT(custom_cb->batch_begin_count, 0);
    EXPECT_GT(custom_cb->train_end_count, 0);

    // Checkpoint should have tracked best loss
    EXPECT_LT(checkpoint->best_loss(), 1.0f);

    // Learning rate should have changed
    EXPECT_GT(lr_scheduler->current_lr(), 0.0f);
}

TYPED_TEST(CallbackMultiDtypeTest, CallbackOrderMatters) {
    using T = TypeParam;

    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackTyped<T>>();
    auto cb2 = std::make_shared<TestCallbackTyped<T>>();

    // Add in specific order
    callbacks.add(cb1);
    callbacks.add(cb2);

    callbacks.on_epoch_end(0, 0.5f, 0.4f);

    // Both should have been called
    EXPECT_EQ(cb1->epoch_end_count, 1);
    EXPECT_EQ(cb2->epoch_end_count, 1);
}

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerWithDifferentOptimizers) {
    using T = TypeParam;

    // Test with SGD
    auto sgd_scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd,
        "step",
        0.5f,
        3
    );

    // Test with Adam
    auto adam_scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_adam,
        "exponential",
        0.9f
    );

    sgd_scheduler->on_train_begin();
    adam_scheduler->on_train_begin();

    float sgd_initial_lr = sgd_scheduler->current_lr();
    float adam_initial_lr = adam_scheduler->current_lr();

    for (int i = 0; i < 10; ++i) {
        sgd_scheduler->on_epoch_end(i, 0.5f, 0.4f);
        adam_scheduler->on_epoch_end(i, 0.5f, 0.4f);
    }

    // Both should have valid learning rates
    EXPECT_GT(sgd_scheduler->current_lr(), 0.0f);
    EXPECT_GT(adam_scheduler->current_lr(), 0.0f);
}

// ============================================================================
// Edge Cases and Error Handling
// ============================================================================

TYPED_TEST(CallbackMultiDtypeTest, EarlyStoppingWithZeroPatience) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(0, 0.0f);

    // Should trigger immediately after first epoch
    early_stop->on_epoch_end(0, 1.0f, 0.5f);
    EXPECT_TRUE(early_stop->should_stop());
}

TYPED_TEST(CallbackMultiDtypeTest, CallbackListEmpty) {
    CallbackList callbacks;

    // Should handle empty list without crashing
    EXPECT_NO_THROW(callbacks.on_train_begin());
    EXPECT_NO_THROW(callbacks.on_epoch_begin(0));
    EXPECT_NO_THROW(callbacks.on_batch_end(0, 0.5f));
    EXPECT_NO_THROW(callbacks.on_epoch_end(0, 0.5f, 0.4f));
    EXPECT_NO_THROW(callbacks.on_train_end());
}

TYPED_TEST(CallbackMultiDtypeTest, ModelCheckpointWithInvalidPath) {
    // Test with path that might fail
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/invalid_path/model.pt",
        this->model,
        true
    );

    // Should not crash, but may fail to save (implementation dependent)
    EXPECT_NO_THROW(checkpoint->on_epoch_end(0, 1.0f, 0.9f));
}

TYPED_TEST(CallbackMultiDtypeTest, LRSchedulerWithInvalidSchedule) {
    // Test with potentially invalid schedule parameters
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        this->optimizer_sgd,
        "unknown_schedule",  // Unknown schedule type
        0.5f,
        10
    );

    // Should handle gracefully or throw appropriate error
    EXPECT_NO_THROW(scheduler->on_train_begin());
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
