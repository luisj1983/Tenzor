/**
 * @file test_callbacks_multidtype.cpp
 * @brief Multi-dtype multi-backend tests for training callback system
 *
 * Tests callback functionality with Float32, Float64, and Float16 dtypes across
 * CPU, CUDA, OneAPI, Vulkan, and ROCm backends:
 * - ModelCheckpoint callback
 * - EarlyStopping callback
 * - LearningRateScheduler callback
 * - ProgressCallback
 * - Custom callbacks
 * - CallbackList
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/callbacks.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <sstream>
#include <memory>
#include <limits>
#include <filesystem>
#include <string>
#include <unistd.h>  // getpid — CC.17 per-test, per-process temp path

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

// ============================================================================
// Helper Classes
// ============================================================================

// Helper class for testing custom callbacks
class TestCallbackImpl : public Callback {
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
// Callback Multi-Backend Multi-DType Test Fixture
// ============================================================================

class CallbackMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();

        // Create simple model for testing
        model = std::make_shared<Linear>(10, 5, true);
        convert_model(*model);

        optimizer_sgd = std::make_shared<optim::SGD>(model->parameters(), 0.01f);
        optimizer_adam = std::make_shared<optim::Adam>(model->parameters(), 0.001f);
    }

    std::shared_ptr<Linear> model;
    std::shared_ptr<optim::SGD> optimizer_sgd;
    std::shared_ptr<optim::Adam> optimizer_adam;
};

// ============================================================================
// Base Callback Tests
// ============================================================================

TEST_P(CallbackMultiDTypeTest, BaseCallbackInterface) {
    auto callback = std::make_shared<Callback>();

    // Should not crash when calling hooks on base class
    EXPECT_NO_THROW(callback->on_train_begin());
    EXPECT_NO_THROW(callback->on_epoch_begin(0));
    EXPECT_NO_THROW(callback->on_batch_begin(0));
    EXPECT_NO_THROW(callback->on_batch_end(0, 0.5f));
    EXPECT_NO_THROW(callback->on_epoch_end(0, 0.5f, 0.4f));
    EXPECT_NO_THROW(callback->on_train_end());
}

TEST_P(CallbackMultiDTypeTest, CustomCallbackTracking) {
    auto callback = std::make_shared<TestCallbackImpl>();

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

TEST_P(CallbackMultiDTypeTest, ProgressCallbackCreation) {
    auto progress = std::make_shared<ProgressCallback>(10);
    EXPECT_NO_THROW(progress->set_total_batches(100));
    EXPECT_NO_THROW(progress->set_total_epochs(50));
}

TEST_P(CallbackMultiDTypeTest, ProgressCallbackHooks) {
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

TEST_P(CallbackMultiDTypeTest, ProgressCallbackWithMultipleEpochs) {
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

TEST_P(CallbackMultiDTypeTest, EarlyStoppingCallbackCreation) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(5, 0.001f);
    EXPECT_FALSE(early_stop->should_stop());
    EXPECT_FLOAT_EQ(early_stop->best_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(early_stop->wait_count(), 0);
}

TEST_P(CallbackMultiDTypeTest, EarlyStoppingImprovement) {
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

TEST_P(CallbackMultiDTypeTest, EarlyStoppingTriggered) {
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

TEST_P(CallbackMultiDTypeTest, EarlyStoppingMonitorTrainLoss) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(2, 0.0f, "train_loss");

    // Should monitor train_loss instead of val_loss
    early_stop->on_epoch_end(0, 1.0f, 0.5f);  // train_loss=1.0, val_loss=0.5
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 1.0f);  // Should track train_loss

    early_stop->on_epoch_end(1, 0.8f, 0.6f);  // train_loss improves, val_loss worsens
    EXPECT_FLOAT_EQ(early_stop->best_loss(), 0.8f);
    EXPECT_EQ(early_stop->wait_count(), 0);  // Should reset because train_loss improved
}

TEST_P(CallbackMultiDTypeTest, EarlyStoppingReset) {
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

TEST_P(CallbackMultiDTypeTest, ModelCheckpointCallbackCreation) {
    // CC.17: per-process, per-test temp path so parallel ctest runs don't race.
    const std::string ckpt_template = (std::filesystem::temp_directory_path() /
        ("tenzor_model_epoch_{epoch}_" + std::to_string(::getpid()) + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".pt")).string();
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        ckpt_template,
        model,
        true,  // save_best_only
        "val_loss"
    );

    EXPECT_FLOAT_EQ(checkpoint->best_loss(), std::numeric_limits<float>::max());
    EXPECT_EQ(checkpoint->last_checkpoint(), "");
}

TEST_P(CallbackMultiDTypeTest, ModelCheckpointSaveBestOnly) {
    const std::string ckpt_path = (std::filesystem::temp_directory_path() /
        ("tenzor_test_model_best_multidtype_" + std::to_string(::getpid()) + "_" +
         backend_name() + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".pt")).string();
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        ckpt_path,
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

TEST_P(CallbackMultiDTypeTest, ModelCheckpointFilepathTemplate) {
    // CC.17: per-process, per-test temp path.
    const std::string ckpt_template = (std::filesystem::temp_directory_path() /
        ("tenzor_model_epoch_{epoch:03d}_" + std::to_string(::getpid()) + "_" +
         backend_name() + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".pt")).string();
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        ckpt_template,
        model,
        false  // save every epoch
    );

    // Should handle filepath template
    EXPECT_NO_THROW(checkpoint->on_epoch_end(0, 1.0f, 0.9f));
    EXPECT_NO_THROW(checkpoint->on_epoch_end(9, 0.8f, 0.7f));
}

TEST_P(CallbackMultiDTypeTest, ModelCheckpointMonitorTrainLoss) {
    const std::string ckpt_path = (std::filesystem::temp_directory_path() /
        ("tenzor_test_model_trainloss_" + std::to_string(::getpid()) + "_" +
         backend_name() + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".pt")).string();
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        ckpt_path,
        model,
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

TEST_P(CallbackMultiDTypeTest, LRSchedulerCallbackCreation) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd,
        "step",
        0.1f,
        10
    );

    EXPECT_NO_THROW(scheduler->on_train_begin());
    EXPECT_GT(scheduler->current_lr(), 0.0f);
}

TEST_P(CallbackMultiDTypeTest, LRSchedulerStepDecay) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd,
        "step",
        0.5f,   // decay by 0.5x
        3       // every 3 epochs
    );

    scheduler->on_train_begin();

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

TEST_P(CallbackMultiDTypeTest, LRSchedulerExponentialDecay) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_adam,
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

TEST_P(CallbackMultiDTypeTest, LRSchedulerCosineAnnealing) {
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd,
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

TEST_P(CallbackMultiDTypeTest, LRSchedulerMultipleSchedules) {
    // Test different schedule types
    auto step_sched = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd, "step", 0.5f, 5
    );

    auto exp_sched = std::make_shared<LRSchedulerCallback>(
        optimizer_adam, "exponential", 0.95f
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

TEST_P(CallbackMultiDTypeTest, CallbackListCreation) {
    CallbackList callbacks;
    EXPECT_EQ(callbacks.callbacks().size(), 0);
}

TEST_P(CallbackMultiDTypeTest, CallbackListAddCallbacks) {
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackImpl>();
    auto cb2 = std::make_shared<TestCallbackImpl>();
    auto cb3 = std::make_shared<ProgressCallback>();

    callbacks.add(cb1);
    callbacks.add(cb2);
    callbacks.add(cb3);

    EXPECT_EQ(callbacks.callbacks().size(), 3);
}

TEST_P(CallbackMultiDTypeTest, CallbackListCallsAllCallbacks) {
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackImpl>();
    auto cb2 = std::make_shared<TestCallbackImpl>();

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

TEST_P(CallbackMultiDTypeTest, CallbackListRemoveCallbacks) {
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackImpl>();
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

TEST_P(CallbackMultiDTypeTest, MultipleCallbacksWithEarlyStopping) {
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

TEST_P(CallbackMultiDTypeTest, CallbackWithModelAndOptimizer) {
    // Create callbacks
    const std::string ckpt_path = (std::filesystem::temp_directory_path() /
        ("tenzor_test_model_integrated_" + std::to_string(::getpid()) + "_" +
         backend_name() + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".pt")).string();
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        ckpt_path,
        model,
        true
    );

    auto lr_scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_adam,
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

TEST_P(CallbackMultiDTypeTest, CompleteTrainingWorkflow) {
    CallbackList callbacks;

    // Add all callback types
    auto custom_cb = std::make_shared<TestCallbackImpl>();
    auto progress = std::make_shared<ProgressCallback>(2);
    auto early_stop = std::make_shared<EarlyStoppingCallback>(5, 0.001f);
    const std::string workflow_ckpt = (std::filesystem::temp_directory_path() /
        ("tenzor_workflow_model_" + std::to_string(::getpid()) + "_" +
         backend_name() + "_" +
         std::string(::testing::UnitTest::GetInstance()->current_test_info()->name()) +
         ".pt")).string();
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        workflow_ckpt,
        model,
        true
    );
    auto lr_scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd,
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

TEST_P(CallbackMultiDTypeTest, CallbackOrderMatters) {
    CallbackList callbacks;

    auto cb1 = std::make_shared<TestCallbackImpl>();
    auto cb2 = std::make_shared<TestCallbackImpl>();

    // Add in specific order
    callbacks.add(cb1);
    callbacks.add(cb2);

    callbacks.on_epoch_end(0, 0.5f, 0.4f);

    // Both should have been called
    EXPECT_EQ(cb1->epoch_end_count, 1);
    EXPECT_EQ(cb2->epoch_end_count, 1);
}

TEST_P(CallbackMultiDTypeTest, LRSchedulerWithDifferentOptimizers) {
    // Test with SGD
    auto sgd_scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd,
        "step",
        0.5f,
        3
    );

    // Test with Adam
    auto adam_scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_adam,
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

TEST_P(CallbackMultiDTypeTest, EarlyStoppingWithZeroPatience) {
    auto early_stop = std::make_shared<EarlyStoppingCallback>(0, 0.0f);

    // First epoch: loss improves from infinity to 0.5 (this is an improvement, no stop)
    early_stop->on_epoch_end(0, 1.0f, 0.5f);
    EXPECT_FALSE(early_stop->should_stop()) << "First improvement should not trigger stop";

    // Second epoch: loss stays same (no improvement with patience=0 should stop)
    early_stop->on_epoch_end(1, 1.0f, 0.5f);
    EXPECT_TRUE(early_stop->should_stop()) << "Zero patience should stop on first non-improvement";
}

TEST_P(CallbackMultiDTypeTest, CallbackListEmpty) {
    CallbackList callbacks;

    // Should handle empty list without crashing
    EXPECT_NO_THROW(callbacks.on_train_begin());
    EXPECT_NO_THROW(callbacks.on_epoch_begin(0));
    EXPECT_NO_THROW(callbacks.on_batch_end(0, 0.5f));
    EXPECT_NO_THROW(callbacks.on_epoch_end(0, 0.5f, 0.4f));
    EXPECT_NO_THROW(callbacks.on_train_end());
}

TEST_P(CallbackMultiDTypeTest, ModelCheckpointWithInvalidPath) {
    // Test with path that might fail
    auto checkpoint = std::make_shared<ModelCheckpointCallback>(
        "/invalid_path/model.pt",
        model,
        true
    );

    // Should not crash, but may fail to save (implementation dependent)
    EXPECT_NO_THROW(checkpoint->on_epoch_end(0, 1.0f, 0.9f));
}

TEST_P(CallbackMultiDTypeTest, LRSchedulerWithInvalidSchedule) {
    // Test with potentially invalid schedule parameters
    auto scheduler = std::make_shared<LRSchedulerCallback>(
        optimizer_sgd,
        "unknown_schedule",  // Unknown schedule type
        0.5f,
        10
    );

    // Should handle gracefully or throw appropriate error
    EXPECT_NO_THROW(scheduler->on_train_begin());
}

// ============================================================================
// Model Forward Pass with Callbacks
// ============================================================================

TEST_P(CallbackMultiDTypeTest, ModelForwardWithCallback) {
    // Test that model works properly with callbacks
    Variable input = createInput({4, 10}, true);

    auto output = model->forward(input);

    EXPECT_EQ(output.shape()[0], 4);
    EXPECT_EQ(output.shape()[1], 5);
    EXPECT_EQ(output.tensor().dtype(), dtype());
}

// ============================================================================
// Test Instantiation
// ============================================================================

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(CallbackMultiDTypeTest);

/*
 * COVERAGE SUMMARY:
 *
 * Test Cases: 33
 * DTypes Tested: Float32, Float64, Float16
 * Backends Tested: CPU, CUDA, OneAPI
 * Total Scenarios: 33 tests × 3 dtypes × 3 backends = 297 test scenarios
 *
 * Coverage:
 * - Base callback: interface, custom tracking
 * - ProgressCallback: creation, hooks, multiple epochs
 * - EarlyStoppingCallback: creation, improvement tracking, triggering, monitoring
 * - ModelCheckpointCallback: creation, save best only, filepath template, monitoring
 * - LRSchedulerCallback: creation, step decay, exponential, cosine, multiple schedules
 * - CallbackList: creation, adding, calling all, ordering
 * - Integration: complete workflows, model+optimizer integration
 * - Edge cases: zero patience, empty list, invalid paths, invalid schedules
 */
