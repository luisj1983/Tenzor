/**
 * @file test_anomaly_mode_full.cpp
 * @brief RAII / nesting / thread-safety tests for AnomalyMode.
 *
 * The existing tests/autograd/test_anomaly_detection.cpp covers the
 * forward-traceback behaviour. The audit (2026-05-02) flagged
 * anomaly_mode.hpp as having only 2 test references and asked for full
 * guard-semantics coverage. This file pins:
 *   - AnomalyMode default-enables on construction.
 *   - Destruction restores the previous flag (RAII).
 *   - Nested guards compose: inner restoration leaves outer enabled.
 *   - Thread-local storage: another thread is unaffected.
 *   - Explicit disable via set_anomaly_detection(false) is observable.
 */

#include <gtest/gtest.h>
#include <tenzor/autograd/anomaly_mode.hpp>
#include <atomic>
#include <thread>

using namespace tenzor;

class AnomalyModeFullTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Reset to a known-disabled state at the start of every test.
        set_anomaly_detection(false);
    }
    void TearDown() override {
        set_anomaly_detection(false);
    }
};

TEST_F(AnomalyModeFullTest, ConstructorEnables) {
    EXPECT_FALSE(is_anomaly_detection_enabled());
    AnomalyMode g;
    EXPECT_TRUE(is_anomaly_detection_enabled());
}

TEST_F(AnomalyModeFullTest, ConstructorWithFalseDisables) {
    set_anomaly_detection(true);
    EXPECT_TRUE(is_anomaly_detection_enabled());
    AnomalyMode g(false);
    EXPECT_FALSE(is_anomaly_detection_enabled());
}

TEST_F(AnomalyModeFullTest, DestructorRestoresPreviousState) {
    EXPECT_FALSE(is_anomaly_detection_enabled());
    {
        AnomalyMode g;
        EXPECT_TRUE(is_anomaly_detection_enabled());
    }
    EXPECT_FALSE(is_anomaly_detection_enabled());
}

TEST_F(AnomalyModeFullTest, NestedGuards_Compose) {
    EXPECT_FALSE(is_anomaly_detection_enabled());
    {
        AnomalyMode outer;            // enable
        EXPECT_TRUE(is_anomaly_detection_enabled());
        {
            AnomalyMode inner(false); // disable
            EXPECT_FALSE(is_anomaly_detection_enabled());
        }
        // Inner destructor restored to outer's state (enabled).
        EXPECT_TRUE(is_anomaly_detection_enabled());
    }
    EXPECT_FALSE(is_anomaly_detection_enabled());
}

TEST_F(AnomalyModeFullTest, ExplicitSetterIsObservable) {
    EXPECT_FALSE(is_anomaly_detection_enabled());
    set_anomaly_detection(true);
    EXPECT_TRUE(is_anomaly_detection_enabled());
    set_anomaly_detection(false);
    EXPECT_FALSE(is_anomaly_detection_enabled());
}

TEST_F(AnomalyModeFullTest, ThreadLocalIsolation) {
    set_anomaly_detection(true);
    EXPECT_TRUE(is_anomaly_detection_enabled());

    std::atomic<bool> child_saw_enabled{false};
    std::thread child([&]{
        // The child thread starts with its own thread-local copy, which
        // should be the default (disabled) — anomaly mode is documented
        // as thread-local so the parent's setting must not leak.
        child_saw_enabled = is_anomaly_detection_enabled();
        // Mutate the child's own copy and verify it doesn't escape.
        AnomalyMode g(false);
        EXPECT_FALSE(is_anomaly_detection_enabled());
    });
    child.join();

    EXPECT_FALSE(child_saw_enabled.load())
        << "Child thread inherited the parent's anomaly-mode flag — "
        << "thread-local isolation is broken.";
    // Parent state is unchanged.
    EXPECT_TRUE(is_anomaly_detection_enabled());
}
