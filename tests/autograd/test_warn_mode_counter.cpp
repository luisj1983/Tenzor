/**
 * @file test_warn_mode_counter.cpp
 * @brief Tests for HigherOrderGradMode::Warn disconnection counter and
 *        is_higher_order_stub() introspection API.
 *
 * Validates that:
 * 1. The disconnection counter increments when Warn mode falls through
 * 2. reset_higher_order_disconnection_count() works
 * 3. Error mode throws instead of incrementing
 * 4. is_higher_order_stub() is true for passthrough stub ops
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/function.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

class WarnModeCounterTest : public ::testing::Test {
protected:
    static bool initialized_;
    HigherOrderGradMode saved_mode_;

    void SetUp() override {
        if (!initialized_) {
            tenzor::initialize();
            initialized_ = true;
        }
        // Save current mode and reset counter before each test
        saved_mode_ = get_higher_order_grad_mode();
        reset_higher_order_disconnection_count();
    }

    void TearDown() override {
        // Restore original mode
        set_higher_order_grad_mode(saved_mode_);
    }
};

bool WarnModeCounterTest::initialized_ = false;

// Counter starts at zero after reset
TEST_F(WarnModeCounterTest, CounterStartsAtZero) {
    EXPECT_EQ(higher_order_disconnection_count(), 0u);
}

// Warn mode increments the counter when an op with a passthrough stub
// is encountered during create_graph=true backward
TEST_F(WarnModeCounterTest, WarnModeIncrements) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    // Build a graph that includes max (a stub op) with create_graph=true
    auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
    auto y = x * x;  // MulBackward — has full higher-order support
    auto z = tenzor::max(y, 1);  // MaxBackward — is_higher_order_stub()

    auto loss = tenzor::sum(z);

    // backward with create_graph=true triggers the Warn fallback for MaxBackward
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    EXPECT_GE(higher_order_disconnection_count(), 1u)
        << "Counter should increment when Warn mode disconnects gradient graph";
}

// Warn mode logs every disconnection (no deduplication)
TEST_F(WarnModeCounterTest, WarnModeLogsEveryDisconnection) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    // Trigger two separate backward passes that each hit a stub op
    for (int i = 0; i < 2; ++i) {
        auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
        auto z = tenzor::max(x * x, 1);
        auto loss = tenzor::sum(z);

        testing::internal::CaptureStderr();
        loss.backward(std::nullopt, false, true);
        std::string output = testing::internal::GetCapturedStderr();

        EXPECT_FALSE(output.empty())
            << "Warn mode should log on iteration " << i
            << " (no per-op deduplication)";
    }

    EXPECT_GE(higher_order_disconnection_count(), 2u)
        << "Counter should increment for each disconnection";
}

// Error mode throws instead of incrementing
TEST_F(WarnModeCounterTest, ErrorModeThrows) {
    set_higher_order_grad_mode(HigherOrderGradMode::Error);

    auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
    auto z = tenzor::max(x * x, 1);
    auto loss = tenzor::sum(z);

    EXPECT_THROW(
        loss.backward(std::nullopt, false, true),
        std::runtime_error
    ) << "Error mode should throw on unsupported higher-order op";

    EXPECT_EQ(higher_order_disconnection_count(), 0u)
        << "Counter should not increment when Error mode throws";
}

// Reset clears the counter
TEST_F(WarnModeCounterTest, ResetClearsCounter) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
    auto z = tenzor::max(x * x, 1);
    auto loss = tenzor::sum(z);
    loss.backward(std::nullopt, false, true);

    EXPECT_GE(higher_order_disconnection_count(), 1u);

    reset_higher_order_disconnection_count();
    EXPECT_EQ(higher_order_disconnection_count(), 0u);
}

// Ops with full higher-order support do NOT increment the counter
TEST_F(WarnModeCounterTest, FullSupportDoesNotIncrement) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    auto x = Variable(randn({3, 4}, DType::Float32, Device::cpu()), true);
    auto y = x * x;  // mul — full support
    auto loss = tenzor::sum(y);  // sum — full support
    loss.backward(std::nullopt, false, true);

    EXPECT_EQ(higher_order_disconnection_count(), 0u)
        << "Ops with full higher-order support should not trigger disconnection";
}

// is_higher_order_stub() returns true for known stub classes
TEST_F(WarnModeCounterTest, IsHigherOrderStubIntrospection) {
    // MaxBackward is a known stub
    auto max_fn = std::make_shared<MaxBackward>(std::optional<int64_t>(1), false);
    EXPECT_TRUE(max_fn->is_higher_order_stub());
    EXPECT_TRUE(max_fn->supports_higher_order());  // still claims support

    // TopKBackward is a known stub
    auto topk_fn = std::make_shared<TopKBackward>(5, 0);
    EXPECT_TRUE(topk_fn->is_higher_order_stub());

    // SortBackward is a known stub
    auto sort_fn = std::make_shared<SortBackward>(0);
    EXPECT_TRUE(sort_fn->is_higher_order_stub());
}
