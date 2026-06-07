/**
 * @file test_warn_mode_counter_multidtype.cpp
 * @brief Multi-backend tests for HigherOrderGradMode::Warn disconnection counter and
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
#include "tenzor/nn/layers/segmentation.hpp"
#include "tenzor/nn/functional.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class WarnModeCounterMultiDTypeTest : public BackendTest {
protected:
    HigherOrderGradMode saved_mode_;

    void SetUp() override {
        BackendTest::SetUp();
        // Save current mode and reset counter before each test
        saved_mode_ = get_higher_order_grad_mode();
        reset_higher_order_disconnection_count();
    }

    void TearDown() override {
        // Restore original mode
        set_higher_order_grad_mode(saved_mode_);
    }
};

// Counter starts at zero after reset
TEST_P(WarnModeCounterMultiDTypeTest, CounterStartsAtZero) {
    EXPECT_EQ(higher_order_disconnection_count(), 0u);
}

// Helper: build a tiny graph ending in upsample_bilinear — the only
// remaining passthrough stub (mul → upsample). Mirrors the single-dtype
// sibling at tests/autograd/test_warn_mode_counter.cpp.
// Note: upsample_bilinear requires CPU Float32 (the kernel is CPU-only),
// so the resulting test body does not vary by the parameterized `device`.
static Variable make_multidtype_stub_graph() {
    // UpsampleBilinear gained full higher-order support (audit D3); the canary
    // is now max_pool2d (MaxPool2dBackward is a passthrough higher-order stub).
    auto x = Variable(randn({1, 1, 4, 4}, DType::Float32, Device::cpu()), true);
    auto y = x * x;
    return tenzor::nn::functional::max_pool2d(y, {2, 2}, {2, 2});
}

// Warn mode increments the counter when an op with a passthrough stub
// is encountered during create_graph=true backward
TEST_P(WarnModeCounterMultiDTypeTest, WarnModeIncrements) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    auto z = make_multidtype_stub_graph();
    auto loss = tenzor::sum(z);

    // backward with create_graph=true triggers the Warn fallback for UpsampleBilinearBackward
    loss.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);

    EXPECT_GE(higher_order_disconnection_count(), 1u)
        << "Counter should increment when Warn mode disconnects gradient graph";
}

// Warn mode logs every disconnection (no deduplication)
TEST_P(WarnModeCounterMultiDTypeTest, WarnModeLogsEveryDisconnection) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    for (int i = 0; i < 2; ++i) {
        auto z = make_multidtype_stub_graph();
        auto loss = tenzor::sum(z);

        ::testing::internal::CaptureStderr();
        loss.backward(std::nullopt, false, true);
        std::string output = ::testing::internal::GetCapturedStderr();

        EXPECT_FALSE(output.empty())
            << "Warn mode should log on iteration " << i
            << " (no per-op deduplication)";
    }

    EXPECT_GE(higher_order_disconnection_count(), 2u)
        << "Counter should increment for each disconnection";
}

// Error mode throws instead of incrementing
TEST_P(WarnModeCounterMultiDTypeTest, ErrorModeThrows) {
    set_higher_order_grad_mode(HigherOrderGradMode::Error);

    auto z = make_multidtype_stub_graph();
    auto loss = tenzor::sum(z);

    EXPECT_THROW(
        loss.backward(std::nullopt, false, true),
        std::runtime_error
    ) << "Error mode should throw on unsupported higher-order op";

    EXPECT_EQ(higher_order_disconnection_count(), 0u)
        << "Counter should not increment when Error mode throws";
}

// Reset clears the counter
TEST_P(WarnModeCounterMultiDTypeTest, ResetClearsCounter) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    auto z = make_multidtype_stub_graph();
    auto loss = tenzor::sum(z);
    loss.backward(std::nullopt, false, true);

    EXPECT_GE(higher_order_disconnection_count(), 1u);

    reset_higher_order_disconnection_count();
    EXPECT_EQ(higher_order_disconnection_count(), 0u);
}

// Ops with full higher-order support do NOT increment the counter
TEST_P(WarnModeCounterMultiDTypeTest, FullSupportDoesNotIncrement) {
    set_higher_order_grad_mode(HigherOrderGradMode::Warn);

    auto x = Variable(randn({3, 4}, DType::Float32, device), true);
    auto y = x * x;  // mul — full support
    auto loss = tenzor::sum(y);  // sum — full support
    loss.backward(std::nullopt, false, true);

    EXPECT_EQ(higher_order_disconnection_count(), 0u)
        << "Ops with full higher-order support should not trigger disconnection";
}

// is_higher_order_stub() distinguishes stub classes from full-support classes.
// Max/Min/TopK/Sort/Scatter/Narrow were upgraded to full higher-order support;
// UpsampleBilinearBackward remains the only passthrough stub (see
// tests/autograd/test_warn_mode_counter.cpp for the sibling single-dtype check).
TEST_P(WarnModeCounterMultiDTypeTest, IsHigherOrderStubIntrospection) {
    // MaxBackward is now full support (no longer a stub)
    auto max_fn = std::make_shared<MaxBackward>(std::optional<int64_t>(1), false);
    EXPECT_FALSE(max_fn->is_higher_order_stub());
    EXPECT_TRUE(max_fn->supports_higher_order());

    // TopKBackward is now full support
    auto topk_fn = std::make_shared<TopKBackward>(5, 0);
    EXPECT_FALSE(topk_fn->is_higher_order_stub());

    // SortBackward is now full support
    auto sort_fn = std::make_shared<SortBackward>(0);
    EXPECT_FALSE(sort_fn->is_higher_order_stub());
}

INSTANTIATE_BACKEND_TESTS(WarnModeCounterMultiDTypeTest);
