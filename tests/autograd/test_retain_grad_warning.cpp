// Isolated test for Phase 3.2: warn-once when .grad is accessed on a
// non-leaf Variable that was not marked with retain_grad().
//
// This test binary is deliberately separate from the other autograd tests
// so the TENZOR_WARN_ONCE call-site static flag is guaranteed to be fresh
// on the first invocation. Mixing it with tests that also access non-leaf
// .grad would cause the first one to win and the warning capture below
// would miss the message.

#include <gtest/gtest.h>

#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/utils/logging.hpp>

#include <sstream>
#include <iostream>

namespace tenzor {
namespace {

class RetainGradWarningTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(RetainGradWarningTest, WarnsOnceOnNonLeafGradAccess) {
    // Build a tiny graph so the intermediate y is non-leaf:
    //   x (leaf, requires_grad=true) → y = x * 2 (non-leaf) → z = sum(y)
    auto data = ones({4}, DType::Float32, Device::cpu());
    Variable x(data, /*requires_grad=*/true);
    auto y = x * 2.0f;
    // Sanity: y is non-leaf *at creation*. The engine clears grad_fn_
    // during backward cleanup so y.is_leaf() would flip to true after
    // loss.backward(), but the warning uses the latched was_non_leaf_
    // flag which stays set.
    ASSERT_FALSE(y.is_leaf());
    ASSERT_TRUE(y.requires_grad());
    ASSERT_FALSE(y.retains_grad());

    auto loss = sum(y);
    loss.backward();

    // Redirect stdout to capture the logger output. Logger::warning()
    // goes to std::cout so this is the cleanest capture.
    std::stringstream captured;
    std::streambuf* old_cout_buf = std::cout.rdbuf(captured.rdbuf());

    // First access — should trigger the warning.
    const auto& grad_first = y.grad();
    (void)grad_first;

    // Second access — must NOT log again (warn-once).
    const auto& grad_second = y.grad();
    (void)grad_second;

    std::cout.rdbuf(old_cout_buf);

    const std::string out = captured.str();
    // Warning text from variable.cpp: we only need to assert it mentions
    // retain_grad (the actionable hint) and WARNING level tag.
    EXPECT_NE(out.find("[WARNING]"), std::string::npos)
        << "expected a warning-level log line, got: " << out;
    EXPECT_NE(out.find("retain_grad"), std::string::npos)
        << "expected 'retain_grad' in the warning, got: " << out;
    EXPECT_NE(out.find("non-leaf"), std::string::npos)
        << "expected 'non-leaf' in the warning, got: " << out;

    // Warn-once: count [WARNING] tags, which appear once per emission.
    size_t pos = 0;
    int count = 0;
    while ((pos = out.find("[WARNING]", pos)) != std::string::npos) {
        ++count;
        pos += 1;
    }
    EXPECT_EQ(count, 1)
        << "TENZOR_WARN_ONCE should have fired exactly once, got "
        << count << " [WARNING] emissions. Captured: " << out;
}

TEST_F(RetainGradWarningTest, LeafAccessDoesNotWarn) {
    // A pristine test binary: since the call-site once_flag has already
    // been consumed by WarnsOnceOnNonLeafGradAccess (or this test runs
    // first), we cannot reliably assert zero warnings over the process.
    // Instead we only assert that accessing a *leaf* .grad is silent in
    // this single test run.
    Variable x(ones({4}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    ASSERT_TRUE(x.is_leaf());

    std::stringstream captured;
    std::streambuf* old_cout_buf = std::cout.rdbuf(captured.rdbuf());
    const auto& g = x.grad();
    (void)g;
    std::cout.rdbuf(old_cout_buf);

    EXPECT_EQ(captured.str().find("[WARNING]"), std::string::npos)
        << "leaf .grad access should not emit a warning, got: "
        << captured.str();
}

TEST_F(RetainGradWarningTest, RetainGradSilencesWarning) {
    // With retain_grad() set, accessing grad() on a non-leaf must not
    // trigger the warning, even in a binary where the once_flag is
    // already consumed: we assert *no* new "non-leaf" substring appears
    // during this capture window.
    Variable x(ones({4}, DType::Float32, Device::cpu()), /*requires_grad=*/true);
    auto y = x * 3.0f;
    ASSERT_FALSE(y.is_leaf());  // latch the non-leaf flag before marking.
    y.retain_grad();
    auto loss = sum(y);
    loss.backward();

    ASSERT_TRUE(y.retains_grad());

    std::stringstream captured;
    std::streambuf* old_cout_buf = std::cout.rdbuf(captured.rdbuf());
    const auto& g = y.grad();
    (void)g;
    std::cout.rdbuf(old_cout_buf);

    EXPECT_EQ(captured.str().find("non-leaf"), std::string::npos)
        << "retain_grad()-marked Variable should not emit the warning, got: "
        << captured.str();
}

} // namespace
} // namespace tenzor
