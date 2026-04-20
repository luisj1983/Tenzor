/**
 * @file test_requires_grad_and_detach.cpp
 * @brief E4: Pin the requires_grad=false and detach() semantics for every
 * core op category.
 *
 * Two invariants this suite enforces:
 *
 *   1. **requires_grad=false inputs do not receive a grad.**
 *      If a leaf Variable has requires_grad=false, its .grad() must remain
 *      null after backward() through any op that uses it. Silent population
 *      of a grad on a non-differentiable leaf would waste memory and, worse,
 *      confuse downstream code that uses .has_grad() as a sentinel.
 *
 *   2. **detach() blocks backward propagation.**
 *      x → op → .detach() → op → loss → backward() must NOT populate x.grad().
 *      A broken detach() would silently leak gradients through frozen
 *      subgraphs (common failure mode for teacher networks / EMA / target
 *      networks in RL).
 *
 * The ops chosen are a representative cross-section of the categories: math,
 * reduction, activation, shape, linear. Adding a new op + backward pair
 * should also add a case here.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/functional.hpp>

using namespace tenzor;

namespace {

// Build a simple forward function F: Variable -> Variable that produces a
// scalar. Returns the Variable loss so the caller can call .backward().
template <typename F>
Variable scalar_of(F&& f, const Variable& x) {
    return sum(f(x));
}

}  // namespace

// ---------------------------------------------------------------------------
// Invariant 1: requires_grad=false inputs never receive a grad.
// ---------------------------------------------------------------------------

TEST(RequiresGradFalse, AddDoesNotPopulateRhsGrad) {
    Variable a(randn({4, 4}), /*requires_grad=*/true);
    Variable b(randn({4, 4}), /*requires_grad=*/false);
    auto loss = sum(a + b);
    loss.backward();
    EXPECT_TRUE(a.has_grad());
    EXPECT_FALSE(b.has_grad()) << "requires_grad=false input received a grad";
}

TEST(RequiresGradFalse, MatMulDoesNotPopulateWeightGrad) {
    Variable x(randn({2, 8}), true);
    Variable w(randn({8, 4}), false);
    auto loss = sum(matmul(x, w));
    loss.backward();
    EXPECT_TRUE(x.has_grad());
    EXPECT_FALSE(w.has_grad()) << "non-trainable weight received a grad";
}

TEST(RequiresGradFalse, ReluOnNonGradLeafLeavesItUngradded) {
    Variable x(randn({6}) + 1.0f, /*requires_grad=*/false);
    auto y = nn::functional::relu(x);
    // Downstream op depends on a requires_grad=true input too so backward has
    // something to do; the non-grad x must still come out ungradded.
    Variable w(randn({6}), true);
    auto loss = sum(y * w);
    loss.backward();
    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(w.has_grad());
}

TEST(RequiresGradFalse, OutputOfNonGradOpHasNoGradFn) {
    Variable x(randn({4}), /*requires_grad=*/false);
    auto y = tenzor::sin(x);
    EXPECT_FALSE(y.requires_grad())
        << "sin(non-grad) should not carry requires_grad";
    EXPECT_EQ(y.grad_fn(), nullptr)
        << "sin(non-grad) should not have a grad_fn";
}

// ---------------------------------------------------------------------------
// Invariant 2: .detach() blocks backward propagation through that point.
// ---------------------------------------------------------------------------

TEST(Detach, BlocksGradientPropagation) {
    Variable x(randn({3, 4}), true);
    auto y = tenzor::exp(x);
    auto y_detached = y.detach();
    // Multiply with a fresh trainable var so backward has a reason to fire.
    Variable w(randn({3, 4}), true);
    auto loss = sum(y_detached * w);
    loss.backward();
    EXPECT_FALSE(x.has_grad())
        << "detach() did not block grad propagation — x got a gradient";
    EXPECT_TRUE(w.has_grad())
        << "w is independent of the detach point and must still receive grad";
}

TEST(Detach, DetachedOutputHasNoGradFn) {
    Variable x(randn({4}), true);
    auto y = x * x;
    ASSERT_NE(y.grad_fn(), nullptr);
    auto y_detached = y.detach();
    EXPECT_EQ(y_detached.grad_fn(), nullptr)
        << "detach() must strip grad_fn";
    EXPECT_FALSE(y_detached.requires_grad())
        << "detach() must produce a non-grad Variable";
}

TEST(Detach, DetachDoesNotModifyOriginal) {
    Variable x(randn({4}), true);
    auto y = x * x;
    auto y_detached = y.detach();
    auto loss = sum(y);  // go through original y, NOT detached.
    loss.backward();
    EXPECT_TRUE(x.has_grad())
        << "detach() should not have disturbed the original graph";
}

TEST(Detach, FrozenTeacherPattern) {
    // Common RL / distillation pattern: teacher output is detached so
    // gradients flow only into the student parameters. Verify that wiring.
    Variable teacher_input(randn({2, 4}), /*teacher inputs fixed*/ false);
    Variable teacher_w(randn({4, 4}), /*teacher params fixed*/ false);
    auto teacher_out = matmul(teacher_input, teacher_w).detach();

    Variable student_input(randn({2, 4}), true);
    Variable student_w(randn({4, 4}), true);
    auto student_out = matmul(student_input, student_w);

    auto loss = sum((student_out - teacher_out) * (student_out - teacher_out));
    loss.backward();

    EXPECT_TRUE(student_input.has_grad());
    EXPECT_TRUE(student_w.has_grad());
    EXPECT_FALSE(teacher_input.has_grad())
        << "teacher input (frozen) must not receive a gradient";
    EXPECT_FALSE(teacher_w.has_grad())
        << "teacher weight (frozen) must not receive a gradient";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    try { tenzor::initialize(); } catch (...) {}
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
