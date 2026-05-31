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
 *
 * Parameterized over all backends via BackendTest: each TEST_P creates its
 * tensors on the fixture's `device`. requires_grad / detach() semantics are
 * device-agnostic graph structure, so the assertions hold identically on
 * every backend.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class RequiresGradFalse : public ::tenzor::testing::BackendTest {};
class Detach : public ::tenzor::testing::BackendTest {};

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

TEST_P(RequiresGradFalse, AddDoesNotPopulateRhsGrad) {
    Variable a(randn({4, 4}, DType::Float32, device), /*requires_grad=*/true);
    Variable b(randn({4, 4}, DType::Float32, device), /*requires_grad=*/false);
    auto loss = sum(a + b);
    loss.backward();
    EXPECT_TRUE(a.has_grad());
    EXPECT_FALSE(b.has_grad()) << "requires_grad=false input received a grad";
}

TEST_P(RequiresGradFalse, MatMulDoesNotPopulateWeightGrad) {
    Variable x(randn({2, 8}, DType::Float32, device), true);
    Variable w(randn({8, 4}, DType::Float32, device), false);
    auto loss = sum(matmul(x, w));
    loss.backward();
    EXPECT_TRUE(x.has_grad());
    EXPECT_FALSE(w.has_grad()) << "non-trainable weight received a grad";
}

TEST_P(RequiresGradFalse, ReluOnNonGradLeafLeavesItUngradded) {
    Variable x(randn({6}, DType::Float32, device) + 1.0f, /*requires_grad=*/false);
    auto y = nn::functional::relu(x);
    // Downstream op depends on a requires_grad=true input too so backward has
    // something to do; the non-grad x must still come out ungradded.
    Variable w(randn({6}, DType::Float32, device), true);
    auto loss = sum(y * w);
    loss.backward();
    EXPECT_FALSE(x.has_grad());
    EXPECT_TRUE(w.has_grad());
}

TEST_P(RequiresGradFalse, OutputOfNonGradOpHasNoGradFn) {
    Variable x(randn({4}, DType::Float32, device), /*requires_grad=*/false);
    auto y = tenzor::sin(x);
    EXPECT_FALSE(y.requires_grad())
        << "sin(non-grad) should not carry requires_grad";
    EXPECT_EQ(y.grad_fn(), nullptr)
        << "sin(non-grad) should not have a grad_fn";
}

// ---------------------------------------------------------------------------
// Invariant 2: .detach() blocks backward propagation through that point.
// ---------------------------------------------------------------------------

TEST_P(Detach, BlocksGradientPropagation) {
    Variable x(randn({3, 4}, DType::Float32, device), true);
    auto y = tenzor::exp(x);
    auto y_detached = y.detach();
    // Multiply with a fresh trainable var so backward has a reason to fire.
    Variable w(randn({3, 4}, DType::Float32, device), true);
    auto loss = sum(y_detached * w);
    loss.backward();
    EXPECT_FALSE(x.has_grad())
        << "detach() did not block grad propagation — x got a gradient";
    EXPECT_TRUE(w.has_grad())
        << "w is independent of the detach point and must still receive grad";
}

TEST_P(Detach, DetachedOutputHasNoGradFn) {
    Variable x(randn({4}, DType::Float32, device), true);
    auto y = x * x;
    ASSERT_NE(y.grad_fn(), nullptr);
    auto y_detached = y.detach();
    EXPECT_EQ(y_detached.grad_fn(), nullptr)
        << "detach() must strip grad_fn";
    EXPECT_FALSE(y_detached.requires_grad())
        << "detach() must produce a non-grad Variable";
}

TEST_P(Detach, DetachDoesNotModifyOriginal) {
    Variable x(randn({4}, DType::Float32, device), true);
    auto y = x * x;
    auto y_detached = y.detach();
    auto loss = sum(y);  // go through original y, NOT detached.
    loss.backward();
    EXPECT_TRUE(x.has_grad())
        << "detach() should not have disturbed the original graph";
}

TEST_P(Detach, FrozenTeacherPattern) {
    // Common RL / distillation pattern: teacher output is detached so
    // gradients flow only into the student parameters. Verify that wiring.
    Variable teacher_input(randn({2, 4}, DType::Float32, device), /*teacher inputs fixed*/ false);
    Variable teacher_w(randn({4, 4}, DType::Float32, device), /*teacher params fixed*/ false);
    auto teacher_out = matmul(teacher_input, teacher_w).detach();

    Variable student_input(randn({2, 4}, DType::Float32, device), true);
    Variable student_w(randn({4, 4}, DType::Float32, device), true);
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

INSTANTIATE_BACKEND_TESTS(RequiresGradFalse);
INSTANTIATE_BACKEND_TESTS(Detach);
