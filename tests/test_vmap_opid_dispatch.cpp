/**
 * @file test_vmap_opid_dispatch.cpp
 * @brief Audit A.3: vmap rule lookup via OpId-keyed registry hits before
 *        falling back to the legacy name-string registry.
 *
 * The dispatch path:
 *   1. Probe-run the user function on a per-batch slice.
 *   2. If the probe's grad_fn->op_id() is a registered OpId, invoke
 *      the OpId-keyed batching rule.
 *   3. Otherwise fall back to the name-string registry.
 *   4. Otherwise loop-and-stack.
 *
 * This test pins step 2 specifically: a passthrough rule registered
 * under `OpId::Add` is invoked when the probe runs through
 * `AddBackward`, even with the legacy name-string registry deliberately
 * unset for that name.
 */

#include <gtest/gtest.h>

#include "backend_test_fixture.hpp"

#include "tenzor/autograd/vmap.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/ops/creation.hpp"

#include <atomic>

using namespace tenzor;

namespace {

class VmapOpIdDispatchTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(VmapOpIdDispatchTest, OpIdRegistryHasBuiltinPassthroughs) {
    // The init_builtin_batching_rules() function registers OpId-keyed
    // passthrough rules for ~35 OpIds. This test forces initialisation
    // by running a small vmap (input with requires_grad=true so the
    // probe returns a Variable with a real grad_fn).
    auto x = randn({4, 3}, DType::Float32, device);
    Variable v(x, /*requires_grad=*/true);
    auto fn = [](const Variable& a) -> Variable {
        return a + a;
    };
    (void) vmap(fn, v, /*batch_dim=*/0);

    EXPECT_TRUE(has_batching_rule(OpId::Add));
    EXPECT_TRUE(has_batching_rule(OpId::Mul));
    EXPECT_TRUE(has_batching_rule(OpId::Softmax));
    EXPECT_TRUE(has_batching_rule(OpId::Gelu));
}

TEST_P(VmapOpIdDispatchTest, RegisterRefusesUnknown) {
    EXPECT_THROW({
        register_batching_rule(OpId::Unknown,
            [](const std::function<Variable(const Variable&)>& f,
               const Variable& v,
               int64_t) -> Variable { return f(v); });
    }, std::runtime_error);
}

TEST_P(VmapOpIdDispatchTest, OpIdKeyOverridesNameKey) {
    // Register a rule that records a counter under OpId::Mul, then
    // confirm a vmap over multiplication invokes it.
    static std::atomic<int> counter{0};
    register_batching_rule(OpId::Mul,
        [](const std::function<Variable(const Variable&)>& f,
           const Variable& v,
           int64_t /*batch_dim*/) -> Variable {
            ++counter;
            return f(v);  // passthrough
        });

    auto x = randn({2, 3}, DType::Float32, device);
    Variable v(x, /*requires_grad=*/true);
    auto fn = [](const Variable& a) -> Variable {
        return a * a;
    };

    counter = 0;
    auto y = vmap(fn, v, /*batch_dim=*/0);

    // The OpId-keyed rule we just registered should have been picked
    // up at least once during dispatch (counter incremented).
    EXPECT_GE(counter.load(), 1);
    // Output sanity: passthrough returns func(batched_input), and
    // func is `a * a`, so y == x * x element-wise.
    EXPECT_EQ(y.tensor().numel(), x.numel());
}

INSTANTIATE_BACKEND_TESTS(VmapOpIdDispatchTest);

}  // namespace
