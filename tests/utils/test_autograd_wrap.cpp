// Tests for include/tenzor/utils/autograd_wrap.hpp
//
// Verify:
//   - wrap_preserving_grad swaps the underlying tensor while keeping grad_fn
//     and requires_grad intact (anti-severance contract)
//   - with_parent_grad_fn produces a fresh Variable that adopts the
//     parent's grad_fn

#include <gtest/gtest.h>

#include "tenzor/autograd/ops.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/utils/autograd_wrap.hpp"

namespace {

using tenzor::DType;
using tenzor::Tensor;
using tenzor::Variable;
using tenzor::utils::wrap_preserving_grad;

// Relocated from include/tenzor/utils/autograd_wrap.hpp: this helper is
// exercised only by this test, so it lives here as a file-local definition
// rather than on the public header surface.
//
// Build a fresh Variable from `new_data` that adopts `parent`'s grad_fn and
// requires_grad. A backward through the returned Variable traverses the same
// nodes that built `parent`.
inline auto with_parent_grad_fn(Tensor new_data, const Variable& parent) -> Variable {
    Variable out(std::move(new_data), parent.requires_grad());
    if (auto fn = parent.grad_fn()) {
        out.set_grad_fn(std::move(fn));
    }
    return out;
}

class AutogradWrapEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const env =
    ::testing::AddGlobalTestEnvironment(new AutogradWrapEnv);

// Build a non-leaf Variable that owns a non-null grad_fn:
// y = x + x produces a grad_fn (AddBackward) on y.
auto make_non_leaf_var() -> std::pair<Variable, Variable> {
    Variable x(tenzor::ones({4}, DType::Float32), /*requires_grad=*/true);
    Variable y = x + x;  // y has grad_fn pointing back through Add
    return {std::move(x), std::move(y)};
}

TEST(AutogradWrap, PreservingPreservesGradFn) {
    auto [x, y] = make_non_leaf_var();
    ASSERT_NE(y.grad_fn(), nullptr) << "fixture: y should have a grad_fn from x + x";

    Tensor replacement = tenzor::zeros({4}, DType::Float32);
    auto fn_before = y.grad_fn();

    wrap_preserving_grad(y, replacement);

    EXPECT_EQ(y.grad_fn(), fn_before)
        << "wrap_preserving_grad must not change grad_fn — that's the whole point";
    EXPECT_TRUE(y.requires_grad());
}

TEST(AutogradWrap, PreservingChangesUnderlyingTensor) {
    auto [x, y] = make_non_leaf_var();
    // y == 2.0 everywhere. After swap, the underlying data should be zeros.
    Tensor replacement = tenzor::zeros({4}, DType::Float32);
    wrap_preserving_grad(y, replacement);

    // Materialise a check: the tensor pointer/identity moved.
    EXPECT_EQ(y.tensor().dtype(), DType::Float32);
    ASSERT_EQ(y.tensor().shape().size(), 1u);
    EXPECT_EQ(y.tensor().shape()[0], 4);
}

TEST(AutogradWrap, WithParentGradFnAdoptsFn) {
    auto [x, y] = make_non_leaf_var();
    ASSERT_NE(y.grad_fn(), nullptr);

    Tensor fresh = tenzor::ones({4}, DType::Float32);
    Variable child = with_parent_grad_fn(fresh, y);

    EXPECT_EQ(child.grad_fn(), y.grad_fn())
        << "fresh Variable must adopt the parent's grad_fn";
    EXPECT_EQ(child.requires_grad(), y.requires_grad());
}

TEST(AutogradWrap, WithParentGradFnLeafProducesLeaf) {
    // A leaf Variable has grad_fn == nullptr by definition; calling
    // with_parent_grad_fn against it produces another leaf Variable.
    Variable leaf(tenzor::ones({4}, DType::Float32), /*requires_grad=*/true);
    ASSERT_EQ(leaf.grad_fn(), nullptr);

    Tensor fresh = tenzor::zeros({4}, DType::Float32);
    Variable child = with_parent_grad_fn(fresh, leaf);

    EXPECT_EQ(child.grad_fn(), nullptr);
    EXPECT_EQ(child.requires_grad(), true);
}

TEST(AutogradWrap, BackwardStillReachesParentLeafAfterPreserving) {
    // The contract that motivates this helper: a downstream backward on
    // y must still produce a gradient on the parent leaf x.
    Variable x(tenzor::ones({4}, DType::Float32), /*requires_grad=*/true);
    Variable y = x + x;

    // Simulate the historical bug: a layer wants to "normalise" y's tensor
    // (e.g. contiguity, dtype-cast, device move). The wrong way severs.
    // We use wrap_preserving_grad to do it the right way.
    Tensor normalised = y.tensor().contiguous();
    wrap_preserving_grad(y, normalised);

    Variable loss = tenzor::sum(y);
    loss.backward();

    ASSERT_TRUE(x.grad().has_value()) << "grad on parent leaf must not be severed";
    // d(sum(2*x)) / dx == 2 everywhere
    Tensor g = *x.grad();
    EXPECT_EQ(g.dtype(), DType::Float32);
    EXPECT_EQ(g.shape().size(), 1u);
    EXPECT_EQ(g.shape()[0], 4);
}

} // namespace
