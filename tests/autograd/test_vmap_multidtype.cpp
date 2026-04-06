/**
 * @file test_vmap_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for vectorized map (vmap) transform
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/vmap.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/indexing.hpp>
#include <tenzor/ops/reduction.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class VmapMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(VmapMultiDTypeTest, VmapIdentity) {
    auto input = createInput({4, 3}, false);

    auto f = [](const Variable& x) -> Variable { return x; };
    auto result = vmap(f, input, 0);

    expectShape(result.tensor(), {4, 3});
    expectTensorNear(result.tensor(), input.tensor());
}

TEST_P(VmapMultiDTypeTest, VmapSquare) {
    auto input = createInput({3, 2}, false);

    auto f = [](const Variable& x) -> Variable { return x * x; };
    auto result = vmap(f, input, 0);

    expectShape(result.tensor(), {3, 2});
    // Verify against direct squaring
    auto expected = tenzor::mul(input.tensor(), input.tensor());
    expectTensorNear(result.tensor(), expected);
}

TEST_P(VmapMultiDTypeTest, VmapNeg) {
    auto input = createInput({5, 4}, false);

    auto f = [](const Variable& x) -> Variable {
        return Variable(tenzor::neg(x.tensor()), false);
    };
    auto result = vmap(f, input, 0);

    expectShape(result.tensor(), {5, 4});
    auto expected = tenzor::neg(input.tensor());
    expectTensorNear(result.tensor(), expected);
}

TEST_P(VmapMultiDTypeTest, VmapBatchDim1) {
    auto input = createInput({3, 4}, false);

    auto f = [](const Variable& x) -> Variable { return x * x; };
    auto result = vmap(f, input, 1);

    expectShape(result.tensor(), {3, 4});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(VmapMultiDTypeTest);
