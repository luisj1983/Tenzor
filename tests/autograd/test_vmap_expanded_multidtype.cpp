/**
 * @file test_vmap_expanded_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for expanded vmap batching rules
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/vmap.hpp>
#include <tenzor/autograd/ops.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class VmapExpandedMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void verify_vmap(std::function<Variable(const Variable&)> func,
                     const Variable& batched_input, int64_t batch_dim,
                     const char* name) {
        auto vmapped = vmap(func, batched_input, batch_dim);

        // Reference: manual loop-and-stack
        auto input_t = batched_input.tensor();
        int64_t batch_size = input_t.shape()[batch_dim];
        std::vector<Tensor> refs;
        for (int64_t i = 0; i < batch_size; ++i) {
            auto slice = tenzor::select(input_t, batch_dim, i);
            Variable sv(slice, false);
            refs.push_back(func(sv).tensor());
        }
        auto ref = tenzor::stack(refs, batch_dim);

        expectTensorNear(vmapped.tensor(), ref, std::max(atol(), 1e-3f));
    }
};

TEST_P(VmapExpandedMultiDTypeTest, ElementWiseExp) {
    auto x = createInput({4, 3}, false);
    verify_vmap([](const Variable& v) { return tenzor::exp(v); }, x, 0, "exp");
}

TEST_P(VmapExpandedMultiDTypeTest, ElementWiseSin) {
    auto x = createInput({4, 3}, false);
    verify_vmap([](const Variable& v) { return tenzor::sin(v); }, x, 0, "sin");
}

TEST_P(VmapExpandedMultiDTypeTest, ElementWiseSqrt) {
    // Ensure positive input for sqrt
    auto x_raw = createInput({4, 3}, false);
    auto x = Variable(tenzor::abs(x_raw.tensor()) + tenzor::full({4, 3}, 0.1f, dtype(), device()), false);
    verify_vmap([](const Variable& v) { return tenzor::sqrt(v); }, x, 0, "sqrt");
}

TEST_P(VmapExpandedMultiDTypeTest, BatchDim1) {
    auto x = createInput({3, 4}, false);
    verify_vmap([](const Variable& v) { return tenzor::neg(v); }, x, 1, "neg_dim1");
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(VmapExpandedMultiDTypeTest);
