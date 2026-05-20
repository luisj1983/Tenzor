/**
 * @file test_scatter_add_autograd_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for scatter_add gradient
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/indexing.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include "../grad_flow_helpers.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ScatterAddAutogradMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ScatterAddAutogradMultiDTypeTest, ForwardCorrectness) {
    // Create index tensor on CPU then transfer
    auto input_t = tenzor::zeros({2, 4}, dtype(), device());
    auto src_t = tenzor::ones({2, 2}, dtype(), device());

    auto index_cpu = Tensor({2, 2}, DType::Int64, Device::cpu());
    auto* idx = index_cpu.data<int64_t>();
    idx[0] = 0; idx[1] = 2;
    idx[2] = 1; idx[3] = 3;
    auto index_t = index_cpu.to(device());

    Variable input(input_t, false);
    Variable src(src_t, false);

    auto result = scatter_add(input, 1, index_t, src);
    expectShape(result.tensor(), {2, 4});
    expectDevice(result.tensor());
    expectDType(result.tensor());
}

TEST_P(ScatterAddAutogradMultiDTypeTest, FirstOrderGradient) {
    auto input_t = tenzor::zeros({2, 4}, dtype(), device());
    auto src_t = tenzor::ones({2, 2}, dtype(), device());

    auto index_cpu = Tensor({2, 2}, DType::Int64, Device::cpu());
    auto* idx = index_cpu.data<int64_t>();
    idx[0] = 0; idx[1] = 2;
    idx[2] = 1; idx[3] = 3;
    auto index_t = index_cpu.to(device());

    Variable input(input_t, true);
    Variable src(src_t, true);

    auto result = scatter_add(input, 1, index_t, src);
    auto loss = tenzor::sum(result);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    expectShape(input.grad().value(), {2, 4});

    EXPECT_GRAD_FLOWS(src);
    expectShape(src.grad().value(), {2, 2});
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ScatterAddAutogradMultiDTypeTest);
