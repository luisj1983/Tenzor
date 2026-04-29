/**
 * @file test_sparse_linear_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for SparseLinear layer
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sparse_linear.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class SparseLinearMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        set_grad_enabled(true);

        // Skip if sparse ops are not available on this backend
        try {
            auto test = tenzor::zeros({2, 2}, dtype(), device());
            (void)test;
        } catch (...) {
            GTEST_SKIP() << "Sparse ops not available on " << backend_name();
        }
    }
};

TEST_P(SparseLinearMultiDTypeTest, ForwardShape) {
    SparseLinear sl(16, 8, /*density=*/0.5, /*bias=*/true);
    convert_model(sl);

    auto input = createInput({4, 16}, false);
    auto output = sl.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 4);
    ASSERT_EQ(output.tensor().shape()[1], 8);
}

TEST_P(SparseLinearMultiDTypeTest, ForwardShapeNoBias) {
    SparseLinear sl(16, 8, /*density=*/0.5, /*bias=*/false);
    convert_model(sl);

    auto input = createInput({4, 16}, false);
    auto output = sl.forward(input);
    ASSERT_EQ(output.tensor().shape()[0], 4);
    ASSERT_EQ(output.tensor().shape()[1], 8);
}

TEST_P(SparseLinearMultiDTypeTest, Backward) {
    SparseLinear sl(8, 4, 0.5, true);
    convert_model(sl);

    auto input = createInput({2, 8}, true);
    auto output = sl.forward(input);
    auto loss = sum(output);
    loss.backward();

    EXPECT_GRAD_FLOWS(input);
    ASSERT_EQ(input.grad().value().shape()[0], 2);
    ASSERT_EQ(input.grad().value().shape()[1], 8);
}

TEST_P(SparseLinearMultiDTypeTest, OutputFinite) {
    SparseLinear sl(8, 4, 0.5, true);
    convert_model(sl);

    auto input = createInput({2, 8}, false);
    auto output = sl.forward(input);

    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out_cpu.data<float>();
    for (int64_t i = 0; i < out_cpu.numel(); ++i) {
        EXPECT_FALSE(std::isnan(data[i]));
    }
}

TEST_P(SparseLinearMultiDTypeTest, DifferentDensities) {
    for (double density : {0.1, 0.5, 0.9}) {
        SparseLinear sl(8, 4, density, true);
        convert_model(sl);

        auto input = createInput({2, 8}, false);
        auto output = sl.forward(input);
        EXPECT_EQ(output.tensor().shape()[0], 2);
        EXPECT_EQ(output.tensor().shape()[1], 4);
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(SparseLinearMultiDTypeTest);
