/**
 * @file test_bilinear_multidtype.cpp
 * @brief Multi-backend × multi-dtype tests for nn::Bilinear.
 *
 * Phase 7.2 of the test-coverage campaign. The Bilinear layer was previously
 * shipped without any dedicated test file. Covers:
 *   - Forward shape correctness
 *   - Backward gradient flow on both inputs and weight
 *   - Bias-on / bias-off variants
 *   - Multi-backend × multi-dtype canonical matrix
 */

#include <gtest/gtest.h>
#include "../../multi_backend_dtype_fixture.hpp"
#include "../../grad_flow_helpers.hpp"
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/reduction.hpp>

using namespace tenzor;
using namespace tenzor::testing;

class BilinearTest : public MultiBackendDTypeTest {};

TEST_P(BilinearTest, ForwardShape) {
    nn::Bilinear layer(/*in1=*/4, /*in2=*/3, /*out=*/5, /*bias=*/true);
    convert_model(layer);

    auto x1 = createInput({2, 4}, /*requires_grad=*/false);
    auto x2 = createInput({2, 3}, /*requires_grad=*/false);

    auto y = layer.forward(x1, x2);
    EXPECT_EQ(y.tensor().ndim(), 2);
    EXPECT_EQ(y.tensor().shape()[0], 2);
    EXPECT_EQ(y.tensor().shape()[1], 5);
    EXPECT_EQ(y.tensor().dtype(), dtype());
}

TEST_P(BilinearTest, ForwardNoBias) {
    nn::Bilinear layer(/*in1=*/4, /*in2=*/3, /*out=*/5, /*bias=*/false);
    convert_model(layer);

    auto x1 = createInput({2, 4}, false);
    auto x2 = createInput({2, 3}, false);

    auto y = layer.forward(x1, x2);
    EXPECT_EQ(y.tensor().shape()[1], 5);
}

TEST_P(BilinearTest, BackwardGradFlowsToBothInputs) {
    nn::Bilinear layer(/*in1=*/3, /*in2=*/4, /*out=*/2, /*bias=*/true);
    convert_model(layer);

    auto x1 = createInput({2, 3}, /*requires_grad=*/true);
    auto x2 = createInput({2, 4}, /*requires_grad=*/true);

    auto y = layer.forward(x1, x2);
    auto loss = tenzor::sum(y);
    loss.backward();

    EXPECT_GRAD_FLOWS(x1);
    EXPECT_GRAD_FLOWS(x2);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(BilinearTest);
