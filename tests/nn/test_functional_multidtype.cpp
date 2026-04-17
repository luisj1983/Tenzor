/**
 * @file test_functional_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for nn::functional API
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/nn/functional.hpp"

namespace F = tenzor::nn::functional;
using namespace tenzor;
using namespace tenzor::testing;

class FunctionalMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(FunctionalMultiDTypeTest, ReluZerosNegatives) {
    auto t = createZeros({4});
    Variable input(t);
    auto output = F::relu(input);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(out_cpu.data<float>()[i], 0.0f);
    }
}

TEST_P(FunctionalMultiDTypeTest, SigmoidOutputRange) {
    auto t = createZeros({4});
    Variable input(t);
    auto output = F::sigmoid(input);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 0.5f, atol());
    }
}

TEST_P(FunctionalMultiDTypeTest, SoftmaxSumsToOne) {
    auto input = createInput({2, 5}, false);
    auto output = F::softmax(input, -1);
    auto row_sums = sum(output.tensor(), 1).to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(row_sums.data<float>()[0], 1.0f, atol());
    EXPECT_NEAR(row_sums.data<float>()[1], 1.0f, atol());
}

TEST_P(FunctionalMultiDTypeTest, DropoutEvalIsIdentity) {
    auto t = createOnes({100});
    Variable input(t);
    auto output = F::dropout(input, 0.5, false);
    auto out_cpu = output.tensor().to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 100; ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 1.0f, atol());
    }
}

TEST_P(FunctionalMultiDTypeTest, L1LossCorrect) {
    auto a = Variable(createOnes({4}));
    auto b = Variable(createZeros({4}));
    auto loss = F::l1_loss(a, b);
    auto loss_cpu = loss.tensor().to(Device::cpu()).to(DType::Float32);
    EXPECT_NEAR(loss_cpu.data<float>()[0], 1.0f, atol());
}

TEST_P(FunctionalMultiDTypeTest, LinearOutputShape) {
    auto x = createInput({2, 3}, false);
    auto w = createInput({4, 3}, false);
    auto b = Variable(createZeros({4}), false);
    auto output = F::linear(x, w, b);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FunctionalMultiDTypeTest);
