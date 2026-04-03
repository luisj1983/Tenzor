/**
 * @file test_functional.cpp
 * @brief Tests for nn::functional API
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/nn/functional.hpp"

namespace F = tenzor::nn::functional;
using namespace tenzor;
using namespace tenzor::testing;

class FunctionalTest : public BackendTest {};

TEST_P(FunctionalTest, ReluZerosNegatives) {
    auto t = zeros({4}, DType::Float32, device);
    Variable input(t);
    auto output = F::relu(input);
    auto out_cpu = output.tensor().to(Device::cpu());
    for (int i = 0; i < 4; ++i) {
        EXPECT_GE(out_cpu.data<float>()[i], 0.0f);
    }
}

TEST_P(FunctionalTest, SigmoidOutputRange) {
    auto t = zeros({4}, DType::Float32, device);
    Variable input(t);
    auto output = F::sigmoid(input);
    auto out_cpu = output.tensor().to(Device::cpu());
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_cpu.data<float>()[i], 0.5f, 1e-5f);
    }
}

TEST_P(FunctionalTest, SoftmaxSumsToOne) {
    auto t = rand({2, 5}, DType::Float32, device);
    Variable input(t);
    auto output = F::softmax(input, -1);
    auto row_sums = sum(output.tensor(), 1).to(Device::cpu());
    EXPECT_NEAR(row_sums.data<float>()[0], 1.0f, 1e-4f);
    EXPECT_NEAR(row_sums.data<float>()[1], 1.0f, 1e-4f);
}

TEST_P(FunctionalTest, DropoutEvalIsIdentity) {
    auto t = ones({100}, DType::Float32, device);
    Variable input(t);
    auto output = F::dropout(input, 0.5, false);
    auto out_cpu = output.tensor().to(Device::cpu());
    for (int i = 0; i < 100; ++i) {
        EXPECT_FLOAT_EQ(out_cpu.data<float>()[i], 1.0f);
    }
}

TEST_P(FunctionalTest, L1LossCorrect) {
    auto a = Variable(ones({4}, DType::Float32, device));
    auto b = Variable(zeros({4}, DType::Float32, device));
    auto loss = F::l1_loss(a, b);
    auto loss_cpu = loss.tensor().to(Device::cpu());
    EXPECT_NEAR(loss_cpu.data<float>()[0], 1.0f, 1e-6f);
}

TEST_P(FunctionalTest, LinearOutputShape) {
    auto x = Variable(rand({2, 3}, DType::Float32, device));
    auto w = Variable(rand({4, 3}, DType::Float32, device));
    auto b = Variable(zeros({4}, DType::Float32, device));
    auto output = F::linear(x, w, b);
    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 4);
}

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    FunctionalTest,
    ::testing::Values("cpu", "cuda", "vulkan", "oneapi", "rocm")
);
