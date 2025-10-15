#include <gtest/gtest.h>
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/ops/creation.hpp"
#include <iostream>

using namespace tenzor;

TEST(KLDivDebug, StepByStep) {
    std::cout << "Creating input and target..." << std::endl;
    auto input = Variable(full({2, 3}, -1.0f, DType::Float32), true);
    auto target = Variable(full({2, 3}, 0.5f, DType::Float32), false);

    std::cout << "Creating loss criterion..." << std::endl;
    auto criterion = nn::KLDivLoss("mean");

    std::cout << "Computing forward pass..." << std::endl;
    auto loss = criterion(input, target);

    std::cout << "Loss computed successfully!" << std::endl;
    std::cout << "Loss shape: [";
    for (size_t i = 0; i < loss.shape().size(); ++i) {
        std::cout << loss.shape()[i];
        if (i < loss.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Calling backward..." << std::endl;
    loss.backward();

    std::cout << "Backward completed!" << std::endl;
    EXPECT_TRUE(input.grad().has_value());
}
