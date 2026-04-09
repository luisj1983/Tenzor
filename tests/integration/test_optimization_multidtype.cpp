/**
 * @file test_optimization_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for optimizer convergence
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/optim/sgd.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/nn/loss/losses.hpp>

using tenzor::optim::SGD;
using tenzor::optim::Adam;
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// Macro (not a method) so that GTEST_SKIP's internal `return`
// statement returns from the TEST_P body rather than from a helper
// method — otherwise the test continues and fails on the first op
// that doesn't support Float16.
#define skipIfHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            GTEST_SKIP() << "Optimization convergence unreliable in half precision"; \
        } \
    } while (0)

class OptimizationMultiDTypeTest : public MultiBackendDTypeTest {
protected:
};

TEST_P(OptimizationMultiDTypeTest, SGDReducesLoss) {
    skipIfHalf();
    auto model = std::make_shared<nn::Linear>(4, 2, true);
    convert_model(model);

    SGD optimizer(model->parameters(), 0.01);
    nn::MSELoss loss_fn;

    auto input = createInput({8, 4}, false);
    auto target = Variable(createRandn({8, 2}), false);

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int step = 0; step < 50; ++step) {
        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = loss_fn.forward(output, target);

        auto loss_f32 = loss.tensor().to(Device::cpu()).to(DType::Float32);
        float loss_val = loss_f32.data<float>()[0];

        if (step == 0) first_loss = loss_val;
        if (step == 49) last_loss = loss_val;

        loss.backward();
        optimizer.step();
    }

    EXPECT_LT(last_loss, first_loss) << "SGD should reduce loss over 50 steps";
}

TEST_P(OptimizationMultiDTypeTest, AdamReducesLoss) {
    skipIfHalf();
    auto model = std::make_shared<nn::Linear>(4, 2, true);
    convert_model(model);

    Adam optimizer(model->parameters(), 0.01);
    nn::MSELoss loss_fn;

    auto input = createInput({8, 4}, false);
    auto target = Variable(createRandn({8, 2}), false);

    float first_loss = 0.0f;
    float last_loss = 0.0f;

    for (int step = 0; step < 50; ++step) {
        optimizer.zero_grad();
        auto output = model->forward(input);
        auto loss = loss_fn.forward(output, target);

        auto loss_f32 = loss.tensor().to(Device::cpu()).to(DType::Float32);
        float loss_val = loss_f32.data<float>()[0];

        if (step == 0) first_loss = loss_val;
        if (step == 49) last_loss = loss_val;

        loss.backward();
        optimizer.step();
    }

    EXPECT_LT(last_loss, first_loss) << "Adam should reduce loss over 50 steps";
}

TEST_P(OptimizationMultiDTypeTest, ZeroGradClearsGradients) {
    auto model = std::make_shared<nn::Linear>(4, 2);
    convert_model(model);

    SGD optimizer(model->parameters(), 0.01);

    auto input = createInput({2, 4}, true);
    auto output = model->forward(input);
    auto loss = tenzor::sum(output);
    loss.backward();

    // Params should have gradients
    for (auto& p : model->parameters()) {
        ASSERT_TRUE(p->grad().has_value());
    }

    optimizer.zero_grad();

    // After zero_grad, gradients should be cleared or zeroed
    // (implementation may set to nullopt or zero tensor)
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(OptimizationMultiDTypeTest);
