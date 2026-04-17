/**
 * @file test_rprop_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Rprop optimizer
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/nn/optim/rprop.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::testing;

class RpropMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(RpropMultiDTypeTest, BasicStep) {
    auto p1_tensor = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto param1 = std::make_shared<Variable>(p1_tensor, true);
    auto p2_tensor = ones({4}, DType::Float32, device()).to(dtype());
    auto param2 = std::make_shared<Variable>(p2_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {param1, param2};

    auto optimizer = Rprop(params, /*lr=*/0.01);

    param1->set_grad(ones({2, 3}, DType::Float32, device()).to(dtype()));
    param2->set_grad(ones({4}, DType::Float32, device()).to(dtype()));

    auto p1_before = param1->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    optimizer.step();

    auto p1_after = param1->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    EXPECT_NE(p1_before, p1_after);
    EXPECT_LT(p1_after, p1_before);
}

TEST_P(RpropMultiDTypeTest, ConvergenceOnQuadratic) {
    auto x_tensor = ones({1}, DType::Float32, device()).to(dtype()) * 10.0f;
    auto x = std::make_shared<Variable>(x_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {x};

    auto optimizer = Rprop(params, /*lr=*/0.1);

    float initial = x->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    for (int i = 0; i < 50; ++i) {
        optimizer.zero_grad();
        x->set_grad(x->tensor());
        optimizer.step();
    }

    float final_val = x->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];
    EXPECT_LT(std::abs(final_val), std::abs(initial));
}

TEST_P(RpropMultiDTypeTest, StepSizeIncreasesOnConsistentGradient) {
    auto x_tensor = zeros({1}, DType::Float32, device()).to(dtype());
    auto x = std::make_shared<Variable>(x_tensor, true);
    auto optimizer = Rprop({x}, /*lr=*/0.01, /*eta_minus=*/0.5, /*eta_plus=*/1.2);

    x->set_grad(ones({1}, DType::Float32, device()).to(dtype()));
    optimizer.step();
    auto after1 = x->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    x->set_grad(ones({1}, DType::Float32, device()).to(dtype()));
    optimizer.step();
    auto after2 = x->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    float delta1 = std::abs(after1 - 0.0f);
    float delta2 = std::abs(after2 - after1);
    EXPECT_GT(delta2, delta1 * 0.99f);
}

TEST_P(RpropMultiDTypeTest, ZeroGrad) {
    auto p1_tensor = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto param1 = std::make_shared<Variable>(p1_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {param1};

    auto optimizer = Rprop(params, 0.01);
    param1->set_grad(ones({2, 3}, DType::Float32, device()).to(dtype()));

    optimizer.zero_grad();

    if (param1->has_grad()) {
        auto g = param1->grad().value().to(Device::cpu()).to(DType::Float32);
        auto* gp = g.data<float>();
        for (int i = 0; i < 6; ++i) {
            EXPECT_NEAR(gp[i], 0.0f, atol());
        }
    }
}

TEST_P(RpropMultiDTypeTest, LearningRateAccessors) {
    auto p1_tensor = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto param1 = std::make_shared<Variable>(p1_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {param1};

    auto optimizer = Rprop(params, 0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05);

    optimizer.set_lr(0.02);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.02);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(RpropMultiDTypeTest);
