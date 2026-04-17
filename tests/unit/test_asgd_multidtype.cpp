/**
 * @file test_asgd_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for ASGD optimizer
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/nn/optim/asgd.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::testing;

class ASGDMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ASGDMultiDTypeTest, BasicStep) {
    auto p1_tensor = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto p2_tensor = ones({4}, DType::Float32, device()).to(dtype());
    auto param1 = std::make_shared<Variable>(p1_tensor, true);
    auto param2 = std::make_shared<Variable>(p2_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {param1, param2};

    auto optimizer = ASGD(params, /*lr=*/0.01);

    param1->set_grad(ones({2, 3}, DType::Float32, device()).to(dtype()));
    param2->set_grad(ones({4}, DType::Float32, device()).to(dtype()));

    auto p1_before = param1->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    optimizer.step();

    auto p1_after = param1->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];

    EXPECT_NE(p1_before, p1_after);
    EXPECT_LT(p1_after, p1_before);
}

TEST_P(ASGDMultiDTypeTest, ConvergenceOnQuadratic) {
    auto x_tensor = ones({3}, DType::Float32, device()).to(dtype()) * 5.0f;
    auto x = std::make_shared<Variable>(x_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {x};

    auto optimizer = ASGD(params, /*lr=*/0.1, /*lambd=*/1e-4, /*alpha=*/0.75, /*t0=*/5.0);

    float initial_norm = 0.0f;
    auto init_cpu = x->tensor().to(Device::cpu()).to(DType::Float32);
    auto* init_data = init_cpu.data<float>();
    for (int i = 0; i < 3; ++i) initial_norm += init_data[i] * init_data[i];

    for (int i = 0; i < 100; ++i) {
        optimizer.zero_grad();
        x->set_grad(x->tensor());
        optimizer.step();
    }

    float final_norm = 0.0f;
    auto result_cpu = x->tensor().to(Device::cpu()).to(DType::Float32);
    auto* result = result_cpu.data<float>();
    for (int i = 0; i < 3; ++i) final_norm += result[i] * result[i];

    EXPECT_LT(final_norm, initial_norm);
}

TEST_P(ASGDMultiDTypeTest, ZeroGrad) {
    auto p1_tensor = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto param1 = std::make_shared<Variable>(p1_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {param1};

    auto optimizer = ASGD(params, 0.01);
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

TEST_P(ASGDMultiDTypeTest, LearningRateAccessors) {
    auto p1_tensor = ones({2, 3}, DType::Float32, device()).to(dtype());
    auto param1 = std::make_shared<Variable>(p1_tensor, true);
    std::vector<std::shared_ptr<Variable>> params = {param1};

    auto optimizer = ASGD(params, 0.05);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.05);

    optimizer.set_lr(0.01);
    EXPECT_DOUBLE_EQ(optimizer.get_lr(), 0.01);
}

TEST_P(ASGDMultiDTypeTest, MultipleStepsDecreaseMonotonically) {
    auto x_tensor = ones({2}, DType::Float32, device()).to(dtype()) * 10.0f;
    auto x = std::make_shared<Variable>(x_tensor, true);
    auto optimizer = ASGD({x}, /*lr=*/0.1, /*lambd=*/1e-4, /*alpha=*/0.75, /*t0=*/0.0);

    float prev_val = x->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];
    for (int i = 0; i < 10; ++i) {
        x->set_grad(ones({2}, DType::Float32, device()).to(dtype()));
        optimizer.step();
        float curr_val = x->tensor().to(Device::cpu()).to(DType::Float32).data<float>()[0];
        EXPECT_LT(curr_val, prev_val) << "Step " << i;
        prev_val = curr_val;
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ASGDMultiDTypeTest);
