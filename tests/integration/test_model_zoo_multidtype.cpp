/**
 * @file test_model_zoo_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for model zoo (inference only)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/models/resnet.hpp>
#include "../multi_backend_dtype_fixture.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class ModelZooMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(ModelZooMultiDTypeTest, ResNet18Creation) {
    auto model = models::resnet18();
    ASSERT_NE(model, nullptr);
    convert_model(model);

    auto input = createInput({1, 3, 32, 32}, false);
    auto output = model->forward(input);

    EXPECT_EQ(output.tensor().shape()[0], 1);
    expectDevice(output.tensor());
    expectDType(output.tensor());
}

TEST_P(ModelZooMultiDTypeTest, ResNet18ParameterCount) {
    auto model = models::resnet18();
    convert_model(model);

    auto params = model->parameters();
    EXPECT_GT(params.size(), 0) << "Model should have parameters";

    size_t total = countParameters(params);
    EXPECT_GT(total, 100000) << "ResNet18 should have >100k parameters";
}

TEST_P(ModelZooMultiDTypeTest, ResNet18EvalMode) {
    auto model = models::resnet18();
    convert_model(model);
    model->eval();

    auto input = createInput({2, 3, 32, 32}, false);
    auto output = model->forward(input);
    EXPECT_EQ(output.tensor().shape()[0], 2);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ModelZooMultiDTypeTest);
