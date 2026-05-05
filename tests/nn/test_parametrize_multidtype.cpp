/**
 * @file test_parametrize_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for nn::utils::parametrize module
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/nn/utils/parametrize.hpp>
#include <tenzor/nn/layers/linear.hpp>

using namespace tenzor;
using namespace tenzor::testing;
using namespace tenzor::nn;
using namespace tenzor::nn::utils;

class ScaleBy2MD : public Parametrization {
public:
    auto forward(const Tensor& X) -> Tensor override {
        return tenzor::mul(X, 2.0);
    }
};

class AbsParamMD : public Parametrization {
public:
    auto forward(const Tensor& X) -> Tensor override {
        return tenzor::abs(X);
    }
};

class ParametrizeMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        // The parametrization registry keys on raw Module* and is global —
        // a fresh Linear in this test can land on the same address as a
        // prior test's Linear, inheriting its parametrization state. Clear
        // the registry so each TEST_P starts clean.
        ::tenzor::nn::utils::clear_parametrization_registry();
    }
};

TEST_P(ParametrizeMultiDTypeTest, RegisterAndCheck) {
    auto linear = std::make_shared<Linear>(4, 2);
    convert_model(linear);

    EXPECT_FALSE(is_parametrized(*linear));
    EXPECT_FALSE(is_parametrized(*linear, "weight"));

    register_parametrization(linear, "weight", std::make_shared<ScaleBy2MD>());

    EXPECT_TRUE(is_parametrized(*linear));
    EXPECT_TRUE(is_parametrized(*linear, "weight"));
    EXPECT_FALSE(is_parametrized(*linear, "bias"));
}

TEST_P(ParametrizeMultiDTypeTest, RemoveParametrization) {
    auto linear = std::make_shared<Linear>(4, 2);
    convert_model(linear);
    register_parametrization(linear, "weight", std::make_shared<ScaleBy2MD>());
    EXPECT_TRUE(is_parametrized(*linear, "weight"));

    remove_parametrizations(linear, "weight");
    EXPECT_FALSE(is_parametrized(*linear, "weight"));
}

TEST_P(ParametrizeMultiDTypeTest, MultipleParametrizations) {
    auto linear = std::make_shared<Linear>(4, 2);
    convert_model(linear);
    register_parametrization(linear, "weight", std::make_shared<ScaleBy2MD>());
    register_parametrization(linear, "weight", std::make_shared<AbsParamMD>());
    EXPECT_TRUE(is_parametrized(*linear, "weight"));
}

TEST_P(ParametrizeMultiDTypeTest, NonExistentParameterThrows) {
    auto linear = std::make_shared<Linear>(4, 2);
    convert_model(linear);
    EXPECT_THROW(
        register_parametrization(linear, "nonexistent", std::make_shared<ScaleBy2MD>()),
        std::exception);
}

TEST_P(ParametrizeMultiDTypeTest, NullModuleThrows) {
    EXPECT_THROW(
        register_parametrization(nullptr, "weight", std::make_shared<ScaleBy2MD>()),
        std::runtime_error);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(ParametrizeMultiDTypeTest);
