/**
 * @file test_parametrize.cpp
 * @brief Tests for nn::utils::parametrize module
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/parametrize.hpp>
#include <tenzor/nn/layers/linear.hpp>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::utils;

class ParametrizeTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new ParametrizeTestEnv);

// Simple parametrization: scale by 2
class ScaleBy2 : public Parametrization {
public:
    auto forward(const Tensor& X) -> Tensor override {
        return tenzor::mul(X, 2.0);
    }
};

// Parametrization: absolute value (ensures non-negative)
class AbsParam : public Parametrization {
public:
    auto forward(const Tensor& X) -> Tensor override {
        return tenzor::abs(X);
    }
};

TEST(Parametrize, RegisterAndCheck) {
    auto linear = std::make_shared<Linear>(4, 2);

    EXPECT_FALSE(is_parametrized(*linear));
    EXPECT_FALSE(is_parametrized(*linear, "weight"));

    register_parametrization(linear, "weight", std::make_shared<ScaleBy2>());

    EXPECT_TRUE(is_parametrized(*linear));
    EXPECT_TRUE(is_parametrized(*linear, "weight"));
    EXPECT_FALSE(is_parametrized(*linear, "bias"));
}

TEST(Parametrize, RemoveParametrization) {
    auto linear = std::make_shared<Linear>(4, 2);
    register_parametrization(linear, "weight", std::make_shared<ScaleBy2>());

    EXPECT_TRUE(is_parametrized(*linear, "weight"));

    remove_parametrizations(linear, "weight");

    EXPECT_FALSE(is_parametrized(*linear, "weight"));
}

TEST(Parametrize, MultipleParametrizations) {
    auto linear = std::make_shared<Linear>(4, 2);

    register_parametrization(linear, "weight", std::make_shared<ScaleBy2>());
    register_parametrization(linear, "weight", std::make_shared<AbsParam>());

    EXPECT_TRUE(is_parametrized(*linear, "weight"));
}

TEST(Parametrize, NonExistentParameterThrows) {
    auto linear = std::make_shared<Linear>(4, 2);
    EXPECT_THROW(
        register_parametrization(linear, "nonexistent", std::make_shared<ScaleBy2>()),
        std::exception  // May throw runtime_error or out_of_range
    );
}

TEST(Parametrize, NullModuleThrows) {
    EXPECT_THROW(
        register_parametrization(nullptr, "weight", std::make_shared<ScaleBy2>()),
        std::runtime_error
    );
}

TEST(Parametrize, NullParametrizationThrows) {
    auto linear = std::make_shared<Linear>(4, 2);
    EXPECT_THROW(
        register_parametrization(linear, "weight", nullptr),
        std::runtime_error
    );
}
