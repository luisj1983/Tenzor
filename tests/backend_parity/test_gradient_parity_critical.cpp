/**
 * @file test_gradient_parity_critical.cpp
 * @brief Cross-backend backward (gradient) parity for the highest-impact ops.
 *
 * tests/backend_parity/ has many forward-only parity files. Per the audit,
 * a backward kernel that diverges between backends would be invisible. This
 * file exercises test_gradient_parity() — which compares both forward AND
 * gradient outputs across backends — for ops that, when broken, cascade
 * silently through training.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

TEST(GradientParityCritical, MatMul) {
    auto a = randn({4, 6}, DType::Float32, Device::cpu());
    auto b = randn({6, 5}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return matmul(in[0], in[1]);
        },
        {a, b}, {},
        1e-5f, 1e-8f, 1e-4f, 1e-6f,
        {}, "GradParity_MatMul");
}

TEST(GradientParityCritical, Add) {
    auto a = randn({8}, DType::Float32, Device::cpu());
    auto b = randn({8}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return in[0] + in[1];
        },
        {a, b}, {}, 1e-5f, 1e-8f, 1e-4f, 1e-6f, {}, "GradParity_Add");
}

TEST(GradientParityCritical, Mul) {
    auto a = randn({8}, DType::Float32, Device::cpu());
    auto b = randn({8}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return in[0] * in[1];
        },
        {a, b}, {}, 1e-5f, 1e-8f, 1e-4f, 1e-6f, {}, "GradParity_Mul");
}

TEST(GradientParityCritical, Softmax_LastDim) {
    auto x = randn({4, 6}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return nn::functional::softmax(in[0], -1);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_Softmax");
}

TEST(GradientParityCritical, GeLU) {
    auto x = randn({16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return nn::functional::gelu(in[0]);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_GeLU");
}

TEST(GradientParityCritical, ReLU) {
    auto x = (randn({16}, DType::Float32, Device::cpu()) + 1.0f);
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return nn::functional::relu(in[0]);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_ReLU");
}

TEST(GradientParityCritical, Sigmoid) {
    auto x = randn({16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return nn::functional::sigmoid(in[0]);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_Sigmoid");
}

TEST(GradientParityCritical, Tanh) {
    auto x = randn({16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return nn::functional::tanh(in[0]);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_Tanh");
}

TEST(GradientParityCritical, SumReduction) {
    auto x = randn({3, 4}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return tenzor::sum(in[0]);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_Sum");
}

TEST(GradientParityCritical, MeanReduction) {
    auto x = randn({3, 4}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            return tenzor::mean(in[0]);
        },
        {x}, {}, 1e-5f, 1e-7f, 1e-4f, 1e-5f, {}, "GradParity_Mean");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    setenv("TENZOR_DISABLE_TF32", "1", 1);
    try {
        if (!::testing::GTEST_FLAG(list_tests)) tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Init failed: " << e.what() << std::endl;
        return 1;
    }
    int rc = RUN_ALL_TESTS();
    try { tenzor::finalize(); } catch (...) {}
    return rc;
}
