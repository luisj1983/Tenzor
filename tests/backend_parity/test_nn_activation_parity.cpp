/**
 * @file test_nn_activation_parity.cpp
 * @brief Backend parity tests for activation functions
 *
 * Tests 12 activation functions (SELU, Mish, Softplus, LogSigmoid,
 * Hardshrink, Hardswish, Hardtanh, PReLU, CELU, Threshold, Softsign,
 * RReLU) across all available backends.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Functional activations — tested via test_operation_parity
// ============================================================================

TEST(NNActivationParity, SELU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::selu(x).tensor();
    }, {input}, 1e-5f, 1e-7f, "SELU");
}

TEST(NNActivationParity, Mish) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::mish(x).tensor();
    }, {input}, 1e-5f, 1e-7f, "Mish");
}

TEST(NNActivationParity, Softplus) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return softplus(x).tensor();
    }, {input}, 1e-5f, 1e-7f, "Softplus");
}

TEST(NNActivationParity, LogSigmoid) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::log_sigmoid(x).tensor();
    }, {input}, 1e-5f, 1e-7f, "LogSigmoid");
}

TEST(NNActivationParity, Hardshrink) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::hardshrink(x, 0.5).tensor();
    }, {input}, 1e-5f, 1e-7f, "Hardshrink");
}

TEST(NNActivationParity, Hardswish) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::hardswish(x).tensor();
    }, {input}, 1e-5f, 1e-7f, "Hardswish");
}

TEST(NNActivationParity, Hardtanh) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::hardtanh(x, -1.0, 1.0).tensor();
    }, {input}, 1e-5f, 1e-7f, "Hardtanh");
}

TEST(NNActivationParity, CELU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::celu(x, 1.0).tensor();
    }, {input}, 1e-5f, 1e-7f, "CELU");
}

TEST(NNActivationParity, Threshold) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::threshold(x, 0.5, 0.0).tensor();
    }, {input}, 1e-5f, 1e-7f, "Threshold");
}

TEST(NNActivationParity, Softsign) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::softsign(x).tensor();
    }, {input}, 1e-5f, 1e-7f, "Softsign");
}

// ============================================================================
// Module-based activations — tested via the NN layer loop pattern
// ============================================================================

TEST(NNActivationParity, PReLU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::PReLU prelu(1);
    prelu.eval();
    auto input = randn({32, 64}, DType::Float32, Device::cpu());
    auto ref = prelu.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::PReLU prelu_dev(1);
            prelu_dev.eval();
            auto params_src = prelu.parameters();
            auto params_dst = prelu_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            prelu_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = prelu_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "PReLU failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNActivationParity, RReLU_Eval) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    // RReLU in eval mode uses the midpoint of (lower, upper) as a fixed
    // negative slope, making it deterministic across backends.
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        return nn::rrelu(x, 1.0 / 8.0, 1.0 / 3.0, /*training=*/false).tensor();
    }, {input}, 1e-5f, 1e-7f, "RReLU_Eval");
}



int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
