/**
 * @file test_gradcheck_parity.cpp
 * @brief Numerical gradient checking across backends
 *
 * Uses finite differences (central differences) to verify that analytical
 * gradients computed by the autograd engine are correct on each backend.
 * This catches bugs where a backend computes the forward pass correctly
 * but has an incorrect backward kernel.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class GradcheckParityTest : public BackendTest {
protected:
    static void SetUpTestSuite() {
        try { tenzor::initialize(); } catch (...) {}
    }
    static void TearDownTestSuite() {
        try { tenzor::finalize(); } catch (...) {}
    }
    void SetUp() override {
        BackendTest::SetUp();
        set_grad_enabled(true);
    }

    /**
     * @brief Run gradcheck on the current device.
     *
     * Computes analytical gradients via backward() and numerical gradients
     * via central finite differences, then verifies they match.
     *
     * The perturbation loop runs on CPU to allow direct data pointer access,
     * but the function is evaluated on the test device.
     *
     * @param fn Function mapping Variable -> scalar Variable
     * @param input_data Input tensor (CPU, Float32)
     * @param eps Finite difference step size
     * @param atol Absolute tolerance for gradient comparison
     */
    void runGradcheck(
        std::function<Variable(const Variable&)> fn,
        const Tensor& input_data,
        float eps = 1e-3f,
        float atol = 1e-2f)
    {
        // Analytical gradient via autograd
        auto x = Variable(input_data.to(device), true);
        auto loss = fn(x);
        loss.backward();
        device.synchronize();
        ASSERT_TRUE(x.has_grad())
            << "No gradient computed on " << backend_name(device);
        auto analytical = x.grad().value().to(Device::cpu());

        // Numerical gradient via central differences
        auto x_cpu = input_data.to(Device::cpu());
        auto numerical = zeros_like(x_cpu);
        auto flat_x = x_cpu.reshape({-1});
        auto flat_num = numerical.reshape({-1});
        int64_t n = flat_x.numel();

        for (int64_t i = 0; i < n; ++i) {
            float orig = flat_x.data<float>()[i];

            // f(x + eps)
            flat_x.data<float>()[i] = orig + eps;
            auto x_plus = Variable(flat_x.reshape(std::vector<int64_t>(input_data.shape().begin(), input_data.shape().end())).to(device), false);
            auto f_plus = fn(x_plus).tensor().to(Device::cpu());
            device.synchronize();

            // f(x - eps)
            flat_x.data<float>()[i] = orig - eps;
            auto x_minus = Variable(flat_x.reshape(std::vector<int64_t>(input_data.shape().begin(), input_data.shape().end())).to(device), false);
            auto f_minus = fn(x_minus).tensor().to(Device::cpu());
            device.synchronize();

            float grad_val = (f_plus.data<float>()[0] - f_minus.data<float>()[0]) / (2.0f * eps);
            flat_num.data<float>()[i] = grad_val;
            flat_x.data<float>()[i] = orig;  // restore
        }

        EXPECT_TRUE(tensors_close(analytical, numerical, 1e-2f, atol))
            << "Gradcheck failed on " << backend_name(device)
            << " (max diff: " << max_abs_diff(analytical, numerical) << ")";
    }
};

// ============================================================================
// Unary Math Function Gradchecks
// ============================================================================

TEST_P(GradcheckParityTest, Sin) {
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 42);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::sin(x));
    }, input);
}

TEST_P(GradcheckParityTest, Cos) {
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 43);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::cos(x));
    }, input);
}

TEST_P(GradcheckParityTest, Exp) {
    // Scale down to avoid overflow: exp(0.5*x)
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 44);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::exp(x * 0.5f));
    }, input);
}

TEST_P(GradcheckParityTest, Log) {
    // Ensure positive inputs: log(|x| + 0.1)
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 45);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::log(tenzor::abs(x) + 0.1f));
    }, input);
}

TEST_P(GradcheckParityTest, Sigmoid) {
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 46);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::sigmoid(x));
    }, input);
}

TEST_P(GradcheckParityTest, Tanh) {
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 47);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::tanh(x));
    }, input);
}

// ============================================================================
// Composite / Multi-op Gradchecks
// ============================================================================

TEST_P(GradcheckParityTest, Softmax) {
    auto input = generate_test_tensor({2, 4}, DType::Float32, Device::cpu(), 48);
    runGradcheck([](const Variable& x) {
        auto s = tenzor::softmax(x, 1);
        return tenzor::sum(s * s);
    }, input);
}

TEST_P(GradcheckParityTest, MatMul) {
    auto input = generate_test_tensor({3, 3}, DType::Float32, Device::cpu(), 49);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(tenzor::matmul(x, tenzor::transpose(x, 0, 1)));
    }, input);
}

TEST_P(GradcheckParityTest, Conv2dGradcheck) {
    // Gradcheck w.r.t. input through a Conv2d with frozen weights.
    // Small input: {1,1,4,4}, small kernel: 1->1, 3x3, pad 1
    auto input = generate_test_tensor({1, 1, 4, 4}, DType::Float32, Device::cpu(), 50);

    // Create conv layer on CPU, copy to device
    nn::Conv2d conv(1, 1, 3, 1, 1);

    // Capture conv parameters as fixed tensors
    auto params = conv.parameters();
    std::vector<Tensor> frozen_params;
    for (auto& p : params) {
        frozen_params.push_back(p->tensor().clone());
    }

    auto test_device = device;
    runGradcheck([&](const Variable& x) {
        // Rebuild conv on the correct device each time
        nn::Conv2d conv_local(1, 1, 3, 1, 1);
        auto local_params = conv_local.parameters();
        for (size_t i = 0; i < frozen_params.size(); ++i) {
            local_params[i]->tensor() = frozen_params[i].clone();
        }
        conv_local.to(x.tensor().device());
        return tenzor::sum(conv_local.forward(x));
    }, input, 1e-3f, 5e-2f);
}

TEST_P(GradcheckParityTest, Cubic) {
    // Control test: f(x) = sum(x^3), f'(x) = 3*x^2
    auto input = generate_test_tensor({2, 3}, DType::Float32, Device::cpu(), 51);
    runGradcheck([](const Variable& x) {
        return tenzor::sum(x * x * x);
    }, input);
}

// ============================================================================

INSTANTIATE_BACKEND_TESTS(GradcheckParityTest);
