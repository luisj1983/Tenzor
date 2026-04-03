#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class SpecialMathTest : public BackendTest {};

// Helper: create a 1-D float tensor from a vector of values
static auto make_tensor(std::vector<float> vals, Device dev) -> Tensor {
    auto t = Tensor::from_blob(vals.data(), {static_cast<int64_t>(vals.size())},
                               DType::Float32, Device::cpu());
    auto out = t.clone();  // Detach from stack-local data
    if (dev.type != Device::Type::CPU) out = out.to(dev);
    return out;
}

// =========================================================================
// Gamma family
// =========================================================================

TEST_P(SpecialMathTest, Gamma) {
    auto input = make_tensor({1.0f, 2.0f, 3.0f, 4.0f, 5.0f}, device);
    auto result = tenzor::gamma(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5);
    EXPECT_NEAR(data[1], 1.0f, 1e-5);
    EXPECT_NEAR(data[2], 2.0f, 1e-5);
    EXPECT_NEAR(data[3], 6.0f, 1e-4);
    EXPECT_NEAR(data[4], 24.0f, 1e-3);
}

TEST_P(SpecialMathTest, Lgamma) {
    auto input = make_tensor({1.0f, 2.0f, 5.0f, 10.0f}, device);
    auto result = tenzor::lgamma(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-5);
    EXPECT_NEAR(data[1], 0.0f, 1e-5);
    EXPECT_NEAR(data[2], std::lgamma(5.0f), 1e-4);
    EXPECT_NEAR(data[3], std::lgamma(10.0f), 1e-3);
}

TEST_P(SpecialMathTest, Digamma) {
    auto input = make_tensor({1.0f, 2.0f, 5.0f}, device);
    auto result = tenzor::digamma(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], -0.5772f, 0.01f);  // ψ(1) = -γ
    EXPECT_NEAR(data[1], 0.4228f, 0.01f);   // ψ(2) = 1 - γ
}

TEST_P(SpecialMathTest, Polygamma) {
    auto input = make_tensor({1.0f, 2.0f}, device);
    auto result = tenzor::polygamma(1, input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 1.6449f, 0.05f);  // ψ¹(1) = π²/6
}

// =========================================================================
// Beta
// =========================================================================

TEST_P(SpecialMathTest, Beta) {
    auto a = make_tensor({1.0f, 2.0f, 1.0f}, device);
    auto b = make_tensor({1.0f, 2.0f, 2.0f}, device);
    auto result = tenzor::beta(a, b).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5);
    EXPECT_NEAR(data[1], 1.0f/6.0f, 1e-4);
    EXPECT_NEAR(data[2], 0.5f, 1e-5);
}

// =========================================================================
// Bessel functions
// =========================================================================

TEST_P(SpecialMathTest, BesselJ0) {
    auto input = make_tensor({0.0f, 1.0f, 5.0f}, device);
    auto result = tenzor::bessel_j0(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5);
    EXPECT_NEAR(data[1], 0.7652f, 0.01f);
}

TEST_P(SpecialMathTest, BesselI0) {
    auto input = make_tensor({0.0f, 1.0f, 2.0f}, device);
    auto result = tenzor::bessel_i0(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5);
    EXPECT_NEAR(data[1], 1.2661f, 0.01f);
}

TEST_P(SpecialMathTest, BesselI1) {
    auto input = make_tensor({0.0f, 1.0f, 2.0f}, device);
    auto result = tenzor::bessel_i1(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-5);
    EXPECT_NEAR(data[1], 0.5652f, 0.01f);
}

// =========================================================================
// ErfInv
// =========================================================================

TEST_P(SpecialMathTest, ErfInv) {
    auto input = make_tensor({0.0f, 0.5f, -0.5f, 0.9f}, device);
    auto result = tenzor::erfinv(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 0.0f, 1e-5);

    // Round-trip: erf(erfinv(x)) ≈ x
    auto roundtrip = tenzor::erf(tenzor::erfinv(input)).to(Device::cpu());
    auto input_cpu = input.to(Device::cpu());
    auto* rt = roundtrip.data<float>();
    auto* inp = input_cpu.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(rt[i], inp[i], 1e-4) << "Round-trip failed at index " << i;
    }
}

// =========================================================================
// Sinc
// =========================================================================

TEST_P(SpecialMathTest, Sinc) {
    auto input = make_tensor({0.0f, 1.0f, 0.5f}, device);
    auto result = tenzor::sinc(input).to(Device::cpu());
    auto* data = result.data<float>();
    EXPECT_NEAR(data[0], 1.0f, 1e-5);
    EXPECT_NEAR(data[1], 0.0f, 1e-5);
    EXPECT_NEAR(data[2], static_cast<float>(2.0 / M_PI), 1e-4);
}

// =========================================================================
// Autograd tests
// =========================================================================

TEST_P(SpecialMathTest, LgammaAutograd) {
    auto x_data = make_tensor({2.0f, 3.0f, 5.0f}, device);
    auto x = Variable(x_data, true);
    auto y = tenzor::lgamma(x);
    auto sum_y = tenzor::sum(y);
    sum_y.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad().value().to(Device::cpu());
    auto expected = tenzor::digamma(x_data).to(Device::cpu());
    auto* gd = grad.data<float>();
    auto* ed = expected.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(gd[i], ed[i], 0.01f) << "lgamma grad mismatch at " << i;
    }
}

TEST_P(SpecialMathTest, BesselI0Autograd) {
    auto x_data = make_tensor({1.0f, 2.0f}, device);
    auto x = Variable(x_data, true);
    auto y = tenzor::bessel_i0(x);
    auto sum_y = tenzor::sum(y);
    sum_y.backward();

    ASSERT_TRUE(x.has_grad());
    auto grad = x.grad().value().to(Device::cpu());
    auto expected = tenzor::bessel_i1(x_data).to(Device::cpu());
    auto* gd = grad.data<float>();
    auto* ed = expected.data<float>();
    for (int i = 0; i < 2; ++i) {
        EXPECT_NEAR(gd[i], ed[i], 0.01f) << "bessel_i0 grad mismatch at " << i;
    }
}

INSTANTIATE_BACKEND_TESTS(SpecialMathTest);
