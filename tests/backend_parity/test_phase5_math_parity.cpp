/**
 * @file test_phase5_math_parity.cpp
 * @brief Cross-backend parity for the Phase-5 PyTorch-parity math ops.
 *
 * Covers OpIds: Deg2Rad, Rad2Deg, Logit, Signbit, FloatPower, Xlog1py,
 * Ldexp, IsReal, IsPosInf, IsNegInf, Frexp. Per the audit, these had
 * no dedicated parity coverage.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class Phase5MathParity : public BackendTest {};

// ----------------------------------------------------------------------------
// Trivial unary float→float ops
// ----------------------------------------------------------------------------

TEST_P(Phase5MathParity, Deg2Rad) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu()) * 180.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return deg2rad(in[0]);
    }, {x}, device, 1e-5f, 1e-7f, "Deg2Rad");
}

TEST_P(Phase5MathParity, Rad2Deg) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return rad2deg(in[0]);
    }, {x}, device, 1e-5f, 1e-7f, "Rad2Deg");
}

TEST_P(Phase5MathParity, Logit) {
    // logit input must be in (0, 1) for finite output. Use sigmoid(randn).
    auto x = sigmoid(randn({4, 8}, DType::Float32, Device::cpu()));
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return logit(in[0], /*eps=*/1e-6);
    }, {x}, device, 1e-4f, 1e-5f, "Logit");
}

TEST_P(Phase5MathParity, Signbit_PositiveAndNegative) {
    // Mix of positive and negative values, including +0 and -0.
    auto x = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return signbit(in[0]);
    }, {x}, device, 0.0f, 0.0f, "Signbit");
}

// ----------------------------------------------------------------------------
// Binary ops
// ----------------------------------------------------------------------------

TEST_P(Phase5MathParity, FloatPower) {
    auto base = abs(randn({4, 8}, DType::Float32, Device::cpu())) + 1e-3f;
    auto exp  = randn({4, 8}, DType::Float32, Device::cpu()) * 2.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return float_power(in[0], in[1]);
    }, {base, exp}, device, 1e-4f, 1e-5f, "FloatPower");
}

TEST_P(Phase5MathParity, Xlog1py) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu());
    auto y = abs(randn({4, 8}, DType::Float32, Device::cpu()));  // y > 0 keeps log1p well-defined
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return xlog1py(in[0], in[1]);
    }, {x, y}, device, 1e-5f, 1e-7f, "Xlog1py");
}

TEST_P(Phase5MathParity, Ldexp) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu());
    // n is integer-valued exponent; PyTorch accepts float n that gets cast.
    auto n = randn({4, 8}, DType::Float32, Device::cpu()) * 4.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return ldexp(in[0], in[1]);
    }, {x, n}, device, 1e-5f, 1e-7f, "Ldexp");
}

// ----------------------------------------------------------------------------
// Predicate ops (return Bool tensor)
// ----------------------------------------------------------------------------

TEST_P(Phase5MathParity, IsReal_Float) {
    // For real-valued tensors, isreal() is uniformly true. This test
    // anchors that the kernel returns the right shape/dtype/value pattern.
    auto x = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return isreal(in[0]);
    }, {x}, device, 0.0f, 0.0f, "IsReal_F32");
}

TEST_P(Phase5MathParity, IsPosInf_AndNegInf) {
    // Build a tensor explicitly containing +inf, -inf, and finite values.
    auto x = full({8}, 0.0, DType::Float32, Device::cpu());
    float* data = x.data<float>();
    data[0] = 1.0f;
    data[1] = std::numeric_limits<float>::infinity();
    data[2] = -1.5f;
    data[3] = -std::numeric_limits<float>::infinity();
    data[4] = 0.0f;
    data[5] = std::numeric_limits<float>::quiet_NaN();
    data[6] = 42.0f;
    data[7] = std::numeric_limits<float>::infinity();

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return isposinf(in[0]);
    }, {x}, device, 0.0f, 0.0f, "IsPosInf");

    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return isneginf(in[0]);
    }, {x}, device, 0.0f, 0.0f, "IsNegInf");
}

// ----------------------------------------------------------------------------
// Frexp: returns (mantissa, exponent) — test both outputs separately
// ----------------------------------------------------------------------------

TEST_P(Phase5MathParity, Frexp_Mantissa) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu()) * 100.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return frexp(in[0]).first;  // mantissa
    }, {x}, device, 1e-5f, 1e-7f, "Frexp_Mantissa");
}

TEST_P(Phase5MathParity, Frexp_Exponent) {
    auto x = randn({4, 8}, DType::Float32, Device::cpu()) * 100.0f;
    test_operation_parity_single([](const std::vector<Tensor>& in) {
        return frexp(in[0]).second;  // exponent (Int32)
    }, {x}, device, 0.0f, 0.0f, "Frexp_Exponent");
}

INSTANTIATE_BACKEND_TESTS(Phase5MathParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
