/**
 * @file test_hardtanh_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Hardtanh activation
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/activations/activations.hpp"

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::testing;

class HardtanhMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(HardtanhMultiDTypeTest, DefaultRangeClampsBelowMin) {
    auto t_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = -5.0f; tp[1] = -2.0f; tp[2] = -1.5f; tp[3] = -1.0f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    Hardtanh act;
    auto result = act.forward_impl(input);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], -1.0f, atol()) << "index " << i;
    }
}

TEST_P(HardtanhMultiDTypeTest, DefaultRangeClampsAboveMax) {
    auto t_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = 1.0f; tp[1] = 1.5f; tp[2] = 2.0f; tp[3] = 5.0f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    Hardtanh act;
    auto result = act.forward_impl(input);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(data[i], 1.0f, atol()) << "index " << i;
    }
}

TEST_P(HardtanhMultiDTypeTest, DefaultRangePassesThrough) {
    auto t_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = -0.5f; tp[1] = 0.0f; tp[2] = 0.5f; tp[3] = 0.99f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    Hardtanh act;
    auto result = act.forward_impl(input);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    EXPECT_NEAR(data[0], -0.5f, atol());
    EXPECT_NEAR(data[1], 0.0f, atol());
    EXPECT_NEAR(data[2], 0.5f, atol());
    EXPECT_NEAR(data[3], 0.99f, atol());
}

TEST_P(HardtanhMultiDTypeTest, DefaultRangeMixed) {
    auto t_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = -3.0f; tp[1] = -0.5f; tp[2] = 0.5f; tp[3] = 3.0f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    Hardtanh act;
    auto result = act.forward_impl(input);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    EXPECT_NEAR(data[0], -1.0f, atol());
    EXPECT_NEAR(data[1], -0.5f, atol());
    EXPECT_NEAR(data[2], 0.5f, atol());
    EXPECT_NEAR(data[3], 1.0f, atol());
}

TEST_P(HardtanhMultiDTypeTest, CustomRange) {
    auto t_cpu = zeros({5}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = -10.0f; tp[1] = -5.0f; tp[2] = 0.0f; tp[3] = 5.0f; tp[4] = 10.0f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    Hardtanh act(-2.0, 3.0);
    auto result = act.forward_impl(input);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    EXPECT_NEAR(data[0], -2.0f, atol());
    EXPECT_NEAR(data[1], -2.0f, atol());
    EXPECT_NEAR(data[2], 0.0f, atol());
    EXPECT_NEAR(data[3], 3.0f, atol());
    EXPECT_NEAR(data[4], 3.0f, atol());
}

TEST_P(HardtanhMultiDTypeTest, FunctionalHardtanh) {
    auto t_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = -3.0f; tp[1] = -0.5f; tp[2] = 0.5f; tp[3] = 3.0f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    auto result = hardtanh(input, -1.0, 1.0);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    EXPECT_NEAR(data[0], -1.0f, atol());
    EXPECT_NEAR(data[1], -0.5f, atol());
    EXPECT_NEAR(data[2], 0.5f, atol());
    EXPECT_NEAR(data[3], 1.0f, atol());
}

TEST_P(HardtanhMultiDTypeTest, ExactBoundaryValues) {
    auto t_cpu = zeros({2}, DType::Float32, Device::cpu());
    auto* tp = t_cpu.data<float>();
    tp[0] = -1.0f; tp[1] = 1.0f;
    auto input = Variable(t_cpu.to(dtype()).to(device()), false);

    Hardtanh act;
    auto result = act.forward_impl(input);

    auto out = result.tensor().to(Device::cpu()).to(DType::Float32);
    auto* data = out.data<float>();
    EXPECT_NEAR(data[0], -1.0f, atol());
    EXPECT_NEAR(data[1], 1.0f, atol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HardtanhMultiDTypeTest);
