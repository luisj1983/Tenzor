/**
 * @file test_autograd_hvp_vhp_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for Hessian-vector product (hvp) and
 *        vector-Hessian product (vhp)
 */

#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/ops.hpp"

using namespace tenzor;
using namespace tenzor::testing;

class HvpVhpMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // HVP/VHP require double-precision second derivatives; use Float64 when
    // the parameterised dtype supports it, otherwise skip.
    void SetUp() override {
        MultiBackendDTypeTest::SetUp();
        // HVP/VHP need at least Float32 precision for meaningful results
    }

    double tol() const {
        if (dtype() == DType::Float64) return 1e-5;
        if (dtype() == DType::Float32) return 1e-3;
        return 5e-2;  // Float16/BFloat16
    }
};

TEST_P(HvpVhpMultiDTypeTest, HvpQuadraticIdentityHessian) {
    // f(x) = 0.5 * sum(x^2), H = I, hvp(f,x,v) = v
    auto x_cpu = ones({4}, DType::Float32, Device::cpu());
    auto x = Variable(x_cpu.to(dtype()).to(device()), true);

    auto v_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* vp = v_cpu.data<float>();
    vp[0] = 1.0f; vp[1] = 2.0f; vp[2] = 3.0f; vp[3] = 4.0f;
    auto v = v_cpu.to(dtype()).to(device());

    auto func = [](const Variable& input) -> Variable {
        auto sq = input * input;
        return sum(sq) * 0.5;
    };

    auto [output, hv] = hvp(func, x, v);

    auto hv_data = hv.to(Device::cpu()).to(DType::Float32).data<float>();
    EXPECT_NEAR(hv_data[0], 1.0f, tol());
    EXPECT_NEAR(hv_data[1], 2.0f, tol());
    EXPECT_NEAR(hv_data[2], 3.0f, tol());
    EXPECT_NEAR(hv_data[3], 4.0f, tol());
}

TEST_P(HvpVhpMultiDTypeTest, HvpQuadraticScaledHessian) {
    // f(x) = x0^2 + 2*x1^2, H = diag(2,4), hvp = [2*v0, 4*v1]
    auto x_cpu = ones({2}, DType::Float32, Device::cpu());
    auto x = Variable(x_cpu.to(dtype()).to(device()), true);

    auto v_cpu = zeros({2}, DType::Float32, Device::cpu());
    v_cpu.data<float>()[0] = 3.0f;
    v_cpu.data<float>()[1] = 5.0f;
    auto v = v_cpu.to(dtype()).to(device());

    auto func = [this](const Variable& input) -> Variable {
        auto sq = input * input;
        auto w_cpu = zeros({2}, DType::Float32, Device::cpu());
        w_cpu.data<float>()[0] = 1.0f;
        w_cpu.data<float>()[1] = 2.0f;
        auto weights = Variable(w_cpu.to(dtype()).to(device()), false);
        return sum(sq * weights);
    };

    auto [output, hv] = hvp(func, x, v);

    auto hv_data = hv.to(Device::cpu()).to(DType::Float32).data<float>();
    EXPECT_NEAR(hv_data[0], 6.0f, tol());
    EXPECT_NEAR(hv_data[1], 20.0f, tol());
}

TEST_P(HvpVhpMultiDTypeTest, VhpQuadraticIdentityHessian) {
    auto x_cpu = ones({4}, DType::Float32, Device::cpu());
    auto x = Variable(x_cpu.to(dtype()).to(device()), true);

    auto v_cpu = zeros({4}, DType::Float32, Device::cpu());
    auto* vp = v_cpu.data<float>();
    vp[0] = 1.0f; vp[1] = 2.0f; vp[2] = 3.0f; vp[3] = 4.0f;
    auto v = v_cpu.to(dtype()).to(device());

    auto func = [](const Variable& input) -> Variable {
        auto sq = input * input;
        return sum(sq) * 0.5;
    };

    auto [output, vh] = vhp(func, x, v);

    auto vh_data = vh.to(Device::cpu()).to(DType::Float32).data<float>();
    EXPECT_NEAR(vh_data[0], 1.0f, tol());
    EXPECT_NEAR(vh_data[1], 2.0f, tol());
    EXPECT_NEAR(vh_data[2], 3.0f, tol());
    EXPECT_NEAR(vh_data[3], 4.0f, tol());
}

TEST_P(HvpVhpMultiDTypeTest, HvpAndVhpAgreeForSymmetricHessian) {
    auto x_cpu = ones({3}, DType::Float32, Device::cpu()) * 2.0f;
    auto x = Variable(x_cpu.to(dtype()).to(device()), true);

    auto v_cpu = zeros({3}, DType::Float32, Device::cpu());
    v_cpu.data<float>()[0] = 1.0f;
    v_cpu.data<float>()[1] = -1.0f;
    v_cpu.data<float>()[2] = 2.0f;
    auto v = v_cpu.to(dtype()).to(device());

    auto func = [](const Variable& input) -> Variable {
        return sum(input * input) * 0.5;
    };

    auto [out_hvp, hv] = hvp(func, x, v);
    auto [out_vhp, vh] = vhp(func, x, v);

    auto hv_data = hv.to(Device::cpu()).to(DType::Float32).data<float>();
    auto vh_data = vh.to(Device::cpu()).to(DType::Float32).data<float>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(hv_data[i], vh_data[i], tol()) << "index " << i;
    }
}

TEST_P(HvpVhpMultiDTypeTest, VhpQuadraticScaledHessian) {
    auto x_cpu = ones({2}, DType::Float32, Device::cpu());
    auto x = Variable(x_cpu.to(dtype()).to(device()), true);

    auto v_cpu = zeros({2}, DType::Float32, Device::cpu());
    v_cpu.data<float>()[0] = 3.0f;
    v_cpu.data<float>()[1] = 5.0f;
    auto v = v_cpu.to(dtype()).to(device());

    auto func = [this](const Variable& input) -> Variable {
        auto sq = input * input;
        auto w_cpu = zeros({2}, DType::Float32, Device::cpu());
        w_cpu.data<float>()[0] = 1.0f;
        w_cpu.data<float>()[1] = 2.0f;
        auto weights = Variable(w_cpu.to(dtype()).to(device()), false);
        return sum(sq * weights);
    };

    auto [output, vh] = vhp(func, x, v);

    auto vh_data = vh.to(Device::cpu()).to(DType::Float32).data<float>();
    EXPECT_NEAR(vh_data[0], 6.0f, tol());
    EXPECT_NEAR(vh_data[1], 20.0f, tol());
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(HvpVhpMultiDTypeTest);
