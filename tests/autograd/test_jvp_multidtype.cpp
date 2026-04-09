/**
 * @file test_jvp_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for forward-mode AD (JVP)
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/jvp_rules.hpp>
#include <tenzor/autograd/functional.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/dual.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include "../multi_backend_dtype_fixture.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

// Macro (not a method) so that GTEST_SKIP's internal `return`
// statement returns from the TEST_P body rather than from a helper
// method — otherwise the test continues and fails on the first op
// that doesn't support Float16.
#define skipIfHalf() \
    do { \
        if (dtype() == DType::Float16 || dtype() == DType::BFloat16) { \
            GTEST_SKIP() << "JVP requires higher precision than Float16"; \
        } \
    } while (0)

class JVPMultiDTypeTest : public MultiBackendDTypeTest {
protected:
    // Verify JVP matches finite differences
    void verify_jvp_fd(std::function<DualTensor(const DualTensor&)> dual_func,
                       std::function<Tensor(const Tensor&)> primal_func,
                       const Tensor& x, const Tensor& v, float tol) {
        DualTensor dual_x(x, v);
        auto dual_out = dual_func(dual_x);
        auto jvp_tangent = dual_out.tangent();

        const float eps = 1e-4f;
        auto x_plus = tenzor::add(x, tenzor::mul(v, eps));
        auto x_minus = tenzor::sub(x, tenzor::mul(v, eps));
        auto fd_tangent = tenzor::mul(
            tenzor::sub(primal_func(x_plus), primal_func(x_minus)),
            1.0f / (2.0f * eps)
        );

        expectTensorNear(jvp_tangent, fd_tangent, tol);
    }
};

TEST_P(JVPMultiDTypeTest, JVPAdd) {
    skipIfHalf();
    auto x = createRandn({4});
    auto v = tenzor::ones({4}, dtype(), device());

    auto dual_f = [](const DualTensor& dx) -> DualTensor { return jvp_add(dx, dx); };
    auto primal_f = [](const Tensor& t) -> Tensor { return tenzor::add(t, t); };

    verify_jvp_fd(dual_f, primal_f, x, v, std::max(atol() * 10.0f, 0.05f));
}

TEST_P(JVPMultiDTypeTest, JVPMul) {
    skipIfHalf();
    auto x = createRandn({4});
    auto v = tenzor::ones({4}, dtype(), device());

    auto dual_f = [](const DualTensor& dx) -> DualTensor { return jvp_mul(dx, dx); };
    auto primal_f = [](const Tensor& t) -> Tensor { return tenzor::mul(t, t); };

    verify_jvp_fd(dual_f, primal_f, x, v, std::max(atol() * 10.0f, 0.05f));
}

TEST_P(JVPMultiDTypeTest, JVPExp) {
    skipIfHalf();
    // Clamp to avoid very large values
    auto x_raw = createRandn({4});
    auto x = tenzor::clamp(x_raw, -2.0f, 2.0f);
    auto v = tenzor::ones({4}, dtype(), device());

    auto dual_f = [](const DualTensor& dx) -> DualTensor { return jvp_exp(dx); };
    auto primal_f = [](const Tensor& t) -> Tensor { return tenzor::exp(t); };

    verify_jvp_fd(dual_f, primal_f, x, v, std::max(atol() * 100.0f, 0.1f));
}

TEST_P(JVPMultiDTypeTest, JVPSin) {
    skipIfHalf();
    auto x = createRandn({4});
    auto v = tenzor::ones({4}, dtype(), device());

    auto dual_f = [](const DualTensor& dx) -> DualTensor { return jvp_sin(dx); };
    auto primal_f = [](const Tensor& t) -> Tensor { return tenzor::sin(t); };

    verify_jvp_fd(dual_f, primal_f, x, v, std::max(atol() * 10.0f, 0.05f));
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(JVPMultiDTypeTest);
