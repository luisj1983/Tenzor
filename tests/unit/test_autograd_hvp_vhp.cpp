/**
 * @file test_autograd_hvp_vhp.cpp
 * @brief Unit tests for Hessian-vector product (hvp) and vector-Hessian product (vhp)
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/functional.hpp"
#include "tenzor/autograd/ops.hpp"
#include <cmath>
#include <functional>
#include <fstream>
#include <sstream>
#include <string>

using namespace tenzor;

class HvpVhpTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

// ============================================================================
// HVP tests
// ============================================================================

TEST_P(HvpVhpTest, HvpQuadraticIdentityHessian) {
    // f(x) = 0.5 * x^T * I * x = 0.5 * sum(x^2)
    // Hessian = I (identity)
    // hvp(f, x, v) = I @ v = v
    auto x = Variable(ones({4}, DType::Float64, device), true);
    float v_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto v = from_data(v_data, {4}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        // 0.5 * sum(x * x)
        auto sq = input * input;
        return sum(sq) * 0.5;
    };

    auto [output, hv] = hvp(func, x, v);

    // H @ v = I @ v = v
    auto hv_cpu = hv.cpu();
    auto hv_data = hv_cpu.data<double>();
    EXPECT_NEAR(hv_data[0], 1.0, 1e-5);
    EXPECT_NEAR(hv_data[1], 2.0, 1e-5);
    EXPECT_NEAR(hv_data[2], 3.0, 1e-5);
    EXPECT_NEAR(hv_data[3], 4.0, 1e-5);
}

TEST_P(HvpVhpTest, HvpQuadraticScaledHessian) {
    // f(x) = 0.5 * (2*x0^2 + 4*x1^2) = x0^2 + 2*x1^2
    // Hessian = diag(2, 4)
    // hvp(f, x, v) = [2*v0, 4*v1]
    float x_data[] = {1.0f, 1.0f};
    auto x = Variable(from_data(x_data, {2}).to(DType::Float64).to(device), true);

    float v_data[] = {3.0f, 5.0f};
    auto v = from_data(v_data, {2}).to(DType::Float64).to(device);

    // Build the diagonal A = diag(2, 4) via element-wise multiply
    auto func = [this](const Variable& input) -> Variable {
        // Manually compute: x0^2 + 2*x1^2
        // which is 0.5 * x^T * diag(2,4) * x
        auto sq = input * input;  // [x0^2, x1^2]
        // We need weights [1, 2] to get [x0^2, 2*x1^2] then sum
        float w[] = {1.0f, 2.0f};
        auto weights = Variable(from_data(w, {2}).to(DType::Float64).to(device), false);
        return sum(sq * weights);
    };

    auto [output, hv] = hvp(func, x, v);

    auto hv_cpu = hv.cpu();
    auto hv_data = hv_cpu.data<double>();
    // Hessian of (x0^2 + 2*x1^2) = diag(2, 4)
    EXPECT_NEAR(hv_data[0], 2.0 * 3.0, 1e-5);  // 6
    EXPECT_NEAR(hv_data[1], 4.0 * 5.0, 1e-5);  // 20
}

// ============================================================================
// VHP tests
// ============================================================================

TEST_P(HvpVhpTest, VhpQuadraticIdentityHessian) {
    // f(x) = 0.5 * sum(x^2), Hessian = I
    // vhp(f, x, v) = v^T @ I = v
    auto x = Variable(ones({4}, DType::Float64, device), true);
    float v_data[] = {1.0f, 2.0f, 3.0f, 4.0f};
    auto v = from_data(v_data, {4}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        auto sq = input * input;
        return sum(sq) * 0.5;
    };

    auto [output, vh] = vhp(func, x, v);

    auto vh_cpu = vh.cpu();
    auto vh_data = vh_cpu.data<double>();
    EXPECT_NEAR(vh_data[0], 1.0, 1e-5);
    EXPECT_NEAR(vh_data[1], 2.0, 1e-5);
    EXPECT_NEAR(vh_data[2], 3.0, 1e-5);
    EXPECT_NEAR(vh_data[3], 4.0, 1e-5);
}

TEST_P(HvpVhpTest, VhpQuadraticScaledHessian) {
    // f(x) = x0^2 + 2*x1^2, Hessian = diag(2, 4)
    // vhp(f, x, v) = v^T @ H = [2*v0, 4*v1] (H is symmetric so same as hvp)
    float x_data[] = {1.0f, 1.0f};
    auto x = Variable(from_data(x_data, {2}).to(DType::Float64).to(device), true);

    float v_data[] = {3.0f, 5.0f};
    auto v = from_data(v_data, {2}).to(DType::Float64).to(device);

    auto func = [this](const Variable& input) -> Variable {
        auto sq = input * input;
        float w[] = {1.0f, 2.0f};
        auto weights = Variable(from_data(w, {2}).to(DType::Float64).to(device), false);
        return sum(sq * weights);
    };

    auto [output, vh] = vhp(func, x, v);

    auto vh_cpu = vh.cpu();
    auto vh_data = vh_cpu.data<double>();
    // v^T @ diag(2,4) = [2*3, 4*5] = [6, 20]
    EXPECT_NEAR(vh_data[0], 6.0, 1e-5);
    EXPECT_NEAR(vh_data[1], 20.0, 1e-5);
}

TEST_P(HvpVhpTest, HvpAndVhpAgreeForSymmetricHessian) {
    // For symmetric H, hvp(f,x,v) and vhp(f,x,v) should give the same result
    // f(x) = 0.5 * sum(x^2), H = I (symmetric)
    auto x = Variable(ones({3}, DType::Float64, device) * 2.0, true);
    float v_data[] = {1.0f, -1.0f, 2.0f};
    auto v = from_data(v_data, {3}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        return sum(input * input) * 0.5;
    };

    auto [out_hvp, hv] = hvp(func, x, v);
    auto [out_vhp, vh] = vhp(func, x, v);

    auto hv_cpu = hv.cpu();
    auto vh_cpu = vh.cpu();
    auto hv_data = hv_cpu.data<double>();
    auto vh_data = vh_cpu.data<double>();

    for (int i = 0; i < 3; ++i) {
        EXPECT_NEAR(hv_data[i], vh_data[i], 1e-5) << "index " << i;
    }
}

// Regression: the Hessian-vector product for a transcendental f must match the
// closed form to high precision. hvp uses a Float64 central difference of the
// gradient (eps=1e-5), which is accurate to ~1e-7 — far tighter than the old
// native-dtype FD path (~1e-4) that the silently-inflated gradcheck tolerance
// used to hide. (An exact analytic H·v awaits the jvp walker's higher-order
// multi-occurrence tangent accumulation; see hvp's implementation note.)
//   f(x) = sum(sin(x))  ->  H = diag(-sin(x))  ->  H·v = -sin(x) ⊙ v
TEST_P(HvpVhpTest, HvpTranscendentalHighPrecision) {
    double x_data[] = {0.3, 1.1, -0.7, 2.4};
    auto x = Variable(from_data(x_data, {4}).to(DType::Float64).to(device), true);
    double v_data[] = {1.0, -2.0, 0.5, 3.0};
    auto v = from_data(v_data, {4}).to(DType::Float64).to(device);

    auto func = [](const Variable& input) -> Variable {
        return sum(sin(input));
    };

    auto [output, hv] = hvp(func, x, v);

    auto hv_cpu = hv.cpu();
    auto hv_data = hv_cpu.data<double>();
    for (int i = 0; i < 4; ++i) {
        double expected = -std::sin(x_data[i]) * v_data[i];
        // 1e-6 is far tighter than the old native-dtype FD path's ~1e-4: only
        // the Float64 central-difference implementation can satisfy it.
        EXPECT_NEAR(hv_data[i], expected, 1e-6) << "index " << i;
    }
}

// Companion: the full Hessian (built from per-basis hvp) is likewise accurate.
//   f(x) = sum(sin(x))  ->  H = diag(-sin(x))
TEST_P(HvpVhpTest, HessianTranscendentalHighPrecision) {
    double x_data[] = {0.3, 1.1, -0.7};
    auto x = Variable(from_data(x_data, {3}).to(DType::Float64).to(device), true);

    auto func = [](const Variable& input) -> Variable {
        return sum(sin(input));
    };

    auto H = hessian(func, x).cpu();
    auto H_data = H.data<double>();
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double expected = (i == j) ? -std::sin(x_data[i]) : 0.0;
            EXPECT_NEAR(H_data[i * 3 + j], expected, 1e-6)
                << "H[" << i << "," << j << "]";
        }
    }
}

INSTANTIATE_BACKEND_TESTS(HvpVhpTest);

// H2 regression: repeated create_graph=true backward passes inside hvp()'s
// grad_func used to form a shared_ptr reference cycle (leaf VariableImpl <->
// its own grad_with_graph_impl_ chain, closed via save_variables_for_backward
// retaining the leaf under HigherOrderGraphRetentionGuard), leaking every
// leaf's VariableImpl and device tensor storage forever — see functional.cpp
// grad_func / audit finding H2. Not parametrized by backend: this is a
// host-side refcounting bug, identical on every device.
static long current_rss_kb() {
    std::ifstream f("/proc/self/status");
    std::string line;
    while (std::getline(f, line)) {
        if (line.rfind("VmRSS:", 0) == 0) {
            std::istringstream iss(line.substr(6));
            long kb = 0;
            iss >> kb;
            return kb;
        }
    }
    return -1;
}

TEST(HvpVhpLeakRegression, RepeatedHvpDoesNotLeakLeafGraph) {
    ::tenzor::testing::EnsureInitialized();
    Tensor x0 = tenzor::full({8}, 1.5, DType::Float32, Device::cpu());
    Tensor v0 = tenzor::full({8}, 1.0, DType::Float32, Device::cpu());
    Variable xvar(x0, false);

    auto func = [](const Variable& p) -> Variable {
        // Must be non-trivial (not purely linear) so p itself becomes a
        // saved operand in the second-order graph, which is what closes the
        // reference cycle.
        Variable y = p * p * p;
        return tenzor::sum(y);
    };

    constexpr int kWarmup = 500;
    constexpr int kIterations = 4000;

    for (int i = 0; i < kWarmup; ++i) {
        auto [out, hv] = tenzor::hvp(func, xvar, v0);
        (void)out;
        (void)hv;
    }
    long rss_before = current_rss_kb();
    ASSERT_GE(rss_before, 0) << "could not read /proc/self/status";

    for (int i = 0; i < kIterations; ++i) {
        auto [out, hv] = tenzor::hvp(func, xvar, v0);
        (void)out;
        (void)hv;
    }
    long rss_after = current_rss_kb();

    // Pre-fix, this leaked the full per-call graph (many KB) every iteration;
    // 4000 iterations grew RSS by tens of MB. A generous 20 MB ceiling
    // comfortably separates "leaking every call" from normal allocator churn.
    EXPECT_LT(rss_after - rss_before, 20 * 1024)
        << "RSS grew by " << (rss_after - rss_before)
        << " kB over " << kIterations
        << " hvp() calls — suspected leaf VariableImpl reference-cycle leak (H2)";
}
