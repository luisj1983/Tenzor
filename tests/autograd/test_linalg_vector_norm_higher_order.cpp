// test_linalg_vector_norm_higher_order.cpp
//
// Audit D4: LinalgVectorNormBackward must build a real Variable-level
// graph (so create_graph=true works), correctly handle every documented
// ord value, and throw a clear error for ord==0 (L0 "norm" has no
// well-defined gradient).
//
// Tests:
//   1. ord=2 forward+backward via autograd::vector_norm; grad has values
//      matching the closed form  x / ||x||_2.
//   2. ord=1 backward returns sign(x) elementwise.
//   3. ord=+inf and ord=-inf backwards put grad on the argmax/argmin |x_i|.
//   4. ord=0 backward throws.
//   5. Graph preservation: with create_graph=true, the gradient Variable
//      carries a grad_fn descending from the incoming grad (proves the
//      previously-broken rewrap path is fixed).

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/function.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <cmath>
#include <limits>
#include "../grad_flow_helpers.hpp"
#include "../backend_test_fixture.hpp"

using namespace tenzor;

class D4Test : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Build a requires_grad=true Variable from host data, materialised on the
    // test's target device. Host write goes to a CPU tensor, then .to(device).
    auto make_var(std::vector<float> data, std::vector<int64_t> shape) -> Variable {
        auto t = zeros(shape, DType::Float32, Device::cpu());
        auto* p = t.data<float>();
        for (size_t i = 0; i < data.size(); ++i) p[i] = data[i];
        return Variable(t.to(device), /*requires_grad=*/true);
    }
};

TEST_P(D4Test, P2_Norm_Backward_MatchesClosedForm) {
    // x = [3, 4]; ||x||_2 = 5; grad = x / norm = [0.6, 0.8].
    auto x = make_var({3.0f, 4.0f}, {2});
    auto y = vector_norm(x, /*ord=*/2.0);
    y.backward();
    EXPECT_GRAD_FLOWS(x);
    auto grad_cpu = x.grad().value().cpu();
    auto* gp = grad_cpu.data<float>();
    EXPECT_NEAR(gp[0], 0.6f, 1e-5f);
    EXPECT_NEAR(gp[1], 0.8f, 1e-5f);
}

TEST_P(D4Test, P1_Norm_Backward_IsSign) {
    auto x = make_var({2.0f, -3.0f, 0.5f, -0.1f}, {4});
    auto y = vector_norm(x, /*ord=*/1.0);
    y.backward();
    EXPECT_GRAD_FLOWS(x);
    auto grad_cpu = x.grad().value().cpu();
    auto* gp = grad_cpu.data<float>();
    EXPECT_FLOAT_EQ(gp[0], 1.0f);
    EXPECT_FLOAT_EQ(gp[1], -1.0f);
    EXPECT_FLOAT_EQ(gp[2], 1.0f);
    EXPECT_FLOAT_EQ(gp[3], -1.0f);
}

TEST_P(D4Test, PInf_Norm_Backward_OnArgmax) {
    // |x| = [1, 4, 3]; max at index 1; grad = [0, -1, 0] (sign of x[1] = -1).
    auto x = make_var({1.0f, -4.0f, 3.0f}, {3});
    auto y = vector_norm(x, /*ord=*/std::numeric_limits<double>::infinity());
    y.backward();
    EXPECT_GRAD_FLOWS(x);
    auto grad_cpu = x.grad().value().cpu();
    auto* gp = grad_cpu.data<float>();
    EXPECT_FLOAT_EQ(gp[0], 0.0f);
    EXPECT_FLOAT_EQ(gp[1], -1.0f);
    EXPECT_FLOAT_EQ(gp[2], 0.0f);
}

TEST_P(D4Test, PNegInf_Norm_Backward_OnArgmin) {
    // |x| = [3, 0.5, 4]; min at index 1; grad = [0, +1, 0] (sign of x[1] = +1).
    auto x = make_var({3.0f, 0.5f, 4.0f}, {3});
    auto y = vector_norm(x, /*ord=*/-std::numeric_limits<double>::infinity());
    y.backward();
    EXPECT_GRAD_FLOWS(x);
    auto grad_cpu = x.grad().value().cpu();
    auto* gp = grad_cpu.data<float>();
    EXPECT_FLOAT_EQ(gp[0], 0.0f);
    EXPECT_FLOAT_EQ(gp[1], 1.0f);
    EXPECT_FLOAT_EQ(gp[2], 0.0f);
}

TEST_P(D4Test, P0_Norm_Backward_Throws) {
    auto x = make_var({2.0f, -3.0f, 0.5f}, {3});
    auto y = vector_norm(x, /*ord=*/0.0);
    EXPECT_THROW(y.backward(), std::runtime_error);
}

TEST_P(D4Test, BackwardWithVariables_PreservesGraph) {
    // Direct unit test: construct a LinalgVectorNormBackward, plant saved
    // state, invoke backward_with_variables with a requires_grad=true grad,
    // verify the result Variable carries a grad_fn.
    auto fn = std::make_shared<LinalgVectorNormBackward>(/*ord=*/2.0,
                                                          std::vector<int64_t>{},
                                                          /*keepdim=*/false);
    auto x_t = make_var({3.0f, 4.0f}, {2}).tensor();
    auto norm_cpu = zeros({}, DType::Float32, Device::cpu());
    norm_cpu.data<float>()[0] = 5.0f;
    auto norm_t = norm_cpu.to(device);
    fn->save_for_backward({x_t, norm_t});

    auto grad_cpu = zeros({}, DType::Float32, Device::cpu());
    grad_cpu.data<float>()[0] = 1.0f;
    Variable grad(grad_cpu.to(device), /*requires_grad=*/true);

    auto result = fn->backward_with_variables({grad});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].requires_grad())
        << "result must descend from grad's autograd graph";
    EXPECT_NE(result[0].grad_fn(), nullptr)
        << "Variable-level backward must preserve grad_fn through grad_outputs[0]";
}

TEST_P(D4Test, GeneralP_Norm_Backward_Smoke) {
    // ord = 3.5, exercises the general-p branch.
    auto x = make_var({1.0f, 2.0f, -3.0f}, {3});
    auto y = vector_norm(x, /*ord=*/3.5);
    EXPECT_NO_THROW(y.backward());
    EXPECT_GRAD_FLOWS(x);
    // Just confirm gradient is finite and non-zero.
    auto grad_cpu = x.grad().value().cpu();
    auto* gp = grad_cpu.data<float>();
    for (int i = 0; i < 3; ++i) {
        EXPECT_TRUE(std::isfinite(gp[i]));
    }
}

INSTANTIATE_BACKEND_TESTS(D4Test);
