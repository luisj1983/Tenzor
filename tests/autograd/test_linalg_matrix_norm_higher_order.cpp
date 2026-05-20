// test_linalg_matrix_norm_higher_order.cpp
//
// Audit D5: LinalgMatrixNormBackward must build a real Variable-level
// graph for `create_graph=true`, handle the previously-missing ord=-2
// (smallest singular value) case, and reuse the SVD-based outer-product
// math for both ord=+2 (largest SV) and ord=-2 (smallest SV).
//
// Tests:
//   1. ord=2 spectral norm backward (sanity / no regression).
//   2. ord=-2 smallest singular value: gradient is u_K v_K^T (verified
//      against a manual SVD of a diagonal matrix where the answer is
//      obvious).
//   3. ord=1 / ord=+inf still work (no regression on subgradient path).
//   4. BackwardWithVariables_PreservesGraph: result Variable carries
//      a grad_fn descending from the incoming grad.

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

using namespace tenzor;

class D5Test : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

static auto make_matrix(std::vector<float> data, int64_t M, int64_t N) -> Variable {
    auto t = zeros({M, N}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (size_t i = 0; i < data.size() && static_cast<int64_t>(i) < M * N; ++i) {
        p[i] = data[i];
    }
    return Variable(t, /*requires_grad=*/true);
}

TEST_F(D5Test, Spectral_Pos2_Backward_NoRegression) {
    // Diagonal matrix diag(3, 1): σ_max = 3, gradient = e_0 e_0^T.
    auto A = make_matrix({3.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    auto y = matrix_norm(A, /*ord=*/2.0);
    y.backward();
    EXPECT_GRAD_FLOWS(A);
    auto* gp = A.grad().value().data<float>();
    EXPECT_NEAR(gp[0], 1.0f, 1e-4f);  // (0,0)
    EXPECT_NEAR(gp[1], 0.0f, 1e-4f);  // (0,1)
    EXPECT_NEAR(gp[2], 0.0f, 1e-4f);  // (1,0)
    EXPECT_NEAR(gp[3], 0.0f, 1e-4f);  // (1,1)
}

TEST_F(D5Test, Spectral_Neg2_Backward_HitsSmallestSV) {
    // Diagonal matrix diag(3, 1): σ_min = 1, gradient = e_1 e_1^T.
    // Audit D5: previously fell through to col-sum mask code → wrong math.
    auto A = make_matrix({3.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    auto y = matrix_norm(A, /*ord=*/-2.0);
    y.backward();
    EXPECT_GRAD_FLOWS(A);
    auto* gp = A.grad().value().data<float>();
    EXPECT_NEAR(gp[0], 0.0f, 1e-4f);  // (0,0)
    EXPECT_NEAR(gp[1], 0.0f, 1e-4f);  // (0,1)
    EXPECT_NEAR(gp[2], 0.0f, 1e-4f);  // (1,0)
    EXPECT_NEAR(gp[3], 1.0f, 1e-4f);  // (1,1)
}

TEST_F(D5Test, Ord1_Backward_NoRegression) {
    // diag(3, 1): max col sum = 3 (column 0), grad = sign of col 0.
    auto A = make_matrix({3.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    auto y = matrix_norm(A, /*ord=*/1.0);
    y.backward();
    EXPECT_GRAD_FLOWS(A);
    auto* gp = A.grad().value().data<float>();
    EXPECT_NEAR(gp[0], 1.0f, 1e-4f);  // (0,0) sign(3) = 1
    EXPECT_NEAR(gp[1], 0.0f, 1e-4f);
    EXPECT_NEAR(gp[2], 0.0f, 1e-4f);  // (1,0) sign(0) = 0 (the col but zero entry)
    EXPECT_NEAR(gp[3], 0.0f, 1e-4f);
}

TEST_F(D5Test, OrdPosInf_Backward_NoRegression) {
    // diag(3, 1): max row sum = 3 (row 0), grad = sign of row 0.
    auto A = make_matrix({3.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    auto y = matrix_norm(A, /*ord=*/std::numeric_limits<double>::infinity());
    y.backward();
    EXPECT_GRAD_FLOWS(A);
    auto* gp = A.grad().value().data<float>();
    EXPECT_NEAR(gp[0], 1.0f, 1e-4f);
    EXPECT_NEAR(gp[1], 0.0f, 1e-4f);
    EXPECT_NEAR(gp[2], 0.0f, 1e-4f);
    EXPECT_NEAR(gp[3], 0.0f, 1e-4f);
}

TEST_F(D5Test, BackwardWithVariables_PreservesGraph_Spectral) {
    auto fn = std::make_shared<LinalgMatrixNormBackward>(/*ord=*/2.0);
    auto A = make_matrix({3.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    auto norm_t = zeros({}, DType::Float32, Device::cpu());
    norm_t.data<float>()[0] = 3.0f;  // ||A||_2 = 3
    fn->save_for_backward({A.tensor(), norm_t});

    auto grad_t = zeros({}, DType::Float32, Device::cpu());
    grad_t.data<float>()[0] = 1.0f;
    Variable grad(grad_t, /*requires_grad=*/true);

    auto result = fn->backward_with_variables({grad});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].requires_grad());
    EXPECT_NE(result[0].grad_fn(), nullptr)
        << "Variable-level backward must preserve grad_fn through grad_outputs[0]";
}

TEST_F(D5Test, BackwardWithVariables_PreservesGraph_Subgradient) {
    auto fn = std::make_shared<LinalgMatrixNormBackward>(/*ord=*/1.0);
    auto A = make_matrix({3.0f, 0.0f, 0.0f, 1.0f}, 2, 2);
    auto norm_t = zeros({}, DType::Float32, Device::cpu());
    norm_t.data<float>()[0] = 3.0f;
    fn->save_for_backward({A.tensor(), norm_t});

    auto grad_t = zeros({}, DType::Float32, Device::cpu());
    grad_t.data<float>()[0] = 1.0f;
    Variable grad(grad_t, /*requires_grad=*/true);

    auto result = fn->backward_with_variables({grad});
    ASSERT_EQ(result.size(), 1u);
    EXPECT_TRUE(result[0].requires_grad());
    EXPECT_NE(result[0].grad_fn(), nullptr);
}
