/**
 * @file test_linalg_norm_ords.cpp
 * @brief Coverage for linalg::norm string-ord variants and vector_norm(ord=0).
 *
 * S6 (release-prep stream): until this test landed, `linalg::norm` accepted
 * only "fro" — every other documented ord ("1", "-1", "2", "-2", "inf",
 * "-inf", "nuc") threw at runtime, and `vector_norm(ord=0)` used an
 * `ax / (ax + 1e-38)` asymptote that was never exactly 1 (silently biasing
 * the L0 count for every real input). Both are now exact.
 *
 * Cross-backend: every suite is parameterized over all backends via
 * BackendTest. Tensors are created on the fixture `device`; gradcheck and
 * finite-difference probes are device-aware (inputs wrap device tensors).
 * Host reads go through `.to(Device::cpu())` before `.data<T>()`.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"

#include <tenzor/tenzor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/linalg.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/ops/reduction.hpp>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/autograd/ops.hpp>
#include <tenzor/backend/fast_dispatch.hpp>

#include <cmath>
#include <limits>
#include <vector>

using namespace tenzor;

// One fixture per suite name (all share BackendTest behaviour).
class LinalgNormOrds : public ::tenzor::testing::BackendTest {};
class LinalgVectorNormNegInf : public ::tenzor::testing::BackendTest {};
class LinalgVectorNormOne : public ::tenzor::testing::BackendTest {};
class LinalgVectorNormOrd0 : public ::tenzor::testing::BackendTest {};
class LinalgVectorNormPosInf : public ::tenzor::testing::BackendTest {};

namespace {

float scalar_f32(const Tensor& t) {
    auto c = t.to(Device::cpu()).to(DType::Float32).contiguous();
    return c.data<float>()[0];
}

double scalar_f64(const Tensor& t) {
    auto c = t.to(Device::cpu()).to(DType::Float64).contiguous();
    return c.data<double>()[0];
}

// 4x3 test matrix (chosen so every advertised norm has a clean closed form).
// Layout (row-major):
//      [  1, -2,  3 ]
//      [  0,  4, -1 ]
//      [ -5,  0,  2 ]
//      [  3,  1,  0 ]
//
// Column abs-sums: |1|+|0|+|-5|+|3| = 9, |-2|+|4|+|0|+|1| = 7, |3|+|-1|+|2|+|0| = 6.
// Row    abs-sums: 6, 5, 7, 4.
//
// Frobenius²: 1 + 4 + 9 + 0 + 16 + 1 + 25 + 0 + 4 + 9 + 1 + 0 = 70.
// → Frobenius = sqrt(70) ≈ 8.366600265.
//
// Spectral / nuclear: computed numerically against svdvals (the test only
// checks `linalg::norm` agrees with `svdvals().max() / .min() / .sum()`).
constexpr int64_t kM = 4;
constexpr int64_t kN = 3;
const std::vector<float> kMatF32 = {
    1.0f, -2.0f,  3.0f,
    0.0f,  4.0f, -1.0f,
   -5.0f,  0.0f,  2.0f,
    3.0f,  1.0f,  0.0f,
};
const std::vector<double> kMatF64 = {
    1.0, -2.0,  3.0,
    0.0,  4.0, -1.0,
   -5.0,  0.0,  2.0,
    3.0,  1.0,  0.0,
};

// Build the test matrices on the requested device. Host data is staged on CPU
// then moved to `device`.
Tensor mk_matrix_f32(const Device& device) {
    return from_data(kMatF32.data(), {kM, kN}, Device::cpu()).contiguous().to(device);
}

Tensor mk_matrix_f64(const Device& device) {
    return from_data(kMatF64.data(), {kM, kN}, Device::cpu()).contiguous().to(device);
}

}  // namespace

// ============================================================================
// linalg::norm — matrix ord variants
// ============================================================================

TEST_P(LinalgNormOrds, FrobeniusFloat32_StillWorks) {
    auto A = mk_matrix_f32(device);
    auto got = linalg::norm(A, "fro");
    EXPECT_NEAR(scalar_f32(got), std::sqrt(70.0f), 1e-4f);
}

TEST_P(LinalgNormOrds, FrobeniusFloat64) {
    auto A = mk_matrix_f64(device);
    auto got = linalg::norm(A, "fro");
    EXPECT_NEAR(scalar_f64(got), std::sqrt(70.0), 1e-9);
}

TEST_P(LinalgNormOrds, One_MaxAbsColumnSum) {
    // max(|col| sum) = max(9, 7, 6) = 9.
    auto A = mk_matrix_f32(device);
    auto got = linalg::norm(A, "1");
    EXPECT_NEAR(scalar_f32(got), 9.0f, 1e-5f);
}

TEST_P(LinalgNormOrds, NegOne_MinAbsColumnSum) {
    // min(9, 7, 6) = 6.
    auto A = mk_matrix_f32(device);
    auto got = linalg::norm(A, "-1");
    EXPECT_NEAR(scalar_f32(got), 6.0f, 1e-5f);
}

TEST_P(LinalgNormOrds, Inf_MaxAbsRowSum) {
    // max(6, 5, 7, 4) = 7.
    auto A = mk_matrix_f32(device);
    auto got = linalg::norm(A, "inf");
    EXPECT_NEAR(scalar_f32(got), 7.0f, 1e-5f);
}

TEST_P(LinalgNormOrds, NegInf_MinAbsRowSum) {
    // min(6, 5, 7, 4) = 4.
    auto A = mk_matrix_f32(device);
    auto got = linalg::norm(A, "-inf");
    EXPECT_NEAR(scalar_f32(got), 4.0f, 1e-5f);
}

TEST_P(LinalgNormOrds, Two_SpectralAgreesWithSvdvalsMax) {
    auto A = mk_matrix_f64(device);
    auto sv = linalg::svdvals(A);  // shape (3,)
    auto sv_cpu = sv.to(Device::cpu()).contiguous();
    auto* sv_data = sv_cpu.data<double>();
    double sigma_max = 0.0;
    for (int64_t i = 0; i < sv_cpu.numel(); ++i) sigma_max = std::max(sigma_max, sv_data[i]);

    auto got = linalg::norm(A, "2");
    EXPECT_NEAR(scalar_f64(got), sigma_max, 1e-9);
}

TEST_P(LinalgNormOrds, NegTwo_SpectralAgreesWithSvdvalsMin) {
    auto A = mk_matrix_f64(device);
    auto sv = linalg::svdvals(A);
    auto sv_cpu = sv.to(Device::cpu()).contiguous();
    auto* sv_data = sv_cpu.data<double>();
    double sigma_min = std::numeric_limits<double>::infinity();
    for (int64_t i = 0; i < sv_cpu.numel(); ++i) sigma_min = std::min(sigma_min, sv_data[i]);

    auto got = linalg::norm(A, "-2");
    EXPECT_NEAR(scalar_f64(got), sigma_min, 1e-9);
}

TEST_P(LinalgNormOrds, Nuc_NuclearAgreesWithSvdvalsSum) {
    auto A = mk_matrix_f64(device);
    auto sv = linalg::svdvals(A);
    auto sv_cpu = sv.to(Device::cpu()).contiguous();
    auto* sv_data = sv_cpu.data<double>();
    double sum = 0.0;
    for (int64_t i = 0; i < sv_cpu.numel(); ++i) sum += sv_data[i];

    auto got = linalg::norm(A, "nuc");
    EXPECT_NEAR(scalar_f64(got), sum, 1e-9);
}

TEST_P(LinalgNormOrds, UnknownOrd_Throws) {
    auto A = mk_matrix_f32(device);
    EXPECT_THROW(linalg::norm(A, "bogus"), std::runtime_error);
}

// ============================================================================
// linalg::vector_norm — ord=0 (count nonzero) and ±inf
// ============================================================================

TEST_P(LinalgVectorNormOrd0, ExactCountOfNonzero_Float32) {
    // 3 nonzero in a 6-vector. Old `ax/(ax+1e-38)` returned ~3.0 but with
    // 1e-19-ish bias per nonzero; here we want exact integer match.
    std::vector<float> v = {0.0f, 1.0f, 0.0f, -2.5f, 0.0f, 3.0f};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/0.0);
    EXPECT_EQ(scalar_f32(got), 3.0f) << "Expected exact L0 count, got " << scalar_f32(got);
}

TEST_P(LinalgVectorNormOrd0, AllZeros_ReturnsZero) {
    std::vector<float> v = {0.0f, 0.0f, 0.0f, 0.0f};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/0.0);
    EXPECT_EQ(scalar_f32(got), 0.0f);
}

TEST_P(LinalgVectorNormOrd0, AllNonzero_ReturnsNumel) {
    std::vector<float> v = {1.0f, -1.0f, 2.0f, -3.0f, 1e-30f};  // tiny but nonzero counts
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/0.0);
    EXPECT_EQ(scalar_f32(got), 5.0f);
}

TEST_P(LinalgVectorNormOrd0, Float64_ExactCount) {
    std::vector<double> v = {0.0, 4.2, 0.0, 0.0, 1e-200};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/0.0);
    EXPECT_EQ(scalar_f64(got), 2.0);
}

TEST_P(LinalgVectorNormPosInf, EqualsMaxAbs) {
    std::vector<float> v = {1.0f, -7.0f, 3.0f, 2.0f, -4.0f};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/std::numeric_limits<double>::infinity());
    EXPECT_NEAR(scalar_f32(got), 7.0f, 1e-6f);
}

TEST_P(LinalgVectorNormNegInf, EqualsMinAbs) {
    std::vector<float> v = {1.0f, -7.0f, 3.0f, 2.0f, -4.0f};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/-std::numeric_limits<double>::infinity());
    EXPECT_NEAR(scalar_f32(got), 1.0f, 1e-6f);
}

TEST_P(LinalgVectorNormOne, EqualsSumAbs) {
    std::vector<float> v = {1.0f, -2.0f, 3.0f, -4.0f};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    auto got = linalg::vector_norm(x, /*ord=*/1.0);
    EXPECT_NEAR(scalar_f32(got), 10.0f, 1e-6f);
}

// ============================================================================
// Gradcheck — analytic backward vs numerical finite-difference
// ============================================================================
// We use a hand-rolled finite-difference probe rather than autograd::gradcheck
// because the matrix-norm ords have non-smooth points (argmax/argmin ties) and
// gradcheck's default tolerance is too strict for the subgradient approach.
// The probe takes a single scalar perturbation per element and compares the
// resulting forward delta to the directional derivative computed from the
// analytic gradient.
namespace {

// Run forward + backward of linalg_norm(A, ord) and return (forward_value,
// analytic_gradient_wrt_A). A_in lives on its own device; the Variable, forward
// and backward all run there.
std::pair<double, Tensor> norm_forward_backward(const Tensor& A_in,
                                                const std::string& ord) {
    tenzor::Variable A_var(A_in, /*requires_grad=*/true);
    auto y = tenzor::linalg_norm(A_var, ord);
    y.backward();
    auto grad_opt = A_var.grad();
    if (!grad_opt.has_value()) {
        throw std::runtime_error("norm_forward_backward: grad missing for ord=" + ord);
    }
    return {scalar_f64(y.tensor()), *grad_opt};
}

// Central finite difference along a single random direction d (||d||_F=1).
// Returns (numerical_dot, analytic_dot) where dot = <d, grad>. All forward
// evaluations run on `device`; the host-side direction is built on CPU and the
// perturbed matrices are moved to `device` before norm().
std::pair<double, double> directional_derivative(const Tensor& A,
                                                 const Tensor& grad,
                                                 const std::string& ord,
                                                 double eps,
                                                 const Device& device) {
    // Build a deterministic direction: d[i,j] = sin(i + 2*j + 1)  → nonzero,
    // bounded, no domain accidents. Normalise to Frobenius unit length.
    std::vector<int64_t> shape(A.shape().begin(), A.shape().end());
    int64_t numel = A.numel();
    std::vector<double> d(numel);
    auto A_cpu = A.to(Device::cpu()).to(DType::Float64).contiguous();
    auto g_cpu = grad.to(Device::cpu()).to(DType::Float64).contiguous();
    auto* a_p = A_cpu.data<double>();
    auto* g_p = g_cpu.data<double>();
    double dnorm = 0.0;
    for (int64_t i = 0; i < numel; ++i) {
        d[i] = std::sin(static_cast<double>(i) * 0.7 + 0.3);
        dnorm += d[i] * d[i];
    }
    dnorm = std::sqrt(dnorm);
    for (int64_t i = 0; i < numel; ++i) d[i] /= dnorm;

    // f(A + eps*d) and f(A - eps*d). Reuse A_cpu's data; clone per side.
    std::vector<double> plus(a_p, a_p + numel);
    std::vector<double> minus(a_p, a_p + numel);
    for (int64_t i = 0; i < numel; ++i) { plus[i]  += eps * d[i]; }
    for (int64_t i = 0; i < numel; ++i) { minus[i] -= eps * d[i]; }
    auto Aplus  = from_data(plus.data(),  shape, Device::cpu()).contiguous().to(device);
    auto Aminus = from_data(minus.data(), shape, Device::cpu()).contiguous().to(device);
    double y_plus  = scalar_f64(linalg::norm(Aplus,  ord));
    double y_minus = scalar_f64(linalg::norm(Aminus, ord));
    double numerical = (y_plus - y_minus) / (2.0 * eps);

    double analytic = 0.0;
    for (int64_t i = 0; i < numel; ++i) analytic += d[i] * g_p[i];
    return {numerical, analytic};
}

}  // namespace

TEST_P(LinalgNormOrds, Gradcheck_Fro) {
    auto A = mk_matrix_f64(device);
    auto [y, grad] = norm_forward_backward(A, "fro");
    EXPECT_GT(y, 0.0);
    auto [num, ana] = directional_derivative(A, grad, "fro", 1e-5, device);
    EXPECT_NEAR(num, ana, 1e-5) << "fro: numerical=" << num << " analytic=" << ana;
}

// H25: NormBackward_Linalg's "fro" branch used to divide by norm_val with
// no zero-guard -- for the exact-zero matrix, norm_val==0 gave
// scale=grad/0=Inf, then Inf*A(==0)=NaN, even though the correct
// subgradient at the origin is 0 (matches PyTorch's convention and the
// already-guarded LinalgVectorNormBackward's identical L2-norm formula).
// This is a first-order bug -- reproducible on a plain backward() call, no
// create_graph needed -- so it's tested with a direct backward() (Variable
// path exercises backward_with_variables() only under create_graph=true;
// covered separately below).
TEST_P(LinalgNormOrds, Fro_ZeroMatrix_GradientIsZeroNotNaN) {
    auto A = zeros({3, 4}, DType::Float64, device);
    tenzor::Variable A_var(A, /*requires_grad=*/true);
    auto y = tenzor::linalg_norm(A_var, "fro");
    EXPECT_NEAR(scalar_f64(y.tensor()), 0.0, 1e-12);
    y.backward();
    auto grad_opt = A_var.grad();
    ASSERT_TRUE(grad_opt.has_value());
    auto grad_cpu = grad_opt->to(Device::cpu()).to(DType::Float64).contiguous();
    auto* g = grad_cpu.data<double>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        ASSERT_FALSE(std::isnan(g[i])) << "grad[" << i << "] is NaN";
        EXPECT_NEAR(g[i], 0.0, 1e-12) << "grad[" << i << "]";
    }
}

// Same check through backward_with_variables() (create_graph=true path).
TEST_P(LinalgNormOrds, Fro_ZeroMatrix_CreateGraphGradientIsZeroNotNaN) {
    auto A = zeros({3, 4}, DType::Float64, device);
    tenzor::Variable A_var(A, /*requires_grad=*/true);
    auto y = tenzor::linalg_norm(A_var, "fro");
    y.backward(std::nullopt, /*retain_graph=*/false, /*create_graph=*/true);
    auto grad_opt = A_var.grad();
    ASSERT_TRUE(grad_opt.has_value());
    auto grad_cpu = grad_opt->to(Device::cpu()).to(DType::Float64).contiguous();
    auto* g = grad_cpu.data<double>();
    for (int64_t i = 0; i < grad_cpu.numel(); ++i) {
        ASSERT_FALSE(std::isnan(g[i])) << "grad[" << i << "] is NaN";
        EXPECT_NEAR(g[i], 0.0, 1e-12) << "grad[" << i << "]";
    }
}

TEST_P(LinalgNormOrds, Gradcheck_One) {
    auto A = mk_matrix_f64(device);
    auto [y, grad] = norm_forward_backward(A, "1");
    EXPECT_NEAR(y, 9.0, 1e-9);
    // ord=1 is piecewise-linear; the subgradient at our point picks the
    // unique max column (column 0). Verify analytic ≈ numerical along a
    // direction that doesn't tip into another column.
    auto [num, ana] = directional_derivative(A, grad, "1", 1e-6, device);
    EXPECT_NEAR(num, ana, 1e-4) << "1: numerical=" << num << " analytic=" << ana;
}

TEST_P(LinalgNormOrds, Gradcheck_Inf) {
    auto A = mk_matrix_f64(device);
    auto [y, grad] = norm_forward_backward(A, "inf");
    EXPECT_NEAR(y, 7.0, 1e-9);
    auto [num, ana] = directional_derivative(A, grad, "inf", 1e-6, device);
    EXPECT_NEAR(num, ana, 1e-4) << "inf: numerical=" << num << " analytic=" << ana;
}

TEST_P(LinalgNormOrds, Gradcheck_Two) {
    auto A = mk_matrix_f64(device);
    auto [y, grad] = norm_forward_backward(A, "2");
    EXPECT_GT(y, 0.0);
    auto [num, ana] = directional_derivative(A, grad, "2", 1e-5, device);
    // Spectral norm gradient is smooth where σ_max is simple — our matrix
    // has well-separated singular values so the tolerance can be tight.
    EXPECT_NEAR(num, ana, 1e-4) << "2: numerical=" << num << " analytic=" << ana;
}

TEST_P(LinalgNormOrds, Gradcheck_Nuc) {
    auto A = mk_matrix_f64(device);
    auto [y, grad] = norm_forward_backward(A, "nuc");
    EXPECT_GT(y, 0.0);
    auto [num, ana] = directional_derivative(A, grad, "nuc", 1e-5, device);
    // Nuclear norm gradient (U @ Vh) is smooth at full-rank, simple-SV
    // matrices. Loose tolerance to absorb LAPACK SVD round-off across the
    // three evaluations (A+εd, A, A-εd) since each does its own SVD.
    EXPECT_NEAR(num, ana, 5e-4) << "nuc: numerical=" << num << " analytic=" << ana;
}

TEST_P(LinalgVectorNormOrd0, BackwardThrows) {
    // ord=0 is piecewise constant — gradient is zero a.e. but undefined at
    // x_i = 0. PyTorch raises here; Tenzor's LinalgVectorNormBackward does
    // the same (see src/autograd/function_new_ops.cpp).
    std::vector<float> v = {0.0f, 1.0f, 0.0f, -2.5f};
    auto x = from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu()).to(device);
    tenzor::Variable xv(x, /*requires_grad=*/true);
    auto y = tenzor::vector_norm(xv, /*ord=*/0.0, /*dim=*/{}, /*keepdim=*/false);
    EXPECT_THROW(y.backward(), std::runtime_error);
}

// ============================================================================
// OpId dispatch-kernel regression (CPU-only): exercises the CPU
// cpu_kernel_registry.cpp entries for LinalgMatrixNorm / LinalgVectorNorm /
// LinalgVecdot directly via tenzor::dispatch(), which is how these OpIds are
// actually invoked (currently only from the JVP adapters in jvp_rules.cpp).
// The dispatch kernels used to reimplement this math a second time,
// divergently, instead of delegating to the already-tested linalg::* free
// functions above -- these tests target that dispatch-layer attribute
// plumbing specifically, not the free-function math (covered above).
// ============================================================================
class LinalgOpIdDispatchCpu : public ::testing::Test {
protected:
    void SetUp() override { tenzor::testing::EnsureInitialized(); }
};

TEST_F(LinalgOpIdDispatchCpu, MatrixNormOrd1IsMaxAbsColumnSumNotNuclear) {
    // Same 4x3 matrix as above: column abs-sums are 9, 7, 6 -> ord=1 should
    // give max()=9. The old dispatch kernel treated ord==1 as the nuclear
    // norm (sum of singular values, a different number entirely).
    auto A = mk_matrix_f32(Device::cpu());
    NewOpAttributes attrs;
    attrs.set(AttrKey::Order, 1.0);
    auto result = tenzor::dispatch(OpId::LinalgMatrixNorm, std::vector<Tensor>{A}, attrs);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_NEAR(scalar_f32(result[0]), 9.0f, 1e-5f);
}

TEST_F(LinalgOpIdDispatchCpu, MatrixNormOrd1PreservesBatchDim) {
    // Stack two copies of the 4x3 matrix into a [2,4,3] batch. The old
    // nuclear/spectral branch reduced with INT64_MIN (reduce-all), collapsing
    // the batch dim to a scalar instead of returning shape [2].
    std::vector<float> batched(kMatF32.size() * 2);
    std::copy(kMatF32.begin(), kMatF32.end(), batched.begin());
    std::copy(kMatF32.begin(), kMatF32.end(), batched.begin() + kMatF32.size());
    auto A = from_data(batched.data(), {2, kM, kN}, Device::cpu()).contiguous();

    NewOpAttributes attrs;
    attrs.set(AttrKey::Order, 1.0);
    auto result = tenzor::dispatch(OpId::LinalgMatrixNorm, std::vector<Tensor>{A}, attrs);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].numel(), 2) << "batch dim was collapsed instead of preserved";
    auto out_cpu = result[0].to(Device::cpu()).to(DType::Float32).contiguous();
    auto* out_data = out_cpu.data<float>();
    EXPECT_NEAR(out_data[0], 9.0f, 1e-5f);
    EXPECT_NEAR(out_data[1], 9.0f, 1e-5f);
}

TEST_F(LinalgOpIdDispatchCpu, VectorNormDimsAttrSupportsMultiAxisReduction) {
    // [2,3,4] tensor of ones; L2-norm reduced over dims {0,1} (6 elements per
    // trailing-axis slot) should give sqrt(6) per output element, shape [4].
    // The old dispatch kernel delegated to cpu::norm_kernel, which only
    // accepts a single dim and could not express this at all.
    auto A = ones({2, 3, 4}, DType::Float32, Device::cpu());
    NewOpAttributes attrs;
    attrs.set(AttrKey::P, 2.0);
    attrs.set(AttrKey::Dims, "0,1");
    attrs.set(AttrKey::Keepdim, false);
    auto result = tenzor::dispatch(OpId::LinalgVectorNorm, std::vector<Tensor>{A}, attrs);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].numel(), 4);
    auto out_cpu = result[0].to(Device::cpu()).to(DType::Float32).contiguous();
    auto* out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < 4; ++i) {
        EXPECT_NEAR(out_data[i], std::sqrt(6.0f), 1e-5f);
    }
}

TEST_F(LinalgOpIdDispatchCpu, VectorNormSingleDimAttrStillWorks) {
    // Regression: the only current real caller (the JVP adapter) sets a
    // single AttrKey::Dim, not AttrKey::Dims. Must keep working unchanged.
    auto A = ones({3, 4}, DType::Float32, Device::cpu());
    NewOpAttributes attrs;
    attrs.set(AttrKey::P, 2.0);
    attrs.set(AttrKey::Dim, static_cast<int64_t>(1));
    attrs.set(AttrKey::Keepdim, false);
    auto result = tenzor::dispatch(OpId::LinalgVectorNorm, std::vector<Tensor>{A}, attrs);
    ASSERT_EQ(result.size(), 1u);
    EXPECT_EQ(result[0].numel(), 3);
    auto out_cpu = result[0].to(Device::cpu()).to(DType::Float32).contiguous();
    auto* out_data = out_cpu.data<float>();
    for (int64_t i = 0; i < 3; ++i) {
        EXPECT_NEAR(out_data[i], 2.0f, 1e-5f);  // sqrt(4 ones^2) = 2
    }
}

TEST_F(LinalgOpIdDispatchCpu, VecdotConjugatesComplexFirstArgument) {
    // vecdot(a, b) = sum(conj(a) * b). Pick a = [i, 0], b = [i, 0]:
    // conj(a)*b = [conj(i)*i, 0] = [(-i)*i, 0] = [1, 0] -> sum = 1+0i.
    // The old dispatch kernel omitted the conjugation entirely and would
    // have computed sum(a*b) = i*i = -1+0i instead.
    auto a = zeros({2}, DType::Complex64, Device::cpu());
    auto b = zeros({2}, DType::Complex64, Device::cpu());
    a.data<std::complex<float>>()[0] = {0.0f, 1.0f};
    a.data<std::complex<float>>()[1] = {0.0f, 0.0f};
    b.data<std::complex<float>>()[0] = {0.0f, 1.0f};
    b.data<std::complex<float>>()[1] = {0.0f, 0.0f};

    NewOpAttributes attrs;
    attrs.set(AttrKey::Dim, static_cast<int64_t>(0));
    auto result = tenzor::dispatch(OpId::LinalgVecdot, std::vector<Tensor>{a, b}, attrs);
    ASSERT_EQ(result.size(), 1u);
    auto out_cpu = result[0].to(Device::cpu()).contiguous();
    auto value = out_cpu.data<std::complex<float>>()[0];
    EXPECT_NEAR(value.real(), 1.0f, 1e-5f);
    EXPECT_NEAR(value.imag(), 0.0f, 1e-5f);
}

// ============================================================================
// Fan every TEST_P above over all five backends (one per suite). BackendTest::
// SetUp skips a backend that is physically absent on the host.
// ============================================================================
INSTANTIATE_BACKEND_TESTS(LinalgNormOrds);
INSTANTIATE_BACKEND_TESTS(LinalgVectorNormNegInf);
INSTANTIATE_BACKEND_TESTS(LinalgVectorNormOne);
INSTANTIATE_BACKEND_TESTS(LinalgVectorNormOrd0);
INSTANTIATE_BACKEND_TESTS(LinalgVectorNormPosInf);
