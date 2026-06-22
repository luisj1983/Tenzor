// =============================================================================
// tests/backends/test_release_audit_cpu.cpp
//
// Consolidated CPU-backend correctness regressions from the release audit.
//   B2  isin must compare at native bit-width (no Float32 truncation).
//   B3  index_select must honour non-contiguous (transposed) inputs.
//   C2  vector norm (p>=2) must not overflow for large-magnitude inputs.
// =============================================================================

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/autograd/gradcheck.hpp>
#include <tenzor/nn/layers/rnn.hpp>

#include <cmath>
#include <limits>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif

using namespace tenzor;

// Per-axis unfold kernel (release-prep B11) — forward-declared for direct test.
// LayerNorm Float32 SIMD kernels (release-prep C6) — forward-declared so the
// test exercises the float-input SIMD reduction path directly.
namespace tenzor { namespace cpu {
auto unfold_kernel(const Tensor& input, int64_t kh, int64_t kw,
                   int64_t sh, int64_t sw, int64_t ph, int64_t pw,
                   int64_t dh, int64_t dw) -> Tensor;
auto layer_norm_kernel(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                       const Tensor& weight, const Tensor& bias, float eps) -> Tensor;
auto layer_norm_kernel_with_stats(const Tensor& input, const std::vector<int64_t>& normalized_shape,
                                  const Tensor& weight, const Tensor& bias, float eps)
    -> std::tuple<Tensor, Tensor, Tensor>;
}}  // namespace tenzor::cpu

namespace {

class ReleaseAuditCpuEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static auto* const env =
    ::testing::AddGlobalTestEnvironment(new ReleaseAuditCpuEnv);

Tensor int64_vec(std::vector<int64_t> v) {
    Tensor t({static_cast<int64_t>(v.size())}, DType::Int64, Device::cpu());
    auto* p = t.data<int64_t>();
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
Tensor f32_vec(std::vector<float> v) {
    Tensor t({static_cast<int64_t>(v.size())}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}
Tensor f64_vec(std::vector<double> v) {
    Tensor t({static_cast<int64_t>(v.size())}, DType::Float64, Device::cpu());
    auto* p = t.data<double>();
    for (size_t i = 0; i < v.size(); ++i) p[i] = v[i];
    return t;
}

}  // namespace

// B2: 2^24 and 2^24+1 are distinct int64 but collide as Float32 (both -> 2^24).
TEST(ReleaseAuditCpu, IsinInt64ExactBeyondFloat32) {
    auto elements = int64_vec({16777217, 16777216, 5});
    auto test = int64_vec({16777216});  // only 2^24 is a member
    auto out = tenzor::isin(elements, test);
    const bool* o = out.data<bool>();
    EXPECT_FALSE(o[0]) << "16777217 must NOT be in {16777216}";
    EXPECT_TRUE(o[1]);
    EXPECT_FALSE(o[2]);
}

// B3: index_select along dim 0 of a transposed (non-contiguous) view.
TEST(ReleaseAuditCpu, IndexSelectNonContiguous) {
    // 2x3 row-major [[0,1,2],[3,4,5]]; transpose -> 3x2 [[0,3],[1,4],[2,5]].
    Tensor m({2, 3}, DType::Float32, Device::cpu());
    auto* p = m.data<float>();
    for (int i = 0; i < 6; ++i) p[i] = static_cast<float>(i);
    auto mt = m.transpose(0, 1);            // [3,2], non-contiguous view
    auto idx = int64_vec({0, 2});           // rows 0 and 2 -> [[0,3],[2,5]]
    auto sel = tenzor::index_select(mt, 0, idx).contiguous();
    ASSERT_EQ(sel.numel(), 4);
    const float* s = sel.data<float>();
    EXPECT_FLOAT_EQ(s[0], 0.0f);
    EXPECT_FLOAT_EQ(s[1], 3.0f);
    EXPECT_FLOAT_EQ(s[2], 2.0f);
    EXPECT_FLOAT_EQ(s[3], 5.0f);
}

// C2: L2 norm of large-magnitude Float32 must stay finite (max-scaling), not
// overflow to +inf from squaring (1e20^2 = 1e40 > FLT_MAX).
TEST(ReleaseAuditCpu, NormL2NoOverflowFloat32) {
    auto v = f32_vec({1e20f, 1e20f, 1e20f, 1e20f});  // true L2 = 2e20
    auto n = tenzor::norm(v, 2.0f);
    float val = n.data<float>()[0];
    EXPECT_TRUE(std::isfinite(val)) << "L2 overflowed to " << val;
    EXPECT_NEAR(val, 2e20f, 2e20f * 1e-4f);
}

// C2 (Lp, 2<p<=10): same overflow guard for general p.
TEST(ReleaseAuditCpu, NormL3NoOverflowFloat32) {
    auto v = f32_vec({1e15f, 1e15f, 1e15f});  // (3 * 1e45)^(1/3) ~ 1.44e15
    auto n = tenzor::norm(v, 3.0f);
    float val = n.data<float>()[0];
    EXPECT_TRUE(std::isfinite(val)) << "L3 overflowed to " << val;
    EXPECT_NEAR(val, std::cbrt(3.0f) * 1e15f, 1e15f * 1e-3f);
}

// C2 dim-wise correctness guard: the max-scaled rewrite must still match the
// plain L2 along a dimension. [[3,4],[6,8]] -> L2 over dim=1 = [5, 10].
TEST(ReleaseAuditCpu, NormDimWiseL2Correct) {
    Tensor m({2, 2}, DType::Float32, Device::cpu());
    auto* p = m.data<float>();
    p[0] = 3; p[1] = 4; p[2] = 6; p[3] = 8;
    auto n = tenzor::norm(m, 2.0f, /*dim=*/1, /*keepdim=*/false).contiguous();
    ASSERT_EQ(n.numel(), 2);
    const float* o = n.data<float>();
    EXPECT_FLOAT_EQ(o[0], 5.0f);
    EXPECT_FLOAT_EQ(o[1], 10.0f);
}

// E: clamp must propagate NaN consistently (SIMD path used to return the bound
// for NaN inputs; scalar/PyTorch propagate NaN). 16 elements hit the AVX2 path.
TEST(ReleaseAuditCpu, ClampPropagatesNaN) {
    const float nan = std::numeric_limits<float>::quiet_NaN();
    auto v = f32_vec({0, 1, 2, 3, nan, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15});
    auto r = tenzor::clamp(v, 0.0, 10.0).contiguous();
    const float* o = r.data<float>();
    EXPECT_TRUE(std::isnan(o[4])) << "clamp must propagate NaN, got " << o[4];
    EXPECT_FLOAT_EQ(o[0], 0.0f);
    EXPECT_FLOAT_EQ(o[15], 10.0f);
}

// E: logit eps contract. Default eps (-1.0 / "None") must NOT silently clamp:
// inputs outside (0,1) produce NaN so the caller sees the invalid region.
// eps>0 clamps to [eps, 1-eps] (finite everywhere). Inside (0,1) both match
// the closed form log(x/(1-x)). The hardcoded-eps kernel used to clamp
// unconditionally, diverging from this canonical (composed-op) behaviour.
TEST(ReleaseAuditCpu, LogitEpsContract) {
    auto x = f64_vec({-0.5, 0.25, 0.5, 0.75, 1.5});

    // Default eps (None): outside-domain entries must be NaN, not clamped.
    auto def = tenzor::logit(x, -1.0).contiguous();
    const double* d = def.data<double>();
    EXPECT_TRUE(std::isnan(d[0])) << "logit(-0.5) must be NaN with eps=None, got " << d[0];
    EXPECT_TRUE(std::isnan(d[4])) << "logit(1.5) must be NaN with eps=None, got " << d[4];
    // Interior values match the closed form.
    EXPECT_NEAR(d[1], std::log(0.25 / 0.75), 1e-12);
    EXPECT_NEAR(d[2], 0.0, 1e-12);
    EXPECT_NEAR(d[3], std::log(0.75 / 0.25), 1e-12);

    // eps>0: clamps to [eps, 1-eps] => every entry is finite.
    const double eps = 1e-6;
    auto cl = tenzor::logit(x, eps).contiguous();
    const double* c = cl.data<double>();
    for (int i = 0; i < 5; ++i)
        EXPECT_TRUE(std::isfinite(c[i])) << "logit with eps>0 must be finite at i=" << i;
    EXPECT_NEAR(c[0], std::log(eps / (1.0 - eps)), 1e-9);
    EXPECT_NEAR(c[4], std::log((1.0 - eps) / eps), 1e-9);
}

// C6: Float32 LayerNorm must accumulate the mean/variance reductions in
// double (its comment promises this and the Float64/Float16 scalar paths
// deliver it). The old SIMD path summed in float registers and only cast
// the reduced scalar to double, so for a large-mean / tiny-variance row the
// accumulated rounding error corrupts the mean by O(1) -- which dominates
// the variance and scrambles the normalized output's sign pattern.
//
// Input: x_i = 1e7 + (-1)^i  (1e7 +/- 1 is exact in Float32; ULP = 1).
// True mean = 1e7, var = 1, so y_i = +/-1. With float accumulation the mean
// drifts by ~O(1), giving var ~ 1+delta^2 and y_i ~ (s_i - delta)/sqrt(1+delta^2)
// -- the even entries (true +1) collapse toward <= 0.
static void check_layernorm_double_accum(bool with_stats) {
    constexpr int64_t N = 32768;
    Tensor x({1, N}, DType::Float32, Device::cpu());
    Tensor w({N}, DType::Float32, Device::cpu());
    Tensor b({N}, DType::Float32, Device::cpu());
    float* xp = x.data<float>();
    float* wp = w.data<float>();
    float* bp = b.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        xp[i] = 1.0e7f + ((i % 2 == 0) ? 1.0f : -1.0f);
        wp[i] = 1.0f;
        bp[i] = 0.0f;
    }
    Tensor out = with_stats
        ? std::get<0>(tenzor::cpu::layer_norm_kernel_with_stats(x, {N}, w, b, 1e-5f))
        : tenzor::cpu::layer_norm_kernel(x, {N}, w, b, 1e-5f);
    out = out.contiguous();
    const float* o = out.data<float>();
    // Double-accumulated reference: y_even = +1, y_odd = -1 (eps shrinks ~1e-5).
    EXPECT_NEAR(o[0], 1.0f, 0.05f)
        << (with_stats ? "with_stats" : "no_stats")
        << ": even entry collapsed -> mean accumulated in float, not double";
    EXPECT_NEAR(o[1], -1.0f, 0.05f);
    EXPECT_NEAR(o[2], 1.0f, 0.05f);
    EXPECT_NEAR(o[3], -1.0f, 0.05f);
}

TEST(ReleaseAuditCpu, LayerNormFloat32DoubleAccumulation) {
    check_layernorm_double_accum(/*with_stats=*/false);
    check_layernorm_double_accum(/*with_stats=*/true);
}

// C4: polygamma must be correct for negative (non-integer) arguments.
// digamma (n=0) uses the reflection formula; n>=1 uses the recurrence
// psi^(n)(x) = psi^(n)(x+1) + (-1)^(n+1) n!/x^(n+1), which is valid for all
// x that are not non-positive integers (it shifts x up past zero into the
// asymptotic region). Closed-form references:
//   psi(-0.5)    = 2 - gamma - 2 ln2          = 0.03648997397857652
//   psi_1(-0.5)  = pi^2 - psi_1(1.5) = pi^2 - (pi^2/2 - 4) = pi^2/2 + 4
//   psi_2(-0.5)  : reflection psi_2(x) - psi_2(1-x) = -d^2/dx^2[pi cot(pi x)]
TEST(ReleaseAuditCpu, PolygammaNegativeArguments) {
    const double gamma = 0.5772156649015328606;
    const double pi = M_PI;

    auto x = f64_vec({-0.5, -1.5, -2.5});

    // digamma (n=0)
    auto d0 = tenzor::polygamma(0, x).contiguous();
    const double* p0 = d0.data<double>();
    // psi(-0.5) = 2 - gamma - 2 ln2
    EXPECT_NEAR(p0[0], 2.0 - gamma - 2.0 * std::log(2.0), 1e-9);
    // psi(-1.5) = psi(-0.5) + 1/(-1.5) + 1/(-0.5)  [psi(x+1)=psi(x)+1/x downward]
    // => psi(-1.5) = psi(-0.5) - 1/(-1.5) ... use upward recurrence:
    // psi(-0.5) = psi(-1.5) + 1/(-1.5)  => psi(-1.5) = psi(-0.5) - 1/(-1.5)
    EXPECT_NEAR(p0[1], (2.0 - gamma - 2.0 * std::log(2.0)) - 1.0 / (-1.5), 1e-9);

    // trigamma (n=1): psi_1(-0.5) = pi^2/2 + 4
    auto d1 = tenzor::polygamma(1, x).contiguous();
    const double* p1 = d1.data<double>();
    EXPECT_NEAR(p1[0], pi * pi / 2.0 + 4.0, 1e-7);
    // psi_1(-1.5) = psi_1(-0.5) - 1/(-1.5)^2  [psi_1(x)=psi_1(x+1)+1/x^2 upward:
    //   psi_1(-1.5) = psi_1(-0.5) + 1/(-1.5)^2]
    EXPECT_NEAR(p1[1], (pi * pi / 2.0 + 4.0) + 1.0 / (1.5 * 1.5), 1e-7);

    // tetragamma (n=2): verify against the recurrence relative to psi_2(0.5).
    // psi_2(0.5) = -14 * zeta(3) = -16.8288... ; psi_2(-0.5) via reflection.
    // Use the recurrence consistency: psi_2(-0.5) = psi_2(0.5) + 2*2!/(-0.5)^3? No --
    // upward: psi_2(-0.5) = psi_2(0.5) - (-1)^2 * 2!/(-0.5)^3 = psi_2(0.5) + 2/0.125
    auto d2 = tenzor::polygamma(2, x).contiguous();
    const double* p2 = d2.data<double>();
    const double psi2_half = -14.0 * 1.2020569031595942854;  // -14 zeta(3)
    EXPECT_NEAR(p2[0], psi2_half + 2.0 / 0.125, 1e-5);
}

// B5: GRUCell backward w.r.t. the previous hidden state must be exact,
// including the reset-gate -> n-gate -> h_prev path (audit flagged it as
// possibly under-counted). f64 gradcheck is decisive.
TEST(ReleaseAuditCpu, GruCellBackwardGradcheckHx) {
    tenzor::manual_seed(7);
    nn::GRUCell gru(3, 4);          // input_size=3, hidden=4
    gru.to(DType::Float64);

    Tensor in_t({2, 3}, DType::Float64, Device::cpu());
    auto* ip = in_t.data<double>();
    for (int i = 0; i < 6; ++i) ip[i] = 0.1 * (i + 1) - 0.3;
    Variable input(in_t, false);

    Tensor hx_t({2, 4}, DType::Float64, Device::cpu());
    auto* hp = hx_t.data<double>();
    for (int i = 0; i < 8; ++i) hp[i] = 0.05 * (i + 1) - 0.2;

    auto f = [&](const Variable& hx) { return gru.forward(input, hx); };
    Variable hx(hx_t, true);
    EXPECT_TRUE(tenzor::gradcheck(f, hx, 1e-6, 1e-4, 1e-3))
        << "GRUCell backward w.r.t. hx failed gradcheck";
}

// B11: CPU unfold must handle ASYMMETRIC kernel/stride/padding/dilation
// (previously threw). 3x3 input 0..8, kernel (kh=2,kw=1), stride=1, pad=0.
TEST(ReleaseAuditCpu, UnfoldAsymmetricKernel) {
    Tensor in({1, 1, 3, 3}, DType::Float32, Device::cpu());
    auto* p = in.data<float>();
    for (int i = 0; i < 9; ++i) p[i] = static_cast<float>(i);
    auto out = tenzor::cpu::unfold_kernel(in, /*kh=*/2, /*kw=*/1,
                                          /*sh=*/1, /*sw=*/1, /*ph=*/0, /*pw=*/0,
                                          /*dh=*/1, /*dw=*/1);
    // Expect shape (1, C*kh*kw=2, L=H_out*W_out=2*3=6).
    ASSERT_EQ(out.numel(), 12);
    auto oc = out.contiguous();
    const float* o = oc.data<float>();
    const float expected[12] = {0,1,2,3,4,5,  3,4,5,6,7,8};
    for (int i = 0; i < 12; ++i)
        EXPECT_FLOAT_EQ(o[i], expected[i]) << "i=" << i;
}

// E/B: abs() must cover Int8/Int16/Int64/BFloat16 (previously only Int32
// among integers; others threw).
TEST(ReleaseAuditCpu, AbsIntegerDtypeCoverage) {
    auto a = int64_vec({-5, 3, -7, 0});
    auto r = tenzor::abs(a);
    ASSERT_EQ(r.dtype(), DType::Int64);
    const int64_t* o = r.data<int64_t>();
    EXPECT_EQ(o[0], 5);
    EXPECT_EQ(o[1], 3);
    EXPECT_EQ(o[2], 7);
    EXPECT_EQ(o[3], 0);
}

// C1: non-MKL Float32 transcendentals must use exact libm (the SIMD
// polynomial gave wrong results for these edge cases).
TEST(ReleaseAuditCpu, TranscendentalExactEdgeCases) {
    const float inf = std::numeric_limits<float>::infinity();
    const float nan = std::numeric_limits<float>::quiet_NaN();
    // exp(NaN) -> NaN (polynomial returned a finite value).
    EXPECT_TRUE(std::isnan(tenzor::exp(f32_vec({nan})).data<float>()[0]));
    // log(+inf) -> +inf (polynomial returned ~88.7 garbage).
    {
        float r = tenzor::log(f32_vec({inf})).data<float>()[0];
        EXPECT_TRUE(std::isinf(r) && r > 0.0f);
    }
    // sin(1e6) must match libm (polynomial f32 range reduction lost all sig).
    EXPECT_NEAR(tenzor::sin(f32_vec({1e6f})).data<float>()[0], std::sin(1e6f), 1e-3f);
    // pow(-2, 3) -> -8 (polynomial clamped negative base to ~0).
    EXPECT_NEAR(tenzor::pow(f32_vec({-2.0f}), 3.0).data<float>()[0], -8.0f, 1e-3f);
}

// C4: Hurwitz/Riemann zeta must include the Euler-Maclaurin Bernoulli
// correction terms. ζ(2,1) = π²/6; the pre-fix truncation erred ~7.6e-5.
TEST(ReleaseAuditCpu, ZetaEulerMaclaurinAccuracy) {
    auto s = f64_vec({2.0});
    auto q = f64_vec({1.0});
    auto z = tenzor::zeta(s, q);
    double val = z.data<double>()[0];
    const double pi2_6 = 3.14159265358979323846 * 3.14159265358979323846 / 6.0;
    EXPECT_NEAR(val, pi2_6, 1e-6) << "zeta(2,1) = " << val << " vs " << pi2_6;
}

// B1: samplers must be reproducible regardless of OMP thread count (Philox
// keyed by global index). Previously each kernel seeded per-OMP-thread, so the
// output depended on OMP_NUM_THREADS.
TEST(ReleaseAuditCpu, RandintReproducibleAcrossThreadCounts) {
#ifdef _OPENMP
    const int64_t N = 8192;
    const int saved = omp_get_max_threads();

    tenzor::manual_seed(2024);
    omp_set_num_threads(1);
    auto a = tenzor::randint(0, 1000, {N}, DType::Int64, Device::cpu());

    tenzor::manual_seed(2024);
    omp_set_num_threads(8);
    auto b = tenzor::randint(0, 1000, {N}, DType::Int64, Device::cpu());

    omp_set_num_threads(saved);

    const int64_t* pa = a.data<int64_t>();
    const int64_t* pb = b.data<int64_t>();
    for (int64_t i = 0; i < N; ++i) ASSERT_EQ(pa[i], pb[i]) << "mismatch at i=" << i;
#else
    GTEST_SKIP() << "OpenMP not enabled in this translation unit";
#endif
}

// Stride family: unary CPU kernels read data<T>() (storage+offset, strides NOT
// applied), so a non-contiguous view must be materialized first. Each op on a
// transposed view must equal the op on an explicit contiguous copy of it.
// Pre-fix, the non-contiguous calls read scrambled storage and diverge.
TEST(ReleaseAuditCpu, UnaryKernelsHonorNonContiguous) {
    Tensor base({3, 4}, DType::Float32, Device::cpu());
    float* bp = base.data<float>();
    for (int i = 0; i < 12; ++i) bp[i] = 0.25f * static_cast<float>(i + 1);  // >0 for log
    Tensor view = base.transpose(0, 1);   // [4,3], non-contiguous
    ASSERT_FALSE(view.is_contiguous());
    Tensor cont = view.contiguous();      // identical logical values, contiguous

    auto check = [&](const char* name, Tensor a, Tensor b) {
        a = a.contiguous();
        b = b.contiguous();
        ASSERT_EQ(a.numel(), b.numel()) << name;
        const float* pa = a.data<float>();
        const float* pb = b.data<float>();
        for (int64_t i = 0; i < a.numel(); ++i)
            EXPECT_FLOAT_EQ(pa[i], pb[i]) << name << " mismatch at " << i;
    };
    check("clamp", tenzor::clamp(view, 0.5, 2.0), tenzor::clamp(cont, 0.5, 2.0));
    check("log",   tenzor::log(view),  tenzor::log(cont));
    check("exp",   tenzor::exp(view),  tenzor::exp(cont));
    check("sin",   tenzor::sin(view),  tenzor::sin(cont));
    check("cos",   tenzor::cos(view),  tenzor::cos(cont));
    check("sign",  tenzor::sign(view), tenzor::sign(cont));
}

// unfold (im2col) indexes input with a contiguous (n*C+c)*H*W formula; a
// non-contiguous NCHW view must be materialized. Calls the kernel directly.
TEST(ReleaseAuditCpu, UnfoldKernelHonorsNonContiguous) {
    Tensor base({1, 2, 3, 3}, DType::Float32, Device::cpu());  // N,C,H,W
    float* bp = base.data<float>();
    for (int i = 0; i < 18; ++i) bp[i] = static_cast<float>(i + 1);
    Tensor view = base.transpose(2, 3);   // [1,2,3,3] non-contiguous (H<->W)
    ASSERT_FALSE(view.is_contiguous());
    Tensor cont = view.contiguous();

    auto uv = tenzor::cpu::unfold_kernel(view, 2, 2, 1, 1, 0, 0, 1, 1).contiguous();
    auto uc = tenzor::cpu::unfold_kernel(cont, 2, 2, 1, 1, 0, 0, 1, 1).contiguous();
    ASSERT_EQ(uv.numel(), uc.numel());
    const float* pv = uv.data<float>();
    const float* pc = uc.data<float>();
    for (int64_t i = 0; i < uv.numel(); ++i)
        EXPECT_FLOAT_EQ(pv[i], pc[i]) << "unfold mismatch at " << i;
}
