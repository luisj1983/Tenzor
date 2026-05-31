// Audit Phase 3 Tasks 3.1+3.2+3.3 regression tests.
//
// 3.1  norm Float64 dim-path no longer round-trips through Float32.
// 3.2  max/min/argmax/argmin F16/BF16/F32 propagate NaN (PyTorch semantics).
// 3.3  Float64 sum benefits from Kahan compensation.
//
// Parameterized over all backends via BackendTest: each TEST_P creates its
// tensors on the fixture's `device`. Host data is filled on a CPU tensor and
// moved to `device`; results are read back via .cpu()/.item(). A (backend,
// dtype) cell the backend does not implement throws and FAILs — intentional.

#include <gtest/gtest.h>
#include <cmath>
#include <cstring>
#include <limits>

#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

class ReductionNumericalFixes : public ::tenzor::testing::BackendTest {};

// ── Helper: write a raw bit pattern into a Float16 element ──────────────────
static void set_f16_bits(tz::Tensor& t, int64_t idx, uint16_t bits) {
    auto* raw = reinterpret_cast<uint16_t*>(t.data_ptr());
    raw[idx] = bits;
}

// ── Task 3.1: norm Float64 precision ─────────────────────────────────────────

TEST_P(ReductionNumericalFixes, NormFloat64FullPrecision) {
    // Use values near 1.0 + i*1e-10: these are close enough that their
    // F32 representations are all equal to 1.0f, yet they differ in F64.
    // A round-trip through F32 collapses the sub-ULP differences and
    // produces a result that differs by ~3e-10 from the true F64 norm.
    auto t_host = tz::zeros({4}, tz::DType::Float64);
    double* p = t_host.data<double>();
    p[0] = 1.0;
    p[1] = 1.0 + 1e-10;
    p[2] = 1.0 + 2e-10;
    p[3] = 1.0 + 3e-10;

    double expected = std::sqrt(p[0]*p[0] + p[1]*p[1] + p[2]*p[2] + p[3]*p[3]);

    auto t = t_host.to(device);

    // Full-tensor norm (no dim) — must stay in Float64.
    auto n0 = tz::norm(t, 2.0f);
    EXPECT_NEAR(n0.cpu().item<double>(), expected, 1e-13)
        << "Full F64 norm lost precision (expected " << expected << ")";

    // Per-dim norm — the previously broken path (was round-tripping via F32).
    // F32 error here would be ~3e-10; we require accuracy to 1e-13.
    auto m_host = tz::zeros({1, 4}, tz::DType::Float64);
    double* mp = m_host.data<double>();
    for (int i = 0; i < 4; ++i) mp[i] = p[i];

    auto m = m_host.to(device);

    auto n1 = tz::norm(m, 2.0f, /*dim=*/1, /*keepdim=*/false);
    EXPECT_NEAR(n1.cpu().item<double>(), expected, 1e-13)
        << "Per-dim F64 norm lost precision (round-trip through F32?): "
        << "got " << n1.cpu().item<double>() << " expected " << expected;
}

// ── Task 3.2: NaN propagation ─────────────────────────────────────────────────

// Helper: read result as float regardless of output dtype (cast to Float32 first).
static float result_as_float(const tz::Tensor& t) {
    return t.cpu().to(tz::DType::Float32).item<float>();
}

// F16 max (global reduction)
TEST_P(ReductionNumericalFixes, MaxNanPropagationFloat16) {
    auto t_host = tz::full({16}, 1.0, tz::DType::Float16);
    // Inject a quiet-NaN at index 4; F16 quiet-NaN = 0x7E00.
    set_f16_bits(t_host, 4, 0x7E00u);
    auto t = t_host.to(device);
    auto m = tz::max(t);
    float val = result_as_float(m);
    EXPECT_TRUE(std::isnan(val))
        << "F16 max: expected NaN but got " << val;
}

// BF16 max (global reduction)
TEST_P(ReductionNumericalFixes, MaxNanPropagationBFloat16) {
    auto t_host = tz::full({16}, 1.0, tz::DType::BFloat16);
    // BF16 quiet-NaN = 0x7FC0 (or any 0x7F80..0x7FFF with mantissa != 0)
    set_f16_bits(t_host, 4, 0x7FC0u);
    auto t = t_host.to(device);
    auto m = tz::max(t);
    float val = result_as_float(m);
    EXPECT_TRUE(std::isnan(val))
        << "BF16 max: expected NaN but got " << val;
}

// F32 max (global reduction via SIMD)
TEST_P(ReductionNumericalFixes, MaxNanPropagationFloat32) {
    auto t_host = tz::ones({16}, tz::DType::Float32);
    t_host.data<float>()[4] = std::numeric_limits<float>::quiet_NaN();
    auto t = t_host.to(device);
    auto m = tz::max(t);
    EXPECT_TRUE(std::isnan(m.cpu().item<float>()))
        << "F32 max: expected NaN but got " << m.cpu().item<float>();
}

// F16 min (global reduction)
TEST_P(ReductionNumericalFixes, MinNanPropagationFloat16) {
    auto t_host = tz::full({16}, 1.0, tz::DType::Float16);
    set_f16_bits(t_host, 7, 0x7E00u);
    auto t = t_host.to(device);
    auto m = tz::min(t);
    float val = result_as_float(m);
    EXPECT_TRUE(std::isnan(val))
        << "F16 min: expected NaN but got " << val;
}

// BF16 min (global reduction)
TEST_P(ReductionNumericalFixes, MinNanPropagationBFloat16) {
    auto t_host = tz::full({16}, 1.0, tz::DType::BFloat16);
    set_f16_bits(t_host, 7, 0x7FC0u);
    auto t = t_host.to(device);
    auto m = tz::min(t);
    float val = result_as_float(m);
    EXPECT_TRUE(std::isnan(val))
        << "BF16 min: expected NaN but got " << val;
}

// F32 min (global reduction)
TEST_P(ReductionNumericalFixes, MinNanPropagationFloat32) {
    auto t_host = tz::ones({16}, tz::DType::Float32);
    t_host.data<float>()[4] = std::numeric_limits<float>::quiet_NaN();
    auto t = t_host.to(device);
    auto m = tz::min(t);
    EXPECT_TRUE(std::isnan(m.cpu().item<float>()))
        << "F32 min: expected NaN but got " << m.cpu().item<float>();
}

// ── Task 3.3: Float64 Kahan compensation ─────────────────────────────────────

TEST_P(ReductionNumericalFixes, SumFloat64KahanCompensation) {
    // 1,000,000 additions of 1e-9 in Float64.
    // Exact result = 1e-3.
    // Naive (uncompensated) sum drifts on the order of N * eps_double (~2e-10).
    // Kahan-compensated sum should be accurate to ~eps_double (~1e-15).
    const int64_t N = 1'000'000;
    auto t = tz::full({N}, 1e-9, tz::DType::Float64, device);
    auto s = tz::sum(t);
    double actual   = s.cpu().item<double>();
    double expected = 1e-3;
    EXPECT_NEAR(actual, expected, 1e-12)
        << "Float64 sum lost precision: actual=" << actual
        << " expected=" << expected
        << " — Kahan compensation not applied to Float64?";
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given reduction dtype throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(ReductionNumericalFixes);
