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
#include "tenzor/ops/advanced.hpp"
#include "tenzor/ops/op_id.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
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

// ── Norm p=0 / p=-inf / empty-dim CPU regression ─────────────────────────────
// p==0 previously fell through to the general Lp branch, where pow(x,0)==1
// for every element (including zero), so it returned the element count
// instead of the count of nonzero elements. p==-inf collapsed to the same
// branch as p==+inf (std::isinf() doesn't distinguish sign), returning
// max(|x|) instead of min(|x|). A per-dim reduction over a zero-size dim
// silently returned 0 instead of throwing, unlike every sibling CPU
// reduction (max/min/median/mode) and unlike Norm's own full-reduction
// empty-tensor path. CPU-only (not run cross-backend): the GPU backends'
// equivalent bugs are tracked as separate findings and fixed independently.

class NormSpecialPCpu : public ::testing::Test {
protected:
    void SetUp() override { tz::testing::EnsureInitialized(); }
};

TEST_F(NormSpecialPCpu, L0FullReductionCountsNonzero) {
    auto t = tz::zeros({5}, tz::DType::Float32);
    float* p = t.data<float>();
    p[0] = 0.0f; p[1] = 3.0f; p[2] = 0.0f; p[3] = -2.0f; p[4] = 0.0f;
    EXPECT_FLOAT_EQ(tz::norm(t, 0.0).item<float>(), 2.0f);
}

TEST_F(NormSpecialPCpu, L0PerDimCountsNonzero) {
    auto t = tz::zeros({2, 3}, tz::DType::Float32);
    float* p = t.data<float>();
    p[0] = 0.0f; p[1] = 3.0f; p[2] = 0.0f;   // row 0: 1 nonzero
    p[3] = -2.0f; p[4] = 0.0f; p[5] = 5.0f;  // row 1: 2 nonzero
    auto n = tz::norm(t, 0.0, /*dim=*/1, /*keepdim=*/false);
    const float* out = n.data<float>();
    EXPECT_FLOAT_EQ(out[0], 1.0f);
    EXPECT_FLOAT_EQ(out[1], 2.0f);
}

TEST_F(NormSpecialPCpu, NegativeInfinityFullReductionReturnsMinAbs) {
    auto t = tz::zeros({4}, tz::DType::Float32);
    float* p = t.data<float>();
    p[0] = 5.0f; p[1] = -2.0f; p[2] = 7.0f; p[3] = -3.0f;
    EXPECT_FLOAT_EQ(tz::norm(t, -std::numeric_limits<double>::infinity()).item<float>(), 2.0f);
    // p=+inf must be unaffected by the sign-check.
    EXPECT_FLOAT_EQ(tz::norm(t, std::numeric_limits<double>::infinity()).item<float>(), 7.0f);
}

TEST_F(NormSpecialPCpu, NegativeInfinityPerDimReturnsMinAbs) {
    auto t = tz::zeros({1, 4}, tz::DType::Float32);
    float* p = t.data<float>();
    p[0] = 5.0f; p[1] = -2.0f; p[2] = 7.0f; p[3] = -3.0f;
    auto n = tz::norm(t, -std::numeric_limits<double>::infinity(), /*dim=*/1, /*keepdim=*/false);
    EXPECT_FLOAT_EQ(n.item<float>(), 2.0f);
}

TEST_F(NormSpecialPCpu, Float64L0CountsNonzero) {
    auto t = tz::zeros({4}, tz::DType::Float64);
    double* p = t.data<double>();
    p[0] = 0.0; p[1] = 3.0; p[2] = 0.0; p[3] = -2.0;
    EXPECT_DOUBLE_EQ(tz::norm(t, 0.0).item<double>(), 2.0);
}

TEST_F(NormSpecialPCpu, Float64NegativeInfinityReturnsMinAbs) {
    auto t = tz::zeros({4}, tz::DType::Float64);
    double* p = t.data<double>();
    p[0] = 5.0; p[1] = -2.0; p[2] = 7.0; p[3] = -3.0;
    EXPECT_DOUBLE_EQ(tz::norm(t, -std::numeric_limits<double>::infinity()).item<double>(), 2.0);
}

TEST_F(NormSpecialPCpu, EmptyDimSliceThrowsInsteadOfSilentZero) {
    auto t = tz::zeros({3, 0}, tz::DType::Float32);
    EXPECT_THROW(tz::norm(t, 2.0, /*dim=*/1, /*keepdim=*/false), std::invalid_argument);
}

// ── Nanmedian even-count lower-median CPU regression ─────────────────────────
// nanmedian_impl indexed the sorted, NaN-filtered slice at n/2 (upper-middle
// for even n), contradicting its own "lower median (no interpolation)"
// comment and this file's own median_kernel (which correctly uses
// (dim_size-1)/2). nanmedian([1,2,3,4]) (no NaNs) must return 2 (the lower
// of the two middle values), not 3. CPU-only: the GPU backends delegate
// nanmedian to nanquantile(q=0.5), a separately tracked divergence.
class NanmedianCpu : public ::testing::Test {
protected:
    void SetUp() override { tz::testing::EnsureInitialized(); }
};

TEST_F(NanmedianCpu, EvenCountReturnsLowerMedian) {
    auto t = tz::zeros({4}, tz::DType::Float32);
    float* p = t.data<float>();
    p[0] = 1.0f; p[1] = 2.0f; p[2] = 3.0f; p[3] = 4.0f;
    auto m = tz::nanmedian(t, /*dim=*/0);
    EXPECT_FLOAT_EQ(m.item<float>(), 2.0f);
}

TEST_F(NanmedianCpu, EvenCountAfterNanFilteringReturnsLowerMedian) {
    // [1, NaN, 2, 3, 4] -> NaN-filtered survivors [1,2,3,4], even count 4.
    auto t = tz::zeros({5}, tz::DType::Float32);
    float* p = t.data<float>();
    p[0] = 1.0f; p[1] = std::numeric_limits<float>::quiet_NaN();
    p[2] = 2.0f; p[3] = 3.0f; p[4] = 4.0f;
    auto m = tz::nanmedian(t, /*dim=*/0);
    EXPECT_FLOAT_EQ(m.item<float>(), 2.0f);
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

// var/std are undefined (NaN) when the number of reduced elements N is <= the
// Bessel correction (PyTorch / CPU semantics). The CUDA backend used to return
// finite values (0 or a clamped m2) here, silently diverging from CPU and any
// parity test that exercises a single-element reduction or a length-1 axis —
// both reachable with the DEFAULT correction=1 (unbiased=true).
TEST_P(ReductionNumericalFixes, VarStdUndefinedReturnsNaN) {
    // Single-element tensor, full reduction (unbiased): N=1, correction=1.
    {
        auto t = tz::full({1}, 3.0f, tz::DType::Float32).to(device);
        EXPECT_TRUE(std::isnan(tz::var(t).cpu().item<float>()))
            << "var of a single element should be NaN on " << device.to_string();
        EXPECT_TRUE(std::isnan(tz::std(t).cpu().item<float>()))
            << "std of a single element should be NaN on " << device.to_string();
    }
    // Length-1 reduced axis (unbiased): dim_size=1, correction=1.
    {
        auto t = tz::randn({3, 1}, tz::DType::Float32).to(device);
        auto v = tz::var(t, /*dim=*/1).cpu();
        const float* vp = v.data<float>();
        for (int64_t i = 0; i < v.numel(); ++i) {
            EXPECT_TRUE(std::isnan(vp[i]))
                << "var over a length-1 axis should be NaN on " << device.to_string();
        }
    }
    // correction=0 (biased): single-element variance IS defined and equals 0.
    {
        auto t = tz::full({1}, 3.0f, tz::DType::Float32).to(device);
        EXPECT_FLOAT_EQ(tz::var(t, std::nullopt, false, /*unbiased=*/false).cpu().item<float>(), 0.0f)
            << "biased var of a single element should be 0 on " << device.to_string();
    }
}

// ── Median/Mode/Quantile zero-size-dim guard ─────────────────────────────────
// CUDA's median_kernel/mode_kernel/quantile_kernel had no guard for a
// zero-size reduced dimension: median's mid=(dim_size-1)/2 truncated to 0
// and read sorted_vals[0] from a 0-byte allocation; mode's find_mode_kernel
// unconditionally read sorted_vals[0] on a 0-size buffer; quantile's
// pos=q*(dim_size-1) went negative and hi=dim_size-1=-1, reading at a
// negative-derived offset -- all genuine out-of-bounds device-memory reads.
// CPU throws std::invalid_argument for median/mode and emits NaN for
// quantile; CUDA must now match exactly.

TEST_P(ReductionNumericalFixes, MedianZeroSizeDimThrowsCleanly) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::median(t, /*dim=*/1), std::invalid_argument)
        << "median over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, ModeZeroSizeDimThrowsCleanly) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::mode(t, /*dim=*/1), std::invalid_argument)
        << "mode over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, QuantileZeroSizeDimReturnsNaN) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto q = tz::quantile(t, 0.5, /*dim=*/1);
    auto q_cpu = q.cpu();
    const float* p = q_cpu.data<float>();
    ASSERT_EQ(q_cpu.numel(), 15);
    for (int64_t i = 0; i < q_cpu.numel(); ++i) {
        EXPECT_TRUE(std::isnan(p[i]))
            << "quantile of a zero-size dim should be NaN at index " << i
            << " on " << device.to_string();
    }
}

// ── Dim-specific empty-reduction identity/throw guard ───────────────────────
// Reducing a size-0 dimension while other dims are non-empty (output_size>0)
// must either fill the mathematically correct identity value (prod=1,
// any=false, all=true, count_nonzero=0, sum=0, logsumexp=-inf) or throw
// cleanly (norm/max/min/argmax/argmin), never leave freshly-allocated output
// as uninitialized device memory. ROCm's dim-reduction launchers previously
// bailed out early on `dim_size == 0` without writing anything.

TEST_P(ReductionNumericalFixes, ProdZeroSizeDimFillsIdentity) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto r = tz::prod(t, /*dim=*/1).cpu();
    ASSERT_EQ(r.numel(), 15);
    const float* p = r.data<float>();
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_EQ(p[i], 1.0f) << "prod over a zero-size dim should be 1 at index " << i
                               << " on " << device.to_string();
    }
}

TEST_P(ReductionNumericalFixes, AnyZeroSizeDimFillsIdentity) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto r = tz::any(t, /*dim=*/1).cpu();
    ASSERT_EQ(r.numel(), 15);
    const bool* p = r.data<bool>();
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_FALSE(p[i]) << "any over a zero-size dim should be false at index " << i
                            << " on " << device.to_string();
    }
}

TEST_P(ReductionNumericalFixes, AllZeroSizeDimFillsIdentity) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto r = tz::all(t, /*dim=*/1).cpu();
    ASSERT_EQ(r.numel(), 15);
    const bool* p = r.data<bool>();
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_TRUE(p[i]) << "all over a zero-size dim should be true at index " << i
                           << " on " << device.to_string();
    }
}

TEST_P(ReductionNumericalFixes, CountNonzeroZeroSizeDimFillsIdentity) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto r = tz::count_nonzero(t, /*dim=*/1).cpu();
    ASSERT_EQ(r.numel(), 15);
    const int64_t* p = r.data<int64_t>();
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_EQ(p[i], 0) << "count_nonzero over a zero-size dim should be 0 at index " << i
                            << " on " << device.to_string();
    }
}

TEST_P(ReductionNumericalFixes, SumZeroSizeDimFillsIdentity) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto r = tz::sum(t, /*dim=*/1).cpu();
    ASSERT_EQ(r.numel(), 15);
    const float* p = r.data<float>();
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_EQ(p[i], 0.0f) << "sum over a zero-size dim should be 0 at index " << i
                               << " on " << device.to_string();
    }
}

TEST_P(ReductionNumericalFixes, LogSumExpZeroSizeDimFillsNegativeInfinity) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    auto r = tz::logsumexp(t, /*dim=*/1).cpu();
    ASSERT_EQ(r.numel(), 15);
    const float* p = r.data<float>();
    for (int64_t i = 0; i < r.numel(); ++i) {
        EXPECT_TRUE(std::isinf(p[i]) && p[i] < 0.0f)
            << "logsumexp over a zero-size dim should be -inf at index " << i
            << " on " << device.to_string();
    }
}

// ── Full-reduction (dim==INT64_MIN sentinel) empty-tensor regression ───────
// tenzor::logsumexp()'s public API always normalizes `dim` before dispatch
// (see normalize_reduce_dim() in src/ops/reduction.cpp), and the op layer
// itself special-cases numel()==0 before ever reaching the backend kernel.
// So the INT64_MIN "reduce every element, no dim specified" sentinel never
// reaches the registered kernel through tz::logsumexp() -- only a raw
// dispatch() call with the Dim attribute left unset does (the kernel reads
// it via attrs.get_int(AttrKey::Dim, INT64_MIN), defaulting to INT64_MIN).
// That registered-kernel contract must still hold for an empty tensor:
// logsumexp of the empty set is -inf. CUDA/ROCm previously wrote the finite
// sentinel -FLT_MAX/-DBL_MAX instead of a real IEEE -infinity; OneAPI had no
// INT64_MIN branch at all and threw "Dimension ... out of range".
TEST_P(ReductionNumericalFixes, LogSumExpFullReductionEmptyTensorFillsNegativeInfinityF32) {
    auto t = tz::zeros({0}, tz::DType::Float32).to(device);
    std::vector<tz::Tensor> inputs = {t};
    auto r = tz::dispatch(tz::OpId::LogSumExp, inputs)[0].cpu();
    ASSERT_EQ(r.numel(), 1);
    float v = r.item<float>();
    EXPECT_TRUE(std::isinf(v) && v < 0.0f)
        << "logsumexp full reduction (dim==INT64_MIN) of an empty Float32 tensor "
        << "should be real -inf, got " << v << " on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, LogSumExpFullReductionEmptyTensorFillsNegativeInfinityF64) {
    auto t = tz::zeros({0}, tz::DType::Float64).to(device);
    std::vector<tz::Tensor> inputs = {t};
    auto r = tz::dispatch(tz::OpId::LogSumExp, inputs)[0].cpu();
    ASSERT_EQ(r.numel(), 1);
    double v = r.item<double>();
    EXPECT_TRUE(std::isinf(v) && v < 0.0)
        << "logsumexp full reduction (dim==INT64_MIN) of an empty Float64 tensor "
        << "should be real -inf, got " << v << " on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, NormZeroSizeDimThrows) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::norm(t, /*p=*/2.0, /*dim=*/1), std::invalid_argument)
        << "norm over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, MaxZeroSizeDimThrows) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::max(t, /*dim=*/1), std::invalid_argument)
        << "max over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, MinZeroSizeDimThrows) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::min(t, /*dim=*/1), std::invalid_argument)
        << "min over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, ArgmaxZeroSizeDimThrows) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::argmax(t, /*dim=*/1), std::invalid_argument)
        << "argmax over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, ArgminZeroSizeDimThrows) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::argmin(t, /*dim=*/1), std::invalid_argument)
        << "argmin over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, AminmaxZeroSizeDimThrows) {
    auto t = tz::zeros({3, 0, 5}, tz::DType::Float32).to(device);
    EXPECT_THROW(tz::aminmax(t, /*dim=*/1), std::invalid_argument)
        << "aminmax over a zero-size dim should throw cleanly on " << device.to_string();
}

TEST_P(ReductionNumericalFixes, ProdFullReductionOfEmptyTensorReturnsOne) {
    auto t = tz::zeros({0}, tz::DType::Float32).to(device);
    auto r = tz::prod(t).cpu();
    ASSERT_EQ(r.numel(), 1);
    EXPECT_EQ(r.item<float>(), 1.0f)
        << "prod() of an empty tensor (no dim) should be 1 on " << device.to_string();
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given reduction dtype throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(ReductionNumericalFixes);
