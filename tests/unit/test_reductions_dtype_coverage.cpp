#include <gtest/gtest.h>
#include <complex>
#include <cmath>
#include <algorithm>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/advanced.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

// Audit Phase 2 Tasks 2.6+2.7+2.8:
// 2.6: mean must work for Complex64/Complex128
// 2.7: sum/mean/var/argmax/argmin/prod must work for integer and Bool dtypes
// 2.8: scatter_reduce must work for Float16/BFloat16
//
// Parameterized over all backends via BackendTest: each TEST_P creates its
// tensors on the fixture's `device`. A (backend, dtype) cell the backend does
// not implement will throw and FAIL the test — intentional, to surface the
// real coverage gap rather than hide it. Backends physically absent on the
// host are skipped by BackendTest::SetUp (availability, not capability).
class ReductionsDtypeCoverage : public ::tenzor::testing::BackendTest {};

namespace {

// Create a tensor of `shape` with dtype `dt` filled with `val` on `device`.
tz::Tensor make_filled(std::vector<int64_t> shape, tz::DType dt, double val,
                       tz::Device device) {
    auto t = tz::empty(shape, dt, device);
    t.fill_(val);
    return t;
}

}  // namespace

// ── Task 2.6: mean Complex64/Complex128 ──────────────────────────────────────

TEST_P(ReductionsDtypeCoverage, MeanComplex64) {
    // mean([1+2i, 3+4i, 5+6i]) = (9+12i)/3 = 3+4i
    auto host = tz::empty({3}, tz::DType::Complex64);
    auto* p = host.data<std::complex<float>>();
    p[0] = {1.0f, 2.0f};
    p[1] = {3.0f, 4.0f};
    p[2] = {5.0f, 6.0f};
    auto t = host.to(device);
    auto m = tz::mean(t);
    auto v = m.cpu().item<std::complex<float>>();
    EXPECT_NEAR(v.real(), 3.0f, 1e-5f);
    EXPECT_NEAR(v.imag(), 4.0f, 1e-5f);
}

TEST_P(ReductionsDtypeCoverage, MeanComplex128) {
    auto host = tz::empty({3}, tz::DType::Complex128);
    auto* p = host.data<std::complex<double>>();
    p[0] = {1.0, 2.0}; p[1] = {3.0, 4.0}; p[2] = {5.0, 6.0};
    auto t = host.to(device);
    auto m = tz::mean(t);
    auto v = m.cpu().item<std::complex<double>>();
    EXPECT_NEAR(v.real(), 3.0, 1e-12);
    EXPECT_NEAR(v.imag(), 4.0, 1e-12);
}

// ── Task 2.7: sum on integer/Bool dtypes ─────────────────────────────────────

TEST_P(ReductionsDtypeCoverage, SumInt8) {
    // 16 elements each = 3, sum = 48; output dtype Int64 (PyTorch convention)
    auto t = make_filled({16}, tz::DType::Int8, 3.0, device);
    auto s = tz::sum(t);
    EXPECT_EQ(s.cpu().item<int64_t>(), 48);
}

TEST_P(ReductionsDtypeCoverage, SumInt16) {
    auto t = make_filled({16}, tz::DType::Int16, 5.0, device);
    auto s = tz::sum(t);
    EXPECT_EQ(s.cpu().item<int64_t>(), 80);
}

TEST_P(ReductionsDtypeCoverage, SumUInt8) {
    auto t = make_filled({16}, tz::DType::UInt8, 7.0, device);
    auto s = tz::sum(t);
    EXPECT_EQ(s.cpu().item<int64_t>(), 112);
}

TEST_P(ReductionsDtypeCoverage, SumUInt16) {
    auto t = make_filled({16}, tz::DType::UInt16, 100.0, device);
    auto s = tz::sum(t);
    EXPECT_EQ(s.cpu().item<int64_t>(), 1600);
}

TEST_P(ReductionsDtypeCoverage, SumBool) {
    // ones({16}, Bool) — all true, sum = 16 in Int64
    auto t = tz::ones({16}, tz::DType::Bool, device);
    auto s = tz::sum(t);
    EXPECT_EQ(s.cpu().item<int64_t>(), 16);
}

// ── Task 2.7: mean on integer dtypes returns Float32 ─────────────────────────

TEST_P(ReductionsDtypeCoverage, MeanInt8) {
    auto t = make_filled({16}, tz::DType::Int8, 4.0, device);
    auto m = tz::mean(t);
    // PyTorch returns Float32 for integer mean
    EXPECT_NEAR(m.cpu().item<float>(), 4.0f, 1e-6f);
}

TEST_P(ReductionsDtypeCoverage, MeanInt16) {
    auto t = make_filled({16}, tz::DType::Int16, 6.0, device);
    auto m = tz::mean(t);
    EXPECT_NEAR(m.cpu().item<float>(), 6.0f, 1e-6f);
}

TEST_P(ReductionsDtypeCoverage, MeanUInt8) {
    auto t = make_filled({16}, tz::DType::UInt8, 3.0, device);
    auto m = tz::mean(t);
    EXPECT_NEAR(m.cpu().item<float>(), 3.0f, 1e-6f);
}

TEST_P(ReductionsDtypeCoverage, MeanBool) {
    // 8 true out of 16 → mean = 0.5f
    auto host = tz::zeros({16}, tz::DType::Bool);
    auto* bp = host.data<bool>();
    for (int i = 0; i < 8; ++i) bp[i] = true;
    auto t = host.to(device);
    auto m = tz::mean(t);
    EXPECT_NEAR(m.cpu().item<float>(), 0.5f, 1e-6f);
}

// ── Task 2.7: argmax on integer/Bool dtypes ───────────────────────────────────

TEST_P(ReductionsDtypeCoverage, ArgmaxInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::Int8, device);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 15);
}

TEST_P(ReductionsDtypeCoverage, ArgmaxInt16) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::Int16, device);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 15);
}

TEST_P(ReductionsDtypeCoverage, ArgmaxUInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::UInt8, device);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 15);
}

TEST_P(ReductionsDtypeCoverage, ArgmaxUInt16) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::UInt16, device);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 15);
}

TEST_P(ReductionsDtypeCoverage, ArgmaxBool) {
    // [false, true, false] → argmax = 1
    auto host = tz::zeros({3}, tz::DType::Bool);
    host.data<bool>()[1] = true;
    auto t = host.to(device);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 1);
}

// ── Task 2.7: argmin on integer/Bool dtypes ───────────────────────────────────

TEST_P(ReductionsDtypeCoverage, ArgminInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::Int8, device);
    auto idx = tz::argmin(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 0);
}

TEST_P(ReductionsDtypeCoverage, ArgminUInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::UInt8, device);
    auto idx = tz::argmin(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 0);
}

TEST_P(ReductionsDtypeCoverage, ArgminBool) {
    // [true, false, true] → argmin = 1
    auto host = tz::ones({3}, tz::DType::Bool);
    host.data<bool>()[1] = false;
    auto t = host.to(device);
    auto idx = tz::argmin(t);
    EXPECT_EQ(idx.cpu().item<int64_t>(), 1);
}

// ── Task 2.7: prod on integer dtypes ─────────────────────────────────────────

TEST_P(ReductionsDtypeCoverage, ProdInt8) {
    // prod([1,2,3,4]) = 24; output Int64
    auto t = tz::arange(1.0, 5.0, 1.0, tz::DType::Int8, device);
    auto p = tz::prod(t);
    EXPECT_EQ(p.cpu().item<int64_t>(), 24);
}

TEST_P(ReductionsDtypeCoverage, ProdInt16) {
    auto t = tz::arange(1.0, 5.0, 1.0, tz::DType::Int16, device);
    auto p = tz::prod(t);
    EXPECT_EQ(p.cpu().item<int64_t>(), 24);
}

TEST_P(ReductionsDtypeCoverage, ProdUInt8) {
    auto t = tz::arange(1.0, 5.0, 1.0, tz::DType::UInt8, device);
    auto p = tz::prod(t);
    EXPECT_EQ(p.cpu().item<int64_t>(), 24);
}

// ── mode / logcumsumexp on Float16 (CPU/CUDA parity) ─────────────────────────

TEST_P(ReductionsDtypeCoverage, ModeFloat16) {
    auto h32 = tz::empty({6}, tz::DType::Float32);
    auto* f = h32.data<float>();
    const float vals[6] = {1, 2, 2, 3, 3, 3};
    for (int i = 0; i < 6; ++i) f[i] = vals[i];
    auto t = h32.to(tz::DType::Float16).to(device);
    auto [mv, mi] = tz::mode(t);
    EXPECT_NEAR(mv.cpu().to(tz::DType::Float32).item<float>(), 3.0f, 1e-2f);
    EXPECT_EQ(mi.cpu().item<int64_t>(), 5);  // last index of the modal run
}

TEST_P(ReductionsDtypeCoverage, LogcumsumexpFloat16) {
    auto h32 = tz::empty({3}, tz::DType::Float32);
    auto* f = h32.data<float>(); f[0] = 0.0f; f[1] = 0.0f; f[2] = 0.0f;
    auto t = h32.to(tz::DType::Float16).to(device);
    auto r = tz::logcumsumexp(t, 0).to(tz::DType::Float32).cpu();
    auto* rp = r.data<float>();
    // logcumsumexp([0,0,0]) = [log1, log2, log3]
    EXPECT_NEAR(rp[0], 0.0f, 2e-2f);
    EXPECT_NEAR(rp[1], 0.6931f, 2e-2f);
    EXPECT_NEAR(rp[2], 1.0986f, 2e-2f);
}

// ── argmax on UInt32/UInt64; empty-reduction guard (CPU/CUDA parity) ─────────

TEST_P(ReductionsDtypeCoverage, ArgmaxUInt32) {
    auto host = tz::empty({4}, tz::DType::UInt32);
    auto* p = host.data<uint32_t>(); p[0] = 7; p[1] = 42; p[2] = 3; p[3] = 9;
    EXPECT_EQ(tz::argmax(host.to(device)).cpu().item<int64_t>(), 1);
}

TEST_P(ReductionsDtypeCoverage, ArgmaxEmptyThrows) {
    // argmax over a zero-size tensor must throw on every backend (no identity).
    auto t = tz::empty({0}, tz::DType::Float32).to(device);
    EXPECT_ANY_THROW(tz::argmax(t));
}

// ── prod on Complex64/Complex128 (CPU/CUDA parity) ───────────────────────────

TEST_P(ReductionsDtypeCoverage, ProdComplex64Full) {
    // (1+1i)*(2-1i) = 3+1i; *(0.5+2i) = -0.5+6.5i
    auto host = tz::empty({3}, tz::DType::Complex64);
    auto* p = host.data<std::complex<float>>();
    p[0] = {1.0f, 1.0f}; p[1] = {2.0f, -1.0f}; p[2] = {0.5f, 2.0f};
    auto pr = tz::prod(host.to(device));
    auto v = pr.cpu().item<std::complex<float>>();
    EXPECT_NEAR(v.real(), -0.5f, 1e-5f);
    EXPECT_NEAR(v.imag(), 6.5f, 1e-5f);
}

TEST_P(ReductionsDtypeCoverage, ProdComplex128Dim) {
    // rows: (1+1i)*(2+0i)=2+2i ; (0+1i)*(3-1i)=1+3i  (prod along dim=1)
    auto host = tz::empty({2, 2}, tz::DType::Complex128);
    auto* p = host.data<std::complex<double>>();
    p[0] = {1.0, 1.0}; p[1] = {2.0, 0.0}; p[2] = {0.0, 1.0}; p[3] = {3.0, -1.0};
    auto pr = tz::prod(host.to(device), /*dim=*/1).cpu();
    auto* r = pr.data<std::complex<double>>();
    EXPECT_NEAR(r[0].real(), 2.0, 1e-12); EXPECT_NEAR(r[0].imag(), 2.0, 1e-12);
    EXPECT_NEAR(r[1].real(), 1.0, 1e-12); EXPECT_NEAR(r[1].imag(), 3.0, 1e-12);
}

// ── argsort on integer dtypes (CPU/CUDA parity) ──────────────────────────────

TEST_P(ReductionsDtypeCoverage, ArgsortInt64) {
    // argsort([5,2,8,1]) ascending = [3,1,0,2]
    auto host = tz::empty({4}, tz::DType::Int64);
    auto* p = host.data<int64_t>(); p[0] = 5; p[1] = 2; p[2] = 8; p[3] = 1;
    auto idx = tz::argsort(host.to(device)).cpu();
    auto* ip = idx.data<int64_t>();
    EXPECT_EQ(ip[0], 3); EXPECT_EQ(ip[1], 1); EXPECT_EQ(ip[2], 0); EXPECT_EQ(ip[3], 2);
}

// ── unique(sorted=false) must be first-appearance order on both backends ─────

TEST_P(ReductionsDtypeCoverage, UniqueSortedFalseFirstAppearance) {
    auto host = tz::empty({7}, tz::DType::Int32);
    const int32_t v[7] = {3, 1, 3, 2, 1, 2, 3};  // first-appearance order: 3, 1, 2
    for (int i = 0; i < 7; ++i) host.data<int32_t>()[i] = v[i];
    auto [ud, invd, cntd] = tz::unique(host.to(device), /*sorted=*/false, false, false);
    auto id = ud.cpu();
    ASSERT_EQ(id.numel(), 3);
    const int32_t* b = id.data<int32_t>();
    EXPECT_EQ(b[0], 3) << "on " << device.to_string();
    EXPECT_EQ(b[1], 1) << "on " << device.to_string();
    EXPECT_EQ(b[2], 2) << "on " << device.to_string();
}

// ── sort/mode tie-break: CPU indices must match the device on duplicates ─────

TEST_P(ReductionsDtypeCoverage, ArgsortTieBreakMatchesCPU) {
    // Duplicate values: both backends return the same stable (ascending-index)
    // permutation. Previously CPU used an unstable std::sort.
    auto host = tz::empty({6}, tz::DType::Float32);
    const float v[6] = {3, 1, 3, 1, 2, 1};
    for (int i = 0; i < 6; ++i) host.data<float>()[i] = v[i];
    auto ic = tz::argsort(host).cpu();          // host is CPU
    auto id = tz::argsort(host.to(device)).cpu();
    const int64_t* a = ic.data<int64_t>();
    const int64_t* b = id.data<int64_t>();
    for (int i = 0; i < 6; ++i)
        EXPECT_EQ(a[i], b[i]) << "argsort tie idx " << i << " on " << device.to_string();
}

TEST_P(ReductionsDtypeCoverage, ModeTieBreakMatchesCPU) {
    // Modal value repeats; the returned index must match across backends.
    auto host = tz::empty({7}, tz::DType::Int32);
    const int32_t v[7] = {5, 2, 5, 2, 5, 2, 9};  // 5 and 2 both appear 3x; 2 is smaller
    for (int i = 0; i < 7; ++i) host.data<int32_t>()[i] = v[i];
    auto [vc, icd] = tz::mode(host);
    auto [vd, idd] = tz::mode(host.to(device));
    EXPECT_EQ(vc.cpu().item<int32_t>(), vd.cpu().item<int32_t>()) << "on " << device.to_string();
    EXPECT_EQ(icd.cpu().item<int64_t>(), idd.cpu().item<int64_t>()) << "on " << device.to_string();
}

// ── any/all on narrow integer dtypes (CPU/CUDA parity) ───────────────────────

TEST_P(ReductionsDtypeCoverage, AnyInt8) {
    auto host = tz::empty({4}, tz::DType::Int8);
    auto* p = host.data<int8_t>(); p[0] = 0; p[1] = 0; p[2] = 3; p[3] = 0;
    EXPECT_TRUE(tz::any(host.to(device)).cpu().item<bool>());
}

TEST_P(ReductionsDtypeCoverage, AllInt16) {
    auto host = tz::empty({4}, tz::DType::Int16);
    auto* p = host.data<int16_t>(); p[0] = 1; p[1] = 2; p[2] = 0; p[3] = 4;
    EXPECT_FALSE(tz::all(host.to(device)).cpu().item<bool>());
}

// ── Task 2.8: scatter_reduce Float16/BFloat16 ────────────────────────────────

TEST_P(ReductionsDtypeCoverage, ScatterReduceFloat16) {
    auto dst = make_filled({4}, tz::DType::Float16, 0.0, device);
    auto src = make_filled({4}, tz::DType::Float16, 1.0, device);
    auto idx = tz::arange(0.0, 4.0, 1.0, tz::DType::Int64, device);
    auto out = tz::scatter_reduce(dst, /*dim=*/0, idx, src, "sum");
    auto as_f64 = out.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], 1.0, 1e-3) << "F16 scatter_reduce at i=" << i;
}

TEST_P(ReductionsDtypeCoverage, ScatterReduceBFloat16) {
    auto dst = make_filled({4}, tz::DType::BFloat16, 0.0, device);
    auto src = make_filled({4}, tz::DType::BFloat16, 1.0, device);
    auto idx = tz::arange(0.0, 4.0, 1.0, tz::DType::Int64, device);
    auto out = tz::scatter_reduce(dst, 0, idx, src, "sum");
    auto as_f64 = out.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], 1.0, 1e-2) << "BF16 scatter_reduce at i=" << i;
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given reduction dtype throws → the corresponding cell FAILS.
// ---------------------------------------------------------------------------
// F065: NaN ordering must be consistent across backends. PyTorch/CPU convention
// (now enforced on both) treats NaN as the LARGEST value — sorts last ascending,
// first descending. These compare the device result to the CPU reference for
// sort/argsort/topk/kthvalue/median/mode on a NaN-containing input.
// ---------------------------------------------------------------------------
namespace {
static tz::Tensor nan_vec_cpu() {
    float vals[10] = {3.0f, NAN, -1.0f, 2.0f, NAN, 0.5f, -4.0f, 7.0f, 1.5f, NAN};
    auto t = tz::empty({10}, tz::DType::Float32, tz::Device::cpu());
    std::copy(vals, vals + 10, t.data<float>());
    return t;
}
static void expect_idx_equal(const tz::Tensor& a, const tz::Tensor& b, const char* what) {
    auto ac = a.to(tz::Device::cpu()).contiguous();
    auto bc = b.to(tz::Device::cpu()).contiguous();
    ASSERT_EQ(ac.numel(), bc.numel());
    const int64_t* ap = ac.data<int64_t>();
    const int64_t* bp = bc.data<int64_t>();
    for (int64_t i = 0; i < ac.numel(); ++i)
        EXPECT_EQ(ap[i], bp[i]) << what << " index " << i;
}
static void expect_val_equal(const tz::Tensor& a, const tz::Tensor& b, const char* what) {
    auto ac = a.to(tz::Device::cpu()).contiguous();
    auto bc = b.to(tz::Device::cpu()).contiguous();
    ASSERT_EQ(ac.numel(), bc.numel());
    const float* ap = ac.data<float>();
    const float* bp = bc.data<float>();
    for (int64_t i = 0; i < ac.numel(); ++i) {
        if (std::isnan(ap[i])) EXPECT_TRUE(std::isnan(bp[i])) << what << " value " << i;
        else EXPECT_NEAR(ap[i], bp[i], 1e-5f) << what << " value " << i;
    }
}
}  // namespace

TEST_P(ReductionsDtypeCoverage, NaNSortArgsortMatchCPU) {
    if (device.type == tz::Device::Type::CPU) return;
    auto x_cpu = nan_vec_cpu();
    auto x_dev = x_cpu.to(device);
    for (bool desc : {false, true}) {
        auto [sv_ref, si_ref] = tz::sort(x_cpu, 0, desc);
        auto [sv_dev, si_dev] = tz::sort(x_dev, 0, desc);
        expect_idx_equal(si_ref, si_dev, desc ? "sort-desc" : "sort-asc");
        expect_val_equal(sv_ref, sv_dev, desc ? "sort-desc" : "sort-asc");
        auto ai_ref = tz::argsort(x_cpu, 0, desc);
        auto ai_dev = tz::argsort(x_dev, 0, desc);
        expect_idx_equal(ai_ref, ai_dev, desc ? "argsort-desc" : "argsort-asc");
    }
}

TEST_P(ReductionsDtypeCoverage, NaNTopkMatchCPU) {
    if (device.type == tz::Device::Type::CPU) return;
    auto x_cpu = nan_vec_cpu();
    auto x_dev = x_cpu.to(device);
    // Compare the top-k VALUES (NaN-aware). The relative order of equal-rank NaN
    // entries is unspecified (as in PyTorch), so raw NaN indices are not asserted.
    for (bool largest : {true, false}) {
        auto [tv_ref, ti_ref] = tz::topk(x_cpu, 4, 0, largest, true);
        auto [tv_dev, ti_dev] = tz::topk(x_dev, 4, 0, largest, true);
        expect_val_equal(tv_ref, tv_dev, largest ? "topk-largest" : "topk-smallest");
    }
}

TEST_P(ReductionsDtypeCoverage, NaNKthvalueMedianModeMatchCPU) {
    if (device.type == tz::Device::Type::CPU) return;
    auto x_cpu = nan_vec_cpu();
    auto x_dev = x_cpu.to(device);
    auto [kv_ref, ki_ref] = tz::kthvalue(x_cpu, 3, 0, false);
    auto [kv_dev, ki_dev] = tz::kthvalue(x_dev, 3, 0, false);
    expect_val_equal(kv_ref, kv_dev, "kthvalue");
    expect_idx_equal(ki_ref, ki_dev, "kthvalue");
    auto [mv_ref, mi_ref] = tz::median(x_cpu, 0, false);
    auto [mv_dev, mi_dev] = tz::median(x_dev, 0, false);
    expect_val_equal(mv_ref, mv_dev, "median");
    auto [ov_ref, oi_ref] = tz::mode(x_cpu, 0, false);
    auto [ov_dev, oi_dev] = tz::mode(x_dev, 0, false);
    expect_val_equal(ov_ref, ov_dev, "mode");
}

INSTANTIATE_BACKEND_TESTS(ReductionsDtypeCoverage);
