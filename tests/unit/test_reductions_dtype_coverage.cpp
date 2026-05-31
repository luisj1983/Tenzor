#include <gtest/gtest.h>
#include <complex>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
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
INSTANTIATE_BACKEND_TESTS(ReductionsDtypeCoverage);
