#include <gtest/gtest.h>
#include <complex>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

// Audit Phase 2 Tasks 2.6+2.7+2.8:
// 2.6: mean must work for Complex64/Complex128
// 2.7: sum/mean/var/argmax/argmin/prod must work for integer and Bool dtypes
// 2.8: scatter_reduce must work for Float16/BFloat16

class ReductionsDtypeCoverageEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new ReductionsDtypeCoverageEnv);

namespace {

// Create a tensor of `shape` with dtype `dt` filled with `val`.
tz::Tensor make_filled(std::vector<int64_t> shape, tz::DType dt, double val) {
    auto t = tz::empty(shape, dt);
    t.fill_(val);
    return t;
}

}  // namespace

// ── Task 2.6: mean Complex64/Complex128 ──────────────────────────────────────

TEST(ReductionsDtypeCoverage, MeanComplex64) {
    // mean([1+2i, 3+4i, 5+6i]) = (9+12i)/3 = 3+4i
    auto t = tz::empty({3}, tz::DType::Complex64);
    auto* p = t.data<std::complex<float>>();
    p[0] = {1.0f, 2.0f};
    p[1] = {3.0f, 4.0f};
    p[2] = {5.0f, 6.0f};
    auto m = tz::mean(t);
    auto v = m.item<std::complex<float>>();
    EXPECT_NEAR(v.real(), 3.0f, 1e-5f);
    EXPECT_NEAR(v.imag(), 4.0f, 1e-5f);
}

TEST(ReductionsDtypeCoverage, MeanComplex128) {
    auto t = tz::empty({3}, tz::DType::Complex128);
    auto* p = t.data<std::complex<double>>();
    p[0] = {1.0, 2.0}; p[1] = {3.0, 4.0}; p[2] = {5.0, 6.0};
    auto m = tz::mean(t);
    auto v = m.item<std::complex<double>>();
    EXPECT_NEAR(v.real(), 3.0, 1e-12);
    EXPECT_NEAR(v.imag(), 4.0, 1e-12);
}

// ── Task 2.7: sum on integer/Bool dtypes ─────────────────────────────────────

TEST(ReductionsDtypeCoverage, SumInt8) {
    // 16 elements each = 3, sum = 48; output dtype Int64 (PyTorch convention)
    auto t = make_filled({16}, tz::DType::Int8, 3.0);
    auto s = tz::sum(t);
    EXPECT_EQ(s.item<int64_t>(), 48);
}

TEST(ReductionsDtypeCoverage, SumInt16) {
    auto t = make_filled({16}, tz::DType::Int16, 5.0);
    auto s = tz::sum(t);
    EXPECT_EQ(s.item<int64_t>(), 80);
}

TEST(ReductionsDtypeCoverage, SumUInt8) {
    auto t = make_filled({16}, tz::DType::UInt8, 7.0);
    auto s = tz::sum(t);
    EXPECT_EQ(s.item<int64_t>(), 112);
}

TEST(ReductionsDtypeCoverage, SumUInt16) {
    auto t = make_filled({16}, tz::DType::UInt16, 100.0);
    auto s = tz::sum(t);
    EXPECT_EQ(s.item<int64_t>(), 1600);
}

TEST(ReductionsDtypeCoverage, SumBool) {
    // ones({16}, Bool) — all true, sum = 16 in Int64
    auto t = tz::ones({16}, tz::DType::Bool);
    auto s = tz::sum(t);
    EXPECT_EQ(s.item<int64_t>(), 16);
}

// ── Task 2.7: mean on integer dtypes returns Float32 ─────────────────────────

TEST(ReductionsDtypeCoverage, MeanInt8) {
    auto t = make_filled({16}, tz::DType::Int8, 4.0);
    auto m = tz::mean(t);
    // PyTorch returns Float32 for integer mean
    EXPECT_NEAR(m.item<float>(), 4.0f, 1e-6f);
}

TEST(ReductionsDtypeCoverage, MeanInt16) {
    auto t = make_filled({16}, tz::DType::Int16, 6.0);
    auto m = tz::mean(t);
    EXPECT_NEAR(m.item<float>(), 6.0f, 1e-6f);
}

TEST(ReductionsDtypeCoverage, MeanUInt8) {
    auto t = make_filled({16}, tz::DType::UInt8, 3.0);
    auto m = tz::mean(t);
    EXPECT_NEAR(m.item<float>(), 3.0f, 1e-6f);
}

TEST(ReductionsDtypeCoverage, MeanBool) {
    // 8 true out of 16 → mean = 0.5f
    auto t = tz::zeros({16}, tz::DType::Bool);
    auto* bp = t.data<bool>();
    for (int i = 0; i < 8; ++i) bp[i] = true;
    auto m = tz::mean(t);
    EXPECT_NEAR(m.item<float>(), 0.5f, 1e-6f);
}

// ── Task 2.7: argmax on integer/Bool dtypes ───────────────────────────────────

TEST(ReductionsDtypeCoverage, ArgmaxInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::Int8);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.item<int64_t>(), 15);
}

TEST(ReductionsDtypeCoverage, ArgmaxInt16) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::Int16);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.item<int64_t>(), 15);
}

TEST(ReductionsDtypeCoverage, ArgmaxUInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::UInt8);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.item<int64_t>(), 15);
}

TEST(ReductionsDtypeCoverage, ArgmaxUInt16) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::UInt16);
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.item<int64_t>(), 15);
}

TEST(ReductionsDtypeCoverage, ArgmaxBool) {
    // [false, true, false] → argmax = 1
    auto t = tz::zeros({3}, tz::DType::Bool);
    t.data<bool>()[1] = true;
    auto idx = tz::argmax(t);
    EXPECT_EQ(idx.item<int64_t>(), 1);
}

// ── Task 2.7: argmin on integer/Bool dtypes ───────────────────────────────────

TEST(ReductionsDtypeCoverage, ArgminInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::Int8);
    auto idx = tz::argmin(t);
    EXPECT_EQ(idx.item<int64_t>(), 0);
}

TEST(ReductionsDtypeCoverage, ArgminUInt8) {
    auto t = tz::arange(0.0, 16.0, 1.0, tz::DType::UInt8);
    auto idx = tz::argmin(t);
    EXPECT_EQ(idx.item<int64_t>(), 0);
}

TEST(ReductionsDtypeCoverage, ArgminBool) {
    // [true, false, true] → argmin = 1
    auto t = tz::ones({3}, tz::DType::Bool);
    t.data<bool>()[1] = false;
    auto idx = tz::argmin(t);
    EXPECT_EQ(idx.item<int64_t>(), 1);
}

// ── Task 2.7: prod on integer dtypes ─────────────────────────────────────────

TEST(ReductionsDtypeCoverage, ProdInt8) {
    // prod([1,2,3,4]) = 24; output Int64
    auto t = tz::arange(1.0, 5.0, 1.0, tz::DType::Int8);
    auto p = tz::prod(t);
    EXPECT_EQ(p.item<int64_t>(), 24);
}

TEST(ReductionsDtypeCoverage, ProdInt16) {
    auto t = tz::arange(1.0, 5.0, 1.0, tz::DType::Int16);
    auto p = tz::prod(t);
    EXPECT_EQ(p.item<int64_t>(), 24);
}

TEST(ReductionsDtypeCoverage, ProdUInt8) {
    auto t = tz::arange(1.0, 5.0, 1.0, tz::DType::UInt8);
    auto p = tz::prod(t);
    EXPECT_EQ(p.item<int64_t>(), 24);
}

// ── Task 2.8: scatter_reduce Float16/BFloat16 ────────────────────────────────

TEST(ReductionsDtypeCoverage, ScatterReduceFloat16) {
    auto dst = make_filled({4}, tz::DType::Float16, 0.0);
    auto src = make_filled({4}, tz::DType::Float16, 1.0);
    auto idx = tz::arange(0.0, 4.0, 1.0, tz::DType::Int64);
    auto out = tz::scatter_reduce(dst, /*dim=*/0, idx, src, "sum");
    auto as_f64 = out.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], 1.0, 1e-3) << "F16 scatter_reduce at i=" << i;
}

TEST(ReductionsDtypeCoverage, ScatterReduceBFloat16) {
    auto dst = make_filled({4}, tz::DType::BFloat16, 0.0);
    auto src = make_filled({4}, tz::DType::BFloat16, 1.0);
    auto idx = tz::arange(0.0, 4.0, 1.0, tz::DType::Int64);
    auto out = tz::scatter_reduce(dst, 0, idx, src, "sum");
    auto as_f64 = out.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], 1.0, 1e-2) << "BF16 scatter_reduce at i=" << i;
}
