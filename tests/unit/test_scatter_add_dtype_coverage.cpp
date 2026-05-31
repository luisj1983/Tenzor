#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

// Audit P0 #4: scatter_add must either succeed for every numeric dtype or
// throw loudly. Pre-fix it silently dropped writes for UInt16/32/64/Int16;
// Bool/Complex inputs hit the throw branch (correct), but the covered dtypes
// were too narrow for the integer type family.
//
// Parameterized over all backends via BackendTest: each TEST_P creates its
// self/index/src tensors on the fixture's `device`. A (backend, dtype) cell
// the backend does not implement will throw and FAIL the test — intentional.
class ScatterAddDtypeCoverage : public ::tenzor::testing::BackendTest {};

namespace {

// Build a length-4 Int64 index [0,1,2,3] on `device`.
tz::Tensor make_idx4(const tz::Device& device) {
    return tz::arange(0, 4, 1, tz::DType::Int64, device);
}

// Create a tensor of `shape` with dtype `dt` filled with `val` on `device`.
// Uses empty+fill_ because ones()/zeros() only cover F32/F64/I32/I64.
tz::Tensor make_filled(std::vector<int64_t> shape, tz::DType dt, double val,
                       const tz::Device& device) {
    auto t = tz::empty(shape, dt, device);
    t.fill_(val);
    return t;
}

}  // namespace

// ── Integer family ────────────────────────────────────────────────────────────

TEST_P(ScatterAddDtypeCoverage, Int16ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::Int16, 1.0, device);
    auto dst = make_filled({4}, tz::DType::Int16, 0.0, device);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(device), src);
    auto as_i32 = out.cpu().to(tz::DType::Int32);
    const int32_t* p = as_i32.data<int32_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "Int16 scatter_add silently dropped write at i=" << i;
}

TEST_P(ScatterAddDtypeCoverage, UInt16ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::UInt16, 1.0, device);
    auto dst = make_filled({4}, tz::DType::UInt16, 0.0, device);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(device), src);
    auto as_i32 = out.cpu().to(tz::DType::Int32);
    const int32_t* p = as_i32.data<int32_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "UInt16 scatter_add silently dropped write at i=" << i;
}

TEST_P(ScatterAddDtypeCoverage, UInt32ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::UInt32, 1.0, device);
    auto dst = make_filled({4}, tz::DType::UInt32, 0.0, device);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(device), src);
    auto as_i64 = out.cpu().to(tz::DType::Int64);
    const int64_t* p = as_i64.data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "UInt32 scatter_add silently dropped write at i=" << i;
}

TEST_P(ScatterAddDtypeCoverage, UInt64ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::UInt64, 1.0, device);
    auto dst = make_filled({4}, tz::DType::UInt64, 0.0, device);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(device), src);
    auto as_i64 = out.cpu().to(tz::DType::Int64);
    const int64_t* p = as_i64.data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "UInt64 scatter_add silently dropped write at i=" << i;
}

// ── Regression: already-covered dtypes still work ────────────────────────────

TEST_P(ScatterAddDtypeCoverage, Float32Regression) {
    auto src = tz::ones({4}, tz::DType::Float32, device);
    auto dst = tz::zeros({4}, tz::DType::Float32, device);
    auto out = tz::scatter_add(dst, 0, make_idx4(device), src);
    auto as_f64 = out.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], 1.0, 1e-9) << "Float32 regression at i=" << i;
}

TEST_P(ScatterAddDtypeCoverage, Int32Regression) {
    auto src = tz::ones({4}, tz::DType::Int32, device);
    auto dst = tz::zeros({4}, tz::DType::Int32, device);
    auto out = tz::scatter_add(dst, 0, make_idx4(device), src);
    auto as_i64 = out.cpu().to(tz::DType::Int64);
    const int64_t* p = as_i64.data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "Int32 regression at i=" << i;
}

// ── Complex: component-wise add is well-defined; if implemented verify result,
//    if not implemented must throw loudly (not silently drop). ─────────────────

TEST_P(ScatterAddDtypeCoverage, Complex64ThrowsLoudlyOrWrites) {
    auto src = make_filled({4}, tz::DType::Complex64, 1.0, device);
    auto dst = make_filled({4}, tz::DType::Complex64, 0.0, device);
    try {
        auto out = tz::scatter_add(dst, 0, make_idx4(device), src);
        // If implemented: real part must equal 1, imag part must equal 0.
        auto re = tz::real(out).cpu().to(tz::DType::Float64);
        const double* p = re.data<double>();
        for (int i = 0; i < 4; ++i)
            EXPECT_NEAR(p[i], 1.0, 1e-9)
                << "Complex64 scatter_add silently dropped write at i=" << i;
    } catch (const std::runtime_error&) {
        SUCCEED();  // explicit throw is acceptable
    }
}

TEST_P(ScatterAddDtypeCoverage, Complex128ThrowsLoudlyOrWrites) {
    auto src = make_filled({4}, tz::DType::Complex128, 1.0, device);
    auto dst = make_filled({4}, tz::DType::Complex128, 0.0, device);
    try {
        auto out = tz::scatter_add(dst, 0, make_idx4(device), src);
        auto re = tz::real(out).cpu().to(tz::DType::Float64);
        const double* p = re.data<double>();
        for (int i = 0; i < 4; ++i)
            EXPECT_NEAR(p[i], 1.0, 1e-9)
                << "Complex128 scatter_add silently dropped write at i=" << i;
    } catch (const std::runtime_error&) {
        SUCCEED();
    }
}

// ── Bool: scatter_add is arithmetic (+=); Bool doesn't have a sensible
//    integer-addition semantics. Must throw loudly. ──────────────────────────

TEST_P(ScatterAddDtypeCoverage, BoolThrowsLoudly) {
    auto src = make_filled({4}, tz::DType::Bool, 1.0, device);
    auto dst = make_filled({4}, tz::DType::Bool, 0.0, device);
    EXPECT_THROW(tz::scatter_add(dst, 0, make_idx4(device), src), std::runtime_error)
        << "scatter_add on Bool must throw — += on bool is UB-prone, "
           "not a silent no-op";
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given scatter_add dtype throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(ScatterAddDtypeCoverage);
