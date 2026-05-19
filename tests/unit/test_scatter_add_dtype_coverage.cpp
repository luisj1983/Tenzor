#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/indexing.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

// Audit P0 #4: scatter_add must either succeed for every numeric dtype or
// throw loudly. Pre-fix it silently dropped writes for UInt16/32/64/Int16;
// Bool/Complex inputs hit the throw branch (correct), but the covered dtypes
// were too narrow for the integer type family.

class ScatterAddDtypeCoverageEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_scatter_add_dtype_env =
    ::testing::AddGlobalTestEnvironment(new ScatterAddDtypeCoverageEnv);

namespace {

// Build a length-4 Int64 index [0,1,2,3].
tz::Tensor make_idx4() {
    return tz::arange(0, 4, 1, tz::DType::Int64);
}

// Create a tensor of `shape` with dtype `dt` filled with `val`.
// Uses empty+fill_ because ones()/zeros() only cover F32/F64/I32/I64.
tz::Tensor make_filled(std::vector<int64_t> shape, tz::DType dt, double val) {
    auto t = tz::empty(shape, dt);
    t.fill_(val);
    return t;
}

}  // namespace

// ── Integer family ────────────────────────────────────────────────────────────

TEST(ScatterAddDtypeCoverage, Int16ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::Int16, 1.0);
    auto dst = make_filled({4}, tz::DType::Int16, 0.0);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(), src);
    auto as_i32 = out.cpu().to(tz::DType::Int32);
    const int32_t* p = as_i32.data<int32_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "Int16 scatter_add silently dropped write at i=" << i;
}

TEST(ScatterAddDtypeCoverage, UInt16ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::UInt16, 1.0);
    auto dst = make_filled({4}, tz::DType::UInt16, 0.0);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(), src);
    auto as_i32 = out.cpu().to(tz::DType::Int32);
    const int32_t* p = as_i32.data<int32_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "UInt16 scatter_add silently dropped write at i=" << i;
}

TEST(ScatterAddDtypeCoverage, UInt32ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::UInt32, 1.0);
    auto dst = make_filled({4}, tz::DType::UInt32, 0.0);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(), src);
    auto as_i64 = out.cpu().to(tz::DType::Int64);
    const int64_t* p = as_i64.data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "UInt32 scatter_add silently dropped write at i=" << i;
}

TEST(ScatterAddDtypeCoverage, UInt64ActuallyWrites) {
    auto src = make_filled({4}, tz::DType::UInt64, 1.0);
    auto dst = make_filled({4}, tz::DType::UInt64, 0.0);
    auto out = tz::scatter_add(dst, /*dim=*/0, make_idx4(), src);
    auto as_i64 = out.cpu().to(tz::DType::Int64);
    const int64_t* p = as_i64.data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "UInt64 scatter_add silently dropped write at i=" << i;
}

// ── Regression: already-covered dtypes still work ────────────────────────────

TEST(ScatterAddDtypeCoverage, Float32Regression) {
    auto src = tz::ones({4}, tz::DType::Float32);
    auto dst = tz::zeros({4}, tz::DType::Float32);
    auto out = tz::scatter_add(dst, 0, make_idx4(), src);
    auto as_f64 = out.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(p[i], 1.0, 1e-9) << "Float32 regression at i=" << i;
}

TEST(ScatterAddDtypeCoverage, Int32Regression) {
    auto src = tz::ones({4}, tz::DType::Int32);
    auto dst = tz::zeros({4}, tz::DType::Int32);
    auto out = tz::scatter_add(dst, 0, make_idx4(), src);
    auto as_i64 = out.cpu().to(tz::DType::Int64);
    const int64_t* p = as_i64.data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], 1) << "Int32 regression at i=" << i;
}

// ── Complex: component-wise add is well-defined; if implemented verify result,
//    if not implemented must throw loudly (not silently drop). ─────────────────

TEST(ScatterAddDtypeCoverage, Complex64ThrowsLoudlyOrWrites) {
    auto src = make_filled({4}, tz::DType::Complex64, 1.0);
    auto dst = make_filled({4}, tz::DType::Complex64, 0.0);
    try {
        auto out = tz::scatter_add(dst, 0, make_idx4(), src);
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

TEST(ScatterAddDtypeCoverage, Complex128ThrowsLoudlyOrWrites) {
    auto src = make_filled({4}, tz::DType::Complex128, 1.0);
    auto dst = make_filled({4}, tz::DType::Complex128, 0.0);
    try {
        auto out = tz::scatter_add(dst, 0, make_idx4(), src);
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

TEST(ScatterAddDtypeCoverage, BoolThrowsLoudly) {
    auto src = make_filled({4}, tz::DType::Bool, 1.0);
    auto dst = make_filled({4}, tz::DType::Bool, 0.0);
    EXPECT_THROW(tz::scatter_add(dst, 0, make_idx4(), src), std::runtime_error)
        << "scatter_add on Bool must throw — += on bool is UB-prone, "
           "not a silent no-op";
}
