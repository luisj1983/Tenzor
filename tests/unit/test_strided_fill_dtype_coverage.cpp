// Audit P0 #9: StridedFill on a non-contiguous view must fill correctly (or
// throw loudly) for every advertised DType. It must never silently no-op.
//
// The bug: cpu_kernel_registry.cpp StridedFill switch had `default: break;`,
// silently no-op'ing for Complex64/128, UInt16/32/64, FP8_*, and quantized.
//
// Test strategy: transpose a 2-D tensor to produce a non-contiguous strided
// view, call fill_(), then verify every element has the expected value.

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

class StridedFillDtypeCoverageEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_env =
    ::testing::AddGlobalTestEnvironment(new StridedFillDtypeCoverageEnv);

namespace {
// Verify all elements of a Float64-cast view equal expected_value.
void expect_all_near(const tz::Tensor& t, double expected_value, double tol = 1e-5) {
    auto cpu_t = t.cpu();
    auto as_f64 = cpu_t.to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_NEAR(p[i], expected_value, tol)
            << "StridedFill silent no-op at element i=" << i
            << " dtype=" << static_cast<int>(t.dtype());
    }
}

// Build a non-contiguous strided view via transpose, fill, verify.
template <tz::DType DT>
void run_strided_fill_check(double fill_value) {
    auto t = tz::empty({4, 4}, DT);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous())
        << "transpose of {4,4} must be non-contiguous to exercise StridedFill";
    view.fill_(fill_value);
    expect_all_near(view, fill_value);
}
} // namespace

// --- scalar integer / float types previously missing from the switch ---

TEST(StridedFillDtypeCoverage, UInt16) { run_strided_fill_check<tz::DType::UInt16>(7.0); }
TEST(StridedFillDtypeCoverage, UInt32) { run_strided_fill_check<tz::DType::UInt32>(7.0); }
TEST(StridedFillDtypeCoverage, UInt64) { run_strided_fill_check<tz::DType::UInt64>(7.0); }

// --- types already in the switch; regression guard ---

TEST(StridedFillDtypeCoverage, Int16)    { run_strided_fill_check<tz::DType::Int16>(7.0);    }
TEST(StridedFillDtypeCoverage, Int32)    { run_strided_fill_check<tz::DType::Int32>(7.0);    }
TEST(StridedFillDtypeCoverage, Int64)    { run_strided_fill_check<tz::DType::Int64>(7.0);    }
TEST(StridedFillDtypeCoverage, Float32)  { run_strided_fill_check<tz::DType::Float32>(2.5);  }
TEST(StridedFillDtypeCoverage, Float64)  { run_strided_fill_check<tz::DType::Float64>(2.5);  }
TEST(StridedFillDtypeCoverage, Float16)  { run_strided_fill_check<tz::DType::Float16>(2.5);  }
TEST(StridedFillDtypeCoverage, BFloat16) { run_strided_fill_check<tz::DType::BFloat16>(2.5); }

TEST(StridedFillDtypeCoverage, Bool) {
    auto t = tz::empty({4, 4}, tz::DType::Bool);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    view.fill_(1.0);
    auto cpu_t = view.cpu();
    auto as_i32 = cpu_t.to(tz::DType::Int32);
    const int32_t* p = as_i32.data<int32_t>();
    for (int64_t i = 0; i < view.numel(); ++i)
        EXPECT_EQ(p[i], 1) << "Bool StridedFill silent no-op at i=" << i;
}

// --- complex types: real part = fill value, imag = 0 ---

TEST(StridedFillDtypeCoverage, Complex64) {
    auto t = tz::empty({4, 4}, tz::DType::Complex64);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    view.fill_(2.5);
    auto re = tz::real(view).cpu().to(tz::DType::Float64);
    auto im = tz::imag(view).cpu().to(tz::DType::Float64);
    for (int64_t i = 0; i < view.numel(); ++i) {
        EXPECT_NEAR(re.data<double>()[i], 2.5, 1e-6)
            << "Complex64 StridedFill real part mismatch at i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-6)
            << "Complex64 StridedFill imag part nonzero at i=" << i;
    }
}

TEST(StridedFillDtypeCoverage, Complex128) {
    auto t = tz::empty({4, 4}, tz::DType::Complex128);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    view.fill_(2.5);
    auto re = tz::real(view).cpu().to(tz::DType::Float64);
    auto im = tz::imag(view).cpu().to(tz::DType::Float64);
    for (int64_t i = 0; i < view.numel(); ++i) {
        EXPECT_NEAR(re.data<double>()[i], 2.5, 1e-12)
            << "Complex128 StridedFill real part mismatch at i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-12)
            << "Complex128 StridedFill imag part nonzero at i=" << i;
    }
}

// --- FP8: 2.0 is exactly representable in both E4M3 and E5M2 ---

TEST(StridedFillDtypeCoverage, FP8E4M3) {
    auto t = tz::empty({4, 4}, tz::DType::FP8_E4M3);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    view.fill_(2.0);
    auto f32 = view.to(tz::DType::Float32);
    const float* p = f32.data<float>();
    for (int64_t i = 0; i < view.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 2.0f) << "FP8_E4M3 StridedFill round-trip at i=" << i;
}

TEST(StridedFillDtypeCoverage, FP8E5M2) {
    auto t = tz::empty({4, 4}, tz::DType::FP8_E5M2);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    view.fill_(2.0);
    auto f32 = view.to(tz::DType::Float32);
    const float* p = f32.data<float>();
    for (int64_t i = 0; i < view.numel(); ++i)
        EXPECT_FLOAT_EQ(p[i], 2.0f) << "FP8_E5M2 StridedFill round-trip at i=" << i;
}

// --- Quantized: fill on strided view with scale/zero_point set ---
//
// NOTE: TensorImpl copy constructor (used by transpose to create the view's
// impl_) does not propagate q_scale_/q_zero_point_ (pre-existing separate
// bug in TensorImpl). We therefore call set_quantization_params on the view
// itself so the strided fill path (which reads self.q_scale()) works.

TEST(StridedFillDtypeCoverage, QInt8Strided) {
    auto t = tz::empty({4, 4}, tz::DType::QInt8);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    // Set quant params on the view's own impl so fill_ can read them.
    view.set_quantization_params(1.0, 0);
    // Should not throw and should encode 3 -> qval=3, stored as int8(3)
    ASSERT_NO_THROW(view.fill_(3.0));
    // Verify via int_repr of the view
    auto repr = view.int_repr();
    const int8_t* p = repr.data<int8_t>();
    for (int64_t i = 0; i < view.numel(); ++i)
        EXPECT_EQ(p[i], static_cast<int8_t>(3))
            << "QInt8 StridedFill storage mismatch at i=" << i;
}

TEST(StridedFillDtypeCoverage, QUInt8Strided) {
    auto t = tz::empty({4, 4}, tz::DType::QUInt8);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    view.set_quantization_params(1.0, 0);
    ASSERT_NO_THROW(view.fill_(5.0));
    auto repr = view.int_repr();
    EXPECT_EQ(repr.dtype(), tz::DType::UInt8);
    const uint8_t* p = repr.data<uint8_t>();
    for (int64_t i = 0; i < view.numel(); ++i)
        EXPECT_EQ(p[i], static_cast<uint8_t>(5))
            << "QUInt8 StridedFill storage mismatch at i=" << i;
}

// Quantized without params must throw, not silently no-op.
TEST(StridedFillDtypeCoverage, QInt8NoParamsThrows) {
    auto t = tz::empty({4, 4}, tz::DType::QInt8);
    auto view = t.transpose(0, 1);
    ASSERT_FALSE(view.is_contiguous());
    // Do NOT call set_quantization_params -- q_scale() == 0.0 must throw
    EXPECT_THROW(view.fill_(3.0), std::runtime_error);
}
