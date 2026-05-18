#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

namespace tenzor { void initialize(); }

namespace tz = ::tenzor;

// Audit P0 #3: fill_kernel must either fill correctly for every advertised
// DType or throw loudly. It must never return uninitialized memory.

// Initialize Tenzor once for the whole binary.
class FillKernelDtypeCoverageEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};

static ::testing::Environment* const g_fill_kernel_dtype_coverage_env =
    ::testing::AddGlobalTestEnvironment(new FillKernelDtypeCoverageEnv);

namespace {
// Helper: confirm every element of `t` equals `expected_value` (within fp tol).
void expect_all_equal(const tz::Tensor& t, double expected_value) {
    auto cpu_t = t.cpu();
    auto as_f64 = cpu_t.to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int64_t i = 0; i < cpu_t.numel(); ++i) {
        EXPECT_NEAR(p[i], expected_value, 1e-6)
            << "fill mismatch at i=" << i
            << " dtype=" << static_cast<int>(t.dtype());
    }
}
}  // namespace

TEST(FillKernelDtypeCoverage, Float16)   { auto t = tz::empty({16}, tz::DType::Float16);   t.fill_(2.5);  expect_all_equal(t, 2.5); }
TEST(FillKernelDtypeCoverage, BFloat16)  { auto t = tz::empty({16}, tz::DType::BFloat16);  t.fill_(2.5);  expect_all_equal(t, 2.5); }
TEST(FillKernelDtypeCoverage, Int16)     { auto t = tz::empty({16}, tz::DType::Int16);     t.fill_(7.0);  expect_all_equal(t, 7.0); }
TEST(FillKernelDtypeCoverage, UInt16)    { auto t = tz::empty({16}, tz::DType::UInt16);    t.fill_(7.0);  expect_all_equal(t, 7.0); }
TEST(FillKernelDtypeCoverage, UInt32)    { auto t = tz::empty({16}, tz::DType::UInt32);    t.fill_(7.0);  expect_all_equal(t, 7.0); }
TEST(FillKernelDtypeCoverage, UInt64)    { auto t = tz::empty({16}, tz::DType::UInt64);    t.fill_(7.0);  expect_all_equal(t, 7.0); }
TEST(FillKernelDtypeCoverage, Bool)      { auto t = tz::empty({16}, tz::DType::Bool);      t.fill_(1.0);  expect_all_equal(t, 1.0); }
TEST(FillKernelDtypeCoverage, Complex64) {
    auto t = tz::empty({4}, tz::DType::Complex64);
    t.fill_(2.5);
    // Real part must equal 2.5; imag must be 0.
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 2.5, 1e-6) << "Complex64 fill real part mismatch i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-6) << "Complex64 fill imag part nonzero i=" << i;
    }
}
TEST(FillKernelDtypeCoverage, Complex128) {
    auto t = tz::empty({4}, tz::DType::Complex128);
    t.fill_(2.5);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 2.5, 1e-6);
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-6);
    }
}
// FP8 / QInt — these are advertised dtypes but quantized requires scale/zero_point.
// Either fill succeeds (FP8 lossy cast acceptable; quantized throws clearly).
TEST(FillKernelDtypeCoverage, FP8E4M3) {
    auto t = tz::empty({4}, tz::DType::FP8_E4M3);
    EXPECT_NO_THROW(t.fill_(2.0));
}
TEST(FillKernelDtypeCoverage, FP8E5M2) {
    auto t = tz::empty({4}, tz::DType::FP8_E5M2);
    EXPECT_NO_THROW(t.fill_(2.0));
}
TEST(FillKernelDtypeCoverage, QuantizedWithoutParamsThrows) {
    auto t = tz::empty({4}, tz::DType::QInt8);
    EXPECT_THROW(t.fill_(2.0), std::runtime_error)
        << "fill_ on QInt8 with no quantization params must throw — "
        << "silent no-op would mask audit P0 #3";
}

TEST(FillKernelDtypeCoverage, QuantizedUIntWithoutParamsThrows) {
    auto t = tz::empty({4}, tz::DType::QUInt8);
    EXPECT_THROW(t.fill_(2.0), std::runtime_error)
        << "fill_ on QUInt8 with no quantization params must throw — "
        << "silent no-op would mask audit P0 #3";
}

TEST(FillKernelDtypeCoverage, QuantizedInt4WithoutParamsThrows) {
    auto t = tz::empty({4}, tz::DType::QInt4x2);
    EXPECT_THROW(t.fill_(2.0), std::runtime_error)
        << "fill_ on QInt4x2 with no quantization params must throw — "
        << "silent no-op would mask audit P0 #3";
}
