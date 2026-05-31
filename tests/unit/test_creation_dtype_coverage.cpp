#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/core/dtype.hpp"
#include "tenzor/ops/creation.hpp"
#include "../backend_test_fixture.hpp"

namespace tz = ::tenzor;

// Audit Phase 2 Tasks 2.1+2.2: creation kernels (zeros/ones/full/arange/linspace)
// must support every advertised dtype without throwing — on EVERY backend.
//
// Parameterized over all backends via BackendTest: each TEST_P creates its
// tensors on the fixture's `device`. A (backend, dtype) cell the backend does
// not implement will throw and FAIL the test — intentional, to surface the
// real coverage gap rather than hide it. Backends physically absent on the
// host are skipped by BackendTest::SetUp (availability, not capability).
class CreationDtypeCoverage : public ::tenzor::testing::BackendTest {};

namespace {

tz::Tensor as_f64(const tz::Tensor& t) { return t.cpu().to(tz::DType::Float64); }

void expect_all_equal(const tz::Tensor& t, double expected, double atol = 1e-6) {
    auto f = as_f64(t);
    const double* p = f.data<double>();
    for (int64_t i = 0; i < f.numel(); ++i) {
        EXPECT_NEAR(p[i], expected, atol)
            << "i=" << i << " dtype=" << static_cast<int>(t.dtype());
    }
}

}  // namespace

// ============================================================================
// zeros — all dtypes
// ============================================================================
TEST_P(CreationDtypeCoverage, ZerosFloat16)  { expect_all_equal(tz::zeros({16}, tz::DType::Float16, device),  0.0); }
TEST_P(CreationDtypeCoverage, ZerosBFloat16) { expect_all_equal(tz::zeros({16}, tz::DType::BFloat16, device), 0.0); }
TEST_P(CreationDtypeCoverage, ZerosInt8)     { expect_all_equal(tz::zeros({16}, tz::DType::Int8, device),     0.0); }
TEST_P(CreationDtypeCoverage, ZerosInt16)    { expect_all_equal(tz::zeros({16}, tz::DType::Int16, device),    0.0); }
TEST_P(CreationDtypeCoverage, ZerosUInt8)    { expect_all_equal(tz::zeros({16}, tz::DType::UInt8, device),    0.0); }
TEST_P(CreationDtypeCoverage, ZerosUInt16)   { expect_all_equal(tz::zeros({16}, tz::DType::UInt16, device),   0.0); }
TEST_P(CreationDtypeCoverage, ZerosUInt32)   { expect_all_equal(tz::zeros({16}, tz::DType::UInt32, device),   0.0); }
TEST_P(CreationDtypeCoverage, ZerosUInt64)   { expect_all_equal(tz::zeros({16}, tz::DType::UInt64, device),   0.0); }
TEST_P(CreationDtypeCoverage, ZerosBool)     { expect_all_equal(tz::zeros({16}, tz::DType::Bool, device),     0.0); }
TEST_P(CreationDtypeCoverage, ZerosComplex64) {
    auto t = tz::zeros({4}, tz::DType::Complex64, device);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 0.0, 1e-9) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}
TEST_P(CreationDtypeCoverage, ZerosComplex128) {
    auto t = tz::zeros({4}, tz::DType::Complex128, device);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 0.0, 1e-9) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}

// ============================================================================
// ones — all standard dtypes
// ============================================================================
TEST_P(CreationDtypeCoverage, OnesFloat16)   { expect_all_equal(tz::ones({16}, tz::DType::Float16, device),  1.0); }
TEST_P(CreationDtypeCoverage, OnesBFloat16)  { expect_all_equal(tz::ones({16}, tz::DType::BFloat16, device), 1.0); }
TEST_P(CreationDtypeCoverage, OnesInt8)      { expect_all_equal(tz::ones({16}, tz::DType::Int8, device),     1.0); }
TEST_P(CreationDtypeCoverage, OnesInt16)     { expect_all_equal(tz::ones({16}, tz::DType::Int16, device),    1.0); }
TEST_P(CreationDtypeCoverage, OnesUInt8)     { expect_all_equal(tz::ones({16}, tz::DType::UInt8, device),    1.0); }
TEST_P(CreationDtypeCoverage, OnesUInt16)    { expect_all_equal(tz::ones({16}, tz::DType::UInt16, device),   1.0); }
TEST_P(CreationDtypeCoverage, OnesUInt32)    { expect_all_equal(tz::ones({16}, tz::DType::UInt32, device),   1.0); }
TEST_P(CreationDtypeCoverage, OnesUInt64)    { expect_all_equal(tz::ones({16}, tz::DType::UInt64, device),   1.0); }
TEST_P(CreationDtypeCoverage, OnesBool)      { expect_all_equal(tz::ones({16}, tz::DType::Bool, device),     1.0); }
TEST_P(CreationDtypeCoverage, OnesComplex64) {
    auto t = tz::ones({4}, tz::DType::Complex64, device);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 1.0, 1e-9) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}
TEST_P(CreationDtypeCoverage, OnesComplex128) {
    auto t = tz::ones({4}, tz::DType::Complex128, device);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 1.0, 1e-9) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}

// ============================================================================
// full — precision and new dtypes
// ============================================================================
TEST_P(CreationDtypeCoverage, FullFloat64Precision) {
    // Pre-fix, full took `float` in the kernel. Float64 caller lost precision.
    double v = 1.234567890123456;  // 16 significant digits
    auto t = tz::full({4}, v, tz::DType::Float64, device);
    const double* p = t.cpu().data<double>();
    for (int i = 0; i < 4; ++i)
        EXPECT_DOUBLE_EQ(p[i], v) << "full() lost precision at i=" << i;
}
TEST_P(CreationDtypeCoverage, FullInt64NoTruncation) {
    // Pre-fix, arange/full took float start — big Int64 would truncate.
    int64_t v = (1LL << 50) + 7;  // > 2^32, won't fit in float
    auto t = tz::full({4}, static_cast<double>(v), tz::DType::Int64, device);
    const int64_t* p = t.cpu().data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], v) << "full() Int64 truncation at i=" << i;
}
TEST_P(CreationDtypeCoverage, FullFloat16)  { expect_all_equal(tz::full({16}, 2.0, tz::DType::Float16, device),  2.0, 1e-3); }
TEST_P(CreationDtypeCoverage, FullBFloat16) { expect_all_equal(tz::full({16}, 2.0, tz::DType::BFloat16, device), 2.0, 1e-2); }
TEST_P(CreationDtypeCoverage, FullInt8)     { expect_all_equal(tz::full({16}, 7.0, tz::DType::Int8, device),     7.0); }
TEST_P(CreationDtypeCoverage, FullInt16)    { expect_all_equal(tz::full({16}, 7.0, tz::DType::Int16, device),    7.0); }
TEST_P(CreationDtypeCoverage, FullUInt8)    { expect_all_equal(tz::full({16}, 7.0, tz::DType::UInt8, device),    7.0); }
TEST_P(CreationDtypeCoverage, FullUInt16)   { expect_all_equal(tz::full({16}, 7.0, tz::DType::UInt16, device),   7.0); }
TEST_P(CreationDtypeCoverage, FullUInt32)   { expect_all_equal(tz::full({16}, 7.0, tz::DType::UInt32, device),   7.0); }
TEST_P(CreationDtypeCoverage, FullUInt64)   { expect_all_equal(tz::full({16}, 7.0, tz::DType::UInt64, device),   7.0); }
TEST_P(CreationDtypeCoverage, FullBool)     { expect_all_equal(tz::full({16}, 1.0, tz::DType::Bool, device),     1.0); }
TEST_P(CreationDtypeCoverage, FullComplex64) {
    auto t = tz::full({4}, 3.5, tz::DType::Complex64, device);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 3.5, 1e-6) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}
TEST_P(CreationDtypeCoverage, FullComplex128) {
    auto t = tz::full({4}, 3.5, tz::DType::Complex128, device);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], 3.5, 1e-9) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}

// ============================================================================
// arange — new dtypes and precision
// ============================================================================
TEST_P(CreationDtypeCoverage, ArangeFloat16) {
    auto t = tz::arange(0.0, 4.0, 1.0, tz::DType::Float16, device);
    ASSERT_EQ(t.numel(), 4);
    auto f = as_f64(t);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(f.data<double>()[i], static_cast<double>(i), 1e-3) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeBFloat16) {
    auto t = tz::arange(0.0, 4.0, 1.0, tz::DType::BFloat16, device);
    ASSERT_EQ(t.numel(), 4);
    auto f = as_f64(t);
    for (int i = 0; i < 4; ++i)
        EXPECT_NEAR(f.data<double>()[i], static_cast<double>(i), 1e-2) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeInt8) {
    auto t = tz::arange(0.0, 8.0, 1.0, tz::DType::Int8, device);
    ASSERT_EQ(t.numel(), 8);
    const int8_t* p = t.cpu().data<int8_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(p[i], static_cast<int8_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeInt16) {
    auto t = tz::arange(0.0, 8.0, 1.0, tz::DType::Int16, device);
    ASSERT_EQ(t.numel(), 8);
    const int16_t* p = t.cpu().data<int16_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(p[i], static_cast<int16_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeUInt8) {
    auto t = tz::arange(0.0, 8.0, 1.0, tz::DType::UInt8, device);
    ASSERT_EQ(t.numel(), 8);
    const uint8_t* p = t.cpu().data<uint8_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(p[i], static_cast<uint8_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeUInt16) {
    auto t = tz::arange(0.0, 8.0, 1.0, tz::DType::UInt16, device);
    ASSERT_EQ(t.numel(), 8);
    const uint16_t* p = t.cpu().data<uint16_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(p[i], static_cast<uint16_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeUInt32) {
    auto t = tz::arange(0.0, 8.0, 1.0, tz::DType::UInt32, device);
    ASSERT_EQ(t.numel(), 8);
    const uint32_t* p = t.cpu().data<uint32_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(p[i], static_cast<uint32_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeUInt64) {
    auto t = tz::arange(0.0, 8.0, 1.0, tz::DType::UInt64, device);
    ASSERT_EQ(t.numel(), 8);
    const uint64_t* p = t.cpu().data<uint64_t>();
    for (int i = 0; i < 8; ++i)
        EXPECT_EQ(p[i], static_cast<uint64_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeInt64Precision) {
    // > 2^32 so float start/step would truncate.
    int64_t base = (1LL << 40);
    auto t = tz::arange(static_cast<double>(base),
                        static_cast<double>(base + 4), 1.0,
                        tz::DType::Int64, device);
    ASSERT_EQ(t.numel(), 4);
    const int64_t* p = t.cpu().data<int64_t>();
    for (int i = 0; i < 4; ++i)
        EXPECT_EQ(p[i], base + i) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, ArangeComplex64) {
    auto t = tz::arange(0.0, 4.0, 1.0, tz::DType::Complex64, device);
    ASSERT_EQ(t.numel(), 4);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], static_cast<double>(i), 1e-6) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}
TEST_P(CreationDtypeCoverage, ArangeComplex128) {
    auto t = tz::arange(0.0, 4.0, 1.0, tz::DType::Complex128, device);
    ASSERT_EQ(t.numel(), 4);
    auto re = tz::real(t).cpu().to(tz::DType::Float64);
    auto im = tz::imag(t).cpu().to(tz::DType::Float64);
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(re.data<double>()[i], static_cast<double>(i), 1e-9) << "re i=" << i;
        EXPECT_NEAR(im.data<double>()[i], 0.0, 1e-9) << "im i=" << i;
    }
}

// ============================================================================
// linspace — new dtypes
// ============================================================================
TEST_P(CreationDtypeCoverage, LinspaceFloat16) {
    auto t = tz::linspace(0.0f, 1.0f, 5, tz::DType::Float16, device);
    ASSERT_EQ(t.numel(), 5);
    auto f = as_f64(t);
    EXPECT_NEAR(f.data<double>()[0], 0.0, 1e-3);
    EXPECT_NEAR(f.data<double>()[4], 1.0, 1e-3);
}
TEST_P(CreationDtypeCoverage, LinspaceBFloat16) {
    auto t = tz::linspace(0.0f, 1.0f, 5, tz::DType::BFloat16, device);
    ASSERT_EQ(t.numel(), 5);
    auto f = as_f64(t);
    EXPECT_NEAR(f.data<double>()[0], 0.0, 1e-2);
    EXPECT_NEAR(f.data<double>()[4], 1.0, 1e-2);
}
TEST_P(CreationDtypeCoverage, LinspaceInt32) {
    auto t = tz::linspace(0.0f, 4.0f, 5, tz::DType::Int32, device);
    ASSERT_EQ(t.numel(), 5);
    const int32_t* p = t.cpu().data<int32_t>();
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(p[i], i) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, LinspaceInt64) {
    auto t = tz::linspace(0.0f, 4.0f, 5, tz::DType::Int64, device);
    ASSERT_EQ(t.numel(), 5);
    const int64_t* p = t.cpu().data<int64_t>();
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(p[i], static_cast<int64_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, LinspaceInt8) {
    auto t = tz::linspace(0.0f, 4.0f, 5, tz::DType::Int8, device);
    ASSERT_EQ(t.numel(), 5);
    const int8_t* p = t.cpu().data<int8_t>();
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(p[i], static_cast<int8_t>(i)) << "i=" << i;
}
TEST_P(CreationDtypeCoverage, LinspaceInt16) {
    auto t = tz::linspace(0.0f, 4.0f, 5, tz::DType::Int16, device);
    ASSERT_EQ(t.numel(), 5);
    const int16_t* p = t.cpu().data<int16_t>();
    for (int i = 0; i < 5; ++i)
        EXPECT_EQ(p[i], static_cast<int16_t>(i)) << "i=" << i;
}

// ============================================================================
// Audit Phase 2A follow-up: FP8 and quantized coverage tests.
// ============================================================================

TEST_P(CreationDtypeCoverage, ZerosFP8_E4M3) {
    auto t = tz::zeros({16}, tz::DType::FP8_E4M3, device);
    // Memset zero is correctly all-bits-zero for FP8 (the encoded zero).
    auto as_f64 = t.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int64_t i = 0; i < 16; ++i)
        EXPECT_NEAR(p[i], 0.0, 1e-9) << "FP8_E4M3 zeros at i=" << i;
}

TEST_P(CreationDtypeCoverage, ZerosFP8_E5M2) {
    auto t = tz::zeros({16}, tz::DType::FP8_E5M2, device);
    auto as_f64 = t.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int64_t i = 0; i < 16; ++i)
        EXPECT_NEAR(p[i], 0.0, 1e-9);
}

TEST_P(CreationDtypeCoverage, ZerosQInt8) {
    // zeros writes raw all-bits-zero. For QInt8 with zero_point=0 this dequantizes
    // to 0; for nonzero zero_point a separately-set-up test would be appropriate.
    auto t = tz::zeros({16}, tz::DType::QInt8, device);
    // Just confirm the call succeeds and produces a properly-sized buffer.
    EXPECT_EQ(t.numel(), 16);
    EXPECT_EQ(t.dtype(), tz::DType::QInt8);
}
TEST_P(CreationDtypeCoverage, ZerosQUInt8)  { auto t = tz::zeros({16}, tz::DType::QUInt8, device); EXPECT_EQ(t.numel(), 16); }
TEST_P(CreationDtypeCoverage, ZerosQInt4x2) { auto t = tz::zeros({8}, tz::DType::QInt4x2, device); EXPECT_EQ(t.numel(), 8); }

TEST_P(CreationDtypeCoverage, OnesFP8_E4M3) {
    auto t = tz::ones({16}, tz::DType::FP8_E4M3, device);
    auto as_f64 = t.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    for (int64_t i = 0; i < 16; ++i) EXPECT_NEAR(p[i], 1.0, 1e-3) << "FP8_E4M3 ones at i=" << i;
}

TEST_P(CreationDtypeCoverage, OnesFP8_E5M2) {
    auto t = tz::ones({16}, tz::DType::FP8_E5M2, device);
    auto as_f64 = t.cpu().to(tz::DType::Float64);
    const double* p = as_f64.data<double>();
    // E5M2 mantissa is wider; 1.0 should be exact.
    for (int64_t i = 0; i < 16; ++i) EXPECT_NEAR(p[i], 1.0, 1e-3) << "FP8_E5M2 ones at i=" << i;
}

TEST_P(CreationDtypeCoverage, OnesQInt8ThrowsWithoutParams) {
    // ones on a fresh quantized dtype with no quant params attached should throw.
    EXPECT_THROW(tz::ones({4}, tz::DType::QInt8, device), std::runtime_error);
}
TEST_P(CreationDtypeCoverage, OnesQUInt8ThrowsWithoutParams) {
    EXPECT_THROW(tz::ones({4}, tz::DType::QUInt8, device), std::runtime_error);
}

// ============================================================================
// linspace double-precision overload tests.
// ============================================================================

TEST_P(CreationDtypeCoverage, LinspaceFloat64Precision) {
    double start = 1.234567890123456;
    double end   = 9.876543210987654;
    auto t = tz::linspace(start, end, /*steps=*/4, tz::DType::Float64, device);
    const double* p = t.cpu().data<double>();
    EXPECT_DOUBLE_EQ(p[0], start) << "linspace lost double precision at start";
    EXPECT_DOUBLE_EQ(p[3], end)   << "linspace lost double precision at end";
}

TEST_P(CreationDtypeCoverage, LinspaceInt64LargeEndpoints) {
    int64_t start = (1LL << 50);
    int64_t end   = (1LL << 50) + 6;
    auto t = tz::linspace(static_cast<double>(start),
                          static_cast<double>(end),
                          /*steps=*/4, tz::DType::Int64, device);
    const int64_t* p = t.cpu().data<int64_t>();
    EXPECT_EQ(p[0], start);
    EXPECT_EQ(p[3], end);
    // Middle values: (1L<<50)+2 and (1L<<50)+4 expected.
    EXPECT_EQ(p[1], start + 2);
    EXPECT_EQ(p[2], start + 4);
}

// Fan every TEST_P above over all five backends. BackendTest::SetUp skips a
// backend that is physically absent on the host; a present backend that does
// not implement a given creation dtype throws → the corresponding cell FAILS.
INSTANTIATE_BACKEND_TESTS(CreationDtypeCoverage);
