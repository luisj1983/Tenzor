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

// FP8: 2.0 is exactly representable in both E4M3 and E5M2; verify round-trip.
TEST(FillKernelDtypeCoverage, FP8E4M3) {
    auto t = tz::empty({4}, tz::DType::FP8_E4M3);
    t.fill_(2.0);
    // Convert to Float32 and check every element is exactly 2.0.
    auto f32 = t.to(tz::DType::Float32);
    const float* p = f32.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(p[i], 2.0f) << "FP8_E4M3 fill_(2.0) round-trip mismatch at i=" << i;
    }
}
TEST(FillKernelDtypeCoverage, FP8E5M2) {
    auto t = tz::empty({4}, tz::DType::FP8_E5M2);
    t.fill_(2.0);
    auto f32 = t.to(tz::DType::Float32);
    const float* p = f32.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_FLOAT_EQ(p[i], 2.0f) << "FP8_E5M2 fill_(2.0) round-trip mismatch at i=" << i;
    }
}

// QInt8 with scale=1, zero_point=0: fill_(3.0) → qval=3, stored as int8=3.
TEST(FillKernelDtypeCoverage, QInt8RoundTrip) {
    auto t = tz::empty({4}, tz::DType::QInt8);
    t.set_quantization_params(1.0, 0);
    t.fill_(3.0);
    auto repr = t.int_repr();  // zero-copy view as Int8
    EXPECT_EQ(repr.dtype(), tz::DType::Int8);
    const int8_t* p = repr.data<int8_t>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(p[i], static_cast<int8_t>(3))
            << "QInt8 fill_(3.0) storage mismatch at i=" << i;
    }
    // Also verify dequantize round-trip
    auto deq = t.dequantize();
    const float* fp = deq.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(fp[i], 3.0f, 1e-6f)
            << "QInt8 dequantize after fill_(3.0) mismatch at i=" << i;
    }
}

// QUInt8 with scale=1, zero_point=0: fill_(200.0) → qval=200, stored as uint8=200.
// Key regression check: the old bug used int8_t* and clamped to [−128,127],
// so qval=200 would have been clamped to 127, not 200.
TEST(FillKernelDtypeCoverage, QUInt8RoundTrip) {
    auto t = tz::empty({4}, tz::DType::QUInt8);
    t.set_quantization_params(1.0, 0);
    t.fill_(200.0);
    auto repr = t.int_repr();  // zero-copy view as UInt8
    EXPECT_EQ(repr.dtype(), tz::DType::UInt8);
    const uint8_t* p = repr.data<uint8_t>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_EQ(p[i], static_cast<uint8_t>(200))
            << "QUInt8 fill_(200.0) storage mismatch at i=" << i
            << " (old bug: int8 branch clamped to 127, uint8 branch gives 200)";
    }
    // Verify dequantize round-trip: (200 - 0) * 1.0 = 200.0
    auto deq = t.dequantize();
    const float* fp = deq.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(fp[i], 200.0f, 1e-4f)
            << "QUInt8 dequantize after fill_(200.0) mismatch at i=" << i;
    }
}

// QInt4x2 with scale=1, zero_point=0: fill_(3.0) → qval=3, clamped to [-8,7],
// packed as byte = (3 & 0xF) | ((3 & 0xF) << 4) = 0x03 | 0x30 = 0x33.
TEST(FillKernelDtypeCoverage, QInt4x2NibblePacking) {
    // 4 logical elements → 2 bytes of packed storage (2 elements per byte)
    auto t = tz::empty({4}, tz::DType::QInt4x2);
    t.set_quantization_params(1.0, 0);
    t.fill_(3.0);
    // Raw storage: int8_t*, but reinterpreted as uint8_t for nibble inspection
    const uint8_t* raw = reinterpret_cast<const uint8_t*>(t.data<int8_t>());
    // numel() for QInt4x2 is the number of bytes = ceil(4/2) = 2
    const int64_t nbytes = t.numel();
    for (int64_t i = 0; i < nbytes; ++i) {
        const uint8_t lo = raw[i] & 0xF;
        const uint8_t hi = (raw[i] >> 4) & 0xF;
        EXPECT_EQ(lo, static_cast<uint8_t>(3))
            << "QInt4x2 fill_(3.0): low nibble mismatch at byte=" << i;
        EXPECT_EQ(hi, static_cast<uint8_t>(3))
            << "QInt4x2 fill_(3.0): high nibble mismatch at byte=" << i
            << " (old bug wrote 0 here — half the elements were silently zero)";
    }
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
