// test_complex_sign_nan_parity.cpp
//
// JIT-R171: sign(z) = z/|z| (and 0 for z == 0) previously silently returned
// 0+0i instead of propagating NaN+NaNi on CUDA/ROCm/OneAPI/Vulkan whenever a
// NaN component made |z| itself NaN (`mag > 0.0` is false for NaN mag under
// IEEE-754, so these four backends took the "zero magnitude" branch). CPU
// used `mag == 0.0f ? 0 : z/mag`, which correctly falls through to z/mag and
// propagates NaN. Reachable via slogdet() on an ill-conditioned/singular
// complex matrix producing a NaN-component determinant.
//
// The existing CPU-only unit test (tests/unit/test_complex_arithmetic.cpp)
// cannot exercise this on the GPU backends since that whole suite is
// currently CPU-only (complex arithmetic isn't fully wired for every op on
// every backend). This test calls sign() directly on each backend that
// implements it, so it verifies exactly the fixed code path without
// depending on that suite's broader instantiation.

#include <gtest/gtest.h>
#include <cmath>
#include <complex>
#include <limits>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;

namespace {

class ComplexSignNanParity : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

template <typename ComplexT, DType kDType>
void run_nan_propagation_check(Device dev) {
    auto t = Tensor({int64_t(2)}, kDType, Device::cpu());
    using ScalarT = typename ComplexT::value_type;
    t.data<ComplexT>()[0] = ComplexT{static_cast<ScalarT>(3), static_cast<ScalarT>(4)};  // |z|=5, sanity case
    t.data<ComplexT>()[1] = ComplexT{std::numeric_limits<ScalarT>::quiet_NaN(),
                                      static_cast<ScalarT>(1)};  // NaN component

    auto t_dev = t.to(dev);
    auto result = sign(t_dev).to(Device::cpu());
    ASSERT_EQ(result.dtype(), kDType);
    auto* rp = result.data<ComplexT>();

    EXPECT_NEAR(rp[0].real(), static_cast<ScalarT>(0.6), static_cast<ScalarT>(1e-5))
        << tenzor::testing::device_type_to_backend_name(dev.type);
    EXPECT_NEAR(rp[0].imag(), static_cast<ScalarT>(0.8), static_cast<ScalarT>(1e-5))
        << tenzor::testing::device_type_to_backend_name(dev.type);

    EXPECT_TRUE(std::isnan(rp[1].real()))
        << "sign() must propagate NaN, not silently return 0, on "
        << tenzor::testing::device_type_to_backend_name(dev.type);
    EXPECT_TRUE(std::isnan(rp[1].imag()))
        << "sign() must propagate NaN, not silently return 0, on "
        << tenzor::testing::device_type_to_backend_name(dev.type);
}

}  // namespace

TEST_F(ComplexSignNanParity, Complex64SignPropagatesNanAcrossBackends) {
    std::vector<Device> devices = {Device::cpu(), Device::cuda(0), Device::rocm(0),
                                    Device::vulkan(0), Device::oneapi(0)};
    int checked = 0;
    for (const auto& dev : devices) {
        if (!tenzor::testing::is_backend_available(dev.type, dev.index)) continue;
        run_nan_propagation_check<std::complex<float>, DType::Complex64>(dev);
        ++checked;
    }
    if (checked == 0) {
        if (tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "No backend available to check Complex64 sign() NaN propagation";
        }
        GTEST_SKIP() << "No backend available";
    }
}

TEST_F(ComplexSignNanParity, Complex128SignPropagatesNanAcrossBackends) {
    std::vector<Device> devices = {Device::cpu(), Device::cuda(0), Device::rocm(0),
                                    Device::vulkan(0), Device::oneapi(0)};
    int checked = 0;
    for (const auto& dev : devices) {
        if (!tenzor::testing::is_backend_available(dev.type, dev.index)) continue;
        run_nan_propagation_check<std::complex<double>, DType::Complex128>(dev);
        ++checked;
    }
    if (checked == 0) {
        if (tenzor::testing::golden::require_multi_backend()) {
            FAIL() << "No backend available to check Complex128 sign() NaN propagation";
        }
        GTEST_SKIP() << "No backend available";
    }
}
