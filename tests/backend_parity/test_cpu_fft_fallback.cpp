// test_cpu_fft_fallback.cpp
//
// Wave Inf-A: CPU FFT now has a non-MKL O(N²) DFT fallback that handles any
// signal length without requiring MKL. The tests below run against whichever
// FFT path was compiled (MKL fast-path or DFT fallback) and verify roundtrip
// + numerical correctness against a hand-rolled reference.
//
// When MKL is present the tests verify the MKL path; when MKL is absent they
// verify the DFT fallback. Either way, FFT must work on CPU — which it didn't
// before Wave Inf-A (the non-MKL path threw `MKL not available`).

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include <complex>

// Qualify FFT functions to avoid clash with tenzor::fft sub-namespace.
namespace tfft = tenzor::fft;
using tenzor::Tensor;
using tenzor::Device;
using tenzor::DType;
using tenzor::zeros;

namespace {

class CpuFftFallback : public ::testing::Test {
protected:
    static void SetUpTestSuite() { tenzor::initialize(); }
};

}  // namespace

TEST_F(CpuFftFallback, FFT_Pow2_RealInput_MatchesHandRolled) {
    constexpr int64_t N = 8;
    auto x = zeros({N}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        xp[i] = static_cast<float>(i) * 0.25f - 1.0f;
    }

    Tensor X = tfft::fft(x, std::nullopt, -1, "backward");
    ASSERT_EQ(X.dtype(), DType::Complex64);
    ASSERT_EQ(X.numel(), N);

    // Hand-rolled DFT reference.
    auto* Xp = X.data<std::complex<float>>();
    for (int64_t k = 0; k < N; ++k) {
        std::complex<double> acc(0, 0);
        for (int64_t n = 0; n < N; ++n) {
            double phase = -2.0 * M_PI * static_cast<double>(k * n) / N;
            acc += std::complex<double>(xp[n], 0.0) *
                   std::complex<double>(std::cos(phase), std::sin(phase));
        }
        EXPECT_NEAR(Xp[k].real(), static_cast<float>(acc.real()), 5e-4f) << " k=" << k;
        EXPECT_NEAR(Xp[k].imag(), static_cast<float>(acc.imag()), 5e-4f) << " k=" << k;
    }
}

TEST_F(CpuFftFallback, FFT_Prime_RealInput_MatchesHandRolled) {
    constexpr int64_t N = 7;  // Prime — exercises non-power-of-2 path.
    auto x = zeros({N}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int64_t i = 0; i < N; ++i) xp[i] = std::sin(2.0f * static_cast<float>(M_PI) * i / N);

    Tensor X = tfft::fft(x, std::nullopt, -1, "backward");
    ASSERT_EQ(X.numel(), N);

    auto* Xp = X.data<std::complex<float>>();
    for (int64_t k = 0; k < N; ++k) {
        std::complex<double> acc(0, 0);
        for (int64_t n = 0; n < N; ++n) {
            double phase = -2.0 * M_PI * static_cast<double>(k * n) / N;
            acc += std::complex<double>(xp[n], 0.0) *
                   std::complex<double>(std::cos(phase), std::sin(phase));
        }
        EXPECT_NEAR(Xp[k].real(), static_cast<float>(acc.real()), 1e-3f) << " k=" << k;
        EXPECT_NEAR(Xp[k].imag(), static_cast<float>(acc.imag()), 1e-3f) << " k=" << k;
    }
}

TEST_F(CpuFftFallback, IFFT_Pow2_RoundtripsExact) {
    constexpr int64_t N = 16;
    auto x_real = zeros({N}, DType::Float32, Device::cpu());
    auto* rp = x_real.data<float>();
    for (int64_t i = 0; i < N; ++i) rp[i] = static_cast<float>(i % 5) * 0.1f;

    Tensor X = tfft::fft(x_real, std::nullopt, -1, "backward");
    Tensor x_back = tfft::ifft(X, std::nullopt, -1, "backward");
    ASSERT_EQ(x_back.dtype(), DType::Complex64);

    auto* xbp = x_back.data<std::complex<float>>();
    for (int64_t i = 0; i < N; ++i) {
        EXPECT_NEAR(xbp[i].real(), rp[i], 1e-4f) << " i=" << i;
        EXPECT_NEAR(std::abs(xbp[i].imag()), 0.0f, 1e-4f) << " i=" << i;
    }
}

TEST_F(CpuFftFallback, RFFT_IRFFT_RoundtripsExact) {
    constexpr int64_t N = 16;
    auto x = zeros({N}, DType::Float32, Device::cpu());
    auto* xp = x.data<float>();
    for (int64_t i = 0; i < N; ++i) xp[i] = std::cos(2.0f * static_cast<float>(M_PI) * i * 3.0f / N);

    Tensor X = tfft::rfft(x, std::nullopt, -1, "backward");
    ASSERT_EQ(X.shape().back(), N / 2 + 1);

    Tensor x_back = tfft::irfft(X, N, -1, "backward");
    ASSERT_EQ(x_back.shape().back(), N);
    ASSERT_EQ(x_back.dtype(), DType::Float32);

    auto* xbp = x_back.data<float>();
    for (int64_t i = 0; i < N; ++i) {
        EXPECT_NEAR(xbp[i], xp[i], 1e-4f) << " i=" << i;
    }
}
