/**
 * @file test_fft_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for FFT operations:
 *        fft, ifft, rfft, irfft, batched FFT, padding
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/fft.hpp>
#include <tenzor/ops/creation.hpp>
#include <cmath>
#include <complex>

using namespace tenzor;
using namespace tenzor::testing;

class FFTMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(FFTMultiDTypeTest, FFTRoundtrip1D) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = ones({8}, dtype(), device());
    auto X = fft::fft(x, std::nullopt, -1, "backward");
    auto y = fft::ifft(X, std::nullopt, -1, "backward");

    auto y_cpu = y.to(Device::cpu());
    // Complex output: convert to check real part
    if (y_cpu.dtype() == DType::Complex64) {
        auto* yp = y_cpu.data<std::complex<float>>();
        for (int i = 0; i < 8; ++i) {
            EXPECT_NEAR(yp[i].real(), 1.0f, atol()) << "index " << i;
            EXPECT_NEAR(yp[i].imag(), 0.0f, atol()) << "index " << i;
        }
    } else if (y_cpu.dtype() == DType::Complex128) {
        auto* yp = y_cpu.data<std::complex<double>>();
        for (int i = 0; i < 8; ++i) {
            EXPECT_NEAR(yp[i].real(), 1.0, 1e-10) << "index " << i;
            EXPECT_NEAR(yp[i].imag(), 0.0, 1e-10) << "index " << i;
        }
    }
}

TEST_P(FFTMultiDTypeTest, DCSignal) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = ones({4}, dtype(), device());
    auto X = fft::fft(x, std::nullopt, -1, "backward");

    auto X_cpu = X.to(Device::cpu());
    if (X_cpu.dtype() == DType::Complex64) {
        auto* Xp = X_cpu.data<std::complex<float>>();
        EXPECT_NEAR(Xp[0].real(), 4.0f, atol());
        EXPECT_NEAR(Xp[0].imag(), 0.0f, atol());
        for (int k = 1; k < 4; ++k) {
            EXPECT_NEAR(Xp[k].real(), 0.0f, atol());
            EXPECT_NEAR(Xp[k].imag(), 0.0f, atol());
        }
    } else if (X_cpu.dtype() == DType::Complex128) {
        auto* Xp = X_cpu.data<std::complex<double>>();
        EXPECT_NEAR(Xp[0].real(), 4.0, 1e-10);
        for (int k = 1; k < 4; ++k) {
            EXPECT_NEAR(Xp[k].real(), 0.0, 1e-10);
        }
    }
}

TEST_P(FFTMultiDTypeTest, ImpulseSignal) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = zeros({4}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 1.0f;
    x = x.to(dtype()).to(device());

    auto X = fft::fft(x, std::nullopt, -1, "backward");
    auto X_cpu = X.to(Device::cpu());
    if (X_cpu.dtype() == DType::Complex64) {
        auto* Xp = X_cpu.data<std::complex<float>>();
        for (int k = 0; k < 4; ++k) {
            EXPECT_NEAR(Xp[k].real(), 1.0f, atol()) << "k=" << k;
            EXPECT_NEAR(Xp[k].imag(), 0.0f, atol()) << "k=" << k;
        }
    } else if (X_cpu.dtype() == DType::Complex128) {
        auto* Xp = X_cpu.data<std::complex<double>>();
        for (int k = 0; k < 4; ++k) {
            EXPECT_NEAR(Xp[k].real(), 1.0, 1e-10);
        }
    }
}

TEST_P(FFTMultiDTypeTest, RFFTOnes) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = ones({8}, dtype(), device());
    auto X = fft::rfft(x, std::nullopt, -1, "backward");

    EXPECT_EQ(X.shape()[0], 5);  // n/2 + 1

    auto X_cpu = X.to(Device::cpu());
    if (X_cpu.dtype() == DType::Complex64) {
        auto* Xp = X_cpu.data<std::complex<float>>();
        EXPECT_NEAR(Xp[0].real(), 8.0f, atol());
        for (int k = 1; k < 5; ++k) {
            EXPECT_NEAR(Xp[k].real(), 0.0f, atol());
        }
    } else if (X_cpu.dtype() == DType::Complex128) {
        auto* Xp = X_cpu.data<std::complex<double>>();
        EXPECT_NEAR(Xp[0].real(), 8.0, 1e-10);
    }
}

TEST_P(FFTMultiDTypeTest, IRFFTRoundtrip) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = ones({8}, dtype(), device());
    auto X = fft::rfft(x, std::nullopt, -1, "backward");
    auto y = fft::irfft(X, 8, -1, "backward");

    auto y_cpu = y.to(Device::cpu()).to(DType::Float32);
    auto* yp = y_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(yp[i], 1.0f, atol()) << "index=" << i;
    }
}

TEST_P(FFTMultiDTypeTest, BatchedFFT) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = ones({3, 4}, dtype(), device());
    auto X = fft::fft(x, std::nullopt, -1, "backward");

    EXPECT_EQ(X.shape()[0], 3);
    EXPECT_EQ(X.shape()[1], 4);

    auto X_cpu = X.to(Device::cpu());
    if (X_cpu.dtype() == DType::Complex64) {
        auto* Xp = X_cpu.data<std::complex<float>>();
        for (int b = 0; b < 3; ++b) {
            EXPECT_NEAR(Xp[b * 4].real(), 4.0f, atol());
            for (int k = 1; k < 4; ++k) {
                EXPECT_NEAR(Xp[b * 4 + k].real(), 0.0f, atol());
            }
        }
    }
}

TEST_P(FFTMultiDTypeTest, FFTWithPadding) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 FFT precision insufficient";
    auto x = ones({4}, dtype(), device());
    auto X = fft::fft(x, 8, -1, "backward");

    EXPECT_EQ(X.shape()[0], 8);
    auto X_cpu = X.to(Device::cpu());
    if (X_cpu.dtype() == DType::Complex64) {
        auto* Xp = X_cpu.data<std::complex<float>>();
        EXPECT_NEAR(Xp[0].real(), 4.0f, atol());
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FFTMultiDTypeTest);
