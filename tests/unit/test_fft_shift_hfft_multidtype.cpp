/**
 * @file test_fft_shift_hfft_multidtype.cpp
 * @brief Multi-backend multi-dtype tests for fftshift, ifftshift, hfft, ihfft
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include <tenzor/ops/fft.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <cmath>
#include <cstring>

using namespace tenzor;
using namespace tenzor::testing;

class FFTShiftHfftMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(FFTShiftHfftMultiDTypeTest, FftshiftEvenLength1D) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for FFT shift";
    auto x = zeros({4}, DType::Float32, Device::cpu());
    for (int i = 0; i < 4; ++i) x.data<float>()[i] = static_cast<float>(i);
    x = x.to(dtype()).to(device());

    auto out = fft::fftshift(x);
    auto o_cpu = out.to(Device::cpu()).to(DType::Float32);
    auto* o = o_cpu.data<float>();
    EXPECT_NEAR(o[0], 2.0f, atol());
    EXPECT_NEAR(o[1], 3.0f, atol());
    EXPECT_NEAR(o[2], 0.0f, atol());
    EXPECT_NEAR(o[3], 1.0f, atol());
}

TEST_P(FFTShiftHfftMultiDTypeTest, FftshiftOddLength1D) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for FFT shift";
    auto x = zeros({5}, DType::Float32, Device::cpu());
    for (int i = 0; i < 5; ++i) x.data<float>()[i] = static_cast<float>(i);
    x = x.to(dtype()).to(device());

    auto out = fft::fftshift(x);
    auto o_cpu = out.to(Device::cpu()).to(DType::Float32);
    auto* o = o_cpu.data<float>();
    EXPECT_NEAR(o[0], 3.0f, atol());
    EXPECT_NEAR(o[1], 4.0f, atol());
    EXPECT_NEAR(o[2], 0.0f, atol());
    EXPECT_NEAR(o[3], 1.0f, atol());
    EXPECT_NEAR(o[4], 2.0f, atol());
}

TEST_P(FFTShiftHfftMultiDTypeTest, IfftshiftInverseFftshift) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for FFT shift";
    for (int64_t N : {4, 5, 8, 9}) {
        auto x = zeros({N}, DType::Float32, Device::cpu());
        for (int i = 0; i < N; ++i) x.data<float>()[i] = static_cast<float>(i);
        x = x.to(dtype()).to(device());

        auto shifted = fft::fftshift(x);
        auto restored = fft::ifftshift(shifted);
        auto r_cpu = restored.to(Device::cpu()).to(DType::Float32);
        for (int i = 0; i < N; ++i) {
            EXPECT_NEAR(r_cpu.data<float>()[i], static_cast<float>(i), atol())
                << "N=" << N << " i=" << i;
        }
    }
}

TEST_P(FFTShiftHfftMultiDTypeTest, Fftshift2D) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for FFT shift";
    auto x = zeros({2, 2}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 0.0f;
    x.data<float>()[1] = 1.0f;
    x.data<float>()[2] = 2.0f;
    x.data<float>()[3] = 3.0f;
    x = x.to(dtype()).to(device());

    auto out = fft::fftshift(x);
    auto o_cpu = out.to(Device::cpu()).to(DType::Float32);
    auto* o = o_cpu.data<float>();
    EXPECT_NEAR(o[0], 3.0f, atol());
    EXPECT_NEAR(o[1], 2.0f, atol());
    EXPECT_NEAR(o[2], 1.0f, atol());
    EXPECT_NEAR(o[3], 0.0f, atol());
}

TEST_P(FFTShiftHfftMultiDTypeTest, IhfftHfftRoundtrip) {
    if (dtype() == DType::Float16) GTEST_SKIP() << "Float16 imprecise for HFFT";
    auto x = zeros({8}, DType::Float32, Device::cpu());
    for (int i = 0; i < 8; ++i)
        x.data<float>()[i] = std::sin(0.5f * static_cast<float>(i));
    x = x.to(dtype()).to(device());

    auto spec = fft::ihfft(x);
    auto recovered = fft::hfft(spec, 8);
    auto r_cpu = recovered.to(Device::cpu()).to(DType::Float32);
    auto x_cpu = x.to(Device::cpu()).to(DType::Float32);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(r_cpu.data<float>()[i], x_cpu.data<float>()[i], atol())
            << "i=" << i;
    }
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(FFTShiftHfftMultiDTypeTest);
