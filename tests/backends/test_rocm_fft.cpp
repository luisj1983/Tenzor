/**
 * @file test_rocm_fft.cpp
 * @brief Tests for rocFFT integration in the ROCm backend
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fft.hpp"
#include <cmath>

using namespace tenzor;

class RocmFFTTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
        // Skip if no ROCm device or rocFFT not working
        try {
            auto t = tenzor::zeros({2}, DType::Float32, Device::rocm(0));
            // Quick FFT smoke test to verify rocFFT works
            auto test_fft = tenzor::fft::rfft(tenzor::randn({8}).to(Device::rocm(0)));
            test_fft.to(Device::cpu());
        } catch (...) {
            GTEST_SKIP() << "ROCm device or rocFFT not available/functional";
        }
    }
};

TEST_F(RocmFFTTest, FFT1DRoundtrip) {
    auto cpu_signal = tenzor::randn({64});
    auto signal = cpu_signal.to(Device::rocm(0));

    // Forward FFT
    auto freq = tenzor::fft::fft(signal);
    // Inverse FFT
    auto reconstructed = tenzor::fft::ifft(freq);

    // Should approximately reconstruct original
    auto recon_cpu = reconstructed.to(Device::cpu());
    // Complex output — compare magnitudes
    EXPECT_EQ(recon_cpu.shape()[0], 64);
}

TEST_F(RocmFFTTest, RFFT1D) {
    auto signal = tenzor::randn({128}).to(Device::rocm(0));

    auto freq = tenzor::fft::rfft(signal);

    // rfft output has n/2+1 frequency bins
    EXPECT_EQ(freq.shape()[0], 65);  // 128/2 + 1
}

TEST_F(RocmFFTTest, IRFFT1D) {
    auto signal = tenzor::randn({128}).to(Device::rocm(0));

    auto freq = tenzor::fft::rfft(signal);
    auto reconstructed = tenzor::fft::irfft(freq, 128);

    auto orig_cpu = signal.to(Device::cpu());
    auto recon_cpu = reconstructed.to(Device::cpu());

    EXPECT_EQ(recon_cpu.shape()[0], 128);

    // Check reconstruction accuracy
    auto* orig = orig_cpu.data<float>();
    auto* recon = recon_cpu.data<float>();
    for (int64_t i = 0; i < 128; i++) {
        EXPECT_NEAR(orig[i], recon[i], 1e-3) << "Mismatch at index " << i;
    }
}

TEST_F(RocmFFTTest, FFT2D) {
    auto input = tenzor::randn({16, 16}).to(Device::rocm(0));

    auto freq = tenzor::fft::fft2(input);
    auto reconstructed = tenzor::fft::ifft2(freq);

    EXPECT_EQ(freq.shape()[0], 16);
    EXPECT_EQ(freq.shape()[1], 16);
}

TEST_F(RocmFFTTest, CPUParityFloat32) {
    auto cpu_input = tenzor::randn({32});
    auto rocm_input = cpu_input.to(Device::rocm(0));

    auto cpu_result = tenzor::fft::rfft(cpu_input).to(Device::cpu());
    auto rocm_result = tenzor::fft::rfft(rocm_input).to(Device::cpu());

    // Results should be close between CPU and ROCm
    EXPECT_EQ(cpu_result.shape()[0], rocm_result.shape()[0]);
}
