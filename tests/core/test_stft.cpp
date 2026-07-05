/**
 * @file test_stft.cpp
 * @brief Tests for STFT and ISTFT operations
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fft.hpp"
#include <cmath>

using namespace tenzor;

class STFTTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

TEST_P(STFTTest, BasicSTFT) {
    // Simple sine wave signal
    int64_t signal_len = 256;
    auto t = tenzor::arange(0.0, static_cast<double>(signal_len), 1.0,
                            DType::Float32, device);
    auto signal = tenzor::sin(t * 0.1f);  // Low frequency sine

    int64_t n_fft = 64;
    auto result = tenzor::fft::stft(signal, n_fft);

    // Output should be complex with shape (n_fft/2+1, num_frames)
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], n_fft / 2 + 1);
    EXPECT_TRUE(result.dtype() == DType::Complex64 || result.dtype() == DType::Complex128);
}

TEST_P(STFTTest, STFTWithWindow) {
    int64_t signal_len = 256;
    auto signal = tenzor::randn({signal_len}, DType::Float32, device);

    int64_t n_fft = 64;
    // Hann window (build on CPU, then move to device)
    auto window_cpu = tenzor::zeros({n_fft}, DType::Float32, Device::cpu());
    auto* w = window_cpu.data<float>();
    for (int64_t i = 0; i < n_fft; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n_fft - 1)));
    }
    auto window = window_cpu.to(device);

    auto result = tenzor::fft::stft(signal, n_fft, /*hop_length=*/-1, /*win_length=*/-1, window);
    EXPECT_EQ(result.shape()[0], n_fft / 2 + 1);
}

TEST_P(STFTTest, STFTRoundtrip) {
    // STFT -> ISTFT should approximately reconstruct
    int64_t signal_len = 256;
    auto signal = tenzor::randn({signal_len}, DType::Float32, device);

    int64_t n_fft = 64;
    int64_t hop_length = 16;

    auto stft_out = tenzor::fft::stft(signal, n_fft, hop_length);
    auto reconstructed = tenzor::fft::istft(stft_out, n_fft, hop_length,
                                     /*win_length=*/-1, /*window=*/Tensor{},
                                     /*center=*/true, /*normalized=*/false,
                                     /*onesided=*/true, /*length=*/signal_len);

    // Check shapes match
    EXPECT_EQ(reconstructed.shape()[0], signal_len);
}

TEST_P(STFTTest, DifferentHopLengths) {
    // Test with different hop lengths
    int64_t signal_len = 256;
    auto signal = tenzor::randn({signal_len}, DType::Float32, device);
    int64_t n_fft = 64;

    auto result1 = tenzor::fft::stft(signal, n_fft, /*hop_length=*/16);
    auto result2 = tenzor::fft::stft(signal, n_fft, /*hop_length=*/32);

    // Smaller hop length = more frames
    EXPECT_GT(result1.shape()[1], result2.shape()[1]);
}

TEST_P(STFTTest, NormalizedRoundtripReconstructs) {
    // With normalized=true on BOTH stft and istft, the ortho-normalized round
    // trip must reconstruct the original amplitude. The old CUDA istft ignored
    // `normalized` and always used the "backward" inverse, mis-scaling the
    // reconstruction by ~sqrt(n_fft).
    const int64_t signal_len = 256, n_fft = 64, hop_length = 16;
    auto host = tenzor::empty({signal_len}, DType::Float32);
    for (int64_t i = 0; i < signal_len; ++i)
        host.data<float>()[i] = std::sin(0.1f * static_cast<float>(i));
    auto signal = host.to(device);

    auto spec = tenzor::fft::stft(signal, n_fft, hop_length, /*win_length=*/-1,
                                  /*window=*/Tensor{}, /*center=*/true,
                                  /*normalized=*/true, /*onesided=*/true);
    auto recon = tenzor::fft::istft(spec, n_fft, hop_length, /*win_length=*/-1,
                                    /*window=*/Tensor{}, /*center=*/true,
                                    /*normalized=*/true, /*onesided=*/true,
                                    /*length=*/signal_len).cpu();
    auto sig_cpu = signal.cpu();
    const float* r = recon.data<float>();
    const float* s = sig_cpu.data<float>();
    // Interior samples (away from edge-window effects) reconstruct closely.
    for (int64_t i = n_fft; i < signal_len - n_fft; ++i)
        EXPECT_NEAR(r[i], s[i], 2e-2f) << "i=" << i << " on " << device.to_string();
}

INSTANTIATE_BACKEND_TESTS(STFTTest);
