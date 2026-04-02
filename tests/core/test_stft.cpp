/**
 * @file test_stft.cpp
 * @brief Tests for STFT and ISTFT operations
 */

#include <gtest/gtest.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fft.hpp"
#include <cmath>

using namespace tenzor;

class STFTTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();
    }
};

TEST_F(STFTTest, BasicSTFT) {
    // Simple sine wave signal
    int64_t signal_len = 256;
    auto t = tenzor::arange(0.0f, static_cast<float>(signal_len), 1.0f);
    auto signal = tenzor::sin(t * 0.1f);  // Low frequency sine

    int64_t n_fft = 64;
    auto result = tenzor::fft::stft(signal, n_fft);

    // Output should be complex with shape (n_fft/2+1, num_frames)
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], n_fft / 2 + 1);
    EXPECT_TRUE(result.dtype() == DType::Complex64 || result.dtype() == DType::Complex128);
}

TEST_F(STFTTest, STFTWithWindow) {
    int64_t signal_len = 256;
    auto signal = tenzor::randn({signal_len});

    int64_t n_fft = 64;
    // Hann window
    auto window = tenzor::zeros({n_fft});
    auto* w = window.data<float>();
    for (int64_t i = 0; i < n_fft; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * M_PI * i / (n_fft - 1)));
    }

    auto result = tenzor::fft::stft(signal, n_fft, /*hop_length=*/-1, /*win_length=*/-1, window);
    EXPECT_EQ(result.shape()[0], n_fft / 2 + 1);
}

TEST_F(STFTTest, STFTRoundtrip) {
    // STFT -> ISTFT should approximately reconstruct
    int64_t signal_len = 256;
    auto signal = tenzor::randn({signal_len});

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

TEST_F(STFTTest, DifferentHopLengths) {
    // Test with different hop lengths
    int64_t signal_len = 256;
    auto signal = tenzor::randn({signal_len});
    int64_t n_fft = 64;

    auto result1 = tenzor::fft::stft(signal, n_fft, /*hop_length=*/16);
    auto result2 = tenzor::fft::stft(signal, n_fft, /*hop_length=*/32);

    // Smaller hop length = more frames
    EXPECT_GT(result1.shape()[1], result2.shape()[1]);
}
