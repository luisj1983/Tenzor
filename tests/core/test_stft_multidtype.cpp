/**
 * @file test_stft_multidtype.cpp
 * @brief Multi-backend + multi-dtype tests for STFT and ISTFT operations
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include "../multi_backend_dtype_fixture.hpp"

#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class STFTMultiDTypeTest : public MultiBackendDTypeTest {};

TEST_P(STFTMultiDTypeTest, BasicSTFTShape) {
    // Create signal in Float32, convert to test dtype
    int64_t signal_len = 256;
    auto t = tenzor::arange(0.0f, static_cast<float>(signal_len), 1.0f, DType::Float32, device());
    auto signal = tenzor::sin(t * 0.1f);
    if (dtype() != DType::Float32) {
        signal = signal.to(dtype());
    }

    int64_t n_fft = 64;
    auto result = tenzor::fft::stft(signal, n_fft);

    // Output should have freq_bins = n_fft/2+1
    EXPECT_EQ(result.ndim(), 2);
    EXPECT_EQ(result.shape()[0], n_fft / 2 + 1);
    EXPECT_TRUE(result.dtype() == DType::Complex64 || result.dtype() == DType::Complex128);
}

TEST_P(STFTMultiDTypeTest, STFTWithWindow) {
    int64_t signal_len = 256;
    auto signal = createRandn({signal_len});

    int64_t n_fft = 64;
    // Hann window -- build in Float32 then convert
    auto window = tenzor::zeros({n_fft}, DType::Float32, device());
    auto w_cpu = window.to(Device::cpu());
    auto* w = w_cpu.data<float>();
    for (int64_t i = 0; i < n_fft; i++) {
        w[i] = 0.5f * (1.0f - std::cos(2.0f * static_cast<float>(M_PI) * i / (n_fft - 1)));
    }
    window = w_cpu.to(device());
    if (dtype() != DType::Float32) {
        window = window.to(dtype());
    }

    auto result = tenzor::fft::stft(signal, n_fft, /*hop_length=*/-1, /*win_length=*/-1, window);
    EXPECT_EQ(result.shape()[0], n_fft / 2 + 1);
}

TEST_P(STFTMultiDTypeTest, STFTRoundtripShape) {
    int64_t signal_len = 256;
    auto signal = createRandn({signal_len});

    int64_t n_fft = 64;
    int64_t hop_length = 16;

    auto stft_out = tenzor::fft::stft(signal, n_fft, hop_length);
    auto reconstructed = tenzor::fft::istft(stft_out, n_fft, hop_length,
                                     /*win_length=*/-1, /*window=*/Tensor{},
                                     /*center=*/true, /*normalized=*/false,
                                     /*onesided=*/true, /*length=*/signal_len);

    EXPECT_EQ(reconstructed.shape()[0], signal_len);
}

TEST_P(STFTMultiDTypeTest, DifferentHopLengths) {
    int64_t signal_len = 256;
    auto signal = createRandn({signal_len});
    int64_t n_fft = 64;

    auto result1 = tenzor::fft::stft(signal, n_fft, /*hop_length=*/16);
    auto result2 = tenzor::fft::stft(signal, n_fft, /*hop_length=*/32);

    // Smaller hop length = more frames
    EXPECT_GT(result1.shape()[1], result2.shape()[1]);
}

INSTANTIATE_MULTI_BACKEND_DTYPE_TESTS(STFTMultiDTypeTest);
