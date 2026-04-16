/**
 * @file test_fft_parity.cpp
 * @brief FFT operation parity tests across backends
 *
 * Verifies that FFT/IFFT/RFFT produce identical results across
 * CPU, CUDA, ROCm, OneAPI, and Vulkan backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// 1-D FFT Parity
// ============================================================================

TEST(FFTParity, FFT_1D_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::fft::fft(inputs[0]);
    }, {input}, 1e-4f, 1e-5f, "FFT 1D");
}

TEST(FFTParity, IFFT_1D_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // FFT then IFFT should round-trip
        auto freq = tenzor::fft::fft(inputs[0]);
        return tenzor::fft::ifft(freq);
    }, {input}, 1e-4f, 1e-5f, "FFT→IFFT roundtrip");
}

TEST(FFTParity, RFFT_1D_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return tenzor::fft::rfft(inputs[0]);
    }, {input}, 1e-4f, 1e-5f, "RFFT 1D");
}

TEST(FFTParity, FFT_1D_WithLength) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Zero-pad to 32 elements
        return tenzor::fft::fft(inputs[0], 32);
    }, {input}, 1e-4f, 1e-5f, "FFT 1D zero-padded");
}

// ============================================================================
// 2D / N-D FFT Parity (Phase 3.4)
// ============================================================================

TEST(FFTParity, FFT2_Basic) {
    auto input = randn({2, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::fft2(ins[0]);
    }, {input}, 1e-4f, 1e-5f, "fft2");
}

TEST(FFTParity, IFFT2_Roundtrip) {
    auto input = randn({2, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        auto f = tenzor::fft::fft2(ins[0]);
        return tenzor::fft::ifft2(f);
    }, {input}, 1e-4f, 1e-5f, "fft2/ifft2 roundtrip");
}

TEST(FFTParity, RFFT2_Basic) {
    auto input = randn({2, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::rfft2(ins[0]);
    }, {input}, 1e-4f, 1e-5f, "rfft2");
}

TEST(FFTParity, FFTN_Basic) {
    auto input = randn({2, 8, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::fftn(ins[0]);
    }, {input}, 1e-4f, 1e-5f, "fftn");
}

TEST(FFTParity, IFFTN_Roundtrip) {
    auto input = randn({2, 8, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        auto f = tenzor::fft::fftn(ins[0]);
        return tenzor::fft::ifftn(f);
    }, {input}, 1e-4f, 1e-5f, "fftn/ifftn roundtrip");
}

TEST(FFTParity, FFTShift) {
    auto input = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::fftshift(ins[0]);
    }, {input}, 1e-5f, 1e-7f, "fftshift");
}

// ============================================================================
// Signal processing (STFT/ISTFT/DCT)
// ============================================================================

TEST(FFTParity, STFT) {
    // Signal length 256, n_fft=64, hop=16
    auto input = randn({1, 256}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::stft(ins[0], /*n_fft=*/64, /*hop=*/16);
    }, {input}, 1e-4f, 1e-5f, "stft");
}

TEST(FFTParity, ISTFT_Roundtrip) {
    auto input = randn({1, 256}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        auto spec = tenzor::fft::stft(ins[0], 64, 16);
        return tenzor::fft::istft(spec, 64, 16);
    }, {input}, 1e-3f, 1e-4f, "stft/istft roundtrip");
}

TEST(FFTParity, DCT) {
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::dct(ins[0]);
    }, {input}, 1e-4f, 1e-5f, "dct");
}

TEST(FFTParity, DCT_IDCT_Roundtrip) {
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        auto d = tenzor::fft::dct(ins[0]);
        return tenzor::fft::idct(d);
    }, {input}, 1e-4f, 1e-5f, "dct/idct roundtrip");
}

TEST(FFTParity, MelScale) {
    // Power spectrogram input: (batch=2, freq_bins=65, time=16). A real
    // pipeline would feed |STFT|^2 here; any positive tensor works for
    // numerical parity.
    auto power = tenzor::abs(randn({2, 65, 16}, DType::Float32, Device::cpu()));
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::mel_scale(ins[0], /*n_mels=*/40,
                                       /*f_min=*/0.0, /*f_max=*/0.0,
                                       /*sample_rate=*/16000);
    }, {power}, 1e-3f, 1e-5f, "mel_scale");
}

TEST(FFTParity, MFCC) {
    // 1 second of 16kHz audio — too large for a parity test; use 4096 samples.
    auto waveform = randn({1, 4096}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& ins) {
        return tenzor::fft::mfcc(ins[0], /*sample_rate=*/16000,
                                  /*n_mfcc=*/20, /*n_mels=*/40,
                                  /*n_fft=*/256, /*hop_length=*/128);
    }, {waveform}, 1e-2f, 1e-3f, "mfcc");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
