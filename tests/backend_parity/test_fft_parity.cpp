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
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class FFTParity : public BackendTest {};
// ============================================================================
// 1-D FFT Parity
// ============================================================================

TEST_P(FFTParity, FFT_1D_Basic) {

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tenzor::fft::fft(inputs[0]);
    }, {input}, device, 1e-4f, 1e-5f, "FFT 1D");
}

TEST_P(FFTParity, IFFT_1D_Basic) {

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // FFT then IFFT should round-trip
        auto freq = tenzor::fft::fft(inputs[0]);
        return tenzor::fft::ifft(freq);
    }, {input}, device, 1e-4f, 1e-5f, "FFT→IFFT roundtrip");
}

TEST_P(FFTParity, RFFT_1D_Basic) {

    auto input = randn({4, 32}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return tenzor::fft::rfft(inputs[0]);
    }, {input}, device, 1e-4f, 1e-5f, "RFFT 1D");
}

TEST_P(FFTParity, FFT_1D_WithLength) {

    auto input = randn({4, 16}, DType::Float32, Device::cpu());

    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        // Zero-pad to 32 elements
        return tenzor::fft::fft(inputs[0], 32);
    }, {input}, device, 1e-4f, 1e-5f, "FFT 1D zero-padded");
}

// ============================================================================
// 2D / N-D FFT Parity (Phase 3.4)
// ============================================================================

TEST_P(FFTParity, FFT2_Basic) {
    auto input = randn({2, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::fft2(ins[0]);
    }, {input}, device, 1e-4f, 1e-5f, "fft2");
}

TEST_P(FFTParity, IFFT2_Roundtrip) {
    auto input = randn({2, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        auto f = tenzor::fft::fft2(ins[0]);
        return tenzor::fft::ifft2(f);
    }, {input}, device, 1e-4f, 1e-5f, "fft2/ifft2 roundtrip");
}

TEST_P(FFTParity, RFFT2_Basic) {
    auto input = randn({2, 16, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::rfft2(ins[0]);
    }, {input}, device, 1e-4f, 1e-5f, "rfft2");
}

TEST_P(FFTParity, FFTN_Basic) {
    auto input = randn({2, 8, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::fftn(ins[0]);
    }, {input}, device, 1e-4f, 1e-5f, "fftn");
}

TEST_P(FFTParity, IFFTN_Roundtrip) {
    auto input = randn({2, 8, 8, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        auto f = tenzor::fft::fftn(ins[0]);
        return tenzor::fft::ifftn(f);
    }, {input}, device, 1e-4f, 1e-5f, "fftn/ifftn roundtrip");
}

TEST_P(FFTParity, FFTShift) {
    auto input = randn({4, 8}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::fftshift(ins[0]);
    }, {input}, device, 1e-5f, 1e-7f, "fftshift");
}

// ============================================================================
// Signal processing (STFT/ISTFT/DCT)
// ============================================================================

TEST_P(FFTParity, STFT) {
    // Signal length 256, n_fft=64, hop=16
    auto input = randn({1, 256}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::stft(ins[0], /*n_fft=*/64, /*hop=*/16);
    }, {input}, device, 1e-4f, 1e-5f, "stft");
}

TEST_P(FFTParity, ISTFT_Roundtrip) {
    auto input = randn({1, 256}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        auto spec = tenzor::fft::stft(ins[0], 64, 16);
        return tenzor::fft::istft(spec, 64, 16);
    }, {input}, device, 1e-3f, 1e-4f, "stft/istft roundtrip");
}

TEST_P(FFTParity, DCT) {
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::dct(ins[0]);
    }, {input}, device, 1e-4f, 1e-5f, "dct");
}

TEST_P(FFTParity, DCT_IDCT_Roundtrip) {
    auto input = randn({4, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        auto d = tenzor::fft::dct(ins[0]);
        return tenzor::fft::idct(d);
    }, {input}, device, 1e-4f, 1e-5f, "dct/idct roundtrip");
}

TEST_P(FFTParity, MelScale) {
    // Power spectrogram input: (batch=2, freq_bins=65, time=16). A real
    // pipeline would feed |STFT|^2 here; any positive tensor works for
    // numerical parity.
    auto power = tenzor::abs(randn({2, 65, 16}, DType::Float32, Device::cpu()));
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::mel_scale(ins[0], /*n_mels=*/40,
                                       /*f_min=*/0.0, /*f_max=*/0.0,
                                       /*sample_rate=*/16000);
    }, {power}, device, 1e-3f, 1e-5f, "mel_scale");
}

TEST_P(FFTParity, MFCC) {
    // 1 second of 16kHz audio — too large for a parity test; use 4096 samples.
    auto waveform = randn({1, 4096}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& ins) {
        return tenzor::fft::mfcc(ins[0], /*sample_rate=*/16000,
                                  /*n_mfcc=*/20, /*n_mels=*/40,
                                  /*n_fft=*/256, /*hop_length=*/128);
    }, {waveform}, device, 1e-2f, 1e-3f, "mfcc");
}

INSTANTIATE_BACKEND_TESTS(FFTParity);


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
