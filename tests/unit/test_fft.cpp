#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/ops/fft.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/backend/fast_dispatch.hpp"
#include "tenzor/backend/op_attributes.hpp"
#include "tenzor/ops/op_id.hpp"
#include <cmath>
#include <complex>

using namespace tenzor;
using namespace tenzor::testing;

class FFTTest : public BackendTest {};

// ============================================================================
// Basic roundtrip: ifft(fft(x)) ≈ x
// ============================================================================

TEST_P(FFTTest, FFTRoundtrip1D) {
    auto x = ones({8}, DType::Float32, device);
    auto X = fft::fft(x, std::nullopt, -1, "backward");
    auto y = fft::ifft(X, std::nullopt, -1, "backward");

    auto y_cpu = y.to(Device::cpu());
    auto* yp = y_cpu.data<std::complex<float>>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(yp[i].real(), 1.0f, 1e-4f) << "Real part mismatch at index " << i;
        EXPECT_NEAR(yp[i].imag(), 0.0f, 1e-4f) << "Imag part mismatch at index " << i;
    }
}

TEST_P(FFTTest, FFTRoundtripOrtho) {
    auto x = ones({16}, DType::Float32, device);
    auto X = fft::fft(x, std::nullopt, -1, "ortho");
    auto y = fft::ifft(X, std::nullopt, -1, "ortho");

    auto y_cpu = y.to(Device::cpu());
    auto* yp = y_cpu.data<std::complex<float>>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(yp[i].real(), 1.0f, 1e-4f);
        EXPECT_NEAR(yp[i].imag(), 0.0f, 1e-4f);
    }
}

// ============================================================================
// Known signal: DC (all ones) -> spike at k=0
// ============================================================================

TEST_P(FFTTest, DCSignal) {
    // fft of N ones with "backward" norm: X[0] = N, X[k!=0] = 0
    auto x = ones({4}, DType::Float32, device);
    auto X = fft::fft(x, std::nullopt, -1, "backward");

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    EXPECT_NEAR(Xp[0].real(), 4.0f, 1e-4f);
    EXPECT_NEAR(Xp[0].imag(), 0.0f, 1e-4f);
    for (int k = 1; k < 4; ++k) {
        EXPECT_NEAR(Xp[k].real(), 0.0f, 1e-4f) << "k=" << k;
        EXPECT_NEAR(Xp[k].imag(), 0.0f, 1e-4f) << "k=" << k;
    }
}

TEST_P(FFTTest, DCSignalForwardNorm) {
    // fft of N ones with "forward" norm: X[0] = 1, X[k!=0] = 0
    auto x = ones({8}, DType::Float32, device);
    auto X = fft::fft(x, std::nullopt, -1, "forward");

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    EXPECT_NEAR(Xp[0].real(), 1.0f, 1e-4f);
    EXPECT_NEAR(Xp[0].imag(), 0.0f, 1e-4f);
    for (int k = 1; k < 8; ++k) {
        EXPECT_NEAR(Xp[k].real(), 0.0f, 1e-4f);
        EXPECT_NEAR(Xp[k].imag(), 0.0f, 1e-4f);
    }
}

// ============================================================================
// Single impulse: x[0]=1, x[n!=0]=0 -> flat spectrum
// ============================================================================

TEST_P(FFTTest, ImpulseSignal) {
    auto x = zeros({4}, DType::Float32, device);
    auto x_cpu = x.to(Device::cpu());
    x_cpu.data<float>()[0] = 1.0f;
    x = x_cpu.to(device);

    auto X = fft::fft(x, std::nullopt, -1, "backward");
    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    // All frequency bins should be 1+0i
    for (int k = 0; k < 4; ++k) {
        EXPECT_NEAR(Xp[k].real(), 1.0f, 1e-4f) << "k=" << k;
        EXPECT_NEAR(Xp[k].imag(), 0.0f, 1e-4f) << "k=" << k;
    }
}

// ============================================================================
// RFFT: real input -> first n/2+1 complex bins
// ============================================================================

TEST_P(FFTTest, RFFTOnes) {
    auto x = ones({8}, DType::Float32, device);
    auto X = fft::rfft(x, std::nullopt, -1, "backward");

    EXPECT_EQ(X.shape()[0], 5);  // n/2 + 1 = 5

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    EXPECT_NEAR(Xp[0].real(), 8.0f, 1e-4f);
    for (int k = 1; k < 5; ++k) {
        EXPECT_NEAR(Xp[k].real(), 0.0f, 1e-4f);
        EXPECT_NEAR(Xp[k].imag(), 0.0f, 1e-4f);
    }
}

// ============================================================================
// IRFFT roundtrip: irfft(rfft(x), n) ≈ x
// ============================================================================

TEST_P(FFTTest, IRFFTRoundtrip) {
    auto x = ones({8}, DType::Float32, device);
    auto X = fft::rfft(x, std::nullopt, -1, "backward");
    auto y = fft::irfft(X, 8, -1, "backward");

    auto y_cpu = y.to(Device::cpu());
    auto* yp = y_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(yp[i], 1.0f, 1e-4f) << "index=" << i;
    }
}

// ============================================================================
// Float64 precision
// ============================================================================

TEST_P(FFTTest, FFTFloat64) {
    auto x = ones({4}, DType::Float64, device);
    auto X = fft::fft(x, std::nullopt, -1, "backward");

    EXPECT_EQ(X.dtype(), DType::Complex128);

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<double>>();
    EXPECT_NEAR(Xp[0].real(), 4.0, 1e-12);
    EXPECT_NEAR(Xp[0].imag(), 0.0, 1e-12);
}

// ============================================================================
// Batched FFT (2D input, FFT along last dim)
// ============================================================================

TEST_P(FFTTest, BatchedFFT) {
    auto x = ones({3, 4}, DType::Float32, device);
    auto X = fft::fft(x, std::nullopt, -1, "backward");

    EXPECT_EQ(X.shape()[0], 3);
    EXPECT_EQ(X.shape()[1], 4);

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    // Each batch: DC = 4, rest = 0
    for (int b = 0; b < 3; ++b) {
        EXPECT_NEAR(Xp[b * 4 + 0].real(), 4.0f, 1e-4f);
        for (int k = 1; k < 4; ++k) {
            EXPECT_NEAR(Xp[b * 4 + k].real(), 0.0f, 1e-4f);
            EXPECT_NEAR(Xp[b * 4 + k].imag(), 0.0f, 1e-4f);
        }
    }
}

// ============================================================================
// FFT with padding (n > input length)
// ============================================================================

TEST_P(FFTTest, FFTWithPadding) {
    auto x = ones({4}, DType::Float32, device);
    // Pad to length 8: x = [1,1,1,1,0,0,0,0]
    auto X = fft::fft(x, 8, -1, "backward");

    EXPECT_EQ(X.shape()[0], 8);

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    // DC = 4 (sum of all ones)
    EXPECT_NEAR(Xp[0].real(), 4.0f, 1e-4f);
    EXPECT_NEAR(Xp[0].imag(), 0.0f, 1e-4f);
}

// ============================================================================
// Validation: n must be positive
// ============================================================================

TEST_P(FFTTest, NegativeNThrows) {
    auto x = ones({4}, DType::Float32, device);
    EXPECT_THROW(fft::fft(x, -1, -1, "backward"), std::runtime_error);
    EXPECT_THROW(fft::ifft(x, -1, -1, "backward"), std::runtime_error);
    EXPECT_THROW(fft::rfft(x, 0, -1, "backward"), std::runtime_error);
    EXPECT_THROW(fft::irfft(x, -1, -1, "backward"), std::runtime_error);
}

// ============================================================================
// Length-1 FFT
// ============================================================================

TEST_P(FFTTest, Length1FFT) {
    auto x = ones({1}, DType::Float32, device);
    auto X = fft::fft(x, std::nullopt, -1, "backward");

    auto X_cpu = X.to(Device::cpu());
    auto* Xp = X_cpu.data<std::complex<float>>();
    EXPECT_NEAR(Xp[0].real(), 1.0f, 1e-4f);
    EXPECT_NEAR(Xp[0].imag(), 0.0f, 1e-4f);
}

// ============================================================================
// CUDA: n<=0 defense-in-depth (kernel-level, bypassing the op layer)
// ============================================================================
// The op-layer (tenzor::fft::fft/ifft/rfft/irfft) already rejects n<=0 in the
// normal call path, but cuda_fft_kernel/cuda_ifft_kernel/cuda_rfft_kernel/
// cuda_irfft_kernel did not independently re-validate it -- a direct
// OpId::FFT/... dispatch with a crafted AttrKey::N (bypassing the op layer
// entirely, e.g. from a hand-built JIT graph) would reach the kernel with
// n<=0 and get a silent degenerate result (0-block no-op launches, or
// cufft_checked_int accepting n==0) instead of a clear error.
TEST(FFTCudaDirectDispatch, ZeroOrNegativeNThrowsAtKernelLevel) {
    using namespace tenzor;
    initialize();
    bool has_cuda = false;
    try { auto t = zeros({1}, DType::Float32, Device::cuda(0)); (void)t; has_cuda = true; }
    catch (...) {}
    if (!has_cuda) GTEST_SKIP();

    auto x = ones({8}, DType::Complex64, Device::cuda(0));
    std::vector<Tensor> ins = {x};

    for (int64_t bad_n : {0, -1, -5}) {
        for (OpId op : {OpId::FFT, OpId::IFFT, OpId::RFFT, OpId::IRFFT}) {
            NewOpAttributes attrs;
            attrs.set(AttrKey::Dim, static_cast<int64_t>(-1));
            attrs.set(AttrKey::N, bad_n);
            EXPECT_THROW(dispatch(op, ins, attrs), std::runtime_error)
                << "OpId " << static_cast<int>(op) << " with n=" << bad_n
                << " did not throw at the kernel level";
        }
    }
}

// ============================================================================
// Instantiate for CPU backend
// ============================================================================
// FFT is only exercised on CPU here. When fft/ifft/rfft/irfft kernels land on
// CUDA (cuFFT), ROCm (rocFFT), Vulkan, or OneAPI (oneMKL), add matching
// INSTANTIATE_TEST_SUITE_P(<Backend>, ...) blocks below — do NOT collapse into
// a single multi-backend instantiation until every op the tests call is
// registered on that backend, or the instantiation will silently skip.

INSTANTIATE_TEST_SUITE_P(
    CPU, FFTTest,
    ::testing::Values("cpu"),
    [](const ::testing::TestParamInfo<std::string>& info) { return info.param; }
);
