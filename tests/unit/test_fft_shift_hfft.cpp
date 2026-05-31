// Tests for fftshift / ifftshift / hfft / ihfft — added in the P2 pass.
// Cross-backend: thin compositions over existing roll/rfft/irfft/conj.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

#include "../backend_test_fixture.hpp"

namespace tenzor {
namespace {

using tenzor::testing::BackendTest;

class FFTShiftTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (IsSkipped()) return;
    }
};

TEST_P(FFTShiftTest, FftshiftEvenLength1D) {
    // Input: [0, 1, 2, 3] -> fftshift -> [2, 3, 0, 1]
    auto x_cpu = zeros({4}, DType::Float32, Device::cpu());
    for (int i = 0; i < 4; ++i) x_cpu.data<float>()[i] = static_cast<float>(i);
    auto x = x_cpu.to(device);
    auto out = fft::fftshift(x);
    auto out_cpu = out.cpu();
    const float* o = out_cpu.data<float>();
    EXPECT_EQ(o[0], 2.0f);
    EXPECT_EQ(o[1], 3.0f);
    EXPECT_EQ(o[2], 0.0f);
    EXPECT_EQ(o[3], 1.0f);
}

TEST_P(FFTShiftTest, FftshiftOddLength1D) {
    // Input: [0, 1, 2, 3, 4] -> fftshift -> [3, 4, 0, 1, 2]
    auto x_cpu = zeros({5}, DType::Float32, Device::cpu());
    for (int i = 0; i < 5; ++i) x_cpu.data<float>()[i] = static_cast<float>(i);
    auto x = x_cpu.to(device);
    auto out = fft::fftshift(x);
    auto out_cpu = out.cpu();
    const float* o = out_cpu.data<float>();
    EXPECT_EQ(o[0], 3.0f);
    EXPECT_EQ(o[1], 4.0f);
    EXPECT_EQ(o[2], 0.0f);
    EXPECT_EQ(o[3], 1.0f);
    EXPECT_EQ(o[4], 2.0f);
}

TEST_P(FFTShiftTest, IfftshiftIsInverseOfFftshift) {
    // Both even and odd sizes should round-trip.
    for (int64_t N : {4, 5, 8, 9}) {
        auto x_cpu = zeros({N}, DType::Float32, Device::cpu());
        for (int i = 0; i < N; ++i) {
            x_cpu.data<float>()[i] = static_cast<float>(i);
        }
        auto x = x_cpu.to(device);
        auto shifted = fft::fftshift(x);
        auto restored = fft::ifftshift(shifted);
        auto restored_cpu = restored.cpu();
        const float* r = restored_cpu.data<float>();
        for (int i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(r[i], static_cast<float>(i))
                << "N=" << N << " i=" << i;
        }
    }
}

TEST_P(FFTShiftTest, Fftshift2D) {
    // 2x2 input: [[0,1],[2,3]]. fftshift over all dims ->
    //   shift along dim 0 by 1: [[2,3],[0,1]]
    //   shift along dim 1 by 1: [[3,2],[1,0]]
    auto x_cpu = zeros({2, 2}, DType::Float32, Device::cpu());
    x_cpu.data<float>()[0] = 0.0f;
    x_cpu.data<float>()[1] = 1.0f;
    x_cpu.data<float>()[2] = 2.0f;
    x_cpu.data<float>()[3] = 3.0f;
    auto x = x_cpu.to(device);
    auto out = fft::fftshift(x);
    auto out_cpu = out.cpu();
    const float* o = out_cpu.data<float>();
    EXPECT_EQ(o[0], 3.0f);
    EXPECT_EQ(o[1], 2.0f);
    EXPECT_EQ(o[2], 1.0f);
    EXPECT_EQ(o[3], 0.0f);
}

TEST_P(FFTShiftTest, IhfftHfftRoundtrip) {
    // ihfft(real) -> Hermitian spectrum; hfft(that) -> real, should recover input.
    auto x_cpu = zeros({8}, DType::Float32, Device::cpu());
    for (int i = 0; i < 8; ++i) {
        x_cpu.data<float>()[i] = std::sin(0.5f * static_cast<float>(i));
    }
    auto x = x_cpu.to(device);
    auto spec = fft::ihfft(x);
    auto recovered = fft::hfft(spec, /*n=*/8);
    auto recovered_cpu = recovered.cpu();
    const float* rec = recovered_cpu.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(rec[i], x_cpu.data<float>()[i], 1e-5f)
            << "i=" << i;
    }
}

INSTANTIATE_BACKEND_TESTS(FFTShiftTest);

} // namespace
} // namespace tenzor
