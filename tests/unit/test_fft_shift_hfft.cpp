// Tests for fftshift / ifftshift / hfft / ihfft — added in the P2 pass.
// CPU-only; these are thin compositions over existing roll/rfft/irfft/conj.

#include <gtest/gtest.h>

#include <cmath>
#include <cstring>

#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>

namespace tenzor {
namespace {

class FFTShiftTest : public ::testing::Test {
protected:
    void SetUp() override { tenzor::initialize(); }
};

TEST_F(FFTShiftTest, FftshiftEvenLength1D) {
    // Input: [0, 1, 2, 3] -> fftshift -> [2, 3, 0, 1]
    auto x = zeros({4}, DType::Float32, Device::cpu());
    for (int i = 0; i < 4; ++i) x.data<float>()[i] = static_cast<float>(i);
    auto out = fft::fftshift(x);
    const float* o = out.data<float>();
    EXPECT_EQ(o[0], 2.0f);
    EXPECT_EQ(o[1], 3.0f);
    EXPECT_EQ(o[2], 0.0f);
    EXPECT_EQ(o[3], 1.0f);
}

TEST_F(FFTShiftTest, FftshiftOddLength1D) {
    // Input: [0, 1, 2, 3, 4] -> fftshift -> [3, 4, 0, 1, 2]
    auto x = zeros({5}, DType::Float32, Device::cpu());
    for (int i = 0; i < 5; ++i) x.data<float>()[i] = static_cast<float>(i);
    auto out = fft::fftshift(x);
    const float* o = out.data<float>();
    EXPECT_EQ(o[0], 3.0f);
    EXPECT_EQ(o[1], 4.0f);
    EXPECT_EQ(o[2], 0.0f);
    EXPECT_EQ(o[3], 1.0f);
    EXPECT_EQ(o[4], 2.0f);
}

TEST_F(FFTShiftTest, IfftshiftIsInverseOfFftshift) {
    // Both even and odd sizes should round-trip.
    for (int64_t N : {4, 5, 8, 9}) {
        auto x = zeros({N}, DType::Float32, Device::cpu());
        for (int i = 0; i < N; ++i) {
            x.data<float>()[i] = static_cast<float>(i);
        }
        auto shifted = fft::fftshift(x);
        auto restored = fft::ifftshift(shifted);
        for (int i = 0; i < N; ++i) {
            EXPECT_FLOAT_EQ(restored.data<float>()[i], static_cast<float>(i))
                << "N=" << N << " i=" << i;
        }
    }
}

TEST_F(FFTShiftTest, Fftshift2D) {
    // 2x2 input: [[0,1],[2,3]]. fftshift over all dims ->
    //   shift along dim 0 by 1: [[2,3],[0,1]]
    //   shift along dim 1 by 1: [[3,2],[1,0]]
    auto x = zeros({2, 2}, DType::Float32, Device::cpu());
    x.data<float>()[0] = 0.0f;
    x.data<float>()[1] = 1.0f;
    x.data<float>()[2] = 2.0f;
    x.data<float>()[3] = 3.0f;
    auto out = fft::fftshift(x);
    const float* o = out.data<float>();
    EXPECT_EQ(o[0], 3.0f);
    EXPECT_EQ(o[1], 2.0f);
    EXPECT_EQ(o[2], 1.0f);
    EXPECT_EQ(o[3], 0.0f);
}

TEST_F(FFTShiftTest, IhfftHfftRoundtrip) {
    // ihfft(real) -> Hermitian spectrum; hfft(that) -> real, should recover input.
    auto x = zeros({8}, DType::Float32, Device::cpu());
    for (int i = 0; i < 8; ++i) {
        x.data<float>()[i] = std::sin(0.5f * static_cast<float>(i));
    }
    auto spec = fft::ihfft(x);
    auto recovered = fft::hfft(spec, /*n=*/8);
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(recovered.data<float>()[i], x.data<float>()[i], 1e-5f)
            << "i=" << i;
    }
}

} // namespace
} // namespace tenzor
