/**
 * @file test_dct_multidtype.cpp
 * @brief Tests for DCT (Discrete Cosine Transform) and IDCT operations
 *
 * Verifies:
 *  - DCT-II of known sequence matches analytical values
 *  - Round-trip: idct(dct(x)) ~= x for all DCT types
 *  - Different DCT types (1-4)
 *  - Ortho normalization produces orthonormal transforms
 */

#include <gtest/gtest.h>
#include "backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/fft.hpp"
#include <cmath>

using namespace tenzor;
using namespace tenzor::testing;

class DCTTest : public BackendTest {};

// DCT-II of a simple known sequence: [1, 2, 3, 4]
// Analytical DCT-II (unnormalized):
//   X[k] = 2 * sum_{n=0}^{N-1} x[n] * cos(pi*(2n+1)*k / (2N))
//   X[0] = 2*(1+2+3+4) = 20
//   X[1] = 2*(1*cos(pi/8) + 2*cos(3pi/8) + 3*cos(5pi/8) + 4*cos(7pi/8))
//   X[2] = 2*(1*cos(pi/4) + 2*cos(3pi/4) + 3*cos(5pi/4) + 4*cos(7pi/4))
//   X[3] = 2*(1*cos(3pi/8) + 2*cos(9pi/8) + 3*cos(15pi/8) + 4*cos(21pi/8))
TEST_P(DCTTest, DCT2KnownValues) {
    // Create input [1, 2, 3, 4] using arange + 1
    auto input = tenzor::add(tenzor::arange(0.0f, 4.0f, 1.0f, DType::Float32, device),
                             tenzor::full({1}, 1.0f, DType::Float32, device));

    auto result = fft::dct(input, 2, std::nullopt, -1, "backward");
    auto result_cpu = result.to(Device::cpu());
    const float* data = result_cpu.data<float>();

    const double pi = 3.14159265358979323846;
    const int N = 4;
    std::vector<float> x = {1.0f, 2.0f, 3.0f, 4.0f};

    // Compute expected DCT-II analytically
    for (int k = 0; k < N; ++k) {
        double expected = 0.0;
        for (int n = 0; n < N; ++n) {
            expected += x[n] * std::cos(pi * (2 * n + 1) * k / (2.0 * N));
        }
        expected *= 2.0;
        EXPECT_NEAR(data[k], static_cast<float>(expected), 1e-4f)
            << "DCT-II mismatch at k=" << k;
    }
}

// Round-trip test: idct(dct(x, type=2), type=2) ~= x
TEST_P(DCTTest, RoundTripDCT2) {
    auto input = tenzor::rand({16}, DType::Float32, device);
    auto dct_result = fft::dct(input, 2, std::nullopt, -1, "ortho");
    auto reconstructed = fft::idct(dct_result, 2, std::nullopt, -1, "ortho");

    expectTensorNear(input, reconstructed, 1e-4f);
}

// Round-trip test for DCT type 1
TEST_P(DCTTest, RoundTripDCT1) {
    auto input = tenzor::rand({8}, DType::Float32, device);
    auto dct_result = fft::dct(input, 1, std::nullopt, -1, "ortho");
    auto reconstructed = fft::idct(dct_result, 1, std::nullopt, -1, "ortho");

    expectTensorNear(input, reconstructed, 1e-4f);
}

// Round-trip test for DCT type 3
TEST_P(DCTTest, RoundTripDCT3) {
    auto input = tenzor::rand({16}, DType::Float32, device);
    auto dct_result = fft::dct(input, 3, std::nullopt, -1, "ortho");
    auto reconstructed = fft::idct(dct_result, 3, std::nullopt, -1, "ortho");

    expectTensorNear(input, reconstructed, 1e-4f);
}

// Round-trip test for DCT type 4
TEST_P(DCTTest, RoundTripDCT4) {
    auto input = tenzor::rand({16}, DType::Float32, device);
    auto dct_result = fft::dct(input, 4, std::nullopt, -1, "ortho");
    auto reconstructed = fft::idct(dct_result, 4, std::nullopt, -1, "ortho");

    expectTensorNear(input, reconstructed, 1e-4f);
}

// DCT along a non-last dimension
TEST_P(DCTTest, DCT2AlongDim0) {
    auto input = tenzor::rand({8, 4}, DType::Float32, device);
    auto dct_result = fft::dct(input, 2, std::nullopt, 0, "ortho");
    auto reconstructed = fft::idct(dct_result, 2, std::nullopt, 0, "ortho");

    expectTensorNear(input, reconstructed, 1e-4f);
}

// Ortho normalization: DCT-II with ortho norm should be energy-preserving
// (Parseval's theorem: ||x||^2 == ||DCT(x)||^2 for ortho norm)
TEST_P(DCTTest, OrthoNormEnergyPreserving) {
    auto input = tenzor::rand({32}, DType::Float32, device);
    auto dct_result = fft::dct(input, 2, std::nullopt, -1, "ortho");

    // Compute energies on CPU
    auto input_cpu = input.to(Device::cpu());
    auto result_cpu = dct_result.to(Device::cpu());
    const float* in_data = input_cpu.data<float>();
    const float* out_data = result_cpu.data<float>();

    double input_energy = 0.0;
    double output_energy = 0.0;
    for (int64_t i = 0; i < 32; ++i) {
        input_energy += in_data[i] * in_data[i];
        output_energy += out_data[i] * out_data[i];
    }

    EXPECT_NEAR(input_energy, output_energy, 1e-3)
        << "Ortho DCT-II should preserve energy (Parseval's theorem)";
}

// Batched DCT: 2D tensor, transform along last dim
TEST_P(DCTTest, BatchedDCT2) {
    auto input = tenzor::rand({4, 16}, DType::Float32, device);
    auto dct_result = fft::dct(input, 2, std::nullopt, -1, "ortho");
    auto reconstructed = fft::idct(dct_result, 2, std::nullopt, -1, "ortho");

    expectTensorNear(input, reconstructed, 1e-4f);
}

// Test invalid type throws
TEST_P(DCTTest, InvalidTypeThrows) {
    auto input = tenzor::rand({8}, DType::Float32, device);
    EXPECT_THROW(fft::dct(input, 0), std::runtime_error);
    EXPECT_THROW(fft::dct(input, 5), std::runtime_error);
}

INSTANTIATE_BACKEND_TESTS(DCTTest);
