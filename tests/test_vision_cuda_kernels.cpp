/**
 * @file test_vision_cuda_kernels.cpp
 * @brief Quick verification test for vision CUDA kernels
 */

#include <gtest/gtest.h>
#include "tenzor/ops/vision.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/core/device.hpp"
#include <iostream>

using namespace tenzor;
using namespace tenzor::ops;

class VisionCUDAKernelTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Check if CUDA is available
        try {
            cuda_available_ = Device::cuda(0).is_available();
        } catch (...) {
            cuda_available_ = false;
        }
    }

    bool cuda_available_ = false;
};

TEST_F(VisionCUDAKernelTest, UnfoldCPU) {
    // Test CPU implementation
    auto input = ones({1, 3, 8, 8}, DType::Float32, Device::cpu());
    auto output = unfold(input, 3, 1, 0, 1);

    // Output shape should be (1, 3*3*3=27, 6*6=36)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 27);  // 3 channels * 3x3 kernel
    EXPECT_EQ(shape[2], 36);  // 6x6 output blocks
}

TEST_F(VisionCUDAKernelTest, UnfoldCUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Test CUDA implementation
    auto input = ones({1, 3, 8, 8}, DType::Float32, Device::cuda(0));
    auto output = unfold(input, 3, 1, 0, 1);

    // Output shape should be (1, 27, 36)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 27);
    EXPECT_EQ(shape[2], 36);

    // Verify output is on CUDA device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
}

TEST_F(VisionCUDAKernelTest, FoldCPU) {
    // Test CPU implementation
    auto input = ones({1, 27, 36}, DType::Float32, Device::cpu());
    auto output = fold(input, {8, 8}, 3, 1, 0, 1);

    // Output shape should be (1, 3, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);
}

TEST_F(VisionCUDAKernelTest, FoldCUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Test CUDA implementation
    auto input = ones({1, 27, 36}, DType::Float32, Device::cuda(0));
    auto output = fold(input, {8, 8}, 3, 1, 0, 1);

    // Output shape should be (1, 3, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);

    // Verify output is on CUDA device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
}

TEST_F(VisionCUDAKernelTest, InterpolateNearestCPU) {
    // Test CPU implementation
    auto input = ones({1, 3, 4, 4}, DType::Float32, Device::cpu());
    auto output = interpolate(input, {8, 8}, "nearest", false);

    // Output shape should be (1, 3, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);
}

TEST_F(VisionCUDAKernelTest, InterpolateNearestCUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Test CUDA implementation
    auto input = ones({1, 3, 4, 4}, DType::Float32, Device::cuda(0));
    auto output = interpolate(input, {8, 8}, "nearest", false);

    // Output shape should be (1, 3, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 3);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);

    // Verify output is on CUDA device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
}

TEST_F(VisionCUDAKernelTest, InterpolateBilinearCPU) {
    // Test CPU implementation
    auto input = ones({1, 2, 4, 4}, DType::Float32, Device::cpu());
    auto output = interpolate(input, {8, 8}, "bilinear", false);

    // Output shape should be (1, 2, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);
}

TEST_F(VisionCUDAKernelTest, InterpolateBilinearCUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Test CUDA implementation
    auto input = ones({1, 2, 4, 4}, DType::Float32, Device::cuda(0));
    auto output = interpolate(input, {8, 8}, "bilinear", false);

    // Output shape should be (1, 2, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 2);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);

    // Verify output is on CUDA device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
}

TEST_F(VisionCUDAKernelTest, InterpolateBicubicCUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Test CUDA bicubic implementation
    auto input = ones({1, 1, 4, 4}, DType::Float32, Device::cuda(0));
    auto output = interpolate(input, {8, 8}, "bicubic", false);

    // Output shape should be (1, 1, 8, 8)
    auto shape = output.shape();
    EXPECT_EQ(shape[0], 1);
    EXPECT_EQ(shape[1], 1);
    EXPECT_EQ(shape[2], 8);
    EXPECT_EQ(shape[3], 8);

    // Verify output is on CUDA device
    EXPECT_EQ(output.device().type, Device::Type::CUDA);
}

TEST_F(VisionCUDAKernelTest, UnfoldFoldRoundTripCPU) {
    // Test that unfold followed by fold recovers original for non-overlapping patches
    auto input = ones({1, 2, 8, 8}, DType::Float32, Device::cpu());
    auto unfolded = unfold(input, 2, 2, 0, 1);  // Non-overlapping 2x2 patches
    auto folded = fold(unfolded, {8, 8}, 2, 2, 0, 1);

    // Shape should match
    EXPECT_EQ(input.shape(), folded.shape());
}

TEST_F(VisionCUDAKernelTest, UnfoldFoldRoundTripCUDA) {
    if (!cuda_available_) {
        GTEST_SKIP() << "CUDA not available, skipping test";
    }

    // Test that unfold followed by fold recovers original for non-overlapping patches
    auto input = ones({1, 2, 8, 8}, DType::Float32, Device::cuda(0));
    auto unfolded = unfold(input, 2, 2, 0, 1);  // Non-overlapping 2x2 patches
    auto folded = fold(unfolded, {8, 8}, 2, 2, 0, 1);

    // Shape should match
    EXPECT_EQ(input.shape(), folded.shape());

    // Both should be on CUDA
    EXPECT_EQ(unfolded.device().type, Device::Type::CUDA);
    EXPECT_EQ(folded.device().type, Device::Type::CUDA);
}
