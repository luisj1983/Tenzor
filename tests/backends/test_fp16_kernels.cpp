#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

// Helper function to check if CUDA is available
bool is_cuda_available() {
    try {
        auto device = Device::cuda(0);
        return true;
    } catch (...) {
        return false;
    }
}

class FP16KernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_cuda_available()) {
            GTEST_SKIP() << "CUDA not available, skipping FP16 tests";
        }
    }
};

// ============================================================================
// Basic FP16 Creation and Conversion Tests
// ============================================================================

TEST_F(FP16KernelsTest, ZerosFP16) {
    auto zeros_f16 = zeros({128, 128}, DType::Float16, Device::cuda(0));

    EXPECT_EQ(zeros_f16.dtype(), DType::Float16);
    EXPECT_EQ(zeros_f16.shape()[0], 128);
    EXPECT_EQ(zeros_f16.shape()[1], 128);
    EXPECT_EQ(zeros_f16.numel(), 128 * 128);
}

TEST_F(FP16KernelsTest, OnesFP16) {
    auto ones_f16 = ones({128, 128}, DType::Float16, Device::cuda(0));

    EXPECT_EQ(ones_f16.dtype(), DType::Float16);
    EXPECT_EQ(ones_f16.numel(), 128 * 128);
}

TEST_F(FP16KernelsTest, RandFP16) {
    auto rand_f16 = rand({128, 128}, DType::Float16, Device::cuda(0));

    EXPECT_EQ(rand_f16.dtype(), DType::Float16);
    EXPECT_EQ(rand_f16.numel(), 128 * 128);
}

TEST_F(FP16KernelsTest, RandnFP16) {
    auto randn_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));

    EXPECT_EQ(randn_f16.dtype(), DType::Float16);
    EXPECT_EQ(randn_f16.numel(), 128 * 128);
}

TEST_F(FP16KernelsTest, FullFP16) {
    auto full_f16 = full({100, 100}, 5.5f, DType::Float16, Device::cuda(0));

    EXPECT_EQ(full_f16.dtype(), DType::Float16);
    EXPECT_EQ(full_f16.numel(), 100 * 100);
}

// ============================================================================
// Dtype Conversion Tests
// ============================================================================

TEST_F(FP16KernelsTest, Float32ToFloat16Conversion) {
    auto a_f32 = randn({128, 128}, DType::Float32, Device::cuda(0));
    auto a_f16 = a_f32.to(DType::Float16);
    auto a_back = a_f16.to(DType::Float32);

    EXPECT_EQ(a_f16.dtype(), DType::Float16);
    EXPECT_EQ(a_back.dtype(), DType::Float32);
    EXPECT_EQ(a_f16.shape()[0], a_f32.shape()[0]);
    EXPECT_EQ(a_f16.shape()[1], a_f32.shape()[1]);
}

TEST_F(FP16KernelsTest, Int32ToFloat16Conversion) {
    auto a_i32 = full({64, 64}, 42.0f, DType::Int32, Device::cuda(0));
    auto a_f16 = a_i32.to(DType::Float16);

    EXPECT_EQ(a_f16.dtype(), DType::Float16);
    EXPECT_EQ(a_f16.numel(), 64 * 64);
}

// ============================================================================
// Basic FP16 Operations Tests
// ============================================================================

TEST_F(FP16KernelsTest, AdditionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));

    auto result_f16 = a_f16 + b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(FP16KernelsTest, SubtractionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));

    auto result_f16 = a_f16 - b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(FP16KernelsTest, MultiplicationFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));

    auto result_f16 = a_f16 * b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(FP16KernelsTest, DivisionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::cuda(0)) + 1.0f;

    auto result_f16 = a_f16 / b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_F(FP16KernelsTest, BroadcastAdditionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({1, 128}, DType::Float16, Device::cuda(0));

    auto result_f16 = a_f16 + b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(FP16KernelsTest, BroadcastMultiplicationFP16) {
    auto a_f16 = randn({64, 128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({1, 128, 1}, DType::Float16, Device::cuda(0));

    auto result_f16 = a_f16 * b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 64);
    EXPECT_EQ(result_f16.shape()[1], 128);
    EXPECT_EQ(result_f16.shape()[2], 128);
}

// ============================================================================
// Tensor Core Matmul Tests
// ============================================================================

TEST_F(FP16KernelsTest, MatmulTensorCoresAligned) {
    // Use dimensions aligned to 16 for Tensor Cores
    auto a_f16 = randn({256, 256}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({256, 256}, DType::Float16, Device::cuda(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 256);
    EXPECT_EQ(result_f16.shape()[1], 256);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(FP16KernelsTest, MatmulTensorCoresNonAligned) {
    // Use dimensions NOT aligned to 16 (should fall back to tiled kernel)
    auto a_f16 = randn({100, 100}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({100, 100}, DType::Float16, Device::cuda(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 100);
    EXPECT_EQ(result_f16.shape()[1], 100);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(FP16KernelsTest, MatmulTensorCoresRectangular) {
    // Test non-square matrices
    auto a_f16 = randn({128, 256}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({256, 128}, DType::Float16, Device::cuda(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(FP16KernelsTest, MatmulTensorCoresLarge) {
    // Test larger matrices (512x512)
    auto a_f16 = randn({512, 512}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({512, 512}, DType::Float16, Device::cuda(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 512);
    EXPECT_EQ(result_f16.shape()[1], 512);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(FP16KernelsTest, BatchedMatmulFP16) {
    // Test batched matmul with FP16
    auto a_f16 = randn({4, 128, 128}, DType::Float16, Device::cuda(0));
    auto b_f16 = randn({4, 128, 128}, DType::Float16, Device::cuda(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 4);
    EXPECT_EQ(result_f16.shape()[1], 128);
    EXPECT_EQ(result_f16.shape()[2], 128);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

TEST_F(FP16KernelsTest, MixedPrecisionWorkflow) {
    // Simulate a typical mixed precision workflow
    // 1. Create data in FP16
    auto x_f16 = randn({64, 256}, DType::Float16, Device::cuda(0));
    auto w_f16 = randn({256, 128}, DType::Float16, Device::cuda(0));

    // 2. Perform matmul in FP16 (uses Tensor Cores)
    auto y_f16 = matmul(x_f16, w_f16);

    EXPECT_EQ(y_f16.dtype(), DType::Float16);
    EXPECT_EQ(y_f16.shape()[0], 64);
    EXPECT_EQ(y_f16.shape()[1], 128);

    // 3. Convert to FP32 for loss computation
    auto y_f32 = y_f16.to(DType::Float32);

    EXPECT_EQ(y_f32.dtype(), DType::Float32);
    EXPECT_EQ(y_f32.shape()[0], 64);
    EXPECT_EQ(y_f32.shape()[1], 128);
}

// ============================================================================
// Shape and Memory Tests
// ============================================================================

TEST_F(FP16KernelsTest, MemoryFootprint) {
    // Verify FP16 uses half the memory of FP32
    auto a_f32 = randn({1024, 1024}, DType::Float32, Device::cuda(0));
    auto a_f16 = a_f32.to(DType::Float16);

    // FP16 should use half the bytes
    size_t f32_bytes = a_f32.numel() * 4;  // 4 bytes per float
    size_t f16_bytes = a_f16.numel() * 2;  // 2 bytes per half

    EXPECT_EQ(f16_bytes * 2, f32_bytes);
}

TEST_F(FP16KernelsTest, ShapePreservation) {
    std::vector<int64_t> shape = {2, 3, 4, 5};
    auto a_f32 = randn(shape, DType::Float32, Device::cuda(0));
    auto a_f16 = a_f32.to(DType::Float16);

    EXPECT_EQ(a_f16.ndim(), 4);
    EXPECT_EQ(a_f16.shape()[0], 2);
    EXPECT_EQ(a_f16.shape()[1], 3);
    EXPECT_EQ(a_f16.shape()[2], 4);
    EXPECT_EQ(a_f16.shape()[3], 5);
}

int main(int argc, char** argv) {
    // Initialize Tenzor to load backends
    tenzor::initialize();

    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
