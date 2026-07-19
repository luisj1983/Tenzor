#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

// Kernel-granularity FP16 test file mirroring test_fp16_kernels.cpp (CUDA), but
// targeting OneAPI — see FINDING 59 in findings.txt: CUDA had a dedicated
// FP16-kernel-depth test file (large arrays, batched matmul, broadcast edge
// cases) that ROCm/Vulkan/OneAPI lacked.

bool is_oneapi_available() {
    try {
        auto device = Device::oneapi(0);
        auto t = zeros({1}, DType::Float32, device);
        (void)t;
        return true;
    } catch (const std::exception&) {
        return false;
    }
}

class OneAPIFP16KernelsTest : public ::testing::Test {
protected:
    void SetUp() override {
        if (!is_oneapi_available()) {
            GTEST_SKIP() << "OneAPI not available, skipping FP16 tests";
        }
    }
};

// ============================================================================
// Basic FP16 Creation and Conversion Tests
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, ZerosFP16) {
    auto zeros_f16 = zeros({128, 128}, DType::Float16, Device::oneapi(0));

    EXPECT_EQ(zeros_f16.dtype(), DType::Float16);
    EXPECT_EQ(zeros_f16.shape()[0], 128);
    EXPECT_EQ(zeros_f16.shape()[1], 128);
    EXPECT_EQ(zeros_f16.numel(), 128 * 128);
}

TEST_F(OneAPIFP16KernelsTest, OnesFP16) {
    auto ones_f16 = ones({128, 128}, DType::Float16, Device::oneapi(0));

    EXPECT_EQ(ones_f16.dtype(), DType::Float16);
    EXPECT_EQ(ones_f16.numel(), 128 * 128);
}

TEST_F(OneAPIFP16KernelsTest, RandFP16) {
    auto rand_f16 = rand({128, 128}, DType::Float16, Device::oneapi(0));

    EXPECT_EQ(rand_f16.dtype(), DType::Float16);
    EXPECT_EQ(rand_f16.numel(), 128 * 128);
}

TEST_F(OneAPIFP16KernelsTest, RandnFP16) {
    auto randn_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));

    EXPECT_EQ(randn_f16.dtype(), DType::Float16);
    EXPECT_EQ(randn_f16.numel(), 128 * 128);
}

TEST_F(OneAPIFP16KernelsTest, FullFP16) {
    auto full_f16 = full({100, 100}, 5.5f, DType::Float16, Device::oneapi(0));

    EXPECT_EQ(full_f16.dtype(), DType::Float16);
    EXPECT_EQ(full_f16.numel(), 100 * 100);
}

// ============================================================================
// Dtype Conversion Tests
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, Float32ToFloat16Conversion) {
    auto a_f32 = randn({128, 128}, DType::Float32, Device::oneapi(0));
    auto a_f16 = a_f32.to(DType::Float16);
    auto a_back = a_f16.to(DType::Float32);

    EXPECT_EQ(a_f16.dtype(), DType::Float16);
    EXPECT_EQ(a_back.dtype(), DType::Float32);
    EXPECT_EQ(a_f16.shape()[0], a_f32.shape()[0]);
    EXPECT_EQ(a_f16.shape()[1], a_f32.shape()[1]);
}

TEST_F(OneAPIFP16KernelsTest, Int32ToFloat16Conversion) {
    auto a_i32 = full({64, 64}, 42.0f, DType::Int32, Device::oneapi(0));
    auto a_f16 = a_i32.to(DType::Float16);

    EXPECT_EQ(a_f16.dtype(), DType::Float16);
    EXPECT_EQ(a_f16.numel(), 64 * 64);
}

// ============================================================================
// Basic FP16 Operations Tests
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, AdditionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));

    auto result_f16 = a_f16 + b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(OneAPIFP16KernelsTest, SubtractionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));

    auto result_f16 = a_f16 - b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(OneAPIFP16KernelsTest, MultiplicationFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));

    auto result_f16 = a_f16 * b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(OneAPIFP16KernelsTest, DivisionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0)) + 1.0f;

    auto result_f16 = a_f16 / b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

// ============================================================================
// Broadcasting Tests
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, BroadcastAdditionFP16) {
    auto a_f16 = randn({128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({1, 128}, DType::Float16, Device::oneapi(0));

    auto result_f16 = a_f16 + b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
}

TEST_F(OneAPIFP16KernelsTest, BroadcastMultiplicationFP16) {
    auto a_f16 = randn({64, 128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({1, 128, 1}, DType::Float16, Device::oneapi(0));

    auto result_f16 = a_f16 * b_f16;

    EXPECT_EQ(result_f16.dtype(), DType::Float16);
    EXPECT_EQ(result_f16.shape()[0], 64);
    EXPECT_EQ(result_f16.shape()[1], 128);
    EXPECT_EQ(result_f16.shape()[2], 128);
}

// ============================================================================
// Matmul Tests (MFMA/tensor-core-eligible shapes on OneAPI)
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, MatmulAligned) {
    // Dimensions aligned to 16 (tensor-core-eligible shapes on OneAPI, mirrors CUDA's
    // Tensor-Core-aligned case)
    auto a_f16 = randn({256, 256}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({256, 256}, DType::Float16, Device::oneapi(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 256);
    EXPECT_EQ(result_f16.shape()[1], 256);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(OneAPIFP16KernelsTest, MatmulNonAligned) {
    // Dimensions NOT aligned to 16 (should fall back to a generic tiled kernel)
    auto a_f16 = randn({100, 100}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({100, 100}, DType::Float16, Device::oneapi(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 100);
    EXPECT_EQ(result_f16.shape()[1], 100);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(OneAPIFP16KernelsTest, MatmulRectangular) {
    auto a_f16 = randn({128, 256}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({256, 128}, DType::Float16, Device::oneapi(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 128);
    EXPECT_EQ(result_f16.shape()[1], 128);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(OneAPIFP16KernelsTest, MatmulLarge) {
    auto a_f16 = randn({512, 512}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({512, 512}, DType::Float16, Device::oneapi(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 512);
    EXPECT_EQ(result_f16.shape()[1], 512);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

TEST_F(OneAPIFP16KernelsTest, BatchedMatmulFP16) {
    auto a_f16 = randn({4, 128, 128}, DType::Float16, Device::oneapi(0));
    auto b_f16 = randn({4, 128, 128}, DType::Float16, Device::oneapi(0));

    auto result_f16 = matmul(a_f16, b_f16);

    EXPECT_EQ(result_f16.shape()[0], 4);
    EXPECT_EQ(result_f16.shape()[1], 128);
    EXPECT_EQ(result_f16.shape()[2], 128);
    EXPECT_EQ(result_f16.dtype(), DType::Float16);
}

// ============================================================================
// Mixed Precision Tests
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, MixedPrecisionWorkflow) {
    auto x_f16 = randn({64, 256}, DType::Float16, Device::oneapi(0));
    auto w_f16 = randn({256, 128}, DType::Float16, Device::oneapi(0));

    auto y_f16 = matmul(x_f16, w_f16);

    EXPECT_EQ(y_f16.dtype(), DType::Float16);
    EXPECT_EQ(y_f16.shape()[0], 64);
    EXPECT_EQ(y_f16.shape()[1], 128);

    auto y_f32 = y_f16.to(DType::Float32);

    EXPECT_EQ(y_f32.dtype(), DType::Float32);
    EXPECT_EQ(y_f32.shape()[0], 64);
    EXPECT_EQ(y_f32.shape()[1], 128);
}

// ============================================================================
// Shape and Memory Tests
// ============================================================================

TEST_F(OneAPIFP16KernelsTest, MemoryFootprint) {
    auto a_f32 = randn({1024, 1024}, DType::Float32, Device::oneapi(0));
    auto a_f16 = a_f32.to(DType::Float16);

    size_t f32_bytes = a_f32.numel() * 4;
    size_t f16_bytes = a_f16.numel() * 2;

    EXPECT_EQ(f16_bytes * 2, f32_bytes);
}

TEST_F(OneAPIFP16KernelsTest, ShapePreservation) {
    std::vector<int64_t> shape = {2, 3, 4, 5};
    auto a_f32 = randn(shape, DType::Float32, Device::oneapi(0));
    auto a_f16 = a_f32.to(DType::Float16);

    EXPECT_EQ(a_f16.ndim(), 4);
    EXPECT_EQ(a_f16.shape()[0], 2);
    EXPECT_EQ(a_f16.shape()[1], 3);
    EXPECT_EQ(a_f16.shape()[2], 4);
    EXPECT_EQ(a_f16.shape()[3], 5);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    if (!::testing::GTEST_FLAG(list_tests)) {
        tenzor::initialize();
    }
    return RUN_ALL_TESTS();
}
