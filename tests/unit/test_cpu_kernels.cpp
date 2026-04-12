#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;

// Global test environment that initializes Tenzor before tests
class TenzorTestEnvironment : public ::testing::Environment {
public:
    void SetUp() override {
        tenzor::initialize();
    }
};

// Register the environment
static ::testing::Environment* const tenzor_env =
    ::testing::AddGlobalTestEnvironment(new TenzorTestEnvironment);

// Helper function to compare floating point values with tolerance
template<typename T>
void ExpectNear(T actual, T expected, T tolerance) {
    EXPECT_NEAR(static_cast<double>(actual), static_cast<double>(expected), static_cast<double>(tolerance));
}

//==============================================================================
// Addition Tests
//==============================================================================

TEST(CPUKernelsTest, AddFloat32_Basic) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({2, 3}, DType::Float32);
    auto c = add(a, b);

    // Check shape
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);
    EXPECT_EQ(c.numel(), 6);

    // Check actual values
    auto data = c.data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, AddFloat32_DifferentValues) {
    // Create tensors with specific values
    auto a = ones({3, 2}, DType::Float32);
    auto b = ones({3, 2}, DType::Float32);

    // Modify tensor values
    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);  // [1, 2, 3, 4, 5, 6]
        b_data[i] = static_cast<float>(i * 2);  // [0, 2, 4, 6, 8, 10]
    }

    auto c = add(a, b);
    auto c_data = c.data<float>();

    // Expected: [1, 4, 7, 10, 13, 16]
    for (int i = 0; i < 6; i++) {
        float expected = static_cast<float>((i + 1) + (i * 2));
        EXPECT_FLOAT_EQ(c_data[i], expected) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, AddFloat64_Basic) {
    auto a = ones({2, 2}, DType::Float64);
    auto b = ones({2, 2}, DType::Float64);
    auto c = add(a, b);

    EXPECT_EQ(c.dtype(), DType::Float64);
    auto data = c.data<double>();
    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(data[i], 2.0) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, AddInt32_Basic) {
    auto a = ones({2, 3}, DType::Int32);
    auto b = ones({2, 3}, DType::Int32);
    auto c = add(a, b);

    EXPECT_EQ(c.dtype(), DType::Int32);
    auto data = c.data<int32_t>();
    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(data[i], 2) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, AddFloat32_WithZeros) {
    auto a = zeros({3, 3}, DType::Float32);
    auto b = ones({3, 3}, DType::Float32);
    auto c = add(a, b);

    auto data = c.data<float>();
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, AddFloat32_Negative) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    // Make b negative
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        b_data[i] = -1.0f;
    }

    auto c = add(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 0.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, AddFloat32_LargeValues) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 1e6f;
        b_data[i] = 2e6f;
    }

    auto c = add(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 3e6f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Subtraction Tests
//==============================================================================

TEST(CPUKernelsTest, SubFloat32_Basic) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({2, 3}, DType::Float32);

    auto a_data = a.data<float>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 5.0f;
    }

    auto c = sub(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 4.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, SubFloat64_Basic) {
    auto a = ones({2, 2}, DType::Float64);
    auto b = ones({2, 2}, DType::Float64);

    auto a_data = a.data<double>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 10.0;
    }

    auto c = sub(a, b);
    auto c_data = c.data<double>();

    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(c_data[i], 9.0) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, SubInt32_Basic) {
    auto a = ones({3, 2}, DType::Int32);
    auto b = ones({3, 2}, DType::Int32);

    auto a_data = a.data<int32_t>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 10;
    }

    auto c = sub(a, b);
    auto c_data = c.data<int32_t>();

    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(c_data[i], 9) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, SubFloat32_ResultNegative) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 2.0f;
        b_data[i] = 5.0f;
    }

    auto c = sub(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], -3.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, SubFloat32_WithZeros) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = zeros({2, 3}, DType::Float32);

    auto c = sub(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 1.0f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Multiplication Tests
//==============================================================================

TEST(CPUKernelsTest, MulFloat32_Basic) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({2, 3}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 3.0f;
        b_data[i] = 4.0f;
    }

    auto c = mul(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 12.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MulFloat64_Basic) {
    auto a = ones({2, 2}, DType::Float64);
    auto b = ones({2, 2}, DType::Float64);

    auto a_data = a.data<double>();
    auto b_data = b.data<double>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 2.5;
        b_data[i] = 4.0;
    }

    auto c = mul(a, b);
    auto c_data = c.data<double>();

    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(c_data[i], 10.0) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MulInt32_Basic) {
    auto a = ones({2, 3}, DType::Int32);
    auto b = ones({2, 3}, DType::Int32);

    auto a_data = a.data<int32_t>();
    auto b_data = b.data<int32_t>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 5;
        b_data[i] = 7;
    }

    auto c = mul(a, b);
    auto c_data = c.data<int32_t>();

    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(c_data[i], 35) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MulFloat32_WithZero) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = zeros({2, 3}, DType::Float32);

    auto a_data = a.data<float>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 100.0f;
    }

    auto c = mul(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 0.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MulFloat32_Negative) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = -3.0f;
        b_data[i] = 4.0f;
    }

    auto c = mul(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], -12.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MulFloat32_LargeValues) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 1e3f;
        b_data[i] = 1e3f;
    }

    auto c = mul(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 1e6f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Division Tests
//==============================================================================

TEST(CPUKernelsTest, DivFloat32_Basic) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = ones({2, 3}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 12.0f;
        b_data[i] = 4.0f;
    }

    auto c = div(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 3.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, DivFloat64_Basic) {
    auto a = ones({2, 2}, DType::Float64);
    auto b = ones({2, 2}, DType::Float64);

    auto a_data = a.data<double>();
    auto b_data = b.data<double>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 10.0;
        b_data[i] = 2.0;
    }

    auto c = div(a, b);
    auto c_data = c.data<double>();

    for (int i = 0; i < 4; i++) {
        EXPECT_DOUBLE_EQ(c_data[i], 5.0) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, DivInt32_Basic) {
    auto a = ones({2, 3}, DType::Int32);
    auto b = ones({2, 3}, DType::Int32);

    auto a_data = a.data<int32_t>();
    auto b_data = b.data<int32_t>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = 20;
        b_data[i] = 5;
    }

    auto c = div(a, b);
    auto c_data = c.data<int32_t>();

    for (int i = 0; i < 6; i++) {
        EXPECT_EQ(c_data[i], 4) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, DivFloat32_Fractional) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 1.0f;
        b_data[i] = 3.0f;
    }

    auto c = div(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(c_data[i], 0.33333333f, 1e-6f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, DivFloat32_Negative) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = -10.0f;
        b_data[i] = 2.0f;
    }

    auto c = div(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], -5.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, DivFloat32_SmallValues) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 1e-3f;
        b_data[i] = 1e-2f;
    }

    auto c = div(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(c_data[i], 0.1f, 1e-6f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Matrix Multiplication Tests
//==============================================================================

TEST(CPUKernelsTest, MatMulFloat32_2x3_3x2) {
    // Create A: 2x3 matrix [[1, 2, 3], [4, 5, 6]]
    auto a = ones({2, 3}, DType::Float32);
    auto a_data = a.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;

    // Create B: 3x2 matrix [[1, 2], [3, 4], [5, 6]]
    auto b = ones({3, 2}, DType::Float32);
    auto b_data = b.data<float>();
    b_data[0] = 1.0f; b_data[1] = 2.0f;
    b_data[2] = 3.0f; b_data[3] = 4.0f;
    b_data[4] = 5.0f; b_data[5] = 6.0f;

    auto c = matmul(a, b);

    // Result should be 2x2
    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    auto c_data = c.data<float>();

    // Expected result: [[22, 28], [49, 64]]
    // Row 0: [1*1 + 2*3 + 3*5, 1*2 + 2*4 + 3*6] = [22, 28]
    // Row 1: [4*1 + 5*3 + 6*5, 4*2 + 5*4 + 6*6] = [49, 64]
    EXPECT_FLOAT_EQ(c_data[0], 22.0f);
    EXPECT_FLOAT_EQ(c_data[1], 28.0f);
    EXPECT_FLOAT_EQ(c_data[2], 49.0f);
    EXPECT_FLOAT_EQ(c_data[3], 64.0f);
}

TEST(CPUKernelsTest, MatMulFloat32_Identity) {
    // Create 3x3 identity-like patterns
    auto a = zeros({3, 3}, DType::Float32);
    auto a_data = a.data<float>();
    a_data[0] = 1.0f; a_data[1] = 0.0f; a_data[2] = 0.0f;
    a_data[3] = 0.0f; a_data[4] = 1.0f; a_data[5] = 0.0f;
    a_data[6] = 0.0f; a_data[7] = 0.0f; a_data[8] = 1.0f;

    auto b = ones({3, 3}, DType::Float32);
    auto b_data = b.data<float>();
    b_data[0] = 1.0f; b_data[1] = 2.0f; b_data[2] = 3.0f;
    b_data[3] = 4.0f; b_data[4] = 5.0f; b_data[5] = 6.0f;
    b_data[6] = 7.0f; b_data[7] = 8.0f; b_data[8] = 9.0f;

    auto c = matmul(a, b);

    // Identity * B should equal B
    auto c_data = c.data<float>();
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(c_data[i], b_data[i]) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MatMulFloat64_Basic) {
    // Create A: 2x2 matrix [[1, 2], [3, 4]]
    auto a = ones({2, 2}, DType::Float64);
    auto a_data = a.data<double>();
    a_data[0] = 1.0; a_data[1] = 2.0;
    a_data[2] = 3.0; a_data[3] = 4.0;

    // Create B: 2x2 matrix [[5, 6], [7, 8]]
    auto b = ones({2, 2}, DType::Float64);
    auto b_data = b.data<double>();
    b_data[0] = 5.0; b_data[1] = 6.0;
    b_data[2] = 7.0; b_data[3] = 8.0;

    auto c = matmul(a, b);

    auto c_data = c.data<double>();

    // Expected: [[19, 22], [43, 50]]
    EXPECT_DOUBLE_EQ(c_data[0], 19.0);
    EXPECT_DOUBLE_EQ(c_data[1], 22.0);
    EXPECT_DOUBLE_EQ(c_data[2], 43.0);
    EXPECT_DOUBLE_EQ(c_data[3], 50.0);
}

TEST(CPUKernelsTest, MatMulInt32_Basic) {
    // Create A: 2x2 matrix [[1, 2], [3, 4]]
    auto a = ones({2, 2}, DType::Int32);
    auto a_data = a.data<int32_t>();
    a_data[0] = 1; a_data[1] = 2;
    a_data[2] = 3; a_data[3] = 4;

    // Create B: 2x2 matrix [[2, 0], [1, 2]]
    auto b = ones({2, 2}, DType::Int32);
    auto b_data = b.data<int32_t>();
    b_data[0] = 2; b_data[1] = 0;
    b_data[2] = 1; b_data[3] = 2;

    auto c = matmul(a, b);

    auto c_data = c.data<int32_t>();

    // Expected: [[4, 4], [10, 8]]
    EXPECT_EQ(c_data[0], 4);
    EXPECT_EQ(c_data[1], 4);
    EXPECT_EQ(c_data[2], 10);
    EXPECT_EQ(c_data[3], 8);
}

TEST(CPUKernelsTest, MatMulFloat32_WithZeros) {
    auto a = ones({2, 3}, DType::Float32);
    auto b = zeros({3, 2}, DType::Float32);

    auto a_data = a.data<float>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    auto c = matmul(a, b);

    // Any matrix multiplied by zero matrix should be zero
    auto c_data = c.data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(c_data[i], 0.0f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MatMulFloat32_SingleElement) {
    // 1x1 matrices (scalars)
    auto a = ones({1, 1}, DType::Float32);
    auto b = ones({1, 1}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    a_data[0] = 3.0f;
    b_data[0] = 7.0f;

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 1);

    auto c_data = c.data<float>();
    EXPECT_FLOAT_EQ(c_data[0], 21.0f);
}

TEST(CPUKernelsTest, MatMulFloat32_VectorVector) {
    // 1x3 * 3x1 = 1x1 (dot product)
    auto a = ones({1, 3}, DType::Float32);
    auto b = ones({3, 1}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    b_data[0] = 4.0f; b_data[1] = 5.0f; b_data[2] = 6.0f;

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 1);
    EXPECT_EQ(c.shape()[1], 1);

    auto c_data = c.data<float>();
    // 1*4 + 2*5 + 3*6 = 4 + 10 + 18 = 32
    EXPECT_FLOAT_EQ(c_data[0], 32.0f);
}

TEST(CPUKernelsTest, MatMulFloat32_LargerMatrix) {
    // 4x3 * 3x2 = 4x2
    auto a = ones({4, 3}, DType::Float32);
    auto b = ones({3, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();

    // Initialize with sequential values
    for (int i = 0; i < 12; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }
    for (int i = 0; i < 6; i++) {
        b_data[i] = static_cast<float>(i + 1);
    }

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 4);
    EXPECT_EQ(c.shape()[1], 2);

    auto c_data = c.data<float>();

    // First row: [1,2,3] * [[1,2],[3,4],[5,6]] = [22, 28]
    EXPECT_FLOAT_EQ(c_data[0], 22.0f);
    EXPECT_FLOAT_EQ(c_data[1], 28.0f);
}

//==============================================================================
// Edge Cases and Error Conditions
//==============================================================================

TEST(CPUKernelsTest, AddFloat32_VerySmallValues) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 1e-8f;
        b_data[i] = 2e-8f;
    }

    auto c = add(a, b);
    auto c_data = c.data<float>();

    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(c_data[i], 3e-8f, 1e-15f) << "Mismatch at index " << i;
    }
}

TEST(CPUKernelsTest, MulFloat32_Overflow) {
    auto a = ones({2, 2}, DType::Float32);
    auto b = ones({2, 2}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    for (int i = 0; i < 4; i++) {
        a_data[i] = 1e20f;
        b_data[i] = 1e20f;
    }

    auto c = mul(a, b);
    auto c_data = c.data<float>();

    // 1e20f * 1e20f overflows float range — result must be infinity per IEEE 754
    for (int i = 0; i < 4; i++) {
        EXPECT_TRUE(std::isinf(c_data[i])) << "Expected overflow at index " << i;
    }
}

TEST(CPUKernelsTest, AllOperations_SingleElement) {
    // Test all operations with 1x1 tensors
    auto a = ones({1, 1}, DType::Float32);
    auto b = ones({1, 1}, DType::Float32);

    auto a_data = a.data<float>();
    auto b_data = b.data<float>();
    a_data[0] = 10.0f;
    b_data[0] = 2.0f;

    auto add_result = add(a, b);
    EXPECT_FLOAT_EQ(add_result.data<float>()[0], 12.0f);

    auto sub_result = sub(a, b);
    EXPECT_FLOAT_EQ(sub_result.data<float>()[0], 8.0f);

    auto mul_result = mul(a, b);
    EXPECT_FLOAT_EQ(mul_result.data<float>()[0], 20.0f);

    auto div_result = div(a, b);
    EXPECT_FLOAT_EQ(div_result.data<float>()[0], 5.0f);
}

TEST(CPUKernelsTest, ShapePreservation) {
    // Verify that operations preserve input shapes correctly
    std::vector<std::vector<int64_t>> shapes = {
        {1, 1}, {2, 3}, {3, 2}, {4, 5}, {10, 1}, {1, 10}
    };

    for (const auto& shape : shapes) {
        auto a = ones(shape, DType::Float32);
        auto b = ones(shape, DType::Float32);

        auto c = add(a, b);

        EXPECT_EQ(c.shape()[0], shape[0]) << "Shape[0] mismatch";
        EXPECT_EQ(c.shape()[1], shape[1]) << "Shape[1] mismatch";
        EXPECT_EQ(c.numel(), shape[0] * shape[1]) << "Numel mismatch";
    }
}


