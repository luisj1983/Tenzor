#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <cmath>
#include <limits>

using namespace tenzor;

// Skip tests if OneAPI is not available
class OneAPIBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        tenzor::initialize();

        // Check if OneAPI backend is available
        try {
            auto test_tensor = ones({2, 2}, DType::Float32, Device::oneapi(0));
            oneapi_available_ = true;
        } catch (...) {
            oneapi_available_ = false;
            GTEST_SKIP() << "OneAPI backend not available, skipping tests";
        }
    }

    bool oneapi_available_ = false;
};

//==============================================================================
// Math Operations Tests
//==============================================================================

TEST_F(OneAPIBackendTest, AddFloat32_Basic) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto b = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto c = add(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 3);

    auto data = c.cpu().data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, SubFloat32_Basic) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 3}, 5.0f, DType::Float32, Device::oneapi(0));
    auto b = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto c = sub(a, b);

    auto data = c.cpu().data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 4.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, MulFloat32_Basic) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 3}, 3.0f, DType::Float32, Device::oneapi(0));
    auto b = full({2, 3}, 4.0f, DType::Float32, Device::oneapi(0));
    auto c = mul(a, b);

    auto data = c.cpu().data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 12.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, DivFloat32_Basic) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 3}, 12.0f, DType::Float32, Device::oneapi(0));
    auto b = full({2, 3}, 4.0f, DType::Float32, Device::oneapi(0));
    auto c = div(a, b);

    auto data = c.cpu().data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, MatMulFloat32_Basic) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto b = ones({3, 2}, DType::Float32, Device::oneapi(0));

    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<float>();
    a_data[0] = 1.0f; a_data[1] = 2.0f; a_data[2] = 3.0f;
    a_data[3] = 4.0f; a_data[4] = 5.0f; a_data[5] = 6.0f;
    a = a_cpu.to(Device::oneapi(0));

    auto b_cpu = b.cpu();
    auto b_data = b_cpu.data<float>();
    b_data[0] = 1.0f; b_data[1] = 2.0f;
    b_data[2] = 3.0f; b_data[3] = 4.0f;
    b_data[4] = 5.0f; b_data[5] = 6.0f;
    b = b_cpu.to(Device::oneapi(0));

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 2);
    EXPECT_EQ(c.shape()[1], 2);

    auto c_cpu = c.cpu();
    auto c_data = c_cpu.data<float>();

    // Expected: [[22, 28], [49, 64]]
    EXPECT_FLOAT_EQ(c_data[0], 22.0f);
    EXPECT_FLOAT_EQ(c_data[1], 28.0f);
    EXPECT_FLOAT_EQ(c_data[2], 49.0f);
    EXPECT_FLOAT_EQ(c_data[3], 64.0f);
}

//==============================================================================
// Unary Operations Tests
//==============================================================================

TEST_F(OneAPIBackendTest, SqrtFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({3, 3}, 4.0f, DType::Float32, Device::oneapi(0));
    auto b = sqrt(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(data[i], 2.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, NegFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 3}, 5.0f, DType::Float32, Device::oneapi(0));
    auto b = neg(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], -5.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, AbsFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 3}, -3.0f, DType::Float32, Device::oneapi(0));
    auto b = abs(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, ExpFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 2}, 1.0f, DType::Float32, Device::oneapi(0));
    auto b = exp(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 2.71828f, 1e-4f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, LogFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 2}, 2.71828f, DType::Float32, Device::oneapi(0));
    auto b = log(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 1.0f, 1e-4f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Activation Functions Tests
//==============================================================================

TEST_F(OneAPIBackendTest, ReLUFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({3, 3}, DType::Float32, Device::oneapi(0));
    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<float>();
    for (int i = 0; i < 9; i++) {
        a_data[i] = static_cast<float>(i - 4);  // [-4, -3, -2, -1, 0, 1, 2, 3, 4]
    }
    a = a_cpu.to(Device::oneapi(0));

    auto b = relu(a);
    auto data = b.cpu().data<float>();

    // Expected: [0, 0, 0, 0, 0, 1, 2, 3, 4]
    for (int i = 0; i < 5; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f) << "Mismatch at index " << i;
    }
    for (int i = 5; i < 9; i++) {
        EXPECT_FLOAT_EQ(data[i], static_cast<float>(i - 4)) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, SigmoidFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = zeros({2, 2}, DType::Float32, Device::oneapi(0));
    auto b = sigmoid(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.5f) << "Sigmoid(0) should be 0.5";
    }
}

TEST_F(OneAPIBackendTest, TanhFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = zeros({2, 2}, DType::Float32, Device::oneapi(0));
    auto b = tanh(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f) << "Tanh(0) should be 0.0";
    }
}

TEST_F(OneAPIBackendTest, GELUFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = zeros({2, 2}, DType::Float32, Device::oneapi(0));
    auto b = gelu(a);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 0.0f, 1e-4f) << "GELU(0) should be approximately 0.0";
    }
}

TEST_F(OneAPIBackendTest, LeakyReLUFloat32) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 2}, -1.0f, DType::Float32, Device::oneapi(0));
    auto b = leaky_relu(a, 0.1f);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_FLOAT_EQ(data[i], -0.1f) << "LeakyReLU(-1, 0.1) should be -0.1";
    }
}

//==============================================================================
// Reduction Operations Tests
//==============================================================================

TEST_F(OneAPIBackendTest, SumReduction) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({3, 4}, DType::Float32, Device::oneapi(0));
    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<float>();
    for (int i = 0; i < 12; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }
    a = a_cpu.to(Device::oneapi(0));

    // Sum over last dimension
    auto b = sum(a, 1, false);

    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape().size(), 1);

    auto data = b.cpu().data<float>();
    // Row 0: 1+2+3+4 = 10
    // Row 1: 5+6+7+8 = 26
    // Row 2: 9+10+11+12 = 42
    EXPECT_FLOAT_EQ(data[0], 10.0f);
    EXPECT_FLOAT_EQ(data[1], 26.0f);
    EXPECT_FLOAT_EQ(data[2], 42.0f);
}

TEST_F(OneAPIBackendTest, MeanReduction) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 4}, 8.0f, DType::Float32, Device::oneapi(0));
    auto b = mean(a, 1, false);

    EXPECT_EQ(b.shape()[0], 2);

    auto data = b.cpu().data<float>();
    for (int i = 0; i < 2; i++) {
        EXPECT_FLOAT_EQ(data[i], 8.0f);
    }
}

TEST_F(OneAPIBackendTest, MaxReduction) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<float>();
    a_data[0] = 1.0f; a_data[1] = 5.0f; a_data[2] = 3.0f;
    a_data[3] = 2.0f; a_data[4] = 8.0f; a_data[5] = 4.0f;
    a = a_cpu.to(Device::oneapi(0));

    auto b = max(a, 1, false);

    auto data = b.cpu().data<float>();
    EXPECT_FLOAT_EQ(data[0], 5.0f);
    EXPECT_FLOAT_EQ(data[1], 8.0f);
}

TEST_F(OneAPIBackendTest, MinReduction) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<float>();
    a_data[0] = 1.0f; a_data[1] = 5.0f; a_data[2] = 3.0f;
    a_data[3] = 2.0f; a_data[4] = 8.0f; a_data[5] = 4.0f;
    a = a_cpu.to(Device::oneapi(0));

    auto b = min(a, 1, false);

    auto data = b.cpu().data<float>();
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 2.0f);
}

//==============================================================================
// Transform Operations Tests
//==============================================================================

TEST_F(OneAPIBackendTest, Reshape) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 6}, DType::Float32, Device::oneapi(0));
    auto b = reshape(a, {3, 4});

    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 4);
    EXPECT_EQ(b.numel(), 12);
}

TEST_F(OneAPIBackendTest, Transpose) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<float>();
    for (int i = 0; i < 6; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }
    a = a_cpu.to(Device::oneapi(0));

    auto b = transpose(a, 0, 1);

    EXPECT_EQ(b.shape()[0], 3);
    EXPECT_EQ(b.shape()[1], 2);

    auto data = b.cpu().data<float>();
    // Original: [[1,2,3],[4,5,6]]
    // Transposed: [[1,4],[2,5],[3,6]]
    EXPECT_FLOAT_EQ(data[0], 1.0f);
    EXPECT_FLOAT_EQ(data[1], 4.0f);
    EXPECT_FLOAT_EQ(data[2], 2.0f);
    EXPECT_FLOAT_EQ(data[3], 5.0f);
}

TEST_F(OneAPIBackendTest, Squeeze) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 1, 3}, DType::Float32, Device::oneapi(0));
    auto b = squeeze(a, 1);

    EXPECT_EQ(b.shape().size(), 2);
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 3);
}

TEST_F(OneAPIBackendTest, Unsqueeze) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float32, Device::oneapi(0));
    auto b = unsqueeze(a, 1);

    EXPECT_EQ(b.shape().size(), 3);
    EXPECT_EQ(b.shape()[0], 2);
    EXPECT_EQ(b.shape()[1], 1);
    EXPECT_EQ(b.shape()[2], 3);
}

//==============================================================================
// Fill Operations Tests
//==============================================================================

TEST_F(OneAPIBackendTest, ZerosCreation) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = zeros({3, 4}, DType::Float32, Device::oneapi(0));

    EXPECT_EQ(a.shape()[0], 3);
    EXPECT_EQ(a.shape()[1], 4);

    auto data = a.cpu().data<float>();
    for (int i = 0; i < 12; i++) {
        EXPECT_FLOAT_EQ(data[i], 0.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, OnesCreation) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 5}, DType::Float32, Device::oneapi(0));

    auto data = a.cpu().data<float>();
    for (int i = 0; i < 10; i++) {
        EXPECT_FLOAT_EQ(data[i], 1.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, FullCreation) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({3, 3}, 7.5f, DType::Float32, Device::oneapi(0));

    auto data = a.cpu().data<float>();
    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(data[i], 7.5f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Float64 (Double) Tests
//==============================================================================

TEST_F(OneAPIBackendTest, AddFloat64) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 3}, DType::Float64, Device::oneapi(0));
    auto b = ones({2, 3}, DType::Float64, Device::oneapi(0));
    auto c = add(a, b);

    auto data = c.cpu().data<double>();
    for (int i = 0; i < 6; i++) {
        EXPECT_DOUBLE_EQ(data[i], 2.0) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, MatMulFloat64) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({2, 2}, DType::Float64, Device::oneapi(0));
    auto b = ones({2, 2}, DType::Float64, Device::oneapi(0));

    auto a_cpu = a.cpu();
    auto a_data = a_cpu.data<double>();
    a_data[0] = 1.0; a_data[1] = 2.0;
    a_data[2] = 3.0; a_data[3] = 4.0;
    a = a_cpu.to(Device::oneapi(0));

    auto b_cpu = b.cpu();
    auto b_data = b_cpu.data<double>();
    b_data[0] = 5.0; b_data[1] = 6.0;
    b_data[2] = 7.0; b_data[3] = 8.0;
    b = b_cpu.to(Device::oneapi(0));

    auto c = matmul(a, b);

    auto c_cpu = c.cpu();
    auto c_data = c_cpu.data<double>();

    // Expected: [[19, 22], [43, 50]]
    EXPECT_DOUBLE_EQ(c_data[0], 19.0);
    EXPECT_DOUBLE_EQ(c_data[1], 22.0);
    EXPECT_DOUBLE_EQ(c_data[2], 43.0);
    EXPECT_DOUBLE_EQ(c_data[3], 50.0);
}

//==============================================================================
// Device Transfer Tests
//==============================================================================

TEST_F(OneAPIBackendTest, CPUToOneAPITransfer) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({3, 3}, DType::Float32);  // CPU tensor
    auto a_data = a.data<float>();
    for (int i = 0; i < 9; i++) {
        a_data[i] = static_cast<float>(i + 1);
    }

    auto b = a.to(Device::oneapi(0));  // Transfer to OneAPI

    EXPECT_EQ(b.device().type, Device::Type::OneAPI);

    auto b_cpu = b.cpu();  // Transfer back to CPU
    auto b_data = b_cpu.data<float>();

    for (int i = 0; i < 9; i++) {
        EXPECT_FLOAT_EQ(b_data[i], a_data[i]) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, OneAPIToCPUTransfer) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 3}, 5.0f, DType::Float32, Device::oneapi(0));
    auto b = a.cpu();

    EXPECT_EQ(b.device().type, Device::Type::CPU);

    auto data = b.data<float>();
    for (int i = 0; i < 6; i++) {
        EXPECT_FLOAT_EQ(data[i], 5.0f) << "Mismatch at index " << i;
    }
}

//==============================================================================
// Edge Cases and Error Handling
//==============================================================================

TEST_F(OneAPIBackendTest, LargeMatrixMultiplication) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({128, 256}, DType::Float32, Device::oneapi(0));
    auto b = ones({256, 64}, DType::Float32, Device::oneapi(0));

    auto c = matmul(a, b);

    EXPECT_EQ(c.shape()[0], 128);
    EXPECT_EQ(c.shape()[1], 64);

    // Check a few values
    auto c_cpu = c.cpu();
    auto c_data = c_cpu.data<float>();
    // Each element should be 256.0 (sum of 256 ones)
    EXPECT_FLOAT_EQ(c_data[0], 256.0f);
    EXPECT_FLOAT_EQ(c_data[100], 256.0f);
}

TEST_F(OneAPIBackendTest, SmallValueOperations) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({2, 2}, 1e-8f, DType::Float32, Device::oneapi(0));
    auto b = full({2, 2}, 2e-8f, DType::Float32, Device::oneapi(0));

    auto c = add(a, b);

    auto data = c.cpu().data<float>();
    for (int i = 0; i < 4; i++) {
        EXPECT_NEAR(data[i], 3e-8f, 1e-15f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, SingleElementTensor) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = full({1}, 5.0f, DType::Float32, Device::oneapi(0));
    auto b = full({1}, 3.0f, DType::Float32, Device::oneapi(0));

    auto c = add(a, b);

    EXPECT_EQ(c.numel(), 1);

    auto data = c.cpu().data<float>();
    EXPECT_FLOAT_EQ(data[0], 8.0f);
}

//==============================================================================
// Performance/Stress Tests
//==============================================================================

TEST_F(OneAPIBackendTest, MultipleSequentialOperations) {
    if (!oneapi_available_) GTEST_SKIP();

    auto a = ones({10, 10}, DType::Float32, Device::oneapi(0));

    // Chain multiple operations
    auto b = add(a, a);      // 2.0
    auto c = mul(b, b);      // 4.0
    auto d = sub(c, a);      // 3.0
    auto e = div(d, a);      // 3.0

    auto data = e.cpu().data<float>();
    for (int i = 0; i < 100; i++) {
        EXPECT_FLOAT_EQ(data[i], 3.0f) << "Mismatch at index " << i;
    }
}

TEST_F(OneAPIBackendTest, MemoryManagement) {
    if (!oneapi_available_) GTEST_SKIP();

    // Create and destroy many tensors to test memory management
    for (int iter = 0; iter < 10; ++iter) {
        auto a = ones({100, 100}, DType::Float32, Device::oneapi(0));
        auto b = ones({100, 100}, DType::Float32, Device::oneapi(0));
        auto c = add(a, b);

        // Tensors should be automatically freed
    }

    // If we got here without crashing, memory management is working
    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
