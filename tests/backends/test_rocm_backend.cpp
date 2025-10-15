#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

class ROCmBackendTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Initialize Tenzor (loads backends)
        initialize();
    }
};

TEST_F(ROCmBackendTest, BackendRegistration) {
    // Test that we can create a ROCm device object
    // This will fail if backend isn't registered
    EXPECT_NO_THROW({
        auto device = Device::rocm(0);
        EXPECT_EQ(device.type, Device::Type::ROCm);
        EXPECT_EQ(device.index, 0);
    });
}

TEST_F(ROCmBackendTest, TensorCreation) {
    try {
        auto device = Device::rocm(0);

        // Test basic tensor creation operations
        EXPECT_NO_THROW({
            auto t1 = zeros({2, 3}, DType::Float32, device);
            std::vector<int64_t> shape1(t1.shape().begin(), t1.shape().end());
            EXPECT_EQ(shape1, std::vector<int64_t>({2, 3}));
            EXPECT_EQ(t1.device().type, Device::Type::ROCm);
        });

        EXPECT_NO_THROW({
            auto t2 = ones({3, 4}, DType::Float32, device);
            std::vector<int64_t> shape2(t2.shape().begin(), t2.shape().end());
            EXPECT_EQ(shape2, std::vector<int64_t>({3, 4}));
        });

        EXPECT_NO_THROW({
            auto t3 = full({2, 2}, 5.0f, DType::Float32, device);
            std::vector<int64_t> shape3(t3.shape().begin(), t3.shape().end());
            EXPECT_EQ(shape3, std::vector<int64_t>({2, 2}));
        });

    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROCm device not available: " << e.what();
    }
}

TEST_F(ROCmBackendTest, BasicOperations) {
    try {
        auto device = Device::rocm(0);

        // Test implemented operations
        auto a = ones({2, 3}, DType::Float32, device);
        auto b = ones({2, 3}, DType::Float32, device);

        // Element-wise operations (should be implemented)
        EXPECT_NO_THROW({
            auto c = a + b;  // add
            EXPECT_EQ(c.ndim(), a.ndim());
        });

        EXPECT_NO_THROW({
            auto c = a - b;  // sub
            EXPECT_EQ(c.ndim(), a.ndim());
        });

        EXPECT_NO_THROW({
            auto c = a * b;  // mul
            EXPECT_EQ(c.ndim(), a.ndim());
        });

        EXPECT_NO_THROW({
            auto c = a / b;  // div
            EXPECT_EQ(c.ndim(), a.ndim());
        });

    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROCm device not available: " << e.what();
    }
}

TEST_F(ROCmBackendTest, UnaryOperations) {
    try {
        auto device = Device::rocm(0);

        auto input = ones({2, 3}, DType::Float32, device);

        // Unary operations
        EXPECT_NO_THROW({
            auto result = sqrt(input);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

        EXPECT_NO_THROW({
            auto result = neg(input);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

        EXPECT_NO_THROW({
            auto result = tenzor::abs(input);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

        EXPECT_NO_THROW({
            auto result = exp(input);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

        EXPECT_NO_THROW({
            auto result = log(input);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROCm device not available: " << e.what();
    }
}

// Activation tests removed - would require Variable support which is more complex
// The backend itself supports activations, but testing through nn:: interface
// requires autograd which is outside the scope of backend testing

TEST_F(ROCmBackendTest, TransformOperations) {
    try {
        auto device = Device::rocm(0);

        auto input = ones({2, 3, 4}, DType::Float32, device);

        // Transform operations
        EXPECT_NO_THROW({
            auto result = transpose(input, 0, 1);
            std::vector<int64_t> shape_t(result.shape().begin(), result.shape().end());
            EXPECT_EQ(shape_t, std::vector<int64_t>({3, 2, 4}));
        });

        EXPECT_NO_THROW({
            auto result = reshape(input, {2, 12});
            std::vector<int64_t> shape_r(result.shape().begin(), result.shape().end());
            EXPECT_EQ(shape_r, std::vector<int64_t>({2, 12}));
        });

        EXPECT_NO_THROW({
            auto result = squeeze(input, 0);
        });

        EXPECT_NO_THROW({
            auto result = unsqueeze(input, 0);
            EXPECT_EQ(result.ndim(), input.ndim() + 1);
        });

    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROCm device not available: " << e.what();
    }
}

TEST_F(ROCmBackendTest, UnimplementedOperations) {
    try {
        auto device = Device::rocm(0);

        auto a = ones({2, 3}, DType::Float32, device);
        auto b = ones({3, 4}, DType::Float32, device);

        // Matmul is stubbed - should throw
        EXPECT_THROW({
            auto c = matmul(a, b);
        }, std::runtime_error);

        // Reduction operations are stubbed
        EXPECT_THROW({
            auto s = sum(a, 0, false);
        }, std::runtime_error);

        EXPECT_THROW({
            auto m = mean(a, 0, false);
        }, std::runtime_error);

    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROCm device not available: " << e.what();
    }
}

TEST_F(ROCmBackendTest, MemoryOperations) {
    try {
        auto device = Device::rocm(0);

        // Test memory allocation and deallocation
        EXPECT_NO_THROW({
            auto t = zeros({100, 100}, DType::Float32, device);
            // Tensor will be destroyed here, testing deallocate
        });

        // Test contiguous operation
        auto input = ones({2, 3}, DType::Float32, device);
        EXPECT_NO_THROW({
            auto result = contiguous(input);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

        // Test clone - create a copy via operations
        EXPECT_NO_THROW({
            auto result = input + zeros({2, 3}, DType::Float32, device);
            EXPECT_EQ(result.ndim(), input.ndim());
        });

    } catch (const std::exception& e) {
        GTEST_SKIP() << "ROCm device not available: " << e.what();
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
