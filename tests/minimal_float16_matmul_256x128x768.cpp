#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

TEST(MinimalFloat16Matmul, Dims256x128x768) {
    std::cerr << "\n===== Testing matmul (256, 128) @ (128, 768) -> (256, 768) =====" << std::endl;

    // Create input tensors matching the failing ALBERT test dimensions
    std::cerr << "Creating A tensor (256, 128)" << std::endl;
    Tensor A = randn({256, 128}, DType::Float16, Device::cpu());
    std::cerr << "A shape: [" << A.shape()[0] << ", " << A.shape()[1] << "]" << std::endl;
    std::cerr << "A total elements: " << (A.shape()[0] * A.shape()[1]) << std::endl;

    std::cerr << "\nCreating B tensor (128, 768)" << std::endl;
    Tensor B = randn({128, 768}, DType::Float16, Device::cpu());
    std::cerr << "B shape: [" << B.shape()[0] << ", " << B.shape()[1] << "]" << std::endl;
    std::cerr << "B total elements: " << (B.shape()[0] * B.shape()[1]) << std::endl;

    std::cerr << "\n===== About to call matmul =====" << std::endl;
    std::cerr << "Expected output shape: (256, 768)" << std::endl;
    std::cerr << "Expected dimensions: M=256, K=128, N=768" << std::endl;

    std::cerr << "\n===== Calling matmul =====\n" << std::endl;
    Tensor C = matmul(A, B);

    std::cerr << "\n===== Matmul completed =====" << std::endl;
    std::cerr << "C shape: [" << C.shape()[0] << ", " << C.shape()[1] << "]" << std::endl;
    std::cerr << "C total elements: " << (C.shape()[0] * C.shape()[1]) << std::endl;

    EXPECT_EQ(C.shape()[0], 256);
    EXPECT_EQ(C.shape()[1], 768);

    std::cerr << "===== Test completed successfully =====" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
