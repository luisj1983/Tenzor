#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

TEST(MinimalFloat16MatmulDebug, SmallMatmul) {
    // Test the specific matmul that's failing: (2, 768) @ (768, 768)
    std::cerr << "\n===== Creating input tensor (2, 768) =====" << std::endl;
    Tensor input = randn({2, 768}, DType::Float16, Device::cpu());
    std::cerr << "Input shape: [" << input.shape()[0] << ", " << input.shape()[1] << "]" << std::endl;
    std::cerr << "Input size_bytes: " << input.size_bytes() << std::endl;
    std::cerr << "Input num_elements: " << (input.size_bytes() / 2) << std::endl;

    std::cerr << "\n===== Creating weight tensor (768, 768) =====" << std::endl;
    Tensor weight = randn({768, 768}, DType::Float16, Device::cpu());
    std::cerr << "Weight shape: [" << weight.shape()[0] << ", " << weight.shape()[1] << "]" << std::endl;
    std::cerr << "Weight size_bytes: " << weight.size_bytes() << std::endl;
    std::cerr << "Weight num_elements: " << (weight.size_bytes() / 2) << std::endl;

    std::cerr << "\n===== About to call matmul =====" << std::endl;
    std::cerr << "Expected output shape: (2, 768)" << std::endl;
    std::cerr << "Expected dimensions: M=2, N=768, K=768" << std::endl;

    // This should compute: output = input @ weight.T
    // But weight is already (768, 768), so we need to transpose it first
    std::cerr << "\n===== Calling matmul =====" << std::endl;
    Tensor output = matmul(input, weight.permute({1, 0}));

    std::cerr << "\n===== Matmul completed =====" << std::endl;
    std::cerr << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;
    std::cerr << "Output size_bytes: " << output.size_bytes() << std::endl;

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 768);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
