#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/nn/layers/linear.hpp"

using namespace tenzor;

TEST(MinimalFloat16Linear, TinyLinear2x768) {
    std::cerr << "\n===== Testing Linear layer (2, 768) -> (2, 768) =====" << std::endl;

    // Create a simple 2x768 Float16 input
    std::cerr << "Creating input tensor (2, 768)" << std::endl;
    Tensor input_tensor = randn({2, 768}, DType::Float16, Device::cpu());
    Variable input(input_tensor, false);

    //  Create linear layer: 768 -> 768
    std::cerr << "Creating Linear layer 768 -> 768" << std::endl;
    nn::Linear linear(768, 768, true);

    // Forward pass
    std::cerr << "Running forward pass" << std::endl;
    Variable output = linear.forward(input);

    std::cerr << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << "]" << std::endl;

    EXPECT_EQ(output.shape()[0], 2);
    EXPECT_EQ(output.shape()[1], 768);

    std::cerr << "===== Test completed successfully =====" << std::endl;
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
