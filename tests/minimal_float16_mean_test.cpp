#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

TEST(MinimalFloat16Mean, ScalarMean) {
    std::cerr << "\n===== Testing scalar mean().backward() with Float16 =====\n";

    // Create a simple 2-element Float16 tensor
    Variable input(Tensor({2}, DType::Float16, Device::cpu()), true);
    auto* data = input.tensor().data<Float16>();
    data[0] = Float16(3.0f);
    data[1] = Float16(5.0f);

    std::cerr << "Input values: " << static_cast<float>(data[0]) << ", " << static_cast<float>(data[1]) << "\n";

    // Compute mean (should be 4.0)
    Variable output = mean(input);
    std::cerr << "Mean: " << static_cast<float>(output.tensor().data<Float16>()[0]) << "\n";

    // Backward
    std::cerr << "Calling backward...\n";
    output.backward();

    // Check gradient
    ASSERT_TRUE(input.grad().has_value()) << "Gradient should exist";
    auto grad = input.grad().value();
    std::cerr << "Gradient dtype: " << static_cast<int>(grad.dtype()) << "\n";

    auto grad_f32 = grad.to(DType::Float32);
    auto* grad_data = grad_f32.data<float>();
    std::cerr << "Gradients: " << grad_data[0] << ", " << grad_data[1] << "\n";

    // Expected: each gradient should be 1/2 = 0.5
    EXPECT_NEAR(grad_data[0], 0.5f, 0.01f);
    EXPECT_NEAR(grad_data[1], 0.5f, 0.01f);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
