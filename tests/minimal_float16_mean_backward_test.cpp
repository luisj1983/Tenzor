#include <iostream>
#include <gtest/gtest.h>
#include "tenzor/core/tensor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

TEST(MinimalFloat16MeanBackward, SimpleCase) {
    std::cerr << "\n===== Testing mean().backward() with Float16 =====" << std::endl;

    // Create a simple Float16 input with gradient tracking
    std::cerr << "Creating Float16 input [1, 10]" << std::endl;
    Variable input(Tensor({1, 10}, DType::Float16, Device::cpu()), true);

    // Fill with ones for simplicity
    auto* data = input.tensor().data<Float16>();
    for (int i = 0; i < 10; ++i) {
        data[i] = Float16(1.0f);
    }

    std::cerr << "Input created, calling mean()" << std::endl;
    Variable output = mean(input);

    std::cerr << "Mean computed: " << static_cast<float>(output.tensor().data<Float16>()[0]) << std::endl;

    std::cerr << "Calling backward()" << std::endl;
    output.backward();

    std::cerr << "Backward completed" << std::endl;

    // Check if gradient exists
    ASSERT_TRUE(input.grad().has_value()) << "Gradient should exist";

    auto grad = input.grad().value();
    std::cerr << "Gradient dtype: " << static_cast<int>(grad.dtype()) << " (Float16=" << static_cast<int>(DType::Float16) << ")" << std::endl;

    // Convert to Float32 for inspection
    auto grad_f32 = grad.to(DType::Float32);
    auto* grad_data = grad_f32.data<float>();

    std::cerr << "First 10 gradient values: ";
    for (int i = 0; i < 10; ++i) {
        std::cerr << grad_data[i] << " ";
    }
    std::cerr << std::endl;

    // Check for non-zero gradients
    bool has_nonzero = false;
    float sum = 0.0f;
    for (int i = 0; i < 10; ++i) {
        sum += std::abs(grad_data[i]);
        if (std::abs(grad_data[i]) > 1e-7f) {
            has_nonzero = true;
        }
    }

    std::cerr << "Sum of absolute gradients: " << sum << std::endl;
    std::cerr << "Has non-zero gradients: " << (has_nonzero ? "YES" : "NO") << std::endl;

    // Expected: each gradient should be 1/10 = 0.1
    EXPECT_TRUE(has_nonzero) << "Gradients should be non-zero";
    EXPECT_NEAR(grad_data[0], 0.1f, 0.01f) << "Gradient should be approximately 0.1";
}

TEST(MinimalFloat16MeanBackward, MatmulBackward) {
    std::cerr << "\n===== Testing matmul().mean().backward() with Float16 =====" << std::endl;

    // Create two small Float16 matrices
    Variable A(Tensor({2, 3}, DType::Float16, Device::cpu()), true);
    Variable B(Tensor({3, 2}, DType::Float16, Device::cpu()), true);

    // Fill with ones
    auto* a_data = A.tensor().data<Float16>();
    auto* b_data = B.tensor().data<Float16>();
    for (int i = 0; i < 6; ++i) {
        a_data[i] = Float16(1.0f);
        b_data[i] = Float16(1.0f);
    }

    std::cerr << "Calling matmul" << std::endl;
    Variable C = matmul(A, B);

    std::cerr << "C shape: [" << C.shape()[0] << ", " << C.shape()[1] << "]" << std::endl;

    std::cerr << "Calling mean" << std::endl;
    Variable loss = mean(C);

    std::cerr << "Loss: " << static_cast<float>(loss.tensor().data<Float16>()[0]) << std::endl;

    std::cerr << "Calling backward" << std::endl;
    loss.backward();

    std::cerr << "Backward completed" << std::endl;

    // Check A gradient
    ASSERT_TRUE(A.grad().has_value()) << "A gradient should exist";
    auto grad_A = A.grad().value().to(DType::Float32);
    auto* grad_A_data = grad_A.data<float>();

    std::cerr << "A gradients: ";
    float sum_A = 0.0f;
    for (int i = 0; i < 6; ++i) {
        std::cerr << grad_A_data[i] << " ";
        sum_A += std::abs(grad_A_data[i]);
    }
    std::cerr << std::endl;
    std::cerr << "Sum of absolute A gradients: " << sum_A << std::endl;

    // Check B gradient
    ASSERT_TRUE(B.grad().has_value()) << "B gradient should exist";
    auto grad_B = B.grad().value().to(DType::Float32);
    auto* grad_B_data = grad_B.data<float>();

    std::cerr << "B gradients: ";
    float sum_B = 0.0f;
    for (int i = 0; i < 6; ++i) {
        std::cerr << grad_B_data[i] << " ";
        sum_B += std::abs(grad_B_data[i]);
    }
    std::cerr << std::endl;
    std::cerr << "Sum of absolute B gradients: " << sum_B << std::endl;

    EXPECT_GT(sum_A, 0.0f) << "A gradients should be non-zero";
    EXPECT_GT(sum_B, 0.0f) << "B gradients should be non-zero";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
