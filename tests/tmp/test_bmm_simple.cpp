#include <iostream>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

int main() {
    // Initialize Tenzor
    tenzor::initialize();

    std::cout << "Testing bmm function..." << std::endl;

    // Test 1: Basic functionality
    {
        Tensor a = ones({2, 3, 4}, DType::Float32, Device::cpu());
        Tensor b = ones({2, 4, 5}, DType::Float32, Device::cpu());

        Tensor result = bmm(a, b);

        std::cout << "Test 1 - Basic functionality:" << std::endl;
        std::cout << "  Input a shape: [" << a.shape()[0] << ", " << a.shape()[1] << ", " << a.shape()[2] << "]" << std::endl;
        std::cout << "  Input b shape: [" << b.shape()[0] << ", " << b.shape()[1] << ", " << b.shape()[2] << "]" << std::endl;
        std::cout << "  Output shape: [" << result.shape()[0] << ", " << result.shape()[1] << ", " << result.shape()[2] << "]" << std::endl;

        bool shape_correct = (result.shape()[0] == 2 && result.shape()[1] == 3 && result.shape()[2] == 5);
        std::cout << "  Shape check: " << (shape_correct ? "PASSED" : "FAILED") << std::endl;

        // Verify values
        auto* data = result.data<float>();
        bool values_correct = true;
        for (int64_t i = 0; i < result.numel(); ++i) {
            if (std::abs(data[i] - 4.0f) > 1e-5f) {
                values_correct = false;
                break;
            }
        }
        std::cout << "  Value check: " << (values_correct ? "PASSED" : "FAILED") << std::endl;
    }

    // Test 2: Invalid input (non-3D)
    {
        std::cout << "\nTest 2 - Invalid input (non-3D):" << std::endl;
        try {
            Tensor a = ones({2, 3}, DType::Float32, Device::cpu());
            Tensor b = ones({3, 4}, DType::Float32, Device::cpu());
            Tensor result = bmm(a, b);
            std::cout << "  FAILED - Should have thrown exception" << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << "  PASSED - Exception caught: " << e.what() << std::endl;
        }
    }

    // Test 3: Dimension mismatch
    {
        std::cout << "\nTest 3 - Dimension mismatch:" << std::endl;
        try {
            Tensor a = ones({2, 3, 4}, DType::Float32, Device::cpu());
            Tensor b = ones({2, 5, 6}, DType::Float32, Device::cpu());
            Tensor result = bmm(a, b);
            std::cout << "  FAILED - Should have thrown exception" << std::endl;
        } catch (const std::runtime_error& e) {
            std::cout << "  PASSED - Exception caught: " << e.what() << std::endl;
        }
    }

    std::cout << "\nAll bmm tests completed!" << std::endl;
    return 0;
}
