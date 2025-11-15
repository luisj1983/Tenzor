#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;

int main() {
    // Initialize Tenzor and load backends
    tenzor::initialize();

    std::cout << "Testing minimal Float16 matmul operations..." << std::endl;

    // Create small Float16 tensors
    auto a = randn({4, 8}, DType::Float16);
    auto b = randn({8, 4}, DType::Float16);

    std::cout << "Created tensors a(4x8) and b(8x4)" << std::endl;

    // Do 100 matmul operations to see if corruption accumulates
    for (int i = 0; i < 100; ++i) {
        auto c = matmul(a, b);  // Should give (4, 4)

        if (i % 10 == 0) {
            std::cout << "Iteration " << i << ": c.shape = ["
                      << c.shape()[0] << ", " << c.shape()[1] << "]" << std::endl;
        }

        // Use c as input for next iteration to keep memory active
        if (i < 99) {
            a = c;
            b = randn({4, 4}, DType::Float16);
        }
    }

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
