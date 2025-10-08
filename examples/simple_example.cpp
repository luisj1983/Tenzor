#include <tenzor/tenzor.hpp>
#include <iostream>

int main() {
    using namespace tenzor;

    std::cout << "Tenzor Simple Example\n";
    std::cout << "=====================\n\n";

    // Create tensors
    auto x = randn({3, 4});
    auto y = randn({3, 4});

    std::cout << "Created two 3x4 tensors\n";

    // Operations
    auto sum = x + y;
    auto product = x * y;

    std::cout << "Performed element-wise operations\n";

    // Matrix multiplication
    auto a = randn({2, 3});
    auto b = randn({3, 4});
    auto c = matmul(a, b);

    std::cout << "Matrix multiplication: (2,3) @ (3,4) = (2,4)\n";
    std::cout << "Result shape: " << c.shape()[0] << "x" << c.shape()[1] << "\n";

    return 0;
}
