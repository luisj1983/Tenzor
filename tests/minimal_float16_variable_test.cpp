#include <iostream>
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"

using namespace tenzor;
namespace autograd = tenzor;

int main() {
    tenzor::initialize();

    std::cout << "Testing minimal Float16 Variable/autograd operations..." << std::endl;

    // Create small Float16 Variables with grad enabled
    auto a_tensor = randn({4, 8}, DType::Float16);
    auto b_tensor = randn({8, 4}, DType::Float16);

    Variable a(a_tensor, true);  // requires_grad=true
    Variable b(b_tensor, true);

    std::cout << "Created Variables a(4x8) and b(8x4) with grad enabled" << std::endl;

    // Do 50 matmul operations with autograd graph building
    for (int i = 0; i < 50; ++i) {
        // This should build autograd graph nodes
        auto c = autograd::matmul(a, b);  // Should give (4, 4)

        if (i % 10 == 0) {
            std::cout << "Iteration " << i << ": c.shape = ["
                      << c.shape()[0] << ", " << c.shape()[1] << "]"
                      << ", requires_grad=" << c.requires_grad() << std::endl;
        }

        // Use c as input for next iteration - this creates deep autograd graph
        if (i < 49) {
            a = c;
            b_tensor = randn({4, 4}, DType::Float16);
            b = Variable(b_tensor, true);
        }
    }

    std::cout << "Test completed successfully!" << std::endl;
    return 0;
}
