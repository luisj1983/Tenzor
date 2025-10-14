#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/linear.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Creating Linear layer (128 -> 128)" << std::endl;
    Linear linear(128, 128, true);

    std::cout << "Creating input (2, 5, 128)" << std::endl;
    Variable input(randn({2, 5, 128}), true);

    std::cout << "Forward pass" << std::endl;
    Variable output = linear.forward(input);
    std::cout << "Output shape: [";
    for (size_t i = 0; i < output.shape().size(); ++i) {
        std::cout << output.shape()[i];
        if (i < output.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Computing mean" << std::endl;
    Variable loss = mean(output);
    std::cout << "Loss shape: [";
    for (size_t i = 0; i < loss.shape().size(); ++i) {
        std::cout << loss.shape()[i];
        if (i < loss.shape().size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;

    std::cout << "Calling backward..." << std::endl;
    try {
        loss.backward();
        std::cout << "Backward successful!" << std::endl;

        std::cout << "Checking input gradient..." << std::endl;
        if (input.has_grad()) {
            std::cout << "Input has gradient! Shape: [";
            for (size_t i = 0; i < input.grad().value().shape().size(); ++i) {
                std::cout << input.grad().value().shape()[i];
                if (i < input.grad().value().shape().size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "Input does NOT have gradient" << std::endl;
        }
    } catch (const std::exception& e) {
        std::cout << "ERROR during backward: " << e.what() << std::endl;
        return 1;
    }

    return 0;
}
