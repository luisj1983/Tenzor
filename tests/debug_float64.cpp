#include <tenzor/tenzor.hpp>
#include <iostream>
#include <iomanip>

using namespace tenzor;

int main() {
    tenzor::initialize();

    // Simple test: f(x) = x^2 + 2*x with Float64
    // Expected gradient: 2x + 2

    auto device = Device::cpu();

    // Test with x = 1.0
    {
        std::cout << "\n=== Test with x = 1.0 (Float64) ===" << std::endl;

        Tensor data = zeros({1}, DType::Float64, device);
        data.data<double>()[0] = 1.0;

        Variable x(data, true);
        Variable y = x * x + x * 2.0;

        std::cout << "x = " << x.tensor().item<double>() << std::endl;
        std::cout << "y = " << y.tensor().item<double>() << std::endl;

        y.backward();

        if (x.has_grad()) {
            double grad = x.grad().value().item<double>();
            std::cout << "Analytical gradient: " << grad << std::endl;
            std::cout << "Expected gradient: 4.0 (2*1 + 2)" << std::endl;
            std::cout << "Error: " << std::abs(grad - 4.0) << std::endl;
        } else {
            std::cout << "ERROR: No gradient computed!" << std::endl;
        }
    }

    // Test with x = 0.5
    {
        std::cout << "\n=== Test with x = 0.5 (Float64) ===" << std::endl;

        Tensor data = zeros({1}, DType::Float64, device);
        data.data<double>()[0] = 0.5;

        Variable x(data, true);
        Variable y = x * x + x * 2.0;

        std::cout << "x = " << x.tensor().item<double>() << std::endl;
        std::cout << "y = " << y.tensor().item<double>() << std::endl;

        y.backward();

        if (x.has_grad()) {
            double grad = x.grad().value().item<double>();
            std::cout << "Analytical gradient: " << std::setprecision(15) << grad << std::endl;
            std::cout << "Expected gradient: 3.0 (2*0.5 + 2)" << std::endl;
            std::cout << "Error: " << std::abs(grad - 3.0) << std::endl;
        } else {
            std::cout << "ERROR: No gradient computed!" << std::endl;
        }
    }

    // Compare with Float32
    {
        std::cout << "\n=== Test with x = 1.0 (Float32 for comparison) ===" << std::endl;

        Tensor data = zeros({1}, DType::Float32, device);
        data.data<float>()[0] = 1.0f;

        Variable x(data, true);
        Variable y = x * x + x * 2.0f;

        std::cout << "x = " << x.tensor().item<float>() << std::endl;
        std::cout << "y = " << y.tensor().item<float>() << std::endl;

        y.backward();

        if (x.has_grad()) {
            float grad = x.grad().value().item<float>();
            std::cout << "Analytical gradient: " << grad << std::endl;
            std::cout << "Expected gradient: 4.0 (2*1 + 2)" << std::endl;
            std::cout << "Error: " << std::abs(grad - 4.0f) << std::endl;
        } else {
            std::cout << "ERROR: No gradient computed!" << std::endl;
        }
    }

    return 0;
}
