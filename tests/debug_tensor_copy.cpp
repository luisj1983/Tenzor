#include <tenzor/tenzor.hpp>
#include <iostream>
#include <iomanip>

using namespace tenzor;

int main() {
    tenzor::initialize();

    std::cout << "=== Testing Tensor copy and modification with Float64 ===" << std::endl;

    // Create a Float64 tensor
    Tensor original = zeros({4}, DType::Float64, Device::cpu());
    double* data = original.data<double>();
    data[0] = 0.1;
    data[1] = 0.2;
    data[2] = 0.3;
    data[3] = 0.4;

    std::cout << "Original: [" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << "]" << std::endl;

    // Clone the tensor
    Tensor cloned = original.clone();

    std::cout << "\nAfter clone, modifying clone..." << std::endl;
    double* cloned_data = cloned.data<double>();
    cloned_data[0] += 0.001;
    cloned_data[1] += 0.001;

    std::cout << "Cloned: [" << cloned_data[0] << ", " << cloned_data[1] << ", " << cloned_data[2] << ", " << cloned_data[3] << "]" << std::endl;
    std::cout << "Original: [" << data[0] << ", " << data[1] << ", " << data[2] << ", " << data[3] << "]" << std::endl;

    // Now test creating a Variable from cloned
    std::cout << "\n=== Testing Variable creation from modified Tensor ===" << std::endl;
    Tensor test_tensor = original.clone();
    double* test_data = test_tensor.data<double>();
    test_data[0] += 1.0;  // Add 1.0 to first element

    std::cout << "After modifying: test_tensor[0] = " << test_data[0] << std::endl;

    Variable var(test_tensor, false);
    const double* var_data = var.tensor().to(Device::cpu()).data<double>();
    std::cout << "Variable tensor[0] = " << var_data[0] << std::endl;

    // Test the exact pattern from gradcheck
    std::cout << "\n=== Testing exact gradcheck pattern ===" << std::endl;
    Tensor input_copy = original.clone();
    Tensor x_plus_cpu = input_copy.clone();

    double* x_plus_data = x_plus_cpu.data<double>();
    std::cout << "Before perturbation: x_plus_cpu[0] = " << x_plus_data[0] << std::endl;

    x_plus_data[0] += 0.001;
    std::cout << "After perturbation: x_plus_cpu[0] = " << x_plus_data[0] << std::endl;

    Variable x_plus(x_plus_cpu, false);
    const double* x_plus_var_data = x_plus.tensor().data<double>();
    std::cout << "Variable x_plus tensor[0] = " << x_plus_var_data[0] << std::endl;

    return 0;
}
