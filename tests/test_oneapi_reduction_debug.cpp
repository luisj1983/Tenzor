#include <iostream>
#include <sycl/sycl.hpp>
#include "tenzor/tenzor.hpp"

using namespace tenzor;

int main() {
    // Initialize Tenzor library (loads all backends)
    tenzor::init();

    std::cout << "OneAPI Reduction Debug Test\n";
    std::cout << "============================\n\n";

    // Test 1: Simple ones tensor sum
    {
        std::cout << "Test 1: Sum of ones tensor [1, 1, 1, 1]\n";
        auto test = ones({4}, DType::Float32, Device::oneapi(0));

        // Print input values
        auto test_cpu = test.cpu();
        std::cout << "Input values: ";
        const float* data = test_cpu.data<float>();
        for (int i = 0; i < 4; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";

        // Compute sum
        auto result = tenzor::sum(test);
        auto result_cpu = result.cpu();
        float sum_value = result_cpu.data<float>()[0];

        std::cout << "Sum result: " << sum_value << " (expected: 4.0)\n";
        std::cout << "Test 1: " << (std::abs(sum_value - 4.0f) < 0.001f ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 2: Simple tensor with value 2.0
    {
        std::cout << "Test 2: Sum of twos tensor [2, 2, 2, 2]\n";
        auto test = full({4}, 2.0f, DType::Float32, Device::oneapi(0));

        // Print input values
        auto test_cpu = test.cpu();
        std::cout << "Input values: ";
        const float* data = test_cpu.data<float>();
        for (int i = 0; i < 4; i++) {
            std::cout << data[i] << " ";
        }
        std::cout << "\n";

        // Compute sum
        auto result = tenzor::sum(test);
        auto result_cpu = result.cpu();
        float sum_value = result_cpu.data<float>()[0];

        std::cout << "Sum result: " << sum_value << " (expected: 8.0)\n";
        std::cout << "Test 2: " << (std::abs(sum_value - 8.0f) < 0.001f ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 3: Mean of ones
    {
        std::cout << "Test 3: Mean of ones tensor [1, 1, 1, 1]\n";
        auto test = ones({4}, DType::Float32, Device::oneapi(0));

        auto result = tenzor::mean(test);
        auto result_cpu = result.cpu();
        float mean_value = result_cpu.data<float>()[0];

        std::cout << "Mean result: " << mean_value << " (expected: 1.0)\n";
        std::cout << "Test 3: " << (std::abs(mean_value - 1.0f) < 0.001f ? "PASS" : "FAIL") << "\n\n";
    }

    // Test 4: Direct memory access test
    {
        std::cout << "Test 4: Direct memory access check\n";
        auto test = full({6}, 3.0f, DType::Float32, Device::oneapi(0));

        // Verify by copying to CPU
        auto test_cpu = test.cpu();
        std::cout << "CPU copy values: ";
        const float* cpu_data = test_cpu.data<float>();
        for (int i = 0; i < 6; i++) {
            std::cout << cpu_data[i] << " ";
        }
        std::cout << "\n";

        // Now check device pointer directly (since we use malloc_shared)
        std::cout << "Direct device pointer check: ";
        const float* dev_data = test.data<float>();
        for (int i = 0; i < 6; i++) {
            std::cout << dev_data[i] << " ";
        }
        std::cout << "\n\n";
    }

    std::cout << "Debug test complete\n";
    return 0;
}
