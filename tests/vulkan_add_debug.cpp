/**
 * @file vulkan_add_debug.cpp
 * @brief Debug Vulkan addition operation specifically
 */

#include <iostream>
#include <iomanip>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

int main() {
    try {
        std::cout << "=== Vulkan Addition Debug Test ===" << std::endl;

        // Initialize
        std::cout << "\n1. Initializing..." << std::endl;
        initialize();

        // Create tensors
        Device vulkan_dev = Device::vulkan(0);
        Device cpu_dev = Device::cpu();

        std::cout << "\n2. Creating CPU tensors..." << std::endl;
        auto a_cpu = zeros({4}, DType::Float32, cpu_dev);
        auto b_cpu = zeros({4}, DType::Float32, cpu_dev);

        auto* a_data = a_cpu.data<float>();
        auto* b_data = b_cpu.data<float>();

        for (int i = 0; i < 4; i++) {
            a_data[i] = 2.0f;
            b_data[i] = 3.0f;
        }

        std::cout << "   A = [";
        for (int i = 0; i < 4; i++) std::cout << a_data[i] << (i < 3 ? ", " : "]\n");
        std::cout << "   B = [";
        for (int i = 0; i < 4; i++) std::cout << b_data[i] << (i < 3 ? ", " : "]\n");

        // Move to Vulkan
        std::cout << "\n3. Copying to Vulkan device..." << std::endl;
        auto a_vulkan = a_cpu.to(vulkan_dev);
        auto b_vulkan = b_cpu.to(vulkan_dev);
        std::cout << "   a_vulkan ptr: " << a_vulkan.data_ptr() << std::endl;
        std::cout << "   b_vulkan ptr: " << b_vulkan.data_ptr() << std::endl;

        // Verify copy worked
        std::cout << "\n4. Verifying Vulkan->CPU transfer..." << std::endl;
        auto a_verify = a_vulkan.to(cpu_dev);
        auto b_verify = b_vulkan.to(cpu_dev);

        auto* a_verify_data = a_verify.data<float>();
        auto* b_verify_data = b_verify.data<float>();

        std::cout << "   A (copied back) = [";
        for (int i = 0; i < 4; i++) std::cout << a_verify_data[i] << (i < 3 ? ", " : "]\n");
        std::cout << "   B (copied back) = [";
        for (int i = 0; i < 4; i++) std::cout << b_verify_data[i] << (i < 3 ? ", " : "]\n");

        // Perform addition
        std::cout << "\n5. Performing Vulkan addition..." << std::endl;
        std::cout << "   Calling operator+..." << std::endl;
        auto result_vulkan = a_vulkan + b_vulkan;
        std::cout << "   result_vulkan ptr: " << result_vulkan.data_ptr() << std::endl;

        // Get result
        std::cout << "\n6. Copying result back to CPU..." << std::endl;
        auto result_cpu = result_vulkan.to(cpu_dev);
        auto* result_data = result_cpu.data<float>();

        std::cout << "   Result = [";
        for (int i = 0; i < 4; i++) std::cout << result_data[i] << (i < 3 ? ", " : "]\n");

        // Verify
        std::cout << "\n7. Verification..." << std::endl;
        bool all_correct = true;
        for (int i = 0; i < 4; i++) {
            float expected = 5.0f;
            if (std::abs(result_data[i] - expected) > 1e-5f) {
                std::cerr << "   ERROR at index " << i << ": expected " << expected
                         << ", got " << result_data[i] << std::endl;
                all_correct = false;
            }
        }

        if (all_correct) {
            std::cout << "   SUCCESS: All values correct!" << std::endl;
        } else {
            std::cerr << "   FAILURE: Some values incorrect!" << std::endl;
            finalize();
            return 1;
        }

        std::cout << "\n=== TEST PASSED ===" << std::endl;
        finalize();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        finalize();
        return 1;
    }
}
