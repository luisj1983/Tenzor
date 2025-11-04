/**
 * @file vulkan_diagnostic.cpp
 * @brief Diagnostic test to identify Vulkan buffer issues
 */

#include <iostream>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

int main() {
    try {
        std::cout << "=== Vulkan Diagnostic Test ===" << std::endl;

        // Initialize
        std::cout << "\n1. Initializing..." << std::endl;
        initialize();
        std::cout << "   OK" << std::endl;

        // Test creating device
        std::cout << "\n2. Creating Vulkan device..." << std::endl;
        Device vulkan_device = Device::vulkan(0);
        std::cout << "   OK - Device: " << vulkan_device.to_string() << std::endl;

        // Test creating tensor
        std::cout << "\n3. Creating tensor on Vulkan device..." << std::endl;
        std::vector<int64_t> shape = {4, 4};
        Tensor tensor(shape, DType::Float32, vulkan_device);
        std::cout << "   OK - Created tensor with " << tensor.numel() << " elements" << std::endl;
        std::cout << "   Data pointer: " << tensor.data_ptr() << std::endl;

        // Test filling tensor on CPU and copying to Vulkan
        std::cout << "\n4. Creating CPU tensor with data..." << std::endl;
        auto cpu_tensor = Tensor(shape, DType::Float32, Device::cpu());
        auto* cpu_data = cpu_tensor.data<float>();
        for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
            cpu_data[i] = 2.0f;
        }
        std::cout << "   OK - CPU tensor filled with 2.0" << std::endl;

        // Test copying to Vulkan
        std::cout << "\n5. Copying data to Vulkan device..." << std::endl;
        try {
            auto vulkan_tensor = cpu_tensor.to(vulkan_device);
            std::cout << "   OK - Data copied to Vulkan" << std::endl;
            std::cout << "   Vulkan tensor data pointer: " << vulkan_tensor.data_ptr() << std::endl;

            // Test copying back to CPU
            std::cout << "\n6. Copying data back to CPU..." << std::endl;
            auto result_tensor = vulkan_tensor.to(Device::cpu());
            std::cout << "   OK - Data copied back to CPU" << std::endl;

            // Verify data
            std::cout << "\n7. Verifying data..." << std::endl;
            auto* result_data = result_tensor.data<float>();
            bool all_correct = true;
            for (int64_t i = 0; i < result_tensor.numel(); ++i) {
                if (std::abs(result_data[i] - 2.0f) > 1e-5f) {
                    std::cerr << "   ERROR at index " << i << ": expected 2.0, got " << result_data[i] << std::endl;
                    all_correct = false;
                    break;
                }
            }
            if (all_correct) {
                std::cout << "   OK - All values correct (2.0)" << std::endl;
            } else {
                std::cerr << "   WARNING: Some values incorrect" << std::endl;
            }
        } catch (const std::exception& e) {
            std::cerr << "   ERROR: Transfer failed: " << e.what() << std::endl;
            finalize();
            return 1;
        }

        std::cout << "\n=== ALL DIAGNOSTICS PASSED ===" << std::endl;
        finalize();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        finalize();
        return 1;
    }
}
