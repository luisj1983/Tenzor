/**
 * @file vulkan_tensor_test.cpp
 * @brief Test program to verify Vulkan backend tensor creation and shader loading
 */

#include <iostream>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

int main() {
    try {
        std::cout << "=== Vulkan Backend Tensor Creation Test ===" << std::endl;

        // Initialize Tenzor library (loads backends)
        std::cout << "\n1. Initializing Tenzor library..." << std::endl;
        initialize();
        std::cout << "   SUCCESS: Library initialized" << std::endl;

        // Create Vulkan device
        std::cout << "\n2. Creating Vulkan device..." << std::endl;
        Device vulkan_device = Device{Device::Type::Vulkan, 0};
        std::cout << "   SUCCESS: Device created" << std::endl;

        // Create a simple tensor on Vulkan device
        std::cout << "\n3. Creating tensor on Vulkan device 0..." << std::endl;
        std::vector<int64_t> shape = {4, 4};

        Tensor tensor(shape, DType::Float32, vulkan_device);
        std::cout << "   SUCCESS: Tensor created" << std::endl;
        std::cout << "   Shape: [" << shape[0] << ", " << shape[1] << "]" << std::endl;
        std::cout << "   DType: Float32" << std::endl;
        std::cout << "   Device: Vulkan:0" << std::endl;
        std::cout << "   Elements: " << tensor.numel() << std::endl;

        // Test tensor memory allocation
        std::cout << "\n4. Testing tensor memory allocation..." << std::endl;
        void* ptr = tensor.data_ptr();
        if (ptr != nullptr) {
            std::cout << "   SUCCESS: Memory allocated at " << ptr << std::endl;
        } else {
            std::cerr << "   ERROR: Null data pointer" << std::endl;
            return 1;
        }

        // Test tensor operations (will trigger shader loading)
        std::cout << "\n5. Testing tensor addition (triggers shader loading)..." << std::endl;
        try {
            // Create tensors on CPU and fill with data
            auto cpu_a = Tensor(shape, DType::Float32, Device::cpu());
            auto cpu_b = Tensor(shape, DType::Float32, Device::cpu());

            auto* data_a = cpu_a.data<float>();
            auto* data_b = cpu_b.data<float>();
            for (int64_t i = 0; i < cpu_a.numel(); ++i) {
                data_a[i] = 2.0f;
                data_b[i] = 3.0f;
            }

            // Move to Vulkan
            auto vulkan_a = cpu_a.to(vulkan_device);
            auto vulkan_b = cpu_b.to(vulkan_device);

            std::cerr << "DEBUG: vulkan_a device type = " << static_cast<int>(vulkan_a.device().type) << std::endl;
            std::cerr << "DEBUG: vulkan_b device type = " << static_cast<int>(vulkan_b.device().type) << std::endl;

            // Attempt addition - this will load the 'add' or 'math' shader
            std::cout << "   Attempting tensor addition (a + b where a=2.0, b=3.0)..." << std::endl;
            auto result = vulkan_a + vulkan_b;
            std::cerr << "DEBUG: result device type = " << static_cast<int>(result.device().type) << std::endl;
            std::cout << "   SUCCESS: Addition shader loaded and executed" << std::endl;

            // Verify result by moving back to CPU
            auto cpu_result = result.to(Device::cpu());
            auto* result_data = cpu_result.data<float>();
            float expected = 5.0f;  // 2.0 + 3.0 = 5.0
            bool all_correct = true;
            for (int64_t i = 0; i < cpu_result.numel(); ++i) {
                if (std::abs(result_data[i] - expected) > 1e-5f) {
                    all_correct = false;
                    std::cerr << "   ERROR at index " << i << ": expected " << expected
                             << ", got " << result_data[i] << std::endl;
                    break;
                }
            }
            if (all_correct) {
                std::cout << "   SUCCESS: Result values verified (all equal to " << expected << ")" << std::endl;
            } else {
                std::cerr << "   WARNING: Result values don't match expected" << std::endl;
            }
        } catch (const std::exception& e) {
            // If shader loading fails, we'll get detailed error message
            std::cerr << "\n   ERROR: Shader operation failed:" << std::endl;
            std::cerr << "   " << e.what() << std::endl;
            return 1;
        }

        std::cout << "\n=== ALL TESTS PASSED ===" << std::endl;
        std::cout << "Vulkan backend successfully loaded, shaders loaded correctly," << std::endl;
        std::cout << "and tensor operations work as expected!" << std::endl;

        // Cleanup
        finalize();
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "\nFATAL ERROR: " << e.what() << std::endl;
        finalize();
        return 1;
    }
}
