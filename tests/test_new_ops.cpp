#include <iostream>
#include "tenzor/core/tensor.hpp"
#include "tenzor/backends/vulkan/vulkan_backend.hpp"

int main() {
    try {
        tenzor::VulkanBackend backend;

        // Test repeat operation
        std::cout << "Testing dispatchRepeat..." << std::endl;
        std::vector<int64_t> shape = {2, 3};
        tenzor::Tensor input(shape, tenzor::DType::Float32, tenzor::Device{tenzor::DeviceType::Vulkan, 0});
        std::vector<int64_t> repeats = {2, 3};
        auto result_repeat = backend.dispatchRepeat(input, repeats);
        std::cout << "Repeat result shape: [";
        for (size_t i = 0; i < result_repeat.shape().size(); ++i) {
            std::cout << result_repeat.shape()[i];
            if (i < result_repeat.shape().size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;

        // Test roll operation
        std::cout << "\nTesting dispatchRoll..." << std::endl;
        auto result_roll = backend.dispatchRoll(input, 1, 0);
        std::cout << "Roll succeeded" << std::endl;

        // Test dot operation
        std::cout << "\nTesting dispatchDot..." << std::endl;
        std::vector<int64_t> vec_shape = {5};
        tenzor::Tensor vec_a(vec_shape, tenzor::DType::Float32, tenzor::Device{tenzor::DeviceType::Vulkan, 0});
        tenzor::Tensor vec_b(vec_shape, tenzor::DType::Float32, tenzor::Device{tenzor::DeviceType::Vulkan, 0});
        auto result_dot = backend.dispatchDot(vec_a, vec_b);
        std::cout << "Dot product succeeded" << std::endl;

        std::cout << "\nAll tests passed!" << std::endl;
        return 0;
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}
