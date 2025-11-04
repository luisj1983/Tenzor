#include "tenzor/tenzor.hpp"
#include <iostream>
#include <exception>

int main() {
    std::cout << "=== Backend Availability Diagnostic ===\n\n";

    // Initialize Tenzor
    tenzor::initialize();
    std::cout << "Tenzor initialized successfully\n\n";

    // Test CPU Backend
    std::cout << "--- CPU Backend ---\n";
    try {
        tenzor::Device cpu_dev = tenzor::Device::cpu();
        auto cpu_tensor = tenzor::zeros({2, 2}, tenzor::DType::Float32, cpu_dev);
        std::cout << "✓ CPU backend works! Created tensor with " << cpu_tensor.numel() << " elements\n";
    } catch (const std::exception& e) {
        std::cout << "✗ CPU backend failed: " << e.what() << "\n";
    }
    std::cout << "\n";

    // Test CUDA Backend
    std::cout << "--- CUDA Backend ---\n";
    try {
        tenzor::Device cuda_dev{tenzor::Device::Type::CUDA, 0};
        auto cuda_tensor = tenzor::zeros({2, 2}, tenzor::DType::Float32, cuda_dev);
        std::cout << "✓ CUDA backend works! Created tensor with " << cuda_tensor.numel() << " elements\n";
    } catch (const std::exception& e) {
        std::cout << "✗ CUDA backend failed: " << e.what() << "\n";
    }
    std::cout << "\n";

    // Test Vulkan Backend
    std::cout << "--- Vulkan Backend ---\n";
    try {
        tenzor::Device vulkan_dev{tenzor::Device::Type::Vulkan, 0};
        std::cout << "  Device created: " << vulkan_dev.to_string() << "\n";

        auto vulkan_tensor = tenzor::zeros({2, 2}, tenzor::DType::Float32, vulkan_dev);
        std::cout << "✓ Vulkan backend works! Created tensor with " << vulkan_tensor.numel() << " elements\n";
    } catch (const std::exception& e) {
        std::cout << "✗ Vulkan backend failed: " << e.what() << "\n";
    }
    std::cout << "\n";

    // Test OneAPI Backend
    std::cout << "--- OneAPI Backend ---\n";
    try {
        tenzor::Device oneapi_dev{tenzor::Device::Type::OneAPI, 0};
        std::cout << "  Device created: " << oneapi_dev.to_string() << "\n";

        auto oneapi_tensor = tenzor::zeros({2, 2}, tenzor::DType::Float32, oneapi_dev);
        std::cout << "✓ OneAPI backend works! Created tensor with " << oneapi_tensor.numel() << " elements\n";
    } catch (const std::exception& e) {
        std::cout << "✗ OneAPI backend failed: " << e.what() << "\n";
    }
    std::cout << "\n";

    std::cout << "=== Diagnostic Complete ===\n";
    return 0;
}
