// Minimal test case to reproduce Vulkan GPU hang with command batching
//
// This test uses the standard Tenzor initialization but adds verbose debugging.
// For RenderDoc, you may need to disable shaderFloat64 in vulkan_backend.cpp
// or use NVIDIA Nsight Graphics instead.

#include <tenzor/tenzor.hpp>
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include <iostream>
#include <cstdlib>

using namespace tenzor;

int main() {
    // Enable Vulkan validation layers for debugging
    setenv("VK_INSTANCE_LAYERS", "VK_LAYER_KHRONOS_validation", 1);

    std::cout << "=== Vulkan GPU Hang Debug Test ===" << std::endl;
    std::cout << "Initializing Tenzor library..." << std::endl;
    initialize();  // Load all backends including Vulkan

    std::cout << "\nCreating Vulkan device..." << std::endl;
    Device device = Device::vulkan(0);

    // Enable gradient computation
    set_grad_enabled(true);

    std::cout << "\n--- Test: Permute Backward Pass ---" << std::endl;
    std::cout << "This test reproduces the GPU hang that occurs during" << std::endl;
    std::cout << "the backward pass of permute when command batching is enabled." << std::endl;

    // Create input tensor
    std::cout << "\nStep 1: Creating input tensor [2, 3, 4]..." << std::endl;
    auto data = ones({2, 3, 4}, DType::Float32, device);
    Variable x(data, true);
    std::cout << "  Input tensor created" << std::endl;

    // Permute
    std::cout << "\nStep 2: Permuting with dims [2, 0, 1]..." << std::endl;
    auto y = permute(x, {2, 0, 1});
    std::cout << "  Permuted shape: [" << y.shape()[0] << ", " << y.shape()[1] << ", " << y.shape()[2] << "]" << std::endl;

    // Sum (reduction)
    std::cout << "\nStep 3: Computing sum reduction..." << std::endl;
    auto loss = sum(y);
    std::cout << "  Sum computed" << std::endl;

    // Backward pass - this is where the GPU hang occurs
    std::cout << "\nStep 4: Running backward pass..." << std::endl;
    std::cout << "  (GPU hang typically occurs during permute backward -> contiguous -> strided_copy)" << std::endl;
    std::cout << "  >>> Starting backward() <<<" << std::endl;
    loss.backward();
    std::cout << "  >>> backward() completed <<<" << std::endl;

    // Check gradient - may trigger another hang when accessing GPU data
    std::cout << "\nStep 5: Checking gradient..." << std::endl;
    if (x.grad().has_value()) {
        std::cout << "  Gradient exists, transferring to CPU..." << std::endl;
        auto grad = x.grad().value().to(Device::cpu());
        std::cout << "  Gradient shape: [" << grad.shape()[0] << ", "
                  << grad.shape()[1] << ", " << grad.shape()[2] << "]" << std::endl;
        std::cout << "  First gradient value: " << grad.data<float>()[0] << std::endl;
    } else {
        std::cout << "  WARNING: No gradient computed" << std::endl;
    }

    std::cout << "\n=== Test PASSED ===" << std::endl;
    return 0;
}
