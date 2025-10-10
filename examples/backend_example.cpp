#include <tenzor/tenzor.hpp>
#include <iostream>

int main() {
    using namespace tenzor;

    // Initialize Tenzor library (loads CPU backend by default)
    initialize();

    std::cout << "\nTenzor Backend Selection Example\n";
    std::cout << "==================================\n\n";

    // ===================================================================
    // Method 1: Specify device when creating tensors
    // ===================================================================

    std::cout << "1. Creating tensors on different devices:\n";
    std::cout << "   ----------------------------------------\n";

    // CPU tensors (default)
    auto cpu_tensor1 = zeros({3, 4}, DType::Float32, Device::cpu());
    auto cpu_tensor2 = ones({3, 4});  // Device::cpu() is default

    std::cout << "   CPU tensor created: " << cpu_tensor1.device().to_string() << "\n";

    // CUDA tensors (if available)
    try {
        auto cuda_tensor = zeros({3, 4}, DType::Float32, Device::cuda(0));
        std::cout << "   CUDA tensor created: " << cuda_tensor.device().to_string() << "\n";
    } catch (const std::exception& e) {
        std::cout << "   CUDA backend not available: " << e.what() << "\n";
    }

    // Multiple GPU support
    try {
        auto gpu0_tensor = zeros({2, 2}, DType::Float32, Device::cuda(0));
        auto gpu1_tensor = zeros({2, 2}, DType::Float32, Device::cuda(1));
        std::cout << "   Multi-GPU tensors created\n";
    } catch (const std::exception& e) {
        std::cout << "   Multi-GPU not available\n";
    }

    std::cout << "\n";

    // ===================================================================
    // Method 2: All creation functions support device parameter
    // ===================================================================

    std::cout << "2. Different creation functions with device:\n";
    std::cout << "   ------------------------------------------\n";

    auto cpu_zeros = zeros({2, 3}, DType::Float32, Device::cpu());
    auto cpu_ones = ones({2, 3}, DType::Float32, Device::cpu());
    auto cpu_rand = rand({2, 3}, DType::Float32, Device::cpu());
    auto cpu_randn = randn({2, 3}, DType::Float32, Device::cpu());
    auto cpu_full = full({2, 3}, 5.0f, DType::Float32, Device::cpu());
    auto cpu_arange = arange(0.0f, 10.0f, 1.0f, DType::Float32, Device::cpu());
    auto cpu_linspace = linspace(0.0f, 1.0f, 10, DType::Float32, Device::cpu());
    auto cpu_eye = eye(3, std::nullopt, DType::Float32, Device::cpu());

    std::cout << "   ✓ zeros, ones, rand, randn\n";
    std::cout << "   ✓ full, arange, linspace, eye\n";
    std::cout << "   All created on: " << cpu_zeros.device().to_string() << "\n";

    std::cout << "\n";

    // ===================================================================
    // Method 3: Operations automatically use tensor's device
    // ===================================================================

    std::cout << "3. Operations follow tensor device:\n";
    std::cout << "   ---------------------------------\n";

    auto x = randn({3, 3}, DType::Float32, Device::cpu());
    auto y = randn({3, 3}, DType::Float32, Device::cpu());

    // Operations use the device of input tensors
    auto sum = x + y;           // Computed on CPU
    auto product = x * y;       // Computed on CPU
    auto mm = matmul(x, y);     // Computed on CPU

    std::cout << "   Input tensors device: " << x.device().to_string() << "\n";
    std::cout << "   Result tensors device: " << sum.device().to_string() << "\n";
    std::cout << "   ✓ Element-wise ops: add, sub, mul, div\n";
    std::cout << "   ✓ Matrix ops: matmul\n";
    std::cout << "   ✓ Reduction ops: sum, mean, max, min\n";
    std::cout << "   ✓ Math ops: exp, log, sqrt, pow\n";
    std::cout << "   ✓ Activations: relu, sigmoid, tanh, softmax\n";

    std::cout << "\n";

    // ===================================================================
    // Method 4: Neural network modules use input tensor device
    // ===================================================================

    std::cout << "4. Neural network modules:\n";
    std::cout << "   ------------------------\n";

    // Create model (device-agnostic)
    auto linear = nn::Linear(10, 5);

    // Forward pass uses input device
    auto cpu_input = Variable(randn({32, 10}, DType::Float32, Device::cpu()), true);
    auto cpu_output = linear.forward(cpu_input);

    std::cout << "   Model created (device-agnostic)\n";
    std::cout << "   Input device:  " << cpu_input.device().to_string() << "\n";
    std::cout << "   Output device: " << cpu_output.device().to_string() << "\n";

    // For CUDA, you would create input on CUDA:
    // auto cuda_input = Variable(randn({32, 10}, DType::Float32, Device::cuda(0)), true);
    // auto cuda_output = linear.forward(cuda_input);  // Computed on CUDA

    std::cout << "\n";

    // ===================================================================
    // Method 5: Device object methods
    // ===================================================================

    std::cout << "5. Device object API:\n";
    std::cout << "   -------------------\n";

    auto device_cpu = Device::cpu();
    auto device_cuda0 = Device::cuda(0);
    auto device_cuda1 = Device::cuda(1);
    auto device_rocm = Device::rocm(0);

    std::cout << "   CPU:      " << device_cpu.to_string() << "\n";
    std::cout << "   CUDA:0:   " << device_cuda0.to_string() << "\n";
    std::cout << "   CUDA:1:   " << device_cuda1.to_string() << "\n";
    std::cout << "   ROCm:0:   " << device_rocm.to_string() << "\n";

    std::cout << "\n";

    // ===================================================================
    // Summary
    // ===================================================================

    std::cout << "Summary:\n";
    std::cout << "========\n\n";
    std::cout << "Backend Selection Methods:\n";
    std::cout << "  1. Specify Device when creating tensors:\n";
    std::cout << "     auto t = zeros({3, 4}, DType::Float32, Device::cuda(0));\n\n";
    std::cout << "  2. Operations automatically use input tensor's device\n\n";
    std::cout << "  3. Available backends:\n";
    std::cout << "     - CPU (always available)\n";
    std::cout << "     - CUDA (if compiled with CUDA support)\n";
    std::cout << "     - ROCm (if compiled with ROCm support)\n\n";
    std::cout << "  4. Backend is loaded dynamically at runtime:\n";
    std::cout << "     - CPU:  tenzor_backend_cpu.so\n";
    std::cout << "     - CUDA: tenzor_backend_cuda.so\n";
    std::cout << "     - ROCm: tenzor_backend_rocm.so\n\n";

    std::cout << "Example complete!\n\n";

    return 0;
}
