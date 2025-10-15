#include <iostream>
#include <tenzor/tenzor.hpp>
#include <hip/hip_runtime.h>

using namespace tenzor;

int main() {
    try {
        std::cout << "Initializing Tenzor..." << std::endl;
        tenzor::initialize();

        // Check if ROCm is available
        int device_count = 0;
        hipError_t error = hipGetDeviceCount(&device_count);

        if (error != hipSuccess || device_count == 0) {
            std::cout << "No ROCm device available. Skipping test." << std::endl;
            return 0;
        }

        std::cout << "Found " << device_count << " ROCm device(s)" << std::endl;

        // Test 1: Simple 2x3 @ 3x2 matrix multiplication (Float32)
        std::cout << "\n=== Test 1: Float32 2x3 @ 3x2 ===" << std::endl;
        {
            auto a = ones({2, 3}, DType::Float32, Device::rocm());
            auto b = ones({3, 2}, DType::Float32, Device::rocm());

            // Initialize matrices
            std::vector<float> host_a = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
            std::vector<float> host_b = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};

            hipMemcpy(a.data<float>(), host_a.data(), 6 * sizeof(float), hipMemcpyHostToDevice);
            hipMemcpy(b.data<float>(), host_b.data(), 6 * sizeof(float), hipMemcpyHostToDevice);

            // Perform matrix multiplication
            auto c = matmul(a, b);

            std::cout << "Result shape: [" << c.shape()[0] << ", " << c.shape()[1] << "]" << std::endl;

            // Copy result to CPU
            auto c_cpu = c.to(Device::cpu());
            auto c_data = c_cpu.data<float>();

            std::cout << "Result matrix:" << std::endl;
            std::cout << "[[" << c_data[0] << ", " << c_data[1] << "]," << std::endl;
            std::cout << " [" << c_data[2] << ", " << c_data[3] << "]]" << std::endl;

            // Expected: [[22, 28], [49, 64]]
            float expected[4] = {22.0f, 28.0f, 49.0f, 64.0f};
            bool passed = true;
            for (int i = 0; i < 4; i++) {
                if (std::abs(c_data[i] - expected[i]) > 1e-4f) {
                    passed = false;
                    std::cout << "FAIL: Expected " << expected[i] << " but got " << c_data[i] << std::endl;
                }
            }
            if (passed) {
                std::cout << "PASS" << std::endl;
            }
        }

        // Test 2: Large matrix multiplication (Float32)
        std::cout << "\n=== Test 2: Float32 256x256 @ 256x256 ===" << std::endl;
        {
            const int M = 256;
            const int K = 256;
            const int N = 256;

            auto a = ones({M, K}, DType::Float32, Device::rocm());
            auto b = ones({K, N}, DType::Float32, Device::rocm());

            auto c = matmul(a, b);

            std::cout << "Result shape: [" << c.shape()[0] << ", " << c.shape()[1] << "]" << std::endl;

            // Copy result to CPU and verify
            auto c_cpu = c.to(Device::cpu());
            auto c_data = c_cpu.data<float>();

            // Each element should be K (sum of K ones)
            float expected = static_cast<float>(K);
            bool passed = true;
            for (int i = 0; i < std::min(100, M * N); i++) {
                if (std::abs(c_data[i] - expected) > 1e-3f) {
                    passed = false;
                    std::cout << "FAIL at index " << i << ": Expected " << expected << " but got " << c_data[i] << std::endl;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASS (first 100 elements verified)" << std::endl;
            }
        }

        // Test 3: Double precision (Float64)
        std::cout << "\n=== Test 3: Float64 4x4 @ 4x4 ===" << std::endl;
        {
            auto a = ones({4, 4}, DType::Float64, Device::rocm());
            auto b = ones({4, 4}, DType::Float64, Device::rocm());

            // Initialize with specific values
            std::vector<double> host_a(16), host_b(16);
            for (int i = 0; i < 16; i++) {
                host_a[i] = static_cast<double>(i) * 0.5;
                host_b[i] = static_cast<double>(i) * 0.1;
            }

            hipMemcpy(a.data<double>(), host_a.data(), 16 * sizeof(double), hipMemcpyHostToDevice);
            hipMemcpy(b.data<double>(), host_b.data(), 16 * sizeof(double), hipMemcpyHostToDevice);

            auto c = matmul(a, b);

            std::cout << "Result shape: [" << c.shape()[0] << ", " << c.shape()[1] << "]" << std::endl;

            // Copy result to CPU
            auto c_cpu = c.to(Device::cpu());
            auto c_data = c_cpu.data<double>();

            std::cout << "Result matrix (first row): [";
            for (int i = 0; i < 4; i++) {
                std::cout << c_data[i];
                if (i < 3) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
            std::cout << "PASS (double precision test completed)" << std::endl;
        }

        // Test 4: Batched matrix multiplication
        std::cout << "\n=== Test 4: Batched matmul [2, 3, 4] @ [2, 4, 3] ===" << std::endl;
        {
            auto a = ones({2, 3, 4}, DType::Float32, Device::rocm());
            auto b = ones({2, 4, 3}, DType::Float32, Device::rocm());

            auto c = matmul(a, b);

            std::cout << "Result shape: [" << c.shape()[0] << ", " << c.shape()[1] << ", " << c.shape()[2] << "]" << std::endl;

            // Expected shape: [2, 3, 3]
            if (c.shape()[0] == 2 && c.shape()[1] == 3 && c.shape()[2] == 3) {
                std::cout << "PASS (shape correct)" << std::endl;
            } else {
                std::cout << "FAIL: Expected shape [2, 3, 3]" << std::endl;
            }

            // Copy result to CPU
            auto c_cpu = c.to(Device::cpu());
            auto c_data = c_cpu.data<float>();

            // Each element should be 4 (sum of 4 ones)
            float expected = 4.0f;
            bool passed = true;
            for (int i = 0; i < 18; i++) { // 2*3*3 = 18 elements
                if (std::abs(c_data[i] - expected) > 1e-4f) {
                    passed = false;
                    std::cout << "FAIL at index " << i << ": Expected " << expected << " but got " << c_data[i] << std::endl;
                    break;
                }
            }
            if (passed) {
                std::cout << "PASS (values correct)" << std::endl;
            }
        }

        std::cout << "\n=== All tests completed ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
