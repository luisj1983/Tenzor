#include <iostream>
#include <cmath>
#include <vector>

// Minimal test without gtest to avoid compilation issues
#include "tenzor/tenzor.hpp"

using namespace tenzor;

bool test_square_matmul() {
    try {
        auto device = Device::oneapi(0);
        
        // Create 3x3 matrices
        auto a = randn({3, 3}, DType::Float32, device);
        auto b = randn({3, 3}, DType::Float32, device);
        
        // Compute matmul on OneAPI
        auto c_oneapi = matmul(a, b);
        
        // Copy to CPU and compute reference
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto c_cpu = matmul(a_cpu, b_cpu);
        
        // Copy OneAPI result to CPU for comparison
        auto c_oneapi_cpu = c_oneapi.to(Device::cpu());
        
        // Check shapes
        auto shape_oneapi = c_oneapi_cpu.shape();
        auto shape_cpu = c_cpu.shape();
        if (shape_oneapi.size() != shape_cpu.size()) {
            std::cerr << "Shape size mismatch!" << std::endl;
            return false;
        }
        
        // Compare results
        const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
        const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());
        
        int mismatches = 0;
        for (int64_t i = 0; i < c_cpu.numel(); ++i) {
            float diff = std::abs(c_oneapi_data[i] - c_cpu_data[i]);
            if (diff > 1e-4f) {
                if (mismatches < 5) {
                    std::cerr << "Mismatch at index " << i << ": OneAPI=" << c_oneapi_data[i]
                              << ", CPU=" << c_cpu_data[i] << ", diff=" << diff << std::endl;
                }
                mismatches++;
            }
        }
        
        if (mismatches > 0) {
            std::cerr << "Total mismatches: " << mismatches << " / " << c_cpu.numel() << std::endl;
            return false;
        }
        
        std::cout << "✓ Square 3x3 MatMul: PASSED" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Square MatMul failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_rectangular_matmul() {
    try {
        auto device = Device::oneapi(0);
        
        // Create matrices: A (4x5), B (5x3), C (4x3)
        auto a = randn({4, 5}, DType::Float32, device);
        auto b = randn({5, 3}, DType::Float32, device);
        
        auto c_oneapi = matmul(a, b);
        
        // Reference on CPU
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto c_cpu = matmul(a_cpu, b_cpu);
        
        auto c_oneapi_cpu = c_oneapi.to(Device::cpu());
        
        // Compare
        const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
        const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());
        
        int mismatches = 0;
        for (int64_t i = 0; i < c_cpu.numel(); ++i) {
            float diff = std::abs(c_oneapi_data[i] - c_cpu_data[i]);
            if (diff > 1e-4f) {
                if (mismatches < 5) {
                    std::cerr << "Mismatch at index " << i << ": OneAPI=" << c_oneapi_data[i]
                              << ", CPU=" << c_cpu_data[i] << ", diff=" << diff << std::endl;
                }
                mismatches++;
            }
        }
        
        if (mismatches > 0) {
            std::cerr << "Total mismatches: " << mismatches << " / " << c_cpu.numel() << std::endl;
            return false;
        }
        
        std::cout << "✓ Rectangular 4x5 @ 5x3 MatMul: PASSED" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Rectangular MatMul failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_large_matmul() {
    try {
        auto device = Device::oneapi(0);
        
        // Create matrices: A (64x128), B (128x32), C (64x32)
        auto a = randn({64, 128}, DType::Float32, device);
        auto b = randn({128, 32}, DType::Float32, device);
        
        auto c_oneapi = matmul(a, b);
        
        // Reference on CPU
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto c_cpu = matmul(a_cpu, b_cpu);
        
        auto c_oneapi_cpu = c_oneapi.to(Device::cpu());
        
        // Compare with relative error
        const float* c_oneapi_data = static_cast<const float*>(c_oneapi_cpu.data_ptr());
        const float* c_cpu_data = static_cast<const float*>(c_cpu.data_ptr());
        
        int mismatches = 0;
        for (int64_t i = 0; i < c_cpu.numel(); ++i) {
            float rel_error = std::abs(c_oneapi_data[i] - c_cpu_data[i]) /
                             (std::abs(c_cpu_data[i]) + 1e-8f);
            if (rel_error > 1e-3f) {
                if (mismatches < 5) {
                    std::cerr << "Mismatch at index " << i << ": OneAPI=" << c_oneapi_data[i]
                              << ", CPU=" << c_cpu_data[i] << ", rel_error=" << rel_error << std::endl;
                }
                mismatches++;
            }
        }
        
        if (mismatches > 0) {
            std::cerr << "Total mismatches: " << mismatches << " / " << c_cpu.numel() << std::endl;
            return false;
        }
        
        std::cout << "✓ Large 64x128 @ 128x32 MatMul: PASSED" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Large MatMul failed: " << e.what() << std::endl;
        return false;
    }
}

bool test_float64_matmul() {
    try {
        auto device = Device::oneapi(0);
        
        // Create 3x3 double matrices
        auto a = randn({3, 3}, DType::Float64, device);
        auto b = randn({3, 3}, DType::Float64, device);
        
        auto c_oneapi = matmul(a, b);
        
        // Reference on CPU
        auto a_cpu = a.to(Device::cpu());
        auto b_cpu = b.to(Device::cpu());
        auto c_cpu = matmul(a_cpu, b_cpu);
        
        auto c_oneapi_cpu = c_oneapi.to(Device::cpu());
        
        // Compare
        const double* c_oneapi_data = static_cast<const double*>(c_oneapi_cpu.data_ptr());
        const double* c_cpu_data = static_cast<const double*>(c_cpu.data_ptr());
        
        int mismatches = 0;
        for (int64_t i = 0; i < c_cpu.numel(); ++i) {
            double diff = std::abs(c_oneapi_data[i] - c_cpu_data[i]);
            if (diff > 1e-10) {
                if (mismatches < 5) {
                    std::cerr << "Mismatch at index " << i << ": OneAPI=" << c_oneapi_data[i]
                              << ", CPU=" << c_cpu_data[i] << ", diff=" << diff << std::endl;
                }
                mismatches++;
            }
        }
        
        if (mismatches > 0) {
            std::cerr << "Total mismatches: " << mismatches << " / " << c_cpu.numel() << std::endl;
            return false;
        }
        
        std::cout << "✓ Float64 3x3 MatMul: PASSED" << std::endl;
        return true;
    } catch (const std::exception& e) {
        std::cerr << "✗ Float64 MatMul failed: " << e.what() << std::endl;
        return false;
    }
}

int main() {
    std::cout << "===== OneAPI MatMul Tests =====" << std::endl;
    
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }
    
    // Check if OneAPI is available
    try {
        auto device = Device::oneapi(0);
        auto test_tensor = ones({2, 2}, DType::Float32, device);
    } catch (...) {
        std::cerr << "OneAPI backend not available, skipping tests" << std::endl;
        return 0;
    }
    
    int passed = 0;
    int total = 0;
    
    total++; if (test_square_matmul()) passed++;
    total++; if (test_rectangular_matmul()) passed++;
    total++; if (test_large_matmul()) passed++;
    total++; if (test_float64_matmul()) passed++;
    
    std::cout << "\n===== Results: " << passed << " / " << total << " tests passed =====" << std::endl;
    
    return (passed == total) ? 0 : 1;
}
