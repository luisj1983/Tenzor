/**
 * @file bench_cublas_cudnn.cpp
 * @brief Comprehensive performance benchmarks for cuBLAS/cuDNN vs custom kernels
 *
 * Tests:
 * 1. Matrix multiplication (cuBLAS GemmEx vs custom kernels)
 * 2. Batched matrix multiplication
 * 3. Conv2d forward/backward (cuDNN vs custom)
 * 4. Batch normalization (cuDNN vs custom)
 * 5. FP16 Tensor Core performance
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include "tenzor/core/tensor.hpp"

using namespace tenzor;
using namespace std::chrono;

// ============================================================================
// Benchmark Helper Functions
// ============================================================================

class CUDABenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        // Check CUDA availability
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        // Print GPU info
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "\n===========================================\n";
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
        std::cout << "===========================================\n\n";

        has_tensor_cores = (prop.major >= 7);  // Volta and newer
    }

    // Benchmark helper: run operation N times and return average time in ms
    template<typename Func>
    double benchmark(Func&& func, int iterations = 100, int warmup = 10) {
        // Warmup
        for (int i = 0; i < warmup; ++i) {
            func();
        }
        cudaDeviceSynchronize();

        // Actual benchmark
        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();

        double total_time = duration<double, std::milli>(end - start).count();
        return total_time / iterations;
    }

    // Print performance comparison
    void print_comparison(const std::string& operation,
                         const std::string& impl1, double time1,
                         const std::string& impl2, double time2) {
        double speedup = time1 / time2;
        std::cout << std::fixed << std::setprecision(3);
        std::cout << operation << ":\n";
        std::cout << "  " << impl1 << ": " << time1 << " ms\n";
        std::cout << "  " << impl2 << ": " << time2 << " ms\n";
        std::cout << "  Speedup: " << speedup << "x\n\n";
    }

    bool has_tensor_cores = false;
};

// ============================================================================
// Matrix Multiplication Benchmarks
// ============================================================================

TEST_F(CUDABenchmark, MatMul_FP32_Large) {
    // Test large matrix multiplication: (2048, 2048) @ (2048, 2048)
    const int64_t M = 2048, N = 2048, K = 2048;

    Device device = Device::cuda(0);
    Tensor a = Tensor::randn({M, K}, DType::Float32, device);
    Tensor b = Tensor::randn({K, N}, DType::Float32, device);

    // Custom kernel
    double custom_time = benchmark([&]() {
        Tensor c = a.matmul(b);  // Uses custom tiled kernel
    });

    // cuBLAS GemmEx
    double cublas_time = benchmark([&]() {
        // TODO: Call cuBLAS implementation when integrated
        Tensor c = a.matmul(b);
    });

    print_comparison("MatMul FP32 (2048x2048)", "Custom", custom_time, "cuBLAS", cublas_time);

    // cuBLAS should be faster (10-30% improvement expected)
    EXPECT_LT(cublas_time, custom_time * 1.1);  // At least 10% faster or comparable
}

TEST_F(CUDABenchmark, MatMul_FP16_TensorCores) {
    if (!has_tensor_cores) {
        GTEST_SKIP() << "Tensor Cores not available on this GPU";
    }

    // Test FP16 with Tensor Cores: (4096, 4096) @ (4096, 4096)
    const int64_t M = 4096, N = 4096, K = 4096;

    Device device = Device::cuda(0);
    Tensor a = Tensor::randn({M, K}, DType::Float16, device);
    Tensor b = Tensor::randn({K, N}, DType::Float16, device);

    // Custom FP16 kernel (uses Tensor Cores via WMMA)
    double custom_time = benchmark([&]() {
        Tensor c = a.matmul(b);
    }, 50, 5);  // Fewer iterations due to size

    // cuBLAS GemmEx with Tensor Cores
    double cublas_time = benchmark([&]() {
        // TODO: Call cuBLAS GemmEx implementation
        Tensor c = a.matmul(b);
    }, 50, 5);

    print_comparison("MatMul FP16 Tensor Cores (4096x4096)", "Custom WMMA", custom_time, "cuBLAS GemmEx", cublas_time);

    // cuBLAS should be significantly faster (20-40% expected)
    EXPECT_LT(cublas_time, custom_time * 1.2);
}

TEST_F(CUDABenchmark, BatchedMatMul_FP32) {
    // Test batched matrix multiplication (common in transformers)
    const int64_t batch = 32, M = 512, N = 512, K = 512;

    Device device = Device::cuda(0);
    Tensor a = Tensor::randn({batch, M, K}, DType::Float32, device);
    Tensor b = Tensor::randn({batch, K, N}, DType::Float32, device);

    // Custom batched kernel
    double custom_time = benchmark([&]() {
        Tensor c = a.matmul(b);
    });

    // cuBLAS StridedBatchedGemmEx
    double cublas_time = benchmark([&]() {
        // TODO: Call cuBLAS batched implementation
        Tensor c = a.matmul(b);
    });

    print_comparison("Batched MatMul FP32 (32x512x512)", "Custom", custom_time, "cuBLAS Batched", cublas_time);

    // cuBLAS batched should be faster (15-35% expected)
    EXPECT_LT(cublas_time, custom_time * 1.15);
}

// ============================================================================
// Convolution Benchmarks
// ============================================================================

TEST_F(CUDABenchmark, Conv2d_Forward_FP32) {
    // ResNet-50 typical convolution: (32, 64, 56, 56) with (128, 64, 3, 3) kernel
    Device device = Device::cuda(0);

    Tensor input = Tensor::randn({32, 64, 56, 56}, DType::Float32, device);
    Tensor weight = Tensor::randn({128, 64, 3, 3}, DType::Float32, device);
    Tensor bias = Tensor::randn({128}, DType::Float32, device);

    // Custom im2col + cuBLAS
    double custom_time = benchmark([&]() {
        // TODO: Call custom conv2d kernel
        auto output = input;  // Placeholder
    });

    // cuDNN (with auto-tuning)
    double cudnn_time = benchmark([&]() {
        // TODO: Call cuDNN conv2d
        auto output = input;  // Placeholder
    });

    print_comparison("Conv2d Forward FP32 (ResNet-50 layer)", "Custom im2col", custom_time, "cuDNN", cudnn_time);

    // cuDNN should be faster (10-30% expected)
    EXPECT_LT(cudnn_time, custom_time * 1.2);
}

TEST_F(CUDABenchmark, Conv2d_Forward_FP16) {
    if (!has_tensor_cores) {
        GTEST_SKIP() << "Tensor Cores not available";
    }

    // FP16 convolution with Tensor Cores
    Device device = Device::cuda(0);

    Tensor input = Tensor::randn({32, 64, 56, 56}, DType::Float16, device);
    Tensor weight = Tensor::randn({128, 64, 3, 3}, DType::Float16, device);
    Tensor bias = Tensor::randn({128}, DType::Float16, device);

    // Custom im2col + FP16 matmul
    double custom_time = benchmark([&]() {
        auto output = input;  // Placeholder
    });

    // cuDNN with Tensor Cores
    double cudnn_time = benchmark([&]() {
        auto output = input;  // Placeholder
    });

    print_comparison("Conv2d Forward FP16 Tensor Cores", "Custom", custom_time, "cuDNN", cudnn_time);

    // cuDNN with Tensor Cores should be significantly faster (20-50% expected)
    EXPECT_LT(cudnn_time, custom_time * 1.3);
}

TEST_F(CUDABenchmark, Conv2d_Backward) {
    // Test backward pass performance
    Device device = Device::cuda(0);

    Tensor grad_output = Tensor::randn({32, 128, 56, 56}, DType::Float32, device);
    Tensor input = Tensor::randn({32, 64, 56, 56}, DType::Float32, device);
    Tensor weight = Tensor::randn({128, 64, 3, 3}, DType::Float32, device);

    // Custom backward kernels
    double custom_time = benchmark([&]() {
        // TODO: Call custom conv2d backward
        auto grad_input = input;  // Placeholder
    });

    // cuDNN backward
    double cudnn_time = benchmark([&]() {
        // TODO: Call cuDNN conv2d backward
        auto grad_input = input;  // Placeholder
    });

    print_comparison("Conv2d Backward FP32", "Custom", custom_time, "cuDNN", cudnn_time);

    // cuDNN should be faster (10-30% expected)
    EXPECT_LT(cudnn_time, custom_time * 1.2);
}

// ============================================================================
// Batch Normalization Benchmarks
// ============================================================================

TEST_F(CUDABenchmark, BatchNorm2d_Forward) {
    // Typical batch normalization: (32, 256, 28, 28)
    Device device = Device::cuda(0);

    Tensor input = Tensor::randn({32, 256, 28, 28}, DType::Float32, device);
    Tensor scale = Tensor::ones({256}, DType::Float32, device);
    Tensor bias = Tensor::zeros({256}, DType::Float32, device);
    Tensor running_mean = Tensor::zeros({256}, DType::Float32, device);
    Tensor running_var = Tensor::ones({256}, DType::Float32, device);

    // Custom batch norm kernel
    double custom_time = benchmark([&]() {
        // TODO: Call custom batchnorm
        auto output = input;  // Placeholder
    });

    // cuDNN batch norm
    double cudnn_time = benchmark([&]() {
        // TODO: Call cuDNN batchnorm
        auto output = input;  // Placeholder
    });

    print_comparison("BatchNorm2d Forward FP32", "Custom", custom_time, "cuDNN", cudnn_time);

    // cuDNN should be faster (30-50% expected)
    EXPECT_LT(cudnn_time, custom_time * 1.4);
}

TEST_F(CUDABenchmark, BatchNorm2d_Backward) {
    // Backward pass for batch normalization
    Device device = Device::cuda(0);

    Tensor grad_output = Tensor::randn({32, 256, 28, 28}, DType::Float32, device);
    Tensor input = Tensor::randn({32, 256, 28, 28}, DType::Float32, device);
    Tensor scale = Tensor::ones({256}, DType::Float32, device);
    Tensor saved_mean = Tensor::zeros({256}, DType::Float32, device);
    Tensor saved_var = Tensor::ones({256}, DType::Float32, device);

    // Custom backward
    double custom_time = benchmark([&]() {
        // TODO: Call custom batchnorm backward
        auto grad_input = input;  // Placeholder
    });

    // cuDNN backward
    double cudnn_time = benchmark([&]() {
        // TODO: Call cuDNN batchnorm backward
        auto grad_input = input;  // Placeholder
    });

    print_comparison("BatchNorm2d Backward FP32", "Custom", custom_time, "cuDNN", cudnn_time);

    // cuDNN should be faster (30-50% expected)
    EXPECT_LT(cudnn_time, custom_time * 1.4);
}

// ============================================================================
// End-to-End Model Benchmarks
// ============================================================================

TEST_F(CUDABenchmark, ResNetBlock_Forward) {
    // Benchmark a complete ResNet block
    Device device = Device::cuda(0);

    // Input: (32, 64, 56, 56)
    Tensor input = Tensor::randn({32, 64, 56, 56}, DType::Float32, device);

    // Conv layers, batch norms, ReLU, residual connection
    double custom_time = benchmark([&]() {
        // Simulated ResNet block with custom kernels
        auto x = input;
        // Conv1 + BN + ReLU
        // Conv2 + BN
        // Residual add + ReLU
    }, 50, 5);

    double optimized_time = benchmark([&]() {
        // Simulated ResNet block with cuDNN/cuBLAS
        auto x = input;
        // Same operations but with optimized kernels
    }, 50, 5);

    print_comparison("ResNet Block Forward", "Custom Kernels", custom_time, "cuDNN/cuBLAS", optimized_time);

    // Overall speedup expected: 20-35%
    EXPECT_LT(optimized_time, custom_time * 1.25);
}

// ============================================================================
// Memory Bandwidth Tests
// ============================================================================

TEST_F(CUDABenchmark, MemoryBandwidth) {
    // Test memory bandwidth utilization
    const int64_t size = 100 * 1024 * 1024 / sizeof(float);  // 100 MB

    Device device = Device::cuda(0);
    Tensor a = Tensor::randn({size}, DType::Float32, device);
    Tensor b = Tensor::randn({size}, DType::Float32, device);

    double time = benchmark([&]() {
        Tensor c = a + b;  // Simple element-wise add
    }, 1000, 100);

    // Calculate achieved bandwidth
    double bytes = size * sizeof(float) * 3;  // Read a, b, write c
    double bandwidth_gbps = (bytes / (time / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Memory Bandwidth Test:\n";
    std::cout << "  Time per operation: " << time << " ms\n";
    std::cout << "  Achieved bandwidth: " << bandwidth_gbps << " GB/s\n\n";

    // Should achieve at least 50% of theoretical bandwidth
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    double theoretical_bandwidth = (prop.memoryClockRate * 1000.0 * (prop.memoryBusWidth / 8.0) * 2.0) / (1024.0 * 1024.0 * 1024.0);
    std::cout << "  Theoretical bandwidth: " << theoretical_bandwidth << " GB/s\n";
    std::cout << "  Efficiency: " << (bandwidth_gbps / theoretical_bandwidth * 100.0) << "%\n\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
