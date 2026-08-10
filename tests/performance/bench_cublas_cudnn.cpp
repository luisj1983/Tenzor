/**
 * @file bench_cublas_cudnn.cpp
 * @brief Performance benchmarks: vendor (cuBLAS/cuDNN) vs custom CUDA kernels.
 *
 * The "custom vs vendor" comparison is made real by a process-global runtime
 * knob, tenzor::cuda::force_custom_kernels() (declared in
 * include/tenzor/backend/cuda_config.hpp). When on, the CUDA backend's matmul
 * bypasses cuBLAS for the custom tiled kernels, and Conv2d / BatchNorm2d bypass
 * cuDNN for the custom composed kernels that live alongside the vendor paths.
 * The "vendor" branch leaves the knob off (the default dispatch through
 * cuBLAS / cuDNN); the "custom" branch forces the knob on for the duration of
 * its timing loop. A scoped RAII guard restores the previous value on exit so
 * the two branches never interfere.
 *
 * This is a manual performance harness (slow, GPU-only): it is built by CMake
 * when TENZOR_BUILD_CUDA is on but is NOT added to ctest. Run the binary
 * directly: ./bin/bench_cublas_cudnn
 */

#include <gtest/gtest.h>
#include <cuda_runtime.h>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>

#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/backend/cuda_config.hpp"

using namespace tenzor;
using namespace std::chrono;

// ============================================================================
// RAII guard: force the CUDA backend into a chosen kernel mode for a scope.
// The knob is process-global (cuBLAS/cuDNN calls run on worker threads), so it
// must be set before the timed loop and restored after — never left on, or the
// "vendor" branch would silently measure the custom path too.
// ============================================================================
class KernelModeGuard {
public:
    explicit KernelModeGuard(bool force_custom)
        : prev_(tenzor::cuda::force_custom_kernels()) {
        tenzor::cuda::set_force_custom_kernels(force_custom);
    }
    ~KernelModeGuard() { tenzor::cuda::set_force_custom_kernels(prev_); }
    KernelModeGuard(const KernelModeGuard&) = delete;
    auto operator=(const KernelModeGuard&) -> KernelModeGuard& = delete;

private:
    bool prev_;
};

// ============================================================================
// Benchmark fixture
// ============================================================================
class CUDABenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        int device_count = 0;
        cudaGetDeviceCount(&device_count);
        if (device_count == 0) {
            GTEST_SKIP() << "No CUDA devices available";
        }

        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, 0);
        std::cout << "\n===========================================\n";
        std::cout << "GPU: " << prop.name << "\n";
        std::cout << "Compute Capability: " << prop.major << "." << prop.minor << "\n";
        std::cout << "Total Global Memory: " << (prop.totalGlobalMem / 1024 / 1024) << " MB\n";
        std::cout << "===========================================\n\n";

        has_tensor_cores = (prop.major >= 7);  // Volta and newer
        device_ = Device::cuda(0);
    }

    // Run `func` `iterations` times (after `warmup` untimed calls) and return
    // the average per-call wall time in ms. cudaDeviceSynchronize brackets the
    // timed region so only kernel execution (not launch enqueue) is counted.
    template<typename Func>
    double benchmark(Func&& func, int iterations = 100, int warmup = 10) {
        for (int i = 0; i < warmup; ++i) {
            func();
        }
        cudaDeviceSynchronize();

        auto start = high_resolution_clock::now();
        for (int i = 0; i < iterations; ++i) {
            func();
        }
        cudaDeviceSynchronize();
        auto end = high_resolution_clock::now();

        double total_time = duration<double, std::milli>(end - start).count();
        return total_time / iterations;
    }

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
    Device device_ = Device::cuda(0);
};

// ============================================================================
// Matrix Multiplication — matmul(a, b) dispatches through the registry; the
// knob selects cuBLAS (off) vs the custom tiled kernel (on).
// ============================================================================

TEST_F(CUDABenchmark, MatMul_FP32_Large) {
    const int64_t M = 2048, N = 2048, K = 2048;
    Tensor a = randn({M, K}, DType::Float32, device_);
    Tensor b = randn({K, N}, DType::Float32, device_);

    double custom_time;
    {
        KernelModeGuard g(/*force_custom=*/true);
        custom_time = benchmark([&]() { Tensor c = matmul(a, b); });
    }
    double cublas_time;
    {
        KernelModeGuard g(/*force_custom=*/false);
        cublas_time = benchmark([&]() { Tensor c = matmul(a, b); });
    }

    print_comparison("MatMul FP32 (2048x2048)", "Custom", custom_time, "cuBLAS", cublas_time);
    EXPECT_LT(cublas_time, custom_time * 1.1);
}

TEST_F(CUDABenchmark, MatMul_FP16_TensorCores) {
    if (!has_tensor_cores) {
        GTEST_SKIP() << "Tensor Cores not available on this GPU";
    }

    const int64_t M = 4096, N = 4096, K = 4096;
    Tensor a = randn({M, K}, DType::Float16, device_);
    Tensor b = randn({K, N}, DType::Float16, device_);

    double custom_time;
    {
        KernelModeGuard g(true);
        custom_time = benchmark([&]() { Tensor c = matmul(a, b); }, 50, 5);
    }
    double cublas_time;
    {
        KernelModeGuard g(false);
        cublas_time = benchmark([&]() { Tensor c = matmul(a, b); }, 50, 5);
    }

    print_comparison("MatMul FP16 Tensor Cores (4096x4096)",
                     "Custom WMMA", custom_time, "cuBLAS GemmEx", cublas_time);
    EXPECT_LT(cublas_time, custom_time * 1.2);
}

TEST_F(CUDABenchmark, BatchedMatMul_FP32) {
    const int64_t batch = 32, M = 512, N = 512, K = 512;
    Tensor a = randn({batch, M, K}, DType::Float32, device_);
    Tensor b = randn({batch, K, N}, DType::Float32, device_);

    double custom_time;
    {
        KernelModeGuard g(true);
        custom_time = benchmark([&]() { Tensor c = matmul(a, b); });
    }
    double cublas_time;
    {
        KernelModeGuard g(false);
        cublas_time = benchmark([&]() { Tensor c = matmul(a, b); });
    }

    print_comparison("Batched MatMul FP32 (32x512x512)",
                     "Custom", custom_time, "cuBLAS Batched", cublas_time);
    EXPECT_LT(cublas_time, custom_time * 1.15);
}

// ============================================================================
// Conv2d — nn::Conv2d::forward dispatches OpId::Conv2dForward; the knob
// selects cuDNN (off) vs the custom im2col+scalar kernel (on). NoGradGuard
// keeps the timing on the kernel, not autograd graph construction.
// ============================================================================

TEST_F(CUDABenchmark, Conv2d_Forward_FP32) {
    // ResNet-50 typical layer: (32, 64, 56, 56) -> (32, 128, 56, 56), 3x3/1/1.
    nn::Conv2d conv(/*in=*/64, /*out=*/128, /*kernel=*/3,
                    /*stride=*/1, /*padding=*/1);
    conv.to(device_);

    Tensor input_t = randn({32, 64, 56, 56}, DType::Float32, device_);
    Variable input(input_t, /*requires_grad=*/false);

    double custom_time;
    {
        KernelModeGuard g(true);
        NoGradGuard ng;
        custom_time = benchmark([&]() { Variable out = conv.forward(input); });
    }
    double cudnn_time;
    {
        KernelModeGuard g(false);
        NoGradGuard ng;
        cudnn_time = benchmark([&]() { Variable out = conv.forward(input); });
    }

    print_comparison("Conv2d Forward FP32 (ResNet-50 layer)",
                     "Custom im2col", custom_time, "cuDNN", cudnn_time);
    EXPECT_LT(cudnn_time, custom_time * 1.2);
}

TEST_F(CUDABenchmark, Conv2d_Forward_FP16) {
    if (!has_tensor_cores) {
        GTEST_SKIP() << "Tensor Cores not available on this GPU";
    }

    nn::Conv2d conv(64, 128, 3, 1, 1);
    conv.to(device_);

    Tensor input_t = randn({32, 64, 56, 56}, DType::Float16, device_);
    Variable input(input_t, /*requires_grad=*/false);

    double custom_time;
    {
        KernelModeGuard g(true);
        NoGradGuard ng;
        custom_time = benchmark([&]() { Variable out = conv.forward(input); }, 50, 5);
    }
    double cudnn_time;
    {
        KernelModeGuard g(false);
        NoGradGuard ng;
        cudnn_time = benchmark([&]() { Variable out = conv.forward(input); }, 50, 5);
    }

    print_comparison("Conv2d Forward FP16 Tensor Cores",
                     "Custom", custom_time, "cuDNN", cudnn_time);
    EXPECT_LT(cudnn_time, custom_time * 1.3);
}

TEST_F(CUDABenchmark, Conv2d_Backward) {
    // Backward: rebuild the graph each iteration (forward creates a fresh
    // graph; backward consumes it). The knob routes both Conv2dForward and the
    // Conv2dBackward* ops through custom vs vendor, so the timed region
    // reflects the kernel difference on both passes.
    nn::Conv2d conv(64, 128, 3, 1, 1);
    conv.to(device_);

    Tensor input_t = randn({32, 64, 56, 56}, DType::Float32, device_);
    Tensor grad_out = randn({32, 128, 56, 56}, DType::Float32, device_);

    double custom_time;
    {
        KernelModeGuard g(true);
        custom_time = benchmark([&]() {
            Variable x(input_t, /*requires_grad=*/true);
            Variable y = conv.forward(x);
            y.backward(grad_out);
        }, 50, 5);
    }
    double cudnn_time;
    {
        KernelModeGuard g(false);
        cudnn_time = benchmark([&]() {
            Variable x(input_t, /*requires_grad=*/true);
            Variable y = conv.forward(x);
            y.backward(grad_out);
        }, 50, 5);
    }

    print_comparison("Conv2d Backward FP32", "Custom", custom_time, "cuDNN", cudnn_time);
    EXPECT_LT(cudnn_time, custom_time * 1.2);
}

// ============================================================================
// BatchNorm2d — nn::BatchNorm2d::forward (train mode) dispatches
// OpId::BatchNorm2dFusedTraining; the knob selects cuDNN (off) vs the custom
// composed path (on). Backward dispatches OpId::BatchNorm2dBackward likewise.
// ============================================================================

TEST_F(CUDABenchmark, BatchNorm2d_Forward) {
    nn::BatchNorm2d bn(/*num_features=*/256);
    bn.to(device_);
    bn.train();

    Tensor input_t = randn({32, 256, 28, 28}, DType::Float32, device_);
    Variable input(input_t, /*requires_grad=*/false);

    double custom_time;
    {
        KernelModeGuard g(true);
        NoGradGuard ng;
        custom_time = benchmark([&]() { Variable out = bn.forward(input); });
    }
    double cudnn_time;
    {
        KernelModeGuard g(false);
        NoGradGuard ng;
        cudnn_time = benchmark([&]() { Variable out = bn.forward(input); });
    }

    print_comparison("BatchNorm2d Forward FP32", "Custom", custom_time, "cuDNN", cudnn_time);
    EXPECT_LT(cudnn_time, custom_time * 1.4);
}

TEST_F(CUDABenchmark, BatchNorm2d_Backward) {
    nn::BatchNorm2d bn(256);
    bn.to(device_);
    bn.train();

    Tensor input_t = randn({32, 256, 28, 28}, DType::Float32, device_);
    Tensor grad_out = randn({32, 256, 28, 28}, DType::Float32, device_);

    double custom_time;
    {
        KernelModeGuard g(true);
        custom_time = benchmark([&]() {
            Variable x(input_t, /*requires_grad=*/true);
            Variable y = bn.forward(x);
            y.backward(grad_out);
        }, 50, 5);
    }
    double cudnn_time;
    {
        KernelModeGuard g(false);
        cudnn_time = benchmark([&]() {
            Variable x(input_t, /*requires_grad=*/true);
            Variable y = bn.forward(x);
            y.backward(grad_out);
        }, 50, 5);
    }

    print_comparison("BatchNorm2d Backward FP32", "Custom", custom_time, "cuDNN", cudnn_time);
    EXPECT_LT(cudnn_time, custom_time * 1.4);
}

// ============================================================================
// End-to-End: a ResNet basic block (two 3x3 convs + BNs + ReLU + residual).
// Both conv and BN dispatch honor the knob, so the whole block flips between
// vendor and custom kernels.
// ============================================================================

TEST_F(CUDABenchmark, ResNetBlock_Forward) {
    nn::Conv2d conv1(64, 64, 3, 1, 1);
    nn::BatchNorm2d bn1(64);
    nn::Conv2d conv2(64, 64, 3, 1, 1);
    nn::BatchNorm2d bn2(64);
    nn::ReLU relu;
    conv1.to(device_); bn1.to(device_);
    conv2.to(device_); bn2.to(device_);

    Tensor input_t = randn({32, 64, 56, 56}, DType::Float32, device_);
    Variable input(input_t, /*requires_grad=*/false);

    auto run_block = [&]() {
        Variable identity = input;
        Variable x = relu.forward(bn1.forward(conv1.forward(input)));
        x = bn2.forward(conv2.forward(x));
        Variable out = relu.forward(x + identity);
        return out;
    };

    double custom_time;
    {
        KernelModeGuard g(true);
        NoGradGuard ng;
        custom_time = benchmark(run_block, 50, 5);
    }
    double vendor_time;
    {
        KernelModeGuard g(false);
        NoGradGuard ng;
        vendor_time = benchmark(run_block, 50, 5);
    }

    print_comparison("ResNet Block Forward", "Custom Kernels", custom_time,
                     "cuDNN/cuBLAS", vendor_time);
    EXPECT_LT(vendor_time, custom_time * 1.25);
}

// ============================================================================
// Memory Bandwidth — element-wise add. Unaffected by the vendor/custom knob
// (no cuBLAS/cuDNN involvement); included as a baseline utilization check.
// ============================================================================

TEST_F(CUDABenchmark, MemoryBandwidth) {
    const int64_t size = 100 * 1024 * 1024 / sizeof(float);  // 100 MB
    Tensor a = randn({size}, DType::Float32, device_);
    Tensor b = randn({size}, DType::Float32, device_);

    double time = benchmark([&]() { Tensor c = a + b; }, 1000, 100);

    double bytes = size * sizeof(float) * 3;  // Read a, b, write c
    double bandwidth_gbps = (bytes / (time / 1000.0)) / (1024.0 * 1024.0 * 1024.0);

    std::cout << "Memory Bandwidth Test:\n";
    std::cout << "  Time per operation: " << time << " ms\n";
    std::cout << "  Achieved bandwidth: " << bandwidth_gbps << " GB/s\n\n";

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    int mem_clock_khz = 0;
    cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, 0);
    double theoretical_bandwidth = (mem_clock_khz * 1000.0 *
                                    (prop.memoryBusWidth / 8.0) * 2.0) /
                                   (1024.0 * 1024.0 * 1024.0);
    std::cout << "  Theoretical bandwidth: " << theoretical_bandwidth << " GB/s\n";
    std::cout << "  Efficiency: "
              << (bandwidth_gbps / theoretical_bandwidth * 100.0) << "%\n\n";
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    tenzor::initialize();  // register backends before any op dispatch
    return RUN_ALL_TESTS();
}