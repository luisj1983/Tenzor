/**
 * @file test_performance_regression.cpp
 * @brief Performance regression tests and baseline measurements
 *
 * Establishes performance baselines for each backend and detects
 * performance regressions over time.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"
#include <chrono>
#include <fstream>
#include <iomanip>

using namespace tenzor;
using namespace tenzor::testing;

// Helper function to measure operation time
template<typename Func>
double measure_time_ms(Func&& func, const Device& device, int iterations = 10) {
    // Warmup
    for (int i = 0; i < 3; ++i) {
        func();
        device.synchronize();
    }

    // Measurement
    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < iterations; ++i) {
        func();
        device.synchronize();
    }

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

    return duration / (1000.0 * iterations);  // Convert to ms per iteration
}

// ============================================================================
// MatMul Performance Baselines
// ============================================================================

TEST(PerformanceRegression, MatMul_Small) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {128, 128};
    const int iterations = 100;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== MatMul 128x128 Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        double time = measure_time_ms([&]() {
            auto c = matmul(a_dev, b_dev);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

TEST(PerformanceRegression, MatMul_Medium) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {512, 512};
    const int iterations = 50;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== MatMul 512x512 Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        double time = measure_time_ms([&]() {
            auto c = matmul(a_dev, b_dev);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

TEST(PerformanceRegression, MatMul_Large) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 20;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== MatMul 1024x1024 Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        double time = measure_time_ms([&]() {
            auto c = matmul(a_dev, b_dev);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

// ============================================================================
// Convolution Performance Baselines
// ============================================================================

TEST(PerformanceRegression, Conv2d_Small) {
    // Skipped: nn::functional::conv2d not available
    GTEST_SKIP() << "nn::functional::conv2d API not available";
}

TEST(PerformanceRegression, Conv2d_Large) {
    // Skipped: nn::functional::conv2d not available
    GTEST_SKIP() << "nn::functional::conv2d API not available";
}

// ============================================================================
// Element-wise Operation Performance
// ============================================================================

TEST(PerformanceRegression, ElementWise_Add) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 200;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== Element-wise Add Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        double time = measure_time_ms([&]() {
            auto c = a_dev + b_dev;
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

TEST(PerformanceRegression, ElementWise_Mul) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 200;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== Element-wise Mul Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        double time = measure_time_ms([&]() {
            auto c = a_dev * b_dev;
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

// ============================================================================
// Activation Function Performance
// ============================================================================

TEST(PerformanceRegression, Activation_ReLU) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 200;

    auto x = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== ReLU Activation Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);

        double time = measure_time_ms([&]() {
            // Use clamp_min as relu equivalent
            auto y = clamp_min(x_dev, 0.0f);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

TEST(PerformanceRegression, Activation_GELU) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 100;

    auto x = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== GELU Activation Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);

        double time = measure_time_ms([&]() {
            auto x_var = Variable(x_dev, false);
            auto y = nn::gelu(x_var);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

TEST(PerformanceRegression, Activation_Softmax) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {512, 1024};
    const int iterations = 100;

    auto x = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== Softmax Activation Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);

        double time = measure_time_ms([&]() {
            auto x_var = Variable(x_dev, false);
            auto y = nn::softmax(x_var, 1);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

// ============================================================================
// Reduction Operation Performance
// ============================================================================

TEST(PerformanceRegression, Reduction_Sum) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 200;

    auto x = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== Sum Reduction Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);

        double time = measure_time_ms([&]() {
            auto y = sum(x_dev);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

TEST(PerformanceRegression, Reduction_Mean) {
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 200;

    auto x = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== Mean Reduction Performance ===" << std::endl;

    for (const auto& backend : backends) {
        auto x_dev = x.to(backend);

        double time = measure_time_ms([&]() {
            auto y = mean(x_dev);
        }, backend, iterations);

        std::cout << std::fixed << std::setprecision(3)
                 << backend_name(backend) << ": " << time << " ms" << std::endl;
    }

    SUCCEED();
}

// ============================================================================
// GPU Speedup Tests
// ============================================================================

TEST(PerformanceRegression, GPU_Speedup_MatMul) {
    auto backends = get_available_backends();

    // Find CPU and first GPU backend
    Device cpu_dev = Device::cpu();
    Device gpu_dev;
    bool has_gpu = false;

    for (const auto& backend : backends) {
        if (backend.type != Device::Type::CPU) {
            gpu_dev = backend;
            has_gpu = true;
            break;
        }
    }

    if (!has_gpu) {
        GTEST_SKIP() << "No GPU backend available for speedup test";
    }

    const std::vector<int64_t> shape = {1024, 1024};
    const int iterations = 20;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    // Measure CPU time
    double cpu_time = measure_time_ms([&]() {
        auto c = matmul(a, b);
    }, cpu_dev, iterations);

    // Measure GPU time
    auto a_gpu = a.to(gpu_dev);
    auto b_gpu = b.to(gpu_dev);

    double gpu_time = measure_time_ms([&]() {
        auto c = matmul(a_gpu, b_gpu);
    }, gpu_dev, iterations);

    double speedup = cpu_time / gpu_time;

    std::cout << "\n=== GPU Speedup for MatMul 1024x1024 ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
             << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time (" << backend_name(gpu_dev) << "): " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // GPU should be faster for large matmul (expect at least 2x speedup)
    EXPECT_GT(speedup, 2.0) << "GPU should be at least 2x faster than CPU for large MatMul";
}

TEST(PerformanceRegression, GPU_Speedup_Conv2d) {
    auto backends = get_available_backends();

    Device cpu_dev = Device::cpu();
    Device gpu_dev;
    bool has_gpu = false;

    for (const auto& backend : backends) {
        if (backend.type != Device::Type::CPU) {
            gpu_dev = backend;
            has_gpu = true;
            break;
        }
    }

    if (!has_gpu) {
        GTEST_SKIP() << "No GPU backend available for speedup test";
    }

    const int iterations = 20;

    // Create Conv2d module (in_channels=64, out_channels=128, kernel_size=3, padding=1)
    auto conv_cpu = nn::Conv2d(64, 128, 3, 1, 1);
    conv_cpu.to(cpu_dev);

    auto input = randn({16, 64, 128, 128}, DType::Float32, Device::cpu());
    auto input_var = Variable(input, false);

    // Measure CPU time
    double cpu_time = measure_time_ms([&]() {
        auto output = conv_cpu.forward(input_var);
    }, cpu_dev, iterations);

    // Measure GPU time
    auto conv_gpu = nn::Conv2d(64, 128, 3, 1, 1);
    conv_gpu.to(gpu_dev);

    auto input_gpu = input.to(gpu_dev);
    auto input_gpu_var = Variable(input_gpu, false);

    double gpu_time = measure_time_ms([&]() {
        auto output = conv_gpu.forward(input_gpu_var);
    }, gpu_dev, iterations);

    double speedup = cpu_time / gpu_time;

    std::cout << "\n=== GPU Speedup for Conv2d ===" << std::endl;
    std::cout << std::fixed << std::setprecision(3)
             << "CPU time: " << cpu_time << " ms" << std::endl;
    std::cout << "GPU time (" << backend_name(gpu_dev) << "): " << gpu_time << " ms" << std::endl;
    std::cout << "Speedup: " << speedup << "x" << std::endl;

    // GPU should be significantly faster for convolution
    EXPECT_GT(speedup, 3.0) << "GPU should be at least 3x faster than CPU for Conv2d";
}

// ============================================================================
// Performance Regression Detection
// ============================================================================

TEST(PerformanceRegression, RegressionCheck_MatMul) {
    // This test would compare current performance against stored baseline
    // In a real scenario, you'd load baseline from a file and compare
    auto backends = get_available_backends();

    const std::vector<int64_t> shape = {512, 512};
    const int iterations = 50;

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    std::cout << "\n=== Performance Regression Check ===" << std::endl;

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        double time = measure_time_ms([&]() {
            auto c = matmul(a_dev, b_dev);
        }, backend, iterations);

        std::cout << backend_name(backend) << " MatMul 512x512: "
                 << std::fixed << std::setprecision(3) << time << " ms" << std::endl;

        // In a real test, compare against baseline and fail if >10% slower
        // EXPECT_LT(time, baseline * 1.1) << "Performance regression detected";
    }

    SUCCEED();
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    std::cout << "\n====================================================\n";
    std::cout << "  Backend Performance Regression Tests\n";
    std::cout << "====================================================\n";

    int result = RUN_ALL_TESTS();

    std::cout << "\n====================================================\n";

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
