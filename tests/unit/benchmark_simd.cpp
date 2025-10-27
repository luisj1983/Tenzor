/**
 * @file benchmark_simd.cpp
 * @brief Performance benchmarks for SIMD dispatch system
 */

#include <gtest/gtest.h>
#include "tenzor/backend/simd_dispatch.hpp"
#include <vector>
#include <chrono>
#include <iostream>
#include <iomanip>
#include <random>

using namespace tenzor::backend;

// Benchmark harness
class SIMDBenchmark : public ::testing::Test {
protected:
    void SetUp() override {
        initialize_simd_dispatch();
        rng.seed(42);
        dist = std::uniform_real_distribution<float>(-10.0f, 10.0f);
    }

    std::vector<float> generate_random_data(size_t size) {
        std::vector<float> data(size);
        for (size_t i = 0; i < size; ++i) {
            data[i] = dist(rng);
        }
        return data;
    }

    template<typename Func>
    double benchmark_kernel(const std::string& name, Func kernel, int iterations = 1000) {
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            kernel();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        return duration.count() / static_cast<double>(iterations);
    }

    void print_speedup(const std::string& op, double scalar_time, double simd_time) {
        double speedup = scalar_time / simd_time;
        std::cout << "  " << std::left << std::setw(20) << op
                  << " Scalar: " << std::setw(10) << std::fixed << std::setprecision(0) << scalar_time << " ns"
                  << " | SIMD: " << std::setw(10) << simd_time << " ns"
                  << " | Speedup: " << std::setprecision(2) << speedup << "x"
                  << std::endl;
    }

    std::mt19937 rng;
    std::uniform_real_distribution<float> dist;
};

// ============================================================================
// Element-wise Operation Benchmarks
// ============================================================================

TEST_F(SIMDBenchmark, AdditionPerformance) {
    std::cout << "\n=== Addition Performance Benchmark ===" << std::endl;
    std::cout << "CPU Features: " << get_cpu_features() << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        auto a = generate_random_data(size);
        auto b = generate_random_data(size);
        std::vector<float> result(size);

        // Benchmark scalar
        auto scalar_lambda = [&]() {
            kernels::add_scalar(result.data(), a.data(), b.data(), size);
        };
        double scalar_time = benchmark_kernel("scalar", scalar_lambda);

        // Benchmark SIMD
        auto simd_kernel = get_optimal_add_kernel();
        auto simd_lambda = [&]() {
            simd_kernel(result.data(), a.data(), b.data(), size);
        };
        double simd_time = benchmark_kernel("simd", simd_lambda);

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("Addition", scalar_time, simd_time);
    }
}

TEST_F(SIMDBenchmark, MultiplicationPerformance) {
    std::cout << "\n=== Multiplication Performance Benchmark ===" << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        auto a = generate_random_data(size);
        auto b = generate_random_data(size);
        std::vector<float> result(size);

        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::mul_scalar(result.data(), a.data(), b.data(), size);
        });

        auto simd_kernel = get_optimal_mul_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), a.data(), b.data(), size);
        });

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("Multiplication", scalar_time, simd_time);
    }
}

// ============================================================================
// Activation Function Benchmarks
// ============================================================================

TEST_F(SIMDBenchmark, ReLUPerformance) {
    std::cout << "\n=== ReLU Performance Benchmark ===" << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        auto input = generate_random_data(size);
        std::vector<float> result(size);

        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::relu_scalar(result.data(), input.data(), nullptr, size);
        });

        auto simd_kernel = get_optimal_relu_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), input.data(), nullptr, size);
        });

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("ReLU", scalar_time, simd_time);
    }
}

TEST_F(SIMDBenchmark, SigmoidPerformance) {
    std::cout << "\n=== Sigmoid Performance Benchmark ===" << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096};

    for (size_t size : sizes) {
        auto input = generate_random_data(size);
        std::vector<float> result(size);

        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::sigmoid_scalar(result.data(), input.data(), nullptr, size);
        }, 100); // Fewer iterations due to exp() cost

        auto simd_kernel = get_optimal_sigmoid_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), input.data(), nullptr, size);
        }, 100);

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("Sigmoid", scalar_time, simd_time);
    }
}

TEST_F(SIMDBenchmark, TanhPerformance) {
    std::cout << "\n=== Tanh Performance Benchmark ===" << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096};

    for (size_t size : sizes) {
        auto input = generate_random_data(size);
        std::vector<float> result(size);

        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::tanh_scalar(result.data(), input.data(), nullptr, size);
        }, 100);

        auto simd_kernel = get_optimal_tanh_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), input.data(), nullptr, size);
        }, 100);

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("Tanh", scalar_time, simd_time);
    }
}

// ============================================================================
// Reduction Operation Benchmarks
// ============================================================================

TEST_F(SIMDBenchmark, ReduceSumPerformance) {
    std::cout << "\n=== Reduce Sum Performance Benchmark ===" << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        auto input = generate_random_data(size);
        float result;

        double scalar_time = benchmark_kernel("scalar", [&]() {
            result = kernels::reduce_sum_scalar(input.data(), size);
        });

        auto simd_kernel = get_optimal_reduce_sum_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            result = simd_kernel(input.data(), size);
        });

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("Reduce Sum", scalar_time, simd_time);

        // Prevent optimization from eliminating the computation
        (void)result;
    }
}

TEST_F(SIMDBenchmark, ReduceMaxPerformance) {
    std::cout << "\n=== Reduce Max Performance Benchmark ===" << std::endl;

    const std::vector<size_t> sizes = {64, 256, 1024, 4096, 16384};

    for (size_t size : sizes) {
        auto input = generate_random_data(size);
        float result;

        double scalar_time = benchmark_kernel("scalar", [&]() {
            result = kernels::reduce_max_scalar(input.data(), size);
        });

        auto simd_kernel = get_optimal_reduce_max_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            result = simd_kernel(input.data(), size);
        });

        std::cout << "Size " << size << ":" << std::endl;
        print_speedup("Reduce Max", scalar_time, simd_time);

        (void)result;
    }
}

// ============================================================================
// Comprehensive Performance Report
// ============================================================================

TEST_F(SIMDBenchmark, ComprehensiveReport) {
    std::cout << "\n=== SIMD Dispatch Comprehensive Performance Report ===" << std::endl;
    std::cout << "CPU Features: " << get_cpu_features() << std::endl;
    std::cout << "  AVX-512: " << (cpu_supports_avx512() ? "YES" : "NO") << std::endl;
    std::cout << "  AVX2: " << (cpu_supports_avx2() ? "YES" : "NO") << std::endl;
    std::cout << "  SSE4.2: " << (cpu_supports_sse42() ? "YES" : "NO") << std::endl;
    std::cout << "  NEON: " << (cpu_supports_neon() ? "YES" : "NO") << std::endl;
    std::cout << std::endl;

    const size_t test_size = 4096;
    auto a = generate_random_data(test_size);
    auto b = generate_random_data(test_size);
    std::vector<float> result(test_size);

    std::cout << "Test size: " << test_size << " elements" << std::endl;
    std::cout << "-----------------------------------------------------------" << std::endl;

    // Addition
    {
        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::add_scalar(result.data(), a.data(), b.data(), test_size);
        });
        auto simd_kernel = get_optimal_add_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), a.data(), b.data(), test_size);
        });
        print_speedup("Addition", scalar_time, simd_time);
    }

    // Multiplication
    {
        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::mul_scalar(result.data(), a.data(), b.data(), test_size);
        });
        auto simd_kernel = get_optimal_mul_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), a.data(), b.data(), test_size);
        });
        print_speedup("Multiplication", scalar_time, simd_time);
    }

    // ReLU
    {
        double scalar_time = benchmark_kernel("scalar", [&]() {
            kernels::relu_scalar(result.data(), a.data(), nullptr, test_size);
        });
        auto simd_kernel = get_optimal_relu_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            simd_kernel(result.data(), a.data(), nullptr, test_size);
        });
        print_speedup("ReLU", scalar_time, simd_time);
    }

    // Reduce Sum
    {
        float sum;
        double scalar_time = benchmark_kernel("scalar", [&]() {
            sum = kernels::reduce_sum_scalar(a.data(), test_size);
        });
        auto simd_kernel = get_optimal_reduce_sum_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            sum = simd_kernel(a.data(), test_size);
        });
        print_speedup("Reduce Sum", scalar_time, simd_time);
        (void)sum;
    }

    // Reduce Max
    {
        float max_val;
        double scalar_time = benchmark_kernel("scalar", [&]() {
            max_val = kernels::reduce_max_scalar(a.data(), test_size);
        });
        auto simd_kernel = get_optimal_reduce_max_kernel();
        double simd_time = benchmark_kernel("simd", [&]() {
            max_val = simd_kernel(a.data(), test_size);
        });
        print_speedup("Reduce Max", scalar_time, simd_time);
        (void)max_val;
    }

    std::cout << "-----------------------------------------------------------" << std::endl;
    std::cout << "\nNote: Speedup depends on CPU architecture and SIMD support." << std::endl;
    std::cout << "Expected speedups:" << std::endl;
    std::cout << "  - SSE4.2 (128-bit): 2-4x" << std::endl;
    std::cout << "  - AVX2 (256-bit): 4-8x" << std::endl;
    std::cout << "  - AVX-512 (512-bit): 8-16x" << std::endl;
    std::cout << "  - NEON (128-bit): 2-4x" << std::endl;
}
