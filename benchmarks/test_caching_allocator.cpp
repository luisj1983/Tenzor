/**
 * @file test_caching_allocator.cpp
 * @brief Benchmark for CPU caching allocator performance
 *
 * Tests the performance improvements from memory caching:
 * - Tensor creation with cache reuse
 * - Repeated allocation/deallocation patterns
 * - Comparison of zeroed vs uninitialized allocation
 */

#include <chrono>
#include <iostream>
#include <iomanip>
#include <vector>
#include <numeric>
#include <algorithm>
#include "tenzor/tenzor.hpp"

using namespace tenzor;
using Clock = std::chrono::high_resolution_clock;

struct BenchmarkResult {
    std::string name;
    double mean_ms;
    double std_ms;
    double min_ms;
    double max_ms;
    double ops_per_sec;
};

template<typename Func>
BenchmarkResult benchmark(const std::string& name, int warmup, int iterations, Func&& func) {
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        func();
    }

    // Benchmark
    std::vector<double> times;
    times.reserve(iterations);

    for (int i = 0; i < iterations; ++i) {
        auto start = Clock::now();
        func();
        auto end = Clock::now();
        double ms = std::chrono::duration<double, std::milli>(end - start).count();
        times.push_back(ms);
    }

    // Calculate statistics
    double sum = std::accumulate(times.begin(), times.end(), 0.0);
    double mean = sum / times.size();

    double sq_sum = 0.0;
    for (auto t : times) {
        sq_sum += (t - mean) * (t - mean);
    }
    double std_dev = std::sqrt(sq_sum / times.size());

    auto [min_it, max_it] = std::minmax_element(times.begin(), times.end());

    return {
        name,
        mean,
        std_dev,
        *min_it,
        *max_it,
        1000.0 / mean  // ops per second
    };
}

void print_result(const BenchmarkResult& r) {
    std::cout << std::fixed << std::setprecision(4);
    std::cout << "\n=== " << r.name << " ===" << std::endl;
    std::cout << "  Mean:     " << std::setw(10) << r.mean_ms << " ms" << std::endl;
    std::cout << "  Std Dev:  " << std::setw(10) << r.std_ms << " ms" << std::endl;
    std::cout << "  Min:      " << std::setw(10) << r.min_ms << " ms" << std::endl;
    std::cout << "  Max:      " << std::setw(10) << r.max_ms << " ms" << std::endl;
    std::cout << "  Ops/sec:  " << std::setw(10) << r.ops_per_sec << std::endl;
}

void print_comparison(const BenchmarkResult& baseline, const BenchmarkResult& optimized) {
    double speedup = baseline.mean_ms / optimized.mean_ms;
    std::cout << "\n  Speedup vs " << baseline.name << ": " << std::fixed
              << std::setprecision(2) << speedup << "x" << std::endl;
}

int main() {
    // Initialize Tenzor library (loads backends)
    tenzor::initialize();

    std::cout << "========================================" << std::endl;
    std::cout << "  CPU Caching Allocator Benchmark" << std::endl;
    std::cout << "========================================" << std::endl;

    constexpr int WARMUP = 10;
    constexpr int ITERATIONS = 100;

    // Test sizes
    std::vector<int64_t> sizes = {1000, 10000, 100000, 1000000, 10000000};

    std::cout << "\n--- Tensor Creation Benchmarks ---" << std::endl;

    for (auto size : sizes) {
        std::cout << "\n\n### Size: " << size << " elements ###" << std::endl;

        // Benchmark: Create tensor (zero-initialized)
        auto r_zeros = benchmark(
            "zeros(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [size]() {
                auto t = zeros({size}, DType::Float32, Device::cpu());
                // Force evaluation
                volatile float x = t.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_zeros);

        // Benchmark: Create tensor (uninitialized via empty)
        auto r_empty = benchmark(
            "empty(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [size]() {
                auto t = empty({size}, DType::Float32, Device::cpu());
                volatile float x = t.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_empty);
        print_comparison(r_zeros, r_empty);

        // Benchmark: Repeated create-destroy cycle (tests cache reuse)
        auto r_cycle = benchmark(
            "create-destroy cycle(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [size]() {
                for (int i = 0; i < 10; ++i) {
                    auto t = empty({size}, DType::Float32, Device::cpu());
                    volatile float x = t.data<float>()[0];
                    (void)x;
                }
            }
        );
        print_result(r_cycle);
        std::cout << "  Per-tensor: " << (r_cycle.mean_ms / 10.0) << " ms" << std::endl;
    }

    std::cout << "\n\n--- Arithmetic Operation Benchmarks ---" << std::endl;
    std::cout << "(includes output tensor allocation)" << std::endl;

    for (auto size : sizes) {
        std::cout << "\n\n### Size: " << size << " elements ###" << std::endl;

        auto a = randn({size}, DType::Float32, Device::cpu());
        auto b = randn({size}, DType::Float32, Device::cpu());

        // Benchmark: Add (allocates output)
        auto r_add = benchmark(
            "add(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [&a, &b]() {
                auto c = a + b;
                volatile float x = c.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_add);

        // Calculate bandwidth
        double bytes = 3.0 * size * sizeof(float);  // 2 reads + 1 write
        double bandwidth_gbps = bytes / (r_add.mean_ms * 1e6);
        std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

        // Benchmark: Mul (allocates output)
        auto r_mul = benchmark(
            "mul(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [&a, &b]() {
                auto c = a * b;
                volatile float x = c.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_mul);

        bandwidth_gbps = bytes / (r_mul.mean_ms * 1e6);
        std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

        // Benchmark: Chained operations (multiple allocations, cache reuse)
        auto r_chain = benchmark(
            "chain a+b+a*b(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [&a, &b]() {
                auto c = a + b;
                auto d = a * b;
                auto e = c + d;
                volatile float x = e.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_chain);
    }

    std::cout << "\n\n--- Reduction Benchmarks ---" << std::endl;

    for (auto size : sizes) {
        std::cout << "\n\n### Size: " << size << " elements ###" << std::endl;

        auto a = randn({size}, DType::Float32, Device::cpu());

        // Benchmark: Sum
        auto r_sum = benchmark(
            "sum(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [&a]() {
                auto s = sum(a);
                volatile float x = s.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_sum);

        double bytes = size * sizeof(float);
        double bandwidth_gbps = bytes / (r_sum.mean_ms * 1e6);
        std::cout << "  Bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

        // Benchmark: Mean
        auto r_mean = benchmark(
            "mean(" + std::to_string(size) + ")",
            WARMUP, ITERATIONS,
            [&a]() {
                auto m = mean(a);
                volatile float x = m.data<float>()[0];
                (void)x;
            }
        );
        print_result(r_mean);
    }

    std::cout << "\n\n--- Memory Allocation Pattern Benchmarks ---" << std::endl;

    // Test varying sizes (stress test cache with different block sizes)
    auto r_varying = benchmark(
        "varying sizes (1K to 1M)",
        WARMUP, ITERATIONS,
        []() {
            std::vector<int64_t> test_sizes = {
                1000, 5000, 10000, 50000, 100000, 500000, 1000000
            };
            for (auto s : test_sizes) {
                auto t = empty({s}, DType::Float32, Device::cpu());
                volatile float x = t.data<float>()[0];
                (void)x;
            }
        }
    );
    print_result(r_varying);
    std::cout << "  Per-allocation: " << (r_varying.mean_ms / 7.0) << " ms" << std::endl;

    // Test same size repeatedly (optimal cache hit scenario)
    auto r_same = benchmark(
        "same size 100K x10",
        WARMUP, ITERATIONS,
        []() {
            for (int i = 0; i < 10; ++i) {
                auto t = empty({100000}, DType::Float32, Device::cpu());
                volatile float x = t.data<float>()[0];
                (void)x;
            }
        }
    );
    print_result(r_same);
    std::cout << "  Per-allocation: " << (r_same.mean_ms / 10.0) << " ms" << std::endl;

    std::cout << "\n\n========================================" << std::endl;
    std::cout << "  Benchmark Complete" << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
