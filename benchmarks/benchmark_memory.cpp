/**
 * @file benchmark_memory.cpp
 * @brief Comprehensive benchmark for memory operations
 *
 * Benchmarks:
 * - Memory allocation/deallocation
 * - Tensor copying (CPU, GPU)
 * - Data transfers
 * - Caching allocator performance
 */

#include "tenzor/tenzor.hpp"
#include "common.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <vector>

using namespace tenzor;
using namespace tenzor::benchmark;

// Global device parsed from argv in main(). Defaults to CPU so unflagged
// invocations stay correct. Previously this file never read argv at all, so
// --device cuda silently benchmarked the CPU backend instead.
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

constexpr size_t WARMUP_ITERATIONS = 5;
constexpr size_t BENCHMARK_ITERATIONS = 100;

/**
 * @brief Benchmark tensor allocation
 */
void benchmark_allocation() {
    std::cout << "\n========================================\n";
    std::cout << "  Tensor Allocation Benchmarks\n";
    std::cout << "========================================\n\n";

    struct AllocConfig {
        std::vector<int64_t> shape;
        std::string name;
    };

    std::vector<AllocConfig> configs = {
        {{1024}, "Vector 1K"},
        {{1024, 1024}, "Matrix 1K x 1K"},
        {{4096, 4096}, "Matrix 4K x 4K"},
        {{8192, 8192}, "Matrix 8K x 8K"},
        {{64, 256, 256}, "3D Tensor 64x256x256"},
        {{32, 128, 128, 128}, "4D Tensor 32x128x128x128"},
    };

    for (const auto& cfg : configs) {
        size_t num_elements = 1;
        for (auto dim : cfg.shape) {
            num_elements *= dim;
        }
        size_t bytes = num_elements * sizeof(float);

        Benchmark bench("Alloc - " + cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto t = zeros(cfg.shape, DType::Float32, g_bench_device);
            volatile void* ptr = t.data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark tensor cloning (deep copy)
 */
void benchmark_clone() {
    std::cout << "\n========================================\n";
    std::cout << "  Tensor Clone Benchmarks\n";
    std::cout << "========================================\n\n";

    struct CloneConfig {
        std::vector<int64_t> shape;
        std::string name;
    };

    std::vector<CloneConfig> configs = {
        {{1024, 1024}, "Clone 1K x 1K"},
        {{2048, 2048}, "Clone 2K x 2K"},
        {{4096, 4096}, "Clone 4K x 4K"},
        {{64, 256, 256}, "Clone 64x256x256"},
    };

    for (const auto& cfg : configs) {
        auto t = randn(cfg.shape, DType::Float32, g_bench_device);

        size_t num_elements = 1;
        for (auto dim : cfg.shape) {
            num_elements *= dim;
        }
        size_t bytes = num_elements * sizeof(float) * 2;  // Read + Write

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto cloned = t.clone();
            volatile void* ptr = cloned.data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark tensor reshaping (view operations)
 */
void benchmark_reshape() {
    std::cout << "\n========================================\n";
    std::cout << "  Tensor Reshape/View Benchmarks\n";
    std::cout << "========================================\n\n";

    auto t = randn({1024, 1024}, DType::Float32, g_bench_device);

    // Reshape (should be fast - just metadata change)
    {
        Benchmark bench("Reshape 1024x1024 -> 1x1048576", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS * 10);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto reshaped = t.reshape({1, 1024 * 1024});
            volatile void* ptr = reshaped.data_ptr();
            (void)ptr;
        });

        result.print();
    }

    // Transpose
    {
        Benchmark bench("Transpose 1024x1024", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS * 10);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto transposed = t.transpose(0, 1);
            volatile void* ptr = transposed.data_ptr();
            (void)ptr;
        });

        result.print();
    }

    // Contiguous (may require copy)
    {
        auto non_contiguous = t.transpose(0, 1);

        Benchmark bench("Make Contiguous 1024x1024", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(1024 * 1024 * sizeof(float) * 2);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto cont = non_contiguous.contiguous();
            volatile void* ptr = cont.data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark slicing operations
 */
void benchmark_slicing() {
    std::cout << "\n========================================\n";
    std::cout << "  Tensor Slicing Benchmarks\n";
    std::cout << "========================================\n\n";

    auto t = randn({1024, 1024}, DType::Float32, g_bench_device);

    // Basic slice
    {
        Benchmark bench("Slice [0:512, :]", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS * 10);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto sliced = t.slice(0, 0, 512);
            volatile void* ptr = sliced.data_ptr();
            (void)ptr;
        });

        result.print();
    }

    // Multiple slices (more complex than single slice)
    {
        Benchmark bench("Multiple Slices [:512, :512]", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto sliced1 = t.slice(0, 0, 512);
            auto sliced2 = sliced1.slice(1, 0, 512);
            volatile void* ptr = sliced2.data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark concatenation
 */
void benchmark_concatenation() {
    std::cout << "\n========================================\n";
    std::cout << "  Tensor Concatenation Benchmarks\n";
    std::cout << "========================================\n\n";

    // Concatenate two tensors
    {
        auto a = randn({512, 1024}, DType::Float32, g_bench_device);
        auto b = randn({512, 1024}, DType::Float32, g_bench_device);

        size_t bytes = (512 * 1024 * 2) * sizeof(float) * 2;  // Read both + write result

        Benchmark bench("Cat 2 tensors [512x1024]", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto c = cat({a, b}, 0);
            volatile void* ptr = c.data_ptr();
            (void)ptr;
        });

        result.print();
    }

    // Concatenate multiple tensors
    {
        std::vector<Tensor> tensors;
        for (int i = 0; i < 8; ++i) {
            tensors.push_back(randn({128, 512}, DType::Float32, g_bench_device));
        }

        size_t bytes = (128 * 512 * 8) * sizeof(float) * 2;

        Benchmark bench("Cat 8 tensors [128x512]", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto c = cat(tensors, 0);
            volatile void* ptr = c.data_ptr();
            (void)ptr;
        });

        result.print();
    }

    // Stack tensors
    {
        std::vector<Tensor> tensors;
        for (int i = 0; i < 4; ++i) {
            tensors.push_back(randn({256, 256}, DType::Float32, g_bench_device));
        }

        size_t bytes = (256 * 256 * 4) * sizeof(float) * 2;

        Benchmark bench("Stack 4 tensors [256x256]", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto s = stack(tensors, 0);
            volatile void* ptr = s.data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark fill operations
 */
void benchmark_fill() {
    std::cout << "\n========================================\n";
    std::cout << "  Tensor Fill Benchmarks\n";
    std::cout << "========================================\n\n";

    struct FillConfig {
        std::vector<int64_t> shape;
        std::string name;
    };

    std::vector<FillConfig> configs = {
        {{1024, 1024}, "Fill 1K x 1K"},
        {{2048, 2048}, "Fill 2K x 2K"},
        {{4096, 4096}, "Fill 4K x 4K"},
    };

    for (const auto& cfg : configs) {
        size_t num_elements = 1;
        for (auto dim : cfg.shape) {
            num_elements *= dim;
        }
        size_t bytes = num_elements * sizeof(float);

        // Zeros
        {
            Benchmark bench("Zeros - " + cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_bytes(bytes);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto t = zeros(cfg.shape, DType::Float32, g_bench_device);
                volatile void* ptr = t.data_ptr();
                (void)ptr;
            });

            result.print();
        }

        // Ones
        {
            Benchmark bench("Ones - " + cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_bytes(bytes);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto t = ones(cfg.shape, DType::Float32, g_bench_device);
                volatile void* ptr = t.data_ptr();
                (void)ptr;
            });

            result.print();
        }

        // Random
        {
            Benchmark bench("Randn - " + cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
            bench.set_bytes(bytes);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto t = randn(cfg.shape, DType::Float32, g_bench_device);
                volatile void* ptr = t.data_ptr();
                (void)ptr;
            });

            result.print();
        }
    }
}

/**
 * @brief Benchmark memory bandwidth
 */
void benchmark_memory_bandwidth() {
    std::cout << "\n========================================\n";
    std::cout << "  Memory Bandwidth Benchmarks\n";
    std::cout << "========================================\n\n";
    std::cout << "Target: Memory overhead < 10% (PyTorch: 15%)\n\n";

    struct BandwidthTest {
        std::vector<int64_t> shape;
        std::string name;
    };

    std::vector<BandwidthTest> tests = {
        {{16 * 1024 * 1024}, "Sequential Read 64MB"},
        {{32 * 1024 * 1024}, "Sequential Read 128MB"},
        {{64 * 1024 * 1024}, "Sequential Read 256MB"},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& test : tests) {
        auto t = randn(test.shape, DType::Float32, g_bench_device);

        size_t num_elements = 1;
        for (auto dim : test.shape) {
            num_elements *= dim;
        }
        size_t bytes = num_elements * sizeof(float);

        Benchmark bench(test.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 5);
        bench.set_bytes(bytes);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto cloned = t.clone();
            volatile void* ptr = cloned.data_ptr();
            (void)ptr;
        });

        result.print();
        results.push_back(result);
    }

    // Print bandwidth summary
    std::cout << "\n========================================\n";
    std::cout << "  Memory Bandwidth Summary\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(30) << "Test"
              << std::right << std::setw(15) << "Bandwidth"
              << "\n";
    std::cout << std::string(45, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(30) << result.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(15) << result.bandwidth_gbs << " GB/s"
                  << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor Memory Benchmark Suite\n";
    std::cout << "  device=" << g_bench_device.to_string() << "\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  Memory Overhead:  < 10% (PyTorch: 15%)\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_allocation();
        benchmark_clone();
        benchmark_reshape();
        benchmark_slicing();
        benchmark_concatenation();
        benchmark_fill();
        benchmark_memory_bandwidth();

        std::cout << "\n========================================\n";
        std::cout << "  Memory Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
