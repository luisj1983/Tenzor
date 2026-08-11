/**
 * @file benchmark_ops.cpp
 * @brief Comprehensive benchmark for matrix and tensor operations
 *
 * Benchmarks critical operations including:
 * - Matrix multiplication (various sizes)
 * - Element-wise operations
 * - Reduction operations
 * - Backward pass timing
 */

#include "tenzor/tenzor.hpp"
#include "common.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/utils/benchmark.hpp"
#include "tenzor/autograd/variable.hpp"
#include <iostream>
#include <sstream>
#include <string>

using namespace tenzor;
using namespace tenzor::benchmark;

// Forward declarations - defined before main()
static void collect_result(const BenchmarkResult& result);

// Global device parsed from argv in main(). Defaults to CPU so unflagged
// invocations stay correct. Previously main() parsed --device into a local
// that was never used to target randn()/Benchmark — see the removed comment
// admitting as much. A --device cuda run silently benchmarked the CPU
// backend instead.
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

// Benchmark configuration
constexpr size_t WARMUP_ITERATIONS = 5;
constexpr size_t BENCHMARK_ITERATIONS = 50;

/**
 * @brief Benchmark matrix multiplication for various sizes
 */
void benchmark_matmul_suite() {
    std::cout << "\n========================================\n";
    std::cout << "  Matrix Multiplication Benchmarks\n";
    std::cout << "========================================\n\n";

    // Test different matrix sizes
    std::vector<std::tuple<size_t, size_t, size_t, std::string>> test_cases = {
        {128, 128, 128, "Small (128x128 x 128x128)"},
        {512, 512, 512, "Medium (512x512 x 512x512)"},
        {1024, 1024, 1024, "Large (1024x1024 x 1024x1024)"},
        {2048, 2048, 2048, "Very Large (2048x2048 x 2048x2048)"},
        {4096, 4096, 4096, "Huge (4096x4096 x 4096x4096)"},
        {256, 1024, 512, "Rectangular (256x1024 x 1024x512)"},
        {1024, 256, 512, "Rectangular (1024x256 x 256x512)"},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& [M, K, N, name] : test_cases) {
        // Create tensors
        auto a = randn({static_cast<int64_t>(M), static_cast<int64_t>(K)}, DType::Float32, g_bench_device);
        auto b = randn({static_cast<int64_t>(K), static_cast<int64_t>(N)}, DType::Float32, g_bench_device);

        // Setup benchmark
        Benchmark bench(name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(flops::matmul(M, N, K));
        bench.set_bytes(memory::matmul(M, N, K, 4));

        // Run benchmark
        auto result = bench.set_device(g_bench_device).run([&]() {
            auto c = matmul(a, b);
            // Force computation (prevent lazy evaluation if any)
            volatile void* ptr = c.data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
        results.push_back(result);
    }

    // Print summary
    std::cout << "\n========================================\n";
    std::cout << "  MatMul Summary\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(35) << "Configuration"
              << std::right << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "GFLOPS"
              << std::setw(12) << "BW (GB/s)"
              << "\n";
    std::cout << std::string(71, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(35) << result.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << (result.stats.mean * 1000.0)
                  << std::setw(12) << (result.tflops * 1000.0)
                  << std::setw(12) << result.bandwidth_gbs
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark batched matrix multiplication
 */
void benchmark_bmm() {
    std::cout << "\n========================================\n";
    std::cout << "  Batched Matrix Multiplication\n";
    std::cout << "========================================\n\n";

    std::vector<std::tuple<size_t, size_t, size_t, size_t, std::string>> test_cases = {
        {8, 256, 256, 256, "Batch=8, 256x256 x 256x256"},
        {16, 512, 512, 512, "Batch=16, 512x512 x 512x512"},
        {32, 128, 128, 128, "Batch=32, 128x128 x 128x128"},
        {64, 64, 64, 64, "Batch=64, 64x64 x 64x64"},
    };

    for (const auto& [batch, M, K, N, name] : test_cases) {
        auto a = randn({static_cast<int64_t>(batch), static_cast<int64_t>(M), static_cast<int64_t>(K)}, DType::Float32, g_bench_device);
        auto b = randn({static_cast<int64_t>(batch), static_cast<int64_t>(K), static_cast<int64_t>(N)}, DType::Float32, g_bench_device);

        Benchmark bench(name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(batch * flops::matmul(M, N, K));
        bench.set_bytes(batch * memory::matmul(M, N, K, 4));

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto c = bmm(a, b);
            volatile void* ptr = c.data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark element-wise operations
 */
void benchmark_elementwise() {
    std::cout << "\n========================================\n";
    std::cout << "  Element-wise Operations\n";
    std::cout << "========================================\n\n";

    // Test different tensor sizes
    std::vector<std::pair<std::vector<int64_t>, std::string>> shapes = {
        {{1024, 1024}, "1024x1024"},
        {{2048, 2048}, "2048x2048"},
        {{4096, 4096}, "4096x4096"},
        {{64, 256, 256}, "64x256x256 (3D)"},
    };

    for (const auto& [shape, name] : shapes) {
        auto a = randn(shape, DType::Float32, g_bench_device);
        auto b = randn(shape, DType::Float32, g_bench_device);

        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }

        // Addition
        {
            Benchmark bench("Add - " + name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops::elementwise(num_elements, 1));
            bench.set_bytes(memory::elementwise(num_elements, 2, 1, 4));

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = add(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
            collect_result(result);
        }

        // Multiplication
        {
            Benchmark bench("Mul - " + name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops::elementwise(num_elements, 1));
            bench.set_bytes(memory::elementwise(num_elements, 2, 1, 4));

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = mul(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
            collect_result(result);
        }

        // Exponential
        {
            Benchmark bench("Exp - " + name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops::elementwise(num_elements, 1));
            bench.set_bytes(memory::elementwise(num_elements, 1, 1, 4));

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = exp(a);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
            collect_result(result);
        }

        // Tanh
        {
            Benchmark bench("Tanh - " + name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops::elementwise(num_elements, 1));
            bench.set_bytes(memory::elementwise(num_elements, 1, 1, 4));

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = tanh(a);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
            collect_result(result);
        }
    }
}

/**
 * @brief Benchmark reduction operations
 */
void benchmark_reductions() {
    std::cout << "\n========================================\n";
    std::cout << "  Reduction Operations\n";
    std::cout << "========================================\n\n";

    std::vector<std::pair<std::vector<int64_t>, std::string>> shapes = {
        {{1024, 1024}, "1024x1024"},
        {{2048, 2048}, "2048x2048"},
        {{64, 256, 256}, "64x256x256"},
    };

    for (const auto& [shape, name] : shapes) {
        auto a = randn(shape, DType::Float32, g_bench_device);

        size_t num_elements = 1;
        for (auto dim : shape) {
            num_elements *= dim;
        }

        // Sum
        {
            Benchmark bench("Sum - " + name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops::elementwise(num_elements, 1));
            bench.set_bytes(num_elements * 4);  // Read all elements

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto s = sum(a);
                volatile void* ptr = s.data_ptr();
                (void)ptr;
            });
            result.print();
            collect_result(result);
        }

        // Mean
        {
            Benchmark bench("Mean - " + name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops::elementwise(num_elements, 2));  // sum + divide
            bench.set_bytes(num_elements * 4);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto m = mean(a);
                volatile void* ptr = m.data_ptr();
                (void)ptr;
            });
            result.print();
            collect_result(result);
        }
    }
}

/**
 * @brief Benchmark backward pass (autograd)
 */
void benchmark_backward() {
    std::cout << "\n========================================\n";
    std::cout << "  Backward Pass (Autograd)\n";
    std::cout << "========================================\n\n";

    // MatMul backward
    {
        auto a = Variable(randn({512, 512}, DType::Float32, g_bench_device), true);
        auto b = Variable(randn({512, 512}, DType::Float32, g_bench_device), true);

        Benchmark bench("MatMul Backward (512x512)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto c = matmul(a.tensor(), b.tensor());
            auto var_c = Variable(c, true);

            // Create gradient
            auto grad = ones_like(c);
            var_c.backward(grad);

            // Clear gradients for next iteration
            a.zero_grad();
            b.zero_grad();
        });
        result.print();
        collect_result(result);
    }

    // Element-wise backward
    {
        auto a = Variable(randn({1024, 1024}, DType::Float32, g_bench_device), true);
        auto b = Variable(randn({1024, 1024}, DType::Float32, g_bench_device), true);

        Benchmark bench("Add Backward (1024x1024)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto c = add(a.tensor(), b.tensor());
            auto var_c = Variable(c, true);

            auto grad = ones_like(c);
            var_c.backward(grad);

            a.zero_grad();
            b.zero_grad();
        });
        result.print();
        collect_result(result);
    }
}

// Global results collector for JSON output
static std::vector<BenchmarkResult> g_all_results;

static void collect_result(const BenchmarkResult& result) {
    g_all_results.push_back(result);
}

int main(int argc, char** argv) {
    bool json_output = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            json_output = true;
        }
    }
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);

    if (!json_output) {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Tenzor Operations Benchmark Suite\n";
        std::cout << "  device=" << g_bench_device.to_string() << "\n";
        std::cout << "========================================\n";
        std::cout << "\nTarget Performance Metrics:\n";
        std::cout << "  MatMul (4096x4096):  < 20ms (PyTorch: 22ms)\n";
        std::cout << "  Backward Pass:       < 2x forward time\n";
        std::cout << "\n";
    }

    try {
        // Initialize Tenzor
        initialize();

        // Suppress stdout for JSON mode by redirecting cout
        std::streambuf* original_buf = nullptr;
        std::ostringstream null_stream;
        if (json_output) {
            original_buf = std::cout.rdbuf(null_stream.rdbuf());
        }

        // Run benchmark suites
        benchmark_matmul_suite();
        benchmark_bmm();
        benchmark_elementwise();
        benchmark_reductions();
        benchmark_backward();

        // Restore cout
        if (json_output && original_buf) {
            std::cout.rdbuf(original_buf);
        }

        if (json_output) {
            // Output JSON array of all results
            BenchmarkSuite suite("ops");
            std::cout << suite.export_json(g_all_results);
        } else {
            std::cout << "\n========================================\n";
            std::cout << "  Benchmark Complete\n";
            std::cout << "========================================\n\n";
        }

        // Finalize
        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
