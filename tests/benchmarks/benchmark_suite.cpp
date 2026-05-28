/**
 * @file benchmark_suite.cpp
 * @brief Comprehensive performance benchmarks for Tenzor operations
 */

#include "tenzor/utils/benchmark.hpp"
#include "tenzor/backends/cpu/simd.hpp"
#include "tenzor/backend/runtime_simd.hpp"
#include <iostream>
#include <vector>
#include <random>

using namespace tenzor;
using namespace tenzor::benchmark;
using namespace tenzor::cpu;

// ============================================================================
// Helper Functions
// ============================================================================

std::vector<float> generate_random(size_t size, float min = -1.0f, float max = 1.0f) {
    std::random_device rd;
    std::mt19937 gen(42);
    std::uniform_real_distribution<float> dis(min, max);

    std::vector<float> result(size);
    for (size_t i = 0; i < size; ++i) {
        result[i] = dis(gen);
    }
    return result;
}

// ============================================================================
// Element-wise Operation Benchmarks
// ============================================================================

void benchmark_elementwise_ops() {
    std::cout << "\n========================================\n";
    std::cout << "  Element-wise Operations\n";
    std::cout << "========================================\n";

    const size_t sizes[] = {1000, 10000, 100000, 1000000};

    for (size_t size : sizes) {
        auto a = generate_random(size);
        auto b = generate_random(size);
        std::vector<float> out(size);

        size_t num_ops = size;  // 1 op per element
        size_t num_bytes = memory::elementwise(size, 2, 1);  // 2 inputs, 1 output

        // Benchmark Add
        {
            auto bench = TENZOR_BENCHMARK("Add_" + std::to_string(size), 10, 100)
                .set_flops(num_ops)
                .set_bytes(num_bytes);

            auto result = bench.run([&]() {
                simd::add(a.data(), b.data(), out.data(), size);
            });

            result.print();
        }

        // Benchmark Mul
        {
            auto bench = TENZOR_BENCHMARK("Mul_" + std::to_string(size), 10, 100)
                .set_flops(num_ops)
                .set_bytes(num_bytes);

            auto result = bench.run([&]() {
                simd::mul(a.data(), b.data(), out.data(), size);
            });

            result.print();
        }

        // Benchmark FMA
        {
            auto c = generate_random(size);
            size_t fma_ops = 2 * size;  // 2 ops per element (mul + add)
            size_t fma_bytes = memory::elementwise(size, 3, 1);

            auto bench = TENZOR_BENCHMARK("FMA_" + std::to_string(size), 10, 100)
                .set_flops(fma_ops)
                .set_bytes(fma_bytes);

            auto result = bench.run([&]() {
                simd::fma(a.data(), b.data(), c.data(), out.data(), size);
            });

            result.print();
        }
    }
}

// ============================================================================
// Activation Function Benchmarks
// ============================================================================

void benchmark_activations() {
    std::cout << "\n========================================\n";
    std::cout << "  Activation Functions\n";
    std::cout << "========================================\n";

    const size_t sizes[] = {1000, 10000, 100000, 1000000};

    for (size_t size : sizes) {
        auto a = generate_random(size, -5.0f, 5.0f);
        std::vector<float> out(size);

        size_t num_bytes = memory::elementwise(size, 1, 1);

        // ReLU
        {
            auto bench = TENZOR_BENCHMARK("ReLU_" + std::to_string(size), 10, 100)
                .set_flops(size)  // 1 comparison per element
                .set_bytes(num_bytes);

            auto result = bench.run([&]() {
                simd::relu(a.data(), out.data(), size);
            });

            result.print();
        }

        // Sigmoid
        {
            auto bench = TENZOR_BENCHMARK("Sigmoid_" + std::to_string(size), 10, 100)
                .set_flops(size * 4)  // Approximate: exp + div + add + mul
                .set_bytes(num_bytes);

            auto result = bench.run([&]() {
                simd::sigmoid(a.data(), out.data(), size);
            });

            result.print();
        }

        // Tanh
        {
            auto bench = TENZOR_BENCHMARK("Tanh_" + std::to_string(size), 10, 100)
                .set_flops(size * 4)
                .set_bytes(num_bytes);

            auto result = bench.run([&]() {
                simd::tanh(a.data(), out.data(), size);
            });

            result.print();
        }

        // GELU
        {
            auto bench = TENZOR_BENCHMARK("GELU_" + std::to_string(size), 10, 100)
                .set_flops(size * 8)  // Complex approximation
                .set_bytes(num_bytes);

            auto result = bench.run([&]() {
                simd::gelu(a.data(), out.data(), size);
            });

            result.print();
        }
    }
}

// ============================================================================
// SIMD vs Scalar Comparison
// ============================================================================

void benchmark_simd_comparison() {
    std::cout << "\n========================================\n";
    std::cout << "  SIMD vs Scalar Comparison\n";
    std::cout << "========================================\n";

    const size_t size = 1000000;
    auto a = generate_random(size);
    auto b = generate_random(size);
    std::vector<float> out(size);

    std::cout << "\n--- Addition ---\n";

    // SIMD Add
    {
        auto bench = TENZOR_BENCHMARK("Add_SIMD", 10, 100);
        auto result = bench.run([&]() {
            simd::add(a.data(), b.data(), out.data(), size);
        });
        result.print();
    }

    // Scalar Add
    {
        auto bench = TENZOR_BENCHMARK("Add_Scalar", 10, 100);
        auto result = bench.run([&]() {
            scalar::add(a.data(), b.data(), out.data(), size);
        });
        result.print();
    }

    std::cout << "\n--- Multiplication ---\n";

    // SIMD Mul
    {
        auto bench = TENZOR_BENCHMARK("Mul_SIMD", 10, 100);
        auto result = bench.run([&]() {
            simd::mul(a.data(), b.data(), out.data(), size);
        });
        result.print();
    }

    // Scalar Mul
    {
        auto bench = TENZOR_BENCHMARK("Mul_Scalar", 10, 100);
        auto result = bench.run([&]() {
            scalar::mul(a.data(), b.data(), out.data(), size);
        });
        result.print();
    }

    std::cout << "\n--- ReLU ---\n";

    // SIMD ReLU
    {
        auto bench = TENZOR_BENCHMARK("ReLU_SIMD", 10, 100);
        auto result = bench.run([&]() {
            simd::relu(a.data(), out.data(), size);
        });
        result.print();
    }

    // Scalar ReLU
    {
        auto bench = TENZOR_BENCHMARK("ReLU_Scalar", 10, 100);
        auto result = bench.run([&]() {
            scalar::relu(a.data(), out.data(), size);
        });
        result.print();
    }
}

// ============================================================================
// Memory Bandwidth Benchmarks
// ============================================================================

void benchmark_memory_bandwidth() {
    std::cout << "\n========================================\n";
    std::cout << "  Memory Bandwidth\n";
    std::cout << "========================================\n";

    const size_t sizes[] = {1000, 10000, 100000, 1000000, 10000000};

    for (size_t size : sizes) {
        auto a = generate_random(size);
        auto b = generate_random(size);
        std::vector<float> out(size);

        size_t bytes_read = size * sizeof(float) * 2;   // a + b
        size_t bytes_write = size * sizeof(float);      // out
        size_t total_bytes = bytes_read + bytes_write;

        auto bench = TENZOR_BENCHMARK("Bandwidth_" + std::to_string(size), 5, 50)
            .set_bytes(total_bytes);

        auto result = bench.run([&]() {
            simd::add(a.data(), b.data(), out.data(), size);
        });

        result.print();
    }
}

// ============================================================================
// Latency vs Throughput
// ============================================================================

void benchmark_latency_throughput() {
    std::cout << "\n========================================\n";
    std::cout << "  Latency vs Throughput\n";
    std::cout << "========================================\n";

    // Small size (latency-bound)
    {
        const size_t size = 100;
        auto a = generate_random(size);
        auto b = generate_random(size);
        std::vector<float> out(size);

        auto bench = TENZOR_BENCHMARK("Latency_Small_100", 10, 1000);
        auto result = bench.run([&]() {
            simd::add(a.data(), b.data(), out.data(), size);
        });
        result.print();
    }

    // Large size (throughput-bound)
    {
        const size_t size = 10000000;
        auto a = generate_random(size);
        auto b = generate_random(size);
        std::vector<float> out(size);

        auto bench = TENZOR_BENCHMARK("Throughput_Large_10M", 5, 20)
            .set_bytes(size * sizeof(float) * 3);

        auto result = bench.run([&]() {
            simd::add(a.data(), b.data(), out.data(), size);
        });
        result.print();
    }
}

// ============================================================================
// Fused Operations
// ============================================================================

void benchmark_fused_ops() {
    std::cout << "\n========================================\n";
    std::cout << "  Fused Operations\n";
    std::cout << "========================================\n";

    const size_t size = 1000000;
    auto a = generate_random(size);
    auto b = generate_random(size);
    auto c = generate_random(size);
    std::vector<float> out(size);
    std::vector<float> temp(size);

    // Unfused: (a * b) + c
    {
        auto bench = TENZOR_BENCHMARK("Unfused_MulAdd", 10, 100)
            .set_flops(2 * size);

        auto result = bench.run([&]() {
            simd::mul(a.data(), b.data(), temp.data(), size);
            simd::add(temp.data(), c.data(), out.data(), size);
        });
        result.print();
    }

    // Fused: FMA
    {
        auto bench = TENZOR_BENCHMARK("Fused_FMA", 10, 100)
            .set_flops(2 * size);

        auto result = bench.run([&]() {
            simd::fma(a.data(), b.data(), c.data(), out.data(), size);
        });
        result.print();
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    std::cout << "========================================\n";
    std::cout << "  Tenzor Benchmark Suite\n";
    std::cout << "========================================\n";

    // Print CPU SIMD features (runtime_simd canonical API).
    const auto& cpu = ::tenzor::backend::get_simd_features();
    std::cout << "\nCPU SIMD Information:\n";
    std::cout << "  Features: " << cpu.to_string() << "\n";

    // Run benchmarks
    benchmark_elementwise_ops();
    benchmark_activations();
    benchmark_memory_bandwidth();
    benchmark_simd_comparison();
    benchmark_latency_throughput();
    benchmark_fused_ops();

    std::cout << "\n========================================\n";
    std::cout << "  Benchmark Suite Complete\n";
    std::cout << "========================================\n";

    return 0;
}
