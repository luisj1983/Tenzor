/**
 * @file benchmark_backends.cpp
 * @brief Comprehensive cross-backend performance benchmarks
 *
 * Compares performance across CPU, CUDA, Vulkan, and OneAPI backends
 * for all major tensor operations.
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/ops/fused_ops.hpp>
#include <iostream>
#include <iomanip>
#include <chrono>
#include <vector>
#include <string>
#include <map>
#include <fstream>
#include <algorithm>
#include <numeric>

using namespace tenzor;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    int warmup_iterations = 5;
    int measure_iterations = 20;
    bool csv_output = false;
    std::string csv_file = "benchmark_results.csv";
    bool verbose = false;
};

// ============================================================================
// Benchmark Result
// ============================================================================

struct BenchmarkResult {
    std::string backend;
    std::string operation;
    std::string shape;
    std::string dtype;
    double mean_time_ms;
    double std_time_ms;
    double min_time_ms;
    double max_time_ms;
    double throughput_gflops;
    double bandwidth_gbps;
    int64_t num_elements;
    bool success;
    std::string error_msg;
};

// ============================================================================
// Timer Utility
// ============================================================================

class Timer {
public:
    void start() {
        start_ = std::chrono::high_resolution_clock::now();
    }

    double stop_ms() {
        auto end = std::chrono::high_resolution_clock::now();
        return std::chrono::duration<double, std::milli>(end - start_).count();
    }

private:
    std::chrono::high_resolution_clock::time_point start_;
};

// ============================================================================
// Benchmark Runner
// ============================================================================

class BackendBenchmark {
public:
    BackendBenchmark(const BenchmarkConfig& config) : config_(config) {}

    // Get available backends
    std::vector<Device> getAvailableBackends() {
        std::vector<Device> backends;

        // CPU is always available
        backends.push_back(Device::cpu());

        // Check CUDA
        try {
            auto cuda_dev = Device::cuda(0);
            auto test = zeros({2, 2}, DType::Float32, cuda_dev);
            backends.push_back(cuda_dev);
        } catch (...) {}

        // Check Vulkan
        try {
            auto vulkan_dev = Device::vulkan(0);
            auto test = zeros({2, 2}, DType::Float32, vulkan_dev);
            backends.push_back(vulkan_dev);
        } catch (const std::exception& e) {
            std::cerr << "Vulkan detection failed: " << e.what() << std::endl;
        }

        // Check OneAPI
        try {
            auto oneapi_dev = Device::oneapi(0);
            auto test = zeros({2, 2}, DType::Float32, oneapi_dev);
            backends.push_back(oneapi_dev);
        } catch (...) {}

        return backends;
    }

    // Synchronize device
    void syncDevice(const Device& device) {
        if (device.type == Device::Type::CUDA) {
            // CUDA sync
            auto t = zeros({1}, DType::Float32, device);
            t.to(Device::cpu());
        } else if (device.type == Device::Type::Vulkan) {
            // Vulkan sync - force completion by transferring to CPU
            auto t = zeros({1}, DType::Float32, device);
            t.to(Device::cpu());
        } else if (device.type == Device::Type::OneAPI) {
            auto t = zeros({1}, DType::Float32, device);
            t.to(Device::cpu());
        }
    }

    // Run a single benchmark
    template<typename Func>
    BenchmarkResult runBenchmark(
        const std::string& backend_name,
        const std::string& op_name,
        const std::string& shape_str,
        const std::string& dtype_str,
        int64_t num_elements,
        int64_t flops,
        int64_t bytes,
        const Device& device,
        Func&& func
    ) {
        BenchmarkResult result;
        result.backend = backend_name;
        result.operation = op_name;
        result.shape = shape_str;
        result.dtype = dtype_str;
        result.num_elements = num_elements;
        result.success = true;

        try {
            // Warmup
            for (int i = 0; i < config_.warmup_iterations; ++i) {
                func();
                syncDevice(device);
            }

            // Measure
            std::vector<double> times;
            Timer timer;

            for (int i = 0; i < config_.measure_iterations; ++i) {
                timer.start();
                func();
                syncDevice(device);
                times.push_back(timer.stop_ms());
            }

            // Calculate statistics
            double sum = std::accumulate(times.begin(), times.end(), 0.0);
            result.mean_time_ms = sum / times.size();

            double sq_sum = std::inner_product(times.begin(), times.end(), times.begin(), 0.0);
            result.std_time_ms = std::sqrt(sq_sum / times.size() - result.mean_time_ms * result.mean_time_ms);

            result.min_time_ms = *std::min_element(times.begin(), times.end());
            result.max_time_ms = *std::max_element(times.begin(), times.end());

            // Calculate throughput
            if (flops > 0) {
                result.throughput_gflops = (flops / 1e9) / (result.mean_time_ms / 1000.0);
            }
            if (bytes > 0) {
                result.bandwidth_gbps = (bytes / 1e9) / (result.mean_time_ms / 1000.0);
            }

        } catch (const std::exception& e) {
            result.success = false;
            result.error_msg = e.what();
        }

        return result;
    }

    // Print result
    void printResult(const BenchmarkResult& r) {
        if (r.success) {
            std::cout << std::left << std::setw(10) << r.backend
                      << std::setw(20) << r.operation
                      << std::setw(20) << r.shape
                      << std::setw(10) << r.dtype
                      << std::right << std::setw(12) << std::fixed << std::setprecision(3) << r.mean_time_ms
                      << std::setw(12) << r.std_time_ms
                      << std::setw(12) << r.throughput_gflops
                      << std::setw(12) << r.bandwidth_gbps
                      << "\n";
        } else {
            std::cout << std::left << std::setw(10) << r.backend
                      << std::setw(20) << r.operation
                      << std::setw(20) << r.shape
                      << std::setw(10) << r.dtype
                      << "  FAILED: " << r.error_msg.substr(0, 40) << "\n";
        }
    }

    void printHeader() {
        std::cout << std::left << std::setw(10) << "Backend"
                  << std::setw(20) << "Operation"
                  << std::setw(20) << "Shape"
                  << std::setw(10) << "DType"
                  << std::right << std::setw(12) << "Mean(ms)"
                  << std::setw(12) << "Std(ms)"
                  << std::setw(12) << "GFLOPS"
                  << std::setw(12) << "GB/s"
                  << "\n";
        std::cout << std::string(108, '-') << "\n";
    }

    // Save results to CSV
    void saveCSV(const std::vector<BenchmarkResult>& results) {
        std::ofstream file(config_.csv_file);
        file << "backend,operation,shape,dtype,mean_ms,std_ms,min_ms,max_ms,gflops,gbps,elements,success\n";
        for (const auto& r : results) {
            file << r.backend << "," << r.operation << "," << r.shape << "," << r.dtype << ","
                 << r.mean_time_ms << "," << r.std_time_ms << "," << r.min_time_ms << "," << r.max_time_ms << ","
                 << r.throughput_gflops << "," << r.bandwidth_gbps << "," << r.num_elements << ","
                 << (r.success ? "true" : "false") << "\n";
        }
        std::cout << "\nResults saved to: " << config_.csv_file << "\n";
    }

private:
    BenchmarkConfig config_;
};

// ============================================================================
// Shape Helper
// ============================================================================

std::string shapeToString(const std::vector<int64_t>& shape) {
    std::string result = "[";
    for (size_t i = 0; i < shape.size(); ++i) {
        result += std::to_string(shape[i]);
        if (i < shape.size() - 1) result += ",";
    }
    result += "]";
    return result;
}

int64_t shapeToNumel(const std::vector<int64_t>& shape) {
    int64_t result = 1;
    for (auto s : shape) result *= s;
    return result;
}

// ============================================================================
// Benchmark Functions
// ============================================================================

void benchmarkElementwise(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  ELEMENTWISE OPERATIONS\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    // Test shapes
    std::vector<std::vector<int64_t>> shapes = {
        {1000},
        {10000},
        {100000},
        {1000000},
        {10000000},
        {1024, 1024},
        {2048, 2048},
    };

    // Test dtypes
    std::vector<DType> dtypes = {DType::Float32, DType::Float16};

    for (const auto& shape : shapes) {
        for (auto dtype : dtypes) {
            int64_t numel = shapeToNumel(shape);
            int64_t dtype_size = (dtype == DType::Float16) ? 2 : 4;
            std::string shape_str = shapeToString(shape);
            std::string dtype_str = (dtype == DType::Float16) ? "fp16" : "fp32";

            for (const auto& device : backends) {
                std::string backend_name = device.to_string();

                // Create tensors
                Tensor a, b;
                try {
                    a = ones(shape, dtype, device);
                    b = full(shape, 2.0f, dtype, device);
                } catch (const std::exception& e) {
                    std::cerr << backend_name << " tensor creation failed: " << e.what() << std::endl;
                    continue;
                }

                // Add
                {
                    int64_t flops = numel;
                    int64_t bytes = numel * dtype_size * 3; // 2 read + 1 write
                    auto r = bench.runBenchmark(backend_name, "add", shape_str, dtype_str,
                                                numel, flops, bytes, device,
                                                [&]() { auto c = add(a, b); });
                    results.push_back(r);
                    bench.printResult(r);
                }

                // Mul
                {
                    int64_t flops = numel;
                    int64_t bytes = numel * dtype_size * 3;
                    auto r = bench.runBenchmark(backend_name, "mul", shape_str, dtype_str,
                                                numel, flops, bytes, device,
                                                [&]() { auto c = mul(a, b); });
                    results.push_back(r);
                    bench.printResult(r);
                }

                // Exp
                {
                    int64_t flops = numel * 10; // approximate
                    int64_t bytes = numel * dtype_size * 2;
                    auto r = bench.runBenchmark(backend_name, "exp", shape_str, dtype_str,
                                                numel, flops, bytes, device,
                                                [&]() { auto c = exp(a); });
                    results.push_back(r);
                    bench.printResult(r);
                }
            }
        }
    }
}

void benchmarkReductions(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  REDUCTION OPERATIONS\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {10000},
        {100000},
        {1000000},
        {10000000},
        {1024, 1024},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = ones(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Sum (all elements)
            {
                int64_t flops = numel;
                int64_t bytes = numel * 4 + 4;
                auto r = bench.runBenchmark(backend_name, "sum_all", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = sum(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Mean
            {
                int64_t flops = numel + 1;
                int64_t bytes = numel * 4 + 4;
                auto r = bench.runBenchmark(backend_name, "mean_all", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = mean(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Max
            {
                int64_t flops = numel;
                int64_t bytes = numel * 4 + 4;
                auto r = bench.runBenchmark(backend_name, "max_all", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = max(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Norm (L2)
            {
                int64_t flops = numel * 3; // square + sum + sqrt
                int64_t bytes = numel * 4 + 4;
                auto r = bench.runBenchmark(backend_name, "norm_l2", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = norm(a); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkMatmul(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  MATRIX MULTIPLICATION\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    // M, N, K dimensions
    std::vector<std::tuple<int64_t, int64_t, int64_t>> dims = {
        {128, 128, 128},
        {256, 256, 256},
        {512, 512, 512},
        {1024, 1024, 1024},
        {2048, 2048, 2048},
        {4096, 4096, 4096},
        // Common transformer shapes
        {1, 768, 768},      // Single token
        {128, 768, 768},    // Batch of 128
        {128, 768, 3072},   // FFN expansion
        {128, 3072, 768},   // FFN contraction
    };

    std::vector<DType> dtypes = {DType::Float32, DType::Float16};

    for (auto [M, N, K] : dims) {
        for (auto dtype : dtypes) {
            std::string shape_str = std::to_string(M) + "x" + std::to_string(K) + " @ " +
                                    std::to_string(K) + "x" + std::to_string(N);
            std::string dtype_str = (dtype == DType::Float16) ? "fp16" : "fp32";
            int64_t dtype_size = (dtype == DType::Float16) ? 2 : 4;

            // FLOPS for matmul: 2*M*N*K (multiply-add)
            int64_t flops = 2 * M * N * K;
            // Bytes: read A (M*K) + read B (K*N) + write C (M*N)
            int64_t bytes = (M * K + K * N + M * N) * dtype_size;
            int64_t numel = M * N;

            for (const auto& device : backends) {
                std::string backend_name = device.to_string();

                Tensor a, b;
                try {
                    a = ones({M, K}, dtype, device);
                    b = ones({K, N}, dtype, device);
                } catch (...) {
                    continue;
                }

                auto r = bench.runBenchmark(backend_name, "matmul", shape_str, dtype_str,
                                            numel, flops, bytes, device,
                                            [&]() { auto c = matmul(a, b); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkBMM(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  BATCHED MATRIX MULTIPLICATION\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    // Batch, M, N, K
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> dims = {
        {8, 64, 64, 64},
        {16, 128, 128, 128},
        {32, 256, 256, 256},
        // Attention patterns
        {12, 128, 128, 64},   // 12 heads, seq=128, head_dim=64
        {12, 512, 512, 64},   // Longer sequence
        {16, 128, 128, 64},   // 16 heads
    };

    for (auto [B, M, N, K] : dims) {
        std::string shape_str = "[" + std::to_string(B) + "," + std::to_string(M) + "," +
                                std::to_string(K) + "]@[" + std::to_string(B) + "," +
                                std::to_string(K) + "," + std::to_string(N) + "]";

        int64_t flops = B * 2 * M * N * K;
        int64_t bytes = B * (M * K + K * N + M * N) * 4;
        int64_t numel = B * M * N;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a, b;
            try {
                a = ones({B, M, K}, DType::Float32, device);
                b = ones({B, K, N}, DType::Float32, device);
            } catch (...) {
                continue;
            }

            auto r = bench.runBenchmark(backend_name, "bmm", shape_str, "fp32",
                                        numel, flops, bytes, device,
                                        [&]() { auto c = bmm(a, b); });
            results.push_back(r);
            bench.printResult(r);
        }
    }
}

void benchmarkActivations(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  ACTIVATION / UNARY FUNCTIONS\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {100000},
        {1000000},
        {10000000},
        {1024, 1024},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);

        int64_t flops = numel;
        int64_t bytes = numel * 4 * 2;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = ones(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // ReLU via clamp_min
            {
                auto r = bench.runBenchmark(backend_name, "relu(clamp)", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = clamp_min(a, 0.0f); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Tanh
            {
                auto r = bench.runBenchmark(backend_name, "tanh", shape_str, "fp32",
                                            numel, flops * 5, bytes, device,
                                            [&]() { auto c = tanh(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Abs
            {
                auto r = bench.runBenchmark(backend_name, "abs", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = abs(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Neg
            {
                auto r = bench.runBenchmark(backend_name, "neg", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() { auto c = neg(a); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkPow(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  POWER / SQRT OPERATIONS\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {100000},
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);

        int64_t bytes = numel * 4 * 2;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = full(shape, 2.0f, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Sqrt
            {
                auto r = bench.runBenchmark(backend_name, "sqrt", shape_str, "fp32",
                                            numel, numel * 5, bytes, device,
                                            [&]() { auto c = sqrt(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Log
            {
                auto r = bench.runBenchmark(backend_name, "log", shape_str, "fp32",
                                            numel, numel * 10, bytes, device,
                                            [&]() { auto c = log(a); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkComparison(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  COMPARISON OPERATIONS\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);

        int64_t bytes = numel * 4 * 3; // 2 inputs + 1 bool output

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a, b;
            try {
                a = ones(shape, DType::Float32, device);
                b = full(shape, 0.5f, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Greater than
            {
                auto r = bench.runBenchmark(backend_name, "gt", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = gt(a, b); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Equal
            {
                auto r = bench.runBenchmark(backend_name, "eq", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = eq(a, b); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkTranspose(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  TRANSPOSE / PERMUTE\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1024, 1024},
        {2048, 2048},
        {4096, 4096},
        {128, 12, 64, 64},     // Attention reshape
        {8, 128, 768},         // Sequence transpose
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);

        int64_t bytes = numel * 4 * 2;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = ones(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            if (shape.size() == 2) {
                auto r = bench.runBenchmark(backend_name, "transpose", shape_str, "fp32",
                                            numel, 0, bytes, device,
                                            [&]() { auto c = transpose(a, 0, 1); });
                results.push_back(r);
                bench.printResult(r);
            } else if (shape.size() >= 3) {
                auto r = bench.runBenchmark(backend_name, "transpose", shape_str, "fp32",
                                            numel, 0, bytes, device,
                                            [&]() { auto c = transpose(a, 1, 2); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkCat(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  CONCATENATION\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1024, 1024},
        {4096, 4096},
        {128, 768},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape) + " x4";

        int64_t bytes = numel * 4 * 5; // 4 inputs + 1 output

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a, b, c, d;
            try {
                a = ones(shape, DType::Float32, device);
                b = ones(shape, DType::Float32, device);
                c = ones(shape, DType::Float32, device);
                d = ones(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            auto r = bench.runBenchmark(backend_name, "cat", shape_str, "fp32",
                                        numel * 4, 0, bytes, device,
                                        [&]() { auto e = cat({a, b, c, d}, 0); });
            results.push_back(r);
            bench.printResult(r);
        }
    }
}

// ============================================================================
// Additional Benchmark Functions (Extended Coverage)
// ============================================================================

void benchmarkMoreElementwise(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  ADDITIONAL ELEMENTWISE (sub, div, pow)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        int64_t bytes = numel * 4 * 3;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a, b;
            try {
                a = full(shape, 4.0f, DType::Float32, device);
                b = full(shape, 2.0f, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Sub
            {
                auto r = bench.runBenchmark(backend_name, "sub", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = sub(a, b); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Div
            {
                auto r = bench.runBenchmark(backend_name, "div", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = div(a, b); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Pow
            {
                auto r = bench.runBenchmark(backend_name, "pow", shape_str, "fp32",
                                            numel, numel * 10, bytes, device,
                                            [&]() { auto c = pow(a, 2.0f); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkTrigonometric(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  TRIGONOMETRIC (sin, cos)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        int64_t bytes = numel * 4 * 2;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = full(shape, 1.0f, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Sin
            {
                auto r = bench.runBenchmark(backend_name, "sin", shape_str, "fp32",
                                            numel, numel * 10, bytes, device,
                                            [&]() { auto c = sin(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Cos
            {
                auto r = bench.runBenchmark(backend_name, "cos", shape_str, "fp32",
                                            numel, numel * 10, bytes, device,
                                            [&]() { auto c = cos(a); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkMoreActivations(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  MORE ACTIVATIONS (sigmoid, tanh, leaky_relu)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        int64_t bytes = numel * 4 * 2;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = randn(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Sigmoid: 1 / (1 + exp(-x))
            {
                auto r = bench.runBenchmark(backend_name, "sigmoid", shape_str, "fp32",
                                            numel, numel * 5, bytes, device,
                                            [&]() {
                                                auto neg_a = neg(a);
                                                auto exp_neg = exp(neg_a);
                                                auto one_plus = add(full(shape, 1.0f, DType::Float32, device), exp_neg);
                                                auto c = div(full(shape, 1.0f, DType::Float32, device), one_plus);
                                            });
                results.push_back(r);
                bench.printResult(r);
            }

            // Tanh: (exp(2x) - 1) / (exp(2x) + 1)
            {
                auto r = bench.runBenchmark(backend_name, "tanh", shape_str, "fp32",
                                            numel, numel * 6, bytes, device,
                                            [&]() {
                                                auto two_x = mul(a, full(shape, 2.0f, DType::Float32, device));
                                                auto exp_2x = exp(two_x);
                                                auto one = full(shape, 1.0f, DType::Float32, device);
                                                auto num = sub(exp_2x, one);
                                                auto denom = add(exp_2x, one);
                                                auto c = div(num, denom);
                                            });
                results.push_back(r);
                bench.printResult(r);
            }

            // Leaky ReLU via clamp + where pattern
            {
                auto r = bench.runBenchmark(backend_name, "leaky_relu", shape_str, "fp32",
                                            numel, numel * 2, bytes, device,
                                            [&]() {
                                                auto alpha = full(shape, 0.01f, DType::Float32, device);
                                                auto scaled = mul(a, alpha);
                                                auto mask = gt(a, zeros(shape, DType::Float32, device));
                                                auto c = where(mask, a, scaled);
                                            });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkSoftmax(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  SOFTMAX OPERATIONS\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    // Typical shapes for softmax (batch, seq_len, vocab or hidden)
    std::vector<std::vector<int64_t>> shapes = {
        {128, 1000},      // Classification
        {128, 50000},     // Large vocabulary
        {32, 128, 128},   // Attention scores
        {16, 12, 128, 128}, // Multi-head attention
        {1024, 1024},     // Square
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        // Softmax: exp + sum + div per element
        int64_t flops = numel * 5;
        int64_t bytes = numel * 4 * 2;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = randn(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Manual softmax: exp(x - max) / sum(exp(x - max))
            {
                int64_t dim = shape.size() - 1;
                auto r = bench.runBenchmark(backend_name, "softmax", shape_str, "fp32",
                                            numel, flops, bytes, device,
                                            [&]() {
                                                // Approximate softmax via exp/sum pattern
                                                auto e = exp(a);
                                                auto s = sum(e, dim, true);
                                                auto c = div(e, s);
                                            });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkConv2d(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  CONVOLUTION 2D (via fused_conv2d_relu)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    // (batch, in_channels, H, W, out_channels, kernel_size)
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t, int64_t>> configs = {
        {1, 3, 224, 224, 64, 7},      // ResNet first layer
        {1, 64, 56, 56, 64, 3},       // ResNet conv
        {1, 128, 28, 28, 128, 3},     // Deeper layer
        {1, 256, 14, 14, 256, 3},     // Even deeper
        {8, 64, 56, 56, 64, 3},       // Batched
        {16, 128, 28, 28, 128, 3},    // Larger batch
        {32, 64, 32, 32, 128, 3},     // Common size
    };

    for (auto [N, C_in, H, W, C_out, K] : configs) {
        std::string shape_str = "N=" + std::to_string(N) + " C=" + std::to_string(C_in) +
                                "->" + std::to_string(C_out) + " " + std::to_string(H) +
                                "x" + std::to_string(W) + " k=" + std::to_string(K);

        int64_t H_out = H - K + 1;  // No padding, stride=1
        int64_t W_out = W - K + 1;
        // FLOPs: 2 * N * C_out * H_out * W_out * C_in * K * K
        int64_t flops = 2 * N * C_out * H_out * W_out * C_in * K * K;
        // Bytes: input + weight + output
        int64_t bytes = (N * C_in * H * W + C_out * C_in * K * K + N * C_out * H_out * W_out) * 4;
        int64_t numel = N * C_out * H_out * W_out;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor input, weight, bias;
            try {
                input = randn({N, C_in, H, W}, DType::Float32, device);
                weight = randn({C_out, C_in, K, K}, DType::Float32, device);
                bias = zeros({C_out}, DType::Float32, device);
            } catch (...) {
                continue;
            }

            auto r = bench.runBenchmark(backend_name, "conv2d+relu", shape_str, "fp32",
                                        numel, flops, bytes, device,
                                        [&]() { auto c = ops::fused_conv2d_relu(input, weight, &bias); });
            results.push_back(r);
            bench.printResult(r);
        }
    }
}

void benchmarkArgOps(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  ARGMAX / ARGMIN\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {128, 50000},   // Classification output
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        int64_t bytes = numel * 4 + 8;  // input + output index

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = randn(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Argmax along last dim
            {
                int64_t dim = shape.size() - 1;
                auto r = bench.runBenchmark(backend_name, "argmax", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = argmax(a, dim); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Argmin along last dim
            {
                int64_t dim = shape.size() - 1;
                auto r = bench.runBenchmark(backend_name, "argmin", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = argmin(a, dim); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

void benchmarkWhere(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  WHERE (CONDITIONAL SELECT)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        // condition + x + y + output
        int64_t bytes = numel * (1 + 4 + 4 + 4);

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor cond, x, y;
            try {
                auto temp = randn(shape, DType::Float32, device);
                cond = gt(temp, zeros(shape, DType::Float32, device));
                x = full(shape, 1.0f, DType::Float32, device);
                y = full(shape, -1.0f, DType::Float32, device);
            } catch (...) {
                continue;
            }

            auto r = bench.runBenchmark(backend_name, "where", shape_str, "fp32",
                                        numel, numel, bytes, device,
                                        [&]() { auto c = where(cond, x, y); });
            results.push_back(r);
            bench.printResult(r);
        }
    }
}

void benchmarkIndexSelect(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  INDEX SELECT (EMBEDDING LOOKUP)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    // (vocab_size, embed_dim, batch_size, seq_len)
    std::vector<std::tuple<int64_t, int64_t, int64_t, int64_t>> configs = {
        {50000, 768, 1, 128},     // BERT-like single
        {50000, 768, 16, 128},    // BERT-like batched
        {50000, 1024, 8, 256},    // Larger model
        {100000, 512, 32, 64},    // Large vocab
    };

    for (auto [vocab, embed, batch, seq] : configs) {
        std::string shape_str = "V=" + std::to_string(vocab) + " E=" + std::to_string(embed) +
                                " B=" + std::to_string(batch) + " S=" + std::to_string(seq);

        int64_t numel = batch * seq * embed;
        // Read: indices + embedding rows, Write: output
        int64_t bytes = (batch * seq * 8) + (batch * seq * embed * 4) + numel * 4;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor embedding, indices;
            try {
                embedding = randn({vocab, embed}, DType::Float32, device);
                // Create indices: floor(rand() * vocab) cast to Int64
                auto rand_vals = rand({batch, seq}, DType::Float32, device);
                auto scaled = mul(rand_vals, full({batch, seq}, static_cast<float>(vocab - 1), DType::Float32, device));
                indices = scaled.to(DType::Int64);
            } catch (...) {
                continue;
            }

            auto r = bench.runBenchmark(backend_name, "index_select", shape_str, "fp32",
                                        numel, numel, bytes, device,
                                        [&]() { auto c = index_select(embedding, 0, indices.flatten()); });
            results.push_back(r);
            bench.printResult(r);
        }
    }
}

void benchmarkMoreReductions(BackendBenchmark& bench, std::vector<BenchmarkResult>& results) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  MORE REDUCTIONS (var, std, prod)\n";
    std::cout << std::string(60, '=') << "\n\n";
    bench.printHeader();

    auto backends = bench.getAvailableBackends();

    std::vector<std::vector<int64_t>> shapes = {
        {1000000},
        {10000000},
        {2048, 2048},
    };

    for (const auto& shape : shapes) {
        int64_t numel = shapeToNumel(shape);
        std::string shape_str = shapeToString(shape);
        int64_t bytes = numel * 4 + 4;

        for (const auto& device : backends) {
            std::string backend_name = device.to_string();

            Tensor a;
            try {
                a = randn(shape, DType::Float32, device);
            } catch (...) {
                continue;
            }

            // Variance
            {
                auto r = bench.runBenchmark(backend_name, "var", shape_str, "fp32",
                                            numel, numel * 3, bytes, device,
                                            [&]() { auto c = var(a); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Std (via sqrt(var))
            {
                auto r = bench.runBenchmark(backend_name, "std", shape_str, "fp32",
                                            numel, numel * 3 + 1, bytes, device,
                                            [&]() { auto c = sqrt(var(a)); });
                results.push_back(r);
                bench.printResult(r);
            }

            // Min
            {
                auto r = bench.runBenchmark(backend_name, "min", shape_str, "fp32",
                                            numel, numel, bytes, device,
                                            [&]() { auto c = min(a); });
                results.push_back(r);
                bench.printResult(r);
            }
        }
    }
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    // Initialize Tenzor library (loads all backends)
    tenzor::initialize();

    std::cout << "\n";
    std::cout << "╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║       TENZOR CROSS-BACKEND BENCHMARK SUITE                   ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n";

    BenchmarkConfig config;

    // Parse arguments
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--csv") {
            config.csv_output = true;
        } else if (arg == "--csv-file" && i + 1 < argc) {
            config.csv_file = argv[++i];
            config.csv_output = true;
        } else if (arg == "--warmup" && i + 1 < argc) {
            config.warmup_iterations = std::stoi(argv[++i]);
        } else if (arg == "--iterations" && i + 1 < argc) {
            config.measure_iterations = std::stoi(argv[++i]);
        } else if (arg == "--help") {
            std::cout << "Usage: " << argv[0] << " [options]\n"
                      << "Options:\n"
                      << "  --csv                Output results to CSV\n"
                      << "  --csv-file <file>    Specify CSV output file\n"
                      << "  --warmup <n>         Warmup iterations (default: 5)\n"
                      << "  --iterations <n>     Measurement iterations (default: 20)\n";
            return 0;
        }
    }

    BackendBenchmark bench(config);
    std::vector<BenchmarkResult> results;

    // Print available backends
    auto backends = bench.getAvailableBackends();
    std::cout << "\nAvailable backends: ";
    for (const auto& b : backends) {
        std::cout << b.to_string() << " ";
    }
    std::cout << "\n";

    std::cout << "Warmup iterations: " << config.warmup_iterations << "\n";
    std::cout << "Measurement iterations: " << config.measure_iterations << "\n";

    // Run all benchmarks
    benchmarkElementwise(bench, results);
    benchmarkMoreElementwise(bench, results);
    benchmarkTrigonometric(bench, results);
    benchmarkReductions(bench, results);
    benchmarkMoreReductions(bench, results);
    benchmarkMatmul(bench, results);
    benchmarkBMM(bench, results);
    benchmarkActivations(bench, results);
    benchmarkMoreActivations(bench, results);
    benchmarkSoftmax(bench, results);
    benchmarkPow(bench, results);
    benchmarkComparison(bench, results);
    benchmarkTranspose(bench, results);
    benchmarkCat(bench, results);
    benchmarkConv2d(bench, results);
    benchmarkArgOps(bench, results);
    benchmarkWhere(bench, results);
    benchmarkIndexSelect(bench, results);

    // Summary
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "  SUMMARY\n";
    std::cout << std::string(60, '=') << "\n\n";

    int total = results.size();
    int passed = std::count_if(results.begin(), results.end(),
                               [](const BenchmarkResult& r) { return r.success; });

    std::cout << "Total benchmarks: " << total << "\n";
    std::cout << "Successful: " << passed << "\n";
    std::cout << "Failed: " << (total - passed) << "\n";

    // Save CSV if requested
    if (config.csv_output) {
        bench.saveCSV(results);
    }

    std::cout << "\n╔══════════════════════════════════════════════════════════════╗\n";
    std::cout << "║                    BENCHMARK COMPLETE                        ║\n";
    std::cout << "╚══════════════════════════════════════════════════════════════╝\n\n";

    return 0;
}
