/**
 * @file benchmark_backends.cpp
 * @brief Comprehensive cross-backend performance benchmarks
 *
 * Compares performance across CPU, CUDA, Vulkan, and OneAPI backends
 * for all major tensor operations.
 */

#include <tenzor/tenzor.hpp>
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
    benchmarkReductions(bench, results);
    benchmarkMatmul(bench, results);
    benchmarkBMM(bench, results);
    benchmarkActivations(bench, results);
    benchmarkPow(bench, results);
    benchmarkComparison(bench, results);
    benchmarkTranspose(bench, results);
    benchmarkCat(bench, results);

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
