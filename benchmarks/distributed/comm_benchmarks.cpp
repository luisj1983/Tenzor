/**
 * @file comm_benchmarks.cpp
 * @brief Communication bandwidth and latency benchmarks for distributed operations
 *
 * Measures performance of collective operations (AllReduce, Broadcast, AllGather,
 * ReduceScatter) for both NCCL and Gloo backends.
 *
 * Usage:
 *   export RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500
 *   ./comm_benchmarks &
 *   export RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500
 *   ./comm_benchmarks
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/distributed/distributed.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/ops/creation.hpp"
#include <iostream>
#include <chrono>
#include <vector>
#include <cstdlib>
#include <iomanip>
#include <cmath>

using namespace tenzor;
using namespace tenzor::distributed;
using namespace std::chrono;

// ============================================================================
// Benchmark Configuration
// ============================================================================

struct BenchmarkConfig {
    std::vector<size_t> message_sizes;  // Message sizes in bytes
    int num_warmup = 5;                 // Warmup iterations
    int num_iterations = 20;            // Measurement iterations
    bool test_gloo = true;              // Test Gloo backend
    bool test_nccl = true;              // Test NCCL backend (if available)
};

// ============================================================================
// Benchmark Results
// ============================================================================

struct BenchmarkResult {
    std::string operation;
    std::string backend;
    size_t message_size_bytes;
    double avg_time_ms;
    double min_time_ms;
    double max_time_ms;
    double bandwidth_gbps;
    double latency_us;
};

// ============================================================================
// Utility Functions
// ============================================================================

auto get_timestamp() -> double {
    return duration_cast<microseconds>(
        high_resolution_clock::now().time_since_epoch()
    ).count() / 1000.0;  // Convert to milliseconds
}

auto bytes_to_gb(size_t bytes) -> double {
    return bytes / (1024.0 * 1024.0 * 1024.0);
}

auto calculate_bandwidth_gbps(size_t bytes, double time_ms) -> double {
    return bytes_to_gb(bytes) / (time_ms / 1000.0);
}

auto print_result(const BenchmarkResult& result, int rank) -> void {
    if (rank != 0) return;  // Only root prints

    std::cout << std::setw(15) << result.operation
              << std::setw(8) << result.backend
              << std::setw(12) << result.message_size_bytes
              << std::setw(12) << std::fixed << std::setprecision(3) << result.avg_time_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << result.min_time_ms
              << std::setw(12) << std::fixed << std::setprecision(3) << result.max_time_ms
              << std::setw(15) << std::fixed << std::setprecision(2) << result.bandwidth_gbps
              << std::setw(15) << std::fixed << std::setprecision(1) << result.latency_us
              << std::endl;
}

// ============================================================================
// Collective Operation Benchmarks
// ============================================================================

auto benchmark_all_reduce(
    const BenchmarkConfig& config,
    const std::string& backend_name,
    int rank,
    int world_size
) -> std::vector<BenchmarkResult> {
    std::vector<BenchmarkResult> results;

    for (size_t msg_size : config.message_sizes) {
        size_t num_elements = msg_size / sizeof(float);
        if (num_elements == 0) num_elements = 1;

        Device device = Device::cpu();
        if (backend_name == "nccl") {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
            device = Device::cuda(0);
#else
            continue;  // Skip NCCL if CUDA not available
#endif
        }

        Tensor data = ones({static_cast<int64_t>(num_elements)}, DType::Float32, device);

        // Warmup
        for (int i = 0; i < config.num_warmup; ++i) {
            all_reduce(data, ReduceOp::SUM);
        }
        barrier();

        // Measurement
        std::vector<double> times;
        for (int i = 0; i < config.num_iterations; ++i) {
            barrier();
            double start = get_timestamp();
            all_reduce(data, ReduceOp::SUM);
            barrier();
            double end = get_timestamp();
            times.push_back(end - start);
        }

        // Calculate statistics
        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        for (double t : times) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        double avg_time = sum / times.size();

        // AllReduce transfers (world_size - 1) * msg_size bytes
        size_t total_bytes = msg_size * (world_size - 1);

        BenchmarkResult result;
        result.operation = "AllReduce";
        result.backend = backend_name;
        result.message_size_bytes = msg_size;
        result.avg_time_ms = avg_time;
        result.min_time_ms = min_time;
        result.max_time_ms = max_time;
        result.bandwidth_gbps = calculate_bandwidth_gbps(total_bytes, avg_time);
        result.latency_us = avg_time * 1000.0;

        results.push_back(result);
    }

    return results;
}

auto benchmark_broadcast(
    const BenchmarkConfig& config,
    const std::string& backend_name,
    int rank,
    int world_size
) -> std::vector<BenchmarkResult> {
    std::vector<BenchmarkResult> results;

    for (size_t msg_size : config.message_sizes) {
        size_t num_elements = msg_size / sizeof(float);
        if (num_elements == 0) num_elements = 1;

        Device device = Device::cpu();
        if (backend_name == "nccl") {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
            device = Device::cuda(0);
#else
            continue;
#endif
        }

        Tensor data = ones({static_cast<int64_t>(num_elements)}, DType::Float32, device);

        // Warmup
        for (int i = 0; i < config.num_warmup; ++i) {
            broadcast(data, 0);
        }
        barrier();

        // Measurement
        std::vector<double> times;
        for (int i = 0; i < config.num_iterations; ++i) {
            barrier();
            double start = get_timestamp();
            broadcast(data, 0);
            barrier();
            double end = get_timestamp();
            times.push_back(end - start);
        }

        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        for (double t : times) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        double avg_time = sum / times.size();

        // Broadcast transfers msg_size to (world_size - 1) ranks
        size_t total_bytes = msg_size * (world_size - 1);

        BenchmarkResult result;
        result.operation = "Broadcast";
        result.backend = backend_name;
        result.message_size_bytes = msg_size;
        result.avg_time_ms = avg_time;
        result.min_time_ms = min_time;
        result.max_time_ms = max_time;
        result.bandwidth_gbps = calculate_bandwidth_gbps(total_bytes, avg_time);
        result.latency_us = avg_time * 1000.0;

        results.push_back(result);
    }

    return results;
}

auto benchmark_all_gather(
    const BenchmarkConfig& config,
    const std::string& backend_name,
    int rank,
    int world_size
) -> std::vector<BenchmarkResult> {
    std::vector<BenchmarkResult> results;

    for (size_t msg_size : config.message_sizes) {
        size_t num_elements = msg_size / sizeof(float);
        if (num_elements == 0) num_elements = 1;

        Device device = Device::cpu();
        if (backend_name == "nccl") {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
            device = Device::cuda(0);
#else
            continue;
#endif
        }

        Tensor local = ones({static_cast<int64_t>(num_elements)}, DType::Float32, device);
        std::vector<Tensor> gathered(world_size);
        for (int i = 0; i < world_size; ++i) {
            gathered[i] = zeros({static_cast<int64_t>(num_elements)}, DType::Float32, device);
        }

        // Warmup
        for (int i = 0; i < config.num_warmup; ++i) {
            all_gather(local, gathered);
        }
        barrier();

        // Measurement
        std::vector<double> times;
        for (int i = 0; i < config.num_iterations; ++i) {
            barrier();
            double start = get_timestamp();
            all_gather(local, gathered);
            barrier();
            double end = get_timestamp();
            times.push_back(end - start);
        }

        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        for (double t : times) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        double avg_time = sum / times.size();

        // AllGather transfers (world_size - 1) * msg_size bytes
        size_t total_bytes = msg_size * (world_size - 1);

        BenchmarkResult result;
        result.operation = "AllGather";
        result.backend = backend_name;
        result.message_size_bytes = msg_size;
        result.avg_time_ms = avg_time;
        result.min_time_ms = min_time;
        result.max_time_ms = max_time;
        result.bandwidth_gbps = calculate_bandwidth_gbps(total_bytes, avg_time);
        result.latency_us = avg_time * 1000.0;

        results.push_back(result);
    }

    return results;
}

auto benchmark_reduce_scatter(
    const BenchmarkConfig& config,
    const std::string& backend_name,
    int rank,
    int world_size
) -> std::vector<BenchmarkResult> {
    std::vector<BenchmarkResult> results;

    for (size_t msg_size : config.message_sizes) {
        size_t chunk_elements = (msg_size / sizeof(float)) / world_size;
        if (chunk_elements == 0) chunk_elements = 1;

        Device device = Device::cpu();
        if (backend_name == "nccl") {
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
            device = Device::cuda(0);
#else
            continue;
#endif
        }

        std::vector<Tensor> input_chunks;
        for (int i = 0; i < world_size; ++i) {
            input_chunks.push_back(
                ones({static_cast<int64_t>(chunk_elements)}, DType::Float32, device)
            );
        }
        Tensor output = zeros({static_cast<int64_t>(chunk_elements)}, DType::Float32, device);

        // Warmup
        for (int i = 0; i < config.num_warmup; ++i) {
            reduce_scatter(input_chunks, output, ReduceOp::SUM);
        }
        barrier();

        // Measurement
        std::vector<double> times;
        for (int i = 0; i < config.num_iterations; ++i) {
            barrier();
            double start = get_timestamp();
            reduce_scatter(input_chunks, output, ReduceOp::SUM);
            barrier();
            double end = get_timestamp();
            times.push_back(end - start);
        }

        double sum = 0.0;
        double min_time = times[0];
        double max_time = times[0];
        for (double t : times) {
            sum += t;
            min_time = std::min(min_time, t);
            max_time = std::max(max_time, t);
        }
        double avg_time = sum / times.size();

        // ReduceScatter transfers (world_size - 1) * msg_size bytes
        size_t total_bytes = msg_size * (world_size - 1);

        BenchmarkResult result;
        result.operation = "ReduceScatter";
        result.backend = backend_name;
        result.message_size_bytes = msg_size;
        result.avg_time_ms = avg_time;
        result.min_time_ms = min_time;
        result.max_time_ms = max_time;
        result.bandwidth_gbps = calculate_bandwidth_gbps(total_bytes, avg_time);
        result.latency_us = avg_time * 1000.0;

        results.push_back(result);
    }

    return results;
}

// ============================================================================
// Main Benchmark Runner
// ============================================================================

int main(int argc, char** argv) {
    // Initialize Tenzor
    tenzor::initialize();

    // Get distributed environment
    const char* rank_env = std::getenv("RANK");
    const char* world_size_env = std::getenv("WORLD_SIZE");
    const char* master_addr = std::getenv("MASTER_ADDR");
    const char* master_port_env = std::getenv("MASTER_PORT");

    if (!rank_env || !world_size_env) {
        std::cerr << "ERROR: RANK and WORLD_SIZE must be set\n";
        std::cerr << "Usage:\n";
        std::cerr << "  export RANK=0 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500\n";
        std::cerr << "  ./comm_benchmarks &\n";
        std::cerr << "  export RANK=1 WORLD_SIZE=2 MASTER_ADDR=localhost MASTER_PORT=29500\n";
        std::cerr << "  ./comm_benchmarks\n";
        return 1;
    }

    int rank = std::atoi(rank_env);
    int world_size = std::atoi(world_size_env);
    std::string master = master_addr ? master_addr : "localhost";
    int master_port = master_port_env ? std::atoi(master_port_env) : 29500;

    if (rank == 0) {
        std::cout << "\n";
        std::cout << "================================================================\n";
        std::cout << "  Tenzor Distributed Communication Benchmarks\n";
        std::cout << "================================================================\n";
        std::cout << "Configuration:\n";
        std::cout << "  World Size: " << world_size << "\n";
        std::cout << "  Master: " << master << ":" << master_port << "\n";
        std::cout << "\n";
    }

    // Configure benchmark
    BenchmarkConfig config;
    config.message_sizes = {
        1024,              // 1 KB
        4096,              // 4 KB
        16384,             // 16 KB
        65536,             // 64 KB
        262144,            // 256 KB
        1048576,           // 1 MB
        4194304,           // 4 MB
        16777216,          // 16 MB
        67108864           // 64 MB
    };
    config.num_warmup = 5;
    config.num_iterations = 20;

    // Print header
    if (rank == 0) {
        std::cout << std::setw(15) << "Operation"
                  << std::setw(8) << "Backend"
                  << std::setw(12) << "Size (B)"
                  << std::setw(12) << "Avg (ms)"
                  << std::setw(12) << "Min (ms)"
                  << std::setw(12) << "Max (ms)"
                  << std::setw(15) << "BW (GB/s)"
                  << std::setw(15) << "Latency (us)"
                  << std::endl;
        std::cout << std::string(99, '-') << std::endl;
    }

    // Benchmark Gloo backend
    if (config.test_gloo) {
        init_process_group("gloo", rank, world_size, master, master_port);

        auto results = benchmark_all_reduce(config, "gloo", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        results = benchmark_broadcast(config, "gloo", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        results = benchmark_all_gather(config, "gloo", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        results = benchmark_reduce_scatter(config, "gloo", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        destroy_process_group();
    }

    // Benchmark NCCL backend
#if defined(TENZOR_USE_CUDA) || defined(TENZOR_USE_ROCM)
    if (config.test_nccl) {
        init_process_group("nccl", rank, world_size, master, master_port);

        auto results = benchmark_all_reduce(config, "nccl", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        results = benchmark_broadcast(config, "nccl", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        results = benchmark_all_gather(config, "nccl", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        results = benchmark_reduce_scatter(config, "nccl", rank, world_size);
        for (const auto& r : results) print_result(r, rank);

        destroy_process_group();
    }
#endif

    if (rank == 0) {
        std::cout << "================================================================\n";
        std::cout << "Benchmarks completed successfully!\n";
        std::cout << "\nTarget bandwidth (from design doc): 100-300 GB/s\n";
        std::cout << "================================================================\n\n";
    }

    return 0;
}
