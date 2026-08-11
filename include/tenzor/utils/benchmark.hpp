/**
 * @file benchmark.hpp
 * @brief Benchmark utilities for performance measurement
 *
 * Provides high-resolution timing, TFLOPS calculation, memory bandwidth
 * measurement, and statistical analysis for performance benchmarking.
 */

#pragma once

#include "tenzor/core/device.hpp"
#include <chrono>
#include <string>
#include <vector>
#include <functional>
#include <cstddef>

namespace tenzor {
namespace benchmark {

/**
 * @brief High-resolution timer for performance measurement
 */
class Timer {
public:
    using Clock = std::chrono::high_resolution_clock;
    using TimePoint = Clock::time_point;
    using Duration = std::chrono::duration<double>;

    /**
     * @brief Start timing
     */
    auto start() -> void {
        start_ = Clock::now();
    }

    /**
     * @brief Stop timing and return elapsed time in seconds
     */
    auto stop() -> double {
        end_ = Clock::now();
        Duration elapsed = end_ - start_;
        return elapsed.count();
    }

    /**
     * @brief Get elapsed time without stopping timer
     */
    auto elapsed() const -> double {
        auto now = Clock::now();
        Duration elapsed = now - start_;
        return elapsed.count();
    }

private:
    TimePoint start_;
    TimePoint end_;
};

/**
 * @brief Benchmark statistics
 */
struct BenchmarkStats {
    double mean;         // Mean execution time (seconds)
    double std_dev;      // Standard deviation (seconds)
    double min;          // Minimum time (seconds)
    double max;          // Maximum time (seconds)
    double median;       // Median time (seconds)
    double p95;          // 95th percentile (seconds)
    double p99;          // 99th percentile (seconds)
    size_t num_runs;     // Number of benchmark runs

    /**
     * @brief Get throughput in operations per second
     */
    auto ops_per_sec(size_t num_ops) const -> double {
        return static_cast<double>(num_ops) / mean;
    }

    /**
     * @brief Get TFLOPS (teraflops)
     */
    auto tflops(size_t num_flops) const -> double {
        return static_cast<double>(num_flops) / (mean * 1e12);
    }

    /**
     * @brief Get GFLOPS (gigaflops)
     */
    auto gflops(size_t num_flops) const -> double {
        return static_cast<double>(num_flops) / (mean * 1e9);
    }

    /**
     * @brief Get memory bandwidth in GB/s
     */
    auto bandwidth_gbs(size_t num_bytes) const -> double {
        return static_cast<double>(num_bytes) / (mean * 1e9);
    }
};

/**
 * @brief Benchmark result
 */
struct BenchmarkResult {
    std::string name;               // Benchmark name
    BenchmarkStats stats;           // Timing statistics
    size_t num_flops{0};           // Number of floating-point operations
    size_t num_bytes{0};           // Number of bytes transferred
    double tflops{0.0};            // TFLOPS
    double bandwidth_gbs{0.0};     // Memory bandwidth (GB/s)

    /**
     * @brief Print formatted results
     */
    auto print() const -> void;

    /**
     * @brief Export results as JSON string
     */
    auto to_json() const -> std::string;
};

/**
 * @brief Benchmark runner
 */
class Benchmark {
public:
    /**
     * @brief Construct benchmark
     *
     * @param name Benchmark name
     * @param num_warmup Number of warmup iterations
     * @param num_runs Number of measured iterations
     */
    Benchmark(
        const std::string& name,
        size_t num_warmup = 5,
        size_t num_runs = 100
    ) : name_(name), num_warmup_(num_warmup), num_runs_(num_runs) {}

    /**
     * @brief Construct a benchmark bound to a workload.
     *
     * The stored workload is what BenchmarkSuite::run_all() executes; without
     * it run_all() can only time loop overhead.
     *
     * @param name Benchmark name
     * @param workload Function under test
     * @param num_warmup Number of warmup iterations
     * @param num_runs Number of measured iterations
     */
    Benchmark(
        const std::string& name,
        std::function<void()> workload,
        size_t num_warmup = 5,
        size_t num_runs = 100
    ) : name_(name), num_warmup_(num_warmup), num_runs_(num_runs),
        workload_(std::move(workload)) {}

    /**
     * @brief Associate the workload to be benchmarked by run_stored()/run_all().
     */
    auto set_workload(std::function<void()> workload) -> Benchmark& {
        workload_ = std::move(workload);
        return *this;
    }

    /**
     * @brief Whether a workload has been associated with this benchmark.
     */
    auto has_workload() const -> bool { return static_cast<bool>(workload_); }

    /**
     * @brief Run the stored workload and collect statistics.
     *
     * @return Benchmark result with statistics
     */
    auto run_stored() -> BenchmarkResult;

    /**
     * @brief Run benchmark and collect statistics
     *
     * @param fn Function to benchmark
     * @return Benchmark result with statistics
     */
    auto run(const std::function<void()>& fn) -> BenchmarkResult;

    /**
     * @brief Run benchmark with custom setup and teardown
     *
     * @param setup Function to run before each iteration
     * @param fn Function to benchmark
     * @param teardown Function to run after each iteration
     * @return Benchmark result
     */
    auto run(
        const std::function<void()>& setup,
        const std::function<void()>& fn,
        const std::function<void()>& teardown
    ) -> BenchmarkResult;

    /**
     * @brief Set the device the timed workload runs on.
     *
     * GPU backends (CUDA/ROCm/Vulkan/OneAPI) queue work asynchronously —
     * the timed function returning does not mean the work is done. Without
     * this, run()/run_stored() measure kernel-launch overhead, not compute
     * time, which previously produced results like "30,293 TFLOPS" on a
     * laptop GPU (~40-80 TFLOPS peak). Defaults to CPU, where synchronize()
     * is a no-op, so existing CPU-only call sites are unaffected.
     */
    auto set_device(Device device) -> Benchmark& {
        device_ = device;
        return *this;
    }

    /**
     * @brief Set number of floating-point operations
     */
    auto set_flops(size_t num_flops) -> Benchmark& {
        num_flops_ = num_flops;
        return *this;
    }

    /**
     * @brief Set number of bytes transferred
     */
    auto set_bytes(size_t num_bytes) -> Benchmark& {
        num_bytes_ = num_bytes;
        return *this;
    }

    /**
     * @brief Compute statistics (mean/min/max/std_dev/median/p95/p99) from a
     *        vector of timing samples (seconds).
     *
     * Pure, stateless utility — exposed so the percentile/interpolation math
     * can be verified directly against a deterministic synthetic timing vector
     * rather than relying on noisy wall-clock measurements.
     */
    static auto compute_stats(const std::vector<double>& times) -> BenchmarkStats;

private:
    std::string name_;
    size_t num_warmup_;
    size_t num_runs_;
    size_t num_flops_{0};
    size_t num_bytes_{0};
    std::function<void()> workload_{};
    Device device_{Device::cpu()};
};

/**
 * @brief Benchmark suite for running multiple benchmarks
 */
class BenchmarkSuite {
public:
    /**
     * @brief Construct benchmark suite
     */
    BenchmarkSuite(const std::string& name) : name_(name) {}

    /**
     * @brief Add benchmark to suite
     */
    auto add(Benchmark benchmark) -> BenchmarkSuite& {
        benchmarks_.push_back(benchmark);
        return *this;
    }

    /**
     * @brief Run all benchmarks in suite
     */
    auto run_all() -> std::vector<BenchmarkResult>;

    /**
     * @brief Print summary of all results
     */
    auto print_summary(const std::vector<BenchmarkResult>& results) const -> void;

    /**
     * @brief Export all results as JSON string
     */
    auto export_json(const std::vector<BenchmarkResult>& results) const -> std::string;

    /**
     * @brief Save results to JSON file
     */
    auto save_json(const std::vector<BenchmarkResult>& results, const std::string& filepath) const -> void;

private:
    std::string name_;
    std::vector<Benchmark> benchmarks_;
};

/**
 * @brief TFLOPS calculation utilities
 */
namespace flops {

/**
 * @brief Calculate FLOPS for matrix multiplication
 *
 * @param M Rows of A
 * @param N Columns of B
 * @param K Columns of A (rows of B)
 * @return Number of floating-point operations
 */
inline auto matmul(size_t M, size_t N, size_t K) -> size_t {
    return 2 * M * N * K;  // 2 ops per element (multiply + add)
}

/**
 * @brief Calculate FLOPS for 2D convolution
 *
 * @param batch Batch size
 * @param out_h Output height
 * @param out_w Output width
 * @param in_channels Input channels
 * @param out_channels Output channels
 * @param kernel_h Kernel height
 * @param kernel_w Kernel width
 * @return Number of floating-point operations
 */
inline auto conv2d(
    size_t batch,
    size_t out_h,
    size_t out_w,
    size_t in_channels,
    size_t out_channels,
    size_t kernel_h,
    size_t kernel_w
) -> size_t {
    return 2 * batch * out_h * out_w * in_channels * out_channels * kernel_h * kernel_w;
}

/**
 * @brief Calculate FLOPS for element-wise operation
 *
 * @param num_elements Number of elements
 * @param ops_per_element Operations per element
 * @return Number of floating-point operations
 */
inline auto elementwise(size_t num_elements, size_t ops_per_element = 1) -> size_t {
    return num_elements * ops_per_element;
}

} // namespace flops

/**
 * @brief Memory transfer calculation utilities
 */
namespace memory {

/**
 * @brief Calculate bytes transferred for matrix multiplication
 *
 * @param M Rows of A
 * @param N Columns of B
 * @param K Columns of A (rows of B)
 * @param element_size Size of each element in bytes
 * @return Number of bytes transferred
 */
inline auto matmul(size_t M, size_t N, size_t K, size_t element_size = 4) -> size_t {
    // Read A (M*K) + B (K*N) + Write C (M*N)
    return element_size * (M * K + K * N + M * N);
}

/**
 * @brief Calculate bytes for element-wise operation
 *
 * @param num_elements Number of elements
 * @param num_inputs Number of input arrays
 * @param num_outputs Number of output arrays
 * @param element_size Size of each element in bytes
 * @return Number of bytes transferred
 */
inline auto elementwise(
    size_t num_elements,
    size_t num_inputs = 2,
    size_t num_outputs = 1,
    size_t element_size = 4
) -> size_t {
    return element_size * num_elements * (num_inputs + num_outputs);
}

} // namespace memory

/**
 * @brief Macro for convenient benchmark definition
 */
#define TENZOR_BENCHMARK(name, warmup, runs) \
    tenzor::benchmark::Benchmark(name, warmup, runs)

} // namespace benchmark
} // namespace tenzor
