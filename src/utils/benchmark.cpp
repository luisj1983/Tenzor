/**
 * @file benchmark.cpp
 * @brief Implementation of benchmark utilities
 */

#include "tenzor/utils/benchmark.hpp"
#include <algorithm>
#include <numeric>
#include <cmath>
#include <iostream>
#include <iomanip>
#include <sstream>
#include <fstream>

namespace tenzor {
namespace benchmark {

auto Benchmark::run(const std::function<void()>& fn) -> BenchmarkResult {
    return run([](){}, fn, [](){});
}

auto Benchmark::run(
    const std::function<void()>& setup,
    const std::function<void()>& fn,
    const std::function<void()>& teardown
) -> BenchmarkResult {
    std::vector<double> times;
    times.reserve(num_runs_);

    Timer timer;

    // Warmup
    for (size_t i = 0; i < num_warmup_; ++i) {
        setup();
        fn();
        teardown();
    }

    // Benchmark runs
    for (size_t i = 0; i < num_runs_; ++i) {
        setup();

        timer.start();
        fn();
        double elapsed = timer.stop();

        teardown();

        times.push_back(elapsed);
    }

    // Compute statistics
    auto stats = compute_stats(times);

    // Create result
    BenchmarkResult result;
    result.name = name_;
    result.stats = stats;
    result.num_flops = num_flops_;
    result.num_bytes = num_bytes_;

    if (num_flops_ > 0) {
        result.tflops = stats.tflops(num_flops_);
    }

    if (num_bytes_ > 0) {
        result.bandwidth_gbs = stats.bandwidth_gbs(num_bytes_);
    }

    return result;
}

auto Benchmark::compute_stats(const std::vector<double>& times) -> BenchmarkStats {
    BenchmarkStats stats;
    stats.num_runs = times.size();

    if (times.empty()) {
        return stats;
    }

    // Sort for percentiles
    std::vector<double> sorted = times;
    std::sort(sorted.begin(), sorted.end());

    // Min/Max
    stats.min = sorted.front();
    stats.max = sorted.back();

    // Mean
    double sum = std::accumulate(sorted.begin(), sorted.end(), 0.0);
    stats.mean = sum / static_cast<double>(sorted.size());

    // Standard deviation
    double sq_sum = 0.0;
    for (double t : sorted) {
        double diff = t - stats.mean;
        sq_sum += diff * diff;
    }
    stats.std_dev = std::sqrt(sq_sum / static_cast<double>(sorted.size()));

    // Median
    if (sorted.size() % 2 == 0) {
        stats.median = (sorted[sorted.size() / 2 - 1] + sorted[sorted.size() / 2]) / 2.0;
    } else {
        stats.median = sorted[sorted.size() / 2];
    }

    // Percentiles (linear interpolation between closest ranks so small N does
    // not bias tail latencies low or collapse p95 == p99).
    auto percentile = [&](double p) -> double {
        double rank = p * static_cast<double>(sorted.size() - 1);
        size_t lo = static_cast<size_t>(std::floor(rank));
        size_t hi = std::min(lo + 1, sorted.size() - 1);
        double frac = rank - static_cast<double>(lo);
        return sorted[lo] * (1.0 - frac) + sorted[hi] * frac;
    };

    stats.p95 = percentile(0.95);
    stats.p99 = percentile(0.99);

    return stats;
}

auto BenchmarkResult::print() const -> void {
    std::cout << "\n=== Benchmark: " << name << " ===\n";
    std::cout << std::fixed << std::setprecision(6);

    std::cout << "  Runs:        " << stats.num_runs << "\n";
    std::cout << "  Mean:        " << (stats.mean * 1000.0) << " ms\n";
    std::cout << "  Std Dev:     " << (stats.std_dev * 1000.0) << " ms\n";
    std::cout << "  Min:         " << (stats.min * 1000.0) << " ms\n";
    std::cout << "  Max:         " << (stats.max * 1000.0) << " ms\n";
    std::cout << "  Median:      " << (stats.median * 1000.0) << " ms\n";
    std::cout << "  95th %ile:   " << (stats.p95 * 1000.0) << " ms\n";
    std::cout << "  99th %ile:   " << (stats.p99 * 1000.0) << " ms\n";

    if (num_flops > 0) {
        std::cout << "  TFLOPS:      " << tflops << "\n";
        std::cout << "  GFLOPS:      " << (tflops * 1000.0) << "\n";
    }

    if (num_bytes > 0) {
        std::cout << "  Bandwidth:   " << bandwidth_gbs << " GB/s\n";
    }

    std::cout << std::endl;
}

auto BenchmarkResult::to_json() const -> std::string {
    std::ostringstream oss;
    oss << std::fixed << std::setprecision(6);

    oss << "{\n";
    oss << "  \"name\": \"" << name << "\",\n";
    oss << "  \"stats\": {\n";
    oss << "    \"num_runs\": " << stats.num_runs << ",\n";
    oss << "    \"mean_ms\": " << (stats.mean * 1000.0) << ",\n";
    oss << "    \"std_dev_ms\": " << (stats.std_dev * 1000.0) << ",\n";
    oss << "    \"min_ms\": " << (stats.min * 1000.0) << ",\n";
    oss << "    \"max_ms\": " << (stats.max * 1000.0) << ",\n";
    oss << "    \"median_ms\": " << (stats.median * 1000.0) << ",\n";
    oss << "    \"p95_ms\": " << (stats.p95 * 1000.0) << ",\n";
    oss << "    \"p99_ms\": " << (stats.p99 * 1000.0) << "\n";
    oss << "  }";

    if (num_flops > 0) {
        oss << ",\n  \"performance\": {\n";
        oss << "    \"tflops\": " << tflops << ",\n";
        oss << "    \"gflops\": " << (tflops * 1000.0) << "\n";
        oss << "  }";
    }

    if (num_bytes > 0) {
        oss << ",\n  \"bandwidth\": {\n";
        oss << "    \"gbs\": " << bandwidth_gbs << "\n";
        oss << "  }";
    }

    oss << "\n}";
    return oss.str();
}

auto BenchmarkSuite::run_all() -> std::vector<BenchmarkResult> {
    std::vector<BenchmarkResult> results;
    results.reserve(benchmarks_.size());

    std::cout << "\n========================================\n";
    std::cout << "  Benchmark Suite: " << name_ << "\n";
    std::cout << "========================================\n";

    for (auto& benchmark : benchmarks_) {
        auto result = benchmark.run([](){});
        result.print();
        results.push_back(result);
    }

    print_summary(results);

    return results;
}

auto BenchmarkSuite::print_summary(const std::vector<BenchmarkResult>& results) const -> void {
    std::cout << "\n========================================\n";
    std::cout << "  Summary\n";
    std::cout << "========================================\n";

    std::cout << std::left << std::setw(30) << "Benchmark"
              << std::right << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "Min (ms)"
              << std::setw(12) << "GFLOPS"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(30) << result.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << (result.stats.mean * 1000.0)
                  << std::setw(12) << (result.stats.min * 1000.0);

        if (result.num_flops > 0) {
            std::cout << std::setw(12) << (result.tflops * 1000.0);
        } else {
            std::cout << std::setw(12) << "-";
        }

        std::cout << "\n";
    }

    std::cout << std::endl;
}

auto BenchmarkSuite::export_json(const std::vector<BenchmarkResult>& results) const -> std::string {
    std::ostringstream oss;
    oss << "{\n";
    oss << "  \"suite_name\": \"" << name_ << "\",\n";
    oss << "  \"benchmarks\": [\n";

    for (size_t i = 0; i < results.size(); ++i) {
        auto json = results[i].to_json();
        // Indent each line by 4 spaces
        std::string indented;
        std::istringstream iss(json);
        std::string line;
        while (std::getline(iss, line)) {
            indented += "    " + line + "\n";
        }
        // Remove trailing newline and add to output
        if (!indented.empty() && indented.back() == '\n') {
            indented.pop_back();
        }
        oss << indented;

        if (i < results.size() - 1) {
            oss << ",";
        }
        oss << "\n";
    }

    oss << "  ]\n";
    oss << "}\n";
    return oss.str();
}

auto BenchmarkSuite::save_json(const std::vector<BenchmarkResult>& results, const std::string& filepath) const -> void {
    std::ofstream file(filepath);
    if (!file.is_open()) {
        throw std::runtime_error("Failed to open file: " + filepath);
    }
    file << export_json(results);
    file.close();
    std::cout << "Benchmark results saved to: " << filepath << std::endl;
}

} // namespace benchmark
} // namespace tenzor
