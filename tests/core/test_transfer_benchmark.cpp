/**
 * @file test_transfer_benchmark.cpp
 * @brief Bandwidth and performance benchmarks for Transfer Engine (Phase 1 - ZeRO Offload)
 *
 * Benchmarks:
 * - Transfer bandwidth for various sizes (1KB to 1GB)
 * - Measure bandwidth in GB/s
 * - Test async overlap benefits
 * - Report results to stdout
 */

#include <gtest/gtest.h>
#include <tenzor/core/transfer_engine.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>
#include <chrono>
#include <vector>
#include <iomanip>
#include <iostream>

using namespace tenzor;
using namespace tenzor::core;

/**
 * Test Fixture for Transfer Benchmarks
 */
class TransferBenchmarkTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();  // Load CUDA backend
    }

    void SetUp() override {
        // Configure engine for optimal performance
        config.num_streams = 8;
        config.queue_capacity = 128;
        config.use_pinned_memory = true;
        config.pinned_pool_size = 512 * 1024 * 1024;  // 512 MB

        cuda_available = checkCudaAvailable();
    }

    void TearDown() override {
        // Report summary
        if (cuda_available && !results.empty()) {
            printSummary();
        }
    }

    TransferEngine::Config config;
    bool cuda_available = false;

    struct BenchmarkResult {
        std::string name;
        size_t size_bytes;
        double time_ms;
        double bandwidth_gbps;
    };

    std::vector<BenchmarkResult> results;

    // Helper to check CUDA availability
    bool checkCudaAvailable() {
        try {
            Device cuda_device = Device::cuda(0);
            Tensor test = zeros({10}, DType::Float32, cuda_device);
            return true;
        } catch (...) {
            return false;
        }
    }

    // Helper to create tensor
    Tensor createTensor(size_t bytes, Device device) {
        size_t numel = bytes / sizeof(float);
        Tensor t(std::vector<int64_t>{static_cast<int64_t>(numel)}, DType::Float32, device);

        // Fill with data to ensure memory is allocated
        if (device.type == Device::Type::CPU) {
            auto* data = t.data<float>();
            for (size_t i = 0; i < numel; ++i) {
                data[i] = static_cast<float>(i % 1000);
            }
        }

        return t;
    }

    // Measure transfer time
    template<typename Func>
    double measureTime(Func&& func) {
        // Warm-up
        func();

        // Actual measurement (average of 3 runs)
        double total_time = 0.0;
        const int num_runs = 3;

        for (int i = 0; i < num_runs; ++i) {
            auto start = std::chrono::high_resolution_clock::now();
            func();
            auto end = std::chrono::high_resolution_clock::now();

            std::chrono::duration<double, std::milli> duration = end - start;
            total_time += duration.count();
        }

        return total_time / num_runs;
    }

    // Record result
    void recordResult(const std::string& name, size_t size_bytes, double time_ms) {
        double bandwidth_gbps = (size_bytes / (time_ms / 1000.0)) / 1e9;

        results.push_back({name, size_bytes, time_ms, bandwidth_gbps});

        std::cout << std::fixed << std::setprecision(2);
        std::cout << name << ": "
                  << (size_bytes / 1024.0 / 1024.0) << " MB in "
                  << time_ms << " ms ("
                  << bandwidth_gbps << " GB/s)" << std::endl;
    }

    // Print summary
    void printSummary() {
        std::cout << "\n========================================\n";
        std::cout << "Transfer Benchmark Summary\n";
        std::cout << "========================================\n";

        double total_bandwidth = 0.0;

        for (const auto& result : results) {
            std::cout << std::setw(30) << std::left << result.name << ": "
                      << std::setw(10) << std::right << std::fixed << std::setprecision(2)
                      << result.bandwidth_gbps << " GB/s" << std::endl;

            total_bandwidth += result.bandwidth_gbps;
        }

        std::cout << "----------------------------------------\n";
        std::cout << std::setw(30) << std::left << "Average Bandwidth" << ": "
                  << std::setw(10) << std::right
                  << (total_bandwidth / results.size()) << " GB/s" << std::endl;
        std::cout << "========================================\n";
    }
};

// =============================================================================
// CPU to GPU Bandwidth Benchmarks
// =============================================================================

TEST_F(TransferBenchmarkTest, Bandwidth_CPUToGPU_1MB) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 1 * 1024 * 1024;  // 1 MB

    Tensor cpu_tensor = createTensor(size, Device::cpu());

    double time_ms = measureTime([&]() {
        Tensor gpu_tensor = engine.cpu_to_gpu(cpu_tensor, Device::cuda(0));
    });

    recordResult("CPU->GPU 1MB", size, time_ms);
}

TEST_F(TransferBenchmarkTest, Bandwidth_CPUToGPU_10MB) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 10 * 1024 * 1024;  // 10 MB

    Tensor cpu_tensor = createTensor(size, Device::cpu());

    double time_ms = measureTime([&]() {
        Tensor gpu_tensor = engine.cpu_to_gpu(cpu_tensor, Device::cuda(0));
    });

    recordResult("CPU->GPU 10MB", size, time_ms);
}

TEST_F(TransferBenchmarkTest, Bandwidth_CPUToGPU_100MB) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 100 * 1024 * 1024;  // 100 MB

    Tensor cpu_tensor = createTensor(size, Device::cpu());

    double time_ms = measureTime([&]() {
        Tensor gpu_tensor = engine.cpu_to_gpu(cpu_tensor, Device::cuda(0));
    });

    recordResult("CPU->GPU 100MB", size, time_ms);
}

// =============================================================================
// GPU to CPU Bandwidth Benchmarks
// =============================================================================

TEST_F(TransferBenchmarkTest, Bandwidth_GPUToCPU_1MB) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 1 * 1024 * 1024;  // 1 MB

    Tensor gpu_tensor = createTensor(size, Device::cuda(0));

    double time_ms = measureTime([&]() {
        Tensor cpu_tensor = engine.gpu_to_cpu(gpu_tensor);
    });

    recordResult("GPU->CPU 1MB", size, time_ms);
}

TEST_F(TransferBenchmarkTest, Bandwidth_GPUToCPU_10MB) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 10 * 1024 * 1024;  // 10 MB

    Tensor gpu_tensor = createTensor(size, Device::cuda(0));

    double time_ms = measureTime([&]() {
        Tensor cpu_tensor = engine.gpu_to_cpu(gpu_tensor);
    });

    recordResult("GPU->CPU 10MB", size, time_ms);
}

TEST_F(TransferBenchmarkTest, Bandwidth_GPUToCPU_100MB) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 100 * 1024 * 1024;  // 100 MB

    Tensor gpu_tensor = createTensor(size, Device::cuda(0));

    double time_ms = measureTime([&]() {
        Tensor cpu_tensor = engine.gpu_to_cpu(gpu_tensor);
    });

    recordResult("GPU->CPU 100MB", size, time_ms);
}

// =============================================================================
// Async Overlap Benchmarks
// =============================================================================

TEST_F(TransferBenchmarkTest, AsyncOverlap_SerialVsParallel) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 10 * 1024 * 1024;  // 10 MB

    const int num_transfers = 8;
    std::vector<Tensor> cpu_tensors;
    for (int i = 0; i < num_transfers; ++i) {
        cpu_tensors.push_back(createTensor(size, Device::cpu()));
    }

    // Serial transfers
    double serial_time_ms = measureTime([&]() {
        for (auto& cpu_t : cpu_tensors) {
            Tensor gpu_t = engine.cpu_to_gpu(cpu_t, Device::cuda(0));
        }
    });
    recordResult("Serial 8x10MB", num_transfers * size, serial_time_ms);

    // Parallel async transfers
    double parallel_time_ms = measureTime([&]() {
        std::vector<TransferHandle> handles;
        for (auto& cpu_t : cpu_tensors) {
            handles.push_back(engine.cpu_to_gpu_async(cpu_t, Device::cuda(0)));
        }
        for (auto& handle : handles) {
            handle.wait();
        }
    });
    recordResult("Parallel 8x10MB", num_transfers * size, parallel_time_ms);

    double speedup = serial_time_ms / parallel_time_ms;
    std::cout << "Async overlap speedup: " << std::fixed << std::setprecision(2)
              << speedup << "x" << std::endl;

    EXPECT_GT(speedup, 1.0);  // Parallel should be faster
}

TEST_F(TransferBenchmarkTest, AsyncOverlap_BidirectionalTransfers) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 20 * 1024 * 1024;  // 20 MB

    const int num_pairs = 4;
    std::vector<Tensor> cpu_tensors;
    std::vector<Tensor> gpu_tensors;

    for (int i = 0; i < num_pairs; ++i) {
        cpu_tensors.push_back(createTensor(size, Device::cpu()));
        gpu_tensors.push_back(createTensor(size, Device::cuda(0)));
    }

    // Bidirectional async transfers
    double time_ms = measureTime([&]() {
        std::vector<TransferHandle> cpu_to_gpu_handles;
        std::vector<TransferHandle> gpu_to_cpu_handles;

        // Launch all transfers
        for (int i = 0; i < num_pairs; ++i) {
            cpu_to_gpu_handles.push_back(engine.cpu_to_gpu_async(cpu_tensors[i], Device::cuda(0)));
            gpu_to_cpu_handles.push_back(engine.gpu_to_cpu_async(gpu_tensors[i]));
        }

        // Wait for all
        for (auto& handle : cpu_to_gpu_handles) handle.wait();
        for (auto& handle : gpu_to_cpu_handles) handle.wait();
    });

    recordResult("Bidirectional 4x20MB", num_pairs * size * 2, time_ms);
}

// =============================================================================
// Sustained Throughput Benchmarks
// =============================================================================

TEST_F(TransferBenchmarkTest, SustainedThroughput_100Transfers) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 1 * 1024 * 1024;  // 1 MB

    const int num_transfers = 100;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < num_transfers; ++i) {
        Tensor cpu_t = createTensor(size, Device::cpu());
        Tensor gpu_t = engine.cpu_to_gpu(cpu_t, Device::cuda(0));
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    size_t total_bytes = num_transfers * size;
    recordResult("Sustained 100x1MB", total_bytes, duration.count());
}

// =============================================================================
// Latency Benchmarks
// =============================================================================

TEST_F(TransferBenchmarkTest, Latency_SmallTransfer) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 4;  // 4 bytes

    Tensor cpu_tensor = createTensor(size, Device::cpu());

    // Measure minimum latency
    auto start = std::chrono::high_resolution_clock::now();
    Tensor gpu_tensor = engine.cpu_to_gpu(cpu_tensor, Device::cuda(0));
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> duration = end - start;

    std::cout << "Minimum transfer latency: " << duration.count() << " μs" << std::endl;
}

TEST_F(TransferBenchmarkTest, Latency_AsyncOverhead) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);
    size_t size = 1024;  // 1 KB

    Tensor cpu_tensor = createTensor(size, Device::cpu());

    // Measure async overhead
    auto start = std::chrono::high_resolution_clock::now();
    TransferHandle handle = engine.cpu_to_gpu_async(cpu_tensor, Device::cuda(0));
    auto launch_end = std::chrono::high_resolution_clock::now();
    handle.wait();
    auto wait_end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double, std::micro> launch_time = launch_end - start;
    std::chrono::duration<double, std::micro> wait_time = wait_end - launch_end;

    std::cout << "Async launch overhead: " << launch_time.count() << " μs" << std::endl;
    std::cout << "Async wait time: " << wait_time.count() << " μs" << std::endl;
}

// =============================================================================
// PCIe Utilization Benchmarks
// =============================================================================

TEST_F(TransferBenchmarkTest, PCIeUtilization_MaxBandwidth) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    TransferEngine engine(config);

    // Test with various sizes to find peak bandwidth
    std::vector<size_t> sizes = {
        1 * 1024 * 1024,    // 1 MB
        10 * 1024 * 1024,   // 10 MB
        50 * 1024 * 1024,   // 50 MB
        100 * 1024 * 1024,  // 100 MB
        200 * 1024 * 1024   // 200 MB
    };

    double max_bandwidth = 0.0;
    size_t optimal_size = 0;

    for (size_t size : sizes) {
        Tensor cpu_tensor = createTensor(size, Device::cpu());

        double time_ms = measureTime([&]() {
            Tensor gpu_tensor = engine.cpu_to_gpu(cpu_tensor, Device::cuda(0));
        });

        double bandwidth_gbps = (size / (time_ms / 1000.0)) / 1e9;

        if (bandwidth_gbps > max_bandwidth) {
            max_bandwidth = bandwidth_gbps;
            optimal_size = size;
        }

        std::cout << "Size: " << (size / 1024 / 1024) << " MB, "
                  << "Bandwidth: " << std::fixed << std::setprecision(2)
                  << bandwidth_gbps << " GB/s" << std::endl;
    }

    std::cout << "\nPeak bandwidth: " << max_bandwidth << " GB/s at "
              << (optimal_size / 1024 / 1024) << " MB transfer size" << std::endl;

    // Typical PCIe 3.0 x16: ~12 GB/s
    // Typical PCIe 4.0 x16: ~25 GB/s
    EXPECT_GT(max_bandwidth, 5.0);  // At least 5 GB/s expected
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "Transfer Engine Bandwidth Benchmarks\n";
    std::cout << "========================================\n";
    std::cout << "\n";

    int result = RUN_ALL_TESTS();

    return result;
}
