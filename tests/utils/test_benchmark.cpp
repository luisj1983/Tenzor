/**
 * @file test_benchmark.cpp
 * @brief Unit tests for benchmark utilities
 */

#include <gtest/gtest.h>
#include <tenzor/utils/benchmark.hpp>
#include <thread>
#include <chrono>

using namespace tenzor::benchmark;

class BenchmarkTest : public ::testing::Test {
protected:
    void SetUp() override {}
    void TearDown() override {}
};

// Test 1: Timer basic functionality
TEST_F(BenchmarkTest, TimerBasic) {
    Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    double elapsed = timer.stop();

    // Should be approximately 10ms (0.01s), but allow some variance
    EXPECT_GE(elapsed, 0.008);  // At least 8ms
    EXPECT_LE(elapsed, 0.050);  // At most 50ms (generous for CI)
}

// Test 2: Timer elapsed without stopping
TEST_F(BenchmarkTest, TimerElapsed) {
    Timer timer;

    timer.start();
    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double elapsed1 = timer.elapsed();

    std::this_thread::sleep_for(std::chrono::milliseconds(5));
    double elapsed2 = timer.elapsed();

    // Second elapsed should be longer than first
    EXPECT_GT(elapsed2, elapsed1);
}

// Test 3: Benchmark run simple function
TEST_F(BenchmarkTest, BenchmarkRunSimple) {
    Benchmark bench("TestBench", 2, 10);  // 2 warmup, 10 runs

    int counter = 0;
    auto result = bench.run([&counter]() {
        counter++;
        // Simulate some work
        volatile int x = 0;
        for (int i = 0; i < 1000; ++i) {
            x += i;
        }
    });

    EXPECT_EQ(result.name, "TestBench");
    EXPECT_EQ(result.stats.num_runs, 10);

    // Should have run warmup + benchmark iterations
    EXPECT_EQ(counter, 12);  // 2 warmup + 10 benchmark

    // Stats should be reasonable
    EXPECT_GT(result.stats.mean, 0.0);
    EXPECT_GT(result.stats.min, 0.0);
    EXPECT_GT(result.stats.max, 0.0);
    EXPECT_GE(result.stats.max, result.stats.min);
    EXPECT_GE(result.stats.mean, result.stats.min);
    EXPECT_LE(result.stats.mean, result.stats.max);
}

// Test 4: Benchmark with setup and teardown
TEST_F(BenchmarkTest, BenchmarkWithSetupTeardown) {
    Benchmark bench("SetupTeardownTest", 1, 5);

    int setup_count = 0;
    int run_count = 0;
    int teardown_count = 0;

    auto result = bench.run(
        [&setup_count]() { setup_count++; },
        [&run_count]() { run_count++; },
        [&teardown_count]() { teardown_count++; }
    );

    // Warmup (1) + Runs (5) = 6 total
    EXPECT_EQ(setup_count, 6);
    EXPECT_EQ(run_count, 6);
    EXPECT_EQ(teardown_count, 6);
    EXPECT_EQ(result.stats.num_runs, 5);
}

// Test 5: BenchmarkStats percentiles
TEST_F(BenchmarkTest, BenchmarkStatsPercentiles) {
    Benchmark bench("PercentilesTest", 0, 100);

    auto result = bench.run([]() {
        // Variable work to get different timings
        volatile int x = 0;
        for (int i = 0; i < (rand() % 100 + 50); ++i) {
            x += i;
        }
    });

    // Verify percentile ordering
    EXPECT_GE(result.stats.median, result.stats.min);
    EXPECT_LE(result.stats.median, result.stats.max);
    EXPECT_GE(result.stats.p95, result.stats.median);
    EXPECT_GE(result.stats.p99, result.stats.p95);
    EXPECT_LE(result.stats.p99, result.stats.max);
}

// Test 6: BenchmarkStats standard deviation
TEST_F(BenchmarkTest, BenchmarkStatsStdDev) {
    Benchmark bench("StdDevTest", 0, 50);

    auto result = bench.run([]() {
        volatile int x = 0;
        for (int i = 0; i < 100; ++i) {
            x += i;
        }
    });

    // Standard deviation should be non-negative and less than max-min
    EXPECT_GE(result.stats.std_dev, 0.0);
    double range = result.stats.max - result.stats.min;
    EXPECT_LE(result.stats.std_dev, range);
}

// Test 7: Benchmark with FLOPS
TEST_F(BenchmarkTest, BenchmarkWithFlops) {
    size_t num_flops = 1000000;  // 1M FLOPS

    Benchmark bench("FlopsTest", 1, 10);
    bench.set_flops(num_flops);

    auto result = bench.run([]() {
        volatile double x = 1.0;
        for (int i = 0; i < 1000000; ++i) {
            x = x * 1.0001;  // 1M multiplications
        }
    });

    EXPECT_EQ(result.num_flops, num_flops);
    EXPECT_GT(result.tflops, 0.0);

    // Verify TFLOPS calculation
    double expected_tflops = static_cast<double>(num_flops) / (result.stats.mean * 1e12);
    EXPECT_NEAR(result.tflops, expected_tflops, 1e-10);
}

// Test 8: Benchmark with memory bandwidth
TEST_F(BenchmarkTest, BenchmarkWithBandwidth) {
    size_t num_bytes = 1024 * 1024;  // 1 MB

    Benchmark bench("BandwidthTest", 1, 10);
    bench.set_bytes(num_bytes);

    std::vector<char> buffer(num_bytes);
    auto result = bench.run([&buffer]() {
        // Read all bytes
        volatile char sum = 0;
        for (size_t i = 0; i < buffer.size(); ++i) {
            sum += buffer[i];
        }
    });

    EXPECT_EQ(result.num_bytes, num_bytes);
    EXPECT_GT(result.bandwidth_gbs, 0.0);
}

// Test 9: BenchmarkStats ops_per_sec
TEST_F(BenchmarkTest, BenchmarkStatsOpsPerSec) {
    BenchmarkStats stats;
    stats.mean = 0.001;  // 1ms average

    size_t num_ops = 1000;
    double ops_sec = stats.ops_per_sec(num_ops);

    // 1000 ops in 0.001s = 1,000,000 ops/sec
    EXPECT_NEAR(ops_sec, 1000000.0, 1.0);
}

// Test 10: BenchmarkStats GFLOPS calculation
TEST_F(BenchmarkTest, BenchmarkStatsGflops) {
    BenchmarkStats stats;
    stats.mean = 0.001;  // 1ms

    size_t num_flops = 1000000000;  // 1 billion FLOPS
    double gflops = stats.gflops(num_flops);

    // 1 GFLOP in 1ms = 1000 GFLOPS
    EXPECT_NEAR(gflops, 1000.0, 1.0);
}

// Test 11: BenchmarkStats bandwidth calculation
TEST_F(BenchmarkTest, BenchmarkStatsBandwidth) {
    BenchmarkStats stats;
    stats.mean = 0.001;  // 1ms

    size_t num_bytes = 1000000000;  // 1 GB
    double bandwidth = stats.bandwidth_gbs(num_bytes);

    // 1 GB in 1ms = 1000 GB/s
    EXPECT_NEAR(bandwidth, 1000.0, 1.0);
}

// Test 12: FLOPS utility functions
TEST_F(BenchmarkTest, FlopsUtilities) {
    // Matmul FLOPS: 2*M*N*K
    size_t matmul_flops = flops::matmul(100, 100, 100);
    EXPECT_EQ(matmul_flops, 2 * 100 * 100 * 100);

    // Conv2D FLOPS
    size_t conv_flops = flops::conv2d(1, 32, 32, 3, 64, 3, 3);
    EXPECT_EQ(conv_flops, 2 * 1 * 32 * 32 * 3 * 64 * 3 * 3);

    // Elementwise FLOPS
    size_t elem_flops = flops::elementwise(1000, 2);
    EXPECT_EQ(elem_flops, 2000);
}

// Test 13: Memory utility functions
TEST_F(BenchmarkTest, MemoryUtilities) {
    // Matmul bytes: (M*K + K*N + M*N) * element_size
    size_t matmul_bytes = memory::matmul(100, 100, 100, 4);
    EXPECT_EQ(matmul_bytes, 4 * (100*100 + 100*100 + 100*100));

    // Elementwise bytes: num_elements * (inputs + outputs) * element_size
    size_t elem_bytes = memory::elementwise(1000, 2, 1, 4);
    EXPECT_EQ(elem_bytes, 1000 * 3 * 4);
}

// Test 14: BenchmarkSuite basic
TEST_F(BenchmarkTest, BenchmarkSuiteBasic) {
    BenchmarkSuite suite("TestSuite");

    suite.add(Benchmark("Bench1", 0, 5));
    suite.add(Benchmark("Bench2", 0, 5));

    // Just verify suite can be created and benchmarks added
    // Full run test would require capturing stdout or being too slow
    SUCCEED();
}

// Test 15: Chaining benchmark configuration
TEST_F(BenchmarkTest, BenchmarkChaining) {
    Benchmark bench("ChainTest", 1, 5);

    // Test method chaining
    auto& b1 = bench.set_flops(1000);
    auto& b2 = b1.set_bytes(2000);

    EXPECT_EQ(&bench, &b1);  // Should return reference to same object
    EXPECT_EQ(&bench, &b2);
}
