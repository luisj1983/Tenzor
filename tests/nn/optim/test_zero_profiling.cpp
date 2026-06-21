/**
 * @file test_zero_profiling.cpp
 * @brief Tests for ZeRO optimizer performance profiling infrastructure
 *
 * Phase 7 - Task 1: Performance Profiling
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/optim/zero_optimizer.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <tenzor/backend/loader.hpp>
#include <memory>
#include <vector>
#include <thread>
#include <chrono>

using namespace tenzor;
using namespace tenzor::optim;
using namespace tenzor::nn;

/**
 * @brief Test fixture for ZeRO profiling tests
 */
class ZeROProfilingTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        // Initialize Tenzor library (loads backends)
        tenzor::initialize();
    }

    void SetUp() override {
        // Create test parameters
        auto param1 = std::make_shared<Variable>(Tensor({128, 256}, DType::Float32, Device::cpu()));
        auto param2 = std::make_shared<Variable>(Tensor({256, 512}, DType::Float32, Device::cpu()));
        auto param3 = std::make_shared<Variable>(Tensor({512, 128}, DType::Float32, Device::cpu()));

        // Initialize with small values (simulating trained weights)
        param1->tensor().fill_(0.01f);
        param2->tensor().fill_(0.01f);
        param3->tensor().fill_(0.01f);

        // Create mock gradients
        param1->set_grad(Tensor({128, 256}, DType::Float32, Device::cpu()));
        param2->set_grad(Tensor({256, 512}, DType::Float32, Device::cpu()));
        param3->set_grad(Tensor({512, 128}, DType::Float32, Device::cpu()));

        param1->mutable_grad()->fill_(0.1f);
        param2->mutable_grad()->fill_(0.1f);
        param3->mutable_grad()->fill_(0.1f);

        parameters_ = {param1, param2, param3};
    }

    std::vector<std::shared_ptr<Variable>> parameters_;
};

/**
 * @brief Test basic profiling enable/disable
 */
TEST_F(ZeROProfilingTest, EnableDisableProfiling) {
    // Create base optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    // Create ZeRO Stage 1 optimizer (single rank for testing)
    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;  // No distributed for this test

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Initially disabled
    EXPECT_FALSE(optimizer.is_profiling_enabled());

    // Enable profiling
    optimizer.enable_profiling(true);
    EXPECT_TRUE(optimizer.is_profiling_enabled());

    // Get stats (should be empty)
    auto stats = optimizer.get_profiling_stats();
    EXPECT_EQ(stats.num_steps, 0);
    EXPECT_EQ(stats.num_gathers, 0);
    EXPECT_EQ(stats.num_scatters, 0);

    // Disable profiling
    optimizer.enable_profiling(false);
    EXPECT_FALSE(optimizer.is_profiling_enabled());
}

/**
 * @brief Test profiling captures step timing
 */
TEST_F(ZeROProfilingTest, CaptureStepTiming) {
    // Create base optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    // Create ZeRO Stage 1 optimizer
    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform optimization steps
    const int num_steps = 5;
    for (int i = 0; i < num_steps; ++i) {
        optimizer.step();
    }

    // Get profiling stats
    auto stats = optimizer.get_profiling_stats();

    // Verify step count
    EXPECT_EQ(stats.num_steps, num_steps);

    // Verify timing is captured
    EXPECT_GT(stats.total_step_time_ms, 0.0);
    EXPECT_GT(stats.avg_step_time_ms, 0.0);
    EXPECT_GT(stats.compute_time_ms, 0.0);

    // Verify average is correct
    EXPECT_DOUBLE_EQ(stats.avg_step_time_ms, stats.total_step_time_ms / num_steps);

    // Verify compute time is reasonable (should be most of the time)
    EXPECT_GT(stats.compute_time_ms, stats.total_step_time_ms * 0.5);
}

/**
 * @brief Test profiling tracks memory statistics
 */
TEST_F(ZeROProfilingTest, TrackMemoryStatistics) {
    // Create base optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    // Create ZeRO Stage 1 optimizer
    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform steps
    optimizer.step();
    optimizer.step();

    // Get profiling stats
    auto stats = optimizer.get_profiling_stats();

    // Verify memory stats are captured
    EXPECT_GT(stats.current_memory_bytes, 0);
    EXPECT_GE(stats.peak_memory_bytes, stats.current_memory_bytes);

    // Verify memory is reasonable for our parameters
    // Adam has 2 state buffers per parameter (momentum + variance)
    size_t expected_min_memory = 0;
    for (const auto& param : parameters_) {
        size_t param_bytes = param->tensor().numel() * dtype_size(param->tensor().dtype());
        expected_min_memory += param_bytes * 2;  // momentum + variance
    }

    EXPECT_GE(stats.current_memory_bytes, expected_min_memory);
}

/**
 * @brief Test profiling reset functionality
 */
TEST_F(ZeROProfilingTest, ResetProfilingStats) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling and perform steps
    optimizer.enable_profiling(true);
    optimizer.step();
    optimizer.step();

    // Verify stats are non-zero
    auto stats_before = optimizer.get_profiling_stats();
    EXPECT_GT(stats_before.num_steps, 0);
    EXPECT_GT(stats_before.total_step_time_ms, 0.0);

    // Reset stats
    optimizer.reset_profiling_stats();

    // Verify stats are reset
    auto stats_after = optimizer.get_profiling_stats();
    EXPECT_EQ(stats_after.num_steps, 0);
    EXPECT_EQ(stats_after.total_step_time_ms, 0.0);
    EXPECT_EQ(stats_after.compute_time_ms, 0.0);
    EXPECT_EQ(stats_after.communication_time_ms, 0.0);
    EXPECT_EQ(stats_after.num_gathers, 0);
    EXPECT_EQ(stats_after.num_scatters, 0);
}

/**
 * @brief Test profiling summary output
 */
TEST_F(ZeROProfilingTest, ProfilingSummaryOutput) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform steps
    optimizer.step();
    optimizer.step();
    optimizer.step();

    // Get stats
    auto stats = optimizer.get_profiling_stats();

    // Test string output
    std::string summary = stats.to_string();
    EXPECT_FALSE(summary.empty());
    EXPECT_NE(summary.find("steps"), std::string::npos);
    EXPECT_NE(summary.find("ms/step"), std::string::npos);

    // Test print_summary (just verify it doesn't crash)
    testing::internal::CaptureStdout();
    stats.print_summary();
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_FALSE(output.empty());
    EXPECT_NE(output.find("ZeRO Optimizer Profiling Summary"), std::string::npos);
    EXPECT_NE(output.find("Timing Statistics"), std::string::npos);
}

/**
 * @brief Test profiling with CPU offload
 *
 * This test verifies that profiling correctly captures statistics when CPU offload is enabled.
 * CPU offload stores optimizer states (momentum/variance) on CPU to save memory.
 */
TEST_F(ZeROProfilingTest, ProfilingWithCPUOffload) {
    // Use CPU parameters (CPU offload is for optimizer states, not parameters)
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = true;  // Offload optimizer states to CPU
    config.cpu_offload_threshold = 0;  // Offload everything
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform steps
    optimizer.step();
    optimizer.step();

    // Get stats
    auto stats = optimizer.get_profiling_stats();

    // Verify profiling captured basic timing
    EXPECT_GT(stats.total_step_time_ms, 0.0);
    EXPECT_GT(stats.compute_time_ms, 0.0);

    // Verify offload is enabled (profiling tracks offload operations)
    EXPECT_TRUE(optimizer.is_cpu_offload_enabled());

    // Memory stats should show optimizer states on CPU
    auto mem_stats = optimizer.get_memory_stats();
    EXPECT_GT(mem_stats.cpu_optimizer_memory, 0);

    // Peak memory should be tracked
    EXPECT_GT(stats.peak_memory_bytes, 0);
}

/**
 * @brief Test profiling accuracy with known delays
 */
TEST_F(ZeROProfilingTest, ProfilingAccuracy) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Measure multiple steps
    const int num_steps = 10;
    for (int i = 0; i < num_steps; ++i) {
        optimizer.step();
    }

    auto stats = optimizer.get_profiling_stats();

    // Verify consistency
    EXPECT_EQ(stats.num_steps, num_steps);

    // Verify averages are computed correctly
    if (stats.num_steps > 0) {
        EXPECT_DOUBLE_EQ(stats.avg_step_time_ms, stats.total_step_time_ms / stats.num_steps);
    }

    // Verify time relationships
    // Total time should include compute time
    EXPECT_GE(stats.total_step_time_ms, stats.compute_time_ms * 0.9);  // Allow small measurement variance

    // All times should be non-negative
    EXPECT_GE(stats.total_step_time_ms, 0.0);
    EXPECT_GE(stats.compute_time_ms, 0.0);
    EXPECT_GE(stats.communication_time_ms, 0.0);
    EXPECT_GE(stats.offload_time_ms, 0.0);
}

/**
 * @brief Test profiling with Stage 2 optimizer
 */
TEST_F(ZeROProfilingTest, Stage2Profiling) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage2Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.gradient_bucketing = true;
    config.gradient_bucket_size = 25 * 1024 * 1024;
    config.process_group = nullptr;

    ZeROStage2Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform steps
    optimizer.step();
    optimizer.step();

    // Get stats
    auto stats = optimizer.get_profiling_stats();

    // Verify stats are captured
    EXPECT_EQ(stats.num_steps, 2);
    EXPECT_GT(stats.total_step_time_ms, 0.0);
    EXPECT_GT(stats.compute_time_ms, 0.0);

    // Stage 2 should have no all-reduce (uses reduce-scatter instead)
    EXPECT_EQ(stats.num_all_reduces, 0);
}

/**
 * @brief Test bandwidth calculation
 */
TEST_F(ZeROProfilingTest, BandwidthCalculation) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform steps
    optimizer.step();

    // Get stats
    auto stats = optimizer.get_profiling_stats();

    // Single rank with no process group performs NO inter-rank communication,
    // so the profiler must report a zero communication state — assert that
    // explicitly rather than skipping the body (the previous `if` was always
    // false here and verified nothing).
    EXPECT_EQ(stats.transferred_bytes, 0u)
        << "single-rank (no process group) must transfer zero bytes";
    EXPECT_DOUBLE_EQ(stats.communication_time_ms, 0.0)
        << "single-rank must record zero communication time";
    EXPECT_DOUBLE_EQ(stats.effective_bandwidth_mbps, 0.0)
        << "with no communication, effective bandwidth must be 0, not garbage";

    // The bandwidth formula must hold whenever communication is non-zero.
    if (stats.communication_time_ms > 0 && stats.transferred_bytes > 0) {
        double expected_bandwidth = (stats.transferred_bytes / (1024.0 * 1024.0)) /
                                   (stats.communication_time_ms / 1000.0);
        EXPECT_NEAR(stats.effective_bandwidth_mbps, expected_bandwidth, 0.01);
    }
}

/**
 * @brief Test overlap ratio calculation
 */
TEST_F(ZeROProfilingTest, OverlapRatioCalculation) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.overlap_comm = true;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform steps
    optimizer.step();
    optimizer.step();

    // Get stats
    auto stats = optimizer.get_profiling_stats();

    // Overlap ratio is always a valid fraction.
    EXPECT_GE(stats.comm_compute_overlap_ratio, 0.0);
    EXPECT_LE(stats.comm_compute_overlap_ratio, 1.0);

    // Stage 1 single-rank (no process group) performs no all-reduce, so there
    // is no communication to overlap: the ratio must be exactly 0 and the
    // communication time must be 0. Asserting the concrete single-rank state
    // turns this from a trivially-true [0,1] bound into a real check.
    EXPECT_DOUBLE_EQ(stats.communication_time_ms, 0.0)
        << "single-rank must record zero communication time";
    EXPECT_DOUBLE_EQ(stats.comm_compute_overlap_ratio, 0.0)
        << "with no communication there is nothing to overlap (ratio must be 0)";
}

/**
 * @brief Benchmark profiling overhead
 */
TEST_F(ZeROProfilingTest, ProfilingOverhead) {
    // Create two optimizers - one with profiling, one without
    auto adam1 = std::make_unique<Adam>(parameters_, 1e-3);
    auto adam2 = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer_with_profiling(std::move(adam1), config);
    ZeROStage1Optimizer optimizer_without_profiling(std::move(adam2), config);

    // Measure time with profiling disabled
    auto start1 = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) {
        optimizer_without_profiling.step();
    }
    auto end1 = std::chrono::steady_clock::now();
    auto duration_without = std::chrono::duration<double, std::milli>(end1 - start1).count();

    // Measure time with profiling enabled
    optimizer_with_profiling.enable_profiling(true);
    auto start2 = std::chrono::steady_clock::now();
    for (int i = 0; i < 100; ++i) {
        optimizer_with_profiling.step();
    }
    auto end2 = std::chrono::steady_clock::now();
    auto duration_with = std::chrono::duration<double, std::milli>(end2 - start2).count();

    // Profiling overhead should be minimal (< 10%)
    double overhead_ratio = (duration_with - duration_without) / duration_without;
    EXPECT_LT(overhead_ratio, 0.10) << "Profiling overhead is " << (overhead_ratio * 100) << "%";

    std::cout << "Profiling overhead: " << (overhead_ratio * 100) << "%" << std::endl;
}

/**
 * @brief Test profiling with multiple steps
 */
TEST_F(ZeROProfilingTest, MultipleStepsProfiling) {
    // Create optimizer
    auto adam = std::make_unique<Adam>(parameters_, 1e-3);

    ZeROStage1Config config;
    config.world_size = 1;
    config.rank = 0;
    config.offload_to_cpu = false;
    config.process_group = nullptr;

    ZeROStage1Optimizer optimizer(std::move(adam), config);

    // Enable profiling
    optimizer.enable_profiling(true);

    // Perform many steps
    const int num_steps = 100;
    for (int i = 0; i < num_steps; ++i) {
        optimizer.step();
    }

    // Get stats
    auto stats = optimizer.get_profiling_stats();

    // Verify all steps are counted
    EXPECT_EQ(stats.num_steps, num_steps);

    // Verify average is computed correctly
    EXPECT_DOUBLE_EQ(stats.avg_step_time_ms, stats.total_step_time_ms / num_steps);

    // Verify total time is sum of averages
    EXPECT_NEAR(stats.total_step_time_ms, stats.avg_step_time_ms * num_steps, 0.01);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
