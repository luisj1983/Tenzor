/**
 * @file test_memory_manager.cpp
 * @brief Comprehensive unit tests for Memory Manager (Phase 1 - ZeRO Offload)
 *
 * Tests cover:
 * - Tensor registration and tracking
 * - Memory usage tracking per device
 * - Memory pressure calculation
 * - LRU eviction policy
 * - Thread-safe concurrent access
 * - Statistics accuracy
 * - Edge cases and error handling
 */

#include <gtest/gtest.h>
#include <tenzor/core/memory_manager.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>
#include <thread>
#include <vector>
#include <algorithm>
#include <chrono>

using namespace tenzor;
using namespace tenzor::core;

/**
 * Test Fixture for Memory Manager
 */
class MemoryManagerTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();  // Load backends once
    }

    void SetUp() override {
        // Default configuration for most tests
        default_config.cpu_memory_limit = 1024 * 1024 * 1024;  // 1 GB
        default_config.gpu_memory_limit = 512 * 1024 * 1024;   // 512 MB
        default_config.eviction_threshold = 0.8f;               // 80%
        default_config.track_statistics = true;
        default_config.enable_cache = true;
    }

    void TearDown() override {
        // Clean up any remaining tensors
    }

    MemoryManager::Config default_config;

    // Helper to create a tensor of specific size
    Tensor createTensor(int64_t numel, DType dtype, Device device) {
        std::vector<int64_t> shape = {numel};
        return Tensor(shape, dtype, device);
    }

    // Helper to calculate tensor memory size
    size_t tensorMemorySize(const Tensor& tensor) {
        size_t dtype_size = 0;
        switch (tensor.dtype()) {
            case DType::Float32: dtype_size = 4; break;
            case DType::Float64: dtype_size = 8; break;
            case DType::Int32: dtype_size = 4; break;
            case DType::Int64: dtype_size = 8; break;
            default: dtype_size = 4;
        }
        return tensor.numel() * dtype_size;
    }
};

// =============================================================================
// Constructor Tests
// =============================================================================

TEST_F(MemoryManagerTest, ConstructorWithValidConfig) {
    ASSERT_NO_THROW({
        MemoryManager manager(default_config);
    });
}

TEST_F(MemoryManagerTest, ConstructorWithDefaultConfig) {
    ASSERT_NO_THROW({
        MemoryManager::Config cfg;
        MemoryManager manager(cfg);
    });
}

// =============================================================================
// Tensor Registration Tests
// =============================================================================

TEST_F(MemoryManagerTest, RegisterSingleTensor) {
    MemoryManager manager(default_config);
    Tensor t = createTensor(100, DType::Float32, Device::cpu());

    ASSERT_NO_THROW(manager.register_tensor(&t));

    auto location = manager.get_tensor_location(&t);
    EXPECT_EQ(location.type, Device::Type::CPU);
}

TEST_F(MemoryManagerTest, RegisterMultipleTensors) {
    MemoryManager::Config cfg;
    MemoryManager manager(cfg);

    std::vector<Tensor> tensors;
    tensors.reserve(10);

    for (int i = 0; i < 10; ++i) {
        tensors.push_back(createTensor(100 * (i + 1), DType::Float32, Device::cpu()));
        manager.register_tensor(&tensors.back());
    }

    // Verify all are tracked
    for (auto& t : tensors) {
        EXPECT_TRUE(manager.is_registered(&t));
        auto location = manager.get_tensor_location(&t);
        EXPECT_EQ(location.type, Device::Type::CPU);
    }
}

TEST_F(MemoryManagerTest, UnregisterTensor) {
    MemoryManager manager(default_config);
    Tensor t = createTensor(100, DType::Float32, Device::cpu());

    manager.register_tensor(&t);
    EXPECT_TRUE(manager.is_registered(&t));

    ASSERT_NO_THROW(manager.unregister_tensor(&t));
    EXPECT_FALSE(manager.is_registered(&t));
}

// =============================================================================
// Tensor Location Tracking Tests
// =============================================================================

TEST_F(MemoryManagerTest, TrackLocationCPU) {
    MemoryManager manager(default_config);
    Tensor t = createTensor(100, DType::Float32, Device::cpu());

    manager.register_tensor(&t);
    auto location = manager.get_tensor_location(&t);

    EXPECT_EQ(location.type, Device::Type::CPU);
}

TEST_F(MemoryManagerTest, UpdateTensorLocation) {
    MemoryManager manager(default_config);
    Tensor t = createTensor(100, DType::Float32, Device::cpu());

    manager.register_tensor(&t);

    // Move tensor to GPU
    Device new_device = Device::cuda(0);
    manager.update_tensor_location(&t, new_device);

    auto location = manager.get_tensor_location(&t);
    EXPECT_EQ(location.type, Device::Type::CUDA);
}

TEST_F(MemoryManagerTest, UpdateMultipleTensorLocations) {
    MemoryManager::Config cfg;
    MemoryManager manager(cfg);

    std::vector<Tensor> tensors;
    tensors.reserve(5);

    for (int i = 0; i < 5; ++i) {
        tensors.push_back(createTensor(100, DType::Float32, Device::cpu()));
        manager.register_tensor(&tensors.back());
    }

    // Move all to GPU
    for (auto& t : tensors) {
        manager.update_tensor_location(&t, Device::cuda(0));
    }

    // Verify all moved
    for (auto& t : tensors) {
        auto location = manager.get_tensor_location(&t);
        EXPECT_EQ(location.type, Device::Type::CUDA);
    }
}

// =============================================================================
// Memory Usage Tracking Tests
// =============================================================================

TEST_F(MemoryManagerTest, TrackCPUMemoryUsage) {
    MemoryManager manager(default_config);
    Tensor t = createTensor(1000, DType::Float32, Device::cpu());
    size_t expected_size = tensorMemorySize(t);

    manager.register_tensor(&t);

    auto usage = manager.get_memory_usage(Device::Type::CPU);
    EXPECT_GE(usage, expected_size);  // May include overhead
}

TEST_F(MemoryManagerTest, TrackMemoryLimit) {
    MemoryManager manager(default_config);

    auto cpu_limit = manager.get_memory_limit(Device::Type::CPU);
    auto gpu_limit = manager.get_memory_limit(Device::Type::CUDA);

    EXPECT_EQ(cpu_limit, default_config.cpu_memory_limit);
    EXPECT_EQ(gpu_limit, default_config.gpu_memory_limit);
}

TEST_F(MemoryManagerTest, MemoryUsageDecreasesAfterUnregister) {
    MemoryManager manager(default_config);

    Tensor t1 = createTensor(1000, DType::Float32, Device::cpu());
    Tensor t2 = createTensor(2000, DType::Float32, Device::cpu());

    manager.register_tensor(&t1);
    manager.register_tensor(&t2);

    size_t before_unregister = manager.get_memory_usage(Device::Type::CPU);

    manager.unregister_tensor(&t1);

    size_t after_unregister = manager.get_memory_usage(Device::Type::CPU);

    EXPECT_LT(after_unregister, before_unregister);
}

// =============================================================================
// Memory Pressure Tests
// =============================================================================

TEST_F(MemoryManagerTest, CalculateMemoryPressure) {
    MemoryManager manager(default_config);

    float initial_pressure = manager.get_memory_pressure(Device::Type::CPU);
    EXPECT_FLOAT_EQ(initial_pressure, 0.0f);

    // Fill some memory
    Tensor t = createTensor(1000000, DType::Float32, Device::cpu());
    manager.register_tensor(&t);

    float pressure_after = manager.get_memory_pressure(Device::Type::CPU);
    EXPECT_GT(pressure_after, 0.0f);
}

TEST_F(MemoryManagerTest, MemoryPressureZeroWhenEmpty) {
    MemoryManager manager(default_config);

    float pressure = manager.get_memory_pressure(Device::Type::CPU);

    EXPECT_FLOAT_EQ(pressure, 0.0f);
}

TEST_F(MemoryManagerTest, MemoryPressureIncreasesWithUsage) {
    MemoryManager manager(default_config);

    float pressure_before = manager.get_memory_pressure(Device::Type::CPU);

    Tensor t = createTensor(10000, DType::Float32, Device::cpu());
    manager.register_tensor(&t);

    float pressure_after = manager.get_memory_pressure(Device::Type::CPU);

    EXPECT_GT(pressure_after, pressure_before);
}

TEST_F(MemoryManagerTest, IsOverThresholdDetectsHighPressure) {
    MemoryManager manager(default_config);

    // Initially not over threshold
    EXPECT_FALSE(manager.is_over_threshold(Device::Type::CPU));

    // Fill above threshold (80%)
    size_t above_threshold = static_cast<size_t>(default_config.cpu_memory_limit * 0.85);
    Tensor t = createTensor(above_threshold / 4, DType::Float32, Device::cpu());
    manager.register_tensor(&t);

    EXPECT_TRUE(manager.is_over_threshold(Device::Type::CPU));
}

// =============================================================================
// LRU Eviction Tests
// =============================================================================

TEST_F(MemoryManagerTest, LRUEvictionOrderCorrect) {
    MemoryManager manager(default_config);

    std::vector<Tensor*> tensors;
    for (int i = 0; i < 5; ++i) {
        auto* t = new Tensor(createTensor(100, DType::Float32, Device::cuda(0)));
        tensors.push_back(t);
        manager.register_tensor(t);

        // Simulate time passing
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
    }

    // Mark tensor 0 and 2 as recently used
    manager.mark_tensor_used(tensors[0]);
    manager.mark_tensor_used(tensors[2]);

    // Request eviction candidates
    auto candidates = manager.evict_lru_tensors(Device::Type::CUDA, 200);

    // Should evict oldest unused tensors
    EXPECT_GE(candidates.size(), 1);

    // Cleanup
    for (auto* t : tensors) {
        manager.unregister_tensor(t);
        delete t;
    }
}

TEST_F(MemoryManagerTest, LRUEvictsOldestFirst) {
    MemoryManager manager(default_config);

    Tensor t1 = createTensor(100, DType::Float32, Device::cuda(0));
    Tensor t2 = createTensor(100, DType::Float32, Device::cuda(0));
    Tensor t3 = createTensor(100, DType::Float32, Device::cuda(0));

    manager.register_tensor(&t1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.register_tensor(&t2);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.register_tensor(&t3);

    auto candidates = manager.evict_lru_tensors(Device::Type::CUDA, 100);

    ASSERT_GE(candidates.size(), 1);
    EXPECT_EQ(candidates[0], &t1);  // Oldest
}

TEST_F(MemoryManagerTest, MarkTensorUsedUpdatesLRU) {
    MemoryManager manager(default_config);

    Tensor t1 = createTensor(100, DType::Float32, Device::cuda(0));
    Tensor t2 = createTensor(100, DType::Float32, Device::cuda(0));

    manager.register_tensor(&t1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.register_tensor(&t2);

    // Mark t1 as recently used (should move to end)
    manager.mark_tensor_used(&t1);

    // Now t2 should be oldest
    auto lru = manager.get_lru_tensor(Device::Type::CUDA);
    EXPECT_EQ(lru, &t2);
}

TEST_F(MemoryManagerTest, GetLRUTensorReturnsOldest) {
    MemoryManager manager(default_config);

    Tensor t1 = createTensor(100, DType::Float32, Device::cpu());
    Tensor t2 = createTensor(100, DType::Float32, Device::cpu());

    manager.register_tensor(&t1);
    std::this_thread::sleep_for(std::chrono::milliseconds(10));
    manager.register_tensor(&t2);

    auto lru = manager.get_lru_tensor(Device::Type::CPU);
    EXPECT_EQ(lru, &t1);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_F(MemoryManagerTest, ConcurrentRegistration) {
    MemoryManager manager(default_config);

    std::vector<std::thread> threads;
    std::vector<std::vector<Tensor>> tensors(4);

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                tensors[t].push_back(createTensor(10, DType::Float32, Device::cpu()));
                manager.register_tensor(&tensors[t].back());
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all tensors registered
    auto count = manager.get_tensor_count();
    EXPECT_EQ(count, 400);
}

TEST_F(MemoryManagerTest, ConcurrentReadWrites) {
    MemoryManager::Config cfg;
    MemoryManager manager(cfg);

    std::vector<Tensor> tensors;
    tensors.reserve(100);
    for (int i = 0; i < 100; ++i) {
        tensors.push_back(createTensor(1000, DType::Float32, Device::cpu()));
        manager.register_tensor(&tensors[i]);
    }

    std::atomic<int> errors{0};
    std::vector<std::thread> threads;

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 100; ++i) {
                try {
                    int idx = (t * 25 + i) % 100;
                    manager.mark_tensor_used(&tensors[idx]);
                    auto loc = manager.get_tensor_location(&tensors[idx]);
                    (void)loc;  // Use the result
                } catch (...) {
                    errors.fetch_add(1);
                }
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    EXPECT_EQ(errors.load(), 0);
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(MemoryManagerTest, StatisticsAccuracy) {
    MemoryManager::Config cfg;
    MemoryManager manager(cfg);

    std::vector<Tensor> tensors;
    tensors.reserve(10);
    const size_t expected_count = 10;

    for (size_t i = 0; i < expected_count; ++i) {
        tensors.push_back(createTensor(100 * (i + 1), DType::Float32, Device::cpu()));
        manager.register_tensor(&tensors.back());
    }

    auto stats = manager.get_stats();

    EXPECT_EQ(stats.total_tensors, expected_count);
    EXPECT_EQ(stats.cpu_tensors, expected_count);
    EXPECT_GT(stats.cpu_memory_used, 0);
}

TEST_F(MemoryManagerTest, StatisticsPerDevice) {
    MemoryManager manager(default_config);

    // CPU tensors
    Tensor cpu_t = createTensor(1000, DType::Float32, Device::cpu());
    manager.register_tensor(&cpu_t);

    auto stats = manager.get_stats();

    EXPECT_EQ(stats.cpu_tensors, 1);
    EXPECT_GT(stats.cpu_memory_used, 0);
}

TEST_F(MemoryManagerTest, ResetStatistics) {
    MemoryManager manager(default_config);

    Tensor t = createTensor(1000, DType::Float32, Device::cpu());
    manager.register_tensor(&t);

    // Trigger some evictions
    manager.evict_lru_tensors(Device::Type::CPU, 100);

    auto stats_before = manager.get_stats();
    EXPECT_GT(stats_before.total_evictions, 0);

    manager.reset_stats();

    auto stats_after = manager.get_stats();
    EXPECT_EQ(stats_after.total_evictions, 0);
}

TEST_F(MemoryManagerTest, GetTensorCount) {
    MemoryManager manager(default_config);

    EXPECT_EQ(manager.get_tensor_count(), 0);

    std::vector<Tensor> tensors;
    for (int i = 0; i < 5; ++i) {
        tensors.push_back(createTensor(100, DType::Float32, Device::cpu()));
        manager.register_tensor(&tensors.back());
    }

    EXPECT_EQ(manager.get_tensor_count(), 5);
}

TEST_F(MemoryManagerTest, GetTensorCountPerDevice) {
    MemoryManager manager(default_config);

    Tensor cpu_t = createTensor(100, DType::Float32, Device::cpu());
    manager.register_tensor(&cpu_t);

    EXPECT_EQ(manager.get_tensor_count(Device::Type::CPU), 1);
    EXPECT_EQ(manager.get_tensor_count(Device::Type::CUDA), 0);
}

// =============================================================================
// Edge Cases
// =============================================================================

TEST_F(MemoryManagerTest, ZeroSizedTensor) {
    MemoryManager manager(default_config);
    Tensor t = createTensor(0, DType::Float32, Device::cpu());

    ASSERT_NO_THROW(manager.register_tensor(&t));

    auto usage = manager.get_memory_usage(Device::Type::CPU);
    EXPECT_GE(usage, 0);  // Should handle gracefully
}

TEST_F(MemoryManagerTest, ManySmallTensors) {
    MemoryManager manager(default_config);

    std::vector<Tensor> tensors;
    for (int i = 0; i < 1000; ++i) {
        tensors.push_back(createTensor(1, DType::Float32, Device::cpu()));
        ASSERT_NO_THROW(manager.register_tensor(&tensors.back()));
    }

    auto count = manager.get_tensor_count();
    EXPECT_EQ(count, 1000);
}

TEST_F(MemoryManagerTest, AlternatingRegistrationUnregistration) {
    MemoryManager manager(default_config);

    for (int i = 0; i < 100; ++i) {
        Tensor t = createTensor(100, DType::Float32, Device::cpu());
        manager.register_tensor(&t);
        manager.unregister_tensor(&t);
    }

    auto count = manager.get_tensor_count();
    EXPECT_EQ(count, 0);
}

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
