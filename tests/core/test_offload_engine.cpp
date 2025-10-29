/**
 * @file test_offload_engine.cpp
 * @brief Comprehensive tests for OffloadEngine (Phase 2 ZeRO Offload)
 */

#include <gtest/gtest.h>
#include "tenzor/core/offload_engine.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/backend/loader.hpp"
#include <chrono>

using namespace tenzor;
using namespace tenzor::core;

class OffloadEngineTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        // Check if CUDA backend is available
        auto* backend = backend_registry().get_backend("cuda");
        cuda_available = (backend != nullptr && backend->is_available());

        if (cuda_available) {
            default_config.pinned_memory_size = 256 * 1024 * 1024;  // 256 MB
            default_config.num_transfer_streams = 4;
            default_config.enable_prefetch = true;
            default_config.prefetch_depth = 2;
            default_config.memory_fraction = 0.25f;
        }
    }

    bool cuda_available = false;
    OffloadEngine::Config default_config;

    // Helper to create test tensor with pattern
    Tensor createPatternTensor(const std::vector<int64_t>& shape, DType dtype, Device device) {
        Tensor cpu_t(shape, dtype, Device::cpu());

        if (dtype == DType::Float32) {
            auto* data = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) {
                data[i] = static_cast<float>(i % 1000);
            }
        }

        if (device.type != Device::Type::CPU) {
            return cpu_t.to(device);
        }

        return cpu_t;
    }

    // Helper to verify tensor data
    bool verifyPatternTensor(const Tensor& t) {
        if (t.dtype() != DType::Float32) return false;

        Tensor cpu_t = (t.device().type == Device::Type::CPU) ? t : t.to(Device::cpu());
        auto* data = cpu_t.data<float>();

        for (int64_t i = 0; i < cpu_t.numel(); ++i) {
            float expected = static_cast<float>(i % 1000);
            if (std::abs(data[i] - expected) > 1e-5f) {
                return false;
            }
        }

        return true;
    }
};

// =============================================================================
// Constructor Tests
// =============================================================================

TEST_F(OffloadEngineTest, ConstructorWithValidConfig) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    ASSERT_NO_THROW({
        OffloadEngine engine(default_config);
    });
}

TEST_F(OffloadEngineTest, ConstructorWithDefaultConfig) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    ASSERT_NO_THROW({
        OffloadEngine engine(config);
    });
}

TEST_F(OffloadEngineTest, ConstructorInitializesResources) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // Verify pinned memory stats
    auto stats = engine.get_pinned_memory_stats();
    EXPECT_EQ(stats.total_size, default_config.pinned_memory_size);
    EXPECT_EQ(stats.allocated_size, 0);
    EXPECT_EQ(stats.free_size, default_config.pinned_memory_size);
}

// =============================================================================
// Synchronous API Tests
// =============================================================================

TEST_F(OffloadEngineTest, SyncOffloadToCPU_BasicTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor gpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cuda());
    Tensor cpu_tensor = engine.offload_to_cpu(gpu_tensor);

    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    EXPECT_TRUE(verifyPatternTensor(cpu_tensor));
}

TEST_F(OffloadEngineTest, SyncOffloadToCPU_LargeTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // 10 MB tensor
    Tensor gpu_tensor = createPatternTensor({2500, 1000}, DType::Float32, Device::cuda());
    Tensor cpu_tensor = engine.offload_to_cpu(gpu_tensor);

    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    EXPECT_TRUE(std::equal(cpu_tensor.shape().begin(), cpu_tensor.shape().end(),
                          gpu_tensor.shape().begin(), gpu_tensor.shape().end()));
}

TEST_F(OffloadEngineTest, SyncLoadToGPU_BasicTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine.load_to_gpu(cpu_tensor);

    EXPECT_EQ(gpu_tensor.device().type, Device::Type::CUDA);
    EXPECT_TRUE(verifyPatternTensor(gpu_tensor));
}

TEST_F(OffloadEngineTest, SyncLoadToGPU_LargeTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor cpu_tensor = createPatternTensor({2500, 1000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine.load_to_gpu(cpu_tensor);

    EXPECT_EQ(gpu_tensor.device().type, Device::Type::CUDA);
    EXPECT_TRUE(std::equal(gpu_tensor.shape().begin(), gpu_tensor.shape().end(),
                          cpu_tensor.shape().begin(), cpu_tensor.shape().end()));
}

TEST_F(OffloadEngineTest, SyncRoundTrip_PreservesData) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor original = createPatternTensor({1000, 500}, DType::Float32, Device::cuda());
    Tensor cpu = engine.offload_to_cpu(original);
    Tensor restored = engine.load_to_gpu(cpu);

    EXPECT_TRUE(verifyPatternTensor(restored));
}

TEST_F(OffloadEngineTest, SyncOffload_MultipleTypes) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // Test Float32
    Tensor float32_gpu = zeros({100}, DType::Float32, Device::cuda());
    Tensor float32_cpu = engine.offload_to_cpu(float32_gpu);
    EXPECT_EQ(float32_cpu.dtype(), DType::Float32);

    // Test Int32
    Tensor int32_gpu = zeros({100}, DType::Int32, Device::cuda());
    Tensor int32_cpu = engine.offload_to_cpu(int32_gpu);
    EXPECT_EQ(int32_cpu.dtype(), DType::Int32);
}

// =============================================================================
// Asynchronous API Tests
// =============================================================================

TEST_F(OffloadEngineTest, AsyncOffloadToCPU_BasicTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor gpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cuda());
    auto handle = engine.offload_to_cpu_async(gpu_tensor);

    EXPECT_TRUE(handle.is_valid());

    Tensor cpu_tensor = handle.get_tensor();
    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    EXPECT_TRUE(verifyPatternTensor(cpu_tensor));
}

TEST_F(OffloadEngineTest, AsyncOffloadToCPU_ReturnsHandle) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor gpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cuda());
    auto handle = engine.offload_to_cpu_async(gpu_tensor);

    EXPECT_TRUE(handle.is_valid());
}

TEST_F(OffloadEngineTest, AsyncOffloadToCPU_CanWait) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor gpu_tensor = createPatternTensor({5000, 1000}, DType::Float32, Device::cuda());
    auto handle = engine.offload_to_cpu_async(gpu_tensor);

    // Wait for completion
    handle.wait();
    EXPECT_TRUE(handle.is_ready());
}

TEST_F(OffloadEngineTest, AsyncLoadToGPU_BasicTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());
    auto handle = engine.load_to_gpu_async(cpu_tensor);

    Tensor gpu_tensor = handle.get_tensor();
    EXPECT_EQ(gpu_tensor.device().type, Device::Type::CUDA);
    EXPECT_TRUE(verifyPatternTensor(gpu_tensor));
}

TEST_F(OffloadEngineTest, AsyncTransfers_MultipleSimultaneous) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // Start multiple async transfers
    std::vector<TransferHandle> handles;
    for (int i = 0; i < 4; ++i) {
        Tensor gpu_tensor = createPatternTensor({500, 250}, DType::Float32, Device::cuda());
        handles.push_back(engine.offload_to_cpu_async(gpu_tensor));
    }

    // Verify all complete
    for (auto& handle : handles) {
        Tensor cpu_tensor = handle.get_tensor();
        EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    }
}

TEST_F(OffloadEngineTest, AsyncTransfers_CorrectOrdering) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    std::vector<Tensor> gpu_tensors;
    std::vector<TransferHandle> handles;

    // Create and offload tensors with unique patterns
    for (int i = 0; i < 3; ++i) {
        gpu_tensors.push_back(createPatternTensor({100 * (i + 1)}, DType::Float32, Device::cuda()));
        handles.push_back(engine.offload_to_cpu_async(gpu_tensors[i]));
    }

    // Verify each tensor
    for (size_t i = 0; i < handles.size(); ++i) {
        Tensor cpu_tensor = handles[i].get_tensor();
        EXPECT_EQ(cpu_tensor.numel(), gpu_tensors[i].numel());
    }
}

TEST_F(OffloadEngineTest, AsyncPrefetch_SingleTensor) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // Create tensor on CPU
    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());

    // Prefetch to GPU
    std::vector<Tensor*> tensors = {&cpu_tensor};
    ASSERT_NO_THROW({
        engine.prefetch_to_gpu(tensors);
    });

    // Wait for prefetch
    engine.wait_for_prefetch();
}

TEST_F(OffloadEngineTest, AsyncPrefetch_MultipleTensors) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // Create tensors on CPU
    Tensor t1 = createPatternTensor({100}, DType::Float32, Device::cpu());
    Tensor t2 = createPatternTensor({200}, DType::Float32, Device::cpu());
    Tensor t3 = createPatternTensor({300}, DType::Float32, Device::cpu());

    // Prefetch all
    std::vector<Tensor*> tensors = {&t1, &t2, &t3};
    ASSERT_NO_THROW({
        engine.prefetch_to_gpu(tensors);
    });
}

// =============================================================================
// Memory Management Tests
// =============================================================================

TEST_F(OffloadEngineTest, PinnedMemoryStats_Accurate) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    auto initial_stats = engine.get_pinned_memory_stats();
    EXPECT_EQ(initial_stats.total_size, default_config.pinned_memory_size);
    EXPECT_EQ(initial_stats.allocated_size, 0);
}

TEST_F(OffloadEngineTest, AutoOffload_Registration) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor tensor = createPatternTensor({1000}, DType::Float32, Device::cuda());

    ASSERT_NO_THROW({
        engine.register_auto_offload(&tensor, OffloadPriority::NORMAL);
    });

    // Unregister
    ASSERT_NO_THROW({
        engine.unregister_auto_offload(&tensor);
    });
}

TEST_F(OffloadEngineTest, AutoOffload_ByPriority) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor low_priority = createPatternTensor({100}, DType::Float32, Device::cuda());
    Tensor high_priority = createPatternTensor({100}, DType::Float32, Device::cuda());

    engine.register_auto_offload(&low_priority, OffloadPriority::LOW);
    engine.register_auto_offload(&high_priority, OffloadPriority::HIGH);

    // Check and offload (would offload low priority first)
    engine.check_and_offload();
}

TEST_F(OffloadEngineTest, MemoryPressure_Calculation) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    float pressure = engine.get_gpu_memory_pressure();
    EXPECT_GE(pressure, 0.0f);
    EXPECT_LE(pressure, 1.0f);
}

// =============================================================================
// Prefetch Tests
// =============================================================================

TEST_F(OffloadEngineTest, Prefetch_StartsTransferEarly) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());

    // Start prefetch
    std::vector<Tensor*> tensors = {&cpu_tensor};
    engine.prefetch_to_gpu(tensors);

    // Prefetch should complete in background
    std::this_thread::sleep_for(std::chrono::milliseconds(100));
}

TEST_F(OffloadEngineTest, Prefetch_WithDisabledConfig) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config = default_config;
    config.enable_prefetch = false;

    OffloadEngine engine(config);

    Tensor cpu_tensor = createPatternTensor({100}, DType::Float32, Device::cpu());

    // Should not crash even with prefetch disabled
    std::vector<Tensor*> tensors = {&cpu_tensor};
    ASSERT_NO_THROW({
        engine.prefetch_to_gpu(tensors);
    });
}

// =============================================================================
// Performance Tests
// =============================================================================

TEST_F(OffloadEngineTest, BandwidthMeasurement_Offload) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // 100 MB tensor
    Tensor gpu_tensor = createPatternTensor({25000, 1000}, DType::Float32, Device::cuda());
    size_t bytes = gpu_tensor.numel() * sizeof(float);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor cpu_tensor = engine.offload_to_cpu(gpu_tensor);
    auto end = std::chrono::high_resolution_clock::now();

    double time_s = std::chrono::duration<double>(end - start).count();
    double bandwidth_gbps = (bytes / 1e9) / time_s;

    std::cout << "Offload bandwidth: " << bandwidth_gbps << " GB/s\n";
    EXPECT_GT(bandwidth_gbps, 0.5);  // At least 0.5 GB/s
}

TEST_F(OffloadEngineTest, BandwidthMeasurement_Load) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // 100 MB tensor
    Tensor cpu_tensor = createPatternTensor({25000, 1000}, DType::Float32, Device::cpu());
    size_t bytes = cpu_tensor.numel() * sizeof(float);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor gpu_tensor = engine.load_to_gpu(cpu_tensor);
    auto end = std::chrono::high_resolution_clock::now();

    double time_s = std::chrono::duration<double>(end - start).count();
    double bandwidth_gbps = (bytes / 1e9) / time_s;

    std::cout << "Load bandwidth: " << bandwidth_gbps << " GB/s\n";
    EXPECT_GT(bandwidth_gbps, 0.5);  // At least 0.5 GB/s
}

// =============================================================================
// Error Handling Tests
// =============================================================================

TEST_F(OffloadEngineTest, OffloadNonGPUTensor_ThrowsError) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor cpu_tensor = zeros({100}, DType::Float32, Device::cpu());

    EXPECT_THROW({
        engine.offload_to_cpu(cpu_tensor);
    }, std::runtime_error);
}

TEST_F(OffloadEngineTest, LoadNonCPUTensor_ThrowsError) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    Tensor gpu_tensor = zeros({100}, DType::Float32, Device::cuda());

    EXPECT_THROW({
        engine.load_to_gpu(gpu_tensor);
    }, std::runtime_error);
}

// =============================================================================
// Synchronization Tests
// =============================================================================

TEST_F(OffloadEngineTest, Synchronize_WaitsForAll) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    // Start multiple async operations
    std::vector<TransferHandle> handles;
    for (int i = 0; i < 4; ++i) {
        Tensor gpu = createPatternTensor({1000, 500}, DType::Float32, Device::cuda());
        handles.push_back(engine.offload_to_cpu_async(gpu));
    }

    // Synchronize should wait for all
    engine.synchronize();

    // All should be ready
    for (auto& handle : handles) {
        EXPECT_TRUE(handle.is_ready());
    }
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_F(OffloadEngineTest, Statistics_TrackOperations) {
    if (!cuda_available) GTEST_SKIP() << "CUDA not available";

    OffloadEngine engine(default_config);

    size_t initial_count = engine.get_offload_count();

    Tensor gpu_tensor = createPatternTensor({100}, DType::Float32, Device::cuda());
    engine.offload_to_cpu(gpu_tensor);

    EXPECT_EQ(engine.get_offload_count(), initial_count + 1);
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
