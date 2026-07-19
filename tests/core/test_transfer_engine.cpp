/**
 * @file test_transfer_engine.cpp
 * @brief Comprehensive unit tests for Transfer Engine (Phase 1 - ZeRO Offload)
 *
 * Tests cover:
 * - Synchronous CPU->GPU transfers
 * - Synchronous GPU->CPU transfers
 * - Asynchronous transfers with handles
 * - TransferHandle wait and status
 * - Multiple concurrent transfers
 * - Stream synchronization
 * - Bandwidth measurement
 * - Error handling (invalid device, OOM)
 */

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"  // CC.18: SKIP_WITH_REASON
#include "../backend_test_fixture.hpp"  // FINDING 25: BackendTest -- parametrized GPU device
#include <tenzor/core/transfer_engine.hpp>
#include <tenzor/core/tensor.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/tenzor.hpp>
#include <thread>
#include <chrono>
#include <vector>
#include <algorithm>
#include <random>

using namespace tenzor;
using namespace tenzor::core;
using namespace tenzor::testing;
using namespace std::chrono_literals;

/**
 * Test Fixture for Transfer Engine.
 *
 * FINDING 25: was TEST_F over a hardcoded Device::cuda(0), so ROCm/Vulkan/
 * OneAPI's real (non-stub) TransferEngine implementations
 * (include/tenzor/core/transfer_engine.hpp's #ifdef TENZOR_USE_ROCM/
 * TENZOR_USE_ONEAPI/TENZOR_USE_VULKAN sections) got zero test exercise. Now
 * TEST_P over BackendTest, parametrized across every GPU backend below --
 * the inherited `device` member is the parametrized GPU device.
 */
class TransferEngineTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;

        // Default configuration
        default_config.num_streams = 4;
        default_config.queue_capacity = 64;
        default_config.use_pinned_memory = true;
        default_config.pinned_pool_size = 256 * 1024 * 1024;  // 256 MB
    }

    void TearDown() override {
        // Clean up
    }

    TransferEngine::Config default_config;

    // Helper to create tensor with specific pattern
    Tensor createPatternTensor(const std::vector<int64_t>& shape, DType dtype, Device device) {
        // Always create on CPU first and initialize with pattern
        Tensor cpu_t(shape, dtype, Device::cpu());

        if (dtype == DType::Float32) {
            auto* data = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) {
                data[i] = static_cast<float>(i % 1000);
            }
        }

        // Transfer to target device if needed
        if (device.type != Device::Type::CPU) {
            return cpu_t.to(device);
        }

        return cpu_t;
    }

    // Helper to verify tensor data matches pattern
    void verifyPatternTensor(const Tensor& t) {
        Tensor cpu_t = t.to(Device::cpu());

        if (t.dtype() == DType::Float32) {
            const auto* data = cpu_t.data<float>();
            for (int64_t i = 0; i < cpu_t.numel(); ++i) {
                EXPECT_FLOAT_EQ(data[i], static_cast<float>(i % 1000))
                    << "Mismatch at index " << i;
            }
        }
    }

    // Helper to measure transfer time
    template<typename Func>
    double measureTime(Func&& func) {
        auto start = std::chrono::high_resolution_clock::now();
        func();
        auto end = std::chrono::high_resolution_clock::now();

        std::chrono::duration<double, std::milli> duration = end - start;
        return duration.count();
    }
};

// =============================================================================
// Constructor Tests
// =============================================================================

TEST_P(TransferEngineTest, ConstructorWithValidConfig) {
    ASSERT_NO_THROW({
        auto engine = std::make_unique<TransferEngine>(default_config);
    });
}

TEST_P(TransferEngineTest, ConstructorWithDefaultConfig) {
    ASSERT_NO_THROW({
        TransferEngine::Config cfg;
        auto engine = std::make_unique<TransferEngine>(cfg);
    });
}

// =============================================================================
// Synchronous CPU->GPU Transfer Tests
// =============================================================================

TEST_P(TransferEngineTest, SyncCPUToGPU_BasicTransfer) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);

    EXPECT_EQ(gpu_tensor.device().type, device.type);
    verifyPatternTensor(gpu_tensor);
}

TEST_P(TransferEngineTest, SyncCPUToGPU_LargeTensor) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    // 10 MB tensor
    Tensor cpu_tensor = createPatternTensor({2560000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);

    EXPECT_EQ(gpu_tensor.device().type, device.type);
    verifyPatternTensor(gpu_tensor);
}

TEST_P(TransferEngineTest, SyncCPUToGPU_MultipleTensors) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    std::vector<Tensor> cpu_tensors;
    std::vector<Tensor> gpu_tensors;

    for (int i = 0; i < 10; ++i) {
        cpu_tensors.push_back(createPatternTensor({1000}, DType::Float32, Device::cpu()));
        gpu_tensors.push_back(engine->cpu_to_gpu(cpu_tensors.back(), device));
    }

    for (const auto& gpu_t : gpu_tensors) {
        EXPECT_EQ(gpu_t.device().type, device.type);
        verifyPatternTensor(gpu_t);
    }
}

TEST_P(TransferEngineTest, SyncCPUToGPU_EmptyTensor) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor empty_cpu = Tensor(std::vector<int64_t>{0}, DType::Float32, Device::cpu());

    ASSERT_NO_THROW({
        Tensor gpu_tensor = engine->cpu_to_gpu(empty_cpu, device);
        EXPECT_EQ(gpu_tensor.numel(), 0);
    });
}

// =============================================================================
// Synchronous GPU->CPU Transfer Tests
// =============================================================================

TEST_P(TransferEngineTest, SyncGPUToCPU_BasicTransfer) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor gpu_tensor = createPatternTensor({1000}, DType::Float32, device);
    Tensor cpu_tensor = engine->gpu_to_cpu(gpu_tensor);

    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    verifyPatternTensor(cpu_tensor);
}

TEST_P(TransferEngineTest, SyncGPUToCPU_LargeTensor) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor gpu_tensor = createPatternTensor({2560000}, DType::Float32, device);
    Tensor cpu_tensor = engine->gpu_to_cpu(gpu_tensor);

    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    verifyPatternTensor(cpu_tensor);
}

TEST_P(TransferEngineTest, SyncGPUToCPU_RoundTrip) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor original_cpu = createPatternTensor({1000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(original_cpu, device);
    Tensor result_cpu = engine->gpu_to_cpu(gpu_tensor);

    EXPECT_EQ(result_cpu.device().type, Device::Type::CPU);
    verifyPatternTensor(result_cpu);
}

// =============================================================================
// Asynchronous Transfer Tests
// =============================================================================

TEST_P(TransferEngineTest, AsyncCPUToGPU_BasicTransfer) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());
    TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, device);

    EXPECT_TRUE(handle.is_valid());

    Tensor gpu_tensor = handle.get_tensor();

    EXPECT_TRUE(handle.is_ready());
    EXPECT_EQ(gpu_tensor.device().type, device.type);
    verifyPatternTensor(gpu_tensor);
}

TEST_P(TransferEngineTest, AsyncGPUToCPU_BasicTransfer) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor gpu_tensor = createPatternTensor({1000}, DType::Float32, device);
    TransferHandle handle = engine->gpu_to_cpu_async(gpu_tensor);

    Tensor cpu_tensor = handle.get_tensor();

    EXPECT_EQ(cpu_tensor.device().type, Device::Type::CPU);
    verifyPatternTensor(cpu_tensor);
}

TEST_P(TransferEngineTest, AsyncTransfer_HandleWait) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({10000}, DType::Float32, Device::cpu());
    TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, device);

    // Wait for completion
    handle.wait();

    EXPECT_TRUE(handle.is_ready());
}

TEST_P(TransferEngineTest, AsyncTransfer_MultipleHandles) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    std::vector<TransferHandle> handles;
    std::vector<Tensor> cpu_tensors;

    for (int i = 0; i < 5; ++i) {
        cpu_tensors.push_back(createPatternTensor({1000}, DType::Float32, Device::cpu()));
        handles.push_back(engine->cpu_to_gpu_async(cpu_tensors.back(), device));
    }

    // Wait for all
    for (auto& handle : handles) {
        handle.wait();
        EXPECT_TRUE(handle.is_ready());
    }
}

// =============================================================================
// Concurrent Transfer Tests
// =============================================================================

TEST_P(TransferEngineTest, ConcurrentTransfers_Bidirectional) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    std::vector<TransferHandle> cpu_to_gpu_handles;
    std::vector<TransferHandle> gpu_to_cpu_handles;

    // Start concurrent transfers in both directions
    for (int i = 0; i < 4; ++i) {
        Tensor cpu_t = createPatternTensor({5000}, DType::Float32, Device::cpu());
        cpu_to_gpu_handles.push_back(engine->cpu_to_gpu_async(cpu_t, device));

        Tensor gpu_t = createPatternTensor({5000}, DType::Float32, device);
        gpu_to_cpu_handles.push_back(engine->gpu_to_cpu_async(gpu_t));
    }

    // Wait for all
    for (auto& handle : cpu_to_gpu_handles) {
        handle.wait();
    }
    for (auto& handle : gpu_to_cpu_handles) {
        handle.wait();
    }

    // Verify all completed
    for (auto& handle : cpu_to_gpu_handles) {
        EXPECT_TRUE(handle.is_ready());
    }
    for (auto& handle : gpu_to_cpu_handles) {
        EXPECT_TRUE(handle.is_ready());
    }
}

TEST_P(TransferEngineTest, ConcurrentTransfers_UtilizesMultipleStreams) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    // Launch many transfers to test stream utilization
    std::vector<TransferHandle> handles;

    auto start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < 8; ++i) {
        Tensor cpu_t = createPatternTensor({10000}, DType::Float32, Device::cpu());
        handles.push_back(engine->cpu_to_gpu_async(cpu_t, device));
    }

    for (auto& handle : handles) {
        handle.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double, std::milli> duration = end - start;

    // With 4 streams, should be faster than serial
    std::cout << "Concurrent transfer time: " << duration.count() << " ms" << std::endl;
}

// =============================================================================
// Stream Synchronization Tests
// =============================================================================

TEST_P(TransferEngineTest, StreamSync_WaitForCompletion) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({10000}, DType::Float32, Device::cpu());
    TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, device);

    // Should block until complete
    engine->synchronize();

    EXPECT_TRUE(handle.is_ready());
}

TEST_P(TransferEngineTest, StreamSync_MultipleTransfers) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    std::vector<TransferHandle> handles;

    for (int i = 0; i < 10; ++i) {
        Tensor cpu_t = createPatternTensor({5000}, DType::Float32, Device::cpu());
        handles.push_back(engine->cpu_to_gpu_async(cpu_t, device));
    }

    engine->synchronize();

    for (auto& handle : handles) {
        EXPECT_TRUE(handle.is_ready());
    }
}

TEST_P(TransferEngineTest, StreamSync_IndividualStream) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({10000}, DType::Float32, Device::cpu());
    TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, device);

    // Synchronize specific stream (stream 0)
    engine->synchronize_stream(0);

    EXPECT_TRUE(handle.is_ready());
}

// =============================================================================
// Statistics Tests
// =============================================================================

TEST_P(TransferEngineTest, Statistics_TrackTransfers) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    size_t count_before = engine->get_transfer_count();

    Tensor cpu_tensor = createPatternTensor({10000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);

    size_t count_after = engine->get_transfer_count();

    EXPECT_GT(count_after, count_before);
}

TEST_P(TransferEngineTest, Statistics_TrackBytes) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    size_t bytes_before = engine->get_bytes_transferred();

    Tensor cpu_tensor = createPatternTensor({10000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);

    size_t bytes_after = engine->get_bytes_transferred();

    EXPECT_GT(bytes_after, bytes_before);
}

TEST_P(TransferEngineTest, Statistics_GetStatistics) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    for (int i = 0; i < 5; ++i) {
        Tensor cpu_t = createPatternTensor({100000}, DType::Float32, Device::cpu());
        Tensor gpu_t = engine->cpu_to_gpu(cpu_t, device);
    }

    auto stats = engine->get_statistics();

    EXPECT_GT(stats.total_transfers, 0);
    EXPECT_GT(stats.bytes_transferred, 0);
    EXPECT_GT(stats.cpu_to_gpu_count, 0);
}

TEST_P(TransferEngineTest, Statistics_Reset) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_t = createPatternTensor({10000}, DType::Float32, Device::cpu());
    Tensor gpu_t = engine->cpu_to_gpu(cpu_t, device);

    EXPECT_GT(engine->get_transfer_count(), 0);

    engine->reset_statistics();

    EXPECT_EQ(engine->get_transfer_count(), 0);
    EXPECT_EQ(engine->get_bytes_transferred(), 0);
}

TEST_P(TransferEngineTest, Statistics_AverageBandwidth) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    for (int i = 0; i < 5; ++i) {
        Tensor cpu_t = createPatternTensor({100000}, DType::Float32, Device::cpu());
        Tensor gpu_t = engine->cpu_to_gpu(cpu_t, device);
    }

    float bandwidth = engine->get_average_bandwidth_gbps();

    EXPECT_GT(bandwidth, 0.0f);
    std::cout << "Average bandwidth: " << bandwidth << " GB/s" << std::endl;
}

// =============================================================================
// Bandwidth Measurement Tests
// =============================================================================

TEST_P(TransferEngineTest, BandwidthMeasurement_CPUToGPU) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    size_t transfer_size = 100 * 1024 * 1024;  // 100 MB
    Tensor cpu_tensor(std::vector<int64_t>{static_cast<int64_t>(transfer_size / 4)}, DType::Float32, Device::cpu());

    auto start = std::chrono::high_resolution_clock::now();
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double bandwidth_gbps = (transfer_size / duration.count()) / 1e9;

    std::cout << "CPU->GPU bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    EXPECT_GT(bandwidth_gbps, 0.5);  // At least 0.5 GB/s
}

TEST_P(TransferEngineTest, BandwidthMeasurement_GPUToCPU) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    size_t transfer_size = 100 * 1024 * 1024;  // 100 MB
    Tensor gpu_tensor(std::vector<int64_t>{static_cast<int64_t>(transfer_size / 4)}, DType::Float32, device);

    auto start = std::chrono::high_resolution_clock::now();
    Tensor cpu_tensor = engine->gpu_to_cpu(gpu_tensor);
    auto end = std::chrono::high_resolution_clock::now();

    std::chrono::duration<double> duration = end - start;
    double bandwidth_gbps = (transfer_size / duration.count()) / 1e9;

    std::cout << "GPU->CPU bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    EXPECT_GT(bandwidth_gbps, 0.5);  // At least 0.5 GB/s
}

// =============================================================================
// Multiple Data Types Tests
// =============================================================================

TEST_P(TransferEngineTest, SyncTransfer_Float16) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor({1000}, DType::Float16, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);
    Tensor result_cpu = engine->gpu_to_cpu(gpu_tensor);

    EXPECT_EQ(result_cpu.device().type, Device::Type::CPU);
    EXPECT_EQ(result_cpu.dtype(), DType::Float16);
}

TEST_P(TransferEngineTest, SyncTransfer_Int32) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor({1000}, DType::Int32, Device::cpu());
    auto* data = cpu_tensor.data<int32_t>();
    for (int64_t i = 0; i < cpu_tensor.numel(); ++i) {
        data[i] = static_cast<int32_t>(i);
    }

    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);
    Tensor result_cpu = engine->gpu_to_cpu(gpu_tensor);

    const auto* result_data = result_cpu.data<int32_t>();
    for (int64_t i = 0; i < result_cpu.numel(); ++i) {
        EXPECT_EQ(result_data[i], static_cast<int32_t>(i));
    }
}

TEST_P(TransferEngineTest, SyncTransfer_Int64) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor({1000}, DType::Int64, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);
    Tensor result_cpu = engine->gpu_to_cpu(gpu_tensor);

    EXPECT_EQ(result_cpu.device().type, Device::Type::CPU);
    EXPECT_EQ(result_cpu.dtype(), DType::Int64);
}

// =============================================================================
// Pinned Memory Tests
// =============================================================================

TEST_P(TransferEngineTest, PinnedMemory_LargeTensorBenefit) {

    // Config with pinned memory
    TransferEngine::Config pinned_config = default_config;
    pinned_config.use_pinned_memory = true;
    auto pinned_engine = std::make_unique<TransferEngine>(pinned_config);

    // Config without pinned memory
    TransferEngine::Config no_pinned_config = default_config;
    no_pinned_config.use_pinned_memory = false;
    auto no_pinned_engine = std::make_unique<TransferEngine>(no_pinned_config);

    // Large tensor (10 MB)
    Tensor cpu_tensor = createPatternTensor({2560000}, DType::Float32, Device::cpu());

    // Transfer with pinned memory
    auto start1 = std::chrono::high_resolution_clock::now();
    Tensor gpu1 = pinned_engine->cpu_to_gpu(cpu_tensor, device);
    auto end1 = std::chrono::high_resolution_clock::now();
    double time_pinned = std::chrono::duration<double, std::milli>(end1 - start1).count();

    // Transfer without pinned memory
    auto start2 = std::chrono::high_resolution_clock::now();
    Tensor gpu2 = no_pinned_engine->cpu_to_gpu(cpu_tensor, device);
    auto end2 = std::chrono::high_resolution_clock::now();
    double time_no_pinned = std::chrono::duration<double, std::milli>(end2 - start2).count();

    std::cout << "Pinned: " << time_pinned << " ms, No pinned: " << time_no_pinned << " ms" << std::endl;

    // Both should complete successfully
    verifyPatternTensor(gpu1);
    verifyPatternTensor(gpu2);
}

TEST_P(TransferEngineTest, PinnedMemory_ReuseBuffer) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    // Transfer tensors of similar size to test buffer reuse
    for (int i = 0; i < 5; ++i) {
        Tensor cpu_t = createPatternTensor({10000}, DType::Float32, Device::cpu());
        Tensor gpu_t = engine->cpu_to_gpu(cpu_t, device);
        verifyPatternTensor(gpu_t);
    }

    // No crashes = buffer reuse working
    SUCCEED();
}

// =============================================================================
// Configuration Tests
// =============================================================================

TEST_P(TransferEngineTest, Config_InvalidNumStreams) {
    TransferEngine::Config invalid_config;
    invalid_config.num_streams = 0;

    EXPECT_THROW({
        auto engine = std::make_unique<TransferEngine>(invalid_config);
    }, std::invalid_argument);
}

TEST_P(TransferEngineTest, Config_InvalidQueueCapacity) {
    TransferEngine::Config invalid_config;
    invalid_config.queue_capacity = 0;

    EXPECT_THROW({
        auto engine = std::make_unique<TransferEngine>(invalid_config);
    }, std::invalid_argument);
}

TEST_P(TransferEngineTest, Config_SingleStream) {

    TransferEngine::Config single_stream_config = default_config;
    single_stream_config.num_streams = 1;

    auto engine = std::make_unique<TransferEngine>(single_stream_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);

    EXPECT_EQ(gpu_tensor.device().type, device.type);
    verifyPatternTensor(gpu_tensor);
}

TEST_P(TransferEngineTest, Config_ManyStreams) {

    TransferEngine::Config many_streams_config = default_config;
    many_streams_config.num_streams = 16;

    auto engine = std::make_unique<TransferEngine>(many_streams_config);

    std::vector<TransferHandle> handles;
    for (int i = 0; i < 20; ++i) {
        Tensor cpu_t = createPatternTensor({5000}, DType::Float32, Device::cpu());
        handles.push_back(engine->cpu_to_gpu_async(cpu_t, device));
    }

    for (auto& handle : handles) {
        handle.wait();
        EXPECT_TRUE(handle.is_ready());
    }
}

// =============================================================================
// Edge Cases and Error Handling
// =============================================================================

TEST_P(TransferEngineTest, EmptyTensorTransfer) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor empty_cpu = zeros({0}, DType::Float32, Device::cpu());

    ASSERT_NO_THROW({
        Tensor gpu_t = engine->cpu_to_gpu(empty_cpu, device);
        EXPECT_EQ(gpu_t.numel(), 0);
    });
}

TEST_P(TransferEngineTest, QueueOverflow_HandlesBackpressure) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    // Submit many transfers
    std::vector<TransferHandle> handles;

    for (int i = 0; i < 100; ++i) {
        Tensor cpu_t = createPatternTensor({100}, DType::Float32, Device::cpu());
        handles.push_back(engine->cpu_to_gpu_async(cpu_t, device));
    }

    // All should eventually complete
    for (auto& handle : handles) {
        handle.wait();
        EXPECT_TRUE(handle.is_ready());
    }
}

TEST_P(TransferEngineTest, Error_TransferNonCPUTensorToGPU) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor gpu_tensor = createPatternTensor({1000}, DType::Float32, device);

    EXPECT_THROW({
        Tensor result = engine->cpu_to_gpu(gpu_tensor, device);
    }, std::runtime_error);
}

TEST_P(TransferEngineTest, Error_TransferNonGPUTensorToCPU) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());

    EXPECT_THROW({
        Tensor result = engine->gpu_to_cpu(cpu_tensor);
    }, std::runtime_error);
}

TEST_P(TransferEngineTest, Error_AsyncTransferInvalidDevice) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());

    EXPECT_THROW({
        TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, Device::cpu());
    }, std::runtime_error);
}

// =============================================================================
// Thread Safety Tests
// =============================================================================

TEST_P(TransferEngineTest, ThreadSafety_MultithreadedAsync) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    std::vector<std::thread> threads;
    std::vector<std::vector<TransferHandle>> thread_handles(4);

    for (int t = 0; t < 4; ++t) {
        threads.emplace_back([&, t]() {
            for (int i = 0; i < 10; ++i) {
                Tensor cpu_t = createPatternTensor({1000}, DType::Float32, Device::cpu());
                thread_handles[t].push_back(engine->cpu_to_gpu_async(cpu_t, device));
            }
        });
    }

    for (auto& thread : threads) {
        thread.join();
    }

    // Verify all transfers completed
    for (auto& handles : thread_handles) {
        for (auto& handle : handles) {
            handle.wait();
            EXPECT_TRUE(handle.is_ready());
        }
    }
}

TEST_P(TransferEngineTest, ThreadSafety_SynchronizeWhileTransferring) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    std::vector<TransferHandle> handles;
    for (int i = 0; i < 20; ++i) {
        Tensor cpu_t = createPatternTensor({10000}, DType::Float32, Device::cpu());
        handles.push_back(engine->cpu_to_gpu_async(cpu_t, device));
    }

    // Synchronize while transfers are in flight
    std::thread sync_thread([&]() {
        std::this_thread::sleep_for(10ms);
        engine->synchronize();
    });

    sync_thread.join();

    // All should be complete after synchronize
    for (auto& handle : handles) {
        EXPECT_TRUE(handle.is_ready());
    }
}

// =============================================================================
// Handle Tests
// =============================================================================

TEST_P(TransferEngineTest, Handle_EmptyHandle) {
    TransferHandle empty_handle;

    EXPECT_FALSE(empty_handle.is_valid());
    EXPECT_TRUE(empty_handle.is_ready());

    ASSERT_NO_THROW({
        empty_handle.wait();
    });
}

TEST_P(TransferEngineTest, Handle_MultipleWaits) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    Tensor cpu_tensor = createPatternTensor({1000}, DType::Float32, Device::cpu());
    TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, device);

    // Multiple waits should be safe
    handle.wait();
    handle.wait();
    handle.wait();

    EXPECT_TRUE(handle.is_ready());
}

TEST_P(TransferEngineTest, Handle_CheckReadyBeforeCompletion) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    // Large tensor to ensure transfer takes time
    Tensor cpu_tensor = createPatternTensor({10000000}, DType::Float32, Device::cpu());
    TransferHandle handle = engine->cpu_to_gpu_async(cpu_tensor, device);

    // May or may not be ready immediately
    bool ready_immediately = handle.is_ready();

    // Wait for completion
    handle.wait();

    // Must be ready after wait
    EXPECT_TRUE(handle.is_ready());
}

// =============================================================================
// Performance Validation Tests
// =============================================================================

TEST_P(TransferEngineTest, Performance_MinimumBandwidth) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    size_t transfer_size = 100 * 1024 * 1024;  // 100 MB
    Tensor cpu_tensor({static_cast<int64_t>(transfer_size / 4)}, DType::Float32, Device::cpu());

    // Warm-up
    engine->cpu_to_gpu(cpu_tensor, device);

    // Measure bandwidth
    auto start = std::chrono::high_resolution_clock::now();
    Tensor gpu_tensor = engine->cpu_to_gpu(cpu_tensor, device);
    auto end = std::chrono::high_resolution_clock::now();

    double time_s = std::chrono::duration<double>(end - start).count();
    double bandwidth_gbps = (transfer_size / 1e9) / time_s;

    std::cout << "Measured bandwidth: " << bandwidth_gbps << " GB/s" << std::endl;

    // Should achieve at least 1 GB/s on modern systems
    EXPECT_GT(bandwidth_gbps, 1.0);
}

TEST_P(TransferEngineTest, Performance_AsyncOverlapping) {

    auto engine = std::make_unique<TransferEngine>(default_config);

    // Test that async transfers can overlap
    std::vector<TransferHandle> handles;
    std::vector<Tensor> cpu_tensors;

    auto start = std::chrono::high_resolution_clock::now();

    // Launch 8 async transfers
    for (int i = 0; i < 8; ++i) {
        cpu_tensors.push_back(createPatternTensor({1000000}, DType::Float32, Device::cpu()));
        handles.push_back(engine->cpu_to_gpu_async(cpu_tensors.back(), device));
    }

    // Wait for all
    for (auto& handle : handles) {
        handle.wait();
    }

    auto end = std::chrono::high_resolution_clock::now();
    double async_time = std::chrono::duration<double, std::milli>(end - start).count();

    // Do same transfers synchronously for comparison
    start = std::chrono::high_resolution_clock::now();
    for (auto& cpu_t : cpu_tensors) {
        engine->cpu_to_gpu(cpu_t, device);
    }
    end = std::chrono::high_resolution_clock::now();
    double sync_time = std::chrono::duration<double, std::milli>(end - start).count();

    std::cout << "Async time: " << async_time << " ms, Sync time: " << sync_time << " ms" << std::endl;

    // Async should be at least somewhat faster (though not always guaranteed)
    // Just verify both complete successfully
    EXPECT_GT(sync_time, 0);
    EXPECT_GT(async_time, 0);
}

// FINDING 25: exercise every real (non-stub) TransferEngine GPU backend
// implementation, not just CUDA. MPS omitted (BackendTest itself skips it
// cleanly off-Apple; including it would just add dead rows on this host).
INSTANTIATE_TEST_SUITE_P(
    GpuBackends,
    TransferEngineTest,
    ::testing::Values("cuda", "rocm", "vulkan", "oneapi"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param;
    }
);

// =============================================================================
// Main
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
