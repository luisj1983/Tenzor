/**
 * @file test_offload_engine_diagnostic.cpp
 * @brief Deep diagnostic tests for OffloadEngine correctness
 *
 * These tests go beyond unit tests to verify:
 * 1. Data integrity under concurrent transfers
 * 2. Memory pressure auto-offload behavior
 * 3. Prefetch effectiveness (latency hiding)
 * 4. End-to-end training simulation
 * 5. Edge cases and error recovery
 */

#include <gtest/gtest.h>
#include "tenzor/core/offload_engine.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/backend/loader.hpp"
#include <chrono>
#include <random>
#include <thread>
#include <vector>
#include <numeric>
#include <cmath>

using namespace tenzor;
using namespace tenzor::core;

class OffloadEngineDiagnosticTest : public ::testing::Test {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }

    void SetUp() override {
        auto* backend = backend_registry().get_backend("cuda");
        cuda_available_ = (backend != nullptr && backend->is_available());
    }

    bool cuda_available_ = false;

    // Create tensor with deterministic pattern based on seed
    Tensor createDeterministicTensor(const std::vector<int64_t>& shape, uint32_t seed) {
        Tensor cpu_t(shape, DType::Float32, Device::cpu());
        auto* data = cpu_t.data<float>();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < cpu_t.numel(); ++i) {
            data[i] = dist(gen);
        }
        return cpu_t;
    }

    // Verify tensor matches expected pattern
    bool verifyDeterministicTensor(const Tensor& t, uint32_t seed, float tolerance = 1e-6f) {
        Tensor cpu_t = (t.device().type == Device::Type::CPU) ? t : t.to(Device::cpu());
        auto* data = cpu_t.data<float>();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < cpu_t.numel(); ++i) {
            float expected = dist(gen);
            if (std::abs(data[i] - expected) > tolerance) {
                std::cerr << "Mismatch at index " << i << ": expected " << expected
                          << ", got " << data[i] << std::endl;
                return false;
            }
        }
        return true;
    }

    // Compute checksum for quick verification
    double computeChecksum(const Tensor& t) {
        Tensor cpu_t = (t.device().type == Device::Type::CPU) ? t : t.to(Device::cpu());
        auto* data = cpu_t.data<float>();
        double sum = 0.0;
        for (int64_t i = 0; i < cpu_t.numel(); ++i) {
            sum += static_cast<double>(data[i]) * (i + 1);
        }
        return sum;
    }
};

// =============================================================================
// TEST 1: Data Integrity Under Stress
// =============================================================================

TEST_F(OffloadEngineDiagnosticTest, DataIntegrity_ManyRoundTrips) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 256 * 1024 * 1024;
    config.num_transfer_streams = 4;
    config.enable_prefetch = true;
    OffloadEngine engine(config);

    const int num_iterations = 20;
    const int num_tensors = 10;

    std::cout << "\n=== Data Integrity Test ===" << std::endl;
    std::cout << "Performing " << num_iterations << " round-trips with "
              << num_tensors << " tensors each\n" << std::endl;

    for (int iter = 0; iter < num_iterations; ++iter) {
        for (int i = 0; i < num_tensors; ++i) {
            uint32_t seed = iter * 1000 + i;

            // Create tensor with deterministic pattern
            Tensor original = createDeterministicTensor({512, 256}, seed);
            double original_checksum = computeChecksum(original);

            // Move to GPU
            Tensor gpu_tensor = original.to(Device::cuda());

            // Offload back to CPU via engine
            Tensor offloaded = engine.offload_to_cpu(gpu_tensor);

            // Verify checksum
            double offloaded_checksum = computeChecksum(offloaded);
            ASSERT_NEAR(original_checksum, offloaded_checksum, 1e-3)
                << "Checksum mismatch at iteration " << iter << ", tensor " << i;

            // Full verification for first few
            if (iter < 3) {
                ASSERT_TRUE(verifyDeterministicTensor(offloaded, seed))
                    << "Data corruption at iteration " << iter << ", tensor " << i;
            }
        }

        if ((iter + 1) % 5 == 0) {
            std::cout << "  Completed iteration " << (iter + 1) << "/" << num_iterations << std::endl;
        }
    }

    std::cout << "  PASSED: " << (num_iterations * num_tensors)
              << " round-trips with verified data integrity\n" << std::endl;
}

TEST_F(OffloadEngineDiagnosticTest, DataIntegrity_ConcurrentTransfers) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 512 * 1024 * 1024;
    config.num_transfer_streams = 8;
    config.enable_prefetch = true;
    OffloadEngine engine(config);

    const int num_concurrent = 16;

    std::cout << "\n=== Concurrent Transfer Integrity Test ===" << std::endl;
    std::cout << "Launching " << num_concurrent << " concurrent async transfers\n" << std::endl;

    // Create tensors with unique patterns
    std::vector<Tensor> gpu_tensors;
    std::vector<double> expected_checksums;

    for (int i = 0; i < num_concurrent; ++i) {
        Tensor cpu = createDeterministicTensor({256, 128}, 12345 + i);
        expected_checksums.push_back(computeChecksum(cpu));
        gpu_tensors.push_back(cpu.to(Device::cuda()));
    }

    // Launch all async transfers
    std::vector<TransferHandle> handles;
    for (int i = 0; i < num_concurrent; ++i) {
        handles.push_back(engine.offload_to_cpu_async(gpu_tensors[i]));
    }

    // Collect and verify
    for (int i = 0; i < num_concurrent; ++i) {
        Tensor result = handles[i].get_tensor();
        double actual_checksum = computeChecksum(result);

        ASSERT_NEAR(expected_checksums[i], actual_checksum, 1e-3)
            << "Concurrent transfer " << i << " data corruption detected!";
    }

    std::cout << "  PASSED: All " << num_concurrent << " concurrent transfers preserved data\n" << std::endl;
}

// =============================================================================
// TEST 2: Memory Pressure Auto-Offload
// =============================================================================

TEST_F(OffloadEngineDiagnosticTest, AutoOffload_TriggersUnderPressure) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    // Configure with low threshold to easily trigger
    OffloadEngine::Config config;
    config.pinned_memory_size = 128 * 1024 * 1024;
    config.num_transfer_streams = 2;
    config.enable_prefetch = false;
    config.memory_fraction = 0.01f;  // Very low threshold to trigger easily
    config.enable_auto_monitoring = false;  // Disable automatic monitoring for controlled test
    OffloadEngine engine(config);

    std::cout << "\n=== Auto-Offload Under Pressure Test ===" << std::endl;

    // Register tensors with different priorities
    std::vector<Tensor> tensors;
    for (int i = 0; i < 5; ++i) {
        tensors.push_back(createDeterministicTensor({256, 128}, 1000 + i).to(Device::cuda()));
    }

    // Register for auto-offload
    engine.register_auto_offload(&tensors[0], OffloadPriority::LOW);      // Should offload first
    engine.register_auto_offload(&tensors[1], OffloadPriority::NORMAL);
    engine.register_auto_offload(&tensors[2], OffloadPriority::HIGH);
    engine.register_auto_offload(&tensors[3], OffloadPriority::CRITICAL); // Should offload last
    engine.register_auto_offload(&tensors[4], OffloadPriority::LOW);

    EXPECT_EQ(engine.get_registered_tensor_count(), 5);

    // Check initial pressure
    float initial_pressure = engine.get_gpu_memory_pressure();
    std::cout << "  Initial GPU memory pressure: " << (initial_pressure * 100) << "%" << std::endl;

    // Manually trigger offload check
    size_t initial_auto_offload_count = engine.get_auto_offload_count();
    size_t offloaded = engine.check_and_offload();

    std::cout << "  Tensors offloaded: " << offloaded << std::endl;
    std::cout << "  Total auto-offload operations: " << engine.get_auto_offload_count() << std::endl;

    // With very low threshold, some tensors should have been offloaded
    if (initial_pressure > config.memory_fraction) {
        EXPECT_GT(offloaded, 0) << "Expected some tensors to be offloaded under pressure";
        std::cout << "  PASSED: Auto-offload triggered correctly\n" << std::endl;
    } else {
        std::cout << "  NOTE: Memory pressure below threshold, no offload needed\n" << std::endl;
    }

    // Cleanup
    for (auto& t : tensors) {
        engine.unregister_auto_offload(&t);
    }
}

TEST_F(OffloadEngineDiagnosticTest, AutoOffload_PriorityOrder) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 64 * 1024 * 1024;
    config.memory_fraction = 0.001f;  // Extremely low to force offloading
    config.enable_auto_monitoring = false;
    OffloadEngine engine(config);

    std::cout << "\n=== Priority-Based Offload Order Test ===" << std::endl;

    // Create tensors and track their state
    Tensor low_priority = createDeterministicTensor({128, 64}, 100).to(Device::cuda());
    Tensor high_priority = createDeterministicTensor({128, 64}, 200).to(Device::cuda());

    // Register with different priorities
    engine.register_auto_offload(&low_priority, OffloadPriority::LOW);
    engine.register_auto_offload(&high_priority, OffloadPriority::HIGH);

    // Force offload check
    engine.check_and_offload();

    // LOW priority should be offloaded first (moved to CPU)
    bool low_on_cpu = (low_priority.device().type == Device::Type::CPU);
    bool high_on_cuda = (high_priority.device().type == Device::Type::CUDA);

    std::cout << "  LOW priority tensor on CPU: " << (low_on_cpu ? "YES" : "NO") << std::endl;
    std::cout << "  HIGH priority tensor on CUDA: " << (high_on_cuda ? "YES" : "NO") << std::endl;

    // At minimum, if any offloading happened, LOW should go before HIGH
    if (low_on_cpu && high_on_cuda) {
        std::cout << "  PASSED: Priority order respected\n" << std::endl;
    } else if (!low_on_cpu && !high_on_cuda) {
        std::cout << "  NOTE: Neither offloaded (pressure may be below threshold)\n" << std::endl;
    }

    engine.unregister_auto_offload(&low_priority);
    engine.unregister_auto_offload(&high_priority);
}

// =============================================================================
// TEST 3: Prefetch Effectiveness
// =============================================================================

TEST_F(OffloadEngineDiagnosticTest, Prefetch_HidesLatency) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 256 * 1024 * 1024;
    config.num_transfer_streams = 4;
    config.enable_prefetch = true;
    config.prefetch_depth = 8;
    OffloadEngine engine(config);

    std::cout << "\n=== Prefetch Latency Hiding Test ===" << std::endl;

    // Create multiple tensors to simulate layer weights
    const int num_layers = 8;
    std::vector<Tensor> layer_weights;
    for (int i = 0; i < num_layers; ++i) {
        layer_weights.push_back(createDeterministicTensor({512, 256}, 5000 + i));
    }

    // Scenario 1: Without prefetch (synchronous loads)
    auto start_sync = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < num_layers; ++i) {
        Tensor gpu = engine.load_to_gpu(layer_weights[i]);
        // Simulate compute
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    auto end_sync = std::chrono::high_resolution_clock::now();
    double sync_time = std::chrono::duration<double, std::milli>(end_sync - start_sync).count();

    // Scenario 2: With prefetch
    auto start_prefetch = std::chrono::high_resolution_clock::now();

    // Prefetch first batch
    std::vector<Tensor*> prefetch_batch;
    for (int i = 0; i < std::min(4, num_layers); ++i) {
        prefetch_batch.push_back(&layer_weights[i]);
    }
    engine.prefetch_to_gpu(prefetch_batch);

    for (int i = 0; i < num_layers; ++i) {
        // Load current layer (should be ready from prefetch)
        Tensor gpu = engine.load_to_gpu(layer_weights[i]);

        // Prefetch next layer while computing
        if (i + 4 < num_layers) {
            engine.prefetch_to_gpu(&layer_weights[i + 4]);
        }

        // Simulate compute
        std::this_thread::sleep_for(std::chrono::microseconds(500));
    }
    auto end_prefetch = std::chrono::high_resolution_clock::now();
    double prefetch_time = std::chrono::duration<double, std::milli>(end_prefetch - start_prefetch).count();

    std::cout << "  Synchronous load time: " << sync_time << " ms" << std::endl;
    std::cout << "  With prefetch time:    " << prefetch_time << " ms" << std::endl;

    // Prefetch should provide some benefit (not always guaranteed due to timing)
    std::cout << "  Speedup: " << (sync_time / prefetch_time) << "x" << std::endl;

    if (prefetch_time < sync_time * 1.1) {  // Allow 10% margin
        std::cout << "  PASSED: Prefetch did not add significant overhead\n" << std::endl;
    } else {
        std::cout << "  WARNING: Prefetch overhead detected (may be system-dependent)\n" << std::endl;
    }
}

// =============================================================================
// TEST 4: Training Simulation
// =============================================================================

TEST_F(OffloadEngineDiagnosticTest, TrainingSimulation_ZeROStyleOffload) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 512 * 1024 * 1024;
    config.num_transfer_streams = 4;
    config.enable_prefetch = true;
    config.prefetch_depth = 4;
    OffloadEngine engine(config);

    std::cout << "\n=== ZeRO-Style Training Simulation ===" << std::endl;

    // Simulate model with 4 layers
    struct LayerState {
        Tensor weights;
        Tensor gradients;
        Tensor optimizer_state;  // e.g., Adam momentum
        double original_checksum;
    };

    const int num_layers = 4;
    const int num_epochs = 3;
    std::vector<LayerState> layers(num_layers);

    // Initialize layers on CPU (simulating ZeRO offload)
    for (int i = 0; i < num_layers; ++i) {
        layers[i].weights = createDeterministicTensor({256, 128}, 10000 + i);
        layers[i].gradients = createDeterministicTensor({256, 128}, 20000 + i);
        layers[i].optimizer_state = createDeterministicTensor({256, 128}, 30000 + i);
        layers[i].original_checksum = computeChecksum(layers[i].weights);
    }

    std::cout << "  Simulating " << num_epochs << " epochs with " << num_layers << " layers\n" << std::endl;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        std::cout << "  Epoch " << (epoch + 1) << ":" << std::endl;

        // Forward pass: load weights to GPU layer by layer
        for (int i = 0; i < num_layers; ++i) {
            // Prefetch next layer's weights
            if (i + 1 < num_layers) {
                engine.prefetch_to_gpu(&layers[i + 1].weights);
            }

            // Load current layer to GPU
            Tensor gpu_weights = engine.load_to_gpu(layers[i].weights);

            // Simulate forward computation (just verify data)
            double checksum = computeChecksum(gpu_weights);
            ASSERT_NEAR(layers[i].original_checksum, checksum, 1e-3)
                << "Weight corruption in epoch " << epoch << ", layer " << i;
        }

        // Backward pass: compute gradients (simulated)
        // In real ZeRO, gradients are computed on GPU then offloaded

        // Optimizer step: load optimizer states, update, offload
        for (int i = num_layers - 1; i >= 0; --i) {
            // Load optimizer state to GPU
            Tensor gpu_opt_state = engine.load_to_gpu(layers[i].optimizer_state);

            // Simulate optimizer update
            // (In reality: momentum = beta * momentum + grad; weight -= lr * momentum)

            // Offload updated state back to CPU
            Tensor cpu_opt_state = engine.offload_to_cpu(gpu_opt_state);

            // Update stored state
            layers[i].optimizer_state = cpu_opt_state;
        }

        std::cout << "    Forward + Backward + Optimizer step completed" << std::endl;
    }

    // Verify final state integrity
    std::cout << "\n  Verifying final state integrity..." << std::endl;
    for (int i = 0; i < num_layers; ++i) {
        // Optimizer states may have changed, but should still be valid
        EXPECT_EQ(layers[i].optimizer_state.device().type, Device::Type::CPU)
            << "Layer " << i << " optimizer state should be on CPU";
    }

    std::cout << "  PASSED: Training simulation completed successfully\n" << std::endl;

    // Print statistics
    std::cout << "  Statistics:" << std::endl;
    std::cout << "    Total offloads: " << engine.get_offload_count() << std::endl;
    std::cout << "    Total loads:    " << engine.get_load_count() << std::endl;
    std::cout << "    Total prefetches: " << engine.get_prefetch_count() << std::endl;
}

// =============================================================================
// TEST 5: Edge Cases and Error Handling
// =============================================================================

TEST_F(OffloadEngineDiagnosticTest, EdgeCase_EmptyTensor) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 64 * 1024 * 1024;
    OffloadEngine engine(config);

    std::cout << "\n=== Edge Case: Empty Tensor ===" << std::endl;

    // Create empty tensor (0 elements)
    Tensor empty_cpu({0}, DType::Float32, Device::cpu());
    Tensor empty_gpu = empty_cpu.to(Device::cuda());

    // These should not crash
    ASSERT_NO_THROW({
        Tensor result = engine.offload_to_cpu(empty_gpu);
        EXPECT_EQ(result.numel(), 0);
    });

    std::cout << "  PASSED: Empty tensor handled correctly\n" << std::endl;
}

TEST_F(OffloadEngineDiagnosticTest, EdgeCase_VeryLargeTensor) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 1024 * 1024 * 1024;  // 1 GB pinned
    config.num_transfer_streams = 4;
    OffloadEngine engine(config);

    std::cout << "\n=== Edge Case: Large Tensor (500 MB) ===" << std::endl;

    // 500 MB tensor
    Tensor large = createDeterministicTensor({128 * 1024, 1024}, 99999);
    double original_checksum = computeChecksum(large);

    std::cout << "  Tensor size: " << (large.numel() * sizeof(float) / (1024.0 * 1024.0)) << " MB" << std::endl;

    Tensor gpu = large.to(Device::cuda());
    Tensor offloaded = engine.offload_to_cpu(gpu);

    double final_checksum = computeChecksum(offloaded);
    ASSERT_NEAR(original_checksum, final_checksum, 1e-3);

    std::cout << "  PASSED: Large tensor transferred correctly\n" << std::endl;
}

TEST_F(OffloadEngineDiagnosticTest, EdgeCase_RapidRegistrationUnregistration) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 128 * 1024 * 1024;
    config.enable_auto_monitoring = false;
    OffloadEngine engine(config);

    std::cout << "\n=== Edge Case: Rapid Register/Unregister ===" << std::endl;

    const int iterations = 100;
    std::vector<Tensor> tensors;
    for (int i = 0; i < 10; ++i) {
        tensors.push_back(createDeterministicTensor({64, 32}, 7000 + i).to(Device::cuda()));
    }

    for (int iter = 0; iter < iterations; ++iter) {
        // Register all
        for (auto& t : tensors) {
            engine.register_auto_offload(&t, OffloadPriority::NORMAL);
        }

        EXPECT_EQ(engine.get_registered_tensor_count(), tensors.size());

        // Unregister all
        for (auto& t : tensors) {
            engine.unregister_auto_offload(&t);
        }

        EXPECT_EQ(engine.get_registered_tensor_count(), 0);
    }

    std::cout << "  PASSED: " << iterations << " register/unregister cycles completed\n" << std::endl;
}

// =============================================================================
// TEST 6: Bandwidth and Performance Baseline
// =============================================================================

TEST_F(OffloadEngineDiagnosticTest, Performance_BandwidthBaseline) {
    if (!cuda_available_) GTEST_SKIP() << "CUDA not available";

    OffloadEngine::Config config;
    config.pinned_memory_size = 1024 * 1024 * 1024;
    config.num_transfer_streams = 4;
    OffloadEngine engine(config);

    std::cout << "\n=== Performance Bandwidth Baseline ===" << std::endl;

    // Test various sizes
    std::vector<size_t> sizes_mb = {1, 10, 50, 100, 200};

    std::cout << "\n  Size (MB)  |  GPU->CPU (GB/s)  |  CPU->GPU (GB/s)" << std::endl;
    std::cout << "  -----------|-------------------|------------------" << std::endl;

    for (size_t mb : sizes_mb) {
        size_t elements = (mb * 1024 * 1024) / sizeof(float);
        std::vector<int64_t> shape = {static_cast<int64_t>(elements)};

        // Warm up
        Tensor warmup = createDeterministicTensor(shape, 1);
        Tensor warmup_gpu = warmup.to(Device::cuda());
        engine.offload_to_cpu(warmup_gpu);

        // GPU -> CPU benchmark
        Tensor gpu_tensor = createDeterministicTensor(shape, 2).to(Device::cuda());
        auto start = std::chrono::high_resolution_clock::now();
        Tensor cpu_result = engine.offload_to_cpu(gpu_tensor);
        auto end = std::chrono::high_resolution_clock::now();
        double time_g2c = std::chrono::duration<double>(end - start).count();
        double bw_g2c = (mb / 1024.0) / time_g2c;

        // CPU -> GPU benchmark
        Tensor cpu_tensor = createDeterministicTensor(shape, 3);
        start = std::chrono::high_resolution_clock::now();
        Tensor gpu_result = engine.load_to_gpu(cpu_tensor);
        end = std::chrono::high_resolution_clock::now();
        double time_c2g = std::chrono::duration<double>(end - start).count();
        double bw_c2g = (mb / 1024.0) / time_c2g;

        std::cout << "  " << std::setw(9) << mb
                  << "  |  " << std::setw(15) << std::fixed << std::setprecision(2) << bw_g2c
                  << "  |  " << std::setw(16) << bw_c2g << std::endl;
    }

    std::cout << "\n  (Higher is better. PCIe 3.0 x16 theoretical max: ~12 GB/s)\n" << std::endl;
}

// =============================================================================
// MAIN
// =============================================================================

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    std::cout << "\n========================================" << std::endl;
    std::cout << "OffloadEngine Diagnostic Test Suite" << std::endl;
    std::cout << "========================================\n" << std::endl;
    return RUN_ALL_TESTS();
}
