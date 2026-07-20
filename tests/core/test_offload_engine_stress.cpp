/**
 * @file test_offload_engine_stress.cpp
 * @brief Stress test for OffloadEngine with data larger than GPU memory
 *
 * Tests the engine's ability to handle workloads that exceed GPU memory,
 * simulating real ZeRO-style training where model parameters must be
 * offloaded and cycled through GPU memory.
 *
 * FINDING 25: was TEST_F over a hardcoded Device::cuda() / "cuda" backend
 * gate, so OffloadEngine's real ROCm/Vulkan/OneAPI TransferEngine-backed
 * paths never got stress-tested at all. Now TEST_P over BackendTest,
 * parametrized across every GPU backend below -- the inherited `device`
 * member is the parametrized GPU device. load_to_gpu/load_to_gpu_async calls
 * use the explicit-device overload so they actually target `device` instead
 * of OffloadEngine's own default_gpu_device_ (which prefers CUDA on a
 * combined-backend host regardless of which backend is under test -- see the
 * same note in test_offload_engine.cpp). prefetch_to_gpu has no
 * explicit-device overload, so (matching that same file's accepted
 * precedent) it always targets default_gpu_device_ rather than `device`.
 */

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"  // FINDING 25: BackendTest -- parametrized GPU device
#include "tenzor/core/offload_engine.hpp"
#include "tenzor/core/tensor.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/backend/loader.hpp"
#include <chrono>
#include <random>
#include <vector>
#include <iomanip>
#include <numeric>

using namespace tenzor;
using namespace tenzor::core;
using namespace tenzor::testing;

class OffloadEngineStressTest : public BackendTest {
protected:
    void SetUp() override {
        BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }

    // Create tensor with deterministic pattern
    Tensor createTensor(const std::vector<int64_t>& shape, uint32_t seed) {
        Tensor cpu_t(shape, DType::Float32, Device::cpu());
        auto* data = cpu_t.data<float>();
        std::mt19937 gen(seed);
        std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
        for (int64_t i = 0; i < cpu_t.numel(); ++i) {
            data[i] = dist(gen);
        }
        return cpu_t;
    }

    // Compute checksum for verification
    double computeChecksum(const Tensor& t) {
        Tensor cpu_t = (t.device().type == Device::Type::CPU) ? t : t.to(Device::cpu());
        auto* data = cpu_t.data<float>();
        double sum = 0.0;
        for (int64_t i = 0; i < cpu_t.numel(); ++i) {
            sum += static_cast<double>(data[i]) * ((i % 10000) + 1);
        }
        return sum;
    }

    // Get tensor size in MB
    double getSizeMB(const Tensor& t) {
        return (t.numel() * sizeof(float)) / (1024.0 * 1024.0);
    }
};

// =============================================================================
// STRESS TEST: Process 2x GPU Memory Worth of Data
// =============================================================================

TEST_P(OffloadEngineStressTest, ProcessDataLargerThanGPU_Sequential) {

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "STRESS TEST: Sequential Processing > GPU Memory" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Configuration
    const size_t GPU_MEMORY_MB = 6 * 1024;  // 6 GB
    const size_t TARGET_DATA_MB = 10 * 1024; // 10 GB total
    const size_t CHUNK_SIZE_MB = 512;        // 512 MB chunks
    const int NUM_CHUNKS = TARGET_DATA_MB / CHUNK_SIZE_MB;

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  GPU Memory:     " << (GPU_MEMORY_MB / 1024) << " GB" << std::endl;
    std::cout << "  Target Data:    " << (TARGET_DATA_MB / 1024) << " GB" << std::endl;
    std::cout << "  Chunk Size:     " << CHUNK_SIZE_MB << " MB" << std::endl;
    std::cout << "  Num Chunks:     " << NUM_CHUNKS << std::endl;

    OffloadEngine::Config config;
    config.pinned_memory_size = 1024ULL * 1024 * 1024;  // 1 GB pinned
    config.num_transfer_streams = 4;
    config.enable_prefetch = true;
    config.prefetch_depth = 4;
    OffloadEngine engine(config);

    // Create chunks on CPU (simulating model parameters stored on CPU)
    std::cout << "\nCreating " << NUM_CHUNKS << " chunks on CPU..." << std::endl;

    std::vector<Tensor> cpu_chunks;
    std::vector<double> expected_checksums;

    size_t elements_per_chunk = (CHUNK_SIZE_MB * 1024 * 1024) / sizeof(float);
    std::vector<int64_t> chunk_shape = {static_cast<int64_t>(elements_per_chunk)};

    auto create_start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cpu_chunks.push_back(createTensor(chunk_shape, 1000 + i));
        expected_checksums.push_back(computeChecksum(cpu_chunks.back()));

        if ((i + 1) % 5 == 0) {
            std::cout << "  Created chunk " << (i + 1) << "/" << NUM_CHUNKS << std::endl;
        }
    }
    auto create_end = std::chrono::high_resolution_clock::now();
    double create_time = std::chrono::duration<double>(create_end - create_start).count();
    std::cout << "  Creation time: " << std::fixed << std::setprecision(2) << create_time << " s" << std::endl;

    // Process each chunk: load to GPU, "compute", offload back
    std::cout << "\nProcessing chunks through GPU..." << std::endl;

    size_t total_bytes_transferred = 0;
    int errors = 0;

    auto process_start = std::chrono::high_resolution_clock::now();

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        // Prefetch next chunk while processing current
        if (i + 1 < NUM_CHUNKS) {
            engine.prefetch_to_gpu(&cpu_chunks[i + 1]);
        }

        // Load to GPU
        Tensor gpu_chunk = engine.load_to_gpu(cpu_chunks[i], device);
        total_bytes_transferred += gpu_chunk.numel() * sizeof(float);

        // Simulate computation (in real use, would do forward/backward pass)
        // Just verify data integrity here

        // Offload back to CPU
        Tensor result = engine.offload_to_cpu(gpu_chunk);
        total_bytes_transferred += result.numel() * sizeof(float);

        // Verify checksum
        double result_checksum = computeChecksum(result);
        if (std::abs(result_checksum - expected_checksums[i]) > 1e-3) {
            std::cerr << "  ERROR: Chunk " << i << " checksum mismatch!" << std::endl;
            errors++;
        }

        // Update CPU chunk with result
        cpu_chunks[i] = result;

        if ((i + 1) % 5 == 0) {
            std::cout << "  Processed chunk " << (i + 1) << "/" << NUM_CHUNKS << std::endl;
        }
    }

    auto process_end = std::chrono::high_resolution_clock::now();
    double process_time = std::chrono::duration<double>(process_end - process_start).count();

    // Results
    std::cout << "\n" << std::string(40, '-') << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Total data processed: " << std::fixed << std::setprecision(2)
              << (total_bytes_transferred / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  Processing time:      " << process_time << " s" << std::endl;
    std::cout << "  Effective bandwidth:  "
              << (total_bytes_transferred / (1024.0 * 1024.0 * 1024.0)) / process_time
              << " GB/s" << std::endl;
    std::cout << "  Errors:               " << errors << std::endl;
    std::cout << "  Offload ops:          " << engine.get_offload_count() << std::endl;
    std::cout << "  Load ops:             " << engine.get_load_count() << std::endl;
    std::cout << "  Prefetch ops:         " << engine.get_prefetch_count() << std::endl;

    EXPECT_EQ(errors, 0) << "Data corruption detected!";

    if (errors == 0) {
        std::cout << "\n  *** PASSED: " << (TARGET_DATA_MB / 1024)
                  << " GB processed with zero errors ***" << std::endl;
    }
}

TEST_P(OffloadEngineStressTest, ProcessDataLargerThanGPU_Concurrent) {

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "STRESS TEST: Concurrent Async Processing > GPU Memory" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Configuration - smaller chunks but many concurrent
    const size_t TARGET_DATA_MB = 8 * 1024;  // 8 GB total
    const size_t CHUNK_SIZE_MB = 128;        // 128 MB chunks
    const int NUM_CHUNKS = TARGET_DATA_MB / CHUNK_SIZE_MB;
    const int MAX_IN_FLIGHT = 8;             // Max concurrent transfers

    std::cout << "\nConfiguration:" << std::endl;
    std::cout << "  Target Data:      " << (TARGET_DATA_MB / 1024) << " GB" << std::endl;
    std::cout << "  Chunk Size:       " << CHUNK_SIZE_MB << " MB" << std::endl;
    std::cout << "  Num Chunks:       " << NUM_CHUNKS << std::endl;
    std::cout << "  Max In-Flight:    " << MAX_IN_FLIGHT << std::endl;

    OffloadEngine::Config config;
    config.pinned_memory_size = 2048ULL * 1024 * 1024;  // 2 GB pinned
    config.num_transfer_streams = 8;
    config.enable_prefetch = true;
    OffloadEngine engine(config);

    // Create chunks
    std::cout << "\nCreating chunks..." << std::endl;

    std::vector<Tensor> cpu_chunks;
    std::vector<double> expected_checksums;

    size_t elements = (CHUNK_SIZE_MB * 1024 * 1024) / sizeof(float);
    std::vector<int64_t> shape = {static_cast<int64_t>(elements)};

    for (int i = 0; i < NUM_CHUNKS; ++i) {
        cpu_chunks.push_back(createTensor(shape, 2000 + i));
        expected_checksums.push_back(computeChecksum(cpu_chunks.back()));
    }
    std::cout << "  Created " << NUM_CHUNKS << " chunks" << std::endl;

    // Process with concurrent async transfers
    std::cout << "\nProcessing with concurrent transfers..." << std::endl;

    size_t total_bytes = 0;
    int errors = 0;

    auto start = std::chrono::high_resolution_clock::now();

    // Use sliding window of in-flight transfers
    std::vector<TransferHandle> upload_handles;
    std::vector<TransferHandle> download_handles;
    std::vector<int> pending_chunks;

    int next_upload = 0;
    int next_download = 0;
    int completed = 0;

    while (completed < NUM_CHUNKS) {
        // Start new uploads if we have capacity
        while (upload_handles.size() < MAX_IN_FLIGHT && next_upload < NUM_CHUNKS) {
            upload_handles.push_back(engine.load_to_gpu_async(cpu_chunks[next_upload], device));
            pending_chunks.push_back(next_upload);
            total_bytes += cpu_chunks[next_upload].numel() * sizeof(float);
            next_upload++;
        }

        // Check for completed uploads and start downloads
        for (size_t i = 0; i < upload_handles.size(); ) {
            if (upload_handles[i].is_ready()) {
                Tensor gpu_tensor = upload_handles[i].get_tensor();
                download_handles.push_back(engine.offload_to_cpu_async(gpu_tensor));
                total_bytes += gpu_tensor.numel() * sizeof(float);

                upload_handles.erase(upload_handles.begin() + i);
            } else {
                i++;
            }
        }

        // Check for completed downloads
        for (size_t i = 0; i < download_handles.size(); ) {
            if (download_handles[i].is_ready()) {
                Tensor result = download_handles[i].get_tensor();
                int chunk_idx = pending_chunks[0];
                pending_chunks.erase(pending_chunks.begin());

                // Verify checksum
                double checksum = computeChecksum(result);
                if (std::abs(checksum - expected_checksums[chunk_idx]) > 1e-3) {
                    errors++;
                }

                completed++;
                download_handles.erase(download_handles.begin() + i);

                if (completed % 10 == 0) {
                    std::cout << "  Completed " << completed << "/" << NUM_CHUNKS << std::endl;
                }
            } else {
                i++;
            }
        }

        // Small sleep to avoid busy waiting
        std::this_thread::sleep_for(std::chrono::microseconds(100));
    }

    auto end = std::chrono::high_resolution_clock::now();
    double time = std::chrono::duration<double>(end - start).count();

    // Results
    std::cout << "\n" << std::string(40, '-') << std::endl;
    std::cout << "Results:" << std::endl;
    std::cout << "  Total data:          " << std::fixed << std::setprecision(2)
              << (total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  Processing time:     " << time << " s" << std::endl;
    std::cout << "  Effective bandwidth: "
              << (total_bytes / (1024.0 * 1024.0 * 1024.0)) / time << " GB/s" << std::endl;
    std::cout << "  Errors:              " << errors << std::endl;

    EXPECT_EQ(errors, 0) << "Data corruption detected!";

    if (errors == 0) {
        std::cout << "\n  *** PASSED: " << (TARGET_DATA_MB / 1024)
                  << " GB concurrent processing with zero errors ***" << std::endl;
    }
}

TEST_P(OffloadEngineStressTest, SimulateLargeModelTraining) {

    std::cout << "\n" << std::string(60, '=') << std::endl;
    std::cout << "STRESS TEST: Simulated Large Model Training" << std::endl;
    std::cout << std::string(60, '=') << std::endl;

    // Simulate a model with:
    // - 12 layers, each with weights + gradients + optimizer state
    // - Total: ~9 GB (exceeds 6 GB GPU)
    // - Train for 5 epochs

    const int NUM_LAYERS = 12;
    const size_t LAYER_SIZE_MB = 256;  // 256 MB per parameter tensor
    const int NUM_EPOCHS = 5;

    size_t total_model_size = NUM_LAYERS * LAYER_SIZE_MB * 3;  // weights + grads + opt_state

    std::cout << "\nModel Configuration:" << std::endl;
    std::cout << "  Layers:           " << NUM_LAYERS << std::endl;
    std::cout << "  Layer Size:       " << LAYER_SIZE_MB << " MB" << std::endl;
    std::cout << "  Total Model:      " << (total_model_size / 1024.0) << " GB" << std::endl;
    std::cout << "  GPU Memory:       6 GB" << std::endl;
    std::cout << "  Epochs:           " << NUM_EPOCHS << std::endl;

    OffloadEngine::Config config;
    config.pinned_memory_size = 2048ULL * 1024 * 1024;
    config.num_transfer_streams = 4;
    config.enable_prefetch = true;
    config.prefetch_depth = 4;
    OffloadEngine engine(config);

    // Create model layers on CPU
    struct Layer {
        Tensor weights;
        Tensor gradients;
        Tensor optimizer_state;
        double weights_checksum;
    };

    std::vector<Layer> layers(NUM_LAYERS);

    size_t elements = (LAYER_SIZE_MB * 1024 * 1024) / sizeof(float);
    std::vector<int64_t> shape = {static_cast<int64_t>(elements)};

    std::cout << "\nInitializing model on CPU..." << std::endl;
    for (int i = 0; i < NUM_LAYERS; ++i) {
        layers[i].weights = createTensor(shape, 3000 + i);
        layers[i].gradients = createTensor(shape, 4000 + i);
        layers[i].optimizer_state = createTensor(shape, 5000 + i);
        layers[i].weights_checksum = computeChecksum(layers[i].weights);
    }
    std::cout << "  Model initialized" << std::endl;

    size_t total_bytes = 0;
    int errors = 0;

    auto train_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < NUM_EPOCHS; ++epoch) {
        std::cout << "\nEpoch " << (epoch + 1) << "/" << NUM_EPOCHS << ":" << std::endl;

        // ===== FORWARD PASS =====
        // Load weights layer by layer, prefetching next
        for (int i = 0; i < NUM_LAYERS; ++i) {
            // Prefetch next layer's weights
            if (i + 1 < NUM_LAYERS) {
                engine.prefetch_to_gpu(&layers[i + 1].weights);
            }

            // Load current weights to GPU
            Tensor gpu_weights = engine.load_to_gpu(layers[i].weights, device);
            total_bytes += gpu_weights.numel() * sizeof(float);

            // Verify weights integrity
            double checksum = computeChecksum(gpu_weights);
            if (std::abs(checksum - layers[i].weights_checksum) > 1e-3) {
                std::cerr << "  Weight corruption at layer " << i << std::endl;
                errors++;
            }

            // "Compute" forward pass (simulated)
            // In reality: output = activation(weights @ input)
        }
        std::cout << "  Forward pass complete" << std::endl;

        // ===== BACKWARD PASS =====
        // Load gradients, compute, offload
        for (int i = NUM_LAYERS - 1; i >= 0; --i) {
            // Load gradient to GPU for backward computation
            Tensor gpu_grad = engine.load_to_gpu(layers[i].gradients, device);
            total_bytes += gpu_grad.numel() * sizeof(float);

            // "Compute" backward pass (simulated)
            // In reality: grad = activation_grad * upstream_grad

            // Offload computed gradient back to CPU
            Tensor result_grad = engine.offload_to_cpu(gpu_grad);
            total_bytes += result_grad.numel() * sizeof(float);
            layers[i].gradients = result_grad;
        }
        std::cout << "  Backward pass complete" << std::endl;

        // ===== OPTIMIZER STEP =====
        // Load optimizer state + weights + gradients, update, offload
        for (int i = 0; i < NUM_LAYERS; ++i) {
            // Load optimizer state
            Tensor gpu_opt = engine.load_to_gpu(layers[i].optimizer_state, device);
            total_bytes += gpu_opt.numel() * sizeof(float);

            // "Update" optimizer state (simulated)
            // In reality: momentum = beta * momentum + grad

            // Offload updated state
            Tensor result_opt = engine.offload_to_cpu(gpu_opt);
            total_bytes += result_opt.numel() * sizeof(float);
            layers[i].optimizer_state = result_opt;
        }
        std::cout << "  Optimizer step complete" << std::endl;
    }

    auto train_end = std::chrono::high_resolution_clock::now();
    double train_time = std::chrono::duration<double>(train_end - train_start).count();

    // Results
    std::cout << "\n" << std::string(40, '-') << std::endl;
    std::cout << "Training Results:" << std::endl;
    std::cout << "  Epochs completed: " << NUM_EPOCHS << std::endl;
    std::cout << "  Data transferred: " << std::fixed << std::setprecision(2)
              << (total_bytes / (1024.0 * 1024.0 * 1024.0)) << " GB" << std::endl;
    std::cout << "  Training time:    " << train_time << " s" << std::endl;
    std::cout << "  Avg bandwidth:    "
              << (total_bytes / (1024.0 * 1024.0 * 1024.0)) / train_time << " GB/s" << std::endl;
    std::cout << "  Errors:           " << errors << std::endl;
    std::cout << "\nEngine Statistics:" << std::endl;
    std::cout << "  Offload ops:      " << engine.get_offload_count() << std::endl;
    std::cout << "  Load ops:         " << engine.get_load_count() << std::endl;
    std::cout << "  Prefetch ops:     " << engine.get_prefetch_count() << std::endl;

    EXPECT_EQ(errors, 0) << "Data corruption detected during training!";

    if (errors == 0) {
        std::cout << "\n  *** PASSED: Large model training simulation completed ***" << std::endl;
    }
}

// FINDING 25: exercise every real (non-stub) OffloadEngine/TransferEngine GPU
// backend, not just CUDA.
INSTANTIATE_TEST_SUITE_P(
    GpuBackends,
    OffloadEngineStressTest,
    ::testing::Values("cuda", "rocm", "vulkan", "oneapi", "mps"),
    [](const ::testing::TestParamInfo<std::string>& info) {
        return info.param;
    }
);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
