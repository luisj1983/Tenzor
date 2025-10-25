/**
 * @file test_backend_stress.cpp
 * @brief Stress tests for backend implementations
 *
 * Tests backend performance and correctness under heavy load including
 * large tensors, many operations, deep graphs, and concurrent execution.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"
#include <chrono>

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Large Tensor Tests
// ============================================================================

TEST(BackendStress, LargeTensor_1GB) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // ~1GB tensor (256M float32 elements)
    auto a = randn({16384, 16384}, DType::Float32, Device::cpu());
    auto b = randn({16384, 16384}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Large Tensor 1GB Add");
}

TEST(BackendStress, LargeTensor_MatMul) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Large matrix multiplication: 2048 x 2048
    auto a = randn({2048, 2048}, DType::Float32, Device::cpu());
    auto b = randn({2048, 2048}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-3f, 1e-5f, "Large MatMul 2048x2048");
}

TEST(BackendStress, LargeBatch_Conv2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Large batch convolution
    auto input = randn({64, 64, 128, 128}, DType::Float32, Device::cpu());
    auto weight = randn({128, 64, 3, 3}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv2d(inputs[0], inputs[1], std::nullopt,
                                     /*stride=*/1, /*padding=*/1);
    }, {input, weight}, 1e-4f, 1e-6f, "Large Batch Conv2d");
}

// ============================================================================
// Many Small Operations
// ============================================================================

TEST(BackendStress, ManySmallOperations_Sequential) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Perform 1000 sequential operations
        for (int i = 0; i < 1000; ++i) {
            result = result + 0.001f;
        }
        return result;
    }, {x}, 1e-4f, 1e-6f, "1000 Sequential Operations");
}

TEST(BackendStress, ManySmallOperations_Chained) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Chain multiple operations
        for (int i = 0; i < 100; ++i) {
            result = nn::relu(result * 0.99f + 0.01f);
        }
        return result;
    }, {x}, 1e-4f, 1e-6f, "100 Chained Operations");
}

// ============================================================================
// Deep Computation Graphs
// ============================================================================

TEST(BackendStress, DeepGraph_100Layers) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({8, 64}, DType::Float32, Device::cpu());
    std::vector<Tensor> weights;

    // Create 100 layer weights
    for (int i = 0; i < 100; ++i) {
        weights.push_back(randn({64, 64}, DType::Float32, Device::cpu()));
    }

    test_operation_parity([&weights](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Deep forward pass
        for (const auto& w : weights) {
            result = nn::relu(matmul(result, w));
        }
        return result;
    }, {x}, 1e-3f, 1e-5f, "Deep Graph 100 Layers");
}

TEST(BackendStress, DeepGraph_Residual) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({16, 128}, DType::Float32, Device::cpu());
    auto w1 = randn({128, 128}, DType::Float32, Device::cpu());
    auto w2 = randn({128, 128}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = inputs[0];
        auto w1 = inputs[1];
        auto w2 = inputs[2];

        // 50 residual blocks
        for (int i = 0; i < 50; ++i) {
            auto residual = x;
            x = nn::relu(matmul(x, w1));
            x = matmul(x, w2);
            x = x + residual;  // Residual connection
        }
        return x;
    }, {x, w1, w2}, 1e-3f, 1e-5f, "Deep Residual 50 Blocks");
}

// ============================================================================
// Memory Pressure Tests
// ============================================================================

TEST(BackendStress, MemoryPressure_ManyTensors) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    std::vector<Tensor> tensors;

    // Create 100 medium-sized tensors
    for (int i = 0; i < 100; ++i) {
        tensors.push_back(randn({256, 256}, DType::Float32, Device::cpu()));
    }

    auto input = randn({256, 256}, DType::Float32, Device::cpu());

    test_operation_parity([&tensors](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Add all tensors
        for (const auto& t : tensors) {
            result = result + t * 0.01f;
        }
        return result;
    }, {input}, 1e-4f, 1e-6f, "Memory Pressure 100 Tensors");
}

TEST(BackendStress, MemoryPressure_AllocDealloc) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({64, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Repeatedly allocate and deallocate
        for (int i = 0; i < 100; ++i) {
            auto temp = randn({64, 64}, DType::Float32, inputs[0].device());
            result = result + temp * 0.01f;
            // temp will be deallocated at end of iteration
        }
        return result;
    }, {x}, 1e-4f, 1e-6f, "Alloc/Dealloc Stress");
}

// ============================================================================
// Complex Operation Chains
// ============================================================================

TEST(BackendStress, ComplexChain_CNN) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 3, 64, 64}, DType::Float32, Device::cpu());
    auto conv1_w = randn({32, 3, 3, 3}, DType::Float32, Device::cpu());
    auto conv2_w = randn({64, 32, 3, 3}, DType::Float32, Device::cpu());
    auto fc_w = randn({64 * 16 * 16, 10}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = inputs[0];
        auto conv1_w = inputs[1];
        auto conv2_w = inputs[2];
        auto fc_w = inputs[3];

        // CNN forward pass
        x = nn::functional::conv2d(x, conv1_w, std::nullopt, 1, 1);
        x = nn::relu(x);
        x = nn::functional::max_pool2d(x, 2);

        x = nn::functional::conv2d(x, conv2_w, std::nullopt, 1, 1);
        x = nn::relu(x);
        x = nn::functional::max_pool2d(x, 2);

        x = x.reshape({x.shape()[0], -1});
        x = matmul(x, fc_w);

        return x;
    }, {input, conv1_w, conv2_w, fc_w}, 1e-3f, 1e-5f, "Complex CNN Chain");
}

TEST(BackendStress, ComplexChain_Transformer) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({2, 32, 256}, DType::Float32, Device::cpu());
    auto q_w = randn({256, 256}, DType::Float32, Device::cpu());
    auto k_w = randn({256, 256}, DType::Float32, Device::cpu());
    auto v_w = randn({256, 256}, DType::Float32, Device::cpu());
    auto o_w = randn({256, 256}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = inputs[0];
        auto q_w = inputs[1];
        auto k_w = inputs[2];
        auto v_w = inputs[3];
        auto o_w = inputs[4];

        // Simplified attention mechanism
        auto q = matmul(x, q_w);
        auto k = matmul(x, k_w);
        auto v = matmul(x, v_w);

        // Attention scores
        auto scores = matmul(q, k.transpose(-2, -1)) / std::sqrt(256.0f);
        auto attn = nn::softmax(scores, -1);

        // Apply attention
        auto out = matmul(attn, v);
        out = matmul(out, o_w);

        return out;
    }, {x, q_w, k_w, v_w, o_w}, 1e-3f, 1e-5f, "Complex Transformer Chain");
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

TEST(BackendStress, Performance_MatMul_Benchmark) {
    auto backends = get_available_backends();

    const int iterations = 10;
    const std::vector<int64_t> shape = {1024, 1024};

    auto a = randn(shape, DType::Float32, Device::cpu());
    auto b = randn(shape, DType::Float32, Device::cpu());

    for (const auto& backend : backends) {
        auto a_dev = a.to(backend);
        auto b_dev = b.to(backend);

        // Warmup
        for (int i = 0; i < 3; ++i) {
            auto c = matmul(a_dev, b_dev);
            backend.synchronize();
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            auto c = matmul(a_dev, b_dev);
            backend.synchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Backend " << backend_name(backend)
                 << " MatMul " << shape[0] << "x" << shape[1]
                 << " average time: " << (duration / iterations) << " ms"
                 << std::endl;
    }

    // No assertion, just benchmark reporting
    SUCCEED();
}

TEST(BackendStress, Performance_Conv2d_Benchmark) {
    auto backends = get_available_backends();

    const int iterations = 10;

    auto input = randn({16, 32, 128, 128}, DType::Float32, Device::cpu());
    auto weight = randn({64, 32, 3, 3}, DType::Float32, Device::cpu());

    for (const auto& backend : backends) {
        auto input_dev = input.to(backend);
        auto weight_dev = weight.to(backend);

        // Warmup
        for (int i = 0; i < 3; ++i) {
            auto output = nn::functional::conv2d(input_dev, weight_dev, std::nullopt, 1, 1);
            backend.synchronize();
        }

        // Benchmark
        auto start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < iterations; ++i) {
            auto output = nn::functional::conv2d(input_dev, weight_dev, std::nullopt, 1, 1);
            backend.synchronize();
        }

        auto end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

        std::cout << "Backend " << backend_name(backend)
                 << " Conv2d average time: " << (duration / iterations) << " ms"
                 << std::endl;
    }

    SUCCEED();
}

// ============================================================================
// Stability Under Load
// ============================================================================

TEST(BackendStress, StabilityUnderLoad_Repeated) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto x = randn({64, 64}, DType::Float32, Device::cpu());

    // Repeat operation 10 times to check consistency
    for (int iter = 0; iter < 10; ++iter) {
        test_operation_parity([](const std::vector<Tensor>& inputs) {
            return matmul(inputs[0], inputs[0].transpose(0, 1));
        }, {x}, 1e-5f, 1e-7f, "Stability Iteration " + std::to_string(iter));
    }
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
