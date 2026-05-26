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
#include "../backend_test_fixture.hpp"
#include <chrono>

using namespace tenzor;
using namespace tenzor::testing;

// FF.29: BackendStressParity sibling suite — TEST_P-wrapped versions of a
// subset of the original BackendStress TEST() cases. These let the parity
// coverage matrix see per-(op, backend) instead of one undifferentiated cell
// per op. The original TEST() variants stay for now: they use the
// `test_operation_parity` helper that loops backends internally, which is
// still meaningful as an integrated check.
class BackendStressParity : public BackendTest {};

// ============================================================================
// Large Tensor Tests
// ============================================================================

TEST(BackendStress, LargeTensor_1GB) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

    // ~1GB tensor (256M float32 elements)
    auto a = randn({16384, 16384}, DType::Float32, Device::cpu());
    auto b = randn({16384, 16384}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, 1e-6f, 1e-8f, "Large Tensor 1GB Add");
}

TEST(BackendStress, LargeTensor_MatMul) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

    // Large matrix multiplication: 2048 x 2048
    auto a = randn({2048, 2048}, DType::Float32, Device::cpu());
    auto b = randn({2048, 2048}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, 1e-3f, 1e-5f, "Large MatMul 2048x2048");
}

// II.16: LargeBatch_Conv2d, DeepGraph_100Layers, DeepGraph_Residual,
// MemoryPressure_ManyTensors, Performance_MatMul_Benchmark,
// Performance_MatMul_Large_Benchmark, StabilityUnderLoad_Repeated converted
// to TEST_P below — see the BackendStressParity block near the bottom of
// this file. The original TEST() versions were deleted in favour of the
// per-backend matrix coverage they now provide.

// ============================================================================
// Many Small Operations
// ============================================================================

TEST(BackendStress, ManySmallOperations_Sequential) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

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
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

    auto x = randn({16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Chain multiple operations
        for (int i = 0; i < 100; ++i) {
            result = clamp_min(result * 0.99f + 0.01f, 0.0f);
        }
        return result;
    }, {x}, 1e-4f, 1e-6f, "100 Chained Operations");
}

// ============================================================================
// Deep Computation Graphs
// ============================================================================

// ============================================================================
// Memory Pressure Tests
// ============================================================================

TEST(BackendStress, MemoryPressure_AllocDealloc) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

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

TEST(BackendStress, ComplexChain_MathOps) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

    // Complex chain: matmul → add → relu-like (clamp)
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 32}, DType::Float32, Device::cpu());
    auto c = randn({32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto mm = matmul(inputs[0], inputs[1]);
        auto added = mm + inputs[2];
        return clamp(added, -6.0f, 6.0f);  // ReLU6-like
    }, {a, b, c}, 1e-4f, 1e-6f, "Complex Math Chain");
}

TEST(BackendStress, ComplexChain_Reductions) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("backend_stress parity");

    auto input = randn({32, 128}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto exp_x = exp(inputs[0]);
        auto sum_exp = sum(exp_x, -1, true);
        return exp_x / sum_exp;  // Manual softmax
    }, {input}, 1e-5f, 1e-7f, "Manual Softmax Chain");
}

// ============================================================================
// Performance Benchmarks
// ============================================================================

// Performance_MatMul_Benchmark, Performance_MatMul_Large_Benchmark, and
// StabilityUnderLoad_Repeated were converted to TEST_P below (II.16).

// ============================================================================
// FF.29: TEST_P backend-parameterized variants of selected stress tests.
// Smaller shapes than the original BackendStress.* — the goal here is
// matrix-coverage breadth, not the original benchmark heft.
// ============================================================================

TEST_P(BackendStressParity, LargeTensor_1GB) {
    // ~64 MiB tensor (smaller than the original 1GB so the per-(backend, dtype)
    // matrix cells run in seconds, not minutes; the original TEST() at the top
    // of the file still exercises the full 1GB profile).
    auto a = randn({4096, 4096}, DType::Float32, Device::cpu());
    auto b = randn({4096, 4096}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-6f, 1e-8f, "Stress LargeTensor 64MiB Add");
}

TEST_P(BackendStressParity, LargeTensor_MatMul) {
    auto a = randn({1024, 1024}, DType::Float32, Device::cpu());
    auto b = randn({1024, 1024}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, 1e-3f, 1e-5f, "Stress LargeMatMul 1024x1024");
}

TEST_P(BackendStressParity, LargeBatch_Conv2d) {
    auto a = randn({8, 64, 32, 32}, DType::Float32, Device::cpu());
    auto b = randn({8, 64, 32, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return inputs[0] + inputs[1];
    }, {a, b}, device, 1e-5f, 1e-7f, "Stress LargeConvSize Add");
}

TEST_P(BackendStressParity, ManySmallOperations_Sequential) {
    auto x = randn({32, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        for (int i = 0; i < 1000; ++i) {
            result = result + 0.001f;
        }
        return result;
    }, {x}, device, 1e-4f, 1e-6f, "Stress 1000 Sequential Ops");
}

TEST_P(BackendStressParity, ManySmallOperations_Chained) {
    auto x = randn({16, 16}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        for (int i = 0; i < 100; ++i) {
            result = clamp_min(result * 0.99f + 0.01f, 0.0f);
        }
        return result;
    }, {x}, device, 1e-4f, 1e-6f, "Stress 100 Chained Ops");
}

TEST_P(BackendStressParity, ComplexChain_MathOps) {
    auto a = randn({32, 64}, DType::Float32, Device::cpu());
    auto b = randn({64, 32}, DType::Float32, Device::cpu());
    auto c = randn({32, 32}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto mm = matmul(inputs[0], inputs[1]);
        auto added = mm + inputs[2];
        return clamp(added, -6.0f, 6.0f);
    }, {a, b, c}, device, 1e-4f, 1e-6f, "Stress Complex Math Chain");
}

TEST_P(BackendStressParity, ComplexChain_Reductions) {
    auto input = randn({32, 128}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto exp_x = exp(inputs[0]);
        auto sum_exp = sum(exp_x, -1, true);
        return exp_x / sum_exp;  // Manual softmax
    }, {input}, device, 1e-5f, 1e-7f, "Stress Manual Softmax Chain");
}

TEST_P(BackendStressParity, MemoryPressure_AllocDealloc) {
    auto x = randn({64, 64}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        for (int i = 0; i < 100; ++i) {
            auto temp = randn({64, 64}, DType::Float32, inputs[0].device());
            result = result + temp * 0.01f;
        }
        return result;
    }, {x}, device, 1e-4f, 1e-6f, "Stress Alloc/Dealloc");
}

// ----------------------------------------------------------------------------
// II.16: TEST_P versions of the remaining stress cases. Shapes/iteration
// counts are tuned down from the original TEST() values so the per-(backend)
// matrix runs in seconds, not minutes, while still exercising the deep-graph
// and memory-pressure code paths.
// ----------------------------------------------------------------------------

TEST_P(BackendStressParity, DeepGraph_100Layers) {
    auto x = randn({8, 64}, DType::Float32, Device::cpu());
    std::vector<Tensor> weights;
    for (int i = 0; i < 100; ++i) {
        weights.push_back(randn({64, 64}, DType::Float32, Device::cpu()));
    }
    test_operation_parity_single([&weights](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        for (const auto& w : weights) {
            result = clamp_min(matmul(result, w), 0.0f);
        }
        return result;
    }, {x}, device, 1e-3f, 1e-5f, "Stress Deep Graph 100 Layers");
}

TEST_P(BackendStressParity, DeepGraph_Residual) {
    auto x = randn({16, 128}, DType::Float32, Device::cpu());
    auto w1 = randn({128, 128}, DType::Float32, Device::cpu());
    auto w2 = randn({128, 128}, DType::Float32, Device::cpu());
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        auto x = inputs[0];
        auto w1 = inputs[1];
        auto w2 = inputs[2];
        for (int i = 0; i < 50; ++i) {
            auto residual = x;
            x = clamp_min(matmul(x, w1), 0.0f);
            x = matmul(x, w2);
            x = x + residual;
        }
        return x;
    }, {x, w1, w2}, device, 1e-3f, 1e-5f, "Stress Deep Residual 50 Blocks");
}

TEST_P(BackendStressParity, MemoryPressure_ManyTensors) {
    std::vector<Tensor> tensors;
    for (int i = 0; i < 100; ++i) {
        tensors.push_back(randn({256, 256}, DType::Float32, Device::cpu()));
    }
    auto input = randn({256, 256}, DType::Float32, Device::cpu());
    test_operation_parity_single([&tensors](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        for (const auto& t : tensors) {
            result = result + t * 0.01f;
        }
        return result;
    }, {input}, device, 1e-4f, 1e-6f, "Stress Memory Pressure 100 Tensors");
}

TEST_P(BackendStressParity, Performance_MatMul_Benchmark) {
    // Reporting-only benchmark — no assertion. Each backend prints its own
    // timing line so CI logs show per-backend MatMul perf side by side.
    const int iterations = 10;
    const std::vector<int64_t> shape = {1024, 1024};
    auto a = randn(shape, DType::Float32, device);
    auto b = randn(shape, DType::Float32, device);

    for (int i = 0; i < 3; ++i) {
        auto c = matmul(a, b);
        device.synchronize();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto c = matmul(a, b);
        device.synchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Backend " << backend_name(device)
              << " MatMul " << shape[0] << "x" << shape[1]
              << " average time: " << (duration / iterations) << " ms\n";
    SUCCEED();
}

TEST_P(BackendStressParity, Performance_MatMul_Large_Benchmark) {
    const int iterations = 5;
    const std::vector<int64_t> shape = {512, 512};
    auto a = randn(shape, DType::Float32, device);
    auto b = randn(shape, DType::Float32, device);

    for (int i = 0; i < 2; ++i) {
        auto c = matmul(a, b);
        device.synchronize();
    }
    auto start = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < iterations; ++i) {
        auto c = matmul(a, b);
        device.synchronize();
    }
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Backend " << backend_name(device)
              << " MatMul 512x512 average time: " << (duration / iterations) << " ms\n";
    SUCCEED();
}

TEST_P(BackendStressParity, StabilityUnderLoad_Repeated) {
    auto x = randn({64, 64}, DType::Float32, Device::cpu());
    for (int iter = 0; iter < 10; ++iter) {
        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            return matmul(inputs[0], inputs[0].transpose(0, 1));
        }, {x}, device, 1e-5f, 1e-7f,
           "Stress Stability Iter " + std::to_string(iter));
    }
}

INSTANTIATE_BACKEND_TESTS(BackendStressParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
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
