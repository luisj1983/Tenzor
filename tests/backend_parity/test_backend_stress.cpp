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

    // Full-FP32 GEMM forward-error bound is O(K * eps * |a||b|); at K=2048 two
    // correct GEMMs that accumulate in a different blocked order (cuBLAS vs MKL)
    // legitimately differ by ~2e-4 on near-zero (catastrophically cancelled)
    // outputs, where rtol gives no slack — measured 2.05e-4 cuBLAS-vs-MKL. atol
    // must absorb that (TF32 is already disabled via EnsureInitialized in main,
    // so this is genuine FP32 rounding, not a tensor-core downgrade). Matches
    // the atol the BackendStressParity.LargeTensor_MatMul sibling uses for K=1024.
    // rtol uses the shared FP32 GEMM floor (parity::MATMUL_RTOL); atol is kept
    // at 5e-4 (looser than parity::MATMUL_ATOL) because K=2048 here is far
    // larger than the small-matrix cases — the forward-error bound grows with K
    // and the measured cuBLAS-vs-MKL near-zero divergence is 2.05e-4.
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, parity::MATMUL_RTOL, 5e-4f, "Large MatMul 2048x2048");
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

    // Pre-generate the per-iteration tensors ONCE on the host. Drawing them
    // inside the parity lambda with randn(..., inputs[0].device()) made each
    // backend consume a *different* RNG stream (device RNGs are independent of
    // the CPU RNG and of each other), so the "parity" compared unrelated random
    // sums — a spurious ~0.48 mismatch. Transferring the same host tensors to
    // each device keeps the inputs identical while still exercising the
    // per-iteration alloc/dealloc churn this test targets.
    std::vector<Tensor> temps;
    temps.reserve(100);
    for (int i = 0; i < 100; ++i) {
        temps.push_back(randn({64, 64}, DType::Float32, Device::cpu()));
    }

    test_operation_parity([temps](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        // Repeatedly allocate and deallocate
        for (int i = 0; i < 100; ++i) {
            auto temp = temps[i].to(inputs[0].device());
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
        // The matmul dominates the error budget of this chain, so it inherits
        // the FP32 cross-device GEMM floor (parity::MATMUL_*).
    }, {a, b, c}, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "Complex Math Chain");
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
    // Float32 GEMM forward-error bound is O(K * eps * |a||b|) ≈ 1e-3 for
    // K = 1024 with N(0,1) inputs; two correct GEMMs that merely sum in a
    // different blocked order legitimately differ by ~1e-4 on near-zero
    // (catastrophically cancelled) outputs, where rtol gives no slack.
    // atol must absorb that: 1e-5 flagged oneMKL-vs-MKL at 9.2e-5.
    // rtol uses the shared FP32 GEMM floor (parity::MATMUL_RTOL); atol kept at
    // 5e-4 because K=1024 here exceeds the small-matrix cases and the forward-
    // error bound grows with K (measured oneMKL-vs-MKL near-zero diff ~9.2e-5).
    test_operation_parity_single([](const std::vector<Tensor>& inputs) {
        return matmul(inputs[0], inputs[1]);
    }, {a, b}, device, parity::MATMUL_RTOL, 5e-4f, "Stress LargeMatMul 1024x1024");
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
        // matmul dominates the chain's error budget → FP32 GEMM floor.
    }, {a, b, c}, device, parity::MATMUL_RTOL, parity::MATMUL_ATOL, "Stress Complex Math Chain");
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
            // Deterministic per-iteration tensor (was randn, whose RNG sequence
            // differs between CPU and the target backend, making the parity
            // comparison meaningless). ones() still exercises alloc/dealloc.
            auto temp = ones({64, 64}, DType::Float32, inputs[0].device());
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
        // Scale so 100 unnormalized matmul+ReLU layers stay finite (otherwise
        // activations overflow to +inf and the parity diff is a meaningless inf).
        weights.push_back(randn({64, 64}, DType::Float32, Device::cpu()) * 0.1f);
    }
    test_operation_parity_single([&weights](const std::vector<Tensor>& inputs) {
        auto result = inputs[0];
        for (const auto& w : weights) {
            // Captured weights are CPU tensors; move to the run's compute device
            // so the device run doesn't mix OneAPI/CPU operands in matmul.
            result = clamp_min(matmul(result, w.to(result.device())), 0.0f);
        }
        return result;
    }, {x}, device, 1e-3f, 1e-5f, "Stress Deep Graph 100 Layers");
}

TEST_P(BackendStressParity, DeepGraph_Residual) {
    // Deterministic inputs: an unseeded draw makes the failing backend random
    // run-to-run (see below), so pin the RNG for reproducibility.
    tenzor::manual_seed(777);
    auto x = randn({16, 128}, DType::Float32, Device::cpu());
    // The residual block x += relu(x@w1)@w2 has per-block gain ≈ 1 + c·scale²·dim.
    // At scale 0.1 that gain is ≈1.9, so 50 blocks amplify activations to ~1e14 —
    // effectively a chaotic map. Tiny, *legitimate* FP differences between
    // backends (e.g. FMA vs non-FMA accumulation order) then diverge to O(1)
    // relative error at some elements, so cross-backend parity is impossible no
    // matter how correct each backend is (measured: Vulkan matched CPU exactly
    // while CUDA drifted 5e7 at 1e14 magnitude). Scale 0.02 keeps the per-block
    // gain just above 1 so activations stay bounded (~O(10)) over 50 blocks —
    // still a genuine deep residual graph, but now numerically well-conditioned
    // and meaningfully comparable across backends.
    auto w1 = randn({128, 128}, DType::Float32, Device::cpu()) * 0.02f;
    auto w2 = randn({128, 128}, DType::Float32, Device::cpu()) * 0.02f;
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
            // Captured tensors are CPU; move to the run's compute device.
            result = result + t.to(result.device()) * 0.01f;
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
    const double avg_ms = static_cast<double>(duration) / iterations;
    std::cout << "Backend " << backend_name(device)
              << " MatMul " << shape[0] << "x" << shape[1]
              << " average time: " << avg_ms << " ms\n";
    // Generous catastrophic-regression floor: a 1024x1024 FP32 matmul is well
    // under a second per iteration on any healthy backend. A SUCCEED()-only
    // body let a 1000x regression or a deadlock pass silently; assert the
    // average iteration time is below an absurd ceiling no backend approaches.
    EXPECT_GE(avg_ms, 0.0);
    EXPECT_LT(avg_ms, 5000.0)
        << "catastrophic MatMul regression on " << backend_name(device)
        << ": " << avg_ms << " ms/iter";
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
    const double avg_ms = static_cast<double>(duration) / iterations;
    std::cout << "Backend " << backend_name(device)
              << " MatMul 512x512 average time: " << avg_ms << " ms\n";
    EXPECT_GE(avg_ms, 0.0);
    EXPECT_LT(avg_ms, 5000.0)
        << "catastrophic MatMul regression on " << backend_name(device)
        << ": " << avg_ms << " ms/iter";
}

TEST_P(BackendStressParity, StabilityUnderLoad_Repeated) {
    auto x = randn({64, 64}, DType::Float32, Device::cpu());
    // K = 64 Float32 GEMM error bound is O(64 * eps * |a||b|) ≈ 7e-5; the
    // previous atol=1e-7 demanded near-bitwise agreement between different
    // GEMM implementations (oneMKL SYCL differed from CPU MKL by 2.3e-5,
    // squarely inside the legitimate bound). Tolerances sized to the math.
    for (int iter = 0; iter < 10; ++iter) {
        test_operation_parity_single([](const std::vector<Tensor>& inputs) {
            return matmul(inputs[0], inputs[0].transpose(0, 1));
        }, {x}, device, 1e-3f, 1e-4f,
           "Stress Stability Iter " + std::to_string(iter));
    }
}

INSTANTIATE_BACKEND_TESTS(BackendStressParity);

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            // Use EnsureInitialized (NOT tenzor::initialize() directly): it sets
            // TENZOR_DISABLE_TF32=1 before bring-up so cuBLAS runs FP32 matmul in
            // full IEEE 754 precision rather than silently downgrading to TF32 on
            // Ampere+. The plain TEST(BackendStress, *) cases here don't inherit
            // BackendTest, so without this the TF32 path blows the CPU↔CUDA
            // matmul parity tolerances (LargeTensor_MatMul, ComplexChain_MathOps).
            tenzor::testing::EnsureInitialized();
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
