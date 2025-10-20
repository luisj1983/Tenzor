/**
 * @file bench_fusion.cpp
 * @brief Performance benchmarks for kernel fusion optimizations
 *
 * Measures actual speedups achieved through fusion compared to unfused operations.
 */

#include <benchmark/benchmark.h>
#include "tenzor/ops/fusion_optimizer.hpp"
#include "tenzor/ops/fused_ops.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/core/device.hpp"
#include <memory>

using namespace tenzor;
using namespace tenzor::ops;

// ==============================================================================
// Helper Functions
// ==============================================================================

static Device get_benchmark_device() {
#ifdef TENZOR_CUDA_AVAILABLE
    return Device::cuda(0);
#else
    return Device::cpu();
#endif
}

// ==============================================================================
// Linear + ReLU Benchmarks
// ==============================================================================

static void BM_UnfusedLinearReLU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t in_features = 512;
    int64_t out_features = 256;

    auto input = randn({batch_size, in_features}, DType::Float32, device);
    auto weight = randn({out_features, in_features}, DType::Float32, device);
    auto bias = randn({out_features}, DType::Float32, device);

    for (auto _ : state) {
        // Unfused: separate operations
        auto linear_out = matmul(input, weight.transpose(0, 1));
        linear_out = linear_out + bias;
        auto relu_out = relu(linear_out);

        benchmark::DoNotOptimize(relu_out);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * out_features);
    state.SetBytesProcessed(state.iterations() * batch_size * out_features * sizeof(float) * 2);
}
BENCHMARK(BM_UnfusedLinearReLU)->Range(8, 128)->Unit(benchmark::kMicrosecond);

static void BM_FusedLinearReLU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t in_features = 512;
    int64_t out_features = 256;

    auto input = randn({batch_size, in_features}, DType::Float32, device);
    auto weight = randn({out_features, in_features}, DType::Float32, device);
    auto bias = randn({out_features}, DType::Float32, device);

    for (auto _ : state) {
        // Fused operation
        auto output = fused_linear_relu(input, weight, &bias);

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * out_features);
    state.SetBytesProcessed(state.iterations() * batch_size * out_features * sizeof(float) * 2);
}
BENCHMARK(BM_FusedLinearReLU)->Range(8, 128)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// BatchNorm + ReLU Benchmarks
// ==============================================================================

static void BM_UnfusedBatchNormReLU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t num_features = 128;
    int64_t spatial_size = 16;

    auto input = randn({batch_size, num_features, spatial_size, spatial_size},
                      DType::Float32, device);
    auto mean = zeros({num_features}, DType::Float32, device);
    auto var = ones({num_features}, DType::Float32, device);
    auto gamma = ones({num_features}, DType::Float32, device);
    auto beta = zeros({num_features}, DType::Float32, device);

    for (auto _ : state) {
        // Unfused: separate batch norm and relu
        auto bn_out = batch_norm(input, mean, var, gamma, beta);
        auto relu_out = relu(bn_out);

        benchmark::DoNotOptimize(relu_out);
    }

    int64_t total_elements = batch_size * num_features * spatial_size * spatial_size;
    state.SetItemsProcessed(state.iterations() * total_elements);
    state.SetBytesProcessed(state.iterations() * total_elements * sizeof(float) * 2);
}
BENCHMARK(BM_UnfusedBatchNormReLU)->Range(4, 32)->Unit(benchmark::kMicrosecond);

static void BM_FusedBatchNormReLU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t num_features = 128;
    int64_t spatial_size = 16;

    auto input = randn({batch_size, num_features, spatial_size, spatial_size},
                      DType::Float32, device);
    auto mean = zeros({num_features}, DType::Float32, device);
    auto var = ones({num_features}, DType::Float32, device);
    auto gamma = ones({num_features}, DType::Float32, device);
    auto beta = zeros({num_features}, DType::Float32, device);

    for (auto _ : state) {
        // Fused operation
        auto output = fused_batchnorm_relu(input, mean, var, gamma, beta);

        benchmark::DoNotOptimize(output);
    }

    int64_t total_elements = batch_size * num_features * spatial_size * spatial_size;
    state.SetItemsProcessed(state.iterations() * total_elements);
    state.SetBytesProcessed(state.iterations() * total_elements * sizeof(float) * 2);
}
BENCHMARK(BM_FusedBatchNormReLU)->Range(4, 32)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// Softmax + CrossEntropy Benchmarks
// ==============================================================================

static void BM_UnfusedSoftmaxCrossEntropy(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t num_classes = 1000;

    auto logits = randn({batch_size, num_classes}, DType::Float32, device);
    auto targets = randint(0, num_classes, {batch_size}, device);

    for (auto _ : state) {
        // Unfused: separate softmax and cross entropy
        auto probs = softmax(logits, 1);
        auto loss = cross_entropy(probs, targets);

        benchmark::DoNotOptimize(loss);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * num_classes);
}
BENCHMARK(BM_UnfusedSoftmaxCrossEntropy)->Range(16, 256)->Unit(benchmark::kMicrosecond);

static void BM_FusedSoftmaxCrossEntropy(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t num_classes = 1000;

    auto logits = randn({batch_size, num_classes}, DType::Float32, device);
    auto targets = randint(0, num_classes, {batch_size}, device);

    for (auto _ : state) {
        // Fused operation
        auto loss = fused_softmax_cross_entropy(logits, targets, "mean");

        benchmark::DoNotOptimize(loss);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * num_classes);
}
BENCHMARK(BM_FusedSoftmaxCrossEntropy)->Range(16, 256)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// Add + ReLU Benchmarks (Residual Connections)
// ==============================================================================

static void BM_UnfusedAddReLU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t features = 2048;

    auto a = randn({batch_size, features}, DType::Float32, device);
    auto b = randn({batch_size, features}, DType::Float32, device);

    for (auto _ : state) {
        // Unfused: separate add and relu
        auto sum = a + b;
        auto output = relu(sum);

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * features);
}
BENCHMARK(BM_UnfusedAddReLU)->Range(8, 256)->Unit(benchmark::kMicrosecond);

static void BM_FusedAddReLU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t features = 2048;

    auto a = randn({batch_size, features}, DType::Float32, device);
    auto b = randn({batch_size, features}, DType::Float32, device);

    for (auto _ : state) {
        // Fused operation
        auto output = fused_add_relu(a, b);

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * features);
}
BENCHMARK(BM_FusedAddReLU)->Range(8, 256)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// GELU Benchmarks
// ==============================================================================

static void BM_UnfusedGELU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t features = 4096;

    auto input = randn({batch_size, features}, DType::Float32, device);

    for (auto _ : state) {
        // Unfused GELU (multiple operations)
        // gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        auto x_cubed = input * input * input;
        auto inner = input + x_cubed * 0.044715f;
        inner = inner * 0.7978845608f;  // sqrt(2/pi)
        auto tanh_val = tanh(inner);
        auto output = input * (tanh_val + 1.0f) * 0.5f;

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * features);
}
BENCHMARK(BM_UnfusedGELU)->Range(8, 256)->Unit(benchmark::kMicrosecond);

static void BM_FusedGELU(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t features = 4096;

    auto input = randn({batch_size, features}, DType::Float32, device);

    for (auto _ : state) {
        // Fused GELU operation
        auto output = fused_gelu(input);

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * features);
}
BENCHMARK(BM_FusedGELU)->Range(8, 256)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// Layer Normalization Benchmarks
// ==============================================================================

static void BM_UnfusedLayerNorm(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t features = 512;

    auto input = randn({batch_size, features}, DType::Float32, device);
    auto weight = ones({features}, DType::Float32, device);
    auto bias = zeros({features}, DType::Float32, device);

    for (auto _ : state) {
        // Unfused: separate mean, var, normalize
        auto mean = input.mean(1, true);
        auto var = input.var(1, true);
        auto normalized = (input - mean) / (var + 1e-5f).sqrt();
        auto output = normalized * weight + bias;

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * features);
}
BENCHMARK(BM_UnfusedLayerNorm)->Range(16, 256)->Unit(benchmark::kMicrosecond);

static void BM_FusedLayerNorm(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t batch_size = state.range(0);
    int64_t features = 512;

    auto input = randn({batch_size, features}, DType::Float32, device);
    auto weight = ones({features}, DType::Float32, device);
    auto bias = zeros({features}, DType::Float32, device);

    for (auto _ : state) {
        // Fused layer norm (single pass)
        auto output = fused_layer_norm(input, {features}, weight, bias);

        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * batch_size * features);
}
BENCHMARK(BM_FusedLayerNorm)->Range(16, 256)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// Graph Optimization Benchmarks
// ==============================================================================

static void BM_FusionGraphConstruction(benchmark::State& state) {
    int64_t num_nodes = state.range(0);

    for (auto _ : state) {
        FusionGraph graph;

        // Build linear chain of operations
        for (int64_t i = 0; i < num_nodes; ++i) {
            OpType type = (i % 2 == 0) ? OpType::Linear : OpType::ReLU;
            std::vector<size_t> inputs = (i > 0) ? std::vector<size_t>{size_t(i-1)} : std::vector<size_t>{};
            graph.add_node(type, "op" + std::to_string(i), inputs);
        }

        benchmark::DoNotOptimize(graph);
    }

    state.SetItemsProcessed(state.iterations() * num_nodes);
}
BENCHMARK(BM_FusionGraphConstruction)->Range(8, 1024)->Unit(benchmark::kMicrosecond);

static void BM_FusionOptimization(benchmark::State& state) {
    int64_t num_patterns = state.range(0);

    // Pre-build graph
    FusionGraph graph;
    for (int64_t i = 0; i < num_patterns; ++i) {
        size_t base = i * 2;
        graph.add_node(OpType::Linear, "linear" + std::to_string(i),
                      i > 0 ? std::vector<size_t>{base - 1} : std::vector<size_t>{});
        graph.add_node(OpType::ReLU, "relu" + std::to_string(i), {base});
    }

    FusionOptimizer optimizer;
    optimizer.add_pattern("linear_relu");

    for (auto _ : state) {
        auto optimized = optimizer.optimize(graph);
        benchmark::DoNotOptimize(optimized);
    }

    state.SetItemsProcessed(state.iterations() * num_patterns);
}
BENCHMARK(BM_FusionOptimization)->Range(4, 128)->Unit(benchmark::kMicrosecond);

static void BM_PatternMatching(benchmark::State& state) {
    int64_t graph_size = state.range(0);

    // Build complex graph
    FusionGraph graph;
    for (int64_t i = 0; i < graph_size; ++i) {
        OpType type;
        switch (i % 4) {
            case 0: type = OpType::Linear; break;
            case 1: type = OpType::ReLU; break;
            case 2: type = OpType::MatMul; break;
            default: type = OpType::Add; break;
        }
        std::vector<size_t> inputs = (i > 0) ? std::vector<size_t>{size_t(i-1)} : std::vector<size_t>{};
        graph.add_node(type, "op" + std::to_string(i), inputs);
    }

    FusionPattern pattern("linear_relu");

    for (auto _ : state) {
        int matches = 0;
        for (size_t i = 0; i < graph_size; ++i) {
            auto match = pattern.match(graph, i);
            if (match) {
                ++matches;
            }
        }
        benchmark::DoNotOptimize(matches);
    }

    state.SetItemsProcessed(state.iterations() * graph_size);
}
BENCHMARK(BM_PatternMatching)->Range(16, 512)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// Memory Bandwidth Tests
// ==============================================================================

static void BM_MemoryBandwidthUnfused(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t size = state.range(0) * 1024 * 1024 / sizeof(float);  // MB

    auto a = randn({size}, DType::Float32, device);
    auto b = randn({size}, DType::Float32, device);
    auto c = randn({size}, DType::Float32, device);

    for (auto _ : state) {
        // Three separate operations = 3 memory passes
        auto temp1 = a + b;
        auto temp2 = temp1 * c;
        auto output = relu(temp2);

        benchmark::DoNotOptimize(output);
    }

    // 5 reads + 3 writes = 8 memory accesses per element
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 8);
}
BENCHMARK(BM_MemoryBandwidthUnfused)->Range(1, 64)->Unit(benchmark::kMillisecond);

static void BM_MemoryBandwidthFused(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t size = state.range(0) * 1024 * 1024 / sizeof(float);  // MB

    auto a = randn({size}, DType::Float32, device);
    auto b = randn({size}, DType::Float32, device);
    auto c = randn({size}, DType::Float32, device);

    for (auto _ : state) {
        // Single fused operation = 1 memory pass
        auto output = fused_elementwise_chain(a, b, c, 0);

        benchmark::DoNotOptimize(output);
    }

    // 3 reads + 1 write = 4 memory accesses per element
    state.SetBytesProcessed(state.iterations() * size * sizeof(float) * 4);
}
BENCHMARK(BM_MemoryBandwidthFused)->Range(1, 64)->Unit(benchmark::kMillisecond);

// ==============================================================================
// Kernel Launch Overhead Tests
// ==============================================================================

static void BM_KernelLaunchOverhead(benchmark::State& state) {
    auto device = get_benchmark_device();
    int64_t num_ops = state.range(0);
    int64_t size = 1024;

    std::vector<Tensor> tensors;
    for (int64_t i = 0; i < num_ops; ++i) {
        tensors.push_back(randn({size}, DType::Float32, device));
    }

    for (auto _ : state) {
        // Multiple kernel launches
        Tensor result = tensors[0];
        for (int64_t i = 1; i < num_ops; ++i) {
            result = result + tensors[i];
        }

        benchmark::DoNotOptimize(result);
    }

    state.SetItemsProcessed(state.iterations() * num_ops);
}
BENCHMARK(BM_KernelLaunchOverhead)->Range(2, 32)->Unit(benchmark::kMicrosecond);

// ==============================================================================
// Main
// ==============================================================================

BENCHMARK_MAIN();
