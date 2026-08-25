/**
 * @file benchmark_compression.cpp
 * @brief Comprehensive benchmarks for model compression (pruning + quantization)
 *
 * Tests:
 * 1. Pruning: 50-90% sparsity levels
 * 2. Quantization: FP32 → INT8 accuracy retention
 * 3. Compression ratios and performance gains
 */

#include <benchmark/benchmark.h>
#include "tenzor/tenzor.hpp"
#include "tenzor/nn/compression/pruning.hpp"
#include "tenzor/nn/quantization.hpp"
#include "common.hpp"
#include <iostream>
#include <iomanip>

// RR.19 (audit-11): global device parsed from argv in main(). Defaults to
// CPU so the historical unflagged Google-Benchmark invocations stay
// correct. Replaces the 11 hardcoded `Device::cpu()` literals that
// previously locked this binary to the host CPU regardless of the
// --device flag the Python runner passes.
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::nn::compression;
using namespace tenzor::nn::quantization;

// =============================================================================
// Test Model for Benchmarks
// =============================================================================

class BenchmarkModel : public Module {
public:
    BenchmarkModel(int input_size, int hidden_size, int num_layers, int output_size) {
        // Build multi-layer network
        layers_.push_back(std::make_shared<Linear>(input_size, hidden_size));
        register_module("layer0", layers_[0]);

        for (int i = 1; i < num_layers - 1; ++i) {
            layers_.push_back(std::make_shared<Linear>(hidden_size, hidden_size));
            register_module("layer" + std::to_string(i), layers_[i]);
        }

        layers_.push_back(std::make_shared<Linear>(hidden_size, output_size));
        register_module("layer" + std::to_string(num_layers - 1), layers_.back());
    }

    Variable forward_impl(const Variable& input) override {
        Variable x = input;
        for (size_t i = 0; i < layers_.size() - 1; ++i) {
            x = layers_[i]->forward(x);
            x = relu(x);
        }
        return layers_.back()->forward(x);
    }

    std::vector<std::shared_ptr<Linear>> layers_;
};

// =============================================================================
// Pruning Benchmarks
// =============================================================================

static void BM_Pruning_Unstructured_50Percent(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);

    for (auto _ : state) {
        auto config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1);
        benchmark::DoNotOptimize(config);
    }

    // Measure actual sparsity achieved
    auto config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1);
    apply_pruning_masks(model, config);
    float sparsity = compute_sparsity(model);

    state.counters["Sparsity"] = sparsity;
    state.counters["TargetSparsity"] = 0.5f;
}
BENCHMARK(BM_Pruning_Unstructured_50Percent)->Unit(benchmark::kMillisecond);

static void BM_Pruning_Unstructured_70Percent(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);

    for (auto _ : state) {
        auto config = prune_unstructured(model, 0.7f, ImportanceCriterion::L1);
        benchmark::DoNotOptimize(config);
    }

    auto config = prune_unstructured(model, 0.7f, ImportanceCriterion::L1);
    apply_pruning_masks(model, config);
    float sparsity = compute_sparsity(model);

    state.counters["Sparsity"] = sparsity;
    state.counters["TargetSparsity"] = 0.7f;
}
BENCHMARK(BM_Pruning_Unstructured_70Percent)->Unit(benchmark::kMillisecond);

static void BM_Pruning_Unstructured_90Percent(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);

    for (auto _ : state) {
        auto config = prune_unstructured(model, 0.9f, ImportanceCriterion::L1);
        benchmark::DoNotOptimize(config);
    }

    auto config = prune_unstructured(model, 0.9f, ImportanceCriterion::L1);
    apply_pruning_masks(model, config);
    float sparsity = compute_sparsity(model);

    state.counters["Sparsity"] = sparsity;
    state.counters["TargetSparsity"] = 0.9f;
}
BENCHMARK(BM_Pruning_Unstructured_90Percent)->Unit(benchmark::kMillisecond);

static void BM_Pruning_Global_vs_Local(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);
    bool global = state.range(0);

    for (auto _ : state) {
        auto config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1, global);
        benchmark::DoNotOptimize(config);
    }
}
BENCHMARK(BM_Pruning_Global_vs_Local)->Arg(0)->Arg(1)->Unit(benchmark::kMillisecond);

static void BM_Pruning_Iterative(benchmark::State& state) {
    int num_iterations = state.range(0);

    for (auto _ : state) {
        auto model = std::make_shared<BenchmarkModel>(784, 256, 2, 10);
        auto config = prune_iterative(model, 0.8f, num_iterations,
                                     PruningSchedule::Polynomial,
                                     ImportanceCriterion::L1);
        benchmark::DoNotOptimize(config);
    }
}
BENCHMARK(BM_Pruning_Iterative)->Arg(3)->Arg(5)->Arg(10)->Unit(benchmark::kMillisecond);

static void BM_Pruning_StructuredChannel(benchmark::State& state) {
    auto conv = std::make_shared<Conv2d>(64, 128, 3);
    float sparsity = state.range(0) / 100.0f;

    for (auto _ : state) {
        auto pruned = prune_channels(conv, sparsity, ImportanceCriterion::L1);
        benchmark::DoNotOptimize(pruned);
    }
}
BENCHMARK(BM_Pruning_StructuredChannel)->Arg(30)->Arg(50)->Arg(70)->Unit(benchmark::kMillisecond);

// =============================================================================
// Quantization Benchmarks
// =============================================================================

static void BM_Quantization_PerTensorSymmetric(benchmark::State& state) {
    Tensor input = randn({1024, 1024}, DType::Float32, g_bench_device);

    for (auto _ : state) {
        auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
        benchmark::DoNotOptimize(q_tensor);
    }

    // Measure quantization error
    auto q_tensor = quantize_per_tensor_symmetric(input, QuantDType::INT8);
    float error = std::get<0>(compute_quantization_error(input, q_tensor));

    state.counters["QuantError"] = error;
    state.SetBytesProcessed(state.iterations() * input.numel() * sizeof(float));
}
BENCHMARK(BM_Quantization_PerTensorSymmetric)->Unit(benchmark::kMillisecond);

static void BM_Quantization_PerTensorAsymmetric(benchmark::State& state) {
    Tensor input = randn({1024, 1024}, DType::Float32, g_bench_device);

    for (auto _ : state) {
        auto q_tensor = quantize_per_tensor_asymmetric(input, QuantDType::INT8);
        benchmark::DoNotOptimize(q_tensor);
    }

    auto q_tensor = quantize_per_tensor_asymmetric(input, QuantDType::INT8);
    float error = std::get<0>(compute_quantization_error(input, q_tensor));

    state.counters["QuantError"] = error;
    state.SetBytesProcessed(state.iterations() * input.numel() * sizeof(float));
}
BENCHMARK(BM_Quantization_PerTensorAsymmetric)->Unit(benchmark::kMillisecond);

static void BM_Quantization_PerChannel(benchmark::State& state) {
    Tensor input = randn({128, 512, 7, 7}, DType::Float32, g_bench_device);
    int64_t axis = 1; // quantize per output channel

    for (auto _ : state) {
        auto q_tensor = quantize_per_channel_symmetric(input, axis, QuantDType::INT8);
        benchmark::DoNotOptimize(q_tensor);
    }

    auto q_tensor = quantize_per_channel_symmetric(input, axis, QuantDType::INT8);
    float error = std::get<0>(compute_quantization_error(input, q_tensor));

    state.counters["QuantError"] = error;
}
BENCHMARK(BM_Quantization_PerChannel)->Unit(benchmark::kMillisecond);

static void BM_Quantization_Observer_MinMax(benchmark::State& state) {
    auto observer = std::make_unique<MinMaxObserver>();

    for (auto _ : state) {
        for (int i = 0; i < 10; ++i) {
            Tensor input = randn({128, 256}, DType::Float32, g_bench_device);
            observer->observe(input);
        }
        observer->reset();
    }
}
BENCHMARK(BM_Quantization_Observer_MinMax)->Unit(benchmark::kMillisecond);

static void BM_Quantization_Observer_Histogram(benchmark::State& state) {
    int num_bins = state.range(0);

    for (auto _ : state) {
        auto observer = std::make_unique<HistogramObserver>(num_bins);
        for (int i = 0; i < 10; ++i) {
            Tensor input = randn({128, 256}, DType::Float32, g_bench_device);
            observer->observe(input);
        }
        auto params = observer->calculate_qparams(QuantDType::INT8,
                                                  QuantizationScheme::PerTensorSymmetric);
        benchmark::DoNotOptimize(params);
    }
}
BENCHMARK(BM_Quantization_Observer_Histogram)->Arg(128)->Arg(256)->Arg(512)->Unit(benchmark::kMillisecond);

static void BM_Quantization_FakeQuantize(benchmark::State& state) {
    auto fake_quant = std::make_shared<FakeQuantize>();
    Tensor input = randn({64, 128}, DType::Float32, g_bench_device);
    Variable var(input, true);

    for (auto _ : state) {
        auto output = fake_quant->forward(var);
        benchmark::DoNotOptimize(output);
    }
}
BENCHMARK(BM_Quantization_FakeQuantize)->Unit(benchmark::kMicrosecond);

// =============================================================================
// Combined Compression Benchmarks
// =============================================================================

static void BM_Compression_Combined(benchmark::State& state) {
    float prune_sparsity = state.range(0) / 100.0f;

    for (auto _ : state) {
        state.PauseTiming();
        auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);
        state.ResumeTiming();

        // Apply pruning
        auto config = prune_unstructured(model, prune_sparsity, ImportanceCriterion::L1);
        apply_pruning_masks(model, config);

        // Quantize weights
        auto params = model->parameters();
        for (auto& param : params) {
            auto q_tensor = quantize_per_tensor_symmetric(param->tensor(), QuantDType::INT8);
            benchmark::DoNotOptimize(q_tensor);
        }
    }

    // Report combined compression ratio
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);
    auto config = prune_unstructured(model, prune_sparsity, ImportanceCriterion::L1);
    apply_pruning_masks(model, config);

    float sparsity = compute_sparsity(model);
    float quant_ratio = 4.0f; // FP32 → INT8
    float combined_ratio = quant_ratio / (1.0f - sparsity);

    state.counters["Sparsity"] = sparsity;
    state.counters["CompressionRatio"] = combined_ratio;
}
BENCHMARK(BM_Compression_Combined)->Arg(50)->Arg(70)->Arg(90)->Unit(benchmark::kMillisecond);

// =============================================================================
// Memory Footprint Benchmarks
// =============================================================================

static void BM_MemoryFootprint_Baseline(benchmark::State& state) {
    int num_params = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        Tensor weights = randn({num_params}, DType::Float32, g_bench_device);
        state.ResumeTiming();

        size_t fp32_bytes = weights.numel() * sizeof(float);
        benchmark::DoNotOptimize(fp32_bytes);
    }

    state.counters["MemoryMB"] = (state.range(0) * sizeof(float)) / (1024.0 * 1024.0);
}
BENCHMARK(BM_MemoryFootprint_Baseline)->Range(1<<20, 1<<24)->Unit(benchmark::kMicrosecond);

static void BM_MemoryFootprint_Quantized(benchmark::State& state) {
    int num_params = state.range(0);

    for (auto _ : state) {
        state.PauseTiming();
        Tensor weights = randn({num_params}, DType::Float32, g_bench_device);
        state.ResumeTiming();

        auto q_tensor = quantize_per_tensor_symmetric(weights, QuantDType::INT8);
        size_t int8_bytes = q_tensor.data().numel() * sizeof(int8_t);
        benchmark::DoNotOptimize(int8_bytes);
    }

    state.counters["MemoryMB"] = (state.range(0) * sizeof(int8_t)) / (1024.0 * 1024.0);
    state.counters["CompressionRatio"] = 4.0f;
}
BENCHMARK(BM_MemoryFootprint_Quantized)->Range(1<<20, 1<<24)->Unit(benchmark::kMicrosecond);

// =============================================================================
// Inference Speed Benchmarks
// =============================================================================

static void BM_Inference_FP32_Baseline(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);
    Tensor input = randn({32, 784}, DType::Float32, g_bench_device);
    Variable var(input, false);

    for (auto _ : state) {
        auto output = model->forward(var);
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * 32); // batch size
}
BENCHMARK(BM_Inference_FP32_Baseline)->Unit(benchmark::kMicrosecond);

static void BM_Inference_Pruned_50Percent(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);
    auto config = prune_unstructured(model, 0.5f, ImportanceCriterion::L1);
    apply_pruning_masks(model, config);

    Tensor input = randn({32, 784}, DType::Float32, g_bench_device);
    Variable var(input, false);

    for (auto _ : state) {
        auto output = model->forward(var);
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * 32);
    state.counters["Sparsity"] = 0.5f;
}
BENCHMARK(BM_Inference_Pruned_50Percent)->Unit(benchmark::kMicrosecond);

static void BM_Inference_Pruned_90Percent(benchmark::State& state) {
    auto model = std::make_shared<BenchmarkModel>(784, 512, 3, 10);
    auto config = prune_unstructured(model, 0.9f, ImportanceCriterion::L1);
    apply_pruning_masks(model, config);

    Tensor input = randn({32, 784}, DType::Float32, g_bench_device);
    Variable var(input, false);

    for (auto _ : state) {
        auto output = model->forward(var);
        benchmark::DoNotOptimize(output);
    }

    state.SetItemsProcessed(state.iterations() * 32);
    state.counters["Sparsity"] = 0.9f;
}
BENCHMARK(BM_Inference_Pruned_90Percent)->Unit(benchmark::kMicrosecond);

// RR.19 (audit-11): custom main() so we can parse the --device flag
// before Google Benchmark consumes argv. Mirrors the parse_device_arg
// pattern used by every other benchmark binary in this directory.
int main(int argc, char** argv) {
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);
    tenzor::initialize();
    std::cout << "[benchmark_compression] device="
              << g_bench_device.to_string() << "\n";
    ::benchmark::Initialize(&argc, argv);
    if (::benchmark::ReportUnrecognizedArguments(argc, argv)) {
        // --device/--device-id were already consumed; Google Benchmark
        // will warn about any other unrecognized flags. Continue anyway
        // so a typo on the runner's side doesn't kill the binary.
    }
    ::benchmark::RunSpecifiedBenchmarks();
    ::benchmark::Shutdown();
    return 0;
}
