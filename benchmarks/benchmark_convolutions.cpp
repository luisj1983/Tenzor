/**
 * @file benchmark_convolutions.cpp
 * @brief Comprehensive benchmark for convolution operations
 *
 * Benchmarks:
 * - Conv2d (various kernel sizes and channels)
 * - ResNet50 layer configurations
 * - Depthwise/Separable convolutions
 * - Pooling operations
 */

#include "tenzor/tenzor.hpp"
#include "common.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <sstream>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

// Global device parsed from argv in main(). Defaults to CPU so unflagged
// invocations stay correct. Previously this file never read argv at all, so
// every randn()/module construction was hardcoded to CPU regardless of the
// --device flag the runner passed.
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

constexpr size_t WARMUP_ITERATIONS = 3;
constexpr size_t BENCHMARK_ITERATIONS = 30;

// Global results collector for JSON output
static std::vector<BenchmarkResult> g_all_results;
static void collect_result(const BenchmarkResult& result) {
    g_all_results.push_back(result);
}

/**
 * @brief Benchmark Conv2d with various configurations
 */
void benchmark_conv2d_basic() {
    std::cout << "\n========================================\n";
    std::cout << "  Conv2d Basic Benchmarks\n";
    std::cout << "========================================\n\n";

    struct ConvConfig {
        int64_t batch;
        int64_t in_channels;
        int64_t out_channels;
        int64_t input_h;
        int64_t input_w;
        int64_t kernel_size;
        int64_t stride;
        std::string name;
    };

    std::vector<ConvConfig> configs = {
        // Small convolutions
        {1, 3, 64, 224, 224, 3, 1, "Conv2d 3x3 (224x224, 3->64)"},
        {1, 64, 64, 56, 56, 3, 1, "Conv2d 3x3 (56x56, 64->64)"},
        {1, 128, 128, 28, 28, 3, 1, "Conv2d 3x3 (28x28, 128->128)"},
        {1, 256, 256, 14, 14, 3, 1, "Conv2d 3x3 (14x14, 256->256)"},
        {1, 512, 512, 7, 7, 3, 1, "Conv2d 3x3 (7x7, 512->512)"},

        // Different kernel sizes
        {1, 64, 64, 56, 56, 1, 1, "Conv2d 1x1 (56x56, 64->64)"},
        {1, 64, 64, 56, 56, 5, 1, "Conv2d 5x5 (56x56, 64->64)"},
        {1, 64, 64, 56, 56, 7, 1, "Conv2d 7x7 (56x56, 64->64)"},

        // Strided convolutions
        {1, 3, 64, 224, 224, 7, 2, "Conv2d 7x7/2 (224x224, 3->64)"},
        {1, 64, 128, 56, 56, 3, 2, "Conv2d 3x3/2 (56x56, 64->128)"},

        // Batch processing
        {8, 64, 64, 56, 56, 3, 1, "Conv2d 3x3 Batch=8 (56x56, 64->64)"},
        {16, 64, 64, 56, 56, 3, 1, "Conv2d 3x3 Batch=16 (56x56, 64->64)"},
        {32, 64, 64, 56, 56, 3, 1, "Conv2d 3x3 Batch=32 (56x56, 64->64)"},
    };

    for (const auto& cfg : configs) {
        // Create conv layer
        auto conv = Conv2d(cfg.in_channels, cfg.out_channels, cfg.kernel_size,
                          cfg.stride, cfg.kernel_size / 2);
        conv.to(g_bench_device);

        // Create input
        auto input = randn({cfg.batch, cfg.in_channels, cfg.input_h, cfg.input_w}, DType::Float32, g_bench_device);

        // Calculate output dimensions
        int64_t out_h = (cfg.input_h + 2 * (cfg.kernel_size / 2) - cfg.kernel_size) / cfg.stride + 1;
        int64_t out_w = (cfg.input_w + 2 * (cfg.kernel_size / 2) - cfg.kernel_size) / cfg.stride + 1;

        // Calculate FLOPs
        size_t flops_count = flops::conv2d(
            cfg.batch, out_h, out_w,
            cfg.in_channels, cfg.out_channels,
            cfg.kernel_size, cfg.kernel_size
        );

        // Setup benchmark
        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(flops_count);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto input_var = Variable(input, false); auto output = conv.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark ResNet50 layer configurations
 */
void benchmark_resnet50_layers() {
    std::cout << "\n========================================\n";
    std::cout << "  ResNet50 Layer Benchmarks\n";
    std::cout << "========================================\n\n";
    std::cout << "Target: < 1ms per layer (PyTorch: 1.2ms)\n\n";

    struct ResNetLayer {
        int64_t in_ch;
        int64_t out_ch;
        int64_t h;
        int64_t w;
        int64_t kernel;
        int64_t stride;
        std::string name;
    };

    // ResNet50 layer configurations
    std::vector<ResNetLayer> layers = {
        // Conv1
        {3, 64, 224, 224, 7, 2, "ResNet50-Conv1"},

        // Conv2_x
        {64, 64, 56, 56, 1, 1, "ResNet50-Conv2_x-1x1"},
        {64, 64, 56, 56, 3, 1, "ResNet50-Conv2_x-3x3"},
        {64, 256, 56, 56, 1, 1, "ResNet50-Conv2_x-expand"},

        // Conv3_x
        {256, 128, 56, 56, 1, 2, "ResNet50-Conv3_x-downsample"},
        {128, 128, 28, 28, 3, 1, "ResNet50-Conv3_x-3x3"},
        {128, 512, 28, 28, 1, 1, "ResNet50-Conv3_x-expand"},

        // Conv4_x
        {512, 256, 28, 28, 1, 2, "ResNet50-Conv4_x-downsample"},
        {256, 256, 14, 14, 3, 1, "ResNet50-Conv4_x-3x3"},
        {256, 1024, 14, 14, 1, 1, "ResNet50-Conv4_x-expand"},

        // Conv5_x
        {1024, 512, 14, 14, 1, 2, "ResNet50-Conv5_x-downsample"},
        {512, 512, 7, 7, 3, 1, "ResNet50-Conv5_x-3x3"},
        {512, 2048, 7, 7, 1, 1, "ResNet50-Conv5_x-expand"},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& layer : layers) {
        auto conv = Conv2d(layer.in_ch, layer.out_ch, layer.kernel,
                          layer.stride, layer.kernel / 2);
        conv.to(g_bench_device);
        auto input = randn({1, layer.in_ch, layer.h, layer.w}, DType::Float32, g_bench_device);

        int64_t out_h = (layer.h + 2 * (layer.kernel / 2) - layer.kernel) / layer.stride + 1;
        int64_t out_w = (layer.w + 2 * (layer.kernel / 2) - layer.kernel) / layer.stride + 1;

        size_t flops_count = flops::conv2d(
            1, out_h, out_w,
            layer.in_ch, layer.out_ch,
            layer.kernel, layer.kernel
        );

        Benchmark bench(layer.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(flops_count);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto input_var = Variable(input, false); auto output = conv.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
        results.push_back(result);
    }

    // Print summary with comparison to target
    std::cout << "\n========================================\n";
    std::cout << "  ResNet50 Layer Summary\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(30) << "Layer"
              << std::right << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "Target"
              << std::setw(12) << "Status"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& result : results) {
        double mean_ms = result.stats.mean * 1000.0;
        std::string status = (mean_ms < 1.0) ? "PASS" : "NEEDS OPT";

        std::cout << std::left << std::setw(30) << result.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << mean_ms
                  << std::setw(12) << "< 1.0ms"
                  << std::setw(12) << status
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark pooling operations
 */
void benchmark_pooling() {
    std::cout << "\n========================================\n";
    std::cout << "  Pooling Operations\n";
    std::cout << "========================================\n\n";

    struct PoolConfig {
        int64_t batch;
        int64_t channels;
        int64_t h;
        int64_t w;
        int64_t kernel;
        int64_t stride;
        std::string name;
    };

    std::vector<PoolConfig> configs = {
        {1, 64, 112, 112, 2, 2, "MaxPool2d 2x2/2 (112x112, 64ch)"},
        {1, 128, 56, 56, 2, 2, "MaxPool2d 2x2/2 (56x56, 128ch)"},
        {1, 256, 28, 28, 2, 2, "MaxPool2d 2x2/2 (28x28, 256ch)"},
        {1, 512, 14, 14, 2, 2, "MaxPool2d 2x2/2 (14x14, 512ch)"},

        {1, 64, 112, 112, 3, 2, "MaxPool2d 3x3/2 (112x112, 64ch)"},
        {1, 128, 56, 56, 3, 2, "MaxPool2d 3x3/2 (56x56, 128ch)"},

        {8, 64, 112, 112, 2, 2, "MaxPool2d 2x2/2 Batch=8 (112x112, 64ch)"},
        {16, 64, 112, 112, 2, 2, "MaxPool2d 2x2/2 Batch=16 (112x112, 64ch)"},
    };

    for (const auto& cfg : configs) {
        auto pool = MaxPool2d(cfg.kernel, cfg.stride);
        pool.to(g_bench_device);
        auto input = randn({cfg.batch, cfg.channels, cfg.h, cfg.w}, DType::Float32, g_bench_device);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto input_var = Variable(input, false); auto output = pool.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark average pooling
 */
void benchmark_avgpool() {
    std::cout << "\n========================================\n";
    std::cout << "  Average Pooling Operations\n";
    std::cout << "========================================\n\n";

    std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>> configs = {
        {64, 112, 112, "AvgPool2d 2x2/2 (112x112, 64ch)"},
        {128, 56, 56, "AvgPool2d 2x2/2 (56x56, 128ch)"},
        {256, 28, 28, "AvgPool2d 2x2/2 (28x28, 256ch)"},
        {512, 7, 7, "AvgPool2d Global (7x7, 512ch)"},
    };

    for (const auto& [channels, h, w, name] : configs) {
        auto pool = AvgPool2d(2, 2);
        pool.to(g_bench_device);
        auto input = randn({1, channels, h, w}, DType::Float32, g_bench_device);

        Benchmark bench(name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto input_var = Variable(input, false); auto output = pool.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

int main(int argc, char** argv) {
    bool json_output = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            json_output = true;
        }
    }
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);

    if (!json_output) {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Tenzor Convolution Benchmark Suite\n";
        std::cout << "  device=" << g_bench_device.to_string() << "\n";
        std::cout << "========================================\n";
        std::cout << "\nTarget Performance Metrics:\n";
        std::cout << "  Conv2d (ResNet50):  < 1ms/layer (PyTorch: 1.2ms)\n";
        std::cout << "\n";
    }

    try {
        initialize();

        std::streambuf* original_buf = nullptr;
        std::ostringstream null_stream;
        if (json_output) {
            original_buf = std::cout.rdbuf(null_stream.rdbuf());
        }

        benchmark_conv2d_basic();
        benchmark_resnet50_layers();
        benchmark_pooling();
        benchmark_avgpool();

        if (json_output && original_buf) {
            std::cout.rdbuf(original_buf);
        }

        if (json_output) {
            BenchmarkSuite suite("convolutions");
            std::cout << suite.export_json(g_all_results);
        } else {
            std::cout << "\n========================================\n";
            std::cout << "  Convolution Benchmark Complete\n";
            std::cout << "========================================\n\n";
        }

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
