/**
 * @file benchmark_normalization.cpp
 * @brief Comprehensive benchmark for normalization layers
 *
 * Benchmarks critical normalization operations:
 * - BatchNorm1d/2d (training and inference modes)
 * - LayerNorm (transformer-style)
 * - GroupNorm
 * - InstanceNorm
 * - RMSNorm (Llama-style)
 * - Fused normalization operations
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/utils/benchmark.hpp"
#include "tenzor/ops/fused_ops.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

constexpr size_t WARMUP_ITERATIONS = 5;
constexpr size_t BENCHMARK_ITERATIONS = 100;

/**
 * @brief Benchmark BatchNorm2d in training mode
 */
void benchmark_batchnorm2d_training() {
    std::cout << "\n========================================\n";
    std::cout << "  BatchNorm2d Training Mode\n";
    std::cout << "========================================\n\n";

    struct BNConfig {
        int64_t batch;
        int64_t channels;
        int64_t height;
        int64_t width;
        std::string name;
    };

    std::vector<BNConfig> configs = {
        // ResNet configurations
        {32, 64, 56, 56, "ResNet stage1 (32, 64, 56, 56)"},
        {32, 128, 28, 28, "ResNet stage2 (32, 128, 28, 28)"},
        {32, 256, 14, 14, "ResNet stage3 (32, 256, 14, 14)"},
        {32, 512, 7, 7, "ResNet stage4 (32, 512, 7, 7)"},

        // Large batch
        {128, 64, 56, 56, "Large batch (128, 64, 56, 56)"},

        // Mobile configurations
        {32, 32, 112, 112, "MobileNet early (32, 32, 112, 112)"},
        {32, 320, 7, 7, "MobileNet late (32, 320, 7, 7)"},
    };

    for (const auto& cfg : configs) {
        auto bn = BatchNorm2d(cfg.channels);
        bn.train();  // Training mode

        auto input = randn({cfg.batch, cfg.channels, cfg.height, cfg.width});
        auto input_var = Variable(input, true);

        size_t num_elements = cfg.batch * cfg.channels * cfg.height * cfg.width;

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(num_elements * sizeof(float) * 3);  // read input, running stats; write output

        auto result = bench.run([&]() {
            auto output = bn.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark BatchNorm2d in inference mode
 */
void benchmark_batchnorm2d_inference() {
    std::cout << "\n========================================\n";
    std::cout << "  BatchNorm2d Inference Mode\n";
    std::cout << "========================================\n\n";

    struct BNConfig {
        int64_t batch;
        int64_t channels;
        int64_t height;
        int64_t width;
        std::string name;
    };

    std::vector<BNConfig> configs = {
        {1, 64, 224, 224, "Single image inference"},
        {32, 64, 56, 56, "Batch inference (32)"},
        {128, 256, 14, 14, "Large batch inference"},
    };

    for (const auto& cfg : configs) {
        auto bn = BatchNorm2d(cfg.channels);
        bn.eval();  // Inference mode

        auto input = randn({cfg.batch, cfg.channels, cfg.height, cfg.width});
        auto input_var = Variable(input, false);

        size_t num_elements = cfg.batch * cfg.channels * cfg.height * cfg.width;

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS * 2);
        bench.set_bytes(num_elements * sizeof(float) * 2);

        auto result = bench.run([&]() {
            auto output = bn.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark LayerNorm (critical for transformers)
 */
void benchmark_layernorm() {
    std::cout << "\n========================================\n";
    std::cout << "  LayerNorm Benchmarks\n";
    std::cout << "========================================\n\n";
    std::cout << "Target: Critical for transformer performance\n\n";

    struct LNConfig {
        int64_t batch;
        int64_t seq_len;
        int64_t hidden;
        std::string name;
    };

    std::vector<LNConfig> configs = {
        // BERT configurations
        {8, 512, 768, "BERT-base (8, 512, 768)"},
        {16, 128, 768, "BERT-base short (16, 128, 768)"},

        // GPT configurations
        {4, 1024, 768, "GPT-2 (4, 1024, 768)"},
        {4, 2048, 768, "GPT-2 long (4, 2048, 768)"},

        // Large models
        {2, 512, 1024, "Large model (2, 512, 1024)"},
        {1, 2048, 1280, "GPT-2 Large (1, 2048, 1280)"},
        {1, 4096, 768, "Very long seq (1, 4096, 768)"},

        // Llama configurations
        {4, 512, 4096, "Llama 7B hidden (4, 512, 4096)"},
        {2, 2048, 4096, "Llama 7B long (2, 2048, 4096)"},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& cfg : configs) {
        auto ln = LayerNorm({cfg.hidden});

        auto input = randn({cfg.batch, cfg.seq_len, cfg.hidden});
        auto input_var = Variable(input, false);

        size_t num_elements = cfg.batch * cfg.seq_len * cfg.hidden;
        // LayerNorm: read input + compute mean/var + normalize + scale/shift
        size_t bytes = num_elements * sizeof(float) * 3;

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.run([&]() {
            auto output = ln.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        results.push_back(result);
    }

    // Summary table
    std::cout << "\n  LayerNorm Throughput Summary:\n";
    std::cout << std::left << std::setw(35) << "Config"
              << std::right << std::setw(15) << "Bandwidth (GB/s)"
              << "\n";
    std::cout << std::string(50, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(35) << result.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(15) << result.bandwidth_gbs
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark GroupNorm
 */
void benchmark_groupnorm() {
    std::cout << "\n========================================\n";
    std::cout << "  GroupNorm Benchmarks\n";
    std::cout << "========================================\n\n";

    struct GNConfig {
        int64_t batch;
        int64_t channels;
        int64_t height;
        int64_t width;
        int64_t groups;
        std::string name;
    };

    std::vector<GNConfig> configs = {
        {32, 256, 14, 14, 32, "groups=32, channels=256"},
        {32, 512, 7, 7, 32, "groups=32, channels=512"},
        {32, 256, 28, 28, 8, "groups=8, channels=256"},
        {16, 1024, 7, 7, 32, "groups=32, channels=1024"},
    };

    for (const auto& cfg : configs) {
        auto gn = GroupNorm(cfg.groups, cfg.channels);

        auto input = randn({cfg.batch, cfg.channels, cfg.height, cfg.width});
        auto input_var = Variable(input, false);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.run([&]() {
            auto output = gn.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark RMSNorm (used in Llama, etc.)
 */
void benchmark_rmsnorm() {
    std::cout << "\n========================================\n";
    std::cout << "  RMSNorm Benchmarks (Llama-style)\n";
    std::cout << "========================================\n\n";

    struct RMSConfig {
        int64_t batch;
        int64_t seq_len;
        int64_t hidden;
        std::string name;
    };

    std::vector<RMSConfig> configs = {
        {4, 512, 4096, "Llama 7B (4, 512, 4096)"},
        {2, 2048, 4096, "Llama 7B long (2, 2048, 4096)"},
        {1, 4096, 4096, "Llama 7B very long"},
        {4, 512, 5120, "Llama 13B (4, 512, 5120)"},
    };

    for (const auto& cfg : configs) {
        auto rms = RMSNorm(cfg.hidden);

        auto input = randn({cfg.batch, cfg.seq_len, cfg.hidden});
        auto input_var = Variable(input, false);

        size_t num_elements = cfg.batch * cfg.seq_len * cfg.hidden;
        size_t bytes = num_elements * sizeof(float) * 2;  // read + write

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.run([&]() {
            auto output = rms.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Compare LayerNorm vs RMSNorm performance
 */
void benchmark_norm_comparison() {
    std::cout << "\n========================================\n";
    std::cout << "  LayerNorm vs RMSNorm Comparison\n";
    std::cout << "========================================\n\n";

    const int64_t batch = 4;
    const int64_t seq_len = 1024;

    std::vector<int64_t> hidden_sizes = {768, 1024, 2048, 4096};

    std::cout << std::left << std::setw(15) << "Hidden Size"
              << std::setw(15) << "LayerNorm"
              << std::setw(15) << "RMSNorm"
              << std::setw(15) << "Speedup"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (auto hidden : hidden_sizes) {
        auto ln = LayerNorm({hidden});
        auto rms = RMSNorm(hidden);

        auto input = randn({batch, seq_len, hidden});
        auto input_var = Variable(input, false);

        // LayerNorm
        Benchmark bench_ln("LN h=" + std::to_string(hidden), 5, 50);
        auto result_ln = bench_ln.run([&]() {
            auto output = ln.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        // RMSNorm
        Benchmark bench_rms("RMS h=" + std::to_string(hidden), 5, 50);
        auto result_rms = bench_rms.run([&]() {
            auto output = rms.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double speedup = result_ln.stats.mean / result_rms.stats.mean;

        std::cout << std::left << std::setw(15) << hidden
                  << std::fixed << std::setprecision(3)
                  << std::setw(15) << (result_ln.stats.mean * 1000.0)
                  << std::setw(15) << (result_rms.stats.mean * 1000.0)
                  << std::setw(15) << speedup
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark fused BatchNorm + ReLU
 */
void benchmark_fused_batchnorm_relu() {
    std::cout << "\n========================================\n";
    std::cout << "  Fused BatchNorm + ReLU\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t batch;
        int64_t channels;
        int64_t height;
        int64_t width;
        std::string name;
    };

    std::vector<Config> configs = {
        {32, 64, 56, 56, "ResNet early (32, 64, 56, 56)"},
        {32, 256, 14, 14, "ResNet mid (32, 256, 14, 14)"},
        {32, 512, 7, 7, "ResNet late (32, 512, 7, 7)"},
    };

    for (const auto& cfg : configs) {
        auto input = randn({cfg.batch, cfg.channels, cfg.height, cfg.width});
        auto mean = zeros({cfg.channels});
        auto var = ones({cfg.channels});
        auto gamma = ones({cfg.channels});
        auto beta = zeros({cfg.channels});

        // Unfused version
        auto bn = BatchNorm2d(cfg.channels);
        bn.eval();

        Benchmark bench_unfused("Unfused " + cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_unfused = bench_unfused.run([&]() {
            auto bn_out = bn.forward(Variable(input, false));
            auto relu_out = relu(bn_out);
            volatile void* ptr = relu_out.tensor().data_ptr();
            (void)ptr;
        });
        result_unfused.print();

        // Fused version
        Benchmark bench_fused("Fused " + cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_fused = bench_fused.run([&]() {
            auto output = ops::fused_batchnorm_relu(input, mean, var, gamma, beta);
            volatile void* ptr = output.data_ptr();
            (void)ptr;
        });
        result_fused.print();

        double speedup = result_unfused.stats.mean / result_fused.stats.mean;
        std::cout << "  Speedup: " << std::fixed << std::setprecision(2)
                  << speedup << "x\n\n";
    }
}

/**
 * @brief Benchmark normalization backward passes
 */
void benchmark_norm_backward() {
    std::cout << "\n========================================\n";
    std::cout << "  Normalization Backward Pass\n";
    std::cout << "========================================\n\n";

    // LayerNorm backward
    {
        const int64_t batch = 8;
        const int64_t seq = 512;
        const int64_t hidden = 768;

        auto ln = LayerNorm({hidden});
        auto input = randn({batch, seq, hidden});
        auto input_var = Variable(input, true);

        Benchmark bench_fwd("LayerNorm Forward", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_fwd = bench_fwd.run([&]() {
            auto output = ln.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_fwd.print();

        Benchmark bench_bwd("LayerNorm Forward+Backward", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
        auto result_bwd = bench_bwd.run([&]() {
            auto output = ln.forward(input_var);
            auto grad = ones_like(output.tensor());
            output.backward(grad);
            input_var.zero_grad();
        });
        result_bwd.print();

        double ratio = result_bwd.stats.mean / result_fwd.stats.mean;
        std::cout << "  Backward/Forward ratio: " << std::fixed << std::setprecision(2)
                  << ratio << "x\n\n";
    }

    // BatchNorm2d backward
    {
        const int64_t batch = 32;
        const int64_t channels = 256;
        const int64_t height = 14;
        const int64_t width = 14;

        auto bn = BatchNorm2d(channels);
        bn.train();
        auto input = randn({batch, channels, height, width});
        auto input_var = Variable(input, true);

        Benchmark bench_fwd("BatchNorm2d Forward", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_fwd = bench_fwd.run([&]() {
            auto output = bn.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_fwd.print();

        Benchmark bench_bwd("BatchNorm2d Forward+Backward", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
        auto result_bwd = bench_bwd.run([&]() {
            auto output = bn.forward(input_var);
            auto grad = ones_like(output.tensor());
            output.backward(grad);
            input_var.zero_grad();
        });
        result_bwd.print();

        double ratio = result_bwd.stats.mean / result_fwd.stats.mean;
        std::cout << "  Backward/Forward ratio: " << std::fixed << std::setprecision(2)
                  << ratio << "x\n\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor Normalization Benchmark Suite\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  BatchNorm2d inference:  Memory bandwidth limited\n";
    std::cout << "  LayerNorm:              Critical for transformers\n";
    std::cout << "  RMSNorm:                1.3-1.5x faster than LayerNorm\n";
    std::cout << "  Fused BN+ReLU:          1.5-2x speedup vs unfused\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_batchnorm2d_training();
        benchmark_batchnorm2d_inference();
        benchmark_layernorm();
        benchmark_groupnorm();
        benchmark_rmsnorm();
        benchmark_norm_comparison();
        benchmark_fused_batchnorm_relu();
        benchmark_norm_backward();

        std::cout << "\n========================================\n";
        std::cout << "  Normalization Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
