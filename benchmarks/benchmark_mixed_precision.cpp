/**
 * @file benchmark_mixed_precision.cpp
 * @brief Comprehensive benchmark for mixed precision and quantization performance
 *
 * Benchmarks critical for modern ML inference and training:
 * - FP32 vs FP16 vs BF16 operations
 * - Automatic Mixed Precision (AMP) training
 * - INT8 quantized inference
 * - Tensor Core utilization (when available)
 * - Memory savings from reduced precision
 */

#include "tenzor/tenzor.hpp"
#include "common.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/amp/autocast.hpp"
#include "tenzor/nn/amp/grad_scaler.hpp"
#include "tenzor/nn/quantization/quantize.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

// Global device parsed from argv in main(). Defaults to CPU so unflagged
// invocations stay correct. Previously this file never read argv at all, so
// every randn()/module construction was hardcoded to CPU regardless of the
// --device flag the runner passed — a run_benchmarks.py "device=cuda" pass
// silently benchmarked the CPU backend instead (see benchmark_attention.cpp
// for the same class of bug, which additionally crashed outright).
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

constexpr size_t WARMUP_ITERATIONS = 5;
constexpr size_t BENCHMARK_ITERATIONS = 50;

/**
 * @brief Simple MLP for precision benchmarking
 */
class BenchmarkMLP : public Module {
public:
    BenchmarkMLP(int64_t in_features, int64_t hidden, int64_t out_features)
        : fc1_(std::make_shared<Linear>(in_features, hidden)),
          fc2_(std::make_shared<Linear>(hidden, hidden)),
          fc3_(std::make_shared<Linear>(hidden, out_features)) {
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc3", fc3_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = relu(fc1_->forward(x));
        h = relu(fc2_->forward(h));
        return fc3_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_, fc2_, fc3_;
};

/**
 * @brief Benchmark matrix multiplication at different precisions
 */
void benchmark_matmul_precision() {
    std::cout << "\n========================================\n";
    std::cout << "  MatMul Precision Comparison\n";
    std::cout << "========================================\n\n";

    struct MatMulConfig {
        int64_t M, N, K;
        std::string name;
    };

    std::vector<MatMulConfig> configs = {
        {1024, 1024, 1024, "1024x1024x1024"},
        {2048, 2048, 2048, "2048x2048x2048"},
        {4096, 4096, 4096, "4096x4096x4096"},
        {8192, 8192, 8192, "8192x8192x8192"},
    };

    for (const auto& cfg : configs) {
        std::cout << "Matrix size: " << cfg.name << "\n";
        std::cout << std::string(50, '-') << "\n";

        size_t flops_count = flops::matmul(cfg.M, cfg.N, cfg.K);

        // FP32
        {
            auto a = randn({cfg.M, cfg.K}, DType::Float32, g_bench_device);
            auto b = randn({cfg.K, cfg.N}, DType::Float32, g_bench_device);

            Benchmark bench("FP32", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops_count);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = matmul(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });

            result.print();
        }

        // FP16
        {
            auto a = randn({cfg.M, cfg.K}, DType::Float16, g_bench_device);
            auto b = randn({cfg.K, cfg.N}, DType::Float16, g_bench_device);

            Benchmark bench("FP16", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops_count);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = matmul(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });

            result.print();
        }

        // BF16 (if supported)
        try {
            auto a = randn({cfg.M, cfg.K}, DType::BFloat16, g_bench_device);
            auto b = randn({cfg.K, cfg.N}, DType::BFloat16, g_bench_device);

            Benchmark bench("BF16", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_flops(flops_count);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = matmul(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });

            result.print();
        } catch (const std::exception& e) {
            std::cout << "  BF16:     Not supported on this platform\n";
        }

        std::cout << "\n";
    }
}

/**
 * @brief Benchmark Linear layer at different precisions
 */
void benchmark_linear_precision() {
    std::cout << "\n========================================\n";
    std::cout << "  Linear Layer Precision Comparison\n";
    std::cout << "========================================\n\n";

    struct LinearConfig {
        int64_t batch;
        int64_t in_features;
        int64_t out_features;
        std::string name;
    };

    std::vector<LinearConfig> configs = {
        {32, 768, 3072, "BERT FFN up (32, 768->3072)"},
        {32, 3072, 768, "BERT FFN down (32, 3072->768)"},
        {16, 4096, 11008, "Llama FFN up (16, 4096->11008)"},
        {16, 11008, 4096, "Llama FFN down (16, 11008->4096)"},
    };

    for (const auto& cfg : configs) {
        std::cout << cfg.name << ":\n";

        // FP32
        {
            auto linear = Linear(cfg.in_features, cfg.out_features);
            linear.to(g_bench_device);
            auto input = randn({cfg.batch, cfg.in_features}, DType::Float32, g_bench_device);
            auto input_var = Variable(input, false);

            Benchmark bench("  FP32", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            auto result = bench.set_device(g_bench_device).run([&]() {
                auto output = linear.forward(input_var);
                volatile void* ptr = output.tensor().data_ptr();
                (void)ptr;
            });
            result.print();
        }

        // FP16
        {
            auto linear = Linear(cfg.in_features, cfg.out_features);
            linear.to(g_bench_device);
            auto input = randn({cfg.batch, cfg.in_features}, DType::Float16, g_bench_device);
            auto input_var = Variable(input, false);

            Benchmark bench("  FP16", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            auto result = bench.set_device(g_bench_device).run([&]() {
                auto output = linear.forward(input_var);
                volatile void* ptr = output.tensor().data_ptr();
                (void)ptr;
            });
            result.print();
        }

        std::cout << "\n";
    }
}

/**
 * @brief Benchmark Conv2d at different precisions
 */
void benchmark_conv2d_precision() {
    std::cout << "\n========================================\n";
    std::cout << "  Conv2d Precision Comparison\n";
    std::cout << "========================================\n\n";

    struct ConvConfig {
        int64_t batch;
        int64_t in_ch;
        int64_t out_ch;
        int64_t h, w;
        int64_t kernel;
        std::string name;
    };

    std::vector<ConvConfig> configs = {
        {32, 64, 128, 56, 56, 3, "ResNet stage (32, 64->128, 56x56)"},
        {32, 256, 512, 14, 14, 3, "ResNet deep (32, 256->512, 14x14)"},
        {64, 3, 64, 224, 224, 7, "First conv (64, 3->64, 224x224)"},
    };

    for (const auto& cfg : configs) {
        std::cout << cfg.name << ":\n";

        // FP32
        {
            auto conv = Conv2d(cfg.in_ch, cfg.out_ch, cfg.kernel, 1, cfg.kernel / 2);
            conv.to(g_bench_device);
            auto input = randn({cfg.batch, cfg.in_ch, cfg.h, cfg.w}, DType::Float32, g_bench_device);
            auto input_var = Variable(input, false);

            Benchmark bench("  FP32", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            auto result = bench.set_device(g_bench_device).run([&]() {
                auto output = conv.forward(input_var);
                volatile void* ptr = output.tensor().data_ptr();
                (void)ptr;
            });
            result.print();
        }

        // FP16
        {
            auto conv = Conv2d(cfg.in_ch, cfg.out_ch, cfg.kernel, 1, cfg.kernel / 2);
            conv.to(g_bench_device);
            auto input = randn({cfg.batch, cfg.in_ch, cfg.h, cfg.w}, DType::Float16, g_bench_device);
            auto input_var = Variable(input, false);

            Benchmark bench("  FP16", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            auto result = bench.set_device(g_bench_device).run([&]() {
                auto output = conv.forward(input_var);
                volatile void* ptr = output.tensor().data_ptr();
                (void)ptr;
            });
            result.print();
        }

        std::cout << "\n";
    }
}

/**
 * @brief Benchmark AMP training simulation
 */
void benchmark_amp_training() {
    std::cout << "\n========================================\n";
    std::cout << "  AMP Training Simulation\n";
    std::cout << "========================================\n\n";

    const int64_t batch = 32;
    const int64_t in_features = 1024;
    const int64_t hidden = 2048;
    const int64_t out_features = 1000;

    auto model = std::make_shared<BenchmarkMLP>(in_features, hidden, out_features);
    model->to(g_bench_device);

    // FP32 training
    {
        auto input = randn({batch, in_features}, DType::Float32, g_bench_device);
        auto input_var = Variable(input, true);

        Benchmark bench("FP32 Training Step", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
        auto result = bench.set_device(g_bench_device).run([&]() {
            model->train();
            auto output = model->forward(input_var);
            auto loss = mean(output);
            loss.backward();

            // Simulate optimizer step
            for (auto& param : model->parameters()) {
                param->zero_grad();
            }
        });
        result.print();
    }

    // AMP training (FP16 forward, FP32 gradients)
    {
        auto input = randn({batch, in_features}, DType::Float16, g_bench_device);
        auto input_var = Variable(input, true);

        Benchmark bench("AMP Training Step", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
        auto result = bench.set_device(g_bench_device).run([&]() {
            model->train();

            // Forward in FP16
            auto output = model->forward(input_var);
            auto loss = mean(output);

            // Backward (would use loss scaling in real AMP)
            loss.backward();

            for (auto& param : model->parameters()) {
                param->zero_grad();
            }
        });
        result.print();
    }
}

/**
 * @brief Benchmark memory savings from reduced precision
 */
void benchmark_memory_precision() {
    std::cout << "\n========================================\n";
    std::cout << "  Memory Usage by Precision\n";
    std::cout << "========================================\n\n";

    std::vector<std::pair<int64_t, std::string>> sizes = {
        {1024 * 1024, "1M elements"},
        {16 * 1024 * 1024, "16M elements"},
        {64 * 1024 * 1024, "64M elements"},
        {256 * 1024 * 1024, "256M elements"},
    };

    std::cout << std::left << std::setw(20) << "Size"
              << std::setw(15) << "FP32 (MB)"
              << std::setw(15) << "FP16 (MB)"
              << std::setw(15) << "INT8 (MB)"
              << std::setw(15) << "Savings"
              << "\n";
    std::cout << std::string(80, '-') << "\n";

    for (const auto& [num_elements, name] : sizes) {
        double fp32_mb = (num_elements * 4) / (1024.0 * 1024.0);
        double fp16_mb = (num_elements * 2) / (1024.0 * 1024.0);
        double int8_mb = (num_elements * 1) / (1024.0 * 1024.0);

        std::cout << std::left << std::setw(20) << name
                  << std::fixed << std::setprecision(1)
                  << std::setw(15) << fp32_mb
                  << std::setw(15) << fp16_mb
                  << std::setw(15) << int8_mb
                  << std::setw(15) << "2x/4x"
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark precision conversion overhead
 */
void benchmark_precision_conversion() {
    std::cout << "\n========================================\n";
    std::cout << "  Precision Conversion Overhead\n";
    std::cout << "========================================\n\n";

    std::vector<int64_t> sizes = {1024 * 1024, 16 * 1024 * 1024, 64 * 1024 * 1024};

    for (auto size : sizes) {
        std::cout << "Size: " << (size / (1024 * 1024)) << "M elements\n";

        auto fp32_tensor = randn({size}, DType::Float32, g_bench_device);

        // FP32 -> FP16
        {
            Benchmark bench("  FP32 -> FP16", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_bytes(size * (4 + 2));  // read FP32, write FP16

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto fp16_tensor = fp32_tensor.to(DType::Float16);
                volatile void* ptr = fp16_tensor.data_ptr();
                (void)ptr;
            });
            result.print();
        }

        // FP16 -> FP32
        {
            auto fp16_tensor = fp32_tensor.to(DType::Float16);

            Benchmark bench("  FP16 -> FP32", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            bench.set_bytes(size * (2 + 4));

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto back_to_fp32 = fp16_tensor.to(DType::Float32);
                volatile void* ptr = back_to_fp32.data_ptr();
                (void)ptr;
            });
            result.print();
        }

        std::cout << "\n";
    }
}

/**
 * @brief Benchmark INT8 quantized operations
 */
void benchmark_int8_inference() {
    std::cout << "\n========================================\n";
    std::cout << "  INT8 Quantized Inference\n";
    std::cout << "========================================\n\n";

    struct QuantConfig {
        int64_t batch;
        int64_t in_features;
        int64_t out_features;
        std::string name;
    };

    std::vector<QuantConfig> configs = {
        {32, 768, 768, "BERT hidden (32, 768->768)"},
        {32, 768, 3072, "BERT FFN up (32, 768->3072)"},
        {16, 4096, 4096, "Llama hidden (16, 4096->4096)"},
    };

    for (const auto& cfg : configs) {
        std::cout << cfg.name << ":\n";

        // FP32 baseline
        {
            auto linear = Linear(cfg.in_features, cfg.out_features);
            linear.to(g_bench_device);
            auto input = randn({cfg.batch, cfg.in_features}, DType::Float32, g_bench_device);
            auto input_var = Variable(input, false);

            Benchmark bench("  FP32 baseline", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            auto result = bench.set_device(g_bench_device).run([&]() {
                auto output = linear.forward(input_var);
                volatile void* ptr = output.tensor().data_ptr();
                (void)ptr;
            });
            result.print();
        }

        // Simulated INT8 (using INT8 storage)
        {
            // Note: Full INT8 quantized linear requires quantization infrastructure
            auto input = randn({cfg.batch, cfg.in_features}, DType::Float32, g_bench_device);

            // Quantize input
            float scale = 127.0f / max(abs(input)).item<float>();

            Benchmark bench("  Quantize+Dequant", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            auto result = bench.set_device(g_bench_device).run([&]() {
                // Simulate quantization overhead
                auto scaled = input * scale;
                auto rounded = round(scaled);
                auto clamped = clamp(rounded, -127.0f, 127.0f);

                // Dequantize
                auto dequant = clamped / scale;
                volatile void* ptr = dequant.data_ptr();
                (void)ptr;
            });
            result.print();
        }

        std::cout << "\n";
    }
}

/**
 * @brief Benchmark element-wise operations at different precisions
 */
void benchmark_elementwise_precision() {
    std::cout << "\n========================================\n";
    std::cout << "  Element-wise Ops Precision\n";
    std::cout << "========================================\n\n";

    const int64_t size = 16 * 1024 * 1024;  // 16M elements

    std::vector<std::pair<std::string, DType>> dtypes = {
        {"FP32", DType::Float32},
        {"FP16", DType::Float16},
    };

    for (const auto& [name, dtype] : dtypes) {
        std::cout << name << ":\n";

        auto a = randn({size}, dtype, g_bench_device);
        auto b = randn({size}, dtype, g_bench_device);

        // Addition
        {
            Benchmark bench("  Add", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            size_t bytes = size * dtype_size(dtype) * 3;  // 2 reads + 1 write
            bench.set_bytes(bytes);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = add(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
        }

        // Multiplication
        {
            Benchmark bench("  Mul", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            size_t bytes = size * dtype_size(dtype) * 3;
            bench.set_bytes(bytes);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = mul(a, b);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
        }

        // Exp
        {
            Benchmark bench("  Exp", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
            size_t bytes = size * dtype_size(dtype) * 2;
            bench.set_bytes(bytes);

            auto result = bench.set_device(g_bench_device).run([&]() {
                auto c = exp(a);
                volatile void* ptr = c.data_ptr();
                (void)ptr;
            });
            result.print();
        }

        std::cout << "\n";
    }
}

int main(int argc, char** argv) {
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor Mixed Precision Benchmark\n";
    std::cout << "  device=" << g_bench_device.to_string() << "\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  FP16 MatMul:      1.5-2x faster than FP32\n";
    std::cout << "  BF16 MatMul:      Similar to FP16 with better numerics\n";
    std::cout << "  AMP Training:     1.3-1.5x faster than FP32\n";
    std::cout << "  INT8 Inference:   2-4x faster than FP32\n";
    std::cout << "  Memory:           2x savings with FP16, 4x with INT8\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_matmul_precision();
        benchmark_linear_precision();
        benchmark_conv2d_precision();
        benchmark_amp_training();
        benchmark_memory_precision();
        benchmark_precision_conversion();
        benchmark_int8_inference();
        benchmark_elementwise_precision();

        std::cout << "\n========================================\n";
        std::cout << "  Mixed Precision Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
