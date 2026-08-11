/**
 * @file benchmark_training.cpp
 * @brief Comprehensive benchmark for full training loops
 *
 * Benchmarks end-to-end training workflows including:
 * - Complete forward-backward passes
 * - Optimizer steps
 * - Small neural network training
 * - Batch processing
 * - Memory patterns during training
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/optim/sgd.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"
#include "tenzor/utils/benchmark.hpp"
#include "common.hpp"
#include <iostream>
#include <memory>
#include <iomanip>
#include <sstream>

// RR.19 (audit-11): global device parsed from argv in main(). Defaults to
// CPU. Replaces the previous "always-on-CPU" behaviour (model->to(device)
// + Variable inputs both created on the default CPU device) so the Python
// runner's --device flag actually selects the training-loop backend.
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

// Benchmark configuration
constexpr size_t WARMUP_ITERATIONS = 3;
constexpr size_t BENCHMARK_ITERATIONS = 20;

// Global results collector for JSON output
static std::vector<BenchmarkResult> g_all_results;
static void collect_result(const BenchmarkResult& result) {
    g_all_results.push_back(result);
}

/**
 * @brief Simple MLP model for benchmarking
 */
class SimpleMLP : public Module {
public:
    SimpleMLP(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : fc1_(std::make_shared<Linear>(input_size, hidden_size)),
          relu_(std::make_shared<ReLU>()),
          fc2_(std::make_shared<Linear>(hidden_size, output_size)) {
        register_module("fc1", fc1_);
        register_module("relu", relu_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = fc1_->forward(x);
        h = relu_->forward(h);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<Linear> fc2_;
};

/**
 * @brief Simple CNN model for benchmarking
 */
class SimpleCNN : public Module {
public:
    SimpleCNN(int64_t in_channels, int64_t num_classes)
        : conv1_(std::make_shared<Conv2d>(in_channels, 32, 3, 1, 1)),
          relu1_(std::make_shared<ReLU>()),
          conv2_(std::make_shared<Conv2d>(32, 64, 3, 1, 1)),
          relu2_(std::make_shared<ReLU>()),
          fc_(std::make_shared<Linear>(64 * 8 * 8, num_classes)) {
        register_module("conv1", conv1_);
        register_module("relu1", relu1_);
        register_module("conv2", conv2_);
        register_module("relu2", relu2_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Conv1 block
        auto h = conv1_->forward(x);
        h = relu1_->forward(h);

        // Max pool (2x2) - manually implemented for benchmark
        auto t = h.tensor();
        auto pooled1 = maxpool2d(t, 2, 2);
        h = Variable(pooled1, h.requires_grad());

        // Conv2 block
        h = conv2_->forward(h);
        h = relu2_->forward(h);

        // Max pool (2x2)
        t = h.tensor();
        auto pooled2 = maxpool2d(t, 2, 2);
        h = Variable(pooled2, h.requires_grad());

        // Flatten
        auto batch_size = h.tensor().shape()[0];
        t = h.tensor();
        auto flattened = t.reshape({batch_size, -1});
        h = Variable(flattened, h.requires_grad());

        // FC layer
        return fc_->forward(h);
    }

private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<ReLU> relu1_;
    std::shared_ptr<Conv2d> conv2_;
    std::shared_ptr<ReLU> relu2_;
    std::shared_ptr<Linear> fc_;

    // Simple max pooling helper
    auto maxpool2d(const Tensor& input, int64_t kernel_size, int64_t stride) -> Tensor {
        auto shape = input.shape();
        auto batch = shape[0];
        auto channels = shape[1];
        auto height = shape[2];
        auto width = shape[3];

        auto out_h = (height - kernel_size) / stride + 1;
        auto out_w = (width - kernel_size) / stride + 1;

        // Naive reference impl below does raw host-pointer arithmetic
        // (data<float>()), which segfaults on a GPU-resident tensor. Route
        // through a CPU copy and move the result back to the input's
        // original device — in production this would use an optimized
        // device kernel instead.
        auto input_cpu = input.device().type == Device::Type::CPU ? input : input.to(Device::cpu());
        auto output = zeros({batch, channels, out_h, out_w}, input.dtype(), Device::cpu());

        // Simple implementation - in production would use optimized kernel
        auto input_ptr = input_cpu.data<float>();
        auto output_ptr = output.data<float>();

        for (int64_t b = 0; b < batch; ++b) {
            for (int64_t c = 0; c < channels; ++c) {
                for (int64_t oh = 0; oh < out_h; ++oh) {
                    for (int64_t ow = 0; ow < out_w; ++ow) {
                        float max_val = -std::numeric_limits<float>::infinity();

                        for (int64_t kh = 0; kh < kernel_size; ++kh) {
                            for (int64_t kw = 0; kw < kernel_size; ++kw) {
                                auto ih = oh * stride + kh;
                                auto iw = ow * stride + kw;
                                auto in_idx = b * channels * height * width +
                                            c * height * width +
                                            ih * width + iw;
                                max_val = std::max(max_val, input_ptr[in_idx]);
                            }
                        }

                        auto out_idx = b * channels * out_h * out_w +
                                     c * out_h * out_w +
                                     oh * out_w + ow;
                        output_ptr[out_idx] = max_val;
                    }
                }
            }
        }

        return input.device().type == Device::Type::CPU ? output : output.to(input.device());
    }
};

/**
 * @brief Benchmark single training iteration (forward + backward + update)
 */
void benchmark_training_iteration() {
    std::cout << "\n========================================\n";
    std::cout << "  Single Training Iteration\n";
    std::cout << "========================================\n\n";

    std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>> configs = {
        {128, 256, 10, "Small MLP (128->256->10)"},
        {512, 512, 100, "Medium MLP (512->512->100)"},
        {1024, 1024, 1000, "Large MLP (1024->1024->1000)"},
    };

    for (const auto& [input_size, hidden_size, output_size, name] : configs) {
        auto model = std::make_shared<SimpleMLP>(input_size, hidden_size, output_size);
        model->to(g_bench_device);
        auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
        auto criterion = std::make_shared<MSELoss>();

        // Create dummy data
        auto input = Variable(randn({32, input_size}, DType::Float32, g_bench_device), true);
        auto target = Variable(randn({32, output_size}, DType::Float32, g_bench_device), false);

        Benchmark bench(name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            // Forward pass
            model->train();
            auto output = model->forward(input);

            // Compute loss
            auto loss = (*criterion)(output, target);

            // Backward pass
            optimizer->zero_grad();
            loss.backward();

            // Update parameters
            optimizer->step();

            // Force synchronization
            volatile void* ptr = loss.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark batch processing
 */
void benchmark_batch_training() {
    std::cout << "\n========================================\n";
    std::cout << "  Batch Training (Multiple Iterations)\n";
    std::cout << "========================================\n\n";

    std::vector<std::pair<int64_t, std::string>> batch_sizes = {
        {16, "Batch Size 16"},
        {32, "Batch Size 32"},
        {64, "Batch Size 64"},
        {128, "Batch Size 128"},
    };

    for (const auto& [batch_size, name] : batch_sizes) {
        auto model = std::make_shared<SimpleMLP>(256, 256, 10);
        model->to(g_bench_device);
        auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
        auto criterion = std::make_shared<MSELoss>();

        // Pre-generate batches
        std::vector<Variable> inputs;
        std::vector<Variable> targets;
        for (size_t i = 0; i < 10; ++i) {
            inputs.push_back(Variable(randn({batch_size, 256}, DType::Float32, g_bench_device), true));
            targets.push_back(Variable(randn({batch_size, 10}, DType::Float32, g_bench_device), false));
        }

        Benchmark bench(name, 2, 10);

        auto result = bench.set_device(g_bench_device).run([&]() {
            model->train();

            // Process all batches
            for (size_t i = 0; i < inputs.size(); ++i) {
                auto output = model->forward(inputs[i]);
                auto loss = (*criterion)(output, targets[i]);

                optimizer->zero_grad();
                loss.backward();
                optimizer->step();
            }

            // Sync
            volatile void* ptr = model->parameters()[0]->tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark CNN training
 */
void benchmark_cnn_training() {
    std::cout << "\n========================================\n";
    std::cout << "  CNN Training Iterations\n";
    std::cout << "========================================\n\n";

    std::vector<std::tuple<int64_t, int64_t, int64_t, std::string>> configs = {
        {8, 32, 10, "Small CNN (batch=8, 32x32, 10 classes)"},
        {16, 32, 100, "Medium CNN (batch=16, 32x32, 100 classes)"},
    };

    for (const auto& [batch_size, image_size, num_classes, name] : configs) {
        auto model = std::make_shared<SimpleCNN>(3, num_classes);
        model->to(g_bench_device);
        auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);
        auto criterion = std::make_shared<CrossEntropyLoss>();

        // Create dummy data
        auto input = Variable(randn({batch_size, 3, image_size, image_size}, DType::Float32, g_bench_device), true);
        // Create random integer targets for classification
        // RR.19 (audit-11): intentionally CPU — target_ptr below uses
        // raw data<int64_t>() host access to fill random labels. The
        // resulting target_tensor is consumed by CrossEntropyLoss which
        // accepts a CPU label tensor regardless of the model device.
        auto target_tensor = zeros({batch_size}, DType::Int64, Device::cpu());
        auto target_ptr = target_tensor.data<int64_t>();
        for (int64_t i = 0; i < batch_size; ++i) {
            target_ptr[i] = static_cast<int64_t>(rand() % num_classes);
        }
        auto target = Variable(target_tensor, false);

        Benchmark bench(name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);

        auto result = bench.set_device(g_bench_device).run([&]() {
            model->train();

            // Forward
            auto output = model->forward(input);

            // Loss - CrossEntropyLoss takes Variable and Tensor
            auto loss = (*criterion)(output, target.tensor());

            // Backward
            optimizer->zero_grad();
            loss.backward();

            // Update
            optimizer->step();

            // Sync
            volatile void* ptr = loss.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark optimizer comparison
 */
void benchmark_optimizers() {
    std::cout << "\n========================================\n";
    std::cout << "  Optimizer Comparison\n";
    std::cout << "========================================\n\n";

    const int64_t input_size = 512;
    const int64_t hidden_size = 512;
    const int64_t output_size = 10;
    const int64_t batch_size = 32;

    // Test SGD
    {
        auto model = std::make_shared<SimpleMLP>(input_size, hidden_size, output_size);
        model->to(g_bench_device);
        auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01, 0.9);
        auto criterion = std::make_shared<MSELoss>();

        auto input = Variable(randn({batch_size, input_size}, DType::Float32, g_bench_device), true);
        auto target = Variable(randn({batch_size, output_size}, DType::Float32, g_bench_device), false);

        Benchmark bench("SGD (momentum=0.9)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto output = model->forward(input);
            auto loss = (*criterion)(output, target);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            volatile void* ptr = loss.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }

    // Test Adam
    {
        auto model = std::make_shared<SimpleMLP>(input_size, hidden_size, output_size);
        model->to(g_bench_device);
        auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
        auto criterion = std::make_shared<MSELoss>();

        auto input = Variable(randn({batch_size, input_size}, DType::Float32, g_bench_device), true);
        auto target = Variable(randn({batch_size, output_size}, DType::Float32, g_bench_device), false);

        Benchmark bench("Adam", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            auto output = model->forward(input);
            auto loss = (*criterion)(output, target);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();
            volatile void* ptr = loss.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark gradient accumulation
 */
void benchmark_gradient_accumulation() {
    std::cout << "\n========================================\n";
    std::cout << "  Gradient Accumulation\n";
    std::cout << "========================================\n\n";

    std::vector<std::pair<size_t, std::string>> accum_steps = {
        {1, "No Accumulation"},
        {2, "Accumulate 2 steps"},
        {4, "Accumulate 4 steps"},
        {8, "Accumulate 8 steps"},
    };

    for (const auto& [steps, name] : accum_steps) {
        auto model = std::make_shared<SimpleMLP>(256, 256, 10);
        model->to(g_bench_device);
        auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
        auto criterion = std::make_shared<MSELoss>();

        // Pre-generate data
        std::vector<Variable> inputs;
        std::vector<Variable> targets;
        for (size_t i = 0; i < steps; ++i) {
            inputs.push_back(Variable(randn({32, 256}, DType::Float32, g_bench_device), true));
            targets.push_back(Variable(randn({32, 10}, DType::Float32, g_bench_device), false));
        }

        Benchmark bench(name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            optimizer->zero_grad();

            // Accumulate gradients
            for (size_t i = 0; i < steps; ++i) {
                auto output = model->forward(inputs[i]);
                auto loss = (*criterion)(output, targets[i]);
                loss.backward();
            }

            // Update once
            optimizer->step();

            // Sync
            volatile void* ptr = model->parameters()[0]->tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

/**
 * @brief Benchmark memory usage patterns
 */
void benchmark_memory_patterns() {
    std::cout << "\n========================================\n";
    std::cout << "  Memory Usage Patterns\n";
    std::cout << "========================================\n\n";

    // Sequential layer processing
    {
        auto model = std::make_shared<SimpleMLP>(512, 512, 10);
        model->to(g_bench_device);
        model->train();

        auto input = Variable(randn({32, 512}, DType::Float32, g_bench_device), true);
        auto target = Variable(randn({32, 10}, DType::Float32, g_bench_device), false);
        auto criterion = std::make_shared<MSELoss>();
        auto optimizer = std::make_shared<optim::SGD>(model->parameters(), 0.01);

        Benchmark bench("Full Training Step with Cleanup", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.set_device(g_bench_device).run([&]() {
            // Training step
            auto output = model->forward(input);
            auto loss = (*criterion)(output, target);
            optimizer->zero_grad();
            loss.backward();
            optimizer->step();

            // Sync
            volatile void* ptr = loss.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        collect_result(result);
    }
}

int main(int argc, char** argv) {
    // RR.19 (audit-11): pick up --device / --device-id from argv. Defaults
    // to CPU when no flag is present, matching pre-RR.19 behaviour.
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);

    bool json_output = false;
    for (int i = 1; i < argc; ++i) {
        if (std::string(argv[i]) == "--json") {
            json_output = true;
        }
    }

    if (!json_output) {
        std::cout << "\n";
        std::cout << "========================================\n";
        std::cout << "  Tenzor Training Benchmark Suite\n";
        std::cout << "  device=" << g_bench_device.to_string() << "\n";
        std::cout << "========================================\n";
        std::cout << "\nTarget Performance Metrics:\n";
        std::cout << "  Training iteration:  < 5ms for small models\n";
        std::cout << "  Batch processing:    Linear scaling with batch size\n";
        std::cout << "  Optimizer overhead:  < 10% of total time\n";
        std::cout << "\n";
    }

    try {
        // Initialize Tenzor
        initialize();

        std::streambuf* original_buf = nullptr;
        std::ostringstream null_stream;
        if (json_output) {
            original_buf = std::cout.rdbuf(null_stream.rdbuf());
        }

        // Run benchmark suites
        benchmark_training_iteration();
        benchmark_batch_training();
        benchmark_cnn_training();
        benchmark_optimizers();
        benchmark_gradient_accumulation();
        benchmark_memory_patterns();

        if (json_output && original_buf) {
            std::cout.rdbuf(original_buf);
        }

        if (json_output) {
            BenchmarkSuite suite("training");
            std::cout << suite.export_json(g_all_results);
        } else {
            std::cout << "\n========================================\n";
            std::cout << "  Training Benchmarks Complete\n";
            std::cout << "========================================\n\n";
        }

        // Finalize
        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
