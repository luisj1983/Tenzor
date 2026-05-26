/**
 * @file benchmark_models.cpp
 * @brief End-to-end model benchmarks for inference and training
 *
 * Critical for real-world performance validation:
 * - ResNet inference/training throughput
 * - BERT/Transformer inference latency
 * - GPT-style autoregressive generation
 * - Vision Transformer (ViT) performance
 * - Batch size scaling analysis
 * - Model size scaling (parameter count)
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/layers/conv.hpp"
#include "tenzor/nn/layers/pooling.hpp"
#include "tenzor/nn/layers/normalization.hpp"
#include "tenzor/nn/layers/batchnorm.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/layers/transformer.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/nn/loss/losses.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

constexpr size_t WARMUP_ITERATIONS = 3;
constexpr size_t BENCHMARK_ITERATIONS = 20;

// ============================================================================
// Model Definitions
// ============================================================================

/**
 * @brief Simplified ResNet-like block for benchmarking
 */
class ResNetBlock : public Module {
public:
    ResNetBlock(int64_t channels, int64_t stride = 1)
        : conv1_(std::make_shared<Conv2d>(channels, channels, 3, stride, 1)),
          bn1_(std::make_shared<BatchNorm2d>(channels)),
          conv2_(std::make_shared<Conv2d>(channels, channels, 3, 1, 1)),
          bn2_(std::make_shared<BatchNorm2d>(channels)) {
        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("conv2", conv2_);
        register_module("bn2", bn2_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto out = relu(bn1_->forward(conv1_->forward(x)));
        out = bn2_->forward(conv2_->forward(out));
        // Residual connection (simplified - assumes same dimensions)
        out = Variable(add(out.tensor(), x.tensor()), out.requires_grad());
        return relu(out);
    }

private:
    std::shared_ptr<Conv2d> conv1_, conv2_;
    std::shared_ptr<BatchNorm2d> bn1_, bn2_;
};

/**
 * @brief Simplified ResNet for benchmarking
 */
class BenchmarkResNet : public Module {
public:
    BenchmarkResNet(int64_t num_classes = 1000, int64_t num_blocks = 4)
        : conv1_(std::make_shared<Conv2d>(3, 64, 7, 2, 3)),
          bn1_(std::make_shared<BatchNorm2d>(64)),
          pool_(std::make_shared<MaxPool2d>(3, 2, 1)),
          fc_(std::make_shared<Linear>(64 * 7 * 7, num_classes)) {
        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("pool", pool_);

        // Add residual blocks
        for (int i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<ResNetBlock>(64);
            blocks_.push_back(block);
            register_module("block" + std::to_string(i), block);
        }

        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto out = relu(bn1_->forward(conv1_->forward(x)));
        out = pool_->forward(out);

        for (auto& block : blocks_) {
            out = block->forward(out);
        }

        // Global average pooling (mean over H and W dimensions)
        auto t = out.tensor();
        auto pooled = mean(mean(t, 3, true), 2, true);  // Average over H, W
        out = Variable(pooled.squeeze(3).squeeze(2), out.requires_grad());

        // Flatten
        auto batch_size = out.tensor().shape()[0];
        auto flat = out.tensor().reshape({batch_size, -1});
        out = Variable(flat, out.requires_grad());

        return fc_->forward(out);
    }

private:
    std::shared_ptr<Conv2d> conv1_;
    std::shared_ptr<BatchNorm2d> bn1_;
    std::shared_ptr<MaxPool2d> pool_;
    std::vector<std::shared_ptr<ResNetBlock>> blocks_;
    std::shared_ptr<Linear> fc_;
};

/**
 * @brief Simplified BERT-like encoder for benchmarking
 */
class BenchmarkBERT : public Module {
public:
    BenchmarkBERT(int64_t vocab_size = 30522,
                  int64_t hidden_size = 768,
                  int64_t num_layers = 12,
                  int64_t num_heads = 12,
                  int64_t intermediate_size = 3072)
        : hidden_size_(hidden_size) {

        // Embedding (simplified - no position/segment)
        embedding_ = std::make_shared<Linear>(vocab_size, hidden_size);
        register_module("embedding", embedding_);

        // Transformer layers
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = std::make_shared<TransformerEncoderLayer>(
                hidden_size, num_heads, intermediate_size, 0.1, "gelu", true);
            layers_.push_back(layer);
            register_module("layer" + std::to_string(i), layer);
        }

        // Classification head
        pooler_ = std::make_shared<Linear>(hidden_size, hidden_size);
        register_module("pooler", pooler_);
    }

    auto forward_impl(const Variable& input_ids) -> Variable override {
        // Simple embedding lookup simulation
        auto x = embedding_->forward(input_ids);

        // Pass through transformer layers
        for (auto& layer : layers_) {
            x = (*layer)(x);
        }

        // Pool first token
        auto first_token = x.tensor().slice(1, 0, 1).squeeze(1);
        auto pooled = Variable(first_token, x.requires_grad());
        // Both ``tenzor::tanh`` (autograd op) and ``tenzor::nn::tanh``
        // (activation wrapper) are pulled in by the using-directives at
        // the top of this file. Disambiguate explicitly so the benchmark
        // keeps building under stricter overload resolution.
        pooled = ::tenzor::nn::tanh(pooler_->forward(pooled));

        return pooled;
    }

private:
    int64_t hidden_size_;
    std::shared_ptr<Linear> embedding_;
    std::vector<std::shared_ptr<TransformerEncoderLayer>> layers_;
    std::shared_ptr<Linear> pooler_;
};

/**
 * @brief Simplified GPT-like decoder for benchmarking
 */
class BenchmarkGPT : public Module {
public:
    BenchmarkGPT(int64_t vocab_size = 50257,
                 int64_t hidden_size = 768,
                 int64_t num_layers = 12,
                 int64_t num_heads = 12)
        : hidden_size_(hidden_size), num_heads_(num_heads) {

        // Token embedding
        token_embedding_ = std::make_shared<Linear>(vocab_size, hidden_size);
        register_module("token_embedding", token_embedding_);

        // Transformer layers (decoder)
        for (int64_t i = 0; i < num_layers; ++i) {
            auto layer = std::make_shared<TransformerDecoderLayer>(
                hidden_size, num_heads, hidden_size * 4, 0.1, "gelu", true);
            layers_.push_back(layer);
            register_module("layer" + std::to_string(i), layer);
        }

        // Output projection
        lm_head_ = std::make_shared<Linear>(hidden_size, vocab_size);
        register_module("lm_head", lm_head_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Embedding
        auto hidden = token_embedding_->forward(x);

        // Create causal mask
        auto seq_len = hidden.tensor().shape()[1];
        auto causal_mask = create_causal_mask(seq_len, hidden.tensor().device());

        // Pass through decoder layers
        for (auto& layer : layers_) {
            hidden = layer->forward(hidden, hidden, Tensor{}, causal_mask);
        }

        // Project to vocabulary
        return lm_head_->forward(hidden);
    }

private:
    int64_t hidden_size_, num_heads_;
    std::shared_ptr<Linear> token_embedding_;
    std::vector<std::shared_ptr<TransformerDecoderLayer>> layers_;
    std::shared_ptr<Linear> lm_head_;
};

// ============================================================================
// Benchmarks
// ============================================================================

/**
 * @brief Benchmark ResNet inference throughput
 */
void benchmark_resnet_inference() {
    std::cout << "\n========================================\n";
    std::cout << "  ResNet Inference Throughput\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t batch;
        int64_t blocks;
        std::string name;
    };

    std::vector<Config> configs = {
        {1, 4, "ResNet-small batch=1"},
        {8, 4, "ResNet-small batch=8"},
        {32, 4, "ResNet-small batch=32"},
        {64, 4, "ResNet-small batch=64"},
        {128, 4, "ResNet-small batch=128"},
    };

    for (const auto& cfg : configs) {
        auto model = std::make_shared<BenchmarkResNet>(1000, cfg.blocks);
        model->eval();

        auto input = randn({cfg.batch, 3, 224, 224});
        auto input_var = Variable(input, false);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.run([&]() {
            auto output = model->forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double throughput = cfg.batch / (result.stats.mean);
        std::cout << cfg.name << ":\n";
        std::cout << "  Latency:    " << std::fixed << std::setprecision(3)
                  << (result.stats.mean * 1000.0) << " ms\n";
        std::cout << "  Throughput: " << std::fixed << std::setprecision(1)
                  << throughput << " images/sec\n\n";
    }
}

/**
 * @brief Benchmark ResNet training iteration
 */
void benchmark_resnet_training() {
    std::cout << "\n========================================\n";
    std::cout << "  ResNet Training Iteration\n";
    std::cout << "========================================\n\n";

    std::vector<int64_t> batch_sizes = {8, 16, 32, 64};

    for (auto batch : batch_sizes) {
        auto model = std::make_shared<BenchmarkResNet>(1000, 4);
        model->train();
        auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
        auto criterion = std::make_shared<CrossEntropyLoss>();

        auto input = randn({batch, 3, 224, 224});
        auto input_var = Variable(input, true);

        // Create random targets
        auto targets = zeros({batch}, DType::Int64);
        auto target_ptr = targets.data<int64_t>();
        for (int64_t i = 0; i < batch; ++i) {
            target_ptr[i] = rand() % 1000;
        }

        Benchmark bench("Batch=" + std::to_string(batch), WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);

        auto result = bench.run([&]() {
            optimizer->zero_grad();
            auto output = model->forward(input_var);
            auto loss = (*criterion)(output, targets);
            loss.backward();
            optimizer->step();
        });

        double throughput = batch / (result.stats.mean);
        std::cout << "Batch=" << batch << ":\n";
        std::cout << "  Iteration time: " << std::fixed << std::setprecision(1)
                  << (result.stats.mean * 1000.0) << " ms\n";
        std::cout << "  Throughput:     " << std::fixed << std::setprecision(1)
                  << throughput << " images/sec\n\n";
    }
}

/**
 * @brief Benchmark BERT-like model inference
 */
void benchmark_bert_inference() {
    std::cout << "\n========================================\n";
    std::cout << "  BERT Inference Latency\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t batch;
        int64_t seq_len;
        int64_t hidden;
        int64_t layers;
        int64_t heads;
        std::string name;
    };

    std::vector<Config> configs = {
        {1, 128, 768, 12, 12, "BERT-base seq=128 batch=1"},
        {8, 128, 768, 12, 12, "BERT-base seq=128 batch=8"},
        {1, 512, 768, 12, 12, "BERT-base seq=512 batch=1"},
        {4, 512, 768, 12, 12, "BERT-base seq=512 batch=4"},
        {1, 128, 1024, 24, 16, "BERT-large seq=128 batch=1"},
    };

    for (const auto& cfg : configs) {
        auto model = std::make_shared<BenchmarkBERT>(
            30522, cfg.hidden, cfg.layers, cfg.heads, cfg.hidden * 4);
        model->eval();

        // Simulate one-hot encoded input (simplified)
        auto input = randn({cfg.batch, cfg.seq_len, 30522});
        auto input_var = Variable(input, false);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.run([&]() {
            auto output = model->forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        std::cout << cfg.name << ":\n";
        std::cout << "  Latency: " << std::fixed << std::setprecision(2)
                  << (result.stats.mean * 1000.0) << " ms\n";
        std::cout << "  P95:     " << std::fixed << std::setprecision(2)
                  << (result.stats.p95 * 1000.0) << " ms\n\n";
    }
}

/**
 * @brief Benchmark GPT-style autoregressive generation
 */
void benchmark_gpt_generation() {
    std::cout << "\n========================================\n";
    std::cout << "  GPT Autoregressive Generation\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t hidden;
        int64_t layers;
        int64_t heads;
        std::string name;
    };

    std::vector<Config> configs = {
        {768, 12, 12, "GPT-2 Small"},
        {1024, 24, 16, "GPT-2 Medium"},
    };

    std::vector<int64_t> seq_lengths = {64, 128, 256, 512};

    for (const auto& cfg : configs) {
        std::cout << cfg.name << ":\n";

        auto model = std::make_shared<BenchmarkGPT>(
            50257, cfg.hidden, cfg.layers, cfg.heads);
        model->eval();

        for (auto seq_len : seq_lengths) {
            // Simulate already-generated context
            auto input = randn({1, seq_len, 50257});
            auto input_var = Variable(input, false);

            Benchmark bench("  seq=" + std::to_string(seq_len), 3, 10);

            auto result = bench.run([&]() {
                auto output = model->forward(input_var);
                volatile void* ptr = output.tensor().data_ptr();
                (void)ptr;
            });

            // Estimate tokens/sec for single token generation
            double tokens_per_sec = 1.0 / result.stats.mean;

            std::cout << "  seq=" << std::setw(4) << seq_len
                      << ": " << std::fixed << std::setprecision(2)
                      << (result.stats.mean * 1000.0) << " ms/step, "
                      << std::setprecision(1) << tokens_per_sec << " tok/s\n";
        }
        std::cout << "\n";
    }
}

/**
 * @brief Benchmark batch size scaling
 */
void benchmark_batch_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  Batch Size Scaling Analysis\n";
    std::cout << "========================================\n\n";

    // Use a simple model to isolate batch scaling behavior
    auto model = std::make_shared<BenchmarkResNet>(1000, 2);
    model->eval();

    std::vector<int64_t> batch_sizes = {1, 2, 4, 8, 16, 32, 64, 128};
    std::vector<double> latencies;

    std::cout << std::left << std::setw(12) << "Batch"
              << std::setw(15) << "Latency (ms)"
              << std::setw(18) << "Throughput (img/s)"
              << std::setw(15) << "Scaling Eff."
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    double base_throughput = 0;

    for (auto batch : batch_sizes) {
        auto input = randn({batch, 3, 224, 224});
        auto input_var = Variable(input, false);

        Benchmark bench("batch=" + std::to_string(batch), 3, 15);

        auto result = bench.run([&]() {
            auto output = model->forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double latency_ms = result.stats.mean * 1000.0;
        double throughput = batch / result.stats.mean;

        if (batch == 1) {
            base_throughput = throughput;
        }

        double efficiency = (throughput / base_throughput) / batch * 100.0;

        std::cout << std::left << std::setw(12) << batch
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << latency_ms
                  << std::setprecision(1)
                  << std::setw(18) << throughput
                  << std::setprecision(1)
                  << std::setw(15) << efficiency << "%"
                  << "\n";

        latencies.push_back(result.stats.mean * 1000.0);
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark model parameter count impact
 */
void benchmark_model_size_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  Model Size Scaling (Transformer)\n";
    std::cout << "========================================\n\n";

    struct ModelConfig {
        int64_t hidden;
        int64_t layers;
        int64_t heads;
        std::string name;
        double approx_params_m;  // Approximate params in millions
    };

    std::vector<ModelConfig> configs = {
        {256, 4, 4, "Tiny (4L)", 5},
        {512, 6, 8, "Small (6L)", 25},
        {768, 12, 12, "Base (12L)", 110},
        {1024, 24, 16, "Large (24L)", 340},
    };

    const int64_t batch = 4;
    const int64_t seq_len = 256;

    std::cout << std::left << std::setw(18) << "Model"
              << std::setw(15) << "~Params (M)"
              << std::setw(15) << "Latency (ms)"
              << std::setw(15) << "ms/M-params"
              << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (const auto& cfg : configs) {
        // Create a BERT-like encoder
        std::vector<std::shared_ptr<TransformerEncoderLayer>> layers;
        for (int64_t i = 0; i < cfg.layers; ++i) {
            layers.push_back(std::make_shared<TransformerEncoderLayer>(
                cfg.hidden, cfg.heads, cfg.hidden * 4, 0.0, "gelu", true));
        }

        auto input = randn({batch, seq_len, cfg.hidden});
        auto input_var = Variable(input, false);

        Benchmark bench(cfg.name, 3, 10);

        auto result = bench.run([&]() {
            auto x = input_var;
            for (auto& layer : layers) {
                x = (*layer)(x);
            }
            volatile void* ptr = x.tensor().data_ptr();
            (void)ptr;
        });

        double latency_ms = result.stats.mean * 1000.0;
        double ms_per_m_params = latency_ms / cfg.approx_params_m;

        std::cout << std::left << std::setw(18) << cfg.name
                  << std::fixed << std::setprecision(0)
                  << std::setw(15) << cfg.approx_params_m
                  << std::setprecision(2)
                  << std::setw(15) << latency_ms
                  << std::setprecision(3)
                  << std::setw(15) << ms_per_m_params
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Memory footprint analysis for models
 */
void benchmark_memory_footprint() {
    std::cout << "\n========================================\n";
    std::cout << "  Model Memory Footprint Analysis\n";
    std::cout << "========================================\n\n";

    struct MemoryConfig {
        std::string name;
        int64_t params_m;
        int64_t batch;
        int64_t seq_len;
        int64_t hidden;
    };

    std::vector<MemoryConfig> configs = {
        {"BERT-base", 110, 8, 512, 768},
        {"BERT-large", 340, 4, 512, 1024},
        {"GPT-2 Small", 117, 4, 1024, 768},
        {"GPT-2 Medium", 345, 2, 1024, 1024},
    };

    std::cout << std::left << std::setw(18) << "Model"
              << std::setw(15) << "Params (MB)"
              << std::setw(18) << "Activations (MB)"
              << std::setw(15) << "Total (MB)"
              << "\n";
    std::cout << std::string(66, '-') << "\n";

    for (const auto& cfg : configs) {
        // Parameters (FP32)
        double params_mb = cfg.params_m * 4.0;  // 4 bytes per param

        // Activations estimate (rough)
        // Each layer stores: attention scores + intermediate activations
        double attn_scores_mb = cfg.batch * 12 * cfg.seq_len * cfg.seq_len * 4.0 / (1024.0 * 1024.0);
        double hidden_acts_mb = cfg.batch * cfg.seq_len * cfg.hidden * 4.0 / (1024.0 * 1024.0);
        double activations_mb = (attn_scores_mb + hidden_acts_mb * 4) * 12;  // ~12 layers

        double total_mb = params_mb + activations_mb;

        std::cout << std::left << std::setw(18) << cfg.name
                  << std::fixed << std::setprecision(0)
                  << std::setw(15) << params_mb
                  << std::setw(18) << activations_mb
                  << std::setw(15) << total_mb
                  << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor Model Benchmark Suite\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  ResNet batch=32:    > 100 images/sec\n";
    std::cout << "  BERT-base seq=128:  < 50ms latency\n";
    std::cout << "  GPT-2 generation:   > 10 tokens/sec\n";
    std::cout << "  Batch scaling:      > 80% efficiency\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_resnet_inference();
        benchmark_resnet_training();
        benchmark_bert_inference();
        benchmark_gpt_generation();
        benchmark_batch_scaling();
        benchmark_model_size_scaling();
        benchmark_memory_footprint();

        std::cout << "\n========================================\n";
        std::cout << "  Model Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
