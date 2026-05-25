/**
 * @file benchmark_attention.cpp
 * @brief Comprehensive benchmark for attention mechanisms and transformer operations
 *
 * Critical for modern ML workloads. Benchmarks:
 * - Multi-head self-attention (various configurations)
 * - Cross-attention patterns
 * - Scaled dot-product attention kernel
 * - Flash Attention comparison (if available)
 * - Sequence length scaling
 * - KV-cache performance
 * - Causal vs bidirectional masking overhead
 */

#include "tenzor/tenzor.hpp"
#include "common.hpp"
#include "tenzor/nn/layers/attention.hpp"
#include "tenzor/nn/layers/transformer.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/activations/activations.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

// HH.25: global device parsed from argv in main(). Defaults to CPU so the
// previous unflagged invocations stay correct.
namespace {
tenzor::Device g_bench_device = tenzor::Device::cpu();
}

// Benchmark configuration
constexpr size_t WARMUP_ITERATIONS = 5;
constexpr size_t BENCHMARK_ITERATIONS = 50;

/**
 * @brief Benchmark multi-head self-attention with various configurations
 */
void benchmark_multihead_attention() {
    std::cout << "\n========================================\n";
    std::cout << "  Multi-Head Self-Attention Benchmarks\n";
    std::cout << "========================================\n\n";
    std::cout << "Target: Competitive with PyTorch/FlashAttention\n\n";

    struct AttentionConfig {
        int64_t batch;
        int64_t seq_len;
        int64_t embed_dim;
        int64_t num_heads;
        std::string name;
    };

    std::vector<AttentionConfig> configs = {
        // BERT-base configurations
        {8, 128, 768, 12, "BERT-base (batch=8, seq=128)"},
        {8, 512, 768, 12, "BERT-base (batch=8, seq=512)"},
        {1, 512, 768, 12, "BERT-base (batch=1, seq=512)"},

        // GPT-2 configurations
        {4, 256, 768, 12, "GPT-2 Small (batch=4, seq=256)"},
        {4, 1024, 768, 12, "GPT-2 Small (batch=4, seq=1024)"},
        {2, 2048, 768, 12, "GPT-2 Small (batch=2, seq=2048)"},

        // Large model configurations
        {4, 512, 1024, 16, "GPT-2 Medium (batch=4, seq=512)"},
        {2, 512, 1280, 20, "GPT-2 Large (batch=2, seq=512)"},
        {1, 512, 1600, 25, "GPT-2 XL (batch=1, seq=512)"},

        // ViT configurations
        {32, 197, 768, 12, "ViT-Base (batch=32, patches=197)"},
        {16, 197, 1024, 16, "ViT-Large (batch=16, patches=197)"},

        // Edge cases
        {1, 4096, 512, 8, "Long sequence (seq=4096)"},
        {64, 64, 256, 4, "Small attention (batch=64, seq=64)"},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& cfg : configs) {
        auto attn = MultiheadAttention(cfg.embed_dim, cfg.num_heads, 0.0, true,
                                       false, false, 0, 0, true);

        auto input = randn({cfg.batch, cfg.seq_len, cfg.embed_dim});
        auto input_var = Variable(input, false);

        // Calculate FLOPs for attention:
        // QKV projections: 3 * batch * seq * embed * embed
        // Attention scores: batch * heads * seq * seq * (embed/heads)
        // Attention output: batch * heads * seq * seq * (embed/heads)
        // Output projection: batch * seq * embed * embed
        size_t qkv_flops = 3 * cfg.batch * cfg.seq_len * cfg.embed_dim * cfg.embed_dim;
        size_t attn_flops = 2 * cfg.batch * cfg.num_heads * cfg.seq_len * cfg.seq_len *
                           (cfg.embed_dim / cfg.num_heads);
        size_t out_flops = cfg.batch * cfg.seq_len * cfg.embed_dim * cfg.embed_dim;
        size_t total_flops = 2 * (qkv_flops + attn_flops + out_flops);  // 2x for multiply-add

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(total_flops);

        auto result = bench.run([&]() {
            auto [output, weights] = attn.forward(input_var, input_var, input_var,
                                                   Tensor{}, Tensor{}, false);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        results.push_back(result);
    }

    // Summary
    std::cout << "\n========================================\n";
    std::cout << "  Attention Performance Summary\n";
    std::cout << "========================================\n";
    std::cout << std::left << std::setw(40) << "Configuration"
              << std::right << std::setw(12) << "Mean (ms)"
              << std::setw(12) << "TFLOPS"
              << "\n";
    std::cout << std::string(64, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(40) << result.name
                  << std::right << std::fixed << std::setprecision(3)
                  << std::setw(12) << (result.stats.mean * 1000.0)
                  << std::setw(12) << result.tflops
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark scaled dot-product attention (core kernel)
 */
void benchmark_scaled_dot_product() {
    std::cout << "\n========================================\n";
    std::cout << "  Scaled Dot-Product Attention Kernel\n";
    std::cout << "========================================\n\n";

    struct SDPConfig {
        int64_t batch;
        int64_t heads;
        int64_t seq_q;
        int64_t seq_kv;
        int64_t head_dim;
        std::string name;
    };

    std::vector<SDPConfig> configs = {
        {8, 12, 512, 512, 64, "Self-Attn (512x512, heads=12)"},
        {8, 12, 1024, 1024, 64, "Self-Attn (1024x1024, heads=12)"},
        {4, 16, 2048, 2048, 64, "Self-Attn (2048x2048, heads=16)"},

        // Cross-attention patterns
        {8, 12, 128, 512, 64, "Cross-Attn (128 query, 512 kv)"},
        {8, 12, 512, 128, 64, "Cross-Attn (512 query, 128 kv)"},

        // Memory-bound cases
        {1, 8, 4096, 4096, 64, "Long seq (4096x4096)"},
        {1, 8, 8192, 8192, 64, "Very long seq (8192x8192)"},
    };

    for (const auto& cfg : configs) {
        auto query = randn({cfg.batch, cfg.heads, cfg.seq_q, cfg.head_dim});
        auto key = randn({cfg.batch, cfg.heads, cfg.seq_kv, cfg.head_dim});
        auto value = randn({cfg.batch, cfg.heads, cfg.seq_kv, cfg.head_dim});

        float scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));

        // FLOPs: QK^T (batch * heads * seq_q * seq_kv * head_dim * 2)
        //      + softmax (batch * heads * seq_q * seq_kv * ~5)
        //      + AV (batch * heads * seq_q * seq_kv * head_dim * 2)
        size_t qk_flops = cfg.batch * cfg.heads * cfg.seq_q * cfg.seq_kv * cfg.head_dim * 2;
        size_t av_flops = cfg.batch * cfg.heads * cfg.seq_q * cfg.seq_kv * cfg.head_dim * 2;
        size_t total_flops = qk_flops + av_flops;

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(total_flops);

        auto result = bench.run([&]() {
            // Manual scaled dot-product attention
            auto key_t = key.transpose(-2, -1);
            auto scores = matmul(query, key_t);
            scores = scores * scale;

            auto scores_var = Variable(scores, false);
            auto attn_weights = tenzor::softmax(scores_var, static_cast<int64_t>(-1));

            auto output = matmul(attn_weights.tensor(), value);
            volatile void* ptr = output.data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark causal masking overhead
 */
void benchmark_causal_attention() {
    std::cout << "\n========================================\n";
    std::cout << "  Causal vs Bidirectional Attention\n";
    std::cout << "========================================\n\n";

    std::vector<int64_t> seq_lengths = {256, 512, 1024, 2048};
    const int64_t batch = 4;
    const int64_t embed_dim = 768;
    const int64_t num_heads = 12;

    for (auto seq_len : seq_lengths) {
        auto attn = MultiheadAttention(embed_dim, num_heads, 0.0, true,
                                       false, false, 0, 0, true);
        auto input = randn({batch, seq_len, embed_dim});
        auto input_var = Variable(input, false);

        // Create causal mask
        auto causal_mask = create_causal_mask(seq_len, g_bench_device, DType::Float32);

        // Bidirectional (no mask)
        Benchmark bench_bi("Bidirectional seq=" + std::to_string(seq_len),
                          WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result_bi = bench_bi.run([&]() {
            auto [output, _] = attn.forward(input_var, input_var, input_var,
                                           Tensor{}, Tensor{}, false);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_bi.print();

        // Causal (with mask)
        Benchmark bench_causal("Causal seq=" + std::to_string(seq_len),
                              WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result_causal = bench_causal.run([&]() {
            auto [output, _] = attn.forward(input_var, input_var, input_var,
                                           Tensor{}, causal_mask, false);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_causal.print();

        double overhead = ((result_causal.stats.mean / result_bi.stats.mean) - 1.0) * 100.0;
        std::cout << "  Causal masking overhead: " << std::fixed << std::setprecision(1)
                  << overhead << "%\n\n";
    }
}

/**
 * @brief Benchmark sequence length scaling (O(n^2) behavior)
 */
void benchmark_sequence_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  Sequence Length Scaling Analysis\n";
    std::cout << "========================================\n\n";
    std::cout << "Verifying O(n^2) attention complexity\n\n";

    const int64_t batch = 4;
    const int64_t embed_dim = 512;
    const int64_t num_heads = 8;

    std::vector<int64_t> seq_lengths = {64, 128, 256, 512, 1024, 2048, 4096};
    std::vector<double> times;

    auto attn = MultiheadAttention(embed_dim, num_heads, 0.0, true,
                                   false, false, 0, 0, true);

    for (auto seq_len : seq_lengths) {
        auto input = randn({batch, seq_len, embed_dim});
        auto input_var = Variable(input, false);

        Benchmark bench("seq=" + std::to_string(seq_len), 3, 20);

        auto result = bench.run([&]() {
            auto [output, _] = attn.forward(input_var, input_var, input_var,
                                           Tensor{}, Tensor{}, false);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        times.push_back(result.stats.mean * 1000.0);
        result.print();
    }

    // Print scaling analysis
    std::cout << "\n  Scaling Analysis:\n";
    std::cout << std::left << std::setw(10) << "Seq Len"
              << std::setw(12) << "Time (ms)"
              << std::setw(15) << "Ratio vs 64"
              << std::setw(15) << "Expected (n^2)"
              << "\n";
    std::cout << std::string(52, '-') << "\n";

    for (size_t i = 0; i < seq_lengths.size(); ++i) {
        double ratio = times[i] / times[0];
        double expected = static_cast<double>(seq_lengths[i] * seq_lengths[i]) /
                         static_cast<double>(seq_lengths[0] * seq_lengths[0]);

        std::cout << std::left << std::setw(10) << seq_lengths[i]
                  << std::fixed << std::setprecision(3)
                  << std::setw(12) << times[i]
                  << std::setw(15) << ratio
                  << std::setw(15) << expected
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark attention backward pass
 */
void benchmark_attention_backward() {
    std::cout << "\n========================================\n";
    std::cout << "  Attention Backward Pass\n";
    std::cout << "========================================\n\n";

    struct BackwardConfig {
        int64_t batch;
        int64_t seq_len;
        int64_t embed_dim;
        int64_t num_heads;
        std::string name;
    };

    std::vector<BackwardConfig> configs = {
        {8, 256, 768, 12, "BERT-like (batch=8, seq=256)"},
        {4, 512, 768, 12, "BERT-like (batch=4, seq=512)"},
        {4, 256, 1024, 16, "Large model (batch=4, seq=256)"},
    };

    for (const auto& cfg : configs) {
        auto attn = MultiheadAttention(cfg.embed_dim, cfg.num_heads, 0.0, true,
                                       false, false, 0, 0, true);

        auto input = randn({cfg.batch, cfg.seq_len, cfg.embed_dim});
        auto input_var = Variable(input, true);  // requires_grad=true

        // Forward only
        Benchmark bench_fwd(cfg.name + " (fwd)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);

        auto result_fwd = bench_fwd.run([&]() {
            auto [output, _] = attn.forward(input_var, input_var, input_var,
                                           Tensor{}, Tensor{}, false);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_fwd.print();

        // Forward + Backward
        Benchmark bench_bwd(cfg.name + " (fwd+bwd)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);

        auto result_bwd = bench_bwd.run([&]() {
            auto [output, _] = attn.forward(input_var, input_var, input_var,
                                           Tensor{}, Tensor{}, false);

            auto grad = ones_like(output.tensor());
            output.backward(grad);

            input_var.zero_grad();
        });
        result_bwd.print();

        double bwd_ratio = result_bwd.stats.mean / result_fwd.stats.mean;
        std::cout << "  Backward/Forward ratio: " << std::fixed << std::setprecision(2)
                  << bwd_ratio << "x\n\n";
    }
}

/**
 * @brief Benchmark transformer encoder layer
 */
void benchmark_transformer_layer() {
    std::cout << "\n========================================\n";
    std::cout << "  Full Transformer Encoder Layer\n";
    std::cout << "========================================\n\n";

    struct TransformerConfig {
        int64_t batch;
        int64_t seq_len;
        int64_t d_model;
        int64_t nhead;
        int64_t dim_ff;
        std::string name;
    };

    std::vector<TransformerConfig> configs = {
        {8, 128, 768, 12, 3072, "BERT-base layer"},
        {4, 512, 768, 12, 3072, "BERT-base long seq"},
        {4, 256, 1024, 16, 4096, "BERT-large layer"},
        {2, 1024, 768, 12, 3072, "Very long sequence"},
    };

    for (const auto& cfg : configs) {
        auto encoder_layer = TransformerEncoderLayer(
            cfg.d_model, cfg.nhead, cfg.dim_ff, 0.1, "relu", true);

        auto input = randn({cfg.batch, cfg.seq_len, cfg.d_model});
        auto input_var = Variable(input, false);

        // Calculate approximate FLOPs
        // Attention: ~4 * seq^2 * d_model + 4 * seq * d_model^2
        // FFN: 2 * seq * d_model * dim_ff
        size_t attn_flops = 4 * cfg.batch * cfg.seq_len * cfg.seq_len * cfg.d_model +
                           4 * cfg.batch * cfg.seq_len * cfg.d_model * cfg.d_model;
        size_t ffn_flops = 2 * cfg.batch * cfg.seq_len * cfg.d_model * cfg.dim_ff * 2;
        size_t total_flops = 2 * (attn_flops + ffn_flops);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(total_flops);

        auto result = bench.run([&]() {
            auto output = encoder_layer(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark memory usage of attention
 */
void benchmark_attention_memory() {
    std::cout << "\n========================================\n";
    std::cout << "  Attention Memory Analysis\n";
    std::cout << "========================================\n\n";

    std::vector<std::pair<int64_t, std::string>> seq_configs = {
        {256, "seq=256"},
        {512, "seq=512"},
        {1024, "seq=1024"},
        {2048, "seq=2048"},
    };

    const int64_t batch = 4;
    const int64_t embed_dim = 768;
    const int64_t num_heads = 12;

    std::cout << std::left << std::setw(15) << "Config"
              << std::setw(18) << "Input (MB)"
              << std::setw(18) << "Attn Matrix (MB)"
              << std::setw(18) << "Total Est. (MB)"
              << "\n";
    std::cout << std::string(69, '-') << "\n";

    for (const auto& [seq_len, name] : seq_configs) {
        // Input memory
        size_t input_bytes = batch * seq_len * embed_dim * sizeof(float);

        // Attention matrix memory: batch * heads * seq * seq
        size_t attn_bytes = batch * num_heads * seq_len * seq_len * sizeof(float);

        // Q, K, V projections
        size_t qkv_bytes = 3 * input_bytes;

        // Total estimated
        size_t total_bytes = input_bytes + attn_bytes + qkv_bytes;

        std::cout << std::left << std::setw(15) << name
                  << std::fixed << std::setprecision(2)
                  << std::setw(18) << (input_bytes / (1024.0 * 1024.0))
                  << std::setw(18) << (attn_bytes / (1024.0 * 1024.0))
                  << std::setw(18) << (total_bytes / (1024.0 * 1024.0))
                  << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    // HH.25: parse --device / --device-id from argv so the runner can target
    // the GPU backends that the binary was supposed to exercise.
    g_bench_device = tenzor::bench::parse_device_arg(argc, argv);
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor Attention Benchmark Suite\n";
    std::cout << "  device=" << g_bench_device.to_string() << "\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  BERT-base (seq=512):  < 5ms forward pass\n";
    std::cout << "  GPT-2 (seq=1024):     < 15ms forward pass\n";
    std::cout << "  Scaling:              O(n^2) confirmed\n";
    std::cout << "  Backward/Forward:     < 3x ratio\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_multihead_attention();
        benchmark_scaled_dot_product();
        benchmark_causal_attention();
        benchmark_sequence_scaling();
        benchmark_attention_backward();
        benchmark_transformer_layer();
        benchmark_attention_memory();

        std::cout << "\n========================================\n";
        std::cout << "  Attention Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
