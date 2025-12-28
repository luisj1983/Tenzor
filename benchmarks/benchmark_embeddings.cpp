/**
 * @file benchmark_embeddings.cpp
 * @brief Comprehensive benchmark for embedding layers and lookup operations
 *
 * Critical for NLP workloads:
 * - Token embedding lookup (vocabulary sizes)
 * - Position embedding
 * - Embedding + Projection fusion
 * - Sparse gradient updates
 * - Vocabulary scaling
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/embedding.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>
#include <random>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

constexpr size_t WARMUP_ITERATIONS = 5;
constexpr size_t BENCHMARK_ITERATIONS = 100;

/**
 * @brief Benchmark embedding forward pass (lookup)
 */
void benchmark_embedding_forward() {
    std::cout << "\n========================================\n";
    std::cout << "  Embedding Forward (Lookup)\n";
    std::cout << "========================================\n\n";

    struct EmbedConfig {
        int64_t vocab_size;
        int64_t embed_dim;
        int64_t batch;
        int64_t seq_len;
        std::string name;
    };

    std::vector<EmbedConfig> configs = {
        // BERT configurations
        {30522, 768, 8, 128, "BERT vocab (30K, batch=8, seq=128)"},
        {30522, 768, 8, 512, "BERT vocab (30K, batch=8, seq=512)"},
        {30522, 768, 32, 128, "BERT vocab large batch (30K, batch=32)"},

        // GPT-2 configurations
        {50257, 768, 4, 256, "GPT-2 vocab (50K, batch=4, seq=256)"},
        {50257, 768, 4, 1024, "GPT-2 vocab (50K, batch=4, seq=1024)"},
        {50257, 1280, 2, 1024, "GPT-2 Large (50K, batch=2, seq=1024)"},

        // Llama configurations
        {32000, 4096, 4, 512, "Llama 7B vocab (32K, batch=4, seq=512)"},
        {32000, 4096, 2, 2048, "Llama 7B long (32K, batch=2, seq=2048)"},

        // Large vocabulary
        {100000, 768, 8, 256, "Large vocab (100K, batch=8, seq=256)"},
        {250000, 768, 4, 256, "Very large vocab (250K, batch=4)"},
    };

    std::vector<BenchmarkResult> results;

    for (const auto& cfg : configs) {
        auto embedding = Embedding(cfg.vocab_size, cfg.embed_dim);

        // Create random indices
        auto indices = zeros({cfg.batch, cfg.seq_len}, DType::Int64);
        auto idx_ptr = indices.data<int64_t>();
        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> dist(0, cfg.vocab_size - 1);
        for (int64_t i = 0; i < cfg.batch * cfg.seq_len; ++i) {
            idx_ptr[i] = dist(gen);
        }
        auto indices_var = Variable(indices, false);

        // Memory: read vocab_size * embed_dim (table) + write batch * seq * embed
        size_t bytes = cfg.batch * cfg.seq_len * cfg.embed_dim * sizeof(float);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_bytes(bytes);

        auto result = bench.run([&]() {
            auto output = embedding.forward(indices_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
        results.push_back(result);
    }

    // Summary
    std::cout << "\n  Embedding Bandwidth Summary:\n";
    std::cout << std::left << std::setw(45) << "Config"
              << std::right << std::setw(15) << "Bandwidth (GB/s)"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    for (const auto& result : results) {
        std::cout << std::left << std::setw(45) << result.name
                  << std::right << std::fixed << std::setprecision(2)
                  << std::setw(15) << result.bandwidth_gbs
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark embedding backward pass (sparse gradient)
 */
void benchmark_embedding_backward() {
    std::cout << "\n========================================\n";
    std::cout << "  Embedding Backward (Sparse Gradient)\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t vocab_size;
        int64_t embed_dim;
        int64_t batch;
        int64_t seq_len;
        std::string name;
    };

    std::vector<Config> configs = {
        {30522, 768, 8, 128, "BERT (30K vocab, 8x128)"},
        {30522, 768, 32, 256, "BERT large batch (30K, 32x256)"},
        {50257, 768, 4, 512, "GPT-2 (50K vocab, 4x512)"},
    };

    for (const auto& cfg : configs) {
        auto embedding = Embedding(cfg.vocab_size, cfg.embed_dim);

        auto indices = zeros({cfg.batch, cfg.seq_len}, DType::Int64);
        auto idx_ptr = indices.data<int64_t>();
        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> dist(0, cfg.vocab_size - 1);
        for (int64_t i = 0; i < cfg.batch * cfg.seq_len; ++i) {
            idx_ptr[i] = dist(gen);
        }
        auto indices_var = Variable(indices, false);

        // Forward only
        Benchmark bench_fwd(cfg.name + " (fwd)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_fwd = bench_fwd.run([&]() {
            auto output = embedding.forward(indices_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_fwd.print();

        // Forward + Backward
        Benchmark bench_bwd(cfg.name + " (fwd+bwd)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
        auto result_bwd = bench_bwd.run([&]() {
            auto output = embedding.forward(indices_var);
            auto grad = ones_like(output.tensor());
            output.backward(grad);

            // Clear gradients
            for (auto& param : embedding.parameters()) {
                param->zero_grad();
            }
        });
        result_bwd.print();

        double ratio = result_bwd.stats.mean / result_fwd.stats.mean;
        std::cout << "  Backward/Forward ratio: " << std::fixed << std::setprecision(2)
                  << ratio << "x\n\n";
    }
}

/**
 * @brief Benchmark combined embedding (token + position)
 */
void benchmark_combined_embedding() {
    std::cout << "\n========================================\n";
    std::cout << "  Combined Token + Position Embedding\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t vocab_size;
        int64_t max_pos;
        int64_t embed_dim;
        int64_t batch;
        int64_t seq_len;
        std::string name;
    };

    std::vector<Config> configs = {
        {30522, 512, 768, 8, 256, "BERT-style (vocab=30K, max_pos=512)"},
        {50257, 1024, 768, 4, 512, "GPT-2 style (vocab=50K, max_pos=1024)"},
        {32000, 4096, 4096, 2, 2048, "Llama style (vocab=32K, max_pos=4096)"},
    };

    for (const auto& cfg : configs) {
        auto token_embed = Embedding(cfg.vocab_size, cfg.embed_dim);
        auto pos_embed = Embedding(cfg.max_pos, cfg.embed_dim);

        // Create indices
        auto token_ids = zeros({cfg.batch, cfg.seq_len}, DType::Int64);
        auto pos_ids = zeros({cfg.batch, cfg.seq_len}, DType::Int64);

        auto tok_ptr = token_ids.data<int64_t>();
        auto pos_ptr = pos_ids.data<int64_t>();

        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> tok_dist(0, cfg.vocab_size - 1);

        for (int64_t b = 0; b < cfg.batch; ++b) {
            for (int64_t s = 0; s < cfg.seq_len; ++s) {
                tok_ptr[b * cfg.seq_len + s] = tok_dist(gen);
                pos_ptr[b * cfg.seq_len + s] = s;
            }
        }

        auto token_var = Variable(token_ids, false);
        auto pos_var = Variable(pos_ids, false);

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);

        auto result = bench.run([&]() {
            auto tok_emb = token_embed.forward(token_var);
            auto pos_emb = pos_embed.forward(pos_var);

            // Combine embeddings
            auto combined = Variable(add(tok_emb.tensor(), pos_emb.tensor()), false);

            volatile void* ptr = combined.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark vocabulary size scaling
 */
void benchmark_vocab_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  Vocabulary Size Scaling\n";
    std::cout << "========================================\n\n";

    const int64_t embed_dim = 768;
    const int64_t batch = 8;
    const int64_t seq_len = 256;

    std::vector<int64_t> vocab_sizes = {10000, 30000, 50000, 100000, 250000, 500000};

    std::cout << std::left << std::setw(15) << "Vocab Size"
              << std::setw(18) << "Table Size (MB)"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "Tokens/ms"
              << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (auto vocab_size : vocab_sizes) {
        auto embedding = Embedding(vocab_size, embed_dim);

        auto indices = zeros({batch, seq_len}, DType::Int64);
        auto idx_ptr = indices.data<int64_t>();
        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> dist(0, vocab_size - 1);
        for (int64_t i = 0; i < batch * seq_len; ++i) {
            idx_ptr[i] = dist(gen);
        }
        auto indices_var = Variable(indices, false);

        Benchmark bench("vocab=" + std::to_string(vocab_size), 5, 50);

        auto result = bench.run([&]() {
            auto output = embedding.forward(indices_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double table_mb = (vocab_size * embed_dim * sizeof(float)) / (1024.0 * 1024.0);
        double time_ms = result.stats.mean * 1000.0;
        double tokens_per_ms = (batch * seq_len) / time_ms;

        std::cout << std::left << std::setw(15) << vocab_size
                  << std::fixed << std::setprecision(1)
                  << std::setw(18) << table_mb
                  << std::setprecision(3)
                  << std::setw(15) << time_ms
                  << std::setprecision(1)
                  << std::setw(15) << tokens_per_ms
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark embedding dimension scaling
 */
void benchmark_embed_dim_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  Embedding Dimension Scaling\n";
    std::cout << "========================================\n\n";

    const int64_t vocab_size = 50000;
    const int64_t batch = 8;
    const int64_t seq_len = 256;

    std::vector<int64_t> embed_dims = {256, 512, 768, 1024, 2048, 4096};

    std::cout << std::left << std::setw(12) << "Embed Dim"
              << std::setw(18) << "Table Size (MB)"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Bandwidth (GB/s)"
              << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (auto embed_dim : embed_dims) {
        auto embedding = Embedding(vocab_size, embed_dim);

        auto indices = zeros({batch, seq_len}, DType::Int64);
        auto idx_ptr = indices.data<int64_t>();
        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> dist(0, vocab_size - 1);
        for (int64_t i = 0; i < batch * seq_len; ++i) {
            idx_ptr[i] = dist(gen);
        }
        auto indices_var = Variable(indices, false);

        size_t bytes = batch * seq_len * embed_dim * sizeof(float);

        Benchmark bench("dim=" + std::to_string(embed_dim), 5, 50);
        bench.set_bytes(bytes);

        auto result = bench.run([&]() {
            auto output = embedding.forward(indices_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double table_mb = (vocab_size * embed_dim * sizeof(float)) / (1024.0 * 1024.0);
        double time_ms = result.stats.mean * 1000.0;

        std::cout << std::left << std::setw(12) << embed_dim
                  << std::fixed << std::setprecision(1)
                  << std::setw(18) << table_mb
                  << std::setprecision(3)
                  << std::setw(15) << time_ms
                  << std::setprecision(2)
                  << std::setw(18) << result.bandwidth_gbs
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark embedding + linear projection (fused pattern)
 */
void benchmark_embed_projection() {
    std::cout << "\n========================================\n";
    std::cout << "  Embedding + Linear Projection\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t vocab_size;
        int64_t embed_dim;
        int64_t proj_dim;
        int64_t batch;
        int64_t seq_len;
        std::string name;
    };

    std::vector<Config> configs = {
        {50257, 768, 768, 8, 256, "GPT-2 (embed=proj=768)"},
        {32000, 4096, 4096, 4, 512, "Llama 7B (embed=proj=4096)"},
        {100000, 256, 768, 8, 256, "Small embed + proj (256->768)"},
    };

    for (const auto& cfg : configs) {
        auto embedding = Embedding(cfg.vocab_size, cfg.embed_dim);
        auto projection = Linear(cfg.embed_dim, cfg.proj_dim);

        auto indices = zeros({cfg.batch, cfg.seq_len}, DType::Int64);
        auto idx_ptr = indices.data<int64_t>();
        std::mt19937 gen(42);
        std::uniform_int_distribution<int64_t> dist(0, cfg.vocab_size - 1);
        for (int64_t i = 0; i < cfg.batch * cfg.seq_len; ++i) {
            idx_ptr[i] = dist(gen);
        }
        auto indices_var = Variable(indices, false);

        // Embedding only
        Benchmark bench_embed(cfg.name + " (embed only)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_embed = bench_embed.run([&]() {
            auto output = embedding.forward(indices_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_embed.print();

        // Embedding + projection
        Benchmark bench_both(cfg.name + " (embed+proj)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_both = bench_both.run([&]() {
            auto emb_output = embedding.forward(indices_var);
            auto proj_output = projection.forward(emb_output);
            volatile void* ptr = proj_output.tensor().data_ptr();
            (void)ptr;
        });
        result_both.print();

        double proj_time = result_both.stats.mean - result_embed.stats.mean;
        std::cout << "  Projection overhead: " << std::fixed << std::setprecision(3)
                  << (proj_time * 1000.0) << " ms\n\n";
    }
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor Embedding Benchmark Suite\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  Embedding lookup:     Memory bandwidth limited\n";
    std::cout << "  Backward (sparse):    < 2x forward time\n";
    std::cout << "  Large vocab (500K):   < 5ms for batch=8\n";
    std::cout << "  Vocab scaling:        Sub-linear growth\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_embedding_forward();
        benchmark_embedding_backward();
        benchmark_combined_embedding();
        benchmark_vocab_scaling();
        benchmark_embed_dim_scaling();
        benchmark_embed_projection();

        std::cout << "\n========================================\n";
        std::cout << "  Embedding Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
