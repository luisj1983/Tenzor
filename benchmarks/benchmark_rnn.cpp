/**
 * @file benchmark_rnn.cpp
 * @brief Comprehensive benchmark for recurrent neural network layers
 *
 * Benchmarks for sequential models:
 * - LSTM forward/backward
 * - GRU forward/backward
 * - Bidirectional variants
 * - Multi-layer stacking
 * - Sequence length scaling
 * - Hidden size scaling
 */

#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/rnn.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/utils/benchmark.hpp"
#include <iostream>
#include <iomanip>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;
using namespace tenzor::benchmark;

constexpr size_t WARMUP_ITERATIONS = 3;
constexpr size_t BENCHMARK_ITERATIONS = 30;

/**
 * @brief Benchmark LSTM forward pass
 */
void benchmark_lstm_forward() {
    std::cout << "\n========================================\n";
    std::cout << "  LSTM Forward Pass\n";
    std::cout << "========================================\n\n";

    struct LSTMConfig {
        int64_t input_size;
        int64_t hidden_size;
        int64_t num_layers;
        int64_t batch;
        int64_t seq_len;
        bool bidirectional;
        std::string name;
    };

    std::vector<LSTMConfig> configs = {
        // Standard configurations
        {256, 256, 1, 32, 100, false, "LSTM 1L (256->256, seq=100)"},
        {512, 512, 1, 32, 100, false, "LSTM 1L (512->512, seq=100)"},
        {256, 512, 2, 32, 100, false, "LSTM 2L (256->512, seq=100)"},
        {512, 512, 4, 16, 100, false, "LSTM 4L (512->512, seq=100)"},

        // Long sequences
        {256, 256, 2, 16, 500, false, "LSTM 2L long seq (256, seq=500)"},
        {256, 256, 2, 8, 1000, false, "LSTM 2L very long (256, seq=1000)"},

        // Bidirectional
        {256, 256, 1, 32, 100, true, "BiLSTM 1L (256->256, seq=100)"},
        {512, 512, 2, 16, 100, true, "BiLSTM 2L (512->512, seq=100)"},

        // Speech/Audio configurations
        {80, 512, 4, 16, 300, false, "Speech LSTM (80->512, 4L, seq=300)"},

        // NLP configurations
        {768, 768, 2, 32, 128, true, "NLP BiLSTM (768->768, seq=128)"},
    };

    for (const auto& cfg : configs) {
        auto lstm = LSTM(cfg.input_size, cfg.hidden_size, cfg.num_layers,
                        true, 0.0, cfg.bidirectional, true);

        auto input = randn({cfg.seq_len, cfg.batch, cfg.input_size});
        auto input_var = Variable(input, false);

        // FLOPs for LSTM: 4 * hidden * (input + hidden) per timestep per layer
        // For bidirectional, double it
        size_t flops_per_step = 4 * cfg.hidden_size * (cfg.input_size + cfg.hidden_size) * 2;
        size_t total_flops = flops_per_step * cfg.seq_len * cfg.num_layers * cfg.batch;
        if (cfg.bidirectional) total_flops *= 2;

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(total_flops);

        auto result = bench.run([&]() {
            auto output = lstm.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark GRU forward pass
 */
void benchmark_gru_forward() {
    std::cout << "\n========================================\n";
    std::cout << "  GRU Forward Pass\n";
    std::cout << "========================================\n\n";

    struct GRUConfig {
        int64_t input_size;
        int64_t hidden_size;
        int64_t num_layers;
        int64_t batch;
        int64_t seq_len;
        bool bidirectional;
        std::string name;
    };

    std::vector<GRUConfig> configs = {
        {256, 256, 1, 32, 100, false, "GRU 1L (256->256, seq=100)"},
        {512, 512, 2, 32, 100, false, "GRU 2L (512->512, seq=100)"},
        {256, 512, 4, 16, 100, false, "GRU 4L (256->512, seq=100)"},
        {256, 256, 2, 16, 500, false, "GRU 2L long seq (256, seq=500)"},
        {256, 256, 1, 32, 100, true, "BiGRU 1L (256->256, seq=100)"},
    };

    for (const auto& cfg : configs) {
        auto gru = GRU(cfg.input_size, cfg.hidden_size, cfg.num_layers,
                      true, 0.0, cfg.bidirectional, true);

        auto input = randn({cfg.seq_len, cfg.batch, cfg.input_size});
        auto input_var = Variable(input, false);

        // GRU has 3 gates instead of 4, so ~75% of LSTM FLOPs
        size_t flops_per_step = 3 * cfg.hidden_size * (cfg.input_size + cfg.hidden_size) * 2;
        size_t total_flops = flops_per_step * cfg.seq_len * cfg.num_layers * cfg.batch;
        if (cfg.bidirectional) total_flops *= 2;

        Benchmark bench(cfg.name, WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        bench.set_flops(total_flops);

        auto result = bench.run([&]() {
            auto output = gru.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        result.print();
    }
}

/**
 * @brief Benchmark LSTM backward pass
 */
void benchmark_lstm_backward() {
    std::cout << "\n========================================\n";
    std::cout << "  LSTM Backward Pass\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t input_size;
        int64_t hidden_size;
        int64_t num_layers;
        int64_t batch;
        int64_t seq_len;
        std::string name;
    };

    std::vector<Config> configs = {
        {256, 256, 2, 32, 100, "LSTM 2L (256->256)"},
        {512, 512, 2, 16, 100, "LSTM 2L (512->512)"},
        {256, 256, 2, 16, 300, "LSTM 2L long (seq=300)"},
    };

    for (const auto& cfg : configs) {
        auto lstm = LSTM(cfg.input_size, cfg.hidden_size, cfg.num_layers,
                        true, 0.0, false, true);

        auto input = randn({cfg.seq_len, cfg.batch, cfg.input_size});
        auto input_var = Variable(input, true);

        // Forward only
        Benchmark bench_fwd(cfg.name + " (fwd)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS);
        auto result_fwd = bench_fwd.run([&]() {
            auto output = lstm.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });
        result_fwd.print();

        // Forward + Backward
        Benchmark bench_bwd(cfg.name + " (fwd+bwd)", WARMUP_ITERATIONS, BENCHMARK_ITERATIONS / 2);
        auto result_bwd = bench_bwd.run([&]() {
            auto output = lstm.forward(input_var);
            auto grad = ones_like(output.tensor());
            output.backward(grad);

            input_var.zero_grad();
            for (auto& param : lstm.parameters()) {
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
 * @brief Benchmark sequence length scaling
 */
void benchmark_sequence_length_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  RNN Sequence Length Scaling\n";
    std::cout << "========================================\n\n";

    const int64_t input_size = 256;
    const int64_t hidden_size = 256;
    const int64_t num_layers = 2;
    const int64_t batch = 16;

    std::vector<int64_t> seq_lengths = {50, 100, 200, 500, 1000, 2000};

    auto lstm = LSTM(input_size, hidden_size, num_layers, true, 0.0, false, true);

    std::cout << std::left << std::setw(12) << "Seq Len"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Time/step (us)"
              << std::setw(15) << "Linear?"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    double base_time_per_step = 0;

    for (auto seq_len : seq_lengths) {
        auto input = randn({seq_len, batch, input_size});
        auto input_var = Variable(input, false);

        Benchmark bench("seq=" + std::to_string(seq_len), 3, 20);

        auto result = bench.run([&]() {
            auto output = lstm.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double time_ms = result.stats.mean * 1000.0;
        double time_per_step_us = (result.stats.mean * 1e6) / seq_len;

        if (seq_len == 50) {
            base_time_per_step = time_per_step_us;
        }

        double linearity = base_time_per_step / time_per_step_us;
        std::string linear_str = (linearity > 0.9) ? "Yes" : "No";

        std::cout << std::left << std::setw(12) << seq_len
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << time_ms
                  << std::setprecision(1)
                  << std::setw(18) << time_per_step_us
                  << std::setw(15) << linear_str
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark hidden size scaling
 */
void benchmark_hidden_size_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  RNN Hidden Size Scaling\n";
    std::cout << "========================================\n\n";

    const int64_t num_layers = 2;
    const int64_t batch = 32;
    const int64_t seq_len = 100;

    std::vector<int64_t> hidden_sizes = {128, 256, 512, 1024, 2048};

    std::cout << std::left << std::setw(15) << "Hidden Size"
              << std::setw(15) << "Time (ms)"
              << std::setw(15) << "GFLOPS"
              << std::setw(18) << "Params (M)"
              << "\n";
    std::cout << std::string(63, '-') << "\n";

    for (auto hidden_size : hidden_sizes) {
        auto lstm = LSTM(hidden_size, hidden_size, num_layers, true, 0.0, false, true);

        auto input = randn({seq_len, batch, hidden_size});
        auto input_var = Variable(input, false);

        size_t flops_per_step = 4 * hidden_size * (hidden_size + hidden_size) * 2;
        size_t total_flops = flops_per_step * seq_len * num_layers * batch;

        // Approximate parameter count
        double params_m = (4 * hidden_size * hidden_size * 2 * num_layers) / 1e6;

        Benchmark bench("hidden=" + std::to_string(hidden_size), 3, 20);
        bench.set_flops(total_flops);

        auto result = bench.run([&]() {
            auto output = lstm.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double time_ms = result.stats.mean * 1000.0;
        double gflops = result.tflops * 1000.0;

        std::cout << std::left << std::setw(15) << hidden_size
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << time_ms
                  << std::setprecision(1)
                  << std::setw(15) << gflops
                  << std::setprecision(2)
                  << std::setw(18) << params_m
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Compare LSTM vs GRU performance
 */
void benchmark_lstm_vs_gru() {
    std::cout << "\n========================================\n";
    std::cout << "  LSTM vs GRU Comparison\n";
    std::cout << "========================================\n\n";

    struct Config {
        int64_t input_size;
        int64_t hidden_size;
        int64_t num_layers;
        int64_t batch;
        int64_t seq_len;
        std::string name;
    };

    std::vector<Config> configs = {
        {256, 256, 2, 32, 100, "256->256, 2L"},
        {512, 512, 2, 32, 100, "512->512, 2L"},
        {256, 512, 4, 16, 100, "256->512, 4L"},
    };

    std::cout << std::left << std::setw(20) << "Config"
              << std::setw(15) << "LSTM (ms)"
              << std::setw(15) << "GRU (ms)"
              << std::setw(15) << "GRU Speedup"
              << "\n";
    std::cout << std::string(65, '-') << "\n";

    for (const auto& cfg : configs) {
        auto lstm = LSTM(cfg.input_size, cfg.hidden_size, cfg.num_layers,
                        true, 0.0, false, true);
        auto gru = GRU(cfg.input_size, cfg.hidden_size, cfg.num_layers,
                      true, 0.0, false, true);

        auto input = randn({cfg.seq_len, cfg.batch, cfg.input_size});
        auto input_var = Variable(input, false);

        // LSTM
        Benchmark bench_lstm("LSTM " + cfg.name, 3, 20);
        auto result_lstm = bench_lstm.run([&]() {
            auto output = lstm.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        // GRU
        Benchmark bench_gru("GRU " + cfg.name, 3, 20);
        auto result_gru = bench_gru.run([&]() {
            auto output = gru.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double lstm_ms = result_lstm.stats.mean * 1000.0;
        double gru_ms = result_gru.stats.mean * 1000.0;
        double speedup = lstm_ms / gru_ms;

        std::cout << std::left << std::setw(20) << cfg.name
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << lstm_ms
                  << std::setw(15) << gru_ms
                  << std::setprecision(2)
                  << std::setw(15) << speedup << "x"
                  << "\n";
    }
    std::cout << std::endl;
}

/**
 * @brief Benchmark layer count scaling
 */
void benchmark_layer_scaling() {
    std::cout << "\n========================================\n";
    std::cout << "  RNN Layer Count Scaling\n";
    std::cout << "========================================\n\n";

    const int64_t input_size = 256;
    const int64_t hidden_size = 256;
    const int64_t batch = 32;
    const int64_t seq_len = 100;

    std::vector<int64_t> layer_counts = {1, 2, 4, 6, 8};

    std::cout << std::left << std::setw(12) << "Layers"
              << std::setw(15) << "Time (ms)"
              << std::setw(18) << "Time/layer (ms)"
              << std::setw(15) << "Linear?"
              << "\n";
    std::cout << std::string(60, '-') << "\n";

    double base_time_per_layer = 0;

    for (auto num_layers : layer_counts) {
        auto lstm = LSTM(input_size, hidden_size, num_layers, true, 0.0, false, true);

        auto input = randn({seq_len, batch, input_size});
        auto input_var = Variable(input, false);

        Benchmark bench("layers=" + std::to_string(num_layers), 3, 20);

        auto result = bench.run([&]() {
            auto output = lstm.forward(input_var);
            volatile void* ptr = output.tensor().data_ptr();
            (void)ptr;
        });

        double time_ms = result.stats.mean * 1000.0;
        double time_per_layer = time_ms / num_layers;

        if (num_layers == 1) {
            base_time_per_layer = time_per_layer;
        }

        double linearity = base_time_per_layer / time_per_layer;
        std::string linear_str = (linearity > 0.9) ? "Yes" : "No";

        std::cout << std::left << std::setw(12) << num_layers
                  << std::fixed << std::setprecision(2)
                  << std::setw(15) << time_ms
                  << std::setw(18) << time_per_layer
                  << std::setw(15) << linear_str
                  << "\n";
    }
    std::cout << std::endl;
}

int main(int argc, char** argv) {
    std::cout << "\n";
    std::cout << "========================================\n";
    std::cout << "  Tenzor RNN Benchmark Suite\n";
    std::cout << "========================================\n";
    std::cout << "\nTarget Performance Metrics:\n";
    std::cout << "  LSTM 2L 256:      < 10ms for seq=100, batch=32\n";
    std::cout << "  GRU speedup:      ~1.3x over LSTM\n";
    std::cout << "  Seq scaling:      Linear with sequence length\n";
    std::cout << "  Layer scaling:    Linear with layer count\n";
    std::cout << "  Backward ratio:   < 3x forward time\n";
    std::cout << "\n";

    try {
        initialize();

        benchmark_lstm_forward();
        benchmark_gru_forward();
        benchmark_lstm_backward();
        benchmark_sequence_length_scaling();
        benchmark_hidden_size_scaling();
        benchmark_lstm_vs_gru();
        benchmark_layer_scaling();

        std::cout << "\n========================================\n";
        std::cout << "  RNN Benchmark Complete\n";
        std::cout << "========================================\n\n";

        finalize();

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        finalize();
        return 1;
    }

    return 0;
}
