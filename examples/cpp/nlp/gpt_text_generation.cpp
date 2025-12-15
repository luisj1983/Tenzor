/**
 * @file gpt_text_generation.cpp
 * @brief Example demonstrating GPT text generation with different sampling strategies
 *
 * This example shows how to:
 * 1. Create and configure GPT-2 models
 * 2. Generate text using different sampling strategies (greedy, top-k, top-p, beam search)
 * 3. Compare generation quality across strategies
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <string>
#include <chrono>
#include "../../include/tenzor/models/gpt.hpp"
#include "../../include/tenzor/autograd/variable.hpp"
#include "../../include/tenzor/core/tensor.hpp"

using namespace tenzor;
using namespace tenzor::models;

/**
 * @brief Simple tokenizer for demonstration purposes.
 *
 * In real applications, use a proper tokenizer like BPE, SentencePiece, or WordPiece.
 */
class SimpleTokenizer {
public:
    SimpleTokenizer(int64_t vocab_size) : vocab_size_(vocab_size) {}

    std::vector<int64_t> encode(const std::string& text) {
        // Simple character-level encoding for demo
        std::vector<int64_t> tokens;
        for (char c : text) {
            tokens.push_back(static_cast<int64_t>(c) % vocab_size_);
        }
        return tokens;
    }

    std::string decode(const std::vector<int64_t>& tokens) {
        std::string text;
        for (int64_t token : tokens) {
            // Map back to ASCII range for display
            text += static_cast<char>((token % 95) + 32);
        }
        return text;
    }

private:
    int64_t vocab_size_;
};

/**
 * @brief Print generation results with formatting.
 */
void print_generation(const std::string& strategy_name,
                     const std::string& prompt,
                     const std::string& generated_text,
                     double time_ms) {
    std::cout << "\n" << std::string(80, '=') << "\n";
    std::cout << "Strategy: " << strategy_name << "\n";
    std::cout << std::string(80, '-') << "\n";
    std::cout << "Prompt: " << prompt << "\n";
    std::cout << "Generated: " << generated_text << "\n";
    std::cout << "Time: " << std::fixed << std::setprecision(2) << time_ms << " ms\n";
    std::cout << std::string(80, '=') << "\n";
}

/**
 * @brief Measure generation time.
 */
template<typename Func>
double measure_time(Func&& func) {
    auto start = std::chrono::high_resolution_clock::now();
    func();
    auto end = std::chrono::high_resolution_clock::now();
    return std::chrono::duration<double, std::milli>(end - start).count();
}

int main(int argc, char** argv) {
    std::cout << "GPT Text Generation Example\n";
    std::cout << "============================\n\n";

    // ============================================================================
    // 1. Model Setup
    // ============================================================================

    std::cout << "Setting up GPT-2 model...\n";

    // Create a small GPT-2 config for demonstration
    auto config = GPT2Config{};
    config.vocab_size = 5000;      // Small vocab for demo
    config.n_positions = 256;      // Max sequence length
    config.n_embd = 256;           // Hidden size
    config.n_layer = 4;            // Number of layers
    config.n_head = 8;             // Attention heads
    config.n_inner = 1024;         // FFN size
    config.attn_pdrop = 0.1;
    config.embd_pdrop = 0.1;
    config.resid_pdrop = 0.1;
    config.activation = "gelu";

    // Create model
    GPT2LMHeadModel model(config);
    model.eval();  // Set to evaluation mode

    std::cout << "Model created with:\n";
    std::cout << "  - Vocabulary size: " << config.vocab_size << "\n";
    std::cout << "  - Hidden size: " << config.n_embd << "\n";
    std::cout << "  - Layers: " << config.n_layer << "\n";
    std::cout << "  - Attention heads: " << config.n_head << "\n\n";

    // Count parameters
    auto params = model.parameters();
    int64_t total_params = 0;
    for (const auto& param : params) {
        total_params += param->tensor().numel();
    }
    std::cout << "Total parameters: " << total_params << "\n\n";

    // ============================================================================
    // 2. Tokenizer Setup
    // ============================================================================

    SimpleTokenizer tokenizer(config.vocab_size);

    // ============================================================================
    // 3. Text Generation with Different Strategies
    // ============================================================================

    std::string prompt = "Hello world";
    std::cout << "Input prompt: \"" << prompt << "\"\n\n";

    // Encode prompt
    auto prompt_tokens = tokenizer.encode(prompt);
    Tensor input_ids({1, static_cast<int64_t>(prompt_tokens.size())},
                     DType::Int64, Device::cpu());
    auto input_data = input_ids.data<int64_t>();
    for (size_t i = 0; i < prompt_tokens.size(); ++i) {
        input_data[i] = prompt_tokens[i];
    }

    // ============================================================================
    // 3.1. Greedy Search
    // ============================================================================

    std::cout << "Generating with Greedy Search...\n";

    GenerationConfig greedy_config;
    greedy_config.max_length = 50;
    greedy_config.strategy = GenerationStrategy::Greedy;

    TextGenerator greedy_generator(model, greedy_config);

    Tensor greedy_output;
    double greedy_time = measure_time([&]() {
        greedy_output = greedy_generator.greedy_search(input_ids);
    });

    // Decode
    std::vector<int64_t> greedy_tokens(greedy_config.max_length);
    auto greedy_data = greedy_output.data<int64_t>();
    for (int64_t i = 0; i < greedy_config.max_length; ++i) {
        greedy_tokens[i] = greedy_data[i];
    }
    std::string greedy_text = tokenizer.decode(greedy_tokens);

    print_generation("Greedy Search", prompt, greedy_text, greedy_time);

    // ============================================================================
    // 3.2. Top-K Sampling
    // ============================================================================

    std::cout << "\nGenerating with Top-K Sampling (k=50, temp=0.8)...\n";

    GenerationConfig topk_config;
    topk_config.max_length = 50;
    topk_config.strategy = GenerationStrategy::TopK;
    topk_config.top_k = 50;
    topk_config.temperature = 0.8;
    topk_config.seed = 42;

    TextGenerator topk_generator(model, topk_config);

    Tensor topk_output;
    double topk_time = measure_time([&]() {
        topk_output = topk_generator.top_k_sampling(input_ids, topk_config.top_k,
                                                     topk_config.temperature);
    });

    std::vector<int64_t> topk_tokens(topk_config.max_length);
    auto topk_data = topk_output.data<int64_t>();
    for (int64_t i = 0; i < topk_config.max_length; ++i) {
        topk_tokens[i] = topk_data[i];
    }
    std::string topk_text = tokenizer.decode(topk_tokens);

    print_generation("Top-K Sampling", prompt, topk_text, topk_time);

    // ============================================================================
    // 3.3. Top-P (Nucleus) Sampling
    // ============================================================================

    std::cout << "\nGenerating with Top-P Sampling (p=0.9, temp=0.9)...\n";

    GenerationConfig topp_config;
    topp_config.max_length = 50;
    topp_config.strategy = GenerationStrategy::TopP;
    topp_config.top_p = 0.9;
    topp_config.temperature = 0.9;
    topp_config.seed = 123;

    TextGenerator topp_generator(model, topp_config);

    Tensor topp_output;
    double topp_time = measure_time([&]() {
        topp_output = topp_generator.top_p_sampling(input_ids, topp_config.top_p,
                                                     topp_config.temperature);
    });

    std::vector<int64_t> topp_tokens(topp_config.max_length);
    auto topp_data = topp_output.data<int64_t>();
    for (int64_t i = 0; i < topp_config.max_length; ++i) {
        topp_tokens[i] = topp_data[i];
    }
    std::string topp_text = tokenizer.decode(topp_tokens);

    print_generation("Top-P (Nucleus) Sampling", prompt, topp_text, topp_time);

    // ============================================================================
    // 3.4. Beam Search
    // ============================================================================

    std::cout << "\nGenerating with Beam Search (beams=4)...\n";

    GenerationConfig beam_config;
    beam_config.max_length = 50;
    beam_config.strategy = GenerationStrategy::BeamSearch;
    beam_config.num_beams = 4;

    TextGenerator beam_generator(model, beam_config);

    Tensor beam_output;
    double beam_time = measure_time([&]() {
        beam_output = beam_generator.beam_search(input_ids, beam_config.num_beams);
    });

    std::vector<int64_t> beam_tokens(beam_config.max_length);
    auto beam_data = beam_output.data<int64_t>();
    for (int64_t i = 0; i < beam_config.max_length; ++i) {
        beam_tokens[i] = beam_data[i];
    }
    std::string beam_text = tokenizer.decode(beam_tokens);

    print_generation("Beam Search", prompt, beam_text, beam_time);

    // ============================================================================
    // 4. Comparison Summary
    // ============================================================================

    std::cout << "\nGeneration Strategy Comparison\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << std::left << std::setw(25) << "Strategy"
              << std::setw(20) << "Time (ms)"
              << std::setw(35) << "Characteristics\n";
    std::cout << std::string(80, '-') << "\n";

    std::cout << std::setw(25) << "Greedy Search"
              << std::setw(20) << greedy_time
              << "Deterministic, fast, may be repetitive\n";

    std::cout << std::setw(25) << "Top-K Sampling"
              << std::setw(20) << topk_time
              << "Stochastic, diverse, controlled\n";

    std::cout << std::setw(25) << "Top-P Sampling"
              << std::setw(20) << topp_time
              << "Stochastic, adaptive, high quality\n";

    std::cout << std::setw(25) << "Beam Search"
              << std::setw(20) << beam_time
              << "Deterministic, slower, best quality\n";

    std::cout << std::string(80, '=') << "\n";

    // ============================================================================
    // 5. Model Configurations Demo
    // ============================================================================

    std::cout << "\nAvailable GPT Model Configurations\n";
    std::cout << std::string(80, '=') << "\n";

    std::cout << "\nGPT-2 Variants:\n";
    std::cout << "  - GPT-2 Small:  768 dim, 12 layers, 12 heads (~117M params)\n";
    std::cout << "  - GPT-2 Medium: 1024 dim, 24 layers, 16 heads (~345M params)\n";
    std::cout << "  - GPT-2 Large:  1280 dim, 36 layers, 20 heads (~774M params)\n";
    std::cout << "  - GPT-2 XL:     1600 dim, 48 layers, 25 heads (~1.5B params)\n";

    std::cout << "\nGPT-3 Variants:\n";
    std::cout << "  - GPT-3 125M:   768 dim, 12 layers\n";
    std::cout << "  - GPT-3 350M:   1024 dim, 24 layers\n";
    std::cout << "  - GPT-3 760M:   1536 dim, 24 layers\n";
    std::cout << "  - GPT-3 1.3B:   2048 dim, 24 layers\n";
    std::cout << "  - GPT-3 2.7B:   2560 dim, 32 layers\n";
    std::cout << "  - GPT-3 6.7B:   4096 dim, 32 layers\n";
    std::cout << "  - GPT-3 13B:    5120 dim, 40 layers\n";
    std::cout << "  - GPT-3 175B:   12288 dim, 96 layers\n";

    // ============================================================================
    // 6. Usage Tips
    // ============================================================================

    std::cout << "\nUsage Tips\n";
    std::cout << std::string(80, '=') << "\n";
    std::cout << "1. Temperature:\n";
    std::cout << "   - Lower (0.1-0.7): More focused, deterministic\n";
    std::cout << "   - Medium (0.7-1.0): Balanced creativity\n";
    std::cout << "   - Higher (1.0-2.0): More random, creative\n\n";

    std::cout << "2. Top-K vs Top-P:\n";
    std::cout << "   - Top-K: Fixed number of candidates (good for controlled generation)\n";
    std::cout << "   - Top-P: Dynamic size based on probability mass (more adaptive)\n\n";

    std::cout << "3. Beam Search:\n";
    std::cout << "   - Best for tasks requiring high quality (translation, summarization)\n";
    std::cout << "   - Slower but more coherent outputs\n";
    std::cout << "   - Increase num_beams for better quality (at cost of speed)\n\n";

    std::cout << "4. Loading Pretrained Weights:\n";
    std::cout << "   - Use models::ModelHub::load_pretrained_weights(model, \"gpt2\")\n";
    std::cout << "   - Supports loading from Hugging Face format\n";
    std::cout << "   - Fine-tune on your specific task for best results\n";

    std::cout << "\nExample completed successfully!\n";

    return 0;
}
