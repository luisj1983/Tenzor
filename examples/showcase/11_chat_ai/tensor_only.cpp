/**
 * @file tensor_only.cpp
 * @brief Chat AI using raw tensor operations with manual backpropagation
 *
 * This example demonstrates building a character-level sequence-to-sequence
 * chatbot using only raw tensor operations. All gradients are computed manually
 * through Backpropagation Through Time (BPTT).
 *
 * Architecture: Encoder-Decoder RNN with character-level encoding
 * Training data: Cornell Movie Dialogs Corpus
 *
 * Usage: ./11_chat_ai_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <random>

using namespace tenzor;

// ============ Manual Activation Functions for Tensor Operations ============

// Manual sigmoid: 1 / (1 + exp(-x))
Tensor sigmoid_tensor(const Tensor& x) {
    auto neg_x = zeros_like(x) - x;
    return ones_like(x) / (ones_like(x) + tenzor::exp(neg_x));
}

// Sigmoid derivative: sigmoid(x) * (1 - sigmoid(x))
Tensor sigmoid_deriv(const Tensor& sigmoid_out) {
    return sigmoid_out * (ones_like(sigmoid_out) - sigmoid_out);
}

// Manual tanh: (exp(x) - exp(-x)) / (exp(x) + exp(-x))
Tensor tanh_tensor(const Tensor& x) {
    auto neg_x = zeros_like(x) - x;
    auto exp_x = tenzor::exp(x);
    auto exp_neg_x = tenzor::exp(neg_x);
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x);
}

// Tanh derivative: 1 - tanh(x)^2
Tensor tanh_deriv(const Tensor& tanh_out) {
    return ones_like(tanh_out) - tanh_out * tanh_out;
}

// Manual softmax with numerical stability
Tensor softmax_tensor(const Tensor& x, int dim = 1) {
    auto max_x = tenzor::max(x, dim, true);
    auto x_stable = x - max_x;
    auto exp_x = tenzor::exp(x_stable);
    auto sum_exp = tenzor::sum(exp_x, dim, true);
    return exp_x / sum_exp;
}

// Gradient clipping by value
Tensor clip_gradient(const Tensor& grad, float max_val) {
    auto clipped = tenzor::clamp(grad, -max_val, max_val);
    return clipped;
}

// Special tokens
const char PAD_CHAR = '\0';
const char SOS_CHAR = '\x01';  // Start of sequence
const char EOS_CHAR = '\x02';  // End of sequence

/**
 * @brief Character vocabulary for encoding/decoding text
 */
class CharVocab {
public:
    CharVocab() {
        // Add special characters
        add_char(PAD_CHAR);  // 0 = padding
        add_char(SOS_CHAR);  // 1 = start
        add_char(EOS_CHAR);  // 2 = end

        // Add printable ASCII characters
        for (char c = ' '; c <= '~'; ++c) {
            add_char(c);
        }
    }

    void add_char(char c) {
        if (char_to_idx_.find(c) == char_to_idx_.end()) {
            int idx = static_cast<int>(idx_to_char_.size());
            char_to_idx_[c] = idx;
            idx_to_char_.push_back(c);
        }
    }

    int encode(char c) const {
        auto it = char_to_idx_.find(c);
        return (it != char_to_idx_.end()) ? it->second : char_to_idx_.at(' ');
    }

    char decode(int idx) const {
        return (idx >= 0 && idx < static_cast<int>(idx_to_char_.size())) ?
               idx_to_char_[idx] : ' ';
    }

    int size() const { return static_cast<int>(idx_to_char_.size()); }
    int pad_idx() const { return 0; }
    int sos_idx() const { return 1; }
    int eos_idx() const { return 2; }

private:
    std::unordered_map<char, int> char_to_idx_;
    std::vector<char> idx_to_char_;
};

/**
 * @brief Load Q&A pairs from training file
 */
std::vector<std::pair<std::string, std::string>> load_training_data(
    const std::string& filepath, int max_pairs = 100) {

    std::vector<std::pair<std::string, std::string>> pairs;
    std::ifstream file(filepath);

    if (!file.is_open()) {
        std::cerr << "Warning: Could not open " << filepath << "\n";
        // Return some built-in examples
        pairs.push_back({"hello", "hi there"});
        pairs.push_back({"how are you", "im good thanks"});
        pairs.push_back({"whats your name", "im a chatbot"});
        pairs.push_back({"goodbye", "see you later"});
        pairs.push_back({"thanks", "youre welcome"});
        return pairs;
    }

    std::string line;
    while (std::getline(file, line) && pairs.size() < static_cast<size_t>(max_pairs)) {
        size_t tab_pos = line.find('\t');
        if (tab_pos != std::string::npos) {
            std::string q = line.substr(0, tab_pos);
            std::string a = line.substr(tab_pos + 1);
            // Convert to lowercase and limit length
            std::transform(q.begin(), q.end(), q.begin(), ::tolower);
            std::transform(a.begin(), a.end(), a.begin(), ::tolower);
            if (q.length() <= 20 && a.length() <= 20) {
                pairs.push_back({q, a});
            }
        }
    }

    return pairs;
}

/**
 * @brief GRU cell state - stores intermediate values for backprop
 */
struct GRUState {
    Tensor z;       // Update gate output
    Tensor r;       // Reset gate output
    Tensor h_tilde; // Candidate hidden state
    Tensor h;       // Output hidden state
    Tensor h_prev;  // Previous hidden state (input)
    Tensor x;       // Input
};

/**
 * @brief GRU cell forward pass with state caching for backprop
 */
GRUState gru_forward(const Tensor& x, const Tensor& h_prev,
                     const Tensor& Wz, const Tensor& Uz, const Tensor& bz,
                     const Tensor& Wr, const Tensor& Ur, const Tensor& br,
                     const Tensor& Wh, const Tensor& Uh, const Tensor& bh) {
    GRUState state;
    state.x = x;
    state.h_prev = h_prev;

    // Update gate: z = sigmoid(x @ Wz + h_prev @ Uz + bz)
    state.z = sigmoid_tensor(matmul(x, Wz) + matmul(h_prev, Uz) + bz);

    // Reset gate: r = sigmoid(x @ Wr + h_prev @ Ur + br)
    state.r = sigmoid_tensor(matmul(x, Wr) + matmul(h_prev, Ur) + br);

    // Candidate hidden: h_tilde = tanh(x @ Wh + (r * h_prev) @ Uh + bh)
    state.h_tilde = tanh_tensor(matmul(x, Wh) + matmul(state.r * h_prev, Uh) + bh);

    // New hidden: h = (1 - z) * h_prev + z * h_tilde
    state.h = (ones_like(state.z) - state.z) * h_prev + state.z * state.h_tilde;

    return state;
}

/**
 * @brief GRU cell backward pass - computes gradients
 */
void gru_backward(const GRUState& state, const Tensor& dL_dh,
                  const Tensor& Wz, const Tensor& Uz,
                  const Tensor& Wr, const Tensor& Ur,
                  const Tensor& Wh, const Tensor& Uh,
                  // Output gradients
                  Tensor& dL_dWz, Tensor& dL_dUz, Tensor& dL_dbz,
                  Tensor& dL_dWr, Tensor& dL_dUr, Tensor& dL_dbr,
                  Tensor& dL_dWh, Tensor& dL_dUh, Tensor& dL_dbh,
                  Tensor& dL_dh_prev, Tensor& dL_dx) {

    // Gradient through h = (1-z)*h_prev + z*h_tilde
    auto dL_dz = dL_dh * (state.h_tilde - state.h_prev);
    auto dL_dh_tilde = dL_dh * state.z;
    auto dL_dh_prev_direct = dL_dh * (ones_like(state.z) - state.z);

    // Gradient through h_tilde = tanh(...)
    auto dL_dpre_h_tilde = dL_dh_tilde * tanh_deriv(state.h_tilde);

    // Gradient through candidate computation
    dL_dWh = dL_dWh + matmul(state.x.transpose(0, 1), dL_dpre_h_tilde);
    auto r_h_prev = state.r * state.h_prev;
    dL_dUh = dL_dUh + matmul(r_h_prev.transpose(0, 1), dL_dpre_h_tilde);
    dL_dbh = dL_dbh + tenzor::sum(dL_dpre_h_tilde, 0, true);

    auto dL_dr_h_prev = matmul(dL_dpre_h_tilde, Uh.transpose(0, 1));
    auto dL_dr = dL_dr_h_prev * state.h_prev;
    auto dL_dh_prev_from_h_tilde = dL_dr_h_prev * state.r;

    // Gradient through z = sigmoid(...)
    auto dL_dpre_z = dL_dz * sigmoid_deriv(state.z);
    dL_dWz = dL_dWz + matmul(state.x.transpose(0, 1), dL_dpre_z);
    dL_dUz = dL_dUz + matmul(state.h_prev.transpose(0, 1), dL_dpre_z);
    dL_dbz = dL_dbz + tenzor::sum(dL_dpre_z, 0, true);
    auto dL_dh_prev_from_z = matmul(dL_dpre_z, Uz.transpose(0, 1));

    // Gradient through r = sigmoid(...)
    auto dL_dpre_r = dL_dr * sigmoid_deriv(state.r);
    dL_dWr = dL_dWr + matmul(state.x.transpose(0, 1), dL_dpre_r);
    dL_dUr = dL_dUr + matmul(state.h_prev.transpose(0, 1), dL_dpre_r);
    dL_dbr = dL_dbr + tenzor::sum(dL_dpre_r, 0, true);
    auto dL_dh_prev_from_r = matmul(dL_dpre_r, Ur.transpose(0, 1));

    // Total gradient w.r.t. h_prev
    dL_dh_prev = dL_dh_prev_direct + dL_dh_prev_from_h_tilde +
                 dL_dh_prev_from_z + dL_dh_prev_from_r;

    // Gradient w.r.t. input x
    dL_dx = matmul(dL_dpre_z, Wz.transpose(0, 1)) +
            matmul(dL_dpre_r, Wr.transpose(0, 1)) +
            matmul(dL_dpre_h_tilde, Wh.transpose(0, 1));
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Chat AI - Tensor Only (Manual BPTT)", device);

    manual_seed(42);

    // Hyperparameters - balanced for manual BPTT performance
    int embed_size = 64;
    int hidden_size = 128;
    float learning_rate = 0.015f;
    float grad_clip = 1.0f;
    int num_epochs = 40;
    int print_every = 10;

    // Build vocabulary
    CharVocab vocab;
    int vocab_size = vocab.size();

    std::cout << "Vocabulary size: " << vocab_size << " characters\n";

    // Load training data (200 pairs from large dataset - manual BPTT is compute-intensive)
    auto data = load_training_data("data/chat_training_large.txt", 200);
    std::cout << "Training pairs: " << data.size() << "\n\n";

    showcase::print_section("Model Architecture");
    std::cout << "Encoder-Decoder GRU with manual BPTT\n";
    std::cout << "  Embedding: " << vocab_size << " -> " << embed_size << "\n";
    std::cout << "  Hidden: " << hidden_size << "\n";
    std::cout << "  Output: " << hidden_size << " -> " << vocab_size << "\n";

    // Initialize weights (Xavier-like initialization)
    float init_scale = 0.1f;
    auto embedding = randn({vocab_size, embed_size}, DType::Float32, device) * init_scale;

    // GRU weights
    auto Wz = randn({embed_size, hidden_size}, DType::Float32, device) * init_scale;
    auto Uz = randn({hidden_size, hidden_size}, DType::Float32, device) * init_scale;
    auto bz = zeros({1, hidden_size}, DType::Float32, device);

    auto Wr = randn({embed_size, hidden_size}, DType::Float32, device) * init_scale;
    auto Ur = randn({hidden_size, hidden_size}, DType::Float32, device) * init_scale;
    auto br = zeros({1, hidden_size}, DType::Float32, device);

    auto Wh = randn({embed_size, hidden_size}, DType::Float32, device) * init_scale;
    auto Uh = randn({hidden_size, hidden_size}, DType::Float32, device) * init_scale;
    auto bh = zeros({1, hidden_size}, DType::Float32, device);

    auto output_proj = randn({hidden_size, vocab_size}, DType::Float32, device) * init_scale;

    showcase::print_section("Training");
    std::cout << "Using full manual BPTT through GRU cells\n\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        int total_tokens = 0;

        // Shuffle data
        std::shuffle(data.begin(), data.end(), gen);

        for (const auto& [question, answer] : data) {
            // Initialize gradient accumulators
            auto dL_dWz = zeros_like(Wz);
            auto dL_dUz = zeros_like(Uz);
            auto dL_dbz = zeros_like(bz);
            auto dL_dWr = zeros_like(Wr);
            auto dL_dUr = zeros_like(Ur);
            auto dL_dbr = zeros_like(br);
            auto dL_dWh = zeros_like(Wh);
            auto dL_dUh = zeros_like(Uh);
            auto dL_dbh = zeros_like(bh);
            auto dL_doutput_proj = zeros_like(output_proj);
            auto dL_dembedding = zeros_like(embedding);

            // ============ Encoder Forward Pass ============
            std::vector<GRUState> encoder_states;
            auto h = zeros({1, hidden_size}, DType::Float32, device);

            for (char c : question) {
                int token = vocab.encode(c);
                std::vector<float> one_hot(vocab_size, 0.0f);
                one_hot[token] = 1.0f;
                auto x = from_data(one_hot.data(), {1, vocab_size}, device);
                auto embedded = matmul(x, embedding);

                auto state = gru_forward(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
                encoder_states.push_back(state);
                h = state.h;
            }

            // ============ Decoder Forward Pass ============
            std::vector<GRUState> decoder_states;
            std::vector<Tensor> logits_list;
            std::vector<int> target_tokens;

            std::string target = std::string(1, SOS_CHAR) + answer + std::string(1, EOS_CHAR);

            for (size_t t = 0; t < target.length() - 1; ++t) {
                int input_token = vocab.encode(target[t]);
                int target_token = vocab.encode(target[t + 1]);
                target_tokens.push_back(target_token);

                std::vector<float> one_hot(vocab_size, 0.0f);
                one_hot[input_token] = 1.0f;
                auto x = from_data(one_hot.data(), {1, vocab_size}, device);
                auto embedded = matmul(x, embedding);

                auto state = gru_forward(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
                decoder_states.push_back(state);
                h = state.h;

                auto logits = matmul(h, output_proj);
                logits_list.push_back(logits);
            }

            // ============ Compute Loss and Output Gradients ============
            float seq_loss = 0.0f;
            std::vector<Tensor> dL_dlogits_list;

            for (size_t t = 0; t < decoder_states.size(); ++t) {
                auto probs = softmax_tensor(logits_list[t], 1);

                // Cross-entropy loss
                auto probs_cpu = probs.cpu();
                const float* prob_data = probs_cpu.data<float>();
                float target_prob = prob_data[target_tokens[t]];
                seq_loss -= std::log(target_prob + 1e-10f);

                // Gradient of cross-entropy w.r.t. logits: probs - one_hot(target)
                std::vector<float> target_oh(vocab_size, 0.0f);
                target_oh[target_tokens[t]] = 1.0f;
                auto target_tensor = from_data(target_oh.data(), {1, vocab_size}, device);
                auto dL_dlogits = probs - target_tensor;
                dL_dlogits_list.push_back(dL_dlogits);

                total_tokens++;
            }

            total_loss += seq_loss;

            // ============ Backward Pass Through Decoder ============
            auto dL_dh = zeros({1, hidden_size}, DType::Float32, device);

            for (int t = static_cast<int>(decoder_states.size()) - 1; t >= 0; --t) {
                // Gradient from output projection
                auto dL_dlogits = dL_dlogits_list[t];
                dL_doutput_proj = dL_doutput_proj + matmul(decoder_states[t].h.transpose(0, 1), dL_dlogits);
                auto dL_dh_from_output = matmul(dL_dlogits, output_proj.transpose(0, 1));

                // Add gradient from next timestep
                auto dL_dh_total = dL_dh + dL_dh_from_output;

                // Backprop through GRU
                Tensor dL_dh_prev, dL_dx;
                gru_backward(decoder_states[t], dL_dh_total,
                            Wz, Uz, Wr, Ur, Wh, Uh,
                            dL_dWz, dL_dUz, dL_dbz,
                            dL_dWr, dL_dUr, dL_dbr,
                            dL_dWh, dL_dUh, dL_dbh,
                            dL_dh_prev, dL_dx);

                // Gradient to embedding (for input token at this step)
                int input_token = vocab.encode(target[t]);
                std::vector<float> one_hot(vocab_size, 0.0f);
                one_hot[input_token] = 1.0f;
                auto input_one_hot = from_data(one_hot.data(), {1, vocab_size}, device);
                dL_dembedding = dL_dembedding + matmul(input_one_hot.transpose(0, 1), dL_dx);

                dL_dh = dL_dh_prev;
            }

            // ============ Backward Pass Through Encoder (truncated) ============
            int bptt_steps = std::min(5, static_cast<int>(encoder_states.size()));
            for (int t = static_cast<int>(encoder_states.size()) - 1;
                 t >= static_cast<int>(encoder_states.size()) - bptt_steps && t >= 0; --t) {

                Tensor dL_dh_prev, dL_dx;
                gru_backward(encoder_states[t], dL_dh,
                            Wz, Uz, Wr, Ur, Wh, Uh,
                            dL_dWz, dL_dUz, dL_dbz,
                            dL_dWr, dL_dUr, dL_dbr,
                            dL_dWh, dL_dUh, dL_dbh,
                            dL_dh_prev, dL_dx);

                // Gradient to embedding
                int token = vocab.encode(question[t]);
                std::vector<float> one_hot(vocab_size, 0.0f);
                one_hot[token] = 1.0f;
                auto input_one_hot = from_data(one_hot.data(), {1, vocab_size}, device);
                dL_dembedding = dL_dembedding + matmul(input_one_hot.transpose(0, 1), dL_dx);

                dL_dh = dL_dh_prev;
            }

            // ============ Update Weights (SGD with gradient clipping) ============
            embedding = embedding - clip_gradient(dL_dembedding, grad_clip) * learning_rate;
            Wz = Wz - clip_gradient(dL_dWz, grad_clip) * learning_rate;
            Uz = Uz - clip_gradient(dL_dUz, grad_clip) * learning_rate;
            bz = bz - clip_gradient(dL_dbz, grad_clip) * learning_rate;
            Wr = Wr - clip_gradient(dL_dWr, grad_clip) * learning_rate;
            Ur = Ur - clip_gradient(dL_dUr, grad_clip) * learning_rate;
            br = br - clip_gradient(dL_dbr, grad_clip) * learning_rate;
            Wh = Wh - clip_gradient(dL_dWh, grad_clip) * learning_rate;
            Uh = Uh - clip_gradient(dL_dUh, grad_clip) * learning_rate;
            bh = bh - clip_gradient(dL_dbh, grad_clip) * learning_rate;
            output_proj = output_proj - clip_gradient(dL_doutput_proj, grad_clip) * learning_rate;
        }

        float avg_loss = total_loss / total_tokens;

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Loss: " << avg_loss << "\n";
        }
    }

    // ============ Interactive Chat ============
    showcase::print_section("Interactive Chat");
    std::cout << "Chat with the trained model! Type 'quit' or 'exit' to stop.\n\n";

    std::string user_input;
    while (true) {
        std::cout << "You: ";
        std::getline(std::cin, user_input);

        // Check for exit commands
        if (user_input == "quit" || user_input == "exit" || user_input.empty()) {
            std::cout << "\nGoodbye!\n";
            break;
        }

        // Convert to lowercase
        std::transform(user_input.begin(), user_input.end(), user_input.begin(), ::tolower);

        // Encode user input
        auto h = zeros({1, hidden_size}, DType::Float32, device);
        for (char c : user_input) {
            int token = vocab.encode(c);
            std::vector<float> one_hot(vocab_size, 0.0f);
            one_hot[token] = 1.0f;
            auto x = from_data(one_hot.data(), {1, vocab_size}, device);
            auto embedded = matmul(x, embedding);
            auto state = gru_forward(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
            h = state.h;
        }

        // Decode (greedy)
        std::string response;
        int current_token = vocab.sos_idx();

        for (int t = 0; t < 50; ++t) {
            std::vector<float> one_hot(vocab_size, 0.0f);
            one_hot[current_token] = 1.0f;
            auto x = from_data(one_hot.data(), {1, vocab_size}, device);
            auto embedded = matmul(x, embedding);
            auto state = gru_forward(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
            h = state.h;

            auto logits = matmul(h, output_proj);
            auto logits_cpu = logits.cpu();
            const float* logit_data = logits_cpu.data<float>();

            int best_idx = 0;
            float best_val = logit_data[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (logit_data[i] > best_val) {
                    best_val = logit_data[i];
                    best_idx = i;
                }
            }

            if (best_idx == vocab.eos_idx()) break;

            char c = vocab.decode(best_idx);
            if (c >= ' ' && c <= '~') {
                response += c;
            }
            current_token = best_idx;
        }

        std::cout << "Bot: " << (response.empty() ? "(no response)" : response) << "\n\n";
    }

    std::cout << "\nChat AI demonstrated with raw tensors and manual BPTT!\n";
    std::cout << "All gradients computed manually through GRU gates.\n";

    finalize();
    return 0;
}
