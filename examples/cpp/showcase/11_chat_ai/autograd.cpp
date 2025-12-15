/**
 * @file autograd.cpp
 * @brief Chat AI using Tenzor's automatic differentiation
 *
 * This example demonstrates building a character-level sequence-to-sequence
 * chatbot using Variable and autograd for automatic gradient computation.
 *
 * Architecture: Encoder-Decoder RNN with character-level encoding
 * Training data: Cornell Movie Dialogs Corpus
 *
 * Usage: ./11_chat_ai_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <fstream>
#include <sstream>
#include <unordered_map>
#include <algorithm>
#include <random>

using namespace tenzor;

// Special tokens
const char PAD_CHAR = '\0';
const char SOS_CHAR = '\x01';
const char EOS_CHAR = '\x02';

/**
 * @brief Character vocabulary for encoding/decoding text
 */
class CharVocab {
public:
    CharVocab() {
        add_char(PAD_CHAR);
        add_char(SOS_CHAR);
        add_char(EOS_CHAR);
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
            std::transform(q.begin(), q.end(), q.begin(), ::tolower);
            std::transform(a.begin(), a.end(), a.begin(), ::tolower);
            if (q.length() <= 30 && a.length() <= 30) {
                pairs.push_back({q, a});
            }
        }
    }

    return pairs;
}

/**
 * @brief GRU cell with autograd support
 */
Variable gru_cell(const Variable& x, const Variable& h_prev,
                  const Variable& Wz, const Variable& Uz, const Variable& bz,
                  const Variable& Wr, const Variable& Ur, const Variable& br,
                  const Variable& Wh, const Variable& Uh, const Variable& bh) {
    // Update gate
    auto z = nn::sigmoid(matmul(x, Wz) + matmul(h_prev, Uz) + bz);

    // Reset gate
    auto r = nn::sigmoid(matmul(x, Wr) + matmul(h_prev, Ur) + br);

    // Candidate hidden
    auto h_tilde = nn::tanh(matmul(x, Wh) + matmul(r * h_prev, Uh) + bh);

    // New hidden state
    auto ones_tensor = ones_like(z.tensor());
    Variable ones_var(ones_tensor, false);
    auto h_new = (ones_var - z) * h_prev + z * h_tilde;

    return h_new;
}

/**
 * @brief Create one-hot encoding as Variable
 */
Variable one_hot_var(int idx, int vocab_size, Device device, bool requires_grad = false) {
    std::vector<float> data(vocab_size, 0.0f);
    data[idx] = 1.0f;
    auto tensor = from_data(data.data(), {1, vocab_size}, device);
    return Variable(tensor, requires_grad);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Chat AI - Autograd (Automatic Differentiation)", device);

    manual_seed(42);

    // Hyperparameters
    int embed_size = 128;
    int hidden_size = 256;
    float learning_rate = 0.03f;
    int num_epochs = 50;
    int print_every = 5;

    // Build vocabulary
    CharVocab vocab;
    int vocab_size = vocab.size();

    std::cout << "Vocabulary size: " << vocab_size << " characters\n";

    // Load full training data
    auto data = load_training_data("data/chat_training_large.txt", 950);
    std::cout << "Training pairs: " << data.size() << "\n\n";

    showcase::print_section("Model Architecture");
    std::cout << "Encoder-Decoder GRU with autograd\n";
    std::cout << "  Embedding: " << vocab_size << " -> " << embed_size << "\n";
    std::cout << "  Hidden: " << hidden_size << "\n";
    std::cout << "  Output: " << hidden_size << " -> " << vocab_size << "\n";

    // Initialize weights as Variables with requires_grad=true
    Variable embedding(randn({vocab_size, embed_size}, DType::Float32, device) * 0.1f, true);

    // GRU weights
    Variable Wz(randn({embed_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable Uz(randn({hidden_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable bz(zeros({1, hidden_size}, DType::Float32, device), true);

    Variable Wr(randn({embed_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable Ur(randn({hidden_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable br(zeros({1, hidden_size}, DType::Float32, device), true);

    Variable Wh(randn({embed_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable Uh(randn({hidden_size, hidden_size}, DType::Float32, device) * 0.1f, true);
    Variable bh(zeros({1, hidden_size}, DType::Float32, device), true);

    Variable output_proj(randn({hidden_size, vocab_size}, DType::Float32, device) * 0.1f, true);

    // Collect all parameters
    std::vector<Variable*> params = {
        &embedding, &Wz, &Uz, &bz, &Wr, &Ur, &br, &Wh, &Uh, &bh, &output_proj
    };

    showcase::print_section("Training");
    std::cout << "Autograd computes gradients automatically through BPTT\n\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        int total_tokens = 0;

        std::shuffle(data.begin(), data.end(), gen);

        for (const auto& [question, answer] : data) {
            // Zero gradients
            for (auto* p : params) {
                p->zero_grad();
            }

            // Encode question
            Variable h(zeros({1, hidden_size}, DType::Float32, device), false);

            for (char c : question) {
                auto x = one_hot_var(vocab.encode(c), vocab_size, device, false);
                auto embedded = matmul(x, embedding);
                h = gru_cell(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
            }

            // Decode with teacher forcing
            std::string target = std::string(1, SOS_CHAR) + answer + std::string(1, EOS_CHAR);
            Variable loss_sum(zeros({1}, DType::Float32, device), true);

            for (size_t t = 0; t < target.length() - 1; ++t) {
                int input_token = vocab.encode(target[t]);
                int target_token = vocab.encode(target[t + 1]);

                auto x = one_hot_var(input_token, vocab_size, device, false);
                auto embedded = matmul(x, embedding);

                h = gru_cell(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);

                // Compute logits and loss
                auto logits = matmul(h, output_proj);
                auto log_probs = log_softmax(logits, 1);

                // Extract target log probability
                auto target_one_hot = one_hot_var(target_token, vocab_size, device, false);
                auto target_log_prob = sum(log_probs * target_one_hot);

                loss_sum = loss_sum - target_log_prob;
                total_tokens++;
            }

            // Backward pass (automatic!)
            loss_sum.backward();

            total_loss += loss_sum.tensor().item<float>();

            // SGD update
            {
                NoGradGuard no_grad;
                for (auto* p : params) {
                    if (p->has_grad()) {
                        auto new_tensor = p->tensor() - (*p->grad() * learning_rate);
                        *p = Variable(new_tensor, true);
                    }
                }
            }
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float avg_loss = total_loss / total_tokens;
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs << "] "
                      << "Loss: " << avg_loss << "\n";
        }
    }

    // Interactive chat
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
        Variable h(zeros({1, hidden_size}, DType::Float32, device), false);
        for (char c : user_input) {
            auto x = one_hot_var(vocab.encode(c), vocab_size, device, false);
            auto embedded = matmul(x, embedding);
            h = gru_cell(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
        }

        // Decode (greedy)
        std::string response;
        int current_token = vocab.sos_idx();

        for (int t = 0; t < 50; ++t) {
            auto x = one_hot_var(current_token, vocab_size, device, false);
            auto embedded = matmul(x, embedding);
            h = gru_cell(embedded, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);

            auto logits = matmul(h, output_proj);
            auto logits_cpu = logits.tensor().cpu();
            const float* data = logits_cpu.data<float>();

            int best_idx = 0;
            float best_val = data[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (data[i] > best_val) {
                    best_val = data[i];
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

    std::cout << "\nChat AI demonstrated with autograd!\n";
    std::cout << "Gradients computed automatically through the entire sequence.\n";

    finalize();
    return 0;
}
