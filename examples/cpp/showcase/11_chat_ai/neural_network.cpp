/**
 * @file neural_network.cpp
 * @brief Chat AI using Tenzor's high-level Neural Network API
 *
 * This example demonstrates building a character-level sequence-to-sequence
 * chatbot using nn::Module, nn::GRUCell, nn::Linear, and optimizers.
 *
 * Architecture: Encoder-Decoder GRU with character-level encoding
 * Training data: Cornell Movie Dialogs Corpus
 *
 * Usage: ./11_chat_ai_neural_network --backend cpu|cuda|vulkan
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
 * @brief Multi-layer Encoder module using stacked GRUCells
 */
class Encoder : public nn::Module {
public:
    Encoder(int vocab_size, int embed_size, int hidden_size, int num_layers = 2)
        : vocab_size_(vocab_size), hidden_size_(hidden_size), num_layers_(num_layers) {

        embedding_ = std::make_shared<nn::Linear>(vocab_size, embed_size, false);
        register_module("embedding", embedding_);

        // Create stacked GRU layers
        for (int i = 0; i < num_layers; ++i) {
            int input_size = (i == 0) ? embed_size : hidden_size;
            auto gru = std::make_shared<nn::GRUCell>(input_size, hidden_size);
            gru_layers_.push_back(gru);
            register_module("gru_layer_" + std::to_string(i), gru);
        }
    }

    auto forward_impl(const Variable& input) -> Variable override {
        return input;
    }

    // Returns vector of hidden states, one per layer
    auto encode(const std::vector<int>& tokens, Device device) -> std::vector<Variable> {
        // Initialize hidden states for each layer
        std::vector<Variable> h_states;
        for (int i = 0; i < num_layers_; ++i) {
            h_states.push_back(Variable(zeros({1, hidden_size_}, DType::Float32, device), false));
        }

        for (int token : tokens) {
            // One-hot encode
            std::vector<float> one_hot(vocab_size_, 0.0f);
            one_hot[token] = 1.0f;
            auto x = Variable(from_data(one_hot.data(), {1, vocab_size_}, device), false);

            // Embed input
            auto layer_input = embedding_->forward(x);

            // Process through each GRU layer
            for (int i = 0; i < num_layers_; ++i) {
                h_states[i] = gru_layers_[i]->forward(layer_input, h_states[i]);
                layer_input = h_states[i];  // Output becomes input to next layer
            }
        }

        return h_states;
    }

    int hidden_size() const { return hidden_size_; }
    int num_layers() const { return num_layers_; }

private:
    int vocab_size_;
    int hidden_size_;
    int num_layers_;
    std::shared_ptr<nn::Linear> embedding_;
    std::vector<std::shared_ptr<nn::GRUCell>> gru_layers_;
};

/**
 * @brief Multi-layer Decoder module using stacked GRUCells
 */
class Decoder : public nn::Module {
public:
    Decoder(int vocab_size, int embed_size, int hidden_size, int num_layers = 2)
        : vocab_size_(vocab_size), hidden_size_(hidden_size), num_layers_(num_layers) {

        embedding_ = std::make_shared<nn::Linear>(vocab_size, embed_size, false);
        register_module("embedding", embedding_);

        // Create stacked GRU layers
        for (int i = 0; i < num_layers; ++i) {
            int input_size = (i == 0) ? embed_size : hidden_size;
            auto gru = std::make_shared<nn::GRUCell>(input_size, hidden_size);
            gru_layers_.push_back(gru);
            register_module("gru_layer_" + std::to_string(i), gru);
        }

        output_proj_ = std::make_shared<nn::Linear>(hidden_size, vocab_size);
        register_module("output_proj", output_proj_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        return input;
    }

    // Step function now takes vector of hidden states (one per layer)
    auto step(int token, std::vector<Variable>& h_states, Device device) -> Variable {
        std::vector<float> one_hot(vocab_size_, 0.0f);
        one_hot[token] = 1.0f;
        auto x = Variable(from_data(one_hot.data(), {1, vocab_size_}, device), false);

        auto layer_input = embedding_->forward(x);

        // Process through each GRU layer
        for (int i = 0; i < num_layers_; ++i) {
            h_states[i] = gru_layers_[i]->forward(layer_input, h_states[i]);
            layer_input = h_states[i];
        }

        // Output projection from top layer
        auto logits = output_proj_->forward(h_states[num_layers_ - 1]);
        return logits;
    }

    int vocab_size() const { return vocab_size_; }
    int hidden_size() const { return hidden_size_; }
    int num_layers() const { return num_layers_; }

private:
    int vocab_size_;
    int hidden_size_;
    int num_layers_;
    std::shared_ptr<nn::Linear> embedding_;
    std::vector<std::shared_ptr<nn::GRUCell>> gru_layers_;
    std::shared_ptr<nn::Linear> output_proj_;
};

/**
 * @brief Seq2Seq ChatBot model with multi-layer GRU
 */
class ChatBot : public nn::Module {
public:
    ChatBot(int vocab_size, int embed_size, int hidden_size, int num_layers = 2) {
        encoder_ = std::make_shared<Encoder>(vocab_size, embed_size, hidden_size, num_layers);
        decoder_ = std::make_shared<Decoder>(vocab_size, embed_size, hidden_size, num_layers);

        register_module("encoder", encoder_);
        register_module("decoder", decoder_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        return input;
    }

    std::shared_ptr<Encoder> encoder() { return encoder_; }
    std::shared_ptr<Decoder> decoder() { return decoder_; }

private:
    std::shared_ptr<Encoder> encoder_;
    std::shared_ptr<Decoder> decoder_;
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

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);

    initialize();

    showcase::print_header("Chat AI - Neural Network API (High-Level)", device);

    manual_seed(42);

    // Hyperparameters
    int embed_size = 128;
    int hidden_size = 256;
    int num_layers = 2;  // Multi-layer GRU for better learning capacity
    float learning_rate = 0.03f;
    int num_epochs = 500;
    int print_every = 5;

    // Build vocabulary
    CharVocab vocab;
    int vocab_size = vocab.size();

    std::cout << "Vocabulary size: " << vocab_size << " characters\n";

    // Load full training data
    auto data = load_training_data("data/chat_training_large.txt", 950);
    std::cout << "Training pairs: " << data.size() << "\n\n";

    showcase::print_section("Model Architecture");
    std::cout << "ChatBot (nn::Module based) - " << num_layers << "-layer GRU:\n";
    std::cout << "  Encoder:\n";
    std::cout << "    - nn::Linear(" << vocab_size << ", " << embed_size << ") [embedding]\n";
    for (int i = 0; i < num_layers; ++i) {
        int in_size = (i == 0) ? embed_size : hidden_size;
        std::cout << "    - nn::GRUCell(" << in_size << ", " << hidden_size << ") [layer " << i << "]\n";
    }
    std::cout << "  Decoder:\n";
    std::cout << "    - nn::Linear(" << vocab_size << ", " << embed_size << ") [embedding]\n";
    for (int i = 0; i < num_layers; ++i) {
        int in_size = (i == 0) ? embed_size : hidden_size;
        std::cout << "    - nn::GRUCell(" << in_size << ", " << hidden_size << ") [layer " << i << "]\n";
    }
    std::cout << "    - nn::Linear(" << hidden_size << ", " << vocab_size << ") [output]\n";

    // Create model with multi-layer GRU
    auto model = std::make_shared<ChatBot>(vocab_size, embed_size, hidden_size, num_layers);
    model->to(device);

    // Get parameters and create optimizer
    auto params = model->parameters();
    std::cout << "\nTotal parameters: " << params.size() << "\n";

    optim::SGD optimizer(params, learning_rate);

    showcase::print_section("Training");
    std::cout << "Using nn::Module, nn::GRUCell, and optim::SGD\n\n";

    std::random_device rd;
    std::mt19937 gen(rd());

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float total_loss = 0.0f;
        int total_tokens = 0;

        model->train();
        std::shuffle(data.begin(), data.end(), gen);

        for (const auto& [question, answer] : data) {
            optimizer.zero_grad();

            // Encode question - returns hidden states for all layers
            std::vector<int> q_tokens;
            for (char c : question) {
                q_tokens.push_back(vocab.encode(c));
            }
            auto h_states = model->encoder()->encode(q_tokens, device);

            // Decode with teacher forcing
            std::string target = std::string(1, SOS_CHAR) + answer + std::string(1, EOS_CHAR);
            Variable loss_sum(zeros({1}, DType::Float32, device), true);

            for (size_t t = 0; t < target.length() - 1; ++t) {
                int input_token = vocab.encode(target[t]);
                int target_token = vocab.encode(target[t + 1]);

                auto logits = model->decoder()->step(input_token, h_states, device);
                auto log_probs = log_softmax(logits, 1);

                // Target one-hot
                std::vector<float> target_oh(vocab_size, 0.0f);
                target_oh[target_token] = 1.0f;
                auto target_var = Variable(from_data(target_oh.data(), {1, vocab_size}, device), false);

                auto target_log_prob = sum(log_probs * target_var);
                loss_sum = loss_sum - target_log_prob;
                total_tokens++;
            }

            // Backward and optimize
            loss_sum.backward();
            optimizer.step();

            total_loss += loss_sum.tensor().item<float>();
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

    model->eval();

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

        // Encode user input - returns hidden states for all layers
        std::vector<int> tokens;
        for (char c : user_input) {
            tokens.push_back(vocab.encode(c));
        }
        auto h_states = model->encoder()->encode(tokens, device);

        // Decode (greedy)
        std::string response;
        int current_token = vocab.sos_idx();

        for (int t = 0; t < 50; ++t) {
            auto logits = model->decoder()->step(current_token, h_states, device);
            auto logits_cpu = logits.tensor().cpu();
            const float* data_ptr = logits_cpu.data<float>();

            int best_idx = 0;
            float best_val = data_ptr[0];
            for (int i = 1; i < vocab_size; ++i) {
                if (data_ptr[i] > best_val) {
                    best_val = data_ptr[i];
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

    std::cout << "\nChat AI solved using Neural Network API!\n";
    std::cout << "Uses multi-layer stacked nn::GRUCell for deeper learning.\n";

    finalize();
    return 0;
}
