/**
 * @file neural_network.cpp
 * @brief LSTM next-character prediction with the nn::LSTM module
 *
 * Uses nn::Embedding + nn::LSTM + nn::Linear to predict the next
 * character in a short repeating sequence. Shows how easy
 * sequence modelling becomes with the high-level API.
 *
 * Usage: ./15_lstm_text_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <string>
#include <vector>

using namespace tenzor;

class CharLSTM : public nn::Module {
public:
    CharLSTM(int64_t vocab, int64_t embed_dim, int64_t hidden)
        : hidden_(hidden) {
        embed = std::make_shared<nn::Embedding>(vocab, embed_dim);
        // batch_first=true: input (B, T, E)
        lstm  = std::make_shared<nn::LSTM>(embed_dim, hidden, 1, true, true);
        head  = std::make_shared<nn::Linear>(hidden, vocab);
        register_module("embed", embed);
        register_module("lstm",  lstm);
        register_module("head",  head);
    }

    // input: (B, T) int64 indices -> logits (B, vocab)
    auto forward_impl(const Variable& input) -> Variable override {
        auto e = embed->forward(input);                 // (B, T, E)
        auto out_state = lstm->forward(e, {Variable{}, Variable{}});
        auto seq_out = out_state.first;                 // (B, T, H)

        auto T = seq_out.shape()[1];
        auto last_step = slice(seq_out, 1, T - 1, T);   // (B, 1, H)
        auto last_h = squeeze(last_step, 1);            // (B, H)
        return head->forward(last_h);                   // (B, vocab)
    }
private:
    std::shared_ptr<nn::Embedding> embed;
    std::shared_ptr<nn::LSTM>      lstm;
    std::shared_ptr<nn::Linear>    head;
    int64_t hidden_;
};

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("LSTM Text - Neural Network API", device);
    manual_seed(42);

    const std::string alphabet = " dehlorw";
    int vocab = static_cast<int>(alphabet.size());
    int embed_dim = 8;
    int hidden = 16;
    int batch_size = 1;

    auto char_idx = [&](char ch) -> int64_t {
        auto p = alphabet.find(ch);
        return p == std::string::npos ? 0 : static_cast<int64_t>(p);
    };

    const std::string input_seq  = "hello worl";
    const char target_char = 'd';
    int T = static_cast<int>(input_seq.size());

    std::vector<int64_t> idx_data(T);
    for (int i = 0; i < T; ++i) idx_data[i] = char_idx(input_seq[i]);
    auto X = from_data(idx_data.data(), {batch_size, T}, device);

    std::vector<int64_t> y_data = {char_idx(target_char)};
    auto y = from_data(y_data.data(), {batch_size}, device);

    auto model = std::make_shared<CharLSTM>(vocab, embed_dim, hidden);
    model->to(device);

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);
    nn::CrossEntropyLoss criterion;

    showcase::print_section("Architecture");
    std::cout << "Embedding(" << vocab << " -> " << embed_dim << ") -> LSTM(" << embed_dim
              << " -> " << hidden << ") -> Linear(" << hidden << " -> " << vocab << ")\n";

    int num_epochs = 300;
    int print_every = 30;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        optimizer.zero_grad();
        Variable x(X, false);
        auto logits = model->forward(x);
        auto loss = criterion(logits, y);  // CrossEntropyLoss target is raw Tensor
        loss.backward();
        optimizer.step();
        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] CrossEntropy: " << loss.tensor().item<float>() << "\n";
        }
    }

    showcase::print_section("Final Results");
    model->eval();
    Variable x_eval(X, false);
    auto logits = model->forward(x_eval).tensor().cpu();
    int best_idx = 0;
    float best = logits.data<float>()[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits.data<float>()[i] > best) { best = logits.data<float>()[i]; best_idx = i; }
    }
    std::cout << "Input: \"" << input_seq << "\"\n";
    std::cout << "Target: '" << target_char << "'\n";
    std::cout << "Predicted: '" << alphabet[best_idx] << "'\n";
    std::cout << "\nLSTM solved using Neural Network API!\n";

    finalize();
    return 0;
}
