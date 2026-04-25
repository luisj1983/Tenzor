/**
 * @file autograd.cpp
 * @brief LSTM next-character prediction with autograd
 *
 * LSTM cell built from Variable ops (matmul, sigmoid, tanh). BPTT is
 * automatic - the computation graph unrolled across time steps is
 * what backward() traverses.
 *
 * Task: learn to complete "hello worl" -> 'd'
 *
 * Usage: ./15_lstm_text_autograd --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <string>
#include <vector>

using namespace tenzor;

static Variable make_onehot(char ch, const std::string& alphabet, const Device& device) {
    std::vector<float> one_hot(alphabet.size(), 0.0f);
    auto idx = alphabet.find(ch);
    if (idx != std::string::npos) one_hot[idx] = 1.0f;
    std::vector<int64_t> shape = {1, static_cast<int64_t>(alphabet.size())};
    return Variable(from_data(one_hot.data(), shape, device), false);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("LSTM Text - Autograd", device);
    manual_seed(42);

    const std::string alphabet = " dehlorw";
    int vocab = static_cast<int>(alphabet.size());
    int hidden = 16;

    // LSTM cell parameters
    Variable W(randn({vocab + hidden, 4 * hidden}, DType::Float32, device)
             * std::sqrt(1.0f / (vocab + hidden)), true);
    Variable b(zeros({1, 4 * hidden}, DType::Float32, device), true);

    // Output projection: hidden -> vocab
    Variable W_out(randn({hidden, vocab}, DType::Float32, device) * std::sqrt(1.0f / hidden), true);
    Variable b_out(zeros({1, vocab}, DType::Float32, device), true);

    auto step = [&](const Variable& x_t, const Variable& h_prev, const Variable& c_prev) {
        std::vector<Variable> parts = {x_t, h_prev};
        auto xh = cat(parts, 1);
        auto gates = matmul(xh, W) + b;
        auto i_gate = nn::sigmoid(slice(gates, 1, 0,        hidden));
        auto f_gate = nn::sigmoid(slice(gates, 1, hidden,   2 * hidden));
        auto g_gate = nn::tanh   (slice(gates, 1, 2*hidden, 3 * hidden));
        auto o_gate = nn::sigmoid(slice(gates, 1, 3*hidden, 4 * hidden));
        auto c_new = f_gate * c_prev + i_gate * g_gate;
        auto h_new = o_gate * nn::tanh(c_new);
        return std::make_pair(h_new, c_new);
    };

    const std::string input_seq  = "hello worl";  // predict 'd'
    const char target_char = 'd';

    float lr = 0.1f;
    int num_epochs = 400;
    int print_every = 40;

    showcase::print_section("Training");

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        Variable h(zeros({1, hidden}, DType::Float32, device), false);
        Variable c(zeros({1, hidden}, DType::Float32, device), false);

        for (char ch : input_seq) {
            auto x = make_onehot(ch, alphabet, device);
            auto [h_new, c_new] = step(x, h, c);
            h = h_new; c = c_new;
        }

        // Predict next char from last hidden state
        auto logits = matmul(h, W_out) + b_out;           // [1, vocab]
        auto log_p  = log_softmax(logits, 1);

        // One-hot target
        std::vector<float> tgt(vocab, 0.0f);
        tgt[alphabet.find(target_char)] = 1.0f;
        Variable target(from_data(tgt.data(), {1, vocab}, device), false);

        auto loss = mean(sum(target * log_p, 1)) * (-1.0f);

        W.zero_grad(); b.zero_grad();
        W_out.zero_grad(); b_out.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W     = Variable(W.tensor()     - (*W.grad()     * lr), true);
            b     = Variable(b.tensor()     - (*b.grad()     * lr), true);
            W_out = Variable(W_out.tensor() - (*W_out.grad() * lr), true);
            b_out = Variable(b_out.tensor() - (*b_out.grad() * lr), true);
        }

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << num_epochs
                      << "] NLL: " << loss.tensor().item<float>() << "\n";
        }
    }

    // Inference
    showcase::print_section("Final Results");
    Variable h(zeros({1, hidden}, DType::Float32, device), false);
    Variable c(zeros({1, hidden}, DType::Float32, device), false);
    for (char ch : input_seq) {
        auto x = make_onehot(ch, alphabet, device);
        auto [h_new, c_new] = step(x, h, c);
        h = h_new; c = c_new;
    }
    auto logits = matmul(h, W_out) + b_out;
    auto logits_cpu = logits.tensor().cpu();
    int best_idx = 0;
    float best = logits_cpu.data<float>()[0];
    for (int i = 1; i < vocab; ++i) {
        if (logits_cpu.data<float>()[i] > best) { best = logits_cpu.data<float>()[i]; best_idx = i; }
    }

    std::cout << "Input: \"" << input_seq << "\"\n";
    std::cout << "Target next char: '" << target_char << "'\n";
    std::cout << "Predicted next char: '" << alphabet[best_idx] << "'\n";
    std::cout << "\nLSTM demonstrated with autograd!\n";
    std::cout << "BPTT happens automatically through the unrolled computation graph.\n";

    finalize();
    return 0;
}
