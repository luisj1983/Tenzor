/**
 * @file autograd_runner.cpp
 * @brief Implementation of the LSTM-text autograd training loop.
 */

#include "autograd_runner.hpp"

#include "../common.hpp"

#include <cmath>
#include <string>
#include <vector>

namespace tenzor::examples::showcase15 {

namespace {

::tenzor::Variable make_onehot(char ch, const std::string& alphabet,
                                const ::tenzor::Device& device) {
    using namespace ::tenzor;
    std::vector<float> one_hot(alphabet.size(), 0.0f);
    auto idx = alphabet.find(ch);
    if (idx != std::string::npos) one_hot[idx] = 1.0f;
    std::vector<int64_t> shape = {1, static_cast<int64_t>(alphabet.size())};
    return Variable(from_data(one_hot.data(), shape, device), false);
}

}  // namespace

int run_lstm_training(int epochs,
                      double* out_initial,
                      double* out_final,
                      ::tenzor::Device device,
                      bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    const std::string alphabet = " dehlorw";
    int vocab = static_cast<int>(alphabet.size());
    int hidden = 16;

    Variable W(randn({vocab + hidden, 4 * hidden}, DType::Float32, device)
             * std::sqrt(1.0f / (vocab + hidden)), true);
    Variable b(zeros({1, 4 * hidden}, DType::Float32, device), true);
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

    const std::string input_seq = "hello worl";
    const char target_char = 'd';

    float lr = 0.1f;
    int print_every = std::max(1, epochs / 10);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable h(zeros({1, hidden}, DType::Float32, device), false);
        Variable c(zeros({1, hidden}, DType::Float32, device), false);

        for (char ch : input_seq) {
            auto x = make_onehot(ch, alphabet, device);
            auto [h_new, c_new] = step(x, h, c);
            h = h_new; c = c_new;
        }

        auto logits = matmul(h, W_out) + b_out;
        auto log_p  = log_softmax(logits, 1);

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

        double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] nll=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase15
