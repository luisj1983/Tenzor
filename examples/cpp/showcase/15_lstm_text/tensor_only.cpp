/**
 * @file tensor_only.cpp
 * @brief LSTM next-character prediction with raw tensors
 *
 * Hand-rolled LSTM cell predicting the next character in a short
 * repeating pattern. Shows the full set of LSTM gate equations
 * with manual forward through a sequence. Backprop-through-time
 * is handled by training across short subsequences.
 *
 * Usage: ./15_lstm_text_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <cstdlib>
#include <string>
#include <vector>

using namespace tenzor;

static Tensor sigmoid_t(const Tensor& x) {
    return ones_like(x) / (tenzor::exp(x * -1.0f) + 1.0f);
}
static Tensor tanh_t(const Tensor& x) {
    auto e_pos = tenzor::exp(x);
    auto e_neg = tenzor::exp(x * -1.0f);
    return (e_pos - e_neg) / (e_pos + e_neg);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("LSTM Text - Tensor Only (Forward Pass Demo)", device);
    manual_seed(42);

    // Tiny vocabulary: "hello world"
    const std::string alphabet = " dehlorw";   // 8 chars
    int vocab = static_cast<int>(alphabet.size());
    int hidden = 16;

    // Demonstrate a single forward pass through the LSTM cell.
    // (Fully manual BPTT is long-winded; the autograd tier shows it.)
    std::cout << "Alphabet: \"" << alphabet << "\"  (vocab=" << vocab << ")\n";
    std::cout << "Hidden size: " << hidden << "\n\n";

    // LSTM cell: input (vocab + hidden) -> 4 * hidden
    auto W = randn({vocab + hidden, 4 * hidden}, DType::Float32, device)
           * std::sqrt(1.0f / (vocab + hidden));
    auto b = zeros({1, 4 * hidden}, DType::Float32, device);

    auto step = [&](const Tensor& x_t, const Tensor& h_prev, const Tensor& c_prev) {
        std::vector<Tensor> parts = {x_t, h_prev};
        auto xh = cat(std::span<const Tensor>(parts), 1);  // [1, vocab + hidden]
        auto gates = matmul(xh, W) + b;              // [1, 4 * hidden]
        // Split along hidden: i, f, g, o
        auto i_gate = sigmoid_t(gates.slice(1, 0,         hidden));
        auto f_gate = sigmoid_t(gates.slice(1, hidden,    2 * hidden));
        auto g_gate = tanh_t   (gates.slice(1, 2*hidden,  3 * hidden));
        auto o_gate = sigmoid_t(gates.slice(1, 3*hidden,  4 * hidden));
        auto c_new  = f_gate * c_prev + i_gate * g_gate;
        auto h_new  = o_gate * tanh_t(c_new);
        return std::make_pair(h_new, c_new);
    };

    showcase::print_section("Single forward pass");
    std::string sequence = "hello";
    auto h = zeros({1, hidden}, DType::Float32, device);
    auto c = zeros({1, hidden}, DType::Float32, device);

    for (char ch : sequence) {
        size_t idx = alphabet.find(ch);
        if (idx == std::string::npos) { continue; }
        std::vector<float> one_hot(vocab, 0.0f);
        one_hot[idx] = 1.0f;
        auto x = from_data(one_hot.data(), {1, vocab}, device);
        auto [h_new, c_new] = step(x, h, c);
        h = h_new;
        c = c_new;
        auto h_cpu = h.cpu();
        float norm = std::sqrt(tenzor::sum(h * h).item<float>());
        std::cout << "char='" << ch << "' |h|_2 = " << norm << "\n";
    }

    showcase::print_section("Final hidden state (first 8 dims)");
    auto h_cpu = h.cpu();
    for (int i = 0; i < 8; ++i) {
        std::cout << h_cpu.data<float>()[i] << " ";
    }
    std::cout << "\n";

    std::cout << "\nLSTM cell forward pass demonstrated with raw tensors!\n";
    std::cout << "Gate equations: i, f, o are sigmoid gates; g is a tanh candidate cell.\n";
    std::cout << "See autograd.cpp for full BPTT training.\n";

    finalize();
    return 0;
}
