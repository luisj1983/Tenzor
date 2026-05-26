/**
 * @file autograd_runner.cpp
 * @brief Implementation of the chat-AI autograd training loop (extracted
 *        from autograd.cpp so a test can drive the same training body).
 *
 * The runner uses a small fixed pair-corpus (the same fallback the
 * standalone exe uses when no --data file is supplied), runs a handful of
 * epochs, and reports the first-epoch and last-epoch per-token mean loss
 * via the out_initial / out_final pointers.
 */

#include "autograd_runner.hpp"

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/utils/clip_grad.hpp>

#include <algorithm>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor::examples::showcase11 {

namespace {

constexpr char kPadChar = '\0';
constexpr char kSosChar = '\x01';
constexpr char kEosChar = '\x02';

struct CharVocab {
    CharVocab() {
        add(kPadChar); add(kSosChar); add(kEosChar);
        for (char c = ' '; c <= '~'; ++c) add(c);
    }
    void add(char c) {
        if (!m_.count(c)) { m_[c] = static_cast<int>(v_.size()); v_.push_back(c); }
    }
    int encode(char c) const {
        auto it = m_.find(c); return it != m_.end() ? it->second : m_.at(' ');
    }
    int size() const { return static_cast<int>(v_.size()); }
    int sos_idx() const { return 1; }
    int eos_idx() const { return 2; }
    std::unordered_map<char, int> m_;
    std::vector<char>             v_;
};

std::shared_ptr<::tenzor::Variable>
make_param(std::vector<int64_t> shape, ::tenzor::Device device, float scale) {
    using namespace ::tenzor;
    return std::make_shared<Variable>(
        randn(shape, DType::Float32, device) * scale, /*requires_grad=*/true);
}

std::shared_ptr<::tenzor::Variable>
make_zero_param(std::vector<int64_t> shape, ::tenzor::Device device) {
    using namespace ::tenzor;
    return std::make_shared<Variable>(
        zeros(shape, DType::Float32, device), /*requires_grad=*/true);
}

::tenzor::Variable gru_step(const ::tenzor::Variable& x,
                            const ::tenzor::Variable& h_prev,
                            const ::tenzor::Variable& Wz, const ::tenzor::Variable& Uz,
                            const ::tenzor::Variable& bz,
                            const ::tenzor::Variable& Wr, const ::tenzor::Variable& Ur,
                            const ::tenzor::Variable& br,
                            const ::tenzor::Variable& Wh, const ::tenzor::Variable& Uh,
                            const ::tenzor::Variable& bh) {
    using namespace ::tenzor;
    auto z = nn::sigmoid(matmul(x, Wz) + matmul(h_prev, Uz) + bz);
    auto r = nn::sigmoid(matmul(x, Wr) + matmul(h_prev, Ur) + br);
    auto h_tilde = nn::tanh(matmul(x, Wh) + matmul(r * h_prev, Uh) + bh);
    Variable ones(ones_like(z.tensor()), false);
    return (ones - z) * h_prev + z * h_tilde;
}

::tenzor::Variable bahdanau_context(const ::tenzor::Variable& h_dec,
                                    const ::tenzor::Variable& H_enc,
                                    const ::tenzor::Variable& H_Wa,
                                    const ::tenzor::Variable& U_a,
                                    const ::tenzor::Variable& v_a) {
    using namespace ::tenzor;
    auto e      = nn::tanh(H_Wa + matmul(h_dec, U_a));
    auto scores = matmul(e, v_a);
    auto alpha  = softmax(scores, 0);
    auto a_T    = transpose(alpha, 0, 1);
    return matmul(a_T, H_enc);
}

::tenzor::Variable one_hot_var(int idx, int vocab_size, ::tenzor::Device device) {
    using namespace ::tenzor;
    std::vector<float> data(vocab_size, 0.0f);
    data[idx] = 1.0f;
    return Variable(from_data(data.data(), {1, vocab_size}, device),
                    /*requires_grad=*/false);
}

}  // namespace

int run_chat_ai_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         ::tenzor::Device device,
                         bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    // Hyperparameters — kept small so the regression test runs fast.
    constexpr int embed_size  = 32;
    constexpr int hidden_size = 48;
    constexpr int attn_size   = 24;
    constexpr float lr        = 2e-3f;
    constexpr float grad_clip = 1.0f;

    CharVocab vocab;
    const int V = vocab.size();

    // Same hardcoded corpus the standalone exe falls back to when no
    // --data is supplied.
    std::vector<std::pair<std::string, std::string>> data = {
        {"hello", "hi there"}, {"hi", "hello"},
        {"how are you", "i am good"},
        {"whats your name", "i am tenzor"},
        {"thanks", "you are welcome"},
        {"bye", "goodbye"}};

    constexpr float init = 0.1f;
    auto embedding = make_param({V, embed_size}, device, init);

    auto Wz = make_param({embed_size,  hidden_size}, device, init);
    auto Uz = make_param({hidden_size, hidden_size}, device, init);
    auto bz = make_zero_param({1, hidden_size}, device);
    auto Wr = make_param({embed_size,  hidden_size}, device, init);
    auto Ur = make_param({hidden_size, hidden_size}, device, init);
    auto br = make_zero_param({1, hidden_size}, device);
    auto Wh = make_param({embed_size,  hidden_size}, device, init);
    auto Uh = make_param({hidden_size, hidden_size}, device, init);
    auto bh = make_zero_param({1, hidden_size}, device);

    auto W_a = make_param({hidden_size, attn_size}, device, init);
    auto U_a = make_param({hidden_size, attn_size}, device, init);
    auto v_a = make_param({attn_size,   1},         device, init);

    auto out_proj = make_param({2 * hidden_size, V}, device, init);
    auto out_bias = make_zero_param({1, V}, device);

    std::vector<std::shared_ptr<Variable>> params = {
        embedding,
        Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh,
        W_a, U_a, v_a,
        out_proj, out_bias};

    optim::Adam optimizer(params, lr, 0.9, 0.999, 1e-8, /*weight_decay=*/1e-5);

    std::mt19937 gen(42);

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        float total_loss = 0.0f;
        int   total_tok  = 0;
        std::shuffle(data.begin(), data.end(), gen);

        for (const auto& [question, answer] : data) {
            optimizer.zero_grad();

            Variable h(zeros({1, hidden_size}, DType::Float32, device), false);
            std::vector<Variable> H_enc;
            H_enc.reserve(question.size() + 1);
            for (char c : question) {
                auto x = one_hot_var(vocab.encode(c), V, device);
                auto e = matmul(x, *embedding);
                h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);
                H_enc.push_back(h);
            }
            if (H_enc.empty()) {
                auto x = one_hot_var(vocab.sos_idx(), V, device);
                auto e = matmul(x, *embedding);
                h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);
                H_enc.push_back(h);
            }
            auto H_cat = cat(H_enc, 0);
            auto H_Wa  = matmul(H_cat, *W_a);

            std::string target = std::string(1, kSosChar) + answer +
                                 std::string(1, kEosChar);

            Variable loss(zeros({1}, DType::Float32, device), true);
            int tokens_this_pair = 0;

            for (size_t t = 0; t + 1 < target.size(); ++t) {
                int in_tok  = vocab.encode(target[t]);
                int tgt_tok = vocab.encode(target[t + 1]);

                auto x = one_hot_var(in_tok, V, device);
                auto e = matmul(x, *embedding);
                h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);

                auto context  = bahdanau_context(h, H_cat, H_Wa, *U_a, *v_a);
                auto combined = cat(std::vector<Variable>{h, context}, 1);
                auto logits   = matmul(combined, *out_proj) + *out_bias;
                auto log_probs = log_softmax(logits, 1);
                auto tgt_one   = one_hot_var(tgt_tok, V, device);
                auto nll       = sum(log_probs * tgt_one);

                loss = loss - nll;
                tokens_this_pair++;
            }

            if (tokens_this_pair > 0) {
                Variable inv_tok(
                    full({1}, 1.0f / static_cast<float>(tokens_this_pair),
                         DType::Float32, device),
                    false);
                loss = loss * inv_tok;
            }

            loss.backward();
            nn::utils::clip_grad_norm_(params, grad_clip);
            optimizer.step();

            total_loss += loss.tensor().item<float>() * tokens_this_pair;
            total_tok  += tokens_this_pair;
        }

        double avg = total_loss / std::max(1, total_tok);
        if (epoch == 0 && out_initial) *out_initial = avg;
        final_loss = avg;

        if (verbose) {
            std::cout << "Epoch [" << std::setw(3) << (epoch + 1) << "/"
                      << epochs << "]  loss=" << std::fixed
                      << std::setprecision(4) << avg
                      << "  ppl=" << std::setprecision(2) << std::exp(avg) << "\n";
        }
    }

    if (out_final) *out_final = final_loss;
    return 0;
}

}  // namespace tenzor::examples::showcase11
