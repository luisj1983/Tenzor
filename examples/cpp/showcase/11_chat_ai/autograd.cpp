/**
 * @file autograd.cpp
 * @brief Chat AI using Tenzor's autograd — GRU seq2seq with Bahdanau attention
 *
 * Demonstrates the mid-level autograd API: every weight is an explicit
 * Variable, every operation (matmul, sigmoid, tanh, softmax, cat, transpose)
 * builds the computation graph, and `loss.backward()` fills every .grad() for
 * us. A Bahdanau attention head lets the decoder look over the full encoder
 * state sequence at each step — dramatically better than the old single
 * fixed-size hidden vector.
 *
 * Trained with optim::Adam, gradient clipping, and per-token-normalized
 * cross-entropy. Sampled at inference with temperature + top-k.
 *
 * Usage: ./11_chat_ai_autograd [--backend cpu|cuda|vulkan|rocm|oneapi]
 *                              [--data <path>] [--epochs N]
 */

#include "../common.hpp"

#include <tenzor/nn/utils/clip_grad.hpp>

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iomanip>
#include <random>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tenzor;

// ============================================================================
// Char vocab (PAD, SOS, EOS + printable ASCII)
// ============================================================================

const char PAD_CHAR = '\0';
const char SOS_CHAR = '\x01';
const char EOS_CHAR = '\x02';

struct CharVocab {
    CharVocab() {
        add(PAD_CHAR); add(SOS_CHAR); add(EOS_CHAR);
        for (char c = ' '; c <= '~'; ++c) add(c);
    }
    void add(char c) {
        if (!m_.count(c)) { m_[c] = static_cast<int>(v_.size()); v_.push_back(c); }
    }
    int  encode(char c) const {
        auto it = m_.find(c); return it != m_.end() ? it->second : m_.at(' ');
    }
    char decode(int i) const {
        return (i >= 0 && i < static_cast<int>(v_.size())) ? v_[i] : ' ';
    }
    int size() const { return static_cast<int>(v_.size()); }
    int pad_idx() const { return 0; }
    int sos_idx() const { return 1; }
    int eos_idx() const { return 2; }
    std::unordered_map<char, int> m_;
    std::vector<char>             v_;
};

// ============================================================================
// Data loader — tab-separated question/answer pairs, optional max length filter
// ============================================================================

static std::vector<std::pair<std::string, std::string>>
load_pairs(const std::string& path, size_t max_len = 50) {
    std::vector<std::pair<std::string, std::string>> pairs;
    std::ifstream f(path);
    if (!f.is_open()) {
        return {{"hello", "hi there"}, {"hi", "hello"},
                {"how are you", "i am good"},
                {"whats your name", "i am tenzor"},
                {"thanks", "you are welcome"},
                {"bye", "goodbye"}};
    }
    std::string line;
    while (std::getline(f, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab);
        std::string a = line.substr(tab + 1);
        std::transform(q.begin(), q.end(), q.begin(), ::tolower);
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        if (q.empty() || a.empty())            continue;
        if (q.size() > max_len || a.size() > max_len) continue;
        pairs.emplace_back(std::move(q), std::move(a));
    }
    return pairs;
}

// ============================================================================
// Parameter helpers
// ============================================================================

static std::shared_ptr<Variable> make_param(std::vector<int64_t> shape,
                                            Device device, float scale) {
    return std::make_shared<Variable>(
        randn(shape, DType::Float32, device) * scale, /*requires_grad=*/true);
}
static std::shared_ptr<Variable> make_zero_param(std::vector<int64_t> shape,
                                                 Device device) {
    return std::make_shared<Variable>(
        zeros(shape, DType::Float32, device), /*requires_grad=*/true);
}

// ============================================================================
// GRU cell (pure autograd ops)
// ============================================================================

static Variable gru_step(const Variable& x, const Variable& h_prev,
                         const Variable& Wz, const Variable& Uz, const Variable& bz,
                         const Variable& Wr, const Variable& Ur, const Variable& br,
                         const Variable& Wh, const Variable& Uh, const Variable& bh)
{
    auto z = nn::sigmoid(matmul(x, Wz) + matmul(h_prev, Uz) + bz);
    auto r = nn::sigmoid(matmul(x, Wr) + matmul(h_prev, Ur) + br);
    auto h_tilde = nn::tanh(matmul(x, Wh) + matmul(r * h_prev, Uh) + bh);
    Variable ones(ones_like(z.tensor()), false);
    return (ones - z) * h_prev + z * h_tilde;
}

// ============================================================================
// Bahdanau additive attention.
//
// For efficiency we split the energy computation into two parts: the
// encoder-only term (H_enc @ W_a) is computed ONCE per pair, outside the
// decoder loop, and passed back in as `H_Wa`. Each decoder step only pays for
// `h_dec @ U_a + tanh + score + softmax + weighted sum`.
// ============================================================================

static Variable bahdanau_context(const Variable& h_dec,
                                 const Variable& H_enc,
                                 const Variable& H_Wa,      // precomputed H_enc @ W_a
                                 const Variable& U_a,
                                 const Variable& v_a)
{
    auto e      = nn::tanh(H_Wa + matmul(h_dec, U_a));  // [T, A]
    auto scores = matmul(e, v_a);                        // [T, 1]
    auto alpha  = softmax(scores, 0);                    // [T, 1]
    auto a_T    = transpose(alpha, 0, 1);                // [1, T]
    return matmul(a_T, H_enc);                           // [1, H]
}

// ============================================================================
// One-hot encoding of a single token index as a non-grad Variable [1, V]
// ============================================================================

static Variable one_hot_var(int idx, int vocab_size, Device device) {
    std::vector<float> data(vocab_size, 0.0f);
    data[idx] = 1.0f;
    return Variable(from_data(data.data(), {1, vocab_size}, device),
                    /*requires_grad=*/false);
}

// ============================================================================
// Temperature + top-k sampler over a CPU float buffer
// ============================================================================

static int sample_next(const float* logits, int V,
                       float temperature, int top_k, std::mt19937& rng)
{
    std::vector<std::pair<float, int>> scored(V);
    for (int i = 0; i < V; ++i) scored[i] = {logits[i] / std::max(1e-6f, temperature), i};
    if (top_k > 0 && top_k < V) {
        std::nth_element(scored.begin(), scored.begin() + top_k, scored.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });
        scored.resize(top_k);
    }
    float m = scored[0].first;
    for (auto& s : scored) m = std::max(m, s.first);
    double Z = 0.0;
    for (auto& s : scored) { s.first = std::exp(s.first - m); Z += s.first; }
    for (auto& s : scored) s.first = static_cast<float>(s.first / Z);
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float r = U(rng), cum = 0.0f;
    for (auto& s : scored) { cum += s.first; if (r <= cum) return s.second; }
    return scored.back().second;
}

// ============================================================================
// CLI parsing
// ============================================================================

struct CliOpts {
    std::string data   = "data/chat_training_mixed.txt";
    int         epochs = 12;
};

static CliOpts parse_cli(int argc, char** argv) {
    CliOpts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--data" || a == "-d") && i + 1 < argc) o.data   = argv[++i];
        else if (a == "--epochs" && i + 1 < argc)         o.epochs = std::stoi(argv[++i]);
    }
    return o;
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    auto   cli    = parse_cli(argc, argv);

    initialize();
    manual_seed(42);

    showcase::print_header("Chat AI — Autograd (GRU + Bahdanau Attention)", device);

    // -------- Hyperparameters --------
    const int embed_size  = 64;
    const int hidden_size = 96;
    const int attn_size   = 48;
    const float lr          = 2e-3f;
    const float grad_clip   = 1.0f;
    const int print_every = 2;

    CharVocab vocab;
    const int V = vocab.size();

    auto data = load_pairs(cli.data);
    std::cout << "Training pairs:      " << data.size() << "\n";
    std::cout << "Vocabulary size:     " << V << " chars\n";
    if (data.empty()) { std::cerr << "No training data.\n"; return 1; }

    showcase::print_section("Model Architecture");
    std::cout << "Encoder-Decoder GRU + Bahdanau attention\n"
              << "  embed:   " << embed_size  << "\n"
              << "  hidden:  " << hidden_size << "\n"
              << "  attn:    " << attn_size   << "\n";

    // -------- Parameters --------
    const float init = 0.1f;

    // Shared embedding
    auto embedding = make_param({V, embed_size}, device, init);

    // GRU weights (shared encoder/decoder for simplicity)
    auto Wz = make_param({embed_size,  hidden_size}, device, init);
    auto Uz = make_param({hidden_size, hidden_size}, device, init);
    auto bz = make_zero_param({1, hidden_size}, device);
    auto Wr = make_param({embed_size,  hidden_size}, device, init);
    auto Ur = make_param({hidden_size, hidden_size}, device, init);
    auto br = make_zero_param({1, hidden_size}, device);
    auto Wh = make_param({embed_size,  hidden_size}, device, init);
    auto Uh = make_param({hidden_size, hidden_size}, device, init);
    auto bh = make_zero_param({1, hidden_size}, device);

    // Bahdanau attention weights
    auto W_a = make_param({hidden_size, attn_size}, device, init);
    auto U_a = make_param({hidden_size, attn_size}, device, init);
    auto v_a = make_param({attn_size,   1},         device, init);

    // Output projection over [h; context] -> vocab
    auto out_proj = make_param({2 * hidden_size, V}, device, init);
    auto out_bias = make_zero_param({1, V}, device);

    std::vector<std::shared_ptr<Variable>> params = {
        embedding,
        Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh,
        W_a, U_a, v_a,
        out_proj, out_bias
    };
    std::cout << "Parameter tensors:   " << params.size() << "\n";

    optim::Adam optimizer(params, /*lr=*/lr, 0.9, 0.999, 1e-8, /*weight_decay=*/1e-5);

    showcase::print_section("Training");
    std::cout << "Optimizer: Adam (lr=" << lr << ")\n"
              << "Grad clip: " << grad_clip << "\n"
              << "Epochs:    " << cli.epochs << "\n\n";

    std::random_device rd;
    std::mt19937 gen(rd());
    std::mt19937 sample_rng(42);

    for (int epoch = 0; epoch < cli.epochs; ++epoch) {
        float total_loss = 0.0f;
        int   total_tok  = 0;
        std::shuffle(data.begin(), data.end(), gen);

        for (const auto& [question, answer] : data) {
            optimizer.zero_grad();

            // ----- Encoder: produce hidden state per input char -----
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
                // Encode an empty input with a single SOS step so attention has something.
                auto x = one_hot_var(vocab.sos_idx(), V, device);
                auto e = matmul(x, *embedding);
                h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);
                H_enc.push_back(h);
            }
            auto H_cat = cat(H_enc, 0);               // [T_src, hidden]
            auto H_Wa  = matmul(H_cat, *W_a);         // [T_src, attn] — cached once per pair

            // ----- Decoder: teacher forcing -----
            std::string target = std::string(1, SOS_CHAR) + answer + std::string(1, EOS_CHAR);

            Variable loss(zeros({1}, DType::Float32, device), true);
            int tokens_this_pair = 0;

            for (size_t t = 0; t + 1 < target.size(); ++t) {
                int in_tok  = vocab.encode(target[t]);
                int tgt_tok = vocab.encode(target[t + 1]);

                // GRU step on input token
                auto x = one_hot_var(in_tok, V, device);
                auto e = matmul(x, *embedding);
                h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);

                // Attention over encoder states (Luong-style: after GRU)
                auto context = bahdanau_context(h, H_cat, H_Wa, *U_a, *v_a);  // [1, H]
                auto combined = cat(std::vector<Variable>{h, context}, 1);    // [1, 2H]

                auto logits    = matmul(combined, *out_proj) + *out_bias;     // [1, V]
                auto log_probs = log_softmax(logits, 1);
                auto tgt_one   = one_hot_var(tgt_tok, V, device);
                auto nll       = sum(log_probs * tgt_one);                    // scalar

                loss = loss - nll;
                tokens_this_pair++;
            }

            // Normalize to per-token loss so gradient scale doesn't depend on reply length
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

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            float avg = total_loss / std::max(1, total_tok);
            std::cout << "Epoch [" << std::setw(3) << (epoch + 1) << "/"
                      << cli.epochs << "]  loss=" << std::fixed
                      << std::setprecision(4) << avg
                      << "  ppl=" << std::setprecision(2) << std::exp(avg) << "\n";
        }
    }

    // ------------------------------------------------------------------
    // Interactive chat — top-k sampling
    // ------------------------------------------------------------------
    const float temperature = 0.4f;
    const int   top_k       = 5;
    const int   max_gen     = 50;

    showcase::print_section("Interactive Chat");
    std::cout << "Chat with the trained model (temperature=" << temperature
              << ", top-k=" << top_k << "). Type 'quit' or 'exit' to stop.\n\n";

    std::string user;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, user)) break;
        if (user == "quit" || user == "exit") { std::cout << "Goodbye!\n"; break; }
        if (user.empty()) continue;
        std::transform(user.begin(), user.end(), user.begin(), ::tolower);

        NoGradGuard no_grad;  // inference doesn't need a graph

        // Encode input
        Variable h(zeros({1, hidden_size}, DType::Float32, device), false);
        std::vector<Variable> H_enc;
        for (char c : user) {
            auto x = one_hot_var(vocab.encode(c), V, device);
            auto e = matmul(x, *embedding);
            h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);
            H_enc.push_back(h);
        }
        if (H_enc.empty()) continue;
        auto H_cat = cat(H_enc, 0);
        auto H_Wa  = matmul(H_cat, *W_a);

        // Decode
        std::string reply;
        int cur = vocab.sos_idx();
        for (int t = 0; t < max_gen; ++t) {
            auto x = one_hot_var(cur, V, device);
            auto e = matmul(x, *embedding);
            h = gru_step(e, h, *Wz, *Uz, *bz, *Wr, *Ur, *br, *Wh, *Uh, *bh);
            auto context  = bahdanau_context(h, H_cat, H_Wa, *U_a, *v_a);
            auto combined = cat(std::vector<Variable>{h, context}, 1);
            auto logits   = matmul(combined, *out_proj) + *out_bias;

            auto cpu = logits.tensor().cpu();
            int next = sample_next(cpu.data<float>(), V, temperature, top_k, sample_rng);
            if (next == vocab.eos_idx()) break;
            char c = vocab.decode(next);
            if (c >= ' ' && c <= '~') reply.push_back(c);
            cur = next;
        }

        std::cout << "Bot: " << (reply.empty() ? "(silence)" : reply) << "\n\n";
    }

    std::cout << "\nTrained an encoder-decoder GRU with Bahdanau attention "
                 "using Tenzor's autograd.\n";
    finalize();
    return 0;
}
