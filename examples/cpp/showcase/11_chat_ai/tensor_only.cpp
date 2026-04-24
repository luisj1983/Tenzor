/**
 * @file tensor_only.cpp
 * @brief Chat AI using only raw Tensors — every gradient computed by hand
 *
 * The purpose of this file is to show how much work autograd is doing for you.
 * There is no Variable, no nn::Module, no loss.backward(). We write the GRU
 * forward pass, the GRU backward pass, and every other gradient by hand,
 * accumulate them through Backprop-Through-Time, apply a global-norm
 * gradient clip, and update the weights with a hand-rolled Adam optimizer.
 *
 * Compared to the previous version this one:
 *   - runs FULL BPTT through both the encoder and the decoder,
 *   - normalizes the loss per token so gradient magnitude doesn't scale with
 *     reply length,
 *   - replaces SGD with a manual Adam implementation,
 *   - replaces value-clipping with proper global-norm gradient clipping,
 *   - samples with temperature + top-k at inference instead of argmax.
 *
 * Usage: ./11_chat_ai_tensor_only [--backend cpu|cuda|vulkan|rocm|oneapi]
 *                                 [--data <path>] [--epochs N]
 */

#include "../common.hpp"

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
// Elementwise helpers built from public tensor ops
// ============================================================================

static Tensor sigmoid_t(const Tensor& x)        { return ones_like(x) / (ones_like(x) + tenzor::exp(zeros_like(x) - x)); }
static Tensor sigmoid_deriv(const Tensor& y)    { return y * (ones_like(y) - y); }
static Tensor tanh_t(const Tensor& x)           { auto ex = tenzor::exp(x); auto en = tenzor::exp(zeros_like(x) - x); return (ex - en) / (ex + en); }
static Tensor tanh_deriv(const Tensor& y)       { return ones_like(y) - y * y; }

static Tensor softmax_t(const Tensor& x, int dim = 1) {
    auto m       = tenzor::max(x, dim, true);
    auto shifted = x - m;
    auto e       = tenzor::exp(shifted);
    auto s       = tenzor::sum(e, dim, true);
    return e / s;
}

// ============================================================================
// Char vocab
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
// Data loader
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
        if (q.empty() || a.empty())             continue;
        if (q.size() > max_len || a.size() > max_len) continue;
        pairs.emplace_back(std::move(q), std::move(a));
    }
    return pairs;
}

// ============================================================================
// GRU forward/backward — saves intermediates needed for BPTT
// ============================================================================

struct GRUState {
    Tensor x, h_prev, z, r, h_tilde, h;
};

static GRUState gru_forward(
    const Tensor& x, const Tensor& h_prev,
    const Tensor& Wz, const Tensor& Uz, const Tensor& bz,
    const Tensor& Wr, const Tensor& Ur, const Tensor& br,
    const Tensor& Wh, const Tensor& Uh, const Tensor& bh)
{
    GRUState s;
    s.x      = x;
    s.h_prev = h_prev;
    s.z      = sigmoid_t(matmul(x, Wz) + matmul(h_prev, Uz) + bz);
    s.r      = sigmoid_t(matmul(x, Wr) + matmul(h_prev, Ur) + br);
    s.h_tilde = tanh_t (matmul(x, Wh) + matmul(s.r * h_prev, Uh) + bh);
    s.h       = (ones_like(s.z) - s.z) * h_prev + s.z * s.h_tilde;
    return s;
}

static void gru_backward(
    const GRUState& s, const Tensor& dL_dh,
    const Tensor& Wz, const Tensor& Uz,
    const Tensor& Wr, const Tensor& Ur,
    const Tensor& Wh, const Tensor& Uh,
    Tensor& dL_dWz, Tensor& dL_dUz, Tensor& dL_dbz,
    Tensor& dL_dWr, Tensor& dL_dUr, Tensor& dL_dbr,
    Tensor& dL_dWh, Tensor& dL_dUh, Tensor& dL_dbh,
    Tensor& dL_dh_prev, Tensor& dL_dx)
{
    // h = (1-z) h_prev + z h_tilde
    auto dL_dz             = dL_dh * (s.h_tilde - s.h_prev);
    auto dL_dh_tilde       = dL_dh * s.z;
    auto dL_dh_prev_direct = dL_dh * (ones_like(s.z) - s.z);

    // h_tilde = tanh(Wh x + Uh (r*h_prev) + bh)
    auto dL_dpre_ht = dL_dh_tilde * tanh_deriv(s.h_tilde);
    dL_dWh = dL_dWh + matmul(s.x.transpose(0, 1), dL_dpre_ht);
    auto r_hp = s.r * s.h_prev;
    dL_dUh = dL_dUh + matmul(r_hp.transpose(0, 1), dL_dpre_ht);
    dL_dbh = dL_dbh + tenzor::sum(dL_dpre_ht, 0, true);

    auto dL_d_rhp = matmul(dL_dpre_ht, Uh.transpose(0, 1));
    auto dL_dr            = dL_d_rhp * s.h_prev;
    auto dL_dh_prev_fromT = dL_d_rhp * s.r;

    // z = sigmoid(Wz x + Uz h_prev + bz)
    auto dL_dpre_z = dL_dz * sigmoid_deriv(s.z);
    dL_dWz = dL_dWz + matmul(s.x.transpose(0, 1),      dL_dpre_z);
    dL_dUz = dL_dUz + matmul(s.h_prev.transpose(0, 1), dL_dpre_z);
    dL_dbz = dL_dbz + tenzor::sum(dL_dpre_z, 0, true);
    auto dL_dh_prev_fromZ = matmul(dL_dpre_z, Uz.transpose(0, 1));

    // r = sigmoid(Wr x + Ur h_prev + br)
    auto dL_dpre_r = dL_dr * sigmoid_deriv(s.r);
    dL_dWr = dL_dWr + matmul(s.x.transpose(0, 1),      dL_dpre_r);
    dL_dUr = dL_dUr + matmul(s.h_prev.transpose(0, 1), dL_dpre_r);
    dL_dbr = dL_dbr + tenzor::sum(dL_dpre_r, 0, true);
    auto dL_dh_prev_fromR = matmul(dL_dpre_r, Ur.transpose(0, 1));

    dL_dh_prev = dL_dh_prev_direct + dL_dh_prev_fromT
               + dL_dh_prev_fromZ  + dL_dh_prev_fromR;

    dL_dx = matmul(dL_dpre_z,  Wz.transpose(0, 1))
          + matmul(dL_dpre_r,  Wr.transpose(0, 1))
          + matmul(dL_dpre_ht, Wh.transpose(0, 1));
}

// ============================================================================
// Hand-rolled Adam — m, v buffers per parameter, bias-corrected update
// ============================================================================

struct AdamState {
    Tensor m, v;
};

class AdamManual {
public:
    AdamManual(std::vector<Tensor*> params, float lr = 1e-3f,
               float beta1 = 0.9f, float beta2 = 0.999f, float eps = 1e-8f)
        : params_(std::move(params)), lr_(lr), b1_(beta1), b2_(beta2), eps_(eps)
    {
        state_.reserve(params_.size());
        for (auto* p : params_) state_.push_back({zeros_like(*p), zeros_like(*p)});
    }

    // Global gradient norm across all grads. No-op clipping if norm ≤ max_norm.
    void clip_grads_global_norm(std::vector<Tensor>& grads, float max_norm) const {
        double sum_sq = 0.0;
        for (auto& g : grads) {
            auto s = tenzor::sum(g * g).cpu();
            sum_sq += static_cast<double>(s.data<float>()[0]);
        }
        float total_norm = std::sqrt(static_cast<float>(sum_sq));
        if (total_norm > max_norm && total_norm > 0.0f) {
            float scale = max_norm / (total_norm + 1e-6f);
            for (auto& g : grads) g = g * scale;
        }
    }

    void step(std::vector<Tensor>& grads) {
        step_++;
        float b1t = 1.0f - std::pow(b1_, step_);
        float b2t = 1.0f - std::pow(b2_, step_);
        for (size_t i = 0; i < params_.size(); ++i) {
            auto& g = grads[i];
            state_[i].m = state_[i].m * b1_ + g * (1.0f - b1_);
            state_[i].v = state_[i].v * b2_ + (g * g) * (1.0f - b2_);
            auto m_hat = state_[i].m * (1.0f / b1t);
            auto v_hat = state_[i].v * (1.0f / b2t);
            auto denom = tenzor::sqrt(v_hat) + eps_;
            *params_[i] = *params_[i] - (m_hat / denom) * lr_;
        }
    }

private:
    std::vector<Tensor*> params_;
    float lr_, b1_, b2_, eps_;
    int   step_ = 0;
    std::vector<AdamState> state_;
};

// ============================================================================
// Temperature + top-k sampler
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

    showcase::print_header("Chat AI — Raw Tensors (manual BPTT + hand-rolled Adam)", device);

    // Hyperparameters
    const int  embed_size   = 48;
    const int  hidden_size  = 96;
    const float lr          = 2e-3f;
    const float grad_clip   = 1.0f;
    const int  print_every  = 2;

    CharVocab vocab;
    const int V = vocab.size();
    auto data   = load_pairs(cli.data);
    std::cout << "Training pairs:      " << data.size() << "\n";
    std::cout << "Vocabulary size:     " << V << " chars\n";
    if (data.empty()) { std::cerr << "No training data.\n"; return 1; }

    showcase::print_section("Model Architecture");
    std::cout << "Encoder-Decoder GRU (manual BPTT, no autograd)\n"
              << "  embed:  " << embed_size << "\n"
              << "  hidden: " << hidden_size << "\n";

    // Parameters
    const float init = 0.1f;
    auto embedding  = randn({V,           embed_size},  DType::Float32, device) * init;
    auto Wz         = randn({embed_size,  hidden_size}, DType::Float32, device) * init;
    auto Uz         = randn({hidden_size, hidden_size}, DType::Float32, device) * init;
    auto bz         = zeros({1, hidden_size},           DType::Float32, device);
    auto Wr         = randn({embed_size,  hidden_size}, DType::Float32, device) * init;
    auto Ur         = randn({hidden_size, hidden_size}, DType::Float32, device) * init;
    auto br         = zeros({1, hidden_size},           DType::Float32, device);
    auto Wh         = randn({embed_size,  hidden_size}, DType::Float32, device) * init;
    auto Uh         = randn({hidden_size, hidden_size}, DType::Float32, device) * init;
    auto bh         = zeros({1, hidden_size},           DType::Float32, device);
    auto output_proj = randn({hidden_size, V},          DType::Float32, device) * init;
    auto output_bias = zeros({1, V},                    DType::Float32, device);

    std::vector<Tensor*> params = {
        &embedding, &Wz, &Uz, &bz, &Wr, &Ur, &br,
        &Wh, &Uh, &bh, &output_proj, &output_bias
    };
    AdamManual optimizer(params, lr);

    showcase::print_section("Training");
    std::cout << "Optimizer: Hand-rolled Adam (lr=" << lr << ")\n"
              << "Grad clip: global-norm " << grad_clip << "\n"
              << "Epochs:    " << cli.epochs << "\n\n";

    std::mt19937 gen(std::random_device{}());
    std::mt19937 sample_rng(42);

    for (int epoch = 0; epoch < cli.epochs; ++epoch) {
        float total_loss = 0.0f;
        int   total_tok  = 0;
        std::shuffle(data.begin(), data.end(), gen);

        for (const auto& [question, answer] : data) {
            // Per-pair gradient accumulators
            auto dL_dWz  = zeros_like(Wz);
            auto dL_dUz  = zeros_like(Uz);
            auto dL_dbz  = zeros_like(bz);
            auto dL_dWr  = zeros_like(Wr);
            auto dL_dUr  = zeros_like(Ur);
            auto dL_dbr  = zeros_like(br);
            auto dL_dWh  = zeros_like(Wh);
            auto dL_dUh  = zeros_like(Uh);
            auto dL_dbh  = zeros_like(bh);
            auto dL_dOP  = zeros_like(output_proj);
            auto dL_dOB  = zeros_like(output_bias);
            auto dL_dE   = zeros_like(embedding);

            // ---------------- Encoder forward ----------------
            std::vector<GRUState> enc;
            auto h = zeros({1, hidden_size}, DType::Float32, device);
            for (char c : question) {
                int tok = vocab.encode(c);
                std::vector<float> oh(V, 0.0f); oh[tok] = 1.0f;
                auto x = from_data(oh.data(), {1, V}, device);
                auto e = matmul(x, embedding);
                auto s = gru_forward(e, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
                enc.push_back(s);
                h = s.h;
            }

            // ---------------- Decoder forward (teacher forcing) ----------------
            std::vector<GRUState> dec;
            std::vector<Tensor>   logits_all;
            std::vector<int>      tgt_tokens;
            std::string target = std::string(1, SOS_CHAR) + answer + std::string(1, EOS_CHAR);
            for (size_t t = 0; t + 1 < target.size(); ++t) {
                int in_tok  = vocab.encode(target[t]);
                int tgt_tok = vocab.encode(target[t + 1]);
                tgt_tokens.push_back(tgt_tok);

                std::vector<float> oh(V, 0.0f); oh[in_tok] = 1.0f;
                auto x = from_data(oh.data(), {1, V}, device);
                auto e = matmul(x, embedding);
                auto s = gru_forward(e, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh);
                dec.push_back(s);
                h = s.h;
                logits_all.push_back(matmul(h, output_proj) + output_bias);
            }

            // ---------------- Loss + dL/dlogits (softmax cross-entropy) ----------------
            float seq_loss = 0.0f;
            std::vector<Tensor> dL_dlogits;
            dL_dlogits.reserve(dec.size());
            const int T_tok = static_cast<int>(dec.size());
            for (int t = 0; t < T_tok; ++t) {
                auto probs = softmax_t(logits_all[t], 1);

                // Per-token cross-entropy (for reporting and gradient scale)
                auto probs_cpu = probs.cpu();
                const float* p = probs_cpu.data<float>();
                float tp = p[tgt_tokens[t]];
                seq_loss -= std::log(std::max(tp, 1e-10f));

                std::vector<float> oh(V, 0.0f); oh[tgt_tokens[t]] = 1.0f;
                auto target_t = from_data(oh.data(), {1, V}, device);
                dL_dlogits.push_back(probs - target_t);
            }
            total_loss += seq_loss;
            total_tok  += T_tok;

            // Per-token normalization for gradient scale — we scale each
            // dL/dlogits by 1/T_tok before accumulating upstream gradients.
            const float norm = 1.0f / std::max(1, T_tok);

            // ---------------- Decoder backward (full BPTT) ----------------
            auto dL_dh = zeros({1, hidden_size}, DType::Float32, device);
            for (int t = T_tok - 1; t >= 0; --t) {
                auto dL_dlog = dL_dlogits[t] * norm;

                // Through out_proj / out_bias
                dL_dOP = dL_dOP + matmul(dec[t].h.transpose(0, 1), dL_dlog);
                dL_dOB = dL_dOB + tenzor::sum(dL_dlog, 0, true);
                auto dL_dh_out = matmul(dL_dlog, output_proj.transpose(0, 1));

                auto dL_dh_total = dL_dh + dL_dh_out;

                Tensor dL_dh_prev, dL_dx;
                gru_backward(dec[t], dL_dh_total,
                             Wz, Uz, Wr, Ur, Wh, Uh,
                             dL_dWz, dL_dUz, dL_dbz,
                             dL_dWr, dL_dUr, dL_dbr,
                             dL_dWh, dL_dUh, dL_dbh,
                             dL_dh_prev, dL_dx);

                // dL/dE for this step's one-hot input
                int in_tok = vocab.encode(target[t]);
                std::vector<float> oh(V, 0.0f); oh[in_tok] = 1.0f;
                auto in_oh = from_data(oh.data(), {1, V}, device);
                dL_dE = dL_dE + matmul(in_oh.transpose(0, 1), dL_dx);

                dL_dh = dL_dh_prev;
            }

            // ---------------- Encoder backward (FULL BPTT — no truncation) ----------------
            for (int t = static_cast<int>(enc.size()) - 1; t >= 0; --t) {
                Tensor dL_dh_prev, dL_dx;
                gru_backward(enc[t], dL_dh,
                             Wz, Uz, Wr, Ur, Wh, Uh,
                             dL_dWz, dL_dUz, dL_dbz,
                             dL_dWr, dL_dUr, dL_dbr,
                             dL_dWh, dL_dUh, dL_dbh,
                             dL_dh_prev, dL_dx);
                int tok = vocab.encode(question[t]);
                std::vector<float> oh(V, 0.0f); oh[tok] = 1.0f;
                auto in_oh = from_data(oh.data(), {1, V}, device);
                dL_dE = dL_dE + matmul(in_oh.transpose(0, 1), dL_dx);
                dL_dh = dL_dh_prev;
            }

            // ---------------- Global-norm clip + Adam update ----------------
            std::vector<Tensor> grads = {
                dL_dE,  dL_dWz, dL_dUz, dL_dbz,
                dL_dWr, dL_dUr, dL_dbr,
                dL_dWh, dL_dUh, dL_dbh,
                dL_dOP, dL_dOB
            };
            optimizer.clip_grads_global_norm(grads, grad_clip);
            optimizer.step(grads);
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
    // Interactive chat — temperature + top-k sampling
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

        // Encode
        auto h = zeros({1, hidden_size}, DType::Float32, device);
        for (char c : user) {
            int tok = vocab.encode(c);
            std::vector<float> oh(V, 0.0f); oh[tok] = 1.0f;
            auto x = from_data(oh.data(), {1, V}, device);
            auto e = matmul(x, embedding);
            h = gru_forward(e, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh).h;
        }

        // Decode
        std::string reply;
        int cur = vocab.sos_idx();
        for (int t = 0; t < max_gen; ++t) {
            std::vector<float> oh(V, 0.0f); oh[cur] = 1.0f;
            auto x = from_data(oh.data(), {1, V}, device);
            auto e = matmul(x, embedding);
            h = gru_forward(e, h, Wz, Uz, bz, Wr, Ur, br, Wh, Uh, bh).h;
            auto logits = matmul(h, output_proj) + output_bias;
            auto cpu    = logits.cpu();
            int next    = sample_next(cpu.data<float>(), V, temperature, top_k, sample_rng);
            if (next == vocab.eos_idx()) break;
            char c = vocab.decode(next);
            if (c >= ' ' && c <= '~') reply.push_back(c);
            cur = next;
        }
        std::cout << "Bot: " << (reply.empty() ? "(silence)" : reply) << "\n\n";
    }

    std::cout << "\nTrained a seq2seq GRU by computing every gradient by hand "
                 "and running hand-rolled Adam.\n";
    finalize();
    return 0;
}
