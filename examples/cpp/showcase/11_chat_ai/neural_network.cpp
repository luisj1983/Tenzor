/**
 * @file neural_network.cpp
 * @brief Chat AI using Tenzor's high-level nn API — tiny decoder-only Transformer
 *
 * A minimal GPT-style causal language model trained on char-level conversation
 * data. Demonstrates nn::Embedding, nn::MultiheadAttention (causal),
 * nn::LayerNorm, nn::Linear, nn::GELU, nn::Dropout, optim::Adam,
 * nn::CrossEntropyLoss (with label smoothing) and gradient clipping.
 *
 * The model is trained next-token-style over random windows of a concatenated
 * corpus, then sampled with temperature + top-k for interactive chat.
 *
 * Usage: ./11_chat_ai_neural_network [--backend cpu|cuda|vulkan|rocm|oneapi]
 *                                    [--data <path>] [--steps N]
 */

#include "../common.hpp"

#include <tenzor/nn/utils/clip_grad.hpp>

#include <algorithm>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Char vocabulary (printable ASCII + a couple of control tokens)
// ============================================================================

struct CharVocab {
    // Token 0 = PAD, 1 = BOS (conversation start). The remaining printable
    // ASCII range fills the rest. Newline '\n' is part of printable range we
    // include so the model can learn turn boundaries.
    CharVocab() {
        idx_to_char_.push_back('\0');  // 0 = PAD
        idx_to_char_.push_back('\x01'); // 1 = BOS
        char_to_idx_['\0'] = 0;
        char_to_idx_['\x01'] = 1;
        for (int c = 0x20; c <= 0x7E; ++c) add(static_cast<char>(c));
        add('\n');
    }

    void add(char c) {
        if (!char_to_idx_.count(c)) {
            char_to_idx_[c] = static_cast<int64_t>(idx_to_char_.size());
            idx_to_char_.push_back(c);
        }
    }

    int64_t encode(char c) const {
        auto it = char_to_idx_.find(c);
        return it != char_to_idx_.end() ? it->second : char_to_idx_.at(' ');
    }

    char decode(int64_t i) const {
        return (i >= 0 && i < static_cast<int64_t>(idx_to_char_.size()))
               ? idx_to_char_[i] : ' ';
    }

    int64_t size()    const { return static_cast<int64_t>(idx_to_char_.size()); }
    int64_t pad_idx() const { return 0; }
    int64_t bos_idx() const { return 1; }
    int64_t nl_idx()  const { return char_to_idx_.at('\n'); }

    std::unordered_map<char, int64_t> char_to_idx_;
    std::vector<char>                 idx_to_char_;
};

// ============================================================================
// Data loading — tab-separated question/answer pairs
// ============================================================================

static std::vector<std::pair<std::string, std::string>>
load_pairs(const std::string& path) {
    std::vector<std::pair<std::string, std::string>> pairs;
    std::ifstream f(path);
    if (!f.is_open()) {
        // Built-in fallback — tiny but well-structured
        pairs = {
            {"hello", "hi there"}, {"hi", "hello"},
            {"how are you", "i am good thanks"},
            {"whats your name", "i am tenzor"},
            {"thanks", "you are welcome"},
            {"bye", "goodbye"}, {"goodbye", "see you later"},
        };
        return pairs;
    }
    std::string line;
    while (std::getline(f, line)) {
        auto tab = line.find('\t');
        if (tab == std::string::npos) continue;
        std::string q = line.substr(0, tab);
        std::string a = line.substr(tab + 1);
        std::transform(q.begin(), q.end(), q.begin(), ::tolower);
        std::transform(a.begin(), a.end(), a.begin(), ::tolower);
        if (q.empty() || a.empty()) continue;
        pairs.emplace_back(std::move(q), std::move(a));
    }
    return pairs;
}

// Encode one pair into a single fixed-length int64 sequence:
//   "u: <q>\nb: <a>\n" followed by PAD tokens to fill `seq_len`.
// The returned sequence is `seq_len` tokens; caller builds `input` = seq[0:T-1]
// and `target` = seq[1:T] for next-token training. If a pair is too long we
// truncate the answer.
static std::vector<int64_t> encode_pair(const std::string& q, const std::string& a,
                                        const CharVocab& v, int64_t seq_len) {
    std::string s = "u: " + q + "\nb: " + a + "\n";
    if (static_cast<int64_t>(s.size()) > seq_len) s.resize(seq_len);
    std::vector<int64_t> out(seq_len, v.pad_idx());
    for (size_t i = 0; i < s.size(); ++i) out[i] = v.encode(s[i]);
    return out;
}

// ============================================================================
// Model — tiny decoder-only Transformer
// ============================================================================

class GPTBlock : public Module {
public:
    GPTBlock(int64_t d_model, int64_t nhead, double dropout)
        : d_model_(d_model)
    {
        // batch_first=true, is_causal=false (mask is passed explicitly every step)
        attn_ = std::make_shared<MultiheadAttention>(
            d_model, nhead,
            /*dropout=*/dropout,
            /*bias=*/true,
            /*add_bias_kv=*/false,
            /*add_zero_attn=*/false,
            /*kdim=*/0, /*vdim=*/0,
            /*batch_first=*/true,
            /*is_causal=*/false);
        ln1_     = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});
        ln2_     = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});
        fc1_     = std::make_shared<Linear>(d_model, 4 * d_model);
        fc2_     = std::make_shared<Linear>(4 * d_model, d_model);
        act_     = std::make_shared<GELU>();
        dropout_ = std::make_shared<Dropout>(static_cast<float>(dropout));

        register_module("attn",    attn_);
        register_module("ln1",     ln1_);
        register_module("ln2",     ln2_);
        register_module("fc1",     fc1_);
        register_module("fc2",     fc2_);
        register_module("gelu",    act_);
        register_module("dropout", dropout_);
    }

    // Pre-norm: x + drop(attn(ln1(x))) ; x + drop(fc2(gelu(fc1(ln2(x)))))
    Variable forward(const Variable& x, const Tensor& causal_mask) {
        auto h1 = ln1_->forward(x);
        auto [att, _w] = attn_->forward(h1, h1, h1,
                                        /*key_padding_mask=*/Tensor{},
                                        /*attn_mask=*/causal_mask,
                                        /*need_weights=*/false);
        auto r1 = x + dropout_->forward(att);
        auto h2 = ln2_->forward(r1);
        auto ff = fc2_->forward(act_->forward(fc1_->forward(h2)));
        return r1 + dropout_->forward(ff);
    }

    Variable forward_impl(const Variable& x) override {
        return forward(x, Tensor{});
    }

private:
    int64_t d_model_;
    std::shared_ptr<MultiheadAttention> attn_;
    std::shared_ptr<LayerNorm> ln1_, ln2_;
    std::shared_ptr<Linear>    fc1_, fc2_;
    std::shared_ptr<GELU>      act_;
    std::shared_ptr<Dropout>   dropout_;
};

class ChatGPT : public Module {
public:
    ChatGPT(int64_t vocab, int64_t d_model, int64_t nhead,
            int64_t n_layer, int64_t max_seq_len, double dropout)
        : vocab_(vocab), d_model_(d_model), max_seq_len_(max_seq_len)
    {
        tok_emb_ = std::make_shared<Embedding>(vocab,        d_model);
        pos_emb_ = std::make_shared<Embedding>(max_seq_len,  d_model);
        drop_    = std::make_shared<Dropout>(static_cast<float>(dropout));
        ln_f_    = std::make_shared<LayerNorm>(std::vector<int64_t>{d_model});
        head_    = std::make_shared<Linear>(d_model, vocab);

        register_module("tok_emb", tok_emb_);
        register_module("pos_emb", pos_emb_);
        register_module("drop",    drop_);
        for (int64_t i = 0; i < n_layer; ++i) {
            auto b = std::make_shared<GPTBlock>(d_model, nhead, dropout);
            blocks_.push_back(b);
            register_module("block_" + std::to_string(i), b);
        }
        register_module("ln_f", ln_f_);
        register_module("head", head_);
    }

    // input_ids: int64 Variable of shape [B, T]. Returns logits [B, T, V].
    Variable forward(const Variable& input_ids) {
        auto shape = input_ids.shape();
        int64_t T  = shape.back();
        Device  dev = input_ids.tensor().device();

        // Token + positional embeddings
        std::vector<int64_t> pos(T);
        for (int64_t i = 0; i < T; ++i) pos[i] = i;
        auto pos_t = from_data(pos.data(), {1, T}, dev);
        Variable pos_v(pos_t, false);

        auto x = tok_emb_->forward(input_ids) + pos_emb_->forward(pos_v);
        x = drop_->forward(x);

        // One causal mask per forward, reused by all blocks
        auto mask = create_causal_mask(T, dev, DType::Float32);
        for (auto& b : blocks_) x = b->forward(x, mask);

        return head_->forward(ln_f_->forward(x));
    }

    Variable forward_impl(const Variable& input_ids) override {
        return forward(input_ids);
    }

    int64_t vocab()       const { return vocab_; }
    int64_t max_seq_len() const { return max_seq_len_; }

private:
    int64_t vocab_;
    int64_t d_model_;
    int64_t max_seq_len_;
    std::shared_ptr<Embedding> tok_emb_, pos_emb_;
    std::shared_ptr<Dropout>   drop_;
    std::vector<std::shared_ptr<GPTBlock>> blocks_;
    std::shared_ptr<LayerNorm> ln_f_;
    std::shared_ptr<Linear>    head_;
};

// ============================================================================
// Sampling helpers — temperature + top-k
// ============================================================================

static int64_t sample_next(const float* logits, int64_t V,
                           float temperature, int top_k,
                           std::mt19937& rng) {
    // Copy into work buffer
    std::vector<std::pair<float, int64_t>> scored(V);
    for (int64_t i = 0; i < V; ++i) scored[i] = {logits[i] / std::max(1e-6f, temperature), i};

    // Top-k filter
    if (top_k > 0 && top_k < V) {
        std::nth_element(scored.begin(), scored.begin() + top_k, scored.end(),
                         [](auto& a, auto& b) { return a.first > b.first; });
        scored.resize(top_k);
    }

    // Softmax over the kept set (numerically stable)
    float m = scored[0].first;
    for (auto& s : scored) m = std::max(m, s.first);
    double Z = 0.0;
    for (auto& s : scored) { s.first = std::exp(s.first - m); Z += s.first; }
    for (auto& s : scored) s.first = static_cast<float>(s.first / Z);

    // Multinomial draw
    std::uniform_real_distribution<float> U(0.0f, 1.0f);
    float r = U(rng);
    float cum = 0.0f;
    for (auto& s : scored) {
        cum += s.first;
        if (r <= cum) return s.second;
    }
    return scored.back().second;
}

// Extract last-timestep logits from [1, T, V] into a CPU buffer.
static std::vector<float> last_token_logits(const Tensor& logits) {
    auto cpu = logits.cpu();
    auto sh  = cpu.shape();          // [1, T, V]
    int64_t T = sh[1], V = sh[2];
    const float* p = cpu.data<float>();
    std::vector<float> out(V);
    std::copy(p + (T - 1) * V, p + T * V, out.begin());
    return out;
}

// ============================================================================
// Training helpers
// ============================================================================

struct TrainConfig {
    int64_t d_model     = 128;
    int64_t nhead       = 4;
    int64_t n_layer     = 3;
    int64_t max_seq_len = 96;
    int64_t batch_size  = 32;
    int64_t seq_len     = 64;     // fixed per-pair sequence length
    int64_t steps       = 1500;
    int     print_every = 50;
    // The mixed corpus is ~750 real pairs (curated templates + chatterbot
    // training data). We keep regularization off and train long enough for
    // the model to commit to confident templated responses; the real-data
    // pairs teach English fluency on top of that.
    float   dropout     = 0.0f;
    float   lr          = 1e-3f;
    float   weight_decay= 0.0f;
    float   grad_clip   = 1.0f;
    float   label_smooth= 0.0f;
};

// ============================================================================
// CLI parsing
// ============================================================================

struct CliOpts {
    std::string data  = "data/chat_training_mixed.txt";
    int64_t     steps = -1;  // -1 => use config default
};

static CliOpts parse_cli(int argc, char** argv) {
    CliOpts o;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if ((a == "--data" || a == "-d") && i + 1 < argc) o.data = argv[++i];
        else if (a == "--steps" && i + 1 < argc)          o.steps = std::stoll(argv[++i]);
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

    showcase::print_header("Chat AI — nn API (Tiny Transformer LM)", device);

    TrainConfig cfg;
    if (cli.steps > 0) cfg.steps = cli.steps;

    CharVocab vocab;
    const int64_t V = vocab.size();

    auto pairs = load_pairs(cli.data);
    if (pairs.empty()) {
        std::cerr << "No training pairs loaded.\n";
        return 1;
    }

    // Pre-encode every (q, a) as a padded fixed-length int64 sequence.
    std::vector<std::vector<int64_t>> encoded;
    encoded.reserve(pairs.size());
    for (auto& [q, a] : pairs) {
        encoded.push_back(encode_pair(q, a, vocab, cfg.seq_len));
    }

    std::cout << "Training pairs:      " << pairs.size()  << "\n";
    std::cout << "Sequence length:     " << cfg.seq_len   << "\n";
    std::cout << "Vocabulary size:     " << V             << " chars\n";

    showcase::print_section("Model Architecture");
    std::cout << "Decoder-only Transformer LM\n"
              << "  d_model:    " << cfg.d_model     << "\n"
              << "  n_heads:    " << cfg.nhead       << "\n"
              << "  n_layers:   " << cfg.n_layer     << "\n"
              << "  seq_len:    " << cfg.seq_len     << "\n"
              << "  dropout:    " << cfg.dropout     << "\n";

    auto model = std::make_shared<ChatGPT>(
        V, cfg.d_model, cfg.nhead, cfg.n_layer, cfg.max_seq_len, cfg.dropout);
    model->to(device);

    auto params = model->parameters();
    std::cout << "Parameter tensors:   " << params.size() << "\n";

    optim::Adam optimizer(params,
                          /*lr=*/cfg.lr,
                          /*beta1=*/0.9, /*beta2=*/0.95,
                          /*eps=*/1e-9,
                          /*weight_decay=*/cfg.weight_decay);

    CrossEntropyLoss criterion(Reduction::Mean, cfg.label_smooth);

    showcase::print_section("Training");
    std::cout << "Optimizer: Adam (lr=" << cfg.lr
              << ", wd=" << cfg.weight_decay << ")\n"
              << "Loss:      CrossEntropy (label_smoothing=" << cfg.label_smooth << ")\n"
              << "Grad clip: " << cfg.grad_clip << "\n"
              << "Batch:     " << cfg.batch_size << " × " << cfg.seq_len << " tokens\n"
              << "Steps:     " << cfg.steps << "\n\n";

    std::mt19937 rng(123);
    std::uniform_int_distribution<size_t> pair_dist(0, encoded.size() - 1);

    model->train();
    const int64_t in_len = cfg.seq_len - 1;  // input is pair[0:T-1], target pair[1:T]
    std::vector<int64_t> batch_in (cfg.batch_size * in_len);
    std::vector<int64_t> batch_tgt(cfg.batch_size * in_len);

    float running = 0.0f;
    int   running_n = 0;

    for (int64_t step = 1; step <= cfg.steps; ++step) {
        // Sample a batch of random (q, a) pairs — each already padded to seq_len
        for (int64_t b = 0; b < cfg.batch_size; ++b) {
            const auto& seq = encoded[pair_dist(rng)];
            for (int64_t t = 0; t < in_len; ++t) {
                batch_in [b * in_len + t] = seq[t];
                batch_tgt[b * in_len + t] = seq[t + 1];
            }
        }

        auto in_t  = from_data(batch_in.data(),
                               {cfg.batch_size, in_len}, device);
        auto tgt_t = from_data(batch_tgt.data(),
                               {cfg.batch_size * in_len}, device);

        Variable in_v(in_t, false);

        optimizer.zero_grad();
        auto logits = model->forward(in_v);                          // [B, T, V]
        auto flat   = reshape(logits, {cfg.batch_size * in_len, V}); // grad-preserving
        auto loss   = criterion.forward(flat, tgt_t);

        loss.backward();
        nn::utils::clip_grad_norm_(params, cfg.grad_clip);
        optimizer.step();

        float lval = loss.tensor().cpu().data<float>()[0];
        running   += lval;
        running_n += 1;

        if (step == 1 || step % cfg.print_every == 0 || step == cfg.steps) {
            float avg = running / std::max(1, running_n);
            float ppl = std::exp(avg);
            std::cout << "Step [" << std::setw(5) << step << "/" << cfg.steps << "]  "
                      << "loss=" << std::fixed << std::setprecision(4) << avg
                      << "  ppl=" << std::setprecision(2) << ppl << "\n";
            running = 0.0f; running_n = 0;
        }
    }

    // ========================================================================
    // Interactive chat
    // ========================================================================

    const float   temperature = 0.2f;
    const int     top_k       = 3;
    const int64_t max_gen     = 50;
    const int64_t NL          = vocab.nl_idx();

    showcase::print_section("Interactive Chat");
    std::cout << "Chat with the trained model. Type 'quit' or 'exit' to stop.\n"
              << "(sampling: temperature=" << temperature
              << ", top-k=" << top_k << ", max " << max_gen << " chars)\n\n";

    model->eval();

    std::string user;
    while (true) {
        std::cout << "You: " << std::flush;
        if (!std::getline(std::cin, user)) break;
        if (user == "quit" || user == "exit") { std::cout << "Goodbye!\n"; break; }
        if (user.empty()) continue;
        std::transform(user.begin(), user.end(), user.begin(), ::tolower);

        // Seed context: "u: <user>\nb: "
        std::string prompt = "u: " + user + "\nb: ";
        std::vector<int64_t> ctx;
        ctx.reserve(cfg.max_seq_len);
        for (char c : prompt) ctx.push_back(vocab.encode(c));

        std::string reply;
        for (int64_t step = 0; step < max_gen; ++step) {
            // Keep context within max_seq_len (drop oldest)
            int64_t T = static_cast<int64_t>(ctx.size());
            if (T > cfg.max_seq_len) {
                ctx.erase(ctx.begin(), ctx.begin() + (T - cfg.max_seq_len));
                T = cfg.max_seq_len;
            }
            auto in_t = from_data(ctx.data(), {1, T}, device);
            Variable in_v(in_t, false);
            auto logits = model->forward(in_v);

            auto last = last_token_logits(logits.tensor());
            int64_t next = sample_next(last.data(), V, temperature, top_k, rng);

            // Stop when the bot turn ends (newline) — that's how training data is structured
            if (next == NL) break;
            ctx.push_back(next);
            char c = vocab.decode(next);
            if (c >= ' ' && c <= '~') reply.push_back(c);
        }

        std::cout << "Bot: " << (reply.empty() ? "(silence)" : reply) << "\n\n";
    }

    std::cout << "\nTrained a tiny decoder-only Transformer LM end-to-end using Tenzor's nn API.\n";
    finalize();
    return 0;
}
