/**
 * @file neural_network.cpp
 * @brief Hierarchical Reasoning Model with Q-Learning Adaptive Compute
 *
 * Trains a full nn::HRM (the brain-inspired two-timescale transformer
 * from Wang et al., 2025) on a variable-difficulty mod-3 sum task and
 * reports how many H-cycles its Adaptive Computational Time module
 * actually consumes per difficulty bucket at eval.
 *
 * Task per sample:
 *   input  : length-6 sequence of digits, one of three difficulty buckets
 *              easy   - tokens drawn from {0, 1}
 *              medium - tokens drawn from {0..2}
 *              hard   - tokens drawn from {0..3}
 *   target : (sum of the digits) mod 3
 *
 * The training loop uses HRM's segment-based forward pass with deep
 * supervision; the ACT halting decisions run inside each forward.
 * After training, the per-bucket eval prints `actual_high_cycles` —
 * the number of high-level cycles the Q-head chose for that input.
 *
 * Usage: ./22_hierarchical_reasoning_neural_network --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <iomanip>
#include <random>
#include <vector>

using namespace tenzor;

struct Batch {
    Tensor inputs;        // (B, T) Int64 tokens
    Tensor labels_pos;    // (B, T) Int64 — per-sample label broadcast across T
    Tensor labels_flat;   // (B,)   Int64 — per-sample label
};

// Build a batch where every digit is sampled uniformly from [0, max_token]
static Batch make_batch(int batch, int seq, int n_classes,
                        int max_token, std::mt19937& rng, Device device) {
    std::uniform_int_distribution<int> draw(0, max_token);
    std::vector<int64_t> tokens(batch * seq);
    std::vector<int64_t> labels(batch);
    std::vector<int64_t> labels_pos(batch * seq);
    for (int b = 0; b < batch; ++b) {
        int s = 0;
        for (int t = 0; t < seq; ++t) {
            int d = draw(rng);
            tokens[b * seq + t] = d;
            s += d;
        }
        labels[b] = s % n_classes;
        for (int t = 0; t < seq; ++t) labels_pos[b * seq + t] = labels[b];
    }
    return Batch{
        from_data(tokens.data(),     {batch, seq}, device),
        from_data(labels_pos.data(), {batch, seq}, device),
        from_data(labels.data(),     {batch},      device),
    };
}

// Argmax over the last (class) dim of a (B, T, C) tensor; pick position
// T-1 (the most context-aware step) and compare to per-sample labels.
static float per_sample_accuracy(const Tensor& logits_btc,
                                 const Tensor& labels_b) {
    auto cpu_logits = logits_btc.cpu();
    auto cpu_labels = labels_b.cpu();
    int B = static_cast<int>(cpu_logits.shape()[0]);
    int T = static_cast<int>(cpu_logits.shape()[1]);
    int C = static_cast<int>(cpu_logits.shape()[2]);
    const float*   lp = cpu_logits.data<float>();
    const int64_t* yp = cpu_labels.data<int64_t>();
    int hits = 0;
    for (int b = 0; b < B; ++b) {
        const float* row = lp + (b * T + (T - 1)) * C;
        int best = 0;
        float bv = row[0];
        for (int c = 1; c < C; ++c) if (row[c] > bv) { bv = row[c]; best = c; }
        if (best == yp[b]) ++hits;
    }
    return static_cast<float>(hits) / static_cast<float>(B);
}

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Hierarchical Reasoning Model - Neural Network API", device);
    manual_seed(42);

    // ------------------------------------------------------------------
    // HRM configuration
    // ------------------------------------------------------------------
    nn::HRMConfig cfg;
    cfg.d_model            = 32;
    cfg.n_heads            = 4;
    cfg.d_feedforward      = 64;
    cfg.n_high_cycles      = 4;     // ACT may halt earlier than this
    cfg.t_low_steps        = 2;
    cfg.dropout            = 0.0;
    cfg.deep_supervision   = true;
    cfg.max_seq_len        = 8;
    cfg.vocab_size         = 4;     // digits 0..3 -> embedding
    cfg.num_classes        = 3;     // sum mod 3
    cfg.use_stablemax      = false; // raw logits, plain CE loss
    cfg.use_act            = true;  // turn on ACT
    cfg.use_qlearning_act  = true;  // Q-learning halting (paper default)
    cfg.act_epsilon        = 0.1;
    cfg.act_gamma          = 0.99;
    cfg.act_lr             = 0.01;
    cfg.max_segments       = 2;     // small: keeps segment-count variance low
    cfg.use_lecun_init     = true;

    nn::HRM model(cfg);
    model.to(device);

    showcase::print_section("Model");
    std::cout << "nn::HRM(d_model=" << cfg.d_model
              << ", heads=" << cfg.n_heads
              << ", N=" << cfg.n_high_cycles
              << ", T=" << cfg.t_low_steps
              << ", vocab=" << cfg.vocab_size
              << ", classes=" << cfg.num_classes << ")\n";
    std::cout << "Adaptive Compute: Q-Learning ACT (eps=" << cfg.act_epsilon
              << ", max_segments=" << cfg.max_segments << ")\n";
    std::cout << "Parameters: " << model.num_parameters() << "\n";

    // ------------------------------------------------------------------
    // Optimizer + loss helpers
    // ------------------------------------------------------------------
    auto params = model.parameters();
    optim::Adam opt(params, 0.001f);

    // Cross-entropy on (B*T, C) logits with (B*T,) Int64 targets.
    auto ce_fn = [](const Variable& pred, const Variable& target) -> Variable {
        auto shape = pred.shape();
        int64_t B = shape[0], T = shape[1], C = shape[2];
        auto pred_flat = reshape(pred, {B * T, C});
        Tensor target_flat = target.tensor().reshape({B * T});
        nn::CrossEntropyLoss ce;
        return ce(pred_flat, target_flat);
    };

    // ------------------------------------------------------------------
    // Training: mixed-difficulty batches with deep supervision + q_loss
    // ------------------------------------------------------------------
    const int seq         = 6;
    const int batch       = 32;
    const int num_epochs  = 300;
    const int print_every = 30;
    std::mt19937 rng(2026);
    std::uniform_int_distribution<int> bucket(0, 2);
    // Difficulty rises with token range — more carry-classes can collide
    // modulo 3, so the model has to combine more partial sums.
    const int max_token_for_bucket[3] = {1, 2, 3};

    showcase::print_section("Training (mixed easy/medium/hard)");
    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model.train();
        opt.zero_grad();

        // Pick a difficulty for the whole batch this step
        int max_tok = max_token_for_bucket[bucket(rng)];
        auto bx = make_batch(batch, seq, cfg.num_classes, max_tok, rng, device);

        Variable input(bx.inputs, false);
        Variable target_pos(bx.labels_pos, false);

        auto [final_out, seg_outs, q_loss] =
            model.forward_with_segments(input, target_pos);

        Variable sup_loss =
            nn::hrm_deep_supervision_loss(seg_outs, target_pos, ce_fn, 0.5);
        Variable loss = sup_loss + q_loss;

        loss.backward();
        opt.step();

        if ((epoch + 1) % print_every == 0 || epoch == 0) {
            auto stats = model.last_forward_stats();
            float acc = per_sample_accuracy(final_out.tensor(), bx.labels_flat);
            std::cout << "Epoch [" << std::setw(3) << (epoch + 1) << "/"
                      << num_epochs << "]"
                      << "  bucket=max" << max_tok
                      << "  loss=" << std::fixed << std::setprecision(3)
                      << loss.tensor().item<float>()
                      << "  acc=" << std::setw(5) << std::setprecision(1)
                      << (acc * 100.0f) << "%"
                      << "  segs=" << stats.actual_segments
                      << "  cycles=" << stats.actual_high_cycles << "\n";
        }

        // Decay exploration so eval-time halting reflects what was learned
        if (auto qact = model.get_qlearning_act()) qact->decay_epsilon(0.99, 0.02);
    }

    // ------------------------------------------------------------------
    // Evaluation: one batch per difficulty, report ACT halt point
    // ------------------------------------------------------------------
    showcase::print_section("Adaptive Compute Report");
    std::cout << "Evaluating on a fresh batch per difficulty bucket.\n"
              << "ACT chooses how many H-cycles (out of " << cfg.n_high_cycles
              << ") to use per forward pass.\n\n";

    std::cout << std::left
              << std::setw(10) << "bucket"
              << std::setw(12) << "tokens"
              << std::setw(14) << "acc@last"
              << std::setw(14) << "H-cycles"
              << "Q(halt) | Q(continue)\n";
    std::cout << std::string(70, '-') << "\n";

    model.eval();
    std::mt19937 eval_rng(9001);
    const char* bucket_name[3] = {"easy", "medium", "hard"};
    for (int b = 0; b < 3; ++b) {
        int max_tok = max_token_for_bucket[b];
        auto eb = make_batch(/*batch*/64, seq, cfg.num_classes,
                             max_tok, eval_rng, device);
        Variable inp(eb.inputs, false);
        auto out = model.forward(inp);
        auto stats = model.last_forward_stats();
        float acc = per_sample_accuracy(out.tensor(), eb.labels_flat);

        std::cout << std::left
                  << std::setw(10) << bucket_name[b]
                  << std::setw(12) << ("0.." + std::to_string(max_tok))
                  << std::setw(14)
                  << (std::to_string(static_cast<int>(acc * 100.0f)) + "%")
                  << std::setw(14)
                  << (std::to_string(stats.actual_high_cycles) + "/"
                      + std::to_string(cfg.n_high_cycles))
                  << std::fixed << std::setprecision(3) << stats.avg_q_halt
                  << "  | " << stats.avg_q_continue << "\n";
    }

    std::cout << "\nThe H-cycles column is the live ACT halt point per input —\n"
                 "the Q-head's decision is recomputed for every batch, so the\n"
                 "compute budget adapts to the data rather than being fixed.\n";

    finalize();
    return 0;
}
