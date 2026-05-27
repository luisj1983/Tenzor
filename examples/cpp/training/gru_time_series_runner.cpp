/**
 * @file gru_time_series_runner.cpp
 * @brief Implementation of the GRU time-series autograd runner.
 *
 * RR.18 (audit-11): exercises GRU + LayerNorm + Dropout + Linear + MSELoss
 * + Adam end-to-end on a tiny synthetic sine-wave batch. Small hyperparams
 * (batch=2, seq_len=16) keep the regression sub-second on CPU.
 */

#include "gru_time_series_runner.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <memory>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::gru_time_series {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::GRU;
using ::tenzor::nn::LayerNorm;
using ::tenzor::nn::Dropout;
using ::tenzor::nn::Linear;
using ::tenzor::nn::MSELoss;

class GRUPredictor : public Module {
public:
    GRUPredictor(int64_t input_size, int64_t hidden_size, int64_t output_size)
        : hidden_size_(hidden_size) {
        gru_     = std::make_shared<GRU>(input_size, hidden_size);
        norm_    = std::make_shared<LayerNorm>(
            std::vector<int64_t>{hidden_size});
        dropout_ = std::make_shared<Dropout>(0.1f);
        fc_      = std::make_shared<Linear>(hidden_size, output_size);
        register_module("gru", gru_);
        register_module("norm", norm_);
        register_module("dropout", dropout_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto gru_out = gru_->forward(input);  // (batch, seq_len, hidden)
        auto batch_size = input.shape()[0];
        auto seq_len = input.shape()[1];
        auto last_hidden = gru_out.tensor()
            .slice(1, seq_len - 1, seq_len)
            .reshape({batch_size, hidden_size_});
        Variable last_h(last_hidden, gru_out.requires_grad());
        auto normed = norm_->forward(last_h);
        auto dropped = dropout_->forward(normed);
        return fc_->forward(dropped);
    }

private:
    int64_t hidden_size_;
    std::shared_ptr<GRU> gru_;
    std::shared_ptr<LayerNorm> norm_;
    std::shared_ptr<Dropout> dropout_;
    std::shared_ptr<Linear> fc_;
};

std::pair<Tensor, Tensor> generate_sine_data(int num_samples, int seq_len,
                                              ::tenzor::Device device) {
    using namespace ::tenzor;
    std::vector<float> X_data(num_samples * seq_len);
    std::vector<float> y_data(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        float phase = static_cast<float>(i) * 0.1f;
        for (int t = 0; t < seq_len; ++t) {
            X_data[i * seq_len + t] = std::sin(phase + t * 0.1f);
        }
        y_data[i] = std::sin(phase + seq_len * 0.1f);
    }
    auto X = from_data(X_data.data(), {num_samples, seq_len, 1}, device);
    auto y = from_data(y_data.data(), {num_samples, 1}, device);
    return {X, y};
}

}  // namespace

int run_gru_training(int num_steps,
                     double* out_initial,
                     double* out_final,
                     ::tenzor::Device device,
                     bool verbose) {
    using namespace ::tenzor;

    const int num_samples = 2;     // batch
    const int seq_len     = 16;
    const int hidden_size = 16;

    manual_seed(42);

    auto [X, y] = generate_sine_data(num_samples, seq_len, device);
    auto model = std::make_shared<GRUPredictor>(1, hidden_size, 1);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.01f);
    MSELoss criterion;

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        optimizer.zero_grad();
        Variable input(X, false);
        auto pred = model->forward(input);
        Variable target(y, false);
        auto loss = criterion(pred, target);
        loss.backward();
        optimizer.step();

        const double loss_v = static_cast<double>(loss.tensor().item<float>());
        if (step == 0) initial_loss = loss_v;
        final_loss = loss_v;
        if (verbose) {
            std::cout << "step " << (step + 1) << "/" << num_steps
                      << " loss=" << loss_v << "\n";
        }
    }

    if (out_initial) *out_initial = initial_loss;
    if (out_final)   *out_final   = final_loss;
    return 0;
}

}  // namespace tenzor::examples::gru_time_series
