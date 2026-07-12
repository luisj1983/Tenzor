/**
 * @file jit_mlp_training_runner.cpp
 * @brief Implementation of the JIT-compiled MLP training/inference example.
 */

#include "jit_mlp_training_runner.hpp"

#include <cmath>
#include <iostream>

#include "tenzor/tenzor.hpp"
#include "tenzor/jit/compile.hpp"

namespace tenzor::examples::jit_mlp {

namespace {

using ::tenzor::DType;
using ::tenzor::NoGradGuard;
using ::tenzor::Tensor;
using ::tenzor::Variable;

// Synthetic regression: y = sin(x0) + 0.5*cos(x1), 2D input -> 1D output.
// Small, fixed dataset (matches the simplicity of the other showcase
// examples' data) so the example runs fast enough to be a real regression
// test, not just a demo.
constexpr int64_t kSamples = 32;

auto make_dataset(const ::tenzor::Device& device)
    -> std::pair<Tensor, Tensor> {
    std::vector<float> x_data(kSamples * 2);
    std::vector<float> y_data(kSamples);
    for (int64_t i = 0; i < kSamples; ++i) {
        const float x0 = -3.14159f + (2.0f * 3.14159f) *
                          (static_cast<float>(i) / static_cast<float>(kSamples - 1));
        const float x1 = 0.5f * x0;
        x_data[2 * i]     = x0;
        x_data[2 * i + 1] = x1;
        y_data[i] = std::sin(x0) + 0.5f * std::cos(x1);
    }
    Tensor x_t({kSamples, 2}, DType::Float32, ::tenzor::Device::cpu());
    Tensor y_t({kSamples, 1}, DType::Float32, ::tenzor::Device::cpu());
    std::copy(x_data.begin(), x_data.end(), x_t.data<float>());
    std::copy(y_data.begin(), y_data.end(), y_t.data<float>());
    return {x_t.to(device), y_t.to(device)};
}

}  // namespace

int run_jit_mlp_training(int epochs,
                         double* out_initial,
                         double* out_final,
                         double* out_jit_vs_eager_diff,
                         ::tenzor::Device device,
                         bool verbose) {
    using namespace ::tenzor;

    manual_seed(42);

    auto [x_t, y_t] = make_dataset(device);

    constexpr int64_t kHidden = 16;
    Variable W1(randn({2, kHidden}, DType::Float32, device) * 0.5f, true);
    Variable b1(zeros({1, kHidden}, DType::Float32, device), true);
    Variable W2(randn({kHidden, 1}, DType::Float32, device) * 0.5f, true);
    Variable b2(zeros({1, 1}, DType::Float32, device), true);

    const float learning_rate = 0.05f;
    const int print_every = std::max(1, epochs / 10);

    // Shared forward-pass logic used both for eager training AND (below,
    // read-only after training) for the JIT-compiled inference comparison.
    auto forward = [&](const Variable& x) -> Variable {
        auto z1 = x.matmul(W1) + b1;
        auto a1 = nn::relu(z1);
        auto z2 = a1.matmul(W2) + b2;
        return z2;
    };

    double final_loss = 0.0;
    for (int epoch = 0; epoch < epochs; ++epoch) {
        Variable x(x_t, false);
        Variable y(y_t, false);

        auto pred = forward(x);
        auto error = pred - y;
        auto loss = mean(error * error);

        W1.zero_grad(); b1.zero_grad();
        W2.zero_grad(); b2.zero_grad();
        loss.backward();

        {
            NoGradGuard ng;
            W1 = Variable(W1.tensor() - (*W1.grad() * learning_rate), true);
            b1 = Variable(b1.tensor() - (*b1.grad() * learning_rate), true);
            W2 = Variable(W2.tensor() - (*W2.grad() * learning_rate), true);
            b2 = Variable(b2.tensor() - (*b2.grad() * learning_rate), true);
        }

        const double loss_val = static_cast<double>(loss.tensor().item<float>());
        if (epoch == 0 && out_initial) *out_initial = loss_val;
        final_loss = loss_val;

        if (verbose && ((epoch + 1) % print_every == 0 || epoch == 0)) {
            std::cout << "Epoch [" << (epoch + 1) << "/" << epochs
                      << "] loss=" << loss_val << "\n";
        }
    }
    if (out_final) *out_final = final_loss;

    // R2-02: with training done, JIT-compile the trained model's inference
    // forward pass and confirm it matches eager on a fresh input -- a real
    // trace -> compile -> execute -> verify regression check, the exact gap
    // this example closes (previously no example in this repo exercised the
    // JIT subsystem at all). Default "nvrtc" backend is used (not "mlir") so
    // this example builds and runs regardless of whether
    // TENZOR_USE_MLIR_JIT was enabled at configure time.
    if (out_jit_vs_eager_diff) {
        auto jit_fn = ::tenzor::jit::CompiledFunction::FnType(
            [&](const Variable& x) -> Variable { return forward(x); });
        ::tenzor::jit::CompileConfig cfg;
        ::tenzor::jit::CompiledFunction compiled(jit_fn, cfg);

        Variable test_input(x_t, /*requires_grad=*/false);
        const Variable eager_out = forward(test_input);
        Variable jit_out;
        try {
            jit_out = compiled(test_input);
        } catch (const std::exception& e) {
            std::cerr << "JIT compile/execute failed: " << e.what() << "\n";
            return 1;
        }
        const double diff = ::tenzor::max(::tenzor::abs(
            eager_out.tensor() - jit_out.tensor())).item<float>();
        *out_jit_vs_eager_diff = diff;
        if (verbose) {
            std::cout << "JIT vs eager max abs diff: " << diff << "\n";
        }
    }

    return 0;
}

}  // namespace tenzor::examples::jit_mlp
