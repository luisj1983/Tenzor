/**
 * @file gradient_checkpointing_runner.cpp
 * @brief Implementation of the gradient-checkpointing autograd runner.
 *
 * KK.27: extracted from gradient_checkpointing.cpp's train_deep_network()
 * so the regression test in tests/examples/test_all_autograd_examples.cpp
 * can drive the same training loop and assert loss decreases.
 *
 * Compared to the standalone example we (a) accept the iteration count as
 * a parameter so the test target can run a 4-iteration smoke test, and
 * (b) report the initial and final loss values so the test can EXPECT_GT
 * the difference.
 */

#include "gradient_checkpointing_runner.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::gradient_checkpointing {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Conv2d;
using ::tenzor::nn::BatchNorm2d;
using ::tenzor::nn::ReLU;
using ::tenzor::nn::AdaptiveAvgPool2d;
using ::tenzor::nn::Linear;
using ::tenzor::nn::MSELoss;

class ResBlock : public Module {
public:
    explicit ResBlock(int64_t channels) {
        conv1_ = std::make_shared<Conv2d>(channels, channels, 3, 1, 1);
        bn1_   = std::make_shared<BatchNorm2d>(channels);
        conv2_ = std::make_shared<Conv2d>(channels, channels, 3, 1, 1);
        bn2_   = std::make_shared<BatchNorm2d>(channels);
        relu_  = std::make_shared<ReLU>();
        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("conv2", conv2_);
        register_module("bn2", bn2_);
        register_module("relu", relu_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = relu_->forward(bn1_->forward(conv1_->forward(x)));
        h = bn2_->forward(conv2_->forward(h));
        return relu_->forward(x + h);
    }

private:
    std::shared_ptr<Conv2d> conv1_, conv2_;
    std::shared_ptr<BatchNorm2d> bn1_, bn2_;
    std::shared_ptr<ReLU> relu_;
};

class DeepResNet : public Module {
public:
    DeepResNet(int64_t num_blocks, int64_t channels) {
        stem_      = std::make_shared<Conv2d>(3, channels, 7, 2, 3);
        stem_bn_   = std::make_shared<BatchNorm2d>(channels);
        stem_relu_ = std::make_shared<ReLU>();
        for (int64_t i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<ResBlock>(channels);
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }
        pool_ = std::make_shared<AdaptiveAvgPool2d>(1, 1);
        fc_   = std::make_shared<Linear>(channels, 10);
        register_module("stem", stem_);
        register_module("stem_bn", stem_bn_);
        register_module("pool", pool_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = stem_relu_->forward(stem_bn_->forward(stem_->forward(x)));
        for (auto& block : blocks_) {
            h = block->forward(h);
        }
        h = pool_->forward(h);
        auto h_tensor = h.tensor();
        int64_t batch_size = h_tensor.shape()[0];
        int64_t channels = h_tensor.shape()[1];
        auto h_flat = h_tensor.reshape({batch_size, channels});
        Variable h_var(h_flat, h.requires_grad());
        return fc_->forward(h_var);
    }

private:
    std::shared_ptr<Conv2d> stem_;
    std::shared_ptr<BatchNorm2d> stem_bn_;
    std::shared_ptr<ReLU> stem_relu_;
    std::vector<std::shared_ptr<ResBlock>> blocks_;
    std::shared_ptr<AdaptiveAvgPool2d> pool_;
    std::shared_ptr<Linear> fc_;
};

}  // namespace

int run_gradient_checkpointing_training(int num_iterations,
                                         double* out_initial,
                                         double* out_final,
                                         ::tenzor::Device device,
                                         bool verbose) {
    using namespace ::tenzor;

    // Keep the test workload small so it stays well under a few seconds
    // even with the deep ResNet. The standalone exe still does the bigger
    // train_deep_network() loop in main().
    const int64_t num_blocks = 2;
    const int64_t channels   = 16;
    const int img_size       = 16;
    const int batch_size     = 4;

    manual_seed(42);

    auto model = std::make_shared<DeepResNet>(num_blocks, channels);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);
    MSELoss criterion;

    std::mt19937 rng(42);
    std::normal_distribution<float> target_dist(0.0f, 1.0f);

    // Reuse a single fixed batch across iterations so loss has to decrease
    // monotonically (modulo BN running-stat warm-up noise). The standalone
    // example sweeps fresh data per iteration, but the regression test
    // needs a stable signal that backward is moving the weights.
    auto images = randn({batch_size, 3, img_size, img_size},
                        DType::Float32, device);
    std::vector<float> target_data(batch_size * 10);
    for (auto& t : target_data) {
        t = target_dist(rng);
    }
    auto targets = from_data(target_data.data(), {batch_size, 10}, device);

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        Variable x(images, true);
        Variable target_var(targets, false);

        optimizer.zero_grad();
        auto output = model->forward(x);
        auto loss = criterion(output, target_var);
        loss.backward();
        optimizer.step();

        const double loss_val = static_cast<double>(
            loss.tensor().item<float>());
        if (i == 0) initial_loss = loss_val;
        final_loss = loss_val;

        if (verbose) {
            std::cout << "iter " << (i + 1) << "/" << num_iterations
                      << " loss=" << loss_val << "\n";
        }
    }

    if (out_initial) *out_initial = initial_loss;
    if (out_final)   *out_final   = final_loss;
    return 0;
}

}  // namespace tenzor::examples::gradient_checkpointing
