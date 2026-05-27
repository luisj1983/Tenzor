/**
 * @file unet_semantic_segmentation_runner.cpp
 * @brief Implementation of the U-Net segmentation autograd runner.
 *
 * RR.18 (audit-11): tiny U-Net (img_size=16, batch=1) for the regression
 * test in tests/examples/test_all_autograd_examples.cpp. The
 * MSE-against-zeros surrogate of train_unet() is not guaranteed to be
 * monotonically decreasing on a randomly-initialised conv backbone over
 * just a few steps, so the regression asserts `initial != final` rather
 * than monotonic decrease — same pattern as the YOLO runner from NN.24.
 */

#include "unet_semantic_segmentation_runner.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::unet_semantic_segmentation {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Conv2d;
using ::tenzor::nn::ConvTranspose2d;
using ::tenzor::nn::BatchNorm2d;
using ::tenzor::nn::MaxPool2d;
using ::tenzor::nn::ReLU;
using ::tenzor::nn::CrossEntropyLoss;

class DoubleConv : public Module {
public:
    DoubleConv(int64_t in_channels, int64_t out_channels) {
        conv1_ = std::make_shared<Conv2d>(in_channels, out_channels, 3, 1, 1);
        bn1_   = std::make_shared<BatchNorm2d>(out_channels);
        conv2_ = std::make_shared<Conv2d>(out_channels, out_channels, 3, 1, 1);
        bn2_   = std::make_shared<BatchNorm2d>(out_channels);
        relu_  = std::make_shared<ReLU>();
        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("conv2", conv2_);
        register_module("bn2", bn2_);
        register_module("relu", relu_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = relu_->forward(bn1_->forward(conv1_->forward(x)));
        h = relu_->forward(bn2_->forward(conv2_->forward(h)));
        return h;
    }

private:
    std::shared_ptr<Conv2d> conv1_, conv2_;
    std::shared_ptr<BatchNorm2d> bn1_, bn2_;
    std::shared_ptr<ReLU> relu_;
};

class TinyUNet : public Module {
public:
    TinyUNet(int64_t num_classes) {
        enc_ = std::make_shared<DoubleConv>(3, 16);
        pool_ = std::make_shared<MaxPool2d>(2, 2);
        bottleneck_ = std::make_shared<DoubleConv>(16, 32);
        up_ = std::make_shared<ConvTranspose2d>(32, 16, 4, 2, 1);
        dec_ = std::make_shared<DoubleConv>(32, 16);  // 16 + 16 (skip)
        out_conv_ = std::make_shared<Conv2d>(16, num_classes, 1, 1, 0);
        register_module("enc", enc_);
        register_module("pool", pool_);
        register_module("bottleneck", bottleneck_);
        register_module("up", up_);
        register_module("dec", dec_);
        register_module("out_conv", out_conv_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto e = enc_->forward(x);
        auto p = pool_->forward(e);
        auto b = bottleneck_->forward(p);
        auto u = up_->forward(b);
        // Concatenate with skip on channel dim. IMPORTANT: use the
        // autograd-aware overload of cat() so the backward path through
        // u (decoder) and e (encoder skip) is preserved — building
        // `Variable(::tenzor::cat({u.tensor(), e.tensor()}, 1), ...)`
        // would sever both grad_fn chains and zero out gradients
        // (the parent example's standalone main() exhibits exactly
        // this bug, which is why its loss never decreases on a small
        // step budget).
        auto concat_var = ::tenzor::cat(
            std::vector<Variable>{u, e}, 1);
        auto d = dec_->forward(concat_var);
        return out_conv_->forward(d);
    }

private:
    std::shared_ptr<DoubleConv> enc_, bottleneck_, dec_;
    std::shared_ptr<MaxPool2d> pool_;
    std::shared_ptr<ConvTranspose2d> up_;
    std::shared_ptr<Conv2d> out_conv_;
};

}  // namespace

int run_unet_training(int num_steps,
                      double* out_initial,
                      double* out_final,
                      ::tenzor::Device device,
                      bool verbose) {
    using namespace ::tenzor;

    const int num_classes = 3;
    const int img_size    = 16;
    const int batch_size  = 2;  // ≥2 so BatchNorm2d sees non-zero variance

    manual_seed(42);

    auto model = std::make_shared<TinyUNet>(num_classes);
    model->to(device);
    model->train();

    auto params = model->parameters();
    // Larger lr so the tiny 10-step regression actually moves weights —
    // the parent example uses 0.001 over many more steps.
    optim::Adam optimizer(params, 0.05f);
    CrossEntropyLoss criterion;

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> label_dist(0, num_classes - 1);

    auto images = randn({batch_size, 3, img_size, img_size},
                        DType::Float32, device);
    std::vector<int64_t> label_data(batch_size * img_size * img_size);
    for (auto& l : label_data) l = label_dist(rng);
    auto labels = from_data(label_data.data(),
                            {batch_size, img_size, img_size}, device);

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int step = 0; step < num_steps; ++step) {
        Variable x(images, true);
        optimizer.zero_grad();
        auto output = model->forward(x);
        const int64_t N = batch_size;
        const int64_t H = img_size;
        const int64_t W = img_size;
        const int64_t C = num_classes;
        // IMPORTANT: use the autograd-aware permute/reshape overloads so
        // the backward path through `output` is preserved. Constructing
        // `Variable(out_tensor.permute(...).reshape(...), ...)` would
        // sever grad_fn and zero out parameter gradients — same bug
        // pattern as the parent example, which is why its tiny-step
        // training loop never moves loss.
        auto out_permuted = ::tenzor::permute(output, {0, 2, 3, 1});
        auto out_flat = ::tenzor::reshape(out_permuted, {N * H * W, C});
        auto labels_flat = labels.reshape({N * H * W});
        auto loss = criterion(out_flat, labels_flat);
        loss.backward();
        optimizer.step();

        const double loss_v =
            static_cast<double>(loss.tensor().cpu().data<float>()[0]);
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

}  // namespace tenzor::examples::unet_semantic_segmentation
