/**
 * @file yolo_object_detection_runner.cpp
 * @brief Implementation of the YOLO-style detection autograd runner.
 *
 * NN.24: extracted from yolo_object_detection.cpp's train_detection_model()
 * training body so the regression test in
 * tests/examples/test_all_autograd_examples.cpp can drive the
 * Conv2d + GroupNorm + Mish + LeakyReLU + ResidualBlock + Adam pipeline
 * end-to-end and assert that backward actually moves the weights.
 *
 * Uses a deliberately small backbone (one ConvBlock stem + one residual
 * block + a 1x1 detection head) so the test wall-time stays in the
 * 1-2 second range. The standalone exe keeps the original deeper
 * backbone.
 */

#include "yolo_object_detection_runner.hpp"

#include <algorithm>
#include <iostream>
#include <memory>
#include <random>
#include <vector>

#include "tenzor/tenzor.hpp"

namespace tenzor::examples::yolo_object_detection {

namespace {

using ::tenzor::Variable;
using ::tenzor::Tensor;
using ::tenzor::nn::Module;
using ::tenzor::nn::Conv2d;
using ::tenzor::nn::GroupNorm;
using ::tenzor::nn::Mish;
using ::tenzor::nn::MSELoss;

/// Conv -> GroupNorm -> Mish (the YOLOv4+ pattern used in the example).
class ConvBlock : public Module {
public:
    ConvBlock(int64_t in_channels, int64_t out_channels, int kernel_size,
              int stride = 1) {
        const int padding = kernel_size / 2;
        conv_ = std::make_shared<Conv2d>(in_channels, out_channels,
                                          kernel_size, stride, padding);
        const int64_t num_groups =
            std::min(static_cast<int64_t>(4), out_channels);
        group_norm_ = std::make_shared<GroupNorm>(num_groups, out_channels);
        mish_ = std::make_shared<Mish>();
        register_module("conv", conv_);
        register_module("group_norm", group_norm_);
        register_module("mish", mish_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv_->forward(x);
        h = group_norm_->forward(h);
        return mish_->forward(h);
    }

private:
    std::shared_ptr<Conv2d> conv_;
    std::shared_ptr<GroupNorm> group_norm_;
    std::shared_ptr<Mish> mish_;
};

/// 1x1 -> 3x3 residual block (mirrors the example's ResidualBlock).
class ResidualBlock : public Module {
public:
    explicit ResidualBlock(int64_t channels) {
        conv1_ = std::make_shared<ConvBlock>(channels, channels / 2, 1);
        conv2_ = std::make_shared<ConvBlock>(channels / 2, channels, 3);
        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv1_->forward(x);
        h = conv2_->forward(h);
        return x + h;
    }

private:
    std::shared_ptr<ConvBlock> conv1_;
    std::shared_ptr<ConvBlock> conv2_;
};

class TinyYOLOBackbone : public Module {
public:
    TinyYOLOBackbone(int64_t num_classes, int64_t channels,
                     int64_t num_anchors) {
        stem_ = std::make_shared<ConvBlock>(3, channels, 3, 1);
        res_  = std::make_shared<ResidualBlock>(channels);
        // Detection head: 1x1 conv mapping to num_anchors * (5 + num_classes)
        const int64_t out_channels = num_anchors * (5 + num_classes);
        head_ = std::make_shared<Conv2d>(channels, out_channels, 1, 1, 0);
        register_module("stem", stem_);
        register_module("res", res_);
        register_module("head", head_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = stem_->forward(x);
        h = res_->forward(h);
        return head_->forward(h);
    }

private:
    std::shared_ptr<ConvBlock> stem_;
    std::shared_ptr<ResidualBlock> res_;
    std::shared_ptr<Conv2d> head_;
};

}  // namespace

int run_yolo_object_detection_training(int num_iterations,
                                        double* out_initial,
                                        double* out_final,
                                        ::tenzor::Device device,
                                        bool verbose) {
    using namespace ::tenzor;

    // Small workload — the standalone exe keeps the original 128x128 image
    // / 100-sample / 5-epoch loop in main().
    const int64_t num_classes = 4;
    const int64_t channels    = 8;
    const int64_t num_anchors = 3;
    const int img_size        = 16;
    const int batch_size      = 2;

    manual_seed(42);

    auto model = std::make_shared<TinyYOLOBackbone>(
        num_classes, channels, num_anchors);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);
    MSELoss criterion;

    // Reuse a single fixed batch + target across iterations so the
    // loss signal is comparable between iter 0 and iter N — the
    // example's main() resamples per iteration, but a stable batch
    // gives the regression test a clean monotonic signal.
    auto images = randn({batch_size, 3, img_size, img_size},
                        DType::Float32, device);

    double initial_loss = 0.0;
    double final_loss   = 0.0;

    for (int i = 0; i < num_iterations; ++i) {
        Variable x(images, true);

        optimizer.zero_grad();
        auto output = model->forward(x);

        auto out_shape = output.shape();
        std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
        auto target = zeros(shape_vec, DType::Float32, device);
        Variable target_var(target, false);
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

}  // namespace tenzor::examples::yolo_object_detection
