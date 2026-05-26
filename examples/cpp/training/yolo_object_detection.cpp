/**
 * @file yolo_object_detection.cpp
 * @brief YOLO-style Object Detection Components Demo
 *
 * This example demonstrates:
 * - Detection head architecture
 * - GroupNorm normalization
 * - Mish activation (YOLOv4+) and LeakyReLU (YOLOv3)
 * - Multi-scale feature processing
 * - Detection loss concepts (objectness + classification + localization)
 * - MaxPool2d for downsampling
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <algorithm>

#include "tenzor/tenzor.hpp"
#include "yolo_object_detection_runner.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// YOLO Building Blocks
// ============================================================================

/**
 * @brief Convolutional block with GroupNorm and Mish activation (YOLOv4+)
 */
class ConvBlock : public Module {
public:
    ConvBlock(int64_t in_channels, int64_t out_channels, int kernel_size,
              int stride = 1)
        : out_channels_(out_channels) {

        int padding = kernel_size / 2;

        conv_ = std::make_shared<Conv2d>(in_channels, out_channels, kernel_size,
                                          stride, padding);

        // GroupNorm - alternative to BatchNorm, works better with small batches
        int64_t num_groups = std::min(static_cast<int64_t>(32), out_channels);
        group_norm_ = std::make_shared<GroupNorm>(num_groups, out_channels);

        // Mish activation (used in YOLOv4 and later)
        mish_ = std::make_shared<Mish>();

        register_module("conv", conv_);
        register_module("group_norm", group_norm_);
        register_module("mish", mish_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv_->forward(x);
        h = group_norm_->forward(h);
        h = mish_->forward(h);
        return h;
    }

private:
    int64_t out_channels_;
    std::shared_ptr<Conv2d> conv_;
    std::shared_ptr<GroupNorm> group_norm_;
    std::shared_ptr<Mish> mish_;
};

/**
 * @brief Convolutional block with GroupNorm and LeakyReLU (YOLOv3 style)
 */
class ConvBlockLeaky : public Module {
public:
    ConvBlockLeaky(int64_t in_channels, int64_t out_channels, int kernel_size,
              int stride = 1)
        : out_channels_(out_channels) {

        int padding = kernel_size / 2;

        conv_ = std::make_shared<Conv2d>(in_channels, out_channels, kernel_size,
                                          stride, padding);

        int64_t num_groups = std::min(static_cast<int64_t>(32), out_channels);
        group_norm_ = std::make_shared<GroupNorm>(num_groups, out_channels);

        // LeakyReLU activation (used in YOLOv3 and earlier)
        leaky_relu_ = std::make_shared<LeakyReLU>(0.1f);

        register_module("conv", conv_);
        register_module("group_norm", group_norm_);
        register_module("leaky_relu", leaky_relu_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv_->forward(x);
        h = group_norm_->forward(h);
        h = leaky_relu_->forward(h);
        return h;
    }

private:
    int64_t out_channels_;
    std::shared_ptr<Conv2d> conv_;
    std::shared_ptr<GroupNorm> group_norm_;
    std::shared_ptr<LeakyReLU> leaky_relu_;
};

/**
 * @brief Residual block for feature extraction
 */
class ResidualBlock : public Module {
public:
    ResidualBlock(int64_t channels) {
        conv1_ = std::make_shared<ConvBlock>(channels, channels / 2, 1);
        conv2_ = std::make_shared<ConvBlock>(channels / 2, channels, 3);

        register_module("conv1", conv1_);
        register_module("conv2", conv2_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = conv1_->forward(x);
        h = conv2_->forward(h);
        return x + h;  // Residual connection
    }

private:
    std::shared_ptr<ConvBlock> conv1_;
    std::shared_ptr<ConvBlock> conv2_;
};

/**
 * @brief YOLO Detection Head
 */
class DetectionHead : public Module {
public:
    DetectionHead(int64_t in_channels, int64_t num_classes, int64_t num_anchors = 3)
        : num_classes_(num_classes), num_anchors_(num_anchors) {

        // Output: num_anchors * (5 + num_classes) per cell
        // 5 = x, y, w, h, objectness
        int64_t out_channels = num_anchors * (5 + num_classes);

        conv_ = std::make_shared<Conv2d>(in_channels, out_channels, 1, 1, 0);
        register_module("conv", conv_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        return conv_->forward(x);
    }

    int64_t num_classes() const { return num_classes_; }
    int64_t num_anchors() const { return num_anchors_; }

private:
    int64_t num_classes_;
    int64_t num_anchors_;
    std::shared_ptr<Conv2d> conv_;
};

// ============================================================================
// Simple YOLO-style Backbone
// ============================================================================

class YOLOBackbone : public Module {
public:
    YOLOBackbone(int64_t num_classes = 20) : num_classes_(num_classes) {
        // Stem
        stem_ = std::make_shared<ConvBlock>(3, 32, 3, 1);

        // Downsample stages
        down1_ = std::make_shared<Conv2d>(32, 64, 3, 2, 1);
        res1_ = std::make_shared<ResidualBlock>(64);

        down2_ = std::make_shared<Conv2d>(64, 128, 3, 2, 1);
        res2_ = std::make_shared<ResidualBlock>(128);

        down3_ = std::make_shared<Conv2d>(128, 256, 3, 2, 1);
        res3_ = std::make_shared<ResidualBlock>(256);

        // Detection head
        head_ = std::make_shared<DetectionHead>(256, num_classes, 3);

        // Mish activation (YOLOv4+)
        mish_ = std::make_shared<Mish>();

        register_module("stem", stem_);
        register_module("down1", down1_);
        register_module("res1", res1_);
        register_module("down2", down2_);
        register_module("res2", res2_);
        register_module("down3", down3_);
        register_module("res3", res3_);
        register_module("head", head_);
        register_module("mish", mish_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Stem
        auto h = stem_->forward(x);

        // Stage 1: 32 -> 64, downsample 2x
        h = mish_->forward(Variable(down1_->forward(h).tensor(), h.requires_grad()));
        h = res1_->forward(h);

        // Stage 2: 64 -> 128, downsample 2x
        h = mish_->forward(Variable(down2_->forward(h).tensor(), h.requires_grad()));
        h = res2_->forward(h);

        // Stage 3: 128 -> 256, downsample 2x
        h = mish_->forward(Variable(down3_->forward(h).tensor(), h.requires_grad()));
        h = res3_->forward(h);

        // Detection head
        return head_->forward(h);
    }

private:
    int64_t num_classes_;
    std::shared_ptr<ConvBlock> stem_;
    std::shared_ptr<Conv2d> down1_, down2_, down3_;
    std::shared_ptr<ResidualBlock> res1_, res2_, res3_;
    std::shared_ptr<DetectionHead> head_;
    std::shared_ptr<Mish> mish_;
};

// ============================================================================
// Detection Data Generation
// ============================================================================

std::pair<Tensor, Tensor> generate_detection_data(int num_samples, int img_size,
                                                   int num_classes, Device device) {
    // Generate random images
    auto images = randn({num_samples, 3, img_size, img_size}, DType::Float32, device);

    // Generate random target labels (simplified - just class labels for demo)
    std::vector<int64_t> labels(num_samples);
    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> dist(0, num_classes - 1);
    for (int i = 0; i < num_samples; ++i) {
        labels[i] = dist(rng);
    }
    auto targets = from_data(labels.data(), {num_samples}, device);

    return {images, targets};
}

// ============================================================================
// Demo Functions
// ============================================================================

void demo_groupnorm() {
    std::cout << "\n=== GroupNorm Demo ===\n\n";

    Device device = Device::cpu();

    std::cout << "GroupNorm vs BatchNorm:\n";
    std::cout << "  BatchNorm: Normalizes over (N, H, W) per channel\n";
    std::cout << "  GroupNorm: Normalizes over (H, W, C/G) per group\n\n";

    auto input = randn({4, 64, 16, 16}, DType::Float32, device);
    Variable x(input, false);

    // GroupNorm with different group sizes
    std::vector<int64_t> group_sizes = {8, 16, 32};

    for (auto num_groups : group_sizes) {
        auto gn = std::make_shared<GroupNorm>(num_groups, 64);
        auto output = gn->forward(x);

        // Check output statistics
        auto out_cpu = output.tensor().cpu();
        float mean_val = tenzor::mean(out_cpu).cpu().data<float>()[0];
        float var_val = tenzor::var(out_cpu).cpu().data<float>()[0];

        std::cout << "  GroupNorm(num_groups=" << num_groups << ", channels=64):\n";
        std::cout << "    Output mean: " << std::fixed << std::setprecision(4) << mean_val << "\n";
        std::cout << "    Output var:  " << var_val << "\n\n";
    }

    std::cout << "  Benefits of GroupNorm:\n";
    std::cout << "    - Works well with small batch sizes\n";
    std::cout << "    - More stable in detection tasks\n";
    std::cout << "    - Commonly used in YOLO, Mask R-CNN\n";
}

void demo_leaky_relu_activation() {
    std::cout << "\n=== LeakyReLU Activation Demo ===\n\n";

    Device device = Device::cpu();

    std::cout << "LeakyReLU: max(alpha*x, x) where alpha=0.1 for YOLO\n\n";

    std::vector<float> test_values = {-4.0f, -2.0f, -1.0f, 0.0f, 1.0f, 2.0f, 4.0f};
    auto input = from_data(test_values.data(),
                           {1, static_cast<int64_t>(test_values.size())}, device);
    Variable x(input, false);

    auto leaky_relu = std::make_shared<LeakyReLU>(0.1f);
    auto output = leaky_relu->forward(x);

    std::cout << "Input:  ";
    for (float v : test_values) {
        std::cout << std::setw(7) << std::fixed << std::setprecision(2) << v << " ";
    }
    std::cout << "\n";

    auto out_cpu = output.tensor().cpu();
    const float* out_data = out_cpu.data<float>();
    std::cout << "Output: ";
    for (size_t i = 0; i < test_values.size(); ++i) {
        std::cout << std::setw(7) << std::fixed << std::setprecision(2) << out_data[i] << " ";
    }
    std::cout << "\n\n";

    std::cout << "Properties of LeakyReLU:\n";
    std::cout << "  - Non-zero gradient for negative inputs\n";
    std::cout << "  - Prevents 'dying ReLU' problem\n";
    std::cout << "  - alpha=0.1 commonly used in YOLO\n";
    std::cout << "  - Fast computation\n";
}

void demo_detection_output() {
    std::cout << "\n=== Detection Output Format ===\n\n";

    int num_classes = 20;
    int num_anchors = 3;
    int grid_size = 13;  // For 416x416 input with stride 32

    int output_channels = num_anchors * (5 + num_classes);

    std::cout << "YOLO Detection Head Output:\n";
    std::cout << "  Grid size: " << grid_size << "x" << grid_size << "\n";
    std::cout << "  Anchors per cell: " << num_anchors << "\n";
    std::cout << "  Classes: " << num_classes << "\n";
    std::cout << "  Output channels: " << output_channels << "\n\n";

    std::cout << "Per-anchor output (5 + num_classes = 25 values):\n";
    std::cout << "  [0]: tx (center x offset)\n";
    std::cout << "  [1]: ty (center y offset)\n";
    std::cout << "  [2]: tw (width scale)\n";
    std::cout << "  [3]: th (height scale)\n";
    std::cout << "  [4]: objectness confidence\n";
    std::cout << "  [5:25]: class probabilities\n\n";

    std::cout << "Box decoding:\n";
    std::cout << "  bx = sigmoid(tx) + cx  (cx = grid cell x)\n";
    std::cout << "  by = sigmoid(ty) + cy  (cy = grid cell y)\n";
    std::cout << "  bw = pw * exp(tw)      (pw = anchor width)\n";
    std::cout << "  bh = ph * exp(th)      (ph = anchor height)\n";
}

void demo_maxpool_downsampling() {
    std::cout << "\n=== MaxPool2d for Downsampling ===\n\n";

    Device device = Device::cpu();

    auto input = randn({1, 64, 32, 32}, DType::Float32, device);

    // Different pooling configurations
    std::cout << "MaxPool2d downsampling configurations:\n\n";

    auto pool_2x = std::make_shared<MaxPool2d>(2, 2);
    auto pool_3x = std::make_shared<MaxPool2d>(3, 2, 1);

    Variable x(input, false);

    auto out_2x = pool_2x->forward(x);
    std::cout << "  MaxPool2d(kernel=2, stride=2):\n";
    std::cout << "    Input:  [1, 64, 32, 32]\n";
    std::cout << "    Output: [" << out_2x.shape()[0] << ", "
              << out_2x.shape()[1] << ", "
              << out_2x.shape()[2] << ", "
              << out_2x.shape()[3] << "] (2x downsample)\n\n";

    auto out_3x = pool_3x->forward(x);
    std::cout << "  MaxPool2d(kernel=3, stride=2, padding=1):\n";
    std::cout << "    Input:  [1, 64, 32, 32]\n";
    std::cout << "    Output: [" << out_3x.shape()[0] << ", "
              << out_3x.shape()[1] << ", "
              << out_3x.shape()[2] << ", "
              << out_3x.shape()[3] << "] (2x downsample)\n\n";

    std::cout << "  Use cases:\n";
    std::cout << "    - Feature map downsampling in backbone\n";
    std::cout << "    - SPP (Spatial Pyramid Pooling) modules\n";
    std::cout << "    - Preserves max activations (important features)\n";
}

// ============================================================================
// Training
// ============================================================================

void train_detection_model(Device device) {
    // NN.24: the tiny-backbone training body is shared with the regression
    // test via examples::yolo_object_detection::run_yolo_object_detection_training.
    // Run it first so this exe exercises the same code path the test does,
    // then continue with the original deeper-backbone loop for the demo.
    std::cout << "\n=== Training YOLO-style Detection Model ===\n\n";

    {
        double init_loss = 0.0;
        double final_loss = 0.0;
        const int runner_iters = 10;
        examples::yolo_object_detection::run_yolo_object_detection_training(
            runner_iters, &init_loss, &final_loss, device, /*verbose=*/false);
        std::cout << "[runner] short-loop training: initial=" << init_loss
                  << " final=" << final_loss << " over " << runner_iters
                  << " iterations\n\n";
    }

    int num_classes = 20;
    int img_size = 128;  // Smaller for faster demo
    int batch_size = 4;
    int num_train = 100;
    int num_epochs = 5;

    auto model = std::make_shared<YOLOBackbone>(num_classes);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    // Use MSE loss for regression (simplified detection loss)
    MSELoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: YOLO-style backbone with GroupNorm + LeakyReLU\n";
    std::cout << "  Input: " << img_size << "x" << img_size << " RGB images\n";
    std::cout << "  Classes: " << num_classes << "\n";
    std::cout << "  Anchors: 3 per grid cell\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            // Generate synthetic data
            auto images = randn({actual_batch, 3, img_size, img_size}, DType::Float32, device);
            Variable x(images, true);

            optimizer.zero_grad();

            auto output = model->forward(x);

            // Simplified loss: MSE on output features
            // Real YOLO uses objectness + class + localization losses
            auto out_shape = output.shape();
            std::vector<int64_t> shape_vec(out_shape.begin(), out_shape.end());
            auto target = zeros(shape_vec, DType::Float32, device);
            Variable target_var(target, false);
            auto loss = criterion(output, target_var);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.tensor().cpu().data<float>()[0];
            num_batches++;
        }

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches) << "\n";
    }

    // Show output shape
    model->eval();
    auto test_input = randn({1, 3, img_size, img_size}, DType::Float32, device);
    Variable test_x(test_input, false);
    auto test_output = model->forward(test_x);

    std::cout << "\nModel output shape: [" << test_output.shape()[0] << ", "
              << test_output.shape()[1] << ", "
              << test_output.shape()[2] << ", "
              << test_output.shape()[3] << "]\n";
    std::cout << "  Channels: 3 anchors * (5 + 20 classes) = 75\n";
    std::cout << "  Grid: " << test_output.shape()[2] << "x" << test_output.shape()[3] << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    tenzor::initialize();

    Device device = Device::cpu();
    if (argc > 1) {
        std::string backend = argv[1];
        if (backend == "cuda") device = Device::cuda();
        else if (backend == "vulkan") device = Device::vulkan();
    }

    std::cout << "======================================================\n";
    std::cout << "   YOLO Object Detection - Component Coverage         \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: Conv2d, GroupNorm, MaxPool2d\n";
    std::cout << "  Activations: LeakyReLU\n";
    std::cout << "  Architecture: YOLO-style backbone, detection head\n";
    std::cout << "  Patterns: Residual blocks, multi-scale features\n";

    try {
        demo_groupnorm();
        demo_leaky_relu_activation();
        demo_detection_output();
        demo_maxpool_downsampling();
        train_detection_model(device);

        std::cout << "\n======================================================\n";
        std::cout << "   All detection examples completed successfully!     \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    tenzor::finalize();
    return 0;
}
