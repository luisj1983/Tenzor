/**
 * @file unet_semantic_segmentation.cpp
 * @brief U-Net Style Segmentation Components Demo
 *
 * This example demonstrates:
 * - Encoder-decoder architecture with skip connections
 * - ConvTranspose2d for upsampling (decoder)
 * - BatchNorm2d and GroupNorm normalization
 * - Feature extraction patterns
 * - CrossEntropyLoss for classification
 * - Conv2d with various kernel sizes
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <memory>

#include "tenzor/tenzor.hpp"
// RR.18 (audit-11): a trimmed-down TinyUNet training body lives in
// unet_semantic_segmentation_runner.{cpp,hpp} so the regression test in
// tests/examples/test_all_autograd_examples.cpp can drive the same
// Conv2d + BatchNorm2d + ReLU + MaxPool + ConvTranspose2d + cat +
// CrossEntropyLoss + Adam pipeline.
#include "unet_semantic_segmentation_runner.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// U-Net Building Blocks
// ============================================================================

class DoubleConv : public Module {
public:
    DoubleConv(int64_t in_channels, int64_t out_channels) {
        conv1_ = std::make_shared<Conv2d>(in_channels, out_channels, 3, 1, 1);
        bn1_ = std::make_shared<BatchNorm2d>(out_channels);
        conv2_ = std::make_shared<Conv2d>(out_channels, out_channels, 3, 1, 1);
        bn2_ = std::make_shared<BatchNorm2d>(out_channels);
        relu_ = std::make_shared<ReLU>();

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

class DownBlock : public Module {
public:
    DownBlock(int64_t in_channels, int64_t out_channels) {
        pool_ = std::make_shared<MaxPool2d>(2, 2);
        conv_block_ = std::make_shared<DoubleConv>(in_channels, out_channels);

        register_module("pool", pool_);
        register_module("conv_block", conv_block_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = pool_->forward(x);
        return conv_block_->forward(h);
    }

private:
    std::shared_ptr<MaxPool2d> pool_;
    std::shared_ptr<DoubleConv> conv_block_;
};

/**
 * @brief Upsampling block with ConvTranspose2d for U-Net decoder
 */
class UpBlock : public Module {
public:
    UpBlock(int64_t in_channels, int64_t out_channels) {
        // ConvTranspose2d for 2x upsampling
        up_conv_ = std::make_shared<ConvTranspose2d>(in_channels, out_channels, 4, 2, 1);
        bn_ = std::make_shared<BatchNorm2d>(out_channels);
        relu_ = std::make_shared<ReLU>();
        // After concatenation, channels double
        conv_block_ = std::make_shared<DoubleConv>(out_channels * 2, out_channels);

        register_module("up_conv", up_conv_);
        register_module("bn", bn_);
        register_module("relu", relu_);
        register_module("conv_block", conv_block_);
    }

    auto forward_with_skip(const Variable& x, const Variable& skip) -> Variable {
        // Upsample
        auto h = up_conv_->forward(x);
        h = relu_->forward(bn_->forward(h));

        // Concatenate with skip connection
        auto h_tensor = h.tensor();
        auto skip_tensor = skip.tensor();
        auto concat = tenzor::cat({h_tensor, skip_tensor}, 1);
        Variable concat_var(concat, h.requires_grad() || skip.requires_grad());

        return conv_block_->forward(concat_var);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Standalone forward (no skip connection)
        auto h = up_conv_->forward(x);
        h = relu_->forward(bn_->forward(h));
        return h;
    }

private:
    std::shared_ptr<ConvTranspose2d> up_conv_;
    std::shared_ptr<BatchNorm2d> bn_;
    std::shared_ptr<ReLU> relu_;
    std::shared_ptr<DoubleConv> conv_block_;
};

/**
 * @brief Full U-Net model with encoder-decoder and skip connections
 */
class UNet : public Module {
public:
    UNet(int64_t num_classes) : num_classes_(num_classes) {
        // Encoder
        enc1_ = std::make_shared<DoubleConv>(3, 64);
        enc2_ = std::make_shared<DownBlock>(64, 128);
        enc3_ = std::make_shared<DownBlock>(128, 256);

        // Bottleneck
        bottleneck_pool_ = std::make_shared<MaxPool2d>(2, 2);
        bottleneck_ = std::make_shared<DoubleConv>(256, 512);

        // Decoder with ConvTranspose2d
        dec3_ = std::make_shared<UpBlock>(512, 256);
        dec2_ = std::make_shared<UpBlock>(256, 128);
        dec1_ = std::make_shared<UpBlock>(128, 64);

        // Output
        out_conv_ = std::make_shared<Conv2d>(64, num_classes, 1, 1, 0);

        register_module("enc1", enc1_);
        register_module("enc2", enc2_);
        register_module("enc3", enc3_);
        register_module("bottleneck", bottleneck_);
        register_module("dec3", dec3_);
        register_module("dec2", dec2_);
        register_module("dec1", dec1_);
        register_module("out_conv", out_conv_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Encoder path
        auto e1 = enc1_->forward(x);      // 64 channels
        auto e2 = enc2_->forward(e1);     // 128 channels
        auto e3 = enc3_->forward(e2);     // 256 channels

        // Bottleneck
        auto b = bottleneck_pool_->forward(e3);
        b = bottleneck_->forward(b);      // 512 channels

        // Decoder path with skip connections
        auto d3 = dec3_->forward_with_skip(b, e3);   // 256 channels
        auto d2 = dec2_->forward_with_skip(d3, e2);  // 128 channels
        auto d1 = dec1_->forward_with_skip(d2, e1);  // 64 channels

        return out_conv_->forward(d1);
    }

private:
    int64_t num_classes_;
    std::shared_ptr<DoubleConv> enc1_, bottleneck_;
    std::shared_ptr<DownBlock> enc2_, enc3_;
    std::shared_ptr<MaxPool2d> bottleneck_pool_;
    std::shared_ptr<UpBlock> dec3_, dec2_, dec1_;
    std::shared_ptr<Conv2d> out_conv_;
};

// ============================================================================
// Feature Pyramid Network Style Model
// ============================================================================

class FeaturePyramidEncoder : public Module {
public:
    FeaturePyramidEncoder(int64_t num_classes) : num_classes_(num_classes) {
        enc1_ = std::make_shared<DoubleConv>(3, 64);
        enc2_ = std::make_shared<DownBlock>(64, 128);
        enc3_ = std::make_shared<DownBlock>(128, 256);
        enc4_ = std::make_shared<DownBlock>(256, 512);

        // Global average pooling and classifier
        pool_ = std::make_shared<AdaptiveAvgPool2d>(1, 1);
        fc_ = std::make_shared<Linear>(512, num_classes);

        register_module("enc1", enc1_);
        register_module("enc2", enc2_);
        register_module("enc3", enc3_);
        register_module("enc4", enc4_);
        register_module("pool", pool_);
        register_module("fc", fc_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto e1 = enc1_->forward(x);
        auto e2 = enc2_->forward(e1);
        auto e3 = enc3_->forward(e2);
        auto e4 = enc4_->forward(e3);

        auto pooled = pool_->forward(e4);
        auto pooled_tensor = pooled.tensor();
        int64_t batch_size = pooled_tensor.shape()[0];
        int64_t channels = pooled_tensor.shape()[1];
        auto flat = pooled_tensor.reshape({batch_size, channels});
        Variable flat_var(flat, pooled.requires_grad());

        return fc_->forward(flat_var);
    }

    // Get intermediate features for skip connections
    std::vector<Variable> get_features(const Variable& x) {
        auto e1 = enc1_->forward(x);
        auto e2 = enc2_->forward(e1);
        auto e3 = enc3_->forward(e2);
        auto e4 = enc4_->forward(e3);
        return {e1, e2, e3, e4};
    }

private:
    int64_t num_classes_;
    std::shared_ptr<DoubleConv> enc1_;
    std::shared_ptr<DownBlock> enc2_, enc3_, enc4_;
    std::shared_ptr<AdaptiveAvgPool2d> pool_;
    std::shared_ptr<Linear> fc_;
};

// ============================================================================
// Fully Convolutional Network (FCN) Style Segmentation
// ============================================================================

class FCNSegmentation : public Module {
public:
    FCNSegmentation(int64_t num_classes) : num_classes_(num_classes) {
        // Encoder
        enc1_ = std::make_shared<DoubleConv>(3, 64);
        enc2_ = std::make_shared<DownBlock>(64, 128);
        enc3_ = std::make_shared<DownBlock>(128, 256);

        // Bottleneck
        bottleneck_ = std::make_shared<DoubleConv>(256, 512);

        // 1x1 convolutions for classification at each scale
        score_pool4_ = std::make_shared<Conv2d>(512, num_classes, 1, 1, 0);
        score_pool3_ = std::make_shared<Conv2d>(256, num_classes, 1, 1, 0);

        register_module("enc1", enc1_);
        register_module("enc2", enc2_);
        register_module("enc3", enc3_);
        register_module("bottleneck", bottleneck_);
        register_module("score_pool4", score_pool4_);
        register_module("score_pool3", score_pool3_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // Encode
        auto e1 = enc1_->forward(x);       // 64x64x64
        auto e2 = enc2_->forward(e1);      // 128x32x32
        auto e3 = enc3_->forward(e2);      // 256x16x16

        // Bottleneck with MaxPool
        auto pool = std::make_shared<MaxPool2d>(2, 2);
        auto b = pool->forward(e3);
        b = bottleneck_->forward(b);       // 512x8x8

        // Get score from deepest features
        auto score = score_pool4_->forward(b);  // num_classes x 8x8

        return score;
    }

private:
    int64_t num_classes_;
    std::shared_ptr<DoubleConv> enc1_, bottleneck_;
    std::shared_ptr<DownBlock> enc2_, enc3_;
    std::shared_ptr<Conv2d> score_pool4_, score_pool3_;
};

// ============================================================================
// Demo Functions
// ============================================================================

void demo_normalization() {
    std::cout << "\n=== Normalization Layers Demo ===\n\n" << std::flush;

    Device device = Device::cpu();

    auto input = randn({4, 64, 16, 16}, DType::Float32, device);
    Variable x(input, false);

    // BatchNorm2d
    auto bn = std::make_shared<BatchNorm2d>(64);
    auto bn_out = bn->forward(x);
    auto bn_cpu = bn_out.tensor().cpu();
    float bn_mean = tenzor::mean(bn_cpu).cpu().data<float>()[0];

    std::cout << "BatchNorm2d(64):\n";
    std::cout << "  Normalizes over (N, H, W) per channel\n";
    std::cout << "  Output mean: " << std::fixed << std::setprecision(4) << bn_mean << "\n\n";

    // GroupNorm
    auto gn = std::make_shared<GroupNorm>(8, 64);
    auto gn_out = gn->forward(x);
    auto gn_cpu = gn_out.tensor().cpu();
    float gn_mean = tenzor::mean(gn_cpu).cpu().data<float>()[0];

    std::cout << "GroupNorm(8, 64):\n";
    std::cout << "  Normalizes over groups of channels\n";
    std::cout << "  Output mean: " << gn_mean << "\n\n";

    std::cout << "Use cases:\n";
    std::cout << "  BatchNorm: Standard for most CV tasks\n";
    std::cout << "  GroupNorm: Small batch sizes, detection tasks\n";
}

void demo_unet_architecture() {
    std::cout << "\n=== U-Net Architecture ===\n\n";

    std::cout << "U-Net Structure:\n\n";
    std::cout << "  Encoder (contracting path):\n";
    std::cout << "    - Progressive downsampling with MaxPool2d\n";
    std::cout << "    - Feature channels increase: 64 -> 128 -> 256 -> 512\n\n";

    std::cout << "  Decoder (expanding path):\n";
    std::cout << "    - Upsampling with ConvTranspose2d or bilinear\n";
    std::cout << "    - Skip connections from encoder\n";
    std::cout << "    - Feature channels decrease: 512 -> 256 -> 128 -> 64\n\n";

    std::cout << "  Skip Connections:\n";
    std::cout << "    - Concatenate/add encoder features with decoder\n";
    std::cout << "    - Preserve fine-grained spatial information\n";
    std::cout << "    - Enable precise localization\n";
}

void demo_feature_pyramid() {
    std::cout << "\n=== Feature Pyramid Demo ===\n\n";

    Device device = Device::cpu();

    auto input = randn({1, 3, 64, 64}, DType::Float32, device);
    Variable x(input, false);

    auto encoder = std::make_shared<FeaturePyramidEncoder>(10);
    auto features = encoder->get_features(x);

    std::cout << "Multi-scale feature extraction:\n\n";
    std::cout << "  Input: [1, 3, 64, 64]\n\n";

    std::vector<std::string> names = {"enc1", "enc2", "enc3", "enc4"};
    for (size_t i = 0; i < features.size(); ++i) {
        auto shape = features[i].shape();
        std::cout << "  " << names[i] << ": [" << shape[0] << ", " << shape[1]
                  << ", " << shape[2] << ", " << shape[3] << "]\n";
    }

    std::cout << "\n  Skip connections allow combining:\n";
    std::cout << "    - High-level semantics from deep layers\n";
    std::cout << "    - Fine details from shallow layers\n";
}

void demo_segmentation_output() {
    std::cout << "\n=== Segmentation Output Format ===\n\n";

    int batch_size = 2;
    int num_classes = 5;
    int height = 64;
    int width = 64;

    std::cout << "Segmentation Model Output:\n";
    std::cout << "  Shape: [batch, num_classes, height, width]\n";
    std::cout << "  Example: [" << batch_size << ", " << num_classes << ", "
              << height << ", " << width << "]\n\n";

    std::cout << "Processing:\n";
    std::cout << "  1. Apply softmax over class dimension (dim=1)\n";
    std::cout << "  2. Take argmax for class predictions\n";
    std::cout << "  3. Result: [batch, height, width] class indices\n\n";

    std::cout << "Loss computation:\n";
    std::cout << "  1. Reshape logits: [batch*H*W, num_classes]\n";
    std::cout << "  2. Reshape targets: [batch*H*W]\n";
    std::cout << "  3. Apply CrossEntropyLoss\n";
}

// ============================================================================
// Training
// ============================================================================

void train_encoder(Device device) {
    std::cout << "\n=== Training Feature Pyramid Encoder ===\n\n" << std::flush;

    int num_classes = 10;
    int img_size = 32;  // Reduced for faster training
    int batch_size = 4;
    int num_train = 20;  // Reduced for demo
    int num_epochs = 3;

    auto model = std::make_shared<FeaturePyramidEncoder>(num_classes);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    CrossEntropyLoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: Feature Pyramid Encoder\n";
    std::cout << "  Input: " << img_size << "x" << img_size << " RGB images\n";
    std::cout << "  Output: " << num_classes << " classes\n";
    std::cout << "  Loss: CrossEntropyLoss\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> label_dist(0, num_classes - 1);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            auto images = randn({actual_batch, 3, img_size, img_size}, DType::Float32, device);
            Variable x(images, true);

            std::vector<int64_t> label_data(actual_batch);
            for (int j = 0; j < actual_batch; ++j) {
                label_data[j] = label_dist(rng);
            }
            auto labels = from_data(label_data.data(), {actual_batch}, device);

            optimizer.zero_grad();

            auto output = model->forward(x);
            auto loss = criterion(output, labels);

            loss.backward();
            optimizer.step();

            auto loss_cpu = loss.tensor().cpu();
            epoch_loss += loss_cpu.data<float>()[0];
            num_batches++;
        }

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches) << "\n";
    }
}

void train_fcn_segmentation(Device device) {
    std::cout << "\n=== Training FCN Segmentation ===\n\n" << std::flush;

    int num_classes = 5;
    int img_size = 32;  // Reduced for faster training
    int batch_size = 2;
    int num_train = 10;  // Reduced for demo
    int num_epochs = 2;

    auto model = std::make_shared<FCNSegmentation>(num_classes);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    CrossEntropyLoss criterion;

    // FCN output size depends on input size and downsampling
    // 3 downsampling layers (2x each) + 1 MaxPool = input / 16
    int out_size = img_size / 8;  // For 32x32 input, output is 4x4

    std::cout << "Configuration:\n";
    std::cout << "  Model: FCN Segmentation\n";
    std::cout << "  Input: " << img_size << "x" << img_size << " RGB images\n";
    std::cout << "  Output: " << num_classes << "-class segmentation (" << out_size << "x" << out_size << ")\n";
    std::cout << "  Loss: CrossEntropyLoss (pixel-wise)\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> label_dist(0, num_classes - 1);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            auto images = randn({actual_batch, 3, img_size, img_size}, DType::Float32, device);
            Variable x(images, true);
            std::vector<int64_t> label_data(actual_batch * out_size * out_size);
            for (auto& l : label_data) {
                l = label_dist(rng);
            }
            auto labels = from_data(label_data.data(), {actual_batch, out_size, out_size}, device);

            optimizer.zero_grad();

            auto output = model->forward(x);

            // Reshape for cross entropy
            int64_t N = actual_batch;
            int64_t H = out_size;
            int64_t W = out_size;
            auto logits_flat = output.tensor().reshape({N * H * W, num_classes});
            auto labels_flat = labels.reshape({N * H * W});

            Variable logits_var(logits_flat, true);
            auto loss = criterion(logits_var, labels_flat);

            loss.backward();
            optimizer.step();

            auto loss_cpu = loss.tensor().cpu();
            epoch_loss += loss_cpu.data<float>()[0];
            num_batches++;
        }

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches) << "\n";
    }

    model->eval();
    auto test_input = randn({1, 3, img_size, img_size}, DType::Float32, device);
    Variable test_x(test_input, false);
    auto test_output = model->forward(test_x);

    std::cout << "\nModel output shape: [" << test_output.shape()[0] << ", "
              << test_output.shape()[1] << ", "
              << test_output.shape()[2] << ", "
              << test_output.shape()[3] << "]\n";
}

void train_unet(Device device) {
    std::cout << "\n=== Training U-Net with ConvTranspose2d ===\n\n" << std::flush;

    int num_classes = 5;
    int img_size = 32;  // Reduced from 64 for faster training
    int batch_size = 1;
    int num_train = 5;  // Reduced for demo
    int num_epochs = 2;

    auto model = std::make_shared<UNet>(num_classes);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    CrossEntropyLoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: U-Net with ConvTranspose2d decoder\n";
    std::cout << "  Input: " << img_size << "x" << img_size << " RGB images\n";
    std::cout << "  Output: " << num_classes << "-class segmentation (full resolution)\n";
    std::cout << "  Decoder: ConvTranspose2d for 2x upsampling\n";
    std::cout << "  Loss: CrossEntropyLoss (pixel-wise)\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    std::mt19937 rng(42);
    std::uniform_int_distribution<int64_t> label_dist(0, num_classes - 1);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            auto images = randn({actual_batch, 3, img_size, img_size}, DType::Float32, device);
            Variable x(images, true);

            // Full resolution output
            std::vector<int64_t> label_data(actual_batch * img_size * img_size);
            for (auto& l : label_data) {
                l = label_dist(rng);
            }
            auto labels = from_data(label_data.data(), {actual_batch, img_size, img_size}, device);

            optimizer.zero_grad();

            auto output = model->forward(x);

            // Reshape for cross entropy: [N, C, H, W] -> [N*H*W, C]
            int64_t N = actual_batch;
            int64_t C = num_classes;
            int64_t H = img_size;
            int64_t W = img_size;
            auto out_tensor = output.tensor();
            auto out_permuted = out_tensor.permute({0, 2, 3, 1});
            auto out_reshaped = out_permuted.reshape({N * H * W, C});
            Variable out_flat(out_reshaped, output.requires_grad());

            auto labels_flat = labels.reshape({N * H * W});

            auto loss = criterion(out_flat, labels_flat);

            loss.backward();
            optimizer.step();

            auto loss_cpu = loss.tensor().cpu();
            epoch_loss += loss_cpu.data<float>()[0];
            num_batches++;
        }

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches) << "\n";
    }

    model->eval();
    auto test_input = randn({1, 3, img_size, img_size}, DType::Float32, device);
    Variable test_x(test_input, false);
    auto test_output = model->forward(test_x);

    std::cout << "\nU-Net output shape: [" << test_output.shape()[0] << ", "
              << test_output.shape()[1] << ", "
              << test_output.shape()[2] << ", "
              << test_output.shape()[3] << "]\n";
    std::cout << "ConvTranspose2d successfully restored full resolution!\n";
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
    std::cout << "   U-Net Semantic Segmentation - Components           \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n" << std::flush;

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: Conv2d, ConvTranspose2d, MaxPool2d, AdaptiveAvgPool2d, Linear\n";
    std::cout << "  Normalization: BatchNorm2d, GroupNorm\n";
    std::cout << "  Architecture: U-Net (encoder-decoder), Feature Pyramid, FCN\n";
    std::cout << "  Loss: CrossEntropyLoss (pixel-wise)\n";

    try {
        demo_normalization();
        demo_unet_architecture();
        demo_feature_pyramid();
        demo_segmentation_output();
        train_encoder(device);
        train_unet(device);
        train_fcn_segmentation(device);

        std::cout << "\n======================================================\n";
        std::cout << "   All segmentation examples completed successfully! \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    tenzor::finalize();
    return 0;
}
