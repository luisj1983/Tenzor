/**
 * @file gradient_checkpointing.cpp
 * @brief Gradient Checkpointing Concepts Demo
 *
 * This example demonstrates:
 * - Memory-efficient training concepts
 * - Deep residual network training
 * - BatchNorm2d normalization
 * - ReLU activation
 * - Adam optimizer
 * - MSELoss for regression
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>
#include <memory>
#include <chrono>

#include "tenzor/tenzor.hpp"
// KK.27: the autograd training loop lives in gradient_checkpointing_runner.{cpp,hpp}
// so the regression test in tests/examples/test_all_autograd_examples.cpp can
// drive the same code path. Include the runner header so the standalone exe's
// main() can invoke run_gradient_checkpointing_training() at the end.
#include "gradient_checkpointing_runner.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Deep Residual Block
// ============================================================================

class ResBlock : public Module {
public:
    ResBlock(int64_t channels) {
        conv1_ = std::make_shared<Conv2d>(channels, channels, 3, 1, 1);
        bn1_ = std::make_shared<BatchNorm2d>(channels);
        conv2_ = std::make_shared<Conv2d>(channels, channels, 3, 1, 1);
        bn2_ = std::make_shared<BatchNorm2d>(channels);
        relu_ = std::make_shared<ReLU>();

        register_module("conv1", conv1_);
        register_module("bn1", bn1_);
        register_module("conv2", conv2_);
        register_module("bn2", bn2_);
        register_module("relu", relu_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        auto h = relu_->forward(bn1_->forward(conv1_->forward(x)));
        h = bn2_->forward(conv2_->forward(h));
        return relu_->forward(x + h);  // Residual connection
    }

private:
    std::shared_ptr<Conv2d> conv1_, conv2_;
    std::shared_ptr<BatchNorm2d> bn1_, bn2_;
    std::shared_ptr<ReLU> relu_;
};

// ============================================================================
// Deep Network Model
// ============================================================================

class DeepResNet : public Module {
public:
    DeepResNet(int64_t num_blocks, int64_t channels = 64) : num_blocks_(num_blocks) {
        stem_ = std::make_shared<Conv2d>(3, channels, 7, 2, 3);
        stem_bn_ = std::make_shared<BatchNorm2d>(channels);
        stem_relu_ = std::make_shared<ReLU>();

        for (int64_t i = 0; i < num_blocks; ++i) {
            auto block = std::make_shared<ResBlock>(channels);
            blocks_.push_back(block);
            register_module("block_" + std::to_string(i), block);
        }

        pool_ = std::make_shared<AdaptiveAvgPool2d>(1, 1);
        fc_ = std::make_shared<Linear>(channels, 10);

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

    int64_t num_blocks() const { return num_blocks_; }

private:
    int64_t num_blocks_;
    std::shared_ptr<Conv2d> stem_;
    std::shared_ptr<BatchNorm2d> stem_bn_;
    std::shared_ptr<ReLU> stem_relu_;
    std::vector<std::shared_ptr<ResBlock>> blocks_;
    std::shared_ptr<AdaptiveAvgPool2d> pool_;
    std::shared_ptr<Linear> fc_;
};

// ============================================================================
// Demo Functions
// ============================================================================

void demo_checkpoint_concept() {
    std::cout << "\n=== Gradient Checkpointing Concept ===\n\n";

    std::cout << "The Memory Problem:\n";
    std::cout << "  During forward pass, activations are saved for backward:\n\n";
    std::cout << "  Input -> [L1] -> a1 -> [L2] -> a2 -> ... -> Output\n";
    std::cout << "           (save)        (save)\n\n";
    std::cout << "  Memory grows linearly with depth: O(n) for n layers\n";
    std::cout << "  Deep networks (100+ layers) can exhaust GPU memory\n\n";

    std::cout << "Checkpointing Solution:\n";
    std::cout << "  Only save activations at checkpoint boundaries:\n\n";
    std::cout << "  Input -> [Seg1] -> c1 -> [Seg2] -> c2 -> ... -> Output\n";
    std::cout << "           (ckpt)          (ckpt)\n\n";
    std::cout << "  During backward: recompute activations within segments\n";
    std::cout << "  Memory: O(sqrt(n)) with proper segment sizing\n\n";

    std::cout << "Trade-off:\n";
    std::cout << "  Memory:  Reduced by factor of segment_size\n";
    std::cout << "  Compute: ~33%% overhead (one extra forward per segment)\n";
    std::cout << "  Benefit: Train larger models or larger batch sizes\n";
}

void demo_when_to_use() {
    std::cout << "\n=== When to Use Checkpointing ===\n\n";

    std::cout << "Use checkpointing when:\n";
    std::cout << "  - Training very deep networks (50+ layers)\n";
    std::cout << "  - Training large transformers (BERT, GPT)\n";
    std::cout << "  - Getting OOM errors with smaller batch sizes\n";
    std::cout << "  - Combining with mixed precision for max efficiency\n\n";

    std::cout << "Implementation strategies:\n";
    std::cout << "  1. Per-layer: Checkpoint every layer (max memory savings)\n";
    std::cout << "  2. Segmented: Checkpoint every N layers (balanced)\n";
    std::cout << "  3. Selective: Checkpoint only memory-heavy layers\n\n";

    std::cout << "Optimal segment size:\n";
    std::cout << "  Rule of thumb: sqrt(num_layers)\n";
    std::cout << "  Example: 64 layers -> segment_size ~8\n";
}

void demo_deep_network_stats() {
    std::cout << "\n=== Deep Network Memory Analysis ===\n\n";

    // Calculate theoretical memory for different depths
    std::vector<int> depths = {8, 16, 32, 64};
    int batch_size = 16;
    int img_size = 64;
    int channels = 64;

    std::cout << "Theoretical activation memory (batch=" << batch_size
              << ", img=" << img_size << "x" << img_size << "):\n\n";

    for (int depth : depths) {
        // Each layer stores: batch * channels * h * w * sizeof(float)
        size_t per_layer = batch_size * channels * img_size * img_size * sizeof(float);
        size_t total_mb = (depth * per_layer) / (1024 * 1024);
        size_t checkpointed_mb = static_cast<size_t>(std::sqrt(depth) * per_layer) / (1024 * 1024);

        std::cout << "  " << std::setw(2) << depth << " layers: ";
        std::cout << std::setw(4) << total_mb << " MB (standard) vs ";
        std::cout << std::setw(4) << checkpointed_mb << " MB (checkpointed)\n";
    }

    std::cout << "\n  Checkpointing can enable 4-8x deeper networks!\n";
}

// ============================================================================
// Training
// ============================================================================

void train_deep_network(Device device) {
    std::cout << "\n=== Training Deep ResNet ===\n\n";

    int64_t num_blocks = 8;
    int64_t channels = 64;
    int img_size = 64;
    int batch_size = 8;
    int num_train = 100;
    int num_epochs = 3;

    auto model = std::make_shared<DeepResNet>(num_blocks, channels);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::Adam optimizer(params, 0.001f);

    MSELoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: DeepResNet with " << num_blocks << " residual blocks\n";
    std::cout << "  Input: " << img_size << "x" << img_size << " RGB images\n";
    std::cout << "  Channels: " << channels << "\n";
    std::cout << "  Loss: MSELoss (regression)\n";
    std::cout << "  Optimizer: Adam (lr=0.001)\n\n";

    std::mt19937 rng(42);
    std::normal_distribution<float> target_dist(0.0f, 1.0f);

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        float epoch_loss = 0.0f;
        int num_batches = 0;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);
            int actual_batch = end - i;

            auto images = randn({actual_batch, 3, img_size, img_size}, DType::Float32, device);
            Variable x(images, true);

            // Generate random targets
            std::vector<float> target_data(actual_batch * 10);
            for (auto& t : target_data) {
                t = target_dist(rng);
            }
            auto targets = from_data(target_data.data(), {actual_batch, 10}, device);
            Variable target_var(targets, false);

            optimizer.zero_grad();

            auto output = model->forward(x);
            auto loss = criterion(output, target_var);

            loss.backward();
            optimizer.step();

            auto loss_cpu = loss.tensor().cpu();
            epoch_loss += loss_cpu.data<float>()[0];
            num_batches++;
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(epoch_end - epoch_start);

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches)
                  << " | Time: " << duration.count() << " ms\n";
    }
}

void demo_training_comparison(Device device) {
    std::cout << "\n=== Training Time Comparison ===\n\n";

    std::vector<int64_t> depths = {4, 8, 16};
    int img_size = 32;
    int batch_size = 4;
    int num_iterations = 5;

    for (int64_t depth : depths) {
        auto model = std::make_shared<DeepResNet>(depth, 32);
        model->to(device);
        model->train();

        auto params = model->parameters();
        optim::Adam optimizer(params, 0.001f);
        MSELoss criterion;

        // Warmup
        auto images = randn({batch_size, 3, img_size, img_size}, DType::Float32, device);
        auto targets = randn({batch_size, 10}, DType::Float32, device);

        for (int i = 0; i < 2; ++i) {
            Variable x(images, true);
            Variable t(targets, false);
            optimizer.zero_grad();
            auto out = model->forward(x);
            auto loss = criterion(out, t);
            loss.backward();
            optimizer.step();
        }

        // Timed iterations
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < num_iterations; ++i) {
            Variable x(images, true);
            Variable t(targets, false);
            optimizer.zero_grad();
            auto out = model->forward(x);
            auto loss = criterion(out, t);
            loss.backward();
            optimizer.step();
        }
        auto end = std::chrono::high_resolution_clock::now();

        auto avg_time = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count() / num_iterations;

        std::cout << "  " << std::setw(2) << depth << " blocks: "
                  << std::setw(4) << avg_time << " ms/iter\n";
    }

    std::cout << "\n  Training time scales approximately linearly with depth.\n";
    std::cout << "  Checkpointing adds ~33%% overhead but enables 4x more depth.\n";
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
    std::cout << "   Gradient Checkpointing - Memory Efficient Training \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: Conv2d, BatchNorm2d, AdaptiveAvgPool2d, Linear\n";
    std::cout << "  Activations: ReLU\n";
    std::cout << "  Architecture: Deep ResNet with residual blocks\n";
    std::cout << "  Loss: MSELoss\n";
    std::cout << "  Optimizer: Adam\n";
    std::cout << "  Concepts: Checkpointing memory trade-offs\n";

    try {
        demo_checkpoint_concept();
        demo_when_to_use();
        demo_deep_network_stats();
        train_deep_network(device);
        demo_training_comparison(device);

        // KK.27: also exercise the regression-tested runner so this
        // standalone exe and the test target stay in lock-step. Small
        // iteration count keeps the runtime negligible.
        double init_loss = 0.0, final_loss = 0.0;
        auto rc = tenzor::examples::gradient_checkpointing::
            run_gradient_checkpointing_training(
                /*num_iterations=*/4, &init_loss, &final_loss, device,
                /*verbose=*/false);
        std::cout << "\nRunner smoke test: initial=" << init_loss
                  << " final=" << final_loss << " rc=" << rc << "\n";

        std::cout << "\n======================================================\n";
        std::cout << "   All checkpointing examples completed successfully! \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    tenzor::finalize();
    return 0;
}
