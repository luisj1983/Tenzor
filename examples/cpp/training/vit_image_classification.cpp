/**
 * @file vit_image_classification.cpp
 * @brief Vision Transformer Components Demo for Image Classification
 *
 * This example demonstrates ViT-related components:
 * - MultiheadAttention mechanism
 * - LayerNorm and GELU activation
 * - AdaptiveAvgPool2d
 * - AdamW optimizer with weight decay
 * - OneCycleLR and CosineAnnealingWarmRestarts schedulers
 * - CrossEntropyLoss for classification
 */

#include <iostream>
#include <iomanip>
#include <vector>
#include <cmath>
#include <random>

#include "tenzor/tenzor.hpp"
// KK.27: a trimmed-down version of the training loop lives in
// vit_image_classification_runner.{cpp,hpp} so the regression test in
// tests/examples/test_all_autograd_examples.cpp can drive the same
// MultiheadAttention + LayerNorm + GELU + AdamW pipeline.
#include "vit_image_classification_runner.hpp"

using namespace tenzor;
using namespace tenzor::nn;

// ============================================================================
// Simple Transformer Block for Classification
// ============================================================================

class TransformerClassifier : public Module {
public:
    TransformerClassifier(int64_t input_dim, int64_t hidden_dim, int64_t num_classes,
                          int64_t num_heads = 4) {
        // Input projection
        fc_in_ = std::make_shared<Linear>(input_dim, hidden_dim);

        // Multihead attention
        attention_ = std::make_shared<MultiheadAttention>(hidden_dim, num_heads);

        // Layer norms
        norm1_ = std::make_shared<LayerNorm>(std::vector<int64_t>{hidden_dim});
        norm2_ = std::make_shared<LayerNorm>(std::vector<int64_t>{hidden_dim});

        // GELU activation
        gelu_ = std::make_shared<GELU>();

        // Feed-forward
        fc1_ = std::make_shared<Linear>(hidden_dim, hidden_dim * 4);
        fc2_ = std::make_shared<Linear>(hidden_dim * 4, hidden_dim);

        // Classification head
        fc_out_ = std::make_shared<Linear>(hidden_dim, num_classes);

        // Dropout
        dropout_ = std::make_shared<Dropout>(0.1f);

        register_module("fc_in", fc_in_);
        register_module("attention", attention_);
        register_module("norm1", norm1_);
        register_module("norm2", norm2_);
        register_module("gelu", gelu_);
        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
        register_module("fc_out", fc_out_);
        register_module("dropout", dropout_);
    }

    auto forward_impl(const Variable& x) -> Variable override {
        // x: [batch, input_dim]
        auto h = fc_in_->forward(x);  // [batch, hidden_dim]

        // Add sequence dimension for attention: [batch, 1, hidden_dim]
        auto h_tensor = h.tensor();
        auto batch_size = h_tensor.shape()[0];
        auto hidden_dim = h_tensor.shape()[1];

        auto h_3d = h_tensor.reshape({batch_size, 1, hidden_dim});
        Variable h_seq(h_3d, h.requires_grad());

        // Self-attention with residual (forward returns pair<output, weights>)
        auto [attn_out, _weights] = attention_->forward(h_seq, h_seq, h_seq);
        attn_out = dropout_->forward(attn_out);

        // Squeeze back: [batch, hidden_dim]
        auto attn_tensor = attn_out.tensor().reshape({batch_size, hidden_dim});
        Variable attn_2d(attn_tensor, attn_out.requires_grad());

        auto h_res = h + attn_2d;
        h_res = norm1_->forward(h_res);

        // Feed-forward with residual
        auto ff = fc1_->forward(h_res);
        ff = gelu_->forward(ff);
        ff = dropout_->forward(ff);
        ff = fc2_->forward(ff);

        auto out = h_res + ff;
        out = norm2_->forward(out);

        // Classification
        return fc_out_->forward(out);
    }

private:
    std::shared_ptr<Linear> fc_in_;
    std::shared_ptr<MultiheadAttention> attention_;
    std::shared_ptr<LayerNorm> norm1_;
    std::shared_ptr<LayerNorm> norm2_;
    std::shared_ptr<GELU> gelu_;
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
    std::shared_ptr<Linear> fc_out_;
    std::shared_ptr<Dropout> dropout_;
};

// ============================================================================
// Data Generation
// ============================================================================

std::pair<Tensor, Tensor> generate_classification_data(int num_samples, int input_dim,
                                                        int num_classes, Device device) {
    std::vector<float> X_data(num_samples * input_dim);
    std::vector<int64_t> y_data(num_samples);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    std::uniform_int_distribution<int> class_dist(0, num_classes - 1);

    for (int i = 0; i < num_samples; ++i) {
        int label = class_dist(rng);
        y_data[i] = label;

        // Add class-specific bias to create learnable patterns
        float bias = static_cast<float>(label) * 0.5f;
        for (int j = 0; j < input_dim; ++j) {
            X_data[i * input_dim + j] = dist(rng) + bias * (j % 10 == label ? 1.0f : 0.0f);
        }
    }

    auto X = from_data(X_data.data(), {num_samples, input_dim}, device);
    auto y = from_data(y_data.data(), {num_samples}, device);

    return {X, y};
}

// ============================================================================
// Training Functions
// ============================================================================

void train_with_adamw_onecycle(Device device) {
    std::cout << "\n=== Training with AdamW + OneCycleLR ===\n\n";

    int num_classes = 10;
    int input_dim = 64;
    int hidden_dim = 128;
    int num_train = 500;
    int num_val = 100;
    int batch_size = 32;
    int num_epochs = 20;

    auto [X_train, y_train] = generate_classification_data(num_train, input_dim, num_classes, device);
    auto [X_val, y_val] = generate_classification_data(num_val, input_dim, num_classes, device);

    auto model = std::make_shared<TransformerClassifier>(input_dim, hidden_dim, num_classes, 4);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::AdamW optimizer(params, 0.001f, 0.9f, 0.999f, 1e-8f, 0.01f);

    int steps_per_epoch = (num_train + batch_size - 1) / batch_size;
    int64_t total_steps = num_epochs * steps_per_epoch;
    optim::OneCycleLR scheduler(optimizer, 0.01, total_steps);

    CrossEntropyLoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Model: TransformerClassifier\n";
    std::cout << "  Components: MultiheadAttention, LayerNorm, GELU\n";
    std::cout << "  Optimizer: AdamW (lr=0.001, weight_decay=0.01)\n";
    std::cout << "  Scheduler: OneCycleLR (max_lr=0.01)\n";
    std::cout << "  Loss: CrossEntropyLoss\n";
    std::cout << "  Batch size: " << batch_size << "\n";
    std::cout << "  Epochs: " << num_epochs << "\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);

            auto batch_X = X_train.slice(0, i, end);
            auto batch_y = y_train.slice(0, i, end);

            optimizer.zero_grad();

            Variable input(batch_X, true);
            auto output = model->forward(input);

            auto loss = criterion(output, batch_y);

            loss.backward();
            optimizer.step();
            scheduler.step();

            epoch_loss += loss.tensor().item<float>();
            num_batches++;
        }

        // Validation
        model->eval();
        int correct = 0;
        int total = 0;

        for (int i = 0; i < num_val; i += batch_size) {
            int end = std::min(i + batch_size, num_val);
            auto batch_X = X_val.slice(0, i, end);
            auto batch_y = y_val.slice(0, i, end);

            Variable input(batch_X, false);
            auto output = model->forward(input);

            auto preds = tenzor::argmax(output.tensor(), 1);
            auto preds_cpu = preds.cpu();
            auto labels_cpu = batch_y.cpu();

            const int64_t* pred_data = preds_cpu.data<int64_t>();
            const int64_t* label_data = labels_cpu.data<int64_t>();

            for (int64_t j = 0; j < preds_cpu.shape()[0]; ++j) {
                if (pred_data[j] == label_data[j]) correct++;
                total++;
            }
        }

        float val_acc = 100.0f * correct / total;

        if ((epoch + 1) % 5 == 0 || epoch == 0) {
            std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                      << " | Loss: " << std::fixed << std::setprecision(4)
                      << (epoch_loss / num_batches)
                      << " | Val Acc: " << std::setprecision(2) << val_acc << "%"
                      << " | LR: " << std::setprecision(6) << scheduler.get_last_lr()
                      << "\n";
        }
    }
}

void train_with_cosine_warm_restarts(Device device) {
    std::cout << "\n=== Training with CosineAnnealingWarmRestarts ===\n\n";

    int num_classes = 10;
    int input_dim = 64;
    int hidden_dim = 64;
    int num_train = 400;
    int batch_size = 32;
    int num_epochs = 15;

    auto [X_train, y_train] = generate_classification_data(num_train, input_dim, num_classes, device);

    auto model = std::make_shared<TransformerClassifier>(input_dim, hidden_dim, num_classes, 2);
    model->to(device);
    model->train();

    auto params = model->parameters();
    optim::AdamW optimizer(params, 0.0001f, 0.9f, 0.999f, 1e-8f, 0.05f);

    optim::CosineAnnealingWarmRestarts scheduler(optimizer, 5, 2, 1e-6);

    CrossEntropyLoss criterion;

    std::cout << "Configuration:\n";
    std::cout << "  Optimizer: AdamW (lr=0.0001, weight_decay=0.05)\n";
    std::cout << "  Scheduler: CosineAnnealingWarmRestarts (T_0=5, T_mult=2)\n\n";

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        model->train();
        float epoch_loss = 0.0f;
        int num_batches = 0;

        for (int i = 0; i < num_train; i += batch_size) {
            int end = std::min(i + batch_size, num_train);

            auto batch_X = X_train.slice(0, i, end);
            auto batch_y = y_train.slice(0, i, end);

            optimizer.zero_grad();

            Variable input(batch_X, true);
            auto output = model->forward(input);

            auto loss = criterion(output, batch_y);

            loss.backward();
            optimizer.step();

            epoch_loss += loss.tensor().item<float>();
            num_batches++;
        }

        scheduler.step();

        std::cout << "Epoch " << std::setw(2) << (epoch + 1) << "/" << num_epochs
                  << " | Loss: " << std::fixed << std::setprecision(4)
                  << (epoch_loss / num_batches)
                  << " | LR: " << std::setprecision(6) << scheduler.get_last_lr()
                  << "\n";
    }
}

void demo_adaptive_pooling(Device device) {
    std::cout << "\n=== AdaptiveAvgPool2d Demo ===\n\n";

    auto pool = std::make_shared<AdaptiveAvgPool2d>(7, 7);

    std::vector<std::pair<int, int>> input_sizes = {
        {32, 32}, {64, 64}, {128, 128}, {224, 224}
    };

    std::cout << "AdaptiveAvgPool2d with output_size=(7, 7):\n\n";

    for (auto [h, w] : input_sizes) {
        auto input_t = randn({2, 64, h, w}, DType::Float32, device);
        Variable input_var(input_t, false);

        auto output = pool->forward(input_var);

        std::cout << "  Input: [2, 64, " << h << ", " << w << "] -> Output: ["
                  << output.shape()[0] << ", "
                  << output.shape()[1] << ", "
                  << output.shape()[2] << ", "
                  << output.shape()[3] << "]\n";
    }

    std::cout << "\nAdaptiveAvgPool2d ensures consistent feature map size!\n";
}

void demo_gelu_and_layernorm(Device device) {
    std::cout << "\n=== GELU and LayerNorm Demo ===\n\n";

    int batch_size = 4;
    int hidden_size = 64;

    auto gelu = std::make_shared<GELU>();
    auto x = randn({batch_size, hidden_size}, DType::Float32, device);
    Variable x_var(x, true);

    auto gelu_out = gelu->forward(x_var);

    std::cout << "GELU activation:\n";
    std::cout << "  Input mean: " << tenzor::mean(x).item<float>() << "\n";
    std::cout << "  Output mean: " << tenzor::mean(gelu_out.tensor()).item<float>() << "\n";

    auto layer_norm = std::make_shared<LayerNorm>(std::vector<int64_t>{hidden_size});
    auto ln_out = layer_norm->forward(x_var);

    std::cout << "\nLayerNorm:\n";
    // Flatten tensors first to get global variance
    auto x_flat = x.reshape({-1});
    auto ln_flat = ln_out.tensor().reshape({-1});
    std::cout << "  Input std: " << std::sqrt(tenzor::var(x_flat).item<float>()) << "\n";
    std::cout << "  Output std: " << std::sqrt(tenzor::var(ln_flat).item<float>()) << "\n";
}

// ============================================================================
// Main
// ============================================================================

int main(int argc, char* argv[]) {
    initialize();

    Device device = Device::cpu();
    if (argc > 1) {
        std::string backend = argv[1];
        if (backend == "cuda") device = Device::cuda();
        else if (backend == "rocm") device = Device::rocm();
        else if (backend == "vulkan") device = Device::vulkan();
        else if (backend == "oneapi") device = Device::oneapi();
        else if (backend == "mps") device = Device::mps();
    }

    std::cout << "======================================================\n";
    std::cout << "   Transformer Classification - ViT Components        \n";
    std::cout << "   Backend: " << device.to_string() << "\n";
    std::cout << "======================================================\n";

    std::cout << "\nComponents demonstrated:\n";
    std::cout << "  Layers: MultiheadAttention, LayerNorm\n";
    std::cout << "  Activations: GELU\n";
    std::cout << "  Pooling: AdaptiveAvgPool2d\n";
    std::cout << "  Optimizers: AdamW\n";
    std::cout << "  Schedulers: OneCycleLR, CosineAnnealingWarmRestarts\n";
    std::cout << "  Losses: CrossEntropyLoss\n";

    try {
        train_with_adamw_onecycle(device);
        train_with_cosine_warm_restarts(device);
        demo_adaptive_pooling(device);
        demo_gelu_and_layernorm(device);

        // KK.27: exercise the regression-tested runner so the standalone
        // exe and the test target stay in lock-step.
        double init_loss = 0.0, final_loss = 0.0;
        auto rc = tenzor::examples::vit_image_classification::
            run_vit_classification_training(
                /*epochs=*/2, &init_loss, &final_loss, device,
                /*verbose=*/false);
        std::cout << "\nRunner smoke test: initial=" << init_loss
                  << " final=" << final_loss << " rc=" << rc << "\n";

        std::cout << "\n======================================================\n";
        std::cout << "   All transformer examples completed successfully!   \n";
        std::cout << "======================================================\n";

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }

    finalize();
    return 0;
}
