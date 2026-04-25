/**
 * @file tensor_only.cpp
 * @brief Layer Normalization with raw tensors
 *
 * Implements LayerNorm from scratch on a 2D tensor:
 *
 *   y = (x - mean(x, -1)) / sqrt(var(x, -1) + eps) * gamma + beta
 *
 * Unlike BatchNorm, LayerNorm normalizes along the feature axis for each
 * sample independently, so it does not depend on batch statistics.
 *
 * Usage: ./19_layer_normalization_tensor_only --backend cpu|cuda|vulkan
 */

#include "../common.hpp"
#include <cmath>
#include <vector>

using namespace tenzor;

int main(int argc, char* argv[]) {
    Device device = showcase::get_device_from_args(argc, argv);
    initialize();
    showcase::print_header("Layer Normalization - Tensor Only", device);
    manual_seed(42);

    int batch = 4;
    int features = 8;

    // Input with wildly different magnitudes per row
    std::vector<float> X_data(batch * features);
    for (int b = 0; b < batch; ++b) {
        float scale = std::pow(10.0f, static_cast<float>(b));
        for (int f = 0; f < features; ++f) {
            X_data[b * features + f] = scale * (0.5f + 0.1f * f);
        }
    }
    auto X = from_data(X_data.data(), {batch, features}, device);

    showcase::print_section("Input (rows have wildly different magnitudes)");
    auto X_cpu = X.cpu();
    for (int b = 0; b < batch; ++b) {
        std::cout << "row " << b << ": [";
        for (int f = 0; f < features; ++f) {
            std::cout << X_cpu.data<float>()[b * features + f];
            if (f < features - 1) std::cout << ", ";
        }
        std::cout << "]\n";
    }

    // Per-row mean and variance (keepdim=true to broadcast)
    float eps = 1e-5f;
    auto mean_x = tenzor::mean(X, -1, true);                        // (batch, 1)
    auto x_centered = X - mean_x;
    auto var_x = tenzor::mean(x_centered * x_centered, -1, true);   // (batch, 1)
    auto x_norm = x_centered / tenzor::sqrt(var_x + eps);

    // Learnable affine (initialized to identity for the demo)
    auto gamma = ones({1, features}, DType::Float32, device);
    auto beta  = zeros({1, features}, DType::Float32, device);

    auto y = x_norm * gamma + beta;

    showcase::print_section("After LayerNorm (each row now zero-mean, unit-variance)");
    auto y_cpu = y.cpu();
    for (int b = 0; b < batch; ++b) {
        // Report per-row stats
        float m = 0, v = 0;
        for (int f = 0; f < features; ++f) {
            m += y_cpu.data<float>()[b * features + f];
        }
        m /= features;
        for (int f = 0; f < features; ++f) {
            float d = y_cpu.data<float>()[b * features + f] - m;
            v += d * d;
        }
        v /= features;
        std::cout << "row " << b << ": mean=" << m << ", var=" << v << "\n";
    }

    std::cout << "\nLayer normalization demonstrated with raw tensors!\n";
    std::cout << "Unlike BatchNorm, LayerNorm stats are computed per-sample, so it\n";
    std::cout << "works identically at train and eval time and for batch size 1.\n";

    finalize();
    return 0;
}
