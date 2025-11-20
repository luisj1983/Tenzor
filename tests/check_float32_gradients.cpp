#include <iostream>
#include <cmath>
#include "tenzor/tenzor.hpp"
#include "../include/tenzor/models/swin_transformer.hpp"

using namespace tenzor;
using namespace tenzor::models;

int main() {
    initialize();
    auto device = Device::cpu();
    const int img_size = 224;

    std::cout << "Testing Float32 Swin Transformer gradient validity...\n\n";

    auto model = swin_tiny(10, img_size, false);
    model->to(DType::Float32);
    model->train();

    Variable input(Tensor({1, 3, img_size, img_size}, DType::Float32, device), true);
    Variable output = model->forward(input);
    Variable loss = tenzor::mean(output);
    loss.backward();

    if (!input.has_grad()) {
        std::cout << "ERROR: Input has no gradient!\n";
        return 1;
    }

    auto grad = input.grad().value();
    auto* data = grad.data<float>();
    int numel = grad.numel();

    int nan_count = 0;
    int inf_count = 0;
    int normal_count = 0;
    int zero_count = 0;

    float min_val = INFINITY;
    float max_val = -INFINITY;

    for (int i = 0; i < numel; i++) {
        float val = data[i];
        if (std::isnan(val)) {
            nan_count++;
        } else if (std::isinf(val)) {
            inf_count++;
        } else if (val == 0.0f) {
            zero_count++;
        } else {
            normal_count++;
            min_val = std::min(min_val, val);
            max_val = std::max(max_val, val);
        }
    }

    std::cout << "Gradient Analysis:\n";
    std::cout << "  Total elements: " << numel << "\n";
    std::cout << "  NaN count: " << nan_count << "\n";
    std::cout << "  Inf count: " << inf_count << "\n";
    std::cout << "  Zero count: " << zero_count << "\n";
    std::cout << "  Normal count: " << normal_count << "\n";

    if (normal_count > 0) {
        std::cout << "  Min normal value: " << min_val << "\n";
        std::cout << "  Max normal value: " << max_val << "\n";
    }

    std::cout << "\nFirst 20 gradient values:\n";
    for (int i = 0; i < std::min(20, numel); i++) {
        std::cout << "  [" << i << "] = " << data[i];
        if (std::isnan(data[i])) std::cout << " (NaN)";
        if (std::isinf(data[i])) std::cout << " (Inf)";
        std::cout << "\n";
    }

    return 0;
}
