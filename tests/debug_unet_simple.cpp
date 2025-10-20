#include "tenzor/tenzor.hpp"
#include "tenzor/models/unet.hpp"
#include <iostream>

using namespace tenzor;
using namespace tenzor::models;

int main() {
    std::cout << "Testing UNet gradient flow with minimal setup..." << std::endl;

    try {
        // Create UNet model (bilinear=false means it uses ConvTranspose2d)
        auto model = std::make_shared<UNet>(3, 2, false);
        model->train();

        std::cout << "Model created successfully" << std::endl;

        // Small input for faster testing
        Variable images(Tensor({1, 3, 64, 64}, DType::Float32, Device::cpu()), true);
        std::cout << "Input created: requires_grad=" << images.requires_grad()
                  << ", is_leaf=" << images.is_leaf() << std::endl;

        // Forward pass
        std::cout << "Running forward pass..." << std::endl;
        Variable output = model->forward(images);
        std::cout << "Forward complete. Output shape: [" << output.shape()[0]
                  << ", " << output.shape()[1] << ", " << output.shape()[2]
                  << ", " << output.shape()[3] << "]" << std::endl;
        std::cout << "Output requires_grad=" << output.requires_grad()
                  << ", has_grad_fn=" << (output.grad_fn() != nullptr) << std::endl;

        // Create loss
        Variable loss = tenzor::sum(output);
        std::cout << "Loss created: value has shape [";
        for (auto s : loss.shape()) std::cout << s << " ";
        std::cout << "], has_grad_fn=" << (loss.grad_fn() != nullptr) << std::endl;

        // Backward pass
        std::cout << "\nCalling backward()..." << std::endl;
        loss.backward();

        std::cout << "\nBackward complete!" << std::endl;
        std::cout << "images.grad().has_value() = " << images.grad().has_value() << std::endl;

        if (images.grad().has_value()) {
            std::cout << "✅ SUCCESS: Gradients computed!" << std::endl;
            return 0;
        } else {
            std::cout << "❌ FAIL: No gradients!" << std::endl;
            return 1;
        }

    } catch (const std::exception& e) {
        std::cout << "❌ Exception: " << e.what() << std::endl;
        return 1;
    }
}
