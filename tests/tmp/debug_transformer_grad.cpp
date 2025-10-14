#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/transformer.hpp>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    tenzor::initialize();

    std::cout << "Testing Transformer gradient flow\n\n";

    // Create minimal transformer: 2 encoder layers, 2 decoder layers
    Transformer model(128, 4, 2, 2, 512, 0.0, "relu", true);

    // Create inputs
    Variable src(randn({2, 5, 128}), true);
    Variable tgt(randn({2, 3, 128}), true);

    std::cout << "src.requires_grad() = " << src.requires_grad() << std::endl;
    std::cout << "tgt.requires_grad() = " << tgt.requires_grad() << std::endl;

    // Forward
    std::cout << "\nCalling model.forward(src, tgt)...\n";
    Variable output = model.forward(src, tgt);
    std::cout << "Output shape: [" << output.shape()[0] << ", " << output.shape()[1] << ", " << output.shape()[2] << "]\n";

    // Loss
    Variable loss = mean(output);
    std::cout << "Loss value: " << loss.tensor().data<float>()[0] << std::endl;

    // Backward
    std::cout << "\nCalling loss.backward()...\n";
    loss.backward();
    std::cout << "Backward completed!\n";

    // Check gradients
    std::cout << "\nChecking gradients:\n";
    std::cout << "src.has_grad() = " << src.has_grad() << std::endl;
    std::cout << "tgt.has_grad() = " << tgt.has_grad() << std::endl;

    if (!src.has_grad() || !tgt.has_grad()) {
        std::cout << "\nFAIL! Gradients not flowing to inputs\n";
        return 1;
    }

    std::cout << "\nSUCCESS! Gradients flow to both src and tgt\n";
    return 0;
}
