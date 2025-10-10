#include <tenzor/tenzor.hpp>
#include <iostream>

int main() {
    using namespace tenzor;

    // Initialize Tenzor library
    initialize();

    std::cout << "\nTenzor MNIST Example\n";
    std::cout << "====================\n\n";

    // Create model
    auto model = nn::Sequential(
        std::make_shared<nn::Linear>(784, 128),
        std::make_shared<nn::ReLU>(),
        std::make_shared<nn::Dropout>(0.2),
        std::make_shared<nn::Linear>(128, 10)
    );

    std::cout << "Created model with:\n";
    std::cout << "  - Linear (784 -> 128)\n";
    std::cout << "  - ReLU activation\n";
    std::cout << "  - Dropout (p=0.2)\n";
    std::cout << "  - Linear (128 -> 10)\n\n";

    // Create optimizer
    auto optimizer = optim::Adam(model.parameters(), 1e-3);

    std::cout << "Created Adam optimizer with lr=0.001\n";

    // Simulate training
    const int epochs = 5;
    const int batch_size = 32;

    std::cout << "\nSimulating training...\n";
    for (int epoch = 0; epoch < epochs; ++epoch) {
        // Forward pass
        auto input = Variable(randn({batch_size, 784}), true);
        auto output = model.forward(input);

        std::cout << "Epoch " << (epoch + 1) << "/" << epochs
                  << " - Output shape: "
                  << output.shape()[0] << "x" << output.shape()[1] << "\n";
    }

    std::cout << "\nTraining simulation complete!\n";

    return 0;
}
