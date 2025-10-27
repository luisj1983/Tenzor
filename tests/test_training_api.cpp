/**
 * @file test_training_api.cpp
 * @brief Test for high-level training API
 */

#include <tenzor/tenzor.hpp>
#include <tenzor/nn/training.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/optim/adam.hpp>
#include <iostream>
#include <vector>

using namespace tenzor;
using namespace tenzor::nn;

int main() {
    try {
        // Initialize Tenzor
        tenzor::initialize();

        std::cout << "=== Testing High-Level Training API ===" << std::endl;

        // 1. Create a simple model (2-layer network)
        auto model = std::make_shared<Sequential>();
        model->add_module(std::make_shared<Linear>(10, 20));
        model->add_module(std::make_shared<ReLU>());
        model->add_module(std::make_shared<Linear>(20, 5));

        std::cout << "✓ Created Sequential model with 2 layers" << std::endl;

        // 2. Create optimizer
        auto optimizer = std::make_shared<optim::Adam>(model->parameters(), 0.001);
        std::cout << "✓ Created Adam optimizer" << std::endl;

        // 3. Create loss function
        auto mse_loss = std::make_shared<MSELoss>();
        std::cout << "✓ Created MSELoss" << std::endl;

        // 4. Create NeuralNetwork wrapper with loss function lambda
        auto loss_fn = [mse_loss](const Variable& pred, const Variable& target) -> Variable {
            return (*mse_loss)(pred, target);
        };
        NeuralNetwork nn(model, optimizer, loss_fn);
        std::cout << "✓ Created NeuralNetwork wrapper" << std::endl;

        // 5. Test training step
        auto input_tensor = Tensor({2, 10}, DType::Float32, Device::cpu());
        auto target_tensor = Tensor({2, 5}, DType::Float32, Device::cpu());

        // Fill with some dummy data
        auto* input_data = input_tensor.data<float>();
        auto* target_data = target_tensor.data<float>();
        for (size_t i = 0; i < input_tensor.numel(); ++i) {
            input_data[i] = static_cast<float>(i) * 0.1f;
        }
        for (size_t i = 0; i < target_tensor.numel(); ++i) {
            target_data[i] = static_cast<float>(i) * 0.01f;
        }

        Variable input_var(input_tensor, false);
        Variable target_var(target_tensor, false);

        std::cout << "\n=== Testing train_step() ===" << std::endl;
        float loss = nn.train_step(input_var, target_var);
        std::cout << "✓ train_step() completed successfully" << std::endl;
        std::cout << "  Loss value: " << loss << std::endl;

        // 6. Test eval step
        std::cout << "\n=== Testing eval_step() ===" << std::endl;
        float eval_loss = nn.eval_step(input_var, target_var);
        std::cout << "✓ eval_step() completed successfully" << std::endl;
        std::cout << "  Eval loss value: " << eval_loss << std::endl;

        // 7. Test mode switching
        std::cout << "\n=== Testing mode switching ===" << std::endl;
        nn.train();
        std::cout << "✓ Set to training mode: " << (nn.is_training() ? "true" : "false") << std::endl;

        nn.eval();
        std::cout << "✓ Set to eval mode: " << (nn.is_training() ? "false" : "true") << std::endl;

        // 8. Test DataLoader
        std::cout << "\n=== Testing DataLoader ===" << std::endl;
        std::vector<std::pair<Tensor, Tensor>> data;
        for (int i = 0; i < 5; ++i) {
            auto inp = Tensor({2, 10}, DType::Float32, Device::cpu());
            auto tgt = Tensor({2, 5}, DType::Float32, Device::cpu());
            data.push_back({inp, tgt});
        }
        DataLoader loader(data, 2);
        std::cout << "✓ Created DataLoader with " << loader.size() << " batches" << std::endl;

        // Iterate through loader
        int batch_count = 0;
        for (auto [inputs, targets] : loader) {
            batch_count++;
        }
        std::cout << "✓ Iterated through " << batch_count << " batches" << std::endl;

        // 9. Test fit() method
        std::cout << "\n=== Testing fit() method ===" << std::endl;
        DataLoader train_loader(data, 2);

        // Create a simple callback
        auto callback = std::make_shared<ProgressCallback>();

        std::cout << "Training for 3 epochs..." << std::endl;
        nn.fit(train_loader, 3, nullptr, {callback});
        std::cout << "✓ fit() completed successfully" << std::endl;

        // 10. Test with validation
        std::cout << "\n=== Testing fit() with validation ===" << std::endl;
        DataLoader val_loader(data, 2);

        std::cout << "Training for 2 epochs with validation..." << std::endl;
        nn.fit(train_loader, 2, &val_loader, {callback});
        std::cout << "✓ fit() with validation completed successfully" << std::endl;

        std::cout << "\n=== All Tests Passed! ===" << std::endl;
        return 0;

    } catch (const std::exception& e) {
        std::cerr << "ERROR: " << e.what() << std::endl;
        return 1;
    }
}
