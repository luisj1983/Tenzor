#include "tenzor/tenzor.hpp"
#include "tenzor/nn/layers/linear.hpp"
#include "tenzor/nn/module.hpp"
#include "tenzor/nn/optim/adam.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/reduction.hpp"
#include <iostream>
#include <memory>

using namespace tenzor;
using namespace tenzor::nn;

// Simple neural network class
class SimpleNetwork : public Module {
public:
    SimpleNetwork(int64_t input_size, int64_t hidden_size, int64_t output_size) {
        fc1_ = std::make_shared<Linear>(input_size, hidden_size);
        fc2_ = std::make_shared<Linear>(hidden_size, output_size);

        register_module("fc1", fc1_);
        register_module("fc2", fc2_);
    }

    auto forward_impl(const Variable& input) -> Variable override {
        auto h = fc1_->forward(input);
        // Apply ReLU manually (if activation not available)
        // h = relu(h);
        return fc2_->forward(h);
    }

private:
    std::shared_ptr<Linear> fc1_;
    std::shared_ptr<Linear> fc2_;
};

int main() {
    // Initialize Tenzor library
    initialize();

    std::cout << "Tenzor Serialization Example\n";
    std::cout << "============================\n\n";

    // Create model
    auto model = std::make_shared<SimpleNetwork>(784, 128, 10);
    std::cout << "Created SimpleNetwork(784 -> 128 -> 10)\n";

    // Get initial parameters
    auto params = model->parameters();
    std::cout << "Model has " << params.size() << " parameters\n\n";

    // Initialize weights to specific values for demonstration
    params[0]->tensor().fill_(0.1f);  // fc1.weight
    params[1]->tensor().fill_(0.01f); // fc1.bias
    params[2]->tensor().fill_(0.2f);  // fc2.weight
    params[3]->tensor().fill_(0.02f); // fc2.bias

    std::cout << "Initialized weights:\n";
    std::cout << "  fc1.weight: 0.1\n";
    std::cout << "  fc1.bias:   0.01\n";
    std::cout << "  fc2.weight: 0.2\n";
    std::cout << "  fc2.bias:   0.02\n\n";

    // Perform a forward pass
    auto input = ones({1, 784}, DType::Float32);
    auto output1 = model->forward(Variable(input));
    std::cout << "Forward pass output (first element): "
              << output1.tensor().data<float>()[0] << "\n\n";

    // Save model
    std::string model_path = "/tmp/model.bin";
    model->save(model_path);
    std::cout << "Saved model to " << model_path << "\n\n";

    // Create optimizer
    optim::Adam optimizer(params, 0.001);
    std::cout << "Created Adam optimizer (lr=0.001)\n";

    // Perform a training step to initialize momentum
    auto loss = sum(output1.tensor());
    auto loss_var = Variable(loss);
    loss_var.backward();
    optimizer.step();
    std::cout << "Performed one training step\n";

    // Save optimizer state
    std::string optim_path = "/tmp/optimizer.bin";
    optimizer.save_state(optim_path);
    std::cout << "Saved optimizer state to " << optim_path << "\n\n";

    // Create new model and load weights
    auto model2 = std::make_shared<SimpleNetwork>(784, 128, 10);
    std::cout << "Created new model instance\n";

    model2->load(model_path);
    std::cout << "Loaded weights from " << model_path << "\n";

    // Verify loaded weights
    auto params2 = model2->parameters();
    std::cout << "Loaded weights:\n";
    std::cout << "  fc1.weight: " << params2[0]->tensor().data<float>()[0] << "\n";
    std::cout << "  fc1.bias:   " << params2[1]->tensor().data<float>()[0] << "\n";
    std::cout << "  fc2.weight: " << params2[2]->tensor().data<float>()[0] << "\n";
    std::cout << "  fc2.bias:   " << params2[3]->tensor().data<float>()[0] << "\n\n";

    // Perform forward pass with loaded model
    auto output2 = model2->forward(Variable(input));
    std::cout << "Forward pass output after loading (first element): "
              << output2.tensor().data<float>()[0] << "\n";

    // Verify outputs match
    float diff = std::abs(output1.tensor().data<float>()[0] -
                          output2.tensor().data<float>()[0]);
    std::cout << "Difference between outputs: " << diff << "\n";

    if (diff < 1e-6f) {
        std::cout << "✓ Outputs match! Serialization successful.\n\n";
    } else {
        std::cout << "✗ Outputs don't match.\n\n";
        return 1;
    }

    // Load optimizer state
    auto params3 = model2->parameters();
    optim::Adam optimizer2(params3, 0.0);  // Different learning rate
    optimizer2.load_state(optim_path);
    std::cout << "Loaded optimizer state from " << optim_path << "\n";
    std::cout << "Optimizer learning rate after loading: "
              << optimizer2.get_lr() << "\n";

    if (std::abs(optimizer2.get_lr() - 0.001) < 1e-9) {
        std::cout << "✓ Optimizer state loaded correctly!\n\n";
    } else {
        std::cout << "✗ Optimizer state loading failed.\n\n";
        return 1;
    }

    // Demonstrate state_dict usage
    std::cout << "State dictionary example:\n";
    auto state = model->state_dict();
    std::cout << "State dict contains " << state.size() << " tensors:\n";
    for (const auto& [name, tensor] : state) {
        std::cout << "  " << name << ": shape [";
        for (size_t i = 0; i < tensor.shape().size(); ++i) {
            if (i > 0) std::cout << ", ";
            std::cout << tensor.shape()[i];
        }
        std::cout << "]\n";
    }

    std::cout << "\n✓ All serialization examples completed successfully!\n";
    return 0;
}
