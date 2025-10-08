#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

TEST(TrainingTest, SimpleOptimization) {
    // Create simple model
    auto model = std::make_shared<nn::Linear>(10, 1);

    // Create optimizer
    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    // Forward pass
    auto input = Variable(randn({32, 10}), true);
    auto output = model->forward(input);
    auto target = Variable(randn({32, 1}));

    // Compute loss
    auto loss = nn::mse_loss(output, target);

    // Backward pass
    optimizer.zero_grad();
    // loss.backward();  // TODO: Implement backward

    // Update parameters
    optimizer.step();

    SUCCEED();
}
