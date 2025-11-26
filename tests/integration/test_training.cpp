#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include <tenzor/tenzor.hpp>

using namespace tenzor;
using namespace tenzor::testing;

//==============================================================================
// Training Tests - Backend Parameterized
//==============================================================================

class TrainingTest : public BackendTest {};

TEST_P(TrainingTest, SimpleOptimization) {
    // Create simple model
    auto model = std::make_shared<nn::Linear>(10, 1);
    model->to(device);

    // Create optimizer
    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    // Forward pass
    auto input = Variable(randn({32, 10}, DType::Float32, device), true);
    auto output = model->forward(input);
    auto target = Variable(randn({32, 1}, DType::Float32, device));

    EXPECT_EQ(output.shape()[0], 32);
    EXPECT_EQ(output.shape()[1], 1);
    EXPECT_EQ(output.device().type, device.type);

    // Compute loss
    auto loss = nn::mse_loss(output, target);
    EXPECT_EQ(loss.tensor().numel(), 1);

    // Get initial loss value
    auto loss_cpu = loss.tensor().to(Device::cpu());
    float initial_loss = loss_cpu.data<float>()[0];
    EXPECT_GT(initial_loss, 0.0f);

    // Backward pass
    optimizer.zero_grad();
    loss.backward();

    // Verify gradients exist
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->has_grad()) << "Parameter should have gradient";
        }
    }

    // Update parameters
    optimizer.step();

    SUCCEED();
}

TEST_P(TrainingTest, MultiStepTraining) {
    // Create model
    auto model = std::make_shared<nn::Linear>(10, 5);
    model->to(device);

    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    const int num_steps = 5;
    std::vector<float> losses;

    for (int step = 0; step < num_steps; ++step) {
        auto input = Variable(randn({16, 10}, DType::Float32, device), true);
        auto target = Variable(randn({16, 5}, DType::Float32, device));

        // Forward
        auto output = model->forward(input);
        auto loss = nn::mse_loss(output, target);

        // Record loss
        auto loss_cpu = loss.tensor().to(Device::cpu());
        losses.push_back(loss_cpu.data<float>()[0]);

        // Backward
        optimizer.zero_grad();
        loss.backward();
        optimizer.step();
    }

    // Verify all losses are valid
    for (float loss_val : losses) {
        EXPECT_GT(loss_val, 0.0f);
        EXPECT_FALSE(std::isnan(loss_val));
        EXPECT_FALSE(std::isinf(loss_val));
    }
}

TEST_P(TrainingTest, AdamOptimizer) {
    auto model = std::make_shared<nn::Linear>(20, 10);
    model->to(device);

    auto params = model->parameters();
    auto optimizer = optim::Adam(params, 0.001);

    auto input = Variable(randn({32, 20}, DType::Float32, device), true);
    auto target = Variable(randn({32, 10}, DType::Float32, device));

    // Forward
    auto output = model->forward(input);
    auto loss = nn::mse_loss(output, target);

    // Backward
    optimizer.zero_grad();
    loss.backward();

    // Verify gradients
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->has_grad());
        }
    }

    // Update
    optimizer.step();

    SUCCEED();
}

TEST_P(TrainingTest, TrainingVsEvalMode) {
    auto model = nn::Sequential(
        std::make_shared<nn::Linear>(10, 20),
        std::make_shared<nn::Dropout>(0.5),
        std::make_shared<nn::Linear>(20, 5)
    );
    model.to(device);

    auto input = Variable(randn({16, 10}, DType::Float32, device), true);

    // Training mode
    model.train();
    auto output_train = model.forward(input);

    // Eval mode
    model.eval();
    auto output_eval = model.forward(input);

    EXPECT_EQ(output_train.shape()[0], 16);
    EXPECT_EQ(output_train.shape()[1], 5);
    EXPECT_EQ(output_eval.shape()[0], 16);
    EXPECT_EQ(output_eval.shape()[1], 5);
}

TEST_P(TrainingTest, GradientAccumulation) {
    auto model = std::make_shared<nn::Linear>(10, 5);
    model->to(device);

    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    const int accumulation_steps = 4;

    optimizer.zero_grad();

    for (int step = 0; step < accumulation_steps; ++step) {
        auto input = Variable(randn({8, 10}, DType::Float32, device), true);
        auto target = Variable(randn({8, 5}, DType::Float32, device));

        auto output = model->forward(input);
        auto loss = nn::mse_loss(output, target);

        // Backward (accumulate gradients)
        loss.backward();
    }

    // Verify gradients accumulated
    for (const auto& param : params) {
        if (param->requires_grad()) {
            EXPECT_TRUE(param->has_grad());
        }
    }

    // Update after accumulation
    optimizer.step();

    SUCCEED();
}

TEST_P(TrainingTest, SimpleMLP_Training) {
    class SimpleMLP : public nn::Module {
    public:
        SimpleMLP(int input_size, int hidden_size, int output_size) {
            fc1 = std::make_shared<nn::Linear>(input_size, hidden_size);
            relu = std::make_shared<nn::ReLU>();
            fc2 = std::make_shared<nn::Linear>(hidden_size, output_size);

            register_module("fc1", fc1);
            register_module("fc2", fc2);
        }

        auto forward_impl(const Variable& x) -> Variable override {
            auto h = fc1->forward(x);
            h = relu->forward(h);
            return fc2->forward(h);
        }

    private:
        std::shared_ptr<nn::Linear> fc1, fc2;
        std::shared_ptr<nn::ReLU> relu;
    };

    auto model = std::make_shared<SimpleMLP>(50, 30, 10);
    model->to(device);

    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    const int num_epochs = 10;
    float initial_loss = 0.0f;
    float final_loss = 0.0f;

    for (int epoch = 0; epoch < num_epochs; ++epoch) {
        auto input = Variable(randn({32, 50}, DType::Float32, device), true);
        auto target = Variable(randn({32, 10}, DType::Float32, device));

        optimizer.zero_grad();

        auto output = model->forward(input);
        auto loss = nn::mse_loss(output, target);

        loss.backward();
        optimizer.step();

        auto loss_cpu = loss.tensor().to(Device::cpu());
        float loss_val = loss_cpu.data<float>()[0];

        if (epoch == 0) {
            initial_loss = loss_val;
        }
        if (epoch == num_epochs - 1) {
            final_loss = loss_val;
        }
    }

    // Verify training happened (loss values are valid)
    EXPECT_GT(initial_loss, 0.0f);
    EXPECT_GT(final_loss, 0.0f);
    EXPECT_FALSE(std::isnan(final_loss));
}

TEST_P(TrainingTest, CrossEntropyLoss) {
    auto model = std::make_shared<nn::Linear>(20, 10);
    model->to(device);

    auto params = model->parameters();
    auto optimizer = optim::SGD(params, 0.01);

    // Create input
    auto input = Variable(randn({32, 20}, DType::Float32, device), true);

    // Create one-hot targets on CPU, then transfer
    auto target_cpu = zeros({32, 10}, DType::Float32, Device::cpu());
    float* target_ptr = const_cast<float*>(target_cpu.data<float>());
    for (int i = 0; i < 32; ++i) {
        int label = i % 10;
        target_ptr[i * 10 + label] = 1.0f;
    }

    auto target_data = (device.type == Device::Type::CPU) ? target_cpu : target_cpu.to(device);
    auto target = Variable(target_data, false);

    // Forward
    auto output = model->forward(input);
    auto loss = nn::cross_entropy(output, target.tensor(), nn::Reduction::Mean);

    EXPECT_EQ(loss.tensor().numel(), 1);

    auto loss_cpu = loss.tensor().to(Device::cpu());
    float loss_val = loss_cpu.data<float>()[0];
    EXPECT_GT(loss_val, 0.0f);

    // Backward
    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    SUCCEED();
}

TEST_P(TrainingTest, ParameterUpdate) {
    auto model = std::make_shared<nn::Linear>(5, 3);
    model->to(device);

    auto params = model->parameters();
    ASSERT_GT(params.size(), 0);

    // Store initial parameter values
    auto initial_param = params[0]->tensor().clone();

    auto optimizer = optim::SGD(params, 0.1);  // High LR for visible change

    // Training step
    auto input = Variable(ones({10, 5}, DType::Float32, device), true);
    auto target = Variable(zeros({10, 3}, DType::Float32, device));

    auto output = model->forward(input);
    auto loss = nn::mse_loss(output, target);

    optimizer.zero_grad();
    loss.backward();
    optimizer.step();

    // Get updated parameter
    auto updated_param = params[0]->tensor();

    // Parameters should have changed
    auto initial_cpu = initial_param.to(Device::cpu());
    auto updated_cpu = updated_param.to(Device::cpu());

    auto* initial_data = initial_cpu.data<float>();
    auto* updated_data = updated_cpu.data<float>();

    bool params_changed = false;
    for (int64_t i = 0; i < initial_cpu.numel(); ++i) {
        if (std::abs(initial_data[i] - updated_data[i]) > 1e-7f) {
            params_changed = true;
            break;
        }
    }

    EXPECT_TRUE(params_changed) << "Parameters should change after optimizer step";
}

INSTANTIATE_BACKEND_TESTS(TrainingTest);
