/**
 * @file test_nn_parity.cpp
 * @brief Neural network operation parity tests
 *
 * Tests 30+ neural network operations including convolutions, pooling,
 * normalization, activations, and loss functions across all backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Convolution Operations
// ============================================================================

TEST(NNOperationParity, Conv2d_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 3, 32, 32}, DType::Float32, Device::cpu());
    auto weight = randn({16, 3, 3, 3}, DType::Float32, Device::cpu());
    auto bias = randn({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv2d(inputs[0], inputs[1], inputs[2],
                                     /*stride=*/1, /*padding=*/1);
    }, {input, weight, bias}, 1e-4f, 1e-6f, "Conv2d Basic");
}

TEST(NNOperationParity, Conv2d_Stride2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 64, 64}, DType::Float32, Device::cpu());
    auto weight = randn({32, 16, 3, 3}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv2d(inputs[0], inputs[1], std::nullopt,
                                     /*stride=*/2, /*padding=*/1);
    }, {input, weight}, 1e-4f, 1e-6f, "Conv2d Stride2");
}

TEST(NNOperationParity, Conv2d_Padding2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 8, 32, 32}, DType::Float32, Device::cpu());
    auto weight = randn({16, 8, 5, 5}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv2d(inputs[0], inputs[1], std::nullopt,
                                     /*stride=*/1, /*padding=*/2);
    }, {input, weight}, 1e-4f, 1e-6f, "Conv2d Padding2");
}

TEST(NNOperationParity, Conv2d_Dilation) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 8, 32, 32}, DType::Float32, Device::cpu());
    auto weight = randn({16, 8, 3, 3}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv2d(inputs[0], inputs[1], std::nullopt,
                                     /*stride=*/1, /*padding=*/2, /*dilation=*/2);
    }, {input, weight}, 1e-4f, 1e-6f, "Conv2d Dilation");
}

TEST(NNOperationParity, Conv2d_Groups) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());
    auto weight = randn({32, 8, 3, 3}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv2d(inputs[0], inputs[1], std::nullopt,
                                     /*stride=*/1, /*padding=*/1,
                                     /*dilation=*/1, /*groups=*/2);
    }, {input, weight}, 1e-4f, 1e-6f, "Conv2d Groups");
}

TEST(NNOperationParity, ConvTranspose2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 16, 16}, DType::Float32, Device::cpu());
    auto weight = randn({16, 32, 3, 3}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::conv_transpose2d(inputs[0], inputs[1], std::nullopt,
                                               /*stride=*/2, /*padding=*/1);
    }, {input, weight}, 1e-4f, 1e-6f, "ConvTranspose2d");
}

// ============================================================================
// Pooling Operations
// ============================================================================

TEST(NNOperationParity, MaxPool2d_2x2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::max_pool2d(inputs[0], /*kernel_size=*/2);
    }, {input}, 1e-6f, 1e-8f, "MaxPool2d 2x2");
}

TEST(NNOperationParity, MaxPool2d_3x3_Stride2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::max_pool2d(inputs[0], /*kernel_size=*/3,
                                         /*stride=*/2, /*padding=*/1);
    }, {input}, 1e-6f, 1e-8f, "MaxPool2d 3x3 Stride2");
}

TEST(NNOperationParity, AvgPool2d_2x2) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::avg_pool2d(inputs[0], /*kernel_size=*/2);
    }, {input}, 1e-6f, 1e-8f, "AvgPool2d 2x2");
}

TEST(NNOperationParity, AdaptiveAvgPool2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::adaptive_avg_pool2d(inputs[0], {7, 7});
    }, {input}, 1e-5f, 1e-7f, "AdaptiveAvgPool2d");
}

TEST(NNOperationParity, AdaptiveMaxPool2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::adaptive_max_pool2d(inputs[0], {7, 7});
    }, {input}, 1e-6f, 1e-8f, "AdaptiveMaxPool2d");
}

// ============================================================================
// Normalization Operations
// ============================================================================

TEST(NNOperationParity, BatchNorm2d_Train) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 16, 32, 32}, DType::Float32, Device::cpu());
    auto weight = ones({16}, DType::Float32, Device::cpu());
    auto bias = zeros({16}, DType::Float32, Device::cpu());
    auto running_mean = zeros({16}, DType::Float32, Device::cpu());
    auto running_var = ones({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::batch_norm(inputs[0], inputs[3], inputs[4],
                                         inputs[1], inputs[2],
                                         /*training=*/true, /*momentum=*/0.1f,
                                         /*eps=*/1e-5f);
    }, {input, weight, bias, running_mean, running_var}, 1e-4f, 1e-6f, "BatchNorm2d Train");
}

TEST(NNOperationParity, BatchNorm2d_Eval) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 16, 32, 32}, DType::Float32, Device::cpu());
    auto weight = ones({16}, DType::Float32, Device::cpu());
    auto bias = zeros({16}, DType::Float32, Device::cpu());
    auto running_mean = randn({16}, DType::Float32, Device::cpu());
    auto running_var = ones({16}, DType::Float32, Device::cpu()) * 0.5f;

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::batch_norm(inputs[0], inputs[3], inputs[4],
                                         inputs[1], inputs[2],
                                         /*training=*/false, /*momentum=*/0.1f,
                                         /*eps=*/1e-5f);
    }, {input, weight, bias, running_mean, running_var}, 1e-5f, 1e-7f, "BatchNorm2d Eval");
}

TEST(NNOperationParity, LayerNorm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({4, 32, 64}, DType::Float32, Device::cpu());
    auto weight = ones({64}, DType::Float32, Device::cpu());
    auto bias = zeros({64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::layer_norm(inputs[0], {64}, inputs[1], inputs[2]);
    }, {input, weight, bias}, 1e-5f, 1e-7f, "LayerNorm");
}

TEST(NNOperationParity, GroupNorm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({2, 16, 32, 32}, DType::Float32, Device::cpu());
    auto weight = ones({16}, DType::Float32, Device::cpu());
    auto bias = zeros({16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::group_norm(inputs[0], /*num_groups=*/4,
                                         inputs[1], inputs[2]);
    }, {input, weight, bias}, 1e-5f, 1e-7f, "GroupNorm");
}

// ============================================================================
// Activation Functions
// ============================================================================

TEST(NNOperationParity, ReLU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::relu(inputs[0]);
    }, {input}, 1e-7f, 1e-9f, "ReLU");
}

TEST(NNOperationParity, ReLU6) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::relu6(inputs[0]);
    }, {input}, 1e-7f, 1e-9f, "ReLU6");
}

TEST(NNOperationParity, LeakyReLU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::leaky_relu(inputs[0], 0.01f);
    }, {input}, 1e-6f, 1e-8f, "LeakyReLU");
}

TEST(NNOperationParity, ELU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::elu(inputs[0], 1.0f);
    }, {input}, 1e-6f, 1e-8f, "ELU");
}

TEST(NNOperationParity, GELU) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::gelu(inputs[0]);
    }, {input}, 1e-5f, 1e-7f, "GELU");
}

TEST(NNOperationParity, Swish) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::swish(inputs[0]);
    }, {input}, 1e-6f, 1e-8f, "Swish");
}

TEST(NNOperationParity, Softmax_Dim1) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::softmax(inputs[0], /*dim=*/1);
    }, {input}, 1e-6f, 1e-8f, "Softmax Dim1");
}

TEST(NNOperationParity, LogSoftmax) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::log_softmax(inputs[0], /*dim=*/1);
    }, {input}, 1e-6f, 1e-8f, "LogSoftmax");
}

// ============================================================================
// Dropout (with fixed seed for reproducibility)
// ============================================================================

TEST(NNOperationParity, Dropout_Eval) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // In eval mode, dropout should be identity
        return nn::functional::dropout(inputs[0], 0.5f, /*training=*/false);
    }, {input}, 1e-7f, 1e-9f, "Dropout Eval");
}

// ============================================================================
// Embedding
// ============================================================================

TEST(NNOperationParity, Embedding) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto indices = (rand({4, 32}, DType::Float32, Device::cpu()) * 100).to(DType::Int64);
    auto weight = randn({1000, 128}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        return nn::functional::embedding(inputs[0], inputs[1]);
    }, {indices, weight}, 1e-7f, 1e-9f, "Embedding");
}

// ============================================================================
// Loss Functions
// ============================================================================

TEST(NNOperationParity, MSELoss) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::MSELoss();
        return loss_fn(inputs[0], inputs[1]);
    }, {pred, target}, 1e-6f, 1e-8f, "MSELoss");
}

TEST(NNOperationParity, L1Loss) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::L1Loss();
        return loss_fn(inputs[0], inputs[1]);
    }, {pred, target}, 1e-6f, 1e-8f, "L1Loss");
}

TEST(NNOperationParity, CrossEntropyLoss) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 10}, DType::Float32, Device::cpu());
    auto target = (rand({32}, DType::Float32, Device::cpu()) * 10).to(DType::Int64);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::CrossEntropyLoss();
        return loss_fn(inputs[0], inputs[1]);
    }, {pred, target}, 1e-5f, 1e-7f, "CrossEntropyLoss");
}

TEST(NNOperationParity, BCELoss) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = sigmoid(randn({32, 64}, DType::Float32, Device::cpu()));
    auto target = rand({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::BCELoss();
        return loss_fn(inputs[0], inputs[1]);
    }, {pred, target}, 1e-5f, 1e-7f, "BCELoss");
}

TEST(NNOperationParity, BCEWithLogitsLoss) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = rand({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::BCEWithLogitsLoss();
        return loss_fn(inputs[0], inputs[1]);
    }, {pred, target}, 1e-5f, 1e-7f, "BCEWithLogitsLoss");
}

TEST(NNOperationParity, SmoothL1Loss) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::SmoothL1Loss();
        return loss_fn(inputs[0], inputs[1]);
    }, {pred, target}, 1e-5f, 1e-7f, "SmoothL1Loss");
}

int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
        return 1;
    }

    int result = RUN_ALL_TESTS();

    try {
        tenzor::finalize();
    } catch (...) {}

    return result;
}
