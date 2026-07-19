/**
 * @file test_nn_parity.cpp
 * @brief Neural network operation parity tests
 *
 * Tests 30+ neural network operations including convolutions, pooling,
 * normalization, activations, and loss functions across all backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class NNOperationParity : public BackendTest {};
// ============================================================================
// Convolution Operations
// ============================================================================

TEST_P(NNOperationParity, Conv2d_Basic) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::Conv2d conv(3, 16, 3, 1, 1);  // 3x3, stride 1, pad 1
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    // CPU reference
    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::Conv2d conv_dev(3, 16, 3, 1, 1);
        // Copy weights from CPU conv
        auto params = conv.parameters();
        auto dev_params = conv_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        conv_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, Conv2d_Stride2) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::Conv2d conv(3, 16, 3, 2, 1);  // stride 2
    auto input = randn({1, 3, 16, 16}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::Conv2d conv_dev(3, 16, 3, 2, 1);
        auto params = conv.parameters();
        auto dev_params = conv_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        conv_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, Conv2d_Padding2) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::Conv2d conv(3, 16, 5, 1, 2);  // 5x5, pad 2
    auto input = randn({1, 3, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::Conv2d conv_dev(3, 16, 5, 1, 2);
        auto params = conv.parameters();
        auto dev_params = conv_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        conv_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, Conv2d_Dilation) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::Conv2d conv(3, 16, 3, 1, 2, 2);  // dilation=2, pad=2
    auto input = randn({1, 3, 16, 16}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::Conv2d conv_dev(3, 16, 3, 1, 2, 2);
        auto params = conv.parameters();
        auto dev_params = conv_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        conv_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, Conv2d_Groups) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::Conv2d conv(16, 16, 3, 1, 1, 1, 16);  // depthwise: groups=16
    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::Conv2d conv_dev(16, 16, 3, 1, 1, 1, 16);
        auto params = conv.parameters();
        auto dev_params = conv_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        conv_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, ConvTranspose2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::ConvTranspose2d conv(16, 3, 4, 2, 1);  // upsample 2x
    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = conv.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::ConvTranspose2d conv_dev(16, 3, 4, 2, 1);
        auto params = conv.parameters();
        auto dev_params = conv_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        conv_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = conv_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

// ============================================================================
// Pooling Operations
// ============================================================================

TEST_P(NNOperationParity, MaxPool2d_2x2) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool2d pool(2);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "MaxPool2d_2x2");
}

TEST_P(NNOperationParity, MaxPool2d_3x3_Stride2) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({1, 16, 16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool2d pool(3, 2, 1);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "MaxPool2d_3x3_Stride2");
}

TEST_P(NNOperationParity, AvgPool2d_2x2) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    // Float32 avg-pool is literally a sum-then-divide; its per-element
    // error is bounded by ~1 ULP (≈1.2e-7). The previous atol=1e-8 was
    // tighter than Float32 machine precision, so any kernel doing the
    // division in a different order from CPU reference could trip it.
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AvgPool2d pool(2);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "AvgPool2d_2x2");
}

TEST_P(NNOperationParity, AdaptiveAvgPool2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({1, 16, 16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AdaptiveAvgPool2d pool(4, 4);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-6f, 1e-8f, "AdaptiveAvgPool2d");
}

TEST_P(NNOperationParity, AdaptiveMaxPool2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AdaptiveMaxPool2d pool(4, 4);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-6f, 1e-8f, "AdaptiveMaxPool2d");
}

// ============================================================================
// Normalization Operations
// ============================================================================

TEST_P(NNOperationParity, BatchNorm2d_Train) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::BatchNorm2d bn(16);
    bn.train();
    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = bn.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::BatchNorm2d bn_dev(16);
        bn_dev.train();
        auto params = bn.parameters();
        auto dev_params = bn_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        bn_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = bn_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, BatchNorm2d_Eval) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::BatchNorm2d bn(16);
    // Run a training forward to populate running stats
    auto train_input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    bn.train();
    bn.forward(Variable(train_input, false));
    bn.eval();

    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    auto ref_output = bn.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::BatchNorm2d bn_dev(16);
        bn_dev.train();
        auto params = bn.parameters();
        auto dev_params = bn_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        bn_dev.to(backends[i]);
        // Run training to populate running stats
        auto train_dev = train_input.to(backends[i]);
        bn_dev.forward(Variable(train_dev, false));
        bn_dev.eval();

        auto input_dev = input.to(backends[i]);
        auto output = bn_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, LayerNorm) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::LayerNorm ln({64});
    auto input = randn({4, 64}, DType::Float32, Device::cpu());

    auto ref_output = ln.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::LayerNorm ln_dev({64});
        auto params = ln.parameters();
        auto dev_params = ln_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        ln_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = ln_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

TEST_P(NNOperationParity, GroupNorm) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::GroupNorm gn(4, 16);  // 4 groups, 16 channels
    auto input = randn({2, 16, 8, 8}, DType::Float32, Device::cpu());

    auto ref_output = gn.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::GroupNorm gn_dev(4, 16);
        auto params = gn.parameters();
        auto dev_params = gn_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        gn_dev.to(backends[i]);
        auto input_dev = input.to(backends[i]);
        auto output = gn_dev.forward(Variable(input_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-4f, 1e-5f);
    }
}

// ============================================================================
// Activation Functions
// ============================================================================

TEST_P(NNOperationParity, ReLU) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::relu(input_var).tensor();
    }, {input}, 1e-7f, 1e-9f, "ReLU");
}

TEST_P(NNOperationParity, ReLU6) {
    // ReLU6 can be implemented as clamp(relu(x), 0, 6) but no dedicated nn::relu6
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto x = Variable(inputs[0], false);
        auto out = nn::relu(x);
        return tenzor::clamp(out, 0.0f, 6.0f).tensor();
    }, {input}, 1e-7f, 1e-9f, "ReLU6");
}

TEST_P(NNOperationParity, LeakyReLU) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::leaky_relu(input_var, 0.01f).tensor();
    }, {input}, 1e-6f, 1e-8f, "LeakyReLU");
}

TEST_P(NNOperationParity, ELU) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    // ELU for x<0 is alpha*(exp(x)-1); the exp evaluation differs by
    // ~0.5 ULP (~6e-8) between CPU (MKL VML) and CUDA (expf intrinsic).
    // The previous atol=1e-8 was tighter than a single Float32 ULP and
    // trivially failed. Raise to atol=1e-6, matching Softplus/GELU.
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::elu(input_var, 1.0f).tensor();
    }, {input}, 1e-5f, 1e-6f, "ELU");
}

TEST_P(NNOperationParity, GELU) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    // atol=1e-6 (not 1e-7): GELU is erf-based on every backend, and a single
    // Float32 ULP of erf(x)≈1 scaled by 0.5·x (x up to ~3 here) is ≈1.8e-7 of
    // absolute error at small-output points (x≈-3) where rtol cannot help. The
    // previous 1e-7 was below Float32's achievable precision for a transcendental
    // and contradicted this file's own standard (see Softplus, raised to 1e-6).
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::gelu(input_var).tensor();
    }, {input}, 1e-5f, 1e-6f, "GELU");
}

TEST_P(NNOperationParity, Swish) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::swish(input_var).tensor();
    }, {input}, 1e-6f, 1e-8f, "Swish");
}

TEST_P(NNOperationParity, Softmax_Dim1) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::softmax(input_var, /*dim=*/1).tensor();
    }, {input}, 1e-6f, 1e-8f, "Softmax Dim1");
}

TEST_P(NNOperationParity, LogSoftmax) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto input_var = Variable(inputs[0], false);
        return nn::log_softmax(input_var, /*dim=*/1).tensor();
    }, {input}, 1e-6f, 1e-8f, "LogSoftmax");
}

// ============================================================================
// Dropout (eval mode - should be identity)
// ============================================================================

TEST_P(NNOperationParity, Dropout_Eval) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto input = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::Dropout dropout(0.5);
        dropout.eval();  // Eval mode: dropout is identity
        return dropout.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "Dropout_Eval");
}

// ============================================================================
// Embedding
// ============================================================================

TEST_P(NNOperationParity, Embedding) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    nn::Embedding emb(100, 32);  // 100 tokens, 32 dims
    auto indices = (rand({4, 8}, DType::Float32, Device::cpu()) * 100).to(DType::Int64);

    auto ref_output = emb.forward(Variable(indices, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        nn::Embedding emb_dev(100, 32);
        auto params = emb.parameters();
        auto dev_params = emb_dev.parameters();
        for (size_t p = 0; p < params.size(); ++p) {
            dev_params[p]->tensor() = params[p]->tensor().clone();
        }
        emb_dev.to(backends[i]);
        auto indices_dev = indices.to(backends[i]);
        auto output = emb_dev.forward(Variable(indices_dev, false)).tensor();
        backends[i].synchronize();
        EXPECT_TENSORS_CLOSE(ref_output, output, 1e-6f, 1e-8f);
    }
}

// ============================================================================
// Loss Functions
// ============================================================================

TEST_P(NNOperationParity, MSELoss) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::MSELoss();
        auto pred_var = Variable(inputs[0], false);
        auto target_var = Variable(inputs[1], false);
        return loss_fn(pred_var, target_var).tensor();
    }, {pred, target}, 1e-6f, 1e-8f, "MSELoss");
}

TEST_P(NNOperationParity, L1Loss) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::L1Loss();
        auto pred_var = Variable(inputs[0], false);
        auto target_var = Variable(inputs[1], false);
        return loss_fn(pred_var, target_var).tensor();
    }, {pred, target}, 1e-6f, 1e-8f, "L1Loss");
}

TEST_P(NNOperationParity, CrossEntropyLoss) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto pred = randn({32, 10}, DType::Float32, Device::cpu());
    auto target = (rand({32}, DType::Float32, Device::cpu()) * 10).to(DType::Int64);

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::CrossEntropyLoss();
        auto pred_var = Variable(inputs[0], false);
        return loss_fn(pred_var, inputs[1]).tensor();
    }, {pred, target}, 1e-5f, 1e-7f, "CrossEntropyLoss");
}

TEST_P(NNOperationParity, BCELoss) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto pred = sigmoid(randn({32, 64}, DType::Float32, Device::cpu()));
    auto target = rand({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::BCELoss();
        auto pred_var = Variable(inputs[0], false);
        auto target_var = Variable(inputs[1], false);
        return loss_fn(pred_var, target_var).tensor();
    }, {pred, target}, 1e-5f, 1e-7f, "BCELoss");
}

TEST_P(NNOperationParity, BCEWithLogitsLoss) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = rand({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::BCEWithLogitsLoss();
        auto pred_var = Variable(inputs[0], false);
        auto target_var = Variable(inputs[1], false);
        return loss_fn(pred_var, target_var).tensor();
    }, {pred, target}, 1e-5f, 1e-7f, "BCEWithLogitsLoss");
}

TEST_P(NNOperationParity, SmoothL1Loss) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn parity");

    auto pred = randn({32, 64}, DType::Float32, Device::cpu());
    auto target = randn({32, 64}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        auto loss_fn = nn::SmoothL1Loss();
        auto pred_var = Variable(inputs[0], false);
        auto target_var = Variable(inputs[1], false);
        return loss_fn(pred_var, target_var).tensor();
    }, {pred, target}, 1e-5f, 1e-7f, "SmoothL1Loss");
}

// ============================================================================
// Circular Padding Layers (fwd + bwd parity)
// ============================================================================
// Commit 77485367 added CircularPad{1,2,3}d. These tests exercise both the
// forward wrap-around semantics and the autograd gradient routing through
// the padding op on every available backend.

TEST_P(NNOperationParity, CircularPad1d_FwdBwd) {
    // Input: (N=2, C=3, W=6); pad 2 on each side => W=10
    auto input = randn({2, 3, 6}, DType::Float32, Device::cpu());

    test_gradient_parity(
        [](std::vector<Variable>& vars) -> Variable {
            nn::CircularPad1d pad(2, 2);
            return pad.forward(vars[0]);
        },
        {input},
        /*grad_output_factory=*/{},
        /*rtol_fwd=*/1e-5f, /*atol_fwd=*/1e-7f,
        /*rtol_bwd=*/1e-4f, /*atol_bwd=*/1e-6f,
        /*backends=*/{},
        "CircularPad1d");
}

TEST_P(NNOperationParity, CircularPad2d_FwdBwd) {
    // Input: (N=2, C=3, H=4, W=5); pad 1 on each side
    auto input = randn({2, 3, 4, 5}, DType::Float32, Device::cpu());

    test_gradient_parity(
        [](std::vector<Variable>& vars) -> Variable {
            nn::CircularPad2d pad(1, 1, 1, 1);
            return pad.forward(vars[0]);
        },
        {input}, {},
        1e-5f, 1e-7f, 1e-4f, 1e-6f, {},
        "CircularPad2d");
}

TEST_P(NNOperationParity, CircularPad3d_FwdBwd) {
    // Input: (N=1, C=2, D=3, H=4, W=4); symmetric pad 1
    auto input = randn({1, 2, 3, 4, 4}, DType::Float32, Device::cpu());

    test_gradient_parity(
        [](std::vector<Variable>& vars) -> Variable {
            nn::CircularPad3d pad(1);
            return pad.forward(vars[0]);
        },
        {input}, {},
        1e-5f, 1e-7f, 1e-4f, 1e-6f, {},
        "CircularPad3d");
}

INSTANTIATE_BACKEND_TESTS(NNOperationParity);


int main(int argc, char** argv) {
    ::testing::InitGoogleTest(&argc, argv);

    try {
        if (!::testing::GTEST_FLAG(list_tests)) {
            tenzor::initialize();
        }
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
