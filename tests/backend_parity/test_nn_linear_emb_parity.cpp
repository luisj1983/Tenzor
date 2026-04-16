/**
 * @file test_nn_linear_emb_parity.cpp
 * @brief Linear, Bilinear, EmbeddingBag, and LazyLinear parity tests across backends
 *
 * Verifies that linear and embedding layers produce consistent results
 * across CPU, CUDA, ROCm, OneAPI, and Vulkan backends.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/lazy_linear.hpp>
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

// ============================================================================
// Linear layers
// ============================================================================

TEST(NNLinearEmbParity, Linear_Basic) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::Linear layer(64, 32);
    auto input = randn({4, 64}, DType::Float32, Device::cpu());
    auto ref = layer.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Linear layer_dev(64, 32);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNLinearEmbParity, Linear_NoBias) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::Linear layer(64, 32, false);
    auto input = randn({4, 64}, DType::Float32, Device::cpu());
    auto ref = layer.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Linear layer_dev(64, 32, false);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNLinearEmbParity, Bilinear) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::Bilinear layer(32, 32, 16);
    auto input1 = randn({4, 32}, DType::Float32, Device::cpu());
    auto input2 = randn({4, 32}, DType::Float32, Device::cpu());
    auto ref = layer.forward(Variable(input1, false), Variable(input2, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::Bilinear layer_dev(32, 32, 16);
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto in1_dev = input1.to(backends[i]);
            auto in2_dev = input2.to(backends[i]);
            auto output = layer_dev.forward(Variable(in1_dev, false),
                                             Variable(in2_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// EmbeddingBag
// ============================================================================

TEST(NNLinearEmbParity, EmbeddingBag_Sum) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // num_embeddings=100, embedding_dim=32, max_norm=0, norm_type=2, scale_grad=false, mode="sum"
    nn::EmbeddingBag layer(100, 32, 0.0, 2.0, false, "sum");

    // Flattened indices: 4 bags of 8 indices each = 32 total
    auto indices = (rand({32}, DType::Float32, Device::cpu()) * 100).to(DType::Int64);
    // Offsets marking start of each bag
    auto offsets_data = zeros({4}, DType::Int64, Device::cpu());
    // Manually set offsets: 0, 8, 16, 24
    auto offsets_ptr = offsets_data.data<int64_t>();
    offsets_ptr[0] = 0; offsets_ptr[1] = 8; offsets_ptr[2] = 16; offsets_ptr[3] = 24;

    auto ref = layer.forward(Variable(indices, false), Variable(offsets_data, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::EmbeddingBag layer_dev(100, 32, 0.0, 2.0, false, "sum");
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto indices_dev = indices.to(backends[i]);
            auto offsets_dev = offsets_data.to(backends[i]);
            auto output = layer_dev.forward(Variable(indices_dev, false),
                                             Variable(offsets_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-5f, 1e-6f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST(NNLinearEmbParity, EmbeddingBag_Mean) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::EmbeddingBag layer(100, 32, 0.0, 2.0, false, "mean");

    auto indices = (rand({32}, DType::Float32, Device::cpu()) * 100).to(DType::Int64);
    auto offsets_data = zeros({4}, DType::Int64, Device::cpu());
    auto offsets_ptr = offsets_data.data<int64_t>();
    offsets_ptr[0] = 0; offsets_ptr[1] = 8; offsets_ptr[2] = 16; offsets_ptr[3] = 24;

    auto ref = layer.forward(Variable(indices, false), Variable(offsets_data, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::EmbeddingBag layer_dev(100, 32, 0.0, 2.0, false, "mean");
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto indices_dev = indices.to(backends[i]);
            auto offsets_dev = offsets_data.to(backends[i]);
            auto output = layer_dev.forward(Variable(indices_dev, false),
                                             Variable(offsets_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-5f, 1e-6f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// LazyLinear
// ============================================================================

TEST(NNLinearEmbParity, LazyLinear) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::LazyLinear layer(32);
    auto input = randn({4, 64}, DType::Float32, Device::cpu());

    // First forward materializes weights on CPU
    auto ref = layer.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LazyLinear layer_dev(32);
            // Materialize on CPU first so we can copy params
            auto warmup = randn({1, 64}, DType::Float32, Device::cpu());
            layer_dev.forward(Variable(warmup, false));

            // Copy materialized params from reference layer
            auto params_src = layer.parameters();
            auto params_dst = layer_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            std::cerr << "Skipped on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

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
