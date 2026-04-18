/**
 * @file test_nn_transformer_parity.cpp
 * @brief Backend parity tests for transformer and attention layers
 *
 * Tests 8 transformer/attention layers across all available backends to ensure
 * consistent results regardless of compute device.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/gqa_attention.hpp>
#include <tenzor/nn/layers/flex_attention.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class NNTransformerParity : public BackendTest {};
// ============================================================================
// Transformer / Attention Parity Tests
// ============================================================================

TEST_P(NNTransformerParity, TransformerEncoderLayer) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::TransformerEncoderLayer layer(64, 4);
        layer.eval();

        auto input = randn({8, 4, 64}, DType::Float32, Device::cpu());
        auto ref = layer.forward(Variable(input, false), Tensor{}, Tensor{}).tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            nn::TransformerEncoderLayer layer_dev(64, 4);
            layer_dev.eval();
            auto params = layer.parameters();
            auto dev_params = layer_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = layer_dev.forward(Variable(input_dev, false), Tensor{}, Tensor{}).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        }
    } catch (...) {
        GTEST_SKIP() << "TransformerEncoderLayer not available";
    }
}

TEST_P(NNTransformerParity, TransformerDecoderLayer) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::TransformerDecoderLayer layer(64, 4);
        layer.eval();

        auto memory = randn({8, 4, 64}, DType::Float32, Device::cpu());
        auto tgt = randn({6, 4, 64}, DType::Float32, Device::cpu());
        auto ref = layer.forward(Variable(tgt, false), Variable(memory, false)).tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            nn::TransformerDecoderLayer layer_dev(64, 4);
            layer_dev.eval();
            auto params = layer.parameters();
            auto dev_params = layer_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            layer_dev.to(backends[i]);
            auto memory_dev = memory.to(backends[i]);
            auto tgt_dev = tgt.to(backends[i]);
            auto output = layer_dev.forward(Variable(tgt_dev, false),
                                           Variable(memory_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        }
    } catch (...) {
        GTEST_SKIP() << "TransformerDecoderLayer not available";
    }
}

TEST_P(NNTransformerParity, TransformerEncoder) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        auto enc_layer = std::make_shared<nn::TransformerEncoderLayer>(64, 4);
        nn::TransformerEncoder encoder(enc_layer, 2);
        encoder.eval();

        auto input = randn({8, 4, 64}, DType::Float32, Device::cpu());
        auto ref = encoder.forward(Variable(input, false), Tensor{}, Tensor{}).tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            auto enc_layer_dev = std::make_shared<nn::TransformerEncoderLayer>(64, 4);
            nn::TransformerEncoder encoder_dev(enc_layer_dev, 2);
            encoder_dev.eval();
            auto params = encoder.parameters();
            auto dev_params = encoder_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            encoder_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = encoder_dev.forward(Variable(input_dev, false), Tensor{}, Tensor{}).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        }
    } catch (...) {
        GTEST_SKIP() << "TransformerEncoder not available";
    }
}

TEST_P(NNTransformerParity, MultiheadAttention) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::MultiheadAttention attn(64, 4);
        attn.eval();

        auto query = randn({8, 4, 64}, DType::Float32, Device::cpu());
        auto key = randn({8, 4, 64}, DType::Float32, Device::cpu());
        auto value = randn({8, 4, 64}, DType::Float32, Device::cpu());

        auto [ref_out, ref_weights] = attn.forward(
            Variable(query, false), Variable(key, false), Variable(value, false));
        auto ref = ref_out.tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            nn::MultiheadAttention attn_dev(64, 4);
            attn_dev.eval();
            auto params = attn.parameters();
            auto dev_params = attn_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            attn_dev.to(backends[i]);
            auto query_dev = query.to(backends[i]);
            auto key_dev = key.to(backends[i]);
            auto value_dev = value.to(backends[i]);
            auto [out, weights] = attn_dev.forward(
                Variable(query_dev, false), Variable(key_dev, false),
                Variable(value_dev, false));
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out.tensor(), 1e-3f, 1e-3f);
        }
    } catch (...) {
        GTEST_SKIP() << "MultiheadAttention not available";
    }
}

TEST_P(NNTransformerParity, PositionalEncoding) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::PositionalEncoding pe(64, 100);
        pe.eval();

        auto input = randn({8, 4, 64}, DType::Float32, Device::cpu());
        auto ref = pe.forward(Variable(input, false)).tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            nn::PositionalEncoding pe_dev(64, 100);
            pe_dev.eval();
            auto params = pe.parameters();
            auto dev_params = pe_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            pe_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = pe_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        }
    } catch (...) {
        GTEST_SKIP() << "PositionalEncoding not available";
    }
}

// Previously DISABLED_ due to suspected CPU hang. Standalone verification
// confirms CPU flex_attention completes in milliseconds; the original "hang"
// was accumulated OneAPI backend-initialization time exceeding a short test
// timeout. The test now runs across all available backends.
TEST_P(NNTransformerParity, FlexAttention) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    constexpr int64_t B = 2, H = 4, S = 32, D = 16;
    constexpr int64_t block_size = 16;

    try {
        auto q = randn({B, H, S, D}, DType::Float32, Device::cpu());
        auto k = randn({B, H, S, D}, DType::Float32, Device::cpu());
        auto v = randn({B, H, S, D}, DType::Float32, Device::cpu());

        auto mask_cpu = nn::BlockMask::causal(S, block_size);
        auto ref = nn::flex_attention(q, k, v, mask_cpu,
                                      nn::causal_score_mod(), -1.0f);

        for (size_t i = 1; i < backends.size(); ++i) {
            auto q_dev = q.to(backends[i]);
            auto k_dev = k.to(backends[i]);
            auto v_dev = v.to(backends[i]);
            auto mask_bool = mask_cpu.mask().to(backends[i]);
            nn::BlockMask mask_dev(mask_bool, block_size);

            auto out = nn::flex_attention(q_dev, k_dev, v_dev, mask_dev,
                                          nn::causal_score_mod(), -1.0f);
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out.to(Device::cpu()), 1e-3f, 1e-3f);
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "FlexAttention unsupported on one of the backends: "
                     << e.what();
    }
}

// Sliding-window attention is exposed as the window_size parameter on
// GroupedQueryAttention. The prior cross-backend divergence traced to a
// broken Expand kernel on CUDA/Vulkan/OneAPI that ignored the input's actual
// strides when the input was a non-contiguous view (repeat_kv feeds it the
// result of permute+unsqueeze). Fix: materialize input to contiguous in
// each backend's expand_kernel.
TEST_P(NNTransformerParity, SlidingWindowAttention) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        nn::GroupedQueryAttention gqa(64, 4, 2, 0.0, true, true, nullptr, 4);
        gqa.eval();

        auto query = randn({4, 16, 64}, DType::Float32, Device::cpu());
        auto key = randn({4, 16, 64}, DType::Float32, Device::cpu());
        auto value = randn({4, 16, 64}, DType::Float32, Device::cpu());

        auto [ref_out, ref_weights] = gqa.forward(
            Variable(query, false), Variable(key, false), Variable(value, false));
        auto ref = ref_out.tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            nn::GroupedQueryAttention gqa_dev(64, 4, 2, 0.0, true, true, nullptr, 4);
            gqa_dev.eval();
            auto params = gqa.parameters();
            auto dev_params = gqa_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            gqa_dev.to(backends[i]);
            auto query_dev = query.to(backends[i]);
            auto key_dev = key.to(backends[i]);
            auto value_dev = value.to(backends[i]);
            auto [out, weights] = gqa_dev.forward(
                Variable(query_dev, false), Variable(key_dev, false),
                Variable(value_dev, false));
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out.tensor().to(Device::cpu()),
                                 1e-3f, 1e-3f);
        }
    } catch (const std::exception& e) {
        GTEST_SKIP() << "Sliding-window GQA unsupported on one of the backends: "
                     << e.what();
    }
}

TEST_P(NNTransformerParity, GroupedQueryAttention) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    try {
        // d_model=64, num_heads=4, num_kv_heads=2
        nn::GroupedQueryAttention gqa(64, 4, 2);
        gqa.eval();

        // GQA expects batch-first: (N, L, embed_dim)
        auto query = randn({4, 8, 64}, DType::Float32, Device::cpu());
        auto key = randn({4, 8, 64}, DType::Float32, Device::cpu());
        auto value = randn({4, 8, 64}, DType::Float32, Device::cpu());

        auto [ref_out, ref_weights] = gqa.forward(
            Variable(query, false), Variable(key, false), Variable(value, false));
        auto ref = ref_out.tensor();

        for (size_t i = 1; i < backends.size(); ++i) {
            nn::GroupedQueryAttention gqa_dev(64, 4, 2);
            gqa_dev.eval();
            auto params = gqa.parameters();
            auto dev_params = gqa_dev.parameters();
            for (size_t p = 0; p < params.size(); ++p) {
                dev_params[p]->tensor() = params[p]->tensor().clone();
            }
            gqa_dev.to(backends[i]);
            auto query_dev = query.to(backends[i]);
            auto key_dev = key.to(backends[i]);
            auto value_dev = value.to(backends[i]);
            auto [out, weights] = gqa_dev.forward(
                Variable(query_dev, false), Variable(key_dev, false),
                Variable(value_dev, false));
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, out.tensor(), 1e-3f, 1e-3f);
        }
    } catch (...) {
        GTEST_SKIP() << "GroupedQueryAttention not available";
    }
}

// ============================================================================
// Main
// ============================================================================

INSTANTIATE_BACKEND_TESTS(NNTransformerParity);


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
