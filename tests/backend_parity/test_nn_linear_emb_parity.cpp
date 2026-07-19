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
#include <tenzor/nn/layers/lazy_conv.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class NNLinearEmbParity : public BackendTest {};
// ============================================================================
// Linear layers
// ============================================================================

TEST_P(NNLinearEmbParity, Linear_Basic) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

    nn::Linear layer(64, 32);
    auto input = randn({4, 64}, DType::Float32, Device::cpu());
    auto ref = layer.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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
            ADD_FAILURE() << "Linear_Basic failed on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNLinearEmbParity, Linear_NoBias) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

    nn::Linear layer(64, 32, false);
    auto input = randn({4, 64}, DType::Float32, Device::cpu());
    auto ref = layer.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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
            ADD_FAILURE() << "Linear_NoBias failed on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNLinearEmbParity, Bilinear) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

    nn::Bilinear layer(32, 32, 16);
    auto input1 = randn({4, 32}, DType::Float32, Device::cpu());
    auto input2 = randn({4, 32}, DType::Float32, Device::cpu());
    auto ref = layer.forward(Variable(input1, false), Variable(input2, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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
            // Bilinear forward is an outer-product @ weight matmul
            // (src/nn/layers/linear.cpp:288) → FP32 cross-device GEMM floor on
            // rtol (parity::MATMUL_RTOL). atol kept at 1e-3: the outer product
            // accumulates one extra reduction stage beyond a plain GEMM.
            EXPECT_TENSORS_CLOSE(ref, output, parity::MATMUL_RTOL, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "Bilinear failed on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// EmbeddingBag
// ============================================================================

TEST_P(NNLinearEmbParity, EmbeddingBag_Sum) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

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

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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
            ADD_FAILURE() << "EmbeddingBag_Sum failed on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNLinearEmbParity, EmbeddingBag_Mean) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

    nn::EmbeddingBag layer(100, 32, 0.0, 2.0, false, "mean");

    auto indices = (rand({32}, DType::Float32, Device::cpu()) * 100).to(DType::Int64);
    auto offsets_data = zeros({4}, DType::Int64, Device::cpu());
    auto offsets_ptr = offsets_data.data<int64_t>();
    offsets_ptr[0] = 0; offsets_ptr[1] = 8; offsets_ptr[2] = 16; offsets_ptr[3] = 24;

    auto ref = layer.forward(Variable(indices, false), Variable(offsets_data, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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
            ADD_FAILURE() << "EmbeddingBag_Mean failed on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// LazyLinear
// ============================================================================

TEST_P(NNLinearEmbParity, LazyLinear) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

    nn::LazyLinear layer(32);
    auto input = randn({4, 64}, DType::Float32, Device::cpu());

    // First forward materializes weights on CPU
    auto ref = layer.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
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
            ADD_FAILURE() << "LazyLinear failed on " << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// LazyConv{1,2,3}d  (commit 37be63b4)
// ============================================================================
// LazyConv infers C_in from the first forward pass, materializing the underlying
// Conv*d. The tests verify two things: (a) forward-pass parity after warmup,
// (b) that materialized weight shapes are identical on every backend.

namespace {
// Helper: warmup-materialize the reference LazyConv on CPU, clone params into
// a new LazyConv on the target backend, run and compare.
template <typename LazyConvT>
void lazy_conv_parity_test(
    LazyConvT&& make_layer,
    const std::vector<int64_t>& warmup_shape,
    const std::vector<int64_t>& test_shape,
    const char* name)
{
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn linear/emb parity");

    auto ref_layer = make_layer();
    // Materialize on CPU first so parameters exist to clone.
    auto warmup = randn(warmup_shape, DType::Float32, Device::cpu());
    ref_layer.forward(Variable(warmup, false));
    auto input = randn(test_shape, DType::Float32, Device::cpu());
    auto ref = ref_layer.forward(Variable(input, false)).tensor();

    for (size_t i = first_gpu_index(backends); i < backends.size(); ++i) {
        try {
            auto dev_layer = make_layer();
            // Warm up on CPU (with a scratch tensor of the same C_in) so the
            // child Conv is materialized before we copy params in.
            auto dev_warmup = randn(warmup_shape, DType::Float32, Device::cpu());
            dev_layer.forward(Variable(dev_warmup, false));

            auto params_src = ref_layer.parameters();
            auto params_dst = dev_layer.parameters();
            ASSERT_EQ(params_src.size(), params_dst.size())
                << name << ": materialized param count differs between layers";
            for (size_t p = 0; p < params_src.size(); ++p) {
                auto src_shape = params_src[p]->tensor().shape();
                auto dst_shape = params_dst[p]->tensor().shape();
                std::vector<int64_t> src_vec(src_shape.begin(), src_shape.end());
                std::vector<int64_t> dst_vec(dst_shape.begin(), dst_shape.end());
                ASSERT_EQ(src_vec, dst_vec)
                    << name << ": materialized weight shape mismatch";
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            dev_layer.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = dev_layer.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            SCOPED_TRACE(std::string(name) + " on " + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-3f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << " failed on "
                      << backend_name(backends[i]) << ": " << e.what() << std::endl;
        }
    }
}
}  // namespace

TEST_P(NNLinearEmbParity, LazyConv1d) {
    lazy_conv_parity_test(
        [] { return nn::LazyConv1d(/*out=*/8, /*k=*/3, /*stride=*/1, /*pad=*/1); },
        /*warmup_shape=*/{1, 4, 16},
        /*test_shape=*/{2, 4, 16},
        "LazyConv1d");
}

TEST_P(NNLinearEmbParity, LazyConv2d) {
    lazy_conv_parity_test(
        [] { return nn::LazyConv2d(/*out=*/8, /*k=*/3, /*stride=*/1, /*pad=*/1); },
        /*warmup_shape=*/{1, 4, 8, 8},
        /*test_shape=*/{2, 4, 8, 8},
        "LazyConv2d");
}

TEST_P(NNLinearEmbParity, LazyConv3d) {
    lazy_conv_parity_test(
        [] { return nn::LazyConv3d(/*out=*/4, /*k=*/3, /*stride=*/1, /*pad=*/1); },
        /*warmup_shape=*/{1, 2, 4, 4, 4},
        /*test_shape=*/{1, 2, 4, 4, 4},
        "LazyConv3d");
}

INSTANTIATE_BACKEND_TESTS(NNLinearEmbParity);


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
