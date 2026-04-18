/**
 * @file test_nn_norm_parity.cpp
 * @brief Backend parity tests for normalization layers
 *
 * Tests InstanceNorm1d/2d/3d, RMSNorm, LocalResponseNorm, BatchNorm2d
 * (no affine), LayerNorm (multi-dim), and SyncBatchNorm across all
 * available backends.
 */

#include <gtest/gtest.h>
#include <iostream>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class NNNormParity : public BackendTest {};
// ============================================================================
// Instance Normalization
// ============================================================================

TEST_P(NNNormParity, InstanceNorm1d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::InstanceNorm1d in1d(16);
    in1d.eval();
    auto input = randn({4, 16, 32}, DType::Float32, Device::cpu());
    auto ref = in1d.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::InstanceNorm1d in1d_dev(16);
            in1d_dev.eval();
            auto params_src = in1d.parameters();
            auto params_dst = in1d_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            in1d_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = in1d_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "InstanceNorm1d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNNormParity, InstanceNorm2d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::InstanceNorm2d in2d(16);
    in2d.eval();
    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    auto ref = in2d.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::InstanceNorm2d in2d_dev(16);
            in2d_dev.eval();
            auto params_src = in2d.parameters();
            auto params_dst = in2d_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            in2d_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = in2d_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "InstanceNorm2d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNNormParity, InstanceNorm3d) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::InstanceNorm3d in3d(16);
    in3d.eval();
    auto input = randn({4, 16, 4, 4, 4}, DType::Float32, Device::cpu());
    auto ref = in3d.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::InstanceNorm3d in3d_dev(16);
            in3d_dev.eval();
            auto params_src = in3d.parameters();
            auto params_dst = in3d_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            in3d_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = in3d_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-3f, 1e-4f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "InstanceNorm3d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// RMSNorm
// ============================================================================

TEST_P(NNNormParity, RMSNorm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::RMSNorm rms(32);
    rms.eval();
    auto input = randn({4, 8, 32}, DType::Float32, Device::cpu());
    auto ref = rms.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::RMSNorm rms_dev(32);
            rms_dev.eval();
            auto params_src = rms.parameters();
            auto params_dst = rms_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            rms_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = rms_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "RMSNorm failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// LocalResponseNorm
// ============================================================================

TEST_P(NNNormParity, LocalResponseNorm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::LocalResponseNorm lrn(5);
    lrn.eval();
    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    auto ref = lrn.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LocalResponseNorm lrn_dev(5);
            lrn_dev.eval();
            // LRN has no learnable parameters, but copy any if present
            auto params_src = lrn.parameters();
            auto params_dst = lrn_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            lrn_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = lrn_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LocalResponseNorm failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// BatchNorm2d without affine parameters
// ============================================================================

TEST_P(NNNormParity, BatchNorm2d_NoAffine) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // affine=false: no learnable weight/bias, only running stats
    nn::BatchNorm2d bn(16, 1e-5, 0.1, /*affine=*/false);
    // Run a training forward to populate running stats
    auto train_input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    bn.train();
    bn.forward(Variable(train_input, false));
    bn.eval();

    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    auto ref = bn.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::BatchNorm2d bn_dev(16, 1e-5, 0.1, /*affine=*/false);
            bn_dev.train();
            auto params_src = bn.parameters();
            auto params_dst = bn_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            bn_dev.to(backends[i]);
            // Run training to populate running stats
            auto train_dev = train_input.to(backends[i]);
            bn_dev.forward(Variable(train_dev, false));
            bn_dev.eval();

            auto input_dev = input.to(backends[i]);
            auto output = bn_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "BatchNorm2d_NoAffine failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// LayerNorm with multi-dimensional normalized shape
// ============================================================================

TEST_P(NNNormParity, LayerNorm_MultiDim) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    nn::LayerNorm ln({8, 32});
    ln.eval();
    auto input = randn({4, 8, 32}, DType::Float32, Device::cpu());
    auto ref = ln.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::LayerNorm ln_dev({8, 32});
            ln_dev.eval();
            auto params_src = ln.parameters();
            auto params_dst = ln_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            ln_dev.to(backends[i]);
            auto input_dev = input.to(backends[i]);
            auto output = ln_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LayerNorm_MultiDim failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// ============================================================================
// SyncBatchNorm (single-GPU: behaves like regular BatchNorm)
// ============================================================================

TEST_P(NNNormParity, SyncBatchNorm) {
    auto backends = get_available_backends();
    if (backends.size() < 2) GTEST_SKIP();

    // Identity all-reduce (single process): no-op
    auto identity_all_reduce = [](Tensor& /*tensor*/) {};

    nn::SyncBatchNorm sbn(16, identity_all_reduce, /*world_size=*/1);
    // Run a training forward to populate running stats
    auto train_input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    sbn.train();
    sbn.forward(Variable(train_input, false));
    sbn.eval();

    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    auto ref = sbn.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            nn::SyncBatchNorm sbn_dev(16, identity_all_reduce, /*world_size=*/1);
            sbn_dev.train();
            auto params_src = sbn.parameters();
            auto params_dst = sbn_dev.parameters();
            for (size_t p = 0; p < params_src.size(); ++p) {
                params_dst[p]->tensor() = params_src[p]->tensor().clone();
            }
            sbn_dev.to(backends[i]);
            // Run training to populate running stats
            auto train_dev = train_input.to(backends[i]);
            sbn_dev.forward(Variable(train_dev, false));
            sbn_dev.eval();

            auto input_dev = input.to(backends[i]);
            auto output = sbn_dev.forward(Variable(input_dev, false)).tensor();
            backends[i].synchronize();
            EXPECT_TENSORS_CLOSE(ref, output, 1e-4f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "SyncBatchNorm failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

INSTANTIATE_BACKEND_TESTS(NNNormParity);




int main(int argc, char** argv) {
    try {
        tenzor::initialize();
    } catch (const std::exception& e) {
        std::cerr << "Failed to initialize Tenzor: " << e.what() << std::endl;
    }
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    try {
        tenzor::finalize();
    } catch (...) {}
    return result;
}
