/**
 * @file test_nn_norm_parity.cpp
 * @brief Backend parity tests for normalization layers
 *
 * Tests InstanceNorm1d/2d/3d, RMSNorm, LocalResponseNorm, BatchNorm2d
 * (no affine), LayerNorm (multi-dim), and SyncBatchNorm across all
 * available backends.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <iomanip>
#include <iostream>
#include <vector>
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
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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

namespace {
// Builds a RMSNorm input row of uniform-magnitude pseudo-random values
// (deterministic, not dependent on global RNG state). Deliberately NOT an
// "outlier-heavy" distribution: a few large-magnitude values dominate the
// sum of squares and make whatever small-magnitude values are lost to
// float32 rounding irrelevant in *relative* terms (verified empirically --
// an outlier-heavy version of this input failed to expose the OneAPI/Vulkan
// accumulation bugs at any practical hidden_dim). With uniform magnitude,
// naive float32 accumulation drifts from the true sum by the standard
// O(sqrt(hidden)) ULPs, which is what actually separates the buggy
// float32 accumulators from a double/Kahan one at a large enough hidden_dim.
auto make_uniform_magnitude_input(int64_t batch, int64_t hidden) -> Tensor {
    auto t = zeros({batch, hidden}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t i = 0; i < hidden; ++i) {
            uint32_t bits = static_cast<uint32_t>((b * 2654435761u) ^ (static_cast<uint32_t>(i) * 40503u));
            float unit = (static_cast<float>(bits & 0xFFFFu) / 65536.0f) - 0.5f;  // [-0.5, 0.5)
            p[b * hidden + i] = unit * 2.0f;  // [-1, 1)
        }
    }
    return t;
}

// Computes the "true" RMSNorm output for a single row by accumulating the
// sum of squares in double precision (independent of any backend
// implementation), so the expected reference cannot itself be contaminated
// by a float32 accumulation bug in whichever backend happens to be used as
// backends[0] in the cross-backend comparison loop.
auto rms_norm_reference_row(const float* row, const float* weight, int64_t hidden, double eps) -> std::vector<float> {
    double ss = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
        double v = static_cast<double>(row[i]);
        ss += v * v;
    }
    double rrms = 1.0 / std::sqrt(ss / static_cast<double>(hidden) + eps);
    std::vector<float> out(hidden);
    for (int64_t i = 0; i < hidden; ++i) {
        out[i] = static_cast<float>(static_cast<double>(row[i]) * rrms * static_cast<double>(weight[i]));
    }
    return out;
}
// Computes the "true" LayerNorm output for a single row by accumulating
// mean and variance in double precision (independent of any backend
// implementation), so the expected reference cannot itself be contaminated
// by a float32 accumulation bug in whichever backend happens to be used as
// backends[0] in the cross-backend comparison loop. Mirrors
// rms_norm_reference_row's rationale/structure above.
auto layer_norm_reference_row(const float* row, const float* weight, const float* bias,
                              int64_t hidden, double eps) -> std::vector<float> {
    double sum = 0.0;
    for (int64_t i = 0; i < hidden; ++i) sum += static_cast<double>(row[i]);
    double mean = sum / static_cast<double>(hidden);
    double var_sum = 0.0;
    for (int64_t i = 0; i < hidden; ++i) {
        double d = static_cast<double>(row[i]) - mean;
        var_sum += d * d;
    }
    double inv_std = 1.0 / std::sqrt(var_sum / static_cast<double>(hidden) + eps);
    std::vector<float> out(hidden);
    for (int64_t i = 0; i < hidden; ++i) {
        double normalized = (static_cast<double>(row[i]) - mean) * inv_std;
        out[i] = static_cast<float>(normalized * static_cast<double>(weight[i]) +
                                    static_cast<double>(bias[i]));
    }
    return out;
}
}  // namespace

// JIT-R153/JIT-R154/JIT-R164: RMSNorm's sum-of-squares accumulation drifted
// from a float32 accumulator to a double-accumulating reference (CPU's
// trained/grad path, CUDA, ROCm, MLIR) in OneAPI, CPU's own no-grad
// "ultra-fast" inference path, and Vulkan's GLSL shaders -- independently,
// in three different implementations. The drift is the standard
// O(sqrt(hidden)) float32 rounding error and only becomes large enough to
// matter at a large hidden_dim, so this test uses hidden_dim=2097152 with
// uniform-magnitude random data. Empirically on this codebase/hardware, the
// pre-fix CPU fast path drifts by ~1e-3 (a huge, unmissable margin above the
// ~1e-7 cross-backend noise floor); OneAPI/Vulkan's declared float32
// accumulators happened not to show a measurable gap on this particular
// compiler/hardware (likely FP-contraction keeping intermediate precision
// wider than the declared type), but the source-level fix (explicit
// double / Kahan accumulation) is still the correct, portable fix and is
// exercised (without regressing) by this same test.
TEST_P(NNNormParity, RMSNorm_LargeHiddenDim_AccumulationPrecision) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

    constexpr int64_t kHidden = 2097152;
    constexpr int64_t kBatch = 2;
    constexpr double kEps = 1e-6;
    nn::RMSNorm rms(kHidden, kEps);
    rms.eval();
    auto input = make_uniform_magnitude_input(kBatch, kHidden);
    auto weight = rms.parameters()[0]->tensor().contiguous();

    // Backend-independent expected output: accumulates sum-of-squares in
    // double precision directly from the raw input, so this reference can't
    // itself be contaminated by a float32 accumulation bug in whichever
    // backend happens to be first in `backends` (using another backend's
    // *output* as "ref" previously let a bug in that backend hide an
    // equally-sized bug in the one being compared against it).
    std::vector<float> expected(kBatch * kHidden);
    {
        auto* in_p = input.data<float>();
        auto* w_p = weight.data<float>();
        for (int64_t b = 0; b < kBatch; ++b) {
            auto row = rms_norm_reference_row(in_p + b * kHidden, w_p, kHidden, kEps);
            std::copy(row.begin(), row.end(), expected.begin() + b * kHidden);
        }
    }
    Tensor expected_t = Tensor::from_blob(expected.data(), {kBatch, kHidden}, DType::Float32).clone();

    // 1e-5 sits comfortably above the ~1e-7 cross-backend rounding noise
    // floor and comfortably below the pre-fix CPU drift (~1e-3), giving a
    // wide margin on both sides.
    for (const auto& dev : backends) {
        try {
            nn::RMSNorm rms_dev(kHidden, kEps);
            rms_dev.eval();
            auto params_dst = rms_dev.parameters();
            params_dst[0]->tensor() = weight.clone();
            rms_dev.to(dev);
            auto input_dev = input.to(dev);
            auto output = rms_dev.forward(Variable(input_dev, false)).tensor();
            dev.synchronize();
            // Non-fatal check (not EXPECT_TENSORS_CLOSE's FAIL()) so one
            // backend failing doesn't abort the loop before later backends
            // are checked -- FAIL() returns from the enclosing function,
            // which previously meant only backends[0] (always "cpu") was
            // ever actually compared.
            if (!tenzor::testing::tensors_close(expected_t, output, 1e-5f, 1e-5f)) {
                float diff = tenzor::testing::max_abs_diff(expected_t, output);
                ADD_FAILURE() << "RMSNorm (large hidden_dim) mismatch on "
                          << backend_name(dev) << ": max abs diff=" << std::scientific << diff;
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "RMSNorm (large hidden_dim) failed on "
                      << backend_name(dev) << ": " << e.what() << std::endl;
        }
    }
}

// JIT-R154 specifically: CPU's no-grad "ultra-fast" inference path
// (fused_rms_norm_f32) must agree with the double-accumulating trained
// (grad-enabled) path for the same input/weights, for a large hidden_dim
// with uniform-magnitude random data (see make_uniform_magnitude_input above).
TEST_P(NNNormParity, RMSNorm_CpuInferenceFastPathMatchesTrainedPath) {
    if (GetParam() != "cpu") {
        GTEST_SKIP() << "CPU-only comparison";
    }
    constexpr int64_t kHidden = 2097152;
    nn::RMSNorm rms(kHidden);
    rms.eval();
    auto input = make_uniform_magnitude_input(2, kHidden);

    // No-grad path: takes fused_rms_norm_f32's "ultra-fast" branch.
    auto out_no_grad = rms.forward(Variable(input, false)).tensor();

    // Grad-enabled path: takes the double-accumulating fused_rms_norm_kernel
    // branch (autograd needs saved stats).
    auto out_grad = rms.forward(Variable(input, true)).tensor();

    // See RMSNorm_LargeHiddenDim_AccumulationPrecision above for why 1e-5 is
    // the right tolerance: the pre-fix CPU float32 accumulator drifts by
    // ~1e-3 at this hidden_dim, far above this threshold, while the fixed
    // double-accumulating fast path agrees with the grad path to ~1e-7.
    EXPECT_TENSORS_CLOSE(out_no_grad, out_grad, 1e-5f, 1e-5f);
}

// ============================================================================
// LocalResponseNorm
// ============================================================================

TEST_P(NNNormParity, LocalResponseNorm) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

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

// cudnn_layer_norm_forward (CUDA, TENZOR_HAS_CUDNN builds) accumulated its
// Float32 mean/variance reduction in float instead of double, a build-flag-
// dependent divergence from the non-cuDNN-build fused_layer_norm_kernel
// (Acc=double) and every other backend for the exact same op. Mirrors
// RMSNorm_LargeHiddenDim_AccumulationPrecision's rationale/structure above:
// a large normalized_shape (matching the finding's own "e.g. 8192" example,
// scaled up further for a robust, clearly-detectable drift signal) with
// uniform-magnitude data is what actually separates a float32 accumulator
// from a double one.
TEST_P(NNNormParity, LayerNorm_LargeNormalizedShape_AccumulationPrecision) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

    constexpr int64_t kHidden = 1048576;
    constexpr int64_t kBatch = 2;
    constexpr double kEps = 1e-5;
    nn::LayerNorm ln({kHidden}, kEps);
    ln.eval();
    auto input = make_uniform_magnitude_input(kBatch, kHidden);
    auto weight = ln.parameters()[0]->tensor().contiguous();
    auto bias = ln.parameters()[1]->tensor().contiguous();

    std::vector<float> expected(kBatch * kHidden);
    {
        auto* in_p = input.data<float>();
        auto* w_p = weight.data<float>();
        auto* b_p = bias.data<float>();
        for (int64_t b = 0; b < kBatch; ++b) {
            auto row = layer_norm_reference_row(in_p + b * kHidden, w_p, b_p, kHidden, kEps);
            std::copy(row.begin(), row.end(), expected.begin() + b * kHidden);
        }
    }
    Tensor expected_t = Tensor::from_blob(expected.data(), {kBatch, kHidden}, DType::Float32).clone();

    for (const auto& dev : backends) {
        try {
            nn::LayerNorm ln_dev({kHidden}, kEps);
            ln_dev.eval();
            auto params_dst = ln_dev.parameters();
            params_dst[0]->tensor() = weight.clone();
            params_dst[1]->tensor() = bias.clone();
            ln_dev.to(dev);
            auto input_dev = input.to(dev);
            auto output = ln_dev.forward(Variable(input_dev, false)).tensor();
            dev.synchronize();
            if (!tenzor::testing::tensors_close(expected_t, output, 1e-5f, 1e-5f)) {
                float diff = tenzor::testing::max_abs_diff(expected_t, output);
                ADD_FAILURE() << "LayerNorm (large normalized_shape) mismatch on "
                          << backend_name(dev) << ": max abs diff=" << std::scientific << diff;
            }
        } catch (const std::exception& e) {
            ADD_FAILURE() << "LayerNorm (large normalized_shape) failed on "
                      << backend_name(dev) << ": " << e.what() << std::endl;
        }
    }
}

TEST_P(NNNormParity, SyncBatchNorm) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn norm parity");

    // Identity all-reduce (single process): no-op
    auto identity_all_reduce = [](Tensor& /*tensor*/) {};

    // intentionally exercising deprecated legacy SyncBatchNorm ctor
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
    nn::SyncBatchNorm sbn(16, identity_all_reduce, /*world_size=*/1);
#pragma GCC diagnostic pop
    // Run a training forward to populate running stats
    auto train_input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    sbn.train();
    sbn.forward(Variable(train_input, false));
    sbn.eval();

    auto input = randn({4, 16, 8, 8}, DType::Float32, Device::cpu());
    auto ref = sbn.forward(Variable(input, false)).tensor();

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            // intentionally exercising deprecated legacy SyncBatchNorm ctor
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
            nn::SyncBatchNorm sbn_dev(16, identity_all_reduce, /*world_size=*/1);
#pragma GCC diagnostic pop
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
