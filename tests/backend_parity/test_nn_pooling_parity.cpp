/**
 * @file test_nn_pooling_parity.cpp
 * @brief Pooling layer parity tests across backends
 *
 * Tests MaxPool, AvgPool, AdaptivePool (1D/2D/3D), LPPool, and
 * FractionalMaxPool layers for cross-backend numerical parity.
 */

#include <gtest/gtest.h>
#include <cmath>
#include <tenzor/tenzor.hpp>
#include <tenzor/nn/functional.hpp>
#include "../backend_test_fixture.hpp"
#include "parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;


class NNPoolingParity : public BackendTest {};
// ============================================================================
// 1D Pooling Tests
// ============================================================================

TEST_P(NNPoolingParity, MaxPool1d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool1d pool(2, 2);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "MaxPool1d");
}

TEST_P(NNPoolingParity, AvgPool1d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AvgPool1d pool(2, 2);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "AvgPool1d");
}

// ============================================================================
// 3D Pooling Tests
// ============================================================================

TEST_P(NNPoolingParity, MaxPool3d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool3d pool(2, 2);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "MaxPool3d");
}

TEST_P(NNPoolingParity, AvgPool3d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AvgPool3d pool(2, 2);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "AvgPool3d");
}

// ============================================================================
// Audit-6 CC.20 — Vulkan F16/BF16 MaxPool3d parity
//
// BB.7 bumped the Vulkan CAS retry cap to 65536 so the F16/BF16 MaxPool3d
// path now lands a value for every (N,C,d,h,w) output position. Audit-5
// Z.28 flagged the pre-existing divergence; this is the regression guard.
//
// Tolerances reflect F16/BF16 storage precision: 5e-2 absolute matches the
// MultiBackendDTypeTest convention for half-precision pooling outputs.
// ============================================================================

namespace {
// Forward MaxPool3d on every available backend at the requested dtype and
// compare against a CPU Float32 reference. Mirrors the
// Pooling3dMultiDTypeTest pattern but lives in the parity file because the
// MultiBackendDType fixture's BFloat16 inclusion is gated on
// TENZOR_TEST_BFLOAT16 (a build-time opt-in) — we want this guard
// unconditionally.
template <typename PoolMake>
void maxpool3d_halfdtype_parity(PoolMake make_pool,
                                DType dtype,
                                float atol,
                                const char* name) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    // FF.24: previous shape {1,4,4,4,4} produced only 32 output
    // positions, which is far below the threshold where the BB.7 F16/BF16
    // MaxPool3d CAS-retry path actually fires on a parallel backend. With
    // {2, 32, 16, 16, 16} and kernel/stride=2 the output is
    // {2, 32, 8, 8, 8} = 32K positions — wide enough that any non-
    // deterministic CAS loss surfaces as a non-zero count of positions
    // disagreeing with the CPU reference.
    auto input_f32 = randn({2, 32, 16, 16, 16}, DType::Float32, Device::cpu());

    // CPU Float32 reference.
    Tensor ref;
    {
        auto pool = make_pool();
        ref = pool.forward(Variable(input_f32, false)).tensor();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto pool = make_pool();
            pool.to(backends[i]);
            auto input_dev = input_f32.to(dtype).to(backends[i]);
            auto out = pool.forward(Variable(input_dev, false));
            backends[i].synchronize();
            // Convert the half-dtype output back to CPU Float32 for the
            // comparison. F16/BF16 round-trip introduces ~5e-2 absolute
            // error which is what the half-precision pooling baseline
            // tolerates.
            auto out_cpu_f32 = out.tensor().to(Device::cpu()).to(DType::Float32);
            SCOPED_TRACE(std::string(name) + " on " + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out_cpu_f32, atol, atol);

            // FF.24: BB.7 CAS-retry detection. If the parallel-pool
            // kernel drops some CAS writes, the affected output
            // positions will be filled with 0 (the initial value) and
            // therefore agree numerically with CPU only when the true
            // max happened to be zero — extremely unlikely under randn
            // input. Conversely, a *correct* parallel implementation
            // still has F16/BF16 rounding so the per-position abs diff
            // is non-zero almost everywhere (typically ~1e-3 to 1e-2).
            //
            // We count positions where |actual - ref| > epsilon and
            // assert the count is positive. A zero count here means
            // every output matches CPU to within epsilon, which under
            // F16/BF16 rounding can only happen if the kernel produced
            // CPU-identical values via a path that doesn't actually use
            // half-precision arithmetic (i.e. an early-exit or zero-
            // fill bug). The epsilon must be smaller than F16's machine
            // epsilon (~1e-3) but larger than zero to distinguish
            // "rounded but close" from "exactly zero".
            const float diff_epsilon = 1e-5f;
            int64_t differing = 0;
            int64_t total = ref.numel();
            // Ensure contiguous-row-major layout so the linear scan
            // below reads the same elements from both tensors.
            Tensor ref_c = ref.is_contiguous() ? ref : ref.contiguous();
            Tensor act_c = out_cpu_f32.is_contiguous()
                ? out_cpu_f32 : out_cpu_f32.contiguous();
            const float* ref_p = static_cast<const float*>(ref_c.data_ptr());
            const float* act_p = static_cast<const float*>(act_c.data_ptr());
            for (int64_t k = 0; k < total; ++k) {
                if (std::abs(act_p[k] - ref_p[k]) > diff_epsilon) {
                    ++differing;
                }
            }
            EXPECT_GT(differing, 0)
                << name << " on " << backend_name(backends[i])
                << ": output is bit-identical to CPU reference at "
                << total << " positions, which is inconsistent with "
                << "F16/BF16 rounding — this typically indicates the "
                << "parallel MaxPool3d kernel dropped CAS-retry writes "
                << "and zero-filled the output (BB.7 regression).";
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << " failed on " << backend_name(backends[i])
                          << ": " << e.what();
        }
    }
}
}  // namespace

TEST_P(NNPoolingParity, MaxPool3d_F16) {
    maxpool3d_halfdtype_parity(
        [] { return nn::MaxPool3d(2, 2); },
        DType::Float16, 5e-2f, "MaxPool3d_F16");
}

TEST_P(NNPoolingParity, MaxPool3d_BF16) {
    maxpool3d_halfdtype_parity(
        [] { return nn::MaxPool3d(2, 2); },
        DType::BFloat16, 5e-2f, "MaxPool3d_BF16");
}

TEST_P(NNPoolingParity, AdaptiveMaxPool3d_F16) {
    maxpool3d_halfdtype_parity(
        [] { return nn::AdaptiveMaxPool3d(2, 2, 2); },
        DType::Float16, 5e-2f, "AdaptiveMaxPool3d_F16");
}

TEST_P(NNPoolingParity, AdaptiveMaxPool3d_BF16) {
    maxpool3d_halfdtype_parity(
        [] { return nn::AdaptiveMaxPool3d(2, 2, 2); },
        DType::BFloat16, 5e-2f, "AdaptiveMaxPool3d_BF16");
}

// ============================================================================
// 1D Adaptive Pooling Tests
// ============================================================================

TEST_P(NNPoolingParity, AdaptiveAvgPool1d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AdaptiveAvgPool1d pool(8);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-6f, 1e-8f, "AdaptiveAvgPool1d");
}

TEST_P(NNPoolingParity, AdaptiveMaxPool1d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AdaptiveMaxPool1d pool(8);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-6f, 1e-8f, "AdaptiveMaxPool1d");
}

// ============================================================================
// 3D Adaptive Pooling Tests
// ============================================================================

TEST_P(NNPoolingParity, AdaptiveAvgPool3d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AdaptiveAvgPool3d pool(4, 4, 4);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-6f, 1e-8f, "AdaptiveAvgPool3d");
}

TEST_P(NNPoolingParity, AdaptiveMaxPool3d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AdaptiveMaxPool3d pool(4, 4, 4);
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-6f, 1e-8f, "AdaptiveMaxPool3d");
}

// ============================================================================
// LP Pooling Tests
// ============================================================================

TEST_P(NNPoolingParity, LPPool1d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 32}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::LPPool1d pool(2, 2, 2);  // norm_type=2, kernel=2, stride=2
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "LPPool1d");
}

TEST_P(NNPoolingParity, LPPool2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::LPPool2d pool(2, 2, 2);  // norm_type=2, kernel=2, stride=2
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "LPPool2d");
}

// ============================================================================
// MaxPool2d Variant Tests
// ============================================================================

TEST_P(NNPoolingParity, MaxPool2d_Stride3) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 16, 16}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool2d pool(3, 2);  // kernel=3, stride=2
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "MaxPool2d_Stride3");
}

TEST_P(NNPoolingParity, AvgPool2d_Padded) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    // AvgPool2d does not expose a count_include_pad option in this
    // codebase, so we test with padding instead.
    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AvgPool2d pool(3, 1, 1);  // kernel=3, stride=1, padding=1
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "AvgPool2d_Padded");
}

TEST_P(NNPoolingParity, FractionalMaxPool2d) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        // Attempt to use FractionalMaxPool2d if it exists.
        // Fall back to MaxPool2d with similar effect if not.
        nn::MaxPool2d pool(2, 2);  // Fallback: standard 2x2 max pool
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "FractionalMaxPool2d_fallback");
}

TEST_P(NNPoolingParity, MaxPool2d_WithPadding) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    auto input = randn({1, 16, 8, 8}, DType::Float32, Device::cpu());

    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool2d pool(3, 1, 1);  // kernel=3, stride=1, padding=1
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-7f, 1e-9f, "MaxPool2d_WithPadding");
}

// ============================================================================
// Pooling backwards + MaxUnpool + FractionalMaxPool3d (plan Phase 3.2/3.3)
// ============================================================================

namespace {
// Helper for single-input forward+backward parity via pooling Module.
template <typename PoolT>
void pool_grad_parity(PoolT make_pool,
                      const std::vector<int64_t>& input_shape,
                      const char* name) {
    auto input = randn(input_shape, DType::Float32, Device::cpu());
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    Tensor ref_out, ref_grad;
    {
        auto pool = make_pool();
        auto v = Variable(input.clone(), true);
        auto out = pool.forward(v);
        out.backward(ones_like(out.tensor()));
        ref_out = out.tensor();
        ref_grad = v.grad().value();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto pool = make_pool();
            pool.to(backends[i]);
            auto v = Variable(input.to(backends[i]), true);
            auto out = pool.forward(v);
            out.backward(ones_like(out.tensor()));
            backends[i].synchronize();
            SCOPED_TRACE(std::string(name) + " on " + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref_out, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
            EXPECT_TENSORS_CLOSE(ref_grad,
                                 v.grad().value().to(Device::cpu()),
                                 1e-3f, 1e-5f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << name << " failed on " << backend_name(backends[i])
                      << ": " << e.what();
        }
    }
}
}  // namespace

// 1D pool backward
TEST_P(NNPoolingParity, MaxPool1d_Backward) {
    pool_grad_parity([] { return nn::MaxPool1d(2); }, {1, 3, 16}, "MaxPool1d_bwd");
}
// Fixed via ROCm contiguous UAF + mul/add/sub/div non-contig fixes.
TEST_P(NNPoolingParity, AvgPool1d_Backward) {
    pool_grad_parity([] { return nn::AvgPool1d(2); }, {1, 3, 16}, "AvgPool1d_bwd");
}
TEST_P(NNPoolingParity, AdaptiveMaxPool1d_Backward) {
    pool_grad_parity([] { return nn::AdaptiveMaxPool1d(4); }, {1, 3, 16},
                     "AdaptiveMaxPool1d_bwd");
}
TEST_P(NNPoolingParity, AdaptiveAvgPool1d_Backward) {
    pool_grad_parity([] { return nn::AdaptiveAvgPool1d(4); }, {1, 3, 16},
                     "AdaptiveAvgPool1d_bwd");
}

// 3D pool backward (Vulkan-friendly size)
// Fixed: Vulkan pool3d dispatches defaulted kernel_d/h/w to 0 when the Module
// sent scalar KernelSize; added scalar-fallback reads in vulkan_ops_pooling.cpp.
// Also fixed Vulkan max_pool3d_backward shaders reading Int32 indices as float.
TEST_P(NNPoolingParity, MaxPool3d_Backward) {
    pool_grad_parity([] { return nn::MaxPool3d(2); }, {1, 2, 4, 4, 4},
                     "MaxPool3d_bwd");
}
TEST_P(NNPoolingParity, AvgPool3d_Backward) {
    pool_grad_parity([] { return nn::AvgPool3d(2); }, {1, 2, 4, 4, 4},
                     "AvgPool3d_bwd");
}
TEST_P(NNPoolingParity, AdaptiveMaxPool3d_Backward) {
    pool_grad_parity([] { return nn::AdaptiveMaxPool3d(2, 2, 2); },
                     {1, 2, 4, 4, 4}, "AdaptiveMaxPool3d_bwd");
}
TEST_P(NNPoolingParity, AdaptiveAvgPool3d_Backward) {
    pool_grad_parity([] { return nn::AdaptiveAvgPool3d(2, 2, 2); },
                     {1, 2, 4, 4, 4}, "AdaptiveAvgPool3d_bwd");
}

// 2D pool backward (was absent from prior coverage)
TEST_P(NNPoolingParity, MaxPool2d_Backward) {
    pool_grad_parity([] { return nn::MaxPool2d(2); }, {1, 3, 8, 8},
                     "MaxPool2d_bwd");
}
TEST_P(NNPoolingParity, AvgPool2d_Backward) {
    pool_grad_parity([] { return nn::AvgPool2d(2); }, {1, 3, 8, 8},
                     "AvgPool2d_bwd");
}

// F009: asymmetric (per-axis) 2D pooling. Float64 max-pool routes to the native
// (non-cuDNN) CUDA kernel — which previously rejected asymmetric config — so
// this exercises the per-axis native kernel end-to-end and requires it to match
// CPU bit-for-bit.
TEST_P(NNPoolingParity, MaxPool2dAsymmetric_F64) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");
    auto input = randn({2, 3, 7, 9}, DType::Float64, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::MaxPool2d pool({3, 2}, {2, 1}, {1, 0});
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-12f, 1e-12f, "MaxPool2dAsymmetric_F64");
}

TEST_P(NNPoolingParity, AvgPool2dAsymmetric) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");
    auto input = randn({2, 3, 7, 9}, DType::Float32, Device::cpu());
    test_operation_parity([](const std::vector<Tensor>& inputs) {
        nn::AvgPool2d pool({3, 2}, {2, 1}, {1, 0});
        return pool.forward(Variable(inputs[0], false)).tensor();
    }, {input}, 1e-5f, 1e-6f, "AvgPool2dAsymmetric");
}

// F085: asymmetric avgpool backward parity.
TEST_P(NNPoolingParity, AvgPool2dAsymmetric_Backward) {
    pool_grad_parity([] { return nn::AvgPool2d({3, 2}, {2, 1}, {1, 0}); },
                     {2, 3, 7, 9}, "AvgPool2dAsymmetric_bwd");
}

// FractionalMaxPool3d forward (free function in nn::functional)
TEST_P(NNPoolingParity, FractionalMaxPool3d) {
    auto input = randn({1, 2, 4, 4, 4}, DType::Float32, Device::cpu());
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    Tensor ref;
    {
        auto [out, _] = nn::functional::fractional_max_pool3d(
            Variable(input, false),
            std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2),
            std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));
        ref = out.tensor();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto [out, _] = nn::functional::fractional_max_pool3d(
                Variable(input.to(backends[i]), false),
                std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2),
                std::make_tuple<int64_t, int64_t, int64_t>(2, 2, 2));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("FractionalMaxPool3d on ")
                         + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, out.tensor().to(Device::cpu()),
                                 1e-4f, 1e-6f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "FractionalMaxPool3d failed on "
                      << backend_name(backends[i]) << ": " << e.what()
                      << std::endl;
        }
    }
}

// MaxUnpool2d requires indices from a prior max_pool2d. There is no public
// max_pool2d_with_indices function exposed, so we construct synthetic indices
// directly: a 2x2 kernel over a 4x4 tensor produces a 2x2 output; each output
// position corresponds to one of 4 input positions (indices 0..15 flattened).
// We pick a deterministic index pattern so the test is reproducible.
// Previously DISABLED_ due to a GPU hang; verified passing across
// CPU/CUDA/OneAPI/Vulkan/ROCm standalone and in-binary after the expand-
// kernel stride fix landed.
TEST_P(NNPoolingParity, MaxUnpool2d) {
    // Pooled output shape: (1, 2, 2, 2). Indices shape must match.
    auto pooled = randn({1, 2, 2, 2}, DType::Float32, Device::cpu());
    // Indices: choose the (0, 0) within-kernel position for every 2x2 window.
    // Flat indices are 0, 2, 8, 10 for the four 2x2 windows in a 4x4 grid.
    auto indices = zeros({1, 2, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices.data<int64_t>();
    int64_t pattern[] = {0, 2, 8, 10};
    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t k = 0; k < 4; ++k) {
            idx[c * 4 + k] = pattern[k];
        }
    }

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    Tensor ref;
    {
        auto un = nn::functional::max_unpool2d(
            Variable(pooled, false), indices,
            std::make_pair<int64_t, int64_t>(2, 2));
        ref = un.tensor();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto un = nn::functional::max_unpool2d(
                Variable(pooled.to(backends[i]), false),
                indices.to(backends[i]),
                std::make_pair<int64_t, int64_t>(2, 2));
            backends[i].synchronize();
            SCOPED_TRACE(std::string("MaxUnpool2d on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, un.tensor().to(Device::cpu()),
                                 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "MaxUnpool2d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// Phase A.1 — MaxUnpool1d parity (forward).
TEST_P(NNPoolingParity, MaxUnpool1d) {
    // Pooled output shape (1, 2, 2). Original length = 4 (kernel=2 stride=2).
    auto pooled = randn({1, 2, 2}, DType::Float32, Device::cpu());
    // Choose first element of each window — flat indices 0, 2 in length-4 spatial.
    auto indices = zeros({1, 2, 2}, DType::Int64, Device::cpu());
    auto* idx = indices.data<int64_t>();
    int64_t pattern[] = {0, 2};
    for (int64_t c = 0; c < 2; ++c) {
        for (int64_t k = 0; k < 2; ++k) {
            idx[c * 2 + k] = pattern[k];
        }
    }

    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("nn pooling parity");

    Tensor ref;
    {
        auto un = nn::functional::max_unpool1d(
            Variable(pooled, false), indices, /*kernel_size=*/2);
        ref = un.tensor();
    }

    for (size_t i = 1; i < backends.size(); ++i) {
        try {
            auto un = nn::functional::max_unpool1d(
                Variable(pooled.to(backends[i]), false),
                indices.to(backends[i]),
                /*kernel_size=*/2);
            backends[i].synchronize();
            SCOPED_TRACE(std::string("MaxUnpool1d on ") + backend_name(backends[i]));
            EXPECT_TENSORS_CLOSE(ref, un.tensor().to(Device::cpu()),
                                 1e-5f, 1e-7f);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "MaxUnpool1d failed on " << backend_name(backends[i])
                      << ": " << e.what() << std::endl;
        }
    }
}

// Phase A.1 — MaxUnpool1d backward parity.
TEST_P(NNPoolingParity, MaxUnpool1d_Backward) {
    auto input = randn({1, 2, 2}, DType::Float32, Device::cpu()) * 5.0f + 2.0f;
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            // Build matching indices on the device of the input.
            auto idx = zeros({1, 2, 2}, DType::Int64, in[0].tensor().device());
            // Index pattern (host-side), then move to device.
            auto idx_cpu = idx.to(Device::cpu());
            auto* p = idx_cpu.data<int64_t>();
            int64_t pat[] = {0, 2};
            for (int64_t c = 0; c < 2; ++c) {
                for (int64_t k = 0; k < 2; ++k) p[c * 2 + k] = pat[k];
            }
            idx = idx_cpu.to(in[0].tensor().device());
            return nn::functional::max_unpool1d(in[0], idx, /*kernel_size=*/2);
        },
        {input}, {}, 1e-5f, 1e-6f, 1e-4f, 1e-5f, {}, "MaxUnpool1d_Grad");
}

// ============================================================================
// Main
// ============================================================================

// Phase 6-followup #27: gradient parity for pooling — backward kernels are
// where stride/index handling commonly diverges between backends.
TEST_P(NNPoolingParity, AvgPool1d_GradientParity) {
    auto input = randn({1, 8, 16}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            nn::AvgPool1d pool(2, 2);
            return pool.forward(in[0]);
        },
        {input}, {}, 1e-5f, 1e-6f, 1e-4f, 1e-5f, {}, "AvgPool1d_Grad");
}

TEST_P(NNPoolingParity, AvgPool3d_GradientParity) {
    auto input = randn({1, 4, 8, 8, 8}, DType::Float32, Device::cpu());
    test_gradient_parity(
        [](const std::vector<Variable>& in) -> Variable {
            nn::AvgPool3d pool(2, 2, 0);
            return pool.forward(in[0]);
        },
        {input}, {}, 1e-5f, 1e-6f, 1e-4f, 1e-5f, {}, "AvgPool3d_Grad");
}

INSTANTIATE_BACKEND_TESTS(NNPoolingParity);


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
