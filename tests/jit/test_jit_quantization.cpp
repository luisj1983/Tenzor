// Cross-backend JIT INT8 quantization parity.
//
// The opt-in QuantizationPass retags a traced Linear -> QuantizedLinear; the JIT
// interpreter executes it via nn::quantization::quantized_linear_dynamic, which
// shares the exact eager nn::QuantizedLinear code path (dynamic per-tensor
// symmetric int8 quantize of weight+activation, then OpId::QuantizedLinear
// dispatch — a real kernel on every backend). This test asserts, on every
// available backend:
//   (a) JIT-quantized output == eager-quantized output (the pass + interpreter
//       wire up correctly and match the eager helper), and
//   (b) the quantized result is within a few percent of the fp32 Linear (the
//       quantization is a faithful approximation, not garbage).
// Plus a cross-backend check that each backend's quantized result agrees.

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include <tenzor/tenzor.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/graph.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/nn/quantization/quantized_layers.hpp>
#include <tenzor/backend/fast_dispatch.hpp>
#include <cmath>

#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

// Copy parameters (by index) from src to dst module.
void copy_params(nn::Module& src, nn::Module& dst) {
    auto sp = src.parameters();
    auto dp = dst.parameters();
    for (size_t p = 0; p < sp.size() && p < dp.size(); ++p) {
        dp[p]->tensor() = sp[p]->tensor().clone();
    }
}

// Relative error on the L2 norm: ||a-b|| / (||a|| + eps).
auto rel_l2(const Tensor& a, const Tensor& b) -> float {
    auto ac = a.to(DType::Float32).to(Device::cpu());
    auto bc = b.to(DType::Float32).to(Device::cpu());
    auto diff = tenzor::sub(ac, bc);
    float num = std::sqrt(tenzor::sum(tenzor::mul(diff, diff)).item<float>());
    float den = std::sqrt(tenzor::sum(tenzor::mul(ac, ac)).item<float>()) + 1e-8f;
    return num / den;
}

}  // namespace

TEST(JITQuantization, QuantizedLinearCrossBackend) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit quantized linear parity");

    const int64_t B = 4, IN = 64, OUT = 32;
    // One shared fp32 Linear (with bias) so every backend quantizes identical
    // weights; a fixed input for determinism.
    nn::Linear fc_ref(IN, OUT, /*bias=*/true);
    auto input_cpu = randn({B, IN}, DType::Float32, Device::cpu());

    std::vector<Tensor> quantized_per_backend;  // collected on CPU for cross-check
    std::vector<std::string> names;

    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            nn::Linear fc(IN, OUT, /*bias=*/true);
            copy_params(fc_ref, fc);
            fc.to(backends[i]);
            auto input = input_cpu.to(backends[i]);

            // Eager fp32 reference and eager-quantized reference (the SAME helper
            // the JIT interpreter uses).
            Tensor fp32 = fc.forward(Variable(input, false)).tensor();
            Tensor q_eager = nn::quantization::quantized_linear_dynamic(
                input, fc.weight()->tensor(),
                std::optional<Tensor>(fc.bias()->tensor()));

            // Trace the fp32 Linear into a graph.
            std::shared_ptr<jit::Graph> graph;
            {
                jit::TracingGuard guard;
                Variable x(input, /*requires_grad=*/false);
                Variable y = fc.forward(x);
                graph = guard.get_graph({x}, {y});
            }
            ASSERT_TRUE(graph != nullptr);

            // Apply ONLY the QuantizationPass (retag Linear -> QuantizedLinear).
            jit::Compiler compiler(/*enable_default_passes=*/false);
            compiler.add_pass(std::make_unique<jit::QuantizationPass>());
            compiler.optimize(*graph);

            // Execute the quantized graph through the interpreter.
            auto outs = graph->forward({Variable(input, /*requires_grad=*/false)});
            ASSERT_FALSE(outs.empty());
            Tensor q_jit = outs[0].tensor();
            backends[i].synchronize();

            SCOPED_TRACE("quantized linear on " + backend_name(backends[i]));

            // (a) JIT-quantized == eager-quantized (same code path).
            EXPECT_TENSORS_CLOSE(q_eager, q_jit, 1e-3f, 1e-3f);

            // (b) quantized ~= fp32 within a few percent (faithful, not garbage).
            EXPECT_LT(rel_l2(fp32, q_jit), 0.1f)
                << "int8-quantized linear diverged too far from fp32 on "
                << backend_name(backends[i]);

            quantized_per_backend.push_back(q_jit.to(Device::cpu()));
            names.push_back(backend_name(backends[i]));
        } catch (const std::exception& e) {
            ADD_FAILURE() << "QuantizedLinearCrossBackend failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }

    // Cross-backend: every backend's quantized result must agree (same weights,
    // same input, same quantization).
    for (size_t a = 0; a + 1 < quantized_per_backend.size(); ++a) {
        SCOPED_TRACE("cross-backend quantized parity " + names[a] + " vs " +
                     names[a + 1]);
        EXPECT_TENSORS_CLOSE(quantized_per_backend[a],
                             quantized_per_backend[a + 1], 2e-3f, 2e-3f);
    }
}

TEST(JITQuantization, QuantizedConv2dCrossBackend) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit quantized conv2d parity");

    const int64_t N = 2, C = 4, H = 8, W = 8, OUT = 8, K = 3;
    nn::Conv2d conv_ref(C, OUT, K, /*stride=*/1, /*padding=*/1);  // square, bias on
    auto input_cpu = randn({N, C, H, W}, DType::Float32, Device::cpu());

    std::vector<Tensor> quantized_per_backend;
    std::vector<std::string> names;

    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            nn::Conv2d conv(C, OUT, K, 1, 1);
            copy_params(conv_ref, conv);
            conv.to(backends[i]);
            auto input = input_cpu.to(backends[i]);

            Tensor fp32 = conv.forward(Variable(input, false)).tensor();

            std::shared_ptr<jit::Graph> graph;
            {
                jit::TracingGuard guard;
                Variable x(input, /*requires_grad=*/false);
                Variable y = conv.forward(x);
                graph = guard.get_graph({x}, {y});
            }
            ASSERT_TRUE(graph != nullptr);

            jit::Compiler compiler(/*enable_default_passes=*/false);
            compiler.add_pass(std::make_unique<jit::QuantizationPass>());
            compiler.optimize(*graph);

            auto outs = graph->forward({Variable(input, /*requires_grad=*/false)});
            ASSERT_FALSE(outs.empty());
            Tensor q_jit = outs[0].tensor();
            backends[i].synchronize();

            SCOPED_TRACE("quantized conv2d on " + backend_name(backends[i]));
            // quantized conv ~= fp32 conv within a few percent.
            EXPECT_LT(rel_l2(fp32, q_jit), 0.1f)
                << "int8-quantized conv2d diverged too far from fp32 on "
                << backend_name(backends[i]);

            quantized_per_backend.push_back(q_jit.to(Device::cpu()));
            names.push_back(backend_name(backends[i]));
        } catch (const std::exception& e) {
            ADD_FAILURE() << "QuantizedConv2dCrossBackend failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }

    for (size_t a = 0; a + 1 < quantized_per_backend.size(); ++a) {
        SCOPED_TRACE("cross-backend quantized conv parity " + names[a] + " vs " +
                     names[a + 1]);
        EXPECT_TENSORS_CLOSE(quantized_per_backend[a],
                             quantized_per_backend[a + 1], 2e-3f, 2e-3f);
    }
}

TEST(JITQuantization, SparseMatMulCrossBackend) {
    auto backends = get_available_backends();
    REQUIRE_MULTI_BACKEND_OR_SKIP("jit sparse matmul parity");

    const int64_t B = 4, IN = 64, OUT = 32;
    nn::Linear fc_ref(IN, OUT, /*bias=*/true);
    // Sparsify the reference weight (~70% zeros, deterministic) so the SparsePass
    // (threshold 0.5) fires. SpMM over from_dense(W) is EXACT, so the JIT sparse
    // result must equal the dense fp32 matmul of the SAME (sparsified) weight.
    {
        auto wc = fc_ref.weight()->tensor().to(Device::cpu()).contiguous();
        float* p = wc.data<float>();
        for (int64_t j = 0; j < wc.numel(); ++j) {
            if (j % 10 < 7) p[j] = 0.0f;
        }
        fc_ref.weight()->tensor() = wc;
    }
    auto input_cpu = randn({B, IN}, DType::Float32, Device::cpu());

    std::vector<Tensor> per_backend;
    std::vector<std::string> names;

    for (size_t i = 0; i < backends.size(); ++i) {
        try {
            nn::Linear fc(IN, OUT, /*bias=*/true);
            copy_params(fc_ref, fc);
            fc.to(backends[i]);
            auto input = input_cpu.to(backends[i]);

            // Dense fp32 reference (of the sparsified weight).
            Tensor fp32 = fc.forward(Variable(input, false)).tensor();

            std::shared_ptr<jit::Graph> graph;
            {
                jit::TracingGuard guard;
                Variable x(input, /*requires_grad=*/false);
                Variable y = fc.forward(x);
                graph = guard.get_graph({x}, {y});
            }
            ASSERT_TRUE(graph != nullptr);

            jit::Compiler compiler(/*enable_default_passes=*/false);
            compiler.add_pass(std::make_unique<jit::SparsePass>(/*threshold=*/0.5f));
            compiler.optimize(*graph);

            auto outs = graph->forward({Variable(input, /*requires_grad=*/false)});
            ASSERT_FALSE(outs.empty());
            Tensor sp_jit = outs[0].tensor();
            backends[i].synchronize();

            SCOPED_TRACE("sparse matmul on " + backend_name(backends[i]));
            // Lossless: sparse SpMM == dense matmul of the same weight.
            EXPECT_TENSORS_CLOSE(fp32, sp_jit, 1e-3f, 1e-3f);

            per_backend.push_back(sp_jit.to(Device::cpu()));
            names.push_back(backend_name(backends[i]));
        } catch (const std::exception& e) {
            ADD_FAILURE() << "SparseMatMulCrossBackend failed on "
                          << backend_name(backends[i]) << ": " << e.what();
        }
    }

    for (size_t a = 0; a + 1 < per_backend.size(); ++a) {
        SCOPED_TRACE("cross-backend sparse parity " + names[a] + " vs " +
                     names[a + 1]);
        EXPECT_TENSORS_CLOSE(per_backend[a], per_backend[a + 1], 2e-3f, 2e-3f);
    }
}

int main(int argc, char** argv) {
    tenzor::initialize();
    ::testing::InitGoogleTest(&argc, argv);
    int result = RUN_ALL_TESTS();
    tenzor::finalize();
    return result;
}

// =========================================================================
// GPU quant-kernel guard/compute regression (JIT-040/041/042/043).
// These directly dispatch the backend kernels with configs the wrappers must
// now handle correctly (fail-loud on unsupported, correct on OutputScale/bias),
// on every AVAILABLE backend. int8 operands via a real symmetric quantize.
// =========================================================================
namespace {
// Quantize an fp32 tensor to int8 (symmetric, per-tensor) and return (q, scale).
std::pair<Tensor, float> quantize_i8(const Tensor& x) {
    Tensor xc = x.to(Device::cpu()).contiguous();
    const float* d = xc.data<float>();
    float amax = 1e-8f;
    for (int64_t i = 0; i < xc.numel(); ++i) amax = std::max(amax, std::fabs(d[i]));
    float scale = amax / 127.0f;
    Tensor q({xc.shape().begin(), xc.shape().end()}, DType::Int8, Device::cpu());
    int8_t* qd = q.data<int8_t>();
    for (int64_t i = 0; i < xc.numel(); ++i) {
        int v = static_cast<int>(std::lround(d[i] / scale));
        qd[i] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
    }
    return {q, scale};
}
}  // namespace

// JIT-R002 (was JIT-041): PerChannelRejectedOnGpu asserted that GPU
// QuantizedLinear THROWS on a per-channel weight scale/zero-point. That
// stopped being true once real per-channel support ("F045") shipped on
// every GPU backend (cuda_kernel_registry.cpp:4642+, oneapi/kernels/
// quantization.cpp:189+, vulkan_ops_linalg.cpp:1798+, the ROCm equivalent) —
// the reject-only test just started failing with no real regression. Per
// JIT-042's pattern, replace the reject check with a genuine numerical-
// parity check: quantize each output channel of the weight independently
// (distinct scales, not the old test's uniform-value placeholder, so the
// per-channel code path is actually exercised rather than coinciding with
// the per-tensor path) and assert every GPU backend matches CPU.
TEST(JITQuantizationGpuKernels, PerChannelQuantizedLinearMatchesCpu_JIT041) {
    REQUIRE_MULTI_BACKEND_OR_SKIP("per-channel quantized linear parity");

    Tensor x_f32 = randn({4, 16}, DType::Float32, Device::cpu());
    Tensor w_f32 = randn({8, 16}, DType::Float32, Device::cpu());
    Tensor bias  = randn({8}, DType::Float32, Device::cpu());
    auto [xq, xs] = quantize_i8(x_f32);

    // Per-output-channel (per-row) symmetric int8 quantization of the weight —
    // each of the 8 rows gets its OWN scale, so the scales genuinely differ.
    Tensor wq({8, 16}, DType::Int8, Device::cpu());
    Tensor w_scales({8}, DType::Float32, Device::cpu());
    Tensor w_zps({8}, DType::Int32, Device::cpu());
    {
        auto w_contig = w_f32.contiguous();
        const float* wd = w_contig.data<float>();
        int8_t* wqd = wq.data<int8_t>();
        float* wsd = w_scales.data<float>();
        int32_t* wzd = w_zps.data<int32_t>();
        for (int64_t r = 0; r < 8; ++r) {
            float amax = 1e-8f;
            for (int64_t c = 0; c < 16; ++c)
                amax = std::max(amax, std::fabs(wd[r * 16 + c]));
            float scale = amax / 127.0f;
            wsd[r] = scale;
            wzd[r] = 0;
            for (int64_t c = 0; c < 16; ++c) {
                int v = static_cast<int>(std::lround(wd[r * 16 + c] / scale));
                wqd[r * 16 + c] = static_cast<int8_t>(std::max(-127, std::min(127, v)));
            }
        }
    }

    auto run = [&](Device dev) {
        std::vector<Tensor> inputs = {xq.to(dev), wq.to(dev), bias.to(dev),
                                      w_scales.to(dev), w_zps.to(dev)};
        OpAttributes attrs;
        attrs.set(AttrKey::InputScale, static_cast<double>(xs));
        return dispatch(OpId::QuantizedLinear, inputs, attrs)[0].to(Device::cpu());
    };

    Tensor cpu_out = run(Device::cpu());
    for (const auto& dev : get_available_backends()) {
        if (dev.type == Device::Type::CPU) continue;
        SCOPED_TRACE(std::string("per-channel quantized linear on ") + backend_name(dev));
        Tensor gpu_out = run(dev);
        EXPECT_TRUE(tensors_close(cpu_out, gpu_out, 5e-2f, 5e-2f))
            << "Per-channel QuantizedLinear diverges from CPU on " << backend_name(dev);
    }
}

// JIT-R002 (was JIT-040): RectangularQuantConvRejected asserted GPU
// QuantizedConv2d THROWS on a rectangular (kH != kW) kernel, and only
// checked Vulkan/ROCm. Real rectangular-kernel support ("F044") shipped on
// Vulkan, ROCm, AND OneAPI (vulkan_kernel_registry.cpp:2498,
// rocm_kernel_registry.cpp:4563, oneapi_kernel_registry.cpp:5185 all say so)
// — the old test both asserted removed behavior AND never covered CUDA/
// OneAPI at all. Replace with a genuine numerical-parity check across every
// available backend (mirroring JIT-042's pattern), closing that coverage gap.
TEST(JITQuantizationGpuKernels, RectangularQuantConvMatchesCpu_JIT040) {
    REQUIRE_MULTI_BACKEND_OR_SKIP("rectangular quantized conv2d parity");

    auto [xq, xs] = quantize_i8(randn({1, 2, 8, 8}, DType::Float32, Device::cpu()));
    auto [wq, ws] = quantize_i8(randn({3, 2, 2, 3}, DType::Float32, Device::cpu()));  // rectangular kH!=kW

    auto run = [&](Device dev) {
        std::vector<Tensor> inputs = {xq.to(dev), wq.to(dev)};
        OpAttributes attrs;
        attrs.set(AttrKey::InputScale, static_cast<double>(xs));
        attrs.set(AttrKey::WeightScaleQ, static_cast<double>(ws));
        return dispatch(OpId::QuantizedConv2d, inputs, attrs)[0].to(Device::cpu());
    };

    Tensor cpu_out = run(Device::cpu());
    for (const auto& dev : get_available_backends()) {
        if (dev.type == Device::Type::CPU) continue;
        SCOPED_TRACE(std::string("rectangular quant-conv on ") + backend_name(dev));
        Tensor gpu_out = run(dev);
        EXPECT_TRUE(tensors_close(cpu_out, gpu_out, 5e-2f, 5e-2f))
            << "Rectangular-kernel QuantizedConv2d diverges from CPU on " << backend_name(dev);
    }
}

TEST(JITQuantizationGpuKernels, VulkanOutputScaleMatchesCpu_JIT042) {
    Device vk{Device::Type::Vulkan, 0};
    bool have_vk = false;
    for (const auto& d : get_available_backends())
        if (d.type == Device::Type::Vulkan) { vk = d; have_vk = true; }
    if (!have_vk) GTEST_SKIP() << "Vulkan not available";

    auto [xq, xs] = quantize_i8(randn({4, 16}, DType::Float32, Device::cpu()));
    auto [wq, ws] = quantize_i8(randn({8, 16}, DType::Float32, Device::cpu()));
    // A NON-ZERO bias exercises the bias/output_scale interaction: the shader adds
    // bias at natural scale, so the /output_scale correction must NOT scale it.
    Tensor bias_cpu = randn({8}, DType::Float32, Device::cpu());
    const double out_scale = 2.0;  // non-unit output scale exercises the /out_scale path
    auto run = [&](Device dev) {
        std::vector<Tensor> inputs = {xq.to(dev), wq.to(dev), bias_cpu.to(dev)};
        OpAttributes attrs;
        attrs.set(AttrKey::InputScale, static_cast<double>(xs));
        attrs.set(AttrKey::WeightScaleQ, static_cast<double>(ws));
        attrs.set(AttrKey::OutputScale, out_scale);
        return dispatch(OpId::QuantizedLinear, inputs, attrs)[0].to(Device::cpu());
    };
    Tensor cpu_out = run(Device::cpu());
    Tensor vk_out  = run(vk);
    // With JIT-042, Vulkan now applies /output_scale like CPU, so they match.
    EXPECT_TRUE(tensors_close(cpu_out, vk_out, 2e-3f, 2e-3f))
        << "Vulkan QuantizedLinear OutputScale diverges from CPU (JIT-042)";
}

TEST(JITQuantizationGpuKernels, OneApiEmptyBiasNoCrash_JIT043) {
    Device oa{Device::Type::OneAPI, 0};
    bool have = false;
    for (const auto& d : get_available_backends())
        if (d.type == Device::Type::OneAPI) { oa = d; have = true; }
    if (!have) GTEST_SKIP() << "OneAPI not available";

    auto [xq, xs] = quantize_i8(randn({4, 16}, DType::Float32, Device::cpu()));
    auto [wq, ws] = quantize_i8(randn({8, 16}, DType::Float32, Device::cpu()));
    OpAttributes attrs;
    attrs.set(AttrKey::InputScale, static_cast<double>(xs));
    attrs.set(AttrKey::WeightScaleQ, static_cast<double>(ws));

    // A 0-element bias placeholder at inputs[2] must be treated as no-bias
    // (JIT-043) rather than read out of bounds.
    Tensor empty_bias({0}, DType::Float32, oa);
    std::vector<Tensor> with_empty = {xq.to(oa), wq.to(oa), empty_bias};
    std::vector<Tensor> no_bias    = {xq.to(oa), wq.to(oa)};

    Tensor a = dispatch(OpId::QuantizedLinear, with_empty, attrs)[0].to(Device::cpu());
    Tensor b = dispatch(OpId::QuantizedLinear, no_bias, attrs)[0].to(Device::cpu());
    EXPECT_TRUE(tensors_close(a, b, 1e-4f, 1e-4f))
        << "OneAPI QuantizedLinear empty-bias placeholder != no-bias (JIT-043)";
}
