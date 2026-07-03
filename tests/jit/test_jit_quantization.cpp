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
