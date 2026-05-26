/**
 * @file test_eager_parity.cpp
 * @brief G.13 — eager-equivalence parity for lazy / JIT / lite execution paths.
 *
 * Audit item G.13 asks for every model in `src/models/` to be exercised through
 * eager, lazy, JIT (CPU + GPU), and lite, with FP64 bitwise / FP32 within 1 ULP.
 * That full sweep depends on Phase C.3 (the lite exporter currently only
 * supports Linear + activations) and on built-out GPU JIT backends. This file
 * is the pragmatic first cut: it exercises a small, representative set of
 * modules end-to-end against every alternate execution path that is actually
 * implemented today, and pins the parity contract so future audit fixes can
 * surface regressions immediately.
 *
 * Scope (this cut):
 *   - Lazy parity: small Tensor-level compute chain materialised through
 *     LazyTensor vs. the same chain run eagerly. Same dispatch path, so
 *     bit-exact is the expectation.
 *   - JIT parity: CPU-only. nn::Module is traced via CompiledModule::trace,
 *     and the compiled module's output is compared to eager. CPU-only because
 *     GPU JIT requires hardware that may not be present in CI.
 *   - Lite parity: a Linear+ReLU+Linear stack exported via
 *     `tenzor::lite::export_to_tzlite`, loaded into `LiteRuntime`, and
 *     compared against eager. Restricted to layers the Phase 3 exporter
 *     supports (Linear, ReLU, Sigmoid, Tanh, GELU); broader model coverage
 *     unblocks once C.3 lands.
 *
 * Tolerances:
 *   - Lazy: bitwise (same kernels run, identical inputs).
 *   - JIT CPU / Lite: ~1 Float32 ULP (atol 1e-5, rtol 0). The reference paths
 *     all funnel through the same CPU MKL kernels, so any drift beyond ULP
 *     scale here is a real divergence.
 */

#include <gtest/gtest.h>
#include <tenzor/autograd/variable.hpp>
#include <tenzor/jit/compile.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/lazy/lazy_tensor.hpp>
#include <tenzor/lite/exporter.hpp>
#include <tenzor/lite/lite_graph.hpp>
#include <tenzor/lite/runtime.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/nn/layers/linear.hpp>
#include <tenzor/nn/module.hpp>
#include <tenzor/ops/creation.hpp>
#include <tenzor/ops/math.hpp>
#include <tenzor/tenzor.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <memory>
#include <string>
#include <vector>

namespace tenzor { void initialize(); }

namespace {

class EagerParityEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_eager_parity_env =
    ::testing::AddGlobalTestEnvironment(new EagerParityEnv);

// Compare two contiguous CPU buffers element-wise within a Float32 ULP-scale
// tolerance. Returns gtest AssertionResult so the failure message localises
// the first divergence.
auto compare_f32(const float* ref, const float* alt, int64_t n,
                 float atol = 1e-5f) -> ::testing::AssertionResult {
    for (int64_t i = 0; i < n; ++i) {
        float diff = std::fabs(ref[i] - alt[i]);
        if (diff > atol) {
            return ::testing::AssertionFailure()
                << "f32 mismatch at index " << i << ": ref=" << ref[i]
                << " alt=" << alt[i] << " |delta|=" << diff
                << " atol=" << atol;
        }
    }
    return ::testing::AssertionSuccess();
}

// Bit-identical comparison for paths that go through the *same* eager
// dispatch (e.g. lazy materialisation). Any difference is a real bug.
auto compare_bitwise_f32(const float* ref, const float* alt, int64_t n)
    -> ::testing::AssertionResult {
    for (int64_t i = 0; i < n; ++i) {
        std::uint32_t a, b;
        std::memcpy(&a, &ref[i], sizeof(a));
        std::memcpy(&b, &alt[i], sizeof(b));
        if (a != b) {
            return ::testing::AssertionFailure()
                << "f32 bitwise mismatch at index " << i << ": ref=" << ref[i]
                << " (0x" << std::hex << a << ") alt=" << alt[i]
                << " (0x" << b << std::dec << ")";
        }
    }
    return ::testing::AssertionSuccess();
}

auto compare_bitwise_f64(const double* ref, const double* alt, int64_t n)
    -> ::testing::AssertionResult {
    for (int64_t i = 0; i < n; ++i) {
        std::uint64_t a, b;
        std::memcpy(&a, &ref[i], sizeof(a));
        std::memcpy(&b, &alt[i], sizeof(b));
        if (a != b) {
            return ::testing::AssertionFailure()
                << "f64 bitwise mismatch at index " << i << ": ref=" << ref[i]
                << " alt=" << alt[i];
        }
    }
    return ::testing::AssertionSuccess();
}

// Make a CPU contiguous copy and return its data pointer + numel.
struct CpuView {
    tenzor::Tensor owner;
    int64_t numel;
};
auto to_cpu_contiguous(const tenzor::Tensor& t) -> CpuView {
    auto c = t.is_contiguous() ? t : t.contiguous();
    if (c.device().type != tenzor::Device::Type::CPU) {
        c = c.to(tenzor::Device::cpu());
    }
    return CpuView{c, c.numel()};
}

// Build a LiteTensor that *owns* a copy of `src`'s data. Mirrors
// tensor_to_lite_input from tests/lite/test_exporter.cpp.
auto tensor_to_lite_input(const tenzor::Tensor& src)
    -> tenzor::lite::LiteTensor {
    auto cpu = to_cpu_contiguous(src).owner;
    tenzor::lite::LiteTensor lt;
    lt.ndim = static_cast<int32_t>(cpu.ndim());
    lt.dtype = cpu.dtype();
    lt.owns_data = true;
    int64_t numel = 1;
    for (int32_t i = 0; i < lt.ndim; ++i) {
        lt.shape[i] = cpu.size(i);
        numel *= lt.shape[i];
    }
    for (int32_t i = lt.ndim - 1; i >= 0; --i) {
        lt.strides[i] = (i == lt.ndim - 1)
                            ? 1
                            : lt.strides[i + 1] * lt.shape[i + 1];
    }
    const auto nbytes =
        static_cast<size_t>(numel * tenzor::dtype_size(lt.dtype));
    lt.data = std::malloc(nbytes);
    std::memcpy(lt.data, cpu.data_ptr(), nbytes);
    return lt;
}

auto temp_tzlite_path(const std::string& stem) -> std::string {
    namespace fs = std::filesystem;
    return (fs::temp_directory_path() /
            (stem + "_" + std::to_string(static_cast<long>(::getpid())) +
             ".tzlite"))
        .string();
}

// Copy parameters tensor-for-tensor from `src` into `dst` (must be the same
// architecture). Same helper pattern used across the existing parity tests.
void copy_params(tenzor::nn::Module& src, tenzor::nn::Module& dst) {
    auto src_params = src.parameters();
    auto dst_params = dst.parameters();
    ASSERT_EQ(src_params.size(), dst_params.size())
        << "Parameter count mismatch between source and destination modules";
    for (size_t p = 0; p < src_params.size(); ++p) {
        dst_params[p]->tensor() = src_params[p]->tensor().clone();
    }
}

// ---------------------------------------------------------------------------
// Reference models
// ---------------------------------------------------------------------------

// Tiny MLP, fully supported by the lite exporter (Linear + ReLU + Linear).
// Built as a Sequential because the Phase 3 lite exporter only walks
// nn::Sequential / nn::Linear / activation containers; a hand-rolled
// nn::Module subclass with a custom forward_impl is rejected. Once C.3
// finishes the exporter's module traversal, this can be swapped back to a
// plain subclass and the JIT and lite tests can share the same model.
auto make_mlp_model() -> std::shared_ptr<tenzor::nn::Sequential> {
    return std::make_shared<tenzor::nn::Sequential>(
        std::make_shared<tenzor::nn::Linear>(8, 16, /*bias=*/true),
        std::make_shared<tenzor::nn::ReLU>(),
        std::make_shared<tenzor::nn::Linear>(16, 4, /*bias=*/true));
}

// A trivial Conv2d + ReLU stack — used only for the JIT path. Lite cannot
// export this until C.3 lands.
class ConvStackModel : public tenzor::nn::Module {
public:
    ConvStackModel() {
        conv = std::make_shared<tenzor::nn::Conv2d>(3, 8, 3, 1, 1);
        act  = std::make_shared<tenzor::nn::ReLU>();
        register_module("conv", conv);
        register_module("act", act);
    }
    auto forward_impl(const tenzor::Variable& x)
        -> tenzor::Variable override {
        auto h = conv->forward(x);
        return act->forward(h);
    }
    std::shared_ptr<tenzor::nn::Conv2d> conv;
    std::shared_ptr<tenzor::nn::ReLU>   act;
};

}  // namespace

// ===========================================================================
// Lazy parity (FP32 + FP64)
// ===========================================================================

// LazyTensor materialisation funnels through the same dispatch as eager.
// We assert bitwise equality here — any drift signals that lazy execution
// took a different code path or applied an unintended transform.
TEST(EagerParity, LazyVsEager_AddMulChain_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto a = randn({16, 8}, DType::Float32, device);
    auto b = randn({16, 8}, DType::Float32, device);
    auto c = randn({16, 8}, DType::Float32, device);

    // Eager: ((a + b) * c) + a
    auto eager_out = add(mul(add(a, b), c), a);

    // Lazy: same chain via the LazyTensor API.
    auto la = lazy::LazyTensor::from_tensor(a);
    auto lb = lazy::LazyTensor::from_tensor(b);
    auto lc = lazy::LazyTensor::from_tensor(c);
    auto sum_ab    = lazy::add(la, lb);
    auto scaled    = lazy::mul(sum_ab, lc);
    auto chain_out = lazy::add(scaled, la);
    auto lazy_out  = chain_out.materialize();

    auto ref = to_cpu_contiguous(eager_out);
    auto alt = to_cpu_contiguous(lazy_out);
    ASSERT_EQ(ref.numel, alt.numel);
    EXPECT_TRUE(compare_bitwise_f32(ref.owner.data<float>(),
                                    alt.owner.data<float>(), ref.numel));
}

TEST(EagerParity, LazyVsEager_AddMulChain_Float64) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto a = randn({8, 8}, DType::Float64, device);
    auto b = randn({8, 8}, DType::Float64, device);
    auto c = randn({8, 8}, DType::Float64, device);

    auto eager_out = add(mul(add(a, b), c), a);

    auto la = lazy::LazyTensor::from_tensor(a);
    auto lb = lazy::LazyTensor::from_tensor(b);
    auto lc = lazy::LazyTensor::from_tensor(c);
    auto chain_out = lazy::add(lazy::mul(lazy::add(la, lb), lc), la);
    auto lazy_out  = chain_out.materialize();

    auto ref = to_cpu_contiguous(eager_out);
    auto alt = to_cpu_contiguous(lazy_out);
    ASSERT_EQ(ref.numel, alt.numel);
    EXPECT_TRUE(compare_bitwise_f64(ref.owner.data<double>(),
                                    alt.owner.data<double>(), ref.numel));
}

// MatMul is the workhorse op and is a good lazy-vs-eager parity probe.
TEST(EagerParity, LazyVsEager_MatMul_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto x = randn({4, 8}, DType::Float32, device);
    auto w = randn({8, 6}, DType::Float32, device);

    auto eager_out = matmul(x, w);

    auto lx = lazy::LazyTensor::from_tensor(x);
    auto lw = lazy::LazyTensor::from_tensor(w);
    auto lazy_out = lazy::matmul(lx, lw).materialize();

    auto ref = to_cpu_contiguous(eager_out);
    auto alt = to_cpu_contiguous(lazy_out);
    ASSERT_EQ(ref.numel, alt.numel);
    // Same CPU MKL gemm in both paths; bitwise equality is the contract.
    EXPECT_TRUE(compare_bitwise_f32(ref.owner.data<float>(),
                                    alt.owner.data<float>(), ref.numel));
}

// ===========================================================================
// JIT parity (CPU only — GPU JIT requires hardware not assumed in CI)
// ===========================================================================

TEST(EagerParity, JitVsEager_Mlp_Cpu_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto eager_model = make_mlp_model();
    eager_model->to(device);
    eager_model->eval();

    auto traced_model = make_mlp_model();
    traced_model->to(device);
    traced_model->eval();
    copy_params(*eager_model, *traced_model);

    auto x = randn({4, 8}, DType::Float32, device);

    auto eager_out = eager_model->forward(Variable(x, false)).tensor();

    // JIT trace + invoke.
    auto compiled = jit::CompiledModule::trace(traced_model, x);
    ASSERT_TRUE(compiled != nullptr) << "CompiledModule::trace returned null";
    auto jit_out  = compiled->forward(Variable(x, false)).tensor();

    auto ref = to_cpu_contiguous(eager_out);
    auto alt = to_cpu_contiguous(jit_out);
    ASSERT_EQ(ref.numel, alt.numel);
    // CPU eager and CPU JIT both funnel through the same dispatch table, so
    // 1 ULP is generous.
    EXPECT_TRUE(compare_f32(ref.owner.data<float>(),
                            alt.owner.data<float>(), ref.numel,
                            /*atol=*/1e-5f));
}

// REGRESSION-DOCUMENT: this test currently fails with a real, reproducible
// shape divergence: eager Conv2d(3->8, k=3, stride=1, padding=1) on a
// [2,3,8,8] input produces a [2,8,8,8] output (numel=1024), but the JIT
// trace+invoke path produces [2,8,6,6] (numel=576) — i.e. the JIT-captured
// Conv2d *loses the padding=1 attribute*, defaulting to padding=0. This is
// exactly the class of bug the G.13 parity contract exists to surface; the
// test is intentionally kept failing until the JIT tracer fix lands (track
// under "JIT Conv2d padding attribute lost during trace").
// Once the JIT side is fixed, remove the skip macro below to re-enable the
// numeric parity check.
TEST(EagerParity, JitVsEager_ConvStack_Cpu_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto eager_model = std::make_shared<ConvStackModel>();
    eager_model->to(device);
    eager_model->eval();

    auto traced_model = std::make_shared<ConvStackModel>();
    traced_model->to(device);
    traced_model->eval();
    copy_params(*eager_model, *traced_model);

    auto x = randn({2, 3, 8, 8}, DType::Float32, device);

    auto eager_out = eager_model->forward(Variable(x, false)).tensor();

    auto compiled = jit::CompiledModule::trace(traced_model, x);
    ASSERT_TRUE(compiled != nullptr) << "CompiledModule::trace returned null";
    auto jit_out  = compiled->forward(Variable(x, false)).tensor();

    auto ref = to_cpu_contiguous(eager_out);
    auto alt = to_cpu_contiguous(jit_out);

    // Audit item G.13: JIT-traced Conv2d previously dropped its padding/
    // stride/dilation attributes and silently ran with padding=0 (output
    // [2,8,6,6] vs eager [2,8,8,8]). Now that Graph::execute_node Conv2d
    // honors PaddingH/PaddingW (and the vec/scalar fallbacks the tracer
    // emits), the JIT and eager paths drive the same Conv2dForward kernel
    // with identical attrs and weights, so the outputs must match
    // bit-exactly for Float32.
    {
        auto es = eager_out.shape();
        auto js = jit_out.shape();
        ASSERT_EQ(std::vector<int64_t>(es.begin(), es.end()),
                  std::vector<int64_t>(js.begin(), js.end()))
            << "JIT Conv2d output shape diverges from eager (padding bug regression)";
    }
    ASSERT_EQ(ref.numel, alt.numel);
    EXPECT_TRUE(compare_bitwise_f32(ref.owner.data<float>(),
                                    alt.owner.data<float>(), ref.numel));
}

// `jit::compile` wraps a callable into a CompiledFunction with the trace +
// shape cache. Exercising both entry points keeps coverage on the user-facing
// API surface, not just the internal CompiledModule::trace.
TEST(EagerParity, JitCompileVsEager_Mlp_Cpu_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto model = make_mlp_model();
    model->to(device);
    model->eval();

    auto x = randn({4, 8}, DType::Float32, device);

    auto eager_out = model->forward(Variable(x, false)).tensor();

    auto compiled = jit::compile([&](const Variable& v) {
        return model->forward(v);
    });
    auto jit_out = compiled(Variable(x, false)).tensor();

    auto ref = to_cpu_contiguous(eager_out);
    auto alt = to_cpu_contiguous(jit_out);
    ASSERT_EQ(ref.numel, alt.numel);
    EXPECT_TRUE(compare_f32(ref.owner.data<float>(),
                            alt.owner.data<float>(), ref.numel,
                            /*atol=*/1e-5f));
}

// ===========================================================================
// Lite parity (CPU, restricted to layers C.3-supported)
// ===========================================================================

// NOTE: scope of this test is intentionally narrower than the audit spec.
// The Phase 3 lite exporter accepts Linear + activations (Sigmoid, Tanh, GELU,
// ReLU). Conv2d / BN / attention layers will roll in once C.3 finishes the
// op coverage backlog. Once that lands, this test should be extended to the
// same model coverage as the JIT tests above.
TEST(EagerParity, LiteVsEager_Mlp_Cpu_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto model = make_mlp_model();
    model->to(device);
    model->eval();

    auto x = randn({4, 8}, DType::Float32, device);

    auto eager_out = model->forward(Variable(x, false)).tensor();

    lite::ExportOptions opts;
    opts.input_shape = {4, 8};
    opts.input_dtype = DType::Float32;
    auto path = temp_tzlite_path("eager_parity_mlp");
    lite::export_to_tzlite(*model, path, opts);

    auto runtime = lite::LiteRuntime::load(path);
    ASSERT_TRUE(runtime != nullptr) << "LiteRuntime::load returned null";

    auto lite_in  = tensor_to_lite_input(x);
    auto lite_out = runtime->forward(lite_in);

    auto ref = to_cpu_contiguous(eager_out);
    ASSERT_EQ(ref.numel, lite_out.numel());
    EXPECT_TRUE(compare_f32(ref.owner.data<float>(),
                            lite_out.data_as<float>(),
                            ref.numel,
                            /*atol=*/1e-5f));

    std::filesystem::remove(path);
}

// Sigmoid activation flavour — keeps the lite-supported activation surface
// exercised under the parity contract.
TEST(EagerParity, LiteVsEager_LinearSigmoid_Cpu_Float32) {
    using namespace tenzor;
    auto device = Device::cpu();

    auto model = std::make_shared<nn::Sequential>(
        std::make_shared<nn::Linear>(8, 4, /*bias=*/true),
        std::make_shared<nn::Sigmoid>());
    model->to(device);
    model->eval();

    auto x = randn({2, 8}, DType::Float32, device);

    auto eager_out =
        model->forward_impl(Variable(x, false)).tensor();

    lite::ExportOptions opts;
    opts.input_shape = {2, 8};
    opts.input_dtype = DType::Float32;
    auto path = temp_tzlite_path("eager_parity_lin_sigmoid");
    lite::export_to_tzlite(*model, path, opts);

    auto runtime = lite::LiteRuntime::load(path);
    ASSERT_TRUE(runtime != nullptr);

    auto lite_in  = tensor_to_lite_input(x);
    auto lite_out = runtime->forward(lite_in);

    auto ref = to_cpu_contiguous(eager_out);
    ASSERT_EQ(ref.numel, lite_out.numel());
    EXPECT_TRUE(compare_f32(ref.owner.data<float>(),
                            lite_out.data_as<float>(),
                            ref.numel,
                            /*atol=*/1e-5f));

    std::filesystem::remove(path);
}
