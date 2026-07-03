/**
 * @file test_codegen.cpp
 * @brief Tests for runtime GPU kernel generation (codegen)
 */

#include <gtest/gtest.h>
#include <cstdlib>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/codegen.hpp>
#include <cmath>
#include <vector>

using namespace tenzor;
using namespace tenzor::jit;

namespace {
// GPU backends the element-wise codegen can target (see the extended-codegen
// test for the rationale). The dtype tests run on EVERY available backend so a
// combined build exercises NVRTC and HIPRTC; they GTEST_SKIP only with no GPU.
auto available_gpu_devices() -> std::vector<Device> {
    std::vector<Device> devs;
    const std::vector<float> probe = {0.0f};
    for (const auto& dev : {Device::cuda(0), Device::rocm(0)}) {
        try {
            Tensor t = from_data(probe.data(), {1}).to(dev);
            (void)t.to(Device::cpu());
            devs.push_back(dev);
        } catch (const std::exception&) {
        }
    }
    return devs;
}

// Honor TENZOR_REQUIRE_MULTI_BACKEND: when set, a run with NO GPU backend is a
// hard failure (a host expected to have a GPU silently skipping GPU codegen
// coverage hides a broken environment). Otherwise skip as before.
#define SKIP_OR_FAIL_NO_GPU()                                                  \
    do {                                                                       \
        const char* _req = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");        \
        if (_req && *_req && *_req != '0')                                     \
            FAIL() << "TENZOR_REQUIRE_MULTI_BACKEND set but no GPU backend "   \
                      "present for GPU codegen test";                          \
        GTEST_SKIP() << "no GPU backend present";                             \
    } while (0)
}  // namespace

class CodegenTestEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
static ::testing::Environment* const env =
    ::testing::AddGlobalTestEnvironment(new CodegenTestEnv);

// =========================================================================
// FusionGroup construction tests
// =========================================================================

TEST(Codegen, BuildFusionGroup) {
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::MulScalar, -1, -1, 2.0},
    }, 1, DType::Float32);

    EXPECT_EQ(group.num_inputs, 1);
    EXPECT_EQ(group.steps.size(), 2u);
    EXPECT_FALSE(group.signature.empty());
}

TEST(Codegen, SignatureUniqueness) {
    auto g1 = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::Sigmoid, -1, -1, 0.0},
    }, 1, DType::Float32);

    auto g2 = build_fusion({
        {ElemOp::Sigmoid, 0, -1, 0.0},
        {ElemOp::Relu, -1, -1, 0.0},
    }, 1, DType::Float32);

    EXPECT_NE(g1.signature, g2.signature);
}

// =========================================================================
// Code generation tests
// =========================================================================

TEST(Codegen, GenerateSimpleKernel) {
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
    }, 1, DType::Float32);

    auto source = KernelCodegen::generate(group);

    // Should contain key elements
    EXPECT_NE(source.find("__global__"), std::string::npos);
    // relu is emitted as a NaN-propagating `x < 0 ? 0 : x` select. It used to
    // use fmax, but fmax(NaN, 0) returns 0 and diverged from the CPU clamp_min
    // fallback (which propagates NaN); the select matches CPU/eager semantics.
    EXPECT_NE(source.find("? 0.0f :"), std::string::npos);
    EXPECT_NE(source.find("blockIdx"), std::string::npos);
    EXPECT_NE(source.find("threadIdx"), std::string::npos);
}

TEST(Codegen, GenerateFusedKernel) {
    auto group = build_fusion({
        {ElemOp::Mul, 0, 1, 0.0},        // tmp = a * b
        {ElemOp::AddScalar, -1, -1, 1.0}, // tmp = tmp + 1
        {ElemOp::Sigmoid, -1, -1, 0.0},   // out = sigmoid(tmp)
    }, 2, DType::Float32);

    auto source = KernelCodegen::generate(group);

    // Should have 2 input pointers
    EXPECT_NE(source.find("inp0"), std::string::npos);
    EXPECT_NE(source.find("inp1"), std::string::npos);
    // Should contain sigmoid formula
    EXPECT_NE(source.find("exp"), std::string::npos);
}

TEST(Codegen, GenerateAllActivations) {
    // Test that all activation codegen compiles to valid source
    for (auto op : {ElemOp::Relu, ElemOp::LeakyRelu, ElemOp::Elu,
                    ElemOp::Selu, ElemOp::Gelu, ElemOp::Mish,
                    ElemOp::Softplus, ElemOp::Sigmoid, ElemOp::Tanh}) {
        auto group = build_fusion({{op, 0, -1, 0.1}}, 1, DType::Float32);
        auto source = KernelCodegen::generate(group);
        EXPECT_NE(source.find("__global__"), std::string::npos)
            << "Failed for op " << static_cast<int>(op);
    }
}

// =========================================================================
// Execution tests (CPU fallback or GPU)
// =========================================================================

TEST(Codegen, ExecuteFused_ReluSigmoid) {
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::Sigmoid, -1, -1, 0.0},
    }, 1, DType::Float32);

    // Create input with mix of positive and negative values
    auto input = tenzor::randn({8}, DType::Float32, Device::cpu());
    auto result = execute_fused(group, {input});

    // Verify: sigmoid(relu(x)) == sigmoid(max(0,x))
    auto expected = tenzor::sigmoid(tenzor::clamp_min(input, 0.0f));
    auto r = result.to(Device::cpu()).contiguous();
    auto e = expected.to(Device::cpu()).contiguous();
    auto* rd = r.data<float>();
    auto* ed = e.data<float>();
    for (int i = 0; i < 8; ++i) {
        EXPECT_NEAR(rd[i], ed[i], 1e-5) << "Mismatch at " << i;
    }
}

TEST(Codegen, ExecuteFused_MulAdd) {
    auto group = build_fusion({
        {ElemOp::Mul, 0, 1, 0.0},         // a * b
        {ElemOp::AddScalar, -1, -1, 3.0},  // + 3.0
    }, 2, DType::Float32);

    auto a = tenzor::ones({4}, DType::Float32, Device::cpu());
    auto b = tenzor::mul(tenzor::ones({4}, DType::Float32, Device::cpu()), 2.0);
    auto result = execute_fused(group, {a, b});

    // Expected: 1*2 + 3 = 5
    auto r = result.to(Device::cpu()).contiguous();
    auto* rd = r.data<float>();
    for (int i = 0; i < 4; ++i) {
        EXPECT_NEAR(rd[i], 5.0f, 1e-5);
    }
}

TEST(Codegen, ExecuteFused_ChainedMath) {
    auto group = build_fusion({
        {ElemOp::Exp, 0, -1, 0.0},        // exp(x)
        {ElemOp::Log, -1, -1, 0.0},       // log(exp(x)) = x
    }, 1, DType::Float32);

    auto input = tenzor::mul(tenzor::randn({16}, DType::Float32, Device::cpu()), 0.5f);
    auto result = execute_fused(group, {input});

    // exp(log(x)) ≈ x
    auto r = result.to(Device::cpu()).contiguous();
    auto inp = input.to(Device::cpu()).contiguous();
    auto* rd = r.data<float>();
    auto* id = inp.data<float>();
    for (int i = 0; i < 16; ++i) {
        EXPECT_NEAR(rd[i], id[i], 1e-4) << "Round-trip failed at " << i;
    }
}

// =========================================================================
// Cache tests
// =========================================================================

// =========================================================================
// GPU execution tests across dtypes + backends (BUG M6 / H-mask coverage)
// =========================================================================

// Float32 fusion must use the single-precision math intrinsics (expf, …) and
// match the eager Float32 exp. Runs on every available GPU backend.
TEST(Codegen, ExecuteFusedF32ExpAllBackends) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    auto group = build_fusion({
        {ElemOp::Exp, 0, -1, 0.0},
        {ElemOp::MulScalar, -1, -1, 0.5},
    }, 1, DType::Float32);

    std::vector<float> in(1024);
    for (size_t i = 0; i < in.size(); ++i) in[i] = -2.0f + 4.0f * (i / 1024.0f);
    Tensor cpu = from_data(in.data(), {1024});

    for (const auto& dev : devs) {
        Tensor gpu = cpu.to(dev);
        auto out = execute_fused(group, {gpu}).to(Device::cpu()).contiguous();
        const float* o = out.data<float>();
        for (size_t i = 0; i < in.size(); ++i) {
            float ref = std::exp(in[i]) * 0.5f;
            EXPECT_NEAR(o[i], ref, 1e-5f + 1e-5f * std::abs(ref))
                << dev.to_string() << " i=" << i;
        }
    }
}

// Float16 / BFloat16 fusion: load the 16-bit storage type, compute in float,
// narrow back. The header claims Float16/BFloat16 support — this asserts it on
// every backend. Reference uses the SAME 16-bit-rounded inputs the kernel sees.
static void run_half_chain(DType dt, double tol) {
    initialize();
    auto devs = available_gpu_devices();
    if (devs.empty()) SKIP_OR_FAIL_NO_GPU();

    // chain: relu(x) * 1.5 -> tanh
    auto group = build_fusion({
        {ElemOp::Relu, 0, -1, 0.0},
        {ElemOp::MulScalar, -1, -1, 1.5},
        {ElemOp::Tanh, -1, -1, 0.0},
    }, 1, dt);

    std::vector<float> in(512);
    for (size_t i = 0; i < in.size(); ++i) in[i] = -3.0f + 6.0f * (i / 512.0f);
    // The exact values the kernel sees are the input rounded to the 16-bit type.
    Tensor rounded_cpu = from_data(in.data(), {512}).to(dt).to(DType::Float32).contiguous();
    const float* xr = rounded_cpu.data<float>();

    for (const auto& dev : devs) {
        Tensor gpu = from_data(in.data(), {512}).to(dt).to(dev);
        auto out = execute_fused(group, {gpu}).to(Device::cpu()).to(DType::Float32).contiguous();
        const float* o = out.data<float>();
        for (size_t i = 0; i < in.size(); ++i) {
            float relu = xr[i] < 0.0f ? 0.0f : xr[i];
            float ref = std::tanh(relu * 1.5f);
            EXPECT_NEAR(o[i], ref, tol) << dev.to_string() << " i=" << i;
        }
    }
}

TEST(Codegen, ExecuteFusedFloat16Chain)  { run_half_chain(DType::Float16, 3e-3); }
TEST(Codegen, ExecuteFusedBFloat16Chain) { run_half_chain(DType::BFloat16, 3e-2); }

TEST(Codegen, KernelCacheHit) {
    auto& cache = KernelCache::instance();
    size_t compilations_before = cache.num_compilations();

    auto group = build_fusion({
        {ElemOp::Neg, 0, -1, 0.0},
        {ElemOp::Abs, -1, -1, 0.0},
    }, 1, DType::Float32);

    // First call compiles
    auto input = tenzor::ones({4}, DType::Float32, Device::cpu());
    execute_fused(group, {input});

    // Second call should hit cache (no new compilation)
    size_t compilations_after_first = cache.num_compilations();
    execute_fused(group, {input});
    size_t compilations_after_second = cache.num_compilations();

    // On CPU fallback, no compilations happen, but cache is still tested
    // On GPU, second call should NOT compile again
    EXPECT_LE(compilations_after_second, compilations_after_first + 0);
}
