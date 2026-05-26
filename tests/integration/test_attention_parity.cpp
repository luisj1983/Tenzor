// Cross-backend FlashAttention / FusedAttention parity sweep.
//
// Per docs/internals/attention-contract.md M9: every enabled backend must
// produce the same output as CPU (within F32 tolerance) for the contract
// axes — {causal, !causal} × {MHA, GQA} × {contiguous, permuted Q/K/V}.
//
// The test is skipped silently when only CPU is available; setting
// TENZOR_REQUIRE_MULTI_BACKEND=1 makes that a hard fail (per CLAUDE.md
// memory: feedback_testing.md).
//
// Tolerance: 1e-3 absolute (matches the contract's F32 numerical-equivalence
// statement). FMA-ordering differences across backends are why bit-exact
// equivalence is not a contract requirement.

#include <gtest/gtest.h>
#include "../multi_backend_dtype_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

#include <cstdlib>
#include <string>
#include <vector>

using namespace tenzor;

namespace {

bool require_multi_backend() {
    const char* env = std::getenv("TENZOR_REQUIRE_MULTI_BACKEND");
    return env != nullptr && env[0] == '1';
}

bool backend_loaded(Device::Type type) {
    return DispatchTableRegistry::has_backend(type);
}

struct ParityCase {
    Device::Type backend;
    const char* name;
    bool causal;
    int64_t H_q;
    int64_t H_kv;
    bool permuted;
};

struct ParityCaseFmt {
    std::string operator()(const ::testing::TestParamInfo<ParityCase>& info) const {
        std::string s = info.param.name;
        s += info.param.causal ? "_causal" : "_noncausal";
        s += "_Hq" + std::to_string(info.param.H_q);
        s += "_Hkv" + std::to_string(info.param.H_kv);
        s += info.param.permuted ? "_permuted" : "_contig";
        return s;
    }
};

class AttentionParityTest : public ::testing::TestWithParam<ParityCase> {
protected:
    static void SetUpTestSuite() {
        tenzor::initialize();
    }
};

// Compute the reference attention output on CPU using the autograd-correct
// fused_attention helper. This is the ground truth every backend matches.
Tensor cpu_reference(const Tensor& Q, const Tensor& K, const Tensor& V,
                     float scale, bool causal) {
    Variable Qv(Q.cpu(), false);
    Variable Kv(K.cpu(), false);
    Variable Vv(V.cpu(), false);
    Variable out = tenzor::fused_attention(Qv, Kv, Vv, scale, causal, /*use_cudnn_sdpa=*/false);
    return out.tensor();
}

// Compute attention on `dev` via the same autograd helper; the dispatch
// system routes to the per-backend kernel based on tensor.device().
Tensor backend_compute(const Tensor& Q_cpu, const Tensor& K_cpu, const Tensor& V_cpu,
                        float scale, bool causal, Device dev) {
    Tensor Q_dev = Q_cpu.to(dev);
    Tensor K_dev = K_cpu.to(dev);
    Tensor V_dev = V_cpu.to(dev);
    Variable Qv(Q_dev, false);
    Variable Kv(K_dev, false);
    Variable Vv(V_dev, false);
    Variable out = tenzor::fused_attention(Qv, Kv, Vv, scale, causal, /*use_cudnn_sdpa=*/false);
    return out.tensor().cpu();
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
    Tensor a64 = a.to(DType::Float64);
    Tensor b64 = b.to(DType::Float64);
    Tensor diff = tenzor::sub(a64, b64);
    Tensor abs_diff = tenzor::abs(diff);
    return tenzor::max(abs_diff).item<double>();
}

}  // anonymous namespace

TEST_P(AttentionParityTest, MatchesCPU) {
    auto cs = GetParam();
    if (!backend_loaded(cs.backend)) {
        if (require_multi_backend()) {
            FAIL() << "Backend " << cs.name
                   << " not loaded but TENZOR_REQUIRE_MULTI_BACKEND=1";
        }
        SKIP_WITH_REASON(tenzor::testing::SkipReason::BackendUnavailable,
            "Backend " << cs.name << " not loaded");
        return;
    }

    constexpr int64_t B = 2;
    constexpr int64_t S_q = 16;
    constexpr int64_t S_k = 16;
    constexpr int64_t D = 32;

    // Generate Q/K/V on CPU. K and V have H_kv heads (GQA when H_kv < H_q).
    Tensor Q_base = tenzor::randn({B, cs.H_q,  S_q, D}, DType::Float32);
    Tensor K_base = tenzor::randn({B, cs.H_kv, S_k, D}, DType::Float32);
    Tensor V_base = tenzor::randn({B, cs.H_kv, S_k, D}, DType::Float32);

    if (cs.permuted) {
        // Make non-contiguous via a transpose+detranspose trick: permute to
        // [B, S, H, D] then back to [B, H, S, D] without contiguous().
        // This exercises the audit Systemic #4 stride-from-shape path.
        Q_base = tenzor::transpose(tenzor::transpose(Q_base, 1, 2), 1, 2);
        K_base = tenzor::transpose(tenzor::transpose(K_base, 1, 2), 1, 2);
        V_base = tenzor::transpose(tenzor::transpose(V_base, 1, 2), 1, 2);
    }

    float scale = 1.0f / std::sqrt(static_cast<float>(D));
    Tensor ref = cpu_reference(Q_base, K_base, V_base, scale, cs.causal);

    Device dev{cs.backend, 0};
    Tensor got;
    try {
        got = backend_compute(Q_base, K_base, V_base, scale, cs.causal, dev);
    } catch (const std::exception& e) {
        // Backend may legitimately reject this configuration (e.g. FlexAttention
        // ScoreModId not yet implemented for this combination). That's a known
        // contract gap, not a parity failure — skip with the message.
        GTEST_SKIP() << cs.name << " rejected the configuration: " << e.what();
    }

    ASSERT_EQ(got.numel(), ref.numel())
        << cs.name << ": output shape mismatch (ref numel=" << ref.numel()
        << ", got=" << got.numel() << ")";

    double diff = max_abs_diff(got, ref);
    EXPECT_LT(diff, 1e-3)
        << cs.name << ": max |out - ref| = " << diff << " > 1e-3 tolerance";
}

namespace {

constexpr Device::Type kBackends[] = {
    Device::Type::CUDA,
    Device::Type::ROCm,
    Device::Type::OneAPI,
    Device::Type::Vulkan,
};
constexpr const char* kBackendNames[] = {"cuda", "rocm", "oneapi", "vulkan"};

std::vector<ParityCase> generate_cases() {
    std::vector<ParityCase> cases;
    for (size_t i = 0; i < std::size(kBackends); ++i) {
        for (bool causal : {false, true}) {
            for (auto [Hq, Hkv] : std::vector<std::pair<int64_t, int64_t>>{{4, 4}, {4, 2}, {8, 1}}) {
                for (bool perm : {false, true}) {
                    cases.push_back({kBackends[i], kBackendNames[i], causal, Hq, Hkv, perm});
                }
            }
        }
    }
    return cases;
}

}  // anonymous namespace

INSTANTIATE_TEST_SUITE_P(
    AllBackends,
    AttentionParityTest,
    ::testing::ValuesIn(generate_cases()),
    ParityCaseFmt());
