// Cross-backend FlashAttention / FusedAttention parity sweep.
//
// Per docs/internals/attention-contract.md M9: every enabled backend must
// produce the same output as CPU (within F32 tolerance) for the contract
// axes — {causal, !causal} × {MHA, GQA} × {contiguous, permuted Q/K/V}.
//
// Backend selection is driven by the BackendTest fixture: each instantiated
// device runs the full contract-axis sweep against a CPU reference. Backends
// that are unavailable are handled by the fixture (skip, or hard fail under
// TENZOR_REQUIRE_MULTI_BACKEND=1).
//
// Tolerance: 1e-3 absolute (matches the contract's F32 numerical-equivalence
// statement). FMA-ordering differences across backends are why bit-exact
// equivalence is not a contract requirement.

#include <gtest/gtest.h>
#include "../backend_test_fixture.hpp"
#include "tenzor/tenzor.hpp"
#include "tenzor/autograd/variable.hpp"
#include "tenzor/autograd/ops.hpp"
#include "tenzor/backend/dispatch_table.hpp"
#include "tenzor/ops/creation.hpp"
#include "tenzor/ops/transform.hpp"
#include "tenzor/ops/math.hpp"
#include "tenzor/ops/reduction.hpp"

#include <cmath>
#include <string>
#include <utility>
#include <vector>

using namespace tenzor;

namespace {

// One axis-combination of the attention contract. The backend dimension is no
// longer part of this struct — it is supplied by the BackendTest fixture's
// `device` member, which lets INSTANTIATE_BACKEND_TESTS enumerate backends.
struct ContractAxis {
    const char* name;
    bool causal;
    int64_t H_q;
    int64_t H_kv;
    bool permuted;
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
Tensor backend_compute(const Tensor& Q_dev, const Tensor& K_dev, const Tensor& V_dev,
                        float scale, bool causal) {
    Variable Qv(Q_dev, false);
    Variable Kv(K_dev, false);
    Variable Vv(V_dev, false);
    Variable out = tenzor::fused_attention(Qv, Kv, Vv, scale, causal, /*use_cudnn_sdpa=*/false);
    return out.tensor().cpu();
}

double max_abs_diff(const Tensor& a, const Tensor& b) {
    Tensor a64 = a.cpu().to(DType::Float64);
    Tensor b64 = b.cpu().to(DType::Float64);
    Tensor diff = tenzor::sub(a64, b64);
    Tensor abs_diff = tenzor::abs(diff);
    return tenzor::max(abs_diff).item<double>();
}

const std::vector<ContractAxis>& contract_axes() {
    static const std::vector<ContractAxis> axes = []() {
        std::vector<ContractAxis> v;
        for (bool causal : {false, true}) {
            for (auto [Hq, Hkv] : std::vector<std::pair<int64_t, int64_t>>{{4, 4}, {4, 2}, {8, 1}}) {
                for (bool perm : {false, true}) {
                    const char* nm = causal ? "causal" : "noncausal";
                    v.push_back({nm, causal, Hq, Hkv, perm});
                }
            }
        }
        return v;
    }();
    return axes;
}

class AttentionParityTest : public ::tenzor::testing::BackendTest {
protected:
    void SetUp() override {
        ::tenzor::testing::BackendTest::SetUp();
        if (::testing::Test::IsSkipped()) return;
    }
};

}  // anonymous namespace

TEST_P(AttentionParityTest, MatchesCPU) {
    constexpr int64_t B = 2;
    constexpr int64_t S_q = 16;
    constexpr int64_t S_k = 16;
    constexpr int64_t D = 32;

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));

    for (const auto& cs : contract_axes()) {
        SCOPED_TRACE(std::string(cs.name) +
                     "_Hq" + std::to_string(cs.H_q) +
                     "_Hkv" + std::to_string(cs.H_kv) +
                     (cs.permuted ? "_permuted" : "_contig"));

        // Generate Q/K/V directly on the target device. K and V have H_kv
        // heads (GQA when H_kv < H_q).
        Tensor Q_dev = tenzor::randn({B, cs.H_q,  S_q, D}, DType::Float32, device);
        Tensor K_dev = tenzor::randn({B, cs.H_kv, S_k, D}, DType::Float32, device);
        Tensor V_dev = tenzor::randn({B, cs.H_kv, S_k, D}, DType::Float32, device);

        if (cs.permuted) {
            // Make non-contiguous via a transpose+detranspose trick: permute to
            // [B, S, H, D] then back to [B, H, S, D] without contiguous().
            // This exercises the audit Systemic #4 stride-from-shape path.
            Q_dev = tenzor::transpose(tenzor::transpose(Q_dev, 1, 2), 1, 2);
            K_dev = tenzor::transpose(tenzor::transpose(K_dev, 1, 2), 1, 2);
            V_dev = tenzor::transpose(tenzor::transpose(V_dev, 1, 2), 1, 2);
        }

        // CPU reference uses the same (host-side) inputs the device sees.
        Tensor Q_cpu = Q_dev.cpu();
        Tensor K_cpu = K_dev.cpu();
        Tensor V_cpu = V_dev.cpu();
        Tensor ref = cpu_reference(Q_cpu, K_cpu, V_cpu, scale, cs.causal);

        Tensor got = backend_compute(Q_dev, K_dev, V_dev, scale, cs.causal);

        ASSERT_EQ(got.numel(), ref.numel())
            << cs.name << ": output shape mismatch (ref numel=" << ref.numel()
            << ", got=" << got.numel() << ")";

        double diff = max_abs_diff(got, ref);
        EXPECT_LT(diff, 1e-3)
            << cs.name << ": max |out - ref| = " << diff << " > 1e-3 tolerance"
            << " on device " << device.to_string();
    }
}

namespace {
INSTANTIATE_BACKEND_TESTS(AttentionParityTest);
}  // anonymous namespace
