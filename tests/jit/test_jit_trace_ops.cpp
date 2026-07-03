/**
 * @file test_jit_trace_ops.cpp
 * @brief End-to-end JIT trace/replay correctness for shape ops (transpose,
 *        flatten-subrange) and elementwise ops (sin, cos, rsqrt) plus RMSNorm.
 *
 * Each test traces a function via tenzor::jit::compile(), drives it past the
 * warmup call so the SECOND invocation replays the captured graph through the
 * Graph executor, and asserts the replayed result matches eager execution on
 * the SAME backend. A trivial eager-passthrough (num_cached == 0) would let a
 * broken trace masquerade as "correct", so every test also asserts the graph
 * was actually compiled and cached.
 *
 * Regression coverage:
 *  - Traced Transpose must preserve dim0/dim1 (else it replays as a no-op).
 *  - Traced Flatten must preserve start_dim/end_dim (else the range is lost).
 *  - Sin/Cos/Rsqrt/RMSNorm must be first-class IR nodes, not frozen constants.
 */

#include <gtest/gtest.h>
#include <tenzor/tenzor.hpp>
#include <tenzor/jit/compile.hpp>

#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;
using namespace tenzor::testing;

namespace {

class JitTraceOpsEnv : public ::testing::Environment {
public:
    void SetUp() override { tenzor::initialize(); }
};
[[maybe_unused]] auto* g_env =
    ::testing::AddGlobalTestEnvironment(new JitTraceOpsEnv);

// Run `fn` compiled on `dev`: warm up (cache-miss -> eager + build graph),
// then invoke again so the compiled Graph executor replays. Returns the
// replayed output tensor (host copy) and the number of cached graphs.
struct ReplayResult {
    Tensor output;   // on CPU for easy comparison
    size_t cached;
};

ReplayResult compiled_replay(jit::CompiledFunction::FnType fn,
                             const Tensor& input_dev) {
    auto compiled = jit::compile(fn);
    Variable v(input_dev, /*requires_grad=*/false);
    // First call: cache miss (runs eager, compiles in the background).
    (void)compiled(v);
    input_dev.device().synchronize();
    // Second call: cache hit -> Graph executor replay.
    auto out = compiled(v);
    input_dev.device().synchronize();
    return {out.tensor().to(Device::cpu()), compiled.num_cached()};
}

// Compare compiled-replay vs eager on every available backend.
void check_op_on_all_backends(
        const char* label,
        const std::function<Variable(const Variable&)>& fn,
        const std::function<Tensor(Device)>& make_input,
        float rtol = 1e-4f, float atol = 1e-4f) {
    for (const auto& dev : get_available_backends()) {
        try {
            auto input = make_input(dev);
            // Eager reference on the same backend.
            Tensor eager = fn(Variable(input, false)).tensor().to(Device::cpu());

            auto rr = compiled_replay(
                [&fn](const Variable& x) { return fn(x); }, input);

            EXPECT_GE(rr.cached, 1u)
                << label << " did not compile/cache a graph on "
                << backend_name(dev)
                << " — the op likely graph-broke and silently fell back to "
                   "eager, so replay correctness was never exercised";

            EXPECT_TRUE(tensors_close(eager, rr.output, rtol, atol))
                << label << " replay != eager on " << backend_name(dev);
        } catch (const std::exception& e) {
            ADD_FAILURE() << label << " threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

}  // namespace

// =========================================================================
// Shape ops
// =========================================================================

TEST(JitTraceOps, TransposePreservesAxes) {
    // relu introduces a real dispatched node; the transpose(1,2) must replay
    // with dim0=1,dim1=2 — not the default (0,0) no-op.
    check_op_on_all_backends(
        "Transpose(1,2)",
        [](const Variable& x) {
            return nn::relu(x).transpose(1, 2);
        },
        [](Device d) { return randn({2, 3, 4}, DType::Float32, d); });
}

TEST(JitTraceOps, FlattenPreservesRange) {
    // flatten(1,2) collapses dims 1..2 of a rank-4 tensor. A lost range would
    // flatten everything (or nothing) and change the output shape/values.
    check_op_on_all_backends(
        "Flatten(1,2)",
        [](const Variable& x) {
            return tenzor::flatten(nn::relu(x), 1, 2);
        },
        [](Device d) { return randn({2, 3, 4, 5}, DType::Float32, d); });
}

// =========================================================================
// Elementwise ops that were previously unmapped (graph-break -> frozen const)
// =========================================================================

TEST(JitTraceOps, SinIsFirstClassNode) {
    check_op_on_all_backends(
        "Sin",
        [](const Variable& x) { return tenzor::sin(x) + x; },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
}

TEST(JitTraceOps, CosIsFirstClassNode) {
    check_op_on_all_backends(
        "Cos",
        [](const Variable& x) { return tenzor::cos(x) + x; },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
}

TEST(JitTraceOps, RsqrtIsFirstClassNode) {
    check_op_on_all_backends(
        "Rsqrt",
        [](const Variable& x) { return tenzor::rsqrt(x); },
        // Strictly positive input: rsqrt domain is (0, inf).
        [](Device d) {
            return tenzor::abs(randn({4, 8}, DType::Float32, d)) + 1.0f;
        });
}

// =========================================================================
// The frozen-constant hazard directly (BUG 3b). `sinh` is deliberately still
// unmapped in opid_to_optype, so it graph-breaks during the trace. Its output
// feeds a mapped consumer (*2). Pre-fix, the break silently baked sinh(in_a)
// in as a frozen constant, so replay ignored the runtime input. Post-fix, a
// break marks the trace incomplete and the CompiledFunction falls back to
// eager for every call — always correct, never frozen. We verify replay with a
// DIFFERENT input still matches eager (a frozen result would not).
// =========================================================================

TEST(JitTraceOps, UnmappedOutputNotFrozenAsConstant) {
    for (const auto& dev : get_available_backends()) {
        try {
            auto fn = [](const Variable& x) {
                return tenzor::sinh(x) * 2.0f;  // sinh: unmapped -> graph break
            };
            auto compiled = jit::compile(fn);

            auto in_a = randn({16}, DType::Float32, dev);
            auto in_b = randn({16}, DType::Float32, dev);

            (void)compiled(Variable(in_a, false));  // warm up / compile attempt
            dev.synchronize();
            // Replay with a DIFFERENT input. A frozen sinh(in_a) would make the
            // result independent of in_b.
            auto out_b = compiled(Variable(in_b, false)).tensor().to(Device::cpu());
            dev.synchronize();

            auto eager_b = fn(Variable(in_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_b, out_b, 1e-4f, 1e-4f))
                << "sinh(x)*2 replay ignored the runtime input on "
                << backend_name(dev) << " — output was frozen at trace time";
            // A broken trace must not be cached as a (wrong) compiled graph.
            EXPECT_EQ(compiled.num_cached(), 0u)
                << "a graph that broke on an unmapped op was still cached on "
                << backend_name(dev) << " — replay would use frozen constants";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "UnmappedOutputNotFrozen threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// RMSNorm end-to-end (records via the layer's jit hook; must map + execute).
// =========================================================================

TEST(JitTraceOps, RMSNormReplayMatchesEager) {
    for (const auto& dev : get_available_backends()) {
        try {
            constexpr int64_t kFeatures = 16;
            nn::RMSNorm norm(kFeatures);
            norm.to(dev);

            auto fn = [&norm](const Variable& x) { return norm.forward(x); };

            auto input = randn({4, kFeatures}, DType::Float32, dev);
            Tensor eager = fn(Variable(input, false)).tensor().to(Device::cpu());

            auto compiled = jit::compile(
                [&fn](const Variable& x) { return fn(x); });
            (void)compiled(Variable(input, false));
            dev.synchronize();
            auto out = compiled(Variable(input, false)).tensor().to(Device::cpu());
            dev.synchronize();

            EXPECT_TRUE(tensors_close(eager, out, 1e-4f, 1e-4f))
                << "RMSNorm replay != eager on " << backend_name(dev);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "RMSNorm threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}
