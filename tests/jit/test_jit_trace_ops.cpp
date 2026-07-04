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
#include <tenzor/nn/functional.hpp>

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

// Data-dependent predicate: a comparison (x > threshold) traced into a graph
// must be a REAL node that re-evaluates at replay, not a constant frozen to the
// trace-time boolean. Before comparison OpIds were mapped to OpTypes, gt/lt/…
// graph-broke and their output was baked as a constant, so a compiled predicate
// ignored its runtime input (breaking data-dependent JIT control flow). Runs on
// every available backend so the comparison OpId dispatches uniformly.
TEST(JitComparison, PredicateIsDataDependentAllBackends) {
    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));

        auto fn = [](const Variable& x) -> Variable {
            Tensor thr = full({4}, 0.5f, DType::Float32, x.tensor().device());
            return Variable(tenzor::gt(x.tensor(), thr), false);
        };

        auto compiled = jit::compile(fn);

        // Warm up (cache miss -> eager + graph build) with all-zeros:
        // pred = (0 > 0.5) = false everywhere.
        Tensor a = zeros({4}, DType::Float32, dev);
        (void)compiled(Variable(a, false));
        dev.synchronize();
        EXPECT_GE(compiled.num_cached(), 1u)
            << "comparison did not compile/cache a graph on " << backend_name(dev);

        // Replay (cache hit) with all-ones: pred must now be TRUE everywhere.
        // A frozen predicate would still yield the trace-time FALSE.
        Tensor b = ones({4}, DType::Float32, dev);
        Tensor replay = compiled(Variable(b, false)).tensor()
                            .to(Device::cpu()).to(DType::Float32).contiguous();
        dev.synchronize();

        const float* r = replay.data<float>();
        for (int i = 0; i < 4; ++i) {
            EXPECT_EQ(r[i], 1.0f)
                << "predicate frozen to trace-time value on "
                << backend_name(dev) << " i=" << i;
        }
    }
}

// count_include_pad=false must be captured by the tracer and honored by the
// interpreted executor at replay. It was previously dropped during tracing, so
// JIT silently used count_include_pad=true and diverged from eager at the padded
// border cells (which should divide by the real element count, not the window
// area). Non-zero padding makes the two modes differ.
TEST(JitAvgPool, CountIncludePadFalseReplayMatchesEagerAllBackends) {
    for (const auto& dev : get_available_backends()) {
        SCOPED_TRACE(backend_name(dev));
        auto fn = [](const Variable& x) -> Variable {
            return nn::functional::avg_pool2d(x, {2, 2}, {2, 2}, {1, 1},
                                              /*count_include_pad=*/false);
        };
        auto input = randn({1, 2, 4, 4}, DType::Float32, dev);
        Tensor eager = fn(Variable(input, false)).tensor().to(Device::cpu());

        auto compiled = jit::compile([&fn](const Variable& x) { return fn(x); });
        (void)compiled(Variable(input, false));
        dev.synchronize();
        EXPECT_GE(compiled.num_cached(), 1u)
            << "avg_pool2d did not compile/cache a graph on " << backend_name(dev);
        Tensor replay = compiled(Variable(input, false)).tensor().to(Device::cpu());
        dev.synchronize();

        EXPECT_TRUE(tensors_close(eager, replay, 1e-4f, 1e-4f))
            << "avg_pool2d(count_include_pad=false) replay != eager on "
            << backend_name(dev) << " — the flag was dropped at trace time";
    }
}

// =========================================================================
// JIT review fix C1: FuseLayerNormActivationPass folds a trailing ReLU/GELU
// into the LayerNorm node (marking fused_activation) and deletes the
// activation node. Pre-fix, the executor never read that marker, so the
// activation was silently DROPPED — replay returned a plain LayerNorm and
// diverged from eager by O(1) on every backend. These assert the fused
// activation is actually applied at replay.
// =========================================================================
TEST(JitTraceOps, LayerNormGeluFusionMatchesEager) {
    check_op_on_all_backends(
        "LayerNorm+GELU",
        [](const Variable& x) {
            int64_t d = x.tensor().shape().back();
            return nn::functional::gelu(
                nn::functional::layer_norm(x, {d}), "none");
        },
        [](Device dv) { return randn({4, 8}, DType::Float32, dv); });
}

TEST(JitTraceOps, LayerNormReluFusionMatchesEager) {
    check_op_on_all_backends(
        "LayerNorm+ReLU",
        [](const Variable& x) {
            int64_t d = x.tensor().shape().back();
            return nn::relu(nn::functional::layer_norm(x, {d}));
        },
        [](Device dv) { return randn({4, 8}, DType::Float32, dv); });
}

// =========================================================================
// JIT review fix C2: previously-unmapped ops (Var/Std/Prod, LeakyReLU/ELU/Mish/
// Softplus, Gather/Scatter/Flip/Roll, GroupNorm/InstanceNorm) are now first-
// class IR nodes, so they are CAPTURED and replayed through the JIT rather than
// graph-breaking to eager. check_op_on_all_backends asserts BOTH that the op
// compiled a graph (num_cached >= 1 — i.e. it did NOT graph-break) AND that the
// replayed result matches eager on every backend.
TEST(JitTraceOps, MappedReductionsMatchEager) {
    check_op_on_all_backends(
        "var(dim=1)",
        [](const Variable& x) { return tenzor::var(nn::relu(x), 1, /*keepdim=*/false); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "std(dim=1)",
        [](const Variable& x) { return tenzor::std(nn::relu(x), 1, /*keepdim=*/false); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "prod(dim=1)",
        [](const Variable& x) { return tenzor::prod(nn::relu(x) + 0.5f, 1, /*keepdim=*/false); },
        [](Device d) { return randn({4, 6}, DType::Float32, d); });
}

// gather / group_norm / flip / roll dispatch through OpId uniformly on EVERY
// backend (CPU flip now routes through cpu::flip_kernel via OpId::Flip instead
// of a manual memcpy that bypassed dispatch), so all are captured on all
// backends — no eager fallback anywhere.
TEST(JitTraceOps, MappedIndexingNormMatchEager) {
    check_op_on_all_backends(
        "flip(dim=1)",
        [](const Variable& x) { return tenzor::flip(x, {1}); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "roll(2,dim=1)",
        [](const Variable& x) { return tenzor::roll(x, 2, 1); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "gather(dim=1)",
        [](const Variable& x) {
            auto idx = zeros({4, 3}, DType::Int64, x.tensor().device());
            return tenzor::gather(x, 1, idx);
        },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "group_norm(2)",
        [](const Variable& x) { return nn::functional::group_norm(x, 2); },
        [](Device d) { return randn({2, 4, 3, 3}, DType::Float32, d); });
}

TEST(JitTraceOps, MappedActivationsMatchEager) {
    check_op_on_all_backends(
        "leaky_relu(0.05)",
        [](const Variable& x) { return tenzor::leaky_relu(x, 0.05); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "elu(1.5)",
        [](const Variable& x) { return tenzor::elu(x, 1.5f); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "mish",
        [](const Variable& x) { return tenzor::mish(x); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
    check_op_on_all_backends(
        "softplus",
        [](const Variable& x) { return tenzor::softplus(x, 1.0f); },
        [](Device d) { return randn({4, 8}, DType::Float32, d); });
}

// Regression (JIT review Fix #1): expand/broadcast_to and cat must be captured
// in the trace on EVERY backend, INCLUDING CPU. They previously bypassed dispatch
// on CPU (inline materialization into a zeros()/empty() output), so the tracer —
// which hooks the dispatch layer — never saw them and their output was frozen as
// a trace-time constant, severing the data dependency on the input. The compiled
// graph then returned the WARMUP's values for any later input. Warm up with input
// A, replay with a DIFFERENT input B (same shape -> same cache key), and require
// the compiled output to track B (== eager(B)); a frozen constant returns
// eager(A) and fails. The num_cached() assert rejects a silent eager passthrough.
TEST(JitTraceOps, ExpandCatTrackInputNotFrozen) {
    auto make_vec = [](std::vector<float> vals, Device dev) {
        auto t = zeros({static_cast<int64_t>(vals.size())}, DType::Float32,
                       Device::cpu());
        auto* d = t.data<float>();
        for (size_t i = 0; i < vals.size(); ++i) d[i] = vals[i];
        return t.to(dev);
    };
    auto fn = [](const Variable& x) -> Variable {
        // x: [3] -> broadcast to [2,3] -> cat with itself along dim 0 -> [4,3].
        auto b = tenzor::expand(x, {2, 3});
        return tenzor::cat({b, b}, 0);
    };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        // Warmup with A: cache miss -> eager + build/cache graph.
        auto a = make_vec({1.0f, 2.0f, 3.0f}, dev);
        (void)compiled(Variable(a, false));
        dev.synchronize();
        // Replay with a DIFFERENT input B.
        auto b = make_vec({10.0f, 20.0f, 30.0f}, dev);
        auto out = compiled(Variable(b, false));
        dev.synchronize();
        ASSERT_GT(compiled.num_cached(), 0u)
            << "expand+cat graph was not compiled/cached on " << dev.to_string();
        auto eager_b = fn(Variable(b, false)).tensor().to(Device::cpu());
        auto out_cpu = out.tensor().to(Device::cpu());
        auto diff = max(abs(eager_b - out_cpu)).template item<float>();
        EXPECT_LT(diff, 1e-5f)
            << "compiled expand+cat did not track input B (frozen constant?) on "
            << dev.to_string();
    }
}
