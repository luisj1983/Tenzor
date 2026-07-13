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
#include <tenzor/ops/linalg.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/nn/layers/flatten.hpp>
#include <tenzor/nn/layers/batchnorm.hpp>
#include <tenzor/nn/layers/sync_batchnorm.hpp>
#include <tenzor/nn/layers/conv.hpp>
#include <tenzor/nn/layers/transformer.hpp>
#include <tenzor/nn/layers/rope.hpp>
#include <tenzor/nn/utils/kv_cache.hpp>
#include <tenzor/nn/layers/sparse_linear.hpp>
#include <tenzor/nn/layers/pooling.hpp>
#include <tenzor/nn/layers/embedding.hpp>
#include <tenzor/nn/layers/dropout.hpp>
#include <tenzor/nn/layers/vision.hpp>
#include <tenzor/ops/vision.hpp>
#include <tenzor/ops/fft.hpp>
#include <tenzor/nn/layers/rnn.hpp>
#include <tenzor/nn/layers/moe.hpp>
#include <tenzor/nn/layers/hrm.hpp>
#include <tenzor/nn/quantization.hpp>
#include <tenzor/nn/loss/losses.hpp>
#include <tenzor/nn/loss/contrastive.hpp>
#include <tenzor/nn/activations/activations.hpp>
#include <tenzor/nested/nested_tensor.hpp>
#include <tenzor/nn/amp/autocast.hpp>
#include <tenzor/nn/utils/spectral_norm.hpp>
#include <tenzor/nn/utils/rnn_utils.hpp>
#include <tenzor/ops/detection.hpp>
#include <tenzor/nn/detection/roi_ops.hpp>
#include <tenzor/nn/detection/anchors.hpp>
#include <tenzor/nn/detection/rpn.hpp>
#include <tenzor/nn/detection/roi_head.hpp>
#include <tenzor/nn/detection/mask_head.hpp>

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

// Regression helper for JIT-R002/R032-R038 (dispatch-bypass capture): traces
// `fn` with a requires_grad=true input — the realistic path for any
// trainable model, since jit.compile()/jit.trace() never disable grad mode —
// then replays with a DIFFERENT input than the trace dummy. A dispatch-
// bypass bug (a raw Tensor-level view/copy op invisible to the tracer)
// bakes the trace-dummy's output as a frozen constant, so replay silently
// returns the WRONG value instead of tracking the new input. This is what
// distinguishes this check from check_op_on_all_backends() above, which
// always uses requires_grad=false and therefore never exercised the
// grad-enabled branch these bugs lived in, and from compiled_replay(),
// which reuses the SAME input on both calls and so can never observe a
// frozen constant either.
void check_op_grad_enabled_tracks_new_input(
        const char* label,
        const std::function<Variable(const Variable&)>& fn,
        const std::function<Tensor(Device)>& make_dummy,
        const std::function<Tensor(Device)>& make_other,
        float rtol = 1e-4f, float atol = 1e-4f) {
    for (const auto& dev : get_available_backends()) {
        try {
            auto dummy = make_dummy(dev);
            auto other = make_other(dev);

            auto compiled = jit::compile(
                [&fn](const Variable& x) { return fn(x); });

            Variable dummy_v(dummy, /*requires_grad=*/true);
            (void)compiled(dummy_v);  // cache miss: eager + trace
            dummy.device().synchronize();

            Variable other_v(other, /*requires_grad=*/true);
            auto replayed = compiled(other_v);  // cache hit: replay
            other.device().synchronize();

            Tensor eager_on_other =
                fn(Variable(other, false)).tensor().to(Device::cpu());
            Tensor replayed_cpu = replayed.tensor().to(Device::cpu());

            EXPECT_GE(compiled.num_cached(), 1u)
                << label << " (grad-enabled) did not compile/cache a graph on "
                << backend_name(dev)
                << " — the op likely graph-broke and silently fell back to "
                   "eager, so replay correctness was never exercised";

            EXPECT_TRUE(tensors_close(eager_on_other, replayed_cpu, rtol, atol))
                << label << " (grad-enabled): replay on a NEW input != eager "
                   "on that input on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << label << " (grad-enabled) threw on "
                          << backend_name(dev) << ": " << e.what();
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

// =========================================================================
// Conv1d / Conv3d rank-dispatch regression (JIT-055).
// Conv1dForward and Conv3dForward both trace to OpType::Conv2d; the executor
// must rank-dispatch on the input's spatial rank, not run everything as 2D.
// =========================================================================
namespace {
// Fixed-weight conv replay-vs-eager check (deterministic weights via ones()).
void check_conv_replay(const char* label,
                       const std::function<Variable(const Variable&, const Variable&)>& fn,
                       const std::vector<int64_t>& in_shape,
                       const std::vector<int64_t>& w_shape) {
    for (const auto& dev : get_available_backends()) {
        try {
            int64_t in_n = 1; for (auto s : in_shape) in_n *= s;
            Tensor xin = arange(0.0, static_cast<double>(in_n), 1.0,
                                DType::Float32, dev).reshape(in_shape);
            Variable w(ones(w_shape, DType::Float32, dev), false);
            Tensor eager = fn(Variable(xin, false), w).tensor().to(Device::cpu());
            auto rr = compiled_replay(
                [&fn, w](const Variable& x) { return fn(x, w); },
                xin);
            EXPECT_GE(rr.cached, 1u)
                << label << " graph-broke on " << backend_name(dev);
            EXPECT_TRUE(tensors_close(eager, rr.output, 1e-3f, 1e-3f))
                << label << " replay != eager on " << backend_name(dev);
        } catch (const std::exception& e) {
            ADD_FAILURE() << label << " threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}
}  // namespace

TEST(JitTraceOps, Conv1dRankDispatch) {
    check_conv_replay(
        "Conv1d",
        [](const Variable& x, const Variable& w) {
            return nn::functional::conv1d(x, w);
        },
        /*in=*/{1, 2, 8}, /*w=*/{3, 2, 3});
}

TEST(JitTraceOps, Conv3dRankDispatch) {
    check_conv_replay(
        "Conv3d",
        [](const Variable& x, const Variable& w) {
            return nn::functional::conv3d(x, w);
        },
        /*in=*/{1, 2, 4, 4, 4}, /*w=*/{3, 2, 2, 2, 2});
}

TEST(JitTraceOps, Conv2dStillWorks) {
    check_conv_replay(
        "Conv2d",
        [](const Variable& x, const Variable& w) {
            return nn::functional::conv2d(x, w);
        },
        /*in=*/{1, 2, 8, 8}, /*w=*/{3, 2, 3, 3});
}

// =========================================================================
// squeeze/unsqueeze must be first-class traced nodes (JIT-069). A graph whose
// OUTPUT is a squeeze/unsqueeze view was previously invisible to the tracer
// (raw Tensor view, no dispatch) and produced no graph output at all.
// =========================================================================
TEST(JitTraceOps, SqueezeOutputIsTraced) {
    check_op_on_all_backends(
        "relu_then_squeeze",
        [](const Variable& x) { return tenzor::squeeze(nn::relu(x), 1); },
        [](Device d) { return arange(0.0, 8.0, 1.0, DType::Float32, d).reshape({2, 1, 4}); });
}
TEST(JitTraceOps, UnsqueezeOutputIsTraced) {
    check_op_on_all_backends(
        "relu_then_unsqueeze",
        [](const Variable& x) { return tenzor::unsqueeze(nn::relu(x), 2); },
        [](Device d) { return arange(0.0, 8.0, 1.0, DType::Float32, d).reshape({2, 4}); });
}

// =========================================================================
// JIT-R002/R032-R038 regression coverage: dispatch-bypass capture on the
// GRAD-ENABLED path. jit.compile()/jit.trace() never disable grad mode, so
// any input derived from trainable weights takes this branch — the ordinary
// case, not a corner case. The bug: these ops called a raw Tensor-level
// view/copy function (invisible to the dispatch-level JIT tracer) instead
// of dispatch()/an explicit tracer registration, so the compiled graph
// froze the trace-time output as a constant and silently ignored the real
// input on every subsequent call. check_op_grad_enabled_tracks_new_input
// traces with one input and replays with a DIFFERENT one — the only way to
// observe a frozen constant (same-input replay can't tell the difference).
// =========================================================================

TEST(JitTraceOps, SqueezeGradEnabledTracksNewInput) {
    // Matches the exact motivating case cited in squeeze()'s own comment:
    // a trailing squeeze on a grad-requiring activation (e.g. F::conv1d).
    check_op_grad_enabled_tracks_new_input(
        "squeeze(grad-enabled)",
        [](const Variable& x) { return tenzor::squeeze(x, 1); },
        [](Device d) { return full({2, 1, 4}, 1.0, DType::Float32, d); },
        [](Device d) { return full({2, 1, 4}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, UnsqueezeGradEnabledTracksNewInput) {
    check_op_grad_enabled_tracks_new_input(
        "unsqueeze(grad-enabled)",
        [](const Variable& x) { return tenzor::unsqueeze(x, 2); },
        [](Device d) { return full({2, 4}, 1.0, DType::Float32, d); },
        [](Device d) { return full({2, 4}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, CloneNoGradTracksNewInput) {
    // clone()'s bug was in the NO-grad branch specifically (the grad-enabled
    // branch already went through Mul, which dispatches correctly).
    check_op_grad_enabled_tracks_new_input(
        "clone(no-grad)",
        [](const Variable& x) {
            return tenzor::clone(Variable(x.tensor(), false)) * 2.0;
        },
        [](Device d) { return full({4}, 1.0, DType::Float32, d); },
        [](Device d) { return full({4}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, NarrowTracksNewInput) {
    // The KV-cache-windowing case: narrow() must track the real input, not
    // freeze the trace-time slice — both branches were previously broken.
    check_op_grad_enabled_tracks_new_input(
        "narrow",
        [](const Variable& x) { return tenzor::narrow(x, 0, 1, 2) * 3.0; },
        [](Device d) { return arange(0.0, 4.0, 1.0, DType::Float32, d); },
        [](Device d) { return arange(10.0, 14.0, 1.0, DType::Float32, d); });
}

TEST(JitTraceOps, AsStridedTracksNewInput) {
    check_op_grad_enabled_tracks_new_input(
        "as_strided",
        [](const Variable& x) {
            std::array<int64_t, 1> size{2};
            std::array<int64_t, 1> stride{1};
            return tenzor::as_strided(x, size, stride, 0) * 2.0;
        },
        [](Device d) { return arange(0.0, 4.0, 1.0, DType::Float32, d); },
        [](Device d) { return arange(10.0, 14.0, 1.0, DType::Float32, d); });
}

TEST(JitTraceOps, ViewAsRealTracksNewInput) {
    check_op_grad_enabled_tracks_new_input(
        "view_as_real",
        [](const Variable& x) { return tenzor::view_as_real(x) * 2.0; },
        [](Device d) {
            return full({2}, 1.0, DType::Float32, d).to(DType::Complex64);
        },
        [](Device d) {
            return full({2}, 7.0, DType::Float32, d).to(DType::Complex64);
        });
}

TEST(JitTraceOps, ViewAsComplexTracksNewInput) {
    check_op_grad_enabled_tracks_new_input(
        "view_as_complex",
        [](const Variable& x) {
            // Keep the result Complex64 (no round-trip back to real) — a
            // view_as_real(view_as_complex(x)) round trip can coincidentally
            // resolve to the live input via tracer fingerprint collision and
            // mask this exact bug class.
            return tenzor::view_as_real(tenzor::view_as_complex(x) * 2.0);
        },
        [](Device d) { return full({2, 2}, 1.0, DType::Float32, d); },
        [](Device d) { return full({2, 2}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, TriuTracksNewInputOnCpu) {
    // The CPU-only device-conditional bypass (JIT-R038): triu/tril/diag/
    // trace only dispatched on non-CPU devices before the fix.
    check_op_grad_enabled_tracks_new_input(
        "triu",
        [](const Variable& x) { return tenzor::triu(x, 0) * 2.0; },
        [](Device) { return full({3, 3}, 1.0, DType::Float32, Device::cpu()); },
        [](Device) { return full({3, 3}, 7.0, DType::Float32, Device::cpu()); });
}

TEST(JitTraceOps, TrilTracksNewInputOnCpu) {
    check_op_grad_enabled_tracks_new_input(
        "tril",
        [](const Variable& x) { return tenzor::tril(x, 0) * 2.0; },
        [](Device) { return full({3, 3}, 1.0, DType::Float32, Device::cpu()); },
        [](Device) { return full({3, 3}, 7.0, DType::Float32, Device::cpu()); });
}

TEST(JitTraceOps, DiagTracksNewInputOnCpu) {
    check_op_grad_enabled_tracks_new_input(
        "diag",
        [](const Variable& x) { return tenzor::diag(x, 0) * 2.0; },
        [](Device) { return arange(1.0, 4.0, 1.0, DType::Float32, Device::cpu()); },
        [](Device) { return arange(10.0, 13.0, 1.0, DType::Float32, Device::cpu()); });
}

TEST(JitTraceOps, TraceTracksNewInputOnCpu) {
    check_op_grad_enabled_tracks_new_input(
        "trace",
        [](const Variable& x) { return tenzor::trace(x) * 2.0; },
        [](Device) { return full({3, 3}, 1.0, DType::Float32, Device::cpu()); },
        [](Device) { return full({3, 3}, 7.0, DType::Float32, Device::cpu()); });
}

TEST(JitTraceOps, ToDeviceTracksNewInputCrossBackend) {
    // Cross-device transfer (JIT-R034) needs a genuine second backend to
    // exercise — CPU-only environments legitimately skip this (see
    // REQUIRE_MULTI_BACKEND_OR_SKIP's doc comment).
    REQUIRE_MULTI_BACKEND_OR_SKIP("ToDevice cross-backend replay");
    auto backends = get_available_backends();
    Device other_dev = (backends[0].type == Device::Type::CPU) ? backends[1] : backends[0];

    check_op_grad_enabled_tracks_new_input(
        "to_device",
        [other_dev](const Variable& x) {
            return tenzor::to_device(x, other_dev) * 2.0;
        },
        [](Device) { return full({4}, 1.0, DType::Float32, Device::cpu()); },
        [](Device) { return full({4}, 7.0, DType::Float32, Device::cpu()); });
}

// =========================================================================
// JIT-R045 regression coverage: nn::Flatten/Unflatten/PixelShuffle/
// PixelUnshuffle/ChannelShuffle. Flatten/Unflatten only call
// Tensor::reshape() (proven above, JIT-R001 investigation, to always
// dispatch OpId::Reshape) so they were NOT actually broken — included here
// to lock that in as a regression test, not because a fix was needed.
// PixelShuffle/PixelUnshuffle/ChannelShuffle DO have a real bug: they chain
// a raw Tensor::permute() (zero dispatch) into .contiguous() (dispatches,
// but aliases its output to the untracked permuted tensor), so the traced
// permute+contiguous+reshape tail froze at the trace-time value. Fixed by
// routing the permute step through the Variable-level, tracer-visible
// tenzor::permute().
// =========================================================================

TEST(JitTraceOps, FlattenModuleGradEnabledTracksNewInput) {
    nn::Flatten flatten(1, -1);
    check_op_grad_enabled_tracks_new_input(
        "nn::Flatten",
        [&flatten](const Variable& x) { return flatten.forward(x) * 2.0; },
        [](Device d) { return full({2, 3, 4}, 1.0, DType::Float32, d); },
        [](Device d) { return full({2, 3, 4}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, UnflattenModuleGradEnabledTracksNewInput) {
    nn::Unflatten unflatten(1, {3, 4});
    check_op_grad_enabled_tracks_new_input(
        "nn::Unflatten",
        [&unflatten](const Variable& x) { return unflatten.forward(x) * 2.0; },
        [](Device d) { return full({2, 12}, 1.0, DType::Float32, d); },
        [](Device d) { return full({2, 12}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, PixelShuffleGradEnabledTracksNewInput) {
    nn::PixelShuffle ps(2);
    check_op_grad_enabled_tracks_new_input(
        "nn::PixelShuffle",
        [&ps](const Variable& x) { return ps.forward(x) * 2.0; },
        [](Device d) { return full({1, 8, 2, 2}, 1.0, DType::Float32, d); },
        [](Device d) { return full({1, 8, 2, 2}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, PixelUnshuffleGradEnabledTracksNewInput) {
    nn::PixelUnshuffle pus(2);
    check_op_grad_enabled_tracks_new_input(
        "nn::PixelUnshuffle",
        [&pus](const Variable& x) { return pus.forward(x) * 2.0; },
        [](Device d) { return full({1, 2, 4, 4}, 1.0, DType::Float32, d); },
        [](Device d) { return full({1, 2, 4, 4}, 7.0, DType::Float32, d); });
}

TEST(JitTraceOps, ChannelShuffleGradEnabledTracksNewInput) {
    nn::ChannelShuffle cs(2);
    check_op_grad_enabled_tracks_new_input(
        "nn::ChannelShuffle",
        [&cs](const Variable& x) { return cs.forward(x) * 2.0; },
        [](Device d) { return full({1, 4, 3, 3}, 1.0, DType::Float32, d); },
        [](Device d) { return full({1, 4, 3, 3}, 7.0, DType::Float32, d); });
}

// =========================================================================
// JIT-R048 regression coverage: BatchNorm1d's 3D (N,C,L) path chains raw
// Tensor::permute()/unsqueeze() (zero dispatch) into .contiguous() (which
// aliases to that untracked value), so a traced BatchNorm1d call froze the
// whole normalize+affine tail at its trace-time value. Uses randomised
// (not `full`) inputs since BatchNorm's own statistics are computed FROM
// the input — a constant input would make every per-channel value
// identical, trivially "passing" even a frozen computation.
// =========================================================================

TEST(JitTraceOps, BatchNorm1d3DTrainingGradEnabledTracksNewInput) {
    nn::BatchNorm1d bn(4);
    bn.train();
    check_op_grad_enabled_tracks_new_input(
        "nn::BatchNorm1d(3D, training)",
        [&bn](const Variable& x) { return bn.forward(x) * 2.0; },
        [](Device d) { return randn({2, 4, 5}, DType::Float32, d) + 1.0f; },
        [](Device d) { return randn({2, 4, 5}, DType::Float32, d) + 100.0f; });
}

TEST(JitTraceOps, BatchNorm1d2DTrainingGradEnabledTracksNewInput) {
    nn::BatchNorm1d bn(4);
    bn.train();
    check_op_grad_enabled_tracks_new_input(
        "nn::BatchNorm1d(2D, training)",
        [&bn](const Variable& x) { return bn.forward(x) * 2.0; },
        [](Device d) { return randn({8, 4}, DType::Float32, d) + 1.0f; },
        [](Device d) { return randn({8, 4}, DType::Float32, d) + 100.0f; });
}

// =========================================================================
// JIT-R049 regression coverage: SyncBatchNorm was 100% tracer-invisible.
//
// Two distinct claims to verify:
//  1. world_size==1 (no real collective fires) IS a pure function of the
//     input and must trace/replay correctly like any other layer — this
//     exercises the packed1 slice()/full() fixes in forward_impl.
//  2. world_size>1 with a genuine all_reduce_fn_ is architecturally
//     untraceable (the all-reduce is a host-side side effect with no graph
//     representation), so tracing it must graph-break to eager fallback
//     rather than silently freeze the trace-time LOCAL (pre-reduce) stats
//     into a compiled graph — the exact bug JIT-R049 described.
// =========================================================================

TEST(JitTraceOps, SyncBatchNormSingleProcessGradEnabledTracksNewInput) {
    // intentionally exercising deprecated legacy SyncBatchNorm ctor
    nn::AllReduceFn identity_all_reduce = [](Tensor&) {};
    nn::SyncBatchNorm sbn(4, identity_all_reduce, /*world_size=*/1);
    sbn.train();
    check_op_grad_enabled_tracks_new_input(
        "nn::SyncBatchNorm(world_size=1, training)",
        [&sbn](const Variable& x) { return sbn.forward(x) * 2.0; },
        [](Device d) { return randn({2, 4, 3, 3}, DType::Float32, d) + 1.0f; },
        [](Device d) { return randn({2, 4, 3, 3}, DType::Float32, d) + 100.0f; });
}

TEST(JitTraceOps, SyncBatchNormMultiWorkerThrowsInsteadOfFreezingLocalStats) {
    // intentionally exercising deprecated legacy SyncBatchNorm ctor
    //
    // A world_size>1 all-reduce is a host-side side effect with NO graph
    // representation: jit::compile()'s trace runs the user closure exactly
    // once (eager result == trace result, JIT-008), so there is no
    // "already-computed correct eager value" to silently fall back to the
    // way an unmapped-op graph break can (compile.cpp's trace_and_compile:
    // `if (!fn_ran) throw;` — a mid-trace exception propagates exactly like
    // a plain eager call throwing). The fix therefore fails LOUDLY: throwing
    // out of forward_impl is the only way to guarantee a compiled graph is
    // never built/cached with frozen pre-reduce local stats.
    nn::AllReduceFn doubling_all_reduce = [](Tensor& t) { t = t * 2.0; };
    nn::SyncBatchNorm sbn(4, doubling_all_reduce, /*world_size=*/2);
    sbn.train();

    auto fn = [&sbn](const Variable& x) { return sbn.forward(x) * 2.0; };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({2, 4, 3, 3}, DType::Float32, dev) + 1.0f;

        EXPECT_THROW(
            { (void)compiled(Variable(in_a, /*requires_grad=*/true)); },
            std::exception)
            << "SyncBatchNorm(world_size=2) did not throw when JIT-traced on "
            << backend_name(dev) << " — a multi-worker all-reduce would be "
               "silently frozen as a constant instead";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "SyncBatchNorm(world_size=2) cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        // The guard must only fire while actually tracing: a plain eager
        // call on the SAME layer instance (no jit.compile involved) must
        // still work normally and call the real all-reduce.
        auto in_eager = randn({2, 4, 3, 3}, DType::Float32, dev) + 1.0f;
        EXPECT_NO_THROW({ (void)fn(Variable(in_eager, false)); })
            << "SyncBatchNorm(world_size=2) incorrectly threw on a plain "
               "eager (non-traced) call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R052 regression coverage: ConvTranspose1d wrapped the (properly
// dispatched) ConvTranspose2dForward call with raw Tensor::unsqueeze(2)/
// squeeze(2) (zero dispatch) and, when padding_>0, a raw ops::slice(Tensor)
// call (also zero dispatch — a thin `return input.slice(...)` wrapper, NOT
// the dispatched Variable-level tenzor::slice()). A traced call froze the
// pre/post-reshape data at its trace-time value.
// =========================================================================

TEST(JitTraceOps, ConvTranspose1dPaddingTracksNewInput) {
    // check_op_grad_enabled_tracks_new_input is USELESS here: ConvTranspose's
    // grad-mode replay is unconditionally unsupported ("ConvTranspose is not
    // wired for differentiable replay"), so a requires_grad=true trace always
    // falls back to genuine eager autograd on EVERY call regardless of
    // whether the unsqueeze/squeeze/slice fix below is present — confirmed
    // by temporarily reverting the fix and observing the grad-enabled test
    // still pass vacuously. Drive the INFERENCE (non-grad) compile path
    // directly instead, which genuinely replays through the graph
    // interpreter and would expose a frozen pre/post-reshape constant.
    nn::ConvTranspose1d ct(4, 6, /*kernel_size=*/3, /*stride=*/1,
                            /*padding=*/1);
    auto fn = [&ct](const Variable& x) { return ct.forward(x) * 2.0; };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({2, 4, 5}, DType::Float32, dev) + 1.0f;
            auto other = randn({2, 4, 5}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::ConvTranspose1d(padding=1) did not compile/cache a "
                   "graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "nn::ConvTranspose1d(padding=1): replay on a NEW input != "
                   "eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::ConvTranspose1d(padding=1) threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R054 regression coverage: F::multi_head_attention_forward's Q/K/V
// split/transpose/merge pipeline used the Tensor-level tenzor::slice()/
// transpose()/permute() overloads — thin wrappers around the raw
// Tensor::slice()/transpose()/permute() metadata ops, zero dispatch(). A
// traced call froze the whole projection+attention pipeline. This function
// is intentionally non-differentiable (Tensor-in/Tensor-out, no grad_fn),
// so we drive it through check_op_grad_enabled_tracks_new_input purely to
// exercise jit.compile()'s always-grad-enabled trace path — not to assert
// any gradient behavior.
// =========================================================================

TEST(JitTraceOps, MultiHeadAttentionForwardTracksNewInput) {
    const int64_t embed_dim = 8, num_heads = 2, batch = 2, seq = 3;
    for (const auto& dev : get_available_backends()) {
        try {
            auto in_proj_weight = randn({3 * embed_dim, embed_dim}, DType::Float32, dev);
            auto in_proj_bias = randn({3 * embed_dim}, DType::Float32, dev);
            auto out_proj_weight = randn({embed_dim, embed_dim}, DType::Float32, dev);
            auto out_proj_bias = randn({embed_dim}, DType::Float32, dev);

            auto fn = [&](const Tensor& q) {
                auto result = nn::functional::multi_head_attention_forward(
                    q, q, q, num_heads,
                    in_proj_weight, in_proj_bias, out_proj_weight, out_proj_bias,
                    std::nullopt, 0.0, false, false);
                return result.first;
            };

            auto compiled = jit::compile(
                [&fn](const Variable& x) { return Variable(fn(x.tensor()), false); });

            auto dummy = randn({batch, seq, embed_dim}, DType::Float32, dev);
            auto other = randn({batch, seq, embed_dim}, DType::Float32, dev) + 50.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/true));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "F::multi_head_attention_forward did not compile/cache a "
                   "graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/true))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(other).to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "F::multi_head_attention_forward: replay on a NEW input != "
                   "eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "F::multi_head_attention_forward threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R057 regression coverage: PositionalEncoding::forward_impl sliced its
// cached PE buffer via raw Tensor::slice() (zero dispatch), so the PE
// contribution to the output was never registered with the tracer — a
// downstream consumer (`x + pe_var`) resolves it as an opaque frozen
// constant rather than a traced Slice node. PE's own VALUE only depends on
// seq_len (not on x's data), so the standard same-shape/different-VALUE
// check can't distinguish frozen-vs-traced here; instead verify a fresh
// compiled function at each of two DIFFERENT seq_lens produces the correct
// (different) PE contribution.
// =========================================================================

TEST(JitTraceOps, PositionalEncodingTracksSeqLen) {
    const int64_t d_model = 8;
    nn::PositionalEncoding pe(d_model, /*max_len=*/64);

    auto fn = [&pe](const Variable& x) { return pe.forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);

            auto zeros_len4 = zeros({1, 4, d_model}, DType::Float32, dev);
            auto out4 = compiled(Variable(zeros_len4, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager4 = fn(Variable(zeros_len4, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager4, out4, 1e-4f, 1e-4f))
                << "PositionalEncoding(seq_len=4) compiled replay != eager on "
                << backend_name(dev);

            auto zeros_len6 = zeros({1, 6, d_model}, DType::Float32, dev);
            auto out6 = compiled(Variable(zeros_len6, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager6 = fn(Variable(zeros_len6, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager6, out6, 1e-4f, 1e-4f))
                << "PositionalEncoding(seq_len=6) compiled replay != eager on "
                << backend_name(dev)
                << " — the PE slice was likely frozen at the seq_len=4 trace";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "PositionalEncoding threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R083 regression coverage: RoPE::forward(input, offset)'s cos/sin
// buffer slice used raw Tensor::slice() (zero dispatch) — the SAME
// primitive as JIT-R033/R057. Fixing that alone is NOT sufficient: `offset`
// is a host int64_t captured by the tracing closure, baked as a Slice
// node's constant attribute. jit.compile() caches purely on input shape, so
// a second call at a DIFFERENT offset but the SAME shape (the canonical
// KV-cache incremental-decode pattern — trace at offset=0/prefill, decode
// at offset=1,2,...) is a cache HIT that replays the captured graph
// directly, silently reusing the WRONG (trace-time) rotation — confirmed
// empirically before adding the fix below. RoPE::forward therefore refuses
// to be traced at all (throws whenever Tracer::is_tracing()), guaranteeing
// no such graph is ever cached. Eager (non-traced) calls are unaffected for
// any offset.
// =========================================================================

TEST(JitTraceOps, RoPERefusesToTraceInsteadOfFreezingOffset) {
    nn::RoPE rope(/*dim=*/8, /*max_seq_len=*/64);
    int64_t offset = 0;
    auto fn = [&rope, &offset](const Variable& x) { return rope.forward(x, offset); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({2, 4, 8}, DType::Float32, dev) + 1.0f;

        EXPECT_THROW(
            { (void)compiled(Variable(in_a, /*requires_grad=*/true)); },
            std::exception)
            << "RoPE::forward did not throw when JIT-traced on "
            << backend_name(dev) << " — a later call at a different offset "
               "would silently replay the wrong (trace-time) rotation";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "RoPE::forward cached a compiled graph on " << backend_name(dev)
            << " despite throwing during trace";

        // The guard must only fire while actually tracing: plain eager calls
        // (no jit.compile involved) must still work normally for ANY offset,
        // including a genuinely varying one across calls.
        for (int64_t eager_offset : {0, 1, 2}) {
            offset = eager_offset;
            auto in_eager = randn({2, 4, 8}, DType::Float32, dev) + 1.0f;
            EXPECT_NO_THROW({ (void)fn(Variable(in_eager, false)); })
                << "RoPE::forward incorrectly threw on a plain eager "
                   "(non-traced) call at offset=" << eager_offset << " on "
                << backend_name(dev);
        }
        offset = 0;
    }
}

// =========================================================================
// JIT-R063 regression coverage: VariationalDropout's mask_ is cached member
// state reused across calls by design (same mask for a whole sequence)
// until reset_mask() or a shape change. A trace can only capture whichever
// mask happened to be resident at trace time, baking it into the compiled
// graph as a constant — a later eager reset_mask() (new sequence) would be
// silently ignored by replay. Mirrors RoPE/KVCache/WindowAttention's
// established refuse-to-trace fix.
// =========================================================================

TEST(JitTraceOps, VariationalDropoutRefusesToTraceInTrainingMode) {
    nn::VariationalDropout vdrop(/*p=*/0.5);
    vdrop.train();
    auto fn = [&vdrop](const Variable& x) { return vdrop.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({4, 8}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(in_a, /*requires_grad=*/true)); },
            std::exception)
            << "VariationalDropout::forward did not throw when JIT-traced "
               "in training mode on " << backend_name(dev)
            << " — a later reset_mask() would silently replay the stale "
               "trace-time mask";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "VariationalDropout::forward cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        // The guard must only fire while actually tracing: plain eager
        // calls (no jit.compile involved) must still work normally.
        vdrop.reset_mask();
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "VariationalDropout::forward incorrectly threw on a plain "
               "eager (non-traced) training-mode call on "
            << backend_name(dev);

        // Eval mode is a pure identity pass-through — no randomness, no
        // state — and must remain safe to call through jit.compile() (no
        // throw). It legitimately produces an EMPTY trace (there is no
        // computation at all to record — `output` literally IS `input`),
        // which is a correct, harmless "graph break" fallback to eager,
        // not a caching failure to assert against.
        vdrop.eval();
        auto compiled_eval = jit::compile(fn);
        Variable eval_out;
        ASSERT_NO_THROW({ eval_out = compiled_eval(Variable(in_a, false)); })
            << "VariationalDropout::forward(eval mode) threw when JIT-traced "
               "on " << backend_name(dev);
        EXPECT_LT(max_abs_diff(in_a, eval_out.tensor()), 1e-6f)
            << "VariationalDropout::forward(eval mode) is not an identity "
               "pass-through on " << backend_name(dev);
        vdrop.train();
    }
}

// =========================================================================
// JIT-R065 regression coverage: FakeQuantize's observer-driven qparams
// (scale/zero_point) are recomputed from observed statistics whenever
// observer_enabled_ && !learnable_ && training_ — the same condition that
// gates calculate_qparams(). Those values are read as plain host floats and
// baked into the STE path as trace-time constants, so a trace would freeze
// QAT calibration at whatever snapshot existed when tracing happened.
// =========================================================================

TEST(JitTraceOps, FakeQuantizeRefusesToTraceWhileCalibrating) {
    nn::quantization::FakeQuantize fq(
        nn::quantization::QuantDType::INT8,
        nn::quantization::QuantizationScheme::PerTensorSymmetric,
        /*learnable=*/false, /*observer_enabled=*/true);
    fq.train();
    auto fn = [&fq](const Variable& x) { return fq.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({4, 8}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(in_a, /*requires_grad=*/true)); },
            std::exception)
            << "FakeQuantize::forward did not throw when JIT-traced while "
               "calibrating on " << backend_name(dev)
            << " — calibration would silently freeze at the trace-time "
               "scale/zero_point snapshot";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "FakeQuantize::forward cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        // The guard must only fire while actually tracing: plain eager
        // calls (no jit.compile involved) must still calibrate normally.
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "FakeQuantize::forward incorrectly threw on a plain eager "
               "(non-traced) calibrating call on " << backend_name(dev);

        // Disabling the observer removes the JIT-R065 hazard (fixed
        // qparams_, nothing left to recalibrate). The JIT-R066 hazard
        // (apply_fake_quantization's raw, dispatch-invisible math) is
        // separate and still applies to the no-grad fall-through path —
        // only the differentiable STE path (input.requires_grad()==true)
        // must not THROW. It legitimately does not produce a cached graph
        // (FakeQuantizeFunction is a hand-written custom autograd Function;
        // grad-mode JIT tracing safely falls back to eager autograd for
        // custom Functions it cannot translate — a correct, harmless
        // "graph break," not a caching failure to assert against).
        fq.enable_observer(false);
        auto compiled_fixed = jit::compile(fn);
        auto in_grad = Variable(in_a, /*requires_grad=*/true);
        EXPECT_NO_THROW({ (void)compiled_fixed(in_grad); })
            << "FakeQuantize::forward(observer disabled, requires_grad) "
               "threw when JIT-traced on " << backend_name(dev);
        fq.enable_observer(true);
    }
}

// =========================================================================
// JIT-R090 regression coverage: KVCache::update()/get_keys()/get_values()
// used raw ops::slice(Tensor,...) (zero dispatch, the JIT-R033 primitive)
// plus a raw std::memcpy fast path that never touches dispatch() at all.
// Fixing dispatch-visibility alone is NOT sufficient: `pos`/`seq_len` are
// baked into the compiled graph as constant Slice/SliceScatter attributes,
// AND the persistent cache buffer is not a declared nn::Module parameter,
// so end_trace() freezes it as an opaque trace-time snapshot. Empirically
// confirmed (before adding the fix below): tracing update() at pos=0 then
// replaying at pos=1 produced both the WRONG shape and WRONG content — the
// silent-corruption scenario this finding describes for the standard
// autoregressive decode loop. update()/get_keys()/get_values() therefore
// refuse to be traced at all, guaranteeing no such graph is ever cached.
// Eager (non-traced) calls are unaffected and correctly advance the cache.
// =========================================================================

TEST(JitTraceOps, KVCacheRefusesToTraceInsteadOfFreezingState) {
    nn::utils::KVCacheConfig cfg;
    cfg.num_layers = 1;
    cfg.max_seq_len = 8;
    cfg.num_kv_heads = 2;
    cfg.head_dim = 4;
    cfg.batch_size = 1;

    for (const auto& dev : get_available_backends()) {
        cfg.device = dev;
        nn::utils::KVCache cache(cfg);
        int64_t pos = 0;
        auto fn = [&cache, &pos](const Variable& k) {
            auto result = cache.update(0, k.tensor(), k.tensor(), pos);
            return Variable(result.first, false);
        };
        auto compiled = jit::compile(fn);
        auto k0 = full({1, 2, 1, 4}, 1.0, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(k0, /*requires_grad=*/true)); },
            std::exception)
            << "KVCache::update did not throw when JIT-traced on "
            << backend_name(dev) << " — a later call at a different pos "
               "would silently replay stale/wrong-shaped cache content";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "KVCache::update cached a compiled graph on " << backend_name(dev)
            << " despite throwing during trace";

        // Plain eager (non-traced) usage must still correctly advance the
        // cache across successive positions.
        nn::utils::KVCache eager_cache(cfg);
        auto ek0 = full({1, 2, 1, 4}, 1.0, DType::Float32, dev);
        auto ek1 = full({1, 2, 1, 4}, 7.0, DType::Float32, dev);
        auto r0 = eager_cache.update(0, ek0, ek0, 0);
        auto r1 = eager_cache.update(0, ek1, ek1, 1);
        EXPECT_EQ(r1.first.shape()[2], 2)
            << "KVCache::update eager path did not correctly grow the "
               "cached length on " << backend_name(dev);
        auto read_back = eager_cache.get_keys(0, 2).to(Device::cpu());
        auto expected = cat({ek0, ek1}, 2).to(Device::cpu());
        EXPECT_TRUE(tensors_close(expected, read_back, 1e-4f, 1e-4f))
            << "KVCache eager update+get_keys did not correctly accumulate "
               "cache content on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R070 regression coverage: WindowAttention::forward's attention-mask
// application used raw Tensor::unsqueeze() (zero dispatch) on the `mask`
// parameter, freezing it as a trace-time constant. Keeps input SHAPE fixed
// across dummy/other and varies only the mask's CONTENT — the standard
// same-shape/different-value frozen-constant check.
// =========================================================================

namespace {
// A UNIFORM mask value shift is invisible to the FINAL output: softmax is
// exactly shift-invariant across a row when every element in that row gets
// the same additive constant, so full(...) masks of different uniform
// values would make this whole test vacuous regardless of whether the mask
// is correctly traced. Build a genuine CHECKERBOARD ((i+j) odd -> -100,
// else 0) so changing `flip` changes the RELATIVE per-row weighting (the
// realistic Swin shifted-window mask shape: some position-pairs masked,
// others not, within the SAME row).
auto make_checkerboard_mask(int64_t num_windows, int64_t N, bool flip, Device dev) -> Tensor {
    auto t_cpu = zeros({num_windows, N, N}, DType::Float32, Device::cpu());
    auto* ptr = t_cpu.data<float>();
    for (int64_t w = 0; w < num_windows; ++w) {
        for (int64_t i = 0; i < N; ++i) {
            for (int64_t j = 0; j < N; ++j) {
                bool masked = ((i + j) % 2 == 1) != flip;
                ptr[(w * N + i) * N + j] = masked ? -100.0f : 0.0f;
            }
        }
    }
    return (dev.type == Device::Type::CPU) ? t_cpu : t_cpu.to(dev);
}
}  // namespace

TEST(JitTraceOps, WindowAttentionRefusesToTraceInsteadOfFreezingMask) {
    // `mask` is a plain (non-Parameter) Tensor argument — end_trace() freezes
    // it as an opaque trace-time constant rather than a live-rebound leaf.
    // Empirically confirmed (before adding the fix below): fixing the raw
    // Tensor::unsqueeze() calls alone was NOT sufficient — tracing once and
    // replaying with a different (checkerboard-flipped) mask value at the
    // SAME shape silently kept applying the trace-time mask. Matches
    // JIT-R083 (RoPE)/JIT-R090 (KVCache)'s "fail loudly instead of producing
    // wrong numerics" pattern: WindowAttention::forward refuses to trace at
    // all whenever a real mask is supplied.
    nn::WindowAttention wa(/*dim=*/8, /*window_size=*/2, /*num_heads=*/2);
    const int64_t num_windows = 2, N = 4;  // N = window_size^2
    Tensor mask;
    auto fn = [&wa, &mask](const Variable& x) { return wa.forward(x, mask); };
    for (const auto& dev : get_available_backends()) {
        mask = make_checkerboard_mask(num_windows, N, /*flip=*/false, dev);
        auto compiled = jit::compile(fn);
        auto in_a = randn({num_windows, N, 8}, DType::Float32, dev) + 1.0f;

        EXPECT_THROW(
            { (void)compiled(Variable(in_a, /*requires_grad=*/true)); },
            std::exception)
            << "nn::WindowAttention did not throw when JIT-traced with a "
               "mask on " << backend_name(dev) << " — a later call with a "
               "different mask would silently replay the wrong (trace-time) "
               "mask";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "nn::WindowAttention cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        // The guard must only fire while actually tracing: plain eager calls
        // must still work normally for any mask, including changing masks
        // across calls.
        for (bool flip : {false, true}) {
            mask = make_checkerboard_mask(num_windows, N, flip, dev);
            auto in_eager = randn({num_windows, N, 8}, DType::Float32, dev) + 1.0f;
            EXPECT_NO_THROW({ (void)fn(Variable(in_eager, false)); })
                << "nn::WindowAttention incorrectly threw on a plain eager "
                   "(non-traced) call on " << backend_name(dev);
        }
    }
}

TEST(JitTraceOps, WindowAttentionNoMaskStillTracesNewInput) {
    // No mask supplied at all: the throw above never fires (gated on
    // mask.is_valid()), so the common no-mask (non-shifted-window) case
    // must remain fully traceable/accelerated.
    nn::WindowAttention wa(/*dim=*/8, /*window_size=*/2, /*num_heads=*/2);
    check_op_grad_enabled_tracks_new_input(
        "nn::WindowAttention(no mask)",
        [&wa](const Variable& x) { return wa.forward(x, Tensor{}); },
        [](Device d) { return randn({2, 4, 8}, DType::Float32, d) + 1.0f; },
        [](Device d) { return randn({2, 4, 8}, DType::Float32, d) + 100.0f; });
}

// =========================================================================
// JIT-R058b regression coverage: QuantizedLinear::forward_quantized's CPU
// path (no OpId::QuantizedLinear kernel registered for CPU) computes the
// dequantize+matmul via a raw host pointer loop, zero dispatch() calls.
// Scoped to the per-tensor (non-per-channel, non-INT4) case, matching
// OpType::QuantizedLinearStatic's attr convention (scalar weight_scale/
// weight_zero_point).
// =========================================================================

TEST(JitTraceOps, QuantizedLinearStaticTracksNewInput) {
    namespace tq = ::tenzor::nn::quantization;
    const int64_t in_features = 8, out_features = 4;

    auto qconfig = tq::DefaultQConfigs::default_qconfig();
    auto weight_obs = qconfig.create_weight_observer();
    Tensor weights({out_features, in_features}, DType::Float32, Device::cpu());
    weights.fill_(0.1f);
    weight_obs->observe(weights);
    auto weight_params = weight_obs->calculate_qparams(
        tq::QuantDType::INT8, tq::QuantizationScheme::PerTensorSymmetric);

    tq::QuantizedLinear q_linear(in_features, out_features, weight_params);
    q_linear.set_weight(tq::quantize_tensor(weights, weight_params));

    // Statically calibrate the activation qparams (as a real static-quant
    // deployment would via a calibration pass) so input_scale/zero_point are
    // genuine per-instance constants. Without this, QuantizedLinear::
    // forward_impl falls back to quantize_per_tensor_symmetric(input) —
    // dynamic, per-call requantization computed by a raw-Tensor (non-
    // Variable) function that is invisible to the tracer regardless of this
    // fix; that gap is JIT-R066 (core (de)quantize math bypasses dispatch/
    // tracing), tracked and scheduled separately, not this finding's scope.
    auto act_obs = qconfig.create_activation_observer();
    Tensor calib_hi({1, in_features}, DType::Float32, Device::cpu());
    calib_hi.fill_(150.0f);
    Tensor calib_lo({1, in_features}, DType::Float32, Device::cpu());
    calib_lo.fill_(-150.0f);
    act_obs->observe(calib_hi);
    act_obs->observe(calib_lo);
    auto act_params = act_obs->calculate_qparams(
        tq::QuantDType::INT8, tq::QuantizationScheme::PerTensorSymmetric);
    q_linear.set_activation_qparams(act_params);

    auto fn = [&q_linear](const Variable& x) { return q_linear.forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({2, in_features}, DType::Float32, dev) + 1.0f;
            auto other = randn({2, in_features}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::quantization::QuantizedLinear did not compile/cache "
                   "a graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-2f, 1e-2f))
                << "nn::quantization::QuantizedLinear: replay on a NEW input "
                   "!= eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::quantization::QuantizedLinear threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// JIT-R058b/R066 companion: WITHOUT set_activation_qparams(), forward_impl
// falls back to quantize_per_tensor_symmetric(input) — dynamic, per-call
// requantization computed by raw-Tensor (non-Variable) code that is
// invisible to the tracer. Baking that as a graph node would silently reuse
// trace-time scale/zero_point (and even trace-time DATA) on every later
// replay call. Matches JIT-R083 (RoPE)/JIT-R090 (KVCache)/WindowAttention's
// established "fail loudly instead of producing wrong numerics" pattern:
// forward_impl refuses to trace at all (throws on the FIRST traced call)
// whenever activation_qparams_ is unset.
TEST(JitTraceOps, QuantizedLinearDynamicQuantRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    const int64_t in_features = 8, out_features = 4;

    auto qconfig = tq::DefaultQConfigs::default_qconfig();
    auto weight_obs = qconfig.create_weight_observer();
    Tensor weights({out_features, in_features}, DType::Float32, Device::cpu());
    weights.fill_(0.1f);
    weight_obs->observe(weights);
    auto weight_params = weight_obs->calculate_qparams(
        tq::QuantDType::INT8, tq::QuantizationScheme::PerTensorSymmetric);

    tq::QuantizedLinear q_linear(in_features, out_features, weight_params);
    q_linear.set_weight(tq::quantize_tensor(weights, weight_params));
    // Deliberately no set_activation_qparams() call — dynamic path.

    auto fn = [&q_linear](const Variable& x) { return q_linear.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({2, in_features}, DType::Float32, dev) + 1.0f;

        EXPECT_THROW(
            { (void)compiled(Variable(in_a, /*requires_grad=*/false)); },
            std::exception)
            << "nn::quantization::QuantizedLinear (uncalibrated activation) "
               "did not throw when JIT-traced on " << backend_name(dev)
            << " — a later call with a different input would silently "
               "replay trace-time-frozen quantization statistics/data";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "nn::quantization::QuantizedLinear cached a compiled graph "
               "on " << backend_name(dev) << " despite throwing during trace";

        // The guard must only fire while actually tracing: plain eager
        // calls must still work normally for any input.
        for (int i = 0; i < 2; ++i) {
            auto in_eager = randn({2, in_features}, DType::Float32, dev) + 1.0f;
            EXPECT_NO_THROW({ (void)fn(Variable(in_eager, false)); })
                << "nn::quantization::QuantizedLinear incorrectly threw on a "
                   "plain eager (non-traced) call on " << backend_name(dev);
        }
    }
}

// =========================================================================
// JIT-R066 regression coverage: unlike QuantizedLinear (JIT-R058b, above),
// QuantizedConv1d/2d/3d/ConvTranspose2d/Conv2dBnReLU/Embedding/EmbeddingBag's
// forward_quantized paths have NO dispatch()-visible lineage at all (100%
// raw host loops for every qparams configuration) — no partial dispatch
// exists to build a manual replay node on top of, unlike QuantizedLinear's
// GPU fast path. All refuse to trace unconditionally.
// =========================================================================

namespace {
auto make_dummy_qparams() -> ::tenzor::nn::quantization::QuantizationParams {
    namespace tq = ::tenzor::nn::quantization;
    Tensor scale({1}, DType::Float32, Device::cpu());
    Tensor zp({1}, DType::Int32, Device::cpu());
    scale.fill_(1.0f);
    zp.fill_(0);
    return tq::QuantizationParams(scale, zp, tq::QuantDType::INT8,
                                  tq::QuantizationScheme::PerTensorSymmetric);
}
}  // namespace

TEST(JitTraceOps, QuantizedConv2dRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    tq::QuantizedConv2d qconv(/*in_channels=*/2, /*out_channels=*/2,
                              /*kernel_size=*/3, /*stride=*/1, /*padding=*/1,
                              /*dilation=*/1, /*groups=*/1, make_dummy_qparams());
    auto fn = [&qconv](const Variable& x) { return qconv.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({1, 2, 4, 4}, DType::Float32, dev);
        EXPECT_THROW({ (void)compiled(Variable(in_a, false)); }, std::exception)
            << "QuantizedConv2d did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedConv2d cached a compiled graph on " << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "QuantizedConv2d incorrectly threw on a plain eager call on "
            << backend_name(dev);
    }
}

TEST(JitTraceOps, QuantizedConv1dRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    tq::QuantizedConv1d qconv(/*in_channels=*/2, /*out_channels=*/2,
                              /*kernel_size=*/3, /*stride=*/1, /*padding=*/1,
                              /*dilation=*/1, /*groups=*/1, make_dummy_qparams());
    auto fn = [&qconv](const Variable& x) { return qconv.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({1, 2, 8}, DType::Float32, dev);
        EXPECT_THROW({ (void)compiled(Variable(in_a, false)); }, std::exception)
            << "QuantizedConv1d did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedConv1d cached a compiled graph on " << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "QuantizedConv1d incorrectly threw on a plain eager call on "
            << backend_name(dev);
    }
}

TEST(JitTraceOps, QuantizedConv3dRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    tq::QuantizedConv3d qconv(/*in_channels=*/2, /*out_channels=*/2,
                              /*kernel_size=*/{3, 3, 3}, /*stride=*/{1, 1, 1},
                              /*padding=*/{1, 1, 1}, /*dilation=*/{1, 1, 1},
                              /*groups=*/1, make_dummy_qparams());
    auto fn = [&qconv](const Variable& x) { return qconv.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({1, 2, 4, 4, 4}, DType::Float32, dev);
        EXPECT_THROW({ (void)compiled(Variable(in_a, false)); }, std::exception)
            << "QuantizedConv3d did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedConv3d cached a compiled graph on " << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "QuantizedConv3d incorrectly threw on a plain eager call on "
            << backend_name(dev);
    }
}

TEST(JitTraceOps, QuantizedConvTranspose2dRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    tq::QuantizedConvTranspose2d qconv(/*in_channels=*/2, /*out_channels=*/2,
                                       /*kernel_size=*/3, /*stride=*/1,
                                       /*padding=*/1, /*output_padding=*/0,
                                       /*groups=*/1, make_dummy_qparams());
    auto fn = [&qconv](const Variable& x) { return qconv.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({1, 2, 4, 4}, DType::Float32, dev);
        EXPECT_THROW({ (void)compiled(Variable(in_a, false)); }, std::exception)
            << "QuantizedConvTranspose2d did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedConvTranspose2d cached a compiled graph on "
            << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "QuantizedConvTranspose2d incorrectly threw on a plain eager "
               "call on " << backend_name(dev);
    }
}

TEST(JitTraceOps, QuantizedConv2dBnReLURefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    Tensor bn_scale({2}, DType::Float32, Device::cpu());
    Tensor bn_bias({2}, DType::Float32, Device::cpu());
    bn_scale.fill_(1.0f);
    bn_bias.fill_(0.0f);
    tq::QuantizedConv2dBnReLU qconv(/*in_channels=*/2, /*out_channels=*/2,
                                    /*kernel_size=*/3, /*stride=*/1,
                                    /*padding=*/1, /*dilation=*/1,
                                    /*groups=*/1, make_dummy_qparams(),
                                    bn_scale, bn_bias);
    auto fn = [&qconv](const Variable& x) { return qconv.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = randn({1, 2, 4, 4}, DType::Float32, dev);
        EXPECT_THROW({ (void)compiled(Variable(in_a, false)); }, std::exception)
            << "QuantizedConv2dBnReLU did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedConv2dBnReLU cached a compiled graph on "
            << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(Variable(in_a, false)); })
            << "QuantizedConv2dBnReLU incorrectly threw on a plain eager "
               "call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R058a regression coverage: QuantizedLSTM/QuantizedGRU/QuantizedLSTMCell
// manage state via raw Tensor::slice()/squeeze()/unsqueeze()/permute() member
// calls — pure TensorImpl metadata tricks with zero dispatch() calls, hence
// invisible to the tracer (which for view ops relies on the Variable-level
// autograd::slice/squeeze/unsqueeze/permute wrappers explicitly calling
// jit_record_shape_op(), not generic dispatch interception). Since h0/c0
// (initial state), the per-timestep input slice, and the gate-split
// boundaries are all LIVE per-call values, freezing them at trace time is a
// silent wrong-numerics bug, not just staleness — matching R058a's own
// "silently freezing dequantized weights, initial-state slices, transposed
// weights, and gate-split boundaries as trace-time constants" description.
// =========================================================================

TEST(JitTraceOps, QuantizedLSTMTracksNewInput) {
    namespace tq = ::tenzor::nn::quantization;
    const int64_t input_size = 4, hidden_size = 3, seq_len = 5, batch = 2;

    nn::LSTM fp_lstm(input_size, hidden_size, /*num_layers=*/1, /*bias=*/true,
                      /*batch_first=*/true);
    auto qconfig = tq::DefaultQConfigs::default_qconfig();
    auto q_lstm = tq::QuantizedLSTM::from_float(fp_lstm, qconfig);

    auto fn = [q_lstm](const Variable& x) { return q_lstm->forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({batch, seq_len, input_size}, DType::Float32, dev) + 1.0f;
            auto other = randn({batch, seq_len, input_size}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::quantization::QuantizedLSTM did not compile/cache a "
                   "graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-2f, 1e-2f))
                << "nn::quantization::QuantizedLSTM: replay on a NEW input != "
                   "eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze trace-time state "
                   "slices/gate splits as constants instead of tracing them";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::quantization::QuantizedLSTM threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

TEST(JitTraceOps, QuantizedGRUTracksNewInput) {
    namespace tq = ::tenzor::nn::quantization;
    const int64_t input_size = 4, hidden_size = 3, seq_len = 5, batch = 2;

    nn::GRU fp_gru(input_size, hidden_size, /*num_layers=*/1, /*bias=*/true,
                    /*batch_first=*/true);
    auto qconfig = tq::DefaultQConfigs::default_qconfig();
    auto q_gru = tq::QuantizedGRU::from_float(fp_gru, qconfig);

    auto fn = [q_gru](const Variable& x) { return q_gru->forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({batch, seq_len, input_size}, DType::Float32, dev) + 1.0f;
            auto other = randn({batch, seq_len, input_size}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::quantization::QuantizedGRU did not compile/cache a "
                   "graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-2f, 1e-2f))
                << "nn::quantization::QuantizedGRU: replay on a NEW input != "
                   "eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze trace-time state "
                   "slices/gate splits as constants instead of tracing them";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::quantization::QuantizedGRU threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

TEST(JitTraceOps, QuantizedLSTMCellTracksNewInput) {
    namespace tq = ::tenzor::nn::quantization;
    const int64_t input_size = 4, hidden_size = 3, batch = 2;

    nn::LSTMCell fp_cell(input_size, hidden_size, /*bias=*/true);
    auto qconfig = tq::DefaultQConfigs::default_qconfig();
    auto q_cell = tq::QuantizedLSTMCell::from_float(fp_cell, qconfig);

    auto fn = [q_cell](const Variable& x) { return q_cell->forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({batch, input_size}, DType::Float32, dev) + 1.0f;
            auto other = randn({batch, input_size}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::quantization::QuantizedLSTMCell did not compile/cache "
                   "a graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-2f, 1e-2f))
                << "nn::quantization::QuantizedLSTMCell: replay on a NEW "
                   "input != eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze trace-time gate "
                   "splits as constants instead of tracing them";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::quantization::QuantizedLSTMCell threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R056 regression coverage: EmbeddingBag's CPU aggregation path (both
// the grad-enabled EmbeddingBagBackward::forward() CPU fallback and the
// no-grad EmbeddingBag::aggregate_embeddings() CPU path) computed the
// per-bag reduction via a raw host loop, zero dispatch() calls — unlike
// Embedding's own already-patched CPU path (jit_record_embedding()).
// Like BatchNorm2d (JIT-R059), the no-grad CPU path is gated on
// is_grad_enabled() as well as the tensor's own requires_grad, so
// NoGradGuard (matching the real eval()-under-no_grad usage pattern) is
// needed to actually reach it.
// =========================================================================

namespace {
auto make_index_tensor(std::initializer_list<int64_t> values, Device dev) -> Tensor {
    auto t_cpu = zeros({static_cast<int64_t>(values.size())}, DType::Int64, Device::cpu());
    auto* ptr = t_cpu.data<int64_t>();
    size_t i = 0;
    for (int64_t v : values) ptr[i++] = v;
    return (dev.type == Device::Type::CPU) ? t_cpu : t_cpu.to(dev);
}
}  // namespace

TEST(JitTraceOps, QuantizedEmbeddingRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    tq::QuantizedEmbedding qembed(/*num_embeddings=*/10, /*embedding_dim=*/4,
                                  make_dummy_qparams());
    auto fn = [&qembed](const Variable& x) { return qembed.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = Variable(make_index_tensor({0, 1, 2, 3}, dev), false);
        EXPECT_THROW({ (void)compiled(in_a); }, std::exception)
            << "QuantizedEmbedding did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedEmbedding cached a compiled graph on "
            << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(in_a); })
            << "QuantizedEmbedding incorrectly threw on a plain eager call "
               "on " << backend_name(dev);
    }
}

TEST(JitTraceOps, QuantizedEmbeddingBagRefusesToTrace) {
    namespace tq = ::tenzor::nn::quantization;
    tq::QuantizedEmbeddingBag qembed(/*num_embeddings=*/10, /*embedding_dim=*/4,
                                     make_dummy_qparams());
    auto fn = [&qembed](const Variable& x) { return qembed.forward(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto in_a = Variable(make_index_tensor({0, 1, 2, 3}, dev), false);
        EXPECT_THROW({ (void)compiled(in_a); }, std::exception)
            << "QuantizedEmbeddingBag did not throw when JIT-traced on "
            << backend_name(dev);
        EXPECT_EQ(compiled.num_cached(), 0u)
            << "QuantizedEmbeddingBag cached a compiled graph on "
            << backend_name(dev);
        EXPECT_NO_THROW({ (void)fn(in_a); })
            << "QuantizedEmbeddingBag incorrectly threw on a plain eager "
               "call on " << backend_name(dev);
    }
}

TEST(JitTraceOps, EmbeddingBagTracksNewIndices) {
    NoGradGuard no_grad;
    nn::EmbeddingBag bag(/*num_embeddings=*/10, /*embedding_dim=*/4,
                         /*max_norm=*/0.0, /*norm_type=*/2.0,
                         /*scale_grad_by_freq=*/false, /*mode=*/"mean");
    for (const auto& dev : get_available_backends()) {
        try {
            auto offsets = Variable(make_index_tensor({0, 2}, dev), false);
            auto fn = [&bag, &offsets](const Variable& indices) {
                return bag.forward(indices, offsets) * 2.0;
            };
            auto compiled = jit::compile(fn);
            auto dummy = Variable(make_index_tensor({0, 1, 2, 3}, dev), false);
            auto other = Variable(make_index_tensor({9, 8, 7, 6}, dev), false);

            (void)compiled(dummy);
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::EmbeddingBag did not compile/cache a graph on "
                << backend_name(dev) << " — the op likely graph-broke and "
                   "silently fell back to eager, so replay correctness was "
                   "never exercised";

            auto replayed = compiled(other).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(other).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "nn::EmbeddingBag: replay on NEW indices != eager on those "
                   "indices on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::EmbeddingBag threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R060 regression coverage: MaxPool2d's ceil_mode was stored as an
// Int64-tagged AttrKey::CeilMode that the generic attr-copy block never
// captured, so a traced ceil_mode=true MaxPool2d silently replayed with the
// floor output-size formula (wrong SHAPE, not a frozen-constant bug).
// ceil_mode=true is only reachable on Vulkan/MPS (CPU rejects it outright,
// matching AvgPool2d's parity design) — this build has neither, so the
// actual ceil-vs-floor SHAPE difference can't be exercised here. This test
// instead verifies: (1) the plumbing added to reach it — F::max_pool2d's
// new ceil_mode parameter — correctly reaches nn::MaxPool2d's own
// CPU-rejection gate (proving the parameter threads through, not silently
// dropped by the new default arg), and (2) the ceil_mode=false path (the
// only one exercisable on CPU) still traces/replays correctly with the new
// AttrKey::CeilMode plumbing present (no regression).
// =========================================================================

TEST(JitTraceOps, MaxPool2dCeilModeParameterReachesCpuRejectionGate) {
    auto x = randn({1, 2, 5, 5}, DType::Float32, Device::cpu());
    EXPECT_THROW(
        { (void)nn::functional::max_pool2d(Variable(x, false), {2, 2}, {2, 2}, {0, 0}, /*ceil_mode=*/true); },
        std::exception)
        << "F::max_pool2d(ceil_mode=true) did not reach nn::MaxPool2d's "
           "CPU-rejection gate — the new ceil_mode parameter may be silently "
           "dropped somewhere in the call chain";
}

TEST(JitTraceOps, MaxPool2dFloorModeStillTracksNewInput) {
    nn::MaxPool2d mp(/*kernel_size=*/2, /*stride=*/2, /*padding=*/0,
                     /*ceil_mode=*/false, /*return_indices=*/false);
    check_op_grad_enabled_tracks_new_input(
        "nn::MaxPool2d(ceil_mode=false)",
        [&mp](const Variable& x) { return mp.forward(x) * 2.0; },
        [](Device d) { return randn({1, 2, 5, 5}, DType::Float32, d) + 1.0f; },
        [](Device d) { return randn({1, 2, 5, 5}, DType::Float32, d) + 100.0f; });
}

// =========================================================================
// JIT-R069 regression coverage: MaxPool2d/AvgPool2d::forward_impl dispatched
// their forward kernel via dispatch_to_device(), which routes through
// DispatchInterceptorStack — the SAME entry point the free dispatch<OpId>
// template uses. Since OpId::MaxPool2dForward/AvgPool2dForward ARE mapped
// in opid_to_optype, the generic interceptor was auto-recording a node in
// addition to the manual jit_record-style block right after (the only one
// carrying the per-axis W / ceil_mode / count_include_pad attrs) — a
// genuine double-record, though harmless in practice because the manual
// block's fresh register_new_tensor() id shadows the auto-recorded node's
// output, and end_trace()'s reachable-from-outputs graph construction
// silently drops the now-unreachable orphan. Fixed by calling the backend
// table directly (bypassing the interceptor entirely), matching
// JIT-R098/R103's established pattern. This test asserts the node count
// directly via dump_graph() rather than only checking numeric correctness
// (which the double-record never affected), so a regression back to the
// double-dispatch form would be caught even though it wouldn't fail any
// value-based test.
// =========================================================================

TEST(JitTraceOps, MaxPool2dTracesExactlyOneNode) {
    nn::MaxPool2d mp(/*kernel_size=*/2, /*stride=*/2, /*padding=*/0,
                     /*ceil_mode=*/false, /*return_indices=*/false);
    auto fn = [&mp](const Variable& x) { return mp.forward(x); };
    auto compiled = jit::compile(fn);
    auto x = randn({1, 2, 4, 4}, DType::Float32, Device::cpu());
    std::string g = compiled.dump_graph(Variable(x, false));
    size_t count = 0, pos = 0;
    while ((pos = g.find("[MaxPool2d]", pos)) != std::string::npos) { count++; pos++; }
    EXPECT_EQ(count, 1u)
        << "MaxPool2d traced " << count << " nodes instead of 1 — "
           "dispatch_to_device is double-recording via the generic "
           "interceptor again:\n" << g;
}

TEST(JitTraceOps, AvgPool2dTracesExactlyOneNode) {
    nn::AvgPool2d ap(/*kernel_size=*/2, /*stride=*/2, /*padding=*/0);
    auto fn = [&ap](const Variable& x) { return ap.forward(x); };
    auto compiled = jit::compile(fn);
    auto x = randn({1, 2, 4, 4}, DType::Float32, Device::cpu());
    std::string g = compiled.dump_graph(Variable(x, false));
    size_t count = 0, pos = 0;
    while ((pos = g.find("[AvgPool2d]", pos)) != std::string::npos) { count++; pos++; }
    EXPECT_EQ(count, 1u)
        << "AvgPool2d traced " << count << " nodes instead of 1 — "
           "dispatch_to_device is double-recording via the generic "
           "interceptor again:\n" << g;
}

// =========================================================================
// JIT-R055 regression coverage: SparseLinear's forward computed
// sparse::spmm(sparse, dense) directly at the Tensor level (comment:
// "avoids constructing the sparse op dispatch graph"), bypassing dispatch()
// entirely. A traced call would freeze the whole output as a trace-time
// constant, silently ignoring subsequent optimizer-updated trainable
// values and new inputs. Uses inference mode: SparseLinearBackward's
// grad-mode replay is unconditionally unsupported (OpType::SparseSpMM
// throws in grad_mode, matching SparseMatMul's established precedent), so
// check_op_grad_enabled_tracks_new_input would be vacuous here too (see
// JIT-R052/R059's notes on this class of test pitfall).
// =========================================================================

TEST(JitTraceOps, SparseLinearTracksNewInput) {
    nn::SparseLinear sl(4, 3, /*density=*/0.5, /*bias=*/false);
    auto fn = [&sl](const Variable& x) { return sl.forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({2, 4}, DType::Float32, dev) + 1.0f;
            auto other = randn({2, 4}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::SparseLinear did not compile/cache a graph on "
                << backend_name(dev) << " — the op likely graph-broke and "
                   "silently fell back to eager, so replay correctness was "
                   "never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "nn::SparseLinear: replay on a NEW input != eager on that "
                   "input on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::SparseLinear threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R098 regression coverage: autograd::spmm/spmv/sparse_add called
// sparse::spmm()/spmv()/add() directly at the Tensor level, bypassing
// dispatch() entirely — the same mechanism as JIT-R055 (SparseLinear),
// now fixed at these shared entry points too. A traced call would freeze
// the output as a trace-time constant. Inference mode, same reason as
// SparseLinearTracksNewInput above: OpType::SparseSpMM/SparseSpMV/
// SparseAdd all throw in grad_mode (unconditionally unsupported replay).
// =========================================================================

TEST(JitTraceOps, SparseSpmmSpmvAddTrackNewInput) {
    // Fixed 3x3 CSR pattern [[2,0,0],[1,2,0],[1,1,2]], shared by spmm/spmv/
    // sparse_add (none of these require a triangular pattern).
    auto crow = Tensor({int64_t(4)}, DType::Int64, Device::cpu());
    auto col = Tensor({int64_t(6)}, DType::Int64, Device::cpu());
    auto values = Tensor({int64_t(6)}, DType::Float32, Device::cpu());
    { auto* cp = crow.data<int64_t>(); cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 6; }
    { auto* colp = col.data<int64_t>();
      colp[0] = 0; colp[1] = 0; colp[2] = 1; colp[3] = 0; colp[4] = 1; colp[5] = 2; }
    { auto* vp = values.data<float>();
      vp[0] = 2; vp[1] = 1; vp[2] = 2; vp[3] = 1; vp[4] = 1; vp[5] = 2; }
    auto sparse = SparseTensor::sparse_csr(crow, col, values, {3, 3});

    {
        auto fn = [&sparse](const Variable& dense) { return spmm(sparse, dense); };
        auto compiled = jit::compile(fn);
        auto dummy = randn({3, 2}, DType::Float32, Device::cpu()) + 1.0f;
        auto other = randn({3, 2}, DType::Float32, Device::cpu()) + 100.0f;
        (void)compiled(Variable(dummy, /*requires_grad=*/false));
        EXPECT_GE(compiled.num_cached(), 1u)
            << "spmm did not compile/cache a graph — likely graph-broke and "
               "silently fell back to eager";
        auto replayed = compiled(Variable(other, false)).tensor();
        auto eager_on_other = fn(Variable(other, false)).tensor();
        EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
            << "spmm: replay on a NEW input != eager on that input — the "
               "compiled graph likely froze the trace-time value";
    }
    {
        auto fn = [&sparse](const Variable& vec) { return spmv(sparse, vec); };
        auto compiled = jit::compile(fn);
        auto dummy = randn({3}, DType::Float32, Device::cpu()) + 1.0f;
        auto other = randn({3}, DType::Float32, Device::cpu()) + 100.0f;
        (void)compiled(Variable(dummy, false));
        EXPECT_GE(compiled.num_cached(), 1u)
            << "spmv did not compile/cache a graph — likely graph-broke and "
               "silently fell back to eager";
        auto replayed = compiled(Variable(other, false)).tensor();
        auto eager_on_other = fn(Variable(other, false)).tensor();
        EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
            << "spmv: replay on a NEW input != eager on that input — the "
               "compiled graph likely froze the trace-time value";
    }
    {
        auto fn = [&sparse](const Variable& dense) { return sparse_add(sparse, dense); };
        auto compiled = jit::compile(fn);
        auto dummy = randn({3, 3}, DType::Float32, Device::cpu()) + 1.0f;
        auto other = randn({3, 3}, DType::Float32, Device::cpu()) + 100.0f;
        (void)compiled(Variable(dummy, false));
        EXPECT_GE(compiled.num_cached(), 1u)
            << "sparse_add did not compile/cache a graph — likely graph-broke "
               "and silently fell back to eager";
        auto replayed = compiled(Variable(other, false)).tensor();
        auto eager_on_other = fn(Variable(other, false)).tensor();
        EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
            << "sparse_add: replay on a NEW input != eager on that input — "
               "the compiled graph likely froze the trace-time value";
    }
}

// JIT-R098: sparse_triangular_solve had the identical direct-sparse::-call
// bypass. Separate test because it needs a genuinely triangular pattern.
TEST(JitTraceOps, SparseTriangularSolveTracksNewInput) {
    // Lower-triangular 3x3 CSR: [[2,0,0],[1,2,0],[1,1,2]].
    auto crow = Tensor({int64_t(4)}, DType::Int64, Device::cpu());
    auto col = Tensor({int64_t(6)}, DType::Int64, Device::cpu());
    auto values = Tensor({int64_t(6)}, DType::Float32, Device::cpu());
    { auto* cp = crow.data<int64_t>(); cp[0] = 0; cp[1] = 1; cp[2] = 3; cp[3] = 6; }
    { auto* colp = col.data<int64_t>();
      colp[0] = 0; colp[1] = 0; colp[2] = 1; colp[3] = 0; colp[4] = 1; colp[5] = 2; }
    { auto* vp = values.data<float>();
      vp[0] = 2; vp[1] = 1; vp[2] = 2; vp[3] = 1; vp[4] = 1; vp[5] = 2; }
    auto L = SparseTensor::sparse_csr(crow, col, values, {3, 3});

    auto fn = [&L](const Variable& b) {
        return sparse_triangular_solve(L, b, /*upper=*/false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = randn({3}, DType::Float32, Device::cpu()) + 1.0f;
    auto other = randn({3}, DType::Float32, Device::cpu()) + 100.0f;
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "sparse_triangular_solve did not compile/cache a graph — likely "
           "graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-3f, 1e-3f))
        << "sparse_triangular_solve: replay on a NEW input != eager on that "
           "input — the compiled graph likely froze the trace-time value";
}

// =========================================================================
// JIT-R100 regression coverage: SparseTensor's structural-conversion
// methods (from_dense/to_dense/coalesce/transpose/to_csr/to_csc/to_bsr,
// src/sparse/sparse_tensor.cpp) were tracer-invisible when called directly
// by user code — a traced function using them recorded no node for the
// conversion, so a downstream consumer (e.g. another sparse op reading
// the result's component tensors) resolved to a frozen trace-time constant
// instead of a real producer. Fixed by manually recording each conversion
// (mirroring JIT-R098's sparse_add pattern) plus mapping the previously-
// unmapped CumSum/Sort/Nonzero/Bincount ops these methods dispatch
// internally — left unmapped, any of them would graph-break the WHOLE
// trace even with the outer node's own manual recording in place (the
// same class of problem JIT-R098 fixed for ScatterAdd/RepeatInterleave).
// See the OpType::SparseFromDense..SparseToBsr doc comments in tracer.hpp
// for the exact recording convention.
//
// Each test traces a function chaining FromDense (or a hand-built
// SparseTensor) -> the method under test -> ToDense, then replays on a
// DIFFERENT input and checks the replayed result tracks that new input
// (rather than the trace-time value) — this directly validates the
// downstream-consumer linkage the bug broke, not just "doesn't crash".
// =========================================================================

namespace {
Tensor jit_r100_test_dense(float offset) {
    auto t = zeros({4, 4}, DType::Float32, Device::cpu());
    auto* p = t.data<float>();
    p[0] = 1.0f + offset; p[5] = 2.0f + offset; p[10] = 3.0f + offset; p[15] = 4.0f + offset;
    p[3] = 0.5f + offset; p[6] = 1.5f + offset;
    return t;
}
}  // namespace

TEST(JitTraceOps, SparseFromDenseToDenseTracksNewInput) {
    auto fn = [](const Variable& dense) -> Variable {
        auto sp = SparseTensor::from_dense(dense.tensor(), SparseLayout::CSR);
        return Variable(sp.to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = jit_r100_test_dense(0.0f);
    auto other = jit_r100_test_dense(100.0f);
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "SparseTensor::from_dense/to_dense did not compile/cache a graph "
           "— likely graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
        << "SparseTensor::from_dense/to_dense: replay on a NEW input != "
           "eager on that input — the compiled graph likely froze the "
           "trace-time value";
}

TEST(JitTraceOps, SparseToCsrTracksNewInput) {
    auto fn = [](const Variable& dense) -> Variable {
        auto sp = SparseTensor::from_dense(dense.tensor(), SparseLayout::COO);
        auto csr = sp.to_csr();
        return Variable(csr.to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = jit_r100_test_dense(0.0f);
    auto other = jit_r100_test_dense(100.0f);
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "SparseTensor::to_csr did not compile/cache a graph — likely "
           "graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
        << "SparseTensor::to_csr: replay on a NEW input != eager on that "
           "input — the compiled graph likely froze the trace-time value";
}

TEST(JitTraceOps, SparseToCscTracksNewInput) {
    auto fn = [](const Variable& dense) -> Variable {
        auto sp = SparseTensor::from_dense(dense.tensor(), SparseLayout::COO);
        auto csc = sp.to_csc();
        return Variable(csc.to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = jit_r100_test_dense(0.0f);
    auto other = jit_r100_test_dense(100.0f);
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "SparseTensor::to_csc did not compile/cache a graph — likely "
           "graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
        << "SparseTensor::to_csc: replay on a NEW input != eager on that "
           "input — the compiled graph likely froze the trace-time value";
}

TEST(JitTraceOps, SparseToBsrTracksNewInput) {
    auto fn = [](const Variable& dense) -> Variable {
        auto sp = SparseTensor::from_dense(dense.tensor(), SparseLayout::CSR);
        auto bsr = sp.to_bsr({2, 2});
        return Variable(bsr.to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = jit_r100_test_dense(0.0f);
    auto other = jit_r100_test_dense(100.0f);
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "SparseTensor::to_bsr did not compile/cache a graph — likely "
           "graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
        << "SparseTensor::to_bsr: replay on a NEW input != eager on that "
           "input — the compiled graph likely froze the trace-time value";
}

TEST(JitTraceOps, SparseTransposeTracksNewInput) {
    // 4x4 input isn't square-symmetric in its nonzero pattern, so a real
    // transpose (not an accidental identity) is exercised. CSR input ->
    // transpose() returns CSR too (per SparseTensor::transpose's CSR
    // branch), so to_dense() directly reflects the transposed 4x4 shape.
    auto fn = [](const Variable& dense) -> Variable {
        auto sp = SparseTensor::from_dense(dense.tensor(), SparseLayout::CSR);
        auto t = sp.transpose();
        return Variable(t.to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = jit_r100_test_dense(0.0f);
    auto other = jit_r100_test_dense(100.0f);
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "SparseTensor::transpose did not compile/cache a graph — likely "
           "graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
        << "SparseTensor::transpose: replay on a NEW input != eager on that "
           "input — the compiled graph likely froze the trace-time value";
}

TEST(JitTraceOps, SparseCoalesceTracksNewInput) {
    // Coalesce needs genuine duplicate COO coordinates to be a meaningful
    // test: two duplicate pairs at (0,0)/(1,1), one unique (2,2). Built
    // directly (not via from_dense, whose nonzero-based construction always
    // produces unique coords) so the values (traced input) get summed
    // across duplicates by the real coalesce() computation.
    auto indices = Tensor({int64_t(2), int64_t(5)}, DType::Int64, Device::cpu());
    {
        auto* ip = indices.data<int64_t>();
        ip[0] = 0; ip[1] = 0; ip[2] = 1; ip[3] = 1; ip[4] = 2;  // row coord
        ip[5] = 0; ip[6] = 0; ip[7] = 1; ip[8] = 1; ip[9] = 2;  // col coord
    }
    auto fn = [&indices](const Variable& values) -> Variable {
        auto sp = SparseTensor::sparse_coo(indices, values.tensor(), {3, 3},
                                           /*validate=*/false);
        auto coalesced = sp.coalesce();
        return Variable(coalesced.to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = full({5}, 1.0, DType::Float32, Device::cpu());
    auto other = full({5}, 7.0, DType::Float32, Device::cpu());
    (void)compiled(Variable(dummy, false));
    EXPECT_GE(compiled.num_cached(), 1u)
        << "SparseTensor::coalesce did not compile/cache a graph — likely "
           "graph-broke and silently fell back to eager";
    auto replayed = compiled(Variable(other, false)).tensor();
    auto eager_on_other = fn(Variable(other, false)).tensor();
    EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
        << "SparseTensor::coalesce: replay on a NEW input != eager on that "
           "input — the compiled graph likely froze the trace-time value";
    // Sanity: the eager result must actually reflect the coalesce sum
    // (2*value at (0,0) and (1,1), 1*value at (2,2)) — otherwise this test
    // would pass vacuously even if coalesce() silently did nothing.
    auto expected = zeros({3, 3}, DType::Float32, Device::cpu());
    { auto* e = expected.data<float>(); e[0] = 14.0f; e[4] = 14.0f; e[8] = 7.0f; }
    EXPECT_TRUE(tensors_close(eager_on_other, expected, 1e-4f, 1e-4f))
        << "sanity check: eager coalesce did not sum duplicate entries as expected";
}

// Structural check (mirrors JIT-R069's node-count regression pattern):
// asserts exactly one [SparseCoalesce] node appears in the compiled graph
// — coalesce() is the highest-risk case (it internally dispatches Sort,
// CumSum, Nonzero AND ScatterAdd), so this directly confirms none of those
// nested ops leak into the final graph as extra/unreachable nodes and that
// the manual recording didn't double-record the same way JIT-R069 did.
TEST(JitTraceOps, SparseCoalesceTracesExactlyOneNode) {
    auto indices = Tensor({int64_t(2), int64_t(3)}, DType::Int64, Device::cpu());
    { auto* ip = indices.data<int64_t>(); ip[0]=0; ip[1]=0; ip[2]=1; ip[3]=0; ip[4]=0; ip[5]=1; }
    auto fn = [&indices](const Variable& values) {
        auto sp = SparseTensor::sparse_coo(indices, values.tensor(), {2, 2},
                                           /*validate=*/false);
        return Variable(sp.coalesce().to_dense(), false);
    };
    auto compiled = jit::compile(fn);
    auto x = full({3}, 1.0, DType::Float32, Device::cpu());
    std::string g = compiled.dump_graph(Variable(x, false));
    size_t count = 0, pos = 0;
    while ((pos = g.find("[SparseCoalesce]", pos)) != std::string::npos) { count++; pos++; }
    EXPECT_EQ(count, 1u)
        << "SparseCoalesce traced " << count << " nodes instead of 1:\n" << g;
}

// =========================================================================
// JIT-R059 regression coverage: BatchNorm2d's CPU eval-mode fast inference
// path (a hand-rolled SIMD pointer loop) never dispatches, unlike the
// STANDARD PATH's dispatch(OpId::BatchNorm2dForwardAffine) — the common
// model.eval()-under-no_grad inference case would freeze the entire output
// as a trace-time constant.
// =========================================================================

TEST(JitTraceOps, BatchNorm2dEvalModeTracksNewInput) {
    // The buggy fast path is gated on !needs_grad — i.e. it only fires when
    // BOTH the trace's grad-mode AND the input's requires_grad are false,
    // the standard model.eval()-under-no_grad inference pattern. Unlike the
    // grad-enabled tracer-bypass bugs elsewhere in this file,
    // check_op_grad_enabled_tracks_new_input can't exercise this: BatchNorm2d
    // grad-mode REPLAY is unconditionally unsupported by design ("not wired
    // for differentiable replay"), so a requires_grad=true trace always
    // falls back to genuine eager autograd on every call regardless of this
    // fix, making that helper's checks vacuously true. Drive the INFERENCE
    // (non-grad) compile path directly instead, matching jit.compile()'s
    // actual behavior for a pure eval()/no_grad() inference call.
    nn::BatchNorm2d bn(4);
    bn.eval();
    // The fast path's `needs_grad` gate ALSO looks at the affine weight/bias
    // parameters' own requires_grad (true by default for a trainable
    // nn::Module) — a merely requires_grad=false INPUT is not enough to
    // reach the buggy branch. Disable grad mode globally (NoGradGuard),
    // matching the real model.eval()-under-torch.no_grad() inference
    // pattern the finding describes, which is what actually makes
    // needs_grad false regardless of the parameters' own flags.
    NoGradGuard no_grad;
    auto fn = [&bn](const Variable& x) { return bn.forward(x) * 2.0; };
    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto dummy = randn({2, 4, 3, 3}, DType::Float32, dev) + 1.0f;
            auto other = randn({2, 4, 3, 3}, DType::Float32, dev) + 100.0f;

            (void)compiled(Variable(dummy, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::BatchNorm2d(eval mode) did not compile/cache a graph "
                   "on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            auto replayed = compiled(Variable(other, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other, false)).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "nn::BatchNorm2d(eval mode): replay on a NEW input != "
                   "eager on that input on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::BatchNorm2d(eval mode) threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R005 regression coverage: without with_buffers(), a module's
// non-trainable buffers (BatchNorm's running_mean_/running_var_) are frozen
// as trace-time constants — a later eager mutation (the EMA update from a
// training loop run outside the compiled graph, since grad-mode replay of
// a training-mode norm layer already throws and correctly falls back to
// eager) is silently invisible to every subsequent replay of the SAME
// compiled graph. with_buffers() registers them as live leaves instead,
// rebound to their current value on every call.
// =========================================================================

TEST(JitTraceOps, BatchNorm2dRunningStatsLiveViaWithBuffers) {
    nn::BatchNorm2d bn(4);
    bn.eval();
    NoGradGuard no_grad;
    auto fn = [&bn](const Variable& x) { return bn.forward(x); };
    for (const auto& dev : get_available_backends()) {
        try {
            bn.to(dev);
            // Direct-init from the by-value compile() result (guaranteed
            // copy elision), then set_buffers() as a separate statement —
            // CompiledFunction is non-copyable (mutex/atomic members), so
            // chaining .with_buffers() (which returns CompiledFunction&)
            // straight into an `auto compiled = ...` would need a copy ctor
            // that doesn't exist.
            auto compiled = jit::compile(fn);
            compiled.set_buffers(bn.buffers());
            auto x = randn({2, 4, 3, 3}, DType::Float32, dev) + 1.0f;

            auto before = compiled(Variable(x, /*requires_grad=*/false))
                              .tensor().to(Device::cpu());
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::BatchNorm2d(with_buffers) did not compile/cache a "
                   "graph on " << backend_name(dev) << " — the op likely "
                   "graph-broke and silently fell back to eager, so replay "
                   "correctness was never exercised";

            // Simulate an eager EMA update happening OUTSIDE the compiled
            // graph (e.g. a training step) by directly mutating the running
            // stats, the same way BatchNorm's own EMA update does (`param->
            // tensor() = ...`, the established optimizer-update pattern).
            auto rm = bn.get_buffer("running_mean");
            ASSERT_TRUE(rm != nullptr)
                << "nn::BatchNorm2d has no 'running_mean' buffer on "
                << backend_name(dev);
            rm->tensor() = rm->tensor() + 5.0;

            auto after = compiled(Variable(x, /*requires_grad=*/false))
                             .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_after = fn(Variable(x, false)).tensor().to(Device::cpu());

            EXPECT_GT(max_abs_diff(before, after), 1e-3f)
                << "nn::BatchNorm2d replay did not react to a running_mean_ "
                   "mutation on " << backend_name(dev)
                << " — running stats are still frozen as trace-time "
                   "constants despite with_buffers()";
            EXPECT_TRUE(tensors_close(eager_after, after, 1e-4f, 1e-4f))
                << "nn::BatchNorm2d replay after mutation != eager after "
                   "mutation on " << backend_name(dev);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::BatchNorm2d(with_buffers) threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R091 regression coverage: CrossEntropyLoss/NLLLoss's segmentation-
// style (>2D integer class-index target) path builds a one-hot tensor via
// dispatch(OpId::OneHot) (correctly traced) but then moves the class axis
// with a raw Tensor::permute() call — pure TensorImpl metadata, zero
// dispatch, invisible to the tracer's view-op recording. Since one_hot is
// derived from the real per-call `target` argument, an untraced permute
// disconnects it from that lineage, freezing the trace-dummy's segmentation
// labels as a constant on replay.
// =========================================================================

TEST(JitTraceOps, CrossEntropyLossSegmentationTracksNewTarget) {
    const int64_t N = 2, C = 3, H = 2, W = 2;
    nn::CrossEntropyLoss loss(nn::Reduction::Mean);
    Tensor logits;  // set per-backend below; captured by reference

    // `target` (integer class-index labels) is the traced Variable argument;
    // logits is fixed across dummy/other since R091 is specifically about
    // whether the one-hot TARGET data survives replay on a new call.
    auto fn = [&loss, &logits](const Variable& target_var) {
        return loss.forward(Variable(logits, false), target_var.tensor());
    };

    for (const auto& dev : get_available_backends()) {
        try {
            logits = randn({N, C, H, W}, DType::Float32, dev);
            auto compiled = jit::compile(fn);
            auto dummy_target = randint(0, C, {N, H, W}, DType::Int64, dev);
            auto other_target = randint(0, C, {N, H, W}, DType::Int64, dev);

            (void)compiled(Variable(dummy_target, /*requires_grad=*/false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::CrossEntropyLoss(segmentation target) did not "
                   "compile/cache a graph on " << backend_name(dev)
                << " — the op likely graph-broke and silently fell back to "
                   "eager, so replay correctness was never exercised";

            auto replayed = compiled(Variable(other_target, /*requires_grad=*/false))
                                .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(Variable(other_target, false))
                                       .tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "nn::CrossEntropyLoss(segmentation target): replay on a "
                   "NEW target != eager on that target on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time "
                   "one-hot target as a constant instead of tracing it";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::CrossEntropyLoss(segmentation target) "
                             "threw on " << backend_name(dev) << ": "
                          << e.what();
        }
    }
}

// =========================================================================
// JIT-R092 regression coverage: CTCLoss's CPU path used to compute the
// entire forward-backward DP loss/gradient via raw pointer loops (zero
// dispatch() calls, invisible to the tracer — silently froze the trace-
// dummy's loss on replay). Fixed by routing CPU through the same
// dispatch<OpId::CTCLossForward> call the GPU path always used. OpId::
// CTCLossForward itself still has no OpType mapping (a custom backward
// captures a precomputed gradient tensor that the generic autograd-replay
// graph can't reconstruct from primitive ops), so tracing CTCLoss now
// safely graph-breaks (matches its own GPU path's pre-existing landmine
// characterization) instead of silently freezing — confirmed below via
// num_cached()==0 and correctness across two DIFFERENT calls (each
// eagerly re-dispatches on cache-miss/graph-break, so results must track
// the real input every time, not just the first).
// =========================================================================

// findings.txt JIT-R134: OpId::CTCLossForward is now mapped in
// tracing_interceptor.cpp's opid_to_optype (previously entirely unmapped,
// hard graph-breaking any traced CTCLoss forward on every backend).
// execute_node's OpType::CTCLossForward case re-dispatches the identical op
// for inference (non-grad) replay — exactly this test's scenario, since both
// logits variables are constructed with requires_grad=false — and throws to
// force an eager fallback for differentiable (grad_mode) replay instead of
// risking an incorrect custom-backward replay. So a no-grad CTCLoss forward
// now DOES compile/cache correctly; this test was renamed and its assertion
// flipped accordingly. The correctness checks (replay on a SECOND, DIFFERENT
// input must match eager on that input, not the first call's frozen result)
// are the important regression coverage and are unchanged.
TEST(JitTraceOps, CTCLossTracesCachesAndStaysCorrect) {
    const int64_t T = 5, N = 2, C = 4, S = 2;
    nn::CTCLoss loss("mean", /*blank=*/0, /*zero_infinity=*/false);

    for (const auto& dev : get_available_backends()) {
        try {
            Tensor targets = randint(1, C, {N, S}, DType::Int64, dev);
            Tensor input_lengths = full({N}, static_cast<double>(T), DType::Int64, dev);
            Tensor target_lengths = full({N}, static_cast<double>(S), DType::Int64, dev);

            auto fn = [&](const Variable& logits) {
                auto log_probs = tenzor::nn::log_softmax(logits, 2);
                return loss.forward(log_probs, targets, input_lengths, target_lengths);
            };

            auto compiled = jit::compile(fn);
            auto logits_a = randn({T, N, C}, DType::Float32, dev);
            auto logits_b = randn({T, N, C}, DType::Float32, dev) + 3.0f;

            auto replayed_a = compiled(Variable(logits_a, /*requires_grad=*/false))
                                   .tensor().to(Device::cpu());
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::CTCLoss (no-grad forward) did not compile/cache a graph "
                   "on " << backend_name(dev) << " — OpId::CTCLossForward "
                   "graph-broke, so the JIT-R134 tracer-visibility fix "
                   "regressed on this backend";

            auto eager_a = fn(Variable(logits_a, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_a, replayed_a, 1e-3f, 1e-3f))
                << "nn::CTCLoss: first compiled() call != eager on " << backend_name(dev);

            auto replayed_b = compiled(Variable(logits_b, /*requires_grad=*/false))
                                   .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_b = fn(Variable(logits_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_b, replayed_b, 1e-3f, 1e-3f))
                << "nn::CTCLoss: second compiled() call (different input) != "
                   "eager on that input on " << backend_name(dev)
                << " — a cached/frozen graph would replay the FIRST call's "
                   "loss regardless of the actual input";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::CTCLoss threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

// The differentiable (grad_mode) path is deliberately NOT wired for replay
// (OpId::CTCLossForward's raw_grad output feeds a custom autograd Function
// that JIT replay does not capture) — verify this fails safely (falls back
// to eager, per CompiledFunction's own eager-fallback contract) rather than
// silently producing wrong gradients.
TEST(JitTraceOps, CTCLossGradModeFallsBackToEagerInsteadOfWrongGradients) {
    const int64_t T = 5, N = 2, C = 4, S = 2;
    nn::CTCLoss loss("mean", /*blank=*/0, /*zero_infinity=*/false);

    for (const auto& dev : get_available_backends()) {
        try {
            Tensor targets = randint(1, C, {N, S}, DType::Int64, dev);
            Tensor input_lengths = full({N}, static_cast<double>(T), DType::Int64, dev);
            Tensor target_lengths = full({N}, static_cast<double>(S), DType::Int64, dev);

            auto fn = [&](const Variable& logits) {
                auto log_probs = tenzor::nn::log_softmax(logits, 2);
                return loss.forward(log_probs, targets, input_lengths, target_lengths);
            };

            auto compiled = jit::compile(fn);
            auto logits_a = randn({T, N, C}, DType::Float32, dev);

            auto replayed_a = compiled(Variable(logits_a, /*requires_grad=*/true))
                                   .tensor().to(Device::cpu());
            dev.synchronize();

            auto eager_a = fn(Variable(logits_a, /*requires_grad=*/true)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_a, replayed_a, 1e-3f, 1e-3f))
                << "nn::CTCLoss (grad-enabled forward) result != eager on "
                << backend_name(dev);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::CTCLoss (grad-enabled) threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// =========================================================================
// findings.txt JIT-R134: OpId::GridSample/AffineGrid/FFT were entirely
// unmapped in tracing_interceptor.cpp, hard graph-breaking any trace that
// called tenzor::grid_sample()/affine_grid()/fft() (real, reachable
// raw-Tensor dispatch call sites — src/ops/vision.cpp, src/ops/fft.cpp).
// None of these three ops have a Variable-level (autograd) overload in this
// codebase, so a traced call necessarily goes through the raw Tensor
// function wrapped in a non-tracking Variable, matching how the detection
// ops above are traced.
// =========================================================================

TEST(JitTraceOps, GridSampleTracksNewInput) {
    const int64_t N = 2, C = 3, Hin = 5, Win = 5, Hout = 4, Wout = 4;
    for (const auto& dev : get_available_backends()) {
        try {
            auto grid_tensor = (randn({N, Hout, Wout, 2}, DType::Float32, dev) * 0.3f);
            auto grid = Variable(grid_tensor, false);
            auto fn = [&](const Variable& input) {
                return tenzor::grid_sample(input, grid, "bilinear", "zeros", false);
            };

            auto compiled = jit::compile(fn);
            auto input_a = randn({N, C, Hin, Win}, DType::Float32, dev);
            auto input_b = randn({N, C, Hin, Win}, DType::Float32, dev) + 5.0f;

            (void)compiled(Variable(input_a, false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "grid_sample did not compile/cache a graph on "
                << backend_name(dev) << " — OpId::GridSample graph-broke, "
                   "regressing the JIT-R134 fix";

            auto replayed_b = compiled(Variable(input_b, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_b = fn(Variable(input_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_b, replayed_b, 1e-4f, 1e-4f))
                << "grid_sample: replay on a NEW input != eager on that input "
                   "on " << backend_name(dev) << " — the compiled graph likely "
                   "froze the trace-time value as a constant";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "grid_sample threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

TEST(JitTraceOps, AffineGridTracksNewInput) {
    const int64_t N = 2;
    for (const auto& dev : get_available_backends()) {
        try {
            auto fn = [&](const Variable& theta) {
                return tenzor::affine_grid(theta, {N, 3, 8, 8}, false);
            };

            auto compiled = jit::compile(fn);
            auto theta_a = randn({N, 2, 3}, DType::Float32, dev) * 0.1f;
            auto theta_b = randn({N, 2, 3}, DType::Float32, dev) * 0.1f + 0.5f;

            (void)compiled(Variable(theta_a, false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "affine_grid did not compile/cache a graph on "
                << backend_name(dev) << " — OpId::AffineGrid graph-broke, "
                   "regressing the JIT-R134 fix";

            auto replayed_b = compiled(Variable(theta_b, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_b = fn(Variable(theta_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_b, replayed_b, 1e-4f, 1e-4f))
                << "affine_grid: replay on a NEW input != eager on that input "
                   "on " << backend_name(dev) << " — the compiled graph likely "
                   "froze the trace-time value as a constant";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "affine_grid threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

TEST(JitTraceOps, IFFTRFFTIRFFTTrackNewInput) {
    const int64_t N = 8;
    for (const auto& dev : get_available_backends()) {
        try {
            auto fn_ifft = [&](const Variable& input) {
                return tenzor::fft_autograd::ifft(input, std::nullopt, -1, "backward");
            };
            auto compiled_ifft = jit::compile(fn_ifft);
            auto ci_a = randn({N}, DType::Float32, dev).to(DType::Complex64);
            auto ci_b = (randn({N}, DType::Float32, dev) + 2.0f).to(DType::Complex64);
            (void)compiled_ifft(Variable(ci_a, false));
            dev.synchronize();
            EXPECT_GE(compiled_ifft.num_cached(), 1u)
                << "ifft did not compile/cache on " << backend_name(dev);
            auto r = compiled_ifft(Variable(ci_b, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto e = fn_ifft(Variable(ci_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(e, r, 1e-3f, 1e-3f))
                << "ifft: replay on new input != eager on " << backend_name(dev);

            auto fn_rfft = [&](const Variable& input) {
                return tenzor::fft_autograd::rfft(input, std::nullopt, -1, "backward");
            };
            auto compiled_rfft = jit::compile(fn_rfft);
            auto rr_a = randn({N}, DType::Float32, dev);
            auto rr_b = randn({N}, DType::Float32, dev) + 2.0f;
            (void)compiled_rfft(Variable(rr_a, false));
            dev.synchronize();
            EXPECT_GE(compiled_rfft.num_cached(), 1u)
                << "rfft did not compile/cache on " << backend_name(dev);
            auto r2 = compiled_rfft(Variable(rr_b, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto e2 = fn_rfft(Variable(rr_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(e2, r2, 1e-3f, 1e-3f))
                << "rfft: replay on new input != eager on " << backend_name(dev);

            auto fn_irfft = [&](const Variable& input) {
                return tenzor::fft_autograd::irfft(input, N, -1, "backward");
            };
            auto compiled_irfft = jit::compile(fn_irfft);
            auto ir_a = randn({N}, DType::Float32, dev).to(DType::Complex64);
            auto ir_b = (randn({N}, DType::Float32, dev) + 2.0f).to(DType::Complex64);
            (void)compiled_irfft(Variable(ir_a, false));
            dev.synchronize();
            EXPECT_GE(compiled_irfft.num_cached(), 1u)
                << "irfft did not compile/cache on " << backend_name(dev);
            auto r3 = compiled_irfft(Variable(ir_b, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto e3 = fn_irfft(Variable(ir_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(e3, r3, 1e-3f, 1e-3f))
                << "irfft: replay on new input != eager on " << backend_name(dev);
        } catch (const std::exception& e) {
            ADD_FAILURE() << "ifft/rfft/irfft threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

TEST(JitTraceOps, FFTTracksNewInput) {
    const int64_t N = 8;
    for (const auto& dev : get_available_backends()) {
        try {
            auto fn = [&](const Variable& input) {
                return tenzor::fft_autograd::fft(input, std::nullopt, -1, "backward");
            };

            auto compiled = jit::compile(fn);
            auto input_a = randn({N}, DType::Float32, dev).to(DType::Complex64);
            auto input_b = (randn({N}, DType::Float32, dev) + 2.0f).to(DType::Complex64);

            (void)compiled(Variable(input_a, false));
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "fft did not compile/cache a graph on " << backend_name(dev)
                << " — OpId::FFT graph-broke, regressing the JIT-R134 fix";

            auto replayed_b = compiled(Variable(input_b, false)).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_b = fn(Variable(input_b, false)).tensor().to(Device::cpu());
            EXPECT_TRUE(tensors_close(eager_b, replayed_b, 1e-3f, 1e-3f))
                << "fft: replay on a NEW input != eager on that input on "
                << backend_name(dev) << " — the compiled graph likely froze "
                   "the trace-time value as a constant";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "fft threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R093 investigation: NTXentLoss builds its diagonal self-similarity
// mask and pair labels via zeros()+raw .data<>() writes sized to
// z_i.shape()[0] AT TRACE TIME (a host-side metadata read, not a traced
// op) — batch-size-baked constants, superficially matching the JIT-R070
// (WindowAttention mask) freeze pattern. UNLIKE R070, though,
// ShapeGuardInsertionPass (compiler.cpp) inserts a ShapeGuard node for
// EVERY graph input unconditionally, and execute_node's ShapeGuard case
// aborts the replay (triggering a full retrace) the instant a runtime
// input's shape disagrees with the trace-time shape — before the
// mismatched-size mask/labels could ever actually be used in a wrong
// computation. This test confirms that protection empirically: replaying
// with a DIFFERENT batch size must retrace and produce a CORRECT result,
// not a shape-mismatch crash or silently wrong loss.
// =========================================================================

TEST(JitTraceOps, NTXentLossRetracesOnBatchSizeChange) {
    nn::NTXentLoss loss(0.5, nn::Reduction::Mean);
    const int64_t D = 8;

    auto fn = jit::CompiledFunction::FnTypeN(
        [&loss](std::span<const Variable> ins) {
            return loss.forward(ins[0], ins[1]);
        });

    for (const auto& dev : get_available_backends()) {
        try {
            auto compiled = jit::compile(fn);
            auto zi_a = randn({4, D}, DType::Float32, dev);
            auto zj_a = randn({4, D}, DType::Float32, dev);
            std::vector<Variable> ins_a = {Variable(zi_a, false), Variable(zj_a, false)};
            (void)compiled(std::span<const Variable>(ins_a.data(), ins_a.size()));
            dev.synchronize();

            // Second call uses a DIFFERENT batch size (6, not 4).
            auto zi_b = randn({6, D}, DType::Float32, dev);
            auto zj_b = randn({6, D}, DType::Float32, dev);
            std::vector<Variable> ins_b = {Variable(zi_b, false), Variable(zj_b, false)};
            auto replayed_b = compiled(std::span<const Variable>(ins_b.data(), ins_b.size()))
                                   .tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_b = loss.forward(Variable(zi_b, false), Variable(zj_b, false))
                                .tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_b, replayed_b, 1e-3f, 1e-3f))
                << "nn::NTXentLoss: replay with a DIFFERENT batch size != "
                   "eager on that batch size on " << backend_name(dev)
                << " — mask/labels baked at the trace-time batch size were "
                   "not correctly rebuilt for the new size";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::NTXentLoss threw on " << backend_name(dev)
                          << ": " << e.what();
        }
    }
}

// =========================================================================
// JIT-R094 regression coverage: entering an autocast scope INSIDE an
// active JIT trace used to silently corrupt the recorded graph. Confirmed
// empirically before the fix: the tracer's interceptor registers an op's
// input ids from the pre-autocast (original dtype) span it receives, but
// its output id from the REAL result (post-autocast-cast dtype) — a
// replay on a NEW input then computed in the ORIGINAL dtype instead of
// the autocast dtype used at trace time (num_cached()==1, but the
// replayed output's dtype didn't match either the trace-time result's
// dtype or a fresh eager call's dtype). Autocast::Autocast() now refuses
// to enter an enabled scope while Tracer::is_tracing() — matching this
// codebase's established "fail loudly instead of producing wrong
// numerics" pattern.
// =========================================================================

TEST(JitTraceOps, AutocastInsideTraceRefusesToCompile) {
    auto fn = [](const Variable& x) {
        ::tenzor::nn::amp::Autocast autocast(true, DType::Float16);
        return tenzor::matmul(x, x);
    };

    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto a = randn({4, 4}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(a, /*requires_grad=*/false)); },
            std::exception)
            << "nn::amp::Autocast did not throw when entered inside an "
               "active JIT trace on " << backend_name(dev)
            << " — the recorded graph would silently replay in the wrong "
               "dtype on any later call";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "nn::amp::Autocast-inside-trace cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        // Plain eager (non-traced) autocast calls must still work normally.
        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "nn::amp::Autocast incorrectly threw on a plain eager "
               "(non-traced) call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R095 regression coverage: NestedTensor's ragged layout (values_ +
// offsets_) is built/consumed entirely via raw host pointer loops (memcpy,
// direct offsets_.data<>() reads) — zero dispatch() calls, 100% invisible
// to the tracer. Segment boundaries are inherently data-dependent (not
// fixable by adding an OpId mapping), so NestedTensor's key entry points
// now refuse to trace instead of silently freezing the trace-dummy's
// per-sample boundaries/padding layout.
// =========================================================================

TEST(JitTraceOps, NestedTensorToPaddedTensorRefusesToTrace) {
    auto fn = [](const Variable& x) {
        Tensor xt = x.tensor();
        std::vector<Tensor> parts = {xt.slice(0, 0, 2), xt.slice(0, 2, 3)};
        NestedTensor nt = NestedTensor::from_tensor_list(parts);
        return Variable(nt.to_padded_tensor(0.0), false);
    };

    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto a = randn({3, 4}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(a, /*requires_grad=*/false)); },
            std::exception)
            << "NestedTensor did not throw when constructed/consumed inside "
               "an active JIT trace on " << backend_name(dev)
            << " — the recorded graph would silently freeze the trace-time "
               "ragged boundaries/padding layout";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "NestedTensor-inside-trace cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "NestedTensor incorrectly threw on a plain eager (non-traced) "
               "call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R096 regression coverage: SpectralNorm's forward pre-hook runs
// power_iteration() (updating u_/v_/sigma_ — raw Tensors, not module
// parameters/buffers, non-differentiable) on every EAGER forward, but
// jit::trace()/compile() only invoke forward()/hooks ONCE at trace time —
// replay never re-runs power_iteration(), so sigma_ used to freeze at its
// trace-time value forever with NO protective throw anywhere downstream
// (Div is universally supported), silently defeating the Lipschitz
// constraint as weight_orig_ continues training. The pre-hook now refuses
// to trace instead.
// =========================================================================

TEST(JitTraceOps, SpectralNormRefusesToTrace) {
    auto linear = std::make_shared<nn::Linear>(4, 3);
    auto sn = nn::utils::SpectralNorm::apply(linear, "weight", 1);
    (void)sn;

    auto fn = [&linear](const Variable& x) { return linear->forward(x); };

    for (const auto& dev : get_available_backends()) {
        linear->to(dev);
        auto compiled = jit::compile(fn);
        auto a = randn({2, 4}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(a, /*requires_grad=*/false)); },
            std::exception)
            << "nn::utils::SpectralNorm did not throw when traced on "
            << backend_name(dev) << " — the compiled graph would silently "
               "freeze sigma_ at its trace-time value forever, defeating "
               "the Lipschitz constraint as the weight continues training";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "SpectralNorm-wrapped module cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "nn::utils::SpectralNorm incorrectly threw on a plain eager "
               "(non-traced) call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R097 regression coverage: pack_padded_sequence/pad_packed_sequence/
// pack_sequence/pad_sequence are entirely raw-Tensor (permute()/zeros()+
// memcpy/direct .data<>() reads) — zero dispatch() calls, invisible to the
// tracer. The packing/padding layout (batch_sizes, sorted order, max_len)
// is computed host-side from the ACTUAL runtime lengths — data-dependent,
// same class as JIT-R095 (NestedTensor). Now refuses to trace instead of
// silently freezing the trace-dummy's layout.
// =========================================================================

TEST(JitTraceOps, PackPaddedSequenceRefusesToTrace) {
    auto fn = [](const Variable& x) {
        Tensor lengths({2}, DType::Int64, Device::cpu());
        lengths.data<int64_t>()[0] = 3;
        lengths.data<int64_t>()[1] = 2;
        auto packed = nn::pack_padded_sequence(x.tensor(), lengths,
                                                /*batch_first=*/true,
                                                /*enforce_sorted=*/false);
        return Variable(packed.data, false);
    };

    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto a = randn({2, 3, 4}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(a, /*requires_grad=*/false)); },
            std::exception)
            << "nn::pack_padded_sequence did not throw when traced on "
            << backend_name(dev) << " — the compiled graph would silently "
               "freeze the trace-time packing layout";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "pack_padded_sequence-inside-trace cached a compiled graph "
               "on " << backend_name(dev) << " despite throwing during trace";

        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "nn::pack_padded_sequence incorrectly threw on a plain eager "
               "(non-traced) call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R050: MixtureOfExperts' per-expert routing (which experts fire, how
// many tokens each gets, capacity-based token dropping) is computed
// host-side from the ACTUAL routing output of the given input — data-
// dependent, same class as JIT-R095 (NestedTensor) and the
// PackPaddedSequence case above. No dedicated MoE OpType exists to
// represent dynamic per-expert dispatch, so a trace would silently freeze
// the trace-dummy's routing decision as permanent graph structure. Refuses
// to trace instead.
// =========================================================================

TEST(JitTraceOps, MixtureOfExpertsRefusesToTrace) {
    auto moe = std::make_shared<nn::MixtureOfExperts>(
        /*input_dim=*/4, /*hidden_dim=*/8, /*num_experts=*/4, /*top_k=*/2);

    auto fn = [&moe](const Variable& x) { return moe->forward(x); };

    for (const auto& dev : get_available_backends()) {
        moe->to(dev);
        auto compiled = jit::compile(fn);
        auto a = randn({3, 4}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(a, /*requires_grad=*/false)); },
            std::exception)
            << "nn::MixtureOfExperts did not throw when traced on "
            << backend_name(dev) << " — the compiled graph would silently "
               "freeze the trace-dummy's per-expert routing decision "
               "forever, producing wrong output for every other input";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "MixtureOfExperts-wrapped function cached a compiled graph "
               "on " << backend_name(dev) << " despite throwing during trace";

        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "nn::MixtureOfExperts incorrectly threw on a plain eager "
               "(non-traced) call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R051: HRM's Adaptive Computation Time (ACT) halting decision reads a
// runtime tensor value host-side and uses it to `break` out of the
// high-cycle loop — data-dependent, same class as JIT-R050 (MoE) above. No
// dedicated OpType exists to represent early-exit over a hand-rolled C++
// loop, so a trace would silently freeze the trace-dummy's halting cycle
// count as permanent graph structure. Refuses to trace instead, but ONLY
// when a halting mechanism (use_act) is actually configured — without it
// the high-cycle loop is a genuinely static, fully traceable unroll.
// =========================================================================

TEST(JitTraceOps, HRMWithACTRefusesToTrace) {
    nn::HRMConfig config;
    config.d_model = 8;
    config.n_heads = 2;
    config.d_feedforward = 16;
    config.n_high_cycles = 2;
    config.t_low_steps = 2;
    config.dropout = 0.0;
    config.use_act = true;
    config.use_qlearning_act = true;
    config.max_seq_len = 8;
    auto hrm = std::make_shared<nn::HRM>(config);
    hrm->eval();  // disable exploration for determinism

    auto fn = [&hrm](const Variable& x) { return hrm->forward(x); };

    for (const auto& dev : get_available_backends()) {
        hrm->to(dev);
        auto compiled = jit::compile(fn);
        auto a = randn({2, 4, 8}, DType::Float32, dev);

        EXPECT_THROW(
            { (void)compiled(Variable(a, /*requires_grad=*/false)); },
            std::exception)
            << "nn::HRM (ACT-enabled) did not throw when traced on "
            << backend_name(dev) << " — the compiled graph would silently "
               "freeze the trace-dummy's halting cycle count forever, "
               "producing wrong output for every other input";

        EXPECT_EQ(compiled.num_cached(), 0u)
            << "HRM (ACT-enabled) cached a compiled graph on "
            << backend_name(dev) << " despite throwing during trace";

        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "nn::HRM (ACT-enabled) incorrectly threw on a plain eager "
               "(non-traced) call on " << backend_name(dev);
    }
}

// =========================================================================
// JIT-R084: Swish/RReLU/LogSigmoid's no-grad-fast-path dispatch<>() call
// used to be unmapped (opid_to_optype had no entry), graph-breaking the
// whole trace — reachable in real deployed inference via nn::Swish
// (MobileNetV3/EfficientNet-style blocks). Now maps to a real OpType with
// a working execute_node case, so these actually compile/cache instead of
// silently degrading to eager. RReLU also needs a training-mode guard
// (mirrors JIT-R013's Dropout fix): the eval-mode fixed-midpoint-slope
// path is genuinely traceable, but training=true draws fresh randomness
// per call that a trace cannot represent.
// =========================================================================

TEST(JitTraceOps, SwishTracesAndCachesAcrossBackends) {
    auto fn = [](const Variable& x) { return nn::swish(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto a = randn({4, 8}, DType::Float32, dev);
        Variable out;
        ASSERT_NO_THROW({ out = compiled(Variable(a, /*requires_grad=*/false)); })
            << "nn::swish threw when traced on " << backend_name(dev);
        EXPECT_GE(compiled.num_cached(), 1u)
            << "nn::swish did not compile/cache a graph on " << backend_name(dev)
            << " — the op likely still graph-broke";
        const auto eager = nn::swish(Variable(a, false));
        EXPECT_LT(max_abs_diff(eager.tensor(), out.tensor()), 1e-4f)
            << "nn::swish JIT output diverged from eager on " << backend_name(dev);
    }
}

TEST(JitTraceOps, LogSigmoidTracesAndCachesAcrossBackends) {
    auto fn = [](const Variable& x) { return nn::log_sigmoid(x); };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto a = randn({4, 8}, DType::Float32, dev);
        Variable out;
        ASSERT_NO_THROW({ out = compiled(Variable(a, /*requires_grad=*/false)); })
            << "nn::log_sigmoid threw when traced on " << backend_name(dev);
        EXPECT_GE(compiled.num_cached(), 1u)
            << "nn::log_sigmoid did not compile/cache a graph on "
            << backend_name(dev) << " — the op likely still graph-broke";
        const auto eager = nn::log_sigmoid(Variable(a, false));
        EXPECT_LT(max_abs_diff(eager.tensor(), out.tensor()), 1e-4f)
            << "nn::log_sigmoid JIT output diverged from eager on "
            << backend_name(dev);
    }
}

TEST(JitTraceOps, RReLUEvalModeTracesAndCachesAcrossBackends) {
    auto fn = [](const Variable& x) {
        return nn::rrelu(x, 1.0 / 8.0, 1.0 / 3.0, /*training=*/false);
    };
    for (const auto& dev : get_available_backends()) {
        auto compiled = jit::compile(fn);
        auto a = randn({4, 8}, DType::Float32, dev);
        Variable out;
        ASSERT_NO_THROW({ out = compiled(Variable(a, /*requires_grad=*/false)); })
            << "nn::rrelu(training=false) threw when traced on "
            << backend_name(dev);
        EXPECT_GE(compiled.num_cached(), 1u)
            << "nn::rrelu(training=false) did not compile/cache a graph on "
            << backend_name(dev) << " — the op likely still graph-broke";
        const auto eager = nn::rrelu(Variable(a, false), 1.0 / 8.0, 1.0 / 3.0, false);
        EXPECT_LT(max_abs_diff(eager.tensor(), out.tensor()), 1e-4f)
            << "nn::rrelu(training=false) JIT output diverged from eager on "
            << backend_name(dev);
    }
}

TEST(JitTraceOps, RReLUTrainingModeRefusesToReplay) {
    // training=true + no-grad still routes through dispatch<OpId::RReLU>
    // (see rrelu()'s needs_grad-gated branch in activations.cpp), so it
    // traces successfully but the compiled graph must refuse to REPLAY it
    // (the kernel draws fresh per-call randomness a frozen trace cannot
    // represent) — never silently replaying a frozen trace-time random
    // slope is the invariant this test guards, not any particular
    // exception-propagation shape.
    //
    // JIT-R120: CompiledFunction::operator()'s cache-hit path used to have
    // no exception-safety net at all, so the graph-level replay-refusal
    // (Graph::execute_node's OpType::RReLU throw, whose own message ends
    // "...that a trace cannot represent); fall back to eager") propagated
    // all the way to the caller as an uncaught hard error even in
    // non-strict mode — inconsistent with every OTHER JIT failure mode in
    // this file (compile failure, graph break, mlir invoke failure), all of
    // which honor config.strict. Now that the cache-hit path has the same
    // strict-aware safety net, non-strict mode must gracefully fall back to
    // a genuinely fresh eager call (fn_ was never run this call, so no
    // double-exec risk) instead of throwing, while strict=true must still
    // surface the incompatibility as a hard error, matching every other
    // JIT fallback's strict contract.
    auto fn = [](const Variable& x) {
        return nn::rrelu(x, 1.0 / 8.0, 1.0 / 3.0, /*training=*/true);
    };
    for (const auto& dev : get_available_backends()) {
        // Non-strict (the default): cache-hit replay failure gracefully
        // falls back to eager instead of throwing.
        auto compiled = jit::compile(fn);
        auto a = randn({4, 8}, DType::Float32, dev);
        // First call: traces successfully (fn_ ran eagerly during trace).
        ASSERT_NO_THROW({ (void)compiled(Variable(a, /*requires_grad=*/false)); })
            << "nn::rrelu(training=true) unexpectedly threw on first "
               "(trace) call on " << backend_name(dev);

        auto b = randn({4, 8}, DType::Float32, dev);
        Variable out1, out2;
        ASSERT_NO_THROW({ out1 = compiled(Variable(b, /*requires_grad=*/false)); })
            << "nn::rrelu(training=true) cache-hit replay must gracefully "
               "fall back to eager (JIT-R120), not throw, in non-strict mode "
               "on " << backend_name(dev);
        ASSERT_NO_THROW({ out2 = compiled(Variable(b, /*requires_grad=*/false)); })
            << "the eager fallback must keep working on every subsequent "
               "call, not just once, on " << backend_name(dev);
        // The safety property the ORIGINAL guard exists for: never silently
        // replay a FROZEN trace-time random slope. Two independent eager
        // fallback calls on the identical input must draw independent
        // randomness (overwhelmingly unlikely to be bit-identical by
        // chance), proving this is a genuine fresh recomputation each time,
        // not a frozen replay slipping through under a different guise.
        EXPECT_GT(tenzor::max(tenzor::abs(
                      out1.tensor() - out2.tensor())).item<float>(), 0.0f)
            << "two eager-fallback calls on the same input produced "
               "bit-identical output — a frozen/stale replay may have "
               "slipped through despite the guard, on " << backend_name(dev);

        // Strict mode: the same incompatibility must surface as a hard
        // error, matching every other JIT fallback's strict contract.
        jit::CompileConfig strict_cfg;
        strict_cfg.backend = "nvrtc";
        strict_cfg.strict = true;
        jit::CompiledFunction strict_compiled(
            jit::CompiledFunction::FnType(fn), strict_cfg);
        auto c = randn({4, 8}, DType::Float32, dev);
        ASSERT_NO_THROW({ (void)strict_compiled(Variable(c, /*requires_grad=*/false)); })
            << "trace (first) call itself must not throw under strict mode "
               "on " << backend_name(dev);
        auto d = randn({4, 8}, DType::Float32, dev);
        EXPECT_THROW(
            { (void)strict_compiled(Variable(d, /*requires_grad=*/false)); },
            std::exception)
            << "nn::rrelu(training=true) must throw on cache-hit replay "
               "under explicit strict=true on " << backend_name(dev);

        EXPECT_NO_THROW({ (void)fn(Variable(a, false)); })
            << "nn::rrelu(training=true) incorrectly threw on a plain "
               "eager (non-traced) call on " << backend_name(dev);
    }
}

// JIT-R064: Embedding::renorm_embeddings' non-differentiable max_norm clamp
// (scale = min(1, max_norm/norm); weight = index_copy(weight, 0, idx,
// rows * scale)) dispatches OpId::Minimum/OpId::IndexCopy, both previously
// unmapped in the tracer — abort_trace_unmappable() killed the WHOLE trace
// on every forward call, deterministically blocking any Embedding(max_norm>0)
// module from ever JIT-compiling.
TEST(JitTraceOps, EmbeddingMaxNormTracesAndCachesAcrossBackends) {
    NoGradGuard no_grad;
    nn::Embedding embed(/*num_embeddings=*/10, /*embedding_dim=*/4,
                        /*padding_idx=*/-1, /*max_norm=*/1.0,
                        /*norm_type=*/2.0);
    for (const auto& dev : get_available_backends()) {
        try {
            embed.to(dev);
            // Scale weight rows well past max_norm so renorm actually clamps
            // (not a no-op scale=1 for every row).
            embed.weight().tensor() = embed.weight().tensor() * 5.0;

            auto fn = [&embed](const Variable& indices) {
                return embed.forward(indices);
            };
            auto compiled = jit::compile(fn);
            auto dummy = Variable(make_index_tensor({0, 1, 2, 3}, dev), false);
            auto other = Variable(make_index_tensor({9, 8, 7, 6}, dev), false);

            (void)compiled(dummy);
            dev.synchronize();

            EXPECT_GE(compiled.num_cached(), 1u)
                << "nn::Embedding(max_norm>0) did not compile/cache a graph "
                   "on " << backend_name(dev)
                << " — the op likely still graph-broke";

            auto replayed = compiled(other).tensor().to(Device::cpu());
            dev.synchronize();
            auto eager_on_other = fn(other).tensor().to(Device::cpu());

            EXPECT_TRUE(tensors_close(eager_on_other, replayed, 1e-4f, 1e-4f))
                << "nn::Embedding(max_norm>0): replay on NEW indices != "
                   "eager on those indices on " << backend_name(dev)
                << " — the compiled graph likely froze the trace-time value "
                   "as a constant instead of tracing the renorm op";
        } catch (const std::exception& e) {
            ADD_FAILURE() << "nn::Embedding(max_norm>0) threw on "
                          << backend_name(dev) << ": " << e.what();
        }
    }
}

// NOTE: this only asserts the trace doesn't THROW (an unrelated, separate
// graph-break cause elsewhere in HRM's forward path — not the ACT halting
// decision this finding is about — currently still prevents a fully
// compiled/cached graph even here; see JIT-R051's residual note in
// findings.txt). Safe (eager fallback), just not yet fully compiled.
TEST(JitTraceOps, HRMWithoutACTStillTracesNewInput) {
    nn::HRMConfig config;
    config.d_model = 8;
    config.n_heads = 2;
    config.d_feedforward = 16;
    config.n_high_cycles = 2;
    config.t_low_steps = 2;
    config.dropout = 0.0;
    config.use_act = false;  // no halting mechanism -> genuinely static loop
    config.max_seq_len = 8;
    auto hrm = std::make_shared<nn::HRM>(config);
    hrm->eval();

    auto fn = [&hrm](const Variable& x) { return hrm->forward(x); };

    for (const auto& dev : get_available_backends()) {
        hrm->to(dev);
        auto compiled = jit::compile(fn);
        auto a = randn({2, 4, 8}, DType::Float32, dev);

        EXPECT_NO_THROW({ (void)compiled(Variable(a, /*requires_grad=*/false)); })
            << "nn::HRM (no ACT) incorrectly threw when traced on "
            << backend_name(dev) << " — the high-cycle loop has no "
               "data-dependent early exit without ACT and should trace fine";
    }
}

// =========================================================================
// JIT-R082/R085-R089: object-detection subsystem regression coverage.
// CPU-only by design — these tests verify TRACER MECHANICS (does replay use
// the real new input vs. a frozen trace-time constant; do genuinely
// data-dependent functions refuse to trace) rather than backend-kernel
// numerics, so a single backend is sufficient and avoids the GPU driver
// flakiness observed elsewhere in this environment.
// =========================================================================

namespace {
Tensor box_tensor(std::initializer_list<std::initializer_list<float>> rows) {
    std::vector<float> flat;
    int64_t ncols = static_cast<int64_t>(rows.begin()->size());
    for (const auto& row : rows) {
        for (float v : row) flat.push_back(v);
    }
    return ::tenzor::from_data(flat.data(),
        {static_cast<int64_t>(rows.size()), ncols}, Device::cpu());
}
Tensor float1d_tensor(std::initializer_list<float> vals) {
    std::vector<float> v(vals);
    return ::tenzor::from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu());
}
Tensor int64_1d_tensor(std::initializer_list<int64_t> vals) {
    std::vector<int64_t> v(vals);
    return ::tenzor::from_data(v.data(), {static_cast<int64_t>(v.size())}, Device::cpu());
}
}  // namespace

TEST(JitDetectionOps, BoxIoUTracksNewInput) {
    // jit::compile's convenient single-callable overload traces a single-
    // Variable function; the second operand is captured by value (a real,
    // fixed tensor for the whole test — not a magic placeholder) matching
    // this file's established check_conv_replay pattern for multi-operand
    // ops. This still directly exercises the fix: `a`'s value changing
    // between trace and replay must flow through, not freeze.
    auto b = box_tensor({{0.0f, 0.0f, 4.0f, 4.0f}});
    auto fn = [b](const Variable& a) {
        return Variable(ops::box_iou(a.tensor(), b), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_a = box_tensor({{0.0f, 0.0f, 1.0f, 1.0f}});
    (void)compiled(Variable(dummy_a, false));

    auto other_a = box_tensor({{1.0f, 1.0f, 3.0f, 3.0f}});
    auto replayed = compiled(Variable(other_a, false));
    auto eager = ops::box_iou(other_a, b);

    EXPECT_GE(compiled.num_cached(), 1u) << "box_iou did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-5f, 1e-5f))
        << "BoxIoU replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitDetectionOps, NMSTracksNewInput) {
    auto scores = float1d_tensor({0.5f, 0.95f, 0.6f});
    auto fn = [scores](const Variable& boxes) {
        return Variable(ops::nms(boxes.tensor(), scores, 0.5), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_boxes = box_tensor({{0.0f, 0.0f, 1.0f, 1.0f}, {0.0f, 0.0f, 1.0f, 1.0f},
                                    {0.0f, 0.0f, 1.0f, 1.0f}});
    (void)compiled(Variable(dummy_boxes, false));

    auto other_boxes = box_tensor({{0.0f, 0.0f, 1.0f, 1.0f}, {5.0f, 5.0f, 6.0f, 6.0f},
                                    {5.1f, 5.1f, 6.1f, 6.1f}});
    auto replayed = compiled(Variable(other_boxes, false));
    auto eager = ops::nms(other_boxes, scores, 0.5);

    EXPECT_GE(compiled.num_cached(), 1u) << "nms did not compile/cache";
    ASSERT_EQ(replayed.tensor().shape()[0], eager.shape()[0])
        << "NMS replay on a NEW input kept a different box count than eager "
           "on that input — frozen trace-time constant";
    EXPECT_TRUE(tensors_close(eager.to(DType::Int64), replayed.tensor().to(DType::Int64), 0.0f, 0.0f))
        << "NMS replay on a NEW input != eager on that input";
}

TEST(JitDetectionOps, ROIAlignTracksNewInputAndReplaysDifferentiably) {
    nn::detection::ROIAlign roi_align(2, 2, 1.0, 2, true);
    auto rois = box_tensor({{0.0f, 0.0f, 0.0f, 4.0f, 4.0f}});
    auto fn = [&roi_align, rois](const Variable& features) {
        return roi_align.forward(features, rois);
    };
    auto compiled = jit::compile(fn);
    auto dummy_features = randn({1, 2, 8, 8}, DType::Float32, Device::cpu());
    (void)compiled(Variable(dummy_features, /*requires_grad=*/true));

    auto other_features = randn({1, 2, 8, 8}, DType::Float32, Device::cpu());
    Variable other_features_v(other_features, /*requires_grad=*/true);
    auto replayed = compiled(other_features_v);
    auto eager = roi_align.forward(Variable(other_features, /*requires_grad=*/true), rois);

    EXPECT_GE(compiled.num_cached(), 1u) << "ROIAlign did not compile/cache";
    EXPECT_TRUE(tensors_close(eager.tensor(), replayed.tensor(), 1e-4f, 1e-4f))
        << "ROIAlign replay on a NEW input != eager on that input — frozen "
           "trace-time constant";
    EXPECT_TRUE(replayed.requires_grad())
        << "ROIAlign replay lost requires_grad — JIT-R082's differentiable-"
           "replay fix regressed";
    ASSERT_NE(replayed.grad_fn(), nullptr)
        << "ROIAlign replay produced no grad_fn — cannot backward through "
           "the compiled graph";
}

TEST(JitDetectionOps, AnchorGeneratorTracksNewFeatureMapSize) {
    nn::detection::AnchorGenerator gen({32.0f, 64.0f}, {0.5f, 1.0f, 2.0f});
    auto fn = [&gen](const Variable& dummy_input) {
        (void)dummy_input;
        return Variable(gen.generate(4, 4, 16, Device::cpu()), false);
    };
    auto compiled = jit::compile(fn);
    auto probe = zeros({1}, DType::Float32, Device::cpu());
    auto first = compiled(Variable(probe, false));
    auto eager_first = gen.generate(4, 4, 16, Device::cpu());
    EXPECT_GE(compiled.num_cached(), 1u) << "AnchorGenerate did not compile/cache";
    EXPECT_TRUE(tensors_close(eager_first, first.tensor(), 1e-5f, 1e-5f))
        << "AnchorGenerator replay != eager generate() — JIT-R085's manual "
           "recording fix produced a wrong value";
}

TEST(JitDetectionOps, EncodeDecodeBoxesTrackNewInput) {
    auto anchors = box_tensor({{0.0f, 0.0f, 4.0f, 4.0f}});
    auto fn = [anchors](const Variable& boxes) {
        auto encoded = ops::encode_boxes(boxes.tensor(), anchors);
        auto decoded = ops::decode_boxes(encoded, anchors);
        return Variable(decoded, false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_boxes = box_tensor({{0.0f, 0.0f, 2.0f, 2.0f}});
    (void)compiled(Variable(dummy_boxes, false));

    auto other_boxes = box_tensor({{1.0f, 1.0f, 5.0f, 7.0f}});
    auto replayed = compiled(Variable(other_boxes, false));
    auto eager_encoded = ops::encode_boxes(other_boxes, anchors);
    auto eager_decoded = ops::decode_boxes(eager_encoded, anchors);

    EXPECT_TRUE(tensors_close(eager_decoded, replayed.tensor(), 1e-3f, 1e-3f))
        << "encode_boxes/decode_boxes replay on a NEW input != eager on "
           "that input — the column-slice tracer-invisibility bug (JIT-R085) "
           "regressed";
}

TEST(JitDetectionOps, ClipBoxesToImageTracksNewInput) {
    auto fn = [](const Variable& boxes) {
        return Variable(ops::clip_boxes_to_image(boxes.tensor(), 10, 10), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = box_tensor({{-5.0f, -5.0f, 15.0f, 15.0f}});
    (void)compiled(Variable(dummy, false));

    auto other = box_tensor({{-3.0f, 2.0f, 20.0f, 8.0f}});
    auto replayed = compiled(Variable(other, false));
    auto eager = ops::clip_boxes_to_image(other, 10, 10);

    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-5f, 1e-5f))
        << "clip_boxes_to_image replay on a NEW input != eager on that "
           "input — JIT-R085 regressed";
}

TEST(JitDetectionOps, RemoveSmallBoxesTracksNewInput) {
    auto scores = float1d_tensor({0.1f, 0.9f, 0.8f});
    auto fn = [scores](const Variable& boxes) {
        return Variable(ops::remove_small_boxes(boxes.tensor(), scores, 1.0), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_boxes = box_tensor({{0.0f, 0.0f, 2.0f, 2.0f}, {0.0f, 0.0f, 0.5f, 0.5f},
                                    {0.0f, 0.0f, 3.0f, 3.0f}});
    (void)compiled(Variable(dummy_boxes, false));

    auto other_boxes = box_tensor({{0.0f, 0.0f, 0.2f, 0.2f}, {0.0f, 0.0f, 5.0f, 5.0f},
                                    {0.0f, 0.0f, 3.0f, 3.0f}});
    auto replayed = compiled(Variable(other_boxes, false));
    auto eager = ops::remove_small_boxes(other_boxes, scores, 1.0);

    ASSERT_EQ(replayed.tensor().shape()[0], eager.shape()[0])
        << "remove_small_boxes (nonzero CPU path) replay on a NEW input kept "
           "a different count than eager — JIT-R085's CPU nonzero manual-"
           "recording fix regressed";
    EXPECT_TRUE(tensors_close(eager.to(DType::Int64), replayed.tensor().to(DType::Int64), 0.0f, 0.0f))
        << "remove_small_boxes replay on a NEW input != eager on that input";
}

TEST(JitDetectionOps, SelectTracksNewInput) {
    auto fn = [](const Variable& x) {
        return Variable(ops::select(x.tensor(), 0, 1), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy = box_tensor({{1.0f, 2.0f}, {3.0f, 4.0f}});
    (void)compiled(Variable(dummy, false));

    auto other = box_tensor({{10.0f, 20.0f}, {30.0f, 40.0f}});
    auto replayed = compiled(Variable(other, false));
    auto eager = ops::select(other, 0, 1);

    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-5f, 1e-5f))
        << "select replay on a NEW input != eager on that input — JIT-R085 "
           "regressed";
}

// -------------------------------------------------------------------------
// JIT-R086/R088/R089: genuinely data-dependent detection-pipeline control
// flow must refuse to trace loudly (matching the established JIT-R050/R051
// MoE/HRM precedent) rather than silently freeze one trace-time branch/
// ground-truth outcome as fixed graph structure.
// -------------------------------------------------------------------------

TEST(JitDetectionOps, GenerateProposalsRefusesToTrace) {
    // forward_proposals is the public traceable entry point; it calls the
    // private generate_proposals() unconditionally per image (training or
    // not), so this exercises JIT-R088's refusal guard through real usage.
    auto rpn = std::make_shared<nn::detection::RegionProposalNetwork>(
        4, std::make_shared<nn::detection::AnchorGenerator>(
               std::vector<float>{32.0f}, std::vector<float>{1.0f}));
    rpn->eval();
    std::vector<std::pair<int64_t, int64_t>> image_shapes{{16, 16}};

    auto fn = [rpn, image_shapes](const Variable& features) {
        auto proposals = rpn->forward_proposals(features, image_shapes, nullptr);
        return Variable(proposals[0], false);
    };
    auto compiled = jit::compile(fn);
    auto probe = randn({1, 4, 4, 4}, DType::Float32, Device::cpu());
    EXPECT_THROW({ (void)compiled(Variable(probe, false)); }, std::runtime_error)
        << "RegionProposalNetwork::generate_proposals should refuse to "
           "trace (JIT-R088) instead of silently baking one trace-time "
           "branch as fixed structure";
}

TEST(JitDetectionOps, ProcessMasksRefusesToTrace) {
    auto mask_logits = randn({1, 1, 4, 4}, DType::Float32, Device::cpu());
    auto boxes = box_tensor({{0.0f, 0.0f, 3.0f, 3.0f}});
    auto class_labels = int64_1d_tensor({1});

    auto fn = [&](const Variable& dummy) {
        (void)dummy;
        return Variable(
            nn::detection::process_masks(mask_logits, boxes, class_labels, 8, 8, 0.5),
            false);
    };
    auto compiled = jit::compile(fn);
    auto probe = zeros({1}, DType::Float32, Device::cpu());
    EXPECT_THROW({ (void)compiled(Variable(probe, false)); }, std::runtime_error)
        << "MaskHead::process_masks should refuse to trace (JIT-R088) "
           "instead of silently freezing its variable-trip-count per-"
           "detection loop";
}

TEST(JitDetectionOps, AssignAnchorsToGtRefusesToTrace) {
    // forward_proposals(..., targets) is the public traceable entry point
    // that reaches the private assign_anchors_to_gt() when training with
    // real ground truth — exercises JIT-R086/R089's refusal guard through
    // real usage, not a direct private-method call.
    auto rpn = std::make_shared<nn::detection::RegionProposalNetwork>(
        4, std::make_shared<nn::detection::AnchorGenerator>(
               std::vector<float>{32.0f}, std::vector<float>{1.0f}));
    rpn->train();
    std::vector<std::pair<int64_t, int64_t>> image_shapes{{16, 16}};
    std::vector<Tensor> targets{box_tensor({{0.0f, 0.0f, 8.0f, 8.0f}})};

    auto fn = [rpn, image_shapes, targets](const Variable& features) {
        auto proposals = rpn->forward_proposals(features, image_shapes, &targets);
        return Variable(proposals[0], false);
    };
    auto compiled = jit::compile(fn);
    auto probe = randn({1, 4, 4, 4}, DType::Float32, Device::cpu());
    EXPECT_THROW({ (void)compiled(Variable(probe, false)); }, std::runtime_error)
        << "RegionProposalNetwork::assign_anchors_to_gt should refuse to "
           "trace (JIT-R086/R089) instead of silently freezing GT-dependent "
           "anchor labels";
}

TEST(JitDetectionOps, PostprocessDetectionsRefusesToTrace) {
    // forward_detections calls the private postprocess_detections()
    // unconditionally (training or not) — exercises JIT-R088's refusal
    // guard through real usage.
    auto roi_head = std::make_shared<nn::detection::RoIHead>(
        /*in_channels=*/4, /*num_classes=*/2, /*roi_output_size=*/2,
        /*spatial_scale=*/1.0, /*sampling_ratio=*/2);
    roi_head->eval();
    std::vector<Tensor> proposals{box_tensor({{0.0f, 0.0f, 4.0f, 4.0f}})};
    std::vector<std::pair<int64_t, int64_t>> image_shapes{{8, 8}};

    auto fn = [roi_head, proposals, image_shapes](const Variable& features) {
        auto detections = roi_head->forward_detections(
            features, proposals, image_shapes, nullptr, nullptr);
        return Variable(detections[0].at("boxes"), false);
    };
    auto compiled = jit::compile(fn);
    auto probe = randn({1, 4, 8, 8}, DType::Float32, Device::cpu());
    EXPECT_THROW({ (void)compiled(Variable(probe, false)); }, std::runtime_error)
        << "RoIHead::postprocess_detections should refuse to trace "
           "(JIT-R088) instead of silently baking one trace-time detection-"
           "count branch as fixed structure";
}

// ============================================================================
// JIT-R110: 11 LAPACK-backed linalg ops previously bypassed the tracer
// entirely on CPU (their LAPACKE calls never went through dispatch(), so
// the tracer never recorded them -- the op's output silently froze as a
// trace-time constant on replay) and hard-broke the WHOLE trace on GPU/
// complex-dtype dispatch (real backend kernels exist and go through
// dispatch(), but no OpType existed to map the OpId to). Each test below
// traces once, replays on a genuinely DIFFERENT input, and verifies the
// replayed result matches fresh eager computation on that new input -- a
// frozen trace-time constant would instead keep returning the FIRST call's
// result regardless of what's passed on replay.
// ============================================================================

namespace {
auto make_mat(std::vector<int64_t> shape, std::vector<float> values) -> Tensor {
    auto t = zeros(shape, DType::Float32, Device::cpu());
    auto* data = t.data<float>();
    for (size_t i = 0; i < values.size() && i < static_cast<size_t>(t.numel()); ++i) {
        data[i] = values[i];
    }
    return t;
}
} // namespace

TEST(JitLinalgOps, SolveTriangularTracksNewInput) {
    auto A = make_mat({3, 3}, {2.0f, 1.0f, 0.5f,
                               0.0f, 3.0f, 1.0f,
                               0.0f, 0.0f, 1.5f});
    auto fn = [A](const Variable& B) {
        return Variable(linalg::solve_triangular(A, B.tensor(), /*upper=*/true,
                                                   /*unitriangular=*/false),
                        false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_B = make_mat({3, 1}, {1.0f, 2.0f, 3.0f});
    (void)compiled(Variable(dummy_B, false));

    auto other_B = make_mat({3, 1}, {4.0f, -1.0f, 2.0f});
    auto replayed = compiled(Variable(other_B, false));
    auto eager = linalg::solve_triangular(A, other_B, true, false);

    EXPECT_GE(compiled.num_cached(), 1u) << "solve_triangular did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "SolveTriangular replay on a NEW input != eager on that input — "
           "frozen trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, SlogdetTracksNewInput) {
    auto fn = [](const Variable& A) {
        auto [sign, logabsdet] = linalg::slogdet(A.tensor());
        return Variable(tenzor::add(sign, logabsdet), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({3, 3}, {4.0f, 1.0f, 0.0f, 1.0f, 4.0f, 1.0f, 0.0f, 1.0f, 4.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({3, 3}, {5.0f, 2.0f, 1.0f, 2.0f, 6.0f, 3.0f, 1.0f, 3.0f, 7.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_sign, eager_logabsdet] = linalg::slogdet(other_A);
    auto eager = tenzor::add(eager_sign, eager_logabsdet);

    EXPECT_GE(compiled.num_cached(), 1u) << "slogdet did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "Slogdet CPU replay on a NEW input != eager on that input — "
           "frozen trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, EigvalshTracksNewInput) {
    auto fn = [](const Variable& A) {
        return Variable(linalg::eigvalsh(A.tensor()), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({3, 3}, {2.0f, 1.0f, 0.0f, 1.0f, 2.0f, 1.0f, 0.0f, 1.0f, 2.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({3, 3}, {3.0f, 1.0f, 0.0f, 1.0f, 3.0f, 1.0f, 0.0f, 1.0f, 3.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto eager = linalg::eigvalsh(other_A);

    EXPECT_GE(compiled.num_cached(), 1u) << "eigvalsh did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "Eigvalsh replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, EigTracksNewInput) {
    auto fn = [](const Variable& A) {
        auto [Wr, Wi, V] = linalg::eig(A.tensor());
        (void)V;
        return Variable(tenzor::add(Wr, Wi), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({3, 3}, {2.0f, 1.0f, 0.0f, 0.0f, 3.0f, 1.0f, 1.0f, 0.0f, 2.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({3, 3}, {3.0f, 0.0f, 1.0f, 1.0f, 2.0f, 0.0f, 0.0f, 1.0f, 4.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_Wr, eager_Wi, eager_V] = linalg::eig(other_A);
    (void)eager_V;
    auto eager = tenzor::add(eager_Wr, eager_Wi);

    EXPECT_GE(compiled.num_cached(), 1u) << "eig did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-3f, 1e-3f))
        << "Eig replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, LuTracksNewInput) {
    auto fn = [](const Variable& A) {
        auto [L, U, pivots] = linalg::lu(A.tensor());
        (void)pivots;
        return Variable(tenzor::matmul(L, U), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({3, 3}, {4.0f, 3.0f, 2.0f, 2.0f, 5.0f, 1.0f, 1.0f, 1.0f, 6.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({3, 3}, {6.0f, 1.0f, 2.0f, 1.0f, 7.0f, 3.0f, 2.0f, 3.0f, 8.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_L, eager_U, eager_pivots] = linalg::lu(other_A);
    (void)eager_pivots;
    auto eager = tenzor::matmul(eager_L, eager_U);

    EXPECT_GE(compiled.num_cached(), 1u) << "lu did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "LU replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, LuSolveTracksNewInput) {
    // Composite trace (lu -> pack -> lu_solve) so LU_data/pivots are always
    // a genuinely valid factorization of whatever A is passed, on every
    // call -- not just the trace-time A.
    auto B = make_mat({3, 1}, {1.0f, 2.0f, 3.0f});
    auto eye3 = tenzor::eye(3);
    auto fn = [B, eye3](const Variable& A) {
        auto [L, U, pivots] = linalg::lu(A.tensor());
        auto LU_data = tenzor::add(tenzor::sub(L, eye3), U);
        return Variable(linalg::lu_solve(LU_data, pivots, B), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({3, 3}, {4.0f, 3.0f, 2.0f, 2.0f, 5.0f, 1.0f, 1.0f, 1.0f, 6.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({3, 3}, {6.0f, 1.0f, 2.0f, 1.0f, 7.0f, 3.0f, 2.0f, 3.0f, 8.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_L, eager_U, eager_pivots] = linalg::lu(other_A);
    auto eager_LU_data = tenzor::add(tenzor::sub(eager_L, eye3), eager_U);
    auto eager = linalg::lu_solve(eager_LU_data, eager_pivots, B);

    EXPECT_GE(compiled.num_cached(), 1u) << "lu_solve did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-3f, 1e-3f))
        << "LuSolve replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, HouseholderProductTracksNewInput) {
    // Composite trace (geqrf -> householder_product) so (input, tau) are
    // always a genuinely valid Householder factorization of whatever A is
    // passed, on every call.
    auto fn = [](const Variable& A) {
        auto [qr_packed, tau] = linalg::geqrf(A.tensor());
        return Variable(linalg::householder_product(qr_packed, tau), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({2, 2}, {2.0f, 1.0f, 1.0f, 3.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_qr, eager_tau] = linalg::geqrf(other_A);
    auto eager = linalg::householder_product(eager_qr, eager_tau);

    EXPECT_GE(compiled.num_cached(), 1u) << "householder_product did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "HouseholderProduct replay on a NEW input != eager on that input "
           "— frozen trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, LdlFactorTracksNewInput) {
    auto fn = [](const Variable& A) {
        auto [LD, pivots] = linalg::ldl_factor(A.tensor());
        (void)pivots;
        return Variable(LD, false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({2, 2}, {4.0f, 2.0f, 2.0f, 3.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({2, 2}, {5.0f, 1.0f, 1.0f, 4.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_LD, eager_pivots] = linalg::ldl_factor(other_A);
    (void)eager_pivots;

    EXPECT_GE(compiled.num_cached(), 1u) << "ldl_factor did not compile/cache";
    EXPECT_TRUE(tensors_close(eager_LD, replayed.tensor(), 1e-4f, 1e-4f))
        << "LdlFactor replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, LdlSolveTracksNewInput) {
    // Composite trace (ldl_factor -> ldl_solve) so LD/pivots are always a
    // genuinely valid factorization of whatever A is passed, on every call.
    auto b = make_mat({2, 1}, {1.0f, 2.0f});
    auto fn = [b](const Variable& A) {
        auto [LD, pivots] = linalg::ldl_factor(A.tensor());
        return Variable(linalg::ldl_solve(LD, pivots, b), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({2, 2}, {4.0f, 2.0f, 2.0f, 3.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({2, 2}, {5.0f, 1.0f, 1.0f, 4.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_LD, eager_pivots] = linalg::ldl_factor(other_A);
    auto eager = linalg::ldl_solve(eager_LD, eager_pivots, b);

    EXPECT_GE(compiled.num_cached(), 1u) << "ldl_solve did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "LdlSolve replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, OrmqrTracksNewInput) {
    // Composite trace (geqrf -> ormqr) so (input, tau) are always a
    // genuinely valid Householder factorization of whatever A is passed.
    auto other = make_mat({2, 2}, {5.0f, 6.0f, 7.0f, 8.0f});
    auto fn = [other](const Variable& A) {
        auto [qr_packed, tau] = linalg::geqrf(A.tensor());
        return Variable(linalg::ormqr(qr_packed, tau, other), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({2, 2}, {2.0f, 1.0f, 1.0f, 3.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_qr, eager_tau] = linalg::geqrf(other_A);
    auto eager = linalg::ormqr(eager_qr, eager_tau, other);

    EXPECT_GE(compiled.num_cached(), 1u) << "ormqr did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "Ormqr replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}

TEST(JitLinalgOps, GeqrfTracksNewInput) {
    auto fn = [](const Variable& A) {
        auto [qr_packed, tau] = linalg::geqrf(A.tensor());
        return Variable(tenzor::add(qr_packed, tenzor::reshape(tau, {2, 1})), false);
    };
    auto compiled = jit::compile(fn);
    auto dummy_A = make_mat({2, 2}, {1.0f, 2.0f, 3.0f, 4.0f});
    (void)compiled(Variable(dummy_A, false));

    auto other_A = make_mat({2, 2}, {2.0f, 1.0f, 1.0f, 3.0f});
    auto replayed = compiled(Variable(other_A, false));
    auto [eager_qr, eager_tau] = linalg::geqrf(other_A);
    auto eager = tenzor::add(eager_qr, tenzor::reshape(eager_tau, {2, 1}));

    EXPECT_GE(compiled.num_cached(), 1u) << "geqrf did not compile/cache";
    EXPECT_TRUE(tensors_close(eager, replayed.tensor(), 1e-4f, 1e-4f))
        << "Geqrf replay on a NEW input != eager on that input — frozen "
           "trace-time constant, not real tracer visibility";
}
