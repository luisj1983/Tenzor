/**
 * @file test_cuda_graph_replay.cpp
 * @brief Regression test for the CUDA-graph replay-over-stale-buffers bug.
 *
 * A captured CUDA/HIP graph re-runs verbatim over the device pointers it
 * recorded at capture time. Before the fix, replay_cuda_graph() validated only
 * input SHAPES and never copied the fresh input data into the captured buffers,
 * so replaying with a CHANGED input silently returned the stale capture-time
 * result. The fix copies each fresh input into the captured buffer
 * (device-to-device) before replay.
 *
 * This test captures `out = x + x`, replays with a different input, and asserts
 * the output reflects the NEW input (i.e. 2 * new_input), not the stale one.
 *
 * Runs only when a CUDA device is available; otherwise it SKIPs (this is a
 * GPU-path regression — there is no CPU fallback for CUDA-graph capture).
 */

#include <gtest/gtest.h>

#include <memory>
#include <optional>
#include <vector>

#include <tenzor/autograd/ops.hpp>
#include <tenzor/backend/cuda_graph.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/control_flow.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/nn/functional.hpp>
#include <tenzor/tenzor.hpp>

#include "../backend_parity/golden_util.hpp"
#include "../backend_parity/parity_test_utils.hpp"

using namespace tenzor;

namespace {
// CUDA-graph capture works on either a CUDA or a ROCm (HIP graph) device — the
// path is identical (routed through the CUDAGraph interface + backend-registered
// factory). Return EVERY available GPU so the regression is exercised on each.
std::vector<Device> available_gpus() {
    std::vector<Device> gpus;
    for (auto type : {Device::Type::CUDA, Device::Type::ROCm}) {
        // JIT-R181: honor TENZOR_SKIP_BACKENDS -- a dev who explicitly opts
        // out of a known-flaky driver on this host must have that respected
        // here too, not just by TENZOR_REQUIRE_MULTI_BACKEND's escalation.
        if (::tenzor::testing::is_backend_skipped_by_env(
                ::tenzor::testing::device_type_to_backend_name(type))) {
            continue;
        }
        auto* b = backend_registry().get_backend(type);
        if (b == nullptr || !b->is_available()) continue;
        // Only include devices whose backend registered a graph-capture factory
        // (CUDA and ROCm both do). create_for() returns null on a build/stack
        // without graph support, in which case the JIT falls back to a normal
        // forward — we must not assert capture on a device that does not support
        // it (a hardware/driver capability gate, not a skipped Tenzor code path).
        if (CUDAGraph::create_for(static_cast<int>(type), 0) != nullptr) {
            gpus.push_back(Device{type, 0});
        }
    }
    return gpus;
}
}  // namespace

class CudaGraphReplay : public ::testing::Test {
protected:
    std::vector<Device> gpus_;

    void SetUp() override {
        tenzor::initialize();
        gpus_ = available_gpus();
        if (gpus_.empty()) {
            // JIT-R129: a bare GTEST_SKIP() here would silently hide a CI box
            // that's supposed to have a CUDA/ROCm GPU but where the driver
            // failed to init -- honor the same TENZOR_REQUIRE_MULTI_BACKEND
            // hard-fail escalation the other GPU-gated suites use (e.g.
            // test_jit_backend_parity.cpp's NvrtcVsMlirDirectParity_NormOpsF16).
            if (::tenzor::testing::golden::require_multi_backend()) {
                FAIL() << "Multi-backend required (TENZOR_REQUIRE_MULTI_BACKEND=1) "
                          "but no CUDA/ROCm device is available; CUDA-graph "
                          "capture is a GPU-only path (no CPU fallback).";
            }
            GTEST_SKIP() << "No CUDA/ROCm device available; graph capture is a "
                            "GPU-only path (no CPU fallback).";
        }
    }
};

TEST_F(CudaGraphReplay, ReplayWithChangedInputsReturnsFreshResult) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    // Build a tiny executable graph: out = x + x, traced on CUDA.
    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {args[0] + args[0]};
    };

    Tensor x0 = full({4}, 1.0f, DType::Float32, cuda);  // capture-time input
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};

    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);

    auto module = std::make_shared<jit::CompiledModule>(graph);

    // Capture with x0 (all ones) -> captured output should be 2.0 everywhere.
    // The graph factory is routed through the backend registry (CUDA backend
    // registers a CUDA-graph factory, ROCm a HIP-graph factory), and the
    // backend's thread-local current-stream makes the forward pass record onto
    // the capture stream — so capture actually contains the work.
    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph());

    // Sanity: capture-time output is 2 * 1 = 2.
    {
        auto outs = module->replay_cuda_graph_outputs();
        ASSERT_EQ(outs.size(), 1u);
        Tensor host = outs[0].to(Device::cpu());
        const float* p = host.data<float>();
        for (int64_t i = 0; i < host.numel(); ++i) {
            EXPECT_FLOAT_EQ(p[i], 2.0f) << "capture-time output at " << i;
        }
    }

    // Replay with a DIFFERENT input (all 5s). Pre-fix this returned the stale
    // 2.0; post-fix the fresh input is copied into the captured buffer so the
    // output must be 2 * 5 = 10.
    Tensor x1 = full({4}, 5.0f, DType::Float32, cuda);
    std::vector<Tensor> replay_inputs = {x1};
    ASSERT_TRUE(module->replay_cuda_graph(replay_inputs));

    auto outs = module->replay_cuda_graph_outputs();
    ASSERT_EQ(outs.size(), 1u);
    Tensor host = outs[0].to(Device::cpu());
    const float* p = host.data<float>();
    for (int64_t i = 0; i < host.numel(); ++i) {
        EXPECT_FLOAT_EQ(p[i], 10.0f)
            << "replay output at " << i
            << " should reflect the NEW input (2*5), not the stale capture (2*1)";
    }

    // Replay again with yet another input (all 3s) -> 6.0, proving copy-in is
    // not a one-shot.
    Tensor x2 = full({4}, 3.0f, DType::Float32, cuda);
    std::vector<Tensor> replay_inputs2 = {x2};
    ASSERT_TRUE(module->replay_cuda_graph(replay_inputs2));
    Tensor host2 = module->replay_cuda_graph_outputs()[0].to(Device::cpu());
    const float* p2 = host2.data<float>();
    for (int64_t i = 0; i < host2.numel(); ++i) {
        EXPECT_FLOAT_EQ(p2[i], 6.0f) << "second replay at " << i;
    }
  }  // for each gpu
}

// capture_cuda_graph retains PRIVATE (cloned) input buffers. replay copies fresh
// data into those buffers device-to-device; if they aliased the caller's
// capture-time tensor (contiguous() returns the same tensor when already
// contiguous), replay would silently overwrite it. Assert the caller's
// capture-time input is untouched after a replay with different data.
TEST_F(CudaGraphReplay, ReplayDoesNotClobberCallerCaptureInput) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {args[0] + args[0]};
    };

    Tensor x0 = full({4}, 1.0f, DType::Float32, cuda);  // caller keeps this
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};
    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);
    auto module = std::make_shared<jit::CompiledModule>(graph);

    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph());

    // Replay with all-5s. Pre-fix (aliased buffer) this device-to-device copy
    // would overwrite x0's storage; post-fix (private clone) x0 stays all-1s.
    Tensor x1 = full({4}, 5.0f, DType::Float32, cuda);
    std::vector<Tensor> replay_inputs = {x1};
    ASSERT_TRUE(module->replay_cuda_graph(replay_inputs));

    Tensor x0_host = x0.to(Device::cpu());
    const float* p = x0_host.data<float>();
    for (int64_t i = 0; i < x0_host.numel(); ++i) {
        EXPECT_FLOAT_EQ(p[i], 1.0f)
            << "caller's capture-time input was clobbered by replay at " << i;
    }
  }  // for each gpu
}

// R3-01: capture_cuda_graph() must pin every INTERMEDIATE node result's device
// buffer for the lifetime of the captured graph, not just its declared inputs
// and outputs. `out = x + x` (the two tests above) has no intermediate at all
// (the output IS the traced node's output), so it cannot exercise this gap.
// Use a genuine 2-op graph (`t = x @ w`, `out = relu(t)`) so `t`'s buffer is a
// true intermediate: allocated during the capture-time forward() pass, never
// exposed via captured_inputs_/captured_outputs_, and — before the fix — freed
// back to the shared CachingAllocator's free list the instant that forward()
// call returns, even though the just-captured graph's kernel-launch nodes have
// that exact device address permanently baked in.
//
// This test does not need to peek at CompiledModule internals: it forces the
// bug to become externally observable. Immediately after capture returns, it
// allocates a same-size "canary" tensor and keeps it ALIVE (unlike the
// warm-up/capture-time intermediate, which is scoped to have already been
// freed by this point). If the intermediate's address was not pinned, the
// CachingAllocator's free list very likely hands that exact freed address to
// this same-size canary allocation (same size, freed at the top of the free
// list). The canary is filled with a sentinel value the captured graph would
// never write. Replaying the graph then physically re-executes the matmul
// kernel that writes into the (baked-in) intermediate address — if the canary
// aliases it, that write silently corrupts the canary's sentinel data
// underneath the caller, who never touched the graph or the canary.
TEST_F(CudaGraphReplay, IntermediateBufferSurvivesUnrelatedAllocationAfterCapture) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    constexpr int64_t N = 256;
    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        Variable t = matmul(args[0], args[1]);  // intermediate buffer
        return {nn::functional::relu(t)};
    };

    Tensor x0 = full({N, N}, 1.0f, DType::Float32, cuda);
    Tensor w0 = full({N, N}, 1.0f, DType::Float32, cuda);
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false),
                                          Variable(w0, /*requires_grad=*/false)};

    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);
    ASSERT_GE(graph->nodes().size(), 2u)
        << "need a genuine multi-node graph with an intermediate buffer";

    auto module = std::make_shared<jit::CompiledModule>(graph);
    module->capture_cuda_graph({x0, w0});
    ASSERT_TRUE(module->has_cuda_graph());

    // Sanity: capture-time output is relu(1s[NxN] @ 1s[NxN]) = N everywhere.
    {
        auto outs = module->replay_cuda_graph_outputs();
        ASSERT_EQ(outs.size(), 1u);
        Tensor host = outs[0].to(Device::cpu());
        EXPECT_FLOAT_EQ(host.data<float>()[0], static_cast<float>(N));
    }

    // Plant the canary right after capture returns -- same size as the
    // intermediate (x0 @ w0 is NxN), same device, so it is the most likely
    // candidate to receive the intermediate's just-freed address if it was
    // not pinned.
    constexpr float kSentinel = -12345.0f;
    Tensor canary = full({N, N}, kSentinel, DType::Float32, cuda);

    // Replay with a fresh input -- this physically re-executes the captured
    // matmul kernel, writing into whatever address it recorded for `t`.
    Tensor x1 = full({N, N}, 2.0f, DType::Float32, cuda);
    Tensor w1 = full({N, N}, 2.0f, DType::Float32, cuda);
    std::vector<Tensor> replay_inputs = {x1, w1};
    ASSERT_TRUE(module->replay_cuda_graph(replay_inputs));

    // The captured graph's own result must still be correct...
    auto outs = module->replay_cuda_graph_outputs();
    ASSERT_EQ(outs.size(), 1u);
    Tensor host = outs[0].to(Device::cpu());
    EXPECT_FLOAT_EQ(host.data<float>()[0], static_cast<float>(4 * N))
        << "replay output should reflect the new inputs: relu(2s @ 2s) = 4*N";

    // ...and the unrelated canary, which the caller never passed to the graph
    // in any way, must be untouched. A corrupted canary means the replay's
    // intermediate write landed on memory the caller believes it privately
    // owns -- exactly the silent-corruption scenario R3-01 describes.
    Tensor canary_host = canary.to(Device::cpu());
    const float* cp = canary_host.data<float>();
    for (int64_t i = 0; i < canary_host.numel(); ++i) {
        ASSERT_FLOAT_EQ(cp[i], kSentinel)
            << "canary tensor at " << i << " was corrupted by graph replay -- "
               "an intermediate buffer's freed address was handed to an "
               "unrelated allocation while the captured graph still "
               "referenced it";
    }
  }  // for each gpu
}

// JIT-R116: OpId::LayerNorm's CUDA kernel returns {output, mean, inv_std} --
// mean/inv_std are written by the SAME kernel launch that writes output
// (needed by fused_layer_norm_cuda's single-pass algorithm), but execute_node
// only exposes `output` as this node's declared Graph output (mean/inv_std
// have no Value id in an inference-only replay). Before the fix, those two
// extra Tensors were local C++ temporaries with no other reference once
// execute_node's `result` vector went out of scope -- freed back to the
// CachingAllocator's pool immediately after capture, even though the SAME
// captured kernel launch keeps writing into their exact address on every
// future replay. This is the identical corruption shape as
// IntermediateBufferSurvivesUnrelatedAllocationAfterCapture above, just for a
// dispatch result that never became a tracked Graph Variable at all.
TEST_F(CudaGraphReplay, LayerNormSavedStatsScratchBufferDoesNotCorruptUnrelatedAllocation) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    constexpr int64_t kBatch = 8;
    constexpr int64_t kFeat = 64;
    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {nn::functional::layer_norm(args[0], {kFeat})};
    };

    Tensor x0 = full({kBatch, kFeat}, 1.0f, DType::Float32, cuda);
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};

    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);
    bool has_layer_norm_node = false;
    for (const auto& node : graph->nodes()) {
        if (node->op_type() == jit::OpType::LayerNorm) has_layer_norm_node = true;
    }
    ASSERT_TRUE(has_layer_norm_node)
        << "test requires nn::functional::layer_norm to trace as a single "
           "OpType::LayerNorm node (the path that discards mean/inv_std)";

    auto module = std::make_shared<jit::CompiledModule>(graph);
    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph());

    // Plant a canary matching mean/inv_std's shape+dtype+device ({kBatch},
    // Float32) right after capture returns -- the most likely candidate to
    // receive the just-freed scratch buffer's address if it was not pinned.
    constexpr float kSentinel = -98765.0f;
    Tensor canary = full({kBatch}, kSentinel, DType::Float32, cuda);

    // Replay with a fresh input -- this physically re-executes the captured
    // LayerNorm kernel, which unconditionally rewrites mean/inv_std at
    // whatever address it recorded for them at capture time.
    Tensor x1 = full({kBatch, kFeat}, 5.0f, DType::Float32, cuda);
    std::vector<Tensor> replay_inputs = {x1};
    ASSERT_TRUE(module->replay_cuda_graph(replay_inputs));

    auto outs = module->replay_cuda_graph_outputs();
    ASSERT_EQ(outs.size(), 1u);

    // The unrelated canary, which the caller never passed to the graph in
    // any way, must be untouched. A corrupted canary means the replay's
    // discarded-scratch-buffer write landed on memory the caller believes
    // it privately owns -- exactly the silent-corruption scenario JIT-R116
    // describes.
    Tensor canary_host = canary.to(Device::cpu());
    const float* cp = canary_host.data<float>();
    for (int64_t i = 0; i < canary_host.numel(); ++i) {
        ASSERT_FLOAT_EQ(cp[i], kSentinel)
            << "canary tensor at " << i << " was corrupted by graph replay -- "
               "LayerNorm's discarded mean/inv_std scratch buffer's freed "
               "address was handed to an unrelated allocation while the "
               "captured graph still wrote into it every replay";
    }
  }  // for each gpu
}

// JIT-R176: investigating whether a nested If/Loop branch's own scratch
// tensors were pinned into an outer CUDA-graph capture surfaced a MORE
// fundamental, previously-unknown gap on real hardware: evaluating an
// If/Loop node's condition (tensor_condition_to_bool) does a synchronous
// device-to-host copy, which CUDA/HIP forbid on a stream that is currently
// being captured ("operation not permitted when stream is capturing") --
// so a graph containing control flow could never be captured successfully
// at all, regardless of scratch-buffer pinning. capture_cuda_graph() now
// detects this up front (Graph::has_control_flow_node()) and refuses with
// a clear, actionable error instead of letting that raw low-level CUDA/HIP
// error surface partway through capture. The out_all_values threading fix
// (Graph::forward's grad_mode/out_all_values now propagate into nested
// If/Loop branch calls) remains in place as correct defense-in-depth for
// whatever DOES reach a nested branch's forward() during capture (e.g. a
// future capture-safe conditional mechanism), even though today's guard
// means the scratch-buffer-corruption scenario itself is unreachable via
// capture_cuda_graph() specifically.
TEST_F(CudaGraphReplay, CaptureRefusesGraphWithControlFlow_JIT176) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    constexpr int64_t kBatch = 8;
    constexpr int64_t kFeat = 64;
    // Both branches call layer_norm identically -- only the condition's
    // truthiness matters for which branch is taken; what matters for this
    // test is that the LayerNorm node ends up inside the If's nested
    // subgraph rather than the top-level graph.
    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        Tensor true_cond = ones({1}, DType::Float32, args[0].tensor().device());
        return jit::cond(
            true_cond,
            [](const std::vector<Variable>& ins) -> std::vector<Variable> {
                return {nn::functional::layer_norm(ins[0], {kFeat})};
            },
            [](const std::vector<Variable>& ins) -> std::vector<Variable> {
                return {nn::functional::layer_norm(ins[0], {kFeat})};
            },
            args);
    };

    Tensor x0 = full({kBatch, kFeat}, 1.0f, DType::Float32, cuda);
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};

    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);
    ASSERT_TRUE(graph->has_control_flow_node())
        << "test setup error: graph must contain an If node";

    // Confirm the LayerNorm node genuinely ended up INSIDE the If's nested
    // branch subgraph, not the top-level graph -- establishes this is a
    // genuinely nested scenario, not just a top-level If with no payload.
    bool has_nested_layer_norm = false;
    for (const auto& node : graph->nodes()) {
        if (node->op_type() == jit::OpType::If) {
            auto check_branch = [&](const std::shared_ptr<jit::Graph>& b) {
                if (!b) return;
                for (const auto& n : b->nodes()) {
                    if (n->op_type() == jit::OpType::LayerNorm) has_nested_layer_norm = true;
                }
            };
            check_branch(node->then_branch());
            check_branch(node->else_branch());
        }
    }
    ASSERT_TRUE(has_nested_layer_norm)
        << "test requires the LayerNorm node to end up inside the If "
           "branch's nested subgraph";

    auto module = std::make_shared<jit::CompiledModule>(graph);
    EXPECT_THROW(module->capture_cuda_graph({x0}), std::runtime_error)
        << "capturing a graph containing an If/Loop node must fail with a "
           "clear error (JIT-R176), not a raw CUDA/HIP 'stream is "
           "capturing' error or silent scratch-buffer corruption";
    EXPECT_FALSE(module->has_cuda_graph())
        << "a refused capture must not leave has_cuda_graph() reporting true";
  }  // for each gpu
}

// R3-02: CompiledModule's internal retrace paths (device/dtype-mismatch and
// ShapeGuard-triggered, both in forward()) reassign graph_ to a freshly
// retraced replacement without invalidating any CUDA/HIP graph already
// captured against the PRE-retrace graph_. Before the fix, has_cuda_graph()
// kept reporting true and replay_cuda_graph() kept replaying the STALE
// graph after an internal retrace -- silently wrong for any direct
// CompiledModule API user who captures, then calls forward() with an input
// whose shape/device/dtype differs enough to trip the internal retrace
// (jit()/CompiledFunction::operator() masks this in the common case since
// its OWN outer shape+dtype+device cache key usually routes around a
// mismatched CompiledModule first).
//
// This exercises the device/dtype-mismatch retrace path specifically (a
// pure shape change, since compute_shape_key() folds shape in too and
// retraces on ANY change to shape/device/dtype -- see forward()'s own
// comment): trace a real nn::Module (needed for source_module_ to be set,
// the precondition for internal retrace being reachable at all), capture a
// graph for one batch size, then call forward() with a DIFFERENT batch
// size on the same device/dtype. has_cuda_graph() must go from true to
// false -- the stale capture must not survive the retrace.
TEST_F(CudaGraphReplay, InternalRetraceInvalidatesStaleCudaGraph) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    auto lin = std::make_shared<nn::Linear>(4, 4);
    lin->to(cuda);

    Tensor x0 = full({2, 4}, 1.0f, DType::Float32, cuda);
    Variable x0_var(x0, /*requires_grad=*/false);

    auto module = jit::CompiledModule::trace(lin, x0_var);
    ASSERT_NE(module, nullptr);

    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph())
        << "capture must succeed before the retrace this test exercises";

    // Different batch size -> compute_shape_key() differs from
    // traced_shape_key_ -> device/dtype-mismatch retrace path fires
    // (source_module_ is set, so it can actually retrace rather than throw).
    Tensor x1 = full({5, 4}, 1.0f, DType::Float32, cuda);
    Variable x1_var(x1, /*requires_grad=*/false);
    ASSERT_NO_THROW({ (void)module->forward(x1_var); })
        << "internal retrace to a new batch size must succeed";

    EXPECT_FALSE(module->has_cuda_graph())
        << "a CUDA/HIP graph captured against the pre-retrace graph_ must "
           "not survive an internal retrace that swaps graph_ out -- "
           "replaying it would silently run stale/mismatched computation";
  }  // for each gpu
}

// findings.txt JIT-R126: unlike the retrace paths covered by
// InternalRetraceInvalidatesStaleCudaGraph above (which swap the graph_
// POINTER, triggering forward()'s own invalidate_cuda_graph() calls),
// optimize_for_inference() and mark_dynamic_dims() mutate *graph_ IN PLACE
// -- no pointer reassignment. Before the fix, has_cuda_graph() stayed true
// after either call, so replay_cuda_graph() would keep replaying the
// PRE-mutation kernel sequence forever while forward() ran the
// newly-mutated (non-numerically-neutral, for optimize_for_inference's
// fusion) graph -- a silent divergence between the two APIs on one module.
TEST_F(CudaGraphReplay, OptimizeForInferenceInvalidatesStaleCudaGraph) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {args[0] + args[0]};
    };
    Tensor x0 = full({4}, 1.0f, DType::Float32, cuda);
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};
    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);

    auto module = std::make_shared<jit::CompiledModule>(graph);
    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph())
        << "capture must succeed before the mutation this test exercises";

    module->optimize_for_inference();

    EXPECT_FALSE(module->has_cuda_graph())
        << "a CUDA/HIP graph captured before optimize_for_inference() must "
           "not survive it -- optimize_for_inference() mutates *graph_ in "
           "place (no pointer swap), so replaying the stale capture would "
           "silently diverge from forward()'s now-optimized graph";
  }  // for each gpu
}

TEST_F(CudaGraphReplay, MarkDynamicDimsInvalidatesStaleCudaGraph) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {args[0] + args[0]};
    };
    Tensor x0 = full({4}, 1.0f, DType::Float32, cuda);
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};
    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);

    auto module = std::make_shared<jit::CompiledModule>(graph);
    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph())
        << "capture must succeed before the mutation this test exercises";

    module->mark_dynamic_dims({jit::CompiledModule::DynamicDimSpec{0, 0, "batch"}});

    EXPECT_FALSE(module->has_cuda_graph())
        << "a CUDA/HIP graph captured before mark_dynamic_dims() must not "
           "survive it -- mark_dynamic_dims() mutates *graph_'s Value shape "
           "metadata in place (no pointer swap)";
  }  // for each gpu
}

// JIT-R122 regression: unlike optimize_for_inference()/mark_dynamic_dims()
// (which mutate *graph_ in place) set_graph() does a full POINTER swap of
// graph_ -- but a captured CUDA/HIP graph replays the OLD graph's physical
// device buffers/kernel-launch geometry regardless of which mutation style
// produced the staleness. Before the fix, set_graph() was the one
// graph_-reassigning entry point with no invalidate_cuda_graph() call at
// all, so has_cuda_graph() stayed true and replay_cuda_graph() would keep
// replaying the PRE-set_graph() kernel sequence against buffers sized/
// shaped for a completely different graph.
TEST_F(CudaGraphReplay, SetGraphInvalidatesStaleCudaGraph) {
  for (const Device& cuda : gpus_) {
    SCOPED_TRACE("device: " + cuda.to_string());

    auto closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {args[0] + args[0]};
    };
    Tensor x0 = full({4}, 1.0f, DType::Float32, cuda);
    std::vector<Variable> trace_inputs = {Variable(x0, /*requires_grad=*/false)};
    std::shared_ptr<jit::Graph> graph = jit::trace(closure, trace_inputs);
    ASSERT_NE(graph, nullptr);

    auto module = std::make_shared<jit::CompiledModule>(graph);
    module->capture_cuda_graph({x0});
    ASSERT_TRUE(module->has_cuda_graph())
        << "capture must succeed before the mutation this test exercises";

    auto other_closure = [](const std::vector<Variable>& args) -> std::vector<Variable> {
        return {args[0] * args[0]};
    };
    std::shared_ptr<jit::Graph> other_graph = jit::trace(other_closure, trace_inputs);
    ASSERT_NE(other_graph, nullptr);
    module->set_graph(other_graph);

    EXPECT_FALSE(module->has_cuda_graph())
        << "a CUDA/HIP graph captured before set_graph() must not survive "
           "it -- set_graph() pointer-swaps graph_ to an entirely different "
           "graph, so replaying the stale capture would run the OLD "
           "graph's kernels/buffers while forward() now runs the NEW one";
  }  // for each gpu
}
