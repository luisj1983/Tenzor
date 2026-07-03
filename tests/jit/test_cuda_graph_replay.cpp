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

#include <tenzor/backend/cuda_graph.hpp>
#include <tenzor/backend/loader.hpp>
#include <tenzor/jit/compiler.hpp>
#include <tenzor/jit/tracer.hpp>
#include <tenzor/tenzor.hpp>

using namespace tenzor;

namespace {
// CUDA-graph capture works on either a CUDA or a ROCm (HIP graph) device — the
// path is identical (routed through the CUDAGraph interface + backend-registered
// factory). Return EVERY available GPU so the regression is exercised on each.
std::vector<Device> available_gpus() {
    std::vector<Device> gpus;
    for (auto type : {Device::Type::CUDA, Device::Type::ROCm}) {
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
