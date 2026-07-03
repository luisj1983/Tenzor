/**
 * @file hip_graph.hpp
 * @brief HIP Graph capture and replay for AMD GPUs
 *
 * Provides RAII-based graph capture using hipGraph_t / hipGraphExec_t.
 * Mirrors the CUDA graph capture pattern for kernel replay optimization.
 *
 * Constraints (same as CUDA graphs):
 * - Tensor shapes must be fixed during capture (no dynamic shapes)
 * - No host-device synchronization during capture
 * - No hipMalloc/hipFree during capture (use caching allocator)
 */

#pragma once

#include <hip/hip_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace rocm {

class HIPGraph {
public:
    HIPGraph() = default;

    ~HIPGraph() {
        cleanup();
    }

    // Non-copyable
    HIPGraph(const HIPGraph&) = delete;
    HIPGraph& operator=(const HIPGraph&) = delete;

    // Movable
    HIPGraph(HIPGraph&& other) noexcept
        : graph_(other.graph_), graph_exec_(other.graph_exec_),
          capturing_(other.capturing_) {
        other.graph_ = nullptr;
        other.graph_exec_ = nullptr;
        other.capturing_ = false;
    }

    HIPGraph& operator=(HIPGraph&& other) noexcept {
        if (this != &other) {
            cleanup();
            graph_ = other.graph_;
            graph_exec_ = other.graph_exec_;
            capturing_ = other.capturing_;
            other.graph_ = nullptr;
            other.graph_exec_ = nullptr;
            other.capturing_ = false;
        }
        return *this;
    }

    /// Begin capturing operations on the given stream
    void begin_capture(hipStream_t stream) {
        if (capturing_) {
            throw std::runtime_error("HIPGraph: already capturing");
        }

        // Destroy previous graph/exec if any
        cleanup();

        auto err = hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("HIPGraph: begin_capture failed: ") +
                hipGetErrorString(err));
        }
        capturing_ = true;
        capture_stream_ = stream;
    }

    /// End capture and compile the graph
    void end_capture(hipStream_t stream) {
        if (!capturing_) {
            throw std::runtime_error("HIPGraph: not capturing");
        }

        auto err = hipStreamEndCapture(stream, &graph_);
        capturing_ = false;
        capture_stream_ = nullptr;

        if (err != hipSuccess || graph_ == nullptr) {
            throw std::runtime_error(
                std::string("HIPGraph: end_capture failed: ") +
                hipGetErrorString(err));
        }

        err = hipGraphInstantiate(&graph_exec_, graph_, nullptr, nullptr, 0);
        if (err != hipSuccess) {
            (void)hipGraphDestroy(graph_);
            graph_ = nullptr;
            throw std::runtime_error(
                std::string("HIPGraph: graph instantiation failed: ") +
                hipGetErrorString(err));
        }
    }

    /// Replay the captured graph on the given stream
    void replay(hipStream_t stream) {
        if (!graph_exec_) {
            throw std::runtime_error("HIPGraph: no graph captured for replay");
        }

        auto err = hipGraphLaunch(graph_exec_, stream);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("HIPGraph: replay failed: ") +
                hipGetErrorString(err));
        }
        // hipGraphLaunch is ASYNCHRONOUS on `stream`. The caller reads the
        // captured output tensors immediately after replay() returns (its
        // device->host copy runs on a different stream), so without this sync the
        // readback can race the still-running graph and observe torn/stale data.
        err = hipStreamSynchronize(stream);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("HIPGraph: replay synchronize failed: ") +
                hipGetErrorString(err));
        }
    }

    /// Check if graph has been captured and compiled
    bool is_compiled() const { return graph_exec_ != nullptr; }

    /// Check if currently in capture mode
    bool is_capturing() const { return capturing_; }

private:
    hipGraph_t graph_ = nullptr;
    hipGraphExec_t graph_exec_ = nullptr;
    hipStream_t capture_stream_ = nullptr;
    bool capturing_ = false;

    void cleanup() {
        if (graph_exec_) {
            (void)hipGraphExecDestroy(graph_exec_);
            graph_exec_ = nullptr;
        }
        if (graph_) {
            (void)hipGraphDestroy(graph_);
            graph_ = nullptr;
        }
    }
};

} // namespace rocm
} // namespace tenzor
