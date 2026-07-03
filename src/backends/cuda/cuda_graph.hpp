/**
 * @file cuda_graph.hpp
 * @brief CUDA Graph capture and replay utility
 *
 * Provides a mechanism to capture a sequence of CUDA kernel launches into
 * a graph that can be replayed efficiently, eliminating per-launch overhead.
 *
 * Constraints:
 * - Tensor shapes must be fixed during capture (no dynamic shapes)
 * - No host-device synchronization during capture
 * - No cudaMalloc/cudaFree during capture (use caching allocator)
 */

#pragma once

#include <cuda_runtime.h>
#include <memory>
#include <stdexcept>
#include <string>

namespace tenzor {
namespace cuda {

/**
 * @brief RAII wrapper for CUDA Graph capture and replay.
 *
 * Usage:
 * @code
 * CUDAGraphCapture graph;
 * graph.begin_capture(stream);
 * // ... launch kernels on stream ...
 * graph.end_capture();
 *
 * // Replay captured work (much faster than re-launching)
 * graph.replay(stream);
 * @endcode
 */
class CUDAGraphCapture {
public:
    CUDAGraphCapture() = default;

    ~CUDAGraphCapture() {
        if (exec_) {
            cudaGraphExecDestroy(exec_);
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
        }
    }

    // Non-copyable
    CUDAGraphCapture(const CUDAGraphCapture&) = delete;
    CUDAGraphCapture& operator=(const CUDAGraphCapture&) = delete;

    // Movable
    CUDAGraphCapture(CUDAGraphCapture&& other) noexcept
        : graph_(other.graph_), exec_(other.exec_),
          stream_(other.stream_), capturing_(other.capturing_) {
        other.graph_ = nullptr;
        other.exec_ = nullptr;
        other.stream_ = nullptr;
        other.capturing_ = false;
    }

    CUDAGraphCapture& operator=(CUDAGraphCapture&& other) noexcept {
        if (this != &other) {
            if (exec_) cudaGraphExecDestroy(exec_);
            if (graph_) cudaGraphDestroy(graph_);

            graph_ = other.graph_;
            exec_ = other.exec_;
            stream_ = other.stream_;
            capturing_ = other.capturing_;

            other.graph_ = nullptr;
            other.exec_ = nullptr;
            other.stream_ = nullptr;
            other.capturing_ = false;
        }
        return *this;
    }

    /**
     * @brief Begin capturing CUDA operations on the given stream.
     *
     * All CUDA operations submitted to the stream after this call will be
     * captured into a graph instead of being executed immediately.
     *
     * @param stream The CUDA stream to capture on
     * @throws std::runtime_error if already capturing or if capture fails
     */
    void begin_capture(cudaStream_t stream) {
        if (capturing_) {
            throw std::runtime_error("CUDAGraphCapture: already capturing");
        }

        // Destroy previous graph/exec if any
        if (exec_) {
            cudaGraphExecDestroy(exec_);
            exec_ = nullptr;
        }
        if (graph_) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
        }

        stream_ = stream;
        auto err = cudaStreamBeginCapture(stream_, cudaStreamCaptureModeGlobal);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDAGraphCapture: begin_capture failed: ") +
                cudaGetErrorString(err));
        }
        capturing_ = true;
    }

    /**
     * @brief End capturing and instantiate the executable graph.
     *
     * @throws std::runtime_error if not capturing or if instantiation fails
     */
    void end_capture() {
        if (!capturing_) {
            throw std::runtime_error("CUDAGraphCapture: not capturing");
        }

        auto err = cudaStreamEndCapture(stream_, &graph_);
        capturing_ = false;

        if (err != cudaSuccess || graph_ == nullptr) {
            throw std::runtime_error(
                std::string("CUDAGraphCapture: end_capture failed: ") +
                cudaGetErrorString(err));
        }

        err = cudaGraphInstantiate(&exec_, graph_, nullptr, nullptr, 0);
        if (err != cudaSuccess) {
            cudaGraphDestroy(graph_);
            graph_ = nullptr;
            throw std::runtime_error(
                std::string("CUDAGraphCapture: instantiation failed: ") +
                cudaGetErrorString(err));
        }
    }

    /**
     * @brief Replay the captured graph on the given stream.
     *
     * @param stream The stream to launch the graph on
     * @throws std::runtime_error if no graph has been captured
     */
    void replay(cudaStream_t stream) const {
        if (!exec_) {
            throw std::runtime_error("CUDAGraphCapture: no graph captured");
        }

        auto err = cudaGraphLaunch(exec_, stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDAGraphCapture: replay failed: ") +
                cudaGetErrorString(err));
        }
        // cudaGraphLaunch is ASYNCHRONOUS on `stream`. The caller reads the
        // captured output tensors immediately after replay() returns (and its
        // device->host copy runs on a DIFFERENT stream), so without this sync the
        // readback can race the still-running graph and observe torn/stale data.
        // Synchronize the launch stream before returning so replay is complete.
        err = cudaStreamSynchronize(stream);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDAGraphCapture: replay synchronize failed: ") +
                cudaGetErrorString(err));
        }
    }

    /**
     * @brief Check if a graph has been captured and is ready for replay.
     */
    bool is_ready() const { return exec_ != nullptr; }

    /**
     * @brief Check if currently in capture mode.
     */
    bool is_capturing() const { return capturing_; }

private:
    cudaGraph_t graph_ = nullptr;
    cudaGraphExec_t exec_ = nullptr;
    cudaStream_t stream_ = nullptr;
    bool capturing_ = false;
};

} // namespace cuda
} // namespace tenzor
