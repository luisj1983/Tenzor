/**
 * @file cuda_graph.hpp
 * @brief Public API for CUDA Graph capture and replay
 *
 * Provides a backend-agnostic interface for graph capture. The actual
 * implementation lives in the CUDA backend plugin.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>

namespace tenzor {

/**
 * @brief Opaque handle to a captured CUDA graph.
 *
 * Usage:
 * @code
 * auto graph = CUDAGraph::create(device_id);
 * graph->begin_capture();
 * // ... run operations ...
 * graph->end_capture();
 * graph->replay();  // fast replay
 * @endcode
 */
class CUDAGraph {
public:
    virtual ~CUDAGraph() = default;

    virtual void begin_capture() = 0;
    virtual void end_capture() = 0;
    virtual void replay() = 0;
    virtual bool is_ready() const = 0;

    /**
     * @brief Create a CUDA graph capture object.
     * @param device_id CUDA device index (default: 0)
     * @return Unique pointer to graph object, or nullptr if CUDA unavailable
     */
    static auto create(int32_t device_id = 0) -> std::unique_ptr<CUDAGraph>;
};

} // namespace tenzor
