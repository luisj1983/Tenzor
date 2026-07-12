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
     * @brief Route dispatch onto this graph's private capture stream WITHOUT
     * starting actual stream capture.
     *
     * Capture callers run an uncaptured "warmup" forward pass first (to
     * pre-populate the caching allocator so the real captured pass never
     * needs a driver allocation, which CUDA/HIP forbid mid-capture). If that
     * warmup runs on the default/legacy stream while begin_capture() later
     * routes the real pass onto a dedicated capture stream, every buffer the
     * warmup allocated is tagged with a DIFFERENT stream than the one that
     * will reuse it from cache during actual capture — a cross-stream reuse,
     * which requires the allocator to insert a stream-to-stream wait that is
     * illegal while the target stream is actively capturing ("operation
     * would make the legacy stream depend on a capturing blocking stream").
     * Calling this before the warmup pass makes warmup allocations land on
     * the SAME stream capture will use, so the real pass's reuse is
     * same-stream (always safely ordered, no cross-stream wait needed).
     */
    virtual void prepare_capture_stream() = 0;

    /**
     * @brief Create a CUDA graph capture object.
     * @param device_id CUDA device index (default: 0)
     * @return Unique pointer to graph object, or nullptr if CUDA unavailable
     */
    static auto create(int32_t device_id = 0) -> std::unique_ptr<CUDAGraph>;

    // ── Backend-registered factory routing ──────────────────────────────────
    // Backends are dlopen'd with RTLD_LOCAL, so the real create() in the backend
    // .so cannot interpose the weak stub symbol linked into tenzor_core. Instead
    // each GPU backend registers a factory for its device type when it loads, and
    // the JIT calls create_for(device_type, id) to obtain the REAL implementation
    // (HIP graph for ROCm, CUDA graph for CUDA). device_type values match
    // Device::Type (passed as int to avoid a header dependency here).
    using Factory = std::unique_ptr<CUDAGraph> (*)(int32_t device_id);
    static void register_factory(int device_type, Factory factory);
    static auto create_for(int device_type, int32_t device_id)
        -> std::unique_ptr<CUDAGraph>;
};

} // namespace tenzor
