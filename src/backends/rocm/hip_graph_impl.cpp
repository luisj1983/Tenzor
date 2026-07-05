/**
 * @file hip_graph_impl.cpp
 * @brief CUDAGraph-interface implementation backed by a HIP graph, plus the
 *        backend-registered factory that lets the JIT reach it across the
 *        RTLD_LOCAL boundary.
 *
 * The JIT (in tenzor_core) drives capture through the abstract CUDAGraph
 * interface. Because backends are dlopen'd RTLD_LOCAL, tenzor_core cannot link
 * the real implementation directly, so this TU registers a factory into the
 * shared registry (src/backend/cuda_graph_stub.cpp) at load time.
 *
 * Crucially, begin_capture() sets the ROCm backend's thread-local "current
 * stream" to the private capture stream, so every kernel the JIT's forward()
 * dispatches (which reads its stream from get_hip_stream()) is recorded onto the
 * captured stream instead of the un-capturable default stream. This is what
 * makes the captured graph actually contain the forward work.
 */

#include "tenzor/backend/cuda_graph.hpp"
#include "tenzor/core/device.hpp"
#include "hip_graph.hpp"
#include "rocm_stream.hpp"
#include "rocm_error.hpp"  // HIP_CHECK for checked device-set paths

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdlib>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>  // std::ignore for intentional discards in dtor/cleanup

namespace tenzor {
namespace {

class HIPGraphImpl : public CUDAGraph {
public:
    explicit HIPGraphImpl(int32_t device_id) : device_id_(device_id) {
        HIP_CHECK(hipSetDevice(device_id_));
        auto err = hipStreamCreate(&stream_);
        if (err != hipSuccess) {
            throw std::runtime_error(
                std::string("HIPGraphImpl: failed to create stream: ") +
                hipGetErrorString(err));
        }
    }

    ~HIPGraphImpl() override {
        // Never leave the thread-local current stream dangling on this thread.
        rocm::rocm_current_stream() = nullptr;
        if (stream_) std::ignore = hipStreamDestroy(stream_);
        std::ignore = hipGetLastError();  // drain any sticky error (dtor must not throw)
    }

    void begin_capture() override {
        HIP_CHECK(hipSetDevice(device_id_));
        // Route all subsequent dispatch launches onto the capture stream.
        rocm::rocm_current_stream() = stream_;
        try {
            graph_.begin_capture(stream_);
        } catch (...) {
            rocm::rocm_current_stream() = nullptr;
            throw;
        }
    }

    void end_capture() override {
        // Restore the default stream for normal execution regardless of outcome.
        try {
            graph_.end_capture(stream_);
        } catch (...) {
            rocm::rocm_current_stream() = nullptr;
            throw;
        }
        rocm::rocm_current_stream() = nullptr;
    }

    void replay() override {
        HIP_CHECK(hipSetDevice(device_id_));
        graph_.replay(stream_);  // launches then synchronizes stream_
    }

    bool is_ready() const override { return graph_.is_compiled(); }

private:
    int32_t device_id_;
    hipStream_t stream_ = nullptr;
    rocm::HIPGraph graph_;
};

auto make_hip_graph(int32_t device_id) -> std::unique_ptr<CUDAGraph> {
    int device_count = 0;
    if (hipGetDeviceCount(&device_count) != hipSuccess) return nullptr;
    if (device_count == 0 || device_id < 0 || device_id >= device_count) {
        return nullptr;
    }
    return std::make_unique<HIPGraphImpl>(device_id);
}

// Register the ROCm graph factory when this backend .so is dlopen'd (global
// constructors run on load). The implementation routes the captured forward()
// onto the capture stream (via the shared rocm_current_stream()) and relies on
// the caching allocator (default ON) to keep capture allocation-free — the AMD
// driver faults on an allocation issued while a stream is capturing.
struct HIPGraphFactoryRegistrar {
    HIPGraphFactoryRegistrar() {
        CUDAGraph::register_factory(static_cast<int>(Device::Type::ROCm),
                                    &make_hip_graph);
    }
};
[[maybe_unused]] const HIPGraphFactoryRegistrar g_hip_graph_registrar;

}  // namespace
}  // namespace tenzor
