#include "tenzor/backend/cuda_graph.hpp"
#include "cuda_graph.hpp"
#include "cuda_stream.hpp"
#include "cuda_error.hpp"
#include "tenzor/core/device.hpp"
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

namespace tenzor {

/**
 * @brief Concrete CUDA Graph implementation using cudaGraph_t.
 */
class CUDAGraphImpl : public CUDAGraph {
public:
    explicit CUDAGraphImpl(int32_t device_id) : device_id_(device_id) {
        // A silently-failed cudaSetDevice here would create the capture stream
        // on the wrong device; the cuBLAS/cuSPARSE/cuSOLVER handle pools all
        // resolve their per-thread handle via cudaGetDevice(), so ops during
        // capture/replay would then silently pick up the wrong device's
        // handle instead of raising a clear error like CUDABackend::set_device().
        CUDA_CHECK(cudaSetDevice(device_id_));
        auto err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDAGraph: failed to create stream: ") +
                cudaGetErrorString(err));
        }
    }

    ~CUDAGraphImpl() override {
        // Never leave the thread-local current stream dangling on this thread.
        cuda::cuda_current_stream() = nullptr;
        if (capture_.is_capturing()) {
            // Abort capture to avoid leaving stream in bad state
            cudaGraph_t dummy = nullptr;
            cudaStreamEndCapture(stream_, &dummy);
            if (dummy) cudaGraphDestroy(dummy);
        }
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
        // A failed cudaStreamEndCapture during an aborted capture (or any error
        // above) leaves a sticky error latched on the runtime for this thread,
        // which would otherwise surface as a spurious failure in the NEXT,
        // unrelated CUDA call. Destructors must not throw, so we deliberately
        // swallow these return codes — but drain the sticky error here so it
        // cannot bleed into subsequent ops.
        cudaGetLastError();
    }

    void prepare_capture_stream() override {
        CUDA_CHECK(cudaSetDevice(device_id_));
        cuda::cuda_current_stream() = stream_;
    }

    void begin_capture() override {
        CUDA_CHECK(cudaSetDevice(device_id_));
        // Route all subsequent dispatch launches onto the capture stream so the
        // forward pass is recorded (the backend otherwise launches on the
        // un-capturable default stream).
        cuda::cuda_current_stream() = stream_;
        try {
            capture_.begin_capture(stream_);
        } catch (...) {
            cuda::cuda_current_stream() = nullptr;
            throw;
        }
    }

    void end_capture() override {
        try {
            capture_.end_capture();
        } catch (...) {
            cuda::cuda_current_stream() = nullptr;
            throw;
        }
        cuda::cuda_current_stream() = nullptr;
    }

    void replay() override {
        CUDA_CHECK(cudaSetDevice(device_id_));
        capture_.replay(stream_);
    }

    bool is_ready() const override {
        return capture_.is_ready();
    }

private:
    int32_t device_id_;
    cudaStream_t stream_ = nullptr;
    cuda::CUDAGraphCapture capture_;
};

namespace {

auto make_cuda_graph(int32_t device_id) -> std::unique_ptr<CUDAGraph> {
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess) {
        return nullptr;
    }
    // Reject a negative device_id too — otherwise it flows into cudaSetDevice as
    // an invalid ordinal (only the upper bound was previously checked).
    if (device_count == 0 || device_id < 0 || device_id >= device_count) {
        return nullptr;
    }
    return std::make_unique<CUDAGraphImpl>(device_id);
}

// Register the CUDA graph factory when this backend .so is dlopen'd. Backends
// use RTLD_LOCAL, so the JIT (in tenzor_core) reaches this implementation via the
// shared registry rather than by linking CUDAGraph::create directly.
struct CUDAGraphFactoryRegistrar {
    CUDAGraphFactoryRegistrar() {
        CUDAGraph::register_factory(static_cast<int>(Device::Type::CUDA),
                                    &make_cuda_graph);
    }
};
[[maybe_unused]] const CUDAGraphFactoryRegistrar g_cuda_graph_registrar;

}  // namespace

} // namespace tenzor
