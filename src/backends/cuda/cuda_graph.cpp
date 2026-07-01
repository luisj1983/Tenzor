#include "tenzor/backend/cuda_graph.hpp"
#include "cuda_graph.hpp"
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
        cudaSetDevice(device_id_);
        auto err = cudaStreamCreate(&stream_);
        if (err != cudaSuccess) {
            throw std::runtime_error(
                std::string("CUDAGraph: failed to create stream: ") +
                cudaGetErrorString(err));
        }
    }

    ~CUDAGraphImpl() override {
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

    void begin_capture() override {
        cudaSetDevice(device_id_);
        capture_.begin_capture(stream_);
    }

    void end_capture() override {
        capture_.end_capture();
    }

    void replay() override {
        cudaSetDevice(device_id_);
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

auto CUDAGraph::create(int32_t device_id) -> std::unique_ptr<CUDAGraph> {
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

} // namespace tenzor
