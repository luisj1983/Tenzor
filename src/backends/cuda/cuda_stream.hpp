#pragma once

#include <cuda_runtime.h>

namespace tenzor {
namespace cuda {

/// Thread-local "current stream" for the CUDA backend. Defaults to nullptr (the
/// default stream), so ordinary execution is unchanged. CUDA-graph capture sets
/// this to the capture stream around a forward() so every kernel launched by the
/// dispatch (which pulls its stream from get_cuda_stream()) is recorded onto the
/// captured stream instead of the un-capturable default stream.
///
/// Inline function → a single shared `static thread_local` instance across every
/// TU in the CUDA backend .so that includes this header.
inline cudaStream_t& cuda_current_stream() {
    static thread_local cudaStream_t s = nullptr;
    return s;
}

}  // namespace cuda
}  // namespace tenzor
