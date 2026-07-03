#pragma once

#include <hip/hip_runtime.h>

namespace tenzor {
namespace rocm {

/// Thread-local "current stream" for the ROCm backend. Defaults to nullptr (the
/// default stream), so ordinary execution is unchanged. HIP-graph capture sets
/// this to the capture stream around a forward() so every kernel launched by the
/// dispatch (which reads its stream from get_hip_stream()) AND every allocation
/// is recorded onto the captured stream instead of the un-capturable default
/// stream.
///
/// This is declared here and DEFINED once in rocm_kernel_registry.cpp so there is
/// exactly ONE shared thread_local across the whole ROCm backend .so. (An inline
/// definition risked separate per-TU instances under the backend's split
/// manual-compile build, which left the capture stream unset in some TUs.)
hipStream_t& rocm_current_stream();

/// RAII helper: set the current stream for the enclosing scope and restore the
/// previous value on exit (exception-safe).
class ScopedCurrentStream {
public:
    explicit ScopedCurrentStream(hipStream_t stream)
        : prev_(rocm_current_stream()) {
        rocm_current_stream() = stream;
    }
    ~ScopedCurrentStream() { rocm_current_stream() = prev_; }
    ScopedCurrentStream(const ScopedCurrentStream&) = delete;
    ScopedCurrentStream& operator=(const ScopedCurrentStream&) = delete;

private:
    hipStream_t prev_;
};

}  // namespace rocm
}  // namespace tenzor
