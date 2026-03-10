/**
 * @file profiling.hpp
 * @brief GPU profiling annotations for NVTX (CUDA) and ROCTX (ROCm).
 *
 * Provides TENZOR_PROFILE_RANGE(name) macro that creates a scoped profiling
 * range visible in nsys (NVIDIA Nsight Systems) or rocprof (AMD ROCm Profiler).
 *
 * Controlled by compile-time defines:
 * - TENZOR_HAS_NVTX: Enable NVTX annotations (CUDA Toolkit)
 * - TENZOR_HAS_ROCTX: Enable ROCTX annotations (ROCm)
 * - Neither: Macro compiles to nothing (zero overhead)
 *
 * Build with -DTENZOR_ENABLE_PROFILING=ON to enable.
 */

#pragma once

#if defined(TENZOR_HAS_NVTX)

#include <nvtx3/nvtx3.hpp>

/// Create a scoped NVTX profiling range with the given name.
/// The range ends when the enclosing scope exits.
#define TENZOR_PROFILE_RANGE(name) nvtx3::scoped_range _tenzor_nvtx_range(name)

#elif defined(TENZOR_HAS_ROCTX)

#include <roctx.h>

namespace tenzor {
namespace detail {

/// RAII wrapper for ROCTX range push/pop.
struct RoctxRange {
    explicit RoctxRange(const char* name) { roctxRangePush(name); }
    ~RoctxRange() { roctxRangePop(); }
    RoctxRange(const RoctxRange&) = delete;
    RoctxRange& operator=(const RoctxRange&) = delete;
};

} // namespace detail
} // namespace tenzor

/// Create a scoped ROCTX profiling range with the given name.
#define TENZOR_PROFILE_RANGE(name) ::tenzor::detail::RoctxRange _tenzor_roctx_range(name)

#else

/// No-op when profiling is disabled.
#define TENZOR_PROFILE_RANGE(name) ((void)0)

#endif
