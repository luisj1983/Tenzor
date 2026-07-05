/**
 * @file cpu_thread_config.cpp
 * @brief Implementation of the single source of truth for OMP thread count.
 *
 * Precedence (highest wins):
 *   1. OMP_NUM_THREADS  — standard, already honoured by the OpenMP runtime; we
 *      observe it explicitly so that get_configured_threads() returns the same
 *      value the OMP runtime ends up using.
 *   2. TENZOR_NUM_THREADS — Tenzor-specific override for callers that want to
 *      cap parallelism without disturbing host OpenMP applications.
 *   3. Auto default — total online cores via sysconf(_SC_NPROCESSORS_ONLN),
 *      falling back to std::thread::hardware_concurrency(). We deliberately do
 *      NOT divide by 2: on modern non-SMT silicon (Apple/ARM, Intel hybrid
 *      P+E, EPYC SMT=off configurations) that produces half the available
 *      compute. The OS scheduler distributes work appropriately.
 */

#include "cpu_thread_config.hpp"

#include <algorithm>
#include <cerrno>
#include <climits>
#include <cstdlib>
#include <limits>
#include <mutex>
#include <thread>

#if defined(__unix__) || defined(__APPLE__)
#  include <unistd.h>
#endif

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace tenzor::backends::cpu {
namespace {

std::once_flag g_once;
int g_threads = 0;

auto parse_positive_int_env(const char* name, bool allow_nested_list = false) -> int {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return 0;
    }
    // std::atoi is UB on overflow; use std::strtol with explicit range checking
    // and reject trailing garbage. Out-of-range / non-positive values yield 0
    // (meaning "unset / auto-detect"). Values above INT_MAX are clamped to
    // INT_MAX here; configure_omp_threads applies a further sane upper bound.
    errno = 0;
    char* end = nullptr;
    long value = std::strtol(env, &end, 10);
    // Terminator handling:
    //   * TENZOR_NUM_THREADS (allow_nested_list=false): reject any trailing
    //     garbage (e.g. "8x", "4,8") — end must reach the terminating NUL.
    //   * OMP_NUM_THREADS (allow_nested_list=true): the OpenMP spec permits a
    //     comma-separated NESTED list, e.g. "4,2", where the FIRST value is the
    //     outer-level (top team) thread count the runtime actually uses. Parse
    //     that leading integer and accept a trailing ",..." remainder so
    //     get_configured_threads() matches the OMP runtime instead of falling
    //     through to auto-detect. A bare "8x" is still rejected.
    const bool ok_terminator =
        (*end == '\0') || (allow_nested_list && *end == ',');
    if (end == env || !ok_terminator || errno == ERANGE || value <= 0) {
        return 0;
    }
    if (value > static_cast<long>(std::numeric_limits<int>::max())) {
        return std::numeric_limits<int>::max();
    }
    return static_cast<int>(value);
}

auto detect_online_cores() -> int {
#if defined(_SC_NPROCESSORS_ONLN)
    long n = sysconf(_SC_NPROCESSORS_ONLN);
    if (n > 0) {
        return static_cast<int>(n);
    }
#endif
    unsigned int hw = std::thread::hardware_concurrency();
    return hw > 0 ? static_cast<int>(hw) : 1;
}

} // namespace

void configure_omp_threads() {
    std::call_once(g_once, [] {
        // Precedence: OMP_NUM_THREADS > TENZOR_NUM_THREADS > auto detect.
        int chosen = parse_positive_int_env("OMP_NUM_THREADS", /*allow_nested_list=*/true);
        if (chosen == 0) {
            chosen = parse_positive_int_env("TENZOR_NUM_THREADS");
        }
        if (chosen == 0) {
            chosen = detect_online_cores();
        }
        if (chosen < 1) {
            chosen = 1;
        }
        // Clamp to a sane upper bound to prevent resource exhaustion / DoS from
        // a hostile or fat-fingered TENZOR_NUM_THREADS/OMP_NUM_THREADS (e.g.
        // 100000 → omp_set_num_threads(100000) would attempt to spawn 100k
        // threads). Allow generous headroom over the detected core count.
        int hw = detect_online_cores();
        int max_threads = std::max(64, hw * 8);
        if (chosen > max_threads) {
            chosen = max_threads;
        }
        g_threads = chosen;
#ifdef _OPENMP
        omp_set_num_threads(g_threads);
#endif
    });
}

int get_configured_threads() {
    // Before configure_omp_threads() has run, g_threads is 0. Return 1 in that
    // case so callers that size/divide by the thread count cannot divide-by-zero
    // or allocate a zero-length per-thread buffer.
    return g_threads > 0 ? g_threads : 1;
}

} // namespace tenzor::backends::cpu
