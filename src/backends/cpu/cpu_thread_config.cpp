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
#include <cstdlib>
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

auto parse_positive_int_env(const char* name) -> int {
    const char* env = std::getenv(name);
    if (env == nullptr || *env == '\0') {
        return 0;
    }
    int value = std::atoi(env);
    return value > 0 ? value : 0;
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
        int chosen = parse_positive_int_env("OMP_NUM_THREADS");
        if (chosen == 0) {
            chosen = parse_positive_int_env("TENZOR_NUM_THREADS");
        }
        if (chosen == 0) {
            chosen = detect_online_cores();
        }
        if (chosen < 1) {
            chosen = 1;
        }
        g_threads = chosen;
#ifdef _OPENMP
        omp_set_num_threads(g_threads);
#endif
    });
}

int get_configured_threads() { return g_threads; }

} // namespace tenzor::backends::cpu
