/**
 * @file cpu_thread_config.cpp
 * @brief Implementation of the single source of truth for OMP thread count.
 */

#include "cpu_thread_config.hpp"

#include <cstdlib>
#include <mutex>
#include <thread>

#ifdef _OPENMP
#  include <omp.h>
#endif

namespace tenzor::backends::cpu {
namespace {
std::once_flag g_once;
int g_threads = 0;
} // namespace

void configure_omp_threads() {
    std::call_once(g_once, [] {
        const char* env = std::getenv("OMP_NUM_THREADS");
        if (env) {
            g_threads = std::atoi(env);
            if (g_threads < 1) g_threads = 1;
        } else {
            // Default: hardware_concurrency / 2 gives physical-core count on
            // hyper-threaded CPUs; clamp to at least 1.
            unsigned int hw = std::thread::hardware_concurrency();
            g_threads = static_cast<int>(hw / 2);
            if (g_threads < 1) g_threads = 1;
        }
#ifdef _OPENMP
        omp_set_num_threads(g_threads);
#endif
    });
}

int get_configured_threads() { return g_threads; }

} // namespace tenzor::backends::cpu
