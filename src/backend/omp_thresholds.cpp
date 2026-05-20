// SPDX-License-Identifier: BSD-3-Clause
//
// Definition for tenzor::OmpThresholds (audit item F.5).
//
// Lives in a .cpp so the public header at include/tenzor/backend/omp_thresholds.hpp
// doesn't have to pull <thread> + <mutex> into every translation unit that
// uses the thresholds.

#include "tenzor/backend/omp_thresholds.hpp"

#include <cstdlib>
#include <mutex>
#include <thread>

namespace tenzor {

namespace {

int64_t simple_  = 65536;
int64_t medium_  = 32768;
int64_t complex_ = 8192;
int64_t matmul_  = 1024;
std::once_flag init_flag_;

int64_t env_or(const char* name, int64_t default_val) {
    const char* val = std::getenv(name);
    if (val) {
        char* end = nullptr;
        long long parsed = std::strtoll(val, &end, 10);
        if (end != val && parsed > 0)
            return static_cast<int64_t>(parsed);
    }
    return default_val;
}

void init() {
    std::call_once(init_flag_, [] {
        int cores = static_cast<int>(std::thread::hardware_concurrency());
        if (cores < 1) cores = 1;
        simple_  = env_or("TENZOR_OMP_THRESHOLD_SIMPLE",  16384LL * cores);
        medium_  = env_or("TENZOR_OMP_THRESHOLD_MEDIUM",   8192LL * cores);
        complex_ = env_or("TENZOR_OMP_THRESHOLD_COMPLEX",  2048LL * cores);
        matmul_  = env_or("TENZOR_OMP_THRESHOLD_MATMUL",   1024LL);
    });
}

}  // namespace

int64_t OmpThresholds::simple()  { init(); return simple_; }
int64_t OmpThresholds::medium()  { init(); return medium_; }
int64_t OmpThresholds::complex() { init(); return complex_; }
int64_t OmpThresholds::matmul()  { init(); return matmul_; }

}  // namespace tenzor
