/**
 * @file cpu_thread_config.hpp
 * @brief Single source of truth for OMP thread count configuration.
 *
 * Calling configure_omp_threads() from any CPU backend initialization path
 * is idempotent — the actual omp_set_num_threads call is guarded by a
 * std::once_flag so only the first call takes effect.
 */

#pragma once

namespace tenzor::backends::cpu {

/**
 * @brief Configure OMP thread count once process-wide.
 *
 * Precedence (highest wins):
 *   1. OMP_NUM_THREADS  — standard OpenMP env var, observed explicitly so
 *      get_configured_threads() agrees with the runtime.
 *   2. TENZOR_NUM_THREADS — Tenzor-specific override.
 *   3. Auto default — total online cores via sysconf(_SC_NPROCESSORS_ONLN),
 *      falling back to std::thread::hardware_concurrency(). We do not divide
 *      by 2 — that is wrong on non-SMT and hybrid hardware.
 *
 * Idempotent — safe to call from multiple translation units.
 */
void configure_omp_threads();

/**
 * @brief Return the thread count that was configured by configure_omp_threads().
 *
 * Returns 1 (not 0) if configure_omp_threads() has not been called yet, so
 * callers can safely use the result as a divisor / buffer count pre-init.
 */
int get_configured_threads();

} // namespace tenzor::backends::cpu
