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
 * Reads OMP_NUM_THREADS env var if set; otherwise defaults to
 * hardware_concurrency() / 2 (physical cores on HT systems), minimum 1.
 * Idempotent — safe to call from multiple translation units.
 */
void configure_omp_threads();

/**
 * @brief Return the thread count that was configured by configure_omp_threads().
 *
 * Returns 0 if configure_omp_threads() has not been called yet.
 */
int get_configured_threads();

} // namespace tenzor::backends::cpu
