/**
 * @file parallel_for.hpp
 * @brief Parallel loop execution primitives
 *
 * Provides high-level parallel programming constructs for data-parallel operations.
 */

#pragma once

#include <functional>
#include <cstdint>

namespace tenzor {

/**
 * @brief Execute loop iterations in parallel
 *
 * Divides iteration range [begin, end) across available threads and executes
 * func(i) for each iteration in parallel. Automatic work distribution and load balancing.
 *
 * @tparam F Function type with signature void(int64_t)
 * @param begin Start of iteration range (inclusive)
 * @param end End of iteration range (exclusive)
 * @param func Function to execute for each iteration
 *
 * @par Complexity
 * - Work: O(end - begin)
 * - Span: O((end - begin) / num_threads)
 *
 * @par Thread Safety
 * func must be thread-safe if it accesses shared data
 *
 * @code
 * parallel_for(0, 1000, [&](int64_t i) {
 *     output[i] = compute(input[i]);
 * });
 * @endcode
 */
template<typename F>
auto parallel_for(int64_t begin, int64_t end, F&& func) -> void;

/**
 * @brief Execute loop with custom grain size
 *
 * Like parallel_for but with explicit grain size control. Grain size determines
 * minimum chunk size per thread, allowing fine-tuning of parallelism granularity.
 *
 * @tparam F Function type with signature void(int64_t)
 * @param begin Start of iteration range
 * @param end End of iteration range
 * @param grain_size Minimum iterations per thread (larger = less overhead)
 * @param func Function to execute
 *
 * @code
 * // Process in chunks of at least 100 items
 * parallel_for(0, 10000, 100, [&](int64_t i) {
 *     // ... work ...
 * });
 * @endcode
 */
template<typename F>
auto parallel_for(int64_t begin, int64_t end, int64_t grain_size, F&& func) -> void;

/**
 * @brief Parallel map-reduce operation
 *
 * Applies map_func to each element in [begin, end), then combines results using
 * reduce_func. Generalizes sum/min/max/product operations.
 *
 * @tparam T Result type
 * @tparam F Map function type: T(int64_t)
 * @tparam R Reduce function type: T(T, T)
 * @param begin Start of range
 * @param end End of range
 * @param init Initial value for reduction
 * @param map_func Maps index to value
 * @param reduce_func Combines two values
 * @return Final reduced value
 *
 * @par Complexity
 * - Work: O(end - begin)
 * - Span: O(log(num_threads))
 *
 * @code
 * // Parallel sum
 * auto sum = parallel_reduce(0, 1000, 0.0,
 *     [&](int64_t i) { return data[i]; },      // map
 *     [](double a, double b) { return a + b; } // reduce
 * );
 * @endcode
 */
template<typename T, typename F, typename R>
auto parallel_reduce(int64_t begin, int64_t end,
                    T init,
                    F&& map_func,
                    R&& reduce_func) -> T;

} // namespace tenzor
