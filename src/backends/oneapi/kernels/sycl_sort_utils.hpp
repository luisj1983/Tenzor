#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>
#include <type_traits>

namespace tenzor {
namespace oneapi {

/**
 * @brief NaN-aware strict-weak-ordering "less than" for device-side sorting.
 *
 * The default operator< is NaN-blind (every comparison against NaN is
 * false), which violates strict-weak-ordering and leaves data effectively
 * unsorted around any NaN. This total order sorts NaN as strictly the
 * largest value (any sign), matching CPU's `nan_less`
 * (src/backends/cpu/kernels/advanced.cpp) and this file's own Quantile/
 * Median NaN-aware sort comparator (src/backends/oneapi/kernels/math.cpp,
 * quantile_impl). Integer T has no NaN concept, so this collapses to a
 * plain `a < b`.
 */
template<typename T>
inline bool sycl_nan_safe_less(T a, T b) {
    if constexpr (std::is_floating_point_v<T>) {
        bool na = sycl::isnan(a);
        bool nb = sycl::isnan(b);
        if (na || nb) return !na && nb;   // finite < NaN; NaN < nothing
        return a < b;
    } else {
        return a < b;
    }
}

/**
 * @brief NaN-aware strict-weak-ordering "greater than" — the mirror of
 * sycl_nan_safe_less(), used by the bitonic sort's descending-stage swap
 * test.
 */
template<typename T>
inline bool sycl_nan_safe_greater(T a, T b) {
    return sycl_nan_safe_less(b, a);
}

/**
 * @brief Largest-possible sentinel value for type @p T.
 *
 * Used to pad a bitonic-sort buffer up to a power-of-two length so that the
 * padding elements always sort to the high end (ascending order) and never
 * displace a real element. For floating-point types this must be a value
 * that dominates every possible real input under sycl_nan_safe_less/greater
 * above — and since those comparators order NaN as strictly the largest
 * value (larger than +infinity), a merely-infinite sentinel would sort
 * BEFORE a genuine NaN in the input, letting the fabricated +inf leak into
 * the real [0,n) result window while the true NaN gets displaced into the
 * discarded padding tail. Only a NaN sentinel dominates every possible real
 * value (including NaN itself) under that ordering, so use quiet_NaN() here
 * instead of infinity().
 */
template<typename T>
constexpr T sycl_sort_max_sentinel() {
    if constexpr (std::numeric_limits<T>::has_infinity) {
        return std::numeric_limits<T>::quiet_NaN();
    } else {
        return std::numeric_limits<T>::max();
    }
}

/**
 * @brief In-place bitonic sort on device memory.
 *
 * Operates on USM device pointers. Fixed number of rounds with
 * no data-dependent branching — well suited for SYCL.
 *
 * For non-power-of-two @p n the routine sorts on an internally allocated
 * power-of-two buffer whose tail is filled with a +max sentinel, then copies
 * the first @p n (smallest) elements back. This avoids the incorrect ordering
 * that results from merely skipping out-of-range comparators (skipping is only
 * equivalent to a sentinel in ascending sub-stages, not descending ones).
 *
 * @tparam T Element type (must support operator< and operator>)
 * @param data Pointer to device memory
 * @param n Number of elements to sort
 * @param queue SYCL queue for kernel submission
 */
template<typename T>
void sycl_bitonic_sort(T* data, int64_t n, sycl::queue& queue) {
    if (n <= 1) return;

    // Round up to next power of 2
    int64_t padded = 1;
    while (padded < n) padded <<= 1;

    // Fast path: already a power of two — sort fully in place.
    if (padded == n) {
        for (int64_t k = 2; k <= padded; k <<= 1) {
            for (int64_t j = k >> 1; j > 0; j >>= 1) {
                queue.parallel_for(sycl::range<1>(padded)  /* full width: low elements span [0,padded); guard below dedups pairs */, [=](sycl::id<1> idx) {
                    int64_t i = static_cast<int64_t>(idx[0]);
                    int64_t ixj = i ^ j;
                    // Only process pairs where i < ixj (avoid double-swap)
                    if (ixj <= i) return;

                    // Determine sort direction: ascending if (i & k) == 0
                    bool ascending = ((i & k) == 0);

                    if (ascending) {
                        if (sycl_nan_safe_greater(data[i], data[ixj])) {
                            T tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                        }
                    } else {
                        if (sycl_nan_safe_less(data[i], data[ixj])) {
                            T tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                        }
                    }
                });
                queue.wait_and_throw();
            }
        }
        return;
    }

    // Non-power-of-two: pad to `padded` with a +max sentinel so the network
    // operates on a valid bitonic input and real elements occupy [0, n).
    const T sentinel = sycl_sort_max_sentinel<T>();
    T* buf = sycl::malloc_device<T>(padded, queue);
    queue.memcpy(buf, data, n * sizeof(T)).wait();
    queue.parallel_for(sycl::range<1>(padded - n), [=](sycl::id<1> idx) {
        buf[n + static_cast<int64_t>(idx[0])] = sentinel;
    }).wait();

    for (int64_t k = 2; k <= padded; k <<= 1) {
        for (int64_t j = k >> 1; j > 0; j >>= 1) {
            queue.parallel_for(sycl::range<1>(padded)  /* full width: low elements span [0,padded); guard below dedups pairs */, [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t ixj = i ^ j;
                if (ixj <= i) return;

                bool ascending = ((i & k) == 0);

                if (ascending) {
                    if (sycl_nan_safe_greater(buf[i], buf[ixj])) {
                        T tmp = buf[i];
                        buf[i] = buf[ixj];
                        buf[ixj] = tmp;
                    }
                } else {
                    if (sycl_nan_safe_less(buf[i], buf[ixj])) {
                        T tmp = buf[i];
                        buf[i] = buf[ixj];
                        buf[ixj] = tmp;
                    }
                }
            });
            queue.wait_and_throw();
        }
    }

    // Copy back the n smallest (real) elements; sentinels stay in [n, padded).
    queue.memcpy(data, buf, n * sizeof(T)).wait();
    sycl::free(buf, queue);
}

/**
 * @brief In-place bitonic sort with paired index tracking.
 *
 * Sorts values in ascending order and permutes the corresponding index
 * array in tandem. Useful for kthvalue where we need both the sorted
 * value and its original position.
 *
 * @tparam T Value type (must support operator<)
 * @param data Pointer to device memory holding values
 * @param indices Pointer to device memory holding indices
 * @param n Number of elements to sort
 * @param queue SYCL queue for kernel submission
 */
template<typename T>
void sycl_bitonic_sort_by_key(T* data, int64_t* indices, int64_t n, sycl::queue& queue) {
    if (n <= 1) return;

    // Round up to next power of 2
    int64_t padded = 1;
    while (padded < n) padded <<= 1;

    // Fast path: already a power of two — sort fully in place.
    if (padded == n) {
        for (int64_t k = 2; k <= padded; k <<= 1) {
            for (int64_t j = k >> 1; j > 0; j >>= 1) {
                queue.parallel_for(sycl::range<1>(padded)  /* full width: low elements span [0,padded); guard below dedups pairs */, [=](sycl::id<1> idx) {
                    int64_t i = static_cast<int64_t>(idx[0]);
                    int64_t ixj = i ^ j;
                    if (ixj <= i) return;

                    bool ascending = ((i & k) == 0);

                    if (ascending) {
                        if (sycl_nan_safe_greater(data[i], data[ixj])) {
                            T tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                            int64_t ti = indices[i];
                            indices[i] = indices[ixj];
                            indices[ixj] = ti;
                        }
                    } else {
                        if (sycl_nan_safe_less(data[i], data[ixj])) {
                            T tmp = data[i];
                            data[i] = data[ixj];
                            data[ixj] = tmp;
                            int64_t ti = indices[i];
                            indices[i] = indices[ixj];
                            indices[ixj] = ti;
                        }
                    }
                });
                queue.wait_and_throw();
            }
        }
        return;
    }

    // Non-power-of-two: pad values with a +max sentinel (and indices with a -1
    // sentinel index) so the comparator network operates on a valid bitonic
    // input. Real elements settle into [0, n); sentinels go to the high end.
    const T sentinel = sycl_sort_max_sentinel<T>();
    T* vbuf = sycl::malloc_device<T>(padded, queue);
    int64_t* ibuf = sycl::malloc_device<int64_t>(padded, queue);
    queue.memcpy(vbuf, data, n * sizeof(T)).wait();
    queue.memcpy(ibuf, indices, n * sizeof(int64_t)).wait();
    queue.parallel_for(sycl::range<1>(padded - n), [=](sycl::id<1> idx) {
        int64_t p = n + static_cast<int64_t>(idx[0]);
        vbuf[p] = sentinel;
        ibuf[p] = -1;
    }).wait();

    for (int64_t k = 2; k <= padded; k <<= 1) {
        for (int64_t j = k >> 1; j > 0; j >>= 1) {
            queue.parallel_for(sycl::range<1>(padded)  /* full width: low elements span [0,padded); guard below dedups pairs */, [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t ixj = i ^ j;
                if (ixj <= i) return;

                bool ascending = ((i & k) == 0);

                if (ascending) {
                    if (sycl_nan_safe_greater(vbuf[i], vbuf[ixj])) {
                        T tmp = vbuf[i];
                        vbuf[i] = vbuf[ixj];
                        vbuf[ixj] = tmp;
                        int64_t ti = ibuf[i];
                        ibuf[i] = ibuf[ixj];
                        ibuf[ixj] = ti;
                    }
                } else {
                    if (sycl_nan_safe_less(vbuf[i], vbuf[ixj])) {
                        T tmp = vbuf[i];
                        vbuf[i] = vbuf[ixj];
                        vbuf[ixj] = tmp;
                        int64_t ti = ibuf[i];
                        ibuf[i] = ibuf[ixj];
                        ibuf[ixj] = ti;
                    }
                }
            });
            queue.wait_and_throw();
        }
    }

    // Copy back the n smallest (real) elements and their original indices.
    queue.memcpy(data, vbuf, n * sizeof(T)).wait();
    queue.memcpy(indices, ibuf, n * sizeof(int64_t)).wait();
    sycl::free(vbuf, queue);
    sycl::free(ibuf, queue);
}

}  // namespace oneapi
}  // namespace tenzor
