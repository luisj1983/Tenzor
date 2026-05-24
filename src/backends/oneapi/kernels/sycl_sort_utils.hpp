#pragma once
#include <sycl/sycl.hpp>
#include <cstdint>

namespace tenzor {
namespace oneapi {

/**
 * @brief In-place bitonic sort on device memory.
 *
 * Operates on USM device pointers. Fixed number of rounds with
 * no data-dependent branching — well suited for SYCL.
 *
 * @tparam T Element type (must support operator<)
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

    // Bitonic sort network
    for (int64_t k = 2; k <= padded; k <<= 1) {
        for (int64_t j = k >> 1; j > 0; j >>= 1) {
            queue.parallel_for(sycl::range<1>(padded / 2), [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t ixj = i ^ j;
                // Only process pairs where i < ixj (avoid double-swap)
                if (ixj <= i) return;
                // Both indices must be in bounds
                if (i >= n || ixj >= n) return;

                // Determine sort direction: ascending if (i & k) == 0
                bool ascending = ((i & k) == 0);

                if (ascending) {
                    if (data[i] > data[ixj]) {
                        T tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                    }
                } else {
                    if (data[i] < data[ixj]) {
                        T tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                    }
                }
            });
            queue.wait_and_throw();
        }
    }
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

    // Bitonic sort network — swap both value and index in lockstep
    for (int64_t k = 2; k <= padded; k <<= 1) {
        for (int64_t j = k >> 1; j > 0; j >>= 1) {
            queue.parallel_for(sycl::range<1>(padded / 2), [=](sycl::id<1> idx) {
                int64_t i = static_cast<int64_t>(idx[0]);
                int64_t ixj = i ^ j;
                if (ixj <= i) return;
                if (i >= n || ixj >= n) return;

                bool ascending = ((i & k) == 0);

                if (ascending) {
                    if (data[i] > data[ixj]) {
                        T tmp = data[i];
                        data[i] = data[ixj];
                        data[ixj] = tmp;
                        int64_t ti = indices[i];
                        indices[i] = indices[ixj];
                        indices[ixj] = ti;
                    }
                } else {
                    if (data[i] < data[ixj]) {
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
}

}  // namespace oneapi
}  // namespace tenzor
