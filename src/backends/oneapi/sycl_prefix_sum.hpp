#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#endif

namespace tenzor {
namespace oneapi {

/**
 * @brief Device-side exclusive prefix sum on int32 array.
 *
 * Uses oneDPL inclusive_scan when available, otherwise falls back to a
 * sequential single_task scan on device (avoids host roundtrips).
 *
 * @param data     Device pointer to int32 array (modified in-place)
 * @param n        Number of elements
 * @param queue    SYCL queue
 * @return         The total sum (last inclusive prefix sum value)
 */
inline int64_t sycl_exclusive_prefix_sum(int32_t* data, int64_t n, sycl::queue& queue) {
    if (n == 0) return 0;

    // Read last element before scan (for inclusive total = exclusive_last + original_last)
    int32_t last_val = 0;
    queue.memcpy(&last_val, data + n - 1, sizeof(int32_t)).wait();

    // Use oneDPL if available for device-side inclusive scan, then shift
#ifdef TENZOR_HAS_ONEDPL
    auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

    // Allocate temp buffer for inclusive scan result
    int32_t* d_inclusive = sycl::malloc_device<int32_t>(n, queue);
    ::oneapi::dpl::inclusive_scan(policy, data, data + n, d_inclusive);

    // Read total count from last element
    int32_t total = 0;
    queue.memcpy(&total, d_inclusive + n - 1, sizeof(int32_t)).wait();

    // Convert inclusive to exclusive: shift right by 1, set first to 0
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        data[i] = (i == 0) ? 0 : d_inclusive[i - 1];
    }).wait();

    sycl::free(d_inclusive, queue);
    return static_cast<int64_t>(total);
#else
    // Without oneDPL: simple sequential scan on device (still avoids D2H roundtrip for data)
    // For moderate sizes this is fine; the bottleneck was the full data transfer, not the scan
    int32_t* d_total = sycl::malloc_device<int32_t>(1, queue);

    queue.single_task([=]() {
        int32_t sum = 0;
        for (int64_t i = 0; i < n; ++i) {
            int32_t val = data[i];
            data[i] = sum;
            sum += val;
        }
        d_total[0] = sum;
    }).wait();

    int32_t total = 0;
    queue.memcpy(&total, d_total, sizeof(int32_t)).wait();
    sycl::free(d_total, queue);
    return static_cast<int64_t>(total);
#endif
}

} // namespace oneapi
} // namespace tenzor
