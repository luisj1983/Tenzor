#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <type_traits>
#include "sycl_buffer_guard.hpp"

#ifdef TENZOR_HAS_ONEDPL
#include <oneapi/dpl/algorithm>
#include <oneapi/dpl/execution>
#endif

namespace tenzor {
namespace oneapi {

/**
 * @brief Device-side exclusive prefix sum on an integral array.
 *
 * Uses oneDPL inclusive_scan when available, otherwise falls back to a
 * sequential single_task scan on device (avoids host roundtrips).
 *
 * @tparam T       Integral type (int32_t, int64_t, etc.)
 * @param data     Device pointer to array (modified in-place)
 * @param n        Number of elements
 * @param queue    SYCL queue
 * @return         The total sum (last inclusive prefix sum value)
 */
template<typename T>
inline int64_t sycl_exclusive_prefix_sum(T* data, int64_t n, sycl::queue& queue) {
    static_assert(std::is_integral_v<T>, "sycl_exclusive_prefix_sum requires an integral type");
    if (n == 0) return 0;

    // Both branches below obtain the true total directly from the scan output
    // (d_inclusive[n-1] for oneDPL, d_total for the fallback), so there is no
    // need for the prior blocking D2H read of the last input element — that
    // stalled the in-order queue on every call (hot path for nonzero /
    // masked_select / sparse index building) for a value that was never used.

    // Use oneDPL if available for device-side inclusive scan, then shift
#ifdef TENZOR_HAS_ONEDPL
    auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

    // Allocate temp buffer for inclusive scan result (RAII: freed on any throw,
    // including a SYCL async exception surfaced by .wait()).
    SyclDeviceBuffer<T> inclusive_buf(n, queue);
    T* d_inclusive = inclusive_buf.get();
    ::oneapi::dpl::inclusive_scan(policy, data, data + n, d_inclusive);

    // Read total count from last element
    T total = 0;
    queue.memcpy(&total, d_inclusive + n - 1, sizeof(T)).wait();

    // Convert inclusive to exclusive: shift right by 1, set first to 0
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        data[i] = (i == 0) ? static_cast<T>(0) : d_inclusive[i - 1];
    }).wait();

    return static_cast<int64_t>(total);
#else
    // Without oneDPL: simple sequential scan on device (still avoids D2H roundtrip for data)
    // For moderate sizes this is fine; the bottleneck was the full data transfer, not the scan
    SyclDeviceBuffer<T> total_buf(1, queue);
    T* d_total = total_buf.get();

    queue.single_task([=]() {
        T sum = 0;
        for (int64_t i = 0; i < n; ++i) {
            T val = data[i];
            data[i] = sum;
            sum += val;
        }
        d_total[0] = sum;
    }).wait();

    T total = 0;
    queue.memcpy(&total, d_total, sizeof(T)).wait();
    return static_cast<int64_t>(total);
#endif
}

} // namespace oneapi
} // namespace tenzor
