#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <functional>
#include <limits>
#include <stdexcept>
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
 *
 * @note The running sum and the per-element exclusive offsets are accumulated in
 *       int64_t internally so the returned total is always correct even when T is
 *       a narrow type (e.g. int32_t). The exclusive offsets are stored back into
 *       the T-typed @p data buffer, which callers read as T to index into output
 *       buffers; if the total count exceeds the representable range of T the
 *       offset map could not be stored without overflow, so the function throws
 *       rather than silently corrupting the index map.
 */
template<typename T>
inline int64_t sycl_exclusive_prefix_sum(T* data, int64_t n, sycl::queue& queue) {
    static_assert(std::is_integral_v<T>, "sycl_exclusive_prefix_sum requires an integral type");
    if (n == 0) return 0;

    // Both branches below obtain the true total directly from the scan output
    // (d_total), so there is no need for the prior blocking D2H read of the last
    // input element — that stalled the in-order queue on every call (hot path for
    // nonzero / masked_select / sparse index building) for a value never used.
    //
    // Accumulation is performed in int64_t regardless of T: a narrow T (int32)
    // would otherwise overflow the running sum and the per-element offsets before
    // the int64_t return value widening, producing a silently corrupted index map
    // (the widened return then giving a false sense of 64-bit safety). We compute
    // the exclusive scan in a dedicated int64_t scratch buffer, derive the true
    // total from it, verify the total still fits in T, and only then narrow the
    // offsets back into data[] (which callers consume as T).
    SyclDeviceBuffer<int64_t> exclusive_buf(n, queue);
    int64_t* d_exclusive = exclusive_buf.get();
    SyclDeviceBuffer<int64_t> total_buf(1, queue);
    int64_t* d_total = total_buf.get();

#ifdef TENZOR_HAS_ONEDPL
    auto policy = ::oneapi::dpl::execution::make_device_policy(queue);

    // Inclusive scan widened to int64_t (the transform widens each element of the
    // T-typed input before summation, so the accumulator never overflows in T).
    ::oneapi::dpl::transform_inclusive_scan(
        policy, data, data + n, d_exclusive,
        ::std::plus<int64_t>{},
        [](T v) { return static_cast<int64_t>(v); });

    // Capture the total (last inclusive value) before shifting in-place. Grab it
    // with a single_task so the subsequent parallel_for can freely overwrite
    // d_exclusive's consumer (data[]) without a read/write race on the total.
    queue.single_task([=]() { d_total[0] = d_exclusive[n - 1]; }).wait();

    // Shift right to make it exclusive. Read d_exclusive[i-1] into data[i].
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        data[i] = (i == 0) ? static_cast<T>(0)
                           : static_cast<T>(d_exclusive[i - 1]);
    }).wait();
#else
    // Without oneDPL: sequential scan on device with an int64_t accumulator.
    queue.single_task([=]() {
        int64_t sum = 0;
        for (int64_t i = 0; i < n; ++i) {
            int64_t val = static_cast<int64_t>(data[i]);
            d_exclusive[i] = sum;  // exclusive offset (int64, never overflows)
            sum += val;
        }
        d_total[0] = sum;
    }).wait();

    // Narrow exclusive offsets back into the T-typed buffer that callers index.
    queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
        data[i] = static_cast<T>(d_exclusive[i]);
    }).wait();
#endif

    int64_t total = 0;
    queue.memcpy(&total, d_total, sizeof(int64_t)).wait();

    // The exclusive offsets just stored into data[] (type T) are read back as T by
    // the caller to address output buffers. If the total count does not fit in T,
    // the largest offsets were truncated on narrowing above and the index map is
    // corrupt. Fail loudly instead of returning a 64-bit total that masks it.
    if constexpr (std::numeric_limits<T>::max() < std::numeric_limits<int64_t>::max()) {
        if (total > static_cast<int64_t>(std::numeric_limits<T>::max())) {
            throw std::overflow_error(
                "sycl_exclusive_prefix_sum: element count exceeds the range of the "
                "offset type T; the in-place offset map would overflow. Instantiate "
                "with a wider index type (int64_t).");
        }
    }

    return total;
}

} // namespace oneapi
} // namespace tenzor
