/**
 * @file async_test_support.hpp
 * @brief Test-only generic async-execution wrapper.
 *
 * Relocated from include/tenzor/ops/async_ops.hpp: the `async_execute`
 * template was exercised only by the async unit tests, never by production
 * code, so it lives here in the test tree. Both test_async_ops.cpp and
 * test_async_ops_multidtype.cpp include this header (the template must be
 * visible at every call site).
 */

#pragma once

#include <tenzor/ops/async_ops.hpp>
#include <tenzor/utils/threading/future.hpp>
#include <tenzor/utils/threading/threadpool.hpp>

#include <exception>
#include <memory>
#include <utility>

namespace tenzor {

/**
 * @brief Generic async operation wrapper.
 *
 * Wraps any synchronous tensor operation to execute asynchronously on the
 * shared thread pool.
 *
 * @tparam F Callable type with signature Tensor(Args...)
 * @tparam Args Argument types
 * @param func Function to execute asynchronously
 * @param args Arguments to pass to func
 * @return Future<Tensor> for operation result
 */
template <typename F, typename... Args>
auto async_execute(F&& func, Args&&... args) -> Future<Tensor> {
    auto promise = std::make_shared<Promise<Tensor>>();
    auto future = Future<Tensor>(promise->get_state());

    thread_pool().submit([promise, func = std::forward<F>(func),
                          ... captured_args = std::forward<Args>(args)]() mutable {
        try {
            Tensor result = func(std::forward<Args>(captured_args)...);
            promise->set_value(std::move(result));
        } catch (...) {
            promise->set_exception(std::current_exception());
        }
    });

    return future;
}

}  // namespace tenzor
