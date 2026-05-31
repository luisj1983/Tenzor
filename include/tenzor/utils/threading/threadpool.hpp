/**
 * @file threadpool.hpp
 * @brief Work-stealing thread pool for parallel tensor operations
 *
 * Provides efficient thread pool implementation for CPU-based parallel computations.
 */

#pragma once

#include <thread>
#include <vector>
#include <queue>
#include <functional>
#include <future>
#include <mutex>
#include <condition_variable>
#include <atomic>

namespace tenzor {

/**
 * @brief Work-stealing thread pool for parallel task execution
 *
 * Maintains a pool of worker threads that execute submitted tasks.
 * Provides efficient work distribution and load balancing.
 *
 * **Features:**
 * - Fixed number of threads (typically hardware concurrency)
 * - Task queue with automatic work distribution
 * - Future-based result retrieval
 * - Graceful shutdown on destruction
 *
 * **Use Cases:**
 * - Parallel tensor operations (element-wise, reductions)
 * - Data loading and preprocessing
 * - Model inference batching
 *
 * @par Thread Safety
 * All methods are thread-safe
 *
 * @code
 * auto pool = ThreadPool(4);  // 4 worker threads
 * auto future = pool.submit([]() {
 *     return expensive_computation();
 * });
 * auto result = future.get();  // Wait for completion
 * @endcode
 */
class ThreadPool {
public:
    /**
     * @brief Construct thread pool with specified number of threads
     * @param num_threads Number of worker threads (default: hardware concurrency)
     */
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());

    /**
     * @brief Destructor - waits for all tasks to complete
     */
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    /**
     * @brief Submit task for asynchronous execution
     *
     * Queues task for execution and returns future for result retrieval.
     *
     * @tparam F Callable type
     * @tparam Args Argument types
     * @param func Callable object to execute
     * @param args Arguments to pass to func
     * @return Future for retrieving result
     * @throws std::runtime_error if pool is stopped
     */
    template<typename F, typename... Args>
    auto submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    /**
     * @brief Get number of threads in pool
     * @return Number of worker threads
     */
    auto num_threads() const -> size_t { return num_threads_; }

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> tasks_;
    std::mutex queue_mutex_;
    std::condition_variable condition_;
    std::atomic<bool> stop_{false};
    std::atomic<size_t> active_threads_{0};
    size_t num_threads_;

    auto worker_thread() -> void;
};

// Template implementations
template<typename F, typename... Args>
auto ThreadPool::submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>> {
    using return_type = std::invoke_result_t<F, Args...>;

    auto task = std::make_shared<std::packaged_task<return_type()>>(
        std::bind(std::forward<F>(func), std::forward<Args>(args)...)
    );

    std::future<return_type> result = task->get_future();
    {
        std::unique_lock lock(queue_mutex_);
        if (stop_) {
            throw std::runtime_error("Cannot submit task to stopped thread pool");
        }
        tasks_.emplace([task]() { (*task)(); });
    }
    condition_.notify_one();
    return result;
}

/**
 * @brief Get global thread pool instance
 *
 * Returns reference to singleton thread pool used by library operations.
 * Thread pool is lazily initialized on first access.
 *
 * @return Reference to global ThreadPool
 */
auto thread_pool() -> ThreadPool&;

} // namespace tenzor
