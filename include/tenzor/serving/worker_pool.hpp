/**
 * @file worker_pool.hpp
 * @brief Thread pool for inference workers
 *
 * Manages a fixed-size pool of worker threads bound to a specific device.
 * Tasks are submitted via a thread-safe queue and executed by the next
 * available worker.
 */

#pragma once

#include "server.hpp"

#include <atomic>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

namespace tenzor::serving {

/**
 * @brief Fixed-size thread pool for running inference tasks on a device.
 *
 * Each worker thread is associated with the given Device so that tensors
 * created inside submitted tasks default to the correct backend.
 */
class WorkerPool {
public:
    /**
     * @brief Construct a worker pool.
     * @param num_workers Number of worker threads to spawn.
     * @param device      Device that workers will target.
     */
    WorkerPool(int num_workers, Device device);

    /** @brief Joins all worker threads (calls shutdown() if still running). */
    ~WorkerPool();

    WorkerPool(const WorkerPool&) = delete;
    auto operator=(const WorkerPool&) -> WorkerPool& = delete;

    /**
     * @brief Submit a task for asynchronous execution.
     * @param task Callable to execute on a worker thread.
     */
    auto submit(std::function<void()> task) -> void;

    /**
     * @brief Gracefully stop all workers.
     *
     * Signals workers to drain remaining tasks and exit. Blocks until all
     * threads have joined.
     */
    auto shutdown() -> void;

private:
    std::vector<std::thread> workers_;
    std::queue<std::function<void()>> task_queue_;
    std::mutex mutex_;
    std::condition_variable cv_;
    std::atomic<bool> running_{false};
    Device device_;
};

} // namespace tenzor::serving
