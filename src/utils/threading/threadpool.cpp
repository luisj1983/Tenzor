#include "tenzor/utils/threading/threadpool.hpp"

#include <algorithm>

namespace tenzor {

ThreadPool::ThreadPool(size_t num_threads)
    // std::thread::hardware_concurrency() is allowed to return 0 (some
    // containers/cgroups/CI). With 0 workers there is no consumer for the task
    // queue, so any future returned by submit() would block forever. Clamp to a
    // minimum of one worker so the pool always makes progress.
    : num_threads_(std::max<size_t>(1, num_threads)) {

    workers_.reserve(num_threads_);

    for (size_t i = 0; i < num_threads_; ++i) {
        workers_.emplace_back(&ThreadPool::worker_thread, this);
    }
}

ThreadPool::~ThreadPool() {
    {
        std::unique_lock lock(queue_mutex_);
        stop_ = true;
    }

    condition_.notify_all();

    for (auto& worker : workers_) {
        if (worker.joinable()) {
            worker.join();
        }
    }
}

auto ThreadPool::worker_thread() -> void {
    while (true) {
        std::function<void()> task;

        {
            std::unique_lock lock(queue_mutex_);

            condition_.wait(lock, [this] {
                return stop_ || !tasks_.empty();
            });

            if (stop_ && tasks_.empty()) {
                return;
            }

            task = std::move(tasks_.front());
            tasks_.pop();
        }

        task();
    }
}

// Global thread pool
auto thread_pool() -> ThreadPool& {
    static ThreadPool pool;
    return pool;
}

} // namespace tenzor
