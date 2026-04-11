#include "tenzor/utils/threading/threadpool.hpp"

namespace tenzor {

ThreadPool::ThreadPool(size_t num_threads)
    : num_threads_(num_threads) {

    workers_.reserve(num_threads);

    for (size_t i = 0; i < num_threads; ++i) {
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

        active_threads_++;
        task();
        active_threads_--;
    }
}

auto ThreadPool::active_threads() const -> size_t {
    return active_threads_.load();
}

// Global thread pool
auto thread_pool() -> ThreadPool& {
    static ThreadPool pool;
    return pool;
}

} // namespace tenzor
