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

// Work-stealing thread pool
class ThreadPool {
public:
    explicit ThreadPool(size_t num_threads = std::thread::hardware_concurrency());
    ~ThreadPool();

    ThreadPool(const ThreadPool&) = delete;
    ThreadPool& operator=(const ThreadPool&) = delete;

    // Submit task
    template<typename F, typename... Args>
    auto submit(F&& func, Args&&... args) -> std::future<std::invoke_result_t<F, Args...>>;

    // Parallel for loop
    template<typename F>
    auto parallel_for(int64_t begin, int64_t end, F&& func) -> void;

    // Get number of threads
    auto num_threads() const -> size_t { return num_threads_; }

    // Get active thread count
    auto active_threads() const -> size_t;

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

template<typename F>
auto ThreadPool::parallel_for(int64_t begin, int64_t end, F&& func) -> void {
    if (begin >= end) return;

    const size_t num_tasks = std::min<size_t>(end - begin, num_threads_ * 4);
    const int64_t chunk_size = (end - begin + num_tasks - 1) / num_tasks;

    std::vector<std::future<void>> futures;
    futures.reserve(num_tasks);

    for (size_t i = 0; i < num_tasks; ++i) {
        int64_t start = begin + i * chunk_size;
        int64_t finish = std::min(start + chunk_size, end);

        futures.push_back(submit([&func, start, finish]() {
            for (int64_t j = start; j < finish; ++j) {
                func(j);
            }
        }));
    }

    for (auto& future : futures) {
        future.wait();
    }
}

// Global thread pool
auto thread_pool() -> ThreadPool&;

} // namespace tenzor
