/**
 * @file worker_pool.cpp
 * @brief Implementation of serving::WorkerPool (audit-2026-05-03 N4).
 *
 * Standard fixed-size thread pool with:
 *   - submit(task) enqueues work to be picked up by any worker.
 *   - shutdown() drains the queue and joins all workers.
 *   - Destructor calls shutdown() if still running.
 */

#include <tenzor/serving/worker_pool.hpp>
#include <tenzor/core/device_guard.hpp>

namespace tenzor::serving {

WorkerPool::WorkerPool(int num_workers, Device device) : device_(device) {
    running_.store(true, std::memory_order_release);
    workers_.reserve(static_cast<size_t>(num_workers));
    for (int i = 0; i < num_workers; ++i) {
        workers_.emplace_back([this] {
            // Bind this worker thread to the pool's device so tensors created
            // inside submitted tasks default to the correct backend (the
            // documented WorkerPool contract). Device state is thread-local,
            // so this affects only this worker thread. CPU is index 0 / no-op.
            detail::switch_device(device_.type, device_.index);
            for (;;) {
                std::function<void()> task;
                {
                    std::unique_lock<std::mutex> lk(mutex_);
                    cv_.wait(lk, [this] {
                        return !running_.load(std::memory_order_acquire) ||
                               !task_queue_.empty();
                    });
                    if (!running_.load(std::memory_order_acquire) &&
                        task_queue_.empty()) {
                        return;
                    }
                    task = std::move(task_queue_.front());
                    task_queue_.pop();
                }
                if (task) task();
            }
        });
    }
}

WorkerPool::~WorkerPool() {
    if (running_.load(std::memory_order_acquire)) {
        shutdown();
    }
}

auto WorkerPool::submit(std::function<void()> task) -> void {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        task_queue_.push(std::move(task));
    }
    cv_.notify_one();
}

auto WorkerPool::shutdown() -> void {
    {
        std::lock_guard<std::mutex> lk(mutex_);
        running_.store(false, std::memory_order_release);
    }
    cv_.notify_all();
    for (auto& w : workers_) {
        if (w.joinable()) w.join();
    }
    workers_.clear();
}

} // namespace tenzor::serving
