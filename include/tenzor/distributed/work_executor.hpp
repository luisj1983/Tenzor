/**
 * @file work_executor.hpp
 * @brief B.3: Single-worker thread-pool for Gloo-backed async collectives.
 *
 * The CommunicationBackend's async API doesn't return a Work handle (the
 * existing contract is `void all_reduce_async(Tensor&, ReduceOp, void*
 * stream)`), so true overlap requires the caller to call `wait_pending()`
 * before depending on the operation's result. The executor runs each
 * enqueued sync collective on a dedicated worker thread so the caller
 * returns immediately; pending work is drained on `wait_pending()` or
 * implicitly at executor destruction.
 *
 * This is not a full Work-handle implementation — that would require a
 * public API change. It IS a real async mechanism: the caller's thread
 * proceeds while the collective runs on the worker, and DDP-style
 * overlap-with-compute patterns can use it via the `wait_pending()`
 * barrier before the optimizer step.
 */

#pragma once

#include <atomic>
#include <condition_variable>
#include <deque>
#include <functional>
#include <mutex>
#include <thread>

namespace tenzor::distributed {

class WorkExecutor {
public:
    WorkExecutor() : stop_(false) {
        worker_ = std::thread([this] { worker_loop(); });
    }

    ~WorkExecutor() {
        {
            std::lock_guard<std::mutex> lk(mu_);
            stop_ = true;
        }
        cv_.notify_all();
        if (worker_.joinable()) worker_.join();
    }

    WorkExecutor(const WorkExecutor&) = delete;
    auto operator=(const WorkExecutor&) -> WorkExecutor& = delete;

    /// Enqueue a sync collective. Returns immediately; the function runs
    /// on the worker thread. Exceptions thrown by `fn` are captured and
    /// re-thrown on `wait_pending()`.
    auto enqueue(std::function<void()> fn) -> void {
        {
            std::lock_guard<std::mutex> lk(mu_);
            queue_.push_back(std::move(fn));
            pending_count_.fetch_add(1, std::memory_order_release);
        }
        cv_.notify_one();
    }

    /// Block until every pending task has completed. Re-throws the first
    /// exception captured from any task.
    auto wait_pending() -> void {
        std::unique_lock<std::mutex> lk(mu_);
        done_cv_.wait(lk, [this] {
            return pending_count_.load(std::memory_order_acquire) == 0;
        });
        if (captured_exception_) {
            auto e = std::move(captured_exception_);
            captured_exception_ = nullptr;
            std::rethrow_exception(e);
        }
    }

    /// True if any enqueued tasks haven't finished.
    auto has_pending() const -> bool {
        return pending_count_.load(std::memory_order_acquire) > 0;
    }

private:
    auto worker_loop() -> void {
        for (;;) {
            std::function<void()> task;
            {
                std::unique_lock<std::mutex> lk(mu_);
                cv_.wait(lk, [this] { return stop_ || !queue_.empty(); });
                if (stop_ && queue_.empty()) return;
                task = std::move(queue_.front());
                queue_.pop_front();
            }
            std::exception_ptr task_exc{nullptr};
            try {
                task();
            } catch (...) {
                task_exc = std::current_exception();
            }
            // Decrement the predicate variable and capture any exception under
            // mu_ so the predicate transition is visible to a waiter that has
            // already acquired the lock. Notifying outside the lock is fine, but
            // the state change MUST be mutex-protected or a notify landing in
            // wait_pending()'s registration window is lost (missed wakeup).
            {
                std::lock_guard<std::mutex> lk(mu_);
                if (task_exc && !captured_exception_) {
                    captured_exception_ = task_exc;
                }
                pending_count_.fetch_sub(1, std::memory_order_release);
            }
            done_cv_.notify_all();
        }
    }

    mutable std::mutex mu_;
    std::condition_variable cv_;
    std::condition_variable done_cv_;
    std::deque<std::function<void()>> queue_;
    std::atomic<int64_t> pending_count_{0};
    std::exception_ptr captured_exception_{nullptr};
    bool stop_;
    std::thread worker_;
};

}  // namespace tenzor::distributed
