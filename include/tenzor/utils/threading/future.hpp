/**
 * @file future.hpp
 * @brief Future/Promise pattern for asynchronous tensor operations
 *
 * Provides a custom Future<T> implementation for non-blocking tensor operations
 * with continuation support and exception propagation.
 */

#pragma once

#include <future>
#include <memory>
#include <functional>
#include <stdexcept>
#include <atomic>
#include <mutex>
#include <condition_variable>
#include <exception>

namespace tenzor {

/**
 * @brief Shared state between Future and Promise
 *
 * Thread-safe state container that holds the computation result,
 * exception, and continuation chain.
 *
 * @tparam T Result type
 */
template<typename T>
class SharedState {
public:
    SharedState() = default;

    /**
     * @brief Drain any pending continuations on destruction.
     *
     * If the state was never satisfied (the owning Promise was abandoned),
     * mark it broken and run the registered continuations so chained futures
     * receive a broken_promise exception instead of leaking. Because the
     * continuations are invoked with a pointer to `this` rather than capturing
     * a shared_ptr back to this state, there is no reference cycle to break and
     * `this` is still fully alive for the duration of the destructor body.
     */
    ~SharedState() {
        break_if_pending();
    }

    /**
     * @brief Mark an unsatisfied state as broken (idempotent).
     *
     * Called from the abandoning Promise's destructor (so blocked waiters and
     * chained continuations are released promptly) and as a last resort from
     * this state's own destructor. If the state is already ready this is a
     * no-op. Wakes get() waiters and drains continuations with a
     * broken_promise exception.
     */
    void break_if_pending() {
        {
            std::unique_lock lock(mutex_);
            if (ready_) {
                return;
            }
            exception_ = std::make_exception_ptr(
                std::future_error(std::future_errc::broken_promise));
            ready_ = true;
        }
        cv_.notify_all();
        execute_continuations();
    }

    /**
     * @brief Set the result value
     * @param value Result to store
     * @throws std::logic_error if already set
     */
    void set_value(T value) {
        std::unique_lock lock(mutex_);
        if (ready_) {
            throw std::logic_error("Promise already satisfied");
        }
        value_ = std::move(value);
        ready_ = true;
        lock.unlock();
        cv_.notify_all();

        // Execute continuations
        execute_continuations();
    }

    /**
     * @brief Set exception instead of value
     * @param exception Exception to store
     */
    void set_exception(std::exception_ptr exception) {
        std::unique_lock lock(mutex_);
        if (ready_) {
            throw std::logic_error("Promise already satisfied");
        }
        exception_ = exception;
        ready_ = true;
        lock.unlock();
        cv_.notify_all();

        // Execute continuations
        execute_continuations();
    }

    /**
     * @brief Wait for result and return it
     * @return The stored value
     * @throws Rethrows stored exception if set
     */
    T get() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return ready_.load(); });

        if (exception_) {
            std::rethrow_exception(exception_);
        }

        return value_;
    }

    /**
     * @brief Read the result of a state already known to be ready.
     *
     * Used by continuations, which only run once the state is ready, to avoid
     * the redundant condition-variable wait that get() incurs. Returns a copy
     * of the stored value rather than moving it out: a single shared state may
     * fan out to multiple continuations (and may additionally be observed via
     * get()/wait()), so every consumer must see the real result. Moving the
     * value out would corrupt the second and later consumers. Rethrows the
     * stored exception if one is set.
     */
    T take() {
        std::unique_lock lock(mutex_);
        if (exception_) {
            std::rethrow_exception(exception_);
        }
        return value_;
    }

    /**
     * @brief Check if result is ready
     * @return true if computation completed
     */
    bool is_ready() const {
        return ready_.load();
    }

    /**
     * @brief Add continuation callback
     *
     * The callback receives a pointer to this ready state so it can take the
     * value directly without re-locking through get(). Storing the callback
     * here (rather than having it capture a shared_ptr back to this state)
     * avoids a self-referential ownership cycle.
     *
     * @param callback Function to call when ready
     */
    void add_continuation(std::function<void(SharedState*)> callback) {
        std::unique_lock lock(mutex_);
        if (ready_) {
            // Already ready, execute immediately
            lock.unlock();
            callback(this);
        } else {
            continuations_.push_back(std::move(callback));
        }
    }

private:
    void execute_continuations() {
        std::unique_lock lock(mutex_);
        auto callbacks = std::move(continuations_);
        lock.unlock();

        for (auto& callback : callbacks) {
            callback(this);
        }
    }

    T value_;
    std::exception_ptr exception_;
    std::atomic<bool> ready_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::function<void(SharedState*)>> continuations_;
};

/**
 * @brief Promise side of Future/Promise pair
 *
 * Used to set the result of an asynchronous computation.
 *
 * @tparam T Result type
 */
template<typename T>
class Promise {
public:
    /**
     * @brief Construct Promise with shared state
     */
    Promise() : state_(std::make_shared<SharedState<T>>()) {}

    // Move-only: a single producer owns the obligation to satisfy the state, so
    // its destruction (without satisfying it) breaks the promise exactly once.
    Promise(const Promise&) = delete;
    Promise& operator=(const Promise&) = delete;
    Promise(Promise&&) noexcept = default;
    Promise& operator=(Promise&&) noexcept = default;

    /**
     * @brief Break the shared state if this producer is abandoned unsatisfied.
     */
    ~Promise() {
        if (state_) {
            state_->break_if_pending();
        }
    }

    /**
     * @brief Set the promise value
     * @param value Result to store
     */
    void set_value(T value) {
        state_->set_value(std::move(value));
    }

    /**
     * @brief Set promise exception
     * @param exception Exception to store
     */
    void set_exception(std::exception_ptr exception) {
        state_->set_exception(exception);
    }

    /**
     * @brief Get shared state pointer
     * @return Shared state
     */
    auto get_state() -> std::shared_ptr<SharedState<T>> {
        return state_;
    }

private:
    std::shared_ptr<SharedState<T>> state_;
};

/**
 * @brief Future for asynchronous tensor operations
 *
 * Represents a value that will be available in the future.
 * Supports waiting, non-blocking checks, and continuation chaining.
 *
 * **Features:**
 * - wait() - block until completion
 * - then() - attach callbacks (continuation pattern)
 * - is_ready() - non-blocking status check
 * - Exception propagation through future chain
 * - Thread-safe state management
 *
 * **Use Cases:**
 * - Asynchronous tensor operations (matmul, conv2d)
 * - Pipeline operations with continuations
 * - Overlapping computation and communication
 *
 * @tparam T Result type
 *
 * @code
 * // Basic usage
 * auto future = async_matmul(a, b);
 * Tensor result = future.wait();
 *
 * // Continuation chaining
 * auto future = async_matmul(a, b)
 *     .then([](Tensor result) {
 *         return relu(result);
 *     })
 *     .then([](Tensor activated) {
 *         return softmax(activated);
 *     });
 * @endcode
 */
template<typename T>
class Future {
public:
    /**
     * @brief Construct Future from shared state
     * @param state Shared state with Promise
     */
    explicit Future(std::shared_ptr<SharedState<T>> state)
        : state_(std::move(state)) {}

    /**
     * @brief Wait for computation to complete and return result
     *
     * Blocks the calling thread until the result is available.
     *
     * @return The computed value
     * @throws Rethrows any exception from the computation
     *
     * @code
     * Future<Tensor> future = async_matmul(a, b);
     * Tensor result = future.wait();  // Blocks here
     * @endcode
     */
    auto wait() -> T {
        if (!state_) {
            throw std::logic_error("Future has no shared state");
        }
        return state_->get();
    }

    /**
     * @brief Attach continuation callback
     *
     * Registers a callback to be executed when the future becomes ready.
     * The callback receives the result value as input.
     *
     * **Continuation Chaining:**
     * Multiple then() calls can be chained to build operation pipelines.
     *
     * @tparam F Callback function type
     * @param callback Function with signature void(T) or R(T)
     * @return Future<R> for chaining, or Future<void> if callback returns void
     *
     * @code
     * // Simple continuation
     * future.then([](Tensor t) {
     *     std::cout << "Result ready: " << t.shape() << std::endl;
     * });
     *
     * // Chained continuations
     * auto final = async_matmul(a, b)
     *     .then([](Tensor t) { return relu(t); })
     *     .then([](Tensor t) { return dropout(t, 0.5); });
     * @endcode
     */
    template<typename F>
    auto then(F&& callback) -> Future<std::invoke_result_t<F, T>> {
        using R = std::invoke_result_t<F, T>;

        auto promise = std::make_shared<Promise<R>>();
        auto next_state = promise->get_state();

        state_->add_continuation([promise, callback = std::forward<F>(callback)](SharedState<T>* st) mutable {
            try {
                // Take the value directly; the state is ready by the time a
                // continuation runs, so this avoids get()'s cv wait + copy and
                // rethrows any stored exception.
                T value = st->take();

                // Execute callback and set result
                if constexpr (std::is_void_v<R>) {
                    callback(std::move(value));
                    promise->set_value(R{});  // For void, use default-constructed value
                } else {
                    R result = callback(std::move(value));
                    promise->set_value(std::move(result));
                }
            } catch (...) {
                // Propagate exception to next future
                promise->set_exception(std::current_exception());
            }
        });

        return Future<R>(next_state);
    }

    /**
     * @brief Check if computation is complete (non-blocking)
     *
     * @return true if result is available, false otherwise
     *
     * @code
     * Future<Tensor> future = async_matmul(a, b);
     *
     * // Do other work
     * process_data();
     *
     * // Check if ready without blocking
     * if (future.is_ready()) {
     *     Tensor result = future.wait();
     * }
     * @endcode
     */
    auto is_ready() const -> bool {
        if (!state_) {
            return false;
        }
        return state_->is_ready();
    }

    /**
     * @brief Check if future has valid shared state
     * @return true if future is valid
     */
    auto valid() const -> bool {
        return state_ != nullptr;
    }

private:
    std::shared_ptr<SharedState<T>> state_;
};

/**
 * @brief Specialization of SharedState for void type
 */
template<>
class SharedState<void> {
public:
    SharedState() = default;

    ~SharedState() {
        break_if_pending();
    }

    void break_if_pending() {
        {
            std::unique_lock lock(mutex_);
            if (ready_) {
                return;
            }
            exception_ = std::make_exception_ptr(
                std::future_error(std::future_errc::broken_promise));
            ready_ = true;
        }
        cv_.notify_all();
        execute_continuations();
    }

    void set_value() {
        std::unique_lock lock(mutex_);
        if (ready_) {
            throw std::logic_error("Promise already satisfied");
        }
        ready_ = true;
        lock.unlock();
        cv_.notify_all();
        execute_continuations();
    }

    void set_exception(std::exception_ptr exception) {
        std::unique_lock lock(mutex_);
        if (ready_) {
            throw std::logic_error("Promise already satisfied");
        }
        exception_ = exception;
        ready_ = true;
        lock.unlock();
        cv_.notify_all();
        execute_continuations();
    }

    void get() {
        std::unique_lock lock(mutex_);
        cv_.wait(lock, [this] { return ready_.load(); });

        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

    /**
     * @brief Rethrow the stored exception (if any) for an already-ready state.
     *
     * Used by continuations to skip get()'s condition-variable wait.
     */
    void take() {
        std::unique_lock lock(mutex_);
        if (exception_) {
            std::rethrow_exception(exception_);
        }
    }

    bool is_ready() const {
        return ready_.load();
    }

    void add_continuation(std::function<void(SharedState*)> callback) {
        std::unique_lock lock(mutex_);
        if (ready_) {
            lock.unlock();
            callback(this);
        } else {
            continuations_.push_back(std::move(callback));
        }
    }

private:
    void execute_continuations() {
        std::unique_lock lock(mutex_);
        auto callbacks = std::move(continuations_);
        lock.unlock();

        for (auto& callback : callbacks) {
            callback(this);
        }
    }

    std::exception_ptr exception_;
    std::atomic<bool> ready_{false};
    std::mutex mutex_;
    std::condition_variable cv_;
    std::vector<std::function<void(SharedState*)>> continuations_;
};

/**
 * @brief Specialization of Promise for void type
 */
template<>
class Promise<void> {
public:
    Promise() : state_(std::make_shared<SharedState<void>>()) {}

    Promise(const Promise&) = delete;
    Promise& operator=(const Promise&) = delete;
    Promise(Promise&&) noexcept = default;
    Promise& operator=(Promise&&) noexcept = default;

    ~Promise() {
        if (state_) {
            state_->break_if_pending();
        }
    }

    void set_value() {
        state_->set_value();
    }

    void set_exception(std::exception_ptr exception) {
        state_->set_exception(exception);
    }

    auto get_state() -> std::shared_ptr<SharedState<void>> {
        return state_;
    }

private:
    std::shared_ptr<SharedState<void>> state_;
};

/**
 * @brief Specialization of Future for void type
 */
template<>
class Future<void> {
public:
    explicit Future(std::shared_ptr<SharedState<void>> state)
        : state_(std::move(state)) {}

    auto wait() -> void {
        if (!state_) {
            throw std::logic_error("Future has no shared state");
        }
        state_->get();
    }

    template<typename F>
    auto then(F&& callback) -> Future<std::invoke_result_t<F>> {
        using R = std::invoke_result_t<F>;

        auto promise = std::make_shared<Promise<R>>();
        auto next_state = promise->get_state();

        state_->add_continuation([promise, callback = std::forward<F>(callback)](SharedState<void>* st) mutable {
            try {
                st->take();

                if constexpr (std::is_void_v<R>) {
                    callback();
                    promise->set_value();
                } else {
                    R result = callback();
                    promise->set_value(std::move(result));
                }
            } catch (...) {
                promise->set_exception(std::current_exception());
            }
        });

        return Future<R>(next_state);
    }

    auto is_ready() const -> bool {
        if (!state_) {
            return false;
        }
        return state_->is_ready();
    }

    auto valid() const -> bool {
        return state_ != nullptr;
    }

private:
    std::shared_ptr<SharedState<void>> state_;
};

} // namespace tenzor
