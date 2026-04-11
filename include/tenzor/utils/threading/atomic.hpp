/**
 * @file atomic.hpp
 * @brief Lock-free atomic operations for thread-safe tensor operations
 *
 * Provides atomic primitives and synchronization mechanisms for concurrent
 * tensor operations on CPU.
 */

#pragma once

#include <atomic>
#include <memory>

namespace tenzor {

/**
 * @brief Atomic add operation with relaxed memory ordering
 *
 * Thread-safe addition to atomic variable. Returns previous value.
 *
 * @tparam T Numeric type (int, float, etc.)
 * @param target Atomic variable to modify
 * @param value Value to add
 * @return Previous value before addition
 *
 * @par Complexity
 * O(1) lock-free operation
 *
 * @par Thread Safety
 * Thread-safe, lock-free
 */
template<typename T>
inline auto atomic_add(std::atomic<T>& target, T value) -> T {
    return target.fetch_add(value, std::memory_order_relaxed);
}

/**
 * @brief Atomic compare-and-swap operation
 *
 * Atomically compares target with expected, and if equal, replaces with desired.
 * Uses weak semantics (may spuriously fail) for better performance.
 *
 * @tparam T Type of values to compare
 * @param target Atomic variable to modify
 * @param expected Expected current value
 * @param desired New value if comparison succeeds
 * @return true if swap occurred, false otherwise
 *
 * @par Complexity
 * O(1) lock-free operation
 *
 * @par Thread Safety
 * Thread-safe, lock-free
 */
template<typename T>
inline auto atomic_cas(std::atomic<T>& target, T expected, T desired) -> bool {
    return target.compare_exchange_weak(expected, desired,
                                       std::memory_order_release,
                                       std::memory_order_relaxed);
}

/**
 * @brief Lightweight spin lock for short critical sections
 *
 * Spin lock that actively spins (busy-waits) until lock is acquired.
 * Suitable for protecting very short critical sections where blocking
 * would be more expensive than spinning.
 *
 * **When to Use:**
 * - Critical sections < 100 CPU cycles
 * - Low contention scenarios
 * - Real-time requirements
 *
 * **When NOT to Use:**
 * - Long critical sections (use std::mutex)
 * - High contention (causes CPU waste)
 * - I/O operations in critical section
 *
 * @par Thread Safety
 * Thread-safe
 *
 * @code
 * SpinLock lock;
 * // ... in thread:
 * lock.lock();
 * // Critical section
 * lock.unlock();
 * @endcode
 */
class SpinLock {
public:
    SpinLock() = default;

    /**
     * @brief Acquire the lock (blocks until available)
     *
     * Spins until lock is acquired. Uses CPU pause instructions on x86/ARM
     * to reduce power consumption and improve performance.
     */
    auto lock() -> void {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin with CPU hints for better performance
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();  // x86 PAUSE instruction
            #elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");  // ARM YIELD instruction
            #endif
        }
    }

    /**
     * @brief Release the lock
     */
    auto unlock() -> void {
        flag_.clear(std::memory_order_release);
    }

    /**
     * @brief Try to acquire lock without blocking
     * @return true if lock acquired, false if already locked
     */
    auto try_lock() -> bool {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

} // namespace tenzor
