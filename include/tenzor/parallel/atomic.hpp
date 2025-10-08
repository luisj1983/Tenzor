#pragma once

#include <atomic>
#include <memory>

namespace tenzor {

// Lock-free atomic operations

// Atomic add
template<typename T>
inline auto atomic_add(std::atomic<T>& target, T value) -> T {
    return target.fetch_add(value, std::memory_order_relaxed);
}

// Atomic compare-and-swap
template<typename T>
inline auto atomic_cas(std::atomic<T>& target, T expected, T desired) -> bool {
    return target.compare_exchange_weak(expected, desired,
                                       std::memory_order_release,
                                       std::memory_order_relaxed);
}

// Spin lock
class SpinLock {
public:
    SpinLock() = default;

    auto lock() -> void {
        while (flag_.test_and_set(std::memory_order_acquire)) {
            // Spin
            #if defined(__x86_64__) || defined(_M_X64)
            __builtin_ia32_pause();
            #elif defined(__aarch64__)
            asm volatile("yield" ::: "memory");
            #endif
        }
    }

    auto unlock() -> void {
        flag_.clear(std::memory_order_release);
    }

    auto try_lock() -> bool {
        return !flag_.test_and_set(std::memory_order_acquire);
    }

private:
    std::atomic_flag flag_ = ATOMIC_FLAG_INIT;
};

} // namespace tenzor
