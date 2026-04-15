/**
 * @file profiling_interceptor.hpp
 * @brief Forward-pass operation profiling via dispatch interceptor.
 *
 * Complements the backward-only AutogradProfiler with forward-pass
 * timing. Push this interceptor to record wall-clock time per OpId.
 *
 * Usage:
 * @code
 * OpProfiler::instance().enable();
 * // ... run forward pass ...
 * OpProfiler::instance().disable();
 * std::cout << OpProfiler::instance().summary();
 * @endcode
 */

#pragma once

#include "dispatch_interceptor.hpp"
#include "../ops/op_id.hpp"
#include "../autograd/profiler.hpp"
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>
#include <algorithm>

namespace tenzor {

/**
 * @brief Profile entry for a single forward-pass operation.
 */
struct OpProfile {
    OpId op;
    std::chrono::nanoseconds total_time{0};
    int64_t call_count{0};
};

/**
 * @brief Singleton profiler for forward-pass dispatch timing.
 *
 * When enabled, the profiling interceptor records wall-clock time
 * for each dispatched operation. Results are aggregated by OpId.
 *
 * Thread-safe. Lock-free enabled check, mutex-guarded recording.
 */
class OpProfiler {
public:
    static auto instance() -> OpProfiler& {
        static OpProfiler prof;
        return prof;
    }

    auto enable() -> void {
        enabled_.store(true, std::memory_order_release);
    }

    auto disable() -> void {
        enabled_.store(false, std::memory_order_release);
    }

    [[nodiscard]] auto is_enabled() const noexcept -> bool {
        return enabled_.load(std::memory_order_acquire);
    }

    auto reset() -> void {
        std::lock_guard lock(mutex_);
        data_.clear();
    }

    auto record(OpId op, std::chrono::nanoseconds elapsed) -> void {
        std::lock_guard lock(mutex_);
        auto& entry = data_[op];
        entry.op = op;
        entry.total_time += elapsed;
        entry.call_count++;
    }

    [[nodiscard]] auto profiles() const -> std::vector<OpProfile> {
        std::lock_guard lock(mutex_);
        std::vector<OpProfile> result;
        result.reserve(data_.size());
        for (const auto& [_, entry] : data_) {
            result.push_back(entry);
        }
        std::sort(result.begin(), result.end(),
                  [](const OpProfile& a, const OpProfile& b) {
                      return a.total_time > b.total_time;
                  });
        return result;
    }

    [[nodiscard]] auto summary() const -> std::string {
        auto profs = profiles();
        if (profs.empty()) return "No forward-pass profiles recorded.\n";

        std::string out = "Forward-Pass Op Profile:\n";
        out += "  OpId   Calls    Total (ms)   Avg (us)\n";
        out += "  -----  -------  -----------  --------\n";
        for (const auto& p : profs) {
            auto total_ms = std::chrono::duration<double, std::milli>(p.total_time).count();
            auto avg_us = p.call_count > 0
                ? std::chrono::duration<double, std::micro>(p.total_time).count() / p.call_count
                : 0.0;
            char line[128];
            std::snprintf(line, sizeof(line), "  %5d  %7lld  %11.3f  %8.1f\n",
                         static_cast<int>(p.op),
                         static_cast<long long>(p.call_count),
                         total_ms, avg_us);
            out += line;
        }
        return out;
    }

private:
    OpProfiler() = default;
    std::atomic<bool> enabled_{false};
    mutable std::mutex mutex_;
    std::unordered_map<OpId, OpProfile> data_;
};

/**
 * @brief Create a dispatch interceptor that records per-op timing.
 *
 * Only records when OpProfiler::instance().is_enabled() is true.
 * When disabled, calls next() with zero overhead beyond the stack check.
 */
inline DispatchInterceptor make_profiling_interceptor() {
    return [](OpId op, std::span<const Tensor> inputs,
              const OpAttributes& attrs, DispatchNext next) -> std::vector<Tensor> {
        if (!OpProfiler::instance().is_enabled()) {
            return next(op, inputs, attrs);
        }

        auto start = std::chrono::steady_clock::now();
        auto result = next(op, inputs, attrs);
        auto end = std::chrono::steady_clock::now();

        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);
        OpProfiler::instance().record(op, ns);

        // Record per-invocation trace event when trace mode is active
        if (AutogradProfiler::instance().is_trace_enabled()) {
            AutogradProfiler::instance().record_trace(
                "OpId:" + std::to_string(static_cast<int>(op)),
                start, ns, ProfilePhase::Forward);
        }

        return result;
    };
}

/**
 * @brief RAII guard that pushes the profiling interceptor for its lifetime.
 *
 * @code
 * {
 *     ProfilingInterceptorGuard guard;
 *     OpProfiler::instance().enable();
 *     // ... forward pass ...
 *     OpProfiler::instance().disable();
 * }
 * std::cout << OpProfiler::instance().summary();
 * @endcode
 */
class ProfilingInterceptorGuard {
public:
    ProfilingInterceptorGuard() {
        DispatchInterceptorStack::push(make_profiling_interceptor());
    }
    ~ProfilingInterceptorGuard() {
        DispatchInterceptorStack::pop();
    }
    ProfilingInterceptorGuard(const ProfilingInterceptorGuard&) = delete;
    ProfilingInterceptorGuard& operator=(const ProfilingInterceptorGuard&) = delete;
};

} // namespace tenzor
