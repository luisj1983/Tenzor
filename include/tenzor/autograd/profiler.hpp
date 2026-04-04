/**
 * @file profiler.hpp
 * @brief Autograd backward pass profiling infrastructure
 *
 * Provides per-function timing instrumentation for the backward pass.
 * Zero overhead when disabled (single atomic bool check).
 *
 * @code
 * auto& prof = AutogradProfiler::instance();
 * prof.enable();
 * loss.backward();
 * prof.disable();
 * std::cout << prof.summary() << std::endl;
 * prof.reset();
 * @endcode
 */

#pragma once

#include <atomic>
#include <chrono>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor {

/**
 * @brief Profile data for a single backward function type.
 */
struct BackwardProfile {
    std::string function_name;
    std::chrono::nanoseconds total_time{0};
    int64_t call_count{0};
};

/**
 * @brief Singleton profiler for autograd backward pass timing.
 *
 * When enabled, records wall-clock time for each Function::backward()
 * call during backpropagation. Results are aggregated by function name.
 *
 * @par Thread Safety
 * Thread-safe. Each thread accumulates into the shared map under a mutex.
 * The is_enabled() check is lock-free (atomic bool).
 */
class AutogradProfiler {
public:
    static auto instance() -> AutogradProfiler& {
        static AutogradProfiler prof;
        return prof;
    }

    /// Enable profiling. Subsequent backward() calls will be timed.
    auto enable() -> void { enabled_.store(true, std::memory_order_release); }

    /// Disable profiling.
    auto disable() -> void { enabled_.store(false, std::memory_order_release); }

    /// Check if profiling is enabled (lock-free).
    [[nodiscard]] auto is_enabled() const noexcept -> bool {
        return enabled_.load(std::memory_order_acquire);
    }

    /// Clear all recorded data.
    auto reset() -> void {
        std::lock_guard lock(mutex_);
        data_.clear();
    }

    /// Record a backward function execution.
    auto record(const std::string& name, std::chrono::nanoseconds elapsed) -> void {
        std::lock_guard lock(mutex_);
        auto& entry = data_[name];
        if (entry.function_name.empty()) {
            entry.function_name = name;
        }
        entry.total_time += elapsed;
        entry.call_count++;
    }

    /// Get all recorded profiles, sorted by total time descending.
    [[nodiscard]] auto profiles() const -> std::vector<BackwardProfile> {
        std::lock_guard lock(mutex_);
        std::vector<BackwardProfile> result;
        result.reserve(data_.size());
        for (const auto& [_, entry] : data_) {
            result.push_back(entry);
        }
        std::sort(result.begin(), result.end(),
                  [](const BackwardProfile& a, const BackwardProfile& b) {
                      return a.total_time > b.total_time;
                  });
        return result;
    }

    /// Human-readable summary string.
    [[nodiscard]] auto summary() const -> std::string {
        auto profs = profiles();
        if (profs.empty()) return "No backward profiles recorded.\n";

        std::string out = "Autograd Backward Profile:\n";
        out += "  Function                          Calls    Total (ms)   Avg (us)\n";
        out += "  --------------------------------  -------  -----------  --------\n";
        for (const auto& p : profs) {
            auto total_ms = std::chrono::duration<double, std::milli>(p.total_time).count();
            auto avg_us = p.call_count > 0
                ? std::chrono::duration<double, std::micro>(p.total_time).count() / p.call_count
                : 0.0;
            char line[128];
            std::snprintf(line, sizeof(line), "  %-34s %7lld  %11.3f  %8.1f\n",
                         p.function_name.c_str(),
                         static_cast<long long>(p.call_count),
                         total_ms, avg_us);
            out += line;
        }
        return out;
    }

private:
    AutogradProfiler() = default;
    std::atomic<bool> enabled_{false};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, BackwardProfile> data_;
};

/**
 * @brief RAII timer that records to AutogradProfiler on destruction.
 *
 * Construct at the start of a backward call; the destructor records
 * the elapsed time. Zero cost when profiler is disabled (never constructed).
 */
class BackwardTimer {
public:
    explicit BackwardTimer(const std::string& name)
        : name_(name), start_(std::chrono::steady_clock::now()) {}

    ~BackwardTimer() {
        auto elapsed = std::chrono::steady_clock::now() - start_;
        AutogradProfiler::instance().record(
            name_, std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed));
    }

    BackwardTimer(const BackwardTimer&) = delete;
    BackwardTimer& operator=(const BackwardTimer&) = delete;

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace tenzor
