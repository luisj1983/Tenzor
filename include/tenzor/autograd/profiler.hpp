/**
 * @file profiler.hpp
 * @brief Unified forward/backward pass profiling infrastructure
 *
 * Provides per-operation timing instrumentation for both forward and backward passes.
 * Zero overhead when disabled (single atomic bool check).
 *
 * @code
 * auto& prof = AutogradProfiler::instance();
 * prof.enable();
 * auto y = model.forward(x);  // forward ops profiled
 * loss.backward();             // backward ops profiled
 * prof.disable();
 * std::cout << prof.summary() << std::endl;
 * prof.reset();
 * @endcode
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

namespace tenzor {

/// Phase of computation being profiled
enum class ProfilePhase : uint8_t {
    Forward,
    Backward
};

/**
 * @brief Profile data for a single operation/function type.
 */
struct AutogradProfile {
    std::string name;
    ProfilePhase phase{ProfilePhase::Backward};
    std::chrono::nanoseconds total_time{0};
    int64_t call_count{0};
};

/**
 * @brief A single trace event for Chrome Trace Event Format export.
 *
 * Captures per-invocation timing (as opposed to AutogradProfile which
 * aggregates). Only recorded when trace mode is active.
 */
struct TraceEvent {
    std::string name;
    ProfilePhase phase{ProfilePhase::Forward};
    std::chrono::steady_clock::time_point start;
    std::chrono::nanoseconds duration{0};
    uint32_t thread_id{0};
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

    /// Enable trace mode. When active, individual TraceEvents are recorded
    /// in addition to the aggregate profiles. Also enables profiling.
    auto enable_trace() -> void {
        trace_enabled_.store(true, std::memory_order_release);
        enable();
    }

    /// Disable profiling (also disables trace mode).
    auto disable() -> void {
        enabled_.store(false, std::memory_order_release);
        trace_enabled_.store(false, std::memory_order_release);
    }

    /// Check if profiling is enabled (lock-free).
    [[nodiscard]] auto is_enabled() const noexcept -> bool {
        return enabled_.load(std::memory_order_acquire);
    }

    /// Check if trace mode is enabled (lock-free).
    [[nodiscard]] auto is_trace_enabled() const noexcept -> bool {
        return trace_enabled_.load(std::memory_order_acquire);
    }

    /// Set the maximum number of trace events retained. When the buffer is full,
    /// record_trace() drops the oldest event (ring-buffer semantics) so trace
    /// mode can run for long sessions without unbounded memory growth. A value
    /// of 0 means "unbounded" (the historical behaviour). The default cap is
    /// kDefaultMaxTraceEvents.
    auto set_max_trace_events(size_t max_events) -> void {
        max_trace_events_.store(max_events, std::memory_order_release);
    }

    /// Current maximum trace-event retention (0 == unbounded).
    [[nodiscard]] auto max_trace_events() const noexcept -> size_t {
        return max_trace_events_.load(std::memory_order_acquire);
    }

    /// Clear all recorded data (aggregate profiles and trace events).
    auto reset() -> void {
        std::lock_guard lock(mutex_);
        data_.clear();
        trace_events_.clear();
        trace_write_pos_ = 0;
    }

    /// Record an operation execution (backward-compatible: defaults to Backward phase).
    auto record(const std::string& name, std::chrono::nanoseconds elapsed) -> void {
        record(name, elapsed, ProfilePhase::Backward);
    }

    /// Record an operation execution with explicit phase.
    auto record(const std::string& name, std::chrono::nanoseconds elapsed,
                ProfilePhase phase) -> void {
        std::string key = (phase == ProfilePhase::Forward ? "fwd:" : "bwd:") + name;
        std::lock_guard lock(mutex_);
        auto& entry = data_[key];
        if (entry.name.empty()) {
            entry.name = name;
            entry.phase = phase;
        }
        entry.total_time += elapsed;
        entry.call_count++;
    }

    /// Record a trace event (individual invocation). Only called when trace mode is active.
    auto record_trace(const std::string& name,
                      std::chrono::steady_clock::time_point start,
                      std::chrono::nanoseconds duration,
                      ProfilePhase phase) -> void {
        TraceEvent evt;
        evt.name = name;
        evt.phase = phase;
        evt.start = start;
        evt.duration = duration;
        evt.thread_id = static_cast<uint32_t>(
            std::hash<std::thread::id>{}(std::this_thread::get_id()));
        std::lock_guard lock(mutex_);
        const size_t cap = max_trace_events_.load(std::memory_order_acquire);
        if (cap == 0 || trace_events_.size() < cap) {
            // Unbounded mode, or still filling the buffer: append.
            trace_events_.push_back(std::move(evt));
        } else {
            // Buffer full: overwrite the oldest event (ring buffer). Each event
            // carries its absolute start time, so export still recovers the
            // correct time origin via min_element regardless of slot order.
            trace_events_[trace_write_pos_] = std::move(evt);
            trace_write_pos_ = (trace_write_pos_ + 1) % cap;
        }
    }

    /// Get all recorded profiles, sorted by total time descending.
    [[nodiscard]] auto profiles() const -> std::vector<AutogradProfile> {
        std::lock_guard lock(mutex_);
        std::vector<AutogradProfile> result;
        result.reserve(data_.size());
        for (const auto& [_, entry] : data_) {
            result.push_back(entry);
        }
        std::sort(result.begin(), result.end(),
                  [](const AutogradProfile& a, const AutogradProfile& b) {
                      return a.total_time > b.total_time;
                  });
        return result;
    }

    /// Get profiles filtered by phase.
    [[nodiscard]] auto profiles(ProfilePhase phase) const -> std::vector<AutogradProfile> {
        auto all = profiles();
        std::vector<AutogradProfile> filtered;
        for (auto& p : all) {
            if (p.phase == phase) filtered.push_back(std::move(p));
        }
        return filtered;
    }

    /// Human-readable summary string (both phases).
    [[nodiscard]] auto summary() const -> std::string {
        auto profs = profiles();
        if (profs.empty()) return "No profiles recorded.\n";

        std::string out = "Operation Profile:\n";
        out += "  Phase  Function                          Calls    Total (ms)   Avg (us)\n";
        out += "  -----  --------------------------------  -------  -----------  --------\n";
        for (const auto& p : profs) {
            auto total_ms = std::chrono::duration<double, std::milli>(p.total_time).count();
            auto avg_us = p.call_count > 0
                ? std::chrono::duration<double, std::micro>(p.total_time).count() / p.call_count
                : 0.0;
            const char* phase_str = p.phase == ProfilePhase::Forward ? "FWD  " : "BWD  ";
            char line[160];
            std::snprintf(line, sizeof(line), "  %s  %-34s %7lld  %11.3f  %8.1f\n",
                         phase_str, p.name.c_str(),
                         static_cast<long long>(p.call_count),
                         total_ms, avg_us);
            out += line;
        }
        return out;
    }

    /// Get a copy of all recorded trace events.
    [[nodiscard]] auto trace_events() const -> std::vector<TraceEvent> {
        std::lock_guard lock(mutex_);
        return trace_events_;
    }

    /**
     * @brief Export recorded trace events to Chrome Trace Event Format JSON.
     *
     * Writes a JSON file compatible with chrome://tracing and Perfetto.
     * All timestamps are relative to the earliest event.
     *
     * @param path  Output file path (e.g. "trace.json").
     */
    auto export_chrome_trace(const std::string& path) const -> void;

    /// Default upper bound on retained trace events when trace mode is active.
    /// Chosen large enough to capture a meaningful window of a training run
    /// while preventing unbounded growth for long-lived sessions.
    static constexpr size_t kDefaultMaxTraceEvents = 1'000'000;

private:
    AutogradProfiler() = default;
    std::atomic<bool> enabled_{false};
    std::atomic<bool> trace_enabled_{false};
    std::atomic<size_t> max_trace_events_{kDefaultMaxTraceEvents};
    mutable std::mutex mutex_;
    std::unordered_map<std::string, AutogradProfile> data_;
    std::vector<TraceEvent> trace_events_;
    size_t trace_write_pos_{0};  // ring-buffer cursor once trace_events_ is full
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
        auto ns = std::chrono::duration_cast<std::chrono::nanoseconds>(elapsed);
        auto& prof = AutogradProfiler::instance();
        prof.record(name_, ns);
        if (prof.is_trace_enabled()) {
            prof.record_trace(name_, start_, ns, ProfilePhase::Backward);
        }
    }

    BackwardTimer(const BackwardTimer&) = delete;
    BackwardTimer& operator=(const BackwardTimer&) = delete;

private:
    std::string name_;
    std::chrono::steady_clock::time_point start_;
};

} // namespace tenzor
