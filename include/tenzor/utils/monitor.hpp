/**
 * @file monitor.hpp
 * @brief Event-based monitoring system for tensor operations
 *
 * Provides a singleton Monitor with named statistics (counters, gauges,
 * aggregations) and a pluggable event-handler interface for dispatching
 * structured events to user-defined sinks.
 */

#pragma once

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <limits>
#include <memory>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace tenzor {
namespace monitor {

/**
 * @brief Aggregation strategy applied when a Stat accumulates values
 *
 * - Sum: running total
 * - Mean: running average (sum / count)
 * - Count: number of add() calls
 * - MinMax: tracks min and max; get() returns the max
 * - Value: last value written (gauge)
 */
enum class Aggregation { Sum, Mean, Count, MinMax, Value };

/**
 * @brief A single named statistic with thread-safe accumulation
 *
 * Values are added via add() and the current aggregate is retrieved
 * via get(). The interpretation of get() depends on the Aggregation
 * strategy chosen at construction time.
 */
class Stat {
public:
    Stat(std::string name, Aggregation agg);

    /// Accumulate a value into this statistic (thread-safe).
    auto add(double value) -> void;

    /// Retrieve the current aggregate value.
    [[nodiscard]] auto get() const -> double;

    /// Number of add() calls since last reset.
    [[nodiscard]] auto count() const -> int64_t;

    /// The name this stat was registered under.
    [[nodiscard]] auto name() const -> const std::string&;

    /// Reset this statistic to its initial state.
    auto reset() -> void;

private:
    std::string name_;
    Aggregation agg_;
    std::atomic<double> value_{0.0};
    std::atomic<double> min_{std::numeric_limits<double>::max()};
    std::atomic<double> max_{std::numeric_limits<double>::lowest()};
    std::atomic<int64_t> count_{0};
    mutable std::mutex mutex_;
};

/**
 * @brief Base class for event handlers
 *
 * Subclass and override handle() to receive structured events
 * dispatched by Monitor::log_event().
 */
class EventHandler {
public:
    virtual ~EventHandler() = default;

    /// Called for every event logged through the Monitor.
    virtual auto handle(const std::string& event_name,
                        const std::unordered_map<std::string, double>& data) -> void = 0;
};

/**
 * @brief Singleton event-based monitoring system
 *
 * Maintains a registry of named Stat objects and dispatches events
 * to a chain of EventHandler instances.
 */
class Monitor {
public:
    /// Obtain the process-wide Monitor instance (Meyer's singleton).
    static auto instance() -> Monitor&;

    /// Register a new statistic. Returns a reference to the (possibly
    /// existing) Stat with the given name.
    auto register_stat(const std::string& name, Aggregation agg) -> Stat&;

    /// Look up a statistic by name. Returns nullptr if not found.
    auto get_stat(const std::string& name) -> Stat*;

    /// Dispatch a named event (with optional key-value payload) to all
    /// registered handlers.
    auto log_event(const std::string& name,
                   const std::unordered_map<std::string, double>& data = {}) -> void;

    /// Add an event handler to the dispatch chain.
    auto add_handler(std::shared_ptr<EventHandler> handler) -> void;

    /// Remove all event handlers.
    auto remove_all_handlers() -> void;

    /// Reset every registered statistic to its initial state.
    auto reset_all_stats() -> void;

    /// Return the names of all registered statistics.
    [[nodiscard]] auto stat_names() const -> std::vector<std::string>;

private:
    Monitor() = default;

    std::unordered_map<std::string, std::unique_ptr<Stat>> stats_;
    std::vector<std::shared_ptr<EventHandler>> handlers_;
    mutable std::mutex mutex_;
};

} // namespace monitor
} // namespace tenzor
