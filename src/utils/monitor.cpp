/**
 * @file monitor.cpp
 * @brief Implementation of the event-based monitoring system
 */

#include <tenzor/utils/monitor.hpp>

namespace tenzor {
namespace monitor {

// ---------------------------------------------------------------------------
// Stat
// ---------------------------------------------------------------------------

Stat::Stat(std::string name, Aggregation agg)
    : name_(std::move(name)), agg_(agg) {}

auto Stat::add(double value) -> void {
    switch (agg_) {
        case Aggregation::Sum:
        case Aggregation::Mean: {
            // Atomic add for doubles via CAS loop. Publish count_ AFTER value_
            // (release), so a concurrent get() that reads count_ (acquire) then
            // value_ never observes a count ahead of the accumulated value — the
            // torn read that previously skewed the mean. (Still best-effort: the
            // value may transiently lead the count, never the reverse.)
            double old = value_.load(std::memory_order_relaxed);
            while (!value_.compare_exchange_weak(
                old, old + value,
                std::memory_order_release, std::memory_order_relaxed)) {
            }
            count_.fetch_add(1, std::memory_order_release);
            break;
        }
        case Aggregation::Count:
            count_.fetch_add(1, std::memory_order_relaxed);
            break;
        case Aggregation::MinMax: {
            count_.fetch_add(1, std::memory_order_relaxed);
            // Update min
            double old_min = min_.load(std::memory_order_relaxed);
            while (value < old_min &&
                   !min_.compare_exchange_weak(
                       old_min, value,
                       std::memory_order_release, std::memory_order_relaxed)) {
            }
            // Update max
            double old_max = max_.load(std::memory_order_relaxed);
            while (value > old_max &&
                   !max_.compare_exchange_weak(
                       old_max, value,
                       std::memory_order_release, std::memory_order_relaxed)) {
            }
            break;
        }
        case Aggregation::Value:
            count_.fetch_add(1, std::memory_order_relaxed);
            value_.store(value, std::memory_order_release);
            break;
    }
}

auto Stat::get() const -> double {
    switch (agg_) {
        case Aggregation::Sum:
            return value_.load(std::memory_order_acquire);
        case Aggregation::Mean: {
            auto c = count_.load(std::memory_order_acquire);
            if (c == 0) return 0.0;
            return value_.load(std::memory_order_acquire) / static_cast<double>(c);
        }
        case Aggregation::Count:
            return static_cast<double>(count_.load(std::memory_order_acquire));
        case Aggregation::MinMax:
            return max_.load(std::memory_order_acquire);
        case Aggregation::Value:
            return value_.load(std::memory_order_acquire);
    }
    return 0.0; // unreachable
}

auto Stat::count() const -> int64_t {
    return count_.load(std::memory_order_acquire);
}

auto Stat::name() const -> const std::string& {
    return name_;
}

auto Stat::reset() -> void {
    value_.store(0.0, std::memory_order_release);
    min_.store(std::numeric_limits<double>::max(), std::memory_order_release);
    max_.store(std::numeric_limits<double>::lowest(), std::memory_order_release);
    count_.store(0, std::memory_order_release);
}

// ---------------------------------------------------------------------------
// Monitor
// ---------------------------------------------------------------------------

auto Monitor::instance() -> Monitor& {
    static Monitor inst;
    return inst;
}

auto Monitor::register_stat(const std::string& name, Aggregation agg) -> Stat& {
    std::lock_guard lock(mutex_);
    auto it = stats_.find(name);
    if (it != stats_.end()) {
        return *it->second;
    }
    auto [inserted, _] = stats_.emplace(name, std::make_unique<Stat>(name, agg));
    return *inserted->second;
}

auto Monitor::get_stat(const std::string& name) -> Stat* {
    std::lock_guard lock(mutex_);
    auto it = stats_.find(name);
    return it != stats_.end() ? it->second.get() : nullptr;
}

auto Monitor::log_event(const std::string& name,
                         const std::unordered_map<std::string, double>& data) -> void {
    // Snapshot the handler list under the lock, then dispatch outside.
    std::vector<std::shared_ptr<EventHandler>> snapshot;
    {
        std::lock_guard lock(mutex_);
        snapshot = handlers_;
    }
    for (auto& handler : snapshot) {
        handler->handle(name, data);
    }
}

auto Monitor::add_handler(std::shared_ptr<EventHandler> handler) -> void {
    std::lock_guard lock(mutex_);
    handlers_.push_back(std::move(handler));
}

auto Monitor::remove_all_handlers() -> void {
    std::lock_guard lock(mutex_);
    handlers_.clear();
}

auto Monitor::reset_all_stats() -> void {
    std::lock_guard lock(mutex_);
    for (auto& [_, stat] : stats_) {
        stat->reset();
    }
}

auto Monitor::stat_names() const -> std::vector<std::string> {
    std::lock_guard lock(mutex_);
    std::vector<std::string> names;
    names.reserve(stats_.size());
    for (const auto& [name, _] : stats_) {
        names.push_back(name);
    }
    return names;
}

} // namespace monitor
} // namespace tenzor
