/**
 * @file monitor.cpp
 * @brief Implementation of the event-based monitoring system
 */

#include <tenzor/utils/monitor.hpp>

#include <stdexcept>

namespace tenzor {
namespace monitor {

// ---------------------------------------------------------------------------
// Stat
// ---------------------------------------------------------------------------

Stat::Stat(std::string name, Aggregation agg)
    : name_(std::move(name)), agg_(agg) {}

auto Stat::add(double value) -> void {
    // Hold mutex_ so a concurrent reset() cannot interleave between the
    // multi-field updates below (e.g. clearing count_ after value_ but before
    // min_/max_), giving reset() true atomicity w.r.t. an in-flight add().
    // The atomics still provide lock-free visibility to get()/count() readers.
    std::lock_guard<std::mutex> lock(mutex_);
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
            // Publish min_/max_ BEFORE bumping count_ (release the count last),
            // mirroring the Sum/Mean path. get()/get_min() guard the
            // uninitialized sentinels (+/-1.8e308) solely via count_ == 0 and do
            // NOT take mutex_. If count_ were incremented first, a lock-free
            // reader could observe count_ == 1 after the increment but before the
            // first min_/max_ store, pass the count==0 guard, and return the raw
            // sentinel as a real metric. Storing min_/max_ first and releasing
            // count_ afterwards guarantees a reader that sees count_ >= 1 also
            // sees at least one real min_/max_ store.
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
            // Release count_ only after min_/max_ are visible.
            count_.fetch_add(1, std::memory_order_release);
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
            // Before any add(), max_ holds numeric_limits::lowest() (~-1.8e308),
            // a sentinel that would leak to telemetry consumers. Return the
            // neutral empty value 0.0, mirroring the count==0 guard in Mean.
            if (count_.load(std::memory_order_acquire) == 0) return 0.0;
            return max_.load(std::memory_order_acquire);
        case Aggregation::Value:
            return value_.load(std::memory_order_acquire);
    }
    return 0.0; // unreachable
}

auto Stat::get_min() const -> double {
    // Before any add(), min_ holds numeric_limits::max() (~1.8e308), a sentinel
    // that would leak to telemetry consumers. Return the neutral empty value 0.0.
    if (count_.load(std::memory_order_acquire) == 0) return 0.0;
    return min_.load(std::memory_order_acquire);
}

auto Stat::count() const -> int64_t {
    return count_.load(std::memory_order_acquire);
}

auto Stat::aggregation() const -> Aggregation {
    return agg_;
}

auto Stat::name() const -> const std::string& {
    return name_;
}

auto Stat::reset() -> void {
    // Serialize with add() so the reset is atomic w.r.t. a concurrent add():
    // observers never see a partially-reset stat (e.g. count_ cleared while
    // value_ still holds the old sum).
    std::lock_guard<std::mutex> lock(mutex_);
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
        // A Stat's aggregation strategy is fixed at construction. Re-registering
        // the same name with a different strategy would silently return a Stat
        // computing under the first-registered aggregation, yielding wrong
        // reported metrics; reject the collision instead.
        if (it->second->aggregation() != agg) {
            throw std::invalid_argument(
                "Monitor::register_stat: stat '" + name +
                "' already registered with a different aggregation strategy");
        }
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
