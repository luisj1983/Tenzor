#include "tenzor/utils/memory_profiler.hpp"

#include <cstdio>
#include <sstream>

namespace tenzor {

auto MemoryProfiler::instance() -> MemoryProfiler& {
    static MemoryProfiler inst;
    return inst;
}

auto MemoryProfiler::on_allocate(size_t bytes) -> void {
    auto signed_bytes = static_cast<int64_t>(bytes);
    total_allocated_bytes_.fetch_add(signed_bytes, std::memory_order_relaxed);
    allocation_count_.fetch_add(1, std::memory_order_relaxed);

    auto current = current_allocated_bytes_.fetch_add(signed_bytes, std::memory_order_relaxed)
                   + signed_bytes;

    // Update peak via CAS loop
    auto peak = peak_allocated_bytes_.load(std::memory_order_relaxed);
    while (current > peak) {
        if (peak_allocated_bytes_.compare_exchange_weak(
                peak, current, std::memory_order_relaxed, std::memory_order_relaxed)) {
            break;
        }
    }
}

auto MemoryProfiler::on_deallocate(size_t bytes) -> void {
    auto signed_bytes = static_cast<int64_t>(bytes);
    current_allocated_bytes_.fetch_sub(signed_bytes, std::memory_order_relaxed);
    deallocation_count_.fetch_add(1, std::memory_order_relaxed);
}

auto MemoryProfiler::memory_stats() const -> MemoryStats {
    return MemoryStats{
        .current_allocated_bytes = current_allocated_bytes_.load(std::memory_order_relaxed),
        .peak_allocated_bytes    = peak_allocated_bytes_.load(std::memory_order_relaxed),
        .total_allocated_bytes   = total_allocated_bytes_.load(std::memory_order_relaxed),
        .allocation_count        = allocation_count_.load(std::memory_order_relaxed),
        .deallocation_count      = deallocation_count_.load(std::memory_order_relaxed),
    };
}

auto MemoryProfiler::reset_peak_memory_stats() -> void {
    auto current = current_allocated_bytes_.load(std::memory_order_relaxed);
    peak_allocated_bytes_.store(current, std::memory_order_relaxed);
}

namespace {

auto format_bytes(int64_t bytes) -> std::string {
    char buf[64];
    // Counters can momentarily go negative under concurrent alloc/dealloc with
    // relaxed ordering; never report a negative size.
    if (bytes < 0) bytes = 0;
    if (bytes >= (1LL << 30)) {
        std::snprintf(buf, sizeof(buf), "%.2f GiB", static_cast<double>(bytes) / (1LL << 30));
    } else if (bytes >= (1LL << 20)) {
        std::snprintf(buf, sizeof(buf), "%.2f MiB", static_cast<double>(bytes) / (1LL << 20));
    } else if (bytes >= (1LL << 10)) {
        std::snprintf(buf, sizeof(buf), "%.2f KiB", static_cast<double>(bytes) / (1LL << 10));
    } else {
        std::snprintf(buf, sizeof(buf), "%lld B", static_cast<long long>(bytes));
    }
    return buf;
}

} // anonymous namespace

auto MemoryProfiler::memory_summary() const -> std::string {
    auto s = memory_stats();
    std::ostringstream oss;
    oss << "MemoryProfiler Summary\n"
        << "  Current allocated : " << format_bytes(s.current_allocated_bytes) << "\n"
        << "  Peak allocated    : " << format_bytes(s.peak_allocated_bytes) << "\n"
        << "  Total allocated   : " << format_bytes(s.total_allocated_bytes) << "\n"
        << "  Allocations       : " << s.allocation_count << "\n"
        << "  Deallocations     : " << s.deallocation_count << "\n";
    return oss.str();
}

} // namespace tenzor
