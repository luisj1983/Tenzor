/**
 * @file chrome_trace.cpp
 * @brief Chrome Trace Event Format JSON exporter for AutogradProfiler.
 *
 * Produces JSON compatible with chrome://tracing and Perfetto UI.
 * Format spec: https://docs.google.com/document/d/1CvAClvFfyA5R-PhYUmn5OOQtYMH4h6I0nSsKchNAySU
 */

#include <tenzor/autograd/profiler.hpp>

#include <algorithm>
#include <cstdio>
#include <fstream>
#include <stdexcept>

namespace tenzor {

auto AutogradProfiler::export_chrome_trace(const std::string& path) const -> void {
    auto events = trace_events();  // thread-safe copy

    if (events.empty()) {
        // Write a valid but empty trace
        std::ofstream out(path);
        if (!out.is_open()) {
            throw std::runtime_error("Failed to open trace file: " + path);
        }
        out << "[]\n";
        return;
    }

    // Find the earliest event to use as time origin
    auto earliest = std::min_element(events.begin(), events.end(),
        [](const TraceEvent& a, const TraceEvent& b) {
            return a.start < b.start;
        });
    auto time_origin = earliest->start;

    std::ofstream out(path);
    if (!out.is_open()) {
        throw std::runtime_error("Failed to open trace file: " + path);
    }

    out << "[\n";

    for (size_t i = 0; i < events.size(); ++i) {
        const auto& evt = events[i];

        // Timestamps in microseconds relative to time origin
        auto ts_us = std::chrono::duration_cast<std::chrono::microseconds>(
            evt.start - time_origin).count();
        auto dur_us = std::chrono::duration_cast<std::chrono::microseconds>(
            evt.duration).count();

        const char* category = (evt.phase == ProfilePhase::Forward) ? "forward" : "backward";

        // Escape the name for JSON (handle quotes and backslashes)
        std::string escaped_name;
        escaped_name.reserve(evt.name.size());
        for (char c : evt.name) {
            if (c == '"') {
                escaped_name += "\\\"";
            } else if (c == '\\') {
                escaped_name += "\\\\";
            } else {
                escaped_name += c;
            }
        }

        // Write the Complete Duration Event ("ph":"X")
        out << "  {\"name\":\"" << escaped_name
            << "\",\"cat\":\"" << category
            << "\",\"ph\":\"X\""
            << ",\"ts\":" << ts_us
            << ",\"dur\":" << dur_us
            << ",\"pid\":1"
            << ",\"tid\":" << evt.thread_id
            << "}";

        if (i + 1 < events.size()) {
            out << ",";
        }
        out << "\n";
    }

    out << "]\n";
}

} // namespace tenzor
