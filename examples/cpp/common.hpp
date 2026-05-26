/**
 * @file examples/cpp/common.hpp
 * @brief Shared helpers for the standalone Tenzor C++ examples.
 *
 * Provides a single helper for building per-run scratch paths that are
 * unique to the current process. Several examples used to hardcode paths
 * like `/tmp/model.bin`, which collide when two examples (or two CI runs
 * on the same host) execute concurrently — the second writer truncates
 * the first reader's file mid-load and the example crashes with an
 * impossible-to-trace serialisation error. Using PID-suffixed paths under
 * the platform's temp directory makes the collision impossible without
 * making the examples harder to read.
 *
 * Usage:
 *     #include "../common.hpp"
 *     std::string path = tenzor::examples::example_tmp_path("my_model");
 *     // -> "/tmp/my_model_12345" (or equivalent on non-POSIX hosts)
 *
 * The helper is header-only so each example translation unit gets its own
 * instantiation; no extra build wiring is required.
 */

#pragma once

#include <filesystem>
#include <string>

#if defined(_WIN32)
#  include <process.h>
#  define TENZOR_EXAMPLES_GETPID _getpid
#else
#  include <unistd.h>
#  define TENZOR_EXAMPLES_GETPID ::getpid
#endif

namespace tenzor::examples {

/// Build a temp-directory path unique to the current process.
///
/// @param prefix  Human-readable name baked into the returned filename, so
///                tmpwatch-style cleanup can still tell two examples apart.
/// @return        std::filesystem::temp_directory_path() / (prefix + "_" + pid)
inline std::string example_tmp_path(const std::string& prefix) {
    namespace fs = std::filesystem;
    return (fs::temp_directory_path() /
            (prefix + "_" + std::to_string(TENZOR_EXAMPLES_GETPID())))
        .string();
}

}  // namespace tenzor::examples
